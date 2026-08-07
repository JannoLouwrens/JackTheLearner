"""
TRAINING PIPELINE - Clean, Minimal, Correct

Replaces the sprawling RobustTrainer.py (8970 lines, 14.7% dead code, 5 duplicate
RL implementations) with a focused pipeline that does one thing well per phase.

Design principles:
1. ONE rl_update() method used everywhere (no duplicates)
2. ONE save/load system (always includes obs_projection + optimizer)
3. ZERO dead code
4. Each phase is a clean function: load → train → save
5. Checkpoint chain is explicit and never breaks

Phase structure:
    Phase 0: Physics Foundation     (SymPy supervision on MuJoCo rollouts)
    Phase 1: Imitation Learning     (MoCap behavior cloning)
    Phase 2: Locomotion RL          (PPO-style on Humanoid-v5)
    Phase 3: Language Grounding     (LLM projector + semantic anchors)
    Phase 4: Perception             (Vision + object detection)
    Phase 5: Manipulation           (Grasping + carrying)
    Phase 6: Planning               (Hierarchical + world model + navigation)
    Phase 7: Integration            (Dual system, all modalities)
    Phase 8: Companion              (Emotional dynamics, movement-mood, personality)

Research backing:
    - PPO: Schulman et al. (2017), RL-Zoo3 Humanoid-v4 hyperparameters
    - EWC: Kirkpatrick et al. (2017)
    - AMP: Peng et al. (2021)
    - Flow Matching: Lipman et al. (2022), pi0 (Physical Intelligence 2024)

Author: Janno Louwrens
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig, compute_physics_loss

try:
    from EmotionalState import EventType
except ImportError:
    EventType = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """All training configuration in one place."""

    # Model
    d_model: int = 512
    n_layers: int = 8
    obs_dim: int = 256
    mujoco_obs_dim: int = 376
    action_dim: int = 17

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.95           # PPO: RL-Zoo3 humanoid tuned
    gae_lambda: float = 0.9       # PPO: RL-Zoo3 humanoid tuned
    clip_range: float = 0.3       # PPO: RL-Zoo3 humanoid tuned
    max_grad_norm: float = 2.0    # PPO: RL-Zoo3 humanoid tuned
    entropy_coef: float = 0.002   # PPO: RL-Zoo3 humanoid tuned
    vf_coef: float = 0.43        # PPO: RL-Zoo3 humanoid tuned
    n_steps: int = 512            # PPO rollout length
    n_epochs_ppo: int = 5         # PPO update epochs per rollout
    ppo_minibatch: int = 512      # PPO minibatch. NOT config.batch_size: at 64,
                                  # minibatch COUNT scales with rollout size, so
                                  # adding envs bought no throughput -- T2.01 v2
                                  # spent ~12s of its 13s/iter in the update on
                                  # a P100. Same total sample-passes either way;
                                  # bigger minibatches just fill the GPU.
    normalize_returns: bool = True  # keep value targets O(1); see rl_update
    action_limit: float = 0.4     # Humanoid-v5 actuator range
    squash_actions: bool = True   # bound the policy MEAN; see policy_mean()
    action_std_init: float = 0.3  # Initial exploration noise
    log_std_min: float = -4.6     # std >= 0.01 — floor, so the policy can commit
    log_std_max: float = 0.0      # std <= 1.0  — ceiling; the entropy bonus is
                                  # otherwise unbounded and inflates std forever

    # Anti-forgetting
    replay_ratio: float = 0.2
    ewc_lambda: float = 1000
    physics_weight: float = 0.1

    # Paths
    checkpoint_dir: str = "checkpoints"
    drive_path: str = "/content/drive/MyDrive/JackTheLearner/checkpoints"


# ==============================================================================
# REPLAY BUFFER (same proven design)
# ==============================================================================

class ReplayBuffer:
    """Phase-tagged replay buffer. Uses _phase tag on samples, not stale index lists."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, sample: Dict, phase: int):
        sample['_phase'] = phase
        self.buffer.append(sample)

    def sample(self, batch_size: int, phase_ratios: Dict[int, float] = None) -> List[Dict]:
        if len(self.buffer) == 0:
            return []
        if phase_ratios is None:
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        phase_samples = {}
        for s in self.buffer:
            p = s.get('_phase', -1)
            if p not in phase_samples:
                phase_samples[p] = []
            phase_samples[p].append(s)

        result = []
        for phase, ratio in phase_ratios.items():
            n = int(batch_size * ratio)
            if phase in phase_samples and phase_samples[phase]:
                result.extend(random.choices(phase_samples[phase], k=min(n, len(phase_samples[phase]))))
        return result

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        torch.save(list(self.buffer), path)

    def load(self, path: str):
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            if isinstance(data, dict):
                data = data.get('buffer', [])
            self.buffer = deque(data, maxlen=self.capacity)
            for s in self.buffer:
                if '_phase' not in s:
                    s['_phase'] = 0


# ==============================================================================
# EWC (Elastic Weight Consolidation)
# ==============================================================================

class EWC:
    """Protects important weights from previous phases."""

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.old_params = {}
        self.fisher = {}

    def compute_fisher(self, dataloader_fn, num_samples: int = 500):
        """Compute Fisher information from a data generator function."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        device = next(self.model.parameters()).device
        count = 0

        for state, action in dataloader_fn():
            if count >= num_samples:
                break
            self.model.zero_grad()
            output = self.model(state.to(device), action=action.to(device))
            loss = output['physics'].pow(2).mean() + output['actions'].pow(2).mean()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)
            count += state.shape[0]

        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher = fisher
        self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        print(f"[EWC] Fisher computed on {count} samples")

    def penalty(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        if not self.fisher:
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n].to(device) * (p - self.old_params[n].to(device)).pow(2)).sum()
        return self.lambda_ewc * loss

    def save(self, path: str):
        torch.save({'fisher': self.fisher, 'old_params': self.old_params}, path)

    def load(self, path: str):
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            self.fisher = data['fisher']
            self.old_params = data['old_params']


# ==============================================================================
# THE TRAINING PIPELINE
# ==============================================================================

class TrainingPipeline:
    """
    Clean training pipeline for Jack.

    One rl_update method. One save/load system. Zero dead code.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 70)
        print("JACK TRAINING PIPELINE")
        print("=" * 70)

        # Create model
        model_config = UnifiedBrainConfig(
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            llm_enabled=False,
            vision_enabled=False,
            audio_enabled=False,
        )
        self.model = UnifiedBrain(model_config).to(self.device)

        # Observation projection: MuJoCo (376) -> model (256)
        self.obs_proj = nn.Sequential(
            nn.Linear(self.config.mujoco_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.config.obs_dim),
            nn.LayerNorm(self.config.obs_dim),
        ).to(self.device)

        # Observation normalization (RL-Zoo3: normalize=True, critical for Humanoid)
        # Running scale of the DISCOUNTED RETURN (SB3 VecNormalize pattern).
        # Ladder spec T2.01 measured why this is not optional: with raw Humanoid
        # returns the value loss ran 540 against a policy loss of 0.27, and with
        # vf_coef=0.43 the value term was ~870x the policy term. value_head sits
        # on the SHARED trunk, so all 57M params were optimised to regress
        # returns while the policy rode along on whatever representation that
        # produced -- training came out at -4334 versus +170 untrained.
        self.ret_var = torch.ones((), device=self.device)
        self.ret_count = 1e-4
        self.obs_mean = torch.zeros(self.config.mujoco_obs_dim, device=self.device)
        self.obs_var = torch.ones(self.config.mujoco_obs_dim, device=self.device)
        self.obs_count = 0

        # Anti-forgetting
        self.replay = ReplayBuffer()
        self.ewc = EWC(self.model, self.config.ewc_lambda)

        # Learnable action std for PPO (RL-Zoo3 pattern: state-independent log_std)
        self.log_std = nn.Parameter(
            torch.ones(self.config.action_dim, device=self.device) * np.log(self.config.action_std_init)
        )

        # State
        self.epoch = 0
        self.global_step = 0
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        print(f"  Device: {self.device}")
        print("=" * 70 + "\n")

    # ─────────────────────────────────────────────────────────────────────
    # OBSERVATION PROJECTION
    # ─────────────────────────────────────────────────────────────────────

    def policy_mean(self, output) -> torch.Tensor:
        """Turn the raw action head output into a BOUNDED policy mean.

        locomotion_head is a bare nn.Linear, so its output is unbounded, and
        clipping alone does not constrain it: once the mean passes the actuator
        limit every action saturates to the same clipped command, the gradient
        can no longer distinguish 0.5 from 43.8, and the drift is unopposed.

        Ladder spec T2.01 measured exactly that runaway. The learning curve, per
        iteration:

            iter    reward   |act|max     std
               1     4.676       1.26   0.304
               4     4.254       6.10   0.317
              11     3.855      17.88   0.352
              31     4.008      43.81   0.506

        Reward flat at ~4.0 while the mean grew 35x past a +-0.4 range — and at
        evaluation the policy was pure bang-bang, returning -19,435 against +171
        for the untrained network. The watch-item written into the journal after
        the previous run said: if reward stalls while |a| grows, squash it. It
        stalled, so it is squashed.

        tanh scaling bounds the mean STRUCTURALLY, so no amount of drift can
        leave the range. Exploration noise is still added on top and clipped for
        the environment, which is the standard Box-space arrangement.
        """
        raw = output["actions"][:, 0, :]
        if not getattr(self.config, "squash_actions", True):
            return raw
        return torch.tanh(raw) * self.config.action_limit

    def normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        """Running observation normalization (RL-Zoo3 pattern).
        Keeps running mean/var and normalizes to ~N(0,1), clipped to [-10, 10].

        Accepts a single observation (D,) or a batch (N, D). The batch path is
        NOT a convenience: feeding a batch through the single-obs update would
        silently corrupt the running statistics — obs_count would advance by 1
        per BATCH instead of per observation, and the (N, D) delta would
        broadcast obs_mean into a matrix. Found while batching the rollout
        (T0.07: the batch-1 policy forward is 155x the physics it drives)."""
        obs = np.asarray(obs_raw, dtype=np.float64)
        if obs.ndim == 2:
            return self._normalize_obs_batch(obs)
        self.obs_count += 1
        if self.obs_count == 1:
            self.obs_mean = torch.tensor(obs, dtype=torch.float32, device=self.device)
            self.obs_var = torch.ones_like(self.obs_mean)
        else:
            delta = torch.tensor(obs, dtype=torch.float32, device=self.device) - self.obs_mean
            self.obs_mean += delta / self.obs_count
            self.obs_var = self.obs_var * (self.obs_count - 1) / self.obs_count + delta.pow(2) / self.obs_count

        std = (self.obs_var + 1e-8).sqrt()
        normalized = (torch.tensor(obs, dtype=torch.float32, device=self.device) - self.obs_mean) / std
        return normalized.clamp(-10, 10).cpu().numpy().astype(np.float32)

    def project_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Map any observation to model's internal dim."""
        if obs.shape[-1] == self.config.obs_dim:
            return obs
        elif obs.shape[-1] == self.config.mujoco_obs_dim:
            return self.obs_proj(obs)
        else:
            pad = self.config.mujoco_obs_dim - obs.shape[-1]
            if pad > 0:
                obs = F.pad(obs, (0, pad))
            else:
                obs = obs[..., :self.config.mujoco_obs_dim]
            return self.obs_proj(obs)

    # ─────────────────────────────────────────────────────────────────────
    # CHECKPOINT SYSTEM
    # ─────────────────────────────────────────────────────────────────────

    def save(self, name: str):
        """Save everything: model + obs_proj + optimizer + log_std."""
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        data = {
            'model': self.model.state_dict(),
            'obs_proj': self.obs_proj.state_dict(),
            'log_std': self.log_std.data,
            'obs_mean': self.obs_mean,
            'obs_var': self.obs_var,
            'obs_count': self.obs_count,
            'epoch': self.epoch,
            'global_step': self.global_step,
        }
        if hasattr(self, 'optimizer'):
            data['optimizer'] = self.optimizer.state_dict()
        torch.save(data, path)
        print(f"[SAVE] {path}")

    def load(self, name_or_path: str) -> bool:
        """Load checkpoint. Returns True if successful."""
        if os.path.exists(name_or_path):
            path = name_or_path
        else:
            path = os.path.join(self.config.checkpoint_dir, f"{name_or_path}.pt")
        if not os.path.exists(path):
            # Try drive path
            drive = os.path.join(self.config.drive_path, os.path.basename(path))
            if os.path.exists(drive):
                path = drive
            else:
                print(f"[WARN] Checkpoint not found: {name_or_path}")
                return False

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=False)
        if 'obs_proj' in ckpt:
            self.obs_proj.load_state_dict(ckpt['obs_proj'], strict=False)
        if 'log_std' in ckpt:
            self.log_std.data.copy_(ckpt['log_std'])
        if 'obs_mean' in ckpt:
            self.obs_mean = ckpt['obs_mean'].to(self.device)
            self.obs_var = ckpt['obs_var'].to(self.device)
            self.obs_count = ckpt['obs_count']
        if 'optimizer' in ckpt and hasattr(self, 'optimizer'):
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError):
                pass
        self.epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        print(f"[LOAD] {path}")
        return True

    def find_checkpoint(self, *names: str) -> Optional[str]:
        """Find the first available checkpoint from a priority list."""
        for name in names:
            for suffix in ['_best', '_latest', '']:
                for base in [self.config.checkpoint_dir, self.config.drive_path]:
                    path = os.path.join(base, f"{name}{suffix}.pt")
                    if os.path.exists(path):
                        return path
        return None

    # ─────────────────────────────────────────────────────────────────────
    # OPTIMIZER FACTORY
    # ─────────────────────────────────────────────────────────────────────

    def make_optimizer(self, phase: int, lr: float = None):
        """Create optimizer for a phase. Always includes obs_proj + log_std."""
        lr = lr or self.config.learning_rate

        # All learnable params
        params = [
            {'params': list(self.model.parameters()), 'lr': lr},
            {'params': list(self.obs_proj.parameters()), 'lr': lr},
            {'params': [self.log_std], 'lr': lr},
        ]

        self.optimizer = torch.optim.AdamW(params, weight_decay=1e-4, eps=1e-5)
        return self.optimizer

    # ─────────────────────────────────────────────────────────────────────
    # THE ONE RL UPDATE (PPO-style, RL-Zoo3 Humanoid tuned)
    # ─────────────────────────────────────────────────────────────────────

    def rl_update(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO update on a rollout buffer. THE SINGLE RL update method.

        Args:
            rollout: dict with keys:
                states: (N, obs_dim)
                actions: (N, action_dim)
                rewards: (N,)
                dones: (N,)
                log_probs: (N,)
                values: (N,)

        Returns:
            dict of training metrics
        """
        states = rollout['states']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        old_values = rollout['values']
        rewards = rollout['rewards']
        dones = rollout['dones']

        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        # ── GAE advantage estimation ──
        # Accepts both layouts: (T,) from collect_rollout, or (T, N) from
        # collect_rollout_vec. The recursion is identical — with (T, N) rows,
        # delta and last_gae are (N,) vectors and every env's advantage chain is
        # computed in parallel. What would NOT work is flattening (T, N) to
        # (T*N,) first: that interleaves envs into one fake trajectory, so env
        # k's first step would bootstrap from env k-1's last value. Time must
        # stay dim 0 until the advantages exist; flattening is safe only after.
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            next_value = old_values[t + 1] if t < T - 1 else torch.zeros_like(rewards[0])
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - old_values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + old_values

        if getattr(self.config, "normalize_returns", True):
            # Scale (not centre) by a running std of returns: centring would bias
            # the value target, scaling only fixes the loss magnitude. Advantages
            # are normalised separately below, per batch, as PPO expects.
            batch_var = returns.detach().var()
            n = returns.numel()
            self.ret_count += n
            w = n / self.ret_count
            self.ret_var = (1 - w) * self.ret_var + w * batch_var
            scale = torch.sqrt(self.ret_var + 1e-8).clamp(min=1e-3)
            returns = returns / scale
            old_values = old_values / scale
            advantages = advantages / scale

        if states.dim() == 3:                      # (T, N, D) -> flat for PPO
            states = states.reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            old_log_probs = old_log_probs.reshape(-1)
            old_values = old_values.reshape(-1)
            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)
        N = states.shape[0]

        # Normalize advantages (per-batch, critical for stability)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update epochs ──
        batch_size = min(getattr(self.config, "ppo_minibatch", 512), N)
        total_pg_loss = 0
        total_vf_loss = 0
        total_entropy = 0

        for _ in range(self.config.n_epochs_ppo):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                idx = indices[start:start + batch_size]
                if len(idx) < 4:
                    continue

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Forward pass. project_obs runs HERE, inside the minibatch,
                # so obs_proj receives a fresh gradient every backward -- it is
                # a no-op for buffers that already stored projected states, so
                # the single-env collect_rollout path is unaffected.
                output = self.model(self.project_obs(mb_states))
                action_mean = self.policy_mean(output)
                values_pred = output['value'].squeeze(-1)

                # Action distribution
                std = self.log_std.clamp(self.config.log_std_min,
                                         self.config.log_std_max
                                         ).exp().expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Policy loss (clipped)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                vf_loss = F.mse_loss(values_pred, mb_returns)

                # EWC penalty (protects Phase 0 physics knowledge)
                ewc_loss = self.ewc.penalty()

                # Replay loss (anti-forgetting: mix Phase 0 physics samples)
                replay_loss = torch.tensor(0.0, device=self.device)
                if len(self.replay) > 0:
                    replay_batch = self.replay.sample(8, phase_ratios={0: 1.0})
                    if replay_batch:
                        r_states = torch.stack([s['state'] for s in replay_batch]).to(self.device)
                        r_physics = torch.stack([s['physics'] for s in replay_batch]).to(self.device)
                        r_out = self.model(r_states)
                        replay_loss = F.mse_loss(r_out['physics'], r_physics) * self.config.replay_ratio

                # Total
                loss = (
                    pg_loss
                    + self.config.vf_coef * vf_loss
                    - self.config.entropy_coef * entropy
                    + ewc_loss
                    + replay_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.obs_proj.parameters()) + [self.log_std],
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                self.global_step += 1

        n_updates = max(1, (N // batch_size) * self.config.n_epochs_ppo)
        return {
            'pg_loss': total_pg_loss / n_updates,
            'vf_loss': total_vf_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    # ─────────────────────────────────────────────────────────────────────
    # ROLLOUT COLLECTOR
    # ─────────────────────────────────────────────────────────────────────

    def collect_rollout(self, env, n_steps: int = None) -> Dict[str, torch.Tensor]:
        """Collect a rollout buffer from the environment."""
        n_steps = n_steps or self.config.n_steps

        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        obs, _ = env.reset()

        for _ in range(n_steps):
            obs_norm = self.normalize_obs(obs)  # Running normalization
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
            state = self.project_obs(obs_tensor)

            with torch.no_grad():
                output = self.model(state)
                action_mean = self.policy_mean(output)
                value = output['value'].squeeze()

                std = self.log_std.clamp(self.config.log_std_min,
                                         self.config.log_std_max
                                         ).exp().expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                action_sampled = dist.sample()
                log_prob = dist.log_prob(action_sampled).sum(dim=-1)

            action_np = action_sampled[0].cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            states.append(state[0])
            actions.append(action_sampled[0])
            log_probs.append(log_prob[0])
            values.append(value)
            rewards.append(float(reward))
            dones.append(float(terminated or truncated))

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values),
            'rewards': torch.tensor(rewards, device=self.device),
            'dones': torch.tensor(dones, device=self.device),
        }

    def _normalize_obs_batch(self, obs: np.ndarray) -> np.ndarray:
        """Chan et al. parallel merge of batch statistics into the running ones.

        The single-obs path stores a biased running variance (M2/count), so the
        batch is merged in the same currency: convert to M2, merge, divide back.
        """
        n = obs.shape[0]
        batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
        b_mean = batch.mean(dim=0)
        b_var = batch.var(dim=0, unbiased=False)

        if self.obs_count == 0:
            self.obs_mean = b_mean
            self.obs_var = torch.clamp(b_var, min=1e-8)
            self.obs_count = n
        else:
            new_count = self.obs_count + n
            delta = b_mean - self.obs_mean
            m2 = (self.obs_var * self.obs_count + b_var * n
                  + delta.pow(2) * self.obs_count * n / new_count)
            self.obs_mean = self.obs_mean + delta * n / new_count
            self.obs_var = m2 / new_count
            self.obs_count = new_count

        std = (self.obs_var + 1e-8).sqrt()
        out = (batch - self.obs_mean) / std
        return out.clamp(-10, 10).cpu().numpy().astype(np.float32)

    def make_vec_envs(self, n_envs: int):
        """N synchronous Humanoid-v5 envs sharing one batched policy forward.

        Sync, not Async: the physics is cheap (1831 steps/s, T0.07) and the win
        is batching the POLICY, so process-per-env overhead buys nothing here.
        SAME_STEP autoreset keeps the classic contract — a terminated env returns
        its reset observation immediately — so the rollout loop needs no
        next-step masking (gymnasium 1.x defaults to NEXT_STEP, which trains on
        a phantom reset transition unless every consumer remembers to mask it).
        """
        from gymnasium.vector import SyncVectorEnv, AutoresetMode
        return SyncVectorEnv([lambda: self.make_env() for _ in range(n_envs)],
                             autoreset_mode=AutoresetMode.SAME_STEP)

    def collect_rollout_vec(self, envs, n_steps: int = None) -> Dict[str, torch.Tensor]:
        """collect_rollout over N envs with ONE policy forward per step.

        Why this exists: collect_rollout is batch-1 — unsqueeze(0), one 58M-param
        forward per physics step — which T0.07 measured at 11.8 steps/s, making
        2M steps a 47-hour job. The physics can already run 155x faster than the
        policy that drives it; batching the forward is the only lever that moves
        rollout throughput.

        SHAPE CONTRACT: returns (T, N, ...) tensors, NOT the flattened (T,)
        layout collect_rollout produces. Flattening (T, N) into (T*N) would
        interleave envs and silently break GAE, which walks dim 0 assuming time
        order. The PPO update that consumes this must compute advantages per env
        (over dim 0) before any flatten.
        """
        n_steps = n_steps or self.config.n_steps
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        obs, _ = envs.reset()
        for _ in range(n_steps):
            obs_norm = self.normalize_obs(obs)                      # (N, D) path
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32,
                                      device=self.device)

            with torch.no_grad():
                state = self.project_obs(obs_tensor)
                output = self.model(state)
                action_mean = self.policy_mean(output)              # (N, act)
                value = output['value'].squeeze(-1)                 # (N,)

                std = self.log_std.clamp(self.config.log_std_min,
                                         self.config.log_std_max
                                         ).exp().expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                action_sampled = dist.sample()
                log_prob = dist.log_prob(action_sampled).sum(dim=-1)

            # CLIP for the environment, keep the UNCLIPPED sample in the buffer
            # (the Gaussian density is over unclipped actions) -- the SB3 Box
            # convention. Measured need: |action| reached 1.20 then 2.37 against
            # an env range of +-0.4 within two iterations, so MuJoCo was silently
            # clipping and the policy was being scored for action components that
            # never touched the physics.
            _lo = envs.single_action_space.low
            _hi = envs.single_action_space.high
            next_obs, reward, term, trunc, _ = envs.step(
                np.clip(action_sampled.cpu().numpy(), _lo, _hi))

            # RAW normalized obs, not the projected state. Storing the projection
            # kept its autograd graph alive into rl_update, where the second PPO
            # minibatch backward hit a freed graph and crashed -- and the only
            # gradient obs_proj ever received flowed through that broken path, so
            # it could not actually train. rl_update projects per minibatch now.
            states.append(obs_tensor)
            actions.append(action_sampled)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32,
                                        device=self.device))
            dones.append(torch.tensor(np.logical_or(term, trunc),
                                      dtype=torch.float32, device=self.device))
            obs = next_obs

        return {
            'states': torch.stack(states),        # (T, N, obs_dim)
            'actions': torch.stack(actions),      # (T, N, act_dim)
            'log_probs': torch.stack(log_probs),  # (T, N)
            'values': torch.stack(values),        # (T, N)
            'rewards': torch.stack(rewards),      # (T, N)
            'dones': torch.stack(dones),          # (T, N)
        }

    # ─────────────────────────────────────────────────────────────────────
    # MuJoCo ENVIRONMENT
    # ─────────────────────────────────────────────────────────────────────

    def make_env(self, render_mode: str = None):
        """Create MuJoCo Humanoid-v5 environment."""
        try:
            import gymnasium as gym
            kwargs = {}
            if render_mode:
                kwargs['render_mode'] = render_mode
            env = gym.make("Humanoid-v5", **kwargs)
            print(f"[ENV] Humanoid-v5: obs={env.observation_space.shape[0]}, act={env.action_space.shape[0]}")
            return env
        except Exception as e:
            print(f"[WARN] Cannot create environment: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 0: PHYSICS FOUNDATION
    # ─────────────────────────────────────────────────────────────────────

    def train_phase0(self, epochs: int = 50, samples_per_epoch: int = 10000):
        """Learn physics from MuJoCo rollouts with SymPy supervision."""
        print("\n" + "=" * 70)
        print("PHASE 0: Physics Foundation")
        print("=" * 70)

        self.make_optimizer(0)       # Create optimizer FIRST
        self.load("phase0_best")     # Then load (can now restore optimizer state)

        env = self.make_env()
        try:
            from SymbolicCalculator import SymbolicPhysicsCalculator
            calc = SymbolicPhysicsCalculator()
        except ImportError:
            calc = None
            print("[WARN] No SymPy calculator")

        best_loss = float('inf')
        env_obs = None
        if env is not None:
            env_obs, _ = env.reset()

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            B = self.config.batch_size

            pbar = tqdm(range(0, samples_per_epoch, B), desc=f"Phase 0 [{epoch+1}/{epochs}]")
            for _ in pbar:
                # Collect real data from MuJoCo
                if env is not None and env_obs is not None:
                    batch_states, batch_actions, batch_next = [], [], []
                    for _ in range(B):
                        action = env.action_space.sample()
                        next_obs, _, term, trunc, _ = env.step(action)
                        batch_states.append(env_obs)
                        batch_actions.append(action)
                        batch_next.append(next_obs)
                        env_obs = next_obs if not (term or trunc) else env.reset()[0]

                    raw_s = torch.tensor(np.array(batch_states), dtype=torch.float32, device=self.device)
                    state = self.project_obs(raw_s)
                    raw_ns = torch.tensor(np.array(batch_next), dtype=torch.float32, device=self.device)
                    next_state = self.project_obs(raw_ns)

                    act_np = np.array(batch_actions)
                    if act_np.shape[-1] < 57:
                        act_np = np.pad(act_np, ((0, 0), (0, 57 - act_np.shape[-1])))
                    action = torch.tensor(act_np, dtype=torch.float32, device=self.device)
                else:
                    # Structured fallback
                    state = torch.zeros(B, self.config.obs_dim, device=self.device)
                    state[:, 2] = 1.3 + torch.randn(B, device=self.device) * 0.1
                    state[:, 3:6] = torch.randn(B, 3, device=self.device) * 0.5
                    state[:, 6:23] = torch.randn(B, 17, device=self.device) * 0.3
                    action = torch.randn(B, 57, device=self.device) * 0.5
                    next_state = state + 0.02 * torch.randn_like(state)

                # Physics targets from SymPy calculator (exact) or inline approximation
                physics_targets = torch.zeros(B, 10, device=self.device)
                for i in range(B):
                    s_np = state[i].detach().cpu().numpy()
                    a_np = action[i].detach().cpu().numpy()

                    if calc is not None:
                        _, phys = calc.predict_robot_state(s_np, a_np)
                        ke = phys['kinetic_energy']
                        pe = phys['potential_energy']
                        mom = phys['momentum']
                        fmag = phys['force_magnitude']
                        physics_targets[i] = torch.tensor([
                            ke, pe, ke + pe, mom, fmag,
                            fmag * 0.3, mom * 0.5, 1.0 / (1 + abs(pe) / 1000), fmag * 0.02, fmag,
                        ])
                    else:
                        v = s_np[3:6] if len(s_np) > 5 else np.zeros(3)
                        h = max(s_np[2], 0) if len(s_np) > 2 else 1.3
                        speed = float(np.linalg.norm(v))
                        mass = 50.0
                        physics_targets[i] = torch.tensor([
                            0.5 * mass * speed**2, mass * 9.81 * h,
                            0.5 * mass * speed**2 + mass * 9.81 * h,
                            mass * speed, float(np.linalg.norm(a_np[:3])),
                            0, 0, 1.0 / (1 + h), 0, 0,
                        ])

                loss, _ = compute_physics_loss(self.model, state, action[:, :self.config.action_dim], next_state, physics_targets)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.obs_proj.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Store in replay
                for i in range(B):
                    self.replay.add({
                        'state': state[i].detach().cpu(),
                        'action': action[i].detach().cpu(),
                        'next_state': next_state[i].detach().cpu(),
                        'physics': physics_targets[i].detach().cpu(),
                    }, phase=0)

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}: loss={avg:.4f}")
            self.save("phase0_latest")
            if avg < best_loss:
                best_loss = avg
                self.save("phase0_best")

        # Save replay buffer for Phase 2 anti-forgetting
        self.replay.save(os.path.join(self.config.checkpoint_dir, "replay.pt"))

        # Compute EWC Fisher information (protects Phase 0 knowledge during Phase 2)
        print("[EWC] Computing Fisher information...")
        def phase0_data_gen():
            samples = self.replay.sample(500, phase_ratios={0: 1.0})
            for s in samples:
                yield s['state'].unsqueeze(0).to(self.device), s['action'][:self.config.action_dim].unsqueeze(0).to(self.device)
        self.ewc.compute_fisher(phase0_data_gen, num_samples=500)
        self.ewc.save(os.path.join(self.config.checkpoint_dir, "ewc.pt"))

        if env:
            env.close()
        print(f"\n[DONE] Phase 0. Best loss: {best_loss:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: LOCOMOTION RL (PPO)
    # ─────────────────────────────────────────────────────────────────────

    def train_phase2(self, total_timesteps: int = 2_000_000):
        """Train walking with PPO. Uses RL-Zoo3 Humanoid-v4 tuned hyperparams."""
        print("\n" + "=" * 70)
        print("PHASE 2: Locomotion RL (PPO)")
        print("=" * 70)

        # RL-Zoo3 tuned LR for Humanoid - create optimizer FIRST so checkpoint can restore it
        self.make_optimizer(2, lr=3.57e-5)

        # Load best previous checkpoint
        ckpt = self.find_checkpoint("phase1_best", "phase0_best")
        if ckpt:
            self.load(ckpt)

        self.replay.load(os.path.join(self.config.checkpoint_dir, "replay.pt"))
        self.ewc.load(os.path.join(self.config.checkpoint_dir, "ewc.pt"))

        env = self.make_env()
        if env is None:
            print("[SKIP] No MuJoCo environment")
            return

        best_reward = -float('inf')
        steps_done = 0
        episode_rewards = deque(maxlen=100)

        print(f"  Training for {total_timesteps:,} timesteps with PPO")
        print(f"  LR={3.57e-5}, gamma={self.config.gamma}, clip={self.config.clip_range}")

        while steps_done < total_timesteps:
            # Collect rollout
            rollout = self.collect_rollout(env, self.config.n_steps)
            steps_done += self.config.n_steps

            # Track episode rewards
            ep_rewards = []
            ep_reward = 0
            for r, d in zip(rollout['rewards'], rollout['dones']):
                ep_reward += r.item()
                if d > 0.5:
                    ep_rewards.append(ep_reward)
                    ep_reward = 0
            episode_rewards.extend(ep_rewards)

            # PPO update
            metrics = self.rl_update(rollout)

            # Log
            if steps_done % (self.config.n_steps * 10) == 0:
                mean_reward = np.mean(list(episode_rewards)) if episode_rewards else 0
                print(f"  [{steps_done:,}/{total_timesteps:,}] "
                      f"reward={mean_reward:.1f} "
                      f"pg={metrics['pg_loss']:.4f} "
                      f"vf={metrics['vf_loss']:.4f} "
                      f"ent={metrics['entropy']:.4f} "
                      f"std={self.log_std.exp().mean().item():.3f}")

                # Save
                self.save("phase2_latest")
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save("phase2_best")

        env.close()
        print(f"\n[DONE] Phase 2. Best reward: {best_reward:.1f}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 8: COMPANION
    # ─────────────────────────────────────────────────────────────────────

    def train_phase8(self, epochs: int = 100):
        """Train companion features: emotional dynamics + movement-mood coupling."""
        print("\n" + "=" * 70)
        print("PHASE 8: Virtual Companion")
        print("=" * 70)

        ckpt = self.find_checkpoint("phase7_best", "phase2_best", "phase0_best")
        if ckpt:
            self.load(ckpt)

        if not hasattr(self.model, 'emotional_state') or self.model.emotional_state is None:
            print("[SKIP] EmotionalState not enabled on model")
            return

        self.make_optimizer(8)

        # Phase 8.1: Emotional dynamics
        print("\n  [8.1] Emotional Dynamics")
        for epoch in range(min(epochs, 50)):
            epoch_loss = 0
            for _ in range(50):
                self.model.emotional_state.pad_vector.requires_grad_(True)
                self.model.emotional_state.pad_vector.data.zero_()
                moods = []

                for step in range(random.randint(10, 40)):
                    event = random.choice(list(EventType)) if EventType else None
                    self.model.emotional_state.update(event_type=event, reward=random.gauss(0, 1), dt=0.1)
                    moods.append(self.model.emotional_state.pad_vector.clone())

                # Smoothness + diversity + range loss
                loss = torch.tensor(0.0, device=self.device)
                if len(moods) > 2:
                    stack = torch.stack(moods)
                    # Smooth: consecutive moods should be similar
                    loss += 0.1 * (stack[1:] - stack[:-1]).pow(2).sum()
                    # Diverse: moods should vary over episode
                    loss += -0.05 * stack.var(dim=0).sum()
                    # In range: stay in [-1, 1]
                    loss += F.relu(stack.abs() - 1.0).sum()

                self.optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.model.emotional_state.pad_vector.requires_grad_(False)
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: loss={epoch_loss/50:.4f}")

        self.save("phase8_best")

        # Phase 8.2: Movement-mood coupling
        if hasattr(self.model, 'movement_mood') and self.model.movement_mood is not None:
            print("\n  [8.2] Movement-Mood Coupling")
            env = self.make_env()
            if env is not None:
                for epoch in range(min(epochs, 30)):
                    for _ in range(5):
                        obs, _ = env.reset()
                        target_pad = torch.randn(3, device=self.device).clamp(-1, 1)
                        self.model.emotional_state.pad_vector.data.copy_(target_pad)

                        for step in range(100):
                            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                            state = self.project_obs(obs_t)
                            with torch.no_grad():
                                out = self.model(state)
                                action = out['actions'][:, 0, :].cpu().numpy()[0]
                            obs, _, term, trunc, _ = env.step(action)
                            if term or trunc:
                                break

                        # Train style: speed should correlate with arousal
                        self.optimizer.zero_grad()
                        speed_raw = self.model.movement_mood.speed_net(target_pad.unsqueeze(0))
                        speed = 0.7 + 0.6 * torch.sigmoid(speed_raw).squeeze()
                        target_speed = 1.0 + 0.3 * target_pad[1]
                        style_loss = (speed - target_speed).pow(2)
                        style_loss.backward()
                        self.optimizer.step()

                    if (epoch + 1) % 10 == 0:
                        print(f"    Epoch {epoch+1}: style_loss={style_loss.item():.4f}")
                env.close()

        self.save("phase8_final")
        print(f"\n[DONE] Phase 8. Jack is ready!")
        print(f"  Run 'python VirtualWorld.py' to meet him.")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Jack Training Pipeline")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["0", "2", "8", "all"],
                        help="Training phase to run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    args = parser.parse_args()

    pipeline = TrainingPipeline()

    if args.phase == "0":
        pipeline.train_phase0(epochs=args.epochs or 50)
    elif args.phase == "2":
        pipeline.train_phase2(total_timesteps=args.timesteps)
    elif args.phase == "8":
        pipeline.train_phase8(epochs=args.epochs or 100)
    elif args.phase == "all":
        pipeline.train_phase0(epochs=args.epochs or 50)
        pipeline.train_phase2(total_timesteps=args.timesteps)
        pipeline.train_phase8(epochs=args.epochs or 100)


if __name__ == "__main__":
    main()
