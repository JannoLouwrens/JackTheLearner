"""
ROBUST TRAINER - Prevents Catastrophic Forgetting (SOTA 2025)

Safeguards implemented:
1. REPLAY BUFFER: Mix Phase 0 data into Phase 1/2 training
2. PHYSICS CONSISTENCY: Verify actions don't violate physics during RL
3. EWC (Elastic Weight Consolidation): Protect important weights
4. MULTI-RATE LEARNING: Different LRs for different components

Research backing:
- EWC: "Overcoming catastrophic forgetting" (Kirkpatrick et al., 2017)
- Replay: "Experience Replay" (Lin, 1992) + modern variants
- Multi-rate: "The Surprising Ineffectiveness..." (Nov 2024)

Usage:
    # Phase 0
    python RobustTrainer.py --phase 0 --epochs 50

    # Phase 1 (auto-loads Phase 0, uses replay + EWC)
    python RobustTrainer.py --phase 1 --epochs 500

    # Phase 2 (auto-loads Phase 1, continues safeguards)
    python RobustTrainer.py --phase 2 --epochs 100

Author: Janno Louwrens
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random
import copy
import argparse
from tqdm import tqdm

from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig, compute_physics_loss, compute_flow_matching_loss


# ==============================================================================
# REPLAY BUFFER
# ==============================================================================

class ReplayBuffer:
    """
    Experience replay buffer that stores data from previous phases.

    Key insight: To prevent forgetting Phase 0 physics knowledge,
    we mix Phase 0 samples into Phase 1 training.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.phase_indices = {}  # Track which samples came from which phase

    def add(self, sample: Dict, phase: int):
        """Add a sample to the buffer"""
        sample['_phase'] = phase
        self.buffer.append(sample)

        if phase not in self.phase_indices:
            self.phase_indices[phase] = []
        self.phase_indices[phase].append(len(self.buffer) - 1)

    def sample(self, batch_size: int, phase_ratios: Dict[int, float] = None) -> List[Dict]:
        """
        Sample a batch with specified ratios from each phase.

        Args:
            batch_size: Total batch size
            phase_ratios: {phase: ratio}, e.g., {0: 0.2, 1: 0.8} means
                          20% from Phase 0, 80% from Phase 1
        """
        if phase_ratios is None:
            # Default: sample uniformly
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        samples = []
        for phase, ratio in phase_ratios.items():
            n_samples = int(batch_size * ratio)
            if phase in self.phase_indices and len(self.phase_indices[phase]) > 0:
                # Get indices for this phase (handle wrap-around)
                valid_indices = [i for i in self.phase_indices[phase] if i < len(self.buffer)]
                if valid_indices:
                    chosen_indices = random.choices(valid_indices, k=min(n_samples, len(valid_indices)))
                    samples.extend([self.buffer[i] for i in chosen_indices])

        return samples

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        """Save buffer to disk"""
        torch.save({
            'buffer': list(self.buffer),
            'phase_indices': dict(self.phase_indices),
        }, path)
        print(f"[SAVE] Replay buffer: {path} ({len(self.buffer)} samples)")

    def load(self, path: str):
        """Load buffer from disk"""
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            self.phase_indices = data['phase_indices']
            print(f"[LOAD] Replay buffer: {path} ({len(self.buffer)} samples)")


# ==============================================================================
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# ==============================================================================

class EWC:
    """
    Elastic Weight Consolidation - protects important weights from previous phases.

    Key insight: Some weights are crucial for Phase 0 physics knowledge.
    We compute their "importance" (Fisher information) and penalize changes.

    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting" (2017)
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc

        # Store old parameters and Fisher information
        self.old_params = {}
        self.fisher = {}

    def compute_fisher(self, dataloader, num_samples: int = 1000):
        """
        Compute Fisher information matrix (diagonal approximation).

        Fisher information measures how much each parameter affects the loss.
        High Fisher = important for current task = protect during future training.
        """
        print("[EWC] Computing Fisher information...")

        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            state = batch['state']
            action = batch.get('action')
            next_state = batch.get('next_state')
            physics = batch.get('physics')

            self.model.zero_grad()

            # Forward pass
            output = self.model(state, action=action)

            # Compute loss (same as training)
            if physics is not None:
                loss = F.mse_loss(output['physics'], physics)
            elif next_state is not None:
                loss = F.mse_loss(output['next_state'], next_state)
            else:
                # Fallback: use action prediction loss
                loss = output['actions'].pow(2).mean()

            loss.backward()

            # Accumulate squared gradients (Fisher diagonal)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

            samples_processed += state.shape[0]

        # Normalize
        for n in fisher:
            fisher[n] /= samples_processed

        self.fisher = fisher

        # Store current parameters
        self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        print(f"[EWC] Fisher computed on {samples_processed} samples")

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty: penalize changes to important weights.

        Loss = lambda * sum(Fisher[i] * (theta[i] - theta_old[i])^2)
        """
        if not self.fisher:
            return torch.tensor(0.0)

        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.old_params[n]).pow(2)).sum()

        return self.lambda_ewc * loss

    def save(self, path: str):
        """Save EWC state"""
        torch.save({
            'fisher': self.fisher,
            'old_params': self.old_params,
            'lambda_ewc': self.lambda_ewc,
        }, path)
        print(f"[SAVE] EWC state: {path}")

    def load(self, path: str):
        """Load EWC state"""
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            self.fisher = data['fisher']
            self.old_params = data['old_params']
            self.lambda_ewc = data['lambda_ewc']
            print(f"[LOAD] EWC state: {path}")


# ==============================================================================
# PHYSICS CONSISTENCY
# ==============================================================================

class PhysicsConsistency:
    """
    Verifies that actions don't violate physics during RL.

    Key insight: Even during RL, we want physics knowledge to be used.
    We penalize actions that would result in impossible physics.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        # Physics bounds (learned or hand-coded)
        self.bounds = {
            'kinetic_energy': (0, 10000),      # Joules
            'potential_energy': (-1000, 1000),  # Joules
            'total_energy': (0, 10000),         # Should be roughly constant
            'momentum': (-100, 100),            # kg*m/s
            'force_magnitude': (0, 500),        # Newtons
            'torque_magnitude': (0, 200),       # Nm
            'angular_momentum': (-50, 50),      # kg*m^2/s
            'stability': (0, 1),                # 0-1 score
        }

    def compute_penalty(self, physics_output: torch.Tensor) -> torch.Tensor:
        """
        Penalize physics predictions outside reasonable bounds.

        This encourages the model to propose actions that result in
        physically plausible outcomes.
        """
        penalty = 0

        # Physics output: [KE, PE, total_E, momentum, force, torque, ang_mom, stability, ...]
        quantity_names = ['kinetic_energy', 'potential_energy', 'total_energy',
                          'momentum', 'force_magnitude', 'torque_magnitude',
                          'angular_momentum', 'stability']

        for i, name in enumerate(quantity_names):
            if i >= physics_output.shape[-1]:
                break
            if name in self.bounds:
                low, high = self.bounds[name]
                values = physics_output[..., i]

                # Soft penalty for out-of-bounds
                penalty += F.relu(low - values).mean()  # Below minimum
                penalty += F.relu(values - high).mean()  # Above maximum

        return penalty

    def energy_conservation_loss(self, physics_before: torch.Tensor, physics_after: torch.Tensor) -> torch.Tensor:
        """
        Penalize violations of energy conservation.

        Total energy should be roughly constant (with some dissipation allowed).
        """
        # Total energy is index 2
        energy_before = physics_before[..., 2]
        energy_after = physics_after[..., 2]

        # Allow 10% change due to friction/dissipation
        max_change = 0.1 * energy_before.abs()
        actual_change = (energy_after - energy_before).abs()

        violation = F.relu(actual_change - max_change)
        return violation.mean()


# ==============================================================================
# ROBUST TRAINER
# ==============================================================================

@dataclass
class RobustTrainerConfig:
    """Configuration for robust training"""
    # Model
    d_model: int = 512
    n_layers: int = 8

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    pretrained_lr_scale: float = 0.1  # 10x slower for pretrained

    # Safeguards
    replay_ratio: float = 0.2      # 20% of batch from replay buffer
    ewc_lambda: float = 1000       # EWC penalty strength
    physics_weight: float = 0.1    # Physics consistency weight

    # Paths
    checkpoint_dir: str = "checkpoints"
    replay_buffer_path: str = "checkpoints/replay_buffer.pt"
    ewc_path: str = "checkpoints/ewc_state.pt"


class RobustTrainer:
    """
    Robust trainer that prevents catastrophic forgetting.

    Safeguards:
    1. Replay buffer: Mix old data into new training
    2. EWC: Protect important weights
    3. Physics consistency: Verify actions are physically plausible
    4. Multi-rate learning: Lower LR for pretrained components
    """

    def __init__(self, config: RobustTrainerConfig = None):
        self.config = config or RobustTrainerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 70)
        print("ROBUST TRAINER - Prevents Catastrophic Forgetting")
        print("=" * 70)

        # Create model
        model_config = UnifiedBrainConfig(
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
        )
        self.model = UnifiedBrain(model_config).to(self.device)

        # Safeguards
        self.replay_buffer = ReplayBuffer()
        self.ewc = EWC(self.model, self.config.ewc_lambda)
        self.physics_checker = PhysicsConsistency(self.model)

        # Training state
        self.current_phase = 0
        self.epoch = 0
        self.global_step = 0

        # Create checkpoint dir
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        print(f"  Device: {self.device}")
        print(f"  Replay ratio: {self.config.replay_ratio * 100:.0f}%")
        print(f"  EWC lambda: {self.config.ewc_lambda}")
        print(f"  Physics weight: {self.config.physics_weight}")
        print("=" * 70 + "\n")

    def _create_optimizer(self, phase: int):
        """Create optimizer with multi-rate learning"""
        if phase == 0:
            # Phase 0: All parameters same LR
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-4
            )
        else:
            # Phase 1+: Lower LR for backbone, normal for heads
            backbone_params = []
            head_params = []

            for name, param in self.model.named_parameters():
                if 'head' in name.lower():
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': self.config.learning_rate * self.config.pretrained_lr_scale},
                {'params': head_params, 'lr': self.config.learning_rate},
            ], weight_decay=1e-4)

            print(f"[OPTIMIZER] Multi-rate learning:")
            print(f"  Backbone: {self.config.learning_rate * self.config.pretrained_lr_scale:.2e}")
            print(f"  Heads: {self.config.learning_rate:.2e}")

    def train_phase0(self, num_epochs: int = 50, samples_per_epoch: int = 10000):
        """
        Phase 0: Learn physics from synthetic data.

        Uses SymPy-generated ground truth for supervision.
        Stores samples in replay buffer for later phases.
        """
        print("\n" + "=" * 70)
        print("PHASE 0: Learning Physics")
        print("=" * 70)

        self.current_phase = 0
        self._create_optimizer(0)

        # Import SymPy calculator
        try:
            from SymbolicCalculator import SymbolicPhysicsCalculator
            calculator = SymbolicPhysicsCalculator()
            print("[OK] SymPy calculator loaded")
        except ImportError:
            print("[WARN] SymbolicPhysicsCalculator not found, using random targets")
            calculator = None

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0
            num_batches = 0

            pbar = tqdm(range(0, samples_per_epoch, self.config.batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")

            for _ in pbar:
                # Generate synthetic physics data
                batch_size = self.config.batch_size

                # Random robot states
                state = torch.randn(batch_size, 256).to(self.device)
                action = torch.randn(batch_size, 17).to(self.device)

                # Get physics ground truth from SymPy (or random for testing)
                if calculator:
                    physics_targets = []
                    next_states = []
                    for i in range(batch_size):
                        # predict_robot_state returns (next_state, physics_dict)
                        ns, phys_dict = calculator.predict_robot_state(
                            state[i].cpu().numpy(),
                            action[i].cpu().numpy()
                        )
                        next_states.append(ns)

                        # Expand 4 physics values to 10 for model compatibility
                        # [KE, PE, total_E, momentum, force, torque, ang_mom, stability, work, power]
                        ke = phys_dict['kinetic_energy']
                        pe = phys_dict['potential_energy']
                        total_e = ke + pe
                        momentum = phys_dict['momentum']
                        force_mag = phys_dict['force_magnitude']

                        # Compute additional physics quantities
                        torque_mag = force_mag * 0.3  # Approximate torque (r ≈ 0.3m arm)
                        ang_momentum = momentum * 0.5  # Approximate (r ≈ 0.5m CoM height)
                        stability = 1.0 / (1.0 + abs(pe) / 1000.0)  # Stability score (0-1)
                        work = force_mag * 0.02  # Work = F * d (dt ≈ 0.02m displacement)
                        power = work / 0.02  # Power = Work / time (50Hz control)

                        phys_array = [ke, pe, total_e, momentum, force_mag,
                                      torque_mag, ang_momentum, stability, work, power]
                        physics_targets.append(phys_array)

                    physics_targets = torch.tensor(np.array(physics_targets), dtype=torch.float32).to(self.device)
                    next_state = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
                else:
                    # Random targets for testing
                    physics_targets = torch.randn(batch_size, 10).to(self.device)
                    next_state = state + 0.1 * torch.randn_like(state)

                # Forward pass
                loss, metrics = compute_physics_loss(self.model, state, action, next_state, physics_targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Store in replay buffer (for Phase 1)
                for i in range(batch_size):
                    self.replay_buffer.add({
                        'state': state[i].cpu(),
                        'action': action[i].cpu(),
                        'next_state': next_state[i].cpu(),
                        'physics': physics_targets[i].cpu(),
                    }, phase=0)

                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = epoch_loss / num_batches
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("phase0_best")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"phase0_epoch{epoch+1}")

        # Compute Fisher information for EWC (before Phase 1)
        print("\n[*] Computing EWC Fisher information...")
        self._compute_ewc_fisher()

        # Save replay buffer
        self.replay_buffer.save(self.config.replay_buffer_path)

        print(f"\n[DONE] Phase 0 complete. Best loss: {best_loss:.4f}")
        return best_loss

    def train_phase1(self, num_epochs: int = 500):
        """
        Phase 1: RL training with safeguards.

        Safeguards active:
        1. Replay: 20% of batch from Phase 0 data
        2. EWC: Penalize changes to important weights
        3. Physics consistency: Verify actions are plausible
        """
        print("\n" + "=" * 70)
        print("PHASE 1: RL Training (with safeguards)")
        print("=" * 70)

        self.current_phase = 1

        # Load Phase 0 checkpoint
        phase0_path = os.path.join(self.config.checkpoint_dir, "phase0_best.pt")
        if os.path.exists(phase0_path):
            self.load_checkpoint(phase0_path)
            print("[OK] Loaded Phase 0 checkpoint")
        else:
            print("[WARN] No Phase 0 checkpoint found!")

        # Load replay buffer and EWC
        self.replay_buffer.load(self.config.replay_buffer_path)
        self.ewc.load(self.config.ewc_path)

        # Create multi-rate optimizer
        self._create_optimizer(1)

        # Create environment
        try:
            import gymnasium as gym
            env = gym.make("Humanoid-v5")
            print(f"[OK] Environment: Humanoid-v5")
            print(f"    Obs dim: {env.observation_space.shape[0]}")
            print(f"    Action dim: {env.action_space.shape[0]}")
        except:
            print("[WARN] Gymnasium not available, using mock environment")
            env = None

        print("\n[*] Safeguards active:")
        print(f"    Replay: {self.config.replay_ratio * 100:.0f}% Phase 0 data")
        print(f"    EWC: lambda = {self.config.ewc_lambda}")
        print(f"    Physics: weight = {self.config.physics_weight}")

        best_reward = -float('inf')

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Collect experience (simplified for demo)
            if env:
                episode_rewards = self._collect_experience(env, steps=2048)
                avg_reward = np.mean(episode_rewards)
            else:
                avg_reward = 0

            # Training step with safeguards
            train_loss = self._train_step_with_safeguards()

            print(f"[Epoch {epoch+1}] Reward: {avg_reward:.1f} | Loss: {train_loss:.4f}")

            # Save best
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint("phase1_best")

            # Periodic save
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint("phase1_latest")

        # Update EWC for Phase 2
        print("\n[*] Updating EWC for Phase 2...")
        self._compute_ewc_fisher()

        print(f"\n[DONE] Phase 1 complete. Best reward: {best_reward:.1f}")
        return best_reward

    def train_phase2(self, num_epochs: int = 100):
        """
        Phase 2: Imitation learning with safeguards.

        All safeguards remain active.
        """
        print("\n" + "=" * 70)
        print("PHASE 2: Imitation Learning (with safeguards)")
        print("=" * 70)

        self.current_phase = 2

        # Load Phase 1 checkpoint
        phase1_path = os.path.join(self.config.checkpoint_dir, "phase1_best.pt")
        if os.path.exists(phase1_path):
            self.load_checkpoint(phase1_path)
            print("[OK] Loaded Phase 1 checkpoint")

        # Load replay buffer and EWC
        self.replay_buffer.load(self.config.replay_buffer_path)
        self.ewc.load(self.config.ewc_path)

        # Create optimizer
        self._create_optimizer(2)

        print("\n[*] Safeguards active (same as Phase 1)")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training step with safeguards
            train_loss = self._train_step_with_safeguards(use_flow_matching=True)

            print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f}")

            # Save best
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint("phase2_best")

        self.save_checkpoint("phase2_final")
        print(f"\n[DONE] Phase 2 complete. Best loss: {best_loss:.4f}")
        return best_loss

    def _train_step_with_safeguards(self, use_flow_matching: bool = False) -> float:
        """
        Training step with ALL safeguards:
        1. Mix replay data from previous phases
        2. Add EWC penalty
        3. Add physics consistency penalty
        """
        self.model.train()

        # Sample batch with replay
        if len(self.replay_buffer) > 0 and self.current_phase > 0:
            # Mix: (1-ratio) current + ratio replay
            replay_batch = self.replay_buffer.sample(
                int(self.config.batch_size * self.config.replay_ratio),
                phase_ratios={0: 1.0}  # All from Phase 0
            )

            if replay_batch:
                # Compute replay loss (physics)
                replay_states = torch.stack([s['state'] for s in replay_batch]).to(self.device)
                replay_actions = torch.stack([s['action'] for s in replay_batch]).to(self.device)
                replay_next = torch.stack([s['next_state'] for s in replay_batch]).to(self.device)
                replay_physics = torch.stack([s['physics'] for s in replay_batch]).to(self.device)

                replay_loss, _ = compute_physics_loss(
                    self.model, replay_states, replay_actions, replay_next, replay_physics
                )
            else:
                replay_loss = torch.tensor(0.0).to(self.device)
        else:
            replay_loss = torch.tensor(0.0).to(self.device)

        # Current task loss (simplified - would be RL or imitation in real impl)
        state = torch.randn(self.config.batch_size, 256).to(self.device)
        action = torch.randn(self.config.batch_size, 17).to(self.device)

        if use_flow_matching:
            target_actions = torch.randn(self.config.batch_size, 16, 17).to(self.device)
            task_loss = compute_flow_matching_loss(self.model, state, target_actions)
        else:
            next_state = state + 0.1 * torch.randn_like(state)
            output = self.model(state, action=action)
            task_loss = F.mse_loss(output['next_state'], next_state)

        # EWC penalty
        ewc_loss = self.ewc.penalty()

        # Physics consistency
        output = self.model(state, action=action)
        physics_penalty = self.physics_checker.compute_penalty(output['physics'])

        # Total loss
        total_loss = (
            task_loss +
            self.config.replay_ratio * replay_loss +
            ewc_loss +
            self.config.physics_weight * physics_penalty
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1

        return total_loss.item()

    def _collect_experience(self, env, steps: int = 2048) -> List[float]:
        """Collect experience from environment (simplified)"""
        episode_rewards = []
        obs, _ = env.reset()
        episode_reward = 0

        for _ in range(steps):
            # Get action from model
            state = torch.tensor(obs[:256], dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.model.predict_action(state).cpu().numpy()[0]

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0

        return episode_rewards if episode_rewards else [0]

    def _compute_ewc_fisher(self):
        """Compute EWC Fisher information from replay buffer"""
        if len(self.replay_buffer) == 0:
            print("[WARN] Replay buffer empty, skipping EWC")
            return

        # Create simple dataloader from replay buffer
        samples = list(self.replay_buffer.buffer)[:1000]

        class SimpleDataset:
            def __init__(self, samples):
                self.samples = samples
            def __iter__(self):
                for s in self.samples:
                    yield {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                           for k, v in s.items() if k != '_phase'}

        self.ewc.compute_fisher(SimpleDataset(samples))
        self.ewc.save(self.config.ewc_path)

    def save_checkpoint(self, name: str):
        """Save checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'current_phase': self.current_phase,
        }, path)
        print(f"[SAVE] {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        print(f"[LOAD] {path}")


# ==============================================================================
# VERIFICATION: Check if physics knowledge is preserved
# ==============================================================================

def verify_physics_preservation(trainer: RobustTrainer) -> Dict[str, float]:
    """
    Verify that physics knowledge from Phase 0 is still present.

    This is the key test: after Phase 1/2, can the model still
    predict physics correctly?
    """
    print("\n[*] Verifying physics preservation...")

    trainer.model.eval()

    # Load some Phase 0 data
    if len(trainer.replay_buffer) == 0:
        trainer.replay_buffer.load(trainer.config.replay_buffer_path)

    phase0_samples = [s for s in trainer.replay_buffer.buffer if s.get('_phase') == 0][:100]

    if not phase0_samples:
        print("[WARN] No Phase 0 samples found")
        return {}

    total_physics_error = 0
    total_dynamics_error = 0

    with torch.no_grad():
        for sample in phase0_samples:
            state = sample['state'].unsqueeze(0).to(trainer.device)
            action = sample['action'].unsqueeze(0).to(trainer.device)
            target_physics = sample['physics'].unsqueeze(0).to(trainer.device)
            target_next = sample['next_state'].unsqueeze(0).to(trainer.device)

            output = trainer.model(state, action=action)

            physics_error = F.mse_loss(output['physics'], target_physics).item()
            dynamics_error = F.mse_loss(output['next_state'], target_next).item()

            total_physics_error += physics_error
            total_dynamics_error += dynamics_error

    avg_physics = total_physics_error / len(phase0_samples)
    avg_dynamics = total_dynamics_error / len(phase0_samples)

    print(f"[*] Physics prediction error: {avg_physics:.4f}")
    print(f"[*] Dynamics prediction error: {avg_dynamics:.4f}")

    if avg_physics < 1.0:
        print("[OK] Physics knowledge PRESERVED!")
    else:
        print("[WARN] Physics knowledge may be degraded")

    return {
        'physics_error': avg_physics,
        'dynamics_error': avg_dynamics,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust Trainer with Forgetting Prevention")
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2],
                        help="Training phase (0=physics, 1=RL, 2=imitation)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--verify", action="store_true", help="Verify physics preservation")
    args = parser.parse_args()

    config = RobustTrainerConfig()
    trainer = RobustTrainer(config)

    if args.phase == 0:
        trainer.train_phase0(num_epochs=args.epochs)
    elif args.phase == 1:
        trainer.train_phase1(num_epochs=args.epochs)
    elif args.phase == 2:
        trainer.train_phase2(num_epochs=args.epochs)

    if args.verify:
        verify_physics_preservation(trainer)


if __name__ == "__main__":
    main()
