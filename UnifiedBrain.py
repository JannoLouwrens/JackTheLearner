"""
UNIFIED BRAIN - ONE NEURAL NETWORK FOR ALL PHASES (SOTA 2025)

This replaces the separate networks (MathReasoner, ScalableRobotBrain, WorldModel, etc.)
with ONE unified transformer that does everything.

SOTA Practices:
1. RMSNorm (LLaMA style) - faster than LayerNorm
2. SwiGLU activation (PaLM/LLaMA) - better than GELU
3. Rotary Position Embeddings (RoPE) - better position encoding
4. Pre-norm architecture - more stable training
5. Joint tokenization - each joint is a token
6. Multi-task heads - action, physics, world model, value, skill

Architecture:
    State -> Tokenizer -> [CLS] [JOINT_1...17] [BODY] [GOAL] [ACTION_1...K]
                                        |
                            Transformer Backbone (8 layers)
                            + Cross-Attention to Physics Rules
                                        |
              +-----------+-----------+-----------+-----------+
              |           |           |           |           |
          ActionHead  PhysicsHead  WorldHead  ValueHead  SkillHead

Training:
    Phase 0: PhysicsHead learns from SymPy
    Phase 1: All heads train with RL
    Phase 2: All heads refine with imitation

References:
- LLaMA (2023): RMSNorm, SwiGLU, RoPE
- Decision Transformer (2021): RL as sequence modeling
- Gato (2022): One model for all tasks
- AlphaGeometry (2024): Neuro-symbolic reasoning
- pi0 (2024): Flow matching for robotics

Author: Janno Louwrens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class UnifiedBrainConfig:
    """Configuration for unified brain"""
    # Model
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # Tokenization
    num_joints: int = 17
    features_per_joint: int = 8
    max_seq_len: int = 64

    # Action
    action_dim: int = 17
    action_chunk_size: int = 16

    # Physics rules
    num_rules: int = 100
    rule_dim: int = 256

    # World model
    latent_dim: int = 256

    # Skills
    num_skills: int = 20

    # Input
    obs_dim: int = 256

    # Flow matching
    use_flow_matching: bool = True


# ==============================================================================
# SOTA COMPONENTS
# ==============================================================================

class RMSNorm(nn.Module):
    """RMSNorm (LLaMA style) - faster than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation (PaLM/LLaMA) - better than GELU"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(q, k, cos, sin):
    """Apply rotary position embedding"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# ATTENTION
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        cos, sin = self.rope(L, x.device)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class CrossAttention(nn.Module):
    """Cross-attention to physics rule bank"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.rule_dim, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.rule_dim, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rules):
        B, L, D = x.shape
        num_rules = rules.shape[0]

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        rules_exp = rules.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(rules_exp).view(B, num_rules, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(rules_exp).view(B, num_rules, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = self.dropout(F.softmax(attn, dim=-1))
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)

        rule_weights = attn_weights.mean(dim=1).mean(dim=1)
        return self.out(out), rule_weights


# ==============================================================================
# TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, RoPE, SwiGLU, and optional cross-attention"""
    def __init__(self, config: UnifiedBrainConfig, use_cross_attn: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm2 = RMSNorm(config.d_model)
            self.cross_attn = CrossAttention(config)

        self.norm3 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(self, x, rules=None, mask=None):
        x = x + self.self_attn(self.norm1(x), mask)

        rule_weights = None
        if self.use_cross_attn and rules is not None:
            cross_out, rule_weights = self.cross_attn(self.norm2(x), rules)
            x = x + cross_out

        x = x + self.ffn(self.norm3(x))
        return x, rule_weights


# ==============================================================================
# TOKENIZER
# ==============================================================================

class JointTokenizer(nn.Module):
    """Tokenize robot state into joint tokens"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        self.joint_embed = nn.Linear(config.features_per_joint, config.d_model)

        remaining = config.obs_dim - (config.num_joints * config.features_per_joint)
        self.remaining_dim = max(remaining, 0)
        if self.remaining_dim > 0:
            self.body_embed = nn.Linear(self.remaining_dim, config.d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.goal_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.action_tokens = nn.Parameter(torch.randn(1, config.action_chunk_size, config.d_model) * 0.02)

        self.token_type_embed = nn.Embedding(5, config.d_model)
        self.action_embed = nn.Linear(config.action_dim, config.d_model)

    def forward(self, state, goal=None, noisy_actions=None):
        B = state.shape[0]
        device = state.device
        tokens, types = [], []

        # [CLS]
        tokens.append(self.cls_token.expand(B, -1, -1))
        types.append(torch.zeros(B, 1, dtype=torch.long, device=device))

        # Joint tokens
        joint_dim = self.config.num_joints * self.config.features_per_joint
        joint_feat = state[:, :min(joint_dim, state.shape[1])]
        if joint_feat.shape[1] < joint_dim:
            joint_feat = F.pad(joint_feat, (0, joint_dim - joint_feat.shape[1]))
        joint_feat = joint_feat.view(B, self.config.num_joints, self.config.features_per_joint)
        tokens.append(self.joint_embed(joint_feat))
        types.append(torch.ones(B, self.config.num_joints, dtype=torch.long, device=device))

        # Body token
        if self.remaining_dim > 0 and state.shape[1] > joint_dim:
            body_feat = state[:, joint_dim:joint_dim + self.remaining_dim]
            if body_feat.shape[1] < self.remaining_dim:
                body_feat = F.pad(body_feat, (0, self.remaining_dim - body_feat.shape[1]))
            tokens.append(self.body_embed(body_feat).unsqueeze(1))
            types.append(torch.full((B, 1), 2, dtype=torch.long, device=device))

        # Goal token
        if goal is not None:
            tokens.append(self.goal_token.expand(B, -1, -1))
            types.append(torch.full((B, 1), 3, dtype=torch.long, device=device))

        # Action tokens
        if noisy_actions is not None:
            tokens.append(self.action_embed(noisy_actions))
        else:
            tokens.append(self.action_tokens.expand(B, -1, -1))
        types.append(torch.full((B, self.config.action_chunk_size), 4, dtype=torch.long, device=device))

        tokens = torch.cat(tokens, dim=1)
        types = torch.cat(types, dim=1)
        tokens = tokens + self.token_type_embed(types)

        mask = torch.ones(B, tokens.shape[1], dtype=torch.bool, device=device)
        return tokens, mask


# ==============================================================================
# PHYSICS RULE BANK
# ==============================================================================

class PhysicsRuleBank(nn.Module):
    """Learnable physics rules for neuro-symbolic reasoning"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.rules = nn.Parameter(torch.randn(config.num_rules, config.rule_dim) * 0.02)
        self.rule_names = [
            "F=ma", "torque", "momentum", "energy_conservation", "work",
            "angular_momentum", "angular_accel", "pendulum",
            "velocity", "position", "acceleration",
            "center_of_mass", "torque_balance", "static_friction", "kinetic_friction",
            "kinetic_energy", "potential_energy", "total_energy", "power",
            "hip_knee_coupling", "ankle_balance", "arm_swing", "symmetry",
            "stance_swing", "ground_reaction",
        ] + [f"learned_{i}" for i in range(config.num_rules - 25)]

    def forward(self):
        return self.rules


# ==============================================================================
# OUTPUT HEADS
# ==============================================================================

class ActionHead(nn.Module):
    """Action prediction with flow matching support"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.decoder = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.action_dim),
        )
        self.velocity_decoder = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.action_dim),
        )

    def forward(self, x):
        return self.decoder(x)

    def predict_velocity(self, x):
        return self.velocity_decoder(x)


class PhysicsHead(nn.Module):
    """Physics quantity prediction"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 10),
        )

    def forward(self, x):
        return self.predictor(x)


class WorldHead(nn.Module):
    """World model for next state prediction"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.encoder = nn.Linear(config.d_model, config.latent_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.obs_dim),
        )
        self.reward_pred = nn.Linear(config.latent_dim, 1)

    def forward(self, cls_feat, action):
        latent = F.silu(self.encoder(cls_feat))
        next_latent = self.dynamics(torch.cat([latent, action], dim=-1))
        return self.decoder(next_latent), self.reward_pred(next_latent), next_latent


class ValueHead(nn.Module):
    """Value function for RL"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, x):
        return self.predictor(x)


class SkillHead(nn.Module):
    """Skill prediction for hierarchical planning"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.skill_embeds = nn.Parameter(torch.randn(config.num_skills, config.d_model) * 0.02)
        self.predictor = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.num_skills),
        )

    def forward(self, x):
        logits = self.predictor(x)
        probs = F.softmax(logits, dim=-1)
        skill_embed = torch.einsum('bn,nd->bd', probs, self.skill_embeds)
        return logits, skill_embed


# ==============================================================================
# UNIFIED BRAIN
# ==============================================================================

class UnifiedBrain(nn.Module):
    """
    ONE NEURAL NETWORK FOR EVERYTHING

    Combines:
    - MathReasoner (physics)
    - ScalableRobotBrain (action)
    - WorldModel (imagination)
    - HierarchicalPlanner (skills)
    - Value function (RL)

    Into ONE transformer with multiple heads.
    """

    def __init__(self, config: UnifiedBrainConfig = None):
        super().__init__()
        self.config = config or UnifiedBrainConfig()
        config = self.config

        print("\n" + "=" * 70)
        print("UNIFIED BRAIN - ONE NETWORK FOR ALL PHASES")
        print("=" * 70)
        print(f"  d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
        print(f"  num_joints={config.num_joints}, action_chunk={config.action_chunk_size}")
        print(f"  num_rules={config.num_rules}, num_skills={config.num_skills}")

        # Tokenizer
        self.tokenizer = JointTokenizer(config)

        # Physics rules
        self.rule_bank = PhysicsRuleBank(config)

        # Transformer backbone (cross-attention every 2 layers)
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_cross_attn=(i % 2 == 1))
            for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

        # Output heads
        self.action_head = ActionHead(config)
        self.physics_head = PhysicsHead(config)
        self.world_head = WorldHead(config)
        self.value_head = ValueHead(config)
        self.skill_head = SkillHead(config)

        # Initialize
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB")
        print("=" * 70 + "\n")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,
        goal: torch.Tensor = None,
        noisy_actions: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.

        Args:
            state: (B, obs_dim) - robot state
            action: (B, action_dim) - for world model prediction
            goal: (B, obs_dim) - optional goal state
            noisy_actions: (B, chunk, action_dim) - for flow matching

        Returns:
            Dict with all predictions
        """
        # Tokenize
        tokens, mask = self.tokenizer(state, goal, noisy_actions)
        rules = self.rule_bank()

        # Transformer
        all_rule_weights = []
        for layer in self.layers:
            tokens, rule_weights = layer(tokens, rules, mask)
            if rule_weights is not None:
                all_rule_weights.append(rule_weights)

        tokens = self.final_norm(tokens)

        # Extract features
        cls_feat = tokens[:, 0, :]
        action_feat = tokens[:, -self.config.action_chunk_size:, :]

        # Average rule weights
        if all_rule_weights:
            avg_rule_weights = torch.stack(all_rule_weights).mean(0)
        else:
            avg_rule_weights = torch.zeros(state.shape[0], self.config.num_rules, device=state.device)

        # All predictions
        output = {
            'cls_features': cls_feat,
            'rule_weights': avg_rule_weights,
            'actions': self.action_head(action_feat),
            'physics': self.physics_head(cls_feat),
            'value': self.value_head(cls_feat),
        }

        # Skill
        skill_logits, skill_embed = self.skill_head(cls_feat)
        output['skill_logits'] = skill_logits
        output['skill_embed'] = skill_embed

        # World model (if action provided)
        if action is not None:
            next_state, reward, next_latent = self.world_head(cls_feat, action)
            output['next_state'] = next_state
            output['reward'] = reward
            output['next_latent'] = next_latent

        return output

    def predict_action(self, state: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """Simple action prediction"""
        output = self.forward(state, goal=goal)
        return output['actions'][:, 0, :]

    def get_active_rules(self, state: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Show which physics rules are active"""
        with torch.no_grad():
            output = self.forward(state)
            weights = output['rule_weights'][0]
        top_w, top_i = torch.topk(weights, k=top_k)
        return [(self.rule_bank.rule_names[i.item()], w.item()) for i, w in zip(top_i, top_w)]


# ==============================================================================
# TRAINING LOSSES
# ==============================================================================

def compute_physics_loss(model, state, action, next_state, physics_targets):
    """Phase 0: Learn physics from SymPy"""
    output = model(state, action=action)

    physics_loss = F.mse_loss(output['physics'], physics_targets)
    dynamics_loss = F.mse_loss(output['next_state'], next_state)

    # Rule diversity
    rule_weights = output['rule_weights']
    entropy = -(rule_weights * torch.log(rule_weights + 1e-8)).sum(-1).mean()
    diversity_loss = -entropy * 0.01

    total = physics_loss + 0.1 * dynamics_loss + diversity_loss

    return total, {
        'physics': physics_loss.item(),
        'dynamics': dynamics_loss.item(),
        'total': total.item(),
    }


def compute_flow_matching_loss(model, state, target_actions, goal=None):
    """Phase 1/2: Flow matching for action generation"""
    B = state.shape[0]
    device = state.device

    t = torch.rand(B, device=device)
    noise = torch.randn_like(target_actions)
    t_exp = t[:, None, None]
    noisy = (1 - t_exp) * noise + t_exp * target_actions
    target_velocity = target_actions - noise

    output = model(state, goal=goal, noisy_actions=noisy)
    pred_velocity = model.action_head.predict_velocity(
        model.final_norm(model.tokenizer(state, goal, noisy)[0])[:, -model.config.action_chunk_size:]
    )

    return F.mse_loss(pred_velocity, target_velocity)


def compute_rl_loss(model, state, action, reward, next_state, done, gamma=0.99):
    """Phase 1: RL value/policy loss"""
    output = model(state, action=action)

    # Value loss (TD error)
    with torch.no_grad():
        next_output = model(next_state)
        target_value = reward + gamma * (1 - done) * next_output['value'].squeeze()

    value_loss = F.mse_loss(output['value'].squeeze(), target_value)

    # World model loss
    world_loss = F.mse_loss(output['next_state'], next_state)
    reward_loss = F.mse_loss(output['reward'].squeeze(), reward)

    return value_loss + 0.1 * world_loss + 0.1 * reward_loss


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("[*] UnifiedBrain - ONE Network Test\n")

    config = UnifiedBrainConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        num_joints=17,
        features_per_joint=8,
        action_dim=17,
        action_chunk_size=16,
        num_rules=100,
        obs_dim=256,
    )

    model = UnifiedBrain(config)

    # Test
    B = 4
    state = torch.randn(B, 256)
    action = torch.randn(B, 17)
    goal = torch.randn(B, 256)

    print("[*] Testing forward pass...")
    output = model(state, action=action, goal=goal)

    print(f"[OK] Actions: {output['actions'].shape}")
    print(f"[OK] Physics: {output['physics'].shape}")
    print(f"[OK] Value: {output['value'].shape}")
    print(f"[OK] Next state: {output['next_state'].shape}")
    print(f"[OK] Skills: {output['skill_logits'].shape}")
    print(f"[OK] Rules: {output['rule_weights'].shape}")

    # Interpretability
    print("\n[*] Active physics rules:")
    for name, weight in model.get_active_rules(state[:1]):
        print(f"  - {name}: {weight:.3f}")

    # Test losses
    print("\n[*] Testing losses...")
    next_state = torch.randn(B, 256)
    physics_targets = torch.randn(B, 10)
    target_actions = torch.randn(B, 16, 17)

    loss, metrics = compute_physics_loss(model, state, action, next_state, physics_targets)
    print(f"[OK] Physics loss: {metrics['physics']:.4f}")

    flow_loss = compute_flow_matching_loss(model, state, target_actions)
    print(f"[OK] Flow matching loss: {flow_loss.item():.4f}")

    print("\n" + "=" * 70)
    print("[SUCCESS] UnifiedBrain validated!")
    print("=" * 70)
    print("\nThis ONE network replaces:")
    print("  - MathReasoner")
    print("  - ScalableRobotBrain")
    print("  - WorldModel")
    print("  - HierarchicalPlanner")
    print("  - ValueHead")
    print("\nAll in ONE transformer with shared backbone!")
    print("=" * 70)
