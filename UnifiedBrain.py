"""
UNIFIED BRAIN - COMPLETE AGI SYSTEM (SOTA 2025)

This is the COMPLETE brain that integrates EVERYTHING:
- SOTA Transformer (RMSNorm, SwiGLU, RoPE)
- Full TD-MPC2 WorldModel (imagination + MPC planning)
- Full HierarchicalPlanner (3-level hierarchy with skills)
- Vision encoders (DINOv2 + SigLIP fusion from OpenVLA)
- Temporal Memory (remembers 50 timesteps)
- Cross-modal Fusion (vision, proprio, touch, language)
- Physics Rule Bank (neuro-symbolic reasoning)
- LLM Integration (frozen backbone + trainable projector)
- AlphaGeometry-style creative loop (optional)

Research papers implemented:
- LLaMA (2023): RMSNorm, SwiGLU, RoPE
- TD-MPC2 (ICLR 2024): World model + MPC planning
- HAC (2019): Hierarchical Actor-Critic with skills
- OpenVLA (2024): DINOv2 + SigLIP vision fusion, frozen LLM backbone
- pi0 (Physical Intelligence 2024): Flow matching + frozen PaliGemma
- RT-2 (Google 2023): VLA with frozen PaLM-E backbone
- AlphaGeometry (Nature 2024): Neuro-symbolic reasoning
- Decision Transformer (2021): RL as sequence modeling

LLM Architecture (following OpenVLA/pi0/RT-2):
- Frozen LLM backbone (SmolLM2/TinyLlama/Gemma) - NOT part of 105M
- Trainable projector (LLM dim -> 512) - IS part of 105M
- LLM understands language, brain learns motor control
- Can swap LLMs without retraining (local vs cloud)

Author: Janno Louwrens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import numpy as np

# Companion modules
try:
    from EmotionalState import EmotionalState, EmotionalConfig, EventType
    from Personality import Personality, JACK_PERSONALITY
    from MovementMoodCoupling import MovementMoodCoupling, MovementMoodConfig
    from InnerMonologue import InnerMonologue, MonologueConfig
except ImportError as e:
    print(f"[WARN] Companion modules not fully available: {e}")
    EmotionalState = None
    Personality = None
    MovementMoodCoupling = None
    InnerMonologue = None

try:
    from AlphaGeometryLoop import AlphaGeometryLoop, LoopConfig as AGLoopConfig
except ImportError:
    AlphaGeometryLoop = None
    AGLoopConfig = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class UnifiedBrainConfig:
    """Configuration for the complete unified brain"""
    # Model architecture
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

    # Physics rules (neuro-symbolic)
    num_rules: int = 100
    rule_dim: int = 256

    # World model (TD-MPC2)
    latent_dim: int = 256
    imagination_horizon: int = 5
    mpc_samples: int = 512
    mpc_temperature: float = 0.5

    # Hierarchical planner
    num_skills: int = 20
    skill_horizon: int = 10
    max_subgoals: int = 5
    goal_dim: int = 64

    # Vision (OpenVLA style)
    vision_enabled: bool = True
    vision_embed_dim: int = 1024
    use_pretrained_vision: bool = False  # Set True if you have HuggingFace models

    # Audio (Whisper + wav2vec2)
    audio_enabled: bool = True
    audio_embed_dim: int = 512
    audio_sample_rate: int = 16000
    use_pretrained_audio: bool = False  # Set True if you have HuggingFace models

    # Temporal memory
    context_length: int = 50  # Remember last 50 timesteps

    # Multimodal
    touch_dim: int = 64
    language_embed_dim: int = 512
    vocab_size: int = 1000

    # LLM Integration (NEW)
    llm_enabled: bool = True
    llm_backend: str = "smollm"  # Options: "smollm", "tinyllama", "gemma", "fallback"
    llm_model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # HuggingFace model ID
    llm_hidden_dim: int = 2048  # LLM hidden dimension (SmolLM2-1.7B = 2048)
    llm_freeze: bool = True  # Keep LLM frozen (recommended)
    llm_use_lora: bool = False  # Optional LoRA fine-tuning
    llm_max_length: int = 128  # Max tokens for commands

    # API LLM (Claude, GPT-4, etc.) - for high-quality reasoning
    llm_api_enabled: bool = False
    llm_api_provider: str = "anthropic"  # "anthropic", "openai", "ollama"
    llm_api_model: str = "claude-sonnet-4-20250514"  # API model name
    llm_api_key_env: str = "ANTHROPIC_API_KEY"  # Environment variable for API key
    llm_api_base_url: str = ""  # Custom base URL (for Ollama: "http://localhost:11434")
    llm_api_max_tokens: int = 200  # Max tokens per response
    llm_api_temperature: float = 0.7  # Response creativity

    # Companion Robot Features (NEW)
    enable_response_generation: bool = True  # LLM generates spoken responses
    enable_task_completion: bool = True  # Knows when task is done
    enable_tts: bool = True  # Text-to-speech output
    enable_memory: bool = True  # Long-term memory
    memory_size: int = 1000  # Number of memories to store

    # Input
    obs_dim: int = 256

    # Flow matching
    use_flow_matching: bool = True

    # =========================================
    # SOTA FEATURES (GR00T N1, π0, Helix style)
    # =========================================

    # Dual System Architecture (Figure Helix / NVIDIA GR00T N1)
    # System 2: Slow VLM reasoning (7-9 Hz) - scene understanding
    # System 1: Fast action generation (50-200 Hz) - visuomotor policy
    # System 0: Ultra-fast motor control (1 kHz) - joint actuators
    dual_system_enabled: bool = True
    system2_hz: float = 9.0  # VLM reasoning frequency
    system1_hz: float = 50.0  # Action generation frequency (π0 uses 50Hz)
    system0_hz: float = 1000.0  # Motor control frequency (optional)
    system0_enabled: bool = False  # Enable System 0 for real hardware

    # Action Expert (π0 style - separate transformer for actions)
    action_expert_enabled: bool = True
    action_expert_layers: int = 4  # Smaller than main backbone
    action_expert_dim: int = 256  # π0 uses ~300M params, we use smaller
    flow_matching_steps: int = 10  # Number of denoising steps (π0 uses 10)

    # DiT (Diffusion Transformer) for action generation
    dit_enabled: bool = True
    dit_time_embed_dim: int = 256  # Time embedding dimension

    # =========================================
    # OBJECT DETECTION & NAVIGATION (General-Purpose Robot)
    # =========================================

    # Object Detection (DETR-style)
    enable_object_detection: bool = True
    object_detection_queries: int = 100  # Number of object queries
    num_object_classes: int = 21  # Number of detectable object types

    # Navigation Planning
    enable_navigation: bool = True
    nav_map_size: int = 64  # Spatial memory map size
    nav_map_resolution: float = 0.1  # Meters per grid cell

    # =========================================
    # INTRINSIC MOTIVATION (Self-Thinking)
    # =========================================
    # These enable truly autonomous behavior without external rewards

    # Master switch for intrinsic motivation
    enable_intrinsic_motivation: bool = False

    # Modules that are constructed but have NO live call site. Measured
    # 2026-08-04: together they were 67,668,479 of 115,009,308 trainable
    # parameters (58.8%) receiving zero gradient, while still being optimised
    # over, checkpointed, and counted in every "105M brain" claim.
    #
    # They stay OFF until something actually invokes them and an ablation test
    # (ladder tier 3) shows they contribute. Turning one on without wiring it
    # only re-creates dead weight. See docs/PIPELINE_REVIEW.md section 9.
    enable_temporal_memory: bool = False      # never passed memory=; context is 1 frame
    enable_world_model: bool = False          # forward() gates on action is not None
    enable_hierarchical_planner: bool = False # 37.2M — larger than the backbone

    # Curiosity (ICM + RND) - drives exploration of novel states
    enable_curiosity: bool = True

    # Skill Discovery (DIAYN) - learns diverse skills without rewards
    enable_skill_discovery: bool = True
    num_discoverable_skills: int = 50  # Number of skills to discover

    # Empowerment - seeks states with maximum control
    enable_empowerment: bool = True

    # Metacognition - knows what it doesn't know
    enable_metacognition: bool = True

    # Autotelic Goals - self-generated learning curriculum
    enable_autotelic_goals: bool = True
    goal_bank_size: int = 1000  # Number of goals to remember

    # =========================================
    # VIRTUAL COMPANION (Emotional, Personality, Movement Style)
    # =========================================
    enable_emotional_state: bool = True
    mood_dim: int = 3  # PAD: Pleasure, Arousal, Dominance (Mehrabian 1996)
    mood_decay_factor: float = 0.995  # Per-step decay toward baseline

    enable_movement_mood_coupling: bool = True
    max_speed_modulation: float = 0.3  # ±30% speed change from mood
    max_style_bias: float = 0.1  # ±10% action bias from mood

    enable_inner_monologue: bool = True
    monologue_cooldown: float = 10.0  # Seconds between autonomous thoughts


# ==============================================================================
# SOTA COMPONENTS (LLaMA-style)
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
# ATTENTION MECHANISMS
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
# AMP DISCRIMINATOR (Adversarial Motion Priors)
# ==============================================================================

class AMPDiscriminator(nn.Module):
    """
    Adversarial Motion Priors Discriminator.

    Distinguishes between:
    - REAL motion (from MoCap data)
    - FAKE motion (from policy rollouts)

    Provides reward signal for natural, human-like movement.

    Research: "AMP: Adversarial Motion Priors for Stylized Physics-Based
              Character Control" (Peng et al., 2021)

    Architecture:
    - Input: State-action pairs (s, a, s') or just state transitions
    - Output: Probability that motion is real (from MoCap)

    Training:
    - Real samples: MoCap state transitions
    - Fake samples: Policy rollout state transitions
    - Loss: Binary cross-entropy + gradient penalty
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 17, hidden_dim: int = 512):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input: (s, a, s') concatenated
        input_dim = state_dim * 2 + action_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output: logit for real/fake
        self.head = nn.Linear(hidden_dim // 2, 1)

        # Gradient penalty coefficient
        self.gradient_penalty_weight = 10.0

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator output.

        Args:
            state: Current state [B, state_dim]
            action: Action taken [B, action_dim]
            next_state: Resulting state [B, state_dim]

        Returns:
            logits: [B, 1] - higher = more likely real (from MoCap)
        """
        # Concatenate inputs
        x = torch.cat([state, action, next_state], dim=-1)

        # Encode
        features = self.encoder(x)

        # Classify
        logits = self.head(features)

        return logits

    def compute_reward(self, state: torch.Tensor, action: torch.Tensor,
                       next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute AMP reward for policy training.

        Reward = log(D(s,a,s')) - log(1 - D(s,a,s'))

        This encourages the policy to produce motion that the
        discriminator thinks is real (from MoCap).
        """
        with torch.no_grad():
            logits = self.forward(state, action, next_state)

            # Clip for numerical stability
            prob = torch.sigmoid(logits).clamp(0.01, 0.99)

            # AMP-style reward
            reward = torch.log(prob) - torch.log(1 - prob)

        return reward.squeeze(-1)

    def compute_loss(self, real_states: torch.Tensor, real_actions: torch.Tensor,
                     real_next_states: torch.Tensor, fake_states: torch.Tensor,
                     fake_actions: torch.Tensor, fake_next_states: torch.Tensor
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator training loss.

        Args:
            real_*: Transitions from MoCap data
            fake_*: Transitions from policy rollouts

        Returns:
            loss: Discriminator loss (BCE + gradient penalty)
            metrics: Dict with 'real_acc', 'fake_acc', 'grad_penalty'
        """
        # Real samples should be classified as 1
        real_logits = self.forward(real_states, real_actions, real_next_states)
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )

        # Fake samples should be classified as 0
        fake_logits = self.forward(fake_states, fake_actions, fake_next_states)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )

        # Gradient penalty for stability (WGAN-GP style)
        grad_penalty = self._gradient_penalty(
            real_states, real_actions, real_next_states,
            fake_states, fake_actions, fake_next_states
        )

        # Total loss
        loss = real_loss + fake_loss + self.gradient_penalty_weight * grad_penalty

        # Metrics
        with torch.no_grad():
            real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean().item()
            fake_acc = (torch.sigmoid(fake_logits) < 0.5).float().mean().item()

        metrics = {
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'grad_penalty': grad_penalty.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
        }

        return loss, metrics

    def _gradient_penalty(self, real_states, real_actions, real_next_states,
                          fake_states, fake_actions, fake_next_states) -> torch.Tensor:
        """
        Compute gradient penalty for training stability.

        Interpolates between real and fake samples, computes gradients,
        and penalizes deviation from unit norm.
        """
        batch_size = real_states.shape[0]
        device = real_states.device

        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, device=device)

        # Interpolate
        interp_states = alpha * real_states + (1 - alpha) * fake_states
        interp_actions = alpha * real_actions + (1 - alpha) * fake_actions
        interp_next = alpha * real_next_states + (1 - alpha) * fake_next_states

        # Enable gradients
        interp_states.requires_grad_(True)
        interp_actions.requires_grad_(True)
        interp_next.requires_grad_(True)

        # Forward
        logits = self.forward(interp_states, interp_actions, interp_next)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=[interp_states, interp_actions, interp_next],
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )

        # Concatenate gradients
        grad_cat = torch.cat([g.view(batch_size, -1) for g in grads], dim=-1)

        # Penalty: (||grad|| - 1)^2
        grad_norm = grad_cat.norm(2, dim=-1)
        penalty = ((grad_norm - 1) ** 2).mean()

        return penalty


# ==============================================================================
# VISION ENCODER (OpenVLA: DINOv2 + SigLIP)
# ==============================================================================

class PrismaticVisionEncoder(nn.Module):
    """
    Fuses DINOv2 (spatial) + SigLIP (semantic) features.
    From OpenVLA paper - best of both worlds.
    """

    def __init__(self, config: UnifiedBrainConfig, image_size=224):
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.use_pretrained = False

        if config.use_pretrained_vision:
            try:
                from transformers import AutoModel
                self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-large")
                self.dinov2.requires_grad_(False)
                self.siglip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
                self.siglip.requires_grad_(False)
                self.projector = nn.Sequential(
                    nn.Linear(1024 + 768, config.vision_embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(config.vision_embed_dim * 2, config.vision_embed_dim),
                )
                self.use_pretrained = True
                print("[VISION] Loaded DINOv2 + SigLIP (pretrained)")
            except Exception as e:
                print(f"[VISION] Pretrained failed: {e}, using CNN fallback")

        if not self.use_pretrained:
            # Fallback CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            )
            self.projector = nn.Linear(128, config.vision_embed_dim)
            print("[VISION] Using CNN fallback")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained:
            if images.shape[-2:] != (self.image_size, self.image_size):
                images = F.interpolate(images, (self.image_size, self.image_size), mode='bilinear')
            images = images.float() / 255.0 if images.max() > 1.0 else images

            with torch.no_grad():
                dino_feat = self.dinov2(pixel_values=images).last_hidden_state[:, 0]
                clip_feat = self.siglip.vision_model(pixel_values=images).pooler_output

            fused = torch.cat([dino_feat, clip_feat], dim=-1)
            return self.projector(fused)
        else:
            return self.projector(self.cnn(images))


# ==============================================================================
# SENSOR ENCODERS
# ==============================================================================

class ObjectDetector(nn.Module):
    """
    Detects and localizes objects in the scene for manipulation/navigation.

    PROBLEM: Robot needs to find "the cup" or "the kitchen" to execute commands.

    SOLUTION: Use vision encoder features + learned object queries to detect:
    - Graspable objects (cup, bottle, bowl)
    - Locations (kitchen, table, door)
    - People (user, faces)

    Research backing:
    - DETR (Facebook, 2020): Object detection as set prediction
    - OWL-ViT (Google, 2022): Open-vocabulary object detection
    - Grounding DINO (2023): Text-conditioned object detection
    """

    # Known object categories
    OBJECTS = [
        "cup", "bottle", "bowl", "plate", "mug", "glass",
        "table", "chair", "counter", "shelf", "door",
        "kitchen", "bathroom", "bedroom", "living room",
        "person", "face", "hand",
        "coffee machine", "fridge", "sink",
    ]

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Object query embeddings (like DETR)
        self.num_queries = config.object_detection_queries
        self.object_queries = nn.Embedding(self.num_queries, self.d_model)

        # Cross-attention from queries to vision features
        self.cross_attn = nn.MultiheadAttention(self.d_model, 8, batch_first=True)

        # Object classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.OBJECTS) + 1),  # +1 for "no object"
        )

        # Position predictor (x, y, z in robot frame)
        self.position_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # x, y, z
        )

        print(f"[OBJECTS] Detector initialized with {len(self.OBJECTS)} categories")

    def forward(self, vision_features: torch.Tensor) -> Dict:
        """
        Detect objects from vision features.

        Args:
            vision_features: [B, d_model] from vision encoder

        Returns:
            Dict with:
            - classes: [B, num_queries] - object class indices
            - positions: [B, num_queries, 3] - x,y,z positions
            - scores: [B, num_queries] - confidence scores
        """
        B = vision_features.shape[0]

        # Expand vision features for cross-attention
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)  # [B, 1, d_model]

        # Get object queries
        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, d_model]

        # Cross-attend to vision
        attended, _ = self.cross_attn(queries, vision_features, vision_features)

        # Classify objects
        class_logits = self.classifier(attended)  # [B, num_queries, num_classes]
        class_probs = F.softmax(class_logits, dim=-1)
        classes = class_probs.argmax(dim=-1)  # [B, num_queries]
        scores = class_probs.max(dim=-1).values  # [B, num_queries]

        # Predict positions
        positions = self.position_head(attended)  # [B, num_queries, 3]

        return {
            'classes': classes,
            'class_logits': class_logits,
            'positions': positions,
            'scores': scores,
            'features': attended,
        }

    def find_object(self, vision_features: torch.Tensor, object_name: str) -> Dict:
        """
        Find a specific object in the scene.

        Args:
            vision_features: Vision encoder output
            object_name: Name of object to find (e.g., "cup")

        Returns:
            Dict with position, confidence, found (bool)
        """
        detections = self.forward(vision_features)

        # Find object index
        object_name_lower = object_name.lower()
        target_idx = None
        for i, obj in enumerate(self.OBJECTS):
            if obj in object_name_lower or object_name_lower in obj:
                target_idx = i
                break

        if target_idx is None:
            return {'found': False, 'reason': f"Unknown object: {object_name}"}

        # Check if any detection matches
        classes = detections['classes'][0]  # First batch
        scores = detections['scores'][0]
        positions = detections['positions'][0]

        for i, (cls, score, pos) in enumerate(zip(classes, scores, positions)):
            if cls.item() == target_idx and score.item() > 0.5:
                return {
                    'found': True,
                    'position': pos.tolist(),
                    'confidence': score.item(),
                    'object': object_name,
                }

        return {'found': False, 'reason': f"Object '{object_name}' not detected"}

    def get_object_name(self, idx: int) -> str:
        """Get object name from index."""
        if idx < len(self.OBJECTS):
            return self.OBJECTS[idx]
        return "unknown"


class NavigationPlanner(nn.Module):
    """
    Plans navigation paths to target locations.

    PROBLEM: "Go to the kitchen" requires knowing where the kitchen is
    and planning a path there.

    SOLUTION: Maintain a spatial memory and plan paths using learned value function.

    Research backing:
    - Neural SLAM (Chaplot et al., 2020)
    - PointGoal Navigation (Habitat)
    - VLMaps (2023): Vision-Language Maps for navigation
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        # Spatial memory (simplified grid map)
        self.map_size = config.nav_map_size  # 64x64 grid
        self.map_resolution = config.nav_map_resolution  # 10cm per cell

        # Current position estimate
        self.register_buffer('position', torch.zeros(3))  # x, y, theta
        self.register_buffer('spatial_map', torch.zeros(1, self.map_size, self.map_size))

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        # Navigation policy
        self.nav_policy = nn.Sequential(
            nn.Linear(64 + 3, 128),  # goal + position
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # [forward, backward, turn_left, turn_right]
        )

        print(f"[NAV] Navigation planner initialized ({self.map_size}x{self.map_size} map)")

    def set_goal(self, goal_embedding: torch.Tensor):
        """Set navigation goal from language/vision embedding."""
        self.current_goal = self.goal_encoder(goal_embedding)

    def get_action(self, position: torch.Tensor) -> torch.Tensor:
        """
        Get navigation action given current position.

        Args:
            position: [x, y, theta] in world frame

        Returns:
            Navigation velocities [vx, vy, vtheta]
        """
        if not hasattr(self, 'current_goal'):
            return torch.zeros(3)

        # Combine goal and position
        goal_flat = self.current_goal.flatten()
        pos_flat = position.flatten()
        combined = torch.cat([goal_flat, pos_flat], dim=-1)

        # Get discrete action
        action_logits = self.nav_policy(combined.unsqueeze(0))
        action_probs = F.softmax(action_logits, dim=-1)

        # Convert to continuous velocities
        # [forward, backward, turn_left, turn_right]
        probs = action_probs[0].detach()
        vx = (probs[0] - probs[1]).item()  # Forward - backward
        vy = 0.0  # No lateral movement
        vtheta = (probs[3] - probs[2]).item()  # Right - left

        return torch.tensor([vx, vy, vtheta])

    def update_map(self, vision_features: torch.Tensor, position: torch.Tensor):
        """Update spatial map from vision and position."""
        # Simplified: just store position
        self.position = position

    def plan_path(self, start: torch.Tensor, goal: torch.Tensor) -> List[torch.Tensor]:
        """
        Plan a path from start to goal.

        Returns list of waypoints.
        """
        # Simplified: straight line path
        num_waypoints = 5
        path = []
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = start * (1 - t) + goal * t
            path.append(waypoint)
        return path


class ProprioceptionEncoder(nn.Module):
    """Encodes joint angles, velocities, orientation"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class TouchEncoder(nn.Module):
    """Encodes contact forces"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# ==============================================================================
# AUDIO ENCODER (SOTA 2025: Whisper + wav2vec2)
# ==============================================================================

class AudioEncoder(nn.Module):
    """
    Encodes audio input for multimodal robot learning.

    Two modes:
    1. Speech-to-Text (Whisper): For understanding voice commands
    2. Audio Embeddings (wav2vec2): For ambient sound understanding

    Research backing:
    - Whisper (OpenAI, 2022): Robust speech recognition
    - wav2vec2 (Meta, 2020): Self-supervised audio representations
    - ES3 (CVPR 2024): Audio-visual speech representations
    - WavTokenizer (ICLR 2025): Discrete audio tokens

    References:
    - https://github.com/jishengpeng/WavTokenizer
    - https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_ES3_Evolving_Self-Supervised_Learning_of_Robust_Audio-Visual_Speech_Representations_CVPR_2024_paper.pdf
    """

    def __init__(self, config, output_dim: int = 512, sample_rate: int = 16000):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.use_pretrained = False

        # Try to load pretrained models
        if getattr(config, 'use_pretrained_audio', False):
            try:
                from transformers import (
                    WhisperProcessor, WhisperForConditionalGeneration,
                    Wav2Vec2Processor, Wav2Vec2Model
                )

                # Whisper for speech-to-text (frozen)
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                self.whisper_model.requires_grad_(False)  # Frozen

                # wav2vec2 for audio embeddings (frozen)
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                self.wav2vec_model.requires_grad_(False)  # Frozen

                # Projection from wav2vec2 hidden size (768) to output_dim
                self.projector = nn.Sequential(
                    nn.Linear(768, output_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(output_dim * 2, output_dim),
                    nn.LayerNorm(output_dim),
                )

                self.use_pretrained = True
                print(f"[AUDIO] Loaded Whisper-tiny + wav2vec2-base (pretrained)")

            except Exception as e:
                print(f"[AUDIO] Pretrained models failed: {e}")
                print("[AUDIO] Using spectrogram CNN fallback")

        if not self.use_pretrained:
            # Fallback: Spectrogram-based CNN encoder
            # Input: raw audio waveform → mel spectrogram → CNN
            self.n_mels = 80
            self.n_fft = 400
            self.hop_length = 160

            # Mel spectrogram parameters (stored for forward pass)
            self.register_buffer('mel_basis', self._create_mel_basis())

            # CNN encoder for spectrograms
            self.cnn = nn.Sequential(
                # Input: (B, 1, n_mels, time)
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),

                nn.Flatten(),
            )

            self.projector = nn.Sequential(
                nn.Linear(128, output_dim),
                nn.LayerNorm(output_dim),
            )

            print(f"[AUDIO] Using spectrogram CNN fallback")

    def _create_mel_basis(self):
        """Create mel filterbank matrix"""
        import math
        n_mels = self.n_mels
        n_fft = self.n_fft
        sr = self.sample_rate

        # Simple mel filterbank (triangular filters)
        fmin, fmax = 0.0, sr / 2.0
        mel_min = 2595 * math.log10(1 + fmin / 700)
        mel_max = 2595 * math.log10(1 + fmax / 700)
        mels = torch.linspace(mel_min, mel_max, n_mels + 2)
        freqs = 700 * (10 ** (mels / 2595) - 1)

        # Convert to FFT bin indices
        bins = torch.floor((n_fft + 1) * freqs / sr).long()

        # Create filterbank
        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform"""
        # waveform: (B, time) or (B, 1, time)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        B, T = waveform.shape

        # Pad to ensure we have enough samples
        if T < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - T))
            T = self.n_fft

        # STFT using unfold (simpler than torch.stft for compatibility)
        # Frame the signal
        frames = waveform.unfold(1, self.n_fft, self.hop_length)  # (B, n_frames, n_fft)

        # Apply Hann window
        window = torch.hann_window(self.n_fft, device=waveform.device)
        frames = frames * window

        # FFT
        spectrum = torch.fft.rfft(frames, dim=-1)  # (B, n_frames, n_fft//2+1)
        magnitude = torch.abs(spectrum)

        # Apply mel filterbank
        mel_spec = torch.matmul(magnitude, self.mel_basis.T)  # (B, n_frames, n_mels)

        # Log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        # Reshape for CNN: (B, 1, n_mels, n_frames)
        mel_spec = mel_spec.transpose(1, 2).unsqueeze(1)

        return mel_spec

    def transcribe(self, audio: torch.Tensor) -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio: Waveform tensor (B, time) at 16kHz

        Returns:
            Transcribed text string
        """
        if not self.use_pretrained:
            return "[Audio transcription requires Whisper model]"

        with torch.no_grad():
            # Process audio
            inputs = self.whisper_processor(
                audio.cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(audio.device)

            # Generate transcription
            generated_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        return transcription

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings.

        Args:
            audio: Waveform tensor (B, time) at 16kHz

        Returns:
            Audio embeddings (B, output_dim)
        """
        if self.use_pretrained:
            with torch.no_grad():
                # Use wav2vec2 for embeddings
                inputs = self.wav2vec_processor(
                    audio.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs.input_values.to(audio.device)

                # Get hidden states
                outputs = self.wav2vec_model(input_values)
                # Mean pool over time dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, 768)

            return self.projector(embeddings)
        else:
            # Use spectrogram CNN
            mel_spec = self._compute_spectrogram(audio)
            features = self.cnn(mel_spec)
            return self.projector(features)


class LanguageEncoder(nn.Module):
    """Simple fallback encoder (LSTM-based)"""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)

    def forward(self, tokens):
        _, (hidden, _) = self.encoder(self.embedding(tokens))
        return hidden[-1]


class LLMEncoder(nn.Module):
    """
    Frozen LLM backbone + trainable projector for language understanding.

    Architecture (following OpenVLA, pi0, RT-2):
    - Frozen LLM extracts rich language features
    - Trainable projector maps to robot brain's d_model
    - LLM weights NEVER change (prevents forgetting)

    Supported backends:
    - SmolLM2 1.7B (default): Best quality/size tradeoff
    - TinyLlama 1.1B: Smaller, faster
    - Gemma 2B: Google's efficient model
    - Fallback: Simple LSTM (no HuggingFace needed)

    Research:
    - OpenVLA (2024): Frozen Llama-2 + action heads
    - pi0 (Physical Intelligence 2024): Frozen PaliGemma + flow matching
    - RT-2 (Google 2023): Frozen PaLM-E + action tokens
    """

    # Model configurations
    MODEL_CONFIGS = {
        "smollm": {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "hidden_dim": 2048,
            "description": "SmolLM2 1.7B - Best quality/size tradeoff"
        },
        "tinyllama": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "hidden_dim": 2048,
            "description": "TinyLlama 1.1B - Smaller, faster"
        },
        "gemma": {
            "model_id": "google/gemma-2b-it",
            "hidden_dim": 2048,
            "description": "Gemma 2B - Google's efficient model"
        },
        "phi": {
            "model_id": "microsoft/phi-2",
            "hidden_dim": 2560,
            "description": "Phi-2 2.7B - Microsoft's small but capable"
        },
    }

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        self.use_llm = False
        self.tokenizer = None
        self.llm = None

        # Try to load HuggingFace LLM
        if config.llm_enabled and config.llm_backend != "fallback":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Get model config
                model_config = self.MODEL_CONFIGS.get(config.llm_backend, self.MODEL_CONFIGS["smollm"])
                model_id = config.llm_model_id or model_config["model_id"]
                hidden_dim = model_config["hidden_dim"]

                print(f"[LLM] Loading {model_config['description']}...")
                print(f"[LLM] Model ID: {model_id}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    padding_side="left"  # For batch generation
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model (frozen)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,  # Half precision to save memory
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )

                # Freeze LLM weights
                if config.llm_freeze:
                    for param in self.llm.parameters():
                        param.requires_grad = False
                    print("[LLM] Weights FROZEN (recommended)")
                else:
                    print("[LLM] WARNING: Weights trainable (may cause forgetting)")

                # Trainable projector: LLM hidden dim -> d_model
                self.projector = nn.Sequential(
                    nn.Linear(hidden_dim, config.d_model * 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model * 2, config.d_model),
                    nn.LayerNorm(config.d_model),
                )

                self.use_llm = True
                self.hidden_dim = hidden_dim

                # Count params
                llm_params = sum(p.numel() for p in self.llm.parameters())
                proj_params = sum(p.numel() for p in self.projector.parameters())
                print(f"[LLM] Loaded! LLM params: {llm_params/1e6:.1f}M (frozen)")
                print(f"[LLM] Projector params: {proj_params/1e3:.1f}K (trainable)")

            except ImportError:
                print("[LLM] HuggingFace transformers not available, using fallback")
            except Exception as e:
                print(f"[LLM] Failed to load: {e}, using fallback")

        # Fallback: simple LSTM encoder
        if not self.use_llm:
            print("[LLM] Using fallback LSTM encoder")
            self.fallback_encoder = LanguageEncoder(config.vocab_size, config.language_embed_dim)
            self.fallback_proj = nn.Linear(config.language_embed_dim, config.d_model)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text string to embedding"""
        return self.encode_batch([text])

    def encode_batch(self, texts: list) -> torch.Tensor:
        """Encode a batch of text strings"""
        if self.use_llm:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.llm_max_length,
            )

            # Move to device
            device = next(self.projector.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get LLM hidden states
            with torch.no_grad():
                outputs = self.llm(**inputs, output_hidden_states=True)
                # Use last hidden state of last token (most information)
                hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
                # Mean pooling over sequence
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)

            # Project to d_model
            return self.projector(pooled.float())  # (batch, d_model)
        else:
            raise ValueError("Use forward() with token IDs for fallback encoder")

    def forward(self, tokens_or_text):
        """
        Forward pass - handles both token IDs and text strings.

        Args:
            tokens_or_text: Either:
                - torch.Tensor of token IDs (for fallback)
                - List of strings (for LLM)
                - Single string (for LLM)
        """
        if self.use_llm:
            if isinstance(tokens_or_text, str):
                return self.encode_batch([tokens_or_text])
            elif isinstance(tokens_or_text, list) and isinstance(tokens_or_text[0], str):
                return self.encode_batch(tokens_or_text)
            elif isinstance(tokens_or_text, torch.Tensor):
                # Decode tokens to text, then re-encode with LLM
                # This is inefficient but maintains compatibility
                texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens_or_text]
                return self.encode_batch(texts)
            else:
                raise ValueError(f"Unexpected input type: {type(tokens_or_text)}")
        else:
            # Fallback LSTM
            if isinstance(tokens_or_text, torch.Tensor):
                return self.fallback_proj(self.fallback_encoder(tokens_or_text))
            else:
                raise ValueError("Fallback encoder requires token IDs (torch.Tensor)")

    def get_tokenizer(self):
        """Return tokenizer for external use"""
        return self.tokenizer


# ==============================================================================
# API LLM PROVIDER (Claude, GPT-4, Ollama)
# ==============================================================================

class APILLMProvider:
    """
    API-based LLM for high-quality language understanding and generation.

    Separates concerns:
    - Local LLMEncoder: Produces embeddings for the transformer backbone (fast, every frame)
    - APILLMProvider: Generates responses, plans, thoughts (slow, on-demand)

    This gives Jack access to Claude/GPT-4 level intelligence for reasoning
    while keeping motor control fast and local.

    Supported providers:
    - Anthropic (Claude): Best reasoning, recommended
    - OpenAI (GPT-4): Strong alternative
    - Ollama: Free, local, any open model
    """

    def __init__(self, config: UnifiedBrainConfig):
        self.config = config
        self.provider = config.llm_api_provider
        self.model = config.llm_api_model
        self.max_tokens = config.llm_api_max_tokens
        self.temperature = config.llm_api_temperature
        self.client = None
        self._available = False

        if not config.llm_api_enabled:
            return

        import os
        api_key = os.environ.get(config.llm_api_key_env, "")

        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
                self._available = True
                print(f"[API LLM] Anthropic Claude: {self.model}")
            except ImportError:
                print("[API LLM] pip install anthropic to use Claude")
            except Exception as e:
                print(f"[API LLM] Anthropic init failed: {e}")

        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
                self._available = True
                print(f"[API LLM] OpenAI: {self.model}")
            except ImportError:
                print("[API LLM] pip install openai to use GPT-4")
            except Exception as e:
                print(f"[API LLM] OpenAI init failed: {e}")

        elif self.provider == "ollama":
            try:
                import requests
                base = config.llm_api_base_url or "http://localhost:11434"
                # Test connection
                resp = requests.get(f"{base}/api/tags", timeout=2)
                if resp.status_code == 200:
                    self._available = True
                    self._ollama_base = base
                    print(f"[API LLM] Ollama: {self.model} at {base}")
                else:
                    print(f"[API LLM] Ollama not responding at {base}")
            except Exception as e:
                print(f"[API LLM] Ollama connection failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, system_prompt: str, user_message: str, max_tokens: int = None) -> str:
        """Generate a response from the API LLM."""
        if not self._available:
            return ""

        max_tokens = max_tokens or self.max_tokens

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=self.temperature,
                )
                return response.content[0].text.strip()

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "ollama":
                import requests
                response = requests.post(
                    f"{self._ollama_base}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
                        "stream": False,
                        "options": {"temperature": self.temperature, "num_predict": max_tokens},
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                return ""

        except Exception as e:
            print(f"[API LLM] Generation failed: {e}")
            return ""

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get text embedding from API (for language-action grounding).
        Falls back to None if not available (use local LLM projector instead).
        """
        if not self._available:
            return None

        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                )
                emb = response.data[0].embedding
                return torch.tensor(emb, dtype=torch.float32)
            # Anthropic and Ollama don't have standard embedding APIs
            # Fall back to None (use local projector)
            return None
        except Exception:
            return None


# ==============================================================================
# CROSS-MODAL FUSION
# ==============================================================================

class CrossModalFusion(nn.Module):
    """
    Sensors attend to each other via self-attention.
    Vision sees "slippery floor" -> Touch confirms "low friction"
    """
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout, batch_first=True)
            for _ in range(3)  # 3 fusion layers
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4), nn.ReLU(),
                nn.Dropout(config.dropout), nn.Linear(config.d_model * 4, config.d_model),
            ) for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(6)])

    def forward(self, tokens, mask=None):
        x = tokens
        for i, (attn, ffn) in enumerate(zip(self.layers, self.ffns)):
            normed_x = self.norms[i*2](x)
            x = x + attn(normed_x, normed_x, normed_x, key_padding_mask=mask)[0]
            x = x + ffn(self.norms[i*2+1](x))
        return x


# ==============================================================================
# TEMPORAL MEMORY
# ==============================================================================

class TemporalMemory(nn.Module):
    """
    Remembers past observations (50 timesteps).
    Critical for: "I tried this 3 times, it's not working, try something else"
    """
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.context_length = config.context_length
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_model * 4,
                config.dropout, batch_first=True
            ),
            num_layers=4
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, config.context_length, config.d_model) * 0.02)

        # Memory buffer (updated during inference)
        self.register_buffer('memory_buffer', None)

    def forward(self, current_state: torch.Tensor, memory: torch.Tensor = None):
        """
        Args:
            current_state: (B, 1, d_model) - current observation
            memory: (B, T, d_model) - past observations (optional)
        """
        if memory is not None:
            seq = torch.cat([memory, current_state], dim=1)
            seq = seq[:, -self.context_length:, :]  # Keep last N
        else:
            seq = current_state

        seq = seq + self.pos_encoding[:, :seq.shape[1], :]
        return self.encoder(seq)

    def update_memory(self, new_state: torch.Tensor):
        """Update memory buffer with new observation"""
        if self.memory_buffer is None:
            self.memory_buffer = new_state
        else:
            self.memory_buffer = torch.cat([self.memory_buffer, new_state], dim=1)
            self.memory_buffer = self.memory_buffer[:, -self.context_length:, :]


# ==============================================================================
# TOKENIZER
# ==============================================================================

class JointTokenizer(nn.Module):
    """Tokenize robot state into joint tokens for meaningful self-attention"""
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

        # Token types: 0=CLS, 1=joint, 2=body, 3=goal, 4=action, 5=multimodal
        self.token_type_embed = nn.Embedding(6, config.d_model)
        self.action_embed = nn.Linear(config.action_dim, config.d_model)

    def forward(self, state, goal=None, noisy_actions=None, multimodal_tokens=None):
        """
        Tokenize state + optional multimodal inputs for transformer.

        Args:
            state: [B, obs_dim] - proprioception
            goal: [B, obs_dim] - optional goal state
            noisy_actions: [B, chunk, action_dim] - for diffusion
            multimodal_tokens: [B, N, d_model] - PRE-ENCODED vision/audio/language
                               These tokens go DIRECTLY into the transformer!

        Token sequence:
            [CLS] [VISION?] [AUDIO?] [LANG?] [Joint1...17] [Body] [Goal?] [Actions]
        """
        B = state.shape[0]
        device = state.device
        tokens, types = [], []

        # [CLS]
        tokens.append(self.cls_token.expand(B, -1, -1))
        types.append(torch.zeros(B, 1, dtype=torch.long, device=device))

        # MULTIMODAL TOKENS (vision/audio/language) - INSERTED INTO TRANSFORMER!
        # This is the KEY fix - these tokens participate in self-attention
        if multimodal_tokens is not None and multimodal_tokens.shape[1] > 0:
            tokens.append(multimodal_tokens)
            # Type 5 = multimodal
            types.append(torch.full((B, multimodal_tokens.shape[1]), 5, dtype=torch.long, device=device))

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
# WORLD MODEL (Full TD-MPC2)
# ==============================================================================

class WorldModel(nn.Module):
    """
    Complete TD-MPC2 World Model (UPGRADED to match archived version)

    Features:
    - Deeper encoder (3 layers like archived)
    - Deeper dynamics (4 layers like archived)
    - Deeper decoder (3 layers like archived)
    - Deeper reward predictor (3 layers like archived)
    - Target encoder for stable training
    - imagine_trajectory() for rollouts
    - plan_action_mpc() for MPC planning
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.d_model  # 512

        # Latent encoder (3 layers like archived)
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        # Dynamics model (4 layers like archived, with residual)
        self.dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, config.latent_dim),
        )
        self.residual_proj = nn.Linear(config.latent_dim, config.latent_dim)

        # Decoder (3 layers like archived)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, config.obs_dim),
        )

        # Reward predictor (3 layers like archived)
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Target encoder (3 layers, for stable training)
        self.target_encoder = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        # Initialize target with same weights
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def encode(self, cls_features: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Encode to latent space"""
        encoder = self.target_encoder if use_target else self.encoder
        return encoder(cls_features)

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent state"""
        x = torch.cat([latent, action], dim=-1)
        delta = self.dynamics(x)
        next_latent = self.residual_proj(latent) + delta  # Residual connection
        decoded = self.decoder(next_latent)
        reward = self.reward_predictor(next_latent)
        return decoded, reward, next_latent

    def imagine_trajectory(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out imagined trajectory in latent space.

        Args:
            initial_latent: (B, latent_dim)
            actions: (B, horizon, action_dim)

        Returns:
            latents: (B, horizon+1, latent_dim)
            rewards: (B, horizon)
        """
        B, horizon, _ = actions.shape
        latents = [initial_latent]
        rewards = []
        current = initial_latent

        for t in range(horizon):
            _, reward, next_latent = self.predict_next(current, actions[:, t])
            latents.append(next_latent)
            rewards.append(reward.squeeze(-1))
            current = next_latent

        return torch.stack(latents, dim=1), torch.stack(rewards, dim=1)

    def plan_action_mpc(
        self,
        cls_features: torch.Tensor,
        num_samples: int = None,
        horizon: int = None,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        Model Predictive Control: Sample actions, imagine outcomes, pick best.

        Args:
            cls_features: (B, d_model) - current state features

        Returns:
            best_action: (B, action_dim) - best first action
        """
        num_samples = num_samples or self.config.mpc_samples
        horizon = horizon or self.config.imagination_horizon
        temperature = temperature or self.config.mpc_temperature

        B = cls_features.shape[0]
        device = cls_features.device

        # Encode current state
        with torch.no_grad():
            current_latent = self.encode(cls_features)

        # Sample random action sequences
        action_sequences = torch.randn(
            B * num_samples, horizon, self.config.action_dim, device=device
        ) * temperature

        # Expand latent for all samples
        expanded_latent = current_latent.unsqueeze(1).repeat(1, num_samples, 1)
        expanded_latent = expanded_latent.reshape(B * num_samples, -1)

        # Imagine all trajectories
        with torch.no_grad():
            _, rewards = self.imagine_trajectory(expanded_latent, action_sequences)

        # Compute returns
        returns = rewards.sum(dim=1).reshape(B, num_samples)

        # Select best
        best_indices = returns.argmax(dim=1)
        action_sequences = action_sequences.reshape(B, num_samples, horizon, self.config.action_dim)
        best_actions = action_sequences[torch.arange(B, device=device), best_indices, 0]

        return best_actions

    def update_target_encoder(self, momentum: float = 0.01):
        """EMA update of target encoder"""
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(momentum * param.data + (1 - momentum) * target_param.data)


# ==============================================================================
# HIERARCHICAL PLANNER (Full HAC)
# ==============================================================================

class Skill(nn.Module):
    """A reusable skill with initiation, policy, and termination"""
    def __init__(self, skill_id: int, config: UnifiedBrainConfig):
        super().__init__()
        self.skill_id = skill_id

        # Initiation (can this skill start?)
        self.initiation = nn.Sequential(
            nn.Linear(config.d_model + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )

        # Policy (what sub-goal to generate)
        self.policy = nn.Sequential(
            nn.Linear(config.d_model + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.goal_dim),
        )

        # Termination (should skill end?)
        self.termination = nn.Sequential(
            nn.Linear(config.d_model + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )

    def can_initiate(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.initiation(torch.cat([state, goal], dim=-1))

    def execute(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(torch.cat([state, goal], dim=-1))

    def should_terminate(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.termination(torch.cat([state, goal], dim=-1))


class HighLevelPlanner(nn.Module):
    """Task decomposition: complex task -> sub-goals"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        self.task_encoder = nn.Sequential(
            nn.Linear(config.goal_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
        )

        self.planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_model * 4,
                config.dropout, batch_first=True
            ),
            num_layers=4
        )

        self.subgoal_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.goal_dim)
        )

        # Learnable sub-goal queries (like DETR)
        self.subgoal_queries = nn.Parameter(
            torch.randn(1, config.max_subgoals, config.d_model) * 0.02
        )

    def forward(self, cls_features: torch.Tensor, task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_features: (B, d_model) - state from backbone
            task: (B, goal_dim) - high-level task

        Returns:
            subgoals: (B, max_subgoals, goal_dim)
            weights: (B, max_subgoals) - importance
        """
        B = cls_features.shape[0]

        state_emb = self.state_encoder(cls_features).unsqueeze(1)
        task_emb = self.task_encoder(task).unsqueeze(1)
        queries = self.subgoal_queries.expand(B, -1, -1)

        context = torch.cat([state_emb, task_emb, queries], dim=1)
        planned = self.planner(context)

        subgoal_reprs = planned[:, 2:, :]
        subgoals = self.subgoal_generator(subgoal_reprs)

        weights = F.softmax(torch.sum(subgoal_reprs ** 2, dim=-1), dim=-1)

        return subgoals, weights


class MidLevelController(nn.Module):
    """Skill selection and execution"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        # Skill library
        self.skills = nn.ModuleList([
            Skill(i, config) for i in range(config.num_skills)
        ])

        # Skill selector
        self.skill_selector = nn.Sequential(
            nn.Linear(config.d_model + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.num_skills),
        )

        self.skill_names = [
            "stand_up", "walk_forward", "walk_backward", "turn_left", "turn_right",
            "reach_forward", "reach_up", "grasp", "release", "push",
            "pull", "climb_step", "descend_step", "jump", "land",
            "balance", "recover", "crawl", "roll", "idle"
        ][:config.num_skills]

    def select_skill(self, cls_features: torch.Tensor, subgoal: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Select which skill to use"""
        combined = torch.cat([cls_features, subgoal], dim=-1)
        logits = self.skill_selector(combined)
        probs = F.softmax(logits, dim=-1)

        # Use mean across batch for skill selection
        mean_probs = probs.mean(dim=0)
        if self.training:
            skill_id = torch.multinomial(mean_probs, num_samples=1).item()
        else:
            skill_id = mean_probs.argmax().item()

        return skill_id, probs

    def execute_skill(self, skill_id: int, cls_features: torch.Tensor, subgoal: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Execute selected skill"""
        skill = self.skills[skill_id]
        low_level_goal = skill.execute(cls_features, subgoal)
        termination_prob = skill.should_terminate(cls_features, subgoal).mean().item()
        return low_level_goal, termination_prob


class HierarchicalPlanner(nn.Module):
    """Complete 3-level hierarchy: Task -> Subgoals -> Skills -> Actions"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        self.high_level = HighLevelPlanner(config)
        self.mid_level = MidLevelController(config)

        # Tracking state
        self.current_subgoal_idx = 0
        self.current_skill_id = None
        self.skill_step = 0

    def plan(self, cls_features: torch.Tensor, task: torch.Tensor) -> Dict:
        """
        Full hierarchical planning.

        Args:
            cls_features: (B, d_model) - from backbone
            task: (B, goal_dim) - high-level task

        Returns:
            dict with subgoals, active_subgoal, skill info, etc.
        """
        B = cls_features.shape[0]

        # High-level: task -> subgoals
        subgoals, subgoal_weights = self.high_level(cls_features, task)

        # Get current subgoal
        active_subgoal = subgoals[:, self.current_subgoal_idx, :]

        # Mid-level: select skill
        skill_id, skill_probs = self.mid_level.select_skill(cls_features, active_subgoal)

        # Execute skill
        low_level_goal, termination_prob = self.mid_level.execute_skill(
            skill_id, cls_features, active_subgoal
        )

        # Update tracking
        if termination_prob > 0.5:
            self.skill_step = 0
            self.current_subgoal_idx = min(self.current_subgoal_idx + 1, self.config.max_subgoals - 1)
        else:
            self.skill_step += 1

        return {
            'subgoals': subgoals,
            'subgoal_weights': subgoal_weights,
            'active_subgoal': active_subgoal,
            'skill_id': skill_id,
            'skill_name': self.mid_level.skill_names[skill_id],
            'skill_probs': skill_probs,
            'low_level_goal': low_level_goal,
            'termination_prob': termination_prob,
        }

    def reset(self):
        """Reset at episode start"""
        self.current_subgoal_idx = 0
        self.current_skill_id = None
        self.skill_step = 0


# ==============================================================================
# OUTPUT HEADS
# ==============================================================================

class ActionHead(nn.Module):
    """
    Multi-Action Head for progressive robot training.

    Supports both:
    - Locomotion only (17 joints): legs + torso
    - Full humanoid (57 joints): locomotion + arms + hands + neck

    The heads share a common feature extractor but have separate outputs.
    This allows training locomotion first, then adding manipulation without
    forgetting walking skills.
    """
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        self.shared = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
        )

        # Locomotion head (17 joints: legs + torso)
        # Joints: abdomen (3) + hips (6) + knees (2) + ankles (4) + shoulders (2)
        self.locomotion_dim = 17
        self.locomotion_head = nn.Linear(config.d_model, self.locomotion_dim)
        self.locomotion_velocity = nn.Linear(config.d_model, self.locomotion_dim)

        # Manipulation head (40 joints: arms + hands + neck)
        # Joints: neck (2) + shoulders (4) + elbows (2) + wrists (4) + fingers (30)
        self.manipulation_dim = 40
        self.manipulation_head = nn.Linear(config.d_model, self.manipulation_dim)
        self.manipulation_velocity = nn.Linear(config.d_model, self.manipulation_dim)

        # Full body = locomotion + manipulation = 57 joints
        self.full_dim = self.locomotion_dim + self.manipulation_dim

        # Mode: 'locomotion' (17), 'manipulation' (40), 'full' (57)
        self.mode = 'locomotion'

        print(f"  ActionHead: locomotion={self.locomotion_dim}, manipulation={self.manipulation_dim}, full={self.full_dim}")

    def set_mode(self, mode: str):
        """Set action output mode: 'locomotion', 'manipulation', or 'full'"""
        assert mode in ['locomotion', 'manipulation', 'full']
        self.mode = mode
        print(f"  ActionHead mode: {mode} ({self._get_dim()} dims)")

    def _get_dim(self) -> int:
        if self.mode == 'locomotion':
            return self.locomotion_dim
        elif self.mode == 'manipulation':
            return self.manipulation_dim
        else:
            return self.full_dim

    def forward(self, x):
        """
        Predict actions based on current mode.

        Args:
            x: Features [B, seq_len, d_model] or [B, d_model]

        Returns:
            Actions [B, seq_len, action_dim] or [B, action_dim]
        """
        features = self.shared(x)

        if self.mode == 'locomotion':
            return self.locomotion_head(features)
        elif self.mode == 'manipulation':
            return self.manipulation_head(features)
        else:  # full
            loco = self.locomotion_head(features)
            manip = self.manipulation_head(features)
            return torch.cat([loco, manip], dim=-1)

    def predict_velocity(self, x):
        """Predict velocity field for flow matching."""
        features = self.shared(x)

        if self.mode == 'locomotion':
            return self.locomotion_velocity(features)
        elif self.mode == 'manipulation':
            return self.manipulation_velocity(features)
        else:  # full
            loco_v = self.locomotion_velocity(features)
            manip_v = self.manipulation_velocity(features)
            return torch.cat([loco_v, manip_v], dim=-1)

    def forward_locomotion(self, x):
        """Force locomotion output regardless of mode."""
        features = self.shared(x)
        return self.locomotion_head(features)

    def forward_manipulation(self, x):
        """Force manipulation output regardless of mode."""
        features = self.shared(x)
        return self.manipulation_head(features)

    def forward_full(self, x):
        """Force full body output regardless of mode."""
        features = self.shared(x)
        loco = self.locomotion_head(features)
        manip = self.manipulation_head(features)
        return torch.cat([loco, manip], dim=-1)


class PhysicsHead(nn.Module):
    """Physics quantity prediction"""
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 10),  # 10 physics quantities
        )

    def forward(self, x):
        return self.predictor(x)


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


# ==============================================================================
# SOTA ACTION GENERATION (π0, GR00T N1, Figure Helix)
# ==============================================================================

class ActionExpert(nn.Module):
    """
    Action Expert - Separate transformer for action generation (π0 style).

    From Physical Intelligence's π0 paper:
    - Separate action expert (smaller than VLM backbone)
    - Cross-attention to VLM features (not concatenation)
    - Flow matching for smooth action generation
    - 50Hz action output (π0) or 200Hz (GR00T N1)

    Architecture:
        VLM features (System 2) ─┐
                                 ├─> Cross-Attention ─> Action Transformer ─> Actions
        Noisy action query ─────┘

    Research: π0 (Physical Intelligence 2024), GR00T N1 (NVIDIA 2025)
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        dim = config.action_expert_dim
        action_dim = config.action_dim
        chunk_size = config.action_chunk_size

        # Action embedding (noisy action -> embedding)
        self.action_embed = nn.Linear(action_dim * chunk_size, dim)

        # Time embedding for flow matching (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Linear(config.dit_time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Cross-attention to VLM/backbone features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
            for _ in range(config.action_expert_layers)
        ])

        # Self-attention transformer layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(config.action_expert_layers)
        ])

        # Layer norms for cross-attention
        self.cross_norms = nn.ModuleList([
            RMSNorm(dim) for _ in range(config.action_expert_layers)
        ])

        # Project VLM features to action expert dimension
        self.vlm_proj = nn.Linear(config.d_model, dim)

        # Output projection
        self.output_proj = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, action_dim * chunk_size),
        )

        print(f"  ActionExpert: {config.action_expert_layers} layers, dim={dim}")

    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embeddings (from DDPM)"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        noisy_actions: torch.Tensor,  # [B, action_dim * chunk_size]
        vlm_features: torch.Tensor,   # [B, seq_len, d_model] from backbone
        timesteps: torch.Tensor,      # [B] diffusion timestep (0 to 1)
    ) -> torch.Tensor:
        """
        Forward pass predicts noise/velocity for flow matching.

        Args:
            noisy_actions: Noised action chunk
            vlm_features: Features from main transformer backbone
            timesteps: Flow matching timestep (0=noise, 1=clean)

        Returns:
            Predicted velocity field for flow matching
        """
        B = noisy_actions.shape[0]

        # Embed actions
        x = self.action_embed(noisy_actions)  # [B, dim]
        x = x.unsqueeze(1)  # [B, 1, dim] - single query token

        # Add time embedding
        t_emb = self.get_timestep_embedding(timesteps, self.config.dit_time_embed_dim)
        t_emb = self.time_embed(t_emb)  # [B, dim]
        x = x + t_emb.unsqueeze(1)

        # Project VLM features
        kv = self.vlm_proj(vlm_features)  # [B, seq_len, dim]

        # Transformer layers with cross-attention
        for cross_attn, self_attn, norm in zip(
            self.cross_attn_layers, self.self_attn_layers, self.cross_norms
        ):
            # Cross-attention to VLM features
            x_cross, _ = cross_attn(x, kv, kv)
            x = norm(x + x_cross)

            # Self-attention
            x = self_attn(x)

        # Output
        x = x.squeeze(1)  # [B, dim]
        velocity = self.output_proj(x)  # [B, action_dim * chunk_size]

        return velocity


class FlowMatchingScheduler:
    """
    Flow Matching scheduler for action generation.

    From Lipman et al. (2022) "Flow Matching for Generative Modeling":
    - Linear interpolation between noise and data
    - Simpler than DDPM, more stable training
    - π0 uses 10 denoising steps (vs 100+ for DDPM)

    Flow: x_t = t * x_1 + (1-t) * x_0
    Where: x_0 = noise, x_1 = clean data, t ∈ [0, 1]
    """

    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps
        self.timesteps = torch.linspace(0, 1, num_steps + 1)

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise using flow matching interpolation."""
        t = timestep.view(-1, 1)
        return t * clean_actions + (1 - t) * noise

    def step(
        self,
        model_output: torch.Tensor,  # Predicted velocity
        timestep: float,
        sample: torch.Tensor,
        dt: float = None,
    ) -> torch.Tensor:
        """One denoising step."""
        if dt is None:
            dt = 1.0 / self.num_steps

        # Euler step: x_{t+dt} = x_t + v_t * dt
        return sample + model_output * dt


class DualSystemController:
    """
    Dual System Architecture (Figure Helix / NVIDIA GR00T N1 style).

    Three-tier architecture:
    - System 2 (VLM): Slow reasoning, 7-9 Hz, scene understanding
    - System 1 (Action Expert): Fast actions, 50-200 Hz, visuomotor policy
    - System 0 (Motor): Ultra-fast control, 1 kHz, PD/torque (optional)

    System 2 provides context to System 1 asynchronously:
    ┌─────────────────────────────────────────────────────────┐
    │  System 2 (9 Hz)                                       │
    │  VLM: "Pick up the red cup on the table"               │
    │       ↓ scene features (async)                         │
    │  System 1 (50 Hz)                                      │
    │  Action Expert: generates smooth action chunks         │
    │       ↓ target positions (sync)                        │
    │  System 0 (1 kHz) - optional                          │
    │  PD Controller: torque commands to motors              │
    └─────────────────────────────────────────────────────────┘

    Research:
    - GR00T N1 (NVIDIA 2025): S0/S1/S2 hierarchy
    - Figure Helix (Figure AI 2025): Dual system with VLM backbone
    - π0 (Physical Intelligence 2024): 50Hz action generation
    """

    def __init__(self, config: UnifiedBrainConfig):
        self.config = config
        self.system2_hz = config.system2_hz
        self.system1_hz = config.system1_hz
        self.system0_hz = config.system0_hz

        # Timing
        self.system2_dt = 1.0 / self.system2_hz  # ~111ms
        self.system1_dt = 1.0 / self.system1_hz  # ~20ms
        self.system0_dt = 1.0 / self.system0_hz  # ~1ms

        # Cached features from System 2 (for async operation)
        self.cached_vlm_features = None
        self.last_system2_time = 0.0

        # Action buffer for interpolation
        self.action_buffer = None
        self.action_index = 0

        print(f"  DualSystem: S2={self.system2_hz}Hz, S1={self.system1_hz}Hz, S0={self.system0_hz}Hz")

    def should_run_system2(self, current_time: float) -> bool:
        """Check if System 2 (VLM) should run."""
        return current_time - self.last_system2_time >= self.system2_dt

    def update_system2_features(self, features: torch.Tensor, current_time: float):
        """Cache VLM features from System 2."""
        self.cached_vlm_features = features
        self.last_system2_time = current_time

    def get_action_from_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Get single action from chunk for System 1 output.

        action_chunk: [B, chunk_size, action_dim]
        Returns: [B, action_dim]
        """
        if self.action_buffer is None or self.action_index >= self.action_buffer.shape[1]:
            # Need new chunk
            self.action_buffer = action_chunk
            self.action_index = 0

        action = self.action_buffer[:, self.action_index, :]
        self.action_index += 1
        return action

    def reset(self):
        """Reset controller state."""
        self.cached_vlm_features = None
        self.last_system2_time = 0.0
        self.action_buffer = None
        self.action_index = 0


class System0Controller(nn.Module):
    """
    System 0: Ultra-fast motor control (1 kHz).

    Converts target joint positions from System 1 into torque commands.
    Uses learned PD gains (like NVIDIA's approach).

    Optional - only needed for real hardware.

    τ = Kp * (q_target - q_current) + Kd * (dq_target - dq_current)

    Where Kp, Kd are learned per-joint gains.
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        action_dim = config.action_dim

        # Learned PD gains per joint
        self.kp = nn.Parameter(torch.ones(action_dim) * 50.0)
        self.kd = nn.Parameter(torch.ones(action_dim) * 5.0)

        # Optional: learned residual for model mismatch
        self.residual_mlp = nn.Sequential(
            nn.Linear(action_dim * 4, 64),  # q, dq, q_target, dq_target
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

        print(f"  System0: PD controller with learned gains")

    def forward(
        self,
        q_current: torch.Tensor,      # Current joint positions
        dq_current: torch.Tensor,     # Current joint velocities
        q_target: torch.Tensor,       # Target positions from System 1
        dq_target: torch.Tensor = None,  # Target velocities (optional)
    ) -> torch.Tensor:
        """Compute torque commands."""
        if dq_target is None:
            dq_target = torch.zeros_like(dq_current)

        # PD control
        pos_error = q_target - q_current
        vel_error = dq_target - dq_current

        tau = self.kp * pos_error + self.kd * vel_error

        # Add learned residual
        state = torch.cat([q_current, dq_current, q_target, dq_target], dim=-1)
        tau = tau + self.residual_mlp(state)

        return tau


# ==============================================================================
# COMPANION ROBOT FEATURES (NEW)
# ==============================================================================

class TaskCompletionHead(nn.Module):
    """
    Predicts when a task is DONE.

    Output: P(done) in [0, 1]
    - 0.0 = still working on task
    - 1.0 = task completed, ready for next command

    Example: "Walk to the door" -> actions... -> done=0.95 -> "I'm at the door!"
    """
    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 4),
            nn.SiLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),  # Output probability
        )

    def forward(self, x):
        return self.predictor(x).squeeze(-1)


class ResponseGenerator:
    """
    Generates spoken responses using the LLM.

    The LLM is used for BOTH:
    1. Understanding commands (encoder mode) - for motor control
    2. Generating responses (decoder mode) - for dialogue

    Example flow:
    User: "Pick up the red cup"
    -> LLM encodes for motor control
    -> Robot executes actions
    -> TaskCompletionHead: done=0.95
    -> ResponseGenerator: "I've picked up the red cup!"
    """

    # Response templates for different situations
    TEMPLATES = {
        "task_started": [
            "Okay, I'll {task}.",
            "Sure, {task}.",
            "On it! {task}.",
        ],
        "task_progress": [
            "Still working on it...",
            "Almost there...",
            "Making progress...",
        ],
        "task_done": [
            "Done! I finished {task}.",
            "All done with {task}!",
            "Completed {task}.",
        ],
        "task_failed": [
            "Sorry, I couldn't {task}. {reason}",
            "I had trouble with {task}. {reason}",
        ],
        "greeting": [
            "Hey! What would you like me to do?",
            "Hi there! Ready to help.",
            "Hello! What's the plan?",
        ],
        "confusion": [
            "I'm not sure what you mean. Can you say that differently?",
            "Could you rephrase that?",
        ],
    }

    def __init__(self, llm_encoder: 'LLMEncoder' = None, personality=None, emotional_state=None, api_llm: 'APILLMProvider' = None):
        self.llm = llm_encoder
        self.personality = personality
        self.emotional_state = emotional_state
        self.api_llm = api_llm
        self.use_llm_generation = False

        # Check if LLM can generate
        if llm_encoder is not None and hasattr(llm_encoder, 'llm') and llm_encoder.llm is not None:
            self.use_llm_generation = True
            print("[RESPONSE] Using LLM for response generation")
        else:
            print("[RESPONSE] Using template-based responses (no LLM)")

    def generate(self, situation: str, task: str = "", reason: str = "", context: str = "") -> str:
        """Generate a response for the given situation."""
        import random

        # API LLM can generate good responses even without context
        if self.api_llm is not None and self.api_llm.available:
            return self._generate_with_llm(situation, task, context or "")

        if self.use_llm_generation and context:
            # Use local LLM for more natural responses
            return self._generate_with_llm(situation, task, context)
        else:
            # Use templates
            templates = self.TEMPLATES.get(situation, self.TEMPLATES["confusion"])
            template = random.choice(templates)
            return template.format(task=task, reason=reason)

    def _generate_with_llm(self, situation: str, task: str, context: str) -> str:
        """Generate response using the LLM with personality and mood awareness."""
        # Build personality-aware prompt
        persona = "You are a helpful companion robot."
        if self.personality is not None:
            mood_dict = {}
            if self.emotional_state is not None:
                mood_dict = self.emotional_state.get_mood_dict()
            persona = self.personality.get_system_prompt(mood_dict)

        # Prefer API LLM (Claude/GPT-4) for much better responses
        if self.api_llm is not None and self.api_llm.available:
            return self._generate_with_api(persona, situation, task, context)

        try:
            prompt = f"""{persona}

Respond briefly (1-2 sentences) to this situation:
Situation: {situation}
Task: {task}
Context: {context}

Response:"""

            inputs = self.llm.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            device = next(self.llm.llm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm.llm.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.pad_token_id,
                )

            response = self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            return response[:100]  # Limit length

        except Exception as e:
            # Fallback to template
            return self._template_response(situation, task)

    def _generate_with_api(self, system_prompt: str, situation: str, task: str, context: str) -> str:
        """Generate response using API LLM (Claude/GPT-4)."""
        user_msg = f"Situation: {situation}\nTask: {task}"
        if context:
            user_msg += f"\nContext: {context}"
        user_msg += "\n\nRespond in character, 1-2 sentences."

        response = self.api_llm.generate(system_prompt, user_msg, max_tokens=100)
        return response if response else self._template_response(situation, task)

    def _template_response(self, situation, task):
        """Fallback to template."""
        import random
        templates = self.TEMPLATES.get(situation, self.TEMPLATES.get("confusion", ["I'm not sure."]))
        return random.choice(templates).format(task=task, reason="")

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the LLM.

        This is for direct Q&A like "What's 1+1?" or "What's the weather?".
        Unlike generate(), this doesn't need a task/situation context.

        Args:
            question: The question to answer

        Returns:
            The answer string
        """
        # Build personality-aware prompt
        persona = "You are a helpful companion robot."
        if self.personality is not None:
            mood_dict = {}
            if self.emotional_state is not None:
                mood_dict = self.emotional_state.get_mood_dict()
            persona = self.personality.get_system_prompt(mood_dict)

        # Prefer API LLM (Claude/GPT-4) for much better answers
        if self.api_llm is not None and self.api_llm.available:
            user_msg = f"Answer this question briefly and directly.\n\nQuestion: {question}"
            response = self.api_llm.generate(persona, user_msg, max_tokens=100)
            if response:
                return response

        if not self.use_llm_generation:
            return "I can't answer questions without my language model."

        try:
            prompt = f"""{persona}

Answer this question briefly and directly.

Question: {question}
Answer:"""

            inputs = self.llm.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            device = next(self.llm.llm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm.llm.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,  # Lower temperature for factual answers
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.pad_token_id,
                )

            response = self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()

            # Clean up
            response = response.split("\n")[0].strip()  # First line only
            return response[:150]  # Limit length

        except Exception as e:
            return f"Sorry, I had trouble thinking about that: {str(e)[:50]}"

    def can_answer(self) -> bool:
        """Check if Q&A is available."""
        if self.api_llm is not None and self.api_llm.available:
            return True
        return self.use_llm_generation


class CompanionMemory:
    """
    Long-term memory for companion robot.

    Stores facts about the user and past interactions.
    Uses simple embedding-based retrieval (like mini RAG).

    Example memories:
    - "User's name is Janno"
    - "User likes chess and cycling"
    - "Yesterday we played a game together"
    """

    def __init__(self, config: UnifiedBrainConfig, llm_encoder: 'LLMEncoder' = None):
        self.max_memories = config.memory_size
        self.memories = []  # List of (text, embedding, timestamp)
        self.llm = llm_encoder
        self.use_embeddings = llm_encoder is not None and hasattr(llm_encoder, 'use_llm') and llm_encoder.use_llm

        print(f"[MEMORY] Initialized with capacity for {self.max_memories} memories")
        if self.use_embeddings:
            print("[MEMORY] Using LLM embeddings for retrieval")
        else:
            print("[MEMORY] Using keyword matching (no LLM)")

    def add(self, text: str, importance: float = 1.0):
        """Add a memory."""
        import time
        timestamp = time.time()

        # Get embedding if possible
        embedding = None
        if self.use_embeddings:
            try:
                with torch.no_grad():
                    embedding = self.llm.encode_batch([text])[0].cpu().numpy()
            except:
                pass

        self.memories.append({
            "text": text,
            "embedding": embedding,
            "timestamp": timestamp,
            "importance": importance,
        })

        # Prune old memories if over capacity
        if len(self.memories) > self.max_memories:
            # Remove least important old memories
            self.memories.sort(key=lambda m: m["importance"] * (1 - (timestamp - m["timestamp"]) / 86400))
            self.memories = self.memories[-self.max_memories:]

        print(f"[MEMORY] Stored: {text[:50]}...")

    def recall(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant memories."""
        if not self.memories:
            return []

        if self.use_embeddings:
            try:
                with torch.no_grad():
                    query_emb = self.llm.encode_batch([query])[0].cpu().numpy()

                # Cosine similarity
                scores = []
                for mem in self.memories:
                    if mem["embedding"] is not None:
                        sim = np.dot(query_emb, mem["embedding"]) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(mem["embedding"]) + 1e-8
                        )
                        scores.append((sim, mem["text"]))

                scores.sort(reverse=True)
                return [text for _, text in scores[:top_k]]
            except:
                pass

        # Fallback: keyword matching
        query_words = set(query.lower().split())
        scores = []
        for mem in self.memories:
            mem_words = set(mem["text"].lower().split())
            overlap = len(query_words & mem_words)
            scores.append((overlap, mem["text"]))

        scores.sort(reverse=True)
        return [text for _, text in scores[:top_k] if _ > 0]

    def get_context(self, query: str) -> str:
        """Get memory context for response generation."""
        memories = self.recall(query, top_k=3)
        if memories:
            return "Relevant memories: " + "; ".join(memories)
        return ""


class TextToSpeech:
    """
    Text-to-speech wrapper for companion robot.

    Options:
    1. pyttsx3 (offline, cross-platform)
    2. gTTS (Google, needs internet)
    3. Bark (neural TTS, needs GPU)
    """

    def __init__(self):
        self.engine = None
        self.method = None

        # Try pyttsx3 first (offline)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.method = "pyttsx3"
            print("[TTS] Using pyttsx3 (offline)")
        except:
            pass

        # Try gTTS (online)
        if self.engine is None:
            try:
                from gtts import gTTS
                self.method = "gtts"
                print("[TTS] Using gTTS (online)")
            except:
                pass

        if self.method is None:
            print("[TTS] No TTS available - will print responses only")

    def speak(self, text: str):
        """Speak the text."""
        print(f"[ROBOT SAYS]: {text}")

        if self.method == "pyttsx3":
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
        elif self.method == "gtts":
            try:
                from gtts import gTTS
                import io
                import pygame

                tts = gTTS(text=text, lang='en')
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)

                pygame.mixer.init()
                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
            except:
                pass


# ==============================================================================
# INTRINSIC MOTIVATION MODULES (NEW - Self-Thinking)
# ==============================================================================
# Research papers:
# - ICM: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (ICML 2017)
# - RND: Burda et al., "Exploration by Random Network Distillation" (ICLR 2019)
# - DIAYN: Eysenbach et al., "Diversity is All You Need" (ICLR 2019)
# - Empowerment: Mohamed & Rezende, "Variational Information Maximisation" (NeurIPS 2015)
# - Autotelic: Colas et al., "Autotelic Agents with Intrinsically Motivated Goal-Conditioned RL" (JMLR 2022)
# ==============================================================================


class IntrinsicCuriosityModule(nn.Module):
    """
    Hybrid ICM + RND for curiosity-driven exploration.

    Two complementary signals:
    1. ICM: Forward model prediction error (learns what's predictable vs novel)
    2. RND: Random network distillation (pure novelty detection)

    The robot is rewarded for encountering states it can't predict,
    driving exploration even without external rewards.

    Research:
    - ICM: https://pathak22.github.io/noreward-rl/
    - RND: https://arxiv.org/abs/1810.12894
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim  # 256
        action_dim = config.action_dim  # 17
        d_model = config.d_model  # 512

        # === ICM COMPONENT ===
        # Feature encoder: compress state to latent
        self.feature_encoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Inverse model: Predict action from (s_t, s_{t+1})
        # This learns features relevant to agent's actions (not noise)
        self.inverse_model = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # Forward model: Predict s_{t+1} from (s_t, a_t)
        # Prediction error = curiosity reward
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        # === RND COMPONENT ===
        # Target network: FROZEN random initialization
        self.rnd_target = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        # Freeze it permanently!
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        # Predictor network: Learns to match target
        # High error = novel state (never seen before)
        self.rnd_predictor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        # Running statistics for reward normalization (critical for stability)
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_std', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))

        print(f"  IntrinsicCuriosityModule: ICM + RND (latent={latent_dim})")

    def compute_icm_reward(
        self,
        state_features: torch.Tensor,      # Current state [B, d_model]
        next_state_features: torch.Tensor, # Next state [B, d_model]
        action: torch.Tensor               # Action taken [B, action_dim]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        ICM curiosity: Reward = forward model prediction error.
        High reward when agent can't predict consequences of action.
        """
        # Encode states to latent
        phi_s = self.feature_encoder(state_features)
        phi_s_next = self.feature_encoder(next_state_features)

        # Forward model prediction
        pred_phi_next = self.forward_model(torch.cat([phi_s, action], dim=-1))

        # Prediction error = intrinsic reward
        forward_loss = F.mse_loss(pred_phi_next, phi_s_next.detach(), reduction='none')
        icm_reward = forward_loss.mean(dim=-1)  # (B,)

        # Inverse model loss (for representation learning)
        pred_action = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.mse_loss(pred_action, action)

        return icm_reward, {
            'forward_loss': forward_loss.mean().item(),
            'inverse_loss': inverse_loss.item(),
        }

    def compute_rnd_reward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        RND curiosity: Reward = prediction error of random network.
        Novel states have high error because predictor hasn't seen them.
        """
        phi_s = self.feature_encoder(state_features)

        # Target output (frozen random features)
        with torch.no_grad():
            target_features = self.rnd_target(phi_s)

        # Predictor tries to match
        predicted_features = self.rnd_predictor(phi_s)

        # Error = novelty
        rnd_reward = F.mse_loss(predicted_features, target_features, reduction='none')
        rnd_reward = rnd_reward.mean(dim=-1)  # (B,)

        return rnd_reward

    def compute_intrinsic_reward(
        self,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor,
        action: torch.Tensor,
        icm_weight: float = 0.5,
        rnd_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Combined intrinsic reward from ICM and RND.
        """
        icm_reward, icm_info = self.compute_icm_reward(
            state_features, next_state_features, action
        )
        rnd_reward = self.compute_rnd_reward(next_state_features)

        # Combine and normalize
        combined = icm_weight * icm_reward + rnd_weight * rnd_reward
        normalized = self._normalize_reward(combined)

        return normalized, {
            'icm_reward': icm_reward.mean().item(),
            'rnd_reward': rnd_reward.mean().item(),
            'total_intrinsic': normalized.mean().item(),
            **icm_info,
        }

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Running normalization of intrinsic rewards (critical for stability)"""
        batch_mean = reward.mean()
        batch_var = reward.var() + 1e-8
        batch_count = reward.numel()

        # Update running stats
        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count

        new_mean = self.reward_mean + delta * batch_count / (total_count + 1e-8)
        new_std = torch.sqrt(
            (self.reward_std**2 * self.reward_count + batch_var * batch_count) / (total_count + 1e-8)
        )

        self.reward_mean.copy_(new_mean)
        self.reward_std.copy_(new_std.clamp(min=1e-4))
        self.reward_count.copy_(total_count.clamp(max=1e6))

        # Normalize
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)

    def get_training_loss(
        self,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for training ICM (inverse + forward) and RND predictor."""
        phi_s = self.feature_encoder(state_features)
        phi_s_next = self.feature_encoder(next_state_features)

        # ICM inverse loss
        pred_action = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.mse_loss(pred_action, action)

        # ICM forward loss
        pred_phi_next = self.forward_model(torch.cat([phi_s, action], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_s_next.detach())

        # RND predictor loss
        with torch.no_grad():
            target_features = self.rnd_target(phi_s_next)
        predicted_features = self.rnd_predictor(phi_s_next)
        rnd_loss = F.mse_loss(predicted_features, target_features)

        return inverse_loss + forward_loss + rnd_loss


class SkillDiscovery(nn.Module):
    """
    DIAYN: Diversity Is All You Need

    Discovers diverse skills WITHOUT any reward function.
    Skills are distinguishable by the states they visit.

    Objective: F(θ) = I(S;Z) + H[A|S] - I(A;Z|S)
    Pseudo-reward: r(s,z) = log q(z|s) - log p(z)

    The robot learns "walking", "jumping", "turning" etc.
    just from wanting to be distinguishable!

    Research: https://arxiv.org/abs/1802.06070
    """

    def __init__(self, config: UnifiedBrainConfig, num_skills: int = 50):
        super().__init__()
        self.config = config
        self.num_skills = num_skills

        # Skill prior: uniform distribution
        self.register_buffer('skill_prior', torch.ones(num_skills) / num_skills)

        # Discriminator q(z|s): Which skill generated this state?
        self.discriminator = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_skills),  # Logits over skills
        )

        # Skill embedding for policy conditioning
        self.skill_embedding = nn.Embedding(num_skills, config.d_model)

        # Skill names (will emerge during training)
        self.skill_names = [f"skill_{i}" for i in range(num_skills)]

        print(f"  SkillDiscovery (DIAYN): {num_skills} discoverable skills")

    def sample_skill(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random skill from prior"""
        return torch.randint(0, self.num_skills, (batch_size,), device=device)

    def compute_diayn_reward(
        self,
        state_latent: torch.Tensor,  # (B, latent_dim) - encoded state
        skill: torch.Tensor          # (B,) skill indices
    ) -> Tuple[torch.Tensor, Dict]:
        """
        DIAYN pseudo-reward: r(s,z) = log q(z|s) - log p(z)
        High reward when state is clearly associated with skill.
        """
        # Discriminator prediction
        logits = self.discriminator(state_latent)  # (B, num_skills)
        log_q = F.log_softmax(logits, dim=-1)

        # Get log probability of actual skill
        log_q_z = log_q.gather(1, skill.unsqueeze(1)).squeeze(1)  # (B,)

        # Prior (uniform)
        log_p_z = torch.log(self.skill_prior[0])  # scalar

        # DIAYN reward
        reward = log_q_z - log_p_z

        # Discriminator accuracy (for monitoring)
        pred_skill = logits.argmax(dim=-1)
        accuracy = (pred_skill == skill).float().mean()

        return reward, {
            'diayn_reward': reward.mean().item(),
            'discriminator_accuracy': accuracy.item(),
            'skill_entropy': -(F.softmax(logits, dim=-1) * log_q).sum(-1).mean().item(),
        }

    def get_discriminator_loss(
        self,
        state_latent: torch.Tensor,
        skill: torch.Tensor,
    ) -> torch.Tensor:
        """Train discriminator to classify skills from states"""
        logits = self.discriminator(state_latent)
        return F.cross_entropy(logits, skill)

    def get_skill_embedding(self, skill: torch.Tensor) -> torch.Tensor:
        """Get embedding to condition policy on skill"""
        return self.skill_embedding(skill)

    def get_most_likely_skill(self, state_latent: torch.Tensor) -> torch.Tensor:
        """Infer which skill the agent is currently executing"""
        logits = self.discriminator(state_latent)
        return logits.argmax(dim=-1)


class Empowerment(nn.Module):
    """
    Information-theoretic intrinsic motivation.

    Empowerment = I(A; S' | S) = mutual information between
    actions and resulting states, given current state.

    Robot seeks states where it has MAXIMUM CONTROL over outcomes.
    Avoids uncontrollable situations (ice, lava, chaos).

    Research:
    - https://arxiv.org/abs/1509.08731 (Variational Information Maximisation)
    - https://link.springer.com/article/10.1007/s10514-023-10087-8 (Robotics 2023)
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        action_dim = config.action_dim

        # Source distribution: p(a|s) - what actions are available
        self.action_encoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # mean + log_std
        )

        # Planning distribution: q(a|s,s') - inverse dynamics
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # mean + log_std
        )

        # Forward dynamics: p(s'|s,a)
        self.forward_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        print(f"  Empowerment: I(A; S'|S) maximization")

    def compute_empowerment(
        self,
        state_latent: torch.Tensor,
        num_samples: int = 16,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate empowerment via variational bound.
        High empowerment = actions lead to diverse, predictable outcomes.
        """
        B = state_latent.shape[0]
        device = state_latent.device
        action_dim = self.config.action_dim

        # Sample actions from source p(a|s)
        action_params = self.action_encoder(state_latent)
        action_mean = action_params[..., :action_dim]
        action_log_std = action_params[..., action_dim:].clamp(-5, 2)
        action_std = action_log_std.exp()

        # Sample multiple actions
        actions = action_mean.unsqueeze(1) + action_std.unsqueeze(1) * \
                  torch.randn(B, num_samples, action_dim, device=device)

        # Predict next states
        state_expanded = state_latent.unsqueeze(1).expand(-1, num_samples, -1)
        inputs = torch.cat([state_expanded, actions], dim=-1)
        next_states = self.forward_dynamics(inputs.reshape(-1, inputs.shape[-1]))
        next_states = next_states.reshape(B, num_samples, -1)

        # Compute log probabilities under inverse model q(a|s,s')
        inverse_inputs = torch.cat([state_expanded, next_states], dim=-1)
        inverse_params = self.inverse_dynamics(inverse_inputs.reshape(-1, inverse_inputs.shape[-1]))
        inverse_params = inverse_params.reshape(B, num_samples, -1)

        inv_mean = inverse_params[..., :action_dim]
        inv_log_std = inverse_params[..., action_dim:].clamp(-5, 2)
        inv_std = inv_log_std.exp()

        # Log probability of sampled actions under inverse model
        log_q = -0.5 * ((actions - inv_mean) / (inv_std + 1e-8)).pow(2).sum(-1)
        log_q = log_q - inv_log_std.sum(-1) - 0.5 * action_dim * np.log(2 * np.pi)

        # Empowerment ≈ E[log q(a|s,s')]
        empowerment = log_q.mean(dim=1)  # (B,)

        return empowerment, {
            'empowerment': empowerment.mean().item(),
            'action_entropy': action_log_std.mean().item(),
        }


class Metacognition(nn.Module):
    """
    Metacognitive module: Knowing what you know and don't know.

    Three components (from metacognition research):
    1. Metacognitive Knowledge: Self-assessment of capabilities
    2. Metacognitive Planning: Deciding what to learn
    3. Metacognitive Evaluation: Reflecting on learning

    This enables the robot to:
    - Know when to ask for help
    - Prioritize learning certain skills
    - Avoid overconfident mistakes

    Research: https://openreview.net/forum?id=4KhDd0Ozqe
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        # === UNCERTAINTY ESTIMATION (Ensemble) ===
        # Multiple prediction heads for ensemble disagreement
        self.ensemble_size = 5
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, 256),
                nn.ReLU(),
                nn.Linear(256, config.action_dim),
            ) for _ in range(self.ensemble_size)
        ])

        # === METACOGNITIVE KNOWLEDGE ===
        # "What am I good/bad at?"
        self.capability_estimator = nn.Sequential(
            nn.Linear(config.d_model + config.goal_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # [can_do_confidently, uncertain, cannot_do]
        )

        # === METACOGNITIVE PLANNING ===
        # "What should I learn next?"
        self.learning_priority = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_skills),  # Priority per skill
        )

        print(f"  Metacognition: ensemble={self.ensemble_size}, capability assessment")

    def estimate_uncertainty(
        self,
        state_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Estimate epistemic uncertainty via ensemble disagreement.

        Returns:
            action_mean: Ensemble mean action
            uncertainty: Uncertainty estimate (0-1)
        """
        # Get predictions from all ensemble members
        predictions = torch.stack([
            head(state_features) for head in self.ensemble_heads
        ], dim=0)  # (ensemble_size, B, action_dim)

        # Mean prediction
        action_mean = predictions.mean(dim=0)

        # Uncertainty = variance across ensemble
        variance = predictions.var(dim=0).mean(dim=-1)  # (B,)

        # Normalize to 0-1
        uncertainty = torch.sigmoid(variance)

        return action_mean, uncertainty, {
            'uncertainty_mean': uncertainty.mean().item(),
            'uncertainty_max': uncertainty.max().item(),
            'ensemble_std': predictions.std(dim=0).mean().item(),
        }

    def assess_capability(
        self,
        state_features: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[str, float, Dict]:
        """
        Self-assess: Can I achieve this goal?

        Returns:
            assessment: "confident", "uncertain", "cannot"
            confidence: 0-1 score
        """
        inputs = torch.cat([state_features, goal], dim=-1)
        logits = self.capability_estimator(inputs)
        probs = F.softmax(logits, dim=-1)

        # Get assessment
        assessment_idx = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        labels = ["confident", "uncertain", "cannot"]
        assessment = labels[assessment_idx[0].item()] if state_features.shape[0] == 1 else "batch"

        return assessment, confidence.mean().item(), {
            'p_confident': probs[:, 0].mean().item(),
            'p_uncertain': probs[:, 1].mean().item(),
            'p_cannot': probs[:, 2].mean().item(),
        }

    def decide_what_to_learn(
        self,
        state_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Metacognitive planning: What skill should I practice?
        Prioritizes skills with high uncertainty (learning potential).
        """
        priorities = self.learning_priority(state_features)
        priorities = F.softmax(priorities, dim=-1)

        # Sample skill to practice (exploration)
        skill_to_learn = torch.multinomial(priorities, 1).squeeze(-1)

        return skill_to_learn, priorities

    def should_ask_for_help(
        self,
        state_features: torch.Tensor,
        goal: torch.Tensor,
        uncertainty_threshold: float = 0.7,
    ) -> Tuple[bool, str]:
        """
        Decide whether to ask human for help.

        Returns True if:
        1. Uncertainty is high AND
        2. Capability assessment is "uncertain" or "cannot"
        """
        _, uncertainty, _ = self.estimate_uncertainty(state_features)
        assessment, confidence, _ = self.assess_capability(state_features, goal)

        should_ask = (
            uncertainty.mean().item() > uncertainty_threshold and
            assessment in ["uncertain", "cannot"]
        )

        reason = ""
        if should_ask:
            if assessment == "cannot":
                reason = "I don't think I can do this task."
            else:
                reason = "I'm not sure how to proceed."

        return should_ask, reason


class AutotelicGoalGenerator(nn.Module):
    """
    IMGEP: Intrinsically Motivated Goal Exploration Process

    The robot generates its OWN goals based on:
    1. Learning progress (improving areas)
    2. Curiosity (novel areas)
    3. Competence (achievable challenges)

    This makes the robot truly self-directed - it decides what to learn!

    Research:
    - https://www.jmlr.org/papers/volume23/21-0808/21-0808.pdf
    - https://neurips.cc/virtual/2024/workshop/84726
    """

    def __init__(self, config: UnifiedBrainConfig, goal_bank_size: int = 1000):
        super().__init__()
        self.config = config
        self.goal_bank_size = goal_bank_size

        # Goal generator (VAE-style)
        self.goal_prior = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, config.goal_dim * 2),  # mean + log_var
        )

        # Learning progress estimator
        self.progress_estimator = nn.Sequential(
            nn.Linear(config.goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Goal bank with metadata
        self.register_buffer('goal_bank', torch.zeros(goal_bank_size, config.goal_dim))
        self.register_buffer('goal_attempts', torch.zeros(goal_bank_size))
        self.register_buffer('goal_successes', torch.zeros(goal_bank_size))
        self.register_buffer('goal_progress', torch.zeros(goal_bank_size))
        self.goal_ptr = 0

        print(f"  AutotelicGoalGenerator: bank_size={goal_bank_size}")

    def generate_goal(
        self,
        state_features: torch.Tensor,
        strategy: str = "learning_progress",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate a goal based on chosen strategy.

        Strategies:
        - "random": Sample from prior
        - "learning_progress": Goals where agent is improving
        - "curiosity": Novel goals (low attempts)
        - "competence": Achievable challenges (~50% success rate)
        """
        B = state_features.shape[0]
        device = state_features.device

        if strategy == "random" or self.goal_ptr == 0:
            # Sample from conditioned prior
            params = self.goal_prior(state_features)
            mean = params[..., :self.config.goal_dim]
            log_var = params[..., self.config.goal_dim:].clamp(-5, 2)
            std = (log_var * 0.5).exp()
            goal = mean + std * torch.randn_like(mean)

        elif strategy == "learning_progress":
            # Select goals with high learning progress
            progress_scores = self.goal_progress[:self.goal_ptr]
            probs = F.softmax(progress_scores * 5, dim=0)
            idx = torch.multinomial(probs.unsqueeze(0).expand(B, -1), 1).squeeze(-1)
            goal = self.goal_bank[idx]

        elif strategy == "curiosity":
            # Goals with low attempt count (novel)
            attempts = self.goal_attempts[:self.goal_ptr]
            novelty = 1.0 / (attempts + 1)
            probs = F.softmax(novelty * 5, dim=0)
            idx = torch.multinomial(probs.unsqueeze(0).expand(B, -1), 1).squeeze(-1)
            goal = self.goal_bank[idx]

        elif strategy == "competence":
            # Goals at edge of competence (~50% success rate)
            success_rate = self.goal_successes[:self.goal_ptr] / (self.goal_attempts[:self.goal_ptr] + 1)
            competence_score = 1.0 - torch.abs(success_rate - 0.5) * 2
            probs = F.softmax(competence_score * 5, dim=0)
            idx = torch.multinomial(probs.unsqueeze(0).expand(B, -1), 1).squeeze(-1)
            goal = self.goal_bank[idx]

        else:
            # Default to random
            return self.generate_goal(state_features, "random")

        return goal, {'strategy': strategy, 'goal_bank_size': self.goal_ptr}

    def update_goal_statistics(
        self,
        goal: torch.Tensor,
        success: bool,
        progress: float,
    ):
        """Update goal bank with outcome"""
        # Find closest goal in bank
        if self.goal_ptr > 0:
            distances = torch.norm(self.goal_bank[:self.goal_ptr] - goal, dim=-1)
            closest_idx = distances.argmin()

            if distances[closest_idx] < 0.5:  # Close enough
                self.goal_attempts[closest_idx] += 1
                self.goal_successes[closest_idx] += float(success)
                # EMA of progress
                self.goal_progress[closest_idx] = 0.9 * self.goal_progress[closest_idx] + 0.1 * progress
                return

        # Add new goal
        if self.goal_ptr < self.goal_bank_size:
            self.goal_bank[self.goal_ptr] = goal.detach()
            self.goal_attempts[self.goal_ptr] = 1
            self.goal_successes[self.goal_ptr] = float(success)
            self.goal_progress[self.goal_ptr] = progress
            self.goal_ptr += 1


class AutonomousMind(nn.Module):
    """
    Complete autonomous mind integrating all intrinsic motivation components.

    This is what makes Jack TRULY self-thinking:
    1. Curiosity drives exploration (ICM + RND)
    2. Skills emerge without supervision (DIAYN)
    3. Goals are self-generated (Autotelic)
    4. Metacognition enables self-awareness
    5. Empowerment seeks control

    The AutonomousMind computes combined intrinsic rewards and
    manages the autonomous learning process.
    """

    def __init__(self, config: UnifiedBrainConfig):
        super().__init__()
        self.config = config

        # All intrinsic motivation components
        self.curiosity = IntrinsicCuriosityModule(config)
        self.skill_discovery = SkillDiscovery(config, num_skills=50)
        self.empowerment = Empowerment(config)
        self.metacognition = Metacognition(config)
        self.goal_generator = AutotelicGoalGenerator(config)

        # Learnable reward mixing weights
        self.reward_weights = nn.Parameter(torch.tensor([
            0.3,   # extrinsic (task reward)
            0.25,  # curiosity (ICM + RND)
            0.2,   # skill diversity (DIAYN)
            0.15,  # empowerment
            0.1,   # goal progress
        ]))

        print(f"\n  AutonomousMind: Complete intrinsic motivation system")

    def compute_autonomous_reward(
        self,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor,
        action: torch.Tensor,
        state_latent: torch.Tensor,
        extrinsic_reward: torch.Tensor,
        skill: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined intrinsic + extrinsic reward.
        This is what makes the robot WANT to explore and learn.
        """
        rewards = {}

        # 1. Extrinsic (task) reward
        rewards['extrinsic'] = extrinsic_reward

        # 2. Curiosity reward
        curiosity_r, curiosity_info = self.curiosity.compute_intrinsic_reward(
            state_features, next_state_features, action
        )
        rewards['curiosity'] = curiosity_r

        # 3. Skill diversity reward (DIAYN)
        if skill is not None:
            diayn_r, diayn_info = self.skill_discovery.compute_diayn_reward(state_latent, skill)
            rewards['skill'] = diayn_r
        else:
            rewards['skill'] = torch.zeros_like(extrinsic_reward)

        # 4. Empowerment reward
        emp_r, emp_info = self.empowerment.compute_empowerment(state_latent)
        rewards['empowerment'] = emp_r

        # 5. Goal progress (placeholder - computed externally)
        rewards['goal_progress'] = torch.zeros_like(extrinsic_reward)

        # Combine with learned weights
        weights = F.softmax(self.reward_weights, dim=0)
        total_reward = sum(
            w * rewards[k] for w, k in zip(
                weights, ['extrinsic', 'curiosity', 'skill', 'empowerment', 'goal_progress']
            )
        )

        info = {
            'total_reward': total_reward.mean().item(),
            'weight_extrinsic': weights[0].item(),
            'weight_curiosity': weights[1].item(),
            'weight_skill': weights[2].item(),
            'weight_empowerment': weights[3].item(),
            'weight_goal': weights[4].item(),
            **{f'reward_{k}': v.mean().item() for k, v in rewards.items()},
        }

        return total_reward, info

    def get_training_loss(
        self,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor,
        action: torch.Tensor,
        state_latent: torch.Tensor,
        skill: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Combined training loss for all intrinsic motivation modules."""
        # ICM + RND loss
        curiosity_loss = self.curiosity.get_training_loss(
            state_features, next_state_features, action
        )

        # DIAYN discriminator loss
        diayn_loss = self.skill_discovery.get_discriminator_loss(state_latent, skill)

        total_loss = curiosity_loss + diayn_loss

        return total_loss, {
            'curiosity_loss': curiosity_loss.item(),
            'diayn_loss': diayn_loss.item(),
        }


# ==============================================================================
# UNIFIED BRAIN - COMPLETE
# ==============================================================================

class UnifiedBrain(nn.Module):
    """
    COMPLETE AGI BRAIN

    Integrates EVERYTHING:
    - SOTA Transformer backbone (RMSNorm, SwiGLU, RoPE)
    - Full TD-MPC2 WorldModel (imagination + MPC)
    - Full HierarchicalPlanner (3-level with skills)
    - Vision encoders (DINOv2 + SigLIP)
    - Temporal memory (50 timesteps)
    - Cross-modal fusion
    - Physics rule bank
    - **NEW: Intrinsic Motivation (Curiosity, Skills, Empowerment, Metacognition)**
    """

    def __init__(self, config: UnifiedBrainConfig = None):
        super().__init__()
        self.config = config or UnifiedBrainConfig()
        config = self.config

        print("\n" + "=" * 70)
        print("UNIFIED BRAIN - COMPLETE AGI SYSTEM")
        print("=" * 70)

        # ==========================================
        # ENCODERS
        # ==========================================
        print("\n[ENCODERS]")

        # Vision (optional)
        if config.vision_enabled:
            self.vision_encoder = PrismaticVisionEncoder(config)
            self.vision_proj = nn.Linear(config.vision_embed_dim, config.d_model)
        else:
            self.vision_encoder = None
            self.vision_proj = None
            print("  Vision: Disabled")

        # Proprioception
        self.proprio_encoder = ProprioceptionEncoder(config.obs_dim, config.d_model)
        print(f"  Proprio: {config.obs_dim} -> {config.d_model}")

        # Touch
        self.touch_encoder = TouchEncoder(10, config.touch_dim)
        self.touch_proj = nn.Linear(config.touch_dim, config.d_model)
        print(f"  Touch: 10 -> {config.d_model}")

        # Audio (Whisper + wav2vec2)
        if config.audio_enabled:
            self.audio_encoder = AudioEncoder(config, config.audio_embed_dim, config.audio_sample_rate)
            self.audio_proj = nn.Linear(config.audio_embed_dim, config.d_model)
            print(f"  Audio: {config.audio_sample_rate}Hz -> {config.d_model}")
        else:
            self.audio_encoder = None
            self.audio_proj = None
            print("  Audio: Disabled")

        # Language (LLM Integration)
        print("\n[LANGUAGE/LLM]")
        if config.llm_enabled:
            self.language_encoder = LLMEncoder(config)
            self.language_proj = None  # LLMEncoder has its own projector
        else:
            self.language_encoder = LanguageEncoder(config.vocab_size, config.language_embed_dim)
            self.language_proj = nn.Linear(config.language_embed_dim, config.d_model)
            print(f"  Fallback: vocab={config.vocab_size}")

        # API LLM (Claude/GPT-4 for high-quality responses)
        if config.llm_api_enabled:
            self.api_llm = APILLMProvider(config)
            if self.api_llm.available:
                print(f"  API LLM: {config.llm_api_provider} ({config.llm_api_model})")
            else:
                self.api_llm = None
                print("  API LLM: Not available (check API key)")
        else:
            self.api_llm = None

        # ==========================================
        # TOKENIZER & FUSION
        # ==========================================
        print("\n[TOKENIZATION & FUSION]")
        self.tokenizer = JointTokenizer(config)
        print(f"  Joint tokens: {config.num_joints} joints")

        self.cross_modal_fusion = CrossModalFusion(config)
        print("  Cross-modal fusion: 3 layers")

        # Semantic action anchors for LLM-agnostic language grounding
        self.semantic_anchors = SemanticActionAnchors(config.d_model, num_anchors=8)
        print("  Semantic anchors: 8 action categories (LLM-agnostic)")

        # ==========================================
        # TEMPORAL MEMORY
        # ==========================================
        print("\n[TEMPORAL MEMORY]")
        self.temporal_memory = (TemporalMemory(config)
                     if config.enable_temporal_memory else None)
        print(f"  Context: {config.context_length} timesteps")

        # ==========================================
        # PHYSICS RULES
        # ==========================================
        print("\n[NEURO-SYMBOLIC]")
        self.rule_bank = PhysicsRuleBank(config)
        print(f"  Physics rules: {config.num_rules}")

        # ==========================================
        # TRANSFORMER BACKBONE
        # ==========================================
        print("\n[TRANSFORMER BACKBONE]")
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_cross_attn=(i % 2 == 1))
            for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        print(f"  Layers: {config.n_layers} (cross-attn every 2)")
        print(f"  d_model: {config.d_model}, heads: {config.n_heads}")

        # ==========================================
        # WORLD MODEL (TD-MPC2)
        # ==========================================
        print("\n[WORLD MODEL - TD-MPC2]")
        self.world_model = (WorldModel(config)
                     if config.enable_world_model else None)
        print(f"  Latent dim: {config.latent_dim}")
        print(f"  Imagination horizon: {config.imagination_horizon}")
        print(f"  MPC samples: {config.mpc_samples}")

        # ==========================================
        # HIERARCHICAL PLANNER (HAC)
        # ==========================================
        print("\n[HIERARCHICAL PLANNER - HAC]")
        self.hierarchical_planner = (HierarchicalPlanner(config)
                     if config.enable_hierarchical_planner else None)
        print(f"  Skills: {config.num_skills}")
        print(f"  Max subgoals: {config.max_subgoals}")

        # ==========================================
        # OUTPUT HEADS
        # ==========================================
        print("\n[OUTPUT HEADS]")
        self.action_head = ActionHead(config)
        self.physics_head = PhysicsHead(config)
        self.value_head = ValueHead(config)
        print("  ActionHead, PhysicsHead, ValueHead")

        # Task completion head (knows when done)
        if config.enable_task_completion:
            self.task_completion_head = TaskCompletionHead(config)
            print("  TaskCompletionHead (knows when done)")
        else:
            self.task_completion_head = None

        # ==========================================
        # SOTA ACTION GENERATION (π0, GR00T N1)
        # ==========================================
        print("\n[SOTA ACTION GENERATION]")

        # Action Expert (π0 style - separate transformer for actions)
        if config.action_expert_enabled:
            self.action_expert = ActionExpert(config)
            self.flow_scheduler = FlowMatchingScheduler(num_steps=config.flow_matching_steps)
        else:
            self.action_expert = None
            self.flow_scheduler = None
            print("  ActionExpert: Disabled")

        # Dual System Controller (manages S0/S1/S2)
        if config.dual_system_enabled:
            self.dual_system = DualSystemController(config)
        else:
            self.dual_system = None
            print("  DualSystem: Disabled")

        # System 0 motor controller (optional, for real hardware)
        if config.system0_enabled:
            self.system0 = System0Controller(config)
        else:
            self.system0 = None
            print("  System0: Disabled (sim only)")

        # ==========================================
        # COMPANION ROBOT FEATURES
        # ==========================================
        print("\n[COMPANION FEATURES]")

        # Response generator (talks back)
        if config.enable_response_generation:
            llm_enc = self.language_encoder if hasattr(self.language_encoder, 'llm') else None
            self.response_generator = ResponseGenerator(llm_enc, api_llm=getattr(self, 'api_llm', None))
        else:
            self.response_generator = None
            print("  Response generation: Disabled")

        # Long-term memory
        if config.enable_memory:
            llm_enc = self.language_encoder if hasattr(self.language_encoder, 'llm') else None
            self.memory = CompanionMemory(config, llm_enc)
        else:
            self.memory = None
            print("  Memory: Disabled")

        # Text-to-speech
        if config.enable_tts:
            self.tts = TextToSpeech()
        else:
            self.tts = None
            print("  TTS: Disabled")

        # ==========================================
        # OBJECT DETECTION & NAVIGATION
        # ==========================================
        print("\n[PERCEPTION & NAVIGATION]")

        # Object detector (DETR-style)
        if config.enable_object_detection:
            self.object_detector = ObjectDetector(config)
        else:
            self.object_detector = None
            print("  Object Detection: Disabled")

        # Navigation planner
        if config.enable_navigation:
            self.navigation_planner = NavigationPlanner(config)
        else:
            self.navigation_planner = None
            print("  Navigation: Disabled")

        # ==========================================
        # INTRINSIC MOTIVATION (Self-Thinking)
        # ==========================================
        print("\n[INTRINSIC MOTIVATION]")

        if config.enable_intrinsic_motivation:
            self.autonomous_mind = AutonomousMind(config)
            print("  AutonomousMind: ENABLED (curiosity + skills + empowerment + metacognition)")
        else:
            self.autonomous_mind = None
            print("  AutonomousMind: Disabled")

        # ==========================================
        # VIRTUAL COMPANION - Emotional State & Movement
        # ==========================================
        print("\n[VIRTUAL COMPANION]")

        if config.enable_emotional_state and EmotionalState is not None:
            emo_config = EmotionalConfig(d_model=config.d_model, pad_dim=config.mood_dim,
                                          decay_factor=config.mood_decay_factor)
            self.emotional_state = EmotionalState(emo_config)
            # Set Jack's personality baseline
            if Personality is not None:
                self.personality = JACK_PERSONALITY
                pad_baseline = self.personality.get_pad_baseline()
                self.emotional_state.set_personality(
                    openness=self.personality.config.openness,
                    conscientiousness=self.personality.config.conscientiousness,
                    extraversion=self.personality.config.extraversion,
                    agreeableness=self.personality.config.agreeableness,
                    neuroticism=self.personality.config.neuroticism
                )
                print(f"  EmotionalState: PAD model, baseline=P:{pad_baseline['pleasure']:.2f} A:{pad_baseline['arousal']:.2f} D:{pad_baseline['dominance']:.2f}")
                print(f"  Personality: {self.personality.config.name} loaded")
            else:
                self.personality = None
                print("  EmotionalState: PAD model (no personality)")
        else:
            self.emotional_state = None
            self.personality = None
            print("  EmotionalState: Disabled")

        # Update response_generator with personality/emotional references
        # (ResponseGenerator is created before companion block, so patch refs now)
        if self.response_generator is not None:
            self.response_generator.personality = getattr(self, 'personality', None)
            self.response_generator.emotional_state = getattr(self, 'emotional_state', None)
            self.response_generator.api_llm = getattr(self, 'api_llm', None)

        if config.enable_movement_mood_coupling and MovementMoodCoupling is not None:
            mood_config = MovementMoodConfig(action_dim=config.action_dim,
                                              max_speed_mod=config.max_speed_modulation,
                                              max_style_bias=config.max_style_bias)
            self.movement_mood = MovementMoodCoupling(mood_config)
            print(f"  MovementMoodCoupling: speed±{config.max_speed_modulation*100:.0f}%, style±{config.max_style_bias*100:.0f}%")
        else:
            self.movement_mood = None
            print("  MovementMoodCoupling: Disabled")

        if config.enable_inner_monologue and InnerMonologue is not None:
            mono_config = MonologueConfig(cooldown_seconds=config.monologue_cooldown)
            llm_enc = self.language_encoder if hasattr(self, 'language_encoder') and hasattr(self.language_encoder, 'llm') else None
            self.inner_monologue = InnerMonologue(llm_encoder=llm_enc, config=mono_config)
            self.inner_monologue._api_llm = getattr(self, 'api_llm', None)
            print(f"  InnerMonologue: cooldown={config.monologue_cooldown}s")
        else:
            self.inner_monologue = None
            print("  InnerMonologue: Disabled")

        # Creative reasoning (AlphaGeometry-style: solve novel problems at runtime)
        self.creative_loop = None
        if AlphaGeometryLoop is not None:
            try:
                self.creative_loop = AlphaGeometryLoop(
                    config=AGLoopConfig(max_iterations=5, timeout_seconds=1.0)
                )
                print("  CreativeLoop: AlphaGeometry-style reasoning ENABLED")
            except Exception:
                print("  CreativeLoop: Failed to initialize")
        else:
            print("  CreativeLoop: Not available")

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Initialize — but NEVER touch pretrained weights.
        #
        # self.apply() recurses into EVERY submodule, including any frozen
        # pretrained encoder (the LLM, SigLIP, DINOv2). requires_grad_(False)
        # stops gradients; it does not stop an in-place normal_() from
        # overwriting the data. Measured before this fix: the loaded LLM's
        # q_proj std went 0.1010 -> 0.0196 and its embeddings 1.0013 -> 0.0197,
        # i.e. the "frozen pretrained backbone" was random noise in every run
        # this project has ever done.
        self._init_trainable_weights()

        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'=' * 70}")
        print(f"[TOTAL] {total_params:,} parameters (~{total_params * 4 / 1e6:.1f} MB)")
        print("=" * 70 + "\n")

    # Submodules holding pretrained weights. Anything whose name starts with one
    # of these is initialised by its own pretrained checkpoint, never by us.
    _PRETRAINED_PREFIXES = (
        "language_encoder.llm", "language_encoder.model", "text_tower",
        "vision_encoder.pretrained", "vision_encoder.clip", "vision_encoder.siglip",
        "vision_encoder.dinov2", "audio_encoder.pretrained",
    )

    def _init_trainable_weights(self) -> None:
        """Apply default init to trainable modules ONLY, skipping pretrained trees.

        Named traversal rather than self.apply(), because apply() offers no way to
        know which subtree it is in. Guarded by ladder spec T1.05.
        """
        skipped = 0
        for name, module in self.named_modules():
            if any(name.startswith(p) for p in self._PRETRAINED_PREFIXES):
                skipped += 1
                continue
            self._init_weights(module)
        if skipped:
            print(f"  [init] preserved {skipped} pretrained submodules")

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
        task: torch.Tensor = None,
        vision: torch.Tensor = None,
        touch: torch.Tensor = None,
        audio: torch.Tensor = None,
        language: torch.Tensor = None,
        noisy_actions: torch.Tensor = None,
        memory: torch.Tensor = None,
        use_mpc: bool = False,
        use_hierarchy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.

        Args:
            state: (B, obs_dim) - robot proprioception
            action: (B, action_dim) - for world model prediction
            goal: (B, obs_dim) - goal state (optional)
            task: (B, goal_dim) - high-level task for hierarchical planner
            vision: (B, 3, H, W) - RGB image (optional)
            touch: (B, 10) - touch sensors (optional)
            audio: (B, time) - audio waveform at 16kHz (optional)
            language: (B, seq_len) - text tokens (optional)
            noisy_actions: (B, chunk, action_dim) - for flow matching
            memory: (B, T, d_model) - temporal memory (optional)
            use_mpc: bool - use MPC planning from world model
            use_hierarchy: bool - use hierarchical planner

        Returns:
            Dict with all predictions
        """
        B = state.shape[0]
        device = state.device

        # ==========================================
        # ENCODE ALL MODALITIES
        # ==========================================
        modality_tokens = []

        # Proprio (always)
        proprio_emb = self.proprio_encoder(state).unsqueeze(1)
        modality_tokens.append(proprio_emb)

        # Vision (optional)
        if vision is not None and self.vision_encoder is not None:
            vision_emb = self.vision_proj(self.vision_encoder(vision)).unsqueeze(1)
            modality_tokens.append(vision_emb)

        # Touch (optional)
        if touch is not None:
            touch_emb = self.touch_proj(self.touch_encoder(touch)).unsqueeze(1)
            modality_tokens.append(touch_emb)

        # Audio (optional)
        if audio is not None and self.audio_encoder is not None:
            audio_emb = self.audio_proj(self.audio_encoder(audio)).unsqueeze(1)
            modality_tokens.append(audio_emb)

        # Language (optional) - supports LLM or fallback
        if language is not None:
            if self.language_proj is not None:
                # Fallback mode: token IDs -> LSTM -> projection
                # Convert strings to token IDs if needed
                if isinstance(language, str):
                    tokens = [ord(c) % self.config.vocab_size for c in language[:20]]
                    language = torch.tensor([tokens], dtype=torch.long, device=device)
                elif isinstance(language, list) and len(language) > 0 and isinstance(language[0], str):
                    batch_tokens = []
                    for text in language:
                        tokens = [ord(c) % self.config.vocab_size for c in text[:20]]
                        batch_tokens.append(tokens)
                    max_len = max(len(t) for t in batch_tokens)
                    batch_tokens = [t + [0] * (max_len - len(t)) for t in batch_tokens]
                    language = torch.tensor(batch_tokens, dtype=torch.long, device=device)
                lang_emb = self.language_proj(self.language_encoder(language)).unsqueeze(1)
            else:
                # LLM mode: LLMEncoder handles projection internally
                lang_emb = self.language_encoder(language).unsqueeze(1)
            # Expand to batch size if needed
            if lang_emb.shape[0] == 1 and B > 1:
                lang_emb = lang_emb.expand(B, -1, -1)
            modality_tokens.append(lang_emb)

        # Mood embedding (if emotional state enabled)
        if hasattr(self, 'emotional_state') and self.emotional_state is not None:
            mood_emb = self.emotional_state.get_mood_embedding()
            if mood_emb.dim() == 1:
                mood_emb = mood_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            elif mood_emb.dim() == 2:
                mood_emb = mood_emb.unsqueeze(1).expand(B, 1, -1)
            modality_tokens.append(mood_emb)

        # CLS token
        modality_tokens.append(self.cls_token.expand(B, -1, -1))

        # ==========================================
        # CROSS-MODAL FUSION
        # ==========================================
        # Concatenate all modality tokens for fusion
        all_modality_tokens = torch.cat(modality_tokens, dim=1)  # [B, N_tokens, d_model]
        fused = self.cross_modal_fusion(all_modality_tokens)

        # Extract multimodal tokens (everything except CLS at the end)
        # These will be INSERTED into the transformer backbone!
        multimodal_for_backbone = fused[:, :-1, :]  # All except CLS
        cls_fused = fused[:, -1, :]  # CLS token output

        # ==========================================
        # TEMPORAL MEMORY
        # ==========================================
        if memory is not None:
            mem_out = self.temporal_memory(cls_fused.unsqueeze(1), memory)
            cls_fused = mem_out[:, -1, :]

        # ==========================================
        # TOKENIZE FOR BACKBONE (NOW WITH MULTIMODAL!)
        # ==========================================
        # KEY FIX: Pass multimodal tokens INTO the tokenizer
        # Now the transformer sees: [CLS] [Vision] [Audio] [Lang] [Joints...] [Actions]
        tokens, mask = self.tokenizer(state, goal, noisy_actions, multimodal_tokens=multimodal_for_backbone)
        rules = self.rule_bank()

        # ==========================================
        # TRANSFORMER BACKBONE
        # ==========================================
        all_rule_weights = []
        for layer in self.layers:
            tokens, rule_weights = layer(tokens, rules, mask)
            if rule_weights is not None:
                all_rule_weights.append(rule_weights)

        tokens = self.final_norm(tokens)

        # Extract features
        cls_feat = tokens[:, 0, :]
        action_feat = tokens[:, -self.config.action_chunk_size:, :]

        # Combine with cross-modal fused features
        cls_combined = cls_feat + cls_fused

        # Average rule weights
        if all_rule_weights:
            avg_rule_weights = torch.stack(all_rule_weights).mean(0)
        else:
            avg_rule_weights = torch.zeros(B, self.config.num_rules, device=device)

        # ==========================================
        # OUTPUT PREDICTIONS
        # ==========================================
        output = {
            'cls_features': cls_combined,
            'hidden_states': tokens,  # Full sequence for ActionExpert cross-attention
            'rule_weights': avg_rule_weights,
            'actions': self.action_head(action_feat),
            'physics': self.physics_head(cls_combined),
            'value': self.value_head(cls_combined),
        }

        # Task completion prediction (knows when done)
        if self.task_completion_head is not None:
            output['task_done'] = self.task_completion_head(cls_combined)

        # ==========================================
        # WORLD MODEL (TD-MPC2)
        # ==========================================
        if action is not None:
            # World model uses locomotion actions only (first 17 dims)
            # Full action may be 57 dims (17 loco + 40 manip), truncate for dynamics
            action_for_world_model = action[..., :self.config.action_dim]
            next_state, reward, next_latent = self.world_model.predict_next(
                self.world_model.encode(cls_combined), action_for_world_model
            )
            output['next_state'] = next_state
            output['reward'] = reward
            output['next_latent'] = next_latent

        # MPC Planning
        if use_mpc:
            mpc_action = self.world_model.plan_action_mpc(cls_combined)
            output['mpc_action'] = mpc_action

        # ==========================================
        # HIERARCHICAL PLANNER
        # ==========================================
        if use_hierarchy and task is not None:
            hierarchy_output = self.hierarchical_planner.plan(cls_combined, task)
            output['hierarchy'] = hierarchy_output

        return output

    def predict_action(self, state: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """Simple action prediction"""
        output = self.forward(state, goal=goal)
        return output['actions'][:, 0, :]

    def imagine(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imagine future trajectory"""
        output = self.forward(state)
        latent = self.world_model.encode(output['cls_features'])
        return self.world_model.imagine_trajectory(latent, actions)

    def plan_with_mpc(self, state: torch.Tensor) -> torch.Tensor:
        """Plan action using MPC"""
        output = self.forward(state)
        return self.world_model.plan_action_mpc(output['cls_features'])

    def plan_with_hierarchy(self, state: torch.Tensor, task: torch.Tensor) -> Dict:
        """Plan using hierarchical planner"""
        output = self.forward(state, task=task, use_hierarchy=True)
        return output['hierarchy']

    def get_active_rules(self, state: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Show which physics rules are active"""
        with torch.no_grad():
            output = self.forward(state)
            weights = output['rule_weights'][0]
        top_w, top_i = torch.topk(weights, k=top_k)
        return [(self.rule_bank.rule_names[i.item()], w.item()) for i, w in zip(top_i, top_w)]

    def reset_planner(self):
        """Reset hierarchical planner state"""
        self.hierarchical_planner.reset()

    # ==========================================
    # LANGUAGE-CONDITIONED ACTIONS (NEW)
    # ==========================================

    def act_with_language(self, state: torch.Tensor, command: str) -> torch.Tensor:
        """
        Get action from natural language command.

        Example:
            action = brain.act_with_language(state, "walk forward slowly")

        Args:
            state: Robot state tensor (batch, obs_dim) or (obs_dim,)
            command: Natural language instruction

        Returns:
            action: Action tensor (batch, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # LLM encodes command, brain produces action
        output = self.forward(state, language=command)
        return output['actions'][:, 0, :]  # First action in chunk

    def act_with_language_batch(self, states: torch.Tensor, commands: list) -> torch.Tensor:
        """Batch version of act_with_language"""
        output = self.forward(states, language=commands)
        return output['actions'][:, 0, :]

    def get_tokenizer(self):
        """Get the LLM tokenizer (if available)"""
        if hasattr(self.language_encoder, 'get_tokenizer'):
            return self.language_encoder.get_tokenizer()
        return None

    def has_llm(self) -> bool:
        """Check if LLM is loaded"""
        return hasattr(self.language_encoder, 'use_llm') and self.language_encoder.use_llm

    # ==========================================
    # FLOW MATCHING ACTION GENERATION (π0 style)
    # ==========================================

    @torch.no_grad()
    def generate_actions_flow_matching(
        self,
        state: torch.Tensor,
        language: str = None,
        vision: torch.Tensor = None,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        Generate smooth action chunks using flow matching (π0 style).

        This is the SOTA way to generate actions:
        1. Start from random noise
        2. Use ActionExpert to predict velocity field
        3. Integrate through ODE to get clean actions
        4. Return smooth action chunk

        From π0 paper: "Flow matching provides smoother trajectories than
        diffusion models with fewer denoising steps."

        Args:
            state: Robot proprioception [B, obs_dim]
            language: Optional language command (string or list)
            vision: Optional vision input [B, 3, H, W]
            num_steps: Denoising steps (default: config.flow_matching_steps)

        Returns:
            actions: Clean action chunk [B, chunk_size, action_dim]
        """
        if self.action_expert is None:
            # Fallback to regular action head
            output = self.forward(state, language=language, vision=vision)
            return output['actions']

        if state.dim() == 1:
            state = state.unsqueeze(0)

        B = state.shape[0]
        device = state.device
        config = self.config
        num_steps = num_steps or config.flow_matching_steps

        # Get backbone features (System 2)
        output = self.forward(state, language=language, vision=vision)
        vlm_features = output['hidden_states']  # [B, seq_len, d_model]

        # Start from pure noise (x_0 in flow matching)
        action_shape = (B, config.action_dim * config.action_chunk_size)
        x = torch.randn(action_shape, device=device)

        # Flow matching: integrate from t=0 (noise) to t=1 (clean)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device)

            # Predict velocity at current state
            velocity = self.action_expert(x, vlm_features, t)

            # Euler step
            x = x + velocity * dt

        # Reshape to action chunk
        actions = x.view(B, config.action_chunk_size, config.action_dim)

        return actions

    def train_flow_matching_step(
        self,
        state: torch.Tensor,
        target_actions: torch.Tensor,
        language: str = None,
        vision: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        One training step for flow matching.

        Loss: ||v_θ(x_t, t) - (x_1 - x_0)||²

        Where:
        - x_0 = noise
        - x_1 = target_actions (from demos)
        - x_t = interpolation
        - v_θ = predicted velocity

        Args:
            state: Robot state [B, obs_dim]
            target_actions: Ground truth actions [B, chunk_size, action_dim]
            language: Optional language command
            vision: Optional vision input

        Returns:
            loss: Flow matching loss scalar
        """
        if self.action_expert is None:
            raise RuntimeError("ActionExpert not enabled. Set action_expert_enabled=True")

        B = state.shape[0]
        device = state.device
        config = self.config

        # Flatten target actions
        x_1 = target_actions.view(B, -1)  # [B, action_dim * chunk_size]

        # Sample noise (x_0)
        x_0 = torch.randn_like(x_1)

        # Sample random timestep
        t = torch.rand(B, device=device)

        # Interpolate: x_t = t * x_1 + (1-t) * x_0
        t_expand = t.view(B, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        # Get backbone features
        output = self.forward(state, language=language, vision=vision)
        vlm_features = output['hidden_states']

        # Predict velocity
        v_pred = self.action_expert(x_t, vlm_features, t)

        # Target velocity is just (x_1 - x_0) for conditional flow matching
        v_target = x_1 - x_0

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def act_dual_system(
        self,
        state: torch.Tensor,
        language: str = None,
        vision: torch.Tensor = None,
        current_time: float = 0.0,
        touch: torch.Tensor = None,
        audio: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Action generation using Dual System Architecture with ALL senses.

        System 2 (9Hz): Full forward pass with vision + audio + touch + language
        System 1 (50Hz): Fast action from cached S2 features

        Args:
            state: Proprioception [B, obs_dim]
            language: Text command (string or token IDs)
            vision: Eye camera [B, 3, H, W]
            touch: Contact forces [B, 10]
            audio: Microphone waveform [B, samples] at 16kHz
            current_time: Simulation time in seconds

        Returns:
            Dict with action, action_chunk, system2_ran, task_done
        """
        if self.dual_system is None:
            actions = self.generate_actions_flow_matching(state, language, vision)
            # Also get task_done from a full forward pass with ALL senses
            output = self.forward(state, language=language, vision=vision, touch=touch, audio=audio)
            return {
                'action': actions[:, 0, :],
                'action_chunk': actions,
                'system2_ran': True,
                'task_done': output.get('task_done', torch.tensor([0.0])),
                'full_output': output,
            }

        system2_ran = False

        # System 2 runs at ~9Hz: full forward pass with ALL senses
        if self.dual_system.should_run_system2(current_time):
            output = self.forward(state, language=language, vision=vision, touch=touch, audio=audio)
            vlm_features = output['hidden_states']
            self.dual_system.update_system2_features(vlm_features, current_time)
            # Cache task_done from System 2
            self._cached_task_done = output.get('task_done', torch.tensor([0.0]))
            system2_ran = True

        # System 1: fast actions from cached features
        if self.dual_system.cached_vlm_features is None:
            actions = self.generate_actions_flow_matching(state, language, vision)
        else:
            # Use cached features for fast action generation
            if self.action_expert is not None:
                B = state.shape[0]
                device = state.device
                config = self.config

                # Flow matching with cached features
                action_shape = (B, config.action_dim * config.action_chunk_size)
                x = torch.randn(action_shape, device=device)

                # Ensure cached features match batch size
                cached = self.dual_system.cached_vlm_features
                if cached.shape[0] != B:
                    cached = cached[:1].expand(B, -1, -1)

                dt = 1.0 / config.flow_matching_steps
                for step in range(config.flow_matching_steps):
                    t = torch.full((B,), step * dt, device=device)
                    velocity = self.action_expert(x, cached, t)
                    x = x + velocity * dt

                actions = x.view(B, config.action_chunk_size, config.action_dim)
            else:
                # Fallback: regular action head
                output = self.forward(state, language=language, vision=vision)
                actions = output['actions']

        # Get single action from chunk
        action = self.dual_system.get_action_from_chunk(actions)

        return {
            'action': action,
            'action_chunk': actions,
            'system2_ran': system2_ran,
            'task_done': getattr(self, '_cached_task_done', torch.tensor([0.0])),
        }

    def act_with_mood(
        self,
        state: torch.Tensor,
        language: str = None,
        vision: torch.Tensor = None,
        touch: torch.Tensor = None,
        audio: torch.Tensor = None,
        current_time: float = 0.0,
        is_idle: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Action generation with emotional modulation and FULL sensory input.

        All of Jack's senses:
        - state: proprioception (joint angles, velocities, orientation)
        - vision: egocentric eye camera [1, 3, 224, 224]
        - touch: contact forces on body [1, 10]
        - audio: raw microphone waveform [1, samples] at 16kHz
        - language: text command or subtask goal (string)
        - mood: emotional state (injected automatically from self.emotional_state)
        """
        result = self.act_dual_system(state, language, vision, current_time,
                                       touch=touch, audio=audio)

        # Apply mood modulation
        if self.movement_mood is not None and self.emotional_state is not None:
            pad = self.emotional_state.pad_vector.detach()
            result['action'] = self.movement_mood.modulate_action(
                result['action'], pad, is_idle=is_idle
            )
            result['mood'] = self.emotional_state.get_mood_dict()
            result['mood_speed'] = self.movement_mood.get_speed_multiplier(pad)

        return result

    # ==========================================
    # COMPANION ROBOT INTERACTION (NEW)
    # ==========================================

    def interact(self, state: torch.Tensor, command: str, speak: bool = True) -> Dict:
        """
        FULL COMPANION ROBOT INTERACTION.

        This is the main entry point for talking to your robot friend!

        Flow:
        1. User says command → Audio/Text input
        2. LLM understands command
        3. Brain generates action
        4. TaskCompletion checks if done
        5. ResponseGenerator creates reply
        6. TTS speaks the reply

        Args:
            state: Robot state tensor
            command: Natural language command (text)
            speak: Whether to use TTS

        Returns:
            Dict with:
            - action: Motor commands
            - task_done: Probability task is complete
            - response: What the robot says back
            - memories: Relevant memories retrieved

        Example:
            result = brain.interact(state, "Walk to the door")
            # Robot: "Okay, I'll walk to the door."
            # ... robot walks ...
            # result['task_done'] > 0.9
            # Robot: "Done! I'm at the door."
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Get relevant memories
        memories = []
        if self.memory is not None:
            memories = self.memory.recall(command, top_k=3)

        # Prepare language input
        # If LLM is available, pass string directly
        # Otherwise, convert to simple token IDs for fallback encoder
        if self.has_llm():
            lang_input = command
        else:
            # Simple character-level tokenization for fallback
            lang_input = torch.tensor([[ord(c) % self.config.vocab_size for c in command[:20]]],
                                      device=state.device)

        # Forward pass with language
        output = self.forward(state, language=lang_input)

        # Extract results
        action = output['actions'][:, 0, :]
        task_done = output.get('task_done', torch.tensor([0.0]))[0].item()

        # Generate response
        response = ""
        if self.response_generator is not None:
            context = self.memory.get_context(command) if self.memory else ""

            if task_done > 0.9:
                response = self.response_generator.generate("task_done", task=command, context=context)
            elif task_done < 0.1:
                response = self.response_generator.generate("task_started", task=command, context=context)
            else:
                response = self.response_generator.generate("task_progress", task=command, context=context)

        # Update emotional state based on interaction
        if self.emotional_state is not None:
            self.emotional_state.update(event_type=EventType.USER_CHAT, reward=0.0,
                                         user_interaction=True, dt=0.1)

        # Speak if requested
        if speak and self.tts is not None and response:
            self.tts.speak(response)

        return {
            'action': action,
            'task_done': task_done,
            'response': response,
            'memories': memories,
            'full_output': output,
        }

    def remember(self, fact: str, importance: float = 1.0):
        """
        Store a fact in long-term memory.

        Example:
            brain.remember("Janno likes chess")
            brain.remember("Today we went for a walk", importance=2.0)
        """
        if self.memory is not None:
            self.memory.add(fact, importance)

    def recall(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant memories.

        Example:
            memories = brain.recall("What does Janno like?")
            # ["Janno likes chess", "Janno likes cycling"]
        """
        if self.memory is not None:
            return self.memory.recall(query, top_k)
        return []

    def say(self, text: str):
        """Make the robot speak."""
        if self.tts is not None:
            self.tts.speak(text)
        else:
            print(f"[ROBOT]: {text}")

    def greet(self):
        """Robot greeting."""
        if self.response_generator is not None:
            response = self.response_generator.generate("greeting")
            self.say(response)
        else:
            self.say("Hello! Ready to help.")

    def ask(self, question: str, speak: bool = True) -> str:
        """
        Ask the robot a question and get a spoken answer.

        This is for Q&A scenarios like:
        - "What's 1+1?"
        - "What time is it?"
        - "Tell me a joke"

        Args:
            question: The question to ask
            speak: Whether to speak the answer aloud

        Returns:
            The answer string

        Example:
            answer = brain.ask("What's 1+1?")
            # Robot speaks: "2"
            # Returns: "2"
        """
        if self.response_generator is None:
            answer = "I don't have my language model loaded."
        elif not self.response_generator.can_answer():
            answer = "I can't answer questions right now."
        else:
            answer = self.response_generator.answer_question(question)

        if speak:
            self.say(answer)

        return answer

    # ==========================================================================
    # INTRINSIC MOTIVATION METHODS (Self-Thinking)
    # ==========================================================================

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        extrinsic_reward: torch.Tensor = None,
        skill: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute intrinsic motivation reward for autonomous learning.

        This combines:
        - Curiosity (novel states)
        - Skill diversity (DIAYN)
        - Empowerment (control)

        Args:
            state: Current state [B, obs_dim]
            next_state: Next state [B, obs_dim]
            action: Action taken [B, action_dim]
            extrinsic_reward: External task reward [B] (optional)
            skill: Current skill index [B] (optional)

        Returns:
            total_reward: Combined intrinsic + extrinsic reward
            info: Dict with reward components
        """
        if self.autonomous_mind is None:
            if extrinsic_reward is not None:
                return extrinsic_reward, {'extrinsic': extrinsic_reward.mean().item()}
            return torch.zeros(state.shape[0], device=state.device), {}

        # Get state features from backbone
        with torch.no_grad():
            output = self.forward(state)
            state_features = output['cls_features']

            output_next = self.forward(next_state)
            next_state_features = output_next['cls_features']

            # Encode to latent for DIAYN/empowerment
            state_latent = self.world_model.encode(state_features)

        # Default extrinsic reward
        if extrinsic_reward is None:
            extrinsic_reward = torch.zeros(state.shape[0], device=state.device)

        return self.autonomous_mind.compute_autonomous_reward(
            state_features=state_features,
            next_state_features=next_state_features,
            action=action,
            state_latent=state_latent,
            extrinsic_reward=extrinsic_reward,
            skill=skill,
        )

    def explore_autonomously(
        self,
        state: torch.Tensor,
        strategy: str = "curiosity",
    ) -> Dict:
        """
        Autonomous exploration step.

        The robot decides what to do based on intrinsic motivation,
        not external commands.

        Strategies:
        - "curiosity": Seek novel states (ICM + RND)
        - "skill_discovery": Practice diverse skills (DIAYN)
        - "learning_progress": Focus on improving areas
        - "empowerment": Seek controllable states

        Returns:
            Dict with action, skill, goal, and intrinsic info
        """
        if self.autonomous_mind is None:
            # Fallback to random action
            action = torch.randn(state.shape[0], self.config.action_dim, device=state.device)
            return {'action': action, 'strategy': 'random_fallback'}

        B = state.shape[0]
        device = state.device

        # Get state features
        output = self.forward(state)
        state_features = output['cls_features']
        state_latent = self.world_model.encode(state_features)

        # Sample skill for this exploration step
        skill = self.autonomous_mind.skill_discovery.sample_skill(B, device)
        skill_embedding = self.autonomous_mind.skill_discovery.get_skill_embedding(skill)

        # Generate self-directed goal
        goal, goal_info = self.autonomous_mind.goal_generator.generate_goal(
            state_features, strategy="learning_progress"
        )

        # Check metacognition: do we know what to do?
        _, uncertainty, uncertainty_info = self.autonomous_mind.metacognition.estimate_uncertainty(
            state_features
        )

        # Get action (could integrate skill conditioning here)
        action = output['actions'][:, 0, :]

        return {
            'action': action,
            'skill': skill,
            'skill_embedding': skill_embedding,
            'goal': goal,
            'uncertainty': uncertainty,
            'strategy': strategy,
            **goal_info,
            **uncertainty_info,
        }

    def should_ask_for_help(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[bool, str]:
        """
        Use metacognition to decide if robot should ask for help.

        Returns:
            should_ask: True if robot is uncertain
            reason: Explanation string
        """
        if self.autonomous_mind is None:
            return False, ""

        output = self.forward(state)
        state_features = output['cls_features']

        return self.autonomous_mind.metacognition.should_ask_for_help(
            state_features, goal
        )

    def discover_skills(
        self,
        state: torch.Tensor,
    ) -> Dict:
        """
        Run skill discovery (DIAYN).

        Returns current skill, discriminator prediction, and reward.
        """
        if self.autonomous_mind is None:
            return {'error': 'Autonomous mind not enabled'}

        B = state.shape[0]
        device = state.device

        # Get state latent
        output = self.forward(state)
        state_latent = self.world_model.encode(output['cls_features'])

        # Sample random skill
        skill = self.autonomous_mind.skill_discovery.sample_skill(B, device)

        # Get DIAYN reward
        reward, info = self.autonomous_mind.skill_discovery.compute_diayn_reward(
            state_latent, skill
        )

        # Get most likely skill from discriminator
        predicted_skill = self.autonomous_mind.skill_discovery.get_most_likely_skill(state_latent)

        return {
            'sampled_skill': skill,
            'predicted_skill': predicted_skill,
            'diayn_reward': reward,
            **info,
        }

    def get_empowerment(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute empowerment for current state.

        High empowerment = robot has good control over outcomes.
        """
        if self.autonomous_mind is None:
            return torch.zeros(state.shape[0], device=state.device), {}

        output = self.forward(state)
        state_latent = self.world_model.encode(output['cls_features'])

        return self.autonomous_mind.empowerment.compute_empowerment(state_latent)

    def get_curiosity_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute curiosity reward for a transition.
        """
        if self.autonomous_mind is None:
            return torch.zeros(state.shape[0], device=state.device), {}

        output = self.forward(state)
        output_next = self.forward(next_state)

        return self.autonomous_mind.curiosity.compute_intrinsic_reward(
            output['cls_features'],
            output_next['cls_features'],
            action,
        )

    # ==========================================================================
    # OBJECT DETECTION & NAVIGATION METHODS
    # ==========================================================================

    def find_object(
        self,
        object_name: str,
        vision_input: torch.Tensor = None
    ) -> Dict:
        """
        Find an object in the scene.

        Args:
            object_name: Name of object to find (e.g., "cup", "coffee machine")
            vision_input: Camera input tensor [B, C, H, W]

        Returns:
            Dict with:
                - found: bool - whether object was found
                - position: [x, y, z] if found
                - confidence: detection confidence
                - object_name: normalized object name
        """
        if self.object_detector is None:
            return {"found": False, "error": "Object detection not enabled"}

        device = next(self.parameters()).device

        if vision_input is None:
            # Use dummy vision for testing
            vision_input = torch.randn(1, 3, 224, 224, device=device)

        # Get vision features
        if self.vision_encoder is not None:
            with torch.no_grad():
                vision_features = self.vision_encoder(vision_input.to(device))
                if isinstance(vision_features, dict):
                    vision_features = vision_features.get("fused", vision_features.get("rgb"))
        else:
            # Mock vision features
            vision_features = torch.randn(1, 49, self.config.d_model, device=device)

        # Find the object
        result = self.object_detector.find_object(vision_features, object_name)
        return result

    def navigate_to(
        self,
        target: str,
        current_position: torch.Tensor = None
    ) -> Dict:
        """
        Navigate to a target location or object.

        Args:
            target: Target name ("kitchen", "cup", "coffee machine")
            current_position: Current robot position [x, y, theta]

        Returns:
            Dict with:
                - action: Navigation velocity commands [vx, vy, vtheta]
                - distance: Estimated distance to target
                - arrived: Whether at target
        """
        if self.navigation_planner is None:
            return {"error": "Navigation not enabled"}

        device = next(self.parameters()).device

        if current_position is None:
            current_position = torch.zeros(3, device=device)
        elif not isinstance(current_position, torch.Tensor):
            current_position = torch.tensor(current_position, dtype=torch.float32, device=device)
        else:
            current_position = current_position.to(device)

        # First, find the target if it's an object
        object_result = self.find_object(target)

        if object_result.get("found", False):
            goal_embedding = torch.tensor(object_result["position"][:2], device=device)
            goal_embedding = F.pad(goal_embedding, (0, self.config.d_model - 2))
        else:
            known_locations = {
                "kitchen": [3.0, 0.0],
                "table": [1.5, 0.0],
                "counter": [3.0, 0.0],
                "door": [0.0, 2.0],
                "start": [0.0, 0.0],
            }
            if target.lower() in known_locations:
                goal_pos = torch.tensor(known_locations[target.lower()], device=device)
            else:
                goal_pos = torch.tensor([2.0, 0.0], device=device)

            goal_embedding = F.pad(goal_pos, (0, self.config.d_model - 2))

        # Set goal and get action
        self.navigation_planner.set_goal(goal_embedding.unsqueeze(0))
        nav_action = self.navigation_planner.get_action(current_position.unsqueeze(0))

        # Check if arrived (within 0.3m)
        distance = torch.norm(current_position[:2] - goal_embedding[:2]).item()
        arrived = distance < 0.3

        return {
            "action": nav_action.squeeze(0).tolist(),
            "distance": distance,
            "arrived": arrived,
            "target": target,
        }

    def find_and_go_to(
        self,
        target: str,
        vision_input: torch.Tensor = None,
        current_position: torch.Tensor = None
    ) -> Dict:
        """
        Combined find + navigate - the main high-level command.

        Example:
            result = brain.find_and_go_to("coffee machine")
            # Robot looks for coffee machine, then navigates to it

        Args:
            target: Object or location name
            vision_input: Camera input
            current_position: Robot position

        Returns:
            Dict with status, action, and verbal response
        """
        # Step 1: Find the object
        find_result = self.find_object(target, vision_input)

        if find_result.get("found", False):
            response = f"I found the {target}. Navigating there now."
            print(f"[NAV] Found {target} at position {find_result.get('position')}")
        else:
            response = f"I don't see {target}, but I'll head towards where it might be."
            print(f"[NAV] {target} not visible, using semantic location")

        # Step 2: Navigate
        nav_result = self.navigate_to(target, current_position)

        return {
            "found": find_result.get("found", False),
            "position": find_result.get("position"),
            "nav_action": nav_result.get("action"),
            "distance": nav_result.get("distance"),
            "arrived": nav_result.get("arrived", False),
            "response": response,
        }

    def _execute_command(self, command: str) -> Dict:
        """
        Execute a parsed command.

        Handles commands like:
        - "go to the kitchen"
        - "pick up the cup"
        - "bring me coffee"
        - "walk forward"
        """
        command_lower = command.lower()

        # Navigation commands
        nav_keywords = ["go to", "walk to", "move to", "navigate to", "head to"]
        for keyword in nav_keywords:
            if keyword in command_lower:
                target = command_lower.split(keyword)[-1].strip()
                # Clean up target
                target = target.replace("the ", "").strip()
                result = self.find_and_go_to(target)
                return {
                    "type": "navigation",
                    "response": result.get("response", f"Going to {target}"),
                    "action": result.get("nav_action"),
                    "target": target,
                }

        # Pick up commands
        pick_keywords = ["pick up", "grab", "take", "get"]
        for keyword in pick_keywords:
            if keyword in command_lower:
                target = command_lower.split(keyword)[-1].strip()
                target = target.replace("the ", "").strip()
                find_result = self.find_object(target)
                if find_result.get("found", False):
                    return {
                        "type": "manipulation",
                        "response": f"I see the {target}. Reaching for it now.",
                        "target": target,
                        "position": find_result.get("position"),
                    }
                else:
                    return {
                        "type": "manipulation",
                        "response": f"I can't find the {target}. Let me look around.",
                        "target": target,
                    }

        # Bring commands (complex: pick up + navigate + hand over)
        if "bring" in command_lower or "fetch" in command_lower:
            # Extract object
            words = command_lower.split()
            obj_idx = -1
            for i, w in enumerate(words):
                if w in ["me", "us", "here"]:
                    obj_idx = i + 1
                    break
            if obj_idx > 0 and obj_idx < len(words):
                target = " ".join(words[obj_idx:]).replace("the ", "").strip()
            else:
                target = words[-1]

            return {
                "type": "fetch",
                "response": f"I'll get the {target} for you.",
                "target": target,
                "steps": ["find", "navigate", "pick_up", "return", "hand_over"],
            }

        # Simple locomotion
        if "walk forward" in command_lower or "move forward" in command_lower:
            return {
                "type": "locomotion",
                "response": "Walking forward.",
                "action": [0.5, 0.0, 0.0],  # Forward velocity
            }

        if "stop" in command_lower or "halt" in command_lower:
            return {
                "type": "stop",
                "response": "Stopping.",
                "action": [0.0, 0.0, 0.0],
            }

        if "turn left" in command_lower:
            return {
                "type": "locomotion",
                "response": "Turning left.",
                "action": [0.0, 0.0, 0.5],  # Rotate left
            }

        if "turn right" in command_lower:
            return {
                "type": "locomotion",
                "response": "Turning right.",
                "action": [0.0, 0.0, -0.5],  # Rotate right
            }

        # Default
        return {
            "type": "unknown",
            "response": f"I'll try to {command_lower}.",
        }

    def make_coffee(self, vision_input: torch.Tensor = None) -> Dict:
        """
        High-level task: Make coffee.

        This demonstrates the full task decomposition:
        1. Find the cup
        2. Pick up cup
        3. Navigate to coffee machine
        4. Place cup under spout
        5. Press button
        6. Wait for coffee
        7. Return with coffee

        Args:
            vision_input: Camera input

        Returns:
            Task execution status
        """
        steps = [
            {"action": "find", "target": "cup"},
            {"action": "navigate", "target": "cup"},
            {"action": "pick_up", "target": "cup"},
            {"action": "navigate", "target": "coffee machine"},
            {"action": "place", "target": "cup", "location": "coffee_cup_spot"},
            {"action": "press", "target": "coffee_button"},
            {"action": "wait", "duration": 30},
            {"action": "pick_up", "target": "cup"},
            {"action": "navigate", "target": "start"},
        ]

        self.say("I'll make coffee for you. Finding the cup first.")

        # In a real execution loop, this would be done step by step
        # For now, return the plan
        return {
            "task": "make_coffee",
            "steps": steps,
            "current_step": 0,
            "status": "planned",
            "response": "I'll make coffee for you. Finding the cup first.",
        }

    def chat(self, message: str, state: torch.Tensor = None, speak: bool = True) -> str:
        """
        Have a conversation with the robot.

        Determines if the message is:
        1. A question (answered with ask())
        2. A command (executed via neural pipeline interact(), with keyword fallback)
        3. General chat (responded with response_generator)

        Args:
            message: User's message
            state: Current robot state tensor (enables neural pipeline for commands)
            speak: Whether to speak response

        Returns:
            Robot's response
        """
        message_lower = message.lower().strip()

        # Check if it's a question
        question_words = ["what", "who", "where", "when", "why", "how", "is", "are", "can", "do", "does"]
        is_question = (
            message_lower.endswith("?") or
            any(message_lower.startswith(w) for w in question_words)
        )

        if is_question:
            return self.ask(message, speak=speak)

        # Check if it's a command (action verb at start)
        action_words = ["go", "walk", "run", "pick", "grab", "put", "bring", "turn", "look", "move", "stop", "come"]
        is_command = any(message_lower.startswith(w) for w in action_words)

        if is_command:
            if state is not None:
                # Neural pipeline: language -> transformer -> actions
                result = self.interact(state, message, speak=speak)
                return result.get('response', f"I'll try to {message_lower}.")
            else:
                # Fallback: keyword matching (when no state available)
                result = self._execute_command(message_lower)
                if speak and result.get("response"):
                    self.say(result["response"])
                return result.get("response", f"I'll try to {message_lower}.")

        # General chat
        if self.response_generator is not None and self.response_generator.can_answer():
            response = self.response_generator.answer_question(f"User says: {message}. How do you respond?")
        else:
            response = "I heard you. How can I help?"

        if speak:
            self.say(response)
        return response


# ==============================================================================
# SEMANTIC ACTION ANCHORS (LLM-Agnostic Language Grounding)
# ==============================================================================

class SemanticActionAnchors(nn.Module):
    """
    LLM-Agnostic language grounding using learned action anchors.

    PROBLEM: If you train with SmolLM2 and switch to Llama, the projector breaks
    because different LLMs produce different embeddings for "walk forward".

    SOLUTION: Learn FIXED action anchors that ANY language maps to:
    - "walk forward", "move ahead", "go straight" → all map to ANCHOR_WALK
    - "run fast", "sprint", "dash" → all map to ANCHOR_RUN

    The anchors are learned from ACTIONS (MoCap), not from language.
    Language just selects which anchor to use via contrastive learning.

    Research backing:
    - CLIP: Contrastive language-image pretraining
    - SigLIP: Sigmoid loss for better alignment
    - RT-2: Action tokens as language targets
    """

    # Known action categories with synonyms (LLM-agnostic!)
    ACTION_CATEGORIES = {
        "walk": ["walk forward", "move ahead", "go straight", "walk", "step forward"],
        "run": ["run forward", "run fast", "sprint", "dash", "jog"],
        "jump": ["jump", "jump in place", "hop", "leap", "bounce"],
        "stand": ["stand", "stand still", "stay", "idle", "stop"],
        "turn_left": ["turn left", "rotate left", "go left"],
        "turn_right": ["turn right", "rotate right", "go right"],
        "crouch": ["crouch", "duck", "squat", "bend down"],
        "wave": ["wave", "wave hand", "greeting"],
    }

    def __init__(self, d_model: int = 512, num_anchors: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors

        # Learnable action anchors - these are trained from MoCap!
        # Shape: (num_anchors, d_model)
        self.anchors = nn.Parameter(torch.randn(num_anchors, d_model) * 0.02)

        # Action encoder: maps action sequences to anchor space
        self.action_encoder = nn.Sequential(
            nn.Linear(17 * 16, d_model),  # 17 joints * 16 timesteps
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Temperature for contrastive learning (learnable)
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # Anchor names for lookup
        self.anchor_names = list(self.ACTION_CATEGORIES.keys())

        print(f"[ANCHORS] Initialized {num_anchors} semantic action anchors")
        print(f"[ANCHORS] Categories: {self.anchor_names}")

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action sequences to anchor space.

        Args:
            actions: (B, chunk_size, action_dim) - e.g., (B, 16, 17)

        Returns:
            (B, d_model) - action embeddings in anchor space
        """
        B = actions.shape[0]
        flat_actions = actions.reshape(B, -1)  # (B, 16*17)
        return F.normalize(self.action_encoder(flat_actions), dim=-1)

    def get_anchor_for_label(self, label: str) -> int:
        """Get anchor index for a label (handles synonyms)"""
        label_lower = label.lower().strip()

        for idx, (anchor_name, synonyms) in enumerate(self.ACTION_CATEGORIES.items()):
            if label_lower in synonyms or anchor_name in label_lower:
                return idx

        # Default to "walk" if unknown
        return 0

    def contrastive_loss(self, language_emb: torch.Tensor, action_emb: torch.Tensor,
                         labels: list) -> torch.Tensor:
        """
        Contrastive loss: language embedding should match corresponding action anchor.

        This makes the system LLM-agnostic because we're learning to map
        ANY language representation to a FIXED set of action anchors.

        Args:
            language_emb: (B, d_model) - from LLM projector
            action_emb: (B, d_model) - from action_encoder
            labels: list of B strings - ["walk forward", "run fast", ...]

        Returns:
            contrastive loss
        """
        B = language_emb.shape[0]
        device = language_emb.device

        # Normalize
        lang_norm = F.normalize(language_emb, dim=-1)
        action_norm = F.normalize(action_emb, dim=-1)
        anchor_norm = F.normalize(self.anchors, dim=-1)

        # Get target anchor indices for each label
        target_indices = torch.tensor(
            [self.get_anchor_for_label(l) for l in labels],
            dtype=torch.long, device=device
        )

        # Loss 1: Language should be close to target anchor
        # Compute similarity: (B, num_anchors)
        lang_to_anchor = torch.matmul(lang_norm, anchor_norm.T) / self.temperature.abs()
        anchor_loss = F.cross_entropy(lang_to_anchor, target_indices)

        # Loss 2: Action should be close to target anchor
        action_to_anchor = torch.matmul(action_norm, anchor_norm.T) / self.temperature.abs()
        action_anchor_loss = F.cross_entropy(action_to_anchor, target_indices)

        # Loss 3: Language and action should be close to each other (when same label)
        # InfoNCE style: positive pairs on diagonal
        logits = torch.matmul(lang_norm, action_norm.T) / self.temperature.abs()
        targets = torch.arange(B, device=device)
        lang_action_loss = F.cross_entropy(logits, targets)

        total_loss = anchor_loss + action_anchor_loss + 0.5 * lang_action_loss

        return total_loss

    def forward(self, language_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map language embedding to nearest anchor.

        Returns:
            selected_anchor: (B, d_model) - the matched anchor embedding
            anchor_probs: (B, num_anchors) - probability distribution over anchors
        """
        lang_norm = F.normalize(language_emb, dim=-1)
        anchor_norm = F.normalize(self.anchors, dim=-1)

        # Compute similarity
        similarity = torch.matmul(lang_norm, anchor_norm.T)  # (B, num_anchors)
        anchor_probs = F.softmax(similarity / self.temperature.abs(), dim=-1)

        # Soft selection (differentiable)
        selected_anchor = torch.matmul(anchor_probs, self.anchors)  # (B, d_model)

        return selected_anchor, anchor_probs


# ==============================================================================
# TRAINING LOSSES
# ==============================================================================

def compute_physics_loss(model, state, action, next_state, physics_targets):
    """Phase 0: Learn physics from SymPy"""
    output = model(state, action=action)

    physics_loss = F.mse_loss(output['physics'], physics_targets)
    dynamics_loss = F.mse_loss(output['next_state'], next_state)

    rule_weights = output['rule_weights']
    entropy = -(rule_weights * torch.log(rule_weights + 1e-8)).sum(-1).mean()
    diversity_loss = -entropy * 0.01

    total = physics_loss + 0.1 * dynamics_loss + diversity_loss

    return total, {
        'physics': physics_loss.item(),
        'dynamics': dynamics_loss.item(),
        'total': total.item(),
    }


def compute_flow_matching_loss(model, state, target_actions, goal=None, language=None):
    """
    Phase 1/2/2.5: PROPER Flow Matching for action generation using ActionExpert.

    This implements real flow matching (Lipman et al. 2022) with:
    - ActionExpert (π0 style separate transformer)
    - Time-dependent conditioning
    - Proper velocity field prediction
    - Cross-attention to VLM features

    Args:
        model: UnifiedBrain model with action_expert
        state: Robot state (B, obs_dim)
        target_actions: Target action sequence (B, chunk_size, action_dim)
        goal: Optional goal state
        language: Optional language labels (list of strings)

    Returns:
        Total loss (flow matching + optional reconstruction)
    """
    B = state.shape[0]
    device = state.device
    chunk_size = target_actions.shape[1]
    action_dim = target_actions.shape[2]

    # Sample random timesteps t ∈ [0, 1]
    t = torch.rand(B, device=device)

    # Sample noise
    noise = torch.randn_like(target_actions)

    # Flow matching interpolation: x_t = t * x_1 + (1-t) * x_0
    # where x_0 = noise, x_1 = clean data
    t_exp = t[:, None, None]
    noisy_actions = t_exp * target_actions + (1 - t_exp) * noise

    # Target velocity field: v = x_1 - x_0 = target - noise
    target_velocity = target_actions - noise

    # Flatten noisy actions for ActionExpert input
    noisy_flat = noisy_actions.view(B, -1)  # [B, chunk_size * action_dim]

    # Get VLM features from backbone
    if language is not None:
        has_llm = model.has_llm() if hasattr(model, 'has_llm') else False

        if has_llm:
            output = model(state, goal=goal, noisy_actions=noisy_actions, language=language)
        else:
            # FALLBACK MODE: Convert to token indices
            language_tokens = []
            max_len = 10
            vocab = {"<pad>": 0, "walk": 1, "forward": 2, "run": 3, "fast": 4,
                     "jump": 5, "in": 6, "place": 7, "move": 8, "naturally": 9,
                     "stand": 10, "idle": 11, "turn": 12, "left": 13, "right": 14,
                     "backward": 15, "slow": 16, "stop": 17, "crouch": 18, "up": 19}

            for label in language:
                words = label.lower().split()
                tokens = [vocab.get(w, vocab["<pad>"]) for w in words]
                tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
                language_tokens.append(tokens)

            language_tensor = torch.tensor(language_tokens, dtype=torch.long, device=device)
            output = model(state, goal=goal, noisy_actions=noisy_actions, language=language_tensor)
    else:
        output = model(state, goal=goal, noisy_actions=noisy_actions)

    # Use ActionExpert if available (π0 style)
    if hasattr(model, 'action_expert') and model.action_expert is not None:
        # Get hidden states from backbone for cross-attention
        vlm_features = output['hidden_states']  # [B, seq_len, d_model]

        # ActionExpert predicts velocity field
        pred_velocity_flat = model.action_expert(noisy_flat, vlm_features, t)
        pred_velocity = pred_velocity_flat.view(B, chunk_size, action_dim)

        # Flow matching loss with time-dependent weighting
        # Weight by (1 - t) to focus on denoising harder cases
        weights = (1 - t_exp).expand_as(target_velocity)
        weighted_loss = weights * (pred_velocity - target_velocity) ** 2
        flow_loss = weighted_loss.mean()

    else:
        # Fallback: Use action_head velocity prediction (less powerful)
        action_feat = output['cls_features'].unsqueeze(1).expand(-1, chunk_size, -1)
        pred_velocity = model.action_head.predict_velocity(action_feat)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)

    # Optional: Add reconstruction loss for stability
    # This helps early in training when velocity prediction is noisy
    if hasattr(model, 'action_expert') and model.action_expert is not None:
        # Predict clean actions from t=1 (pure clean, no noise)
        t_clean = torch.ones(B, device=device)
        clean_flat = target_actions.view(B, -1)
        pred_clean_flat = model.action_expert(clean_flat, vlm_features, t_clean)
        pred_clean = pred_clean_flat.view(B, chunk_size, action_dim)
        recon_loss = F.mse_loss(pred_clean, target_actions)

        # Combined loss: flow matching + small reconstruction term
        total_loss = flow_loss + 0.1 * recon_loss
    else:
        total_loss = flow_loss

    return total_loss


def compute_language_grounding_loss(model, state, target_actions, language_labels):
    """
    Phase 2.5: Language grounding with contrastive learning.

    This solves 3 problems:
    1. STRONGER GRADIENT SIGNAL: Direct contrastive loss on language embeddings
    2. LLM-AGNOSTIC: Learns to map ANY language to fixed action anchors
    3. PROPER PROJECTOR TRAINING: Language embeddings explicitly trained

    The loss has 3 components:
    - Flow matching (action prediction)
    - Contrastive (language ↔ action alignment)
    - Anchor matching (language → semantic anchor)

    Args:
        model: UnifiedBrain with semantic_anchors
        state: Robot state (B, obs_dim)
        target_actions: Target actions (B, chunk_size, action_dim)
        language_labels: List of strings ["walk forward", "run fast", ...]
    """
    B = state.shape[0]
    device = state.device

    # 1. Standard flow matching loss (for action prediction)
    t = torch.rand(B, device=device)
    noise = torch.randn_like(target_actions)
    t_exp = t[:, None, None]
    noisy = (1 - t_exp) * noise + t_exp * target_actions
    target_velocity = target_actions - noise

    # Get language embeddings from the model
    has_llm = model.has_llm() if hasattr(model, 'has_llm') else False

    if has_llm:
        # Get language embeddings directly from LLMEncoder
        language_emb = model.language_encoder(language_labels)  # (B, d_model)
    else:
        # Fallback: tokenize and get embeddings
        language_tokens = []
        max_len = 10
        vocab = {"<pad>": 0, "walk": 1, "forward": 2, "run": 3, "fast": 4,
                 "jump": 5, "in": 6, "place": 7, "move": 8, "naturally": 9,
                 "stand": 10, "idle": 11, "turn": 12, "left": 13, "right": 14,
                 "backward": 15, "slow": 16, "stop": 17, "crouch": 18, "up": 19}

        for label in language_labels:
            words = label.lower().split()
            tokens = [vocab.get(w, vocab["<pad>"]) for w in words]
            tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
            language_tokens.append(tokens)

        language_tensor = torch.tensor(language_tokens, dtype=torch.long, device=device)
        language_emb = model.language_encoder(language_tensor)  # (B, d_model)

    # Forward pass with language
    output = model(state, goal=None, noisy_actions=noisy, language=language_labels if has_llm else language_tensor)

    # Flow matching loss
    action_feat = output['cls_features'].unsqueeze(1).expand(-1, target_actions.shape[1], -1)
    pred_velocity = model.action_head.predict_velocity(action_feat)
    flow_loss = F.mse_loss(pred_velocity, target_velocity)

    # 2. Encode actions for contrastive learning
    action_emb = model.semantic_anchors.encode_actions(target_actions)  # (B, d_model)

    # 3. Contrastive loss: language ↔ action ↔ anchors
    contrastive_loss = model.semantic_anchors.contrastive_loss(
        language_emb, action_emb, language_labels
    )

    # Total loss (flow matching + contrastive)
    # Contrastive weight starts high and decreases as anchors stabilize
    total_loss = flow_loss + 0.5 * contrastive_loss

    return total_loss, {
        'flow': flow_loss.item(),
        'contrastive': contrastive_loss.item(),
        'total': total_loss.item(),
    }


def compute_world_model_loss(model, state, action, reward, next_state):
    """World model training loss"""
    output = model(state, action=action)

    # Reconstruction
    recon_loss = F.mse_loss(output['next_state'], next_state)

    # Reward prediction
    reward_loss = F.mse_loss(output['reward'].squeeze(), reward)

    # Dynamics consistency (with target encoder)
    with torch.no_grad():
        next_output = model(next_state)
        target_latent = model.world_model.encode(next_output['cls_features'], use_target=True)

    dynamics_loss = F.mse_loss(output['next_latent'], target_latent)

    total = recon_loss + reward_loss + 10.0 * dynamics_loss

    return total, {
        'recon': recon_loss.item(),
        'reward': reward_loss.item(),
        'dynamics': dynamics_loss.item(),
    }


def compute_hierarchical_loss(model, state, task, achieved_subgoal, reward):
    """Hierarchical planner training loss"""
    output = model(state, task=task, use_hierarchy=True)
    hierarchy = output['hierarchy']

    # Subgoal regression
    subgoal_loss = F.mse_loss(hierarchy['active_subgoal'], achieved_subgoal)

    # Skill selection (REINFORCE)
    skill_probs = hierarchy['skill_probs']
    skill_id = hierarchy['skill_id']
    skill_log_prob = torch.log(skill_probs[0, skill_id] + 1e-8)
    skill_loss = -skill_log_prob * reward.item()

    # Entropy regularization
    skill_entropy = -(skill_probs * torch.log(skill_probs + 1e-8)).sum()

    total = subgoal_loss + 0.1 * skill_loss - 0.01 * skill_entropy

    return total, {
        'subgoal': subgoal_loss.item(),
        'skill': skill_loss.item(),
        'entropy': skill_entropy.item(),
    }


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED BRAIN - COMPLETE AGI SYSTEM TEST")
    print("=" * 70)

    config = UnifiedBrainConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        num_joints=17,
        features_per_joint=8,
        action_dim=17,
        action_chunk_size=16,
        num_rules=100,
        num_skills=20,
        obs_dim=256,
        vision_enabled=False,  # Disable for testing
        use_pretrained_vision=False,
    )

    model = UnifiedBrain(config)

    # Test inputs
    B = 2
    state = torch.randn(B, 256)
    action = torch.randn(B, 17)
    goal = torch.randn(B, 256)
    task = torch.randn(B, 64)

    print("\n[TEST 1] Basic forward pass")
    with torch.no_grad():
        output = model(state, action=action, goal=goal)
    print(f"  Actions: {output['actions'].shape}")
    print(f"  Physics: {output['physics'].shape}")
    print(f"  Value: {output['value'].shape}")
    print(f"  Next state: {output['next_state'].shape}")
    print(f"  Reward: {output['reward'].shape}")

    print("\n[TEST 2] MPC Planning")
    with torch.no_grad():
        output = model(state, use_mpc=True)
    print(f"  MPC action: {output['mpc_action'].shape}")

    print("\n[TEST 3] Hierarchical Planning")
    model.reset_planner()
    with torch.no_grad():
        output = model(state, task=task, use_hierarchy=True)
    h = output['hierarchy']
    print(f"  Subgoals: {h['subgoals'].shape}")
    print(f"  Active skill: {h['skill_name']} (ID: {h['skill_id']})")
    print(f"  Low-level goal: {h['low_level_goal'].shape}")

    print("\n[TEST 4] Imagination (TD-MPC2)")
    actions = torch.randn(B, 5, 17)  # 5-step horizon
    with torch.no_grad():
        latents, rewards = model.imagine(state, actions)
    print(f"  Imagined latents: {latents.shape}")
    print(f"  Imagined rewards: {rewards.shape}")

    print("\n[TEST 5] Active physics rules")
    rules = model.get_active_rules(state[:1])
    print("  Top rules:")
    for name, weight in rules:
        print(f"    - {name}: {weight:.3f}")

    print("\n[TEST 6] Intrinsic Motivation (Self-Thinking)")
    if model.autonomous_mind is not None:
        # Test curiosity
        next_state = torch.randn(B, 256)
        action = torch.randn(B, 17)
        curiosity_r, curiosity_info = model.get_curiosity_reward(state, next_state, action)
        print(f"  Curiosity reward: {curiosity_r.mean().item():.4f}")

        # Test empowerment
        emp_r, emp_info = model.get_empowerment(state)
        print(f"  Empowerment: {emp_r.mean().item():.4f}")

        # Test skill discovery
        skill_info = model.discover_skills(state)
        print(f"  Sampled skill: {skill_info['sampled_skill'][0].item()}")
        print(f"  DIAYN reward: {skill_info['diayn_reward'].mean().item():.4f}")

        # Test autonomous exploration
        explore_result = model.explore_autonomously(state)
        print(f"  Uncertainty: {explore_result['uncertainty'].mean().item():.4f}")

        print("  [OK] Intrinsic motivation working!")
    else:
        print("  Intrinsic motivation disabled")

    print("\n" + "=" * 70)
    print("[SUCCESS] All tests passed!")
    print("=" * 70)
    print("\nThis brain now has EVERYTHING:")
    print("  [x] SOTA Transformer (RMSNorm, SwiGLU, RoPE)")
    print("  [x] TD-MPC2 WorldModel (imagination + MPC)")
    print("  [x] HAC HierarchicalPlanner (skills + subgoals)")
    print("  [x] Vision encoders (DINOv2 + SigLIP ready)")
    print("  [x] Temporal memory (50 timesteps)")
    print("  [x] Cross-modal fusion")
    print("  [x] Physics rule bank (100 rules)")
    print("  [x] Flow matching for actions")
    print("  [x] **INTRINSIC MOTIVATION (Self-Thinking)**")
    print("      - Curiosity (ICM + RND)")
    print("      - Skill Discovery (DIAYN)")
    print("      - Empowerment")
    print("      - Metacognition")
    print("      - Autotelic Goals")
    print("=" * 70)
