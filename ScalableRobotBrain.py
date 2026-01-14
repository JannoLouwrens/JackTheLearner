"""
SCALABLE ROBOT BRAIN - SYSTEM 1 (Fast Thinking, 50Hz)

This is the reactive brain that handles immediate control:
- Vision encoding (DINOv2 + SigLIP fusion from OpenVLA)
- Proprioception, touch, language encoding
- Cross-modal fusion (sensors attend to each other)
- Temporal memory (remembers past 50 timesteps)
- Diffusion policy with flow matching (1-step action inference)

Research papers implemented:
- OpenVLA (2024): DINOv2 + SigLIP vision fusion
- Physical Intelligence pi0 (2024): Flow matching for 1-step inference
- Boston Dynamics Atlas (2024): 48-action chunking
- Diffusion Policy (Columbia 2023): Denoising transformer

This is System 1 from Kahneman's "Thinking Fast and Slow":
- Fast, automatic, always-on
- Pattern matching, reactive control
- Runs at 50Hz for real-time robot control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class BrainConfig:
    """Configuration for System 1 brain"""

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

    # Context
    context_length: int = 50        # Remember last 50 timesteps
    action_chunk_size: int = 48     # Predict 48 actions ahead (Boston Dynamics)

    # Modalities
    vision_embed_dim: int = 1024    # DINOv2 + SigLIP fused
    proprio_dim: int = 256
    touch_dim: int = 64
    language_embed_dim: int = 512

    # Actions
    action_dim: int = 17            # Humanoid DOF

    # Diffusion
    use_diffusion: bool = True
    use_flow_matching: bool = True  # 1-step inference
    diffusion_steps: int = 15       # DDIM steps (if not flow matching)
    flow_matching_steps: int = 1

    # Training
    dropout: float = 0.1
    use_pretrained_vision: bool = True
    vlm_backbone: str = "prismatic"


# ==============================================================================
# VISION ENCODER (OpenVLA: DINOv2 + SigLIP Fusion)
# ==============================================================================

class PrismaticVisionEncoder(nn.Module):
    """
    Fuses DINOv2 (self-supervised) + SigLIP (vision-language) features.

    From OpenVLA paper:
    - DINOv2: Good at spatial features (where things are)
    - SigLIP: Good at semantic features (what things are)
    - Fusion: Concatenate + project = best of both
    """

    def __init__(self, config: BrainConfig, image_size=224):
        super().__init__()
        self.config = config
        self.image_size = image_size

        if config.use_pretrained_vision and config.vlm_backbone == "prismatic":
            try:
                from transformers import AutoModel, AutoImageProcessor

                # DINOv2 (1024-dim features)
                self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-large")
                self.dinov2.requires_grad_(False)

                # CLIP/SigLIP (768-dim features)
                self.siglip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
                self.siglip.requires_grad_(False)

                # Fusion projector: 1792 -> 1024
                self.projector = nn.Sequential(
                    nn.Linear(1024 + 768, config.vision_embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(config.vision_embed_dim * 2, config.vision_embed_dim),
                )
                self.use_pretrained = True

            except Exception as e:
                print(f"[WARNING] Pretrained vision failed: {e}")
                self.use_pretrained = False
        else:
            self.use_pretrained = False

        if not self.use_pretrained:
            # Fallback CNN for testing
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            )
            self.projector = nn.Linear(128, config.vision_embed_dim)

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

class ProprioceptionEncoder(nn.Module):
    """Encodes joint angles, velocities, orientation"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim),
        )
    def forward(self, x): return self.encoder(x)


class TouchEncoder(nn.Module):
    """Encodes contact forces"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.LayerNorm(output_dim),
        )
    def forward(self, x): return self.encoder(x)


class LanguageEncoder(nn.Module):
    """Encodes text instructions"""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
    def forward(self, tokens):
        _, (hidden, _) = self.encoder(self.embedding(tokens))
        return hidden[-1]


# ==============================================================================
# CROSS-MODAL FUSION
# ==============================================================================

class CrossModalFusion(nn.Module):
    """
    Sensors attend to each other via self-attention.

    Example: Vision sees "slippery floor" -> Touch confirms "low friction"
             -> Proprio adjusts "widen stance"
    """
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout, batch_first=True)
            for _ in range(config.n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4), nn.ReLU(),
                nn.Dropout(config.dropout), nn.Linear(config.d_model * 4, config.d_model),
            ) for _ in range(config.n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.n_layers * 2)])

    def forward(self, tokens, mask=None):
        x = tokens
        for i, (attn, ffn) in enumerate(zip(self.layers, self.ffns)):
            x = x + attn(self.norms[i*2](x), self.norms[i*2](x), self.norms[i*2](x), key_padding_mask=mask)[0]
            x = x + ffn(self.norms[i*2+1](x))
        return x


# ==============================================================================
# TEMPORAL MEMORY
# ==============================================================================

class TemporalMemory(nn.Module):
    """
    Remembers past observations.

    Critical for: "I tried this 3 times, it's not working, try something else"
    """
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.d_model * 4, config.dropout, batch_first=True),
            num_layers=4
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, config.context_length, config.d_model) * 0.02)

    def forward(self, seq):
        return self.encoder(seq + self.pos_encoding[:, :seq.shape[1], :])


# ==============================================================================
# DIFFUSION POLICY (Flow Matching)
# ==============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Timestep embedding for diffusion"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FlowMatchingActionDecoder(nn.Module):
    """
    Diffusion policy with flow matching for 1-step inference.

    From Physical Intelligence pi0:
    - Traditional diffusion: 15-100 denoising steps
    - Flow matching: learns velocity field, 1 step sufficient
    - Result: Real-time 50Hz control
    """
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2), nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.action_emb = nn.Linear(config.action_dim * config.action_chunk_size, config.d_model)

        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.d_model * 4, config.dropout, batch_first=True),
            num_layers=6
        )

        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2), nn.SiLU(),
            nn.Linear(config.d_model * 2, config.action_dim * config.action_chunk_size),
        )
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(self, memory, actions=None, timesteps=None):
        B = memory.shape[0]

        if self.training and actions is not None and timesteps is not None:
            # Training: denoise given noisy actions
            act_emb = self.action_emb(actions.reshape(B, -1)).unsqueeze(1)
            time_emb = self.time_emb(timesteps).unsqueeze(1)
            out = self.denoiser(torch.cat([memory, act_emb + time_emb], dim=1))
            output = self.action_head(out[:, -1, :]).reshape(B, self.config.action_chunk_size, self.config.action_dim)
        else:
            # Inference: 1-step flow matching
            output = self._flow_matching_inference(memory)

        return output, self.value_head(memory[:, -1, :])

    def _flow_matching_inference(self, memory):
        B, device = memory.shape[0], memory.device
        noise = torch.randn(B, self.config.action_chunk_size * self.config.action_dim, device=device)

        act_emb = self.action_emb(noise).unsqueeze(1)
        time_emb = self.time_emb(torch.ones(B, device=device)).unsqueeze(1)

        out = self.denoiser(torch.cat([memory, act_emb + time_emb], dim=1))
        velocity = self.action_head(out[:, -1, :])

        return (noise + velocity).reshape(B, self.config.action_chunk_size, self.config.action_dim)


# ==============================================================================
# SCALABLE ROBOT BRAIN (System 1)
# ==============================================================================

class ScalableRobotBrain(nn.Module):
    """
    System 1: Fast, reactive brain (50Hz)

    Takes: vision, proprioception, touch, language
    Outputs: 48-action chunk via diffusion policy

    This is the core VLA (Vision-Language-Action) transformer.
    """

    def __init__(self, config: BrainConfig, obs_dim: int):
        super().__init__()
        self.config = config

        # Encoders
        self.vision_encoder = PrismaticVisionEncoder(config)
        self.proprio_encoder = ProprioceptionEncoder(obs_dim, config.proprio_dim)
        self.touch_encoder = TouchEncoder(10, config.touch_dim)
        self.language_encoder = LanguageEncoder(1000, config.language_embed_dim)

        # Projections to d_model
        self.vision_proj = nn.Linear(config.vision_embed_dim, config.d_model)
        self.proprio_proj = nn.Linear(config.proprio_dim, config.d_model)
        self.touch_proj = nn.Linear(config.touch_dim, config.d_model)
        self.language_proj = nn.Linear(config.language_embed_dim, config.d_model)

        # Core
        self.fusion = CrossModalFusion(config)
        self.memory = TemporalMemory(config)
        self.decoder = FlowMatchingActionDecoder(config)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, proprio, vision=None, touch=None, language=None, history=None):
        B = proprio.shape[0]
        tokens = [self.proprio_proj(self.proprio_encoder(proprio)).unsqueeze(1)]

        if vision is not None:
            tokens.append(self.vision_proj(self.vision_encoder(vision)).unsqueeze(1))
        if touch is not None:
            tokens.append(self.touch_proj(self.touch_encoder(touch)).unsqueeze(1))
        if language is not None:
            tokens.append(self.language_proj(self.language_encoder(language)).unsqueeze(1))

        tokens.append(self.cls_token.expand(B, -1, -1))

        # Fuse modalities
        fused = self.fusion(torch.cat(tokens, dim=1))

        # Add temporal context
        if history is not None:
            seq = torch.cat([history, fused[:, -1:, :]], dim=1)
            seq = seq[:, -self.config.context_length:, :]
        else:
            seq = fused[:, -1:, :]

        mem = self.memory(seq)
        actions, values = self.decoder(mem)

        return actions, values, mem


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

def flow_matching_loss(model_output, target_actions, noisy_actions, timesteps):
    """
    Flow matching loss: learn velocity field from noise to data.

    true_velocity = target - noise
    loss = MSE(predicted_velocity, true_velocity)
    """
    true_velocity = target_actions - noisy_actions
    return F.mse_loss(model_output, true_velocity)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("Testing ScalableRobotBrain (System 1)...")

    config = BrainConfig(d_model=256, n_heads=4, n_layers=3, action_dim=17, use_pretrained_vision=False)
    brain = ScalableRobotBrain(config, obs_dim=348)

    proprio = torch.randn(2, 348)
    with torch.no_grad():
        actions, values, memory = brain(proprio)

    print(f"Actions: {actions.shape}")  # (2, 48, 17)
    print(f"Values: {values.shape}")    # (2, 1)
    print(f"Memory: {memory.shape}")    # (2, 1, 256)
    print("OK!")
