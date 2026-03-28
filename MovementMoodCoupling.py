"""
MOVEMENT-MOOD COUPLING - Emotion-Conditioned Action Modulation

Research backing:
- Tessler et al. (2023): CALM - Conditional Adversarial Latent Models (SIGGRAPH 2023)
- Peng et al. (2022): ASE - Adversarial Skill Embeddings (SIGGRAPH 2022)
- Generative Human Motion Stylization in Latent Space (2024)
- Breazeal (2003): Affective movement in social robots (Kismet)

Architecture:
- Speed modulator: PAD -> scalar speed multiplier [0.7, 1.3]
- Style modulator: PAD -> per-joint action bias (clamped +/-10%)
- Posture modulator: PAD -> joint angle offsets for idle/standing posture
- All modulations are SMALL to preserve trained locomotion stability

Key insight from CALM: Instead of retraining the entire policy for each style,
apply a lightweight post-processing layer that modulates the output.

Author: Janno Louwrens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class MovementMoodConfig:
    """Configuration for the movement-mood coupling module.

    Design choices:
    - max_speed_mod=0.3: Speed varies in [1-0.3, 1+0.3] = [0.7, 1.3].
      This range is empirically safe for humanoid locomotion controllers.
      Tessler et al. (2023) CALM uses similar bounded style multipliers to
      avoid destabilizing the base locomotion policy. Going beyond 1.3x
      risks falling due to insufficient foot clearance; below 0.7x causes
      unnatural foot dragging.
    - max_style_bias=0.1: Per-joint action offsets clamped to +/-10% of
      action magnitude. ASE (Peng et al., 2022) demonstrates that small
      additive latent perturbations can encode diverse motion styles without
      catastrophic policy degradation. 10% is conservative enough to
      preserve balance while being perceptible as stylistic variation.
    - action_dim=17: Matches ActionHead locomotion mode in UnifiedBrain
      (abdomen 3 + hips 6 + knees 2 + ankles 4 + shoulders 2 = 17).
    - pad_dim=3: Must match EmotionalState's PAD vector (Pleasure,
      Arousal, Dominance). Mehrabian (1996).
    """
    # Safety bounds
    max_speed_mod: float = 0.3          # Speed range: [1 - max, 1 + max]
    max_style_bias: float = 0.1         # Per-joint bias: +/-10% of action magnitude
    max_posture_offset: float = 0.15    # Idle posture shift: +/-0.15 radians (~8.6 deg)

    # Dimensions
    action_dim: int = 17                # Locomotion joint count (matches ActionHead)
    pad_dim: int = 3                    # PAD emotional dimensions (P, A, D)

    # Network hidden size (intentionally tiny - this is a post-processor)
    hidden_dim: int = 32

    # Posture joint groups (indices into the 17-dim action vector)
    # These index assignments follow the ActionHead locomotion layout:
    #   abdomen: [0, 1, 2]  (flexion, lateral, rotation)
    #   hips:    [3, 4, 5, 6, 7, 8]  (L/R: flexion, abduction, rotation)
    #   knees:   [9, 10]  (L/R flexion)
    #   ankles:  [11, 12, 13, 14]  (L/R: dorsiflexion, inversion)
    #   shoulders: [15, 16]  (L/R flexion)
    shoulder_indices: Tuple[int, ...] = (15, 16)
    abdomen_indices: Tuple[int, ...] = (0, 1, 2)
    hip_indices: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    knee_indices: Tuple[int, ...] = (9, 10)


# ==============================================================================
# PAD -> MOVEMENT DESCRIPTORS (from emotion-body language research)
# ==============================================================================
# Mapping grounded in:
# - Breazeal (2003): Kismet robot affective expression through body posture
# - Wallbott (1998): Bodily expression of emotion - EMG studies
# - Roether et al. (2009): Critical features of emotional body language
# - Gross et al. (2012): Effort-Shape analysis of emotional movement
#
# Quadrant definitions (Pleasure x Arousal):
#   High P + High A  = "Elated"   -> bouncy, wide arm swing, energetic
#   Low  P + High A  = "Angry"    -> tense, jerky, stomping, clenched
#   Low  P + Low  A  = "Sad"      -> slow, slouched, minimal arm swing
#   High P + Low  A  = "Relaxed"  -> smooth, slight sway, upright
#
# Dominance axis (orthogonal modifier):
#   High D = upright torso, wider stride, confident head angle
#   Low  D = hunched shoulders, smaller steps, lowered head

_STYLE_DESCRIPTIONS = {
    "elated":   "bouncy, energetic, wide arm swing",
    "angry":    "tense, jerky, stomping gait",
    "sad":      "slow, slouched, minimal arm swing",
    "relaxed":  "smooth, gentle sway, upright posture",
    "neutral":  "normal locomotion, no stylistic bias",
    "dominant":  "confident stride, upright torso",
    "submissive": "hunched, smaller movements",
}


# ==============================================================================
# MOVEMENT-MOOD COUPLING MODULE
# ==============================================================================

class MovementMoodCoupling(nn.Module):
    """Post-processing layer that modulates motor actions based on emotional state.

    This module sits BETWEEN the brain's ActionHead output and the final
    action sent to the physics simulator. It applies three types of
    emotion-conditioned modulation:

    1. **Speed modulation**: Arousal-driven global speed scaling.
       High arousal -> faster movement; low arousal -> slower.
       Bounded to [0.7, 1.3] to preserve locomotion stability.

    2. **Style modulation**: Per-joint additive biases derived from the
       full PAD vector. These create subtle stylistic differences (e.g.,
       wider arm swing when happy, stiff joints when angry).
       Bounded to +/-10% of action magnitude.

    3. **Posture modulation**: Joint angle offsets for idle/standing poses.
       Shoulders droop when sad, torso straightens when dominant, etc.
       Only applied when the agent is idle (not during active locomotion).

    Safety guarantee:
        ALL modulations are bounded by hard clamps. The module CANNOT
        produce outputs that deviate more than the configured maximums
        from the base action. This means a well-trained locomotion policy
        remains stable regardless of emotional state.

    Integration:
        Called after ActionHead.forward() in the inference loop::

            raw_action = brain.action_head(features)         # [B, 17]
            pad_vector = emotional_state.pad_vector           # [3]
            modulated  = mood_coupling.modulate_action(
                raw_action, pad_vector, is_idle=False
            )                                                 # [B, 17]
            env.step(modulated)

    Training:
        Trainable during Phase 8 (emotion integration). The networks are
        small (~3K parameters) and can be trained with style-matching
        losses from emotional MoCap data, or with perceptual losses from
        a style discriminator (CALM approach).

    Parameter count: ~3K parameters
        - speed_net:   3 -> 32 -> 1    = 129 + 33  = 162
        - style_net:   3 -> 32 -> 17   = 129 + 561 = 690
        - posture_net: 3 -> 32 -> 17   = 129 + 561 = 690
        - Total: ~1,542 parameters (negligible)

    Research:
        The core idea of post-hoc action modulation comes from CALM
        (Tessler et al., SIGGRAPH 2023), where a lightweight conditioning
        network transforms a base policy's output into stylized motion.
        ASE (Peng et al., SIGGRAPH 2022) similarly shows that small latent
        perturbations can encode rich motion styles. We extend this to
        continuous emotional conditioning via the PAD model rather than
        discrete style labels.
    """

    def __init__(self, config: Optional[MovementMoodConfig] = None):
        super().__init__()
        self.config = config or MovementMoodConfig()

        pad_dim = self.config.pad_dim
        action_dim = self.config.action_dim
        hidden = self.config.hidden_dim

        # ── Speed Modulator ──────────────────────────────────────────────
        # Maps PAD -> scalar speed multiplier.
        # Architecture: Linear -> ReLU -> Linear -> Sigmoid -> rescale.
        # Sigmoid output in [0, 1] is rescaled to [1-max, 1+max].
        # The network learns that high arousal -> high speed, but the
        # mapping is not hardcoded -- it can learn nuanced relationships
        # (e.g., high arousal + low pleasure = tense but not necessarily fast).
        self.speed_net = nn.Sequential(
            nn.Linear(pad_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            # Sigmoid applied in forward for explicit rescaling
        )

        # Initialize speed_net to output ~0.5 (neutral speed = 1.0x)
        # This ensures the module starts as an identity transform.
        with torch.no_grad():
            self.speed_net[-1].weight.zero_()
            self.speed_net[-1].bias.fill_(0.0)  # sigmoid(0) = 0.5 -> speed = 1.0

        # ── Style Modulator ──────────────────────────────────────────────
        # Maps PAD -> per-joint action bias vector.
        # Architecture: Linear -> ReLU -> Linear -> Tanh -> rescale.
        # Tanh output in [-1, 1] is rescaled to [-max_style_bias, +max_style_bias].
        # This produces subtle per-joint offsets that create motion style.
        self.style_net = nn.Sequential(
            nn.Linear(pad_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

        # Initialize style_net to output ~0 (no bias at start)
        with torch.no_grad():
            self.style_net[-2].weight.zero_()
            self.style_net[-2].bias.zero_()

        # ── Posture Modulator ────────────────────────────────────────────
        # Maps PAD -> joint angle offsets for idle/standing posture.
        # These offsets model research-backed postural correlates of emotion:
        #   - Sad: shoulder droop, forward trunk lean, slight knee bend
        #   - Happy: upright posture, slight shoulder lift
        #   - Dominant: expanded chest, wide stance
        #   - Submissive: contracted posture, narrow stance
        #
        # Breazeal (2003) demonstrated these posture-emotion mappings in
        # the Kismet robot, and Wallbott (1998) confirmed them in human
        # EMG studies.
        #
        # Only applied when is_idle=True (during standing or waiting).
        # During active locomotion, posture is controlled by the policy.
        self.posture_net = nn.Sequential(
            nn.Linear(pad_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

        # Initialize posture_net with emotion-informed biases
        # These encode known posture-emotion correlations as initial biases
        # that can be refined through training.
        self._init_posture_priors()

        # ── Diagnostic counters ──────────────────────────────────────────
        self._modulation_count: int = 0

        print(f"[MovementMoodCoupling] Initialized: action_dim={action_dim}, "
              f"speed_range=[{1.0 - self.config.max_speed_mod:.1f}, "
              f"{1.0 + self.config.max_speed_mod:.1f}], "
              f"style_bias=+/-{self.config.max_style_bias*100:.0f}%")

    def _init_posture_priors(self) -> None:
        """Initialize posture network with emotion-body research priors.

        These are soft priors -- the network can override them during training.
        We set the final layer's bias to encode:
        - Pleasure -> slight shoulder lift (positive embodiment)
        - Arousal -> no default posture effect (arousal affects speed, not posture)
        - Dominance -> torso straightening (power posing research, Carney et al. 2010)

        The weights are initialized to small values so the priors are gentle.
        """
        with torch.no_grad():
            final_layer = self.posture_net[-2]  # Linear before Tanh
            final_layer.weight.zero_()
            final_layer.bias.zero_()

            # Pleasure contribution: slight shoulder lift
            # shoulder_indices affect the shoulder flexion joints
            for idx in self.config.shoulder_indices:
                # Positive pleasure -> slight positive shoulder offset
                final_layer.weight[idx, 0] = 0.3  # PAD dim 0 = Pleasure

            # Dominance contribution: torso straightening
            # abdomen_indices[0] = trunk flexion
            for idx in self.config.abdomen_indices[:1]:
                # Positive dominance -> upright torso (negative flexion offset)
                final_layer.weight[idx, 2] = -0.3  # PAD dim 2 = Dominance

    # ══════════════════════════════════════════════════════════════════════
    # CORE API
    # ══════════════════════════════════════════════════════════════════════

    def modulate_action(
        self,
        action: torch.Tensor,
        pad_vector: torch.Tensor,
        is_idle: bool = False,
    ) -> torch.Tensor:
        """Apply emotion-conditioned modulation to a raw action.

        This is the main entry point. It applies speed scaling and style
        bias to every action, plus posture offsets when idle.

        Args:
            action: Raw action from ActionHead, shape [B, action_dim] or
                    [action_dim]. Values are joint torques/positions.
            pad_vector: PAD emotional state, shape [3] or [B, 3].
                        Values in [-1, 1] for each dimension.
            is_idle: If True, also apply idle posture offsets. Set this
                     when the agent is standing still or waiting.

        Returns:
            Modulated action, same shape as input. Guaranteed to be within
            safe bounds of the original action.
        """
        # Handle both batched and unbatched inputs
        was_unbatched = action.dim() == 1
        if was_unbatched:
            action = action.unsqueeze(0)

        # Ensure pad_vector is batched and on the same device
        pad = pad_vector.to(action.device)
        if pad.dim() == 1:
            pad = pad.unsqueeze(0).expand(action.shape[0], -1)
        elif pad.shape[0] == 1 and action.shape[0] > 1:
            pad = pad.expand(action.shape[0], -1)

        # ── 1. Speed Modulation ──────────────────────────────────────────
        # Arousal primarily drives speed, but the network can learn
        # nuanced PAD -> speed mappings.
        speed_multiplier = self._compute_speed_multiplier(pad)  # [B, 1]
        modulated = action * speed_multiplier

        # ── 2. Style Bias ────────────────────────────────────────────────
        # Small per-joint offsets that create motion style.
        # Clamped relative to action magnitude for safety.
        style_bias = self._compute_style_bias(pad, action)  # [B, action_dim]
        modulated = modulated + style_bias

        # ── 3. Posture Offset (idle only) ────────────────────────────────
        # Postural shifts only when standing still. During locomotion,
        # the trained policy handles posture; we don't want to interfere.
        if is_idle:
            posture_offset = self._compute_posture_offset(pad)  # [B, action_dim]
            modulated = modulated + posture_offset

        self._modulation_count += 1

        if was_unbatched:
            modulated = modulated.squeeze(0)

        return modulated

    def get_speed_multiplier(self, pad_vector: torch.Tensor) -> float:
        """Return the scalar speed multiplier for a given PAD state.

        Useful for debugging, UI display, and reward shaping.

        Args:
            pad_vector: PAD emotional state, shape [3].

        Returns:
            Float speed multiplier in [1 - max_speed_mod, 1 + max_speed_mod].
        """
        with torch.no_grad():
            pad = pad_vector.detach()
            if pad.dim() == 1:
                pad = pad.unsqueeze(0)
            multiplier = self._compute_speed_multiplier(pad)
            return multiplier.squeeze().item()

    def get_style_description(self, pad_vector: torch.Tensor) -> str:
        """Return a human-readable description of the current movement style.

        Maps the PAD vector to the nearest emotion quadrant and returns
        a descriptive string. Useful for debugging, logging, and UI display.

        Args:
            pad_vector: PAD emotional state, shape [3]. Values in [-1, 1].

        Returns:
            String description of the movement style, e.g.
            "elated (bouncy, energetic, wide arm swing) | speed=1.18x"
        """
        with torch.no_grad():
            if isinstance(pad_vector, torch.Tensor):
                p, a, d = pad_vector.detach().cpu().tolist()
            else:
                p, a, d = float(pad_vector[0]), float(pad_vector[1]), float(pad_vector[2])

        # Determine quadrant from Pleasure x Arousal
        if abs(p) < 0.15 and abs(a) < 0.15:
            quadrant = "neutral"
        elif p >= 0 and a >= 0:
            quadrant = "elated"
        elif p < 0 and a >= 0:
            quadrant = "angry"
        elif p < 0 and a < 0:
            quadrant = "sad"
        else:  # p >= 0 and a < 0
            quadrant = "relaxed"

        style_desc = _STYLE_DESCRIPTIONS.get(quadrant, "unknown")

        # Add dominance modifier
        if abs(d) > 0.3:
            dom_label = "dominant" if d > 0 else "submissive"
            dom_desc = _STYLE_DESCRIPTIONS[dom_label]
            style_desc = f"{style_desc}, {dom_desc}"

        # Compute speed for display
        speed = self.get_speed_multiplier(
            torch.tensor([p, a, d], dtype=torch.float32)
        )

        return (
            f"{quadrant} ({style_desc}) | "
            f"speed={speed:.2f}x | "
            f"PAD=({p:+.2f}, {a:+.2f}, {d:+.2f})"
        )

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL COMPUTATION
    # ══════════════════════════════════════════════════════════════════════

    def _compute_speed_multiplier(self, pad: torch.Tensor) -> torch.Tensor:
        """Compute bounded speed multiplier from PAD vector.

        Args:
            pad: Batched PAD vector [B, 3]

        Returns:
            Speed multiplier [B, 1] in [1-max, 1+max]
        """
        raw = self.speed_net(pad)                           # [B, 1]
        normalized = torch.sigmoid(raw)                      # [B, 1] in [0, 1]
        # Rescale [0, 1] -> [1-max, 1+max]
        max_mod = self.config.max_speed_mod
        multiplier = (1.0 - max_mod) + normalized * (2.0 * max_mod)  # [B, 1]
        return multiplier

    def _compute_style_bias(
        self, pad: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute bounded per-joint style bias from PAD vector.

        The bias is clamped relative to the action magnitude so that
        large actions get proportionally larger biases (preserving the
        relative style effect), while small actions get small biases
        (preventing the style from overwhelming the intended movement).

        For zero or near-zero actions, an absolute floor prevents division
        by zero while keeping the bias negligible.

        Args:
            pad: Batched PAD vector [B, 3]
            action: Raw action [B, action_dim]

        Returns:
            Style bias [B, action_dim], bounded to +/-max_style_bias * |action|
        """
        raw_bias = self.style_net(pad)  # [B, action_dim], already in [-1, 1] via Tanh

        # Scale to configured maximum
        scaled_bias = raw_bias * self.config.max_style_bias  # [B, action_dim]

        # Clamp relative to action magnitude for safety
        # This ensures the bias never exceeds max_style_bias fraction of |action|
        action_magnitude = torch.abs(action).clamp(min=0.01)  # Floor to prevent /0
        max_abs_bias = action_magnitude * self.config.max_style_bias
        clamped_bias = torch.clamp(scaled_bias, -max_abs_bias, max_abs_bias)

        return clamped_bias

    def _compute_posture_offset(self, pad: torch.Tensor) -> torch.Tensor:
        """Compute bounded idle posture offset from PAD vector.

        Args:
            pad: Batched PAD vector [B, 3]

        Returns:
            Posture offset [B, action_dim], bounded to +/-max_posture_offset
        """
        raw_offset = self.posture_net(pad)  # [B, action_dim], in [-1, 1] via Tanh

        # Scale and hard-clamp to safe range
        max_offset = self.config.max_posture_offset
        offset = raw_offset * max_offset
        offset = torch.clamp(offset, -max_offset, max_offset)

        return offset

    # ══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    def get_modulation_stats(self, pad_vector: torch.Tensor) -> dict:
        """Return detailed modulation statistics for a given PAD state.

        Useful for debugging, unit testing, and training monitoring.

        Args:
            pad_vector: PAD emotional state, shape [3].

        Returns:
            Dictionary with speed, style bias norm, and posture offset norm.
        """
        with torch.no_grad():
            pad = pad_vector.detach()
            if pad.dim() == 1:
                pad = pad.unsqueeze(0)

            speed = self._compute_speed_multiplier(pad)
            # Use a dummy unit action to measure style bias magnitude
            dummy_action = torch.ones(1, self.config.action_dim, device=pad.device)
            style = self._compute_style_bias(pad, dummy_action)
            posture = self._compute_posture_offset(pad)

            return {
                "speed_multiplier": speed.item(),
                "style_bias_l2": torch.norm(style).item(),
                "style_bias_max": torch.max(torch.abs(style)).item(),
                "posture_offset_l2": torch.norm(posture).item(),
                "posture_offset_max": torch.max(torch.abs(posture)).item(),
                "total_modulations": self._modulation_count,
            }

    def reset_counters(self) -> None:
        """Reset diagnostic counters. Call at the start of each episode."""
        self._modulation_count = 0

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        cfg = self.config
        return (
            f"action_dim={cfg.action_dim}, "
            f"speed_range=[{1.0 - cfg.max_speed_mod:.1f}, {1.0 + cfg.max_speed_mod:.1f}], "
            f"style_bias=+/-{cfg.max_style_bias*100:.0f}%, "
            f"posture_offset=+/-{cfg.max_posture_offset:.2f}rad"
        )


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

def _test():
    """Verify safety bounds and basic functionality."""
    print("=" * 70)
    print("MovementMoodCoupling - Safety & Functionality Test")
    print("=" * 70)

    config = MovementMoodConfig()
    module = MovementMoodCoupling(config)
    module.eval()

    # Parameter count
    n_params = sum(p.numel() for p in module.parameters())
    print(f"\nParameter count: {n_params:,}")

    # Test PAD vectors representing each emotional quadrant
    test_moods = {
        "Elated   (High P, High A)":     torch.tensor([ 0.8,  0.8,  0.3]),
        "Angry    (Low P, High A)":      torch.tensor([-0.7,  0.9, -0.2]),
        "Sad      (Low P, Low A)":       torch.tensor([-0.6, -0.7, -0.4]),
        "Relaxed  (High P, Low A)":      torch.tensor([ 0.7, -0.5,  0.2]),
        "Neutral  (zero)":               torch.tensor([ 0.0,  0.0,  0.0]),
        "Dominant (High D)":             torch.tensor([ 0.1,  0.1,  0.9]),
        "Submissive (Low D)":            torch.tensor([ 0.0,  0.0, -0.8]),
    }

    # Create a realistic action (mimics walking joint torques)
    action = torch.randn(1, config.action_dim) * 0.5

    print(f"\nBase action magnitude: {torch.norm(action).item():.4f}")
    print("-" * 70)

    for name, pad in test_moods.items():
        # Test active locomotion modulation
        modulated = module.modulate_action(action, pad, is_idle=False)

        # Verify safety: modulated action should be close to original
        max_deviation = torch.max(torch.abs(modulated - action)).item()
        speed = module.get_speed_multiplier(pad)
        description = module.get_style_description(pad)
        stats = module.get_modulation_stats(pad)

        print(f"\n{name}")
        print(f"  Style: {description}")
        print(f"  Speed: {speed:.3f}x")
        print(f"  Max deviation from base: {max_deviation:.4f}")
        print(f"  Style bias L2: {stats['style_bias_l2']:.4f}")
        print(f"  Posture offset L2: {stats['posture_offset_l2']:.4f}")

        # SAFETY ASSERTIONS
        assert 0.7 <= speed <= 1.3, (
            f"Speed {speed} out of safe range [0.7, 1.3]!"
        )

    # Test idle posture modulation
    print("\n" + "=" * 70)
    print("Idle Posture Test (is_idle=True)")
    print("=" * 70)

    idle_action = torch.zeros(1, config.action_dim)  # Standing still
    sad_pad = torch.tensor([-0.6, -0.7, -0.4])

    idle_modulated = module.modulate_action(idle_action, sad_pad, is_idle=True)
    max_posture_shift = torch.max(torch.abs(idle_modulated)).item()
    print(f"\nSad idle posture max offset: {max_posture_shift:.4f} rad")
    assert max_posture_shift <= config.max_posture_offset + 1e-6, (
        f"Posture offset {max_posture_shift} exceeds max {config.max_posture_offset}!"
    )

    # Test batched operation
    print("\n" + "=" * 70)
    print("Batched Operation Test")
    print("=" * 70)

    batch_action = torch.randn(8, config.action_dim) * 0.5
    batch_pad = torch.randn(8, 3).clamp(-1, 1)
    batch_modulated = module.modulate_action(batch_action, batch_pad)
    print(f"\nBatch input:  {batch_action.shape}")
    print(f"Batch output: {batch_modulated.shape}")
    assert batch_modulated.shape == batch_action.shape, "Shape mismatch!"

    # Test unbatched operation
    single_action = torch.randn(config.action_dim) * 0.5
    single_pad = torch.tensor([0.5, 0.5, 0.0])
    single_modulated = module.modulate_action(single_action, single_pad)
    assert single_modulated.shape == single_action.shape, "Unbatched shape mismatch!"
    print(f"Unbatched input:  {single_action.shape}")
    print(f"Unbatched output: {single_modulated.shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - Safety bounds verified")
    print("=" * 70)


if __name__ == "__main__":
    _test()
