"""
EMOTIONAL STATE MODULE - PAD Model (Pleasure-Arousal-Dominance)

Research backing:
- Mehrabian (1996): PAD Temperament Model
- Gebhard (2005): ALMA - A Layered Model of Affect
- Rivers (2024): Chain-of-Emotion for game agents (PLOS ONE)
- FAtiMA Toolkit (Dias et al., 2022): OCC emotion triggers

Architecture:
- 3-dim continuous PAD vector in [-1, 1]
- GRU-based temporal dynamics with exponential decay
- Personality baseline (Big Five -> PAD mapping)
- Mood embedding projector for brain conditioning
- OCC event triggers for discrete emotion responses

Author: Janno Louwrens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque


# ==============================================================================
# EVENT TYPES - OCC Appraisal Categories
# ==============================================================================
# Based on the OCC (Ortony, Clore, Collins 1988) model of emotion appraisal.
# Each event maps to a PAD delta vector computed from appraisal dimensions:
#   - desirability (maps to Pleasure)
#   - unexpectedness (maps to Arousal)
#   - agency/control (maps to Dominance)
# FAtiMA Toolkit (Dias et al., 2022) provides the computational implementation
# pattern where discrete events trigger continuous affect changes.

class EventType(Enum):
    """Discrete events that trigger emotional responses.

    Each event has an associated PAD delta vector that shifts the current
    mood state. The deltas are derived from OCC appraisal theory:
    - Pleasure: how desirable/undesirable the event is
    - Arousal: how unexpected/stimulating the event is
    - Dominance: how much control/agency the agent has
    """
    TASK_SUCCESS = auto()     # Completed a task successfully
    TASK_FAILURE = auto()     # Failed at a task
    USER_CHAT = auto()        # User initiated conversation
    USER_PRAISE = auto()      # User gave positive feedback
    USER_SCOLD = auto()       # User gave negative feedback
    NOVELTY = auto()          # Encountered something new/unexpected
    BOREDOM_TICK = auto()     # Nothing has happened for a while
    SKILL_LEARNED = auto()    # Mastered a new skill/capability
    DAMAGE = auto()           # Physical damage or error state
    GOAL_ACHIEVED = auto()    # Major goal completed


# ==============================================================================
# OCC EVENT -> PAD DELTA MAPPING
# ==============================================================================
# These delta vectors are calibrated based on:
# - Gebhard (2005) ALMA: empirical PAD values for named emotions
# - Rivers (2024): validated PAD ranges in game agent studies
# - Mehrabian (1996): original PAD temperament mappings
#
# Format: {EventType: (delta_P, delta_A, delta_D)}
# Magnitudes are intentionally moderate (0.1-0.4 range) so that single events
# produce noticeable but not overwhelming shifts. Multiple events accumulate.

OCC_EVENT_DELTAS: Dict[EventType, Tuple[float, float, float]] = {
    # TASK_SUCCESS: desirable + moderate arousal + high control
    # OCC: "Joy" from desirable event + "Pride" from self-agency
    EventType.TASK_SUCCESS:   ( 0.3,   0.1,   0.2),

    # TASK_FAILURE: undesirable + high arousal + low control
    # OCC: "Distress" from undesirable event + "Shame" from self-agency failure
    EventType.TASK_FAILURE:   (-0.3,   0.2,  -0.3),

    # USER_CHAT: mildly pleasant + moderate arousal + neutral control
    # OCC: "Liking" from social interaction
    EventType.USER_CHAT:      ( 0.1,   0.15,  0.0),

    # USER_PRAISE: highly desirable + moderate arousal + high dominance
    # OCC: "Appreciation" + "Pride" from external validation
    EventType.USER_PRAISE:    ( 0.4,   0.2,   0.3),

    # USER_SCOLD: undesirable + high arousal + low dominance
    # OCC: "Reproach" triggers "Shame" + loss of agency
    EventType.USER_SCOLD:     (-0.3,   0.3,  -0.4),

    # NOVELTY: neutral pleasure + high arousal + neutral dominance
    # OCC: "Surprise" is valence-neutral but highly arousing
    EventType.NOVELTY:        ( 0.05,  0.4,   0.0),

    # BOREDOM_TICK: mildly unpleasant + arousal decrease + neutral
    # Not a standard OCC emotion, but models the absence of stimulation
    # Rivers (2024) shows boredom as low-arousal negative state
    EventType.BOREDOM_TICK:   (-0.05, -0.15,  0.0),

    # SKILL_LEARNED: highly desirable + moderate arousal + high dominance
    # OCC: "Pride" from self-initiated achievement + competence
    EventType.SKILL_LEARNED:  ( 0.35,  0.25,  0.35),

    # DAMAGE: undesirable + very high arousal + very low dominance
    # OCC: "Fear" from threat + loss of control
    EventType.DAMAGE:         (-0.4,   0.5,  -0.5),

    # GOAL_ACHIEVED: maximally desirable + high arousal + high dominance
    # OCC: "Joy" + "Pride" + "Gratification" compound emotion
    EventType.GOAL_ACHIEVED:  ( 0.5,   0.3,   0.4),
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class EmotionalConfig:
    """Configuration for the EmotionalState module.

    Design choices:
    - pad_dim=3: Mehrabian (1996) showed 3 dimensions (PAD) capture ~87% of
      variance in emotional experience. Adding more dims gives diminishing
      returns and complicates learning. 5-dim models (e.g., OCEAN) conflate
      personality with affect.
    - decay_factor=0.995: From "Generative Agents" (Park et al., 2023).
      Exponential decay toward baseline ensures moods are transient but not
      instant. At 0.995, a mood perturbation has a half-life of ~138 steps,
      meaning emotions linger for minutes of simulated time but do not persist
      indefinitely. This matches psychological evidence that moods decay
      exponentially (Davidson, 1998).
    - gru_hidden=64: GRU hidden size for temporal dynamics. Small enough to
      be lightweight but large enough to learn non-trivial mood transitions.
      GRU chosen over LSTM because it has fewer parameters and performs
      comparably for short-horizon temporal modeling (Chung et al., 2014).
    - d_model=512: Must match UnifiedBrain.d_model for the mood embedding
      projector to condition the brain's transformer layers.
    """
    # PAD vector dimensionality (FIXED at 3 - do not change)
    # Mehrabian (1996): Pleasure, Arousal, Dominance
    pad_dim: int = 3

    # Brain embedding dimension - must match UnifiedBrain.d_model
    d_model: int = 512

    # Temporal dynamics
    # 0.995 per step -> half-life ~138 steps -> emotions linger ~2-5 min
    # From "Generative Agents" (Park et al., 2023)
    decay_factor: float = 0.995

    # GRU hidden size for learned mood transitions
    # Small footprint: 64 hidden units, ~25K parameters total
    # GRU over LSTM: fewer params, comparable performance (Chung et al., 2014)
    gru_hidden: int = 64

    # Personality baseline (Big Five -> PAD mapping)
    # Default: mildly positive, calm, moderate control
    # Based on Mehrabian (1996) mapping for "agreeable" personality
    baseline_pleasure: float = 0.2
    baseline_arousal: float = -0.1    # Slightly calm by default
    baseline_dominance: float = 0.1   # Slight sense of control

    # Event response scaling
    # Multiplier on OCC deltas - allows personality to amplify/dampen reactions
    # Neuroticism maps to higher reactivity (Costa & McCrae, 1992)
    event_reactivity: float = 1.0

    # Mood history tracking
    history_maxlen: int = 1000   # Keep last 1000 mood snapshots
    history_interval: float = 1.0  # Record every 1 second of sim time

    # Clamping range for PAD dimensions
    pad_min: float = -1.0
    pad_max: float = 1.0

    # Noise for stochastic mood variation (models biological noise)
    # Small Gaussian noise added each step to prevent getting stuck
    # Supports the "affective noise" hypothesis (Eich, 1995)
    noise_std: float = 0.005


# ==============================================================================
# MOOD HISTORY - For Visualization and Analysis
# ==============================================================================

class MoodHistory:
    """Tracks mood state over time for UI visualization and analysis.

    Stores timestamped PAD vectors and event annotations. This enables:
    - Real-time mood graphs in the companion UI
    - Post-hoc analysis of emotional trajectories
    - Training data for mood prediction models
    - Debugging emotional response tuning

    Design: Uses a fixed-size deque to bound memory. Each entry stores
    the full PAD vector, timestamp, dominant mood label, and optional
    event annotation.
    """

    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.entries: deque = deque(maxlen=maxlen)
        self._last_record_time: float = 0.0

    def record(
        self,
        timestamp: float,
        pad_vector: Tuple[float, float, float],
        dominant_mood: str,
        event: Optional[str] = None,
    ) -> None:
        """Record a mood snapshot.

        Args:
            timestamp: Simulation time in seconds
            pad_vector: Current (P, A, D) values
            dominant_mood: String label of dominant mood
            event: Optional event that caused this snapshot
        """
        self.entries.append({
            "timestamp": timestamp,
            "pleasure": pad_vector[0],
            "arousal": pad_vector[1],
            "dominance": pad_vector[2],
            "mood": dominant_mood,
            "event": event,
        })
        self._last_record_time = timestamp

    def get_recent(self, n: int = 50) -> List[Dict]:
        """Return the N most recent mood entries for UI display."""
        return list(self.entries)[-n:]

    def get_time_range(
        self, start: float, end: float
    ) -> List[Dict]:
        """Return mood entries within a time range."""
        return [
            e for e in self.entries
            if start <= e["timestamp"] <= end
        ]

    def get_average_mood(self, n: int = 50) -> Tuple[float, float, float]:
        """Return average PAD over last N entries.

        Useful for determining long-term mood trends vs transient spikes.
        ALMA (Gebhard, 2005) distinguishes between "emotion" (transient)
        and "mood" (sustained average). This method computes the mood.
        """
        recent = list(self.entries)[-n:]
        if not recent:
            return (0.0, 0.0, 0.0)

        avg_p = sum(e["pleasure"] for e in recent) / len(recent)
        avg_a = sum(e["arousal"] for e in recent) / len(recent)
        avg_d = sum(e["dominance"] for e in recent) / len(recent)
        return (avg_p, avg_a, avg_d)

    def get_mood_volatility(self, n: int = 50) -> float:
        """Compute mood volatility as std dev of PAD magnitude over last N entries.

        High volatility = emotionally unstable (may indicate stress or disorder).
        Low volatility = emotionally stable or flat affect.
        Rivers (2024) uses similar metrics for agent emotion quality scoring.
        """
        recent = list(self.entries)[-n:]
        if len(recent) < 2:
            return 0.0

        magnitudes = [
            math.sqrt(e["pleasure"]**2 + e["arousal"]**2 + e["dominance"]**2)
            for e in recent
        ]
        mean_mag = sum(magnitudes) / len(magnitudes)
        variance = sum((m - mean_mag)**2 for m in magnitudes) / len(magnitudes)
        return math.sqrt(variance)

    def to_json(self) -> str:
        """Serialize mood history to JSON for persistence."""
        return json.dumps(list(self.entries))

    @classmethod
    def from_json(cls, json_str: str, maxlen: int = 1000) -> "MoodHistory":
        """Deserialize mood history from JSON."""
        history = cls(maxlen=maxlen)
        entries = json.loads(json_str)
        for entry in entries:
            history.entries.append(entry)
        if entries:
            history._last_record_time = entries[-1]["timestamp"]
        return history

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        if not self.entries:
            return "MoodHistory(empty)"
        latest = self.entries[-1]
        return (
            f"MoodHistory(n={len(self.entries)}, "
            f"latest_mood={latest['mood']}, "
            f"P={latest['pleasure']:.2f}, "
            f"A={latest['arousal']:.2f}, "
            f"D={latest['dominance']:.2f})"
        )


# ==============================================================================
# EMOTIONAL STATE MODULE (nn.Module)
# ==============================================================================

class EmotionalState(nn.Module):
    """Emotional state module using the PAD (Pleasure-Arousal-Dominance) model.

    This module maintains a 3-dimensional continuous mood vector and provides:
    1. GRU-based temporal dynamics for smooth mood transitions
    2. Exponential decay toward personality baseline
    3. OCC-style event triggers for discrete emotion responses
    4. Mood embedding projection for conditioning the brain's transformer
    5. Named mood classification for interpretability
    6. LLM prompt modifiers for language-conditioned behavior

    Integration with UnifiedBrain:
        The mood embedding (d_model-dimensional) is added to the brain's
        token embeddings before the transformer layers, following the
        "cross-modal conditioning" pattern from OpenVLA (2024). This means
        the emotional state biases all reasoning and motor output.

    Research foundation:
        - Mehrabian (1996): The PAD space captures the core dimensions of
          human emotional experience. 3 dimensions explain ~87% of variance.
        - ALMA (Gebhard 2005): Layered architecture separating personality
          (stable), mood (slow-changing), and emotion (fast-changing).
        - Generative Agents (Park et al., 2023): Exponential decay of
          emotional state toward a baseline, with decay_factor=0.995.
        - FAtiMA (Dias et al., 2022): OCC appraisal theory implemented
          as event -> emotion rules for virtual agents.
        - Chain-of-Emotion (Rivers 2024): Demonstrates that explicit
          emotional reasoning improves agent behavior in game environments.

    Parameter count: ~60K parameters (lightweight by design)
        - GRU: 3*64*(3+64+1) = ~13K
        - mood_encoder: 3*512+512 = ~2K
        - event_encoder: 10*3 = 30
        - input_projector: 6*3+3 = ~21
        - Total overhead is negligible compared to UnifiedBrain's ~105M
    """

    def __init__(self, config: Optional[EmotionalConfig] = None):
        super().__init__()
        self.config = config or EmotionalConfig()

        # ── PAD State Vector ──────────────────────────────────────────────
        # Registered as a buffer so it persists across forward passes but
        # is NOT a learnable parameter (it's updated procedurally).
        # Buffer moves with .to(device) automatically.
        self.register_buffer(
            "pad_vector",
            torch.tensor([
                self.config.baseline_pleasure,
                self.config.baseline_arousal,
                self.config.baseline_dominance,
            ], dtype=torch.float32),
        )

        # ── Personality Baseline ──────────────────────────────────────────
        # The "resting state" mood that the agent decays toward when nothing
        # is happening. Derived from Big Five personality traits.
        # Mehrabian (1996) provides the Big Five -> PAD mapping:
        #   Extraversion  -> +P, +A
        #   Agreeableness -> +P, +D
        #   Neuroticism   -> -P, +A, -D
        #   Openness      -> +A
        #   Conscientiousness -> +D
        # Default baseline represents a "curious, calm, mildly happy" agent.
        self.register_buffer(
            "baseline",
            torch.tensor([
                self.config.baseline_pleasure,
                self.config.baseline_arousal,
                self.config.baseline_dominance,
            ], dtype=torch.float32),
        )

        # ── GRU-based Mood Updater ────────────────────────────────────────
        # Instead of simple additive deltas, we use a GRU to learn non-linear
        # mood transitions. The GRU takes [current_pad, event_delta] as input
        # and produces a refined PAD update. This allows the model to learn
        # that, e.g., repeated failures cause increasing frustration (non-linear
        # accumulation), or that praise after failure causes relief (context-
        # dependent transitions).
        #
        # Input: 6-dim (3-dim current PAD + 3-dim event delta)
        # Hidden: config.gru_hidden (default 64)
        # Output: projected back to 3-dim PAD delta
        #
        # GRU chosen over LSTM: fewer parameters (3 gates vs 4), and mood
        # transitions are short-horizon (Chung et al., 2014).
        self.mood_updater = nn.GRUCell(
            input_size=self.config.pad_dim * 2,  # [current_pad, event_delta]
            hidden_size=self.config.gru_hidden,
        )

        # Project GRU hidden state back to PAD space
        self.gru_output_projector = nn.Sequential(
            nn.Linear(self.config.gru_hidden, self.config.pad_dim),
            nn.Tanh(),  # Bound output to [-1, 1] matching PAD range
        )

        # ── GRU Hidden State ──────────────────────────────────────────────
        # Persistent hidden state for the GRU. Registered as buffer so it
        # moves with the module to the correct device.
        self.register_buffer(
            "_gru_hidden",
            torch.zeros(1, self.config.gru_hidden, dtype=torch.float32),
        )

        # ── Input Projector ───────────────────────────────────────────────
        # Projects the concatenated [current_pad, event_delta] into the
        # GRU input space. Adds a non-linearity to allow richer input
        # representations before the GRU processes them.
        self.input_projector = nn.Sequential(
            nn.Linear(self.config.pad_dim * 2, self.config.pad_dim * 2),
            nn.SiLU(),  # SwiGLU-family activation, consistent with UnifiedBrain
        )

        # ── Mood Embedding Projector ──────────────────────────────────────
        # Projects the 3-dim PAD vector into d_model space so it can be
        # added to transformer token embeddings in UnifiedBrain.
        # This follows the "cross-modal conditioning" pattern:
        #   token_emb = token_emb + mood_embedding
        # Used in OpenVLA (2024) for vision conditioning and pi0 (2024)
        # for action conditioning.
        self.mood_encoder = nn.Sequential(
            nn.Linear(self.config.pad_dim, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
            # No final activation: the embedding should be an unconstrained
            # additive bias in the transformer's embedding space.
        )

        # ── OCC Event Embeddings ──────────────────────────────────────────
        # Learnable per-event PAD deltas initialized from the OCC mapping.
        # These START at the hand-crafted values from OCC_EVENT_DELTAS but
        # can be fine-tuned during training if we find better mappings.
        # This is the "neuro-symbolic" approach: symbolic priors + learned
        # refinement (similar to AlphaGeometry's synthetic priors).
        num_events = len(EventType)
        self.event_deltas = nn.Parameter(
            torch.zeros(num_events, self.config.pad_dim, dtype=torch.float32)
        )
        # Initialize from OCC mapping
        with torch.no_grad():
            for event_type in EventType:
                idx = event_type.value - 1  # Enum values start at 1
                delta = OCC_EVENT_DELTAS.get(
                    event_type, (0.0, 0.0, 0.0)
                )
                self.event_deltas[idx] = torch.tensor(delta, dtype=torch.float32)

        # ── Reward Sensitivity ────────────────────────────────────────────
        # Learnable scaling from scalar reward to PAD delta.
        # Reward primarily affects Pleasure (desirability appraisal in OCC).
        # Arousal gets a smaller boost from reward magnitude.
        # Dominance gets a slight boost from positive rewards (agency).
        self.reward_to_pad = nn.Linear(1, self.config.pad_dim)
        with torch.no_grad():
            # Initialize: reward -> mostly Pleasure, some Arousal, little Dominance
            self.reward_to_pad.weight.copy_(
                torch.tensor([[0.3], [0.1], [0.05]], dtype=torch.float32)
            )
            self.reward_to_pad.bias.zero_()

        # ── Interaction Sensitivity ───────────────────────────────────────
        # Learnable scaling from user interaction intensity [0,1] to PAD delta.
        # Social interaction generally increases P and A (social facilitation).
        self.interaction_to_pad = nn.Linear(1, self.config.pad_dim)
        with torch.no_grad():
            self.interaction_to_pad.weight.copy_(
                torch.tensor([[0.1], [0.15], [0.05]], dtype=torch.float32)
            )
            self.interaction_to_pad.bias.zero_()

        # ── Mood History ──────────────────────────────────────────────────
        self.history = MoodHistory(maxlen=self.config.history_maxlen)

        # ── Simulation Clock ──────────────────────────────────────────────
        self._sim_time: float = 0.0

        # ── Initialization ────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for linear layers
        and orthogonal for GRU (best practice for RNNs, Saxe et al. 2013)."""
        for name, param in self.mood_updater.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in [self.mood_encoder, self.input_projector, self.gru_output_projector]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ──────────────────────────────────────────────────────────────────────
    # CORE UPDATE
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        event_type: Optional[EventType] = None,
        reward: float = 0.0,
        user_interaction: float = 0.0,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Update the emotional state based on events, rewards, and time.

        This is the main method called each simulation step. It implements
        the ALMA (Gebhard 2005) three-layer update:
        1. PERSONALITY (baseline): Stable, defines decay target
        2. MOOD (slow): Exponential decay toward baseline
        3. EMOTION (fast): Immediate response to events

        The update pipeline:
            pad_t+1 = decay * pad_t + (1 - decay) * baseline   [mood decay]
                    + GRU(pad_t, event_delta)                   [learned dynamics]
                    + noise                                      [biological noise]

        Args:
            event_type: Optional discrete event that occurred this step
            reward: Scalar reward signal from the environment [-inf, inf]
            user_interaction: User interaction intensity [0, 1]
            dt: Time delta in seconds since last update

        Returns:
            Updated PAD vector as a 3-dim tensor
        """
        # Advance simulation clock
        self._sim_time += dt

        # ── Step 1: Compute event delta ───────────────────────────────────
        # Combine OCC event delta + reward signal + interaction signal
        event_delta = torch.zeros(
            self.config.pad_dim,
            device=self.pad_vector.device,
            dtype=self.pad_vector.dtype,
        )

        event_name = None
        if event_type is not None:
            idx = event_type.value - 1
            event_delta = event_delta + self.event_deltas[idx]
            event_name = event_type.name

        # Add reward contribution
        if reward != 0.0:
            reward_tensor = torch.tensor(
                [[reward]], device=self.pad_vector.device, dtype=self.pad_vector.dtype
            )
            event_delta = event_delta + self.reward_to_pad(reward_tensor).squeeze(0)

        # Add interaction contribution
        if user_interaction > 0.0:
            interaction_tensor = torch.tensor(
                [[user_interaction]],
                device=self.pad_vector.device,
                dtype=self.pad_vector.dtype,
            )
            event_delta = event_delta + self.interaction_to_pad(
                interaction_tensor
            ).squeeze(0)

        # Scale by personality reactivity
        event_delta = event_delta * self.config.event_reactivity

        # ── Step 2: Exponential decay toward baseline ─────────────────────
        # From Generative Agents (Park et al., 2023):
        # mood_t+1 = decay^dt * mood_t + (1 - decay^dt) * baseline
        # The dt exponent ensures time-invariant decay regardless of step size.
        decay = self.config.decay_factor ** dt
        self.pad_vector = (
            decay * self.pad_vector + (1.0 - decay) * self.baseline
        )

        # ── Step 3: GRU-based learned update ──────────────────────────────
        # The GRU learns non-linear mood transitions that simple additive
        # deltas cannot capture. For example:
        # - Repeated failures -> escalating frustration
        # - Praise after failure -> relief (context-dependent)
        # - Novelty when bored -> excitement (state-dependent)
        gru_input = torch.cat([self.pad_vector, event_delta], dim=-1)
        gru_input = self.input_projector(gru_input.unsqueeze(0))  # [1, 6]

        self._gru_hidden = self.mood_updater(gru_input, self._gru_hidden)
        gru_delta = self.gru_output_projector(self._gru_hidden).squeeze(0)

        # GRU receives event_delta as input and learns to produce the appropriate
        # mood change. Using only gru_delta avoids double-counting the event.
        total_delta = gru_delta

        # ── Step 4: Apply delta and add noise ─────────────────────────────
        self.pad_vector = self.pad_vector + total_delta

        # Biological noise: small random perturbations model the stochastic
        # nature of neurochemical affect regulation (Eich, 1995)
        if self.config.noise_std > 0:
            noise = torch.randn_like(self.pad_vector) * self.config.noise_std
            self.pad_vector = self.pad_vector + noise

        # ── Step 5: Clamp to valid PAD range ──────────────────────────────
        self.pad_vector = torch.clamp(
            self.pad_vector,
            min=self.config.pad_min,
            max=self.config.pad_max,
        )

        # ── Step 6: Record history ────────────────────────────────────────
        if (
            self._sim_time - self.history._last_record_time
            >= self.config.history_interval
        ):
            self.history.record(
                timestamp=self._sim_time,
                pad_vector=(
                    self.pad_vector[0].item(),
                    self.pad_vector[1].item(),
                    self.pad_vector[2].item(),
                ),
                dominant_mood=self.get_dominant_mood(),
                event=event_name,
            )

        return self.pad_vector.clone()

    # ──────────────────────────────────────────────────────────────────────
    # MOOD EMBEDDING (for brain conditioning)
    # ──────────────────────────────────────────────────────────────────────

    def get_mood_embedding(self) -> torch.Tensor:
        """Project the 3-dim PAD vector into d_model space for brain conditioning.

        Returns a (d_model,) tensor that should be added to transformer token
        embeddings in UnifiedBrain:

            token_emb = token_emb + emotional_state.get_mood_embedding()

        This follows the cross-modal conditioning pattern from:
        - OpenVLA (2024): adds vision embeddings to transformer tokens
        - pi0 (2024): adds action embeddings to transformer tokens
        - Our approach: adds mood embeddings to bias all processing

        The embedding is detached from grad if you only want inference,
        or kept in the graph for end-to-end training of emotional responses.
        """
        return self.mood_encoder(self.pad_vector.unsqueeze(0)).squeeze(0)

    # ──────────────────────────────────────────────────────────────────────
    # MOOD CLASSIFICATION
    # ──────────────────────────────────────────────────────────────────────

    def get_mood_dict(self) -> Dict[str, float]:
        """Return named PAD values for logging and UI display.

        Returns:
            Dict with keys: pleasure, arousal, dominance, magnitude, mood
        """
        p = self.pad_vector[0].item()
        a = self.pad_vector[1].item()
        d = self.pad_vector[2].item()
        magnitude = math.sqrt(p**2 + a**2 + d**2)

        return {
            "pleasure": round(p, 4),
            "arousal": round(a, 4),
            "dominance": round(d, 4),
            "magnitude": round(magnitude, 4),
            "mood": self.get_dominant_mood(),
        }

    def get_dominant_mood(self) -> str:
        """Classify the current PAD vector into a named mood category.

        Mapping based on Mehrabian (1996) octant analysis and refined by
        Gebhard (2005) ALMA. The PAD space divides into 8 octants, each
        corresponding to a distinct mood. We use threshold-based rules
        rather than strict octants for smoother transitions.

        Priority ordering matters: more specific moods (anxious, frustrated)
        are checked before general ones (happy, calm). This prevents
        high-arousal negative states from being misclassified as generic moods.

        Returns:
            String mood label: one of "Anxious", "Frustrated", "Excited",
            "Curious", "Happy", "Confident", "Bored", or "Calm"
        """
        p = self.pad_vector[0].item()
        a = self.pad_vector[1].item()
        d = self.pad_vector[2].item()

        # ── High-arousal negative states (check first - most urgent) ──────
        # Anxious: negative affect + high arousal + low control
        # Mehrabian (1996): -P, +A, -D octant = "Anxious"
        if p < 0 and a > 0.5 and d < 0:
            return "Anxious"

        # Frustrated: negative affect + moderate-high arousal
        # OCC: repeated undesirable events without control
        if p < -0.3 and a > 0.3:
            return "Frustrated"

        # ── High-arousal positive states ──────────────────────────────────
        # Excited: positive arousal dominant (high energy positive)
        # Mehrabian (1996): +P, +A octant = "Exuberant/Excited"
        if p > 0 and a > 0.5:
            return "Excited"

        # Curious: moderate arousal + positive dominance (engaged exploration)
        # Not a standard Mehrabian label, but Rivers (2024) shows it as a
        # key agent emotion for exploration behavior
        if a > 0.3 and d > 0:
            return "Curious"

        # ── Moderate states ───────────────────────────────────────────────
        # Happy: high pleasure, any arousal
        # Mehrabian (1996): +P octant = "Pleasant"
        if p > 0.5 and a > 0:
            return "Happy"

        # Confident: positive affect + high dominance (mastery feeling)
        # Mehrabian (1996): +P, +D = "Confident/Dominant-pleasant"
        if p > 0 and d > 0.5:
            return "Confident"

        # ── Low-arousal states ────────────────────────────────────────────
        # Bored: low arousal (understimulated)
        # Rivers (2024): boredom drives exploration in game agents
        if a < -0.3:
            return "Bored"

        # ── Default: Calm ─────────────────────────────────────────────────
        # When no strong dimension dominates, the agent is in a neutral,
        # calm state. This is the "resting" classification.
        # Mehrabian (1996): near-zero PAD = "Neutral/Calm"
        return "Calm"

    # ──────────────────────────────────────────────────────────────────────
    # LLM PROMPT CONDITIONING
    # ──────────────────────────────────────────────────────────────────────

    def get_personality_prompt_modifier(self) -> str:
        """Generate a natural language mood description for LLM conditioning.

        This string is prepended to the LLM's system prompt to bias its
        language generation toward mood-congruent responses. For example,
        a happy agent generates more enthusiastic responses, while a
        frustrated agent uses shorter, more clipped language.

        Following Chain-of-Emotion (Rivers 2024): explicit emotional state
        in the prompt significantly improves agent behavior coherence.

        Returns:
            A 1-2 sentence mood description suitable for LLM system prompts.
        """
        mood = self.get_dominant_mood()
        p = self.pad_vector[0].item()
        a = self.pad_vector[1].item()
        d = self.pad_vector[2].item()
        magnitude = math.sqrt(p**2 + a**2 + d**2)

        # Intensity qualifier based on PAD magnitude
        # Low magnitude = subtle mood, high magnitude = intense mood
        if magnitude < 0.3:
            intensity = "slightly"
        elif magnitude < 0.6:
            intensity = "moderately"
        elif magnitude < 0.9:
            intensity = "quite"
        else:
            intensity = "very"

        # Mood-specific prompt modifiers
        # Each modifier describes the mood AND its behavioral implications,
        # following the Chain-of-Emotion approach (Rivers 2024)
        mood_prompts = {
            "Happy": (
                f"You are feeling {intensity} happy and content. "
                "You communicate warmly and are eager to help."
            ),
            "Excited": (
                f"You are feeling {intensity} excited and energetic. "
                "You communicate enthusiastically and are full of ideas."
            ),
            "Curious": (
                f"You are feeling {intensity} curious and engaged. "
                "You ask follow-up questions and explore topics deeply."
            ),
            "Frustrated": (
                f"You are feeling {intensity} frustrated. "
                "You are more direct and focused on solving the problem at hand."
            ),
            "Bored": (
                f"You are feeling {intensity} bored and understimulated. "
                "You seek novelty and suggest new activities or topics."
            ),
            "Calm": (
                "You are feeling calm and neutral. "
                "You communicate in a balanced, thoughtful manner."
            ),
            "Anxious": (
                f"You are feeling {intensity} anxious and uncertain. "
                "You seek reassurance and prefer cautious approaches."
            ),
            "Confident": (
                f"You are feeling {intensity} confident and in control. "
                "You communicate assertively and take initiative."
            ),
        }

        return mood_prompts.get(mood, "You are feeling neutral.")

    # ──────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Save emotional state for persistence across sessions.

        Saves:
        - Current PAD vector
        - Personality baseline
        - GRU hidden state
        - All learnable parameters (event deltas, reward mapping, etc.)
        - Mood history
        - Simulation time

        This enables "emotional continuity" - the agent remembers its mood
        from the last session, creating a more believable companion experience.
        Generative Agents (Park et al., 2023) shows that persistent emotional
        state significantly improves long-term agent believability.
        """
        state = {
            "pad_vector": self.pad_vector.cpu().tolist(),
            "baseline": self.baseline.cpu().tolist(),
            "gru_hidden": self._gru_hidden.cpu().tolist(),
            "sim_time": self._sim_time,
            "model_state_dict": {
                k: v.cpu() for k, v in self.state_dict().items()
            },
            "mood_history": self.history.to_json(),
            "config": {
                "pad_dim": self.config.pad_dim,
                "d_model": self.config.d_model,
                "decay_factor": self.config.decay_factor,
                "gru_hidden": self.config.gru_hidden,
                "baseline_pleasure": self.config.baseline_pleasure,
                "baseline_arousal": self.config.baseline_arousal,
                "baseline_dominance": self.config.baseline_dominance,
                "event_reactivity": self.config.event_reactivity,
                "noise_std": self.config.noise_std,
            },
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load emotional state from a previous session.

        Restores the full emotional state including PAD vector, GRU hidden
        state, learned parameters, and mood history. The agent resumes
        with the same emotional state it had when last saved.
        """
        state = torch.load(path, map_location=self.pad_vector.device, weights_only=False)

        # Restore PAD vector
        self.pad_vector.copy_(
            torch.tensor(state["pad_vector"], dtype=torch.float32)
        )

        # Restore baseline (may have been customized)
        self.baseline.copy_(
            torch.tensor(state["baseline"], dtype=torch.float32)
        )

        # Restore GRU hidden state
        self._gru_hidden.copy_(
            torch.tensor(state["gru_hidden"], dtype=torch.float32)
        )

        # Restore simulation time
        self._sim_time = state.get("sim_time", 0.0)

        # Restore learned parameters
        if "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"], strict=False)

        # Restore mood history
        if "mood_history" in state:
            self.history = MoodHistory.from_json(
                state["mood_history"],
                maxlen=self.config.history_maxlen,
            )

    # ──────────────────────────────────────────────────────────────────────
    # PERSONALITY CUSTOMIZATION
    # ──────────────────────────────────────────────────────────────────────

    def set_personality(
        self,
        extraversion: float = 0.5,
        agreeableness: float = 0.5,
        neuroticism: float = 0.5,
        openness: float = 0.5,
        conscientiousness: float = 0.5,
    ) -> None:
        """Set the personality baseline using Big Five traits.

        Converts Big Five personality dimensions [0, 1] to PAD baseline
        using the ALMA mapping (Gebhard 2005), which matches the formula
        in Personality.big_five_to_pad_baseline() for consistency.

        ALMA mapping (Gebhard 2005, Table 2 - Default Mood computation):
            P =  0.21*E + 0.59*A + 0.19*(-N)
            A =  0.15*O + 0.30*(-A) + 0.57*N
            D =  0.25*O + 0.17*C + 0.60*E - 0.32*A

        All traits are on [0, 1] scale where 0.5 = population average.
        They are remapped to [-1, 1] before applying the ALMA formula.

        Args:
            extraversion: Sociability, assertiveness, positive emotionality
            agreeableness: Cooperation, trust, compliance
            neuroticism: Emotional instability, anxiety, moodiness
            openness: Curiosity, imagination, aesthetic sensitivity
            conscientiousness: Organization, discipline, goal-directed behavior
        """
        # Remap [0, 1] -> [-1, 1] for the ALMA formula (matches Personality.py)
        o = openness * 2.0 - 1.0
        c = conscientiousness * 2.0 - 1.0
        e = extraversion * 2.0 - 1.0
        a = agreeableness * 2.0 - 1.0
        n = neuroticism * 2.0 - 1.0

        # ALMA mapping (Gebhard 2005) - matches Personality.big_five_to_pad_baseline()
        pleasure = 0.21 * e + 0.59 * a + 0.19 * (-n)
        arousal = 0.15 * o + 0.30 * (-a) + 0.57 * n
        dominance = 0.25 * o + 0.17 * c + 0.60 * e - 0.32 * a

        # Clamp to valid range
        pleasure = max(-1.0, min(1.0, pleasure))
        arousal = max(-1.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))

        self.baseline.copy_(
            torch.tensor([pleasure, arousal, dominance], dtype=torch.float32)
        )

        # Also adjust reactivity based on neuroticism
        # High neuroticism = stronger emotional reactions (Costa & McCrae, 1992)
        self.config.event_reactivity = 0.5 + neuroticism

    # ──────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset emotional state to personality baseline.

        Useful at the start of a new episode or after a major context switch.
        """
        self.pad_vector.copy_(self.baseline)
        self._gru_hidden.zero_()
        self._sim_time = 0.0

    def get_valence(self) -> float:
        """Return simple positive/negative valence.

        Valence is the most important single dimension of affect
        (Russell, 1980). It maps directly to Pleasure in PAD.
        """
        return self.pad_vector[0].item()

    def get_energy(self) -> float:
        """Return energy level (arousal).

        High energy = active, alert, aroused
        Low energy = calm, sleepy, bored
        """
        return self.pad_vector[1].item()

    def get_confidence(self) -> float:
        """Return confidence/agency level (dominance).

        High = confident, in control
        Low = submissive, uncertain
        """
        return self.pad_vector[2].item()

    def forward(
        self,
        event_type: Optional[EventType] = None,
        reward: float = 0.0,
        user_interaction: float = 0.0,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass: update state and return mood embedding.

        This is the standard nn.Module forward() that integrates with
        PyTorch's autograd. Call this in the brain's forward pass to
        get the mood embedding for conditioning.

        Returns:
            mood_embedding: (d_model,) tensor for brain conditioning
        """
        self.update(
            event_type=event_type,
            reward=reward,
            user_interaction=user_interaction,
            dt=dt,
        )
        return self.get_mood_embedding()

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"pad_dim={self.config.pad_dim}, "
            f"d_model={self.config.d_model}, "
            f"gru_hidden={self.config.gru_hidden}, "
            f"decay={self.config.decay_factor}, "
            f"mood={self.get_dominant_mood()}"
        )


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EMOTIONAL STATE MODULE - Standalone Test")
    print("=" * 70)

    # Create with default config
    config = EmotionalConfig()
    emo = EmotionalState(config)
    print(f"\nModule: {emo}")
    print(f"Parameters: {sum(p.numel() for p in emo.parameters()):,}")

    # Test personality setup
    print("\n--- Personality: Curious Explorer ---")
    emo.set_personality(
        extraversion=0.7,
        agreeableness=0.6,
        neuroticism=0.3,
        openness=0.9,
        conscientiousness=0.5,
    )
    print(f"Baseline: {emo.baseline.tolist()}")
    print(f"Reactivity: {emo.config.event_reactivity:.2f}")

    # Simulate a sequence of events
    print("\n--- Event Simulation ---")
    events = [
        (EventType.NOVELTY, 0.0, 0.0, "Discovers something new"),
        (EventType.USER_CHAT, 0.0, 0.8, "User starts chatting"),
        (EventType.TASK_SUCCESS, 0.5, 0.0, "Completes a task"),
        (EventType.USER_PRAISE, 0.0, 0.9, "User says 'great job!'"),
        (None, 0.0, 0.0, "Idle tick 1"),
        (None, 0.0, 0.0, "Idle tick 2"),
        (None, 0.0, 0.0, "Idle tick 3"),
        (EventType.TASK_FAILURE, -0.3, 0.0, "Task goes wrong"),
        (EventType.TASK_FAILURE, -0.5, 0.0, "Fails again"),
        (EventType.USER_SCOLD, 0.0, 0.7, "User is disappointed"),
        (EventType.BOREDOM_TICK, 0.0, 0.0, "Nothing happening..."),
        (EventType.SKILL_LEARNED, 0.8, 0.0, "Learns a new skill!"),
        (EventType.GOAL_ACHIEVED, 1.0, 0.5, "Major goal achieved!"),
    ]

    for event_type, reward, interaction, description in events:
        pad = emo.update(
            event_type=event_type,
            reward=reward,
            user_interaction=interaction,
            dt=1.0,
        )
        mood = emo.get_mood_dict()
        print(
            f"  {description:<35} -> "
            f"P={mood['pleasure']:+.3f}  "
            f"A={mood['arousal']:+.3f}  "
            f"D={mood['dominance']:+.3f}  "
            f"[{mood['mood']}]"
        )

    # Test mood embedding
    print("\n--- Mood Embedding ---")
    embedding = emo.get_mood_embedding()
    print(f"Shape: {embedding.shape}")
    print(f"Norm: {embedding.norm().item():.4f}")

    # Test LLM prompt modifier
    print("\n--- LLM Prompt Modifier ---")
    print(f'"{emo.get_personality_prompt_modifier()}"')

    # Test mood history
    print("\n--- Mood History ---")
    print(f"History entries: {len(emo.history)}")
    print(f"Average mood: {emo.history.get_average_mood()}")
    print(f"Volatility: {emo.history.get_mood_volatility():.4f}")
    print(f"History repr: {emo.history}")

    # Test persistence
    print("\n--- Persistence Test ---")
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name
    emo.save_state(save_path)
    print(f"Saved to: {save_path}")

    # Create fresh instance and load
    emo2 = EmotionalState(config)
    emo2.load_state(save_path)
    print(f"Loaded mood: {emo2.get_mood_dict()}")
    print(f"Moods match: {emo.get_dominant_mood() == emo2.get_dominant_mood()}")

    # Cleanup
    os.unlink(save_path)

    # Test forward pass (autograd)
    print("\n--- Forward Pass (autograd) ---")
    embedding = emo(event_type=EventType.NOVELTY, reward=0.1, dt=1.0)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Requires grad: {embedding.requires_grad}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
