"""
INNER MONOLOGUE - LLM-Based Autonomous Self-Talk

Research backing:
- Ahn et al. (2022): Inner Monologue - Embodied Reasoning through Planning with LLMs
- Ahn et al. (2022): SayCan - Do As I Can, Not As I Say (affordance grounding)
- Wang et al. (2023): Voyager - An Open-Ended Embodied Agent with LLMs
- Rivers (2024): Chain-of-Emotion appraisal architecture
- Yao et al. (2022): ReAct - Synergizing Reasoning and Acting

Architecture:
- Observe -> Appraise -> Plan -> Act -> Verify loop
- LLM generates chain-of-thought conditioned on mood + personality
- SayCan-style affordance grounding: P(skill) = P_LLM(skill) * P_affordance(skill)
- Thoughts stored in history for UI display and memory integration

Author: Janno Louwrens
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # Allow module to load without PyTorch for testing


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MonologueConfig:
    """Configuration for the inner monologue system.

    Attributes:
        cooldown_seconds: Minimum seconds between autonomous thoughts.
                          Prevents thought-flooding and keeps CPU/GPU usage sane.
        max_thought_length: Character limit for a single thought string.
                            Keeps UI display and memory footprint bounded.
        max_history: Rolling window of retained thoughts (oldest evicted first).
    """
    cooldown_seconds: float = 10.0
    max_thought_length: int = 100
    max_history: int = 50


# =============================================================================
# THOUGHT TYPES
# =============================================================================

THOUGHT_TYPES = (
    "observation",   # Perceiving the environment
    "appraisal",     # Emotional evaluation of a situation (Chain-of-Emotion)
    "plan",          # Next-step reasoning (ReAct / Inner Monologue)
    "reflection",    # Post-hoc outcome analysis (Voyager-style)
    "curiosity",     # Intrinsic-motivation-driven wondering
    "social",        # People-directed thoughts or empathy signals
)


# =============================================================================
# TEMPLATE BANK (fallback when no LLM is available)
# =============================================================================

_TEMPLATES: Dict[str, List[str]] = {
    "observation": [
        "I notice {detail} around me.",
        "Something changed - {detail}.",
        "The environment looks {detail}.",
        "I can sense {detail} nearby.",
        "Interesting - {detail} is different now.",
    ],
    "appraisal": [
        "This feels {mood_word}.",
        "I'm sensing a {mood_word} vibe about this.",
        "My current state is rather {mood_word}.",
        "Overall the mood is {mood_word}.",
        "Emotionally this situation is {mood_word}.",
    ],
    "plan": [
        "I should try to {goal} next.",
        "My plan is to {goal}.",
        "Next step: {goal}.",
        "Let me work on {goal}.",
        "Focusing on {goal} now.",
    ],
    "reflection": [
        "That went {outcome} - I {verb} at {task}.",
        "Reflecting: {task} was a {outcome}.",
        "Looking back, {task} turned out {outcome}.",
        "I {verb} the {task} task. Noted.",
        "Outcome for {task}: {outcome}.",
    ],
    "curiosity": [
        "I wonder what would happen if I explored more.",
        "There might be something interesting to discover.",
        "My curiosity is pulling me toward the unknown.",
        "What if I tried a different approach?",
        "I feel like exploring new possibilities.",
    ],
    "social": [
        "I wonder how the human is doing.",
        "Is there anything I can help with?",
        "It's nice to have company.",
        "I should be ready if someone needs me.",
        "I hope I'm being useful.",
    ],
}


def _mood_to_word(mood_dict: Dict[str, float]) -> str:
    """Convert a mood dictionary to a single descriptive word.

    The mood dict maps emotion names (e.g. 'valence', 'arousal', 'happy',
    'calm') to float intensities.  We pick the most dominant emotion name
    and fall back to a valence-based adjective if only valence/arousal are
    present.
    """
    if not mood_dict:
        return "neutral"

    # Common named emotions -> adjective mapping
    _adjectives = {
        "happy": "positive", "sad": "melancholy", "angry": "tense",
        "calm": "calm", "anxious": "uneasy", "curious": "curious",
        "excited": "energetic", "bored": "restless", "fear": "wary",
        "surprise": "surprised", "trust": "trusting", "disgust": "uneasy",
    }

    # If the mood dict uses named emotions, pick the dominant one
    named = {k: v for k, v in mood_dict.items()
             if k in _adjectives}
    if named:
        dominant = max(named, key=named.get)
        return _adjectives[dominant]

    # Fall back to valence
    valence = mood_dict.get("valence", 0.0)
    if valence > 0.3:
        return "positive"
    elif valence < -0.3:
        return "uneasy"
    return "neutral"


# =============================================================================
# INNER MONOLOGUE
# =============================================================================

class InnerMonologue:
    """LLM-powered autonomous self-talk for Jack.

    When an ``LLMEncoder`` (from ``UnifiedBrain``) is provided **and** its
    underlying causal-LM is loaded, thoughts are generated via prompted
    inference.  Otherwise the system degrades gracefully to a deterministic
    template engine -- useful for unit tests, headless training, and
    environments without GPU.

    Thread safety
    -------------
    Every public method acquires ``self._lock`` before mutating shared state
    (``thought_history``, ``last_thought_time``).  The game loop can call
    ``should_think`` / ``think`` from any thread.

    SayCan-style affordance grounding
    ----------------------------------
    ``think()`` accepts ``available_skills`` -- a mapping from skill name to
    a ``[0, 1]`` affordance probability.  The LLM proposes a skill, and the
    final score is::

        P(skill) = P_LLM(skill | context) * P_affordance(skill)

    This prevents the agent from planning actions it cannot physically
    execute (Ahn et al. 2022, *SayCan*).
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_encoder: Any = None,
        config: Optional[MonologueConfig] = None,
    ) -> None:
        self.config = config or MonologueConfig()

        # API LLM (Claude/GPT-4) - set externally after construction
        self._api_llm = None

        # Thread safety
        self._lock = threading.Lock()

        # Rolling history: (timestamp, thought_text, thought_type)
        self.thought_history: deque[Tuple[float, str, str]] = deque(
            maxlen=self.config.max_history,
        )
        self.last_thought_time: float = 0.0

        # LLM plumbing
        self._llm_encoder = llm_encoder
        self._use_llm: bool = False

        if llm_encoder is not None:
            # The LLMEncoder in UnifiedBrain exposes .llm (the causal-LM)
            # and .tokenizer when a HuggingFace backend was loaded
            # successfully.
            has_llm = (
                hasattr(llm_encoder, "llm")
                and llm_encoder.llm is not None
            )
            has_tok = (
                hasattr(llm_encoder, "tokenizer")
                and llm_encoder.tokenizer is not None
            )
            if has_llm and has_tok:
                self._use_llm = True
                print("[InnerMonologue] LLM backend active - rich thought generation enabled")
            else:
                print("[InnerMonologue] LLM encoder present but no causal-LM loaded - using templates")
        else:
            print("[InnerMonologue] No LLM encoder - using template-based thoughts")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def should_think(self, current_time: float) -> bool:
        """Return True if the cooldown has elapsed since the last thought.

        Args:
            current_time: Monotonic clock value (e.g. ``time.monotonic()``).
        """
        with self._lock:
            return (current_time - self.last_thought_time) >= self.config.cooldown_seconds

    def think(
        self,
        personality_prompt: str,
        mood_dict: Dict[str, float],
        recent_memories: List[str],
        current_goal: str,
        available_skills: Optional[Dict[str, float]] = None,
    ) -> str:
        """Generate an autonomous thought (the main entry point).

        Implements the *Observe -> Appraise -> Plan* loop from Inner
        Monologue (Ahn et al. 2022).  The thought is conditioned on:

        * ``personality_prompt`` -- a short string describing Jack's persona
        * ``mood_dict``         -- current emotional state (Chain-of-Emotion)
        * ``recent_memories``   -- short textual summaries of recent events
        * ``current_goal``      -- active goal from the AutonomousMind
        * ``available_skills``  -- affordance probabilities (SayCan grounding)

        Returns:
            The generated thought string (already stored in history).
            Empty string if cooldown has not elapsed (thread-safe).
        """
        # Atomic cooldown check to prevent TOCTOU race
        with self._lock:
            now = time.monotonic()
            if (now - self.last_thought_time) < self.config.cooldown_seconds:
                return ""

        # --- Build LLM prompt or pick a template category ---
        if self._use_llm:
            prompt = self._build_think_prompt(
                personality_prompt, mood_dict, recent_memories,
                current_goal, available_skills,
            )
            thought = self._generate_with_llm(prompt, max_tokens=50)
            thought_type = self._classify_thought(thought)
        else:
            thought, thought_type = self._template_think(
                mood_dict, current_goal, available_skills,
            )

        thought = self._clamp(thought)
        self._record(thought, thought_type)
        return thought

    def appraise(
        self,
        situation_description: str,
        mood_dict: Dict[str, float],
    ) -> str:
        """Emotional appraisal of a situation (Chain-of-Emotion, Rivers 2024).

        Evaluates the situation through the lens of the current mood and
        returns a short appraisal string.

        Args:
            situation_description: Free-text description of what is happening.
            mood_dict: Current emotional state.

        Returns:
            The appraisal thought string.
        """
        if self._use_llm:
            mood_summary = ", ".join(
                f"{k}={v:.2f}" for k, v in mood_dict.items()
            )
            prompt = (
                f"You are an embodied robot reflecting on a situation.\n"
                f"Current mood: {mood_summary}\n"
                f"Situation: {situation_description}\n"
                f"Give a brief emotional appraisal (one sentence):\n"
            )
            thought = self._generate_with_llm(prompt, max_tokens=40)
        else:
            mood_word = _mood_to_word(mood_dict)
            template = random.choice(_TEMPLATES["appraisal"])
            thought = template.format(mood_word=mood_word)

        thought = self._clamp(thought)
        self._record(thought, "appraisal")
        return thought

    def narrate_goal(
        self,
        goal_info: Dict[str, Any],
        mood_dict: Dict[str, float],
    ) -> str:
        """Narrate the current goal in first person.

        ``goal_info`` should contain at least a ``'description'`` key with a
        human-readable goal string.  Optional keys:

        * ``'strategy'`` -- the autotelic strategy that generated this goal
        * ``'progress'`` -- a ``[0, 1]`` completion estimate

        Args:
            goal_info: Goal metadata from the AutonomousMind.
            mood_dict: Current emotional state.

        Returns:
            The narration thought string.
        """
        description = goal_info.get("description", "explore the environment")
        strategy = goal_info.get("strategy", "")
        progress = goal_info.get("progress", 0.0)

        if self._use_llm:
            mood_summary = ", ".join(
                f"{k}={v:.2f}" for k, v in mood_dict.items()
            )
            prompt = (
                f"You are a goal-directed embodied robot.\n"
                f"Current mood: {mood_summary}\n"
                f"Goal: {description}\n"
                f"Strategy: {strategy}\n"
                f"Progress: {progress:.0%}\n"
                f"Briefly narrate your current objective (one sentence):\n"
            )
            thought = self._generate_with_llm(prompt, max_tokens=40)
        else:
            template = random.choice(_TEMPLATES["plan"])
            thought = template.format(goal=description)

        thought = self._clamp(thought)
        self._record(thought, "plan")
        return thought

    def reflect(
        self,
        task_name: str,
        success: bool,
        mood_change: Dict[str, float],
    ) -> str:
        """Post-hoc reflection on a completed task (Voyager-style).

        After a task finishes, the agent reflects on the outcome and how
        it affected its emotional state.  Successful reflections can be
        stored in long-term memory for skill reuse (Wang et al. 2023,
        *Voyager*).

        Args:
            task_name: Human-readable name of the completed task.
            success: Whether the task succeeded.
            mood_change: Delta to the mood dict caused by the outcome.

        Returns:
            The reflection thought string.
        """
        outcome = "success" if success else "failure"
        verb = "succeeded" if success else "failed"

        if self._use_llm:
            mood_delta = ", ".join(
                f"{k} {'+'if v >= 0 else ''}{v:.2f}" for k, v in mood_change.items()
            )
            prompt = (
                f"You are an embodied robot reflecting on a completed task.\n"
                f"Task: {task_name}\n"
                f"Outcome: {outcome}\n"
                f"Mood change: {mood_delta}\n"
                f"Give a brief reflection on what happened (one sentence):\n"
            )
            thought = self._generate_with_llm(prompt, max_tokens=40)
        else:
            template = random.choice(_TEMPLATES["reflection"])
            thought = template.format(
                task=task_name, outcome=outcome, verb=verb,
            )

        thought = self._clamp(thought)
        self._record(thought, "reflection")
        return thought

    def get_recent_thoughts(
        self,
        n: int = 5,
    ) -> List[Tuple[float, str, str]]:
        """Return the *n* most recent thoughts (newest first).

        Each element is ``(timestamp, thought_text, thought_type)``.
        """
        with self._lock:
            items = list(self.thought_history)
        # deque appends to the right, so newest is last
        return list(reversed(items[-n:]))

    # ------------------------------------------------------------------
    # LLM generation (wraps the existing LLMEncoder from UnifiedBrain)
    # ------------------------------------------------------------------

    def _generate_with_llm(self, prompt: str, max_tokens: int = 50) -> str:
        """Prompt the causal-LM and return the decoded continuation.

        Uses the same generation path as ``ResponseGenerator`` in
        ``UnifiedBrain.py`` -- tokenize, generate, decode -- but with
        parameters tuned for internal self-talk (lower temperature, shorter
        output).

        If an API LLM (Claude/GPT-4) is available, it is preferred over the
        local causal-LM for higher-quality thought generation.

        Falls back to a generic template string on *any* exception so that a
        GPU OOM or tokenizer hiccup never crashes the game loop.
        """
        # Prefer API LLM (Claude/GPT-4) for much richer thoughts
        if self._api_llm is not None and self._api_llm.available:
            try:
                result = self._api_llm.generate(
                    "You are Jack's inner thoughts. Think in first person, one sentence.",
                    prompt,
                    max_tokens=max_tokens,
                )
                if result:
                    return result
            except Exception as exc:
                print(f"[InnerMonologue] API LLM failed ({exc}), falling back to local LLM")

        try:
            encoder = self._llm_encoder
            inputs = encoder.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            device = next(encoder.llm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = encoder.llm.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=encoder.tokenizer.pad_token_id,
                )

            decoded = encoder.tokenizer.decode(
                outputs[0], skip_special_tokens=True,
            )

            # Strip the prompt echo that many causal LMs include
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            # Take only the first sentence / line
            for sep in ("\n", ". ", "! ", "? "):
                if sep in decoded:
                    decoded = decoded[: decoded.index(sep) + len(sep)]
                    break

            return decoded.strip()

        except Exception as exc:
            # Robust fallback -- never let LLM failures bubble up
            print(f"[InnerMonologue] LLM generation failed ({exc}), using fallback")
            return random.choice(_TEMPLATES["observation"]).format(
                detail="something in my surroundings",
            )

    # ------------------------------------------------------------------
    # template fallback (no LLM)
    # ------------------------------------------------------------------

    def _template_think(
        self,
        mood_dict: Dict[str, float],
        current_goal: str,
        available_skills: Optional[Dict[str, float]],
    ) -> Tuple[str, str]:
        """Pick a thought type stochastically and fill a template."""
        weights = {
            "observation": 0.25,
            "appraisal":   0.15,
            "plan":        0.25,
            "curiosity":   0.20,
            "social":      0.15,
        }

        # Bias toward planning when a concrete goal exists
        if current_goal:
            weights["plan"] += 0.15
            weights["curiosity"] -= 0.10

        types = list(weights.keys())
        probs = [weights[t] for t in types]
        total = sum(probs)
        probs = [p / total for p in probs]

        thought_type = random.choices(types, weights=probs, k=1)[0]

        if thought_type == "plan":
            goal_text = current_goal or "explore the environment"
            # SayCan-style: if skills are available, pick the most
            # affordable one to mention
            if available_skills:
                best_skill = max(available_skills, key=available_skills.get)
                goal_text = f"{goal_text} using {best_skill}"
            template = random.choice(_TEMPLATES["plan"])
            thought = template.format(goal=goal_text)

        elif thought_type == "appraisal":
            mood_word = _mood_to_word(mood_dict)
            template = random.choice(_TEMPLATES["appraisal"])
            thought = template.format(mood_word=mood_word)

        elif thought_type == "observation":
            template = random.choice(_TEMPLATES["observation"])
            thought = template.format(detail="something new")

        else:
            template = random.choice(_TEMPLATES[thought_type])
            thought = template

        return thought, thought_type

    # ------------------------------------------------------------------
    # prompt construction
    # ------------------------------------------------------------------

    def _build_think_prompt(
        self,
        personality_prompt: str,
        mood_dict: Dict[str, float],
        recent_memories: List[str],
        current_goal: str,
        available_skills: Optional[Dict[str, float]],
    ) -> str:
        """Assemble the chain-of-thought prompt for the LLM.

        Follows the ReAct pattern (Yao et al. 2022): the LLM is asked to
        produce a single thought step conditioned on observations, mood, and
        goals.  Available skills are listed so the model can ground its plan
        in what the body can actually do (SayCan).
        """
        mood_summary = ", ".join(
            f"{k}={v:.2f}" for k, v in mood_dict.items()
        ) if mood_dict else "neutral"

        memory_block = "\n".join(
            f"- {m}" for m in (recent_memories or [])[-5:]
        ) or "- (none)"

        skill_block = ""
        if available_skills:
            ranked = sorted(
                available_skills.items(), key=lambda kv: kv[1], reverse=True,
            )[:8]
            skill_block = (
                "Available skills (name: affordance probability):\n"
                + "\n".join(f"- {name}: {prob:.2f}" for name, prob in ranked)
            )

        prompt = (
            f"{personality_prompt}\n"
            f"Current mood: {mood_summary}\n"
            f"Recent memories:\n{memory_block}\n"
            f"Current goal: {current_goal or 'none'}\n"
            f"{skill_block}\n"
            f"Generate a single brief inner thought (one sentence):\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # thought classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_thought(thought: str) -> str:
        """Heuristically classify an LLM-generated thought into a type.

        A lightweight keyword classifier keeps us from needing a second LLM
        call just for tagging.
        """
        lower = thought.lower()

        _keywords: Dict[str, List[str]] = {
            "plan":        ["should", "plan", "next", "let me", "going to", "will try", "i'll"],
            "reflection":  ["went", "learned", "outcome", "reflect", "realized", "succeeded", "failed"],
            "appraisal":   ["feel", "mood", "emotion", "sense", "vibe", "comfort"],
            "curiosity":   ["wonder", "curious", "what if", "explore", "discover", "interesting"],
            "social":      ["human", "help", "company", "someone", "together", "people"],
            "observation": ["notice", "see", "sense", "detect", "observe", "environment", "change"],
        }

        best_type = "observation"
        best_score = 0
        for ttype, words in _keywords.items():
            score = sum(1 for w in words if w in lower)
            if score > best_score:
                best_score = score
                best_type = ttype

        return best_type

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _clamp(self, text: str) -> str:
        """Enforce max thought length."""
        text = text.strip()
        if len(text) > self.config.max_thought_length:
            text = text[: self.config.max_thought_length - 3].rstrip() + "..."
        return text

    def _record(self, thought: str, thought_type: str) -> None:
        """Append a thought to history and update the cooldown timestamp.

        Thread-safe: acquires ``self._lock``.
        """
        now = time.monotonic()
        with self._lock:
            self.thought_history.append((now, thought, thought_type))
            self.last_thought_time = now

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "LLM" if self._use_llm else "template"
        n = len(self.thought_history)
        return (
            f"InnerMonologue(mode={mode}, thoughts={n}, "
            f"cooldown={self.config.cooldown_seconds}s)"
        )
