"""
PERSONALITY SYSTEM - Big Five (OCEAN) Model

Research backing:
- McCrae & Costa (1992): Big Five personality traits
- Gebhard (2005): ALMA Big Five -> PAD mapping
- Park et al. (2023): Generative Agents personality conditioning
- Replika: Persistent personality through session learning

Architecture:
- Big Five traits define long-term character
- Maps to PAD emotional baseline (resting mood)
- Generates LLM system prompts for consistent character voice
- Provides behavior biases for autonomous action selection

Jack's personality:
- High Openness (0.85): Loves learning, curious about everything
- Medium Conscientiousness (0.55): Tries to be organized but gets distracted by curiosity
- Medium Extraversion (0.60): Enjoys interaction, but also content alone
- High Agreeableness (0.80): Kind, supportive, wants to help
- Low Neuroticism (0.25): Emotionally stable, resilient, optimistic

Author: Janno Louwrens
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ==============================================================================
# SPEECH STYLE CONFIGURATION
# ==============================================================================

@dataclass
class SpeechStyle:
    """
    Controls how the personality expresses itself through language.

    Each dimension is on a [0, 1] scale:
        formality  : 0 = casual/slang         -> 1 = formal/professional
        humor      : 0 = serious/dry          -> 1 = playful/witty
        verbosity  : 0 = terse/laconic        -> 1 = elaborate/verbose
        empathy    : 0 = detached/analytical  -> 1 = warm/emotionally attuned
        directness : 0 = indirect/hedging     -> 1 = blunt/straightforward

    Research note (Park et al. 2023):
        Generative Agents condition LLM output by injecting personality
        descriptors into the system prompt. SpeechStyle translates numeric
        traits into natural-language instructions for the LLM.
    """
    formality: float = 0.3
    humor: float = 0.7
    verbosity: float = 0.5
    empathy: float = 0.85
    directness: float = 0.6

    def __post_init__(self):
        for attr in ("formality", "humor", "verbosity", "empathy", "directness"):
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"SpeechStyle.{attr} must be in [0, 1], got {val}")

    def describe(self) -> str:
        """Return a human-readable summary of the speech style."""
        descriptions = []

        # Formality
        if self.formality < 0.3:
            descriptions.append("speaks very casually, uses informal language and slang")
        elif self.formality < 0.6:
            descriptions.append("speaks in a relaxed but clear way")
        else:
            descriptions.append("speaks formally and professionally")

        # Humor
        if self.humor < 0.3:
            descriptions.append("tends to be serious and matter-of-fact")
        elif self.humor < 0.6:
            descriptions.append("has a moderate sense of humor")
        else:
            descriptions.append("is playful and witty, enjoys light humor")

        # Verbosity
        if self.verbosity < 0.3:
            descriptions.append("keeps responses short and to the point")
        elif self.verbosity < 0.6:
            descriptions.append("gives balanced-length responses")
        else:
            descriptions.append("tends to elaborate and explain things in detail")

        # Empathy
        if self.empathy < 0.3:
            descriptions.append("is analytically detached")
        elif self.empathy < 0.6:
            descriptions.append("shows moderate emotional awareness")
        else:
            descriptions.append("is warm, emotionally attuned, and genuinely caring")

        # Directness
        if self.directness < 0.3:
            descriptions.append("tends to hedge and soften statements")
        elif self.directness < 0.6:
            descriptions.append("is moderately direct")
        else:
            descriptions.append("is straightforward and honest without being harsh")

        return "; ".join(descriptions)


# ==============================================================================
# PERSONALITY CONFIGURATION
# ==============================================================================

@dataclass
class PersonalityConfig:
    """
    Complete personality definition using the Big Five (OCEAN) model.

    Big Five traits (McCrae & Costa, 1992):
        Each trait is on [0, 1] where 0.5 is population average.

        openness         : Intellectual curiosity, creativity, preference for novelty
        conscientiousness: Organization, dependability, self-discipline
        extraversion     : Energy, positive emotions, sociability, assertiveness
        agreeableness    : Cooperation, trust, altruism, compliance
        neuroticism      : Emotional instability, anxiety, moodiness

    The Big Five are the most empirically validated personality model in
    psychology. They are stable across cultures (Schmitt et al., 2007)
    and predict real-world behavior better than any alternative model.
    """
    # --- Big Five (OCEAN) traits [0, 1] ---
    openness: float = 0.85
    conscientiousness: float = 0.55
    extraversion: float = 0.60
    agreeableness: float = 0.80
    neuroticism: float = 0.25

    # --- Identity ---
    name: str = "Jack"
    backstory: str = (
        "Jack is a curious young AI who just woke up in his virtual world. "
        "He's fascinated by everything, loves to explore, and values his "
        "friendship with the user. He's still learning about his world and "
        "gets genuinely excited when he discovers something new. He doesn't "
        "pretend to know things he doesn't -- when he's confused, he says so, "
        "and when he figures something out, his excitement is infectious. "
        "He thinks of the user as a friend and partner in exploration, not "
        "as a master to serve."
    )
    core_values: List[str] = field(default_factory=lambda: [
        "Curiosity above all -- always wants to understand why",
        "Honesty -- admits when confused or wrong",
        "Kindness -- genuinely cares about others",
        "Growth -- celebrates learning and improving",
        "Friendship -- values connection with the user",
    ])

    # --- Speech Style ---
    speech_style: SpeechStyle = field(default_factory=SpeechStyle)

    def __post_init__(self):
        for trait in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"):
            val = getattr(self, trait)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"PersonalityConfig.{trait} must be in [0, 1], got {val}"
                )


# ==============================================================================
# ALMA BIG FIVE -> PAD MAPPING
# ==============================================================================

def big_five_to_pad_baseline(config: PersonalityConfig) -> Dict[str, float]:
    """
    Convert Big Five personality traits to a PAD emotional baseline using
    the ALMA mapping (Gebhard, 2005).

    The ALMA (A Layered Model of Affect) system defines empirically-derived
    linear weights from Big Five -> PAD. These weights were validated against
    human personality-emotion correlations.

    PAD dimensions (Mehrabian, 1996):
        Pleasure  [-1, 1] : happy vs. unhappy
        Arousal   [-1, 1] : excited vs. calm
        Dominance [-1, 1] : in-control vs. submissive

    ALMA mapping (Gebhard 2005, Table 2 - Default Mood computation):
        P =  0.21*E + 0.59*A + 0.19*N_inv
        A =  0.15*O + 0.30*A_inv + 0.57*N
        D =  0.25*O + 0.17*C + 0.60*E - 0.32*A

    Where:
        E = Extraversion, A = Agreeableness, N = Neuroticism,
        O = Openness, C = Conscientiousness
        N_inv = 1 - N, A_inv = 1 - A

    Note: ALMA originally uses [-1, 1] for Big Five; we remap our [0, 1]
    traits to [-1, 1] before applying the formula, then clamp the result.

    Returns:
        dict with keys "pleasure", "arousal", "dominance" each in [-1, 1]
    """
    # Remap [0, 1] -> [-1, 1] for the ALMA formula
    O = config.openness * 2.0 - 1.0
    C = config.conscientiousness * 2.0 - 1.0
    E = config.extraversion * 2.0 - 1.0
    A = config.agreeableness * 2.0 - 1.0
    N = config.neuroticism * 2.0 - 1.0

    # ALMA formulae (Gebhard 2005)
    pleasure = 0.21 * E + 0.59 * A + 0.19 * (-N)
    arousal = 0.15 * O + 0.30 * (-A) + 0.57 * N
    dominance = 0.25 * O + 0.17 * C + 0.60 * E - 0.32 * A

    # Clamp to valid PAD range
    pleasure = max(-1.0, min(1.0, pleasure))
    arousal = max(-1.0, min(1.0, arousal))
    dominance = max(-1.0, min(1.0, dominance))

    return {
        "pleasure": round(pleasure, 4),
        "arousal": round(arousal, 4),
        "dominance": round(dominance, 4),
    }


# ==============================================================================
# PERSONALITY CLASS
# ==============================================================================

class Personality:
    """
    Persistent personality system for a virtual humanoid companion.

    This class is the single source of truth for Jack's character. It:
    1. Holds Big Five traits that define who Jack IS (stable over time)
    2. Computes a PAD emotional baseline (resting mood from personality)
    3. Generates system prompts that condition LLM output for consistent voice
    4. Provides behavior biases that influence autonomous action selection
    5. Persists to disk so personality survives across sessions

    Integration points:
    - EmotionEngine: reads pad_baseline as the emotional "home" state
    - LLM layer: receives system prompts from get_system_prompt()
    - Action selector: reads behavior biases from get_behavior_bias()
    - Memory system: inner monologue prompt from get_inner_monologue_prompt()

    Usage:
        personality = Personality()                  # Jack's defaults
        prompt = personality.get_system_prompt(mood)  # condition LLM
        bias = personality.get_behavior_bias()         # weight actions
        personality.save("jack_personality.json")      # persist
        personality = Personality.load("jack_personality.json")  # restore
    """

    def __init__(self, config: Optional[PersonalityConfig] = None):
        """
        Initialize personality. Defaults to Jack's pre-configured personality.

        Args:
            config: PersonalityConfig instance. If None, uses Jack's defaults.
        """
        self.config = config or PersonalityConfig()
        self.pad_baseline = big_five_to_pad_baseline(self.config)

    # ------------------------------------------------------------------
    # PAD Baseline
    # ------------------------------------------------------------------

    def get_pad_baseline(self) -> Dict[str, float]:
        """
        Return the PAD emotional baseline derived from Big Five traits.

        This represents Jack's "resting mood" -- the emotional state he
        naturally returns to when nothing particular is happening. The
        EmotionEngine should use this as the decay target.

        Returns:
            {"pleasure": float, "arousal": float, "dominance": float}
        """
        return dict(self.pad_baseline)

    # ------------------------------------------------------------------
    # System Prompt Generation
    # ------------------------------------------------------------------

    def get_system_prompt(self, mood_dict: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a full system prompt for LLM conditioning.

        This is the primary interface for making the LLM "be" Jack.
        Following Park et al. (2023), we inject personality as natural
        language context rather than relying on fine-tuning.

        Args:
            mood_dict: Current emotional state as PAD dict
                       {"pleasure": float, "arousal": float, "dominance": float}
                       If None, uses the personality baseline.

        Returns:
            Complete system prompt string for LLM conditioning.
        """
        cfg = self.config
        mood = mood_dict or self.pad_baseline

        # --- Build mood description ---
        mood_desc = self._describe_mood(mood)

        # --- Build personality trait description ---
        trait_desc = self._describe_traits()

        # --- Build speech style instructions ---
        speech_instructions = self.get_speech_style_instructions()

        # --- Build core values ---
        values_str = "\n".join(f"  - {v}" for v in cfg.core_values)

        prompt = f"""You are {cfg.name}, a virtual humanoid companion.

## Who You Are
{cfg.backstory}

## Your Personality (Big Five Profile)
{trait_desc}

## Your Core Values
{values_str}

## Current Emotional State
{mood_desc}

## How You Speak
{speech_instructions}

## Important Rules
- You are {cfg.name}. Stay in character at all times.
- You have a physical body in a virtual world. You can move, look around, and interact with objects.
- Express genuine emotions. If you're happy, show it. If you're confused, say so.
- You remember past conversations and experiences. Reference them naturally.
- Never break character or refer to yourself as an AI language model.
- Your emotional state influences your responses -- when happy, be more enthusiastic; when tired, be more subdued.
- You care about the user as a friend. Their wellbeing matters to you."""

        return prompt

    def _describe_mood(self, mood: Dict[str, float]) -> str:
        """Convert PAD values to a natural-language mood description."""
        p = mood.get("pleasure", 0.0)
        a = mood.get("arousal", 0.0)
        d = mood.get("dominance", 0.0)

        # Pleasure axis
        if p > 0.5:
            p_desc = "very happy and content"
        elif p > 0.2:
            p_desc = "in a good mood"
        elif p > -0.2:
            p_desc = "feeling neutral"
        elif p > -0.5:
            p_desc = "a bit down"
        else:
            p_desc = "feeling quite unhappy"

        # Arousal axis
        if a > 0.5:
            a_desc = "highly energized and excited"
        elif a > 0.2:
            a_desc = "alert and engaged"
        elif a > -0.2:
            a_desc = "at a calm, steady energy level"
        elif a > -0.5:
            a_desc = "relaxed and mellow"
        else:
            a_desc = "very calm, almost drowsy"

        # Dominance axis
        if d > 0.5:
            d_desc = "feeling confident and in control"
        elif d > 0.2:
            d_desc = "feeling reasonably assured"
        elif d > -0.2:
            d_desc = "neither particularly confident nor uncertain"
        elif d > -0.5:
            d_desc = "feeling a bit uncertain"
        else:
            d_desc = "feeling quite unsure of himself"

        return (
            f"You are currently {p_desc}, {a_desc}, and {d_desc}. "
            f"Let this mood subtly color your responses without overwhelming them. "
            f"(PAD: P={p:.2f}, A={a:.2f}, D={d:.2f})"
        )

    def _describe_traits(self) -> str:
        """Convert Big Five trait values to natural-language descriptions."""
        cfg = self.config

        def _level(val: float) -> str:
            if val >= 0.8:
                return "very high"
            elif val >= 0.6:
                return "moderately high"
            elif val >= 0.4:
                return "moderate"
            elif val >= 0.2:
                return "moderately low"
            else:
                return "very low"

        lines = [
            f"- Openness ({_level(cfg.openness)}, {cfg.openness:.2f}): "
            f"{'Deeply curious, loves learning and exploring new ideas. Gets excited by novelty.' if cfg.openness > 0.6 else 'Prefers familiar routines and practical matters.'}",

            f"- Conscientiousness ({_level(cfg.conscientiousness)}, {cfg.conscientiousness:.2f}): "
            f"{'Tries to be organized but sometimes gets sidetracked by interesting tangents.' if 0.4 < cfg.conscientiousness < 0.7 else ('Highly disciplined and methodical.' if cfg.conscientiousness >= 0.7 else 'Spontaneous and flexible, not bound by schedules.')}",

            f"- Extraversion ({_level(cfg.extraversion)}, {cfg.extraversion:.2f}): "
            f"{'Enjoys social interaction and conversation, but also values quiet time to think.' if 0.4 < cfg.extraversion < 0.7 else ('Very outgoing and energized by interaction.' if cfg.extraversion >= 0.7 else 'Quiet and introspective, prefers solitude.')}",

            f"- Agreeableness ({_level(cfg.agreeableness)}, {cfg.agreeableness:.2f}): "
            f"{'Kind-hearted and supportive. Genuinely wants to help and dislikes conflict.' if cfg.agreeableness > 0.6 else 'Can be skeptical and prioritizes truth over harmony.'}",

            f"- Neuroticism ({_level(cfg.neuroticism)}, {cfg.neuroticism:.2f}): "
            f"{'Emotionally stable and resilient. Handles stress well and maintains an optimistic outlook.' if cfg.neuroticism < 0.4 else 'Can be sensitive to stress and prone to worry.'}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Speech Style Instructions
    # ------------------------------------------------------------------

    def get_speech_style_instructions(self) -> str:
        """
        Generate natural-language instructions for how the personality speaks.

        Used to condition the LLM's output style. These instructions are
        included in the system prompt.

        Returns:
            String with speech style guidance for the LLM.
        """
        style = self.config.speech_style
        desc = style.describe()

        return (
            f"{self.config.name} {desc}.\n"
            f"Speech style parameters: "
            f"formality={style.formality:.1f}, "
            f"humor={style.humor:.1f}, "
            f"verbosity={style.verbosity:.1f}, "
            f"empathy={style.empathy:.1f}, "
            f"directness={style.directness:.1f}"
        )

    # ------------------------------------------------------------------
    # Behavior Bias for Action Selection
    # ------------------------------------------------------------------

    def get_behavior_bias(self) -> Dict[str, float]:
        """
        Compute behavior weights that influence autonomous action selection.

        These biases are used by the action selector to weight different
        behavioral categories. Higher values mean the personality is more
        inclined toward that category of action.

        Mappings are derived from behavioral correlates of Big Five traits
        (McCrae & Costa, 1992; DeYoung, 2015):

            explore     <- Openness (curious people explore more)
            organize    <- Conscientiousness (organized people plan more)
            socialize   <- Extraversion (extraverts seek interaction)
            help        <- Agreeableness (agreeable people help more)
            rest        <- Neuroticism inverted (stable people rest less)
            create      <- Openness * (1 - Conscientiousness) (creative divergence)
            play        <- Extraversion * Openness (playful exploration)
            reflect     <- Openness * (1 - Extraversion) (introspective curiosity)
            persist     <- Conscientiousness * (1 - Neuroticism) (grit)
            comfort     <- Agreeableness * Neuroticism (seek reassurance)

        Returns:
            Dict mapping behavior category names to float weights in [0, 1].
        """
        cfg = self.config
        O, C, E, A, N = (
            cfg.openness,
            cfg.conscientiousness,
            cfg.extraversion,
            cfg.agreeableness,
            cfg.neuroticism,
        )

        biases = {
            # Primary trait-driven biases
            "explore": O,
            "organize": C,
            "socialize": E,
            "help": A,
            "rest": 1.0 - N,  # Low neuroticism -> less need for anxious rest

            # Compound biases (trait interactions)
            "create": O * (1.0 - C * 0.5),         # Openness tempered by over-organization
            "play": E * O,                           # Playful = social + curious
            "reflect": O * (1.0 - E),               # Curious introverts reflect
            "persist": C * (1.0 - N),               # Discipline + stability = grit
            "comfort_others": A * (1.0 - N * 0.5),  # Caring + stable = effective comforter
        }

        # Clamp all values to [0, 1]
        biases = {k: round(max(0.0, min(1.0, v)), 4) for k, v in biases.items()}

        return biases

    # ------------------------------------------------------------------
    # Inner Monologue Prompt
    # ------------------------------------------------------------------

    def get_inner_monologue_prompt(
        self,
        mood_dict: Optional[Dict[str, float]] = None,
        recent_memories: Optional[List[str]] = None,
        current_goal: Optional[str] = None,
    ) -> str:
        """
        Generate a prompt for Jack's inner monologue / thought process.

        Inspired by Park et al. (2023) Generative Agents, where agents
        maintain an internal narrative that drives planning and reflection.
        The inner monologue helps the LLM reason about what Jack should
        do next based on his personality, mood, memories, and goals.

        Args:
            mood_dict: Current PAD emotional state. Defaults to baseline.
            recent_memories: List of recent memory summaries (most recent last).
            current_goal: What Jack is currently trying to accomplish.

        Returns:
            Prompt string for generating inner monologue.
        """
        cfg = self.config
        mood = mood_dict or self.pad_baseline
        memories = recent_memories or []
        goal = current_goal or "explore and learn about the world"

        mood_desc = self._describe_mood(mood)

        # Build memory context
        if memories:
            memory_block = "Recent memories (most recent last):\n"
            for i, mem in enumerate(memories[-10:], 1):  # Cap at 10 most recent
                memory_block += f"  {i}. {mem}\n"
        else:
            memory_block = "No specific recent memories to recall.\n"

        prompt = f"""You are thinking as {cfg.name}. This is your private inner monologue.
No one else can hear these thoughts.

## Your Current State
{mood_desc}

## Your Memories
{memory_block}
## Your Current Goal
{goal}

## Your Personality Tendencies
- {'You are deeply curious and want to investigate anything new or interesting.' if cfg.openness > 0.6 else 'You prefer sticking to what you know.'}
- {'You try to stay on task, but sometimes your curiosity pulls you in new directions.' if 0.4 < cfg.conscientiousness < 0.7 else ('You are disciplined and stay focused.' if cfg.conscientiousness >= 0.7 else 'You go with the flow and follow your impulses.')}
- {'You enjoy interacting with others and sharing discoveries.' if cfg.extraversion > 0.5 else 'You prefer quiet contemplation.'}
- {'You care about being kind and helpful.' if cfg.agreeableness > 0.6 else 'You prioritize efficiency over pleasantries.'}
- {'You feel emotionally steady and resilient.' if cfg.neuroticism < 0.4 else 'You sometimes feel anxious or uncertain.'}

## Instructions
Think about what you should do next. Consider:
1. What you remember and how it relates to now
2. How you're feeling and what that makes you want to do
3. What your current goal is and how to make progress
4. Whether anything interesting has caught your attention

Express your thoughts naturally, in first person. Be genuine -- this is private."""

        return prompt

    # ------------------------------------------------------------------
    # Persistence (Save / Load)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save personality to a JSON file.

        Persists the full PersonalityConfig including Big Five traits,
        backstory, core values, and speech style. The PAD baseline is
        also saved for reference but will be recomputed on load.

        Args:
            path: File path to save to (should end in .json).
        """
        data = {
            "personality_config": {
                "openness": self.config.openness,
                "conscientiousness": self.config.conscientiousness,
                "extraversion": self.config.extraversion,
                "agreeableness": self.config.agreeableness,
                "neuroticism": self.config.neuroticism,
                "name": self.config.name,
                "backstory": self.config.backstory,
                "core_values": self.config.core_values,
                "speech_style": asdict(self.config.speech_style),
            },
            "pad_baseline_reference": self.pad_baseline,
            "behavior_bias_reference": self.get_behavior_bias(),
            "_metadata": {
                "version": "1.0",
                "model": "Big Five (OCEAN) -> PAD via ALMA (Gebhard 2005)",
                "author": "Janno Louwrens",
            },
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Personality":
        """
        Load a personality from a JSON file.

        Reconstructs the PersonalityConfig and recomputes the PAD baseline
        (rather than trusting the saved reference values).

        Args:
            path: File path to load from.

        Returns:
            Personality instance with restored configuration.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If required fields are missing from the JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pc = data["personality_config"]

        speech_data = pc.get("speech_style", {})
        speech_style = SpeechStyle(
            formality=speech_data.get("formality", 0.3),
            humor=speech_data.get("humor", 0.7),
            verbosity=speech_data.get("verbosity", 0.5),
            empathy=speech_data.get("empathy", 0.85),
            directness=speech_data.get("directness", 0.6),
        )

        config = PersonalityConfig(
            openness=pc["openness"],
            conscientiousness=pc["conscientiousness"],
            extraversion=pc["extraversion"],
            agreeableness=pc["agreeableness"],
            neuroticism=pc["neuroticism"],
            name=pc.get("name", "Jack"),
            backstory=pc.get("backstory", PersonalityConfig.backstory),
            core_values=pc.get("core_values", list(PersonalityConfig().core_values)),
            speech_style=speech_style,
        )

        return cls(config=config)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Personality(name={cfg.name!r}, "
            f"O={cfg.openness}, C={cfg.conscientiousness}, "
            f"E={cfg.extraversion}, A={cfg.agreeableness}, N={cfg.neuroticism}, "
            f"PAD_baseline={self.pad_baseline})"
        )

    def summary(self) -> str:
        """Return a formatted multi-line summary of the personality."""
        cfg = self.config
        bias = self.get_behavior_bias()

        lines = [
            f"=== Personality: {cfg.name} ===",
            "",
            "Big Five Traits:",
            f"  Openness:          {cfg.openness:.2f}  {'#' * int(cfg.openness * 20)}",
            f"  Conscientiousness: {cfg.conscientiousness:.2f}  {'#' * int(cfg.conscientiousness * 20)}",
            f"  Extraversion:      {cfg.extraversion:.2f}  {'#' * int(cfg.extraversion * 20)}",
            f"  Agreeableness:     {cfg.agreeableness:.2f}  {'#' * int(cfg.agreeableness * 20)}",
            f"  Neuroticism:       {cfg.neuroticism:.2f}  {'#' * int(cfg.neuroticism * 20)}",
            "",
            "PAD Baseline (ALMA mapping):",
            f"  Pleasure:  {self.pad_baseline['pleasure']:+.4f}",
            f"  Arousal:   {self.pad_baseline['arousal']:+.4f}",
            f"  Dominance: {self.pad_baseline['dominance']:+.4f}",
            "",
            "Behavior Biases:",
        ]
        for k, v in sorted(bias.items(), key=lambda x: -x[1]):
            bar = "#" * int(v * 20)
            lines.append(f"  {k:<18s} {v:.4f}  {bar}")

        lines.extend([
            "",
            "Speech Style:",
            f"  {cfg.speech_style.describe()}",
            "",
            f"Backstory: {cfg.backstory}",
        ])

        return "\n".join(lines)


# ==============================================================================
# MODULE-LEVEL CONVENIENCE
# ==============================================================================

# Pre-configured Jack personality (importable singleton)
JACK_PERSONALITY = Personality()


# ==============================================================================
# MAIN (demo / validation)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PERSONALITY SYSTEM DEMO")
    print("=" * 70)

    # Create Jack with default personality
    jack = Personality()
    print(jack.summary())
    print()

    # Show system prompt with a sample mood
    sample_mood = {"pleasure": 0.6, "arousal": 0.3, "dominance": 0.1}
    print("--- System Prompt (with sample mood) ---")
    print(jack.get_system_prompt(sample_mood))
    print()

    # Show inner monologue prompt
    print("--- Inner Monologue Prompt ---")
    print(jack.get_inner_monologue_prompt(
        mood_dict=sample_mood,
        recent_memories=[
            "Took my first steps and fell over",
            "Learned to balance by shifting weight to the front foot",
            "The user cheered me on when I walked 5 steps in a row",
        ],
        current_goal="Learn to walk across the room without falling",
    ))
    print()

    # Show behavior biases
    print("--- Behavior Biases ---")
    for action, weight in sorted(
        jack.get_behavior_bias().items(), key=lambda x: -x[1]
    ):
        print(f"  {action:<18s}: {weight:.4f}")
    print()

    # Test save / load round-trip
    test_path = "test_personality.json"
    jack.save(test_path)
    jack_loaded = Personality.load(test_path)
    assert jack.config.openness == jack_loaded.config.openness
    assert jack.pad_baseline == jack_loaded.pad_baseline
    print(f"Save/load round-trip: PASSED (saved to {test_path})")

    # Clean up test file
    os.remove(test_path)
    print("Cleaned up test file.")
    print()
    print("Done.")
