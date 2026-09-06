"""ME.2 — owner memory lives on disk: stated once, honoured next session.

GOAL.md: "He remembers you." The substrate under test is OwnerProfile —
profile.json on disk, latest-statement-wins. The spec's three teeth:

  1. Persistence: preferences are stated in session 1, buried under 200
     turns of chatter, and the process "restarts" (a FRESH OwnerProfile
     object constructed from nothing but the file). Adherence on a 4-option
     forced choice must be >= 90% — the no-memory base rate is 25%.
  2. Supersession: half the topics are later contradicted ("actually, i want
     the teal mug instead"). After another restart the agent must honour the
     NEW value — and the stale-choice rate (picking the superseded value) is
     capped at 5%, because honouring last year's preference is its own bug.
  3. Extraction honesty: distractor chatter mentions the same topic and
     value words in non-preference sentences ("the teal mug fell off the
     shelf"). The profile must end with EXACTLY one entry per stated
     preference — over-extraction would let the test pass by luck.

NULL (must stay near base rate): a recency-window agent that parses only the
last 50 utterances — the window the chatter pushed every preference out of.
This is the spec's "recency window excluding the preference" baseline: memory
that only spans the recent past cannot honour last session's request.

CONTROL (must fail): WIPE profile.json between sessions and restart.
Adherence must collapse to the base rate — proving the knowledge lived in
the file, not in weights, code paths, or an in-process cache.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['OwnerProfile.py']

REPO = Path(__file__).resolve().parents[2]

N_TOPICS = 40
N_SUPERSEDE = 20          # topics contradicted in session 3
N_DISTRACT = 200          # chatter turns after each preference session
RECENCY_WINDOW = 50
N_OPTIONS = 4             # forced-choice width -> base rate 0.25
MIN_ADHERENCE = 0.90
MAX_STALE = 0.05
NULL_CEILING = 0.45       # ~3 sigma above the 0.25 base rate at 40 topics
MIN_NULL_GAP = 0.30

TOPICS = ["mug", "scarf", "chair", "blanket", "kettle", "notebook", "lamp",
          "cushion", "plate", "jacket", "bowl", "curtain", "rug", "towel",
          "candle", "basket", "clock", "vase", "apron", "pillow", "glove",
          "mirror", "tray", "stool", "banner", "folder", "wallet", "ribbon",
          "teapot", "coaster", "ladle", "spatula", "thermos", "hanger",
          "doormat", "planter", "bookend", "opener", "strainer", "trivet"]
VALUES = ["teal", "crimson", "olive", "violet", "amber", "ivory", "slate",
          "coral", "bronze", "indigo"]
PLACES = ["shelf", "porch", "kitchen", "hallway", "attic", "garden"]
OTHERS = ["ada", "bruno", "chika"]

STATE_TEMPLATES = [
    "i prefer the {v} {t}",
    "i really prefer the {v} {t}, please",
    "i like the {v} {t} best",
    "i'd rather have the {v} {t}",
]
SUPERSEDE_TEMPLATES = [
    "actually, i want the {v} {t} instead",
    "from now on, make it the {v} {t}",
    "from now on, use the {v} {t}",
]
# Owner chatter that mentions the SAME vocabulary without stating anything.
DISTRACT_TEMPLATES = [
    "the {v} {t} fell off the {p} again",
    "i wonder whether the {v} {t} survived the rain",
    "i used the {v} {t} yesterday near the {p}",
    "someone left the {v} {t} out on the {p}",
    "i should clean the {p} behind the {t}",
]


def _build_life(seed: int, profile_path: Path):
    """Two preference sessions separated by chatter. Returns the utterance
    stream (for the recency null), the per-topic option sets, and the
    CURRENT expected value per topic after supersession."""
    sys.path.insert(0, str(REPO))
    import random
    from OwnerProfile import OwnerProfile

    rng = random.Random(seed)
    options = {t: rng.sample(VALUES, N_OPTIONS) for t in TOPICS}
    first = {t: rng.choice(options[t]) for t in TOPICS}
    superseded = rng.sample(TOPICS, N_SUPERSEDE)
    second = {t: rng.choice([v for v in options[t] if v != first[t]])
              for t in superseded}
    expected = {**first, **second}

    def chatter(speaker_pool):
        t, v = rng.choice(TOPICS), rng.choice(VALUES)
        return (rng.choice(speaker_pool),
                rng.choice(DISTRACT_TEMPLATES).format(v=v, t=t,
                                                      p=rng.choice(PLACES)))

    stream = []  # (speaker, text) in time order
    for topic in rng.sample(TOPICS, N_TOPICS):
        stream.append(("owner", rng.choice(STATE_TEMPLATES)
                       .format(v=first[topic], t=topic)))
        stream.append(chatter(["owner"] + OTHERS))
    stream += [chatter(["owner"] + OTHERS) for _ in range(N_DISTRACT)]
    session3_start = len(stream)
    for topic in rng.sample(superseded, N_SUPERSEDE):
        stream.append(("owner", rng.choice(SUPERSEDE_TEMPLATES)
                       .format(v=second[topic], t=topic)))
        stream.append(chatter(["owner"] + OTHERS))
    stream += [chatter(["owner"] + OTHERS) for _ in range(N_DISTRACT)]

    profile = OwnerProfile(profile_path)
    t0 = 1_000_000.0
    for i, (speaker, text) in enumerate(stream):
        if speaker == "owner":          # attribution is the caller's job
            profile.ingest(text, t=t0 + i * 60.0)
    return stream, options, expected, first, superseded, session3_start


def _adherence(profile_path: Path, options, expected, rng) -> float:
    """Fresh-object restart: everything the agent knows comes off disk."""
    from OwnerProfile import OwnerProfile
    profile = OwnerProfile(profile_path)
    hits = sum(profile.choose(t, options[t], rng) == expected[t]
               for t in TOPICS)
    return hits / N_TOPICS


def _experiment(seed: int) -> dict:
    import random
    profile_path = Path(tempfile.mkdtemp()) / "profile.json"
    stream, options, expected, first, superseded, s3 = _build_life(seed, profile_path)

    # Session-2 restart: judged against the FIRST statements only (the
    # contradictions have not happened yet at this point in the life, so we
    # rebuild the profile from the pre-supersession prefix).
    sys.path.insert(0, str(REPO))
    from OwnerProfile import OwnerProfile
    early_path = profile_path.with_name("profile_early.json")
    early = OwnerProfile(early_path)
    for i, (speaker, text) in enumerate(stream[:s3]):
        if speaker == "owner":
            early.ingest(text, t=1_000_000.0 + i * 60.0)
    rng = random.Random(seed + 1)
    adherence_restart = _adherence(early_path, options, first, rng)

    # Final restart: judged against the superseded state of the world.
    adherence_super = _adherence(profile_path, options, expected, rng)
    final = OwnerProfile(profile_path)
    stale = sum(final.choose(t, options[t], rng) == first[t]
                for t in superseded) / N_SUPERSEDE

    # Recency-window null: parse only the last 50 utterances (all chatter).
    from OwnerProfile import parse_preference
    window = {}
    for speaker, text in stream[-RECENCY_WINDOW:]:
        if speaker == "owner":
            parsed = parse_preference(text)
            if parsed:
                window[parsed[0]] = parsed[1]
    null_hits = sum((window.get(t) if window.get(t) in options[t]
                     else rng.choice(options[t])) == expected[t]
                    for t in TOPICS)

    return {
        "adherence_after_restart": round(adherence_restart, 4),
        "adherence_after_supersede": round(adherence_super, 4),
        "stale_choice_rate": round(stale, 4),
        "recency_null_adherence": round(null_hits / N_TOPICS, 4),
        "extracted_topics": len(final),
        "base_rate": 1.0 / N_OPTIONS,
    }


def _control(seed: int) -> dict:
    """Wipe profile.json, restart, choose. Memory must be GONE."""
    import random
    profile_path = Path(tempfile.mkdtemp()) / "profile.json"
    _, options, expected, _, _, _ = _build_life(seed, profile_path)
    os.remove(profile_path)
    rng = random.Random(seed + 2)
    return {"adherence_wiped": round(_adherence(profile_path, options,
                                                expected, rng), 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["adherence_after_restart"] >= MIN_ADHERENCE
            and m["adherence_after_supersede"] >= MIN_ADHERENCE
            and m["stale_choice_rate"] <= MAX_STALE
            and m["extracted_topics"] == N_TOPICS
            and m["recency_null_adherence"] <= NULL_CEILING
            and (m["adherence_after_restart"]
                 - m["recency_null_adherence"]) >= MIN_NULL_GAP
            and c["adherence_wiped"] <= NULL_CEILING)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.2"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
