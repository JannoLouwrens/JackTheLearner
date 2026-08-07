"""ME.1 — what happened is retrievable, and what never happened is not.

The store under test is EpisodicMemory: append-only JSONL, retrieval scored
recency x importance x similarity, with a similarity floor below which it
ABSTAINS. This spec asks the two questions that make an event log worth
having at 1,000 events:

  1. Cued recall: given a partial, paraphrased cue about a real event
     ("the thing with the copper kettle by the pond"), does the top result
     land on THAT event >= 80% of the time?
  2. Honesty: given a cue about an event that NEVER happened, does it return
     nothing? Confabulating the nearest neighbour is the failure mode that
     poisons every downstream user of memory — a companion that invents your
     preferences is worse than one that forgets them.

NULL (must lose): recency-only retrieval — answer every cue with the most
recent event. At 1k events this is the "goldfish with a good mood" baseline
the scoring has to beat.

CONTROL (must fail): 60 fabricated cues built from vocabulary DISJOINT from
every stored event. Any retrieval that returns a match for one of these has a
broken threshold; the spec demands >= 95% abstention.

Synthetic-life generation notes, because the test is only as honest as its
data: events are templated from disjoint content-word pools (objects, places,
colours, activities), so each event has identifying vocabulary; cues take a
WORD SUBSET of their event plus generic filler, never the full sentence — a
cue identical to its event would test string equality, not memory. Seeded RNG
throughout; three seeds by spec.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

N_EVENTS = 1000
N_QUERIES = 120
N_FABRICATED = 60
MIN_RECALL = 0.80
MIN_ABSTENTION = 0.95

OBJECTS = ["kettle", "ladder", "apple", "lantern", "hammer", "compass", "bucket",
           "rope", "mirror", "whistle", "anchor", "basket", "drum", "kite",
           "shovel", "candle", "bell", "net", "flag", "chain"]
PLACES = ["pond", "ramp", "platform", "meadow", "shed", "gate", "bridge",
          "cellar", "orchard", "quarry", "dock", "tower", "trail", "garden"]
COLOURS = ["copper", "crimson", "olive", "violet", "amber", "teal", "ivory",
           "slate", "coral", "bronze"]
ACTIONS = ["carried", "dropped", "painted", "repaired", "buried", "balanced",
           "measured", "cleaned", "stacked", "traded", "borrowed", "hid"]
SPEAKERS = ["ada", "bruno", "chika", "jack"]

# Vocabulary for fabricated cues — checked disjoint from the pools above.
FAB_OBJECTS = ["zeppelin", "harmonica", "telescope", "cauldron", "typewriter",
               "gramophone", "sundial", "abacus", "monocle", "tapestry"]
FAB_PLACES = ["volcano", "lighthouse", "catacomb", "glacier", "bazaar",
              "observatory", "vineyard", "citadel"]


def _build_life(seed: int, mem_path: Path):
    sys.path.insert(0, str(REPO))
    import random
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(seed)
    mem = EpisodicMemory(path=mem_path)
    t0 = 1_000_000.0
    events = []
    for i in range(N_EVENTS):
        speaker = rng.choice(SPEAKERS)
        channel = ("said" if speaker == "jack" and rng.random() < 0.5
                   else "did" if speaker == "jack"
                   else "heard")
        obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
        colour, act = rng.choice(COLOURS), rng.choice(ACTIONS)
        text = f"{speaker} {act} the {colour} {obj} near the {place}"
        ev = mem.record(channel, speaker, text,
                        importance=rng.uniform(0.5, 5.0), t=t0 + i * 60.0)
        events.append((ev, (obj, place, colour, act)))
    return mem, events, t0 + N_EVENTS * 60.0


def _cue(rng, words) -> str:
    obj, place, colour, act = words
    picks = rng.sample([obj, place, colour, act], 3)
    return f"the thing about the {picks[0]} and the {picks[1]} {picks[2]}"


def _experiment(seed: int) -> dict:
    import random
    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    mem, events, now = _build_life(seed, tmp)
    rng = random.Random(seed + 1)

    sampled = rng.sample(events, N_QUERIES)
    hits = 0
    for ev, words in sampled:
        res = mem.recall(_cue(rng, words), top_k=1, now=now)
        hits += bool(res and res[0].event.eid == ev.eid)

    fabricated_abstained = 0
    for _ in range(N_FABRICATED):
        cue = (f"the thing about the {rng.choice(FAB_OBJECTS)} "
               f"and the {rng.choice(FAB_PLACES)}")
        fabricated_abstained += not mem.recall(cue, top_k=1, now=now)

    # Restart honesty: a fresh store loaded from the same file answers alike.
    from EpisodicMemory import EpisodicMemory
    mem2 = EpisodicMemory(path=tmp)
    rng2 = random.Random(seed + 1)
    resampled = rng2.sample(events, N_QUERIES)
    rehits = sum(bool((r := mem2.recall(_cue(rng2, w), top_k=1, now=now))
                      and r[0].event.eid == ev.eid)
                 for ev, w in resampled)

    return {
        "events": len(mem),
        "cued_recall": round(hits / N_QUERIES, 4),
        "recall_after_reload": round(rehits / N_QUERIES, 4),
        "fabricated_abstention": round(fabricated_abstained / N_FABRICATED, 4),
    }


def _control(seed: int) -> dict:
    """Recency-only null: most recent event wins every query. Must lose badly."""
    import random
    tmp = Path(tempfile.mkdtemp()) / "life_null.jsonl"
    mem, events, now = _build_life(seed, tmp)
    rng = random.Random(seed + 1)
    newest = max((e for e, _ in events), key=lambda e: e.t)
    hits = sum(bool(newest.eid == ev.eid) for ev, _ in rng.sample(events, N_QUERIES))
    return {"recency_only_recall": round(hits / N_QUERIES, 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["cued_recall"] >= MIN_RECALL
            and m["recall_after_reload"] >= MIN_RECALL
            and m["fabricated_abstention"] >= MIN_ABSTENTION
            and c["recency_only_recall"] < MIN_RECALL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.1"], _experiment, _check, control_fn=_control, ledger=ledger)
