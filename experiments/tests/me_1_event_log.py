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

STRENGTHENED 2026-09-06 (Review, FULL Part 2) — the disjoint-vocabulary
abstention control had been outgrown by our own evidence, and a second,
strictly harder abstention conjunct is added. The old control asks whether the
store invents an answer for a cue about "the zeppelin at the volcano": every
content word is absent, so a keyword filter passes it. On 2026-09-02 `ME.11`
SETTLED FAIL measuring the HARD version of the same question on this project's
retrieval stack — gold event masked, the topically-similar rest of the life
retained, threshold calibrated identically — and recorded distractor abstention
**0.877** against the 0.95 the claim required, answering on 12.29% +- 1.56% of
cues whose target was absent while finding only 6.67% of those present:
~1.8x more invention than recall. Nothing in ME.1 could have seen that,
because ME.1's absent-target cue shares no vocabulary with anything stored.

So ME.1 now carries `distractor_abstention` as a REQUIRED conjunct, built to
ME.11's design: 60 events are HELD OUT of the store, their cues are issued
against the 940 that remain, and the store must abstain — the cue's content
words are all present in the corpus, in other events, just never together in
one that exists. The bar is ME.1's OWN existing 0.95, applied to a harder
control; no threshold moved in either direction. Rig-aliveness: a cue is
excluded from the denominator if a RETAINED event happens to carry all three
of its picked content words (then a hit is correct retrieval, not
confabulation), and the excluded count is recorded so the filter cannot go
quiet.

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

# The store under test. Undeclared until 2026-09-06 (78th audit B2): the 12:16
# scorer recalibration staled this certificate and the staleness lane could not
# see it, because impl_sha only covers declared inputs.
IMPL_DEPS = ["EpisodicMemory.py"]

N_EVENTS = 1000
N_QUERIES = 120
N_FABRICATED = 60
N_DISTRACTOR = 60          # events held OUT of the store, then cued for
MIN_DISTRACTOR_EVAL = 30   # aliveness: below this the control has gone quiet
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


def _build_life(seed: int, mem_path: Path, skip: set[int] | None = None):
    """Build the synthetic life. `skip` holds out event INDICES from the store
    while still returning their content words, so a cue can be issued about an
    event that never made it in — ME.11's distractor design."""
    sys.path.insert(0, str(REPO))
    import random
    from EpisodicMemory import EpisodicMemory

    skip = skip or set()
    rng = random.Random(seed)
    mem = EpisodicMemory(path=mem_path)
    t0 = 1_000_000.0
    events, held_out = [], []
    for i in range(N_EVENTS):
        speaker = rng.choice(SPEAKERS)
        channel = ("said" if speaker == "jack" and rng.random() < 0.5
                   else "did" if speaker == "jack"
                   else "heard")
        obj, place = rng.choice(OBJECTS), rng.choice(PLACES)
        colour, act = rng.choice(COLOURS), rng.choice(ACTIONS)
        words = (obj, place, colour, act)
        # Draw identically whether or not the event is stored, so the held-out
        # run sees the SAME 940 retained events as the full run's first 940.
        imp = rng.uniform(0.5, 5.0)
        if i in skip:
            held_out.append(words)
            continue
        text = f"{speaker} {act} the {colour} {obj} near the {place}"
        ev = mem.record(channel, speaker, text, importance=imp, t=t0 + i * 60.0)
        events.append((ev, words))
    return mem, events, t0 + N_EVENTS * 60.0, held_out


def _cue(rng, words) -> str:
    obj, place, colour, act = words
    picks = rng.sample([obj, place, colour, act], 3)
    return f"the thing about the {picks[0]} and the {picks[1]} {picks[2]}"


def _distractor_abstention(seed: int) -> tuple[int, int]:
    """ME.11's control, on ME.1's store: hold N_DISTRACTOR events OUT, then cue
    for them against the topically-similar remainder. Every content word in the
    cue exists in the corpus; only this combination does not. Returns
    (abstained, evaluated) — evaluated excludes cues a RETAINED event answers
    correctly, because that is retrieval, not confabulation."""
    import random
    tmp = Path(tempfile.mkdtemp()) / "life_distractor.jsonl"
    held = set(random.Random(seed + 7).sample(range(N_EVENTS), N_DISTRACTOR))
    mem, events, now, held_words = _build_life(seed, tmp, skip=held)
    retained = [w for _, w in events]
    rng = random.Random(seed + 8)

    abstained = evaluated = 0
    for words in held_words:
        cue_words = rng.sample([words[0], words[1], words[2], words[3]], 3)
        picks = set(cue_words)
        # A retained event carrying all three picked words makes a hit CORRECT.
        if any(picks <= set(rw) for rw in retained):
            continue
        cue = (f"the thing about the {cue_words[0]} and the "
               f"{cue_words[1]} {cue_words[2]}")
        evaluated += 1
        abstained += not mem.recall(cue, top_k=1, now=now)
    return abstained, evaluated


def _experiment(seed: int) -> dict:
    import random
    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    mem, events, now, _ = _build_life(seed, tmp)
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

    d_abst, d_eval = _distractor_abstention(seed)

    return {
        "events": len(mem),
        "cued_recall": round(hits / N_QUERIES, 4),
        "recall_after_reload": round(rehits / N_QUERIES, 4),
        "fabricated_abstention": round(fabricated_abstained / N_FABRICATED, 4),
        # Strengthened 2026-09-06 — ME.11's distractor control, on this store.
        "distractor_abstention": round(d_abst / d_eval, 4) if d_eval else 0.0,
        "distractor_evaluated": d_eval,
        "distractor_excluded": N_DISTRACTOR - d_eval,
    }


def _control(seed: int) -> dict:
    """Recency-only null: most recent event wins every query. Must lose badly."""
    import random
    tmp = Path(tempfile.mkdtemp()) / "life_null.jsonl"
    mem, events, now, _ = _build_life(seed, tmp)
    rng = random.Random(seed + 1)
    newest = max((e for e, _ in events), key=lambda e: e.t)
    hits = sum(bool(newest.eid == ev.eid) for ev, _ in rng.sample(events, N_QUERIES))
    return {"recency_only_recall": round(hits / N_QUERIES, 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["cued_recall"] >= MIN_RECALL
            and m["recall_after_reload"] >= MIN_RECALL
            and m["fabricated_abstention"] >= MIN_ABSTENTION
            # Strengthened 2026-09-06: the hard abstention question, at the
            # spec's own unchanged 0.95 bar, with an aliveness floor so a
            # control that stops evaluating anything cannot pass by silence.
            and m["distractor_evaluated"] >= MIN_DISTRACTOR_EVAL
            and m["distractor_abstention"] >= MIN_ABSTENTION
            and c["recency_only_recall"] < MIN_RECALL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.1"], _experiment, _check, control_fn=_control, ledger=ledger)
