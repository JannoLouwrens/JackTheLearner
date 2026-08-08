"""T2.10 — retrieval SCORING beats a pure-recency baseline on recall questions.

ME.1 already killed the degenerate recency null (answer everything with the
single newest event: ~0.001 recall). That is not the interesting fight. The
interesting fight is the one Generative Agents scoring (recency x importance x
relevance — registry notes) was designed for, and this spec stages both halves
of it:

  Arm A — recency is COMPETITIVE. Recall questions are recency-biased the way
  real conversation is (35% of questions ask about the last ~10 minutes of
  life). "Return the k most recent events" now scores ~0.15-0.20 recall@5
  instead of ~0.01 — a real baseline. Scored retrieval must still beat it by
  a pre-registered margin, because most questions are NOT about just now.

  Arm B — recency is LOAD-BEARING inside the score. 30 recurring situations
  (same obj/place/colour/action 4-tuple lived through 5 separate times). A cue
  naming all four words matches all 5 occurrences at similarity 1.0; the
  right answer to "the ladder thing" in a life where the ladder recurred is
  the LATEST occurrence, and only the recency term can say so. A
  similarity-only scorer (w_recency=0, w_importance=0 — the control that must
  fail) has no basis to prefer any occurrence and lands on the wrong one.

So the claim is precisely the spec's: the COMBINED scoring beats pure recency
(arm A) and is not secretly pure similarity either (arm B). Occurrences of a
recurring tuple share one fixed importance so arm B isolates the recency term;
distinct events keep random importance as in ME.1.

NULL (must lose): the k=5 most recent events answer every question.
CONTROL (must fail): similarity-only scoring on arm B's latest-occurrence
questions — expected ~0, since without recency the tie among 5 identical
matches is broken arbitrarily.

Data notes: 650 events one minute apart (10.8 h of life vs the store's 6 h
recency half-life, so recency differences are material). 500 distinct events
carry unique 4-tuples (a 4-word cue identifies exactly one event — oracle 1.0
by construction); recurring occurrences sit in the first 600 slots so the
life's tail, where recency-biased questions concentrate, is distinct events.
Seeded RNG; 3 seeds by spec.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

N_DISTINCT = 500
N_RECUR_TUPLES = 30
RECUR_TIMES = 5
N_TOTAL = N_DISTINCT + N_RECUR_TUPLES * RECUR_TIMES   # 650
N_QUESTIONS_A = 120
P_RECENT = 0.35          # fraction of arm-A questions about the recent past
RECENT_WINDOW = 10       # "recent" = the last 10 events of the life
TOP_K = 5

MIN_RECALL_AT_5 = 0.90
MIN_LATEST_AT_1 = 0.90
MAX_NULL_RECALL = 0.50
MIN_MARGIN = 0.30
MAX_SIMONLY_LATEST = 0.50

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

T0 = 1_000_000.0
STEP_S = 60.0


def _unique_tuples(rng, n, seen):
    out = []
    while len(out) < n:
        tup = (rng.choice(OBJECTS), rng.choice(PLACES),
               rng.choice(COLOURS), rng.choice(ACTIONS))
        if tup not in seen:
            seen.add(tup)
            out.append(tup)
    return out


def _build_life(seed: int, mem_path: Path):
    """One life: distinct events everywhere, recurring tuples in slots 0..599.

    Returns (mem, distinct, recurring, now) where distinct is [(Event, tuple)]
    and recurring maps tuple -> [Event] in time order.
    """
    sys.path.insert(0, str(REPO))
    import random
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(seed)
    seen = set()
    distinct_tuples = _unique_tuples(rng, N_DISTINCT, seen)
    recur_tuples = _unique_tuples(rng, N_RECUR_TUPLES, seen)

    recur_slots = sorted(rng.sample(range(N_TOTAL - 50),
                                    N_RECUR_TUPLES * RECUR_TIMES))
    slot_to_tuple = {}
    order = [t for t in recur_tuples for _ in range(RECUR_TIMES)]
    rng.shuffle(order)
    for slot, tup in zip(recur_slots, order):
        slot_to_tuple[slot] = tup

    mem = EpisodicMemory(path=mem_path)
    distinct, recurring = [], {t: [] for t in recur_tuples}
    di = 0
    for i in range(N_TOTAL):
        if i in slot_to_tuple:
            tup, importance = slot_to_tuple[i], 1.0
        else:
            tup, importance = distinct_tuples[di], rng.uniform(0.5, 5.0)
            di += 1
        obj, place, colour, act = tup
        speaker = rng.choice(SPEAKERS)
        channel = ("said" if speaker == "jack" and rng.random() < 0.5
                   else "did" if speaker == "jack"
                   else "heard")
        text = f"{speaker} {act} the {colour} {obj} near the {place}"
        ev = mem.record(channel, speaker, text, importance=importance,
                        t=T0 + i * STEP_S)
        if i in slot_to_tuple:
            recurring[tup].append(ev)
        else:
            distinct.append((ev, tup))
    return mem, distinct, recurring, T0 + N_TOTAL * STEP_S


def _cue(rng, tup) -> str:
    obj, place, colour, act = tup
    words = [obj, place, colour, act]
    rng.shuffle(words)
    return (f"what about the {words[0]} and the {words[1]} "
            f"and the {words[2]} {words[3]}")


def _questions(seed: int, distinct, recurring):
    """Same question stream for experiment and controls (same seed, same rng)."""
    import random
    rng = random.Random(seed + 1)
    recent = distinct[-RECENT_WINDOW:]   # tail of the life is distinct-only
    arm_a = []
    for _ in range(N_QUESTIONS_A):
        ev, tup = rng.choice(recent if rng.random() < P_RECENT else distinct)
        arm_a.append((ev, _cue(rng, tup)))
    arm_b = [(max(evs, key=lambda e: e.t), _cue(rng, tup))
             for tup, evs in recurring.items()]
    return arm_a, arm_b


def _experiment(seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    mem, distinct, recurring, now = _build_life(seed, tmp)
    arm_a, arm_b = _questions(seed, distinct, recurring)

    hits5 = hits1 = 0
    for ev, cue in arm_a:
        res = mem.recall(cue, top_k=TOP_K, now=now)
        hits5 += any(r.event.eid == ev.eid for r in res)
        hits1 += bool(res and res[0].event.eid == ev.eid)

    latest1 = sum(bool((r := mem.recall(cue, top_k=1, now=now))
                       and r[0].event.eid == ev.eid)
                  for ev, cue in arm_b)

    return {
        "events": len(mem),
        "recall_at_5": round(hits5 / len(arm_a), 4),
        "recall_at_1": round(hits1 / len(arm_a), 4),
        "latest_at_1": round(latest1 / len(arm_b), 4),
    }


def _control(seed: int) -> dict:
    """The two scorers that must lose: pure recency, and similarity-only."""
    from EpisodicMemory import EpisodicMemory
    tmp = Path(tempfile.mkdtemp()) / "life_null.jsonl"
    mem, distinct, recurring, now = _build_life(seed, tmp)
    arm_a, arm_b = _questions(seed, distinct, recurring)

    # Null: answer every question with the k most recent events.
    newest_k = {ev.eid for ev in
                sorted(mem.events, key=lambda e: e.t, reverse=True)[:TOP_K]}
    rec5 = sum(ev.eid in newest_k for ev, _ in arm_a)
    rec_latest = sum(ev.eid in newest_k for ev, _ in arm_b)

    # Control: same machinery, similarity only — no recency, no importance.
    sim_only = EpisodicMemory(path=tmp, w_recency=0.0, w_importance=0.0)
    sim_latest = sum(bool((r := sim_only.recall(cue, top_k=1, now=now))
                          and r[0].event.eid == ev.eid)
                     for ev, cue in arm_b)

    return {
        "recency_recall_at_5": round(rec5 / len(arm_a), 4),
        "recency_latest_at_1": round(rec_latest / len(arm_b), 4),
        "simonly_latest_at_1": round(sim_latest / len(arm_b), 4),
    }


def _check(m: dict, c: dict) -> bool:
    return (m["recall_at_5"] >= MIN_RECALL_AT_5
            and m["latest_at_1"] >= MIN_LATEST_AT_1
            and c["recency_recall_at_5"] < MAX_NULL_RECALL
            and m["recall_at_5"] - c["recency_recall_at_5"] >= MIN_MARGIN
            and c["simonly_latest_at_1"] < MAX_SIMONLY_LATEST)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.10"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
