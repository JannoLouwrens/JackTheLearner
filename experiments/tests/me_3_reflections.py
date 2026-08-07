"""ME.3 — reflections beat raw events on aggregation questions, equal tokens.

The claim (docs/research/MEMORY.md 2.3, Generative Agents 2304.03442): for
questions about what USUALLY happens — "which place is ada usually near?" —
a consolidated belief with a count over the whole log beats top-k raw events,
because at a fixed token budget raw retrieval can only show a handful of
episodes while one reflection line carries the tally of hundreds.

Setup: a 1,200-event synthetic life where each of four speakers has a habit —
a favourite object, place, colour, and action drawn with probability P_FAV,
everything else uniform. The habit is real but MILD (P_FAV = 0.18): a handful
of retrieved events is a noisy sample of it, the full log is not. Questions
are 4-way forced choice over the EMPIRICAL mode of the speaker's log (the
truth about the log, not the generator), 8 candidate draws per speaker x
dimension, speakers ada/bruno/chika (jack's own channels are ME.9's beat;
his name is also a retrieval stop-word, which would starve both arms alike).

Both arms are mechanically identical — same cue string, same retrieval
contract, same greedy packing to TOKEN_BUDGET whitespace tokens, same reader
(tally candidate mentions, weighted by a parsed "(N of M events)" count when
a line states one, weight 1 otherwise; max wins, ties random):

  reflect arm: Reflections consolidated from the log, RELOADED FROM DISK
               before answering — beliefs must live in the file.
  raw arm    : top-k events from EpisodicMemory at the same budget. This is
               the null; equal tokens is enforced by a check that the raw arm
               packed AT LEAST as many tokens as the reflect arm.

CONTROL (must hurt): answer the same questions from reflections consolidated
from ANOTHER agent's log (same machinery, different life). Habit beliefs that
transfer between different lives were never about THIS life — accuracy must
fall below the raw-events null.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

N_EVENTS = 1200
P_FAV = 0.18                  # habit strength: mild on purpose (see docstring)
N_CANDIDATES = 4              # forced-choice width -> base rate 0.25
N_REPS = 8                    # candidate re-draws per (speaker, dimension)
TOKEN_BUDGET = 40             # ~5 raw events or ~4 reflection lines
MIN_REFLECT = 0.90
MIN_GAIN = 0.15               # aggregation_qa_gain = reflect - raw
MIN_CONTROL_DROP = 0.30       # reflect minus wrong-agent accuracy

OBJECTS = ["kettle", "ladder", "apple", "lantern", "hammer", "compass",
           "bucket", "rope", "mirror", "whistle", "anchor", "basket", "drum",
           "kite", "shovel", "candle", "bell", "net", "flag", "chain"]
PLACES = ["pond", "ramp", "platform", "meadow", "shed", "gate", "bridge",
          "cellar", "orchard", "quarry", "dock", "tower", "trail", "garden"]
COLOURS = ["copper", "crimson", "olive", "violet", "amber", "teal", "ivory",
           "slate", "coral", "bronze"]
ACTIONS = ["carried", "dropped", "painted", "repaired", "buried", "balanced",
           "measured", "cleaned", "stacked", "traded", "borrowed", "hid"]
DIMENSIONS = {"object": OBJECTS, "place": PLACES,
              "colour": COLOURS, "action": ACTIONS}
SPEAKERS = ["ada", "bruno", "chika", "jack"]
ASKABLE = ["ada", "bruno", "chika"]

_COUNT = re.compile(r"\((\d+) of \d+ events\)")


def _build_life(seed: int, mem_path: Path):
    """A life with per-speaker habits. Returns the store, the empirical
    per-(speaker, dimension) value counts, and end-of-life time."""
    sys.path.insert(0, str(REPO))
    import random
    from collections import Counter
    from EpisodicMemory import EpisodicMemory

    rng = random.Random(seed)
    fav = {s: {d: rng.choice(pool) for d, pool in DIMENSIONS.items()}
           for s in SPEAKERS}

    def draw(s, d):
        pool = DIMENSIONS[d]
        if rng.random() < P_FAV:
            return fav[s][d]
        return rng.choice([v for v in pool if v != fav[s][d]])

    mem = EpisodicMemory(path=mem_path)
    tally = {s: {d: Counter() for d in DIMENSIONS} for s in SPEAKERS}
    t0 = 1_000_000.0
    for i in range(N_EVENTS):
        s = rng.choice(SPEAKERS)
        vals = {d: draw(s, d) for d in DIMENSIONS}
        channel = ("said" if s == "jack" and rng.random() < 0.5
                   else "did" if s == "jack" else "heard")
        mem.record(channel, s,
                   f"{s} {vals['action']} the {vals['colour']} "
                   f"{vals['object']} near the {vals['place']}",
                   importance=rng.uniform(0.5, 5.0), t=t0 + i * 60.0)
        for d, v in vals.items():
            tally[s][d][v] += 1
    return mem, tally, t0 + N_EVENTS * 60.0


def _questions(seed: int, tally):
    """4-way forced choice on the empirical mode; skip the rare non-unique
    mode rather than invent a truth the log does not contain."""
    import random
    rng = random.Random(seed + 1)
    out = []
    for s in ASKABLE:
        for d, pool in DIMENSIONS.items():
            ranked = tally[s][d].most_common(2)
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                continue
            truth = ranked[0][0]
            for _ in range(N_REPS):
                decoys = rng.sample([v for v in pool if v != truth],
                                    N_CANDIDATES - 1)
                cands = [truth] + decoys
                rng.shuffle(cands)
                out.append((s, cands, truth))
    return out


def _pack(lines):
    """Greedy packing under the shared whitespace-token budget."""
    packed, used = [], 0
    for text in lines:
        n = len(text.split())
        if used + n > TOKEN_BUDGET:
            break
        packed.append(text)
        used += n
    return packed, used


def _read(lines, candidates, rng):
    """The shared reader: tally candidate mentions, weighted by a stated
    evidence count when the line carries one. Purely mechanical — the
    information advantage must be in the lines, not the reader."""
    from EpisodicMemory import _tokens
    score = {c: 0 for c in candidates}
    for text in lines:
        m = _COUNT.search(text)
        w = int(m.group(1)) if m else 1
        tok = _tokens(text)
        for c in candidates:
            if c in tok:
                score[c] += w
    best = max(score.values())
    if best == 0:
        return rng.choice(candidates)
    return rng.choice([c for c in candidates if score[c] == best])


def _answer_all(questions, reflect_store, mem, now, rng):
    """Accuracy of each arm over the same questions, cue, budget, reader."""
    r_hits = e_hits = r_tok = e_tok = 0
    for s, cands, truth in questions:
        cue = " ".join([s] + cands)
        r_lines, rt = _pack([b.text for b in reflect_store.recall(cue, top_k=32)])
        e_lines, et = _pack([r.event.text
                             for r in mem.recall(cue, top_k=32, now=now)])
        r_hits += _read(r_lines, cands, rng) == truth
        e_hits += _read(e_lines, cands, rng) == truth
        r_tok += rt
        e_tok += et
    n = len(questions)
    return (r_hits / n, e_hits / n, r_tok / n, e_tok / n)


def _experiment(seed: int) -> dict:
    import random
    sys.path.insert(0, str(REPO))
    from Reflections import Reflections

    tmp = Path(tempfile.mkdtemp())
    mem, tally, now = _build_life(seed, tmp / "life.jsonl")
    questions = _questions(seed, tally)

    refl = Reflections(path=tmp / "reflections.jsonl")
    refl.consolidate(mem, now=now)
    reloaded = Reflections(path=tmp / "reflections.jsonl")   # off disk only

    rng = random.Random(seed + 2)
    r_acc, e_acc, r_tok, e_tok = _answer_all(questions, reloaded, mem, now, rng)
    return {
        "n_questions": len(questions),
        "reflect_acc": round(r_acc, 4),
        "raw_acc": round(e_acc, 4),
        "aggregation_qa_gain": round(r_acc - e_acc, 4),
        "reflect_tokens_mean": round(r_tok, 1),
        "raw_tokens_mean": round(e_tok, 1),
        "n_beliefs": len(reloaded),
        "base_rate": 1.0 / N_CANDIDATES,
    }


def _control(seed: int) -> dict:
    """Answer THIS life's questions from ANOTHER life's reflections."""
    import random
    sys.path.insert(0, str(REPO))
    from Reflections import Reflections

    tmp = Path(tempfile.mkdtemp())
    mem, tally, now = _build_life(seed, tmp / "life.jsonl")
    questions = _questions(seed, tally)
    other, _, other_now = _build_life(seed + 7919, tmp / "other.jsonl")

    wrong = Reflections(path=tmp / "wrong.jsonl")
    wrong.consolidate(other, now=other_now)

    rng = random.Random(seed + 3)
    w_acc, _, _, _ = _answer_all(questions, wrong, mem, now, rng)
    return {"wrong_agent_acc": round(w_acc, 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["reflect_acc"] >= MIN_REFLECT
            and m["aggregation_qa_gain"] >= MIN_GAIN
            and m["raw_tokens_mean"] >= m["reflect_tokens_mean"]  # equal-tokens honesty
            and m["reflect_tokens_mean"] <= TOKEN_BUDGET
            and c["wrong_agent_acc"] < m["raw_acc"]
            and (m["reflect_acc"] - c["wrong_agent_acc"]) >= MIN_CONTROL_DROP)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.3"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
