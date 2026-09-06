"""ME.1 floor calibration probe — arms measured, not argued (law 3).

The queue row `me1-similarity-floor-never-abstains` (DUE 2026-09-13) owes a
calibration of `EpisodicMemory`'s similarity floor that abstains on absent
targets without costing `cued_recall`, at ME.1's own unchanged bars. Two
repair families exist and choosing between them by reasoning is exactly what
SYSTEM.md law 3 forbids, so this probe scores them on ME.1's own harness —
the same `_build_life`, the same `_cue`, the same distractor construction —
plus two robustness probes that PAST scars say the calibration must not
re-open:

  terse cues    ME.9's scar: Jaccard made every one-word attribution question
                abstain (0.0 across the board), which is why containment was
                adopted. A floor repair that re-breaks terse cues has traded
                one failure for the old one.
  verbose cues  ME.11's scar, inverted: real cues carry junk words the store
                has never seen. A raw-containment floor high enough to refuse
                a 2-of-3 near-neighbour (sim 0.50) also refuses the TRUE
                target the moment the cue carries four junk words
                (3/(4+4) = 0.375). Verbosity robustness is where the two
                repair families genuinely differ.

ARMS (all leave ME.1's 0.95 abstention bar and its exclusion filter alone —
the calibration is the MODULE's, the spec does not move):

  A0 shipped        raw containment |q∩e|/|q|, floor 0.34. The measured
                    failure: distractor_abstention 0.0000 (ME.1 attempt 5).
  A1 raw-0.60       raw containment, floor 0.60. Separates target (0.75)
                    from 2-of-3 neighbour (0.50) on ME.1's cue shape.
  A2 coverage-0.95  drop cue words absent from the STORE'S VOCABULARY (the
                    union of stored-event tokens), then require >= 0.95
                    coverage of the words that remain. A word the store has
                    never heard is noise; a word it knows but this event
                    lacks is evidence of mismatch. All-unknown cue -> abstain.
  A3 margin-0.20    shipped floor 0.34, plus abstain unless the top match
                    beats the runner-up's similarity by 0.20 (a lone match
                    passes — nothing to dominate).
  A4 cov+raw        A2's coverage conjunct AND A0's raw floor 0.34 — the raw
                    floor guards the one hole coverage opens (a cue whose
                    only KNOWN word is common answers with full confidence).

Decision rule, pre-stated: an arm must clear ME.1's four conjunct analogues
(cued_recall >= 0.80, fabricated >= 0.95, distractor >= 0.95, distractor
evaluated >= 30) on ALL THREE seeds; among survivors, higher terse answer
rate then higher verbose recall then simpler implementation. The winner is
adopted in EpisodicMemory.py and VERIFIED by re-running ME.1 through the
runner — this probe decides, the ledger certifies.

Run: /data/venvs/jackthelearner/bin/python -m experiments.tests.me1_floor_probe
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.tests.me_1_event_log import (  # noqa: E402
    N_EVENTS, N_QUERIES, N_FABRICATED, N_DISTRACTOR,
    FAB_OBJECTS, FAB_PLACES, _build_life, _cue,
)

SEEDS = (0, 1, 2)
JUNK = ["please", "kindly", "sometime", "earlier"]   # all off-vocabulary, none in _STOP


def variant_recall(mem, query, now, arm, top_k=1):
    """The five scoring variants through one code path, mirroring
    EpisodicMemory.recall's loop so the only difference IS the floor rule."""
    from EpisodicMemory import _tokens
    q = _tokens(query)
    if not q:
        return []
    vocab = set().union(*mem._tok) if mem._tok else set()
    q_known = q & vocab
    wr, wi, ws = mem.w
    cands = []
    for ev, tok in zip(mem.events, mem._tok):
        inter_raw = len(q & tok)
        if inter_raw == 0:
            continue
        sim_raw = inter_raw / len(q)
        sim_cov = (len(q_known & tok) / len(q_known)) if q_known else 0.0
        if arm == "A0" and sim_raw < 0.34:
            continue
        if arm == "A1" and sim_raw < 0.60:
            continue
        if arm == "A2" and sim_cov < 0.95:
            continue
        if arm == "A3" and sim_raw < 0.34:
            continue
        if arm == "A4" and (sim_cov < 0.95 or sim_raw < 0.34):
            continue
        sim = sim_cov if arm in ("A2", "A4") else sim_raw
        import math
        age = max(0.0, now - ev.t)
        recency = math.pow(0.5, age / mem.half_life_s)
        importance = min(max(ev.importance, 0.0), 10.0) / 10.0
        cands.append((wr * recency + wi * importance + ws * sim, sim, ev))
    cands.sort(key=lambda c: c[0], reverse=True)
    if arm == "A3" and len(cands) >= 2 and cands[0][1] - cands[1][1] < 0.20:
        return []
    return cands[:top_k]


def measure(seed: int, arm: str) -> dict:
    tmp = Path(tempfile.mkdtemp()) / "life.jsonl"
    mem, events, now, _ = _build_life(seed, tmp)
    rng = random.Random(seed + 1)

    sampled = rng.sample(events, N_QUERIES)
    hits = verbose_hits = 0
    for ev, words in sampled:
        cue = _cue(rng, words)
        res = variant_recall(mem, cue, now, arm)
        hits += bool(res and res[0][2].eid == ev.eid)
        vres = variant_recall(mem, cue + " " + " ".join(JUNK), now, arm)
        verbose_hits += bool(vres and vres[0][2].eid == ev.eid)

    fab = 0
    for _ in range(N_FABRICATED):
        cue = (f"the thing about the {rng.choice(FAB_OBJECTS)} "
               f"and the {rng.choice(FAB_PLACES)}")
        fab += not variant_recall(mem, cue, now, arm)

    terse = 0
    for ev, words in rng.sample(events, 60):
        res = variant_recall(mem, f"the {words[0]}", now, arm)
        terse += bool(res and words[0] in res[0][2].text)

    # Distractor control, ME.1's own construction.
    tmp2 = Path(tempfile.mkdtemp()) / "life_d.jsonl"
    held = set(random.Random(seed + 7).sample(range(N_EVENTS), N_DISTRACTOR))
    mem2, events2, now2, held_words = _build_life(seed, tmp2, skip=held)
    retained = [w for _, w in events2]
    rng2 = random.Random(seed + 8)
    d_abst = d_eval = 0
    for words in held_words:
        cue_words = rng2.sample(list(words), 3)
        if any(set(cue_words) <= set(rw) for rw in retained):
            continue
        cue = (f"the thing about the {cue_words[0]} and the "
               f"{cue_words[1]} {cue_words[2]}")
        d_eval += 1
        d_abst += not variant_recall(mem2, cue, now2, arm)

    return {"cued_recall": hits / N_QUERIES,
            "verbose_recall": verbose_hits / N_QUERIES,
            "fabricated_abst": fab / N_FABRICATED,
            "distractor_abst": (d_abst / d_eval) if d_eval else 0.0,
            "distractor_eval": d_eval,
            "terse_answer": terse / 60}


def main():
    arms = ("A0", "A1", "A2", "A3", "A4")
    print(f"{'arm':4} {'seed':4} {'cued':>6} {'verb':>6} {'fab':>6} "
          f"{'distr':>6} {'d_ev':>4} {'terse':>6}")
    table = {}
    for arm in arms:
        rows = [measure(s, arm) for s in SEEDS]
        table[arm] = rows
        for s, r in zip(SEEDS, rows):
            print(f"{arm:4} {s:4} {r['cued_recall']:6.3f} {r['verbose_recall']:6.3f} "
                  f"{r['fabricated_abst']:6.3f} {r['distractor_abst']:6.3f} "
                  f"{r['distractor_eval']:4d} {r['terse_answer']:6.3f}")
    print()
    for arm in arms:
        rows = table[arm]
        ok = all(r["cued_recall"] >= 0.80 and r["fabricated_abst"] >= 0.95
                 and r["distractor_abst"] >= 0.95 and r["distractor_eval"] >= 30
                 for r in rows)
        worst_v = min(r["verbose_recall"] for r in rows)
        worst_t = min(r["terse_answer"] for r in rows)
        print(f"{arm}: conjuncts {'PASS' if ok else 'fail'}  "
              f"worst verbose {worst_v:.3f}  worst terse {worst_t:.3f}")
    print("EXIT 0", flush=True)


if __name__ == "__main__":
    main()
