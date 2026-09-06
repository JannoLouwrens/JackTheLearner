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

THIRD SCAR, added 2026-09-06 per the queue row's own precondition ("any
recalibration must add ME.3's cue shape to the probe's arms before
adoption"). The adopted A2 was chosen by a probe that never measured a
DISJUNCTIVE cue, and ME.3 attempt 4 (commit 42ad5c9) paid for it: its raw
arm's cue is `" ".join([speaker] + candidates)` — 5 tokens, ALL known to the
store, mutually exclusive by construction (an event carries the speaker plus
at most ONE candidate, so best-event coverage is capped at 2/5 = 0.4) — and
the 0.95 coverage floor abstains on EVERY question (raw_tokens 40 -> 0.0,
raw_acc 0.625 -> 0.2917 vs base 0.25; the equal-tokens gate refused the
starved null). This section adds two measurements:

  disjunctive   ME.3's exact construction (its `_build_life`, `_questions`,
                `_pack`, `_read`, its token budget) scored per arm. Bars,
                pre-stated before the run: disj_answer >= 0.95 (the store
                must surface evidence on essentially every question) and
                disj_acc >= 0.45 on all seeds (A0 measured 0.625 on this
                harness at attempts 2-3; base rate 0.25; the band allows
                seed noise without admitting chance).
  separability  the statistic every floor in this family thresholds on is
                bestcov(q) = max over events of |q_known ∩ tok|/|q_known|.
                A floor answers a disjunctive cue only if it sits AT OR
                BELOW that cue's bestcov, and abstains on a distractor cue
                only if it sits ABOVE that cue's bestcov. So a floor serving
                both exists iff max(bestcov over distractor cues) <
                min(bestcov over disjunctive cues) — ON THE SAME STORE. The
                probe prints both distributions and the gap. A negative gap
                is a MEASURED impossibility for every monotone floor on this
                statistic, not an argument: the abstain-required cues score
                HIGHER than the answer-required cues on the only number the
                scorer sees. MEASURED 2026-09-06: gap −0.267 on all three
                seeds (distractor bestcov 0.667 exactly, disjunctive 0.400
                exactly), overlap 1.000. No single-cue floor arm can pass
                both; A0–A4 confirm it empirically below.

A5, added after that measurement and REFUSED BY `decisions.py` as an owner
escalation first (MEANS-ESCALATED: a means fork is settled by bakeoff, not by
the owner — law 3, enforced): the fork the gap leaves open is not a floor
value, it is WHERE THE ASKER'S INTENT LIVES. The two cue populations are
identical to the scorer (bags of known words that never co-occur in one
event); what differs is that "the thing about X and Y" asserts a conjunction
and "was it A or B or C?" lists alternatives — and intent is not recoverable
from a token bag. A5 moves it to the call site:

  A5 alt-declared   single cues score exactly as A2 (coverage 0.95 — nothing
                    is declared, nothing changes). A cue that ARRIVES as
                    alternatives is scored per alternative: each candidate
                    becomes its own sub-cue (speaker + candidate), each
                    sub-cue is a CONJUNCTION under the same 0.95 coverage
                    floor, results union-ranked. Abstention is preserved
                    per sub-cue by construction — a never-lived pairing
                    still clears nothing.

Decision rule for the third scar, pre-stated: an adoptable mechanism must
clear ME.1's four conjunct analogues AND disj_answer >= 0.95 AND disj_acc
>= 0.45 on all seeds. If only A5 survives, the bakeoff's verdict is that the
CONTRACT SPLIT is the mechanism — single-cue recall keeps ME.1's abstention
contract, OR-intent must be declared at the call site — and what remains is
NOT a module calibration: adopting it means ME.3's harness declares its
alternatives instead of packing them into one string, which is a spec
redesign and routes through the Review on the queue row. No EpisodicMemory.py
edit happens from this probe; the module and its 16 declared certificates do
not move.

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
    """The scoring variants through one code path, mirroring
    EpisodicMemory.recall's loop so the only difference IS the floor rule.
    A5 without declared alternatives IS A2 — the contract split only changes
    behaviour at a call site that declares them (see recall_alternatives)."""
    from EpisodicMemory import _tokens
    if arm == "A5":
        arm = "A2"
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


def recall_alternatives(mem, speaker, cands, now, top_k=32):
    """A5's declared-alternatives path: each candidate is its own
    conjunctive sub-cue under the SAME A2 floor, results union-ranked.
    A never-lived (speaker, candidate) pairing still clears nothing —
    abstention survives per sub-cue by construction."""
    by_eid = {}
    for cand in cands:
        for c in variant_recall(mem, f"{speaker} {cand}", now, "A2",
                                top_k=top_k):
            eid = c[2].eid
            if eid not in by_eid or c[0] > by_eid[eid][0]:
                by_eid[eid] = c
    out = sorted(by_eid.values(), key=lambda c: c[0], reverse=True)
    return out[:top_k]


def measure_disjunctive(seed: int, arm: str) -> dict:
    """ME.3's cue shape through each arm: speaker + 4 mutually-exclusive
    candidates, all words known, scored by ME.3's own packer and reader."""
    import random
    from experiments.tests.me_3_reflections import (
        _build_life as _build_me3, _questions, _pack, _read,
    )
    tmp = Path(tempfile.mkdtemp()) / "life3.jsonl"
    mem, tally, now = _build_me3(seed, tmp)
    questions = _questions(seed, tally)
    rng = random.Random(seed + 3)
    answered = hits = tok_sum = 0
    for s, cands, truth in questions:
        if arm == "A5":
            res = recall_alternatives(mem, s, cands, now, top_k=32)
        else:
            cue = " ".join([s] + cands)
            res = variant_recall(mem, cue, now, arm, top_k=32)
        lines, used = _pack([c[2].text for c in res])
        answered += bool(lines)
        tok_sum += used
        hits += _read(lines, cands, rng) == truth
    n = len(questions)
    return {"disj_answer": answered / n, "disj_acc": hits / n,
            "disj_tokens": tok_sum / n, "n_questions": n}


def separability(seed: int) -> dict:
    """bestcov(q) distributions for the two cue populations, SAME store:
    ME.3's disjunctive questions (must answer) vs ME.3's own distractor
    construction (must abstain). A monotone floor serving both exists iff
    max(distractor bestcov) < min(disjunctive bestcov)."""
    import random
    from EpisodicMemory import _tokens
    from experiments.tests.me_3_reflections import (
        _build_life as _build_me3, _questions,
        OBJECTS, PLACES, COLOURS, ACTIONS, N_DISTRACTOR,
    )
    tmp = Path(tempfile.mkdtemp()) / "life3s.jsonl"
    mem, tally, now = _build_me3(seed, tmp)
    vocab = set().union(*mem._tok)

    def bestcov(cue: str) -> float:
        q_known = _tokens(cue) & vocab
        if not q_known:
            return 0.0
        return max((len(q_known & tok) / len(q_known) for tok in mem._tok),
                   default=0.0)

    disj = [bestcov(" ".join([s] + cands))
            for s, cands, _ in _questions(seed, tally)]

    stored = mem._tok
    rng = random.Random(seed + 11)          # ME.3's own distractor seeding
    distr = []
    for _ in range(N_DISTRACTOR):
        combo = (rng.choice(OBJECTS), rng.choice(PLACES),
                 rng.choice(COLOURS), rng.choice(ACTIONS))
        picks = rng.sample(combo, 3)
        pick_set = set(picks)
        if not pick_set <= vocab or any(pick_set <= s for s in stored):
            continue
        distr.append(bestcov(
            f"the thing about the {picks[0]} and the {picks[1]} {picks[2]}"))

    gap = min(disj) - max(distr) if disj and distr else float("nan")
    overlap = (sum(d >= min(disj) for d in distr) / len(distr)
               if disj and distr else float("nan"))
    return {"disj_min": min(disj), "disj_max": max(disj),
            "disj_mean": sum(disj) / len(disj),
            "distr_min": min(distr), "distr_max": max(distr),
            "distr_mean": sum(distr) / len(distr),
            "n_disj": len(disj), "n_distr": len(distr),
            "gap": gap, "distr_at_or_above_disj_min": overlap}


def main():
    arms = ("A0", "A1", "A2", "A3", "A4", "A5")
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
    print(f"{'arm':4} {'seed':4} {'d_ans':>6} {'d_acc':>6} {'d_tok':>6} {'n_q':>4}")
    dtable = {}
    for arm in arms:
        drows = [measure_disjunctive(s, arm) for s in SEEDS]
        dtable[arm] = drows
        for s, r in zip(SEEDS, drows):
            print(f"{arm:4} {s:4} {r['disj_answer']:6.3f} {r['disj_acc']:6.3f} "
                  f"{r['disj_tokens']:6.1f} {r['n_questions']:4d}")
    print()
    print("separability (same store: distractor cues must abstain, "
          "disjunctive cues must answer; a floor serving both needs gap > 0)")
    for s in SEEDS:
        sep = separability(s)
        print(f"seed {s}: disj bestcov [{sep['disj_min']:.3f}, "
              f"{sep['disj_mean']:.3f}, {sep['disj_max']:.3f}] n={sep['n_disj']}  "
              f"distr bestcov [{sep['distr_min']:.3f}, {sep['distr_mean']:.3f}, "
              f"{sep['distr_max']:.3f}] n={sep['n_distr']}  "
              f"gap {sep['gap']:+.3f}  "
              f"distr>=disj_min {sep['distr_at_or_above_disj_min']:.3f}")
    print()
    for arm in arms:
        rows, drows = table[arm], dtable[arm]
        ok = all(r["cued_recall"] >= 0.80 and r["fabricated_abst"] >= 0.95
                 and r["distractor_abst"] >= 0.95 and r["distractor_eval"] >= 30
                 for r in rows)
        d_ok = all(r["disj_answer"] >= 0.95 and r["disj_acc"] >= 0.45
                   for r in drows)
        worst_v = min(r["verbose_recall"] for r in rows)
        worst_t = min(r["terse_answer"] for r in rows)
        print(f"{arm}: conjuncts {'PASS' if ok else 'fail'}  "
              f"disjunctive {'PASS' if d_ok else 'fail'}  "
              f"worst verbose {worst_v:.3f}  worst terse {worst_t:.3f}")
    print("EXIT 0", flush=True)


if __name__ == "__main__":
    main()
