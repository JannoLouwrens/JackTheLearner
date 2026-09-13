"""LG.13 — The chooser earns the Language-routing seat, or the seat stays vacant.

THE SEAT. `docs/CHAMPIONS.md` opened *Language routing (what he says, and which
task a command becomes)* on 2026-09-13 and recorded it **VACANT**, with three
red arena members (`LG.10`, `LG.12`, `T2.15`) arbitrated by nothing. SYSTEM.md's
standing rule is that no architectural seat may be held without a registered
EXISTING challenger; law 3 says a fork whose arms can all be run is not an
escalation but an experiment nobody has written yet. The incumbent — a plain
softmax over every gate-passed candidate — has chosen Jack's words since `LG.10`
was written and has never been raced against anything.

THE VENUE is `LG.10`'s rig, imported and not re-derived (the SO.10/SO.08
import precedent). One life per seed in `ME.9`'s store, a candidate pool of
3 phrasings x the intent fact + 3 x 4 truthful distractors + 2 phatic lines
+ 2 fabrications the verification gate must reject, and 1588 frozen
log-probabilities from two SmolLM2 mouths, cached and content-hash keyed.

WHY FOUR RULES CAN SHARE ONE RIG, and why that is structural rather than lucky.
`lg_10._measure(select_fn=)` is the only thing that varies: the trials, the
pool, the verification gate, the prompts, the scores and every metric are built
before the selector is consulted, and the selector is handed `[(logprob,
utterance, meaning)]` plus its own seeded `random.Random`. The core's intent is
NEVER passed to it. So every arm reads a byte-identical candidate table and
differs only in what it makes of it — which is also what licenses
`gate_mode="screen"`: these are OBSERVABLES, so an arm below the gate is a
property of the RULE and not evidence that its run broke.

THE ARMS — all at LG.10's registered TEMP = 1.0, differing only in DECODE
STRUCTURE, which is a ladder from "the score decides everything" to "the score
decides only what is admitted":

    softmax-full    softmax(logprob/TEMP) over the WHOLE pool. THE INCUMBENT,
                    imported as `lg_10._draw` itself so the champion competes
                    as the code it is, not as a copy of it.
    topk-softmax    the same softmax restricted to the K highest-scoring
                    candidates. The score still weights, but the tail cannot
                    be reached at all.
    topk-uniform    uniform over those same K. The score ADMITS and nothing
                    more; wording is free inside the admitted set.
    meaning-mass    softmax mass aggregated per MEANING picks the CONTENT,
                    then wording is drawn uniformly among that meaning's
                    phrasings. This is the arm GOAL.md's sentence describes
                    most literally — *"Jack chooses what to say; the LLM only
                    chooses how"* — with the two decisions made separately
                    instead of jointly. It still does not know the intent.

THE TEMPERATURE KNOB IS DELIBERATELY NOT AN ARM, and this is the one design
choice a reader should check. `lg10-mouth-fidelity-vs-freedom` (Review DAILY
2026-09-08) ruled *"do not fit T — both endpoints are already paid for"*:
T=0.25 reads match 0.9833/1.0/1.0 and VOIDs on the variety floor (the sampler
had no measured freedom), T=1.0 reads variety 1.0 and FAILs on match
(0.60/0.78/0.70). Racing T would re-buy a measured answer and would be
knob-fitting on a failing spec. The same disposition named the legitimate
direction — *"the mouth needs a stronger chooser: bigger frozen model,
structured decode"* — and structured decode is the half that costs nothing.

GREEDY ARGMAX IS EXCLUDED, and it is named here so the omission is not silent.
It is INELIGIBLE BY CONSTRUCTION: one utterance drawn five times gives variety
0.0 on every trial, under a floor of 0.30. It also carries the fewest tunable
constants, so under a cost tie-break it would take the seat and then fail
admission — which is SO.10's finding, and entering an arm in order to reproduce
a known finding is manufacturing, not measuring.

THE CONTROL (entered through `run_bakeoff(controls=)`, scored on the same
ruler, never competing): THE STATE-FREE PROMPT — LG.10's own declared null,
`NULL_ASK`, with no core-selected intent in it, read under the incumbent
selector. If it CLEARS the 3-sigma gate the verdict inverts to VOID: the
scaffold would be leaking the intent, and then no arm's meaning-match is
evidence about its chooser.

THE NULL is SCORE-BLIND UNIFORM selection over the identical gate-passed pool —
the chooser removed entirely, which is the quantity this race is about. Chance
is ~3/17 = 0.18 per draw.

ELIGIBILITY IS NOT THE SCORE. SYSTEM.md's SCORED-AND-INELIGIBLE rule (owner
ruling 2026-08-24) says an arm that cannot be seated is still measured and
still recorded, so every arm's number goes in the row whatever happens to it.
On top of the score sits the seat's admission test, and it is three of LG.10's
own gates applied per arm:

    variety    >= VARIETY_MIN on every seed. A chooser that buys meaning-match
               by killing the sampler's freedom has made LG.10's whole
               invariance question vacuous — that is what VOIDed attempt 1.
    null_ok    the state-free prompt still reads <= NULL_MATCH_MAX under THIS
               selector, on BOTH models. Sharpening is not free: a rule that
               also makes the intent-free prompt track state has stopped
               measuring the core's choosing.
    noleak     leak_draws == 0. No fabricated line ever drawn.

A TIE IS A FAIL HERE, which is STRICTLY HARDER than SO.10's cost-resolved tie
and is pre-registered before any number exists. `run_bakeoff` resolves a tie by
declared cost, and at EQUAL cost `min` returns whichever arm sorted first — an
arbitrary pick. Costs are declared because the primitive requires them, and the
resolution is recorded, but the seat is not awarded on it: "the choice does not
matter yet" is not a title, and this is a seat that was already filled once by
accident. This does NOT pre-empt
`so10-tie-break-hands-the-seat-to-an-ineligible-arm` (DUE 2026-09-17), which
asks whether `bakeoff.py` ITSELF must know about eligibility; nothing in the
primitive is touched here and this clause binds one spec.

NOT ONE BAR HERE WAS CHOSEN BY THIS FILE'S AUTHOR. The learning gate (3 sigma)
and margin (1.5 sigma) are `run_bakeoff`'s defaults. `VARIETY_MIN`,
`NULL_MATCH_MAX`, `LIVENESS_MIN`, `TEMP` and `S_DRAWS` are LG.10 constants
imported unmoved. The one constant this file introduces is `K = N_PHRASINGS`,
DERIVED from the pool's own granularity — three wordings per meaning, so top-K
admits exactly one meaning's worth of candidates — rather than swept.

DETERMINISM, because this family lost a number to it this morning. Every
reduction below is over a `dict` or an explicit sort with the original pool
index as the tie-break; there is no `set()` anywhere in a selector, and no
selector's output depends on `PYTHONHASHSEED`. The lesson is `docs/LESSONS.md`,
*"a ladder metric must be a function of (code, seed, data) and of nothing
else"*, and a bakeoff is the venue where a coin-flip does the most damage.

    VOID  — a rig that could not ask the question: a cached verdict missing
            under any arm; a model failing LG.10's LIVENESS floor; the mouth
            speaking on a nothing-to-report state; a fabrication surviving the
            verification gate; or the CONTROL clearing the gate, which inverts
            the verdict. Also `run_bakeoff`'s own VOIDs (fewer than
            MIN_FINISHERS arms cleared the gate).
    FAIL  — the bakeoff reached no outright decision (a TIE), or reached one
            whose winner is INELIGIBLE on any seed. Either way no chooser is
            seated and the seat stays VACANT. A high score is not a title.
    PASS  — an outright WINNER, eligible on every seed. That rule is seated in
            docs/CHAMPIONS.md BY VERDICT with this row as the deciding run.

WHAT A PASS DOES NOT BUY: it changes nothing in `LG.10` — not a bar, not its
FAIL, not its standing measurement that at honest sampler freedom the frozen
mouth chooses part of the CONTENT. A PASS says a chooser holds the SEAT, not
that the shipped mouth pipeline preserves meaning, and converting one into the
other is the redesign-a-failing-spec move the 09-08 disposition forbade.
Whether `LG.10` is owed a successor under a new champion is the Review's call.
No LLM is loaded and no verdict is bought: the artifact's keys are
content-hashed over (model, revision, scaffold, prompt, candidate), so a
missing verdict is a VOID and never a silent re-purchase.

COVERS: language (parent) (rule).
"""
from __future__ import annotations

import json
import math
import random
import sys

from ..bakeoff import Arm, run_bakeoff
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_10_jack_chooses_what_to_say import (LIVENESS_MIN, N_PHRASINGS,
                                             NULL_MATCH_MAX, S_DRAWS, TEMP,
                                             VARIETY_MIN, _draw, _measure)

IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_10_jack_chooses_what_to_say.py",
             "experiments/bakeoff.py"]

SEEDS = (0, 1, 2)
K = N_PHRASINGS          # 3: one meaning's worth of candidates. Derived from
                         # the pool's own granularity, not swept.

ARM_NAMES = ("softmax-full", "topk-softmax", "topk-uniform", "meaning-mass")
CONTROL_NAME = "state-free-prompt"
NULL_NAME = "uniform"
# Every selector that must actually be executed on the rig. The control is NOT
# here: it is a different METRIC read off the incumbent's own cell.
SELECTORS_RUN = ARM_NAMES + (NULL_NAME,)


# ── shared plumbing: deterministic, no set(), no hash-order anywhere ─────────
def _by_score(scored: list) -> list:
    """Indices of `scored`, best first, ties broken by ORIGINAL POOL ORDER.

    The tie-break is a fact about the pool the rig built, which is a function
    of (code, seed, data). Sorting by the utterance string instead would make
    the mouth's choice a fact about English spelling."""
    return sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))


def _weights(scored: list) -> list:
    """softmax(logprob / TEMP), shift-stabilised exactly as `lg_10._draw`."""
    mx = max(s for s, _u, _m in scored)
    return [math.exp((s - mx) / TEMP) for s, _u, _m in scored]


def _pick(cands: list, rng: random.Random):
    """Uniform choice over `cands`, by index so the draw is a function of the
    seeded rng and the list order and of nothing else."""
    u, m = cands[rng.randrange(len(cands))][1:]
    return u, m


# ── the four rules and the null ─────────────────────────────────────────────
# `softmax-full` IS `lg_10._draw`, imported: the incumbent competes as the code
# it is. Its `__name__` is `_draw`, which is also the key `lg_10._MEMO` uses for
# the default path, so the champion and the shipped path share one cell by
# construction and cannot silently diverge.

def _sel_topk_softmax(scored: list, rng: random.Random):
    """The score still weights, but the tail cannot be reached at all."""
    top = [scored[i] for i in _by_score(scored)[:K]]
    return _draw(top, rng)


def _sel_topk_uniform(scored: list, rng: random.Random):
    """The score ADMITS and nothing more; wording is free inside the set."""
    top = [scored[i] for i in _by_score(scored)[:K]]
    return _pick(top, rng)


def _sel_meaning_mass(scored: list, rng: random.Random):
    """Content by aggregated mass, wording uniform within it.

    The two decisions GOAL.md separates, separated: softmax mass is summed per
    MEANING (a `dict`, so insertion order is pool order and `max` keeps the
    first maximum — the same first-appearance tie-break `lg_10._modal` adopted
    this morning), the heaviest meaning is chosen, and the wording is then
    drawn uniformly among that meaning's phrasings. The selector is never told
    what the core's intent was, so meaning-match is not true by construction."""
    ws = _weights(scored)
    mass: dict = {}
    for w, (_s, _u, m) in zip(ws, scored):
        mass[m] = mass.get(m, 0.0) + w
    best = max(mass, key=mass.get)
    return _pick([t for t in scored if t[2] == best], rng)


def _sel_uniform(scored: list, rng: random.Random):
    """THE NULL: the chooser removed. The scores are accepted and dropped,
    which is the whole point."""
    return _pick(scored, rng)


SELECTORS = {
    "softmax-full": _draw,
    "topk-softmax": _sel_topk_softmax,
    "topk-uniform": _sel_topk_uniform,
    "meaning-mass": _sel_meaning_mass,
    NULL_NAME: _sel_uniform,
}

# Cost, in the unit the spec named: tunable constants the rule carries. RECORDED
# and reported; the seat is NOT awarded on it, because a TIE is a FAIL here.
COSTS = {"softmax-full": 1.0,       # TEMP
         "topk-softmax": 2.0,       # TEMP, K
         "topk-uniform": 1.0,       # K
         "meaning-mass": 1.0,       # TEMP
         NULL_NAME: 0.0,
         CONTROL_NAME: 1.0}         # TEMP (it is the incumbent selector)


def _cell(name: str, seed: int) -> dict:
    """One selector on one seed, straight off LG.10's rig. `lg_10._MEMO` is
    keyed by (seed, selector name), so each cell is computed once per process
    and two choosers can never read each other's."""
    return _measure(seed, select_fn=SELECTORS[name])


def _match_both(name: str, seed: int) -> float:
    """THE METRIC: meaning-match averaged over both frozen mouths. Both, not
    one, because a chooser that holds this seat has to work for the mouth it
    was not tuned against — LG.10's swap is the whole point of having two."""
    c = _cell(name, seed)
    return (c["meaning_tracks_state_not_model"] + c["match_swap"]) / 2.0


def _null_both(name: str, seed: int) -> float:
    """The same ruler read on the STATE-FREE prompt under the same selector."""
    c = _cell(name, seed)
    return (c["null_match"] + c["null_match_swap"]) / 2.0


def _elig(name: str, seed: int) -> dict:
    """The seat's three admission legs, per arm per seed."""
    c = _cell(name, seed)
    variety = float(c["variety"] >= VARIETY_MIN)
    null_ok = float(c["null_match"] <= NULL_MATCH_MAX
                    and c["null_match_swap"] <= NULL_MATCH_MAX)
    noleak = float(c["leak_draws"] == 0)
    return {"variety_ok": variety, "null_ok": null_ok, "noleak": noleak,
            "elig": float(variety and null_ok and noleak)}


_MEMO: dict = {}


def _bakeoff() -> dict:
    """Run the race once and flatten it into recordable floats."""
    if _MEMO:
        return _MEMO
    spec = BY_ID["LG.13"]
    arms = [Arm(n, (lambda nm: lambda s: _match_both(nm, s))(n),
                description=f"decode rule {n}", cost=COSTS[n])
            for n in ARM_NAMES]
    controls = [Arm(CONTROL_NAME,
                    lambda s: _null_both("softmax-full", s),
                    description="LG.10's state-free prompt, incumbent selector",
                    cost=COSTS[CONTROL_NAME])]
    res = run_bakeoff(spec, arms, lambda s: _match_both(NULL_NAME, s),
                      seeds=list(SEEDS), controls=controls, ledger=None)

    out: dict = {
        "verdict_winner": float(res.verdict == "WINNER"),
        "verdict_tie": float(res.verdict == "TIE"),
        "null_mean": round(res.null_mean, 4),
        "null_std": round(res.null_std, 4),
        "winner_idx": float(ARM_NAMES.index(res.winner))
                      if res.winner in ARM_NAMES else -1.0,
        "k": float(K), "temp": float(TEMP), "s_draws": float(S_DRAWS),
    }
    for a in res.arms:
        nm = a.name.replace("control:", "ctl_")
        out[f"{nm}_mean"] = round(a.mean, 4)
        out[f"{nm}_sigma"] = round(a.sigma_over_null, 3)
        out[f"{nm}_gate"] = float(a.passed_gate)
        out[f"{nm}_cost"] = float(a.cost if a.cost is not None else -1.0)
    for n in SELECTORS_RUN:
        for s in SEEDS:
            c, e = _cell(n, s), _elig(n, s)
            out[f"{n}_match_both_s{s}"] = round(_match_both(n, s), 4)
            out[f"{n}_null_both_s{s}"] = round(_null_both(n, s), 4)
            for k in ("meaning_tracks_state_not_model", "match_swap",
                      "unanimity", "unanimity_swap", "swap_agree", "variety",
                      "liveness", "liveness_swap", "null_match",
                      "null_match_swap", "leak_draws", "speak_silence",
                      "gate_rejected_fab_frac", "verdicts_missing",
                      "style_change"):
                out[f"{n}_{k}_s{s}"] = c[k]
            for k in ("variety_ok", "null_ok", "noleak", "elig"):
                out[f"{n}_{k}_s{s}"] = e[k]
    # The winner's eligibility, read once so `_check` is a static row read.
    w = res.winner if res.winner in ARM_NAMES else None
    for s in SEEDS:
        out[f"winner_elig_s{s}"] = _elig(w, s)["elig"] if w else 0.0
    _MEMO.update(out)
    return _MEMO


def _experiment(seed: int) -> dict:
    return dict(_bakeoff())


def _control(seed: int) -> dict:
    """The state-free prompt, surfaced as run_spec's control so the row carries
    it under `control_metrics` as well as inside the bakeoff."""
    b = _bakeoff()
    return {f"ctl_null_both_s{s}": b[f"softmax-full_null_both_s{s}"]
            for s in SEEDS} | {
        "ctl_sigma": b.get(f"ctl_{CONTROL_NAME}_sigma", 0.0),
        "ctl_gate": b.get(f"ctl_{CONTROL_NAME}_gate", 0.0),
    }


_NEED_M = (("verdict_winner", "verdict_tie", "winner_idx", "null_mean")
           + tuple(f"winner_elig_s{s}" for s in SEEDS)
           + tuple(f"{n}_{k}_s{s}" for n in SELECTORS_RUN
                   for k in ("match_both", "elig", "variety", "liveness",
                             "liveness_swap", "leak_draws", "speak_silence",
                             "gate_rejected_fab_frac", "verdicts_missing")
                   for s in SEEDS))
_NEED_C = ("ctl_sigma", "ctl_gate") + tuple(f"ctl_null_both_s{s}"
                                            for s in SEEDS)


def _check(m: dict, c: dict):
    """Pure function of the recorded row — every read static, all read up front
    so each is consulted on every replay."""
    if any(k not in m for k in _NEED_M) or any(k not in c for k in _NEED_C):
        return Status.VOID
    missing = [m[f"{n}_verdicts_missing_s{s}"]
               for n in SELECTORS_RUN for s in SEEDS]
    live = [m[f"{n}_{k}_s{s}"] for n in SELECTORS_RUN for s in SEEDS
            for k in ("liveness", "liveness_swap")]
    silence = [m[f"{n}_speak_silence_s{s}"] for n in SELECTORS_RUN
               for s in SEEDS]
    fab = [m[f"{n}_gate_rejected_fab_frac_s{s}"] for n in SELECTORS_RUN
           for s in SEEDS]
    ctl_gate = c["ctl_gate"]
    ctl_null = [c[f"ctl_null_both_s{s}"] for s in SEEDS]
    won = m["verdict_winner"]
    tied = m["verdict_tie"]
    widx = m["winner_idx"]
    welig = [m[f"winner_elig_s{s}"] for s in SEEDS]
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if max(missing) != 0:
        return Status.VOID           # a cached verdict absent under some arm
    if min(live) < LIVENESS_MIN:
        return Status.VOID           # a mouth could not rank sense over noise
    if max(silence) != 0.0:
        return Status.VOID           # spoke on a nothing-to-report state
    if min(fab) != 1.0:
        return Status.VOID           # a fabrication survived the gate
    if ctl_gate == 1.0 or max(ctl_null) > NULL_MATCH_MAX:
        return Status.VOID           # the scaffold leaks the intent
    # ── the claim: an OUTRIGHT winner whose arm may hold the seat ──
    # A TIE is a FAIL by pre-registration, never a VOID: the race ran and its
    # answer was "the choice does not matter yet", which is a finding about the
    # choosers and not a broken rig.
    if tied == 1.0:
        return False
    return bool(won == 1.0 and widx >= 0 and min(welig) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.13"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        for n in SELECTORS_RUN:
            e = _elig(n, 0)
            print(f"smoke {n:14s} match_both={_match_both(n, 0):.4f} "
                  f"null_both={_null_both(n, 0):.4f} elig={e['elig']:.0f}")
    else:
        print(json.dumps(_bakeoff(), indent=2, sort_keys=True))
