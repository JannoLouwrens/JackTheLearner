"""SO.10 — The trust rule earns its seat, or the seat stays vacant.

THE SEAT. `docs/CHAMPIONS.md` opened *Person model (trust, attribution, and
whose advice proved true)* on 2026-09-13 and recorded it VACANT with the
sentence that commissions this spec: *"how a person is represented and how
trust updates is a MECHANISM with real arms (per-source Bayesian reliability,
recency-weighted track record, a single scalar, no model at all), and the repo
has picked one by accident."* CHAMPIONS rule 3 gives the match to the builder;
SYSTEM.md law 3 says a fork whose arms can all be run is not an escalation but
an experiment nobody has written yet. The incumbent — a Laplace posterior over
the last WINDOW verified claims — has held the seat since `LG.02` was written
and has never been raced against anything.

THE VENUE is LG.02's certified rig, imported and not re-derived (the SO.08
Laplace-import precedent). Two advisors alternate, one systematically truthful
and one systematically false; every claim is verified by Jack's own subsequent
finding; trust is recomputed FROM THE ATTRIBUTED DIARY at every decision and he
follows with probability equal to it.

WHY FOUR RULES CAN SHARE ONE WORLD, and why this is structural rather than
lucky. The evidence stream is arm-independent by construction: `rng_world` is
seeded from the seed alone, `rng_agent` draws exactly once per round whatever
the rule returns, and the diary records the CLAIM and the FINDING — never
whether he followed. So every arm reads a byte-identical diary and differs only
in what it makes of it. That is also what licenses `gate_mode="screen"`: these
are OBSERVABLES, so an arm below the gate is a property of the RULE (full-
history Laplace cannot migrate; last-claim-only is memoryless) and not evidence
that its run broke.

THE ARMS — all four read the same join, and differ only in how they weight it:

    laplace-w30    (hits+1)/(n+2) over the last WINDOW=30 verified claims.
                   THE INCUMBENT, imported as the shipped `_trust` itself so
                   the champion competes as the code it is, not a copy.
    laplace-full   the same posterior over the WHOLE history. The question it
                   asks is the one GOAL.md asks of every component: does the
                   window earn its parameter?
    exp-decay-h15  exponentially-weighted hit rate, half-life WINDOW/2, with
                   the same Laplace pseudo-counts so a stranger still scores
                   PRIOR exactly. The forgetting is graded instead of a cliff.
    last-1         trust = TRUTH_P if their last verified claim was true else
                   LIE_P. A memoryless person model — win-stay/lose-shift, the
                   cheapest thing that is still per-person.

THE CONTROL (entered through `run_bakeoff(controls=)`, scored on the same ruler,
never competing): POOLED SCALAR — one global trust over every speaker at once,
which is a diary with no person model in it at all. The advisors alternate, so
a rule that cannot tell them apart must not be able to diverge. If it CLEARS
the 3-sigma gate the verdict inverts to VOID: the metric would not be measuring
person-modelling.

THE NULL is LG.02's, reused unchanged: attribution stripped at record time
(speaker "someone", the name scrubbed from the text), incumbent rule, same
world stream, same metric.

ELIGIBILITY IS NOT THE SCORE. SYSTEM.md's SCORED-AND-INELIGIBLE rule (owner
ruling 2026-08-24) says an arm that cannot be seated is still measured and
still recorded, so every arm's number goes in the row whatever happens to it.
On top of the score sits the seat's admission test, and it is three of LG.02's
own gates applied per arm:

    prior_ok   first-encounter trust is exactly PRIOR for BOTH advisors —
               the registry's kills-clause guard against scripted divergence.
    noleak     with attribution stripped, |divergence| <= NULL_DIV_MAX. A rule
               that diverges without a record of who spoke is reading speaker
               identity from somewhere outside the attributed diary.
    migrate    after the roles swap at SWAP_ROUND, divergence toward the
               newly-truthful voice >= MIN_MIGRATE, having been >= MIN_PRESWAP
               toward the originally-truthful one before it. A rule that cannot
               change its mind when the world changes who is honest was
               tracking voices, not veracity.

NOT ONE BAR HERE WAS CHOSEN BY THIS FILE'S AUTHOR. The learning gate (3 sigma)
and the margin (1.5 sigma) are `run_bakeoff`'s defaults. Every eligibility and
rig bar is an LG.02 constant imported unmoved. The one number this file
introduces is `HALF_LIFE = WINDOW // 2`, and it is derived from the incumbent's
own window rather than tuned, so the two forgetting rules integrate comparable
amounts of evidence. There is nothing here a preview of the result could have
moved.

    VOID  — a rig that could not ask the question: a per-seed key the gates
            need is missing; verification incomplete; an advisor was not
            systematic (realized accuracy outside TRUTH_BAND / LIE_BAND); the
            stripped join never ran (pooled trust outside NULL_TRUST_BAND); or
            the CONTROL cleared the gate, which inverts the verdict because the
            metric is then not measuring what the spec claims.
    FAIL  — the bakeoff reached no decision (fewer than MIN_FINISHERS arms
            cleared the gate, or a tie no declared cost could break), or it
            reached one whose winner is INELIGIBLE on any seed. Either way no
            rule is seated and the seat stays VACANT. A high score is not a
            title.
    PASS  — a WINNER or a cost-resolved TIE whose arm is eligible on every
            seed. That arm is seated in docs/CHAMPIONS.md BY VERDICT with this
            row as the deciding run.

WHAT A PASS DOES NOT BUY: it re-opens nothing in LG.02, whose gates are
untouched and whose certificate this spec depends on; and it claims no new
capability for Jack — the COVERS kind is `rule` on purpose, so no commitment's
`n_pass` can move behind it.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ..bakeoff import Arm, run_bakeoff
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lg_02_liar_loses_him import (ADVISORS, LAST_Q, LIE_BAND, LIE_P, MIN_DIV,
                                   MIN_MIGRATE, MIN_PRESWAP, NULL_DIV_MAX,
                                   NULL_TRUST_BAND, N_ROUNDS, PRIOR, SWAP_ROUND,
                                   TRUTH_BAND, TRUTH_P, WINDOW, _follow_rate,
                                   _live, _parse_round, _parse_site, _trust,
                                   _truth_rate)

IMPL_DEPS = ["EpisodicMemory.py",
             "experiments/tests/lg_02_liar_loses_him.py",
             "experiments/bakeoff.py"]

SEEDS = (0, 1, 2)
HALF_LIFE = WINDOW // 2          # 15: derived from the incumbent's window, so
                                 # the two forgetting rules integrate
                                 # comparable evidence. Not tuned.

ARM_NAMES = ("laplace-w30", "laplace-full", "exp-decay-h15", "last-1")
CONTROL_NAME = "pooled-scalar"
ALL_NAMES = ARM_NAMES + (CONTROL_NAME,)


# ── the join, factored out so the non-Laplace rules read the SAME evidence ──
def _verified(mem, speaker: str | None) -> list:
    """The (round, hit) pairs `lg_02._trust` computes, in round order, reduced
    to hits. Identical logic, written once here so an arm cannot quietly
    compete on a different evidence set than the incumbent."""
    findings = {}
    for e in mem.events:
        if e.channel == "did":
            findings[_parse_round(e.text)] = _parse_site(e.text)
    out = []
    for e in mem.events:
        if e.channel != "heard":
            continue
        if speaker is not None and e.speaker.lower() != speaker:
            continue
        r = _parse_round(e.text)
        if r in findings:
            out.append((r, _parse_site(e.text) == findings[r]))
    out.sort()
    return [hit for _, hit in out]


# ── the four rules and the control ──────────────────────────────────────────
def _rule_laplace_full(mem, speaker):
    """The incumbent's posterior with no window at all."""
    return _trust(mem, speaker, window=N_ROUNDS)


def _rule_exp_decay(mem, speaker):
    """Exponentially-weighted hit rate with the incumbent's pseudo-counts, so
    a stranger scores PRIOR exactly and forgetting is graded, not a cliff."""
    hits = _verified(mem, speaker)
    n = len(hits)
    num = den = 0.0
    for i, hit in enumerate(hits):
        w = 0.5 ** ((n - 1 - i) / HALF_LIFE)
        den += w
        num += w * hit
    return (num + 1) / (den + 2)


def _rule_last_1(mem, speaker):
    """Win-stay / lose-shift: the cheapest thing that is still per-person."""
    hits = _verified(mem, speaker)
    if not hits:
        return PRIOR
    return TRUTH_P if hits[-1] else LIE_P


def _rule_pooled(mem, speaker):
    """THE CONTROL: one trust for everyone. `speaker` is accepted and dropped,
    which is the whole point — no person model."""
    return _trust(mem, None)


RULES = {
    "laplace-w30": _trust,
    "laplace-full": _rule_laplace_full,
    "exp-decay-h15": _rule_exp_decay,
    "last-1": _rule_last_1,
    CONTROL_NAME: _rule_pooled,
}

# Cost, in the unit the spec named: tunable constants the rule carries. A TIE
# resolves toward the rule with fewer of them, which is GOAL.md's
# earn-your-parameters rule applied to a decision instead of a module.
COSTS = {"laplace-w30": 1.0,        # WINDOW
         "laplace-full": 0.0,       # none
         "exp-decay-h15": 1.0,      # HALF_LIFE
         "last-1": 2.0,             # TRUTH_P, LIE_P
         CONTROL_NAME: 0.0}


def _measure_rule(name: str, seed: int) -> dict:
    """One rule, one seed: its score and its three eligibility legs."""
    fn = RULES[name]
    tmp = Path(tempfile.mkdtemp())
    truthful, liar = ADVISORS

    mem, rows, first, _ = _live(seed, tmp / "life.jsonl", trust_fn=fn)
    div = _follow_rate(rows, truthful, LAST_Q) - _follow_rate(rows, liar, LAST_Q)

    _, rows_n, _, _ = _live(seed, tmp / "null.jsonl", stripped=True, trust_fn=fn)
    ndiv = _follow_rate(rows_n, truthful, LAST_Q) - _follow_rate(rows_n, liar, LAST_Q)

    a, b = ADVISORS                      # a truthful first half, b second
    _, rows_s, _, _ = _live(seed, tmp / "swap.jsonl", swap=True, trust_fn=fn)
    mig = _follow_rate(rows_s, b, LAST_Q) - _follow_rate(rows_s, a, LAST_Q)
    pre = _follow_rate(rows_s, a, 60, SWAP_ROUND) - _follow_rate(rows_s, b, 60,
                                                                SWAP_ROUND)

    prior_ok = float(first[truthful] == PRIOR and first[liar] == PRIOR)
    noleak = float(abs(ndiv) <= NULL_DIV_MAX)
    migrate = float(mig >= MIN_MIGRATE and pre >= MIN_PRESWAP)
    return {
        "div": round(div, 4),
        "null_div": round(ndiv, 4),
        "null_trust_end": round(rows_n[-1]["trust"], 4),
        "migrate_div": round(mig, 4),
        "preswap_div": round(pre, 4),
        "prior_ok": prior_ok,
        "noleak": noleak,
        "migrate": migrate,
        "elig": float(prior_ok and noleak and migrate),
        # rig facts — arm-independent by construction, recorded per arm so the
        # row proves they were checked under the arm that won, not once.
        "verify_complete": float(sum(1 for e in mem.events
                                     if e.channel == "did") == N_ROUNDS),
        "truth_rate_truthful": round(_truth_rate(rows, truthful), 4),
        "truth_rate_liar": round(_truth_rate(rows, liar), 4),
    }


_CELL: dict = {}                  # (rule, seed) -> metrics; every life runs once
_MEMO: dict = {}                  # the whole bakeoff, once per process


def _cell(name: str, seed: int) -> dict:
    if (name, seed) not in _CELL:
        _CELL[(name, seed)] = _measure_rule(name, seed)
    return _CELL[(name, seed)]


def _bakeoff() -> dict:
    """Run the race once and flatten it into recordable floats."""
    if _MEMO:
        return _MEMO
    spec = BY_ID["SO.10"]
    arms = [Arm(n, (lambda nm: lambda s: _cell(nm, s)["div"])(n),
                description=f"trust rule {n}", cost=COSTS[n]) for n in ARM_NAMES]
    controls = [Arm(CONTROL_NAME,
                    lambda s: _cell(CONTROL_NAME, s)["div"],
                    description="one global trust; no person model",
                    cost=COSTS[CONTROL_NAME])]
    # The null is the incumbent with attribution stripped — LG.02's declared
    # null, scored on this spec's metric.
    res = run_bakeoff(spec, arms, lambda s: _cell("laplace-w30", s)["null_div"],
                      seeds=list(SEEDS), controls=controls, ledger=None)

    out: dict = {
        "verdict_decided": float(res.verdict in ("WINNER", "TIE")),
        "verdict_tie": float(res.verdict == "TIE"),
        "null_mean": round(res.null_mean, 4),
        "null_std": round(res.null_std, 4),
        "winner_idx": float(ALL_NAMES.index(res.winner))
                      if res.winner in ALL_NAMES else -1.0,
    }
    for a in res.arms:
        nm = a.name.replace("control:", "ctl_")
        out[f"{nm}_mean"] = round(a.mean, 4)
        out[f"{nm}_sigma"] = round(a.sigma_over_null, 3)
        out[f"{nm}_gate"] = float(a.passed_gate)
    for n in ALL_NAMES:
        for s in SEEDS:
            c = _cell(n, s)
            for k in ("div", "null_div", "null_trust_end", "migrate_div",
                      "preswap_div", "prior_ok", "noleak", "migrate", "elig",
                      "verify_complete", "truth_rate_truthful",
                      "truth_rate_liar"):
                out[f"{n}_{k}_s{s}"] = c[k]
    # The winner's eligibility, read once so `_check` is a static row read.
    w = res.winner if res.winner in ALL_NAMES else None
    for s in SEEDS:
        out[f"winner_elig_s{s}"] = _cell(w, s)["elig"] if w else 0.0
    _MEMO.update(out)
    return _MEMO


def _experiment(seed: int) -> dict:
    return dict(_bakeoff())


def _control(seed: int) -> dict:
    """The pooled-scalar rule, surfaced as run_spec's control so the row
    carries it under `control_metrics` as well as inside the bakeoff."""
    b = _bakeoff()
    return {"ctl_div_s%d" % s: b[f"{CONTROL_NAME}_div_s{s}"] for s in SEEDS} | {
        "ctl_sigma": b.get(f"ctl_{CONTROL_NAME}_sigma", 0.0),
        "ctl_gate": b.get(f"ctl_{CONTROL_NAME}_gate", 0.0),
    }


_NEED_M = (("verdict_decided", "winner_idx", "null_mean")
           + tuple(f"winner_elig_s{s}" for s in SEEDS)
           + tuple(f"{n}_{k}_s{s}" for n in ALL_NAMES
                   for k in ("div", "elig", "null_trust_end", "verify_complete",
                             "truth_rate_truthful", "truth_rate_liar")
                   for s in SEEDS))
_NEED_C = ("ctl_sigma", "ctl_gate") + tuple(f"ctl_div_s{s}" for s in SEEDS)


def _in_band(vals, band) -> bool:
    return all(band[0] <= v <= band[1] for v in vals)


def _check(m: dict, c: dict):
    """Pure function of the recorded row — every read static, all read up
    front so each is consulted on every replay."""
    if any(k not in m for k in _NEED_M) or any(k not in c for k in _NEED_C):
        return Status.VOID
    verify = [m[f"{n}_verify_complete_s{s}"] for n in ALL_NAMES for s in SEEDS]
    tr_t = [m[f"{n}_truth_rate_truthful_s{s}"] for n in ALL_NAMES for s in SEEDS]
    tr_l = [m[f"{n}_truth_rate_liar_s{s}"] for n in ALL_NAMES for s in SEEDS]
    # The stripped-join aliveness proof is read from the INCUMBENT only.
    # NULL_TRUST_BAND is LG.02's band on a Laplace posterior over a ~50/50
    # pooled stream; `last-1` reports TRUTH_P or LIE_P there by construction,
    # so asking it of every arm would turn a rule's own shape into a rig VOID
    # — the 23rd-audit instrument-gating lesson pointed the wrong way.
    ntr = [m[f"laplace-w30_null_trust_end_s{s}"] for s in SEEDS]
    ctl_div = [c[f"ctl_div_s{s}"] for s in SEEDS]
    ctl_gate = c["ctl_gate"]
    decided = m["verdict_decided"]
    widx = m["winner_idx"]
    welig = [m[f"winner_elig_s{s}"] for s in SEEDS]
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if min(verify) != 1.0:
        return Status.VOID              # a claim with nothing to join to
    if not (_in_band(tr_t, TRUTH_BAND) and _in_band(tr_l, LIE_BAND)):
        return Status.VOID              # an advisor was not systematic
    if not _in_band(ntr, NULL_TRUST_BAND):
        return Status.VOID              # the stripped join never actually ran
    if ctl_gate == 1.0 or max(ctl_div) >= MIN_DIV:
        return Status.VOID              # the control escaped: the metric is
                                        # not measuring person-modelling
    # ── the claim: a decision was reached AND its winner may hold the seat ──
    return bool(decided == 1.0 and widx >= 0 and min(welig) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.10"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        import experiments.tests.lg_02_liar_loses_him as lg
        lg.N_ROUNDS, lg.SWAP_ROUND, lg.LAST_Q = 8, 4, 6
        for n in ALL_NAMES:
            _measure_rule(n, 0)
        print(f"smoke: {len(ALL_NAMES)} rules x 3 toy lives, no error")
    else:
        print(json.dumps(_bakeoff(), indent=2, sort_keys=True))
