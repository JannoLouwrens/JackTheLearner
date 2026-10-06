"""W1.04 — The horizon is longer than the consequence.

THE QUESTION. BA.03 and LF.01 are one defect seen from two ends: a blind
twin holding 98.9% of a 12.0 s horizon and a forager dying of integrity at
~25 min are both "the window closed before the thing we are claiming had
time to happen". The Review FULL's published W1 design (REVIEW_QUEUE
`w0-too-shallow`, 2026-09-06, STRENGTHENED 2026-09-10 with conjunct (c))
turns that into a world-fidelity gate over the registered W1 claims: the
windows those claims declare must be real — long enough for the claimed
mechanism (conjunct a), and actually survivable by the lives that are
supposed to fill them (conjunct c). Under the `w0-too-shallow` row's
2026-10-02 split this spec RUNS on W0 as built; the world edit is needed to
PASS, not to RUN. The registered prediction follows from numbers already on
the ledger: lives in this venue end by energy death at ~52–60 s while the
declared horizon is 5600 s, so conjunct (c) is expected red — a FAIL here
is the registered, gated form of LF.01's "the window is fiction", not an
error to repair by softening. Conjunct (c) forbids the one repair that
would be a lie: shortening the horizon to fit the deaths.

THE QUANTIFICATION SET, declared and guarded. "Every registered W1 claim"
is today {W1.01, W1.02}. W1.00 is registered but runs ZERO lives (it reads
LC.03's recorded artifacts; the `w0-too-shallow` row's own holding text
says conjunct (c) "has nothing to measure on it"), and W1.04 is this spec.
W1.03 is not registered (its twin is not writable against a world with no
traps — 1^17's prohibition). The set is enforced, not assumed: any
registry id matching `W1.*` outside {W1.00, W1.01, W1.02, W1.04} VOIDs the
run ("claim set stale") — if W1.03 ever registers, this spec refuses to
quantify over a set it was not written for, and must be strengthened to
include it rather than silently under-covering. Strengthen-only.

THE HORIZON, imported not copied. Both claims run N_DEC decisions at
SIM_S_PER_DECISION (w1_01 imports N_DEC from w1_02, so the family has ONE
declared horizon by construction — asserted at import): HORIZON_S =
N_DEC * SIM_S_PER_DECISION = 5600.0 s. Reported on the row per the design
("BOTH numbers reported").

TIME-TO-CONSEQUENCE, measured per claim per seed, never typed:

  W1.01 ("Passivity dies"): the claimed consequence is the passive arm's
  DEATH — the world punishing inaction. t2c = the measured mean completed-
  life span of the passive arm (survival.py's statue) at the certified
  envelope. Its analytic anchor is the E0/BASAL_B basal ceiling (~60 s),
  but the MEASURED spans gate, not the formula.

  W1.02 ("Outcomes have resolution"): the claimed mechanism is a per-life
  graded outcome — one completed life is one sample of the statistic. t2c
  = the measured mean completed-life span of the repeat-process arm (the
  process W1.02's own row certifies as its venue), i.e. the time the venue
  needs to produce ONE resolvable outcome.

  Conjunct (a): HORIZON_S / t2c >= RATIO_MIN (3.0, the design's "3x") for
  every claim on every seed.

SURVIVAL, conjunct (c), measured per claim per seed. Spans of ALL lives:
completed spans plus the trailing fragment censored at the horizon (NE.08
conventions via W1.02's C3 — cause-tagged, censored-vs-fatal by tag).
Including censored spans at their censoring time can only UNDERSTATE
survival, so it can make (c) harder, never easier. p5 = 5th percentile
(the design's own statistic — "NOT the mean"). W1.01's lives are BOTH its
arms' (passive and oracle; the claim's statistic reads both), gated at the
worse arm; W1.02's are the repeat arm's. Conjunct (c): p5 >= HORIZON_S for
every claim on every seed. Per-life termination causes ride on the row as
per-cause counts (death_cause() returns "energy" | "integrity"; the
fragment is "censored").

THE TRIVIAL-PASS BRANCH, pre-registered by the falsifier: if measured
time-to-consequence is strictly under a third of the horizon everywhere
AND survival clears, the spec PASSes trivially and its claim_branch says
it should be RETIRED rather than kept as decoration.

THE CONTROL (must show the gate alive — registry: control-red VOIDs,
never FAILs, T0.22): the identical measurement with the terminating
mechanism disabled — the same repeat process, lethal=False, nothing else
touched (the one-flag twin; the local loop is w1_02's verbatim with the
flag exposed, because w1_02's own _rollout pins lethal=True). With death
off the single life runs to the horizon and is censored there, so the
control's p5 EQUALS the horizon — the maximum the measurement can observe,
since the measurement itself censors at HORIZON_S. The gate is therefore
p5_ctrl >= HORIZON_S, equality legal and expected; a control p5 UNDER the
horizon means the survival gate cannot tell a fatal world from a
survivable one, and the run VOIDs.

VOID LANES, in test order (an instrument fault is not a world reading):
  V1 PS.01 borrow unavailable                       "uncalibrated borrow"
  V2 a W1.* id outside the declared set registered  "claim set stale"
  V3 non-finite physics in any run                  "non-finite physics"
  V4 a lethal arm under MIN_LIVES completed lives   "under minimum lives"
     (MIN_LIVES = 20 is a measurement floor for a 5th percentile, not a
     claim bar; the venue measures ~80–106 lives per arm per seed)
  V5 a completed life with an empty cause tag       "untagged terminal"
  V6 control p5 under the horizon                   "control: survival
     gate cannot tell a fatal world from a survivable one"

COST, from W1.01's and W1.02's recorded runs (~830–1500 core-s per
28000-decision rollout): 4 arms (w101-passive, w101-oracle, w102-repeat,
control) x 3 seeds = 12 rollouts over 3 niced spawn workers, ~40–60 min
wall. CPU_LONG.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..survival import HOLD_K
from ..w0 import SIM_S_PER_DECISION, W0, random_action
from .w0_diag_exploration_reaches_food import E0
from .w1_01_passivity_dies import _rollout as _w101_rollout
from .w1_02_outcomes_have_resolution import N_DEC, _borrow
from . import w1_01_passivity_dies as _w101_mod

IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py", "playground.py",
             "experiments/survival.py",
             "experiments/tests/w0_diag_exploration_reaches_food.py",
             "experiments/tests/w1_01_passivity_dies.py",
             "experiments/tests/w1_02_outcomes_have_resolution.py"]

# the family has ONE declared horizon by construction (w1_01 imports N_DEC
# from w1_02); if that ever stops being true this line refuses loudly
assert _w101_mod.N_DEC == N_DEC
HORIZON_S = float(N_DEC) * SIM_S_PER_DECISION   # 5600.0 s

RATIO_MIN = 3.0          # the design's "horizon >= 3x time-to-consequence"
MIN_LIVES = 20           # measurement floor for a 5th percentile (V4)
KNOWN_W1 = {"W1.00", "W1.01", "W1.02", "W1.04"}
PILOT_SEED = 90


def _claimset_ok() -> float:
    """V2: refuse to quantify over a W1 family this spec was not written
    for. A W1.03 registration must strengthen this spec, not slip past it."""
    return float(all(sid in KNOWN_W1
                     for sid in BY_ID if sid.startswith("W1.")))


def _repeat_rollout(seed: int, j0: float, alpha: float, lethal: bool,
                    n_decisions: int = N_DEC) -> dict:
    """W1.02's repeat-action loop verbatim with the lethal flag exposed —
    the control is this exact measurement differing in ONE argument."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=lethal)
    w.drives.state = drives.DriveState(e=E0)
    rng = np.random.RandomState(seed * 6553 + 11)
    held_a, held_left = None, 0
    lives: list = []
    t0 = time.perf_counter()
    for _ in range(n_decisions):
        if held_left == 0:
            held_a, held_left = random_action(rng), HOLD_K
        a, held_left = held_a, held_left - 1
        w.decide(a)
        if w.died_this_decision:
            lives.append((w.life_lengths[-1], str(w.last_death_cause)))
            w.drives.state = drives.DriveState(e=E0)
    frag_span = float(w.sim_seconds - w._life_started_at)
    return {
        "lives": lives,                      # (span_s, cause) per completed
        "frag_span": frag_span,              # censored at the horizon
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
        "wall_s": float(time.perf_counter() - t0),
    }


def _spans_and_causes(lives: list, frag_span: float) -> Tuple[list, dict]:
    """All survival spans (completed + fragment censored at the horizon)
    and the per-cause counts. Censored spans enter at censoring time —
    this can only understate survival (see docstring)."""
    spans = [s for s, _c in lives]
    counts = {"energy": 0.0, "integrity": 0.0, "other": 0.0,
              "censored": 0.0}
    for _s, cause in lives:
        counts[cause if cause in counts else "other"] += 1.0
    if frag_span > 0.0:
        spans.append(frag_span)
        counts["censored"] += 1.0
    return spans, counts


def _p5(spans: list) -> float:
    return float(np.percentile(spans, 5.0)) if spans else 0.0


def _mean_completed(lives: list) -> float:
    spans = [s for s, _c in lives]
    return float(np.mean(spans)) if spans else 0.0


_CACHE: Dict[Tuple[int, str], dict] = {}


def _arm(seed: int, arm: str) -> Optional[dict]:
    key = (seed, arm)
    if key not in _CACHE:
        j0, alpha, ok = _borrow()
        if not ok:
            return None
        if arm == "w102":
            _CACHE[key] = _repeat_rollout(seed, j0, alpha, lethal=True)
        elif arm == "ctrl":
            _CACHE[key] = _repeat_rollout(seed, j0, alpha, lethal=False)
        else:   # w101p / w101o — W1.01's own arms, its loop, its oracle
            out = _w101_rollout(seed, j0, alpha,
                                "oracle" if arm == "w101o" else "passive",
                                twin=False)
            _CACHE[key] = {
                "lives": [(s, c) for s, _g, c in out["lives"]],
                "frag_span": float(out["frag"][0]),
                "physics_finite": out["physics_finite"],
                "wall_s": out["wall_s"],
            }
    return _CACHE[key]


def _experiment(seed: int) -> dict:
    p = _arm(seed, "w101p")
    o = _arm(seed, "w101o")
    r = _arm(seed, "w102")
    if p is None or o is None or r is None:
        return {"borrowed_ok": 0.0}

    sp_p, cc_p = _spans_and_causes(p["lives"], p["frag_span"])
    sp_o, cc_o = _spans_and_causes(o["lives"], o["frag_span"])
    sp_r, cc_r = _spans_and_causes(r["lives"], r["frag_span"])

    # conjunct (a): measured time-to-consequence per claim
    t2c_w101 = _mean_completed(p["lives"])   # the statue's death
    t2c_w102 = _mean_completed(r["lives"])   # one outcome sample's price
    ratio_w101 = HORIZON_S / t2c_w101 if t2c_w101 > 0.0 else 0.0
    ratio_w102 = HORIZON_S / t2c_w102 if t2c_w102 > 0.0 else 0.0

    # conjunct (c): 5th-pct survival per claim; W1.01 gated at its worse arm
    p5_w101 = min(_p5(sp_p), _p5(sp_o))
    p5_w102 = _p5(sp_r)

    lethal_ns = [len(p["lives"]), len(o["lives"]), len(r["lives"])]
    tagged = all(c != "" for arm_out in (p, o, r)
                 for _s, c in arm_out["lives"])
    causes = {k: cc_p[k] + cc_o[k] + cc_r[k] for k in cc_p}

    return {
        "borrowed_ok": 1.0,
        "claimset_ok": _claimset_ok(),
        "physics_finite": min(p["physics_finite"], o["physics_finite"],
                              r["physics_finite"]),
        "min_lives_ok": float(min(lethal_ns) >= MIN_LIVES),
        "tagged_ok": float(tagged),
        "horizon_s": HORIZON_S,
        "t2c_w101": t2c_w101, "t2c_w102": t2c_w102,
        "ratio_w101": ratio_w101, "ratio_w102": ratio_w102,
        "ratio_ok": float(ratio_w101 >= RATIO_MIN
                          and ratio_w102 >= RATIO_MIN),
        "p5_w101_passive": _p5(sp_p), "p5_w101_oracle": _p5(sp_o),
        "p5_w101": p5_w101, "p5_w102": p5_w102,
        "p5_ok": float(p5_w101 >= HORIZON_S and p5_w102 >= HORIZON_S),
        "n_w101p": float(len(p["lives"])), "n_w101o": float(len(o["lives"])),
        "n_w102": float(len(r["lives"])),
        "cause_energy": causes["energy"],
        "cause_integrity": causes["integrity"],
        "cause_other": causes["other"],
        "cause_censored": causes["censored"],
        "trivial": float(t2c_w101 * 3.0 < HORIZON_S
                         and t2c_w102 * 3.0 < HORIZON_S
                         and p5_w101 >= HORIZON_S
                         and p5_w102 >= HORIZON_S),
        "wall_s": p["wall_s"] + o["wall_s"] + r["wall_s"],
    }


def _control(seed: int) -> dict:
    """The one-flag twin: terminating mechanism off, identical measurement."""
    t = _arm(seed, "ctrl")
    if t is None:
        return {"borrowed_ok": 0.0}
    spans, _cc = _spans_and_causes(t["lives"], t["frag_span"])
    p5 = _p5(spans)
    return {
        "borrowed_ok": 1.0,
        "claimset_ok": _claimset_ok(),
        "physics_finite": t["physics_finite"],
        "horizon_s": HORIZON_S,
        "p5_ctrl": p5,
        "p5_ok": float(p5 >= HORIZON_S),
        "n_ctrl_lives": float(len(t["lives"])),
        "wall_s": t["wall_s"],
    }


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1: the borrow
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    # V2: the quantification set (means across seeds; 1.0 means every seed)
    if m.get("claimset_ok", 0.0) != 1.0 or c.get("claimset_ok", 0.0) != 1.0:
        return _void(m, "claim set stale: a W1 spec outside the declared "
                        "quantification set is registered — strengthen this "
                        "spec to include it")
    # V3-V5: the rig
    if (m.get("physics_finite", 0.0) != 1.0
            or c.get("physics_finite", 0.0) != 1.0):
        return _void(m, "non-finite physics")
    if m.get("min_lives_ok", 0.0) != 1.0:
        return _void(m, "under minimum lives for a percentile")
    if m.get("tagged_ok", 0.0) != 1.0:
        return _void(m, "untagged terminal")
    # V6: the control — a survival gate that cannot tell a fatal world from
    # a survivable one is measuring nothing (control-red VOIDs, never FAILs)
    if c.get("p5_ok", 0.0) != 1.0:
        return _void(m, "control: survival gate cannot tell a fatal world "
                        "from a survivable one")

    ratio_ok = m.get("ratio_ok", 0.0) == 1.0
    p5_ok = m.get("p5_ok", 0.0) == 1.0
    if ratio_ok and p5_ok:
        if m.get("trivial", 0.0) == 1.0:
            m["claim_branch"] = (
                "TRIVIAL PASS, per the pre-registered branch: measured "
                "time-to-consequence is under a third of the horizon "
                "everywhere AND the lives outlast the horizon — this spec "
                "should be RETIRED rather than kept as decoration "
                f"(t2c {m.get('t2c_w101', float('nan')):.1f}/"
                f"{m.get('t2c_w102', float('nan')):.1f} s, p5 "
                f"{m.get('p5_w101', float('nan')):.1f}/"
                f"{m.get('p5_w102', float('nan')):.1f} s vs horizon "
                f"{HORIZON_S:.0f} s)")
        else:
            m["claim_branch"] = (
                "the horizon is real: every registered W1 claim's window is "
                ">= 3x its mechanism's measured time-to-consequence and its "
                "lives outlast the horizon")
        return True
    failed = []
    if not ratio_ok:
        failed.append(
            "(a) horizon under 3x time-to-consequence (ratio "
            f"W1.01 {m.get('ratio_w101', float('nan')):.2f}, "
            f"W1.02 {m.get('ratio_w102', float('nan')):.2f} vs {RATIO_MIN})")
    if not p5_ok:
        failed.append(
            "(c) lives end before the horizon closes (5th-pct survival "
            f"W1.01 {m.get('p5_w101', float('nan')):.1f} s, "
            f"W1.02 {m.get('p5_w102', float('nan')):.1f} s vs horizon "
            f"{HORIZON_S:.0f} s) — the horizon is fiction, and it may NOT "
            "be repaired by shortening the horizon to fit the deaths")
    m["claim_branch"] = "FAILED conjunct(s): " + "; ".join(failed)
    return False


def _run_task(args: tuple) -> tuple:
    seed, arm = args
    return args, _arm(seed, arm)


def _worker_init():
    import os
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


def run(ledger: Ledger | None = None):
    """12 arm tasks over 3 single-threaded niced workers (W1.01/W1.02's
    pool pattern), memoised into run_spec."""
    import multiprocessing as mp

    spec = BY_ID["W1.04"]
    tasks = [(seed, arm)
             for seed in range(spec.seeds)
             for arm in ("w101p", "w101o", "w102", "ctrl")]
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        for key, out in pool.map(_run_task, tasks):
            if out is not None:
                _CACHE[key] = out
    return run_spec(spec, _experiment, _check, control_fn=_control,
                    ledger=ledger or Ledger())


# ── fixtures: every _check branch, including both boundaries ───────────────
def _fixtures() -> int:
    H = HORIZON_S

    def gm(**kw):
        m = {"borrowed_ok": 1.0, "claimset_ok": 1.0, "physics_finite": 1.0,
             "min_lives_ok": 1.0, "tagged_ok": 1.0, "horizon_s": H,
             "t2c_w101": 60.0, "t2c_w102": 53.0,
             "ratio_w101": H / 60.0, "ratio_w102": H / 53.0,
             "ratio_ok": 1.0, "p5_w101": H, "p5_w102": H, "p5_ok": 1.0,
             "trivial": 1.0}
        m.update(kw)
        return m

    def gc(**kw):
        c = {"borrowed_ok": 1.0, "claimset_ok": 1.0, "physics_finite": 1.0,
             "horizon_s": H, "p5_ctrl": H, "p5_ok": 1.0}
        c.update(kw)
        return c

    n_pass = 0

    # 1 borrow red -> VOID
    r = _check(gm(borrowed_ok=0.0), gc())
    assert r is Status.VOID
    n_pass += 1
    # 2 claim set stale -> VOID
    m = gm(claimset_ok=0.0)
    assert _check(m, gc()) is Status.VOID and "stale" in m["void_reason"]
    n_pass += 1
    # 3 non-finite physics (control side) -> VOID
    assert _check(gm(), gc(physics_finite=0.0)) is Status.VOID
    n_pass += 1
    # 4 under minimum lives -> VOID
    assert _check(gm(min_lives_ok=0.0), gc()) is Status.VOID
    n_pass += 1
    # 5 untagged terminal -> VOID
    assert _check(gm(tagged_ok=0.0), gc()) is Status.VOID
    n_pass += 1
    # 6 control p5 UNDER the horizon -> VOID (control-red, never FAIL)
    m = gm()
    r = _check(m, gc(p5_ctrl=H - 1.0, p5_ok=0.0))
    assert r is Status.VOID and "fatal world" in m["void_reason"]
    n_pass += 1
    # 7 trivial PASS -> True, branch says RETIRE
    m = gm()
    assert _check(m, gc()) is True and "RETIRED" in m["claim_branch"]
    n_pass += 1
    # 8 boundary: ratio EXACTLY 3.0 satisfies (a); not trivial -> plain PASS
    m = gm(t2c_w101=H / 3.0, ratio_w101=3.0, trivial=0.0)
    assert _check(m, gc()) is True and "RETIRED" not in m["claim_branch"]
    n_pass += 1
    # 9 boundary: p5 EXACTLY at the horizon satisfies (c)
    m = gm(p5_w101=H, p5_w102=H)
    assert _check(m, gc()) is True
    n_pass += 1
    # 10 ratio under 3 on one claim -> FAIL naming (a)
    m = gm(ratio_w101=2.9, ratio_ok=0.0)
    assert _check(m, gc()) is False and "(a)" in m["claim_branch"]
    n_pass += 1
    # 11 p5 under the horizon -> FAIL naming (c) and forbidding the repair
    m = gm(p5_w101=55.0, p5_ok=0.0, trivial=0.0)
    r = _check(m, gc())
    assert r is False and "(c)" in m["claim_branch"] \
        and "shortening" in m["claim_branch"]
    n_pass += 1
    # 12 both conjuncts red -> FAIL naming both
    m = gm(ratio_ok=0.0, p5_ok=0.0, trivial=0.0)
    r = _check(m, gc())
    assert r is False and "(a)" in m["claim_branch"] \
        and "(c)" in m["claim_branch"]
    n_pass += 1
    # 13 control red NEVER reaches the claim: even a failing claim VOIDs
    m = gm(ratio_ok=0.0, p5_ok=0.0, trivial=0.0)
    assert _check(m, gc(p5_ok=0.0)) is Status.VOID
    n_pass += 1

    print(f"fixtures: {n_pass}/13 green")
    return n_pass


def _pilot(n_decisions: int = 3000):
    """Cheap smoke at PILOT_SEED (disjoint from recorded seeds; records
    nothing): plumbing, cause tags, and the control's censoring shape."""
    j0, alpha, ok = _borrow()
    if not ok:
        print("pilot: borrow unavailable")
        return
    for arm, leth in (("w102", True), ("ctrl", False)):
        out = _repeat_rollout(PILOT_SEED, j0, alpha, lethal=leth,
                              n_decisions=n_decisions)
        spans, cc = _spans_and_causes(out["lives"], out["frag_span"])
        print(f"{arm}: lives {len(out['lives'])} p5 {_p5(spans):.1f}s "
              f"causes {cc} frag {out['frag_span']:.1f}s "
              f"finite {out['physics_finite']} wall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    import sys
    if "--pilot" in sys.argv:
        _pilot()
    else:
        _fixtures()
