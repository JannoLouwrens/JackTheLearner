"""W1.01 — Passivity dies.

THE QUESTION. Seven instruments read W0 as too shallow to host a learning
claim, and the strongest single form of that reading is SH.02's pilot: twin,
privileged oracle and both cosmetic controls all at exactly 1.0000 — a venue
where doing nothing already holds the roof. The Review FULL's published W1
design (REVIEW_QUEUE `w0-too-shallow`, 2026-09-06) turns that prose into a
registered gate: in W0 AS BUILT, on the W1.02-certified graded outcome, is
there measurable headroom between a do-nothing agent and a hand-coded
competent oracle? The registered PREDICTION is FAIL (the registry's
`falsified_by` says so, recorded before any run); a FAIL here is the
registered, gated form of the `w0-too-shallow` finding, not an error to
repair by softening. W0.DIAG's +12.12 life-second food result cuts the other
way, so the run decides, not the paragraphs.

THE OUTCOME is W1.02's, unchanged: per-life G = sum over the life's
observable decisions of (1 - min(1, d(h))) * SIM_S_PER_DECISION
(satisfied-seconds; the dying decision excluded, exactly W1.02's loop), at
the W1.02-certified envelope — repeat venue, E0 respawn energy, PS.01's
borrowed j0/alpha, lethal=True, N_DEC decisions per arm (N_DEC imported from
w1_02, not copied). The per-arm statistic is the FRAGMENT-INCLUSIVE mean
per-life G ("no uncensored-only mean stands alone", W1.02 C3): a never-dying
arm has zero completed lives and one censored fragment, and the mean must be
defined for it — the whole claim is that such an arm exists.

THE ARMS, both at action-space zeros except the drive channel so they differ
in exactly one place:

  PASSIVE  np.zeros(8) every decision — survival.py's statue, verbatim
           (LC.03's control (a), measured 599.92 s on its 600 s basal
           ceiling). At this envelope its ceiling is E0/BASAL_B = 60 s.
  ORACLE   privileged scripted need-serving (the design's words; competence
           is the oracle's job, honesty the twin control's). It reads the
           live food geom positions, the DriveLayer's own respawn timers and
           the root velocity straight from the instrument — privileged by
           declaration — and servos the gated horizontal drive onto the
           nearest AVAILABLE reachable food, else toward the earliest-to-
           respawn one, resting (the statue's zeros: heal, zero power) once
           within PARK_DIST of a cooldown target. Reachable = geom centre
           below REACH_Z and outside the ACTUAL pool box (the pool is a
           0.2-1.2 m hole where the gated drive is dead — a food floating
           on it is not worth the trap; a margin-inflated box measurably
           excluded a dry food at seed 90 and starved the oracle on one
           food's sub-basal income). The apple on the platform is excluded
           by REACH_Z unless it has fallen. While
           TRAVELLING it pumps ONE lift slide at low amplitude (PUMP_AMP)
           with adhesion off, because the drive alone cannot break ground
           friction at this envelope's gear (E0=0.1 -> 0.46 x 600 N =
           276 N against a ~324 N body on mu 1.0 — the bare servo crawls
           at 0.02 m/s and starves). The gait is sized by MEASUREMENT and
           for ENERGY, never for the bar: 0.135 m/s at 82 W mechanical,
           drain 1.86e-3/s against the floor-food income ceiling
           2.39e-3/s, zero integrity loss; the obvious full-amplitude
           pump moved barely faster at 1471 W and died faster than the
           statue. All the gait numbers are seed-90 measurements taken
           before the recorded run; none is a gate.

THE CLAIM (pre-registered; the bar is W1.02's certificate, not this desk's):
on EVERY seed, gap = mean_all_lives(oracle) - mean_all_lives(passive)
> GAP_BAR = 3 x quantum, where quantum is READ from W1.02's recorded PASS
row (pinned by ran_at; a moved row VOIDs, never silently re-anchors). The
row reports FLOOR = the passive reading and ROOF = the oracle reading per
the design's `passive <= FLOOR < ROOF <= oracle`; the binding statistic is
the gap (registry metric `min_over_seeds_oracle_minus_passive_gap`, enforced
as the per-seed conjunct gap_ok on all seeds).

THE CONTROL (must show nothing — registry: control-red VOIDs, never FAILs):
the identical measurement on a deliberately-benign twin world in which the
nutrition channel is dead — every food geom's energy value set to 0.0 on the
twin's own DriveLayer instance, nothing else touched. In that world
competence buys nothing, so the oracle-minus-passive gap must sit AT or
BELOW the bar on every seed; a control gap ABOVE the bar means the
instrument fabricates headroom in a world that has none, and the run VOIDs
(instrument fault, T0.22). The twin carries its own alive-proof BOTH ways:
its oracle must still EAT (contact the dead food — proving the mechanism
ran with the channel removed, not that the oracle sat down), and the claim
world's passive arm must sit on its basal ceiling (LC.03's statue gate,
10% tolerance) — a statue that lives the wrong length is a broken
instrument, not a world reading. DIVERGENCE FROM THE DESIGN'S WORD
"benign", NAMED: the twin is kind to no one — it is the world with the
headroom CHANNEL removed, which is the operative property the design's own
sentence states ("a headroom test that cannot detect a world without
headroom is measuring nothing"); a twin whose needs never deplete would
make the control's gap identically zero by construction, i.e. a control
that cannot fire, which law 2 forbids.

VOID LANES, in test order (an instrument fault is not a world reading):
  V1 PS.01 borrow unavailable                       "uncalibrated borrow"
  V2 W1.02 row missing, moved, or not PASS          "recorded inputs moved"
  V3 non-finite physics in any arm                  "non-finite physics"
  V4 claim-world passive off the basal ceiling      "passive arm off the
     (|mean life - E0/BASAL_B| > 10%)                basal ceiling"
  V5 food present and the claim-world oracle never
     ate — an incompetent oracle could manufacture
     the predicted FAIL, so it may not score        "oracle never ate"
  V6 food present and the TWIN oracle never ate     "twin mechanism dead"
  V7 twin gap above the bar on any seed             "control shows headroom
                                                     in a world with none"
If a seed's world carries NO food geoms (a mutation may drop them), V5/V6 do
not apply and the gap is measured as found: a foodless world admitting no
gap is the claim failing honestly, not the instrument breaking.

FALSIFIED BY (the registry's, binding): any seed's gap <= 3x quantum — the
world admits no such gap, and no claim about learning is admissible in a
venue where doing nothing already wins.

COST, from W1.02's measured envelope (~830 core-s per 28000-decision
rollout): 4 rollouts x 3 seeds = 12 rollouts ~ 2.8 core-h, ~55-70 min wall
on the 3-worker niced pool. cpu<2h (Budget.CPU_LONG).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..w0 import POOL_XY, SIM_S_PER_DECISION, W0
from .w0_diag_exploration_reaches_food import E0
from .w1_02_outcomes_have_resolution import N_DEC

IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py", "playground.py",
             "experiments/tests/w0_diag_exploration_reaches_food.py",
             "experiments/tests/w1_02_outcomes_have_resolution.py"]

GAP_BAR_MULT = 3.0            # the design's "3x the W1.02 quantum", unmoved
BASAL_TOL = 0.10              # LC.03 statue gate's own tolerance, borrowed
REACH_Z = 0.5                 # m; a food centre above this is not ground food
PARK_DIST = 0.45              # m; this close to a cooldown target it rests
KP, KD = 1.2, 0.6             # drive servo gains — behaviour, not a bar
PUMP_AMP = 0.3                # lift-pump amplitude — sized for energy, below
PILOT_SEED = 90

# pinned recorded row — a moved row VOIDs (V2), never silently re-anchors
W102_RAN_AT = "2026-09-06T11:32:50"


def _borrow():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if b.values is None:
        return None, None, 0.0
    return b.values["j0_ms"], b.values["alpha"], 1.0


def _quantum() -> Optional[float]:
    """W1.02's certified quantum, read from its recorded PASS row, pinned by
    ran_at. Never hard-coded: the bar follows W1.02's certificate and moves
    only if W1.02 is re-bought."""
    path = Path(__file__).resolve().parents[1] / "ledger.json"
    try:
        row = json.loads(path.read_text())["results"]["W1.02"]
        if row["ran_at"] != W102_RAN_AT or row["status"] != "PASS":
            return None
        return float(row["metrics"]["quantum"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


class _Oracle:
    """The design's hand-coded competent oracle: privileged, scripted
    need-serving. Privileged by declaration — it reads the world's own food
    positions and the DriveLayer's respawn timers, which no learner may."""

    def __init__(self, w: W0):
        self.w = w
        self.pool_half = float(w.params.pool_size)
        self.k = 0                     # decision counter for the travel gait

    def _in_pool(self, xy: np.ndarray, margin: float) -> bool:
        return (abs(float(xy[0]) - POOL_XY[0]) < self.pool_half + margin
                and abs(float(xy[1]) - POOL_XY[1]) < self.pool_half + margin)

    def act(self) -> np.ndarray:
        w = self.w
        a = np.zeros(8)
        pos = np.asarray(w.data.xpos[w.rover_bid][:2], dtype=float)
        targets = []          # (available, eta_s, dist, xy)
        for name, (gid, _nu) in w.drives._food.items():
            fxy = np.asarray(w.data.geom_xpos[gid][:2], dtype=float)
            fz = float(w.data.geom_xpos[gid][2])
            # exclusion is the ACTUAL pool box (margin 0): a food NEAR the
            # water is reachable from dry ground; one IN it is not worth a
            # drowning. Seed 90 measured the inflated box excluding a dry
            # obj1 and starving the oracle on one food's sub-basal income.
            if fz > REACH_Z or self._in_pool(fxy, 0.0):
                continue
            eta = max(0.0, float(w.drives._respawn_at[name]) - float(w.drives.t))
            dist = float(np.linalg.norm(fxy - pos))
            targets.append((eta <= 0.0, eta, dist, fxy))
        if not targets:
            return a
        avail = [t for t in targets if t[0]]
        tgt = (min(avail, key=lambda t: t[2]) if avail
               else min(targets, key=lambda t: (t[1], t[2])))
        delta = tgt[3] - pos
        dist = float(np.linalg.norm(delta))
        if not tgt[0] and dist < PARK_DIST:
            return a                         # waiting on a respawn: rest (heal,
        self.k += 1                          # zero power), the statue's zeros
        da = w.ix["root_dofadr"]
        vel = np.asarray(w.data.qvel[da:da + 2], dtype=float)
        drv = KP * delta - KD * vel
        if self._in_pool(pos, 0.3):          # inside the water: leave, shortest way
            drv = pos - np.asarray(POOL_XY)
        n = float(np.linalg.norm(drv))
        if n > 1.0:
            drv = drv / n
        # Travel gait: the drive alone cannot break ground friction at this
        # envelope's gear (E0=0.1 -> 276 N vs a ~324 N weight on mu 1.0; the
        # bare servo crawls at 0.02 m/s and starves), so ONE lift slide pumps
        # at LOW amplitude to unload the foot. Sized by measurement, for
        # energy and never for the bar: 0.135 m/s at 82 W mechanical
        # (drain 1.86e-3/s against the floor-food income ceiling 2.39e-3/s),
        # where the full-amplitude two-arm pump moved barely faster at
        # 1471 W (drain 5.08e-3/s) and died FASTER than the statue.
        # Adhesion off while travelling; zero integrity loss measured.
        a[1] = PUMP_AMP if self.k % 2 == 0 else -PUMP_AMP
        a[4:6] = -1.0
        a[6:8] = drv
        return a


def _rollout(seed: int, j0: float, alpha: float, policy: str, twin: bool,
             n_decisions: int = N_DEC) -> dict:
    """One arm, W1.02's loop verbatim: per-life (span, G, cause), the trailing
    fragment censored, the dying decision excluded from the integral."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    if twin:
        # the benign twin: nutrition channel dead, nothing else touched
        w.drives._food = {name: (gid, 0.0)
                          for name, (gid, _nu) in w.drives._food.items()}
    food_present = float(len(w.drives._food) > 0)
    w.drives.state = drives.DriveState(e=E0)
    oracle = _Oracle(w) if policy == "oracle" else None
    zeros = np.zeros(8)
    g_acc = 0.0
    lives: list = []
    t0 = time.perf_counter()
    for _ in range(n_decisions):
        w.decide(oracle.act() if oracle else zeros)
        if w.died_this_decision:
            lives.append((w.life_lengths[-1], g_acc, str(w.last_death_cause)))
            g_acc = 0.0
            w.drives.state = drives.DriveState(e=E0)
        else:
            s = 1.0 - min(1.0, w.drives.state.d())
            g_acc += s * SIM_S_PER_DECISION
    frag_span = float(w.sim_seconds - w._life_started_at)
    return {
        "lives": lives,
        "frag": (frag_span, g_acc),
        "ate": float(sum(w.drives.ate_total.values())),
        "food_present": food_present,
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
        "wall_s": float(time.perf_counter() - t0),
    }


def _mean_all_lives(out: dict) -> float:
    """Fragment-inclusive mean per-life G — defined even at zero completed
    lives, which is the shape the claim itself predicts for the oracle."""
    vals = [g for _, g, _ in out["lives"]]
    if out["frag"][0] > 0.0:
        vals.append(out["frag"][1])
    return float(np.mean(vals)) if vals else 0.0


def _mean_life_s(out: dict) -> float:
    spans = [s for s, _, _ in out["lives"]]
    return float(np.mean(spans)) if spans else float(out["frag"][0])


_CACHE: Dict[Tuple[int, str, bool], dict] = {}


def _arm(seed: int, policy: str, twin: bool) -> Optional[dict]:
    key = (seed, policy, twin)
    if key not in _CACHE:
        j0, alpha, ok = _borrow()
        if not ok:
            return None
        _CACHE[key] = _rollout(seed, j0, alpha, policy, twin)
    return _CACHE[key]


def _seed_metrics(seed: int, twin: bool) -> dict:
    q = _quantum()
    p = _arm(seed, "passive", twin)
    o = _arm(seed, "oracle", twin)
    if p is None or o is None:
        return {"borrowed_ok": 0.0}
    m: dict = {"borrowed_ok": 1.0,
               "recorded_ok": float(q is not None),
               "food_present": min(p["food_present"], o["food_present"]),
               "physics_finite": min(p["physics_finite"], o["physics_finite"])}
    if q is None:
        return m
    bar = GAP_BAR_MULT * q
    floor = _mean_all_lives(p)
    roof = _mean_all_lives(o)
    gap = roof - floor
    ceiling = E0 / drives.BASAL_B
    m.update({
        "quantum": q, "gap_bar": bar,
        "floor_passive": floor, "roof_oracle": roof, "gap": gap,
        "gap_ok": float(gap > bar),
        "ctrl_quiet": float(gap <= bar),
        "p_n_lives": float(len(p["lives"])),
        "p_mean_life_s": _mean_life_s(p),
        "basal_ok": float(abs(_mean_life_s(p) - ceiling) <= BASAL_TOL * ceiling),
        "o_n_lives": float(len(o["lives"])),
        "o_mean_life_s": _mean_life_s(o),
        "o_frag_s": float(o["frag"][0]),
        "o_ate": o["ate"],
        "oracle_fed": float(o["ate"] >= 1.0 or m["food_present"] == 0.0),
        "wall_s": p["wall_s"] + o["wall_s"],
    })
    return m


def _experiment(seed: int) -> dict:
    return _seed_metrics(seed, twin=False)


def _control(seed: int) -> dict:
    return _seed_metrics(seed, twin=True)


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1-V2: the borrow and the pinned certificate the bar is read from
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    if m.get("recorded_ok", 0.0) != 1.0 or c.get("recorded_ok", 0.0) != 1.0:
        return _void(m, "recorded inputs moved")
    # V3-V4: the rig (means across seeds; 1.0 means every seed)
    if m.get("physics_finite", 0.0) != 1.0 or c.get("physics_finite", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("basal_ok", 0.0) != 1.0:
        return _void(m, "passive arm off the basal ceiling")
    # V5-V6: competence alive-proofs, claim world and twin
    if m.get("food_present", 0.0) == 1.0 and m.get("oracle_fed", 0.0) != 1.0:
        return _void(m, "oracle never ate")
    if c.get("food_present", 0.0) == 1.0 and c.get("oracle_fed", 0.0) != 1.0:
        return _void(m, "twin mechanism dead")
    # V7: the control — a gap in a world with no headroom channel is the
    # instrument's, and control-red VOIDs, never FAILs (registry)
    if c.get("ctrl_quiet", 0.0) != 1.0:
        return _void(m, "control shows headroom in a world with none")

    if m.get("gap_ok", 0.0) == 1.0:
        m["claim_branch"] = (
            "headroom: the oracle clears the passive arm by more than 3x the "
            "W1.02 quantum on every seed — doing nothing does not hold the "
            "outcome roof in W0 as built")
        return True
    m["claim_branch"] = (
        "no admissible headroom: on at least one seed the oracle-minus-"
        "passive gap sits at or under 3x the W1.02 quantum — the registered, "
        "gated form of the w0-too-shallow finding (gap mean "
        f"{m.get('gap', float('nan')):.4f} vs bar "
        f"{m.get('gap_bar', float('nan')):.4f})")
    return False


def _run_task(args: tuple) -> tuple:
    seed, policy, twin = args
    return args, _arm(seed, policy, twin)


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
    """12 arm tasks over 3 single-threaded niced workers (W1.02's pool
    pattern), memoised into run_spec."""
    import multiprocessing as mp

    spec = BY_ID["W1.01"]
    tasks = [(seed, policy, twin)
             for seed in range(spec.seeds)
             for policy in ("passive", "oracle")
             for twin in (False, True)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        for key, out in pool.map(_run_task, tasks):
            if out is not None:
                _CACHE[key] = out
    return run_spec(spec, _experiment, _check, control_fn=_control,
                    ledger=ledger or Ledger())


# ── fixtures: every _check branch, including both boundaries ───────────────
def _fixtures() -> int:
    q = 0.0083681
    bar = GAP_BAR_MULT * q

    def base(**kw):
        d = {"borrowed_ok": 1.0, "recorded_ok": 1.0, "physics_finite": 1.0,
             "basal_ok": 1.0, "food_present": 1.0, "oracle_fed": 1.0,
             "gap": 10.0, "gap_bar": bar, "gap_ok": 1.0, "ctrl_quiet": 0.0}
        d.update(kw)
        return d

    def ctrl(**kw):
        d = base(gap=0.0, gap_ok=0.0, ctrl_quiet=1.0)
        d.update(kw)
        return d

    cases = [
        ("V1 borrow", base(borrowed_ok=0.0), ctrl(), Status.VOID,
         "uncalibrated borrow"),
        ("V2 moved row", base(recorded_ok=0.0), ctrl(), Status.VOID,
         "recorded inputs moved"),
        ("V3 physics (ctrl side)", base(), ctrl(physics_finite=0.0),
         Status.VOID, "non-finite physics"),
        ("V4 basal", base(basal_ok=0.0), ctrl(), Status.VOID,
         "passive arm off the basal ceiling"),
        ("V5 oracle starved", base(oracle_fed=0.0), ctrl(), Status.VOID,
         "oracle never ate"),
        ("V5 foodless world skips V5",
         base(food_present=0.0, oracle_fed=0.0, gap=0.0, gap_ok=0.0),
         ctrl(food_present=0.0, oracle_fed=0.0), False, None),
        ("V6 twin dead", base(), ctrl(oracle_fed=0.0), Status.VOID,
         "twin mechanism dead"),
        ("V7 control gap above bar", base(),
         ctrl(gap=bar + 1e-6, ctrl_quiet=0.0), Status.VOID,
         "control shows headroom in a world with none"),
        ("V7 boundary: control gap AT bar is quiet", base(),
         ctrl(gap=bar, ctrl_quiet=1.0), True, None),
        ("PASS just above bar", base(gap=bar + 1e-6), ctrl(), True, None),
        ("FAIL at bar exactly", base(gap=bar, gap_ok=0.0), ctrl(), False,
         None),
        ("FAIL below bar", base(gap=0.0, gap_ok=0.0), ctrl(), False, None),
        ("FAIL one seed of three (mean 2/3)", base(gap_ok=2.0 / 3.0), ctrl(),
         False, None),
    ]
    bad = 0
    for name, m, c, want, want_reason in cases:
        got = _check(dict(m), dict(c)) if want is not Status.VOID else None
        if want is Status.VOID:
            mm = dict(m)
            got = _check(mm, dict(c))
            ok = got is Status.VOID and mm.get("void_reason") == want_reason
        else:
            ok = got is want
        print(f"  [{'ok' if ok else 'XX'}] {name}")
        bad += 0 if ok else 1
    return bad


def _pilot():
    """Plumbing + oracle-competence smoke at seed 90 (disjoint from recorded
    seeds 0-2), short envelope — prints JSON, records NOTHING, reads no gate,
    freezes no bar. Competence is the oracle's job; this is where it is
    proven before a recorded run can be manufactured by an incompetent one."""
    j0, alpha, ok = _borrow()
    if not ok:
        raise SystemExit("borrow unavailable (PS.01)")
    print("quantum:", _quantum(), " bar:", (_quantum() or 0) * GAP_BAR_MULT)
    out = {}
    for policy in ("passive", "oracle"):
        for twin in (False, True):
            r = _rollout(PILOT_SEED, j0, alpha, policy, twin,
                         n_decisions=3000)
            out[f"{policy}{'_twin' if twin else ''}"] = {
                "n_lives": len(r["lives"]),
                "mean_life_s": round(_mean_life_s(r), 2),
                "mean_all_lives_G": round(_mean_all_lives(r), 3),
                "ate": r["ate"],
                "frag_s": round(r["frag"][0], 1),
                "physics_finite": r["physics_finite"],
                "wall_s": round(r["wall_s"], 1),
                "causes": sorted({c for _, _, c in r["lives"]}),
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    if "--fixtures" in sys.argv:
        raise SystemExit(_fixtures())
    _pilot()
