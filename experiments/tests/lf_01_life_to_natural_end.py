"""LF.01 — a life runs to its natural end, and the harness survives it.

T1.06 certified 1000 steps (15 sim-s). This is the 240x extension: one life of
>= 1 simulated HOUR in lethal W0, then a fast to a natural death, with the
harness holding four promises the whole way — bounded memory, bounded diary
growth, finite state, and a DEATH the record can tell from a CRASH. The claim
is about the HARNESS, not about Jack: no learner runs here, and the policy is
declared apparatus (below), exactly as XL.00's statue and drift arms were.

WHY A SCRIPTED FORAGER IS HONEST, and why the hour is reachable at all. The
energy economy was CALIBRATED to make subsistence possible (drives.py C2:
floor supply S_f = 2x0.08/66.9 = 2.391e-3 /s versus basal 1.667e-3 /s, funding
a duty cycle of 0.217), and metabolic power is integrated from the ACTUATOR
dofs (drives.substep: qfrc_actuator x qvel), so the world-frame drive force —
W0's gated locomotion cheat — moves the body without metabolic surcharge. A
privileged servo that shuttles between the two floor foods therefore lives on
the world's own arithmetic, not on a loophole this spec invented: it eats what
contact-eating awards anyone standing there. The forager reads live state
(geom positions, eat counters) because it is a FIXTURE, like XL.00's samplers;
nothing here claims he learned to forage — SH.01's corpse says exactly that
claim must not be smuggled.

THE LIFE, in two declared phases:
  FORAGE  drive between obj0/obj1 (and the apple, only if it has left its
          platform and sits on the ground outside the pool — the lc03 food
          probe measured that it can), eating on contact, until the clock
          reads HOUR_SIM_S. Pool avoidance is geometric (skirt the disc) so
          drowning cannot end the life the claim needs alive. MEASURED
          CAVEAT (attempt 1, 2026-09-03): drowning was the only damage route
          this design excluded, and it was not the one that fired — W0
          killed the forager by INTEGRITY (body damage) mid-forage at
          sim_s 1476.9 +/- 382.0, min_energy 0.128, eats 23.3. The damage
          exclusion did NOT hold at 240x exposure; only its pool-shaped
          corner did.
  FAST    drive to a rest point away from food and pool, then zero action.
          Basal drain empties a full tank in 600 s (1/BASAL_B), so death by
          "energy" arrives within FAST_CAP_DECISIONS by arithmetic, and W0
          records it: exactly one death, exactly one diary row, cause named.

THE FOUR GATES, all exact or pre-registered:
  bounded memory   peak RSS <= PEAK_RSS_MB_MAX (the box's 1.5 GB tenant rule,
                   with margin), and RSS drift — mean of the last quarter of
                   samples minus mean of the second quarter — <= RSS_DRIFT_MB_MAX.
                   The first quarter is excluded because page-in and allocator
                   warmup live there (T0.07's warmup lesson, applied to RSS).
  bounded diary    exact structural invariant: W0 writes the diary ONLY at
                   death, so len(diary.events) == deaths == 1, and 0 rows
                   before the death. A per-decision writer fails this by
                   thousands, not by a margin.
  finite state     zero non-finite readings of (qpos, qvel, e, i, w) at any of
                   the >= 18,000 decision boundaries.
  death != crash   the record carries outcome in {"death", "crash"} from two
                   INDEPENDENT detectors: death is W0's own died_this_decision
                   (drive floor), crash is the watchdog (non-finite state, a
                   raising step, OR the engine's own corruption counters). The
                   CONTROL poisons qvel with NaN mid-life: the drive layer's
                   comparisons (NaN <= 0 is False) mean a world that cannot
                   crash-detect would run on silently or — worse — call it
                   mortality; the harness must classify it as a crash within
                   CRASH_DETECT_MAX_DECISIONS and the death counter must not
                   move. A NaN scored as death is exactly falsified_by, and
                   returns FAIL, never VOID.

                   THE CONTROL EARNED ITS KEEP BEFORE THE RUN (smoke, seed 0,
                   2026-09-03): the first watchdog checked only finiteness,
                   and the poison was classified NOTHING — MuJoCo's mj_step
                   detects the bad qvel itself, increments mjWARN_BADQVEL, and
                   AUTO-RESETS the state, so by the decision boundary every
                   number is finite again and a corrupted life continues from
                   a silently teleported pose. A crash detector that reads
                   only state finiteness cannot see an engine-healed
                   corruption; the watchdog therefore also reads the four
                   BAD-state warning counters (BADQPOS/BADQVEL/BADQACC/
                   BADCTRL) every decision. Law 2's gift, verbatim: the
                   control localised the bug, pre-registration.

THE REAL-TIME FACTOR IS A GATE, TWICE (the registry's own words: "a GATE, not
a note"). Before stepping, each seed measures rtf on a probe world and calls
rtf.require_feasible over the WHOLE child's declared sim total (T0.32's
standing refusal — this file is the first long-run spec bound by it, 66th
audit B2's bind-from-now). A refusal VOIDs with the Decision recorded. After
the life, the MEASURED whole-life rtf must be >= RTF_MIN = 1.0: a harness
slower than real time cannot host "a life of hours" and the fixture's own
existence proof (SO.01) already cleared 1.0 with a render stream attached.

VOID LANES, none of them FAIL: PS.01 calibration refused (borrow_metrics — an
uncalibrated drive layer cannot host a calibrated starvation); rtf projection
refused; the forager DYING before the hour by EITHER mortality route —
starved (cause=energy) or wrecked (cause=integrity) — because either way the
fixture failed to buy the 240x exposure, and the world working as designed
refutes the SCRIPT, not the harness (attempt 1 took the integrity route: the
lane's original text named only starving, and the record corrected it — 67th
audit B4); the fast phase failing to kill by its cap (ditto, inverted). Each
records what it measured. FAIL is reserved for the falsified_by list:
unbounded memory or diary, non-finite state, a death the record cannot tell
from a crash.

THE FIXTURE-LANE VOIDS ARE CAPPED, pre-registered (67th audit B5, the
SM.02/SH.01 both-fail idiom): FIXTURE_VOID_CAP attempts whose VOID comes from
the fixture lanes (calibration and rtf both admitted, and the life still
missed the hour or the fast missed its cap), and then `run()` REFUSES —
the repair is a forager/world redesign routed to the Review, never another
pilot. Without the cap the hour gate is an infinite VOID lane and the
headline claim is unfailable. Refusal-lane VOIDs (borrow/rtf) do not count:
they are the apparatus declining to spend, not the fixture missing.

PER-SEED LOCALISATION (67th audit B3, the LG.00 `<key>_s<seed>` idiom):
attempt 1's seed-mean metrics could not say WHICH seed died of what, or when.
Every life's cause, outcome, sim_s, hour_mark, min_energy and death decision
index are now recorded per seed as explicit `<key>_s<seed>` keys, identical
in every seed's returned dict (a module memo runs each life once; the first
call computes all seeds, later calls read the memo), so `_aggregate` carries
them into the row verbatim. Each seed's rtf probe still runs BEFORE that
seed's life (T0.32's ordering, preserved inside the memo).
"""
from __future__ import annotations

import time

import numpy as np

from .. import drives, rtf
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..w0 import POOL_XY, SIM_S_PER_DECISION, SUBSTEPS, W0
# After `..w0`, deliberately: importing it puts the repo root on sys.path
# (XL.00's idiom), and EpisodicMemory lives there rather than in the package.
from EpisodicMemory import EpisodicMemory  # noqa: E402

IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "experiments/rtf.py"]

# ── the pre-registered numbers, all of them, before the run ────────────────
HOUR_SIM_S = 3600.0            # the claim: >= 1 simulated hour alive
HOUR_DECISIONS = int(HOUR_SIM_S / SIM_S_PER_DECISION)   # 18,000
MIN_CONTROL_STEPS = 240_000    # the registry's number; one decision is 40
                               # physics steps, so the hour delivers 720,000 —
                               # reported, and gated as exact arithmetic
FAST_CAP_DECISIONS = 5_000     # 1000 sim-s. A full tank empties in 600 s of
                               # basal drain; travel to the rest point plus one
                               # incidental en-route meal (+0.16 e = 96 s)
                               # still leaves 300 s of margin.
NAN_INJECT_DECISION = 750      # control: poison qvel at 150 sim-s, mid-life
CRASH_DETECT_MAX_DECISIONS = 50   # 10 sim-s from poison to classification
CONTROL_CAP_DECISIONS = NAN_INJECT_DECISION + CRASH_DETECT_MAX_DECISIONS + 200

FIXTURE_VOID_CAP = 3           # fixture-lane VOIDs before run() refuses and
                               # the repair is a Review redesign (67th audit
                               # B5). Attempt 1 (2026-09-03, integrity at ~25
                               # min) is the first of the three.
SEEDS = (0, 1, 2)              # run_spec's seeds = range(spec.seeds); named
                               # here because the per-seed keys cite them

PEAK_RSS_MB_MAX = 1400.0       # under the box's 1.5 GB tenant ceiling
RSS_DRIFT_MB_MAX = 32.0        # last-quarter mean minus second-quarter mean
RSS_SAMPLE_EVERY = 250         # decisions between /proc/self/statm samples
RTF_MIN = 1.0                  # measured whole-life sim-s per real s

# rtf.require_feasible projection: the whole child (3 seeds, experiment +
# control), in sim-seconds. Declared here so the projection cannot drift from
# the loop that spends it.
_SEED_SIM_S = (HOUR_SIM_S + FAST_CAP_DECISIONS * SIM_S_PER_DECISION
               + CONTROL_CAP_DECISIONS * SIM_S_PER_DECISION)
TOTAL_CHILD_SIM_S = 3 * _SEED_SIM_S

# Forager servo. Gains are apparatus, not claims — they were chosen at the
# smoke run and are frozen with it; the registered gates never read them.
KP, KD = 2.0, 1.0
POOL_MARGIN = 1.0              # m of clearance when skirting the pool disc
APPLE_GROUND_Z = 0.30          # apple targetable only below this height
REST_CLEARANCE = 0.5           # m; "arrived" at the rest point


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * 4096 / 1e6


def _calibration():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


class _Forager:
    """Privileged servo: drive toward the current food target, eat on contact,
    switch on eat; in the fast phase, drive to the rest point and stop."""

    def __init__(self, w: W0):
        self.w = w
        self.floor = [n for n in ("obj0", "obj1") if n in w.drives._food]
        self.gids = {n: w.drives._food[n][0] for n in list(w.drives._food)}
        self.target = self.floor[0] if self.floor else None
        self.prev_xy = self._xy()
        self.prev_ate = dict(w.drives.ate_total)
        a = float(w.params.arena_size)
        self.rest_xy = np.array([-0.6 * a, 0.6 * a])   # opposite the pool
        self.pool_xy = np.array(POOL_XY, dtype=float)
        self.pool_r = float(w.params.pool_size) + POOL_MARGIN

    def _xy(self) -> np.ndarray:
        return np.array(self.w.data.xpos[self.w.rover_bid][:2], dtype=float)

    def _in_pool(self, xy: np.ndarray) -> bool:
        return bool(np.linalg.norm(xy - self.pool_xy) < self.pool_r)

    def _food_xy(self, name: str) -> np.ndarray:
        return np.array(self.w.data.geom_xpos[self.gids[name]][:2], dtype=float)

    def _pick_target(self) -> np.ndarray:
        # Eat observed on the current target -> go to the other floor food.
        ate = self.w.drives.ate_total
        for n, k in ate.items():
            if k > self.prev_ate.get(n, 0) and n == self.target and self.floor:
                others = [f for f in self.floor if f != n]
                self.target = others[0] if others else n
        self.prev_ate = dict(ate)
        # The apple outranks floor food when it is on the ground and dry:
        # nu 0.50 / 129.6 s alone out-earns basal drain.
        if "apple" in self.gids:
            axy = self._food_xy("apple")
            az = float(self.w.data.geom_xpos[self.gids["apple"]][2])
            if az < APPLE_GROUND_Z and not self._in_pool(axy):
                return axy
        if self.target is not None:
            txy = self._food_xy(self.target)
            if self._in_pool(txy) and self.floor:
                others = [f for f in self.floor if f != self.target]
                if others and not self._in_pool(self._food_xy(others[0])):
                    self.target = others[0]
                    txy = self._food_xy(self.target)
            return txy
        return self.rest_xy

    def action(self, phase: str) -> np.ndarray:
        xy = self._xy()
        vel = (xy - self.prev_xy) / SIM_S_PER_DECISION
        self.prev_xy = xy
        tgt = self.rest_xy if phase == "fast" else self._pick_target()
        d = tgt - xy
        a = np.zeros(8)
        a[4:6] = -1.0                                  # adhesion off
        if phase == "fast" and float(np.linalg.norm(d)) < REST_CLEARANCE:
            return a                                   # arrived: lie still, starve
        # Skirt the pool: if the commanded direction points into the disc from
        # nearby, steer along its tangent instead of through it.
        to_pool = self.pool_xy - xy
        dp = float(np.linalg.norm(to_pool))
        dirv = d / (float(np.linalg.norm(d)) + 1e-9)
        if dp < self.pool_r + 1.2 and float(dirv @ (to_pool / max(dp, 1e-9))) > 0.3:
            tang = np.array([-to_pool[1], to_pool[0]]) / max(dp, 1e-9)
            dirv = tang if float(dirv @ tang) >= 0 else -tang
            d = dirv * max(float(np.linalg.norm(d)), 1.0)
        a[6:8] = np.clip(KP * d - KD * vel, -1.0, 1.0)
        return a


def _run_life(seed: int, *, j0: float, alpha: float,
              hour_sim_s: float = HOUR_SIM_S,
              fast_cap: int = FAST_CAP_DECISIONS,
              nan_inject: int | None = None,
              control_cap: int | None = None) -> dict:
    """One harnessed life. Returns the raw record both conditions share.

    `nan_inject`/`control_cap` non-None is the control; the experiment leaves
    them None. Everything else is byte-identical between the two — the control
    differs from the experiment by exactly one poisoned array.
    """
    import mujoco
    diary = EpisodicMemory()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True, diary=diary)
    forager = _Forager(w)
    # The engine's own corruption counters. mj_step detects a bad state,
    # warns, and AUTO-RESETS — state finiteness alone is blind to it (the
    # smoke control proved exactly that), so these four are the crash
    # detector's second eye. Read as cumulative counts, checked for growth.
    _bad = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL]

    def _warnings() -> int:
        return int(sum(w.data.warning[int(k)].number for k in _bad))
    phase = "forage"
    hour_mark = None            # decisions on the clock when the hour struck
    outcome, cause, crash_at = "", "", -1
    death_at = -1               # decision index of death (B3: localisation)
    finite_violations = 0
    rss = [_rss_mb()]
    min_e = 1.0
    t0 = time.perf_counter()
    while True:
        if nan_inject is not None and w.decisions == nan_inject:
            w.data.qvel[:] = np.nan            # the poison, between decisions
        try:
            w.decide(forager.action(phase))
        except Exception as e:                 # a raising step IS a crash
            outcome, crash_at = "crash", w.decisions
            cause = f"raise:{type(e).__name__}"
            break
        s = w.drives.state
        if not (np.isfinite(w.data.qpos).all() and np.isfinite(w.data.qvel).all()
                and np.isfinite([s.e, s.i, s.w]).all()) or _warnings() > 0:
            finite_violations += 1
            outcome, crash_at = "crash", w.decisions
            break
        min_e = min(min_e, float(s.e))
        if w.decisions % RSS_SAMPLE_EVERY == 0:
            rss.append(_rss_mb())
        if w.died_this_decision:
            outcome, cause = "death", w.last_death_cause
            death_at = w.decisions
            break
        if phase == "forage" and w.sim_seconds >= hour_sim_s:
            phase, hour_mark = "fast", w.decisions
        if control_cap is not None and w.decisions >= control_cap:
            outcome = "no_crash_by_cap"
            break
        if hour_mark is not None and w.decisions >= hour_mark + fast_cap:
            outcome = "no_death_by_cap"
            break
    wall = time.perf_counter() - t0
    rss = np.array(rss + [_rss_mb()])
    q = max(1, len(rss) // 4)
    return {
        "outcome": outcome, "cause": cause, "crash_at": float(crash_at),
        "death_at": float(death_at),
        "decisions": float(w.decisions), "sim_s": float(w.sim_seconds),
        "substeps": float(w.decisions * SUBSTEPS),
        "deaths": float(w.deaths), "diary_rows": float(len(diary.events)),
        "hour_mark": float(-1 if hour_mark is None else hour_mark),
        "finite_violations": float(finite_violations),
        "peak_rss_mb": float(rss.max()),
        "rss_drift_mb": float(rss[-q:].mean() - rss[q:2 * q].mean()),
        "rtf_life": float(w.sim_seconds / wall) if wall > 0 else 0.0,
        "wall_s": float(wall), "min_energy": float(min_e),
        "eats": float(sum(w.drives.ate_total.values())),
        "drive_gate_frac": float(w.report()["drive_gate_frac"]),
    }


def _feasibility(seed: int, j0: float, alpha: float) -> dict:
    """T0.32's standing refusal, measured on a probe world then discarded."""
    probe = W0(seed=seed, j0=j0, alpha=alpha)
    zero = np.zeros(8)
    reading = rtf.measure_rtf(lambda: probe.decide(zero),
                              SIM_S_PER_DECISION, n_steps=50, warmup=10,
                              trials=2)
    try:
        d = rtf.require_feasible(reading, TOTAL_CHILD_SIM_S, spec=BY_ID["LF.01"])
        return {"rtf_probe": reading.rtf, "rtf_projected_s": d.projected_s,
                "rtf_limit_s": d.limit_s, "rtf_admitted": 1.0}
    except rtf.RTFRefusal as e:
        return {"rtf_probe": reading.rtf, "rtf_admitted": 0.0,
                "rtf_refusal": f"VOID: {e.decision.reason}"}


# seed -> that seed's feasibility record (+ life record when admitted).
# LG.00's memo idiom: each life runs ONCE; the first _experiment call computes
# every seed so that every seed's returned dict can carry the full per-seed
# key set and _aggregate records it verbatim (a key absent from runs[0] is
# dropped, so uniform emission is what makes the row able to localise).
_LIVES: dict = {}

# The per-seed keys B3 names, emitted for every seed in every run's dict.
_PER_SEED = ("outcome", "cause", "sim_s", "hour_mark", "min_energy",
             "death_at")


def _ensure_life(seed: int, j0: float, alpha: float) -> dict:
    if seed not in _LIVES:
        rec: dict = dict(_feasibility(seed, j0, alpha))
        if rec["rtf_admitted"] == 1.0:
            # probe-then-step, per seed, inside the memo: T0.32's ordering.
            rec.update(_run_life(seed, j0=j0, alpha=alpha))
        _LIVES[seed] = rec
    return _LIVES[seed]


def _experiment(seed: int) -> dict:
    j0, alpha, prov = _calibration()
    m: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return m
    for s in SEEDS:
        _ensure_life(s, j0, alpha)
    r = _LIVES[seed]
    m.update(r)
    for s in SEEDS:
        rs = _LIVES[s]
        for k in _PER_SEED:
            m[f"{k}_s{s}"] = rs.get(
                k, "" if k in ("outcome", "cause") else -1.0)
    if m["rtf_admitted"] != 1.0:
        return m
    m["survived_hour"] = float(r["hour_mark"] >= 0)
    m["died_naturally"] = float(r["outcome"] == "death"
                                and r["cause"] in ("energy", "integrity"))
    m["one_death_one_row"] = float(r["deaths"] == 1.0
                                   and r["diary_rows"] == 1.0)
    m["control_steps_ok"] = float(r["hour_mark"] >= 0
                                  and r["hour_mark"] * SUBSTEPS
                                  >= MIN_CONTROL_STEPS)
    m["finite_ok"] = float(r["finite_violations"] == 0.0)
    m["rss_ok"] = float(r["peak_rss_mb"] <= PEAK_RSS_MB_MAX
                        and r["rss_drift_mb"] <= RSS_DRIFT_MB_MAX)
    m["rtf_ok"] = float(r["rtf_life"] >= RTF_MIN)
    m["conjunction"] = float(
        m["survived_hour"] == 1.0 and m["died_naturally"] == 1.0
        and m["one_death_one_row"] == 1.0 and m["control_steps_ok"] == 1.0
        and m["finite_ok"] == 1.0 and m["rss_ok"] == 1.0
        and m["rtf_ok"] == 1.0)
    return m


def _control(seed: int) -> dict:
    """The NaN mid-life. Must be classified CRASH; the death counter must not
    move; detection must land within CRASH_DETECT_MAX_DECISIONS of the poison."""
    j0, alpha, prov = _calibration()
    c: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return c
    r = _run_life(seed, j0=j0, alpha=alpha,
                  nan_inject=NAN_INJECT_DECISION,
                  control_cap=CONTROL_CAP_DECISIONS)
    c.update(r)
    c["crash_ok"] = float(
        r["outcome"] == "crash"
        and r["deaths"] == 0.0 and r["diary_rows"] == 0.0
        and NAN_INJECT_DECISION
        <= r["crash_at"] <= NAN_INJECT_DECISION + CRASH_DETECT_MAX_DECISIONS)
    c["nan_scored_as_death"] = float(r["outcome"] == "death")
    return c


def _check(m: dict, c: dict):
    if m.get("calibrated") != 1.0 or c.get("calibrated") != 1.0:
        return Status.VOID          # PS.01 supplied no usable j0/alpha —
        # borrow_metrics wrote the refusal into the metrics.
    if m.get("rtf_admitted") != 1.0:
        return Status.VOID          # T0.32 refused the projection; the
        # Decision's reason is in the metrics. Not a refutation.
    # A NaN scored as mortality is the falsified_by list verbatim: FAIL.
    if c.get("nan_scored_as_death", 0.0) > 0.0:
        return False
    if m.get("survived_hour") != 1.0:
        # The forager DIED before the hour — starved (energy) or wrecked
        # (integrity), both are the FIXTURE failing to buy the 240x exposure,
        # so the harness claim was never tested. Attempt 1 took the integrity
        # route; the per-seed keys say which seed died of what, and when.
        return Status.VOID
    if m.get("outcome") == "no_death_by_cap":
        # The fast failed to kill by its own arithmetic cap: fixture fault.
        return Status.VOID
    return bool(m.get("conjunction") == 1.0 and c.get("crash_ok") == 1.0)


def _fixture_void_count(ledger: Ledger) -> int:
    """VOIDs from the fixture lanes only: calibration AND rtf admitted, yet
    the row is VOID — i.e. the life missed the hour or the fast missed its
    cap. Refusal-lane VOIDs (borrow/rtf declined to spend) do not count.
    Reads the live row plus its history, so a re-registration cannot forget."""
    row = ledger.results.get("LF.01")
    if row is None:
        return 0
    rows = list(getattr(row, "history", None) or []) + [row.__dict__ if not
                                                        isinstance(row, dict)
                                                        else row]
    n = 0
    for r in rows:
        d = r if isinstance(r, dict) else r.__dict__
        status = d.get("status")
        status = getattr(status, "value", status)
        mm = d.get("metrics") or {}
        if (status == "VOID" and mm.get("calibrated") == 1.0
                and mm.get("rtf_admitted") == 1.0):
            n += 1
    return n


def run(ledger: Ledger | None = None):
    ledger = ledger or Ledger()
    n = _fixture_void_count(ledger)
    if n >= FIXTURE_VOID_CAP:
        raise RuntimeError(
            f"LF.01 has {n} fixture-lane VOIDs against the pre-registered cap "
            f"of {FIXTURE_VOID_CAP} (67th audit B5). The hour gate does not "
            f"get an infinite VOID lane: the repair is a forager/world "
            f"redesign routed to the Review "
            f"(docs/REVIEW_QUEUE.md: w0-kills-a-forager-by-integrity-at-25-"
            f"minutes), never another pilot.")
    return run_spec(BY_ID["LF.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
