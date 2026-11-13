"""LC.02 — A core that cannot live a life at survivable wall-clock is not a core.

ADMISSION-2 of `docs/research/LEARNING_CORE.md` §5.1. LC.01 asked whether an arm
takes every sense into one latent; this asks whether it can be RUN. Both are
evaluated before any arm is scored, and both can only exclude — neither can
crown anything.

    T >= 5.0 simulated seconds of Jack's life per real second,
    on 3 ARM cores of this box at nice 19, with the learner in the loop.

The floor is not taste. §5.1 derives it: the climber-rover's physics ceiling was
DERIVED there at ~16.2 sim-s/real-s (this run measures the world plus its senses
at 10.09 — see the correction filed in §5.1), and 5.0 is what keeps the LC.04
envelope inside
`Budget.CPU_LONG` — i.e. what makes "Jack lives for an hour" a sentence with a
price. GOAL.md requires lives, death and cross-life learning; an arm that cannot
produce a second life inside a builder iteration cannot deliver them at any
sample efficiency, which is why exclusion here is final and not a handicap.

WHAT IS BEING TIMED, precisely, because a throughput number is only as honest as
its denominator:

  * the world is `experiments/w0.py` — the climber-rover in the real playground,
    with all six W0-4 senses computed every decision (16-ray retina, 8-band
    binaural contact audio, 4-site touch, proprioception, the drive vector, and
    `language` handled as a MISSING input condition). Not a stub. A throughput
    measured without the senses would licence a train_ratio the senses cannot
    afford.
  * the learner is `cores.lc_update` — the SAME function LC.03 will call. It is
    in `cores.py` rather than here for exactly that reason: a spec that timed a
    stand-in update would licence a ratio the real update cannot afford, and
    nothing downstream could see the substitution.
  * one decision is 40 substeps of 0.005 s = 0.2 simulated seconds, so
    5 decisions buy 1 simulated second. §5.1's accounting unit, unchanged.

THE SELECTION RULE, fixed before the run. For each arm, walk the power-of-two
train_ratios upward from 1/8 and take the LARGEST that clears 5.0; stop at the
first that does not (throughput is monotone decreasing in gradient steps per
decision, so the first failure ends the walk). An arm that fails at 1/8 is
EXCLUDED — recorded, not silently dropped.

TWO ANTI-GAMING PROVISIONS, from §5.1, and one added here:

  1. **This spec may not read `life_gain`.** Selecting a hyperparameter by its
     score is tuning on the metric. Grep this file: there is no reward function,
     no return, and `lc_targets` feeds the update NOISE on purpose — a timing
     measurement must not depend on the agent being any good.
  2. **The chosen ratio is committed to the ledger before LC.03 runs**, and
     reported for every arm, so "it lost at the ratio our hardware allows" is
     visible next to the verdict rather than discovered afterwards.
  3. **ADDED HERE: the committed ratio is the one every seed can afford.**
     The registered spec says "the largest power-of-two value that clears that
     floor" and does not say what happens when three seeds disagree. `committed_
     ratio` below fixes that before the run: the largest ratio that cleared on
     EVERY seed. It can only ever licence less compute than a seed can afford,
     which is the direction a selection step should err in.

CONTROL (the spec's own): **the 57M UnifiedBrain trunk in the control path must
FAIL the floor**, and it is measured at the SMALLEST train_ratio — the most
favourable setting it will ever see — so a failure there is a failure
everywhere. `DIRECTION_AUDIT.md` §4.1 measured that trunk at 0.17 sim-s/real-s
against a 160K MLP's 22.97. If it passes a 5.0 floor, the instrument is wrong
and not the trunk. The control is the REAL module: `UnifiedBrain.TransformerBlock`
x8 plus its `PhysicsRuleBank` — 36.74M of trunk, 36.92M with the encoders and
readout the arm needs — wired as the arm's `encode`. `trunk_params` is asserted
above 30M so the control cannot pass by being a cheap imitation.

NULL BASELINE: physics and senses alone, zero action, no learner at all — the
ceiling no arm can exceed. Reported every run. If an arm ever exceeds the null
the instrument is measuring something other than what it claims.

BODY SANITY, carried here rather than assumed. PG.8's lesson is that seven
honest fixtures composed into an empty room; this is the first spec to put the
climber-rover in the world, so it asserts the body is present and live before it
reports how fast it runs: `model.nu == 6` MJCF actuators plus the 2-dimensional
gated drive = the 8 action dimensions `cores.ACTION_DIM` declares, actuation
moves the body measurably more than zero control does, and the drive gate is a
gate (`drive_gate_frac < 1.0`). A throughput number for a dead body would be
both true and worthless.

MEASURED AND REPORTED, NOT GATED — read this before using the rover for
anything about climbing: under random action the rover TOPPLES within ~20
decisions and then slides on its side (`upright_cos` goes to ~0). That is the
body `CURIOSITY_BAKEOFF.md` §2.3 specifies — a 30 kg capsule on a 0.09 m
spherical foot is an inverted pendulum and the rig has no balance mechanism —
and it is consistent with that document's own pilot (`z_rest` 0.360 m, zero
engaged ladder attempts in 9,000 random decisions). It does not affect this
spec's claim, which is about wall-clock. It very much affects LC.03's, and it is
queued as a BAKEOFF in `docs/INTEGRATION_QUEUE.md` rather than patched here —
picking a balance mechanism by argument is exactly what law 3 forbids.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..cores import (ACTION_DIM, CANDIDATE_ARMS, LC_BATCH, build_arm,
                     lc_targets, lc_update, make_optimizer, n_params)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..survival_rec import REC_ARMS, build_rec_arm
from ..w0 import W0, SIM_S_PER_DECISION

# ── CHALLENGER PRICING (ORDERS J3, 2026-11-13 23:00) ────────────────────
# LC.09 races two arms that CARRY STATE (survival_rec.REC_ARMS: wm-dyn and
# wm-latent+ring) and its registration says each runs at "LC.02's committed
# train_ratio". Their update is not CANDIDATE_ARMS' update: every replay row
# hands its stored recurrent state back through `core.use_stored` before
# `lc_update`, and every decision advances the state with `core.step` +
# `core.efference`. So they are timed through THEIR loop (`_run_rec`), never
# borrowed from a sibling's ratio. Same floor, same walk, same keys
# (`<arm>/clears@<r>`), same conservative `committed_ratio`. They are PRICED
# here, not admitted: they do not enter `throughput_floor_conjunction` (the
# registered claim is about CANDIDATE_ARMS and its gate is byte-unmoved). A
# challenger that clears no ratio is EXCLUDED from LC.09 by LC.09 itself
# (no committed ratio -> UnpricedArm), which is LC.02's own exclusion rule.
CHALLENGER_ARMS = REC_ARMS

# This spec certifies a property of the WORLD, so the world hashes into
# impl_sha. Change playground.py and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
# WIDENED 2026-08-10 to the two modules this spec actually times. It declared
# only `playground.py`, so a change to `w0.py` — the module whose decision loop
# IS the throughput being measured — left this PASS standing over code it had
# never run. The narrower list was not a smaller claim; it was a blind spot in
# exactly the direction the guard exists to cover.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             # 2026-10-20 widened domain (T0.35): nested `from ..cores import`
             "experiments/cores.py",
             # 11-13: the challengers' loop and cores are timed here too
             "experiments/cores_dyn.py", "experiments/cores_ring.py",
             "experiments/survival_rec.py"]

FLOOR = 5.0                      # sim-s per real-s, LEARNING_CORE.md §5.1
N_THREADS = 3                    # "3 ARM cores of this box"
NICE = 19
TRAIN_RATIOS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WARMUP = LC_BATCH                # fill the replay buffer; never timed
TIMED = 40                       # decisions inside the stopwatch
BUFFER = 256                     # decisions of replay; LC_BATCH is drawn from it

# j0 and alpha have no defaults in `drives.py` and PS.01 has not run, so they
# are passed explicitly here and are THROUGHPUT-ONLY: every arithmetic path in
# the drive integrator executes identically whatever their values, so they
# cannot move a wall-clock number, and nothing in this spec reads e, i or w.
# PS.01 must measure them before any spec reads the drive state as a quantity.
J0_TIMING_ONLY, ALPHA_TIMING_ONLY = 1.0, 0.01

# ── CONTENTION PRECONDITION (ORDERS J1 step 1, 2026-11-13) ──────────────
# This is a wall-clock claim on a SHARED 4-core box. Attempt 19's FAIL on
# 11-12 carried the PASS impl_sha and was taken while a detached race loaded
# the box; attempt 20's one red arm-seed (dreamer-xs 5.91) was taken at load
# ~2 and the same code read 7.41 on a quiet box (attempt 21, PASS). Other
# tenants' CPU can only make a decision SLOWER, never faster, so:
#   * a PASS earned under contention is a valid lower bound and STANDS;
#   * a would-be FAIL earned while other processes used more than
#     FOREIGN_MAX_CORES of the box inside any timed window is VOID (it could
#     not test the claim: the floor was measured on cores we did not have).
# The meter reads the thing itself, not loadavg (which counts our own 3
# threads): box-wide busy jiffies from /proc/stat minus this process's own
# utime+stime, over exactly the stopwatch window. 4 cores - our 3 threads =
# 1 core others may use before they take one of ours.
# It is proven live on every seed by a PLANTED load: two niced spinners for
# a fixed window must read >= PLANTED_MIN_CORES, else the meter is blind and
# the run is VOID (LESSONS_CORE 9). No bar, control or ratio moved.
FOREIGN_MAX_CORES = 1.0
PLANTED_SPINNERS = 2
PLANTED_MIN_CORES = 1.5
_CLK = os.sysconf("SC_CLK_TCK")


def _cpu_jiffies() -> tuple:
    """(box busy jiffies, this process's jiffies) — all threads included."""
    f = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    v = [int(x) for x in f]
    # user nice system idle iowait irq softirq steal [guest guest_nice]:
    # guest time is already inside user/nice, so it is not added again.
    busy = v[0] + v[1] + v[2] + v[5] + v[6] + (v[7] if len(v) > 7 else 0)
    st = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    own = int(st[11]) + int(st[12])          # utime, stime (fields 14, 15)
    return busy, own


def _foreign_cores(j0: tuple, j1: tuple, wall: float) -> float:
    """Cores' worth of CPU used by OTHER processes between two readings."""
    foreign = (j1[0] - j0[0]) - (j1[1] - j0[1])
    return max(0.0, foreign / _CLK / max(wall, 1e-9))


def _planted_load_reading(window: float = 1.0) -> float:
    """Known positive: plant PLANTED_SPINNERS busy processes, read the meter."""
    code = ("import os, time\nos.nice(19)\ne = time.time() + %f\n"
            "while time.time() < e:\n    pass\n" % (window + 0.6))
    procs = [subprocess.Popen([sys.executable, "-I", "-c", code])
             for _ in range(PLANTED_SPINNERS)]
    try:
        time.sleep(0.3)                      # past interpreter start-up
        j0, t0 = _cpu_jiffies(), time.perf_counter()
        time.sleep(window)
        j1, wall = _cpu_jiffies(), time.perf_counter() - t0
    finally:
        for p in procs:
            p.wait(timeout=10)
    return _foreign_cores(j0, j1, wall)


def _nice_and_threads() -> tuple:
    """Put this process where the spec says the measurement is made."""
    torch.set_num_threads(N_THREADS)
    cur = os.nice(0)
    if cur < NICE:
        os.nice(NICE - cur)      # nice() may only increase; never fails upward
    return float(os.nice(0)), float(torch.get_num_threads())


class _Buffer:
    """A ring of decisions, as tensors, one stack per modality key."""

    def __init__(self, obs, cap: int = BUFFER):
        self.cap = cap
        self.keys = list(obs)
        self.buf = {k: torch.zeros(cap, obs[k].shape[0]) for k in self.keys}
        self.n = 0

    def add(self, obs) -> None:
        i = self.n % self.cap
        for k in self.keys:
            self.buf[k][i] = torch.from_numpy(obs[k])
        self.n += 1

    def sample(self, gen, size: int = LC_BATCH):
        hi = min(self.n, self.cap)
        idx = torch.randint(0, hi, (size,), generator=gen)
        return {k: self.buf[k][idx] for k in self.keys}


def _to_tensor(obs):
    return {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}


def _run(seed: int, arm, train_ratio, timed: int = TIMED) -> dict:
    """Time `timed` decisions of one life. `arm=None` is the no-learner null."""
    w = W0(seed=seed, j0=J0_TIMING_ONLY, alpha=ALPHA_TIMING_ONLY)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)

    core = opt = buf = None
    if arm is not None:
        torch.manual_seed(9_000 + seed)
        core = build_arm(arm) if isinstance(arm, str) else arm(seed)
        core.eval()                                   # F1
        opt = make_optimizer(core)
        buf = _Buffer(w.observe())

    def one() -> None:
        obs = w.observe()
        if core is None:
            w.decide(np.zeros(ACTION_DIM))
            return
        buf.add(obs)
        with torch.no_grad():
            a = core.act_deterministic(_to_tensor(obs), dropped=W0.DROPPED)
        w.decide(a.squeeze(0).numpy())
        if train_ratio >= 1.0:
            n_up = int(train_ratio)
        else:
            n_up = 1 if (w.decisions % int(round(1.0 / train_ratio)) == 0) else 0
        for _ in range(n_up):
            lc_update(core, buf.sample(gen), lc_targets(gen), opt,
                      dropped=W0.DROPPED)

    for _ in range(WARMUP):
        one()
    j0 = _cpu_jiffies()
    t0 = time.perf_counter()
    for _ in range(timed):
        one()
    wall = time.perf_counter() - t0
    j1 = _cpu_jiffies()

    out = {"T": timed * SIM_S_PER_DECISION / wall,
           "foreign_cores": _foreign_cores(j0, j1, wall),
           "decisions_per_s": timed / wall,
           "core_s_per_1k": 1000.0 * wall / timed}
    out.update(w.report())
    if core is not None:
        out["params"] = float(n_params(core))
    return out


class _RecBuffer(_Buffer):
    """_Buffer + the recurrent state each row was acted from (R2D2 stored
    state, as survival_rec._StateRing), sampled with the SAME indices."""

    def __init__(self, obs, cap: int = BUFFER):
        super().__init__(obs, cap)
        self.st = None

    def add_rec(self, obs, stored) -> None:
        stored = stored if isinstance(stored, tuple) else (stored,)
        if self.st is None:
            self.st = [torch.zeros(self.cap, x.shape[-1]) for x in stored]
        i = self.n % self.cap
        for b, x in zip(self.st, stored):
            b[i] = x
        self.add(obs)

    def sample_rec(self, gen, size: int = LC_BATCH):
        hi = min(self.n, self.cap)
        idx = torch.randint(0, hi, (size,), generator=gen)
        st = tuple(b[idx] for b in self.st)
        return ({k: self.buf[k][idx] for k in self.keys},
                st if len(st) > 1 else st[0])


def _stored_now(core):
    """The state this decision is acted FROM (survival_rec's expression)."""
    from ..cores import DETER, LATENT
    if hasattr(core, "stored_now"):
        return core.stored_now()
    return ((core._h if core._h is not None
             else torch.zeros(1, DETER)).squeeze(0).clone(),
            (core._z if core._z is not None
             else torch.zeros(1, LATENT)).squeeze(0).clone(),
            core._a.squeeze(0).clone())


def _run_rec(seed: int, arm: str, train_ratio, timed: int = TIMED) -> dict:
    """`_run` for a core that carries state, through its REAL loop: step +
    efference every decision, stored state handed back on every update.
    Same world, seeds, warm-up, stopwatch and NOISE targets as `_run`."""
    w = W0(seed=seed, j0=J0_TIMING_ONLY, alpha=ALPHA_TIMING_ONLY)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)
    torch.manual_seed(9_000 + seed)
    core = build_rec_arm(arm)
    core.eval()
    opt = make_optimizer(core)
    buf = _RecBuffer(w.observe())

    def one() -> None:
        obs = w.observe()
        buf.add_rec(obs, _stored_now(core))
        obs_t = _to_tensor(obs)
        with torch.no_grad():
            s = core.step(obs_t, dropped=W0.DROPPED)
            a = core.act(obs_t, s["state"]).squeeze(0).numpy()
        core.efference(a)
        w.decide(a)
        if train_ratio >= 1.0:
            n_up = int(train_ratio)
        else:
            n_up = 1 if (w.decisions % int(round(1.0 / train_ratio)) == 0) else 0
        for _ in range(n_up):
            batch, st = buf.sample_rec(gen)
            core.use_stored(st)
            try:
                lc_update(core, batch, lc_targets(gen), opt,
                          dropped=W0.DROPPED)
            finally:
                core.use_stored(None)

    for _ in range(WARMUP):
        one()
    j0 = _cpu_jiffies()
    t0 = time.perf_counter()
    for _ in range(timed):
        one()
    wall = time.perf_counter() - t0
    j1 = _cpu_jiffies()
    out = {"T": timed * SIM_S_PER_DECISION / wall,
           "foreign_cores": _foreign_cores(j0, j1, wall),
           "decisions_per_s": timed / wall,
           "core_s_per_1k": 1000.0 * wall / timed,
           "params": float(n_params(core))}
    out.update(w.report())
    return out


def _body_sanity(seed: int) -> dict:
    """PG.8's question for the rover: is anybody home, and does the drive gate?

    Actuation is measured on the four ARM SLIDES between the two ctrl extremes,
    not on `qpos` as a whole. The first version of this check compared "zero
    control" against random control and read a NEGATIVE margin — because the
    rover topples either way and the free root's fall swamps every slide. A
    body-wide drift number cannot tell a working actuator from gravity, which
    is the T2.00 lesson (measure the quantity you are claiming, not a proxy)
    arriving on a second body.
    """
    import playground as pg

    def settle(level: float) -> np.ndarray:
        w = W0(seed=seed, j0=J0_TIMING_ONLY, alpha=ALPHA_TIMING_ONLY)
        a = np.zeros(ACTION_DIM)
        a[:4] = level
        for _ in range(20):
            w.decide(a)
        qa = w.ix["jnt_qposadr"]
        return w, np.array([float(w.data.qpos[qa[n]])
                            for n in ("reachL", "liftL", "reachR", "liftR")])

    w_lo, lo = settle(-1.0)
    w_hi, hi = settle(+1.0)
    r = w_hi.report()
    return {"nu": float(w_hi.model.nu), "action_dim": float(w_hi.action_dim),
            "expected_action_dim": float(ACTION_DIM),
            "mjcf_nu_expected": float(pg.ROVER_NU),
            "slide_qpos_ctrl_lo": float(np.abs(lo).max()),
            "slide_qpos_ctrl_hi": float(np.abs(hi).max()),
            "actuation_margin": float(np.abs(hi - lo).max()),
            "drive_gate_frac": r["drive_gate_frac"],
            "upright_cos_after_20": r["upright_cos"]}


def _walk(seed: int, arm: str, runner, m: dict, foreign: list) -> int:
    """THE SELECTION RULE for one arm on one seed; writes its keys into m."""
    chosen, t_min, n_measured = None, None, 0
    clears = {r: 0.0 for r in TRAIN_RATIOS}
    times: dict = {r: None for r in TRAIN_RATIOS}
    for r in TRAIN_RATIOS:
        rr = runner(seed, arm, r)
        t = rr["T"]
        foreign.append(rr["foreign_cores"])
        times[r], n_measured = t, n_measured + 1
        if t_min is None:
            t_min = t
        if t >= FLOOR:
            clears[r], chosen = 1.0, r
        else:
            # Monotonicity, not an assumption: more gradient steps per
            # decision cannot make a decision cheaper. Everything above a
            # failing ratio fails, so the walk stops and those ratios are
            # recorded as not-cleared with their time left UNMEASURED
            # (null, never a zero — LESSONS.md, "a default of zero is not
            # 'unknown'"). `ratios_measured` says how far the walk got.
            break
    for r in TRAIN_RATIOS:
        m[f"{arm}/clears@{r}"] = clears[r]
        m[f"{arm}/T@{r}"] = times[r]
    m[f"{arm}/train_ratio_this_seed"] = float(chosen or 0.0)
    m[f"{arm}/cleared"] = float(chosen is not None)
    m[f"{arm}/T_at_min_ratio"] = float(t_min)
    m[f"{arm}/ratios_measured"] = float(n_measured)
    return int(chosen is not None)


def _experiment(seed: int) -> dict:
    nice_level, threads = _nice_and_threads()
    m: dict = {"nice": nice_level, "torch_threads": threads,
               "floor": FLOOR, "timed_decisions": float(TIMED)}
    m.update(_body_sanity(seed))
    planted = _planted_load_reading()
    m["meter_planted_cores"] = planted
    m["meter_live"] = float(planted >= PLANTED_MIN_CORES)
    foreign: list = []

    null = _run(seed, None, None)
    foreign.append(null["foreign_cores"])
    m["null_T"] = null["T"]
    m["null_decisions_per_s"] = null["decisions_per_s"]

    n_cleared = 0
    for arm in CANDIDATE_ARMS:
        n_cleared += _walk(seed, arm, _run, m, foreign)
    # Challengers: priced, never gating (see CHALLENGER_ARMS).
    m["challengers_cleared"] = float(sum(
        _walk(seed, arm, _run_rec, m, foreign) for arm in CHALLENGER_ARMS))
    m["challengers_probed"] = float(len(CHALLENGER_ARMS))

    m["foreign_cores_max"] = float(max(foreign))
    # Per seed 0/1, so the across-seed MEAN is > 0 iff ANY seed was contended.
    m["contended"] = float(max(foreign) > FOREIGN_MAX_CORES)
    m["arms_cleared"] = float(n_cleared)
    m["arms_probed"] = float(len(CANDIDATE_ARMS))
    m["throughput_floor_conjunction"] = float(
        n_cleared == len(CANDIDATE_ARMS) and m["null_T"] >= FLOOR)
    return m


def committed_ratio(m: dict, arm: str):
    """The train_ratio LC.03 must use for `arm`, derived from the ledger entry.

    THE CONSERVATIVE RULE, fixed here before the run: the largest power-of-two
    ratio that cleared the floor on EVERY seed. `clears@r` aggregates to a mean
    over seeds, so `== 1.0` is exactly "every seed cleared it".

    The rejected alternative was "require the seeds to agree, otherwise FAIL".
    It is stricter but wrong: an arm whose throughput straddles the floor is a
    fact about the hardware, not a broken measurement, and answering it with a
    red ladder entry would push the next iteration toward the floor rather than
    toward the ratio. Taking the minimum can only ever licence LESS compute than
    a seed can afford, which is the direction a selection step should err in.
    """
    ok = [r for r in TRAIN_RATIOS if m.get(f"{arm}/clears@{r}", 0.0) == 1.0]
    return max(ok) if ok else None


def _control(seed: int) -> dict:
    """The 57M UnifiedBrain trunk on the control path, at the EASIEST ratio."""
    from ..cores import TRUNK_CONTROL_ARM

    r = _run(seed, TRUNK_CONTROL_ARM, TRAIN_RATIOS[0], timed=8)
    return {"trunk_T": r["T"], "trunk_decisions_per_s": r["decisions_per_s"],
            "trunk_params": r["params"], "trunk_ratio": TRAIN_RATIOS[0]}


def _check(m: dict, c: dict):
    # ── the instrument first ────────────────────────────────────────────
    if m.get("nu", -1.0) != m.get("mjcf_nu_expected", -2.0):
        return Status.VOID                      # not the body this spec times
    if m.get("action_dim", -1.0) != m.get("expected_action_dim", -2.0):
        return Status.VOID
    if m.get("actuation_margin", 0.0) <= 0.10:
        return Status.VOID                      # nobody home: a dead body's
        # throughput is a true number about nothing (PG.8)
    if m.get("drive_gate_frac", 1.0) >= 1.0:
        return Status.VOID                      # the gate is not a gate
    if c.get("trunk_params", 0.0) < 30e6:
        return Status.VOID                      # the control is not the trunk
    if c.get("trunk_T", 0.0) >= FLOOR:
        return Status.VOID                      # a 36.7M transformer clearing a
        # 5.0 floor means the stopwatch, not the trunk, is what we measured

    if m.get("meter_live", 0.0) < 1.0:
        return Status.VOID                      # the contention meter could
        # not see a planted load on some seed: it is blind, so a FAIL below
        # could not be told from contention

    # ── the claim ───────────────────────────────────────────────────────
    # Every admissible arm must have a train_ratio it can afford ON EVERY SEED.
    # An arm with none is EXCLUDED (the spec's `falsified_by`) and the
    # hypothesis "every admissible arm sustains 5.0" is false.
    ok = (m.get("null_T", 0.0) >= FLOOR
          and all(committed_ratio(m, arm) is not None for arm in CANDIDATE_ARMS)
          and m.get("throughput_floor_conjunction", 0.0) == 1.0)
    if ok:
        return True                             # contention only slows: a
        # PASS under load is a lower bound and stands
    if m.get("contended", 1.0) > 0.0:
        return Status.VOID                      # a red measured on cores other
        # tenants were using (> FOREIGN_MAX_CORES in a timed window) cannot
        # test the claim; re-run on a quiet box
    if m.get("null_T", 0.0) < FLOOR:
        return Status.VOID                      # the WORLD cannot be lived in
        # at this speed; no arm can, and that is a world result, not an arm one
    return False


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LC.02"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
