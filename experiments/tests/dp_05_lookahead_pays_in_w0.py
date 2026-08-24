"""DP.05 — Lookahead pays in the world he ACTUALLY lives in.

DP.00 PASSED (2026-08-10, gap 75.8 steps) and was cited since as "Jack's world
rewards deliberation" — but it ran in LC.00's 12x12 tabular gridworld, not in
W0. This spec re-points DP.00's design, unchanged in structure, at the MuJoCo
climber-rover world with drives, food, water and death that every learning-core
number is actually about. The registry entry says why; this docstring says how.

  The only difference between the arms is PLANNING DEPTH.
  Same candidate machinery, same scorer, same world, same spawns.

DESIGN, mirrored from dp_00_lookahead_pays.py piece by piece:

WORLD. `W0(seed, lethal=True)`, j0/alpha borrowed live from PS.01's ledger row
(`borrow_metrics` — a refusal is VOID, never a default). ONE declared change,
exactly as DP.00 declared LIFE_CAP=200: **E0 = 0.20** (XL.00's precedent for
piloting fast deaths; `run_survival` exposes the same knob). A resting body at
E0 starves in E0/BASAL_B = 120 sim-s, so lives are decidable inside the cpu<2h
budget; LIFE_CAP_DEC = 1000 decisions (200 sim-s) censors both arms, and
censoring can only UNDERSTATE the planner's advantage — the test gets harder,
never easier. At E0=0.20 the gear scale starts at ~0.52 (GEAR_FLOOR + 0.6*e):
both arms live in the same weakened body; declared, not hidden.

THE PLANNER'S MODEL IS THE SIMULATOR — literally, not by duplication. DP.00
mirrored the step function and policed the copy with fidelity probes. Here
there is no copy at all: candidates are rolled out in the live W0 via full
state snapshot/restore (qpos/qvel/act/qacc_warmstart/time + every mutable
DriveLayer field + the spawn RNG), so the model cannot drift from the world by
construction. What CAN fail is the snapshot itself — a missed field would leak
rollout state into the real timeline — so the fidelity gate probes exactly
that: restore must reproduce a 3-decision trace BYTE-IDENTICALLY (qpos/qvel
bytes, drives, food timers, death counters), a different action sequence must
diverge (a restore check that cannot fail is not a check), and the probes must
exercise the EAT and DEATH branches, engineered if necessary — a probe set
that never ate and never died would certify the interior of the dynamics and
miss the transitions that decide a life (DP.00's own probe lesson).

ARMS, all driven by `_plan` with identical candidate sets per call:
  react_k5    H=1 (one 0.2 s decision), K=5 candidates. The best no-lookahead
              policy this machinery can express: it sees the one-step drive
              gradient and nothing else. At one decision the drive change is
              near action-independent (basal drain dominates; arm power is the
              only lever), which is the world's own statement that it offers no
              immediate gradient — DP.00's react_greedy, verbatim.
  react_k10   H=1, K=10 — the STRENGTHENED null: the reactive arm gets TWICE
              the planner's per-call search, so "the planner just searches
              more" is answered inside the design (DP.00's react_persist role;
              the reported null is the per-seed MAX of the two).
  plan_h4     H=4 (0.8 s), K=5, replan every 4.
  plan_h10    H=10 (2.0 s), K=5, replan every 5 — the GATED arm.
Candidates are hold-constant 8-vectors: [previous executed action] + fresh
uniform draws. H_MAX=10 is NOT "unlimited lookahead" and this line says so out
loud, exactly as DP.00 said it of H=8: every gap below is a LOWER BOUND on
what deeper/denser search would find, which can only strengthen a PASS, and
the H=4/H=10 sweep must show dose-response for the gap to be attributable to
lookahead at all.

METRIC `gap_s` = mean lifespan(plan_h10) - max(mean lifespan(react_k5),
mean lifespan(react_k10)), in SIMULATED SECONDS, N_LIVES per arm per seed,
identical spawn sequences across arms (same seed -> same W0 -> same
spawn-RNG draw per life index). Lifespan, not return, for DP.00's reason:
drive return telescopes; lifespan is the consequential quantity.

ATTAINABLE RANGE, computed before the gates were chosen. Lifespan lies in
(0, 200]; the no-food ceiling for a RESTING body is 120 s and every mover
sits below it, so the exploitable range above the best non-eater is roughly
80 s. MIN_GAP_S = 20.0 is 25% of that range; one floor food (+0.08 e) buys
48 s of basal life, so the margin is under half of one exploited food.
SIGMA_GATE = 3.0, unpaired, bakeoff.py's ruler.

THE REFERENCE ARM (SH.01's lesson: pilot the must-succeed arm first, and keep
it in the registered run as a gate). A scripted chaser that drives straight at
the nearest ACTIVE floor food using privileged state. It is an INSTRUMENT, not
an arm: if the reference cannot eat (>= 3 events) and outlive the resting
no-food ceiling by 10%, then food does not pay at this envelope, no arm could
have shown lookahead paying, and the verdict is VOID (rig), never FAIL.
MEASURED before registration (PILOT RECORD): seed 90 lives to the 200 s cap
with 8 eats; seed 91 183.8 s with 6 — food pays, so FAIL would be meaningful.

CONTROL — DP.00's, re-pointed: W0 DISARMED (lethal=False, so no death; no
needs in the objective) with a DENSE IMMEDIATE reward: episodes start at a
legal spawn >= 4 m from a beacon (another legal spawn), reward
0.02 * (dist_before - dist_after) per decision, capped at 100 decisions.
Greedy descent on a dense potential is near-optimal for a driven rover on
mostly-flat ground, so planning must gain ~nothing there. MuJoCo momentum
makes "exactly nothing" false in a way a gridworld never was, so the
tolerance is calibrated, in the open, on pilot seeds 90-91 (disjoint from run
seeds 0-2) and fixed before the registered run. Three gates, DP.00 verbatim:
  ctrl_gain          plan return minus react return <= CTRL_TOL.
  ctrl_gain_broken   react must beat uniform-random actions by >=
                     CTRL_BROKEN_FLOOR — the control's own positive control.
  (react-optimality has no provable analog in MuJoCo; its job — proving the
  null arm is not handicapped — is carried by react_k10's doubled search.)

VOID rather than FAIL when the instrument failed (order matters, DP.00's):
  - PS.01's constants refuse to borrow (stale/absent source);
  - any fidelity probe mismatches, the divergence check cannot fire, or the
    eat/death branches went unexercised;
  - the reference chaser cannot demonstrate exploitable food structure;
  - the control's positive control cannot fire;
  - the control itself shows a gain (the metric is measuring compute).
FAIL fires only past every gate: rig sound, food exploitable, and the planner
still cannot beat the best reactive arm by 20 s at 3 sigma. Then, at THIS
declared K x H envelope, lookahead buys nothing in W0 — the finding DP.05's
registry entry names: fix the world (traps, delays, irreversibility) before
any dual-process claim is made in it, and BO.01 does not run.

CALIBRATION LEDGER (constants fixed against seeds 90-91, disjoint from run
seeds; DP.00's precedent):
  - decision cost measured 2026-08-24 (seed 90, this box, nice 19): 28.7 ms
    plain, 29.1 ms per rollout decision incl. snapshot/restore. The envelope
    above was SHRUNK to fit that price (E0 0.25->0.20, LIFE_CAP 1500->1000,
    K 6->5, K_strong 12->10, REPLAN[4] 2->4, control 8x100->6x80): worst-case
    ~115 min for 3 seeds inside cpu<2h.
  - CTRL_TOL = 0.02 / CTRL_BROKEN_FLOOR = 0.04 were written before the pilot
    and HELD: measured ctrl_gain -0.0058 (s90) / -0.0276 (s91) — the planner
    does not gain, it slightly loses to momentum in the dense world — and
    broken_gap 0.1117 / 0.1252, nearly 3x the floor.

PILOT RECORD (2026-08-24, seeds 90-91; a number measured off-run is a design
input, never a finding):
  - fidelity: FIRST RUN CAUGHT TWO REAL HOLES. (1) Water.apply writes a
    body's xfrc row only while it is in the pool, so a body that leaves keeps
    its last force row forever — the whole array is dynamics state and is now
    snapshotted. (2) W0 carries one-substep-stale xipos/cvel/contacts across
    decision boundaries; restore's mj_forward refreshes them, so live-vs-
    restored traces diverged at 1e-14 on seed 90 (a floating object). Fixed by
    refreshing EVERY compared path at the snapshot boundary (see _probe_once).
    After both: probe_mismatch 0, diverge_ok 1, ate 2, died 2 on both seeds.
  - reference: seed 90 span 200.0 s (cap) / 8 eats; seed 91 183.8 s / 6 eats,
    against the 132 s gated ceiling.
  - control: numbers above; both gates hold with the pre-written constants.
  - arm smoke (LIFE_CAP=60, not a finding): the machinery runs end to end;
    in a 12 s window plan_h10 ate once, react_k5 zero times.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from .. import drives                                             # noqa: E402
from ..w0 import W0, SIM_S_PER_DECISION                           # noqa: E402

IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py"]

# ── the declared envelope (sized to cpu<2h at 29 ms/decision, see PILOT) ───
E0 = 0.20                     # starting/respawn energy; 120 s resting ceiling
LIFE_CAP_DEC = 1000           # 200 sim-s; censors BOTH arms (harder, not easier)
N_LIVES = 2                   # lives per arm per seed
K_CANDS = 5                   # planner candidate count per call
K_REACT_STRONG = 10           # the strengthened null's doubled search
H_SWEEP = (4, 10)             # planning depths reported beside H=1
H_MAX = 10                    # the gated horizon: 2.0 s of lookahead
REPLAN = {1: 1, 4: 4, 10: 5}  # execute this many actions per planning call
GAMMA = 0.95                  # per-decision discount, DP.00's

MIN_GAP_S = 20.0              # sim-seconds, every seed
SIGMA_GATE = 3.0              # bakeoff.py's ruler, unpaired

NO_FOOD_CEILING_S = E0 / drives.BASAL_B      # 150.0; asserted below
REF_MIN_EATS = 3
REF_HEADROOM = 1.1            # reference must outlive the ceiling by 10%

# ── the control world ──────────────────────────────────────────────────────
CTRL_EPISODES = 6
CTRL_CAP_DEC = 80             # 16 s per episode
CTRL_MIN_D = 4.0              # m, start-to-beacon
CTRL_SHAPE = 0.02             # per metre closed — DP.00's FLAT_SHAPE, in metres
CTRL_DONE_D = 0.6             # m, episode ends inside this radius
CTRL_TOL = 0.02               # return units; pilot-calibrated (PILOT RECORD)
CTRL_BROKEN_FLOOR = 0.04      # react minus broken must exceed this

# ── fidelity probes ────────────────────────────────────────────────────────
N_PROBE_GENERIC = 4
N_PROBE_EAT = 2
N_PROBE_DEATH = 2
PROBE_TRACE_DEC = 3           # decisions per replayed trace

PILOT_SEEDS = (90, 91)        # calibration only; run_spec uses 0-2

# The floor/ceiling the gates were checked against; asserted so they cannot rot.
assert abs(NO_FOOD_CEILING_S - 120.0) < 1e-9, "E0 or BASAL_B moved; re-derive"
assert MIN_GAP_S < (LIFE_CAP_DEC * SIM_S_PER_DECISION - NO_FOOD_CEILING_S), (
    "the gap gate exceeds the attainable range above the no-food ceiling")
assert H_MAX in H_SWEEP, "the gated horizon must be one the run measures"
assert all(h in REPLAN for h in (1,) + H_SWEEP), "every horizon needs a replan"


# ── PS.01's measured constants, borrowed live (refusal preserved) ──────────
_BORROW_CACHE = None


def _borrowed():
    global _BORROW_CACHE
    if _BORROW_CACHE is None:
        _BORROW_CACHE = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    return _BORROW_CACHE


def _mkworld(seed: int, lethal: bool) -> W0:
    b = _borrowed()
    return W0(seed=seed, j0=b.values["j0_ms"], alpha=b.values["alpha"],
              lethal=lethal)


# ── snapshot/restore: the model IS the simulator ───────────────────────────
_W0_SCALARS = ("decisions", "sim_seconds", "drive_gate_open",
               "audio_events_total", "died_this_decision", "last_death_cause",
               "life", "deaths", "_life_started_at")
_DRV_SCALARS = ("state", "t", "_submerged_since", "_touching_world",
                "_prev_speed", "last_j", "last_power_w", "last_dt",
                "last_rest_dt", "n_onsets")


def _snap(w: W0) -> dict:
    d = w.data
    return {
        "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
        "warm": d.qacc_warmstart.copy(), "time": float(d.time),
        "ctrl": d.ctrl.copy(),
        # Water.apply writes a body's force row only WHILE it is in the pool;
        # a body that leaves keeps its last row forever, so the whole array is
        # dynamics state (found by the seed-90 fidelity pilot, 5/8 mismatches).
        "xfrc": d.xfrc_applied.copy(),
        "w0": {k: getattr(w, k) for k in _W0_SCALARS},
        "lists": (list(w.life_lengths), list(w.death_sites),
                  list(w.spawn_sites)),
        "prev_drive": w._prev_drive,          # DriveState: replaced, not mutated
        "audio": w._audio.copy(),
        "spawn_rng": w._spawn_rng.get_state(),
        "vis_rng": w._rng.get_state(),
        "drv": {k: getattr(w.drives, k) for k in _DRV_SCALARS},
        "respawn_at": dict(w.drives._respawn_at),
        "ate_total": dict(w.drives.ate_total),
    }


def _restore(w: W0, s: dict) -> None:
    d = w.data
    d.qpos[:] = s["qpos"]
    d.qvel[:] = s["qvel"]
    if d.act.size:
        d.act[:] = s["act"]
    d.time = s["time"]
    d.ctrl[:] = s["ctrl"]
    d.xfrc_applied[:] = s["xfrc"]
    w.mujoco.mj_forward(w.model, w.data)
    d.qacc_warmstart[:] = s["warm"]           # after forward, which rewrites it
    w.mujoco.mj_rnePostConstraint(w.model, w.data)
    for k, v in s["w0"].items():
        setattr(w, k, v)
    w.life_lengths, w.death_sites, w.spawn_sites = (
        list(s["lists"][0]), list(s["lists"][1]), list(s["lists"][2]))
    w._prev_drive = s["prev_drive"]
    w._audio = s["audio"].copy()
    w._spawn_rng.set_state(s["spawn_rng"])
    w._rng.set_state(s["vis_rng"])
    for k, v in s["drv"].items():
        setattr(w.drives, k, v)
    w.drives._respawn_at = dict(s["respawn_at"])
    w.drives.ate_total = dict(s["ate_total"])
    w.synth.events = []                       # rollout sounds do not leak back


# ── one scorer for every arm: potential-based, DP.00's reward re-pointed ───
def _phi_drive(w: W0) -> float:
    return -w.drives.state.d()


def _score_rollout(w: W0, action: np.ndarray, horizon: int, phi) -> float:
    total, g = 0.0, 1.0
    for _ in range(horizon):
        p0 = phi(w)
        w.decide(action)
        total += g * (phi(w) - p0)
        g *= GAMMA
        if w.died_this_decision:
            break                             # DP.00: dead -> r only
    return total


def _plan(w: W0, rng: np.random.RandomState, prev_a: np.ndarray,
          horizon: int, k: int, phi) -> np.ndarray:
    """Best of k hold-constant candidates under depth-`horizon` rollout.

    The candidate set is [previous executed action] + (k-1) fresh uniform
    draws; every arm uses this same function, so the ONLY thing that differs
    between arms is `horizon` (and react_k12's k, which only ever helps the
    null). Rollouts run in the live simulator under snapshot/restore.
    """
    cands = [np.asarray(prev_a, dtype=float)]
    cands += [rng.uniform(-1.0, 1.0, w.action_dim) for _ in range(k - 1)]
    snap = _snap(w)
    best_a, best_v = cands[0], -1e18
    for a in cands:
        _restore(w, snap)
        v = _score_rollout(w, a, horizon, phi)
        if v > best_v:
            best_v, best_a = v, a
    _restore(w, snap)
    return best_a


# ── the survival arms ──────────────────────────────────────────────────────
def _fresh_life(w: W0, first: bool) -> None:
    if not first:
        w.respawn()                           # one spawn-RNG draw per life end,
    w.drives.state = drives.DriveState(e=E0)  # same count in every arm
    w._prev_drive = drives.DriveState(e=E0)


def _run_arm(seed: int, horizon: int, k: int) -> dict:
    w = _mkworld(seed, lethal=True)
    w.legal_spawns()                          # prime the cache off the clock
    rng = np.random.RandomState(seed * 9973 + horizon * 131 + k)
    spans, eats = [], 0
    for life in range(N_LIVES):
        _fresh_life(w, first=(life == 0))
        start = w.sim_seconds
        prev_a = np.zeros(w.action_dim)
        queue: list = []
        dec = 0
        ate0 = sum(w.drives.ate_total.values())
        while dec < LIFE_CAP_DEC:
            if not queue:
                a = _plan(w, rng, prev_a, horizon, k, _phi_drive)
                queue = [a] * REPLAN[horizon]
            a = queue.pop(0)
            w.decide(a)
            prev_a = a
            dec += 1
            if w.died_this_decision:
                break
        spans.append(min(w.sim_seconds - start,
                         LIFE_CAP_DEC * SIM_S_PER_DECISION))
        eats += sum(w.drives.ate_total.values()) - ate0
    return {"span": float(np.mean(spans)), "eats": float(eats)}


# ── the reference chaser: an instrument, not an arm ────────────────────────
def _food_target(w: W0):
    """(x, y) of the nearest ACTIVE floor food, or the nearest either way."""
    t = w.drives.t
    px, py = (float(w.data.xpos[w.rover_bid][0]),
              float(w.data.xpos[w.rover_bid][1]))
    best, best_d, best_active = None, 1e18, False
    for name in ("obj0", "obj1"):
        ent = w.drives._food.get(name)
        if ent is None:
            continue
        gid, _nu = ent
        gx, gy = float(w.data.geom_xpos[gid][0]), float(w.data.geom_xpos[gid][1])
        active = t >= w.drives._respawn_at[name]
        dist = math.hypot(gx - px, gy - py)
        if (active, -dist) > (best_active, -best_d):
            best, best_d, best_active = (gx, gy), dist, active
    return best


def _reference(seed: int) -> dict:
    w = _mkworld(seed, lethal=True)
    w.legal_spawns()
    spans, eats = [], 0
    for life in range(N_LIVES):
        _fresh_life(w, first=(life == 0))
        start = w.sim_seconds
        dec = 0
        ate0 = sum(w.drives.ate_total.values())
        while dec < LIFE_CAP_DEC:
            a = np.zeros(w.action_dim)
            tgt = _food_target(w)
            if tgt is not None:
                px, py = (float(w.data.xpos[w.rover_bid][0]),
                          float(w.data.xpos[w.rover_bid][1]))
                vx, vy = tgt[0] - px, tgt[1] - py
                n = math.hypot(vx, vy)
                if n > 1e-9:
                    a[6], a[7] = vx / n, vy / n
            w.decide(a)
            dec += 1
            if w.died_this_decision:
                break
        spans.append(min(w.sim_seconds - start,
                         LIFE_CAP_DEC * SIM_S_PER_DECISION))
        eats += sum(w.drives.ate_total.values()) - ate0
    return {"ref_span": float(np.mean(spans)), "ref_eats": float(eats)}


# ── fidelity: the snapshot IS complete, proven per branch ──────────────────
def _trace(w: W0, seq) -> list:
    out = []
    for a in seq:
        w.decide(a)
        s = w.drives.state
        out.append((w.data.qpos.tobytes(), w.data.qvel.tobytes(),
                    s.e, s.i, s.w, w.drives.t, w.deaths,
                    tuple(sorted(w.drives._respawn_at.items())),
                    tuple(sorted(w.drives.ate_total.items()))))
    return out


def _probe_once(w: W0, rng) -> dict:
    """One snapshot at the current state: replay-equal AND divergence-able.

    Every trace starts from `_restore(snap)` — including the first — because
    the snapshot boundary is REFRESHED: W0 carries one-substep-stale derived
    state (xipos/cvel, which Water.apply reads; contacts, which the drive gate
    reads) across decision boundaries, and `_restore`'s mj_forward recomputes
    it. `_plan` passes the executed path through the same refresh at every
    planning call, so what the probes certify is exactly the boundary the arms
    run on. (Found by the seed-90 pilot: a floating object's stale water force
    made live-vs-restored traces diverge at 1e-14 and grow.)"""
    seq = [rng.uniform(-1.0, 1.0, w.action_dim) for _ in range(PROBE_TRACE_DEC)]
    alt = [rng.uniform(-1.0, 1.0, w.action_dim) for _ in range(PROBE_TRACE_DEC)]
    snap = _snap(w)
    _restore(w, snap)
    t1 = _trace(w, seq)
    _restore(w, snap)
    t2 = _trace(w, seq)
    _restore(w, snap)
    t3 = _trace(w, alt)
    _restore(w, snap)
    ate = sum(n for _, n in t1[-1][8]) - sum(
        n for _, n in snap["ate_total"].items())
    died = t1[-1][6] - snap["w0"]["deaths"]
    return {"match": float(t1 == t2), "diverged": float(t1 != t3),
            "ate": float(ate), "died": float(died)}


def _fidelity(seed: int) -> dict:
    w = _mkworld(seed, lethal=True)
    w.legal_spawns()
    w.drives.state = drives.DriveState(e=E0)
    rng = np.random.RandomState(seed * 31337 + 5)
    for _ in range(25):                       # burn-in off the spawn pose
        w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
    mism = div = ate_n = died_n = 0
    for _ in range(N_PROBE_GENERIC):
        r = _probe_once(w, rng)
        mism += int(r["match"] != 1.0)
        div += int(r["diverged"] == 1.0)
        for _ in range(10):
            w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
    # EAT branch, engineered: park him touching-distance from obj0, timer live.
    for _ in range(N_PROBE_EAT):
        ent = w.drives._food.get("obj0")
        if ent is None:
            break
        gid, _nu = ent
        gx, gy = float(w.data.geom_xpos[gid][0]), float(w.data.geom_xpos[gid][1])
        w._place(gx - 0.25, gy)
        w.mujoco.mj_forward(w.model, w.data)
        w.drives._respawn_at["obj0"] = 0.0
        w.drives.state = drives.DriveState(e=0.5)
        eat_seq_rng = np.random.RandomState(int(rng.randint(2 ** 31)))

        class _EatRng:                        # drive straight at the food
            def uniform(self, lo, hi, n):
                a = eat_seq_rng.uniform(lo, hi, n) * 0.0
                a[6] = 1.0
                return a
        r = _probe_once(w, _EatRng())
        mism += int(r["match"] != 1.0)
        ate_n += int(r["ate"] > 0)
    # DEATH branch, engineered: one basal decision from the floor.
    for _ in range(N_PROBE_DEATH):
        w.drives.state = drives.DriveState(e=2e-4)
        r = _probe_once(w, rng)
        mism += int(r["match"] != 1.0)
        died_n += int(r["died"] > 0)
        w.drives.state = drives.DriveState(e=E0)   # back from the brink
        for _ in range(5):
            w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
    return {"probe_mismatch": float(mism),
            "probe_diverge_ok": float(div == N_PROBE_GENERIC),
            "probe_ate": float(ate_n), "probe_died": float(died_n)}


# ── the control: disarmed world, dense reward, planning must not gain ──────
def _ctrl_episodes(w: W0, seed: int) -> list:
    legal = w.legal_spawns()
    setup = np.random.RandomState(seed * 6151 + 17)   # identical per arm
    eps = []
    while len(eps) < CTRL_EPISODES:
        i, j = setup.randint(len(legal)), setup.randint(len(legal))
        (sx, sy), (bx, by) = legal[i], legal[j]
        if math.hypot(bx - sx, by - sy) >= CTRL_MIN_D:
            eps.append(((float(sx), float(sy)), (float(bx), float(by))))
    return eps


def _run_ctrl_arm(seed: int, horizon: int, broken: bool = False) -> float:
    w = _mkworld(seed, lethal=False)
    eps = _ctrl_episodes(w, seed)
    rng = np.random.RandomState(seed * 7907 + horizon * 101 + int(broken))
    rets = []
    for (sx, sy), (bx, by) in eps:
        w.respawn(at=(sx, sy))

        def phi(wv: W0) -> float:
            px, py = (float(wv.data.xpos[wv.rover_bid][0]),
                      float(wv.data.xpos[wv.rover_bid][1]))
            return -CTRL_SHAPE * math.hypot(bx - px, by - py)

        ret = 0.0
        prev_a = np.zeros(w.action_dim)
        queue: list = []
        for _ in range(CTRL_CAP_DEC):
            if broken:
                a = rng.uniform(-1.0, 1.0, w.action_dim)
            else:
                if not queue:
                    a = _plan(w, rng, prev_a, horizon, K_CANDS, phi)
                    queue = [a] * REPLAN[horizon]
                a = queue.pop(0)
            p0 = phi(w)
            w.decide(a)
            ret += phi(w) - p0                # undiscounted realised return
            prev_a = a
            if -phi(w) / CTRL_SHAPE < CTRL_DONE_D:
                break
        rets.append(ret)
    return float(np.mean(rets))


# ── the spec ───────────────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    b = _borrowed()
    if not b.ok:
        return {"borrow_ok": 0.0}
    m = {"borrow_ok": 1.0}
    m.update(_fidelity(seed))
    m.update(_reference(seed))
    r5 = _run_arm(seed, 1, K_CANDS)
    r10 = _run_arm(seed, 1, K_REACT_STRONG)
    m["react_k5"], m["react_k5_eats"] = r5["span"], r5["eats"]
    m["react_k10"], m["react_k10_eats"] = r10["span"], r10["eats"]
    for h in H_SWEEP:
        p = _run_arm(seed, h, K_CANDS)
        m[f"plan_h{h}"], m[f"plan_h{h}_eats"] = p["span"], p["eats"]
    m["react_best"] = max(m["react_k5"], m["react_k10"])
    m["gap_s"] = m[f"plan_h{H_MAX}"] - m["react_best"]
    m["gap_clear"] = float(m["gap_s"] >= MIN_GAP_S)
    return m


def _control(seed: int) -> dict:
    b = _borrowed()
    if not b.ok:
        return {"ctrl_borrow_ok": 0.0}
    plan = _run_ctrl_arm(seed, H_MAX)
    react = _run_ctrl_arm(seed, 1)
    broken = _run_ctrl_arm(seed, 1, broken=True)
    return {"ctrl_borrow_ok": 1.0,
            "ctrl_plan_ret": plan, "ctrl_react_ret": react,
            "ctrl_broken_ret": broken,
            "ctrl_gain": plan - react,
            "ctrl_gain_broken": react - broken}


def _sigma(mean_a, std_a, mean_b, std_b) -> float:
    return (mean_a - mean_b) / max(std_a, std_b, 1e-9)


def _check(m: dict, c: dict):
    # --- the instrument, before the hypothesis -------------------------
    if m.get("borrow_ok", 0.0) != 1.0 or c.get("ctrl_borrow_ok", 0.0) != 1.0:
        return Status.VOID          # PS.01's constants refused to borrow
    if m.get("probe_mismatch", 1.0) != 0.0:
        return Status.VOID          # the snapshot is not the simulator
    if m.get("probe_diverge_ok", 0.0) != 1.0:
        return Status.VOID          # a restore check that cannot fail
    if m.get("probe_ate", 0.0) <= 0.0 or m.get("probe_died", 0.0) <= 0.0:
        return Status.VOID          # the probes missed the deciding branches
    if (m.get("ref_eats", 0.0) < REF_MIN_EATS
            or m.get("ref_span", 0.0) <= NO_FOOD_CEILING_S * REF_HEADROOM):
        return Status.VOID          # food does not pay at this envelope
    if c.get("ctrl_gain_broken", 0.0) < CTRL_BROKEN_FLOOR:
        return Status.VOID          # the control's gate could not have fired
    if c.get("ctrl_gain", 1e9) > CTRL_TOL:
        return Status.VOID          # the gap is compute, not lookahead

    # --- the hypothesis -------------------------------------------------
    if m.get("gap_clear", 0.0) != 1.0:
        return False                # some seed missed the 20 s margin
    return _sigma(m[f"plan_h{H_MAX}"], m.get(f"plan_h{H_MAX}_std", 0.0),
                  m["react_best"], m.get("react_best_std", 0.0)) >= SIGMA_GATE


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["DP.05"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())


# ── pilots (seeds 90-91 only; calibration, never findings) ─────────────────
def _pilot_timing() -> None:
    import time
    seed = PILOT_SEEDS[0]
    w = _mkworld(seed, lethal=True)
    w.legal_spawns()
    rng = np.random.RandomState(1)
    t0 = time.perf_counter()
    for _ in range(100):
        w.decide(rng.uniform(-1, 1, w.action_dim))
    plain = (time.perf_counter() - t0) / 100
    snap = _snap(w)
    t0 = time.perf_counter()
    for _ in range(20):
        _restore(w, snap)
        _score_rollout(w, rng.uniform(-1, 1, w.action_dim), 5, _phi_drive)
    roll = (time.perf_counter() - t0) / 100
    _restore(w, snap)
    print(f"decision: {plain * 1000:.1f} ms plain, "
          f"{roll * 1000:.1f} ms per rollout decision (incl. restore)")
    per_seed = (2 * N_LIVES * LIFE_CAP_DEC * (1 + K_CANDS) * roll          # react
                + N_LIVES * LIFE_CAP_DEC *
                sum(1 + K_CANDS * h / REPLAN[h] for h in H_SWEEP) * roll   # plans
                + N_LIVES * LIFE_CAP_DEC * plain                           # ref
                + CTRL_EPISODES * CTRL_CAP_DEC *
                (2 + K_CANDS * (1 + H_MAX / REPLAN[H_MAX])) * roll)        # ctrl
    print(f"worst-case per-seed estimate: {per_seed / 60:.1f} min "
          f"(x3 seeds = {per_seed / 20:.1f} min)")


def _pilot_ref() -> None:
    for seed in PILOT_SEEDS:
        r = _reference(seed)
        print(f"seed {seed}: ref_span {r['ref_span']:.1f} s "
              f"(ceiling {NO_FOOD_CEILING_S * REF_HEADROOM:.0f}), "
              f"ref_eats {r['ref_eats']:.0f} (need {REF_MIN_EATS})")


def _pilot_fidelity() -> None:
    for seed in PILOT_SEEDS:
        print(f"seed {seed}: {_fidelity(seed)}")


def _pilot_ctrl() -> None:
    for seed in PILOT_SEEDS:
        plan = _run_ctrl_arm(seed, H_MAX)
        react = _run_ctrl_arm(seed, 1)
        broken = _run_ctrl_arm(seed, 1, broken=True)
        print(f"seed {seed}: plan {plan:.4f} react {react:.4f} "
              f"broken {broken:.4f} | gain {plan - react:+.4f} "
              f"(tol {CTRL_TOL}) broken_gap {react - broken:.4f} "
              f"(floor {CTRL_BROKEN_FLOOR})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "timing"
    {"timing": _pilot_timing, "ref": _pilot_ref,
     "fidelity": _pilot_fidelity, "ctrl": _pilot_ctrl}[cmd]()
