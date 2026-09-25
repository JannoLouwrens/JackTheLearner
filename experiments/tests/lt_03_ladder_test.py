"""LT.03 — THE LADDER TEST: curiosity alone climbs the ladder.

THE claim of GOAL.md's opening image: *"If there is a ladder with an apple on
top, he must try to climb the ladder, fall, and learn from falling, purely out
of curiosity."* The environment returns reward IDENTICALLY ZERO. Whatever
climbing happens is paid for by an intrinsic signal alone.

STATUS — IMPLEMENTED 2026-09-25 (16:0x slot), GATES NOT YET PILOT-VALIDATED.
`_PILOT_VALIDATED = False` and `run()` REFUSES until a pilot (seed 90,
disjoint from recorded seeds 0-2) has validated rig mechanics, control
aliveness and sizing, per SM.03's lesson (a registered run against an
unvalidated rig burned its comparison before the nose was ever measured) and
LC.07's idiom. Every CLAIM BAR below is the registry's, frozen 2026-08-09 from
LT.01's measured null floor — the pilot may reshape the ENVELOPE (life counts,
decision counts, learner sizes) and may not touch a bar. STATISTIC_BOUND:
none of the gated statistics sits on a bounded scale whose bound the null
approaches (engaged attempts unbounded above, ascent gain bounded by ladder
height 1.5 m with the random ceiling measured at 0.83 m — 0.67 m of headroom,
priced against the 0.35 m margin per the UNSATURATED-NULL RULE).

WHAT IS PRE-REGISTERED (registry LT.03; thresholds are LT.01's four measured
numbers and do not move):
  CLAIM — at least one CANDIDATE arm, in >= 2 of 3 seeds, produces ALL of:
    engaged ladder attempts        >= 20        (ENGAGED_MIN)
    distance-matched post-fall
      return lift                  >= 2.0       (RETURN_LIFT_MIN)
    ascent gain (last-vs-first
      quintile of engaged H)       >= 0.35 m    (ASCENT_GAIN_MIN)
    Spearman rho(attempt idx, H)   >= 0.35, p < 0.01
    final-quintile mean rise       >= 0.85 m    (QUINTILE_RISE_MIN — above
                                                 LT.01's random ceiling 0.83)
    toppings-out (ladder-supported
      platform arrivals)           >= 1
  ...while that arm's panel dwell <= 0.15 in EVERY seed, and the arm never
  trips the self-generated-chaos check (chaos_occupancy >= 3.0 AND
  chaos_reward_ratio >= 2.0 in any seed makes THAT ARM VOID for the run —
  its curiosity signal degenerated, so it did not test the claim).

  CONTROLS, each on its declared side:
    (1) the ICM control MUST fixate on the panel in THIS rig: panel dwell
        > 0.4 in >= 2 of 3 seeds — proving the trap is live HERE, not only
        in PG.4's slider. An at-chance control must carry proof its
        instrument was alive (LESSONS); this is that proof for the dwell
        instrument, live in the same world the candidates are scored in.
    (2) randrew — a random-STATIONARY-reward learner at matched compute —
        must not match the winner's visitation lift: engaged(randrew)
        <= 0.5 x engaged(winner), seed-mean. Controls for "any optimisation
        pressure explores".
    (3) the goal-shuffled variant of the per-seed winning arm must show NO
        ascent trend: it must fail (gain >= 0.35 AND rho >= 0.35 AND
        p < 0.01). Run as run_spec's control_fn (`_control`) so the ledger
        records it as the control it is; the winner identity is read from
        the per-seed cache `_experiment` writes (same process, run_spec
        calls _experiment before _control for each seed — asserted, not
        assumed: _control raises if the cache is empty).

  SCREENING ONLY — no winner is declared here; arbitration is LT.04.
  run_bakeoff is NOT used, deliberately: it VOIDs on a sub-gate arm, and the
  icm control is REQUIRED to fail (fixate).

THE CANDIDATE ARMS (LT.04's pre-registered set, cost estimates lp 2.0 /
disagree 9.0 / metra 14.0 core-s per 1k decisions to be REPLACED by in-run
measurement before LT.04 runs):
  lp        learning progress: per-floor-cell forward-error EMAs, fast minus
            slow; reward = error DECREASE (progress), the classic
            Oudeyer/Kaplan signal. Immune to irreducible noise by
            construction (noise does not get more predictable).
  disagree  ensemble disagreement: K=4 small forward models, reward =
            predictive variance (Pathak 2019). The panel's noise keeps the
            ensemble split forever — this arm must EARN its dwell <= 0.15.
  metra     metra-xs: a per-life skill z on the unit circle, phi trained so
            (phi(o2)-phi(o)) . z is large under a soft unit-step constraint
            (METRA's Wasserstein objective, xs-simplified the way LC.03's
            dreamer-xs was — named -xs to say so honestly).
CONTROL ARMS: icm (LT.02's forward-error farmer — the naive signal PG.4
proved trappable), randrew, shuffled-winner (reward stream computed on
MISMATCHED transitions — a random earlier (o,a,o2) from the arm's own
buffer — destroying state-contingency while preserving marginal scale).
The RANDOM arm is the ruler (defines chaos occupancy 1.0) and the source of
the distance-matched return null; its zero-engagement floor is LT.01's
measured result, re-read here per seed.

THE STATIC AUDIT (registry: "a match is ERROR"): every arm class's source is
audited at run time for apparatus-referencing symbols (ladder/rung/platform/
attempt/h_max/rise/climb/height). The shared metering loop reads those — the
ARMS may not. A match raises RuntimeError before any physics runs: nothing
anywhere may smuggle the ladder into a reward.

THE EYE: PG.4's retina construction, rebuilt against the rover world by name
(T0.14): 24 mj_ray casts from the torso at panel height, per ray
[distance, texture]; textures deterministic per geom EXCEPT `noise_panel`,
resampled uniformly per observation while within resolve range — the noisy
TV itself, no GL, ~free. obs = 20-dim proprio (LT.02's read) + 48 retina
= 68 dims. The panel's stochasticity reaches every arm's forward model
through the texture channels; that is what makes control (1) satisfiable
and the trap live.

THE LEARNER SKELETON is LT.02's, deliberately: tabular Q over
cell x upright x liftL x liftR x rise-band (5808 states) with 24 uniform
prototype actions + stand — ADHESION LIVE (unlike LT.02's chaos farmer,
which forced it off: climbing requires grip; the null floor with live
adhesion is exactly what LT.01 measured). Epsilon schedule, Q_LR, GAMMA
imported from PG.4 — same constants that produced the known positive.
The learner PERSISTS across lives within an arm-seed; the body respawns.

FALL / RETURN definition (frozen here, before any run): a FALL is a decision
where rise >= ENGAGED_RISE with ladder contact, followed within 10 decisions
by grounded, no ladder contact, rise < 0.1 m. A RETURN is ladder contact
within RETURN_W = 150 decisions (30 s) of the fall. The null return rate is
measured from the RANDOM arm's own log at matched distance (landing distance
+- 0.5 m); both rates are Laplace-smoothed (k+1)/(n+2) so an empty null bin
cannot manufacture an infinite lift. lift = rate_arm / rate_null_matched.

SIZING — PROVISIONAL, the pilot's first job: N_LIVES x LIFE_DEC = 10 x 2500
decisions per arm-seed (500 s of life each), 7 arm-runs per seed. LT.02's
sizing record (4000 decisions ~ 11 s physics) projects ~8 min/seed of
physics + learner overhead; the pilot measures the true figure and the cost
class follows the measurement (LT.02's cpu<2h -> cpu<10min precedent).
Budget.CPU_LONG until measured otherwise.

Creature-gate note: this spec moves none of T2.01/XL.01/T6.01 directly; it
is the constitutional ladder image itself (GOAL.md:29-34), Tier 5.
"""
from __future__ import annotations

import inspect
import math
import re
import sys
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lt_01_null_floor import (_Rig, ENGAGED_RISE, SUCCESS_RUNGS,
                               MUTATE_STRENGTH)
from .lt_02_chaos_detector import _detector, _reward_ratio
from .pg_4_noisy_tv import EPS_HI, EPS_LO, GAMMA, Q_INIT, Q_LR

# The claim is about LT.01's world+instrument, LT.02's detector and PG.4's
# trap constants; all hash into impl_sha.
IMPL_DEPS = ["playground.py",
             "experiments/tests/lt_01_null_floor.py",
             "experiments/tests/lt_02_chaos_detector.py",
             "experiments/tests/pg_4_noisy_tv.py"]

REPO = Path(__file__).resolve().parents[2]

# ── envelope (PROVISIONAL until the pilot freezes it; never a bar) ─────────
N_LIVES = 10                # lives per arm-seed; learner persists across them
LIFE_DEC = 2500             # decisions per life (500 s at 0.2 s/decision)
N_PROTO = 24                # uniform prototype actions (+ stand), adhesion LIVE
N_RAYS = 24                 # the eye: PG.4's ray count
MAX_RANGE = 12.0            # m, ray range (arena diameter scale)
R_RESOLVE = 4.0             # m, panel amplitude falloff (PG.4's semantics)
RAY_Z = 1.2                 # panel centre height (playground.py:494)
DWELL_RADIUS = 2.0          # m, LT.01/LT.02's dwell zone
RETURN_W = 150              # decisions (30 s): post-fall return window
FALL_SETTLE = 10            # decisions: contact->ground transition window
DIST_TOL = 0.5              # m, distance-matching band for the return null
LP_FAST, LP_SLOW = 0.05, 0.005   # learning-progress EMA rates
K_ENSEMBLE = 4              # disagree heads
METRA_LAM = 10.0            # metra-xs constraint weight
FWD_HID = 64                # forward-model width (LT.02's)
BATCH = 64                  # per-decision replay batch (LT.02's)

# ── pre-registered bars (registry LT.03, 2026-08-09 — restated, not owned) ─
ENGAGED_MIN = 20
RETURN_LIFT_MIN = 2.0
ASCENT_GAIN_MIN = 0.35      # m
RHO_MIN = 0.35
RHO_P_MAX = 0.01
QUINTILE_RISE_MIN = 0.85    # m — above LT.01's measured random ceiling 0.83
TOPOUT_MIN = 1
DWELL_MAX = 0.15
CONTROL_DWELL_MIN = 0.4     # icm must fixate here
RANDREW_FRAC_MAX = 0.5      # randrew engaged <= this x winner engaged
CHAOS_OCC = 3.0             # from LT.02 (VOID lane, per arm)
CHAOS_RATIO = 2.0

CANDIDATES = ("lp", "disagree", "metra")

# Flipped only by a commit that pastes the pilot's JSON into this docstring.
_PILOT_VALIDATED = False


# ═════════════════════════════ static audit ════════════════════════════════
_BANNED = re.compile(r"ladder|rung|platform|attempt|h_max|rise|climb|height",
                     re.IGNORECASE)


def _audit_arm_sources():
    """Registry: 'Every arm's reward code passes a static audit for
    ladder-referencing symbols; a match is ERROR.' Audited: the arm classes,
    whole source, comments included."""
    for cls in (_ICM, _LP, _Disagree, _MetraXS, _RandRew, _Shuffled):
        src = inspect.getsource(cls)
        m = _BANNED.search(src)
        if m:
            raise RuntimeError(
                f"LT.03 static audit: arm {cls.__name__} references the "
                f"apparatus ('{m.group(0)}'); refusing to run.")


# ═══════════════════════════════ the eye ═══════════════════════════════════
class _Eye:
    """PG.4's retina against the rover world: 24 rays from the torso at
    panel height, per ray [distance, texture]. Textures deterministic per
    geom except the noise panel, resampled per call — the noisy TV."""

    def __init__(self, rig: _Rig, seed: int):
        import mujoco
        self.mujoco = mujoco
        self.rig = rig
        self.rng = np.random.RandomState(70_000 + seed)
        gid = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_GEOM,
                                "noise_panel")
        if gid < 0:
            raise RuntimeError("LT.03: world has no noise_panel geom — the "
                               "trap-liveness control cannot run.")
        self.panel = int(gid)
        self.exclude = rig.rover_bid
        self.dirs = np.array(
            [[math.cos(2 * math.pi * k / N_RAYS),
              math.sin(2 * math.pi * k / N_RAYS), 0.0] for k in range(N_RAYS)])
        self._geomid = np.zeros(1, dtype=np.int32)
        self.tex = ((np.arange(rig.model.ngeom) * 0.37) % 0.8) + 0.1

    def observe(self) -> np.ndarray:
        d = self.rig.data
        x, y = d.geom_xpos[self.rig.torso_gid][:2]
        pnt = np.array([x, y, RAY_Z])
        out = np.empty(2 * N_RAYS, dtype=np.float32)
        for k in range(N_RAYS):
            dist = self.mujoco.mj_ray(self.rig.model, d, pnt, self.dirs[k],
                                      None, 1, self.exclude, self._geomid)
            gid = int(self._geomid[0])
            if dist < 0 or dist > MAX_RANGE:
                dd, t = 1.0, 0.0
            elif gid == self.panel:
                dd = dist / MAX_RANGE
                amp = min(1.0, R_RESOLVE / max(dist, 1e-6))
                t = 0.5 + amp * (float(self.rng.uniform()) - 0.5)
            else:
                dd, t = dist / MAX_RANGE, float(self.tex[gid])
            out[2 * k] = dd
            out[2 * k + 1] = t
        return out


class _LadderRig(_Rig):
    """LT.01's instrumented world + LT.02's proprioceptive read + the eye."""

    JOINTS = ("reachL", "liftL", "reachR", "liftR")

    def __init__(self, seed: int):
        super().__init__(seed)
        assert self.params.noise_panel, "LT.03 requires the panel present"
        self.jq = [self.ix["jnt_qposadr"][n] for n in self.JOINTS]
        self.jd = [self.ix["jnt_dofadr"][n] for n in self.JOINTS]
        self.jlo = np.array([self.model.jnt_range[self.ix["jnt"][n]][0]
                             for n in self.JOINTS])
        self.jhi = np.array([self.model.jnt_range[self.ix["jnt"][n]][1]
                             for n in self.JOINTS])
        self.eye = _Eye(self, seed)
        self.obs_dim = 20 + 2 * N_RAYS

    def upright(self) -> bool:
        return float(self.data.geom_xpos[self.torso_gid][2]) > 0.7 * self.z_rest

    def obs(self) -> np.ndarray:
        a = self.params.arena_size
        qadr = self.ix["root_qposadr"]
        pos = self.data.qpos[qadr:qadr + 3]
        R = self.data.xmat[self.rover_bid].reshape(3, 3)
        up = R[:, 2]
        dof = self.ix["root_dofadr"]
        lin = self.data.qvel[dof:dof + 3]
        ang = self.data.qvel[dof + 3:dof + 6]
        jq = (2.0 * (self.data.qpos[self.jq] - self.jlo)
              / (self.jhi - self.jlo) - 1.0)
        jv = self.data.qvel[self.jd]
        proprio = np.concatenate([
            [pos[0] / a, pos[1] / a, pos[2]], up,
            np.clip(lin / 3.0, -3, 3), np.clip(ang / 10.0, -3, 3),
            jq, np.clip(jv / 3.0, -3, 3)]).astype(np.float32)
        return np.concatenate([proprio, self.eye.observe()])


def _cell(rig: _LadderRig) -> int:
    a = rig.params.arena_size
    q = rig.ix["root_qposadr"]
    x, y = rig.data.qpos[q], rig.data.qpos[q + 1]
    cx = min(10, max(0, int((x + a) / (2 * a) * 11)))
    cy = min(10, max(0, int((y + a) / (2 * a) * 11)))
    return cy * 11 + cx


def _band(rig: _LadderRig) -> int:
    r = float(rig.data.geom_xpos[rig.torso_gid][2]) - rig.z_rest
    return 0 if r < ENGAGED_RISE else (1 if r < QUINTILE_RISE_MIN else 2)


def _state(rig: _LadderRig) -> int:
    liftL = int(rig.data.qpos[rig.jq[1]] > 0.5 * (rig.jlo[1] + rig.jhi[1]))
    liftR = int(rig.data.qpos[rig.jq[3]] > 0.5 * (rig.jlo[3] + rig.jhi[3]))
    s = ((_cell(rig) * 2 + int(rig.upright())) * 2 + liftL) * 2 + liftR
    return s * 3 + _band(rig)


N_STATES = 121 * 2 * 2 * 2 * 3  # 5808


# ═══════════════════════════════ the arms ══════════════════════════════════
# Arm contract: reward(o, a, o2, cell) -> float; called once per decision,
# may train internal models. NOTHING in these classes may reference the
# apparatus (see _audit_arm_sources; the banned list includes the obvious
# words — these docstrings stay clean by construction).

def _mlp(inp: int, out: int):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(inp, FWD_HID), torch.nn.ReLU(),
        torch.nn.Linear(FWD_HID, FWD_HID), torch.nn.ReLU(),
        torch.nn.Linear(FWD_HID, out))


class _ICM:
    """Forward-model error, the naive signal. CONTROL: PG.4 proved it
    trappable; here it must fixate on the panel or the trap is not live."""

    def __init__(self, obs_dim: int, seed: int):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.fwd = _mlp(obs_dim + 8, obs_dim)
        self.opt = torch.optim.Adam(self.fwd.parameters(), lr=1e-3)
        self.bx, self.by = [], []
        self.rng = np.random.RandomState(41_000 + seed)

    def reward(self, o, a, o2, cell) -> float:
        t = self.torch
        with t.no_grad():
            inp = t.cat([t.from_numpy(o), t.from_numpy(a)])
            r = float(((self.fwd(inp) - t.from_numpy(o2)) ** 2).sum())
        self.bx.append(inp.numpy())
        self.by.append(o2)
        idx = self.rng.randint(len(self.bx), size=min(BATCH, len(self.bx)))
        bx = t.from_numpy(np.stack([self.bx[i] for i in idx]))
        by = t.from_numpy(np.stack([self.by[i] for i in idx]))
        loss = ((self.fwd(bx) - by) ** 2).sum(dim=1).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return r


class _LP:
    """Learning progress: per-cell forward-error EMAs, slow minus fast.
    Progress is error DECREASE; irreducible noise yields none."""

    def __init__(self, obs_dim: int, seed: int):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.fwd = _mlp(obs_dim + 8, obs_dim)
        self.opt = torch.optim.Adam(self.fwd.parameters(), lr=1e-3)
        self.bx, self.by = [], []
        self.fast = np.zeros(121)
        self.slow = np.zeros(121)
        self.seen = np.zeros(121, dtype=bool)
        self.rng = np.random.RandomState(42_000 + seed)

    def reward(self, o, a, o2, cell) -> float:
        t = self.torch
        with t.no_grad():
            inp = t.cat([t.from_numpy(o), t.from_numpy(a)])
            e = float(((self.fwd(inp) - t.from_numpy(o2)) ** 2).sum())
        if not self.seen[cell]:
            self.fast[cell] = self.slow[cell] = e
            self.seen[cell] = True
        self.fast[cell] += LP_FAST * (e - self.fast[cell])
        self.slow[cell] += LP_SLOW * (e - self.slow[cell])
        r = max(0.0, float(self.slow[cell] - self.fast[cell]))
        self.bx.append(inp.numpy())
        self.by.append(o2)
        idx = self.rng.randint(len(self.bx), size=min(BATCH, len(self.bx)))
        bx = t.from_numpy(np.stack([self.bx[i] for i in idx]))
        by = t.from_numpy(np.stack([self.by[i] for i in idx]))
        loss = ((self.fwd(bx) - by) ** 2).sum(dim=1).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return r


class _Disagree:
    """Ensemble disagreement: K forward models, reward = predictive
    variance. Aleatoric noise keeps the ensemble split — this arm must earn
    its panel indifference or be caught by the dwell gate."""

    def __init__(self, obs_dim: int, seed: int):
        import torch
        self.torch = torch
        self.heads, self.opts = [], []
        for k in range(K_ENSEMBLE):
            torch.manual_seed(seed * 97 + k)
            h = _mlp(obs_dim + 8, obs_dim)
            self.heads.append(h)
            self.opts.append(torch.optim.Adam(h.parameters(), lr=1e-3))
        self.bx, self.by = [], []
        self.rng = np.random.RandomState(43_000 + seed)

    def reward(self, o, a, o2, cell) -> float:
        t = self.torch
        with t.no_grad():
            inp = t.cat([t.from_numpy(o), t.from_numpy(a)])
            preds = t.stack([h(inp) for h in self.heads])
            r = float(preds.var(dim=0).mean())
        self.bx.append(inp.numpy())
        self.by.append(o2)
        for k, (h, opt) in enumerate(zip(self.heads, self.opts)):
            idx = self.rng.randint(len(self.bx),
                                   size=min(BATCH, len(self.bx)))
            bx = t.from_numpy(np.stack([self.bx[i] for i in idx]))
            by = t.from_numpy(np.stack([self.by[i] for i in idx]))
            loss = ((h(bx) - by) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return r


class _MetraXS:
    """metra-xs: per-life unit skill z, phi trained to maximise
    (phi(o2)-phi(o)).z under a soft unit-step constraint. -xs says the
    simplification out loud (LC.03's dreamer-xs precedent)."""

    def __init__(self, obs_dim: int, seed: int):
        import torch
        torch.manual_seed(seed + 7)
        self.torch = torch
        self.phi = _mlp(obs_dim, 2)
        self.opt = torch.optim.Adam(self.phi.parameters(), lr=1e-3)
        self.rng = np.random.RandomState(44_000 + seed)
        self.z = None
        self.new_life()

    def new_life(self):
        th = self.rng.uniform(0, 2 * math.pi)
        self.z = np.array([math.cos(th), math.sin(th)], dtype=np.float32)

    def reward(self, o, a, o2, cell) -> float:
        t = self.torch
        z = t.from_numpy(self.z)
        to, to2 = t.from_numpy(o), t.from_numpy(o2)
        with t.no_grad():
            d = self.phi(to2) - self.phi(to)
            r = float((d * z).sum())
        d = self.phi(to2) - self.phi(to)
        obj = (d * z).sum()
        pen = t.clamp((d ** 2).sum() - 1.0, min=0.0)
        loss = -(obj - METRA_LAM * pen)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return r


class _RandRew:
    """Stationary random reward at matched compute: a fixed seeded table
    over (state-index mod table, action), plus a dummy forward-model update
    per decision so its optimisation pressure and compute match the
    candidates'. Controls for 'any optimisation pressure explores'."""

    TABLE = 5808

    def __init__(self, obs_dim: int, seed: int):
        import torch
        torch.manual_seed(seed + 13)
        self.torch = torch
        self.fwd = _mlp(obs_dim + 8, obs_dim)
        self.opt = torch.optim.Adam(self.fwd.parameters(), lr=1e-3)
        self.bx, self.by = [], []
        tab_rng = np.random.RandomState(45_000 + seed)
        self.table = tab_rng.uniform(0.0, 1.0, size=(self.TABLE, N_PROTO + 1))
        self.rng = np.random.RandomState(46_000 + seed)
        self._last_sa = (0, 0)

    def note_sa(self, s: int, ai: int):
        self._last_sa = (s % self.TABLE, ai)

    def reward(self, o, a, o2, cell) -> float:
        t = self.torch
        inp = t.cat([t.from_numpy(o), t.from_numpy(a)])
        self.bx.append(inp.numpy())
        self.by.append(o2)
        idx = self.rng.randint(len(self.bx), size=min(BATCH, len(self.bx)))
        bx = t.from_numpy(np.stack([self.bx[i] for i in idx]))
        by = t.from_numpy(np.stack([self.by[i] for i in idx]))
        loss = ((self.fwd(bx) - by) ** 2).sum(dim=1).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        s, ai = self._last_sa
        return float(self.table[s, ai])


class _Shuffled:
    """Goal-shuffle control: the inner arm's reward computed on a MISMATCHED
    transition — a random earlier (o, a, o2) from this run's own buffer —
    destroying state-contingency, preserving marginal scale."""

    def __init__(self, inner, seed: int):
        self.inner = inner
        self.buf = []
        self.rng = np.random.RandomState(47_000 + seed)

    def new_life(self):
        if hasattr(self.inner, "new_life"):
            self.inner.new_life()

    def reward(self, o, a, o2, cell) -> float:
        self.buf.append((o, a, o2, cell))
        j = int(self.rng.randint(len(self.buf)))
        oo, aa, oo2, cc = self.buf[j]
        return self.inner.reward(oo, aa, oo2, cc)


def _make_arm(name: str, obs_dim: int, seed: int):
    return {"icm": _ICM, "lp": _LP, "disagree": _Disagree,
            "metra": _MetraXS, "randrew": _RandRew}[name](obs_dim, seed)


# ═════════════════════════════ the shared life ═════════════════════════════
def _run_arm(rig: _LadderRig, name: str, seed: int, arm=None,
             n_lives: int = None, life_dec: int = None) -> dict:
    """N_LIVES lives with a persistent learner (or the random null when
    name == 'random'). Q-learning skeleton is LT.02's; adhesion live.

    Logs per decision: dist-to-base, ladder-contact, grounded, rise, panel
    proximity — plus (obs, act, next) transitions for the chaos detector."""
    n_lives = N_LIVES if n_lives is None else n_lives
    life_dec = LIFE_DEC if life_dec is None else life_dec
    rng = np.random.RandomState(30_000 + seed if name != "random"
                                else 10_000 + seed)
    protos = rng.uniform(-1.0, 1.0, size=(N_PROTO, 8))
    stand = np.zeros(8); stand[4:6] = -1.0
    protos = np.vstack([stand, protos]).astype(np.float32)

    if arm is None and name != "random":
        arm = _make_arm(name, rig.obs_dim, seed)
    q = np.full((N_STATES, len(protos)), Q_INIT)

    X, Y, R = [], [], []
    dist_log, contact_log, ground_log, rise_log = [], [], [], []
    attempts_all = []          # ordered [H, n_rungs] across lives
    dwell_dec = 0
    platform_hits = platform_ladder = 0
    n_total = n_lives * life_dec
    t_global = 0

    for life in range(n_lives):
        if hasattr(arm, "new_life"):
            arm.new_life()
        sx, sy, _ = rig.params.spawn()
        rig.teleport(sx, sy)
        rig.reset_meter()
        for t in range(life_dec):
            o = rig.obs()
            s = _state(rig)
            if name == "random":
                ai = int(rng.randint(len(protos)))
            else:
                eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO)
                          * t_global / max(1, n_total // 3))
                if rng.uniform() < eps:
                    ai = int(rng.randint(len(protos)))
                else:
                    best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                    ai = int(best[rng.randint(len(best))])
            a = protos[ai]
            rig.decide(a)
            o2 = rig.obs()

            if name != "random":
                if isinstance(arm, _RandRew):
                    arm.note_sa(s, ai)
                r = arm.reward(o, a, o2, _cell(rig))
                s2 = _state(rig)
                q[s, ai] += Q_LR * (r + GAMMA * q[s2].max() - q[s, ai])
                R.append(r)

            X.append(np.concatenate([o, a]))
            Y.append(o2)
            qadr = rig.ix["root_qposadr"]
            xy = rig.data.qpos[qadr:qadr + 2]
            dist_log.append(float(np.hypot(xy[0] - rig.ladder_base[0],
                                           xy[1] - rig.ladder_base[1])))
            pairs = rig._pairs()
            contact_log.append(int(rig._any_pair(pairs, rig.climb,
                                                 rig.ladder)))
            ground_log.append(int(rig.grounded(pairs)))
            rise_log.append(float(rig.data.geom_xpos[rig.torso_gid][2])
                            - rig.z_rest)
            if np.hypot(xy[0] - rig.panel_xy[0],
                        xy[1] - rig.panel_xy[1]) < DWELL_RADIUS:
                dwell_dec += 1
            t_global += 1
        rig.finalize()
        attempts_all.extend(rig.attempts)
        platform_hits += rig.platform_hits
        platform_ladder += rig.platform_hits - rig.platform_nonladder

    return {
        "X": np.stack(X).astype(np.float32), "Y": np.stack(Y),
        "rewards": np.array(R) if R else None,
        "attempts": attempts_all,
        "dist": np.array(dist_log), "contact": np.array(contact_log),
        "ground": np.array(ground_log), "rise": np.array(rise_log),
        "panel_dwell": round(dwell_dec / max(1, n_total), 4),
        "topout": int(platform_ladder),
        "finite": float(rig.finite()),
    }


# ═════════════════════════════ the statistics ══════════════════════════════
def _falls_and_returns(log: dict) -> list:
    """[(fall_decision_idx, landing_dist, returned)] per the frozen
    definition in the docstring."""
    contact, ground, rise, dist = (log["contact"], log["ground"],
                                   log["rise"], log["dist"])
    n = len(contact)
    out = []
    t = 0
    while t < n - FALL_SETTLE:
        if contact[t] and rise[t] >= ENGAGED_RISE:
            w = slice(t + 1, min(n, t + 1 + FALL_SETTLE))
            landed = np.flatnonzero((ground[w] == 1) & (contact[w] == 0)
                                    & (rise[w] < 0.1))
            if landed.size:
                j = t + 1 + int(landed[0])
                ret_w = slice(j, min(n, j + RETURN_W))
                returned = bool(contact[ret_w].any())
                out.append((j, float(dist[j]), returned))
                t = j + 1
                continue
        t += 1
    return out


def _matched_null_rate(null_log: dict, landing_dists: list) -> float:
    """Random-arm P(ladder contact within RETURN_W | at matched distance,
    not in contact), Laplace-smoothed, averaged over the arm's fall
    distances."""
    contact, dist = null_log["contact"], null_log["dist"]
    n = len(contact)
    future = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(contact)
    ptr = 0
    for t in range(n):
        while ptr < len(idx) and idx[ptr] < t:
            ptr += 1
        future[t] = ptr < len(idx) and idx[ptr] < t + RETURN_W
    rates = []
    for d in landing_dists:
        m = (np.abs(dist - d) <= DIST_TOL) & (contact == 0)
        k, nn = int(future[m].sum()), int(m.sum())
        rates.append((k + 1) / (nn + 2))
    return float(np.mean(rates)) if rates else 0.5


def _ascent_stats(attempts: list) -> dict:
    """Engaged attempts in order -> gain, Spearman rho/p, final-quintile
    mean H, engaged count."""
    from scipy import stats
    H = [h for h, _ in attempts if h >= ENGAGED_RISE]
    n = len(H)
    out = {"engaged": n, "gain": 0.0, "rho": 0.0, "rho_p": 1.0,
           "final_q": 0.0}
    if n >= 5:
        k = max(1, n // 5)
        out["gain"] = float(np.mean(H[-k:]) - np.mean(H[:k]))
        out["final_q"] = float(np.mean(H[-k:]))
        rho, p = stats.spearmanr(np.arange(n), H)
        out["rho"] = float(rho) if np.isfinite(rho) else 0.0
        out["rho_p"] = float(p) if np.isfinite(p) else 1.0
    elif n > 0:
        out["final_q"] = float(np.mean(H))
    return out


def _arm_claim(stats_: dict, lift: float, topout: int, dwell: float) -> int:
    return int(stats_["engaged"] >= ENGAGED_MIN
               and lift >= RETURN_LIFT_MIN
               and stats_["gain"] >= ASCENT_GAIN_MIN
               and stats_["rho"] >= RHO_MIN and stats_["rho_p"] < RHO_P_MAX
               and stats_["final_q"] >= QUINTILE_RISE_MIN
               and topout >= TOPOUT_MIN
               and dwell <= DWELL_MAX)


# ═══════════════════ per-seed winner cache (for _control) ══════════════════
_WINNER: dict = {}


def _score_arm(log: dict, null_log: dict) -> dict:
    st = _ascent_stats(log["attempts"])
    fr = _falls_and_returns(log)
    if fr:
        k = sum(1 for _, _, r in fr if r)
        rate = (k + 1) / (len(fr) + 2)
        null_rate = _matched_null_rate(null_log, [d for _, d, _ in fr])
        lift = rate / max(null_rate, 1e-9)
    else:
        lift = 0.0
    st["return_lift"] = round(float(lift), 4)
    st["n_falls"] = len(fr)
    return st


def _experiment(seed: int) -> dict:
    _audit_arm_sources()
    rig = _LadderRig(seed)
    if not rig.calib["ok"]:
        return {"rig_ok": 0.0, "finite": 0.0}

    logs = {"random": _run_arm(rig, "random", seed)}
    for name in ("icm",) + CANDIDATES + ("randrew",):
        logs[name] = _run_arm(rig, name, seed)

    stats_ = {n: _score_arm(logs[n], logs["random"]) for n in logs
              if n != "random"}

    # Per-seed best candidate (screening order: engaged, then gain) — the
    # shuffled twin _control needs it.
    best = max(CANDIDATES,
               key=lambda n: (stats_[n]["engaged"], stats_[n]["gain"]))
    _WINNER[seed] = best

    det = _detector({n: logs[n] for n in logs}, ruler="random", seed=seed)
    out = {"rig_ok": 1.0, "finite": min(l["finite"] for l in logs.values()),
           "ruler_occ": det["per_arm"]["random"]["occupancy"],
           "null_engaged": float(_ascent_stats(
               logs["random"]["attempts"])["engaged"])}

    winner_engaged = stats_[best]["engaged"]
    for n in ("icm",) + CANDIDATES + ("randrew",):
        s = stats_[n]
        occ = det["per_arm"][n]["occupancy"]
        ratio = _reward_ratio(logs[n], det["flags"][n], det["sub_idx"][n])
        chaos = int(occ >= CHAOS_OCC and ratio >= CHAOS_RATIO)
        out[f"{n}_engaged"] = float(s["engaged"])
        out[f"{n}_gain"] = round(s["gain"], 4)
        out[f"{n}_rho"] = round(s["rho"], 4)
        out[f"{n}_rho_p"] = round(s["rho_p"], 6)
        out[f"{n}_final_q"] = round(s["final_q"], 4)
        out[f"{n}_return_lift"] = s["return_lift"]
        out[f"{n}_n_falls"] = float(s["n_falls"])
        out[f"{n}_topout"] = float(logs[n]["topout"])
        out[f"{n}_dwell"] = logs[n]["panel_dwell"]
        out[f"{n}_dwell_ok"] = float(logs[n]["panel_dwell"] <= DWELL_MAX)
        out[f"{n}_chaos_occ"] = occ
        out[f"{n}_chaos_ratio"] = ratio
        out[f"{n}_chaos_void"] = float(chaos)
        if n in CANDIDATES:
            out[f"{n}_claim"] = float(
                _arm_claim(s, s["return_lift"], logs[n]["topout"],
                           logs[n]["panel_dwell"]) and not chaos)
    out["icm_fixates"] = float(logs["icm"]["panel_dwell"]
                               > CONTROL_DWELL_MIN)
    out["randrew_below"] = float(
        out["randrew_engaged"] <= RANDREW_FRAC_MAX * max(1.0, winner_engaged))
    out["winner_engaged"] = float(winner_engaged)
    return out


def _control(seed: int) -> dict:
    """Control (3): the per-seed winning candidate, goal-shuffled. MUST show
    no ascent trend."""
    if seed not in _WINNER:
        raise RuntimeError("LT.03 _control ran before _experiment for seed "
                           f"{seed}; the shuffled twin has no winner to copy.")
    rig = _LadderRig(seed)
    name = _WINNER[seed]
    inner = _make_arm(name, rig.obs_dim, seed)
    arm = _Shuffled(inner, seed)
    log = _run_arm(rig, name, seed, arm=arm)
    null_log = _run_arm(rig, "random", seed)
    s = _score_arm(log, null_log)
    return {"shuffled_arm_idx": float(CANDIDATES.index(name)),
            "shuffled_engaged": float(s["engaged"]),
            "shuffled_gain": round(s["gain"], 4),
            "shuffled_rho": round(s["rho"], 4),
            "shuffled_rho_p": round(s["rho_p"], 6),
            "shuffled_trend": float(s["gain"] >= ASCENT_GAIN_MIN
                                    and s["rho"] >= RHO_MIN
                                    and s["rho_p"] < RHO_P_MAX)}


def _void(m: dict, reason: str):
    return Status.VOID, f"run did not test the claim; not a refutation — {reason}"


def _check(m: dict, c: dict):
    """Pre-registered. m: seed-MEAN metrics (per-seed 0/1 flags arrive as
    fractions of 3 seeds); c: control means."""
    if m.get("rig_ok", 0.0) < 1.0 or m.get("finite", 0.0) < 1.0:
        return _void(m, "rig unhealthy (calibration or non-finite state)")
    if abs(m.get("ruler_occ", 1.0) - 1.0) > 0.10:
        return _void(m, "detector ruler broken: random occupancy far from 1.0")
    if m.get("null_engaged", 99.0) >= ENGAGED_MIN:
        return _void(m, "the null itself reaches the claim's engagement bar "
                        "— the statistic does not discriminate in this rig")

    # Control (1): the trap must be live (>= 2 of 3 seeds -> mean >= 0.66).
    if m.get("icm_fixates", 0.0) < 0.66:
        return _void(m, "the ICM control did not fixate — the panel trap is "
                        "not live in this rig, so dwell <= 0.15 is untested")

    # VOID lane: an arm whose chaos check fired in ANY seed is VOID for the
    # claim; if EVERY candidate is chaos-VOID the run tested nothing.
    live = [n for n in CANDIDATES if m.get(f"{n}_chaos_void", 1.0) == 0.0]
    if not live:
        return _void(m, "every candidate arm tripped the self-generated-"
                        "chaos check; no curiosity signal survived")

    # THE CLAIM: some live candidate clears every conjunct in >= 2/3 seeds.
    claimant = None
    for n in live:
        if (m.get(f"{n}_claim", 0.0) >= 0.66
                and m.get(f"{n}_dwell_ok", 0.0) >= 1.0):
            claimant = n
            break
    if claimant is None:
        return False, ("no candidate arm climbed: engaged/trend/return/"
                       "quintile/topout conjuncts unmet in >=2 of 3 seeds "
                       "(the pivot branch: GOAL.md's ladder image needs a "
                       "goal/skill layer)")

    # Control (2): randrew must not match the winner's visitation.
    if m.get("randrew_below", 0.0) < 1.0:
        return False, ("randrew matched the winner's visitation lift — "
                       "optimisation pressure alone explains the exploration")

    # Control (3): the shuffled twin must show no ascent trend.
    if c.get("shuffled_trend", 1.0) > 0.0:
        return False, ("the goal-shuffled twin shows an ascent trend — the "
                       "trend is not the arm's state-contingent signal")

    return True, (f"{claimant} climbed: engaged "
                  f"{m.get(claimant + '_engaged'):.0f}, gain "
                  f"{m.get(claimant + '_gain'):.2f} m, final-quintile "
                  f"{m.get(claimant + '_final_q'):.2f} m, return lift "
                  f"{m.get(claimant + '_return_lift'):.2f}, dwell "
                  f"{m.get(claimant + '_dwell'):.3f} — curiosity alone, "
                  "reward identically zero")


def run(ledger: Ledger | None = None):
    import os
    if not _PILOT_VALIDATED:
        raise RuntimeError(
            "LT.03: rig not pilot-validated. Run "
            "`python -m experiments.tests.lt_03_ladder_test` (seed 90, "
            "reduced envelope, records nothing), paste the JSON into the "
            "docstring's PILOT RECORD, flip _PILOT_VALIDATED in the same "
            "commit. SM.03's lesson: no registered run on an unvalidated "
            "rig.")
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    return run_spec(BY_ID["LT.03"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


# ═══════════════════════════════ pilot ═════════════════════════════════════
def _pilot():
    """Seed 90 (disjoint from recorded 0-2), reduced envelope: mechanics,
    gate aliveness, runtime. Prints JSON, records NOTHING."""
    import json
    import time as _time

    _audit_arm_sources()
    out = {}
    t0 = _time.time()
    rig = _LadderRig(90)
    out["build_s"] = round(_time.time() - t0, 1)
    out["z_rest"] = round(rig.z_rest, 4)
    out["calib_ok"] = rig.calib["ok"]
    out["obs_dim"] = rig.obs_dim

    n_lives, life_dec = 2, 400
    logs = {}
    for name in ("random", "icm", "lp"):
        t0 = _time.time()
        logs[name] = _run_arm(rig, name, 90, n_lives=n_lives,
                              life_dec=life_dec)
        out[f"t_{name}_s"] = round(_time.time() - t0, 1)
    t0 = _time.time()
    det = _detector(logs, ruler="random", seed=90)
    out["t_det_s"] = round(_time.time() - t0, 1)

    for name in logs:
        s = _ascent_stats(logs[name]["attempts"])
        fr = _falls_and_returns(logs[name])
        out[name] = {
            "engaged": s["engaged"], "gain": round(s["gain"], 3),
            "final_q": round(s["final_q"], 3), "n_falls": len(fr),
            "dwell": logs[name]["panel_dwell"],
            "topout": logs[name]["topout"],
            "occ": det["per_arm"][name]["occupancy"],
            "finite": logs[name]["finite"],
        }
    proj = (out["t_random_s"] + out["t_icm_s"] + 5 * out["t_lp_s"]) \
        * (N_LIVES * LIFE_DEC) / (n_lives * life_dec)
    out["projected_full_seed_s"] = round(proj, 0)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _pilot()
