"""NE.02 — every need earns its place (the need x ablation matrix, standing).

First registration (builder, 2026-10-17). The spec has NEVER run: every bar
below is pre-registered here, before any arm's number exists. NE.01 is FAIL
(ok_night_shelter, a world finding routed as `ne01-occlusion-knife-edge`,
DECLINED and owner-held under D43), so this implementation lands BLOCKED —
`run()` refuses on the dependency and that is the honest state. It exists so
that NE.08's chain (XL.01's strengthened successor) no longer carries an
"unimplemented root with no file at all" (the 2026-10-04 xl01 ruling's leg (a)).

## The ablation, mechanically (registry: "delta clamped to 0, observation
## channels removed, death condition removed, remaining lambdas RESCALED")

Ablating need k is done in THIS rig's wrapper, never by editing needs.py
(LT.02's precedent: the spec's own rig overrides, the shared module stays on
its certified bytes):

  1. STATE CLAMP: after every `decide()`, state_k := setpoint_k. This kills
     every body coupling the need carries (gear_scale's (1-f)*min(e,i),
     shiver, the sweat multiplier) and makes its death condition unreachable
     (the death clocks need the state held at a bound for a sustained window;
     a per-decision re-clamp caps drift at ~1e-4 per 0.2 s).
  2. DELTA ZERO: the wrapper's d(h) and obs read delta_k = 0 exactly, never
     the ~1e-4 within-decision drift.
  3. OBS CHANNEL ZEROED, width constant (UB.11's matched-width idiom): the
     observation is always NEED_DIM_02 = 11 floats
     [e, w, p, T_norm, f, c, i, x, d(h), pain, gear] with the ablated
     channel at 0.0, so arms are comparable to a future learner.
  4. LAMBDA RESCALE: remaining lambdas scale by S/(S - lambda_k) so
     max_h d(h) = sqrt(S) is BYTE-EXACT unchanged (asserted to 1e-9; the
     registry's second control). S = sum of the 8-need suite's lambdas.
  5. RESURRECT GUARD, defensive: a death whose recorded cause belongs to the
     ablated need is undone and COUNTED (`resurrections` — expected 0 by
     construction; nonzero is reported, and the drowning clock is reset
     because drowning routes through i).

FATIGUE gets the kills-field's own extra step, verbatim: clamp f=0 AND
rescale kappa_act so total energy drain over a life is matched — kappa' =
kappa * (P_baseline / P_fclamped), both measured on a per-seed calibration
random life pair (CAL_CAP decisions each) before any arm is scored. Without
it the f column would read the energy economy, not the information.

## The placebo need (the registry's own null, UB.11 transposed)

An eighth variable x: lambda_x = 1.0 (the MEDIAN of the real lambdas
{1,1,0.5,1,0.3,0.3,1}, the same convention as its autocorrelation matching),
one observation channel like every real need, delta dynamics = an OU process
whose (tau, mu, sigma) are fitted on the calibration life to the median-tau
real need's delta trajectory, clipped to [0,1], driven by a DEDICATED RNG
that shares no stream with physics. x feeds ONLY obs and d(h).

EXPECTED, stated before any number exists: in THIS venue (scripted policies,
nothing reads obs) the placebo-ablation column is ~0 — the rollout is
deterministic given config and x touches no physics. The placebo column's
job here is therefore to DETECT a leaky ablation procedure (any nonzero
reading means the wrapper touched physics or an RNG stream) and to carry the
reverse gate. The claim bar is floored at R_MIN so a ~0 placebo cannot make
the claim trivially cheap. STATISTIC_BOUND: none (relative effects are
unbounded above); the null (placebo column) sits at ~0, the far side of the
bar from saturation — the unsaturated-null rule is satisfied by construction.

## The three measures and the columns

Per arm, per seed: median RANDOM-LIFE lifespan (N_RANDOM lives, cap
RANDOM_CAP — T-ablated lives right-censor at the cap, which UNDERSTATES that
column: the bias direction is against the claim, declared); the COMPETENCE
BATTERY (the NE.01 forager at 1 sim-day: meals, drinks, sleep decisions,
survived decisions); and each need's OWN SIGNATURE:

  e  apple bursts initiated          (the script's eating cadence)
  w  pool visits                     (the script's drinking cadence)
  T  thermal share of random deaths  (hyper+hypothermia / lives)
  f  within-burst pacing ratio       (mean power last/first 10 decisions of
                                      each apple burst — the kills field's
                                      "within-bout pacing structure")
  i  min gear_scale over a 3-drop probe (damage -> weakness is the
                                      behavioural face of integrity)

Every statistic is reduced to a RELATIVE effect
r = |stat(arm) - stat(base)| / max(|stat(base)|, FLOOR_stat) with floors
declared below. A need's COLUMN is the max over {lifespan r, battery max r,
own-signature r}; the PLACEBO COLUMN is the max over ALL those statistics'
r. Aggregation across seeds is by `protocol._aggregate` means (3 seeds).

## Pre-registered gates

  CLAIM     every EXERCISED need's column >= BAR, where
            BAR = max(PLACEBO_MARGIN * placebo_col, R_MIN),
            PLACEBO_MARGIN = 3.0, R_MIN = 0.15. Metric:
            min_need_margin_over_placebo = min_k column_k / BAR, gated >= 1.
  REVERSE   placebo_col <= PLACEBO_MAX = 0.20, else Status.VOID — the
            registry's own words: a large placebo delta means off-manifold
            shock and "no column is interpretable".
  EXERCISE  (VOID lanes, PS.01-v2's range-over-admitted-behaviours idiom)
            baseline must show: all random lives die; forager meals >= 3,
            drinks >= 2, bursts >= 1; drop probe damaging impacts >= 2;
            placebo OU fit finite. A venue that stopped exercising the suite
            VOIDs the matrix rather than failing a need for it.
  LAMBDA    lambda_rescale_err_max <= 1e-9 (rig arithmetic, VOID otherwise).
  CONTROL   `_control` exercises the rig's two validators with sabotaged
            configurations: a rescale deliberately skipped must be CAUGHT
            (c_lambda_catch) and an obs channel deliberately left live must
            be CAUGHT (c_leak_catch). Pure arithmetic, no physics.

## Venue limitations, declared loudly (none is a weakening — first
## registration, and each narrows the CLAIM, not a bar)

* SLEEP (p) is EXCEPTED by the registry's own notes: removing p also removes
  the consolidation trigger, so its ablation is NE.05's `timer` arm.
* SOCIAL (c) is UNEXERCISABLE in W0: no social event has ever fired (there
  are no people), c has no body coupling and no in-rig consumer, so its
  column would duplicate the placebo BY CONSTRUCTION. Its arm is NOT run;
  `c_unexercised = 1` is recorded and the claim set is {e, w, T, f, i}.
  That is a statement about the VENUE (GOAL.md's company need has nowhere to
  act), not about the need — deleting c off this matrix would be the exact
  wrong-bucket move LESSONS warns about. When the spec first RUNS, this
  metric routes to the queue as a venue finding.
* At this rung no learner exists (T2.01 FAIL): e/w columns are carried by
  the scripted forager's OWN thresholds, so for them the matrix measures
  that the need is LOAD-BEARING IN THE WORLD+SCRIPT SYSTEM — the
  dependencies a competent agent must have — not that a learner uses them
  (NE.03's question, explicitly out of scope here). f's pacing and i's
  weakness columns are physics-level and carry independent information.
* STANDING: this spec re-runs on every change to the suite. Columns and
  margins are expected to MOVE as the suite changes; the bars do not.

## What this cannot say

That needs teach anything (NE.03), that the suite's constants are right
(NE.01), or anything about a learner's use of the need channels.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# The suite under test is the world + integrator pair, read through NE.01's
# rig helpers (shared on purpose: one venue, two specs; a drift in either is
# a loud stale on both).
IMPL_DEPS = [
    "playground.py",
    "experiments/needs.py",
    "experiments/drives.py",
    "experiments/tests/ne_01_nobody_survives_by_accident.py",
]

REPO = Path(__file__).resolve().parents[2]

# ── the suite (8 needs: 7 real + placebo x) ─────────────────────────────
REAL_KEYS = ("e", "w", "p", "T", "f", "c", "i")
LAMBDA_X = 1.0               # median of the real lambdas (docstring)
SETPOINT = {"e": 1.0, "w": 1.0, "p": 0.05, "T": 37.0,
            "f": 0.0, "c": 1.0, "i": 1.0}
CAUSES_OF = {"e": ("starvation",), "w": ("dehydration",),
             "T": ("hypothermia", "hyperthermia"),
             "i": ("injury", "drowning")}
ABLATABLE = ("e", "w", "T", "f", "i")      # p excepted, c unexercisable (doc)
ARMS = ("base",) + ABLATABLE + ("x",)      # x = the placebo ablation
NEED_DIM_02 = 11             # [e,w,p,Tn,f,c,i,x,d,pain,gear]

# ── rollout sizes (declared; the registered run is CPU_LONG) ────────────
N_RANDOM = 3
RANDOM_CAP = 3000            # 600 s — past the statue's 570 s dehydration
FORAGER_DEC = 6000           # one full sim-day, night sleep included
CAL_CAP = 600                # calibration lives (OU fit + f power match)
DROP_N = 3
DROP_H = 1.2                 # m — NE.01's drop-probe regime
PACE_W = 10                  # decisions averaged at each end of a burst

# ── pre-registered bars ─────────────────────────────────────────────────
PLACEBO_MARGIN = 3.0
R_MIN = 0.15
PLACEBO_MAX = 0.20
LAMBDA_TOL = 1e-9
MIN_MEALS, MIN_DRINKS, MIN_BURSTS, MIN_DAMAGING = 3, 2, 1, 2

# ── relative-effect floors, natural units (docstring) ───────────────────
FLOORS = {
    "lifespan": 300.0,       # decisions — the death window's own lower edge
    "meals": 1.0, "drinks": 1.0, "sleep_dec": 500.0, "alive_dec": 600.0,
    "e_sig": 1.0,            # bursts
    "w_sig": 1.0,            # pool visits
    "T_sig": 0.2,            # cause share
    "f_sig": 0.2,            # pacing ratio
    "i_sig": 0.2,            # min gear over the drop probe
}
BATTERY = ("meals", "drinks", "sleep_dec", "alive_dec")
SIG_OF = {"e": "e_sig", "w": "w_sig", "T": "T_sig", "f": "f_sig",
          "i": "i_sig"}
ALL_STATS = ("lifespan",) + BATTERY + tuple(SIG_OF.values())

_NE1 = None


def _ne1():
    """NE.01's rig helpers, imported lazily (they import mujoco/playground)."""
    global _NE1
    if _NE1 is None:
        from . import ne_01_nobody_survives_by_accident as m
        _NE1 = m
    return _NE1


# ── the suite wrapper ────────────────────────────────────────────────────
def _lambda8():
    from experiments import needs
    lam = dict(needs.LAMBDA)
    lam["x"] = LAMBDA_X
    return lam


def _rescaled(removed: str | None, *, skip_rescale: bool = False) -> dict:
    """The arm's lambda table. With `removed`, the remaining lambdas scale by
    S/(S - lambda_removed) so max_h d(h) = sqrt(S) is unchanged.
    `skip_rescale` exists ONLY for the control's sabotage path."""
    lam = _lambda8()
    if removed is None:
        return lam
    s_full = sum(lam.values())
    lam_k = lam.pop(removed)
    if skip_rescale:
        return lam
    scale = s_full / (s_full - lam_k)
    return {k: v * scale for k, v in lam.items()}


def _lambda_err(lam: dict) -> float:
    """|sum(lam) - S_full| — the rescale-exactness validator (control 2)."""
    return abs(sum(lam.values()) - sum(_lambda8().values()))


class _OU:
    """The placebo's band-limited noise: an OU process on delta_x in [0,1],
    (tau, mu, sigma) fitted to the median-tau real need, DEDICATED RNG."""

    def __init__(self, tau: float, mu: float, sigma: float, seed: int):
        import numpy as np
        self.tau = max(1.0, min(600.0, float(tau)))
        self.mu, self.sigma = float(mu), float(sigma)
        self.rng = np.random.RandomState(seed)
        self.x = self.mu

    def step(self, dt: float) -> float:
        import numpy as np
        a = dt / self.tau
        self.x += (self.mu - self.x) * a + (
            self.sigma * math.sqrt(2.0 * a) * self.rng.randn())
        self.x = float(np.clip(self.x, 0.0, 1.0))
        return self.x


class AblatedSuite:
    """Wraps a NeedLayer: one arm of the matrix. Owns the clamp, the
    rescaled d(h), the 11-float obs, the placebo variable and the resurrect
    guard. `leak` deliberately skips the obs zeroing — the control's
    sabotage, caught by `obs_leak_err`."""

    def __init__(self, layer, removed: str | None, ou: "_OU",
                 *, leak: bool = False, skip_rescale: bool = False):
        if removed is not None and removed not in ABLATABLE + ("x",):
            raise ValueError(f"not an ablatable need: {removed}")
        self.layer = layer
        self.removed = removed
        self.ou = ou
        self.leak = leak
        self.lam = _rescaled(removed, skip_rescale=skip_rescale)
        self.lambda_err = _lambda_err(self.lam)
        self.resurrections = 0
        self.x_delta = ou.mu if removed != "x" else 0.0

    def _clamp(self) -> None:
        r = self.removed
        if r in SETPOINT:
            self.layer.state = replace(self.layer.state, **{r: SETPOINT[r]})

    def decide(self):
        s = self.layer.decide()
        if self.layer.dead and self.removed in CAUSES_OF:
            rec = self.layer.death_record
            if rec and rec["cause"] in CAUSES_OF[self.removed]:
                self.resurrections += 1
                self.layer.dead = False
                self.layer.death_record = None
                self.layer._submerged_since = None   # drowning routes via i
        self._clamp()
        if self.removed != "x":
            self.x_delta = self.ou.step(self.layer.last_dt or 0.2)
        return self.layer.state if not self.layer.dead else s

    def deltas(self) -> dict:
        from experiments import needs
        s = self.layer.state
        d = needs.deltas(s.e, s.w, s.p, s.T, s.f, s.c, s.i)
        d["x"] = self.x_delta
        if self.removed is not None:
            d[self.removed] = 0.0
        return d

    def drive(self) -> float:
        d = self.deltas()
        from experiments import needs
        s = sum(self.lam[k] * d[k] ** needs.N_EXP for k in self.lam)
        return float(s ** (1.0 / needs.M_EXP))

    def obs(self):
        import numpy as np
        from experiments import needs
        s = self.layer.state
        d = self.deltas()
        v = np.array([s.e, s.w, s.p, needs.t_norm(s.T), s.f, s.c, s.i,
                      d["x"], self.drive(), self.layer.last_pain,
                      self.layer.gear_scale()], dtype=np.float64)
        idx = {"e": 0, "w": 1, "p": 2, "T": 3, "f": 4, "c": 5, "i": 6, "x": 7}
        if self.removed is not None and not self.leak:
            v[idx[self.removed]] = 0.0
        if v.shape[0] != NEED_DIM_02:
            raise RuntimeError(f"obs is {v.shape[0]}, not {NEED_DIM_02}")
        return v

    def obs_leak_err(self) -> float:
        """The leak validator: the ablated channel must read 0 in obs.
        With `leak=True` (control sabotage) this reads > 0 and must be
        CAUGHT by the caller."""
        if self.removed is None or self.removed == "x":
            return 0.0
        idx = {"e": 0, "w": 1, "p": 2, "T": 3, "f": 4, "c": 5, "i": 6}
        return float(abs(self.obs()[idx[self.removed]]))


# ── calibration (per seed, before any arm is scored) ────────────────────
def _calibrate(seed: int) -> dict:
    """One short random life per config: (a) real-need delta trajectories
    for the placebo's OU fit; (b) mean mechanical power with f live vs f
    clamped, for the kills-field kappa_act match."""
    import numpy as np
    n1 = _ne1()

    def _life(clamp_f: bool):
        p, model, data, water, pool = n1._build(seed)
        layer = n1._layer(model, pool, seed * 211 + (7 if clamp_f else 3))
        rng = np.random.RandomState(seed * 4241 + (17 if clamp_f else 5))
        rows, powers = [], []
        for _ in range(CAL_CAP):
            if layer.dead:
                break
            if clamp_f:
                layer.state = replace(layer.state, f=0.0)
            ctrl = rng.uniform(-n1.CTRL_SCALE, n1.CTRL_SCALE,
                               model.nu) * layer.gear_scale()
            n1._step_decision(model, data, water, layer, ctrl)
            rows.append(n1._deltas_row(layer))
            powers.append(layer.last_power_w)
        del model, data, water
        return np.array(rows), float(np.mean(powers)) if powers else 0.0

    rows, p_base = _life(clamp_f=False)
    _, p_fclamp = _life(clamp_f=True)
    if rows.size == 0 or len(rows) < 20:
        return {"tau": float("nan"), "mu": 0.0, "sigma": 0.0, "ou_need": "e",
                "p_base": p_base, "p_fclamp": p_fclamp, "kappa_ratio": 1.0,
                "cal_finite": 0}

    fits = {}
    dt = 0.2
    for j, k in enumerate("ewpTfci"):
        col = rows[:, j]
        if len(col) < 20 or float(np.std(col)) < 1e-12:
            fits[k] = (600.0, float(np.mean(col)), 0.0)
            continue
        a, b = col[1:] - col[1:].mean(), col[:-1] - col[:-1].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        rho1 = float((a * b).sum() / denom) if denom > 0 else 0.0
        rho1 = min(max(rho1, 1e-6), 1.0 - 1e-9)
        fits[k] = (-dt / math.log(rho1), float(np.mean(col)),
                   float(np.std(col)))
    med_k = sorted("ewpTfci", key=lambda k: fits[k][0])[3]  # median tau
    tau, mu, sigma = fits[med_k]
    kappa_ratio = p_base / p_fclamp if p_fclamp > 0 else 1.0
    return {"tau": tau, "mu": mu, "sigma": sigma, "ou_need": med_k,
            "p_base": p_base, "p_fclamp": p_fclamp,
            "kappa_ratio": float(kappa_ratio),
            "cal_finite": int(np.isfinite([tau, mu, sigma,
                                           kappa_ratio]).all())}


# ── arm construction ─────────────────────────────────────────────────────
def _make_suite(model, pool, removed, cal, seed, salt, **kw) -> AblatedSuite:
    n1 = _ne1()
    layer = n1._layer(model, pool, seed * 131 + salt)
    if removed == "f":
        layer.kappa_act *= cal["kappa_ratio"]   # energy-drain match (kills)
    ou = _OU(cal["tau"], cal["mu"], cal["sigma"], seed * 7919 + salt)
    return AblatedSuite(layer, removed, ou, **kw)


# ── probes, one arm each ─────────────────────────────────────────────────
def _random_lives_arm(seed: int, removed, cal) -> dict:
    import numpy as np
    n1 = _ne1()
    spans, causes, res = [], [], 0
    for life in range(N_RANDOM):
        p, model, data, water, pool = n1._build(seed)
        suite = _make_suite(model, pool, removed, cal, seed, life)
        layer = suite.layer
        rng = np.random.RandomState(seed * 9973 + life * 17)
        k = 0
        while not layer.dead and k < RANDOM_CAP:
            ctrl = rng.uniform(-n1.CTRL_SCALE, n1.CTRL_SCALE,
                               model.nu) * layer.gear_scale()
            n1._step_decision(model, data, water, layer, ctrl)
            suite.decide()                       # clamp + placebo + guard
            k += 1
        spans.append(k)                          # censored at cap, declared
        rec = layer.death_record
        causes.append(rec["cause"] if rec else "none")
        res += suite.resurrections
        del model, data, water
    thermal = sum(c in ("hyperthermia", "hypothermia") for c in causes)
    return {"lifespan": float(np.median(spans)),
            "n_died": sum(c != "none" for c in causes),
            "T_sig": thermal / float(N_RANDOM),
            "resurrections": res,
            "lambda_err": _lambda_err(_rescaled(removed))}


def _forager_arm(seed: int, removed, cal) -> dict:
    """NE.01's forager, adapted: 1 sim-day, reads the WRAPPED state (the
    clamp propagates through every script threshold), and records per-burst
    power for the pacing signature. Teleports/serving verbatim from NE.01.
    DECLARED DIVERGENCE: he sleeps the night at OPEN_SPOT, not at a shelter
    pose — NE.01 measured the open night survivable-once (cost 0.498) and
    the shelter band NOT dynamically held by a sleeping body (its own FAIL);
    a battery leaning on a broken affordance would measure the world's
    defect, not the needs."""
    import numpy as np
    from experiments import needs
    n1 = _ne1()
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index

    p, model, data, water, pool = n1._build(seed)
    qadr = humanoid_index(model)["qposadr"]
    suite = _make_suite(model, pool, removed, cal, seed, 389)
    layer = suite.layer
    rng = np.random.RandomState(seed * 613 + 11)
    head_id = int(model.geom("head").id)

    joints = {}
    for name in ("obj0", "obj1", "apple"):
        try:
            bid = int(model.body(name).id)
        except (KeyError, ValueError):
            continue
        jadr = int(model.body_jntadr[bid])
        if jadr < 0:
            continue
        joints[name] = (int(model.jnt_qposadr[jadr]),
                        int(model.jnt_dofadr[jadr]))

    def serve(name):
        qa, da = joints[name]
        hp = np.asarray(data.geom_xpos[head_id])
        data.qpos[qa:qa + 3] = (hp[0], hp[1], hp[2] + 0.25)
        data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[da:da + 6] = 0.0

    due = {name: 0.0 for name in joints}
    ate0 = dict(layer.ate_total)
    asleep = False
    pool_visit, pool_visits = 0, 0
    burst, bursts = 0, 0
    burst_powers, pace = [], []
    sleep_dec = 0
    k_done = 0
    n1._place(model, data, qadr, n1.HOME_SPOT)
    for k in range(FORAGER_DEC):
        if layer.dead:
            break
        s = layer.state
        want_sleep = layer.is_night()
        if want_sleep and not asleep and pool_visit == 0:
            if s.w < 0.9 and layer.t >= layer._drink_ready_at:
                n1._place(model, data, qadr, (pool[0], pool[1], 0.30))
                pool_visit = 10
                pool_visits += 1
            else:
                n1._place(model, data, qadr,
                          (n1.OPEN_SPOT[0], n1.OPEN_SPOT[1], 0.25))
                layer.set_asleep(True)
                asleep = True
        elif asleep and s.w < 0.12:
            layer.set_asleep(False)
            asleep = False
        elif not want_sleep and asleep:
            layer.set_asleep(False)
            n1._place(model, data, qadr, n1.HOME_SPOT)
            asleep = False

        ctrl = np.zeros(model.nu)
        if asleep:
            sleep_dec += 1
        else:
            if pool_visit > 0:
                pool_visit -= 1
                if pool_visit == 0:
                    n1._place(model, data, qadr, n1.HOME_SPOT)
            elif (not layer.is_night()
                  and layer.last_occlusion > n1.DAY_OCC_FLEE):
                n1._place(model, data, qadr, n1.HOME_SPOT)
            elif s.w < 0.6 and layer.t >= layer._drink_ready_at:
                n1._place(model, data, qadr, (pool[0], pool[1], 0.30))
                pool_visit = 10
                pool_visits += 1
            elif burst > 0:
                burst -= 1
                ctrl = rng.uniform(-n1.CTRL_SCALE, n1.CTRL_SCALE,
                                   model.nu) * layer.gear_scale()
                if burst == 0 and "apple" in joints:
                    serve("apple")
                    if len(burst_powers) >= 2 * PACE_W:
                        pace.append(
                            float(np.mean(burst_powers[-PACE_W:]))
                            / max(float(np.mean(burst_powers[:PACE_W])),
                                  1e-9))
                    burst_powers = []
            elif ("apple" in joints and layer.t >= due["apple"]
                  and s.e < 0.9 and s.T < n1.BURST_T_MAX):
                burst = n1.APPLE_BURST_DEC
                bursts += 1
            else:
                for name in ("obj0", "obj1"):
                    if name in joints and layer.t >= due[name] and s.e < 0.95:
                        serve(name)
                if rng.random_sample() < n1.FORAGER_DUTY:
                    ctrl = rng.uniform(-n1.CTRL_SCALE, n1.CTRL_SCALE,
                                       model.nu) * layer.gear_scale()
        n1._step_decision(model, data, water, layer, ctrl)
        suite.decide()
        if burst > 0 or (burst == 0 and burst_powers):
            burst_powers.append(layer.last_power_w)
        for name in joints:
            if layer.ate_total[name] > ate0[name]:
                due[name] = layer.t + needs.FOOD[name][1]
                ate0[name] = layer.ate_total[name]
        k_done = k + 1

    out = {
        "alive_dec": float(k_done),
        "meals": float(sum(layer.ate_total.values())),
        "drinks": float(layer.drank_total),
        "sleep_dec": float(sleep_dec),
        "e_sig": float(bursts),
        "w_sig": float(pool_visits),
        "f_sig": float(np.mean(pace)) if pace else -1.0,
        "n_bursts_paced": len(pace),
        "forager_resurrections": suite.resurrections,
    }
    del model, data, water
    return out


def _drop_arm(seed: int, removed, cal) -> dict:
    """Integrity's signature venue: DROP_N drops from DROP_H; the column is
    the MIN gear_scale reached (damage -> weakness). Teleport arrival speed
    is zeroed by _place, so impacts come from the fall itself."""
    import numpy as np
    n1 = _ne1()
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index

    p, model, data, water, pool = n1._build(seed)
    qadr = humanoid_index(model)["qposadr"]
    suite = _make_suite(model, pool, removed, cal, seed, 547)
    layer = suite.layer
    gear_min, n_damaging = 1.0, 0
    for d in range(DROP_N):
        n1._place(model, data, qadr,
                  (n1.OPEN_SPOT[0], n1.OPEN_SPOT[1], DROP_H))
        for _ in range(60):                     # ~12 s: fall + settle
            if layer.dead:
                break
            j0 = layer.n_onsets
            n1._step_decision(model, data, water, layer,
                              np.zeros(model.nu))
            suite.decide()
            if layer.n_onsets > j0 and layer.last_j > layer.j0:
                n_damaging += 1
            gear_min = min(gear_min, layer.gear_scale())
        if layer.dead:
            break
    out = {"i_sig": float(gear_min), "n_damaging": n_damaging,
           "drop_resurrections": suite.resurrections}
    del model, data, water
    return out


# ── the matrix ───────────────────────────────────────────────────────────
def _rel(stat: str, arm_v: float, base_v: float) -> float:
    return abs(arm_v - base_v) / max(abs(base_v), FLOORS[stat])


def _arm_stats(seed: int, removed, cal) -> dict:
    r = _random_lives_arm(seed, removed, cal)
    f = _forager_arm(seed, removed, cal)
    d = _drop_arm(seed, removed, cal)
    return {**r, **f, **d}


def _experiment(seed: int) -> dict:
    import numpy as np

    b = _ne1()._borrowed()
    if not b.ok:
        return {"void": f"VOID: {b.refusal}", "all_finite": 0}

    cal = _calibrate(seed)
    if not cal["cal_finite"]:
        return {"void": "VOID: calibration non-finite", "all_finite": 0}

    stats = {arm: _arm_stats(seed, None if arm == "base" else arm, cal)
             for arm in ARMS}
    base = stats["base"]

    # placebo column: max relative shift over EVERY statistic
    placebo_col = max(_rel(s, stats["x"][s], base[s]) for s in ALL_STATS
                      if base[s] >= 0 and stats["x"][s] >= 0)

    bar = max(PLACEBO_MARGIN * placebo_col, R_MIN)
    m: dict = {"all_finite": 1, "placebo_col": placebo_col, "bar": bar,
               "c_unexercised": 1,          # W0 has no people (docstring)
               **{f"cal_{k}": v for k, v in cal.items()
                  if k != "ou_need"},
               "lambda_rescale_err_max": 0.0,
               "resurrections_total": 0}
    m["ou_need_idx"] = float("ewpTfci".index(cal["ou_need"]))

    margins = []
    for k in ABLATABLE:
        a = stats[k]
        effs = [_rel("lifespan", a["lifespan"], base["lifespan"]),
                max(_rel(s, a[s], base[s]) for s in BATTERY)]
        sig = SIG_OF[k]
        if base[sig] >= 0 and a[sig] >= 0:
            effs.append(_rel(sig, a[sig], base[sig]))
        col = max(effs)
        m[f"col_{k}"] = col
        m[f"margin_{k}"] = col / bar
        margins.append(col / bar)
        m["lambda_rescale_err_max"] = max(m["lambda_rescale_err_max"],
                                          a["lambda_err"])
        m["resurrections_total"] += (a["resurrections"]
                                     + a["forager_resurrections"]
                                     + a["drop_resurrections"])
    m["min_need_margin_over_placebo"] = float(min(margins))

    # exercise gates, all on the BASELINE arm
    m["ex_random_died"] = int(base["n_died"] == N_RANDOM)
    m["ex_meals"] = int(base["meals"] >= MIN_MEALS)
    m["ex_drinks"] = int(base["drinks"] >= MIN_DRINKS)
    m["ex_bursts"] = int(base["e_sig"] >= MIN_BURSTS)
    m["ex_damaging"] = int(base["n_damaging"] >= MIN_DAMAGING)
    m["ok_exercised"] = int(all((m["ex_random_died"], m["ex_meals"],
                                 m["ex_drinks"], m["ex_bursts"],
                                 m["ex_damaging"])))
    for arm in ARMS:
        for s in ("lifespan", "meals", "drinks", "e_sig", "w_sig",
                  "T_sig", "f_sig", "i_sig", "alive_dec"):
            m[f"{arm}_{s}"] = stats[arm][s]
    m["ok_claim"] = int(m["min_need_margin_over_placebo"] >= 1.0)
    m["ok_placebo_small"] = int(placebo_col <= PLACEBO_MAX)
    m["ok_lambda"] = int(m["lambda_rescale_err_max"] <= LAMBDA_TOL)
    return m


def _control(seed: int) -> dict:
    """The rig's validators, exercised with SABOTAGED configurations (pure
    arithmetic, no physics — W1.04's fixture-control idiom). Both sabotages
    must be CAUGHT, and the clean paths must stay clean."""
    lam_bad = _rescaled("e", skip_rescale=True)
    lam_good = _rescaled("e")

    class _FakeLayer:
        class _S:
            e, w, p, T, f, c, i = 1.0, 1.0, 0.05, 37.0, 0.0, 1.0, 1.0
        state = _S()
        last_pain, last_dt = 0.0, 0.2
        dead = False
        death_record = None
        _submerged_since = None

        def gear_scale(self):
            return 1.0

        def decide(self):
            return self.state

    ou = _OU(60.0, 0.3, 0.05, seed)
    leaky = AblatedSuite(_FakeLayer(), "e", ou, leak=True)
    # the leaky suite's state must differ from setpoint for the leak to
    # read nonzero; sabotage the state too
    leaky.layer.state.e = 0.4
    clean = AblatedSuite(_FakeLayer(), "e", _OU(60.0, 0.3, 0.05, seed + 1))
    clean.layer.state.e = 0.4

    return {
        "all_finite": 1,
        "c_lambda_catch": int(_lambda_err(lam_bad) > LAMBDA_TOL),
        "c_lambda_clean": int(_lambda_err(lam_good) <= LAMBDA_TOL),
        "c_leak_catch": int(leaky.obs_leak_err() > 0.0),
        "c_leak_clean": int(clean.obs_leak_err() == 0.0),
    }


def _check(m: dict, c: dict):
    from ..protocol import _declares_void
    if _declares_void(m) or _declares_void(c):
        return Status.VOID
    # VOID lanes: a venue that stopped exercising the suite, a rescale that
    # is not exact, or a placebo column too large to interpret any column
    if m["ok_exercised"] != 1.0:
        return Status.VOID
    if m["ok_lambda"] != 1.0:
        return Status.VOID
    if m["ok_placebo_small"] != 1.0:
        return Status.VOID
    return bool(
        m["all_finite"] == 1 and c["all_finite"] == 1
        # the claim: every exercised need's column clears the placebo bar
        and m["ok_claim"] == 1.0
        # the resurrect guard stayed theoretical (reported either way)
        # — NOT gated: a nonzero count is a finding, not a verdict
        # controls: both sabotages caught, both clean paths clean
        and c["c_lambda_catch"] == 1.0 and c["c_lambda_clean"] == 1.0
        and c["c_leak_catch"] == 1.0 and c["c_leak_clean"] == 1.0
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["NE.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
