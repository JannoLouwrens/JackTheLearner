"""BA.02 — He catches himself: the felt fall changes what he does.

HYPOTHESIS (registry, unchanged). A learner given BA.01's vestibular channel
ACTS on it: trained in a topple-costly regime, it stays upright measurably
longer than an identical learner trained with the channel deleted (>= 3 sigma
across seeds), and the gain vanishes when the channel is replaced by
matched-statistics noise.

THE CHANNEL IS BA.01'S, BY REFERENCE. BA.01 registered the orientation channel
as the whole graviceptive suffix — touch (8, plantar graviception: N ~ m(g-a_z))
plus vestibular (11: gravity-in-body 3, canals 3, otoliths 3, root vx/vy) — and
its control deletes exactly that block. BA.02 inherits the block, its ordering,
and the tilt/kick draw constants by IMPORT from `ba_01_feels_the_fall` (the
"when you can reference, reference" rule): if BA.01's rig moves, this claim
goes stale with it via IMPL_DEPS, never silently.

THE LEARNER IS THE SIMPLEST ONE THAT COULD ACT ON THE CHANNEL (the T1.02
reference-arm lesson, pointed forward): a LINEAR reactive policy
a[:4] = tanh(W z + b) over the standardized 27-dim observation, trained by
cross-entropy method (CEM) directly on upright time — no reward shaping, no
value function, no learning-rate fragility to hide a null result behind. The
claim is existential ("a learner given the channel acts on it"), so the
simplest learner PASSing is sufficient; and if it FAILs, the failure is about
the information, not about tuning, precisely because there is nothing to tune.
CEM fitness = the pre-registered metric itself.

ARMS — all four trained/evaluated on the SAME worlds with PAIRED episode draws
(spawn site, tilt angle, kick magnitude and direction identical across arms,
episode for episode — the PS.03 paired-draws pattern):

  vest      (experiment)  the full standardized observation.
  deprived  (null twin)   graviceptive suffix pinned to its training mean
                          (zero after standardization): identical architecture,
                          zero information in the channel. This is BA.01's own
                          control, inherited as the registry demands. Note the
                          twin is NOT helpless: open-loop postural strategies
                          (where to hold the arms) are fully available to it,
                          which is exactly why it is the right null.
  noise     (declared control, `_control`) suffix replaced each decision by
                          fresh N(0,1) draws — matched statistics in
                          standardized space, zero information. Separates
                          information content from input-width/regularisation
                          effects; its gain over the deprived twin must VANISH.
  random    (null)        the untrained policy class: a[:4] ~ U[-1,1] each
                          decision. The registry's second null.

THE RIG. Uniform LEGAL spawns, arms at spawn target (hold action derived from
the live ctrlrange — PS.03's phantom-servo scar: a neutral-looking action value
is not a neutral action; a=0 commands mid-range), adhesion OFF (a=-1 maps to
the ctrlrange floor) and drive zero: catching yourself is an arm-mass problem,
not a thruster problem. T_SETTLE decisions of hold, then BA.01's v4 tilt+kick
draw (log-uniform tilt over two decades, tilt-proportional kick, random
direction), then the policy owns the slides until upright cosine < TOPPLE_UP
or HORIZON. Fitness and metric: decisions upright, reported in sim-seconds.

DECLARED DEVIATION from BA.01's v4 rig: NO boundary spawns and NO aimed falls
(P_STRUCT = 0 here). BA.01 added those to spread FALL TIMES for its clock null
— a probe-side need this spec does not have — and a body that topples against
a wall and rests on it would score "upright longer" by leaning, which is
shelter, not catching. Every fall here is on open ground where only the body's
own action can change the outcome.

Every episode starts through `w.respawn(at=site)`, which resets the drive
state to a fresh body — so gear_scale is 1.0 in every episode and no
energy-decay weakness drifts across a CEM run (checked: drives.new_body).

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The gain is input-width, not information** — 27 learnable weights beat 8
   for optimisation-landscape reasons unrelated to sensing. Caught by the
   declared control: the noise arm has identical width and matched per-dim
   statistics, and its gain must vanish.

2. **The rig cannot be caught in** — every draw is unrecoverable (or none
   topple), so vest-vs-deprived measures seed noise on a task with no
   headroom. Caught by per-seed rig gates (VOID, not FAIL): the random policy
   must topple on >= TOPPLED_FRAC_MIN of eval episodes AND survive on average
   under RANDOM_UP_FRAC_MAX of the horizon (falls exist, headroom exists),
   and at least one TRAINED arm must beat the random policy by
   IMPROVE_MARGIN_MIN (the learner demonstrably functioned in this world —
   open-loop postural learning is available to every arm, so a world where
   no arm learns anything is a world that tested nothing).

3. **The gain is a seed lottery** — one lucky world carries the mean. Caught
   by the registry's own bar: all seeds individually positive
   (gain_positive == 1.0) and the across-seed t >= 3 (sample-std form,
   derivation at the gate).

## WHAT FAIL WOULD MEAN

FAIL is reserved for the sense going unused: trained with the felt channel, he
stays upright no longer than his deprived twin (or the "gain" survives the
noise substitution, meaning it was never information). Then balance is decoded
but not acted on — BA.01 measured a spectator — and every claim that assumes
he can use up-from-down inherits that hole (the registry's kills field).

## ANATOMY, REPORTED NEVER GATED (registry: report the channels separately)

The trained vest policy is re-evaluated on the paired eval draws with one
sub-block at a time pinned to its mean: touch, grav, canals, otoliths, vx/vy.
The upright-time drop per block says which organ the winning policy actually
reads. Retraining-per-ablation would be the stronger claim and is T3-family
work; this is the cheap honest version and is labelled as such.

PILOT: seed 90, disjoint from the registered seeds 0/1/2 (BA.01/PS.02/PS.03
precedent). The pilot measures throughput (the registry pre-authorises
re-costing the TIER, never the thresholds), catchability, and the null/noise
levels. Auxiliary gate constants marked PILOT-FINAL below are finalised in the
registration commit BEFORE the registered run; T_GAIN_MIN = 3.0 is the
registry's own bar and does not move.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from ..w0 import W0, SIM_S_PER_DECISION                           # noqa: E402
# BA.01's rig constants and helpers, by reference (one definition of the fall).
from .ba_01_feels_the_fall import (GRAVITY, KICK_JIT, KICK_OMEGA_P,  # noqa: E402
                                   TILT0_LOG10_DEG, TOPPLE_UP, VEST_DIM,
                                   _tilt_quat)

# The claim goes stale when the world, the body, the drive layer or the sense's
# own defining rig moves.
IMPL_DEPS = ["experiments/w0.py", "playground.py", "experiments/drives.py",
             "experiments/tests/ba_01_feels_the_fall.py"]

# ── observation layout (BA.01's ordering, verbatim) ─────────────────────
BLIND_DIM = 8                 # 4 arm slide positions + 4 slide velocities
TOUCH_DIM = 8
CH_DIM = TOUCH_DIM + VEST_DIM  # the graviceptive suffix BA.01 registered (19)
OBS_DIM = BLIND_DIM + CH_DIM   # 27
ACT_DIM = 4                    # the slides; adhesion OFF, drive zero
THETA_DIM = ACT_DIM * OBS_DIM + ACT_DIM  # 112 linear-policy parameters

# ── the rig envelope (PRE-REGISTERED) ───────────────────────────────────
HORIZON = 60                  # decisions = 12 s; catches run to the horizon
T_SETTLE = 3                  # hold decisions before the tilt+kick (BA.01)
N_STATS_EP = 12               # random-policy pre-pass sizing mu/sd + noise match
N_EVAL = 48                   # paired eval episodes per arm per seed

# ── the learner (PRE-REGISTERED; identical across arms by construction) ─
CEM_POP = 24
CEM_ELITE = 6
CEM_ITERS = 12
CEM_K_FIT = 3                 # episodes per candidate, COMMON DRAWS per iter
CEM_SIG_INIT = 0.5
CEM_SIG_FLOOR = 0.05

# ── gates ───────────────────────────────────────────────────────────────
# THE REGISTRY'S BAR (constitutional here, does not move): >= 3 sigma across
# seeds, every seed positive.
T_GAIN_MIN = 3.0
# Rig gates (VOID, not FAIL — a world that could not test the claim).
# PILOT-FINAL: values below are candidates; finalised in the registration
# commit with the seed-90 pilot numbers beside them, per BA.01 precedent.
TOPPLED_FRAC_MIN = 0.60       # random policy must actually fall   [PILOT-FINAL]
RANDOM_UP_FRAC_MAX = 0.80     # ...but not survive ~the horizon    [PILOT-FINAL]
IMPROVE_MARGIN_MIN = 0.20     # sim-s: best trained arm over random [PILOT-FINAL]
# Control gates (the noise gain must VANISH).
NOISE_GAIN_FRAC_MAX = 0.50    # gain_noise <= this fraction of gain [PILOT-FINAL]
VEST_OVER_NOISE_MIN = 0.20    # sim-s: gain - gain_noise floor      [PILOT-FINAL]

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's j0/alpha, or a refusal. W0 has no defaults for them (BA.01)."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _arm_hold(w: W0) -> np.ndarray:
    """The 8-vector that commands the SPAWN pose and nothing else.

    Slides spawn at joint target 0 (`_place` zeroes them); the action that
    holds them there is derived from the live ctrlrange (PS.03's scar: a=0 is
    mid-range, and the servo snap read as a phantom fall). Adhesion a=-1 maps
    to the range floor (off); drive dims are a force scale, 0 is genuinely
    none.
    """
    lo = np.asarray(w.model.actuator_ctrlrange[:4, 0], float)
    hi = np.asarray(w.model.actuator_ctrlrange[:4, 1], float)
    a = np.zeros(8)
    a[:4] = 2.0 * (0.0 - lo) / (hi - lo) - 1.0
    a[4:6] = -1.0
    return a


def _draw_pack(rng: np.random.RandomState, legal: np.ndarray) -> dict:
    """One episode's world draw — everything that must be PAIRED across arms."""
    k = int(rng.randint(len(legal)))
    theta = math.radians(10.0 ** rng.uniform(*TILT0_LOG10_DEG))
    mag = theta * KICK_OMEGA_P * 10.0 ** rng.uniform(*KICK_JIT)
    u = rng.randn(3)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return {"site": (float(legal[k][0]), float(legal[k][1])),
            "aim": float(rng.uniform(0.0, 2.0 * math.pi)),
            "theta": theta, "kick": u * mag}


def _obs_row(w: W0, v_prev: np.ndarray) -> tuple:
    """BA.01's 27-dim row, ordering verbatim: blind(8) + touch(8) + vest(11)."""
    da = w.ix["root_dofadr"]
    xmat = w.data.xmat[w.rover_bid]
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    grav_body = -R.T @ np.array([0.0, 0.0, 1.0])
    canals = w.data.qvel[da + 3:da + 6].copy()
    v = w.data.qvel[da:da + 3].copy()
    a_world = (v - v_prev) / SIM_S_PER_DECISION
    otoliths = R.T @ (a_world - np.array([0.0, 0.0, -GRAVITY]))
    p = w._proprio()
    row = np.concatenate([p[:8], w._touch(),
                          grav_body, canals, otoliths, p[10:12]])
    assert row.shape[0] == OBS_DIM
    return row, v, float(xmat[8])


def _episode(w: W0, pack: dict, act_fn, hold: np.ndarray,
             horizon: int = HORIZON) -> tuple:
    """One catch attempt. Returns (decisions upright, rows) — rows only when
    the caller collects statistics (act_fn may ignore them otherwise).

    `respawn(at=...)` resets pose, arms, velocities AND the drive state (a
    fresh body: gear 1.0), so episodes are exchangeable across a whole run.
    """
    mujoco = w.mujoco
    qa, da = w.ix["root_qposadr"], w.ix["root_dofadr"]
    w.respawn(at=pack["site"])
    for _ in range(T_SETTLE):
        w.decide(hold)
    q0 = w.data.qpos[qa + 3:qa + 7].copy()
    qt = _tilt_quat(pack["theta"], pack["aim"])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qt, q0)
    w.data.qpos[qa + 3:qa + 7] = out
    w.data.qvel[da:da + 6] = 0.0
    w.data.qvel[da + 3:da + 6] = pack["kick"]
    mujoco.mj_forward(w.model, w.data)

    v_prev = w.data.qvel[da:da + 3].copy()
    rows = []
    for t in range(horizon):
        row, v_prev, up = _obs_row(w, v_prev)
        if up < TOPPLE_UP:
            return t, rows
        rows.append(row)
        a = np.array(hold)
        a[:4] = act_fn(row)
        w.decide(a)
    return horizon, rows


# ── the policy class and its conditions ─────────────────────────────────
def _policy(theta: np.ndarray, mu: np.ndarray, sd: np.ndarray, cond: str,
            noise_rng: np.random.RandomState | None):
    """cond: 'vest' full obs | 'deprived' suffix at its mean | 'noise'
    suffix i.i.d. N(0,1) per decision | or 'ablate:<a>:<b>' pinning dims
    [a:b) of the STANDARDIZED row to zero (the anatomy evals)."""
    W = theta[:ACT_DIM * OBS_DIM].reshape(ACT_DIM, OBS_DIM)
    b = theta[ACT_DIM * OBS_DIM:]

    def act(row: np.ndarray) -> np.ndarray:
        z = (row - mu) / sd
        if cond == "deprived":
            z[BLIND_DIM:] = 0.0
        elif cond == "noise":
            z[BLIND_DIM:] = noise_rng.randn(CH_DIM)
        elif cond.startswith("ablate:"):
            _, a0, b0 = cond.split(":")
            z[int(a0):int(b0)] = 0.0
        return np.tanh(W @ z + b)
    return act


def _random_policy(rng: np.random.RandomState):
    def act(row: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, ACT_DIM)
    return act


def _cem_train(w: W0, legal: np.ndarray, hold: np.ndarray, mu, sd, cond: str,
               seed: int, iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE,
               k_fit=CEM_K_FIT, horizon=HORIZON) -> tuple:
    """CEM on upright time. Common draws per iteration: every candidate in an
    iteration faces the same k_fit packs, so within-iteration ranking is
    paired. The noise condition's per-decision draws come from a stream
    seeded by (seed, iter, candidate, episode) — deterministic, never shared
    with the world draws."""
    theta = np.zeros(THETA_DIM)
    sig = np.full(THETA_DIM, CEM_SIG_INIT)
    g = np.random.RandomState(seed * 9973 + 101)
    curve = []
    for it in range(iters):
        packs = [_draw_pack(g, legal) for _ in range(k_fit)]
        cands = [theta + sig * g.randn(THETA_DIM) for _ in range(pop)]
        fits = []
        for ci, th in enumerate(cands):
            ups = []
            for ei, pack in enumerate(packs):
                nr = (np.random.RandomState(
                    seed * 1_000_003 + it * 10_007 + ci * 101 + ei)
                      if cond == "noise" else None)
                up, _ = _episode(w, pack, _policy(th, mu, sd, cond, nr),
                                 hold, horizon)
                ups.append(up)
            fits.append(float(np.mean(ups)))
        order = np.argsort(fits)[::-1]
        el = np.stack([cands[i] for i in order[:elite]])
        theta = el.mean(0)
        sig = np.maximum(el.std(0), CEM_SIG_FLOOR)
        curve.append(float(np.mean([fits[i] for i in order[:elite]])))
    return theta, curve


def _eval_policy(w: W0, packs: list, hold: np.ndarray, act_builder,
                 horizon=HORIZON) -> np.ndarray:
    """Mean upright time (sim-s) over the PAIRED eval packs. `act_builder(ei)`
    returns the per-episode act_fn (noise streams differ per episode)."""
    ups = []
    for ei, pack in enumerate(packs):
        up, _ = _episode(w, pack, act_builder(ei), hold, horizon)
        ups.append(up * SIM_S_PER_DECISION)
    return np.asarray(ups)


def _collect(seed: int, iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE,
             k_fit=CEM_K_FIT, n_eval=N_EVAL, n_stats=N_STATS_EP,
             horizon=HORIZON) -> dict:
    """Everything this spec needs for one seed, once. Cached: `_control`
    reuses the same trained arms and paired eval draws (BA.01's pattern —
    re-simulating would let the two sides differ by something other than
    the channel)."""
    key = (seed, iters, pop, k_fit, n_eval, horizon)
    if key in _CACHE:
        return _CACHE[key]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[key] = {"refused": prov}
        return _CACHE[key]
    t0 = time.time()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False)
    legal = w.legal_spawns()
    hold = _arm_hold(w)

    # Stats pre-pass: mu/sd for standardization AND the matched-noise scale.
    srng = np.random.RandomState(seed * 613 + 7)
    rows_all = []
    for _ in range(n_stats):
        pack = _draw_pack(srng, legal)
        _, rows = _episode(w, pack, lambda r: srng.uniform(-1, 1, ACT_DIM),
                           hold, horizon)
        rows_all.extend(rows)
    X = np.asarray(rows_all)
    mu, sd = X.mean(0), X.std(0) + 1e-8

    arms = {}
    curves = {}
    for cond in ("vest", "deprived", "noise"):
        arms[cond], curves[cond] = _cem_train(
            w, legal, hold, mu, sd, cond, seed,
            iters=iters, pop=pop, elite=elite, k_fit=k_fit, horizon=horizon)

    # Paired eval draws, one list per seed, shared by every arm.
    erng = np.random.RandomState(seed * 271 + 17)
    packs = [_draw_pack(erng, legal) for _ in range(n_eval)]

    def builder(cond, th):
        def build(ei):
            nr = (np.random.RandomState(seed * 41 + 900_000 + ei)
                  if cond == "noise" else None)
            return _policy(th, mu, sd, cond, nr)
        return build

    ev = {c: _eval_policy(w, packs, hold, builder(c, arms[c]), horizon)
          for c in arms}
    rrng = np.random.RandomState(seed * 83 + 5)
    ev["random"] = _eval_policy(w, packs, hold,
                                lambda ei: _random_policy(rrng), horizon)
    toppled_random = float(np.mean(ev["random"] <
                                   horizon * SIM_S_PER_DECISION - 1e-9))

    # Anatomy: the trained vest policy with one sub-block pinned (reported).
    nb = BLIND_DIM
    blocks = {"touch": (nb, nb + 8), "grav": (nb + 8, nb + 11),
              "canals": (nb + 11, nb + 14), "otoliths": (nb + 14, nb + 17),
              "vxvy": (nb + 17, nb + 19)}
    anatomy = {}
    for name, (a0, b0) in blocks.items():
        cond = f"ablate:{a0}:{b0}"
        anatomy[name] = float(np.mean(_eval_policy(
            w, packs, hold,
            lambda ei: _policy(arms["vest"], mu, sd, cond, None), horizon)))

    _CACHE[key] = {"ev": ev, "curves": curves, "anatomy": anatomy,
                   "toppled_random": toppled_random, "prov": prov,
                   "wall_s": time.time() - t0, "horizon": horizon}
    return _CACHE[key]


def _rig(c: dict) -> dict:
    """Per-seed rig health; the conjunction is VOID-gated in _check."""
    h_s = c["horizon"] * SIM_S_PER_DECISION
    up_random = float(c["ev"]["random"].mean())
    best_trained = max(float(c["ev"][k].mean())
                       for k in ("vest", "deprived", "noise"))
    ok = (c["toppled_random"] >= TOPPLED_FRAC_MIN
          and up_random <= RANDOM_UP_FRAC_MAX * h_s
          and best_trained - up_random >= IMPROVE_MARGIN_MIN)
    return {"toppled_frac_random": c["toppled_random"],
            "up_random": up_random, "best_trained": best_trained,
            "seed_rig_ok": 1.0 if ok else 0.0}


def _experiment(seed: int, **env) -> dict:
    c = _collect(seed, **env)
    if "refused" in c:
        return {"probe": "VOID", "gain": float("nan"), **c["refused"]}
    up_v = float(c["ev"]["vest"].mean())
    up_d = float(c["ev"]["deprived"].mean())
    gain = up_v - up_d
    out = {"up_vest": up_v, "up_deprived": up_d, "gain": gain,
           "gain_positive": 1.0 if gain > 0 else 0.0,
           **_rig(c),
           "vest_fit_first": c["curves"]["vest"][0],
           "vest_fit_last": c["curves"]["vest"][-1],
           "deprived_fit_first": c["curves"]["deprived"][0],
           "deprived_fit_last": c["curves"]["deprived"][-1],
           "wall_s": c["wall_s"]}
    for k, v in c["anatomy"].items():
        out[f"up_ablate_{k}"] = v
    return out


def _control(seed: int, **env) -> dict:
    """The declared control: matched-statistics noise in the channel. Its gain
    over the deprived twin must vanish."""
    c = _collect(seed, **env)
    if "refused" in c:
        return {"probe": "VOID", "gain_noise": float("nan"), **c["refused"]}
    up_n = float(c["ev"]["noise"].mean())
    up_d = float(c["ev"]["deprived"].mean())
    return {"up_noise": up_n, "gain_noise": up_n - up_d,
            "noise_fit_last": c["curves"]["noise"][-1],
            "seed_rig_ok": _rig(c)["seed_rig_ok"]}


def _declared_void(m: dict) -> bool:
    return m.get("probe") == "VOID" or not np.isfinite(m.get("gain",
                                                             m.get("gain_noise",
                                                                   np.nan)))


def _check(m: dict, c: dict):
    if _declared_void(m) or _declared_void(c):
        return Status.VOID
    # Rig degeneracy is VOID, per seed (mean of the conjunction is 1.0 only
    # when EVERY seed's world both toppled and was learnable at all).
    if m["seed_rig_ok"] < 1.0:
        return Status.VOID
    # The registry's bar: >= 3 sigma across seeds, every seed positive.
    # _aggregate hands the POPULATION std over n=3 seeds; the t statistic
    # wants the SAMPLE std over the seed mean: t = mean / (s/sqrt(n)) with
    # s = std_pop*sqrt(n/(n-1)), which reduces to mean*sqrt(2)/std_pop at n=3.
    t_gain = m["gain"] * math.sqrt(2.0) / max(m.get("gain_std", 0.0), 1e-9)
    ok = (m["gain_positive"] == 1.0
          and t_gain >= T_GAIN_MIN
          # the control must FAIL: the matched-noise gain vanishes
          and c["gain_noise"] <= NOISE_GAIN_FRAC_MAX * m["gain"]
          and m["gain"] - c["gain_noise"] >= VEST_OVER_NOISE_MIN)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["BA.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── smoke and pilot ─────────────────────────────────────────────────────
def _smoke():
    """Tiny envelope, every entry point once, both list positions of nothing
    grown (no list.remove anywhere here — the LC.03 scar's cheap half is
    simply not building that shape)."""
    env = dict(iters=2, pop=6, elite=2, k_fit=1, n_eval=4, n_stats=2,
               horizon=24)
    m = _experiment(0, **env)
    c = _control(0, **env)
    print("smoke experiment:", json.dumps(m, indent=1, default=float))
    print("smoke control:", json.dumps(c, indent=1, default=float))
    # The check must run on smoke output (aggregate shape: single seed).
    print("smoke check path:", _check({**m, "gain_std": 1.0}, c))


def _pilot():
    """Seed 90, full envelope, JSON to stdout — no ledger write."""
    t0 = time.time()
    m = _experiment(90)
    c = _control(90)
    print(json.dumps({"seed": 90, "experiment": m, "control": c,
                      "pilot_wall_s": time.time() - t0}, default=float))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        _pilot()
    else:
        print(run().status)
