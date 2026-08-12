"""BA.01 — He feels himself falling before he falls.

HYPOTHESIS (registry, unchanged). Jack carries a sensed orientation signal —
gravity's direction in his own body frame — from which a linear probe recovers
tilt, and from which time-to-topple is predictable while he is still upright.

THE BODY IS AN INVERTED PENDULUM AND THAT IS THE POINT. W0.BAL (integration
queue, raised by LC.02's measurement 2026-08-09): a 30 kg capsule on a 0.09 m
spherical foot topples under random action within ~20 decisions and slides on
its side. This spec does not fix that — fixing it is a control problem the
LC bakeoff owns. It asks the question that must be answerable BEFORE any
balance controller could exist: while the body is still upright, do his senses
carry the fall that is coming? A creature that cannot feel falling cannot learn
to catch itself, however good its learning rule.

THE VESTIBULAR CHANNEL, REGISTERED AS BIOLOGY HANDS IT TO US (registry notes):
not one signal but two —

  otoliths  linear acceleration: specific force in the body frame,
            f = R^T (a_world - g). At rest it reads 9.81 body-up; in free
            fall it reads zero. This is what distinguishes FALLING from
            BEING CARRIED — a body moved at constant velocity feels rest.
  canals    angular velocity in the body (local) frame, straight from the
            free joint's rotational dofs.

plus gravity's direction in the body frame (the static orientation signal the
hypothesis names) and the root's horizontal velocity. Root vx/vy belong on the
balance side deliberately: on a pivoting foot the origin's horizontal velocity
is tilt rate times height — a control that kept them would read the fall
through kinematic coupling and the spec would be measuring bookkeeping.

TOUCH IS A BALANCE ORGAN TOO, AND THE PILOT PROVED IT. The seed-90 pilot put
the eight touch floats on the blind side and they alone scored AUC 0.94: the
foot's normal force unloads as the fall accelerates (N ~ m(g - a_z)), which is
plantar graviception — the same somatosensory balance contribution biology
gives the sole of the foot. A "control" keeping them would not be reading a
clock, it would be reading a second balance organ. So the ORIENTATION CHANNEL
this spec registers — and its control deletes — is the whole graviceptive
suffix: touch + vestibular, with the vestibular-proper numbers (auc_vest,
auc_grav, auc_canals, auc_otoliths, auc_touch) reported separately so the
claim's anatomy is visible and no single organ hides behind the block.

THE BLIND SIDE is what remains and it is measured blind, not assumed blind:
the four arm slide positions and their velocities read tilt at R^2 = 0.04 in
the pilot. Vision is excluded from BOTH sides: the rays rotate with the body,
so a tilted body sees tilted distances — a feature that smuggles orientation
in through the eyes would make the control unfailable. Needs are excluded
because they drain monotonically with time — feeding the probe a clock while
gating against a clock-reading control would be arming both sides of the same
test. The headline probe runs on the VESTIBULAR block itself, because the
hypothesis says "FROM WHICH": the claim is that the orientation signal carries
the fall, not that something in a wider bag of floats does (the pilot measured
the dilution directly: vestibular-alone 0.95 AUC, vestibular+touch 0.82,
all 27 floats 0.81 — a generic kernel probe pays for every uninformative
dimension it is handed). Touch is deleted ALONGSIDE the vestibular block in
the control and reported separately in the metrics.

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The probe reads the episode clock** — falls cluster at a characteristic
   time after spawn, so elapsed time alone predicts "topple soon". Caught
   twice, by design: the declared control deletes the balance suffix from the
   SAME rollouts (physics identical, decision for decision) and must fail; and
   the null predictor is the SAME probe machinery given elapsed time as its
   only feature — the headline must beat it by a pre-registered margin, not by
   rounding.

2. **The rig makes every episode identical** — if every spawn topples on the
   same schedule (the zero-perturbation pilot measured exactly this: passive
   topple at ~10 decisions on most spawns), time-to-topple degenerates into
   the clock and the control could not have failed however honest the probe.
   So each episode draws a pre-registered LOG-uniform initial tilt (fall
   time goes as log(1/theta), so only a log draw spreads it) plus an
   angular-velocity kick, and respawns to a fresh site; TF_SPREAD_MIN gates
   that the spread actually happened, and rows are only eligible while
   upright cosine >= UPRIGHT_ROW — the claim is about feeling the fall
   EARLY, not about reading a body already at 45 degrees.

3. **The task is unscoreable and reports FAIL anyway** — too few topples, too
   few eligible rows, or a test set with one class. Those are rig failures,
   not refutations: they return Status.VOID (the T2.02 lesson — only a run
   that tested the claim may say FAIL).

## WHAT FAIL WOULD MEAN

FAIL is reserved for the sense failing: tilt unrecoverable by a linear probe,
the AUC no better than the clock, or the control passing (the "signal" was
never the balance channel). Then balance is not a sense he has, it is an
outcome he suffers — and every climbing and locomotion claim that assumes he
can tell up from down inherits that hole.

PILOT: seed 90, disjoint from the registered seeds 0/1/2 (PS.02, PG.6, SM.01
precedent). Gates were set with margin after the pilot; the pilot numbers are
pre-registered in docs/LOOP_JOURNAL.md under this spec before the recorded run.
"""
from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from ..w0 import W0, SIM_S_PER_DECISION                           # noqa: E402

# The claim is about the world's body and its senses, so it goes stale when
# either moves.
IMPL_DEPS = ["experiments/w0.py", "playground.py"]

# ── the rollouts ────────────────────────────────────────────────────────
# The rig was shaped by three pilot measurements, not by intention (the PS.02
# order: the control shaped the world). (1) A uniform tilt draw gave median
# topple at 5 decisions with std 1.5 — pendulum fall time goes as log(1/theta),
# so uniform theta is a clock and the AUC was unscoreable (7 negative rows).
# (2) The tilt floor is structural: from EXACT upright with zero velocity the
# contact solver injects ~0.8 deg within one decision, so every free-standing
# spawn falls by ~13 decisions regardless of how small the draw is. Fall times
# are therefore spread by drawing theta LOG-uniformly above that floor, and by
# a strong angular kick whose direction matters (a kick against the tilt swings
# the body through upright and buys it decisions — an energy state the
# vestibular channel reads and the clock cannot). (3) Spawn sites vary: some
# poses lean on structure and never topple, which is where the honest
# late-time negative rows come from — so every episode respawns.
N_EP_TRAIN = 84              # episodes the probes fit on
N_EP_TEST = 36               # episodes they are scored on — held out by EPISODE
HORIZON = 80                 # decisions = 16 s; passive topple is far inside
TILT0_LOG10_DEG = (-1.0, 1.15)   # theta ~ 10^U[...]: 0.1 to 14 deg
OMEGA0_STD = 0.3             # rad/s, initial angular-velocity kick per axis
ARM_NOISE = 0.3              # slide actions ~ U[-1,1] * this; drive+adhesion 0
GRAVITY = 9.81

# ── labels ──────────────────────────────────────────────────────────────
TOPPLE_UP = 0.5              # upright cosine below this = toppled (60 deg)
UPRIGHT_ROW = 0.9            # rows eligible for the AUC while above this
W_WARN = 5                   # decisions: "topples within the next 1.0 s"

# ── the probes ──────────────────────────────────────────────────────────
N_RFF = 300                  # PS.02's generic probe, one fixed draw
RFF_SEED = 20260812
RIDGE_LAMBDA = 1.0
VEST_DIM = 11                # grav_body 3 + canals 3 + otoliths 3 + vx,vy
GRAV_DIM = 8 + VEST_DIM      # the orientation channel: touch + vestibular

# ── pre-registered gates (set with margin after the seed-90 pilot;
#    pilot numbers recorded beside each gate and in LOOP_JOURNAL.md) ─────
TOPPLED_FRAC_MIN = 0.60      # a world with nothing falling tests nothing (VOID)
TF_SPREAD_MIN = 2.5          # decisions, std of topple times (pilot 5.69)
MIN_CLASS_ROWS = 25          # test rows per class, else unscoreable (VOID)
TILT_R2_MIN = 0.90           # linear probe recovers tilt-cosine (pilot 0.998)
TILT_SHUF_R2_MAX = 0.05      # shuffled pairing must collapse (pilot < 0)
TILT_CONTROL_R2_MAX = 0.30   # arm slides must not recover tilt (pilot 0.04)
AUC_MIN = 0.85               # the headline (pilot 0.95)
AUC_TIME_MARGIN_MIN = 0.10   # headline minus elapsed-time null (pilot ~0.23)
CONTROL_AUC_MAX = 0.70       # blind AUC cap (pilot 0.64)
CONTROL_MARGIN_MIN = 0.15    # headline minus blind AUC (pilot ~0.31)

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's j0/alpha, or a refusal. W0 has no defaults for them."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _tilt_quat(rng: np.random.RandomState) -> np.ndarray:
    theta = math.radians(10.0 ** rng.uniform(*TILT0_LOG10_DEG))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0),
                     math.cos(phi) * s, math.sin(phi) * s, 0.0])


def _episode(w: W0, rng: np.random.RandomState) -> dict:
    """One perturbed passive episode: respawn, tilt, fall (or don't).

    Actions move ONLY the four arm slides. The drive dims are a world-frame
    force — a hidden push knob the blind features could read through its
    consequences — and adhesion glues hands; neither belongs in a passive
    balance measurement. Arm noise is kept so the blind block is a live signal
    the control probe genuinely gets to try: a control fed constants cannot
    fail meaningfully (PS.02: a control that cannot fail is not a control —
    the mirror rule: a control that cannot PASS proves nothing either).
    """
    mujoco = w.mujoco
    w.respawn()
    qa, da = w.ix["root_qposadr"], w.ix["root_dofadr"]
    # Tilt the root about the world frame, then kick its angular velocity.
    q0 = w.data.qpos[qa + 3:qa + 7].copy()
    qt = _tilt_quat(rng)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qt, q0)
    w.data.qpos[qa + 3:qa + 7] = out
    w.data.qvel[da:da + 6] = 0.0
    w.data.qvel[da + 3:da + 6] = rng.randn(3) * OMEGA0_STD
    mujoco.mj_forward(w.model, w.data)

    rows, uprights = [], []
    v_prev = w.data.qvel[da:da + 3].copy()
    t_f = None
    for t in range(HORIZON):
        xmat = w.data.xmat[w.rover_bid]
        R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        up = float(xmat[8])
        grav_body = -R.T @ np.array([0.0, 0.0, 1.0])       # gravity direction
        canals = w.data.qvel[da + 3:da + 6].copy()          # ang vel, body frame
        v = w.data.qvel[da:da + 3].copy()
        a_world = (v - v_prev) / SIM_S_PER_DECISION
        otoliths = R.T @ (a_world - np.array([0.0, 0.0, -GRAVITY]))
        v_prev = v
        p = w._proprio()          # 4 slide pos, 4 slide vel, z, upright, vx, vy
        touch = w._touch()
        blind = np.concatenate([p[:8], touch])
        balance = np.concatenate([grav_body, canals, otoliths, p[10:12]])
        assert balance.shape[0] == VEST_DIM
        rows.append(np.concatenate([blind, balance]))
        uprights.append(up)
        if up < TOPPLE_UP:
            t_f = t
            break
        act = np.zeros(8)
        act[:4] = rng.uniform(-1.0, 1.0, 4) * ARM_NOISE
        w.decide(act)
    return {"X": np.asarray(rows, dtype=np.float64),
            "upright": np.asarray(uprights, dtype=np.float64),
            "t_f": t_f}


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control reuses it.

    The control is the SAME rollouts observed without the balance suffix —
    re-simulating would risk the two arms differing by something other than
    the sense, which is the only difference the spec is allowed to have.
    """
    if seed in _CACHE:
        return _CACHE[seed]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False)
    rng = np.random.RandomState(seed * 7907 + 11)
    eps = [_episode(w, rng) for _ in range(N_EP_TRAIN + N_EP_TEST)]
    _CACHE[seed] = {"eps": eps, "prov": prov}
    return _CACHE[seed]


# ── probes: linear ridge (tilt), RFF ridge (topple), one fixed draw ─────
def _ridge_predict(Xtr, ytr, Xte, rff: bool) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if rff:
        rng = np.random.RandomState(RFF_SEED)
        d = Ztr.shape[1]
        W = rng.randn(d, N_RFF) / math.sqrt(d)
        b = rng.uniform(0.0, 2.0 * math.pi, N_RFF)
        Ztr = np.cos(Ztr @ W + b) * math.sqrt(2.0 / N_RFF)
        Zte = np.cos(Zte @ W + b) * math.sqrt(2.0 / N_RFF)
    ybar = ytr.mean()
    A = Ztr.T @ Ztr + RIDGE_LAMBDA * np.eye(Ztr.shape[1])
    beta = np.linalg.solve(A, Ztr.T @ (ytr - ybar))
    return Zte @ beta + ybar


def _r2(y, yhat) -> float:
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _auc(y, score) -> float:
    """Mann–Whitney rank AUC; nan when the test set has one class."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # midranks for ties
    allsc = np.concatenate([neg, pos])
    for v in np.unique(allsc):
        m = allsc == v
        ranks[m] = ranks[m].mean()
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _label_rows(ep: dict) -> tuple:
    """Eligible rows (still upright) with the 'topples within W_WARN' label.

    A surviving episode's last W_WARN rows are trimmed: they could belong to a
    topple just past the horizon, and a label that might be wrong is censored,
    not asserted (PS.02's nan-target lesson, one step earlier).
    """
    up = ep["upright"]
    n = len(up)
    t_f = ep["t_f"]
    last = n if t_f is not None else n - W_WARN
    rows, ys, ts = [], [], []
    for t in range(max(last, 0)):
        if up[t] < UPRIGHT_ROW:
            continue
        y = 1.0 if (t_f is not None and t_f - t <= W_WARN) else 0.0
        rows.append(ep["X"][t])
        ys.append(y)
        ts.append(t * SIM_S_PER_DECISION)
    return rows, ys, ts


def _stack(eps: list) -> tuple:
    X, y, t = [], [], []
    for ep in eps:
        r, yy, tt = _label_rows(ep)
        X.extend(r); y.extend(yy); t.extend(tt)
    return (np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(t, dtype=np.float64)[:, None])


def _tilt_sets(eps: list) -> tuple:
    """All not-yet-toppled rows with the tilt COSINE as the target.

    The hypothesis asks a LINEAR probe to recover tilt, and the upright cosine
    is tilt in monotone units — the same information, bijectively. The angle
    itself goes as sqrt(1 - u) exactly where the rows cluster (near upright),
    so a linear probe of the ANGLE fails for the reader's reasons rather than
    the sensor's: the seed-90 pilot measured R^2 0.195 on the angle and 0.998
    on the cosine from the same features. Chosen after the pilot, disclosed
    here and in the journal pre-registration.
    """
    X, y = [], []
    for ep in eps:
        X.extend(ep["X"])
        y.extend(ep["upright"])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _evaluate(seed: int, blind: bool) -> dict:
    """The headline probe reads the graviceptive suffix; the control reads
    the blind prefix. Same rollouts, one slice apart — the only difference
    the spec allows."""
    c = _collect(seed)
    if "refused" in c:
        return {"probe": "VOID", **{k: float("nan") for k in ("auc",)},
                **c["refused"]}
    eps = c["eps"]
    tr, te = eps[:N_EP_TRAIN], eps[N_EP_TRAIN:]
    sl_x = slice(None, -GRAV_DIM) if blind else slice(-VEST_DIM, None)

    t_fs = [ep["t_f"] for ep in eps if ep["t_f"] is not None]
    toppled_frac = len(t_fs) / len(eps)
    tf_spread = float(np.std(t_fs)) if t_fs else 0.0

    Xtr, ytr, ttr = _stack(tr)
    Xte, yte, tte = _stack(te)
    n_pos, n_neg = int(yte.sum()), int((1 - yte).sum())

    out = {"toppled_frac": toppled_frac, "tf_spread": tf_spread,
           "median_t_f": float(np.median(t_fs)) if t_fs else float("nan"),
           "n_rows_train": float(len(ytr)), "n_pos_test": float(n_pos),
           "n_neg_test": float(n_neg)}
    if n_pos < MIN_CLASS_ROWS or n_neg < MIN_CLASS_ROWS or len(ytr) < 100:
        out["probe"] = "VOID"
        out["auc"] = float("nan")
        return out

    out["auc"] = _auc(yte, _ridge_predict(Xtr[:, sl_x], ytr, Xte[:, sl_x],
                                          rff=True))

    # Tilt: the linear probe of the hypothesis, on all not-toppled rows.
    Ttr_X, Ttr_y = _tilt_sets(tr)
    Tte_X, Tte_y = _tilt_sets(te)
    out["tilt_r2"] = _r2(Tte_y, _ridge_predict(Ttr_X[:, sl_x], Ttr_y,
                                               Tte_X[:, sl_x], rff=False))

    if not blind:
        # The elapsed-time null: the same machinery, the clock as its only
        # feature. What survivorship makes predictable, it gets for free.
        out["auc_time"] = _auc(yte, _ridge_predict(ttr, ytr, tte, rff=True))
        # Shuffled tilt pairing (chance for tilt, the registry's null).
        sh = np.random.RandomState(RFF_SEED + 1).permutation(len(Ttr_y))
        out["tilt_r2_shuffled"] = _r2(
            Tte_y, _ridge_predict(Ttr_X[:, sl_x], Ttr_y[sh],
                                  Tte_X[:, sl_x], rff=False))
        # The organs of the graviceptive block, reported separately as the
        # registry demands — a system given only gravity's direction cannot
        # tell falling from being carried, and touch must not hide the
        # vestibular numbers (or vice versa). Reported, not gated.
        nb = Xtr.shape[1] - GRAV_DIM
        for name, sl in (("touch", slice(nb, nb + 8)),
                         ("vest", slice(nb + 8, None)),
                         ("grav", slice(nb + 8, nb + 11)),
                         ("canals", slice(nb + 11, nb + 14)),
                         ("otoliths", slice(nb + 14, nb + 17))):
            out[f"auc_{name}"] = _auc(
                yte, _ridge_predict(Xtr[:, sl], ytr, Xte[:, sl], rff=True))
        gates = (toppled_frac >= TOPPLED_FRAC_MIN
                 and tf_spread >= TF_SPREAD_MIN
                 and out["auc"] >= AUC_MIN
                 and out["auc"] - out["auc_time"] >= AUC_TIME_MARGIN_MIN
                 and out["tilt_r2"] >= TILT_R2_MIN)
        out["seed_gates_ok"] = 1.0 if gates else 0.0
    return out


def _experiment(seed: int) -> dict:
    return _evaluate(seed, blind=False)


def _control(seed: int) -> dict:
    """The declared control: the balance suffix deleted, physics identical."""
    return _evaluate(seed, blind=True)


def _declared_void(m: dict) -> bool:
    return m.get("probe") == "VOID" or not np.isfinite(m.get("auc", np.nan))


def _check(m: dict, c: dict):
    if _declared_void(m) or _declared_void(c):
        return Status.VOID
    ok = (m["seed_gates_ok"] == 1.0
          and m["toppled_frac"] >= TOPPLED_FRAC_MIN
          and m["tf_spread"] >= TF_SPREAD_MIN
          and m["tilt_r2"] >= TILT_R2_MIN
          and m["tilt_r2_shuffled"] <= TILT_SHUF_R2_MAX
          and m["auc"] >= AUC_MIN
          and m["auc"] - m["auc_time"] >= AUC_TIME_MARGIN_MIN
          # the control must FAIL, and by a margin, on both probes
          and c["tilt_r2"] <= TILT_CONTROL_R2_MAX
          and c["auc"] <= CONTROL_AUC_MAX
          and m["auc"] - c["auc"] >= CONTROL_MARGIN_MIN)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["BA.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    print(run().status)
