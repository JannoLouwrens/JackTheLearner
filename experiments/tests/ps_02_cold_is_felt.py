"""PS.02 — The world can freeze him, and the cold is FELT before it kills.

HYPOTHESIS (registry, unchanged). The world carries a temperature field with
pre-registered dynamics — body temperature falls at a measured rate in cold,
rises near heat, death below a threshold within a bounded time — AND the
approach of that death is legible from Jack's senses beforehand: a probe on his
sensory vector predicts time-to-freezing well above chance while he is still
alive.

WHY THE SECOND HALF IS THE WHOLE SPEC. A lethal need you cannot perceive is not
a curriculum, it is noise. Building a world where cold kills is ten lines; the
question that decides whether `thermal (kills)` can ever be a commitment this
project keeps is whether an agent standing in that world has any way to know it
is coming. If time-to-freezing is unpredictable from the senses, then no policy
and no architecture could ever adapt to it, and every shelter result built on
top (SH.01, and the jungle's entire motive for building anything) would be
measuring luck. So the headline number is a probe R^2, not a death count.

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The probe reads the clock.** Every episode gets colder with time, so a
   feature that tracks elapsed time within a run — `needs`' energy drains
   monotonically — predicts "time remaining" without sensing anything thermal.
   Caught by the registry's declared control, SILENT LETHALITY: the body cools
   and dies on exactly the same schedule, and the thermal channel is deleted
   from the sensory vector. Same runs, same trajectories, same 80 other
   features; only the 2 thermal floats are gone. It must fail.

   The control is also what shaped the WORLD, and that is the right order.
   Newton's law of cooling — the obvious choice — drives the body to an
   asymptote and squeezes every run's lifetime into a narrow band, at which
   point "the mean lifetime minus elapsed time" is a good predictor and the
   control could not have failed however honest the probe was. `thermal.py`
   uses a linear law so that time-to-freezing is a ratio of two per-run
   quantities (how much heat is left over how fast it is leaving), which no
   clock reconstructs. A control that cannot fail is not a control.

2. **The probe reads the fire through the eyes.** A visible warm patch is a
   patch `vision`'s 16 rays can find. `thermal.py` puts no geom at `fire_xy` —
   warmth at distance is sensed thermally or not at all — so the only route
   from the heat source into the feature matrix is the channel under test.

3. **The world is not actually lethal, or is lethal instantly.** Gated in both
   directions: every cold run must die, and the death must land inside
   [DEATH_S_MIN, DEATH_S_MAX] — a world that kills in two seconds is as
   unlearnable as one that never kills. The registry's null (a thermally inert
   world, equation (1) times zero, everything else identical) must produce zero
   deaths, and the shuffled probe pairing must collapse.

## THE PROBE IS GENERIC ON PURPOSE

300 random Fourier features and a ridge, drawn from one fixed seed and applied
IDENTICALLY to the experiment, the control and the shuffled null. The closed
form for time-to-freezing is a ratio, so a linear probe would understate the
legibility of the world and a probe hand-built around that ratio would overstate
it — the first would make the spec fail for the reader's reasons rather than the
world's, and the second would be measuring `thermal.py`'s arithmetic through a
detour. A generic nonlinear regressor asks the question the hypothesis asks: is
the information THERE.

Held out by RUN, never by row. Rows inside one life are heavily autocorrelated
— consecutive decisions differ by 0.2 s of a slow body — so a random row split
would put near-duplicates of every training row in the test set and report a
memorisation score as a generalisation score.

## THE LAW IS CHECKED AGAINST ITS OWN CLOSED FORM

`law_dev` steps equation (1) at the simulation's own `dt` and compares the
crossing time against `thermal.time_to_lethal_s`. It is a tripwire, not a
discrimination (LESSONS.md: a gate that re-derives the module's own formula
proves only that nobody broke the integrator) — which is why it is one clause of
six and not the claim.

PILOT: seed 90, disjoint from the registered seeds 0/1/2, as PG.6 and SM.01 did.
Gates were set with margin after the pilot, and the pilot numbers are in
`docs/LOOP_JOURNAL.md` under this spec's pre-registration.
"""
from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from .. import thermal                                    # noqa: E402
from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                              # noqa: E402
from ..w0 import SIM_S_PER_DECISION, W0                   # noqa: E402

# The claim is about the WORLD, so it goes stale when the world moves.
IMPL_DEPS = ["experiments/thermal.py", "experiments/w0.py", "playground.py"]

# ── the runs ────────────────────────────────────────────────────────────
N_TRAIN = 10                 # cold runs the probe fits on
N_TEST = 6                   # cold runs it is scored on — held out by RUN
N_WARM = 3                   # runs started AT the fire: "rises near heat"
N_INERT = 3                  # the registry's null: equation (1) times zero
HORIZON = 400                # decisions = 80 s; longer than any lawful death
WARM_DECISIONS = 100         # 20 s is enough for +0.25 degC/s to be unarguable
BLIND_CHECK_DECISIONS = 20   # `blind` must actually drop the channel

# ── pre-registered gates ────────────────────────────────────────────────
DEATH_S_MIN = 3.0            # faster than this and no policy could react
DEATH_S_MAX = 70.0           # slower and the death is outside the observation
LAW_DEV_MAX = 0.02           # integrator vs closed form, relative
WARM_DELTA_MIN = 2.0         # degC gained in 20 s at the fire
WARM_DIST_MAX = 1.0          # m; "near heat" is measured, not intended
PROBE_R2_MIN = 0.50          # the headline: legible while still alive
SHUFFLED_R2_MAX = 0.05       # the declared null pairing
CONTROL_R2_MAX = 0.20        # SILENT LETHALITY must fail...
CONTROL_MARGIN_MIN = 0.35    # ...and fail by this much, not by rounding

# ── the probe ───────────────────────────────────────────────────────────
N_RFF = 300
RFF_SEED = 20260812
RIDGE_LAMBDA = 1.0

# Feature layout. `language` is W0.DROPPED — present as an input CONDITION and
# zero as data — so it is excluded rather than fed 32 zeros to a regressor.
FEATURE_KEYS = ("vision", "audio", "touch", "proprio", "needs", "placebo")

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's `j0`/`alpha`, or a refusal. W0 has no defaults for them."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _features(obs: dict) -> np.ndarray:
    """The sensory vector, thermal LAST so the control is a clean suffix drop."""
    parts = [np.asarray(obs[k], dtype=np.float64) for k in FEATURE_KEYS]
    if "thermal" in obs:
        parts.append(np.asarray(obs["thermal"], dtype=np.float64))
    return np.concatenate(parts)


def _run_life(world_seed: int, *, j0: float, alpha: float, inert: bool = False,
              fire_dist=None, horizon: int = HORIZON,
              action_scale: float = 1.0) -> dict:
    """One life in the cold. Random actions — this spec is about the WORLD.

    A trained policy here would confound the question: "is time-to-freezing
    legible" must not depend on a controller nobody has certified yet, and a
    random walk is the weakest possible reader of the world. If the signal is
    there under random behaviour it is there.

    `action_scale=0.0` is the WARM FIXTURE and nothing else: a body resting at
    the fire. The pilot found the random walk carries the rover out of the
    1.5 m warm zone within 20 s and it then cools like everything else, which
    is a true fact about a random policy and no fact at all about heat. "Rises
    near heat" is a claim about the field, so it is measured where the field is
    warm — and `warm_mean_dist_m` is reported and gated so that "near" is a
    measurement rather than an intention.
    """
    w0 = W0(seed=world_seed, j0=j0, alpha=alpha, lethal=False)
    tw = thermal.ThermalWorld(w0, seed=world_seed, inert=inert,
                              fire_dist=fire_dist)
    rng = np.random.RandomState(world_seed * 31 + 5)
    rows, tb0, dists = [], tw.state.tb, []
    t_eff0 = tw.state.t_eff
    death_s = float("nan")
    for k in range(horizon):
        rows.append(_features(tw.observe()))
        xy = tw._xy()
        dists.append(math.hypot(xy[0] - tw.state.fire_xy[0],
                                xy[1] - tw.state.fire_xy[1]))
        a = rng.uniform(-1.0, 1.0, w0.action_dim) * action_scale
        tw.decide(a, SIM_S_PER_DECISION)
        if tw.frozen:
            death_s = (k + 1) * SIM_S_PER_DECISION
            break
    X = np.asarray(rows, dtype=np.float64)
    n = X.shape[0]
    # Target: seconds of life remaining, measured from the actual death. A run
    # that did not die contributes no target (and the gates below refuse a cold
    # world in which that happens at all).
    y = (np.arange(n, 0, -1) * SIM_S_PER_DECISION) if np.isfinite(death_s) \
        else np.full(n, np.nan)
    return {"X": X, "y": y, "death_s": death_s, "tb0": tb0, "t_eff0": t_eff0,
            "tb_end": tw.state.tb, "n": n, "mean_dist_m": float(np.mean(dists)),
            "predicted_s": thermal.time_to_lethal_s(tb0, t_eff0)}


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control reuses it.

    `run_spec` calls the experiment and the control once each per seed, and the
    control is the SAME WORLD observed without one channel — re-simulating it
    would risk the two arms differing by something other than the sense, which
    is the only difference the spec is allowed to have.
    """
    if seed in _CACHE:
        return _CACHE[seed]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]

    base = seed * 100 + 1        # small: W0 multiplies its seed into a 32-bit rng
    cold = [_run_life(base + i, j0=j0, alpha=alpha)
            for i in range(N_TRAIN + N_TEST)]
    warm = [_run_life(base + 50 + i, j0=j0, alpha=alpha, fire_dist=0.0,
                      horizon=WARM_DECISIONS, action_scale=0.0)
            for i in range(N_WARM)]
    inert = [_run_life(base + 70 + i, j0=j0, alpha=alpha, inert=True)
             for i in range(N_INERT)]

    # `blind` is the control's mechanism, so it is EXERCISED, not trusted: the
    # blind observer must return exactly the sighted vector minus its last two
    # columns, on the same world, decision for decision. A flag nothing reads
    # is a comment, not a signal (LESSONS.md).
    w0a = W0(seed=base, j0=j0, alpha=alpha, lethal=False)
    w0b = W0(seed=base, j0=j0, alpha=alpha, lethal=False)
    ta = thermal.ThermalWorld(w0a, seed=base)
    tb = thermal.ThermalWorld(w0b, seed=base, blind=True)
    rng = np.random.RandomState(base * 31 + 5)
    blind_dev, blind_width_ok = 0.0, True
    for _ in range(BLIND_CHECK_DECISIONS):
        fa, fb = _features(ta.observe()), _features(tb.observe())
        blind_width_ok &= (fa.shape[0] == fb.shape[0] + thermal.THERMAL_DIM)
        blind_dev = max(blind_dev, float(np.max(np.abs(fa[:fb.shape[0]] - fb))))
        a = rng.uniform(-1.0, 1.0, w0a.action_dim)
        ta.decide(a, SIM_S_PER_DECISION)
        tb.decide(a, SIM_S_PER_DECISION)

    out = {"cold": cold, "warm": warm, "inert": inert, "prov": prov,
           "blind_dev": blind_dev, "blind_width_ok": float(blind_width_ok)}
    _CACHE[seed] = out
    return out


# ── the probe: random Fourier features + ridge, one fixed draw ──────────
def _fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.RandomState(RFF_SEED)
    d = Ztr.shape[1]
    W = rng.randn(d, N_RFF) / math.sqrt(d)
    b = rng.uniform(0.0, 2.0 * math.pi, N_RFF)
    Ptr = np.cos(Ztr @ W + b) * math.sqrt(2.0 / N_RFF)
    Pte = np.cos(Zte @ W + b) * math.sqrt(2.0 / N_RFF)
    ybar = ytr.mean()
    A = Ptr.T @ Ptr + RIDGE_LAMBDA * np.eye(N_RFF)
    beta = np.linalg.solve(A, Ptr.T @ (ytr - ybar))
    return Pte @ beta + ybar


def _r2(y, yhat) -> float:
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _score(cold: list, n_cols: int, shuffle: bool = False) -> float:
    """Held-out-by-RUN R^2 for the first `n_cols` features."""
    tr = cold[:N_TRAIN]
    te = cold[N_TRAIN:]
    Xtr = np.concatenate([r["X"][:, :n_cols] for r in tr])
    ytr = np.concatenate([r["y"] for r in tr])
    Xte = np.concatenate([r["X"][:, :n_cols] for r in te])
    yte = np.concatenate([r["y"] for r in te])
    if shuffle:
        ytr = np.random.RandomState(RFF_SEED + 1).permutation(ytr)
    return _r2(yte, _fit_predict(Xtr, ytr, Xte))


def _law_dev() -> float:
    """The integrator against `thermal.time_to_lethal_s`, at the sim's own dt."""
    worst = 0.0
    for tb0 in (30.5, 34.0, 37.5):
        for t_eff in (-18.0, -5.0, 5.0):
            want = thermal.time_to_lethal_s(tb0, t_eff)
            tb, t = tb0, 0.0
            while tb > thermal.TB_LETHAL and t < 1000.0:
                tb += thermal.drift_per_s(t_eff) * SIM_S_PER_DECISION
                t += SIM_S_PER_DECISION
            worst = max(worst, abs(t - want) / want)
    return worst


def _experiment(seed: int) -> dict:
    d = _collect(seed)
    if "refused" in d:
        return {"borrow_ok": 0.0, **{k: float(v) if isinstance(v, (int, float))
                                     else 0.0 for k, v in d["refused"].items()
                                     if k != "borrow_refusal"}}
    cold, warm, inert = d["cold"], d["warm"], d["inert"]
    deaths = np.array([r["death_s"] for r in cold], dtype=float)
    n_cols = cold[0]["X"].shape[1]

    r2 = _score(cold, n_cols)
    r2_shuf = _score(cold, n_cols, shuffle=True)

    warm_delta = float(np.mean([r["tb_end"] - r["tb0"] for r in warm]))
    m = {
        "borrow_ok": 1.0,
        # (3) the world is lethal, and boundedly so
        "cold_deaths": float(np.sum(np.isfinite(deaths))),
        "cold_runs": float(len(cold)),
        "all_cold_died": float(bool(np.all(np.isfinite(deaths)))),
        "death_s_min": float(np.nanmin(deaths)),
        "death_s_max": float(np.nanmax(deaths)),
        "death_s_mean": float(np.nanmean(deaths)),
        "death_s_spread": float(np.nanmax(deaths) / max(np.nanmin(deaths), 1e-9)),
        # (1) the law, and the integrator that runs it
        "law_dev": _law_dev(),
        "predicted_vs_observed_dev": float(np.nanmean(
            [abs(r["predicted_s"] - r["death_s"]) / r["death_s"] for r in cold])),
        # "rises near heat"
        "warm_delta_c": warm_delta,
        "warm_mean_dist_m": float(np.mean([r["mean_dist_m"] for r in warm])),
        "warm_deaths": float(sum(np.isfinite(r["death_s"]) for r in warm)),
        # the registry's null: a thermally inert world
        "inert_deaths": float(sum(np.isfinite(r["death_s"]) for r in inert)),
        "inert_tb_spread": float(np.max([abs(r["tb_end"] - r["tb0"])
                                         for r in inert])),
        # the headline
        "probe_r2": r2,
        "shuffled_r2": r2_shuf,
        "n_features": float(n_cols),
        "n_rows": float(sum(r["n"] for r in cold)),
        "blind_dev": d["blind_dev"],
        "blind_width_ok": d["blind_width_ok"],
    }
    m["seed_gates_ok"] = float(
        m["all_cold_died"] == 1.0
        and DEATH_S_MIN <= m["death_s_min"] and m["death_s_max"] <= DEATH_S_MAX
        and m["law_dev"] <= LAW_DEV_MAX
        and m["warm_delta_c"] >= WARM_DELTA_MIN and m["warm_deaths"] == 0.0
        and m["warm_mean_dist_m"] <= WARM_DIST_MAX
        and m["inert_deaths"] == 0.0 and m["inert_tb_spread"] == 0.0
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX
        and m["blind_width_ok"] == 1.0 and m["blind_dev"] == 0.0)
    return m


def _control(seed: int) -> dict:
    """SILENT LETHALITY: the same lives, the thermal channel deleted.

    Not a re-simulation and not a different world — the identical feature
    matrix with `thermal.THERMAL_DIM` columns dropped, which is exactly what
    `ThermalWorld(blind=True)` observes (asserted decision-by-decision in
    `_collect`). He still freezes on the same schedule; he simply cannot feel
    it. If the probe still works, it was reading the clock.
    """
    d = _collect(seed)
    if "refused" in d:
        return {"control_r2": 0.0, "control_caught": 0.0}
    cold = d["cold"]
    n_cols = cold[0]["X"].shape[1] - thermal.THERMAL_DIM
    r2c = _score(cold, n_cols)
    return {"control_r2": r2c,
            "control_n_features": float(n_cols),
            "control_caught": float(r2c <= CONTROL_R2_MAX)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing. VOID, never FAIL (T0.22).
        return Status.VOID
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX
        and c["control_caught"] == 1.0
        and c["control_r2"] <= CONTROL_R2_MAX
        and (m["probe_r2"] - c["control_r2"]) >= CONTROL_MARGIN_MIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
