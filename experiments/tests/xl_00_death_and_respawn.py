"""XL.00 — he dies, he reappears somewhere he did not choose, and the diary crosses.

`LEARNING_CORE.md` §5.0 lists six things the learning-core bakeoff needs from
World Zero. Four of them shipped with `w0.py`. **W0-2 (death) and W0-3
(cross-life memory) did not** — the module's own header said "NOT YET" — and
LC.03 scores `life_gain` over `n_lives >= 12` and `cross_life_transfer`, so
until this spec passes the learning-core bakeoff is scoring two quantities that
do not exist. This certifies the mechanism, cheaply, before 20 core-hours are
spent on top of it.

WHAT IS ACTUALLY AT RISK HERE, and it is not "does the code run". §5.0 W0-2
requires that death is not a **free teleport to a good state**: the respawn must
be uniform over the legal set and blind to where he died. `LT` §2.1's objection
is that an episode boundary is an experimenter-supplied curriculum, and the
random respawn is the whole answer to it. A respawn that quietly correlated with
the death site — because the sampler saw it, because the legal set was carved
around it, because a fall always ends in the same corner — would make every
`life_gain` in LC.03/LC.04 partly a measurement of the reset rule. So the
independence test is the load-bearing half of this spec, and it carries a
positive control (a sampler that DOES respawn at the death site) which must
trip it. A detector that has never seen its own positive control has measured
nothing — T0.13, and it is why three of the five controls here are positive.

THE SHORT-LIFE FIXTURE, declared. A random policy on full charge lives ~300
simulated seconds; sixteen deaths per seed at three seeds is ~16 real minutes of
drain and nothing else. Every claim in this spec except the drain arithmetic is
INVARIANT to the starting charge — a spawn sampler, a legality predicate, a
diary and a life counter do not know how full the tank was — so lives after the
first start at `SHORT_E0`. The drain arithmetic is then certified separately and
more strictly, at two independent full charges, by requiring the IMPLIED 1/b to
agree with `drives.BASAL_B` and with itself: that checks the RATE, where one
death at e=1.0 checks a single endpoint.

`j0` and `alpha` are read from PS.01's LEDGER ENTRY. They are measurements, and
a measurement copied into a second file is a constant that drifts away from what
produced it (T0.14's 28 dead padded columns). If PS.01 is not PASS this spec is
VOID, not FAIL: an uncalibrated drive layer cannot refute anything.
"""
from __future__ import annotations

import math

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..w0 import (MIN_LEGAL_SPAWNS, SIM_S_PER_DECISION, W0, random_action,
                  uniform_legal_spawn)
# After `..w0`, deliberately: importing it is what puts the repo root on
# `sys.path`, and `EpisodicMemory` lives there rather than in the package.
from EpisodicMemory import EpisodicMemory   # noqa: E402

# The world and the drive layer are what this spec certifies, so both hash into
# `impl_sha`. LC.02 declares only `playground.py` and therefore could not see a
# change to `w0.py` — the module whose throughput it measures. Widened here for
# LC.02 too, in the same commit that changes `w0.py`, because a certificate
# whose scope grows retroactively invalidates every entry recorded under the
# narrow one.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py"]

# ── the pre-registered numbers, all of them, before the run ────────────────
N_DECISIONS = 3000            # 600 simulated seconds per condition. A life at
                              # SHORT_E0 measured 40.2 s in the smoke run, so
                              # this buys ~15 lives against MIN_LIVES=12 — the
                              # margin is deliberate: sizing the budget to land
                              # exactly on the floor makes the gate a coin toss
                              # on seed noise rather than a claim.
SHORT_E0 = 0.10               # the declared short-life fixture
MIN_LIVES = 12                # LC.03's own floor, so this certifies what it needs
STATUE_CHARGES = (0.2, 0.5)   # two independent drains of the resting body
BASAL_TOL = 0.02              # implied 1/b within 2% of 1/BASAL_B, and of itself
UNIFORM_DRAWS = 20_000        # chi-square draws from the sampler alone
UNIFORM_Z_MAX = 4.0           # |z| on the chi-square, normal approximation
INDEP_Z_MAX = 3.0             # |z| on paired-vs-shuffled death->spawn distance
TREND_Z_MAX = 3.0             # |z| on the non-learner's life-length slope
N_PERM = 2000                 # permutations for both null distributions
DRIFT_E0 = (0.05, 0.025)      # the drifting world: e0 = 0.05 + 0.025 * life


def _calibration() -> tuple:
    """(j0, alpha) as PS.01 measured them, or (None, None) if it has not passed."""
    entry = Ledger().results.get("PS.01")
    if entry is None or entry.status != Status.PASS:
        return None, None
    j0, alpha = entry.metrics.get("j0_ms"), entry.metrics.get("alpha")
    if j0 is None or alpha is None:
        return None, None
    return float(j0), float(alpha)


def _biased_sampler(legal, rng, death_xy):
    """Draws only from the half of the legal set nearest the origin.

    The positive control for the uniformity detector. It is still a legal spawn
    every time — which is the point: legality and uniformity are two claims, and
    a test that checked only the first would call this sampler correct.
    """
    r = np.hypot(legal[:, 0], legal[:, 1])
    near = np.argsort(r)[: len(legal) // 2]
    k = int(rng.randint(len(near)))
    return float(legal[near[k]][0]), float(legal[near[k]][1])


def _at_death_sampler(legal, rng, death_xy):
    """Respawns exactly where he died. The positive control for independence."""
    return float(death_xy[0]), float(death_xy[1])


# ── statistics, both permutation-based, both computed from the run's own draws ─
def _perm_z(paired: float, null: np.ndarray) -> float:
    sd = float(null.std())
    if sd == 0.0:
        return float("nan")            # a null with no spread cannot detect
    return float((float(null.mean()) - paired) / sd)


def _independence_z(deaths: np.ndarray, spawns: np.ndarray, seed: int) -> tuple:
    """Is the spawn closer to the death site than a shuffled pairing would be?

    The statistic is the MEAN death->spawn distance. Under independence the
    paired value is a draw from the shuffled distribution; under a leak it is
    smaller. Reported as a z, positive meaning "closer than chance".
    """
    if len(deaths) < 3:
        return float("nan"), float("nan")
    paired = float(np.linalg.norm(spawns - deaths, axis=1).mean())
    rng = np.random.RandomState(seed * 6011 + 17)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        null[b] = np.linalg.norm(spawns[rng.permutation(len(spawns))] - deaths,
                                 axis=1).mean()
    return _perm_z(paired, null), paired


def _slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _trend_z(lengths: np.ndarray, seed: int) -> tuple:
    """Do lives lengthen across the run, beyond what shuffling them produces?

    Gated against a PERMUTATION null rather than a fixed slope, because a life
    length here is not a low-variance quantity: one accidental apple is worth
    150 simulated seconds under the short fixture, so a fixed threshold would
    read variance as a trend. The null is the same lives in a shuffled order.
    """
    if len(lengths) < 4:
        return float("nan"), float("nan")
    s = _slope(lengths)
    rng = np.random.RandomState(seed * 7717 + 31)
    null = np.array([_slope(lengths[rng.permutation(len(lengths))])
                     for _ in range(N_PERM)])
    sd = float(null.std())
    if sd == 0.0:
        return float("nan"), s
    return float((s - float(null.mean())) / sd), s


def _uniformity_z(sampler, legal: np.ndarray, seed: int) -> float:
    """Chi-square of `UNIFORM_DRAWS` draws against the flat multinomial.

    Normal approximation (z = (chi2 - df) / sqrt(2 df)) rather than a p-value:
    df is ~600 here, where the approximation is excellent, and it avoids a scipy
    dependency for one number. The SAMPLER is exercised directly — this claim is
    about the draw, not about the simulation, and 20,000 draws cost nothing
    while 20,000 deaths would cost days.
    """
    rng = np.random.RandomState(seed * 104729 + 7)
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(legal)}
    counts = np.zeros(len(legal))
    for _ in range(UNIFORM_DRAWS):
        counts[index[sampler(legal, rng, (0.0, 0.0))]] += 1.0
    exp = UNIFORM_DRAWS / len(legal)
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    df = len(legal) - 1
    return float((chi2 - df) / math.sqrt(2.0 * df))


# ── the conditions ─────────────────────────────────────────────────────────
def _live(seed: int, *, j0: float, alpha: float, lethal: bool = True,
          sampler=None, wipe_diary: bool = False, drift: bool = False) -> dict:
    """Run `N_DECISIONS` of uniform-random action under one condition."""
    diary = EpisodicMemory()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=lethal, diary=diary,
           spawn_sampler=sampler or uniform_legal_spawn)
    rng = np.random.RandomState(seed * 31 + 3)

    def charge(life: int) -> float:
        if drift:
            return min(1.0, DRIFT_E0[0] + DRIFT_E0[1] * life)
        return SHORT_E0

    w.drives.state = drives.DriveState(e=charge(0))
    for _ in range(N_DECISIONS):
        w.decide(random_action(rng))
        if w.died_this_decision:
            # The fixture, applied from OUTSIDE the world: `respawn()` has
            # already put a full body at the setpoint, exactly as it will in
            # LC.03. Nothing in `w0.py` knows this spec exists.
            w.drives.state = drives.DriveState(e=charge(w.life))
            if wipe_diary:
                diary.events.clear()
                diary._tok.clear()

    lives = np.asarray(w.life_lengths, dtype=float)
    deaths = np.asarray(w.death_sites, dtype=float).reshape(-1, 2)
    spawns = np.asarray(w.spawn_sites, dtype=float).reshape(-1, 2)

    legal = w.legal_spawns()
    legal_set = {(round(x, 6), round(y, 6)) for x, y in legal}
    n_legal_spawn = sum(1 for x, y in spawns
                        if (round(x, 6), round(y, 6)) in legal_set)

    indep_z, paired = _independence_z(deaths, spawns, seed)
    trend_z, slope = _trend_z(lives, seed)

    # W0-3. Read from the store, not from a recall score: "the rows are still
    # there, indexed by life" and "retrieval reaches across a death" are two
    # different claims and the aggregate would hide which one carried (ME.10's
    # rule, applied one level down).
    rows = [e for e in diary.events if e.channel == "did"]
    seen_lives = {int(e.meta.get("life", -1)) for e in rows}
    hits = diary.recall("life ended energy gone", channel="did", top_k=5)
    crossed = any(int(h.event.meta.get("life", -1)) < w.life - 1 for h in hits)

    finite = bool(np.all(np.isfinite(w.data.qpos))
                  and np.all(np.isfinite(w.data.qvel)))
    return {
        "n_lives": float(len(lives)),
        "deaths": float(w.deaths),
        "mean_life_s": float(lives.mean()) if len(lives) else 0.0,
        "death_by_energy": float(sum(1 for e in rows
                                     if e.meta.get("cause") == "energy")),
        "death_by_integrity": float(sum(1 for e in rows
                                        if e.meta.get("cause") == "integrity")),
        "spawn_legal_frac": (float(n_legal_spawn / len(spawns))
                             if len(spawns) else 0.0),
        "indep_z": indep_z,
        "paired_death_spawn_dist": paired,
        "trend_z": trend_z,
        "life_slope_s_per_life": slope,
        "life_drift_frac": (abs(slope) * max(0, len(lives) - 1)
                            / lives.mean() if len(lives) else 0.0),
        "diary_life0_rows": float(any(int(e.meta.get("life", -1)) == 0
                                      for e in rows)),
        "diary_life_index_covers": float(
            bool(rows) and seen_lives == set(range(w.life))),
        "diary_recall_crosses_death": float(bool(crossed)),
        "diary_rows": float(len(rows)),
        "n_legal": float(len(legal)),
        "physics_finite": float(finite),
        "sim_seconds": float(w.sim_seconds),
    }


def _statue_1_over_b(seed: int, e0: float, *, j0: float, alpha: float) -> float:
    """Sim-seconds a RESTING body survives, divided by its starting charge.

    Under the zero action the arms hold their mid-position and mechanical power
    is ~0, so §2.2's drain reduces to `BASAL_B` alone and the implied 1/b must
    come out at 600 s from any starting charge. Two charges, so this checks the
    RATE and not one endpoint.
    """
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    w.drives.state = drives.DriveState(e=e0)
    a = np.zeros(w.action_dim)
    cap = int(2.0 * e0 / drives.BASAL_B / SIM_S_PER_DECISION)
    while w.deaths == 0 and w.decisions < cap:
        w.decide(a)
    if not w.life_lengths:
        return float("nan")
    return float(w.life_lengths[0] / e0)


def _experiment(seed: int) -> dict:
    j0, alpha = _calibration()
    m: dict = {"calibrated": float(j0 is not None),
               "j0": float(j0 or 0.0), "alpha": float(alpha or 0.0),
               "min_lives": float(MIN_LIVES), "short_e0": SHORT_E0}
    if j0 is None:
        return m

    # ── the drain arithmetic, at two charges ────────────────────────────
    implied = [_statue_1_over_b(seed, e0, j0=j0, alpha=alpha)
               for e0 in STATUE_CHARGES]
    target = 1.0 / drives.BASAL_B
    m["statue_1_over_b_lo"], m["statue_1_over_b_hi"] = implied
    m["statue_1_over_b_target"] = float(target)
    m["statue_rate_agrees"] = float(
        all(abs(v - target) <= BASAL_TOL * target for v in implied)
        and abs(implied[0] - implied[1]) <= BASAL_TOL * target)

    # ── the legality predicate, known-answer both ways ──────────────────
    import playground as pg
    probe = W0(seed=seed, j0=j0, alpha=alpha)
    probe._place(pg.LADDER_X, pg.LADDER_Y)
    probe.mujoco.mj_forward(probe.model, probe.data)
    m["ladder_pose_rejected"] = float(probe._penetrating())
    a = float(probe.params.arena_size) - 0.75
    probe._place(a, a)
    probe.mujoco.mj_forward(probe.model, probe.data)
    m["corner_pose_accepted"] = float(not probe._penetrating())
    legal = probe.legal_spawns()
    m["legal_spawns"] = float(len(legal))
    m["legal_spawns_floor"] = float(MIN_LEGAL_SPAWNS)

    # ── the sampler, exercised directly ─────────────────────────────────
    m["uniform_z"] = _uniformity_z(uniform_legal_spawn, legal, seed)

    # ── the lives ───────────────────────────────────────────────────────
    m.update(_live(seed, j0=j0, alpha=alpha))
    m["conjunction"] = float(
        m["statue_rate_agrees"] == 1.0
        and m["ladder_pose_rejected"] == 1.0 and m["corner_pose_accepted"] == 1.0
        and m["n_lives"] >= MIN_LIVES
        and m["spawn_legal_frac"] == 1.0
        and abs(m["uniform_z"]) <= UNIFORM_Z_MAX
        and abs(m["indep_z"]) <= INDEP_Z_MAX
        and abs(m["trend_z"]) <= TREND_Z_MAX
        and m["diary_life0_rows"] == 1.0
        and m["diary_life_index_covers"] == 1.0
        and m["diary_recall_crosses_death"] == 1.0
        and m["physics_finite"] == 1.0)
    return m


def _control(seed: int) -> dict:
    """Five conditions. (b) and (d) share one run, and they may: the spawn
    sampler cannot read the diary and the diary cannot read the sampler, so
    neither rigging can move the other's statistic. (a), (c) and (e) are their
    own — (c) needs no simulation at all."""
    j0, alpha = _calibration()
    if j0 is None:
        return {"c_calibrated": 0.0}

    immortal = _live(seed, j0=j0, alpha=alpha, lethal=False)
    rigged = _live(seed, j0=j0, alpha=alpha, sampler=_at_death_sampler,
                   wipe_diary=True)
    drifting = _live(seed, j0=j0, alpha=alpha, drift=True)
    legal = W0(seed=seed, j0=j0, alpha=alpha).legal_spawns()
    return {
        "c_calibrated": 1.0,
        "c_immortal_deaths": immortal["deaths"],
        "c_immortal_lives": immortal["n_lives"],
        "c_at_death_indep_z": rigged["indep_z"],
        "c_at_death_paired_dist": rigged["paired_death_spawn_dist"],
        "c_wiped_life0_rows": rigged["diary_life0_rows"],
        "c_wiped_rows": rigged["diary_rows"],
        "c_biased_uniform_z": _uniformity_z(_biased_sampler, legal, seed),
        "c_drift_trend_z": drifting["trend_z"],
        "c_drift_slope": drifting["life_slope_s_per_life"],
        "c_drift_lives": drifting["n_lives"],
    }


def _check(m: dict, c: dict):
    # ── the instrument ──────────────────────────────────────────────────
    if m.get("calibrated", 0.0) != 1.0 or c.get("c_calibrated", 0.0) != 1.0:
        return Status.VOID          # PS.01 has not measured j0/alpha
    if m.get("legal_spawns", 0.0) < m.get("legal_spawns_floor", 1e9):
        return Status.VOID
    for k in ("indep_z", "trend_z", "uniform_z"):
        if not math.isfinite(m.get(k, float("nan"))):
            return Status.VOID      # a null with no spread detects nothing
    if not math.isfinite(c.get("c_at_death_indep_z", float("nan"))) or \
            not math.isfinite(c.get("c_drift_trend_z", float("nan"))):
        return Status.VOID

    # ── the controls, each on its declared side ─────────────────────────
    if c.get("c_immortal_deaths", 1.0) != 0.0:
        return False                # (a) death fires where there is no death
    if c.get("c_at_death_indep_z", 0.0) <= INDEP_Z_MAX:
        return False                # (b) the independence detector is blind
    if c.get("c_biased_uniform_z", 0.0) <= UNIFORM_Z_MAX:
        return False                # (c) the uniformity detector is blind
    if c.get("c_wiped_life0_rows", 1.0) != 0.0:
        return False                # (d) wiping the diary changed nothing
    if c.get("c_drift_trend_z", 0.0) <= TREND_Z_MAX:
        return False                # (e) the trend detector is blind

    # ── the claim ───────────────────────────────────────────────────────
    return bool(m.get("conjunction", 0.0) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["XL.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
