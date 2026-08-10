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
from ..w0 import (MIN_LEGAL_SPAWNS, SIM_S_PER_DECISION, SPAWN_PENETRATION, W0,
                  random_action, uniform_legal_spawn)
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
                              # The drifting control is the one condition this
                              # budget does NOT buy ~15 lives for, because its
                              # manipulation lengthens them — see
                              # DRIFT_DECISIONS below.
SHORT_E0 = 0.10               # the declared short-life fixture
MIN_LIVES = 12                # LC.03's own floor, so this certifies what it needs
STATUE_CHARGES = (0.2, 0.5)   # two independent drains of the resting body
BASAL_TOL = 0.02              # implied 1/b within 2% of 1/BASAL_B, and of itself
UNIFORM_DRAWS = 20_000        # chi-square draws from the sampler alone
UNIFORM_Z_MAX = 4.0           # |z| on the chi-square, normal approximation
N_PERM = 100_000              # permutations for both null distributions. Sized
                              # by the assertion below, not by taste: at 20,000
                              # the attainable floor is 1.0e-4 and the control
                              # gate 1.0e-3 clears it by 1.0001x, which passes
                              # the letter of the margin check and none of its
                              # intent. 100,000 buys 5x.
# THE PERMUTATION GATES ARE p-VALUES, NOT z-SCORES, AND THE FIRST VERSION OF
# THIS SPEC FAILED BECAUSE THEY WERE z. A permutation z for a linear statistic
# is bounded above by exactly sqrt(n - 1) — the extreme pairing is r = 1, and
# z = r * sqrt(n - 1) — so a threshold of 3.0 is UNREACHABLE below n = 10
# however strong the effect. The drifting-world control produced a slope of
# +9.31 s per life across n = 9 lives, which is as monotone as a sequence can
# be, and scored 2.69 against sqrt(8) = 2.83. The gate was not strict; it was
# impossible. A rank p-value has no such ceiling: the same control reaches
# 1/(N_PERM + 1).
#
# This is STRICTER on the experiment side, not looser, and that is checked
# rather than asserted: the old |z| <= 3.0 admits everything out to a
# two-sided p of ~0.003, where P_MIN_NULL = 0.01 rejects at 0.01. Both
# directions are gated (a respawn that lands systematically FAR from the death
# site is as much a leak as one that lands near it).
P_MIN_NULL = 0.01             # experiment: two-sided permutation p must EXCEED
P_MAX_CONTROL = 0.001         # control: two-sided permutation p must FALL BELOW

# AND THE SAME MISTAKE HAS A SECOND FORM, CAUGHT IN THE SMOKE RUN OF THE REPAIR:
# a rank p also has a FLOOR — the most extreme possible observation still scores
# `2 / (N_PERM + 1)`, and separately `2 / n!` when there are fewer orderings than
# draws. At the N_PERM = 2000 of the first repair the floor was 0.0009995 against
# a control gate of 0.001, so the positive controls would have passed by 5e-7 and
# a single tied draw would have failed them. A gate must clear the statistic's
# attainable range by a MARGIN, and the margin is asserted at import rather than
# hoped for: this is the one line that makes the whole class of error impossible
# to reintroduce here, and it costs nothing.
PERM_MARGIN = 10.0
PERM_P_FLOOR = 2.0 / (N_PERM + 1.0)
assert P_MAX_CONTROL >= PERM_MARGIN * PERM_P_FLOOR, (
    f"control gate {P_MAX_CONTROL} is within {PERM_MARGIN}x of the attainable "
    f"floor {PERM_P_FLOOR:.2e} at N_PERM={N_PERM}: a positive control could "
    f"only clear it by rounding")
assert P_MIN_NULL > P_MAX_CONTROL, "the two gates would overlap"
DRIFT_E0 = (0.05, 0.025)      # the drifting world: e0 = 0.05 + 0.025 * life

# THE SAME MARGIN RULE, APPLIED TO A GEOMETRIC FIXTURE. The occupied-pose
# control asks `_penetrating()` a question whose answer it claims to know in
# advance, and `_penetrating()` thresholds at `SPAWN_PENETRATION`. A fixture
# that overlaps by 0.02 m is not a known answer just because 0.02 > 0.001 — it
# is a known answer only if NO mutation of the world can push it under. So the
# depth is measured, not assumed, and required to clear the tolerance by this
# factor; a run whose fixture cannot VOIDs instead of answering. `welded_block`
# measures 90x on every seed tried, a 9x cushion over the margin.
PENETRATION_MARGIN = 10.0

# THE DRIFT CONTROL PAYS FOR ITS OWN MANIPULATION, SO IT GETS A LONGER BUDGET.
# It plants a trend by LENGTHENING each successive life, which means that at a
# fixed decision budget it collects FEWER lives than any other condition — 9,
# where the experiment gets 13.7 — and the permutation p it can reach is set by
# how many inversions n lives can absorb. Measured at N_DECISIONS = 3000: seeds
# 0/2/3/4 came out perfectly monotone and hit the floor at 2.0e-5, while seed 1
# drew two genuine inversions ([20, 31, 72, 50, 89, 69, 76, 87, 97]) and could
# only reach p = 0.00262 against a gate of 0.001. The detector was not blind;
# n = 9 was too small for the noise. Checked, not guessed: a rank statistic is
# WORSE here (Spearman gave 0.00802 on that seed), because this is sampling
# noise and not an outlier, so the fix is lives rather than a better statistic.
# At 2.5x the budget all three seeds hit the floor — n = 15/16, and seed 1
# clears its gate by 50x while carrying SIX inversions.
# The gate P_MAX_CONTROL is UNCHANGED at 0.001. This buys the control the
# evidence to reach it; it does not move it.
DRIFT_DECISIONS = 7500


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
def _perm_matrix(n: int, seed: int) -> np.ndarray:
    """(N_PERM, n) independent permutations of range(n), as one array."""
    rng = np.random.RandomState(seed)
    return np.argsort(rng.rand(N_PERM, n), axis=1)


def _attainable_p(n: int) -> float:
    """The smallest two-sided p this many lives and draws can ever produce.

    Two floors, and the binding one is whichever is larger: the draw count
    (`2 / (N_PERM + 1)`) and the number of distinct orderings (`2 / n!`). A gate
    below this is not strict, it is unreachable — which is exactly how the first
    version of this spec failed.
    """
    if n < 2:
        return 1.0
    orderings = math.factorial(n) if n <= 20 else float("inf")
    return max(PERM_P_FLOOR, 2.0 / orderings)


def _deepest_obstacle_contact(w) -> float:
    """The most negative contact distance between the body and a non-ground geom.

    The same filter `w0._penetrating()` applies — rover-rover pairs excluded (the
    arms fold against the torso in every pose) and ground excluded — but it
    returns the DEPTH rather than a bool, so a fixture's margin over
    `SPAWN_PENETRATION` can be measured instead of asserted. 0.0 means the pose
    touches no obstacle at all, which is the failure mode that made two versions
    of the occupied-pose control read as a coin flip.
    """
    deepest = 0.0
    for k in range(int(w.data.ncon)):
        con = w.data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        mine = (g1 in w.body_gids, g2 in w.body_gids)
        if not any(mine) or all(mine):
            continue
        if (g2 if mine[0] else g1) in w.ground_gids:
            continue
        deepest = min(deepest, float(con.dist))
    return deepest


def _perm_p_and_z(observed: float, null: np.ndarray) -> tuple:
    """Two-sided rank p (the gate) and the z (a diagnostic, never a gate).

    `p = 2 * min(P[null <= obs], P[null >= obs])`, each with the observed value
    added to its own null — the standard +1 correction, which also makes p
    strictly positive so "p = 0" can never mean "no permutation was as extreme"
    and "the null was empty" at the same time.
    """
    n = len(null)
    lo = (1.0 + float((null <= observed).sum())) / (n + 1.0)
    hi = (1.0 + float((null >= observed).sum())) / (n + 1.0)
    p = min(1.0, 2.0 * min(lo, hi))
    sd = float(null.std())
    z = float("nan") if sd == 0.0 else float((observed - float(null.mean())) / sd)
    return p, z


def _independence(deaths: np.ndarray, spawns: np.ndarray, seed: int) -> tuple:
    """Is the spawn closer (or farther) from the death site than chance?

    The statistic is the MEAN death->spawn distance. Under independence the
    paired value is one exchangeable draw from the shuffled pairings; under a
    leak it sits in a tail. Returns (p, z, paired distance); the z is signed so
    that NEGATIVE means "closer than chance", which is the leak direction.
    """
    n = len(deaths)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    paired = float(np.linalg.norm(spawns - deaths, axis=1).mean())
    # The full death x spawn distance matrix, so a permutation is an index, not
    # a recomputation: 20,000 nulls in one vectorised pass.
    dist = np.linalg.norm(deaths[:, None, :] - spawns[None, :, :], axis=2)
    perm = _perm_matrix(n, seed * 6011 + 17)
    null = dist[np.arange(n), perm].mean(axis=1)
    p, z = _perm_p_and_z(paired, null)
    return p, z, paired


def _slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _trend(lengths: np.ndarray, seed: int) -> tuple:
    """Do lives lengthen across the run, beyond what shuffling them produces?

    Gated against a PERMUTATION null rather than a fixed slope, because a life
    length here is not a low-variance quantity: one accidental apple is worth
    150 simulated seconds under the short fixture, so a fixed slope threshold
    would read variance as a trend. The null is the same lives in a shuffled
    order. Returns (p, z, slope).
    """
    n = len(lengths)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    s = _slope(lengths)
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    y = lengths[_perm_matrix(n, seed * 7717 + 31)]
    null = (y - y.mean(axis=1, keepdims=True)) @ x / (x * x).sum()
    p, z = _perm_p_and_z(s, null)
    return p, z, s


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
    for _ in range(DRIFT_DECISIONS if drift else N_DECISIONS):
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

    indep_p, indep_z, paired = _independence(deaths, spawns, seed)
    trend_p, trend_z, slope = _trend(lives, seed)

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
        "indep_p": indep_p,
        "indep_z": indep_z,
        "perm_z_ceiling": (math.sqrt(len(lives) - 1) if len(lives) > 1
                           else float("nan")),
        "perm_p_attainable": _attainable_p(len(lives)),
        "paired_death_spawn_dist": paired,
        "trend_p": trend_p,
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
    # THE OCCUPIED POSE IS READ FROM THE MODEL, AND ITS MARGIN IS MEASURED.
    # This fixture has now been wrong twice, in the same way both times, and
    # the second wrong version was a THEORY about geometry that was never
    # checked against a contact:
    #   v1  probed the literal (LADDER_X, LADDER_Y) — the point BETWEEN the
    #       rails, where penetration depends on the torso radius against a
    #       0.25 m half-width in a per-seed mutated world. 2 of 3 seeds agreed.
    #   v2  probed `ladder_railL`'s live position, on the stated reasoning that
    #       "a body standing at its own centre overlaps it under every
    #       mutation". IT DOES NOT OVERLAP IT AT ALL. The whole ladder is
    #       collision group contype/conaffinity = 4 and the rails never reach
    #       the body; the ONLY obstacle contact at that pose is the TIP of
    #       `rung1`, whose height is `ladder_rung_spacing` — a parameter
    #       `mutate()` jitters. Measured across seeds 0..4: -0.023, +0.013,
    #       -0.020, -0.025, -0.059 m against a 0.001 m tolerance. Reading the
    #       pose off the live model fixed the wrong half; the answer was still
    #       a coin flip on a mutated parameter, so v2 scored 0.667 exactly as
    #       v1 had.
    # `welded_block` is the fixture that cannot be dodged: an unconditional
    # 0.15 m box welded at a fixed pos (playground.py:422 — note `fulcrum` is
    # deeper but sits behind `if p.seesaw`, so it is not unconditional), in the
    # body's own collision group. Measured depth -0.090 m on every seed of
    # 0..4, invariant, and the contact names the block itself.
    #
    # AND THE MARGIN IS NO LONGER A CLAIM. The depth is recorded and gated:
    # 90x the tolerance is the cushion, and a run whose fixture cannot clear
    # PENETRATION_MARGIN VOIDs rather than answering. That is the same rule the
    # permutation floor already carries, applied to geometry — a fixture must
    # clear the ATTAINABLE range of the predicate it is interrogating, and the
    # margin is measured rather than reasoned about.
    probe = W0(seed=seed, j0=j0, alpha=alpha)
    block = int(probe.model.geom("welded_block").id)
    probe.mujoco.mj_forward(probe.model, probe.data)
    rx, ry = (float(probe.data.geom_xpos[block][0]),
              float(probe.data.geom_xpos[block][1]))
    probe._place(rx, ry)
    probe.mujoco.mj_forward(probe.model, probe.data)
    m["occupied_pose_rejected"] = float(probe._penetrating())
    m["occupied_probe_depth"] = _deepest_obstacle_contact(probe)
    m["occupied_probe_margin"] = float(
        abs(m["occupied_probe_depth"]) / SPAWN_PENETRATION)
    m["occupied_probe_x"], m["occupied_probe_y"] = rx, ry
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
        and m["occupied_pose_rejected"] == 1.0
        and m["corner_pose_accepted"] == 1.0
        and m["n_lives"] >= MIN_LIVES
        and m["spawn_legal_frac"] == 1.0
        and abs(m["uniform_z"]) <= UNIFORM_Z_MAX
        and m["indep_p"] >= P_MIN_NULL
        and m["trend_p"] >= P_MIN_NULL
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
    biased_z = _uniformity_z(_biased_sampler, legal, seed)

    # EVERY CONTROL GATE IS REDUCED TO A PER-SEED BOOLEAN HERE, NOT LEFT AS A
    # RAW STATISTIC FOR `_check` TO THRESHOLD. `run_spec` hands `_check` the
    # MEAN over seeds (`protocol.py:_aggregate`), so thresholding a raw p there
    # asks "was the control detective ON AVERAGE" — and averages of p-values
    # near a floor are dominated by the seeds that saturate it. Measured on the
    # 11:05 run: `c_drift_trend_p` came back mean 8.87e-4 against a 1e-3 gate
    # with std 1.23e-3, i.e. two seeds pinned at the 2e-5 floor and ONE SEED
    # BLIND at p ~ 2.6e-3. The gate passed. A control that is blind on a third
    # of the seeds is not a control on those seeds, and law 2 does not average.
    # The `*_ok` fields below are 1.0 only when THAT seed's control fired, so
    # their mean is 1.0 iff every seed fired — the same trick the experiment
    # side already uses for `conjunction`. The raw statistics stay in the
    # ledger beside them; they are the diagnosis, the booleans are the gate.
    return {
        "c_calibrated": 1.0,
        "c_immortal_deaths": immortal["deaths"],
        "c_immortal_lives": immortal["n_lives"],
        "c_at_death_indep_p": rigged["indep_p"],
        "c_at_death_indep_z": rigged["indep_z"],
        "c_at_death_paired_dist": rigged["paired_death_spawn_dist"],
        "c_at_death_perm_p_attainable": rigged["perm_p_attainable"],
        "c_drift_perm_p_attainable": drifting["perm_p_attainable"],
        "c_wiped_life0_rows": rigged["diary_life0_rows"],
        "c_wiped_rows": rigged["diary_rows"],
        "c_biased_uniform_z": biased_z,
        "c_drift_trend_p": drifting["trend_p"],
        "c_drift_trend_z": drifting["trend_z"],
        "c_drift_perm_z_ceiling": drifting["perm_z_ceiling"],
        "c_drift_slope": drifting["life_slope_s_per_life"],
        "c_drift_lives": drifting["n_lives"],
        # ── the five gates, evaluated inside the seed ───────────────────
        "c_immortal_ok": float(immortal["deaths"] == 0.0),
        "c_at_death_ok": float(rigged["indep_p"] <= P_MAX_CONTROL),
        "c_biased_ok": float(biased_z > UNIFORM_Z_MAX),
        "c_wiped_ok": float(rigged["diary_life0_rows"] == 0.0),
        "c_drift_ok": float(drifting["trend_p"] <= P_MAX_CONTROL),
        # ── and the two attainability preconditions, likewise ───────────
        "c_at_death_attainable_ok": float(
            rigged["perm_p_attainable"] * PERM_MARGIN <= P_MAX_CONTROL),
        "c_drift_attainable_ok": float(
            drifting["perm_p_attainable"] * PERM_MARGIN <= P_MAX_CONTROL),
        # And the power precondition that `attainable` cannot see: the extreme
        # ordering is reachable at n = 9, but two ordinary inversions are not.
        # Hold the control to the same life floor the experiment is held to, so
        # a control starved of lives VOIDs instead of being read as a verdict.
        "c_drift_lives_ok": float(drifting["n_lives"] >= MIN_LIVES),
    }


def _check(m: dict, c: dict):
    # ── the instrument ──────────────────────────────────────────────────
    if m.get("calibrated", 0.0) != 1.0 or c.get("c_calibrated", 0.0) != 1.0:
        return Status.VOID          # PS.01 has not measured j0/alpha
    if m.get("legal_spawns", 0.0) < m.get("legal_spawns_floor", 1e9):
        return Status.VOID
    for k in ("indep_p", "trend_p", "uniform_z"):
        if not math.isfinite(m.get(k, float("nan"))):
            return Status.VOID      # too few lives to permute, or a null with
            # no spread; either way the statistic detects nothing
    if not math.isfinite(c.get("c_at_death_indep_p", float("nan"))) or \
            not math.isfinite(c.get("c_drift_trend_p", float("nan"))):
        return Status.VOID
    # A positive control that CANNOT reach its own gate has not been run; it has
    # been asked for the impossible, and reading its miss as a verdict is what
    # produced the FAIL this spec was revised from. Gated per seed (`_control`
    # explains why the mean of a p-value is not a gate): ANY seed whose control
    # was asked for the impossible VOIDs the run.
    for k in ("c_at_death_attainable_ok", "c_drift_attainable_ok",
              "c_drift_lives_ok"):
        if c.get(k, 0.0) != 1.0:
            return Status.VOID
    # And the geometric fixture, on the same rule: a probe pose that does not
    # clear `SPAWN_PENETRATION` by PENETRATION_MARGIN has not asked the legality
    # predicate a question it could be right or wrong about.
    if m.get("occupied_probe_margin", 0.0) < PENETRATION_MARGIN:
        return Status.VOID

    # ── the controls, each on its declared side, EVERY SEED ─────────────
    # These are means of per-seed booleans, so `== 1.0` reads "fired on every
    # seed". Thresholding the raw statistics here instead let a control that
    # was blind on 1 of 3 seeds pass on the strength of the other two.
    if c.get("c_immortal_ok", 0.0) != 1.0:
        return False                # (a) death fires where there is no death
    if c.get("c_at_death_ok", 0.0) != 1.0:
        return False                # (b) the independence detector is blind
    if c.get("c_biased_ok", 0.0) != 1.0:
        return False                # (c) the uniformity detector is blind
    if c.get("c_wiped_ok", 0.0) != 1.0:
        return False                # (d) wiping the diary changed nothing
    if c.get("c_drift_ok", 0.0) != 1.0:
        return False                # (e) the trend detector is blind

    # ── the claim ───────────────────────────────────────────────────────
    return bool(m.get("conjunction", 0.0) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["XL.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
