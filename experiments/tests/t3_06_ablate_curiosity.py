"""T3.06 — remove the intrinsic reward and unprompted coverage must collapse.

Tier 3 is "earn your parameters": ablate the component, something measurable
must degrade. T2.08 established that a boredom-form pseudo-count bonus drives
coverage above random IN A WORLD WITH NO EXTRINSIC REWARD ANYWHERE. That is
not yet a reason to keep the component, because the world Jack lives in has
needs (GOAL.md: "curiosity is the explorer, needs are the reason"). The
question this spec asks is the one that decides whether curiosity survives
contact with a task: **once there is something to want, does the curiosity
term still buy exploration, or does the task reward simply absorb it?**

THE VACUITY THIS SPEC EXISTS TO AVOID, stated before the design, because the
obvious implementation is circular. The intrinsic reward IS a coverage bonus:
`r = 1/sqrt(N(s')) - 0.5` pays for entering rarely-visited cells. Measuring
"does a coverage bonus raise coverage" is not an experiment, it is arithmetic,
and T2.08 already spent that question. Three things make this spec ask a new
one:

  1. **There is an extrinsic task, and the ablated arm must be shown to have
     LEARNED it.** The null is not a do-nothing arm; it is a competent
     goal-seeker. `task_dwell` is a pre-registered RIG gate on the WORST life
     of the ablated arm, and a run in which the extrinsic arm never learned to
     occupy the goal is VOID, not FAIL — there was no ablation, only two
     random walks, and FAIL would fire this spec's `kills` field off a run
     that never asked the question. (T2.09's lesson, generalised: an ablation
     whose ablated arm is dead is measuring nothing, exactly as an at-chance
     control whose instrument is dead is measuring nothing.)
  2. **The predicted mechanism runs the OTHER way from the circular one.**
     T2.08's pilot established that in bootstrapped tabular Q every
     always-positive reward ANTI-explores: the visited core's accumulated Q
     beats one-shot frontier bonuses ("bonus myopia"). A goal reward is
     always-positive, so the extrinsic-only arm is predicted to camp and its
     coverage to fall BELOW random — the degradation this spec claims is a
     property of the task reward, and the curiosity term's job is to survive
     it. A design where the ablated arm merely fails to be helped is weaker
     than one where it is actively harmed and the component rescues it.
  3. **The control adds a reward of the same magnitude carrying no
     information.** `shuftask` is extrinsic + a uniform draw from the agent's
     own past bonuses. If it also recovers coverage, the measurement is about
     reward magnitude or about Q-value noise, not about curiosity, and the
     claim is void of content whatever the experiment arm did.

WHAT THIS REGIME REMOVES, said out loud (LESSONS: list what the chosen regime
removes). Inherited from T2.08 and PG.4's certified rig: observation noise
(T2.09's subject), movable clutter, and percept-driven novelty — the bonus is
a position-state pseudo-count, so no arm reads the retina. Added here and
removed deliberately: the goal is STATIC within a life (a respawning goal
would make the task itself reward exploration and destroy the contrast the
spec is built on), and it is a bare reward with no satiation, no death and no
second need. This spec therefore measures the exploration/exploitation
tension in its simplest honest form. The needs-world version is the NE family.

THE RIG is T2.08's, which is PG.4's certified rover: velocity-controlled
slider, `contype 0`, `n_objects=0`, panel static, so every one of the 484
cells is reachable and coverage's ceiling is a true 1.0. `IMPL_DEPS` hashes
the world contract and both parents, so moving any of them makes this
certificate go stale loudly rather than stand over a rig it no longer
describes.

ARMS (each arm's per-seed number is the mean over the INFORMATIVE lives of
that seed — see THE LIFE PROTOCOL — because a null measured by one draw is a
sample, not a null, and a null averaged over lives in which it never met the
task is not a null at all):
    task      extrinsic only                      — the registered null
    curious   extrinsic + boredom bonus           — the full system
    shuftask  extrinsic + time-permuted bonus     — the CONTROL, must fail
    random    random walk, no learning            — the dwell instrument's zero

THE LIFE PROTOCOL (v2, written 2026-08-30 BEFORE the v2 pilot drew a number;
it is the repair the v1 pilot pre-registered in this docstring, and the two
deviations from that text are named and justified below).

A LIFE IS INFORMATIVE IFF THE ABLATED ARM LEARNED THE TASK IN IT — i.e. the
`task` arm of that sub-seed reached `dwell >= INFORMATIVE_DWELL_MIN`. The
claim is scored ONLY over informative lives, paired arm-to-arm on the shared
sub-seed (same world, same goal cell). A seed with fewer than
`MIN_INFORMATIVE_LIVES` informative lives is VOID, never FAIL: there was no
ablation to measure.

  Why the selection is safe. The criterion reads ONE number — the ABLATED
  arm's dwell — and never the curious arm's coverage, which is the quantity
  under test. No life can be dropped for being unflattering to the claim,
  because nothing about the claim is visible to the selector. Every life,
  informative or not, is recorded in `per_life` with all four arms' numbers,
  so the subset is recomputable from the record by someone who is not its
  author. (T2.09's informative-seed protocol, applied one level down.)

  DEVIATION 1, and it is a STRENGTHENING. The v1 text pre-registered
  `dwell > 0` as the criterion. That is too weak to do the job it was written
  for: a life in which the ablated arm brushed the goal once at decision 3900
  and never returned has `dwell = 0.00025 > 0`, so it would enter the claim
  subset while the arm demonstrably did not learn to camp — reintroducing the
  vacuity the whole gate exists to prevent. Setting the criterion AT the rig
  bar (`INFORMATIVE_DWELL_MIN = TASK_DWELL_MIN`) selects a STRICT SUBSET of
  `dwell > 0`: strictly fewer lives qualify, so it is harder to reach
  `MIN_INFORMATIVE_LIVES` and harder to record a PASS. The alternative —
  select on `dwell > 0` and ALSO gate the worst selected life at
  `TASK_DWELL_MIN` — is stricter still and is REJECTED for a stated reason:
  it VOIDs an entire seed whenever any single life finds the goal late, which
  is the same bimodality failure one level down, and a gate that VOIDs
  everything measures nothing.

  DEVIATION 2. `task_dwell_worst_life` survives as a gate but is now
  tautologically satisfied on the informative subset (min over lives selected
  for being >= the bar). It is KEPT anyway, and kept in the record, as a
  recomputable receipt: a reader can check the fold was applied by verifying
  that number never sits below the bar. It is no longer load-bearing; the
  load moved to `n_informative`.

PRE-REGISTERED GATES. **PROVISIONAL — `_GATES_FROZEN = False` and `run()`
refuses.** The bars below are placeholders derived from the parent specs'
measured numbers, NOT from a pilot of this rig; a pilot freezes them and the
values it freezes are recorded in this docstring under PILOT RECORD, in the
open, with any bar that moved named and justified (SM.02's idiom, T2.09's
precedent).

  RIG (any violated -> VOID, not FAIL):
    n_informative - 1.5*std >= MIN_INFORMATIVE_LIVES — enough lives in which
        the ABLATED arm actually learned the task, IN EVERY SEED. THIS IS THE
        GATE THE v1 PILOT BOUGHT: `task_dwell_worst_life = 0.0000` reproduced
        on both seed families, so demanding every life be informative would
        VOID nearly every seed, and gating the MEAN would certify a rig on
        evidence a quarter of which is a random walk. Counting them is the
        third option and the only honest one — but `_check` reads metrics
        already meaned ACROSS SEEDS, so the raw count would let seeds of 2 and
        10 average to a healthy 6 and rebuild the same bimodality one level
        up. The 1.5*std bound is the file's own exact all-seeds rule, and it
        collapses to the raw count on a single-seed pilot.
    task_dwell_worst_life >= TASK_DWELL_MIN — retained as a receipt that the
        fold ran (see DEVIATION 2). Not load-bearing.
    random_dwell_worst_life + 1.5*std <= RANDOM_DWELL_MAX — the dwell
        instrument reads, on its worst seed as well as its worst life, near
        its chance value (1/484 = 0.0021) on a non-learner. If a random
        walk also occupies the goal cell, the cell is a physical attractor and
        `task_dwell` is certifying geometry, not learning. Scored over ALL
        lives, not the informative subset: an instrument's zero must be
        checked everywhere it is read, not only where the claim is scored.
    coverage_random in [RANDOM_COV_LO, RANDOM_COV_HI] — PG.4/T2.08's
        construction check: the world is reachable and coverage is not
        saturated at the horizon, so there is room for an arm to be worse.

  CLAIM (all three), every coverage scored over the INFORMATIVE lives only:
    delta_coverage = mean over informative lives of
        [cov(curious, life) - cov(task, life)] >= DELTA_MIN
    delta_coverage - 1.5 * std > 0 — the all-seeds rule, exact: for n=3 and
        the recorder's ddof=0 std the extreme deviation is <= sqrt(2)*std, so
        1.5 guarantees every seed's delta is positive (T2.08's idiom).
    delta_coverage * sqrt(3) / std >= 3.0 — the house 3-sigma learning gate on
        the paired delta. Paired is the right ruler: both arms run on the SAME
        sub-seed worlds with the SAME goal cells.

  CONTROL (must fail): delta_shuf = mean over the SAME informative lives of
    [cov(shuftask, life) - cov(task, life)] < DELTA_MIN. The control inherits
    the claim's life subset exactly — it is selected by the `task` arm, which
    both share — so a control cannot be scored on an easier set of worlds.

WHERE THE v2 NUMBERS COME FROM — every one of them exogenous, because the v1
pilot's warning was precisely that a bar anchored to its own pilot's bulk
reads as a per-run coin flip (the BA.01-v3 / T2.08-v1 lottery disease):

  DELTA_MIN = 0.05, UNCHANGED, and deliberately NOT re-derived from the two
      observed deltas (+0.0558, +0.1658) in either direction. It is T2.08's
      registered `MARGIN_MIN` — the same quantity, on the same rig, in the
      same units, fixed before this spec existed and with no knowledge of its
      numbers, which is what "exogenous" means. Its anti-collapse content,
      stated in cells: 0.05 x 484 = 24 cells, so the curious arm must reach a
      region about five cells on a side that the ablated arm never enters.
      Anything smaller is not a behavioural difference worth a `kills` field.
      **A bar this close to the smaller observed delta may well FAIL. That is
      the bar doing its job, not a reason to move it.**

  MIN_INFORMATIVE_LIVES = 6, from a statistical requirement, not a count seen
      in a pilot: six paired lives is the smallest n at which a one-sided
      paired sign test can reach p < 0.05 (2^-6 = 0.0156). Below it the
      informative subset cannot in principle carry the claim, whatever the
      mean says, so VOID is the correct verdict rather than FAIL.

  LIVES_PER_ARM = 16, up from 4, sized so MIN_INFORMATIVE_LIVES is reachable
      without a lottery. The v1 pilot saw one dead life in four on both
      families; taking that at face value as p ~ 0.75 informative, 16 lives
      put ~12 in the subset. **If p is really nearer 0.4 the seed VOIDs, and
      that is the honest outcome** — the fix would be more lives or a longer
      horizon, both of which are re-registrations, not bar moves. Cost is the
      reason 16 and not 40: ~4.3 s per life measured, so 16 lives x 4 arms x
      3 seeds ~ 14 min of CPU.

FALSIFICATION, restated so it cannot be quietly narrowed: if the extrinsic
arm learns the task (rig green) and its coverage is not measurably below the
curious arm's, curiosity did not earn its parameters in the presence of a
need, and the registry's `kills: IntrinsicCuriosityModule` fires. That is a
result about the architecture, not a tuning miss, and it routes to the Review
rather than to a re-roll.

BUDGET. Registered `gpu<2h`. MEASURED at ~17 s per arm-seed (4 lives x 4000
decisions) on 4 ARM cores, so a registered run of 4 arms x 3 seeds is ~3.5
minutes of pure numpy + MuJoCo. **This spec is CPU and the registry must be
corrected to say so** (LESSONS: a declared attribute consumed by routing must
match behaviour) — but NOT before the gates are frozen, because moving the
budget also moves which queue-depth class this spec stocks, and a spec whose
`run()` still refuses stocks nothing. Do not spend Kaggle hours on it; the
expiring free hours belong to specs that need a GPU.

PILOT RECORD — seed-90 family, 2026-08-29 20:15 UTC, /data/t3_06_pilot.json,
4 lives/arm at the registered 4000 decisions. **GATES STAY PROVISIONAL: the
pilot found a design fault, not a set of bars.**

    arm        cov      cov_lo   dwell    dwell_lo  dwell_hi
    task       0.5553   0.4360   0.1104   0.0000    0.2983
    curious    0.6111   0.3017   0.1442   0.0000    0.5427
    shuftask   0.4576   0.2562   0.1216   0.0000    0.4838
    random     0.6152   0.5888   0.0018   0.0000    0.0043
    delta_coverage +0.0558   delta_shuf -0.0977   task_vs_random -0.0599

Seed-91 family, same launch (total wall 134.9 s for 32 lives):

    arm        cov      cov_lo   dwell    dwell_lo  dwell_hi
    task       0.4561   0.2975   0.2702   0.0000    0.7143
    curious    0.6219   0.4855   0.1653   0.0000    0.5750
    shuftask   0.4468   0.4112   0.2190   0.0000    0.4170
    random     0.6612   0.4070   0.0010   0.0000    0.0032
    delta_coverage +0.1658   delta_shuf -0.0093   task_vs_random -0.2051

Both families agree on the three confirmations and on the fault. Two numbers
from seed 91 that the freeze must respect and that seed 90 alone would have
hidden:

  - **`task_cov_vs_random` is -0.2051 here against -0.0599 there.** The
    camping effect is real in both but its SIZE varies by 3.4x across seed
    families, so `delta_coverage` (+0.0558 / +0.1658) has a seed spread wider
    than its own provisional bar. `DELTA_MIN = 0.05` sits below the smaller of
    the two observed values with almost no room — a pilot-bulk-anchored bar,
    which is precisely the BA.01-v3 / T2.08-v1 lottery disease. The freeze must
    re-derive `DELTA_MIN` from an exogenous purpose (an anti-collapse floor),
    not from these two numbers, or it will read as a per-run coin flip.
  - **`delta_shuf = -0.0093` here against -0.0977 there.** The control still
    fails on both families, but on seed 91 it fails by one tenth as much. The
    information-free bonus is not reliably harmful — it is reliably *not
    helpful*, which is the weaker claim the control is entitled to make and
    the one the gate already encodes (`delta_shuf < DELTA_MIN`, not
    `delta_shuf < 0`). Do not strengthen the control gate to `< 0` on the
    strength of seed 90.

  - **And the fault reproduces: `task_dwell_worst_life = 0.0000` on BOTH
    families.** Two for two, so the bimodality is a property of the design and
    not of a draw. The informative-life protocol below is not optional.

THREE THINGS THE PILOT CONFIRMED, and they are the reasons to keep this
design rather than start over:

  1. **The predicted camping mechanism fired.** `task_cov_vs_random = -0.0599`:
     the extrinsic-only arm explores measurably LESS than a random walk. The
     ablation is not merely un-helped, it is actively harmed, exactly as
     T2.08's bonus-myopia finding predicts of any always-positive reward in
     bootstrapped tabular Q. That is what makes the claim non-circular.
  2. **The control fails in the right direction and by a wide margin.**
     `delta_shuf = -0.0977` — a magnitude-matched, information-free bonus does
     not recover the lost coverage, it costs more. The effect is not reward
     magnitude and not Q-value noise.
  3. **The dwell instrument's zero reads at chance.** Random dwell 0.0018
     against the analytic 1/484 = 0.0021, worst life 0.0043. `RANDOM_DWELL_MAX
     = 0.02` is ~5x the worst observed and is confirmed, not tuned.

AND THE FAULT, which is why nothing is frozen. **`task_dwell_worst_life =
0.0000`: one life in four of the ABLATED arm never found the goal at all.**
The apparatus is bimodal across LIVES exactly as T2.09's was across SEEDS —
the goal is a single cell of 484 behind an epsilon-greedy search, so finding
it is close to a coin flip within a life's budget. As written, the rig gate
`task_dwell_worst_life >= TASK_DWELL_MIN` would VOID nearly every seed, and
the two available repairs are both wrong:

  - **Lower the bar to 0.** That deletes the only instrument proving the
    ablated arm learned the task, which is the entire defence against the
    vacuity named at the top of this docstring. Forbidden.
  - **Gate the MEAN dwell (0.1104, comfortably over any bar).** That is
    precisely the defect T2.09 was rewritten to remove: a mean over a bimodal
    apparatus certifies a rig on evidence one quarter of which is a random
    walk.

THE REPAIR, pre-registered here before the next pilot draws a number, so it
cannot be chosen to flatter a result: adopt T2.09's informative-unit protocol
one level down. A LIFE is informative iff its ablated arm found the goal
(`dwell > 0` for the `task` arm of that sub-seed); the claim scores only
informative lives, paired across arms on the shared sub-seed; a seed is VOID
below a pre-registered minimum count of informative lives out of
`LIVES_PER_ARM`. The selection criterion reads ONLY the ablated arm's dwell —
never the curious arm's coverage — so no life can be dropped for being
unflattering, and every life, informative or not, is recorded in `per_life`
so the subset is recomputable from the record by someone who is not its
author. `LIVES_PER_ARM` will need raising from 4 for the count to be
meaningful; the run costs ~17 s per arm-seed, so it is affordable.
"""
from __future__ import annotations

import math

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# The world contract and BOTH parents hash into this certificate: PG.4 owns
# the rover rig, T2.08 owns the boredom-bonus construction this spec ablates.
IMPL_DEPS = ["playground.py",
             "experiments/tests/pg_4_noisy_tv.py",
             "experiments/tests/t2_08_curiosity_coverage.py"]

_GATES_FROZEN = False           # provisional — see PRE-REGISTERED GATES

# --- rig constants, inherited verbatim from T2.08 so the ablation is of the
# --- certified apparatus and not of a lookalike ---------------------------
CELL_M = 0.5
GRID_LO, GRID_HI = -5.5, 5.5
GRID_N = int(round((GRID_HI - GRID_LO) / CELL_M))    # 22
N_CELLS = GRID_N * GRID_N                             # 484
N_DECISIONS = 4000              # T2.08's discriminating horizon
LIVES_PER_ARM = 16              # v2: was 4. See WHERE THE v2 NUMBERS COME FROM.
SUBSTEPS = 40
SPEED = 1.5
GAMMA = 0.95
Q_LR = 0.2
EPS_HI, EPS_LO = 1.0, 0.10
BORED_BASELINE = 0.5

# --- what this spec adds: the extrinsic task ------------------------------
R_GOAL = 1.0                    # paid on every decision that ENDS in the goal
                                # cell. Scaled to the bonus, whose range is
                                # (-0.5, +0.5]: the task is worth at least as
                                # much as the strongest possible novelty, so
                                # the ablated arm has a real reason to camp.
GOAL_MIN_CELLS = 6              # the goal is drawn at least this many cells
                                # (3.0 m) from the start cell, so reaching it
                                # is a search and not an accident.

# --- PROVISIONAL bars -----------------------------------------------------
TASK_DWELL_MIN = 0.10           # placeholder: an informative life is one in
                                # which the ablated arm spends >=10% of its
                                # decisions in one cell of 484 (chance 0.0021,
                                # ~48x). Pilot freezes it.
INFORMATIVE_DWELL_MIN = TASK_DWELL_MIN   # ONE threshold, not two: the life
                                # selector IS the rig instrument, so a life
                                # can never be admitted to the claim subset by
                                # a laxer rule than the one that certifies the
                                # arm learned. See THE LIFE PROTOCOL.
MIN_INFORMATIVE_LIVES = 6       # fewer -> VOID, not FAIL. Sign-test floor.
RANDOM_DWELL_MAX = 0.02         # ~10x chance; a random walk must not camp.
RANDOM_COV_LO = 0.40            # T2.08 measured random 0.602-0.638 at this
RANDOM_COV_HI = 0.95            # horizon; the band is wide on purpose — it is
                                # a construction check, not a performance bar.
DELTA_MIN = 0.05                # T2.08's MARGIN_MIN, same rig, same units —
                                # exogenous, and NOT re-derived from this
                                # spec's pilot in either direction.
SEED_SPREAD_FACTOR = 1.5
DELTA_TSTAT_MIN = 3.0

_ACTIONS = [(0.0, 0.0)] + [
    (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)
]

_ARMS = ("task", "curious", "shuftask", "random")


def _cell(x: float, y: float) -> int:
    cx = min(GRID_N - 1, max(0, int((x - GRID_LO) / CELL_M)))
    cy = min(GRID_N - 1, max(0, int((y - GRID_LO) / CELL_M)))
    return cy * GRID_N + cx


def _goal_cell(sub_seed: int) -> int:
    """Deterministic per LIFE, not per arm: every arm of a given life searches
    the SAME world for the SAME goal, which is what makes the delta paired."""
    import numpy as np

    rng = np.random.RandomState((sub_seed * 7919 + 11) % (2 ** 32 - 1))
    sx, sy = GRID_N // 2, GRID_N // 2          # the rover starts at (0, 0)
    while True:
        cx, cy = int(rng.randint(GRID_N)), int(rng.randint(GRID_N))
        if abs(cx - sx) + abs(cy - sy) >= GOAL_MIN_CELLS:
            return cy * GRID_N + cx


def _life(sub_seed: int, arm: str, n_decisions: int = N_DECISIONS) -> tuple:
    """One life. Returns (final coverage, goal-dwell fraction).

    The reward buffer is filled with the TRUE bonus sequence in every arm that
    computes one, so `shuftask`'s reward distribution is magnitude-matched by
    construction and only the information is destroyed — a fresh uniform draw
    per step, never one fixed permutation shared across seeds (LESSONS).
    """
    import mujoco
    import numpy as np

    from .pg_4_noisy_tv import _build

    model, data, _panel_gid, _rover_bid, (ax, ay) = _build()
    agent_rng = np.random.RandomState((sub_seed * 104729 + 7) % (2 ** 32 - 1))
    goal = _goal_cell(sub_seed)

    q = np.zeros((N_CELLS, len(_ACTIONS)))
    counts = np.zeros(N_CELLS)
    rbuf: list = []
    visited: set = set()
    in_goal = 0
    for t in range(n_decisions):
        s = _cell(float(data.qpos[-2]), float(data.qpos[-1]))
        visited.add(s)
        if arm == "random":
            a = int(agent_rng.randint(len(_ACTIONS)))
        else:
            eps = max(EPS_LO,
                      EPS_HI - (EPS_HI - EPS_LO) * t / (n_decisions // 3))
            if agent_rng.uniform() < eps:
                a = int(agent_rng.randint(len(_ACTIONS)))
            else:
                best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                a = int(best[agent_rng.randint(len(best))])

        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        s2 = _cell(float(data.qpos[-2]), float(data.qpos[-1]))
        counts[s2] += 1
        if s2 == goal:
            in_goal += 1

        bonus = 1.0 / math.sqrt(counts[s2]) - BORED_BASELINE
        rbuf.append(bonus)
        r_ext = R_GOAL if s2 == goal else 0.0
        if arm == "task":
            r = r_ext
        elif arm == "curious":
            r = r_ext + bonus
        elif arm == "shuftask":
            r = r_ext + rbuf[int(agent_rng.randint(len(rbuf)))]
        else:
            r = 0.0
        if arm != "random":
            q[s, a] += Q_LR * (r + GAMMA * q[s2].max() - q[s, a])

    return len(visited) / N_CELLS, in_goal / n_decisions


def _sub_seeds(seed: int) -> list:
    """Distinct from T2.08's registered families (seed*101 + k*17 + 3) and
    from its pilot's 90-family, so no life is shared with the parent's
    certificate: seed 0 -> 5..44, seed 1 -> 312..351, seed 2 -> 619..658."""
    return [seed * 307 + k * 13 + 5 for k in range(LIVES_PER_ARM)]


_ARM_CACHE: dict = {}


def _arm(seed: int, arm: str) -> dict:
    """Per-LIFE rows, keyed by sub-seed: {sub_seed: (coverage, dwell)}.

    Cached because `_experiment` and `_control` both need the `task` arm (it
    is the null AND the life selector) and `run_spec` calls them separately;
    the lives are deterministic in the sub-seed, so recomputing them would
    cost a third of the run's wall time to produce identical numbers.
    """
    key = (seed, arm, LIVES_PER_ARM, N_DECISIONS)
    if key not in _ARM_CACHE:
        _ARM_CACHE[key] = {s: _life(s, arm) for s in _sub_seeds(seed)}
    return _ARM_CACHE[key]


def _informative_lives(seed: int) -> list:
    """The sub-seeds whose ABLATED arm learned the task.

    Reads exactly one number — the `task` arm's dwell — and never touches the
    curious arm's coverage, which is the quantity under test. That is what
    makes dropping a life safe: nothing about the claim is visible here.
    """
    task = _arm(seed, "task")
    return sorted(s for s, (_cov, dwell) in task.items()
                  if dwell >= INFORMATIVE_DWELL_MIN)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _experiment(seed: int) -> dict:
    arms = {name: _arm(seed, name) for name in ("task", "curious", "random")}
    live = _informative_lives(seed)

    # Paired per-life deltas on the shared sub-seed: same world, same goal.
    paired = [arms["curious"][s][0] - arms["task"][s][0] for s in live]
    n_pos = sum(1 for d in paired if d > 0.0)

    def cov(name):
        return _mean([arms[name][s][0] for s in live])

    def dwell(name):
        return _mean([arms[name][s][1] for s in live])

    all_seeds = _sub_seeds(seed)
    return {
        # THE RIG GATE the v1 pilot bought. Everything below is scored on the
        # subset this number counts.
        "n_informative": float(len(live)),
        "lives_per_arm": float(LIVES_PER_ARM),
        "delta_coverage": round(_mean(paired), 4),
        "coverage_curious": round(cov("curious"), 4),
        "coverage_task": round(cov("task"), 4),
        "coverage_random": round(cov("random"), 4),
        # RIG instruments.
        "task_dwell": round(dwell("task"), 4),
        # Tautological on the informative subset by construction — kept as a
        # receipt that the fold ran (DEVIATION 2), not as a live gate.
        "task_dwell_worst_life": round(
            min([arms["task"][s][1] for s in live], default=0.0), 4),
        "curious_dwell": round(dwell("curious"), 4),
        "random_dwell": round(dwell("random"), 4),
        # Over ALL lives, not the subset: an instrument's zero must be checked
        # everywhere it is read.
        "random_dwell_worst_life": round(
            max(arms["random"][s][1] for s in all_seeds), 4),
        # Reported, not gated.
        "delta_paired_worst_life": round(min(paired, default=0.0), 4),
        "delta_paired_best_life": round(max(paired, default=0.0), 4),
        "n_lives_curious_wins": float(n_pos),
        "curious_cov_worst_life": round(
            min([arms["curious"][s][0] for s in live], default=0.0), 4),
        "task_cov_best_life": round(
            max([arms["task"][s][0] for s in live], default=0.0), 4),
        # Reported, not gated: is the task reward actually costing coverage?
        # Negative means the extrinsic-only arm explores less than a random
        # walk — the predicted camping mechanism, visible in the record.
        "task_cov_vs_random": round(cov("task") - cov("random"), 4),
        # EVERY life, informative or not, so the subset is recomputable by a
        # reader who is not its author. Named for what survives `_aggregate`:
        # non-numeric fields take the FIRST seed's value, so this is seed 0's
        # rows only, and the pilot artifacts carry all of them.
        "per_life_first_seed_only": [
            [s,
             1.0 if s in live else 0.0,
             round(arms["task"][s][0], 4), round(arms["task"][s][1], 4),
             round(arms["curious"][s][0], 4), round(arms["curious"][s][1], 4),
             round(arms["random"][s][0], 4), round(arms["random"][s][1], 4)]
            for s in all_seeds],
        "per_life_cols": ("sub_seed informative task_cov task_dwell "
                          "curious_cov curious_dwell random_cov random_dwell"),
    }


def _control(seed: int) -> dict:
    """Extrinsic + a time-permuted, magnitude-matched bonus. Must NOT recover
    the coverage the ablation cost: if it does, the effect is reward
    magnitude, not curiosity.

    Scored on the SAME informative lives as the claim — the subset is selected
    by the `task` arm, which both share — so the control can never be measured
    on an easier set of worlds than the arm it is supposed to shadow.
    """
    shuf, task = _arm(seed, "shuftask"), _arm(seed, "task")
    live = _informative_lives(seed)
    paired = [shuf[s][0] - task[s][0] for s in live]
    return {"coverage_shuftask": round(_mean([shuf[s][0] for s in live]), 4),
            "shuftask_dwell": round(_mean([shuf[s][1] for s in live]), 4),
            "control_n_informative": float(len(live)),
            "delta_shuf": round(_mean(paired), 4)}


def _check(m: dict, c: dict):
    # An ablated arm that never learned the task is an APPARATUS outcome, not
    # a refutation: there was no ablation to measure. VOID, so that a dead rig
    # can never fire this spec's `kills` field.
    # `_check` sees metrics ALREADY MEANED across the registered seeds, so a
    # gate written on the mean of a worst-case instrument re-opens, one level
    # up, exactly the bimodality this spec's life fold closes: seeds with 2
    # and 10 informative lives average to a healthy-looking 6. Both worst-case
    # rig instruments are therefore bounded to their WORST SEED by the file's
    # own exact all-seeds rule — for n=3 and the recorder's ddof=0 std the
    # extreme deviation is <= sqrt(2)*std, so SEED_SPREAD_FACTOR = 1.5 bounds
    # every seed. On a single-seed pilot `_aggregate` emits no `_std` and the
    # bound collapses to the raw number, which is correct.
    def worst_lo(key):
        return m[key] - SEED_SPREAD_FACTOR * m.get(key + "_std", 0.0)

    def worst_hi(key):
        return m[key] + SEED_SPREAD_FACTOR * m.get(key + "_std", 0.0)

    rig = (worst_lo("n_informative") >= MIN_INFORMATIVE_LIVES
           # Tautological per seed (min over lives selected for clearing the
           # bar), so it is read on the mean as a receipt that the fold ran —
           # NOT bounded to the worst seed, because a conservative bound on a
           # per-seed tautology VOIDs on spread alone. See DEVIATION 2.
           and m["task_dwell_worst_life"] >= TASK_DWELL_MIN
           and worst_hi("random_dwell_worst_life") <= RANDOM_DWELL_MAX
           and RANDOM_COV_LO <= m["coverage_random"] <= RANDOM_COV_HI)
    if not rig:
        return Status.VOID

    std = m.get("delta_coverage_std", 0.0)
    delta_floor = m["delta_coverage"] - SEED_SPREAD_FACTOR * std
    delta_t = m["delta_coverage"] * (3 ** 0.5) / max(std, 1e-9)
    return bool(m["delta_coverage"] >= DELTA_MIN
                and delta_floor > 0.0
                and delta_t >= DELTA_TSTAT_MIN
                and c["delta_shuf"] < DELTA_MIN)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T3.06 gates are provisional — pilot first, freeze the bars in "
            "this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T3.06"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
