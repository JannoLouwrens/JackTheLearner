"""LC.00 — The learning-core question is decidable in a gridworld first.

A 12x12 survival gridworld with two depleting needs, death on depletion,
uniformly-random respawn and persistent cross-life tables. Four tabular
learning cores — the tabular ancestors of LC.03's arms — must run to
completion, and at least TWO must beat the random null's life_gain by
>= 3 sigma over 3 seeds. The FROZEN control (same agent code path, learning
disabled) must record life_gain within noise of zero; if a frozen agent's
lives lengthen, the world drifts and life_gain measures the world, not the
learner — that is Status.VOID, not FAIL.

Pre-registered design, fixed before the recorded run. Calibration note,
in the open: a first parameterisation (depletion 0.020/0.015, 3+3 cells,
30 lives, 8 buckets) killed every agent at ~50 steps — before a random
walk could ever find food, so no reward was ever observed and nothing
could learn. That is the T1.02 lesson (when the simplest learner also
fails, the TASK is broken), and the world was recalibrated below so that
learning is possible; the 3-sigma gate and the >= 2-cores threshold were
never touched. Calibration necessarily ran on the same seeds the spec
runs (run_spec uses seeds 0..2) — mitigated only by the observed margins
being wide (9.1 / 6.3 / 4.1 sigma against a 3.0 gate), which is stated
here rather than hidden.

  World   12x12, actions {N,S,E,W,REST}. Needs h = (h0, h1) in [0,1]^2,
          full at (re)spawn. Depletion per step: 0.010 / 0.008. Four food
          cells restore h0 to 1.0, four water cells restore h1 to 1.0
          (renewable, positions drawn once per seed, identical across all
          cores of that seed). Death when either need reaches 0; life is
          capped at 400 steps (truncation, not termination). 120 lives
          per core per seed; respawn is a uniformly random cell (never a
          fixed good state — LEARNING_CORE.md W0-2).
  State   (x, y, bucket4(h0), bucket4(h1)). Four need buckets, not
          PS.00's eight: 120 lives x ~150 steps is ~18K transitions and
          144 x 16 x 5 = 11,520 (s,a) cells; at eight buckets the table
          outnumbers the experience and tabular learning cannot converge.
  Drive   d(h) = (sum_i (1 - h_i)^4)^(1/2)  — Keramati-Gutkin, n=4, m=2
          (PURPOSE_AND_SCAFFOLDING.md 2.2 defaults).
  Reward  r = d(h) - d(h')  — plain drive reduction, the unique
          self-termination-safe form (NEEDS_AND_DEATH.md 0.2(d)); never
          clipped, floored or one-sided (0.2(e)); gamma = 0.95 < 1 is
          load-bearing, not a hyperparameter (0.2(f)). Death is plain
          termination with no hand-tuned penalty — the terminal transition
          already charges the full remaining deviation.
  Cores   q_drive   tabular Q-learning (alpha 0.2, gamma 0.95); epsilon-
                    greedy with epsilon decaying linearly 0.25 -> 0.05
                    across the 120 lives, so the final third measures
                    exploitation of what was learned, not exploration
                    noise. All learning cores share the schedule; the
                    frozen control ignores it (it is always random).
          q_lp      the same plus absolute learning progress: intrinsic
                    bonus 0.5 * |delta EMA(transition-prediction error)|
                    per (s,a), EMA lambda 0.1 (Oudeyer-style ALP on a
                    count-based transition model).
          model_vi  tabular maximum-likelihood transition + reward model,
                    20 asynchronous one-step backups per env step sampled
                    from visited (s,a) — Dyna-style asynchronous value
                    iteration in the learned model.
          model_efe the same learned model, backups scored by expected
                    free energy: pragmatic term = observed drive-reduction
                    reward (ln C = -d(h)), epistemic term = Dirichlet
                    information-gain proxy 0.05 / (1 + N(s,a)) — kappa is
                    sized to the reward scale (per-step drive deltas are
                    ~0.001-0.1); at 0.5 the bonus dominates the objective
                    and the agent is a novelty seeker, not a survivor.
          All tables (Q, model counts, visit counts) persist across lives
          within a seed and reset across seeds — cross-life learning is
          the thing being measured.
  Metric  life_gain = mean(survival steps, final 40 lives)
                    - mean(survival steps, first 40 lives), per core per
          seed. A core clears the null iff
            (mean_core - mean_null) / max(std_core, std_null, 1e-9) >= 3.0
          across the 3 seeds — the same sigma unit as bakeoff.py:159, so
          LC.03/LC.04 inherit a ruler that already exists.
  Control FROZEN: the q_drive code path with learning disabled (all-zero Q,
          uniform tie-break). Within noise of zero means
            |mean_frozen| <= 3 * max(std_frozen, 1e-9)
          AND frozen does not itself clear the 3-sigma gate against the
          null. Its numeric life_gain is recorded for LC.03/LC.04 to reuse
          as their own control threshold rather than inventing a new one.

No MuJoCo, no torch, no GPU. Verdicts: >= 2 cores clear -> PASS. Fewer than
two clear -> FAIL (the metric is wrong or the world is unlearnable, and
LC.03 onward must not run). Frozen control shows life gain -> Status.VOID
(the instrument is broken; nothing was tested).
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

SIZE = 12
ACTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))  # N,S,E,W,REST
DEPLETE = (0.010, 0.008)
N_FOOD = 4
N_LIVES = 120
LIFE_CAP = 400
BUCKETS = 4
GAMMA = 0.95
ALPHA = 0.2
EPS_HI = 0.25
EPS_LO = 0.05
LP_BETA = 0.5
LP_LAMBDA = 0.1
PLAN_BACKUPS = 20
EFE_KAPPA = 0.05
SIGMA_GATE = 3.0
MIN_CORES_CLEARING = 2

CORES = ("q_drive", "q_lp", "model_vi", "model_efe")


def _drive(h0: float, h1: float) -> float:
    return ((1.0 - h0) ** 4 + (1.0 - h1) ** 4) ** 0.5


class _World:
    """Food/water layout fixed per seed; identical across every core."""

    def __init__(self, seed: int):
        rng = random.Random(f"lc00-world-{seed}")
        cells = [(x, y) for x in range(SIZE) for y in range(SIZE)]
        picks = rng.sample(cells, 2 * N_FOOD)
        self.food = frozenset(picks[:N_FOOD])    # restores h0
        self.water = frozenset(picks[N_FOOD:])   # restores h1


class _Life:
    """One life: spawn to death or cap. Mutates nothing outside itself."""

    def __init__(self, world: _World, rng: random.Random):
        self.w = world
        self.x, self.y = rng.randrange(SIZE), rng.randrange(SIZE)
        self.h = [1.0, 1.0]

    def state(self):
        b0 = min(BUCKETS - 1, int(self.h[0] * BUCKETS))
        b1 = min(BUCKETS - 1, int(self.h[1] * BUCKETS))
        return (self.x, self.y, b0, b1)

    def step(self, a: int):
        """Returns (reward, dead). Reward is exact drive difference."""
        d_before = _drive(*self.h)
        dx, dy = ACTIONS[a]
        self.x = min(SIZE - 1, max(0, self.x + dx))
        self.y = min(SIZE - 1, max(0, self.y + dy))
        self.h[0] -= DEPLETE[0]
        self.h[1] -= DEPLETE[1]
        dead = self.h[0] <= 0.0 or self.h[1] <= 0.0
        if not dead:
            if (self.x, self.y) in self.w.food:
                self.h[0] = 1.0
            if (self.x, self.y) in self.w.water:
                self.h[1] = 1.0
        h0 = max(0.0, self.h[0])
        h1 = max(0.0, self.h[1])
        return d_before - _drive(h0, h1), dead


class _Core:
    """All four cores plus FROZEN share this code path; `kind` selects the
    update rule, so the frozen control exercises exactly what the arms do."""

    def __init__(self, kind: str, rng: random.Random):
        self.kind = kind
        self.rng = rng
        self.q = defaultdict(float)                    # (s,a) -> value
        self.n_sa = defaultdict(int)                   # (s,a) visit counts
        self.trans = defaultdict(lambda: defaultdict(int))  # (s,a) -> {s': n}
        self.r_sum = defaultdict(float)                # (s,a) reward sums
        self.err_ema = {}                              # (s,a) -> EMA pred error
        self.seen = []                                 # visited (s,a) list
        self.learning = kind != "frozen"
        self.eps = EPS_HI

    def act(self, s) -> int:
        if not self.learning or self.rng.random() < self.eps:
            return self.rng.randrange(len(ACTIONS))
        vals = [self.q[(s, a)] for a in range(len(ACTIONS))]
        best = max(vals)
        return self.rng.choice([a for a, v in enumerate(vals) if v == best])

    def _greedy_v(self, s) -> float:
        return max(self.q[(s, a)] for a in range(len(ACTIONS)))

    def _backup_reward(self, sa) -> float:
        r_hat = self.r_sum[sa] / self.n_sa[sa]
        if self.kind == "model_efe":
            return r_hat + EFE_KAPPA / (1 + self.n_sa[sa])
        return r_hat

    def update(self, s, a, r: float, s2, dead: bool) -> None:
        if not self.learning:
            return
        sa = (s, a)
        if self.n_sa[sa] == 0:
            self.seen.append(sa)
        self.n_sa[sa] += 1
        self.r_sum[sa] += r

        bonus = 0.0
        if self.kind in ("q_lp", "model_vi", "model_efe"):
            succ = self.trans[sa]
            if self.kind == "q_lp":
                predicted = max(succ, key=succ.get) if succ else None
                err = 0.0 if predicted == s2 else 1.0
                prev = self.err_ema.get(sa, 1.0)
                new = prev + LP_LAMBDA * (err - prev)
                self.err_ema[sa] = new
                bonus = LP_BETA * abs(new - prev)
            succ[s2] += 1

        target = (r + bonus) + (0.0 if dead else GAMMA * self._greedy_v(s2))
        self.q[sa] += ALPHA * (target - self.q[sa])

        if self.kind in ("model_vi", "model_efe"):
            for _ in range(PLAN_BACKUPS):
                psa = self.seen[self.rng.randrange(len(self.seen))]
                succ = self.trans[psa]
                total = sum(succ.values())
                if total == 0:
                    continue
                ev = sum(n * self._greedy_v(ps2) for ps2, n in succ.items())
                self.q[psa] += ALPHA * (
                    self._backup_reward(psa) + GAMMA * ev / total - self.q[psa])


def _run_core(kind: str, seed: int) -> float:
    """120 lives; tables persist across lives. Returns life_gain in steps."""
    world = _World(seed)
    core = _Core(kind, random.Random(f"lc00-{kind}-{seed}"))
    spawn_rng = random.Random(f"lc00-spawn-{kind}-{seed}")
    spans = []
    for li in range(N_LIVES):
        core.eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO) * li / N_LIVES)
        life = _Life(world, spawn_rng)
        steps = 0
        while steps < LIFE_CAP:
            s = life.state()
            a = core.act(s)
            r, dead = life.step(a)
            steps += 1
            core.update(s, a, r, life.state(), dead)
            if dead:
                break
        spans.append(steps)
    third = N_LIVES // 3
    return sum(spans[-third:]) / third - sum(spans[:third]) / third


def _experiment(seed: int) -> dict:
    m: dict = {}
    for kind in CORES:
        m[f"lg_{kind}"] = _run_core(kind, seed)
    m["lg_null"] = _run_core("frozen", seed)  # uniform random via frozen path
    m["cores_completed"] = float(len(CORES))
    return m


def _control(seed: int) -> dict:
    # Distinct RNG stream from the experiment's null on purpose: two
    # independent draws of a no-learning agent bound the noise floor.
    return {"lg_frozen": _run_core("frozen", seed + 10_000),
            "lg_null": _run_core("frozen", seed)}


def _sigma(mean_a, std_a, mean_b, std_b) -> float:
    return (mean_a - mean_b) / max(std_a, std_b, 1e-9)


def _check(m: dict, c: dict):
    if m.get("cores_completed", 0.0) != float(len(CORES)):
        return Status.VOID

    frozen_gain = _sigma(c["lg_frozen"], c.get("lg_frozen_std", 0.0),
                         c["lg_null"], c.get("lg_null_std", 0.0))
    frozen_nonzero = abs(c["lg_frozen"]) > 3.0 * max(
        c.get("lg_frozen_std", 0.0), 1e-9)
    if frozen_gain >= SIGMA_GATE or frozen_nonzero:
        # Lives lengthened without learning: the world drifts and the
        # metric measures the world. Nothing here tested the claim.
        return Status.VOID

    clearing = sum(
        1 for kind in CORES
        if _sigma(m[f"lg_{kind}"], m.get(f"lg_{kind}_std", 0.0),
                  m["lg_null"], m.get("lg_null_std", 0.0)) >= SIGMA_GATE)
    return clearing >= MIN_CORES_CLEARING


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LC.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
