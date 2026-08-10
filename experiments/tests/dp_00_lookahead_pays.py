"""DP.00 — This world rewards looking ahead at all.

The cheapest falsifier of the whole fast/slow story, run BEFORE any
dual-process machinery is built. It costs oracle rollouts, not a training
run: the planner is handed **the simulator itself** as its model, so the
question is purely *does lookahead pay in this world*, with learning removed
as a confound.

  The only difference between the arms is PLANNING DEPTH.
  Same code, same model, same world, same tie-break machinery.

That is what makes the gap attributable to lookahead rather than to compute,
samples or a better optimiser — and the control below is what proves it.

WORLD. LC.00's survival gridworld, imported rather than re-typed: 12x12,
five actions, two depleting needs, death on depletion, four food and four
water cells drawn once per seed, drive `d(h)` and reward `d(h) - d(h')`
exactly as LC.00 certified them. This spec changes ONE world constant and
declares it: `LIFE_CAP = 200` (LC.00 uses 400), because a depth-8 planner
holds ~40 s of CPU per seed at this cap and the budget is `cpu<10min`.

ARMS, all five driven by the same `_best_action`:
  reactive_greedy   H=1, uniform tie-break. This is the BEST possible
                    no-lookahead policy given the true reward: depletion is
                    action-independent, so at H=1 every action ties unless a
                    resource is one step away or death is one step away.
                    That is not a strawman, it is the world's own statement
                    that it offers no immediate gradient.
  reactive_persist  H=1, tie-break prefers repeating the previous action —
                    a persistent walker covers more ground than a random one,
                    and covering ground is how a blind agent finds food. A
                    STRENGTHENED null: the reported null is the per-seed
                    MAXIMUM of the two reactive arms, so the test is harder,
                    never easier. (On the calibration seeds persistence in
                    fact HURT, 115.8 vs 129.6 — recorded because a strengthened
                    null that turns out not to be stronger is still evidence.)
  plan_h2/h4/h8     receding-horizon exhaustive search on the true simulator,
                    memoised on (x, y, h0, h1, depth), discount GAMMA=0.95.
                    H_MAX = 8 is NOT "unlimited rollouts" and the sweep says
                    so out loud. On the calibration seeds the gain looked like
                    it was saturating (129.6 -> 147.9 -> 184.0 -> 190.6 for
                    H=1/2/4/8); on the recorded seeds it was still climbing
                    fastest at the end (121.7 -> 125.8 -> 139.2 -> 197.5).
                    So the depth axis is NOT exhausted, and every number here
                    is a lower bound: a deeper planner would widen the gap,
                    which can only strengthen the hypothesis. What the sweep
                    does establish is a dose-response — more lookahead, more
                    life, monotonically — which a compute artifact would not
                    have to produce.

METRIC `return_gap_oracle_plan_vs_reactive` = mean lifespan(plan_h8) - mean
lifespan(best reactive), in steps. Lifespan, not episodic return: return here
TELESCOPES (sum of d(h)-d(h') over a life is d(h_0) - d(h_T) = -d(h_T)), so
an agent that dies at step 100 and one that dies at step 400 score the same.
Lifespan is the consequential quantity and it does not telescope.

ATTAINABLE RANGE, computed before the gate was chosen (LESSONS.md:1784).
Hunger depletes 0.010/step from 1.0, so NOTHING dies before step 100 and
nothing outlives `LIFE_CAP = 200`: lifespan lies in [100, 200] and the gap in
[-100, +100]. The 20-step gate is 20% of that ceiling; the 3.0-sigma gate sits
against a sigma ceiling of about 100/13 = 7.7 at the observed seed spread.
Both arms are CENSORED at the cap — the planner reaches it — so the measured
gap UNDERSTATES the advantage. Censoring can only make this test harder.

CALIBRATION, in the open (LC.00's precedent). Horizons, life counts and the
gate were fixed against seeds **100-102**, which are disjoint from the seeds
`run_spec` uses (0-2); no number below was chosen after seeing a run seed.
Calibration gaps were 40.7 / 86.2 / 56.2 steps, sigma 4.6. Calibration seeds
set the GATES honestly and described the SHAPE wrongly — see the sweep above,
where the saturation visible at 100-102 was absent at 0-2. A number measured
off-run is a design input, never a finding.

CONTROL — a world variant that is provably reactive-solvable, so planning must
NOT gain there. Same grid, same movement, same planner: one beacon, no needs,
no death, no traps, no irreversible states, and a DENSE immediate reward
`0.02 * (dist_before - dist_after)`. Greedy on that signal walks the shortest
path, and the shortest path is optimal, so an H=8 planner can only tie. Three
gates, because a control that cannot be seen failing is not a control
(LESSONS.md:1580):
  ctrl_gain          plan_steps must not beat react_steps by more than 0.5.
  ctrl_react_optimal the reactive arm must hit the shortest path EXACTLY, and
                     the shortest path is derived from the live episode, not
                     from a constant (LESSONS.md:1806).
  ctrl_gain_broken   a deliberately broken reactive arm (uniform random) must
                     produce a gain of >= 10 steps. This is the control's own
                     positive control: it proves the statistic CAN fire.
                     Calibration measured 49 steps against a 10-step floor.

VOID rather than FAIL when the instrument, not the hypothesis, is what failed:
  - the planner's model is not the simulator (2000 probes per seed, on
    resource, death and interior branches, each of which must be exercised —
    a probe that cannot produce the event cannot measure it, LESSONS.md:1410);
  - the reactive arm is not optimal in the world where greedy is provably
    optimal (then "best reactive" is a handicapped arm and every number here
    inherits it);
  - the control's positive control could not fire;
  - the control itself shows a gain (then the metric is measuring compute, not
    lookahead, LESSONS.md:350 — and DP.00's `kills` field takes out the whole
    DP family, so only a hypothesis that was really tested may fire it).

FALSIFIED BY: no gap. Then this world has no slow system to find, DP.01-DP.03
are unregistrable as written, and the finding is about the WORLD — it needs
traps, delays or irreversibility before any dual-process claim can be made in
it. No MuJoCo, no torch, no GPU.
"""
from __future__ import annotations

import random

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .lc_00_gridworld_decidable import (ACTIONS, DEPLETE, GAMMA, SIZE, _drive,
                                        _Life, _World)

IMPL_DEPS = ["experiments/tests/lc_00_gridworld_decidable.py"]

LIFE_CAP = 200            # LC.00 uses 400; halved for the CPU budget, declared
N_LIVES = 10              # lives per arm per seed
H_MAX = 8
SWEEP = (2, 4, 8)         # planning depths reported beside H=1

MIN_GAP = 20.0            # steps, every seed
SIGMA_GATE = 3.0          # the bakeoff.py:159 / LC.00 ruler, unpaired

FLAT_T = 60               # control: episode cap, steps
FLAT_EPISODES = 20
FLAT_MIN_D = 6            # control: minimum start->beacon manhattan distance
FLAT_SHAPE = 0.02
CTRL_TOL = 0.5            # steps the planner may "gain" in the control
CTRL_BROKEN_FLOOR = 10.0  # the broken-null gain that proves the gate can fire

N_PROBES = 2000           # model-fidelity probes per seed
PROBE_TOL = 1e-12

# The floor/ceiling the gate was checked against; asserted so it cannot rot.
LIFE_FLOOR = int(1.0 / DEPLETE[0])          # nothing dies before this step
assert LIFE_CAP > LIFE_FLOOR, "lifespan has no room above its own floor"
assert MIN_GAP < (LIFE_CAP - LIFE_FLOOR), (
    "the gap gate exceeds the attainable range of the statistic")
assert H_MAX in SWEEP, "the gated horizon must be one the run actually measures"


# ---------------------------------------------------------------- survival

def _sim(world, x: int, y: int, h0: float, h1: float, a: int):
    """A PURE mirror of `lc_00._Life.step` — the planner's model.

    Duplicated deliberately (the planner needs a side-effect-free step), and
    the duplication is what `_model_fidelity` exists to police: a planner whose
    model has silently drifted from the simulator is not an oracle planner, and
    every number in this file would be about a world nobody lives in.
    """
    d_before = _drive(h0, h1)
    dx, dy = ACTIONS[a]
    nx = min(SIZE - 1, max(0, x + dx))
    ny = min(SIZE - 1, max(0, y + dy))
    n0 = h0 - DEPLETE[0]
    n1 = h1 - DEPLETE[1]
    dead = n0 <= 0.0 or n1 <= 0.0
    if not dead:
        if (nx, ny) in world.food:
            n0 = 1.0
        if (nx, ny) in world.water:
            n1 = 1.0
    r = d_before - _drive(max(0.0, n0), max(0.0, n1))
    return nx, ny, n0, n1, r, dead


def _action_scores(world, x, y, h0, h1, horizon: int) -> list:
    """Discounted value of each first action under depth-`horizon` search."""
    memo: dict = {}

    def val(x, y, h0, h1, d):
        if d == 0:
            return 0.0
        key = (x, y, round(h0, 6), round(h1, 6), d)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = -1e18
        for a in range(len(ACTIONS)):
            nx, ny, n0, n1, r, dead = _sim(world, x, y, h0, h1, a)
            v = r if dead else r + GAMMA * val(nx, ny, n0, n1, d - 1)
            if v > best:
                best = v
        memo[key] = best
        return best

    out = []
    for a in range(len(ACTIONS)):
        nx, ny, n0, n1, r, dead = _sim(world, x, y, h0, h1, a)
        out.append(r if dead else r + GAMMA * val(nx, ny, n0, n1, horizon - 1))
    return out


def _pick(scores, rng, prev, persist: bool) -> int:
    top = max(scores)
    best = [a for a, v in enumerate(scores) if v >= top - 1e-12]
    if persist and prev in best:
        return prev
    return rng.choice(best)


def _run_arm(seed: int, horizon: int, persist: bool = False) -> float:
    """Mean lifespan over N_LIVES, censored at LIFE_CAP. No learning anywhere:
    tables, weights and counts do not exist in this spec by design."""
    world = _World(seed)
    rng = random.Random(f"dp00-{horizon}-{persist}-{seed}")
    spawn = random.Random(f"dp00-spawn-{seed}")   # identical spawns per arm
    spans = []
    for _ in range(N_LIVES):
        x, y = spawn.randrange(SIZE), spawn.randrange(SIZE)
        h0 = h1 = 1.0
        prev, steps = None, 0
        while steps < LIFE_CAP:
            a = _pick(_action_scores(world, x, y, h0, h1, horizon),
                      rng, prev, persist)
            prev = a
            x, y, h0, h1, _r, dead = _sim(world, x, y, h0, h1, a)
            steps += 1
            if dead:
                break
        spans.append(steps)
    return sum(spans) / len(spans)


# ------------------------------------------------------- model fidelity

def _model_fidelity(seed: int) -> dict:
    """Is the planner's model the simulator? 2000 probes over three branches.

    Counted per branch and reported: a probe set that never eats and never
    dies has checked the interior of the dynamics only, and would certify a
    model that is wrong about exactly the transitions that decide a life.
    """
    world = _World(seed)
    rng = random.Random(f"dp00-probe-{seed}")
    cells = sorted(world.food | world.water)
    mismatch = 0
    ate = died = 0
    for i in range(N_PROBES):
        branch = i % 3
        if branch == 0:                                   # interior
            x, y = rng.randrange(SIZE), rng.randrange(SIZE)
            h0, h1 = rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
        elif branch == 1:                                 # onto a resource
            cx, cy = cells[rng.randrange(len(cells))]
            dx, dy = ACTIONS[rng.randrange(4)]
            x, y = min(SIZE - 1, max(0, cx - dx)), min(SIZE - 1, max(0, cy - dy))
            h0, h1 = rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
        else:                                             # one step from death
            x, y = rng.randrange(SIZE), rng.randrange(SIZE)
            h0 = rng.uniform(0.0, DEPLETE[0])
            h1 = rng.uniform(0.0, 1.0)
        a = rng.randrange(len(ACTIONS))

        life = _Life(world, random.Random(0))
        life.x, life.y, life.h = x, y, [h0, h1]
        r_ref, dead_ref = life.step(a)
        nx, ny, n0, n1, r, dead = _sim(world, x, y, h0, h1, a)
        if (nx, ny) != (life.x, life.y) or dead != dead_ref \
                or abs(r - r_ref) > PROBE_TOL \
                or abs(n0 - life.h[0]) > PROBE_TOL \
                or abs(n1 - life.h[1]) > PROBE_TOL:
            mismatch += 1
        if dead:
            died += 1
        elif n0 == 1.0 or n1 == 1.0:
            ate += 1
    return {"probe_mismatch": float(mismatch),
            "probe_ate": float(ate),
            "probe_died": float(died)}


# ------------------------------------------------------ the control world

def _flat_sim(bx, by, x, y, a):
    """Dense immediate reward, no needs, no death, no irreversible states."""
    d_before = abs(x - bx) + abs(y - by)
    dx, dy = ACTIONS[a]
    nx = min(SIZE - 1, max(0, x + dx))
    ny = min(SIZE - 1, max(0, y + dy))
    d_after = abs(nx - bx) + abs(ny - by)
    return nx, ny, FLAT_SHAPE * (d_before - d_after), d_after == 0


def _flat_scores(bx, by, x, y, horizon):
    memo: dict = {}

    def val(x, y, d):
        if d == 0:
            return 0.0
        key = (x, y, d)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = -1e18
        for a in range(len(ACTIONS)):
            nx, ny, r, done = _flat_sim(bx, by, x, y, a)
            v = r if done else r + GAMMA * val(nx, ny, d - 1)
            if v > best:
                best = v
        memo[key] = best
        return best

    out = []
    for a in range(len(ACTIONS)):
        nx, ny, r, done = _flat_sim(bx, by, x, y, a)
        out.append(r if done else r + GAMMA * val(nx, ny, horizon - 1))
    return out


def _run_flat(seed: int, horizon: int, broken: bool = False):
    """Returns (mean steps to the beacon, mean shortest path).

    The shortest path is read off the live episode rather than assumed, so the
    known answer survives any change to the layout draw.
    """
    rng = random.Random(f"dp00-flat-{horizon}-{broken}-{seed}")
    setup = random.Random(f"dp00-flatsetup-{seed}")   # identical per arm
    taken, shortest = [], []
    for _ in range(FLAT_EPISODES):
        while True:
            bx, by = setup.randrange(SIZE), setup.randrange(SIZE)
            x, y = setup.randrange(SIZE), setup.randrange(SIZE)
            if abs(x - bx) + abs(y - by) >= FLAT_MIN_D:
                break
        shortest.append(abs(x - bx) + abs(y - by))
        steps = 0
        while steps < FLAT_T:
            if broken:
                a = rng.randrange(len(ACTIONS))
            else:
                a = _pick(_flat_scores(bx, by, x, y, horizon), rng, None, False)
            x, y, _r, done = _flat_sim(bx, by, x, y, a)
            steps += 1
            if done:
                break
        taken.append(steps)
    n = len(taken)
    return sum(taken) / n, sum(shortest) / n


# ------------------------------------------------------------- the spec

def _experiment(seed: int) -> dict:
    m = _model_fidelity(seed)
    m["react_greedy"] = _run_arm(seed, 1, persist=False)
    m["react_persist"] = _run_arm(seed, 1, persist=True)
    for h in SWEEP:
        m[f"plan_h{h}"] = _run_arm(seed, h)
    react = max(m["react_greedy"], m["react_persist"])   # strengthened null
    m["react_best"] = react
    m["return_gap_oracle_plan_vs_reactive"] = m[f"plan_h{H_MAX}"] - react
    m["gap_clear"] = float(m["return_gap_oracle_plan_vs_reactive"] >= MIN_GAP)
    return m


def _control(seed: int) -> dict:
    plan, short = _run_flat(seed, H_MAX)
    react, short_r = _run_flat(seed, 1)
    broken, _ = _run_flat(seed, 1, broken=True)
    return {"ctrl_plan_steps": plan,
            "ctrl_react_steps": react,
            "ctrl_broken_steps": broken,
            "ctrl_shortest": short,
            "ctrl_gain": react - plan,
            "ctrl_gain_broken": broken - plan,
            # greedy is provably optimal here, so anything but an exact match
            # means the null arm is handicapped, not that planning won.
            "ctrl_react_optimal": float(abs(react - short_r) < 1e-9)}


def _sigma(mean_a, std_a, mean_b, std_b) -> float:
    return (mean_a - mean_b) / max(std_a, std_b, 1e-9)


def _check(m: dict, c: dict):
    # --- the instrument, before the hypothesis -------------------------
    if m.get("probe_mismatch", 1.0) != 0.0:
        return Status.VOID          # the planner is not planning in this world
    if m.get("probe_ate", 0.0) <= 0.0 or m.get("probe_died", 0.0) <= 0.0:
        return Status.VOID          # the probes never exercised the branches
    if c.get("ctrl_react_optimal", 0.0) != 1.0:
        return Status.VOID          # "best reactive" is a handicapped arm
    if c.get("ctrl_gain_broken", 0.0) < CTRL_BROKEN_FLOOR:
        return Status.VOID          # the control's gate could not have fired

    # --- the control must NOT show a gain ------------------------------
    # VOID, not FAIL: a control that succeeds means the metric is not measuring
    # lookahead (LESSONS.md:350), and DP.00's `kills` field takes out the whole
    # DP family — only a hypothesis that was really tested may fire it.
    if c.get("ctrl_gain", 1e9) > CTRL_TOL:
        return Status.VOID          # the gap is compute, not lookahead

    # --- the hypothesis -------------------------------------------------
    if m.get("gap_clear", 0.0) != 1.0:
        return False                # some seed missed the 20-step margin
    return _sigma(m[f"plan_h{H_MAX}"], m.get(f"plan_h{H_MAX}_std", 0.0),
                  m["react_best"], m.get("react_best_std", 0.0)) >= SIGMA_GATE


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["DP.00"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
