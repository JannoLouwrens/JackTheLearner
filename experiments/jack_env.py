"""jack_env.py — his world behind the standard interface (ORDERS J1 step 1).

PLAN stage 1, step 2 (combine). Every published learner (SB3 SAC, DreamerV3's
gym wrapper, CleanRL) plugs into `gymnasium.Env`; until W0 is one, neither
brain of the plan can touch his world (PUBLISHED_PRIORS.md §1).

WHAT IT IS. A thin wrapper around `w0.W0` (lethal, drives.py's needs, the W0-4
senses). It WRAPS `w0.py` and `drives.py`; it edits neither, and it adds no
rule to the world. Default options reproduce today's world exactly — the same
world `survival.run_survival` lives in.

  action       Box(-1, 1, (8,)) float32 — W0.decide's 8-vector unchanged
               (4 arm slides, 2 adhesion, 2 gated horizontal drive).
  observation  Box(-inf, inf, (N,)) float32 — his senses flattened in W0-4
               order: vision, audio, touch, proprio, needs. NOT included:
               `language` (present-and-dropped in W0; zeros) and the placebo
               channel (a noise probe for controls, not a sense). W0.observe()
               is still called once per decision exactly as the old loop calls
               it, because the placebo draw consumes the world's RNG.
  step         one 0.2 s decision (W0.SUBSTEPS x timestep).
  reward       homeo-dr, as survival.py: d(h_t) - d(h_{t+1}); a death
               charges d_before - survival.D_DEATH. Options: see OPTIONS.
  terminated   True on the decision he dies (W0 lethal). Death only.
  truncated    False by default; True when the `max_episode_decisions`
               administrative cap is hit (OPTIONS). Never on a death step.
  info         meals (total eat events, all foods, cumulative over the world),
               ate_total (per food, cumulative), meals_this_life, death_cause
               ("" alive / "energy" / "integrity"), life, decisions, sim_s.
               Drinks: today's W0 (drives.py, 3 needs e/i/w) has no drinking
               event of its own — `ate_total` is passed through verbatim, so a
               drink appears there the day the world gains one.

LIVES. The world persists across deaths, as in every loop here: W0 respawns
him in the SAME world (regrowth timers, diary carried). So `reset()` after a
death does NOT rebuild the world — it hands back the respawned body's
observation. `reset(seed=s)` (or the first reset) builds a fresh W0(seed=s).
That is what makes a run of lives through this env the same run as
`run_survival`'s.

PROOF IT IS THE SAME WORLD: experiments/tests/test_jack_env.py —
`check_env` passes, and fixed-seed random-action runs through the wrapper
match the same runs through `survival.run_survival(policy="random")`
decision for decision (life ends, eat events). Numbers: see SPEED / PROOF
below, filled from measured runs only.

OPTIONS (ORDERS J2 items 3 and 4; every default = today's world, proof:
experiments/tests/test_jack_env_options.py). Ours beside theirs:

  reward="homeo-dr"      default, above. Unchanged.
  reward="dr100-eat"     r = 100*(d_t - d_{t+1}) + eat_pay * (eat events this
                         decision); a death charges 100*(d_before - D_DEATH).
                         Theirs: EVAAA main text pays drive reduction
                         sum_i beta_i (D_i(t) - D_i(t+1)); EVAAA's released code
                         has a branch paying +100 on consuming a resource
                         (evaaa_train/envs/interoceptiveAI.py L177-204;
                         PUBLISHED_PRIORS §3 long form line 70). DIFFERS: we
                         pay BOTH (the order's recipe), one beta (100) over
                         our 3 needs, eat_pay default 100 = their branch.
  reward="dr100-eat-pos" as dr100-eat, but an ALIVE decision is never
                         negative: the drive term is max(0, .) while alive
                         (being alive and hungry costs nothing); a death
                         still charges 100*(d_before - D_DEATH) < 0. Theirs:
                         §3 "remove any way to profit from ending a life";
                         a reward paying on intake/drive reduction has no
                         sign problem (long form line 161). DIFFERS: no
                         published world uses exactly this clip; it is the
                         order's "never negative while alive" option.
  max_episode_decisions  None (default) = no cap. N = after N decisions
                         since the last reset() with no death, truncated=True.
                         The world is NOT touched: reset() after a
                         truncation hands back that step's observation (no
                         extra observe(), so the world RNG stream is the
                         uncapped run's) and the life continues (life,
                         meals_this_life carry on). Theirs: §5 "death is
                         terminated=True, every administrative stop
                         truncated=True"; §8 "the 3,000-decision cap is a
                         truncation"; Crafter caps at 10,000 steps. DIFFERS:
                         Crafter's cap builds a new world; ours keeps it
                         (lives persist in W0).
  difficulty=None       J2 item 1 (meal size, regrowth delay, spawn
                         distance, food count). None
                         or 1.0 = today's world, untouched. At d in [0,1):
                         floor-food meal nu(d) = 1.0 + d*(0.08 - 1.0) (a
                         full bar at d=0) and regrowth delay d*66.9 s
                         (instant at d=0, floored at the rest of the eating
                         decision, <= 0.2 s). The apple is not dialled.
                         Spawn distance: birth and every respawn are drawn
                         uniformly from W0's legal cells within r(d) = 0.5 +
                         17*d m of the nearest floor food's current x, y
                         (nearest legal cell if none is that close). DIFFERS:
                         Avalon puts the food 0.5 m IN FRONT; W0 spawns
                         every body facing one fixed yaw, so ours is a
                         distance, not a heading; the linear map is ours.
                         Food count: W0's non-food floor objects obj2, obj3,
                         obj4 become floor food (same meal and regrowth
                         dial) — the first round-half-up 3*(1-d) of them
                         that the seed's world carries (W0 draws 4 or 6
                         objects per seed; seed 155 and seeds 201-202 have
                         no obj4): up to 5 floor foods at d=0, 4 at d<1/2,
                         3 at d<5/6, 2 above.
                         Each keeps W0's own per-geom ray texture, so to
                         his eyes each food is its own stable texture, as
                         obj0/obj1 already are. Counted in `meals` and
                         `ate_total` (world_card's floor_meals still counts
                         obj0+obj1 only — J2 item 5 must count all).
                         DIFFERS: Avalon raises food count by placing new
                         items; ours re-labels objects already in the room
                         (no world file), at their W0 places.
                         FOOD-COUNT PROOF (2026-11-15, `python -m
                         experiments.tests.test_jack_env_foodcount 400`,
                         seeds 153-155): d=1 == None byte for byte, food set
                         unchanged; at d=0/0.25/0.6/0.9 the food set, meal
                         and regrowth are as stated; extras eaten 4 times
                         (seed 153); control with extras stripped: eaten 0
                         times, stream differs. PASS. Seen there: at d=0
                         (instant regrowth) a body resting on a food eats
                         every decision — seed 153 stripped ate obj1 84
                         times in 400 decisions, with extras 20 — so d=0
                         "meals" partly counts contact time (J2 item 5 must
                         read that). Spawn proof re-run against all floor
                         foods PASS; dial proof PASS (pooled d=0 eats 31).
                         SPAWN PROOF (2026-11-15, `python -m experiments.
                         tests.test_jack_env_spawn 400`, seeds 153-155,
                         e0=0.05): every d=0 spawn 0.25-0.45 m from food,
                         every d=0.25 spawn <= 4.59 m (bound 4.75); control:
                         today's world breaks the 0.5 m bound on 12/12
                         spawns (0.91-6.22 m). PASS. test_jack_env_dial re-run
                         PASS (d=1 byte-identical; pooled floor eats d=0 17,
                         d=0.5 3, d=1 3 in 3x400 random decisions).
                         Theirs: Avalon eat, one ceiling d in [0,1] per task,
                         food 0.5 m in front at d=0 so "even random agents
                         succeed"; Avalon food restores 100% of the bar
                         (§3 table); two-resource ant: instant respawn at
                         10% meals (trp_env). DIFFERS: Avalon dials distance
                         and food count, not meal size or regrowth (no
                         survival world ablates those alone, §3 "Not
                         verified"); the linear map is ours.
  random_needs=None     J2 item 2 (part). None = today (birth and every
                         respawn at e=e0, i=1, w=0). A float s in (0, 1]:
                         at birth and at EVERY respawn, energy e and
                         integrity i are each drawn uniform in [1-s, 1]
                         (their distance from the setpoint up to s of the
                         distance to death; w, not lethal, stays 0). Drawn
                         from the env's own RNG seeded by (seed, 2), never
                         W0's, so the world's streams are untouched. Needs
                         e0 == 1.0 (it replaces e0). Theirs: trp_env draws
                         each internal state uniform within +/-1/6 of the
                         setpoint at every reset (two_resource_env.py L88,
                         L297-300; long form line 341) -> s = 1/6 is the
                         published value (NEEDS_S_TRP). DIFFERS: trp_env's
                         needs are two-sided (too full also kills), so it
                         draws both sides; ours are one-sided (full is the
                         setpoint), so only the hungry side is drawn. The
                         needs sense's change-since-last reads 0 at a drawn
                         start (W0 would report a fake drop from full).
                         NEEDS PROOF (2026-11-15, `python -m experiments.
                         tests.test_jack_env_needs 600`, seeds 153-155, 3
                         forced respawns each): None == default byte for
                         byte; s=1/6 starts e 0.842-0.952, s=0.5 e
                         0.526-0.857, all in bounds, every life a new draw;
                         W0's RNG states equal the default's; control (s=0.5)
                         stream differs. PASS.
  random_layout=False   J2 item 2 (rest). False = today (foods lie where
                         W0 put them and where eating shoves them). True: at
                         birth and at EVERY respawn, every floor food (the
                         dial's extras too) is re-placed at a uniformly random
                         cell of W0's body-legal spawn grid (0.44 m pitch), at
                         its own W0 rest height and pose, >= LAYOUT_SEP 0.5 m
                         from the other foods, touching nothing but ground;
                         then legal cells within 0.8 m of a food are re-tested
                         with W0's own penetration test and dropped if the body
                         would overlap a food; then the body is drawn by
                         today's rule (uniform) or the dial's. Food cells come
                         from the env's own stream (seed, 3); the body's from
                         W0's spawn stream. Other objects never move.
                         Theirs: §3 "every life is a new layout; no level
                         repeats" (Crafter, Avalon, Neural MMO). DIFFERS: they
                         build a new world each life; ours moves only the food
                         inside the persistent W0 room (walls, decoys, apple
                         and clocks stay), on W0's grid, not continuous; the
                         grid is the one computed against the ORIGINAL food
                         places, so cells next to those stay excluded. With
                         the dial at d=0 the body lands exactly one grid cell
                         (0.44 m) from a food every time (the proof's 12/12).
                         LAYOUT PROOF (2026-11-16, `python -m experiments.
                         tests.test_jack_env_layout 600`, seeds 153-155, 3
                         forced respawns each, 3.0 CPU-min): False == default
                         byte for byte; on (alone and with d=0) every life's
                         foods on grid, at rest height, >= 0.5 m apart, body
                         not penetrating, 0 placement failures, 4/4 layouts
                         distinct per run; same seed same layout, next seed
                         differs; control: the same check on today's world
                         FAILS on all 3 seeds, layout stream differs. PASS.
  terminal_obs           "reborn" (default, today's) | "dying" (J9). On the
                         decision he dies, "reborn" returns the RESPAWNED
                         body's first senses with terminated=True; "dying"
                         returns the DYING body's (its place, its empty need,
                         its last sounds), captured by `W0.observe_pure()`
                         before the respawn, and reset() hands out the reborn
                         one exactly as before. The world's random stream is
                         not touched: every other observation, reward and
                         death equals the default run's. TERMINAL PROOF
                         (2026-11-16, `python -m experiments.tests.
                         test_jack_env_terminal 300`, seeds 153-155): (1) J1
                         identity vs run_survival, OFF and ON, e0 1.0 and
                         0.05 (deaths @98/206, 99/205, 105/213): 12/12 MATCH,
                         controls differ; (2) ON stream == OFF stream on
                         every non-terminal step and reset, 6 deaths, each
                         OFF terminal obs == ON next reset(); (3) ON terminal
                         proprio+needs == a draw-free spy on W0.respawn, e 0.0
                         at death vs 0.05 reborn; control (OFF terminal)
                         fails on all 3. PASS. Place is carried by the dying
                         body's own senses (rays, proprio); x, y is no sense.
  move_cost_dial=False  J11 (1). False = today (every joule of the body's
                         mechanical work costs drives.KAPPA of energy at
                         every d). True: this world charges dial_move_cost(d)
                         * KAPPA — 0 at and below the entry rung d0 = 0.75
                         (D0_ENTRY; moving drains nothing beyond basal),
                         linear in (d - d0)/(1 - d0), the full KAPPA at
                         d = 1; with difficulty None (= 1) nothing is
                         patched. Per instance: drives.py and the module
                         constant are untouched; `last_power_w` still reports
                         the true power. Theirs: no published survival world
                         charges for moving (§13); the cure proven on the
                         lie-still trap is a cost ramped in (Hwangbo 2019:
                         factor from 0.3, k <- k^0.997; Solo12: 0 -> 1;
                         Kwiatkowski 2023). DIFFERS: theirs ramps over
                         TRAINING TIME inside one run; ours ties the cost to
                         the world's difficulty dial (a fixed world per run),
                         from 0 not 0.3, and is a reward-free WORLD change
                         (energy), not a reward term.
  spawn_clear=None      J11 (1). None = today. R (m): no birth or rebirth
                         with the body's root within R (x, y) of any floor
                         food where it lies at that moment — the legal cells
                         inside R are dropped before today's draw (uniform,
                         the dial's, or after the layout's food placement;
                         same W0 spawn stream). The apple is on its platform
                         and is not counted. Why: J8's floor arms crossed the
                         sustained mark 5 times in 15 runs, every time a
                         do-nothing body born touching a food and fed by
                         contact at every regrowth (PILOT_B.md). R to use =
                         SPAWN_CLEAR_R 1.55 m, MEASURED 11-17
                         (/data/jack-logs/j11/reach_probe.py, seeds 381-382,
                         180 do-nothing lives of 250 decisions spawned within
                         1.6 m of a food in the pilot world): the body tips
                         and slides a median 0.71 m, max 1.07 m (all of it in
                         the first 100 decisions for the median life); 5
                         lives ate by contact, all at decisions 9-10, from
                         spawns 0.44 m (4 of 28) and 0.88 m (1 of 17), 0 of
                         123 from >= 0.98 m. R = 1.07 + torso bound 0.27 +
                         largest food bound 0.175 = 1.52, rounded up. Theirs:
                         Avalon starts food 0.5 m in FRONT, never in contact;
                         no published world feeds a motionless body (§13).
                         DIFFERS: with the dial below d ~ 0.06 (r(d) < R) the
                         dial's "nearest cell" fallback applies, so d = 0
                         then means "as near as the clearance allows".
  CURES PROOF (measured 2026-11-17, `python -m
  experiments.tests.test_jack_env_cures`, seeds 153-155): see RESULT in that
  file's docstring.
  OPTIONS PROOF (measured 2026-11-15, `python -m
  experiments.tests.test_jack_env_options 300`, seeds 153/154/155): (a)
  explicit defaults == JackEnv() byte for byte; (b) dr100-eat == 100*homeo-dr
  + 100*eats on every decision (eats on 153/155); (c) pos: all alive
  decisions >= 0, deaths < 0, plain dr100-eat had negative alive ones; (d)
  K=40: 6 truncations + 2 deaths per seed, observations/rewards/deaths equal
  the uncapped run, meals_this_life carries; control stream differs. PASS.
  The J1 proof (test_jack_env) re-run after the change: PASS, unchanged.
  Still owed for J2: item 5 (measure d0).

PROOF (measured 2026-11-15, `python -m experiments.tests.test_jack_env 300`):
  check_env PASS. Seeds 153/154/155 x 300 random decisions, against
  run_survival(policy="random"), identical: e0=1.0 eat events
  {obj1:1} / {} / {obj0:1,obj1:1}, reward sums -0.034212046 / -0.069107120 /
  -0.048612778; e0=0.05 deaths at decisions [98,206] / [99,205] / [105,213].
  Control (action stream seed+1) differs on all 6. Caveat: identity holds for
  float64 actions; a float32 action (what SB3 emits) rounds the 8-vector and
  drifts the reward sum by ~1e-9..4e-3 in 300 decisions — a different action,
  not a different world.
  Observation width 74. By default the terminal observation of a death step
  is the RESPAWNED body (W0 respawns inside decide); `terminal_obs="dying"`
  (J9) returns the dying body's instead.
SPEED (measured): one env on the box, nice 19, 400 random decisions:
  38.8 decisions/s wall (38.9 per CPU-second) -> 1M decisions ~7.2 h on one
  core. 4-process vector and the Kaggle CPU figure: still owed (J1 step 3).
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments import drives
from experiments.survival import D_DEATH

SENSES = ("vision", "audio", "touch", "proprio", "needs")
REWARDS = ("homeo-dr", "dr100-eat", "dr100-eat-pos")
TERMINAL_OBS = ("reborn", "dying")
INTEGRATORS = ("shipped", "implicitfast")  # J6(4); shipped = RK4
FLOOR_FOODS = ("obj0", "obj1")
# J2 item 1, food count: W0's own non-food floor objects that the dial turns
# into floor food at low d, in this order (obj5 is never food).
EXTRA_FOODS = ("obj2", "obj3", "obj4")
NEEDS_S_TRP = 1.0 / 6.0  # trp_env: start state uniform within +/-1/6 of setpoint
MEAL_AT_D0 = 1.0     # Avalon: a food fills the bar (PUBLISHED_PRIORS §3 table)
SPAWN_R_D0 = 0.5     # m; Avalon eat: food 0.5 m in front of the agent at d=0
SPAWN_R_SPAN = 17.0  # m; ~ the arena diagonal (2*sqrt(2)*6): r(1) covers all
D0_ENTRY = 0.75      # ruling 22: the entry rung; move_cost_dial charges 0 there
# J11: the contact-eating reach of a body that does nothing (measured, see
# OPTIONS spawn_clear): it tips and slides <= 1.07 m, + torso 0.27 + food 0.175.
SPAWN_CLEAR_R = 1.55  # m


# J2 item 2, random layout each life (see OPTIONS).
LAYOUT_TRIES = 50          # random cells tried per food before it stays put
LAYOUT_SEP = 0.5           # m between floor foods (no stacked foods)
LAYOUT_BODY_CHECK = 0.8    # m: legal cells this near a food are re-tested


def dial_spawn_radius(d: float) -> float:
    """Spawn-to-nearest-floor-food radius at dial d: 0.5 m at d=0, linear."""
    return SPAWN_R_D0 + d * SPAWN_R_SPAN


def dial_extra_foods(d: float) -> int:
    """Extra floor foods at dial d: 3 at d=0 (5 in all), 0 at d=1, linear,
    rounded half up: d<1/6 -> 3, <1/2 -> 2, <5/6 -> 1, else 0."""
    return int(np.floor(len(EXTRA_FOODS) * (1.0 - d) + 0.5))


def dial_move_cost(d: float, d0: float = D0_ENTRY) -> float:
    """J11 `move_cost_dial`: the share of drives.KAPPA charged at dial d —
    0 at and below the entry rung d0, 1 at d = 1, linear between."""
    return float(min(1.0, max(0.0, (d - d0) / (1.0 - d0))))


def dial_meal(nu1: float, d: float) -> float:
    """Meal size at dial d: today's nu1 at d=1, a full bar at d=0, linear."""
    return MEAL_AT_D0 + d * (nu1 - MEAL_AT_D0)


def default_cal() -> dict:
    """The j0/alpha every forager/LC run uses (borrowed LC.09 calibration)."""
    from experiments.forager_race import _cal
    c = _cal()
    return {"j0": c["j0"], "alpha": c["alpha"]}


class JackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, j0: Optional[float] = None,
                 alpha: Optional[float] = None, seed: int = 0,
                 e0: float = 1.0, diary=None, reward: str = "homeo-dr",
                 eat_pay: float = 100.0,
                 max_episode_decisions: Optional[int] = None,
                 difficulty: Optional[float] = None,
                 random_needs: Optional[float] = None,
                 random_layout: bool = False,
                 terminal_obs: str = "reborn",
                 lazy_contact_audio: bool = False,
                 multiray_vision: bool = False,
                 integrator: str = "shipped",
                 move_cost_dial: bool = False,
                 spawn_clear: Optional[float] = None):
        # e0: birth AND respawn energy, exactly survival.run_survival's e0.
        if j0 is None or alpha is None:
            c = default_cal()
            j0 = c["j0"] if j0 is None else j0
            alpha = c["alpha"] if alpha is None else alpha
        self.j0, self.alpha, self.e0, self.diary = float(j0), float(alpha), float(e0), diary
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        self.reward, self.eat_pay = reward, float(eat_pay)
        self.max_episode_decisions = (None if max_episode_decisions is None
                                      else int(max_episode_decisions))
        if difficulty is not None and not 0.0 <= float(difficulty) <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {difficulty!r}")
        # None and 1.0 are both today's world: nothing is touched.
        self.difficulty = (None if difficulty is None or float(difficulty) == 1.0
                           else float(difficulty))
        if random_needs is not None:
            if not 0.0 < float(random_needs) <= 1.0:
                raise ValueError(f"random_needs must be in (0, 1], got {random_needs!r}")
            if float(e0) != 1.0:
                raise ValueError("random_needs replaces e0; leave e0 = 1.0")
        self.random_needs = None if random_needs is None else float(random_needs)
        self.random_layout = bool(random_layout)
        if terminal_obs not in TERMINAL_OBS:
            raise ValueError(f"terminal_obs must be one of {TERMINAL_OBS}, "
                             f"got {terminal_obs!r}")
        self.terminal_obs = terminal_obs
        # J6(2a), default OFF: identical audio events, force computed only
        # for a contact pair that fires (experiments/fast_sim.py).
        self.lazy_contact_audio = bool(lazy_contact_audio)
        # J6(3), default OFF: the 16 vision rays in one mj_multiRay call,
        # identical observations (experiments/tests/test_multiray.py).
        self.multiray_vision = bool(multiray_vision)
        # J6(4), default OFF: "implicitfast" swaps MuJoCo's integrator (the
        # world ships RK4). A WORLD CHANGE, not an identity: measured in
        # experiments/tests/test_integrator.py. W0.__init__ takes no physics
        # step, so setting it after construction equals a native build.
        if integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}, "
                             f"got {integrator!r}")
        self.integrator = integrator
        # J11 (1), default OFF: the movement drain ramped in with the dial.
        self.move_cost_dial = bool(move_cost_dial)
        # J11 (1), default None: no birth or rebirth within R m of a floor food.
        if spawn_clear is not None and not float(spawn_clear) > 0.0:
            raise ValueError(f"spawn_clear must be > 0 m, got {spawn_clear!r}")
        self.spawn_clear = None if spawn_clear is None else float(spawn_clear)
        self.spawn_clear_failed = 0
        self._ep_decisions = 0
        self._truncated = False
        self._seed = int(seed)
        self.w = None
        self._build(self._seed)
        o = self._obs(self.w.observe())
        self._pending_obs = o
        self.action_space = spaces.Box(-1.0, 1.0, (self.w.action_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, o.shape, np.float32)
        self._died = False

    def _build(self, seed: int) -> None:
        from experiments.w0 import W0
        sampler = None
        if self.random_layout:
            sampler = self._layout_spawn
        elif self.difficulty is not None:
            sampler = self._near_food_spawn
        elif self.spawn_clear is not None:
            sampler = self._clear_spawn
        self.w = W0(seed=seed, j0=self.j0, alpha=self.alpha, lethal=True,
                    diary=self.diary, spawn_sampler=sampler)
        self.w.capture_dying_obs = self.terminal_obs == "dying"
        if self.lazy_contact_audio:
            from experiments.fast_sim import apply_lazy_contact_audio
            apply_lazy_contact_audio(self.w)
        if self.multiray_vision:
            from experiments.fast_sim import apply_multiray_vision
            apply_multiray_vision(self.w)
        if self.integrator == "implicitfast":
            self.w.model.opt.integrator = self.w.mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.move_cost_scale = 1.0
        if self.move_cost_dial and self.difficulty is not None:
            self.move_cost_scale = dial_move_cost(self.difficulty)
            if self.move_cost_scale != 1.0:
                self._apply_move_cost(self.move_cost_scale)
        self.floor_foods = FLOOR_FOODS
        if self.difficulty is not None:
            # J2 item 1, food count: extras are food before birth is placed.
            self._add_extra_foods()
        if self.random_layout:
            # J2 item 2: own stream; each food keeps its W0 rest height/pose.
            self._layout_rng = np.random.default_rng([seed, 3])
            m = self.w.model
            self._food_rest = {}
            for name in self.floor_foods:
                if name not in self.w.drives._food:
                    continue
                ja = int(m.body_jntadr[m.body(name).id])
                qa, da = int(m.jnt_qposadr[ja]), int(m.jnt_dofadr[ja])
                self._food_rest[name] = (qa, da, self.w.data.qpos[qa + 2],
                                         self.w.data.qpos[qa + 3:qa + 7].copy())
            self.layout_failed = 0
        if (self.difficulty is not None or self.random_layout
                or self.spawn_clear is not None):
            # J2 items 1/2: birth too is placed by the dial and/or the layout
            # (W0's own respawn, body drawn from W0's spawn stream); with
            # both options off this is never reached.
            self.w.respawn()
        if self.e0 != 1.0:      # survival.run_survival's e0, applied the same way
            self.w.drives.state = drives.DriveState(e=self.e0)
        if self.random_needs is not None:
            # J2 item 2: own stream, so W0's RNG streams are the default run's
            self._needs_rng = np.random.default_rng([seed, 2])
            self._draw_needs()
        self._eats_at_life_start = 0
        self._truncated = False
        if self.difficulty is not None:
            # J2 item 1 (part): meal size, per instance; drives.py untouched.
            for name in self.floor_foods:
                if name in self.w.drives._food:
                    gid, nu = self.w.drives._food[name]
                    self.w.drives._food[name] = (gid, dial_meal(nu, self.difficulty))

    def _apply_move_cost(self, scale: float) -> None:
        """J11 (1) `move_cost_dial`: this world's DriveLayer charges
        scale * KAPPA per joule instead of KAPPA. drives.decide computes
        e -= BASAL_B*dt + KAPPA*_power_dt; the accumulated mechanical work
        `_power_dt` is scaled just before that line runs (drives.py is not
        edited, the module constant is not touched, so no other world sees
        it) and the published `last_power_w` is put back to the TRUE mean
        power, so a reader of the body's work is not lied to."""
        dl = self.w.drives
        inner = dl.decide

        def decide():
            p, dt = dl._power_dt, dl._dt_acc
            dl._power_dt = p * scale
            st = inner()
            dl.last_power_w = p / dt if dt > 0 else 0.0
            return st

        dl.decide = decide

    def _food_xy(self) -> np.ndarray:
        w = self.w
        return np.array([w.data.xpos[w.model.body(n).id][:2]
                         for n in getattr(self, "floor_foods", FLOOR_FOODS)
                         if n in w.drives._food]).reshape(-1, 2)

    def _clear_cells(self, legal: np.ndarray) -> np.ndarray:
        """J11 (1) `spawn_clear`: drop the legal cells within R m (body root
        to food body, x, y) of any floor food where it lies NOW. If none is
        left (never seen), the farthest cell(s) are kept and
        `spawn_clear_failed` counts it."""
        if self.spawn_clear is None:
            return legal
        foods = self._food_xy()
        if len(foods) == 0:
            return legal
        dist = np.min(np.linalg.norm(legal[:, None, :] - foods[None, :, :],
                                     axis=2), axis=1)
        keep = dist > self.spawn_clear
        if not keep.any():
            self.spawn_clear_failed += 1
            keep = dist >= dist.max() - 1e-9
        return legal[keep]

    def _clear_spawn(self, legal: np.ndarray, rng: np.random.RandomState,
                     death_xy) -> tuple:
        """W0 spawn_sampler for spawn_clear alone (no dial, no layout):
        today's uniform draw over the cleared cells, W0's own spawn stream."""
        from experiments.w0 import uniform_legal_spawn
        return uniform_legal_spawn(self._clear_cells(legal), rng, death_xy)

    def _draw_needs(self) -> None:
        """J2 item 2: e and i uniform in [1-s, 1], w = 0 (see OPTIONS)."""
        from experiments import drives
        lo = 1.0 - self.random_needs
        e, i = self._needs_rng.uniform(lo, 1.0, size=2)
        self.w.drives.state = drives.DriveState(e=float(e), i=float(i))
        # the needs sense reports change since the last decision; a body born
        # hungry has not just LOST energy, so its first change reads zero.
        self.w._prev_drive = drives.DriveState(e=float(e), i=float(i))
        self.last_start_needs = (float(e), float(i))

    def _add_extra_foods(self) -> None:
        """J2 item 1, food count: make the first dial_extra_foods(d) of
        EXTRA_FOODS floor food in THIS world's drive layer, exactly as
        drives.DriveLayer.__init__ registers obj0/obj1 (geom id, today's
        NU_FLOORFOOD before the meal dial, regrowth timer 0, eat counter 0).
        drives.substep looks the regrowth delay up in the module dict
        RESPAWN_S by name, so the extras' entries (= the floor food's) are
        added there; no FOOD_GEOMS entry is added, so no other world ever
        treats them as food. Must run before the first respawn draw that
        should see them: _build calls it just before the dial's birth
        respawn."""
        from experiments import drives
        dl = self.w.drives
        added = []
        for name in EXTRA_FOODS[:dial_extra_foods(self.difficulty)]:
            try:
                gid = int(self.w.model.geom(name).id)
            except (KeyError, ValueError):
                continue
            drives.RESPAWN_S.setdefault(name, drives.RESPAWN_FLOORFOOD_S)
            dl._food[name] = (gid, drives.NU_FLOORFOOD)
            dl._respawn_at[name] = 0.0
            dl.ate_total[name] = 0
            added.append(name)
        self.floor_foods = FLOOR_FOODS + tuple(added)

    def _near_food_spawn(self, legal: np.ndarray, rng: np.random.RandomState,
                         death_xy) -> tuple:
        """W0 spawn_sampler at dial d: uniform over the LEGAL cells whose
        distance to the nearest floor food (its body's current x, y — the
        foods are free bodies and get pushed) is <= dial_spawn_radius(d).
        If no legal cell is that close (25x25 grid, 0.44 m pitch), the
        nearest legal cell(s) are used. `death_xy` unused, as in
        w0.uniform_legal_spawn. Same RNG stream (W0's _spawn_rng)."""
        w = self.w
        if not self.random_layout:      # _layout_spawn has cleared already
            legal = self._clear_cells(legal)
        foods = np.array([w.data.xpos[w.model.body(n).id][:2]
                          for n in getattr(self, "floor_foods", FLOOR_FOODS)
                          if n in w.drives._food])
        if len(foods) == 0:
            k = int(rng.randint(len(legal)))
            return float(legal[k][0]), float(legal[k][1])
        dist = np.min(np.linalg.norm(legal[:, None, :] - foods[None, :, :],
                                     axis=2), axis=1)
        r = max(dial_spawn_radius(self.difficulty), float(dist.min()) + 1e-9)
        cand = np.flatnonzero(dist <= r)
        k = int(cand[rng.randint(len(cand))])
        self.last_spawn_food_dist = float(dist[k])
        return float(legal[k][0]), float(legal[k][1])

    def _place_foods(self, legal: np.ndarray) -> None:
        """J2 item 2: re-place every floor food at a random LEGAL cell (W0's
        body-legal grid), uniformly, from the env's own stream; keep the
        first draw at which the food touches nothing but ground (his old body
        excepted: it is about to move) and is >= LAYOUT_SEP m from the foods
        already placed. LAYOUT_TRIES misses -> the food stays where it lies
        (counted in layout_failed)."""
        w = self.w
        m, d = w.model, w.data
        placed = []
        for name, (qa, da, z0, quat0) in self._food_rest.items():
            gid = int(m.geom(name).id)
            for _ in range(LAYOUT_TRIES):
                x, y = legal[int(self._layout_rng.integers(len(legal)))]
                if any(math.hypot(x - px, y - py) < LAYOUT_SEP for px, py in placed):
                    continue
                d.qpos[qa:qa + 3] = (x, y, z0)
                d.qpos[qa + 3:qa + 7] = quat0
                d.qvel[da:da + 6] = 0.0
                w.mujoco.mj_forward(m, d)
                clash = False
                for i in range(d.ncon):
                    g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
                    if gid not in (g1, g2):
                        continue
                    o = g2 if g1 == gid else g1
                    if o not in w.ground_gids and o not in w.body_gids \
                            and float(d.contact[i].dist) < -1e-3:
                        clash = True
                        break
                if not clash:
                    placed.append((x, y))
                    break
            else:
                self.layout_failed += 1
                xy = d.xpos[m.body(name).id][:2]
                placed.append((float(xy[0]), float(xy[1])))
        self.last_layout = {n: tuple(map(float, w.data.qpos[v[0]:v[0] + 2]))
                            for n, v in self._food_rest.items()}

    def _layout_spawn(self, legal: np.ndarray, rng: np.random.RandomState,
                      death_xy) -> tuple:
        """W0 spawn_sampler under random_layout: lay the foods out first,
        then drop the legal cells his body would now penetrate a food from
        (tested by placing the body there, as W0.legal_spawns does), then
        draw the body by today's rule (uniform) or the dial's, from W0's
        own spawn stream."""
        from experiments.w0 import uniform_legal_spawn
        self._place_foods(legal)
        w = self.w
        foods = np.array(list(self.last_layout.values()))
        near = np.min(np.linalg.norm(legal[:, None, :] - foods[None, :, :],
                                     axis=2), axis=1) < LAYOUT_BODY_CHECK
        keep = []
        for k, (x, y) in enumerate(legal):
            if near[k]:
                w._place(float(x), float(y))
                w.mujoco.mj_forward(w.model, w.data)
                if w._penetrating():
                    continue
            keep.append(k)
        legal = self._clear_cells(legal[keep])
        if self.difficulty is not None:
            return self._near_food_spawn(legal, rng, death_xy)
        return uniform_legal_spawn(legal, rng, death_xy)

    def _dial_respawn(self, ate_before: dict) -> None:
        """Shorten the floor-food regrowth timer just set by an eat event.

        drives.substep set `_respawn_at = t_eat + RESPAWN_S`; at dial d it
        becomes t_eat + d*RESPAWN_S. Applied at the decision boundary, so the
        delay is never shorter than the rest of that decision (<= 0.2 s)."""
        from experiments.drives import RESPAWN_S
        dl = self.w.drives
        for name in self.floor_foods:
            if name in dl.ate_total and dl.ate_total[name] > ate_before.get(name, 0):
                dl._respawn_at[name] -= (1.0 - self.difficulty) * RESPAWN_S[name]

    @staticmethod
    def _obs(o: dict) -> np.ndarray:
        return np.concatenate([o[k] for k in SENSES]).astype(np.float32)

    def _meals(self) -> int:
        return int(sum(self.w.drives.ate_total.values()))

    def _info(self) -> dict:
        w = self.w
        return {"meals": self._meals(),
                "ate_total": {k: int(v) for k, v in w.drives.ate_total.items()},
                "meals_this_life": self._meals() - self._eats_at_life_start,
                "death_cause": w.last_death_cause if self._died else "",
                "life": int(w.life), "decisions": int(w.decisions),
                "sim_s": float(w.sim_seconds)}

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None and (self.w is None or seed != self._seed
                                 or self.w.decisions > 0):
            self._seed = int(seed)
            self._build(self._seed)
            self._pending_obs = self._obs(self.w.observe())
        if self._pending_obs is None:
            # no observation held (e.g. reset mid-life): the world's next one
            self._pending_obs = self._obs(self.w.observe())
        o, self._pending_obs = self._pending_obs, None
        continuing = self._truncated and not self._died
        self._died = False
        self._truncated = False
        self._ep_decisions = 0
        if not continuing:          # a truncation does not start a new life
            self._eats_at_life_start = self._meals()
        return o, self._info()

    def step(self, action):
        w = self.w
        d_before = w.drives.state.d()
        meals_before = self._meals()
        ate_before = dict(w.drives.ate_total)
        w.decide(np.asarray(action, dtype=float))
        if self.difficulty is not None:
            self._dial_respawn(ate_before)
        eats = self._meals() - meals_before
        self._died = bool(w.died_this_decision)
        d_after = D_DEATH if self._died else w.drives.state.d()
        info = self._info()
        if self._died:
            # the respawned body's first observation, handed out by reset();
            # the world is NOT rebuilt (lives share one world, as in W0).
            info["meals_this_life"] = info["meals"] - self._eats_at_life_start
        if self._died and self.e0 != 1.0:
            # run_survival re-applies e0 at every respawn, before the next
            # observe(); same order here (found by the proof: without it the
            # second life lived on the world's default energy).
            w.drives.state = drives.DriveState(e=self.e0)
        if self._died and self.random_needs is not None:
            self._draw_needs()      # same place e0 is re-applied: before observe()
        o = self._obs(w.observe())
        self._ep_decisions += 1
        self._truncated = (not self._died
                           and self.max_episode_decisions is not None
                           and self._ep_decisions >= self.max_episode_decisions)
        if self._died or self._truncated:
            self._pending_obs = o
        if self._died and self.terminal_obs == "dying":
            # J9: the reborn senses above were still observed (same stream
            # position as the default) and wait for reset(); the transition
            # that ends the life carries the body that died.
            o = self._obs(w.dying_obs)
        dr = float(d_before - d_after)
        if self.reward == "homeo-dr":
            r = dr
        else:
            drive = 100.0 * dr
            if self.reward == "dr100-eat-pos" and not self._died:
                drive = max(0.0, drive)
            r = drive + self.eat_pay * eats
        return o, r, self._died, self._truncated, info


def make_vec(n: int = 4, *, seeds=None, asynchronous: bool = True, **kw):
    """J6(1): `n` JackEnvs stepped in `n` worker processes — default OFF
    (nothing uses it unless asked). It changes nothing in the world: each
    worker holds an ordinary `JackEnv(seed=seeds[i], **kw)`, and the proof
    (`experiments/tests/test_jack_env_vec.py`) checks every worker's stream
    equals a single env's, field for field.

    gymnasium `AsyncVectorEnv` (`SyncVectorEnv` if `asynchronous=False`) with
    `autoreset_mode=SAME_STEP`: on the step that ends a life (or a cap) the
    returned observation is the next life's first one (what `reset()` hands
    out) and the transition's own last observation is in
    `info["final_obs"][i]` — so a learner bootstraps from the right body, and
    with `terminal_obs="dying"` that is the body that died.

    Speed (decisions per wall-second, random actions; box = 4 shared ARM
    cores under load, so the box rows are a floor, not a measurement of the
    vector): single 35.3/s, make_vec(2) 56.2/s wall (1.59x), per CPU-second
    unchanged (35.9 vs 33.3) — 11-16, load 3.8. Kaggle 4-core row owed.
    """
    import functools
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, AutoresetMode
    seeds = list(range(n)) if seeds is None else [int(x) for x in seeds]
    if len(seeds) != n:
        raise ValueError(f"need {n} seeds, got {len(seeds)}")
    fns = [functools.partial(JackEnv, seed=s, **kw) for s in seeds]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    return cls(fns, autoreset_mode=AutoresetMode.SAME_STEP)
