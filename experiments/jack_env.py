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
  difficulty=None       J2 item 1, PART (meal size, regrowth delay, spawn
                         distance; food count still owed). None
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
  OPTIONS PROOF (measured 2026-11-15, `python -m
  experiments.tests.test_jack_env_options 300`, seeds 153/154/155): (a)
  explicit defaults == JackEnv() byte for byte; (b) dr100-eat == 100*homeo-dr
  + 100*eats on every decision (eats on 153/155); (c) pos: all alive
  decisions >= 0, deaths < 0, plain dr100-eat had negative alive ones; (d)
  K=40: 6 truncations + 2 deaths per seed, observations/rewards/deaths equal
  the uncapped run, meals_this_life carries; control stream differs. PASS.
  The J1 proof (test_jack_env) re-run after the change: PASS, unchanged.
  Still owed for J2: item 1 (the dial d), item 2 (random needs/layout), 5.

PROOF (measured 2026-11-15, `python -m experiments.tests.test_jack_env 300`):
  check_env PASS. Seeds 153/154/155 x 300 random decisions, against
  run_survival(policy="random"), identical: e0=1.0 eat events
  {obj1:1} / {} / {obj0:1,obj1:1}, reward sums -0.034212046 / -0.069107120 /
  -0.048612778; e0=0.05 deaths at decisions [98,206] / [99,205] / [105,213].
  Control (action stream seed+1) differs on all 6. Caveat: identity holds for
  float64 actions; a float32 action (what SB3 emits) rounds the 8-vector and
  drifts the reward sum by ~1e-9..4e-3 in 300 decisions — a different action,
  not a different world.
  Observation width 74. The terminal observation of a death step is the
  RESPAWNED body (W0 respawns inside decide); the dead body's is not kept.
SPEED (measured): one env on the box, nice 19, 400 random decisions:
  38.8 decisions/s wall (38.9 per CPU-second) -> 1M decisions ~7.2 h on one
  core. 4-process vector and the Kaggle CPU figure: still owed (J1 step 3).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SENSES = ("vision", "audio", "touch", "proprio", "needs")
REWARDS = ("homeo-dr", "dr100-eat", "dr100-eat-pos")
FLOOR_FOODS = ("obj0", "obj1")
MEAL_AT_D0 = 1.0     # Avalon: a food fills the bar (PUBLISHED_PRIORS §3 table)
SPAWN_R_D0 = 0.5     # m; Avalon eat: food 0.5 m in front of the agent at d=0
SPAWN_R_SPAN = 17.0  # m; ~ the arena diagonal (2*sqrt(2)*6): r(1) covers all


def dial_spawn_radius(d: float) -> float:
    """Spawn-to-nearest-floor-food radius at dial d: 0.5 m at d=0, linear."""
    return SPAWN_R_D0 + d * SPAWN_R_SPAN


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
                 difficulty: Optional[float] = None):
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
        from experiments import drives
        from experiments.w0 import W0
        self.w = W0(seed=seed, j0=self.j0, alpha=self.alpha, lethal=True,
                    diary=self.diary,
                    spawn_sampler=(None if self.difficulty is None
                                   else self._near_food_spawn))
        if self.difficulty is not None:
            # J2 item 1: birth too is placed by the dial (W0's own respawn,
            # drawn from W0's spawn stream); d=1 never reaches here.
            self.w.respawn()
        if self.e0 != 1.0:      # survival.run_survival's e0, applied the same way
            self.w.drives.state = drives.DriveState(e=self.e0)
        self._eats_at_life_start = 0
        self._truncated = False
        if self.difficulty is not None:
            # J2 item 1 (part): meal size, per instance; drives.py untouched.
            for name in FLOOR_FOODS:
                if name in self.w.drives._food:
                    gid, nu = self.w.drives._food[name]
                    self.w.drives._food[name] = (gid, dial_meal(nu, self.difficulty))

    def _near_food_spawn(self, legal: np.ndarray, rng: np.random.RandomState,
                         death_xy) -> tuple:
        """W0 spawn_sampler at dial d: uniform over the LEGAL cells whose
        distance to the nearest floor food (its body's current x, y — the
        foods are free bodies and get pushed) is <= dial_spawn_radius(d).
        If no legal cell is that close (25x25 grid, 0.44 m pitch), the
        nearest legal cell(s) are used. `death_xy` unused, as in
        w0.uniform_legal_spawn. Same RNG stream (W0's _spawn_rng)."""
        w = self.w
        foods = np.array([w.data.xpos[w.model.body(n).id][:2]
                          for n in FLOOR_FOODS if n in w.drives._food])
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

    def _dial_respawn(self, ate_before: dict) -> None:
        """Shorten the floor-food regrowth timer just set by an eat event.

        drives.substep set `_respawn_at = t_eat + RESPAWN_S`; at dial d it
        becomes t_eat + d*RESPAWN_S. Applied at the decision boundary, so the
        delay is never shorter than the rest of that decision (<= 0.2 s)."""
        from experiments.drives import RESPAWN_S
        dl = self.w.drives
        for name in FLOOR_FOODS:
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
        from experiments.survival import D_DEATH
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
            from experiments import drives
            w.drives.state = drives.DriveState(e=self.e0)
        o = self._obs(w.observe())
        self._ep_decisions += 1
        self._truncated = (not self._died
                           and self.max_episode_decisions is not None
                           and self._ep_decisions >= self.max_episode_decisions)
        if self._died or self._truncated:
            self._pending_obs = o
        dr = float(d_before - d_after)
        if self.reward == "homeo-dr":
            r = dr
        else:
            drive = 100.0 * dr
            if self.reward == "dr100-eat-pos" and not self._died:
                drive = max(0.0, drive)
            r = drive + self.eat_pay * eats
        return o, r, self._died, self._truncated, info
