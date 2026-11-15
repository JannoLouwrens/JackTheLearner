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
               charges d_before - survival.D_DEATH. (J5 adds the record's
               reward options; default stays this.)
  terminated   True on the decision he dies (W0 lethal).
  truncated    always False here (J5 adds a time limit as an option).
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


def default_cal() -> dict:
    """The j0/alpha every forager/LC run uses (borrowed LC.09 calibration)."""
    from experiments.forager_race import _cal
    c = _cal()
    return {"j0": c["j0"], "alpha": c["alpha"]}


class JackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, j0: Optional[float] = None,
                 alpha: Optional[float] = None, seed: int = 0,
                 e0: float = 1.0, diary=None):
        # e0: birth AND respawn energy, exactly survival.run_survival's e0.
        if j0 is None or alpha is None:
            c = default_cal()
            j0 = c["j0"] if j0 is None else j0
            alpha = c["alpha"] if alpha is None else alpha
        self.j0, self.alpha, self.e0, self.diary = float(j0), float(alpha), float(e0), diary
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
                    diary=self.diary)
        if self.e0 != 1.0:      # survival.run_survival's e0, applied the same way
            self.w.drives.state = drives.DriveState(e=self.e0)
        self._eats_at_life_start = 0

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
        self._died = False
        self._eats_at_life_start = self._meals()
        return o, self._info()

    def step(self, action):
        from experiments.survival import D_DEATH
        w = self.w
        d_before = w.drives.state.d()
        w.decide(np.asarray(action, dtype=float))
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
        if self._died:
            self._pending_obs = o
        return o, float(d_before - d_after), self._died, False, info
