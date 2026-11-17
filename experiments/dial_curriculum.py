"""dial_curriculum.py — the difficulty dial's SUCCESS GATE (ORDERS Next-up 7).

PLAN stage 1, part "His world" (staging) + "Learning to eat", step 2
(COMBINE: build from the published design and code). Nothing here is run on a
learner yet: it is an apparatus option, default OFF (nothing imports it), so
the pilot that first wants a curriculum arm can name it. No world file is
touched (jack_env, w0, drives, needs, survival unchanged), so no certificate
goes stale.

PRIOR (PUBLISHED_PRIORS.md §3 "Already known"; long form
world_needs_curriculum.md L203, L208, L417, L433, L443): a success-gated
difficulty dial is the cheapest method with the largest measured effect —
Avalon eat 0.716 with it vs 0.001 without (IMPALA, 50M steps, Table 1).

THE PUBLISHED CODE (read 11-17 from github.com/Avalon-Benchmark/avalon at
HEAD 1c8f9fec, copies in /data/tmp/avalon_src/):
  godot_gym.py L609-635 `CurriculumWrapper.step`: on `done`, if
      info["success"] == 1: difficulties[task] += task_difficulty_update
      elif == 0:             difficulties[task] -= task_difficulty_update
      then clip to [0.0, 1.0] and hand it to the world generator.
  godot_gym.py L172-179: task_difficulty_update = 0.01 (the wrapper default;
      the paper's launchers use 3e-4 PPO, 6e-4 Dreamer, 5e-3 IMPALA — "per
      global env step depends on the number of workers", L609),
      initial_difficulty = 0.
  world_generator.py L470-480, L524, L559-561: the generator keeps
      max(min_difficulty, ceiling) and draws EVERY world's difficulty
      `rand.uniform(low=min_difficulty, high=ceiling)`; with the curriculum
      off it draws `rand.uniform()` = U[0, 1] (the 0.001 arm).
  Every episode is a NEW generated world.

OURS BESIDE THEIRS (every setting that differs):
  - min_difficulty = initial ceiling = jack_env.D0_ENTRY (0.75), not 0:
    ruling 22 fixes the entry rung at d0 = 0.75 and forbids any rung below
    0.5. `lo` is a parameter; the default is the ruling's. Clipping: Avalon's
    wrapper clips its own value to [0, 1] and the generator applies
    max(min_difficulty, .) on top, so with min > 0 a run of failures would
    bank a value below min that successes must climb back through; Avalon
    never runs min > 0 (L420 passes 0). Ours clips the ceiling at lo itself.
    With lo = 0 the two are identical (proof (a)).
  - the uniform (no-curriculum) arm draws U[lo, 1], not U[0, 1], same reason.
  - step default 0.01 = the wrapper default (L173), applied per LIFE in one
    process. Avalon's launchers apply theirs across many workers sharing one
    ceiling; ours has one worker per run, so the per-life rate is the
    wrapper's own number. A launcher-scale step (3e-4..5e-3) is a parameter.
  - one skill, one ceiling (Avalon: one per task, plus a meta-difficulty that
    picks tasks — we have one task, eating, so no meta-difficulty).
  - success = pilot_b's `sustained` row (life reached the cap AND intake >=
    basal drain over it) — the test §3 item 7 names ("net intake at or above
    drain"); Avalon's eat success = "it eats all of the available food before
    the episode timer is finished". `success_fn` is a parameter.
  - the world is rebuilt from a fresh seed at the drawn d when a life ends
    (Avalon generates a new world per episode). Today's runs keep lives in
    one persisting world; under this wrapper they do not. World seeds come
    from the run seed's own stream `default_rng([seed, 0xD1A2])`, in
    [20,000, 40,000) (W0's RandomState(seed * 104729 + 7) caps seeds < 41,010).
  - draws are made at the moment the next life starts, from the ceiling as
    it stands; Avalon pre-generates worlds in a buffer, so its draws lag the
    ceiling by the buffer depth. Ours has no lag.
  - a ceiling of exactly 1.0 builds today's world (jack_env treats
    difficulty 1.0 as None, "nothing touched").

USE: DialCurriculum(PilotLives(JackEnv(seed, difficulty=lo, ...)), seed=...).
It must wrap pilot_b.PilotLives (it reads the finished-life row there). Each
life's drawn d and the ceiling it was drawn under are in `self.history`.

Proof: experiments/tests/test_dial_curriculum.py.
"""
from __future__ import annotations

import numpy as np

AVALON_STEP = 0.01          # godot_gym.py L173, task_difficulty_update default
MODES = ("gated", "uniform", "fixed")
# World seeds for rebuilt lives: W0 seeds RandomState(seed * 104729 + 7), so a
# seed must stay < 41,010; this band is clear of every seed range used here
# (0-1000s; card 601-616; pilots 301-345, 401+).
WORLD_SEED_LO, WORLD_SEED_HI = 20_000, 40_000


class SuccessGatedDial:
    """Avalon's controller, one ceiling. Pure: no world inside.

    gated    d ~ U[lo, ceiling]; ceiling += step on success, -= on failure,
             clipped to [lo, hi]           (godot_gym L623-633, wg L524/L561)
    uniform  d ~ U[lo, hi] every life, success ignored   (wg L526, curriculum off)
    fixed    d = lo every life (today's runs: one rung)
    """

    def __init__(self, lo: float, hi: float = 1.0, step: float = AVALON_STEP,
                 mode: str = "gated", rng=None, initial: float | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"need 0 <= lo <= hi <= 1, got {lo}, {hi}")
        self.lo, self.hi, self.step, self.mode = float(lo), float(hi), float(step), mode
        self.ceiling = float(lo if initial is None else initial)
        self.ceiling = max(self.lo, min(self.ceiling, self.hi))
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def draw(self) -> float:
        if self.mode == "fixed":
            return self.lo
        top = self.hi if self.mode == "uniform" else self.ceiling
        return float(self.rng.uniform(low=self.lo, high=top))

    def update(self, success: bool) -> float:
        if self.mode == "gated":
            self.ceiling += self.step if success else -self.step
            self.ceiling = max(self.lo, min(self.ceiling, self.hi))
        return self.ceiling


def sustained(row: dict) -> bool:
    """§3 item 7's success test, as pilot_b logs it."""
    return bool(row.get("sustained"))


try:
    import gymnasium as _gym
    _Wrapper = _gym.Wrapper
except ImportError:  # pragma: no cover
    _Wrapper = object


class DialCurriculum(_Wrapper):
    """Each life in a NEW world at a difficulty drawn by SuccessGatedDial.

    The first life uses the world the inner JackEnv was built with (build it
    at difficulty=lo); every later life rebuilds the world at the draw.
    """

    def __init__(self, pilot, seed: int, lo: float | None = None, hi: float = 1.0,
                 step: float = AVALON_STEP, mode: str = "gated", success_fn=sustained):
        super().__init__(pilot)
        from experiments.jack_env import D0_ENTRY
        self.pilot = pilot
        self.jenv = pilot.env
        lo = D0_ENTRY if lo is None else float(lo)
        rng = np.random.default_rng([int(seed), 0xD1A1])
        self.dial = SuccessGatedDial(lo, hi, step, mode, rng=rng)
        self._world_rng = np.random.default_rng([int(seed), 0xD1A2])
        self.success_fn = success_fn
        cur = 1.0 if self.jenv.difficulty is None else self.jenv.difficulty
        self.history = []          # one dict per life: d, ceiling at draw, world seed, success
        self._current = {"d": float(cur), "ceiling": self.dial.ceiling,
                         "world_seed": int(self.jenv._seed)}
        self._lives_seen = 0
        self._need_rebuild = False

    def set_world(self, d: float, world_seed: int) -> None:
        """Rebuild the inner JackEnv's world at difficulty d (fresh state)."""
        e = self.jenv
        e.difficulty = None if float(d) == 1.0 else float(d)
        e._seed = int(world_seed)
        e._build(int(world_seed))
        e._pending_obs = None
        e._truncated = False
        e._died = False
        self.pilot._capped = False      # the new world's birth, not a respawn

    def reset(self, *, seed=None, options=None):
        if self._need_rebuild:
            d = self.dial.draw()
            ws = int(self._world_rng.integers(WORLD_SEED_LO, WORLD_SEED_HI))
            self._current = {"d": d, "ceiling": self.dial.ceiling, "world_seed": ws}
            self.set_world(d, ws)
            self._need_rebuild = False
        return self.pilot.reset(seed=seed, options=options)

    def step(self, action):
        o, r, term, trunc, info = self.pilot.step(action)
        if (term or trunc) and len(self.pilot.lives) > self._lives_seen:
            self._lives_seen = len(self.pilot.lives)
            ok = bool(self.success_fn(self.pilot.lives[-1]))
            self.history.append(dict(self._current, success=ok))
            self.dial.update(ok)
            self._need_rebuild = True
            info = dict(info, dial_d=self._current["d"], dial_ceiling=self.dial.ceiling)
        return o, r, term, trunc, info
