"""J15 E2c — the two-number wrapper: a learner HANDED the direction to food.

A handed sense, computed from the simulator: scored-and-ineligible, never his
(GOAL line 2). The rule is frozen in `experiments/forager_clone.py` ("RULE
E2c"); this file is only the wrapper it names, written before J11's rule so
the run needs no new code when that rule freezes. Nothing here runs a learner.

HandedSense(env, mode, encoding) appends to every observation:
  mode "food"          the nearest UNEATEN floor food (the rule's arm)
  mode "random_point"  a random fixed point re-drawn each life (the rule's
                       control, which MUST FAIL)
  mode "zero"          the numbers present but always zero (INBOX 16:14
                       proposal: the field's control is the input removed;
                       reported only unless the Director rules it in)
  encoding "radpi"     2 numbers: bearing (body frame, radians / pi, in
                       [-1, 1); jumps at +-pi) and distance (m / 10, clip 1)
                       — the FROZEN rule's form, the default
  encoding "cossin"    3 numbers: cos(bearing), sin(bearing), distance — the
                       INBOX 16:14 proposal (no jump at +-pi); an option only
No uneaten food (all regrowing): bearing 0 (cos 0, sin 0) and distance 1.

Body frame: bearing = atan2(dy, dx) - yaw of the rover body (the yaw the body
eye and body motor use, fast_sim.body_yaw), wrapped to [-pi, pi). "Uneaten":
drives' own eat test — `t + dt_acc >= _respawn_at[name]` (drives.py line 354).

Random point (control): uniform over the axis-aligned box spanned by this
world's floor foods' positions, re-drawn at the start of each life from the
wrapper's own RandomState(seed + 7919) — it lands where food could be, it
moves with nothing, and a truncation (not a death) keeps the same point. A
fixed point read in the body frame still leaks odometry (§17); said so.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.pilot_b import _Wrapper  # noqa: E402  (gymnasium.Wrapper)

MODES = ("food", "random_point", "zero")
ENCODINGS = ("radpi", "cossin")
DIST_SCALE = 10.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _floor_names(env):
    dl = env.w.drives
    from experiments.jack_env import FLOOR_FOODS
    return [n for n in getattr(env, "floor_foods", FLOOR_FOODS) if n in dl._food]


def uneaten_food_xy(env) -> np.ndarray:
    w, dl = env.w, env.w.drives
    now = dl.t + dl._dt_acc
    xy = [w.data.xpos[w.model.body(n).id][:2] for n in _floor_names(env)
          if now >= dl._respawn_at[n]]
    return np.array(xy, dtype=float).reshape(-1, 2)


def encode(target_xy, body_xy, yaw: float, encoding: str) -> np.ndarray:
    if target_xy is None:
        b, d, have = 0.0, 1.0, False
    else:
        dx, dy = float(target_xy[0] - body_xy[0]), float(target_xy[1] - body_xy[1])
        b = _wrap(math.atan2(dy, dx) - yaw)
        d = min(math.hypot(dx, dy) / DIST_SCALE, 1.0)
        have = True
    if encoding == "radpi":
        return np.array([b / math.pi, d], np.float32)
    c, s = (math.cos(b), math.sin(b)) if have else (0.0, 0.0)
    return np.array([c, s, d], np.float32)


class HandedSense(_Wrapper):
    def __init__(self, env, mode: str = "food", encoding: str = "radpi",
                 seed: int = 0):
        if mode not in MODES or encoding not in ENCODINGS:
            raise ValueError(f"mode {mode!r} / encoding {encoding!r}")
        super().__init__(env)
        self.mode, self.encoding = mode, encoding
        self.n_extra = 2 if encoding == "radpi" else 3
        self._rng = np.random.RandomState(int(seed) + 7919)
        self.point = None
        from gymnasium import spaces
        n = env.observation_space.shape[0] + self.n_extra
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)

    # Found 11-19 wiring E2c: gymnasium 1.x wrappers forward nothing, and
    # pilot_b.PilotLives (run_b wraps every world in it) READS env.w /
    # env.ate_total ... and WRITES env._pending_obs / env._truncated. Through
    # this wrapper the reads raised and the writes would have landed here,
    # silently leaving JackEnv's flags stale. Reads fall through to the world;
    # the writes PilotLives makes are named and forwarded, nothing else.
    FORWARD_SET = ("_pending_obs", "_truncated", "_died")

    def __getattr__(self, name):
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in HandedSense.FORWARD_SET and "env" in self.__dict__:
            setattr(self.env, name, value)
        else:
            object.__setattr__(self, name, value)

    def _draw_point(self):
        xy = np.array([self.env.w.data.xpos[self.env.w.model.body(n).id][:2]
                       for n in _floor_names(self.env)]).reshape(-1, 2)
        lo, hi = xy.min(0), xy.max(0)
        self.point = lo + self._rng.rand(2) * (hi - lo)

    def extra(self) -> np.ndarray:
        if self.mode == "zero":
            return np.zeros(self.n_extra, np.float32)
        from experiments.fast_sim import body_yaw
        w = self.env.w
        body = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        if self.mode == "random_point":
            tgt = self.point
        else:
            f = uneaten_food_xy(self.env)
            tgt = (f[np.argmin(((f - body) ** 2).sum(1))] if len(f) else None)
        return encode(tgt, body, body_yaw(w), self.encoding)

    def reset(self, **kw):
        e = self.env
        continuing = bool(getattr(e, "_truncated", False)
                          and not getattr(e, "_died", False))
        o, info = e.reset(**kw)
        if self.mode == "random_point" and (self.point is None or not continuing):
            self._draw_point()
        return np.concatenate([o, self.extra()]).astype(np.float32), info

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        # a death: the reborn body's sense is read by reset(); this o is the
        # transition's own (dying or reborn, as JackEnv hands it)
        return np.concatenate([o, self.extra()]).astype(np.float32), r, term, trunc, info
