"""J6(d) DRINK race — arms as declared in docs/research/NEEDS_CLOCK_RACE.md
("J6(d) step 2 — DRINK race DECLARED", 2026-11-14 18:25 UTC, commit 4c316fa).
The declaration came before this module and before any cell. Rule, bars, seeds and apparatus checks are stated there and
NOT here, so the two cannot drift.

  dB   drink is an act: a gulp needs the mouth in today's drink zone
       continuously for SWALLOW_S = 1.0 s with every mouth geom's speed
       < SWALLOW_VMAX = 0.2 m/s throughout (leaving the zone or moving resets).
       NU_DRINK, DRINK_DEPTH_M, DRINK_REFRACTORY_S unchanged.
  dB0  apparatus arm (ii), NOT raced: SWALLOW_S = 0, SWALLOW_VMAX = inf. Must
       reproduce swp_k72 exactly (s1 l3 = 26 gulps) or dB is VOID.
  dA / dC  pool far: NOT BUILT. The pool's (2.6, -2.4) is a literal in
       playground.py (lines 421, 886), ne_01._build and needs.py:820, so
       moving it needs a world-file refactor that re-bills every certificate
       resting on playground.py. That refactor needs its own sitting.

Base for every arm: water_clock_race._apply("swp_k72") (a fresh reload of
experiments.needs each call, so the patch below never leaks across arms).
"""
from __future__ import annotations

import math

from experiments import water_clock_race as WR

SWALLOW = {"dB": (1.0, 0.2), "dB0": (0.0, math.inf)}
ARMS = ("swp_k72",) + tuple(SWALLOW)


def _patch_swallow(N, swallow_s: float, vmax: float) -> None:
    import numpy as np
    L = N.NeedLayer
    orig_substep, orig_reset = L.substep, L._reset_body_state

    def _reset_body_state(self):
        orig_reset(self)
        self._sw_t = 0.0
        self._sw_prev = None

    def substep(self, model, data, dt):
        # Update the swallow timer from the post-step mouth pose BEFORE the
        # original substep consults _mouth_near_surface (it does so only when
        # off refractory; the timer must run regardless).
        pos = np.array([data.geom_xpos[g] for g in sorted(self._mouth_geoms)])
        prev = getattr(self, "_sw_prev", None)
        v = 0.0 if prev is None else float(np.max(np.linalg.norm(pos - prev, axis=1)) / dt)
        self._sw_prev = pos.copy()
        if orig_near(self, data) and v < vmax:
            self._sw_t = getattr(self, "_sw_t", 0.0) + dt
        else:
            self._sw_t = 0.0
        orig_substep(self, model, data, dt)

    orig_near = L._mouth_near_surface

    def _mouth_near_surface(self, data):
        if not orig_near(self, data):
            return False
        ok = getattr(self, "_sw_t", 0.0) >= swallow_s - 1e-12
        if ok:
            self._sw_t = 0.0      # one swallow, one gulp
        return ok

    L._reset_body_state = _reset_body_state
    L.substep = substep
    L._mouth_near_surface = _mouth_near_surface


def _apply(arm: str):
    N, mods = WR._apply("swp_k72")
    if arm in SWALLOW:
        _patch_swallow(N, *SWALLOW[arm])
    elif arm in ("dA", "dC"):
        raise NotImplementedError("pool-far arms need the playground.py pool "
                                  "literal refactor; see module docstring")
    elif arm != "swp_k72":
        raise KeyError(arm)
    return N, mods
