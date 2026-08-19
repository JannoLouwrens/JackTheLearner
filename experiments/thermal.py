"""thermal.py — cold, heat, and a body that can freeze. The substrate PS.02 tests.

GOAL.md's survival directive says "too cold kills him, too hot kills him", and
until this file existed the world had no temperature in it at all: `run senses`
listed `thermal (kills)` with two declared specs and nothing passing. This is
the cold half.

**CAVEMAN REALISM, NOT THERMODYNAMICS** (owner, 2026-08-09: *"we don't actually
need to understand chemistry for this — just like cavemen didn't"*). What a
caveman needs to learn is: cold takes heat out of you at a rate that depends on
how cold it is, fire puts it back, and if your body gets cold enough you die.
That is three rules and they are all here. There is no wind chill, no
conduction/convection split, no clothing, no basal-rate model. The world must be
CONSISTENT, DISCOVERABLE and CONSEQUENTIAL (GOAL.md), and it is exactly those
three properties PS.02 gates — never "realism", which was never falsifiable.

## THE DECLARED LAW — pre-registered before the probe was ever run

    dTb/dt = G_RATE * (T_eff - T_NEUTRAL)                            (1)

    T_eff(x, y) = T_cold + f * (T_FIRE - T_cold),
                  f = exp(-(d / R_FIRE)^2),  d = |xy - fire_xy|      (2)

    death when Tb <= TB_LETHAL                                       (3)

Linear, not Newtonian, and that is deliberate. Newton's law of cooling drives
`Tb` to an asymptote `T_amb + metabolic/k`, which compresses time-to-freezing
into a narrow band no matter how cold the world is — the spread across runs
would then be small next to the spread WITHIN a run, and a probe reading the
episode clock could score well on a body it cannot feel. Equation (1) makes
time-to-freezing `(Tb - TB_LETHAL) / (G_RATE * (T_NEUTRAL - T_eff))`, a ratio of
two per-run quantities that a clock cannot reconstruct. The design constraint
came from the CONTROL, which is the right way round: the silent-lethality
control must be able to fail.

## WHY THE HEAT SOURCE IS INVISIBLE

`fire_xy` puts no geom in the world. A warm patch you can SEE is a warm patch a
vision ray can find, and PS.02's control — the thermal channel deleted from the
sensory vector — would then still be predictable from `vision`. Heat is not
light; making the source thermally-sensed-only is both what a fire's warmth
actually is at distance and what makes the control honest. (SM.01's occlusion
clause is the same discipline: a sense earns its place by carrying something no
other channel carries.)

## WHY THIS IS A WRAPPER AND NOT AN EDIT TO `w0.py`

`ThermalWorld` HOLDS a `W0` and appends its own 2-float channel. It does not
touch `cores.MODALITIES`, `W0.observe()` or the drive layer. The obs-dim scar
is the reason: T2.02 is VOID to this day because an observation width changed
under arms that had been admitted at the old width, and LC.01/LC.02/XL.00 all
carry certificates measured against the current dict. A new sense arrives as an
overlay until a spec says it is load-bearing; then, and only then, is widening
the contract a change worth paying for.

## WHAT IS NOT HERE, DELIBERATELY

Heat death (`too hot kills him`) has no constants and no code: it is a separate
claim, it would need its own lethal threshold and its own probe, and inventing
its numbers here would be exactly the "default nobody measured wearing the
costume of a measurement" this repo refuses. Wetness is likewise NOT coupled to
`T_eff`, though a soaked body plainly cools faster: `w` lives in the drive
layer's observation, so coupling it would hand the silent-lethality control a
back channel into the target. Both are recorded as follow-ups rather than
guessed at.

Every constant below is a PROPOSAL in the PS.01 sense — a world parameter, not a
measurement — and the spec's job is to certify that the world they build is
lethal, bounded and legible, not that the numbers are right in nature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# ── the declared constants (pre-registered 2026-08-12, before any probe) ────
T_NEUTRAL = 20.0       # degC ambient at which the body neither gains nor loses
G_RATE = 0.010         # degC of body temperature per second, per degC of offset
T_FIRE = 45.0          # degC at the centre of the heat source
R_FIRE = 1.5           # m, Gaussian radius of the warm zone
TB_LETHAL = 28.0       # degC; at or below this the life ends of cold
TB_HEALTHY = 37.0      # degC; the reference a rested warm body sits at
TB_SHIVER = 35.0       # degC; below this the cold is unmistakable in the sense

# The per-life draws. A world whose cold is the same every life is a world
# whose time-to-freezing is a constant, and a constant is predicted perfectly
# by a probe that senses nothing at all.
T_COLD_RANGE = (-20.0, 0.0)      # degC, the night's ambient away from the fire
TB0_RANGE = (30.0, 38.0)         # degC, how chilled he already is at spawn
FIRE_DIST_RANGE = (2.5, 6.0)     # m from spawn; never close enough to save him

# The sense's normalisation. Two floats, both in roughly [-1, 1] over the world
# this file builds, so no downstream learner has to rescale them.
CORE_SPAN = TB_HEALTHY - TB_LETHAL       # 9 degC from healthy to dead
SKIN_SPAN = 30.0                          # degC of ambient offset per unit
THERMAL_DIM = 2

# ── shelter (SH.01's substrate; pre-registered 2026-08-19, before any run) ──
# A WORKING shelter is a wind-break: inside it, only this fraction of the
# ambient's offset from thermoneutral is felt.
#
#     T_eff_inside = T_NEUTRAL + SHELTER_LEAK * (T_eff_outside - T_NEUTRAL) (4)
#
# 0.15 keeps the world CONSEQUENTIAL in both directions: at T_cold = -20 degC a
# body inside still cools (felt ambient 14 degC, drift -0.06 degC/s), so a
# shelter postpones freezing ~6.7x rather than abolishing it — an agent that
# shelters and then never leaves still dies, which is what makes leaving to
# warm at the fire a decision worth learning. A COSMETIC shelter applies no
# term at all: same geoms (playground._shelter_fragments emits identical
# boxes), the difference is entirely in this equation firing or not.
# Geometry stays playground's: the footprint half-extent is REFERENCED from
# the file that builds the walls, never transcribed (the PG.8 rule).
SHELTER_LEAK = 0.15


def _shelter_half() -> float:
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from playground import SHELTER_HALF
    return float(SHELTER_HALF)


def inside_shelter(xy, sxy, half: Optional[float] = None) -> bool:
    """Is a horizontal position within a shelter's (axis-aligned) footprint?"""
    h = _shelter_half() if half is None else float(half)
    return (abs(float(xy[0]) - float(sxy[0])) <= h and
            abs(float(xy[1]) - float(sxy[1])) <= h)


def ambient(xy, fire_xy, t_cold: float) -> float:
    """Equation (2): the felt air temperature at a horizontal position."""
    d = math.hypot(float(xy[0]) - float(fire_xy[0]),
                   float(xy[1]) - float(fire_xy[1]))
    f = math.exp(-(d / R_FIRE) ** 2)
    return t_cold + f * (T_FIRE - t_cold)


def drift_per_s(t_eff: float) -> float:
    """Equation (1): degC per second of simulated time, signed."""
    return G_RATE * (t_eff - T_NEUTRAL)


def time_to_lethal_s(tb: float, t_eff: float) -> float:
    """Closed form for equation (1)+(3). `inf` when the body is not losing heat.

    Written here so the test can compare the INTEGRATOR against the law rather
    than against itself, and so a caller can price a world before running it.
    """
    rate = drift_per_s(t_eff)
    if rate >= 0.0:
        return float("inf")
    return (tb - TB_LETHAL) / (-rate)


def sense(tb: float, t_eff: float) -> np.ndarray:
    """The 2-float thermal channel: what he feels of himself and of the air.

    `core`  1.0 at a healthy body, 0.0 at the lethal threshold. This is the
            interoceptive half — the chill you feel in yourself.
    `skin`  the ambient offset from thermoneutral. This is the exteroceptive
            half — how cold the air is where he is standing right now, which is
            the only thing that tells him whether moving would help.

    Both are needed and neither is sufficient: `core` alone says how much is
    left, `skin` alone says how fast it is going.
    """
    return np.array([(tb - TB_LETHAL) / CORE_SPAN,
                     (t_eff - T_NEUTRAL) / SKIN_SPAN], dtype=np.float32)


@dataclass
class ThermalState:
    tb: float
    t_eff: float
    t_cold: float
    fire_xy: Tuple[float, float]
    frozen: bool = False


class ThermalWorld:
    """A `W0` with a temperature over it. Owns no physics; steps after W0 does.

    `inert=True` is the registry's declared null: equation (1) is multiplied by
    zero, so `Tb` never moves and nothing may die of cold. Every other line of
    this class — the draws, the sense, the death check — runs unchanged, so the
    null differs from the world in exactly one term.

    `blind=True` is the SILENT LETHALITY control: the body still cools and still
    dies on the same schedule, but `channel()` returns nothing. It is a property
    of the OBSERVER, never of the world, which is what makes it a control rather
    than a second experiment.
    """

    def __init__(self, w0, seed: int, *, inert: bool = False,
                 blind: bool = False, fire_dist: Optional[float] = None,
                 shelters: tuple = ()):
        self.w0 = w0
        self.inert = bool(inert)
        self.blind = bool(blind)
        # `shelters`: ((x, y, working), ...). Geometry is the CALLER's job —
        # build the W0 with matching playground shelters, or these are ghosts
        # no ray can see; SH.01's visual-identity control needs the geoms.
        self.shelters = tuple((float(x), float(y), bool(wk))
                              for x, y, wk in shelters)
        self._shalf = _shelter_half() if self.shelters else 0.0
        rng = np.random.RandomState(seed * 7717 + 101)
        t_cold = float(rng.uniform(*T_COLD_RANGE))
        tb0 = float(rng.uniform(*TB0_RANGE))
        d = float(rng.uniform(*FIRE_DIST_RANGE)) if fire_dist is None \
            else float(fire_dist)
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        xy = self._xy()
        fire_xy = (xy[0] + d * math.cos(theta), xy[1] + d * math.sin(theta))
        self.state = ThermalState(tb=tb0,
                                  t_eff=self._felt(xy, fire_xy, t_cold),
                                  t_cold=t_cold, fire_xy=fire_xy)
        self.tb_trace = [tb0]

    # ── the shelter law ─────────────────────────────────────────────────
    def shelter_index(self, xy=None) -> int:
        """Index into `self.shelters` of the one he is inside, else -1."""
        p = self._xy() if xy is None else xy
        for i, (sx, sy, _wk) in enumerate(self.shelters):
            if inside_shelter(p, (sx, sy), self._shalf):
                return i
        return -1

    def _felt(self, xy, fire_xy, t_cold: float) -> float:
        """Equation (2), then equation (4) if inside a WORKING shelter."""
        t = ambient(xy, fire_xy, t_cold)
        i = self.shelter_index(xy)
        if i >= 0 and self.shelters[i][2]:
            t = T_NEUTRAL + SHELTER_LEAK * (t - T_NEUTRAL)
        return t

    def _xy(self) -> Tuple[float, float]:
        p = self.w0.data.xpos[self.w0.rover_bid]
        return float(p[0]), float(p[1])

    # ── the sense ───────────────────────────────────────────────────────
    def channel(self) -> np.ndarray:
        """The thermal half of the observation — empty under `blind`."""
        if self.blind:
            return np.zeros(0, dtype=np.float32)
        return sense(self.state.tb, self.state.t_eff)

    def observe(self) -> Dict[str, np.ndarray]:
        """W0's dict, plus `thermal` when the observer has the sense."""
        obs = self.w0.observe()
        ch = self.channel()
        if ch.size:
            obs["thermal"] = ch
        return obs

    # ── one decision ────────────────────────────────────────────────────
    def decide(self, action, dt_s: float) -> None:
        """Advance W0 by one decision, then integrate equation (1) over it."""
        self.w0.decide(action)
        s = self.state
        s.t_eff = self._felt(self._xy(), s.fire_xy, s.t_cold)
        if not self.inert:
            s.tb += drift_per_s(s.t_eff) * dt_s
        self.tb_trace.append(s.tb)
        if s.tb <= TB_LETHAL:
            s.frozen = True

    @property
    def frozen(self) -> bool:
        return self.state.frozen

    def time_to_lethal_s(self) -> float:
        """What the law says is left, from where he is right now."""
        if self.inert:
            return float("inf")
        return time_to_lethal_s(self.state.tb, self.state.t_eff)


# ── self-test: `python -m experiments.thermal` ──────────────────────────────
def _smoke() -> None:
    """The shelter substrate's own checks, kept runnable so a change to this
    file or to playground's shelter geometry re-verifies the contract in
    seconds. NOT a spec: SH.01 owns the claims; this owns the arithmetic.
    """
    import numpy as np
    from experiments.w0 import W0, MIN_LEGAL_SPAWNS

    SH = (("shelterA", -1.0, 1.0), ("shelterB", 1.0, 1.0))
    j0, alpha = 0.001157, 0.5           # world params, any sane value works

    # 1. unused shelters change nothing: no shelter geom, XML identical
    import playground as pg
    p = pg.PlaygroundParams(seed=0)
    assert pg.build_mjcf(p) == pg.build_mjcf(p, shelters=())
    assert "shelter" not in pg.build_mjcf(p)

    # 2. shelters compile, are geometrically identical, leave the spawn set legal
    w = W0(seed=0, j0=j0, alpha=alpha, shelters=SH)
    for suf in ("_wallN", "_wallW", "_wallE"):
        a, b = w.model.geom("shelterA" + suf), w.model.geom("shelterB" + suf)
        assert np.allclose(a.size, b.size) and np.allclose(a.rgba, b.rgba)
    assert len(w.legal_spawns()) >= MIN_LEGAL_SPAWNS

    # 3. the law: equation (4) at the working hut, no term at the cosmetic one,
    #    untouched outside, and the empty tuple reproduces ambient() exactly
    tw = ThermalWorld(w, seed=3, fire_dist=5.0,
                      shelters=((-1.0, 1.0, True), (1.0, 1.0, False)))
    fire, tc = tw.state.fire_xy, tw.state.t_cold
    for xy, working in [((-1.0, 1.0), True), ((1.0, 1.0), False),
                        ((0.0, -1.5), False)]:
        base = ambient(xy, fire, tc)
        want = T_NEUTRAL + SHELTER_LEAK * (base - T_NEUTRAL) if working else base
        assert abs(tw._felt(xy, fire, tc) - want) < 1e-12, xy
    tw0 = ThermalWorld(W0(seed=5, j0=j0, alpha=alpha), seed=5)
    for xy in [(0, 0), (2, -2), (-1.0, 1.0)]:
        assert tw0._felt(xy, tw0.state.fire_xy, tw0.state.t_cold) == \
               ambient(xy, tw0.state.fire_xy, tw0.state.t_cold)

    # 4. membership: edge-inclusive, -1 outside
    assert tw.shelter_index((-1.0, 1.0)) == 0
    assert tw.shelter_index((1.0, 1.0)) == 1
    assert tw.shelter_index((-1.0 + tw._shalf, 1.0)) == 0
    assert tw.shelter_index((-1.0 + tw._shalf + 1e-6, 1.0)) == -1

    # 5. a live decide() integrates the sheltered rate (ratio ~ SHELTER_LEAK)
    def drop_at(xy):
        wi = W0(seed=3, j0=j0, alpha=alpha, shelters=SH)
        ti = ThermalWorld(wi, seed=3, fire_dist=5.0,
                          shelters=((-1.0, 1.0, True), (1.0, 1.0, False)))
        wi._place(*xy)
        tb0 = ti.state.tb
        ti.decide(np.zeros(8), dt_s=0.2)
        return tb0 - ti.state.tb
    d_in, d_out = drop_at((-1.0, 1.0)), drop_at((0.0, -1.5))
    assert 0.0 < d_in < d_out * 0.25, (d_in, d_out)
    print(f"thermal._smoke OK: leak {SHELTER_LEAK}, live dTb ratio "
          f"{d_in / d_out:.3f}, spawns legal, worlds byte-identical unused")


if __name__ == "__main__":
    _smoke()
