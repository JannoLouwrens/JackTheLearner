"""odour.py — smell, as a field in the world rather than a second vision.

Certified by `SM.01`. Built for `SM.02` (smell finds what vision cannot see) and
for `UB.11`, which gives every modality its own ablation row and its own placebo.

WHY THIS EXISTS AT ALL, AND WHY IT IS NOT A DISTANCE SENSOR
-----------------------------------------------------------
`docs/research/FROZEN_VS_PLASTIC.md` §8.3 makes the case with three pieces of
evidence and they converge on one design constraint: **a smooth analytic field
is not a cheap approximation of smell, it is a different sense.**

  * FlyGym's shipped `OdorArena` is a static inverse-square field and its
    bundled demo solves it with a hand-written bilateral-difference controller
    and *no learning at all*. A smooth field is not a learning problem.
  * Celani, Villermaux & Vergassola (Phys. Rev. X 4:041015): moths in a steady
    uniform stimulus stop flying upwind and start casting; they resume only when
    the stimulus is PULSED. The intermittency is the message.
  * Farrell et al. (Environ. Fluid Mech. 2:143-169, 2002) against Jones' field
    data: real plumes are blank 85.2% of the time at 2 m, 90.1% at 5 m, 83.7%
    at 10 m, with peak/mean concentration ratios of 36 / 78 / 112.

So this module ships TWO field models and keeps both, because they are the arms
of a comparison that has to be run rather than argued:

    O1  StaticField   sum over sources of A*exp(-d/LAMBDA_M). No wind, no
                      occlusion, no time. It is the CONTROL SM.02 must beat: if
                      O1 buys as much as O2, smell in this world is a distance
                      sensor wearing the word "smell" and the intermittency
                      literature does not apply to us.
    O2  PuffField     Poisson puff emission, wind advection, crosswind Gaussian
                      spread, and per-puff line-of-sight occlusion by GADEN's
                      trick (a 3-sigma distance cutoff, then one ray-cast per
                      surviving candidate). Whiffs, blanks, and the one property
                      that makes smell non-redundant.

O3 (baked CFD, replayed from disk) is deliberately NOT implemented. It is the
cheapest at runtime and it is the wrong shape for this project: a jungle Jack
rebuilds is exactly the world a pre-baked plume cannot follow.

THE PROPERTY THAT DECIDES WHETHER SMELL IS WORTH ANYTHING
----------------------------------------------------------
Odour passes occlusion; light does not. That is the whole non-redundancy
argument, and it is not a flag in this file — it is `mj_ray` against the same
geometry the eye sees. A puff behind a wall is invisible to the receiver; a puff
that has DRIFTED PAST the wall is not. The source can be hidden while its odour
is not, which is what a wall does to a smell in the real world and what no
amount of vision recovers.

Note what that implies and why the occlusion test in `SM.01` is not vacuous:
occlusion must *attenuate* (some puffs are blocked, or the ray-cast is a no-op
and "odour passes occlusion" is true by construction) while leaving the reading
*non-zero* (or odour is just light with extra steps). Both directions are gated.

WHAT THIS MODULE OWNS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
It owns the field and the sensor. It does not step physics — the caller owns
`mj_step` and calls `step()` after it, exactly as `drives` and `plants` do, for
the same reason: a layer that owned the loop would be a second copy of the
stepping code. It carries no policy, no reward and no memory of where food was.
Searching is Jack's problem.

THE CONSTANTS, and which kind each one is
------------------------------------------
Two kinds, and the difference matters (the `plants.py` precedent):

  * DECLARATIONS -- `LAMBDA_M`, `A0`, `PUFF_R0`, `PUFF_GROWTH`,
    `CROSSWIND_SIGMA`, `EMIT_HZ`, `MAX_PUFFS`, `NOISE_SIGMA`. These are choices.
    `SM.01`'s gates constrain them: the falloff must be the declared
    exponential, the wind term must displace the peak in proportion to wind
    speed, an occluded receiver must still read odour, and a disabled source
    must read the noise floor. Changing any of them re-runs SM.01 rather than
    silently making SM.02 easier.
  * DERIVED -- `SNIFF_HZ` is 5 Hz because that is both what this box can afford
    and inside the 4-12 Hz mammalian sniff band, and `SNIFF_BAND_HZ` is carried
    here so the claim is checkable rather than asserted in a comment.

`C = 4` channels (food / decay / smoke / water) tagged per source, never
chemistry. That is GOAL.md's caveman standard: a caveman did not know what
pyrazines are and could still smell that the meat had turned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ── the channels ────────────────────────────────────────────────────────
# Four, tagged per source. A fifth is a one-line change; a chemistry model is
# not, and is not wanted (GOAL.md: "realistic means what it meant to a caveman").
CHANNELS: Tuple[str, ...] = ("food", "decay", "smoke", "water")
C = len(CHANNELS)
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}

# ── sampling rate ───────────────────────────────────────────────────────
SNIFF_HZ = 5.0
SNIFF_BAND_HZ = (4.0, 12.0)         # mammalian sniff band; SM.01 gates on it
HEAD_SEP_M = 0.16                   # bilateral receiver separation (two nostrils)

# ── O1: the static field ────────────────────────────────────────────────
A0 = 1.0                            # concentration at the source, per unit strength
LAMBDA_M = 2.0                      # e-folding length, metres

# ── O2: the puff plume ──────────────────────────────────────────────────
PUFF_R0 = 0.05                      # m, radius at emission
PUFF_GROWTH = 0.02                  # m^2/s;  r^2(age) = PUFF_R0^2 + 2*PUFF_GROWTH*age
CROSSWIND_SIGMA = 0.05              # m / sqrt(s), per-puff small-scale turbulence
EMIT_HZ = 20.0                      # puffs per second per enabled source
MAX_PUFFS = 500                     # the configuration measured at 124 us/step
PUFF_MASS = 0.05                    # per-puff tracer mass, in the same units as A0
LOS_CUTOFF_SIGMA = 3.0              # GADEN: skip (and never ray-cast) beyond this

# MEANDER — the wind's own crosswind wander, shared by every puff, as an
# Ornstein-Uhlenbeck velocity. This is the term that makes the plume
# INTERMITTENT rather than merely spread out, and it is the difference between
# a plume and a blurred distance sensor: per-puff diffusion alone produces a
# smooth Gaussian plume (measured: 3.5% blanks at 2 m, i.e. essentially always
# on), while a wandering centreline produces whiffs and blanks because the
# plume physically leaves the receiver and comes back. Farrell's field data is
# 83-90% blank; this model reaches roughly 40-55% (reported by SM.01, not
# gated by it — see the INTERMITTENCY note at the end of this docstring block).
MEANDER_SIGMA = 0.6                 # m/s, stationary sigma of the crosswind gust
MEANDER_TAU = 3.0                   # s, its correlation time

# ── the sensor ──────────────────────────────────────────────────────────
# The noise floor is a property of the NOSE, not of the field, so it is applied
# only when an rng is supplied. A field asked for a value without an rng returns
# the exact field: that is what lets SM.01 check the declared rule to 1% instead
# of checking it to the sensor's noise.
NOISE_SIGMA = 1e-3


@dataclass
class Source:
    """An odour source. `enabled=False` is the spec's declared null: same
    geometry, same receiver, same distance, nothing emitting."""
    name: str
    channel: str
    pos: Tuple[float, float, float]
    strength: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.channel not in CHANNEL_INDEX:
            raise ValueError(f"unknown odour channel {self.channel!r}; "
                             f"known: {CHANNELS}")
        self.pos = tuple(float(v) for v in self.pos)


def _as_points(points) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if p.shape[1] != 3:
        raise ValueError(f"receiver points must be (n, 3); got {p.shape}")
    return p


def _sensor_noise(out: np.ndarray, rng, noise_sigma: float) -> np.ndarray:
    """Additive nose noise, clipped at zero.

    Clipped because a concentration cannot be negative, and a nose that reported
    -0.002 would be reporting an artefact of the model rather than a smell. The
    consequence is that a disabled source reads the half-normal mean
    0.3989*sigma rather than 0 — which is exactly what "the noise floor" means
    and what SM.01's null gate is written against.
    """
    if rng is None or noise_sigma <= 0:
        return out
    return np.maximum(out + rng.normal(0.0, noise_sigma, size=out.shape), 0.0)


class StaticField:
    """O1 — sum over sources of `A0 * strength * exp(-d / LAMBDA_M)`.

    No wind, no time, no occlusion. This is deliberately the weakest thing that
    can be called smell, and it is kept precisely so SM.02 has something to beat:
    a channel that buys no more than O1 is a distance sensor.
    """

    name = "O1_static"

    def __init__(self, sources: Sequence[Source], noise_sigma: float = NOISE_SIGMA,
                 lam: float = LAMBDA_M, amp: float = A0):
        self.sources: List[Source] = list(sources)
        self.noise_sigma = float(noise_sigma)
        self.lam = float(lam)
        self.amp = float(amp)
        self.t = 0.0

    def step(self, dt: float, model=None, data=None) -> None:
        """Static by definition; only the clock moves."""
        self.t += float(dt)

    def sample(self, points, model=None, data=None, rng=None) -> np.ndarray:
        pts = _as_points(points)
        out = np.zeros((pts.shape[0], C))
        for s in self.sources:
            if not s.enabled:
                continue
            d = np.linalg.norm(pts - np.asarray(s.pos), axis=1)
            out[:, CHANNEL_INDEX[s.channel]] += (
                self.amp * s.strength * np.exp(-d / self.lam))
        return _sensor_noise(out, rng, self.noise_sigma)

    # Costing hook: how many rays this model casts per sample (zero, by design).
    def rays_last_sample(self) -> int:
        return 0


class PuffField:
    """O2 — Poisson puffs, wind advection, crosswind spread, per-puff occlusion.

    A puff is a Gaussian blob of tracer with a radius that grows with age:

        r(age)^2 = PUFF_R0^2 + 2 * PUFF_GROWTH * age
        c(x)     = mass / (2*pi*r^2)^(3/2) * exp(-|x - p|^2 / (2 r^2))

    ADVECTION is `p += wind * dt`, and CROSSWIND SPREAD is a Gaussian increment
    of sigma `CROSSWIND_SIGMA * sqrt(dt)` on each of the two axes perpendicular
    to the wind. Along-wind position therefore carries no diffusive noise, which
    is Farrell's formulation and which is what makes the displacement rule in
    SM.01 an exact statement (`peak displacement == |wind| * elapsed`) rather
    than a statistical one that a large enough sample could always be found to
    satisfy.

    MEANDER is an Ornstein-Uhlenbeck crosswind gust shared by every puff, added
    to the wind vector. It is what makes the plume intermittent; see the
    MEANDER_SIGMA comment above. It is CROSSWIND ONLY, so it cannot move the
    along-wind position of anything, which is why SM.01's displacement rule
    stays an exact statement with meander on or off. SM.01 nonetheless measures
    that rule at `meander_sigma=0`, because a rule about the wind term is
    isolated by holding everything else steady — and the control is run under
    the identical condition.

    OCCLUSION is GADEN's trick, in this order and for its reason: cull by the
    3-sigma distance cutoff FIRST (a puff further than that contributes less
    than 1.1e-2 of its peak and is not worth a ray), then cast one ray per
    surviving candidate from the receiver toward the puff centre. A hit closer
    than the puff means solid matter is in the way and the puff contributes
    nothing to that receiver.

    `drop_wind=True` is SM.01's declared control: the advection term is skipped
    and NOTHING ELSE changes — the wind vector still defines the crosswind
    basis, the meander state still integrates, the emission is drawn from the
    same rng in the same order. It must be caught by the displacement gate, or
    that gate is decorative.
    """

    name = "O2_puffs"

    def __init__(self, sources: Sequence[Source], wind=(0.0, 0.0, 0.0),
                 emit_hz: float = EMIT_HZ, seed: int = 0,
                 max_puffs: int = MAX_PUFFS, los: bool = True,
                 noise_sigma: float = NOISE_SIGMA,
                 crosswind_sigma: float = CROSSWIND_SIGMA,
                 meander_sigma: float = MEANDER_SIGMA,
                 meander_tau: float = MEANDER_TAU,
                 drop_wind: bool = False):
        self.sources: List[Source] = list(sources)
        self.wind = np.asarray(wind, dtype=float)
        self.emit_hz = float(emit_hz)
        self.max_puffs = int(max_puffs)
        self.los = bool(los)
        self.noise_sigma = float(noise_sigma)
        self.crosswind_sigma = float(crosswind_sigma)
        self.meander_sigma = float(meander_sigma)
        self.meander_tau = float(meander_tau)
        self.drop_wind = bool(drop_wind)
        self.rng = np.random.RandomState(seed)
        self.t = 0.0
        self._rays = 0
        self.gust = 0.0                 # OU state, m/s along the crosswind axis

        # Puff state as parallel arrays: position (n,3), age (n,), channel (n,),
        # mass (n,). Arrays rather than objects because 500 puffs x 5 Hz x a
        # whole life is the inner loop that decides whether this sense is
        # affordable at all.
        self.pos = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.chan = np.zeros(0, dtype=int)
        self.mass = np.zeros(0)

        self._basis = self._crosswind_basis()

    # ── geometry ────────────────────────────────────────────────────────
    def _crosswind_basis(self) -> np.ndarray:
        """Two unit vectors perpendicular to the wind.

        Derived from the DECLARED wind vector, not from whether advection is
        applied, so `drop_wind` changes exactly one term.
        """
        w = self.wind
        n = float(np.linalg.norm(w))
        if n < 1e-12:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        w = w / n
        ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(w, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        e2 /= np.linalg.norm(e2)
        return np.stack([e1, e2])

    def radius(self, age=None) -> np.ndarray:
        a = self.age if age is None else np.asarray(age, dtype=float)
        return np.sqrt(PUFF_R0 ** 2 + 2.0 * PUFF_GROWTH * a)

    # ── emission and transport ──────────────────────────────────────────
    def emit(self, n: int = 1) -> None:
        """Emit `n` puffs from every enabled source, at the source position."""
        live = [s for s in self.sources if s.enabled]
        if not live or n <= 0:
            return
        pos = np.repeat(np.array([s.pos for s in live]), n, axis=0)
        chan = np.repeat(np.array([CHANNEL_INDEX[s.channel] for s in live]), n)
        mass = np.repeat(np.array([PUFF_MASS * s.strength for s in live]), n)
        self.pos = np.vstack([self.pos, pos])
        self.age = np.concatenate([self.age, np.zeros(pos.shape[0])])
        self.chan = np.concatenate([self.chan, chan])
        self.mass = np.concatenate([self.mass, mass])
        self._cull()

    def _cull(self) -> None:
        """Oldest first: a puff that has aged out has spread below detection
        anyway, and an unbounded list is how a 5 Hz sense becomes a 40% frame."""
        if self.pos.shape[0] <= self.max_puffs:
            return
        keep = np.argsort(self.age)[: self.max_puffs]
        keep.sort()
        self.pos, self.age = self.pos[keep], self.age[keep]
        self.chan, self.mass = self.chan[keep], self.mass[keep]

    def step(self, dt: float, model=None, data=None) -> None:
        dt = float(dt)
        # The gust integrates whether or not advection is applied, so that
        # `drop_wind` differs from the real field in exactly one term and not
        # also in its random stream.
        if self.meander_sigma > 0 and self.meander_tau > 0:
            self.gust += (-self.gust / self.meander_tau) * dt + self.meander_sigma * \
                math.sqrt(2.0 * dt / self.meander_tau) * self.rng.normal()
        n = self.pos.shape[0]
        if n:
            if not self.drop_wind:
                self.pos = self.pos + (self.wind + self.gust * self._basis[0]) * dt
            if self.crosswind_sigma > 0:
                jit = self.rng.normal(0.0, self.crosswind_sigma * math.sqrt(dt),
                                      size=(n, 2))
                self.pos = self.pos + jit @ self._basis
            self.age = self.age + dt
        if self.emit_hz > 0:
            k = self.rng.poisson(self.emit_hz * dt)
            if k:
                self.emit(int(k))
        self.t += dt

    # ── the nose ────────────────────────────────────────────────────────
    def sample(self, points, model=None, data=None, rng=None) -> np.ndarray:
        pts = _as_points(points)
        out = np.zeros((pts.shape[0], C))
        self._rays = 0
        if self.pos.shape[0]:
            r = self.radius()
            peak = self.mass / (2.0 * math.pi * r ** 2) ** 1.5
            cutoff = LOS_CUTOFF_SIGMA * r
            for i, x in enumerate(pts):
                delta = self.pos - x
                d = np.linalg.norm(delta, axis=1)
                near = np.nonzero(d <= cutoff)[0]          # GADEN's cutoff first
                if near.size == 0:
                    continue
                if self.los and model is not None and data is not None:
                    near = near[self._visible(model, data, x, delta[near], d[near])]
                    if near.size == 0:
                        continue
                c = peak[near] * np.exp(-(d[near] ** 2) / (2.0 * r[near] ** 2))
                np.add.at(out[i], self.chan[near], c)
        return _sensor_noise(out, rng, self.noise_sigma)

    def _visible(self, model, data, x, delta, dist) -> np.ndarray:
        """Boolean mask: which candidate puffs have line of sight to `x`.

        One `mj_ray` per candidate, cast from the receiver toward the puff. A hit
        strictly nearer than the puff is solid matter in between. `flg_static=1`
        so walls and terrain — the things that actually occlude — are included.
        """
        import mujoco
        gid = np.zeros(1, dtype=np.int32)
        pnt = np.asarray(x, dtype=float)
        vis = np.ones(delta.shape[0], dtype=bool)
        for j in range(delta.shape[0]):
            if dist[j] <= 1e-9:
                continue
            vec = delta[j] / dist[j]
            hit = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, gid)
            self._rays += 1
            if hit >= 0.0 and hit < dist[j] - 1e-6:
                vis[j] = False
        return vis

    def rays_last_sample(self) -> int:
        return int(self._rays)

    def blank_fraction(self, readings, thresh: float) -> float:
        """Intermittency: the fraction of samples below `thresh`.

        Reported rather than gated by SM.01 — the field's blankness is a
        property SM.02's difficulty rides on, and Farrell's 83-90% is a target
        to be compared against, not a rule this world is claimed to obey.
        """
        a = np.asarray(readings, dtype=float)
        return float((a < thresh).mean()) if a.size else 0.0


# ── the sensor Jack actually carries ────────────────────────────────────
# Two receiver sites, left and right of the head (mammals do sample
# bilaterally), reading C channels each, plus the temporal derivative of the
# bilateral mean: 2*C + C = 12 floats at C = 4. FROZEN_VS_PLASTIC.md §8.3.
OBS_DIM = 2 * C + C


class OdourSensor:
    """Bilateral nose. Owns no field and no clock — the caller supplies both."""

    def __init__(self, field, sep: float = HEAD_SEP_M):
        self.field = field
        self.sep = float(sep)
        self._prev_mean: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    def sites(self, head_pos, heading_rad: float) -> np.ndarray:
        """(left, right) receiver positions for a head at `head_pos` facing
        `heading_rad` in the world xy-plane."""
        p = np.asarray(head_pos, dtype=float)
        lateral = np.array([-math.sin(heading_rad), math.cos(heading_rad), 0.0])
        return np.stack([p + lateral * self.sep / 2.0,
                         p - lateral * self.sep / 2.0])

    def obs(self, head_pos, heading_rad: float, t: float,
            model=None, data=None, rng=None) -> np.ndarray:
        """12 floats: [left C, right C, d(mean)/dt C].

        The derivative is against WALL-CLOCK SIMULATED TIME supplied by the
        caller, not against "one call ago": a sense whose derivative depended on
        how often it happened to be polled would report a different world at a
        different frame skip.
        """
        conc = self.field.sample(self.sites(head_pos, heading_rad),
                                 model=model, data=data, rng=rng)
        mean = conc.mean(axis=0)
        if self._prev_mean is None or self._prev_t is None or t <= self._prev_t:
            deriv = np.zeros(C)
        else:
            deriv = (mean - self._prev_mean) / (t - self._prev_t)
        self._prev_mean, self._prev_t = mean, float(t)
        return np.concatenate([conc[0], conc[1], deriv])

    def reset(self) -> None:
        self._prev_mean = self._prev_t = None


def line_of_sight(model, data, a, b) -> Tuple[bool, float]:
    """Does light get from `a` to `b`? Returns (clear, hit_distance).

    This is the function that makes "odour passes occlusion; light does not"
    a measurement instead of a slogan: the SAME geometry and the SAME ray-caster
    that gate a puff's contribution decide whether the source is visible.
    """
    import mujoco
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-9:
        return True, -1.0
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, a, delta / dist, None, 1, -1, gid)
    blocked = bool(hit >= 0.0 and hit < dist - 1e-6)
    return (not blocked), float(hit)
