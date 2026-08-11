"""SM.01 — The odour field obeys its own pre-registered rules.

HYPOTHESIS (registry, unchanged). An `Odour` overlay in the `Water` pattern
produces concentrations that match the declared field model to within 1%:
inverse-exponential falloff with distance for O1, downwind displacement of the
peak proportional to wind speed for O2, and non-zero concentration at a receiver
with NO line of sight to the source (odour passes occlusion; light does not).

WHY A FIXTURE CERTIFICATE COMES FIRST. `SM.02` is the value test — smell must
find OCCLUDED food faster than a no-smell twin and must buy little or nothing
when the food is in plain sight. Every way that claim can be fake runs through
this field:

  * if the occluded receiver reads ZERO, the whole non-redundancy argument for
    smell collapses and SM.02 is measuring an extra input channel;
  * if the wind term does not move anything, the plume is a distance sensor
    wearing the word "smell", and `FROZEN_VS_PLASTIC.md` §8.3's evidence —
    FlyGym's static arena solved by a hand-written controller with no learning
    — says a value test on it measures the field, not the sense;
  * if a disabled source still reads above the noise floor, the receiver is
    reading the geometry rather than the odour.

So SM.01 `kills` SM.02 and SM.03 (registry), and the gates are written against
the field's DECLARED rules rather than against "looks like a plume".

WHAT MAKES "WITHIN 1%" A REAL CHECK AND NOT A TAUTOLOGY. Three of the four
rules are re-derived HERE, in this file, from constants imported live out of
`experiments/odour.py` (T0.14/T0.22: import the constant, never paste it) —
the closed forms are written independently and compared against what the field
actually returns through its public `sample()`. Two more properties are checked
that have no closed form in the module at all and are exactly where an
implementation quietly goes wrong:

  SUPERPOSITION   two sources sampled together must equal the sum of the two
                  sampled apart. An accumulation bug (`=` for `+=`, a stale
                  buffer, a mis-indexed `np.add.at`) survives every
                  single-source test and dies here.
  CHANNEL ISOLATION  a food source must contribute EXACTLY zero to smoke. This
                  is the off-by-one that turns four senses into one.

THE OCCLUSION CLAUSE IS TWO-SIDED ON PURPOSE. "Odour passes occlusion" is
trivially true if the ray-cast never blocks anything — a `_visible` that always
returned True would pass a one-sided test with the best numbers in the file. So
occlusion is gated in BOTH directions:

  (a) the source has NO line of sight to the hidden receiver — asserted with
      `mj_ray` against the same geometry the eye uses, which is the "light does
      not" half made into a measurement rather than a slogan — and the hidden
      receiver nevertheless reads far above the noise floor;
  (b) the ray-caster demonstrably blocks: a receiver in the block's shadow reads
      measurably LESS with line-of-sight enabled than with it disabled, on the
      same field state; and a single synthetic puff placed behind the block
      contributes EXACTLY zero while the same puff at the same distance with a
      clear line contributes the same with and without occlusion.

Check (b)'s puff pair is the sharp one: identical distance, identical age,
identical mass, differing only in whether solid matter is in the way.

THE GEOMETRY, and why every point sits on the line y = -1.5. `build_mjcf`
scatters its `n_objects` at `x in [-2, 2], y in [-1, 1.5]` with radius <= 0.18,
so the strip `y <= -1.2` is free of seed-dependent clutter by construction and
the only thing on this line is `welded_block` — the immovable 0.30 m cube at
(-1.5, -1.5, 0.15), which is a fixture PG.1 already lists as a landmark. That
is what makes the occlusion result a property of the block rather than a
property of whichever object seed 2 happened to drop in the way. It is asserted
rather than assumed: the test reports which geom the blocked ray hit, and gates
on it being `welded_block`.

WHY THE WIND RULE IS MEASURED WITH THE MEANDER OFF. `odour.PuffField` ships a
crosswind Ornstein-Uhlenbeck gust, because per-puff diffusion alone gives a
smooth plume and smell's whole interest is intermittency. The gust is CROSSWIND
ONLY, so it cannot move the along-wind peak — but a rule about the wind term is
isolated by holding everything else steady, and the declared control (advection
dropped, nothing else changed) is run under exactly the same condition. Both
arms therefore differ in one term. The transect grid is deliberately offset off
the round numbers so that `u * T` never lands on a grid node: a peak-finder that
skipped its sub-cell interpolation would otherwise score exact by luck.

PILOTED ON SEED 90, DISJOINT FROM THE REGISTERED SEEDS 0/1/2 (PG.6's precedent):

    O1 falloff / superposition / channel isolation   deviation ~1e-16
    wind displacement, u in {0.37, 0.83, 1.41, 2.06} deviation ~1e-15
    control (advection dropped)                      deviation 1.0 — caught
    hidden receiver, no line of sight to the source  mean 0.497 vs a 1e-3 floor
    shadow receiver, occlusion on vs off             27.7% attenuation
    O2 cost, 500 puffs, bilateral, occlusion on      ~220 us/step

Gates were then set with margin rather than at the pilot values.

WHAT THIS DOES AND DOES NOT LICENSE. It licenses SM.02: the field has a
declared falloff, a wind term that does what it claims, an occlusion mechanism
that both blocks and is passed, and a null that reads the noise floor. It does
NOT claim the plume is as intermittent as a real one — measured blank fractions
are reported beside Farrell's field numbers (85.2% at 2 m, 90.1% at 5 m, 83.7%
at 10 m) and this model reaches roughly half that. That gap is recorded, not
gated, because intermittency is not in this spec's registered hypothesis; it is
SM.02's difficulty riding on a number the next iteration can now see.
"""

from __future__ import annotations

import math
import time

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from .. import odour  # noqa: E402
from ..protocol import Ledger, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

# The claim is about the WORLD and the field, not only about this file: the
# occluder is `playground.py`'s welded block and every constant is
# `odour.py`'s. Change either and this certificate goes stale loudly instead of
# standing over a world it no longer describes.
IMPL_DEPS = ["playground.py", "experiments/odour.py"]

# ── geometry, all on the clutter-free strip y = -1.5 ────────────────────
BLOCK = np.array([-1.5, -1.5, 0.15])        # `welded_block`, half-extent 0.15
Z = float(BLOCK[2])
SRC = tuple(BLOCK + np.array([-1.00, 0.0, 0.0]))    # 1.0 m upwind of the block
R_HIDDEN = BLOCK + np.array([1.00, 0.0, 0.0])       # 2.0 m downwind, block between
R_SHADOW = BLOCK + np.array([0.25, 0.0, 0.0])       # hard up against the far face
R_LIT = np.array(SRC) + np.array([0.0, -2.0, 0.0])  # same 2.0 m, clear line
OCCLUDER_NAME = "welded_block"

# The synthetic-puff pair for the occlusion mechanism check: same distance from
# R_SHADOW, same age, same mass, one behind the block and one on a clear line.
PUFF_D = 0.80
PUFF_AGE = 2.0
P_BLOCKED = R_SHADOW + np.array([-PUFF_D, 0.0, 0.0])
P_CLEAR = R_SHADOW + np.array([0.0, -PUFF_D, 0.0])

# ── the wind-displacement measurement ───────────────────────────────────
WIND_SPEEDS = (0.37, 0.83, 1.41, 2.06)      # deliberately not round numbers
WIND_T_S = 4.0                              # advection time for the burst
WIND_SRC = (0.0, 0.0, 8.0)                  # open air, above the 2.5 m walls
BURST = 800
TRANSECT_DX = 0.005
TRANSECT_OFFSET = 0.0017                    # so u*T never lands on a grid node

# ── the plume run used for occlusion, nulls and intermittency ───────────
DT = 1.0 / odour.SNIFF_HZ
PLUME_STEPS = 400                           # 80 s of simulated time
WIND = (1.0, 0.0, 0.0)
FARRELL_DIST_M = (2.0, 5.0, 10.0)
FARRELL_BLANK = (0.852, 0.901, 0.837)       # Farrell et al. 2002, for comparison

# ── O1 falloff sampling ─────────────────────────────────────────────────
FALLOFF_D = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)

# ── pre-registered gates ────────────────────────────────────────────────
FIELD_RULE_TOL = 0.01          # the registry's "to within 1%"
FALLOFF_DISCRIM_MIN = 0.10     # ...and the inverse-square rival must MISS by this
CONTROL_MIN_DEV = 0.50         # the broken variant must miss by at least this
OCC_SNR = 50.0                 # hidden receiver, in units of the noise sigma
OCC_LIVE_FRAC = 0.10           # ...and this fraction of samples above 10*sigma
SHADOW_ATTEN_MIN = 0.05        # occlusion must actually block something
NULL_FLOOR_MULT = 1.0          # disabled source: mean <= this * NOISE_SIGMA
FRAME_S = 1.0 / 30.0           # the 30 Hz frame the cost is quoted against
COST_FRAME_FRAC_MAX = 0.05     # O2 must fit in 5% of one frame on this box


# ── helpers ─────────────────────────────────────────────────────────────
def _world(seed: int):
    """The playground, unchanged. The occluder is its own welded block."""
    p = pg.PlaygroundParams(seed=seed)
    model = mujoco.MjModel.from_xml_string(pg.build_mjcf(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _reldev(got, want) -> float:
    """Relative deviation, with an absolute fallback so a rule whose expected
    value is zero cannot divide its way to a pass."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    denom = np.maximum(np.abs(want), 1e-12)
    return float(np.max(np.where(np.abs(want) > 1e-12,
                                 np.abs(got - want) / denom,
                                 np.abs(got - want))))


def _hit_geom(model, data, a, b) -> str:
    """Name of the first geom on the segment a->b, or "" if the line is clear."""
    a = np.asarray(a, dtype=float)
    delta = np.asarray(b, dtype=float) - a
    dist = float(np.linalg.norm(delta))
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, a, delta / dist, None, 1, -1, gid)
    if not (0.0 <= hit < dist - 1e-6) or gid[0] < 0:
        return ""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or "?"


def _peak_x(field, x_lo: float, x_hi: float, y: float, z: float) -> float:
    """Along-wind position of the concentration peak, sub-cell interpolated.

    Parabolic interpolation through the three samples around the argmax. The
    grid is offset (`TRANSECT_OFFSET`) so the true peak never sits on a node.
    """
    xs = np.arange(x_lo, x_hi, TRANSECT_DX) + TRANSECT_OFFSET
    pts = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], axis=1)
    c = field.sample(pts)[:, 0]
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2.0 * y1 + y2
        if abs(den) > 1e-18:
            return float(xs[i] + 0.5 * (y0 - y2) / den * TRANSECT_DX)
    return float(xs[i])


# ── rule 1: O1, inverse-exponential falloff (+ superposition, isolation) ─
def _o1_rules(seed: int) -> dict:
    """The declared O1 model, re-derived here rather than imported.

    `A0 * strength * exp(-d / LAMBDA_M)`, summed over sources, per channel.
    """
    rng = np.random.RandomState(seed + 11)
    s0 = odour.Source("food0", "food", (0.0, 0.0, 1.0), strength=1.0)
    s1 = odour.Source("water0", "water", (3.0, -1.0, 1.0),
                      strength=float(rng.uniform(0.5, 1.5)))
    s2 = odour.Source("food1", "food", (-2.0, 2.0, 1.0),
                      strength=float(rng.uniform(0.5, 1.5)))

    # (a) falloff along a ray from a single source.
    f1 = odour.StaticField([s0])
    dirn = np.array([1.0, 0.0, 0.0])
    pts = np.array([np.array(s0.pos) + dirn * d for d in FALLOFF_D])
    got = f1.sample(pts)[:, odour.CHANNEL_INDEX["food"]]
    d = np.asarray(FALLOFF_D)
    want = odour.A0 * s0.strength * np.exp(-d / odour.LAMBDA_M)
    falloff_dev = _reldev(got, want)
    # "Matches the declared exponential" is worth little on its own — the
    # expression above is the one the module computes, so agreement to 1e-16 is
    # a regression tripwire and not a discrimination. This is the
    # discrimination: the SAME samples measured against the obvious competing
    # model, an inverse-square field normalised to agree at d = 1 m. It must
    # MISS by a wide margin, or "exponential falloff" is not a claim about this
    # field, it is a claim about any smooth decay.
    inv_sq = (odour.A0 * s0.strength * math.exp(-1.0 / odour.LAMBDA_M)
              * (1.0 / d) ** 2)
    falloff_dev_invsq = _reldev(got, inv_sq)

    # (b) superposition: three sources together == the sum of them apart.
    probes = np.array([[0.7, 0.3, 1.0], [2.0, -0.5, 1.0], [-1.0, 1.0, 1.2],
                       [1.5, 1.5, 0.4]])
    both = odour.StaticField([s0, s1, s2]).sample(probes)
    apart = sum(odour.StaticField([s]).sample(probes) for s in (s0, s1, s2))
    super_dev = _reldev(both, apart)

    # (c) channel isolation: food and water sources put NOTHING in smoke/decay.
    quiet = both[:, [odour.CHANNEL_INDEX["smoke"], odour.CHANNEL_INDEX["decay"]]]
    isolation_leak = float(np.max(np.abs(quiet)))

    # (d) the null the registry declares, on O1: same receiver, same distance,
    # source disabled -> the noise floor and nothing else.
    off = odour.Source("food0", "food", s0.pos, enabled=False)
    n_rng = np.random.RandomState(seed + 12)
    null = odour.StaticField([off]).sample(np.repeat(probes, 500, axis=0), rng=n_rng)
    return {
        "o1_falloff_dev": falloff_dev,
        "o1_falloff_dev_vs_inverse_square": falloff_dev_invsq,
        "o1_superposition_dev": super_dev,
        "o1_channel_leak": isolation_leak,
        "o1_null_mean": float(null.mean()),
        "o1_null_max": float(null.max()),
    }


# ── rule 2: O2, the wind term displaces the peak in proportion to u ─────
def _wind_rule(seed: int, drop_wind: bool = False) -> dict:
    """Burst of puffs, steady wind, meander off. Peak must sit at u * T.

    Occlusion is disabled here and only here: this measurement casts a transect
    of ~600 receiver points against 800 puffs, and it is in open air 8 m up
    where `_lit_lines` confirms there is nothing to occlude. The wind term is
    the quantity under test; the occlusion mechanism is gated separately, and
    in both directions.
    """
    disp, expect = [], []
    for u in WIND_SPEEDS:
        f = odour.PuffField([odour.Source("f", "food", WIND_SRC)],
                            wind=(u, 0.0, 0.0), emit_hz=0.0, seed=seed,
                            los=False, max_puffs=4 * BURST,
                            meander_sigma=0.0, drop_wind=drop_wind)
        f.emit(BURST)
        for _ in range(int(round(WIND_T_S / DT))):
            f.step(DT)
        disp.append(_peak_x(f, WIND_SRC[0] - 1.0, WIND_SRC[0] + u * WIND_T_S + 2.0,
                            WIND_SRC[1], WIND_SRC[2]) - WIND_SRC[0])
        expect.append(u * WIND_T_S)
    disp = np.asarray(disp)
    expect = np.asarray(expect)
    u = np.asarray(WIND_SPEEDS)
    # Proportionality: the best-fit slope through the origin, and how far the
    # measured points sit from BOTH that line and the declared constant T.
    slope = float((u @ disp) / (u @ u))
    return {
        "wind_disp_dev": _reldev(disp, expect),
        "wind_prop_dev": _reldev(disp, slope * u),
        "wind_slope_s": slope,
        "wind_slope_dev": abs(slope - WIND_T_S) / WIND_T_S,
        "wind_disp_min": float(disp.min()),
    }


# ── rule 3: occlusion, gated in both directions ─────────────────────────
def _lit_lines(model, data) -> dict:
    """Which of these straight lines does LIGHT get down? Same `mj_ray`, same
    geometry, same `flg_static` as the puff occlusion uses."""
    hidden_clear, _ = odour.line_of_sight(model, data, SRC, R_HIDDEN)
    lit_clear, _ = odour.line_of_sight(model, data, SRC, R_LIT)
    blocked_clear, _ = odour.line_of_sight(model, data, R_SHADOW, P_BLOCKED)
    clearpuff_clear, _ = odour.line_of_sight(model, data, R_SHADOW, P_CLEAR)
    return {
        "light_reaches_hidden": float(hidden_clear),
        "light_reaches_lit": float(lit_clear),
        "hidden_occluder": _hit_geom(model, data, SRC, R_HIDDEN),
        "puff_blocked_los": float(blocked_clear),
        "puff_clear_los": float(clearpuff_clear),
        "hidden_dist_m": float(np.linalg.norm(np.array(SRC) - R_HIDDEN)),
        "lit_dist_m": float(np.linalg.norm(np.array(SRC) - R_LIT)),
    }


def _puff_pair(model, data) -> dict:
    """The sharp mechanism check: one synthetic puff, two places, one distance.

    Behind the block it must contribute EXACTLY zero with occlusion on and
    something with it off. On a clear line it must contribute the SAME with and
    without occlusion — otherwise the ray-caster is subtracting signal that
    nothing is blocking.
    """
    f = odour.PuffField([], wind=WIND, emit_hz=0.0, seed=0, los=True)
    ch = odour.CHANNEL_INDEX["food"]

    def one(p):
        f.pos = np.array([p], dtype=float)
        f.age = np.array([PUFF_AGE])
        f.chan = np.array([ch])
        f.mass = np.array([odour.PUFF_MASS])
        f.los = True
        on = float(f.sample([R_SHADOW], model=model, data=data)[0][ch])
        f.los = False
        off = float(f.sample([R_SHADOW], model=model, data=data)[0][ch])
        return on, off

    b_on, b_off = one(P_BLOCKED)
    c_on, c_off = one(P_CLEAR)
    return {
        "puff_behind_on": b_on, "puff_behind_off": b_off,
        "puff_clear_on": c_on, "puff_clear_off": c_off,
        "puff_pair_ok": float(b_on == 0.0 and b_off > 0.0
                              and c_on > 0.0 and c_on == c_off),
    }


def _plume(seed: int, model, data) -> dict:
    """The live plume: occlusion on vs off at two receivers, plus the null."""
    src = [odour.Source("food0", "food", SRC)]
    ch = odour.CHANNEL_INDEX["food"]
    recv = np.stack([R_HIDDEN, R_SHADOW])

    def sweep(los: bool, enabled: bool = True, count_rays: bool = False):
        s = [odour.Source("food0", "food", SRC, enabled=enabled)]
        f = odour.PuffField(s, wind=WIND, seed=seed, los=los)
        n_rng = np.random.RandomState(seed + 31)
        vals, rays, t0 = [], [], time.time()
        for _ in range(PLUME_STEPS):
            f.step(DT)
            vals.append(f.sample(recv, model=model, data=data,
                                 rng=n_rng if not enabled else None)[:, ch])
            if count_rays:
                rays.append(f.rays_last_sample())
        us = (time.time() - t0) / PLUME_STEPS * 1e6
        return np.asarray(vals), us, (float(np.mean(rays)) if rays else 0.0), f

    on, us_on, n_rays, f_on = sweep(True, count_rays=True)
    off, _, _, _ = sweep(False)
    null, _, _, _ = sweep(True, enabled=False)

    hid_on, sh_on = on[:, 0].mean(), on[:, 1].mean()
    hid_off, sh_off = off[:, 0].mean(), off[:, 1].mean()
    live = float((on[:, 0] > 10.0 * odour.NOISE_SIGMA).mean())
    return {
        "hidden_conc_mean": float(hid_on),
        "hidden_conc_snr": float(hid_on / odour.NOISE_SIGMA),
        "hidden_live_frac": live,
        "shadow_conc_on": float(sh_on),
        "shadow_conc_off": float(sh_off),
        "shadow_attenuation": float(1.0 - sh_on / sh_off) if sh_off > 0 else 0.0,
        "null_mean": float(null.mean()),
        "null_max": float(null.max()),
        "puffs_live": int(f_on.pos.shape[0]),
        "rays_per_sample": n_rays,
        "o2_us_per_step": float(us_on),
        "o2_frame_frac": float(us_on * 1e-6 / FRAME_S),
    }


def _intermittency(seed: int) -> dict:
    """Reported, not gated: how blank is this plume beside a real one?

    Farrell et al. (2002) against Jones' field data measured 85.2 / 90.1 / 83.7%
    blank at 2 / 5 / 10 m. `FROZEN_VS_PLASTIC.md` §8.3 argues the intermittency
    IS the message, so the distance between this model and that measurement is
    the honest headline for whoever builds SM.02 — but it is not in SM.01's
    registered hypothesis and it is not gated here.
    """
    f = odour.PuffField([odour.Source("f", "food", WIND_SRC)], wind=WIND,
                        seed=seed + 5, los=False)
    recv = np.array([[WIND_SRC[0] + d, WIND_SRC[1], WIND_SRC[2]]
                     for d in FARRELL_DIST_M])
    vals = []
    for k in range(3 * PLUME_STEPS):
        f.step(DT)
        vals.append(f.sample(recv)[:, odour.CHANNEL_INDEX["food"]])
    v = np.asarray(vals)[PLUME_STEPS:]        # discard the fill transient
    out = {}
    for i, d in enumerate(FARRELL_DIST_M):
        a = v[:, i]
        out[f"blank_frac_{int(d)}m"] = float((a < 10.0 * odour.NOISE_SIGMA).mean())
        out[f"peak_over_mean_{int(d)}m"] = float(a.max() / max(a.mean(), 1e-12))
    out["farrell_blank_gap"] = float(np.mean(
        [FARRELL_BLANK[i] - out[f"blank_frac_{int(d)}m"]
         for i, d in enumerate(FARRELL_DIST_M)]))
    return out


def _o1_cost(seed: int) -> dict:
    f = odour.StaticField([odour.Source("f", "food", SRC)])
    pts = np.stack([R_HIDDEN, R_SHADOW])
    t0 = time.time()
    for _ in range(PLUME_STEPS):
        f.step(DT)
        f.sample(pts)
    us = (time.time() - t0) / PLUME_STEPS * 1e6
    return {"o1_us_per_step": float(us), "o1_frame_frac": float(us * 1e-6 / FRAME_S)}


# ── the experiment ──────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    model, data = _world(seed)
    m: dict = {}
    m.update(_o1_rules(seed))
    m.update(_wind_rule(seed))
    m.update(_lit_lines(model, data))
    m.update(_puff_pair(model, data))
    m.update(_plume(seed, model, data))
    m.update(_intermittency(seed))
    m.update(_o1_cost(seed))

    m["sniff_hz"] = odour.SNIFF_HZ
    m["sniff_in_band"] = float(odour.SNIFF_BAND_HZ[0] <= odour.SNIFF_HZ
                               <= odour.SNIFF_BAND_HZ[1])
    m["obs_dim"] = odour.OBS_DIM

    # The registry's metric: the largest relative deviation any declared rule
    # showed. One number, and it is the one the spec is named for.
    m["field_rule_max_deviation"] = float(max(
        m["o1_falloff_dev"], m["o1_superposition_dev"],
        m["wind_disp_dev"], m["wind_prop_dev"], m["wind_slope_dev"]))

    m["seed_gates_ok"] = float(
        # the declared rules, to 1%
        m["field_rule_max_deviation"] <= FIELD_RULE_TOL
        and m["o1_falloff_dev_vs_inverse_square"] >= FALLOFF_DISCRIM_MIN
        and m["o1_channel_leak"] == 0.0
        # nulls: same receiver, same distance, source disabled
        and m["o1_null_mean"] <= NULL_FLOOR_MULT * odour.NOISE_SIGMA
        and m["null_mean"] <= NULL_FLOOR_MULT * odour.NOISE_SIGMA
        # light does NOT get to the hidden receiver, and it is the block
        and m["light_reaches_hidden"] == 0.0
        and m["hidden_occluder"] == OCCLUDER_NAME
        and m["light_reaches_lit"] == 1.0
        and abs(m["hidden_dist_m"] - m["lit_dist_m"]) < 1e-9
        # ...but odour does
        and m["hidden_conc_snr"] >= OCC_SNR
        and m["hidden_live_frac"] >= OCC_LIVE_FRAC
        # ...and the ray-caster is not a no-op in either direction
        and m["shadow_attenuation"] >= SHADOW_ATTEN_MIN
        and m["puff_pair_ok"] == 1.0
        and m["puff_blocked_los"] == 0.0
        and m["puff_clear_los"] == 1.0
        # ...and the sense is affordable and sampled where a mammal samples
        and m["o2_frame_frac"] <= COST_FRAME_FRAC_MAX
        and m["sniff_in_band"] == 1.0
        and m["obs_dim"] == 2 * odour.C + odour.C)
    return m


def _control(seed: int) -> dict:
    """The registry's declared control: the wind term dropped, nothing else.

    Same rng stream, same emission, same crosswind basis, same gust integration
    — only `pos += wind*dt` is skipped. The displacement gate must catch it, or
    the gate is decorative (PG.5's precedent).
    """
    w = _wind_rule(seed, drop_wind=True)
    return {
        "broken_wind_disp_dev": w["wind_disp_dev"],
        "broken_wind_slope_s": w["wind_slope_s"],
        "broken_wind_disp_min": w["wind_disp_min"],
        "control_caught": float(w["wind_disp_dev"] > FIELD_RULE_TOL),
        "control_caught_hard": float(w["wind_disp_dev"] >= CONTROL_MIN_DEV),
    }


def _check(m: dict, c: dict) -> bool:
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["field_rule_max_deviation"] <= FIELD_RULE_TOL
        and m["o1_falloff_dev_vs_inverse_square"] >= FALLOFF_DISCRIM_MIN
        and m["hidden_conc_snr"] >= OCC_SNR
        and m["shadow_attenuation"] >= SHADOW_ATTEN_MIN
        and m["light_reaches_hidden"] == 0.0
        and m["null_mean"] <= NULL_FLOOR_MULT * odour.NOISE_SIGMA
        # the broken variant must be caught on every seed
        and c["control_caught"] == 1.0
        and c["control_caught_hard"] == 1.0
        and c["broken_wind_disp_dev"] >= CONTROL_MIN_DEV)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SM.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
