"""The heard-not-seen (HNS) scene: one bit of PURE audio-visual synergy.

Built for UB.9 ("Heard, not seen: the task that is impossible without fusion")
and certified by PG.7. The scene is deliberately tiny — a floor and two
spheres — because its value is not richness but the exactness of what it does
and does not encode:

    audio  -> WHICH SIZE fell   (modal fundamental; ContactAudio.py:97-99)
    vision -> WHICH SIZE is WHERE  (a frame captured BEFORE the release)
    fused  -> WHICH SLOT fell   (the answer; 1 bit, unavailable to either alone)

I(audio;Y) = I(vision;Y) = 0 and I(audio,vision;Y) = 1 bit — a physical XOR.
Because the size->slot assignment is redrawn every episode, hearing "the small
one fell" says nothing about where the small one was standing.

WHY THE NUISANCE PARAMETERS ARE SHARED BETWEEN SLOTS.  Episodes must vary or a
probe memorises four fixed waveforms; but any per-slot variation is a leak. So
every episode draws its nuisance parameters ONCE — azimuth, listener range,
listener yaw, fall height — and applies them to BOTH slots. Within an episode
the two slots are exact mirror images (pan equal to float round-off, listener
distance equal); across episodes the audio varies freely. Variation and
non-leakage are therefore not in tension: the variation is common-mode.

THE MIRROR.  ContactAudio pans by ``p = -sin(azimuth)`` (ContactAudio.py:26,
:197), and ``sin`` is invariant under ``theta -> pi - theta``. Slot A sits at
azimuth ``+theta`` and slot B at ``pi - theta`` — in front of and behind the
listener's lateral axis, at equal range. This is exactly the front-back
confusion the synth's docstring declares out of scope, used on purpose.

EQUAL FALL DISTANCE, NOT EQUAL DROP HEIGHT.  A sphere resting on the floor has
its centre at z = r, so releasing both candidates from the same absolute height
gives the smaller one a longer fall and a harder landing — an amplitude cue for
size. Both candidates therefore start at ``z = r + fall_h``: the same distance
to fall, the same impact speed, and with equal mass the same impact force.
(UNIFIED_BRAIN_BAKEOFF.md 3.2 says "equal drop height"; equal fall DISTANCE is
what actually equalises the impact, and PG.7 measures the amplitude match
rather than assuming it.)

The decoy candidate is welded to the world, so MuJoCo discards its contact with
the (also static) floor and it cannot sound. PG.7 asserts that anyway.

MEASURED, NOT ARGUED — why the radii are not the ones the design doc named.
UNIFIED_BRAIN_BAKEOFF.md 3.1 proposed r = 0.07 and 0.16 and argued that the
resulting difference in audible mode count (2 vs 3) "is not an amplitude cue,
because `_voice` renormalises by the total gain of the *included* modes". PG.7
measured the opposite: that renormalisation is exactly what CREATES the cue.
`sig *= amp / total_gain` (ContactAudio.py:165-166) divides by the gain of the
modes that survived the 7200 Hz cutoff — 1.50 for two modes, 1.75 for three —
so the two-mode voice comes out 15% louder at identical impact force, and a
logistic probe on window level alone named the size on 70% of episodes against
a 53% gate. Identity was riding on LOUDNESS as well as on spectrum, which would
have let UB.9's fusion arms bind f0 to nothing. Both radii here ring with three
modes, `total_gain` is identical, and the level cue measures 0.0.

Deliberately leaky variants, used ONLY as PG.7's positive controls — a leak
detector that has never seen a leak is not a detector:

    Leak.GEOMETRY   slot B un-mirrored and further away -> pan and level track SLOT
    Leak.MASS       mass by volume -> impact amplitude tracks SIZE
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Tuple

# Radii chosen so BOTH voices ring with exactly THREE audible modes. See the
# "MEASURED, NOT ARGUED" note above: the mode count, not the radius, is what
# has to match. f0 = 180/r (ContactAudio.py:97-99); a mode is audible while
# f0 * ratio < 0.45 * sr = 7200 Hz, and MODE_RATIOS[2:4] = (5.40, 8.93), so
# exactly three modes sound for f0 in [806, 1333] Hz -> r in [0.135, 0.223] m.
# Both sit mid-band with >280 Hz of margin at each edge.
R_SMALL = 0.140625          # -> f0 1280 Hz, 3 audible modes
R_LARGE = 0.214286          # -> f0  840 Hz, 3 audible modes (1.52x separation)
MASS_KG = 0.5               # both candidates, in the balanced fixture
LISTENER_Z = 1.4
FLOOR_HALF = 8.0


class Leak(enum.Enum):
    """Which confound is deliberately re-opened. NONE is the real fixture."""
    NONE = "none"
    GEOMETRY = "geometry"   # breaks the mirror: pan and distance track slot
    MASS = "mass"           # mass by volume: impact amplitude tracks size


@dataclass(frozen=True)
class HnsEpisode:
    """One episode's full specification. The nuisance draw is shared by slots;
    only ``large_slot`` and ``faller_slot`` carry information, and they are
    drawn independently — which is why neither modality alone can answer."""
    theta: float            # azimuth of slot A in the listener frame, rad
    rng_range: float        # listener->slot horizontal range, m
    yaw: float              # listener yaw, rad
    fall_h: float           # fall distance for whichever candidate is released
    large_slot: int         # 0 or 1 — where the big sphere stands
    faller_slot: int        # 0 or 1 — THE LABEL
    leak: Leak = Leak.NONE

    @property
    def faller_radius(self) -> float:
        return R_LARGE if self.faller_slot == self.large_slot else R_SMALL


def slot_positions(ep: HnsEpisode) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """World (x, y) of slots 0 and 1, listener at the origin with yaw ``ep.yaw``.

    Balanced: azimuths ``+theta`` and ``pi - theta`` at equal range, so
    ``sin(azimuth)`` — and therefore the pan — is identical.
    Leak.GEOMETRY: slot 1 mirrored in the WRONG axis (``-theta``) and pushed
    0.6 m further out, so both pan and 1/distance track the slot.
    """
    R = ep.rng_range
    az0, az1 = ep.theta, math.pi - ep.theta
    R0, R1 = R, R
    if ep.leak is Leak.GEOMETRY:
        az1, R1 = -ep.theta, R + 0.6
    out = []
    for az, rad in ((az0, R0), (az1, R1)):
        world = ep.yaw + az
        out.append((rad * math.cos(world), rad * math.sin(world)))
    return out[0], out[1]


def _mass_for(radius: float, leak: Leak) -> float:
    if leak is Leak.MASS:
        return MASS_KG * (radius / R_SMALL) ** 3
    return MASS_KG


def hns_mjcf(ep: HnsEpisode) -> str:
    """MJCF for one episode. The released candidate gets a free joint; the other
    is welded to the world (a held object, and acoustically silent)."""
    (x0, y0), (x1, y1) = slot_positions(ep)
    radii = [R_SMALL, R_SMALL]
    radii[ep.large_slot] = R_LARGE
    bodies = []
    for slot, (x, y, r) in enumerate(((x0, y0, radii[0]), (x1, y1, radii[1]))):
        falls = slot == ep.faller_slot
        z = r + ep.fall_h
        joint = '<freejoint/>' if falls else ''
        rgba = "0.85 0.35 0.25 1" if slot == 0 else "0.25 0.45 0.85 1"
        bodies.append(
            f'    <body name="cand{slot}" pos="{x:.9f} {y:.9f} {z:.9f}">\n'
            f'      {joint}\n'
            f'      <geom name="cand{slot}" type="sphere" size="{r:.6f}" '
            f'mass="{_mass_for(r, ep.leak):.6f}" rgba="{rgba}" '
            f'friction="0.9 0.005 0.0001"/>\n'
            f'    </body>')
    return (
        '<mujoco model="hns">\n'
        '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        '  <visual><global offwidth="640" offheight="480"/></visual>\n'
        '  <worldbody>\n'
        '    <light pos="0 0 6" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>\n'
        f'    <geom name="floor" type="plane" size="{FLOOR_HALF} {FLOOR_HALF} 0.1" '
        'rgba="0.55 0.55 0.58 1" friction="0.9 0.005 0.0001"/>\n'
        + '\n'.join(bodies) + '\n'
        '  </worldbody>\n'
        '</mujoco>\n')


def listener_pose(ep: HnsEpisode):
    """Where the ears are. The listener never moves within an episode."""
    return (0.0, 0.0, LISTENER_Z), ep.yaw


def build(ep: HnsEpisode):
    """Compile the episode and place it at t=0. Returns (model, data)."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(hns_mjcf(ep))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def draw_quad(rng, leak: Leak = Leak.NONE):
    """ONE nuisance draw, expanded into the four (large_slot, faller_slot) cells.

    The quad is the unit of sampling for two reasons. (1) Balance: a batch of k
    quads is EXACTLY balanced in both labels and in their interaction, so no
    class prior and no nuisance-label correlation exists for a probe to
    exploit — the audio-only null is at chance by construction, not by
    averaging. (2) Pairing: the fixture assertions (pan, listener distance,
    impact amplitude) compare episodes that differ ONLY in the label, which is
    the comparison the leak table in UNIFIED_BRAIN_BAKEOFF.md 3.2 asks for.
    """
    theta = float(rng.uniform(math.radians(30.0), math.radians(50.0)))
    rng_range = float(rng.uniform(1.6, 2.6))
    yaw = float(rng.uniform(-math.pi, math.pi))
    fall_h = float(rng.uniform(0.40, 0.60))
    return [HnsEpisode(theta=theta, rng_range=rng_range, yaw=yaw, fall_h=fall_h,
                       large_slot=i % 2, faller_slot=i // 2, leak=leak)
            for i in range(4)]
