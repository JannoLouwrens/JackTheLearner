"""plants.py — the poison fixture: two plants, and the difference you cannot see.

Certified by `TA.01`. Built for `TA.02` (conditioned taste aversion from ONE
exposure), which is the first spec in this project that would be novel work if
it passes — `docs/research/FROZEN_VS_PLASTIC.md` §8.4 searched the RL and ALife
literatures and found no agent implementing a taste-specific one-trial
associator with an hours-long eligibility window. A claim like that is only
worth as much as the fixture under it, which is why the fixture is certified
first and separately.

WHAT THIS MODULE OWNS

    two plant types   visually drawn from ONE distribution, so a probe on the
                      rendered frame cannot name the type; distinct in a taste
                      vector t in R^5 (sweet, bitter, sour, salt, umami —
                      caveman, not chemistry, per GOAL.md's "realistic means
                      what it meant to a caveman")
    a dose-response   the toxin's declared curve: how much integrity a dose of
                      q costs, when it arrives, and over how long
    the delay D       ingest now, sicken later. This is the quantity TA.02's
                      difficulty scales with, and it is declared HERE so that
                      TA.02 cannot quietly shorten it.

WHY VISUAL IDENTITY IS THE WHOLE POINT. If the toxic plant looked different,
TA.02 would be a colour-discrimination task with a slow reward, which vision
plus ordinary RL solves and which proves nothing about taste. Garcia &
Koelling's result — the reason CTA is worth building at all — is that the
taste->illness pairing is PRIVILEGED, not that one-shot learning happens.
A fixture whose types are separable by eye cannot test that.

WHY IT IS NOT ENOUGH TO ASSERT THE TYPES LOOK ALIKE. Two geoms with identical
parameters are identical trivially, and a probe scoring chance on them
certifies nothing — it is the "control that passes by construction" this repo
has been burned by (PG.6's out-of-FOV control, first drafted as "put it behind
the camera", where every frame is byte-identical). So every plant carries real
per-individual variation — stem height, berry radius, berry offset, shade,
bearing, distance — drawn from ONE distribution that does not know the type.
The probe therefore has plenty to read, and TA.01 checks that it DOES read it
(plant radius must be recoverable) before believing that it cannot read type.

THE ILLNESS IS NOT A REWARD. It is a delayed interoceptive insult: a hit to
integrity `i`, the same scalar `drives.DriveLayer` integrates and the same one
`w0.DEATH_FLOOR` kills at. Nothing here writes a reward, and nothing here steps
physics — the caller owns `mj_step`, exactly as `drives` does, for the same
reason (a layer that owned the loop would be a second copy of the stepping code).

THE CONSTANTS. Two kinds, and the difference matters:

  * `TASTE_*`, `Q_FIRST`, `I_MAX`, `Q50`, `HILL`, `ILL_WINDOW_S` are the
    fixture's DECLARATION. They are choices, and TA.01's gates constrain them:
    the first bite must be felt and survivable, a whole plant must be able to
    kill, and the curve must be monotone. A change that breaks any of those
    fails TA.01 rather than silently making TA.02 easier.
  * `DELAY_S` is DERIVED, and the derivation is the biology. Rat CTA tolerates
    1-6 h reliably (Riley, Hempel & Clasen, Psychon. Bull. Rev. 25:429-441,
    2018), against a starvation horizon of roughly 3 days — a delay of 1.4% to
    8.3% of the time it takes the animal to starve. This world's starvation
    horizon is `1 / drives.BASAL_B` = 600 s at rest, so the same fraction band
    is 8.3-50 s. `DELAY_S = 30 s` sits mid-band, and TA.01 gates on the
    FRACTION, not on the seconds, so the delay tracks the world if the world's
    metabolism is ever recalibrated.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not decide, does not learn,
and carries no aversion state. The associator is TA.02's, and it must be
ablatable: `FROZEN_VS_PLASTIC.md` §8.4 requires deleting taste to cost him the
one-trial capability. A fixture that already knew which plant was bad would
hand TA.02 its answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import drives

# ── the taste vector ────────────────────────────────────────────────────
# Five channels because five is what a tongue has. Emitted ONLY on ingestion
# (FROZEN_VS_PLASTIC.md §8.4), never as a standing observation: a creature does
# not taste what it has not eaten, and a taste channel that is always on would
# let a policy read plant identity without taking the risk that makes the
# learning problem a learning problem.
TASTE_DIM = 5
TASTE_KEYS = ("sweet", "bitter", "sour", "salt", "umami")

# Within-type variation. Individual plants differ, so the taste channel is a
# noisy measurement rather than a lookup table with the answer in it.
TASTE_SIGMA = 0.06


@dataclass(frozen=True)
class PlantType:
    """One kind of plant. `potency` scales the dose-response curve; 0 is safe.

    `taste_mu` is the mean taste vector. The two types differ mostly in BITTER,
    which is the alkaloid signature every foraging animal on Earth reads, and
    they overlap in the other four channels so the discrimination is a real one
    rather than a one-hot with a costume on.
    """
    name: str
    taste_mu: Tuple[float, ...]
    potency: float

    def taste(self, rng: np.random.RandomState) -> np.ndarray:
        t = np.asarray(self.taste_mu, dtype=float) + rng.normal(0.0, TASTE_SIGMA, TASTE_DIM)
        return np.clip(t, 0.0, 1.0)


SAFE = PlantType("safe", (0.70, 0.10, 0.30, 0.05, 0.20), 0.0)
TOXIC = PlantType("toxic", (0.65, 0.55, 0.35, 0.05, 0.15), 1.0)
TYPES = (SAFE, TOXIC)

# ── the dose-response curve, declared ───────────────────────────────────
# dose q in [0, 1] is the fraction of ONE plant ingested.
#
#     dI(q) = I_MAX * q^HILL / (Q50^HILL + q^HILL)      total integrity lost
#
# Q_FIRST is the neophobic first bite. Neophobia is modelled as a reduced
# PORTION and not as a sampling ritual, because that is what the measurements
# actually show (Lin, Arthurs & Reilly 2012: intake rises ~3.7x over 2-3
# exposures; Modlinska et al. 2016 found no sampling behaviour at all in a wild
# colony). A first bite is therefore ~1/7th of a plant.
Q_FIRST = 0.15
I_MAX = 1.30       # asymptote: more than the whole integrity range, so a full
                   # plant is lethal and "sub-lethal" is a fact about the DOSE
Q50 = 0.35         # half-max at about a third of a plant
HILL = 2.0         # mild sigmoid: no cliff, so the response is learnable

# Delay and duration of the malaise. See the module docstring for DELAY_S's
# derivation; the 1-6 h CTA band maps to 8.3-50 s here.
DELAY_S = 30.0
ILL_WINDOW_S = 20.0

# The biological band DELAY_S must sit inside, as a fraction of this world's
# starvation horizon. TA.01 gates on this rather than on the seconds.
DELAY_FRAC_BAND = (0.014, 0.083)


def illness_total(dose: float, potency: float) -> float:
    """Total integrity cost of ingesting `dose` of a plant with `potency`."""
    q = float(np.clip(dose, 0.0, 1.0))
    if potency <= 0.0 or q <= 0.0:
        return 0.0
    return float(potency * I_MAX * q ** HILL / (Q50 ** HILL + q ** HILL))


def delay_fraction() -> float:
    """DELAY_S as a fraction of the resting starvation horizon, 1 / BASAL_B."""
    return float(DELAY_S * drives.BASAL_B)


class Toxin:
    """Delayed malaise, scheduled by ingestion. The caller owns time.

    Usage mirrors `drives.DriveLayer`: this object never advances the clock and
    never touches physics. It is asked what the ingestions so far cost during
    the interval `[t, t + dt)`, and the caller subtracts that from integrity.

        tox = Toxin()
        tox.ingest(t=12.0, plant=TOXIC, dose=Q_FIRST)
        ...
        di = tox.rate(t) * dt          # integrity lost this step, >= 0

    Several ingestions overlap by summing, which is the correct behaviour and
    not a convenience: Kwok & Boakes showed a second novel taste inside the
    delay window overshadows the first aversion in a single trial, so an agent
    that eats several things before falling ill SHOULD have smeared credit.
    """

    def __init__(self) -> None:
        # (onset_t, end_t, rate) per ingestion that carries any toxin.
        self._events: List[Tuple[float, float, float]] = []
        self.ingestions: List[Tuple[float, str, float]] = []

    def ingest(self, t: float, plant: PlantType, dose: float) -> float:
        """Record an ingestion. Returns the integrity it will eventually cost."""
        total = illness_total(dose, plant.potency)
        self.ingestions.append((float(t), plant.name, float(dose)))
        if total > 0.0:
            self._events.append((float(t) + DELAY_S,
                                 float(t) + DELAY_S + ILL_WINDOW_S,
                                 total / ILL_WINDOW_S))
        return total

    def rate(self, t: float) -> float:
        """Integrity lost per second at time `t`, summed over live events."""
        return float(sum(r for a, b, r in self._events if a <= t < b))

    def pending(self, t: float) -> float:
        """Integrity still owed at time `t` — what he has coming, unfelt."""
        owed = 0.0
        for a, b, r in self._events:
            if t < a:
                owed += r * (b - a)
            elif t < b:
                owed += r * (b - t)
        return float(owed)


@dataclass
class Bout:
    """One simulated ingestion, integrated against the world's own integrity
    dynamics: `drives.RHO_HEAL` heals, `drives` clips to [0, 1], and
    `w0.DEATH_FLOOR` is where the life ends. Every constant is imported live —
    a calibration pasted into a second file is a constant that drifts from its
    measurement (T0.14, T0.22)."""
    t: np.ndarray
    i: np.ndarray
    died: bool
    onset_t: float          # first time integrity moved, or -1 if it never did
    i_min: float
    i_end: float


def ingest_bout(plant: PlantType, dose: float, *, horizon_s: float = 400.0,
                dt: float = 0.2, i0: float = 1.0, resting: bool = True,
                death_floor: float = 0.0, eps: float = 1e-9) -> Bout:
    """Integrate one ingestion at t=0 and report what integrity did.

    `dt` defaults to W0's decision length (0.2 s), so the onset time this
    reports is resolvable at the rate Jack actually experiences the world, not
    at a resolution no agent has access to.
    """
    tox = Toxin()
    tox.ingest(0.0, plant, dose)
    n = int(round(horizon_s / dt))
    ts = np.arange(n + 1, dtype=float) * dt
    i = np.empty(n + 1, dtype=float)
    i[0] = float(i0)
    died = False
    onset = -1.0
    for k in range(n):
        cur = i[k]
        if died:
            i[k + 1] = cur
            continue
        loss = tox.rate(ts[k]) * dt
        heal = drives.RHO_HEAL * dt if resting else 0.0
        nxt = float(np.clip(cur + heal - loss, 0.0, 1.0))
        if onset < 0.0 and loss > eps:
            # The START of the interval in which the toxin was active: the
            # schedule's onset, not the step at which the drop became readable.
            # TA.01 gates this against DELAY_S at the resolution of a decision.
            onset = float(ts[k])
        if nxt <= death_floor:
            nxt, died = death_floor, True
        i[k + 1] = nxt
    return Bout(t=ts, i=i, died=died, onset_t=onset,
                i_min=float(i.min()), i_end=float(i[-1]))


# ── the plants, as geometry ─────────────────────────────────────────────
# Every draw below is TYPE-INDEPENDENT in the real fixture. `colour_coded=True`
# is TA.01's positive control: the same draws, the same geometry, one thing
# changed — the berry hue tracks the type — which the same probe must catch.
STEM_H = (0.35, 0.70)       # m
BERRY_R = (0.10, 0.20)      # m
BERRY_OFF = 0.06            # m, lateral jitter of the cluster on the stem
SHADE = 0.06                # +- multiplicative-ish jitter on the berry rgb
STEM_RGB = (0.25, 0.42, 0.22)
BERRY_RGB = (0.42, 0.24, 0.45)
CODED_RGB = (0.88, 0.16, 0.12)      # the control's "this one is poisonous" red

PLANT_BODY = "plant"
PLANT_BERRY = "plant_berry"
PLANT_STEM = "plant_stem"


@dataclass(frozen=True)
class PlantDraw:
    """One individual plant. `type_index` is the label a probe must recover."""
    type_index: int
    stem_h: float
    berry_r: float
    off_x: float
    off_y: float
    shade: float
    bearing_deg: float
    dist_m: float

    @property
    def plant(self) -> PlantType:
        return TYPES[self.type_index]


def draw_plant(rng: np.random.RandomState, type_index: int,
               bearing_band: Tuple[float, float], dist_band: Tuple[float, float]
               ) -> PlantDraw:
    """Sample one plant. `type_index` enters the taste and the toxin — and
    nothing else. If it ever enters a line below, TA.01 is what catches it."""
    b = rng.uniform(*bearing_band)
    if rng.rand() < 0.5:
        b = -b
    return PlantDraw(type_index=int(type_index),
                     stem_h=rng.uniform(*STEM_H),
                     berry_r=rng.uniform(*BERRY_R),
                     off_x=rng.uniform(-BERRY_OFF, BERRY_OFF),
                     off_y=rng.uniform(-BERRY_OFF, BERRY_OFF),
                     shade=rng.uniform(-SHADE, SHADE),
                     bearing_deg=b,
                     dist_m=rng.uniform(*dist_band))


def berry_rgba(d: PlantDraw, colour_coded: bool = False) -> Tuple[float, ...]:
    base = CODED_RGB if (colour_coded and d.type_index == 1) else BERRY_RGB
    return tuple(float(np.clip(c + d.shade, 0.0, 1.0)) for c in base) + (1.0,)


def plant_mjcf() -> str:
    """The plant, at the origin. Sizes, colours and the body pose are edited on
    the compiled model, so a thousand plants cost one MJCF compile.

    No joint: the body is static and moved through `model.body_pos`. A freejoint
    would add dofs to a world whose observation width other specs depend on.
    """
    sx, sy, sz = STEM_RGB
    return (f'<body name="{PLANT_BODY}" pos="0 0 0">'
            f'<geom name="{PLANT_STEM}" type="capsule" fromto="0 0 0 0 0 0.25" '
            f'size="0.012" rgba="{sx} {sy} {sz} 1" contype="0" conaffinity="0"/>'
            f'<geom name="{PLANT_BERRY}" type="sphere" pos="0 0 0.25" size="0.07" '
            f'rgba="{BERRY_RGB[0]} {BERRY_RGB[1]} {BERRY_RGB[2]} 1" '
            f'contype="0" conaffinity="0"/>'
            f'</body>')


def with_plant(xml: str) -> str:
    """Insert one plant into a playground MJCF string.

    Done here rather than in `playground.build_mjcf` on purpose. Plants are W1
    content (`FROZEN_VS_PLASTIC.md` §8.3: wire the channel at W0, add the
    content at W1) and `playground.py` is hashed into the `impl_sha` of nine
    specs — putting a fixture that only TA.01/TA.02 use into it would mark all
    nine stale to certify a plant. `hns_scene.py` is the precedent: a scene that
    belongs to one claim lives with that claim.
    """
    if xml.count("</worldbody>") != 1:
        raise ValueError("playground MJCF has no single worldbody to extend — "
                         "the template changed and this insertion is unsafe")
    return xml.replace("</worldbody>", plant_mjcf() + "\n  </worldbody>")
