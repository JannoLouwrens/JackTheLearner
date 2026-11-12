"""LAB — give him SMELL, raced against a smell that tells him nothing (no ledger row).

The J1/LAB finding (02ecf94): W0 gives him no sense of food before contact —
rays pass over floor food, there is no smell and no x,y. The eye-height race
(`food_sense.py`, e38780a) was KILLED by its declared rule: seeing floor food
on ~12% of poses did not make him approach it (s91-93 meals split 1/2/3 vs
split_blind 2/2/3). Smell is the other candidate in the order: a GRADIENT
that is non-zero everywhere, so a policy can climb it from any birth.

THE SENSE. `odour.StaticField` (O1, certified by SM.01: A0*exp(-d/2 m) per
source), one source per food geom in W0 (obj0, obj1 on the floor; the apple
on its tree — a creature smells the apple too), each source ENABLED only while
that food is ready to eat (drives' own respawn clock). Read by
`odour.OdourSensor`: bilateral nose 0.16 m apart at the body, facing the
body's +x, 12 floats [left C, right C, d(mean)/dt C], sensor noise 1e-3 from a
private rng. It enters him as a NEW modality `smell` (12 wide) with its own
encoder — his shape grows by one sense, per process: `cores.MODALITIES` and
`W0.observe` are patched in `install`, w0.py/cores.py on disk are untouched and
no certificate stales. O1, not O2 (puffs): the order asks first whether a
gradient he CAN climb gets climbed; intermittency is a later race.

ARMS (same seed, core, AWR policy loss, exploration, births — wean_pilot's
`null` arm, uniform legal spawn):
  smell           the sense above, read at his own nose
  smell_shuffled  CONTROL THAT MUST FAIL: the same field, the same sensor, the
                  same shape, but read at a uniformly random legal pose with a
                  random heading every decision — the marginal of smell over
                  the world, decoupled from where he is. If it eats as well as
                  `smell`, the meals came from the extra encoder, not from
                  smelling food.
The no-smell null is food_sense's `level` arm (today's W0, same learner, same
13k decisions, s91-93 meals 1/2/3 — already on disk).

DECLARED BEFORE ANY SMELL SEED IS READ (13k decisions/arm, seeds 91-93):
KILL `smell` if its meals summed over the three seeds are not greater than
BOTH smell_shuffled's sum and level's (6), or if it out-eats both on fewer
than 2 of the 3 seeds. Secondary reading, not gated: mean distance to the
nearest floor food while alive (`d_mean`), smell vs shuffled.

    nice -n 19 $PY -m experiments.smell_sense probe [seed]
    nice -n 19 $PY -m experiments.smell_sense pilot <arm> [seed] [total]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

ARMS = ("smell", "smell_shuffled")
SMELL_DIM = 12                 # OdourSensor.obs: [left C, right C, d/dt C], C=4


def install(arm: str, seed: int) -> None:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    from . import cores
    from . import w0 as W
    from .odour import OdourSensor, Source, StaticField

    # The core reads its key list at construction: grow it before build_arm.
    cores.MODALITIES["smell"] = SMELL_DIM
    cores.LOSS_SCALES["smell"] = 1.0
    orig_observe = W.W0.observe

    def _setup(self):
        dr = self.drives
        self._sm_names = list(dr._food)
        self._sm_src = [Source(name=n, channel="food",
                               pos=tuple(float(x) for x in
                                         self.data.geom_xpos[g]))
                        for n, (g, _) in dr._food.items()]
        self._sm_field = StaticField(self._sm_src)
        self._sm_nose = OdourSensor(self._sm_field)
        self._sm_rng = np.random.RandomState(7_177 + int(seed))
        self._sm_legal = self.legal_spawns()
        self._sm_last = None

    def _smell(self) -> np.ndarray:
        if not hasattr(self, "_sm_field"):
            _setup(self)
        dr = self.drives
        for s, n in zip(self._sm_src, self._sm_names):
            s.enabled = bool(dr.t >= dr._respawn_at[n])
        body = np.array(self.data.xpos[self.rover_bid], dtype=float)
        mat = np.array(self.data.xmat[self.rover_bid]).reshape(3, 3)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        if arm == "smell_shuffled":
            k = self._sm_rng.randint(len(self._sm_legal))
            body = np.array([self._sm_legal[k][0], self._sm_legal[k][1],
                             body[2]])
            yaw = float(self._sm_rng.uniform(-math.pi, math.pi))
        # a respawn is a new nose: no derivative across the jump
        if (self._sm_last is not None
                and np.linalg.norm(body[:2] - self._sm_last) > 1.0
                and arm == "smell"):
            self._sm_nose.reset()
        self._sm_last = body[:2].copy()
        return self._sm_nose.obs(body, yaw, float(dr.t),
                                 rng=self._sm_rng).astype(np.float32)

    def observe(self):
        obs = orig_observe(self)
        obs["smell"] = _smell(self)
        return obs

    W.W0.observe = observe


def probe(seed: int = 90) -> dict:
    """Is the field informative where he lives? Over every legal pose (food
    all ready): the food-channel mean, and whether the bilateral difference
    points toward the nearest floor food when he faces across it."""
    install("smell", seed)
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w = L.w
    w.observe()
    f = w._sm_field
    floor = np.array([s.pos[:2] for s in w._sm_src if s.pos[2] <= 0.5])
    legal = w._sm_legal
    vals, right = [], 0
    for x, y in legal:
        p = np.array([x, y, 0.6])
        c = f.sample(p[None])[0, 0]
        vals.append(c)
        t = floor[np.argmin(np.linalg.norm(floor - p[:2], axis=1))]
        ang = math.atan2(t[1] - y, t[0] - x)
        # face 90 deg right of the food: the LEFT nostril is nearer
        left, rgt = w._sm_nose.sites(p, ang - math.pi / 2)
        cl, cr = f.sample(np.stack([left, rgt]))[:, 0]
        right += int(cl > cr)
    vals = np.array(vals)
    rec = {"seed": seed, "poses": int(len(legal)),
           "food_mean": round(float(vals.mean()), 4),
           "food_min": round(float(vals.min()), 4),
           "food_max": round(float(vals.max()), 4),
           "bilateral_points_to_food": round(right / len(legal), 3)}
    print(json.dumps(rec), flush=True)
    return rec


def pilot(arm: str, seed: int = 90, total: int = 13000) -> None:
    install(arm, seed)
    from . import wean_pilot
    wean_pilot.main("null", seed, 0, total,
                    f"/data/tmp/smell_sense/{arm}_s{seed}.jsonl")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "probe":
        probe(int(a[1]) if len(a) > 1 else 90)
    elif a[0] == "pilot":
        pilot(a[1], int(a[2]) if len(a) > 2 else 90,
              int(a[3]) if len(a) > 3 else 13000)
    else:
        raise SystemExit(__doc__)
