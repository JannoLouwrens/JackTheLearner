"""LAB — can a food sense be given to him without changing his shape? (no ledger row)

The J1/LAB finding (02ecf94): in W0 his 16 rays leave horizontally at the body
origin + 0.05 m (z ~0.70), floor food tops out at ~0.33 m, so 0 of 584 legal
poses let any ray touch floor food. `LF.04` asks whether living teaches him; a
creature that cannot perceive food before bumping into it can only learn
stillness (the AWR pilot: 536 s, 1 meal in 12 lives). That is a WORLD repair
(his body's eyes), and it is raced here, not assumed.

ARMS — every arm keeps `vision` 32 wide (16 rays x [distance, texture]), so
the core, its encoders and every seat stay byte-identical; only where the rays
leave from changes. The patch is per process (`install`), w0.py is untouched
and no certificate stales.
  level   today's W0: 16 level rays from z_body + 0.05         (the null)
  low     16 level rays from LOW_Z above the ground (whisker / snout height)
  split   even rays as `level`, odd rays as `low` (keeps the high view of
          walls and panel on half the spokes)
  split_blind  CONTROL THAT MUST FAIL: `split`'s eyes, but every ray passes
          THROUGH floor food (re-cast 0.35 m past it) — same low view of the
          floor and walls, no food. If it eats as well as `split`, the meals
          came from the eye height, not from seeing food.

STEP 1 `perceive`: over every legal spawn pose, the fraction of poses where at
least one ray's first hit is a floor-food geom, and the nearest such hit.
The `level` arm must read 0 (the reproduction of the LAB finding — the
control that must fail to see).
STEP 2 `pilot`: the AWR learner (wean_pilot's `null` arm: uniform births,
nothing staged) living `total` decisions with the arm's eyes; meals/1000 dec.

SEED-90 PILOT (13k decisions/arm, AWR): meals level 0 (= wean_pilot null),
low 0, split 3 (lives 1 and 3). One seed: an anecdote, so —
DECLARED BEFORE SEEDS 91-93 ARE READ (13k decisions/arm, arms level, split,
split_blind): KILL `split` if its meals summed over the three seeds are not
greater than BOTH level's and split_blind's sums, or if it out-eats both on
fewer than 2 of the 3 seeds. split_blind verified through the real _vision:
blind on 76/76 seed-90 poses where split sees food, identical on 535/535 others.

    nice -n 19 $PY -m experiments.food_sense perceive <arm> [seed]
    nice -n 19 $PY -m experiments.food_sense pilot <arm> [seed] [total]
"""
from __future__ import annotations

import json
import sys

import numpy as np

ARMS = ("level", "low", "split", "split_blind")
LOW_Z = 0.15           # m above the ground: under the 0.33 m top of floor food
FLOOR_Z = 0.5          # a food geom centred above this is not floor food


def install(arm: str) -> None:
    """Patch W0._vision for this process. `level` leaves it untouched."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    if arm == "level":
        return
    from . import w0 as W

    def _vision(self) -> np.ndarray:
        body = np.array(self.data.xpos[self.rover_bid], dtype=float)
        hi = body.copy()
        hi[2] += W.RAY_Z_OFFSET
        lo = body.copy()
        lo[2] = LOW_Z
        out = np.empty(W.MODALITIES["vision"], dtype=np.float32)
        if not hasattr(self, "_fs_floor"):
            self._fs_floor = {g for g, _ in self.drives._food.values()
                              if self.data.geom_xpos[g][2] <= FLOOR_Z}
        for k in range(W.N_RAYS):
            pos = lo if (arm == "low" or k % 2 == 1) else hi
            dist = self.mujoco.mj_ray(self.model, self.data, pos,
                                      self._ray_dirs[k], W._RAY_GROUPS, 1,
                                      -1, self._geomid)
            gid = int(self._geomid[0])
            if arm == "split_blind" and gid in self._fs_floor and dist >= 0:
                skip = dist + 0.35
                d2 = self.mujoco.mj_ray(self.model, self.data,
                                        pos + skip * self._ray_dirs[k],
                                        self._ray_dirs[k], W._RAY_GROUPS, 1,
                                        -1, self._geomid)
                gid = int(self._geomid[0])
                dist = d2 + skip if d2 >= 0 else -1.0
            if dist < 0 or dist > W.RAY_MAX_RANGE or gid < 0:
                d, t = 1.0, 0.0
            else:
                d = dist / W.RAY_MAX_RANGE
                t = float(self._tex[gid])
                if gid == self.panel_gid:
                    amp = min(1.0, 2.5 / max(dist, 1e-6))
                    t = 0.5 + amp * (float(self._rng.uniform()) - 0.5)
            out[2 * k] = d
            out[2 * k + 1] = t
        return out

    W.W0._vision = _vision


def _ray_gids(w, arm):
    """(gid, dist) of each ray's first hit, by the arm's geometry."""
    from . import w0 as W
    body = np.array(w.data.xpos[w.rover_bid], dtype=float)
    hits = []
    gid_buf = np.zeros(1, dtype=np.int32)
    for k in range(W.N_RAYS):
        pos = body.copy()
        if arm == "low" or (arm.startswith("split") and k % 2 == 1):
            pos[2] = LOW_Z
        else:
            pos[2] += W.RAY_Z_OFFSET
        d = w.mujoco.mj_ray(w.model, w.data, pos, w._ray_dirs[k],
                            W._RAY_GROUPS, 1, -1, gid_buf)
        hits.append((int(gid_buf[0]), float(d)))
    return hits


def perceive(arm: str, seed: int = 90) -> dict:
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w = L.w
    floor = {gid for gid, _ in w.drives._food.values()
             if w.data.geom_xpos[gid][2] <= FLOOR_Z}
    legal = w.legal_spawns()
    seen, dists, rays_on_food = 0, [], 0
    for x, y in legal:
        w.respawn(at=(float(x), float(y)))
        hits = [(g, d) for g, d in _ray_gids(w, arm)
                if g in floor and 0 <= d <= 8.0]
        rays_on_food += len(hits)
        if hits:
            seen += 1
            dists.append(min(d for _, d in hits))
    rec = {"arm": arm, "seed": seed, "poses": int(len(legal)),
           "n_floor_food": len(floor), "poses_seeing_food": seen,
           "frac": round(seen / len(legal), 3),
           "rays_on_food_per_pose": round(rays_on_food / len(legal), 3),
           "nearest_seen_m": (round(float(np.median(dists)), 2)
                              if dists else None)}
    print(json.dumps(rec), flush=True)
    return rec


def pilot(arm: str, seed: int = 90, total: int = 13000) -> None:
    install(arm)
    from . import wean_pilot
    wean_pilot.main("null", seed, 0, total,
                    f"/data/tmp/food_sense/{arm}_s{seed}.jsonl")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "perceive":
        perceive(a[1], int(a[2]) if len(a) > 2 else 90)
    elif a[0] == "pilot":
        pilot(a[1], int(a[2]) if len(a) > 2 else 90,
              int(a[3]) if len(a) > 3 else 13000)
    else:
        raise SystemExit(__doc__)
