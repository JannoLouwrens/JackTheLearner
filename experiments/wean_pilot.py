"""LAB pilot — WEANING: a curriculum that stages where he is BORN (no ledger row).

Seat: the curriculum (ORDERS J5; no seat yet, null = "dropped into W0 and
scored"). The J1 finding this answers: in lethal W0 the learner meets food
once in 12 lives, so living carries no signal that food ends hunger — the AWR
learner learns stillness (536 s) and never forages.

THE BET. A mammal is not born a forager: it is born AT the food, and the
distance grows as it succeeds. Arm `nursed`: an adult of his world (the
mother — a spawn rule, not a part of him and not his actions) puts each new
body within radius r of a floor food that is ready to eat; a life with a meal
widens r x1.5, a life without narrows it /1.5 (floor 0.5 m, cap 6 m). After
NURSE decisions the mother leaves: uniform legal spawn, the standing W0 rule,
for every arm. He never stops learning in either phase.

ARMS (same seed, core, AWR policy loss, exploration, cadence, world):
  null     uniform legal spawn throughout (today's W0)
  nursed   weaned as above for the first NURSE decisions, then uniform
  decoy    CONTROL THAT MUST FAIL: the same rule aimed at fixed non-food
           points (>= 2 m from every food) — staged births that teach nothing

READING (declared before the run): in the WEANED window (lives born under
uniform spawn after decision NURSE; the same decision window for null) —
meals per 1000 decisions, and mean distance to the nearest floor food.
KILL: nursed does not eat more than BOTH null and decoy in that window.

    nice -n 19 $PY -m experiments.wean_pilot <arm> [seed] [nurse] [total] [out]
"""
from __future__ import annotations

import functools
import json
import sys
import time

import numpy as np

ARMS = ("null", "nursed", "decoy")
R0, R_MIN, R_MAX, R_STEP = 0.6, 0.5, 6.0, 1.5
FLOOR_Z = 0.5          # a food geom above this is not floor food (the apple)
DECOY_CLEAR = 2.0      # m: a decoy point is at least this far from any food


def main(arm: str, seed: int = 90, nurse: int = 9000, total: int = 15000,
         out_path: str = "") -> None:
    import torch

    from . import cores
    from .policy_loss import lc_update_with
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    from .w0 import uniform_legal_spawn

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    out_path = out_path or f"/data/tmp/wean/{arm}_s{seed}.jsonl"
    torch.set_num_threads(1)
    cores.lc_update = functools.partial(lc_update_with, policy_loss="awr")
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=True)
    w, dr = L.w, L.w.drives
    floor = {n: gid for n, (gid, _) in dr._food.items()}

    def food_xy(ready_only: bool):
        pts = []
        for n, gid in floor.items():
            p = w.data.geom_xpos[gid]
            if p[2] > FLOOR_Z:
                continue
            if ready_only and dr.t < dr._respawn_at[n]:
                continue
            pts.append((float(p[0]), float(p[1])))
        return np.array(pts).reshape(-1, 2)

    legal = w.legal_spawns()
    f0 = food_xy(False)
    far = legal[np.min(np.linalg.norm(legal[:, None] - f0[None], axis=2),
                       axis=1) >= DECOY_CLEAR]
    drng = np.random.RandomState(seed * 31 + 5)
    decoys = far[drng.choice(len(far), size=max(1, len(f0)), replace=False)]

    st = {"r": R0, "phase": "nurse" if arm != "null" else "uniform",
          "spawn_d": None}

    def sampler(legal_, rng, death_xy):
        if arm == "null" or L.k >= nurse:
            st["phase"] = "uniform"
            return uniform_legal_spawn(legal_, rng, death_xy)
        tg = food_xy(True) if arm == "nursed" else decoys
        if len(tg) == 0:
            tg = food_xy(False) if arm == "nursed" else decoys
        t = tg[int(rng.randint(len(tg)))]
        d = np.linalg.norm(legal_ - t[None], axis=1)
        ok = np.flatnonzero(d <= st["r"])
        k = int(ok[rng.randint(len(ok))]) if len(ok) else int(np.argmin(d))
        return float(legal_[k][0]), float(legal_[k][1])

    w.spawn_sampler = sampler
    if arm != "null":                      # the first body is born staged too
        w.respawn(at=sampler(legal, w._spawn_rng, (0.0, 0.0)))

    def nearest():
        f = food_xy(False)
        p = w.data.xpos[w.rover_bid]
        return float(np.min(np.linalg.norm(f - np.array([p[0], p[1]]), axis=1)))

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.perf_counter()
    life, k_life, ate0 = 0, 0, 0
    born_phase, born_r, born_d = st["phase"], st["r"], nearest()
    dsum, dmin = 0.0, 1e9
    with open(out_path, "w") as out:
        while L.k < total:
            L.step()
            k_life += 1
            died = w.died_this_decision
            if not died:
                d = nearest()
                dsum += d
                dmin = min(dmin, d)
            if died or L.k >= total:
                ate = int(sum(dr.ate_total.values()))
                span = (float(w.life_lengths[-1]) if died
                        else float(w.sim_seconds - w._life_started_at))
                row = {"arm": arm, "life": life, "phase": born_phase,
                       "r": round(born_r, 2), "born_d": round(born_d, 2),
                       "sim_s": round(span, 1), "decisions": k_life,
                       "censored": int(not died), "ate": ate - ate0,
                       "d_mean": round(dsum / max(1, k_life), 2),
                       "d_min": round(dmin, 2), "k": L.k,
                       "cause": w.last_death_cause if died else "",
                       "wall_s": round(time.perf_counter() - t0, 1)}
                out.write(json.dumps(row) + "\n")
                out.flush()
                if born_phase == "nurse":
                    st["r"] = (min(R_MAX, st["r"] * R_STEP) if row["ate"]
                               else max(R_MIN, st["r"] / R_STEP))
                    # the staircase moves AFTER the respawn that _die already
                    # made with the old r: one life of lag, same for decoy
                life, k_life, ate0 = life + 1, 0, ate
                born_phase, born_r, born_d = st["phase"], st["r"], nearest()
                dsum, dmin = 0.0, 1e9


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 90,
         int(a[2]) if len(a) > 2 else 9000,
         int(a[3]) if len(a) > 3 else 15000,
         a[4] if len(a) > 4 else "")
