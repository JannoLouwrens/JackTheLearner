"""LAB pilot CR.00: the curriculum cook-off's pilot (ORDERS J5 step 3; no ledger row).

Design: docs/research/CURRICULUM.md section 4. This is the pilot only: ONE seed
(not a claim seed), three arms. It exists to set N (exam lives) and the margin
from the null's own life-to-life spread BEFORE CR.01's claim seeds are read.

The arms stage only the WORLD, never his actions: an adult of his world (a
spawn rule) chooses where each new body is born during TRAINING.
  null      uniform legal spawn throughout (today's W0)
  staged    STAGES radii in order (reverse-curriculum shape): born within r of
            a ready floor food, r widening block by block; the last block is
            uniform. Each block is an equal share of the training decisions.
  shuffled  CONTROL THAT MUST FAIL against staged: the SAME multiset of blocks
            in a seeded random order. If it matches staged, ordering carries
            nothing and the "curriculum" is only extra easy lives.
(The matched-meal null and the LP/frontier/teacher arms belong to CR.01.)

Then every arm lives an identical EXAM: EXAM decisions of uniform W0 lives,
still learning. Only exam lives are scored, by FED-LIFE FRACTION (lives with
>= 1 meal / exam lives) and MEDIAN SPAN (sim-s, censored last life excluded),
NOT meal counts: the 11-13 J1 reading put 73 % of meals in 5 jackpot lives.

    nice -n 19 $PY -m experiments.cr00_pilot <arm> [seed] [train] [exam] [out]
    $PY -m experiments.cr00_pilot tally [dir]
"""
from __future__ import annotations

import functools
import glob
import json
import os
import sys
import time

import numpy as np

ARMS = ("null", "staged", "shuffled")
STAGES = (0.6, 1.2, 2.4, 4.0, None)   # None = uniform; declared before any run
FLOOR_Z = 0.5                         # a food geom above this is not floor food


def block_order(arm: str, seed: int) -> list:
    if arm == "null":
        return [None] * len(STAGES)
    if arm == "staged":
        return list(STAGES)
    perm = np.random.RandomState(seed * 7919 + 13).permutation(len(STAGES))
    return [STAGES[i] for i in perm]


def main(arm: str, seed: int = 89, train: int = 40000, exam: int = 20000,
         out_path: str = "", smell: bool = False) -> None:
    import torch

    from . import cores
    from .policy_loss import lc_update_with
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    from .w0 import uniform_legal_spawn

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    if smell:
        from . import smell_sense
        smell_sense.install("smell", seed)
    out_path = out_path or f"/data/tmp/cr00/{arm}_s{seed}.jsonl"
    torch.set_num_threads(1)
    cores.lc_update = functools.partial(lc_update_with, policy_loss="awr")
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=True)
    w, dr = L.w, L.w.drives
    floor = {n: gid for n, (gid, _) in dr._food.items()}
    order = block_order(arm, seed)
    block = max(1, train // len(order))

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

    def stage_now():
        if L.k >= train:
            return "exam", None
        return "train", order[min(len(order) - 1, L.k // block)]

    def sampler(legal_, rng, death_xy):
        _, r = stage_now()
        if r is None:
            return uniform_legal_spawn(legal_, rng, death_xy)
        tg = food_xy(True)
        if len(tg) == 0:
            tg = food_xy(False)
        t = tg[int(rng.randint(len(tg)))]
        d = np.linalg.norm(legal_ - t[None], axis=1)
        ok = np.flatnonzero(d <= r)
        k = int(ok[rng.randint(len(ok))]) if len(ok) else int(np.argmin(d))
        return float(legal_[k][0]), float(legal_[k][1])

    w.spawn_sampler = sampler
    if order[0] is not None:               # the first body is born staged too
        w.respawn(at=sampler(w.legal_spawns(), w._spawn_rng, (0.0, 0.0)))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.perf_counter()
    life, k_life, ate0 = 0, 0, 0
    born = stage_now()
    with open(out_path, "w") as out:
        while L.k < train + exam:
            L.step()
            k_life += 1
            died = w.died_this_decision
            if died or L.k >= train + exam:
                ate = int(sum(dr.ate_total.values()))
                span = (float(w.life_lengths[-1]) if died
                        else float(w.sim_seconds - w._life_started_at))
                row = {"arm": arm, "seed": seed, "life": life,
                       "phase": born[0], "r": born[1], "sim_s": round(span, 1),
                       "decisions": k_life, "censored": int(not died),
                       "ate": ate - ate0, "k": L.k,
                       "cause": w.last_death_cause if died else "",
                       "wall_s": round(time.perf_counter() - t0, 1)}
                out.write(json.dumps(row) + "\n")
                out.flush()
                life, k_life, ate0 = life + 1, 0, ate
                born = stage_now()


def tally(d: str = "/data/tmp/cr00") -> None:
    """Exam lives only = lives BORN after the training decisions."""
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [json.loads(x) for x in open(f) if x.strip()]
        ex = [r for r in rows if r["phase"] == "exam"]
        done = [r for r in ex if not r["censored"]]
        fed = sum(1 for r in ex if r["ate"] > 0)
        med = float(np.median([r["sim_s"] for r in done])) if done else float("nan")
        spans = [r["sim_s"] for r in done]
        print(f"{os.path.basename(f):24s} train_lives {len(rows) - len(ex):4d} "
              f"exam_lives {len(ex):4d} fed {fed}/{len(ex)} "
              f"({fed / max(1, len(ex)):.3f}) median_span {med:.0f} "
              f"span_sd {np.std(spans) if spans else float('nan'):.0f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "tally":
        tally(*(a[1:2]))
    else:
        main(a[0], int(a[1]) if len(a) > 1 else 89,
             int(a[2]) if len(a) > 2 else 40000,
             int(a[3]) if len(a) > 3 else 20000,
             a[4] if len(a) > 4 else "",
             smell=os.environ.get("CR00_SMELL") == "1")
