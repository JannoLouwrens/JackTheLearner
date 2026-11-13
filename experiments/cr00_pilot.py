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
CR.01's machine arms (added 11-13, docs/research/CURRICULUM.md section 4), all
over the SAME five birth bins as staged (STAGES), chosen per BIRTH, never per
action; a bin's outcome is whether the life born in it ate at least once:
  lp        self-chosen by absolute learning progress (IAC/SAGG-RIAC shape): each
            bin keeps its fed outcomes; ALP = |mean(last W) - mean(W before)|;
            a bin with < 2W outcomes is sampled first (lowest count), then
            eps-greedy on ALP (EPS), ties to the wider bin. Birth-radius bins
            only: W0 has no food-density knob, so the density axis is absent.
  frontier  GoalGAN-style band: one radius index, starts at the nearest bin;
            after every W lives born at it, fed rate > BAND_HI -> widen one bin,
            < BAND_LO -> narrow one bin, else stay.
  matched   matched-meal null (CONTROL): each birth draws a bin uniformly from
            STAGES - the staged arm's expected easy-birth share, with no
            order and no adaptation. Separates "a curriculum" from "more
            easy lives". (The teacher arm waits on the Judge admission gate.)

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

ARMS = ("null", "staged", "shuffled", "lp", "frontier", "matched")
ADAPTIVE = ("lp", "frontier", "matched")   # bin chosen per birth
STAGES = (0.6, 1.2, 2.4, 4.0, None)   # None = uniform; declared before any run
FLOOR_Z = 0.5                         # a food geom above this is not floor food
W, EPS = 5, 0.2                       # LP window (lives) and exploration; declared 11-13
BAND_LO, BAND_HI = 0.2, 0.8           # frontier band (GoalGAN's ~0.1-0.9, CURRICULUM.md)


class BinChooser:
    """Per-birth bin choice for the adaptive arms. Pure: no world access."""

    def __init__(self, arm: str, seed: int):
        self.arm, self.n = arm, len(STAGES)
        self.rng = np.random.RandomState(seed * 6007 + 29)
        self.hist = [[] for _ in range(self.n)]
        self.fi, self.fwin = 0, []                # frontier index + its window

    def alp(self, i: int) -> float:
        h = self.hist[i]
        if len(h) < 2 * W:
            return float("nan")
        return abs(float(np.mean(h[-W:])) - float(np.mean(h[-2 * W:-W])))

    def choose(self) -> int:
        if self.arm == "matched":
            return int(self.rng.randint(self.n))
        if self.arm == "frontier":
            return self.fi
        young = [i for i in range(self.n) if len(self.hist[i]) < 2 * W]
        if young:
            m = min(len(self.hist[i]) for i in young)
            return max(i for i in young if len(self.hist[i]) == m)
        if self.rng.rand() < EPS:
            return int(self.rng.randint(self.n))
        a = [self.alp(i) for i in range(self.n)]
        best = max(a)
        return max(i for i in range(self.n) if a[i] == best)

    def credit(self, i: int, fed: bool) -> None:
        self.hist[i].append(int(fed))
        if self.arm == "frontier" and i == self.fi:
            self.fwin.append(int(fed))
            if len(self.fwin) >= W:
                rate = float(np.mean(self.fwin))
                if rate > BAND_HI:
                    self.fi = min(self.n - 1, self.fi + 1)
                elif rate < BAND_LO:
                    self.fi = max(0, self.fi - 1)
                self.fwin = []


def block_order(arm: str, seed: int) -> list:
    if arm in ADAPTIVE:
        return [None] * len(STAGES)          # placeholder; chosen per birth
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
    chooser = BinChooser(arm, seed) if arm in ADAPTIVE else None
    cur = {"bin": None, "ate0": 0}          # the living body's birth bin

    def ate_now() -> int:
        return int(sum(dr.ate_total.values()))

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
        if chooser is not None:
            return "train", (STAGES[cur["bin"]] if cur["bin"] is not None
                             else None)
        return "train", order[min(len(order) - 1, L.k // block)]

    def pick_bin():
        """Credit the dying body's bin, then choose the newborn's."""
        if chooser is None:
            return
        if cur["bin"] is not None:
            chooser.credit(cur["bin"], ate_now() - cur["ate0"] > 0)
        cur["bin"] = chooser.choose() if L.k < train else None
        cur["ate0"] = ate_now()

    def sampler(legal_, rng, death_xy):
        pick_bin()
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
    if chooser is not None:                # the first body is born by choice too
        w.respawn(at=sampler(w.legal_spawns(), w._spawn_rng, (0.0, 0.0)))
    elif order[0] is not None:             # the first body is born staged too
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
