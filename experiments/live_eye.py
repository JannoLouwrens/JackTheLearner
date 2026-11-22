"""LAB — THE HINDSIGHT EYE LIT FROM EMPTY, AND UNLEARNING A WRONG IMPRINT (no
ledger row; a pilot. Seat: Vision encoder, BY DEFAULT; PLAN parts "Learning
to eat" and "His world" right-hand columns; follow-up to `hindsight_eye.py`,
ALIVE 11-21 with a table CARRIED from 215 meals — ruling 84 entered it as
J17's next candidate and its own docstring says: '"filled as he lives, from
empty" is NOT tested'. This file tests that, and nothing else.)

WHY. A newborn has no table. His first meal comes from babble (about one in
2,200 decisions on the entry rung), and the stage-1 read says only 74% of
imprints are on food. A carried table of 215 meals out-votes the wrong ones.
A table of ONE meal cannot: if his first imprint is a wall, the reflex walks
him to walls, the walls are always in sight, he never babbles again, never
eats again, and the table is never corrected. The sense as entered may lock
a newborn onto the wrong thing one birth in four.

THE BET. (1) From an empty table, in a world he has never seen, the loop
lights itself: after his first meal he eats several times what babble eats.
(2) It needs UNLEARNING to do so: a colour he has walked up to and stood at
without being fed loses its place in the table (extinction — a cue with no
meal). Without it some births lock on.

PRIOR (SYSTEM.md 3a; from memory, not re-read today; the Director's
`RESEARCH:` line of 11-22 for this sense is still open). Acquisition and
extinction of a cue by its outcome is Rescorla & Wagner 1972 — a table, a
handful of trials, the size we are at. Optimistic tabular bandits lock onto
a wrong arm without it (Sutton & Barto ch. 2). No published case of a
from-scratch forager whose only teacher of WHICH thing is food is the look
at its own mouth, at one meal. Our budget reaches the published size (a
table; tens of trials).

WORLD. `taste_eye.make_env` unchanged (world K, entry rung, the 128-ray
body-frame eye, mujoco 3.17.0 in `/data/venvs/jack-std`). NOT a learner
(ruling 58): a hand reflex as a ruler, as in `hindsight_eye` stage 2. The
table is his: it starts EMPTY in every cell, is written only by his own
meals, and is kept across his deaths inside the cell.

ARMS (same world, same seed, same babble stream; all four are the SAME run
until the first meal, so they share its decision index t1 — asserted)
  live        at each meal: table[texture of his nearest ray] += 1. When a
              ray wears a tabled texture, push along the nearest ray wearing
              the most-eaten one (`hindsight_eye.reflex_cell`'s reflex);
              else babble (Held 8). EXTINCTION: each decision he pushes at a
              tabled texture whose ray reads nearer than NEAR and is not fed
              adds 1 to that texture's unfed count; past PATIENCE the
              texture leaves the table. A meal imprinted on a texture zeroes
              its unfed count.
  live-noext  the same with no extinction (the sense exactly as entered).
  yoked       MUST FAIL: the same loop and the same extinction, but at each
              meal the imprint is the nearest ray's texture from one of his
              OWN looks 200-1,000 decisions earlier (the mouth did not name
              it).
  held8       the floor: babble only.

PINS, set on train worlds only (dev: 1101-1106, lived before by
`taste_eye`'s kernel; no judge world lived): NEAR, PATIENCE, AFTER, CAP
below. Dev said, before the judge (after = 1,500): PATIENCE 50 forgot real
food (live 2/1/1/0 vs no-extinction 5/5/5/5 on 1102/1104/1101/1103);
PATIENCE 400: live 5/5/7/5, yoked 0/5/0/0 on 1102/1103/1105/1106. All 6 dev
first imprints were on food, so bet (2) may well read NOT NEEDED. The judge worlds 1401..1408 have never been lived by anything.

RULE — DECLARED BEFORE ANY JUDGE WORLD IS LIVED.
  A world IGNITES if held8's first meal comes by decision CAP - AFTER. The
  measure, per ignited world and arm: MEALS IN THE `AFTER` DECISIONS THAT
  FOLLOW THE FIRST MEAL (the first meal itself not counted).
  VOID if fewer than 6 of the 8 worlds ignite, or an arm's first meal is
  not at held8's t1 (the arms were not the same run).
  Bet (1) is ALIVE iff `live` eats more than held8 AND more than yoked on
  at least three quarters of the ignited worlds (6 of 8, 6 of 7, 5 of 6),
  and at least 2x each in total. Else DEAD.
  Bet (2), read only if (1) is ALIVE: UNLEARNING NEEDED iff live-noext has
  at least one ignited world with ZERO meals after the first where `live`
  has one or more, and `live` >= live-noext in total; NOT NEEDED iff
  live-noext's total is within 10% of `live`'s or above it; else UNCLEAR.
  No extension; a miss is a miss.
  Reported beside, gating nothing: t1 per world (his world's own number —
  how long babble takes to find a first meal); imprints on food (simulator
  truth, a diagnostic); extinctions, and how many struck a true food
  texture; deaths; the carried table's 10 meals per 2,400 decisions
  (`hindsight_eye` stage 2) as the yardstick.
  Cannot say, whatever happens: that a LEARNER uses the sense (J17's
  question); anything about an eye whose colours are not W0's one clean
  number per object; that the table should be carried between worlds.

    python -m experiments.live_eye dev            # train worlds only
    python -m experiments.live_eye judge [workers] # the one judged read
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import taste_eye as te  # noqa: E402

N = te.N
OUT = Path("/data/jack-logs/live_eye")
ARMS = ("held8", "live", "live-noext", "yoked")
JUDGE_SEEDS = tuple(range(1401, 1409))
DEV_SEEDS = (1101, 1102, 1103, 1104, 1105, 1106)
NEAR = 0.13        # normalised ray distance: 95th pct of the nearest ray at a meal
PATIENCE = 400     # unfed decisions near a tabled texture (80 s) before it is dropped;
#                    dev 11-22 on worlds 1101-1104: at 50 (10 s) he forgot REAL food, 8 of 8
#                    extinctions — an eaten food stays in sight while it regrows (50 s =
#                    250 decisions), so patience must outlast the regrow clock.
AFTER = 2500       # decisions measured after the first meal (500 s)
CAP = 9000         # a cell never runs longer (first meal by 6,500 or not ignited)


def cell(arm: str, seed: int, after: int = AFTER, cap: int = CAP) -> dict:
    from experiments.world_card import Held
    ext = arm in ("live", "yoked")
    ft = te.food_tex(seed)
    ft = ft[~np.isnan(ft)]

    def is_food(v) -> bool:             # simulator truth: a diagnostic only
        return bool((np.abs(ft - v) < 0.006).any())
    w = te.make_env(seed)
    pol = Held(seed, 8)
    rng = np.random.RandomState(seed * 7919 + 5)
    o, info = w.reset(seed=seed)
    pol.new_life()
    table: dict[float, int] = {}
    unfed: dict[float, int] = {}
    hist: list[float] = []              # his nearest ray's texture, every look
    last, t1, meals_after, deaths = int(info["meals"]), None, 0, 0
    steered = imprints = on_food = ext_n = ext_food = 0
    t0, i = time.process_time(), 0
    while i < cap and (t1 is None or i <= t1 + after):
        dist = o[0:2 * N:2].astype(np.float32)
        tex = o[1:2 * N:2].astype(np.float32)
        hist.append(float(tex[dist.argmin()]))
        a = pol.act(o)                  # always drawn: one babble stream for every arm
        target = None
        if arm != "held8" and table:
            sc = np.zeros(N, np.float32)
            for v, c_ in table.items():
                sc = np.where(np.abs(tex - v) < 0.006, float(c_), sc)
            if sc.max() > 0:
                k = int((sc - dist * 0.5).argmax())
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
                steered += 1
                target = min(table, key=lambda v: abs(float(tex[k]) - v))
                near = bool(dist[k] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            if t1 is None:
                t1 = i
            else:
                meals_after += 1
            if arm != "held8":
                if arm == "yoked":
                    back = int(rng.randint(200, 1001))
                    c = hist[max(0, len(hist) - 1 - back)]
                else:
                    c = hist[-1]
                v = round(c, 2)
                table[v] = table.get(v, 0) + 1
                unfed[v] = 0
                imprints += 1
                on_food += is_food(v)
        elif ext and target is not None and near:
            unfed[target] = unfed.get(target, 0) + 1
            if unfed[target] > PATIENCE:
                del table[target]
                unfed[target] = 0
                ext_n += 1
                ext_food += is_food(target)
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            deaths += 1
            last = int(info["meals"])
        i += 1
    return {"arm": arm, "seed": seed, "t1": t1, "meals_after": meals_after,
            "decisions": i, "deaths": deaths, "steered": steered,
            "imprints": imprints, "imprints_on_food": on_food,
            "extinctions": ext_n, "extinctions_of_food": ext_food,
            "table": {str(k): v for k, v in sorted(table.items())},
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(args):
    r = cell(*args)
    print(json.dumps(r), file=sys.stderr, flush=True)
    return r


def judge_rows(rows, seeds, after=AFTER, cap=CAP) -> dict:
    R = {(r["arm"], r["seed"]): r for r in rows}
    ign = [s for s in seeds if R[("held8", s)]["t1"] is not None
           and R[("held8", s)]["t1"] <= cap - after]
    same = all(R[(a, s)]["t1"] == R[("held8", s)]["t1"] for a in ARMS for s in seeds)
    M = {a: [R[(a, s)]["meals_after"] for s in ign] for a in ARMS}
    T = {a: sum(v) for a, v in M.items()}
    void = []
    if len(ign) < 6:
        void.append(f"{len(ign)} of {len(seeds)} worlds ignite (< 6)")
    if not same:
        void.append("an arm's first meal is not at held8's t1")
    need = math.ceil(0.75 * len(ign)) if ign else 0
    cl = {}
    for o_ in ("held8", "yoked"):
        cl[f"worlds over {o_} >= {need}"] = sum(a > b for a, b in zip(M["live"], M[o_])) >= need
        cl[f"2x {o_}"] = T["live"] >= 2 * T[o_] and T["live"] > 0
    b1 = "VOID" if void else "ALIVE" if all(cl.values()) else "DEAD"
    locked = sum(n == 0 and l >= 1 for n, l in zip(M["live-noext"], M["live"]))
    if b1 != "ALIVE":
        b2 = "not read"
    elif locked >= 1 and T["live"] >= T["live-noext"]:
        b2 = "UNLEARNING NEEDED"
    elif T["live-noext"] >= 0.9 * T["live"]:
        b2 = "NOT NEEDED"
    else:
        b2 = "UNCLEAR"
    diag = {a: {k: sum(R[(a, s)][k] for s in seeds) for k in
                ("imprints", "imprints_on_food", "extinctions", "extinctions_of_food",
                 "deaths", "steered")} for a in ARMS}
    return {"bet1": b1, "bet2": b2, "void": void, "clauses": cl, "ignited": ign,
            "t1": {str(s): R[("held8", s)]["t1"] for s in seeds},
            "meals_after": M, "totals": T, "noext_locked_worlds": locked,
            "diag": diag, "pins": {"NEAR": NEAR, "PATIENCE": PATIENCE,
                                   "AFTER": after, "CAP": cap},
            "cpu_s": round(sum(r["cpu_s"] for r in rows), 1)}


def run(seeds, workers=1, after=AFTER, cap=CAP) -> dict:
    jobs = [(a, s, after, cap) for s in seeds for a in ARMS]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as p:
            rows = p.map(_cell, jobs, chunksize=1)
    else:
        rows = [_cell(j) for j in jobs]
    r = judge_rows(rows, seeds, after, cap)
    r["rows"] = rows
    return r


def main(argv):
    cmd = argv[0] if argv else "dev"
    workers = int(argv[1]) if len(argv) > 1 else 1
    if cmd == "dev":
        seeds = tuple(int(s) for s in argv[2:]) or DEV_SEEDS[:2]
        r = run(seeds, workers, after=1500, cap=6000)
    elif cmd == "judge":
        r = run(JUDGE_SEEDS, workers)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "judge.json").write_text(json.dumps(r, indent=1))
    else:
        raise SystemExit(__doc__)
    r.pop("rows", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
