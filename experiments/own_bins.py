"""LAB — THE CLASS IS HIS OWN: closeness bins whose "food" class is named by his
mouth (no ledger row; a pilot, no learner. Seat: Vision encoder, BY DEFAULT;
PLAN parts "Learning to eat" + "Eye" right-hand columns; builds on order J24,
which STOPPED on 11-25 07:10 because the ray row carries no class channel.)

WHY. The record feeds a ray sense to a learner as ~16 closeness bins PER
OBJECT CLASS (PRIORS §27: Safety Gym, FoodCollector) — and in every published
case the simulator hands the class to the lidar. His rays return a per-object
id, not a class (`j24_class_probe.py`), so J24's `eye="bins16"` cannot be built
"from the rays and nothing else". The INBOX line asks for a ray that returns a
material class. This file prices the other answer: the class column is filled
by HIM. The lab's live table (ruling 98: the texture at his mouth when he ate,
empty at birth, unlearned when it does not feed him) says which rays are
"food"; the record's 16-bin closeness form is computed from the rays and that
table and nothing else. Nothing is handed, and the learner would get the
published input shape.

THE BET. From an empty table, a forager that sees ONLY the 16 own-class food
bins (22.5 degrees each — the learner's future input, not the 128 rays) eats
several times what babble and a yoked table eat, and at least half of what
the same forager eats on bins whose class the simulator hands it.

PRIOR (SYSTEM.md 3a; PRIORS §26, §27 read today). Bins per class under a
replay learner: Safety Gym / OmniSafe SAC 27.0 at 3e6 steps; class always
handed. A self-taught object channel fed to a learner: Transporter (frozen
before the learner starts; Atari 100k). A sense built online from outcome and
read as input: "the experiment itself is unpublished" (§26). This pilot is a
table and a reflex — the published size (Rescorla-Wagner: tens of trials) is
ours. It cannot say a learner uses the bins; that is J24 (3)-(6) once ruled.

WORLD. `taste_eye.make_env` unchanged (as `live_eye`: world K, entry rung,
128-ray body-frame eye, mujoco 3.17.0, `/data/venvs/jack-std`).

THE SENSE (`food_bins`). Bin b covers rays 8b-4 .. 8b+3 (bin 0 is dead ahead).
food[b] = max over the bin's rays wearing an ACTIVE tabled texture of
(1 - distance/range); 0 when none. The ruler: push at the centre of the
fullest food bin; no lit bin -> babble (Held 8). The ruler never sees a ray.

ARMS (same world, seed and babble stream; identical until the first meal)
  held8     babble only — the floor.
  own       table from EMPTY, imprint = nearest ray's texture at each meal,
            deletion after PATIENCE unfed decisions near it (ruling 98's
            table, `live_eye` pins unchanged).
  own-sup   PRIORS §26 (3), the lab's standing task (ruling 107): suppression
            in place of deletion — past PATIENCE the texture is MUTED for
            MUTE decisions (doubling each time it is muted again) and keeps
            its count; a meal on it un-mutes and resets the doubling.
  yoked     MUST FAIL: `own`, but each imprint is the nearest ray's texture
            from one of his own looks 200-1,000 decisions earlier.
  oracle    the INBOX proposal as a ceiling: from the first meal on, the
            food class is the simulator's (every floor food's texture).
  live128   `live_eye.cell("live")` — the same table steering on all 128
            rays; prices what pooling to 16 bins costs. Gates nothing.

RULE — DECLARED BEFORE ANY JUDGE WORLD IS LIVED (worlds 1601-1608, never
lived; dev = 1101, 1102 only to see the code run; no pin is tuned here —
NEAR, PATIENCE, AFTER, CAP are `live_eye`'s; MUTE = 1,000 is set blind).
  Measure, per ignited world and arm: meals in the AFTER decisions after the
  first meal. A world ignites as in `live_eye`.
  VOID if fewer than 6 of 8 ignite; or an arm's first meal is not at held8's;
  or oracle is not >= 2x held8 in total (then the bin ruler is broken — our
  code, not the task).
  ALIVE iff `own` eats more than held8 AND more than yoked on at least three
  quarters of the ignited worlds, >= 2x each in total, AND own's total is
  >= half of oracle's. Else DEAD. No extension; a miss is a miss.
  Suppression, read only if ALIVE: BETTER iff own-sup >= 1.1x own in total;
  WORSE iff <= 0.9x; else SAME.
  Reported beside, gating nothing: own / live128 (the price of 16 bins);
  own / oracle; mutes, deletions and how many struck true food.

RESULT (2026-11-25, box, jack-std, 2,365 CPU-s, rule frozen at `4628b97a`) —
**DEAD BY ITS RULE: ahead of the yoked twin on 5 of 7 ignited worlds, bar 6.**
Meals in the 2,500 decisions after the first meal, worlds 1601-1604, 1606-1608
(1605 never ignited): own 65 (10/10/18/9/9/0/9) = oracle 65 (10/10/9/9/9/9/9);
live128 55; yoked 17 (4/0/1/1/1/0/10); babble 10. Every other clause held
(over babble 6/7, 2x babble, 2x yoked, >= half of oracle; not VOID).
 The two worlds that decided it: 1607 — his FIRST imprint was not food (first
 meal at decision 5,523), it was unlearned after 400 decisions, and babble did
 not find a second meal in the time left: own 0, live128 0, oracle 9. 1608 —
 the yoked twin's stale looks landed on food 7 times of 11 and it ate 10 to
 own's 9 (the disclosed weakness of `live_eye`'s twin, again).
 What the numbers do say, gating nothing: pooling 128 rays to 16 bins cost
 nothing (65 vs 55); where his first imprint was food (6 of 7) his own class
 ate as much as a handed class or more; the whole loss against a handed
 class is the one wrong first imprint. Suppression: not read (rule), and it
 changed no meal count (65 = 65; 2 mutes, both of non-food).
 Not explained: own ate 18 in world 1603 where oracle ate 9 (one tabled
 texture, 19 imprints on food) — above the one-food regrow ceiling; not
 examined. Without that world's surplus own would read 56 to oracle's 65.
 Cannot say: that a learner uses these bins; anything at 500k.

    python -m experiments.own_bins dev [workers]
    python -m experiments.own_bins judge [workers]
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

from experiments import live_eye as le  # noqa: E402
from experiments import taste_eye as te  # noqa: E402

N = te.N
NB = 16
PER = N // NB
OUT = Path("/data/jack-logs/own_bins")
ARMS = ("held8", "own", "own-sup", "yoked", "oracle", "live128")
JUDGE_SEEDS = tuple(range(1601, 1609))
DEV_SEEDS = (1101, 1102)
NEAR, PATIENCE, AFTER, CAP = le.NEAR, le.PATIENCE, le.AFTER, le.CAP
MUTE = 1000
TOL = 0.006


def food_bins(dist: np.ndarray, tex: np.ndarray, active) -> tuple[np.ndarray, np.ndarray]:
    """(16,) closeness of the nearest ray per bin wearing an active tabled
    texture (0 = none), and that ray's index (-1 = none). Rays and table only."""
    hit = np.zeros(N, bool)
    for v in active:
        hit |= np.abs(tex - v) < TOL
    close = np.where(hit, 1.0 - np.minimum(dist, 1.0), 0.0).astype(np.float32)
    idx = (np.arange(N) - PER // 2) % N            # bin 0 = rays -4 .. 3
    c = close[idx].reshape(NB, PER)
    j = c.argmax(1)
    fb = c[np.arange(NB), j]
    ray = np.where(fb > 0, idx.reshape(NB, PER)[np.arange(NB), j], -1)
    return fb, ray


def cell(arm: str, seed: int, after: int = AFTER, cap: int = CAP) -> dict:
    if arm == "live128":
        r = le.cell("live", seed, after, cap)
        r["arm"] = arm
        return r
    from experiments.world_card import Held
    ft = te.food_tex(seed)
    ft = [float(v) for v in ft[~np.isnan(ft)]]

    def is_food(v) -> bool:             # simulator truth: a diagnostic only
        return any(abs(f - v) < TOL for f in ft)
    w = te.make_env(seed)
    pol = Held(seed, 8)
    rng = np.random.RandomState(seed * 7919 + 5)
    o, info = w.reset(seed=seed)
    pol.new_life()
    table: dict[float, int] = {}
    unfed: dict[float, int] = {}
    muted_until: dict[float, int] = {}
    mute_len: dict[float, int] = {}
    hist: list[float] = []
    last, t1, meals_after, deaths = int(info["meals"]), None, 0, 0
    steered = imprints = on_food = ext_n = ext_food = 0
    t0, i = time.process_time(), 0
    while i < cap and (t1 is None or i <= t1 + after):
        dist = o[0:2 * N:2].astype(np.float32)
        tex = o[1:2 * N:2].astype(np.float32)
        hist.append(float(tex[dist.argmin()]))
        a = pol.act(o)                  # always drawn: one babble stream for every arm
        target, near = None, False
        if arm == "oracle":
            active = ft if t1 is not None else []
        elif arm == "held8":
            active = []
        else:
            active = [v for v in table if muted_until.get(v, -1) <= i]
        if active:
            fb, ray = food_bins(dist, tex, active)
            if fb.max() > 0:
                b = int(fb.argmax())
                ang = 2 * math.pi * (PER * b - 0.5) / N       # the bin's centre
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(ang), math.sin(ang)
                steered += 1
                if arm != "oracle":
                    target = min(active, key=lambda v: abs(float(tex[ray[b]]) - v))
                    near = bool(1.0 - fb[b] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            if t1 is None:
                t1 = i
            else:
                meals_after += 1
            if arm in ("own", "own-sup", "yoked"):
                if arm == "yoked":
                    back = int(rng.randint(200, 1001))
                    c = hist[max(0, len(hist) - 1 - back)]
                else:
                    c = hist[-1]
                v = round(c, 2)
                table[v] = table.get(v, 0) + 1
                unfed[v] = 0
                muted_until.pop(v, None)
                mute_len.pop(v, None)
                imprints += 1
                on_food += is_food(v)
        elif target is not None and near:
            unfed[target] = unfed.get(target, 0) + 1
            if unfed[target] > PATIENCE:
                unfed[target] = 0
                ext_n += 1
                ext_food += is_food(target)
                if arm == "own-sup":
                    mute_len[target] = 2 * mute_len[target] if target in mute_len else MUTE
                    muted_until[target] = i + mute_len[target]
                else:
                    del table[target]
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
    if not (T["oracle"] >= 2 * T["held8"] and T["oracle"] > 0):
        void.append("oracle bins not >= 2x babble: the bin ruler is broken")
    need = math.ceil(0.75 * len(ign)) if ign else 0
    cl = {}
    for o_ in ("held8", "yoked"):
        cl[f"worlds over {o_} >= {need}"] = sum(a > b for a, b in zip(M["own"], M[o_])) >= need
        cl[f"2x {o_}"] = T["own"] >= 2 * T[o_] and T["own"] > 0
    cl["own >= half of oracle"] = 2 * T["own"] >= T["oracle"]
    bet = "VOID" if void else "ALIVE" if all(cl.values()) else "DEAD"
    if bet != "ALIVE":
        sup = "not read"
    elif T["own-sup"] >= 1.1 * T["own"]:
        sup = "BETTER"
    elif T["own-sup"] <= 0.9 * T["own"]:
        sup = "WORSE"
    else:
        sup = "SAME"
    diag = {a: {k: sum(R[(a, s)][k] for s in seeds) for k in
                ("imprints", "imprints_on_food", "extinctions", "extinctions_of_food",
                 "deaths", "steered")} for a in ARMS}
    return {"bet": bet, "suppression": sup, "void": void, "clauses": cl, "ignited": ign,
            "t1": {str(s): R[("held8", s)]["t1"] for s in seeds},
            "meals_after": M, "totals": T, "diag": diag,
            "pins": {"NEAR": NEAR, "PATIENCE": PATIENCE, "MUTE": MUTE,
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
        r = run(DEV_SEEDS, workers, after=1500, cap=6000)
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
