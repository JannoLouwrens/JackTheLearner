"""LAB — A FIRST IMPRINT THAT CANNOT BE WRONG: his mouth names everything
within reach, and going unfed crosses names off (no ledger row; a pilot, no
learner. Seat: Vision encoder, BY DEFAULT; PLAN parts "Learning to eat" +
"Eye" right-hand columns. Repairs the one measured loss of the lab's ENTERED
hindsight eye (rulings 84, 98, 107) before its registered race.)

WHY. `own_bins` (11-25) died on ONE world: at his first meal the NEAREST ray
was a wall beside the food (1607: wall .072, food .081), the one-name table
took the wall, unlearned it 400 decisions later, and babble found no second
meal in time — own 0, handed class 9. Probe of 12 already-lived worlds
(`/data/jack-logs/first_imprint/probe.py`, 1601-1608 + 1401-1405): at every
babbled meal the food he ate is among the 2-3 nearest textures of the look
before the bite or the one before that (0.014-0.176 of range) — it is just
not always THE nearest. The label was right; the pointer was too narrow.

THE BET. Imprint the SET — every texture within SET_R in his last two looks
before a meal — and let the table's existing unlearning (unfed PATIENCE
decisions standing at a texture -> deleted) cross off the names that do not
feed him. Then no first imprint is wrong: it eats as much as the one-name
table where that was right, and eats where the one-name table locked out.

PRIOR (SYSTEM.md 3a; PRIORS §26, §28 as read 11-25, the rest from memory).
Candidate elimination over a version space (Mitchell 1982): keep every
hypothesis consistent with the positives, strike those a negative refutes —
exact for a conjunctive concept, tens of examples. Cue competition in
conditioning (Rescorla-Wagner 1972; overshadowing / blocking): every cue
present at the outcome is credited, extinction sorts them — tens of trials,
a table: our size. §26: a sense built online from outcome is unpublished;
§28: a look whose meaning is re-drawn per world has only been learned by
agents with memory — this table is that memory, and it is his.

WORLD. `taste_eye.make_env` unchanged (world K, entry rung d0.75, 128-ray
body-frame eye, mujoco 3.17.0, `/data/venvs/jack-std`). Ruler and sense:
`own_bins.food_bins` and its fullest-bin push, unchanged.

ARMS (same world, seed and babble stream; identical until the first meal)
  held8      babble only — the floor (`own_bins.cell`).
  own        the holder: one name per meal = the nearest ray's texture
             (`own_bins.cell("own")`, unchanged).
  set        the bet: at a meal every texture (2 dp) whose nearest ray is
             within SET_R in the look before the bite or the one before it
             is counted and its unfed clock zeroed; none within SET_R ->
             the nearest ray's texture (own's rule). Deletion as `own`.
  yoked-set  MUST FAIL: `set`, but each meal's set is read from his own
             pair of looks 200-1,000 decisions earlier.
  oracle     ceiling: the simulator's food class from the first meal on.

PINS. NEAR, PATIENCE, AFTER, CAP are `live_eye`'s, untouched. SET_R = 0.20,
set from the probe above (largest food distance in the two looks: .176) and
tuned on nothing else. DEV = lived worlds 1607, 1608, 1601 only (does the
code run; does 1607 — the known positive — now eat).

RULE — DECLARED BEFORE ANY JUDGE WORLD IS LIVED (worlds 1701-1712, never
lived by anything). Measure, per ignited world and arm: meals in the AFTER
decisions after the first meal; a world ignites as in `live_eye`.
  VOID if fewer than 9 of 12 ignite; or an arm's first meal is not at
  held8's; or oracle is not >= 2x held8 in total.
  ALIVE iff ALL of:
   (a) `set` eats more than held8 AND more than yoked-set on at least three
       quarters of the ignited worlds, and >= 2x each in total;
   (b) no price for the insurance: set total >= 0.9 x own total;
   (c) the repair itself: over the ignited worlds where the nearest ray at
       the first meal was NOT food (simulator truth, judge only), set eats
       >= half of oracle's total there. If the 12 worlds draw no such world,
       (c) is read on the known positive, lived world 1607, and the verdict
       line says so.
  Else DEAD. No extension; a miss is a miss.
  Reported beside, gating nothing: set / oracle; size of the first set and
  whether it held the food; deletions and how many struck true food; worlds
  where own ate <= 1.

RESULT (2026-11-25, box, jack-std, 2,895 CPU-s, rule frozen at `aa9ea10d`) —
**DEAD BY ITS RULE: clause (c). In the one judged world whose nearest ray at
the first meal was not food (1708) set ate 6, the handed class 19 — bar 9.5.**
Worlds 1701-1712, 10 ignited (1710, 1711 never ate). Meals in the 2,500
decisions after the first meal: set 74 (2/6/9/9/9/9/9/6/9/6), own 79
(9/10/9/9/10/9/9/0/9/5), oracle 102, yoked-set 12, babble 19. Clauses (a)
and (b) held: over babble 8/10 and over the yoked twin 9/10 (bar 8), 2x
each in total, 74 >= 0.9 x 79.
 What it did: the lock-out is gone — own ate <= 1 in one world (1708: 0),
 set in none (6 there); the first set held the food in 10 of 10 worlds
 (size 1-4, median 2) where the nearest ray alone was food in 9 of 10. Dev,
 lived world 1607: own 0 -> set 9 = oracle 9.
 What it cost: extra names are paid for in waiting. 14 deletions against
 own's 4 (none of true food) at 400 decisions each: 1701 set 2 vs own 9
 (5 deletions), 1702 6 vs 10 (4). And names that sit beside the food are
 never struck (1704: .18 counted 10 of 10 meals) — harmless there, since
 pushing at them feeds him.
 The yoked twin failed as it must this time (12; ahead in no world, level
 in one) — the bigger stale set is not a lucky guess.
 Not explained: oracle 19 in 1708 (two foods in one push?) — the same
 above-the-regrow-ceiling surplus `own_bins` left open in 1603.
 Cannot say: that a learner uses it; anything about a kind-coloured world
 (INBOX 11-25 16:20), where one name per KIND would make a wrong first
 name cost one deletion in a lifetime, not one per world.
 A new reason to return: strike a name faster than 400 decisions when a
 better-fed name is in sight (counts 7 vs 1 already say which is food).

    python -m experiments.first_imprint dev [workers]
    python -m experiments.first_imprint judge [workers]
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

from experiments import own_bins as ob  # noqa: E402
from experiments import taste_eye as te  # noqa: E402

N, PER = ob.N, ob.PER
OUT = Path("/data/jack-logs/first_imprint")
ARMS = ("held8", "own", "set", "yoked-set", "oracle")
JUDGE_SEEDS = tuple(range(1701, 1713))
DEV_SEEDS = (1607, 1608, 1601)
KNOWN_POSITIVE = 1607
NEAR, PATIENCE, AFTER, CAP, TOL = ob.NEAR, ob.PATIENCE, ob.AFTER, ob.CAP, ob.TOL
SET_R = 0.20


def near_set(dist: np.ndarray, tex: np.ndarray) -> frozenset:
    """Textures (2 dp) with a ray inside SET_R. Rays only."""
    return frozenset(round(float(v), 2) for v in tex[dist < SET_R])


def cell(arm: str, seed: int, after: int = AFTER, cap: int = CAP) -> dict:
    if arm in ("held8", "own", "oracle"):
        return ob.cell(arm, seed, after, cap)
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
    sets: list[frozenset] = []
    nearest: list[float] = []
    last, t1, meals_after, deaths = int(info["meals"]), None, 0, 0
    steered = imprints = with_food = ext_n = ext_food = 0
    first = None
    t0, i = time.process_time(), 0
    while i < cap and (t1 is None or i <= t1 + after):
        dist = o[0:2 * N:2].astype(np.float32)
        tex = o[1:2 * N:2].astype(np.float32)
        sets.append(near_set(dist, tex))
        nearest.append(round(float(tex[dist.argmin()]), 2))
        a = pol.act(o)                  # always drawn: one babble stream for every arm
        target, near = None, False
        if table:
            active = list(table)
            fb, ray = ob.food_bins(dist, tex, active)
            if fb.max() > 0:
                b = int(fb.argmax())
                ang = 2 * math.pi * (PER * b - 0.5) / N
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(ang), math.sin(ang)
                steered += 1
                target = min(active, key=lambda v: abs(float(tex[ray[b]]) - v))
                near = bool(1.0 - fb[b] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            k = len(sets) - 1
            if arm == "yoked-set":
                k = max(0, k - int(rng.randint(200, 1001)))
            cands = set(sets[k]) | (set(sets[k - 1]) if k > 0 else set())
            if not cands:
                cands = {nearest[k]}
            if t1 is None:
                t1 = i
                first = {"size": len(cands), "has_food": any(is_food(v) for v in cands),
                         "nearest_is_food": is_food(nearest[len(sets) - 1])}
            else:
                meals_after += 1
            for v in cands:
                table[v] = table.get(v, 0) + 1
                unfed[v] = 0
            imprints += 1
            with_food += any(is_food(v) for v in cands)
        elif target is not None and near:
            unfed[target] = unfed.get(target, 0) + 1
            if unfed[target] > PATIENCE:
                unfed[target] = 0
                ext_n += 1
                ext_food += is_food(target)
                del table[target]
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            deaths += 1
            last = int(info["meals"])
        i += 1
    return {"arm": arm, "seed": seed, "t1": t1, "meals_after": meals_after,
            "decisions": i, "deaths": deaths, "steered": steered,
            "imprints": imprints, "imprints_with_food": with_food,
            "extinctions": ext_n, "extinctions_of_food": ext_food, "first": first,
            "table": {str(k): v for k, v in sorted(table.items())},
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(args):
    r = cell(*args)
    print(json.dumps(r), file=sys.stderr, flush=True)
    return r


def judge_rows(rows, seeds, after=AFTER, cap=CAP, known=None) -> dict:
    """`known` = the `set` and `oracle` rows of the known-positive world, used
    for clause (c) only when no judged world draws a wrong nearest ray."""
    R = {(r["arm"], r["seed"]): r for r in rows}
    ign = [s for s in seeds if R[("held8", s)]["t1"] is not None
           and R[("held8", s)]["t1"] <= cap - after]
    same = all(R[(a, s)]["t1"] == R[("held8", s)]["t1"] for a in ARMS for s in seeds)
    M = {a: [R[(a, s)]["meals_after"] for s in ign] for a in ARMS}
    T = {a: sum(v) for a, v in M.items()}
    void = []
    if len(ign) < math.ceil(0.75 * len(seeds)):
        void.append(f"{len(ign)} of {len(seeds)} worlds ignite")
    if not same:
        void.append("an arm's first meal is not at held8's t1")
    if not (T["oracle"] >= 2 * T["held8"] and T["oracle"] > 0):
        void.append("oracle bins not >= 2x babble: the bin ruler is broken")
    need = math.ceil(0.75 * len(ign)) if ign else 0
    cl = {}
    for o_ in ("held8", "yoked-set"):
        cl[f"(a) worlds over {o_} >= {need}"] = sum(
            a > b for a, b in zip(M["set"], M[o_])) >= need
        cl[f"(a) 2x {o_}"] = T["set"] >= 2 * T[o_] and T["set"] > 0
    cl["(b) set >= 0.9 x own"] = T["set"] >= 0.9 * T["own"]
    wrong = [s for s in ign if not R[("set", s)]["first"]["nearest_is_food"]]
    if wrong:
        sw = sum(R[("set", s)]["meals_after"] for s in wrong)
        ow = sum(R[("oracle", s)]["meals_after"] for s in wrong)
        c_src = f"judge worlds {wrong}"
    elif known is not None:
        sw, ow = known["set"]["meals_after"], known["oracle"]["meals_after"]
        c_src = f"known positive {KNOWN_POSITIVE} (no wrong nearest ray drawn)"
    else:
        sw, ow, c_src = 0, 1, "no wrong nearest ray and no known positive given"
    cl["(c) set >= half of oracle where the nearest ray was wrong"] = 2 * sw >= ow and sw > 0
    bet = "VOID" if void else "ALIVE" if all(cl.values()) else "DEAD"
    diag = {a: {k: sum(R[(a, s)].get(k, 0) for s in seeds) for k in
                ("imprints", "extinctions", "extinctions_of_food", "deaths", "steered")}
            for a in ARMS}
    return {"bet": bet, "void": void, "clauses": cl, "ignited": ign,
            "clause_c": {"source": c_src, "set": sw, "oracle": ow},
            "t1": {str(s): R[("held8", s)]["t1"] for s in seeds},
            "meals_after": M, "totals": T, "diag": diag,
            "first_set": {str(s): R[("set", s)]["first"] for s in seeds},
            "own_locked_out": [s for s in ign if R[("own", s)]["meals_after"] <= 1],
            "set_locked_out": [s for s in ign if R[("set", s)]["meals_after"] <= 1],
            "pins": {"NEAR": NEAR, "PATIENCE": PATIENCE, "SET_R": SET_R,
                     "AFTER": after, "CAP": cap},
            "cpu_s": round(sum(r["cpu_s"] for r in rows), 1)}


def run(seeds, workers=1, after=AFTER, cap=CAP, known=None) -> dict:
    jobs = [(a, s, after, cap) for s in seeds for a in ARMS]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as p:
            rows = p.map(_cell, jobs, chunksize=1)
    else:
        rows = [_cell(j) for j in jobs]
    r = judge_rows(rows, seeds, after, cap, known)
    r["rows"] = rows
    return r


def main(argv):
    cmd = argv[0] if argv else "dev"
    workers = int(argv[1]) if len(argv) > 1 else 1
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "dev":
        r = run(DEV_SEEDS, workers)
        (OUT / "dev.json").write_text(json.dumps(r, indent=1))
    elif cmd == "judge":
        dev = json.loads((OUT / "dev.json").read_text())
        known = {x["arm"]: x for x in dev["rows"] if x["seed"] == KNOWN_POSITIVE}
        r = run(JUDGE_SEEDS, workers, known=known)
        (OUT / "judge.json").write_text(json.dumps(r, indent=1))
    else:
        raise SystemExit(__doc__)
    r.pop("rows", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
