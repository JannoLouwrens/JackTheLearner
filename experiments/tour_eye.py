"""LAB — A SPENT FOOD IS SUPPRESSED, NOT FORGOTTEN: HE LEAVES IT AND TOURS (no
ledger row; a pilot. Seat: Vision encoder, BY DEFAULT; PLAN parts "Learning
to eat" and "Eye" right-hand columns; follow-up to `live_eye.py`, ALIVE
11-22, rulings 84 and 98; answers PRIORS §26 "what the record does NOT
answer", third row, which asks for exactly this: box-minutes, no learner.)

WHY. `live_eye` eats 8-9 meals in 2,500 decisions on 8 of 8 worlds — and that
is a CEILING, not a skill: he reaches one food and camps on it; 9 meals is
its 50 s regrow clock. His world holds three foods, each its own colour.
Three things in the table keep him camping: (a) unlearning is DELETION after
400 unfed decisions, so patience must outlast the regrow clock (at 50 he
forgot real food 8 of 8); (b) nothing tells him a food he just ate is spent;
(c) a food he never ate is never in the table, and only babble finds it.
And the reflex can wedge (2,983 decisions steered at a food never reached).

THE BET. Make the table's unlearning what §26 says biology's is:
  SUPPRESSION, not deletion — a colour pursued unfed is set aside for a
    time that DOUBLES with each unfed bout (400, 800, ...) and a meal on it
    restores it at once. Nothing is ever erased.
  A GIVE-UP FROM HIS OWN MEALS — the shortest gap he has lived between two
    meals on a colour is that colour's regrow wait W. After a meal on it the
    colour is SPENT for 0.8 W (he leaves), and a pursuit of it unfed for
    longer than W is given up (Krebs 1974: giving-up time follows his own
    capture rate). Before W is known: one camp of <= 400 decisions, as now.
    The pursuit clock runs near AND far, so the wedge ends too.
  (arm `mouth` only) AN INNATE BIAS — with nothing wanted in view he walks
    to the nearest colour he has never stood at, instead of babbling; 25
    unfed decisions at it (or 300 chasing it) and it is tried, for good.
    (§26's table: "chicks peck small 3-D things; ours none; costs the
    ~950-decision wait".)
Then he should eat MORE than the camp ceiling: A is spent -> go to B -> C.

PRIOR (SYSTEM.md 3a; PRIORS §26, read today). Extinction suppresses and
does not erase (Bouton); giving-up time falls as capture rate rises (Krebs
1974); one flavour pairing gives 69-71% choice (Ackroff 2009) — animals,
tens of trials: our size. In agents: nothing — §26 "the experiment itself is
unpublished"; its prediction for this row: "suppression ends the patience-50
failure and keeps the 2-of-8 repair". Inhibition of return as a foraging
facilitator is Klein 1988 (from memory, not re-read). No published forager
learns each food's regrow wait from its own meal gaps at this size. Budget:
a table and tens of meals — reached.

WORLD. `taste_eye.make_env` unchanged (world K, entry rung, 128-ray
body-frame eye, mujoco 3.17.0, `/data/venvs/jack-std`). NOT a learner
(ruling 58): the reflex is a ruler; the table (counts, waits, suppression
clocks, the tried set) is his — empty at birth, written only by his own
meals and his own unfed waits, kept across his deaths inside the cell.

ARMS (same world, same seed, same babble stream)
  held8   babble only. Its first meal t1 sets every arm's stop: t1 + AFTER.
  live    the holder: `live_eye.cell("live")` exactly as entered.
  tour    live's imprint and reflex + suppression + the learned wait.
  mouth   tour + the innate bias.
  yoked   MUST FAIL: `mouth`, but each meal's imprint is the nearest ray's
          colour from one of his OWN looks 200-1,000 decisions earlier.

PINS (set on dev worlds 1101-1106 only; see RULE for what dev said).
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
OUT = Path("/data/jack-logs/tour_eye")
ARMS = ("held8", "live", "tour", "tour-yoked", "mouth", "mouth-yoked")
JUDGE_SEEDS = tuple(range(1501, 1509))   # never lived (fed_once died before them)
DEV_SEEDS = le.DEV_SEEDS
NEAR, AFTER, CAP = le.NEAR, le.AFTER, le.CAP
CAMP = le.PATIENCE   # 400: unfed pursuit allowed while a colour's wait is unknown
BASE = 400           # first suppression after a give-up; doubles each unfed bout
SPENT = 0.8          # after a meal the colour is set aside for SPENT x its wait
MOUTH = 25           # unfed decisions standing at a new colour before it is tried
CHASE = 300          # decisions chasing a new colour before it is tried anyway


def cell(arm: str, seed: int, stop: int) -> dict:
    """One arm in one world for `stop` decisions. Meals are counted from birth."""
    from experiments.world_card import Held
    ft = te.food_tex(seed)
    ft = {int(round(float(v) * 100)) for v in ft[~np.isnan(ft)]}   # diagnostic only
    w = te.make_env(seed)
    pol = Held(seed, 8)
    rng = np.random.RandomState(seed * 7919 + 5)
    o, info = w.reset(seed=seed)
    pol.new_life()
    n: dict[int, int] = {}          # meals imprinted per colour
    away: dict[int, int] = {}       # how long it is set aside after a meal (learned)
    unfed: dict[int, int] = {}      # decisions he stood at it unfed since its last meal
    sup: dict[int, int] = {}        # suppressed until this decision
    back: dict[int, int] = {}       # unfed bouts since the last meal on it
    pursuit: dict[int, int] = {}
    m_near: dict[int, int] = {}
    m_chase: dict[int, int] = {}
    hist: list[int] = []
    last, t1, meals, deaths = int(info["meals"]), None, 0, 0
    steered = mouthed = tried = giveups = giveups_food = on_food = 0
    run_unfed = longest_unfed = 0   # the wedge: longest unbroken unfed pursuit
    meal_at: list[int] = []
    t0 = time.process_time()
    for i in range(stop):
        dist = o[0:2 * N:2].astype(np.float32)
        tex = np.rint(o[1:2 * N:2].astype(np.float64) * 100).astype(np.int64)
        hist.append(int(tex[dist.argmin()]))
        a = pol.act(o)              # always drawn: one babble stream for every arm
        mode, target, k, near = None, None, -1, False
        if arm != "held8":
            sc = np.zeros(N, np.float32)
            for v, c_ in n.items():
                if sup.get(v, 0) <= i:
                    sc = np.where(tex == v, float(c_), sc)
            if sc.max() > 0:
                k = int((sc - dist * 0.5).argmax())
                mode, target = "want", int(tex[k])
            elif arm.startswith("mouth"):
                new = np.array([int(t) not in n and sup.get(int(t), 0) <= i for t in tex])
                if new.any():
                    k = int(np.where(new, dist, np.inf).argmin())
                    mode, target = "mouth", int(tex[k])
            if mode:
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
                near = bool(dist[k] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            meals += 1
            meal_at.append(i)
            run_unfed = 0
            if t1 is None:
                t1 = i
            if arm != "held8":
                if arm.endswith("yoked"):
                    v = hist[max(0, len(hist) - 1 - int(rng.randint(200, 1001)))]
                elif mode and near:
                    v = target      # the colour he was reaching for when fed
                else:
                    v = hist[-1]    # not reaching: the nearest ray, as `live`
                n[v] = n.get(v, 0) + 1
                on_food += v in ft
                back[v] = pursuit[v] = 0
                # he was fed after `away` set aside + `unfed` standing at it: next
                # time stay away SPENT of that (first meal on it: 0 — he camps)
                away[v] = int(SPENT * (away[v] + unfed.get(v, 0))) if v in away else 0
                unfed[v] = 0
                sup[v] = i + away[v]
        elif mode == "want":
            steered += 1
            run_unfed += 1
            longest_unfed = max(longest_unfed, run_unfed)
            pursuit[target] = pursuit.get(target, 0) + 1
            unfed[target] = unfed.get(target, 0) + near
            if pursuit[target] > CAMP:
                sup[target] = i + BASE * 2 ** back.get(target, 0)
                back[target] = back.get(target, 0) + 1
                pursuit[target] = 0
                run_unfed = 0
                giveups += 1
                giveups_food += target in ft
        elif mode == "mouth":
            mouthed += 1
            run_unfed = 0
            m_chase[target] = m_chase.get(target, 0) + 1
            m_near[target] = m_near.get(target, 0) + near
            if m_near[target] > MOUTH or m_chase[target] > CHASE:
                sup[target] = i + BASE * 2 ** back.get(target, 0)
                back[target] = back.get(target, 0) + 1
                m_near[target] = m_chase[target] = 0
                tried += 1
        else:
            run_unfed = 0
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            deaths += 1
            last = int(info["meals"])
    return {"arm": arm, "seed": seed, "t1": t1, "meals": meals, "decisions": stop,
            "deaths": deaths, "steered": steered, "mouthed": mouthed,
            "giveups": giveups, "giveups_of_food": giveups_food,
            "imprints_on_food": on_food, "longest_unfed_pursuit": longest_unfed,
            "foods_tabled": len(set(n) & ft), "tried": tried,
            "table": {str(k_): v for k_, v in sorted(n.items())},
            "away": {str(k_): v for k_, v in sorted(away.items())},
            "meal_at": meal_at, "imprints": sum(n.values()), "cpu_s": round(time.process_time() - t0, 1)}


def _world(args) -> list[dict]:
    seed, after, cap = args
    h = le.cell("held8", seed, after, cap)
    rows = [{"arm": "held8", "seed": seed, "t1": h["t1"], "decisions": h["decisions"],
             "meals": (h["t1"] is not None) + h["meals_after"], "deaths": h["deaths"]}]
    if h["t1"] is not None and h["t1"] <= cap - after:
        stop = h["t1"] + after + 1          # live_eye runs i = 0 .. t1 + after
        lv = le.cell("live", seed, after, cap)
        rows.append({"arm": "live", "seed": seed, "t1": lv["t1"], "decisions": lv["decisions"],
                     "meals": 1 + lv["meals_after"], "deaths": lv["deaths"],
                     "table": lv["table"]})
        rows += [cell(a, seed, stop) for a in ARMS[2:]]
    for r in rows:
        print(json.dumps(r), file=sys.stderr, flush=True)
    return rows


def run(seeds, workers=1, after=AFTER, cap=CAP) -> dict:
    jobs = [(s, after, cap) for s in seeds]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as p:
            rows = sum(p.map(_world, jobs, chunksize=1), [])
    else:
        rows = sum((_world(j) for j in jobs), [])
    R = {(r["arm"], r["seed"]): r for r in rows}
    ign = [s for s in seeds if ("live", s) in R]
    M = {a: [R[(a, s)]["meals"] for s in ign] for a in ARMS}
    out = {"ignited": ign, "t1_held8": {str(s): R[("held8", s)]["t1"] for s in seeds},
           "meals": M, "totals": {a: sum(v) for a, v in M.items()},
           "t1": {a: [R[(a, s)]["t1"] for s in ign] for a in ARMS},
           "same_t1": all(R[(a, s)]["t1"] == R[("held8", s)]["t1"]
                          for a in ("live", "tour", "tour-yoked") for s in ign),
           "diag": {a: {k_: [R[(a, s)].get(k_) for s in ign] for k_ in
                        ("deaths", "giveups", "giveups_of_food", "foods_tabled",
                         "longest_unfed_pursuit", "tried")}
                    for a in ARMS[2:]},
           "pins": {"NEAR": NEAR, "CAMP": CAMP, "BASE": BASE, "SPENT": SPENT,
                    "MOUTH": MOUTH, "CHASE": CHASE, "AFTER": after, "CAP": cap},
           "rows": rows}
    return out


def main(argv):
    cmd = argv[0] if argv else "dev"
    workers = int(argv[1]) if len(argv) > 1 else 1
    if cmd == "dev":
        seeds = tuple(int(s) for s in argv[2:]) or DEV_SEEDS
        r = run(seeds, workers)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "dev.json").write_text(json.dumps(r, indent=1))
    else:
        raise SystemExit(__doc__)
    r.pop("rows", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
