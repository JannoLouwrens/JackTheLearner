"""LAB — WHICH KIND FEEDS HIM: THE MOUTH-TAUGHT TABLE ON THE KIND PAINT, AND
CARRIED INTO WORLDS HE HAS NEVER SEEN (no ledger row; a pilot. Seat: Vision
encoder, BY DEFAULT; PLAN parts "Learning to eat" / "Eye" / "Memory and
sleep" right-hand columns; ORDERS 11-26 "Lab's next unit"; follow-up to
`live_eye.py` (ALIVE 11-22) and `first_imprint.py` (DEAD 11-25, "return only
... in a kind-coloured world where a wrong name costs one wait in a
lifetime"). The new reason is that world: `paint="kind"` exists since 11-26.)

WHY. J24's eye (`eye="bins16"`, `paint="kind"`) gives him 5 kinds x 16
closeness bins, the same in every world, and NOTHING says which kind feeds
him. The hand forager that eats 25.5 meals a life on it is TOLD: it reads
channel 0. A learner handed "the food bins" is handed the same thing. The
lab's table was never told; on the old paint it had to re-learn a new code
per object per world, and one wrong first imprint locked a birth out.

THE BET. (1) From an EMPTY table of five counts, his own mouth names the
kind: after the first babbled meal he eats several times what babble eats,
as on 11-22. (2) Because a kind looks the same everywhere, the table he
ends one world with feeds him FROM BIRTH in the next world, with no babbled
first meal (median ~950 decisions on 11-22) — as much as a forager told the
channel. Nothing of the kind could be carried on the old paint.

PRIOR (SYSTEM.md 3a). §28: every published forager's ray returns a CLASS
and the class that pays is fixed by the reward, learned over 2M-25M steps
by gradient (FoodCollector 2M; SAC PointGoal 3e6). Biology in the same
section: taste labels the look, bees in 1-3 rewarded visits. §26: no
published learner takes "what was at my mouth when I ate" as an input.
Rescorla-Wagner 1972 acquisition and extinction (from memory): a table,
tens of trials — our size. A count table over 5 kinds is inside the budget;
whether ONE meal is enough and whether it survives a change of world is not
on record.

WORLD. `taste_eye.make_env`'s world (world K, entry rung, body frames,
mujoco 3.17.0 in `/data/venvs/jack-std`) with ONLY the eye changed:
eye="bins16", paint="kind". Not a learner (ruling 58): a hand reflex as a
ruler, `live_eye`'s reflex unchanged — push at the bin that scores highest
on (meals counted for its kind) - 0.5 * (1 - closeness); else babble
(Held 8). The table is his: five counts, written only by his own meals.

ARMS (same world, same seed, same babble stream)
 after the first meal (all the same run until it; t1 asserted equal):
  held8       the floor: babble only.
  live        at each meal: count[the kind of the CLOSEST lit bin in the
              look before the bite] += 1. EXTINCTION as 11-22: each decision
              he pushes at a counted kind nearer than NEAR unfed adds 1 to
              its unfed count; past PATIENCE the kind leaves the table.
  live-noext  the same, no extinction.
  yoked       MUST FAIL: `live`, but each imprint is the closest kind in one
              of his OWN looks 200-1,000 decisions earlier.
  told        the yardstick, gating nothing in bet 1: after the first meal
              the table is {food: 1} by hand, no imprint, no extinction.
 from birth (decisions 0..AFTER of the world):
  carried       starts with the table `live` ENDED THE PREVIOUS judge world
                with (1401's comes from 1408: every table is from another
                world); goes on imprinting and unlearning.
  carried-yoked MUST FAIL: the same with `yoked`'s final table.
  told-birth    the yardstick: {food: 1} by hand from decision 0.
  held8-birth   babble, read over the same window (from held8's own run).

PINS. NEAR, PATIENCE, AFTER, CAP are `live_eye`'s, unmoved (0.13, 400,
2,500, 9,000). Dev worlds 1101-1106 only, for bugs, not for pins.
Judge worlds 1401-1408 — the 11-22 worlds, as ordered ("same judge worlds
and kill bars"). Disclosed: they were lived on 11-22 by `live_eye` on the
OLD paint; nothing in this file was fitted to them and no arm here has
lived them on this eye.

RULE — DECLARED BEFORE ANY JUDGE WORLD IS LIVED ON THIS EYE.
  Bet (1): `live_eye`'s rule word for word. A world IGNITES if held8's
  first meal comes by CAP - AFTER. Measure: meals in the AFTER decisions
  after the first meal. VOID if < 6 of 8 ignite or an arm's t1 differs.
  ALIVE iff `live` > held8 AND `live` > yoked on >= 3/4 of ignited worlds
  and >= 2x each in total. Else DEAD.
  Unlearning (read only if (1) ALIVE): NEEDED iff live-noext has an ignited
  world with ZERO meals where `live` has >= 1 and live >= live-noext in
  total; NOT NEEDED iff live-noext >= 0.9 x live; else UNCLEAR.
  Bet (2), read only if (1) is ALIVE, on all 8 worlds, measure = meals in
  decisions 0..AFTER from birth: CARRIES iff `carried` > held8-birth AND
  > carried-yoked on >= 6 of 8 worlds, >= 2x each in total, AND carried
  >= 0.9 x told-birth in total. Else DOES NOT CARRY.
  No extension; a miss is a miss.
  Beside, gating nothing: told vs live (what not being told costs after
  the first meal); first imprints on food of 8; extinctions and how many
  struck food; t1; each world's final table.
  Cannot say, whatever happens: that a LEARNER uses the table (J24's and
  the hindsight eye's later race); anything about a camera eye; that kinds
  as an adult painted them are the kinds he would carve himself.

    python -m experiments.own_kind dev [workers] [seeds...]
    python -m experiments.own_kind judge [workers]
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

OUT = Path("/data/jack-logs/own_kind")
K, B = 5, 16                       # fast_sim.KINDS x fast_sim.BINS, kind-major
FOOD = 0                           # simulator truth: told arms and diagnostics only
T1_ARMS = ("held8", "live", "live-noext", "yoked", "told")
BIRTH_ARMS = ("carried", "carried-yoked", "told-birth")
JUDGE_SEEDS = le.JUDGE_SEEDS
DEV_SEEDS = le.DEV_SEEDS
NEAR, PATIENCE, AFTER, CAP = le.NEAR, le.PATIENCE, le.AFTER, le.CAP


def make_env(seed: int):
    """taste_eye.make_env with only the eye changed."""
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b import LIFE_CAP, REWARD, PilotLives
    from experiments.sim_speed_kcpu import _compat_multiray
    from experiments.taste_eye import D0
    _compat_multiray()
    env = JackEnv(seed=seed, difficulty=D0, reward=REWARD,
                  random_needs=NEEDS_S_TRP, random_layout=True,
                  lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                  move_cost_dial=True, food_floor_only=True, eye="bins16",
                  paint="kind", eye_frame="body", motor_frame="body",
                  native_substep=True)
    return PilotLives(env, cap=LIFE_CAP)


def closest_kind(close: np.ndarray):
    """close: (K, B). The kind of the closest lit bin, None if nothing lit."""
    if close.max() <= 0:
        return None
    return int(np.unravel_index(int(close.argmax()), close.shape)[0])


def steer(close: np.ndarray, table: dict):
    """live_eye's reflex on bins: (bin, kind) of the best counted bin, or None."""
    best, pick = 0.0, None
    for k, c in table.items():
        lit = close[k] > 0
        if not lit.any():
            continue
        sc = np.where(lit, float(c) - (1.0 - close[k]) * 0.5, -np.inf)
        j = int(sc.argmax())
        if pick is None or sc[j] > best:
            best, pick = float(sc[j]), (j, k)
    return pick


def cell(arm: str, seed: int, after: int = AFTER, cap: int = CAP,
         table0: dict | None = None) -> dict:
    from experiments.world_card import Held
    birth = arm in BIRTH_ARMS
    imprinting = arm in ("live", "live-noext", "yoked", "carried", "carried-yoked")
    ext = arm in ("live", "yoked", "carried", "carried-yoked")
    yoked = arm in ("yoked", "carried-yoked")
    w = make_env(seed)
    pol = Held(seed, 8)
    rng = np.random.RandomState(seed * 7919 + 5)
    o, info = w.reset(seed=seed)
    pol.new_life()
    table: dict[int, int] = {}
    if arm == "told-birth":
        table = {FOOD: 1}
    elif birth:
        table = {int(k): int(v) for k, v in (table0 or {}).items()}
    unfed: dict[int, int] = {}
    hist: list = []
    last, t1, meals_after, deaths = int(info["meals"]), None, 0, 0
    meal_at: list[int] = []
    steered = imprints = on_food = ext_n = ext_food = 0
    first_imprint = None
    t0, i = time.process_time(), 0
    while i < cap and ((i < after) if birth else (t1 is None or i <= t1 + after)):
        close = np.asarray(o[:K * B], dtype=np.float32).reshape(K, B)
        hist.append(closest_kind(close))
        a = pol.act(o)                  # always drawn: one babble stream for every arm
        target, near = None, False
        if table:
            pick = steer(close, table)
            if pick is not None:
                j, target = pick
                ang = 2 * math.pi * j / B
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(ang), math.sin(ang)
                steered += 1
                near = bool(1.0 - close[target, j] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            meal_at.append(i)
            if t1 is None:
                t1 = i
                if arm == "told":
                    table = {FOOD: 1}
            else:
                meals_after += 1
            if imprinting:
                if yoked:
                    c = hist[max(0, len(hist) - 1 - int(rng.randint(200, 1001)))]
                else:
                    c = hist[-1]
                if c is not None:
                    table[c] = table.get(c, 0) + 1
                    unfed[c] = 0
                    imprints += 1
                    on_food += c == FOOD
                    if first_imprint is None:
                        first_imprint = c
        elif ext and target is not None and near:
            unfed[target] = unfed.get(target, 0) + 1
            if unfed[target] > PATIENCE:
                del table[target]
                unfed[target] = 0
                ext_n += 1
                ext_food += target == FOOD
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            deaths += 1
            last = int(info["meals"])
        i += 1
    return {"arm": arm, "seed": seed, "t1": t1, "meals_after": meals_after,
            "meals_birth": sum(m < after for m in meal_at),
            "decisions": i, "deaths": deaths, "steered": steered,
            "imprints": imprints, "imprints_on_food": on_food,
            "first_imprint": first_imprint,
            "extinctions": ext_n, "extinctions_of_food": ext_food,
            "table0": {str(k): v for k, v in sorted((table0 or {}).items())},
            "table": {str(k): v for k, v in sorted(table.items())},
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(args):
    r = cell(*args)
    print(json.dumps(r), file=sys.stderr, flush=True)
    return r


def judge_rows(rows, seeds, after=AFTER, cap=CAP) -> dict:
    R = {(r["arm"], r["seed"]): r for r in rows}
    # bet 1: live_eye's own judge, on its four arms
    r1 = le.judge_rows([R[(a, s)] for a in le.ARMS for s in seeds], seeds, after, cap)
    told_t1 = all(R[("told", s)]["t1"] == R[("held8", s)]["t1"] for s in seeds)
    if not told_t1:
        r1["void"].append("told's first meal is not at held8's t1")
    Mb = {a: [R[(a, s)]["meals_birth"] for s in seeds] for a in BIRTH_ARMS}
    Mb["held8-birth"] = [R[("held8", s)]["meals_birth"] for s in seeds]
    Tb = {a: sum(v) for a, v in Mb.items()}
    need = math.ceil(0.75 * len(seeds))
    cl = {}
    for o_ in ("held8-birth", "carried-yoked"):
        cl[f"worlds over {o_} >= {need}"] = sum(
            a > b for a, b in zip(Mb["carried"], Mb[o_])) >= need
        cl[f"2x {o_}"] = Tb["carried"] >= 2 * Tb[o_] and Tb["carried"] > 0
    cl["0.9x told-birth"] = Tb["carried"] >= 0.9 * Tb["told-birth"]
    b2 = ("not read" if r1["bet1"] != "ALIVE"
          else "CARRIES" if all(cl.values()) else "DOES NOT CARRY")
    ign = r1["ignited"]
    r1.update({
        "bet2": b2, "unlearning": r1.pop("bet2"), "clauses2": cl,
        "meals_birth": Mb, "totals_birth": Tb,
        "told_after": [R[("told", s)]["meals_after"] for s in ign],
        "first_imprint": {str(s): R[("live", s)]["first_imprint"] for s in seeds},
        "tables": {str(s): R[("live", s)]["table"] for s in seeds},
        "tables_yoked": {str(s): R[("yoked", s)]["table"] for s in seeds},
        "cpu_s": round(sum(r["cpu_s"] for r in rows), 1)})
    return r1


def run(seeds, workers=1, after=AFTER, cap=CAP) -> dict:
    def go(jobs):
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(workers) as p:
                return p.map(_cell, jobs, chunksize=1)
        return [_cell(j) for j in jobs]
    rows = go([(a, s, after, cap) for s in seeds for a in T1_ARMS])
    R = {(r["arm"], r["seed"]): r for r in rows}
    jobs = []
    for n, s in enumerate(seeds):
        prev = seeds[n - 1]                      # 1401 takes 1408's: another world
        jobs.append(("carried", s, after, cap, R[("live", prev)]["table"]))
        jobs.append(("carried-yoked", s, after, cap, R[("yoked", prev)]["table"]))
        jobs.append(("told-birth", s, after, cap))
    rows += go(jobs)
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
