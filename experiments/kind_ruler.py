"""LAB — A RULER THAT CAN FAIL, FOR THE HINDSIGHT EYE ON THE KIND PAINT
(no ledger row; a probe of the RULER, not of Jack. Seat: Vision encoder —
the hindsight eye stays ENTERED and is raced against the holder `bins16`
only once this ruler exists and has been shown able to fail: rulings 134,
140; ORDERS "the lab's next unit (box-minutes, no learner)".)

WHY. `own_kind.py` died by its rule (11-26): on `paint="kind"` the yoked
twin ate 57 against live 73, because with five names a wrong imprint is
crossed off in one 400-decision wait and BOTH arms sit on the regrow
ceiling (~9 meals in 2,500 decisions). "Meals in the window" cannot tell
"the mouth named it" from "going unfed crossed the others off".

THE RULER (what is new; rebuilt 11-28 after the dev smoke, before any
validation world was lived). The first draft measured decisions from the
first meal to the fifth inside one world. Dev world 1101 killed it: `told`,
a random-imprint table and the no-unlearning yoked twin ate at the SAME
decisions (2720, 2970, 3221 ... every 251): after his first meal he sits
on the food while it regrows, and any name that keeps him near it eats as
well as the right one. own_kind's rows say the same of the end tables: the
yoked twin's often ends with food on top, because the looks before a meal
are full of the food he is sitting on. Inside one world nothing below the
ceiling reads the name. Where a name shows is a world he has NOT lived:
  F = decisions from BIRTH to the first meal in an unseen world, carrying
      the table his arm ENDED its previous world with (world s takes the
      table of the world before it in VAL_SEEDS, as own_kind), censored at
      AFTER (no meal by AFTER reads AFTER: a tie, never a win). Lower wins.
      own_kind's carried arm read ~54 (median) where babble reads ~943.
  Every naming runs in two FORMS: with unlearning (ext) and with none
  (noext) — the no-unlearning twin of ruling 134. The carried form keeps
  its form in the new world.
  NAMINGS (the rule that writes the table at each meal, in world prev):
    live    the kind of the closest lit bin in the look before the bite
            (the hindsight eye; `own_kind`'s live)
    yoked   the closest kind in one of his OWN looks drawn uniformly from
            those BEFORE his first meal in that world (up to 1,100 of them,
            looks with nothing lit skipped;
            v2, see below): what he sees, without what his mouth said
    yoked2  the same with a second, independent stream (the self-check)
    oracle  KNOWN POSITIVE: food (simulator truth) at every meal
    chance  MUST FAIL: one of the five kinds uniformly at every meal
  plus `told-blind` (birth only): {food: 1} steering on a look whose 16
  bins are rolled by a fresh random offset every decision — the right
  name, direction destroyed (J30's note: sight must be separated from
  living and moving), and `babble` = held8's own first meal in world s.

RE-REGISTERED v2 (11-28, before any validation world; v1 at `8ae32ad4`).
  v1's yoke (own_kind's: a look 200-1,000 decisions earlier) read RULER
  BLIND on dev 1101-1102: he sits on the regrowing food for most of the
  window, so earlier looks show food closest too — the yoked end table in
  1101 was {food: 7} and tied the oracle exactly (F 14 = 14). A twin that
  holds the answer is not a control. v2 draws the yoke from his looks
  before he ever ate in that world. Nothing else moved: arms, worlds,
  pins, 6 of 8, 1.5x. Dev v1 numbers: oracle F 48 / 14, babble 2500 / 76,
  chance-noext 2500 / 2500, told-blind 2500 / 2500.

RULE OF THE RULER — DECLARED BEFORE ANY VALIDATION WORLD IS LIVED.
  X BEATS Y on a form iff F_X < F_Y on >= 6 of the 8 worlds AND
  sum F_Y >= 1.5 x sum F_X.
  The race's rule (as it will be applied to the hindsight eye):
  naming N PASSES iff, in BOTH forms, N BEATS babble AND N BEATS yoked.
  RULER CAN FAIL iff ALL of:
    known positive  `oracle` PASSES            (else RULER BLIND)
    must fail       `chance` does NOT pass
    must fail       `told-blind` does NOT pass (its one form read for both)
    self-check      yoked2 does NOT beat yoked AND yoked does NOT beat
                    yoked2, in either form    (a twin against itself)
  else RULER CANNOT FAIL (named). No extension; a miss is a miss.
  Not gated by ignition: every birth arm starts at decision 0 of world s;
  world prev's arm runs to t1 + AFTER or CAP as own_kind (a world whose
  held8 never eats hands an EMPTY table on, which reads as babble).
  Pins NEAR 0.13, PATIENCE 400, AFTER 2,500, CAP 9,000 are `live_eye`'s,
  unmoved; 6 of 8 and 1.5x are declared here, unfitted to any world.
  Beside, gating nothing: `live` on the ruler (the hindsight eye's own
  read; its race against the holder is later, on worlds none of these
  arms has lived); each end table; meals in world prev.
  Worlds: dev 1101-1102 for bugs only; validation 1801-1808, never lived
  by any lab file (1101-1720 are taken).
  Cannot say: that a LEARNER is measured fairly by this ruler (a learner
  brings its own blind twin, J30); anything about a camera eye; what
  happens in a world whose kinds are painted differently.

PRIOR (SYSTEM.md 3a). Not a capability claim, an instrument check:
LESSONS_CORE 2, 9 (every ruler carries a known positive and a control that
must fail, through the real code path); PLAN rules ("a harness must first
report no difference when a learner is compared with itself"). The arms are
`own_kind`'s reflex and pins; Rescorla-Wagner (1972, from memory) as there.

    python -m experiments.kind_ruler dev [workers] [seeds...]
    python -m experiments.kind_ruler validate [workers]
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

from experiments import own_kind as ok  # noqa: E402

OUT = Path("/data/jack-logs/kind_ruler")
K, B, FOOD = ok.K, ok.B, ok.FOOD
NEAR, PATIENCE, AFTER, CAP = ok.NEAR, ok.PATIENCE, ok.AFTER, ok.CAP
VAL_SEEDS = tuple(range(1801, 1809))
DEV_SEEDS = (1101, 1102)
RATIO, SHARE = 1.5, 0.75
NAMINGS = ("live", "yoked", "yoked2", "oracle", "chance")
FORMS = ("ext", "noext")


def live_cell(naming: str, form: str, seed: int, table0: dict | None = None,
              birth: bool = False, blind: bool = False,
              after: int = AFTER, cap: int = CAP) -> dict:
    """One arm in one world. birth=False: world prev, from an empty table,
    runs to t1 + after (own_kind's t1 arms). birth=True: from decision 0
    with table0, stops at the first meal or `after` (the ruler's F)."""
    from experiments.world_card import Held
    ext = form == "ext"
    w = ok.make_env(seed)
    pol = Held(seed, 8)
    yoke = np.random.RandomState(seed * 7919 + (5 if naming != "yoked2" else 17))
    ch = np.random.RandomState(seed * 104729 + 3)
    roll = np.random.RandomState(seed * 31337 + 1)
    o, info = w.reset(seed=seed)
    pol.new_life()
    table = {int(k): int(v) for k, v in (table0 or {}).items()}
    unfed: dict[int, int] = {}
    hist: list = []
    last, t1 = int(info["meals"]), None
    pre: list = []
    meal_at: list[int] = []
    imprints = on_food = ext_n = ext_food = deaths = 0
    t0, i = time.process_time(), 0
    while i < cap:
        if birth and (t1 is not None or i >= after):
            break
        if not birth and t1 is not None and i > t1 + after:
            break
        close = np.asarray(o[:K * B], dtype=np.float32).reshape(K, B)
        hist.append(ok.closest_kind(close))
        a = pol.act(o)                      # one babble stream for every arm
        look = np.roll(close, int(roll.randint(B)), axis=1) if blind else close
        target, near = None, False
        if table:
            pick = ok.steer(look, table)
            if pick is not None:
                j, target = pick
                ang = 2 * math.pi * j / B
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(ang), math.sin(ang)
                near = bool(1.0 - look[target, j] < NEAR)
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            meal_at.append(i)
            if t1 is None:
                t1 = i
                pre = [k for k in hist[:-1] if k is not None]   # looks before the bite-look
            if naming == "oracle":
                c = FOOD
            elif naming == "chance":
                c = int(ch.randint(K))
            elif naming in ("yoked", "yoked2"):
                c = pre[int(yoke.randint(len(pre)))] if pre else None
            elif naming == "live":
                c = hist[-1]
            else:                           # told-blind / babble: no imprint
                c = None
            if c is not None:
                table[c] = table.get(c, 0) + 1
                unfed[c] = 0
                imprints += 1
                on_food += c == FOOD
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
        hist = hist[-1100:]
        i += 1
    F = t1 if (t1 is not None and t1 < after) else after
    return {"naming": naming, "form": form, "seed": seed, "birth": birth,
            "blind": blind, "t1": t1, "F": int(F),
            "meals": len(meal_at), "meal_at": meal_at[:12], "decisions": i,
            "deaths": deaths, "imprints": imprints, "imprints_on_food": on_food,
            "extinctions": ext_n, "extinctions_of_food": ext_food,
            "table0": {str(k): v for k, v in sorted((table0 or {}).items())},
            "table": {str(k): v for k, v in sorted(table.items())},
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(kw):
    r = live_cell(**kw)
    print(json.dumps(r), file=sys.stderr, flush=True)
    return r


def beats(fx, fy) -> dict:
    need = math.ceil(SHARE * len(fx))
    w = sum(a < b for a, b in zip(fx, fy))
    return {"worlds": w, "need": need, "sum": sum(fx), "sum_ref": sum(fy),
            "ok": w >= need and sum(fy) >= RATIO * sum(fx)}


def judge(birth_rows, seeds) -> dict:
    R = {(r["naming"], r["form"], r["seed"]): r for r in birth_rows}
    F = {f"{n}/{f}": [R[(n, f, s)]["F"] for s in seeds] for n in NAMINGS for f in FORMS}
    F["told-blind"] = [R[("told-blind", "noext", s)]["F"] for s in seeds]
    F["babble"] = [R[("babble", "noext", s)]["F"] for s in seeds]

    def passes(n):
        out = {}
        for f in FORMS:
            x = F["told-blind"] if n == "told-blind" else F[f"{n}/{f}"]
            out[f] = {"vs babble": beats(x, F["babble"]),
                      "vs yoked": beats(x, F[f"yoked/{f}"])}
        out["PASSES"] = all(v["ok"] for f in FORMS for v in out[f].values())
        return out

    V = {n: passes(n) for n in ("oracle", "chance", "told-blind", "live")}
    selfc = {f: {"yoked2>yoked": beats(F[f"yoked2/{f}"], F[f"yoked/{f}"]),
                 "yoked>yoked2": beats(F[f"yoked/{f}"], F[f"yoked2/{f}"])} for f in FORMS}
    self_ok = not any(v["ok"] for f in FORMS for v in selfc[f].values())
    bad = [n for n in ("chance", "told-blind") if V[n]["PASSES"]]
    if not self_ok:
        bad.append("yoked2 vs yoked self-check")
    if not V["oracle"]["PASSES"]:
        ruler = "RULER BLIND"
    elif bad:
        ruler = "RULER CANNOT FAIL (" + ", ".join(bad) + ")"
    else:
        ruler = "RULER CAN FAIL"
    return {"ruler": ruler, "live_beside": V["live"]["PASSES"], "F": F,
            "totals": {k: sum(v) for k, v in F.items()}, "verdicts": V,
            "self_check": selfc,
            "pins": {"NEAR": NEAR, "PATIENCE": PATIENCE, "AFTER": AFTER,
                     "CAP": CAP, "RATIO": RATIO, "SHARE": SHARE}}


def run(seeds, workers=1) -> dict:
    def go(jobs):
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(workers) as p:
                return p.map(_cell, jobs, chunksize=1)
        return [_cell(j) for j in jobs]
    a_rows = go([{"naming": n, "form": f, "seed": s} for s in seeds
                 for n in NAMINGS for f in FORMS]
                + [{"naming": "babble", "form": "noext", "seed": s} for s in seeds])
    A = {(r["naming"], r["form"], r["seed"]): r for r in a_rows}
    jobs = []
    for k, s in enumerate(seeds):
        prev = seeds[k - 1]                  # the first world takes the last's: another world
        for n in NAMINGS:
            for f in FORMS:
                jobs.append({"naming": n, "form": f, "seed": s, "birth": True,
                             "table0": A[(n, f, prev)]["table"]})
        jobs.append({"naming": "told-blind", "form": "noext", "seed": s,
                     "birth": True, "blind": True, "table0": {str(FOOD): 1}})
    b_rows = go(jobs)
    for s in seeds:                          # babble's F: held8's own first meal in world s
        r = dict(A[("babble", "noext", s)])
        r["F"] = r["t1"] if (r["t1"] is not None and r["t1"] < AFTER) else AFTER
        r["birth"] = True
        b_rows.append(r)
    out = judge(b_rows, seeds)
    out["prev_world"] = {f"{n}/{f}/{s}": {"t1": A[(n, f, s)]["t1"],
                                          "meals": A[(n, f, s)]["meals"],
                                          "table": A[(n, f, s)]["table"]}
                         for s in seeds for n in NAMINGS for f in FORMS}
    out["cpu_s"] = round(sum(r["cpu_s"] for r in a_rows + b_rows), 1)
    out["rows"] = a_rows + b_rows
    return out


def main(argv):
    cmd = argv[0] if argv else "dev"
    workers = int(argv[1]) if len(argv) > 1 else 1
    if cmd == "dev":
        seeds = tuple(int(s) for s in argv[2:]) or DEV_SEEDS
        r, name = run(seeds, workers), "dev.json"
    elif cmd == "validate":
        r, name = run(VAL_SEEDS, workers), "validate.json"
    else:
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(r, indent=1))
    r.pop("rows", None)
    r.pop("prev_world", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
