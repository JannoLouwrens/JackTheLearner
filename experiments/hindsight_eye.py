"""LAB — THE MOUTH NAMES IT, THE EYE REMEMBERS BACKWARDS (no ledger row; a
pilot. Seat: Vision encoder, BY DEFAULT; PLAN part "Learning to eat", right-
hand column; follow-up to `taste_eye.py`, DEAD 11-21 at .080).

WHY. `taste_eye` died on its LABEL, not its net: the same ring handed the
food's ray scores .829, trained on "I eat within 2 s" .080 (untrained .082).
A look-level label 2 s early does not say WHICH ray. But at the moment of the
meal the food is at his mouth: it is the nearest thing he sees.

THE BET. At each meal he imprints the texture of his NEAREST ray (his own
eye, his own mouth; no simulator). He then walks back through the looks he
remembers from the 8 s before that meal and marks, in each, the nearest ray
wearing that texture — the same thing, further off. Those marks are the
labels `told` got from the simulator. Same ring, same optimiser, same steps
as `told`; only the label's source changes.

PRIOR (SYSTEM.md 3a; from memory, not re-read today). Hindsight relabelling
(Andrychowicz 2017) and labels from tracking an object back in time (Pathak
2017, "Learning features by watching objects move"; Wang & Gupta 2015) are
published at 10^4-10^6 clips. Nothing on record at ~200 meals with taste as
the anchor. Biology: food imprinting / conditioned taste preference.

DATA. `taste_eye`'s kernel data, as it lies on disk: train worlds 1101-1124
(215 meals), judge worlds 1201-1220. Development (label design, the dev
numbers in LAB) touched ONLY worlds 1101-1124 (fit 1101-1118, look at
1119-1124). DISCLOSED: the judge worlds were read once by `taste_eye`'s rule
(pooled numbers only; no look of them was inspected for this design).

ARMS (3 initialisations each for the nets)
  hindsight   the bet as a NET: the ring, trained on the walked-back marks.
  colours     the bet as a MEMORY, no net: a count of the textures that were
              at his mouth; peak = the nearest ray wearing the most-eaten
              texture in sight. (Dev, train worlds only, said before the
              judge: table .948, net .414 at 1,000 steps and one init.)
  yoked / colours-yoked   MUST FAIL: the same walk back through the same
              pre-meal looks (the same table), but the imprint is the nearest
              ray's texture at a RANDOM non-meal look of the same world (the
              mouth did not name it).
  untrained   MUST FAIL: the ring, never trained.
  nearest     hand rule: the nearest ray.
  told        CEILING: NOT refitted (box budget) — `taste_eye`'s read of the
              same looks with the same judge code stands: .829 >= .80.

JUDGE. `taste_eye`'s, unchanged (judge_tables / hits): over judge looks with
an uneaten floor food in sight, HIT = the peak ray is on an uneaten food.

RULE — DECLARED BEFORE ANY JUDGE-WORLD LOOK IS SCORED BY THIS FILE. Two
forms of one bet, each judged alone by taste_eye's thresholds, word for word,
with the yoked twin in shuffled's place:
  A form is ALIVE iff, pooled over the 20 judge worlds, its HIT >= .50, AND
  it is >= .25 above its yoked twin, untrained AND nearest, AND ahead of each
  of its yoked twin and untrained on >= 16 of 20 worlds. Else that form is
  DEAD. The bet is ALIVE if either form is, and the entry names which.
  VOID if fewer than 60 imprinted meals or fewer than 500 judge looks.
  No extension; a miss is a miss.
  Cannot say, whatever happens: that a learner USES it; that it survives his
  own changing behaviour; that it finds a food he has never eaten the colour
  of (reported beside: HIT split by the judge world's colour set); and for
  `colours`: anything about an eye whose colours are not W0's one clean
  number per object (a camera would need the net).
  If ALIVE, the next step is the workers' and needs the Director (ruling 58):
  E2c's arm with its handed direction replaced by this sense's own (bearing
  of the peak ray, that ray's distance), the table filled as he lives.

STAGE 2 — DOES POINTING FEED HIM? DECLARED 2026-11-21 23:3x, BEFORE ANY
STAGE-2 WORLD IS LIVED and before the stage-1 read is home; run only if the
`colours` form reads ALIVE. NOT a learner: a hand reflex as a ruler
(taste_eye's stage 2, its worlds — never lived, its reflex was not run —
and its rule word for word). The table is CARRIED from the 215 train-world
meals and frozen here; "filled as he lives, from empty" is NOT tested.
  Worlds 1301..1305, 2,400 decisions each (box budget; taste_eye wrote
  3,600), the same world K, mujoco 3.17.0 (`/data/venvs/jack-std`).
  mouth-reflex   when a ray wears an eaten texture, push (body frame) along
                 the nearest ray wearing the most-eaten one; else held8.
  yoked-reflex   MUST FAIL: the same reflex on the yoked table.
  held8          the floor (world_card.Held(seed, 8)).
  ALIVE-2 iff mouth-reflex eats more meals than held8 AND than yoked-reflex
  on >= 4 of 5 worlds, and >= 2x each in total. Else the table points and the
  pointing does not feed him: said so, stage 1 stands.

    python -m experiments.hindsight_eye dev     # train worlds only
    python -m experiments.hindsight_eye read    # the one judged read
    python -m experiments.hindsight_eye reflex  # stage 2 (jack-std venv)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import taste_eye as te  # noqa: E402

N, WIN = te.N, te.KEEP_WIN
OUT = Path("/data/jack-logs/hindsight_eye")


def approaches(ttm: np.ndarray, seed: np.ndarray):
    """For each meal look m: the kept looks m, m-1, .. while they are the
    consecutive decisions before it (ttm 0, 1, 2, ..), same world."""
    out = []
    for m in np.flatnonzero(ttm == 0):
        idx, j = [m], 1
        while j < WIN and m - j >= 0 and ttm[m - j] == j and seed[m - j] == seed[m]:
            idx.append(m - j)
            j += 1
        out.append(np.array(idx))
    return out


def labels(x, ttm, seed, yoked: bool, rs: int = 11):
    """(look index, target ray) from his own eye and mouth only."""
    dist = x[:, 0::2].astype(np.float32)
    tex = x[:, 1::2].astype(np.float32)
    rng = np.random.RandomState(rs)
    li, lr, imprints = [], [], []
    for idx in approaches(ttm, seed):
        m = idx[0]
        if yoked:
            pool = np.flatnonzero((seed == seed[m]) & (ttm >= WIN))
            a = int(rng.choice(pool))
        else:
            a = m
        c = tex[a, dist[a].argmin()]
        imprints.append(float(c))
        for i in idx:
            same = np.abs(tex[i] - c) < te.TEX_TOL
            if same.any():
                li.append(i)
                lr.append(int(np.where(same, dist[i], 9.0).argmin()))
    return np.array(li, int), np.array(lr, int), np.array(imprints)


def colour_peak(xj, imprints):
    """No net: the nearest ray wearing the most-eaten texture in sight."""
    dist = xj[:, 0::2].astype(np.float32)
    tex = xj[:, 1::2].astype(np.float32)
    vals, cnt = np.unique(np.round(imprints, 2), return_counts=True)
    sc = np.zeros_like(tex)
    for v, c_ in zip(vals, cnt):
        sc = np.where(np.abs(tex - v) < 0.006, float(c_), sc)
    return (sc - dist * 0.5).argmax(1)       # dist in [0,1]: breaks ties only


def run(train, judge, steps=3000, inits=te.INITS, enforce=True) -> dict:
    import torch
    torch.set_num_threads(1)
    z = np.load(te.OUT / te.DATA_NAME)
    x, foods, ttm, seed = z["x"], z["foods"], z["ttm"], z["seed"]
    tr, ju = np.isin(seed, train), np.isin(seed, judge)
    xt, tt, st = x[tr], ttm[tr], seed[tr]
    xj, sj = x[ju], seed[ju]
    on, _kf, d = te.judge_tables(xj, foods[ju], sj)
    el = on.any((1, 2))
    xj, sj, on, d = xj[el], sj[el], on[el], d[el]
    dist = xj[:, 0::2].astype(np.float32)
    worlds = [s for s in judge if (sj == s).sum() > 0]
    arms, diag = {}, {}

    def enter(name, hit_list):
        hm = np.mean(hit_list, axis=0)
        arms[name] = {"pooled": float(hm.mean()),
                      "world": np.array([hm[sj == s].mean() for s in worlds])}

    # label quality, TRAIN worlds only (simulator truth, a diagnostic)
    on_t, _k, _d = te.judge_tables(xt, foods[tr], st)
    lab = {}
    for name, yk in (("hindsight", False), ("yoked", True)):
        li, lr, imp = labels(xt, tt, st, yk)
        lab[name] = (li, lr, imp)
        diag[name] = {"imprints": int(len(imp)), "labelled_looks": int(len(li)),
                      "label_on_food": (float(on_t[li, :, lr].any(1).mean())
                                        if len(li) else None)}
        nets = []
        for i in inits:
            nets.append(te.train_told(xt[li], lr, i, steps=steps) if len(li) >= 64
                        else te.build_net(N, i))
            print(f"fit {name} init {i} done {time.strftime('%H:%M:%S')}",
                  file=sys.stderr, flush=True)
        enter(name, [te.hits(te.scores(n_, xj).argmax(1), on) for n_ in nets])
    enter("untrained", [te.hits(te.scores(te.build_net(N, i), xj).argmax(1), on)
                        for i in inits])
    enter("nearest", [te.hits(dist.argmin(1), on)])
    enter("colours", [te.hits(colour_peak(xj, lab["hindsight"][2]), on)])
    enter("colours-yoked", [te.hits(colour_peak(xj, lab["yoked"][2]), on)])

    sets = {s_: " ".join(f"{v:.2f}" for v in te.food_tex(int(s_))[:3]) for s_ in worlds}
    by_set = {k: {u: round(float(np.mean([v["world"][i] for i, s_ in enumerate(worlds)
                                           if sets[s_] == u])), 3)
                  for u in sorted(set(sets.values()))} for k, v in arms.items()}
    P = {k: v["pooled"] for k, v in arms.items()}
    need = 16 if enforce else max(1, int(0.8 * len(worlds)))
    clauses, ahead = {}, {}
    for form, twin in (("hindsight", "yoked"), ("colours", "colours-yoked")):
        t = arms[form]
        ah = {o: int((t["world"] > arms[o]["world"]).sum()) for o in (twin, "untrained")}
        ahead[form] = ah
        clauses[form] = {"hit>=.50": P[form] >= 0.50,
                         "+.25 over yoked twin": P[form] - P[twin] >= 0.25,
                         "+.25 over untrained": P[form] - P["untrained"] >= 0.25,
                         "+.25 over nearest": P[form] - P["nearest"] >= 0.25,
                         f"ahead of yoked twin >={need}": ah[twin] >= need,
                         f"ahead of untrained >={need}": ah["untrained"] >= need}
    forms = {f: "ALIVE" if all(c.values()) else "DEAD" for f, c in clauses.items()}
    void = []
    if diag["hindsight"]["imprints"] < 60:
        void.append("fewer than 60 imprinted meals")
    if len(xj) < 500:
        void.append(f"{len(xj)} judge looks < 500")
    return {"verdict": "VOID" if void else "ALIVE" if "ALIVE" in forms.values() else "DEAD",
            "forms": forms, "void": void, "clauses": clauses, "pooled": P, "ahead": ahead,
            "worlds": len(worlds), "judge_looks": int(len(xj)), "labels": diag,
            "by_set": by_set, "steps": steps, "inits": list(inits),
            "world_hits": {k: [round(float(v_), 3) for v_ in v["world"]]
                           for k, v in arms.items()}}


REFLEX_SEEDS, REFLEX_DEC = (1301, 1302, 1303, 1304, 1305), 2400


def tables() -> dict:
    """The two carried tables: eaten texture (2 dp) -> count, train worlds."""
    z = np.load(te.OUT / te.DATA_NAME)
    tr = np.isin(z["seed"], te.TRAIN_SEEDS)
    x, tt, st = z["x"][tr], z["ttm"][tr], z["seed"][tr]
    out = {}
    for name, yk in (("mouth-reflex", False), ("yoked-reflex", True)):
        v, c = np.unique(np.round(labels(x, tt, st, yk)[2], 2), return_counts=True)
        out[name] = (v, c)
    return out


def reflex_cell(arm: str, seed: int, table, n_dec: int = REFLEX_DEC) -> dict:
    import math
    from experiments.world_card import Held
    w = te.make_env(seed)
    pol = Held(seed, 8)
    o, info = w.reset(seed=seed)
    pol.new_life()
    m0, t0, steered = int(info["meals"]), time.process_time(), 0
    for _ in range(n_dec):
        a = pol.act(o)
        if table is not None:
            dist = o[0:2 * N:2].astype(np.float32)
            tex = o[1:2 * N:2].astype(np.float32)
            sc = np.zeros(N, np.float32)
            for v, c_ in zip(*table):
                sc = np.where(np.abs(tex - v) < 0.006, float(c_), sc)
            if sc.max() > 0:
                k = int((sc - dist * 0.5).argmax())
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
                steered += 1
        o, _r, term, trunc, info = w.step(a)
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
    return {"arm": arm, "seed": seed, "meals": int(info["meals"]) - m0,
            "lives": len(w.lives), "steered": steered,
            "cpu_s": round(time.process_time() - t0, 1)}


def reflex(seeds=REFLEX_SEEDS, n_dec=REFLEX_DEC) -> dict:
    tb = tables()
    tb["held8"] = None
    rows = []
    for s_ in seeds:
        for a in ("mouth-reflex", "yoked-reflex", "held8"):
            rows.append(reflex_cell(a, s_, tb[a], n_dec))
            print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    M = {a: [r["meals"] for r in rows if r["arm"] == a] for a in tb}
    t, cl = M["mouth-reflex"], {}
    for o_ in ("held8", "yoked-reflex"):
        cl[f"worlds over {o_} >=4"] = sum(a > b for a, b in zip(t, M[o_])) >= 4
        cl[f"2x {o_}"] = sum(t) >= 2 * sum(M[o_]) and sum(t) > 0
    return {"verdict": "ALIVE-2" if all(cl.values()) else "DEAD-2", "clauses": cl,
            "meals": M, "totals": {a: sum(v) for a, v in M.items()},
            "lives": {a: [r["lives"] for r in rows if r["arm"] == a] for a in tb},
            "steered": {a: [r["steered"] for r in rows if r["arm"] == a] for a in tb},
            "tables": {k: [[round(float(a), 2), int(b)] for a, b in zip(*v)]
                       for k, v in tb.items() if v is not None},
            "decisions_per_cell": n_dec, "cpu_s": sum(r["cpu_s"] for r in rows)}


def main(argv):
    cmd = argv[0] if argv else "dev"
    if cmd == "dev":
        r = run(tuple(range(1101, 1119)), tuple(range(1119, 1125)),
                steps=1000, inits=(0,), enforce=False)
    elif cmd == "read":
        r = run(te.TRAIN_SEEDS, te.JUDGE_SEEDS)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "read.json").write_text(json.dumps(r, indent=1, default=str))
    elif cmd == "reflex":
        r = reflex()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "reflex.json").write_text(json.dumps(r, indent=1, default=str))
    elif cmd == "reflex-smoke":            # smoke seed 153 only, 300 decisions
        r = reflex(seeds=(153,), n_dec=300)
    else:
        raise SystemExit(__doc__)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
