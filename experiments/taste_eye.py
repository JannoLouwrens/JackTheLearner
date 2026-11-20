"""LAB — TASTE TEACHES THE EYE: a "where is food" sense grown from his own
meals (no ledger row; a pilot. Seat: Vision encoder, BY DEFAULT; PLAN part
"Learning to eat", right-hand column "can his rays tell food from a wall?").

WHY NOW. J15 E2c (11-20, 25/25): a plain learner HANDED the direction to food
eats (sustained 4/5 vs zero 0/5); the same learner with only his eye never
did (J11, 0 of 25). "J11's miss is the eye." So the missing organ is the one
that turns a look into "food is THAT way" — and the handed one is the
simulator's, never his.

THE BET. The only teacher is his own mouth. A small ring-shaped network over
his row of rays (the same weights at every ray, like the place key's ring)
gives each ray a score; the scores are pooled into ONE number trained to say
"I eat within 2 s" — the meal event is the whole label, no direction is ever
given. Because the anatomy is a ring, the ray that carries the pooled answer
IS a bearing in his own body frame: where he looks when he is about to eat.
Random held pushes (the world's own floor, held8) are the only behaviour.

PRIOR (SYSTEM.md 3a). Weak labels localise: a network trained only on "is
there a dog in this image" points at the dog (Oquab 2015, "Is object
localization for free?"; Zhou 2016 class activation maps, ImageNet scale,
10^6 images, from memory — not re-read today). Multiple-instance pooling is
the mechanism. Nothing on record at 10^2 meals with reward-timing as the
label; LAB 11-14 `steerwant` (a linear want from 14 meals, 16 rays, no ring)
DIED at 7/40 vs shuffled 7. New reasons to retry: ~200 meals not 14, the
128-ray body eye, shared ring weights, and E2c's proof that the direction is
what a learner lacks. PRIORS §12 leaves "can rays tell food from wall" open.

WORLD. The pilot world K at the entry rung, exactly E2c's (pilot_b2
make_world: d = 0.75, random needs, random layout, floor food only, move-cost
dial, body eye, body motor, lazy audio, native substep) with eye="row128".
Policy: world_card.Held(seed, 8). Lives capped at 9,000 (PilotLives).

DATA (one Kaggle CPU kernel, < 1 h; nothing is fitted there).
  train worlds 1101..1124, 20,000 decisions each (never run by anything)
  judge worlds 1201..1220,  4,000 decisions each (never run by anything)
  kept: every look within 40 decisions before a meal, 1 in 4 of the others
  (train) / 1 in 2 (judge), drawn by RandomState(seed + 31).
  Per kept look: the 256 eye numbers; decisions until his next meal in that
  life; and FOR THE JUDGE ONLY the simulator's body-frame bearing, distance
  and uneaten flag of every floor food.

ARMS (all from scratch; 3 initialisations each, 0..2)
  taste128    the bet. Label y = "a meal within H = 10 decisions".
  shuffled    MUST FAIL: taste128 with y permuted across training looks.
  untrained   MUST FAIL: the same network, never trained.
  nearest     hand rule, reported and in the rule: the ray with the smallest
              distance (is the map more than "the closest thing"?).
  taste16     reported only: the same on today's 16 rays (row rays 0, 8, ..).
  told        CEILING, INELIGIBLE (simulator-taught): the same network with
              the food's ray given as the label. If it scores < .80 the
              judge or the network is broken: the run is VOID, not DEAD.

JUDGE. A judge look = a kept judge-world look with at least one UNEATEN floor
food IN SIGHT (a ray within 2 rays of its true bearing reads its distance to
0.5 m). HIT = the peak ray is within 3 rays (8.4 deg) of an in-sight uneaten
food's bearing AND reads that food's distance to 0.5 m. (taste16: the ray
nearest the bearing, same distance clause, same looks.) Per world: HIT rate,
mean over the 3 initialisations.

RULE — DECLARED IN THIS COMMIT, BEFORE THE KERNEL RUNS (2026-11-20):
  ALIVE iff, pooled over the 20 judge worlds, taste128 HIT >= .50, AND it is
  >= .25 above shuffled, untrained AND nearest, AND ahead of each of
  shuffled and untrained on >= 16 of 20 worlds. Anything else: DEAD.
  VOID if told < .80, or fewer than 60 training meals, or fewer than 500
  judge looks. No extension; a miss is a miss.
  If ALIVE, the next step is the workers': E2c's arm with its two handed
  numbers replaced by this sense's own (bearing of the peak ray, that ray's
  distance) — a learner run, so it needs the Director (ruling 58).
  Cannot say, whatever happens: that a learner USES it; that it survives his
  own changing behaviour (the looks are random-push looks).

    python -m experiments.taste_eye smoke            # box, ~1 min
    python -m experiments.taste_eye kernel           # the data kernel
    python -m experiments.taste_eye read [datafile]  # fit + judge + rule
    python -m experiments.taste_eye reflex           # stage 2, box

STAGE 2 — DOES POINTING FEED HIM? DECLARED 2026-11-20 BEFORE ANY REFLEX WORLD
IS LIVED, and run only if stage 1 reads ALIVE. NOT a learner and not his: a
hand reflex used as a ruler — every decision push (body frame) straight along
the map's peak ray, the other six actions zero.
  Worlds 1301..1305 (never run), 3,600 decisions each, the same world K.
  taste-reflex   the reflex on taste128 (initialisation 0, kernel data)
  shuf-reflex    MUST FAIL: the same reflex on the shuffled-label twin
  held8          the floor (world_card.Held(seed, 8))
  told-reflex    reported only: the reflex on the simulator's bearing to the
                 nearest uneaten floor food (handed_sense), the ceiling
  ALIVE-2 iff taste-reflex eats more meals than held8 AND than shuf-reflex on
  >= 4 of 5 worlds, and >= 2x each in total. Else the map points and the
  pointing does not feed him: said so, stage 1 stands, no entry.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TRAIN_SEEDS = tuple(range(1101, 1125))
JUDGE_SEEDS = tuple(range(1201, 1221))
TRAIN_DEC, JUDGE_DEC = 20_000, 4_000
D0 = 0.75
N = 128
H = 10
KEEP_WIN = 40
MAX_FOODS = 5
SIGHT_RAYS, HIT_RAYS, DIST_TOL = 2, 3, 0.5
RAY_RANGE = 8.0
INITS = (0, 1, 2)
OUT = Path("/data/jack-logs/taste_eye")
DATA_NAME = "taste_eye_data.npz"
KERNEL_PINS = ["gymnasium==1.4.0", "mujoco==3.17.0"]


# ── the world ───────────────────────────────────────────────────────────────
def make_env(seed: int):
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b import LIFE_CAP, REWARD, PilotLives
    from experiments.sim_speed_kcpu import _compat_multiray
    _compat_multiray()                # mujoco >= 3.3 on Kaggle; no-op here
    env = JackEnv(seed=seed, difficulty=D0, reward=REWARD,
                  random_needs=NEEDS_S_TRP, random_layout=True,
                  lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                  move_cost_dial=True, food_floor_only=True, eye="row128",
                  eye_frame="body", motor_frame="body", native_substep=True)
    return PilotLives(env, cap=LIFE_CAP)


def _foods(env) -> np.ndarray:
    """JUDGE ONLY: (MAX_FOODS, 3) = body-frame bearing, distance, uneaten."""
    from experiments.fast_sim import body_yaw
    from experiments.handed_sense import _floor_names, _wrap
    w, dl = env.w, env.w.drives
    now = dl.t + dl._dt_acc
    body = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
    yaw = body_yaw(w)
    out = np.full((MAX_FOODS, 3), np.nan, np.float32)
    for i, n in enumerate(_floor_names(env)[:MAX_FOODS]):
        p = w.data.xpos[w.model.body(n).id][:2]
        dx, dy = float(p[0] - body[0]), float(p[1] - body[1])
        out[i] = (_wrap(math.atan2(dy, dx) - yaw), math.hypot(dx, dy),
                  float(now >= dl._respawn_at[n]))
    return out


def collect_world(seed: int, n_dec: int, keep_1_in: int) -> dict:
    from experiments.world_card import Held
    w = make_env(seed)
    pol = Held(seed, 8)
    o, info = w.reset(seed=seed)
    pol.new_life()
    X = np.empty((n_dec, 2 * N), np.float16)
    F = np.empty((n_dec, MAX_FOODS, 3), np.float32)
    meal = np.zeros(n_dec, bool)
    life = np.zeros(n_dec, np.int32)
    last, lf = int(info["meals"]), 0
    t0 = time.process_time()
    for i in range(n_dec):
        X[i] = o[:2 * N]
        F[i] = _foods(w.env)
        life[i] = lf
        o, _r, term, trunc, info = w.step(pol.act(o))
        if int(info["meals"]) > last:
            meal[i], last = True, int(info["meals"])
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            lf += 1
            last = int(info["meals"])
    # decisions until the next meal in the same life (0 = this decision)
    ttm = np.full(n_dec, 30_000, np.int32)
    nxt, cur = None, None
    for i in range(n_dec - 1, -1, -1):
        if life[i] != cur:
            cur, nxt = life[i], None
        if meal[i]:
            nxt = i
        if nxt is not None:
            ttm[i] = nxt - i
    rng = np.random.RandomState(seed + 31)
    keep = (ttm < KEEP_WIN) | (rng.randint(keep_1_in, size=n_dec) == 0)
    return {"x": X[keep], "foods": F[keep], "ttm": ttm[keep],
            "seed": np.full(int(keep.sum()), seed, np.int32),
            "meals": int(meal.sum()), "decisions": n_dec, "lives": lf,
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(args):
    return collect_world(*args)


def collect_all(outdir, train=TRAIN_SEEDS, judge=JUDGE_SEEDS,
                train_dec=TRAIN_DEC, judge_dec=JUDGE_DEC, workers=4) -> dict:
    from multiprocessing import Pool
    cells = ([(s, train_dec, 4) for s in train] + [(s, judge_dec, 2) for s in judge])
    with Pool(workers) as p:
        res = p.map(_cell, cells, chunksize=1)
    cat = {k: np.concatenate([r[k] for r in res]) for k in ("x", "foods", "ttm", "seed")}
    summ = {"worlds": [{k: (int(r["seed"][0]) if k == "seed" else r[k])
                        for k in ("seed", "meals", "decisions", "lives", "cpu_s")}
                       for r in res]}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / DATA_NAME, summary=json.dumps(summ), **cat)
    return summ


# ── the judge (simulator truth; never an input to him) ──────────────────────
def _circ(a, b, n):
    d = np.abs(a - b) % n
    return np.minimum(d, n - d)


def judge_tables(x: np.ndarray, foods: np.ndarray):
    """Per look and food: its ray, and whether it is uneaten AND in sight."""
    dist = x[:, 0::2].astype(np.float32) * RAY_RANGE              # (n, N)
    b, d, un = foods[..., 0], foods[..., 1], foods[..., 2]
    ok = ~np.isnan(b)
    kf = np.where(ok, np.rint(np.nan_to_num(b) / (2 * math.pi) * N), 0).astype(int) % N
    sight = np.zeros(b.shape, bool)
    for off in range(-SIGHT_RAYS, SIGHT_RAYS + 1):
        rd = np.take_along_axis(dist, (kf + off) % N, axis=1)
        sight |= np.abs(rd - np.nan_to_num(d, nan=1e9)) < DIST_TOL
    return dist, kf, np.nan_to_num(d, nan=1e9), sight & ok & (un > 0.5)


def hits(peak: np.ndarray, dist, kf, d, vis, tol=HIT_RAYS, stride=1) -> np.ndarray:
    """peak: ray index on the 128 row. HIT per look (looks with any vis)."""
    pd = dist[np.arange(len(peak)), peak][:, None]
    near = _circ(peak[:, None], kf, N) <= tol
    if stride > 1:   # the coarse eye: the ray nearest the bearing
        near = (np.rint(kf / stride).astype(int) * stride) % N == peak[:, None]
    return (vis & near & (np.abs(pd - d) < DIST_TOL)).any(1)


# ── his part: the ring ──────────────────────────────────────────────────────
def build_net(n_rays: int, init: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(1000 + init)
    k = 5 if n_rays >= 64 else 3
    net = nn.Sequential(
        nn.Conv1d(2, 32, 1), nn.ReLU(),
        nn.Conv1d(32, 32, 1), nn.ReLU(),
        nn.Conv1d(32, 16, k, padding=k // 2, padding_mode="circular"), nn.ReLU(),
        nn.Conv1d(16, 1, 1))
    with torch.no_grad():
        net[-1].bias.fill_(-4.0)
    return net


def _as_rays(x: np.ndarray, stride: int):
    import torch
    r = x.reshape(len(x), N, 2)[:, ::stride].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(r.transpose(0, 2, 1)))


def scores(net, x, stride=1) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.cat([net(_as_rays(x[i:i + 8192], stride)).squeeze(1)
                          for i in range(0, len(x), 8192)]).numpy()


def train_taste(x, y, init: int, stride=1, steps=3000, batch=256):
    """Pooled logit = logsumexp over rays; balanced batches; BCE on the meal."""
    import torch
    net = build_net(N // stride, init)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.RandomState(50 + init)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    X = _as_rays(x, stride)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch // 2)])
        t = torch.zeros(batch)
        t[:batch // 2] = 1.0
        loss = lossf(torch.logsumexp(net(X[idx]).squeeze(1), 1), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def train_told(x, target_ray, init: int, steps=3000, batch=256):
    """CEILING, INELIGIBLE: the food's ray is the label (cross-entropy)."""
    import torch
    net = build_net(N, init)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.RandomState(70 + init)
    X, T = _as_rays(x, 1), torch.from_numpy(target_ray.astype(np.int64))
    for _ in range(steps):
        idx = rng.randint(len(x), size=batch)
        loss = torch.nn.functional.cross_entropy(net(X[idx]).squeeze(1), T[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


# ── fit + judge + rule ──────────────────────────────────────────────────────
def read(path=None, train=TRAIN_SEEDS, judge=JUDGE_SEEDS, steps=3000,
         enforce=True, keep_nets=False, only=None) -> dict:
    import torch
    torch.set_num_threads(2)
    z = np.load(path or OUT / DATA_NAME)
    x, foods, ttm, seed = z["x"], z["foods"], z["ttm"], z["seed"]
    summ = json.loads(str(z["summary"]))
    tr, ju = np.isin(seed, train), np.isin(seed, judge)
    n_meals = sum(w["meals"] for w in summ["worlds"] if w["seed"] in train)
    xt, yt = x[tr], ttm[tr] < H
    dist_t, kf_t, d_t, vis_t = judge_tables(xt, foods[tr])
    xj, sj = x[ju], seed[ju]
    dist, kf, d, vis = judge_tables(xj, foods[ju])
    el = vis.any(1)
    xj, sj, dist, kf, d, vis = xj[el], sj[el], dist[el], kf[el], d[el], vis[el]
    dmin = np.where(vis, d, 1e9).min(1)
    worlds = [s for s in judge if (sj == s).sum() > 0]

    def per_world(h):
        return np.array([h[sj == s].mean() for s in worlds])

    arms, bands = {}, {}

    def enter(name, hit_list):
        hm = np.mean(hit_list, axis=0)                 # per look, over inits
        arms[name] = {"pooled": float(hm.mean()), "world": per_world(hm)}
        bands[name] = [float(hm[(dmin >= a) & (dmin < b)].mean())
                       if ((dmin >= a) & (dmin < b)).any() else None
                       for a, b in ((0, 2), (2, 4), (4, 9))]

    rng = np.random.RandomState(7)
    yshuf = rng.permutation(yt)
    # told: training looks with an uneaten food in sight; label = nearest one
    has = vis_t.any(1)
    tgt = kf_t[has][np.arange(int(has.sum())), np.where(vis_t[has], d_t[has], 1e9).argmin(1)]
    out_nets = {}
    for name, fit in (
            ("taste128", lambda i: train_taste(xt, yt, i, steps=steps)),
            ("shuffled", lambda i: train_taste(xt, yshuf, i, steps=steps)),
            ("untrained", lambda i: build_net(N, i)),
            ("told", lambda i: train_told(xt[has], tgt, i, steps=steps))):
        if only and name not in only:
            continue
        nets = [fit(i) for i in INITS]
        out_nets[name] = nets
        enter(name, [hits(scores(n_, xj).argmax(1), dist, kf, d, vis) for n_ in nets])
    if only:
        return {"_nets": out_nets}
    enter("nearest", [hits(dist.argmin(1), dist, kf, d, vis)])
    enter("taste16", [hits(scores(train_taste(xt, yt, i, stride=8, steps=steps), xj, 8)
                           .argmax(1) * 8, dist, kf, d, vis, stride=8) for i in INITS])

    t, P = arms["taste128"], {k: v["pooled"] for k, v in arms.items()}
    ahead = {o: int((t["world"] > arms[o]["world"]).sum()) for o in ("shuffled", "untrained")}
    need = min(16, len(worlds)) if not enforce else 16
    clauses = {
        "hit>=.50": P["taste128"] >= 0.50,
        "+.25 over shuffled": P["taste128"] - P["shuffled"] >= 0.25,
        "+.25 over untrained": P["taste128"] - P["untrained"] >= 0.25,
        "+.25 over nearest": P["taste128"] - P["nearest"] >= 0.25,
        "ahead of shuffled >=16": ahead["shuffled"] >= need,
        "ahead of untrained >=16": ahead["untrained"] >= need}
    void = []
    if P["told"] < 0.80:
        void.append(f"told {P['told']:.3f} < .80")
    if n_meals < 60:
        void.append(f"{n_meals} training meals < 60")
    if len(xj) < 500:
        void.append(f"{len(xj)} judge looks < 500")
    verdict = ("VOID" if void else "ALIVE" if all(clauses.values()) else "DEAD")
    res = {"verdict": verdict, "void": void, "clauses": clauses, "pooled": P,
           "ahead_of_20": ahead, "worlds": len(worlds), "judge_looks": int(len(xj)),
           "train_looks": int(tr.sum()), "train_pos": int(yt.sum()),
           "train_meals": int(n_meals),
           "sd_world": {k: float(v["world"].std()) for k, v in arms.items()},
           "bands_0-2_2-4_4-8m": bands,
           "band_looks": [int(((dmin >= a) & (dmin < b)).sum()) for a, b in ((0, 2), (2, 4), (4, 9))]}
    if keep_nets:
        res["_nets"] = out_nets
    return res


REFLEX_SEEDS = (1301, 1302, 1303, 1304, 1305)
REFLEX_DEC = 3600


def reflex_cell(arm: str, seed: int, net=None, n_dec: int = REFLEX_DEC) -> dict:
    import torch
    from experiments.world_card import Held
    from experiments.handed_sense import HandedSense
    w = make_env(seed)
    pol = Held(seed, 8)
    hs = HandedSense(w.env, "food", "cossin") if arm == "told-reflex" else None
    o, info = w.reset(seed=seed)
    pol.new_life()
    m0, t0 = int(info["meals"]), time.process_time()
    for _ in range(n_dec):
        if arm == "held8":
            a = pol.act(o)
        else:
            a = np.zeros(8, np.float32)
            if hs is not None:
                c, s_, _d = hs.extra()
                a[6], a[7] = c, s_
            else:
                with torch.no_grad():
                    k = int(net(_as_rays(o[None, :2 * N], 1)).squeeze().argmax())
                a[6], a[7] = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
        o, _r, term, trunc, info = w.step(a)
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
    return {"arm": arm, "seed": seed, "meals": int(info["meals"]) - m0,
            "lives": len(w.lives), "cpu_s": round(time.process_time() - t0, 1)}


def reflex(path=None) -> dict:
    import torch
    torch.set_num_threads(1)
    r = read(path, keep_nets=True, only=("taste128", "shuffled"))
    nets = {"taste-reflex": r["_nets"]["taste128"][0],
            "shuf-reflex": r["_nets"]["shuffled"][0], "held8": None, "told-reflex": None}
    rows = [reflex_cell(a, s, nets[a]) for s in REFLEX_SEEDS for a in nets]
    M = {a: [x["meals"] for x in rows if x["arm"] == a] for a in nets}
    t = M["taste-reflex"]
    cl = {}
    for o_ in ("held8", "shuf-reflex"):
        cl[f"worlds over {o_} >=4"] = sum(a > b for a, b in zip(t, M[o_])) >= 4
        cl[f"2x {o_}"] = sum(t) >= 2 * sum(M[o_]) and sum(t) > 0
    return {"verdict": "ALIVE-2" if all(cl.values()) else "DEAD-2", "clauses": cl,
            "meals": M, "totals": {a: sum(v) for a, v in M.items()},
            "cpu_s": sum(x["cpu_s"] for x in rows)}


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.taste_eye import collect_all
r = collect_all(os.environ["JACK_OUT"])
print("RESULT " + json.dumps(r), flush=True)
print("DONE", flush=True)
'''


def kernel() -> dict:
    from experiments.gpu import build_job, submit
    sub = submit(build_job(_JOB.replace("__PINS__", repr(KERNEL_PINS))),
                 prefer="kaggle-cpu", est_hours=0.9, timeout_s=3 * 3600,
                 fetch=[DATA_NAME])
    if not sub.ok:
        raise RuntimeError(f"taste_eye kernel: {sub.message}")
    src = sub.artifacts.get(DATA_NAME)
    if not src:
        raise RuntimeError(f"no {DATA_NAME} in {sorted(sub.artifacts)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / DATA_NAME).write_bytes(Path(src).read_bytes())
    print("data home:", OUT / DATA_NAME, flush=True)
    return {"data": str(OUT / DATA_NAME)}


def main(argv):
    cmd = argv[0] if argv else "read"
    if cmd == "smoke":
        d = Path("/data/tmp/taste_eye_smoke")
        s = collect_all(d, train=(153,), judge=(154,), train_dec=3000,
                        judge_dec=1500, workers=2)
        print(json.dumps(s))
        print(json.dumps(read(d / DATA_NAME, train=(153,), judge=(154,),
                              steps=200, enforce=False), indent=1, default=str))
    elif cmd == "kernel":
        print(json.dumps(kernel()))
    elif cmd == "read":
        r = read(argv[1] if len(argv) > 1 else None)
        print(json.dumps(r, indent=1, default=str))
    elif cmd == "reflex":
        print(json.dumps(reflex(argv[1] if len(argv) > 1 else None), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
