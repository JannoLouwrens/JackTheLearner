"""J24 (2), box proof 1: PUBLISHED_PRIORS §28's toy, with the REAL geom ids of
worlds 941-943, 5 seeds, 200k steps, and the kind look as a third arm.

Not a learner in his world, not a ledger spec: a supervised known-answer
check with clean labels. Question (§28 (c)): can brain B's own 2x256 MLP read
"a ray hit food" from (i) a class handed one-hot, (ii) the kind paint's 5
channels, (iii) W0's scalar per-geom code?

=============================================================================
FROZEN in this commit, before any of its runs (11-26). Pass line: ORDERS J24
(2) — "kind look >= 99% recall within 10x one-hot's steps AND scalar code
< 50%".
=============================================================================
Worlds  941, 942, 943 as J24 builds them (the pilot's world). From each REAL
        world: the geoms a ray can hit (group != 3, i.e. not his body), the
        three floor-food geom ids, each geom's W0 code `_tex[gid]`, and each
        geom's kind from `fast_sim.kind_table` (the table paint="kind" uses).
Task    the scratch toy's (`/data/jack-research/scratch_sec28C/code/toy.py`),
        unchanged: R rays; each ray hits a uniformly drawn hittable geom or
        misses (one extra "miss" id, plus 30% of rays forced to miss); the
        distance number is uniform noise; label = any of the first min(R, 4)
        rays hit a floor food. MLP 2x256 ReLU, Adam 3e-4, batch 256, inputs
        standardised on a 200k sample, test set 20k, read every 250 steps;
        a run stops early at recall >= 99% and accuracy >= 99.5%.
Arms    onehot  [d, one-hot over (food, water, other, miss)]  — class handed
        kind    [d, food, water, obstacle, wall, panel], zeros on a miss
        code    [d, W0's scalar code], 0 on a miss            — today's row
R       128 (the eye he has; THE READ) and 1 (the bare slot question;
        reported beside it, decides nothing).
Seeds   torch seeds 0..4 per (world, arm, R); cap 200,000 steps.
Read    per world, over its 5 seeds, at R = 128:
          K: on EVERY seed the kind arm reaches 99% recall within 10 x the
             steps the onehot arm took on the same seed;
          C: the code arm's MEAN final food recall < 50%.
        PASS = K and C in all three worlds. Per-seed numbers are printed.
        A miss is a miss: INBOX line, no learner seed. No extension.
Cost    timed on the box first (11-26 07:21, load 6): 21 CPU-ms a step at
        R = 128, 6.7 at R = 1; the kind arm hit 99% at step 250. The 15
        full-length code runs at R = 128 alone are ~63,000 CPU-s — over the
        box's day. So: ONE Kaggle CPU kernel, 4 processes, one torch thread
        each (the box timing used 2; thread count changes no number that is
        read), R = 128 runs first; est <= 5 h.

    python -m experiments.j24_toy time
    python -m experiments.j24_toy run [R ...]     # resumable, one JSON a run
    python -m experiments.j24_toy shard I N       # every N-th run from the I-th
    python -m experiments.j24_toy kernel          # box: submit, wait, write OUT
    python -m experiments.j24_toy score
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("JACK_TOY_OUT", "/data/jack-logs/j24/toy"))
THREADS = int(os.environ.get("JACK_TOY_THREADS", "2"))
KERNEL_PINS = ["gymnasium==1.4.0", "mujoco==3.17.0"]
REMOTE_TIMEOUT_S = int(7.0 * 3600)
WORLDS = (941, 942, 943)
ARMS = ("onehot", "kind", "code")
RS = (128, 1)
SEEDS = (0, 1, 2, 3, 4)
STEPS = 200_000


def world_tables(seed: int) -> dict:
    """Real ids: hittable geoms, their codes, kinds and which are floor food."""
    from experiments import fast_sim as FS
    from experiments.jack_env import JackEnv
    from experiments.tests.test_jack_env_paint_cases import pilot_kw
    e = JackEnv(seed=seed, eye="row128", paint="kind", **pilot_kw())
    w, m = e.w, e.w.model
    hit = [g for g in range(m.ngeom) if int(m.geom_group[g]) != 3]
    foods = sorted(int(w.drives._food[n][0]) for n in e.floor_foods
                   if n in w.drives._food)
    return {"seed": seed, "hit": hit, "foods": foods,
            "code": [float(w._tex[g]) for g in hit],
            "kind": [int(w._kind_of[g]) for g in hit],
            "n_kinds": FS.N_KINDS, "food_kind": FS.KINDS.index("food"),
            "water_kind": FS.KINDS.index("water")}


def run_one(tab: dict, arm: str, R: int, seed: int, steps: int = STEPS) -> dict:
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    N = len(tab["hit"])                       # index N = miss
    isfood = torch.zeros(N + 1)
    for i, g in enumerate(tab["hit"]):
        if g in tab["foods"]:
            isfood[i] = 1
    codes = torch.tensor(tab["code"] + [0.0])
    kind = torch.tensor(tab["kind"] + [0])
    K = tab["n_kinds"]
    kind1h = torch.nn.functional.one_hot(kind, K).float()
    kind1h[N] = 0.0                           # a miss: all channels zero
    cls = torch.full((N + 1,), 2)             # handed class: food/water/other/miss
    cls[kind == tab["food_kind"]] = 0
    cls[kind == tab["water_kind"]] = 1
    cls[N] = 3
    cls1h = torch.nn.functional.one_hot(cls, 4).float()

    def sample(B):
        ids = torch.randint(0, N + 1, (B, R))
        ids[torch.rand(B, R) < 0.3] = N
        return ids

    def feat(ids):
        d = torch.rand(ids.shape).unsqueeze(-1)
        if arm == "code":
            x = torch.cat([d, codes[ids].unsqueeze(-1)], -1)
        elif arm == "kind":
            x = torch.cat([d, kind1h[ids]], -1)
        else:
            x = torch.cat([d, cls1h[ids]], -1)
        return x.flatten(1)

    def lab(ids):
        return (isfood[ids[:, :min(R, 4)]].sum(1) > 0).float()

    x0 = feat(sample(200_000 if R == 1 else 20_000))
    mu, sd = x0.mean(0), x0.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(x0.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    te = sample(20_000)
    xt, yt = (feat(te) - mu) / sd, lab(te)
    hit, acc, rec, s = {}, 0.0, 0.0, 0
    t0 = time.process_time()
    for s in range(1, steps + 1):
        b = sample(256)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net((feat(b) - mu) / sd).squeeze(1), lab(b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 250 == 0:
            with torch.no_grad():
                p = (net(xt).squeeze(1) > 0).float()
                acc = (p == yt).float().mean().item()
                rec = (p[yt == 1] == 1).float().mean().item()
            for th in (0.5, 0.9, 0.99):
                if rec >= th and th not in hit:
                    hit[th] = s
            if rec >= 0.99 and acc >= 0.995:
                break
    return {"world": tab["seed"], "arm": arm, "R": R, "seed": seed, "steps": s,
            "acc": round(acc, 4), "recall": round(rec, 4),
            "base_rate": round(float(yt.mean()), 4),
            "steps_to": {str(k): v for k, v in hit.items()},
            "cpu_s": round(time.process_time() - t0, 1)}


def _file(world, arm, R, seed) -> Path:
    return OUT / f"w{world}_{arm}_R{R}_s{seed}.json"


def tasks() -> list:
    """Every run, the long ones (code, R = 128) first."""
    order = [(R, arm) for R in RS for arm in ("code", "kind", "onehot")]
    return [(R, arm, w, s) for R, arm in order for s in SEEDS for w in WORLDS]


def shard(i: int, n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = {w: world_tables(w) for w in WORLDS}
    for R, arm, w, s in tasks()[i::n]:
        f = _file(w, arm, R, s)
        if f.exists():
            continue
        rec = run_one(tabs[w], arm, R, s)
        f.write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)
    print(f"SHARD {i} DONE", flush=True)


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import glob, json, os
os.chdir("/tmp/jack")
out = os.environ["JACK_OUT"]
env = dict(os.environ, JACK_TOY_OUT=os.path.join(out, "toy"), JACK_TOY_THREADS="1")
ps = [_s.Popen([_y.executable, "-m", "experiments.j24_toy", "shard", str(i), "4"],
               env=env, stdout=open(os.path.join(out, f"shard{i}.log"), "w"),
               stderr=_s.STDOUT) for i in range(4)]
rcs = [p.wait() for p in ps]
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "toy", "*.json")))]
tails = {i: open(os.path.join(out, f"shard{i}.log")).read()[-600:] for i in range(4)}
json.dump({"rcs": rcs, "runs": runs, "tails": tails},
          open(os.path.join(out, "j24_toy.json"), "w"))
print("DONE", rcs, len(runs), flush=True)
'''


def kernel() -> dict:
    """One Kaggle CPU kernel for all 90 runs; each run's JSON is written into
    OUT as `run` would have, plus the kernel's whole answer."""
    from experiments.gpu import build_job, result_json, submit
    body = _JOB.replace("__PINS__", repr(KERNEL_PINS))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=5.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["j24_toy.json"])
    if not sub.ok:
        raise RuntimeError(f"j24_toy kernel: {sub.message}")
    r = result_json(sub, "j24_toy.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernel.json").write_text(json.dumps(r) + "\n")
    for rec in r["runs"]:
        _file(rec["world"], rec["arm"], rec["R"], rec["seed"]).write_text(json.dumps(rec))
    print(f"HOME rcs {r['rcs']} runs {len(r['runs'])} of {len(tasks())}", flush=True)
    return r


def run(rs=RS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = {w: world_tables(w) for w in WORLDS}
    for w in WORLDS:
        t = tabs[w]
        print(f"world {w}: {len(t['hit'])} hittable geoms, foods {t['foods']} codes "
              f"{[round(t['code'][t['hit'].index(g)], 2) for g in t['foods']]}", flush=True)
    for R in rs:
        for arm in ARMS:                      # cheap arms first
            for w in WORLDS:
                for s in SEEDS:
                    f = _file(w, arm, R, s)
                    if f.exists():
                        continue
                    rec = run_one(tabs[w], arm, R, s)
                    f.write_text(json.dumps(rec))
                    print(json.dumps(rec), flush=True)
    print("DONE", flush=True)


def score() -> dict:
    out = {}
    for R in RS:
        res = {}
        for w in WORLDS:
            cell = {a: [json.loads(_file(w, a, R, s).read_text())
                        for s in SEEDS if _file(w, a, R, s).exists()] for a in ARMS}
            if any(len(v) < len(SEEDS) for v in cell.values()):
                res[w] = {"incomplete": {a: len(v) for a, v in cell.items()}}
                continue
            oh = [r["steps_to"].get("0.99") for r in cell["onehot"]]
            kd = [r["steps_to"].get("0.99") for r in cell["kind"]]
            code = [r["recall"] for r in cell["code"]]
            Kc = all(o is not None and k is not None and k <= 10 * o
                     for o, k in zip(oh, kd))
            Cc = float(np.mean(code)) < 0.5
            res[w] = {"onehot_steps_to_99": oh, "kind_steps_to_99": kd,
                      "code_recall": code, "code_recall_mean": round(float(np.mean(code)), 4),
                      "code_steps": [r["steps"] for r in cell["code"]],
                      "K": Kc, "C": Cc}
        out[f"R{R}"] = res
    r128 = out["R128"]
    if any("incomplete" in v for v in r128.values()):
        out["verdict"] = "INCOMPLETE"
    else:
        out["verdict"] = "PASS" if all(v["K"] and v["C"] for v in r128.values()) else "MISS"
    return out


def main(argv):
    if argv[:1] == ["time"]:
        tab = world_tables(941)
        for arm, R in (("code", 128), ("code", 1), ("kind", 128)):
            r = run_one(tab, arm, R, 0, steps=1000)
            print(arm, R, r, f"-> {r['cpu_s'] / r['steps'] * 1000:.1f} CPU-ms a step",
                  flush=True)
        return 0
    if argv[:1] == ["run"]:
        run(tuple(int(x) for x in argv[1:]) or RS)
        return 0
    if argv[:1] == ["shard"]:
        shard(int(argv[1]), int(argv[2]))
        return 0
    if argv[:1] == ["kernel"]:
        kernel()
        print(json.dumps(score(), indent=1))
        return 0
    if argv[:1] == ["score"]:
        print(json.dumps(score(), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
