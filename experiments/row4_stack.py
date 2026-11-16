"""Row 4c — the stack hunt for J3 row 4 (PPO, Zoo entry, MountainCarContinuous-v0).

Runs the Zoo's PPO MountainCarContinuous-v0 entry with whichever SB3/gym stack
the interpreter carries. The same file runs under both stacks (SB3's
make_vec_env / VecNormalize / PPO API is shared), so the STACK is the only
factor that differs between arms.

THE RULE — row 4c, committed before any run (11-16 ~10:35 UTC).
Why. Rows 4 (5 seeds) and 4b (15 fresh seeds) MISSED: 4 of 20 seeds solved,
then collapsed to saturated actions (returns -30..-100) late in the 20,000
steps. The public run store (wandb openrlbenchmark/sb3, 10 runs, re-read
11-16 through its GraphQL API) has finals 81.6 83.5 83.8 82.9 87.9 88.9 89.1
92.0 92.0 93.2 (mean 87.47) and NO collapse. Its runs' requirements.txt (run
2guwltm4): Python 3.9.16, stable-baselines3 1.7.0a12, rl-zoo3 1.7.0a12,
gym 0.21.0, torch 1.12.0, numpy 1.23.1. Ours: Python 3.12, SB3 2.9.2,
gymnasium 1.4.0, torch 2.15, numpy 2.5.
Source diff done first (11-16): gym 0.21 vs gymnasium 1.4
continuous_mountain_car.step is identical line for line (float32 state, same
clip, reward, 999-step limit); rl-zoo3 1.7.0's exp_manager copies the entry's
gamma into VecNormalize exactly as 2.9.2 does, and the 1.7.0 ppo.yml entry is
byte-equal in values; SB3 PPO.collect_rollouts (timeout bootstrap included)
and VecNormalize are behaviourally equal; the ONLY behavioural line found is
SB3 1.7's PPO.train calling policy.reset_noise(batch_size) per minibatch,
which draws torch RNG but changes no update (the rollout resamples the gSDE
matrix at its start in both versions). So the code says the stacks are the
same distribution; this run asks the machine.
Arms   r4pub   this file under /data/venvs/sb3-17: /usr/bin/python3.9 (3.9.25),
               stable-baselines3 1.7.0 (release; the store's 1.7.0a12 is not
               on PyPI), gym 0.21.0, torch 1.12.0 (CPU), numpy 1.23.1.
       r4pubctrl  CONTROL THAT MUST FAIL: r4pub with learning_rate 0.0.
Seeds  r4pub 0..19 (the labels rows 4 + 4b used; streams differ by stack, so
       seeds are labels, not pairings); r4pubctrl 0..4.
Task, steps (20,000), settings, metric: exactly row 4's (mean of
       ep_info_buffer at the end of learn(); Monitor inside VecNormalize, raw
       returns). Box, nice 19, 2 workers, one torch thread per cell.
Reading (decided now):
  A  r4pub collapses (metric < 60) on >= 3 of 20  -> the collapse is the
     published code's own tail, sampled by 20 seeds and missed by the store's
     10; our build is not broken. Row 4 is then re-registered as a
     same-against-same reproduction (ours vs the public stack run here, a
     rule written before it runs) — the band from 10 public finals is not
     re-used. No threshold of row 4 / 4b is moved; both stay MISSES.
  B  r4pub collapses on <= 1 of 20 AND its 20-seed mean lies in the k=15-style
     band for n=20, 87.5 +/- 3*4.2*sqrt(1/20+1/10) = [82.6, 92.4]  -> the
     stack matters although the source diff did not show where: bisect one
     factor at a time (SB3 1.8 on gym 0.21 / SB3 2.0 on gymnasium 0.28 /
     torch), rule first each time.
  C  anything else (2 collapses, or <= 1 with the mean outside) -> inconclusive;
     20 more r4pub seeds (20..39) under this same rule, pooled.
  Also reported: Fisher exact, ours 4/20 collapses vs r4pub k/20.
VOID unless r4pubctrl completes and every r4pubctrl seed is below 60.

Usage: python -m experiments.row4_stack run OUTDIR   (all cells, 2 workers)
       python -m experiments.row4_stack cell ARM SEED OUTDIR
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = 20_000
PUB_SEEDS = tuple(range(0, 20))
CTRL_SEEDS = tuple(range(0, 5))
BAND_N20 = (82.6, 92.4)
COLLAPSE = 60.0


def cells():
    return ([("r4pub", s) for s in PUB_SEEDS]
            + [("r4pubctrl", s) for s in CTRL_SEEDS])


def run_cell(arm: str, seed: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    t0 = time.time()
    env = VecNormalize(make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=seed),
                       gamma=0.9999)
    model = PPO("MlpPolicy", env, batch_size=256, n_steps=8, gamma=0.9999,
                learning_rate=(0.0 if arm == "r4pubctrl" else 7.77e-05),
                ent_coef=0.00429, clip_range=0.1, n_epochs=10, gae_lambda=0.9,
                max_grad_norm=5, vf_coef=0.19, use_sde=True,
                policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
                seed=seed, device="cpu")
    model.learn(total_timesteps=STEPS)
    last = [float(e["r"]) for e in model.ep_info_buffer]
    lens = [int(e["l"]) for e in model.ep_info_buffer]
    try:
        import gym as g
    except ImportError:
        import gymnasium as g
    import numpy
    return {"row": "4c", "arm": arm, "seed": seed, "steps": STEPS,
            "timesteps": int(model.num_timesteps), "last_n": len(last),
            "mean_return": sum(last) / max(1, len(last)), "last_returns": last,
            "last_lens": lens, "wall_s": round(time.time() - t0, 1),
            "versions": {"python": sys.version.split()[0], "sb3": sb3.__version__,
                         "gym": f"{g.__name__} {g.__version__}",
                         "torch": torch.__version__, "numpy": numpy.__version__}}


def verdict(recs: list, ours_collapses: int = 4, ours_n: int = 20) -> dict:
    pub = sorted((r for r in recs if r["arm"] == "r4pub"), key=lambda r: r["seed"])
    ctrl = sorted((r for r in recs if r["arm"] == "r4pubctrl"), key=lambda r: r["seed"])
    vals = [r["mean_return"] for r in pub]
    k = sum(v < COLLAPSE for v in vals)
    mean = sum(vals) / len(vals) if vals else float("nan")
    out = {"r4pub": {"seeds": [r["seed"] for r in pub], "per_seed": [round(v, 1) for v in vals],
                     "mean": round(mean, 1), "collapsed_lt60": k},
           "r4pubctrl": {"seeds": [r["seed"] for r in ctrl],
                         "per_seed": [round(r["mean_return"], 1) for r in ctrl]}}
    void = ([r["seed"] for r in ctrl] != list(CTRL_SEEDS)
            or any(r["mean_return"] >= COLLAPSE for r in ctrl))
    complete = [r["seed"] for r in pub] == list(PUB_SEEDS)
    if void or not complete:
        out["reading"] = "VOID" if void else "INCOMPLETE"
    elif k >= 3:
        out["reading"] = "A"
    elif k <= 1 and BAND_N20[0] <= mean <= BAND_N20[1]:
        out["reading"] = "B"
    else:
        out["reading"] = "C"
    try:
        from scipy.stats import fisher_exact
        out["fisher_two_sided_p"] = round(float(fisher_exact(
            [[ours_collapses, ours_n - ours_collapses], [k, len(vals) - k]])[1]), 4)
    except Exception as e:  # scipy absent in the py3.9 stack: computed at harvest
        out["fisher_two_sided_p"] = f"unavailable ({type(e).__name__})"
    return out


def run_all(outdir: str, workers: int = 2) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells() if not (out / f"{c[0]}_{c[1]}.json").exists()]
    running = []
    while todo or running:
        while todo and len(running) < workers:
            arm, seed = todo.pop(0)
            p = subprocess.Popen(["nice", "-n", "19", sys.executable, "-m",
                                  "experiments.row4_stack", "cell", arm, str(seed), str(out)])
            running.append((p, arm, seed))
            print(f"START {arm} {seed} pid {p.pid}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, arm, seed = item
            if p.poll() is not None:
                running.remove(item)
                print(f"END {arm} {seed} rc {p.returncode}", flush=True)
    recs = [json.loads(f.read_text()) for f in sorted(out.glob("r4pub*.json"))]
    v = verdict(recs)
    (out / "verdict.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if len(a) >= 2 and a[1] == "cell":
        rec = run_cell(a[2], int(a[3]))
        Path(a[4], f"{a[2]}_{a[3]}.json").write_text(json.dumps(rec))
        print(f"{a[2]} {a[3]} mean {rec['mean_return']:.1f} n {rec['last_n']} "
              f"wall {rec['wall_s']}s", flush=True)
    elif len(a) >= 3 and a[1] == "run":
        run_all(a[2])
    else:
        print(__doc__)
        sys.exit(2)
