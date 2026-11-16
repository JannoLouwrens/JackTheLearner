"""brain_b.py — Brain B (the replay learner) from published code, reproducing
published numbers before it is trusted. ORDERS J3; PLAN stage 1, part "His
brain", step 2->3 (combine, reproduce). Prior: PUBLISHED_PRIORS.md §5.

NOT A REWRITE. Brain B *is* SBX's `SAC` (sbx-rl 0.28.0) with
`policy_kwargs=dict(layer_norm=True)`. This file only builds it, runs §5's
known-answer rows and writes their numbers. No ledger row (ORDERS J3: the
table goes to `docs/research/KNOWN_ANSWERS.md`).

Environment: `/data/venvs/jack-std` (Python 3.12.13), built 11-16 with §5's
pins exactly: stable-baselines3==2.9.2, sb3-contrib==2.9.2, sbx-rl==0.28.0,
rl-zoo3==2.9.2, gymnasium==1.4.0, mujoco==3.17.0, dm-control==1.0.50,
shimmy==2.0.1, pink-noise-rl==2.0.1 (+ torch 2.15.0+cpu, jax/jaxlib 0.11.2,
flax 0.12.12 as resolved). The project venv is not touched. On Kaggle the
pins row 1 imports are pip-installed over Kaggle's image (Python 3.13 there,
3.12 here — a stated difference); each cell records the versions it ran with.

Sources
- SAC: Haarnoja et al. 2018 (arXiv 1801.01290, 1812.05905); code SB3
  `stable_baselines3.SAC` (PyTorch) and SBX `sbx.SAC` (JAX), both DLR-RM.
- Settings: rl-zoo3 2.9.2 `hyperparams/sac.yml` entry `Pendulum-v1`:
  n_timesteps 20000, MlpPolicy, learning_rate 1e-3; all else library default.
- Reference: Open RL Benchmark run store, SB3 SAC Pendulum-v1, 13 seeds,
  mean return of all 100 training episodes -298.4 +- 10.3 (§5 row 1).

=============================================================================
THE RULE — row 1, committed before any run (11-16, this commit).
=============================================================================
Task       gymnasium 1.4.0 `Pendulum-v1`, default (200-step time limit =
           truncated), wrapped in SB3 `Monitor`.
Steps      20,000 environment steps = exactly 100 training episodes.
Metric     per seed: mean of the returns of ALL 100 training episodes (as the
           reference states it — training episodes, exploration included, not
           a deterministic evaluation).
Seeds      0, 1, 2, 3, 4 (5 seeds; `seed=` given to the algorithm, which seeds
           env, torch/jax and numpy).
Arms       lib   SB3 SAC (PyTorch), Zoo entry unmodified: lr 1e-3, every other
                 setting the library default (buffer 1e6, batch 256,
                 learning_starts 100, tau 0.005, gamma 0.99, train_freq 1,
                 gradient_steps 1, ent_coef "auto", net_arch [256, 256]).
           ours  SBX SAC, the same Zoo entry, plus layer_norm=True.
           ctrl  CONTROL THAT MUST FAIL: SB3 SAC with learning_starts = 20,001,
                 i.e. uniform random actions for all 100 episodes and no
                 gradient step — the same plumbing with the learning removed.
Accept     an arm REPRODUCES iff its 5-seed mean lies in [-314.7, -282.1]
           (§5: -298.4 +- 1.58 x 10.3). `lib` is run first; `ours` counts only
           if `lib` reproduces (otherwise the apparatus is broken, not the
           arm). The run is VOID if `ctrl`'s mean lies inside the band or
           above -282.1.
A miss means the build is broken (ORDERS J3): fix it, do not go on to row 3.

Ours beside theirs (every setting of `ours` that differs from the reference run)
| setting        | reference (SB3 SAC, Zoo) | ours (Brain B)          |
| library        | PyTorch SB3 2.x          | SBX 0.28.0 (JAX)        |
| LayerNorm      | none                     | layer_norm=True         |
| everything else| Zoo Pendulum-v1 entry    | identical               |
§5 "Take as-is" lists B's in-world settings (lr 3e-4, 5-10k warm-up,
observations normalised, 1-2 updates per decision). Those are NOT used on
row 1: the row reproduces the Zoo entry, and §5 says "SAC, Zoo entry (lr
1e-3) ... then ours". The in-world settings are J3's later step and stage 2's.

Threads: one per cell (torch.set_num_threads(1); XLA single-threaded) so the
box rule (<= 2-3 workers, nice 19) and Kaggle's 4 cores are both respected.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/data/jack-logs/brain_b")
STD_PY = "/data/venvs/jack-std/bin/python"

ROW1_SEEDS = (0, 1, 2, 3, 4)
ROW1_STEPS = 20_000
ROW1_BAND = (-314.7, -282.1)
ROW1_ARMS = ("lib", "ours", "ctrl")
PINS = ["stable-baselines3==2.9.2", "sb3-contrib==2.9.2", "sbx-rl==0.28.0",
        "rl-zoo3==2.9.2", "gymnasium==1.4.0", "mujoco==3.17.0",
        "dm-control==1.0.50", "shimmy==2.0.1", "pink-noise-rl==2.0.1"]
# Row 1 on Kaggle installs only what row 1 imports: Kaggle's image is Python
# 3.13 and dm-control's `labmaze` has no 3.13 wheel (kernel jack-cpu-1794791632
# died in pip on it, 11-16 01:14, before any cell ran). Same pinned versions.
ROW1_KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0"]
CHILD_TIMEOUT_S = int(5.0 * 3600)
REMOTE_TIMEOUT_S = int(6.0 * 3600)


def _single_thread():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def make_row1(arm: str, seed: int, steps: int = ROW1_STEPS):
    """Build the arm's model on a Monitor-wrapped Pendulum-v1."""
    _single_thread()
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(gym.make("Pendulum-v1"))
    if arm == "lib":
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed, device="cpu")
    if arm == "ours":
        from sbx import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed,
                   policy_kwargs=dict(layer_norm=True))
    if arm == "ctrl":
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import SAC
        return SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed, device="cpu",
                   learning_starts=steps + 1)
    raise ValueError(f"arm {arm!r} not in {ROW1_ARMS}")


def versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    out = {"python": sys.version.split()[0]}
    for p in ["stable-baselines3", "sbx-rl", "gymnasium", "torch", "jax",
              "jaxlib", "flax", "numpy"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def run_row1_cell(arm: str, seed: int, steps: int = ROW1_STEPS) -> dict:
    t0 = time.time()
    model = make_row1(arm, seed, steps)
    model.learn(total_timesteps=steps)
    env = model.get_env().envs[0]
    rets = [float(r) for r in env.get_episode_rewards()]
    return {"row": 1, "arm": arm, "seed": seed, "steps": steps,
            "episodes": len(rets), "mean_return": sum(rets) / max(1, len(rets)),
            "returns": rets, "wall_s": round(time.time() - t0, 1),
            "versions": versions()}


def verdict_row1(cells: list) -> dict:
    lo, hi = ROW1_BAND
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        seeds = sorted(c["seed"] for c in cs)
        m = sum(c["mean_return"] for c in cs) / len(cs)
        sd = (sum((c["mean_return"] - m) ** 2 for c in cs) / max(1, len(cs) - 1)) ** .5
        complete = (seeds == list(ROW1_SEEDS)
                    and all(c["episodes"] == ROW1_STEPS // 200 for c in cs))
        out[arm] = {"seeds": seeds, "mean": round(m, 1), "sd": round(sd, 1),
                    "per_seed": [round(c["mean_return"], 1) for c in sorted(cs, key=lambda c: c["seed"])],
                    "in_band": lo <= m <= hi, "complete": complete}
    ctrl = out.get("ctrl")
    void = ctrl is None or not ctrl["complete"] or ctrl["mean"] >= lo
    lib_ok = "lib" in out and out["lib"]["complete"] and out["lib"]["in_band"]
    ours_ok = lib_ok and "ours" in out and out["ours"]["complete"] and out["ours"]["in_band"]
    out["verdict"] = ("VOID" if void else
                      ("lib REPRODUCES; ours " + ("REPRODUCES" if ours_ok else "MISSES"))
                      if lib_ok else "lib MISSES (apparatus broken)")
    return out


def run_parallel(cells: list, outdir, workers: int = 4) -> list:
    """Run (arm, seed) cells as child processes, `workers` at a time."""
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pending = list(cells)
    running, done = [], []
    while pending or running:
        while pending and len(running) < workers:
            arm, seed = pending.pop(0)
            f = outdir / f"row1_{arm}_s{seed}.json"
            p = subprocess.Popen([sys.executable, "-m", "experiments.brain_b", "cell",
                                  arm, str(seed), str(f)], cwd=REPO,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((arm, seed, f, p, time.time()))
        time.sleep(5)
        for item in list(running):
            arm, seed, f, p, t0 = item
            if p.poll() is None and time.time() - t0 < CHILD_TIMEOUT_S:
                continue
            if p.poll() is None:
                p.kill()
            err = p.stderr.read().decode()[-800:]
            running.remove(item)
            rec = (json.loads(f.read_text()) if f.exists()
                   else {"arm": arm, "seed": seed, "error": f"rc={p.returncode}: {err}"})
            done.append(rec)
            print(f"{arm} s{seed}: {rec.get('mean_return', rec.get('error'))}", flush=True)
    return done


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.brain_b import run_parallel, verdict_row1
r = run_parallel(__CELLS__, os.environ["JACK_OUT"])
ok = [c for c in r if "error" not in c]
json.dump({"cells": r, "verdict": verdict_row1(ok) if ok else None},
          open(os.path.join(os.environ["JACK_OUT"], "brain_b_row1.json"), "w"))
print("DONE", flush=True)
'''


def row1_cells() -> list:
    # lib first (the rule: the unmodified library is run first), then ours, ctrl
    return [(a, s) for a in ROW1_ARMS for s in ROW1_SEEDS]


def row1_kernel() -> dict:
    """Submit row 1 (3 arms x 5 seeds) as one Kaggle CPU kernel; write the
    cells and verdict to OUT/row1.json."""
    from experiments.gpu import build_job, result_json, submit
    body = _JOB.replace("__CELLS__", repr(row1_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b row1 kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "row1.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def main(argv):
    if len(argv) >= 5 and argv[1] == "cell":      # cell <arm> <seed> <out.json> [steps]
        steps = int(argv[5]) if len(argv) > 5 else ROW1_STEPS
        rec = run_row1_cell(argv[2], int(argv[3]), steps)
        Path(argv[4]).write_text(json.dumps(rec))
        return 0
    if len(argv) >= 2 and argv[1] == "row1-kernel":
        row1_kernel()
        return 0
    if len(argv) >= 2 and argv[1] == "smoke":     # smoke [steps]: all arms, seed 0
        steps = int(argv[2]) if len(argv) > 2 else 1000
        for arm in ROW1_ARMS:
            rec = run_row1_cell(arm, 0, steps)
            print(arm, rec["episodes"], round(rec["mean_return"], 1), rec["wall_s"], "s",
                  flush=True)
        print(json.dumps(versions()))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
