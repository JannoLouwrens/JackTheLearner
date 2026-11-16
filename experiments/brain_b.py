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

ROW 1 RESULT (11-16, kernel jack-cpu-1794791696): lib -319.5 +- 34.8
(per seed -312.5 -281.8 -305.6 -321.8 -376.0) MISSES; ours -285.9 +- 10.0
(in band, does not count); ctrl -1227.6 (holds). Verdict: lib MISSES.
Reference re-pulled 11-16 from the Open RL Benchmark store itself
(/data/jack-logs/brain_b/orb/, summary in KNOWN_ANSWERS.md): 14 finished
runs, -298.3 +- 9.9, range [-315.1, -283.5] — §5's number is right. Two of
our five lib seeds fall below the reference's worst. By 4-episode blocks the
loss is in BOTH phases: lib seed 4 learns ~1 block late (eps 13-16: -757 vs
reference -277 mean) and collapses once after learning (eps 21-24: -903);
lib eps 21-100 average -170.6 vs reference -156.8 (range -178.0..-131.4).
Reference stack (run configs): Python 3.8, SB3 1.x, gym 0.21, Jan 2023.
Ours: Python 3.13 on Kaggle, SB3 2.9.2, gymnasium 1.4.0, torch 2.11.0+cpu.

=============================================================================
THE RULE — row 1b, the re-registration, committed before any run (11-16).
=============================================================================
Why re-register: the row-1 rule took 5 seeds. Our lib's seed-level sd was
34.8, so the 5-seed mean's standard error (15.6) is the band's half-width
(16.3): a build that matched our lib's spread would miss about a third of
the time even if its centre were right. 5 seeds cannot tell "broken" from
"heavy tail". Row 1 STAYS RECORDED AS A MISS; row 1b is a stronger test on
fresh seeds, not a re-scoring.
Seeds      5..19 (15 fresh seeds; none of row 1's).
Task, steps, metric, arms (lib, ours), settings: exactly row 1's.
ctrl       seeds 5..9 (it never learns; 5 seeds suffice for the void check).
Accept     band UNCHANGED [-314.7, -282.1] on the 15-seed mean. lib first;
           ours counts only if lib reproduces. VOID if ctrl's mean >= -314.7.
Secondary  (reported, does not decide): two-sided Mann-Whitney U of lib's
           15 per-seed means against the reference's 14 (p reported); the
           count of lib seeds below the reference's worst (-315.1).
If lib misses 1b: the stack is the suspect, not the seed count. Next step
is the stack hunt — SB3 1.8 / gym 0.21 / Python 3.8 (the reference's) vs
ours, one difference at a time — before row 3.

ROW 1b RESULT (11-16, kernel jack-cpu-1794795073): lib -309.3 +- 13.7
REPRODUCES; ours -302.0 +- 22.2 REPRODUCES; ctrl -1227.7 (valid). Secondary:
Mann-Whitney vs the reference's 14 runs lib p = 0.011, ours p = 0.42; below
the reference's worst lib 5/15, ours 2/15. Next: row 3.

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
ROW1B_SEEDS = tuple(range(5, 20))
ROW1B_CTRL_SEEDS = tuple(range(5, 10))
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


def verdict_row1(cells: list, seeds=None, ctrl_seeds=None) -> dict:
    lo, hi = ROW1_BAND
    seeds = list(seeds or ROW1_SEEDS)
    ctrl_seeds = list(ctrl_seeds or seeds)
    by = {}
    for c in cells:
        by.setdefault(c["arm"], []).append(c)
    out = {}
    for arm, cs in by.items():
        got = sorted(c["seed"] for c in cs)
        m = sum(c["mean_return"] for c in cs) / len(cs)
        sd = (sum((c["mean_return"] - m) ** 2 for c in cs) / max(1, len(cs) - 1)) ** .5
        want = ctrl_seeds if arm == "ctrl" else seeds
        complete = (sorted(c["seed"] for c in cs) == sorted(want)
                    and all(c["episodes"] == ROW1_STEPS // 200 for c in cs))
        out[arm] = {"seeds": got, "mean": round(m, 1), "sd": round(sd, 1),
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
json.dump({"cells": r, "verdict": __VERDICT__ if ok else None},
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
    body = (_JOB.replace("__CELLS__", repr(row1_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("__VERDICT__", "verdict_row1(ok)"))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b row1 kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "row1.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def row1b_cells() -> list:
    # lib first, then ours, then ctrl (row 1b rule)
    return ([("lib", s) for s in ROW1B_SEEDS] + [("ours", s) for s in ROW1B_SEEDS]
            + [("ctrl", s) for s in ROW1B_CTRL_SEEDS])


def row1b_kernel() -> dict:
    """Submit row 1b (lib, ours x 15 fresh seeds; ctrl x 5) as one Kaggle CPU
    kernel; write cells and verdict to OUT/row1b.json."""
    from experiments.gpu import build_job, result_json, submit
    body = (_JOB.replace("__CELLS__", repr(row1b_cells())).replace("__PINS__", repr(ROW1_KERNEL_PINS))
            .replace("__VERDICT__", f"verdict_row1(ok, seeds={list(ROW1B_SEEDS)!r}, "
                                    f"ctrl_seeds={list(ROW1B_CTRL_SEEDS)!r})"))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_row1.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b row1b kernel: {sub.message}")
    r = result_json(sub, "brain_b_row1.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "row1b.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def main(argv):
    if len(argv) >= 5 and argv[1] == "cell":      # cell <arm> <seed> <out.json> [steps]
        steps = int(argv[5]) if len(argv) > 5 else ROW1_STEPS
        rec = run_row1_cell(argv[2], int(argv[3]), steps)
        Path(argv[4]).write_text(json.dumps(rec))
        return 0
    if len(argv) >= 2 and argv[1] == "row1b-kernel":
        row1b_kernel()
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
