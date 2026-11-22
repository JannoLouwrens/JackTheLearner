"""brain_b_dmc.py — Brain B's known-answer rows 5 and 6 (PUBLISHED_PRIORS §5):
plain SAC on proprioceptive DeepMind Control. ORDERS "Next up" item 2 (rule +
kernel bodies only; no run in the slot that wrote this). PLAN part "His brain",
step 3 (reproduce). Row 7 (recurrent SAC, Ni et al. 2022) is NOT here: it needs
a different learner (separate GRU-128 encoders) and gets its own rule.

NOT A REWRITE. Brain B is SBX `SAC` (sbx-rl 0.28.0); see `experiments/brain_b.py`.

Source of the reference (code read 11-18 at SonyResearch/simba@7d0358b1):
- `configs/agent/sac.yaml`: actor and critic = one `MLPBlock` of width 256 =
  Dense(256)-ReLU-Dense(256)-ReLU (2 x 256), orthogonal init sqrt(2); lr 3e-4
  (actor, critic, temperature); temperature init 1.0, target entropy
  -1.0 x action_dim; clipped double Q; tau 0.005; no observation normalisation.
- `configs/base.yaml`: updates_per_interaction_step 2; gamma from TD-MPC2's
  heuristic = 0.99 for 1000-frame episodes at repeat 2; n_step 1; evaluation
  every 25,000 interaction steps, 10 episodes, deterministic (temperature 0).
- `configs/env/dmc_em.yaml`: action_repeat 2, max_episode_steps 1000 frames,
  rescale_action [-1, 1], no_termination false, reward_scale 1.
- `configs/buffer/numpy_uniform.yaml`: 1e6, min_length 5,000 (uniform random
  actions until then), batch 256.
- env: `dm_control.suite.load(domain, task, task_kwargs={"random": seed})`,
  shimmy `DmControlCompatibilityV0`, `FlattenObservation`, `RescaleAction`,
  `TimeLimit(1000)`, then `RepeatAction(2)` (rewards summed over the 2 frames).
- numbers: `results/sac.csv` (10 seeds; frames on the step axis), read per
  seed at the logged step nearest the target; mean +- sample SD over seeds
  (`$N/control_arm_engineering.md`, which reproduces the paper's Table 12 SAC
  mean 679.42 by that procedure).

=============================================================================
THE RULE — rows 5 and 6, committed 2026-11-18 before any seed runs.
=============================================================================
| row | task (DMC proprio)       | decisions (frames) | published mean +- SD (10) | band (5-seed mean) |
|-----|--------------------------|--------------------|---------------------------|--------------------|
| 5   | cartpole-swingup         |  50,000 (100K)     | 853.0 +- 9.2              | 837.9 .. 868.1     |
| 6a  | cartpole-swingup_sparse  | 150,000 (300K)     | 776.5 +- 30.2             | 726.9 .. 826.1     |
| 6b  | reacher-easy             | 150,000 (300K)     | 968.0 +- 23.2             | 929.9 .. 1000      |
| 6c  | reacher-hard             | 150,000 (300K)     | 955.8 +- 30.5             | 905.7 .. 1000      |

Band = rows 4b's formula: mean +- 3 x SD x sqrt(1/k + 1/n), k = 5 ours, n = 10
reference; sqrt(0.3) = 0.548. Upper edges above 1000 are cut at the task max.
Metric      per seed: mean return of 10 deterministic evaluation episodes
            (actor mean through tanh) at the target decision, on a separate
            eval env (seed + 10,000); each episode = 500 decisions = 1000 frames.
            Reported, not judged: the same evaluation at 25,000-decision
            marks (the reference's cadence) and every training return.
Seeds       0..4 per arm per task.
Arms        lib    SBX SAC as close to the SimBa recipe as SBX's knobs allow:
                   net_arch [256, 256], lr 3e-4, buffer 1e6, batch 256,
                   learning_starts 5,000, tau 0.005, gamma 0.99, train_freq 1,
                   gradient_steps 2, ent_coef "auto" (init 1.0), target
                   entropy "auto" (= -action_dim). layer_norm False.
            ours   lib + layer_norm=True — Brain B as built (rows 1b, 3b).
            ctrl   CONTROL THAT MUST FAIL: lib with learning_starts past the
                   run (uniform random actions, no gradient step; the
                   evaluation plays the untrained actor's mean). Seeds 0..2.
Accept      a task REPRODUCES for an arm iff its 5-seed mean lies in the band
            and all 5 seeds completed. `lib` counts first; `ours` counts only
            if `lib` reproduces on that task (else the apparatus is broken).
            A task is VOID if any ctrl seed scores at or above the band's
            lower edge, or ctrl did not complete.
            Row 6 is judged per task; row 6 REPRODUCES only if all three do.
            Also reported, not judged (§5: "sparse near zero at 50K on most
            seeds"): swingup_sparse's evaluation at 25k/50k decisions.
Extension   ONE, stated now: a task that misses at 5 seeds runs seeds 5..14
            (fresh) and is judged once more at k = 15 with the formula's
            narrower band (row 5: 853.0 +- 11.3 = 841.7 .. 864.3). Nothing after.
Order       row 5 first (one Kaggle CPU session); row 6 only after row 5's
            `lib` reproduces — a miss on row 5 is fixed first, not stacked.

Settings that differ from the reference (ours beside theirs):
- library: SBX 0.28.0 JAX SAC vs SimBa's own JAX SAC.
- init: flax default (lecun-normal) vs orthogonal sqrt(2).
- actor log-std: clipped to [-20, 2] vs tanh-rescaled into [-10, 2].
- temperature lr: shared 3e-4 (SBX has one lr) — equal to theirs (3e-4).
- eval env seed: seed + 10,000 vs SimBa's eval env built from the same seed.
- Python 3.13 on Kaggle (dm-control installed without labmaze, which has no
  3.13 wheel and is only imported by the locomotion suite) vs theirs unstated.
- ours (only): LayerNorm in the critic.
A miss can mean a broken build OR one of these; it is written with the
candidates named, not explained away.

Price (box probe 11-18, load ~4, `probe 1500`, seed 0, NOT a reading): 1,000
learning decisions = 2,000 updates in 31 s (ours) / 40 s (lib) incl. JIT,
~15-20 ms/update; one eval episode 0.6 s. Row 5 = 13 cells, 10 learning at
~90k updates ~ 26-30 min each -> ~1.5-2 h on 4 workers (est 3 h). Row 6 =
~82 min per learning cell -> ~10 h for all three tasks: ONE KERNEL PER TASK
(~3.5 h each, est 5 h). Kaggle core speed vs box unmeasured for SBX.
Kaggle (pushed HEAD first), rank of "Next up" item 2:
  scripts/launch_detached.sh /data/jack-logs/brain_b/launch_row5.log
      $PY -m experiments.brain_b_dmc kernel 5
  ... kernel 6 cartpole-swingup_sparse | reacher-easy | reacher-hard
The rule's ONE extension (11-21, after row 5's lib missed at 5 seeds, 796.2):
  ... kernel 5 ext   -> lib + ours on EXT_SEEDS 5..14 (20 cells, no ctrl re-run),
      judged by `judge_ext` with row5_dmc.json's cells: seeds 0..14, k = 15,
      band 841.7..864.3, ctrl = the first kernel's seeds 0..2. Written into
      row5_dmc_ext.json. Code added only; the rule text above is unchanged.

THE CAUSE TWIN (ORDERS J19, ruling 81; 11-22, code added only, rule unchanged):
  ... twin cartpole-swingup_sparse | reacher-hard
  = the row's `lib` arm ONLY, the same seeds 0..4, the same 150,000 decisions,
  with TWO additions and nothing else: the model is saved at every evaluation
  mark (`twin_lib_<task>_s<seed>_<n>.msgpack`: actor + critic weights only,
  no optimiser, no target, no buffer -- `save_weights`/`load_weights`), and
  every 10,000 decisions one trace row holds the learner's last logged
  critic_loss, actor_loss (mean Q is read as -actor_loss, as j17_onset does),
  ent_coef and n_updates. Both additions only READ (test_twin_reads_only: a
  cell with them on and off gives identical evals and training returns).
  NOT a third try at the row: rows 5, 6a, 6c stay named MISSES whatever it
  scores; it has no verdict (`verdict` is None in its file) and no ctrl.
  Home: OUT/j19_twin_<task>/ (summary + 30 model files) for the box
  re-evaluation, whose reading is written into KNOWN_ANSWERS.md B-6 first.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from experiments.brain_b import _single_thread, versions

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/data/jack-logs/brain_b")
STD_PY = "/data/venvs/jack-std/bin/python"

REF_N = 10
K = 5
ROWS = {
    "5": {"cartpole-swingup": (853.0, 9.2)},
    "6": {"cartpole-swingup_sparse": (776.5, 30.2),
          "reacher-easy": (968.0, 23.2),
          "reacher-hard": (955.8, 30.5)},
}
STEPS = {"5": 50_000, "6": 150_000}
ARMS = ("lib", "ours", "ctrl")
SEEDS = (0, 1, 2, 3, 4)
EXT_SEEDS = tuple(range(5, 15))
CTRL_SEEDS = (0, 1, 2)
EVAL_EPISODES = 10
EVAL_EVERY = 25_000
EVAL_SEED_OFFSET = 10_000
ACTION_REPEAT = 2
MAX_FRAMES = 1000
LEARNING_STARTS = 5_000
# Kaggle's image is Python 3.13; dm-control's labmaze has no 3.13 wheel
# (brain_b.py, kernel jack-cpu-1794791632). suite imports without it (checked
# 11-18 on the box with labmaze import-blocked: cartpole-swingup loads).
KERNEL_PINS = ["stable-baselines3==2.9.2", "sbx-rl==0.28.0", "gymnasium==1.4.0",
               "mujoco==3.17.0", "shimmy==2.0.1", "dm-env", "dm-tree", "absl-py",
               "glfw", "lxml", "pyopengl", "pyparsing"]
KERNEL_NODEPS = ["dm-control==1.0.50"]
CHILD_TIMEOUT_S = int(8.0 * 3600)
REMOTE_TIMEOUT_S = int(9.5 * 3600)


def band(mean: float, sd: float, k: int = K, n: int = REF_N) -> tuple:
    h = 3 * sd * math.sqrt(1 / k + 1 / n)
    return (round(mean - h, 1), round(min(1000.0, mean + h), 1))


def make_env(task: str, seed: int):
    """The SimBa env stack, wrapper for wrapper (see the docstring)."""
    import gymnasium as gym
    from dm_control import suite
    from gymnasium import spaces
    from gymnasium.wrappers import FlattenObservation, RescaleAction, TimeLimit
    from shimmy import DmControlCompatibilityV0

    class RepeatAction(gym.Wrapper):          # SimBa scale_rl/envs/wrappers/repeat_action.py
        def __init__(self, env, k):
            super().__init__(env)
            self._k = k

        def step(self, action):
            total, info = 0.0, {}
            for _ in range(self._k):
                obs, r, term, trunc, i = self.env.step(action)
                total += float(r)
                info.update(i)
                if term or trunc:
                    break
            return obs, total, term, trunc, info

    domain, name = task.split("-")
    env = DmControlCompatibilityV0(suite.load(domain, name, task_kwargs={"random": seed}))
    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    env = RescaleAction(env, -1.0, 1.0)
    env = TimeLimit(env, MAX_FRAMES)
    env = RepeatAction(env, ACTION_REPEAT)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


def make_model(arm: str, task: str, seed: int, steps: int):
    _single_thread()
    from stable_baselines3.common.monitor import Monitor
    from sbx import SAC
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    env = Monitor(make_env(task, seed))
    kw = dict(learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
              learning_starts=LEARNING_STARTS, tau=0.005, gamma=0.99,
              train_freq=1, gradient_steps=2, ent_coef="auto",
              target_entropy="auto", seed=seed,
              policy_kwargs=dict(net_arch=[256, 256], layer_norm=(arm == "ours")))
    if arm == "ctrl":
        kw["learning_starts"] = steps + 1
    return SAC("MlpPolicy", env, **kw)


def evaluate(model, task: str, seed: int, episodes: int = EVAL_EPISODES) -> list:
    env = make_env(task, seed + EVAL_SEED_OFFSET)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=(seed + EVAL_SEED_OFFSET) if ep == 0 else None)
        done, total = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(float(total))
    return rets


TWIN_TRACE_EVERY = 10_000


def twin_model_name(arm: str, task: str, seed: int, n: int) -> str:
    return f"twin_{arm}_{task}_s{seed}_{n}.msgpack"


def save_weights(model, path) -> None:
    """Actor + critic WEIGHTS only (flax msgpack, < 1 MB). SB3's `model.save`
    is 8.4 MB a mark here (optimiser moments, targets): 250 MB a kernel on a
    disk 1 GB above its floor. Enough to re-evaluate and to read Q; not
    enough to resume training (that is `brain_b.save_life`)."""
    from flax.serialization import to_bytes
    pol = model.policy
    Path(path).write_bytes(to_bytes({"actor": pol.actor_state.params, "qf": pol.qf_state.params}))


def load_weights(arm: str, task: str, seed: int, path):
    """A fresh model of the cell's own architecture carrying the saved actor
    and critic weights (its optimiser and target are fresh: evaluate, don't train)."""
    from flax.serialization import from_bytes
    model = make_model(arm, task, seed, 1)
    pol = model.policy
    w = from_bytes({"actor": pol.actor_state.params, "qf": pol.qf_state.params},
                   Path(path).read_bytes())
    pol.actor_state = pol.actor_state.replace(params=w["actor"])
    pol.qf_state = pol.qf_state.replace(params=w["qf"])
    return model


def run_cell(arm: str, task: str, seed: int, steps: int, save_dir=None) -> dict:
    """One cell of the rule. `save_dir` (J19's cause twin; default None = the
    row's cell, unchanged, no key added): save the model at every evaluation
    mark and keep a trace row every TWIN_TRACE_EVERY decisions. Reads only."""
    from stable_baselines3.common.callbacks import BaseCallback
    t0 = time.time()
    model = make_model(arm, task, seed, steps)
    marks, trace, last = {}, [], {}

    def keep(n):
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_weights(model, Path(save_dir) / twin_model_name(arm, task, seed, n))

    class Marks(BaseCallback):
        def _on_training_start(self):
            if save_dir is None:
                return
            rec = self.model.logger.record

            def record(key, value, *a, **k):
                if key.startswith("train/"):
                    last[key[6:]] = value
                return rec(key, value, *a, **k)
            self.model.logger.record = record

        def _on_step(self):
            n = self.num_timesteps
            if save_dir is not None and n % TWIN_TRACE_EVERY == 0:
                trace.append({"t": int(n), **{k: (round(float(v), 5) if isinstance(v, (int, float))
                                                  or hasattr(v, "__float__") else None)
                                              for k, v in last.items()}})
            if n % EVAL_EVERY == 0 and n < steps:
                marks[n] = evaluate(self.model, task, seed)
                keep(n)
            return True

    model.learn(total_timesteps=steps, callback=Marks())
    final = evaluate(model, task, seed)
    marks[steps] = final
    keep(steps)
    mon = model.get_env().envs[0]
    return {"arm": arm, "task": task, "seed": seed, "steps": steps,
            "score": sum(final) / len(final), "eval": final,
            "marks": {str(k): sum(v) / len(v) for k, v in sorted(marks.items())},
            "train_returns": [float(x) for x in mon.get_episode_rewards()],
            "wall_s": round(time.time() - t0, 1), "versions": versions(),
            **({"mark_evals": {str(k): v for k, v in sorted(marks.items())},
                "trace": trace, "trace_every": TWIN_TRACE_EVERY,
                "models": [twin_model_name(arm, task, seed, n) for n in sorted(marks)]}
               if save_dir is not None else {})}


def verdict(cells: list, row: str, seeds=SEEDS, k: int = K, tasks=None) -> dict:
    out = {}
    for task, (m, sd) in ROWS[row].items():
        if tasks and task not in tasks:
            continue
        lo, hi = band(m, sd, k)
        t = {"band": [lo, hi]}
        for arm in ARMS:
            cs = sorted((c for c in cells if c.get("task") == task and c.get("arm") == arm
                         and "error" not in c), key=lambda c: c["seed"])
            want = sorted(CTRL_SEEDS if arm == "ctrl" else seeds)
            sc = [c["score"] for c in cs]
            mean = sum(sc) / len(sc) if sc else None
            t[arm] = {"per_seed": [round(s, 1) for s in sc], "mean": None if mean is None else round(mean, 1),
                      "complete": [c["seed"] for c in cs] == want}
        ctrl = t["ctrl"]
        void = not ctrl["complete"] or any(s >= lo for s in ctrl["per_seed"])
        lib_ok = t["lib"]["complete"] and lo <= t["lib"]["mean"] <= hi
        ours_ok = lib_ok and t["ours"]["complete"] and lo <= t["ours"]["mean"] <= hi
        t["verdict"] = ("VOID" if void else
                        (f"lib REPRODUCES; ours {'REPRODUCES' if ours_ok else 'MISSES'}"
                         if lib_ok else "lib MISSES (apparatus or a listed difference)"))
        out[task] = t
    return out


def cells_for(row: str, seeds=SEEDS, tasks=None, arms=ARMS) -> list:
    """lib first (the rule), then ours, then ctrl; the row's tasks (or a subset).
    `arms` narrows it for the rule's ONE extension (lib + ours on EXT_SEEDS; the
    ctrl seeds 0..2 already ran with the first five and are not re-run)."""
    steps = STEPS[row]
    tasks = list(tasks or ROWS[row])
    bad = [t for t in tasks if t not in ROWS[row]]
    if bad:
        raise ValueError(f"tasks {bad} not in row {row}")
    return [(a, task, s, steps) for a in arms for task in tasks
            for s in (CTRL_SEEDS if a == "ctrl" else seeds)]


def run_parallel(cells: list, outdir, workers: int = 4, save: bool = False) -> list:
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pending, running, done = list(cells), [], []
    while pending or running:
        while pending and len(running) < workers:
            arm, task, seed, steps = pending.pop(0)
            f = outdir / f"dmc_{arm}_{task}_s{seed}.json"
            p = subprocess.Popen([sys.executable, "-m", "experiments.brain_b_dmc", "cell",
                                  arm, task, str(seed), str(steps), str(f)]
                                 + ([str(outdir)] if save else []), cwd=REPO,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((arm, task, seed, f, p, time.time()))
        time.sleep(5)
        for item in list(running):
            arm, task, seed, f, p, t0 = item
            if p.poll() is None and time.time() - t0 < CHILD_TIMEOUT_S:
                continue
            if p.poll() is None:
                p.kill()
            err = p.stderr.read().decode()[-800:]
            running.remove(item)
            rec = (json.loads(f.read_text()) if f.exists()
                   else {"arm": arm, "task": task, "seed": seed, "error": f"rc={p.returncode}: {err}"})
            done.append(rec)
            print(f"{arm} {task} s{seed}: {rec.get('score', rec.get('error'))}", flush=True)
    return done


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
_s.run([_y.executable, "-m", "pip", "install", "-q", "--no-deps"] + __NODEPS__, check=True)
import json, os
os.environ.setdefault("MUJOCO_GL", "disable")
os.chdir("/tmp/jack")
from experiments.brain_b_dmc import run_parallel, verdict
r = run_parallel(__CELLS__, os.environ["JACK_OUT"])
json.dump({"cells": r, "verdict": verdict(r, __ROW__, tasks=__TASKS__) if __JUDGE__ else None},
          open(os.path.join(os.environ["JACK_OUT"], "brain_b_dmc.json"), "w"))
print("DONE", flush=True)
'''


def job_body(row: str, seeds=SEEDS, tasks=None, ext: bool = False) -> str:
    cells = (cells_for(row, EXT_SEEDS, tasks, arms=("lib", "ours")) if ext
             else cells_for(row, seeds, tasks))
    return (_JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__NODEPS__", repr(KERNEL_NODEPS))
            .replace("__CELLS__", repr(cells)).replace("__ROW__", repr(row)).replace("__TASKS__", repr(tasks))
            .replace("__JUDGE__", repr(not ext)))


def judge_ext(first: dict, ext: dict, row: str, tasks=None) -> dict:
    """The rule's extension reading: seeds 0..14 per arm (k = 15, the narrower
    band), ctrl = the first kernel's seeds 0..2. Pure, so it is testable."""
    return verdict(list(first["cells"]) + list(ext["cells"]), row,
                   seeds=SEEDS + EXT_SEEDS, k=len(SEEDS) + len(EXT_SEEDS), tasks=tasks)


def kernel(row: str, est_hours: float, task: str | None = None, ext: bool = False) -> dict:
    """Row 5: one kernel. Row 6: one kernel PER TASK (`kernel 6 <task>`) — the
    three together price at ~10 h on box timing, over one session."""
    from experiments.gpu import build_job, result_json, submit
    if row == "6" and task is None:
        raise ValueError("row 6 runs one task per kernel: kernel 6 <task>")
    tasks = [task] if task else None
    name = f"row{row}_dmc" + (f"_{task}" if task else "") + ("_ext" if ext else "")
    first = OUT / (f"row{row}_dmc" + (f"_{task}" if task else "") + ".json")
    if ext and not first.exists():
        raise RuntimeError(f"the extension is judged with {first}; it is missing")
    sub = submit(build_job(job_body(row, SEEDS, tasks, ext=ext)), prefer="kaggle-cpu", est_hours=est_hours,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_dmc.json"])
    if not sub.ok:
        raise RuntimeError(f"brain_b_dmc row {row} kernel: {sub.message}")
    r = result_json(sub, "brain_b_dmc.json")
    OUT.mkdir(parents=True, exist_ok=True)
    if ext:
        r["verdict"] = judge_ext(json.loads(first.read_text()), r, row, tasks)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("verdict"), indent=1), flush=True)
    return r


def twin_cells(task: str) -> list:
    """J19's cause twin: the row-6 `lib` arm only, the row's seeds and steps."""
    return cells_for("6", SEEDS, [task], arms=("lib",))


def twin_models(task: str) -> list:
    marks = list(range(EVAL_EVERY, STEPS["6"] + 1, EVAL_EVERY))
    return [twin_model_name(a, t, s, n) for a, t, s, _ in twin_cells(task) for n in marks]


def twin_job_body(task: str) -> str:
    return (_JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__NODEPS__", repr(KERNEL_NODEPS))
            .replace("run_parallel(__CELLS__, os.environ[\"JACK_OUT\"])",
                     f"run_parallel({twin_cells(task)!r}, os.environ[\"JACK_OUT\"], save=True)")
            .replace("__ROW__", repr("6")).replace("__TASKS__", repr([task]))
            .replace("__JUDGE__", "False"))


def twin_kernel(task: str, est_hours: float = 5.0) -> dict:
    """Submit the cause twin for one row-6 task; bring the summary AND every
    saved model home to OUT/j19_twin_<task>/ (the kernel's work dir is a
    temp dir). Refuses to report success with a model file missing."""
    import shutil
    from experiments.gpu import build_job, result_json, submit
    want = twin_models(task)
    sub = submit(build_job(twin_job_body(task)), prefer="kaggle-cpu", est_hours=est_hours,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["brain_b_dmc.json"])   # whole output comes home; a
    # download timeout re-fetches the summary only -> the models stay on Kaggle (`kernels output`)
    home = OUT / f"j19_twin_{task}"
    home.mkdir(parents=True, exist_ok=True)
    for name, path in sub.artifacts.items():            # whatever came home, first
        if name.endswith((".msgpack", ".json")):
            shutil.copy2(path, home / name)
    if not sub.ok:
        raise RuntimeError(f"brain_b_dmc twin {task}: {sub.message} (partial files in {home})")
    r = result_json(sub, "brain_b_dmc.json")
    (home / "twin.json").write_text(json.dumps(r, indent=1))
    missing = [m for m in want if not (home / m).exists()]
    for c in r["cells"]:
        print(f"{c.get('arm')} {c.get('task')} s{c.get('seed')}: "
              f"{c.get('score', c.get('error'))} marks {c.get('marks')}", flush=True)
    if missing:
        raise RuntimeError(f"twin {task}: {len(missing)} model files missing, e.g. {missing[:3]}; "
                           f"pull them: kaggle kernels output <user>/{getattr(sub, 'job_id', '<slug>')} -p {home}")
    print(f"twin {task}: {len(want)} models + twin.json in {home}", flush=True)
    return r


def main(argv):
    if len(argv) >= 7 and argv[1] == "cell":     # cell <arm> <task> <seed> <steps> <out>
        rec = run_cell(argv[2], argv[3], int(argv[4]), int(argv[5]),
                       save_dir=argv[7] if len(argv) > 7 else None)
        Path(argv[6]).write_text(json.dumps(rec))
        return 0
    if len(argv) == 3 and argv[1] == "twin":     # twin <row-6 task>: J19's cause twin
        twin_kernel(argv[2])
        return 0
    if len(argv) >= 2 and argv[1] == "probe":    # probe [steps]: price one cell per arm
        steps = int(argv[2]) if len(argv) > 2 else 1500
        global LEARNING_STARTS, EVAL_EVERY
        LEARNING_STARTS, EVAL_EVERY = min(LEARNING_STARTS, steps // 3), 10 ** 9
        for arm in ARMS:
            t0 = time.time()
            model = make_model(arm, "cartpole-swingup", 0, steps)
            model.learn(total_timesteps=steps)
            t1 = time.time()
            ev = evaluate(model, "cartpole-swingup", 0, episodes=1)
            t2 = time.time()
            print(f"{arm}: {steps} decisions ({LEARNING_STARTS} random) {t1 - t0:.1f}s, "
                  f"1 eval ep {t2 - t1:.1f}s score {ev[0]:.1f}", flush=True)
        return 0
    if len(argv) >= 3 and argv[1] == "kernel":
        row, rest = argv[2], argv[3:]
        ext = "ext" in rest
        rest = [a for a in rest if a != "ext"]
        kernel(row, {"5": 3.0, "6": 5.0}[row] * (2.0 if ext else 1.0), rest[0] if rest else None, ext=ext)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
