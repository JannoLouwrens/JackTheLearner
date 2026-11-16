"""Brain A speed probe (J5 step 2): the pinned runner's smallest preset on DMC
cartpole-swingup, timed on a T4 and on a free CPU kernel.

Source: `sheeprl==0.8.5` from PyPI, algorithm `dreamer_v3_5` with
`algo=dreamer_v3_5_1M` (`size1m`); see `experiments/brain_a/README.md` for the
pins and why. Code only, never weights: every run here starts from zero.

THIS IS A SPEED MEASUREMENT, NOT A LEARNING RESULT. No score is read from it,
no seat moves on it, one seed. It answers §2's "speed is measured nowhere".

What is run (the runner is not edited; these are command-line overrides):

    exp=dreamer_v3_5_dmc_walker_walk algo=dreamer_v3_5_1M
    env.wrapper.domain_name=cartpole env.wrapper.task_name=swingup
    env.wrapper.from_pixels=False env.wrapper.from_vectors=True
    algo.cnn_keys.encoder=[] algo.mlp_keys.encoder=[state]

Ours beside theirs — every setting that differs from the shipped experiment
file (`dreamer_v3_5_dmc_walker_walk.yaml`, pixels, 12M) and why:

| setting            | shipped example | here   | why                                             |
|--------------------|-----------------|--------|-------------------------------------------------|
| model              | 12M             | 1M     | the order: smallest preset                      |
| task               | walker walk     | cartpole swingup | the order; §2's first known-answer rows |
| observation        | pixels          | vector | §2's rows are proprioceptive                    |
| env.num_envs       | 16              | 4      | §2's published protocol (4 envs)                |
| env.capture_video  | True            | False  | no renderer needed for vectors                  |
| metric.log_every   | 5000            | 500 (T4), 100 (CPU) | timing resolution only             |
| fabric.accelerator | cpu             | cuda on the T4 | the thing being measured                |
| unchanged          | action repeat 2, replay_ratio 0.5 (= train ratio 512), batch 16 x 64, precision 32-true, compile off |

The run is stopped by wall clock, not by step count (the speed is unknown
before the run). Speed is read from the runner's own TensorBoard scalars
(`Params/replay_ratio` events carry policy_step and wall time; the runner's
`Time/sps_train` and `Time/sps_env_interaction` are reported beside it).
Steady state = events after training starts, first interval dropped.

A "decision" below is one policy step of one environment (the runner's
`policy_step`); with action repeat 2 it is two simulator steps. The train
ratio is enforced synchronously by the runner, so "achieved ratio" is checked
(replay_ratio x 1024), not assumed.

A second arm, reported only, turns on the runner's own `algo.compile.enabled`
(torch.compile) — a shipped switch, default off.

Usage:  python -m experiments.brain_a.speed colab|colab2|kaggle-cpu|local [budget_s]

`colab2` = the second round: compiled (budget + 10 min for compilation), `bf16-mixed`, and three
default runs sharing one T4 at once (seeds 42-44; throughput = their sum).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/brain_a")

BASE = ["exp=dreamer_v3_5_dmc_walker_walk", "algo=dreamer_v3_5_1M",
        "env.wrapper.domain_name=cartpole", "env.wrapper.task_name=swingup",
        "env.wrapper.from_pixels=False", "env.wrapper.from_vectors=True",
        "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]",
        "env.num_envs=4", "env.capture_video=False"]

_JOB = r'''
import glob, json, os, signal, subprocess, sys, time
DEV = __DEV__
RUNS = __RUNS__
BASE = __BASE__
PIP = __PIP__
INSTALL = "none"
if PIP:
    # the full pinned install first; if the resolver refuses on this image, say why and install the
    # same pinned runner without its optional video/GUI dependencies (the image's own are used)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + PIP, capture_output=True, text=True)
    INSTALL = "full"
    if r.returncode != 0:
        print("PIP_FULL_FAILED", (r.stdout + r.stderr)[-3000:], flush=True)
        # measured on Colab 11-16 (Python 3.13): dm_control's dependency `labmaze` has no wheel and
        # does not build. The control suite does not import it, so: the pinned runner with its own
        # dependencies, then dm_control without `labmaze`.
        INSTALL = "runner full; dm_control --no-deps + its deps minus labmaze"
        r1 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIP[0]],
                            capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + PIP[1:],
                            capture_output=True, text=True)
        r3 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dm-env", "dm-tree", "mujoco", "glfw",
                             "lxml", "pyopengl", "pyparsing", "absl-py", "scipy", "tqdm", "protobuf", "requests"],
                            capture_output=True, text=True)
        print("PIP_FALLBACK", r1.returncode, r2.returncode, r3.returncode,
              (r1.stderr + r2.stderr + r3.stderr)[-3000:], flush=True)
        r2.returncode = r2.returncode or r3.returncode
        if r1.returncode or r2.returncode:
            raise SystemExit("pip install failed on both paths")
os.environ["MUJOCO_GL"] = "disable"   # vectors only: no renderer is imported
WORK = os.environ.get("JACK_OUT") or os.getcwd()
os.makedirs(WORK, exist_ok=True)


def _versions():
    import importlib.metadata as md, platform
    v = {"python": platform.python_version(), "cpus": os.cpu_count(), "install": INSTALL}
    for p in ("sheeprl", "torch", "lightning", "dm_control", "mujoco", "numpy", "gymnasium"):
        try:
            v[p] = md.version(p)
        except Exception as e:
            v[p] = "missing"
    try:
        v["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                   "--format=csv,noheader"], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
    except Exception:
        v["gpu"] = ""
    return v


def _scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for f in glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
        ea = EventAccumulator(os.path.dirname(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((e.step, e.wall_time, e.value) for e in ea.Scalars(tag))
    return {k: sorted(v) for k, v in out.items()}


def _cfg(logdir):
    import yaml
    for f in glob.glob(os.path.join(logdir, "**", "config.yaml"), recursive=True):
        if ".hydra" in f:
            continue
        c = yaml.safe_load(open(f))
        a, e = c.get("algo", {}), c.get("env", {})
        rm = a.get("world_model", {}).get("recurrent_model", {})
        return {"dense_units": a.get("dense_units"), "discrete_size": a.get("world_model", {}).get("discrete_size"),
                "recurrent_state_size": rm.get("recurrent_state_size"), "replay_ratio": a.get("replay_ratio"),
                "batch": a.get("per_rank_batch_size"), "seq": a.get("per_rank_sequence_length"),
                "learning_starts": a.get("learning_starts"), "num_envs": e.get("num_envs"),
                "action_repeat": e.get("action_repeat"), "env_id": e.get("id"),
                "compile": a.get("compile", {}).get("enabled"), "mlp_keys": a.get("mlp_keys", {}).get("encoder"),
                "accelerator": c.get("fabric", {}).get("accelerator"), "precision": c.get("fabric", {}).get("precision")}
    return None


def _params(logdir):
    try:
        import torch
        cks = sorted(glob.glob(os.path.join(logdir, "**", "*.ckpt"), recursive=True))
        if not cks:
            return None
        sd = torch.load(cks[-1], map_location="cpu", weights_only=False)
        n = {}
        for k in ("world_model", "actor", "critic"):
            if k in sd:
                n[k] = int(sum(t.numel() for t in sd[k].values() if hasattr(t, "numel")))
        return n
    except Exception as e:
        return {"error": repr(e)[:200]}


def _harvest(rec, logdir):
    try:
        rec["config"] = _cfg(logdir)
        sc = _scalars(logdir)
        rr = sc.get("Params/replay_ratio", [])
        ls = (rec["config"] or {}).get("learning_starts") or 1024
        rec["events"] = len(rr)
        rec["last_policy_step"] = rr[-1][0] if rr else None
        tr = [e for e in rr if e[0] > ls]
        if len(tr) >= 3:
            a, b = tr[1], tr[-1]          # first training interval dropped (warm-up, compile)
            dps = (b[0] - a[0]) / (b[1] - a[1])
            rep = rec["config"]["replay_ratio"]
            rec["steady"] = {"from_step": a[0], "to_step": b[0], "seconds": round(b[1] - a[1], 1),
                             "decisions_per_s": round(dps, 2),
                             "gradient_steps_per_s": round(dps * rep, 3),
                             "cumulative_replay_ratio_last": round(b[2], 4),
                             "train_ratio": rep * rec["config"]["batch"] * rec["config"]["seq"],
                             "wall_s_per_100k_decisions": round(1e5 / dps, 0),
                             "simulator_steps_per_s": round(dps * rec["config"]["action_repeat"], 2)}
        elif len(tr) >= 1 and rr[0][0] <= ls:
            # too few logged intervals: one coarse figure from the last prefill event (includes warm-up)
            pre = [e for e in rr if e[0] <= ls][-1]
            rec["coarse"] = {"from_step": pre[0], "to_step": tr[-1][0], "seconds": round(tr[-1][1] - pre[1], 1),
                             "decisions_per_s": round((tr[-1][0] - ls) / (tr[-1][1] - pre[1]), 3)}
        for tag in ("Time/sps_train", "Time/sps_env_interaction"):
            v = [e[2] for e in sc.get(tag, [])]
            if v:
                v2 = v[1:] if len(v) > 2 else v
                rec[tag] = {"n": len(v), "median": round(sorted(v2)[len(v2) // 2], 3), "last": round(v[-1], 3)}
        rew = sc.get("Rewards/rew_avg", [])
        rec["rew_avg_last"] = round(rew[-1][2], 1) if rew else None
        rec["params"] = _params(logdir)
    except Exception as e:
        rec["error"] = repr(e)[:400]
    return rec


def one(name, budget_s, extra, copies=1):
    # copies > 1: that many identical runs (seeds 42, 43, ...) share the one device at once; the
    # throughput figure is their SUM. Copy 0 keeps the runner's default seed and the old log path.
    logdirs = [os.path.join(WORK, "logs_" + name + ("" if i == 0 else "_%d" % i)) for i in range(copies)]
    t0 = time.time()
    procs, logs = [], []
    for i, logdir in enumerate(logdirs):
        cmd = ([sys.executable, "-m", "sheeprl"] + BASE + ["fabric.accelerator=" + DEV, "log_root=" + logdir]
               + extra + ([] if i == 0 else ["seed=%d" % (42 + i)]))
        print("RUN", name, i, " ".join(cmd), flush=True)
        log = open(os.path.join(WORK, name + ("" if i == 0 else "_%d" % i) + ".log"), "w")
        procs.append(subprocess.Popen(cmd, cwd=WORK, stdout=log, stderr=subprocess.STDOUT))
        logs.append(log)
    smi = ""
    while any(p.poll() is None for p in procs) and time.time() - t0 < budget_s:
        time.sleep(10)
        if DEV == "cuda" and not smi and time.time() - t0 > budget_s * 0.6:
            try:
                smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=30).stdout.strip()
            except Exception:
                pass
    early = [p.poll() for p in procs]
    for p in procs:
        if p.poll() is None:
            # the event writer flushes every 120 s at worst: read what is on disk, then stop the run
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
    wall = time.time() - t0
    for log in logs:
        log.close()
    recs = []
    for i, logdir in enumerate(logdirs):
        rec = {"name": name if i == 0 else "%s_%d" % (name, i), "budget_s": budget_s, "wall_s": round(wall, 1),
               "extra": extra, "exited_early_rc": early[i], "gpu_util_mem": smi}
        _harvest(rec, logdir)
        if "steady" not in rec:
            rec["log_tail"] = open(os.path.join(WORK, rec["name"] + ".log")).read()[-1500:]
        recs.append(rec)
    if copies == 1:
        rec = recs[0]
    else:
        st = [r.get("steady") for r in recs]
        rec = {"name": name, "copies": copies, "budget_s": budget_s, "wall_s": round(wall, 1), "extra": extra,
               "gpu_util_mem": smi, "runs": recs}
        if all(st):
            dps = sum(x["decisions_per_s"] for x in st)
            rec["sum"] = {"decisions_per_s": round(dps, 2),
                          "gradient_steps_per_s": round(sum(x["gradient_steps_per_s"] for x in st), 3),
                          "wall_s_per_100k_decisions_each": [x["wall_s_per_100k_decisions"] for x in st],
                          "device_s_per_100k_decisions": round(1e5 / dps, 0)}
    print("REC", json.dumps(rec), flush=True)
    return rec


res = {"device": DEV, "versions": _versions(), "runs": []}
print("VERSIONS", json.dumps(res["versions"]), flush=True)
for run in RUNS:
    res["runs"].append(one(*run))
    json.dump(res, open(os.path.join(WORK, "brain_a_speed.json"), "w"))
print("RESULT " + json.dumps(res), flush=True)
print("DONE", flush=True)
'''


def body(dev: str, runs: list, pip=("sheeprl==0.8.5", "dm_control")) -> str:
    return (_JOB.replace("__DEV__", repr(dev)).replace("__RUNS__", repr(runs))
            .replace("__BASE__", repr(BASE)).replace("__PIP__", repr(list(pip))))


def main(argv) -> int:
    where = argv[1] if len(argv) > 1 else "local"
    OUT.mkdir(parents=True, exist_ok=True)
    if where == "local":
        # box smoke: the job body itself, already-installed runner, tiny budget
        import os
        import subprocess
        budget = int(argv[2]) if len(argv) > 2 else 150
        work = OUT / "smoke"
        work.mkdir(parents=True, exist_ok=True)
        f = work / "job.py"
        f.write_text(body("cpu", [("smoke", budget, ["metric.log_every=20"])], pip=()))
        return subprocess.run([sys.executable, str(f)], cwd=work,
                              env={**os.environ, "JACK_OUT": str(work)}).returncode
    from experiments.gpu import build_job, result_json, submit
    if where == "colab":
        budget = int(argv[2]) if len(argv) > 2 else 400
        runs = [("t4_default", budget, ["metric.log_every=500"]),
                ("t4_compile", budget + 120, ["metric.log_every=500", "algo.compile.enabled=True"])]
        sub = submit(build_job(body("cuda", runs)), prefer="colab", est_hours=0.45,
                     timeout_s=2 * budget + 120 + 600, fetch=["brain_a_speed.json"])
    elif where == "colab2":
        # J5 step 2, second round: the three arms the first left owed, each with a budget that
        # outlasts compilation. Same unmodified runner; shipped switches only.
        budget = int(argv[2]) if len(argv) > 2 else 1500
        runs = [("t4_compile", budget + 600, ["metric.log_every=500", "algo.compile.enabled=True"]),
                ("t4_bf16", budget, ["metric.log_every=500", "fabric.precision=bf16-mixed"]),
                ("t4_default_x3", budget, ["metric.log_every=500"], 3)]
        sub = submit(build_job(body("cuda", runs)), prefer="colab", est_hours=1.4,
                     timeout_s=3 * budget + 600 + 1800, fetch=["brain_a_speed.json"])
    elif where == "kaggle-cpu":
        budget = int(argv[2]) if len(argv) > 2 else 2400
        runs = [("cpu_4threads", budget, ["metric.log_every=100", "num_threads=4"]),
                ("cpu_default", budget, ["metric.log_every=100"])]
        sub = submit(build_job(body("cpu", runs)), prefer="kaggle-cpu", est_hours=1.6,
                     timeout_s=2 * budget + 1800, fetch=["brain_a_speed.json"])
    else:
        print(__doc__)
        return 2
    print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
    if not sub.ok:
        return 1
    r = result_json(sub, "brain_a_speed.json")
    (OUT / f"speed_{where}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
