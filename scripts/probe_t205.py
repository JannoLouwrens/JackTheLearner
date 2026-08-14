"""Probe kernel for T2.05 sizing — times the PRODUCTION configuration on the
target GPU before the real dispatch (the T2.04/B1 lesson: a cost measured on
the smoke's configuration is not a cost for the production configuration).

Run from the repo root on this box:
    /data/venvs/jackthelearner/bin/python scripts/probe_t205.py

Submits a short Kaggle kernel (est 0.1 h) that times, at PipelineConfig()
defaults (d_model=512, n_layers=8) with enable_world_model=True, K=5:
  - per-collect-row cost (600 windows)
  - pipeline build cost
  - per-train-step cost (5 warmup + 25 timed, batch 256, deep supervision)
  - full-eval cost (600 rows, both passes)
Prints the JSON the dispatch arithmetic needs. The numbers go into the
_submit comment in t2_05_world_model.py, committed before dispatch.
"""

import json
from pathlib import Path

from experiments.gpu import build_job, submit

JOB = r'''
import subprocess as _sp, sys as _sys, os as _o, time
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
import json
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import experiments.tests.t2_05_world_model as t
from TrainingPipeline import TrainingPipeline, PipelineConfig

out = {}
env = gym.make("Humanoid-v5")
t0 = time.time()
X0, A, Y, ep, falls, short = t._collect_windows(env, 600, 0, 0)
out["collect_s_per_row"] = round((time.time() - t0) / 600, 5)

torch.manual_seed(0)
t0 = time.time()
tp = TrainingPipeline(PipelineConfig(enable_world_model=True))
out["build_s"] = round(time.time() - t0, 2)
out["gpu"] = (torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "cpu")
out["wm_params"] = sum(p.numel() for p in tp.model.world_model.parameters())

readout = nn.Linear(tp.model.config.obs_dim, t.OBS_DIM).to(tp.device)
tp.normalize_obs(np.concatenate([X0, Y.reshape(-1, t.OBS_DIM)]))
mu, sd = t._z_stats(tp)
X0z, Yz = t._z(X0, mu, sd), t._z(Y, mu, sd)

t.WM_STEPS = 5                       # warmup (cudnn autotune, allocator)
t._train_wm(tp, readout, X0z, A, Yz, 0)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t.WM_STEPS = 25
t0 = time.time()
t._train_wm(tp, readout, X0z, A, Yz, 0)
if torch.cuda.is_available():
    torch.cuda.synchronize()
out["train_s_per_step"] = round((time.time() - t0) / 25, 4)

t0 = time.time()
_pred, det_ok = t._eval_deterministic(tp, readout, X0z, A)
out["eval_s_per_600rows_2pass"] = round(time.time() - t0, 2)
out["det_ok"] = bool(det_ok)

json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "probe205.json"),
                    "w"), indent=1)
print("PROBE", json.dumps(out), flush=True)
'''


def main():
    job = build_job(JOB)
    res = submit(job, prefer="kaggle", est_hours=0.1, timeout_s=1500,
                 fetch=["probe205.json"])
    if not res.ok:
        raise SystemExit(f"probe failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["probe205.json"]).read_text())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
