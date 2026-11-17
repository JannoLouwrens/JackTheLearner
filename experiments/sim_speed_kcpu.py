"""J6 — the speed table's Kaggle CPU column (ORDERS J6, last step).

The box column is already measured, each option on its own proof file:
shipped ~40 dec/CPU-s; `lazy_contact_audio` 51.8 (tests/test_fast_sim.py);
`integrator="implicitfast"` 70.1, with lazy audio 108.5 (tests/test_integrator.py,
a WORLD CHANGE — default stays OFF); `multiray_vision` 39.2 (tests/test_multiray.py);
`make_vec` vec2 1.59x wall, per CPU-s unchanged (tests/test_jack_env_vec.py).

This measures the same options in ONE process on one Kaggle CPU kernel, on
J8's pilot world (`pilot_b.make_world`: difficulty 0.75, dr100-eat,
random needs, random layout), seeded uniform-random actions, so the two
columns are comparable to each other — not to a learner's decisions/s
(no learning runs here). Every config gets the same seed and action stream;
CPU-s = this process + reaped children (vec workers are closed before reading).
Identity is NOT re-proven here (the box tests own that); this is speed only.

Usage: $PY -m experiments.sim_speed_kcpu box [T]     # local, small T
       $PY -m experiments.sim_speed_kcpu kernel      # one Kaggle CPU kernel
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

CONFIGS = {
    "shipped": {},
    "lazy": {"lazy_contact_audio": True},
    "implicitfast": {"integrator": "implicitfast"},
    "lazy+implicitfast": {"lazy_contact_audio": True, "integrator": "implicitfast"},
    "multiray": {"multiray_vision": True},
}
VEC_CONFIGS = {"vec4 shipped": {}, "vec4 lazy+implicitfast": CONFIGS["lazy+implicitfast"]}
SEED = 301
OUT = Path("/data/jack-logs/j6")


def _cpu() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _world_kw() -> dict:
    from experiments.jack_env import NEEDS_S_TRP
    from experiments.pilot_b import DIFFICULTY, REWARD
    return dict(difficulty=DIFFICULTY, reward=REWARD,
                random_needs=NEEDS_S_TRP, random_layout=True)


def single(kw: dict, T: int) -> dict:
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=SEED, **_world_kw(), **kw)
    rng = np.random.default_rng(7)
    env.reset(seed=SEED)
    lo, hi = env.action_space.low, env.action_space.high
    acts = rng.uniform(lo, hi, size=(T,) + lo.shape).astype(env.action_space.dtype)
    ends = 0
    c0, t0 = _cpu(), time.time()
    for t in range(T):
        _, _, te, tr, _ = env.step(acts[t])
        if te or tr:
            ends += 1
            env.reset()
    w, c = time.time() - t0, _cpu() - c0
    env.close()
    return {"T": T, "wall_s": round(w, 2), "cpu_s": round(c, 2), "ends": ends,
            "dec_per_wall_s": round(T / w, 1), "dec_per_cpu_s": round(T / max(c, 1e-9), 1)}


def vec(kw: dict, T: int, n: int = 4) -> dict:
    from experiments.jack_env import make_vec
    seeds = [SEED + k for k in range(n)]
    c0, t0 = _cpu(), time.time()
    ve = make_vec(n, seeds=seeds, **_world_kw(), **kw)
    ve.reset(seed=seeds)
    rng = np.random.default_rng(7)
    sp = ve.single_action_space
    for _ in range(T):
        ve.step(rng.uniform(sp.low, sp.high, size=(n,) + sp.low.shape).astype(sp.dtype))
    ve.close()
    w, c = time.time() - t0, _cpu() - c0
    return {"T_per_env": T, "n": n, "wall_s": round(w, 2), "cpu_s": round(c, 2),
            "dec_per_wall_s": round(n * T / w, 1),
            "dec_per_cpu_s": round(n * T / max(c, 1e-9), 1),
            "note": "includes worker start-up"}


def run_all(T: int) -> dict:
    import platform
    res = {"host": platform.node(), "cpus": os.cpu_count(),
           "load": os.getloadavg()[0], "configs": {}}
    for name, kw in CONFIGS.items():
        res["configs"][name] = single(kw, T)
        print(name, res["configs"][name], flush=True)
    for name, kw in VEC_CONFIGS.items():
        res["configs"][name] = vec(kw, T)
        print(name, res["configs"][name], flush=True)
    import gymnasium, mujoco
    res["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                       "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__}
    return res


# mujoco/gymnasium pinned to the box venv's versions so the columns compare.
KERNEL_PINS = ["mujoco==3.2.3", "gymnasium==1.1.1"]
KERNEL_T = 3000
REMOTE_TIMEOUT_S = int(3 * 3600)

_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from experiments.sim_speed_kcpu import run_all
r = run_all(__T__)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "sim_speed.json"), "w"))
print("DONE", flush=True)
'''


def kernel() -> dict:
    from .gpu import build_job, result_json, submit
    body = _JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__T__", repr(KERNEL_T))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=1.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["sim_speed.json"])
    if not sub.ok:
        raise RuntimeError(f"sim_speed_kcpu kernel: {sub.message}")
    r = result_json(sub, "sim_speed.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kcpu_speed.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r, indent=1), flush=True)
    return r


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kernel":
        kernel()
    elif len(sys.argv) > 1 and sys.argv[1] == "box":
        T = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        r = run_all(T)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"box_speed_T{T}.json").write_text(json.dumps(r, indent=1) + "\n")
    else:
        raise SystemExit(__doc__)
