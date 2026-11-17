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

RESULT — Kaggle CPU column (kernel jack-cpu-1794897525, 11-17 06:44, mujoco
3.17.0, python 3.13, 4 cpus, T=3000 decisions each, `/data/jack-logs/j6/kcpu_speed.json`):
  config                    dec/CPU-s   x shipped   (box T=40 smoke, x)
  shipped                      72.7       1.00
  lazy_contact_audio           88.0       1.21      (1.27)
  implicitfast (WORLD CHANGE)  98.6       1.36      (1.60)
  lazy + implicitfast         124.1       1.71      (2.44)
  multiray_vision              72.5       1.00      (not a speed lever)
  vec4 shipped                145.3 wall  41.6/CPU-s (incl. worker start-up)
  vec4 lazy + implicitfast    263.4 wall  77.9/CPU-s (incl. worker start-up)
  Kaggle's shipped core is ~1.4-1.8x the box's per CPU-s; the levers' ratios
  are smaller there (1.71x vs 2.44x at T=40 on the box; T=3000 is the
  sturdier figure). One Kaggle kernel, 4 envs, lazy+implicitfast: 263
  decisions/s wall -> 1M decisions ~1.05 h; shipped vec4 ~1.9 h; shipped
  single ~3.8 h. Item (2b) (40 sub-steps in one native call) not measured.

BOX ROW with (2b) (11-17 ~12:40, `box 300`, same world/seed/actions, one
run each, `/data/jack-logs/j6/box_speed_T300.json`; identity proven in
tests/test_native_substep_env.py, which needs lazy audio):
  config                      dec/CPU-s   x shipped
  shipped                        49.0       1.00
  lazy                           60.6       1.24
  implicitfast (WORLD CHANGE)    80.8       1.65
  lazy + implicitfast           120.0       2.45
  lazy + native                  93.7       1.91   (identical world)
  lazy + native + implicitfast  281.3       5.74   (world change = implicitfast's)
  vec4 shipped                  113.6 wall  38.6/CPU-s
  vec4 lazy + implicitfast      237.5 wall  86.3/CPU-s
  The native gain under implicitfast (2.34x here, 2.0-2.2x in the test) is
  above (2b)'s 1.97x ceiling, which was priced on a different world/seed
  (substep_split.py, seed 153, J8 shipped cal) — read the test's per-seed
  range as the sturdier figure. Kaggle column for native: NOT measured (the
  C build and its identity must be proven on Kaggle's BLAS first).

NATIVE ON KAGGLE — kernel READY 11-17 13:00, not yet run. `kernel` now first
runs `native_identity(300)` (rule in its docstring, fixed before any run:
cases A + D x seeds 153-155 exact, >= 1 death each, dt+1ulp control must
differ on all 3 seeds; else the option is box-only), builds the C under
JACK_NATIVE_DIR=/tmp/jack_native against the wheel's own headers, then the
speed rows incl. the two native ones. PRE-CHECK on the box under Kaggle's
pins (jack-std venv: mujoco 3.17.0, gymnasium 1.4.0, py3.12; still THIS
box's aarch64 — Kaggle's x86 FMA order is what the kernel tests): the C
compiles clean against 3.17 headers; A 0/0/0 of 300 differing (deaths
2/2/1), D 0/0/0 (2/2/1), control seed 153 differs 85/300 (first at 108);
lazy -> lazy+native 64.8-93.8 -> 99.3-153.4 dec/CPU-s; under implicitfast
108-142 -> 254-363. (3.17 changes the world: case A seed 155 dies once
where 3.2.3 died twice — a pin difference, not a native one.)

Usage: $PY -m experiments.sim_speed_kcpu box [T] [identity_N]   # local, small T
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
    "lazy+native": {"lazy_contact_audio": True, "native_substep": True},
    "lazy+native+implicitfast": {"lazy_contact_audio": True, "native_substep": True,
                                 "integrator": "implicitfast"},
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


def _compat_multiray() -> None:
    """mujoco >= 3.3 added a `normal` out-arg to mj_multiRay (kernel 2 died on
    it, 11-17 06:3x). fast_sim calls the 3.2.3 signature; for the Kaggle column
    only, shim the call by inserting normal=None. The box path is untouched."""
    import mujoco
    f = mujoco.mj_multiRay
    if getattr(f, "_jack_shim", False):
        return
    new_sig = []  # decided on the first call, so later calls pay no exception

    def shim(*a):
        if not new_sig:
            try:
                f(*a)
                new_sig.append(False)
                return None
            except TypeError:
                if len(a) != 11:
                    raise
                new_sig.append(True)
        if new_sig[0]:
            return f(*a[:9], None, *a[9:])
        return f(*a)
    shim._jack_shim = True
    mujoco.mj_multiRay = shim


def native_identity(N: int = 300) -> dict:
    """J6 (2b) on THIS host's mujoco/BLAS: tests/test_native_substep_env cases
    A (dying, e0=0.05) and D (A + implicitfast), seeds 153-155, N decisions,
    exact equality every decision; the dt+1ulp control must differ on >= 1
    decision a seed. Rule (fixed before the kernel runs): PASS = 0 differing
    in all 6 pairs, >= 1 death in each, and the control differs on all 3
    seeds; otherwise the option is box-only. A C build error is a FAIL with
    its message, and the native speed rows are then skipped."""
    out = {"N": N, "pairs": {}, "control": {}}
    try:
        import experiments.tests.test_native_substep_env as T
        T.N = N
        ok = True
        for case in ("A", "D"):
            for seed in T.SEEDS:
                r = T.pair(seed, case)
                out["pairs"][f"{case}{seed}"] = r
                ok &= r["bad"] == 0 and r["deaths"] >= 1
                print("identity", case, seed, r, flush=True)
        for seed in T.SEEDS:
            r = T.pair(seed, "A", control="ulp")
            out["control"][str(seed)] = r
            ok &= r["bad"] >= 1
            print("control dt+1ulp", seed, r, flush=True)
        out["verdict"] = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001 — a build failure is a reading
        out["verdict"] = "FAIL"
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
        print("native identity error:", out["error"], flush=True)
    return out


def run_all(T: int, identity_n: int = 0) -> dict:
    import platform
    _compat_multiray()
    res = {"host": platform.node(), "cpus": os.cpu_count(),
           "load": os.getloadavg()[0], "configs": {}}
    if identity_n:
        res["native_identity"] = native_identity(identity_n)
    for name, kw in CONFIGS.items():
        if (identity_n and "native" in name
                and "error" in res["native_identity"]):
            res["configs"][name] = {"skipped": "native build failed"}
            continue
        res["configs"][name] = single(kw, T)
        print(name, res["configs"][name], flush=True)
    for name, kw in VEC_CONFIGS.items():
        res["configs"][name] = vec(kw, T)
        print(name, res["configs"][name], flush=True)
    import gymnasium, mujoco
    res["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                       "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__}
    return res


# The box venv runs mujoco 3.2.3 / gymnasium 1.1.1, but Kaggle is Python 3.13
# and mujoco 3.2.3 has no 3.13 wheel (first kernel 11-17 06:27 died building
# it). pilot_b's Kaggle pin mujoco==3.17.0 is used instead — a listed
# difference: compare the columns by RATIO to their own `shipped` row, not
# absolute dec/CPU-s. Versions are written into the result.
KERNEL_PINS = ["mujoco==3.17.0", "gymnasium==1.1.1", "cffi"]
KERNEL_T = 3000
KERNEL_IDENTITY_N = 300
REMOTE_TIMEOUT_S = int(3 * 3600)

_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["JACK_NATIVE_DIR"] = "/tmp/jack_native"
from experiments.sim_speed_kcpu import run_all
r = run_all(__T__, __N__)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "sim_speed.json"), "w"))
print("DONE", flush=True)
'''


def kernel() -> dict:
    from .gpu import build_job, result_json, submit
    body = (_JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__T__", repr(KERNEL_T))
            .replace("__N__", repr(KERNEL_IDENTITY_N)))
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
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        r = run_all(T, n)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"box_speed_T{T}.json").write_text(json.dumps(r, indent=1) + "\n")
    else:
        raise SystemExit(__doc__)
