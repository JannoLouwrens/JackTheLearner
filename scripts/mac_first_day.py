#!/usr/bin/env python3
"""mac_first_day.py — the Mac mini's first-day speed test, one command (ORDERS "Next up" 2).

Source of the twelve measurements: PUBLISHED_PRIORS.md §10 "Reproduce first"
and its long form, `/data/jack-research/reports/Mac mini M6 speed for Jack.md`
§"The first day". Every speed figure in §10 is SCALED from other machines;
this script replaces them with measurements. Day one's numbers are the first
anywhere, so they are logged with versions and settings and repeated a week
later.

    python scripts/mac_first_day.py                  # everything it can run
    python scripts/mac_first_day.py --only 3,4       # chosen measurements
    python scripts/mac_first_day.py --quick          # short timings (a smoke)
    python scripts/mac_first_day.py --procs 1,2      # cap the process ladder

Output: `first_day_<host>_<UTC stamp>.json` in `--out` (default: the current
directory), one record per measurement: `status` RAN / SKIPPED (with the
reason: a missing library or device is a SKIP, never a zero) / FAILED (with
the error tail), its settings, and its numbers beside the estimate it
replaces. Each measurement runs in its own child process with a timeout, so
an MPS crash cannot take the others down.

Rules this script obeys (§10 "Take as-is"):
  * It never lowers its own priority: no nice, no `taskpolicy -b`, no
    background QoS. On macOS background priority pins work to efficiency
    cores (2-3x loss, no error). Run it from a normal Terminal, under
    `caffeinate -dimsu`, with `sudo pmset -a sleep 0 disksleep 0` set.
  * Threads are capped in code (OMP/MKL/OPENBLAS = 1 per worker unless the
    measurement is the thread count itself), never by priority.
  * Portable: standard library + whichever of numpy/mujoco/torch/sb3/sbx/
    mlx/jax are installed; no /proc, no Linux-only call outside an explicit
    `platform.system()` branch. On the Linux box it runs as a smoke (CPU
    rows only; MPS/MLX rows SKIP).

Automated here: 1-8 and 11's MPS-context line. NOT automated (printed as
NOT_AUTOMATED with the command to run by hand):
  9  Dreamer steps/s on official JAX, sheeprl CPU and sheeprl MPS — needs
     the J5 pins installed on the Mac (experiments/brain_a/README.md);
  10 sheeprl CPU-vs-MPS 2,000-step correctness — same installs;
  11 renders from a logged-in session, SSH, LaunchAgent, LaunchDaemon — needs
     the four contexts set up; this script records which context it ran in;
  12 the 24 h soak behind a wall meter.

Box smoke, 11-16 22:25-22:34 (Ampere VM, 4 cores, load ~0.7, nice 19 from
outside, `--quick --procs 1,2`; a smoke of the script, not a Mac estimate):
  1 Linux aarch64, 4 cpus, mujoco 3.17.0, torch 2.15 cpu, sbx 0.28, no mlx
  2 humanoid python mj_step 6,032 steps/s/proc; 2 procs 12,028 (no testspeed)
  3 Jack's world raw sub-steps/s: Euler 17,210  RK4 4,876  implicitfast 16,561
  4 decisions/s, 1 proc, 10 s: shipped 50.2, implicitfast 73.0, +lazy audio 124.3
    (J6's per-CPU-s 41 / 70 / 108 — agrees)
  5 shipped 1 -> 2 procs 50.5 -> 98.8; implicitfast+lazy 123.5 -> 234.3
  6 SAC updates/s (50 warm, 200 timed): SB3 t1 32.7, SB3 t4 55.6, SBX t1 63.3;
    SBX 2 runs 39.5 each (1.28x) before XLA single-threading was added
  7 numpy f32 1024^2 0.04 TFLOPS; 8 torch GRU cpu b16 1,459 us, b1024 37,492 us
  9 10 12 NOT_AUTOMATED; 11 context terminal, MPS false; MPS/MLX rows SKIP.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ESTIMATE = {  # §10's scaled figures, verbatim, so each row prints beside its guess
    1: "2 super + 4 performance + 6 efficiency cores",
    2: "50,000-90,000 humanoid steps/s per fast core; whole chip 4.5-8x the ARM server",
    3: "RK4 7,000-12,500 and implicitfast 26,000-48,000 sub-steps/s per fast core",
    4: "85-150 (RK4) and 160-250 (implicitfast) decisions/s, one process",
    5: "750-1,300 decisions/s on six fast cores; +15-35% from the efficiency cores",
    6: "SB3 CPU 70-300, SBX 150-500, SB3 MPS 50-200 updates/s; near-linear across fast cores",
    7: "2.5-5 TFLOPS achieved on 4096^2 (MPS / MLX)",
    8: "15-30 us fixed cost per small GPU job (GRU cell)",
    11: "MPS available in which launch contexts",
}
NOT_AUTOMATED = {
    9: "Dreamer grad steps/s after 500 warm-up, dummy 30-number world, pre-filled replay: "
       "official JAX (CPU), sheeprl dreamer_v3_5_1M on CPU and MPS (PYTORCH_ENABLE_MPS_FALLBACK=0), "
       "then _12M for the two fastest; same script on the T4. Needs the J5 pins "
       "(experiments/brain_a/README.md).",
    10: "2,000 sheeprl steps, fixed seed and data, CPU vs MPS: compare loss curves and per-module "
        "gradient norms; any divergence bars MPS from decisive runs (PyTorch issue 204359).",
    11: "2,000 renders of a 64x64 eye frame logged in, over SSH, from a LaunchAgent and from a "
        "LaunchDaemon after a no-login reboot; run this script with --only 11 in each context.",
    12: "24 h soak: 6 world processes (--only 5 --procs 6 --seconds 86400) + 1 Dreamer learner on MPS, "
        "behind a wall meter; log vm_stat, memory_pressure, powermetrics each minute.",
}


def _one_thread_env(n: int = 1) -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS_THREADS"):
        env[k] = str(n)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _try_version(mod: str):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        return f"absent ({type(e).__name__})"


def _cmd(args, timeout=20):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


class Skip(Exception):
    pass


# ---------------------------------------------------------------- measurements
def m1(a):
    sysname = platform.system()
    rec = {"system": sysname, "release": platform.release(), "machine": platform.machine(),
           "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
           "versions": {k: _try_version(k) for k in
                        ("numpy", "mujoco", "torch", "stable_baselines3", "sbx", "jax", "mlx",
                         "gymnasium")}}
    if sysname == "Darwin":
        levels = {}
        for lv in range(4):
            out = _cmd(["sysctl", "-n", f"hw.perflevel{lv}.physicalcpu", f"hw.perflevel{lv}.name"])
            if out and "unavailable" not in out and out.strip():
                levels[f"perflevel{lv}"] = out.split()
        rec.update(perflevels=levels, macos=platform.mac_ver()[0],
                   cpu_brand=_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
                   memsize=_cmd(["sysctl", "-n", "hw.memsize"]), pmset=_cmd(["pmset", "-g"]))
    elif sysname == "Linux":
        rec["lscpu"] = _cmd(["lscpu"])[:2000]
    return rec


def _humanoid_xml():
    try:
        import gymnasium
        p = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "humanoid.xml"
        if p.exists():
            return p
    except Exception:  # noqa: BLE001
        pass
    raise Skip("no humanoid.xml (install gymnasium[mujoco])")


def _humanoid_worker(xml, steps, q):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    t = time.perf_counter()
    mujoco.mj_step(m, d, nstep=steps)
    q.put(steps / (time.perf_counter() - t))


def _parallel(fn, args, n):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # macOS default; same on the box
    q = ctx.Queue()
    ps = [ctx.Process(target=fn, args=(*args, q)) for _ in range(n)]
    t = time.perf_counter()
    for p in ps:
        p.start()
    rates = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return {"procs": n, "per_proc": [round(r, 1) for r in rates], "sum": round(sum(rates), 1),
            "wall_s": round(time.perf_counter() - t, 2)}


def m2(a):
    xml = _humanoid_xml()
    rec = {"model": str(xml)}
    tb = os.environ.get("MUJOCO_TESTSPEED") or _which("testspeed")
    if tb:  # the release-build binary §10 names
        rec["testspeed"] = {n: _cmd([tb, str(xml), "10000", str(n)], timeout=600)[-600:]
                            for n in a.procs}
    else:
        rec["testspeed"] = "SKIPPED: no testspeed binary (set MUJOCO_TESTSPEED); python mj_step used"
    steps = 5000 if a.quick else 50000
    rec["python_mj_step"] = [_parallel(_humanoid_worker, (xml, steps), n) for n in a.procs]
    return rec


def _which(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(d) / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def _jack_world():
    sys.path.insert(0, str(REPO))
    from experiments.jack_env import JackEnv
    env = JackEnv(j0=1.0, alpha=1.0)
    env.reset(seed=153)
    return env


def m3(a):
    import copy
    import mujoco
    env = _jack_world()
    n = 4000 if a.quick else 40000
    out = {}
    for name, integ in (("Euler", 0), ("RK4", 1), ("implicitfast", 3)):
        m = copy.deepcopy(env.w.model)
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = env.w.data.qpos, env.w.data.qvel
        m.opt.integrator = integ
        mujoco.mj_forward(m, d)
        t = time.perf_counter()
        mujoco.mj_step(m, d, nstep=n)
        out[name] = round(n / (time.perf_counter() - t), 1)
    return {"substeps": n, "substeps_per_s": out, "shipped_integrator": int(env.w.model.opt.integrator)}


LOOP_ARMS = {"shipped_rk4": {}, "implicitfast": {"integrator": "implicitfast"},
             "implicitfast+lazy_audio": {"integrator": "implicitfast", "lazy_contact_audio": True}}


def _loop_worker(kw, seconds, seed, q):
    sys.path.insert(0, str(REPO))
    import numpy as np
    from experiments.jack_env import JackEnv
    env = JackEnv(j0=1.0, alpha=1.0, **kw)
    env.reset(seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(20):
        env.step(rng.uniform(-1, 1, env.action_space.shape[0]))
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds:
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape[0]))
        n += 1
        if term or trunc:
            env.reset()
    q.put(n / (time.perf_counter() - t))


def m4(a):
    s = 10 if a.quick else 60
    return {"seconds": s, "note": "watch core placement: sudo powermetrics --samplers cpu_power",
            "decisions_per_s": {arm: _parallel(_loop_worker, (kw, s, 153), 1)["sum"]
                                for arm, kw in LOOP_ARMS.items()}}


def m5(a):
    s = a.seconds or (10 if a.quick else 60)
    return {"seconds": s, "ladder": {arm: [_parallel(_loop_worker, (kw, s, 153), n) for n in a.procs]
                                     for arm, kw in LOOP_ARMS.items() if arm != "implicitfast"}}


def _sac_worker(lib, device, threads, warm, timed, q):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import numpy as np
    import gymnasium as gym
    if lib == "sb3":
        import torch
        torch.set_num_threads(threads)
        from stable_baselines3 import SAC
    else:
        # JAX ignores OMP; without this one SBX worker spreads over all cores and
        # the 1..6-runs ladder measures contention, not scaling (box: 2 procs 1.28x).
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   " --xla_cpu_multi_thread_eigen=false"
                                   " intra_op_parallelism_threads=1").strip()
        from sbx import SAC
    obs_dim, act_dim = 64, 8  # synthetic; 2x256 nets, batch 256 as §10 row 6

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (obs_dim,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (act_dim,), np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, False, {}

    kw = dict(policy_kwargs=dict(net_arch=[256, 256]), batch_size=256, learning_starts=0,
              buffer_size=10000, verbose=0, seed=0)
    if lib == "sb3":
        kw["device"] = device
    model = SAC("MlpPolicy", Dummy(), **kw)
    model.learn(total_timesteps=1000)  # fills replay with 1,000 synthetic transitions
    model.train(gradient_steps=warm, batch_size=256)
    t = time.perf_counter()
    model.train(gradient_steps=timed, batch_size=256)
    q.put(timed / (time.perf_counter() - t))


def m6(a):
    warm, timed = (50, 200) if a.quick else (200, 2000)
    rec = {"warmup": warm, "timed": timed, "single": {}, "concurrent": {}}
    arms = [("sb3", "cpu", 1), ("sb3", "cpu", 4), ("sbx", "cpu", 1), ("sb3", "mps", 1)]
    for lib, dev, th in arms:
        key = f"{lib}_{dev}_t{th}"
        try:
            if lib == "sbx":
                __import__("sbx")
            else:
                import torch
                if dev == "mps" and not (hasattr(torch.backends, "mps")
                                         and torch.backends.mps.is_available()):
                    raise Skip("MPS not available")
            rec["single"][key] = _parallel(_sac_worker, (lib, dev, th, warm, timed), 1)["sum"]
        except Skip as e:
            rec["single"][key] = f"SKIPPED: {e}"
        except Exception as e:  # noqa: BLE001
            rec["single"][key] = f"SKIPPED: {type(e).__name__}: {e}"[:200]
    best = "sbx" if isinstance(rec["single"].get("sbx_cpu_t1"), float) else "sb3"
    rec["concurrent"][f"{best}_cpu_t1"] = [_parallel(_sac_worker, (best, "cpu", 1, warm, timed), n)
                                           for n in a.procs]
    return rec


def _time(fn, reps):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def m7(a):
    n = 1024 if a.quick else 4096
    flop = 2 * n ** 3
    rec = {"n": n}
    import numpy as np
    x = np.random.rand(n, n).astype(np.float32)
    rec["numpy_cpu_f32_tflops"] = round(flop / _time(lambda: x @ x, 3) / 1e12, 3)
    try:
        import torch
        if torch.backends.mps.is_available():
            for dt in (torch.float32, torch.float16):
                y = torch.rand(n, n, device="mps", dtype=dt)
                rec[f"mps_{str(dt)[6:]}_tflops"] = round(
                    flop / _time(lambda: (y @ y, torch.mps.synchronize()), 10) / 1e12, 3)
        else:
            rec["mps"] = "SKIPPED: MPS not available"
    except ImportError:
        rec["mps"] = "SKIPPED: no torch"
    try:
        import mlx.core as mx
        for dt in (mx.float32, mx.float16):
            y = mx.random.uniform(shape=(n, n)).astype(dt)
            rec[f"mlx_{dt}_tflops"] = round(flop / _time(lambda: mx.eval(y @ y), 10) / 1e12, 3)
    except ImportError:
        rec["mlx"] = "SKIPPED: no mlx"
    return rec


def m8(a):
    reps = 20 if a.quick else 200
    rec = {"hidden": 256, "input": 64, "reps": reps}
    try:
        import torch
    except ImportError:
        raise Skip("no torch")
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        cell = torch.nn.GRUCell(64, 256).to(dev)
        for b in (16, 1024):
            x = torch.randn(b, 64, device=dev)
            h = torch.randn(b, 256, device=dev)

            def step():
                cell(x, h).sum().backward()
                if dev == "mps":
                    torch.mps.synchronize()
            rec[f"torch_{dev}_b{b}_us"] = round(_time(step, reps) * 1e6, 1)
    if "mps" not in devs:
        rec["mps"] = "SKIPPED: MPS not available"
    try:
        import mlx.core as mx
        import mlx.nn as mnn
        cell = mnn.GRU(64, 256)
        for b in (16, 1024):
            x = mx.random.normal((b, 1, 64))
            g = mx.grad(lambda xx: cell(xx).sum())
            rec[f"mlx_b{b}_us"] = round(_time(lambda: mx.eval(g(x)), reps) * 1e6, 1)
    except ImportError:
        rec["mlx"] = "SKIPPED: no mlx"
    return rec


def m11(a):
    ctx = ("ssh" if os.environ.get("SSH_CONNECTION") else
           "launchd" if os.environ.get("XPC_SERVICE_NAME", "0") not in ("0", "") else "terminal")
    rec = {"context": ctx, "user": os.environ.get("USER")}
    try:
        import torch
        rec["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        rec["mps_available"] = "SKIPPED: no torch"
    return rec


MEASURE = {1: m1, 2: m2, 3: m3, 4: m4, 5: m5, 6: m6, 7: m7, 8: m8, 11: m11}


def child(k, a):
    try:
        print(json.dumps({"status": "RAN", "result": MEASURE[k](a)}))
    except Skip as e:
        print(json.dumps({"status": "SKIPPED", "reason": str(e)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma list of measurement numbers")
    ap.add_argument("--quick", action="store_true", help="short timings: a smoke, not a reading")
    ap.add_argument("--procs", default="1,2,4,6,8,12")
    ap.add_argument("--seconds", type=float, default=0, help="measurement 5 duration override")
    ap.add_argument("--out", default=".")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--child", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.procs = [int(x) for x in a.procs.split(",") if x]
    if a.child:
        return child(a.child, a)
    want = [int(x) for x in a.only.split(",") if x] or sorted(set(MEASURE) | set(NOT_AUTOMATED))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report = {"host": socket.gethostname(), "utc": stamp, "argv": sys.argv,
              "quick": a.quick, "measurements": {}}
    path = Path(a.out) / f"first_day_{socket.gethostname()}_{stamp}.json"
    for k in want:
        if k in NOT_AUTOMATED and k not in MEASURE:
            rec = {"status": "NOT_AUTOMATED", "do_by_hand": NOT_AUTOMATED[k]}
        else:
            t0 = time.perf_counter()
            args = [sys.executable, str(Path(__file__).resolve()), "--child", str(k),
                    "--procs", ",".join(map(str, a.procs)), "--seconds", str(a.seconds)]
            if a.quick:
                args.append("--quick")
            try:
                p = subprocess.run(args, capture_output=True, text=True, timeout=a.timeout,
                                   env=_one_thread_env(), cwd=str(REPO))
                line = (p.stdout.strip().splitlines() or [""])[-1]
                rec = json.loads(line) if p.returncode == 0 and line.startswith("{") else \
                    {"status": "FAILED", "rc": p.returncode, "stderr": p.stderr[-1500:]}
            except subprocess.TimeoutExpired:
                rec = {"status": "FAILED", "reason": f"timeout {a.timeout}s"}
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            if k in NOT_AUTOMATED:
                rec["also_by_hand"] = NOT_AUTOMATED[k]
        rec["estimate_replaced"] = ESTIMATE.get(k, "")
        report["measurements"][str(k)] = rec
        path.write_text(json.dumps(report, indent=1))  # written after every row
        print(f"[{k}] {rec['status']}: " + json.dumps(rec.get("result", rec.get("reason", rec.get("do_by_hand", rec.get("stderr", "")))))[:400],
              flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
