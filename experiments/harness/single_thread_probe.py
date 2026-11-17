"""J11 item (4): why did brain B use ~2.4 cores under `_single_thread`?

INBOX 11-16 19:40: check8_fixed_point's probe ran at ~2.4 cores although
`pilot_b._single_thread()` was called. This probe measures, it does not argue.

Each variant runs in a FRESH child process (threading env is read at import):
pilot_b's own `run_b` path (world, VecNormalize, SBX SAC, gSDE), with WARMUP
cut to 300 and T = 1,300, so ~1,000 decisions are learning decisions (32
gradient steps every 32 decisions). It reports cores = process CPU-s / wall-s
over `model.learn`, and the CPU time of every thread grouped by thread name
(/proc/self/task/*/{comm,stat}), so the extra cores are attributed to a pool.

Variants:
  inherited  the shell's env as is (this box exports OMP_NUM_THREADS=2,
             MKL_NUM_THREADS=2), `_single_thread` as shipped.
  clean      OMP/MKL/XLA vars removed before the child starts, as shipped.
  clean      the XLA/OMP vars set in the child's env from its start (proves
             it is not a setdefault-vs-inherited problem).
  pinned     the fix: `_single_thread(pin_cpu=3)` (affinity before jax).
  taskset1   control: inherited env, pinned to one CPU with taskset — must
             read <= ~1.05 cores, else the meter is wrong.
Each child also prints a fingerprint of the learned actor and the lives, so
pinned vs unpinned is shown identical or different (not assumed).

RESULT 11-17 13:28 (box, jax 0.11.2, seed 401, 1,312 decisions, ~1,000 learning):
  variant    cores  learn wall-s  learn CPU-s  fingerprint (actor sum / lives sha)
  inherited  1.60   34.7          55.4         -257.40708004548327 / 509c0df6616c
  clean      1.54   35.7          54.8         same
  pinned     1.00   50.7          50.5         same
  taskset1   1.00   50.7          50.5         same   (control reads 1.00: meter OK)
  Threads (inherited): `tf_XLAEigen` x4 = 29.4 CPU-s of 64 (46%); pinned: x1.
  Reading: the XLA_FLAGS do nothing in jax 0.11 even when set from process
  start (`clean`); the pool follows the affinity mask. Pinning makes every
  learned weight and every life bit-identical, costs 1.46x the wall and
  saves 9% of the CPU. Not run: a fingerprint control (another seed) — the
  fingerprint's power to differ is assumed, not shown.

Run: nice -n 19 /data/venvs/jack-std/bin/python -m experiments.harness.single_thread_probe
Out: /data/jack-logs/harness/single_thread_probe.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = Path("/data/jack-logs/harness/single_thread_probe.json")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "XLA_FLAGS", "JAX_PLATFORMS")


def _threads() -> dict:
    tick = os.sysconf("SC_CLK_TCK")
    by = {}
    for t in Path("/proc/self/task").iterdir():
        try:
            name = (t / "comm").read_text().strip()
            f = (t / "stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(f[11]) + int(f[12])) / tick
        except OSError:
            continue
        g = by.setdefault(name, {"n": 0, "cpu_s": 0.0})
        g["n"] += 1
        g["cpu_s"] = round(g["cpu_s"] + cpu, 2)
    return dict(sorted(by.items(), key=lambda kv: -kv[1]["cpu_s"]))


def child(pin: bool) -> dict:
    from experiments import pilot_b as pb
    if pin:
        pb._single_thread(pin_cpu=3)
    pb.WARMUP = 300
    env_seen = {k: os.environ.get(k) for k in THREAD_VARS}
    t0w, t0c = time.time(), time.process_time()
    import hashlib
    import numpy as np
    holder = {}
    real_make_b = pb.make_b

    def _keep(*a, **k):
        holder["m"] = real_make_b(*a, **k)
        return holder["m"]
    pb.make_b = _keep
    r = pb.run_b(seed=401, T=1300)
    import jax
    leaves = jax.tree_util.tree_leaves(holder["m"].policy.actor_state.params)
    flat = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in leaves])
    fp = {"actor_sum": float(flat.sum()), "actor_abs": float(np.abs(flat).sum()),
          "lives_sha": hashlib.sha1(json.dumps(r["lives"], sort_keys=True,
                                               default=str).encode()).hexdigest()[:12]}
    wall, cpu = time.time() - t0w, time.process_time() - t0c
    import torch
    return {"env_seen_after": env_seen, "torch_threads": torch.get_num_threads(),
            "learn_wall_s": r["wall_s"], "learn_cpu_s": r["cpu_s"],
            "learn_cores": round(r["cpu_s"] / max(r["wall_s"], 1e-9), 2),
            "total_cores": round(cpu / max(wall, 1e-9), 2),
            "decisions": r["decisions"], "fingerprint": fp, "threads": _threads()}


def run_variant(name: str) -> dict:
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "experiments.harness.single_thread_probe",
           "child", "pin" if name == "pinned" else "plain"]
    if name == "clean":
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu",
                    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false "
                                 "intra_op_parallelism_threads=1"})
    if name == "taskset1":
        cmd = ["taskset", "-c", "3"] + cmd
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=1800)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        return {"variant": name, "rc": p.returncode, "stderr": p.stderr[-2000:]}
    return {"variant": name, **json.loads(line[-1][7:])}


def main(argv):
    if argv[:1] == ["child"]:
        print("RESULT " + json.dumps(child(argv[1:2] == ["pin"])), flush=True)
        return 0
    names = argv or ["inherited", "clean", "pinned", "taskset1"]
    out = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "box_env": {k: os.environ.get(k) for k in THREAD_VARS},
           "variants": []}
    for n in names:
        r = run_variant(n)
        print(json.dumps({k: r.get(k) for k in ("variant", "learn_cores",
              "learn_wall_s", "learn_cpu_s", "fingerprint", "rc")}), flush=True)
        out["variants"].append(r)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
