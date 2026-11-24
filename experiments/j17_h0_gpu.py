"""J17 hunt h0 (SimBa) on a Kaggle GPU: the TIMING cell first (ORDERS J17, 11-24).

Why: h0 on Kaggle CPU was VOID — 1.12 decisions/s per cell, 100k ~ 25 h
(`WORLD_CARD.md` § J17 hunt). SimBa (Lee et al. 2024) is a JAX learner whose
published runs are on accelerators; sbx 0.28.0's SimbaPolicy is the build.
ORDERS: "time 5k decisions first and launch only if 100k fits 8 h".

The cell: arm `H-simba-row128` (the sighted h0 arm, pilot_b2.J17_HUNT_ARMS,
unchanged), seed 923, PILOT_RULE's world, run in a child process per platform
(JAX_PLATFORMS=cuda; CPU's rate is already known, 1.12/s), each for
T = 5,000 (= pilot_b.WARMUP: no gradient step yet, env + acting only) and
T = 10,000 (5,000 LEARNING decisions). Learning rate = 5,000 / (t10k - t5k).
The read, written before the kernel runs:
  * fits  — 95k learning decisions at the GPU rate + the warmup <= 8 h for
            ONE cell; the pair (sighted + blind) is then two processes on one
            GPU, so the launch also needs the pair rate (2 children at once).
  * does not fit -> INBOX line, no h0 launch on GPU (branch 4 then rests on
            3 of 4 arms + h0 VOID on every free machine).
No row, no ledger: a timing, not a reading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/pilot_b2/j17_h0_gpu_timing.json")
ARM = "H-simba-row128"
BLIND = "H-simba-row128-blind"
SEED = 923
BUDGET_S = 8 * 3600

_JOB = r"""
import subprocess as _s, sys as _y, os, json, time
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
# 11-24 10:26 first try (Colab, jax 0.11.1 preinstalled): sbx's actor.init died in
# flax -- `jax.core.get_opaque_trace_state` removed in JAX 0.11. Pin the trio the
# box's jack-std venv runs this arm with (jax 0.11.2, flax 0.12.12, optax 0.2.9).
jv = "0.11.2"
inst = _s.run([_y.executable, "-m", "pip", "install", "-q", "jax[cuda12]==" + jv,
               "flax==0.12.12", "optax==0.2.9"], capture_output=True, text=True)
res = {"jax": jv, "cuda_install_rc": inst.returncode, "cuda_install_err": inst.stderr[-1500:]}
os.chdir("/tmp/jack")
CHILD = r'''
import json, sys, time
t0 = time.time()
import jax
dev = [str(d) for d in jax.devices()]
from experiments.pilot_b2 import run_pilot_cell
t = time.time()
r = run_pilot_cell(sys.argv[2], __SEED__, True, int(sys.argv[1]), eye_frame="body",
                   motor_frame="body", native_substep=True)
print("RESULT " + json.dumps({"devices": dev, "wall_s": {sys.argv[2]: time.time() - t},
                              "lives": len(r.get("lives", [])),
                              "total_s": time.time() - t0}), flush=True)
'''
open("/tmp/j17_child.py", "w").write(CHILD)
def run(plat, T, arms):
    env = dict(os.environ, JAX_PLATFORMS=plat, PYTHONPATH="/tmp/jack",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t = time.time()
    ps = [_s.Popen([_y.executable, "/tmp/j17_child.py", str(T), a], env=env,
                   stdout=_s.PIPE, stderr=_s.PIPE, text=True) for a in arms]
    outs = [p.communicate(timeout=3 * 3600) for p in ps]
    wall, devs, errs = {}, None, []
    for p, (o, e) in zip(ps, outs):
        line = [l for l in o.splitlines() if l.startswith("RESULT ")]
        if line:
            d = json.loads(line[-1][7:]); wall.update(d["wall_s"]); devs = d["devices"]
        if p.returncode:
            errs.append(e[-2000:])
    return {"rc": max(p.returncode for p in ps), "outer_s": time.time() - t,
            "result": {"devices": devs, "wall_s": wall} if wall else None, "err": errs}
def save():
    json.dump(res, open(os.path.join(os.environ["JACK_OUT"], "j17_h0_gpu_timing.json"), "w"), indent=1)
# CPU is NOT re-timed here: its rate is known (1.12 decisions/s, Kaggle CPU,
# h0 VOID 11-22) and would cost ~75 GPU-min for 5k learning decisions.
for T in (5000, 10000):
    k = f"cuda_T{T}_single"
    res[k] = run("cuda", T, ["__ARM__"])
    print(k, json.dumps(res[k])[:800], flush=True); save()
if res["cuda_T10000_single"]["rc"] == 0:
    res["cuda_T10000_pair"] = run("cuda", 10000, ["__ARM__", "__BLIND__"])
    print("cuda_T10000_pair", json.dumps(res["cuda_T10000_pair"])[:800], flush=True); save()
print("DONE", flush=True)
"""


def job_body() -> str:
    from experiments.pilot_b import KERNEL_PINS
    return (_JOB.replace("__PINS__", repr(list(KERNEL_PINS)))
            .replace("__SEED__", str(SEED)).replace("__ARM__", ARM)
            .replace("__BLIND__", BLIND))


def learn_rate(r: dict, key5: str, key10: str, arm: str = ARM) -> float | None:
    """Learning decisions/s = 5,000 / (wall at 10k - wall at 5k)."""
    try:
        a = r[key5]["result"]["wall_s"][arm]
        b = r[key10]["result"]["wall_s"][arm]
    except (KeyError, TypeError):
        return None
    return 5000.0 / (b - a) if b > a else None


def read(r: dict) -> dict:
    """The pre-written read: does one 100k cell (and the pair) fit 8 h?"""
    out = {}
    for plat in ("cuda",):
        lr = learn_rate(r, f"{plat}_T5000_single", f"{plat}_T10000_single")
        try:
            warm = r[f"{plat}_T5000_single"]["result"]["wall_s"][ARM]
        except (KeyError, TypeError):
            warm = None
        est = (warm + 95_000 / lr) if (lr and warm is not None) else None
        out[plat] = {"learn_dps": lr, "warmup_s": warm, "est_100k_h": est and est / 3600,
                     "fits_8h": bool(est and est <= BUDGET_S)}
    try:
        pair = r["cuda_T10000_pair"]["result"]["wall_s"]
        w5 = r["cuda_T5000_single"]["result"]["wall_s"][ARM]
        worst = max(pair.values())
        lr2 = 5000.0 / (worst - w5)          # warmup assumed unchanged by pairing
        est2 = w5 + 95_000 / lr2
        out["cuda_pair"] = {"learn_dps": lr2, "est_100k_h": est2 / 3600,
                            "fits_8h": est2 <= BUDGET_S}
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        out["cuda_pair"] = None
    return out


def launch() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle", est_hours=1.0,
                 timeout_s=4 * 3600, fetch=["j17_h0_gpu_timing.json"])
    if not (sub.ok and "j17_h0_gpu_timing.json" in sub.artifacts):
        raise RuntimeError(f"j17 h0 GPU timing: {sub.message}")
    r = result_json(sub, "j17_h0_gpu_timing.json")
    r["read"] = read(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["read"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:] == ["launch"]:
        launch()
    elif sys.argv[1:] == ["read"]:
        print(json.dumps(read(json.loads(OUT.read_text())), indent=1))
    elif sys.argv[1:] != ["pair"]:
        print("usage: python -m experiments.j17_h0_gpu launch|read|pair")


# ---- the h0 pair itself (11-24 10:40, after the timing read: cuda single
# 42.6 learning decisions/s, pair 24.0/s -> 100k ~ 1.1 h, fits 8 h). The SAME
# j17_hunt_kernel cells and world (PILOT_RULE, T = J17_HUNT_T, seed 923, both
# arms in one job = one machine), only the platform differs: JAX on cuda, the
# timing cell's pinned trio, no XLA preallocation (two processes share one GPU).
# Written to its OWN file, so the CPU VOID file j17_hunt_h0.json is kept.
PAIR_OUT = Path("/data/jack-logs/pilot_b2/j17_hunt_h0_gpu.json")
_GPU_ENV = (
    'import os as _o; _o.environ["JAX_PLATFORMS"] = "cuda"; '
    '_o.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"\n')
_TRIO = ('_s.run([_y.executable, "-m", "pip", "install", "-q", "jax[cuda12]==0.11.2", '
         '"flax==0.12.12", "optax==0.2.9"], check=True)\n')


def pair_body() -> str:
    from experiments.pilot_b2 import (J17_HUNT_GROUPS, J17_HUNT_SEED, J17_HUNT_T,
                                      PILOT_RULE, pilot_job_body)
    cells = [(a, J17_HUNT_SEED) for a in J17_HUNT_GROUPS["h0"]]
    rule = ("J17 critic hunt PILOT h0 on GPU 11-24 (CPU run VOID 11-22), SimBa "
            "(PUBLISHED_PRIORS §24), 1 seed, no verdict; world = " + PILOT_RULE["rule"])
    body = pilot_job_body(cells, PILOT_RULE["food_floor_only"], J17_HUNT_T, rule,
                          PILOT_RULE["eye_frame"], PILOT_RULE["motor_frame"],
                          PILOT_RULE["native_substep"])
    first = body.index("\n", body.index("check=True)")) + 1   # after the pins install
    return _GPU_ENV + body[:first] + _TRIO + body[first:]


def launch_pair() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(pair_body()), prefer="kaggle", est_hours=2.0,
                 timeout_s=int(3.9 * 3600), fetch=["pilot_b2_cells.json"])
    if not (sub.ok and "pilot_b2_cells.json" in sub.artifacts):
        raise RuntimeError(f"j17 h0 GPU pair: {sub.message}")
    r = result_json(sub, "pilot_b2_cells.json")
    PAIR_OUT.write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives" + (" (cut)" if rec.get("cut") else ""), flush=True)
    return r


if __name__ == "__main__" and sys.argv[1:] == ["pair"]:
    launch_pair()
