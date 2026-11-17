"""MLX port stage 0, steps 0.2a + 0.3 Kaggle half: decide gate G0 on one Kaggle CPU kernel.

Source: docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md, stage 0 and gate G0
(thresholds are that report's choice, not published, and are NOT changed here):

    G0 PASS  iff  size12m, mx.compile, cold cache: first step (trace + compile + run) < 600 s
                  AND peak MLX memory < 8 GB (8192 MB), with finite loss/grad-norm.
    G0 FAIL  otherwise -> the report's fallback (compile one time step, loop in Python) must be
                  measured next; a failed fallback stops the port.

The kernel (x86, 4 cores, PyPI `mlx[cpu]==0.32.4` — the box uses conda-forge because the PyPI
aarch64 wheel needs a newer glibc) runs, in order:
  1. `mlx_smoke.py` (0.1's eight checks; the PyPI binary's traps may differ from conda-forge's);
  2. `mlx_spike.py --size 1m` compiled and eager (cross-check against the box: 37.1 s cold,
     2.34 s steady, 341 MB on 1 ARM core);
  3. `mlx_spike.py --size 12m` compiled (G0) and eager (the no-compile reference).
Each run is a fresh process (cold JIT cache for the first compiled run of each size; TMPDIR is
fresh on a Kaggle session). Every run has its own timeout so one hang cannot eat the others.

NOT in this kernel: 0.2b (official dreamerv3@e01491f `debug` preset, 200 steps, on a 74-in/8-out
dummy world). It needs a dummy-env wrapper for the official driver that is not written yet; it
is the next step and can share a later session with 0.4 (golden generator v0).

Usage:  python -m experiments.brain_a.g0_kernel kernel     (submits; writes /data/jack-logs/mlx/g0.json)
        python -m experiments.brain_a.g0_kernel verdict FILE
"""
import json
import sys
from pathlib import Path

PIN = "mlx[cpu]==0.32.4"
OUT = Path("/data/jack-logs/mlx")
G0_COMPILE_S = 600.0
G0_PEAK_MB = 8192.0
RUNS = [("smoke", None, None, 900), ("1m", "1m", "compiled", 1800), ("1m", "1m", "eager", 1800),
        ("12m", "12m", "compiled", 3 * 3600), ("12m", "12m", "eager", 3 * 3600)]

_JOB = r'''
import subprocess as _s, sys as _y, json, os, platform, time
_s.run([_y.executable, "-m", "pip", "install", "-q", __PIN__], check=True)
os.chdir("/tmp/jack")
out_dir = os.environ["JACK_OUT"]
res = {"python": platform.python_version(), "machine": platform.machine(),
       "cpus": os.cpu_count(), "runs": {}}
for tag, size, mode, tmo in __RUNS__:
    key = "smoke" if size is None else f"{size}_{mode}"
    if size is None:
        cmd = [_y.executable, "experiments/brain_a/mlx_smoke.py"]
    else:
        cmd = [_y.executable, "experiments/brain_a/mlx_spike.py", "--size", size, "--mode", mode,
               "--steps", "6", "--out", os.path.join(out_dir, f"spike_{key}.json")]
    t0 = time.time()
    try:
        p = _s.run(cmd, capture_output=True, text=True, timeout=tmo)
        rec = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1),
               "stdout_tail": p.stdout[-6000:], "stderr_tail": p.stderr[-3000:]}
    except _s.TimeoutExpired:
        rec = {"rc": "TIMEOUT", "wall_s": round(time.time() - t0, 1), "timeout_s": tmo}
    f = os.path.join(out_dir, f"spike_{key}.json")
    if size is not None and os.path.exists(f):
        rec["spike"] = json.load(open(f))
    res["runs"][key] = rec
    print(key, rec.get("rc"), rec["wall_s"], flush=True)
    json.dump(res, open(os.path.join(out_dir, "g0.json"), "w"))
print("DONE", flush=True)
'''


def job_body() -> str:
    return _JOB.replace("__PIN__", repr(PIN)).replace("__RUNS__", repr(RUNS))


def verdict(res: dict) -> dict:
    r = res.get("runs", {}).get("12m_compiled", {})
    s = r.get("spike")
    if not s:
        return {"G0": "FAIL" if r.get("rc") == "TIMEOUT" else "NO-READING",
                "why": f"12m compiled rc={r.get('rc')}, no spike record"}
    ok = s["first_step_s"] < G0_COMPILE_S and s["peak_mb"] < G0_PEAK_MB and s["finite"]
    return {"G0": "PASS" if ok else "FAIL", "first_step_s": s["first_step_s"],
            "peak_mb": s["peak_mb"], "steady_median_s": s["steady_median_s"],
            "params": s["params"], "finite": s["finite"],
            "rule": f"first_step < {G0_COMPILE_S} s and peak < {G0_PEAK_MB} MB"}


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=int(9.5 * 3600), fetch=["g0.json"])
    if not sub.ok:
        raise RuntimeError(f"g0_kernel: {sub.message}")
    r = result_json(sub, "g0.json")
    OUT.mkdir(parents=True, exist_ok=True)
    r["verdict"] = verdict(r)
    (OUT / "g0.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["verdict"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif sys.argv[1:2] == ["verdict"]:
        print(json.dumps(verdict(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    else:
        raise SystemExit(__doc__)
