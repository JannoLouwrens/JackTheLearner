"""J16 (d): rung 1 + rung 2 on the PyPI MLX wheel — one Kaggle CPU kernel.

Why a kernel: the parity rungs PASS here on conda-forge `mlx 0.32.4` (rung 1
9/9, rung 2 4/4, eager and compiled), but the Mac will run the PyPI wheel, and
the PyPI aarch64 wheel cannot load on this box (needs GLIBC 2.35 /
GLIBCXX 3.4.30; box has 2.34 / 3.4.29 — README stage 0.1). A Kaggle CPU
session (x86, Ubuntu 22.04, glibc 2.35) loads it. Different binary, same
version, same tests, unchanged: the report's rows are not edited here.

The kernel installs `mlx[cpu]==0.32.4` from PyPI and runs, each in a fresh
process with its own timeout:
  rung1          experiments/brain_a/rung1.py            exit 0 + "RUNG 1: 9/9 ... -> PASS"
  rung1_planted  experiments/brain_a/rung1.py --planted  exit 0 + "PLANTED: n/n caught -> OK"
  rung2          experiments/brain_a/rung2.py            exit 0 + "RUNG 2: 4/4 ... -> PASS"
  rung2_planted  experiments/brain_a/rung2.py --planted  exit 0 + "PLANTED: n/n caught -> OK"

WHEEL LINE (verdict, fixed before any run): PASS iff all four exit 0 with their
line printed AND the printed mlx version is 0.32.4 (so the box's conda-forge
build cannot be what ran); FAIL names the first differing run and its tail;
NO-READING when the result file lacks a run (the kernel died — not a FAIL).
No speed is read (only the Mac's speed counts, J16 prior).

Usage:  python -m experiments.brain_a.wheel_kernel kernel     (submits; writes OUT/wheel.json)
        python -m experiments.brain_a.wheel_kernel verdict FILE
"""
import json
import re
import sys
from pathlib import Path

PIN = "mlx[cpu]==0.32.4"
VERSION = "0.32.4"
OUT = Path("/data/jack-logs/mlx")
RUNS = [("rung1", "rung1.py", [], 1200), ("rung1_planted", "rung1.py", ["--planted"], 1200),
        ("rung2", "rung2.py", [], 1200), ("rung2_planted", "rung2.py", ["--planted"], 1200)]
EXPECT = {"rung1": r"RUNG 1: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS",
          "rung1_planted": r"PLANTED: (\d+)/(\d+) caught -> OK",
          "rung2": r"RUNG 2: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS",
          "rung2_planted": r"PLANTED: (\d+)/(\d+) caught -> OK"}

_JOB = r'''
import subprocess as _s, sys as _y, json, os, platform, time
_s.run([_y.executable, "-m", "pip", "install", "-q", __PIN__], check=True)
os.chdir("/tmp/jack")
out_dir = os.environ["JACK_OUT"]
tmp = "/tmp/jack_mlx_tmp"; os.makedirs(tmp, exist_ok=True)
env = dict(os.environ, TMPDIR=tmp)
res = {"python": platform.python_version(), "machine": platform.machine(),
       "libc": list(platform.libc_ver()), "cpus": os.cpu_count(), "runs": {}}
for key, script, args, tmo in __RUNS__:
    cmd = [_y.executable, os.path.join("experiments", "brain_a", script)] + args
    t0 = time.time()
    try:
        p = _s.run(cmd, capture_output=True, text=True, timeout=tmo, env=env)
        rec = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1),
               "stdout_tail": p.stdout[-8000:], "stderr_tail": p.stderr[-3000:]}
    except _s.TimeoutExpired:
        rec = {"rc": "TIMEOUT", "wall_s": round(time.time() - t0, 1), "timeout_s": tmo}
    res["runs"][key] = rec
    print(key, rec.get("rc"), rec["wall_s"], flush=True)
    json.dump(res, open(os.path.join(out_dir, "wheel.json"), "w"))
print("DONE", flush=True)
'''


def job_body() -> str:
    return _JOB.replace("__PIN__", repr(PIN)).replace("__RUNS__", repr(RUNS))


def verdict(res: dict) -> dict:
    runs = res.get("runs", {})
    lines = {}
    for key, *_ in RUNS:
        r = runs.get(key)
        if r is None:
            return {"WHEEL": "NO-READING", "why": f"{key} missing (kernel died before it)"}
        m = re.search(EXPECT[key], r.get("stdout_tail", ""))
        if r.get("rc") != 0 or not m or m.group(1) != m.group(2):
            return {"WHEEL": "FAIL", "first_differing": key, "rc": r.get("rc"),
                    "stdout_tail": r.get("stdout_tail", "")[-1500:],
                    "stderr_tail": r.get("stderr_tail", "")[-800:]}
        if key in ("rung1", "rung2") and m.group(3) != VERSION:
            return {"WHEEL": "FAIL", "first_differing": key,
                    "why": f"mlx {m.group(3)} ran, not the pinned {VERSION}"}
        lines[key] = m.group(0)
    return {"WHEEL": "PASS", "lines": lines, "machine": res.get("machine"),
            "libc": res.get("libc"), "python": res.get("python")}


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-cpu", est_hours=0.3,
                 timeout_s=int(2 * 3600), fetch=["wheel.json"])
    if not sub.ok:
        raise RuntimeError(f"wheel_kernel: {sub.message}")
    r = result_json(sub, "wheel.json")
    OUT.mkdir(parents=True, exist_ok=True)
    r["verdict"] = verdict(r)
    (OUT / "wheel.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["verdict"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif sys.argv[1:2] == ["verdict"]:
        print(json.dumps(verdict(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    else:
        raise SystemExit(__doc__)
