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

RUNGS 3/6/7 (11-21, J16's "a wheel re-check is one < 1 h Kaggle CPU kernel a
rung"): `kernel 367` runs rung3/rung6/rung7 and each `--planted` the same way,
result `wheel_367.json`; same verdict rule (PASS iff all six exit 0 with their
line, n == m, and mlx 0.32.4 printed by each rung line). Default `kernel` is
byte-identical to the rung 1/2 run above.

Usage:  python -m experiments.brain_a.wheel_kernel kernel [367|s1m|9]  (submits; writes OUT/wheel[_367|_s1m].json)
        python -m experiments.brain_a.wheel_kernel verdict FILE [367]
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
RUNS_367 = [(f"rung{k}{t}", f"rung{k}.py", a, 1200) for k in (3, 6, 7)
            for t, a in (("", []), ("_planted", ["--planted"]))]
_PL = r"PLANTED: (\d+)/(\d+) caught, (?:control|unpatched port) passes -> OK"
EXPECT.update({f"rung{k}": rf"RUNG {k}: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS"
               for k in (3, 6, 7)})
EXPECT.update({f"rung{k}_planted": _PL for k in (3, 6, 7)})
# J21 (11-25): the size1m rungs since 367 — rung 7 at --preset size1m, then the world-model loss
# rungs 8a (dyn/rep free bits), 8b (reward/continue), 8c (decoder vector). Same pin, same shape of
# check (rc 0, n == m, mlx 0.32.4 on each rung line). Box timing 11-25: all 8 runs <= 4 s each.
RUNS_S1M = ([("rung7s1m", "rung7.py", ["--preset", "size1m"], 1200),
             ("rung7s1m_planted", "rung7.py", ["--preset", "size1m", "--planted"], 1200)]
            + [(f"rung{k}{t}", f"rung{k}.py", a, 1200) for k in ("8a", "8b", "8c")
               for t, a in (("", []), ("_planted", ["--planted"]))])
EXPECT["rung7s1m"] = EXPECT["rung7"]
EXPECT["rung7s1m_planted"] = _PL
EXPECT.update({f"rung{k}": rf"RUNG {k}: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS"
               for k in ("8a", "8b", "8c")})
EXPECT.update({f"rung{k}_planted": _PL for k in ("8a", "8b", "8c")})
# J21 (11-25): rung 9 — the FULL world-model loss (L240 sum(mean x scale)) + its gradient-routing
# test, and --planted (5 bugs incl. the forward-blind dyn<->rep swap). Same pin, same check shape.
# Box timing 11-25 (conda-forge mlx 0.32.4): both runs a few s each.
RUNS_9 = [("rung9", "rung9.py", [], 1200), ("rung9_planted", "rung9.py", ["--planted"], 1200)]
EXPECT["rung9"] = r"RUNG 9: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS"
EXPECT["rung9_planted"] = _PL
SETS = {"12": (RUNS, "wheel.json"), "367": (RUNS_367, "wheel_367.json"),
        "s1m": (RUNS_S1M, "wheel_s1m.json"), "9": (RUNS_9, "wheel_9.json")}

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


def job_body(which: str = "12") -> str:
    runs, fname = SETS[which]
    body = _JOB.replace("__PIN__", repr(PIN)).replace("__RUNS__", repr(runs))
    return body if which == "12" else body.replace('"wheel.json"', repr(fname))


def verdict(res: dict, which: str = "12") -> dict:
    runs = res.get("runs", {})
    lines = {}
    for key, *_ in SETS[which][0]:
        r = runs.get(key)
        if r is None:
            return {"WHEEL": "NO-READING", "why": f"{key} missing (kernel died before it)"}
        m = re.search(EXPECT[key], r.get("stdout_tail", ""))
        if r.get("rc") != 0 or not m or m.group(1) != m.group(2):
            return {"WHEEL": "FAIL", "first_differing": key, "rc": r.get("rc"),
                    "stdout_tail": r.get("stdout_tail", "")[-1500:],
                    "stderr_tail": r.get("stderr_tail", "")[-800:]}
        if not key.endswith("_planted") and m.group(3) != VERSION:
            return {"WHEEL": "FAIL", "first_differing": key,
                    "why": f"mlx {m.group(3)} ran, not the pinned {VERSION}"}
        lines[key] = m.group(0)
    return {"WHEEL": "PASS", "lines": lines, "machine": res.get("machine"),
            "libc": res.get("libc"), "python": res.get("python")}


def kernel(which: str = "12") -> dict:
    from experiments.gpu import build_job, result_json, submit
    fname = SETS[which][1]
    sub = submit(build_job(job_body(which)), prefer="kaggle-cpu", est_hours=0.3,
                 timeout_s=int(2 * 3600), fetch=[fname])
    if not sub.ok:
        raise RuntimeError(f"wheel_kernel: {sub.message}")
    r = result_json(sub, fname)
    OUT.mkdir(parents=True, exist_ok=True)
    r["verdict"] = verdict(r, which)
    (OUT / fname).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["verdict"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel(sys.argv[2] if len(sys.argv) > 2 else "12")
    elif sys.argv[1:2] == ["verdict"]:
        print(json.dumps(verdict(json.loads(Path(sys.argv[2]).read_text()),
                                 sys.argv[3] if len(sys.argv) > 3 else "12"), indent=1))
    else:
        raise SystemExit(__doc__)
