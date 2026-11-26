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

Usage:  python -m experiments.brain_a.wheel_kernel kernel [367|s1m|9|10_12|13]  (submits; writes OUT/wheel[_367|_s1m].json)
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
# J25 (11-25): rungs 10 (imagination + lambda-returns), 11 (critic loss), 12a (return normalisation
# + advantage), 12b (policy heads + loss), each + --planted. Same pin, same check shape. Box timing
# 11-25 (conda-forge mlx 0.32.4): 10/10, 10 planted 6/6, 11 4/4, 6/6, 12a 2/2, 5/5, 12b 3/3, 5/5; <= 5 s each.
RUNS_10_12 = [(f"rung{k}{t}", f"rung{k}.py", a, 1200) for k in ("10", "11", "12a", "12b")
              for t, a in (("", []), ("_planted", ["--planted"]))]
EXPECT.update({f"rung{k}": rf"RUNG {k}: (\d+)/(\d+) tests pass eager AND compiled \(mlx ([\w.]+)\) -> PASS"
               for k in ("10", "11", "12a", "12b")})
EXPECT.update({f"rung{k}_planted": _PL for k in ("10", "11", "12a", "12b")})
SETS = {"12": (RUNS, "wheel.json"), "367": (RUNS_367, "wheel_367.json"),
        "s1m": (RUNS_S1M, "wheel_s1m.json"), "9": (RUNS_9, "wheel_9.json"),
        "10_12": (RUNS_10_12, "wheel_10_12.json")}
# J25 (11-26): rung 13 on the wheel. Its goldens live only on /data (golden A 251 MB, R13 75 MB), so the
# kernel REGENERATES them with the official code (golden_kernel's run A + rung13's A13 --dump-params, the
# same pins/commit/args) and the reading counts only if both regenerated files equal the box's byte for
# byte (np.savez is byte-reproducible: checked 11-26 under numpy 1.26.4 and 2.0.2). Then mlx[cpu]==0.32.4
# from PyPI into the same venv, and rung13 --port + rung13b's ten modes with JACK_GOLDEN_A / JACK_R13 set.
# LINE (fixed before the run): PASS iff both sha256 equal BOX_SHA, every run rc 0 with its PASS/HOLD line;
# a sha mismatch or a missing run -> NO-READING (the kernel tested a different golden / died); else FAIL.
BOX_SHA = {"A.npz": "84d42bb6d0126ab5bf94d6204ceb136e4331fad070ec434b96e14d71352c97f4",
           "R13.npz": "8e0ee53f93656435ca287fb9e62b420d4230bd883ba1a0c3f6ae687c5d8abf00"}
MODES_13B = ["", "--losses", "--obs", "--scan", "--wm", "--imag", "--imagloss-q2", "--imagloss-vq",
             "--repval", "--grads"]
RUNS_13 = ([("rung13_port", "rung13.py", ["--port"], 1800)]
           + [("rung13b" + (m.replace("--", "_") if m else "_enc"), "rung13b.py", [m] if m else [], 2400)
              for m in MODES_13B])
EXPECT["rung13_port"] = r"RUNG 13a PORT \(mlx ([\w.]+)\) PASS"
for _k, *_ in RUNS_13[1:]:
    EXPECT[_k] = (r"RUNG 13b-losses: target HOLDS \(planted (\d+)/(\d+)\)" if _k == "rung13b_losses" else
                  r"RUNG 13b-enc: references HOLD; port PASS" if _k == "rung13b_enc" else
                  rf"RUNG 13b-{re.escape(_k[len('rung13b_'):])}(?: \([^)]*\))?: port PASS \(planted caught (\d+)/(\d+)\)")
SETS["13"] = (RUNS_13, "wheel_13.json")
# J25 (11-26 04:30): wheel 13 read FAIL at rung13b_grads ONLY (10/11 runs PASS, sha256 match): pol vs truth at
# call 2 port 5.80e+01 vs official 3.08e-01 (13/13 over), call 6 1/13 over; on the box call 2 port == official
# (3.24e-01). Set 13p is a PROBE, not a test (no verdict, writes no ledger): the same regenerated goldens, then
# rung13b_polgrad_probe.py on the wheel — prints |val32 - val64| and port_gate for adv=port/f64/gret per call,
# to tell an x86 f32 value-decode rounding difference from a port bug (box: call 2 |val32-val64| 5.7e-04).
RUNS_13P = [("polgrad_probe", "rung13b_polgrad_probe.py", [], 2400)]
SETS["13p"] = (RUNS_13P, "wheel_13p.json")
# J25 (11-26 04:55): 13p read — at call 2 the wheel's |val32 - val64| is 5.835e-01 (box 5.723e-04), adv=f64 on the
# port's own graph reads global-rel 8.27e-02 (box 9.25e-02), adv=gret fails like adv=port: the miss is in val32 alone.
# Box: one ulp on one tail prob swings the decode by 0.226 (bins to 4.85e8, |val| 1.1e-3). Set 13u is a PROBE (no
# verdict, no ledger): ulp distance of x86 mlx's f32 softmax vs the correctly rounded f64 softmax, per call.
RUNS_13U = [("softmax_ulp_probe", "rung13b_softmax_ulp_probe.py", [], 1200)]
SETS["13u"] = (RUNS_13U, "wheel_13u.json")
# 13u READ (11-26 05:10, `jack-cpu-1795668030`; its receipt says NO-READING only because kernel() sent the probe set
# to the test verdict, which looks in res["runs"] — fixed: 13u is a probe like 13p). Probe rc 0, sha256 = box:
# softmax ulp vs rounded-f64 max 31 on 99% of probs (as the box); call 2 |val_mlx - val_rounded| 5.838e-01,
# |val_rounded - val_f64| 1.145e-03 -> mlx's softmax rounding carries the 0.58. Ruling T2 in rung13b.py.


def job_body_13(which: str = "13") -> str:
    runs, fname = SETS[which]
    from experiments.brain_a import rung13
    s = rung13.job_body()
    old = repr([("A13", ["--seed", "0", "--dump-params"])])
    assert s.count(old) == 1
    s = s.replace(old, repr([("A", ["--seed", "0"]), ("A13", ["--seed", "0", "--dump-params"])]))
    tail = s.rindex('print("DONE", flush=True)')
    s = s[:tail] + r'''res["sha256"] = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()
                 if os.path.exists(os.path.join(out, f)) else None for f in ("A.npz", "R13.npz")}
_s.run(([_y.executable, "-m", "uv", "pip", "install", "-q", "--python", PY] if PY != _y.executable
        else [PY, "-m", "pip", "install", "-q"]) + [__PIN__], check=True)
tmp = "/tmp/jack_mlx_tmp"; os.makedirs(tmp, exist_ok=True)
env = dict(os.environ, TMPDIR=tmp, JACK_GOLDEN_A=os.path.join(out, "A.npz"), JACK_R13=os.path.join(out, "R13.npz"))
res["runs13"] = {}
for key, script, args, tmo in __RUNS__:
    t1 = time.time()
    try:
        p = _s.run([PY, "-I", os.path.join("experiments", "brain_a", script)] + args,
                   capture_output=True, text=True, timeout=tmo, env=env)
        rec = {"rc": p.returncode, "stdout_tail": p.stdout[-8000:], "stderr_tail": p.stderr[-3000:]}
    except _s.TimeoutExpired:
        rec = {"rc": "TIMEOUT", "timeout_s": tmo}
    rec["wall_s"] = round(time.time() - t1, 1)
    res["runs13"][key] = rec
    print(key, rec["rc"], rec["wall_s"], flush=True)
    json.dump(res, open(os.path.join(out, __FNAME__), "w"))
for f in ("A.npz", "B.npz", "A13.npz", "R13.npz"):
    if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
json.dump(res, open(os.path.join(out, __FNAME__), "w"))
print("DONE", flush=True)'''
    return s.replace("__PIN__", repr(PIN)).replace("__RUNS__", repr(runs)).replace("__FNAME__", repr(fname))


def verdict_13(res: dict) -> dict:
    sha = res.get("sha256") or {}
    for f, want in BOX_SHA.items():
        if sha.get(f) != want:
            return {"WHEEL": "NO-READING", "why": f"regenerated {f} sha256 {sha.get(f)} != box {want[:8]}"}
    v = verdict({"runs": res.get("runs13", {})}, "13")
    v["sha256_match"] = True
    return v


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
        n_m = m and m.re.groups >= 2
        if r.get("rc") != 0 or not m or (n_m and m.group(1) != m.group(2)):
            return {"WHEEL": "FAIL", "first_differing": key, "rc": r.get("rc"),
                    "stdout_tail": r.get("stdout_tail", "")[-1500:],
                    "stderr_tail": r.get("stderr_tail", "")[-800:]}
        ver = m.group(3) if m.re.groups >= 3 else (m.group(1) if m.re.groups == 1 else VERSION)
        if not key.endswith("_planted") and ver != VERSION:
            return {"WHEEL": "FAIL", "first_differing": key,
                    "why": f"mlx {ver} ran, not the pinned {VERSION}"}
        lines[key] = m.group(0)
    return {"WHEEL": "PASS", "lines": lines, "machine": res.get("machine"),
            "libc": res.get("libc"), "python": res.get("python")}


def kernel(which: str = "12") -> dict:
    from experiments.gpu import build_job, result_json, submit
    fname = SETS[which][1]
    body = job_body_13(which) if which.startswith("13") else job_body(which)
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=1.0 if which.startswith("13") else 0.3,
                 timeout_s=int(2 * 3600), fetch=[fname])
    if not sub.ok:
        raise RuntimeError(f"wheel_kernel: {sub.message}")
    r = result_json(sub, fname)
    OUT.mkdir(parents=True, exist_ok=True)
    r["verdict"] = (verdict_13(r) if which == "13" else
                    {"PROBE": "no verdict", "sha256": r.get("sha256")} if which in ("13p", "13u") else verdict(r, which))
    (OUT / fname).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["verdict"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel(sys.argv[2] if len(sys.argv) > 2 else "12")
    elif sys.argv[1:2] == ["verdict"]:
        _w = sys.argv[3] if len(sys.argv) > 3 else "12"
        _r = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps(verdict_13(_r) if _w == "13" else verdict(_r, _w), indent=1))
    else:
        raise SystemExit(__doc__)
