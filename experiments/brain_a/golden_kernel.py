"""J16 row 2.1: the size1m golden generator — one Kaggle CPU kernel (rank 0).

Why a kernel: one eager size1m train call peaks at 3.4 GB on the box (> its 1.5 GB budget, and a
batch cut does not fix it), and a golden needs >= 6 calls (call 1 has 14/81 grads exactly zero).

Source: danijar/dreamerv3 @ e01491f (MIT), code only, never weights; fresh init.
The kernel clones it, pins the box's `jack-dv3` venv (Python 3.12) and runs
`experiments/brain_a/g1_probe.py --preset size1m` three times, each a fresh process:
  A        seed 0, file noise seed 0, STEPS calls, batch 16, episode start at EP_COL
  B        A again, byte for byte                       -> must equal A (every array, every param)
  control  A with --seed 1 (the agent's init seed)      -> must differ from A
GOLDEN LINE (fixed before any run): PASS iff all three runs exit 0 (rc 3 = some grad is
exactly zero at EVERY dumped call = a golden blind there -> FAIL), A == B on every dumped
array and every param hash, A != control on params, and A's dump has in_reset 1 on the even rows
at observe step EP_COL-1 (replay_context 1) in every dumped call, and 0 elsewhere. NO-READING if a run is missing.

Ours beside theirs (listed differences, nothing else): jax cpu + prealloc False (what `debug`
sets); compute_dtype float32 (published bfloat16); agent.opt.warmup=0 (so call 1 moves the
weights; the warm-up ramp is rung 3's, PASS); sampler noise from a file (`file_noise.py`, 0.4c
iii) so the Mac can replay it; a synthetic 74-in/8-out batch, not a replay buffer (even rows
cross an episode start at EP_COL); ninjax 3.6.2 (3.6.3 fails opt.update's unflatten assert on
this checkout — box, 11-21).

Usage:  python -m experiments.brain_a.golden_kernel kernel      (submits; writes OUT/golden.json)
        python -m experiments.brain_a.golden_kernel verdict FILE
"""
import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/mlx/g1v1")
COMMIT = "e01491fad6434b2245a3b8ca201dd7faedcc458c"
PINS = ["jax==0.4.33", "jaxlib==0.4.33", "ninjax==3.6.2", "elements==3.22.4", "granular==0.25.0",
        "portal==3.8.3", "optax==0.2.5", "chex==0.1.90", "einops==0.8.2", "scope==0.8.1",
        "jaxtyping==0.3.11", "numpy==1.26.4", "ruamel.yaml==0.19.1", "msgpack==1.2.4",
        "cloudpickle==3.1.2"]
STEPS = 8
DUMP = "1,2,6,8"
BATCH = 16
EP_COL = 33          # size1m: T = 64 + 1 context; observe step 32 carries the reset
NOISE_N = 64_000_000  # B=4 drew 1.67M uniform a call (smoke_b4) -> B=16 x 8 calls ~ 54M
RUNS = [("A", ["--seed", "0"]), ("B", ["--seed", "0"]), ("control", ["--seed", "1"])]

_JOB = r'''
import subprocess as _s, sys as _y, json, os, platform, time
t0 = time.time()
# The pins are the box's jack-dv3 venv (Python 3.12); jaxlib 0.4.33 has no wheel for the Kaggle
# image's Python 3.13 (kernel jack-cpu-1795270995, 11-21) -> a uv-made 3.12 venv, pins unchanged.
if _y.version_info[:2] == (3, 12):
    PY = _y.executable
else:
    _s.run([_y.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    _s.run([_y.executable, "-m", "uv", "venv", "-q", "-p", "3.12", "/tmp/v312"], check=True)
    PY = "/tmp/v312/bin/python"
_s.run([_y.executable, "-m", "uv", "pip", "install", "-q", "--python", PY] + __PINS__ if PY != _y.executable
       else [PY, "-m", "pip", "install", "-q"] + __PINS__, check=True)
_s.run(["git", "clone", "-q", "https://github.com/danijar/dreamerv3", "/tmp/dv3"], check=True)
_s.run(["git", "-C", "/tmp/dv3", "checkout", "-q", __COMMIT__], check=True)
os.chdir("/tmp/jack")
out = os.environ["JACK_OUT"]
res = {"python": _s.run([PY, "-c", "import platform; print(platform.python_version())"],
                        capture_output=True, text=True).stdout.strip(), "kernel_python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
       "install_s": round(time.time() - t0, 1), "runs": {}}
noise = "/tmp/noise1m_s0.npz"
for key, extra in __RUNS__:
    cmd = [PY, "-m", "experiments.brain_a.g1_probe", "--src", "/tmp/dv3",
           "--preset", "size1m", "--jit", "False", "--steps", str(__STEPS__),
           "--noise-file", noise, "--noise-n", str(__NOISE_N__), "--dump-step", __DUMP__,
           "--dump", os.path.join(out, key + ".npz"), "--set", "agent.opt.warmup=0",
           "--set", "batch_size=" + str(__BATCH__), "--episode-start", str(__EP_COL__),
           "--out", os.path.join(out, key + ".json")] + extra
    t1 = time.time()
    try:
        p = _s.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
        rec = {"rc": p.returncode, "stdout_tail": p.stdout[-6000:], "stderr_tail": p.stderr[-3000:]}
    except _s.TimeoutExpired:
        rec = {"rc": "TIMEOUT"}
    rec["wall_s"] = round(time.time() - t1, 1)
    res["runs"][key] = rec
    print(key, rec["rc"], rec["wall_s"], flush=True)
    json.dump(res, open(os.path.join(out, "golden_runs.json"), "w"))
import numpy as np, hashlib
res["noise_sha256"] = hashlib.sha256(open(noise, "rb").read()).hexdigest()
def cmp(x, y):
    a, b = np.load(os.path.join(out, x + ".npz")), np.load(os.path.join(out, y + ".npz"))
    keys = sorted(set(a) | set(b))
    return {"n": len(keys), "equal": sum(k in a and k in b and np.array_equal(a[k], b[k]) for k in keys)}
for x, y in (("A", "B"), ("A", "control")):
    try:
        res["cmp_" + x + "_" + y] = cmp(x, y)
    except Exception as e:
        res["cmp_" + x + "_" + y] = {"error": repr(e)}
try:
    a = np.load(os.path.join(out, "A.npz"))
    res["in_reset"] = {k: np.asarray(a[k]).astype(int).tolist() for k in a if k.endswith("in_reset")}
except Exception as e:
    res["in_reset"] = {"error": repr(e)}
json.dump(res, open(os.path.join(out, "golden_runs.json"), "w"))
print("DONE", flush=True)
'''


def job_body() -> str:
    rep = {"__PINS__": repr(PINS), "__COMMIT__": repr(COMMIT), "__RUNS__": repr(RUNS),
           "__STEPS__": repr(STEPS), "__NOISE_N__": repr(NOISE_N), "__DUMP__": repr(DUMP),
           "__BATCH__": repr(BATCH), "__EP_COL__": repr(EP_COL)}
    s = _JOB
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def _probe(r):
    return r is not None and r.get("rc") == 0


def verdict(res: dict, a_json=None, b_json=None, c_json=None) -> dict:
    runs = res.get("runs", {})
    for key, _ in RUNS:
        if key not in runs:
            return {"GOLDEN": "NO-READING", "why": f"{key} missing (kernel died before it)"}
        if not _probe(runs[key]):
            return {"GOLDEN": "FAIL", "first": key, "rc": runs[key].get("rc"),
                    "stderr_tail": runs[key].get("stderr_tail", "")[-800:]}
    ab, ac = res.get("cmp_A_B", {}), res.get("cmp_A_control", {})
    if "error" in ab or ab.get("equal") != ab.get("n") or not ab.get("n"):
        return {"GOLDEN": "FAIL", "why": "A != B (not repeatable)", "cmp": ab}
    if a_json and b_json and a_json.get("params") != b_json.get("params"):
        return {"GOLDEN": "FAIL", "why": "A != B param hashes"}
    if a_json and c_json and a_json.get("params") == c_json.get("params"):
        return {"GOLDEN": "FAIL", "why": "control (seed 1) == A: the seed is not read"}
    step = f"rssm/t{EP_COL - 1:03d}/in_reset"  # observe step k = column k + 1 (box smoke: col 5 -> t004)
    want = [1 if i % 2 == 0 else 0 for i in range(BATCH)]
    resets = res.get("in_reset", {})
    hits = {k: v for k, v in resets.items() if any(v)}
    n_calls = len(DUMP.split(","))
    if sum(k.endswith(step) and v == want for k, v in hits.items()) != n_calls or any(
            not k.endswith(step) for k in hits):
        return {"GOLDEN": "FAIL", "why": f"reset not exactly at {step} on even rows",
                "hits": sorted(hits)[:8]}
    return {"GOLDEN": "PASS", "cmp_A_B": ab, "cmp_A_control": ac, "noise_sha256": res.get("noise_sha256")}


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=int(11 * 3600),
                 fetch=["golden_runs.json", "A.npz", "A.json", "B.json", "control.json"])
    if not sub.ok:
        raise RuntimeError(f"golden_kernel: {sub.message}")
    r = result_json(sub, "golden_runs.json")
    OUT.mkdir(parents=True, exist_ok=True)
    js = {}
    for k in ("A", "B", "control"):
        try:
            js[k] = result_json(sub, k + ".json")
        except Exception:
            js[k] = None
    npz = sub.artifacts.get("A.npz")
    if npz:  # the golden itself (A == B by the line); kept beside its run record
        import shutil
        shutil.copy(npz, OUT / "golden_size1m_A.npz")
    r["golden_npz"] = str(OUT / "golden_size1m_A.npz") if npz else None
    r["verdict"] = verdict(r, js["A"], js["B"], js["control"])
    (OUT / "golden.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["verdict"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif sys.argv[1:2] == ["verdict"]:
        print(json.dumps(verdict(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    else:
        raise SystemExit(__doc__)
