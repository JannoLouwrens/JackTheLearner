"""S12 (a), the owner's gate of 2026-11-20: is Kaggle's free TPU (TpuV6E8 — v3-8 `Tpu1VmV38` is retired: 11-20 Kaggle took the push and ran a CPU box, 0 TPU h billed; 20 h/week) usable for Brain A?

One kernel, the official danijar/dreamerv3 @ e01491f, nothing inside the agent touched:
  1. `official_debug.py` (the `debug` preset, 200 env steps, Dummy74 world) with `--jax.platform tpu`
     -> runs clean, or the traceback names the unsupported op.
  2. `official_speed.py` (row 9's shapes: 30-in / 6-out, batch 16 x 65, compiled) on the TPU at
     size1m float32, size1m bfloat16, size12m bfloat16 -> grad steps/s + peak device memory.
JAX: the image's own first; if step 1 fails on it, the pin's version `jax[tpu]==0.4.33` is installed
and both steps run again. Every attempt is recorded. One TPU core (the official default device list).

Read beside: 25.9 grad steps/s = sheeprl Dreamer-V3 size1m compiled on a Colab T4 (J5, `ca4dcf00`).
Same network size and batch, a different implementation — the table says so.

    python -m experiments.brain_a.tpu_probe kernel      # submits; writes /data/jack-logs/mlx/tpu_probe.json
"""
import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/mlx")
T4_REF = 25.9
PIN_SHA = "e01491fad6434b2245a3b8ca201dd7faedcc458c"

_JOB = r'''
import subprocess as _s, sys as _y, json, os, time
out = os.environ["JACK_OUT"]; res = {"attempts": []}
def sh(cmd, tmo):
    t0 = time.time()
    try:
        p = _s.run(cmd, capture_output=True, text=True, timeout=tmo, cwd="/tmp/jack")
        return {"rc": p.returncode, "wall_s": round(time.time() - t0, 1),
                "out": p.stdout[-4000:], "err": p.stderr[-6000:]}
    except _s.TimeoutExpired:
        return {"rc": "TIMEOUT", "wall_s": round(time.time() - t0, 1)}
def save(): json.dump(res, open(os.path.join(out, "tpu_probe.json"), "w"), indent=1)
res["clone"] = sh(["git", "clone", "-q", "https://github.com/danijar/dreamerv3", "/tmp/dv3"], 600)
sh(["git", "-C", "/tmp/dv3", "checkout", "-q", __PIN__], 120)
res["deps"] = sh([_y.executable, "-m", "pip", "install", "-q", "elements>=3.19.1", "ninjax>=3.5.1",
    "portal>=3.5.0", "granular>=0.20.3", "scope>=0.4.4", "chex", "optax", "einops", "jaxtyping",
    "colored_traceback", "ruamel.yaml", "tqdm"], 1200)
PROBE = r"""
import json, sys, jax
sys.path.insert(0, "/tmp/jack")
from experiments.brain_a.official_speed import speed
size, dtype = sys.argv[1], sys.argv[2]
r = speed("/tmp/dv3", size, 20, 300, 0, dtype, "tpu")
st = jax.devices()[0].memory_stats() or {}
r["peak_mb"] = round(st.get("peak_bytes_in_use", 0) / 2**20, 1)
r["device_kind"] = jax.devices()[0].device_kind; r["n_devices"] = len(jax.devices())
print("SPEED " + json.dumps(r))
"""
DEBUG = r"""
import sys
sys.path.insert(0, "/tmp/dv3"); sys.path.insert(0, "/tmp/jack")
from dreamerv3 import main as m
_orig = m.main
m.main = lambda flags: _orig(flags + ["--jax.platform", "tpu"])
from experiments.brain_a import official_debug
raise SystemExit(official_debug.main(["--src", "/tmp/dv3", "--logdir", "/tmp/dbg_" + sys.argv[1]]))
"""
for jaxv in ("image", "0.4.33"):
    if jaxv != "image":
        res["install_" + jaxv] = sh([_y.executable, "-m", "pip", "install", "-q", "jax[tpu]==0.4.33",
            "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"], 1200)
    a = {"jax": jaxv, "versions": sh([_y.executable, "-c",
         "import jax, numpy; print(jax.__version__, numpy.__version__, jax.devices())"], 600)}
    res["attempts"].append(a); save()
    if "tpu" not in a["versions"].get("out", "").lower():
        break  # no TPU attached: nothing on this box can answer the gate
    a["debug"] = sh([_y.executable, "-c", DEBUG, jaxv], 2400); save()
    print(jaxv, "debug rc", a["debug"]["rc"], flush=True)
    if a["debug"]["rc"] != 0:
        continue
    for size, dtype in (("1m", "float32"), ("1m", "bfloat16"), ("12m", "bfloat16")):
        r = sh([_y.executable, "-c", PROBE, size, dtype], 2400)
        line = [l for l in r.get("out", "").splitlines() if l.startswith("SPEED ")]
        a[f"speed_{size}_{dtype}"] = json.loads(line[-1][6:]) if line else r
        print(size, dtype, r["rc"], flush=True); save()
    break
print("DONE", flush=True); save()
'''


def job_body() -> str:
    return _JOB.replace("__PIN__", repr(PIN_SHA))


def table(r: dict) -> str:
    rows = ["| jax | step | result | grad steps/s | peak MB | compile s |", "|---|---|---|---|---|---|"]
    for a in r.get("attempts", []):
        d = a.get("debug", {})
        why = "clean" if d.get("rc") == 0 else f"rc={d.get('rc')}: " + (d.get("err", "").strip().splitlines() or ["?"])[-1][:160]
        rows.append(f"| {a['jax']} ({a.get('versions', {}).get('out', '').split(' ')[0]}) | debug 200 | {why} | | | |")
        for k, v in a.items():
            if k.startswith("speed_"):
                ok = isinstance(v, dict) and "grad_steps_per_s" in v
                rows.append(f"| {a['jax']} | {k[6:]} | {'ran' if ok else 'rc=' + str(v.get('rc'))} | "
                            f"{v.get('grad_steps_per_s', '')} | {v.get('peak_mb', '')} | {v.get('compile_s', '')} |")
    return "\n".join(rows)


def verdict(r: dict) -> str:
    # 11-20: a retired accelerator id was accepted and ran on CPU; that is a refusal, not a TPU result.
    if not any("tpu" in a.get("versions", {}).get("out", "").lower() for a in r.get("attempts", [])):
        return "NO TPU ATTACHED: the session's jax.devices() never listed a TPU — no verdict on the TPU"
    best = max((v.get("grad_steps_per_s", 0) for a in r.get("attempts", []) for k, v in a.items()
                if k.startswith("speed_1m") and isinstance(v, dict)), default=0)
    clean = any(a.get("debug", {}).get("rc") == 0 for a in r.get("attempts", []))
    ok = clean and best > 0
    return (f"TPU {'viable' if ok else 'NOT viable'}: official debug preset "
            f"{'ran clean' if clean else 'failed on every jax tried'}; best size1m {best} grad steps/s "
            f"vs T4 sheeprl {T4_REF}")


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-tpu", est_hours=2.0,
                 timeout_s=int(4 * 3600), fetch=["tpu_probe.json"], fetch_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not sub.ok:
        (OUT / "tpu_probe.json").write_text(json.dumps({"submit_failed": sub.message}) + "\n")
        raise RuntimeError(f"tpu_probe: {sub.message}")
    r = result_json(sub, "tpu_probe.json")
    r["job_id"], r["head"], r["verdict"] = sub.job_id, sub.head, verdict(r)
    (OUT / "tpu_probe.json").write_text(json.dumps(r, indent=1) + "\n")
    print(r["verdict"], "\n" + table(r), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif sys.argv[1:2] == ["table"]:
        r = json.loads(Path(sys.argv[2]).read_text())
        print(verdict(r), "\n" + table(r))
    else:
        raise SystemExit(__doc__)
