"""J28 cell 1: does MuJoCo Playground's HumanoidStand + Brax PPO at 8,192 copies RUN on the GPU Kaggle
gives, and how fast? A TIMING CELL — no verdict, no learner of Jack's, nothing kept but three numbers
(does JAX run / env steps a second / peak device memory) for PUBLISHED_PRIORS.md §29 "ours".

Published recipe, code only (no weights): `mujoco_playground` from PyPI, its own
`dm_control_suite_params.brax_ppo_config("HumanoidStand")` with ONE change: `num_timesteps` cut to
TIMESTEPS (a timing cell, not a training run) and `num_evals` to 6 so progress is timestamped.
`num_envs` = the config's own (8,192 expected; printed). Known risk (§29): JAX fails on a P100
convolution (jax#33224) — so the GPU kind is printed first and a failure's text is the reading.

    python -m experiments.body.mjx_timing kernel     # writes /data/jack-logs/body/mjx_timing.json
"""
import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/body")
FETCH = "mjx_timing.json"
TIMESTEPS = 10_000_000
EST_HOURS = 0.5

_JOB = r'''
import subprocess as _s, sys as _y, json, os, time
out = os.environ["JACK_OUT"]; res = {"cell": "J28-1 MJX HumanoidStand timing", "timing_only": True}
def sh(cmd, tmo):
    t0 = time.time()
    try:
        p = _s.run(cmd, capture_output=True, text=True, timeout=tmo)
        return {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "out": p.stdout[-4000:], "err": p.stderr[-6000:]}
    except _s.TimeoutExpired:
        return {"rc": "TIMEOUT", "wall_s": round(time.time() - t0, 1)}
def save(): json.dump(res, open(os.path.join(out, __FETCH__), "w"), indent=1)
res["gpu"] = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 60); save()
res["install"] = sh([_y.executable, "-m", "pip", "install", "-q", "jax[cuda12]", "mujoco_playground"], 1800); save()
PROBE = r"""
import functools, json, time, jax
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params
from brax.training.agents.ppo import train as ppo, networks as ppo_networks
print("JAX", jax.__version__, jax.devices(), flush=True)
name = "HumanoidStand"
env = registry.load(name)
p = dm_control_suite_params.brax_ppo_config(name)
p.num_timesteps = __STEPS__; p.num_evals = 6
kw = dict(p)
net = ppo_networks.make_ppo_networks
if "network_factory" in kw:   # the Playground notebook's own unpacking
    del kw["network_factory"]
    net = functools.partial(ppo_networks.make_ppo_networks, **p.network_factory)
marks = []
def prog(step, metrics):
    marks.append((int(step), time.time())); print("PROGRESS", step, round(time.time() - t0, 1), flush=True)
t0 = time.time()
ppo.train(environment=env, network_factory=net, progress_fn=prog,
          wrap_env_fn=wrapper.wrap_for_brax_training, **kw)
st = jax.devices()[0].memory_stats() or {}
sps = None
if len(marks) >= 3:   # first interval carries the compile: dropped
    (s1, t1), (s2, t2) = marks[1], marks[-1]
    sps = round((s2 - s1) / (t2 - t1), 1) if t2 > t1 else None
print("TIMING " + json.dumps({"num_envs": kw.get("num_envs"), "marks": marks, "env_steps_per_s": sps,
      "compile_plus_first_s": round(marks[1][1] - t0, 1) if len(marks) > 1 else None,
      "peak_mb": round(st.get("peak_bytes_in_use", 0) / 2**20, 1), "device": str(jax.devices()[0])}))
"""
r = sh([_y.executable, "-c", PROBE], 3000)
line = [l for l in r.get("out", "").splitlines() if l.startswith("TIMING ")]
res["probe"] = r; res["jax_runs"] = bool(line) and r["rc"] == 0
res["timing"] = json.loads(line[-1][7:]) if line else None
print("RESULT " + json.dumps({"gpu": res["gpu"].get("out"), "jax_runs": res["jax_runs"], "timing": res["timing"]}), flush=True)
print("DONE", flush=True); save()
'''


def job_body() -> str:
    return _JOB.replace("__FETCH__", repr(FETCH)).replace("__STEPS__", str(TIMESTEPS))


def main(argv) -> int:
    if (argv[1] if len(argv) > 1 else "") != "kernel":
        print(__doc__)
        return 2
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle", est_hours=EST_HOURS,
                 timeout_s=int(2 * 3600), fetch=[FETCH], fetch_only=True)
    print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
    if not sub.ok:
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / FETCH).write_text(json.dumps(result_json(sub, FETCH)))
    print("WROTE", OUT / FETCH, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
