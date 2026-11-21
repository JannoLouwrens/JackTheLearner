"""J10 why, step 2: CHAOS or RECIPE? Seed 1 continued from its own 1.5M checkpoint.

Row J10-1 FAILED: seed 1 stood and walked at 1.5M (box re-eval R 5,621, M 9.2,
0/10 fell) and had lost it at 2M (R 581-585, 10/10 fell). PUBLISHED_PRIORS §21
found no record of Zoo SAC losing a Humanoid walk late. Two readings are open:
  RECIPE  the Zoo recipe on this stack drives this policy off its walk from
          this point, whatever the minibatch / env randomness;
  CHAOS   the loss was one unlucky draw; most continuations keep the walk.
This is a DIAGNOSTIC, not part of the frozen row: it cannot change Row J10-1
(a miss is a miss) and it is not a repair. No ledger row.

RULE (written in this commit, before either fork exists):
  start     s1 seg2's kernel output (model.zip + replay.pkl at 1.5M,
            kernel jannolouwrens/jack-cpu-1795167808), mounted as in the row.
  forks     fork ids 1 and 2 (stated now; no extension). Fork f re-seeds
            the model with 1000 + f (SB3 set_random_seed; sbx's JAX `key`
            when the attribute exists — which attributes were reseeded is
            WRITTEN to the row, so a reseed that never took is visible) and
            the env with 1_000_000 + f. Everything else is train_segment's
            path: same SAC_KW from the checkpoint, same 500k to 2M, same eval
            (10 deterministic episodes, env seeds 10_000..10_009) on Kaggle CPU.
  reading   per fork at 2M: LOST = R < 1,000 AND >= 8/10 episodes fell;
            KEPT = R >= 5,000 AND <= 2/10 fell; else BETWEEN.
            both LOST           -> RECIPE (the collapse is not one draw; a
                                   repair from §21's list is the next race)
            both KEPT           -> CHAOS (seed 1's loss was a draw; the row's
                                   2/3 reading is noise-limited, say so)
            anything else       -> MIXED (fragile: the walk is lost in some
                                   continuations; neither reading is shown)
  limits    2 forks is a pilot of "how often": a MIXED or CHAOS reading says
            nothing about a rate. Eval runs on Kaggle's CPU, like the row
            (the box re-eval is not comparable — 1fdd3d58's gate failed 4/5).
"""
from __future__ import annotations

import json
import sys

START_KERNEL = "jannolouwrens/jack-cpu-1795167808"   # s1 seg2 -> 1.5M
FORKS = (1, 2)
SEED = 1
SEGMENT = 3


def reading(rows: dict) -> dict:
    """rows: fork -> {"R": float, "fell": int}. Pure reading of the rule."""
    per = {}
    for f, r in rows.items():
        if r["R"] < 1000 and r["fell"] >= 8:
            per[f] = "LOST"
        elif r["R"] >= 5000 and r["fell"] <= 2:
            per[f] = "KEPT"
        else:
            per[f] = "BETWEEN"
    if set(per) != set(FORKS):
        return {"per_fork": per, "reading": "INCOMPLETE"}
    v = set(per.values())
    return {"per_fork": per,
            "reading": "RECIPE" if v == {"LOST"} else "CHAOS" if v == {"KEPT"} else "MIXED"}


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.humanoid_sac_row import train_segment, resolve_prev_dir, _ckpt_paths
import sbx
FORK = __FORK__
_p = resolve_prev_dir(__PREV__)
print("PREV_DIR " + repr(_p), flush=True)
_orig_load = sbx.SAC.load
reseeded = []
def _load(path, env=None, **kw):
    m = _orig_load(path, env=env, **kw)
    m.set_random_seed(1000 + FORK)
    reseeded.append("set_random_seed")
    if hasattr(m, "key"):
        import jax
        m.key = jax.random.PRNGKey(1000 + FORK)
        reseeded.append("key")
    env.reset(seed=1_000_000 + FORK)
    reseeded.append("env")
    return m
sbx.SAC.load = staticmethod(_load)
out = os.environ["JACK_OUT"]
r = train_segment(__SEED__, out, _p, __SEGMENT__)
r["fork"] = FORK
r["reseeded"] = reseeded
if "eval" in r:
    r["eval"]["fell"] = sum(1 for e in r["eval"]["episodes"] if e.get("fell"))
with open(os.path.join(out, "segment.json"), "w") as f:
    json.dump(r, f, indent=1)
print("RESULT " + json.dumps(r), flush=True)
'''


def job_body(fork: int) -> str:
    from experiments.humanoid_sac_row import KERNEL_PINS
    if fork not in FORKS:
        raise ValueError(f"fork {fork} is not in the rule's forks {FORKS}")
    prev = f"/kaggle/input/{START_KERNEL.split('/', 1)[1]}"
    return (_JOB.replace("__PINS__", repr(KERNEL_PINS)).replace("__FORK__", repr(int(fork)))
            .replace("__PREV__", repr(prev)).replace("__SEED__", repr(SEED))
            .replace("__SEGMENT__", repr(SEGMENT)))


def kernel(fork: int) -> dict:
    """Submit one fork as a Kaggle CPU kernel -> /data/jack-logs/j10/fork/f<fork>.json."""
    from pathlib import Path
    from experiments.gpu import build_job, result_json, submit
    from experiments.humanoid_sac_row import KERNEL_REMOTE_TIMEOUT_S, SEGMENT_FETCH
    sub = submit(build_job(job_body(fork)), prefer="kaggle-cpu", est_hours=10.5,
                 timeout_s=KERNEL_REMOTE_TIMEOUT_S, fetch=SEGMENT_FETCH,
                 kernel_sources=[START_KERNEL], fetch_only=True)
    r = result_json(sub, "segment.json")
    r["kernel"] = sub.job_id
    out = Path("/data/jack-logs/j10/fork")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"f{fork}.json").write_text(json.dumps(r, indent=1) + "\n")
    return r


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        print(json.dumps(kernel(int(sys.argv[2]))))
    else:
        raise SystemExit("usage: -m experiments.j10_fork kernel <fork 1|2>")
