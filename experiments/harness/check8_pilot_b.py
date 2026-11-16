"""Harness self-check #8 (J4/§8), paid on J8's Brain B pipeline: a death does
not bootstrap, a life cap does.

Rule (written before the run, 11-16 18:35 UTC):
  The REAL pilot plumbing — `pilot_b.PilotLives(pilot_b.make_world(seed))`
  inside `VecNormalize(DummyVecEnv)`, `pilot_b.make_b` (SBX SAC, its own SB3
  ReplayBuffer) — collects decisions with learning disabled (learning_starts
  beyond the run, so nothing trains; collection is the same code path as the
  pilot's warm-up). Then for every life end the wrapper logged, the stored
  transition at that buffer index is read through `ReplayBuffer._get_samples`,
  i.e. exactly the `dones` the SAC target multiplies by:
    death  (cause != capped)  -> sampled done == 1   (target = r, no bootstrap)
    capped                     -> sampled done == 0   (target bootstraps)
  and every non-final decision -> sampled done == 0.
  PASS iff >= 1 death and >= 2 caps are observed and every row holds.
  Control that MUST fail: the same run with a mutated wrapper that reports
  the cap as `terminated` (the classic time-limit bug) -> the caps must read
  done == 1; if the check still passes on it, the check is VOID.

  Cap is shortened to CAP_TEST decisions for the cap rows (run A) and left at
  the pilot's 9,000 for the death rows (run B: a random body at d 0.75 dies in
  ~1,400 decisions), so both kinds of ending are seen in a few CPU-minutes.

  Scope: "reward-forever 1/(1-gamma)" (the other half of #8) is a property
  of the target formula SB3/SBX implement, r + gamma (1 - done) V'; this
  check proves the `done` that formula receives is right at both endings.
  It does not train a critic to its fixed point.

Run: /data/venvs/jack-std/bin/python -m experiments.harness.check8_pilot_b
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CAP_TEST = 150
SEED = 301


def _collect(cap: int, decisions: int, mutate: bool) -> dict:
    from experiments import pilot_b as pb
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class Mutated(pb.PilotLives):
        def step(self, action):
            o, r, term, trunc, info = super().step(action)
            if trunc and not term:          # the bug: cap reported as death
                term, trunc = True, False
                self._capped = True
            return o, r, term, trunc, info

    holder = {}

    def _mk():
        w = (Mutated if mutate else pb.PilotLives)(pb.make_world(SEED), cap=cap)
        holder["w"] = w
        return w

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False,
                        clip_obs=10.0)
    venv.seed(SEED)
    model = pb.make_b(venv, SEED, decisions)
    model.learning_starts = decisions + 1          # collect only, never train
    model.learn(total_timesteps=decisions, progress_bar=False)
    w = holder["w"]
    rb = model.replay_buffer
    import numpy as np
    n = rb.pos if not rb.full else rb.buffer_size
    sampled = np.asarray(rb._get_samples(np.arange(n)).dones).reshape(-1)
    ends, rows, i = set(), [], -1
    for life in w.lives:
        i += life["decisions"]
        ends.add(i)
        rows.append({"cause": life["cause"], "index": i,
                     "done": float(sampled[i])})
    inner_bad = int(sum(1 for j in range(n) if j not in ends and sampled[j] != 0))
    return {"cap": cap, "decisions": decisions, "mutate": mutate,
            "buffer_n": int(n), "ends": rows, "inner_nonzero_done": inner_bad}


def _verdict(runs: list) -> dict:
    ends = [e for r in runs for e in r["ends"]]
    deaths = [e for e in ends if e["cause"] != "capped"]
    caps = [e for e in ends if e["cause"] == "capped"]
    ok = (len(deaths) >= 1 and len(caps) >= 2
          and all(e["done"] == 1.0 for e in deaths)
          and all(e["done"] == 0.0 for e in caps)
          and all(r["inner_nonzero_done"] == 0 for r in runs))
    return {"deaths": len(deaths), "caps": len(caps),
            "death_done": sorted({e["done"] for e in deaths}),
            "cap_done": sorted({e["done"] for e in caps}), "pass": ok}


def main() -> int:
    t = time.process_time()
    real = [_collect(CAP_TEST, 3 * CAP_TEST + 10, False),
            _collect(9000, 4000, False)]
    ctrl = [_collect(CAP_TEST, 3 * CAP_TEST + 10, True),
            _collect(9000, 4000, True)]
    out = {"real": _verdict(real), "control": _verdict(ctrl),
           "cpu_s": round(time.process_time() - t, 1)}
    if out["real"]["deaths"] < 1:
        out["result"] = "VOID (no death observed)"
    elif out["control"]["pass"]:
        out["result"] = "VOID (control passed)"
    else:
        out["result"] = "PASS" if out["real"]["pass"] else "FAIL"
    out["runs"] = real + ctrl
    p = Path("/data/jack-logs/harness/check8_pilot_b.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    return 0 if out["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
