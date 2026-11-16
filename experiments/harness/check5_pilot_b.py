"""Harness self-check #5 (J4/§8), paid on J8's pilot: run k sees the same
world in every arm.

Rule (written before the run, 11-16 19:00 UTC):
  For seed 301, each arm's REAL construction path from `pilot_b` is built:
    b, blr0, bwhite, bpink  `run_b`'s stack: PilotLives(make_world(s)) in
                            VecNormalize(DummyVecEnv), venv.seed(s), then the
                            arm's `make_b` (seed s, its lr / use_sde) and, for
                            bpink, `_pink_selection` — every global-RNG side
                            effect the learner's construction has happens;
    noop, held8             `run_floor`'s stack: PilotLives(make_world(s)),
                            the arm's policy object built, w.reset(seed=s).
  Then the world in each stack is driven by ONE fixed action sequence (uniform
  in [-1, 1], numpy default_rng(12345), independent of every arm) for N_DEC
  decisions, through the stack's own reset path at every life end (DummyVecEnv
  auto-reset for B-family; w.reset() for floor). A pass-through recorder
  wrapped directly around PilotLives (same in both stacks; it changes nothing)
  hashes the RAW observation of every reset and step, with reward and
  terminated/truncated. Learning never runs (no .learn()).
  PASS iff all six arms give the identical hash sequence AND >= 1 life end
  (death or cap) is inside the window, so respawn draws are covered.
  Control that MUST fail: the `b` stack built with seed 302 must differ from
  seed 301's sequence; if it matches, the check is VOID.

  Scope: with the same actions, every arm's world is the same world (layout,
  needs, respawns, noise). It does not and cannot say arms see the same world
  under their OWN actions — the world reacts to the body, which is the point.

Run: /data/venvs/jack-std/bin/python -m experiments.harness.check5_pilot_b
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

N_DEC = 2000
SEED = 301
CONTROL_SEED = 302
ARMS = ("b", "blr0", "bwhite", "bpink", "noop", "held8")
OUT = Path("/data/jack-logs/harness/check5_pilot_b.json")


class _Rec:
    """Pass-through: forwards everything, hashes what PilotLives returns."""

    def __init__(self, inner):
        import gymnasium as gym

        class R(gym.Wrapper):
            def reset(s2, **kw):
                o, info = s2.env.reset(**kw)
                self.hashes.append(_h(o, 0.0, False, False))
                return o, info

            def step(s2, a):
                o, r, te, tr, info = s2.env.step(a)
                self.hashes.append(_h(o, r, te, tr))
                if te or tr:
                    self.ends += 1
                return o, r, te, tr, info

        self.hashes, self.ends = [], 0
        self.env = R(inner)


def _actions(shape):
    return np.random.default_rng(12345).uniform(-1, 1, size=(N_DEC,) + shape).astype(np.float32)


def _h(o, r, te, tr) -> str:
    m = hashlib.sha256(np.ascontiguousarray(o, dtype=np.float64).tobytes())
    m.update(repr((float(r), bool(te), bool(tr))).encode())
    return m.hexdigest()[:16]


def trace_b_family(arm: str, seed: int) -> dict:
    from experiments import pilot_b as pb
    pb._single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    holder = {}

    def _mk():
        w = pb.PilotLives(pb.make_world(seed))
        holder["w"] = w
        holder["rec"] = _Rec(w)
        return holder["rec"].env

    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.seed(seed)
    lr = 0.0 if arm == "blr0" else 3e-4
    model = pb.make_b(venv, seed, pb.T_RUN, lr, use_sde=(arm not in pb.EXPLORE_ARMS))
    if arm == "bpink":
        pb._pink_selection(model, seed)
    venv.reset()
    for a in _actions(venv.action_space.shape):
        venv.step(a[None])
    rec = holder["rec"]
    return {"hashes": rec.hashes, "ends": rec.ends, "lives": list(holder["w"].lives)}


def trace_floor(arm: str, seed: int) -> dict:
    from experiments import pilot_b as pb
    from experiments.world_card import DoNothing, Held
    _pol = DoNothing(seed) if arm == "noop" else Held(seed, 8)
    pl = pb.PilotLives(pb.make_world(seed))
    rec = _Rec(pl)
    w = rec.env
    w.reset(seed=seed)
    _pol.new_life()
    for a in _actions(w.action_space.shape):
        _o, _r, te, tr, _info = w.step(a)
        if te or tr:
            w.reset()
            _pol.new_life()
    return {"hashes": rec.hashes, "ends": rec.ends, "lives": list(pl.lives)}


def trace(arm: str, seed: int) -> dict:
    t = time.process_time()
    r = trace_floor(arm, seed) if arm in ("noop", "held8") else trace_b_family(arm, seed)
    r["cpu_s"] = round(time.process_time() - t, 1)
    return r


def main() -> int:
    res = {a: trace(a, SEED) for a in ARMS}
    ctrl = trace("b", CONTROL_SEED)
    ref = res["b"]["hashes"]
    rows = {}
    for a, r in res.items():
        first_diff = next((i for i, (x, y) in enumerate(zip(ref, r["hashes"])) if x != y), None)
        if first_diff is None and len(ref) != len(r["hashes"]):
            first_diff = min(len(ref), len(r["hashes"]))
        rows[a] = {"identical_to_b": first_diff is None, "first_diff": first_diff,
                   "n": len(r["hashes"]), "ends": r["ends"], "cpu_s": r["cpu_s"],
                   "lives": r["lives"]}
    ctrl_differs = ctrl["hashes"] != ref
    all_same = all(v["identical_to_b"] for v in rows.values())
    ends_ok = all(v["ends"] >= 1 for v in rows.values())
    verdict = ("VOID" if not ctrl_differs else
               "PASS" if (all_same and ends_ok) else "FAIL")
    out = {"check": 5, "seed": SEED, "n_dec": N_DEC, "verdict": verdict,
           "control_seed": CONTROL_SEED, "control_differs": ctrl_differs,
           "control_first_diff": next((i for i, (x, y) in enumerate(zip(ref, ctrl["hashes"])) if x != y), None),
           "arms": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}))
    for a, v in rows.items():
        print(a, {k: v[k] for k in ("identical_to_b", "first_diff", "n", "ends", "cpu_s")})
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
