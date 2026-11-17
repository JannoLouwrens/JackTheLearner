"""pilot_b's two knobs for J11's pilot arms (`world`, `gamma`) — default =
unchanged, and each knob does what its arm will claim.

ORDERS J11 item 3 prep, written 11-17 BEFORE the rule and before the floor
card is complete. NOT a pilot run and not a reading of any arm: seed 381 (the
card's smoke seed; never 301-345, never the pilot's 401+), WARMUP cut to 300
and T = 700 decisions, so the learner makes ~12 rounds of updates. Nothing
here says whether anything eats.

Run (needs sbx; taskset, not JACK_PIN_CPU — the pin guard refuses a second
run_b in one process):  taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python \
                      -m experiments.tests.test_pilot_b_knobs

  identity  run_b(world=make_world, gamma=GAMMA) == run_b(): every life row
            and every actor weight bit for bit.
  gamma     gamma=0.999 reaches the model (model.gamma) and moves the actor
            (control: same call with GAMMA does not — that is `identity`).
  world     world = pilot_b2.make_world(move_cost_dial=True): the env the
            learner stepped has move_cost_dial, spawn_clear, lazy audio
            (read off the env), is stamped in the run dict, and the lives
            differ from the default world's.
  lr0       K's floor arm (that world, learning rate 0): the actor is bit
            identical to its initial weights after the same 700 decisions;
            control: lr 3e-4 in the same world moves them.
  floor     run_floor(world=) default identical; with the K world stamped.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED, T, WARM = 381, 700, 300


def _flat(model):
    import jax
    leaves = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in leaves])


def _run(pb, **kw):
    """run_b, keeping the model, its initial actor weights and its env."""
    keep = {}
    real = pb.make_b

    def _mk(venv, *a, **k):
        m = real(venv, *a, **k)
        keep.update(model=m, init=_flat(m).copy(),
                    env=venv.venv.envs[0].env)
        return m
    pb.make_b = _mk
    try:
        r = pb.run_b(SEED, T, **kw)
    finally:
        pb.make_b = real
    keep["run"] = r
    keep["actor"] = _flat(keep["model"])
    keep["lives_sha"] = hashlib.sha1(json.dumps(
        r["lives"], sort_keys=True, default=str).encode()).hexdigest()[:12]
    return keep


def main() -> int:
    from experiments import pilot_b as pb
    from experiments import pilot_b2 as p2
    pb._single_thread()
    pb.WARMUP = WARM
    out = {}

    def kworld(seed):
        return p2.make_world(seed, move_cost_dial=True)

    base = _run(pb)
    same = _run(pb, world=pb.make_world, gamma=pb.GAMMA)
    out["identity"] = {
        "lives_equal": base["run"]["lives"] == same["run"]["lives"],
        "actor_max_abs_diff": float(np.abs(base["actor"] - same["actor"]).max()),
        "base_moved_from_init": float(np.abs(base["actor"] - base["init"]).max()),
        "default_dict_has_no_new_keys": not ({"gamma", "world"} & set(base["run"])),
        "lives": len(base["run"]["lives"])}
    assert out["identity"]["lives_equal"]
    assert out["identity"]["actor_max_abs_diff"] == 0.0
    assert out["identity"]["base_moved_from_init"] > 0.0   # the run did learn
    assert out["identity"]["default_dict_has_no_new_keys"]

    g = _run(pb, gamma=0.999)
    out["gamma"] = {
        "model_gamma": float(g["model"].gamma), "stamped": g["run"].get("gamma"),
        "default_model_gamma": float(base["model"].gamma),
        "actor_max_abs_diff_vs_default": float(np.abs(g["actor"] - base["actor"]).max())}
    assert abs(out["gamma"]["model_gamma"] - 0.999) < 1e-12
    assert abs(out["gamma"]["default_model_gamma"] - 0.995) < 1e-12
    assert abs(out["gamma"]["stamped"] - 0.999) < 1e-12
    assert out["gamma"]["actor_max_abs_diff_vs_default"] > 0.0

    k = _run(pb, world=kworld)
    e = k["env"]
    out["world"] = {
        "env_move_cost_dial": bool(e.move_cost_dial), "env_spawn_clear": e.spawn_clear,
        "env_lazy_audio": bool(e.lazy_contact_audio), "env_reward": e.reward,
        "default_env_move_cost_dial": bool(base["env"].move_cost_dial),
        "default_env_spawn_clear": base["env"].spawn_clear,
        "stamp": k["run"].get("world"),
        "lives_differ_from_default": k["lives_sha"] != base["lives_sha"]}
    assert e.move_cost_dial and e.spawn_clear is not None and e.lazy_contact_audio
    assert not base["env"].move_cost_dial and base["env"].spawn_clear is None
    assert k["run"]["world"]["move_cost_dial"] is True
    assert k["run"]["world"]["spawn_clear"] == e.spawn_clear
    assert out["world"]["lives_differ_from_default"]

    z = _run(pb, world=kworld, lr=0.0)
    out["lr0"] = {
        "actor_max_abs_diff_vs_init": float(np.abs(z["actor"] - z["init"]).max()),
        "control_lr3e-4_same_world_moved": float(np.abs(k["actor"] - k["init"]).max()),
        "same_init_as_trained_twin": bool((z["init"] == k["init"]).all()),
        "decisions": z["run"]["decisions"]}
    assert out["lr0"]["actor_max_abs_diff_vs_init"] == 0.0
    assert out["lr0"]["control_lr3e-4_same_world_moved"] > 0.0
    assert out["lr0"]["same_init_as_trained_twin"]

    f0 = pb.run_floor("noop", SEED, 400)
    f1 = pb.run_floor("noop", SEED, 400, world=pb.make_world)
    f2 = pb.run_floor("held8", SEED, 400, world=kworld)
    out["floor"] = {"default_lives_equal": f0["lives"] == f1["lives"],
                    "default_has_no_world_key": "world" not in f0,
                    "k_world_stamp": f2.get("world")}
    assert out["floor"]["default_lives_equal"] and "world" not in f0
    assert f2["world"]["move_cost_dial"] is True

    print(json.dumps(out, indent=1))
    print("PASS test_pilot_b_knobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
