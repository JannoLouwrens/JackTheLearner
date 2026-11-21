"""J14 (b): brain A in his world — the smoke's overrides, the registered world and the DONE judge.
Each check carries a control that must fail. No sheeprl needed."""
import math
import sys

from experiments.brain_a import in_world as W
from experiments.brain_a import rows as R


def test_overrides_keep_rows_settings_and_swap_only_the_env():
    ovr = W.overrides(total=20_000, seed=3, motor="body", device="cuda")
    # A's row settings ride along unchanged (preset, replay ratio, vectors, envs)
    for keep in ("algo=dreamer_v3_5_1M", "algo.replay_ratio=0.5", "env.num_envs=4",
                 "algo.mlp_keys.encoder=[state]", "algo.cnn_keys.encoder=[]"):
        assert keep in ovr, keep
    # the DMC-only keys are gone, his world is in, each key given once (Hydra refuses repeats)
    assert not any(o.startswith("env.wrapper.from_") for o in ovr)
    assert "env=gym" in ovr and f"env.id={W.GYM_ID}" in ovr and "env.action_repeat=1" in ovr
    keys = [o.split("=")[0] for o in ovr]
    assert len(keys) == len(set(keys)), keys
    assert "algo.total_steps=20000" in ovr and "+env.wrapper.motor_frame=body" in ovr
    # control: rows' own A2 overrides would NOT be his world
    assert "env=gym" not in R.overrides("A2", 0)


def test_registered_world_is_the_pilot_world_with_body_frame_eye():
    import gymnasium as gym
    env = gym.make(W.GYM_ID, render_mode="rgb_array", motor_frame="body").unwrapped
    assert env.eye_frame == "body" and env.motor_frame == "body"
    assert env.food_floor_only and env.random_layout and env.move_cost_dial
    assert env.native_substep and env.spawn_clear == 1.55
    o, _ = env.reset(seed=0)
    assert o.ndim == 1 and env.action_space.shape[0] > 0
    o2, r, term, trunc, _ = env.step(env.action_space.sample())
    assert o2.shape == o.shape and math.isfinite(r)
    # control: --motor world gives today's push
    assert gym.make(W.GYM_ID, motor_frame="world").unwrapped.motor_frame == "world"


def _rec(**kw):
    base = {"rc": 0, "cut": False, "last_step": 20_000, "wall_s": 400.0, "episodes": [(800, 12.0, 1)],
            "losses": {"Loss/world_model_loss": [(2000, 9.1), (3000, 7.2)],
                       "Loss/policy_loss": [(2000, 0.1), (3000, 0.05)]}}
    base.update(kw)
    return base


def test_judge_passes_a_clean_run_and_each_clause_can_fail():
    v = W.judge(_rec(), 20_000)
    assert v["done"] and abs(v["decisions_per_s"] - 50.0) < 1e-9
    fails = {
        "steps": [_rec(rc=1), _rec(cut=True), _rec(last_step=10_000)],
        "learns": [_rec(losses={"Loss/world_model_loss": [(2000, 9.1)]})],
        "no_nan": [_rec(losses={"Loss/world_model_loss": [(2000, 9.1), (3000, float("nan"))]}),
                   _rec(episodes=[(800, float("inf"), 0)]), _rec(losses={})],
    }
    for clause, recs in fails.items():
        for r in recs:
            v = W.judge(r, 20_000)
            assert not v["clauses"][clause] and not v["done"], (clause, r)


def test_run_reports_not_done_when_no_gradient_step_is_logged(tmp_path):
    # a fake runner that prints episodes and exits 0 but writes no losses must NOT pass
    fake = tmp_path / "fake.py"
    fake.write_text("import sys\nfor i in range(4):\n    print(f'Rank-0: policy_step=20000, reward_env_{i}=1.0')\n")
    r = W.run(str(tmp_path), total=20_000, device="cpu", compile_on=False, deadline_s=60,
              cmd_prefix=(sys.executable, str(fake)))
    assert r["rc"] == 0 and r["verdict"]["clauses"]["steps"]
    assert not r["verdict"]["clauses"]["learns"] and not r["verdict"]["done"]


def test_per_sense_overrides_name_five_keys_for_encoder_and_decoder():
    ovr = W.overrides(total=600, keys="per_sense")
    lst = "[vision,audio,touch,proprio,needs]"
    assert f"algo.mlp_keys.encoder={lst}" in ovr and f"algo.mlp_keys.decoder={lst}" in ovr
    assert "+env.wrapper.keys=per_sense" in ovr
    keys = [o.split("=")[0] for o in ovr]
    assert len(keys) == len(set(keys)), keys
    # every other setting is the flat smoke's, unchanged
    flat = W.overrides(total=600)
    diff = set(ovr) ^ set(flat)
    assert diff == {f"algo.mlp_keys.encoder={lst}", f"algo.mlp_keys.decoder={lst}",
                    "+env.wrapper.keys=per_sense", "algo.mlp_keys.encoder=[state]"}, diff
    # control: the flat path still feeds one key and no wrapper key
    assert "algo.mlp_keys.encoder=[state]" in flat and not any("wrapper.keys" in o for o in flat)
    try:
        W.overrides(keys="perr_sense")
        raise AssertionError("a misspelt keys arm was accepted")
    except ValueError:
        pass


def test_per_sense_split_is_exact_and_a_shifted_split_is_caught():
    import gymnasium as gym
    import numpy as np
    from experiments.jack_env import SENSES
    flat = gym.make(W.GYM_ID, motor_frame="body").unwrapped
    per = gym.make(W.GYM_ID, motor_frame="body", keys="per_sense")
    assert isinstance(per.observation_space, gym.spaces.Dict)
    # gymnasium sorts a Dict's keys; sheeprl reads them by name, so the set is what binds
    assert sorted(per.observation_space.spaces) == sorted(SENSES) and W.per_sense_keys() == list(SENSES)
    widths = {k: per.observation_space[k].shape[0] for k in SENSES}
    assert sum(widths.values()) == flat.observation_space.shape[0]
    of, _ = flat.reset(seed=5)
    od, _ = per.reset(seed=5)
    rng = np.random.default_rng(0)
    same = [np.array_equal(of, np.concatenate([od[k] for k in SENSES]))]
    for _ in range(40):
        a = rng.uniform(-1, 1, flat.action_space.shape).astype(np.float32)
        of, rf, tf, _, _ = flat.step(a)
        od, rd, td, _, _ = per.step(a)
        same.append(np.array_equal(of, np.concatenate([od[k] for k in SENSES])) and rf == rd and tf == td)
        assert all(per.observation_space[k].contains(od[k]) for k in SENSES)
    assert all(same), same.index(False)
    # planted: a split one number off (vision 31, audio 17) must not round-trip to the same senses
    cuts = np.cumsum([0] + [widths[k] for k in SENSES])
    bad = cuts.copy()
    bad[1] -= 1
    good_v, bad_v = of[cuts[0]:cuts[1]], of[bad[0]:bad[1]]
    assert bad_v.shape != good_v.shape or not np.array_equal(bad_v, good_v)
    assert not np.array_equal(of[bad[1]:bad[2]], od["audio"])


def test_make_refuses_an_unknown_keys_arm():
    try:
        W.make(0, keys="sum")
        raise AssertionError("an unknown keys arm built a world")
    except ValueError:
        pass


def main():
    import pathlib
    import tempfile
    import traceback
    ok = True
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(pathlib.Path(tempfile.mkdtemp(dir="/data/tmp")))
            else:
                fn()
            print("PASS", name)
        except Exception as e:
            ok = False
            traceback.print_exc()
            print("FAIL", name, repr(e))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


# 11-20 Next-up 1 (2): `kernel --per-sense`. sha256 of kernel_body() taken from the file
# BEFORE the keys arg existed (HEAD 69ce1eac): the flat GPU job must not move.
# 11-21: moved ON PURPOSE by speed._JOB's torch 2.6.0+cu126 re-pin (sheeprl 0.8.5 needs torch>=2.6;
# Kaggle's preamble pinned 2.5.1 -> A1/A2 died at pip). HEAD f11fb1dc's body hashed to the old value
# 853d1a46...; the new body = old + only the re-pin block (checked by recomputing with HEAD's _JOB).
FLAT_KERNEL_SHA = "c0ad8d1067efb8ed449e8cadf6865b1ffd7af6c3fe44ccc015710b4400c6b606"


def test_kernel_body_flat_unchanged_and_per_sense_only_adds_keys():
    import hashlib
    from experiments.brain_a import in_world as w
    flat = w.kernel_body()
    assert hashlib.sha256(flat.encode()).hexdigest() == FLAT_KERNEL_SHA
    assert w.kernel_body("flat") == flat
    ps = w.kernel_body("per_sense")
    assert ps != flat   # control: the arg reaches the job
    assert ps == flat.replace('run(WORK, device="cuda")', 'run(WORK, device="cuda", keys=\'per_sense\')')
    compile(ps, "job", "exec")
    assert w.kernel_out_name("flat") == "brain_a_inworld.json"
    # the name the GPU run writes must be the name the kernel fetches
    assert w.kernel_out_name("per_sense") == "inworld_s0_body_per_sense.json"
    try:
        w.kernel_body("per-sense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown keys arm reached the job")
