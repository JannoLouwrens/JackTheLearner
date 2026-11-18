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
