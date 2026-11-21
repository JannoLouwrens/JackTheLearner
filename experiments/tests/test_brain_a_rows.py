"""J14: brain A's row kernel — parsing, unit check, never-trains control arithmetic, row driver, merge.
Each check carries a control that must fail. No sheeprl needed (a fake runner stands in)."""
import ast
import json
import pathlib
import sys
import tempfile

from experiments.brain_a import rows as R


def _line(ps, i, r):
    return f"Rank-0: policy_step={ps}, reward_env_{i}={r}\n"


def test_parse_and_unit_check():
    good = "".join(_line(2000, i, 10.0 * i) for i in range(4)) + _line(4000, 0, 5.5)
    eps = R.parse_episodes(good)
    assert eps[0] == (4000, 0.0, 0) and eps[-1] == (8000, 5.5, 0)   # env step = 2 x policy_step
    assert R.unit_check(good)[0] is True
    assert R.unit_check("no episode yet") is None
    # controls: per-decision counting (500) and a missing env must both fail
    assert R.unit_check("".join(_line(500, i, 1) for i in range(4)))[0] is False
    assert R.unit_check("".join(_line(2000, i, 1) for i in range(3)))[0] is False


def test_control_never_reaches_a_gradient_step():
    # the runner trains when iter >= learning_starts // num_envs, iters run 1..total // num_envs
    for row in R.ROWS:
        ovr = R.overrides(row, 0, control=True)
        total = int([o for o in ovr if o.startswith("algo.total_steps=")][0].split("=")[1])
        ls = int([o for o in ovr if o.startswith("algo.learning_starts=")][0].split("=")[1])
        assert ls // R.NUM_ENVS > total // R.NUM_ENVS, row
        # planted: the pre-correction form total + 1 does reach the last iteration
        assert (total + 1) // R.NUM_ENVS <= total // R.NUM_ENVS
        assert not any(o.startswith("algo.learning_starts=") for o in R.overrides(row, 0))


def _fake_runner(tmp_path, first_ps):
    f = tmp_path / "fake.py"
    f.write_text(
        "import sys, time\n"
        f"for k in range(1, 4):\n"
        f"    for i in range(4):\n"
        f"        print('Rank-0: policy_step=%d, reward_env_%d=%f' % ({first_ps} * k, i, 100.0 * k), flush=True)\n"
        "    time.sleep(0.2)\n")
    return (sys.executable, str(f))


def test_run_row_with_fake_runner(tmp_path):
    res = R.run_row("A1", (0, 1), str(tmp_path), "cpu", cmd_prefix=_fake_runner(tmp_path, 2000))
    assert res["stopped"] is None and [r["name"] for r in res["runs"]] == ["A1_s0_control", "A1_s0", "A1_s1"]
    assert all(r["unit"]["ok"] and r["rc"] == 0 and len(r["episodes"]) == 12 for r in res["runs"])
    on_disk = json.load(open(tmp_path / "brain_a_rows_A1.json"))
    assert len(on_disk["runs"]) == 3
    # control: a runner counting per decision stops the row after the first run
    d2 = tmp_path / "bad"
    d2.mkdir()
    bad = R.run_row("A1", (0, 1), str(d2), "cpu", cmd_prefix=_fake_runner(tmp_path, 500))
    assert bad["stopped"] and len(bad["runs"]) == 1 and bad["runs"][0]["unit"]["ok"] is False


def _file(tmp_path, name, runs):
    p = tmp_path / name
    p.write_text(json.dumps({"runs": runs}))
    return str(p)


def _run(seed, ctl, ret, cut=False):
    eps = [(s, ret, 0) for s in range(4000, 200_001, 4000)]
    return {"name": f"A1_s{seed}{'_control' if ctl else ''}", "seed": seed, "control": ctl, "rc": 0,
            "cut": cut, "unit": {"ok": True}, "episodes": eps}


def test_judge_files_merge_and_refusals(tmp_path):
    base = _file(tmp_path, "rows_A1.json", [_run(0, True, 100)] + [_run(s, False, 800) for s in (0, 1, 2)])
    j = R.judge_files("A1", [base])
    assert j["points"][100_000]["reading"] == "in band" and j["verdict"] == "MISS"   # 800 < 932 at 200K
    assert j["extension_due"] is True
    ext = _file(tmp_path, "rows_A1_ext.json", [_run(s, False, 800) for s in (3, 4)])
    j5 = R.judge_files("A1", [base, ext])
    assert j5["seeds"] == [0, 1, 2, 3, 4] and j5["extension_due"] is False
    # a cut seed is missing, never a zero
    cutf = _file(tmp_path, "rows_A1_c.json", [_run(0, True, 100), _run(0, False, 800), _run(1, False, 800),
                                              _run(2, False, 800, cut=True)])
    assert R.judge_files("A1", [cutf])["verdict"] == "NOT JUDGED"
    # a control inside the band voids the row
    easy = _file(tmp_path, "rows_A1_e.json", [_run(0, True, 950)] + [_run(s, False, 950) for s in (0, 1, 2)])
    assert R.judge_files("A1", [easy])["verdict"] == "VOID"


def test_kernel_body_is_python_and_names_the_row():
    for ext in (False, True):
        src = R.kernel_body("A3", R.EXTENSION_SEEDS if ext else R.SEEDS, control=not ext)
        ast.parse(src)
        assert "run_row('A3', [3, 4], WORK, \"cuda\", control=False)" in src if ext else \
               "run_row('A3', [0, 1, 2], WORK, \"cuda\", control=True)" in src
        assert "sheeprl==0.8.5" in src and "__" + "ROW__" not in src


# ---- arm A2-v1S (J14 (b), ruling 67) ------------------------------------------------------------
FROZEN_A2_S0 = ["exp=dreamer_v3_5_dmc_walker_walk", "algo=dreamer_v3_5_1M",
                "env.wrapper.from_pixels=False", "env.wrapper.from_vectors=True",
                "algo.cnn_keys.encoder=[]", "algo.mlp_keys.encoder=[state]",
                "env.num_envs=4", "env.capture_video=False", "env.action_repeat=2",
                "algo.replay_ratio=0.5", "algo.compile.enabled=True",
                "env.wrapper.domain_name=cartpole", "env.wrapper.task_name=swingup",
                "seed=0", "algo.total_steps=100000"]


def test_frozen_rows_unmoved_by_the_new_arm():
    # the A1-A3 command lines, written out here by hand, are what overrides() still returns
    assert R.overrides("A2", 0) == FROZEN_A2_S0
    assert R.overrides("A2", 0, control=True) == FROZEN_A2_S0 + ["algo.learning_starts=200000"]
    # control: the new arm's line is NOT the frozen one (a test that cannot tell them apart is none)
    assert R.overrides("A2", 0, base=R.BASE_V1S) != FROZEN_A2_S0
    assert R.BASE[0] == "exp=dreamer_v3_5_dmc_walker_walk" and "algo.learning_starts=1024" not in R.BASE


def test_v1s_overrides_carry_ruling_67_pins():
    o = R.overrides(R.V1S_ROW, 1, base=R.BASE_V1S)
    for pin in ("exp=dreamer_v3_dmc_walker_walk", "algo=dreamer_v3_S", "fabric.precision=32-true",
                "algo.learning_starts=1024", "algo.replay_ratio=0.5", "env.num_envs=4",
                "env.action_repeat=2", "env.wrapper.task_name=swingup", "seed=1", "algo.total_steps=100000"):
        assert o.count(pin) == 1, pin
    keys = [x.split("=")[0] for x in o]
    assert len(keys) == len(set(keys))                    # Hydra refuses a key given twice
    assert not any("bf16" in x or "12M" in x or "v3_5" in x for x in o)   # struck settings stay out
    c = R.overrides(R.V1S_ROW, 0, control=True, base=R.BASE_V1S)
    ck = [x.split("=")[0] for x in c]
    assert len(ck) == len(set(ck)) and "algo.learning_starts=200000" in c and "algo.learning_starts=1024" not in c
    assert 200000 // R.NUM_ENVS > 100000 // R.NUM_ENVS    # never reaches a gradient step
    assert R.ROWS["A2"]["points"] == {100_000: (478, 20), 200_000: (778, 19)}   # bands unchanged
    assert R.band(478, 20) == (438, 518) and R.band(778, 19) == (740, 816)


def test_v1s_run_row_has_its_own_names(tmp_path):
    res = R.run_row(R.V1S_ROW, (0,), str(tmp_path), "cpu", cmd_prefix=_fake_runner(tmp_path, 2000),
                    base=R.BASE_V1S, tag=R.V1S_TAG)
    assert [r["name"] for r in res["runs"]] == ["A2v1S_s0_control", "A2v1S_s0"] and res["arm"] == "A2v1S"
    assert (tmp_path / "brain_a_rows_A2v1S.json").exists() and not (tmp_path / "brain_a_rows_A2.json").exists()
    assert "algo=dreamer_v3_S" in res["runs"][1]["cmd"] and "algo.learning_starts=200000" in res["runs"][0]["cmd"]
    # the harvested file is judged by row A2's own rule
    j = R.judge_files(R.V1S_ROW, [str(tmp_path / "brain_a_rows_A2v1S.json")])
    assert j["verdict"] == "NOT JUDGED" and "seeds missing: [1, 2]" in j["notes"]


def test_v1s_kernel_bodies():
    src = R.kernel_body_v1s((0,), True)
    ast.parse(src)
    assert 'run_row(V1S_ROW, [0], WORK, "cuda", control=True, base=BASE_V1S, tag=V1S_TAG)' in src
    assert "'brain_a_v1s_s0_c.json'" in src and "sheeprl==0.8.5" in src and "__" + "SEEDS__" not in src
    assert R.v1s_fetch_name((2,), False) == "brain_a_v1s_s2.json"
    t = R.kernel_body_timing()
    ast.parse(t)
    assert "timing_cell(WORK" in t and "run_row(" not in t
    # one seed per kernel: the command refuses two seeds and a seed outside the rule's list
    assert R.main(["rows", "v1s-kernel", "0", "1"]) == 2 and R.main(["rows", "v1s-kernel", "7"]) == 2


def test_timing_projection_arithmetic_and_unknown_is_not_zero(tmp_path):
    ts = {"Time/sps_train": [(1000, 0.5), (2000, 10.0), (3000, 10.0)],
          "Time/sps_env_interaction": [(1000, 50.0), (2000, 100.0), (3000, 100.0)]}
    p = R.project_seed_hours(ts, 100_000)
    # first window (compile) dropped: (100000-1024)*0.5/10 + 100000/100 = 4948.8 + 1000 s
    assert p["windows"] == 2 and p["sps_train"] == 10.0 and abs(p["hours"] - 5948.8 / 3600) < 0.01
    # planted: with the compile window kept, the train time ALONE (2.01 h) exceeds the whole estimate (1.65 h)
    assert (98976 * 0.5 / ((0.5 + 10 + 10) / 3)) / 3600 > p["hours"] + 0.3
    # one window only, or no train rate at all: no number, never a zero
    assert R.project_seed_hours({k: v[:1] for k, v in ts.items()}, 100_000) is None
    assert R.project_seed_hours({"Time/sps_env_interaction": ts["Time/sps_env_interaction"]}, 100_000) is None
    assert R.project_seed_hours({"error": "x"}, 100_000) is None
    # the cell itself, through a fake runner: file written, timing-only, no projection without scalars
    res = R.timing_cell(str(tmp_path), "cpu", cmd_prefix=_fake_runner(tmp_path, 2000))
    on_disk = json.load(open(tmp_path / "brain_a_v1s_timing.json"))
    assert on_disk["timing_only"] is True and on_disk["projection"] is None and res["unit"]["ok"]
    assert "algo.total_steps=6000" in on_disk["cmd"] and "algo.total_steps=100000" not in on_disk["cmd"]
    assert "scores" not in on_disk and "episodes" not in on_disk


def main():
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
            print("FAIL", name, repr(e))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
