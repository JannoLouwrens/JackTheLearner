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
