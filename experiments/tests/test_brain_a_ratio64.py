"""A1-r64 (prereg a2_ratio64.md, frozen 97762429): the arm moves ONE key; the judge is per-seed 3/3."""
import json

from experiments.brain_a import ratio64 as Q
from experiments.brain_a import rows as R
from experiments.tests.test_brain_a_rows import _fake_runner


def test_arm_moves_only_replay_ratio():
    b = Q.base()
    diff = [(x, y) for x, y in zip(R.BASE, b) if x != y]
    assert len(b) == len(R.BASE) and diff == [("algo.replay_ratio=0.5", "algo.replay_ratio=0.0625")]
    ovr = R.overrides("A1", 20, False, b)
    assert "env.wrapper.task_name=balance" in ovr and "seed=20" in ovr


def test_kernel_body_compiles_and_names_its_file():
    t = Q.kernel_body((20,), True)
    compile(t, "r64", "exec")
    assert "brain_a_a1r64_s20_c.json" in t and "torch==2.6.0" in t


def _eps(v100, v200):
    return [(s, v100) for s in range(72_000, 100_000, 2_000)] + [(s, v200) for s in range(172_000, 200_000, 2_000)]


def test_judge_per_seed_rule_planted():
    twin = _eps(300, 300)
    ok = {20: _eps(800, 990), 21: _eps(700, 950), 22: _eps(880, 1000)}
    assert Q.judge(ok, twin)["verdict"] == "HOLDS"
    # mean in band (793) but one seed under 690 -> MISSES (stricter than rows.py's mean rule)
    one_low = {20: _eps(850, 990), 21: _eps(850, 950), 22: _eps(680, 990)}
    j = Q.judge(one_low, twin)
    assert j["verdict"] == "MISSES" and j["points"][100_000]["in_band"] == "2/3"
    assert j["points"][200_000]["reading"] == "in band 3/3"
    # twin inside the band -> VOID; a missing seed -> NOT JUDGED
    assert Q.judge(ok, _eps(700, 300))["verdict"] == "VOID"
    assert Q.judge({20: ok[20], 21: ok[21]}, twin)["verdict"] == "NOT JUDGED"


def test_read_window_is_ruling_78s_half_open():
    # an episode ending exactly at P is outside [P-30K, P) but inside rows.py's (0.9P, P]
    eps = [(100_000, 999.0), (75_000, 500.0)]
    assert Q.score_read(eps, 100_000) == 500.0 and R.score_at(eps, 100_000) == 999.0


def test_run_row_with_fake_runner_writes_tagged_file(tmp_path):
    res = R.run_row("A1", (20,), str(tmp_path), "cpu", cmd_prefix=_fake_runner(tmp_path, 2000),
                    base=Q.base(), tag=Q.TAG, control_seed=Q.CONTROL_SEED)
    assert [r["name"] for r in res["runs"]] == ["A1r64_s20_control", "A1r64_s20"]
    d = json.load(open(tmp_path / "brain_a_rows_A1r64.json"))
    assert len(d["runs"]) == 2


def main():
    import pathlib, sys, tempfile
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
