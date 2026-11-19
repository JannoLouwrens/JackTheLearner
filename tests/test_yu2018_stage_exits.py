"""J10 (2): the released curriculum's per-stage exit log tells threshold from cap.

Planted runs: each label must fire on its own plant and spare the others.
"""
from experiments.yu2018_walk import CAP, MPI_RANKS, job_body, stage_exits, stage_summary


def _stage(n_iters, ts_last, rew=100.0):
    step = ts_last // n_iters
    return [{"Iter": i + 1, "TimestepsSoFar": step * (i + 1) if i < n_iters - 1 else ts_last,
             "EpRewMean": rew, "EpLenMean": 200.0} for i in range(n_iters)]


def _stdout(stages_heads_reverts, schedules):
    out = []
    for k, (h, rv) in enumerate(stages_heads_reverts):
        for _ in range(h):
            out += ["********** Iteration 3 ************"] * MPI_RANKS
        for _ in range(rv):
            out += ["Revert Iteration!!!!!"] * MPI_RANKS
        if k < len(schedules):
            out += [f"Current Schedule:  {schedules[k]}"] * MPI_RANKS
    return out


def test_threshold_cap_cut_and_hidden_samples():
    rows = _stage(60, 1_200_000) + _stage(251, CAP + 4_000) + _stage(10, 200_000)
    so = _stdout([(60, 0), (254, 3), (10, 0)], ["[[0.0,[1500,1500]]]", "[[0.0,[1125,1125]]]"])
    st = stage_exits(rows, so, finished=False)
    assert [s["exit"] for s in st] == ["THRESHOLD", "CAP", "CUT"]
    assert st[1]["reverts"] == 3 and st[1]["iter_headers"] == 254
    assert st[1]["samples_drawn"] == 254 * 20_000 > st[1]["timesteps_logged"]
    assert st[0]["schedule_after"] == "[[0.0,[1500,1500]]]"
    assert stage_summary(st) == "1 of 2 stages advanced on the cap"


def test_finished_run_reads_last_stage():
    rows = _stage(5, 100_000) + _stage(7, 140_000)
    st = stage_exits(rows, [], finished=True)
    assert [s["exit"] for s in st] == ["THRESHOLD", "THRESHOLD"]
    rows = _stage(5, 100_000) + _stage(250, CAP)
    assert [s["exit"] for s in stage_exits(rows, [], finished=True)] == ["THRESHOLD", "CAP"]


def test_boundary_and_noise():
    # one sample below the cap is THRESHOLD (the only other exit); rows without Iter are skipped
    rows = [{"x": 1}] + _stage(3, CAP - 1) + _stage(2, 10)
    assert [s["exit"] for s in stage_exits(rows, [], finished=True)] == ["THRESHOLD", "THRESHOLD"]
    assert stage_exits([], [], finished=True) == []


def test_job_body_carries_the_parser():
    src = job_body(60)
    compile(src, "job", "exec")
    assert "def stage_exits" in src and "stages_curriculum" in src


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
    print(f"PASS {n}/{n}")
