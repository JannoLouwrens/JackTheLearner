"""Brain B rows 5-6 (Next-up item 2): bands, cells, verdict and the SimBa env stack.
Run: MUJOCO_GL=disable /data/venvs/jack-std/bin/python -m experiments.tests.test_brain_b_dmc"""
from experiments import brain_b_dmc as D


def _cell(arm, task, seed, score):
    return {"arm": arm, "task": task, "seed": seed, "score": score}


def test_bands_are_the_rule_table():
    assert D.band(853.0, 9.2) == (837.9, 868.1)
    assert D.band(776.5, 30.2) == (726.9, 826.1)
    assert D.band(968.0, 23.2) == (929.9, 1000.0)
    assert D.band(955.8, 30.5) == (905.7, 1000.0)
    assert D.band(853.0, 9.2, k=15) == (841.7, 864.3)     # the one extension


def test_cells_lib_first_and_row6_split():
    c = D.cells_for("5")
    assert c[0][0] == "lib" and len(c) == 5 + 5 + 3 and all(x[3] == 50_000 for x in c)
    assert len(D.cells_for("6")) == 3 * 13
    assert {x[1] for x in D.cells_for("6", tasks=["reacher-easy"])} == {"reacher-easy"}
    try:
        D.cells_for("5", tasks=["reacher-easy"]); raise AssertionError("foreign task accepted")
    except ValueError:
        pass
    try:
        D.kernel("6", 1.0); raise AssertionError("row 6 without a task accepted")
    except ValueError:
        pass
    b = D.job_body("6", tasks=["reacher-hard"])
    compile(b, "job", "exec")
    assert "'reacher-hard'" in b and "dm-control==1.0.50" in b and "--no-deps" in b


def test_verdict_planted():
    t = "cartpole-swingup"
    base = [_cell("lib", t, s, 850) for s in D.SEEDS] + [_cell("ctrl", t, s, 60) for s in D.CTRL_SEEDS]
    ours_in = [_cell("ours", t, s, 845) for s in D.SEEDS]
    assert D.verdict(base + ours_in, "5")[t]["verdict"] == "lib REPRODUCES; ours REPRODUCES"
    # control at the band's lower edge voids the task
    bad = base[:-1] + [_cell("ctrl", t, 2, 837.9)] + ours_in
    assert D.verdict(bad, "5")[t]["verdict"] == "VOID"
    # lib below the band: ours cannot count
    low = [_cell("lib", t, s, 800) for s in D.SEEDS] + base[5:] + ours_in
    assert D.verdict(low, "5")[t]["verdict"].startswith("lib MISSES")
    # a missing lib seed is not a pass
    gap = base[1:] + ours_in
    assert D.verdict(gap, "5")[t]["verdict"].startswith("lib MISSES")
    # an errored ctrl seed leaves ctrl incomplete -> VOID
    err = base[:-1] + [{"arm": "ctrl", "task": t, "seed": 2, "error": "rc=1"}] + ours_in
    assert D.verdict(err, "5")[t]["verdict"] == "VOID"


def test_env_stack_repeat2_500_decisions():
    import numpy as np
    env = D.make_env("cartpole-swingup", 0)
    env.reset(seed=0)
    n, total, done = 0, 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert 0.0 <= r <= 2.0 + 1e-6          # two frames' rewards summed, each in [0, 1]
        total += r; n += 1; done = term or trunc
    assert n == 500 and trunc and not term and total <= 1000.0
    assert float(env.action_space.low[0]) == -1.0 and float(env.action_space.high[0]) == 1.0


def test_extension_cells_and_judge_planted():
    """The rule's ONE extension: lib + ours on seeds 5..14, no ctrl re-run; judged
    at k = 15 over seeds 0..14 with the first kernel's ctrl (band 841.7..864.3)."""
    t = "cartpole-swingup"
    c = D.cells_for("5", D.EXT_SEEDS, arms=("lib", "ours"))
    assert len(c) == 20 and {x[2] for x in c} == set(range(5, 15)) and {x[0] for x in c} == {"lib", "ours"}
    b = D.job_body("5", ext=True)
    compile(b, "job", "exec")
    assert "'ctrl'" not in b and "if False else None" in b
    assert "if True else None" in D.job_body("5")            # the first kernel still judges itself
    ctrl = [_cell("ctrl", t, s, 70) for s in D.CTRL_SEEDS]
    first = {"cells": [_cell("lib", t, s, 850) for s in D.SEEDS] + [_cell("ours", t, s, 850) for s in D.SEEDS] + ctrl}
    ext = {"cells": [_cell("lib", t, s, 850) for s in D.EXT_SEEDS] + [_cell("ours", t, s, 850) for s in D.EXT_SEEDS]}
    v = D.judge_ext(first, ext, "5")[t]
    assert v["band"] == [841.7, 864.3] and v["verdict"] == "lib REPRODUCES; ours REPRODUCES"
    # 838 is inside the k=5 band but outside k=15's: the narrower band binds
    low = {"cells": [_cell("lib", t, s, 838) for s in D.EXT_SEEDS] + ext["cells"][10:]}
    first_low = {"cells": [_cell("lib", t, s, 838) for s in D.SEEDS] + first["cells"][5:]}
    assert D.judge_ext(first_low, low, "5")[t]["verdict"].startswith("lib MISSES")
    # a missing extension seed is not a pass
    gap = {"cells": ext["cells"][1:]}
    assert D.judge_ext(first, gap, "5")[t]["verdict"].startswith("lib MISSES")
    # ctrl still voids
    first_bad = {"cells": first["cells"][:-1] + [_cell("ctrl", t, 2, 841.7)]}
    assert D.judge_ext(first_bad, ext, "5")[t]["verdict"] == "VOID"


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
    print(f"test_brain_b_dmc PASS {n}/{n}")
