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


def test_twin_cells_and_job():
    """J19's cause twin: lib only, the row's seeds and steps, no ctrl, no verdict."""
    c = D.twin_cells("reacher-hard")
    assert c == [("lib", "reacher-hard", s, 150_000) for s in D.SEEDS]
    b = D.twin_job_body("reacher-hard")
    compile(b, "job", "exec")
    assert "save=True" in b and "'ours'" not in b and "'ctrl'" not in b and "if False else None" in b
    assert "save=True" not in D.job_body("6", tasks=["reacher-hard"])     # the row's kernel is unchanged
    m = D.twin_models("reacher-hard")
    assert len(m) == 30 and m[0] == "twin_lib_reacher-hard_s0_25000.msgpack" and m[-1].endswith("_s4_150000.msgpack")
    try:
        D.twin_cells("cartpole-swingup"); raise AssertionError("a row-5 task accepted")
    except ValueError:
        pass


def test_twin_reads_only():
    """The twin's two additions (save at every mark, trace) change NOTHING the
    row measures: same seed with them on and off -> identical evals, marks and
    training returns. Control: another seed differs (the check can fail). And
    each saved model IS the one that was evaluated: loaded back, it replays
    its mark's ten episode returns exactly. Control: the first mark's model
    does not replay the final mark's returns."""
    import tempfile
    old = (D.LEARNING_STARTS, D.EVAL_EVERY, D.TWIN_TRACE_EVERY)
    D.LEARNING_STARTS, D.EVAL_EVERY, D.TWIN_TRACE_EVERY = 300, 500, 250
    try:
        task, steps = "reacher-hard", 1500
        d = tempfile.mkdtemp(prefix="j19twin_test_", dir="/data/tmp")
        on = D.run_cell("lib", task, 0, steps, save_dir=d)
        off = D.run_cell("lib", task, 0, steps)
        other = D.run_cell("lib", task, 1, steps)
        for k in ("eval", "marks", "train_returns", "score"):
            assert on[k] == off[k], k
        assert other["train_returns"] != off["train_returns"], "control: another seed must differ"
        assert "trace" not in off and "models" not in off
        assert [r["t"] for r in on["trace"]] == [250, 500, 750, 1000, 1250, 1500]
        late = on["trace"][-1]
        assert all(isinstance(late[k], float) for k in ("critic_loss", "actor_loss", "ent_coef")), late
        assert on["models"] == [D.twin_model_name("lib", task, 0, n) for n in (500, 1000, 1500)]
        import os
        for n in (500, 1000, 1500):
            m = D.load_weights("lib", task, 0, os.path.join(d, D.twin_model_name("lib", task, 0, n)))
            assert D.evaluate(m, task, 0) == on["mark_evals"][str(n)], n
        fresh = D.make_model("lib", task, 0, 1)          # control: unloaded weights do not replay it
        assert D.evaluate(fresh, task, 0) != on["mark_evals"]["1500"], "control: a fresh model must differ"
        assert on["mark_evals"]["500"] != on["mark_evals"]["1500"], "control: marks must differ"
        # J19 step (2)'s re-evaluation, on the files just written:
        r = D.reeval_seed(task, 0, d, marks=(500, 1500), episodes=10, kernel_cell=on)
        for n in ("500", "1500"):                       # fixed: its 10 ARE the kernel's 10, at every mark
            assert r["marks"][n]["fixed"] == on["mark_evals"][n] and r["marks"][n]["replay_gap"] == 0.0
        m = D.load_weights("lib", task, 0, os.path.join(d, on["models"][-1]))
        env = D.make_env(task, 0)
        a, b = D._episodes(m, env, 3), D._episodes(m, env, 3)
        assert a != b, "stream: the same model must meet NEW targets on the next draw"
        fx = [D._episodes(m, D.make_env(task, 10_000), 3, reset_seed=10_000) for _ in range(2)]
        assert fx[0] == fx[1], "fixed: the same model must meet the SAME targets again"
        assert r["marks"]["1500"]["stream"] != r["marks"]["1500"]["fixed"]
        planted = {"task": task, "band": [905.7, 1000.0], "fixed_mean_at_target": 1.0, "fixed_in_band": False,
                   "stream_mean_at_target": 950.0, "stream_in_band": True,
                   "seeds": {"0": {"marks": {str(D.STEPS["6"]): r["marks"]["1500"]}}}}
        tb = D.reeval_table(planted)
        assert "fixed 1.0 (OUT OF band" in tb and "stream 950.0 (IN band)" in tb, tb
        size = os.path.getsize(os.path.join(d, on["models"][-1]))
        assert size < 1_500_000, size
        print("model file bytes:", size)
    finally:
        D.LEARNING_STARTS, D.EVAL_EVERY, D.TWIN_TRACE_EVERY = old
        import shutil
        shutil.rmtree(d, ignore_errors=True)




def test_b6c2_cells_job_and_score_planted():
    """Row B-6c-2 (J23, rule 07df7d63): lib on FRESH seeds 5-9 + ctrl 0-2, reacher-hard, 150k,
    every cell saves its models; the verdict reads stream-100 at 150k, band unchanged."""
    c = D.b6c2_cells()
    assert c == ([("lib", "reacher-hard", s, 150_000) for s in (5, 6, 7, 8, 9)]
                 + [("ctrl", "reacher-hard", s, 150_000) for s in (0, 1, 2)])
    b = D.b6c2_job_body()
    compile(b, "job", "exec")
    assert "save=True" in b and "'ours'" not in b and "if False else None" in b
    m = D.b6c2_models()
    assert len(m) == 8 * 6 and "twin_ctrl_reacher-hard_s2_150000.msgpack" in m
    lo, _ = D.band(*D.ROWS["6"]["reacher-hard"])

    def res(lib, ctrl):
        seeds = {}
        for arm, xs, ss in (("lib", lib, (5, 6, 7, 8, 9)), ("ctrl", ctrl, (0, 1, 2))):
            for s, x in zip(ss, xs):
                seeds[f"{arm}_s{s}"] = {"marks": {"150000": {"stream_mean": x, "fixed_mean": 0.0}}}
        return seeds
    assert D.b6c2_verdict(res([910] * 5, [1, 2, 3]))["verdict"] == "lib REPRODUCES"
    assert D.b6c2_verdict(res([900] * 5, [1, 2, 3]))["verdict"] == "lib MISSES (named, final)"
    assert D.b6c2_verdict(res([990] * 5, [1, 2, lo]))["verdict"] == "VOID"       # a control that does not fail
    assert D.b6c2_verdict(res([990] * 4, [1, 2, 3]))["verdict"] == "INCOMPLETE"
    # the fixed reading is printed beside, never read: a fixed miss does not move a stream pass
    r = res([910] * 5, [1, 2, 3]); r["lib_s5"]["marks"]["150000"]["fixed_mean"] = 0.0
    assert D.b6c2_verdict(r)["verdict"] == "lib REPRODUCES"


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
    print(f"test_brain_b_dmc PASS {n}/{n}")
