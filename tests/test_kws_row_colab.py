"""kws_row's Colab chain for N7-kws-2 (ruling 106): plumbing on, 2.4 h run deadline, one run a job, stop on a bad run."""
import json

from experiments import kws_row as K


def _full(name, seed, twin=False):
    return {"name": name, "seed": seed, "twin": twin, "test": 96.9, "rc": 0, "cut": False, "stopped": None,
            "counts": list(K.EXPECT_COUNTS), "valid": [90.0] * K.EPOCHS,
            **({"shuffled": [36923, 36923]} if twin else {})}


def test_body_carries_plumb_and_deadline():
    b = K.kernel_body(["s0"], "kws2_s0", plumb=True, deadline_s=K.COLAB_RUN_DEADLINE_S)
    assert "if True:" in b and 'os.environ["KWS_PLUMB"] = "1"' in b
    assert "deadline_s=%r" % K.COLAB_RUN_DEADLINE_S in b
    plain = K.kernel_body(["s0"], "t")
    assert "if False:" in plain and "deadline_s=None or RUN_DEADLINE_S" in plain   # Kaggle path unchanged
    compile(b, "k", "exec"); compile(plain, "k", "exec")
    assert K.COLAB_RUN_DEADLINE_S == 2.4 * 3600   # the frozen fit rule


def test_chain_one_job_per_run_in_order(tmp_path):
    seen = []

    def fake(body, tag):
        seen.append(tag)
        assert 'os.environ["KWS_PLUMB"] = "1"' in body
        run = tag.split("_", 1)[1]
        seed, twin = K.plan_from_args([run])[0]
        return {"stopped": None, "runs": [_full(K.run_name(seed, twin), seed, twin)]}
    assert K.colab_chain(["s0", "s1", "s2", "twin"], fake, str(tmp_path)) == 0
    assert seen == ["kws2_s0", "kws2_s1", "kws2_s2", "kws2_twin"]
    # relaunch: every run home and complete -> nothing submitted
    seen.clear()
    assert K.colab_chain(["s0", "s1", "s2", "twin"], fake, str(tmp_path)) == 0 and seen == []


def test_chain_stops_on_cut_run(tmp_path):   # planted: a cut run must not let the chain spend the next session
    seen = []

    def fake(body, tag):
        seen.append(tag)
        r = _full("s0", 0); r["cut"] = True
        return {"stopped": None, "runs": [r]}
    assert K.colab_chain(["s0", "s1"], fake, str(tmp_path)) == 1
    assert seen == ["kws2_s0"]
    assert json.load(open(tmp_path / "kws2_s0.json"))["runs"][0]["cut"] is True


def test_chain_stops_on_submit_fail(tmp_path):
    assert K.colab_chain(["s0", "s1"], lambda b, t: None, str(tmp_path)) == 1


if __name__ == "__main__":   # no pytest in the project venv
    import pathlib, sys, tempfile
    bad = 0
    for n, f in [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]:
        try:
            f(pathlib.Path(tempfile.mkdtemp())) if f.__code__.co_argcount else f()
            print("PASS", n, flush=True)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e), flush=True)
    sys.exit(1 if bad else 0)
