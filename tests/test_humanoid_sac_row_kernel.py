"""J10 (2'): the Kaggle job text for a resume segment is the frozen rule's call."""
from experiments import humanoid_sac_row as H


def _raises(exc, f, *a):
    try:
        f(*a)
    except exc:
        return
    raise AssertionError(f"{f.__name__}{a} did not raise {exc.__name__}")


def test_segment0_body_calls_train_segment_fresh():
    b = H.segment_job_body(0, 0)
    assert "_p = None" in b and "train_segment(0, out, _p, 0)" in b
    assert "sbx-rl==0.28.0" in b and "gymnasium==1.4.0" in b and "mujoco==3.17.0" in b
    compile(b, "job", "exec")


def test_later_segment_needs_prev_and_fresh_refuses_prev():
    b = H.segment_job_body(1, 2, "/kaggle/input/x")
    assert "_p = '/kaggle/input/x'" in b and "resolve_prev_dir(_p)" in b
    assert "train_segment(1, out, _p, 2)" in b
    compile(b, "job", "exec")
    _raises(ValueError, H.segment_job_body, 0, 1)
    _raises(ValueError, H.segment_job_body, 0, 0, "/kaggle/input/x")


def test_seed_outside_rule_refused_and_checkpoint_never_fetched():
    _raises(ValueError, H.segment_job_body, 3, 0)
    assert H.SEGMENT_FETCH == ["segment.json"]


def test_kernel_refuses_later_segment_without_previous_record():
    import os
    if not os.path.exists("/data/jack-logs/j10/sac_row/s2_seg4.json"):
        _raises(RuntimeError, H.kernel, 2, 5)


def test_prev_dir_found_where_kaggle_actually_mounts_it():
    import os, tempfile
    with tempfile.TemporaryDirectory() as root:
        def ckpt(*parts, files=("model.zip", "replay.pkl")):
            d = os.path.join(root, *parts); os.makedirs(d, exist_ok=True)
            for f in files:
                open(os.path.join(d, f), "w").close()
            return d
        exp = os.path.join(root, "jack-cpu-1")
        _raises(FileNotFoundError, H.resolve_prev_dir, exp, root)          # nothing mounted
        ckpt("jack-cpu-1", files=("model.zip",))                            # half a checkpoint
        _raises(FileNotFoundError, H.resolve_prev_dir, exp, root)
        ckpt("other", "jack-cpu-2")                                         # another kernel's
        _raises(FileNotFoundError, H.resolve_prev_dir, exp, root)
        deep = ckpt("notebooks", "owner", "jack-cpu-1")                     # the deeper mount
        assert H.resolve_prev_dir(exp, root) == deep
        ckpt("x", "jack-cpu-1")                                             # ambiguous
        _raises(FileNotFoundError, H.resolve_prev_dir, exp, root)
        ckpt("jack-cpu-1")                                                  # the rule's own path wins
        assert H.resolve_prev_dir(exp, root) == exp


def test_output_pattern_selects_only_fetched_basenames():
    import re
    from experiments.gpu import output_pattern
    assert output_pattern(None) == [] and output_pattern([]) == []
    flag, pat = output_pattern(["segment.json", "a.b+c.json"])
    assert flag == "--file-pattern"
    for name, want in [("segment.json", True), ("sub/segment.json", True),
                       ("a.b+c.json", True), ("model.zip", False),
                       ("replay.pkl", False), ("xsegment.json", False),
                       ("segment.json.bak", False), ("aXb+c.json", False)]:
        assert bool(re.search(pat, name)) is want, name


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
    print(f"PASS {n}/{n}")
