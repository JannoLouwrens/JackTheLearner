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
    assert "train_segment(0, out, None, 0)" in b
    assert "sbx-rl==0.28.0" in b and "gymnasium==1.4.0" in b and "mujoco==3.17.0" in b
    compile(b, "job", "exec")


def test_later_segment_needs_prev_and_fresh_refuses_prev():
    assert "train_segment(1, out, '/kaggle/input/x', 2)" in H.segment_job_body(1, 2, "/kaggle/input/x")
    _raises(ValueError, H.segment_job_body, 0, 1)
    _raises(ValueError, H.segment_job_body, 0, 0, "/kaggle/input/x")


def test_seed_outside_rule_refused_and_checkpoint_never_fetched():
    _raises(ValueError, H.segment_job_body, 3, 0)
    assert H.SEGMENT_FETCH == ["segment.json"]


def test_kernel_refuses_later_segment_until_kernel_sources_exist():
    _raises(RuntimeError, H.kernel, 0, 1)


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
    print(f"PASS {n}/{n}")
