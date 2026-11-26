"""J28 cell 1: the MJX timing body compiles, both levels, and carries the cut and the fetch name."""
from experiments.body import mjx_timing as M


def test_job_and_probe_compile():
    b = M.job_body()
    compile(b, "job", "exec")
    probe = b.split('PROBE = r"""')[1].split('"""')[0]
    compile(probe, "probe", "exec")
    assert str(M.TIMESTEPS) in probe and "HumanoidStand" in probe and repr(M.FETCH) in b
    assert "nvidia-smi" in b          # the GPU kind is printed before anything can fail
