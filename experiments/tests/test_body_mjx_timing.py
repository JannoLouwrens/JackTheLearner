"""J28 cell 1: the MJX timing body compiles, both levels, and carries the cut and the fetch name."""
from experiments.body import mjx_timing as M


def test_job_and_probe_compile():
    b = M.job_body()
    compile(b, "job", "exec")
    probe = b.split('PROBE = r"""')[1].split('"""')[0]
    compile(probe, "probe", "exec")
    assert str(M.TIMESTEPS) in probe and "HumanoidStand" in probe and repr(M.FETCH) in b
    assert "nvidia-smi" in b          # the GPU kind is printed before anything can fail


def test_installs_a_real_pypi_name_and_forces_the_jax_path():
    """RESEARCH 11-27 04:11: `mujoco_playground` is a PyPI 404 and 0.2.0's humanoid defaults to
    impl="warp" — the cell must install `playground==0.2.0` and time the JAX/MJX path."""
    b = M.job_body()
    install = [l for l in b.splitlines() if '"pip", "install"' in l]
    assert len(install) == 1 and '"playground==0.2.0"' in install[0]
    assert '"mujoco_playground"' not in install[0]
    probe = b.split('PROBE = r"""')[1].split('"""')[0]
    assert 'config_overrides={"impl": "jax"}' in probe and "VERSIONS " in probe
