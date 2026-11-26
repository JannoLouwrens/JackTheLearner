"""J28 cell 2: fp16 on/off timing — one key moves; a crashed cell reads None, never a speed."""
from experiments.brain_a import fp16_timing as F
from experiments.brain_a import rows as R


def test_arms_differ_in_precision_only():
    off, on = F.arm(F.PRECISIONS["off"]), F.arm(F.PRECISIONS["on"])
    assert off == R.enc_base()          # OFF is the frozen arm itself
    diff = [(x, y) for x, y in zip(off, on) if x != y]
    assert len(on) == len(off) and diff == [("fabric.precision=32-true", "fabric.precision=16-mixed")]


def test_kernel_body_compiles():
    t = F.kernel_body()
    compile(t, "fp16", "exec")
    assert "torch==2.6.0" in t and "TORCHINDUCTOR_COMPILE_THREADS" in t and "from experiments.brain_a.fp16_timing import" in t


def test_summary_planted():
    p = {"sps_env": 100.0, "sps_train": 10.0}          # 10 s + 50 s = 60 s per 1k
    q = {"sps_env": 200.0, "sps_train": 20.0}          # 30 s
    s = F.summarise({"off": {"rc": 0, "projection": p}, "on": {"rc": 0, "projection": q}})
    assert s["arms"]["off"]["sec_per_1k_decisions"] == 60.0 and s["arms"]["on"]["sec_per_1k_decisions"] == 30.0
    assert s["speedup_on_vs_off"] == 2.0
    crashed = F.summarise({"off": {"rc": 0, "projection": p}, "on": {"rc": -11, "projection": None}})
    assert crashed["arms"]["on"]["sec_per_1k_decisions"] is None and "speedup_on_vs_off" not in crashed
