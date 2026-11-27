"""S12 (a): `gpu.py` grows an `accelerator` argument.

1. Every existing caller pushes the BYTE-IDENTICAL job: kernel.py, kernel-
   metadata.json and the push command line are compared against the gpu.py of
   commit c4db9a61 (before the edit), loaded side by side, same frozen clock.
2. `submit(prefer="kaggle-tpu")` asks for TpuV6E8 with enable_tpu, skips the
   Pascal torch fix, bills its own weekly line (20 h), never touches the T4
   line and never fails over to colab/T4.
3. Control: the pre-edit gpu.py has no TPU door — the same call is refused.

Run: python -m tests.test_gpu_accelerator  (CLI stubbed; no network, no quota)
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import gpu  # noqa: E402

PRE_EDIT = "c4db9a61"
EPOCH = 1795200000.0


class _Clock:
    def __init__(self):
        self.now = EPOCH

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def __getattr__(self, k):
        import time
        return getattr(time, k)


class _Sub:
    @staticmethod
    def run(cmd, **kw):
        class R:
            returncode, stdout, stderr = 0, "  - username: stubuser\n", ""
        return R()


def _drive(mod, call):
    """Run `call(mod, script, budget, journal)` with mod's CLI stubbed; return
    (result, pushes, budget path, colab calls). Each push = (argv minus the
    temp dir, kernel.py bytes, metadata dict)."""
    pushes, colab = [], []

    def cli(cmd, timeout):
        if "push" in cmd:
            d = Path(cmd[cmd.index("-p") + 1])
            pushes.append(([c for c in cmd if c != str(d)],
                           (d / "kernel.py").read_bytes(),
                           json.loads((d / "kernel-metadata.json").read_text())))
            return 0, "Kernel version 1 successfully pushed.", ""
        if "status" in cmd:
            return 0, 'has status "KernelWorkerStatus.COMPLETE"', ""
        return 0, "", ""

    saved = (mod._run, mod.subprocess, mod.time, mod.run_on_colab, mod.STDOUT_DIR)
    td = tempfile.mkdtemp(dir="/data")
    try:
        mod._run, mod.subprocess, mod.time = cli, _Sub(), _Clock()
        mod.run_on_colab = lambda *a, **k: (colab.append(1) or mod.JobResult(
            "colab", False, message="stub"))
        mod.STDOUT_DIR = Path(td) / "stdout"
        script = Path(td) / "job.py"
        script.write_text("print('job')\n")
        bpath = Path(td) / "budget.json"
        res = call(mod, script, mod.Budget(bpath), Path(td) / "receipts.jsonl")
    finally:
        mod._run, mod.subprocess, mod.time, mod.run_on_colab, mod.STDOUT_DIR = saved
    return res, pushes, bpath, colab


def _load_pre_edit():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{PRE_EDIT}:experiments/gpu.py"],
                         capture_output=True, text=True, check=True).stdout
    f = Path(tempfile.mkdtemp(dir="/data")) / "gpu_pre_s12.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("experiments.gpu_pre_s12", f)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "experiments"
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    old = _load_pre_edit()
    checks = {}
    calls = {
        "t4": lambda m, s, b, j: m.submit(s, prefer="kaggle", est_hours=0.1,
                                          timeout_s=600, budget=b, journal=j),
        "cpu": lambda m, s, b, j: m.submit(s, prefer="kaggle-cpu", est_hours=0.1,
                                           timeout_s=600, budget=b, journal=j),
        "direct": lambda m, s, b, j: m.run_on_kaggle(s, 600, None, j),
    }
    for name, call in calls.items():
        _, new_p, _, _ = _drive(gpu, call)
        _, old_p, _, _ = _drive(old, call)
        checks[f"default_{name}_byte_identical"] = bool(new_p) and new_p == old_p

    tpu = lambda m, s, b, j: m.submit(s, prefer="kaggle-tpu", est_hours=0.5,
                                      timeout_s=600, budget=b, journal=j)
    res, p, bpath, colab = _drive(gpu, tpu)
    cmd, body, meta = p[-1] if p else ([], b"", {})
    b = gpu.Budget(bpath)
    checks["tpu_asks_for_tpu"] = (cmd[-2:] == ["--accelerator", "TpuV6E8"]
                                  and meta.get("enable_tpu") is True
                                  and meta.get("enable_gpu") is False
                                  and meta.get("id", "").startswith("stubuser/jack-tpu-"))
    checks["tpu_no_torch_fix"] = body == b"print('job')\n"
    # read on the stub clock the charge was filed with: on the wall clock the
    # check went red once the real week left EPOCH's W46 (failed from 11-21)
    real_time, gpu.time = gpu.time, _Clock()
    try:
        checks["tpu_own_weekly_line"] = (res.ok and b.used_hours("kaggle-tpu") > 0
                                         and b.used_hours("kaggle") == 0.0
                                         and b.remaining("kaggle-tpu") < 20.0)
    finally:
        gpu.time = real_time
    checks["tpu_never_fails_over"] = len(p) == 1 and not colab

    # refused when the TPU line is spent: no push at all
    def spent(m, s, bud, j):
        bud.charge("kaggle-tpu", 20 * 3600, ok=True, job_id="x/spent")
        return m.submit(s, prefer="kaggle-tpu", est_hours=0.5, timeout_s=600,
                        budget=bud, journal=j)
    res2, p2, _, colab2 = _drive(gpu, spent)
    checks["tpu_spent_is_refused"] = (not res2.ok and not p2 and not colab2
                                      and "kaggle-tpu" in res2.message)

    # control: the pre-edit module cannot reach a TPU
    _, cp, _, _ = _drive(old, tpu)
    control_fails = not any(c[0][-2:] == ["--accelerator", "TpuV6E8"] for c in cp)

    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(f"{'PASS' if control_fails else 'FAIL'} control_pre_edit_has_no_tpu")
    ok = all(checks.values()) and control_fails
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
