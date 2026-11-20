"""J16 (d) wheel kernel: the job body and the WHEEL verdict, before any session.

Known answers, each with controls that must not PASS:
- the job body is valid Python, pins the PyPI wheel, runs exactly the four
  rung commands (no -I: a pip-installed wheel may sit outside the isolated path);
- the verdict PASSES on the box's real output of those four commands
  (conda-forge mlx 0.32.4 — same version, same tests, so the parsing is real);
- it FAILS on: one rc != 0; a "PASS" line with k < n; a different mlx version;
  a planted suite that says HARNESS BLIND; and is NO-READING on a missing run.
"""
import ast
import copy
import os
import subprocess
import sys

from experiments.brain_a import wheel_kernel as W

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_a")
MLX_PY = "/data/venvs/jack-mlx/bin/python"


def test_job_body_parses_and_pins():
    body = W.job_body()
    ast.parse(body)
    assert repr("mlx[cpu]==0.32.4") in body and "__" not in body.replace("__name__", "")
    assert [r[0] for r in W.RUNS] == ["rung1", "rung1_planted", "rung2", "rung2_planted"]
    assert '"-I"' not in body


def _fake(stdouts, rcs=None):
    rcs = rcs or {}
    return {"machine": "x86_64", "libc": ["glibc", "2.35"], "python": "3.11",
            "runs": {k: {"rc": rcs.get(k, 0), "wall_s": 1.0, "stdout_tail": s, "stderr_tail": ""}
                     for k, s in stdouts.items()}}


GOOD = {"rung1": "RUNG 1: 9/9 tests pass eager AND compiled (mlx 0.32.4) -> PASS\n",
        "rung1_planted": "PLANTED: 9/9 caught -> OK\n",
        "rung2": "RUNG 2: 4/4 tests pass eager AND compiled (mlx 0.32.4) -> PASS\n",
        "rung2_planted": "PLANTED: 4/4 caught -> OK\n"}


def test_verdict_planted():
    assert W.verdict(_fake(GOOD))["WHEEL"] == "PASS"
    assert W.verdict(_fake(GOOD, rcs={"rung2": 1}))["first_differing"] == "rung2"
    bad = dict(GOOD, rung1="RUNG 1: 8/9 tests pass eager AND compiled (mlx 0.32.4) -> PASS\n")
    assert W.verdict(_fake(bad))["WHEEL"] == "FAIL"
    bad = dict(GOOD, rung1=GOOD["rung1"].replace("0.32.4", "0.31.0"))
    assert W.verdict(_fake(bad))["WHEEL"] == "FAIL"
    bad = dict(GOOD, rung2_planted="PLANTED: 3/4 caught -> HARNESS BLIND\n")
    assert W.verdict(_fake(bad))["first_differing"] == "rung2_planted"
    part = copy.deepcopy(_fake(GOOD)); del part["runs"]["rung2_planted"]
    assert W.verdict(part)["WHEEL"] == "NO-READING"
    timeout = _fake(GOOD); timeout["runs"]["rung1"] = {"rc": "TIMEOUT", "wall_s": 1200}
    assert W.verdict(timeout)["WHEEL"] == "FAIL"


def test_verdict_on_real_box_output():
    """The four commands exactly as the job runs them (no -I), on the box build."""
    if not os.path.exists(MLX_PY):
        print("SKIP: box conda-forge MLX venv absent"); return
    stdouts, rcs = {}, {}
    for key, script, args, tmo in W.RUNS:
        p = subprocess.run(["nice", "-n", "19", MLX_PY, os.path.join(HERE, script)] + args,
                           capture_output=True, text=True, timeout=tmo)
        stdouts[key], rcs[key] = p.stdout, p.returncode
    v = W.verdict(_fake(stdouts, rcs))
    assert v["WHEEL"] == "PASS", v


def main():
    for t in (test_job_body_parses_and_pins, test_verdict_planted, test_verdict_on_real_box_output):
        t()
        print(f"{t.__name__}: PASS")
    print("test_brain_a_wheel_kernel PASS")


if __name__ == "__main__":
    main()
