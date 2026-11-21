"""INBOX 11-21: a finished kernel's output download may not eat its result.

A2b/A3 (J14) left 0.6-1.2 GB of logs; the whole-output pull timed out and
`TimeoutExpired` escaped after the GPU job completed (no receipt, no charge).
1. first pull times out -> named files re-fetched into a fresh dir, note says so
2. both pulls time out -> no exception (the caller still charges), note says so
3. a pull that finishes -> note "" and the first dir
Control: the pre-fix inline call (bare `_run`) under the same stub raises.

Run: python -m tests.test_gpu_output_timeout  (CLI stubbed; no network)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import gpu  # noqa: E402


def stub(timeouts):
    calls = []

    def run(cmd, timeout):
        calls.append(cmd)
        d = Path(cmd[cmd.index("-p") + 1])
        if len(calls) <= timeouts:
            (d / "partial.json").write_text("{trunc")
            raise subprocess.TimeoutExpired(cmd, timeout)
        (d / "r.json").write_text('{"ok": 1}')
        (d / "slug.log").write_text("[]")
        return 0, "", ""
    return run, calls


def main():
    bad, real = 0, gpu._run
    try:
        for n, want_dir, want_note in [(1, "out_named", "re-fetched only r.json"),
                                       (2, "out", "timed out"), (0, "out", "")]:
            gpu._run, calls = stub(n)
            with tempfile.TemporaryDirectory() as t:
                d, note = gpu._kaggle_fetch_output("u/slug", Path(t), ["r.json"], False)
                ok = d.name == want_dir and (want_note in note if want_note else note == "")
                if n == 1:
                    ok &= (d / "r.json").exists() and not (d / "partial.json").exists()
                    ok &= "--file-pattern" in calls[1] and "--file-pattern" not in calls[0]
                print("PASS" if ok else "FAIL", f"timeouts={n}", d.name, repr(note))
                bad += not ok
        gpu._run, _ = stub(1)                    # control: the pre-fix call
        with tempfile.TemporaryDirectory() as t:
            try:
                gpu._run([gpu.KAGGLE, "kernels", "output", "u/slug", "-p", t], 300)
                print("FAIL control: bare _run did not raise"); bad += 1
            except subprocess.TimeoutExpired:
                print("PASS control: bare _run raises TimeoutExpired")
    finally:
        gpu._run = real
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
