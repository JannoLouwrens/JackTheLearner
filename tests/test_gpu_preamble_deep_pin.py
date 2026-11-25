"""Planted: a pin > 50 commits behind main must still check out (J18, 11-25).

Runs the preamble's own clone lines against a local 60-commit repo, so the
code under test is the text the kernel executes, not a re-statement of it.
"""
import subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import gpu


def _git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _run(origin, dest, ref):
    text = gpu.repo_preamble(ref).replace(gpu.REPO_URL, f"file://{origin}")
    text = text.replace("/tmp/jack", str(dest)).split("_sys.path.insert")[0]
    return subprocess.run([sys.executable, "-c", text], capture_output=True, text=True)


def test_deep_pin_and_shallow_pin_and_bad_pin():
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "origin"; o.mkdir()
        _git("init", "-q", "-b", "main", cwd=o)
        shas = []
        for i in range(60):
            (o / "f").write_text(str(i))
            _git("add", "f", cwd=o)
            _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", str(i), cwd=o)
            shas.append(_git("rev-parse", "HEAD", cwd=o))
        for k, ref in enumerate([shas[0], shas[-3]]):   # 59 behind, 2 behind
            d = Path(td) / f"c{k}"
            r = _run(o, d, ref)
            assert r.returncode == 0, r.stderr
            assert _git("rev-parse", "HEAD", cwd=d) == ref
        r = _run(o, Path(td) / "bad", "0" * 40)          # control: must fail
        assert r.returncode != 0


if __name__ == "__main__":
    test_deep_pin_and_shallow_pin_and_bad_pin(); print("PASS")
