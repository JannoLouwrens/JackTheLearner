"""S7(3) planted test: JACK_REUSE_KERNEL applies to the first submission only.

(a) The first ask for a slug is granted; (b) planted: a second ask for the
same slug (bare or user/slug) is refused through `run_on_kaggle` itself, before
any Kaggle call; (c) a different slug is still granted; (d) no env → no-op.
Control that must fail: with the claim forgotten (the old behaviour, where the
env var alone decided), the second ask is granted — else (b) measured nothing.
"""
import os
from pathlib import Path

from experiments import gpu


def main():
    gpu._REUSED_SLUGS.clear()
    assert gpu._claim_reuse("u/jack-cpu-1700000000") is None, "(a)"
    os.environ["JACK_REUSE_KERNEL"] = "jack-cpu-1700000000"
    try:
        res = gpu.run_on_kaggle(Path("/nonexistent.py"), cpu=True)
    finally:
        os.environ.pop("JACK_REUSE_KERNEL", None)
    assert not res.ok and "already used" in res.message, f"(b) {res}"
    assert gpu._claim_reuse("jack-cpu-1700000001") is None, "(c)"
    assert gpu._claim_reuse("") is None and "" not in gpu._REUSED_SLUGS, "(d)"
    gpu._REUSED_SLUGS.clear()  # control: the old, env-only rule
    assert gpu._claim_reuse("jack-cpu-1700000000") is None, "control"
    gpu._REUSED_SLUGS.clear()
    print("PASS test_reuse_kernel_once: 4/4")


if __name__ == "__main__":
    main()
