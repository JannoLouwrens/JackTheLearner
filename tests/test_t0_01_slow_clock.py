"""S21 (ruling 132): T0.01 judges the FASTEST of three fresh-process imports.

A module that sleeps 6 s at every import is slow on all three tries and FAILS;
one stalled once (first try only) passes, with every try's time printed.
"""
import tempfile
from pathlib import Path

from experiments.tests import t0_01_imports as T


def _plant(d: Path):
    (d / "sleeper.py").write_text("import time\ntime.sleep(6)\n")
    (d / "stalled_once.py").write_text(
        "import time, pathlib\n"
        "m = pathlib.Path(__file__).with_suffix('.stalled')\n"
        "if not m.exists():\n    m.write_text('x')\n    time.sleep(6)\n")
    (d / "broken.py").write_text("raise ImportError('planted')\n")


def test_sleeper_fails_stall_passes_broken_is_failed():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _plant(d)
        res = T.time_imports(["sleeper", "stalled_once", "broken"], d)
        for n, r in res.items():
            print(n, r)
        sl, st, br = res["sleeper"], res["stalled_once"], res["broken"]
        assert len(sl["times"]) == 3 and min(sl["times"]) > T.SLOW_LINE_S
        assert st["error"] is None and len(st["times"]) == 2
        assert st["times"][0] > T.SLOW_LINE_S and min(st["times"]) <= T.SLOW_LINE_S
        assert br["error"] == "ImportError" and br["times"] == []


if __name__ == "__main__":
    test_sleeper_fails_stall_passes_broken_is_failed()
    print("PASS 1/1")
