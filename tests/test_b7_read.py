"""Planted tests for the B-7 reader (experiments/b7_read.py): the frozen rule can say
REPRODUCES, MISS and VOID, and refuses a draft pin or a missing seed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile  # noqa: E402

from experiments import b7_read  # noqa: E402
from experiments.b7mem_launch import FROZEN_MARK  # noqa: E402

PIN = "c" * 40
FROZEN_ROW = f"## Row B-7 x\n{FROZEN_MARK} ...\n"


def _evals(level):
    return [{"step": s, "ret": level} for s in range(2000, 51001, 1000)]


def _plant(tmp, mem_level, flat_level, lost=None, pin=PIN, drop=None):
    m, f = tmp / "mem", tmp / "flat"
    m.mkdir(); f.mkdir()
    for s in range(4):
        rec = {"arm": "B7mem", "seed": s, "complete": s != lost, "evals": _evals(mem_level), "pin": pin}
        if s == lost:
            rec["lost"] = "stalled"
        if s != drop:
            (m / f"b7mem_s{s}.json").write_text(json.dumps(rec))
        (f / f"b7flat_s{s}.json").write_text(json.dumps(
            {"arm": "B7flat", "seed": s, "complete": True, "evals": _evals(flat_level)}))
    return m, f


def _read(m, f, row=FROZEN_ROW):
    return b7_read.read(m, f, row_fn=lambda pin: row)


def test_reproduces(tmp_path):
    assert _read(*_plant(tmp_path, -170.0, -1490.0))["verdict"] == "REPRODUCES"


def test_miss(tmp_path):
    r = _read(*_plant(tmp_path, -300.0, -1490.0))
    assert r["verdict"] == "MISS" and r["arm_score"] == -300.0


def test_twin_inside_band_voids(tmp_path):
    assert _read(*_plant(tmp_path, -170.0, -180.0))["verdict"] == "VOID"


def test_lost_seed_voids_never_replaced(tmp_path):
    r = _read(*_plant(tmp_path, -170.0, -1490.0, lost=2))
    assert r["verdict"] == "VOID" and r["mem_seed_scores"][2] is None


def test_draft_pin_refused(tmp_path):
    try:
        _read(*_plant(tmp_path, -170.0, -1490.0), row="## Row B-7 x\n- draft only\n")
    except b7_read.Refused:
        return
    raise AssertionError("draft pin read")


def test_missing_seed_refused(tmp_path):
    try:
        _read(*_plant(tmp_path, -170.0, -1490.0, drop=3))
    except b7_read.Refused:
        return
    raise AssertionError("missing seed read")


def test_real_freeze_commit():
    """The real pre-freeze commit is refused and the real freeze commit accepted."""
    from experiments.b7mem_launch import frozen, row_at
    assert frozen(row_at("c848d888")) and not frozen(row_at("c848d888^"))


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            try:
                if inspect.signature(f).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        f(Path(d))
                else:
                    f()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    raise SystemExit(1 if fails else 0)
