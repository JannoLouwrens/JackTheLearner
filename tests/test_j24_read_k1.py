"""j24_read --variant k1: the default J24 read is unchanged; the K1 read refuses a
draft prereg / a missing pin, and reads K1 raw files by J24's rule unchanged (planted).

    python tests/test_j24_read_k1.py
"""
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

READ_JSON = Path("/data/jack-logs/j24/read.json")


def fresh():
    import experiments.j24_read as r
    return importlib.reload(r)


def refuses(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"did not refuse ({needle})")


def test_default_read_unchanged():
    if not READ_JSON.exists():
        print("SKIP: J24 raw not on this box"); return
    r = fresh()
    out = r.main([])
    want = json.loads(READ_JSON.read_text())
    assert out["branch"] == want["branch"] == "NOT SHOWN AT 500k"
    for a in r.ARMS:
        for sd in r.SEEDS:
            assert math.isclose(out["scores"][a][sd], want["scores"][a][str(sd)])


def test_k1_refuses_a_draft_and_a_missing_pin():
    r = fresh()
    refuses(lambda: r.main(["--variant", "k1"]), "needs --pin")
    refuses(lambda: r.set_k1("HEAD"), "not FROZEN")          # the prereg is a draft
    refuses(lambda: r.set_k1("deadbeef" * 5), "not a commit")
    refuses(lambda: r.main(["--variant", "k2"]), "no variant")
    assert r.PIN.startswith("03e6285d") and r.TAG == "j24"   # a refusal moves nothing


def test_k1_reads_its_own_files_by_the_same_rule():
    r = fresh()
    r.PIN, r.ARMS, r.TAG = "f" * 40, r.K1["arms"], r.K1["tag"]
    real = r.pilot_b2.j17_late_meals
    r.pilot_b2.j17_late_meals = lambda rec, late: rec["late"]
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            r.RAW = d
            for sd in r.SEEDS:
                cells = {f"{a}_s{sd}": {"decisions": r.T, "late": m,
                                        "lives": [{"decisions": r.T}]}
                         for a, m in zip(r.ARMS, (9, 2, 3))}
                (d / f"j24k1_s{sd}.json").write_text(
                    json.dumps({"pin": "f" * 40, "cells": cells}))
            assert r.main([])["branch"] == "EATS"             # 9 >= 2 x 3 on 3/3
            for sd in r.SEEDS:                                # 5 < 2 x 3: not EATS
                p = d / f"j24k1_s{sd}.json"
                j = json.loads(p.read_text())
                j["cells"][f"K1-bins16_s{sd}"]["late"] = 5
                p.write_text(json.dumps(j))
            assert r.main([])["branch"] == "EXTEND_1.5M"
            (d / "j24k1_s941.json").write_text(json.dumps({"pin": "0" * 40, "cells": {}}))
            refuses(lambda: r.main([]), "not the freeze")
    finally:
        r.pilot_b2.j17_late_meals = real


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("PASS", n)
