"""j17_onset.hunt: the SimBa hunt reading, tested on planted files built from
the 20 MLP J17 runs BEFORE the hunt kernel is home (11-22)."""
import json
import tempfile
from pathlib import Path

from experiments import j17_onset as J

S, T = J.HUNT_ARMS


def _plant(sighted, twin, d):
    raw = {}
    for f in sorted(J.DATA.glob("j17_r*.json")):
        raw.update(json.loads(f.read_text())["cells"])
    cells = {f"{S}_s923": raw[sighted], f"{T}_s923": raw[twin]}
    p = Path(d) / "h.json"
    p.write_text(json.dumps({"cells": cells}))
    return J.hunt(p)


def test_cut_points_are_the_mlp_group_edges():
    q = J.mlp_q_climbs()
    assert max(q[J.BLIND]) <= J.GAP_LO < J.GAP_HI <= min(q["K-row128"] + q["K-rays16"])


def test_planted_readings():
    with tempfile.TemporaryDirectory() as d:
        assert _plant("K-row128_s923", "K-row128-blind_s923", d)["verdict"] == "STILL RUNS AWAY"
        assert _plant("K-row128-blind_s921", "K-row128-blind_s923", d)["verdict"] == "HELD"
        assert _plant("K-row128-blind_s923", "K-row128_s923", d)["verdict"] == "NO CONTROL"
        assert J.hunt(Path(d) / "absent.json")["verdict"].startswith("VOID")


if __name__ == "__main__":
    test_cut_points_are_the_mlp_group_edges(); test_planted_readings(); print("PASS")
