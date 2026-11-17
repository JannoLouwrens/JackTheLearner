"""pilot_b2.cells_from_artifacts: a kernel with no summary file is assembled
from its per-cell files; unfinished cells are marked cut, missing ones
not_found, a .tmp is ignored, and nothing at all returns None (control)."""
import json

from experiments import pilot_b2 as P


def _cell(arm, seed, done):
    life = {"decisions": 2576, "seconds": 515.2, "floor_meals": 0, "apples": 0,
            "cause": "energy", "intake": 0.0, "basal_drain": 1.3,
            "sustained": False, "reward": 0.0, "spawn_dist": 3.0}
    return {"arm": arm, "seed": seed, "done": done, "lives": [life] * 10}


def test_assemble_from_cell_files(tmp_path):
    cells = P.card_groups()["c0"]
    art = {}
    for (arm, seed), done in zip(cells[:2], (True, False)):
        f = tmp_path / f"card_{arm}_s{seed}.json"
        f.write_text(json.dumps(_cell(arm, seed, done)))
        art[f.name] = str(f)
    art["card_bogus.json.tmp"] = str(f)
    out = P.cells_from_artifacts(art, cells)["cells"]
    k0, k1 = (f"{a}_s{s}" for a, s in cells[:2])
    assert len(out) == 8
    assert not out[k0].get("deadline_cut") and len(out[k0]["lives"]) == 10
    assert out[k1]["deadline_cut"]
    assert sum(bool(v.get("not_found")) for v in out.values()) == 6
    (tmp_path / "c0.json").write_text(json.dumps({"cells": out}))
    assert "| 1 (1) | 10 |" in P.card_table([tmp_path / "c0.json"])
    assert P.cells_from_artifacts({}, cells) is None


def main():
    import sys
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(dir="/data") as d:
        test_assemble_from_cell_files(Path(d))
    print("test_pilot_b2_recover PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
