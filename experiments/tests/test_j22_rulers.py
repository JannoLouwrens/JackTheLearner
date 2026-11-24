"""J22 (0): the J17 row's Q ÷ SUPPORTED ruler, re-derived with the run's own T,
must reproduce the J17 addendum's printed table at T = 300k (WORLD_CARD.md)."""
import json
from pathlib import Path

from experiments.j17_onset import DATA, hunt_rulers, q_over_supported

PRINTED = {"K-rays16": [1.8, 1.2, 1.8, 1.9, 2.2], "K-row128": [1.7, 2.0, 1.7, 1.4, 1.6],
           "K-row128-blind": [0.4, 0.6, 0.4, 0.8, 0.6]}


def test_reproduces_j17_table_at_300k():
    got = {}
    for f in sorted(DATA.glob("j17_r*.json")):
        for k, c in json.loads(f.read_text())["cells"].items():
            got.setdefault(k.rsplit("_s", 1)[0], {})[c["seed"]] = q_over_supported(c, 300_000)["ratio"]
    for arm, want in PRINTED.items():
        assert [round(got[arm][s], 1) for s in sorted(got[arm])] == want, arm


def test_hunt_rulers_reads_both_arms():
    r = hunt_rulers(DATA / "j17_hunt_h0_gpu.json")
    assert round(r["H-simba-row128"]["climb"], 3) == 2.360
    assert round(r["H-simba-row128"]["ratio"], 2) == 0.17


def main():
    test_reproduces_j17_table_at_300k()
    test_hunt_rulers_reads_both_arms()
    print("test_j22_rulers PASS")


if __name__ == "__main__":
    main()
