"""Next-up 2 (11-20): world_card's per-food eat count per life.

    python -m experiments.tests.test_world_card_by_food

(a) PLANTED, known answer: foods A, B, C; A eaten at decisions 10, 400, 800
    (regrowth-spaced, each > BOUT_GAP apart), B at 50, 52 (one bout, two
    eats), C never. -> by_food {A:3, B:2, C:0}, max 3, eats_all 5, bouts 4.
(b) CONTROL that shows the old fields could not read it: a life eating ONE
    food 3 times and a life eating THREE foods once each give the same
    (floor_eats_all, floor_bouts) = (3, 3) but different max_eats_one_food
    (3 vs 1).
(c) pre-eat counts are subtracted: a food with ate_total 7 at life start and
    9 at the end counts 2; a non-floor key (apple) is never counted.
(d) REAL WORLD, old fields unchanged: run_seed("d8_d00", 153, lives=1,
    cap=N) from HEAD's pre-edit file vs this file -> every pre-existing key
    identical, by_food sums to floor_eats_all.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.world_card import FoodTally, BOUT_GAP  # noqa: E402


def drive(foods, start, events, n_end=1000):
    """events: {decision: food} ; ate_total cumulative from `start`."""
    tot = dict(start)
    t = FoodTally(foods, tot)
    for n in range(1, n_end + 1):
        if n in events:
            for k in events[n]:
                tot[k] = tot.get(k, 0) + 1
        t.step(n, dict(tot))
    return t.row()


def test_planted():
    assert 50 + 2 - 50 <= BOUT_GAP < 390
    r = drive(("A", "B", "C"), {}, {10: "A", 400: "A", 800: "A", 50: "B", 52: "B"})
    assert r["floor_eats_by_food"] == {"A": 3, "B": 2, "C": 0}, r
    assert r["max_eats_one_food"] == 3 and r["floor_eats_all"] == 5 and r["floor_bouts"] == 4, r


def test_old_fields_blind():
    one = drive(("A", "B", "C"), {}, {10: "A", 400: "A", 800: "A"})
    three = drive(("A", "B", "C"), {}, {10: "A", 400: "B", 800: "C"})
    old = lambda r: (r["floor_eats_all"], r["floor_bouts"])
    assert old(one) == old(three) == (3, 3), (one, three)
    assert one["max_eats_one_food"] == 3 and three["max_eats_one_food"] == 1


def test_offsets_and_nonfloor():
    r = drive(("A",), {"A": 7, "apple": 4}, {100: "AA", 300: ["apple"]})
    assert r["floor_eats_by_food"] == {"A": 2} and r["floor_eats_all"] == 2, r


def test_real_world_old_fields_unchanged(n=1500):
    src = subprocess.run(["git", "show", "HEAD:experiments/world_card.py"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    if "class FoodTally" in src:
        print("  (d) HEAD already has FoodTally; comparing against its parent")
        src = subprocess.run(["git", "log", "-1", "--format=%H", "-S", "class FoodTally", "--",
                              "experiments/world_card.py"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        src = subprocess.run(["git", "show", f"{src}^:experiments/world_card.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    p = Path("/tmp/_wc_old_pre_foodtally.py"); p.write_text(src)
    spec = importlib.util.spec_from_file_location("_wc_old", p)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    from experiments import world_card as new
    a = old.run_seed("d8_d00", 153, lives=1, cap=n)[0]
    b = new.run_seed("d8_d00", 153, lives=1, cap=n)[0]
    for k in a:
        assert a[k] == b[k], (k, a[k], b[k])
    assert sum(b["floor_eats_by_food"].values()) == b["floor_eats_all"], b
    print(f"  (d) {n} decisions d8_d00 seed 153: old keys identical; eats_all "
          f"{b['floor_eats_all']} bouts {b['floor_bouts']} by_food {b['floor_eats_by_food']}")


if __name__ == "__main__":
    for f in (test_planted, test_old_fields_blind, test_offsets_and_nonfloor,
              test_real_world_old_fields_unchanged):
        f(); print("PASS", f.__name__)
