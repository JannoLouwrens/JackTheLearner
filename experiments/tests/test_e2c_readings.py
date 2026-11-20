"""J15 11-20 19:45: `pilot_b2 e2c-readings` — E2c's reading transcribed before
the book is complete (18/25 home). Known answers, each branch planted:
- auto food sustains 4/5, fixed food 0/5 -> "WALL"; and no other branch.
- both food arms 1/5 -> "BOTH FOOD ARMS MISS".
- fixed food 5/5, auto 5/5, zero arms 0 -> NO BRANCH (the INBOX 17:15 hole);
  pass clause PASS.
- E-zero-auto eats -> "E-zero-auto EATS"; control: it fires for no other arm.
- the late-meals clause alone (sus 0, late 4/5) counts as eating; late 3/5 not.
- SEAM (s1): auto food sustains exactly 3/5 -> J11 reading eats, STRICT not,
  both branches printed.
- 2 crashed runs in one arm -> NO READING, no branch.
- E-food sustains 3 but E-zero 3 -> pass clause "VOID — control sustains".
- a partial book raises (no reading on a partial book).
"""
import json
import tempfile
from pathlib import Path

from experiments import pilot_b2 as P

ARMS = ("E-food", "E-zero", "E-random_point", "E-food-auto", "E-zero-auto")


def _c(**over):
    c = {a: {"sus": 0, "late": 0, "crash": 0, "short": 0} for a in ARMS}
    for a, d in over.items():
        c[a.replace("_", "-").replace("random-point", "random_point")].update(d)
    return c


def test_wall():
    r = P.e2c_branch(_c(E_food_auto={"sus": 4}))
    assert r["verdict"] == "READ" and len(r["branch"]) == 1, r
    assert r["branch"][0].startswith("THE FIXED 0.1 WAS THE WALL"), r
    assert r["pass_clause"] == "MISS" and "seam" not in r


def test_both_miss():
    r = P.e2c_branch(_c(E_food={"sus": 1}, E_food_auto={"sus": 1}))
    assert r["branch"] == [b for b in r["branch"] if b.startswith("BOTH FOOD ARMS MISS")] and r["branch"], r


def test_no_branch_hole():
    r = P.e2c_branch(_c(E_food={"sus": 5}, E_food_auto={"sus": 5}))
    assert r["branch"][0].startswith("NO BRANCH") and len(r["branch"]) == 1, r
    assert r["pass_clause"] == "PASS"


def test_zero_eats():
    r = P.e2c_branch(_c(E_food={"sus": 5}, E_food_auto={"sus": 5}, E_zero_auto={"late": 4}))
    assert r["branch"] == ["E-zero-auto EATS — the handed sense carries nothing; "
                           "E-food-auto's eating not credited to it"], r


def test_late_clause():
    assert P.e2c_branch(_c(E_food_auto={"late": 4}))["eats"]["E-food-auto"] is True
    assert P.e2c_branch(_c(E_food_auto={"late": 3}))["eats"]["E-food-auto"] is False


def test_seam():
    r = P.e2c_branch(_c(E_food_auto={"sus": 3}))
    assert r["branch"][0].startswith("THE FIXED 0.1 WAS THE WALL"), r
    assert r["seam"]["strict_branch"][0].startswith("BOTH FOOD ARMS MISS"), r


def test_void_and_control():
    r = P.e2c_branch(_c(E_zero={"crash": 2}, E_food_auto={"sus": 5}))
    assert r["verdict"].startswith("NO READING") and "branch" not in r, r
    r = P.e2c_branch(_c(E_food={"sus": 3}, E_zero={"sus": 3}))
    assert r["pass_clause"] == "VOID — control sustains", r


def test_partial_raises():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "e2c_x0.json"
        p.write_text(json.dumps({"cells": {}}))
        try:
            P.e2c_readings([p])
        except ValueError as e:
            assert "25 of 25 missing" in str(e)
        else:
            raise AssertionError("partial book read")


def main():
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("PASS", k)


if __name__ == "__main__":
    main()
