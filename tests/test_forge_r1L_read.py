"""Planted tests for round 1-L's reader (ORDERS S20 (2)) on a fake r1L folder.
Expected verdicts are worked from the rule: K1 crossing early in all ten
runs against a baseline that never crosses differs at look 1; identical arms
need look 2; a moved spec, a moved metric, a missing run all refuse.
Run: /data/venvs/jackthelearner/bin/python -m tests.test_forge_r1L_read
"""
import json

from experiments.forge import Refused
from experiments.forge import r1L_read as R
from experiments.forge.metric import metric_hash

SPEC = json.loads((R.ROUND / "spec.json").read_text())
SHA, PIN = "s" * 64, "b" * 40
T = SPEC["rungs"]["L"]["decisions"]


def life(cause, intake=0.0, drain=1.0):
    return {"decisions": 9000, "floor_meals": 6 if intake else 0, "apples": 0,
            "cause": cause, "intake": intake, "basal_drain": drain}


FED, STARVED = life("capped", 3.0, 1.5), life("starved")
N = T // 9000 + 1                                   # 112 lives: past T


def rec(arm, seed, fed):
    lives = [FED if fed else STARVED] * N
    return {"arm": arm, "seed": seed, "T": T, "decisions": T + 8000, "rc": 0,
            "world": {k: SPEC["world"][k] for k in R._WORLD},
            "forge": {"rung": "L", "code_head": PIN, "settings": R.rule_settings(SPEC, arm)},
            "lives": lives}


def folder(fed_arms=("K1",), drop=None):
    plan = R.manifest(SPEC, [1])
    cells = [(a, s) for a, ss in plan.items() for s in ss if (a, s) != drop]
    return {f"L_g1-{j}.json": {"round": "r1L", "rung": "L", "pin": PIN, "spec_sha256": SHA,
                               "cells": {f"{a}_s{s}": rec(a, s, a in fed_arms)
                                         for a, s in cells[j:j + 4]}}
            for j in range(0, len(cells), 4)}


def go(files, **kw):
    args = dict(spec_sha=SHA, freeze_sha=SHA, code_hash=metric_hash())
    args.update(kw)
    return R.read(SPEC, files, [1], **args)


def refused(fn, text):
    try:
        fn()
    except Refused as e:
        assert text in str(e), e
        return
    raise AssertionError(f"not refused ({text})")


def test_planted_winner_stops_at_look_1_with_books_and_rates():
    out = go(folder(("K1", "K2")))
    v = out["verdict"]
    assert v.decision == "A_DIFFERS" and v.look == 1 and v.p1 <= 0.025, v
    assert out["books"] == {"K1": 10, "E-food": 10, "K2": 10, "E-food-lr0": 2}
    assert out["crossed"] == {"K1": 10, "E-food": 0}
    assert out["rate_M"]["K1"] == [1.0] * 10 and out["rate_M"]["E-food"] == [0.0] * 10


def test_null_pair_needs_look_2():
    v = go(folder(("K1", "E-food")))["verdict"]
    assert v.decision == "NEEDS_LOOK_2" and v.look == 1, v


def test_refusals():
    refused(lambda: go(folder(), spec_sha="x" * 64), "differs from the freeze")
    refused(lambda: go(folder(), code_hash="0" * 64), "metric code")
    refused(lambda: go(folder(drop=("E-food", 1425))), "missing")
    f = folder()
    next(iter(f.values()))["spec_sha256"] = "y" * 64
    refused(lambda: go(f), "frozen spec")
    f = folder()
    c = next(iter(next(iter(f.values()))["cells"].values()))
    c["forge"]["code_head"] = "c" * 40
    refused(lambda: go(f), "pin")


def test_freeze_sha_is_the_committed_spec():
    import hashlib
    assert R.frozen_sha() == hashlib.sha256((R.ROUND / "spec.json").read_bytes()).hexdigest()


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
