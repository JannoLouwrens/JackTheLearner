"""Planted tests for round 1-L's beside readings and keep rule
(`experiments/forge/r1L_beside.py`) on a fake r1L folder. Every expected
value is worked by hand from the frozen rule's text, not from the code.

Lives are 10,000 decisions each, so life i ends at i x 10,000: lives 31-50
end in (300k, 500k], lives 51-100 are the late half, lives 81-100 end in
(800k, 1M], life 101 ends past the rung.
Run: /data/venvs/jackthelearner/bin/python -m tests.test_forge_r1L_beside
"""
import json

from experiments.forge import Refused
from experiments.forge import r1L_beside as B
from experiments.forge import r1L_read as R
from experiments.forge.metric import metric_hash

SPEC = json.loads((R.ROUND / "spec.json").read_text())
SHA, PIN = "s" * 64, "b" * 40
T = SPEC["rungs"]["L"]["decisions"]


def life(fed, capped=True, intake=3.0):
    return {"decisions": 10_000, "floor_meals": 6 if fed else 0, "apples": 0,
            "cause": "capped" if (fed and capped) else "starved",
            "intake": intake if fed else 0.0, "basal_drain": 1.5}


def lives(fed_set):
    return [life(i in fed_set) for i in range(1, 102)]


def late(k):
    """Fed in every early life and in the first k of the 50 late lives."""
    return set(range(1, 51 + k))


def rec(arm, seed, fed_set, **over):
    r = {"arm": arm, "seed": seed, "T": T, "decisions": T + 10_000, "rc": 0,
         "world": {k: SPEC["world"][k] for k in R._WORLD},
         "forge": {"rung": "L", "code_head": PIN, "settings": R.rule_settings(SPEC, arm)},
         "lives": lives(fed_set)}
    r.update(over)
    return r


def folder(plant, over=None):
    """`plant(arm, i)` -> the fed-life set of arm's i-th tranche-1 seed."""
    plan = R.manifest(SPEC, [1])
    cells = [(a, s, i) for a, ss in plan.items() for i, s in enumerate(ss)]
    return {f"L_g1-{j}.json": {"round": "r1L", "rung": "L", "pin": PIN, "spec_sha256": SHA,
                               "cells": {f"{a}_s{s}": rec(a, s, plant(a, i),
                                                          **((over or {}).get((a, s), {})))
                                         for a, s, i in cells[j:j + 4]}}
            for j in range(0, len(cells), 4)}


def go(files, **kw):
    args = dict(spec_sha=SHA, freeze_sha=SHA, code_hash=metric_hash())
    args.update(kw)
    return B.read(SPEC, files, [1], **args)


def plant(k1, k2, base=lambda i: set()):
    return lambda arm, i: {"K1": k1, "K2": k2, "E-food": base}.get(arm, lambda i: set())(i)


# ---- the window reading, by hand ------------------------------------------
def test_window_edges_belong_to_the_life_that_ends_on_them():
    ls = lives({30, 50, 80, 100, 101})
    # life 30 ends AT 300k: outside (300k, 500k]; life 50 ends AT 500k: inside.
    assert B.window_rate(ls, *B.MID) == (1 / 20, 20)
    # life 80 ends AT 800k: outside; life 100 ends AT 1M: inside; 101 is past the rung.
    assert B.window_rate(ls, *B.LAST) == (1 / 20, 20)


def test_a_long_life_that_did_not_eat_is_not_sustained():
    ls = [life(True, intake=1.0) for _ in range(101)]       # capped, intake 1.0 < drain 1.5
    assert B.window_rate(ls, *B.LAST) == (0.0, 20)
    ls = [life(True, capped=False) for _ in range(101)]     # ate, but died before the cap
    assert B.window_rate(ls, *B.LAST) == (0.0, 20)


def test_a_window_with_no_life_reads_zero_and_says_so():
    one = [dict(life(True), decisions=1_000_000)]
    assert B.window_rate(one, *B.MID) == (0.0, 0)
    assert B.window_rate(one, *B.LAST) == (1.0, 1)


# ---- the keep rule, by hand -----------------------------------------------
def test_keep_rule_cases():
    D = "A_DIFFERS"
    k1 = [0.25, 0.75] * 5                                   # mean 0.5, range 0.5
    assert B.keep(D, k1, [1.0] * 10)["keep"] == "K2"        # higher by exactly the range
    assert B.keep(D, k1, [0.875] * 10)["keep"] == "K1"      # higher, but closer than the range
    assert B.keep(D, k1, [0.25] * 10)["keep"] == "K1"       # lower
    assert B.keep(D, [1.0] * 10, [1.0] * 10)["keep"] == "K1"    # exact tie, range 0
    assert B.keep(D, [0.5] * 10, [0.75] * 10)["keep"] == "K2"   # range 0, K2 higher
    for d in ("FUTILE", "NOT_DISTINGUISHED", "NEEDS_LOOK_2"):
        assert B.keep(d, [0.0] * 10, [1.0] * 10)["keep"] is None, d
    assert "not yet" in B.keep("NEEDS_LOOK_2", k1, k1)["why"]
    try:
        B.keep(D, [1.0] * 10, [1.0] * 9)
    except Refused:
        return
    raise AssertionError("unequal seed lists not refused")


# ---- through the reader, on a planted folder ------------------------------
def test_k2_inside_k1s_range_keeps_k1_and_long_life_is_read():
    # K1: late rates 0.8 / 1.0 alternating (40 or 50 of 50) -> mean 0.9, range 0.2.
    # K2: every late life fed -> 1.0; 0.1 ahead, inside 0.2 -> K1.
    out = go(folder(plant(lambda i: late(40 + 10 * (i % 2)), lambda i: late(50))))
    assert out["verdict"].decision == "A_DIFFERS" and out["verdict"].look == 1
    k = out["keep"]
    assert k["keep"] == "K1" and "closer" in k["why"], k
    assert abs(k["mean_K1"] - 0.9) < 1e-12 and k["mean_K2"] == 1.0
    assert abs(k["K1_range"] - 0.2) < 1e-12
    ll = out["long_life"]
    # K1 seed 0: fed through life 90 -> (800k, 1M] has lives 81-90 fed of 20 = 0.5.
    assert (ll["K1"][0]["mid"], ll["K1"][0]["last"]) == (1.0, 0.5)
    assert (ll["K1"][1]["mid"], ll["K1"][1]["last"]) == (1.0, 1.0)
    assert all(r["mid"] == 0.0 and r["last"] == 0.0 for r in ll["E-food"])
    assert len(ll["E-food-lr0"]) == 2 and all(r["mid_lives"] == 20 for r in ll["E-food-lr0"])


def test_k2_clearly_ahead_is_kept():
    out = go(folder(plant(lambda i: late(25), lambda i: late(50))))     # 0.5 flat vs 1.0
    assert out["keep"]["keep"] == "K2" and out["keep"]["K1_range"] == 0.0, out["keep"]


def test_win_then_lose_shows_in_the_long_life_reading():
    # K2 eats through life 60 and never again: late-half rate 10/50, mid 1.0, last 0.0.
    out = go(folder(plant(lambda i: late(50), lambda i: late(10))))
    assert out["rate_M"]["K2"] == [0.2] * 10
    assert all((r["mid"], r["last"]) == (1.0, 0.0) for r in out["long_life"]["K2"])
    assert out["keep"]["keep"] == "K1"


def test_a_crashed_run_counts_zero_against_its_arm():
    out = go(folder(plant(lambda i: late(50), lambda i: late(50)),
                    over={("K2", 1425): {"error": "planted"}}))
    assert out["status"] == {"K2_s1425": "crash"}
    assert abs(out["keep"]["mean_K2"] - 0.9) < 1e-12        # nine at 1.0, the crash at 0
    row = [r for r in out["long_life"]["K2"] if r["seed"] == 1425][0]
    assert row == {"seed": 1425, "mid": 0.0, "mid_lives": 0, "last": 0.0, "last_lives": 0,
                   "status": "crash"}


def test_no_keep_setting_without_a_verdict_that_k1_differs():
    same = lambda i: late(50)                                # noqa: E731
    out = go(folder(plant(same, same, base=same)))           # K1 == baseline
    assert out["verdict"].decision == "NEEDS_LOOK_2" and out["keep"]["keep"] is None
    out = go(folder(plant(lambda i: set(), same)))           # K1 and baseline never eat
    assert out["verdict"].decision == "FUTILE" and out["keep"]["keep"] is None
    assert "none chosen" in out["keep"]["why"]


def test_the_readers_refusals_come_first():
    f = folder(plant(lambda i: late(50), lambda i: late(50)))
    for kw, text in ((dict(spec_sha="x" * 64), "differs from the freeze"),
                     (dict(code_hash="0" * 64), "metric code")):
        try:
            go(f, **kw)
        except Refused as e:
            assert text in str(e), e
        else:
            raise AssertionError(f"not refused ({text})")


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
