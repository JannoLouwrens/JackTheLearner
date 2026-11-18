"""J11 item 5's descriptive table (11-18 06:40, before the rule and before any
pilot run): planted pilot kernel files with known answers.

K s402: 6 starving lives, then 2 capped-and-fed lives in a row, then a starving
tail -> crosses W=1 and W=2, not W=3. Control K s403: the same lives with one
starving life BETWEEN the two fed ones -> crosses W=1 only (a broken streak must
not count as W=2). B0 s402 is cut (decisions sum < T) -> flagged, its lives kept.
Every other registered run is absent -> PARTIAL, exit code 3 path."""
import json

from experiments import pilot_b2 as P

SEEDS = (402, 403)


def life(dec, cause, meals, intake, drain, sustained):
    return {"decisions": dec, "cause": cause, "floor_meals": meals, "apples": 0,
            "intake": intake, "basal_drain": drain, "sustained": sustained,
            "seconds": dec / 4.0, "reward": float(meals)}


STARVE = life(4000, "hunger", 0, 0.0, 1.0, False)
FED = life(9000, "capped", 12, 2.0, 1.0, True)


def run(lives, T=P.PILOT_T):
    rest = T - sum(l["decisions"] for l in lives)
    if rest > 0:
        lives = lives + [life(rest, "unfinished", 0, 0.0, 1.0, False)]
    return {"lives": lives, "T": P.PILOT_T, "rc": 0}


def write(tmp_path):
    k402 = run([STARVE] * 6 + [FED, FED] + [STARVE] * 3)
    k403 = run([STARVE] * 6 + [FED, STARVE, FED] + [STARVE] * 3)
    b0cut = {"lives": [STARVE] * 5, "T": P.PILOT_T, "rc": 0, "cut": True}
    f = tmp_path / "pilot_p0.json"
    f.write_text(json.dumps({"cells": {"K_s402": k402, "K_s403": k403, "B0_s402": b0cut}}))
    return f


def row(md, arm):
    return next(l for l in md.splitlines() if l.startswith(f"| {arm} | ") and " of 2 |" in l)


def test_crossings_and_control(tmp_path):
    md, missing = P.pilot_table([write(tmp_path)], SEEDS)
    k = row(md, "K").split(" | ")
    assert k[1] == "2 of 2" and k[2] == "0"
    assert k[3] == "2/1/0"            # s402 crosses W1+W2; control s403 only W1
    assert k[5] == "4"                # 2 + 2 sustained lives
    b0 = row(md, "B0").split(" | ")
    assert b0[1] == "1 of 2" and b0[2] == "1" and b0[3] == "0/0/0"


def test_partial_and_duplicates(tmp_path):
    f = write(tmp_path)
    md, missing = P.pilot_table([f], SEEDS)
    assert md.startswith("**PARTIAL**")
    assert len(missing) == len(P.pilot_cells(SEEDS)) - 3
    assert "K_s402" not in missing
    g = tmp_path / "pilot_p1.json"
    g.write_text(f.read_text())
    try:
        P.pilot_table([f, g], SEEDS)
    except ValueError as e:
        assert "K_s402" in str(e)
    else:
        raise AssertionError("a run in two files must refuse")


def test_no_verdict(tmp_path):
    md, _ = P.pilot_table([write(tmp_path)], SEEDS)
    low = md.lower()
    for word in ("slides", "eats but starves", "pass", "fail", "verdict"):
        assert word not in low, word


def main():
    import tempfile
    from pathlib import Path
    for t in (test_crossings_and_control, test_partial_and_duplicates, test_no_verdict):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"{t.__name__}: PASS")
    print("test_pilot_b2_table PASS")


if __name__ == "__main__":
    main()
