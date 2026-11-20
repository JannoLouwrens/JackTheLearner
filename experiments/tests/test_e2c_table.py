"""J15: `pilot_b2 e2c-table` — E2c's descriptive table (11-20, before the book
is complete; it applies no reading).

Known answers, each with a control that must fail:
- A planted book of two runs: a food run whose capped lives end at 12k and
  30k decisions, then only starved lives; a zero run with none. The table
  prints `[12, 30]` and `none` -- control: a life that ENDS at the cap mark
  but is not `capped` is not listed.
- 2 of 25 registered runs in -> PARTIAL, the 23 others named missing; an
  unregistered run id is reported as ignored, never scored.
- pilot_table's default path (J11's forty runs) is unchanged by the new
  arms/cells arguments: same registered ids as before the edit.
- (11-20 18:30) the critic section: a run that never eats with -actor_loss
  2,000 at 300k reads Q 300k 2000 and a ratio > 4 over SUPPORTED (= 100*d_max
  + 0.1*8ln2/(1-0.995) = 151.7 + 110.9 = 262.6); control: a run eating 1 meal
  per 100 decisions (SUPPORTED 200 + 262.6) with Q 300 reads ratio < 1.
"""
import json
import tempfile
from pathlib import Path

from experiments import pilot_b2 as P


def _life(dec, cause, meals):
    return {"decisions": dec, "seconds": dec / 5.0, "floor_meals": meals,
            "apples": 0, "cause": cause, "intake": 1.0, "basal_drain": 3.0,
            "sustained": cause == "capped", "reward": 0.0}


def _rec(arm, seed, lives):
    return {"arm": arm, "seed": seed, "T": P.PILOT_T, "rc": 0, "lives": lives}


def _book(d: Path):
    food = [_life(4000, "energy", 1), _life(8000, "capped", 30),
            _life(18000, "capped", 40), _life(9000, "energy", 0),
            _life(3000, "capped_lookalike", 0), _life(258000, "unfinished", 0)]
    zero = [_life(5000, "energy", 0), _life(295000, "unfinished", 0)]
    cells = {"E-food_s801": _rec("E-food", 801, food),
             "E-zero-auto_s801": _rec("E-zero-auto", 801, zero),
             "E-food_s999": _rec("E-food", 999, zero)}
    p = d / "e2c_x0.json"
    p.write_text(json.dumps({"cells": cells}))
    return p


def test_planted_book():
    with tempfile.TemporaryDirectory() as t:
        md, missing = P.e2c_table([_book(Path(t))])
    assert "- E-food_s801: [12, 30]" in md, md        # 4k+8k, +18k
    assert "- E-zero-auto_s801: none" in md
    assert "E-food_s999" in md.split("\n")[0] and "NOT registered" in md
    assert "- E-food_s999" not in md                  # never scored
    assert md.startswith("**PARTIAL** -- 2 of 25 registered runs in")
    assert len(missing) == 23 and "E-food-auto_s805" in missing
    return md


def test_pilot_default_unchanged():
    from experiments import pilot_b_report as R
    seeds = P.PILOT_RULE["seeds"]
    reg = [R.rid(a, s) for a, s in P.pilot_cells(seeds)]
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "pilot_p0.json"
        p.write_text(json.dumps({"cells": {}}))
        md, missing = P.pilot_table([p], seeds)
    assert missing == reg and len(reg) == 40
    assert md.count("| 0 of ") == len(P.PILOT_LEARNERS + P.PILOT_FLOORS)


def _qrec(arm, seed, q300, meal_every):
    lives = [_life(9000, "capped", 9000 // meal_every if meal_every else 0)
             for _ in range(300_000 // 9000)]
    lives.append(_life(300_000 - 9000 * len(lives), "unfinished", 0))
    trace = [{"t": t, "actor_loss": -q300 * t / 300_000, "ent_coef": 0.1}
             for t in range(5000, 300_001, 5000)]
    return dict(_rec(arm, seed, lives), trace=trace)


def test_critic_section():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "e2c_q0.json"
        p.write_text(json.dumps({"cells": {
            "E-zero_s801": _qrec("E-zero", 801, 2000.0, 0),
            "E-food_s801": _qrec("E-food", 801, 300.0, 100)}}))
        md, _ = P.e2c_table([p])
    rows = {l.split(" | ")[0][2:]: l for l in md.splitlines()
            if l.startswith("| E-") and l.count("|") == 10}
    z = rows["E-zero_s801"].strip("| ").split(" | ")
    assert z[4] == "2000" and z[7] == "263" and float(z[8]) > 4, z
    f = rows["E-food_s801"].strip("| ").split(" | ")
    assert f[4] == "300" and float(f[8]) < 1, f      # control: supported
    assert f[1] == "50", f                           # Q at 50k = 300*50/300


def main():
    test_critic_section()
    test_planted_book()
    test_pilot_default_unchanged()
    print("test_e2c_table PASS")


if __name__ == "__main__":
    main()
