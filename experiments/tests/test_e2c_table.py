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
- (11-20 19:30) the entropy section: H = -8 + ent_coef_loss / log(alpha). A
  planted fixed-0.1 run whose true H falls 5.36 -> -17 linearly prints 5.0 at
  10k, -17.0 at 300k, H min -17.0 and its first H <= 0 at the
  planted mark (75k); control: a run whose H stays 5.36 prints 5.4 throughout
  and NO crossing. An auto run (alpha 0.3) with the same planted H reads the
  same H (the formula divides by log alpha, not alpha -- dividing by alpha
  would print -7.4, asserted absent). alpha = 1 or no ent_coef_loss -> "–".
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


def _hrec(arm, seed, h_of_t, alpha=0.1, drop=False):
    import math
    trace = []
    for t in range(5000, 300_001, 5000):
        w = {"t": t, "sat": 0.05, "actor_loss": -1.0, "ent_coef": alpha}
        if not drop:
            w["ent_coef_loss"] = math.log(alpha) * (h_of_t(t) - (-8.0)) if alpha != 1 else 0.0
        trace.append(w)
    return dict(_rec(arm, seed, [_life(300_000, "unfinished", 0)]), trace=trace)


def test_entropy_section():
    fall = lambda t: 5.36 - (5.36 + 17.0) * (t - 5000) / 295_000   # <= 0 from 75,716 -> window 80k
    flat = lambda t: 5.36
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "e2c_h0.json"
        p.write_text(json.dumps({"cells": {
            "E-food_s801": _hrec("E-food", 801, fall),
            "E-zero_s801": _hrec("E-zero", 801, flat),
            "E-food-auto_s801": _hrec("E-food-auto", 801, fall, alpha=0.3),
            "E-zero-auto_s801": _hrec("E-zero-auto", 801, fall, alpha=1.0),
            "E-random_point_s801": _hrec("E-random_point", 801, fall, drop=True)}}))
        md, _ = P.e2c_table([p])
    rows = {l.split(" | ")[0][2:]: l.strip("| ").split(" | ") for l in md.splitlines()
            if l.startswith("| E-") and l.count("|") == 13}
    assert len(rows) == 5, rows
    f = rows["E-food_s801"]
    assert f[1] == "5.0" and f[5] == "-17.0" and f[6] == "-17.0" and f[7] == "80", f
    z = rows["E-zero_s801"]                           # control: no sharpening
    assert z[1:7] == ["5.4"] * 6 and z[7] == "–", z
    a = rows["E-food-auto_s801"]
    assert a[1:8] == f[1:8], (a, f)                   # log(alpha), not alpha
    assert "-7.4" not in a
    assert rows["E-zero-auto_s801"][1:8] == ["–"] * 7       # alpha = 1
    assert rows["E-random_point_s801"][1:8] == ["–"] * 7    # nothing logged
    assert P.e2c_entropy({"ent_coef": 0.1, "ent_coef_loss": -30.888}) is not None
    assert abs(P.e2c_entropy({"ent_coef": 0.1, "ent_coef_loss": -30.888}) - 5.415) < 0.01


def main():
    test_critic_section()
    test_entropy_section()
    test_planted_book()
    test_pilot_default_unchanged()
    print("test_e2c_table PASS")


if __name__ == "__main__":
    main()
