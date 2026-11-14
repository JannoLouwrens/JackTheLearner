"""J6(c) gate-2 probe: do the humidity arms kill NE.01's random flailers inside one day?

WHY (water race tally, 11-14 05:04): both sweat arms lost on NE.01 gate 2.
`ok_random_deaths` is 1.0 for shipped, and 0/25 random lives died in swp_w72 or in swp_k72
(`random_death_dec` -1). The arithmetic: with sweat paid, the only thing that killed a
random flailer was THIRST (sweat_paid s0 had 5/5 dehydration deaths, last at decision 2479).
At B_W 1/3600 a flailer draining 3.5x basal needs about 1150 s, and RANDOM_LIFE_CAP is
6000 decisions = 1200 s, so in practice nobody dies. Under the declared rule
(seed-mean of every ok_* >= shipped's), an arm with no acute killer cannot ship. That holds
for every k = 72 arm, whatever its thermal tuning.
The humidity arms (humid_race) are the only candidates with an acute killer, so before
their NE.01 legs (~2 h CPU) are spent, this measures gate 2 alone, through NE.01's OWN
`_random_lives` under each arm's own `_apply`.

Not a race leg and not a ledger row. It prices the next race. No threshold is touched.
APPARATUS (known negative): swp_k72 s0 must reproduce the water race's cell, 0/5 died.

Usage (nice 19, <= 2 workers, detached):
  python -m experiments.random_death_probe run ARM SEED
  python -m experiments.random_death_probe tally
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import humid_race as HR

OUT = Path("/data/jack-logs/random_death_probe")


def run(arm: str, seed: int) -> Path:
    _, mods = HR._apply(arm)
    ne = mods["NE.01"]
    r = ne._random_lives(int(seed))
    rec = {"arm": arm, "seed": int(seed), "deaths": r["deaths"],
           "cap": ne.RANDOM_LIFE_CAP,
           "window": [ne.DEATH_MIN_DEC, ne.DEATH_MAX_DEC]}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{arm}_s{seed}.json"
    p.write_text(json.dumps(rec))
    return p


def tally() -> dict:
    out: dict = {}
    for p in sorted(OUT.glob("*_s*.json")):
        d = json.loads(p.read_text())
        lo, hi = d["window"]
        ds = d["deaths"]
        ok = all(x["died"] and lo <= x["decisions"] <= hi for x in ds)
        out[p.stem] = {"died": sum(x["died"] for x in ds), "n": len(ds),
                       "ok_random_deaths": int(ok),
                       "decisions": [x["decisions"] for x in ds],
                       "causes": [x["cause"] for x in ds]}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        print(run(sys.argv[2], int(sys.argv[3])))
    else:
        print(json.dumps(tally(), indent=1))
