"""J24 — the frozen read of `docs/research/prereg/eye_bins16.md` (frozen 03e6285d).

Score: late meals a life (lives STARTING at decision >= 250,000), `pilot_b2.j17_late_meals`.
EATS = sighted above BOTH twins on 3/3 seeds AND sighted mean >= 2 x the higher twin mean.
Extension = sighted above both on >= 2/3 but not EATS -> carry to 1.5M.
Else NOT SHOWN AT 500k. (The prereg's `j17_onset` beside-reading is the critic-ruler
onset of `experiments/j17_onset.py`; it decides nothing and is not printed here.) A cell that is not whole (decisions != 500k, wrong pin) refuses.

    python -m experiments.j24_read            # prints the nine numbers and the branch
"""
import json
import sys
from pathlib import Path

from experiments import pilot_b2

PIN = "03e6285d7302095fa433c9692cfe085c6c28edbc"
T = 500_000
LATE_FROM = 250_000
SEEDS = (941, 942, 943)
ARMS = ("K-bins16", "K-bins16-blind", "K-bins16-shuffled")
RAW = Path("/data/jack-logs/j24/chains")


def load(seed: int, root: Path = RAW) -> dict:
    d = json.loads((root / f"j24_s{seed}.json").read_text())
    if d.get("pin") != PIN:
        raise ValueError(f"s{seed}: pin {d.get('pin')} is not the freeze {PIN}")
    out = {}
    for arm in ARMS:
        rec = d["cells"][f"{arm}_s{seed}"]
        if rec["decisions"] != T or sum(l["decisions"] for l in rec["lives"]) != T:
            raise ValueError(f"{arm}_s{seed}: not whole ({rec['decisions']} decisions)")
        out[arm] = rec
    return out


def verdict(scores: dict) -> dict:
    """scores[arm][seed] -> late meals a life. THE RULE, as frozen."""
    s, b, h = (scores[a] for a in ARMS)
    above = [sd for sd in SEEDS if s[sd] > b[sd] and s[sd] > h[sd]]
    mean = {a: sum(scores[a].values()) / len(SEEDS) for a in ARMS}
    twin_hi = max(mean[ARMS[1]], mean[ARMS[2]])
    eats = len(above) == 3 and mean[ARMS[0]] >= 2 * twin_hi
    branch = ("EATS" if eats else "EXTEND_1.5M" if len(above) >= 2
              else "NOT SHOWN AT 500k")
    return {"above_both": above, "means": mean, "twin_hi": twin_hi, "branch": branch}


def main(argv=()):
    scores = {a: {} for a in ARMS}
    for sd in SEEDS:
        cells = load(sd)
        for a in ARMS:
            scores[a][sd] = pilot_b2.j17_late_meals(cells[a], LATE_FROM)
    v = verdict(scores)
    out = {"pin": PIN, "late_from": LATE_FROM, "scores": scores, **v}
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
