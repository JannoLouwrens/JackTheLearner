"""J24 — the frozen read of `docs/research/prereg/eye_bins16.md` (frozen 03e6285d).

Score: late meals a life (lives STARTING at decision >= 250,000), `pilot_b2.j17_late_meals`.
EATS = sighted above BOTH twins on 3/3 seeds AND sighted mean >= 2 x the higher twin mean.
Extension = sighted above both on >= 2/3 but not EATS -> carry to 1.5M.
Else NOT SHOWN AT 500k. (The prereg's `j17_onset` beside-reading is the critic-ruler
onset of `experiments/j17_onset.py`; it decides nothing and is not printed here.) A cell that is not whole (decisions != 500k, wrong pin) refuses.

    python -m experiments.j24_read            # prints the nine numbers and the branch
    python -m experiments.j24_read --variant k1 --pin <freeze sha>
        # J24 follow-up (prereg/eye_bins16_k1.md): the SAME rule on the K1 cells'
        # raw files under /data/jack-logs/j24k1; refuses unless the prereg at
        # <pin> is FROZEN and every raw file carries that pin.
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
TAG = "j24"
# J24 follow-up: the K1 cells (J24K1_ARMS, resets every 50k) read by J24's rule
# unchanged. Its pin is the K1 prereg's freeze commit, given on the command line.
K1 = {"arms": ("K1-bins16", "K1-bins16-blind", "K1-bins16-shuffled"),
      "raw": Path("/data/jack-logs/j24k1/chains"), "tag": "j24k1",
      "prereg": "docs/research/prereg/eye_bins16_k1.md"}


def set_k1(pin: str) -> None:
    """Point the reader at the K1 variant. Refuses unless the K1 prereg at `pin`
    is FROZEN (j24_launch's own guard) - a draft is never read."""
    global PIN, ARMS, RAW, TAG
    import subprocess
    from experiments.j24_launch import REPO, frozen
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify",
                           f"{pin}^{{commit}}"], capture_output=True, text=True)
    if full.returncode != 0:
        raise ValueError(f"j24_read k1: {pin!r} is not a commit")
    full = full.stdout.strip()
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{full}:{K1['prereg']}"],
                          capture_output=True)
    if text.returncode != 0 or not frozen(text.stdout):
        raise ValueError(f"j24_read k1: {K1['prereg']} is not FROZEN at {full[:8]}")
    PIN, ARMS, RAW, TAG = full, K1["arms"], K1["raw"], K1["tag"]


def load(seed: int, root=None) -> dict:
    root = RAW if root is None else root
    d = json.loads((root / f"{TAG}_s{seed}.json").read_text())
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
    argv = list(argv)
    if "--variant" in argv:
        v = argv[argv.index("--variant") + 1]
        if v == "k1":
            if "--pin" not in argv:
                raise ValueError("j24_read --variant k1 needs --pin <K1 prereg freeze sha>")
            set_k1(argv[argv.index("--pin") + 1])
        elif v != "j24":
            raise ValueError(f"j24_read: no variant {v!r} (have j24, k1)")
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
