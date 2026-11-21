"""j17_onset.py — WHEN does B's critic run away on eye content? (J17, read-only)

ORDERS J17 NEXT names a cause hunt "at box-minutes on 1 seed" for the critic
that runs away when the eye carries content (WORLD_CARD J17 + addendum
`c905410a`). Before anyone sizes that hunt, this reads the 20 harvested J17
runs (`docs/research/world_card_data/j17/j17_r*.json`, Kaggle CPU, mujoco
3.17.0) and asks, per seed, from which decision the sighted run's critic
signature stays above its blind twin's FOR GOOD. No run, no verdict, no
ledger; DESCRIPTIVE, the 1.5x margin and the 3-window median are this file's
choices (stated, not registered).

  signature  Q (= -actor_loss, min twin critic), critic_loss (median of the
             last 3 five-k windows: one minibatch per window is spiky),
             ent_coef (same median).
  onset(k)   first window t such that sighted(u) > MARGIN x blind(u) for every
             window u >= t, same seed. None = never for good.
  control    the same rule on blind(seed) vs blind(next seed): both arms carry
             no content, so an onset here means the rule fires on seed noise.
             It must read None (or late) on most seeds.

Usage: python -m experiments.j17_onset
"""
from __future__ import annotations

import json
import statistics as S
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "docs/research/world_card_data/j17"
MARGIN = 1.5
KEYS = ("Q", "critic_loss", "ent_coef")
BLIND = "K-row128-blind"


def load(data: Path = DATA) -> dict:
    cells = {}
    for f in sorted(data.glob("j17_r*.json")):
        for k, c in json.loads(f.read_text())["cells"].items():
            cells[(k.rsplit("_s", 1)[0], c["seed"])] = {
                w["t"]: w for w in c["trace"] if "critic_loss" in w}
    return cells


def value(tr: dict, ts: list, t: int, key: str) -> float:
    if key == "Q":
        return -tr[t]["actor_loss"]
    i = ts.index(t)
    return S.median(tr[x][key] for x in ts[max(0, i - 2):i + 1])


def onset(a: dict, b: dict, key: str, margin: float = MARGIN):
    ts = sorted(set(a) & set(b))
    for i, t in enumerate(ts):
        if all(value(a, ts, u, key) > margin * value(b, ts, u, key) for u in ts[i:]):
            return t
    return None


def table(cells: dict) -> dict:
    seeds = sorted({s for _, s in cells})
    out = {}
    for arm in ("K-row128", "K-rays16"):
        for key in KEYS:
            out[(arm, key)] = [onset(cells[(arm, s)], cells[(BLIND, s)], key) for s in seeds]
    for key in KEYS:   # control: blind vs a different seed's blind
        out[("CONTROL blind(s+1) vs blind(s)", key)] = [
            onset(cells[(BLIND, seeds[(i + 1) % len(seeds)])], cells[(BLIND, s)], key)
            for i, s in enumerate(seeds)]
    return {"seeds": seeds, "rows": out}


def main(argv) -> int:
    r = table(load())
    print(f"onset (k decisions) from which arm > {MARGIN}x blind twin for good; seeds {r['seeds']}")
    for (arm, key), v in r["rows"].items():
        ks = ["never" if t is None else f"{t // 1000}k" for t in v]
        print(f"  {arm:32s} {key:11s} {' '.join(f'{k:>6s}' for k in ks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
