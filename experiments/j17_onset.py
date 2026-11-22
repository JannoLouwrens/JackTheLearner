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

Usage: python -m experiments.j17_onset [box-twin | hunt [path]]
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


# ---- box twin (11-21 22:30; reading RE-WRITTEN 22:40, still BEFORE either
# box cell is home) ----------------------------------------------------------
# The cause hunt is to run "at box-minutes". Box `/data/venvs/jack-std` = the
# kernel pins (mujoco 3.17.0) but another CPU. Seed 923, K-row128 + its blind
# twin, 30k decisions, PILOT_RULE's world, are run on the box.
# The first draft read "box row128 > 1.5x box blind at 25k and 30k". STRUCK
# before any result: on the 20 Kaggle runs (`noise()` below) the same-seed
# row128/blind ratio at 30k is 0.65-4.59 while blind-vs-blind across seeds
# reaches 7.34 (9/20 pairs > 1.5x) -- a 30k read is inside seed noise. The
# absolute critic loss first separates sighted from blind 5/5 vs 5/5 at 100k.
# So this twin asks ONLY the apparatus question:
#   SAME TRAJECTORY  every 5k window of BOTH box cells equals Kaggle's
#                    s923 window key for key (same pins, same seed) -> a box
#                    hunt may sit beside the Kaggle traces as its controls.
#   ANOTHER DRAW     any window differs -> the box is a fresh seed draw: a box
#                    hunt needs its own blind twin AND must read at >= 100k.
#   VOID             a cell missing, rc != 0, or a trace short of 30k.
BOX = Path("/data/jack-logs/j17_box_twin")
TWIN_SEED, TWIN_T = 923, 30_000


def noise(cells: dict, t: int, key: str = "critic_loss") -> dict:
    """Same-seed sighted/blind ratio vs cross-seed blind/blind ratio at t,
    and whether the absolute values separate (min sighted > max blind)."""
    seeds = sorted({s for _, s in cells})
    v = lambda c: value(c, sorted(c), t, key)
    row = [v(cells[("K-row128", s)]) for s in seeds]
    bl = [v(cells[(BLIND, s)]) for s in seeds]
    cross = [x / y for i, x in enumerate(bl) for j, y in enumerate(bl) if i != j]
    return {"t": t, "same_seed_ratio": [round(r / b, 2) for r, b in zip(row, bl)],
            "cross_blind_max": round(max(cross), 2),
            "cross_blind_over_margin": f"{sum(c > MARGIN for c in cross)}/{len(cross)}",
            "separates": min(row) > max(bl)}


def _box_cell(arm: str, box: Path = BOX):
    f = box / f"pilot_{arm}_s{TWIN_SEED}.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    if rec.get("error") or rec.get("rc", 0) != 0 or not rec.get("trace"):
        return None
    return {w["t"]: w for w in rec["trace"] if "critic_loss" in w}


def box_twin(box: Path = BOX) -> dict:
    kg = load()
    out = {"noise": [noise(kg, t) for t in (30_000, 60_000, 100_000, 140_000)]}
    pairs = []
    for arm in ("K-row128", BLIND):
        b, k = _box_cell(arm, box), kg[(arm, TWIN_SEED)]
        if b is None or max(b, default=0) < TWIN_T:
            out["verdict"] = f"VOID ({arm}: missing, errored or short of 30k)"
            return out
        ts = [t for t in sorted(b) if t <= TWIN_T]
        eq = [t for t in ts if t in k and b[t] == k[t]]
        pairs.append((arm, f"{len(eq)}/{len(ts)} windows equal",
                      [(t, round(b[t]["critic_loss"], 2), round(k[t]["critic_loss"], 2)) for t in ts]))
    out["beside"] = pairs
    out["verdict"] = ("SAME TRAJECTORY" if all(p[1].split("/")[0] == p[1].split("/")[1].split()[0]
                                               for p in pairs) else "ANOTHER DRAW")
    return out

# ---- SimBa hunt reader (11-22 02:45; written BEFORE `jack-cpu-1795311660`
# is home, from the 20 MLP J17 runs only) -------------------------------------
# One seed, so the read must use a signature that separates sighted from
# blind on EVERY MLP run at 100k. critic_loss climb 50k->100k does not
# (sighted 1.83-6.20, blind 0.65-2.37: overlap). Q climb does:
#   Q(100k)/Q(50k)  MLP K-row128 1.32-1.57 (5/5), K-rays16 1.34-2.03 (5/5),
#                   MLP blind 0.92-1.20 (5/5).   (median-free: Q = -actor_loss
#                   at the window, as value() reads it.)
# Reading (SimBa sighted = S, its own blind twin = T, both s923, 100k):
#   VOID          a cell missing, errored, or its trace short of 100k.
#   NO CONTROL    T's Q climb >= GAP_HI: the twin itself climbs like a sighted
#                 run, so this seed cannot tell sight from SimBa.
#   STILL RUNS    S >= GAP_HI (inside every MLP sighted run) -> SimBa is not
#   AWAY          the cure; next §24 arm.
#   HELD          S <= GAP_LO (inside every MLP blind run) -> SimBa holds the
#                 critic on this seed; then a 5-seed row by rule.
#   UNREAD        GAP_LO < S < GAP_HI: between the groups; one seed says nothing.
# Beside, not read: critic_loss at 100k S/T (MLP s923: 56.3/2.6 = 21.7x).
# The 1.20/1.32 cut points are the MLP groups' own edges; NOT a registered
# threshold, a pilot's reading.
HUNT = Path("/data/jack-logs/pilot_b2/j17_hunt_h0.json")   # = pilot_b2.OUT
HUNT_ARMS = ("H-simba-row128", "H-simba-row128-blind")
GAP_LO, GAP_HI = 1.20, 1.32


def q_climb(tr: dict, t0: int = 50_000, t1: int = 100_000) -> float:
    ts = sorted(tr)
    return value(tr, ts, t1, "Q") / value(tr, ts, t0, "Q")


def hunt(path: Path = HUNT) -> dict:
    if not path.exists():
        return {"verdict": f"VOID ({path} not home)"}
    cells = {}
    for k, c in json.loads(path.read_text())["cells"].items():
        if c.get("error") or c.get("rc", 0) != 0 or not c.get("trace"):
            continue
        cells[k.rsplit("_s", 1)[0]] = {w["t"]: w for w in c["trace"] if "critic_loss" in w}
    for a in HUNT_ARMS:
        if a not in cells or max(cells[a], default=0) < 100_000 or 50_000 not in cells[a]:
            return {"verdict": f"VOID ({a}: missing, errored or short of 100k)"}
    s, t = (q_climb(cells[a]) for a in HUNT_ARMS)
    v = lambda a: value(cells[a], sorted(cells[a]), 100_000, "critic_loss")
    out = {"S_q_climb": round(s, 3), "T_q_climb": round(t, 3),
           "beside_critic_loss_100k": (round(v(HUNT_ARMS[0]), 2), round(v(HUNT_ARMS[1]), 2))}
    out["verdict"] = ("NO CONTROL" if t >= GAP_HI else "STILL RUNS AWAY" if s >= GAP_HI
                      else "HELD" if s <= GAP_LO else "UNREAD")
    return out


def mlp_q_climbs(cells: dict | None = None) -> dict:
    """The MLP groups the cut points come from (re-derivable, tested)."""
    cells = cells or load()
    return {arm: [round(q_climb(cells[(arm, s)]), 3) for s in sorted({s for _, s in cells})]
            for arm in ("K-row128", "K-rays16", BLIND)}


def main(argv) -> int:
    if argv[1:2] == ["hunt"]:
        print("MLP Q climb 50k->100k:", mlp_q_climbs())
        for k, v in hunt(Path(argv[2]) if len(argv) > 2 else HUNT).items():
            print(f"{k}: {v}")
        return 0
    if argv[1:2] == ["box-twin"]:
        for k, v in box_twin().items():
            print(f"{k}: {v}")
        return 0
    r = table(load())
    print(f"onset (k decisions) from which arm > {MARGIN}x blind twin for good; seeds {r['seeds']}")
    for (arm, key), v in r["rows"].items():
        ks = ["never" if t is None else f"{t // 1000}k" for t in v]
        print(f"  {arm:32s} {key:11s} {' '.join(f'{k:>6s}' for k in ks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
