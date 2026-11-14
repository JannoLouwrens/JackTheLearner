"""J6(d) DRINK race — the tally, under the rule DECLARED in
docs/research/NEEDS_CLOCK_RACE.md ("J6(d) step 2 — DRINK race DECLARED",
4c316fa, 18:25 UTC 11-14, before any cell). Written before any race cell
exists (11-14 19:40 UTC); no ledger.

Why a separate module: experiments/drink_race_kcpu.py is in its own cells'
CODE_FILES, so a tally living there would stale every finished cell the moment
the tally is touched. This file is on no cell's code path.

Cells: /data/jack-logs/drink_race/<arm>_<spec>_s<seed>.json, written by
drink_race_kcpu. A cell COUNTS only if its `code` equals the current tree's
code_shas(spec) (unknown is not a match). Stale and missing cells both leave
the race incomplete -> verdict None. The old heat_race / water_race cells have
no `code` stamp and predate NE.01's re-registration, so none are reused.

Rule (humid_race's verbatim + clause 4), race arm dB vs base `shipped`,
seeds 0-4, W.1 / W.2 / NE.01:
  dB SHIPS iff
  (1) every W.1/W.2 boolean `*_pass` and every NE.01 `ok_*` seed-mean >= shipped's
      (keys whose values are not all 0/1 excluded — water_clock_race._summ),
  (2) NE.01 pooled max random-death cause share STRICTLY below shipped's,
  (3) W.1 _check passes on every seed shipped's does,
  (4) NE.01 gate 2 `ok_random_deaths` == 1 on >= 4 of 5 seeds.
  Apparatus (VOIDs the arm, not a FAIL): (ii) dB0 replay byte-same — PASSED
  c9dd457 (random_life_drink_probe, recorded, not re-run here); (iii) NE.01
  forager_drinks >= 3 on every seed, else the arm is WORLD-BREAKING.
  dA / dC were declared but are not built (pool literal = world-file refactor,
  Director call) — reported as NOT BUILT, never as lost.
Reported, not gated: swp_k72 against the same clauses, causes, W.2 c_pass,
forager drinks per seed.

Usage: python -m experiments.drink_race_tally [OUTDIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import drink_race_kcpu as K
from experiments import water_clock_race as WR

BASE, RACE, REF = "shipped", ("dB",), ("swp_k72",)
NOT_BUILT = ("dA", "dC")
GATE2 = "ok_random_deaths"
GATE2_MIN_SEEDS = 4
MIN_FORAGER_DRINKS = 3


def load_current(outdir: Path, arm: str, cur: dict | None = None) -> dict:
    cur = cur or {s: K.code_shas(s) for s in K.SPECS}
    cells, stale = [], []
    for p in sorted(Path(outdir).glob(f"{arm}_*.json")):
        rec = json.loads(p.read_text())
        if rec.get("arm") != arm or rec.get("spec") not in cur:
            continue
        (cells if rec.get("code") == cur[rec["spec"]] else stale).append(rec)
    return {"cells": cells, "stale": [f"{c['spec']}_s{c['seed']}" for c in stale]}


def _complete(cells) -> bool:
    have = {(c["spec"], c["seed"]) for c in cells}
    return all((s, d) in have for s in K.SPECS for d in K.SEEDS)


def _gate2_seeds(cells) -> int:
    return sum(1 for c in cells if c["spec"] == "NE.01"
               and float(c["metrics"].get(GATE2, 0)) == 1.0)


def _forager_drinks(cells) -> dict:
    return {c["seed"]: c["metrics"].get("forager_drinks")
            for c in cells if c["spec"] == "NE.01"}


def judge(arm_cells, base_cells) -> list:
    """Clauses the arm loses on; empty list = ships. Both sets complete."""
    r, b = WR._summ(arm_cells), WR._summ(base_cells)
    lost = [k for k, v in b["flags"].items() if r["flags"].get(k, 0.0) < v]
    if not r["max_cause_share"] < b["max_cause_share"]:
        lost.append("max_cause_share not lower")
    if not all(r["W.1_check_by_seed"].get(s) for s, ok in
               b["W.1_check_by_seed"].items() if ok):
        lost.append("W.1 check lost a seed shipped passes")
    if _gate2_seeds(arm_cells) < GATE2_MIN_SEEDS:
        lost.append(f"gate 2 on {_gate2_seeds(arm_cells)}/5 seeds < {GATE2_MIN_SEEDS}")
    return lost


def tally(outdir: Path = K.OUT) -> dict:
    cur = {s: K.code_shas(s) for s in K.SPECS}
    loaded = {a: load_current(outdir, a, cur) for a in (BASE,) + RACE + REF}
    res: dict = {"not_built": list(NOT_BUILT),
                 "apparatus_ii": "PASS c9dd457 (dB0 replays swp_k72 s1 l3 byte-same)"}
    for a, L in loaded.items():
        s = WR._summ(L["cells"])
        res[a] = {"n_current": len(L["cells"]), "stale": L["stale"],
                  "complete": _complete(L["cells"]),
                  "gate2_seeds": _gate2_seeds(L["cells"]),
                  "forager_drinks": _forager_drinks(L["cells"]),
                  "flags": s["flags"], "causes": s.get("causes"),
                  "max_cause_share": s.get("max_cause_share"),
                  "W.2_c_pass_by_seed": s["W.2_c_pass_by_seed"]}
    if not all(res[a]["complete"] for a in (BASE,) + RACE):
        res["verdict"] = None
        return res
    base = loaded[BASE]["cells"]
    for a in REF:                                   # reported, not gated
        if res[a]["complete"]:
            res[a]["would_lose_on"] = judge(loaded[a]["cells"], base)
    wins = {}
    for a in RACE:
        fd = res[a]["forager_drinks"]
        if not all(float(v or 0) >= MIN_FORAGER_DRINKS for v in fd.values()):
            res[a]["void"] = f"WORLD-BREAKING: forager drinks {fd} (< {MIN_FORAGER_DRINKS} on a seed)"
            wins[a] = False
            continue
        res[a]["lost_on"] = judge(loaded[a]["cells"], base)
        wins[a] = not res[a]["lost_on"]
    res["wins"] = wins
    res["verdict"] = next((f"{a} SHIPS" for a in RACE if wins[a]), "none ships")
    return res


if __name__ == "__main__":
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else K.OUT
    print(json.dumps(tally(od), indent=1, default=str))
