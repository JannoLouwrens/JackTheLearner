"""J6(c) next arm — the water clock on the sweating body (sweat_paid + B_W toward k = 72).

WHY (docs/research/NEEDS_CLOCK_RACE.md, heat race 11-14): paid sweat moved W0's
binding fault off heat — random flailers now die of THIRST at a clock-set
~1450 decisions — and the water clock still runs ~6.3x past the declared
k = 72 (ruling 1). This race moves the water clock on the body that sweats.
No world file is edited until an arm WINS.

Arms (patched in-process on `experiments.needs`, on top of heat_race's
`sweat_paid` patch — same G_SW, E_SW_MAX, C_SW 2.82):
  shipped     the law as shipped (the null). Cells REUSED from the heat race
              (/data/jack-logs/heat_race/shipped_*), byte-same code path.
  swp_w72     sweat_paid + B_W 1/3600 (thirst at k = 72); hunger shipped.
  swp_k72     sweat_paid + the full J6(b) k72 clock (B_E 1/25200, B_W 1/3600,
              TAU_WAKE 910, TAU_SLEEP 210).
W.2's harness horizons are scaled to each arm's own constants exactly as
needs_clock_race._apply does (its APPARATUS NOTE; bars untouched).

ARITHMETIC (no run): sweat_paid flailers run T* ~37.9 C, so the water drain is
~1 + 2.82 x 0.9 = 3.5x basal. Shipped-clock thirst deadline 450/3.5 + 120 =
~250 s (observed ~1450 dec = 290 s). At B_W 1/3600: 3600/3.5 + 120 = ~1150 s =
~5750 decisions — against NE.01's RANDOM_LIFE_CAP 6000 (one 1200 s day). Gate 2
is a knife edge for swp_w72, which is why this is run, not argued. In swp_k72
hunger cannot kill inside a day, so every random death is thirst: clause (2)
below is expected to fail it; it is run as the full-clock reference.
In swp_w72 hunger (1800 s basal, faster when active) can compete with thirst —
the only arm that might break gate 5's one-clock share.

DECLARED RULE (written 2026-11-14 ~03:35 UTC, before any cell of this race):
  heat_race's rule verbatim, with arms (swp_w72, swp_k72) vs shipped, seeds 0-4,
  W.1 / W.2 / NE.01 through each spec's own _experiment (+_control/_check, W.1):
  X SHIPS iff (1) every W.1/W.2 boolean `*_pass` and every NE.01 `ok_*`
  seed-mean >= shipped's, (2) NE.01 pooled max random-death cause share
  STRICTLY below shipped's, (3) W.1 _check passes on every seed shipped's does.
  If both win: swp_w72 (the smaller edit). ADDITIONALLY reported, not gated:
  W.2 c_pass per seed (ruling 1's target), random-life decisions, causes.
  APPARATUS (known positive): sweat_paid NE.01 s0 and W.2 s0 re-run here must
  equal the heat race's sweat_paid cells exactly, else VOID.
  TALLY FIX (heat race's apparatus note): `*_pass` keys whose values are not
  0/1 (W.2:c_widest_tol_that_would_pass, a tolerance) are excluded from (1).

Usage (nice 19, <= 2 workers, detached via water_clock_race.sh):
  python -m experiments.water_clock_race check
  python -m experiments.water_clock_race run ARM SPEC SEED
  python -m experiments.water_clock_race tally
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import heat_race as H

OUT = Path("/data/jack-logs/water_race")
HEAT = H.OUT
SEEDS = H.SEEDS
SPECS = H.SPECS
ARMS = ("shipped", "swp_w72", "swp_k72")
CLOCKS = {"swp_w72": {"B_W": 1 / 3600.0},
          "swp_k72": {"B_E": 1 / 25200.0, "B_W": 1 / 3600.0,
                      "TAU_WAKE": 910.0, "TAU_SLEEP": 210.0}}


def _apply(arm: str):
    base = "shipped" if arm == "shipped" else "sweat_paid"
    N, mods = H._apply(base)
    if arm in CLOCKS:
        for k, v in CLOCKS[arm].items():
            setattr(N, k, v)
        w2 = mods["W.2"]
        deadline = max(1.0 / N.B_W + N.DEATH_W_S, 1.0 / N.B_E + N.DEATH_E_S)
        w2.DEPRIVED_HORIZON_S = max(3000.0, 1.5 * deadline)
        w2.WAKE_FIT_S = max(3500.0, 5.0 * N.TAU_WAKE)
        w2.SLEEP_FIT_S = max(1200.0, 7.5 * N.TAU_SLEEP)
    return N, mods


def check() -> dict:
    out = {}
    for arm in ARMS[1:] + ("sweat_paid",):
        N, mods = _apply(arm)
        w2 = mods["W.2"]
        out[arm] = {"B_E": N.B_E, "B_W": N.B_W, "C_SW": N.C_SW,
                    "TAU_WAKE": N.TAU_WAKE, "TAU_SLEEP": N.TAU_SLEEP,
                    "W2_DEPRIVED_HORIZON_S": w2.DEPRIVED_HORIZON_S,
                    "Tstar_open_day_M300": H._t_star(N, 300.0, 0.0, N.T_DAY),
                    "thirst_deadline_at_3.5x_s": 1 / N.B_W / 3.5 + N.DEATH_W_S}
    H._apply("shipped")
    return out


def run_cell(arm: str, spec: str, seed: int) -> Path:
    _, mods = _apply(arm)
    mod = mods[spec]
    m = mod._experiment(int(seed))
    rec = {"arm": arm, "spec": spec, "seed": int(seed), "metrics": m}
    if spec == "W.1":
        c = mod._control(int(seed))
        rec["control"] = c
        rec["check"] = bool(mod._check(m, c))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{arm}_{spec}_s{seed}.json"
    p.write_text(json.dumps(rec, default=H._jsonable))
    return p


def _load(d: Path, arm: str):
    return [json.loads(p.read_text()) for p in sorted(d.glob(f"{arm}_*.json"))]


def _is_flag(v) -> bool:
    try:
        return float(v) in (0.0, 1.0)
    except (TypeError, ValueError):
        return False


def _apparatus() -> dict:
    bad, n = [], 0
    for spec in ("NE.01", "W.2"):
        a, b = OUT / f"sweat_paid_{spec}_s0.json", HEAT / f"sweat_paid_{spec}_s0.json"
        if not (a.exists() and b.exists()):
            continue
        n += 1
        ma, mb = json.loads(a.read_text())["metrics"], json.loads(b.read_text())["metrics"]
        if ma != mb:
            bad.append(spec + ": " + ",".join(sorted(k for k in set(ma) | set(mb)
                                                    if ma.get(k) != mb.get(k))[:6]))
    return {"compared": n, "mismatched": bad, "ok": n == 2 and not bad}


def _summ(cells) -> dict:
    by = {s: [c for c in cells if c["spec"] == s] for s in SPECS}
    r = {f"n_{s}": len(v) for s, v in by.items()}
    flags = {}
    for s, v in by.items():
        if not v:
            continue
        keys = sorted({k for c in v for k, x in c["metrics"].items()
                       if (k.startswith("ok_") if s == "NE.01" else k.endswith("_pass"))
                       and all(_is_flag(cc["metrics"].get(k, 0)) for cc in v)})
        for k in keys:
            flags[f"{s}:{k}"] = sum(float(c["metrics"].get(k, 0)) for c in v) / len(v)
    r["flags"] = flags
    r["W.1_check_by_seed"] = {c["seed"]: c.get("check") for c in by["W.1"]}
    r["W.2_c_pass_by_seed"] = {c["seed"]: c["metrics"].get("c_pass") for c in by["W.2"]}
    ne = [c["metrics"] for c in by["NE.01"]]
    if ne:
        causes = [k for k in ne[0] if k.startswith("deaths_") and "microsleep" not in k]
        tot = {k[7:]: sum(float(m[k]) for m in ne) for k in causes}
        s = sum(tot.values())
        r["causes"] = tot
        r["max_cause_share"] = max(tot.values()) / s if s else 1.0
        r["random_death_dec"] = [min(float(m["death_dec_min"]) for m in ne),
                                 max(float(m["death_dec_max"]) for m in ne)]
        r["forager"] = [(m.get("forager_cause"), m.get("forager_survived_dec")) for m in ne]
    return r


def tally() -> dict:
    res: dict = {"apparatus": _apparatus()}
    res["shipped"] = _summ(_load(HEAT, "shipped"))
    for arm in ARMS[1:]:
        res[arm] = _summ(_load(OUT, arm))
    complete = all(res[a][f"n_{s}"] == len(SEEDS) for a in ARMS for s in SPECS)
    if not complete:
        res["verdict"] = None
        return res
    if not res["apparatus"]["ok"]:
        res["verdict"] = "VOID (sweat_paid did not replay the heat race)"
        return res
    base, wins = res["shipped"], {}
    for arm in ARMS[1:]:
        r = res[arm]
        worse = [k for k, v in base["flags"].items() if r["flags"].get(k, 0.0) < v]
        share = r["max_cause_share"] < base["max_cause_share"]
        w1 = all(r["W.1_check_by_seed"].get(s) for s, ok in
                 base["W.1_check_by_seed"].items() if ok)
        r["lost_on"] = worse + ([] if share else ["max_cause_share not lower"]) \
            + ([] if w1 else ["W.1 check lost a seed shipped passes"])
        wins[arm] = not r["lost_on"]
    res["wins"] = wins
    res["verdict"] = ("swp_w72" if wins["swp_w72"] else
                      "swp_k72" if wins["swp_k72"] else "none ships")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "run":
        print(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif cmd == "tally":
        print(json.dumps(tally(), indent=1, default=str))
    else:
        print(json.dumps(check(), indent=1))
