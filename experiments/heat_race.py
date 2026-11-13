"""J6(c) — the heat race: does W0's body lack the one human heat-loss channel?

THE FAULT (read from experiments/needs.py, 11-13): W0 CHARGES sweat but never
PAYS it. The water drain rises 0.5x per C above 37 (`C_SW`; needs.py's own
self-check reads "sweat: water drain at 39 C = 2x"), yet `e_evap` cools only
skin that is EXTERNALLY wet (pool), so a hot dry body has no evaporative loss
at all: its only exit is k_dry = 14.29 W/C. Random flailing (M 150-238 W)
then equilibrates at 40.5-46.6 C, and every random life in both arms of the
J6(b) needs-clock race died of hyperthermia at 445-575 decisions
(docs/research/NEEDS_CLOCK_RACE.md). A human sheds most exercise heat by
sweating; biology is the oracle, so this is raced, not argued.

Arms (patched in-process on `experiments.needs`; no world file is edited
until an arm WINS — the same discipline as J6(b)):
  shipped     the law as shipped (the null).
  sweat       e_evap += min(E_SW_MAX, G_SW * max(0, T - 37)) on DRY skin too;
              water law unchanged (C_SW 0.5 — the sweat it already charges).
  sweat_paid  same cooling; water drain re-priced so every evaporated joule
              is paid at the human ratio: C_SW = WATER_PER_EVAP * G_SW/M_BASAL.
              The wet-skin (pool) evaporation keeps its shipped 0.5/C term.

CONSTANTS, declared from Gagge's two-node model (Gagge, Fobelets & Berglund
1986, ASHRAE Trans 92:709), scaled per basal so W0's compressed clocks keep
their ratios:
  G_SW  = CSW 170 g/(h m2 K) x 0.68 W h/g x 1.8 m2 / 104.8 W basal
        = 1.99 per K  ->  2.0 * M_BASAL  W/K
  E_SW_MAX = 500 g/(h m2) regulatory clamp x 0.68 x 1.8 / 104.8 = 5.8 * M_BASAL
             (W0 has no humidity; the evaporative ceiling is the sweat clamp)
  WATER_PER_EVAP = 1.47 g/h per W evaporated / 104 g/h basal turnover
                   (2.5 L/day) per W-basal-unit = 1.41 (basal-drain units per
                   M_BASAL of evaporation)  ->  C_SW_paid = 1.41 * 2.0 = 2.82
Arithmetic (no run): open day, full activity (M 300 W): shipped T* = 51.0 C,
sweat T* = 37.93 C. Sealed roof by day asleep (k_eff 4.29): shipped T* =
53.2 C (the "oven"), sweat T* = 37.34 C. Night open (T < 37): sweat inactive,
the cold pedagogy is byte-unchanged.

DECLARED RULE (written 2026-11-13 ~16:30 UTC, before any cell exists):
  Per arm, seeds 0-4, on W.1, W.2 and NE.01 run through each spec's own
  `_experiment` (+ `_control` and `_check` for W.1, whose verdict is pass/fail):
  an arm X != shipped SHIPS iff
    (1) on every W.1 / W.2 `*_pass` metric and every NE.01 `ok_*` metric,
        X's seed-mean >= shipped's (no worse anywhere), and
    (2) NE.01's pooled max random-death cause share is STRICTLY below
        shipped's (it must move W0's binding fault, not just tie it), and
    (3) W.1's full `_check` verdict passes on every seed where shipped's does.
  If both win: sweat_paid (the only arm whose water ledger pays for the heat
  it sheds). If neither: the tables go to ORDERS J6 with the clause that lost.
  Clause (2) is the race's self-control: shipped vs itself fails it by
  construction. APPARATUS CHECK (known positive): the shipped NE.01 and W.2
  cells must reproduce /data/jack-logs/needs_race/shipped_* metrics exactly
  (the world is deterministic per seed; same code since 386bfba); a mismatch
  makes the race VOID, not a verdict. REPORTED, not gated: random-life
  decisions to death, the pooled cause table, forager cause/sleep occ, the
  NE.01 gate-5 bar (CAUSE_SHARE_MAX 0.60) against each arm's share.

Usage (nice 19, <= 2 workers, run detached via heat_race.sh):
  python -m experiments.heat_race check            # arithmetic + patch sanity, seconds
  python -m experiments.heat_race run ARM SPEC SEED
  python -m experiments.heat_race tally
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/heat_race")
REF = Path("/data/jack-logs/needs_race")
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("shipped", "sweat", "sweat_paid")
SPECS = {"W.1": "experiments.tests.w_1_heat_balance",
         "W.2": "experiments.tests.w_2_needs_ledger",
         "NE.01": "experiments.tests.ne_01_nobody_survives_by_accident"}

G_SW_PER_BASAL = 2.0          # per K
E_SW_MAX_PER_BASAL = 5.8
WATER_PER_EVAP = 1.41
C_SW_WET = 0.5                # the shipped wet-skin evaporation slope, frozen


def _apply(arm: str):
    N = importlib.reload(importlib.import_module("experiments.needs"))
    if arm != "shipped":
        g, cap = G_SW_PER_BASAL * N.M_BASAL, E_SW_MAX_PER_BASAL * N.M_BASAL

        def e_evap(skin_wetness: float, T: float) -> float:
            warm = max(0.0, T - N.T_SETPOINT)
            wet = N.C_EV * skin_wetness * (1.0 + C_SW_WET * warm)
            return wet + min(cap, g * warm)
        N.e_evap = e_evap
        if arm == "sweat_paid":
            N.C_SW = WATER_PER_EVAP * G_SW_PER_BASAL
    mods = {s: importlib.reload(importlib.import_module(p)) for s, p in SPECS.items()}
    return N, mods


def _t_star(N, M: float, occ: float, T_env: float) -> float:
    T = 37.0
    for _ in range(20000):
        T = N.thermal_step(T, 0.0, 0.0, occ, T_env, 0.2, 0.0, shiver_on=False) \
            + (M - N.M_BASAL) / N.C_EFF * 0.2
    return T


def check() -> dict:
    out = {}
    for arm in ARMS:
        N, _ = _apply(arm)
        out[arm] = {"C_SW": N.C_SW,
                    "e_evap_dry_38C": N.e_evap(0.0, 38.0),
                    "Tstar_open_day_M300": _t_star(N, 300.0, 0.0, N.T_DAY),
                    "Tstar_sealed_day_rest": _t_star(N, 100.0, 1.0, N.T_DAY),
                    "Tstar_open_night_rest": _t_star(N, 100.0, 0.0, N.T_DAY - N.DELTA_T_NIGHT),
                    "w_drain_x_at_38C": 1.0 + N.C_SW * 1.0}
    _apply("shipped")
    return out


def _jsonable(o):
    return getattr(o, "tolist", lambda: str(o))()


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
    p.write_text(json.dumps(rec, default=_jsonable))
    return p


def _cells():
    return [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]


def _apparatus(cells) -> dict:
    """Known positive: shipped must replay the J6(b) cells exactly."""
    bad, n = [], 0
    for c in cells:
        if c["arm"] != "shipped" or c["spec"] not in ("NE.01", "W.2"):
            continue
        ref = REF / f"shipped_{c['spec']}_s{c['seed']}.json"
        if not ref.exists():
            continue
        n += 1
        a = json.loads(json.dumps(c["metrics"], default=_jsonable))
        b = json.loads(ref.read_text())["metrics"]
        if a != b:
            bad.append(f"{c['spec']} s{c['seed']}: " + ",".join(
                sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:6]))
    return {"compared": n, "mismatched": bad, "ok": n > 0 and not bad}


def tally() -> dict:
    cells = _cells()
    res: dict = {"apparatus": _apparatus(cells)}
    for arm in ARMS:
        by = {s: [c for c in cells if c["arm"] == arm and c["spec"] == s] for s in SPECS}
        r = {f"n_{s}": len(v) for s, v in by.items()}
        flags = {}
        for s, v in by.items():
            pref = "ok_" if s == "NE.01" else None
            keys = sorted({k for c in v for k in c["metrics"]
                           if (k.startswith(pref) if pref else k.endswith("_pass"))})
            for k in keys:
                flags[f"{s}:{k}"] = sum(float(c["metrics"].get(k, 0)) for c in v) / len(v)
        r["flags"] = flags
        r["W.1_check_by_seed"] = {c["seed"]: c.get("check") for c in by["W.1"]}
        ne = [c["metrics"] for c in by["NE.01"]]
        if ne:
            causes = [k for k in ne[0] if k.startswith("deaths_") and "microsleep" not in k]
            tot = {k[7:]: sum(float(m[k]) for m in ne) for k in causes}
            s = sum(tot.values())
            r["causes"] = tot
            r["max_cause_share"] = max(tot.values()) / s if s else 1.0
            r["random_death_dec"] = [min(float(m["death_dec_min"]) for m in ne),
                                     max(float(m["death_dec_max"]) for m in ne)]
            r["forager"] = [(m.get("forager_cause"), m.get("forager_survived_dec"),
                             round(float(m.get("forager_sleep_occ", -1)), 3)) for m in ne]
        res[arm] = r
    complete = all(res[a][f"n_{s}"] == len(SEEDS) for a in ARMS for s in SPECS)
    if not complete:
        res["verdict"] = None
        return res
    if not res["apparatus"]["ok"]:
        res["verdict"] = "VOID (shipped did not replay J6(b))"
        return res
    base = res["shipped"]
    wins = {}
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
    res["verdict"] = ("sweat_paid" if wins["sweat_paid"] else
                      "sweat" if wins["sweat"] else "none ships")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "run":
        print(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif cmd == "tally":
        print(json.dumps(tally(), indent=1, default=str))
    else:
        print(json.dumps(check(), indent=1))
