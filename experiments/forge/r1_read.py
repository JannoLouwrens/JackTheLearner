"""Round 1's READER (F3 / ORDERS J18): raw kernel files -> `results.jsonl` rows.

This file is part of the JUDGE: it is named in `rounds/r1/judge_lock.json`,
which is committed alone BEFORE the first raw file is opened (S13 b). It was
written without opening one. It imports only the locked judge (the ladder
rule and the metric) — not the kernel file, which J18 lets change between
rungs — so nothing it depends on can move under the lock.

What it does, and nothing else:
  refuses   unless the judge lock holds; unless the rung's kernel files hold
            exactly the rule's cells (arms x seeds), once each; unless every
            file is this round, this rung, this frozen spec and ONE pinned
            commit, and every finished run carries that commit, the rule's
            settings for its arm and the rule's world; and if the rung was
            already read (a rung is read once).
  a cut     a file in which NO run finished is a cut kernel: refused, to be
  kernel    re-run from the pinned commit, not scored (J18).
  scores    a finished run by `metric.sustained_rate` over its lives as the
            harness reads them (an unfinished life is not capped), late half.
            A crashed, NaN, cut or missing run scores 0 and counts against
            its arm (the rule's `crash` clause).
  tiebreak  late meals per life: mean meals (floor + apples) over the same
            late lives the score counts. Only orders arms equal on the score.
  cost      a run held one core for its kernel's duration:
            core_hours = kernel duration; kernel_hours = duration / runs in
            that kernel (so the rows of a kernel sum to its duration).

    python -m experiments.forge.r1_read --rung S          # writes results.jsonl
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from experiments.forge import Refused, check_lock, entrants, validate_spec, yardstick
from experiments.forge.metric import metric_hash, sustained_rate

ROUND = Path(__file__).resolve().parent / "rounds" / "r1"
RAW = Path("/data/jack-logs/forge/r1/raw")
_NOT_LEARNER = ("note", "reset_every", "buffer", "norm_obs")   # prose / fixed for every arm
_WORLD = ("food_floor_only", "eye_frame", "motor_frame", "native_substep",
          "lazy_contact_audio", "difficulty", "reward")


def rung_arms(spec: dict, rows: list, rung: str) -> list:
    """Who runs at `rung`: baseline, twin(s), then every candidate at S; above
    S the ladder's climbers and the yardstick control."""
    who = [spec["baseline"], *spec["twins"], *entrants(spec, rows, rung)]
    y = yardstick(spec, rows, rung)
    return who + ([y] if y and y not in who else [])


def rule_settings(spec: dict, arm: str) -> dict:
    """The rule's learner settings for one arm (baseline overlaid by its own)."""
    st = dict(spec["baseline_settings"])
    if arm in spec["twins"]:
        st.update(spec["twin_settings"][arm])
    elif arm != spec["baseline"]:
        st.update({c["id"]: c for c in spec["candidates"]}[arm]["settings"])
    return {k: v for k, v in st.items() if k not in _NOT_LEARNER}


def _late(lives: list, total: float, late_frac: float) -> list:
    """The lives `sustained_rate` counts as late (same walk, same bounds)."""
    start, cum, out = late_frac * total, 0.0, []
    for life in lives:
        cum += float(life["decisions"])
        if cum > total:
            break
        if cum > start:
            out.append(life)
    return out


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def score_run(rec: dict, T: int, late_frac: float) -> dict:
    """One run -> {status, score, tiebreak, late_lives, late_sustained, lives}."""
    lives = rec.get("lives") or []
    out = {"status": "ok", "score": 0.0, "tiebreak": 0.0, "late_lives": 0,
           "late_sustained": 0, "lives": len(lives)}
    if rec.get("not_found"):
        out["status"] = "missing"
    elif rec.get("cut") or rec.get("from_partial"):
        out["status"] = "cut"
    elif "error" in rec or rec.get("rc", 0) != 0:
        out["status"] = "crash"
    elif not _finite(rec.get("decisions")) or rec["decisions"] < T:
        out["status"] = "cut"
    elif any(not all(_finite(l.get(k)) for k in ("decisions", "intake", "basal_drain",
                                                  "floor_meals", "apples"))
             or l["decisions"] < 0 for l in lives):
        out["status"] = "crash"
    if out["status"] != "ok":
        return out
    tuples = [(l["decisions"], l["cause"] == "capped", l["intake"], l["basal_drain"])
              for l in lives]
    late = _late(lives, T, late_frac)
    out["score"] = float(sustained_rate(tuples, T, late_frac))
    out["late_lives"] = len(late)
    out["late_sustained"] = sum(l["cause"] == "capped" and l["intake"] >= l["basal_drain"]
                                for l in late)
    out["tiebreak"] = (sum(l["floor_meals"] + l["apples"] for l in late) / len(late)
                       if late else 0.0)
    return out


def score_rung(spec: dict, rung: str, files: dict, prior_rows: list, *,
               spec_sha: str, code_hash: str | None = None) -> list:
    """`files`: {file name: parsed kernel payload} for ONE rung. Pure: returns
    the rung's result rows or raises Refused; writes nothing."""
    validate_spec(spec)
    frozen = spec["outcome"]["metric_sha256"]
    if (metric_hash() if code_hash is None else code_hash) != frozen:
        raise Refused("the metric code on disk differs from the round's frozen hash")
    if any(r.get("rung") == rung for r in prior_rows):
        raise Refused(f"rung {rung} is already read; a rung is read once")
    if not files:
        raise Refused(f"no kernel file for rung {rung}")
    T, seeds = spec["rungs"][rung]["decisions"], spec["rungs"][rung]["seeds"]
    late_frac = spec["outcome"]["late_frac"]
    want = {(a, s) for a in rung_arms(spec, prior_rows, rung) for s in seeds}
    pins = {p.get("pin") for p in files.values()}
    if len(pins) != 1 or not isinstance(next(iter(pins)), str) or len(next(iter(pins))) != 40:
        raise Refused(f"rung {rung}'s kernels do not share one pinned commit: {sorted(map(str, pins))}")
    pin = next(iter(pins))
    rows, seen = [], set()
    for name in sorted(files):
        p = files[name]
        if p.get("round") != spec["round"] or p.get("rung") != rung:
            raise Refused(f"{name}: not round {spec['round']} rung {rung}")
        if p.get("spec_sha256") != spec_sha:
            raise Refused(f"{name}: ran under another spec than the frozen rule")
        dur = (p.get("kernel") or {}).get("duration_s")
        if not _finite(dur) or dur <= 0:
            raise Refused(f"{name}: no kernel duration — its cost cannot be stated")
        cells = p.get("cells") or {}
        scored = []
        for key, rec in sorted(cells.items()):
            cell = (rec.get("arm"), rec.get("seed"))
            if key != f"{cell[0]}_s{cell[1]}" or cell not in want:
                raise Refused(f"{name}: run {key} is not a cell of rung {rung}")
            if cell in seen:
                raise Refused(f"run {key} is in two kernel files")
            seen.add(cell)
            got = score_run(rec, T, late_frac)
            if got["status"] == "ok":
                f = rec.get("forge") or {}
                if f.get("code_head") != pin:
                    raise Refused(f"{name}: {key} ran commit {str(f.get('code_head'))[:10]}, "
                                  f"not the pin {pin[:10]}")
                if f.get("rung") != rung or rec.get("T") != T:
                    raise Refused(f"{name}: {key} is not a rung-{rung} run of {T} decisions")
                if f.get("settings") != rule_settings(spec, cell[0]):
                    raise Refused(f"{name}: {key} did not run the rule's settings for its arm")
                bad = [k for k in _WORLD if (rec.get("world") or {}).get(k) != spec["world"][k]]
                if bad:
                    raise Refused(f"{name}: {key} ran another world than the rule's: {bad}")
            scored.append((cell, got))
        if not any(g["status"] == "ok" for _, g in scored):
            raise Refused(f"{name}: no run finished — a cut kernel is re-run from the pinned "
                          f"commit {pin[:10]}, not scored (J18)")
        for (arm, seed), g in scored:
            rows.append({"arm": arm, "rung": rung, "seed": seed, "score": g["score"],
                         "tiebreak": round(g["tiebreak"], 6), "status": g["status"],
                         "late_lives": g["late_lives"], "late_sustained": g["late_sustained"],
                         "lives": g["lives"], "core_hours": round(dur / 3600.0, 4),
                         "kernel_hours": round(dur / 3600.0 / len(scored), 4),
                         "metric_sha256": frozen, "code_head": pin, "kernel": name})
    if seen != want:
        raise Refused(f"rung {rung}: runs missing from the kernel files (re-run their kernel "
                      f"from the pin {pin[:10]}): {sorted(want - seen)}")
    order = {a: i for i, a in enumerate(rung_arms(spec, prior_rows, rung))}
    return sorted(rows, key=lambda r: (order[r["arm"]], r["seed"]))


def read(rung: str, round_dir=ROUND, raw_dir=RAW) -> list:
    """Lock first, then open the raw files, then append the rung's rows."""
    round_dir, raw_dir = Path(round_dir), Path(raw_dir)
    check_lock(round_dir)               # before any raw file is opened
    spec_path, res = round_dir / "spec.json", round_dir / "results.jsonl"
    spec = json.loads(spec_path.read_text())
    prior = ([json.loads(x) for x in res.read_text().splitlines() if x.strip()]
             if res.is_file() else [])
    files = {f.name: json.loads(f.read_text()) for f in sorted(raw_dir.glob(f"{rung}_g*.json"))}
    rows = score_rung(spec, rung, files, prior,
                      spec_sha=hashlib.sha256(spec_path.read_bytes()).hexdigest())
    tmp = res.with_name(res.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in prior + rows))
    os.replace(tmp, res)
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=["S", "M"])
    a = ap.parse_args(argv)
    try:
        rows = read(a.rung)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1
    for r in rows:
        print(f"{r['arm']:<11} s{r['seed']} {r['status']:<7} score {r['score']:.4f} "
              f"late {r['late_sustained']}/{r['late_lives']} meals/life {r['tiebreak']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
