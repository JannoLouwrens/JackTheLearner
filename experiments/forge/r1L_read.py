"""Round 1-L's READER (ORDERS S20 (2), rulings 122/123): raw kernel files ->
the harness verdict for K1 against the baseline, check #13's books, and
beside it rung M's late-half rate per arm under the frozen metric hash.

`r1_read` stays locked to round 1 and is not edited; this file imports its
per-run scorer and the rule's settings/world, which sit under r1's lock.
Writes nothing: it prints. Refuses
  * a spec whose sha256 differs from the freeze (`FREEZE`'s blob), and any
    kernel file run under another spec, round, rung, or more than one pin;
  * metric code whose hash differs from the spec's frozen `metric_sha256`;
  * a run that is not a cell of a named tranche, or one in two files;
  * a named tranche with a run missing (a cut kernel is re-run, J18);
  * a finished run that carried another commit, settings or world.
Primary: `outcome_sustained(lives, W, T)` per run (a crashed or cut run's
lives end in "crash": censored at T, counted against its arm, never dropped),
in launch order (tranche, then seed), through `harness.looks.compare_looks`
— look 1 with tranche 1 home, look 2 with both. The spec names no
permutation seed or count: `compare`'s own n_perm (2000) and seed 0, printed.

    python -m experiments.forge.r1L_read --tranches 1        # look 1
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from experiments.forge import Refused
from experiments.forge.metric import metric_hash
from experiments.forge.r1_read import _WORLD, rule_settings, score_run
from experiments.harness import check_runs_accounted, outcome_sustained
from experiments.harness.looks import compare_looks

REPO = Path(__file__).resolve().parents[2]
ROUND = Path(__file__).resolve().parent / "rounds" / "r1L"
RAW = Path("/data/jack-logs/forge/r1L/raw")
FREEZE = "52033df2"
N_PERM, SEED = 2000, 0


def frozen_sha(commit: str = FREEZE) -> str:
    p = subprocess.run(["git", "-C", str(REPO), "show",
                        f"{commit}:experiments/forge/rounds/r1L/spec.json"], capture_output=True)
    if p.returncode:
        raise Refused(f"no r1L spec at the freeze {commit}")
    return hashlib.sha256(p.stdout).hexdigest()


def manifest(spec: dict, tranches) -> dict:
    """{arm: [seed, ...]} in launch order for the named tranches."""
    out = {}
    for t in tranches:
        for grp in spec["rungs"]["L"]["tranches"][str(t)]:
            for arm in grp["arms"]:
                out.setdefault(arm, []).extend(grp["seeds"])
    return out


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else math.nan


def harness_lives(rec: dict, status: str) -> list:
    lives = [(_num(l.get("decisions")), l.get("cause") == "capped",
              _num(l.get("intake")), _num(l.get("basal_drain"))) for l in rec.get("lives") or []]
    return lives if status == "ok" else lives + ["crash"]


def read(spec: dict, files: dict, tranches, *, spec_sha: str, freeze_sha: str,
         code_hash: str | None = None) -> dict:
    if spec_sha != freeze_sha:
        raise Refused("rounds/r1L/spec.json differs from the freeze")
    if (metric_hash() if code_hash is None else code_hash) != spec["outcome"]["metric_sha256"]:
        raise Refused("the metric code on disk differs from the round's frozen hash")
    prim = spec["outcome"]["primary"]
    T, W, late = spec["rungs"]["L"]["decisions"], prim["W"], spec["outcome"]["late_frac"]
    plan = manifest(spec, tranches)
    want = {(a, s) for a, ss in plan.items() for s in ss}
    if not files:
        raise Refused("no kernel file home yet")
    pins = {p.get("pin") for p in files.values()}
    if len(pins) != 1 or not isinstance(next(iter(pins)), str) or len(next(iter(pins))) != 40:
        raise Refused(f"the kernels do not share one pinned commit: {sorted(map(str, pins))}")
    pin, runs = next(iter(pins)), {}
    for name in sorted(files):
        p = files[name]
        if p.get("round") != "r1L" or p.get("rung") != "L" or p.get("spec_sha256") != freeze_sha:
            raise Refused(f"{name}: not round r1L rung L under the frozen spec")
        for key, rec in sorted((p.get("cells") or {}).items()):
            cell = (rec.get("arm"), rec.get("seed"))
            if key != f"{cell[0]}_s{cell[1]}" or cell not in want:
                raise Refused(f"{name}: run {key} is not a cell of tranche(s) {list(tranches)}")
            if cell in runs:
                raise Refused(f"run {key} is in two kernel files")
            got = score_run(rec, T, late)
            if got["status"] == "ok":
                f = rec.get("forge") or {}
                if f.get("code_head") != pin or f.get("rung") != "L" or rec.get("T") != T:
                    raise Refused(f"{name}: {key} is not a rung-L run of the pin {pin[:10]}")
                if f.get("settings") != rule_settings(spec, cell[0]):
                    raise Refused(f"{name}: {key} did not run the rule's settings for its arm")
                if [k for k in _WORLD if (rec.get("world") or {}).get(k) != spec["world"][k]]:
                    raise Refused(f"{name}: {key} ran another world than the rule's")
            runs[cell] = (got, harness_lives(rec, got["status"]))
    if set(runs) != want:
        raise Refused(f"runs missing (re-run their kernel from {pin[:10]}): "
                      f"{sorted(want - set(runs))}")
    a, b = prim["a"], prim["b"]
    scored = {arm: {f"{arm}_s{s}": outcome_sustained(runs[(arm, s)][1], W, T) for s in plan[arm]}
              for arm in (a, b)}
    check_runs_accounted({arm: list(scored[arm]) for arm in (a, b)}, scored)
    v = compare_looks(list(scored[a].values()), list(scored[b].values()), N_PERM, SEED)
    beside = {arm: [runs[(arm, s)][0]["score"] for s in ss] for arm, ss in plan.items()}
    return {"verdict": v, "books": {arm: len(ss) for arm, ss in plan.items()},
            "crossed": {arm: sum(o.event for o in scored[arm].values()) for arm in (a, b)},
            "status": {f"{c[0]}_s{c[1]}": g["status"] for c, (g, _) in runs.items()
                       if g["status"] != "ok"},
            "rate_M": beside, "pin": pin}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranches", default="1", help="comma list, e.g. 1 or 1,2")
    tr = [int(x) for x in ap.parse_args(argv).tranches.split(",")]
    sp = ROUND / "spec.json"
    try:
        out = read(json.loads(sp.read_text()),
                   {f.name: json.loads(f.read_text()) for f in sorted(RAW.glob("L_g*.json"))},
                   tr, spec_sha=hashlib.sha256(sp.read_bytes()).hexdigest(),
                   freeze_sha=frozen_sha())
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1
    v = out["verdict"]
    print(f"PRIMARY K1 vs E-food: {v.decision} at look {v.look}  p1 {v.p1:.4f}  p2 {v.p2:.4f}"
          f"  (n_perm {N_PERM}, seed {SEED}, pin {out['pin'][:10]})")
    print(f"check #13 books: {out['books']}  crossed: {out['crossed']}  not ok: {out['status']}")
    for arm, xs in out["rate_M"].items():
        print(f"  {arm:<11} rung-M late-half rate mean {sum(xs) / len(xs):.3f}  "
              + " ".join(f"{x:.2f}" for x in xs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
