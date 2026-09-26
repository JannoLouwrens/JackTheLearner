"""One-shot measurement of the DECIDING set for option (iv) of
`hash-salt-lottery-in-a-gated-metric` (docs/REVIEW_QUEUE.md, disposition
2026-09-19) — ordered to run and be REPORTED before any implementation.

DECLARED MARGIN — written down here, with its reasoning, BEFORE any spec is
scanned, per the disposition's first binding ("a margin chosen after seeing
which specs it captures is the venue-selection defect wearing a threshold"):

    MARGIN_REL = 0.10   (relative: a metric is perturbed by 0.10 * |value|)
    ZERO_ABS   = 0.10   (a metric recorded exactly 0.0 gets +/-0.10 absolute)

Reasoning: the one measured instance of the defect (the row's own table)
moved aggregate `swap_agree` across 0.8333-0.8889 under salts {0,1,7,42,
12345} — a spread of 0.0556, which is 6.2% of its 0.90 bar. A 10% relative
margin covers that measured spread with ~1.6x headroom without being fit to
this ladder's current values. ZERO_ABS: most gated metrics on this ladder are
unit-scaled (accuracies, fractions, ratios in [0,1]); 0.10 absolute equals
the relative delta at value 1.0, so a metric sitting exactly at 0.0 is probed
on the same scale as its unit-scaled siblings rather than skipped.

BINDING is detected mechanically, with no hand-mapping of metric names to
threshold constants: replay the spec's own pre-registered `_check` offline
against the recorded row (the 2026-09-25 sweep idiom), then perturb ONE
numeric metric at a time by the declared margin in each direction and replay
again. If the normalized verdict (PASS/FAIL/VOID) changes, the recorded
verdict is sensitive to that metric within the margin — i.e. the metric is
DECIDING, by the actual gate logic rather than by a name-pairing heuristic.

Scope per the disposition: CPU cost classes only (Budget.CPU, CPU_FAST,
CPU_LONG, CPU_DAYS); GPU classes are excluded unconditionally. Latest row per
spec. Known limits, disclosed rather than hidden: impure `_checks` (LG.10 /
LG.12 read module globals) and unreplayable rows (drifted impls, GL env,
renamed modules) are counted and listed, not silently dropped.
"""
import copy
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/opc/jackthelearner")

from experiments.protocol import Status, module_path_for
from experiments.registry import BY_ID

MARGIN_REL = 0.10
ZERO_ABS = 0.10
CPU_CLASSES = {"Budget.CPU", "Budget.CPU_FAST", "Budget.CPU_LONG",
               "Budget.CPU_DAYS"}
CPU_DAY_CEILING_S = 57600.0

led = json.load(open("/home/opc/jackthelearner/experiments/ledger.json"))["results"]


def norm(v):
    if isinstance(v, Status):
        return v.name
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, (int, float)) and float(v) in (0.0, 1.0):
        return "PASS" if v else "FAIL"
    return f"OTHER:{type(v).__name__}"


def call_check(mod, m, c):
    import inspect
    fn = mod._check
    n = len(inspect.signature(fn).parameters)
    return fn(m, c) if n >= 2 else fn(m)


def replay(mod, m, c):
    try:
        return norm(call_check(mod, m, c)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


out = {"deciding": {}, "unreplayable": {}, "no_check": [], "no_row": [],
       "clean": [], "baseline_mismatch": {}}

for sid, spec in sorted(BY_ID.items()):
    if str(spec.budget) not in CPU_CLASSES:
        continue
    row = led.get(sid)
    if not row:
        out["no_row"].append(sid)
        continue
    path = module_path_for(sid)
    if path is None:
        out["unreplayable"][sid] = "no implementation file"
        continue
    modname = "experiments.tests." + Path(path).stem
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        out["unreplayable"][sid] = f"import: {type(e).__name__}: {e}"
        continue
    if not hasattr(mod, "_check"):
        out["no_check"].append(sid)
        continue
    m0 = row.get("metrics") or {}
    c0 = row.get("control_metrics") or {}
    base, err = replay(mod, copy.deepcopy(m0), copy.deepcopy(c0))
    if base is None:
        out["unreplayable"][sid] = err
        continue
    if base != row.get("status"):
        # replay disagrees with the recorded verdict (impure _check or
        # hand-repaired row) — a lead, not a verdict; excluded from the
        # perturbation scan because the baseline is not trustworthy.
        out["baseline_mismatch"][sid] = {"recorded": row.get("status"),
                                         "replay": base}
        continue
    flips = []
    for which, d0 in (("m", m0), ("c", c0)):
        for k, v in d0.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            delta = MARGIN_REL * abs(v) if v != 0.0 else ZERO_ABS
            dirs = []
            for sgn in (+1, -1):
                m1, c1 = copy.deepcopy(m0), copy.deepcopy(c0)
                (m1 if which == "m" else c1)[k] = v + sgn * delta
                got, err2 = replay(mod, m1, c1)
                if got is not None and got != base:
                    dirs.append((sgn, got))
            if dirs:
                flips.append({"dict": which, "metric": k, "value": v,
                              "dirs": dirs,
                              # both directions flipping = an exact-equality /
                              # band gate: the value sits AT its bound, distance
                              # zero. One direction = an inequality bar within
                              # the declared margin.
                              "equality_like": len(dirs) == 2,
                              "at_bound_zero": v == 0.0})
    if flips:
        out["deciding"][sid] = {
            "status": row.get("status"), "budget": str(spec.budget),
            "duration_s": row.get("duration_s"), "flips": flips}
    else:
        out["clean"].append(sid)

# ---- summary ----
dec = out["deciding"]
zero_only = [s for s, d in dec.items()
             if all(f["at_bound_zero"] for f in d["flips"])]
near = [s for s in dec if s not in zero_only]
eq_only = [s for s, d in dec.items()
           if all(f["equality_like"] for f in d["flips"])]
has_onesided = [s for s, d in dec.items()
                if any(not f["equality_like"] for f in d["flips"])]
cost = sum(d["duration_s"] or 0 for d in dec.values())
cost_near = sum(dec[s]["duration_s"] or 0 for s in near)
cost_onesided = sum(dec[s]["duration_s"] or 0 for s in has_onesided)
print(json.dumps(out, indent=1, default=str))
print("=" * 60, file=sys.stderr)
print(f"CPU specs with rows scanned: "
      f"{len(dec) + len(out['clean']) + len(out['baseline_mismatch'])}",
      file=sys.stderr)
print(f"DECIDING (any flip): {len(dec)}  "
      f"[near-margin: {len(near)}, at-bound-zero only: {len(zero_only)}]",
      file=sys.stderr)
print(f"clean: {len(out['clean'])}  unreplayable: {len(out['unreplayable'])}  "
      f"baseline-mismatch: {len(out['baseline_mismatch'])}  "
      f"no_check: {len(out['no_check'])}  no_row: {len(out['no_row'])}",
      file=sys.stderr)
print(f"doubled cost of DECIDING set: {cost:.0f} s "
      f"({100 * cost / CPU_DAY_CEILING_S:.1f}% of CPU_DAY_CEILING_S "
      f"{CPU_DAY_CEILING_S:.0f} s)", file=sys.stderr)
print(f"doubled cost of near-margin subset alone: {cost_near:.0f} s "
      f"({100 * cost_near / CPU_DAY_CEILING_S:.1f}%)", file=sys.stderr)
print(f"near-margin specs: {sorted(near)}", file=sys.stderr)
print(f"at-bound-zero-only specs: {sorted(zero_only)}", file=sys.stderr)
print(f"equality-gate-only specs (every flip is two-sided): {len(eq_only)} "
      f"{sorted(eq_only)}", file=sys.stderr)
print(f"specs with >=1 one-sided (inequality-bar-within-margin) flip: "
      f"{len(has_onesided)}, doubled cost {cost_onesided:.0f} s "
      f"({100 * cost_onesided / CPU_DAY_CEILING_S:.1f}%)", file=sys.stderr)
n_eq = sum(1 for d in dec.values() for f in d["flips"] if f["equality_like"])
n_one = sum(1 for d in dec.values() for f in d["flips"]
            if not f["equality_like"])
print(f"flipping metrics: {n_eq + n_one} total = {n_one} one-sided + "
      f"{n_eq} equality/band", file=sys.stderr)
