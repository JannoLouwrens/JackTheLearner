"""T0.13 — no gate in the ladder is decorative.

Law 1 says a capability may be claimed only by a test that COULD have failed.
This spec applies that law one level up, to the machine rather than the
science: an assertion inside a `_check` that cannot change the check's verdict
is a green tick nobody earned, and the ladder has no way to see it.

The motivating case, found by the 2026-08-09 overseer audit:

    return (m["ok"] and m["cuda_available"] and m["matmul_finite"]
            and "NVIDIA" in m["gpu"].upper() or "TESLA" in m["gpu"].upper()) \\
        and m["artifact_bytes"] > 0 and not c["ok"]

`and` binds tighter than `or`, so this reads
`(ok and cuda and matmul and "NVIDIA" in gpu) or ("TESLA" in gpu)`. Colab's
device string is literally "Tesla T4", so the right branch was true on every
real run and the three assertions on the left were never consulted. T0.09 —
the certification spec for the backend that ran T1.07, T1.12 and every T1.02
attempt — would have passed a job that reported failure, had no CUDA, and
returned non-finite results.

Two detectors, because they fail differently:

  1. SENSITIVITY (the general one). For every PASSing spec, replay its stored
     `_check` against its stored metrics, then perturb one referenced key at a
     time. A key that cannot move the verdict under any perturbation is inert.
     Deliberately measured AT THE OPERATING POINT — the values the run actually
     produced — not over a hypothetical input space. "Could this assertion have
     fired on a run like the ones we really get?" is the question Law 1 asks;
     "is it reachable for some fictional input" is not.

  2. PRECEDENCE (the narrow one). Parse every `_check` and flag an `or` whose
     operands include an unparenthesised `and`, which is the exact shape above.
     Kept separate because it fires on source that is *currently* harmless but
     one metric value away from silently disarming a gate.

Control: the pre-fix T0.09 check, verbatim, against T0.09's recorded metrics.
Both detectors must flag it. Without that, a scan finding nothing would be
indistinguishable from a scan that does not work — which is precisely the
"silence is not success" failure this repo has paid for repeatedly.
"""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
from typing import Any, Callable

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# The pre-fix T0.09 gate, preserved verbatim as the control fixture. Do not
# "tidy" the precedence here — the missing brackets ARE the fixture.
CONTROL_SRC = '''
def _check(m, c):
    return (m["ok"] and m["cuda_available"] and m["matmul_finite"]
            and "NVIDIA" in m["gpu"].upper() or "TESLA" in m["gpu"].upper()) \\
        and m["artifact_bytes"] > 0 and not c["ok"]
'''

# The metrics T0.09 actually recorded on 2026-08-04, so the control is scored
# at a real operating point rather than an invented one.
CONTROL_M = {"ok": True, "gpu": "Tesla T4, 15360 MiB", "cuda_available": True,
             "device": "Tesla T4", "matmul_finite": True, "artifact_bytes": 124,
             "duration_s": 96.0, "message": ""}
CONTROL_C = {"ok": False, "message": "no such accelerator"}


def _perturbations(v: Any) -> list:
    """Same-type alternatives. Type-preserving on purpose: swapping in a None
    would raise a TypeError inside most checks, and an exception would read as
    'the key matters' when it only means the key was compared."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, (int, float)):
        return [0, 1, -1, v + 1, v - 1, 1e9, -1e9]
    if isinstance(v, str):
        return ["", "ZZZ_NOT_A_REAL_VALUE"]
    if isinstance(v, (list, tuple)):
        return [type(v)()]
    if isinstance(v, dict):
        return [{}]
    return []


def _source(fn: Callable, src: str | None) -> str | None:
    """Explicit source wins. The control fixture is built with exec(), and
    `inspect.getsource` raises OSError on a function with no file on disk — so
    relying on inspect alone made the control scan ZERO gates and report a
    clean bill of health for the known-bad one. Caught by the control itself on
    T0.13's first run."""
    if src is not None:
        return _dedent(src)
    try:
        return _dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None


def _fdef(src: str | None):
    if src is None:
        return None
    return next((n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef)), None)


def _referenced_keys(fn: Callable, src: str | None) -> tuple[set, set, set]:
    """Keys READ as m["..."] / c["..."], by parameter POSITION not by name —
    checks in this repo variously call them m/c, m/_c, metrics/control.

    Load context only. Several checks compute a derived figure and store it
    back (`m["resume_fidelity_ratio"] = ...` in T0.04, `m["verdict"]` in
    T2.02); those are OUTPUTS of the gate, not inputs to it, and counting them
    reported three inert keys that were never assertions at all.

    Third return value: the subset that appears ONLY inside an `or`. An inert
    key there is redundancy by design — T1.09's control asserts `absurd_oom or
    absurd_peak_gb > MAX_GB` because a run that OOMs has no peak to read, so
    one branch is necessarily dead and no correct rewrite avoids it. An inert
    key under pure `and` is a disarmed assertion, which is the real defect.
    """
    fdef = _fdef(src)
    if fdef is None:
        return set(), set(), set()
    names = [a.arg for a in fdef.args.args]
    m_name = names[0] if names else None
    c_name = names[1] if len(names) > 1 else None

    # Every subscript that sits somewhere beneath an `or`.
    under_or = set()
    for node in ast.walk(fdef):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for sub in ast.walk(node):
                under_or.add(id(sub))

    m_keys, c_keys = set(), set()
    inside, outside = set(), set()
    for node in ast.walk(fdef):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and isinstance(node.ctx, ast.Load)):
            continue
        key = node.slice.value
        if node.value.id == m_name:
            tag, bucket = "m", m_keys
        elif node.value.id == c_name:
            tag, bucket = "c", c_keys
        else:
            continue
        bucket.add(key)
        (inside if id(node) in under_or else outside).add(f"{tag}[{key!r}]")
    # A key read anywhere OUTSIDE an `or` is load-bearing as a conjunct, even
    # if it is also read inside one. Only reads that are exclusively disjunctive
    # get the redundancy exemption.
    return m_keys, c_keys, inside - outside


def _dedent(src: str) -> str:
    lines = src.splitlines()
    pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[pad:] for l in lines)


def _verdict(fn: Callable, m: dict, c: dict):
    """Deep-copied every call: some checks WRITE to their metrics (T2.02 sets
    m["verdict"]), and a detector that let that leak would score the next
    perturbation against a mutated baseline."""
    try:
        out = fn(copy.deepcopy(m), copy.deepcopy(c))
    except Exception as e:
        return ("RAISED", type(e).__name__)
    if isinstance(out, Status):
        return ("STATUS", out.value)
    return ("BOOL", bool(out))


def _inert_keys(fn: Callable, src: str | None, m: dict, c: dict) -> tuple[list, list]:
    """Referenced keys that cannot move the verdict at this operating point.

    Returns (disarmed, redundant): keys read under pure conjunction, and keys
    read only inside an `or`. Only the first class is gated.
    """
    base = _verdict(fn, m, c)
    if base[0] == "RAISED":
        return [], []        # cannot score a gate we cannot evaluate; reported separately
    m_keys, c_keys, disjunct_only = _referenced_keys(fn, src)
    disarmed, redundant = [], []
    for which, keys, store in (("m", m_keys, m), ("c", c_keys, c)):
        for k in sorted(keys):
            if k not in store:
                continue
            moved = False
            for alt in _perturbations(store[k]):
                probe = dict(store)
                probe[k] = alt
                got = _verdict(fn, probe, c) if which == "m" else _verdict(fn, m, probe)
                if got != base:
                    moved = True
                    break
            if not moved:
                label = f"{which}[{k!r}]"
                (redundant if label in disjunct_only else disarmed).append(label)
    return disarmed, redundant


def _precedence_hazards(fn: Callable, src: str | None) -> int:
    """An `or` with an `and` among its operands. Python's grammar drops
    redundant parens, so `(a and b) or c` and `a and b or c` share an AST —
    both are reported. That is intended: the first is only safe by luck of
    where the author happened to stop typing, and the audit's finding is that
    nothing in the ladder distinguishes them.

    This is the detector that separates a DISARMED gate from honest
    redundancy. T1.09's `absurd_oom or absurd_peak_gb > MAX_GB` has no `and`
    among its operands and is not flagged; the pre-fix T0.09 gate is.
    """
    if src is None:
        return 0
    n = 0
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            if any(isinstance(v, ast.BoolOp) and isinstance(v.op, ast.And)
                   for v in node.values):
                n += 1
    return n


def _scan(entries: list) -> dict:
    """entries: (spec_id, check_fn, source_or_None, metrics, control_metrics)."""
    disarmed, redundant, hazards, unevaluable, unreadable, scanned = {}, {}, {}, [], [], 0
    for spec_id, fn, raw_src, m, c in entries:
        scanned += 1
        src = _source(fn, raw_src)
        if src is None or _fdef(src) is None:
            # A gate whose source cannot be read is a gate that was not
            # scanned. Reported, never silently skipped.
            unreadable.append(spec_id)
            continue
        if _verdict(fn, m, c)[0] == "RAISED":
            unevaluable.append(spec_id)
        h = _precedence_hazards(fn, src)
        if h:
            hazards[spec_id] = h
        dis, red = _inert_keys(fn, src, m, c)
        # The redundancy exemption is forfeited by a gate that ALSO carries a
        # precedence hazard. Structure alone cannot separate honest redundancy
        # from a disarmed assertion — the pre-fix T0.09 keys and T1.09's
        # `absurd_peak_gb` are both inert operands of an `or`, and both look
        # identical to the sensitivity detector. What distinguishes them is
        # that T0.09's `or` was never written: `and` binding tighter turned an
        # intended conjunction into one. So an `or` that swallows an `and` is
        # evidence of intent, and its dead operands are defects, not slack.
        if h:
            dis, red = sorted(dis + red), []
        if dis:
            disarmed[spec_id] = dis
        if red:
            redundant[spec_id] = red
    return {
        "gates_scanned": scanned,
        "disarmed_conjunct_keys": sum(len(v) for v in disarmed.values()),
        "specs_with_disarmed_keys": len(disarmed),
        "redundant_disjunct_keys": sum(len(v) for v in redundant.values()),
        "inert_gate_keys": (sum(len(v) for v in disarmed.values())
                            + sum(len(v) for v in redundant.values())),
        "precedence_hazards": sum(hazards.values()),
        "specs_with_precedence_hazards": len(hazards),
        "unevaluable_gates": len(unevaluable),
        "unreadable_gates": len(unreadable),
        "disarmed_detail": "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(disarmed.items())),
        "redundant_detail": "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(redundant.items())),
        "hazard_detail": ", ".join(sorted(hazards)),
        "unevaluable_detail": ", ".join(sorted(unevaluable)),
        "unreadable_detail": ", ".join(sorted(unreadable)),
    }


def _experiment(seed: int) -> dict:
    ledger = Ledger()
    entries = []
    for spec_id, r in _passing(ledger):
        mod_name = spec_id.lower().replace(".", "_")
        try:
            mod = importlib.import_module(f"..tests.{_module_stem(mod_name)}", __package__)
        except Exception:
            continue
        fn = getattr(mod, "_check", None)
        if fn is None:
            continue
        entries.append((spec_id, fn, None, dict(r.get("metrics") or {}),
                        dict(r.get("control_metrics") or {})))
    return _scan(entries)


def _module_stem(prefix: str) -> str:
    from ..run import TESTS_DIR
    matches = sorted(p.stem for p in TESTS_DIR.glob(f"{prefix}_*.py"))
    if not matches:
        raise FileNotFoundError(prefix)
    return matches[0]


def _passing(ledger: Ledger):
    for spec_id, r in sorted(ledger.results.items()):
        if r.status is Status.PASS and (r.metrics or r.control_metrics):
            yield spec_id, {"metrics": r.metrics, "control_metrics": r.control_metrics}


def _control(seed: int) -> dict:
    ns: dict = {}
    exec(compile(CONTROL_SRC, "<pre-fix T0.09 gate>", "exec"), ns)
    # Source passed explicitly — inspect cannot recover it for exec'd code, and
    # a detector that reads nothing reports a clean bill of health.
    return _scan([("T0.09_prefix", ns["_check"], CONTROL_SRC,
                   dict(CONTROL_M), dict(CONTROL_C))])


def _check(m: dict, c: dict) -> bool:
    # The detectors must come up clean on the real ladder...
    ladder_clean = (m["disarmed_conjunct_keys"] == 0
                    and m["precedence_hazards"] == 0
                    and m["unevaluable_gates"] == 0
                    and m["unreadable_gates"] == 0)
    # ...and BOTH must fire on the known-bad gate, or a clean scan is
    # indistinguishable from a scan that does not run. The pre-fix T0.09 check
    # has exactly three unreachable assertions (ok, cuda_available,
    # matmul_finite) and one precedence hazard.
    control_caught = (c["disarmed_conjunct_keys"] >= 3
                      and c["precedence_hazards"] >= 1
                      and c["gates_scanned"] == 1
                      and c["unreadable_gates"] == 0)
    # A scan of nothing is not a clean scan.
    scanned_enough = m["gates_scanned"] >= 30
    return ladder_clean and control_caught and scanned_enough


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.13"], _experiment, _check, control_fn=_control, ledger=ledger)
