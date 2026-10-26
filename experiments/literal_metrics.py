"""METRIC RECORDED AS A SOURCE LITERAL — the decorative-metric reading.

ADOPTED by the Review's 2026-09-29 ruling on `d27-screen-measures-95-percent-
false`, SEPARATELY from `D27` and EXPLICITLY NOT AS `D27`'s SCREEN. Quote this
reading as answering `D27` and you are misquoting it: `D27`'s class is *"the
run measured the quantity that would have indicted it and then did not look at
it"*; this module's class is a key that was never measured at all — its
recorded value is a LITERAL in the source. The scar that earned it is
`T0.06 steps_ok`: the literal `5` written straight into the metrics dict,
never computed, never read, standing on a PASS certificate — the single true
positive of the deleted `metric_recorded_but_unread` screen (19/20 false,
deleted by the same ruling; the screen and its history are in git at the
commit this module arrived in, and in `docs/DECISIONS_RESOLVED.md` under
`D27`).

WHY IT HAS NO FALSE POSITIVES BY CONSTRUCTION (the property the ruling adopted
it for): it claims only what the AST shows — every site that records this key
assigns it a bare constant. It does not claim the key SHOULD have been
computed, does not pair bars, does not follow dataflow. A key with even one
computed recording site anywhere in the module is never flagged.

TWO SUB-CLASSES, printed apart because they indict different things:
  * UNREAD literal — decorative: a constant wearing a measurement's name.
  * READ literal — a `_check` conjunct gates on a constant, which can never
    fail at that site; that is `T0.13`'s subject (decorative GATES) arriving
    from the metrics side, printed here so the two readers can be joined by a
    human.

REPORTING-ONLY and UNFLOORED: it refuses nothing, stales nothing, reddens
nothing, and no rule may be armed on its output without `run blast-radius`.

Conservatism, stated — three rules, each bought by a measured near-miss on the
day this shipped: (1) recording sites are read ONLY inside `_experiment` and
`_control` bodies — the first module-wide draft flagged `VO.02 cic=0.05`, whose
literal sites were its own self-test FIXTURE dicts while the real recording is
the dynamic `out[f"{arm}_cic"]`; (2) a key assigned DIFFERENT literals at
different sites is computed by control flow (the branch choice is the
measurement) and is never flagged — the first draft classed `LT.01 c2_branch`'s
branch-selected strings as a gated constant, which it is not; (3) a key
recorded only through dynamic construction (f-string keys, helper-built dicts)
has no literal site to read and is NEVER flagged — unknown is not zero.
Recording sites recognised are dict literals (`{"k": <const>}`) and constant
subscript stores (`out["k"] = <const>`).
"""

from __future__ import annotations

import ast
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Same attribution set the deleted screen declared; recorded for provenance,
#: never a quantity a claim gates on. Named so a reader can contest the list.
PROVENANCE = frozenset({
    "backend", "gpu", "device", "torch", "hardware", "capability",
    "landed_on", "message", "duration_s", "wall_s",
    "gpu_job_id", "gpu_repo_sha", "commit", "impl_sha", "ran_at",
})


def _check_fn(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_check":
            return n
    return None


def _recording_sites(tree: ast.AST) -> Dict[str, List[ast.AST]]:
    """key -> every value expression that records it, read ONLY inside
    `_experiment`/`_control` bodies: dict-literal entries and constant-key
    subscript stores (`out["k"] = expr`). Module-wide reading flagged a
    self-test fixture dict as a recording site on day one (VO.02)."""
    out: Dict[str, List[ast.AST]] = {}
    roots = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)
             and n.name in ("_experiment", "_control")]
    for root in roots:
        for n in ast.walk(root):
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out.setdefault(k.value, []).append(v)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.slice, ast.Constant)
                            and isinstance(t.slice.value, str)):
                        out.setdefault(t.slice.value, []).append(n.value)
    return out


def _is_bare_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    # -1.0 parses as UnaryOp(USub, Constant); it is still a literal
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)):
        return True
    return False


def _minted_std(metrics: set) -> set:
    return {k for k in metrics if k.endswith("_std") and k[:-4] in metrics}


def scan_spec(row: dict, path: pathlib.Path) -> Dict[str, Any]:
    """One certificate: which of its recorded keys are literals at every
    recording site. Returns {"literal_unread", "literal_read", "n_considered"}."""
    tree = ast.parse(path.read_text())
    sites = _recording_sites(tree)
    ch = _check_fn(tree)
    read_names = ({n.value for n in ast.walk(ch)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                  if ch is not None else set())
    metrics = set(row.get("metrics") or {})
    authored = metrics - _minted_std(metrics) - PROVENANCE
    lit_unread: List[Tuple[str, Any]] = []
    lit_read: List[Tuple[str, Any]] = []
    for k in sorted(authored):
        exprs = sites.get(k)
        if not exprs:
            continue                       # no literal site readable: never flag
        if all(_is_bare_constant(e) for e in exprs):
            vals = {repr(ast.literal_eval(e)) for e in exprs}
            if len(vals) > 1:
                continue    # branch-selected literals: the choice IS computed
            val = ast.literal_eval(exprs[0])
            (lit_read if k in read_names else lit_unread).append((k, val))
    return {"literal_unread": lit_unread, "literal_read": lit_read,
            "n_considered": len(authored)}


def scan(ledger: Any = None) -> Dict[str, Any]:
    """Every standing PASS row. Reporting-only; raises nothing on a finding."""
    from .protocol import module_path_for

    if ledger is None:
        results = json.loads(
            (REPO / "experiments" / "ledger.json").read_text())["results"]
    else:
        results = getattr(ledger, "results", None) or ledger.data["results"]

    unread: Dict[str, List[Tuple[str, Any]]] = {}
    read: Dict[str, List[Tuple[str, Any]]] = {}
    refused: List[str] = []
    scanned = 0
    for sid, row in sorted(results.items()):
        if row.get("status") != "PASS":
            continue
        try:
            path = module_path_for(sid)
        except Exception as exc:
            refused.append(f"{sid}: {type(exc).__name__}")
            continue
        if path is None:
            continue
        try:
            r = scan_spec(row, path)
        except SyntaxError as exc:
            refused.append(f"{sid}: SyntaxError {exc.lineno}")
            continue
        scanned += 1
        if r["literal_unread"]:
            unread[sid] = r["literal_unread"]
        if r["literal_read"]:
            read[sid] = r["literal_read"]
    return {"literal_unread": unread, "literal_read": read,
            "n_unread": sum(len(v) for v in unread.values()),
            "n_read": sum(len(v) for v in read.values()),
            "scanned": scanned, "refused": refused}


def render(indent: str = "  ", result: Optional[dict] = None) -> str:
    try:
        res = result or scan()
    except Exception as exc:                            # never break `status`
        return (f"{indent}LITERAL METRICS — scan refused "
                f"({type(exc).__name__}: {exc}). An instrument going quiet is "
                f"a fault,\n{indent}  not a quiet day.\n")
    lines = [
        f"{indent}METRIC RECORDED AS A SOURCE LITERAL — {res['n_unread']} "
        f"decorative metric(s) on {len(res['literal_unread'])} certificate(s): "
        f"every recording site",
        f"{indent}  assigns a bare constant, and no conjunct of `_check` reads "
        f"the key. Mechanical, no false",
        f"{indent}  positives by construction (adopted by the 2026-09-29 "
        f"ruling on `d27-screen-measures-95-percent-false`;",
        f"{indent}  NOT the D27 screen and never quotable as answering D27 — "
        f"that answer is 0, recorded under D27).",
        f"{indent}  REPORTING-ONLY and UNFLOORED: refuses nothing, stales "
        f"nothing, reddens nothing.",
    ]
    for sid, pairs in sorted(res["literal_unread"].items()):
        shown = ", ".join(f"{k}={v!r}" for k, v in pairs[:4])
        lines.append(f"{indent}    {sid:10s} {len(pairs):3d}  {shown}"
                     + (" ..." if len(pairs) > 4 else ""))
    if res["literal_read"]:
        n_pairs = res["n_read"]
        lines.append(
            f"{indent}  and {n_pairs} GATED literal(s) on "
            f"{len(res['literal_read'])} certificate(s) — a `_check` conjunct "
            f"reads a key that is a bare")
        lines.append(
            f"{indent}  constant at every recording site (T0.13's subject "
            f"from the metrics side):")
        for sid, pairs in sorted(res["literal_read"].items()):
            shown = ", ".join(f"{k}={v!r}" for k, v in pairs[:4])
            lines.append(f"{indent}    {sid:10s} {len(pairs):3d}  {shown}"
                         + (" ..." if len(pairs) > 4 else ""))
    if res["refused"]:
        lines.append(f"{indent}  !! {len(res['refused'])} spec(s) refused the "
                     f"scan: {', '.join(res['refused'][:4])}")
    lines.append(f"{indent}  scanned: {res['scanned']} standing PASS row(s); "
                 f"a key with no literal recording site is never flagged — "
                 f"unknown is not zero.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# known-answer fixture — per-property, authored before first render, the
# AT.01 idiom: the battery can fail, so the reading is alive.
# --------------------------------------------------------------------------

_FIX_SRC = '''
def _experiment(seed):
    a = compute()
    out = {"computed": a, "decorative": 5, "gated_literal": 1.0,
           "negative_lit": -1.0, "mixed": 3}
    out["stored_lit"] = 7
    out["mixed"] = a
    if a > 0:
        out["branchy"] = "high"
    else:
        out["branchy"] = "low"
    return out

def _selftest():
    fixture = {"decorative": 99, "fixture_only": 5}
    return fixture

def _check(m, c):
    return m["computed"] > 0 and m["gated_literal"] == 1.0
'''


def _check() -> List[str]:
    """P1 a computed key is never flagged.
    P2 an unread literal IS flagged as decorative.
    P3 a read literal lands in the GATED class, not the decorative one.
    P4 a key with one computed site among literal sites is never flagged.
    P5 provenance and minted `_std` keys are never flagged.
    P6 a constant subscript store is a recognised recording site.
    P7 a negated literal is still a literal.
    P8 render() carries the not-D27 disclaimer — the one property the
       adoption ruling binds.
    P9 a literal in a dict OUTSIDE `_experiment`/`_control` (a self-test
       fixture) is not a recording site — the VO.02 near-miss, asserted.
    P10 a key assigned DIFFERENT literals in different branches is computed
        by control flow and never flagged — the LT.01 near-miss, asserted."""
    import tempfile
    fails: List[str] = []
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_FIX_SRC)
        p = pathlib.Path(f.name)
    try:
        mets = {"computed": 2.0, "decorative": 5, "gated_literal": 1.0,
                "negative_lit": -1.0, "mixed": 3.0, "stored_lit": 7,
                "branchy": "high", "fixture_only": 5,
                "computed_std": 0.1, "backend": "kaggle"}
        r = scan_spec({"metrics": mets}, p)
        unread = dict(r["literal_unread"])
        read = dict(r["literal_read"])
        if "computed" in unread or "computed" in read:
            fails.append("P1: a computed key was flagged")
        if "decorative" not in unread:
            fails.append("P2: the unread literal was NOT flagged — the "
                         "detector cannot see its own known positive")
        if "gated_literal" not in read:
            fails.append("P3: the read literal missed the GATED class")
        if "gated_literal" in unread:
            fails.append("P3: the read literal was classed decorative")
        if "mixed" in unread or "mixed" in read:
            fails.append("P4: a key with a computed site was flagged")
        if "computed_std" in unread or "backend" in unread:
            fails.append("P5: a minted _std or provenance key was flagged")
        if "stored_lit" not in unread:
            fails.append("P6: a constant subscript store was not a site")
        if "negative_lit" not in unread:
            fails.append("P7: a negated literal was not a literal")
        if "fixture_only" in unread or "fixture_only" in read:
            fails.append("P9: a self-test fixture dict was read as a "
                         "recording site — the VO.02 near-miss")
        if unread.get("decorative") != 5:
            fails.append("P9: the fixture's foreign literal for `decorative` "
                         "leaked into the flagged value")
        if "branchy" in unread or "branchy" in read:
            fails.append("P10: branch-selected literals were flagged — the "
                         "LT.01 near-miss")
    finally:
        p.unlink()

    fake = {"literal_unread": {"A.1": [("x", 5)]}, "literal_read": {},
            "n_unread": 1, "n_read": 0, "scanned": 3, "refused": []}
    txt = render(result=fake)
    if "NOT the D27 screen" not in txt:
        fails.append("P8: render() lost the not-D27 disclaimer — the one "
                     "property the adoption ruling binds")

    if fails:
        raise AssertionError("literal_metrics fixture RED:\n  "
                             + "\n  ".join(fails))
    return fails


def measure() -> None:
    """`python -m experiments.literal_metrics` — the block on its own."""
    _check()
    print(render(), end="")


if __name__ == "__main__":
    measure()
