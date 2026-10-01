"""Worst-seed gate audit — arm (c) of `aggregate-hides-worst-seed`.

THE DEFECT THIS AUDITS (routed 2026-08-30, ruled 2026-09-30, Review DAILY):
`protocol._aggregate` means every numeric metric across seeds before `_check`
runs once, so a metric whose NAME and PURPOSE are "the worst X" — a per-seed
min/max/len — is silently gated on the MEAN of the per-seed worst cases.
Seeds with 2, 6 and 10 informative lives average to a healthy 6 and clear a
gate no seed clears.

WHAT THIS MODULE IS, PER THE RULING ("(c) THEN (b); (a) IS REFUSED"):
arm (c) — the static audit that PRICES arm (b) (the recorder refusing to
flatten worst-case keys). It is the 2026-09-12 sweep (90th audit B2, attached
to the row) turned from a one-off into a standing instrument and widened from
multi-seed PASS rows to every registered spec. Two lanes:

  LANE A (FOLD-FLAGGED, prices arm (b)): a bare `_check` comparison on
    `m["<key>"]`/`c["<key>"]` where the SAME FILE computes that key with a
    min()/max()/len() fold. Detection is by the FOLD, never by a naming
    convention — a worst-case key renamed to dodge `*_min`/`*_worst_*` is
    still flagged, which is the loophole the ruling closes in advance
    (see `_selftest`'s rename-dodge fixture). `max(1, x)`-style clamps
    (a numeric-literal argument) are NOT folds.
  LANE B (RECORD-ADMITS-VIOLATION, the sweep standing): for every bare gate
    with a committed multi-seed row, test the exact ddof=0 extreme-value
    bound max|x_i - mu| <= sigma*sqrt(n-1) against the gate's bar. A pair is
    WRONG when the committed record ADMITS a seed on the failing side of the
    bar the gate claims to enforce.

THE TWO REFINEMENTS WITHOUT WHICH THE FIRST RUN LIES (both from the row):
  * `if <bad>: return Status.VOID/FAIL` INVERTS the comparison — the failing
    tail is the opposite one from the operator. A scan that reads the operator
    raw reports T3.01 `ref_min` and W0.DIAG `jit_delta_up` as false positives
    (the row's own warning; both verified CORRECT here).
  * integer arithmetic: LG.01's `retained_min_per_category` admits 19.9449
    numerically, but the metric is integer-valued and any admissible integer
    >= 19.9449 is >= 20 — SAFE, exactly as the 09-12 sweep classified it.
Plus `std == 0` (to fp precision) proves idiom 1 — the worst seed was folded
in BEFORE aggregation (T3.01's across-seed `min` in `_experiment`), so the
mean IS the worst case and the gate is CORRECT.

WHAT THIS MODULE IS NOT, SAID HERE BECAUSE TWO FREEZES ARE LIVE:
reporting-only, deliberately unfloored, exit 0 always from the audit itself
(`--selftest` is the only path that can exit nonzero). It records nothing in
any certificate, moves no threshold, and gates no run — the SEMANTIC bill of
arm (c) is zero by the ruling's own words. The ruling asks for "a standing
T0-family gate"; the STANDING FREEZE closes Tier 0 at 39 and D35 clause 2
forbids new checkers, so this ships the audit SUBSTANCE and leaves the
T0-spec packaging to the Review desk, with the tension disclosed in the
shipping commit (the `418f015` precedent). The row does NOT close on this:
step 2 (the recorder refusal, arm (b)) returns to the desk with Lane A's
count in hand.

KNOWN LIMITS, DISCLOSED RATHER THAN SILENT: comparisons whose bar does not
resolve to a number from the file's own constants (one import hop allowed)
are printed as UNRESOLVED, not dropped; derived gates (arithmetic on m[...]
that is not a lone subscript) are counted but not classified — a derived gate
that references `<key>_std` is recorded as PROTECTION for that key; specs
with no committed row classify as UNMEASURED in Lane A and are absent from
Lane B. Nothing is truncated: every flagged pair prints.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent / "tests"
LEDGER = Path(__file__).resolve().parent / "ledger.json"

#: std at or below this is floating-point zero: every seed equals the mean,
#: i.e. the worst seed was folded in before aggregation (idiom 1).
STD_ZERO = 1e-12
#: tolerance for deciding a metric is integer-valued from (n*mu, n*E[x^2]) —
#: both sums are exact integers for integer data; stored values are _round6ed.
INT_TOL_SUM, INT_TOL_SUMSQ = 0.02, 0.05

_FOLD_NAMES = {"min", "max", "len"}
_BAD_RETURNS = {"VOID", "FAIL"}


# ---------------------------------------------------------------- AST helpers

def _is_metric_sub(node: ast.AST):
    """`m["key"]` / `c["key"]` / `m.get("key"[, d])` -> (source, key) or None."""
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id in ("m", "c")):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return node.value.id, sl.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("m", "c") and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return node.func.value.id, node.args[0].value
    return None


def _metric_keys_in(node: ast.AST) -> set:
    """Every (source, key) metric reference anywhere under `node`."""
    out = set()
    for n in ast.walk(node):
        hit = _is_metric_sub(n)
        if hit:
            out.add(hit)
    return out


def _contains_fold(node: ast.AST) -> str | None:
    """Name of the first genuine fold call under `node`, else None.

    min/max with a numeric-literal argument is a CLAMP (`max(1, x)`), not a
    worst-case fold, and is excluded — PG.4's `panel_hits_dwell /
    max(1, dwell_steps)` must not read as a fold.
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        name = None
        if isinstance(n.func, ast.Name) and n.func.id in _FOLD_NAMES:
            name = n.func.id
        elif isinstance(n.func, ast.Attribute) and n.func.attr in ("min", "max"):
            name = n.func.attr          # np.min(x) / arr.min()
        if name is None:
            continue
        if name in ("min", "max") and len(n.args) > 1 and any(
                isinstance(a, ast.Constant) and isinstance(a.value, (int, float))
                for a in n.args):
            continue                    # clamp, not a fold
        return name
    return None


def folded_keys(tree: ast.AST) -> dict:
    """{metric key -> fold name} for every dict-literal entry or
    `something["key"] = ...` assignment whose value expression folds."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and v is not None):
                    f = _contains_fold(v)
                    if f:
                        out.setdefault(k.value, f)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and isinstance(t.slice.value, str)):
                f = _contains_fold(node.value)
                if f:
                    out.setdefault(t.slice.value, f)
    return out


# ------------------------------------------------------------ constant tables

def _module_consts(tree: ast.AST) -> dict:
    """Module-level `NAME = <number>` (two passes, so NAME = OTHER*2 resolves)."""
    consts: dict = {}
    for _ in range(2):
        for node in tree.body if hasattr(tree, "body") else []:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                v = _const_eval(node.value, consts)
                if v is not None:
                    consts[node.targets[0].id] = v
                elif isinstance(node.value, (ast.List, ast.Tuple)):
                    name = node.targets[0].id
                    consts[f"len:{name}"] = len(node.value.elts)
                    for i, el in enumerate(node.value.elts):
                        ev = _const_eval(el, consts)
                        if ev is not None:
                            consts[f"{name}[{i}]"] = ev
    return consts


def _const_eval(node: ast.AST, consts: dict):
    """Evaluate a bar expression against known numeric constants, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_eval(node.operand, consts)
        return None if v is None else -v
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)):
        return consts.get(f"len:{node.args[0].id}")
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)):
        return consts.get(f"{node.value.id}[{node.slice.value}]")
    if isinstance(node, ast.BinOp):
        l, r = _const_eval(node.left, consts), _const_eval(node.right, consts)
        if l is None or r is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
        except ZeroDivisionError:
            return None
    return None


def _imported_consts(tree: ast.AST, tests_dir: Path) -> dict:
    """One hop: `from .sibling import NAME` resolved from the sibling's own
    module-level constants (T3.01 gates on T2.03's MIN_CANARY_COLORS)."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        mod = node.module.rsplit(".", 1)[-1]
        sib = tests_dir / f"{mod}.py"
        if not sib.exists():
            continue
        try:
            sib_consts = _module_consts(ast.parse(sib.read_text()))
        except SyntaxError:
            continue
        for alias in node.names:
            tgt = alias.asname or alias.name
            if alias.name in sib_consts:
                out[tgt] = sib_consts[alias.name]
            for k, v in sib_consts.items():   # tuple elements / lengths
                if k.startswith(f"{alias.name}[") or k == f"len:{alias.name}":
                    out[k.replace(alias.name, tgt, 1)] = v
    return out


# ------------------------------------------------------------- gate extraction

_INVERT = {ast.Lt: ">=", ast.LtE: ">", ast.Gt: "<=", ast.GtE: "<",
           ast.Eq: "!=", ast.NotEq: "=="}
_STRAIGHT = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
             ast.Eq: "==", ast.NotEq: "!="}


def _returns_bad(body: list) -> bool:
    """Does this `if` body hand back a failing verdict (VOID/FAIL/False)?"""
    for stmt in body:
        if isinstance(stmt, ast.Return):
            v = stmt.value
            if isinstance(v, ast.Constant) and v.value is False:
                return True
            if (isinstance(v, ast.Attribute) and v.attr in _BAD_RETURNS):
                return True
            # `return _void(m, "...")` / `return _fail(...)` helper idiom
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                    and any(w in v.func.id.lower() for w in ("void", "fail"))):
                return True
    return False


def _expand(node: ast.AST, env: dict, depth: int = 0) -> ast.AST:
    """Substitute `_check`-local simple assignments (one Name = expr each),
    so `margin_floor > 0.0` is seen as the std-bounded expression it is."""
    if depth > 5:
        return node

    class Sub(ast.NodeTransformer):
        def visit_Name(self, n):            # noqa: N802
            if isinstance(n.ctx, ast.Load) and n.id in env:
                return _expand(env[n.id], env, depth + 1)
            return n
    return Sub().visit(__import__("copy").deepcopy(node))


class Gate:
    """One `_check` comparison, normalised to its PASS direction."""

    def __init__(self, source, key, pass_op, raw_op, bar, bar_text, lineno):
        self.source, self.key = source, key
        self.pass_op, self.raw_op = pass_op, raw_op
        self.bar, self.bar_text, self.lineno = bar, bar_text, lineno
        self.inverted = pass_op != raw_op

    def __repr__(self):
        src = "m" if self.source == "m" else "c"
        inv = "  [if-bad inverted]" if self.inverted else ""
        return f'{src}["{self.key}"] {self.pass_op} {self.bar_text}{inv}'


def extract_gates(tree: ast.AST, consts: dict):
    """(bare gates, protected keys, derived-unclassified count) from `_check`.

    Polarity: a comparison inside `if <cond>: return Status.VOID/FAIL/False`
    is a BAD condition — the pass region is its NEGATION. Everything else
    (`return a and b`, assignments returned as booleans) is read straight.
    A `not` toggles. This is the operator-inversion handling without which
    T3.01 and W0.DIAG read as false positives (the row's warning).
    """
    check = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_check"), None)
    if check is None:
        return [], {}, 0

    env = {}
    for stmt in check.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            env[stmt.targets[0].id] = stmt.value

    gates, protected, derived = [], {}, 0

    def handle_compare(cmp_node: ast.Compare, neg: bool):
        nonlocal derived
        if len(cmp_node.ops) != 1 or len(cmp_node.comparators) != 1:
            return
        if type(cmp_node.ops[0]) not in _STRAIGHT:
            return                      # is / in / not in — not a numeric gate
        sides = [_expand(cmp_node.left, env), _expand(cmp_node.comparators[0], env)]
        op = cmp_node.ops[0]
        for i, (side, other) in enumerate(((sides[0], sides[1]),
                                           (sides[1], sides[0]))):
            hit = _is_metric_sub(side)
            if hit is None:
                continue
            source, key = hit
            bar = _const_eval(other, consts)
            raw = _STRAIGHT[type(op)] if i == 0 else _flip(_STRAIGHT[type(op)])
            pas = (_INVERT[type(op)] if i == 0 else _flip(_INVERT[type(op)])) \
                if neg else raw
            gates.append(Gate(source, key, pas, raw, bar,
                              ast.unparse(other), cmp_node.lineno))
            return
        # no lone-subscript side: derived. Std references = protection.
        refs = _metric_keys_in(cmp_node.left) | {
            k for c in cmp_node.comparators for k in _metric_keys_in(c)}
        for ex in sides:
            refs |= _metric_keys_in(ex)
        plain = {(s, k) for s, k in refs if not k.endswith("_std")}
        stds = {(s, k[:-4]) for s, k in refs if k.endswith("_std")}
        for s, k in plain & stds:
            protected.setdefault((s, k), []).append(cmp_node.lineno)
        if plain and not (plain & stds):
            derived += 1

    def walk_expr(node: ast.AST, neg: bool):
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                walk_expr(v, neg)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            walk_expr(node.operand, not neg)
        elif isinstance(node, ast.Compare):
            handle_compare(node, neg)
        else:
            for child in ast.iter_child_nodes(node):
                walk_expr(child, neg)

    def walk_stmt(stmt: ast.stmt):
        if isinstance(stmt, ast.If):
            walk_expr(stmt.test, _returns_bad(stmt.body))
            for s in stmt.body + stmt.orelse:
                walk_stmt(s)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            walk_expr(stmt.value, False)
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt):
                    walk_stmt(child)
                elif isinstance(child, ast.expr):
                    pass            # bare expressions carry no verdict
    for stmt in check.body:
        walk_stmt(stmt)
    return gates, protected, derived


def _flip(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<=",
            "==": "==", "!=": "!="}[op]


# ------------------------------------------------------------- classification

def _integer_valued(mu: float, sigma: float, n: int) -> bool:
    s1, s2 = mu * n, n * (sigma * sigma + mu * mu)
    return (abs(s1 - round(s1)) < INT_TOL_SUM
            and abs(s2 - round(s2)) < INT_TOL_SUMSQ)


def classify(gate: Gate, mu, sigma, n):
    """(verdict, detail) for one bare gate against its committed row.

    verdicts: CORRECT (the record cannot contain a seed on the failing side),
    WRONG (the gate reads GREEN on the mean while the record admits a seed on
    the failing side — the hidden-seed signature), FIRED (the mean itself
    fails: the gate caught this record, nothing is hidden — a red row is not
    a worst-seed defect), UNMEASURED, UNRESOLVED (bar irreducible).
    """
    if gate.bar is None:
        return "UNRESOLVED", f"bar `{gate.bar_text}` is not a resolvable constant"
    if mu is not None and sigma is None and n == 1:
        return "CORRECT", ("single-seed row: _aggregate flattens nothing, the "
                           "gate reads the true value")
    if mu is None or sigma is None:
        return "UNMEASURED", "no committed row carries this key with a std"
    mean_passes = _passes(mu, gate.pass_op, gate.bar)
    if sigma <= STD_ZERO:
        if mean_passes:
            return "CORRECT", ("std==0: every seed equals the mean — the worst "
                               "seed was folded in before aggregation")
        return "FIRED", (f"the gate fired on this record (mu {mu:.6g} fails "
                         f"`{gate.pass_op} {gate.bar}` outright, std==0)")
    if not mean_passes:
        return "FIRED", (f"the gate fired on this record (mu {mu:.6g} fails "
                         f"`{gate.pass_op} {gate.bar}` outright)")
    if n is None or n < 2:
        return "UNMEASURED", "seed count unknown"
    dev = sigma * math.sqrt(n - 1)
    if gate.pass_op in (">=", ">"):
        worst = mu - dev
        viol = worst < gate.bar if gate.pass_op == ">=" else worst <= gate.bar
        if viol and _integer_valued(mu, sigma, n):
            w = math.ceil(worst - 1e-9)
            viol = w < gate.bar if gate.pass_op == ">=" else w <= gate.bar
            if not viol:
                return "CORRECT", (f"worst admissible {worst:.6g} but the metric "
                                   f"is integer-valued: ceil {w} clears {gate.bar}")
    elif gate.pass_op in ("<=", "<"):
        worst = mu + dev
        viol = worst > gate.bar if gate.pass_op == "<=" else worst >= gate.bar
        if viol and _integer_valued(mu, sigma, n):
            w = math.floor(worst + 1e-9)
            viol = w > gate.bar if gate.pass_op == "<=" else w >= gate.bar
            if not viol:
                return "CORRECT", (f"worst admissible {worst:.6g} but the metric "
                                   f"is integer-valued: floor {w} clears {gate.bar}")
    elif gate.pass_op == "==":
        return ("WRONG", "== gate on a key with seed spread: the mean can equal "
                         "the bar while no seed does")
    else:                                   # "!="
        worst_lo, worst_hi = mu - dev, mu + dev
        viol = worst_lo <= gate.bar <= worst_hi
        worst = gate.bar if viol else mu
    if viol:
        return "WRONG", (f"record admits a seed at {worst:.6g} on the failing "
                         f"side of `{gate.pass_op} {gate.bar}` "
                         f"(mu {mu:.6g}, sigma {sigma:.6g}, bound sigma*sqrt({n - 1}))")
    return "CORRECT", (f"worst admissible seed {worst:.6g} still satisfies "
                       f"`{gate.pass_op} {gate.bar}`")


def _passes(x, op, bar):
    return {"<": x < bar, "<=": x <= bar, ">": x > bar, ">=": x >= bar,
            "==": x == bar, "!=": x != bar}[op]


# ------------------------------------------------------------------ the audit

def _registered_files():
    """[(spec_id, path)] for every registered spec with an implementation."""
    from .registry import LADDER
    from .protocol import module_path_for
    out = []
    for spec in LADDER:
        try:
            p = module_path_for(spec.id)
        except RuntimeError:
            p = None
        if p is not None:
            out.append((spec.id, p))
    return out


def audit(ledger_path: Path = LEDGER):
    """Run both lanes over every registered spec. Returns a dict of findings."""
    results = json.loads(ledger_path.read_text())["results"] \
        if ledger_path.exists() else {}
    fold_rows, bound_rows, skipped = [], [], []
    for sid, path in _registered_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            skipped.append((sid, f"unparseable: {e}"))
            continue
        consts = _module_consts(tree)
        consts.update(_imported_consts(tree, path.parent))
        folds = folded_keys(tree)
        gates, protected, derived = extract_gates(tree, consts)

        row = results.get(sid) or {}
        mrow, crow = row.get("metrics") or {}, row.get("control_metrics") or {}
        n = len(row.get("seeds") or []) or None

        for g in gates:
            vals = mrow if g.source == "m" else crow
            mu, sigma = vals.get(g.key), vals.get(g.key + "_std")
            verdict, detail = classify(g, mu, sigma, n)
            sibling = (g.source, g.key) in protected
            entry = {"spec": sid, "gate": g, "fold": folds.get(g.key),
                     "verdict": verdict, "detail": detail,
                     "partially_protected": sibling and verdict == "WRONG",
                     "derived_gates": derived}
            if g.key in folds:
                fold_rows.append(entry)
            if verdict == "WRONG":
                bound_rows.append(entry)
    return {"fold": fold_rows, "bound": bound_rows, "skipped": skipped}


def render(res: dict) -> str:
    out = []
    w = out.append
    wrong_first = sorted(res["fold"],
                         key=lambda e: {"WRONG": 0, "UNRESOLVED": 1,
                                        "UNMEASURED": 2, "FIRED": 3,
                                        "CORRECT": 4}[e["verdict"]])
    fold_wrong = [e for e in res["fold"] if e["verdict"] == "WRONG"]
    bound_specs = sorted({e["spec"] for e in res["bound"]})
    w("WORST-SEED GATE AUDIT — arm (c) of `aggregate-hides-worst-seed` "
      "(ruled 2026-09-30: (c) prices (b); (a) refused)")
    w("  A bare `_check` gate on an _aggregate-meaned key cannot see its worst "
      "seed. Reporting-only; nothing here gates a run.")
    w("")
    w(f"LANE B — RECORD ADMITS A VIOLATING SEED ({len(res['bound'])} gate(s), "
      f"{len(bound_specs)} spec(s)) — the 09-12 sweep, standing:")
    if not res["bound"]:
        w("  none — no committed row admits a seed on the failing side of its bar")
    for e in sorted(res["bound"], key=lambda e: e["spec"]):
        g = e["gate"]
        pp = "  [std-bounded sibling conjunct exists: partially protected]" \
            if e["partially_protected"] else ""
        w(f"  {e['spec']:<8} {g!r:<52} WRONG: {e['detail']}{pp}")
    w("")
    w(f"LANE A — FOLD-FLAGGED, the set that prices arm (b) "
      f"({len(res['fold'])} pair(s), {len(fold_wrong)} WRONG):")
    if not res["fold"]:
        w("  none")
    for e in wrong_first:
        g = e["gate"]
        w(f"  {e['spec']:<8} {g!r:<52} fold={e['fold']:<4} "
          f"{e['verdict']}: {e['detail']}")
    for sid, why in res["skipped"]:
        w(f"  SKIPPED {sid}: {why}")
    w("")
    w(f"summary: fold-flagged {len(res['fold'])} pair(s) "
      f"({len(fold_wrong)} WRONG — arm (b)'s breakage price), "
      f"record-admits-violation {len(res['bound'])} gate(s) across "
      f"{len(bound_specs)} spec(s): {', '.join(bound_specs) or 'none'}")
    return "\n".join(out) + "\n"


# -------------------------------------------------------------------- selftest

_FIXTURE_RENAME_DODGE = '''
BAR = 0.5
def _experiment(seed):
    xs = [1.0, 2.0]
    return {"blorp": min(xs)}          # worst-case fold, innocuous name
def _check(m, c):
    return m["blorp"] >= BAR
'''

_FIXTURE_NAME_ONLY = '''
BAR = 0.5
def _experiment(seed):
    return {"foo_min": 0.9}            # convention name, NO fold
def _check(m, c):
    return m["foo_min"] >= BAR
'''

_FIXTURE_CLAMP = '''
BAR = 0.5
def _experiment(seed):
    a, b = 3.0, 4.0
    return {"rate": a / max(1, b)}     # clamp, not a fold
def _check(m, c):
    return m["rate"] >= BAR
'''

_FIXTURE_INVERTED = '''
REF_FLOOR = 0.38
class Status:
    VOID = "VOID"
def _experiment(seed):
    rows = [0.45, 0.45]
    return {"ref_min": min(rows)}
def _check(m, c):
    if m["ref_min"] < REF_FLOOR:
        return Status.VOID
    return True
'''


def _selftest() -> int:
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # 1. rename-dodge: fold detection flags a worst-case key however named.
    tree = ast.parse(_FIXTURE_RENAME_DODGE)
    folds = folded_keys(tree)
    gates, _, _ = extract_gates(tree, _module_consts(tree))
    check("rename-dodge: fold found", folds.get("blorp") == "min")
    check("rename-dodge: gate found",
          any(g.key == "blorp" and g.pass_op == ">=" for g in gates))

    # 2. naming convention alone is NOT a flag (detection by the fold).
    check("name-only: not fold-flagged",
          "foo_min" not in folded_keys(ast.parse(_FIXTURE_NAME_ONLY)))

    # 3. clamps are not folds.
    check("clamp: not fold-flagged",
          "rate" not in folded_keys(ast.parse(_FIXTURE_CLAMP)))

    # 4. `if <bad>: VOID` inversion: pass direction is the NEGATION, and the
    #    classification flips from false-positive WRONG to CORRECT.
    tree = ast.parse(_FIXTURE_INVERTED)
    gates, _, _ = extract_gates(tree, _module_consts(tree))
    g = next(g for g in gates if g.key == "ref_min")
    check("inverted: pass op is >=", g.pass_op == ">=" and g.raw_op == "<")
    v_fixed, _ = classify(g, 0.4467, 0.0, 3)
    naive = Gate(g.source, g.key, g.raw_op, g.raw_op, g.bar, g.bar_text, 0)
    v_naive, _ = classify(naive, 0.4467, 0.0, 3)
    check("inverted: fixed reads CORRECT", v_fixed == "CORRECT")
    # naive direction reads the gate as "fired" on a PASS certificate — the
    # contradiction that made T3.01/W0.DIAG false positives in a raw scan
    check("inverted: naive direction misreads", v_naive != "CORRECT")

    # 5. integer refinement: LG.01's exact numbers (admits 19.9449, but any
    #    admissible INTEGER >= 19.9449 is >= 20).
    g_int = Gate("m", "retained_min_per_category", ">=", ">=", 20, "RETAIN_MIN", 0)
    v, d = classify(g_int, 23.0, 2.160247, 3)
    check("integer: LG.01 CORRECT", v == "CORRECT" and "integer" in d)
    v, _ = classify(Gate("m", "k", ">=", ">=", 0.25, "B", 0), 0.370367, 0.0944234, 3)
    check("float: ME.10 WRONG", v == "WRONG")

    # 6. the 09-12 reference classification, reproduced on the live tree +
    #    ledger (the row's table is the reference; a drifted row shows here).
    res = audit()
    by_pair = {(e["spec"], e["gate"].key): e["verdict"]
               for e in res["fold"] + res["bound"]}
    wrong = {(e["spec"], e["gate"].key) for e in res["bound"]}
    for pair in [("PG.4", "icm_dwell_share"), ("PG.4", "dwell_margin"),
                 ("PG.4", "panel_reward_ratio"),
                 ("PG.4", "rays_on_panel_while_dwelling"),
                 ("ME.10", "skill_gain"), ("PS.02", "shuffled_r2"),
                 ("T2.08", "coverage_margin")]:
        check(f"reference WRONG: {pair}", pair in wrong)
    for spec, key in [("LG.01", "retained_min_per_category"),
                      ("T3.01", "ref_min"), ("W0.DIAG", "jit_delta_up")]:
        check(f"reference safe: {(spec, key)}", (spec, key) not in wrong)
    t208 = [e for e in res["bound"]
            if (e["spec"], e["gate"].key) == ("T2.08", "coverage_margin")]
    check("T2.08 partially protected",
          bool(t208) and t208[0]["partially_protected"])

    if failures:
        print("SELFTEST FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("selftest: all fixtures pass (rename-dodge, name-only, clamp, "
          "if-bad inversion, integer refinement, 09-12 reference classification)")
    return 0


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return _selftest()
    print(render(audit()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
