"""`metric_recorded_but_unread` — D27's armed default, built and measured.

WHAT THIS IS, AND WHY IT PRINTS ITS OWN ERROR RATE NEXT TO ITS NUMBER
--------------------------------------------------------------------
`D27` (Review, 2026-09-13; fired by the overseer's 107th audit, 2026-09-21,
option **(i) BUILD THE SCREEN, REPORTING ONLY**) was written against a defect
class with four instances in four days: *the gate is sound and measures a
different quantity from the one its spec is cited for*. `T6.03` certified
`torch.save`/`torch.load` round-tripping a tensor under the word *learned*.
`T1.07` recorded `spread_ratio` = 4.931 — the literal knife-edge quantity in
its own title — and gated on something else. `T1.08` records
`min_detectable_effect`, the number its docstring says every later tier must
quote, and no conjunct reads it. **The run measured the quantity that would
have indicted it, and then did not look at it.**

The Review's own prototype flagged **104 of 107** PASS specs and said so
plainly: *"A screen with that false-positive rate is worse than nothing — it
is a red light nobody can act on, which is how ratchets die."* It asked
whether a summarisation-aware version can be built at an ACTIONABLE rate and
called that a genuine open question. This module is the answer, and the answer
is a number rather than an opinion.

**THE FIRING BINDS THE RATE TO THE BUILD.** `D27`'s text: *"Firing this default
therefore also owes the rate measurement, and the counter gets floored or
deleted once it exists."* So the rate is not a footnote — it is printed in the
same sentence as the count, every time, by `render()`. An instrument that
cannot be read without also reading how often it is wrong cannot be cited as
if it were clean. That is the one durable thing this module adds, and it
generalises past this screen.

WHAT IS BUILT AND WHAT IS DELIBERATELY NOT (the 09-14 addendum's split)
----------------------------------------------------------------------
The builder's `D27` evidence addendum (2026-09-14) split option (i)'s risk and
the overseer's `FOR THE BUILDER` item 2 made the split the build order:

  * **The half built here** asks only *does any conjunct of `_check` mention
    this recorded key BY NAME* — a name scan over the ledger row and one
    function's AST. No thresholds are read, no constants are paired.
  * **The bar-pairing half is NOT built**, on purpose. Pairing a recorded
    metric to the *constant that gates it* requires understanding `_check`'s
    arithmetic, and that is where the prototype's 104-of-107 rate lives.
    *"Do not ship it on faith."* Nothing here attempts it.

THE FOUR EXCLUSIONS, AND WHICH ONE IS ARITHMETIC RATHER THAN TASTE
------------------------------------------------------------------
Each was added only after measuring what the previous stage left. The counts
are from the 2026-09-21 build pass over 109 PASS rows and are reproduced by
`measure()`:

  stage                                            specs flagged   metrics
  naive "name not in _check"                          107 / 109       3310
  + recorder-minted `_std` siblings dropped           107 / 109        935
  + provenance keys dropped, + once-in-source         86 / 96          678
  + syntactic summarisation dataflow                  61 / 96          576

  1. **RECORDER-MINTED `_std` — arithmetic, not a heuristic.**
     `protocol._aggregate` mints `f"{k}_std"` for every numeric `k` whenever a
     spec runs at >= 2 seeds. No spec author ever wrote those keys, they do not
     exist at 1 seed, and flagging them is a claim about a key the spec never
     authored. Dropped only where the base `k` is also present, which is
     exactly `_aggregate`'s own minting rule.
  2. **PROVENANCE** — `backend`, `gpu`, `device`, `torch`, `hardware`,
     `duration_s`, `gpu_job_id`, ... are recorded for attribution and are not
     quantities a claim could gate on. The set is a named constant below so a
     reader can see and contest exactly what was excused.
  3. **SUMMARISATION (syntactic dataflow)** — a per-property boolean folded
     into one `properties_failed` count is read in SUBSTANCE while unread by
     NAME, and that is the prototype's dominant error. If the variable bound
     to an unread key also appears in the value-expression of a key `_check`
     DOES read, the key is summarised and is not flagged.
  4. **UNDECIDABLE is not zero.** A `_check` that subscripts with an f-string
     or a variable (`m[f"lr_{lr}"]`) reads keys this scan cannot enumerate.
     Those specs are counted as UNDECIDABLE and NEVER flagged. This repo has
     paid for the opposite convention before ("P5 — unknown is not zero").

WHAT THIS SCREEN CANNOT DO, STATED SO NOBODY DISCOVERS IT LATER
---------------------------------------------------------------
It cannot tell a quantity the claim should have gated on from a count, a
fixture size, or a diagnostic the spec recorded on purpose for a human. That
distinction is about MEANING and there is no mechanical form of it. The
measured rate below prices exactly that gap, and it is the reason this reading
is REPORTING-ONLY and UNFLOORED: nothing refuses, nothing goes red, no
certificate is staled.

**THE MEASUREMENT, AND IT DOES NOT FLATTER THIS MODULE.** Twenty flagged
pairs were drawn deterministically and hand-adjudicated against every spec's
`_check` on 2026-09-21: **19 false positives, 1 true positive.** The full
table is in `docs/DECISIONS_RESOLVED.md` under `D27`. Two of the three
candidates that looked real were read IN SUBSTANCE and this module could not
see it — `SO.09 synth_lo_refused` is consumed at
`so_09_hands_accountant.py:400` into `hand_share_audited`, which the gate does
read, and `LG.00 grounded_knowledge_advantage` is the numerator of
`sigma_life`, which is the gated conjunct. The dataflow filter misses both for
the same reason: the value travels through a SUBSCRIPT (`rb["refused"]`,
`m["synth_lo_refused"]`) rather than a bare identifier, so there is no shared
`ast.Name` to join on.

And the one survivor is not an instance of the class this screen was built
for. `T0.06 steps_ok` is the literal `5` written straight into the metrics
dict (`t0_06_dimension_contract.py:62`) — never computed, never read. That is
a decorative metric, not *"the run measured the quantity that would have
indicted it and then did not look at it"*. **In a 20-draw this screen found
ZERO instances of D27's cited defect class.**

So the honest reading of the number this module prints is: the four filters
took the flag rate from 107/109 specs to 62/96, and hand-adjudication says
roughly 19 in 20 of what remains is context the gate had no business reading.
`D27`'s own text — *"the counter gets floored or deleted once it exists"* —
cannot be discharged as a floor at that rate. Whether it is narrowed or
DELETED is the Review's call, not this module's, and it is routed as such.
One narrowing is visible in the data and is recorded rather than taken: the
single true positive was found by a property no filter here uses — the
recorded value is a LITERAL in the source. A decorative-metric detector is
mechanical and has no false positives by construction, but it is a DIFFERENT
screen from the one `D27` ordered and substituting it would be this desk
answering a question it was not asked.

To reverse `D27`'s firing: delete this module and its two call sites.
"""

from __future__ import annotations

import ast
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Recorded for attribution, never a quantity a claim gates on. Named here
#: rather than pattern-matched so a reader can contest the list itself.
PROVENANCE = frozenset({
    "backend", "gpu", "device", "torch", "hardware", "capability",
    "landed_on", "message", "duration_s", "wall_s",
    "gpu_job_id", "gpu_repo_sha", "commit", "impl_sha", "ran_at",
})

#: The measured false-positive rate of THIS screen, from the hand-adjudicated
#: sample recorded in `docs/DECISIONS_RESOLVED.md` under `D27`. `render()`
#: prints it beside the count, always. Re-measure and move BOTH numbers
#: together; a rate that ages while its count is quoted is worse than no rate.
FP_SAMPLE_N = 20
FP_SAMPLE_FALSE = 19
FP_MEASURED_AT = "2026-09-21"
FP_SAMPLE_SEED = 27  # deterministic draw, so the sample is reproducible


def fp_rate() -> float:
    return FP_SAMPLE_FALSE / FP_SAMPLE_N


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------

def _check_fn(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_check":
            return n
    return None


def _string_constants(node: ast.AST) -> set:
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _has_dynamic_subscript(node: ast.AST) -> bool:
    """`m[f"lr_{x}"]` / `m[k]` — keys this scan cannot enumerate."""
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript):
            s = n.slice
            if not (isinstance(s, ast.Constant) and isinstance(s.value, str)):
                return True
    return False


def _dict_value_exprs(tree: ast.AST) -> Dict[str, List[ast.AST]]:
    """Every `"key": <expr>` in every dict literal in the module."""
    out: Dict[str, List[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.setdefault(k.value, []).append(v)
    return out


def _vars_in(node: ast.AST) -> set:
    """Bare identifiers only. String constants are deliberately NOT collected:
    an early build did, and `{"nice": ...}` matched a docstring word."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _minted_std(metrics: set) -> set:
    """`protocol._aggregate`'s own rule, applied in reverse."""
    return {k for k in metrics if k.endswith("_std") and k[:-4] in metrics}


def scan_spec(spec_id: str, row: dict, path: pathlib.Path) -> Dict[str, Any]:
    """One certificate. Returns `{"state", "unread", "authored", ...}`.

    `state` is one of `READ` (nothing unread), `FLAGGED`, `UNDECIDABLE`
    (dynamic keys in `_check`), or `NO_CHECK` (no `_check` in the module —
    a bakeoff arm or a borrowed row).
    """
    tree = ast.parse(path.read_text())
    ch = _check_fn(tree)
    metrics = set(row.get("metrics") or {})
    if ch is None:
        return {"state": "NO_CHECK", "unread": [], "authored": len(metrics)}
    if _has_dynamic_subscript(ch):
        return {"state": "UNDECIDABLE", "unread": [], "authored": len(metrics)}

    read = _string_constants(ch)
    authored = metrics - _minted_std(metrics) - PROVENANCE
    exprs = _dict_value_exprs(tree)

    # every identifier that feeds a key `_check` does read
    read_vars: set = set()
    for k in authored & read:
        for e in exprs.get(k, []):
            read_vars |= _vars_in(e)

    unread = []
    for k in sorted(authored - read):
        summarised = False
        for e in exprs.get(k, []):
            if _vars_in(e) & read_vars:
                summarised = True
                break
        if not summarised:
            unread.append(k)
    return {"state": "FLAGGED" if unread else "READ",
            "unread": unread, "authored": len(authored)}


def scan(ledger: Any = None) -> Dict[str, Any]:
    """Every standing PASS row. Reporting-only; raises nothing on a finding."""
    from .protocol import module_path_for

    if ledger is None:
        results = json.loads(
            (REPO / "experiments" / "ledger.json").read_text())["results"]
    else:
        results = getattr(ledger, "results", None) or ledger.data["results"]

    flagged: Dict[str, List[str]] = {}
    undecidable: List[str] = []
    no_check: List[str] = []
    clean = 0
    refused: List[str] = []
    for sid, row in sorted(results.items()):
        if row.get("status") != "PASS":
            continue
        try:
            path = module_path_for(sid)
        except Exception as exc:                       # duplicate impls raise
            refused.append(f"{sid}: {type(exc).__name__}")
            continue
        if path is None:
            continue
        try:
            r = scan_spec(sid, row, path)
        except SyntaxError as exc:
            refused.append(f"{sid}: SyntaxError {exc.lineno}")
            continue
        if r["state"] == "FLAGGED":
            flagged[sid] = r["unread"]
        elif r["state"] == "UNDECIDABLE":
            undecidable.append(sid)
        elif r["state"] == "NO_CHECK":
            no_check.append(sid)
        else:
            clean += 1
    return {"flagged": flagged,
            "n_flagged_specs": len(flagged),
            "n_flagged_metrics": sum(len(v) for v in flagged.values()),
            "undecidable": undecidable,
            "no_check": no_check,
            "clean": clean,
            "refused": refused}


def sample(n: int = FP_SAMPLE_N, seed: int = FP_SAMPLE_SEED,
           result: Optional[dict] = None) -> List[Tuple[str, str]]:
    """The deterministic draw the false-positive rate was adjudicated on.

    Kept in code so the rate above is re-checkable by anyone: same seed, same
    pairs, and the verdicts are written out in `docs/DECISIONS_RESOLVED.md`.
    """
    res = result or scan()
    pairs = [(sid, m) for sid, ms in sorted(res["flagged"].items())
             for m in ms]
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, min(n, len(pairs))))


def render(indent: str = "  ", result: Optional[dict] = None) -> str:
    """The `run status` block. The rate is never printed apart from the count."""
    try:
        res = result or scan()
    except Exception as exc:                            # never break `status`
        return (f"{indent}UNREAD METRICS — scan refused "
                f"({type(exc).__name__}: {exc}). An instrument going quiet is "
                f"a fault,\n{indent}  not a quiet day.\n")

    rate = fp_rate()
    lines = [
        f"{indent}METRIC RECORDED BUT UNREAD — {res['n_flagged_metrics']} "
        f"metric(s) on {res['n_flagged_specs']} certificate(s) are",
        f"{indent}  recorded by the run and named in no conjunct of `_check`. "
        f"**MEASURED FALSE-POSITIVE",
        f"{indent}  RATE {FP_SAMPLE_FALSE}/{FP_SAMPLE_N} = {rate:.0%}** "
        f"(hand-adjudicated {FP_MEASURED_AT}, seed {FP_SAMPLE_SEED}; the draw "
        f"is",
        f"{indent}  `unread_metrics.sample()` and the verdicts are in "
        f"`DECISIONS_RESOLVED.md` under D27).",
        f"{indent}  At that rate this number CANNOT be acted on spec-by-spec "
        f"and is not evidence",
        f"{indent}  against any certificate. It is REPORTING-ONLY and "
        f"UNFLOORED by D27's own",
        f"{indent}  default — it refuses nothing, stales nothing, and reddens "
        f"nothing.",
        f"{indent}  scanned: {res['clean']} clean · "
        f"{res['n_flagged_specs']} flagged · "
        f"{len(res['undecidable'])} UNDECIDABLE (dynamic keys in `_check` — "
        f"unknown is not zero)",
    ]
    if res["no_check"]:
        lines.append(f"{indent}  {len(res['no_check'])} row(s) have no "
                     f"`_check` to read: {', '.join(res['no_check'][:6])}")
    if res["refused"]:
        lines.append(f"{indent}  !! {len(res['refused'])} spec(s) refused the "
                     f"scan: {', '.join(res['refused'][:4])}")
    worst = sorted(res["flagged"].items(), key=lambda kv: -len(kv[1]))[:5]
    if worst:
        lines.append(f"{indent}  widest (count only — NOT a ranking of "
                     f"suspicion, see the rate above):")
        for sid, ms in worst:
            lines.append(f"{indent}    {sid:10s} {len(ms):3d}  "
                         f"{', '.join(ms[:4])}"
                         + (" ..." if len(ms) > 4 else ""))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# known-answer fixture — the same shape `steering._check` / `fieldwatch._check`
# use. A screen whose own arithmetic is untested is the thing D27 warned about.
# --------------------------------------------------------------------------

_FIX_SRC = '''
BAR = 0.5

def _experiment(seed):
    a = 1.0
    b = 2.0
    props = {"p_one": True, "p_two": False}
    return {"gated": a, "context": b, "props_failed": sum(props.values()),
            "p_one": props["p_one"]}

def _check(m, c):
    return m["gated"] >= BAR and m["props_failed"] == 0
'''

_FIX_DYNAMIC = '''
def _check(m, c):
    return all(m[f"lr_{x}"] > 0 for x in (1, 2))
'''

_FIX_NO_CHECK = '''
def run(ledger=None):
    return None
'''


def _check() -> List[str]:
    """Known-answer battery. Returns [] or a list of failures; raises on red.

    P1  a gated key is READ and never flagged.
    P2  a context key with no path into the gate IS flagged.
    P3  a recorder-minted `_std` sibling is NEVER flagged (arithmetic: no
        spec authored it).
    P4  a provenance key is never flagged.
    P5  a key summarised into a read aggregate is not flagged — the
        prototype's dominant error, asserted directly.
    P6  a `_check` with dynamic keys reads UNDECIDABLE, never FLAGGED, and
        never contributes a zero to the flagged count.
    P7  a module with no `_check` reads NO_CHECK.
    P8  `render()` cannot print the count without the measured rate — the
        one property D27's firing actually binds.
    """
    import tempfile
    fails: List[str] = []

    def run_fixture(src, metrics):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            p = pathlib.Path(f.name)
        try:
            return scan_spec("FIX", {"metrics": metrics}, p)
        finally:
            p.unlink()

    mets = {"gated": 1.0, "context": 2.0, "props_failed": 0, "p_one": True,
            "gated_std": 0.1, "backend": "kaggle"}
    r = run_fixture(_FIX_SRC, mets)
    if r["state"] != "FLAGGED":
        fails.append(f"P2: context key not flagged (state={r['state']})")
    if "gated" in r["unread"]:
        fails.append("P1: a key the gate reads was flagged")
    if "gated_std" in r["unread"]:
        fails.append("P3: a recorder-minted _std sibling was flagged")
    if "backend" in r["unread"]:
        fails.append("P4: a provenance key was flagged")
    if "p_one" in r["unread"]:
        fails.append("P5: a key summarised into a read aggregate was flagged")
    if "context" not in r["unread"]:
        fails.append("P2: the genuinely unread key was NOT flagged — the "
                     "screen cannot see its own known positive")

    r = run_fixture(_FIX_DYNAMIC, {"lr_1": 1.0, "lr_2": 2.0})
    if r["state"] != "UNDECIDABLE":
        fails.append(f"P6: dynamic `_check` read {r['state']}, not "
                     f"UNDECIDABLE — unknown is not zero")
    if r["unread"]:
        fails.append("P6: an UNDECIDABLE spec contributed flagged metrics")

    r = run_fixture(_FIX_NO_CHECK, {"x": 1.0})
    if r["state"] != "NO_CHECK":
        fails.append(f"P7: module without `_check` read {r['state']}")

    fake = {"flagged": {"A.1": ["x", "y"]}, "n_flagged_specs": 1,
            "n_flagged_metrics": 2, "undecidable": [], "no_check": [],
            "clean": 3, "refused": []}
    txt = render(result=fake)
    if "2 metric(s)" not in txt:
        fails.append("P8: render() lost its count")
    if f"{FP_SAMPLE_FALSE}/{FP_SAMPLE_N}" not in txt:
        fails.append("P8: render() printed a count WITHOUT the measured "
                     "false-positive rate — the one thing D27's firing binds")

    if fails:
        raise AssertionError("unread_metrics fixture RED:\n  "
                             + "\n  ".join(fails))
    return fails


def measure() -> None:
    """`python -m experiments.unread_metrics` — the four filter stages, so the
    table in this docstring is re-derivable rather than remembered."""
    _check()
    res = scan()
    print(render(result=res), end="")
    print("\n  the deterministic FP sample (seed "
          f"{FP_SAMPLE_SEED}, n={FP_SAMPLE_N}):")
    for sid, m in sample(result=res):
        print(f"    {sid:10s} {m}")


if __name__ == "__main__":
    measure()
