"""T0.35 — an implementation dependency cannot go undeclared.

THE SCAR (78th audit, 2026-09-06, and it fired the same morning it was
measured). `impl_sha` answers "is this certificate about the code that
produced it" by hashing the test file plus every file the test DECLARES in
`IMPL_DEPS` — and declaring was voluntary. On 2026-09-06, 54 of 105 PASS
specs declared nothing at all, and 35 implemented specs imported a repo-root
module by name without declaring it. That morning `EpisodicMemory.py`'s
scorer was replaced (abstain floor 0.34 raw containment -> 0.95 coverage).
Eight PASS specs import it; the six that declared it were correctly flagged
and re-bought in the same slot; seven certificates that did not declare —
`ME.3`, `ME.4`, `ME.5`, `ME.9`, `ME.10`, `T2.20`, `XL.00` — stayed green
while `run status`'s staleness lane printed **0 stale PASS rows**. One of
them, `ME.9`, is named by id in `GOAL.md`.

(The audit's list also named `ME.11.A`; measured here, that module never
imports `EpisodicMemory` — its only mention is prose in the docstring. The
count this spec ratchets is its OWN instrument's reading: 35 implemented
specs, 30 of them PASS, enumerated below by name.)

THE GENERALISABLE FORM, from LESSONS.md: an opt-in integrity check reports
the health of the population that opted in, and that population is invisible
in the output. The tool cannot distinguish "this certificate is fresh" from
"this certificate cannot become stale", and it prints the first for both.
The repair is not a rule ("always declare") — rules here have a measured
~12-hour half-life — it is an instrument: parse every registered spec's
module statically, and make an undeclared import a red on the board.

FIVE PROPERTIES, each able to fail on its own:

  P1  KNOWN-POSITIVE: a module whose `_experiment` lazily imports a
      repo-root module and declares nothing is caught. The import is
      deliberately nested inside a function, because in this codebase the
      imports are overwhelmingly lazy — a top-level-only walk found 2 of the
      35 real violators, so the nested case is the load-bearing one.
  P2  KNOWN-NEGATIVE: the identical module plus the one-line declaration is
      clean. A detector that fires on declarers gets ignored in a fortnight.
  P3  THE REAL LADDER, ratcheted by NAME, shrink-only: every registered spec
      with an implementation file is swept; a violator absent from
      `GRANDFATHERED` — a NEW undeclared import, on a new spec or added to
      an old one — is a FAIL with the spec and module named. A count would
      let one spec repair and another break even; the set cannot.
  P4  THE SET CANNOT GO STALE: every `GRANDFATHERED` entry must still match
      its spec's live violation exactly. When a spec adds its declaration,
      its entry must be DELETED in the same commit (the floor follows the
      number down — `UNREACHABLE_BASELINE`'s rule, applied to names); an
      entry for a spec that no longer exists or no longer violates is a FAIL,
      because a stale allowlist entry is a licence for silent regression.
  P5  EVERY REGISTERED MODULE COMPILES — `compile()`, not `ast.parse`.
      `dp_04_slow_path_verbal.py` carried a statement above its
      `from __future__` import for seven days; `ast.parse` accepts that and
      `compile()` refuses it, so every battery that parsed the tree saw
      nothing while the module was unimportable. The fixture asserts exactly
      that split (parse OK, compile SyntaxError), so this property cannot
      quietly degrade to a parse check.

THE CONTROL is the blind spot re-enacted: the OPT-IN instrument — examine
only modules that DECLARE `IMPL_DEPS`, the shape of both the pre-audit
staleness lane and the corrected-then-recorrected queue-note grep — run over
a two-module population containing one undeclared importer. It MUST read
zero violations where the full instrument reads one. A control that also
sees the violator would mean opt-in was never the defect and this spec
measures nothing.

Static throughout: nothing here imports a test module, touches the real
ledger, or runs a spec. `module_path_for(strict=True)` resolves files;
`undeclared_impl_imports` (protocol.py, declared below) does the walking.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..protocol import (Ledger, module_path_for, run_spec,
                        undeclared_impl_imports)
from ..registry import BY_ID, LADDER

# The predicate lives in protocol.py beside impl_deps_of/impl_sha_of — one
# code path for writer and reader, T0.17's precedent. Without this line an
# edit to the walker leaves this PASS describing a detector that no longer
# exists — the exact defect this spec polices, one level up.
IMPL_DEPS = ["experiments/protocol.py"]

REPO = Path(__file__).resolve().parents[2]

# ── P3/P4's ratchet: today's violators, BY NAME, shrink-only ────────────────
# Measured 2026-09-06 by the shipped predicate over all 150 implemented
# registered specs: 35 violators, 30 holding PASS. DO NOT ADD AN ENTRY —
# a new violator's repair is its declaration, never a bigger allowlist
# (adding one here is the widening move UNREACHABLE_BASELINE's header
# forbids for floors). DELETE an entry in the same commit that declares its
# imports; P4 fails until you do.
# 2026-09-06, 78th audit B2: ME.1/ME.3/ME.4/ME.5/ME.9/ME.10/T2.20/XL.00
# declared their imports and their entries are deleted here, same commit —
# the set follows the repairs down, 35 -> 27.
GRANDFATHERED = {
    "D1.0": ("TrainingPipeline", "UnifiedBrain"),
    "LF.01": ("EpisodicMemory",),
    "ME.2": ("OwnerProfile",),
    "ME.8": ("WorkingMemory",),
    "PG.7": ("ContactAudio",),
    "PG.8": ("TrainingPipeline",),
    "PL.00": ("UnifiedBrain", "playground"),
    "T0.03": ("UnifiedBrain",),
    "T0.04": ("UnifiedBrain",),
    "T0.06": ("VirtualWorld",),
    "T0.07": ("UnifiedBrain", "VirtualWorld"),
    "T0.14": ("TrainingPipeline",),
    "T0.16": ("TrainingPipeline",),
    "T0.25": ("TrainingPipeline",),
    "T1.01": ("UnifiedBrain",),
    "T1.03": ("UnifiedBrain",),
    "T1.04": ("UnifiedBrain",),
    "T1.05": ("UnifiedBrain",),
    "T1.06": ("UnifiedBrain",),
    "T1.10": ("UnifiedBrain",),
    "T1.11": ("UnifiedBrain",),
    "T1.12": ("UnifiedBrain",),
    "T1.13": ("MoCapLoader",),
    "T2.00": ("TrainingPipeline",),
    "T2.10": ("EpisodicMemory",),
    "T2.12": ("EmotionalState",),
    "T3.09": ("EpisodicMemory",),
}

# ── fixtures: source bytes, never files on disk ─────────────────────────────
# P1's import is INSIDE a function on purpose (see the docstring). The module
# name is a real repo-root module so the existence check is live, but nothing
# is ever imported — the predicate is static.
_FIXTURE_VIOLATOR = b'''
"""fixture: lazy repo-root import, no declaration."""
def _experiment(seed):
    from EpisodicMemory import EpisodicMemory
    return {}
'''

_FIXTURE_DECLARER = b'''
"""fixture: same import, one-line declaration."""
IMPL_DEPS = ["EpisodicMemory.py"]
def _experiment(seed):
    from EpisodicMemory import EpisodicMemory
    return {}
'''

# P5's fixture is dp_04's defect verbatim: a statement between the docstring
# and the future-import. `ast.parse` accepts it; `compile()` raises.
_FIXTURE_LATE_FUTURE = b'''
"""fixture: banner above the future import."""
_BANNER = "looks harmless"
from __future__ import annotations
'''


def _parse_ok(src: bytes) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _compile_ok(src: bytes, name: str) -> bool:
    try:
        compile(src, name, "exec")
        return True
    except SyntaxError:
        return False


def _sweep() -> tuple[dict, list]:
    """(violators, compile_failures) over every implemented registered spec."""
    violators, compile_bad = {}, []
    for spec in LADDER:
        path = module_path_for(spec.id, strict=True)
        if path is None:
            continue
        src = path.read_bytes()
        if not _compile_ok(src, str(path)):
            compile_bad.append(spec.id)
        missing, _problem = undeclared_impl_imports(path, source=src)
        if missing:
            violators[spec.id] = tuple(missing)
    return violators, compile_bad


def _experiment(seed: int) -> dict:
    # P1 / P2 — the detector on known ground.
    pos, _ = undeclared_impl_imports("fixture_violator.py",
                                     source=_FIXTURE_VIOLATOR)
    neg, _ = undeclared_impl_imports("fixture_declarer.py",
                                     source=_FIXTURE_DECLARER)

    # P5's fixture — asserting the parse/compile SPLIT, so the sweep below
    # cannot quietly weaken to ast.parse.
    late_future_split = (_parse_ok(_FIXTURE_LATE_FUTURE)
                         and not _compile_ok(_FIXTURE_LATE_FUTURE, "fx"))

    # P3 / P4 / P5 — the real ladder.
    violators, compile_bad = _sweep()
    new_violations = {sid: mods for sid, mods in violators.items()
                      if sid not in GRANDFATHERED}
    widened = {sid: mods for sid, mods in violators.items()
               if sid in GRANDFATHERED and GRANDFATHERED[sid] != mods}
    stale_entries = {sid: mods for sid, mods in GRANDFATHERED.items()
                     if sid not in BY_ID
                     or violators.get(sid) != mods}

    return {
        "detector_fires_on_lazy_undeclared": pos == ("EpisodicMemory",),
        "detector_silent_on_declarer": neg == (),
        "late_future_import_parse_compile_split": late_future_split,
        "undeclared_importers": len(violators),
        "grandfathered": len(GRANDFATHERED),
        "new_undeclared": len(new_violations) + len(widened),
        "new_undeclared_named": {**new_violations, **widened},
        "grandfather_stale": len(stale_entries),
        "grandfather_stale_named": dict(stale_entries),
        "compile_failures": len(compile_bad),
        "compile_failures_named": list(compile_bad),
    }


def _control(seed: int) -> dict:
    """The OPT-IN instrument over a population with one hidden violator.

    Examine only modules that declare `IMPL_DEPS` — the pre-audit staleness
    lane's shape, and the corrected queue-note grep's first stage. Over
    {violator, declarer} it must read ZERO violations: the violator never
    enters its domain. That wrong answer is the measured defect; a control
    that sees the violator would mean this spec measures nothing.
    """
    from ..protocol import impl_deps_of

    population = {"fixture_violator.py": _FIXTURE_VIOLATOR,
                  "fixture_declarer.py": _FIXTURE_DECLARER}
    optin_violations = 0
    examined = 0
    for name, src in population.items():
        deps, _ = impl_deps_of(name, source=src)
        if not deps:            # declared nothing -> outside the domain
            continue
        examined += 1
        missing, _ = undeclared_impl_imports(name, source=src)
        optin_violations += len(missing)

    full_missing, _ = undeclared_impl_imports("fixture_violator.py",
                                              source=_FIXTURE_VIOLATOR)
    return {"optin_examined": examined,
            "optin_violations": optin_violations,
            "full_instrument_sees_violator": len(full_missing) > 0}


def _check(m: dict, c: dict) -> bool:
    props = {
        "detector_fires_on_lazy_undeclared":
            m.get("detector_fires_on_lazy_undeclared", False),
        "detector_silent_on_declarer":
            m.get("detector_silent_on_declarer", False),
        "late_future_import_parse_compile_split":
            m.get("late_future_import_parse_compile_split", False),
        "no_new_undeclared": m.get("new_undeclared", 1) == 0,
        "no_stale_grandfather": m.get("grandfather_stale", 1) == 0,
        "all_modules_compile": m.get("compile_failures", 1) == 0,
    }
    m["properties_failed"] = sum(1 for v in props.values() if not v)
    m["failed_properties"] = [k for k, v in props.items() if not v]

    # The control must have produced the WRONG answer: the opt-in instrument
    # reads clean over a population the full instrument sees a violator in.
    # `.get(..., 1)` so an empty control reads as "opt-in saw it", i.e. FAIL.
    control_is_blind = (c.get("optin_violations", 1) == 0
                        and c.get("full_instrument_sees_violator", False))
    m["control_optin_reads_clean"] = control_is_blind
    return all(props.values()) and control_is_blind


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.35"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
