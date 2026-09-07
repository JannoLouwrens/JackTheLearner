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

THE SECOND SCAR, ONE LEVEL UP (81st audit, 2026-09-07): every property above
is ONE HOP deep, and the hop past it broke a live certificate the same week
the one-hop detector shipped. `me_11_a_lexical_incumbent.py` imports
`_build_life` from a SIBLING test module; `ME.1`'s strengthening gave that
helper a fourth return value; `ME.11.A`'s control started raising
`ValueError` while holding a PASS, and eight `ME.11` modules kept certifying
a retriever (`EpisodicMemory.py`, reached at 1–3 hops through
`me_1_event_log.py` or `fixtures/paraphrase_eval.py`) that had been replaced
underneath them. Six of the eight declared `paraphrase_eval.py` — the honest
instinct, catching nothing, because declaring the door does not hash what is
behind it. The audit's B2 hand-counted eight violators; the shipped walker
reads TWENTY (`LC.03`, `LC.07`, `T0.26` were never in anyone's count). Four
more properties:

  P6  TRANSITIVE KNOWN-POSITIVE, and it must split against P1's instrument:
      a module reaching `EpisodicMemory` only through a sibling test module
      is caught by `transitive_impl_imports` while `undeclared_impl_imports`
      reads CLEAN on the same bytes. The split is the property — a chain the
      one-hop walker could see would mean the transitive class was never
      invisible and this strengthening measures nothing.
  P7  TRANSITIVE KNOWN-NEGATIVE: the identical chain with both far ends
      declared is clean.
  P8  THE REAL LADDER, transitively, ratcheted BY NAME, shrink-only:
      `TRANSITIVE_GRANDFATHERED` under exactly P3/P4's rules — a new or
      widened reach is a FAIL, a stale entry is a FAIL, the set only drains.
  P9  THE MUTATION FALSIFIER the audit ordered: mutate `EpisodicMemory.py`'s
      bytes (via `impl_sha_of`'s `dep_bytes` lane — nothing on disk moves)
      and `ME.11.A`'s impl_sha MUST change. This is only true because
      `ME.11.A` now declares its transitive reach; delete that declaration
      and P9 fails. A staleness detector that cannot be shown to fire is the
      thing the 81st audit was about.

THE CONTROL is the blind spot re-enacted, twice: the OPT-IN instrument —
examine only modules that DECLARE `IMPL_DEPS`, the shape of both the
pre-audit staleness lane and the corrected-then-recorrected queue-note grep
— run over a two-module population containing one undeclared importer. It
MUST read zero violations where the full instrument reads one. And the
ONE-HOP instrument — the corrected shape that closed the 78th audit — run
over the transitive chain, where it MUST also read zero while the transitive
walker sees two undeclared reaches. A control that sees either violator
would mean the corresponding blindness was never the defect and this spec
measures nothing.

Static throughout: nothing here imports a test module, touches the real
ledger, or runs a spec. `module_path_for(strict=True)` resolves files;
`undeclared_impl_imports` and `transitive_impl_imports` (protocol.py,
declared below) do the walking.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..protocol import (Ledger, impl_deps_of, impl_sha_of, module_path_for,
                        run_spec, transitive_impl_imports,
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
# 2026-09-06 17:xx: the 18 cheap PASS certificates declared (ME.2, ME.8,
# PG.7, PG.8, T0.03, T0.04, T0.06, T0.07, T0.14, T0.16, T0.25, T1.03,
# T1.04, T1.05, T1.10, T1.11, T1.13, T2.12) and re-bought in the same slot,
# 27 -> 9. What remains, and why each stays: D1.0's attempt 2 is IN FLIGHT
# (editing its module under a live watcher risks a sha mismatch on the row
# the watcher writes); T1.01/T1.06/T1.12/T2.00 are PASS certs whose re-buys
# cost 17-63 min each — declare each ONLY in a slot that re-runs it, or the
# declaration itself manufactures a DRIFTED claim; LF.01/T2.10/T3.09
# hold FAIL/VOID rows whose re-runs are routed or held, so a declaration
# would push them into the stale lane with no sanctioned way out.
# 2026-09-07: PL.00 declared (playground.py, UnifiedBrain.py, eye_quality.py)
# and deleted from BOTH sets in the same commit — its re-run was sanctioned
# by the renderer-bakeoff disposition, 9 -> 8.
GRANDFATHERED = {
    "D1.0": ("TrainingPipeline", "UnifiedBrain"),
    "LF.01": ("EpisodicMemory",),
    "T1.01": ("UnifiedBrain",),
    "T1.06": ("UnifiedBrain",),
    "T1.12": ("UnifiedBrain",),
    "T2.00": ("TrainingPipeline",),
    "T2.10": ("EpisodicMemory",),
    "T3.09": ("EpisodicMemory",),
}

# ── P8's ratchet: transitive violators, BY NAME, shrink-only ────────────────
# Measured 2026-09-07 by `transitive_impl_imports` over all implemented
# registered specs: 20 violators. The audit hand-counted 8; the walker found
# LC.03/LC.07/T0.26 and the direct-set overlap nobody added up. Same rules as
# GRANDFATHERED above: DO NOT ADD OR WIDEN AN ENTRY — the repair is the
# spec's own declaration; DELETE the entry in the same commit that declares
# (P8 fails until you do). ME.11.A and ME.11.0 — the two LIVE PASSes the
# 81st audit flagged — declared their reaches in this same commit and were
# re-bought; they are deliberately absent. Everything below holds a FAIL,
# VOID, PARKED or held row whose re-run is routed elsewhere, or a PASS whose
# re-buy costs 17-63 min (T1.01/T1.06/T1.12/T2.00, same reasoning as the
# direct set above): declare each ONLY in a slot that re-runs it.
TRANSITIVE_GRANDFATHERED = {
    "D1.0": ("TrainingPipeline.py", "UnifiedBrain.py"),
    "LC.03": ("experiments/tests/lc_02_throughput_floor.py",),
    "LC.07": ("experiments/tests/lc_02_throughput_floor.py",),
    "LF.01": ("EpisodicMemory.py",),
    "ME.11": ("EpisodicMemory.py",
              "experiments/tests/me_11_a_lexical_incumbent.py",
              "experiments/tests/me_1_event_log.py"),
    "ME.11.B": ("EpisodicMemory.py", "experiments/tests/me_1_event_log.py"),
    "ME.11.C": ("EpisodicMemory.py", "experiments/tests/me_1_event_log.py"),
    "ME.11.D": ("EpisodicMemory.py", "experiments/tests/me_1_event_log.py"),
    "ME.11.E": ("EpisodicMemory.py",
                "experiments/tests/me_11_a_lexical_incumbent.py",
                "experiments/tests/me_1_event_log.py"),
    "ME.11.F": ("EpisodicMemory.py",
                "experiments/tests/me_11_a_lexical_incumbent.py",
                "experiments/tests/me_11_b_bm25s_stemming.py",
                "experiments/tests/me_1_event_log.py"),
    "T0.26": ("experiments/tests/ba_01_feels_the_fall.py",),
    "T1.01": ("UnifiedBrain.py",),
    "T1.06": ("UnifiedBrain.py",),
    "T1.12": ("UnifiedBrain.py",),
    "T2.00": ("TrainingPipeline.py",),
    "T2.10": ("EpisodicMemory.py", "experiments/tests/me_1_event_log.py"),
    "T3.09": ("EpisodicMemory.py",),
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

# P6/P7's chain is the ME.11.A defect verbatim: the entry module imports a
# helper from a SIBLING test module, and only the sibling touches the impl.
# The chain lives in an overlay, never on disk, so the known-positive cannot
# rot when the real ladder's violators drain. The entry path is what gives
# the walker its package for the relative import.
_FIXTURE_TRANSITIVE_ENTRY = "experiments/tests/fixture_transitive.py"
_FIXTURE_TRANSITIVE = b'''
"""fixture: reaches EpisodicMemory only through a sibling test module."""
def _experiment(seed):
    from .fixture_bridge import build
    return {}
'''
_FIXTURE_TRANSITIVE_DECLARED = b'''
"""fixture: the identical chain, both far ends declared."""
IMPL_DEPS = ["EpisodicMemory.py", "experiments/tests/fixture_bridge.py"]
def _experiment(seed):
    from .fixture_bridge import build
    return {}
'''
_OVERLAY = {"experiments/tests/fixture_bridge.py": b'''
"""fixture bridge: the sibling that actually imports the impl."""
def build():
    from EpisodicMemory import EpisodicMemory
    return EpisodicMemory
'''}
_CHAIN_REACH = ("EpisodicMemory.py", "experiments/tests/fixture_bridge.py")


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


def _sweep() -> tuple[dict, dict, list]:
    """(violators, transitive_violators, compile_failures) over every
    implemented registered spec."""
    violators, trans_violators, compile_bad = {}, {}, []
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
        t_missing, _problem = transitive_impl_imports(path, source=src)
        if t_missing:
            trans_violators[spec.id] = tuple(t_missing)
    return violators, trans_violators, compile_bad


def _experiment(seed: int) -> dict:
    # P1 / P2 — the one-hop detector on known ground.
    pos, _ = undeclared_impl_imports("fixture_violator.py",
                                     source=_FIXTURE_VIOLATOR)
    neg, _ = undeclared_impl_imports("fixture_declarer.py",
                                     source=_FIXTURE_DECLARER)

    # P5's fixture — asserting the parse/compile SPLIT, so the sweep below
    # cannot quietly weaken to ast.parse.
    late_future_split = (_parse_ok(_FIXTURE_LATE_FUTURE)
                         and not _compile_ok(_FIXTURE_LATE_FUTURE, "fx"))

    # P6 / P7 — the transitive detector on known ground, and the SPLIT: the
    # one-hop walker must be blind to the same chain, or the transitive class
    # was never invisible and this strengthening measures nothing.
    t_pos, _ = transitive_impl_imports(_FIXTURE_TRANSITIVE_ENTRY,
                                       source=_FIXTURE_TRANSITIVE,
                                       overlay=_OVERLAY)
    t_pos_onehop, _ = undeclared_impl_imports("fixture_transitive.py",
                                              source=_FIXTURE_TRANSITIVE)
    t_neg, _ = transitive_impl_imports(_FIXTURE_TRANSITIVE_ENTRY,
                                       source=_FIXTURE_TRANSITIVE_DECLARED,
                                       overlay=_OVERLAY)

    # P9 — the 81st audit's mutation falsifier: mutate EpisodicMemory.py's
    # bytes through impl_sha_of's dep_bytes lane (disk untouched) and
    # ME.11.A's sha must move. Only true while ME.11.A declares the reach.
    me11a = module_path_for("ME.11.A", strict=True)
    me11a_deps, _ = impl_deps_of(me11a)
    sha_now = impl_sha_of(me11a)
    sha_mut = impl_sha_of(me11a, dep_bytes={
        "EpisodicMemory.py": b"# mutated by T0.35's P9 falsifier\n"})
    mutation_fires = ("EpisodicMemory.py" in me11a_deps
                      and sha_now is not None and sha_mut is not None
                      and sha_now != sha_mut)

    # P3 / P4 / P5 / P8 — the real ladder.
    violators, trans_violators, compile_bad = _sweep()
    new_violations = {sid: mods for sid, mods in violators.items()
                      if sid not in GRANDFATHERED}
    widened = {sid: mods for sid, mods in violators.items()
               if sid in GRANDFATHERED and GRANDFATHERED[sid] != mods}
    stale_entries = {sid: mods for sid, mods in GRANDFATHERED.items()
                     if sid not in BY_ID
                     or violators.get(sid) != mods}
    t_new = {sid: mods for sid, mods in trans_violators.items()
             if sid not in TRANSITIVE_GRANDFATHERED}
    t_widened = {sid: mods for sid, mods in trans_violators.items()
                 if sid in TRANSITIVE_GRANDFATHERED
                 and TRANSITIVE_GRANDFATHERED[sid] != mods}
    t_stale = {sid: mods for sid, mods in TRANSITIVE_GRANDFATHERED.items()
               if sid not in BY_ID
               or trans_violators.get(sid) != mods}

    return {
        "detector_fires_on_lazy_undeclared": pos == ("EpisodicMemory",),
        "detector_silent_on_declarer": neg == (),
        "late_future_import_parse_compile_split": late_future_split,
        "transitive_fires_where_onehop_blind": (t_pos == _CHAIN_REACH
                                                and t_pos_onehop == ()),
        "transitive_silent_on_declarer": t_neg == (),
        "mutation_stales_me11a": mutation_fires,
        "undeclared_importers": len(violators),
        "grandfathered": len(GRANDFATHERED),
        "new_undeclared": len(new_violations) + len(widened),
        "new_undeclared_named": {**new_violations, **widened},
        "grandfather_stale": len(stale_entries),
        "grandfather_stale_named": dict(stale_entries),
        "transitive_undeclared": len(trans_violators),
        "transitive_grandfathered": len(TRANSITIVE_GRANDFATHERED),
        "new_transitive": len(t_new) + len(t_widened),
        "new_transitive_named": {**t_new, **t_widened},
        "transitive_grandfather_stale": len(t_stale),
        "transitive_grandfather_stale_named": dict(t_stale),
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

    # The second blindness, re-enacted the same way: the CORRECTED one-hop
    # instrument — the shape that closed the 78th audit — over the transitive
    # chain. It must read zero where the transitive walker sees two.
    onehop_on_chain, _ = undeclared_impl_imports("fixture_transitive.py",
                                                 source=_FIXTURE_TRANSITIVE)
    trans_on_chain, _ = transitive_impl_imports(_FIXTURE_TRANSITIVE_ENTRY,
                                                source=_FIXTURE_TRANSITIVE,
                                                overlay=_OVERLAY)
    return {"optin_examined": examined,
            "optin_violations": optin_violations,
            "full_instrument_sees_violator": len(full_missing) > 0,
            "onehop_violations_on_chain": len(onehop_on_chain),
            "transitive_sees_chain": len(trans_on_chain) > 0}


def _check(m: dict, c: dict) -> bool:
    props = {
        "detector_fires_on_lazy_undeclared":
            m.get("detector_fires_on_lazy_undeclared", False),
        "detector_silent_on_declarer":
            m.get("detector_silent_on_declarer", False),
        "late_future_import_parse_compile_split":
            m.get("late_future_import_parse_compile_split", False),
        "transitive_fires_where_onehop_blind":
            m.get("transitive_fires_where_onehop_blind", False),
        "transitive_silent_on_declarer":
            m.get("transitive_silent_on_declarer", False),
        "mutation_stales_me11a": m.get("mutation_stales_me11a", False),
        "no_new_undeclared": m.get("new_undeclared", 1) == 0,
        "no_stale_grandfather": m.get("grandfather_stale", 1) == 0,
        "no_new_transitive": m.get("new_transitive", 1) == 0,
        "no_stale_transitive_grandfather":
            m.get("transitive_grandfather_stale", 1) == 0,
        "all_modules_compile": m.get("compile_failures", 1) == 0,
    }
    m["properties_failed"] = sum(1 for v in props.values() if not v)
    m["failed_properties"] = [k for k, v in props.items() if not v]

    # The control must have produced the WRONG answer, twice: the opt-in
    # instrument reads clean over a population the full instrument sees a
    # violator in, and the one-hop instrument reads clean over a chain the
    # transitive walker sees through. `.get(..., 1)` so an empty control
    # reads as "the blind instrument saw it", i.e. FAIL.
    control_is_blind = (c.get("optin_violations", 1) == 0
                        and c.get("full_instrument_sees_violator", False))
    m["control_optin_reads_clean"] = control_is_blind
    control_onehop_blind = (c.get("onehop_violations_on_chain", 1) == 0
                            and c.get("transitive_sees_chain", False))
    m["control_onehop_reads_clean"] = control_onehop_blind
    return all(props.values()) and control_is_blind and control_onehop_blind


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.35"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
