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

Three detectors, because they fail differently:

  1. SENSITIVITY (the general one). For every PASSing spec, replay its stored
     `_check` against its stored metrics, then perturb one referenced key at a
     time. A key that cannot move the verdict under any perturbation is inert.
     Deliberately measured AT THE OPERATING POINT — the values the run actually
     produced — not over a hypothetical input space. "Could this assertion have
     fired on a run like the ones we really get?" is the question Law 1 asks;
     "is it reachable for some fictional input" is not.

     **THE SUBJECT IS THE RECORDED METRIC, NOT THE DICT SLOT** (repair,
     2026-08-29, after this spec's own FAIL at attempt 22 named four keys in
     `T0.24`, `T1.02` and `T2.04`). Reading `m["k"]` in a `_check` is not the
     same act as consulting the number the run recorded under `k`, and the
     first version of this detector could not tell the two apart. It counted a
     key as an assertion whenever the AST mentioned it, so it reported as
     "disarmed" two things that are not defects at all:

       - `T1.02` computes `m["beats_mean_baseline"]` from `m["mean_baseline"]`
         and `m["structured_heldout"]` and *then* asserts on it. Perturbing the
         recorded slot cannot move the verdict because the gate overwrites it
         first — but the assertion is live, on its inputs, and those inputs are
         scanned. Same shape in `T0.24` (`m["control_reproduces_scar"]` is
         computed from two `c` keys).
       - `T2.04` reads `m["ridge_beats_null_any"]` only inside
         `if not claim and not m[...]`, which converts a FAIL into a VOID. On a
         PASSing row that branch never executes, so the key is unread — not
         unarmed. Demanding that a VOID-escalation guard be exercised by a PASS
         asks for something impossible by construction.

     So the base evaluation now runs against a RECORDING dict that logs every
     read and write in order, and each key lands in exactly one class:

       CONSULTED  read from the record before the gate wrote to it → perturbed,
                  and if it cannot move the verdict it is DISARMED. The defect.
       COMPUTED   the gate stored it before reading it back. Exempt — but ONLY
                  if some store of that key derives from at least one other
                  record read. `m["x"] = True; return m["x"]` is a constant
                  asserting against itself and stays DISARMED.
       UNREACHED  never read at this operating point (branch not taken, or
                  short-circuited away). Counted and reported, not gated.

     Both exemptions forfeit inside a gate that carries a precedence hazard,
     exactly as the `or`-redundancy exemption already does, and the control
     carries a fixture for each so an exemption can never quietly become a
     hole. The hole the UNREACHED class *does* open — a gate that consults
     nothing at all, `def _check(m, c): return True` — is closed by a fourth
     detector below.

  0. KEYLESS GATES (added with the repair above, and gated). A `_check` whose
     base evaluation reads NOTHING from the record is decorative in the purest
     sense available: no perturbation of any recorded number can move it. The
     sensitivity detector cannot see this, because it has no keys to perturb —
     a scan of zero keys returns zero disarmed keys and reads as clean. Fixture
     F3 in the control is exactly this gate and must be caught on every run.

  2. PRECEDENCE (the narrow one). Parse every `_check` and flag an `or` whose
     operands include an unparenthesised `and`, which is the exact shape above.
     Kept separate because it fires on source that is *currently* harmless but
     one metric value away from silently disarming a gate. It is also what
     licenses the redundancy exemption in detector 1: structure alone cannot
     tell T1.09's deliberate `absurd_oom or absurd_peak_gb > MAX_GB` from an
     `or` that `and`-precedence manufactured, so a gate carrying a hazard
     forfeits the exemption and its dead operands count as disarmed.

  3. STALENESS. A PASS asserts that THIS gate accepted THOSE metrics, but the
     gate can be edited afterwards and nothing re-checks the pairing. Every
     PASSing spec's current `_check` is replayed against its recorded metrics
     and must still return True. This is not a substitute for re-running the
     experiment — it certifies only that the stored numbers still clear the
     current bar. It is how T0.09's precedence fix was verified without
     spending a Colab round-trip: the 2026-08-04 metrics satisfy the repaired
     gate, so the recorded PASS was substantively sound and only the guard was
     off.

Nothing is skipped silently. Gates that will not import, expose no `_check`,
have unreadable source, or raise when replayed are COUNTED and gated, because
an unaudited gate that leaves the numerator alone is the same lie one level up.
T0.13 excludes only itself, and says so in `self_excluded_gates`: its own entry
is written after the scan, so it always reflects the previous version of this
file. Its own gate is exercised by the control instead.

Control: three known-bad gates, each scanned separately, each targeting a
different detector. Without them a scan finding nothing would be
indistinguishable from a scan that does not work — which is precisely the
"silence is not success" failure this repo has paid for repeatedly, and
precisely what happened on this spec's own second attempt.

  F1  the pre-fix T0.09 check, verbatim, against T0.09's recorded metrics.
      The sensitivity AND precedence detectors must both flag it (>=3 disarmed
      keys, >=1 hazard).
  F2  `m["all_good"] = True; return m["ok"] and m["all_good"]` — a slot the
      gate writes from a CONSTANT and then asserts on. The COMPUTED exemption
      must refuse it, because nothing recorded is under assertion.
  F3  `return True` — a gate that consults no record at all. The KEYLESS
      detector must flag it; every other detector is blind to it.

F2 and F3 were added on 2026-08-29 with the recording-dict repair, so that the
two new exemptions and the new class are each proved live on every run rather
than trusted. Amendment to `_check` is strengthen-only under the T1.02
precedent: three more conjuncts, none removed, and the prior version stands in
the ledger's history.
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

# F2 — the hole the COMPUTED exemption would open if it were unconditional: a
# slot the gate writes from a CONSTANT and then asserts on. `m["all_good"]` is
# recorded False and the gate still passes, because nothing recorded is under
# assertion. `_record_derived_stores` must refuse the exemption here.
CONSTANT_SRC = '''
def _check(m, c):
    m["all_good"] = True
    return m["ok"] and m["all_good"]
'''
CONSTANT_M = {"ok": True, "all_good": False}
CONSTANT_C = {"ok": False}

# F3 — the hole the UNREACHED class would open: a gate with no keys at all.
# Every other detector on this page is blind to it by construction, because
# each of them starts from a key.
KEYLESS_SRC = '''
def _check(m, c):
    return True
'''
KEYLESS_M = {"anything": 1}
KEYLESS_C = {"anything": 0}

# F4 — a key the AST cannot name, read through `.get()` and thrown away. The
# DYNAMIC detector must see it; every AST-driven detector on this page is blind
# to it, which is how 49 such keys sat unscanned until 2026-08-29.
DYNAMIC_SRC = '''
def _check(m, c):
    ok = m.get("real", False)
    m.get("decorative", False)
    return ok
'''
DYNAMIC_M = {"real": True, "decorative": True}
DYNAMIC_C = {"unused": 0}

_FIXTURES = (
    ("T0.09_prefix", CONTROL_SRC, CONTROL_M, CONTROL_C),
    ("F2_constant", CONSTANT_SRC, CONSTANT_M, CONSTANT_C),
    ("F3_keyless", KEYLESS_SRC, KEYLESS_M, KEYLESS_C),
    ("F4_dynamic", DYNAMIC_SRC, DYNAMIC_M, DYNAMIC_C),
)

# ── the dynamic-key backlog, adjudicated once and frozen ────────────────────
#
# Recording the reads (2026-08-29) made this detector see keys no AST walk can
# name: `.get(k)` inside a comprehension, and f-string subscripts. That found
# 54 inert keys the previous version was structurally blind to. Five were the
# detector's own gap (XL.00's `math.isfinite` VOID guards, unfalsifiable
# against a finite-only perturbation set — fixed above). The other 49 are all
# one shape, and it is a shape this spec ALREADY exempts in its `or` spelling:
#
#   LC.00 (9)   `clearing = sum(1 for kind in CORES if _sigma(...) >= GATE)`
#               then `clearing >= MIN_CORES_CLEARING`. One core's numbers move
#               the count by at most 1 and the margin is wider than that.
#   LC.02 (39)  `m[f"{arm}/clears@{x}"]` over 5 arms x 8 budgets, same count.
#   T0.08 (1)   `not all(c.get(k, True) for k in _STALE_PROPS)` — an `any` over
#               the control's reverted properties; one member is slack while
#               another carries it.
#
# A member of an `any`/`all`/count aggregation is redundant, not disarmed —
# the same judgement `redundant_disjunct_keys` already makes for T1.09's
# `absurd_oom or absurd_peak_gb > MAX_GB`. The AST exemption recognises the
# `or` keyword and cannot recognise the loop that means the same thing, and
# writing that detector is a unit of its own (see LOOP_JOURNAL 2026-08-29).
#
# So the gate here is a SET, not a count: these specs are adjudicated as
# aggregation slack and their key counts may move freely when they re-run,
# but a NEW spec appearing in this list is unadjudicated and turns T0.13
# red. Shrink-only by construction — removing an id can only tighten it.
#
# Three more adjudicated 2026-09-02 (one by one, per the do-not-batch order;
# full records in LOOP_JOURNAL ~17:1x), after landing post-08-29 and firing
# T0.13 attempt 22:
#
#   T2.09 (1)   `c["claim_dwell"]` — AGGREGATION SLACK, T0.08's exact class.
#               The gate `not _claim_holds(c)` is a De Morgan OR over three
#               conjuncts; the recorded PASS row's control fails ALL three
#               (claim_dwell 1.0 vs NULL_DWELL_MAX 0.20, claim_fed_ratio
#               9.5e11 vs FED_RATIO_MAX 1.5, coverage 0.4298 vs EXPLORE_FRAC
#               0.80), so any one member is carried slack by the other two.
#               No correct rewrite avoids `not (a and b and c)`, and the AST
#               or-exemption cannot see through a helper call.
#   VO.02 (3)   `c["coord"]`, `c["coord_std"]`, `m["untrained_coord_std"]` —
#               same De Morgan class through `if _claim(c): return False` and
#               the untrained-null merge. The scrambled control fails the
#               information conjunct outright (mi_ear 0.0400 < perm_p95
#               0.0577; cic 0.0160 < perm_p95 0.0448) and the untrained null
#               fails both conjuncts (coord 0.392 vs COORD_MIN 0.70; mi
#               0.0996) — the coordination keys are carried slack by MI, and
#               a null failing several gates at once is the design.
#   W0.DIAG (1) `c["margin_down_std"]` — NOT aggregation; a distinct
#               sub-shape: SIGN-CARRIED SCALE SLACK. V6 is a one-sided house
#               t-gate (`mean*sqrt(3)/max(std,1e-9) >= SIGMA_GATE -> VOID`)
#               and the recorded mean sits on the safe side (margin_down
#               -11.054): a scale factor cannot flip a sign, so no std
#               perturbation (0, ±1, ±1e9, nan, ±inf) moves the verdict —
#               while the paired MEAN key is fully live (perturb margin_down
#               positive and V6 fires). Every correct t-stat spelling has
#               this property; the record names the new sub-shape rather
#               than shoehorning it into "aggregation".
DYNAMIC_ADJUDICATED = frozenset({"LC.00", "LC.02", "T0.08",
                                 "T2.09", "VO.02", "W0.DIAG"})


def _perturbations(v: Any) -> list:
    """Same-type alternatives. Type-preserving on purpose: swapping in a None
    would raise a TypeError inside most checks, and an exception would read as
    'the key matters' when it only means the key was compared."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, float):
        # NaN and inf are here because a whole class of gate exists only to
        # catch them: `XL.00` reads `indep_p`, `trend_p`, `uniform_z`,
        # `c_at_death_indep_p` and `c_drift_trend_p` ONLY inside
        # `math.isfinite(...)` VOID guards. Against a finite-only perturbation
        # set those five assertions were unfalsifiable by construction and this
        # detector called them disarmed — the detector's gap, not the spec's.
        # Floats only: an int slot is usually an index or a count, and a NaN
        # there raises rather than measures.
        return [0.0, 1.0, -1.0, v + 1, v - 1, 1e9, -1e9,
                float("nan"), float("inf"), float("-inf")]
    if isinstance(v, int):
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


class _Recording(dict):
    """A metrics dict that logs every read and write, in order.

    This is the instrument that separates "the gate consulted the number the
    run recorded" from "the gate mentioned a slot of that name". `__getitem__`
    and `get` both log, so a check that reads `c.get("k", "")` is seen exactly
    like one that writes `c["k"]` — the AST-only version of this detector saw
    neither, and the four false positives of attempt 22 all lived in the gap.
    """

    def __init__(self, data: dict, log: list, tag: str):
        super().__init__(data)
        self._log = log
        self._tag = tag

    def __getitem__(self, key):
        self._log.append((self._tag, key, "get"))
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._log.append((self._tag, key, "get"))
        return dict.get(self, key, default)

    def __setitem__(self, key, value):
        self._log.append((self._tag, key, "set"))
        dict.__setitem__(self, key, value)


def _verdict(fn: Callable, m: dict, c: dict, log: list | None = None):
    """Deep-copied every call: some checks WRITE to their metrics (T2.02 sets
    m["verdict"]), and a detector that let that leak would score the next
    perturbation against a mutated baseline.

    `log`, when given, receives the (tag, key, "get"/"set") trace of the call.
    """
    trace = [] if log is None else log
    try:
        out = fn(_Recording(copy.deepcopy(m), trace, "m"),
                 _Recording(copy.deepcopy(c), trace, "c"))
    except Exception as e:
        return ("RAISED", type(e).__name__)
    if isinstance(out, Status):
        return ("STATUS", out.value)
    return ("BOOL", bool(out))


def _access_classes(log: list) -> tuple[set, set]:
    """(consulted, computed) as (tag, key) pairs.

    A key is CONSULTED if the gate read it before writing it — the recorded
    number reached the assertion. It is COMPUTED if the first access was a
    write: whatever the run recorded under that name was discarded, and the
    value asserted on is the gate's own. Read-then-write-then-read counts as
    consulted, because the first read is the one the record answered.
    """
    consulted, computed, written = set(), set(), set()
    for tag, key, op in log:
        if op == "set":
            written.add((tag, key))
        elif (tag, key) in written:
            computed.add((tag, key))
        else:
            consulted.add((tag, key))
    return consulted, computed - consulted


def _record_derived_stores(fn: Callable, src: str | None) -> dict:
    """(tag, key) -> did any assignment to that slot read the record?

    This is what keeps the COMPUTED exemption from becoming a hole. `T1.02`
    computes `m["beats_mean_baseline"]` from two recorded numbers, so asserting
    on it asserts on them. `m["all_good"] = True` (control fixture F2) reads
    nothing, so asserting on it asserts on nothing — and a slot with no parsed
    assignment at all defaults to False, i.e. stays a defect. Unparsed is not
    exempt.
    """
    fdef = _fdef(src)
    out: dict = {}
    if fdef is None:
        return out
    names = [a.arg for a in fdef.args.args]
    m_name = names[0] if names else None
    c_name = names[1] if len(names) > 1 else None

    def tag_of(nm):
        return "m" if nm == m_name else ("c" if nm == c_name else None)

    def reads_record(node) -> bool:
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)
                    and tag_of(sub.value.id) and isinstance(sub.ctx, ast.Load)):
                return True
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and tag_of(sub.func.value.id)):
                return True
        return False

    for node in ast.walk(fdef):
        if not isinstance(node, ast.Assign):
            continue
        derived = reads_record(node.value)
        for t in node.targets:
            if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                    and tag_of(t.value.id) and isinstance(t.slice, ast.Constant)
                    and isinstance(t.slice.value, str)):
                k = (tag_of(t.value.id), t.slice.value)
                out[k] = out.get(k, False) or derived
    return out


def _key_classes(fn: Callable, src: str | None, m: dict, c: dict) -> dict:
    """Sort every key this gate touches into one of four classes.

    Returns {"disarmed", "redundant", "computed", "unreached", "consulted_n"}.
    Only `disarmed` is gated; `consulted_n` == 0 is the keyless case, gated
    separately by the caller.
    """
    empty = {"disarmed": [], "redundant": [], "computed": [], "unreached": [],
             "dynamic": [], "consulted_n": None}
    log: list = []
    base = _verdict(fn, m, c, log)
    if base[0] == "RAISED":
        return empty         # cannot score a gate we cannot evaluate; reported separately
    consulted, computed = _access_classes(log)
    m_keys, c_keys, disjunct_only = _referenced_keys(fn, src)
    stores = _record_derived_stores(fn, src)
    named = {("m", k) for k in m_keys} | {("c", k) for k in c_keys}

    out = {"disarmed": [], "redundant": [], "computed": [], "unreached": [],
           "dynamic": [], "consulted_n": len(consulted)}
    for tag, k in sorted(named | consulted | computed):
        # Keys the recorder saw but the AST never named — reached through
        # `.get(k)` in a comprehension, or through an f-string subscript like
        # `m[f"{arm}/clears@{x}"]`. They are REPORTED on their own line rather
        # than mixed into the pre-registered count; see DYNAMIC_ADJUDICATED.
        dynamic_only = (tag, k) not in named
        store = m if tag == "m" else c
        label = f"{tag}[{k!r}]"
        if (tag, k) in consulted:
            if k not in store:
                # Read via `.get()` with the key absent from the record. There
                # is no recorded number to perturb, so there is nothing to say
                # about it here; `stale_gates` is the detector that owns a gate
                # whose record no longer answers it.
                continue
            moved = False
            for alt in _perturbations(store[k]):
                probe = dict(store)
                probe[k] = alt
                got = _verdict(fn, probe, c) if tag == "m" else _verdict(fn, m, probe)
                if got != base:
                    moved = True
                    break
            if not moved:
                bucket = ("dynamic" if dynamic_only else
                          "redundant" if label in disjunct_only else "disarmed")
                out[bucket].append(label)
        elif (tag, k) in computed:
            # Exempt only if the gate built this value out of the record. A
            # constant written and then asserted on is the defect itself.
            out["computed" if stores.get((tag, k)) else
                "dynamic" if dynamic_only else "disarmed"].append(label)
        else:
            out["unreached"].append(label)
    return out


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
    disarmed, redundant, hazards = {}, {}, {}
    computed, unreached, dynamic, keyless = {}, {}, {}, []
    unevaluable, unreadable, stale, scanned = [], [], [], 0
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
        # A PASS is a claim that THIS gate accepted THOSE metrics. The gate can
        # be edited afterwards — T0.09's was, to fix the precedence bug — and
        # nothing re-checks that the stored entry still satisfies it. Replay is
        # free and turns "the ledger agrees with the code" from an assumption
        # into a measurement. It is not a substitute for re-running the
        # experiment; it only certifies that the recorded numbers still clear
        # the current bar.
        if _verdict(fn, m, c) not in (("BOOL", True), ("STATUS", Status.PASS.value)):
            stale.append(spec_id)
        h = _precedence_hazards(fn, src)
        if h:
            hazards[spec_id] = h
        cls = _key_classes(fn, src, m, c)
        dis, red = cls["disarmed"], cls["redundant"]
        comp, unre, dyn = cls["computed"], cls["unreached"], cls["dynamic"]
        # EVERY exemption is forfeited by a gate that ALSO carries a precedence
        # hazard. Structure alone cannot separate honest redundancy from a
        # disarmed assertion — the pre-fix T0.09 keys and T1.09's
        # `absurd_peak_gb` are both inert operands of an `or`, and both look
        # identical to the sensitivity detector. What distinguishes them is
        # that T0.09's `or` was never written: `and` binding tighter turned an
        # intended conjunction into one. So an `or` that swallows an `and` is
        # evidence of intent, and its dead operands are defects, not slack.
        # The COMPUTED and UNREACHED exemptions forfeit on the same reasoning:
        # in a gate whose control flow is already known to be an accident, a
        # branch that did not execute is not evidence that it was not meant to.
        if h:
            dis = sorted(dis + red + comp + unre + dyn)
            red = comp = unre = dyn = []
        # A gate that consulted NOTHING from the record cannot be moved by any
        # number the run produced. The sensitivity detector is structurally
        # blind to this — zero keys perturbed yields zero disarmed keys, which
        # reads as clean — so it is counted on its own. Only reached when the
        # gate evaluated: `consulted_n` is None for an unevaluable gate, which
        # is already gated by `unevaluable_gates`.
        if cls["consulted_n"] == 0:
            keyless.append(spec_id)
        if dis:
            disarmed[spec_id] = dis
        if red:
            redundant[spec_id] = red
        if comp:
            computed[spec_id] = comp
        if unre:
            unreached[spec_id] = unre
        if dyn:
            dynamic[spec_id] = dyn

    def _detail(d):
        return "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(d.items()))

    return {
        "gates_scanned": scanned,
        "disarmed_conjunct_keys": sum(len(v) for v in disarmed.values()),
        "specs_with_disarmed_keys": len(disarmed),
        "redundant_disjunct_keys": sum(len(v) for v in redundant.values()),
        "computed_gate_keys": sum(len(v) for v in computed.values()),
        "unreached_gate_keys": sum(len(v) for v in unreached.values()),
        "dynamic_inert_keys": sum(len(v) for v in dynamic.values()),
        "dynamic_inert_specs": ",".join(sorted(dynamic)),
        "keyless_gates": len(keyless),
        # Every key that cannot move its gate, whichever exemption applies.
        # Widened 2026-08-29 to include the dynamic class: reporting 0 here
        # while 49 keys sit in `dynamic_inert_keys` would be the same kind of
        # understatement this spec exists to catch.
        "inert_gate_keys": (sum(len(v) for v in disarmed.values())
                            + sum(len(v) for v in redundant.values())
                            + sum(len(v) for v in dynamic.values())),
        "precedence_hazards": sum(hazards.values()),
        "specs_with_precedence_hazards": len(hazards),
        "unevaluable_gates": len(unevaluable),
        "unreadable_gates": len(unreadable),
        "stale_gates": len(stale),
        "disarmed_detail": _detail(disarmed),
        "redundant_detail": _detail(redundant),
        "computed_detail": _detail(computed),
        "unreached_detail": _detail(unreached),
        "dynamic_inert_detail": _detail(dynamic),
        "keyless_detail": ", ".join(sorted(keyless)),
        "hazard_detail": ", ".join(sorted(hazards)),
        "unevaluable_detail": ", ".join(sorted(unevaluable)),
        "unreadable_detail": ", ".join(sorted(unreadable)),
        "stale_detail": ", ".join(sorted(stale)),
    }


def _experiment(seed: int) -> dict:
    ledger = Ledger()
    entries, unloadable = [], []
    self_excluded = 0
    for spec_id, r in _passing(ledger):
        # A gate cannot audit its own ledger entry. That entry is written AFTER
        # this scan, so it always reflects the PREVIOUS version of this file —
        # on the run where T0.13's own `_check` changes, replaying it against
        # last run's metrics raises KeyError and reports T0.13 as stale, every
        # time, forever. The exclusion is stated in the metrics rather than
        # silent: T0.13's own gate is instead exercised by the control fixture,
        # which must fire on a known-bad check on every run.
        if spec_id == "T0.13":
            self_excluded += 1
            continue
        mod_name = spec_id.lower().replace(".", "_")
        # Counted, not swallowed. A PASSing spec whose module will not import,
        # or which exposes no `_check`, is a gate this scan did NOT audit — and
        # an unaudited gate must not quietly leave the numerator alone.
        try:
            mod = importlib.import_module(f"..tests.{_module_stem(mod_name)}", __package__)
        except Exception:
            unloadable.append(spec_id)
            continue
        fn = getattr(mod, "_check", None)
        if fn is None:
            unloadable.append(spec_id)
            continue
        entries.append((spec_id, fn, None, dict(r.get("metrics") or {}),
                        dict(r.get("control_metrics") or {})))
    out = _scan(entries)
    out["self_excluded_gates"] = self_excluded
    out["unloadable_gates"] = len(unloadable)
    out["unloadable_detail"] = ", ".join(sorted(unloadable))
    return out


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
    entries = []
    for label, src, m, c in _FIXTURES:
        ns: dict = {}
        exec(compile(src, f"<{label}>", "exec"), ns)
        # Source passed explicitly — inspect cannot recover it for exec'd code,
        # and a detector that reads nothing reports a clean bill of health.
        entries.append((label, ns["_check"], src, dict(m), dict(c)))
    out = _scan(entries)
    out["self_excluded_gates"] = 0
    out["unloadable_gates"] = 0
    out["unloadable_detail"] = ""
    return out


def _dynamic_specs(m: dict) -> frozenset:
    """The spec ids carrying dynamic-key slack, as a set. A missing field reads
    as the whole ladder rather than as nothing: a row recorded before this
    field existed has not been scanned by this detector, and an unscanned
    ladder must not certify itself clean."""
    raw = m.get("dynamic_inert_specs")
    if raw is None:
        return frozenset({"<field absent — this row predates the detector>"})
    return frozenset(x for x in raw.split(",") if x)


def _check(m: dict, c: dict) -> bool:
    # The detectors must come up clean on the real ladder...
    ladder_clean = (m["disarmed_conjunct_keys"] == 0
                    and m["precedence_hazards"] == 0
                    and m["unevaluable_gates"] == 0
                    and m["unreadable_gates"] == 0
                    and m["stale_gates"] == 0
                    and m["unloadable_gates"] == 0
                    and m["keyless_gates"] == 0        # added 2026-08-29
                    and _dynamic_specs(m) <= DYNAMIC_ADJUDICATED)
    # ...and every detector must fire on the fixture built for it, or a clean
    # scan is indistinguishable from a scan that does not run. F1, the pre-fix
    # T0.09 check, has exactly three unreachable assertions (ok,
    # cuda_available, matmul_finite) and one precedence hazard; F2 contributes
    # the fourth disarmed key (a constant asserting against itself, which the
    # COMPUTED exemption must refuse); F3 contributes the keyless gate.
    control_caught = (c["disarmed_conjunct_keys"] >= 4
                      and c["precedence_hazards"] >= 1
                      and c["keyless_gates"] >= 1      # added 2026-08-29 (F3)
                      and c["dynamic_inert_keys"] >= 1  # added 2026-08-29 (F4)
                      and c["gates_scanned"] == 4      # was 1, before F2-F4
                      and c["unreadable_gates"] == 0)
    # A scan of nothing is not a clean scan.
    scanned_enough = m["gates_scanned"] >= 30
    return ladder_clean and control_caught and scanned_enough


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.13"], _experiment, _check, control_fn=_control, ledger=ledger)
