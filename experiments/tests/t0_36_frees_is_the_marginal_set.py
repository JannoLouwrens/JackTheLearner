"""T0.36 — `run blocked`'s `frees` is the set that repair actually buys.

THE SCAR IS FOUR HOURS OLD AND IT IS ON THE PRIORITY PAGE RIGHT NOW.

2026-09-13 10:05 UTC. `T1.08` re-ran under a conjunct armed four hours earlier
and returned FAIL — honestly, on a pre-registered bar that did not move. Three
specs declare `depends_on: T1.08`, and one of them is `T2.01`, a settled FAIL
that had been the project's largest blocker for five weeks at `frees 35`.

`run blocked` immediately printed, and four documents immediately quoted:

    T1.08 = FAIL  frees 41  (blocks 45)  — Seed variance measured

**The true marginal value of repairing `T1.08` alone is 3.** Measured by
counterfactual against this repository's own ledger at `2ed9f85`: assume
`T1.08 := PASS` and exactly `D1.0`, `T2.01` and `T2.02` leave the unreachable
set. The other 38 fall straight back — 35 onto `T2.01`, 3 onto `T2.02` — both
of which are settled non-PASS verdicts that somebody still has to repair. The
headline number overstated the marginal value of the project's top-ranked unit
by **13x**, and `unreachable` moved only 93 -> 97 on the same event, which is
the arithmetic that gives it away: a root cannot acquire 41 dependents while
only 4 specs become stuck.

THE MECHANISM, and it is one line. `_terminal_blockers.walk` resolved a stuck
dependency to ITS roots and dropped the dependency itself:

    roots |= upstream if upstream else {d}

So `T2.01` — FAIL, and blocking 35 specs on its own account — was substituted
away the moment it acquired an unsatisfied dependency of its own, and its
entire mass was credited to the spec underneath it. The comment above that line
named the two cases it handled ("itself stuck" / "merely not-yet-run") and the
third case is the one that bit: **a dependency can be BOTH — stuck behind
something else AND a settled verdict that must itself be repaired.**

WHY IT SURVIVED: THE KNOWN-ANSWER FIXTURE WAS NAMED BUT NEVER WRITTEN.
`_terminal_blockers`' own docstring says *"`ladder`/`by_id` are injectable so
the ranking below can be checked against a graph whose answer is known — see
`_RANKER_FIXTURE`"*. `grep -rn _RANKER_FIXTURE experiments/` returns exactly
that one line. The fixture does not exist, has never existed, and the seam
built to admit it was used by nothing. That is yesterday's lesson arriving one
file over (`3e5fc4f`: *an option stays unpriced when the request names an
instrument that does not exist*) — here it cost a priority ranking rather than
a price. `_RANKER_FIXTURE` below is that promise made real.

THIS IS THE RANKER'S FOUNDING BUG WEARING NEW CLOTHES. `_rank_blockers`' own
docstring records the first one: it ranked by MENTIONS, mentions double-count,
it reported `T2.03 blocks 11` when fixing T2.03 freed two, and *"the ranking
sent the loop at the wrong unit."* The repair was `frees` — the marginal set.
Today's defect is the same overstatement arriving through substitution instead
of double-counting, and it was invisible for the same reason: nothing ever
checked `frees` against the counterfactual it claims to be.

WHAT THIS SPEC CLAIMS, narrowly and on purpose: `frees` is SOUND. What it says
a single repair buys, that repair buys. It does NOT claim completeness, and the
difference is not a hedge — a spec resting on `{T1.08, T2.01}` is freed by the
unphysical counterfactual `T2.01 := PASS` alone, because `T1.08` entered its
root set only THROUGH `T2.01`'s own blockage. You cannot make `T2.01` pass
without repairing `T1.08` first, so the pair is the honest operational answer
and the single-root counterfactual is the wrong oracle for it. Soundness is the
half that is exactly checkable, and it is the half that failed.

THE CONTROL is the ranker as it stood at `2ed9f85`, reconstructed BY DELETION
rather than paraphrase (T0.08 property 5, T0.30's precedent): `_walk_legacy`
below is the shipped body with the `own` term removed and nothing else changed.
It must fail P1 (it claims 41 and delivers 3), P3 and P7 (both fixtures'
known answers), and P5 (`groups` reads empty where the pair belongs). It PASSES
P2, P4 and P6, and those passes are named rather than hidden: P2 is an
alive-proof that the legacy walk also satisfies, P4 is the narrow case the
repair deliberately leaves alone, and P6 is invariance against a baseline the
control IS — vacuous for it by construction. Invariance without detection is
not the property, which is why `_check` does not ask those three of it.
"""
from __future__ import annotations

from ..protocol import Budget, Ledger, Spec, Status, run_spec
from ..registry import BY_ID
from ..run import (LADDER, _AssumeStatus, _rank_blockers, _terminal_blockers,
                   unreachable_count)

SPEC_ID = "T0.36"
N_PROPERTIES = 7.0
IMPL_DEPS = ["experiments/run.py"]

#: The event this spec exists for, asserted rather than remembered. `T1.08`'s
#: row is live at registration; if the ladder ever repairs it these names stop
#: being the witness and P1 carries the property on whatever roots exist then.
SCAR_ROOT = "T1.08"


def _spec(sid: str, deps: list) -> Spec:
    """A registry-shaped Spec carrying only what the walk reads."""
    return Spec(sid, 0, f"fixture {sid}",
                hypothesis="fixture", falsified_by="fixture",
                null_baseline="fixture", metric="fixture",
                budget=Budget.CPU_FAST, depends_on=list(deps))


class _FixtureLedger:
    """A Ledger double over a hand-built graph with a known answer.

    Implements the two methods the walk and the ranker actually call. It is
    NOT duck-typed by coincidence — `Ledger.unsatisfied` is the one rule both
    readers ask (the `T0.22` repair), so this double restates that rule for a
    synthetic graph and nothing else. `t0_15`'s stub went dead for 18 days at a
    rename precisely because nothing pinned its shape; this one is exercised by
    P3/P4/P7 on every run, so a rename breaks it loudly.
    """

    def __init__(self, status: dict):
        self._status = status

    def status(self, spec_id: str) -> Status:
        return self._status.get(spec_id, Status.NOT_RUN)

    def unsatisfied(self, spec) -> list:
        return [(d, "fixture") for d in (spec.depends_on or [])
                if self.status(d) is not Status.PASS]


def _fixture(kind: str) -> tuple:
    """`(ladder, by_id, ledger)` for one named known-answer graph.

    THE PROMISE `_terminal_blockers`' DOCSTRING HAS MADE SINCE IT WAS WRITTEN.
    Three graphs, chosen so that the two walks disagree on two of them and
    agree on the third — an oracle that only ever disagrees cannot show the
    repair is NARROW.

      chain  A = FAIL, B = FAIL <- A, {C,D,E} = NOT_RUN <- B
             Repairing A alone frees B and nothing else: B is still FAIL.
             C/D/E need A and B both, which is what `groups` is for.
      narrow A = FAIL, B = NOT_RUN <- A, C = NOT_RUN <- B
             B carries no verdict of its own, so repairing A really does
             buy B and C. Both walks must say so; the repair may not
             over-fire here or it would bury every ordinary chain.
      void   A = FAIL, B = VOID <- A, C = NOT_RUN <- B
             The rule keys on "carries a settled verdict", never on the FAIL
             token. A VOID is a verdict somebody must repair.
    """
    graphs = {
        "chain": ({"X.A": Status.FAIL, "X.B": Status.FAIL},
                  [("X.A", []), ("X.B", ["X.A"]), ("X.C", ["X.B"]),
                   ("X.D", ["X.B"]), ("X.E", ["X.B"])]),
        "narrow": ({"Y.A": Status.FAIL},
                   [("Y.A", []), ("Y.B", ["Y.A"]), ("Y.C", ["Y.B"])]),
        "void": ({"Z.A": Status.FAIL, "Z.B": Status.VOID},
                 [("Z.A", []), ("Z.B", ["Z.A"]), ("Z.C", ["Z.B"])]),
    }
    status, nodes = graphs[kind]
    ladder = [_spec(sid, deps) for sid, deps in nodes]
    return ladder, {s.id: s for s in ladder}, _FixtureLedger(status)


#: The known answers, written down BEFORE either walk is run on them.
_RANKER_FIXTURE = {
    "chain": {"frees": {"X.A": {"X.B"}},
              "groups": {frozenset({"X.A", "X.B"}): {"X.C", "X.D", "X.E"}}},
    "narrow": {"frees": {"Y.A": {"Y.B", "Y.C"}}, "groups": {}},
    "void": {"frees": {"Z.A": {"Z.B"}},
             "groups": {frozenset({"Z.A", "Z.B"}): {"Z.C"}}},
}


def _walk_legacy(ledger, ladder, by_id) -> dict:
    """`_terminal_blockers` as it stood at `2ed9f85`, BY DELETION.

    Byte-for-byte the shipped body with the `own` term and its comment removed.
    Not a paraphrase and not a crippled copy: this is the instrument that
    printed `frees 41` this morning, and it is the whole of what four documents
    were quoting.
    """
    terminal: dict = {}

    def walk(sid: str, seen: frozenset) -> set:
        if sid in terminal:
            return terminal[sid]
        spec = by_id.get(sid)
        if spec is None or sid in seen:
            return {sid}
        roots: set = set()
        for d, _why in ledger.unsatisfied(spec):
            upstream = walk(d, seen | {sid})
            roots |= upstream if upstream else {d}
        terminal[sid] = roots
        return roots

    for s in ladder:
        walk(s.id, frozenset())
    return terminal


def _unreachable_set(walk, ledger, ladder=None, by_id=None) -> set:
    """The specs `blocked` calls unreachable, under whichever walk is given."""
    ladder = LADDER if ladder is None else ladder
    by_id = BY_ID if by_id is None else by_id
    terminal = walk(ledger, ladder, by_id)
    mentions, _frees, _groups = _rank_blockers(terminal, ledger, ladder)
    return {sid for ids in mentions.values() for sid in ids}


def _obstructed(walk, ledger, ladder=None, by_id=None) -> set:
    """Specs with something BROKEN still in the way — the oracle for `frees`.

    THE OBVIOUS ORACLE IS THE WRONG ONE, and getting it wrong once is how this
    property nearly shipped asking for something false. "Freed" cannot mean
    "immediately runnable": in a chain `A = FAIL <- B <- C` with B and C never
    run, repairing A leaves C waiting on B — and B is ordinary work, not a
    blocker. `frees` says `{B, C}` and is RIGHT to; `blocked` exists to rank
    what must be REPAIRED, and running a runnable spec is the loop's day job.

    So the distinction that carries the property is repair-vs-run: a spec is
    obstructed while any of its terminal roots carries a settled verdict
    (FAIL/VOID/BLOCKED/ERROR, or a PASS gone stale — anything not NOT_RUN).
    That is the same predicate the walk itself now uses to decide whether a
    dependency is a root in its own right, asked from the other end, and two
    readers of one quantity share the rule or they drift.
    """
    ladder = LADDER if ladder is None else ladder
    by_id = BY_ID if by_id is None else by_id
    terminal = walk(ledger, ladder, by_id)
    out = set()
    for s in ladder:
        roots = {r for r in terminal.get(s.id, set()) if r != s.id}
        if any(ledger.status(r) is not Status.NOT_RUN for r in roots):
            out.add(s.id)
    return out


def _probe(walk) -> dict:
    """Every property, under one walk. `walk(ledger, ladder, by_id) -> dict`."""
    failed, checked = [], 0

    def prop(name: str, ok: bool) -> None:
        nonlocal checked
        checked += 1
        if not ok:
            failed.append(name)

    # ---- the live ladder -----------------------------------------------
    led = Ledger()
    terminal = walk(led, LADDER, BY_ID)
    mentions, frees, groups = _rank_blockers(terminal, led, LADDER)
    live_unreachable = {sid for ids in mentions.values() for sid in ids}

    # P1 SOUNDNESS. For every root, everything it claims to free must have
    # nothing BROKEN left in the way once that root alone is repaired and
    # re-run. This is the quantity `_rank_blockers`' docstring calls "the
    # marginal value of this fix alone", and it is what the ranking is sorted
    # by. See `_obstructed` for why this, and not "immediately runnable".
    overstated, worst_gap, worst_root = {}, 0, ""
    for root, claimed in frees.items():
        if not claimed:
            continue
        hyp = _AssumeStatus(led, root, Status.PASS)
        still = _obstructed(walk, hyp)
        missing = sorted(x for x in claimed if x in still)
        if missing:
            overstated[root] = missing
            if len(missing) > worst_gap:
                worst_gap, worst_root = len(missing), root
    prop("p1_frees_is_the_set_that_repair_actually_buys", not overstated)

    # P2 THE INSTRUMENT IS ALIVE. A walk that returns no roots at all satisfies
    # P1 vacuously and would rank an empty board as clean. LESSONS.md's rule
    # for an at-chance control, applied to a zero-violation reading.
    prop("p2_the_ranking_is_not_empty",
         bool(live_unreachable) and any(frees.values()))

    # P3/P4/P7 KNOWN ANSWERS. The fixture the shipped docstring has named since
    # it was written, finally asked.
    def fixture_says(kind: str) -> tuple:
        ladder, by_id, fled = _fixture(kind)
        t = walk(fled, ladder, by_id)
        _m, f, g = _rank_blockers(t, fled, ladder)
        got_f = {r: set(ids) for r, ids in f.items() if ids}
        got_g = {frozenset(r): set(ids) for r, ids in g.items() if ids}
        want = _RANKER_FIXTURE[kind]
        return got_f == want["frees"], got_g == want["groups"]

    chain_f, chain_g = fixture_says("chain")
    narrow_f, narrow_g = fixture_says("narrow")
    void_f, void_g = fixture_says("void")
    prop("p3_chain_fixture_frees_only_the_next_verdict", chain_f)
    prop("p4_narrow_fixture_is_unchanged_by_the_repair", narrow_f and narrow_g)
    prop("p5_chain_fixture_names_the_pair_in_groups", chain_g)
    prop("p7_a_void_intermediate_is_a_root_like_a_fail", void_f and void_g)

    # P6 THE RATCHET DOES NOT MOVE. `unreachable` is shrink-only with a
    # declared floor. An attribution repair re-labels WHO blocks a spec; it may
    # not change WHICH specs are stuck, or it would move a floor as a side
    # effect of a reporting fix.
    legacy_unreachable = _unreachable_set(_walk_legacy, led)
    prop("p6_unreachable_set_is_invariant_under_reattribution",
         live_unreachable == legacy_unreachable)

    n_unreachable, ladder_size = unreachable_count(led, LADDER, mentions=mentions)
    return {
        "properties_checked": float(checked),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "overstated_roots": float(len(overstated)),
        "worst_overstatement": float(worst_gap),
        "worst_root_is_the_scar": 1.0 if worst_root == SCAR_ROOT else 0.0,
        "scar_root_claims": float(len(frees.get(SCAR_ROOT, []))),
        "unreachable": float(n_unreachable),
        "ladder_size": float(ladder_size),
    }


def _experiment(seed: int) -> dict:
    return _probe(_terminal_blockers)


def _control(seed: int) -> dict:
    """The ranker as it stood at `2ed9f85` — see `_walk_legacy`."""
    return _probe(_walk_legacy)


def _check(m: dict, c: dict) -> Status | bool:
    # Every property ran AND every property held. Gating on `properties_failed
    # == 0` alone lets a battery that stopped early read as clean — T0.13's own
    # first bug, and every T0.1x/T0.2x battery since.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that name what it
    # cannot do. P2/P4/P6 are deliberately NOT required of it: P2 is an
    # alive-proof it also satisfies, P4 is the case the repair leaves alone,
    # and P6 asks invariance against a baseline the control is.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p1_frees_is_the_set_that_repair_actually_buys",
                           "p3_chain_fixture_frees_only_the_next_verdict",
                           "p5_chain_fixture_names_the_pair_in_groups",
                           "p7_a_void_intermediate_is_a_root_like_a_fail",
                           } <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
