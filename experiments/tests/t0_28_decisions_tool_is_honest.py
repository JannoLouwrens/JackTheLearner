"""T0.28 — the escalation tool can be shown catching a deadlock and a claim-death.

`experiments/decisions.py` stands over eleven pre-registered constitutional
defaults, ten of which fire on the same date. Every audit opens by running it.
Until this spec it was certified by fixtures its own author wrote — the precise
self-certification `SYSTEM.md`'s first law exists to distrust — and it had
already been wrong once, for six days, in the worst direction: `SYSTEM.md` said
*"experiments/decisions.py enforces this"* of the three-part safety clause while
the tool read no default's content at all. A governing document naming an
enforcement is making a capability claim, and law 1 binds it like any other.

WHAT THE KNOWN-POSITIVES ARE. They are events, not inventions. On 2026-08-29
`D8`'s armed default read *"PARK BA.02"* and `BA.02` was the only claim-kind
spec behind `balance`, a commitment the owner listed as constitutional
(`GOAL.md:41`). One unanswered calendar date would have left that commitment
with nothing falsifiable behind it. `BA.03` was registered on the morning of
2026-08-30 — the prescribed repair, *register a successor* — and that, not any
change to the tool, is why the live board is green today.

WHICH IS EXACTLY WHY THE FIXTURE IS SYNTHETIC. Rows are injected through
`audit(rows_for_safety=...)` rather than read from the live ledger, because a
known-positive pinned to the live repository stops being exercised the moment
somebody repairs the repository. That is the same disease one level up:
a guard that is green because its subject vanished rather than because it was
fixed. P5 makes the two silences distinguishable INSIDE the tool; building the
fixture this way makes them distinguishable for the tool itself.

THE CONTROL is the organ as it stood before 2026-08-30, kept executable: an
`audit()` with no safety pass, and a `--check` whose blocking set omits
`NO-DEFAULT`. Both holes were real and both are reconstructed by deletion
rather than by paraphrase (T0.08 property 5) — the safety pass only ever
APPENDS `SAFETY-CLAIM-DEAD` violations, so dropping them from the return value
reproduces the old return value exactly. It must miss the `D8` positive, miss
the both-named case, and exit 0 on a goal-class entry that arms nothing.

NO ledger writes, no training, no world: the documents are strings built
in-process and the rows are dicts, so the numbers hold still while the RULE
varies. Same shape as T0.19, T0.20 and T0.21.

THE NEWEST KNOWN-POSITIVE IS THE ORGAN'S OWN (2026-09-04, 70th audit B1). `D21`
was armed by the overseer with `default: … the 2026-09-06 FULL Review takes the
W1 design as the FIRST item on its docket …` and `decide_by: 2026-09-11`. The
clock fell five days after the event it commanded: on the morning it became due
it would have ordered a Sunday that had already happened to re-order its docket.
Every field was legal and nothing printed. It was caught by hand, the next
morning, by the organ that wrote it — which is the whole reason this pass
exists, and it is the third time a guard on this desk shipped checking the FORM
of a declaration rather than what it says.

WHAT THIS SPEC DOES NOT CERTIFY, stated so no later reader repeats SYSTEM.md's
mistake in this file's name: two of the three safety clauses — *never edits
GOAL.md*, *never weakens a threshold* — are still enforced by nobody. They are
properties of the COMMIT that fires a default, not of the text that arms it,
and no battery over `decisions.py` can see them. `T0.29` (`champions.py`) is
the companion certificate and is owed.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import re

from ..coverage import _claim_dead
from ..decisions import (BASELINE_ACTION_EXPIRED, BASELINE_UNDECLARED, DOC,
                         audit, blast_radius, check_rc, default_dates,
                         expired_actions, main, parse)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

SPEC_ID = "T0.28"

# A claim about decisions.py must die when decisions.py changes: a certificate
# that survives edits to its own subject is a certificate about nothing (PG.6
# hashing playground.py; T0.21 hashing coverage.py).
IMPL_DEPS = ["experiments/decisions.py"]

N_PROPERTIES = 13

# The pre-2026-08-30 blocking set, verbatim. `NO-DEFAULT` is absent — that is
# the hole, not an abbreviation.
LEGACY_BLOCKING = ("MEANS-ESCALATED", "CLASS", "DATE", "SAFETY-CLAIM-DEAD")

TODAY = _dt.date(2026, 8, 30)

# ── the documents. Each is the smallest thing that carries one defect. ──

# The four parse defects plus one correctly armed entry, and D95 in D1's real
# shape: a header calling an OPTION stale must not read as the DECISION being
# settled.
DOC_PARSE = """
## D90 — an open decision with no declaration at all (OPEN, owner)
## D91 — a means fork sent to the owner instead of to a bakeoff (OPEN, owner)

DECIDE: D91
  class:     means
  default:   ask the owner
  decide_by: 2026-01-01

## D92 — a goal fork with no default (OPEN, owner)

DECIDE: D92
  class:     goal
  decide_by: 2026-01-01

## D93 — correctly armed (OPEN, owner)

DECIDE: D93
  class:     goal
  default:   keep the conservative arm, journal the firing
  decide_by: 2099-01-01

## D94 — RESOLVED, must NOT be reported as open
## D95 — THE OPTION SET IS STALE: an option contradicts a later decree
## D95 — the original question (OPEN, owner)
"""

# `D8` as it actually stood on 2026-08-29, reduced to the sentence that names
# the spec. Wrapped across two physical lines ON PURPOSE: the id lives on the
# continuation line, so a parser that truncates a default at its first line
# computes an empty blast radius and the guard silently never fires. That
# truncation shipped for one run.
DOC_D8 = """
## D80 — the body question (OPEN, owner)

DECIDE: D80
  class:     goal
  default:   Option 1 — PARK the balance claim until a body with directional
             catch authority exists, re-parenting BA.02 behind the
             playground-humanoid line. BA.01 stands untouched.
  decide_by: 2026-08-31
"""

# The same document with the successor ALSO named — the case where registering
# BA.03 is not the repair, because one date still reaches every claim.
DOC_D8_BOTH = DOC_D8.replace("re-parenting BA.02 behind the",
                             "re-parenting BA.02 and BA.03 behind the")

# A means fork whose default names the only two claims behind `smell`. A means
# entry is never armed, so its radius must not be consulted at all — and it
# must block on its own account.
DOC_MEANS = """
## D82 — which nose (OPEN, owner)

DECIDE: D82
  class:     means
  default:   PARK SM.02 and SM.03 both.
  decide_by: 2026-08-31
"""

# A default that names nothing resolvable. `ZZ.99` is a typo, not a reference.
DOC_TYPO = """
## D83 — a default full of typos (OPEN, owner)

DECIDE: D83
  class:     goal
  default:   PARK ZZ.99 and QQ.01 until the world grows traps.
  decide_by: 2026-08-31
"""

# The registry the fixture resolves ids against. Values are unused — only
# membership decides whether an id is a reference (`blast_radius`).
FIXTURE_BY_ID = {i: 1 for i in ("BA.01", "BA.02", "BA.03", "T2.01",
                                "SM.02", "SM.03")}


# ── the owner's OTHER desk (2026-09-04). Real pages, reduced to the items. ──
#
# `docs/PROGRESS.md` is CURRENT-STATE BY DESIGN: the Review rewrites it whole,
# every run. These two are the 09-03 and 09-04 `FOR THE OWNER` sections with
# each item cut to the clause that carries it. Item 2 of the 09-03 page is the
# known-positive and it is an event, not an invention — that recommendation
# vanished unanswered at 24 hours old and the overseer lifted it onto the
# owner's desk by hand as `D21` the next morning.
PAGE_0903 = """
## FOR THE OWNER

1. **Sunday 2026-09-06 is oversubscribed, and I am telling you the order I will
   take it in rather than discovering it at turn 100.** Six OPEN queue rows come
   due that day, on the same run that owes Part 2.

2. **The world is now the measured bottleneck on six independent instruments.**
   **My recommendation: W1 stops being a queue row and becomes the project's
   stated stage.** This is the strategic fork; the `D1.0` gate is a detail
   beside it.

3. **`run blocked` cannot see the project's largest unblock.** `T2.01` blocks 38
   specs; its repair runs through `D1.0`; no spec declares `depends_on: D1.0`.
   That is a real design change to `run blocked`, so it is yours to authorise.

4. **Organ liveness, all green.** builder 06:07, overseer 06:37, field watch
   08-31 05:53 (Mondays — next fire 09-07, inside cadence). No organ is silent.
"""

PAGE_0904 = """
## FOR THE OWNER

1. **THE FORK, and it is new: design throughput is now the binding constraint.**
   My recommendation: let the builder DRAFT redesigns; keep ratification here.

2. **`D20`'s input, measured rather than argued.** The CPU day-meter's first
   full day billed 5,906.8 s and every line item is a re-buy.

3. **Sunday 2026-09-06, order unchanged from yesterday's page.** Six OPEN queue
   rows come due that day, on the same run that owes Part 2.

4. **Organ liveness, all green.** builder 06:17, overseer 06:37, field watch
   08-31 05:53 (Mondays — next fire 09-07, inside cadence). No organ is silent.

---
"""

# The decision file the fixture resolves cites and quotations against. `D20`
# exists; `D21` exists and QUOTES the 09-03 recommendation, which is exactly the
# repair the overseer performed — quoting is the match, so the check rewards the
# thing the system already does.
NEEDED_ASKS = """
## D20 — what should the CPU day-ceiling count? (2026-09-04, overseer)

## D21 — the Review has recommended that W1 stop being a queue row (2026-09-04)

> *"My recommendation: W1 stops being a queue row and becomes the project's
> stated stage."* — docs/PROGRESS.md, Review 2026-09-03 (`f529ab1`)
"""


# ── the clock defect (2026-09-04, 70th audit B1). Also a real entry. ────────
#
# `D21` as the 69th audit ARMED it, cut to the two lines that carry the defect:
# an action on 2026-09-06, a clock on 2026-09-11. On the morning it became due
# it would have ordered a Sunday that had already happened to re-order its
# docket. Both spellings of the date are here because the real entry wrote both
# — an ISO-only reader would have caught this one by luck, and the defaults in
# the live document routinely write only the short form.
DOC_D21 = """
## D21 — the W1 recommendation, lifted onto the owner's desk (OPEN, owner)

DECIDE: D21
  class:     goal
  default:   NEITHER (ii) NOR (iii). What fires instead is the narrowest
             already-permitted action: the 2026-09-06 FULL Review takes the W1
             design as the FIRST item on its docket, and `w0-too-shallow` is
             already dated 09-06, so this re-orders a scheduled item and
             creates no new permission.
  decide_by: 2026-09-11
"""

# The 70th audit's repair, verbatim in kind: SHORTEN the clock so the firing
# lands the morning before the action. A deadline may tighten on its own — it
# widens nothing — and this is the only direction available to a default.
DOC_D21_SHORTENED = DOC_D21.replace("2026-09-11", "2026-09-05")


def _row(commitment: str, kinds: dict, n_pass: int = 0) -> dict:
    """One `coverage.report()` row, reduced to the keys the safety pass reads."""
    return {"commitment": commitment, "n_pass": n_pass, "kinds": kinds,
            "parked": {}, "n_specs": len(kinds), "specs": sorted(kinds)}


# `balance` on 2026-08-29: one sensor, one claim, no PASS.
ROWS_BEFORE = [_row("balance", {"BA.01": "sensor", "BA.02": "claim"})]
# ...after the 08-30 repair: the successor exists and no default names it.
ROWS_REPAIRED = [_row("balance", {"BA.01": "sensor", "BA.02": "claim",
                                  "BA.03": "claim"})]
# ...and the state that must NOT be mistaken for the repair: the claim is gone.
ROWS_VANISHED = [_row("balance", {"BA.01": "sensor"})]


def _violations(text: str, rows, *, safety_enforced: bool) -> list:
    """The organ under test, or the organ as it stood before 2026-08-30.

    The pre-fix `audit()` did not contain the safety loop. That loop only ever
    APPENDS, so deleting its output from the return value reconstructs the old
    behaviour exactly rather than paraphrasing it.
    """
    v, _rows = audit(text, TODAY, rows_for_safety=rows, by_id=FIXTURE_BY_ID)
    if safety_enforced:
        return v
    return [x for x in v
            if x[0] not in ("SAFETY-CLAIM-DEAD", "DEFAULT-ACTION-EXPIRED")]


def _rc(violations: list, *, safety_enforced: bool) -> int:
    """`--check`'s exit code, under today's blocking set or the legacy one."""
    if safety_enforced:
        return check_rc(violations)
    undeclared = sum(1 for k, _, _ in violations if k == "UNDECLARED")
    if undeclared > BASELINE_UNDECLARED:
        return 1
    return 1 if any(v[0] in LEGACY_BLOCKING for v in violations) else 0


def _expired(text: str, *, safety_enforced: bool) -> set:
    """`{decision_id}` this organ reports as naming an out-of-date action."""
    return {did for kind, did, _ in _violations(text, [],
                                                safety_enforced=safety_enforced)
            if kind == "DEFAULT-ACTION-EXPIRED"}


def _asks(progress: str, prev, needed: str = NEEDED_ASKS, resolved: str = "",
          *, safety_enforced: bool) -> set:
    """`{(kind, key)}` for the two owner-ask classes, through the real `audit`.

    The control arm is `decisions.py` as it stood before 2026-09-04, when the
    owner's other desk had no reader at all — reconstructed by DELETION, like
    the safety pass above it, because the ask pass only ever APPENDS.
    """
    v, _rows = audit(needed, TODAY, rows_for_safety=[], by_id=FIXTURE_BY_ID,
                     progress_text=progress, prev_progress_text=prev,
                     resolved_text=resolved)
    if not safety_enforced:
        return set()
    return {(k, key) for k, key, _ in v if k.endswith("-OWNER-ASK")}


def _hazards(text: str, rows, *, safety_enforced: bool) -> list:
    """`(id, commitment, claims)` triples this organ reports for `text`."""
    return [(did, why.split("'")[1], why)
            for kind, did, why in _violations(text, rows,
                                              safety_enforced=safety_enforced)
            if kind == "SAFETY-CLAIM-DEAD"]


def _live_asks() -> tuple:
    """`(unrouted, vanished)` on the real pages, for the record only.

    Never gated — see the metric's comment. Returns `(0, 0)` if the live
    documents cannot be read, because a certificate about a scanner must not
    turn a missing file into a science result.
    """
    try:
        from ..decisions import PROGRESS, RESOLVED, _previous_page
        v, _ = audit(DOC.read_text(), _dt.date.today(),
                     rows_for_safety=[], by_id=BY_ID,
                     progress_text=PROGRESS.read_text(),
                     prev_progress_text=_previous_page(PROGRESS),
                     resolved_text=RESOLVED.read_text())
    except Exception:
        return (0, 0)
    return (sum(1 for k, _, _ in v if k == "UNROUTED-OWNER-ASK"),
            sum(1 for k, _, _ in v if k == "VANISHED-OWNER-ASK"))


def _probe(safety_enforced: bool) -> dict:
    failed: list[str] = []
    S = safety_enforced

    # P1 — the four parse defects and the one correct entry. A scanner that
    # flags everything is as useless as one that flags nothing, so the negative
    # (D93 armed, D94 resolved) is half the property. D95 is D1's real shape:
    # a header calling an OPTION stale is not a resolution of the DECISION.
    v, rows = audit(DOC_PARSE, TODAY, rows_for_safety=[], by_id=FIXTURE_BY_ID)
    kinds = {did: kind for kind, did, _ in v}
    if (kinds.get("D90") != "UNDECLARED"
            or kinds.get("D91") != "MEANS-ESCALATED"
            or kinds.get("D92") != "NO-DEFAULT"
            or kinds.get("D95") != "UNDECLARED"
            or "D93" in kinds or "D94" in kinds
            or [r["id"] for r in rows] != ["D93"]):
        failed.append("p1_parse_defects_are_flagged_and_armed_is_not")

    # P2 — KNOWN POSITIVE, and it is a thing that happened. `D8` on 2026-08-29:
    # its default names BA.02, which was the only claim behind `balance`. The
    # tool must name the decision, the commitment and the claim, and the gate
    # must STOP — a violation the exit code ignores is a violation nobody acts
    # on. The pre-fix organ read no default's content and misses it entirely.
    haz = _hazards(DOC_D8, ROWS_BEFORE, safety_enforced=S)
    if (len(haz) != 1 or haz[0][0] != "D80" or haz[0][1] != "balance"
            or "BA.02" not in haz[0][2]
            or _rc(_violations(DOC_D8, ROWS_BEFORE, safety_enforced=S),
                   safety_enforced=S) != 1):
        failed.append("p2_d8_known_positive_fires")

    # P3 — it must go quiet ONLY for the prescribed repair, and the property is
    # the CONTRAST: the same document, the same default, one registered
    # successor between them — loud before, quiet after. Asserting only the
    # quiet half would be passed by an organ that is quiet about everything,
    # which is precisely the organ this spec's control is.
    if (not _hazards(DOC_D8, ROWS_BEFORE, safety_enforced=S)
            or _hazards(DOC_D8, ROWS_REPAIRED, safety_enforced=S)):
        failed.append("p3_successor_is_the_repair")

    # P4 — and registering a successor is NOT the repair when the same date
    # still reaches it. A default naming BA.02 AND BA.03 fires again: the
    # question is never "does a successor exist", it is "does one calendar
    # event reach every claim". Without this, the prescribed repair could be
    # discharged by writing a spec the default already covers.
    hb = _hazards(DOC_D8_BOTH, ROWS_REPAIRED, safety_enforced=S)
    if len(hb) != 1 or hb[0][0] != "D80" or "BA.03" not in hb[0][2]:
        failed.append("p4_both_named_fires")

    # P5 — THE TWO SILENCES ARE DIFFERENT, and this is the property the
    # existing fixture did not have. Delete the claim instead of succeeding it
    # and the hazard ALSO disappears — because there is nothing left to put at
    # risk. That must not read as health: `coverage._claim_dead` is True on the
    # vanished row and False on the repaired one, so quiet-because-fixed and
    # quiet-because-gone are distinguishable, and the second is `coverage.py`'s
    # red rather than a green here. A guard that cannot tell them apart can be
    # discharged by deleting its subject.
    vanished_quiet = not _hazards(DOC_D8, ROWS_VANISHED, safety_enforced=S)
    if (not vanished_quiet
            or not _claim_dead(ROWS_VANISHED[0])
            or _claim_dead(ROWS_REPAIRED[0])
            or _claim_dead(ROWS_BEFORE[0])):
        failed.append("p5_vanished_is_not_repaired")

    # P6 — a recorded PASS is never put at risk by a calendar. A default cannot
    # un-record a certificate, so a commitment with a passing claim is not a
    # hazard however thoroughly the default names it — and the same row with
    # the PASS removed MUST fire, or "no hazard" is coming from somewhere else
    # entirely. Both directions or neither.
    doc_loco = DOC_D8.replace("BA.02", "T2.01")
    passed = [_row("locomotion", {"T2.01": "claim"}, n_pass=1)]
    unpassed = [_row("locomotion", {"T2.01": "claim"}, n_pass=0)]
    if (_hazards(doc_loco, passed, safety_enforced=S)
            or not _hazards(doc_loco, unpassed, safety_enforced=S)):
        failed.append("p6_a_pass_is_not_at_calendar_risk")

    # P7 — an id that resolves to nothing is a typo, not a reference; counting
    # it would inflate every radius with noise. And the over-approximation runs
    # the other way ON PURPOSE: a resolvable id ANYWHERE in the text is in the
    # radius, including on a wrapped continuation line, because the radius says
    # what a firing could REACH and never what it will do. The continuation
    # half is a real regression — the first parser truncated a default at its
    # first physical line, which would have computed {} for `D8` itself.
    joined = parse(DOC_D8)[0]["D80"]
    if (blast_radius("PARK BA.02 and ZZ.99", FIXTURE_BY_ID) != {"BA.02"}
            or blast_radius(DOC_TYPO, FIXTURE_BY_ID)
            or _hazards(DOC_TYPO, ROWS_BEFORE, safety_enforced=S)
            or "BA.02" not in joined.get("default", "")
            or blast_radius(joined.get("default", ""),
                            FIXTURE_BY_ID) != {"BA.01", "BA.02"}):
        failed.append("p7_typo_is_not_a_reference")

    # P8 — a MEANS fork is never armed, so its blast radius must not be
    # consulted: it is not a thing the owner may leave unanswered, it is a
    # bakeoff somebody has not written. Its default names both claims behind
    # `smell` and must still raise no hazard — while blocking on its own
    # account, in both organs.
    smell = [_row("smell", {"SM.02": "claim", "SM.03": "claim"})]
    mv = _violations(DOC_MEANS, smell, safety_enforced=S)
    if (_hazards(DOC_MEANS, smell, safety_enforced=S)
            or not any(k == "MEANS-ESCALATED" for k, _, _ in mv)
            or _rc(mv, safety_enforced=S) != 1):
        failed.append("p8_means_is_never_armed")

    # P9 — the exit code blocks on every class the report calls fatal, not on
    # one. `NO-DEFAULT` — a goal entry that declares itself and arms nothing —
    # was printed and exited 0 until 2026-08-30: D1's exact disease with a
    # DECIDE block on top. The same one-class-ratchet shape the 40th audit
    # found in `champions.py`. The second direction keeps the fix honest: an
    # UNDECLARED backlog at or below its baseline is a queue, not a wall, and
    # a ratchet that blocked on everything would be no ratchet at all.
    nodefault = _violations(DOC_PARSE, [], safety_enforced=S)
    nodefault = [x for x in nodefault if x[0] == "NO-DEFAULT"]
    backlog = [("UNDECLARED", f"D{i}", "") for i in range(BASELINE_UNDECLARED)]
    if (_rc(nodefault, safety_enforced=S) != 1
            or _rc(backlog, safety_enforced=S) != 0
            or _rc(backlog + [("UNDECLARED", "Dx", "")],
                   safety_enforced=S) != 1):
        failed.append("p9_ratchet_counts_every_class")

    # P10 — against the LIVE document, and this is the half a fixture can
    # never do. It must parse into an armed set with no blocking violation,
    # every armed row must carry an ISO date and a non-empty default, and the
    # report must print each default IN FULL. That last one is not decoration:
    # `main()` sliced defaults at 110 characters while the live ones run
    # 369-1041, so 70-89% of every constitutional clause the owner armed had
    # never appeared in any report anyone read. A clause nobody can read is an
    # unreviewed clause, and the check that would catch a truncation is
    # whether the LONGEST live default's last words survive to stdout.
    live_v, live_rows = audit(DOC.read_text(), TODAY)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([])
    printed = re.sub(r"\s+", " ", buf.getvalue())
    longest = max((r["default"] for r in live_rows), key=len, default="")
    tail = re.sub(r"\s+", " ", longest)[-60:]
    if (not live_rows
            or any(v[0] in ("SAFETY-CLAIM-DEAD", "MEANS-ESCALATED", "CLASS",
                            "DATE", "NO-DEFAULT") for v in live_v)
            or any(not r["default"] or not isinstance(r["due"], _dt.date)
                   for r in live_rows)
            or len(longest) < 200 or tail not in printed):
        failed.append("p10_live_document_is_armed_and_readable")

    # P11 — THE SECOND DESK. An item under `## FOR THE OWNER` that reaches no
    # decision entry is reported, and the burden runs the blunt way ON PURPOSE:
    # every numbered item is an ask until it cites a live `D`, is quoted into a
    # decision file, or DECLARES `NO-DECISION:` with a reason. Both directions,
    # because a scanner that flags everything is as useless as one that flags
    # nothing — item 2 cites `D20` and must stay quiet, and an item outside the
    # section must be invisible.
    a04 = _asks(PAGE_0904, None, safety_enforced=S)
    exempted = PAGE_0904.replace(
        "   My recommendation: let the builder DRAFT redesigns; keep ratification here.",
        "   NO-DECISION: a report, nothing here to rule on")
    outside = "## FOR THE BUILDER\n\n1. **Not an owner ask.** Invisible.\n"
    if ({k for k in a04 if k[0] == "UNROUTED-OWNER-ASK"}
            != {("UNROUTED-OWNER-ASK", "PROGRESS #1"),
                ("UNROUTED-OWNER-ASK", "PROGRESS #3"),
                ("UNROUTED-OWNER-ASK", "PROGRESS #4")}
            or ("UNROUTED-OWNER-ASK", "PROGRESS #1")
            in _asks(exempted, None, safety_enforced=S)
            or _asks(outside, None, safety_enforced=S)):
        failed.append("p11_unrouted_owner_ask_is_reported")

    # P12 — KNOWN POSITIVE, and it is a thing that happened. The 09-03 page's
    # item 3 left the owner's desk when the 09-04 page replaced it and reached
    # no decision file; item 2 vanished too and is QUIET because `D21` quotes
    # it, which is the repair the overseer actually performed. Three further
    # directions, each a real trap:
    #   - a rewritten-but-surviving item (1 and 4, reworded, still on the page)
    #     must NOT read as vanished, or the class fires on every edit;
    #   - `D1.0` is a SPEC and must not route an ask to decision `D1` — that
    #     lookahead bug silenced this exact positive during development;
    #   - no baseline must manufacture nothing (`review_queue`'s rule).
    van = {k[1] for k in _asks(PAGE_0904, PAGE_0903, safety_enforced=S)
           if k[0] == "VANISHED-OWNER-ASK"}
    repaired = {k[1] for k in _asks(
        PAGE_0904, PAGE_0903,
        resolved="> *\"`run blocked` cannot see the project's largest "
                 "unblock.\"* Ruled builder work; implemented at 9e847cf.",
        safety_enforced=S) if k[0] == "VANISHED-OWNER-ASK"}
    if (van != {"PROGRESS(prev) #3"} or repaired
            or any(k[0] == "VANISHED-OWNER-ASK"
                   for k in _asks(PAGE_0904, None, safety_enforced=S))):
        failed.append("p12_vanished_owner_ask_is_the_known_positive")

    # P13 — KNOWN POSITIVE, and it is the newest thing that happened: `D21`,
    # armed 2026-09-04 by the overseer with an action on 09-06 and a clock on
    # 09-11, caught by hand the next morning because no instrument in the system
    # would say a word. Five directions, and the first two are the property:
    #   - the defect fires, on the real path, on the real entry;
    #   - the REPAIR silences it — shortening the clock, which is the only
    #     direction a deadline may move on its own. Quiet-because-fixed, not
    #     quiet-because-the-subject-left (P5's rule, one class over);
    #   - the equality case is the same defect, because `main()` marks overdue
    #     at `(today - decide_by).days > 0` and so the earliest fire is
    #     `decide_by + 1`. An action dated ON decide_by has already passed;
    #   - a bare `MM-DD` takes the NEAREST year, not decide_by's blindly, so a
    #     December action behind a January clock reads as the December just
    #     gone rather than the one eleven months out;
    #   - it is a RATCHET and not a wall: at baseline the gate passes, one over
    #     and it stops. A class that blocked would have to be switched off the
    #     day a default legitimately cites another decision's clock.
    at_base = [("DEFAULT-ACTION-EXPIRED", f"D{i}", "")
               for i in range(BASELINE_ACTION_EXPIRED)]
    if (_expired(DOC_D21, safety_enforced=S) != {"D21"}
            or _expired(DOC_D21_SHORTENED, safety_enforced=S)
            or not expired_actions("act on 2026-09-05", _dt.date(2026, 9, 5))
            or expired_actions("act on 2026-09-06", _dt.date(2026, 9, 5))
            or default_dates("by 12-28", _dt.date(2027, 1, 3))
            != [_dt.date(2026, 12, 28)]
            or _rc(at_base, safety_enforced=S) != 0
            or _rc(at_base + [("DEFAULT-ACTION-EXPIRED", "Dx", "")],
                   safety_enforced=S) != 1):
        failed.append("p13_expired_default_action_is_the_known_positive")

    live_asks = _live_asks()
    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "live_armed": float(len(live_rows)),
        "live_violations": float(len(live_v)),
        "live_longest_default_chars": float(len(longest)),
        "live_specs_in_radius": float(len(
            {s for r in live_rows for s in blast_radius(r["default"])})),
        # Recorded, NOT gated. The live reading moves whenever the Review
        # rewrites its page, and a certificate that flapped daily on somebody
        # else's prose would be re-bought for no scientific reason. The ratchet
        # in `decisions.py` owns the threshold; the ledger owns the number, so
        # a day the owner's desk grows is visible in the record either way.
        "live_unrouted_asks": float(live_asks[0]),
        "live_vanished_asks": float(live_asks[1]),
        # Also recorded and not gated, for the same reason: the live count moves
        # when the owner's desk does. `decisions.py`'s ratchet owns the
        # threshold; the ledger owns the number.
        "live_expired_actions": float(
            sum(1 for v in live_v if v[0] == "DEFAULT-ACTION-EXPIRED")),
    }


def _experiment(seed: int) -> dict:
    return _probe(safety_enforced=True)


def _control(seed: int) -> dict:
    """`decisions.py` as it stood before each pass this file added, kept
    executable.

    Four holes, all real and all reconstructed by DELETION rather than by
    paraphrase (T0.08 property 5), because every pass only ever APPENDS:
    `audit()` carried no safety pass, so no default's content was ever read;
    `--check`'s blocking set omitted `NO-DEFAULT`, so a goal-class entry that
    armed nothing was printed and exited 0; the owner's other desk had no
    reader at all; and no pass asked whether a default's action still existed
    on the day the default fires. It must miss the `D8` known-positive (P2),
    miss the both-named case (P4), pass a document containing an unarmed
    escalation (P9), miss both owner-ask classes (P11, P12) and miss `D21`'s
    expired clock (P13).
    """
    return _probe(safety_enforced=False)


def _check(m: dict, c: dict) -> Status | bool:
    # Every property ran AND every property held, on both arms. Gating on
    # `properties_failed == 0` alone lets a battery that stopped early read as
    # clean — T0.13's own first bug, and T0.19/T0.20/T0.21 all carry this guard.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that name the two holes
    # this file closed. A control that fails for some other reason is not the
    # disease reproduced, it is a different bug.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p2_d8_known_positive_fires",
                           "p4_both_named_fires",
                           "p9_ratchet_counts_every_class",
                           "p11_unrouted_owner_ask_is_reported",
                           "p12_vanished_owner_ask_is_the_known_positive",
                           "p13_expired_default_action_is_the_known_positive"}
                      <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
