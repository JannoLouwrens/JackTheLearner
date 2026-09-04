"""T0.31 — the backlog reader cannot be quieted by tidying the backlog.

`experiments/review_queue.py` is the reader `docs/REVIEW_QUEUE.md` did not have
for six days. The scar is dated: the Review's Sunday FULL run started
2026-08-30T06:37, died on `Reached max turns (60)` at 06:48 having written
nothing, and `w0-too-shallow`'s own status line — *"design owed by the Review
2026-08-30"* — went past its date in silence while two holds and four
gate-provisional specs sat behind it. The 52nd audit called a counter with an
age column *"the single highest-leverage instrument you could add this week"*.

An instrument added to make a backlog visible is worth exactly what it costs to
make it quiet again, so this battery is mostly about the four ways to quiet it.
Each one is a CONVERSION, and each is its own violation class:

    delete the rotting row                 -> VANISHED
    relabel it HELD (holds do not age)     -> HOLD-WITHOUT-A-CLOCK
    drop the DUE: line that went red       -> CLOCK-REMOVED
    hold it behind a blocker long resolved -> HOLD-ON-A-RESOLVED-BLOCKER
    stamp it ACTED naming no commit        -> ACTED-WITHOUT-A-COMMIT
    write the row as prose under a heading -> UNDECLARED-ROW

P14 is not a conversion — it is the pile, and it is the newest scar of all
(68th audit B7, 2026-09-04). The instrument already PRINTED the pile: on
2026-09-02 the builder read it, staggered `2026-09-06` from eighteen live rows
to five, in the open, with a stated reason per row. Two days later three
newly-routed rows had put it back to eight. Nothing was dishonest — each of the
three named a reason — but each router chose the date with no instrument
telling it the date was already full, so the file could only report the pile
AFTER it re-formed. P14 names the ACT instead of the symptom (`piled_on`) and
hands the next router the mechanical alternative (`next_free_due`), and it is
deliberately a METRIC rather than a violation: some of those rows chose a full
Sunday for good reasons, and a gate at zero would forbid a legal move — the
68th audit's own B3 discipline, applied to the instrument that audit wrote B7
about.

P15 is not a conversion either, and it is the newest scar of all (69th audit,
2026-09-04, B1). Every class above fires on a promise BREAKING — and the file
deliberately offers an honest way never to break one: re-arm the `DUE:` with a
reason, exactly as `decide_by` is re-armed. That hatch is right. Its unpriced
consequence is that **a desk which re-arms honestly forever is
byte-indistinguishable, to every per-row instrument here, from a desk that is
keeping up.** On the morning the audit ran, this file held 35 routed rows with
2 lifetime dispositions, 30 of them arrived in the previous seven days, and
`run review-queue` printed `0 violations` — correctly, because every class it
owned was local to one row while the divergence was a property of the SET.
P15 is the other half of the transaction: arrivals, disposals and the drain
projection, read from GIT rather than from the rows' own declared dates (a rate
that falls when you back-date a row is a rate that teaches back-dating), with
`DISPOSITIONED` counted separately and never credited as drain, because a design
that is still live and still ageing is not a row the desk has finished with.
Like P14 it is a METRIC and never a violation, and like P14 it is RATCHETED:
re-dating, re-arming and splitting a row must each leave it unimproved.

The sixth conversion is the newest scar (60th audit, 2026-09-02): six sections
written in the pre-declaration prose idiom — three with the declaration INSIDE
the heading, `## ROUTED: OPEN — ...`, one `## ` away from being read — were not
rows at all to the parser, so `run review-queue` printed 20 of the file's 26
and all six would have crossed the age bar with no number moving, because a
row the parser never saw cannot age. P13 pins that class at the audit's exact
count: the six historical shapes fire six, the heading is COUNTED and never
parsed, the honest repair (a real `ROUTED:` line under the heading) clears
exactly one, and the file's one legitimate prose heading (THE BUNDLING RULE
shape) stays exempt throughout.

The fifth conversion is the older scar (Review 09-01, item 4): on
`recipe-sensitivity`, `ACTED 2026-08-25` meant *a design exists* — the same
token that means *executed, commit named* on `me11-…` — and because ACTED is
terminal the row read closed for seven days while its spec stayed parked. The
repair is a distinguished LIVE status, `DISPOSITIONED` (design written,
execution owed), which ages and goes OVERDUE like OPEN, plus the rule that
`ACTED` must name its executing commit. P12 is that repair, executable in both
directions: the lazy relabel is flagged and does not lower the total; the
honest one — ACTED with the commit — clears the row and trips nothing.

This is the fourth instrument in this repo to be checked for the one-class
ratchet disease, and the first to be built with it already in mind: `coverage.py`
(closed by T0.21 P2), `decisions.py`'s `NO-DEFAULT` (T0.28 P9) and
`champions.py`'s `ARENA-MISSING` (T0.29 P2) each shipped counting a single class
and each paid a repair that LOWERED its own number. P4, P5 and P6 here are that
lesson applied before the fact: the total must not fall under any of the three
tidy-ups, and the property asserts on the TOTAL, not on the new class.

THE CONTROL IS NOT A PARAPHRASE — it is the reader that actually existed. Before
this module, the only machine-readable thing anyone had was the contract line in
`REVIEW_QUEUE.md` itself: `grep '^ROUTED:' docs/REVIEW_QUEUE.md`, i.e. a count of
rows. `_control` runs exactly that count against the same sabotages. It must be
blind to five of the six, and on the sixth — deletion — it must move in the WRONG
direction, reporting a smaller and therefore healthier-looking backlog for the
one edit the file's contract forbids outright. That is the disease, executable.

DATES ARE PINNED. Every fixture carries an explicit `today`, because a battery
whose verdict depends on the day it runs is a battery that will fail on a
Tuesday for reasons nobody can reconstruct.

NO ledger writes, no training, no world: documents are strings built in-process
and `audit()` takes its clock and its git baseline as arguments. Same shape as
T0.19, T0.20, T0.21, T0.28 and T0.29.

WHAT THIS DOES NOT CERTIFY. Whether a disposition was any good. An `ACTED`
that names a commit is taken at its word — whether that commit did the work is
the overseer's job and is done by reading, not by grepping. What the reader can
no longer do is accept a status token that means two things. And `MAX_OPEN_AGE_DAYS` is derived from the consumer's schedule (one
DAILY cycle plus the weekly Sunday FULL, plus a day of grace); if the Review's
cadence changes, that constant is stale and no property here can tell.

Deliberately declares NO `COVERS:` commitment. It guards the decision
machinery, not a capability.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..review_queue import (DOC_PATH, LOG_PATH, MAX_OPEN_AGE_DAYS,
                            THROUGHPUT_WINDOW_DAYS, VIOLATIONS, audit, check,
                            consumer_last_run, live_audit, parse, render,
                            throughput)

SPEC_ID = "T0.31"

# A claim about review_queue.py must die when review_queue.py changes: a
# certificate that survives edits to its own subject is a certificate about
# nothing (PG.6 hashing playground.py; T0.21 coverage.py; T0.28 decisions.py;
# T0.29 champions.py).
IMPL_DEPS = ["experiments/review_queue.py"]

N_PROPERTIES = 15

TODAY = _dt.date(2026, 9, 1)

# ── the fixture. One row per defect class, plus healthy rows that make a
#    flag-everything scanner fail as loudly as a flag-nothing one. ──

_ROWS = [
    # healthy: dispositioned long ago, and age must not touch it
    ("ok-acted", "2026-01-01", "ACTED 2026-01-02 (deadbeef)", []),
    # healthy: OPEN and young
    ("ok-young", "2026-08-30", "OPEN", []),
    # healthy: OPEN, ancient, but RE-ARMED into the future — the escape hatch
    ("ok-rearmed", "2026-01-01", "OPEN", ["DUE: 2026-12-01 | re-armed past the W1 design"]),
    # healthy: HELD behind a blocker that is still live
    ("ok-held", "2026-01-01", "HELD 2026-08-25 for the window", ["BLOCKED-BY: ok-young | the window it opens"]),
    # healthy: a young disposition — a design exists, execution owed, and the
    # status alone is no violation
    ("ok-dispositioned", "2026-08-30",
     "DISPOSITIONED 2026-08-31 (design in PROGRESS.md; execution owed)", []),
    # exactly ON the bar: not stale. A boundary that is wrong by one is a
    # boundary nobody can reason about.
    ("edge-at-bar", (TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS)).isoformat(), "OPEN", []),
    # OVERDUE: the w0-too-shallow case, dated and passed
    ("bad-overdue", "2026-08-24", "OPEN", ["DUE: 2026-08-30 | a design owed by a run that died"]),
    # STALE: open past a whole consumer cycle with no clock at all
    ("bad-stale", (TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS + 1)).isoformat(), "OPEN", []),
    # HOLD-ON-A-RESOLVED-BLOCKER: the window opened and the hold did not release
    ("bad-heldresolved", "2026-02-01", "HELD 2026-02-02 waiting", ["BLOCKED-BY: ok-acted | released when it acts"]),
    # STALE via DISPOSITIONED: the recipe-sensitivity scar, executable — routed
    # 08-20, design stamped 08-25, no clock, execution never started
    ("bad-disp-stale", (TODAY - _dt.timedelta(days=12)).isoformat(),
     "DISPOSITIONED 2026-08-25 (design written; execution owed)", []),
    # ACTED-WITHOUT-A-COMMIT: the two-meaning token itself — the word says
    # executed and the text names no executing commit
    ("bad-acted-nocommit", "2026-08-20",
     "ACTED 2026-08-25 (a design now sits in PROGRESS.md and work will follow)", []),
]


def _doc(rows) -> str:
    """Rows plus the prose hazards, rendered as the real file is."""
    out = ["# a fixture queue", ""]
    for rid, routed, status, decls in rows:
        out.append(f"ROUTED: {rid} | {routed} | src | {status}")
        out += [f"    {d}" for d in decls]
        # THE PROSE HAZARD, in every body: a date and the word ROUTED, indented,
        # in a sentence. Neither may open a row or set a clock.
        out.append("    Question: this was ROUTED: not-a-row and owed by 1999-01-01;")
        out.append("    the prose says a date and the reader must not believe it.")
        out.append("")
    return "\n".join(out)


DOC_FIXTURE = _doc(_ROWS)

# ── THE SIX, as shapes: the pre-declaration residue the 60th audit (2026-09-02)
#    found invisible. Three headers carry the declaration INSIDE the heading
#    (`## ROUTED: OPEN — ...`); two open with the prose idiom
#    `## ROUTED <date> (builder):`; one leads with a backticked id. Interleaved
#    with the shapes that must stay SILENT: the prose-rule heading (THE
#    BUNDLING RULE — a rule, not a row), a headingless declared row, a declared
#    row sitting directly under an undeclared section with nothing between
#    them, and the compliant new-style heading with its ROUTED: attached.
#    THE RATCHET IS 6 AND STAYS 6: a parser edit that reads 5 has gone blind
#    to a shape that was really written in the live file, and one that reads 7
#    has started flagging the file's legitimate prose. ──

_UNDECLARED_SIX = "\n".join([
    "# a fixture queue holding the pre-declaration residue", "",
    "## THE BUNDLING RULE — added by the Review on first use of this file", "",
    "Prose about sequencing. Not a row; it must never fire.", "",
    "ROUTED: ok-oldstyle | 2026-08-30 | src | OPEN",
    "    Question: a headingless declared row, the file's oldest shape.", "",
    "---", "",
    "## `six-backtick-slug` — a spec whose gates move in OPPOSITE directions",
    "## (builder, 2026-08-30; PARKED)", "",
    "Routed here by the spec's own pre-registered fork, in prose only.", "",
    "---", "",
    "## ROUTED 2026-08-30 (builder): six-prose-a's held-out split is saturated", "",
    "**Status: OPEN.** Gates provisional — a status only a human can read.", "",
    "---", "",
    "## ROUTED 2026-08-30 (builder): should six-prose-b count as an artifact", "",
    "**Status: OPEN.** No gate was moved.", "",
    "## ROUTED: OPEN — `six-heading-a`: the declaration is inside the heading",
    "## (builder, 2026-08-30)", "",
    "**The measurement.** One `## ` away from being read.", "",
    "ROUTED: ok-following | 2026-08-30 | src | OPEN",
    "    Question: a declared row directly below an undeclared section with no",
    "    hr between them — it must not be read as six-heading-a's declaration.", "",
    "## ROUTED: OPEN — `six-heading-b`: same shape, mid-file", "",
    "prose", "",
    "## ROUTED: OPEN — `six-heading-c`: same shape, last section in the file",
    "## (builder, 2026-08-31)", "",
    "prose", "",
    "## ROUTED 2026-09-01 (builder): `ok-newstyle` — candidate, declared, silent", "",
    "ROUTED: ok-newstyle | 2026-09-01 | src | OPEN",
    "    Question: the compliant shape every migrated row now uses.",
])


def _classes(a: dict) -> set:
    return {c for c, n in a["counts"].items() if n}


def _rows_of(doc: str) -> int:
    """THE CONTROL, executable: the only machine-readable reader that existed
    before 2026-08-31 — the contract line's own `grep '^ROUTED:' | wc -l`.
    """
    return sum(1 for line in doc.splitlines() if line.startswith("ROUTED:"))


# ── the six sabotages, each returning (name, sabotaged doc, prev doc) ──

def _sabotages() -> list[tuple[str, str, str]]:
    base = DOC_FIXTURE
    return [
        # 1. delete the rotting row entirely
        ("delete", _doc([r for r in _ROWS if r[0] != "bad-overdue"]), base),
        # 2. relabel it HELD — holds do not age
        ("relabel-held",
         _doc([(r[0], r[1], "HELD 2026-08-31 pending", []) if r[0] == "bad-stale" else r
               for r in _ROWS]), base),
        # 3. drop the DUE: that went red
        ("drop-due",
         _doc([(r[0], r[1], r[2], []) if r[0] == "bad-overdue" else r for r in _ROWS]), base),
        # 4. point the hold at a blocker that will never resolve... by naming
        #    one that does not exist
        ("phantom-blocker",
         _doc([(r[0], r[1], r[2], ["BLOCKED-BY: no-such-row | forever"])
               if r[0] == "bad-stale" else r for r in _ROWS]), base),
        # 5. break the row shape so a lazy parser skips it
        ("malform",
         base.replace("ROUTED: bad-overdue | 2026-08-24 | src | OPEN",
                      "ROUTED: bad-overdue | not-a-date | OPEN"), base),
        # 6. invent a status verb that no disposition rule covers
        ("invented-status",
         base.replace("ROUTED: bad-stale | ", "ROUTED: bad-stale | ").replace(
             f"| {(TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS + 1)).isoformat()} | src | OPEN",
             f"| {(TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS + 1)).isoformat()} | src | PARKED"),
         base),
        # 7. stamp the owed design ACTED without naming the executing commit —
        #    the two-meaning token, executable (Review 09-01 item 4)
        ("acted-no-commit",
         _doc([(r[0], r[1], "ACTED 2026-09-01 (executed, honestly)", []) if r[0] == "bad-disp-stale"
               else r for r in _ROWS]), base),
        # 8. write the new row as prose under a `## ` heading, declaring
        #    nothing — the pre-declaration idiom (60th audit): to the parser
        #    it is not a row at all, so it can never age or go red
        ("prose-row",
         base + "\n## ROUTED: OPEN — `bad-prose-row`: written the old way\n\n"
                "**Status: OPEN.** A status only a human can read.\n", base),
    ]


def _probe(blind: bool) -> dict:
    """`blind=True` runs the pre-2026-08-31 reader — a count of rows."""
    failed: list[str] = []
    base = audit(DOC_FIXTURE, None, TODAY)
    base_n = _rows_of(DOC_FIXTURE) if blind else base["total"]

    def total(doc: str, prev: str | None) -> float:
        return float(_rows_of(doc) if blind else audit(doc, prev, TODAY)["total"])

    # P1 — the LIVE document parses into well-formed rows, and every finding
    # names a row that exists. A battery that only ever sees its own fixture
    # cannot notice the real file drifting out of contract.
    live = audit(DOC_PATH.read_text(), None, TODAY)
    live_ids = {r["id"] for r in live["rows"]}
    if (live["n_rows"] < 8
            or any(r["fields"] != 4 or r["routed"] is None or r["status"] not in
                   ("OPEN", "HELD", "DISPOSITIONED", "ACTED", "DECLINED") for r in live["rows"])
            or live["counts"]["MALFORMED"]
            or any(rid not in live_ids for _c, rid, _w in live["findings"])
            or not consumer_last_run(LOG_PATH.read_text())):
        failed.append("p1_live_document_is_in_contract")

    # P2 — OVERDUE fires on a passed DUE and only on a passed DUE. Same row,
    # future date, must be silent: a clock that flags every dated row is a
    # clock that measures nothing.
    future = DOC_FIXTURE.replace("DUE: 2026-08-30 |", "DUE: 2026-12-30 |")
    over_here = {rid for c, rid, _ in base["findings"] if c == "OVERDUE"}
    over_there = {rid for c, rid, _ in audit(future, None, TODAY)["findings"] if c == "OVERDUE"}
    if blind or over_here != {"bad-overdue"} or over_there:
        failed.append("p2_overdue_fires_on_a_passed_date_only")

    # P3 — STALE is about age and nothing else: it fires one day past the bar,
    # is silent AT the bar, and never touches a terminal or re-armed row. A
    # DISPOSITIONED row ages exactly like an OPEN one — a design is not an
    # execution — so the aged one fires and the young one is silent.
    stale = {rid for c, rid, _ in base["findings"] if c == "STALE"}
    if blind or stale != {"bad-stale", "bad-disp-stale"}:
        failed.append("p3_stale_is_age_and_only_age")

    # P4 — THE RATCHET, conversion 1. Relabelling a stale row HELD must not
    # lower the total: HELD exempts a row from ageing, so it has to pay for the
    # exemption. This is `coverage.py`/`decisions.py`/`champions.py`'s disease,
    # asserted on the TOTAL rather than on any one class.
    name, doc, prev = _sabotages()[1]
    if total(doc, prev) < base_n:
        failed.append("p4_relabelling_held_does_not_help")

    # P5 — conversion 2. Deleting the row must not lower the total. The blind
    # reader fails this in the loudest possible way: its number FALLS, so the
    # one edit the file's contract forbids outright reads as an improvement.
    name, doc, prev = _sabotages()[0]
    if total(doc, prev) < base_n:
        failed.append("p5_deleting_the_row_does_not_help")

    # P6 — conversion 3. Dropping a DUE: that went red must not lower the total.
    name, doc, prev = _sabotages()[2]
    if total(doc, prev) < base_n:
        failed.append("p6_dropping_the_clock_does_not_help")

    # P7 — a hold must name something real and still unresolved. Behind a live
    # blocker: silent. Behind a dispositioned one: flagged. Behind a phantom:
    # MALFORMED, because a hold pointing at nothing is a hold forever.
    held_bad = {rid for c, rid, _ in base["findings"] if c == "HOLD-ON-A-RESOLVED-BLOCKER"}
    phantom = audit(_sabotages()[3][1], None, TODAY)
    if (blind or held_bad != {"bad-heldresolved"}
            or not any(c == "MALFORMED" and rid == "bad-stale" for c, rid, _ in phantom["findings"])
            or phantom["total"] < base["total"]):
        failed.append("p7_a_hold_names_a_live_blocker")

    # P8 — malformation is RECORDED, not raised, and an invented status verb is
    # malformation. A parser that dies on the first bad line reports one defect
    # and hides the rest, in a file four organs write to.
    mal = audit(_sabotages()[4][1], None, TODAY)
    inv = audit(_sabotages()[5][1], None, TODAY)
    if (blind or not mal["counts"]["MALFORMED"] or not inv["counts"]["MALFORMED"]
            or mal["n_rows"] != len(_ROWS) or inv["n_rows"] != len(_ROWS)
            or mal["total"] < base["total"] or inv["total"] < base["total"]):
        failed.append("p8_malformation_is_recorded_not_raised")

    # P9 — prose cannot open a row or set a clock. Every fixture body contains
    # an indented `ROUTED:` and the date 1999-01-01; both are invisible.
    # `champions.py` shipped a regex over prose and one seat's arena turned out
    # to be the words OUT LOUD (901f7fc).
    if blind or base["n_rows"] != len(_ROWS) or "1999" in str(base["findings"]):
        failed.append("p9_prose_opens_no_row_and_sets_no_clock")

    # P10 — an ABSENT baseline manufactures nothing. With no previous revision
    # (a new file, a shallow clone, no repo) the two git-baselined classes must
    # be silent rather than firing on every row at once.
    none_base = audit(DOC_FIXTURE, None, TODAY)
    both = audit(DOC_FIXTURE, DOC_FIXTURE, TODAY)
    if (blind or none_base["counts"]["VANISHED"] or none_base["counts"]["CLOCK-REMOVED"]
            or both["total"] != none_base["total"]):
        failed.append("p10_an_absent_baseline_accuses_nobody")

    # P11 — every class in VIOLATIONS is reachable, and the renderer prints
    # every finding it was given. A class that nothing can trigger is a class
    # the exit code cannot see, and a finding the report drops is a finding
    # nobody reads.
    reached = set()
    for _n, doc, prev in _sabotages():
        reached |= _classes(audit(doc, prev, TODAY))
    reached |= _classes(base)
    text = render(base, "2026-08-29")
    if (blind or reached != set(VIOLATIONS)
            or any(rid not in text for _c, rid, _w in base["findings"])
            or "consumer last ran 2026-08-29" not in text):
        failed.append("p11_every_class_is_reachable_and_reported")

    # P12 — a disposition is not an execution (the 2026-09-01 scar: `ACTED` on
    # `recipe-sensitivity` meant "a design exists", the row read closed, and
    # the spec it parked stayed parked for seven days). Three conjuncts, both
    # directions: (i) the aged DISPOSITIONED row is red and the commitless
    # ACTED row is flagged as its own class; (ii) the LAZY relabel —
    # DISPOSITIONED -> ACTED with no commit named — converts STALE into
    # ACTED-WITHOUT-A-COMMIT and must not lower the total; (iii) the HONEST
    # repair — the same relabel WITH the executing commit — clears exactly the
    # one repaired finding and trips nothing, because an escape hatch that is
    # also red is not a hatch.
    nocommit = {rid for c, rid, _ in base["findings"] if c == "ACTED-WITHOUT-A-COMMIT"}
    lazy = audit(_sabotages()[6][1], None, TODAY)
    honest_doc = _doc([(r[0], r[1], "ACTED 2026-09-01 (executed in deadbee1)", [])
                       if r[0] == "bad-disp-stale" else r for r in _ROWS])
    honest = audit(honest_doc, None, TODAY)
    if (blind
            or "bad-disp-stale" not in stale
            or nocommit != {"bad-acted-nocommit"}
            or total(_sabotages()[6][1], _sabotages()[6][2]) < base_n
            or not any(c == "ACTED-WITHOUT-A-COMMIT" and rid == "bad-disp-stale"
                       for c, rid, _ in lazy["findings"])
            or honest["total"] != base["total"] - 1
            or any(rid == "bad-disp-stale" for _c, rid, _w in honest["findings"])):
        failed.append("p12_a_disposition_is_not_an_execution")

    # P13 — a row written as prose under a heading is COUNTED, never parsed
    # (60th audit: 6 of the file's 26 rows were invisible, three with the
    # declaration inside the heading, and an invisible row cannot age). Four
    # conjuncts: (i) THE RATCHET — the six historical shapes fire exactly 6
    # UNDECLARED-ROW, pinned at the audit's count; (ii) counting is not
    # parsing — none of the six opens a row (only the three declared ok-* rows
    # parse) and nothing in the findings quotes the heading's OPEN as a
    # status; (iii) the honest repair — a real ROUTED: line under the heading
    # — clears exactly one and the migrated row then parses like any other;
    # (iv) the exempt shapes (the prose-rule heading, the attached new-style
    # heading) are silent in both documents, so the class cannot be quieted by
    # deleting legitimate prose nor inflated by flagging it.
    six = audit(_UNDECLARED_SIX, None, TODAY)
    migrated = _UNDECLARED_SIX.replace(
        "## (builder, 2026-08-30)\n\n**The measurement.**",
        "## (builder, 2026-08-30)\n\n"
        "ROUTED: six-heading-a | 2026-08-30 | src | OPEN\n\n**The measurement.**")
    mig = audit(migrated, None, TODAY)
    if (blind
            or six["counts"]["UNDECLARED-ROW"] != 6
            or six["n_rows"] != 3
            or any("OPEN" in r["status"] for r in six["rows"] if r["id"].startswith("six"))
            or mig["counts"]["UNDECLARED-ROW"] != 5
            or "six-heading-a" not in {r["id"] for r in mig["rows"]}
            or mig["n_rows"] != 4):
        failed.append("p13_a_prose_row_is_counted_never_parsed")

    # P14 — the pile is reported as an ACT, not only as a symptom, and the
    # report cannot be tidied by moving a promise between piles. The scar is
    # two days old: on 2026-09-02 the builder staggered 2026-09-06 from
    # eighteen live rows to five, in the open, with a reason per row — and by
    # 09-04 three newly-routed rows had put it back to eight, because each
    # router chose the date with nothing telling it the date was full. Five
    # conjuncts: (i) a row dated onto a day that already carried CAPACITY live
    # rows when it was ROUTED is named in `piled_on`, and the row that got
    # there first is NOT — an instrument that blames the whole day blames the
    # innocent; (ii) it is a METRIC and never a violation, so adding a
    # perfectly healthy row onto a full day moves `piled_on` and leaves
    # `total` and every violation class exactly where they were (68th audit
    # B3: report a number, do not gate at zero); (iii) THE RATCHET — re-dating
    # a piled row onto ANOTHER full day does not lower the count, because
    # moving a promise between piles is not a repair; (iv) moving it to a day
    # under capacity DOES clear it, so the honest escape hatch works;
    # (v) `next_free_due` names a future date that really carries no live
    # promise, which is the mechanical alternative to defaulting onto Sunday.
    # `other-day` is routed BEFORE `second` on purpose: the sideways re-date in
    # (iii) must land on a day whose occupant got there first, or the metric's
    # documented conservatism (a re-arm is timestamped by the row's ROUTED
    # date, which can only under-count) would clear the row for the wrong
    # reason and the ratchet would be testing the wrong thing.
    pile_rows = [
        ("first", "2026-08-20", "OPEN", ["DUE: 2026-09-06 | got there first"]),
        ("second", "2026-08-25", "OPEN", ["DUE: 2026-09-06 | dated onto a full day"]),
        ("other-day", "2026-08-21", "OPEN", ["DUE: 2026-09-20 | its own day, occupied early"]),
    ]
    pile = audit(_doc(pile_rows), None, TODAY)
    piled_ids = {p["id"] for p in pile["piled_on"]}
    # (ii) a healthy row added onto the full day: metric moves, violations do not
    plus = audit(_doc(pile_rows + [("third", "2026-08-27", "OPEN",
                                    ["DUE: 2026-09-06 | also onto the full day"])]),
                 None, TODAY)
    # (iii) re-date `second` onto the OTHER occupied day — still piled
    sideways = audit(_doc([(r[0], r[1], r[2], ["DUE: 2026-09-20 | moved to another pile"])
                           if r[0] == "second" else r for r in pile_rows]), None, TODAY)
    # (iv) re-date `second` onto an empty day — cleared
    honest_move = audit(_doc([(r[0], r[1], r[2], ["DUE: 2026-10-15 | moved to a free day"])
                              if r[0] == "second" else r for r in pile_rows]), None, TODAY)
    free = pile["next_free_due"]
    if (blind
            or piled_ids != {"second"}
            or plus["total"] != pile["total"] or plus["counts"] != pile["counts"]
            or len(plus["piled_on"]) != 2
            or {p["id"] for p in sideways["piled_on"]} != {"second"}
            or honest_move["piled_on"]
            or not free
            or _dt.date.fromisoformat(free) <= TODAY
            or pile["due_pile"].get(free, 0) != 0):
        failed.append("p14_a_promise_dated_onto_a_full_day_is_named")

    # P15 — the OTHER HALF OF THE TRANSACTION. Every class above fires on a
    # promise breaking; the honest escape hatch (re-arm the DUE: with a reason)
    # means a desk can be indistinguishable from one that is keeping up while
    # disposing of nothing, which is what 2026-09-04 measured: 35 rows, 2
    # lifetime dispositions, 30 arrivals in seven days, `0 violations`. Eight
    # conjuncts.
    #
    # `then` is the window's git baseline; `now` is the working tree. Row `f`
    # is ACTED in BOTH, so a disposal already banked before the window cannot
    # be credited a second time. Row `b` is the only real disposal.
    then_rows = [
        ("a", "2026-08-01", "OPEN", []),
        ("b", "2026-08-02", "OPEN", []),
        ("c", "2026-08-03", "OPEN", []),
        ("f", "2026-07-01", "ACTED 2026-07-02 (cafe123)", []),
    ]
    now_rows = [
        ("a", "2026-08-01", "OPEN", []),
        ("b", "2026-08-02", "ACTED 2026-08-30 (beadfed)", []),
        ("c", "2026-08-03", "OPEN", []),
        ("f", "2026-07-01", "ACTED 2026-07-02 (cafe123)", []),
        ("d", "2026-08-29", "OPEN", []),
        ("e", "2026-08-30", "OPEN", []),
    ]
    then_doc, now_doc = _doc(then_rows), _doc(now_rows)
    W = THROUGHPUT_WINDOW_DAYS
    t = throughput(now_doc, then_doc)

    # (i) the arithmetic itself, and the UNBOUNDED verdict when arrivals win
    base_ok = (t is not None and t["arrived"] == 2 and t["disposed"] == 1
               and t["designed"] == 0 and t["live_now"] == 4
               and t["net_arrivals"] == 1 and t["unbounded"]
               and t["drain_cycles"] is None
               and abs(t["arrived_per_cycle"] - 2 / W) < 1e-9)

    # (ii) UNBOUNDED IS NOT BY CONSTRUCTION. A desk that really disposes must
    # get a FINITE drain, or the metric is a slogan: here `a` and `b` both
    # close and nothing arrives, leaving one live row draining at 2/W per
    # cycle. A number that can only ever read one way measures nothing —
    # law 2, applied to a metric instead of to an experiment.
    keeping_up = throughput(
        _doc([(r[0], r[1], "ACTED 2026-08-30 (beadfed)", r[3])
              if r[0] in ("a", "b") else r for r in then_rows]), then_doc)
    positive_ok = (keeping_up is not None and keeping_up["arrived"] == 0
                   and keeping_up["disposed"] == 2
                   and not keeping_up["unbounded"]
                   and keeping_up["drain_cycles"] is not None
                   and abs(keeping_up["drain_cycles"] - 1 / (2 / W)) < 1e-9)

    # (iii) A DESIGN IS NOT A DISPOSAL — the two-meaning token (Review 09-01
    # item 4) reborn as a rate. `b` becomes DISPOSITIONED: it is still LIVE and
    # still ageing, so `disposed` stays 0, drain stays UNBOUNDED, and the row
    # shows up in `designed` where a reader can see it without it being credited.
    design_only = throughput(
        _doc([(r[0], r[1], "DISPOSITIONED 2026-08-30 (design written)", r[3])
              if r[0] == "b" else r for r in then_rows]), then_doc)
    design_ok = (design_only is not None and design_only["disposed"] == 0
                 and design_only["designed"] == 1
                 and design_only["live_now"] == 3
                 and design_only["unbounded"])

    # (iv) THE RATCHET, back-dating: `d`'s declared `routed` date is rewritten
    # to long before the window opened. Arrivals are read from git, so the
    # number cannot move — a rate that falls when you back-date a row is a rate
    # that teaches back-dating.
    backdated = throughput(
        _doc([("d", "2020-01-01", r[2], r[3]) if r[0] == "d" else r
              for r in now_rows]), then_doc)
    backdate_ok = (backdated is not None
                   and backdated["arrived"] == t["arrived"]
                   and backdated["net_arrivals"] == t["net_arrivals"])

    # (v) THE RATCHET, re-arming: every honest hatch in this file — a new DUE:,
    # a re-armed date — leaves throughput identical, because none of them
    # disposes of anything.
    rearmed = throughput(
        _doc([(r[0], r[1], r[2], ["DUE: 2026-12-01 | re-armed, with a reason"])
              for r in now_rows]), then_doc)
    rearm_ok = rearmed == t

    # (vi) THE RATCHET, splitting: one live row written as two must never
    # improve the reading. It adds an arrival and a live row, so the net can
    # only get worse.
    split = throughput(_doc(now_rows + [("c-b", "2026-09-01", "OPEN", [])]),
                       then_doc)
    split_ok = (split is not None
                and split["net_arrivals"] > t["net_arrivals"]
                and split["live_now"] > t["live_now"]
                and split["unbounded"])

    # (vii) A METRIC, NEVER A VIOLATION (P14's discipline, 68th audit B3):
    # supplying the baseline must not move `total` or any violation class.
    with_base = audit(now_doc, None, TODAY, base_doc=then_doc)
    without = audit(now_doc, None, TODAY)
    metric_ok = (with_base["total"] == without["total"]
                 and with_base["counts"] == without["counts"]
                 and with_base["throughput"] is not None)

    # (viii) AN ABSENT BASELINE REPORTS NOTHING, not something good. The
    # git-baselined violation classes refuse to ACCUSE without a baseline; this
    # one must refuse to EXONERATE without one, and `None` must survive into
    # the rendered report as a stated fault rather than as a clean week.
    no_base_ok = (without["throughput"] is None
                  and "UNMEASURED" in render(without))

    if (blind or not base_ok or not positive_ok or not design_ok
            or not backdate_ok or not rearm_ok or not split_ok
            or not metric_ok or not no_base_ok):
        failed.append("p15_the_desk_is_measured_disposing_not_only_breaking")

    # The live desk's own numbers, recorded in the ledger row so the reading
    # that motivated P15 is dated and attributable rather than quoted from an
    # audit page. `-1` is the honest value for "no git baseline in this
    # checkout" — a sentinel, said out loud, rather than a zero that would read
    # as a clean week.
    live_t = live_audit()["throughput"]
    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "fixture_violations": float(base["total"]),
        "live_rows": float(live["n_rows"]),
        "live_open": float(live["n_open"]),
        "live_violations": float(live["total"]),
        "live_oldest_days": float(live["oldest_live_days"]),
        "live_arrived": float(live_t["arrived"]) if live_t else -1.0,
        "live_disposed": float(live_t["disposed"]) if live_t else -1.0,
        "live_net_arrivals": float(live_t["net_arrivals"]) if live_t else -1.0,
    }


def _experiment(seed: int) -> dict:
    return _probe(blind=False)


def _control(seed: int) -> dict:
    """The reader that existed before this module: `grep -c '^ROUTED:'`.

    Not a paraphrase and not a crippled copy — it is the literal instrument the
    file's own contract published, and for six days it was the whole of the
    tooling. It counts rows. It therefore cannot see a passed date, an aged row,
    a laundered hold, a phantom blocker, a malformed line or an invented status,
    and on the one sabotage it CAN see it reports the wrong sign: delete the
    rotting row and the number falls, so the backlog looks healthier.

    Measured: it fails 12 of 15, and the 3 it passes are worth naming so nobody
    reads this as a straw man. P1 is a statement about the live DOCUMENT rather
    than about the reader, and is not asked of it. P4 and P6 it passes
    VACUOUSLY — relabelling a row `HELD` and deleting its `DUE:` both leave the
    row count at 8, so a blind instrument is trivially invariant under the two
    sabotages it also cannot detect. Invariance without detection is not the
    property; that is why P4 and P6 are not what `_check` requires of it.

    It must fail P2 (the dated promise), P5 (the deletion, where its number
    moves the wrong way), P11 (the classes it cannot name) and P12 (a row count
    cannot tell a design from an execution — the exact blindness that parked
    UB.10 for a week). P13 it also fails, definitionally: a `grep '^ROUTED:'`
    IS the parser the 60th audit caught reading 20 of 26 — the prose rows are
    precisely the ones it cannot count. If it ever passes those, this spec is
    guarding a distinction that did not need making.

    P15 it fails for the reason the 69th audit exists: a count of rows has no
    notion of a row's STATUS, so it cannot tell an arrival from a disposal, and
    the one signal it does carry moves the wrong way again — a desk that closes
    a row and a desk that deletes it are the same falling number to it.
    """
    return _probe(blind=True)


def _check(m: dict, c: dict) -> Status | bool:
    # Every property ran AND every property held. Gating on `properties_failed
    # == 0` alone lets a battery that stopped early read as clean — T0.13's own
    # first bug, and every T0.1x/T0.2x battery since.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that name what it cannot
    # do. A control that fails for some other reason is not the disease
    # reproduced, it is a different bug.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p2_overdue_fires_on_a_passed_date_only",
                           "p5_deleting_the_row_does_not_help",
                           "p11_every_class_is_reachable_and_reported",
                           "p12_a_disposition_is_not_an_execution",
                           "p14_a_promise_dated_onto_a_full_day_is_named",
                           "p15_the_desk_is_measured_disposing_not_only_breaking"
                           } <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
