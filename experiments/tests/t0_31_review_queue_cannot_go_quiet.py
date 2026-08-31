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

WHAT THIS DOES NOT CERTIFY. Whether a DISPOSITION was any good. `ACTED
2026-08-25` is the Review's word about its own work and this module takes it —
the audit of dispositions is the overseer's job and is done by reading, not by
grepping. And `MAX_OPEN_AGE_DAYS` is derived from the consumer's schedule (one
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
from ..review_queue import (DOC_PATH, LOG_PATH, MAX_OPEN_AGE_DAYS, VIOLATIONS,
                            audit, check, consumer_last_run, parse, render)

SPEC_ID = "T0.31"

# A claim about review_queue.py must die when review_queue.py changes: a
# certificate that survives edits to its own subject is a certificate about
# nothing (PG.6 hashing playground.py; T0.21 coverage.py; T0.28 decisions.py;
# T0.29 champions.py).
IMPL_DEPS = ["experiments/review_queue.py"]

N_PROPERTIES = 11

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
    # exactly ON the bar: not stale. A boundary that is wrong by one is a
    # boundary nobody can reason about.
    ("edge-at-bar", (TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS)).isoformat(), "OPEN", []),
    # OVERDUE: the w0-too-shallow case, dated and passed
    ("bad-overdue", "2026-08-24", "OPEN", ["DUE: 2026-08-30 | a design owed by a run that died"]),
    # STALE: open past a whole consumer cycle with no clock at all
    ("bad-stale", (TODAY - _dt.timedelta(days=MAX_OPEN_AGE_DAYS + 1)).isoformat(), "OPEN", []),
    # HOLD-ON-A-RESOLVED-BLOCKER: the window opened and the hold did not release
    ("bad-heldresolved", "2026-02-01", "HELD 2026-02-02 waiting", ["BLOCKED-BY: ok-acted | released when it acts"]),
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
                   ("OPEN", "HELD", "ACTED", "DECLINED") for r in live["rows"])
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
    # is silent AT the bar, and never touches a dispositioned or re-armed row.
    stale = {rid for c, rid, _ in base["findings"] if c == "STALE"}
    if blind or stale != {"bad-stale"}:
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

    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "fixture_violations": float(base["total"]),
        "live_rows": float(live["n_rows"]),
        "live_open": float(live["n_open"]),
        "live_violations": float(live["total"]),
        "live_oldest_days": float(live["oldest_live_days"]),
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

    Measured: it fails 8 of 11, and the 3 it passes are worth naming so nobody
    reads this as a straw man. P1 is a statement about the live DOCUMENT rather
    than about the reader, and is not asked of it. P4 and P6 it passes
    VACUOUSLY — relabelling a row `HELD` and deleting its `DUE:` both leave the
    row count at 8, so a blind instrument is trivially invariant under the two
    sabotages it also cannot detect. Invariance without detection is not the
    property; that is why P4 and P6 are not what `_check` requires of it.

    It must fail P2 (the dated promise), P5 (the deletion, where its number
    moves the wrong way) and P11 (the classes it cannot name). If it ever
    passes those, this spec is guarding a distinction that did not need making.
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
                           "p11_every_class_is_reachable_and_reported"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
