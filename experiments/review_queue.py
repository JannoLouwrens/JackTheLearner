"""A backlog nobody can count is indistinguishable from a backlog nobody has.

WHY THIS EXISTS. `docs/REVIEW_QUEUE.md` was built on 2026-08-24 (27th audit, B2)
because *"routed to Review"* was a phrase in commit messages with no file, so the
backlog was invisible — nothing could print *"3 routed, 0 acted on, oldest 4
days"*. It succeeded at holding the rows and it never gained a reader. Six days
later the failure recurred one layer up:

    2026-08-30 06:37   review.sh starts in FULL mode — the only mode that does
                       Part 2 and the run that owed `w0-too-shallow`'s design.
    2026-08-30 06:48   dies on `Reached max turns (60)`, 11 minutes into a
                       40-minute budget, having written NOTHING.
    2026-08-31 00:45   52nd audit finds it. `w0-too-shallow` still says
                       "design owed by the Review 2026-08-30" — a date that had
                       passed — and no number anywhere had gone red.

That row's promise was made in the open, dated, and broken in silence, while four
experiments (`DP.04`, `SH.02`, `SM.03`, `T2.11`) sat gate-provisional waiting on
it and both GPU cost classes read EMPTY because of it. The 27th audit's own
corollary had already been written and never built: *"an organ that is the
destination of routed work must have liveness watched by something other than
itself."* `scripts/lib_liveness.sh:review_liveness` (52nd audit, B1) now watches
whether the CONSUMER ran. This file watches whether the WORK MOVED — the two are
different failures and neither implies the other. A desk can open every morning
and dispose of nothing.

THE DECLARATION, not a regex over prose. `champions.py` learned this the
expensive way on 2026-08-31 (`901f7fc`): a seat's ring was inferred by regex and
one seat's arena turned out to be the words OUT LOUD. So every fact this module
gates on is DECLARED, at the start of a line, in the `DECIDE:`/`COVERS:` idiom:

    ROUTED: <id> | <YYYY-MM-DD> | <source> | <STATUS ...>
        DUE: <YYYY-MM-DD> | what is owed, and by whom
        BLOCKED-BY: <another row id> | what releases this hold

The `ROUTED:` line's four-field shape is the file's own published contract and is
unchanged; `DUE:` and `BLOCKED-BY:` are new indented body lines, so every reader
that greps `^ROUTED:` keeps working. Prose dates are NOT read. `w0-too-shallow`'s
prose date was migrated into a `DUE:` line by hand, once, in the commit that
added this module — a migration is a human act, an inference is a bug.

THE FIVE STATUSES, from the file's own contract plus the two the consumer's
practice forced: `OPEN`, `HELD` (the bundling rule, 2026-08-25),
`DISPOSITIONED`, `ACTED`, `DECLINED`. `ACTED`/`DECLINED` are TERMINAL and
exempt from everything below; `OPEN`, `HELD` and `DISPOSITIONED` are LIVE.

`DISPOSITIONED` exists because one token carried two meanings and the cheap
reading won (Review 09-01, FOR THE BUILDER item 4). On `recipe-sensitivity`
the Review stamped `ACTED 2026-08-25` meaning *a design now exists*; on
`me11-…` the same word meant *the builder executed one, commit named*. ACTED
is terminal here, so the first sense closed a row whose work had not started —
`run review-queue` printed `ACTED 12 d` and no violation while the design sat
unexecuted and a frees-4 spec (`UB.10`) stayed parked for seven days. So the
intermediate state gets its own word: a design without an executing commit is
`DISPOSITIONED`, and it KEEPS AGEING — it can go STALE and OVERDUE exactly
like `OPEN`, because owed execution is live work whatever the design's
quality. `ACTED` now means executed, and must NAME the executing commit in
its status text (>=7 hex chars containing a letter, so a bare date-stamp
cannot pass as one); an `ACTED` naming no commit is the two-meaning token
reborn and is its own violation, `ACTED-WITHOUT-A-COMMIT`.

THE PRE-DECLARATION RESIDUE IS COUNTED, NEVER PARSED (60th audit, 2026-09-02).
Replacing the prose convention with a declaration syntax made the un-migrated
rows INVISIBLE rather than untidy: six sections written in the pre-declaration
idiom — each under its own `## ` heading, each saying "Status: OPEN" or
"ROUTED: OPEN" in its own words, three carrying the declaration INSIDE the
heading (`## ROUTED: OPEN — ...`, one `## ` away from being read) — were not
rows at all to this parser. `run review-queue` printed 20 of the file's 26,
and all six would have crossed MAX_OPEN_AGE_DAYS with no number moving,
because a row the parser never saw cannot age. `decisions.py` and
`champions.py` both ship an UNDECLARED class for exactly this residue; this
module alone had none. So: a `## ` heading that ANNOUNCES a row — it opens
with the word ROUTED or with a backticked row id — is a CANDIDATE, and a
candidate whose heading block is not immediately followed (blank lines aside)
by a real column-0 `ROUTED:` declaration is `UNDECLARED-ROW`. The heading is
never parsed into a row: a declaration inside a heading will keep being
written, and counting it is the guard — reading it would be the champions.py
regex mistake (`901f7fc`) again. Known limit, stated so nobody discovers it
as a scar: a routed section whose heading carries neither marker is still
invisible; the two marker shapes are the two that have actually been written,
and the T0.31 fixture pins both at the audit's count of six.

THE RATCHET COUNTS EVERY CLASS, because counting one is how the other three
instruments in this repo were gamed by accident (`coverage.py`, closed by
`T0.21` P2; `decisions.py`'s `NO-DEFAULT`, closed by `T0.28` P9;
`champions.py`'s `ARENA-MISSING`, closed by `T0.29` P2 — three instruments, one
disease, found three separate times). Here the conversions a tidy-up would
reach for are named and each is its own violation:

    write the new row as prose under
      a heading, declaring nothing       -> UNDECLARED-ROW  (an invisible row
                                                             cannot age)
    delete a rotting row                 -> VANISHED        (rows are never
                                                             deleted: the file's
                                                             contract, T1.02)
    relabel OPEN as HELD                 -> HOLD-WITHOUT-A-CLOCK  (a hold must
                                                             pay for its
                                                             exemption)
    drop the DUE line that went red      -> CLOCK-REMOVED
    hold forever behind a blocker that   -> HOLD-ON-A-RESOLVED-BLOCKER
      was resolved months ago

VANISHED and CLOCK-REMOVED are computed against the PREVIOUS COMMITTED revision
of the file, so the baseline is git and there is no baseline constant to edit.

THE ESCAPE HATCH IS RE-ARMING, IN THE OPEN. A red `STALE` or `OVERDUE` row has
three honest repairs and no dishonest one: do the work (`ACTED`, naming the
executing commit), refuse it (`DECLINED`), or move the date by writing a new
`DUE:` with a reason — exactly
what `SYSTEM.md` already blesses for `decide_by` ("answer `D1` and `D10`, or
re-arm both past the W1 design"). What it must not be able to do is go quiet.

WHAT THIS DOES NOT WATCH, stated so no reader mistakes the scope. It does not
know whether a disposition was any GOOD — an `ACTED` that names a commit is
taken at its word; whether that commit did the work is the overseer's to read.
What it can no longer do is mean two things: the commitless form is a violation
and the design-only form has its own live status. It does not read `PROGRESS.md`. It reports the
consumer's last-run date off `docs/PROGRESS_LOG.md` for context and deliberately
does not gate on it: `review_liveness` owns that alarm, and two organs owning one
fact is how a number ends up watched by nobody in particular.

THROUGHPUT: THE HALF OF THE TRANSACTION THIS MODULE COULD NOT SEE (69th audit,
2026-09-04, B1). Every violation class above fires on a promise BREAKING, and
the file deliberately provides an honest way never to break one — re-arm with a
new `DUE:` and a reason, exactly as `decide_by` is re-armed. That hatch is
right; a deadline that cannot move when the work genuinely changes is a deadline
that gets deleted instead. But it has a consequence nobody priced: **a desk that
re-arms honestly forever is byte-indistinguishable, to every per-row instrument,
from a desk that is keeping up.** On 2026-09-04 this file held 35 routed rows
with 2 lifetime dispositions, 30 of them arrived in the previous seven days —
and this module printed `0 violations`, correctly, because every class it owns
is local to one row and the divergence is a property of the SET.

So the reader now also counts the other direction, over a trailing window, from
the file's own git history:

    arrived   ids present now and absent at the window's baseline revision
    disposed  ids TERMINAL now that were not TERMINAL at the baseline
    designed  ids DISPOSITIONED now that were not DISPOSITIONED then
    drain     live rows / (disposed - arrived) per cycle, or UNBOUNDED

Four properties of that design are load-bearing and each is a scar somewhere
else in this repo:

**It is a METRIC and never a violation** — the same discipline `piled_on` was
given one layer down (68th audit B3). A slow week is legal, the consumer is a
colleague, and a gate here would forbid a legal move.

**Arrivals and disposals are read from GIT, not from the declared `routed`
date.** The declared date is the file's contract and is trusted everywhere else
here — but it is writable, and a metric that improves when you back-date a row
is a metric that teaches back-dating. Git is the baseline `VANISHED` and
`CLOCK-REMOVED` already use, and nothing a working-tree edit can do reaches it.

**A DISPOSITIONED transition is NOT credited as a disposal**, which departs from
the letter of the audit's request (it named `ACTED`/`DECLINED`/`DISPOSITIONED`
together) for the reason that audit's own file records: `DISPOSITIONED` is LIVE
and keeps ageing, so counting it as drain would let a desk report throughput by
writing designs it never executes — the two-meaning token (Review 09-01 item 4)
reborn as a rate. It is counted and printed separately, as `designed`, so a
design-only desk is visible without being credited.

**An absent baseline reports nothing rather than something good.** The
git-baselined violation classes above refuse to ACCUSE without a baseline; this
one refuses to EXONERATE without one. `throughput` is `None`, the ratchet
counter `review_queue_net_arrivals` goes LOST in `run status`, and LOST is a
fault there, not a quiet day.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "REVIEW_QUEUE.md"
LOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "PROGRESS_LOG.md"

#: An OPEN row survives at most one full consumer cycle before it is rotting.
#: DERIVED, not chosen: the Review runs DAILY with exactly one FULL run per week
#: (Sundays), and the FULL run is the only mode that does Part 2, so a row whose
#: repair is a redesign can wait 7 days for its window. One day of grace for a
#: run that starts late. A row older than this has outlived the consumer's whole
#: schedule, which is the condition `w0-too-shallow` was in when nothing noticed.
MAX_OPEN_AGE_DAYS = 8

LIVE = ("OPEN", "HELD", "DISPOSITIONED")
TERMINAL = ("ACTED", "DECLINED")
STATUSES = LIVE + TERMINAL

#: The consumer's only MEASURED discharge capacity, in rows per cycle: the
#: 2026-08-30 FULL run died at eleven minutes owing exactly ONE dated row
#: (`w0-too-shallow`), and no cycle since has demonstrably discharged more.
#: A due date carrying more live rows than this is a pile of promises
#: scheduled to break together — worth predicting before Sunday rather than
#: reporting on Monday (65th audit B6, written the week seven rows shared
#: 2026-09-06). Raise it only by citing a cycle that actually discharged N.
MEASURED_DISCHARGE_CAPACITY = 1

#: One consumer cycle, in days. DERIVED: `review.sh` runs DAILY, so a day is
#: the finest cadence at which this desk can dispose of anything, and it is the
#: unit `MEASURED_DISCHARGE_CAPACITY` is already denominated in (a due DATE
#: carrying more than one live row is the pile).
CONSUMER_CYCLE_DAYS = 1

#: The trailing window the throughput reading is computed over, in days.
#: DERIVED, and deliberately NOT the audit's suggested three cycles: the
#: consumer's schedule is one DAILY run plus exactly one FULL run per week, and
#: the FULL run is the only mode that does Part 2 — so a window shorter than
#: seven days cannot contain the mode that discharges a redesign row, and would
#: read UNBOUNDED for a desk that is in fact on schedule. Seven days is the
#: shortest window in which every row in this file could honestly have been
#: disposed of. It is also the window the Review and the 69th audit both
#: computed by hand ("30 arrived in the last 7"), so the number is comparable
#: to the one two organs already published.
THROUGHPUT_WINDOW_DAYS = 7

#: How far ahead `next_free_due` will look for a date carrying no promise yet.
#: A board booked solid past a full quarter should print "" and say so rather
#: than scan forever; the empty string is a reading, not a failure.
FREE_DATE_HORIZON_DAYS = 120

#: How many `piled_on` rows the report prints, worst-crowded first. The rest
#: are counted out loud in the elision line — a cap that hides its own size
#: reads as coverage.
PILED_ON_SHOWN = 8

#: Every violation class, named once. `--check` is red on any of them; a class
#: absent from this tuple is a class the exit code cannot see.
VIOLATIONS = ("MALFORMED", "OVERDUE", "STALE", "HOLD-WITHOUT-A-CLOCK",
              "HOLD-ON-A-RESOLVED-BLOCKER", "VANISHED", "CLOCK-REMOVED",
              "ACTED-WITHOUT-A-COMMIT", "UNDECLARED-ROW")

_ROUTED = re.compile(r"^ROUTED:\s*(.*)$")
#: A `## ` heading that ANNOUNCES a routed row. Two shapes have actually been
#: written (60th audit): the prose idiom `## ROUTED 2026-08-30 (builder): ...`
#: / `## ROUTED: OPEN — ...`, and the backticked-id lead `## \`t310-...\` — `.
#: A prose-rule heading (THE BUNDLING RULE) matches neither and is exempt —
#: this file legitimately holds prose that is not a row.
_CANDIDATE = re.compile(r"^##\s*(?:ROUTED\b|`)")
_HEADING = re.compile(r"^#{1,6}\s")
_DECL = re.compile(r"^(DUE|BLOCKED-BY):\s*(.*)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOG_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|")
#: An executing commit: >=7 hex chars WITH at least one letter. The letter is
#: required so a compact date-stamp (`20260901`) cannot quietly satisfy the
#: ACTED contract; the rare all-digit abbreviated sha loses to that trade and
#: the repair is to paste one more character of it.
_COMMIT = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")


def _date(s: str):
    """A date or None. Never a guess: the string either IS an ISO date or is not."""
    s = s.strip()
    if not _DATE.match(s):
        return None
    try:
        return _dt.date.fromisoformat(s)
    except ValueError:
        return None


def parse(doc: str) -> list[dict]:
    """Rows in file order. Malformation is RECORDED on the row, never raised —
    a parser that dies on the first bad line reports one defect and hides the
    rest, and this file is edited by four organs.
    """
    rows: list[dict] = []
    cur: dict | None = None
    for raw in doc.splitlines():
        m = _ROUTED.match(raw)              # column 0 only: prose quoting the
        if m:                               # word cannot open a row
            fields = [f.strip() for f in m.group(1).split("|")]
            cur = {"fields": len(fields), "raw": raw, "bad": [],
                   "id": fields[0] if fields else "",
                   "routed": None, "source": "", "status": "", "status_text": "",
                   "due": None, "due_text": "", "blocked_by": "", "blocked_text": ""}
            if len(fields) != 4:
                cur["bad"].append(f"{len(fields)} pipe-separated fields, expected 4")
            else:
                cur["routed"] = _date(fields[1])
                if cur["routed"] is None:
                    cur["bad"].append(f"routed date {fields[1]!r} is not an ISO date")
                cur["source"] = fields[2]
                cur["status_text"] = fields[3]
                verb = fields[3].split()[0].upper() if fields[3].split() else ""
                cur["status"] = verb
                if verb not in STATUSES:
                    cur["bad"].append(f"status {verb!r} is not one of {'/'.join(STATUSES)}")
            rows.append(cur)
            continue
        if cur is None:
            continue
        if raw.strip() and not raw[:1].isspace():
            cur = None                      # a non-indented line ends the body
            continue
        d = _DECL.match(raw.strip())
        if not d:
            continue
        key, rest = d.group(1), d.group(2)
        head = rest.split("|")[0].strip()
        if key == "DUE":
            if cur["due"] is not None:      # a second DUE is a re-arm, and the
                cur["due"] = None           # LAST one wins — but only if it
                cur["due_text"] = ""        # parses; ambiguity is malformation
            got = _date(head)
            if got is None:
                cur["bad"].append(f"DUE: {head!r} is not an ISO date")
            else:
                # only what is owed; the date is already `due` and repeating it
                # in the finding reads as two different facts
                cur["due"] = got
                cur["due_text"] = rest.split("|", 1)[1].strip() if "|" in rest else ""
        else:
            if not head:
                cur["bad"].append("BLOCKED-BY: names no row")
            cur["blocked_by"], cur["blocked_text"] = head, rest
    return rows


def undeclared_rows(doc: str) -> list[tuple[int, str]]:
    """(line number, heading text) for every candidate heading with no
    declaration attached — attached meaning the first line after the heading
    block that is neither blank nor a further heading line is a column-0
    `ROUTED:` line. The heading is COUNTED, never parsed: no id, no date and
    no status are read out of it, however loudly it states them.
    """
    lines = doc.splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if not _CANDIDATE.match(lines[i]):
            i += 1
            continue
        lineno, head = i + 1, lines[i].lstrip("#").strip()
        j = i + 1
        while j < len(lines) and (_HEADING.match(lines[j]) or not lines[j].strip()):
            j += 1
        if j >= len(lines) or not _ROUTED.match(lines[j]):
            out.append((lineno, head))
        i = j
    return out


def throughput(doc: str, base_doc: str | None,
               window_days: int = THROUGHPUT_WINDOW_DAYS,
               cycle_days: int = CONSUMER_CYCLE_DAYS) -> dict | None:
    """The desk's disposal rate against its arrival rate over a trailing
    window, or None when there is no baseline to measure against.

    `base_doc` is the file as of the last commit before the window opened —
    git, deliberately, not the rows' declared `routed` dates. The declared date
    is the file's contract and is trusted by every other reading here, but it
    is writable, and a rate that falls when you back-date a row is a rate that
    teaches back-dating.

    Pure. `check()` supplies the revision; the properties can hold git still.
    """
    if base_doc is None:
        return None
    now = {r["id"]: r for r in parse(doc) if r["id"]}
    then = {r["id"]: r for r in parse(base_doc) if r["id"]}
    cycles = window_days / cycle_days

    arrived = sum(1 for rid in now if rid not in then)
    # Terminal NOW and not terminal THEN — so a row that both arrived and
    # closed inside the window is credited, and a row that was already ACTED
    # before the window is not credited twice.
    disposed = sum(1 for rid, r in now.items()
                   if r["status"] in TERMINAL
                   and (rid not in then or then[rid]["status"] not in TERMINAL))
    # Counted, NEVER added to `disposed`: a design is not an execution, the row
    # stays LIVE and keeps ageing, and crediting it would make drain reportable
    # by writing prose (Review 09-01 item 4, the two-meaning token).
    designed = sum(1 for rid, r in now.items()
                   if r["status"] == "DISPOSITIONED"
                   and (rid not in then or then[rid]["status"] != "DISPOSITIONED"))
    live_now = sum(1 for r in now.values() if r["status"] in LIVE)

    net = arrived - disposed
    drain = None if net >= 0 or live_now == 0 else live_now / (-net / cycles)
    return {"window_days": window_days, "cycle_days": cycle_days,
            "cycles": cycles,
            "arrived": arrived, "disposed": disposed, "designed": designed,
            "live_now": live_now, "net_arrivals": net,
            "arrived_per_cycle": arrived / cycles,
            "disposed_per_cycle": disposed / cycles,
            "designed_per_cycle": designed / cycles,
            "drain_cycles": drain,
            "unbounded": drain is None and live_now > 0}


def audit(doc: str, prev_doc: str | None = None, today: _dt.date | None = None,
          base_doc: str | None = None) -> dict:
    """Every violation in `doc`, with `prev_doc` (the previous committed
    revision) as the only baseline. Pure: no clock, no git, no filesystem —
    `main()` supplies all three, so the properties can hold the world still.

    `base_doc` — the file as of the last commit before the throughput window
    opened — is a SECOND, older baseline and is optional. Absent, the
    throughput reading is `None`: no violation moves either way, because a
    missing baseline must no more exonerate this desk than it may accuse it.
    """
    today = today or _dt.date.today()
    rows = parse(doc)
    by_id = {r["id"]: r for r in rows if r["id"]}
    findings: list[tuple[str, str, str]] = []      # (class, row id, why)

    for lineno, head in undeclared_rows(doc):
        findings.append(("UNDECLARED-ROW", f"line {lineno}",
                         f"a `## ` heading announces a row ({head[:72]!r}) and no "
                         "ROUTED: declaration follows it — to this parser it is "
                         "not a row at all, so it cannot age, go OVERDUE, or be "
                         "counted; migrate it by writing the ROUTED: line under "
                         "the heading"))

    for r in rows:
        rid = r["id"] or "(unnamed)"
        for why in r["bad"]:
            findings.append(("MALFORMED", rid, why))
        if r["status"] == "ACTED" and not _COMMIT.search(r["status_text"]):
            findings.append(("ACTED-WITHOUT-A-COMMIT", rid,
                             "ACTED names no executing commit; a design without "
                             "one is DISPOSITIONED, which keeps ageing"))
        if r["status"] not in LIVE:
            continue
        if r["blocked_by"]:
            tgt = by_id.get(r["blocked_by"])
            if tgt is None:
                findings.append(("MALFORMED", rid,
                                 f"BLOCKED-BY names {r['blocked_by']!r}, which is not a row here"))
            elif tgt["status"] in TERMINAL:
                findings.append(("HOLD-ON-A-RESOLVED-BLOCKER", rid,
                                 f"held behind {tgt['id']}, which is {tgt['status']} — "
                                 "the window it was waiting for has opened"))
        if r["due"] is not None and r["due"] < today:
            findings.append(("OVERDUE", rid,
                             f"promised {r['due'].isoformat()} ({(today - r['due']).days} d ago): "
                             f"{r['due_text']}"))
        if r["status"] == "HELD" and r["due"] is None and not r["blocked_by"]:
            findings.append(("HOLD-WITHOUT-A-CLOCK", rid,
                             "HELD exempts a row from ageing; it must declare a DUE: "
                             "or a BLOCKED-BY: to earn that"))
        if (r["status"] in ("OPEN", "DISPOSITIONED") and r["due"] is None
                and r["routed"]
                and (today - r["routed"]).days > MAX_OPEN_AGE_DAYS):
            what = ("OPEN" if r["status"] == "OPEN"
                    else "DISPOSITIONED (design written, execution owed)")
            findings.append(("STALE", rid,
                             f"{what} for {(today - r['routed']).days} d, past the "
                             f"{MAX_OPEN_AGE_DAYS}-day consumer cycle, with no DUE: to re-arm it"))

    if prev_doc is not None:
        prev = {r["id"]: r for r in parse(prev_doc) if r["id"]}
        for pid, prow in prev.items():
            now = by_id.get(pid)
            if now is None:
                findings.append(("VANISHED", pid,
                                 "present in the previous committed revision and gone now; "
                                 "rows are dispositioned, never deleted"))
            elif (prow["due"] is not None and now["due"] is None
                  and now["status"] in LIVE):
                findings.append(("CLOCK-REMOVED", pid,
                                 f"had DUE: {prow['due'].isoformat()} and no longer does, "
                                 "while still live"))

    counts = {c: sum(1 for f in findings if f[0] == c) for c in VIOLATIONS}
    live = [r for r in rows if r["status"] in LIVE]
    ages = [(today - r["routed"]).days for r in live if r["routed"]]
    due_pile: dict = {}
    for r in live:
        if r["due"] is not None:
            k = r["due"].isoformat()
            due_pile[k] = due_pile.get(k, 0) + 1

    # THE ACT THE PILE IS MADE OF, named. The pile line reports a symptom after
    # the fact; on 2026-09-02 the builder staggered 2026-09-06 from eighteen
    # rows to five, and by 09-04 three newly-routed rows had put it back to
    # eight — because each router picked the date without being told it was
    # already full. A row is `piled_on` when at least CAPACITY other live rows
    # were ALREADY promised on its date at the moment it was routed.
    #
    # METRIC, NEVER A VIOLATION (68th audit B3's discipline: report a number
    # rather than gate at zero). Every row that did this named a reason, and
    # some reasons are good — `cross-organ-doc-race-voids-certificates` chose a
    # full Sunday knowingly because its trap re-arms nightly. A gate here would
    # forbid a legal move; a number makes the move visible.
    #
    # CONSERVATIVE BY CONSTRUCTION: a re-armed row's `DUE:` was chosen later
    # than its `routed` date, so using `routed` as the moment of choice can
    # only UNDER-count. An instrument on a shared file errs toward silence.
    piled_on: list[dict] = []
    for r in live:
        if r["due"] is None or r["routed"] is None:
            continue
        prior = sum(1 for o in live
                    if o is not r and o["due"] == r["due"]
                    and o["routed"] is not None and o["routed"] < r["routed"])
        if prior >= MEASURED_DISCHARGE_CAPACITY:
            piled_on.append({"id": r["id"], "due": r["due"].isoformat(),
                             "prior": prior})

    # The mechanical alternative to defaulting onto Sunday: the first FUTURE
    # date that could take a new promise without becoming a pile — i.e. the
    # first date carrying fewer than CAPACITY live rows. "" when the board is
    # booked past the horizon, which is itself the answer.
    next_free_due = ""
    d = today + _dt.timedelta(days=1)
    for _ in range(FREE_DATE_HORIZON_DAYS):
        if due_pile.get(d.isoformat(), 0) < MEASURED_DISCHARGE_CAPACITY:
            next_free_due = d.isoformat()
            break
        d += _dt.timedelta(days=1)

    return {"rows": rows, "findings": findings, "counts": counts,
            "due_pile": due_pile, "piled_on": piled_on,
            "next_free_due": next_free_due,
            "throughput": throughput(doc, base_doc),
            "total": len(findings), "today": today,
            "n_rows": len(rows),
            "n_open": sum(1 for r in rows if r["status"] == "OPEN"),
            "n_held": sum(1 for r in rows if r["status"] == "HELD"),
            "n_dispositioned": sum(1 for r in rows if r["status"] == "DISPOSITIONED"),
            "n_acted": sum(1 for r in rows if r["status"] == "ACTED"),
            "n_declined": sum(1 for r in rows if r["status"] == "DECLINED"),
            "oldest_live_days": max(ages) if ages else 0}


def consumer_last_run(log: str) -> str:
    """The newest date in `PROGRESS_LOG.md`'s table, or "" — context, not a gate."""
    dates = [m.group(1) for m in (_LOG_ROW.match(l) for l in log.splitlines()) if m]
    return max(dates) if dates else ""


def render(a: dict, last_run: str = "") -> str:
    """The one line the 52nd audit said nothing in this repo could print."""
    out: list[str] = []
    oldest = a["oldest_live_days"]
    head = (f"{a['n_open']} OPEN, {a['n_held']} HELD, "
            f"{a['n_dispositioned']} DISPOSITIONED, {a['n_acted']} ACTED, "
            f"{a['n_declined']} DECLINED of {a['n_rows']} routed; "
            f"oldest live {oldest} d")
    if last_run:
        age = (a["today"] - _dt.date.fromisoformat(last_run)).days
        head += f"; consumer last ran {last_run} ({age} d ago)"
    out.append("\n  REVIEW QUEUE — " + head)
    out.append("")
    t = a.get("throughput")
    if t is None:
        out.append("  THROUGHPUT — no baseline revision for the trailing "
                   f"{THROUGHPUT_WINDOW_DAYS} days, so the")
        out.append("  disposal rate is UNMEASURED. That is a fault, not a "
                   "clean week: an absent")
        out.append("  baseline may not exonerate this desk any more than it "
                   "may accuse it.")
        out.append("")
    else:
        out.append(f"  THROUGHPUT — trailing {t['window_days']} d "
                   f"({t['cycles']:.0f} consumer cycles), measured against "
                   "the file's git")
        out.append("  history rather than its declared dates (69th audit B1). "
                   "A METRIC, never a")
        out.append("  violation: a slow week is legal and a gate here would "
                   "forbid a legal move.")
        out.append(f"    arrived   {t['arrived']:>3}  "
                   f"({t['arrived_per_cycle']:.2f}/cycle)")
        out.append(f"    disposed  {t['disposed']:>3}  "
                   f"({t['disposed_per_cycle']:.2f}/cycle)  "
                   "ACTED or DECLINED — the row left the live set")
        out.append(f"    designed  {t['designed']:>3}  "
                   f"({t['designed_per_cycle']:.2f}/cycle)  "
                   "DISPOSITIONED — NOT counted as disposal;")
        out.append("                            the row is still live and "
                   "still ageing")
        if t["unbounded"]:
            out.append(f"    drain     UNBOUNDED — the desk is not keeping "
                       f"up. {t['live_now']} live rows, arrivals")
            out.append(f"              exceed disposals by "
                       f"{t['net_arrivals']} over the window; the backlog has "
                       "no")
            out.append("              projected end. Every dated promise in "
                       "it is downstream of this.")
        elif t["drain_cycles"] is None:
            out.append("    drain     0 live rows — the desk is clear.")
        else:
            out.append(f"    drain     {t['drain_cycles']:.0f} cycles to "
                       f"clear {t['live_now']} live rows at the measured net "
                       "rate.")
        out.append("")
    for r in a["rows"]:
        if not r["id"]:
            continue
        age = f"{(a['today'] - r['routed']).days:>3} d" if r["routed"] else "  ? d"
        clock = ""
        if r["due"] is not None:
            late = (a["today"] - r["due"]).days
            clock = f"  DUE {r['due'].isoformat()}" + (f" (+{late} d)" if late > 0 else "")
        elif r["blocked_by"]:
            clock = f"  BLOCKED-BY {r['blocked_by']}"
        out.append(f"    {r['status']:<13} {age}  {r['id']}{clock}")
    if a["due_pile"]:
        out.append("")
        out.append("  DUE-DATE PILE — live rows per promised date (65th audit "
                   "B6). The consumer's one")
        out.append("  measured cycle discharged "
                   f"{MEASURED_DISCHARGE_CAPACITY} dated row; a date carrying "
                   "more is amber:")
        for d in sorted(a["due_pile"]):
            n = a["due_pile"][d]
            flag = ("  !! AMBER: pile" if n > MEASURED_DISCHARGE_CAPACITY
                    else "")
            out.append(f"    {d}  {n:>2}  {'#' * n}{flag}")
        worst_d = max(a["due_pile"], key=lambda k: a["due_pile"][k])
        worst_n = a["due_pile"][worst_d]
        if worst_n > MEASURED_DISCHARGE_CAPACITY:
            out.append(f"  {worst_n} rows share {worst_d} against a measured "
                       f"capacity of {MEASURED_DISCHARGE_CAPACITY}/cycle —")
            out.append("  that many promises are scheduled to break together. "
                       "Re-date the ones that can")
            out.append("  wait (a new DUE: with a reason is honest; a broken "
                       "date is a violation).")
        if a["piled_on"]:
            out.append("")
            out.append(f"  DATED ONTO A FULL DAY — {len(a['piled_on'])} live "
                       "row(s) named a date that already")
            out.append("  carried its measured capacity when the row was "
                       "routed. This is the ACT the")
            out.append("  pile above is made of, and it is a METRIC, not a "
                       "violation: each of these")
            out.append("  may have had a good reason, and a gate here would "
                       "forbid a legal move.")
            worst = sorted(a["piled_on"], key=lambda p: (-p["prior"], p["due"]))
            for p in worst[:PILED_ON_SHOWN]:
                out.append(f"    {p['id']:<52} -> {p['due']}  "
                           f"({p['prior']} already promised there)")
            if len(worst) > PILED_ON_SHOWN:
                # A cap that does not say what it dropped reads as coverage.
                out.append(f"    ... and {len(worst) - PILED_ON_SHOWN} more, "
                           "least-crowded first, elided from this print only "
                           "(all are in `piled_on`).")
        if a["next_free_due"]:
            out.append(f"  Next date carrying no promise yet: "
                       f"{a['next_free_due']} — the mechanical answer for the")
            out.append("  next router, instead of defaulting onto Sunday.")
        else:
            out.append(f"  NO date in the next {FREE_DATE_HORIZON_DAYS} days is "
                       "free of promises. There is no")
            out.append("  honest re-date left: the repair is to ACT or to "
                       "DECLINE.")
    if a["total"]:
        out.append("")
        out.append(f"  {a['total']} VIOLATION(S) — "
                   + ", ".join(f"{c} {n}" for c, n in a["counts"].items() if n))
        for cls, rid, why in a["findings"]:
            out.append(f"    {cls}: {rid}")
            out.append(f"        {why}")
        out.append("")
        out.append("  Three honest repairs and no dishonest one: do the work")
        out.append("  (ACTED, naming the executing commit), refuse it (DECLINED),")
        out.append("  or move the date by writing a new DUE: with a reason. A")
        out.append("  design alone is DISPOSITIONED and keeps ageing. Deleting")
        out.append("  the row, relabelling it HELD, dropping its DUE:, stamping")
        out.append("  ACTED with no commit, or writing the row as prose under a")
        out.append("  heading with no ROUTED: line are each their own violation")
        out.append("  — see VIOLATIONS in experiments/review_queue.py.")
    else:
        out.append("")
        out.append("  0 violations. A row going STALE is normal work arriving;")
        out.append("  a row going OVERDUE is a dated promise that was broken.")
    return "\n".join(out) + "\n"


def _prev_revision(path: Path) -> str | None:
    """The file as HEAD has it, or None if git cannot say (a brand-new file, no
    repo, or a detached environment). None means "no baseline available" and
    the two git-baselined classes simply do not fire — an absent baseline must
    never manufacture a violation.
    """
    try:
        rel = path.relative_to(Path(__file__).resolve().parent.parent)
        r = subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"],
                           cwd=path.parent.parent, capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def _revision_before(path: Path, when: _dt.date) -> str | None:
    """The file's content at the last commit strictly before UTC midnight on
    `when`, or None when git cannot say. Same channel as `_prev_revision` and
    for the same reason: it is the one baseline a working-tree edit cannot
    reach.
    """
    try:
        repo = Path(__file__).resolve().parent.parent
        rel = path.relative_to(repo).as_posix()
        sha = subprocess.run(
            ["git", "log", "-1", "--format=%H",
             f"--before={when.isoformat()}T00:00:00Z", "--", rel],
            cwd=repo, capture_output=True, text=True, timeout=20)
        if sha.returncode != 0 or not sha.stdout.strip():
            return None
        r = subprocess.run(["git", "show", f"{sha.stdout.strip()}:{rel}"],
                           cwd=repo, capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def live_audit(doc_path: Path | None = None, today: _dt.date | None = None) -> dict:
    """`audit` on the real file with both git baselines supplied. The single
    entry point for every caller that wants the live reading — `check()` here
    and the ratchet block in `run status` — so the two can never disagree
    about which baselines were used."""
    p = doc_path or DOC_PATH
    today = today or _dt.date.today()
    base = _revision_before(p, today - _dt.timedelta(days=THROUGHPUT_WINDOW_DAYS))
    return audit(p.read_text(), _prev_revision(p), today, base_doc=base)


def check(doc_path: Path | None = None) -> int:
    p = doc_path or DOC_PATH
    if not p.exists():
        print(f"  REVIEW QUEUE — {p} does not exist; the backlog file is the instrument.")
        return 2
    a = live_audit(p)
    last = consumer_last_run(LOG_PATH.read_text()) if LOG_PATH.exists() else ""
    print(render(a, last))
    return 2 if a["total"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 2 if any row is malformed, overdue, stale, held "
                         "without a clock, held behind a resolved blocker, "
                         "deleted, or stripped of its clock")
    args = ap.parse_args(argv)
    rc = check()
    return rc if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
