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


def audit(doc: str, prev_doc: str | None = None, today: _dt.date | None = None) -> dict:
    """Every violation in `doc`, with `prev_doc` (the previous committed
    revision) as the only baseline. Pure: no clock, no git, no filesystem —
    `main()` supplies all three, so the properties can hold the world still.
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
    return {"rows": rows, "findings": findings, "counts": counts,
            "due_pile": due_pile,
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


def check(doc_path: Path | None = None) -> int:
    p = doc_path or DOC_PATH
    if not p.exists():
        print(f"  REVIEW QUEUE — {p} does not exist; the backlog file is the instrument.")
        return 2
    doc = p.read_text()
    a = audit(doc, _prev_revision(p))
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
