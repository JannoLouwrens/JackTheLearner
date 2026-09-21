#!/usr/bin/env python3
"""Who actually drew on the meter that rations the builder — D26's armed default.

D26's default fired 2026-09-12 (the owner did not rule by 2026-09-10):
**(iv) MEASURE ONLY, GATE NOTHING, RELAX NOTHING.** This file gates nothing,
refuses nothing, and is read by exactly one caller: `pace_gate`'s skip line in
`scripts/lib_usage.sh`, which prints its output beside the shared total.

THE SCAR. `pace_gate` rations the builder against `week:all models`, which is a
SHARED pool — the owner's interactive sessions and anything else on the account
draw on the same meter. Between 2026-09-08T08:23 and 2026-09-12T17:07 the builder
skipped **104 consecutive slots** while every instrument in the project read
healthy, because the one organ that would report a skip streak is the organ being
skipped. Two independent windows had already measured that 71-75% of the meter's
rise falls in hours when this box issues ZERO requests. That number existed only
inside audit prose, recomputed by hand each time; below it is computed from the
`usage_ledger.jsonl` rows this project already writes.

**WHY THE UNION AND NOT THE SUM, which is the one thing easy to get wrong.** The
overseer and the Review run daily and their sessions OVERLAP. Summing per-session
deltas double-counts every overlapping minute and inflates the desks' share — so
a span of meter rise is attributed to the SET of organs alive during it, and each
span is counted exactly once. A span with a builder alive and a desk alive is
`both`, not one point each.

WHAT IT DELIBERATELY DOES NOT DO:
  - it never gates. `pace_gate`'s arithmetic is untouched; this changes the
    printed LINE, never the branch. D26 option (i) — pace against our own spend
    rather than the shared total — WIDENS what the builder may spend, and a
    default may not loosen a gate (`SYSTEM.md` law 4). It did not fire.
  - it never blocks. Every failure path returns a reading marked `known=False`
    and the caller prints "unattributed"; an unreadable ledger must not cost an
    iteration. The gates above own refusal.
  - it does not model the meter. It reports where the rise FELL, which is
    arithmetic over recorded readings. Three attempts to PRICE organ-hours
    against this meter have been falsified inside a week (42nd/44th audits);
    this file makes no forecast and computes no rate.

UNKNOWN IS NOT ZERO. A span whose pct reading is missing is dropped from every
bucket rather than attributed to nobody — silently folding it into
`unattributed` would let a broken meter read as "somebody else's fault", which is
the conclusion this instrument exists to support or refute honestly.

Run:  python3 scripts/usage_attribution.py          (human-readable)
      python3 scripts/usage_attribution.py --line   (the one-line form pace_gate prints)
      python3 scripts/usage_attribution.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

LEDGER = Path("/data/jack-logs/usage_ledger.jsonl")
LADDER_LOG = Path("/data/jack-logs/ladder.log")

#: `iteration end rc=N — ...`. The only line in `ladder.log` that says what
#: happened to a slot that actually attempted to run.
_END_RE = re.compile(r"^(\S+)\s+iteration end rc=(\d+)")

#: The organs that are NOT the builder. `pace_gate` applies to the builder
#: alone — `scripts/review.sh` calls `usage_gate` without `pace_gate`, so the
#: two organs that report to the owner are rationed by nothing. That asymmetry
#: is the finding D26 was escalated on, so the split is builder vs everyone
#: else, and `selftest` is deliberately included: a probe's spend is this
#: project's spend.
DESK_ORGANS = ("overseer", "review", "field-watch", "selftest")


def _rows(text: str) -> list[dict]:
    """Parsed rows with a usable ts, in time order.

    A malformed line is SKIPPED rather than raising: this is a shared append-only
    log and a detector on one must bound itself to what it can read
    (`lib_credits.sh`'s rule). A row with no `ts` cannot be placed on the
    timeline and is not evidence of anything.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and r.get("ts") and r.get("organ"):
            out.append(r)
    out.sort(key=lambda r: r["ts"])
    return out


def _week_rows(rows: list[dict]) -> list[dict]:
    """Rows since the last weekly reset, detected as the meter FALLING.

    The reset time is not recorded anywhere this file can read, and hard-coding
    it is the mistake `CLAUDE.md` has made twice (a cached reset date went five
    days stale on the one page that opens by saying no number is cached on it).
    A monotone-within-a-week meter that drops has reset — that is derivable from
    the data and cannot go stale.
    """
    last_drop = 0
    prev = None
    for i, r in enumerate(rows):
        pct = r.get("pct")
        if not isinstance(pct, int):
            continue
        if prev is not None and pct < prev:
            last_drop = i
        prev = pct
    return rows[last_drop:]


def _sessions(rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """`organ -> [(start_ts, end_ts)]`, pairing each start with its next end.

    An unmatched trailing `start` (the organ is running RIGHT NOW — which is
    always true of the builder when `pace_gate` calls this) is closed at the
    last timestamp on the timeline, so the live session is not silently
    dropped. An `end` with no start is ignored.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    open_at: dict[str, str] = {}
    last_ts = rows[-1]["ts"] if rows else ""
    for r in rows:
        organ, phase, ts = r["organ"], r.get("phase"), r["ts"]
        if phase == "start":
            open_at[organ] = ts
        elif phase == "end" and organ in open_at:
            out.setdefault(organ, []).append((open_at.pop(organ), ts))
    for organ, ts in open_at.items():
        if ts <= last_ts:
            out.setdefault(organ, []).append((ts, last_ts))
    return out


def _alive(sessions: dict[str, list[tuple[str, str]]], organs, lo: str, hi: str) -> bool:
    """Was any of `organs` alive for any part of the span (lo, hi]?"""
    for organ in organs:
        for s, e in sessions.get(organ, ()):
            if s < hi and e > lo:
                return True
    return False


def slot_outcomes(log_text: str) -> list[tuple[str, int]]:
    """`[(timestamp, rc)]` for every slot that ENDED, oldest first.

    A `PACING:` line is not a slot outcome — it is a slot that never started.
    Nothing here conflates the two; see `failed_streak` for why that matters.
    """
    out = []
    for raw in log_text.splitlines():
        m = _END_RE.match(raw.strip())
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def failed_streak(outcomes: list[tuple[str, int]]) -> int:
    """Consecutive slot ENDINGS with `rc != 0`, newest backwards.

    `PACING:` lines are transparent here ON PURPOSE. On 2026-09-20/21 six
    slots died `rc=126` (`Argument list too long` — the steering file crossed
    `MAX_ARG_STRLEN`) with eighteen paced skips interleaved between them. The
    honest reading of that log is *"every slot that tried to run, died"*, and
    a counter that let a skip reset it would have printed 1 six times instead
    of 6 once.
    """
    n = 0
    for _ts, rc in reversed(outcomes):
        if rc == 0:
            break
        n += 1
    return n


def hours_since_ok(outcomes: list[tuple[str, int]],
                   now: str | None = None) -> float | None:
    """Hours since the last slot that ended `rc=0`. None if there is none.

    This is the reading the outage had no instrument for. `dark_slots` was 0,
    `iteration start` lines were being counted one file over, and the ledger
    was clean — every liveness surface in the project read healthy while the
    builder had not completed an iteration in 23 hours. A number that only
    goes up while nothing works cannot be satisfied by a log line.
    """
    last = next((ts for ts, rc in reversed(outcomes) if rc == 0), None)
    if last is None:
        return None
    now = now or time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    try:
        import datetime as _dt
        a = _dt.datetime.fromisoformat(last)
        b = _dt.datetime.fromisoformat(now)
    except Exception:
        return None
    return round((b - a).total_seconds() / 3600.0, 2)


def attribution(text: str | None = None, log_text: str | None = None,
                now: str | None = None) -> dict:
    """Where this usage week's meter rise fell, in meter points.

    Returns `{"known", "total", "builder", "desks", "both", "unattributed",
    "dark_slots", "dark_known"}`. `known` is False when the ledger cannot be
    read or holds no usable pct readings — and then every bucket is None, never
    0, because a meter this project cannot read is not a meter it did not move.
    """
    out = {"known": False, "total": None, "builder": None, "desks": None,
           "both": None, "unattributed": None,
           "dark_slots": None, "dark_known": False,
           "failed_slots": None, "hours_since_rc0": None}

    # The dark-slot streak is independent of the ledger: it is the one fault
    # `pace_gate` cannot report about itself, because the organ that would
    # report it is the organ being skipped.
    if log_text is None:
        try:
            log_text = LADDER_LOG.read_text(errors="replace")
        except Exception:
            log_text = None
    if log_text is not None:
        streak = 0
        for line in reversed(log_text.splitlines()):
            if not line.strip():
                continue
            if "PACING:" in line:
                streak += 1
            elif line[:4].isdigit():       # a real slot line ends the streak
                break
        out["dark_slots"] = streak
        out["dark_known"] = True
        # THE SECOND READING, and it is a SEPARATE quantity (107th audit,
        # RANK 2 / FOR THE BUILDER 3). `dark_slots` counts slots that were
        # SKIPPED, which is a real thing and is left alone. It cannot see a
        # slot that STARTED AND DIED: the `break` above fires on any line
        # beginning with four digits, so `iteration end rc=126` reads as a
        # healthy slot and resets the streak to zero. On 2026-09-21 at 07:07
        # this printed `0 dark slots` with the builder 23 hours idle and six
        # consecutive slots dead. Widening the streak would have destroyed a
        # good number to paper over a missing one; this adds the missing one.
        outcomes = slot_outcomes(log_text)
        out["failed_slots"] = failed_streak(outcomes)
        out["hours_since_rc0"] = hours_since_ok(outcomes, now=now)

    if text is None:
        try:
            text = LEDGER.read_text(errors="replace")
        except Exception:
            return out
    rows = _week_rows(_rows(text))
    marks = [(r["ts"], r["pct"]) for r in rows if isinstance(r.get("pct"), int)]
    if len(marks) < 2:
        return out

    sessions = _sessions(rows)
    desks = [o for o in DESK_ORGANS]
    buckets = {"builder": 0, "desks": 0, "both": 0, "unattributed": 0}
    for (lo, plo), (hi, phi) in zip(marks, marks[1:]):
        delta = phi - plo
        if delta <= 0:                     # flat or a reset boundary
            continue
        b = _alive(sessions, ("builder",), lo, hi)
        d = _alive(sessions, desks, lo, hi)
        key = "both" if (b and d) else "builder" if b else "desks" if d else "unattributed"
        buckets[key] += delta
    out.update(known=True, total=sum(buckets.values()), **buckets)
    return out


def _failed_phrase(a: dict) -> str:
    """The dead-launcher half of the liveness sentence. Never silent: a dash
    is not a reading, and `0` must be distinguishable from `unknown`."""
    f, h = a.get("failed_slots"), a.get("hours_since_rc0")
    if f is None:
        return "failed slots unknown"
    age = (f"{h:.1f} h since the last rc=0" if h is not None
           else "NO rc=0 in this log at all")
    if f:
        return (f"!! {f} consecutive slot(s) ENDED rc!=0 — the launcher is "
                f"dying, not pacing ({age})")
    return f"0 failed slots ({age})"


def line(a: dict) -> str:
    """The compact form `pace_gate` appends to its skip line.

    SKIPPED and DEAD print in the same sentence, by the 107th audit's explicit
    instruction: the outage was invisible because the only liveness phrase on
    this line was `0 dark slots`, which was TRUE and useless.
    """
    if a["dark_known"] and a["dark_slots"]:
        dark = f"{a['dark_slots']} consecutive dark slot(s)"
    elif a["dark_known"]:
        dark = "0 dark slots"
    else:
        dark = "dark slots unknown"
    dark += "; " + _failed_phrase(a)
    if not a["known"]:
        return f"attribution unreadable (this project's own share unknown, NOT zero); {dark}"
    t = a["total"] or 0
    def pc(n: int) -> str:
        return f"{n}" + (f" ({100 * n // t}%)" if t else "")
    return (f"of this week's {t} shared point(s): builder {pc(a['builder'])}, "
            f"desks {pc(a['desks'])}, both {pc(a['both'])}, "
            f"NOT THIS PROJECT {pc(a['unattributed'])}; {dark}")


def _selftest() -> int:
    """Known-answer battery. Every arm is planted beside the state it must not
    be confused with — the same discipline as `coverage.py`'s fixtures."""
    fails: list[str] = []

    def row(organ, ts, pct, phase):
        return json.dumps({"organ": organ, "ts": ts, "pct": pct,
                           "model_pct": None, "phase": phase})

    # P1 — THE LOAD-BEARING ARM: the union, not the sum. The overseer and the
    # review overlap completely across a 4-point rise. Summing sessions would
    # bill the desks 8; the union bills 4.
    doc = "\n".join([
        row("overseer", "2026-09-10T01:00:00+00:00", 10, "start"),
        row("review",   "2026-09-10T01:05:00+00:00", 10, "start"),
        row("review",   "2026-09-10T02:00:00+00:00", 14, "end"),
        row("overseer", "2026-09-10T02:05:00+00:00", 14, "end"),
    ])
    a = attribution(doc, log_text="")
    if not a["known"] or a["desks"] != 4 or a["total"] != 4:
        fails.append(f"union: two fully-overlapping desk sessions across a "
                     f"4-point rise must bill 4, not 8 — got desks="
                     f"{a['desks']} total={a['total']}")

    # P2 — the split, and the span nobody was awake for. This is the whole
    # reason D26 exists: a rise with no organ running is NOT the builder's.
    doc = "\n".join([
        row("builder",  "2026-09-10T01:00:00+00:00", 10, "start"),
        row("builder",  "2026-09-10T01:30:00+00:00", 12, "end"),
        row("overseer", "2026-09-10T02:00:00+00:00", 12, "start"),
        row("overseer", "2026-09-10T02:30:00+00:00", 15, "end"),
        row("selftest", "2026-09-10T05:00:00+00:00", 40, "probe"),
    ])
    a = attribution(doc, log_text="")
    if a["builder"] != 2 or a["desks"] != 3 or a["unattributed"] != 25:
        fails.append(f"split: expected builder=2 desks=3 unattributed=25, got "
                     f"builder={a['builder']} desks={a['desks']} "
                     f"unattributed={a['unattributed']} — a rise while nothing "
                     f"of ours ran is the finding, not a rounding error")

    # P3 — an overlapping builder+desk span is `both`, billed once to neither.
    doc = "\n".join([
        row("builder",  "2026-09-10T01:00:00+00:00", 10, "start"),
        row("overseer", "2026-09-10T01:10:00+00:00", 10, "start"),
        row("overseer", "2026-09-10T01:50:00+00:00", 16, "end"),
        row("builder",  "2026-09-10T02:00:00+00:00", 16, "end"),
    ])
    a = attribution(doc, log_text="")
    if a["both"] != 6 or a["builder"] or a["desks"]:
        fails.append(f"overlap: a span with both alive must bill `both`, not "
                     f"be double-counted — got both={a['both']} "
                     f"builder={a['builder']} desks={a['desks']}")

    # P4 — the weekly reset. A drop means the meter reset; points before it
    # belong to a spent week and must not inflate this one.
    doc = "\n".join([
        row("builder", "2026-09-05T01:00:00+00:00", 80, "start"),
        row("builder", "2026-09-05T02:00:00+00:00", 88, "end"),
        row("builder", "2026-09-07T01:00:00+00:00", 3, "start"),
        row("builder", "2026-09-07T02:00:00+00:00", 9, "end"),
    ])
    a = attribution(doc, log_text="")
    if a["total"] != 6 or a["builder"] != 6:
        fails.append(f"reset: a falling meter is a weekly reset and the spent "
                     f"week must not be counted — expected 6, got "
                     f"total={a['total']} builder={a['builder']}")

    # P5 — UNKNOWN IS NOT ZERO. An unreadable ledger must report unknown, and
    # the caller must not be handed a comfortable 0. (Arm.cost's lesson: a
    # sentinel that is also a valid value cannot be detected.)
    a = attribution("", log_text="")
    if a["known"] or a["total"] is not None or a["builder"] is not None:
        fails.append(f"unknown: an empty ledger must report known=False with "
                     f"None buckets, never 0 — got {a}")
    if "unreadable" not in line(a):
        fails.append("unknown: the printed line must say so in words")

    # P6 — the dark-slot streak, and that it STOPS at the last SUCCESSFUL
    # slot. THE PREMISE OF THIS PROPERTY WAS WRONG UNTIL 2026-09-21 and the
    # comment said so out loud: "a real slot line ends the streak". It does,
    # and that is correct for a counter of SKIPS — but the assertion was the
    # only thing anyone checked, so nothing in this project noticed that a
    # slot line saying `rc=126` also ended it. The streak's arithmetic is
    # UNCHANGED (widening it would destroy a good number). What changed is
    # that the property is now stated against a successful slot, and P6b
    # below asserts the reading that covers the case this one cannot see.
    log = ("2026-09-08T08:23:00+00:00 iteration end rc=0\n"
           "2026-09-08T09:07:00+00:00 PACING: skipping\n"
           "2026-09-08T10:07:00+00:00 PACING: skipping\n")
    a = attribution("", log_text=log, now="2026-09-08T10:23:00+00:00")
    if not a["dark_known"] or a["dark_slots"] != 2:
        fails.append(f"dark: the streak is the TRAILING run of PACING lines "
                     f"and must end at the last real slot — expected 2, got "
                     f"{a['dark_slots']}")
    if a["failed_slots"] != 0 or a["hours_since_rc0"] != 2.0:
        fails.append(f"dark: a healthy loop being paced must read 0 failed "
                     f"slots and an HONEST age — got failed="
                     f"{a['failed_slots']} hours={a['hours_since_rc0']}")
    a = attribution("", log_text="2026-09-08T08:23:00+00:00 iteration end rc=0\n")
    if not a["dark_known"] or a["dark_slots"] != 0:
        fails.append("dark: a loop that just ran is 0 dark slots, and 0 must "
                     "be distinguishable from unknown")
    a = attribution("", log_text=None) if not LADDER_LOG.exists() else {"dark_known": True}
    if not a["dark_known"]:
        pass  # no log on this box is a legitimate unknown, not a failure

    # P6b — THE OUTAGE, REPLAYED. This is the real 2026-09-20/21 shape: dead
    # slots with paced skips interleaved. `dark_slots` reads 1 — TRUE, and
    # useless. The two new readings must both fire, and the printed line must
    # SAY the launcher is dying rather than leaving a reader to infer it.
    log = ("2026-09-20T06:07:00+00:00 iteration end rc=0 — 109 -> 109\n"
           "2026-09-20T07:07:00+00:00 iteration end rc=126 — 109 -> 109\n"
           "2026-09-20T08:07:00+00:00 iteration end rc=126 — 109 -> 109\n"
           "2026-09-20T09:07:00+00:00 PACING: acting on 'week:all models'\n"
           "2026-09-21T06:07:00+00:00 iteration end rc=126 — 109 -> 109\n"
           "2026-09-21T07:07:00+00:00 PACING: acting on 'week:all models'\n")
    a = attribution("", log_text=log, now="2026-09-21T08:07:00+00:00")
    if a["dark_slots"] != 1:
        fails.append(f"outage: `dark_slots` must keep its own meaning — "
                     f"expected 1 trailing PACING line, got {a['dark_slots']}")
    if a["failed_slots"] != 3:
        fails.append(f"outage: three slots ended rc!=0 with a paced skip "
                     f"BETWEEN them; a skip must not reset the failed streak "
                     f"— expected 3, got {a['failed_slots']}")
    if a["hours_since_rc0"] != 26.0:
        fails.append(f"outage: 26 hours since the last rc=0 — got "
                     f"{a['hours_since_rc0']}")
    txt = line(a)
    if "0 dark slots" in txt and "ENDED rc!=0" not in txt:
        fails.append("outage: the line printed a reassuring dark-slot count "
                     "with no word about the dead launcher — that is the "
                     "exact sentence that hid a 23-hour outage")
    if "the launcher is dying, not pacing" not in txt:
        fails.append(f"outage: the line must NAME the fault, not leave it to "
                     f"be inferred — got {txt!r}")

    # P6c — UNKNOWN IS NOT ZERO, for the new readings too. A log with no
    # successful slot in it must not report a comfortable age.
    a = attribution("", log_text="2026-09-21T06:07:00+00:00 iteration end rc=126\n")
    if a["failed_slots"] != 1 or a["hours_since_rc0"] is not None:
        fails.append(f"no-rc0: a log with no successful slot must report age "
                     f"None, never 0 — got {a['hours_since_rc0']}")
    if "NO rc=0 in this log at all" not in line(a):
        fails.append("no-rc0: the printed line must say so in words")
    a = attribution("", log_text=None)
    if a["failed_slots"] is not None and not LADDER_LOG.exists():
        fails.append("no-log: an unreadable log must report None, not 0")
    if "unknown" not in _failed_phrase({"failed_slots": None}):
        fails.append("no-log: the printed line must say failed slots unknown")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"usage_attribution selftest: {len(fails)} failure(s)")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--line", action="store_true",
                    help="the one-line form pace_gate prints")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    a = attribution()
    if args.line:
        print(line(a))
        return 0
    for k, v in a.items():
        print(f"  {k:<14} {v}")
    print(f"\n  {line(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
