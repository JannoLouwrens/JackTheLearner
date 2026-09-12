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
import sys
from pathlib import Path

LEDGER = Path("/data/jack-logs/usage_ledger.jsonl")
LADDER_LOG = Path("/data/jack-logs/ladder.log")

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


def attribution(text: str | None = None, log_text: str | None = None) -> dict:
    """Where this usage week's meter rise fell, in meter points.

    Returns `{"known", "total", "builder", "desks", "both", "unattributed",
    "dark_slots", "dark_known"}`. `known` is False when the ledger cannot be
    read or holds no usable pct readings — and then every bucket is None, never
    0, because a meter this project cannot read is not a meter it did not move.
    """
    out = {"known": False, "total": None, "builder": None, "desks": None,
           "both": None, "unattributed": None,
           "dark_slots": None, "dark_known": False}

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


def line(a: dict) -> str:
    """The compact form `pace_gate` appends to its skip line."""
    if a["dark_known"] and a["dark_slots"]:
        dark = f"{a['dark_slots']} consecutive dark slot(s)"
    elif a["dark_known"]:
        dark = "0 dark slots"
    else:
        dark = "dark slots unknown"
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

    # P6 — the dark-slot streak, and that it STOPS at the last real slot.
    log = ("2026-09-08T08:23:00+00:00 iteration end rc=0\n"
           "2026-09-08T09:07:00+00:00 PACING: skipping\n"
           "2026-09-08T10:07:00+00:00 PACING: skipping\n")
    a = attribution("", log_text=log)
    if not a["dark_known"] or a["dark_slots"] != 2:
        fails.append(f"dark: the streak is the TRAILING run of PACING lines "
                     f"and must end at the last real slot — expected 2, got "
                     f"{a['dark_slots']}")
    a = attribution("", log_text="2026-09-08T08:23:00+00:00 iteration end rc=0\n")
    if not a["dark_known"] or a["dark_slots"] != 0:
        fails.append("dark: a loop that just ran is 0 dark slots, and 0 must "
                     "be distinguishable from unknown")
    a = attribution("", log_text=None) if not LADDER_LOG.exists() else {"dark_known": True}
    if not a["dark_known"]:
        pass  # no log on this box is a legitimate unknown, not a failure

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
