"""Real Claude usage, read from the CLI itself.

The owner asked whether `claude -p "/usage"` works headlessly. I assumed it did
not — slash commands looked like a REPL feature — and said so without testing.
It works, and it returns the authoritative numbers:

    Current session:            25% used · resets Aug 9, 11pm (UTC)
    Current week (all models):  81% used · resets Aug 12, 12pm (UTC)
    Current week (Fable):      100% used · resets Aug 12, 11:59am (UTC)

That is strictly better than the transcript-summing proxy this file used to
contain, which could not see per-model limits, could not see reset times, and
inferred its ceiling from a number the owner read out loud. Deleted.

Lesson (also in LESSONS.md): a two-line experiment beats a confident assumption
about a mechanism. This one cost an hour of building the wrong thing.

    python scripts/claude_usage.py           # human summary
    python scripts/claude_usage.py --pct     # weekly all-models percent
    python scripts/claude_usage.py --model Fable --pct
    python scripts/claude_usage.py --week-elapsed   # 0-100, how far into the week

`--week-elapsed` exists so a caller can PACE rather than only STOP. The percent
alone says how much is gone; it cannot say whether that is early or late. The
CLI already prints the reset instant ("resets Aug 31, 5am (UTC)"), so the week's
position is readable and does not need to be assumed — which matters, because
the reset moved once already (Aug 12 noon -> Aug 31 5am) and a hardcoded
"Mondays at 05:00" would have silently drifted.
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

TIMEOUT = 90
WEEK_S = 7 * 24 * 3600

_MONTHS = {m: i + 1 for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split())}


def reset_epoch(entry: dict) -> int | None:
    """Parse 'Aug 31, 5am (UTC)' / 'Aug 12, 11:59am (UTC)' into a UTC epoch.

    The year is absent from the CLI's own output, so it is inferred: the reset
    is a future instant, and a parse landing in the past means the year rolled.
    Returns None on anything unrecognised — the caller must treat that as
    'unknown', and pacing must fail OPEN when it is (the 90% stop is the real
    limit; a pace line that fails closed would be a second limit nobody set).
    """
    m = re.match(r"([A-Za-z]{3})\w*\s+(\d{1,2}),\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
                 (entry.get("resets") or "").strip(), re.I)
    if not m:
        return None
    mon, day, hh, mm, ap = m.groups()
    month = _MONTHS.get(mon.lower())
    if month is None:
        return None
    hour = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    now = datetime.now(timezone.utc)
    for year in (now.year, now.year + 1):
        try:
            t = datetime(year, month, int(day), hour, int(mm or 0), tzinfo=timezone.utc)
        except ValueError:
            return None
        if t > now - timedelta(days=3):
            return int(t.timestamp())
    return None


def read() -> dict:
    """Parse `claude -p /usage`. Returns {} on any failure — a meter must never
    break the thing it measures, and callers must treat missing data as
    'unknown', never as 'zero'."""
    try:
        out = subprocess.run(["claude", "-p", "/usage", "--max-turns", "1"],
                             capture_output=True, text=True, timeout=TIMEOUT).stdout
    except Exception:
        return {}
    d: dict = {"raw": out.strip()}
    for line in out.splitlines():
        m = re.match(r"\s*Current (session|week)\s*(?:\(([^)]+)\))?\s*:\s*(\d+)%\s*used"
                     r"(?:\s*·\s*resets\s*(.+))?", line)
        if not m:
            continue
        scope, model, pct, resets = m.groups()
        key = scope if scope == "session" else f"week:{(model or 'all models').strip()}"
        d[key] = {"pct": int(pct), "resets": (resets or "").strip()}
    return d


def main() -> int:
    d = read()
    if not d or len([k for k in d if k != "raw"]) == 0:
        print("usage unavailable (CLI did not report) — treat as UNKNOWN, not zero")
        return 2

    want_pct = "--pct" in sys.argv
    model = None
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    if "--week-elapsed" in sys.argv:
        entry = d.get(f"week:{model}" if model else "week:all models")
        end = reset_epoch(entry) if entry else None
        if end is None:
            return 2
        frac = 1.0 - (end - datetime.now(timezone.utc).timestamp()) / WEEK_S
        print(max(0, min(100, round(frac * 100))))
        return 0

    if want_pct:
        key = f"week:{model}" if model else "week:all models"
        entry = d.get(key)
        if entry is None:
            return 2
        print(entry["pct"])
        return 0

    for key in sorted(k for k in d if k != "raw"):
        e = d[key]
        bar = "#" * (e["pct"] // 5)
        resets = f"  resets {e['resets']}" if e["resets"] else ""
        print(f"{key:22s} [{bar:<20}] {e['pct']:3d}%{resets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
