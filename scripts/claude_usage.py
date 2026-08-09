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
"""
from __future__ import annotations

import re
import subprocess
import sys

TIMEOUT = 90


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
