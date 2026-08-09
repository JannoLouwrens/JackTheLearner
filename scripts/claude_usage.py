"""How many Claude tokens the machine has spent in a rolling window.

The scar: on 2026-08-09 the builder hit "out of usage credits" FOUR times and
the meta-audit recorded credits as "the binding resource, and the only one
that is unmetered" — GPU hours are tracked to the second while the thing that
powers every organ had no accounting at all.

No CLI exposes the plan's remaining percentage; that lives server-side. But
every session writes its own usage records to the transcript, so the machine
CAN meter its own consumption. This is the Claude equivalent of
experiments/gpu_budget.json: a ceiling in a file, checked before spending.

    python scripts/claude_usage.py            # human summary
    python scripts/claude_usage.py --pct      # percent of ceiling, for scripts

Ceiling lives in scripts/claude_budget.json so the owner can change it without
touching code. Set it from OBSERVED usage, not a guess — run this for a few
days first.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TRANSCRIPTS = Path.home() / ".claude" / "projects"
BUDGET_FILE = Path(__file__).parent / "claude_budget.json"
DEFAULT = {
    "weekly_output_token_ceiling": 40_000_000,
    "pause_at_pct": 90,
    "_comment": "Set the ceiling from OBSERVED usage. pause_at_pct is where "
                "the autonomous organs stop; the owner is never blocked.",
}


def budget() -> dict:
    if BUDGET_FILE.exists():
        try:
            return {**DEFAULT, **json.loads(BUDGET_FILE.read_text())}
        except Exception:
            pass
    BUDGET_FILE.write_text(json.dumps(DEFAULT, indent=2) + "\n")
    return DEFAULT


def spent(days: float = 7.0) -> int:
    """Output tokens across all transcripts touched in the window.

    Output tokens only: they dominate cost and are the cleanest single signal.
    Counted by FILE MTIME rather than per-message timestamps — coarse on
    purpose. A meter that is expensive to read gets read rarely, and this one
    runs before every organ fires.
    """
    cutoff = time.time() - days * 86400
    total = 0
    if not TRANSCRIPTS.exists():
        return 0
    for f in TRANSCRIPTS.rglob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                continue
            for line in f.read_text(errors="ignore").splitlines():
                i = line.find('"output_tokens":')
                while i != -1:
                    j = i + 16
                    k = j
                    while k < len(line) and line[k].isdigit():
                        k += 1
                    if k > j:
                        total += int(line[j:k])
                    i = line.find('"output_tokens":', k)
        except Exception:
            continue          # a meter must never break the thing it measures
    return total


def main() -> int:
    b = budget()
    used = spent()
    ceiling = max(1, int(b["weekly_output_token_ceiling"]))
    pct = 100.0 * used / ceiling
    if "--pct" in sys.argv:
        print(f"{pct:.1f}")
        return 0
    bar = "#" * int(min(pct, 100) // 5)
    print(f"Claude output tokens, last 7 days: {used:,} of {ceiling:,}")
    print(f"  [{bar:<20}] {pct:.1f}%   (organs pause at {b['pause_at_pct']}%)")
    if pct >= b["pause_at_pct"]:
        print("  OVER THRESHOLD — autonomous organs will pause. You are not blocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
