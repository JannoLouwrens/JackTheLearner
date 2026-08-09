#!/bin/bash
# FIELD WATCH — weekly scout of the research frontier. Nominates, never adopts.
# Owner directive 2026-08-09: the system must "be part of everything" — outside
# models enter as bakeoff arms, and this is the process that finds them.
#
# Runs Mondays 05:37 — after Sunday's Kaggle reset, so fresh nominations land
# just as the week's GPU budget does. No lock against the builder: read-only
# on everything except its own two output files.
#
# Install:  37 5 * * 1  /home/opc/jackthelearner/scripts/field_watch.sh
# Read:     docs/FIELD_WATCH.md
# Stop:     touch /home/opc/jackthelearner/.fieldwatch-paused
set -uo pipefail
REPO=/home/opc/jackthelearner
LOG=/data/jack-logs/field_watch.log
PAUSE="$REPO/.fieldwatch-paused"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }
[ -f "$PAUSE" ] && { say "paused"; exit 0; }
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt 3 ] && { say "ABORT: ${FREE_GB}GB free on /"; exit 0; }
LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN{exit !(l>6.0)}' && { say "ABORT: load ${LOAD} — tenants first"; exit 0; }
# STOP AT 90%. Owner's rule, 2026-08-09: pause ALL agents at 90% weekly usage
# until the owner resumes them. Real number from `claude -p /usage`, not a
# proxy. Nothing else is throttled — this is the only limit.
# UNKNOWN IS NOT ZERO: if usage cannot be read, do NOT run. A meter that fails
# open is not a limit.
PCT=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null)
case "$PCT" in ''|*[!0-9]*) say "ABORT: usage unreadable — refusing to run"; exit 0;; esac
if [ "$PCT" -ge 90 ]; then
  say "STOPPED at ${PCT}% weekly usage — all agents paused until the owner resumes"
  exit 0
fi
cd "$REPO" || exit 0
MODEL="${JACK_FIELDWATCH_MODEL:-opus}"
say "sweep start — model ${MODEL}"
nice -n 19 env TMPDIR=/data/tmp timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
RC=$?
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  nice -n 19 env TMPDIR=/data/tmp timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
  RC=$?
fi
say "sweep end rc=${RC} — $(grep -c NOMINAT docs/FIELD_WATCH.md 2>/dev/null || echo 0) nomination lines"
exit 0
