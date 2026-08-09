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
cd "$REPO" || exit 0
MODEL="${JACK_FIELDWATCH_MODEL:-opus}"
say "sweep start — model ${MODEL}"
nice -n 19 timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
RC=$?
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  nice -n 19 timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
  RC=$?
fi
say "sweep end rc=${RC} — $(grep -c NOMINAT docs/FIELD_WATCH.md 2>/dev/null || echo 0) nomination lines"
exit 0
