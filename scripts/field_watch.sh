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
# CLAUDE BUDGET GATE. Credits were the one binding resource with no meter
# (META_AUDIT 2026-08-09, four exhaustion events in a day). scripts/
# claude_usage.py sums this machine's own token consumption; the organs stop
# at the owner-set threshold so the OWNER always has headroom left. The owner
# is never blocked by this — only the autonomous organs are.
PCT=$(/data/venvs/jackthelearner/bin/python "$REPO/scripts/claude_usage.py" --pct 2>/dev/null || echo 0)
LIMIT=$(/data/venvs/jackthelearner/bin/python -c "import json,sys;print(json.load(open('$REPO/scripts/claude_budget.json'))['pause_at_pct'])" 2>/dev/null || echo 90)
if awk -v p="$PCT" -v l="$LIMIT" 'BEGIN{exit !(p>=l)}'; then
  say "ABORT: Claude usage ${PCT}% of ceiling (>= ${LIMIT}%) — leaving headroom for the owner"
  exit 0
fi
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
