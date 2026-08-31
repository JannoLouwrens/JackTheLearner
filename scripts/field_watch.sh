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
. "$REPO/scripts/lib_credits.sh"
. "$REPO/scripts/lib_usage.sh"
. "$REPO/scripts/lib_pause.sh"
. "$REPO/scripts/lib_seal.sh"
pause_gate say "$PAUSE" || exit 0
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt 3 ] && { say "ABORT: ${FREE_GB}GB free on /"; exit 0; }
LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN{exit !(l>6.0)}' && { say "ABORT: load ${LOAD} — tenants first"; exit 0; }
# STOP AT 90%. Owner's rule, 2026-08-09: pause ALL agents at 90% weekly usage
# until the owner resumes them. Real number from `claude -p /usage`, not a
# proxy. Nothing else is throttled — this is the only limit.
# UNKNOWN IS NOT ZERO: if usage cannot be read, do NOT run. A meter that fails
# open is not a limit.
usage_gate say || exit 0
cd "$REPO" || exit 0
# Turn budget derived from the clock at the Review's rate (3 turns/min over a
# 30m timeout), not hard-coded. Seven max-turns deaths across the three organs,
# every one with time left on the clock — see scripts/review.sh.
MAXTURNS=90
MODEL="${JACK_FIELDWATCH_MODEL:-opus}"
say "sweep start — model ${MODEL}"
mark_log
nice -n 19 env TMPDIR=/data/tmp timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
RC=$?
if credits_out; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  mark_log
  nice -n 19 env TMPDIR=/data/tmp timeout 30m claude -p "$(cat "$REPO/scripts/field_watch_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
  RC=$?
fi
# Same seal as the overseer and the review: a late death leaves a written report
# that nothing marks as a draft, and a death before writing leaves a page that
# still claims to be current (scripts/lib_seal.sh). 169 h = this organ's weekly
# cadence plus an hour; below that the last report is still the current one.
seal_output "$RC" docs/FIELD_WATCH.md field-watch say 169
say "sweep end rc=${RC} — $(grep -c NOMINAT docs/FIELD_WATCH.md 2>/dev/null || echo 0) nomination lines"
exit 0
