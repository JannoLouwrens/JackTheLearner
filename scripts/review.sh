#!/bin/bash
# THE REVIEW — weekly chief scientist: big-picture progress + test re-examination.
# Owner directive 2026-08-09: "recurring checks that rewrite tests and look at
# the bigger picture of progress." May STRENGTHEN specs, may never weaken —
# the overseer independently audits every spec diff, so the powers check each
# other. Runs Sundays 06:37.
#
# Install:  37 6 * * 0  /home/opc/jackthelearner/scripts/field_watch.sh
# Read:     docs/PROGRESS.md
# Stop:     touch /home/opc/jackthelearner/.review-paused
set -uo pipefail
REPO=/home/opc/jackthelearner
LOG=/data/jack-logs/review.log
PAUSE="$REPO/.review-paused"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }
. "$REPO/scripts/lib_credits.sh"
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
MODEL="${JACK_REVIEW_MODEL:-opus}"
# Two gears, one organ (owner, 2026-08-09: "every 24 hours the loop must be
# reviewed... to fix itself... the senior engineer"). Steering rots in hours
# — the builder's map went stale twice in one day. Science needs a week of
# data. So: DAILY = the morning walk-through (steering only, ~15 min);
# FULL on Sundays = everything, including test re-examination.
if [ "$(date +%u)" = "7" ]; then MODE=FULL; TMOUT=40m; else MODE=DAILY; TMOUT=20m; fi
say "review start — mode ${MODE}, model ${MODEL}"
mark_log
nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
RC=$?
if credits_out; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  mark_log
  nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
  RC=$?
fi
say "sweep end rc=${RC} — $(grep -c STRENGTHEN docs/PROGRESS.md 2>/dev/null || echo 0) strengthen lines"
exit 0
