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
MODEL="${JACK_REVIEW_MODEL:-opus}"
# Two gears, one organ (owner, 2026-08-09: "every 24 hours the loop must be
# reviewed... to fix itself... the senior engineer"). Steering rots in hours
# — the builder's map went stale twice in one day. Science needs a week of
# data. So: DAILY = the morning walk-through (steering only, ~15 min);
# FULL on Sundays = everything, including test re-examination.
if [ "$(date +%u)" = "7" ]; then MODE=FULL; TMOUT=40m; else MODE=DAILY; TMOUT=20m; fi
say "review start — mode ${MODE}, model ${MODEL}"
nice -n 19 timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
RC=$?
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  nice -n 19 timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
  RC=$?
fi
say "sweep end rc=${RC} — $(grep -c STRENGTHEN docs/PROGRESS.md 2>/dev/null || echo 0) strengthen lines"
exit 0
