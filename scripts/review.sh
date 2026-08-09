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
cd "$REPO" || exit 0
MODEL="${JACK_REVIEW_MODEL:-opus}"
say "sweep start — model ${MODEL}"
nice -n 19 timeout 40m claude -p "$(cat "$REPO/scripts/review_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
RC=$?
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  nice -n 19 timeout 40m claude -p "$(cat "$REPO/scripts/review_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns 60 >> "$LOG" 2>&1
  RC=$?
fi
say "sweep end rc=${RC} — $(grep -c STRENGTHEN docs/PROGRESS.md 2>/dev/null || echo 0) strengthen lines"
exit 0
