#!/bin/bash
# THE REVIEW — weekly chief scientist: big-picture progress + test re-examination.
# Owner directive 2026-08-09: "recurring checks that rewrite tests and look at
# the bigger picture of progress." May STRENGTHEN specs, may never weaken —
# the overseer independently audits every spec diff, so the powers check each
# other. Runs DAILY at 06:37; FULL (Part 2, the test re-examination) on Sundays.
#
# Install:  37 6 * * *  /home/opc/jackthelearner/scripts/review.sh
# Read:     docs/PROGRESS.md
# Stop:     touch /home/opc/jackthelearner/.review-paused
set -uo pipefail
REPO=/home/opc/jackthelearner
LOG=/data/jack-logs/review.log
PAUSE="$REPO/.review-paused"
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
# Declare this run to procwatch (65th audit B5) — same reason as overseer.sh:
# an auditor's instrument call must not read as a builder leftover. Children
# of a declared pid are attributed; dead declarations are pruned.
. "$REPO/scripts/lib_procwatch.sh"
proc_declare $$ "review.sh consumer slot"
MODEL="${JACK_REVIEW_MODEL:-opus}"
# Two gears, one organ (owner, 2026-08-09: "every 24 hours the loop must be
# reviewed... to fix itself... the senior engineer"). Steering rots in hours
# — the builder's map went stale twice in one day. Science needs a week of
# data. So: DAILY = the morning walk-through (steering only, ~15 min);
# FULL on Sundays = everything, including test re-examination.
# THE TURN BUDGET SCALES WITH THE CLOCK, and until 2026-08-31 it did not.
# `--max-turns 60` was hard-coded identically for DAILY (20m) and FULL (40m), so
# the mode that does TWICE THE WORK got twice the clock and the same turns — and
# FULL is the mode that owns the world redesign. On 2026-08-30 the project's
# first-ever FULL run died at `Reached max turns (60)` after ELEVEN minutes of a
# forty-minute budget, having written nothing (see scripts/lib_liveness.sh).
# There have been 7 max-turns deaths across the three organs (ladder 4,
# overseer 2, review 1); every one of them left time on the clock.
#
# The rate is DAILY's own, unchanged: 60 turns / 20 min = 3 turns/min. FULL gets
# the same rate over its own clock. This does not raise the spend ceiling —
# `timeout` still does that — it stops an organ being killed early by a budget
# that was never derived from anything.
TURNS_PER_MIN=3
if [ "$(date +%u)" = "7" ]; then MODE=FULL; TMOUT=40m; MINUTES=40; else MODE=DAILY; TMOUT=20m; MINUTES=20; fi
MAXTURNS=$(( MINUTES * TURNS_PER_MIN ))
say "review start — mode ${MODE}, model ${MODEL}, ${TMOUT} / ${MAXTURNS} turns"
mark_log
nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
RC=$?
if credits_out; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  mark_log
  nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
    --model sonnet --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
  RC=$?
elif [ "$RC" -ne 0 ] && api_overloaded; then
  # Transient server-side 5xx (the 08-24 daily died on one 529 and never ran).
  # Same model, one retry, after a pause — this is not a credit event.
  say "API overloaded on ${MODEL} — waiting 120s, retrying once"
  sleep 120
  mark_log
  nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
    --model "$MODEL" --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
  RC=$?
fi
# A run that dies LATE has already written its report -> stamp it a DRAFT. A run
# that dies BEFORE writing leaves a clean file that is nonetheless no longer
# current state -> stamp it STALE, but only once it is older than this organ's
# 24 h cadence (both branches in scripts/lib_seal.sh; the 08-30 FULL death is
# the scar for the second one).
seal_output "$RC" docs/PROGRESS.md review say 25
say "sweep end rc=${RC} — $(grep -c STRENGTHEN docs/PROGRESS.md 2>/dev/null || echo 0) strengthen lines"
exit 0
