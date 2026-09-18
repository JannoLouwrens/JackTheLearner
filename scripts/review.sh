#!/bin/bash
# THE REVIEW — weekly chief scientist: big-picture progress + test re-examination.
# Owner directive 2026-08-09: "recurring checks that rewrite tests and look at
# the bigger picture of progress." May STRENGTHEN specs, may never weaken —
# the overseer independently audits every spec diff, so the powers check each
# other. Runs DAILY at 06:37; FULL (Part 2, the test re-examination) on Sundays.
#
# Install:  37 6 * * *          /home/opc/jackthelearner/scripts/review.sh
#           22 9,12,15,18 * * * /home/opc/jackthelearner/scripts/review.sh --retry
# Read:     docs/PROGRESS.md
# Stop:     touch /home/opc/jackthelearner/.review-paused
set -uo pipefail
REPO=/home/opc/jackthelearner
LOG=/data/jack-logs/review.log
PAUSE="$REPO/.review-paused"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# --- THE DEFERRED SITTING (99th audit B2, RANK 2) ---------------------------
# One poll a day meant one usage_gate refusal cost this desk a full day even
# when the meter reset hours later: on 09-17 and 09-18 the 06:37 slot was
# refused at 91-100%, the weekly reset landed mid-morning, budget sat abundant
# all afternoon, and the queue went 0 -> 11 OVERDUE with no sitting either day.
# A refused sitting now writes a deferral marker; later same-day polls
# (--retry, crontab above) hold the sitting AT MOST ONCE if and only if it has
# not happened yet. The once-a-day cap is on sittings actually HELD: a run the
# CLI refuses at the door in seconds (a limit wording, nothing spent, no page
# touched) is not a held sitting and re-arms the deferral — see the
# door-refusal block after the run chain below, and its scar of 2026-09-18.
# THE 90% STOP IS NOT TOUCHED: every retry passes through
# the same pause/disk/load/usage gates below — this changes only WHEN the
# question is asked, never what it answers, and a paused or over-budget desk
# stays paused or refused. A retry poll with nothing deferred exits in
# milliseconds before any gate or CLI call.
DEFER=/data/jack-logs/review-deferred
SAT=/data/jack-logs/review-last-sitting
TODAY=$(date -u +%F)
if [ "${1:-}" = "--retry" ]; then
  [ -f "$DEFER" ] && [ "$(cat "$DEFER" 2>/dev/null)" = "$TODAY" ] || exit 0
  if [ "$(cat "$SAT" 2>/dev/null)" = "$TODAY" ]; then rm -f "$DEFER"; exit 0; fi
  say "retry poll: the 06:37 sitting was deferred and has not been held — attempting it now (99th audit B2)"
fi

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
usage_gate say || {
  echo "$TODAY" > "$DEFER"
  say "sitting DEFERRED — the usage gate refused it; a later --retry poll today may hold it once the meter allows (99th audit B2)"
  exit 0
}
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
#
# RATE CORRECTED 3 -> 6, 2026-09-05 (74th audit B3, the day before the 09-06
# FULL). The 3/min figure was DAILY's ALLOWANCE, not its measured speed, and
# the measurement says turns bind before the clock ever does: the 08-30 FULL
# died at 60 turns in 11 of 40 minutes (~5.5 turns/min consumed), the 09-05
# DAILY died at 60 turns in 15 of 20 (~4/min), and all seven max-turns deaths
# across the organs left time on the clock — 4 of 4 cron-fired Sunday FULLs
# among them. At 6/min the `timeout` becomes the binding ceiling, which is the
# one that actually caps spend; --max-turns returns to being the runaway
# backstop it was meant to be. This raises no science threshold and weakens no
# control — it is an organ's own turn budget, sized to its measured consumption
# so the most consequential scheduled run is not killed by an allowance derived
# from nothing (said here explicitly so no audit reads it as a silent loosening).
TURNS_PER_MIN=6
if [ "$(date +%u)" = "7" ]; then MODE=FULL; TMOUT=40m; MINUTES=40; else MODE=DAILY; TMOUT=20m; MINUTES=20; fi
MAXTURNS=$(( MINUTES * TURNS_PER_MIN ))
# The sitting is now HAPPENING: stamp it and consume any deferral, so retry
# polls stop attempting today. Stamped before the agent runs, not after — a
# sitting that dies mid-run spent its budget and must not be re-held; the
# seal/INCOMPLETE-row machinery below already reports that death honestly.
echo "$TODAY" > "$SAT"
rm -f "$DEFER"
say "review start — mode ${MODE}, model ${MODEL}, ${TMOUT} / ${MAXTURNS} turns"
# The seal's sweep bound (74th audit B1): only dirty files whose mtime is at or
# after this moment are this run's own acts. Captured before the agent starts.
RUN_START=$(date +%s)
mark_log
usage_ledger review start "$MODEL"    # D15 (d): attribution; model passed, not env-read (76th B1)
nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
  --model "$MODEL" --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
RC=$?
if credits_out; then
  say "OUT OF CREDITS on ${MODEL} — retrying on sonnet"
  mark_log
  MODEL=sonnet   # the ledger append below must name what RAN (76th B1)
  nice -n 19 env TMPDIR=/data/tmp timeout "$TMOUT" claude -p "$(printf "REVIEW MODE TODAY: %s\n\n" "$MODE"; cat "$REPO/scripts/review_prompt.md")" \
    --model "$MODEL" --dangerously-skip-permissions --max-turns "$MAXTURNS" >> "$LOG" 2>&1
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
usage_ledger review end "$MODEL"      # D15 (d): after any retry, one line whatever RC says;
                                      # MODEL tracks the retry branch so this names what RAN
# A LIMIT REFUSAL AT THE DOOR IS NOT A HELD SITTING (2026-09-18). The SAT stamp
# above deliberately treats a mid-run death as spent budget — right, and kept.
# But on 2026-09-18 the 15:22 retry was refused by the CLI in 8 seconds on the
# session-limit wording, spent nothing, wrote nothing — and the stamp still
# consumed the day's one deferred sitting while the meter sat at 35% and the
# limit itself reset at 16:20. lib_credits.sh has recognised that wording since
# the 14th audit; this script sourced it and never acted on it. So: rc!=0 AND a
# limit wording in this run's OWN log bytes (bounded read, after the last
# mark_log) AND the whole chain over inside 120 s — every recorded door-refusal
# is 3–9 s, a real sitting runs minutes — restores the deferral and un-stamps,
# so a later --retry poll may still hold the day's sitting through the same
# gates. A day whose every poll is refused ends with the deferral armed and no
# sitting, which the INCOMPLETE trend row below still reports as a labelled
# hole. The 90% stop and the once-a-day cap on HELD sittings are unchanged.
if [ "$RC" -ne 0 ] && limit_hit && [ $(( $(date +%s) - RUN_START )) -le 120 ]; then
  echo "$TODAY" > "$DEFER"
  rm -f "$SAT"
  say "sitting REFUSED at the door by a usage limit in $(( $(date +%s) - RUN_START ))s — nothing was held; deferral restored so a later --retry poll may hold today's sitting (door-refusal is not a sitting)"
fi
# A run that dies LATE has already written its report -> stamp it a DRAFT. A run
# that dies BEFORE writing leaves a clean file that is nonetheless no longer
# current state -> stamp it STALE, but only once it is older than this organ's
# 24 h cadence (both branches in scripts/lib_seal.sh; the 08-30 FULL death is
# the scar for the second one).
# THE TAIL RECEIPT (D25's armed default, fired 2026-09-14). The append to
# docs/PROGRESS_LOG.md below is the LAST item on this organ's checklist, so its
# presence is the one honest signal available at seal time that the run reached
# the end of its own list before the wall clock killed it — the 2026-09-06
# FULL's row landed at 07:12 and `timeout` killed it at 07:17, and the seal
# called that finished page a draft. Passed as a pattern rather than hard-coded
# in lib_seal.sh so the library stays organ-agnostic. NOTE THE ORDER: this call
# precedes the INCOMPLETE-row fallback below, so a row found here is the
# agent's own — and lib_seal.sh refuses an INCOMPLETE row as a receipt anyway.
seal_output "$RC" docs/PROGRESS.md review say 25 "$RUN_START" \
  docs/PROGRESS_LOG.md "^\| $(date -u +%F) "
# The trend row must not die with the run (76th audit B4). The append to
# docs/PROGRESS_LOG.md sits at the END of the agent's checklist, and the last
# two runs both died before reaching it — so the file that exists "so trends
# survive any single Review" lost its 09-05 row entirely, and a trend file
# that only records the runs that finished over-reports this desk's own
# throughput, which is the exact quantity D22 turns on. A dead run writes a
# row that SAYS it is a hole; a completed run's own append is untouched (the
# grep guard means this never duplicates one the agent already wrote).
if [ "$RC" -ne 0 ] && ! grep -q "^| $(date -u +%F) " docs/PROGRESS_LOG.md 2>/dev/null; then
  echo "| $(date -u +%F) | $MODE | — | — | — | — | INCOMPLETE — the ${MODE} run exited rc=${RC} before appending its own row; written by review.sh (76th audit B4) so the trend has a labelled hole instead of a silent gap. The exit code is in review.log; any sealed draft is bannered in docs/PROGRESS.md. |" >> docs/PROGRESS_LOG.md
  git add -- docs/PROGRESS_LOG.md 2>/dev/null
  git commit -q -m "review: rc=${RC} run recorded as an INCOMPLETE row in PROGRESS_LOG.md (76th audit B4)

Committed by review.sh, not by the organ's agent, because the agent died
before its own append. A trend file that only records the runs that finished
over-reports the desk's throughput." -- docs/PROGRESS_LOG.md 2>/dev/null \
    && say "wrote the INCOMPLETE trend row for $(date -u +%F)" \
    || say "WARNING: could not commit the INCOMPLETE trend row"
fi
say "sweep end rc=${RC} — $(grep -c STRENGTHEN docs/PROGRESS.md 2>/dev/null || echo 0) strengthen lines"
exit 0
