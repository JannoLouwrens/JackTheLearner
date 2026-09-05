#!/bin/bash
# The OVERSEER — adversarial audit of the ladder system.
#
# Runs INDEPENDENTLY of the builder loop, on purpose. A builder cannot audit
# itself: it is optimistic by construction (it wants specs to pass), and the
# failures that matter most here are the ones that look like success —
# a threshold quietly loosened, a control that never ran, GPU hours spent with
# nothing to show, a green ladder drifting away from GOAL.md.
#
# It runs at :37, half an hour offset from the builder's :07, so the two are
# never competing for the box. It takes NO lock: it only reads, and it must be
# able to report on a builder that is currently wedged.
#
# Install:  37 */6 * * *  /home/opc/jackthelearner/scripts/overseer.sh
# Read:     docs/OVERSIGHT.md   (current-state report, rewritten each run)
# Stop:     touch /home/opc/jackthelearner/.overseer-paused
set -uo pipefail

REPO=/home/opc/jackthelearner
LOGDIR=/data/jack-logs
LOCK=/tmp/jack-overseer.lock
PAUSE="$REPO/.overseer-paused"
MIN_FREE_GB=2

mkdir -p "$LOGDIR"
LOG="$LOGDIR/overseer.log"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }
. "$REPO/scripts/lib_credits.sh"
. "$REPO/scripts/lib_usage.sh"
. "$REPO/scripts/lib_pause.sh"
. "$REPO/scripts/lib_seal.sh"
. "$REPO/scripts/lib_liveness.sh"

pause_gate say "$PAUSE" || exit 0

exec 9>"$LOCK"
flock -n 9 || { say "previous audit still running — skipping"; exit 0; }

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ] && { say "ABORT: ${FREE_GB}GB free"; exit 0; }

# STOP AT 90%. Owner's rule, 2026-08-09: pause ALL agents at 90% weekly usage
# until the owner resumes them. Real number from `claude -p /usage`, not a
# proxy. Nothing else is throttled — this is the only limit.
# UNKNOWN IS NOT ZERO: if usage cannot be read, do NOT run. A meter that fails
# open is not a limit.
usage_gate say || exit 0
cd "$REPO" || exit 0

# Declare this audit to procwatch (65th audit B5): the overseer's read-only
# instrument calls are project pythons, and undeclared they are
# indistinguishable from an abandoned builder leftover — the 2026-09-03
# LEFTOVER=1 was this organ's own `python -m experiments.coverage` at 0 s CPU.
# Children of a declared pid are attributed through the ppid chain
# (lib_procwatch.sh), and dead declarations are pruned, so one line here
# covers every call this run makes and outlives none of them.
. "$REPO/scripts/lib_procwatch.sh"
proc_declare $$ "overseer.sh audit slot"

# D13's CHANGE-GATED NO-OP (default fired 2026-09-01; DECIDE block in
# docs/DECISIONS_NEEDED.md). Skip this slot ONLY when ALL FOUR hold:
#   (1) HEAD unchanged since the last COMPLETED audit;
#   (2) zero builder iteration starts in ladder.log since it;
#   (3) no OPEN decision's decide_by falls before the next slot — read from
#       `experiments.decisions`, NOT by grepping decide_by: resolved entries
#       keep their past dates in the file forever, so a raw grep would trip
#       on history and quietly turn this no-op off for good;
#   (4) fewer than 3 consecutive slots already skipped (the organ can never
#       go dark past 24 h on its own decision).
# The state file records the last completed audit; a run that dies does not
# update it, so a dead audit forces the next slot to run in full — the guard
# fails toward MORE oversight, never less.
NOOP_STATE="$LOGDIR/overseer_noop.state"    # "<head> <iso-ts> <skips>"
noop_eligible() {
  [ -f "$NOOP_STATE" ] || return 1
  read -r LAST_HEAD LAST_TS SKIPS < "$NOOP_STATE" || return 1
  [ -n "$LAST_HEAD" ] && [ -n "$LAST_TS" ] || return 1
  [ "${SKIPS:-3}" -lt 3 ] || return 1                              # (4)
  [ "$(git rev-parse HEAD)" = "$LAST_HEAD" ] || return 1           # (1)
  ITER=$(awk -v ts="$LAST_TS" '$1 > ts && /iteration start/' \
         "$LOGDIR/ladder.log" 2>/dev/null | wc -l)
  [ "${ITER:-1}" -eq 0 ] || return 1                               # (2)
  NEXT_SLOT=$(( $(date +%s) + 6*3600 ))                            # (3)
  DUES=$(/data/venvs/jackthelearner/bin/python -m experiments.decisions \
         2>/dev/null | grep -oE 'OVERDUE — DEFAULT IS DUE TO FIRE|due [0-9]{4}-[0-9]{2}-[0-9]{2}')
  [ -n "$DUES" ] || return 1     # unreadable is not quiet — audit runs
  echo "$DUES" | grep -q OVERDUE && return 1
  while read -r line; do
    d="${line#due }"
    [ -n "$d" ] || continue
    [ "$(date -d "$d" +%s 2>/dev/null || echo 0)" -lt "$NEXT_SLOT" ] && return 1
  done <<< "$(echo "$DUES" | grep '^due' )"
  return 0
}
if noop_eligible; then
  review_liveness say || true    # the Review watch never lapses with the audit
  echo "$LAST_HEAD $LAST_TS $((SKIPS + 1))" > "$NOOP_STATE"
  say "overseer: no-op, HEAD $(git rev-parse --short HEAD) unchanged and 0 builder iterations since ${LAST_TS} (skip $((SKIPS + 1))/3, D13 change-gated)"
  exit 0
fi

MODEL="${JACK_OVERSEER_MODEL:-opus}"
# Turn budget derived from the clock, at the Review's own rate of 3 turns/min
# (60 turns / 20 min), instead of the hard-coded 60 that killed this organ twice
# with time still on it. `timeout` remains the spend ceiling; this only stops an
# early death by a number nothing derived. See scripts/review.sh.
MAXTURNS=75
say "audit start — model ${MODEL}, $(git rev-parse --short HEAD)"

# WATCH THE ORGAN NEXT DOOR, BEFORE AUDITING ANYTHING. The Review's own
# instruments cannot see it fail to run — two of its three Sunday FULL slots
# were refused by the usage gate before `review.sh` executed a line, and the
# third died before writing. This runs 4x/day, takes no lock and only reads, so
# it is the right place: an organ that is the destination of routed work must
# have its liveness watched by something other than itself (27th audit's
# corollary; scripts/lib_liveness.sh is where the scar is written up).
#
# It runs BEFORE the agent so the audit sees the banner in its own working tree,
# and it is deliberately not gated on RC: a dead Review is a finding whether or
# not this audit completes.
review_liveness say || true

# The seal's sweep bound (74th audit B1): only dirty files whose mtime is at or
# after this moment are this run's own acts. Captured before the agent starts.
RUN_START=$(date +%s)
mark_log

nice -n 19 ionice -c3 env TMPDIR=/data/tmp \
  timeout 25m claude -p "$(cat "$REPO/scripts/overseer_prompt.md")" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --max-turns "$MAXTURNS" \
    >> "$LOG" 2>&1
RC=$?

# Same credit-exhaustion trap the builder fell into: the CLI prints a message
# and exits in ~3s, which looks identical to a clean fast run.
if credits_out; then
  say "OUT OF CREDITS on ${MODEL} — audit skipped, not performed"
  exit 0
fi

# OVERSIGHT.md is the PREVIOUS audit's file until this run rewrites it. A run
# that died (rc!=0) did not write it, so grepping it publishes the STALE verdict
# in the vocabulary of success — a dead audit logged "ON TRACK" on 2026-08-24
# 12:37 after 2s on a session limit (Review 2026-08-25, FOR THE BUILDER 1).
#
# AND THE OTHER HALF, added 2026-08-30 because the sentence above is only true
# of a run that dies EARLY. The 49th audit died at max turns AFTER writing the
# whole file, so the log said UNKNOWN while docs/OVERSIGHT.md said ON TRACK and
# nothing joined them. `seal_output` stamps the file itself and commits it.
#
# AND THE THIRD CASE, 2026-08-31: a run that dies before writing leaves a CLEAN
# file, which the seal skipped. 7 h = this organ's 6-hourly cadence plus an
# hour, so a 06:37 report is not stamped stale when the 12:37 run dies — only a
# page that has actually outlived the schedule is.
if [ "$RC" -ne 0 ]; then
  seal_output "$RC" docs/OVERSIGHT.md overseer say 7 "$RUN_START"
  say "audit end rc=${RC} — verdict: UNKNOWN (audit did not complete)"
  exit 0
fi
VERDICT=$(grep -m1 -oE "ON TRACK|DRIFTING|INTEGRITY RISK" docs/OVERSIGHT.md 2>/dev/null || echo "no-verdict")
say "audit end rc=${RC} — verdict: ${VERDICT}"
# A COMPLETED audit is the only thing that resets D13's no-op state: skips
# back to 0, HEAD and timestamp stamped. Dead runs leave it stale on purpose.
echo "$(git rev-parse HEAD) $(date -Iseconds) 0" > "$NOOP_STATE"
exit 0
