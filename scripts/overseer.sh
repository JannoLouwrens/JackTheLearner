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

[ -f "$PAUSE" ] && { say "paused"; exit 0; }

exec 9>"$LOCK"
flock -n 9 || { say "previous audit still running — skipping"; exit 0; }

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ] && { say "ABORT: ${FREE_GB}GB free"; exit 0; }

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
MODEL="${JACK_OVERSEER_MODEL:-opus}"
say "audit start — model ${MODEL}, $(git rev-parse --short HEAD)"

nice -n 19 ionice -c3 \
  timeout 25m claude -p "$(cat "$REPO/scripts/overseer_prompt.md")" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --max-turns 60 \
    >> "$LOG" 2>&1
RC=$?

# Same credit-exhaustion trap the builder fell into: the CLI prints a message
# and exits in ~3s, which looks identical to a clean fast run.
if tail -5 "$LOG" | grep -qi "out of usage credits"; then
  say "OUT OF CREDITS on ${MODEL} — audit skipped, not performed"
  exit 0
fi

VERDICT=$(grep -m1 -oE "ON TRACK|DRIFTING|INTEGRITY RISK" docs/OVERSIGHT.md 2>/dev/null || echo "no-verdict")
say "audit end rc=${RC} — verdict: ${VERDICT}"
exit 0
