#!/bin/sh
# launch_detached.sh LOGFILE CMD [ARGS...] — detach a long-running job and
# PROVE it survived its imports before reporting success.
#
# Closes four failure modes (LESSONS "verify the artifact"): wrong cwd
# (pinned), severed stderr (2>&1), reaped with the session (setsid), and
# "launched" believed on exit code (at 15 s the process must be alive and the
# log non-empty, else exit 1 with the log tail). A job that legitimately
# finishes under 15 s is reported DEAD — read the tail.
set -u
MAIN=/home/opc/jackthelearner
# JACK_REPO (S7 step 2): launch from a clean `git worktree` of this repo, so
# another session's uncommitted files in the main tree cannot reach the run.
# Anything that is not a worktree of MAIN is refused; the CPU day budget stays
# MAIN's file (a per-worktree budget would split the day's limit).
REPO="${JACK_REPO:-$MAIN}"
if [ "$REPO" != "$MAIN" ]; then
    common=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
    [ "$common" = "$MAIN/.git" ] || { echo "REFUSED: JACK_REPO=$REPO is not a worktree of $MAIN" >&2; exit 2; }
    export JACK_CPU_BUDGET="${JACK_CPU_BUDGET:-$MAIN/experiments/cpu_budget.json}"
fi
export JACK_REPO="$REPO"
[ $# -ge 2 ] || { echo "usage: launch_detached.sh LOGFILE CMD [ARGS...]" >&2; exit 2; }
LOG=$1; shift
cd "$REPO" || { echo "REFUSED: cannot cd $REPO" >&2; exit 2; }
: > "$LOG" || { echo "REFUSED: cannot write $LOG" >&2; exit 2; }
echo "LAUNCH $(date -u +%FT%TZ) cwd=$REPO cmd: $*" >> "$LOG"
# ADMIT, THEN METER (T0.34): the day's CPU ledger refuses a new launch from an
# exhausted/overloaded day or full seats (fails closed); once admitted the
# wrapper bills wall per heartbeat, marking overruns, never killing. The
# read-only `run lane` probe (test_lane_guard.sh) takes no seat (S7(4)).
PYBIN=/data/venvs/jackthelearner/bin/python
LABEL="${JACK_AWAITING_SPEC:-detached:$(basename "$LOG")}"
SEAT=; [ "$*" = "$PYBIN -m experiments.run lane" ] && SEAT=--no-seat
admit_msg=$("$PYBIN" -m experiments.cpu_budget admit "$LABEL" "${JACK_DETACHED_EST_S:-0}" $SEAT 2>&1) || {
    echo "REFUSED by the CPU day budget (T0.34): $admit_msg" >&2
    exit 3
}
# env -u: the session's iteration deadline must not bind a setsid child.
# JACK_DETACHED_LANE marks every spend launched here as detached (visibility
# only; the run.py lane guard cannot see past `cpu_budget wrap`).
setsid nice -n 19 env -u JACK_ITER_DEADLINE JACK_DETACHED_LANE="launch_detached.sh $LOG" "$PYBIN" -m experiments.cpu_budget wrap "$LABEL" "$@" >> "$LOG" 2>&1 < /dev/null &
pid=$!
# DECLARE IT, so lib_procwatch's leftover check can tell it from an orphan.
. "$MAIN/scripts/lib_procwatch.sh" 2>/dev/null && proc_declare "$pid" "launch_detached $LOG: $*" 2>/dev/null
# AND CLAIM THE RESULT: JACK_AWAITING_SPEC=<id> writes an AWAITING row; `run
# next` refuses new work while it has neither a ledger entry nor a live pid.
if [ -n "${JACK_AWAITING_SPEC:-}" ]; then
    proc_await "$JACK_AWAITING_SPEC" "$pid" "launch_detached $LOG" 2>/dev/null \
        && echo "AWAITING $JACK_AWAITING_SPEC since $(date -u +%FT%T) pid=$pid"
fi
sleep 15
alive=$(ps -o args= -p "$pid" 2>/dev/null || true)
bytes=$(wc -c < "$LOG")
if [ -z "$alive" ]; then
    echo "DEAD at 15s: pid $pid gone. Log $LOG ($bytes bytes) tail:" >&2
    tail -20 "$LOG" >&2
    exit 1
fi
if [ "$bytes" -le 80 ]; then
    # only the header (or nothing): the payload has not written a byte.
    # Alive-but-silent is not proof of death, but it is not proof of life
    # either — warn, do not fail; the caller owns the follow-up check.
    echo "WARN: pid $pid alive but log has only the header at 15s" >&2
fi
echo "ALIVE pid=$pid log=$LOG bytes=$bytes"
echo "  cmd: $alive"
echo "  meter: $admit_msg (label $LABEL)"
exit 0
