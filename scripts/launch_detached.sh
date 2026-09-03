#!/bin/sh
# launch_detached.sh LOGFILE CMD [ARGS...] — detach a long-running job and
# PROVE it survived its imports before reporting success.
#
# This is the mechanical form of the LESSONS rule "verify the artifact ~10 s
# after launch" (overseer 21st audit B1). Four detached launches in 24 h died
# at import with exit 0 at the launch site and a 0-byte log, and three
# iterations idled waiting on processes that did not exist. The failure modes
# it closes, each of which has actually happened here:
#   - died at import from a wrong cwd  -> cwd is pinned to the repo
#   - traceback into a severed stderr  -> 2>&1 into the log
#   - reaped with the launching session -> setsid
#   - "launched" believed on exit code -> at 15 s the PROCESS must be alive
#     and the LOG non-empty, or this script exits 1 loudly with the log tail
#
# The header line written before exec makes "log non-empty" meaningful even
# when the payload buffers its stdout. Meant for jobs that run minutes or
# hours: a job that legitimately finishes in under 15 s is reported DEAD —
# read the printed log tail to tell success from a crash.
set -u
REPO=/home/opc/jackthelearner
[ $# -ge 2 ] || { echo "usage: launch_detached.sh LOGFILE CMD [ARGS...]" >&2; exit 2; }
LOG=$1; shift
cd "$REPO" || { echo "REFUSED: cannot cd $REPO" >&2; exit 2; }
: > "$LOG" || { echo "REFUSED: cannot write $LOG" >&2; exit 2; }
echo "LAUNCH $(date -u +%FT%TZ) cwd=$REPO cmd: $*" >> "$LOG"
# ADMIT, THEN METER (T0.34). The day's CPU ledger (experiments/cpu_budget.py,
# T0.33's file) refuses a NEW detached launch from an exhausted or overloaded
# day — loudly, before anything detaches — and once admitted the wrapper
# bills measured wall clock per heartbeat, split across the calendar days it
# spans. Accounting never kills a running child; overruns are marked. An
# unreachable venv python refuses too: a meter that fails open is not a
# limit (T0.12's rule).
PYBIN=/data/venvs/jackthelearner/bin/python
LABEL="${JACK_AWAITING_SPEC:-detached:$(basename "$LOG")}"
admit_msg=$("$PYBIN" -m experiments.cpu_budget admit "$LABEL" 2>&1) || {
    echo "REFUSED by the CPU day budget (T0.34): $admit_msg" >&2
    exit 3
}
setsid nice -n 19 "$PYBIN" -m experiments.cpu_budget wrap "$LABEL" "$@" >> "$LOG" 2>&1 < /dev/null &
pid=$!
# DECLARE IT. A detached run is compute this system MEANT to leave behind, and
# the loop's leftover check (scripts/lib_procwatch.sh, 52nd audit B2) must be
# able to tell it from the orphan that cost 1.26 core-hours. Undeclared is not
# fatal — it prints a LEFTOVER line naming this command, which is the correct
# failure direction for a guard that never kills.
. "$REPO/scripts/lib_procwatch.sh" 2>/dev/null && proc_declare "$pid" "launch_detached $LOG: $*" 2>/dev/null
# AND CLAIM THE RESULT, not just the process (67th audit B2). When the caller
# names the spec this launch is buying (JACK_AWAITING_SPEC=LF.01 ...), an
# AWAITING row is written beside the pid declaration; `run next` then refuses
# to select new work while that row has neither a ledger entry nor a live pid.
# A launch without the env var behaves exactly as before — but a registered
# run launched without it is a handoff that exists only in prose, which is
# the scar this closes.
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
