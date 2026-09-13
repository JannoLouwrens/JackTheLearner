#!/bin/bash
# dispatch.sh — launch a GPU-dispatching spec DETACHED from the calling session.
#
# Why this exists: a dispatch watcher launched as a child of a Claude session
# dies when the session dies, and the session WILL die under it (T2.01 v3's
# waiter, SIGPIPEd at ~80 min; T2.04 on 2026-08-14, watcher dead 53 min into a
# 1 h kernel). The kernel keeps computing either way — only the process waiting
# to fetch and record the result is lost, and recovery costs an iteration of
# archaeology. setsid+nohup orphans the watcher from the session so the result
# lands in the ledger no matter what happens to the caller.
#
# Usage:
#   scripts/dispatch.sh T2.04 --projected-hours 0.5             # fresh dispatch
#   JACK_REUSE_KERNEL=jack-ladder-<ts> scripts/dispatch.sh T2.04 --projected-hours 0
#                                                               # reattach (free)
#
# Prints the log path and the reattach command. Poll the log; do NOT relaunch
# while the pid it prints is alive.
#
# THE REFUSALS (91st audit, RANK 1 / B1, 2026-09-13). Until today this script
# refused exactly two things — an unpushed HEAD and a held lock — while three
# separate documents said `D1.0` may not be re-dispatched unchanged and its two
# attempts had already spent 33.78 GPU-hours for one VOID row. The prohibitions
# were excellent and nothing read them: "a prohibition is a sentence in a
# document; a refusal is a branch that returns non-zero" (LESSONS.md, 09-13).
# The budget/authorisation/projection refusals live in
# `experiments/dispatch_guard.py` — Python, because they read JSON accounting
# and a module's declared IMPL_DEPS, and because a refusal that cannot be
# exercised against a constructed case is not a refusal anyone has tested.
# A CLEAR pre-flight means no rule refused this; it is never an argument that
# the dispatch is worth making.
set -eu
SPEC="${1:?usage: dispatch.sh <SPEC_ID> --projected-hours <H>}"
shift
PROJECTED=""
while [ $# -gt 0 ]; do
    case "$1" in
        --projected-hours) PROJECTED="${2:?--projected-hours needs a value}"; shift 2 ;;
        --projected-hours=*) PROJECTED="${1#*=}"; shift ;;
        *) echo "REFUSING: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=/data/venvs/jackthelearner/bin/python
LOG="/data/tmp/dispatch_$(echo "$SPEC" | tr '.' '_' | tr '[:upper:]' '[:lower:]').log"

cd "$REPO"

# A REATTACH IS EXEMPT, and the carve-out is named rather than left implicit.
# `JACK_REUSE_KERNEL` recovers a kernel that is already running or finished:
# it buys no fresh GPU time (`gpu.submit` skips `afford()` for exactly this
# reason — the affordability gate once turned a zero-cost T2.01 v4 recovery
# into a Colab failover and an ERROR), and refusing it as an "unchanged
# re-dispatch" would forbid the recovery path itself. The guard is skipped,
# loudly. Everything else goes through it.
REUSE="${JACK_REUSE_KERNEL:-}"
if [ -n "$REUSE" ]; then
    echo "pre-flight SKIPPED: JACK_REUSE_KERNEL=$REUSE — a reattach buys no" >&2
    echo "fresh quota and is not a re-dispatch. Budget/authorisation refusals" >&2
    echo "do not apply; the kernel must already exist." >&2
else
    # REFUSAL 3 — state the projection. Without a number, the budget refusal
    # cannot be computed and projected-vs-actual can never be audited: `D1.0`'s
    # 17.61 h estimate has existed only in prose for a week.
    if [ -z "$PROJECTED" ]; then
        echo "REFUSING: no --projected-hours. State what this dispatch is expected" >&2
        echo "to cost in GPU-hours; it is checked against the week's remaining quota" >&2
        echo "and recorded beside the job so projected-vs-actual is auditable." >&2
        echo "Run: scripts/dispatch.sh $SPEC --projected-hours <H>" >&2
        exit 1
    fi
fi
if [ "$(git rev-list --count origin/main..HEAD)" != 0 ]; then
    echo "REFUSING: HEAD is not pushed and the GPU VM clones from GitHub." >&2
    echo "Run: git push   (owner answered D3: yes, push)" >&2
    exit 1
fi

# A held GPU lock makes the runner print "Wait for it" and exit ZERO — and it
# takes >2s of imports to get there, so the liveness check below reports
# success for a watcher that is already doomed (T2.04, 2026-08-19: dispatched
# beside T2.03, "watcher pid (detached)" printed, watcher dead on the lock
# seconds later, nothing queued). Refuse loudly up front instead.
GPULOCK=/tmp/jack-ladder-gpu.lock
if [ -e "$GPULOCK" ] && ! flock -n "$GPULOCK" true 2>/dev/null; then
    # NB: the pid inside the file is unreliable — every failed contender opens
    # it with mode "w" and truncates it. lsof sees who holds it open.
    echo "REFUSING: $GPULOCK is held (holder pid $(lsof -t "$GPULOCK" 2>/dev/null | tr '\n' ' ' || true))." >&2
    echo "GPU runs are serialised; a second one exits without queuing." >&2
    echo "Wait for the holder, then re-run: scripts/dispatch.sh $SPEC" >&2
    exit 1
fi

# REFUSALS 1 and 2 — budget, and unchanged re-dispatch. Deliberately the LAST
# refusal: `--record` writes a projection receipt to `gpu_budget.json`, and the
# receipt log must mean "was allowed to go", not "was considered". Running it
# ahead of the push/lock checks would file a projection for a dispatch that
# never left. Non-zero here means the dispatch does not happen; the guard
# prints its own arithmetic either way.
if [ -z "$REUSE" ]; then
    if ! "$PY" -m experiments.dispatch_guard "$SPEC" \
            --projected-hours "$PROJECTED" --record; then
        exit 1
    fi
fi

setsid nohup "$PY" -m experiments.run "$SPEC" >"$LOG" 2>&1 </dev/null &
PID=$!
# Declared, so the loop's leftover check reads this watcher (and the runner it
# forks) as intended compute rather than a stranded orphan — see
# scripts/lib_procwatch.sh.
. "$REPO/scripts/lib_procwatch.sh" 2>/dev/null && proc_declare "$PID" "dispatch $SPEC" 2>/dev/null
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "watcher died immediately — read $LOG" >&2
    tail -5 "$LOG" >&2
    exit 1
fi
echo "watcher pid $PID (detached), log $LOG"
echo "if this watcher dies mid-run: find the slug in experiments/gpu_submissions.jsonl"
echo "(last 'attempt' row for $SPEC), verify the kernel with 'kaggle kernels status',"
echo "then: JACK_REUSE_KERNEL=<slug> scripts/dispatch.sh $SPEC"
