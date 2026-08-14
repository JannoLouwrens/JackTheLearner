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
#   scripts/dispatch.sh T2.04                        # fresh dispatch
#   JACK_REUSE_KERNEL=jack-ladder-<ts> scripts/dispatch.sh T2.04   # reattach
#
# Prints the log path and the reattach command. Poll the log; do NOT relaunch
# while the pid it prints is alive.
set -eu
SPEC="${1:?usage: dispatch.sh <SPEC_ID>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=/data/venvs/jackthelearner/bin/python
LOG="/data/tmp/dispatch_$(echo "$SPEC" | tr '.' '_' | tr '[:upper:]' '[:lower:]').log"

cd "$REPO"
if [ "$(git rev-list --count origin/main..HEAD)" != 0 ]; then
    echo "REFUSING: HEAD is not pushed and the GPU VM clones from GitHub." >&2
    echo "Run: git push   (owner answered D3: yes, push)" >&2
    exit 1
fi

setsid nohup "$PY" -m experiments.run "$SPEC" >"$LOG" 2>&1 </dev/null &
PID=$!
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
