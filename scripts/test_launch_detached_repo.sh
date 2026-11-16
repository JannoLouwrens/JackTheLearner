#!/bin/sh
# Planted test for S7 step 2: launch_detached.sh honours JACK_REPO.
#   (a) a real worktree of the repo: the child's cwd IS the worktree, and the
#       CPU day is billed to MAIN's budget file, not the worktree's copy;
#   (b) control that must fail: a directory that is not a worktree is REFUSED.
# Run: sh scripts/test_launch_detached_repo.sh   (exit 0 = green; ~20 s)
set -u
MAIN=/home/opc/jackthelearner; PY=/data/venvs/jackthelearner/bin/python
WT=/data/tmp/wt_s7test_$$; LOG=/data/tmp/s7test_$$.log; fail=0
git -C "$MAIN" worktree add -q --detach "$WT" HEAD || exit 1
trap 'git -C "$MAIN" worktree remove --force "$WT"; rm -f "$LOG" "$LOG.b"' EXIT
JACK_REPO="$WT" sh "$MAIN/scripts/launch_detached.sh" "$LOG" "$PY" -c \
  "import os,time; from experiments import cpu_budget as c; print('CWD', os.getcwd()); print('BUDGET', c._budget_path(), flush=True); time.sleep(17)" >/dev/null
sleep 1
grep -qx "CWD $WT" "$LOG" && echo "PASS (a) child cwd = worktree" || { echo "FAIL (a) cwd"; fail=1; }
grep -qx "BUDGET $MAIN/experiments/cpu_budget.json" "$LOG" && echo "PASS (a) budget = main's file" || { echo "FAIL (a) budget"; fail=1; }
JACK_REPO=/data/tmp sh "$MAIN/scripts/launch_detached.sh" "$LOG.b" true 2>/dev/null
rc=$?; [ $rc -eq 2 ] && echo "PASS (b) non-worktree refused" || { echo "FAIL (b) rc=$rc"; fail=1; }
[ $fail -eq 0 ] || cat "$LOG"
sleep 3; exit $fail
