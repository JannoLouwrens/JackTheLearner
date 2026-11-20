#!/bin/sh
# tpu_wait.sh — S12 (a): Kaggle's 5-session batch cap counts TPU sessions too, so the TPU gate kernel
# waits for a free one. Every 10 min: count our RUNNING/QUEUED kernels among the 12 newest (status
# calls only — no push, no receipt); below 5, run tpu_probe once and exit. Gives up at $1 (UTC epoch,
# default 21:30 today) so the run can finish before the TPU week lapses (Sat 00:00 UTC).
# One-shot for this week's gate; delete once the verdict is in.
K=/data/venvs/kaggle/bin/kaggle
PY=/data/venvs/jackthelearner/bin/python
STOP=${1:-$(date -u -d 'today 21:30' +%s)}
while [ "$(date +%s)" -lt "$STOP" ]; do
    n=0
    for k in $($K kernels list --mine --sort-by dateRun --page-size 12 2>/dev/null | awk '/^jannolouwrens\//{print $1}'); do
        $K kernels status "$k" 2>/dev/null | grep -qiE 'RUNNING|QUEUED' && n=$((n+1))
    done
    echo "$(date -u +%FT%TZ) busy=$n"
    if [ "$n" -lt 5 ]; then
        exec $PY -m experiments.brain_a.tpu_probe kernel
    fi
    sleep 600
done
echo "$(date -u +%FT%TZ) GAVE UP: no free Kaggle session before the deadline"
exit 4
