#!/bin/sh
# j10_chain.sh SEED SEG WAIT_PID — ORDERS J10 (2), ruling 109: the next segment
# of a J10-2 seed must not wait for a worker to be awake.
#
# Waits (sleeping; no CPU) for WAIT_PID — the watcher of segment SEG-1 — to
# exit, then launches `humanoid_sac_row kernel2 SEED SEG` from the clean
# worktree through launch_detached.sh, exactly as a worker would by hand.
# REFUSES (and says so in its log) unless segment SEG-1's record is on disk
# with hit_target true and a kernel slug: a segment that fell short is not
# resumed as if it had finished. One link only; it never retries a refusal.
#   setsid sh scripts/j10_chain.sh 7 3 <pid> >> /data/jack-logs/j10/chain_s7.log 2>&1 &
set -u
SEED=$1; SEG=$2; WAIT_PID=$3
MAIN=/home/opc/jackthelearner
WT=/data/wt/jack-clean
PY=/data/venvs/jackthelearner/bin/python
PREV=/data/jack-logs/j10/sac_row2/s${SEED}_seg$((SEG - 1)).json
THIS=/data/jack-logs/j10/sac_row2/s${SEED}_seg${SEG}.json
say() { echo "$(date -u +%FT%TZ) j10_chain s$SEED seg$SEG: $*"; }
say "waiting on pid $WAIT_PID ($(ps -o args= -p "$WAIT_PID" 2>/dev/null | cut -c1-120))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
say "pid $WAIT_PID gone"
[ -e "$THIS" ] && { say "REFUSED: $THIS already exists"; exit 0; }
if pgrep -f "humanoid_sac_row kernel2 $SEED $SEG\$" >/dev/null; then
    say "REFUSED: kernel2 $SEED $SEG is already running"; exit 0
fi
ok=$("$PY" -c "
import json,sys
r=json.load(open(sys.argv[1]))
print('OK' if r.get('hit_target') is True and r.get('kernel') and r.get('segment')==int(sys.argv[2]) else 'NO %r'%({k:r.get(k) for k in ('segment','end','target','hit_target','kernel')},))
" "$PREV" "$((SEG - 1))" 2>&1)
case "$ok" in OK) ;; *) say "REFUSED: previous segment record $PREV: $ok"; exit 1;; esac
say "previous segment home and on target -> launching"
JACK_REPO=$WT "$MAIN/scripts/launch_detached.sh" "$WT/launch_j10_2_s${SEED}_seg${SEG}.log" \
    "$PY" -m experiments.humanoid_sac_row kernel2 "$SEED" "$SEG"
say "launch_detached rc=$? (log $WT/launch_j10_2_s${SEED}_seg${SEG}.log; receipt in $WT/experiments/gpu_submissions.jsonl)"
