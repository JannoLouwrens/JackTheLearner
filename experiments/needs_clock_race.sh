#!/bin/sh
# J6(b) needs-clock race launcher (no ledger row): waits for W1.00 attempt 3's wrapper (pid $1) to exit,
# then 2 workers, nice 19. W.2 cells first (cheap), then NE.01 (~40 min/seed on this box). Tally at the end.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "gate pid $1 gone $(date -u +%FT%TZ), race starts"
mkdir -p /data/jack-logs/needs_race
for spec in W.2 NE.01; do for s in 0 1 2 3 4; do for a in shipped k72; do echo "$a $spec $s"; done; done; done |
  xargs -P 2 -L 1 sh -c 'f=/data/jack-logs/needs_race/$0_$1_s$2.json; [ -s $f ] && exit 0;
    t=$(date +%s); nice -n 19 '"$PY"' -m experiments.needs_clock_race run $0 $1 $2 > /data/jack-logs/needs_race/log_$0_$1_s$2.txt 2>&1;
    echo "DONE $0 $1 s$2 rc=$? $(( $(date +%s) - t ))s $(date -u +%T)"'
$PY -m experiments.needs_clock_race tally
echo "NEEDS RACE EXIT"
