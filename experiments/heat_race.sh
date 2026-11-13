#!/bin/sh
# J6(c) heat race launcher (no ledger row). Launch via scripts/launch_detached.sh (the CPU day gate admits it).
# 2 workers, nice 19. W.1 cells are already on disk (3 s each); W.2 first (cheap), then NE.01 (~9 min wall/cell). Tally at the end.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
mkdir -p /data/jack-logs/heat_race
for spec in W.1 W.2 NE.01; do for s in 0 1 2 3 4; do for a in shipped sweat sweat_paid; do echo "$a $spec $s"; done; done; done |
  xargs -P 2 -L 1 sh -c 'f=/data/jack-logs/heat_race/$0_$1_s$2.json; [ -s $f ] && exit 0;
    t=$(date +%s); nice -n 19 '"$PY"' -m experiments.heat_race run $0 $1 $2 > /data/jack-logs/heat_race/log_$0_$1_s$2.txt 2>&1;
    echo "DONE $0 $1 s$2 rc=$? $(( $(date +%s) - t ))s $(date -u +%T)"'
$PY -m experiments.heat_race tally
echo "HEAT RACE EXIT"
