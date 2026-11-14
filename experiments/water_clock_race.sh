#!/bin/sh
# J6(c) water-clock race launcher (no ledger row). Launch via scripts/launch_detached.sh.
# 2 workers, nice 19. Apparatus replays first (sweat_paid W.2/NE.01 s0), then W.1, W.2, NE.01 per arm. Tally at the end.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
D=/data/jack-logs/water_race
mkdir -p $D
{ echo "sweat_paid W.2 0"; echo "sweat_paid NE.01 0";
  for spec in W.1 W.2 NE.01; do for s in 0 1 2 3 4; do for a in swp_w72 swp_k72; do echo "$a $spec $s"; done; done; done; } |
  xargs -P 2 -L 1 sh -c 'f='"$D"'/$0_$1_s$2.json; [ -s $f ] && exit 0;
    t=$(date +%s); nice -n 19 '"$PY"' -m experiments.water_clock_race run $0 $1 $2 > '"$D"'/log_$0_$1_s$2.txt 2>&1;
    echo "DONE $0 $1 s$2 rc=$? $(( $(date +%s) - t ))s $(date -u +%T)"'
$PY -m experiments.water_clock_race tally
echo "WATER RACE EXIT"
