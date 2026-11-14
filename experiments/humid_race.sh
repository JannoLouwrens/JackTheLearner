#!/bin/sh
# J6(c) humidity race launcher (no ledger row). Launch via scripts/launch_detached.sh.
# Waits for the water race to exit (box cap: 2 of our workers), then 2 workers, nice 19.
# Apparatus first (swp_k72 W.2 s0 through humid_race._apply), then W.1, W.2, NE.01 per arm. Tally at the end.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
D=/data/jack-logs/humid_race
mkdir -p $D
while pgrep -f "experiments.water_clock_race run" >/dev/null; do sleep 60; done
echo "START $(date -u +%T)"
{ echo "swp_k72 W.2 0";
  for spec in W.1 W.2 NE.01; do for s in 0 1 2 3 4; do for a in hum30 hum34; do echo "$a $spec $s"; done; done; done; } |
  xargs -P 2 -L 1 sh -c 'f='"$D"'/$0_$1_s$2.json; [ -s $f ] && exit 0;
    t=$(date +%s); nice -n 19 '"$PY"' -m experiments.humid_race run $0 $1 $2 > '"$D"'/log_$0_$1_s$2.txt 2>&1;
    echo "DONE $0 $1 s$2 rc=$? $(( $(date +%s) - t ))s $(date -u +%T)"'
$PY -m experiments.humid_race tally
echo "HUMID RACE EXIT"
