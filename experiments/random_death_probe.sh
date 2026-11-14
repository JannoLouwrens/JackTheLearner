#!/bin/sh
# J6(c) gate-2 probe launcher (no ledger row). Launch via scripts/launch_detached.sh. 2 workers, nice 19.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
D=/data/jack-logs/random_death_probe
mkdir -p $D
printf '%s\n' "swp_k72 0" "hum34 0" "hum30 0" "hum34 1" "hum30 1" "hum34 2" |
  xargs -P 2 -L 1 sh -c 'f='"$D"'/$0_s$1.json; [ -s $f ] && exit 0;
    t=$(date +%s); nice -n 19 '"$PY"' -m experiments.random_death_probe run $0 $1 > '"$D"'/log_$0_s$1.txt 2>&1;
    echo "DONE $0 s$1 rc=$? $(( $(date +%s) - t ))s $(date -u +%T)"'
$PY -m experiments.random_death_probe tally
echo "PROBE EXIT"
