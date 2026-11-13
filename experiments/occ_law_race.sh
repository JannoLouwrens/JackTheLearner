#!/bin/sh
# LAB occlusion-law race launcher (ORDERS J6; no ledger row). Rule declared in
# experiments/occ_law_race.py's docstring before any row is read.
# Waits for J5's CR.00 pilot to exit, then 4 arms x 5 seeds on 2 workers, nice 19.
cd /home/opc/jackthelearner
O=/data/tmp/occrace; mkdir -p $O
while pgrep -f 'cr00_pilot_race.sh' > /dev/null; do sleep 60; done
echo "cr00 gone $(date -u +%T)"
for s in 0 1 2 3 4; do for a in shipped graded damped open; do echo "$a $s"; done; done |
  xargs -P 2 -L 1 sh -c 'nice -n 19 /data/venvs/jackthelearner/bin/python -m experiments.occ_law_race $0 $1 '$O'/$0_s$1.jsonl > '$O'/run_$0_s$1.log 2>&1; echo "DONE $0 $1 rc=$? $(date -u +%T)"'
/data/venvs/jackthelearner/bin/python -m experiments.occ_law_race tally $O
echo "OCCRACE EXIT"
