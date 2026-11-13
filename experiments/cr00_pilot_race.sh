#!/bin/sh
# LAB CR.00 pilot launcher (ORDERS J5 step 3; no ledger row): 3 arms, seed 89 (not a claim seed),
# train 40000, exam 20000, no smell (W0 as seated). Waits for the J1 wean-smell race to exit, then 3 workers.
# Purpose: set CR.01's N (exam lives) and margin from the null's life-to-life spread BEFORE claim seeds are read.
cd /home/opc/jackthelearner
O=/data/tmp/cr00; mkdir -p $O
while pgrep -f 'wean_smell_race.sh' > /dev/null; do sleep 60; done
echo "wean_smell gone $(date -u +%T)"
one() { nice -n 19 /data/venvs/jackthelearner/bin/python -m experiments.cr00_pilot "$1" 89 40000 20000 $O/$1_s89.jsonl > $O/run_$1_s89.log 2>&1; echo "DONE $1 rc=$? $(date -u +%T)"; }
one null & one staged & one shuffled; wait
/data/venvs/jackthelearner/bin/python -m experiments.cr00_pilot tally $O
echo "CR00 EXIT"
