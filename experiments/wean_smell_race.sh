#!/bin/sh
# LAB 11-12: WEANING re-raced WITH smell on the repaired world (the LAB entry's own declared next step).
# wean_pilot arms nursed / decoy, smell_sense 'smell' installed, s91-93, nurse 26000, total 52000.
# Null = dose2 smell_s9x (same process, uniform births throughout; already on disk).
# RULE DECLARED BEFORE ANY ROW IS READ (experiments/wean_smell_tally.py): meals in lives with phase == "uniform" AND k > 26000.
# nursed ALIVE iff it out-eats decoy on >= 2 of 3 seeds AND out-eats null (dose2 smell 2/0/3) on >= 2 of 3; else DEAD.
cd /home/opc/jackthelearner
O=/data/tmp/wean_smell; mkdir -p $O
one() { nice -n 19 /data/venvs/jackthelearner/bin/python -c 'import sys; from experiments import smell_sense as S, wean_pilot; S.install("smell", int(sys.argv[2])); wean_pilot.main(sys.argv[1], int(sys.argv[2]), 26000, 52000, "/data/tmp/wean_smell/%s_s%s.jsonl" % (sys.argv[1], sys.argv[2]))' "$1" "$2" > $O/run_$1_s$2.log 2>&1; echo "DONE $1 $2 rc=$?"; }
one nursed 91 &
# the J1 gamma99 race holds 2 cores until ~01:40: wait for it, then two at a time
while pgrep -f 'smell_sense/gamma99/' > /dev/null; do sleep 120; done
echo "gamma99 gone $(date -u +%T)"
one decoy 91 &
one nursed 92; wait
one decoy 92 &
one nursed 93; wait
one decoy 93
echo "WEAN_SMELL EXIT"
