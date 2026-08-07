#!/bin/bash
# T2.01 v4 (gpu<8h on Kaggle), then 3-seed T1 re-verification.
# Detached from any Claude session; survives restarts. Serialised by the
# ladder lock inside experiments.run itself — this script just orders the work.
cd /home/opc/jackthelearner
PY=/data/venvs/jackthelearner/bin/python
{
  echo "=== T2.01 v4 launch $(date -u +%FT%TZ) ==="
  timeout 39600 $PY -m experiments.run T2.01
  echo "=== reverify start $(date -u +%FT%TZ) ==="
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  for spec in T1.05 T1.04 T1.03 T1.12 T1.01 T1.02 T1.06; do
    nice -n 19 timeout 7000 $PY -m experiments.run "$spec"
  done
  $PY -m experiments.run status
  echo "=== all done $(date -u +%FT%TZ) ==="
} >> /tmp/v4_chain.log 2>&1
