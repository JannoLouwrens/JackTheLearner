#!/bin/bash
# Record T2.01 v4 from the COMPLETED kernel once the reverify chain frees the
# ladder lock. Reuse costs zero GPU quota (artifact already exists).
cd /home/opc/jackthelearner
while kill -0 409067 2>/dev/null; do sleep 120; done
export JACK_REUSE_KERNEL=jack-ladder-1786091013
timeout 3600 /data/venvs/jackthelearner/bin/python -m experiments.run T2.01 >> /tmp/t201v4_record.log 2>&1
/data/venvs/jackthelearner/bin/python -m experiments.run status >> /tmp/t201v4_record.log 2>&1
