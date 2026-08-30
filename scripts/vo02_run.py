#!/usr/bin/env python
"""VO.02's registered run. Detached so the verdict lands in the ledger whether
or not the launching session survives (the T2.01 v3 / T2.04 archaeology tax)."""
import os
import sys

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

import experiments.tests.vo_02_two_jacks_signal as M  # noqa: E402

print("VO.02 registered run: 3 seeds x 4 arms, ~0.95 h projected", flush=True)
r = M.run()
print("VERDICT", r.status, r.message, flush=True)
