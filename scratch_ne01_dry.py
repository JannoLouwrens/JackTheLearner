"""Abbreviated dry run of NE.01's probes (NOT the registered run)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments.tests.ne_01_nobody_survives_by_accident as t

t.N_RANDOM_LIVES = 2
t.FORAGER_DECISIONS = 7000       # ~1.17 sim-days: day 1 + night 1 + morning
t.N_PUSH_DECISIONS = 300

t0 = time.time()
m = t._experiment(0)
print(f"\n_experiment(0) in {time.time()-t0:.0f}s")
for k in sorted(m):
    if not isinstance(m[k], str):
        print(f"  {k} = {m[k]}")
t0 = time.time()
c = t._control(0)
print(f"\n_control(0) in {time.time()-t0:.0f}s")
for k in sorted(c):
    print(f"  {k} = {c[k]}")
