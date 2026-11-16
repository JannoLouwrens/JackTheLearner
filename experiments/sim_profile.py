"""sim_profile.py — where one decision of his world spends its CPU (ORDERS J6).

Read-only probe: builds one JackEnv (hand cal j0 = alpha = 1.0; the split is
cal-independent), warms 50 random decisions, then times
  (a) the full `JackEnv.step` (process CPU time, N decisions),
  (b) the same model/data stepped by bare `mj_step` x SUBSTEPS — the physics
      floor, i.e. what "all Python moved to native" could at best reach,
  (c) (b) on deep copies of the model under each MuJoCo integrator — the
      world ships integrator 1 (RK4); J6 item (4) is implicitfast (3).
It never touches the world files and writes nothing but stdout.
(b)/(c) step with the last decision's ctrl held and no drive force, so they
are an approximation of the physics share (same contacts at start: 24).

Box, 11-16 14:10, load ~0.8, seed 153, N 500 (two repeats agree to 0.3 ms):
  full step                    27.7 ms  (36.1 decisions/CPU-s)
  mj_step x40, RK4 (shipped)   15.1 ms  (54%) -> zero-Python ceiling 66 /CPU-s
  mj_step x40, Euler            3.9 ms
  mj_step x40, implicit         4.3 ms
  mj_step x40, implicitfast     3.9 ms  (RK4 costs 3.9x)
  Python per decision          12.6 ms  (cProfile: ContactAudio.step ~27% of
                               the decision incl. 24 mj_contactForce calls per
                               substep; drives.substep ~12%; water ~3%;
                               _grounded ~3%)
Reading. §4's "the loop, not physics, is the bottleneck" is half right here:
removing every line of per-substep Python caps at 1.9x (66 /CPU-s), because
RK4 is 54% of the decision. Implicitfast alone with today's Python: ~60
/CPU-s (1.7x). Both together: ~(3.9 + Python left) ms — the only route
toward §4's 500-2,000 band. The integrator is a world change (integrity
.879 -> .972 in one life); the Director rules on it with the stale list.

Usage: $PY -m experiments.sim_profile [N]
"""
from __future__ import annotations

import copy
import sys
import time

import mujoco
import numpy as np

from experiments import w0 as W
from experiments.jack_env import JackEnv


def main(n: int = 500) -> None:
    env = JackEnv(j0=1.0, alpha=1.0)
    env.reset(seed=153)
    rng = np.random.RandomState(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, 8))
    t = time.process_time()
    for _ in range(n):
        env.step(rng.uniform(-1, 1, 8))
    full = (time.process_time() - t) / n
    w = env.w
    print(f"SUBSTEPS {W.SUBSTEPS} timestep {w.model.opt.timestep} "
          f"integrator {w.model.opt.integrator} ncon {w.data.ncon}")
    print(f"full step {full * 1e3:.2f} ms ({1 / full:.1f} decisions/CPU-s)")
    for name, integ in (("Euler", 0), ("RK4", 1), ("implicit", 2), ("implicitfast", 3)):
        m = copy.deepcopy(w.model)
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = w.data.qpos, w.data.qvel, w.data.ctrl
        m.opt.integrator = integ
        mujoco.mj_forward(m, d)
        t = time.process_time()
        for _ in range(n * W.SUBSTEPS):
            mujoco.mj_step(m, d)
        p = (time.process_time() - t) / n
        print(f"  {name:13s} mj_step x{W.SUBSTEPS} {p * 1e3:.2f} ms "
              f"({p / full:.0%} of a decision; zero-Python ceiling {1 / p:.0f}/CPU-s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
