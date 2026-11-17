"""substep_split.py — J6 item (2b): what "all 40 sub-steps of a decision in one
native call" can buy, measured before anyone writes it (ORDERS J6).

Read-only probe. Builds one JackEnv with the SHIPPED cal and
`lazy_contact_audio=True` (ruling 34: every new run passes it), warms 50
random decisions, then times N decisions with every per-sub-step piece of
`W0.decide`'s loop wrapped by a perf_counter timer on the instance only
(`_grounded`, `water.apply`, `mujoco.mj_step`, `synth.step`,
`drives.substep`). Nothing in the world files is touched; the wrappers call
the originals with the same arguments, so the trajectory is the shipped one.
Then, on a deep copy of the model/data, it times 40 Python-level `mj_step`
calls against ONE `mj_step(m, d, nstep=40)` — the native-call saving itself.
Run once under each integrator ("shipped" RK4 and "implicitfast").

Reading rule (written before the first run): (2b) is worth building only if
its CEILING — the decision with every per-sub-step Python piece and the
Python loop removed, i.e. full − (Σ pieces − mj_step) — is ≥ 1.3× the
lazy-audio decision. Identity: `_grounded`, `water.apply`, `synth.step` and
`drives.substep` each read the state the PREVIOUS sub-step left, so one
`mj_step(nstep=40)` is identical only if all four move into native code (a
C extension or `mjcb_control`, which calls back into Python every sub-step
and saves nothing). Moving them to once a decision is a world change.

Box, 11-17 09:24, load 0.4, seed 153, N 300, nice 19 (ms per decision, wall):
                       shipped RK4        implicitfast
  full decision        21.45 (46.8/CPU-s) 10.44 (96.4/CPU-s)
  mj_step x40          14.99  70%          4.66  45%
  drives.substep        2.85  13%          2.66  25%
  water.apply           1.25   6%          0.84   8%
  synth.step (lazy)     0.91   4%          0.88   8%
  _grounded             0.73   3%          0.71   7%
  per-decision rest     0.72   3%          0.68   7%
  40 py mj_step calls vs one mj_step(nstep=40): 13.70 vs 13.66 / 3.88 vs 3.83
  (2b) CEILING         15.67 -> 1.37x      5.30 -> 1.97x
Reading. The native CALL itself saves 0.04 ms (0.2%): `mj_step(nstep=40)`
alone is not a lever. Every gain is in the four per-sub-step Python pieces,
~5.7 ms a decision (drives.substep ~70 us a sub-step, small-numpy overhead in
an already vectorised body). Both ceilings clear the 1.3x rule, so (2b) IS
worth building — but only as those four pieces in C, identity-tested against
today's Python; under implicitfast it is the larger half of the decision.
Buildable here: gcc, cffi 2.0.0, mujoco 3.2.3's `include/` and
`libmujoco.so.3.2.3` ship in the venv's site-packages.

Usage: $PY -m experiments.substep_split [N]
"""
from __future__ import annotations

import copy
import sys
import time
from collections import defaultdict

import mujoco
import numpy as np

from experiments import w0 as W
from experiments.jack_env import JackEnv


class _Timed:
    def __init__(self, fn, acc, key):
        self.fn, self.acc, self.key = fn, acc, key

    def __call__(self, *a, **k):
        t = time.perf_counter()
        r = self.fn(*a, **k)
        self.acc[self.key] += time.perf_counter() - t
        return r


class _MjProxy:
    def __init__(self, mod, acc):
        self._mod, self._acc = mod, acc
        self.mj_step = _Timed(mod.mj_step, acc, "mj_step")

    def __getattr__(self, name):
        return getattr(self._mod, name)


def run(integrator: str, n: int) -> dict:
    env = JackEnv(lazy_contact_audio=True, integrator=integrator)
    env.reset(seed=153)
    rng = np.random.RandomState(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, 8))
    w = env.w
    acc: dict = defaultdict(float)
    w._grounded = _Timed(w._grounded, acc, "_grounded")
    if w.water is not None:
        w.water.apply = _Timed(w.water.apply, acc, "water.apply")
    w.synth.step = _Timed(w.synth.step, acc, "synth.step")
    w.drives.substep = _Timed(w.drives.substep, acc, "drives.substep")
    w.mujoco = _MjProxy(w.mujoco, acc)
    tp, tw = time.process_time(), time.perf_counter()
    for _ in range(n):
        env.step(rng.uniform(-1, 1, 8))
    full_cpu = (time.process_time() - tp) / n
    full_wall = (time.perf_counter() - tw) / n
    per = {k: v / n for k, v in acc.items()}

    m = copy.deepcopy(w.model)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = w.data.qpos, w.data.qvel, w.data.ctrl
    mujoco.mj_forward(m, d)
    d0 = copy.deepcopy(d)
    t = time.perf_counter()
    for _ in range(n):
        for _ in range(W.SUBSTEPS):
            mujoco.mj_step(m, d)
    loop40 = (time.perf_counter() - t) / n
    d = d0
    t = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(m, d, nstep=W.SUBSTEPS)
    native40 = (time.perf_counter() - t) / n
    return dict(integrator=integrator, full_cpu=full_cpu, full_wall=full_wall,
                per=per, loop40=loop40, native40=native40)


def report(r: dict) -> None:
    ms = 1e3
    print(f"== integrator {r['integrator']}, lazy_contact_audio on ==")
    print(f"full decision  {r['full_cpu'] * ms:6.2f} ms CPU ({1 / r['full_cpu']:.1f} dec/CPU-s), "
          f"{r['full_wall'] * ms:.2f} ms wall")
    py_sub = 0.0
    for k, v in sorted(r["per"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:15s} {v * ms:6.2f} ms ({v / r['full_wall']:.0%})")
        if k != "mj_step":
            py_sub += v
    rest = r["full_wall"] - sum(r["per"].values())
    print(f"  per-decision + loop/timer rest {rest * ms:6.2f} ms ({rest / r['full_wall']:.0%})")
    print(f"  bare mj_step x40 python loop {r['loop40'] * ms:.2f} ms; "
          f"one mj_step(nstep=40) {r['native40'] * ms:.2f} ms "
          f"(native-call saving {(r['loop40'] - r['native40']) * ms:.3f} ms)")
    ceiling = r["full_wall"] - py_sub - max(0.0, r["loop40"] - r["native40"])
    print(f"  (2b) CEILING {ceiling * ms:.2f} ms -> {r['full_wall'] / ceiling:.2f}x "
          f"(rule: build only if >= 1.3x)")


def main(n: int = 300) -> None:
    for integ in ("shipped", "implicitfast"):
        report(run(integ, n))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
