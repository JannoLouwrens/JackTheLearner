"""native_substep.py — J6 item (2b), piece 1 of 4: the per-sub-step contact
scans in C, read straight from MuJoCo's own `mjData*` (ORDERS J6).

`experiments/substep_split.py` priced (2b): the native CALL saves nothing;
every gain is in four per-sub-step Python pieces (`drives.substep`,
`water.apply`, `synth.step`, `W0._grounded`). Identity forbids moving any of
them to once a decision, so they move into C one at a time, each proven
identical against today's Python on live trajectories BEFORE any `jack_env`
option wires them in (a `jack_env` edit stales its certificates — one edit,
when all four are proven, not four).

This file is that pipeline plus its first two functions, both integer-exact:
  jn_grounded(d, body_mask, ground_mask)  == W0._grounded()
  jn_partners(d, mask, out)               == drives._contact_partners(d, mask)
                                             (as a per-geom 0/1 row)
C reads `d->contact[k].geom[0..1]` through `MjData._address` — no copy, no
numpy view. Built with gcc + cffi ABI mode (`dlopen`) against the venv's own mujoco 3.2.3
`include/` and `libmujoco.so.3.2.3`, output under /data/caches/jack_native
(never /home or /tmp: `/` is near its floor).

Not world code; nothing here is imported by the world. Proof:
`tests/test_native_substep.py` (identity on live JackEnv sub-steps, 3 seeds,
plus a control that must differ).

Scar for pieces 2-4 (written before they exist): `drives.substep`'s power is
`np.abs(tau*omega).sum()` — numpy's PAIRWISE sum. A C loop that sums left to
right differs in the last bits and is NOT identical; the C side must
reproduce numpy's pairwise order (blocks of 8, unroll 128) or the option is a
world change. Same for every `np.linalg.norm` (it is sqrt(dot), BLAS order).

Measured 11-17 09:50, box, nice 19 (seed 153, ncon 23): `_grounded` Python
13.69 us, C 1.02 us a call -> x40 = 0.51 ms of a decision (of 0.73 priced).
Identity (`tests/test_native_substep.py`): 18,000/18,000 sub-step calls of
each function match on seeds 153-155; the control (ground mask cleared)
differs on all 17,456 grounded sub-steps.

Usage: $PY -m experiments.native_substep   # build + timing on one live env
"""
from __future__ import annotations

import os
import sys
import time

import mujoco
import numpy as np

BUILD_DIR = "/data/caches/jack_native"
_MJ_DIR = os.path.dirname(mujoco.__file__)

CDEF = """
int jn_grounded(uintptr_t d, const uint8_t* body, const uint8_t* ground);
int jn_partners(uintptr_t d, const uint8_t* mask, uint8_t* out, int ngeom);
"""

SRC = r"""
#include <stdint.h>
#include <string.h>
#include <mujoco/mujoco.h>

int jn_grounded(uintptr_t dp, const uint8_t* body, const uint8_t* ground) {
    const mjData* d = (const mjData*)dp;
    for (int k = 0; k < d->ncon; k++) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        if ((body[a] && ground[b]) || (body[b] && ground[a])) return 1;
    }
    return 0;
}

int jn_partners(uintptr_t dp, const uint8_t* mask, uint8_t* out, int ngeom) {
    const mjData* d = (const mjData*)dp;
    int n = 0;
    memset(out, 0, (size_t)ngeom);
    for (int k = 0; k < d->ncon; k++) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        if (mask[a] && !out[b]) { out[b] = 1; n++; }
        if (mask[b] && !out[a]) { out[a] = 1; n++; }
    }
    return n;
}
"""


def build(force: bool = False):
    """Compile SRC once into BUILD_DIR/libjack_native.so; return (ffi, lib).

    ABI mode (gcc + `ffi.dlopen`): the box has no Python headers
    (`pyconfig.h` missing), so cffi's API mode cannot compile here."""
    import subprocess
    from cffi import FFI
    os.makedirs(BUILD_DIR, exist_ok=True)
    so = os.path.join(BUILD_DIR, "libjack_native.so")
    c = os.path.join(BUILD_DIR, "jack_native.c")
    if force or not os.path.exists(so) or open(c).read() != SRC:
        with open(c, "w") as f:
            f.write(SRC)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, c,
                        "-I", os.path.join(_MJ_DIR, "include")], check=True)
    ffi = FFI()
    ffi.cdef(CDEF)
    return ffi, ffi.dlopen(so)


class NativeContacts:
    """C twins of `W0._grounded` and `drives._contact_partners` for one env."""

    def __init__(self, body_mask, ground_mask):
        self.ffi, self.lib = build()
        self._body = np.ascontiguousarray(body_mask, dtype=np.uint8)
        self._ground = np.ascontiguousarray(ground_mask, dtype=np.uint8)
        self._pb = self.ffi.from_buffer("uint8_t[]", self._body)
        self._pg = self.ffi.from_buffer("uint8_t[]", self._ground)
        self._out = np.zeros(len(self._body), dtype=np.uint8)
        self._po = self.ffi.from_buffer("uint8_t[]", self._out)

    def grounded(self, data) -> bool:
        return bool(self.lib.jn_grounded(data._address, self._pb, self._pg))

    def partners_row(self, data, mask_u8_ptr) -> np.ndarray:
        self.lib.jn_partners(data._address, mask_u8_ptr, self._po,
                             len(self._out))
        return self._out


def _main() -> None:
    from experiments.jack_env import JackEnv
    env = JackEnv(seed=153, lazy_contact_audio=True)
    env.reset(seed=153)
    w = env.w
    nc = NativeContacts(w._body_mask, w._ground_mask)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, env.action_space.shape))
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        w._grounded()
    tp = (time.perf_counter() - t0) / N * 1e6
    t0 = time.perf_counter()
    for _ in range(N):
        nc.grounded(w.data)
    tc = (time.perf_counter() - t0) / N * 1e6
    print(f"ncon={w.data.ncon} _grounded python {tp:.2f} us, C {tc:.2f} us"
          f" -> x40 saves {(tp - tc) * 40 / 1000:.3f} ms a decision")


if __name__ == "__main__":
    _main()
