"""fast_sim.py — J6(2): per-sub-step Python moved out of the decision where the
result is identical. Each piece is a `jack_env` option, default OFF; nothing
here edits `ContactAudio.py` or `w0.py` (so no standing certificate stales).

(a) `lazy_contact_audio` — `ContactAudioSynth.step` with the work it throws
    away removed. The shipped step, every one of the 40 sub-steps of a
    decision, computes `mj_contactForce` and copies `con.pos` for EVERY
    contact (~24 at rest) and only then discards every pair that is already
    touching or inside its refractory window — which is nearly all of them.
    Here pairs are read from MuJoCo's contact arrays in one call, and force and
    position are computed only for a pair that will actually fire, over the
    same contact indices in the same order, so the event's centroid and summed
    force are the same floating-point operations on the same numbers.
    `_prev_pairs` and `_last_fired` evolve exactly as before. Proof:
    `experiments/tests/test_fast_sim.py` (stream identity incl. every audio
    event field, and a control that must differ).

(b) `multiray_vision` — J6(3): `W0._vision`'s 16 `mj_ray` calls replaced by
    ONE `mj_multiRay` over the same origin, directions, geom groups,
    flg_static=1 and bodyexclude=-1, cutoff mjMAXVAL (so the
    "> RAY_MAX_RANGE reads empty" clause stays where it was). The panel's
    noise draw happens per ray in the same order k = 0..15, so the world's
    random stream is consumed identically. MuJoCo 3.2.3 returns the same
    distance and geom id bit for bit (2,400 rays, 0 differences, 11-16).
    Proof: `experiments/tests/test_multiray.py`. Speed: today's eye is
    0.057 ms of a ~20 ms decision, so this is NOT a speed lever for the
    16-ray eye (<0.3%); its value is the batched call §12's retina needs
    (16x16 0.10 ms, 32x32 0.34 ms, 5-7x the loop, §12).

Speed table: in `experiments/tests/test_fast_sim.py`'s docstring.
"""
from __future__ import annotations

import types

import numpy as np

import experiments.w0  # noqa: F401  (puts ContactAudio on the path)
from ContactAudio import REFRACTORY_S


def _lazy_step(self, data) -> None:
    import mujoco
    t = float(data.time)
    n = int(data.ncon)
    by_pair: dict = {}
    if n:
        g = data.contact.geom[:n]
        for i, (a, b) in enumerate(g.tolist()):
            pair = (a, b) if a <= b else (b, a)
            lst = by_pair.get(pair)
            if lst is None:
                by_pair[pair] = [i]
            else:
                lst.append(i)
    prev, last = self._prev_pairs, self._last_fired
    for pair, idx in by_pair.items():
        if pair in prev:
            continue
        if t - last.get(pair, -1e9) < REFRACTORY_S:
            continue
        pts = []
        for i in idx:
            con = data.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, f6)
            pts.append((np.array(con.pos), abs(float(f6[0]))))
        pos = np.mean([p for p, _ in pts], axis=0)
        force = float(sum(f for _, f in pts))
        last[pair] = t
        self.events.append(self._make_event(t, pair, pos, force))
    self._prev_pairs = set(by_pair)


def _multiray_vision(self) -> np.ndarray:
    import mujoco
    from experiments import w0 as W
    pos = np.array(self.data.xpos[self.rover_bid], dtype=float)
    pos[2] += W.RAY_Z_OFFSET
    n = W.N_RAYS
    if not hasattr(self, "_mr_vec"):
        self._mr_vec = np.ascontiguousarray(self._ray_dirs, dtype=np.float64).reshape(-1)
        self._mr_gid = np.zeros(n, dtype=np.int32)
        self._mr_dist = np.zeros(n, dtype=np.float64)
    mujoco.mj_multiRay(self.model, self.data, pos, self._mr_vec, W._RAY_GROUPS,
                       1, -1, self._mr_gid, self._mr_dist, n, mujoco.mjMAXVAL)
    out = np.empty(W.MODALITIES["vision"], dtype=np.float32)
    for k in range(n):
        dist = float(self._mr_dist[k])
        gid = int(self._mr_gid[k])
        if dist < 0 or dist > W.RAY_MAX_RANGE or gid < 0:
            d, t = 1.0, 0.0
        else:
            d = dist / W.RAY_MAX_RANGE
            t = float(self._tex[gid])
            if gid == self.panel_gid:
                amp = min(1.0, 2.5 / max(dist, 1e-6))
                t = 0.5 + amp * (float(self._rng.uniform()) - 0.5)
        out[2 * k] = d
        out[2 * k + 1] = t
    # keep the shipped side effect: _geomid holds the LAST ray's hit
    self._geomid[0] = self._mr_gid[n - 1]
    return out


def apply_multiray_vision(w) -> None:
    """Swap the instance's `_vision` for the one-call version (this W0 only)."""
    w._vision = types.MethodType(_multiray_vision, w)


def apply_lazy_contact_audio(w) -> None:
    """Swap the instance's `synth.step` for the lazy one (this W0 only)."""
    w.synth.step = types.MethodType(_lazy_step, w.synth)
