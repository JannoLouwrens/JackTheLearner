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


def apply_lazy_contact_audio(w) -> None:
    """Swap the instance's `synth.step` for the lazy one (this W0 only)."""
    w.synth.step = types.MethodType(_lazy_step, w.synth)
