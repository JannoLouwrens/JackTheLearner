"""cores_ring.py — LC.09's `wm-latent+ring` arm: the seated Learning-core
(wm-latent, WorldModelCore decoder=False) with the seated working memory
(RingMemory, WM.02) carried across each life. Built 2026-11-13 (ORDERS J3).

WIRING (declared here because the spec says only "carried across each life in
place of no working memory" and "wires RingMemory ... into the core's input"):

  * The ring READS the fused latent z_t = encode(obs_t) (64 floats) — the
    one representation every sense has reached; it holds 32 floats (16
    planes), RingMemory's seated size.
  * Its readout `head(h_t)` (64 floats) is ADDED to z_t before the RSSM:
    z'_t = z_t + head(h_t). So the ring sits on the core's input and every
    downstream dimension (GRU, post/prior, actor, critic, the guard's roll)
    is byte-for-byte the seated arm's. Nothing is concatenated, so no head
    of the seat is re-shaped.
  * The ring is RingMemory UNCHANGED (constructor, init, rotate=True): this
    file instantiates it and steps it through `ring_cell`, a pure copy of
    `RingMemory.step`'s arithmetic that takes the state as an argument
    (RingMemory.py is in WM.02's IMPL_DEPS, so it is not edited; the
    selftest below asserts `ring_cell` == `RingMemory.step` bit-for-bit).
  * Life: `reset()` at every death puts the ring at its learned resting
    state h0 (a respawned body starts with nothing held).
  * Learning: R2D2 stored state, burn-in 0, exactly as survival_rec does for
    wm-dyn: each replay row keeps the ring state h_{t-1} it was acted from;
    a batch advances one ring step from it (a life's first row is flagged
    at-rest and reads the LIVE h0, so the resting state learns too), so the
    actor/critic/binding
    losses send gradient into the ring's angle, gate, write and head
    (truncated BPTT of length 1 — declared, the stored h was made by older
    weights). Without a stored state (the guard's buffered reads, U-tests)
    the ring starts from h0: the core is read as if each observation began
    a life. The guard's rank and roll therefore read the ring arm with its
    memory at rest — declared, conservative (the reading cannot be inflated
    by the memory).
  * Size: obs 64, hidden 32, out 64 -> 5,329 ring params (the spec's
    "1,031" is the WM.02 fixture's size, obs/out dims of a 40-step register;
    the module is the same, its I/O widths are Jack's latent here).

The guard: this arm has no next-latent head, so the action-gain clause
VOIDs it as unmeasurable, as it VOIDs the seat (unknown is not zero). That is
the registered rule, unmoved.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

from .cores import ACTION_DIM, WorldModelCore

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from RingMemory import RingMemory  # noqa: E402

RING_HIDDEN = 32                    # RingMemory's seated state size (WM.02)


def ring_cell(ring: RingMemory, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """RingMemory.step's state update, pure: (h_{t-1}, x_t) -> h_t."""
    if ring.rotate:
        th = ring.ang(x)
        c, s = th.cos(), th.sin()
        a, b = h[:, :ring.k], h[:, ring.k:]
        h = torch.cat([c * a - s * b, s * a + c * b], dim=-1)
    g = torch.sigmoid(ring.gate(x))
    return (1 - g) * h + g * torch.tanh(ring.write(x))


class RingWorldCore(WorldModelCore):
    binding_term = ("wm-latent's latent prediction + KL, on z + ring readout "
                    "(RingMemory carried across the life)")

    def __init__(self, with_placebo: bool = True):
        super().__init__(with_placebo, decoder=False)
        from .cores import LATENT
        self.ring = RingMemory(LATENT, LATENT, RING_HIDDEN)
        self._stored: Optional[torch.Tensor] = None
        self.reset()

    # -- rollout state -------------------------------------------------------
    def reset(self) -> None:
        self._rh: Optional[torch.Tensor] = None      # None = at rest (h0)

    def efference(self, a) -> None:                  # the ring does not read it
        return None

    def stored_now(self):
        """(ring state the NEXT decision is acted from, at-rest flag). The
        flag marks a life's first decision: replay then reads the LIVE h0, so
        the learned resting state gets gradient (a detached copy would not —
        measured 11-13 22:35: h0 moved 0.0 in 68 living updates)."""
        if self._rh is None:
            return (self.ring.h0.detach().clone(), torch.ones(1))
        return (self._rh.detach().squeeze(0).clone(), torch.zeros(1))

    def use_stored(self, stored) -> None:
        self._stored = stored

    def _h_prev(self, n: int) -> torch.Tensor:
        st = self._stored
        if st is None:
            return self.ring.h0.unsqueeze(0).expand(n, -1)
        if isinstance(st, tuple):                    # (h, at_rest) from replay
            h, rest = st
            return rest * self.ring.h0.unsqueeze(0) + (1 - rest) * h
        return st                                    # a bare state (rollout)

    # -- the core ------------------------------------------------------------
    def _with_ring(self, obs, dropped, h_prev):
        z = WorldModelCore.encode(self, obs, dropped)
        h = ring_cell(self.ring, h_prev, z)
        return z + self.ring.head(h), h

    def encode(self, obs, dropped=()):
        """z' = z + ring readout. Without a stored state the ring is at rest."""
        n = next(iter(obs.values())).shape[0]
        return self._with_ring(obs, dropped, self._h_prev(n))[0]

    @torch.no_grad()
    def step(self, obs, dropped=()) -> Dict[str, torch.Tensor]:
        """One decision of a living rollout (batch 1): advance the ring, keep
        it, return the RSSM state the actor reads."""
        h_prev = (self._rh if self._rh is not None
                  else self.ring.h0.unsqueeze(0))
        old = self._stored
        self._stored = h_prev
        try:
            s = self.rssm(obs, dropped)
            _, self._rh = self._with_ring(obs, dropped, h_prev)
        finally:
            self._stored = old
        return s


# ── selftest: the wiring, no world ──────────────────────────────────────────
def _selftest() -> dict:
    from .cores import (LC_BATCH, lc_targets, lc_update, make_batch,
                        make_optimizer, n_params, build_arm)
    res = {}
    torch.manual_seed(0)
    core = RingWorldCore().eval()
    # (1) ring_cell is RingMemory.step's arithmetic, bit for bit
    x = torch.randn(3, core.ring.obs_dim)
    core.ring.reset(3)
    h0 = core.ring.h.clone()
    core.ring.step(x)
    res["cell_matches_step"] = bool(torch.equal(ring_cell(core.ring, h0, x),
                                                core.ring.h))
    # (2) the seat's dimensions are untouched
    seat = build_arm("wm-latent")
    res["state_dim_same"] = core.state_dim == seat.state_dim
    res["ring_params"] = n_params(core.ring)
    res["params"], res["seat_params"] = n_params(core), n_params(seat)
    # (3) memory carries: two lives see the same obs stream; the second
    #     decision's state depends on the first decision (ring held it)
    gen = torch.Generator().manual_seed(1)
    bb = make_batch(1, n=8)          # n=1 would standardise to NaN
    o1, o2, o1b = ({k: v[i:i + 1] for k, v in bb.items()} for i in (0, 1, 2))
    D = ("language",)
    core.reset(); core.step(o1, D); s_a = core.step(o2, D)["state"]
    core.reset(); core.step(o1b, D); s_b = core.step(o2, D)["state"]
    res["history_moves_state"] = float((s_a - s_b).abs().max())
    core.reset(); s_r = core.step(o2, D)["state"]
    core.reset(); s_r2 = core.step(o2, D)["state"]
    res["reset_is_rest"] = bool(torch.equal(s_r, s_r2))
    # (4) a stored-state update reaches the ring's angle columns
    batch = make_batch(4, n=LC_BATCH)
    opt = make_optimizer(core)
    ang0 = core.ring.ang.weight.detach().clone()
    st = (torch.randn(LC_BATCH, RING_HIDDEN) * 0.5,
          (torch.arange(LC_BATCH) % 2).float().unsqueeze(1))   # half at rest
    h00 = core.ring.h0.detach().clone()
    core.use_stored(st)
    try:
        lc_update(core, batch, lc_targets(gen), opt, dropped=("language",))
    finally:
        core.use_stored(None)
    res["ang_moved"] = float((core.ring.ang.weight.detach() - ang0).abs().max())
    res["h0_moved"] = float((core.ring.h0.detach() - h00).abs().max())
    print(res)
    assert res["cell_matches_step"], "ring_cell drifted from RingMemory.step"
    assert res["state_dim_same"]
    assert res["history_moves_state"] > 0.0, "ring holds nothing"
    assert res["reset_is_rest"]
    assert res["ang_moved"] > 0.0, "no gradient into the ring"
    assert res["h0_moved"] > 0.0, "resting state never learns"
    return res


if __name__ == "__main__":
    _selftest()
