"""survival_rec.py — the lethal-W0 learner loop for a core that CARRIES STATE
(ORDERS J3, LC.09's wm-dyn arm; 2026-11-13).

Why a new file and not an edit of `survival.py`: 10 certificates declare
`survival.py` in IMPL_DEPS (LC.03/07/08, LF.03, SH.01/02, W0.diag, W1.00/02/04);
touching it stales all ten before the new arm has won anything. This loop
reuses survival.py's ring, GAE and update schedule by import and differs only
where a recurrent core must:

  * ROLLOUT: `core.step(obs)` advances deter from the previous decision, and
    `core.efference(a)` hands it the motor command actually issued (noise
    included). `core.reset()` at every death: a respawned body starts from
    zero state.
  * REPLAY (R2D2 stored state, Kapturowski et al. 2019, burn-in 0): every
    ring row keeps the (deter_{t-1}, z_{t-1}, a_{t-1}) it was acted from; a
    sampled batch hands them to `core.use_stored` before `lc_update`, so the
    actor/critic read the state the rollout read, and the binding loss's
    L_next sends gradient into the GRU's action columns.

ARMS (11-13 22:30): wm-dyn, and wm-latent+ring (cores_ring.RingWorldCore,
the seat with RingMemory carried across the life; its stored state is the ring
state alone, read via `core.stored_now()`; it ignores `efference`).

Scope: policy="core" only. The nulls (random, random-repeat, statue) carry no
state and run through `survival.run_survival` unchanged. No lp channel, no
record_xy/record_transitions (LC.09 does not read them). Returns the facts
LC.09 reads under the same keys as `run_survival`.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from . import drives
from .cores import (ACTION_DIM, DETER, LATENT, LC_BATCH, build_arm,
                    lc_update, make_optimizer, n_params)
from .cores_dyn import DynWorldCore
from .survival import (D_DEATH, EXPLORE_STD, SEG, _TargetRing, _gae,
                       _updates_due)
from .w0 import W0

REC_ARMS = ("wm-dyn", "wm-latent+ring")


def build_rec_arm(name: str):
    if name == "wm-dyn":
        return DynWorldCore()
    if name == "wm-latent+ring":
        from .cores_ring import RingWorldCore
        return RingWorldCore()
    raise ValueError(f"{name!r} is not a recurrent arm {REC_ARMS}")


class _StateRing(_TargetRing):
    """_TargetRing + the stored recurrent state each row was acted from.
    `stored` is a tuple of 1-d tensors (wm-dyn: deter, z, a; wm-latent+ring:
    the ring state alone); widths are taken from the first push."""

    def __init__(self, obs, cap=None):
        super().__init__(obs) if cap is None else super().__init__(obs, cap)
        self.st: Optional[List[torch.Tensor]] = None

    def push_rec(self, obs, action, value_target, advantage, stored):
        stored = stored if isinstance(stored, tuple) else (stored,)
        if self.st is None:
            self.st = [torch.zeros(self.cap, x.shape[-1]) for x in stored]
            self._tuple = len(stored) > 1
        i = self.n % self.cap
        for buf, x in zip(self.st, stored):
            buf[i] = x
        self.push(obs, action, value_target, advantage)

    def sample_rec(self, gen, size: int = LC_BATCH):
        hi = min(self.n, self.cap)
        idx = torch.randint(0, hi, (size,), generator=gen)
        batch = {k: self.obs[k][idx] for k in self.keys}
        targets = {"value": self.value[idx], "value_lp": self.value_lp[idx],
                   "action": self.action[idx], "advantage": self.advantage[idx]}
        st = tuple(b[idx] for b in self.st)
        return batch, targets, (st if self._tuple else st[0])


def run_survival_rec(seed: int, *, j0: float, alpha: float, n_decisions: int,
                     arm: str = "wm-dyn", train: bool = True,
                     train_ratio: float = 0.25, e0: float = 1.0,
                     explore_std: tuple = EXPLORE_STD,
                     reward_fn=None,
                     return_core: bool = False) -> dict:
    """`reward_fn(r_h, w, obs, core) -> float`: survival.run_survival's hook,
    same call point (after `w.decide`, before the segment push), so LC.09's
    GuardRecorder rides this loop unchanged. None = homeo-dr unchanged."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    if e0 != 1.0:
        w.drives.state = drives.DriveState(e=e0)
    rng = np.random.RandomState(seed * 6553 + 11)       # survival.py's seeds
    gen = torch.Generator().manual_seed(seed * 7919 + 3)
    torch.manual_seed(9_000 + seed)
    core = build_rec_arm(arm)
    core.eval()
    opt = make_optimizer(core) if train else None
    ring = _StateRing(w.observe())

    seg: List[tuple] = []          # (obs, a, r, v, stored)
    optimiser_steps = 0
    reward_sum = 0.0
    act_log: List[np.ndarray] = []
    life_ends: List[tuple] = []
    t_cpu0, t_wall0 = time.process_time(), time.perf_counter()

    def _close(boot: float, terminal: bool):
        nonlocal seg
        if not seg:
            return
        adv, tgt = _gae([s[2] for s in seg], [s[3] for s in seg], boot, terminal)
        for i, (o, a, _r, _v, st) in enumerate(seg):
            ring.push_rec(o, a, float(tgt[i]), float(adv[i]), st)
        seg = []

    for k in range(n_decisions):
        obs = w.observe()
        obs_t = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in obs.items()}
        # the state this decision is acted FROM, as step() will read it
        if hasattr(core, "stored_now"):
            stored = core.stored_now()
        else:
            stored = ((core._h if core._h is not None
                       else torch.zeros(1, DETER)).squeeze(0).clone(),
                      (core._z if core._z is not None
                       else torch.zeros(1, LATENT)).squeeze(0).clone(),
                      core._a.squeeze(0).clone())
        with torch.no_grad():
            s = core.step(obs_t, dropped=W0.DROPPED)
            mean = core.act(obs_t, s["state"]).squeeze(0).numpy()
            v = float(core.critic(s["state"]))
        frac = min(1.0, k / max(1, n_decisions - 1))
        std = explore_std[0] + (explore_std[1] - explore_std[0]) * frac
        a = np.clip(mean + std * rng.randn(ACTION_DIM), -1.0, 1.0)
        core.efference(a)

        d_before = w.drives.state.d()
        w.decide(a)
        died = w.died_this_decision
        r = d_before - (D_DEATH if died else w.drives.state.d())
        if reward_fn is not None:
            r = float(reward_fn(r, w, obs, core))
        reward_sum += r
        act_log.append(a)
        seg.append(({kk: torch.from_numpy(vv.copy()) for kk, vv in obs.items()},
                    torch.from_numpy(a.astype(np.float32)), r, v, stored))
        if died or len(seg) >= SEG:
            _close(boot=v, terminal=died)
        if train and ring.n >= LC_BATCH:
            for _ in range(_updates_due(w.decisions, train_ratio)):
                batch, targets, st = ring.sample_rec(gen)
                core.use_stored(st)
                try:
                    lc_update(core, batch, targets, opt, dropped=W0.DROPPED)
                finally:
                    core.use_stored(None)
                optimiser_steps += 1
        if died:
            life_ends.append((int(w.decisions),
                              round(time.process_time() - t_cpu0, 3),
                              int(optimiser_steps)))
            core.reset()
            if e0 != 1.0:
                w.drives.state = drives.DriveState(e=e0)

    spans = list(w.life_lengths)
    third = len(spans) // 3
    acts = np.asarray(act_log)
    out = {
        "life_gain": (float(np.mean(spans[-third:]) - np.mean(spans[:third]))
                      if third >= 1 else 0.0),
        "n_lives": float(len(spans)),
        "mean_life_s": float(np.mean(spans)) if spans else 0.0,
        "life_spans": [round(float(x), 3) for x in spans],
        "reward_sum": float(reward_sum),
        "action_std_final_third": (float(acts[-(len(acts) // 3):].std())
                                   if len(acts) >= 3 else 0.0),
        "decisions": float(w.decisions),
        "sim_seconds": float(w.sim_seconds),
        "optimiser_steps": float(optimiser_steps),
        "process_time_s": float(time.process_time() - t_cpu0),
        "wall_s": float(time.perf_counter() - t_wall0),
        "life_ends": life_ends,
        "params": float(n_params(core)),
    }
    if return_core:
        out["core"] = core
    return out


def _smoke(n: int = 400) -> dict:
    """Wiring check (records nothing): a short trained life must take
    optimiser steps and MOVE the GRU's action columns (the seated stateless
    path gives them exactly 0.0 gradient, ef335cf); the untrained twin must
    leave them bit-identical (control).

    WORLD: j0 = alpha = 1.0 is NOT the calibrated W0 (PS.01: j0 2.255 ms,
    alpha .0275). Measured 11-13 21:40 (seed 0, 400 dec): every death here is
    an INTEGRITY death after 0.2-1.0 sim-s, so `n_lives` counts how hard the
    body thrashes, not how well he lives - trained 120 vs frozen 32 lives, and
    the stateless loop shows the same direction (wm-latent 55 vs 8; random
    96). Under PS.01's calibration all five read 0 deaths in 400 dec (seeds
    0, 1). Read only the wiring asserts from this smoke, never its lives."""
    torch.manual_seed(9_000)
    w0 = DynWorldCore().gru.weight_ih[:, LATENT:].detach().clone()
    res = {}
    for train in (True, False):
        out = run_survival_rec(0, j0=1.0, alpha=1.0, n_decisions=n,
                               train=train, return_core=True)
        wa = out.pop("core").gru.weight_ih[:, LATENT:].detach()
        res[train] = {k: out[k] for k in ("decisions", "n_lives",
                                          "optimiser_steps", "process_time_s")}
        res[train]["action_cols_moved"] = float((wa - w0).abs().max())
    print(res)
    assert res[True]["optimiser_steps"] > 0
    assert res[True]["action_cols_moved"] > 0.0, "no gradient to action cols"
    assert res[False]["action_cols_moved"] == 0.0, "frozen twin moved"
    return res


if __name__ == "__main__":
    _smoke()
