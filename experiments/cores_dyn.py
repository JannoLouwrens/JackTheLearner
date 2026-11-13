"""cores_dyn.py — a world-model core that HAS dynamics (ORDERS J3, 2026-11-13).

Why this file exists. ef335cf measured that `cores.WorldModelCore.rssm()` steps
its GRU once from a zero state with a zero action, so the action columns and
`weight_hh` of dreamer-xs / wm-efe / wm-latent get exactly 0.0 gradient: the
seated Learning-core is a feed-forward encoder called a world model. This is
the repair as a NEW ARM to be raced (LC.09 re-registration = one more arm,
every gate unmoved). It lives in its own file on purpose: `cores.py` is
hashed raw by 8 certificates (LC.01/02/03/07/08, LF.03, SH.01/02); a new arm
must not stale them before it has won anything.

`wm-dyn` — the arm, declared:
  * RSSM convention: deter_t = GRU([z_{t-1}, a_{t-1}], deter_{t-1});
    post_t = q(deter_t, z_t); prior_t = p(deter_t). State = [deter | post],
    the same width and layout as WorldModelCore (the actor/critic, the LC.09
    guard's `roll` and `rssm()["state"]` all read it unchanged).
  * ROLLOUT carries deter across decisions (`step`), the executed action
    arrives as an EFFERENCE COPY (`efference(a)` — the motor command actually
    issued, noise included, as a body knows its own command), and `reset()`
    at death (a respawned body starts from zero state).
  * BINDING / LEARNING: on windows of real consecutive experience
    (obs_t, a_t) of length SEQ, filtered from zero state:
      L_next = || pred(deter_t) - sg(z_t) ||^2  — the latent at t predicted
               from (past state, previous ACTION) before seeing obs_t;
      L_kl   = KL(post_t || prior_t) with free bits (as WorldModelCore);
      L_var  = VICReg std hinge on z (Bardes et al. 2022, arXiv:2105.04906),
               mean(relu(1 - std_d(z))) — the stop-gradient target alone lets
               the encoder collapse to a constant (that is what LC.09's rank
               clause exists to catch; this term is the arm's declared
               defence, not a guard bypass: the guard still reads rank).
    Gradient therefore reaches the GRU's action columns and weight_hh by
    construction; `_probe` measures that it does.
  * The core keeps its own window buffer (SEQ_BUF windows) of what it lived.
    Not wired into `survival.py` yet: survival's ring samples i.i.d. single
    observations, so the actor/critic batch carries no recurrent state. The
    planned wiring is R2D2's stored-state strategy (Kapturowski et al. 2019):
    the deter at each decision is kept with the observation. NEXT STEP.

Probe (records nothing): `python -m experiments.cores_dyn probe`.

MEASURED 2026-11-13 18:50 (probe, W0 seed 0, 1200 random-policy decisions,
1 life, 884/295 train/test windows, init 9000, 300 Adam updates, 69 s CPU):
  * gradient REACHES the dynamics: first update GRU action cols max |g|
    0.0565, weight_hh 0.1432 (seated arms: exactly 0.0). Structural repair OK.
  * Delta_5 on held-out windows, FILTERED start state: before 0.0147, after
    0.0093 — training did NOT make his actions matter more to the roll.
    L_next 0.74 -> 0.17 (u50) then back to 0.55 as the VICReg hinge expanded
    z (var 0.58 -> 0.0): unnormalised MSE on a widening target. Not a pass.
  * the SEATED wm-latent, untrained, through the same filtered measurement:
    +0.0147 (same init -> same number). ef335cf's -0.006..-0.008 came from a
    ZERO start state; from a real posterior an untrained random GRU reads
    POSITIVE. So Delta_5 > 0 does not certify learned dynamics and the
    "predictable VOID" of the seat is not structural — LC.09's guard needs a
    comparison to the arm's own untrained/shuffled-action roll (INBOX).
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .cores import (ACTION_DIM, DETER, LATENT, STOCH_C, STOCH_N, UNIMIX,
                    FREE_BITS, LC_LR, _mlp, WorldModelCore)

SEQ = 16            # window length for the dynamics loss
SEQ_BUF = 64        # windows kept
VAR_COEF = 1.0      # VICReg std-hinge weight (VICReg's own default is 25 on
                    # a 2048-d head; 1.0 here, declared, untuned)


class DynWorldCore(WorldModelCore):
    binding_term = ("next-latent prediction from (deter_{t-1}, a_{t-1}) "
                    "+ posterior/prior KL + VICReg std hinge")

    def __init__(self, with_placebo: bool = True):
        super().__init__(with_placebo, decoder=False)
        self.next_pred = _mlp(DETER, 256, LATENT)
        self.reset()

    # -- rollout state -------------------------------------------------------
    def reset(self):
        self._h: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._a = torch.zeros(1, ACTION_DIM)

    def efference(self, a) -> None:
        self._a = torch.as_tensor(np.asarray(a, dtype=np.float32)).view(1, -1)

    def _cell(self, z_prev, a_prev, h_prev, z):
        deter = self.gru(torch.cat([z_prev, a_prev], -1), h_prev)
        post = self._probs(self.post(torch.cat([deter, z], -1)))
        prior = self._probs(self.prior(deter))
        n = z.shape[0]
        return {"z": z, "deter": deter, "post": post, "prior": prior,
                "state": torch.cat([deter, post.reshape(n, -1)], -1)}

    @torch.no_grad()
    def step(self, obs, dropped=()) -> Dict[str, torch.Tensor]:
        """One decision of a living rollout (batch 1): advance and keep."""
        z = self.encode(obs, dropped)
        h = self._h if self._h is not None else torch.zeros(1, DETER)
        zp = self._z if self._z is not None else torch.zeros_like(z)
        s = self._cell(zp, self._a, h, z)
        self._h, self._z = s["deter"], z
        return s

    def rssm(self, obs, dropped=(), h=None, z_prev=None, a_prev=None):
        """Stateless call (batch): zero state unless one is supplied. The
        stored-state wiring will supply it; without it this degrades to the
        seated arm's behaviour, which is why the probe filters sequences."""
        z = self.encode(obs, dropped)
        n = z.shape[0]
        h = torch.zeros(n, DETER) if h is None else h
        zp = torch.zeros_like(z) if z_prev is None else z_prev
        a = torch.zeros(n, ACTION_DIM) if a_prev is None else a_prev
        return self._cell(zp, a, h, z)

    # -- learning on real sequences ------------------------------------------
    def filter_seq(self, obs_seq: List[Dict[str, torch.Tensor]],
                   acts: torch.Tensor, dropped=()) -> List[Dict]:
        """obs_seq: T dicts of (B, d); acts: (T, B, A), a_t executed after obs_t."""
        B = acts.shape[1]
        h = torch.zeros(B, DETER)
        z_prev = None
        a_prev = torch.zeros(B, ACTION_DIM)
        out = []
        for t, o in enumerate(obs_seq):
            z = self.encode(o, dropped)
            zp = torch.zeros_like(z) if z_prev is None else z_prev
            s = self._cell(zp, a_prev, h, z)
            out.append(s)
            h, z_prev, a_prev = s["deter"], z, acts[t]
        return out

    def seq_loss(self, obs_seq, acts, dropped=()) -> Tuple[torch.Tensor, Dict]:
        ss = self.filter_seq(obs_seq, acts, dropped)
        l_next = torch.stack([F.mse_loss(self.next_pred(s["deter"]),
                                         s["z"].detach()) for s in ss[1:]]).mean()
        l_kl = torch.stack([self._kl(s["post"], s["prior"]) for s in ss]).mean()
        z = torch.cat([s["z"] for s in ss], 0)
        l_var = F.relu(1.0 - z.std(0)).mean()
        total = l_next + l_kl + VAR_COEF * l_var
        return total, {"next": float(l_next.detach()), "kl": float(l_kl.detach()),
                       "var": float(l_var.detach())}


# ── probe: does gradient reach the dynamics, and does the guard see it? ─────
def _windows(hist, idx, T):
    obs = [{k: torch.from_numpy(np.stack([hist[i + t][0][k] for i in idx]))
            for k in hist[0][0]} for t in range(T)]
    acts = torch.from_numpy(np.stack(
        [np.stack([hist[i + t][1] for i in idx]) for t in range(T)]))
    return obs, acts


@torch.no_grad()
def delta5_seq(core, hist, starts, K=5, dropped=("language",)) -> float:
    """LC.09's declared Delta_5, with each window FILTERED from its own past
    (W = SEQ decisions of burn-in) instead of a zero state: the roll starts
    from the real posterior at t, the target is the filtered state at t+K."""
    from .tests.lc_09_learning_core_3x import _cos
    T = SEQ + K
    obs, acts = _windows(hist, [i - SEQ for i in starts], T)
    ss = core.filter_seq(obs, acts, dropped)
    s_t, target = ss[SEQ - 1], ss[SEQ - 1 + K]["state"]
    roll_a = acts[SEQ - 1:SEQ - 1 + K]          # a_t .. a_{t+K-1}

    def roll(a_seq):
        deter, z = s_t["deter"], s_t["z"]
        for k in range(K):
            deter = core.gru(torch.cat([z, a_seq[k]], -1), deter)
        prior = core._probs(core.prior(deter)).reshape(deter.shape[0], -1)
        return torch.cat([deter, prior], -1)
    return float((_cos(roll(roll_a), target)
                  - _cos(roll(torch.zeros_like(roll_a)), target)).mean())


def _probe(n_dec: int = 1200, n_updates: int = 300, seed: int = 0) -> dict:
    from .cores import build_arm, make_optimizer
    from .survival import random_action
    from .w0 import W0
    from .tests.lc_03_survival_screening import _borrow
    torch.set_num_threads(1)
    cal, prov = _borrow()
    assert cal is not None, prov
    w = W0(seed=seed, j0=cal["j0"], alpha=cal["alpha"], lethal=True)
    rng = np.random.RandomState(seed)
    hist, life = [], 0
    for _ in range(n_dec):
        obs = w.observe()
        a = random_action(rng).astype(np.float32)
        w.decide(a)
        hist.append((obs, a, life))
        if w.died_this_decision:
            life += 1
    K = 5
    ok = [i for i in range(SEQ, len(hist) - K)
          if hist[i - SEQ][2] == hist[i + K][2]]
    split = int(0.75 * len(ok))
    train_i, test_i = ok[:split], ok[split:]
    out = {"decisions": n_dec, "lives": life + 1, "windows_train": len(train_i),
           "windows_test": len(test_i)}
    dropped = ("language",)
    torch.manual_seed(9_000)
    core = DynWorldCore()
    opt = make_optimizer(core, LC_LR)
    gen = np.random.RandomState(1)
    out["delta5_test_before"] = round(delta5_seq(core.eval(), hist, test_i), 6)
    starts_ok = [i - SEQ for i in train_i]
    log = []
    core.train()
    for u in range(n_updates):
        idx = gen.choice(starts_ok, 16)
        obs, acts = _windows(hist, list(idx), SEQ)
        loss, parts = core.seq_loss(obs, acts, dropped)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if u == 0:
            gi = core.gru.weight_ih.grad
            out["grad_first_update"] = {
                "action_cols_max": float(gi[:, LATENT:].abs().max()),
                "weight_hh_max": float(core.gru.weight_hh.grad.abs().max())}
        opt.step()
        if u % 50 == 0 or u == n_updates - 1:
            log.append({"u": u, **{k: round(v, 5) for k, v in parts.items()}})
    core.eval()
    out["loss_log"] = log
    out["delta5_test_after"] = round(delta5_seq(core, hist, test_i), 6)
    # the seated arm through the SAME filtered measurement, for comparison
    torch.manual_seed(9_000)
    seated = build_arm("wm-latent").eval()
    seated.filter_seq = lambda o, a, d=(): DynWorldCore.filter_seq(seated, o, a, d)
    seated._cell = lambda zp, a, h, z: DynWorldCore._cell(seated, zp, a, h, z)
    out["seated_wm_latent_delta5_untrained"] = round(
        delta5_seq(seated, hist, test_i), 6)
    return out


if __name__ == "__main__":
    if (sys.argv[1:] or [""])[0] == "probe":
        print(json.dumps(_probe(), indent=1))
    else:
        raise SystemExit("usage: python -m experiments.cores_dyn probe")
