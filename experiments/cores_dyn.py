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

MEASURED 2026-11-13 18:39 (L_next NORMALISED by sg per-dim var(z); probe
seeds 0/1/2, same 1200 dec / 300 upd; + shuffled-action control = roll with
the NEXT window's actions from the same start):
  * normalised L_next 4.50/4.79/4.11 -> 0.42/0.42/0.35 (u299), no rebound:
    ~60% of z's variance at t is predicted from (deter_{t-1}, a_{t-1}).
    The 18:50 rebound WAS the unnormalised target widening. Loss fix OK.
  * action grad first update .071/.126/.083, weight_hh .182/.244/.214.
  * Delta_5 (real - zero actions) before .0147/.0151/.0159 -> after
    .0069/.0084/.0085 — FALLS on 3/3 seeds. Real - shuffled actions before
    .0114/.0110/.0119 -> after .0077/.0088/.0078 — also FALLS 3/3.
    KL sits at the free-bits floor (1.00) from u50: posterior == prior, so
    the GRU need not carry action information to win L_next; the next
    latent is predicted from state, and training WASHES OUT the random
    init's action sensitivity. Not a model of what he does. Open question
    the probe cannot answer: how much does a random-policy action change
    W0's observation 5 decisions later at all (the oracle ceiling)?

MEASURED 2026-11-13 19:35 (`action_share`, same probe, seeds 0/1/2; held-out
test windows unless marked; units = fraction of z_t's per-dim variance):
  * untrained: gap_shuf -.013/-.004/-.015 (the random init credits nothing
    to the action it is told), unexpl_real 6.4/5.5/4.9.
  * after 300 updates: gap_shuf +.146/+.132/+.067 (train windows +.064/
    +.084/+.067), gap_zero +.081/+.066/+.028; unexpl_real 1.01/.53/.45
    (train .42/.39/.44: seed 0 overfits 884 windows); copying z_{t-1} as the
    guess leaves 1.97/.96/.76 -> the arm beats persistence on 3/3.
READING: training does NOT wash actions out of ONE step - it goes from 0 to
~.07-.15 of z's variance credited to the told action (world ceiling for one
decision: .43 of OBS variance, a different space, so a rough bound). The
Delta_5 fall (ada42ca) is therefore not "actions washed out"; the untrained
GRU's +.015 is arbitrary random-init sensitivity that a cosine-on-state
metric scores as dynamics. Delta_5 vs zero actions cannot tell learned from
random sensitivity; gap_shuf from an untrained start (~0) can. Next: the
same gap at K=5 open-loop (prior roll), then a K-step loss to raise it.

MEASURED 2026-11-13 19:45 (`action_share_k`, K=5 OPEN-LOOP, the arm feeding
its own next_pred back as z; held-out; seeds 0/1/2):
  * untrained gap_shuf -.014/+.007/-.028; trained +.135/+.108/+.055
    (gap_zero +.091/+.062/+.021); unexpl_real 1.03/.64/.53 vs copying z_t
    1.95/1.05/.88 -> beats persistence 3/3. Same run: Delta_5 .0069/.0084/.0085
    (below its untrained .015).
READING: 5 decisions out, the trained arm's imagined z depends on the
actions it is told (untrained: no dependence), on 3/3 seeds - it IS a model
of what he does, weakly (seed 2 smallest). Delta_5 (`delta5_seq`, LC.09's
guard form) rolls with z HELD at z_t - a roll this arm never trained on -
and scores cosine on state: it reads the untrained random GRU as more
"dynamic" than the trained one. The guard measures the wrong thing for any
arm whose GRU input is a moving z. Proposed (INBOX; LC.09 is J3's spec, the
change strengthens it): replace Delta_5 > 0 with gap_shuf(K=5, own-roll)
> its untrained-init value by a declared margin, on >= 3 seeds.
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

    # -- R2D2 stored state (survival_rec.py, 11-13 20:50) --------------------
    # The rollout keeps (deter_{t-1}, z_{t-1}, a_{t-1}) with each observation;
    # a replay batch hands them back through `use_stored` before lc_update, so
    # shared_state/binding_loss advance ONE GRU step from the stored state
    # (Kapturowski et al. 2019, stored-state strategy, burn-in 0; the stored
    # deter was made by older weights - declared staleness, not hidden).
    _stored: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def use_stored(self, stored) -> None:
        self._stored = stored

    def _from_stored(self, obs, dropped):
        st = self._stored
        if st is None:
            return self.rssm(obs, dropped)
        h, zp, a = st
        return self.rssm(obs, dropped, h=h, z_prev=zp, a_prev=a)

    def shared_state(self, obs, dropped=()):
        return self._from_stored(obs, dropped)["state"]

    def binding_loss(self, obs, dropped=()):
        """With a stored state: the one-step form of `seq_loss` (L_next from
        (deter_{t-1}, a_{t-1}) normalised by z's batch variance + KL + VICReg),
        so gradient reaches the GRU's action columns from i.i.d. replay.
        Without one: the seated WorldModelCore binding loss, unchanged."""
        if self._stored is None:
            return super().binding_loss(obs, dropped)
        s = self._from_stored(obs, dropped)
        zt = s["z"].detach()
        zvar = zt.var(0, unbiased=False).clamp_min(1e-4)
        l_next = ((self.next_pred(s["deter"]) - zt).pow(2) / zvar).mean()
        l_kl = self._kl(s["post"], s["prior"])
        l_var = F.relu(1.0 - s["z"].std(0)).mean()
        total = l_next + l_kl + VAR_COEF * l_var
        return total, {"next": l_next.detach(), "kl": l_kl.detach(),
                       "var": l_var.detach()}

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
        # NORMALISED (11-13 19:00): MSE divided by the target's own per-dim
        # variance (stop-grad), so the VICReg hinge widening z cannot inflate
        # L_next and the term reads "fraction of z's variance unexplained".
        zt = torch.cat([s["z"] for s in ss[1:]], 0).detach()
        zvar = zt.var(0, unbiased=False).clamp_min(1e-4)
        l_next = torch.stack([((self.next_pred(s["deter"]) - s["z"].detach())
                               .pow(2) / zvar).mean() for s in ss[1:]]).mean()
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
def delta5_seq(core, hist, starts, K=5, dropped=("language",),
               shuffled: bool = False) -> float:
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
    real = _cos(roll(roll_a), target)
    if shuffled:   # control: same start, ANOTHER window's actions
        perm = torch.roll(torch.arange(roll_a.shape[1]), 1)
        return float((real - _cos(roll(roll_a[:, perm]), target)).mean())
    return float((real - _cos(roll(torch.zeros_like(roll_a)), target)).mean())


@torch.no_grad()
def action_share(core, hist, starts, dropped=("language",)) -> Dict[str, float]:
    """The arm's OWN one-step action share (11-13 19:30), on held-out windows
    filtered from their past: at each step t >= 1 of a SEQ window,
    pred_real = next_pred(GRU([z_{t-1}, a_{t-1}], h_{t-1})) vs the same with
    a_{t-1} replaced by ANOTHER window's action (shuffled) and by zeros.
    Errors are per-dim normalised by z_t's variance (as L_next), so
      unexpl_real  = fraction of z_t's variance the arm leaves unexplained;
      gap_shuf     = unexpl_shuf - unexpl_real = share of z_t's variance
                     the arm credits to the ACTION it was told.
    The world's ceiling for one decision is .43 of obs variance (K=1,
    w0_action_ceiling_probe); gap_shuf ~ 0 means the arm predicts from
    state alone."""
    obs, acts = _windows(hist, [i - SEQ for i in starts], SEQ)
    ss = core.filter_seq(obs, acts, dropped)
    zt = torch.cat([s["z"] for s in ss[1:]], 0)
    zvar = zt.var(0, unbiased=False).clamp_min(1e-4)
    perm = torch.roll(torch.arange(acts.shape[1]), 1)
    err = {"real": [], "shuf": [], "zero": [], "copy": []}
    for t in range(1, SEQ):
        h, zp, a = ss[t - 1]["deter"], ss[t - 1]["z"], acts[t - 1]
        z = ss[t]["z"]
        for k, aa in (("real", a), ("shuf", a[perm]),
                      ("zero", torch.zeros_like(a))):
            d = core.gru(torch.cat([zp, aa], -1), h)
            err[k].append(((core.next_pred(d) - z).pow(2) / zvar).mean())
        err["copy"].append(((zp - z).pow(2) / zvar).mean())   # z_{t-1} as guess
    u = {k: float(torch.stack(v).mean()) for k, v in err.items()}
    return {"unexpl_real": round(u["real"], 4), "unexpl_shuf": round(u["shuf"], 4),
            "unexpl_zero": round(u["zero"], 4),
            "unexpl_copy_prev_z": round(u["copy"], 4),
            "gap_shuf": round(u["shuf"] - u["real"], 4),
            "gap_zero": round(u["zero"] - u["real"], 4)}


@torch.no_grad()
def action_share_k(core, hist, starts, K=5, dropped=("language",)) -> Dict[str, float]:
    """K-step OPEN-LOOP version of `action_share` (11-13 19:40): from the
    filtered state at t (SEQ burn-in), imagine K decisions feeding the arm's
    OWN prediction back as the next z (z <- next_pred(deter)), under the real
    actions a_t..a_{t+K-1}, another window's (shuffled) or zeros; score
    next_pred(deter_K) against the encoder's z_{t+K}, per-dim normalised by
    z_{t+K}'s variance. (delta5_seq rolls with z held fixed at z_t, which an
    arm trained on moving z never saw - this is the roll it was trained for.)"""
    T = SEQ + K
    obs, acts = _windows(hist, [i - SEQ for i in starts], T)
    ss = core.filter_seq(obs, acts, dropped)
    s_t = ss[SEQ - 1]
    zK = ss[SEQ - 1 + K]["z"]
    zvar = zK.var(0, unbiased=False).clamp_min(1e-4)
    roll_a = acts[SEQ - 1:SEQ - 1 + K]
    perm = torch.roll(torch.arange(roll_a.shape[1]), 1)

    def roll(a_seq):
        deter, z = s_t["deter"], s_t["z"]
        for k in range(K):
            deter = core.gru(torch.cat([z, a_seq[k]], -1), deter)
            z = core.next_pred(deter)
        return float(((z - zK).pow(2) / zvar).mean())
    r, sh, ze = roll(roll_a), roll(roll_a[:, perm]), roll(torch.zeros_like(roll_a))
    cp = float(((s_t["z"] - zK).pow(2) / zvar).mean())
    return {"K": K, "unexpl_real": round(r, 4), "unexpl_shuf": round(sh, 4),
            "unexpl_zero": round(ze, 4), "unexpl_copy_z_t": round(cp, 4),
            "gap_shuf": round(sh - r, 4), "gap_zero": round(ze - r, 4)}


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
    out["action_share_before"] = action_share(core.eval(), hist, test_i)
    out["action_share5_before"] = action_share_k(core, hist, test_i)
    out["delta5_test_before"] = round(delta5_seq(core.eval(), hist, test_i), 6)
    out["delta5shuf_test_before"] = round(
        delta5_seq(core, hist, test_i, shuffled=True), 6)
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
    out["action_share_after"] = action_share(core, hist, test_i)
    out["action_share_train_after"] = action_share(core, hist, train_i[:300])
    out["action_share5_after"] = action_share_k(core, hist, test_i)
    out["delta5_test_after"] = round(delta5_seq(core, hist, test_i), 6)
    out["delta5shuf_test_after"] = round(
        delta5_seq(core, hist, test_i, shuffled=True), 6)
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
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        print(json.dumps(_probe(seed=sd), indent=1))
    else:
        raise SystemExit("usage: python -m experiments.cores_dyn probe")
