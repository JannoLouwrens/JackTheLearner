"""cores.py — the candidate learning cores of the LC bakeoff, in ONE place.

Why this file exists rather than a copy inside each spec: LESSONS.md, *"two
kernels re-implementing one operation is the defect; the mode bug was only its
symptom"*. LC.01 (unison admission), LC.02 (throughput) and LC.03/LC.04 (the
bakeoff itself) all need the same five arms. One definition, one place to fix,
one place to guard.

The arms are `docs/research/LEARNING_CORE.md` §5.4:

    A0  ppo-needs    tuned PPO (F9) + L_masked_cross_modal   [admissible only
    A1  ppo-lp       A0 + a learning-progress value head       with the MCM term]
    A2  dreamer-xs   RSSM + per-modality decoder + KL
    A3  wm-efe       A2's world model byte-identical, EFE actor
    A4  wm-latent    A2 with the decoder deleted; latent prediction vs an
                     EMA target encoder (the JEPA move)

plus the two things LC.01 needs in order to be a measurement rather than a
ceremony:

    N   unbound      per-modality encoders, concat, NO cross-modal loss term.
                     Its U2 finite difference must read EXACTLY 0.0 — that
                     number is what U2 is measured against.
    C1  leaky        a core with a private path from proprioception to the
                     action, bypassing the shared latent. MUST FAIL U1.
    C2  dreamer-naive A2 with DreamerV3's shipped shared `loss_scales.rec`
                     semantics — per-key loss SUMMED over dimensions instead
                     of averaged. MUST FAIL U4. This is §3.2.6 landmine 4
                     built as a fixture so the detector has a positive control.

DESIGN NOTES THAT ARE PART OF THE PRE-REGISTRATION

* **Modality dropout is a supported input condition, not a zero-fill.** Each
  encoder owns a learned `missing` embedding; dropping a modality substitutes
  it. (U3.)
* **Every probe uses the deterministic path.** The stochastic state is taken as
  its expectation (the categorical probabilities) rather than a straight-through
  sample, everywhere in this file. LC.01 measures finite differences of
  gradients; a sampler in the graph would make the difference measure the
  sampler. LC.03 trains with sampling; that is a different question and it is
  stated here so nobody reads this file as the training implementation.
* **`loss_scales` are declared, not defaulted.** `needs` carries 2.0 and every
  other key 1.0, and per-key losses are MEAN-reduced over dimensions. Both
  halves are deliberate: mean reduction removes the dimension-count bias
  (a 32-ray retina against a 6-dim drive vector) and the 2.0 says out loud that
  the need-state is the modality this project is about. F10 then checks the
  OUTCOME rather than trusting the declaration.
* **No dropout anywhere** (F2), and every module is constructed in `.eval()`
  discipline by the caller (F1).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# The modality contract.
#
# U1 (LEARNING_CORE.md §5.0b) names the full token set: vision, audio, touch,
# proprioception, need-state, language. W0 itself supplies a subset at runtime
# (W0-4: proprio + touch + drive + the ray retina) — which is exactly why U3
# exists: a core must handle a missing modality as an input condition. The
# ADMISSION test is about the core, so it is run against the full set.
# ---------------------------------------------------------------------------
MODALITIES: Dict[str, int] = {
    "vision": 32,     # the _Retina rays PG.4 already uses
    "audio": 16,      # ContactAudio bands
    "touch": 8,       # contact sensors
    "proprio": 12,    # climber-rover, 8 actuated DoF + body pose
    "needs": 6,       # drive vector [e, i, w, d(h), e_dot, i_dot]
    "language": 32,   # a sentence embedding from the frozen text tower
}
PLACEBO_KEY = "placebo"
PLACEBO_DIM = MODALITIES["needs"]      # matched dimension to the need-state
ACTION_DIM = 8                         # climber-rover, 8 actuated DoF

LOSS_SCALES: Dict[str, float] = {k: 1.0 for k in MODALITIES}
LOSS_SCALES["needs"] = 2.0
LOSS_SCALES[PLACEBO_KEY] = 1.0

EMB = 32          # per-modality embedding width
LATENT = 64       # the shared latent every modality must reach
DETER = 256       # RSSM recurrent state (LEARNING_CORE.md §5.4 A2)
STOCH_N, STOCH_C = 32, 8   # 32 categoricals of 8 classes, 1% unimix
UNIMIX = 0.01
FREE_BITS = 1.0


def obs_keys(with_placebo: bool = True) -> List[str]:
    ks = list(MODALITIES)
    if with_placebo:
        ks.append(PLACEBO_KEY)
    return ks


def obs_dims(with_placebo: bool = True) -> Dict[str, int]:
    d = dict(MODALITIES)
    if with_placebo:
        d[PLACEBO_KEY] = PLACEBO_DIM
    return d


def make_batch(seed: int, n: int = 64, with_placebo: bool = True,
               shift: float = 0.0) -> Dict[str, torch.Tensor]:
    """A W0-shaped multimodal batch drawn from a SHARED latent cause.

    Cross-modal prediction has to be a real task or U2/U4 measure nothing: the
    modalities are linear projections of one 8-dim situation latent plus
    per-key noise, standardised per key. `placebo` is drawn from the same
    marginal (mean 0, unit scale, matched dimension) and is independent of the
    latent — matched statistics, zero information, which is what makes it a
    control for U4 rather than just another key.
    """
    g = torch.Generator().manual_seed(1_000_003 + seed)
    z = torch.randn(n, 8, generator=g)
    out: Dict[str, torch.Tensor] = {}
    for k, d in obs_dims(with_placebo).items():
        w = torch.randn(8, d, generator=g) / (8 ** 0.5)
        eps = torch.randn(n, d, generator=g)
        x = eps if k == PLACEBO_KEY else (z @ w + 0.5 * eps)
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        out[k] = x + shift
    return out


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    # LayerNorm before every dense layer, tanh, no dropout (F2, F9).
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.Tanh(),
                         nn.LayerNorm(h), nn.Linear(h, o))


class Core(nn.Module):
    """The shared skeleton: per-modality encoders -> ONE shared latent.

    Subclasses supply `binding_term`, `binding_loss` and `uncertainty`. The
    actor signature is `act(obs, z)` on purpose: an honest core ignores `obs`
    and reads only the shared state, and U1's second half detaches `z` and
    requires the action's gradient to every raw modality to be exactly zero.
    A core with a private path cannot hide from that.
    """

    binding_term = "none"

    def __init__(self, with_placebo: bool = True):
        super().__init__()
        self.keys = obs_keys(with_placebo)
        self.dims = obs_dims(with_placebo)
        self.enc = nn.ModuleDict(
            {k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, EMB), nn.Tanh())
             for k, d in self.dims.items()})
        self.missing = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(EMB)) for k in self.keys})
        self.trunk = _mlp(EMB * len(self.keys), 128, LATENT)
        self.actor = _mlp(LATENT, 128, ACTION_DIM)
        # F9: separate policy and value networks, policy last layer 100x small.
        self.critic = _mlp(LATENT, 128, 1)
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    # -- encoding ----------------------------------------------------------
    def embed(self, obs: Dict[str, torch.Tensor],
              dropped: Iterable[str] = ()) -> Dict[str, torch.Tensor]:
        dropped = set(dropped)
        n = next(iter(obs.values())).shape[0]
        out = {}
        for k in self.keys:
            if k in dropped:
                out[k] = self.missing[k].expand(n, EMB)
            else:
                out[k] = self.enc[k](obs[k])
        return out

    def encode(self, obs: Dict[str, torch.Tensor],
               dropped: Iterable[str] = ()) -> torch.Tensor:
        e = self.embed(obs, dropped)
        return self.trunk(torch.cat([e[k] for k in self.keys], dim=-1))

    def shared_state(self, obs: Dict[str, torch.Tensor],
                     dropped: Iterable[str] = ()) -> torch.Tensor:
        """THE shared representation the actor is allowed to read, and the
        only thing it is allowed to read. For the PPO arms that is the fused
        latent; the world-model arms override it with the RSSM state, because
        an actor that read the pre-RSSM latent would leave the model off the
        action path and U1 would certify a core the bakeoff never runs."""
        return self.encode(obs, dropped)

    def act(self, obs: Dict[str, torch.Tensor], s: torch.Tensor) -> torch.Tensor:
        return self.actor(s)                      # `obs` deliberately unused

    def act_deterministic(self, obs: Dict[str, torch.Tensor],
                          dropped: Iterable[str] = ()) -> torch.Tensor:
        return self.act(obs, self.shared_state(obs, dropped))

    def encoder_params(self, key: str) -> List[nn.Parameter]:
        return [p for p in self.enc[key].parameters() if p.requires_grad]

    # -- the two things every arm must declare ------------------------------
    def binding_loss(self, obs, dropped: Iterable[str] = ()
                     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def uncertainty(self, obs, dropped: Iterable[str] = ()) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# A0 / A1 — PPO with L_masked_cross_modal (bare PPO is inadmissible, §5.0b)
# ---------------------------------------------------------------------------
class PPOCore(Core):
    """`ppo-needs` (+ `ppo-lp` when `lp=True`).

    The declared binding term is L_masked_cross_modal and nothing else: mask
    one modality's embedding, predict that modality's raw input from the SHARED
    latent built out of the others. Modality A's gradient reaches modality B's
    encoder because B is an input to the latent from which A is predicted.
    "It's a shared trunk" is not an answer (MULTIMODAL_BINDING.md: pi-0.5 reads
    its prompt at 99.3% linear-probe accuracy while behaving invariantly to it),
    so the term is named here and measured by U2 there.
    """

    binding_term = "L_masked_cross_modal"

    def __init__(self, with_placebo: bool = True, lp: bool = False):
        super().__init__(with_placebo)
        self.pred = nn.ModuleDict(
            {k: _mlp(LATENT, 128, d) for k, d in self.dims.items()})
        self.lp = lp
        if lp:
            # Two value heads (RND's design), not a summed scalar — removal is
            # then the deletion of a head. PURPOSE_AND_SCAFFOLDING.md §2.8.
            self.critic_lp = _mlp(LATENT, 128, 1)

    def _per_key(self, obs, dropped) -> Dict[str, torch.Tensor]:
        per = {}
        for k in self.keys:
            z = self.encode(obs, set(dropped) | {k})
            per[k] = LOSS_SCALES[k] * F.mse_loss(self.pred[k](z), obs[k])
        return per

    def binding_loss(self, obs, dropped=()):
        per = self._per_key(obs, dropped)
        return torch.stack(list(per.values())).sum(), per

    def uncertainty(self, obs, dropped=()):
        # The core's internal uncertainty IS its masked cross-modal surprise:
        # how badly it predicts each sense from the others. Declared here so
        # U3 measures a named quantity rather than whatever moved.
        per = self._per_key(obs, dropped)
        return torch.stack(list(per.values())).mean()


class LeakyPPOCore(PPOCore):
    """C1, the positive control for U1: proprioception has a private wire to
    the action, bypassing the shared latent. Must FAIL U1."""

    def __init__(self, with_placebo: bool = True):
        super().__init__(with_placebo)
        self.private = nn.Linear(self.dims["proprio"], ACTION_DIM)

    def act(self, obs, z):
        return self.actor(z) + self.private(obs["proprio"])


# ---------------------------------------------------------------------------
# N — the unbound null. Per-modality encoders, concat, no cross-modal term.
# ---------------------------------------------------------------------------
class UnboundCore(Core):
    """Each modality reconstructs itself from its OWN embedding. The action
    still sees a concatenation of everything, so it looks like a unified brain
    at the shape level — and its cross-modal finite difference is exactly 0.0,
    because dL/d(theta_B) has no path to x_A. That number is U2's ruler."""

    binding_term = "per-modality autoencoding (NO cross-modal term)"

    def __init__(self, with_placebo: bool = True):
        super().__init__(with_placebo)
        self.dec = nn.ModuleDict(
            {k: _mlp(EMB, 128, d) for k, d in self.dims.items()})

    def _per_key(self, obs, dropped) -> Dict[str, torch.Tensor]:
        e = self.embed(obs, dropped)
        return {k: LOSS_SCALES[k] * F.mse_loss(self.dec[k](e[k]), obs[k])
                for k in self.keys}

    def binding_loss(self, obs, dropped=()):
        per = self._per_key(obs, dropped)
        return torch.stack(list(per.values())).sum(), per

    def uncertainty(self, obs, dropped=()):
        per = self._per_key(obs, dropped)
        return torch.stack(list(per.values())).mean()


# ---------------------------------------------------------------------------
# A2 / A3 / C2 — the RSSM world model
# ---------------------------------------------------------------------------
class WorldModelCore(Core):
    """`dreamer-xs`: RSSM (GRU deter 256, 32x8 categorical, 1% unimix),
    per-modality decoder, KL with free bits.

    Every sense enters one latent by construction and the learning rule itself
    is the binding term — this is the arm in which question (a) is answered by
    the objective rather than by concatenation.

    `sum_reduce=True` is C2, the U4 positive control: DreamerV3's shipped
    shared `loss_scales.rec` semantics, where a key's contribution scales with
    its DIMENSION COUNT. Here that is 32 retina rays against a 6-dim drive
    vector; in the shipped configuration it is 12,288 against 10.
    """

    binding_term = "per-modality prediction loss + posterior/prior KL"

    def __init__(self, with_placebo: bool = True, decoder: bool = True,
                 ensemble: int = 0, sum_reduce: bool = False):
        super().__init__(with_placebo)
        self.stoch_dim = STOCH_N * STOCH_C
        self.state_dim = DETER + self.stoch_dim
        self.gru = nn.GRUCell(LATENT + ACTION_DIM, DETER)
        self.post = _mlp(DETER + LATENT, 256, self.stoch_dim)
        self.prior = _mlp(DETER, 256, self.stoch_dim)
        self.sum_reduce = sum_reduce
        self.decoder = decoder
        if decoder:
            self.dec = nn.ModuleDict(
                {k: _mlp(self.state_dim, 256, d) for k, d in self.dims.items()})
        else:
            # A4: latent prediction against an EMA target encoder.
            self.latent_pred = _mlp(self.state_dim, 256, LATENT)
        self.ensemble = nn.ModuleList(
            [_mlp(self.state_dim + ACTION_DIM, 128, LATENT)
             for _ in range(ensemble)])
        # The actor reads the MODEL STATE (deter + stoch), as DreamerV3's does.
        self.actor = _mlp(self.state_dim, 128, ACTION_DIM)
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def shared_state(self, obs, dropped=()):
        return self.rssm(obs, dropped)["state"]

    # -- RSSM --------------------------------------------------------------
    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits.view(-1, STOCH_N, STOCH_C), dim=-1)
        return (1 - UNIMIX) * p + UNIMIX / STOCH_C

    def rssm(self, obs, dropped=()) -> Dict[str, torch.Tensor]:
        z = self.encode(obs, dropped)
        n = z.shape[0]
        h0 = torch.zeros(n, DETER)
        a0 = torch.zeros(n, ACTION_DIM)
        deter = self.gru(torch.cat([z, a0], -1), h0)
        post = self._probs(self.post(torch.cat([deter, z], -1)))
        prior = self._probs(self.prior(deter))
        # Deterministic path: the stochastic state is its EXPECTATION, never a
        # sample. See the module docstring — a sampler in the graph would make
        # U2's finite difference measure the sampler.
        state = torch.cat([deter, post.reshape(n, -1)], -1)
        return {"z": z, "deter": deter, "post": post, "prior": prior,
                "state": state}

    def _kl(self, post, prior) -> torch.Tensor:
        kl = (post * (torch.log(post + 1e-8) - torch.log(prior + 1e-8))).sum(-1)
        return torch.clamp(kl.mean(0), min=FREE_BITS / STOCH_N).sum()

    def _reduce(self, pred, target, key) -> torch.Tensor:
        se = (pred - target) ** 2
        # mean over dims (the fix) vs sum over dims (the landmine).
        per_sample = se.sum(-1) if self.sum_reduce else se.mean(-1)
        return LOSS_SCALES[key] * per_sample.mean()

    def _per_key(self, obs, dropped) -> Tuple[Dict[str, torch.Tensor], Dict]:
        s = self.rssm(obs, dropped)
        if self.decoder:
            per = {k: self._reduce(self.dec[k](s["state"]), obs[k], k)
                   for k in self.keys}
        else:
            # A4 has no decoder, so F10's "per-modality share of the
            # prediction loss" is defined by leave-one-out attribution on the
            # latent-prediction objective: how much of the predicted latent
            # each modality is responsible for, weighted by the same declared
            # loss_scales. Declared here because it is a choice, not a default.
            target = s["z"].detach()
            per = {}
            for k in self.keys:
                sk = self.rssm(obs, set(dropped) | {k})
                per[k] = LOSS_SCALES[k] * F.mse_loss(
                    self.latent_pred(sk["state"]), target)
        return per, s

    def binding_loss(self, obs, dropped=()):
        per, s = self._per_key(obs, dropped)
        total = torch.stack(list(per.values())).sum() + self._kl(s["post"], s["prior"])
        return total, per

    def uncertainty(self, obs, dropped=()):
        s = self.rssm(obs, dropped)
        if self.ensemble:
            # A3's epistemic term: disagreement across K latent dynamics heads
            # (§3.3.3 — this IS the EFE epistemic term's standard estimator).
            a = torch.zeros(s["state"].shape[0], ACTION_DIM)
            preds = torch.stack([h(torch.cat([s["state"], a], -1))
                                 for h in self.ensemble])
            return preds.var(0).mean()
        # A2/A4: mean categorical entropy of the posterior.
        return -(s["post"] * torch.log(s["post"] + 1e-8)).sum(-1).mean()


# ---------------------------------------------------------------------------
# The arm table. `admissible` names the five §5.4 candidates; the rest are the
# controls LC.01 needs in order to be able to fail.
# ---------------------------------------------------------------------------
def build_arm(name: str, with_placebo: bool = True) -> Core:
    if name == "ppo-needs":
        return PPOCore(with_placebo, lp=False)
    if name == "ppo-lp":
        return PPOCore(with_placebo, lp=True)
    if name == "dreamer-xs":
        return WorldModelCore(with_placebo, decoder=True)
    if name == "wm-efe":
        return WorldModelCore(with_placebo, decoder=True, ensemble=5)
    if name == "wm-latent":
        return WorldModelCore(with_placebo, decoder=False)
    if name == "unbound":
        return UnboundCore(with_placebo)
    if name == "leaky":
        return LeakyPPOCore(with_placebo)
    if name == "dreamer-naive":
        return WorldModelCore(with_placebo, decoder=True, sum_reduce=True)
    raise KeyError(f"unknown arm: {name}")


CANDIDATE_ARMS: Tuple[str, ...] = (
    "ppo-needs", "ppo-lp", "dreamer-xs", "wm-efe", "wm-latent")
CONTROL_ARMS: Tuple[str, ...] = ("unbound", "leaky", "dreamer-naive")


def n_params(core: nn.Module) -> int:
    return sum(p.numel() for p in core.parameters() if p.requires_grad)


def n_dropout_modules(core: nn.Module) -> int:
    """F2: dropout is ABSENT from every learning core, not disabled at eval."""
    return sum(1 for m in core.modules()
               if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                 nn.Dropout3d, nn.AlphaDropout)))
