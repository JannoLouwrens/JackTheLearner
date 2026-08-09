"""LC.01 — Every candidate core takes every sense into one latent, or it is
not a candidate.

ADMISSION-1 of `docs/research/LEARNING_CORE.md` §5.0b, which exists because
SYSTEM.md's first hard constraint is constitutional: *"No learning core without
unison. A core that wins the task but fails binding has not won; it has changed
the subject."* This spec runs BEFORE any learning, on CPU, and decides which
arms are even allowed into LC.03/LC.04.

The four requirements, and how each is measured on the live modules in
`experiments/cores.py` (never on a docstring — a declared binding term is a
claim, and this is the test that could refute it):

  U1  Every modality reaches the ONE shared state, and no modality has a
      private path to the action.
      (a) d(action)/d(x_m) != 0 for every m, through the deterministic path;
      (b) with the shared latent DETACHED, d(action)/d(x_m) == 0 for every m.
      (b) is the half that bites: a core can pass (a) while wiring one sense
      straight to the actuators.
  U2  A named loss term by which modality A's gradient reaches modality B's
      ENCODER. Measured by finite difference, not asserted: take
      g_B = d(L_bind)/d(theta_B), perturb x_A by eps, recompute, and require
      ||g_B' - g_B|| / eps > 0 for every ordered pair A != B.
      "It's a shared trunk" is not an answer — MULTIMODAL_BINDING.md records
      pi-0.5 encoding its language prompt at 99.3% linear-probe accuracy while
      behaving invariantly to it.
  U3  A missing modality is an input condition the core handles (a learned
      `missing` embedding), not a zero-fill convention: dropping each modality
      raises no shape error AND changes the core's declared internal
      uncertainty.
  U4  No modality is silently down-weighted out of existence: the need-state
      holds >= 1/|M| of the total prediction loss at init, with |M| = 6.
      This is §3.2.6 landmine 4 — with a shared `loss_scales.rec` a 64x64x3
      image contributes 12,288 reconstruction terms and a 10-dim needs vector
      contributes 10. It is the nn.Dropout(p=0.1) shape exactly: every loss
      curve looks correct while the modality this project is about is deleted.

PRE-REGISTERED DECISION RULE (fixed before the run; the spec text states what
happens to a failing ARM but not what happens to the SPEC, so it is stated
here rather than decided afterwards):

  PASS  all five §5.4 candidate arms — ppo-needs, ppo-lp, dreamer-xs, wm-efe,
        wm-latent — satisfy U1-U4 on all 3 seeds, and all three controls
        behave as pre-registered.
  FAIL  a candidate arm fails a requirement. The metrics name it; per the
        spec's `falsified_by` that arm is EXCLUDED from LC.03/LC.04 — and the
        remedy is to fix the ARM, never the requirement (SYSTEM.md law 4).
  VOID  a CONTROL misbehaves. Then the probe cannot discriminate and nothing
        here tested the claim.

THE CONTROLS — three, because a detector that cannot see its own positive
control has measured nothing (T0.13's lesson):

  unbound        per-modality encoders, concat, no cross-modal term. Its U2
                 finite difference must read EXACTLY 0.0 — dL/d(theta_B) has
                 no path to x_A, so the two backward passes are bit-identical.
                 That zero is the number U2 is measured against.
  leaky          a core with a private wire from proprioception to the action.
                 MUST FAIL U1. If it passes, U1(b) is not reading routing.
  dreamer-naive  dreamer-xs with DreamerV3's shipped shared-`loss_scales`
                 semantics: per-key loss SUMMED over dimensions instead of
                 averaged, so a key's weight is its dimension count. MUST FAIL
                 U4. If it passes, U4 is not reading loss balance.
  placebo        a seventh input key of matched dimension (6) and matched
                 statistics (mean 0, unit scale) carrying zero information. It
                 must NOT acquire a loss share above 1/|M| in any candidate
                 arm — if noise binds as well as a sense, U4 measures capacity
                 rather than binding. Checked inside the experiment, on every
                 arm, because it is a property of each arm and not of a
                 separate run.

Also asserted, cheaply, because both are fixed invariants of §5.1 and this is
the first spec that instantiates the arms at all: F2 (dropout is ABSENT, not
disabled — 0 dropout modules in any arm) and F1 (the deterministic action path
is bit-identical across two forwards of one input). Parameter counts are
recorded for LC.02/LC.03 to inherit rather than re-declare.

WHAT THIS SPEC DOES NOT CLAIM. It is a structural probe at initialisation.
It says nothing about whether an admitted arm learns anything — that is LC.03,
and admission is a necessary condition, never a sufficient one.
"""

from __future__ import annotations

import torch

from ..cores import (CANDIDATE_ARMS, MODALITIES, PLACEBO_KEY, build_arm,
                     make_batch, n_dropout_modules, n_params)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

N_BATCH = 64
EPS = 1e-2                      # finite-difference step on a unit-scale input
UNC_REL_TOL = 1e-6              # a "changed" uncertainty must move by this
N_MODALITIES = len(MODALITIES)  # |M| = 6; the placebo key is NOT one of them
NEEDS_FLOOR = 1.0 / N_MODALITIES
PLACEBO_CEIL = 1.0 / N_MODALITIES


def _obs(seed: int, grad: bool = False):
    o = make_batch(seed, N_BATCH)
    if grad:
        o = {k: v.clone().requires_grad_(True) for k, v in o.items()}
    return o


def _u1(core, seed) -> tuple[float, float, float]:
    """Returns (passes, min |d action / d x_m|, max private-path gradient)."""
    obs = _obs(seed, grad=True)
    keys = list(obs)
    a = core.act_deterministic(obs)
    g = torch.autograd.grad(a.sum(), [obs[k] for k in keys],
                            allow_unused=True, retain_graph=False)
    reach = {k: (0.0 if gi is None else gi.abs().sum().item())
             for k, gi in zip(keys, g)}

    obs2 = _obs(seed, grad=True)
    s = core.shared_state(obs2).detach()    # cut the ONE legitimate path
    a2 = core.act(obs2, s)
    g2 = torch.autograd.grad(a2.sum(), [obs2[k] for k in keys],
                             allow_unused=True, retain_graph=False)
    private = max(0.0 if gi is None else gi.abs().sum().item() for gi in g2)

    min_reach = min(reach[k] for k in MODALITIES)   # placebo not required
    ok = (min_reach > 0.0) and (private == 0.0)
    return float(ok), min_reach, private


def _enc_grads(core, obs) -> dict:
    loss, _ = core.binding_loss(obs)
    keys = list(obs)
    params, index = [], {}
    for k in keys:
        ps = core.encoder_params(k)
        index[k] = (len(params), len(params) + len(ps))
        params.extend(ps)
    g = torch.autograd.grad(loss, params, allow_unused=True)
    out = {}
    for k in keys:
        lo, hi = index[k]
        out[k] = torch.cat([torch.zeros(1) if gi is None else gi.reshape(-1)
                            for gi in g[lo:hi]])
    return out


def _u2(core, seed) -> tuple[float, float]:
    """Finite-difference cross-modal gradient flow. Returns (passes, min fd)."""
    base_obs = _obs(seed)
    base = _enc_grads(core, base_obs)
    gen = torch.Generator().manual_seed(4242 + seed)
    worst = float("inf")
    for a_key in base_obs:
        pert = {k: v.clone() for k, v in base_obs.items()}
        d = torch.randn(pert[a_key].shape, generator=gen)
        pert[a_key] = pert[a_key] + EPS * d
        g = _enc_grads(core, pert)
        for b_key in base_obs:
            if b_key == a_key:
                continue
            fd = (g[b_key] - base[b_key]).norm().item() / EPS
            worst = min(worst, fd)
    return float(worst > 0.0), worst


def _u3(core, seed) -> tuple[float, float]:
    obs = _obs(seed)
    with torch.no_grad():
        full = core.uncertainty(obs).item()
        worst = float("inf")
        for k in MODALITIES:
            drop = core.uncertainty(obs, dropped={k}).item()   # shape error -> ERROR
            worst = min(worst, abs(drop - full) / (abs(full) + 1e-12))
    return float(worst > UNC_REL_TOL), worst


def _u4(core, seed) -> tuple[float, float, float]:
    obs = _obs(seed)
    with torch.no_grad():
        _, per = core.binding_loss(obs)
        total = sum(v.item() for v in per.values())
        needs = per["needs"].item() / total
        placebo = per[PLACEBO_KEY].item() / total
    ok = (needs >= NEEDS_FLOOR) and (placebo <= PLACEBO_CEIL)
    return float(ok), needs, placebo


def _deterministic(core, seed) -> float:
    obs = _obs(seed)
    with torch.no_grad():
        a1 = core.act_deterministic(obs)
        a2 = core.act_deterministic(obs)
    return float(torch.equal(a1, a2))


def _probe(name: str, seed: int) -> dict:
    torch.manual_seed(9_000 + seed)
    core = build_arm(name)
    core.eval()                                     # F1
    u1, reach, private = _u1(core, seed)
    u2, fd = _u2(core, seed)
    u3, unc = _u3(core, seed)
    u4, needs, placebo = _u4(core, seed)
    return {"u1": u1, "u2": u2, "u3": u3, "u4": u4,
            "admitted": float(u1 and u2 and u3 and u4),
            "min_action_reach": reach, "private_path_grad": private,
            "min_cross_modal_fd": fd, "min_uncertainty_shift": unc,
            "needs_loss_share": needs, "placebo_loss_share": placebo,
            "params": float(n_params(core)),
            "dropout_modules": float(n_dropout_modules(core)),
            "deterministic": _deterministic(core, seed)}


def _experiment(seed: int) -> dict:
    m: dict = {}
    admitted = 0
    for name in CANDIDATE_ARMS:
        p = _probe(name, seed)
        for k, v in p.items():
            m[f"{name}/{k}"] = v
        admitted += int(p["admitted"])
        m[f"{name}/dropout_modules"] = p["dropout_modules"]
    m["arms_probed"] = float(len(CANDIDATE_ARMS))
    m["arms_admitted"] = float(admitted)
    m["dropout_modules_total"] = sum(
        m[f"{n}/dropout_modules"] for n in CANDIDATE_ARMS)
    m["deterministic_arms"] = sum(
        m[f"{n}/deterministic"] for n in CANDIDATE_ARMS)
    m["unison_admission_conjunction"] = float(
        admitted == len(CANDIDATE_ARMS)
        and m["dropout_modules_total"] == 0.0
        and m["deterministic_arms"] == float(len(CANDIDATE_ARMS)))
    return m


def _control(seed: int) -> dict:
    """The three cores that MUST fail their respective requirement."""
    torch.manual_seed(9_000 + seed)
    unbound = build_arm("unbound"); unbound.eval()
    leaky = build_arm("leaky"); leaky.eval()
    naive = build_arm("dreamer-naive"); naive.eval()
    _, unbound_fd = _u2(unbound, seed)
    leaky_u1, _, leaky_private = _u1(leaky, seed)
    naive_u4, naive_needs, _ = _u4(naive, seed)
    return {"unbound_cross_modal_fd": unbound_fd,
            "leaky_u1": leaky_u1,
            "leaky_private_path_grad": leaky_private,
            "naive_u4": naive_u4,
            "naive_needs_loss_share": naive_needs}


def _check(m: dict, c: dict):
    # --- the instrument first. A control that behaves the wrong way means the
    # probe cannot tell binding from plumbing, and nothing here tested the
    # claim: VOID, not FAIL.
    if c.get("unbound_cross_modal_fd", -1.0) != 0.0:
        return Status.VOID          # U2 is reading autograd plumbing
    if c.get("leaky_u1", 1.0) != 0.0:
        return Status.VOID          # U1 cannot see a private path to the action
    if c.get("naive_u4", 1.0) != 0.0:
        return Status.VOID          # U4 cannot see a dimension-count landmine
    if c.get("leaky_private_path_grad", 0.0) <= 0.0:
        return Status.VOID          # the leak fixture is not leaking

    if m.get("arms_probed", 0.0) != float(len(CANDIDATE_ARMS)):
        return Status.VOID

    return bool(m.get("unison_admission_conjunction", 0.0) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LC.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())
