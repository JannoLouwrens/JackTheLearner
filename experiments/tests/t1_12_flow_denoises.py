"""T1.12 — the sampler the robot runs must actually reconstruct actions.

T1.11 proved gradient reaches ActionExpert. That is plumbing, not learning, and
the distinction matters more here than almost anywhere else in the ladder:
flow matching trains a velocity field at random t, but INFERENCE integrates ten
Euler steps from pure noise. A mathematically correct loss can drive its training
objective down while the integrated trajectory still lands nowhere near the
target — wrong step size, wrong time conditioning, a sign error in the velocity —
and none of that shows up in the loss curve.

So this measures the thing the robot actually executes:
`generate_actions_flow_matching`, the @torch.no_grad() sampler called by
act_dual_system.

Two controls, because there are two distinct ways to be fooled:
  - UNTRAINED sampler: the same integration on a fresh model. If trained is no
    better, training changed nothing that the sampler uses.
  - SHUFFLED conditioning: the same trained model asked to reconstruct targets
    paired with the WRONG states. If that reconstructs equally well, the sampler
    is ignoring its conditioning and has merely learned the average action.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 250
N = 4


def _build(seed: int):
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    return UnifiedBrain(cfg), cfg


def _sample_error(brain, obs, target) -> float:
    """Error of the SAMPLER — what the runtime executes — not of the loss."""
    import torch
    brain.eval()
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(obs)
    brain.train()
    return float(torch.nn.functional.mse_loss(pred.float(), target.float()))


def _experiment(seed: int) -> dict:
    import torch
    brain, cfg = _build(seed)
    g = torch.Generator().manual_seed(seed + 100)
    obs = torch.randn(N, cfg.obs_dim, generator=g)
    target = torch.randn(N, cfg.action_chunk_size, cfg.action_dim, generator=g)

    before = _sample_error(brain, obs, target)

    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(STEPS):
        loss = brain.action_training_loss(obs, target)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    after = _sample_error(brain, obs, target)

    # Shuffled conditioning: same model, targets paired with the wrong states.
    shuffled_target = target[torch.randperm(N, generator=g)]
    shuffled = _sample_error(brain, obs, shuffled_target)

    return {
        "sampler_error_before": round(before, 5),
        "sampler_error_after": round(after, 5),
        "reconstruction_improvement": round(before / max(after, 1e-9), 3),
        "shuffled_conditioning_error": round(shuffled, 5),
        "conditioning_ratio": round(shuffled / max(after, 1e-9), 3),
        "final_train_loss": round(float(loss), 5),
    }


def _control(seed: int) -> dict:
    """An untrained sampler on the same data — the null this must beat."""
    import torch
    brain, cfg = _build(seed + 7777)
    g = torch.Generator().manual_seed(seed + 100)
    obs = torch.randn(N, cfg.obs_dim, generator=g)
    target = torch.randn(N, cfg.action_chunk_size, cfg.action_dim, generator=g)
    return {"sampler_error_untrained": round(_sample_error(brain, obs, target), 5)}


def _check(m: dict, c: dict) -> bool:
    # 1. Training must measurably improve the SAMPLER, not just the loss.
    # 2. It must beat an untrained sampler.
    # 3. It must actually use its conditioning — shuffled targets must be worse.
    return (m["reconstruction_improvement"] >= 2.0
            and m["sampler_error_after"] < c["sampler_error_untrained"]
            and m["conditioning_ratio"] >= 1.5)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.12"], _experiment, _check, control_fn=_control, ledger=ledger)
