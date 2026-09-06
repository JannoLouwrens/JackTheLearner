"""T1.11 — the module that drives the robot must be the module that learns.

The defect this exists for was invisible to every other check. The runtime path
(VirtualWorld -> act_dual_system -> generate_actions_flow_matching -> ActionExpert)
is decorated @torch.no_grad(); training went through a different module entirely
(action_head); and train_flow_matching_step — the only bridge between them — had
zero callers in the repo. The system could have trained to convergence with the
4.6M-parameter actuator module still at its random initialisation, and the loss
curve would have looked perfect throughout.

Method: discover which parameters the INFERENCE path uses by tracing the modules
the runtime actually calls, then check the TRAINING loss delivers gradient to
them. Not "does some loss exist" — does the loss the pipeline optimises reach the
thing that moves the joints.

Control: a loss touching only the unused head must FAIL. Without that, this would
pass for any loss that happened to be large enough.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['UnifiedBrain.py']

REPO = Path(__file__).resolve().parents[2]

# Modules on the live inference path. Sourced from the runtime trace, not guessed:
# VirtualWorld._update_brain -> act_with_mood -> act_dual_system ->
# generate_actions_flow_matching -> ActionExpert (+ backbone via cross-attention).
INFERENCE_MODULES = ("action_expert", "layers", "proprio_encoder")


def _build():
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    torch.manual_seed(0)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    return UnifiedBrain(cfg).train(), cfg


def _gradient_coverage(loss_fn) -> dict:
    import torch
    brain, cfg = _build()
    obs = torch.randn(4, cfg.obs_dim, generator=torch.Generator().manual_seed(1))
    tgt = torch.randn(4, cfg.action_chunk_size, cfg.action_dim,
                      generator=torch.Generator().manual_seed(2))

    brain.zero_grad(set_to_none=True)
    loss = loss_fn(brain, obs, tgt)
    loss.backward()

    reached, total = 0, 0
    per_module = {}
    for name, p in brain.named_parameters():
        top = name.split(".")[0]
        if top not in INFERENCE_MODULES or not p.requires_grad:
            continue
        total += p.numel()
        got = p.grad is not None and float(p.grad.abs().sum()) > 0
        if got:
            reached += p.numel()
        per_module[top] = per_module.get(top, 0) + (p.numel() if got else 0)

    return {
        "inference_params_total": total,
        "inference_params_trained": reached,
        "inference_params_trained_frac": round(reached / max(1, total), 4),
        "per_module": "; ".join(f"{k}={v:,}" for k, v in sorted(per_module.items())),
    }


def _experiment(seed: int) -> dict:
    return _gradient_coverage(
        lambda b, obs, tgt: b.action_training_loss(obs, tgt)["loss"])


def _control(seed: int) -> dict:
    """The old training loss: forward()['actions'] only. It must NOT cover the
    inference path — that is precisely the bug."""
    import torch
    return _gradient_coverage(
        lambda b, obs, tgt: torch.nn.functional.mse_loss(
            b(obs)["actions"].float(), tgt.float()))


def _check(m: dict, c: dict) -> bool:
    # Near-total coverage under the real loss, and the old loss must miss badly.
    return (m["inference_params_trained_frac"] >= 0.99
            and c["inference_params_trained_frac"] < 0.9)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.11"], _experiment, _check, control_fn=_control, ledger=ledger)
