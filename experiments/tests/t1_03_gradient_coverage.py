"""T1.03 — every trainable parameter must receive gradient.

This is the test the repo needed and never had. The pipeline review measured
45,538,295 parameters (38.6% of 117,888,028) with no live call site: a
hierarchical_planner larger than the backbone it sits on, a temporal_memory never
passed `memory=`, a world_model gated on an argument nothing supplies.

None of that is visible from a loss curve. It is visible in one backward pass.

The test reports the orphan fraction and names the worst offenders, so the
remedy is unambiguous: wire it, or delete it.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# Anything above this is a structural problem, not an oversight.
MAX_ORPHAN_FRACTION = 0.05


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    brain = UnifiedBrain(cfg)
    brain.train()

    total = sum(p.numel() for p in brain.parameters())
    trainable = sum(p.numel() for p in brain.parameters() if p.requires_grad)

    # A representative forward: proprioception only, which is what the live
    # runtime path actually supplies.
    obs = torch.randn(2, cfg.obs_dim)
    out = brain(obs)
    loss = sum(v.float().pow(2).mean() for v in out.values()
               if torch.is_tensor(v) and v.dtype.is_floating_point)
    loss.backward()

    orphan_params, orphan_by_module = 0, {}
    for name, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            orphan_params += p.numel()
            top = name.split(".")[0]
            orphan_by_module[top] = orphan_by_module.get(top, 0) + p.numel()

    worst = sorted(orphan_by_module.items(), key=lambda kv: -kv[1])[:6]
    return {
        "total_params": total,
        "trainable_params": trainable,
        "params_without_grad": orphan_params,
        "orphan_fraction": round(orphan_params / max(1, trainable), 4),
        "worst_offenders": "; ".join(f"{k}={v:,}" for k, v in worst) or "none",
    }


def _check(m: dict, _c: dict) -> bool:
    return m["orphan_fraction"] <= MAX_ORPHAN_FRACTION


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.03"], _experiment, _check, ledger=ledger)
