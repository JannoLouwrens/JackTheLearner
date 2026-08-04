"""T1.04 — every trainable module's weights must actually change.

Distinct from T1.03, which asks whether gradient ARRIVES. A parameter can receive
a gradient and still not move: a zero learning rate for its group, a scheduler
that never warms up, weight decay exactly cancelling the update, or an optimiser
built over a stale parameter list that omits modules added later. Each of those
leaves T1.03 green and the module frozen in practice.

The optimiser-list failure is the realistic one here. Training code repeatedly
does `Adam(model.parameters())` at construction; anything attached afterwards, or
any module swapped during setup, silently never updates.

Control: the same steps with lr=0 must move NOTHING. Without it, floating-point
noise alone could pass this.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 20


def _deltas(seed: int, lr: float) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).train()

    before = {n: p.detach().clone() for n, p in brain.named_parameters() if p.requires_grad}
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=lr)

    g = torch.Generator().manual_seed(seed + 3)
    obs = torch.randn(4, cfg.obs_dim, generator=g)
    tgt = torch.randn(4, cfg.action_chunk_size, cfg.action_dim, generator=g)
    for _ in range(STEPS):
        loss = brain.action_training_loss(obs, tgt)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    # Per top-level module, so a single frozen submodule is named rather than
    # averaged away by the millions of parameters that did move.
    moved, stuck = {}, {}
    for n, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        top = n.split(".")[0]
        d = float((p.detach() - before[n]).abs().max())
        if d > 0:
            moved[top] = moved.get(top, 0) + p.numel()
        else:
            stuck[top] = stuck.get(top, 0) + p.numel()

    total = sum(moved.values()) + sum(stuck.values())
    return {
        "params_moved": sum(moved.values()),
        "params_stuck": sum(stuck.values()),
        "moved_frac": round(sum(moved.values()) / max(1, total), 4),
        "stuck_modules": "; ".join(f"{k}={v:,}" for k, v in sorted(stuck.items())) or "none",
    }


def _experiment(seed: int) -> dict:
    return _deltas(seed, lr=3e-4)


def _control(seed: int) -> dict:
    """lr=0 — nothing may move. Proves the measurement is not reading noise."""
    return _deltas(seed, lr=0.0)


def _check(m: dict, c: dict) -> bool:
    # Everything the gradient reaches must move (T1.03 already caps orphans at
    # 5%), and the zero-lr control must move nothing at all.
    return m["moved_frac"] >= 0.95 and c["params_moved"] == 0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.04"], _experiment, _check, control_fn=_control, ledger=ledger)
