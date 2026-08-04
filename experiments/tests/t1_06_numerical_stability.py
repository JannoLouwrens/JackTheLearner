"""T1.06 — no NaN or Inf across a sustained run.

Short tests hide instability. A model can train cleanly for 50 steps and blow up
at 400 when Adam's second-moment estimate has warmed, or when a rare large input
meets an unnormalised layer. Discovering that eleven hours into a Kaggle session
is expensive; discovering it here costs CPU minutes.

Checked every step, on three surfaces, because they fail at different times:
  loss        NaN from a bad forward
  gradients   Inf from an exploding backward before the weights show it
  weights     the damage, once it has landed

Control: the same loop with a deliberately absurd learning rate MUST go
non-finite. If it does not, the detector is not watching the right tensors and a
clean pass would mean nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 400


def _loop(seed: int, lr: float) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).train()
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=lr)

    g = torch.Generator().manual_seed(seed + 11)
    obs = torch.randn(4, cfg.obs_dim, generator=g)
    tgt = torch.randn(4, cfg.action_chunk_size, cfg.action_dim, generator=g)

    bad_loss = bad_grad = bad_weight = 0
    first_bad = -1
    max_grad = 0.0
    for step in range(STEPS):
        loss = brain.action_training_loss(obs, tgt)["loss"]
        if not torch.isfinite(loss):
            bad_loss += 1
            if first_bad < 0:
                first_bad = step
        opt.zero_grad(); loss.backward()

        for p in brain.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    bad_grad += 1
                    if first_bad < 0:
                        first_bad = step
                    break
                max_grad = max(max_grad, float(p.grad.abs().max()))
        opt.step()

        if step % 50 == 0:
            for p in brain.parameters():
                if not torch.isfinite(p).all():
                    bad_weight += 1
                    if first_bad < 0:
                        first_bad = step
                    break

    return {"steps": STEPS, "nonfinite_loss": bad_loss, "nonfinite_grad": bad_grad,
            "nonfinite_weight": bad_weight, "first_bad_step": first_bad,
            "max_grad_abs": round(max_grad, 3), "final_loss": round(float(loss), 5)}


def _experiment(seed: int) -> dict:
    return _loop(seed, lr=3e-4)


def _control(seed: int) -> dict:
    """An absurd learning rate must break it — proof the detector works."""
    return _loop(seed, lr=1e4)


def _check(m: dict, c: dict) -> bool:
    clean = m["nonfinite_loss"] == 0 and m["nonfinite_grad"] == 0 and m["nonfinite_weight"] == 0
    control_broke = (c["nonfinite_loss"] + c["nonfinite_grad"] + c["nonfinite_weight"]) > 0
    return clean and control_broke


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.06"], _experiment, _check, control_fn=_control, ledger=ledger)
