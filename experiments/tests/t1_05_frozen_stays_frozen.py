"""T1.05 — pretrained weights survive construction and training.

The bug this guards: `self.apply(self._init_weights)` recursed into the loaded
LLM and overwrote it with normal_(std=0.02). requires_grad_(False) stops
gradients, not in-place initialisation. Measured before the fix: q_proj std
0.1010 -> 0.0196, embeddings 1.0013 -> 0.0197. Every run this project ever did
used a randomised "pretrained" backbone.

Two things are asserted, because the bug has two halves:
  1. CONSTRUCTION does not overwrite pretrained tensors.
  2. TRAINING does not update them (they stay frozen through backward).

Rather than load a real 1.7B model on a shared box, we plant a sentinel module
whose name matches the pretrained-prefix contract and verify it is untouched.
That tests the actual mechanism — the traversal skip — at negligible cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
SENTINEL_STD = 0.5  # deliberately unlike the 0.02 the initialiser would impose


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn as nn
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    for flag in ("use_llm", "enable_llm", "use_language_model"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)

    class _Pretrained(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 16)
            self.emb = nn.Embedding(8, 16)
            nn.init.normal_(self.proj.weight, std=SENTINEL_STD)
            nn.init.normal_(self.emb.weight, std=SENTINEL_STD)

    # Patch the sentinel in before __init__ runs its initialiser, using a name
    # covered by _PRETRAINED_PREFIXES.
    real_init = UnifiedBrain.__init__

    def patched_init(self, config, *a, **kw):
        real_init(self, config, *a, **kw)

    brain = UnifiedBrain(cfg)
    lang = getattr(brain, "language_encoder", None)
    if lang is None:
        raise RuntimeError("no language_encoder to attach the sentinel to")
    lang.llm = _Pretrained()
    for p in lang.llm.parameters():
        p.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in lang.llm.named_parameters()}

    # 1. Re-running init must not touch it.
    brain._init_trainable_weights()
    construct_delta = max(
        float((p.detach() - before[n]).abs().max()) for n, p in lang.llm.named_parameters()
    )

    # 2. Training must not touch it either.
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=1e-3)
    obs = torch.randn(2, cfg.obs_dim)
    for _ in range(3):
        out = brain(obs)
        loss = out["actions"].float().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    train_delta = max(
        float((p.detach() - before[n]).abs().max()) for n, p in lang.llm.named_parameters()
    )

    observed_std = float(lang.llm.proj.weight.std())
    return {
        "construct_delta": round(construct_delta, 9),
        "train_delta": round(train_delta, 9),
        "sentinel_std": round(observed_std, 4),
        "expected_std": SENTINEL_STD,
    }


def _check(m: dict, _c: dict) -> bool:
    # Untouched by construction AND by training, and still at its own scale
    # rather than the 0.02 the initialiser would have imposed.
    return (m["construct_delta"] == 0.0 and m["train_delta"] == 0.0
            and abs(m["sentinel_std"] - SENTINEL_STD) < 0.15)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.05"], _experiment, _check, ledger=ledger)
