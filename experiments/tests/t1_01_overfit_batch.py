"""T1.01 — real training, real improvement, on the real model.

The single most informative test in machine learning: take ONE fixed batch and
train until the model memorises it. A network that cannot drive loss to ~0 on a
batch it sees 500 times will never learn a task, and every downstream result from
it is noise. This catches a wrong loss, a detached graph, a shape bug, dead
activations and a bad learning rate in one run.

Crucially it carries a CONTROL: the identical loop with requires_grad=False must
NOT improve. If the frozen model's loss also falls, the metric is measuring
something other than learning and no result built on it can be trusted.

This runs on CPU in minutes. GPU quota is spent only once this passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

STEPS = 400
TARGET_LOSS = 1e-2
BATCH = 8


def _train(seed: int, frozen: bool) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    # Keep the frozen 1.7B LLM out of this test — it is not what we are probing,
    # and loading it makes a CPU sanity check cost 7 GB.
    for flag in ("use_llm", "enable_llm", "use_language_model"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    brain = UnifiedBrain(cfg)

    if frozen:
        for p in brain.parameters():
            p.requires_grad_(False)

    params = [p for p in brain.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4) if params else None

    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(BATCH, cfg.obs_dim, generator=g)
    # The action head emits a CHUNK: [B, chunk_len, action_dim], not [B, action_dim].
    # (Binding review: chunk is 16, though README prose claims 48.)
    with torch.no_grad():
        probe = brain(obs)["actions"]
    target = torch.randn(*probe.shape, generator=g)

    curve = []
    for step in range(STEPS):
        out = brain(obs)
        action = out["actions"] if isinstance(out, dict) and "actions" in out else None
        if action is None:
            raise RuntimeError(f"forward() returned no actions key: {list(out)[:8]}")
        loss = torch.nn.functional.mse_loss(action.float(), target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 40 == 0 or step == STEPS - 1:
            curve.append(round(loss.item(), 5))

    return {
        "final_loss": round(curve[-1], 6),
        "initial_loss": round(curve[0], 6),
        "improvement_ratio": round(curve[0] / max(curve[-1], 1e-9), 2),
        "curve": ";".join(str(c) for c in curve),
        "trainable_tensors": len(params),
    }


def _experiment(seed: int) -> dict:
    return _train(seed, frozen=False)


def _control(seed: int) -> dict:
    return _train(seed, frozen=True)


def _check(m: dict, c: dict) -> bool:
    learned = m["final_loss"] < TARGET_LOSS
    control_did_not_learn = c["final_loss"] >= TARGET_LOSS
    return learned and control_did_not_learn


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.01"], _experiment, _check, control_fn=_control, ledger=ledger)
