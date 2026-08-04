"""T0.03 — a checkpoint must restore the model exactly.

Everything downstream assumes this. A 12-hour Kaggle session that cannot reload
what it saved has produced nothing, and the failure is silent: the model loads,
the loss looks plausible, and the weights are wrong. `strict=False` on
`load_state_dict` — which this repo uses in several places — hides exactly that.

So the test compares FORWARD OUTPUTS after a round trip, not just tensor
equality: it catches a key mismatch that silently dropped a module, which
comparing only the keys that did load would miss entirely.

Control: a freshly initialised model must show a LARGE delta. If it does not,
the comparison is insensitive and a pass would mean nothing.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-6


def _build(seed: int):
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    for flag in ("llm_enabled", "enable_intrinsic_motivation"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    return UnifiedBrain(cfg), cfg


def _outputs(brain, obs):
    import torch
    brain.eval()
    with torch.no_grad():
        out = brain(obs)
    return {k: v for k, v in out.items()
            if hasattr(v, "dtype") and v.dtype.is_floating_point}


def _max_delta(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    if not shared:
        return float("inf")
    return max(float((a[k] - b[k]).abs().max()) for k in shared)


def _experiment(seed: int) -> dict:
    import torch
    brain, cfg = _build(seed)
    obs = torch.randn(2, cfg.obs_dim, generator=torch.Generator().manual_seed(seed))
    before = _outputs(brain, obs)

    with tempfile.TemporaryDirectory(dir="/data") as td:
        path = Path(td) / "ckpt.pt"
        torch.save({"model": brain.state_dict()}, path)
        size_mb = path.stat().st_size / 1e6

        restored, _ = _build(seed + 9999)  # different init on purpose
        ck = torch.load(path, map_location="cpu", weights_only=True)
        # strict=True deliberately: a silently dropped module is the failure
        # mode this test exists to catch.
        missing, unexpected = restored.load_state_dict(ck["model"], strict=True)

    after = _outputs(restored, obs)
    return {
        "output_delta": round(_max_delta(before, after), 12),
        "checkpoint_mb": round(size_mb, 1),
        "tensors_compared": len(set(before) & set(after)),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def _control(seed: int) -> dict:
    """A different random init must NOT match — proves the comparison is sensitive."""
    import torch
    a, cfg = _build(seed)
    b, _ = _build(seed + 1)
    obs = torch.randn(2, cfg.obs_dim, generator=torch.Generator().manual_seed(seed))
    return {"output_delta": round(_max_delta(_outputs(a, obs), _outputs(b, obs)), 12)}


def _check(m: dict, c: dict) -> bool:
    return (m["output_delta"] < TOL
            and m["missing_keys"] == 0 and m["unexpected_keys"] == 0
            and m["tensors_compared"] > 0
            and c["output_delta"] > TOL * 100)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.03"], _experiment, _check, control_fn=_control, ledger=ledger)
