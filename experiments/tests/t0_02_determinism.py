"""T0.02 — same seed reproduces the same loss trace; different seeds do not.

Both halves matter. If seeds do not reproduce, no A/B comparison in the ladder
means anything. If different seeds produce IDENTICAL traces, the seed is being
ignored and the 3-seed variance requirement is silently vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]


def _trace(seed: int, steps: int = 30) -> list[float]:
    sys.path.insert(0, str(REPO))
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 8))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 32, generator=g)
    y = torch.randn(16, 8, generator=g)
    out = []
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        out.append(loss.item())
    return out


def _experiment(seed: int) -> dict:
    a, b = _trace(0), _trace(0)
    delta = max(abs(x - y) for x, y in zip(a, b))
    return {"max_abs_trace_delta": delta, "steps": len(a), "final_loss": a[-1]}


def _control(seed: int) -> dict:
    """Different seeds MUST diverge."""
    a, b = _trace(0), _trace(1)
    return {"max_abs_trace_delta": max(abs(x - y) for x, y in zip(a, b))}


def _check(m: dict, c: dict) -> bool:
    return m["max_abs_trace_delta"] < 1e-9 and c["max_abs_trace_delta"] > 1e-6


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.02"], _experiment, _check, control_fn=_control, ledger=ledger)
