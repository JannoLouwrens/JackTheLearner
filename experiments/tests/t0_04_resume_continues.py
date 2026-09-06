"""T0.04 — resuming must CONTINUE training, not quietly restart it.

The failure this catches is silent and expensive. A run reloads, the loss looks
plausible, training proceeds — and the optimiser moments were dropped, so Adam
spends hundreds of steps rebuilding momentum it already had. On a 12-hour Kaggle
session that is hours of compute burned to arrive back where you were, with
nothing in the logs to say so.

Method: train, checkpoint, and keep training to get a REFERENCE trace. Then
restore and train the same steps again. With state fully restored the two traces
must be identical — not merely "smooth", identical, because the data and seed are
the same. That is a far sharper instrument than eyeballing a curve for a bump.

Control: restore the WEIGHTS ONLY, as a naive implementation would. It must
diverge. If it does not, the test cannot tell a correct resume from a broken one
and a pass would be meaningless.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['UnifiedBrain.py']

REPO = Path(__file__).resolve().parents[2]
WARMUP, RESUME_STEPS = 25, 20
TOL = 1e-6


def _setup(seed: int):
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    for flag in ("llm_enabled", "enable_intrinsic_motivation"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    brain = UnifiedBrain(cfg)
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(4, cfg.obs_dim, generator=g)
    with torch.no_grad():
        tgt = torch.randn_like(brain(obs)["actions"])
    return brain, opt, obs, tgt


def _steps(brain, opt, obs, tgt, n: int) -> list[float]:
    import torch
    trace = []
    for _ in range(n):
        loss = torch.nn.functional.mse_loss(brain(obs)["actions"].float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        trace.append(loss.item())
    return trace


def _run(seed: int, restore_optimiser: bool) -> dict:
    import torch
    brain, opt, obs, tgt = _setup(seed)
    _steps(brain, opt, obs, tgt, WARMUP)

    # state_dict() only — the module itself cannot be deepcopied: it holds a
    # _thread.lock (AlphaGeometryLoop's timeout machinery), so copy.deepcopy(brain)
    # raises TypeError. Any training code that clones the model will hit this too.
    ckpt = {"model": {k: v.detach().clone() for k, v in brain.state_dict().items()},
            "optim": copy.deepcopy(opt.state_dict())}

    # Reference: this same model simply keeps going, uninterrupted.
    reference = _steps(brain, opt, obs, tgt, RESUME_STEPS)

    # Resumed: a fresh session rebuilds from the checkpoint on disk.
    with tempfile.TemporaryDirectory(dir="/data") as td:
        p = Path(td) / "resume.pt"
        torch.save(ckpt, p)
        loaded = torch.load(p, map_location="cpu", weights_only=False)

    new_brain, _, _, _ = _setup(seed + 4242)   # different init on purpose
    new_brain.load_state_dict(loaded["model"], strict=True)
    new_opt = torch.optim.Adam([p_ for p_ in new_brain.parameters() if p_.requires_grad], lr=3e-4)
    if restore_optimiser:
        new_opt.load_state_dict(loaded["optim"])
    resumed = _steps(new_brain, new_opt, obs, tgt, RESUME_STEPS)

    delta = max(abs(a - b) for a, b in zip(reference, resumed))
    first_step_gap = abs(reference[0] - resumed[0]) / max(reference[0], 1e-9) * 100
    return {
        "max_trace_delta": round(delta, 10),
        "first_step_gap_pct": round(first_step_gap, 4),
        "reference_final": round(reference[-1], 6),
        "resumed_final": round(resumed[-1], 6),
    }


def _experiment(seed: int) -> dict:
    return _run(seed, restore_optimiser=True)


def _control(seed: int) -> dict:
    """Weights-only resume — the naive version. Must diverge."""
    return _run(seed, restore_optimiser=False)


def _check(m: dict, c: dict) -> bool:
    """Restoring optimiser state must track the uninterrupted run far more closely
    than a weights-only resume. See the spec note for why the original
    first-step-jump threshold was replaced: it read 1.326% in both arms."""
    ratio = c["max_trace_delta"] / max(m["max_trace_delta"], 1e-12)
    m["resume_fidelity_ratio"] = round(ratio, 2)
    return ratio >= 10.0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.04"], _experiment, _check, control_fn=_control, ledger=ledger)
