"""T1.10 — the model must compute the same function on CPU and GPU.

The ladder's evidence is split across two machines: Tiers 0-1 baselines were
measured on this box's ARM CPU, Tier 2 onward trains on ephemeral x86+CUDA VMs.
Every cross-tier comparison silently assumes both compute the same function. That
assumption has real ways to fail: a custom layer with device-dependent branching,
fp32 accumulation-order drift large enough to matter, a buffer left on the wrong
device, or a kernel bug like Kaggle's sm_60/torch mismatch (T0.10) — which
produced WRONG results, not crashes, until pinned.

So: construct the identical model (same seed) here and on the GPU, feed the
IDENTICAL input tensor, compare forward outputs elementwise.

Two traps this design dodges, learned the hard way:
  - torch.randn on CUDA and CPU produce DIFFERENT sequences from the same seed,
    so all tensors are generated on CPU and shipped/re-generated identically;
    the stochastic flow sampler is not compared, only the deterministic forward.
  - weights must come from the same draw: the model is built with
    torch.manual_seed on CPU in both places and only THEN moved to CUDA.

Tolerance is 2e-3 max-abs: fp32 across ARM/x86/CUDA legitimately drifts at 1e-5
or so through 58M params of accumulation; 2e-3 catches anything structural while
ignoring rounding. CONTROL: a model built from a different seed must disagree by
far more than the tolerance — proving the comparison could have failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

MODEL_SEED = 0
INPUT_SEED = 1
CONTROL_SEED = 99
TOL = 2e-3
BATCH = 4

JOB = r'''
import json, torch
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

def outputs(model_seed):
    torch.manual_seed(model_seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg)                      # built on CPU: same weight draw
    brain = brain.to("cuda").eval()
    g = torch.Generator().manual_seed(__INPUT_SEED__)   # CPU generator: same input
    obs = torch.randn(__BATCH__, cfg.obs_dim, generator=g).to("cuda")
    with torch.no_grad():
        out = brain.forward(obs)
    return out["actions"].detach().cpu().flatten().tolist()

res = {"gpu": torch.cuda.get_device_name(0),
       "actions": outputs(__MODEL_SEED__)}
import os as _o
json.dump(res, open(_o.path.join(_o.environ["JACK_OUT"], "t110.json"), "w"))
print("DONE", len(res["actions"]), "values", flush=True)
'''


def _local_outputs(model_seed: int) -> list:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(model_seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).eval()
    g = torch.Generator().manual_seed(INPUT_SEED)
    obs = torch.randn(BATCH, cfg.obs_dim, generator=g)
    with torch.no_grad():
        out = brain.forward(obs)
    return out["actions"].detach().flatten().tolist()


_CACHE: dict = {}


def _experiment(seed: int) -> dict:
    body = (JOB.replace("__MODEL_SEED__", repr(MODEL_SEED))
               .replace("__INPUT_SEED__", repr(INPUT_SEED))
               .replace("__BATCH__", repr(BATCH)))
    job = build_job(body)
    res = submit(job, prefer="colab", est_hours=0.15, timeout_s=1500,
                 fetch=["t110.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t110.json")
    if not path:
        raise RuntimeError(f"no artifact from {res.backend}. message={res.message!r}")
    gpu = json.loads(Path(path).read_text())

    local = _local_outputs(MODEL_SEED)
    _CACHE["gpu_actions"] = gpu["actions"]
    if len(local) != len(gpu["actions"]):
        raise RuntimeError(f"shape mismatch: local {len(local)} vs gpu {len(gpu['actions'])}")

    diffs = [abs(a - b) for a, b in zip(local, gpu["actions"])]
    return {
        "gpu": gpu["gpu"], "backend": res.backend,
        "values_compared": len(diffs),
        "max_abs_diff": round(max(diffs), 8),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 9),
        "tolerance": TOL,
    }


def _control(seed: int) -> dict:
    """A different model seed must NOT agree — else the comparison sees nothing."""
    other = _local_outputs(CONTROL_SEED)
    gpu = _CACHE["gpu_actions"]
    diffs = [abs(a - b) for a, b in zip(other, gpu)]
    return {"wrong_seed_max_diff": round(max(diffs), 6)}


def _check(m: dict, c: dict) -> bool:
    return (m["max_abs_diff"] <= TOL
            and c["wrong_seed_max_diff"] > 10 * TOL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.10"], _experiment, _check, control_fn=_control, ledger=ledger)
