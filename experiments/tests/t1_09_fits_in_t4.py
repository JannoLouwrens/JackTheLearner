"""T1.09 — the training step must fit a 16 GB P100 at the intended batch size.

Every GPU backend we can reach for free tops out at a T4 (Colab) or P100
(Kaggle), both 16 GB. If a training step OOMs there, every Tier 2 plan is
fiction regardless of how good the specs are — so this is measured before any
long run is scheduled, not discovered eleven hours into one.

RE-AIMED AT THE P100 (Review 2026-08-31 item 5, executed 2026-09-01): the spec
and this docstring named a Colab T4 the project has not certified on since
08-12, while the attempt-1 PASS itself already recorded a Kaggle P100 — submit
fell back. Every long run in this project now lands on the P100, so the aim
(`prefer="kaggle"`) and the claim text now name the device the certificate is
actually about. The ceilings are IDENTICAL (bar 12 GB of 16, absurd batch
1024): correct device, not a weakening.

Measured, not asserted: torch.cuda.max_memory_allocated() across a full
forward+backward+step at the batch size Tier 2 intends (64, the same the T1.07/
T1.08 arms use), plus a probe at 2x that to learn where the ceiling actually is.
The margin matters as much as the pass: fragmentation, the sampler's working set
and CUDA context overhead all land on top of what this measures, so the bar is
12 GB, not 15.

CONTROL: a deliberately absurd batch (1024) must either OOM or exceed the bar.
If the measurement cannot detect a batch 16x larger, it is not measuring
allocation, and a pass would mean nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

INTENDED_BATCH = 64
MAX_GB = 12.0            # of 15 usable — headroom for fragmentation + context
ABSURD_BATCH = 1024      # control: must OOM or blow past the bar

JOB = r'''
import json, torch
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

DEV = "cuda"
assert torch.cuda.is_available(), "this spec is meaningless on CPU"

def peak_gb(batch):
    """Peak allocated GB across one full optimiser step at this batch size."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(0)
        cfg = UnifiedBrainConfig()
        cfg.llm_enabled = False
        cfg.enable_intrinsic_motivation = False
        brain = UnifiedBrain(cfg).to(DEV).train()
        opt, step_fn = brain.make_action_optimizer(lr=3e-4)
        g = torch.Generator().manual_seed(1)
        obs = torch.randn(batch, cfg.obs_dim, generator=g).to(DEV)
        tgt = torch.randn(batch, cfg.action_chunk_size, cfg.action_dim,
                          generator=g).to(DEV)
        # Two steps: the SECOND is the honest one — Adam's moment buffers only
        # exist after the first step, and they are 2x the parameter memory.
        for _ in range(2):
            loss = brain.action_training_loss(obs, tgt)["loss"]
            opt.zero_grad(); loss.backward(); step_fn()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        del brain, opt, obs, tgt
        torch.cuda.empty_cache()
        return {"batch": batch, "peak_gb": round(peak, 3), "oom": False}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"batch": batch, "peak_gb": float("inf"), "oom": True}

out = {"gpu": torch.cuda.get_device_name(0),
       "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
       "intended": peak_gb(__BATCH__),
       "double": peak_gb(__BATCH__ * 2),
       "absurd": peak_gb(__ABSURD__)}
import os as _o
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t109.json"), "w"), indent=1)
print("DONE", json.dumps(out), flush=True)
'''


def _submit() -> dict:
    job = build_job(JOB.replace("__BATCH__", repr(INTENDED_BATCH))
                       .replace("__ABSURD__", repr(ABSURD_BATCH)))
    res = submit(job, prefer="kaggle", est_hours=0.2, timeout_s=1800,
                 fetch=["t109.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t109.json")
    if not path:
        raise RuntimeError(
            f"no artifact from {res.backend}. message={res.message!r} "
            f"stdout_tail={res.stdout[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    return d


_CACHE: dict = {}


def _experiment(seed: int) -> dict:
    _CACHE.update(_submit())
    i, d = _CACHE["intended"], _CACHE["double"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "vram_total_gb": _CACHE["total_gb"],
        "peak_gb_at_batch_64": i["peak_gb"],
        "oom_at_batch_64": i["oom"],
        "peak_gb_at_batch_128": d["peak_gb"] if not d["oom"] else -1,
        "headroom_gb": round(MAX_GB - i["peak_gb"], 3) if not i["oom"] else -1,
    }


def _control(seed: int) -> dict:
    a = _CACHE["absurd"]
    return {"absurd_batch": a["batch"], "absurd_oom": a["oom"],
            "absurd_peak_gb": a["peak_gb"] if not a["oom"] else -1}


def _check(m: dict, c: dict) -> bool:
    fits = (not m["oom_at_batch_64"]) and m["peak_gb_at_batch_64"] <= MAX_GB
    # The control must show the measurement can detect excess: either OOM, or a
    # peak beyond the bar.
    detects = c["absurd_oom"] or (c["absurd_peak_gb"] > MAX_GB)
    return fits and detects


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.09"], _experiment, _check, control_fn=_control, ledger=ledger)
