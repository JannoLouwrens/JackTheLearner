"""T0.09 — a job reaches a real GPU and its results come back.

The whole of Tier 2 rests on this. If a script cannot leave this box, run on a
GPU, and return an artifact, then no training result can ever be produced —
regardless of how good the model is.

What must be true, each independently checkable:
  - a real NVIDIA GPU is present (not a CPU runtime silently substituted)
  - torch sees CUDA and can compute on it
  - a file written on the VM comes back to this box intact
  - the VM is released afterwards, so a forgotten session cannot burn quota

Control: the same submission with an impossible accelerator must FAIL. Without
it, a pass could mean "we always report success", which is the failure mode of
every wrapper around someone else's API.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import run_on_colab

PROBE = textwrap.dedent('''
    import json, subprocess, torch
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    # Compute something real — a CUDA context that never runs a kernel proves nothing.
    a = torch.randn(512, 512, device="cuda")
    b = (a @ a.T).sum().item()
    out = {
        "gpu": smi,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0),
        "matmul_finite": bool(b == b),
        "torch": torch.__version__,
    }
    print("PROBE_JSON " + json.dumps(out))
    with open("probe_result.json", "w") as f:
        json.dump(out, f)
''')


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "probe.py"
        script.write_text(PROBE)
        res = run_on_colab(script, gpu="T4", timeout_s=900,
                           fetch=["probe_result.json"])

    payload = {}
    for line in res.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            payload = json.loads(line[len("PROBE_JSON "):])
            break

    artifact = res.artifacts.get("probe_result.json")
    artifact_bytes = Path(artifact).stat().st_size if artifact and Path(artifact).exists() else 0

    return {
        "ok": res.ok,
        "gpu": payload.get("gpu", res.gpu_name or "unknown"),
        "cuda_available": bool(payload.get("cuda_available", False)),
        "device": payload.get("device", ""),
        "matmul_finite": bool(payload.get("matmul_finite", False)),
        "artifact_bytes": artifact_bytes,
        "duration_s": round(res.duration_s, 1),
        "message": res.message[:160],
    }


def _control(seed: int) -> dict:
    """An impossible accelerator must not report success."""
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "probe.py"
        script.write_text("print('should never run')")
        res = run_on_colab(script, gpu="NOT_A_REAL_GPU", timeout_s=240)
    return {"ok": res.ok, "message": res.message[:160]}


def _check(m: dict, c: dict) -> bool:
    # The GPU-name disjunction MUST be parenthesised. Without the inner
    # brackets `and` binds tighter than `or`, so the whole gate collapses to
    # `(... and "NVIDIA" in gpu) or ("TESLA" in gpu)` — and Colab's device
    # string is literally "Tesla T4", so the right branch was true on every
    # real run and ok/cuda_available/matmul_finite were never consulted.
    # Found by the 2026-08-09 overseer audit; T0.13 now gates the whole class.
    is_nvidia = "NVIDIA" in m["gpu"].upper() or "TESLA" in m["gpu"].upper()
    return (m["ok"] and m["cuda_available"] and m["matmul_finite"]
            and is_nvidia and m["artifact_bytes"] > 0 and not c["ok"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.09"], _experiment, _check, control_fn=_control, ledger=ledger)
