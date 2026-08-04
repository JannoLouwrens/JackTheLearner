"""T0.11 — the same job runs on either backend, unmodified.

Failover is only useful if a job never learns where it landed. If a script needs
editing to move from Colab to Kaggle, then "failover" is a rewrite and the
promise is empty — you would discover that at the worst moment, when one backend
is down mid-training.

Three asymmetries make this non-obvious, and each could break it:
  - Colab runs synchronously and returns in ~30s; Kaggle pushes, queues, polls,
    and fetches output, taking minutes.
  - Artifact retrieval differs: Colab needs an ABSOLUTE remote path and an
    explicit download; Kaggle collects an output directory automatically.
  - The torch versions differ by necessity. Colab's T4 (sm_75) runs
    2.11.0+cu128; Kaggle's P100 (sm_60) must be given 2.5.1+cu121 because
    Kaggle's own build dropped Pascal. A job must tolerate both.

So the test submits ONE unmodified script, forces the preferred backend to fail,
and requires the fallback to produce the same artifact under the same key.

Control: with BOTH backends made impossible, submit() must report failure rather
than inventing success — the standard failure mode of a wrapper around someone
else's API.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .. import gpu

# Deliberately backend-agnostic: no /content, no Kaggle paths, no assumptions
# about the torch version beyond CUDA being usable.
JOB = textwrap.dedent('''
    import json, torch
    assert torch.cuda.is_available(), "no CUDA"
    a = torch.randn(256, 256, device="cuda")
    s = float((a @ a.T).sum())
    out = {"device": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "finite": s == s}
    print("JOB_JSON " + json.dumps(out))
    with open("job_result.json", "w") as f:
        json.dump(out, f)
''')


def _submit(prefer: str, colab_gpu: str) -> dict:
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "job.py"
        script.write_text(JOB)
        res = gpu.submit(script, prefer=prefer, gpu=colab_gpu,
                         timeout_s=1500, fetch=["job_result.json"])
    art = res.artifacts.get("job_result.json")
    payload = {}
    if art and Path(art).exists():
        payload = json.loads(Path(art).read_text())
    return {
        "backend": res.backend, "ok": res.ok,
        "artifact_key_present": bool(art),
        "device": payload.get("device", ""),
        "torch": payload.get("torch", ""),
        "finite": bool(payload.get("finite", False)),
        "message": res.message[:120],
    }


def _experiment(seed: int) -> dict:
    # Force Colab to refuse by asking for an accelerator that cannot exist.
    # The job text is IDENTICAL to what a Colab run would receive.
    r = _submit(prefer="colab", colab_gpu="NOT_A_REAL_GPU")
    return {
        "landed_on": r["backend"],
        "ok": r["ok"],
        "fell_back": r["backend"] == "kaggle",
        "artifact_returned": r["artifact_key_present"],
        "device": r["device"],
        "torch": r["torch"],
        "matmul_finite": r["finite"],
        "message": r["message"],
    }


def _control(seed: int) -> dict:
    """Both backends impossible — submit() must report failure, not success."""
    real_kaggle = gpu.run_on_kaggle
    gpu.run_on_kaggle = lambda *a, **k: gpu.JobResult("kaggle", False, message="forced failure")
    try:
        r = _submit(prefer="colab", colab_gpu="NOT_A_REAL_GPU")
    finally:
        gpu.run_on_kaggle = real_kaggle
    return {"ok": r["ok"], "message": r["message"]}


def _check(m: dict, c: dict) -> bool:
    return (m["ok"] and m["fell_back"] and m["artifact_returned"]
            and m["matmul_finite"] and not c["ok"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.11"], _experiment, _check, control_fn=_control, ledger=ledger)
