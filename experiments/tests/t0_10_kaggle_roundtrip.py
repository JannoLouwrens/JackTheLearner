"""T0.10 — a job reaches a real Kaggle GPU and its results come back.

Same contract as T0.09, different executor. Kaggle has no ephemeral-run
primitive: a kernel is pushed, queued, polled, and its output collected
afterwards, so the wrapper has more moving parts and more ways to lie.

It has already lied twice, which is why each property is checked separately:
  - before phone verification, kernels ran to COMPLETE on CPU with torch+cpu
    installed and nothing anywhere signalling a problem;
  - after verification, a real P100 was attached but Kaggle's own preinstalled
    torch 2.10+cu128 ships sm_70+ kernels only, so every CUDA op raised "no
    kernel image is available" — the GPU was real and unusable.

So this asserts a device IS present, CUDA IS usable, and a kernel ACTUALLY RAN,
not merely that the job reported success.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import run_on_kaggle

PROBE = textwrap.dedent('''
    import json, torch
    out = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        a = torch.randn(512, 512, device="cuda")
        s = float((a @ a.T).sum())
        out["matmul_finite"] = (s == s)
    print("PROBE_JSON " + json.dumps(out))
    with open("probe_result.json", "w") as f:
        json.dump(out, f)
''')


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "probe.py"
        script.write_text(PROBE)
        res = run_on_kaggle(script, timeout_s=1500, fetch=["probe_result.json"])

    payload = {}
    art = res.artifacts.get("probe_result.json")
    if art and Path(art).exists():
        payload = json.loads(Path(art).read_text())

    return {
        "ok": res.ok,
        "cuda_available": bool(payload.get("cuda_available", False)),
        "device": payload.get("device", ""),
        "capability": ".".join(str(x) for x in payload.get("capability", [])),
        "matmul_finite": bool(payload.get("matmul_finite", False)),
        "torch": payload.get("torch", ""),
        "artifact_bytes": Path(art).stat().st_size if art and Path(art).exists() else 0,
        "duration_s": round(res.duration_s, 1),
        "message": res.message[:120],
    }


def _check(m: dict, _c: dict) -> bool:
    # Every property separately — each has already failed independently once.
    return (m["ok"] and m["cuda_available"] and m["matmul_finite"]
            and m["device"] != "" and m["artifact_bytes"] > 0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.10"], _experiment, _check, ledger=ledger)
