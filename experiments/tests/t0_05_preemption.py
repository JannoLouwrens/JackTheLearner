"""T0.05 — a SIGKILL mid-write must not corrupt the checkpoint.

Ephemeral GPUs die without warning: Colab reclaims the VM, Kaggle caps the
session at 12 hours, a spot instance vanishes. The dangerous failure is not
losing progress — it is a checkpoint that was half-written when the process died,
because the next session loads it, gets a truncated or corrupt file, and either
crashes hours later or silently trains from garbage.

Method: write checkpoints in a loop in a child process, SIGKILL it at an
unpredictable moment, then verify the newest checkpoint on disk still loads and
still contains what it claims. Repeat, because the interesting kill lands DURING
a write and hitting that window takes several attempts.

Control: the same loop writing non-atomically (torch.save straight to the final
path). It must produce a corrupt file, or the test cannot tell a safe writer from
an unsafe one.

RNG state is included in the payload here — T0.04 showed a freshly constructed
model draws different stochastic values, so exact resumption needs it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"
ATTEMPTS = 12

WRITER = textwrap.dedent("""
    import os, sys, time, torch
    outdir, atomic = sys.argv[1], sys.argv[2] == "1"
    payload = {
        "step": 0,
        "model": {f"w{i}": torch.randn(256, 256) for i in range(8)},   # ~2 MB, wide enough to be interrupted
        "optim": {f"m{i}": torch.randn(256, 256) for i in range(8)},
        "rng": torch.get_rng_state(),
    }
    final = os.path.join(outdir, "ckpt.pt")
    step = 0
    while True:
        step += 1
        payload["step"] = step
        payload["rng"] = torch.get_rng_state()
        if atomic:
            tmp = final + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, final)      # atomic on POSIX: never a partial final file
        else:
            torch.save(payload, final)  # the naive version
        time.sleep(0.01)
""")


def _hammer(atomic: bool) -> dict:
    """Kill a writer repeatedly and see whether the checkpoint survives."""
    import torch

    corrupt, checked, steps_seen = 0, 0, []
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "writer.py"
        script.write_text(WRITER)
        for attempt in range(ATTEMPTS):
            proc = subprocess.Popen(
                [PY, str(script), td, "1" if atomic else "0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            # Wait for the writer to be genuinely writing before killing it.
            # Guessing at a sleep killed it during `import torch` on this ARM box
            # and the test measured nothing (checkpoints_checked = 0). Poll for a
            # real checkpoint, then kill at a drifting offset so SIGKILL lands at
            # different points inside a write across attempts.
            import time
            ck_path = Path(td) / "ckpt.pt"
            deadline = time.time() + 60
            while not ck_path.exists() and time.time() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            time.sleep(0.013 + (attempt % 7) * 0.011)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

            ck = Path(td) / "ckpt.pt"
            if not ck.exists():
                continue
            checked += 1
            try:
                loaded = torch.load(ck, map_location="cpu", weights_only=False)
                # Loading is not enough — assert the payload is complete.
                assert set(loaded) == {"step", "model", "optim", "rng"}
                assert len(loaded["model"]) == 8 and len(loaded["optim"]) == 8
                assert loaded["rng"].numel() > 0
                steps_seen.append(int(loaded["step"]))
            except Exception:
                corrupt += 1
    return {
        "kills": ATTEMPTS,
        "checkpoints_checked": checked,
        "corrupt": corrupt,
        "max_step_recovered": max(steps_seen) if steps_seen else 0,
    }


def _experiment(seed: int) -> dict:
    return _hammer(atomic=True)


def _control(seed: int) -> dict:
    return _hammer(atomic=False)


def _check(m: dict, c: dict) -> bool:
    # Atomic writer: never corrupt, and it actually got exercised.
    # Control: must corrupt at least once, or the hammer is not hitting writes
    # and a pass would be vacuous.
    return (m["corrupt"] == 0 and m["checkpoints_checked"] >= 5
            and c["corrupt"] >= 1)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.05"], _experiment, _check, control_fn=_control, ledger=ledger)
