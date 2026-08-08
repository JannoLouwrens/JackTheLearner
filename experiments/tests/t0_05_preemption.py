"""T0.05 — a SIGKILL mid-write must not corrupt the checkpoint.

Ephemeral GPUs die without warning: Colab reclaims the VM, Kaggle caps the
session at 12 hours, a spot instance vanishes. The dangerous failure is not
losing progress — it is a checkpoint that was half-written when the process died,
because the next session loads it, gets a truncated or corrupt file, and either
crashes hours later or silently trains from garbage.

Method: write checkpoints in a loop in a child process, then kill it AT THE
DANGEROUS MOMENT — while a write is demonstrably in progress. The parent first
waits until one complete checkpoint exists (so there is something to lose),
then polls the in-progress file's size at sub-millisecond granularity and
delivers SIGKILL the instant it observes a partial file. Every attempt lands
inside a write by construction.

The v1 hammer killed at a drifting fixed offset and HOPED to hit a write.
Measured on 2026-08-08 after the gate caught it vacuous: torch.save buffers
and flushes in one short burst, so a random kill lands mid-write only ~8-19%
of the time — at 12 attempts the control had a ~35% chance of corrupting
nothing, which makes a pass meaningless. Same pre-registered thresholds here;
only the kill got surgical (targeted kills: control corrupts 12/12, atomic
survives 12/12, three standalone trials).

Control: the same loop writing non-atomically (torch.save straight to the final
path). Killed mid-write, it must produce a corrupt file, or the test cannot
tell a safe writer from an unsafe one.

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
import time
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
        "model": {f"w{i}": torch.randn(256, 256) for i in range(8)},   # ~4 MB, wide enough to be interrupted
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
        time.sleep(0.005)               # a stable complete-state window for the parent to observe
""")


def _ref_size() -> int:
    """Size of a complete checkpoint. Shapes are fixed, so this is stable."""
    import torch

    payload = {
        "step": 0,
        "model": {f"w{i}": torch.randn(256, 256) for i in range(8)},
        "optim": {f"m{i}": torch.randn(256, 256) for i in range(8)},
        "rng": torch.get_rng_state(),
    }
    with tempfile.NamedTemporaryFile(dir="/data", suffix=".pt") as f:
        torch.save(payload, f.name)
        return os.path.getsize(f.name)


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def _hammer(atomic: bool) -> dict:
    """Kill a writer mid-write, repeatedly, and see whether the checkpoint survives."""
    import torch

    ref = _ref_size()
    corrupt, checked, steps_seen = 0, 0, []
    with tempfile.TemporaryDirectory(dir="/data") as td:
        script = Path(td) / "writer.py"
        script.write_text(WRITER)
        final = os.path.join(td, "ckpt.pt")
        # The file that is mid-write at the dangerous moment: the naive writer
        # rewrites `final` in place; the atomic writer stages into `.tmp`.
        watch = final + ".tmp" if atomic else final
        for attempt in range(ATTEMPTS):
            proc = subprocess.Popen(
                [PY, str(script), td, "1" if atomic else "0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            deadline = time.time() + 60
            # Phase 1: a complete checkpoint must exist before we strike — the
            # question is whether a mid-write death destroys what was there.
            while time.time() < deadline and _size(final) < ref * 0.99:
                if proc.poll() is not None:
                    break
                time.sleep(0.0002)   # sub-ms poll, but never a hot spin on a shared box
            # Phase 2: strike the instant we SEE an in-progress partial write.
            while time.time() < deadline:
                s = _size(watch)
                if 0 < s < ref * 0.9:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.0002)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

            if not os.path.exists(final):
                continue
            checked += 1
            try:
                loaded = torch.load(final, map_location="cpu", weights_only=False)
                # Loading is not enough — assert the payload is complete.
                assert set(loaded) == {"step", "model", "optim", "rng"}
                assert len(loaded["model"]) == 8 and len(loaded["optim"]) == 8
                assert loaded["rng"].numel() > 0
                steps_seen.append(int(loaded["step"]))
            except Exception:
                corrupt += 1
            for p in (final, final + ".tmp"):
                Path(p).unlink(missing_ok=True)
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
