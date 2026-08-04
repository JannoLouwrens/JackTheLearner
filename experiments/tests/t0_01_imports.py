"""T0.01 — every live module imports cleanly, with no side effects.

An import that starts training, opens a window, or downloads a model makes every
later timing and memory measurement meaningless.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

LIVE_MODULES = [
    "UnifiedBrain", "VirtualWorld", "TaskManager", "Persistence", "EmotionalState",
    "Personality", "MovementMoodCoupling", "InnerMonologue", "SymbolicCalculator",
    "AlphaGeometryLoop", "MoCapLoader", "TrainingPipeline", "AudioListener",
]


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    ok, failed, slow = 0, [], []
    for name in LIVE_MODULES:
        t0 = time.time()
        try:
            importlib.import_module(name)
            ok += 1
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}")
        dt = time.time() - t0
        # A slow import is a side effect: model download, dataset scan, or worse.
        if dt > 5.0:
            slow.append(f"{name}:{dt:.1f}s")
    return {
        "modules_imported": ok,
        "modules_total": len(LIVE_MODULES),
        "failed": ";".join(failed) or "none",
        "slow_imports": ";".join(slow) or "none",
    }


def _check(m: dict, _c: dict) -> bool:
    return m["modules_imported"] == m["modules_total"] and m["slow_imports"] == "none"


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.01"], _experiment, _check, ledger=ledger)
