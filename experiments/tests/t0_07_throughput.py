"""T0.07 — measure real throughput before believing any compute estimate.

Not a pass/fail claim about quality; a measurement that every later plan depends
on. The pipeline review measured ~27 env-steps/s and traced it to 6,095 ATen
dispatches per B=1 forward — the rollout loop is CPU-dispatch-bound, not FLOP-
bound. That number decides whether a T4 helps at all: if the bottleneck is Python
dispatch on one environment, a GPU sits idle waiting and the honest answer is to
vectorise before renting anything.

So this records these figures and their ratios:
  bare      — MuJoCo physics alone, no policy. The ceiling.
  vectorised— N environments stepped together, amortising Python overhead.
  policy    — physics plus a full brain forward. What training actually costs.

The policy figure is measured TWICE, because there are two different brains:
`UnifiedBrainConfig()` as constructed today loads SmolLM2-1.7B in-process (6.9 GB
resident, 1.77B params, of which 57.8M are trainable), and the same config with
`llm_enabled=False`. Decision D-dialogue puts the chat model out-of-process, so
the second is the number to plan the refactor toward — but the first is the
number every other test on this ladder is actually paying today, including the
T1.03 gradient sweep. Quoting only one of them was the first version's mistake.

The pass condition is only that the measurement is sound (stable across repeats,
the harness recovers a ground truth it cannot fake, and the environment is
genuinely stepping). The numbers themselves are recorded for planning. A test
that "fails" because a box is slow would be measuring the box, not the code.
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 300
N_VEC = 8

# Pre-registered validity thresholds. These gate whether the measurement is
# believable, not whether the box is fast.
MAX_REPEAT_SPREAD = 0.25       # bare physics, trial to trial
MAX_POLICY_REL_STD = 0.20      # policy forward, trial to trial
MAX_CALIBRATION_ERROR = 0.15   # harness must recover a known scaling
CALIB_WORK = 20_000


def _setup():
    os.environ.setdefault("MUJOCO_GL", "disabled")
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------
# Timing harness. One path, used by every measurement below, so a broken timer
# cannot flatter one figure and not another.
# --------------------------------------------------------------------------

def _rate(body, n: int, warmup: int = 0) -> float:
    for _ in range(warmup):
        body()
    t0 = time.perf_counter()
    for _ in range(n):
        body()
    dt = time.perf_counter() - t0
    return n / dt if dt > 0 else float("inf")


def _trials(body, n: int, warmup: int, trials: int = 3):
    """Warmup before EVERY trial, not just the first.

    This is not ceremony. Warming once and then timing the policy measured
    3.67 Hz with 28.9% spread, while the same forward *plus* a 224x224 vision
    encode measured 4.38 Hz — a heavier computation cannot be faster, so the
    timer was catching the 6.9 GB model still being paged in rather than the
    computation. Per-trial warmup removed the anomaly and the spread fell to
    6.4%. The reproducibility threshold was not what changed.
    """
    rates = [_rate(body, n, warmup) for _ in range(trials)]
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return mean, (var ** 0.5) / mean if mean else float("inf")


def _calibration_work(units: int):
    """Cost exactly linear in `units`. Nothing lazy, nothing cached, no I/O."""
    acc = 0
    for i in range(units * CALIB_WORK):
        acc += i
    return acc


def _calibrate() -> tuple[float, float]:
    """Ground truth the harness cannot fake: doubling the work must halve the rate.

    Calibrating against `time.sleep(1/200)` instead was tried and failed at 16.3%
    error — but the fault was the ground truth, not the timer. sleep() promises
    only a LOWER bound, and delivered 5.8 ms for a 5 ms request under co-tenant
    load, so "200 Hz" was never the true rate. Scaling is a real invariant
    whatever the scheduler is doing.
    """
    one, _ = _trials(lambda: _calibration_work(1), n=30, warmup=5)
    two, _ = _trials(lambda: _calibration_work(2), n=15, warmup=3)
    return one, abs((one / two) - 2.0) / 2.0


# --------------------------------------------------------------------------
# The three subjects.
# --------------------------------------------------------------------------

def _bare():
    import gymnasium as gym
    e = gym.make("Humanoid-v5"); e.reset(seed=0)
    a = e.action_space.sample()

    def step():
        _, _, term, trunc, _ = e.step(a)
        if term or trunc:
            e.reset()

    return _trials(step, n=STEPS, warmup=50)


def _vectorised(n: int = N_VEC):
    """Measured with the same repeat discipline as everything else.

    The first version timed this once and the commit message concluded
    "vectorisation is SLOWER than one env, 0.87x". Re-measured, it read 1.17x.
    A ratio that changes sign between runs is not a finding — it is an
    un-repeated measurement, which is the failure mode this whole ladder exists
    to prevent. So the spread is now reported and gated, and no directional
    claim is made that the spread does not support.
    """
    import gymnasium as gym
    v = gym.make_vec("Humanoid-v5", num_envs=n, vectorization_mode="sync")
    v.reset(seed=0)
    a = v.action_space.sample()
    hz, rel_std = _trials(lambda: v.step(a), n=STEPS // 4, warmup=20)
    return hz * n, rel_std


def _with_policy(llm: bool) -> dict:
    """One rollout step: brain forward -> width-checked ctrl write -> physics."""
    import gymnasium as gym, numpy as np, torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    from VirtualWorld import apply_action

    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = llm                     # a real field (UnifiedBrain.py:122)
    brain = UnifiedBrain(cfg).eval()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    e = gym.make("Humanoid-v5"); u = e.unwrapped
    e.reset(seed=0)
    obs = torch.zeros(1, cfg.obs_dim)

    def step():
        with torch.no_grad():
            act = brain(obs)["actions"]
        # Chunked head: take the first step of the chunk, width-checked.
        a = act.reshape(-1)[: u.model.nu].double().numpy()
        apply_action(u.data, u.model, a)
        _, _, term, trunc, _ = e.step(np.zeros(u.model.nu))
        if term or trunc:
            e.reset()

    hz, rel_std = _trials(step, n=10, warmup=3)
    out = {
        "hz": hz,
        "rel_std": rel_std,
        "peak_rss_mb": rss_mb,
        "total_params": sum(p.numel() for p in brain.parameters()),
        "trainable_params": sum(p.numel() for p in brain.parameters() if p.requires_grad),
    }
    del brain, e
    gc.collect()
    return out


def _experiment(seed: int) -> dict:
    _setup()
    calib_hz, calib_err = _calibrate()

    bare, spread = _bare()    # repeat: a measurement that will not repeat is not one
    vec, vec_spread = _vectorised()

    # LLM-off first: it is the smaller resident set, so the 6.9 GB build does not
    # colour the peak-RSS figure attributed to it.
    off = _with_policy(llm=False)
    on = _with_policy(llm=True)

    return {
        "calibration_hz": round(calib_hz, 1),
        "calibration_error": round(calib_err, 4),
        "bare_steps_per_s": round(bare, 1),
        "bare_repeat_spread": round(spread, 4),
        "vectorised_steps_per_s": round(vec, 1),
        "vectorised_rel_std": round(vec_spread, 4),
        "vectorised_speedup": round(vec / bare, 2),

        # Today's default brain — what every other test on this ladder pays.
        "policy_steps_per_s": round(on["hz"], 2),
        "policy_rel_std": round(on["rel_std"], 4),
        "policy_slowdown_vs_bare": round(bare / max(on["hz"], 1e-9), 1),
        "hours_for_2M_steps_policy": round(2_000_000 / max(on["hz"], 1e-9) / 3600, 1),
        "policy_total_params": on["total_params"],
        "policy_trainable_params": on["trainable_params"],
        "policy_peak_rss_mb": round(on["peak_rss_mb"], 1),

        # The same brain with the chat model out of the graph (decision D-dialogue).
        "policy_no_llm_steps_per_s": round(off["hz"], 2),
        "policy_no_llm_rel_std": round(off["rel_std"], 4),
        "hours_for_2M_steps_no_llm": round(2_000_000 / max(off["hz"], 1e-9) / 3600, 1),
        "policy_no_llm_total_params": off["total_params"],
        "llm_removal_speedup": round(off["hz"] / max(on["hz"], 1e-9), 2),
    }


def _control(seed: int) -> dict:
    """The scaffold must contribute nothing.

    Same `_rate` harness, body removed. If an empty loop is not orders of
    magnitude faster than the measured work, the reported rate is dominated by
    timing overhead and measures the harness, not the subject.
    """
    return {"empty_loop_hz": round(_rate(lambda: None, n=200_000, warmup=1000), 1)}


def _check(m: dict, c: dict) -> bool:
    # Sound measurement, not a performance bar: the harness recovers a known
    # scaling, repeats agree, the sim genuinely ran, and the numbers come from
    # the work rather than the loop.
    return (m["calibration_error"] < MAX_CALIBRATION_ERROR
            and m["bare_repeat_spread"] < MAX_REPEAT_SPREAD
            and m["vectorised_rel_std"] < MAX_REPEAT_SPREAD
            and m["policy_rel_std"] < MAX_POLICY_REL_STD
            and m["policy_no_llm_rel_std"] < MAX_POLICY_REL_STD
            and m["bare_steps_per_s"] > 1
            and m["policy_steps_per_s"] > 0
            and c["empty_loop_hz"] > 100 * m["bare_steps_per_s"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.07"], _experiment, _check, ledger=ledger, control_fn=_control)
