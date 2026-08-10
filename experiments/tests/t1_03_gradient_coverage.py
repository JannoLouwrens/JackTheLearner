"""T1.03 — every trainable parameter must receive gradient.

This is the test the repo needed and never had. The pipeline review measured
45,538,295 parameters (38.6% of 117,888,028) with no live call site: a
hierarchical_planner larger than the backbone it sits on, a temporal_memory never
passed `memory=`, a world_model gated on an argument nothing supplies.

None of that is visible from a loss curve. It is visible in one backward pass.

The test reports the orphan fraction and names the worst offenders, so the
remedy is unambiguous: wire it, or delete it.

CONTROL, added 2026-08-10 (OVERSIGHT §1.3, asked for over four audits): this
spec had none, so "0 orphans" was never shown to be a statement the measurement
was CAPABLE of contradicting on this build. Two parameters are planted into the
same brain, driven through the same loss and read by the same scan, one for each
branch of the detector — a module that is never called (`grad is None`) and a
parameter that IS reached but multiplied by zero (`grad` present and all-zero).
The second is the insidious one: a live wire behind a dead gate looks wired from
every angle except its gradient. Both must be reported, or a green T1.03 means
only that the scan found nothing anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# Anything above this is a structural problem, not an oversight.
MAX_ORPHAN_FRACTION = 0.05


#: The two planted orphans, by parameter-name prefix. Named here so the check
#: can assert WHICH parameters were caught: a count alone would be satisfied by
#: the scan flagging two unrelated tensors.
PLANT_NEVER_CALLED = "orphan_never_called"     # grad stays None
PLANT_ZERO_GRAD = "orphan_zero_grad"           # grad exists and is all-zero


def _measure(seed: int, plant: bool = False) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    brain = UnifiedBrain(cfg)
    brain.train()

    if plant:
        # Planted AFTER construction and BEFORE the loss, so both plants travel
        # the identical path the real parameters do. `torch.nn` is reached
        # through `torch` so this control needs no import the experiment lacks.
        brain.add_module(PLANT_NEVER_CALLED, torch.nn.Linear(8, 8))
        setattr(brain, PLANT_ZERO_GRAD, torch.nn.Parameter(torch.randn(8)))

    total = sum(p.numel() for p in brain.parameters())
    trainable = sum(p.numel() for p in brain.parameters() if p.requires_grad)

    # Feed EVERY modality at the shape the code ACTUALLY expects. Earlier this
    # test guessed from config names and mis-fed two of them: touch takes width
    # 10 (hardcoded in TouchEncoder(10, config.touch_dim) — touch_dim is the
    # OUTPUT), and audio takes a raw waveform of audio_sample_rate samples, not
    # a feature vector. Both were reported as orphaned when they were merely
    # never exercised. A test that withholds a module's input and then calls it
    # dead is worse than no test.
    obs = torch.randn(2, cfg.obs_dim)
    candidates = {
        "vision": (2, 3, 224, 224),
        "touch": (2, 10),
        "audio": (2, getattr(cfg, "audio_sample_rate", 16000)),
        "goal": (2, cfg.d_model),
        "task": (2, cfg.d_model),
    }
    kwargs = {}
    for name, shape in candidates.items():
        try:
            brain(obs, **{name: torch.randn(*shape)})
            kwargs[name] = torch.randn(*shape)
        except Exception:
            pass  # genuinely not accepted by this build

    # Measure under the REAL training objective, not a bare forward. The runtime
    # drives the robot from ActionExpert via flow matching, which forward() does
    # not touch at all — measuring a forward would score the wrong path.
    target = torch.randn(2, cfg.action_chunk_size, cfg.action_dim)
    losses = brain.action_training_loss(obs, target)["loss"]

    aux = brain(obs, **kwargs)
    losses = losses + 0.01 * sum(
        v.float().pow(2).mean() for k, v in aux.items()
        if torch.is_tensor(v) and v.dtype.is_floating_point and k != "actions")
    if plant:
        # The second plant IS reached by autograd and still receives nothing:
        # d(0 * x)/dx is a present, all-zero gradient. This is the branch a
        # never-called module cannot exercise.
        losses = losses + 0.0 * getattr(brain, PLANT_ZERO_GRAD).sum()
    losses.backward()

    orphan_params, orphan_by_module = 0, {}
    caught = set()
    for name, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            orphan_params += p.numel()
            top = name.split(".")[0]
            orphan_by_module[top] = orphan_by_module.get(top, 0) + p.numel()
            caught.add(top)

    worst = sorted(orphan_by_module.items(), key=lambda kv: -kv[1])[:6]
    out = {
        "modalities_fed": ",".join(kwargs) or "proprio-only",
        "total_params": total,
        "trainable_params": trainable,
        "params_without_grad": orphan_params,
        "orphan_fraction": round(orphan_params / max(1, trainable), 4),
        "worst_offenders": "; ".join(f"{k}={v:,}" for k, v in worst) or "none",
    }
    if plant:
        # Reported as separate keys, not folded into the fraction: 80 planted
        # params against ~50M trainable move `orphan_fraction` by 1.6e-6, so a
        # gate on the fraction could not have told a caught plant from a missed
        # one. "Gate on the quantity a reader would act on" (LESSONS.md).
        out["planted_none_detected"] = float(PLANT_NEVER_CALLED in caught)
        out["planted_zero_grad_detected"] = float(PLANT_ZERO_GRAD in caught)
    return out


def _experiment(seed: int) -> dict:
    return _measure(seed, plant=False)


def _control(seed: int) -> dict:
    """Two deliberately dead parameters must both be REPORTED as orphaned.

    Same brain, same objective, same scan — the plants are inserted into the
    live model rather than into a tidied restatement of it, because a control
    that re-implements the thing under test certifies the re-implementation
    (T0.16's shipped-kernel lesson).
    """
    m = _measure(seed, plant=True)
    return {"planted_none_detected": m["planted_none_detected"],
            "planted_zero_grad_detected": m["planted_zero_grad_detected"],
            "orphan_fraction_with_plants": m["orphan_fraction"],
            "worst_offenders_with_plants": m["worst_offenders"]}


def _check(m: dict, c: dict) -> bool:
    return (m["orphan_fraction"] <= MAX_ORPHAN_FRACTION
            # …and the scan must be able to say so. Both branches of the
            # detector are exercised, and both must fire on the plants.
            and c["planted_none_detected"] == 1.0
            and c["planted_zero_grad_detected"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
