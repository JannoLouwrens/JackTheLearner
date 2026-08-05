"""T1.04 — every trainable module's weights must actually change.

Distinct from T1.03, which asks whether gradient ARRIVES. A parameter can receive
a gradient and still not move: a zero learning rate for its group, a scheduler
that never warms up, weight decay exactly cancelling the update, or an optimiser
built over a stale parameter list that omits modules added later. Each of those
leaves T1.03 green and the module frozen in practice.

The optimiser-list failure is the realistic one here. Training code repeatedly
does `Adam(model.parameters())` at construction; anything attached afterwards, or
any module swapped during setup, silently never updates.

Control: the same steps with lr=0 must move NOTHING. Without it, floating-point
noise alone could pass this.

FED MODALITIES, 2026-08-05. The first run scored moved_frac 0.8863 with 6.6M
parameters stuck, and the diagnosis is the same mistake T1.03 made: the test fed
proprioception ONLY, so vision/audio/touch/language encoders received no input
and therefore no gradient. They were not frozen, they were unasked. Correct
shapes matter and are not guessable from config names -- touch takes width 10
(TouchEncoder(10, cfg.touch_dim); touch_dim is the OUTPUT) and audio takes a raw
16000-sample waveform.

A SECOND, GENUINE finding survived that fix and must not be blurred into it: some
heads have no loss term in action_training_loss AT ALL -- value_head,
physics_head, task_completion_head, emotional_state, movement_mood. They are not
starved of input; nothing in the action objective refers to them. That is an
architecture question (do they earn their parameters?), which Tier 3 answers by
ablation, not a plumbing bug. So they are declared here explicitly: listed, with
a reason, and any module going stuck OUTSIDE that list fails the spec loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 20

# Modules the action objective does not train, declared BEFORE running so a newly
# stuck module cannot hide among them. Each needs its own loss or its own deletion
# (Tier 3); both are recorded in docs/DECISIONS_NEEDED.md.
NOT_IN_ACTION_LOSS = {
    "value_head":           "RL critic — trained by a value loss that does not exist yet",
    "physics_head":         "predicts dynamics; no physics objective is wired",
    "task_completion_head": "task-success classifier; no task labels in the corpus",
    "emotional_state":      "affect model; no supervision signal defined",
    "movement_mood":        "couples affect to gait; downstream of emotional_state",
    "tokenizer":            "embedding table for the local text tower, frozen by design",
    # ActionHead carries four output heads; this embodiment drives exactly one.
    # MuJoCo Humanoid-v5 has nu=17 and is position-controlled, so the 40-DOF
    # manipulation heads and both velocity heads have no consumer. Confirmed by
    # inspection: locomotion_head moves, the other three never receive a grad.
    # Tier 3 decides whether they earn their 49,761 parameters or get deleted.
    "action_head.locomotion_velocity":    "velocity control mode; unused (position control)",
    "action_head.manipulation_head":      "40-DOF manipulator; this embodiment has nu=17",
    "action_head.manipulation_velocity":  "both of the above at once",
}


def _declared(param_name: str) -> bool:
    """Declared untrained by top-level module OR by dotted submodule prefix.

    Submodule granularity matters: action_head as a whole DOES train (its shared
    trunk and locomotion head move), so declaring the whole module would hide a
    future regression in the part that matters.
    """
    return any(param_name == k or param_name.startswith(k + ".")
               for k in NOT_IN_ACTION_LOSS)


def _deltas(seed: int, lr: float) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).train()

    before = {n: p.detach().clone() for n, p in brain.named_parameters() if p.requires_grad}
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=lr)

    g = torch.Generator().manual_seed(seed + 3)
    obs = torch.randn(4, cfg.obs_dim, generator=g)
    tgt = torch.randn(4, cfg.action_chunk_size, cfg.action_dim, generator=g)
    # Every modality fed, at the shapes the encoders actually take. Withholding
    # an input and then calling its encoder frozen is measuring the test, not
    # the model.
    feeds = {
        "vision": torch.randn(4, 3, 224, 224, generator=g),
        "touch": torch.randn(4, 10, generator=g),
        "audio": torch.randn(4, getattr(cfg, "audio_sample_rate", 16000), generator=g),
    }
    lang = ["walk forward", "turn left", "stand still", "run"]
    for _ in range(STEPS):
        loss = brain.action_training_loss(obs, tgt, language=lang, **feeds)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    # Per top-level module, so a single frozen submodule is named rather than
    # averaged away by the millions of parameters that did move.
    moved, stuck = {}, {}
    for n, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        # Report at submodule granularity when the module is partially declared,
        # so "action_head" cannot mask which of its four heads is dead.
        top = ".".join(n.split(".")[:2]) if _declared(n) else n.split(".")[0]
        d = float((p.detach() - before[n]).abs().max())
        if d > 0:
            moved[top] = moved.get(top, 0) + p.numel()
        else:
            stuck[top] = stuck.get(top, 0) + p.numel()

    total = sum(moved.values()) + sum(stuck.values())
    # Split the stuck set: declared-untrained is an architecture question,
    # undeclared is a plumbing bug and fails the spec.
    declared = {k: v for k, v in stuck.items() if _declared(k)}
    undeclared = {k: v for k, v in stuck.items() if not _declared(k)}
    return {
        "params_moved": sum(moved.values()),
        "params_stuck": sum(stuck.values()),
        "moved_frac": round(sum(moved.values()) / max(1, total), 4),
        "undeclared_stuck_params": sum(undeclared.values()),
        "undeclared_stuck": "; ".join(f"{k}={v:,}" for k, v in sorted(undeclared.items())) or "none",
        "declared_untrained": "; ".join(f"{k}={v:,}" for k, v in sorted(declared.items())) or "none",
    }


def _experiment(seed: int) -> dict:
    return _deltas(seed, lr=3e-4)


def _control(seed: int) -> dict:
    """lr=0 — nothing may move. Proves the measurement is not reading noise."""
    return _deltas(seed, lr=0.0)


def _check(m: dict, c: dict) -> bool:
    # No module may be stuck unless it was declared untrained in advance, and the
    # zero-lr control must move nothing at all. Stricter than a bare moved_frac
    # bar: a newly-orphaned encoder fails here even if it is small.
    return m["undeclared_stuck_params"] == 0 and c["params_moved"] == 0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.04"], _experiment, _check, control_fn=_control, ledger=ledger)
