"""T1.05 — pretrained weights survive construction and training.

The bug this guards: `self.apply(self._init_weights)` recursed into the loaded
LLM and overwrote it with normal_(std=0.02). requires_grad_(False) stops
gradients, not in-place initialisation. Measured before the fix: q_proj std
0.1010 -> 0.0196, embeddings 1.0013 -> 0.0197. Every run this project ever did
used a randomised "pretrained" backbone.

Two things are asserted, because the bug has two halves:
  1. CONSTRUCTION does not overwrite pretrained tensors.
  2. TRAINING does not update them (they stay frozen through backward).

Rather than load a real 1.7B model on a shared box, we plant a sentinel module
whose name matches the pretrained-prefix contract and verify it is untouched.
That tests the actual mechanism — the traversal skip — at negligible cost.

CONTROL, added 2026-08-10 (OVERSIGHT §1.3, asked for over four audits): an
IDENTICAL sentinel attached OUTSIDE the pretrained-prefix contract must move on
both halves — re-randomised by construction, and updated by training. Without
it, `construct_delta == 0.0 and train_delta == 0.0` is satisfied by a
measurement that reads zero for reasons of its own: a `.clone()` compared
against itself, an initialiser that never ran, an optimiser with an empty
parameter list. The experiment asserts two zeros; the control is what makes
those zeros mean "skipped" rather than "not looked at".

NOTE ON THE HYPOTHESIS, unchanged: the PLASTIC-ONLY decree (owner, 2026-08-09)
means nothing inside Jack ships frozen, so this spec no longer implies a frozen
part of him. It remains a live MECHANISM test — `requires_grad_(False)` stops
gradients, not in-place initialisation, and that trap is waiting for any tensor
loaded from disk (the parent LLM in his world included). No threshold touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
SENTINEL_STD = 0.5  # deliberately unlike the 0.02 the initialiser would impose
#: Attachment point for the control's sentinel. Must NOT be a prefix of
#: anything in `UnifiedBrain._PRETRAINED_PREFIXES` — asserted below rather
#: than assumed, because a control that is accidentally protected passes by
#: looking exactly like the experiment.
UNPROTECTED_NAME = "t1_05_unprotected_sentinel"


def _measure(seed: int, protected: bool = True) -> dict:
    """Attach the sentinel INSIDE (`protected`) or OUTSIDE the pretrained tree.

    One function, two arms, so the control cannot diverge from the experiment in
    anything but the one variable it is meant to vary.
    """
    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn as nn
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    for flag in ("llm_enabled", "enable_intrinsic_motivation"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)

    class _Pretrained(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 16)
            self.emb = nn.Embedding(8, 16)
            nn.init.normal_(self.proj.weight, std=SENTINEL_STD)
            nn.init.normal_(self.emb.weight, std=SENTINEL_STD)

    # Patch the sentinel in before __init__ runs its initialiser, using a name
    # covered by _PRETRAINED_PREFIXES.
    real_init = UnifiedBrain.__init__

    def patched_init(self, config, *a, **kw):
        real_init(self, config, *a, **kw)

    brain = UnifiedBrain(cfg)
    lang = getattr(brain, "language_encoder", None)
    if lang is None:
        raise RuntimeError("no language_encoder to attach the sentinel to")
    sentinel = _Pretrained()
    if protected:
        # `language_encoder.llm` is in _PRETRAINED_PREFIXES: the traversal must
        # skip it, and gradients are off as a loaded backbone's would be.
        lang.llm = sentinel
        for p in sentinel.parameters():
            p.requires_grad_(False)
    else:
        # CONTROL: same module, same scale, attached under a name NOTHING
        # protects, and left trainable. Every mechanism this spec claims must
        # now visibly fail. The name is checked against the live contract, not
        # trusted: an accidentally-protected control would pass by looking
        # exactly like the experiment (LESSONS.md — assert contracts against the
        # source of truth, never against another constant).
        if any(f"{UNPROTECTED_NAME}." .startswith(pref) or
               pref.startswith(UNPROTECTED_NAME)
               for pref in UnifiedBrain._PRETRAINED_PREFIXES):
            raise RuntimeError(
                f"{UNPROTECTED_NAME} collides with _PRETRAINED_PREFIXES")
        brain.add_module(UNPROTECTED_NAME, sentinel)
    before = {n: p.detach().clone() for n, p in sentinel.named_parameters()}

    # 1. Re-running init must not touch it.
    brain._init_trainable_weights()
    construct_delta = max(
        float((p.detach() - before[n]).abs().max()) for n, p in sentinel.named_parameters()
    )

    # 2. Training must not touch it either.
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=1e-3)
    obs = torch.randn(2, cfg.obs_dim)
    for _ in range(3):
        out = brain(obs)
        loss = out["actions"].float().pow(2).mean()
        if not protected:
            # The sentinel is not on the forward path in either arm, so the
            # control puts it there explicitly. Otherwise `train_delta == 0`
            # would hold for the control too — for the trivial reason that
            # nothing asked it for a gradient — and the control would "fail"
            # while proving nothing about the freeze.
            loss = loss + sentinel.proj.weight.float().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    train_delta = max(
        float((p.detach() - before[n]).abs().max()) for n, p in sentinel.named_parameters()
    )

    observed_std = float(sentinel.proj.weight.std())
    return {
        "construct_delta": round(construct_delta, 9),
        "train_delta": round(train_delta, 9),
        "sentinel_std": round(observed_std, 4),
        "expected_std": SENTINEL_STD,
    }


def _experiment(seed: int) -> dict:
    return _measure(seed, protected=True)


def _control(seed: int) -> dict:
    """An UNPROTECTED sentinel must move on BOTH halves.

    Re-randomised by construction (`construct_delta > 0`, and its std pulled to
    the initialiser's 0.02 from 0.5) and updated by training (`train_delta > 0`).
    A zero here would mean the two zeros in the experiment are a property of the
    measurement rather than of the traversal skip.
    """
    m = _measure(seed, protected=False)
    return {"unprotected_construct_delta": m["construct_delta"],
            "unprotected_train_delta": m["train_delta"],
            "unprotected_std": m["sentinel_std"]}


#: The control's sentinel must move by more than float noise on both halves.
#: 1e-6 is four orders below the initialiser's own 0.02 scale, so this is a
#: floor on "moved at all", not a tuned effect size — and `_round6` keeps six
#: SIGNIFICANT figures below 1.0, so a real 3e-7 cannot be stored as 0.0 the way
#: it silently was before T0.15 (LESSONS.md, the recorder's resolution).
MIN_UNPROTECTED_DELTA = 1e-6


def _check(m: dict, c: dict) -> bool:
    # Untouched by construction AND by training, and still at its own scale
    # rather than the 0.02 the initialiser would have imposed.
    protected_held = (m["construct_delta"] == 0.0 and m["train_delta"] == 0.0
                      and abs(m["sentinel_std"] - SENTINEL_STD) < 0.15)
    # …and the same measurement, pointed at an unprotected twin, must report
    # movement on both halves. Otherwise the zeros above are not evidence.
    control_moved = (c["unprotected_construct_delta"] >= MIN_UNPROTECTED_DELTA
                     and c["unprotected_train_delta"] >= MIN_UNPROTECTED_DELTA
                     # Re-randomised to the initialiser's scale, not merely
                     # nudged: this separates "init ran on it" from "training
                     # moved it a little".
                     and c["unprotected_std"] < 0.15)
    return protected_held and control_moved


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.05"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
