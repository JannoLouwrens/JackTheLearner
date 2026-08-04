"""T1.02 — does the architecture exploit structure, or only memorise?

The original version measured training FIT on a single batch and came out at
0.999 — structured and shuffled targets fit identically. That is not the model
failing; it is the experiment being unable to ask the question. A 58M network
memorises 8 arbitrary pairs whether or not a state->action mapping exists, so fit
measures capacity. The original spec said exactly this in its own null_baseline
and I built it anyway.

Only GENERALISATION can detect structure exploitation. So: train on 64 samples,
score 16 states never seen.

  structured   actions ARE a function of state -> held-out error should fall
  shuffled     the same targets permuted, mapping destroyed -> held-out error
               should be no better than predicting the mean

This is strictly harder than the version it replaces. If it fails, the finding is
serious: the architecture cannot learn a state->action mapping, and no amount of
GPU time in Tier 2 will fix that.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
N_TRAIN, N_TEST, STEPS = 64, 16, 400


def _run(seed: int, shuffle: bool) -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).train()

    g = torch.Generator().manual_seed(seed + 900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    W = torch.randn(cfg.obs_dim, cfg.action_chunk_size * cfg.action_dim, generator=g) * 0.05
    tgt = (obs @ W).view(n, cfg.action_chunk_size, cfg.action_dim)

    tr_o, tr_t = obs[:N_TRAIN], tgt[:N_TRAIN]
    te_o, te_t = obs[N_TRAIN:], tgt[N_TRAIN:]

    if shuffle:
        # Destroy the mapping on the TRAINING set only. Held-out targets stay
        # correct, so a model that learned real structure still scores well and
        # one that memorised noise cannot.
        tr_t = tr_t[torch.randperm(N_TRAIN, generator=g)]

    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=3e-4)
    bs = 8
    for step in range(STEPS):
        i = (step * bs) % N_TRAIN
        loss = brain.action_training_loss(tr_o[i:i + bs], tr_t[i:i + bs])["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    brain.eval()
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(te_o)
        heldout = float(torch.nn.functional.mse_loss(pred.float(), te_t.float()))
        # The floor any model gets for free by ignoring the input entirely.
        mean_baseline = float(torch.nn.functional.mse_loss(
            tr_t.mean(0, keepdim=True).expand_as(te_t).float(), te_t.float()))

    return {"heldout_error": round(heldout, 5),
            "mean_baseline": round(mean_baseline, 5),
            "train_loss_final": round(float(loss), 5)}


def _experiment(seed: int) -> dict:
    r = _run(seed, shuffle=False)
    return {"structured_heldout": r["heldout_error"],
            "mean_baseline": r["mean_baseline"],
            "structured_train_loss": r["train_loss_final"]}


def _control(seed: int) -> dict:
    r = _run(seed, shuffle=True)
    return {"shuffled_heldout": r["heldout_error"],
            "shuffled_train_loss": r["train_loss_final"]}


def _check(m: dict, c: dict) -> bool:
    adv = c["shuffled_heldout"] / max(m["structured_heldout"], 1e-9)
    m["heldout_structure_advantage"] = round(adv, 3)
    m["beats_mean_baseline"] = round(m["mean_baseline"] / max(m["structured_heldout"], 1e-9), 3)
    # Structure must generalise better than destroyed structure, AND the
    # structured model must beat the do-nothing baseline of predicting the mean.
    return adv >= 1.25 and m["beats_mean_baseline"] >= 1.1


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.02"], _experiment, _check, control_fn=_control, ledger=ledger)
