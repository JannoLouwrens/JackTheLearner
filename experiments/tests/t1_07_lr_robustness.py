"""T1.07 — training must not sit on a knife-edge of learning rate.

Why this matters more than it looks. If exactly one learning rate works, every
later comparison in the ladder is contaminated: an ablation that "hurts" may
simply have shifted the optimal LR, and a component that "helps" may have been
luckier with the one value we happened to pick. Tier 3 deletes components on the
strength of such comparisons, so a knife-edge here would mean deleting working
code on the basis of a tuning artifact.

So: train the same identifiable task at 1e-4, 3e-4 and 1e-3 — a 10x span — and
require ALL of them to beat the do-nothing baseline of predicting the mean action.
Not "converge to the same value"; different LRs legitimately land in different
places. The claim is only that the model is not so brittle that a 3x change in
step size destroys it.

Two guards against a vacuous pass:

  CONTROL   lr=1.0 is absurd for Adam on this model and MUST fail. If it passes,
            the bar is too low to discriminate anything and the result is void.
  REFERENCE a plain MLP trained on the same task must succeed. This is the lesson
            from T1.02 v2, which was unpassable for two redesigns because the TASK
            was underdetermined, not because the model was broken. When the
            simplest possible learner also fails, suspect the experiment.

Runs on GPU. On this box a single arm is ~40 minutes of CPU; four arms on a T4 is
a few minutes, and T0.07 measured why — the forward costs 155x the physics it
drives, and 4 shared ARM cores are the wrong instrument for anything that trains.

GRADIENT CLIPPING ADDED 2026-08-05, with evidence, and NOT to move the bar. The
first run scored 5.443 / 13.195 / 0.643 across the three LRs — at 1e-3 the model
came out WORSE than predicting the mean action (0.643x), 20.5x spread end to end.
The cause was a discrepancy between this test and real training: TrainingPipeline
clips gradients at max_grad_norm=2.0 (lines 495 and 681) and these arms did not,
so the test measured a configuration nobody runs. The threshold is unchanged; the
arms now train the way the pipeline trains.

If it still fails with clipping, that is a genuine model finding rather than a
harness artifact, and the reference arm distinguishes them: it scored 7.605, so
the task is learnable and this is not another T1.02-style unidentifiable task.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# Pre-registered, before any run.
LRS = [1e-4, 3e-4, 1e-3]        # a 10x span
ABSURD_LR = 1.0                 # control: must fail
MIN_BEAT_MEAN = 1.15            # each LR must beat mean-prediction by this factor
MAX_GRAD_NORM = 2.0             # TrainingPipeline.py:76 — match real training
WARMUP_STEPS = 100              # 1500-step run; warmup is the fix for the 1e-3 collapse

JOB = r'''
import json, torch, torch.nn.functional as F
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_TEST, STEPS, BS, RANK = 2048, 512, 1500, 64, 8
SEED = 0

def make_task(cfg, seed):
    """Identifiable by construction: a rank-8 tanh map, 2048 samples for 8
    latent directions. T1.02 v2 failed because 64 samples for obs_dim=348 is
    underdetermined — no architecture can fit what is not determined."""
    g = torch.Generator().manual_seed(seed + 900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    A = torch.randn(cfg.obs_dim, RANK, generator=g) / (cfg.obs_dim ** 0.5)
    B = torch.randn(RANK, cfg.action_chunk_size * cfg.action_dim, generator=g)
    tgt = (torch.tanh(obs @ A) @ B).view(n, cfg.action_chunk_size, cfg.action_dim) * 0.3
    return obs.to(DEV), tgt.to(DEV)

def arm(lr, seed=SEED):
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).to(DEV).train()
    obs, tgt = make_task(cfg, seed)
    tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]

    # One source of truth for the recipe, so a spec and the pipeline cannot
    # drift apart -- the drift is what produced the previous false diagnosis.
    opt, step_fn = brain.make_action_optimizer(lr=lr, warmup_steps=__WARMUP__,
                                               max_grad_norm=__CLIP__)
    for step in range(STEPS):
        i = (step * BS) % (N_TRAIN - BS)
        loss = brain.action_training_loss(tr_o[i:i+BS], tr_t[i:i+BS])["loss"]
        opt.zero_grad(); loss.backward(); step_fn()
        if not torch.isfinite(loss):
            return {"lr": lr, "heldout": float("inf"), "diverged": True}

    brain.eval()
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(te_o)
        heldout = float(F.mse_loss(pred.float(), te_t.float()))
        mean_base = float(F.mse_loss(tr_t.mean(0, keepdim=True).expand_as(te_t).float(),
                                     te_t.float()))
    return {"lr": lr, "heldout": heldout, "mean_baseline": mean_base, "diverged": False}

def reference(seed=SEED):
    """Plain MLP, no flow matching. If THIS fails the task is void, not the model."""
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    obs, tgt = make_task(cfg, seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(cfg.obs_dim, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, cfg.action_chunk_size * cfg.action_dim)).to(DEV)
    tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(STEPS):
        i = (step * BS) % (N_TRAIN - BS)
        p = net(tr_o[i:i+BS]).view(-1, cfg.action_chunk_size, cfg.action_dim)
        loss = F.mse_loss(p, tr_t[i:i+BS])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(te_o).view(-1, cfg.action_chunk_size, cfg.action_dim)
        return {"heldout": float(F.mse_loss(p, te_t)),
                "mean_baseline": float(F.mse_loss(
                    tr_t.mean(0, keepdim=True).expand_as(te_t), te_t))}

out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "arms": [arm(lr) for lr in __LRS__],
       "absurd": arm(__ABSURD__),
       "reference": reference()}
import os as _o
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t107.json"), "w"), indent=1)
print("DONE", json.dumps(out)[:600], flush=True)
'''


def _submit() -> dict:
    body = (JOB.replace("__LRS__", repr(LRS))
               .replace("__ABSURD__", repr(ABSURD_LR))
               .replace("__CLIP__", repr(MAX_GRAD_NORM))
               .replace("__WARMUP__", repr(WARMUP_STEPS)))
    job = build_job(body)
    res = submit(job, prefer="colab", est_hours=0.4, timeout_s=3000,
                 fetch=["t107.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t107.json")
    if not path:
        raise RuntimeError(f"no artifact returned; stdout tail: {res.stdout[-300:]}")
    data = json.loads(Path(path).read_text())
    data["backend"] = res.backend
    return data


_CACHE: dict = {}


def _experiment(seed: int) -> dict:
    _CACHE.update(_submit())
    arms, ref = _CACHE["arms"], _CACHE["reference"]
    ratios = {f"lr_{a['lr']:g}": round(a["mean_baseline"] / max(a["heldout"], 1e-9), 3)
              for a in arms}
    worst = min(ratios.values())
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        **ratios,
        "worst_lr_advantage": worst,
        "lrs_beating_baseline": sum(1 for v in ratios.values() if v >= MIN_BEAT_MEAN),
        "reference_advantage": round(ref["mean_baseline"] / max(ref["heldout"], 1e-9), 3),
        "spread_ratio": round(max(a["heldout"] for a in arms)
                              / max(min(a["heldout"] for a in arms), 1e-9), 3),
    }


def _control(seed: int) -> dict:
    a = _CACHE.get("absurd", {})
    return {"absurd_lr": ABSURD_LR,
            "absurd_diverged": bool(a.get("diverged")),
            "absurd_advantage": round(a.get("mean_baseline", 0.0)
                                      / max(a.get("heldout", 1e9), 1e-9), 4)}


def _check(m: dict, c: dict) -> bool:
    # Every LR in the 10x span beats mean-prediction; the reference arm proves the
    # task is learnable at all; the absurd LR must NOT clear the same bar.
    return (m["lrs_beating_baseline"] == len(LRS)
            and m["worst_lr_advantage"] >= MIN_BEAT_MEAN
            and m["reference_advantage"] >= MIN_BEAT_MEAN
            and c["absurd_advantage"] < MIN_BEAT_MEAN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.07"], _experiment, _check, control_fn=_control, ledger=ledger)
