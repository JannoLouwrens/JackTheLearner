"""T1.08 — measure the noise floor, so later claims can be believed.

This spec produces a number that every subsequent tier depends on and that no
other spec provides: how much a result moves when NOTHING changes but the seed.

Without it, "the world model improved held-out error by 8%" is uninterpretable.
If reruns of the identical configuration span 12%, that 8% is noise wearing a
result's clothes. RL and generative-policy literature is littered with exactly
this error, which is why the Spec dataclass carries a `seeds` field and says so:
"RL effect sizes are routinely smaller than seed noise."

So: train the SAME configuration at three seeds, changing nothing else, and
report the spread. The spec passes when the effect being measured — the model's
improvement over predicting the mean action — is large compared to the
seed-to-seed spread of that same quantity.

  effect          mean improvement over the do-nothing baseline
  noise           std of that improvement across seeds
  snr             effect / noise

CONTROL: three arms that differ ONLY by seed must still differ. If the spread is
exactly zero, seeding is over-constrained (or the arms are not actually
independent) and the noise floor is a fiction — which would make every later
comparison look more significant than it is. T0.02 established that different
seeds must produce different traces; this checks the same property survives a
full training run.

The number this produces should be quoted whenever a later tier claims an
improvement. An effect smaller than the noise floor recorded here is not a result.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

SEEDS = [0, 1, 2]
MIN_SNR = 3.0          # effect must be >= 3x the seed spread
MIN_SPREAD = 1e-6      # ... but the spread must not be zero, or seeding is fake

JOB = r'''
import json, torch, torch.nn.functional as F
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_TEST, STEPS, BS, RANK = 2048, 512, 1500, 64, 8

def make_task(cfg, seed):
    # Task fixed across arms: only the TRAINING seed varies, so the spread
    # measured is optimisation noise, not task noise.
    g = torch.Generator().manual_seed(900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    A = torch.randn(cfg.obs_dim, RANK, generator=g) / (cfg.obs_dim ** 0.5)
    B = torch.randn(RANK, cfg.action_chunk_size * cfg.action_dim, generator=g)
    tgt = (torch.tanh(obs @ A) @ B).view(n, cfg.action_chunk_size, cfg.action_dim) * 0.3
    return obs.to(DEV), tgt.to(DEV)

def arm(seed):
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).to(DEV).train()
    obs, tgt = make_task(cfg, seed)
    tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]
    # Same recipe as T1.07 and TrainingPipeline. A noise floor measured under a
    # different configuration would not bound the claims it is supposed to bound.
    opt, step_fn = brain.make_action_optimizer(lr=3e-4, warmup_steps=100,
                                               max_grad_norm=2.0)
    for step in range(STEPS):
        i = (step * BS) % (N_TRAIN - BS)
        loss = brain.action_training_loss(tr_o[i:i+BS], tr_t[i:i+BS])["loss"]
        opt.zero_grad(); loss.backward(); step_fn()
    brain.eval()
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(te_o)
        heldout = float(F.mse_loss(pred.float(), te_t.float()))
        base = float(F.mse_loss(tr_t.mean(0, keepdim=True).expand_as(te_t).float(),
                                te_t.float()))
    return {"seed": seed, "heldout": heldout, "mean_baseline": base,
            "improvement": base - heldout}

out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "arms": [arm(s) for s in __SEEDS__]}
import os as _o
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t108.json"), "w"), indent=1)
print("DONE", json.dumps(out)[:600], flush=True)
'''


def _submit() -> dict:
    job = build_job(JOB.replace("__SEEDS__", repr(SEEDS)))
    res = submit(job, prefer="colab", est_hours=0.3, timeout_s=3000,
                 fetch=["t108.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t108.json")
    if not path:
        raise RuntimeError(
            f"no artifact from {res.backend}. message={res.message!r} "
            f"stdout_tail={res.stdout[-400:]!r} stderr_tail={res.stderr[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    return d


_CACHE: dict = {}


def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    return mean, var ** 0.5


def _experiment(seed: int) -> dict:
    _CACHE.update(_submit())
    arms = _CACHE["arms"]
    imps = [a["improvement"] for a in arms]
    held = [a["heldout"] for a in arms]
    eff, noise = _stats(imps)
    h_mean, h_std = _stats(held)
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"], "seeds": len(arms),
        "heldout_mean": round(h_mean, 5),
        "heldout_std": round(h_std, 6),
        "heldout_cv_pct": round(100 * h_std / max(abs(h_mean), 1e-9), 3),
        "effect": round(eff, 5),
        "seed_noise": round(noise, 6),
        "snr": round(eff / max(noise, 1e-9), 2),
        # THE number later tiers must quote: an improvement smaller than this is
        # indistinguishable from rerunning the same code.
        "min_detectable_effect": round(2 * noise, 6),
    }


def _control(seed: int) -> dict:
    """Seeds must actually change the outcome, or the noise floor is fictional."""
    held = [a["heldout"] for a in _CACHE["arms"]]
    return {"distinct_results": len(set(round(h, 9) for h in held)),
            "spread": round(max(held) - min(held), 8)}


def _check(m: dict, c: dict) -> bool:
    return (m["snr"] >= MIN_SNR
            and c["spread"] > MIN_SPREAD
            and c["distinct_results"] == len(SEEDS))


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.08"], _experiment, _check, control_fn=_control, ledger=ledger)
