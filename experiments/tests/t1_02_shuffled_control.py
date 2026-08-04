"""T1.02 — does the architecture exploit structure, or only memorise?

Redesigned twice, both times because the EXPERIMENT was wrong. Neither threshold
has ever been moved.

  v1  measured training FIT on a single batch. Result 0.999 — structured and
      shuffled targets indistinguishable. A 58M network memorises 8 arbitrary
      pairs whether or not a mapping exists, so fit measures capacity. The
      original spec predicted exactly this in its own null_baseline.
  v2  measured GENERALISATION, which is the right question, but drew 64 training
      samples for a map with obs_dim=348. That system is underdetermined by a
      factor of five: infinitely many maps fit those points and essentially none
      generalise. What proved it was adding a plain-MSE reference arm with no
      flow matching anywhere — it failed identically (0.925 against the mean
      predictor). When the simplest possible learner also fails, the task is
      unlearnable and the model is not the story.
  v3  this one. 2048 training samples so the map is identifiable, 256 held-out
      states never seen, and it runs on a GPU because on this box it did not fit
      in a sane budget.

Measured on a T4 with the identifiable task, mean-predictor baseline 0.635:
    regress (no flow, reference)  0.238
    x1 parameterisation           0.266
    velocity + Beta(1,1.5) t      0.407
    velocity + uniform t          0.620
which is what moved the repo to flow_parameterisation="x1".
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

_CACHE: dict = {}

JOB = '''
import json, torch, torch.nn.functional as F
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_TEST, STEPS, BS = 2048, 256, 2000, 48
SEED = %d

def run(shuffle):
    torch.manual_seed(SEED)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).to(dev).train()

    g = torch.Generator().manual_seed(SEED + 900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    W = torch.randn(cfg.obs_dim, cfg.action_chunk_size * cfg.action_dim, generator=g) * 0.05
    tgt = (obs @ W).view(n, cfg.action_chunk_size, cfg.action_dim)
    tro, trt = obs[:N_TRAIN].to(dev), tgt[:N_TRAIN].to(dev)
    teo, tet = obs[N_TRAIN:].to(dev), tgt[N_TRAIN:].to(dev)

    if shuffle:
        # Destroy the mapping on TRAIN only. Held-out targets stay correct, so a
        # model that learned real structure still scores well here and one that
        # merely memorised cannot.
        trt = trt[torch.randperm(N_TRAIN, generator=g).to(dev)]

    base = float(F.mse_loss(trt.mean(0, keepdim=True).expand_as(tet), tet))
    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=3e-4)
    for s in range(STEPS):
        i = (s * BS) %% (N_TRAIN - BS)
        L = brain.action_training_loss(tro[i:i+BS], trt[i:i+BS])["loss"]
        opt.zero_grad(); L.backward(); opt.step()

    brain.eval()
    with torch.no_grad():
        err = float(F.mse_loss(brain.generate_actions_flow_matching(teo).float(), tet.float()))
    return {"heldout": round(err, 5), "mean_baseline": round(base, 5)}

out = {"real": run(False), "shuffled": run(True)}
print("RESULT", json.dumps(out), flush=True)
open("/content/t102.json", "w").write(json.dumps(out))
'''


def _gpu(seed: int) -> dict:
    """One GPU job trains both arms; both hooks read the same result."""
    if seed in _CACHE:
        return _CACHE[seed]
    job = build_job(JOB % seed)
    r = submit(job, prefer="colab", est_hours=0.8, timeout_s=4200,
               fetch=["/content/t102.json"])
    if not r.ok:
        raise RuntimeError(f"GPU job failed on {r.backend}: {r.message}")
    art = r.artifacts.get("/content/t102.json")
    if art and Path(art).exists():
        data = json.loads(Path(art).read_text())
    else:
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")][-1]
        data = json.loads(line[len("RESULT "):])
    data["_backend"] = r.backend
    _CACHE[seed] = data
    return data


def _experiment(seed: int) -> dict:
    d = _gpu(seed)
    return {"structured_heldout": d["real"]["heldout"],
            "mean_baseline": d["real"]["mean_baseline"],
            "backend": d["_backend"]}


def _control(seed: int) -> dict:
    d = _gpu(seed)
    return {"shuffled_heldout": d["shuffled"]["heldout"]}


def _check(m: dict, c: dict) -> bool:
    adv = c["shuffled_heldout"] / max(m["structured_heldout"], 1e-9)
    m["heldout_structure_advantage"] = round(adv, 3)
    m["beats_mean_baseline"] = round(m["mean_baseline"] / max(m["structured_heldout"], 1e-9), 3)
    # Thresholds unchanged from pre-registered v2. Structure must generalise
    # better than destroyed structure, AND beat the do-nothing mean predictor.
    return adv >= 1.25 and m["beats_mean_baseline"] >= 1.1


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.02"], _experiment, _check, control_fn=_control, ledger=ledger)
