"""T1.02 — does the architecture exploit structure, or only memorise?

Third design. The first two failed for reasons that were mine, not the model's,
and the history is kept because it is the most instructive thing in this tier.

  v1  measured training FIT on one batch -> 0.999. A 58M network memorises 8
      arbitrary pairs whether or not a mapping exists, so fit measures capacity.
  v2  measured generalisation, correctly, but drew 64 training samples for an
      obs_dim=348 input. Underdetermined; nothing could pass it. The tell was
      beats_mean_baseline=0.415 — the trained model lost to predicting the mean.

So v3 changes two things.

FIRST, the task is made identifiable: 2048 training samples and a true map of
rank 8. Rank 8 is not a convenience — actions genuinely depend on a
low-dimensional function of proprioception, so this is closer to the real problem
than a full-rank random map ever was.

SECOND, and this is the part worth carrying to every later spec: a REFERENCE ARM.
A plain MLP trained with MSE, no flow matching anywhere. It exists to answer the
question v2 could not — when the model fails, is the model wrong or is the task
impossible? If the simplest possible learner also fails, the run is VOID, not a
verdict on the architecture. Without this arm an unlearnable task looks exactly
like a broken model, and I would have spent GPU hours "fixing" working code.

Three arms:
  structured  the brain on a real mapping        -> must generalise
  shuffled    the brain on permuted targets      -> must NOT generalise
  regress     a plain MLP on the real mapping    -> must succeed, or the run is void
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# Pre-registered, before any v3 result was seen.
MIN_REFERENCE_GAIN = 1.5    # below this the task is unlearnable -> void, not FAIL
MIN_STRUCTURE_ADV = 1.25    # structured must beat shuffled by this
MIN_BEATS_MEAN = 1.10       # structured must beat the do-nothing baseline

JOB = r'''
import json, torch, torch.nn.functional as F
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_TEST, STEPS, BS, RANK = 2048, 512, 1500, 64, 8


def make_task(cfg, seed):
    """A rank-8 map: actions depend on a low-dimensional function of the state.

    Identifiable from 2048 samples, unlike v2's full-rank map from 64.
    """
    g = torch.Generator().manual_seed(seed + 900)
    n = N_TRAIN + N_TEST
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    A = torch.randn(cfg.obs_dim, RANK, generator=g) / (cfg.obs_dim ** 0.5)
    B = torch.randn(RANK, cfg.action_chunk_size * cfg.action_dim, generator=g)
    tgt = (torch.tanh(obs @ A) @ B).view(n, cfg.action_chunk_size, cfg.action_dim) * 0.3
    return obs.to(DEV), tgt.to(DEV)


def brain_arm(seed, shuffle):
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).to(DEV).train()

    obs, tgt = make_task(cfg, seed)
    tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]
    if shuffle:
        # Destroy the mapping on the TRAINING set only; held-out targets stay
        # honest, so a model that learned real structure still scores well.
        gp = torch.Generator().manual_seed(seed + 7)
        tr_t = tr_t[torch.randperm(N_TRAIN, generator=gp).to(DEV)]

    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad], lr=3e-4)
    for step in range(STEPS):
        i = (step * BS) % (N_TRAIN - BS)
        loss = brain.action_training_loss(tr_o[i:i + BS], tr_t[i:i + BS])["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    brain.eval()
    with torch.no_grad():
        pred = brain.generate_actions_flow_matching(te_o)
        err = float(F.mse_loss(pred.float(), te_t.float()))
        mean_err = float(F.mse_loss(tr_t.mean(0, keepdim=True).expand_as(te_t).float(),
                                    te_t.float()))
    return {"heldout": round(err, 5), "mean_baseline": round(mean_err, 5),
            "train_loss": round(float(loss), 5)}


def reference_arm(seed):
    """Plain MLP, plain MSE. No flow matching. The task-validity check."""
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    obs, tgt = make_task(cfg, seed)
    out_dim = cfg.action_chunk_size * cfg.action_dim
    net = torch.nn.Sequential(
        torch.nn.Linear(cfg.obs_dim, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, out_dim)).to(DEV)
    tr_o, tr_t, te_o, te_t = obs[:N_TRAIN], tgt[:N_TRAIN], obs[N_TRAIN:], tgt[N_TRAIN:]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(STEPS):
        i = (step * BS) % (N_TRAIN - BS)
        loss = F.mse_loss(net(tr_o[i:i + BS]).view(-1, cfg.action_chunk_size, cfg.action_dim),
                          tr_t[i:i + BS])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        err = float(F.mse_loss(net(te_o).view_as(te_t), te_t))
        mean_err = float(F.mse_loss(tr_t.mean(0, keepdim=True).expand_as(te_t), te_t))
    return {"heldout": round(err, 5), "mean_baseline": round(mean_err, 5)}


SEED = 0
out = {"device": DEV,
       "reference": reference_arm(SEED),
       "structured": brain_arm(SEED, shuffle=False),
       "shuffled": brain_arm(SEED, shuffle=True)}
json.dump(out, open("/content/out.json", "w"), indent=2)
print("RESULT", json.dumps(out), flush=True)
'''


def _submit(seed: int) -> dict:
    job = build_job(JOB)
    r = submit(job, prefer="colab", est_hours=0.4, timeout_s=2400,
               fetch=["/content/out.json"])
    if not r.ok:
        raise RuntimeError(f"GPU job failed on {r.backend}: {r.message}")
    path = r.artifacts.get("/content/out.json") or next(iter(r.artifacts.values()), None)
    if path:
        return json.loads(Path(path).read_text())
    # Colab buffers stdout to the end; the printed line is the fallback.
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError("job produced no result")


_CACHE: dict = {}


def _experiment(seed: int) -> dict:
    _CACHE.update(_submit(seed))
    s, ref = _CACHE["structured"], _CACHE["reference"]
    return {"structured_heldout": s["heldout"],
            "mean_baseline": s["mean_baseline"],
            "reference_heldout": ref["heldout"],
            "device": _CACHE["device"]}


def _control(seed: int) -> dict:
    sh = _CACHE["shuffled"]
    return {"shuffled_heldout": sh["heldout"], "shuffled_train_loss": sh["train_loss"]}


def _check(m: dict, c: dict) -> bool:
    ref_gain = m["mean_baseline"] / max(m["reference_heldout"], 1e-9)
    m["reference_gain"] = round(ref_gain, 3)
    m["heldout_structure_advantage"] = round(
        c["shuffled_heldout"] / max(m["structured_heldout"], 1e-9), 3)
    m["beats_mean_baseline"] = round(
        m["mean_baseline"] / max(m["structured_heldout"], 1e-9), 3)

    if ref_gain < MIN_REFERENCE_GAIN:
        # The task, not the model. Do not report this as an architecture failure.
        m["verdict"] = ("VOID — a plain MLP could not learn this task either "
                        f"(reference_gain={ref_gain:.2f} < {MIN_REFERENCE_GAIN}). "
                        "The task is unidentifiable; redesign it, do not blame the model.")
        return False

    return (m["heldout_structure_advantage"] >= MIN_STRUCTURE_ADV
            and m["beats_mean_baseline"] >= MIN_BEATS_MEAN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.02"], _experiment, _check, control_fn=_control, ledger=ledger)
