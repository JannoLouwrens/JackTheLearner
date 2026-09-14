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

SPREAD GATED 2026-09-13 (Review PROGRESS FOR THE BUILDER item 5, 2026-09-13).
STRENGTHENING, strictly additive: `MIN_BEAT_MEAN` is untouched and no arm is
removed. The Review's finding, and it is about THIS file rather than about the
model: "not knife-edge" is a claim about SPREAD, and for five weeks the gate
read only ABSENCE OF COLLAPSE — three booleans saying no LR fell below 1.15x
mean-prediction. `spread_ratio` was computed, recorded on the ledger, and read
by no conjunct, while the advantage swung 1.38 -> 6.80 across the 10x span.
A spec is not entitled to a title whose quantity it never gates.

REACHABILITY, pre-registered before the re-run (92nd audit B3 item 1 — the two
free numbers that would have caught LG.12's foreclosed knob with zero seeds):

  required to clear   spread_ratio <= 6.0
  recorded range      4.304 (attempt 1, T4, 2026-08-05, commit 1a69db6)
                      4.931 (attempt 2, P100, 2026-08-14, commit e29bd82)
  range the mechanism DOES produce: ~20.5 — the pre-clipping configuration of
                      2026-08-05 scored 5.443 / 13.195 / 0.643, documented
                      eleven lines above this block.

So the bar sits INSIDE the band this project has actually measured, with both
sides reachable: a configuration we ran clears it at 4.3-4.9, and a
configuration we ran fires it at ~20.5. It is not a bar fitted to the only
number we have. Headroom 6.0 / 4.931 = 1.217x — deliberately the same ratio the
Review chose for this bar, applied unchanged to T1.08's sibling conjunct.

VENUE PRICING OF ALL FOUR CONJUNCTS (builder, 2026-09-14, zero GPU; the
follow-on the T1.08 annotation of 09-13 flagged as unpriced). It corrects that
annotation on one point before it prices anything: it said computing this
spec's false-fail rate "needs the per-arm seed noise, which nobody has
measured", and no run of this spec can ever produce that number. `SEED = 0` is
a module constant inside JOB and the registry declares `seeds = 1`, so the
statistic is not SAMPLED across seeds at all. The only variation T1.07 has ever
exhibited is VENUE, and there are two observations of it on the ledger.

  within-venue determinism is demonstrated on one of the two venues: attempts 2
  and 3 (P100, 2026-08-14 `e29bd82` and 2026-09-13 `445b9e1`, a month and a code
  change apart) agree on EVERY metric and EVERY control metric to the recorded
  digit — 3.917 / 6.804 / 1.380, reference 7.605, spread 4.931, absurd 0.9162.
  The T4 has ONE observation; it is not known to be deterministic.

  code drift eliminated between attempt 1 (T4) and attempt 2 (P100), by the same
  method the T1.08 row used: this file's diff across `1a69db6..e29bd82` is 4
  insertions / 3 deletions and all of it is the artifact-path contract
  (`/content/` -> `JACK_OUT`); `UnifiedBrain.py`'s diff is two hunks, both in the
  PRETRAINED vision path (an `allow_vision_fallback` flag and a raise replacing a
  silent CNN downgrade), which `use_pretrained_vision=False` never reaches. The
  CNN fallback branch that this job does construct is byte-identical, so RNG
  consumption at build time is identical too. Same computation, both runs.
  NOTE this had to be done BY HAND: T1.07 declares no `IMPL_DEPS`, so a
  `UnifiedBrain.py` change does not stale its certificate (`protocol._impl_sha`
  hashes the module plus declared deps only — the gap T0.35 exists to count).

  "venue" here is three things confounded and NOT separable from two rows:
  Colab/T4/sm_75/Colab-torch versus Kaggle/P100/sm_60/torch 2.5.1+cu121.

  PER-ARM MOVEMENT, T4 -> P100:
    reference (plain MLP + Adam)   7.605 -> 7.605    x1.0000
    lr 1e-4                        4.088 -> 3.917    x0.958
    lr 1e-3                        1.298 -> 1.380    x1.063
    lr 3e-4                        5.585 -> 6.804    x1.218
    absurd lr 1.0                 0.0092 -> 0.9162   x99.59
  The one venue-INVARIANT arm is the one that is not the brain. The task, the
  data and plain Adam reproduce to four significant figures across both venues;
  everything that moves is inside the UnifiedBrain training path, and it moves
  most where the optimisation is most violent.

  MARGIN vs MEASURED MOVEMENT, at the live (P100) reading:
    worst_lr_advantage >= 1.15   1.3800   room x1.200   moved x1.063    34%
    spread_ratio       <= 6.00   4.9310   room x1.217   moved x1.146    69%
    reference_adv      >= 1.15   7.6050   room x6.613   moved x1.000     0%
    absurd_advantage   <  1.15   0.9162   room x1.255   moved x99.59  2024%
  (last column: the venue movement as a percentage of the remaining log-headroom)

THE FINDING, AND IT IS ABOUT THE CONTROL, NOT THE CLAIM. The bar flagged on
09-13 is the third-thinnest of the four. The binding one is the CONTROL: this
docstring says twenty lines up that if lr=1.0 clears MIN_BEAT_MEAN then "the bar
is too low to discriminate anything and the result is void" — and on the venue
the live certificate was bought on, lr=1.0 does not diverge at all
(`absurd_diverged` False) and lands at 0.9162x mean-prediction, 1.255x from
making this spec's own guard vacuous. On the other venue it read 0.0092. The
control's mechanism is 100x weaker on the P100 than on the T4, and its margin is
79.3x smaller than the one venue change we have on record.

That is a bound on what is KNOWN, not a probability: the direction happened to
run toward the bar, a third venue could run the other way, and n=2 supports no
rate. The honest statement is that this margin has never been shown to survive a
venue change and the only venue change on record would have destroyed it.

NOTHING MOVES ON THIS FINDING. `MIN_BEAT_MEAN`, `MAX_SPREAD_RATIO`, `LRS` and
`ABSURD_LR` are pre-registered and law 4 is unconditional; a margin with a
measured exposure is better governed than the same margin with an unmeasured
one, and which repair is right (pin the venue, add seeds inside JOB the way
T1.08 does, widen the control's separation, or accept it) is a design question,
not this file's to settle. WHAT IT WOULD COST to price the seed term, so the
next reader prices the measurement and not the re-read: attempt 3 ran 5 trainings
in 1673 s (~335 s each), so k seeds is ~0.465*k GPU-h — k=5 is 2.33 h.
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
MAX_SPREAD_RATIO = 6.0          # ... and the held-out spread across the span is
                                # the knife-edge quantity itself: see REACHABILITY
                                # in the docstring. Recorded 4.304 / 4.931;
                                # the unclipped configuration produced ~20.5.
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
    # Every LR in the 10x span beats mean-prediction; the held-out error across
    # that span stays inside a 6x band — the knife-edge quantity in the title,
    # gated from 2026-09-13; the reference arm proves the task is learnable at
    # all; the absurd LR must NOT clear the same bar.
    return (m["lrs_beating_baseline"] == len(LRS)
            and m["worst_lr_advantage"] >= MIN_BEAT_MEAN
            and m["spread_ratio"] <= MAX_SPREAD_RATIO
            and m["reference_advantage"] >= MIN_BEAT_MEAN
            and c["absurd_advantage"] < MIN_BEAT_MEAN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.07"], _experiment, _check, control_fn=_control, ledger=ledger)
