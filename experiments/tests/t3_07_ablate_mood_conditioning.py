"""T3.07 — Ablate mood conditioning: does mood reach BEHAVIOUR, or only text?

HYPOTHESIS (registry). Mood measurably changes behaviour, not just text.
Falsified by: identical action distributions across moods. Null: mood token
zeroed. Kills: MovementMoodCoupling as anything but cosmetics.

WHAT IS MEASURED, SAID PLAINLY. In the shipped brain, mood has exactly ONE
path to action: UnifiedBrain.act() takes emotional_state.pad_vector and pipes
the action head's output through MovementMoodCoupling.modulate_action
(UnifiedBrain.py, "Apply mood modulation" — the only call site). InnerMonologue
and the LLM persona read mood too, but those are TEXT; this spec gates the
action path the kills-clause names. The experiment composes with T2.12's
certificate: T2.12 proved four event regimes (thriving / struggling /
exploring / neglected) leave separable PAD trajectories; T3.07 asks whether
those same trajectories, driven through the shipped coupling, leave separable
ACTION streams. Together they test events -> mood -> behaviour. The PAD source
is imported from t2_12 (reference, don't transcribe): the identical
_trajectories rig its PASS certified, 4 regimes x 40 trajectories x 120 steps.

THE SHIPPED TRAINING, REPRODUCED — AND WHAT IT OMITS, WITH THE ARITHMETIC.
Untrained, the coupling is identity during locomotion BY INIT (speed_net and
style_net final layers are zeroed: multiplier == 1.0 and bias == 0 for every
PAD — MovementMoodCoupling.__init__), so the untrained module cannot pass and
testing it alone would be arithmetic, not measurement. The system's shipped
training is TrainingPipeline.train_phase8 Phase 8.2: 30 epochs x 5 updates,
target_pad ~ randn(3).clamp(-1,1), loss = (0.7 + 0.6*sigmoid(speed_net(pad))
- (1 + 0.3*pad[1]))^2, AdamW lr=3e-4 wd=1e-4 eps=1e-5 (make_optimizer). This
file reproduces those loss lines verbatim and OMITS the 100-step env rollout
that wraps them, because the rollout has no gradient path to the loss: the
loss reads speed_net(target_pad) only, and the rollout's actions are computed
under no_grad and never touched again — it advances RNG and the wall clock,
nothing else (the decorative-critic disease, T2.01 v5). Consequence, stated
so the reader needn't derive it: the shipped training touches ONLY speed_net
(162 params); style_net and posture_net never receive a gradient anywhere in
the repo. Post-training, mood's entire locomotion footprint is whatever
speed map those 150 single-sample AdamW steps at lr 3e-4 carve out of a
zero-initialised head — and the pre-registration smoke measured that span to
be SMALL (well under the designed 0.6 range; per-coordinate Adam movement is
bounded by steps*lr = 0.045, so this is arithmetic as much as measurement).
A weak shipped training is a property of the SYSTEM UNDER TEST, so it must
be able to produce a FAIL, not a VOID — which forces the attribution
question: is a near-identity map the shipped system's defect, or this
harness's? The rig answers it with a MUST-SUCCEED reference arm (the T1.02
plain-MSE-reference lesson, and PG.7's corollary that a probe which must
succeed belongs beside the probes that must fail): the same architecture and
the same loss trained with an adequate budget (batch 256, lr 1e-2, 500
steps — a 162-param MSE fit). Reference span >= 0.30 proves the map is
learnable BY THIS RIG; only then may the shipped arm's weakness be read as
the system's. Reference fails -> VOID, nothing attributable.

DESIGN — paired base, open loop. Per (regime, trajectory): an independent
base action stream (120 x 17, 0.5*randn clipped to [-1,1] — the scale the
module's own _test uses) is modulated step-by-step by that trajectory's PAD
(is_idle=False, the locomotion path). Episode features: per-joint mean (17) +
per-joint std (17) + mean and std over time of the mean |action| (2) = 36-D.
Nearest-centroid 4-way classification, even trajectories train / odd test
(t2_12's protocol on this spec's own feature space — the classifier operates
on precomputed vectors, so t2_12's trajectory-bound helper is not reusable
without editing a PASSed spec's file, which would stale its certificate).
Base streams are drawn per-episode rather than shared across regimes ON
PURPOSE: with a shared stream the bypass control's features would be
class-identical by construction and the control could never fail
("assert at a point where the state can still tell the two outcomes apart").
Open loop measures the coupling's DIRECT effect on the action distribution —
the quantity the registry's falsified_by names. Closed-loop compounding
(a 1.3x-torque body visits different states) is out of scope here and can
only add divergence on top of what this measures.

NULL (registry): mood token zeroed — the same episodes re-modulated with
pad = 0 through the LIVE path, real labels kept. Chance by construction
unless the rig leaks. CONTROL (declared in the registry, must fail): the
ablation itself — the same episodes with the coupling BYPASSED (raw base
actions, real labels). Mood cannot reach the features except through the
coupling, so bypass accuracy must sit at chance; if it clears, the measured
separability is not attributable to mood conditioning and the run is VOID,
not evidence. CONDITION on this control: it certifies non-leakage of the
BASE-STREAM rig (seed arithmetic, feature code); it cannot certify the PAD
source, which is T2.12's certificate's job.

GATES, all exogenous (chance = 1/4; 80 test episodes -> sd at chance
= sqrt(.25*.75/80) ~= 0.048; nothing calibrated from a pilot):
  MIN_ACC        0.45  chance + ~4.1 sd, gated on the WORST seed
  MAX_NULL_ACC   0.40  chance + ~3.1 sd; any seed above -> VOID (leak)
  MAX_CTRL_ACC   0.40  same, for the bypass control -> VOID (not attributable)
  MIN_PAD_SEP    t2_12.MIN_ACC (0.80), t2_12's own registered bar: the drawn
                 PAD trajectories must be separable THIS run or mood->action
                 transmission cannot be tested -> VOID (referenced constant,
                 with t2_12's file in IMPL_DEPS so the loan goes stale loudly)
  MIN_REF_SPAN   0.30  the REFERENCE arm's multiplier(A=+1) - multiplier(A=-1);
                 the designed span is 0.6, so 0.30 = half. Below -> VOID: the
                 rig cannot demonstrate the map is learnable, so a weak
                 shipped arm cannot be attributed to the shipped system.
                 The SHIPPED arm's span is reported beside it, ungated — its
                 weakness is a candidate finding, never a candidate excuse.
Headline metric action_dist_divergence := worst-seed accuracy - chance, so 0
reads exactly as the registry's "identical action distributions".

DIAGNOSTICS, reported ungated: shipped-arm speed span and training-loss
descent; per-regime mean arousal and mean speed multiplier (which moods
moved); reference-trained-module accuracy (would the claim hold if the
shipped training CONVERGED? — separates "the training is too weak" from
"even a converged speed map cannot transmit these regimes' differences");
idle-path accuracy (is_idle=True — the hand-initialised posture priors, the
cosmetic channel); untrained-module accuracy (identity by init — should sit
at chance with the control); post-actuator-clip accuracy at |a| <= 0.4
(Humanoid-v5 ctrlrange, transcribed for a diagnostic only — MuJoCo clips
ctrl silently, so a multiplier on an already-saturated action changes
nothing physical; if the gated accuracy passes while this reads chance,
that gap is the next spec).

WHY THIS RUNS LOCALLY DESPITE A gpu<20min BUDGET. The whole rig is the
1,542-param coupling + T2.12's PAD generator; T2.12's recorded 3-seed run
took 38.6 s on this box. A GPU submission would spend its entire runtime on
clone+provision overhead (~7 min) to accelerate ~2 min of CPU arithmetic —
"USE THE GPU" is about training steps a GPU speeds up, and there are none
here. The budget stays a ceiling; run.py's derived timeout (1200 s x seeds
x 2) fits with two orders of magnitude to spare.
"""

from __future__ import annotations

import copy

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .t2_12_emotion_separability import (MIN_ACC as T212_MIN_ACC,
                                         _centroid_accuracy, _trajectories)

# The claim is about the shipped mood->action stack: the coupling, its PAD
# source, the pipeline that trains it, the brain that wires it, and the PAD
# rig this spec borrows. Any of them moving must stale this certificate.
IMPL_DEPS = ["MovementMoodCoupling.py", "EmotionalState.py",
             "TrainingPipeline.py", "UnifiedBrain.py",
             "experiments/tests/t2_12_emotion_separability.py"]

SEEDS = [0, 1, 2]
SMOKE_SEED = 90

CHANCE = 0.25
MIN_ACC = 0.45            # worst seed; chance + ~4.1 sd of the 80-episode test
MAX_NULL_ACC = 0.40       # any seed above -> VOID (zeroed-token arm leaks)
MAX_CTRL_ACC = 0.40       # any seed above -> VOID (bypass arm leaks)
MIN_REF_SPAN = 0.30       # reference arm below half the designed 0.6 -> VOID
CLIP_DIAG = 0.4           # Humanoid-v5 ctrlrange — DIAGNOSTIC only, ungated

BASE_SCALE = 0.5          # base action stream: 0.5*randn, clipped to [-1, 1]
N_EPOCHS, N_INNER = 30, 5  # TrainingPipeline.train_phase8: min(epochs,30) x 5
LR, WD, EPS = 3e-4, 1e-4, 1e-5   # make_optimizer: config.learning_rate, AdamW
REF_BATCH, REF_STEPS, REF_LR = 256, 500, 1e-2   # the must-succeed reference


def _build_module():
    """The coupling exactly as UnifiedBrain builds it (UnifiedBrain.py, the
    enable_movement_mood_coupling branch) — config values referenced from the
    live UnifiedBrainConfig, not transcribed."""
    from MovementMoodCoupling import MovementMoodConfig, MovementMoodCoupling
    from UnifiedBrain import UnifiedBrainConfig
    ucfg = UnifiedBrainConfig()
    return MovementMoodCoupling(MovementMoodConfig(
        action_dim=ucfg.action_dim,
        max_speed_mod=ucfg.max_speed_modulation,
        max_style_bias=ucfg.max_style_bias))


def _train_shipped(module, seed: int) -> dict:
    """Phase 8.2's gradient-relevant lines, verbatim (see docstring for why
    the env rollout is omitted: no gradient path to the loss)."""
    import torch
    torch.manual_seed(seed * 31 + 7)
    opt = torch.optim.AdamW([{"params": list(module.parameters()), "lr": LR}],
                            weight_decay=WD, eps=EPS)
    losses = []
    for _epoch in range(N_EPOCHS):
        for _ in range(N_INNER):
            target_pad = torch.randn(3).clamp(-1, 1)
            opt.zero_grad()
            speed_raw = module.speed_net(target_pad.unsqueeze(0))
            speed = 0.7 + 0.6 * torch.sigmoid(speed_raw).squeeze()
            target_speed = 1.0 + 0.3 * target_pad[1]
            style_loss = (speed - target_speed).pow(2)
            style_loss.backward()
            opt.step()
            losses.append(float(style_loss.detach()))
    q = max(1, len(losses) // 4)
    return {"loss_first": float(np.mean(losses[:q])),
            "loss_last": float(np.mean(losses[-q:]))}


def _train_reference(module, seed: int) -> None:
    """The must-succeed arm: same architecture, same loss, adequate budget
    (full-batch MSE on 162 params). Proves the map is LEARNABLE by this rig,
    so the shipped arm's weakness is attributable to the shipped training."""
    import torch
    torch.manual_seed(seed * 47 + 3)
    pads = torch.randn(REF_BATCH, 3).clamp(-1, 1)
    target = 1.0 + 0.3 * pads[:, 1]
    opt = torch.optim.AdamW([{"params": list(module.parameters()),
                              "lr": REF_LR}], weight_decay=WD, eps=EPS)
    for _ in range(REF_STEPS):
        opt.zero_grad()
        speed = 0.7 + 0.6 * torch.sigmoid(module.speed_net(pads)).squeeze(1)
        (speed - target).pow(2).mean().backward()
        opt.step()


def _features(stream: np.ndarray) -> list:
    """36-D episode features of an action stream [T, action_dim]."""
    per_joint_mean = stream.mean(0)
    per_joint_std = stream.std(0)
    mag = np.abs(stream).mean(1)
    return (list(per_joint_mean) + list(per_joint_std)
            + [float(mag.mean()), float(mag.std())])


def _centroid_acc(feats: dict) -> float:
    """t2_12's classification protocol on precomputed feature vectors:
    z-score on train stats, nearest class centroid, even train / odd test."""
    train = [(ri, f) for ri, fs in feats.items()
             for i, f in enumerate(fs) if i % 2 == 0]
    test = [(ri, f) for ri, fs in feats.items()
            for i, f in enumerate(fs) if i % 2 == 1]
    nfeat = len(train[0][1])
    mu = [sum(f[j] for _, f in train) / len(train) for j in range(nfeat)]
    sd = [max(1e-9, (sum((f[j] - mu[j]) ** 2 for _, f in train)
                     / len(train)) ** 0.5) for j in range(nfeat)]
    z = lambda f: [(f[j] - mu[j]) / sd[j] for j in range(nfeat)]
    cents = {}
    for ri in feats:
        fs = [z(f) for l, f in train if l == ri]
        cents[ri] = [sum(f[j] for f in fs) / len(fs) for j in range(nfeat)]
    hits = 0
    for ri, f in test:
        zf = z(f)
        pred = min(cents, key=lambda c: sum((zf[j] - cents[c][j]) ** 2
                                            for j in range(nfeat)))
        hits += pred == ri
    return hits / len(test)


def _run_seed(seed: int) -> dict:
    import torch

    # 1. The PAD source T2.12 certified, and its rig-health gate.
    trajs, _baseline = _trajectories(seed)
    pad_sep = _centroid_accuracy(trajs)

    # 2. The shipped module, untrained snapshot kept, then the shipped
    #    training — plus the must-succeed reference arm from the same init.
    torch.manual_seed(seed * 31 + 7)
    module = _build_module().eval()
    untrained = copy.deepcopy(module)
    reference = copy.deepcopy(module)
    train_stats = _train_shipped(module.train(), seed)
    module.eval()
    _train_reference(reference.train(), seed)
    reference.eval()

    def _span(mod):
        return float(mod.get_speed_multiplier(torch.tensor([0.0, 1.0, 0.0]))
                     - mod.get_speed_multiplier(torch.tensor([0.0, -1.0, 0.0])))

    span, ref_span = _span(module), _span(reference)

    # 3. Episodes: independent base stream per (regime, trajectory); one
    #    batched modulate_action call per arm (pad rows align with steps).
    arms = {k: {} for k in ("exp", "null", "ctrl", "idle", "untrained",
                            "clip", "ref")}
    arousal_by_regime, speed_by_regime = {}, {}
    with torch.no_grad():
        for ri, trs in trajs.items():
            for k in arms:
                arms[k][ri] = []
            ar_acc, sp_acc = [], []
            for ti, tr in enumerate(trs):
                rng = np.random.RandomState(
                    (seed * 611_953 + ri * 10_007 + ti * 97 + 13) % 2**32)
                base = np.clip(BASE_SCALE * rng.randn(len(tr),
                                                      module.config.action_dim),
                               -1.0, 1.0).astype(np.float32)
                base_t = torch.from_numpy(base)
                pads = torch.tensor(tr, dtype=torch.float32)      # [T, 3]
                zero = torch.zeros_like(pads)
                mod = module.modulate_action(base_t, pads).numpy()
                arms["exp"][ri].append(_features(mod))
                arms["null"][ri].append(_features(
                    module.modulate_action(base_t, zero).numpy()))
                arms["ctrl"][ri].append(_features(base))
                arms["idle"][ri].append(_features(
                    module.modulate_action(base_t, pads, is_idle=True).numpy()))
                arms["untrained"][ri].append(_features(
                    untrained.modulate_action(base_t, pads).numpy()))
                arms["clip"][ri].append(_features(
                    np.clip(mod, -CLIP_DIAG, CLIP_DIAG)))
                arms["ref"][ri].append(_features(
                    reference.modulate_action(base_t, pads).numpy()))
                ar_acc.append(float(pads[:, 1].mean()))
                sp_acc.append(float(
                    module._compute_speed_multiplier(pads).mean()))
            arousal_by_regime[ri] = round(float(np.mean(ar_acc)), 4)
            speed_by_regime[ri] = round(float(np.mean(sp_acc)), 4)

    acc = {k: round(_centroid_acc(v), 4) for k, v in arms.items()}
    return {
        "pad_separability": round(pad_sep, 4),
        "speed_span": round(span, 4),
        "ref_speed_span": round(ref_span, 4),
        "loss_first": round(train_stats["loss_first"], 6),
        "loss_last": round(train_stats["loss_last"], 6),
        "acc": acc,
        "arousal_by_regime": arousal_by_regime,
        "speed_by_regime": speed_by_regime,
        "n_params": int(sum(p.numel() for p in module.parameters())),
    }


# All seeds are computed on the first call and shared between _experiment and
# _control (the T4.02 cache pattern): worst-seed gates need every seed's
# number in one place, and run_spec calls once per seed.
_CACHE: dict = {}


def _rows() -> list:
    if not _CACHE:
        _CACHE["rows"] = [_run_seed(s) for s in SEEDS]
    return _CACHE["rows"]


def _experiment(seed: int) -> dict:
    rows = _rows()
    accs = [r["acc"]["exp"] for r in rows]
    return {
        "action_dist_divergence": round(min(accs) - CHANCE, 4),
        "separability_acc_min": min(accs),
        "acc_per_seed": accs,
        "chance": CHANCE,
        "pad_separability_min": min(r["pad_separability"] for r in rows),
        "ref_speed_span_min": min(r["ref_speed_span"] for r in rows),
        "shipped_speed_span_per_seed": [r["speed_span"] for r in rows],
        "train_loss_decreased_all": float(all(
            r["loss_last"] < r["loss_first"] for r in rows)),
        "ref_acc_per_seed": [r["acc"]["ref"] for r in rows],
        "idle_acc_per_seed": [r["acc"]["idle"] for r in rows],
        "untrained_acc_per_seed": [r["acc"]["untrained"] for r in rows],
        "clip_acc_per_seed": [r["acc"]["clip"] for r in rows],
        "arousal_by_regime_per_seed": [r["arousal_by_regime"] for r in rows],
        "speed_by_regime_per_seed": [r["speed_by_regime"] for r in rows],
        "n_params": rows[0]["n_params"],
    }


def _control(seed: int) -> dict:
    rows = _rows()
    return {
        "ctrl_bypass_acc_max": max(r["acc"]["ctrl"] for r in rows),
        "null_zeroed_acc_max": max(r["acc"]["null"] for r in rows),
        "ctrl_acc_per_seed": [r["acc"]["ctrl"] for r in rows],
        "null_acc_per_seed": [r["acc"]["null"] for r in rows],
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if m["pad_separability_min"] < T212_MIN_ACC:
        return Status.VOID   # mood states indistinct this draw — nothing to transmit
    if m["ref_speed_span_min"] < MIN_REF_SPAN:
        return Status.VOID   # rig cannot learn the map — weakness unattributable
    if c["ctrl_bypass_acc_max"] > MAX_CTRL_ACC:
        return Status.VOID   # separability without the coupling — leak
    if c["null_zeroed_acc_max"] > MAX_NULL_ACC:
        return Status.VOID   # zeroed token still separable — leak
    # The claim, on the worst seed. The shipped arm's own training strength is
    # part of the system under test: a map too weak to separate moods is the
    # hypothesis FAILING, never a VOID (the reference arm above holds the
    # attribution).
    return m["separability_acc_min"] >= MIN_ACC


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T3.07"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # One full production seed (the rig is small enough that the smoke IS
        # a production-shaped run: full 40 trajectories, full 120 steps, all
        # six arms — the seed-derivation extremes included, per the LC.03
        # smoke lesson). Asserts rig mechanics only, never the science.
        row = _run_seed(SMOKE_SEED)
        print(json.dumps(row, indent=1))
        assert row["loss_last"] < row["loss_first"], "shipped training did not descend"
        assert row["ref_speed_span"] > MIN_REF_SPAN, \
            "reference arm cannot learn the map — the rig, not the system, is broken"
        assert abs(row["acc"]["untrained"] - row["acc"]["ctrl"]) < 1e-9, \
            "untrained module is not identity during locomotion — init changed?"
        assert all(0.0 <= a <= 1.0 for a in row["acc"].values())
        print("SMOKE OK")
    else:
        run()
