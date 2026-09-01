"""D1.0 — Control-path bakeoff: who does motor control (D1's four permitted arms).

PROVENANCE. D1 ("does the 57M trunk stay in the control path?") sat OPEN for
twenty days blocking 35+ specs, was armed 2026-08-24 under SYSTEM.md rule 3 as
amended, and its default FIRED 2026-09-01: the PLASTIC-ONLY decree stands
verbatim, option A (freeze the trunk) is STRUCK as unconstitutional, and the
four permitted arms go to this registered bakeoff (DECISIONS_RESOLVED.md D1).
The registry entry is the authority. docs/research/D1_CONTROL_ARCHITECTURE.md
designed a six-arm bakeoff BEFORE the strike — its A0 (SB3 reference), A4
(frozen pretrained trunk) and A5 (frozen trunk + flow expert) are NOT arms
here; what survives of it is the repairs table (R1–R6), the controller shapes
(2006.05990), and the harness-vs-architecture reading recorded below.

THE FOUR ARMS, mapped to code. Every arm trains under the ONE PPO loop this
repo owns (`TrainingPipeline.collect_rollout_vec` + `rl_update` — T0.14/T0.16
guard its mode discipline), with identical N_ENVS, rollout length, minibatch,
epochs, learning rate and log_std init. Matched env-steps AND matched
optimiser-steps therefore hold BY CONSTRUCTION (T2.02's Rank-2 scar: matched
env-steps alone hid a 16x optimiser-step gap), and both counts are recorded
per arm (repair R5). Injection surface: `tp.model` is swapped and
`tp.project_obs` overridden to identity, so each arm owns its FULL input path
from the normalised raw 348-d observation; no arm shares parameters with
another.

  C  (`c_e2e`)   — end-to-end: stock TrainingPipeline, the 57M trunk trained
                   by PPO exactly as T2.01 v5 ran it, at MORE steps than v5's
                   704,512/seed. Reclassified UNTESTED, not refuted (the v5
                   plateau was measured pre-T0.25 on a broken estimator's
                   history; v5 itself reached 2.67 sigma post-fix).
  B  (`b_split`) — split value/policy: policy = own 348->256 projection stem +
                   UnifiedBrain trunk + action head (PPO pg gradient); value =
                   a SEPARATE MLP critic 348->256->256->1 reading the raw
                   normalised observation. Zero shared parameters, so value
                   gradient cannot touch the policy trunk (T2.00 measured the
                   vf/pg interference this isolates).
  A' (`aprime`)  — a small dedicated control head that LEARNS: policy MLP
                   512->128->128->17 and value MLP 512->256->256->1 read the
                   trunk's `cls_features` through a STOP-GRADIENT (detach), so
                   PPO trains only the head + log_std; the trunk (own stem +
                   UnifiedBrain) stays PLASTIC under its other objective — a
                   next-observation prediction head trained on the same
                   rollouts by a separate optimiser, 1 epoch per PPO
                   iteration. The doc's A4 pretrained this trunk on mocap;
                   that lane is FORECLOSED here (the mocap loader's URLs 404
                   and it fabricates sinusoids — GOAL.md context, 2026-08-09),
                   so the trunk's "other objective" is the cheapest one that
                   exists in-kernel: predict o_{t+1} from o_t on lived data.
  D  (`d_mlp`)   — transformer out of the control path: SeparateActorCritic,
                   policy 348->128->128->17 (tanh, last layer init 100x
                   small), value 348->256->256->1 (tanh), per Andrychowicz
                   2006.05990. No trunk anywhere in the motor path.

PRE-REGISTERED DECISION RULE (the registry's, spelled out; verdicts name
their branch WITH the comparison — the BA.03 lesson):
  VOID   — any arm's step count < MIN_STEP_MATCH x STEP_TARGET (comparison
           not at matched experience; raise the cap, do not compare); OR any
           arm misses the 3-sigma learning gate vs the random null (two
           non-learners cannot arbitrate — T2.02's precedent; the verdict
           records WHICH arms learned, per falsified_by); OR any UNTRAINED
           twin clears 3 sigma (the gate would be measuring architecture
           bias, not learning — T2.02's untrained MLP hit 2.74 sigma, hence
           twins per arm, not just random).
  FAIL   — no arm beats the runner-up by >= WIN_MARGIN_SIGMA x pooled seed
           spread: a TIE, resolved to the cheapest arm by PPO-trainable
           parameters. A real result: the control-path choice does not matter
           at this scale.
  VOID (SPLIT-PENDING) — a margin exists but the runner-up's final-third
           curve slope is positive AND its extrapolated crossover with the
           winner's final-third mean lands inside CROSSOVER_MULT x
           STEP_TARGET. The owner's convergence check: no winner while the
           runner-up is still closing.
  PASS   — one arm wins by the margin and survives the convergence check.
           If that arm is D, the verdict RECORDS the pre-registered cost (a D
           win forecloses DP.02 — private control representations, the "two
           brains wearing one wrapper" signature) and the adoption routes
           through the Review, per the registry's kills field.

KNOWN RISK, pre-registered rather than discovered: UB.10 measured RECIPE
SENSITIVITY — no single uniform recipe trained all six of its matched arms.
This bakeoff deliberately runs all four arms under the trunk-tuned recipe
(that IS the matched condition), so an MLP-arm gate miss is possible. Context
for reading it: T2.02's SB3 MLP scored 530 / 7.11 sigma under SB3's own
recipe, so a D gate-miss here is evidence about the shared recipe, not about
MLPs — the VOID verdict must say so and the repair is a recipe question for
the Review, not a silent re-roll.

GATES FROZEN 2026-09-01 — PILOT RECORD (kernel jack-ladder-1788225926,
Kaggle P100, wall 27.1 min, 0.50 h charged to 2026-W35, artifact
/data/d10_pilot.json, dispatched at eda570d). The per-condition rule of
2026-08-31 is satisfied: all four arms appear in the record, no arm assumed
covered by a sibling.

  arm       steps/s   h/seed@750k  wiring (asserted live on the P100)
  aprime     176.55      1.18      trunk UNCHANGED by PPO, CHANGED by aux;
                                   head updated (ppo 281,763 / aux 57.5M)
  b_split    105.09      1.98      trunk AND critic updated (ppo 57.8M)
  c_e2e      104.94      1.99      trunk updated (ppo 57.4M)
  d_mlp     1390.37      0.15      pol AND val updated (ppo 530,339)
  All losses finite on every arm.

STEP_TARGET stays 750,000 (legal floor 704,513). MINUTES_CAP frozen at
1.26x-1.67x measured raw: aprime 90, b_split 150, c_e2e 150, d_mlp 15
minutes per arm-seed.

THE PRE-REGISTERED ESCALATION BRANCH FIRED — no <=2-submission split fits
at ANY legal STEP_TARGET. The arithmetic, from measured throughput: total
compute 15.9 h across 4 arms x 3 seeds; the best possible 2-way split by
max side is {aprime,b_split}/{c_e2e,d_mlp} = 9.49 h / 6.41 h at 750k, and
8.91 h even at the 704,513 floor — both ABOVE the 8.89 h child timeout
(timeout_s=32000), with zero margin, before caps or kernel overhead. Per
the clause this section pre-registered, the registry's "one Kaggle
submission per arm-pair at most" line MOVES LOUDLY, here and in the
registry note: `_KERNEL_SPLIT` is THREE kernels — ("aprime","d_mlp") at
5.25 h, ("b_split",) at 7.5 h, ("c_e2e",) at 7.5 h by caps — every kernel
>= 1.39 h under the child timeout, worst-case charge 20.25 h against ~28 h
free in the live GPU week. This widens a logistics budget in the open; no
gate moved: STEP_TARGET, both sigma bars, MIN_STEP_MATCH and every control
are exactly as registered.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# ---------------------------------------------------------------- constants --
SEEDS = [0, 1, 2]
ARMS = ["aprime", "b_split", "c_e2e", "d_mlp"]     # canonical order everywhere

N_ENVS = 32                 # T2.01 v4/v5's measured-throughput configuration
ROLLOUT_STEPS = 128         # -> 4096-sample PPO batches, minibatch 512, 5 epochs
EVAL_EPISODES = 5
RANDOM_EPISODES = 10
EVAL_SEED_BASE = 9000       # repair R6: identical eval env seeds across arms,
                            # twins and the random null — paired evaluation.

MIN_LEARN_SIGMA = 3.0       # learning gate, every arm, vs random
WIN_MARGIN_SIGMA = 1.5      # winner over runner-up, pooled seed spread
MIN_STEP_MATCH = 0.9        # any arm below this fraction of STEP_TARGET -> VOID
CROSSOVER_MULT = 3.0        # convergence check horizon, x STEP_TARGET

# FROZEN 2026-09-01 from the PILOT RECORD in the docstring (kernel
# jack-ladder-1788225926, measured per-arm steps/s on the real P100).
# STEP_TARGET must stay strictly above T2.01 v5's 704,512 env-steps/seed
# (arm C's "more steps" clause).
STEP_TARGET = 750_000
MINUTES_CAP = {"aprime": 90, "b_split": 150, "c_e2e": 150, "d_mlp": 15}
_KERNEL_SPLIT = (("aprime", "d_mlp"), ("b_split",), ("c_e2e",))
                            # THREE kernels — the pre-registered escalation
                            # branch fired (no <=2-submission split fits at
                            # any legal STEP_TARGET; arithmetic in the
                            # docstring). Caps per kernel: 5.25/7.5/7.5 h,
                            # all >=1.39 h under timeout_s=32000.

PILOT_MINUTES_PER_ARM = 6   # timed throughput window, after the wiring checks

_GATES_FROZEN = True        # frozen 2026-09-01; PILOT RECORD in docstring
_PILOT_OWED = (
    "The pilot freezes STEP_TARGET, MINUTES_CAP and _KERNEL_SPLIT from "
    "measured per-arm env-steps/s on the real Kaggle GPU, and asserts each "
    "arm's gradient-isolation wiring (A' trunk untouched by PPO, changed by "
    "its aux objective; B critic/trunk both live; C trunk live; D both MLPs "
    "live). It can succeed because every arm composes components already "
    "measured to train on this hardware: the identical trunk+loop ran T2.01 "
    "v4/v5 at ~106-128 env-steps/s on the P100, and the MLP shapes are "
    "2006.05990's, trained by the same loop."
)
_PILOT_ARTIFACT = "/data/d10_pilot.json"


# =============================================================================
# KERNEL SIDE — everything below `_lazy_torch` runs on the GPU VM (build_job
# clones this repo, so the kernel is 3 lines invoking this module) and also
# locally for the smoke lane. Heavy imports stay inside functions so that
# registry scans importing this module never pay for torch.
# =============================================================================

def _lazy_torch():
    import numpy as np
    import torch
    import torch.nn as nn
    return np, torch, nn


def _identity_project(obs):
    """Replaces `tp.project_obs` for arms that own their input path: the model
    receives the normalised RAW 348-d observation in rollout, update and
    deterministic evaluation alike (all three call sites route through
    `project_obs`, which is the whole reason this injection point is safe)."""
    return obs


def _mlp(sizes, act, small_last=False):
    _, torch, nn = _lazy_torch()
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        if small_last and i == len(sizes) - 2:
            # Last policy layer init 100x small (2006.05990): the policy starts
            # near zero-mean so early exploration is log_std's job, not the
            # initialisation lottery's.
            with torch.no_grad():
                lin.weight.mul_(0.01)
                lin.bias.mul_(0.0)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _build_arm_modules(name, action_dim=17, mujoco_dim=348, d_model=512):
    """The per-arm nn.Module that becomes `tp.model`. Returns (module, aux)
    where aux is the A' auxiliary-objective closure or None.

    Every module's forward takes the normalised raw observation (B, 348) and
    returns the dict contract `collect_rollout_vec`/`rl_update`/`policy_mean`
    consume: 'actions' (B, 1, action_dim) and 'value' (B, 1).
    """
    np, torch, nn = _lazy_torch()
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    def make_trunk():
        return UnifiedBrain(UnifiedBrainConfig(
            d_model=d_model, n_layers=8, obs_dim=256, action_dim=action_dim,
            enable_world_model=False, llm_enabled=False,
            vision_enabled=False, audio_enabled=False))

    def make_stem():
        # Mirrors TrainingPipeline.obs_proj (348 -> 256) so B and A' feed the
        # trunk exactly what arm C's trunk receives — arms differ ONLY in
        # control-path placement, never in the trunk's input contract.
        return nn.Sequential(
            nn.Linear(mujoco_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256))

    if name == "d_mlp":
        class SeparateActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.pol = _mlp([mujoco_dim, 128, 128, action_dim], nn.Tanh,
                                small_last=True)
                self.val = _mlp([mujoco_dim, 256, 256, 1], nn.Tanh)

            def forward(self, x):
                return {"actions": self.pol(x).unsqueeze(1),
                        "value": self.val(x)}
        return SeparateActorCritic(), None

    if name == "b_split":
        class SplitTrunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = make_stem()
                self.trunk = make_trunk()
                self.critic = _mlp([mujoco_dim, 256, 256, 1], nn.Tanh)

            def forward(self, x):
                out = self.trunk(self.stem(x))
                # Policy path: stem+trunk+action head. Value path: the MLP
                # critic on the raw observation. ZERO shared parameters, so
                # the vf gradient structurally cannot reach the policy trunk.
                return {"actions": out["actions"], "value": self.critic(x)}
        return SplitTrunk(), None

    if name == "aprime":
        class PlasticTrunkHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = make_stem()
                self.trunk = make_trunk()
                self.pred = nn.Linear(d_model, mujoco_dim)   # o_{t+1} head
                self.pol = _mlp([d_model, 128, 128, action_dim], nn.Tanh,
                                small_last=True)
                self.val = _mlp([d_model, 256, 256, 1], nn.Tanh)

            def features(self, x):
                return self.trunk(self.stem(x))["cls_features"]

            def forward(self, x):
                z = self.features(x).detach()   # STOP-GRADIENT: PPO ends here
                return {"actions": self.pol(z).unsqueeze(1),
                        "value": self.val(z)}

            def control_params(self):
                return list(self.pol.parameters()) + list(self.val.parameters())

            def trunk_params(self):
                return (list(self.stem.parameters())
                        + list(self.trunk.parameters())
                        + list(self.pred.parameters()))
        return PlasticTrunkHead(), "aux"

    raise ValueError(name)


def _make_arm(name, config_cls=None):
    """A TrainingPipeline configured as one arm. Returns (tp, aux_step, meta).

    aux_step(rollout) is a no-op for C/B/D; for A' it runs one epoch of the
    trunk's next-observation objective on the rollout just consumed by PPO.
    """
    np, torch, nn = _lazy_torch()
    from TrainingPipeline import TrainingPipeline, PipelineConfig, EWC

    tp = TrainingPipeline(PipelineConfig())
    lr = tp.config.learning_rate
    aux_opt = None
    aux_steps = [0]

    if name == "c_e2e":
        tp.make_optimizer(phase=3)
    else:
        module, aux = _build_arm_modules(name)
        del tp.model                     # the stock 57M trunk is not this arm
        tp.model = module.to(tp.device)
        tp.ewc = EWC(tp.model, tp.config.ewc_lambda)   # penalty stays 0; the
        tp.project_obs = _identity_project             # ref must not dangle
        if name == "aprime":
            # PPO optimiser covers ONLY the control head + log_std. The trunk
            # is excluded on purpose — detach already stops the gradient, and
            # excluding it makes the pilot's wiring assertion mean something.
            tp.optimizer = torch.optim.AdamW(
                [{"params": tp.model.control_params(), "lr": lr},
                 {"params": [tp.log_std], "lr": lr}],
                weight_decay=1e-4, eps=1e-5)
            aux_opt = torch.optim.AdamW(tp.model.trunk_params(), lr=lr,
                                        weight_decay=1e-4, eps=1e-5)
        else:
            tp.make_optimizer(phase=3)

    def aux_step(rollout):
        if aux_opt is None:
            return 0.0
        # One epoch of next-obs prediction over the rollout: (T-1, N) pairs,
        # done-transitions masked (predicting across a reset would train the
        # trunk on a discontinuity the world never produced).
        tp.model.train()
        states = rollout["states"]            # (T, N, 348) raw normalised
        dones = rollout["dones"]
        s_t = states[:-1].reshape(-1, states.shape[-1])
        s_t1 = states[1:].reshape(-1, states.shape[-1])
        keep = (dones[:-1].reshape(-1) < 0.5)
        s_t, s_t1 = s_t[keep], s_t1[keep]
        mb = 512
        total, nb = 0.0, 0
        perm = torch.randperm(s_t.shape[0], device=s_t.device)
        for start in range(0, s_t.shape[0], mb):
            idx = perm[start:start + mb]
            if len(idx) < 4:
                continue
            pred = tp.model.pred(tp.model.features(s_t[idx]))
            loss = torch.nn.functional.mse_loss(pred, s_t1[idx])
            aux_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tp.model.trunk_params(),
                                     tp.config.max_grad_norm)
            aux_opt.step()
            total += float(loss.item())
            nb += 1
            aux_steps[0] += 1
        return total / max(nb, 1)

    ppo_trainable = sum(p.numel() for g in tp.optimizer.param_groups
                        for p in g["params"])
    aux_trainable = (sum(p.numel() for g in aux_opt.param_groups
                         for p in g["params"]) if aux_opt else 0)
    meta = {"ppo_trainable_params": int(ppo_trainable),
            "aux_trainable_params": int(aux_trainable),
            "aux_steps": aux_steps}
    return tp, aux_step, meta


def _eval_policy(tp, episodes, random_actions=False):
    """Deterministic paired evaluation (repair R6): every arm, every twin and
    the random null face the SAME eval env seeds, so seed-to-seed initial-state
    luck cancels out of every pairwise comparison."""
    np, _, _ = _lazy_torch()
    env = tp.make_env()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        done, total = False, 0.0
        while not done:
            if random_actions:
                act = env.action_space.sample()
            else:
                act = tp.act_deterministic(obs)   # THE one eval path (T0.16)
            act = np.clip(act, env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(act)
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return returns


def _param_sig(params):
    """Cheap change-detector for the wiring assertions: sum of |p| per tensor.
    A single optimiser step moves every updated tensor's signature; an
    untouched tensor's signature is bit-identical."""
    return [float(p.detach().abs().sum().item()) for p in params]


def _train_arm(name, seed, step_target, minutes_cap, curve_max=200):
    np, torch, nn = _lazy_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tp, aux_step, meta = _make_arm(name)

    untrained = _eval_policy(tp, EVAL_EPISODES)

    envs = tp.make_vec_envs(N_ENVS)
    deadline = time.time() + minutes_cap * 60
    t_start = time.time()
    iters = steps = opt_steps = 0
    mb = min(tp.config.ppo_minibatch, ROLLOUT_STEPS * N_ENVS)
    opt_per_iter = tp.config.n_epochs_ppo * (
        (ROLLOUT_STEPS * N_ENVS + mb - 1) // mb)
    curve = []
    while steps < step_target and time.time() < deadline:
        buf = tp.collect_rollout_vec(envs, n_steps=ROLLOUT_STEPS)
        tp.rl_update(buf)
        aux_loss = aux_step(buf)
        iters += 1
        steps += ROLLOUT_STEPS * N_ENVS
        opt_steps += opt_per_iter
        curve.append({"iter": iters, "steps": steps,
                      "mean_reward": float(buf["rewards"].mean()),
                      "action_std": float(tp.log_std.exp().mean()),
                      "value_mean": float(buf["values"].mean()),
                      "aux_loss": round(float(aux_loss), 6)})
    envs.close()

    trained = _eval_policy(tp, EVAL_EPISODES)
    # Decimate the FULL curve to <= curve_max points spanning all iterations
    # (repair R5 — the [:8] truncation hid T2.01 v4's plateau from its readers).
    if len(curve) > curve_max:
        idx = [round(i * (len(curve) - 1) / (curve_max - 1))
               for i in range(curve_max)]
        curve = [curve[i] for i in sorted(set(idx))]
    wall_s = time.time() - t_start
    return {
        "arm": name, "seed": seed,
        "env_steps": steps, "optimiser_steps": opt_steps,
        "aux_optimiser_steps": int(meta["aux_steps"][0]),
        "updates_per_env_step": round(opt_steps / max(steps, 1), 6),
        "steps_per_s": round(steps / max(wall_s, 1e-9), 2),
        "ppo_trainable_params": meta["ppo_trainable_params"],
        "aux_trainable_params": meta["aux_trainable_params"],
        "wall_minutes": round(wall_s / 60, 1),
        "curve": curve,
        "untrained_returns": untrained,
        "untrained_mean": float(np.mean(untrained)),
        "trained_returns": trained,
        "trained_mean": float(np.mean(trained)),
    }


def _pilot_one(name):
    """Wiring assertions + a timed throughput window for ONE arm (the
    per-condition rule: no arm's envelope is certified by a sibling's)."""
    np, torch, nn = _lazy_torch()
    torch.manual_seed(0)
    np.random.seed(0)
    rec = {"arm": name}
    tp, aux_step, meta = _make_arm(name)
    rec.update({k: meta[k] for k in
                ("ppo_trainable_params", "aux_trainable_params")})

    # Snapshot the groups whose (non-)movement the arm's definition asserts.
    if name == "aprime":
        groups = {"trunk": tp.model.trunk_params(),
                  "head": tp.model.control_params()}
    elif name == "b_split":
        groups = {"trunk": (list(tp.model.stem.parameters())
                            + list(tp.model.trunk.parameters())),
                  "critic": list(tp.model.critic.parameters())}
    elif name == "d_mlp":
        groups = {"pol": list(tp.model.pol.parameters()),
                  "val": list(tp.model.val.parameters())}
    else:
        groups = {"trunk": list(tp.model.parameters())}
    before = {g: _param_sig(ps) for g, ps in groups.items()}

    envs = tp.make_vec_envs(N_ENVS)
    buf = tp.collect_rollout_vec(envs, n_steps=ROLLOUT_STEPS)
    stats = tp.rl_update(buf)
    rec["losses_finite"] = all(
        np.isfinite(v) for v in stats.values() if isinstance(v, float))
    after_ppo = {g: _param_sig(ps) for g, ps in groups.items()}
    changed_ppo = {g: any(a != b for a, b in zip(after_ppo[g], before[g]))
                   for g in groups}

    wiring_ok = True
    if name == "aprime":
        # THE arm-defining invariant: PPO must not move the trunk; the aux
        # objective must. Asserted live, not assumed from the detach call.
        wiring_ok &= (not changed_ppo["trunk"]) and changed_ppo["head"]
        aux_step(buf)
        after_aux = _param_sig(groups["trunk"])
        changed_aux = any(a != b for a, b in zip(after_aux, after_ppo["trunk"]))
        rec["trunk_changed_by_aux"] = bool(changed_aux)
        wiring_ok &= changed_aux
    elif name == "b_split":
        wiring_ok &= changed_ppo["trunk"] and changed_ppo["critic"]
    elif name == "d_mlp":
        wiring_ok &= changed_ppo["pol"] and changed_ppo["val"]
    else:
        wiring_ok &= changed_ppo["trunk"]
    rec["ppo_changed"] = {g: bool(v) for g, v in changed_ppo.items()}
    rec["wiring_ok"] = bool(wiring_ok)

    # Timed throughput window, measured AFTER the first update so one-time
    # costs (cudnn autotune, allocator warmup) do not flatter the estimate.
    t0 = time.time()
    steps = 0
    while time.time() - t0 < PILOT_MINUTES_PER_ARM * 60:
        buf = tp.collect_rollout_vec(envs, n_steps=ROLLOUT_STEPS)
        tp.rl_update(buf)
        aux_step(buf)
        steps += ROLLOUT_STEPS * N_ENVS
    elapsed = time.time() - t0
    envs.close()
    rec["timed_env_steps"] = steps
    rec["timed_seconds"] = round(elapsed, 1)
    rec["steps_per_s"] = round(steps / max(elapsed, 1e-9), 2)
    rec["projected_hours_per_seed_at_750k"] = round(
        750_000 / max(rec["steps_per_s"], 1e-9) / 3600, 2)

    del tp
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def remote_run(mode, arms=None, seeds=None):
    """Kernel entry point. mode='pilot' (all four arms, seed 0, throughput +
    wiring) or mode='full' (train the given arms x seeds to STEP_TARGET)."""
    np, torch, _ = _lazy_torch()
    from TrainingPipeline import TrainingPipeline, PipelineConfig
    arms = arms or ARMS
    seeds = seeds or SEEDS
    t0 = time.time()
    out = {"mode": mode,
           "gpu": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
           "config": {"n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS,
                      "step_target": STEP_TARGET,
                      "eval_seed_base": EVAL_SEED_BASE}}

    tp0 = TrainingPipeline(PipelineConfig())
    out["random_returns"] = _eval_policy(tp0, RANDOM_EPISODES,
                                         random_actions=True)
    del tp0

    def _dump():
        out["wall_minutes"] = round((time.time() - t0) / 60, 1)
        p = os.environ.get("JACK_OUT")
        if p:
            fn = "d10_pilot.json" if mode == "pilot" else "d10_full.json"
            json.dump(out, open(os.path.join(p, fn), "w"), indent=1)

    if mode == "pilot":
        out["arms"] = []
        for a in arms:
            rec = _pilot_one(a)
            out["arms"].append(rec)
            _dump()   # partial dump per arm: a timeout costs the last arm only
            print("PILOT", a, rec["steps_per_s"], "steps/s wiring_ok",
                  rec["wiring_ok"], flush=True)
    else:
        out["runs"] = []
        for a in arms:
            for s in seeds:
                r = _train_arm(a, s, STEP_TARGET, MINUTES_CAP[a])
                out["runs"].append(r)
                _dump()
                print("RUN", a, "seed", s, round(r["trained_mean"], 1), "@",
                      r["env_steps"], "steps", flush=True)
    _dump()
    return out


# =============================================================================
# HOST SIDE — submission, experiment, check.
# =============================================================================

_PILOT_JOB = r'''
import subprocess as _sp, sys as _sys, os as _os
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
from experiments.tests.d1_0_control_path_bakeoff import remote_run
remote_run("pilot")
'''

_FULL_JOB = r'''
import subprocess as _sp, sys as _sys, os as _os
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
from experiments.tests.d1_0_control_path_bakeoff import remote_run
remote_run("full", arms=__ARMS__)
'''


def pilot(out_path=_PILOT_ARTIFACT):
    """Dispatch the throughput/wiring pilot to Kaggle and write its artifact.
    ~40 min GPU: 4 arms x (construct + 1 update + 6 timed minutes)."""
    job = build_job(_PILOT_JOB)
    res = submit(job, prefer="kaggle", est_hours=1.2, timeout_s=7000,
                 fetch=["d10_pilot.json"])
    if not res.ok:
        raise RuntimeError(f"pilot failed on {res.backend}: {res.message}")
    path = res.artifacts.get("d10_pilot.json")
    if not path:
        raise RuntimeError(f"no pilot artifact. message={res.message!r} "
                           f"stdout_tail={res.stdout[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    Path(out_path).write_text(json.dumps(d, indent=1))
    print("PILOT ARTIFACT", out_path)
    for a in d.get("arms", []):
        print(f"  {a['arm']:8s} {a['steps_per_s']:8.1f} steps/s  "
              f"wiring_ok={a['wiring_ok']}  "
              f"~{a['projected_hours_per_seed_at_750k']}h/seed@750k")
    return d


_CACHE: dict = {}


def _submit_full() -> dict:
    if _KERNEL_SPLIT is None:
        raise RuntimeError(
            "D1.0 _KERNEL_SPLIT is not frozen — run the pilot, freeze the "
            "split from its measured steps/s, then submit. Submitting on an "
            "estimate is how a 9h kernel dies at hour 8 with two arms done.")
    merged = {"runs": [], "kernels": []}
    for arms in _KERNEL_SPLIT:
        body = _FULL_JOB.replace("__ARMS__", repr(list(arms)))
        job = build_job(body)
        est = sum(MINUTES_CAP[a] for a in arms) * len(SEEDS) / 60.0
        res = submit(job, prefer="kaggle", est_hours=min(est * 1.15, 8.8),
                     timeout_s=32000, fetch=["d10_full.json"])
        if not res.ok:
            raise RuntimeError(f"kernel {arms} failed on {res.backend}: "
                               f"{res.message}")
        path = res.artifacts.get("d10_full.json")
        if not path:
            raise RuntimeError(f"no artifact from kernel {arms}: "
                               f"{res.message!r}")
        d = json.loads(Path(path).read_text())
        merged["runs"].extend(d["runs"])
        merged["kernels"].append({"arms": list(arms), "gpu": d["gpu"],
                                  "wall_minutes": d["wall_minutes"],
                                  "backend": res.backend})
        if "random_returns" not in merged:
            merged["random_returns"] = d["random_returns"]
    return merged


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    return m, (sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5


def _slope(points):
    """Least-squares slope of mean_reward vs env-steps over the given curve
    points — the convergence check's instrument, pre-registered here."""
    if len(points) < 2:
        return 0.0
    xs = [p["steps"] for p in points]
    ys = [p["mean_reward"] for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(denom, 1e-9)


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit_full())
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    m = {"kernels": _CACHE["kernels"],
         "random_mean": round(rnd_mean, 1), "random_std": round(rnd_std, 2),
         "arms": {}}
    for a in ARMS:
        runs = [r for r in _CACHE["runs"] if r["arm"] == a]
        means = [r["trained_mean"] for r in runs]
        am, astd = _stats(means)
        m["arms"][a] = {
            "trained_means": [round(x, 1) for x in means],
            "mean": round(am, 1), "std": round(astd, 2),
            "sigma_vs_random": round((am - rnd_mean)
                                     / max(astd, rnd_std, 1e-6), 2),
            "env_steps": [r["env_steps"] for r in runs],
            "optimiser_steps": [r["optimiser_steps"] for r in runs],
            "aux_optimiser_steps": [r["aux_optimiser_steps"] for r in runs],
            "updates_per_env_step": runs[0]["updates_per_env_step"],
            "ppo_trainable_params": runs[0]["ppo_trainable_params"],
            "step_match": round(min(r["env_steps"] for r in runs)
                                / STEP_TARGET, 3),
            "final_third_slope": round(_slope(
                [p for r in runs for p in
                 r["curve"][2 * len(r["curve"]) // 3:]]), 9),
            "final_third_reward": round(sum(
                p["mean_reward"] for r in runs
                for p in r["curve"][2 * len(r["curve"]) // 3:]) / max(sum(
                    len(r["curve"]) - 2 * len(r["curve"]) // 3
                    for r in runs), 1), 3),
        }
    ranked = sorted(ARMS, key=lambda a: m["arms"][a]["mean"], reverse=True)
    w, r_up = ranked[0], ranked[1]
    pooled = max((m["arms"][w]["std"] ** 2 / 2
                  + m["arms"][r_up]["std"] ** 2 / 2) ** 0.5, 1e-6)
    m["winner"], m["runner_up"] = w, r_up
    m["margin_sigma"] = round(
        (m["arms"][w]["mean"] - m["arms"][r_up]["mean"]) / pooled, 2)
    return m


def _control(seed: int) -> dict:
    """Untrained twins of ALL FOUR arms must miss the learning gate."""
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    c = {}
    for a in ARMS:
        runs = [r for r in _CACHE["runs"] if r["arm"] == a]
        um, ustd = _stats([r["untrained_mean"] for r in runs])
        c[f"untrained_{a}_mean"] = round(um, 1)
        c[f"untrained_{a}_sigma"] = round(
            (um - rnd_mean) / max(ustd, rnd_std, 1e-6), 2)
    return c


def _check(m: dict, c: dict):
    # Every verdict names its branch WITH the comparison (the BA.03 lesson: a
    # one-bit verdict over a conjunction gets read as its most familiar
    # branch, and the readout must carry the comparison, not the operand).
    low = {a: m["arms"][a]["step_match"] for a in ARMS
           if m["arms"][a]["step_match"] < MIN_STEP_MATCH}
    if low:
        m["verdict"] = (
            "VOID — not at matched experience: "
            + ", ".join(f"{a} reached {v:.0%} of STEP_TARGET {STEP_TARGET}"
                        f" (floor {MIN_STEP_MATCH:.0%})"
                        for a, v in low.items())
            + ". Raise that arm's MINUTES_CAP; do not compare.")
        return Status.VOID
    missed = {a: m["arms"][a]["sigma_vs_random"] for a in ARMS
              if m["arms"][a]["sigma_vs_random"] < MIN_LEARN_SIGMA}
    if missed:
        learned = {a: m["arms"][a]["sigma_vs_random"] for a in ARMS
                   if a not in missed}
        m["verdict"] = (
            "VOID — learning gate: "
            + ", ".join(f"{a} at {v} sigma vs random (bar {MIN_LEARN_SIGMA})"
                        for a, v in missed.items())
            + f"; arms that DID learn: {learned or 'none'}. Two non-learners "
              "cannot arbitrate an architecture (T2.02's precedent). If d_mlp "
              "is among the missed while T2.02's SB3 MLP holds 530/7.11 sigma, "
              "this is the shared trunk-tuned recipe failing the MLP — a "
              "recipe question for the Review (UB.10's finding), not an "
              "architecture verdict and not a re-roll.")
        return Status.VOID
    hot_twins = {a: c[f"untrained_{a}_sigma"] for a in ARMS
                 if c[f"untrained_{a}_sigma"] >= MIN_LEARN_SIGMA}
    if hot_twins:
        m["verdict"] = (
            "VOID — untrained twin(s) cleared the learning gate: "
            + ", ".join(f"{a} at {v} sigma (bar {MIN_LEARN_SIGMA})"
                        for a, v in hot_twins.items())
            + ". The gate is measuring architectural bias, not learning.")
        return Status.VOID
    w, r_up = m["winner"], m["runner_up"]
    if m["margin_sigma"] < WIN_MARGIN_SIGMA:
        cheapest = min(ARMS, key=lambda a: m["arms"][a]["ppo_trainable_params"])
        m["verdict"] = (
            f"FAIL (TIE) — {w} leads {r_up} by {m['margin_sigma']} sigma, "
            f"under the {WIN_MARGIN_SIGMA}-sigma margin. Resolved to the "
            f"cheapest arm by PPO-trainable parameters: {cheapest} "
            f"({m['arms'][cheapest]['ppo_trainable_params']:,}). A real "
            "result: the control-path choice does not matter at this scale.")
        return False
    slope = m["arms"][r_up]["final_third_slope"]
    if slope > 0:
        gap = (m["arms"][w]["final_third_reward"]
               - m["arms"][r_up]["final_third_reward"])
        crossover = (STEP_TARGET + gap / slope if gap > 0 else STEP_TARGET)
        m["crossover_steps"] = round(crossover)
        if crossover <= CROSSOVER_MULT * STEP_TARGET:
            m["verdict"] = (
                f"VOID (SPLIT-PENDING) — {w} wins by {m['margin_sigma']} "
                f"sigma but runner-up {r_up} is still climbing (final-third "
                f"slope {slope:.2e}/step) with extrapolated crossover at "
                f"{m['crossover_steps']:,} steps, inside {CROSSOVER_MULT}x "
                f"STEP_TARGET ({int(CROSSOVER_MULT * STEP_TARGET):,}). The "
                "owner's convergence check: no winner while the runner-up is "
                "closing.")
            return Status.VOID
    m["verdict"] = (
        f"PASS — {w} beats {r_up} by {m['margin_sigma']} sigma "
        f"(bar {WIN_MARGIN_SIGMA}) at matched env-steps and optimiser-steps; "
        f"runner-up final-third slope {slope:.2e} "
        + ("<= 0 (converged)" if slope <= 0
           else f"with crossover beyond {CROSSOVER_MULT}x budget")
        + ". "
        + ("RECORDED COST (pre-registered): a d_mlp win forecloses DP.02 — "
           "private control representations, the two-brains-in-one-wrapper "
           "signature. Adoption routes through the Review, not this row."
           if w == "d_mlp" else
           f"{w} takes the Control-architecture seat's verdict subject to "
           "CHAMPIONS.md process."))
    return True


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "D1.0 gates are provisional — the pilot has not frozen "
            "STEP_TARGET / MINUTES_CAP / _KERNEL_SPLIT from measured per-arm "
            "throughput. Run the pilot (python -m experiments.tests."
            "d1_0_control_path_bakeoff pilot), freeze the envelope in this "
            "file with the pilot record, then run (SM.02's _GATES_FROZEN "
            "idiom).")
    return run_spec(BY_ID["D1.0"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    # `pilot`  — dispatch the GPU pilot from this box (pushes required first;
    #            gpu.assert_ref_is_current refuses an unpushed HEAD).
    # `smoke`  — local CPU sanity: tiny envelope, every arm constructs, takes
    #            one PPO step, and its wiring assertions hold. Minutes, not
    #            hours; run before ever paying GPU quota.
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "pilot":
        pilot()
    elif cmd == "smoke":
        # globals(), NOT a re-import: under `python -m` this module is
        # `__main__`, and importing it by name would build a SECOND module
        # object whose constants remote_run never reads.
        g = globals()
        g["N_ENVS"] = 2
        g["ROLLOUT_STEPS"] = 16
        g["PILOT_MINUTES_PER_ARM"] = 0.0
        g["EVAL_EPISODES"] = 1
        g["RANDOM_EPISODES"] = 1
        out = remote_run("pilot")
        ok = all(a["wiring_ok"] and a["losses_finite"] for a in out["arms"])
        print(json.dumps({a["arm"]: {"wiring_ok": a["wiring_ok"],
                                     "losses_finite": a["losses_finite"],
                                     "ppo_changed": a["ppo_changed"]}
                          for a in out["arms"]}, indent=1))
        print("SMOKE", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    else:
        raise SystemExit(f"unknown command {cmd!r} (pilot|smoke)")
