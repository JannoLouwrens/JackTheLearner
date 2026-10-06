"""T4.05 — Full regression gate: Tier 0-3's proven capabilities, re-checked
ON the composed brain.

HYPOTHESIS (registry). Every passing Tier 0-3 test still passes after
composition. Falsified by: any regression. Metric: regressions. Budget
GPU_LONG, depends_on T4.04.

WHAT "EVERY TEST" CAN HONESTLY MEAN HERE, DECLARED LOUDLY RATHER THAN
SILENTLY NARROWED. A Tier 0-3 certificate names one of two kinds of subject:
(1) a BRAIN CAPABILITY (the shipped UnifiedBrain/TrainingPipeline paths that
composition trains), or (2) a RIG, WORLD or INSTRUMENT (PG.*'s playground,
T0.*'s harness, SM/TA/W.*'s world laws). Training a brain cannot regress
kind (2) — no parameter of theirs is in a training event's causal reach, and
their re-sweep duty already has an owner: `python -m experiments.run --gate`,
whose hygiene is itself certified by T0.30. Re-implementing that sweep inside
a spec would be a second ladder-sweeping checker, which the 2026-09-17 freeze
(clause 2: no new audit organ or checker) forbids — and it would re-run specs
whose rows only the runner may write (T0.08). So THIS spec is the kind-(1)
gate: the Tier 0-3 claims whose subject is the brain composition touches,
each mapped BY SPEC ID onto a battery item, evaluated with the IDENTICAL
instrument and IDENTICAL pre-registered bar BEFORE and AFTER composition.
A regression = an item that read True on the pre-composition brain and False
on the composed one. The registry's "any regression" is the claim: the count
must be ZERO on every seed.

THE COMPOSITION EVENT is T4.04's, verbatim: phase 1 trains task A (T2.04's
behaviour cloning, same committed script law, seed 240814) on the shipped
action path; the composed brain is that brain after task B (T2.06's
grounding objective, shipped compute_language_grounding_loss, committed
synthetic family, seed 41004) for B_STEPS more. All task constants, data
laws and training loops are T4.04's own — the module-level pieces are
IMPORTED from it and the torch closures are reproduced verbatim, with the
coupling DECLARED: t4_04_task_interference.py is in IMPL_DEPS, so any drift
between the two rigs stales this certificate rather than hiding.

THE BATTERY — six items, each derived from a named Tier 0-3 spec's own
claim, each a boolean with its bar committed before any run:
  t101_overfit      (T1.01) a fresh Adam on ONE fixed batch (first OVERFIT_N
                    train rows) for OVERFIT_STEPS steps drives that batch's
                    MSE to <= OVERFIT_RATIO x its first-step value (or the
                    first-step value is already under OVERFIT_EPS). The
                    brain can still LEARN — plasticity, not memory.
  t103_grad_reach   (T1.03) the union gradient support of task A's loss and
                    task B's loss covers REF — the support measured on the
                    phase-1 brain. A tensor in REF with zero gradient after
                    composition went DEAD. (REF is measured, not assumed, so
                    this item is True pre-composition by construction and
                    binds post.)
  t104_weights_move (T1.04) one Adam step on task A's loss changes every
                    tensor that received nonzero gradient.
  t106_stable       (T1.06) task A's and task B's losses on fixed probe
                    batches are finite, and the deterministic policy output
                    on the probe batch is finite everywhere.
  t014_det_eval     (T0.14/T2.04's convention) the held-out eval is
                    bit-deterministic across two passes with the running-
                    normalisation state snapshot-restored between them.
  t204_beats_mean   (T2.04) held-out action MSE <= A_LEARN_RATIO x the
                    mean-action predictor's MSE. The capability itself —
                    NOT T4.04's 10% retention tolerance, which is T4.04's
                    claim and is not re-tested here; losing to the TRIVIAL
                    predictor is what "the Tier-2 test no longer passes"
                    means.
Battery discipline: the battery snapshots the brain on entry and restores
after every item that takes an optimizer step (t101, t104), so measuring
never mutates what is measured. All probe batches are FIXED slices, and the
battery forks and PINS the RNG for its whole body (the shipped brain has
live dropout in train mode, so train-mode forwards draw masks from the RNG;
union_support pins its own stream the same way for REF), so pre and post run
the bit-identical instrument and the ambient RNG stream is untouched.

THE PREMISE LANE (VOID, not FAIL). Items t101/t104/t106/t014/t204 must ALL
read True on the phase-1 brain on every seed: a capability that was not
there cannot regress, and a False pre-battery means the rig or a bar is
miscalibrated — an instrument statement, never evidence about composition.
(t103 is True pre by construction; it is excluded from the premise test.)

COMPOSITION-REALITY LANE (VOID). The composed brain's language-side
retrieval (T2.06's anchors-argmax eval) must beat 1/8 chance by >=
Z_B_LEARNED sigma on every seed. If B never trained, "nothing regressed" is
free and empty — the same vacuity T4.04's b_learned lane refuses.

THE CONTROL (MUST be caught; control-red is VOID, never FAIL). T4.04's
corrupted-supervision twin: from the same phase-1 snapshot, task A's own
inputs trained against SHUFFLED labels for the identical B_STEPS. That
certainly overwrites the A mapping, so the SAME battery must register >= 1
regression on it on every seed. A twin reading zero regressions means this
battery cannot see a lost capability that is certainly lost — a dead ruler
measures nothing.

STATISTIC_BOUND: regressions is bounded below at 0 and the claim sits AT
that bound — by the REGISTRY'S OWN TEXT ("Any regression" falsifies), not by
this implementation's choice. The unsaturated-null rule is satisfied the only
way a zero-defect claim can satisfy it: the null's distance from the bound is
not assumed but ENFORCED per run — the corrupted twin must measure >= 1
(control lane above), so a rig in which the statistic cannot leave its bound
reads VOID, never a clean 0. No envelope growth can flatter this gate in
either direction; the battery bars are constants committed here.

RIG -> VOID, not FAIL: dims contract (348/17, the T0.14 scar); every
training loss finite; every eval deterministic (the dropout scar).

WHAT A FAIL RETIRES (pre-registered): the claim that T4.04-style sequential
composition preserves Tier 0-3 capability on the one brain — i.e. T6.01 may
not inherit a composed brain whose battery regressed; the repair is a
composition recipe change routed through the Review, never a battery edit.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the 5.5-GPU-hour scar). Sizing borrows T4.04's
borrowed T2.04 probe rates one hop further (0.4225 s/step, 0.0157 s/eval-
row): per seed 1200 + 2x1200 train steps + 2x(60+1) battery steps ~ 3722
steps -> ~1600 s, + 3 evals x 1000 rows ~ 47 s each, + B-step overhead
unprobed -> est_hours 2.5, timeout_s 18000 (inside the runner's 21600 s
child cap). THE RATES ARE BORROWED TWICE REMOVED, NOT PROBED: this spec is
registered behind T4.04 (NOT_RUN, itself behind T2.01's FAIL) and cannot
dispatch today; before any real dispatch a probe at production config is
owed (16th audit B1), and the B-step op has never been timed.

Pilot modes (out-of-band, NO ledger write — "running a spec writes the
ledger"):
    python -m experiments.tests.t4_05_full_regression smoke
    python -m experiments.tests.t4_05_full_regression fixtures
`fixtures` replays every _check branch on synthetic dicts; `smoke` runs the
real code path at the production SHAPE (d512 brain) with tiny step counts on
CPU — it exercises plumbing (battery isolation, snapshot/restore, pre/post
instrument identity) and its gates are NOT the claim's (too few steps to
learn, so premise bools are reported, not asserted).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit
from .t4_04_task_interference import (
    ACT_DIM, OBS_DIM, A_BATCH, A_LR, A_STEPS, B_BATCH, B_LR, B_STEPS,
    A_LEARN_RATIO, CHANCE, Z_B_LEARNED, N_B_TEST_PER_CAT, N_B_TRAIN,
    N_TEST, N_TRAIN, _N_CATS, _binom_z, _collect, gen_action)

# The claim is about the shipped brain + shipped training paths, composed by
# T4.04's recipe — all three hash into impl_sha.
IMPL_DEPS = ["TrainingPipeline.py", "UnifiedBrain.py",
             "experiments/tests/t4_04_task_interference.py"]

SEEDS = [0, 1, 2]

# ── pre-registered battery constants. FINAL — committed before the first
# run. These are PREMISE bars (miscalibration -> VOID), except that the same
# bar read post-composition is what defines a regression. ────────────────
BATTERY = ("t101_overfit", "t103_grad_reach", "t104_weights_move",
           "t106_stable", "t014_det_eval", "t204_beats_mean")
PREMISE_ITEMS = ("t101_overfit", "t104_weights_move", "t106_stable",
                 "t014_det_eval", "t204_beats_mean")   # t103 pre-True by constr.
OVERFIT_N = 64          # rows in the single batch (fixed slice, no RNG)
OVERFIT_STEPS = 60      # fresh-Adam steps on that one batch
OVERFIT_RATIO = 0.5     # final <= 0.5 x first, or
OVERFIT_EPS = 1e-9      # ... first already at the floor


# ── everything below runs remotely (or locally for the smoke) ────────────
def remote_run(seeds: list, n_train: int = N_TRAIN, n_test: int = N_TEST,
               a_steps: int = A_STEPS, b_steps: int = B_STEPS,
               a_batch: int = A_BATCH, b_batch: int = B_BATCH,
               n_b_train: int = N_B_TRAIN,
               n_b_test_per_cat: int = N_B_TEST_PER_CAT,
               overfit_steps: int = OVERFIT_STEPS,
               double_pre: bool = False) -> dict:
    import gymnasium as gym
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig
    from UnifiedBrain import (SemanticActionAnchors,
                              compute_language_grounding_loss,
                              grounding_fallback_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(SemanticActionAnchors.ACTION_CATEGORIES.keys())
    syns = SemanticActionAnchors.ACTION_CATEGORIES

    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "seeds": []}
    env = gym.make("Humanoid-v5")
    obs_env = int(env.observation_space.shape[0])
    act_env = int(env.action_space.shape[0])

    # ── T4.04's loops, reproduced verbatim (coupling declared: IMPL_DEPS) ─
    def train_a(tp, Xn, Y, steps, gen_seed):
        X = torch.tensor(Xn, dtype=torch.float32, device=tp.device)
        T = torch.tensor(Y, dtype=torch.float32, device=tp.device)
        params = list(tp.model.parameters()) + list(tp.obs_proj.parameters())
        opt = torch.optim.Adam(params, lr=A_LR)
        g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
        tp.model.train()
        tp.obs_proj.train()
        last, fin = float("nan"), True
        for _ in range(steps):
            idx = torch.randint(0, len(X), (a_batch,), generator=g).to(tp.device)
            mean = tp.policy_mean(tp.model(tp.project_obs(X[idx])))
            loss = torch.nn.functional.mse_loss(mean, T[idx])
            if not bool(torch.isfinite(loss)):
                fin = False
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = float(loss.item())
        for p in params:
            p.grad = None
        return last, fin

    def train_b(tp, states, acts, phrases, steps, gen_seed):
        S = torch.tensor(states, dtype=torch.float32, device=tp.device)
        AT = torch.tensor(acts, dtype=torch.float32, device=tp.device)
        params = list(tp.model.parameters())
        opt = torch.optim.Adam(params, lr=B_LR)
        g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
        tp.model.train()
        first, last, fin = float("nan"), float("nan"), True
        for s in range(steps):
            idx = torch.randint(0, len(S), (b_batch,), generator=g)
            loss, _parts = compute_language_grounding_loss(
                tp.model, S[idx].to(tp.device), AT[idx].to(tp.device),
                [phrases[i] for i in idx.tolist()])
            if not bool(torch.isfinite(loss)):
                fin = False
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if s == 0:
                first = float(loss.item())
            last = float(loss.item())
        for p in params:
            p.grad = None
        return first, last, fin

    def eval_a(tp, Xte, Yte):
        snap = (tp.obs_mean.clone(), tp.obs_var.clone(), tp.obs_count)
        p1 = np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])
        tp.obs_mean, tp.obs_var, tp.obs_count = \
            snap[0].clone(), snap[1].clone(), snap[2]
        p2 = np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])
        tp.obs_mean, tp.obs_var, tp.obs_count = snap
        return float(((p1 - Yte) ** 2).mean()), bool(np.array_equal(p1, p2))

    def eval_b_lang(tp, phrases, cats):
        tp.model.eval()
        with torch.no_grad():
            toks = grounding_fallback_tokens(phrases).to(tp.device)
            lang = tp.model.language_encoder(toks)
            _sel, probs = tp.model.semantic_anchors(lang)
            pred = probs.argmax(-1).cpu().numpy()
        return float((pred == cats).mean())

    def snapshot(tp):
        return {
            "model": {k: v.detach().clone()
                      for k, v in tp.model.state_dict().items()},
            "proj": {k: v.detach().clone()
                     for k, v in tp.obs_proj.state_dict().items()},
            "norm": (tp.obs_mean.clone(), tp.obs_var.clone(), tp.obs_count),
        }

    def restore(tp, snap):
        tp.model.load_state_dict(snap["model"])
        tp.obs_proj.load_state_dict(snap["proj"])
        tp.obs_mean, tp.obs_var, tp.obs_count = \
            snap["norm"][0].clone(), snap["norm"][1].clone(), snap["norm"][2]

    def union_support(tp, Xn, Y, states, acts, phrases):
        """Tensor names receiving nonzero grad from task A's OR task B's
        loss, on fixed probe slices. One backward each, no step, grads
        cleared after (T4.04's grad_overlap, union instead of overlap)."""
        every = (list(tp.model.named_parameters())
                 + [("obs_proj." + k, v)
                    for k, v in tp.obs_proj.named_parameters()])

        def support(loss_fn):
            for _n, p in every:
                p.grad = None
            loss_fn().backward()
            s = {n for n, p in every
                 if p.grad is not None and float(p.grad.abs().sum()) > 0}
            for _n, p in every:
                p.grad = None
            return s

        X = torch.tensor(Xn[:a_batch], dtype=torch.float32, device=tp.device)
        T = torch.tensor(Y[:a_batch], dtype=torch.float32, device=tp.device)
        S = torch.tensor(states[:b_batch], dtype=torch.float32,
                         device=tp.device)
        AT = torch.tensor(acts[:b_batch], dtype=torch.float32,
                          device=tp.device)
        tp.model.train()
        tp.obs_proj.train()
        # The brain has live dropout in train mode; pin the mask stream so
        # support is a fixed instrument (same masks on every invocation)
        # and the ambient RNG stream is untouched.
        devs = [0] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(900_032)
            sa = support(lambda: torch.nn.functional.mse_loss(
                tp.policy_mean(tp.model(tp.project_obs(X))), T))
            sb = support(lambda: compute_language_grounding_loss(
                tp.model, S, AT, phrases[:b_batch])[0])
        return sa | sb

    def battery(tp, Xn, Y, Xte, Yte, states, acts, phrases, mse_mean,
                ref_support):
        """The six items. Snapshot on entry; every step-taking item is
        restored before the next, so measuring never mutates the brain.
        All probe batches are fixed slices, and the RNG is forked and
        PINNED for the battery's whole body (the brain has live dropout in
        train mode, so train-mode forwards draw masks from the RNG), making
        pre and post the bit-identical instrument while leaving the ambient
        RNG stream untouched."""
        base = snapshot(tp)
        devs = [0] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(900_031)
            return _battery_body(tp, Xn, Y, Xte, Yte, states, acts,
                                 phrases, mse_mean, ref_support, base)

    def _battery_body(tp, Xn, Y, Xte, Yte, states, acts, phrases, mse_mean,
                      ref_support, base):
        r = {}

        # t101_overfit — fresh Adam, ONE fixed batch
        Xb = torch.tensor(Xn[:OVERFIT_N], dtype=torch.float32,
                          device=tp.device)
        Tb = torch.tensor(Y[:OVERFIT_N], dtype=torch.float32,
                          device=tp.device)
        params = list(tp.model.parameters()) + list(tp.obs_proj.parameters())
        opt = torch.optim.Adam(params, lr=A_LR)
        tp.model.train()
        tp.obs_proj.train()
        first = last = float("nan")
        of_fin = True
        for s in range(overfit_steps):
            loss = torch.nn.functional.mse_loss(
                tp.policy_mean(tp.model(tp.project_obs(Xb))), Tb)
            if not bool(torch.isfinite(loss)):
                of_fin = False
                break
            if s == 0:
                first = float(loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = float(loss.item())
        for p in params:
            p.grad = None
        restore(tp, base)
        r["overfit_first"] = first
        r["overfit_last"] = last
        r["t101_overfit"] = bool(of_fin and (
            first <= OVERFIT_EPS or last <= OVERFIT_RATIO * first))

        # t103_grad_reach — union support covers REF
        sup = union_support(tp, Xn, Y, states, acts, phrases)
        r["support_n"] = len(sup)
        r["support_missing"] = sorted(ref_support - sup) if ref_support \
            else []
        r["t103_grad_reach"] = bool(not ref_support
                                    or ref_support.issubset(sup))

        # t104_weights_move — one Adam step changes every grad-carrying tensor
        every = (list(tp.model.named_parameters())
                 + [("obs_proj." + k, v)
                    for k, v in tp.obs_proj.named_parameters()])
        before = {n: p.detach().clone() for n, p in every}
        X = torch.tensor(Xn[:a_batch], dtype=torch.float32, device=tp.device)
        T = torch.tensor(Y[:a_batch], dtype=torch.float32, device=tp.device)
        params = [p for _n, p in every]
        opt = torch.optim.Adam(params, lr=A_LR)
        tp.model.train()
        tp.obs_proj.train()
        loss = torch.nn.functional.mse_loss(
            tp.policy_mean(tp.model(tp.project_obs(X))), T)
        opt.zero_grad()
        loss.backward()
        had_grad = {n for n, p in every
                    if p.grad is not None and float(p.grad.abs().sum()) > 0}
        opt.step()
        moved = {n for n, p in every
                 if not torch.equal(p.detach(), before[n])}
        for _n, p in every:
            p.grad = None
        restore(tp, base)
        r["grad_tensors"] = len(had_grad)
        r["t104_weights_move"] = bool(had_grad and had_grad.issubset(moved))

        # t106_stable — both losses + deterministic output finite
        with torch.no_grad():
            tp.model.eval()
            tp.obs_proj.eval()
            outp = tp.policy_mean(tp.model(tp.project_obs(X)))
            out_fin = bool(torch.isfinite(outp).all())
        tp.model.train()
        tp.obs_proj.train()
        la = torch.nn.functional.mse_loss(
            tp.policy_mean(tp.model(tp.project_obs(X))), T)
        S = torch.tensor(states[:b_batch], dtype=torch.float32,
                         device=tp.device)
        AT = torch.tensor(acts[:b_batch], dtype=torch.float32,
                          device=tp.device)
        lb, _parts = compute_language_grounding_loss(
            tp.model, S, AT, phrases[:b_batch])
        r["loss_a_probe"] = float(la.item())
        r["loss_b_probe"] = float(lb.item())
        for _n, p in every:
            p.grad = None
        r["t106_stable"] = bool(out_fin and torch.isfinite(la)
                                and torch.isfinite(lb))

        # t014_det_eval + t204_beats_mean — one held-out eval serves both
        mse, det = eval_a(tp, Xte, Yte)
        r["mse"] = mse
        r["t014_det_eval"] = bool(det)
        r["t204_beats_mean"] = bool(mse <= A_LEARN_RATIO * mse_mean)
        restore(tp, base)
        return r

    for seed in seeds:
        # A data (T2.04 domains), B data (T4.04's committed family)
        Xtr, Ytr = _collect(env, n_train, seed, domain=0)
        Xte, Yte = _collect(env, n_test, seed, domain=1)
        rng = np.random.RandomState(int(seed) * 100_003 % 2**32)
        cats_tr = rng.randint(0, _N_CATS, size=n_b_train)
        phr_tr = [syns[names[c]][rng.randint(len(syns[names[c]]))]
                  for c in cats_tr]
        acts_tr = np.stack([gen_action(int(c), rng) for c in cats_tr])
        states_tr = 0.1 * rng.randn(n_b_train, 256)
        n_b_test = n_b_test_per_cat * _N_CATS
        cats_te = np.repeat(np.arange(_N_CATS), n_b_test_per_cat)
        phr_te = [syns[names[c]][rng.randint(len(syns[names[c]]))]
                  for c in cats_te]

        # the brain: shipped pipeline + the anchors module the flag gates
        torch.manual_seed(int(seed))
        from UnifiedBrain import SemanticActionAnchors as _SAA
        tp = TrainingPipeline(PipelineConfig())
        tp.model.semantic_anchors = _SAA(
            tp.model.config.d_model, num_anchors=8).to(tp.device)
        dims_ok = (obs_env == tp.config.mujoco_obs_dim == OBS_DIM
                   and act_env == tp.config.action_dim == ACT_DIM)

        Xtr_norm = tp.normalize_obs(Xtr)
        mse_mean = float(((Ytr.mean(0) - Yte) ** 2).mean())

        # phase 1: task A — the pre-composition brain
        loss_a, fin_a = train_a(tp, Xtr_norm, Ytr, a_steps, seed * 7 + 1)

        # REF support, then the PRE battery on the phase-1 brain
        ref = union_support(tp, Xtr_norm, Ytr, states_tr, acts_tr, phr_tr)
        pre = battery(tp, Xtr_norm, Ytr, Xte, Yte, states_tr, acts_tr,
                      phr_tr, mse_mean, ref)
        battery_idempotent = True
        if double_pre:
            # isolation proof (smoke): a second identical battery on the
            # same brain must read bit-identically, or the battery mutates
            # what it measures
            pre2 = battery(tp, Xtr_norm, Ytr, Xte, Yte, states_tr, acts_tr,
                           phr_tr, mse_mean, ref)
            battery_idempotent = bool(
                pre2["mse"] == pre["mse"]
                and all(pre2[k] == pre[k] for k in BATTERY))
        snap = snapshot(tp)

        # SEQ: task B on the same brain — the COMPOSED brain
        lb_first, lb_last, fin_b = train_b(
            tp, states_tr, acts_tr, phr_tr, b_steps, seed * 7 + 2)
        acc_b_lang = eval_b_lang(tp, phr_te, cats_te)
        post = battery(tp, Xtr_norm, Ytr, Xte, Yte, states_tr, acts_tr,
                       phr_tr, mse_mean, ref)

        # CTRL: corrupted supervision from the same snapshot, matched steps
        restore(tp, snap)
        perm = np.random.RandomState(int(seed) + 41).permutation(len(Ytr))
        loss_c, fin_c = train_a(tp, Xtr_norm, Ytr[perm], b_steps, seed * 7 + 4)
        ctrl = battery(tp, Xtr_norm, Ytr, Xte, Yte, states_tr, acts_tr,
                       phr_tr, mse_mean, ref)

        reg_seq = [k for k in BATTERY if pre[k] and not post[k]]
        reg_ctrl = [k for k in BATTERY if pre[k] and not ctrl[k]]
        out["seeds"].append({
            "seed": int(seed), "dims_ok": bool(dims_ok),
            "battery_idempotent": bool(battery_idempotent),
            "premise_ok": bool(all(pre[k] for k in PREMISE_ITEMS)),
            "pre": {k: bool(pre[k]) for k in BATTERY},
            "post": {k: bool(post[k]) for k in BATTERY},
            "ctrl": {k: bool(ctrl[k]) for k in BATTERY},
            "regressions": len(reg_seq), "regressed_items": reg_seq,
            "regressions_ctrl": len(reg_ctrl), "ctrl_items": reg_ctrl,
            "mse_pre": round(pre["mse"], 6), "mse_post": round(post["mse"], 6),
            "mse_ctrl": round(ctrl["mse"], 6),
            "mse_mean": round(mse_mean, 6),
            "overfit_pre": [round(pre["overfit_first"], 6),
                            round(pre["overfit_last"], 6)],
            "overfit_post": [round(post["overfit_first"], 6),
                             round(post["overfit_last"], 6)],
            "support_ref_n": len(ref), "support_post_n": post["support_n"],
            "support_missing_post": post["support_missing"],
            "acc_b_lang": round(acc_b_lang, 4),
            "z_b_lang": round(_binom_z(acc_b_lang, n_b_test), 3),
            "n_b_test": int(n_b_test),
            "loss_b_first": round(lb_first, 6),
            "loss_b_final": round(lb_last, 6),
            "loss_a_final": round(loss_a, 6),
            "loss_ctrl_final": round(loss_c, 6),
            "det_ok": bool(pre["t014_det_eval"] and post["t014_det_eval"]
                           and ctrl["t014_det_eval"]),
            "steps": {"a_phase1": a_steps, "phase2_each_branch": b_steps,
                      "overfit": overfit_steps, "a_batch": a_batch,
                      "b_batch": b_batch},
            "finite": bool(np.isfinite(
                [pre["mse"], post["mse"], ctrl["mse"], loss_a, lb_last,
                 loss_c]).all() and fin_a and fin_b and fin_c),
        })
    env.close()
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _sp, sys as _sys, os as _o
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
import json
from experiments.tests.t4_05_full_regression import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t405.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Sizing: T4.04's borrowed T2.04 probe rates, one hop further (see
    # docstring — a probe at production config is OWED before any real
    # dispatch; this spec is behind T4.04 and cannot dispatch today).
    res = submit(job, prefer="kaggle", est_hours=2.5, timeout_s=18000,
                 fetch=["t405.json"])
    if not res.ok:
        raise RuntimeError(f"T4.05 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t405.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE.get("backend", "local"),
        "regressions": max(r["regressions"] for r in rows),
        "regressions_all": [r["regressions"] for r in rows],
        "regressed_items_all": [r["regressed_items"] for r in rows],
        "mse_pre_all": [r["mse_pre"] for r in rows],
        "mse_post_all": [r["mse_post"] for r in rows],
        "mse_mean_all": [r["mse_mean"] for r in rows],
        "overfit_pre_all": [r["overfit_pre"] for r in rows],
        "overfit_post_all": [r["overfit_post"] for r in rows],
        "support_ref_n_all": [r["support_ref_n"] for r in rows],
        "support_missing_post_all": [r["support_missing_post"]
                                     for r in rows],
        "acc_b_lang_all": [r["acc_b_lang"] for r in rows],
        "z_b_lang_all": [r["z_b_lang"] for r in rows],
        "loss_b_first_all": [r["loss_b_first"] for r in rows],
        "loss_b_final_all": [r["loss_b_final"] for r in rows],
        "steps": rows[0]["steps"],
        "claim_zero_regressions": all(r["regressions"] == 0 for r in rows),
        "premise_all": all(r["premise_ok"] for r in rows),
        "b_learned_all": all(r["z_b_lang"] >= Z_B_LEARNED for r in rows),
        "dims_ok_all": all(r["dims_ok"] for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "regressions_ctrl_all": [r["regressions_ctrl"] for r in rows],
        "ctrl_items_all": [r["ctrl_items"] for r in rows],
        "mse_ctrl_all": [r["mse_ctrl"] for r in rows],
        "control_caught_all": all(r["regressions_ctrl"] >= 1 for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig lanes first: an invalid run is VOID, not evidence.
    if not m["dims_ok_all"]:
        return Status.VOID          # config contract broken (T0.14 scar)
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval not deterministic (dropout scar)
    if not m["premise_all"]:
        return Status.VOID          # a capability was never there pre-
                                    # composition: nothing could regress,
                                    # and a bar is miscalibrated
    if not m["b_learned_all"]:
        return Status.VOID          # B never trained: composition never
                                    # happened, "nothing regressed" is empty
    if not c["control_caught_all"]:
        return Status.VOID          # planted damage invisible: dead ruler
    return bool(m["claim_zero_regressions"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T4.05"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


def _fixture_rows(regressions, regressions_ctrl, **over):
    m = {
        "dims_ok_all": True, "finite_all": True, "det_ok_all": True,
        "premise_all": True, "b_learned_all": True,
        "claim_zero_regressions": all(r == 0 for r in regressions),
    }
    m.update({k: v for k, v in over.items() if k in m})
    c = {"control_caught_all": all(r >= 1 for r in regressions_ctrl)}
    return m, c


def _fixtures():
    """Replay every _check branch on synthetic dicts. Exit non-zero on any
    branch answering wrongly."""
    cases = [
        # (name, m-overrides, regressions, regressions_ctrl, expected)
        ("pass", {}, [0, 0, 0], [2, 1, 3], True),
        ("fail_one_seed", {}, [0, 1, 0], [2, 1, 3], False),
        ("fail_all", {}, [3, 2, 4], [2, 1, 3], False),
        ("void_dims", {"dims_ok_all": False}, [0] * 3, [1] * 3, Status.VOID),
        ("void_nonfinite", {"finite_all": False}, [0] * 3, [1] * 3,
         Status.VOID),
        ("void_nondet", {"det_ok_all": False}, [0] * 3, [1] * 3, Status.VOID),
        ("void_premise", {"premise_all": False}, [0] * 3, [1] * 3,
         Status.VOID),
        ("void_b_unlearned", {"b_learned_all": False}, [0] * 3, [1] * 3,
         Status.VOID),
        ("void_dead_ruler", {}, [0] * 3, [0, 2, 2], Status.VOID),
        ("void_dead_ruler_all", {}, [0] * 3, [0, 0, 0], Status.VOID),
        # the dead ruler outranks the red claim: a battery that cannot see
        # planted damage voids the run even when the claim side also reads red
        ("void_beats_fail", {}, [1, 0, 0], [0, 0, 0], Status.VOID),
    ]
    bad = 0
    for name, over, rs, rc, want in cases:
        m, c = _fixture_rows(rs, rc, **over)
        got = _check(m, c)
        ok = got is want if isinstance(want, Status) else got == want
        print(f"  {'OK ' if ok else 'BAD'} {name}: _check -> {got}")
        bad += 0 if ok else 1
    print(f"fixtures: {len(cases) - bad}/{len(cases)} green")
    return bad


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fixtures":
        sys.exit(_fixtures())
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, tiny, CPU, out-of-band: the REAL code path at the production
        # SHAPE (default d512 brain). Step counts are far too small to learn;
        # the smoke asserts PLUMBING — dims, battery isolation (pre battery
        # twice -> identical bools and identical mse, proving the battery
        # restores what it measures), determinism, finiteness, all three
        # branches run. Premise bools are REPORTED, not asserted: at 25 A
        # steps they may read either way, and the claim's gates live in
        # _check, not here.
        out = remote_run([0], n_train=300, n_test=40, a_steps=25, b_steps=15,
                         a_batch=32, b_batch=16, n_b_train=200,
                         n_b_test_per_cat=5, overfit_steps=10,
                         double_pre=True)
        print(json.dumps(out, indent=1))
        r = out["seeds"][0]
        assert r["dims_ok"] and r["det_ok"] and r["finite"], r
        assert r["battery_idempotent"], "battery mutates what it measures"
        assert set(r["pre"]) == set(BATTERY), r
        assert r["regressions"] == len(r["regressed_items"]), r
        print("SMOKE OK — pre", r["pre"], "post", r["post"],
              "ctrl", r["ctrl"], "regressions", r["regressions"],
              "ctrl_regressions", r["regressions_ctrl"])
    else:
        run()
