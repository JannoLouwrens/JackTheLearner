"""T4.04 — Task interference: training task B does not degrade task A.

HYPOTHESIS (registry). Training task B does not degrade task A beyond a set
tolerance. Falsified by: A drops >10% while learning B. Null: A trained
alone. Metric: task_a_retention. Budget GPU_LONG, 3 seeds, depends_on T2.01.

WHAT THIS MEASURES, SAID PLAINLY. Tier 4 is COMPOSITION: Tiers 1-3 prove each
capability alone, and alone is not together. This is the first spec that
trains TWO different capabilities SEQUENTIALLY on ONE brain and asks whether
the second buys its skill out of the first's hide. GOAL.md's one-brain bet is
that what he hears can teach what he sees — the cheap failure mode of that
bet is that what he hears OVERWRITES what he can do. T5.03 (continual
learning, depends_on T4.04) owns the full forgetting curve with replay
baselines; this spec owns the one-step question that gates it: does the
shipped brain survive ONE task handoff inside tolerance?

THE TWO TASKS — both already proven individually, both shipped paths:
  TASK A (motor): T2.04's behaviour cloning. The shipped action path — obs →
      normalize_obs → obs_proj → UnifiedBrain trunk → policy_mean — trained
      by Adam/MSE to imitate T2.04's committed deterministic script law on
      Humanoid-v5 rollouts (the law's constants are re-derived here from the
      SAME literal seed 240814, so task A here IS task A there; T2.04 PASSed
      it 2026-08-07). Score: held-out action MSE through act_deterministic.
  TASK B (language): T2.06's grounding. The shipped objective
      compute_language_grounding_loss — flow matching through the trunk +
      action head, contrastive language↔action↔anchor alignment — on a
      committed synthetic (command, action) family over the shipped
      ACTION_CATEGORIES synonyms (T2.06 PASSed the mechanism 2026-08-16).

ONE CONSTRUCTION DISCLOSURE. TrainingPipeline builds its UnifiedBrain without
enable_semantic_anchors (the flag is not in PipelineConfig), so this rig
attaches the module post-construction with the IDENTICAL call the flag gates
(UnifiedBrain.py:3930: SemanticActionAnchors(d_model, num_anchors=8)), seeded
inside the same torch.manual_seed(seed) scope as the rest of the brain.
semantic_anchors has zero forward() reads (UnifiedBrainConfig's own comment)
— nothing about task A's path changes by its presence; only
compute_language_grounding_loss reads it.

THE ARMS — all three phase-2 branches start from the SAME phase-1 brain
(state snapshot/restore), all three get the SAME number of optimizer steps
(B_STEPS) and a fresh Adam, so "B degraded A" cannot hide inside "more
training happened" or "a different brain happened":
  phase 1 (shared)   train A for A_STEPS; measure mse_A0.
  seq                train B for B_STEPS (the shipped grounding loss, model
                     params incl. anchors); re-measure A -> mse_seq.
  alone (THE NULL)   train A for B_STEPS more (fresh draws from the same
                     committed law); re-measure A -> mse_alone. This is the
                     registry's "A trained alone" read at matched TOTAL
                     optimizer steps — the D1 lesson (match optimizer steps,
                     report both) applied to a retention claim.
  ctrl (MUST FAIL)   train A's own inputs against SHUFFLED labels for
                     B_STEPS; re-measure A -> mse_ctrl. Corrupted supervision
                     on A's own input distribution actively overwrites the
                     A mapping; if THIS twin reads as retained, the retention
                     instrument cannot see degradation that is certainly
                     there, and the run is VOID (the at-chance-control
                     lesson, LESSONS.md 2026-08-21).

THE STATISTIC, pre-registered. drop = (mse - mse_alone) / mse_alone, i.e.
task A's held-out error rise relative to the matched-steps null, per seed.
  CLAIM     drop_seq <= DROP_MAX on EVERY seed (DROP_MAX = 0.10 — the
            registry's own ">10%", committed before any run).
  CONTROL   drop_ctrl > DROP_MAX on EVERY seed (the planted degradation must
            be SEEN; a miss is VOID, not PASS).
  task_a_retention (the registry metric) = per-seed mse_alone / mse_seq,
            reported per seed; headline = min over seeds. retention 0.9091
            corresponds exactly to drop 0.10.
STATISTIC_BOUND: none — drop is unbounded above and below; neither the null
(drop identically 0 by construction) nor the claim sits at a ceiling, so the
unsaturated-null rule is satisfied by construction and growth of the
envelope is not a tempting repair here.

WHY "no interference" CANNOT PASS VACUOUSLY — the three learning gates, all
VOID lanes (an arm that learned nothing can neither interfere nor retain):
  A-learned   mse_A0 <= A_LEARN_RATIO * mse_mean (mean-action predictor) on
              every seed. A brain that never learned A has nothing to lose.
  B-learned   task B's language-side retrieval (shipped anchors argmax, the
              T2.06 eval) must beat 1/8 chance by >= 3 sigma binomial on
              every seed — the LC/T2.02 learning-gate convention. A B that
              never trained makes "B did not degrade A" free and empty.
              loss_b_first/final recorded per seed (the live-twin
              observable, 24th audit B3).
  SHARED SUBSTRATE, measured not assumed: one backward pass of each task's
              loss on the phase-1 brain (no optimizer step), then the
              intersection of parameter tensors receiving nonzero gradient
              from BOTH. If the intersection is EMPTY the two tasks occupy
              disjoint parameters, interference is impossible BY
              CONSTRUCTION, and a PASS would be decorative (T0.13's
              disease; GOAL.md: "an architecture can drift into two private
              towers while every capability number keeps improving") ->
              VOID. shared/a_only/b_only tensor counts recorded per seed.

RIG -> VOID, not FAIL (an invalid run is not evidence): dims contract
(348/17, T0.14 scar); every loss finite; every A eval bit-deterministic
across two passes with the running-normalisation state snapshot-restored
between them (the dropout scar, T2.04's exact assertion).

MATCHED-BUDGET DISCLOSURES. (1) Each phase-2 branch gets a FRESH Adam:
optimizer state does not cross the phase boundary in ANY branch, so the
reset is matched, not a confound. (2) Step counts are matched (B_STEPS per
branch); per-step batch follows each task's own proven recipe (A: 256,
T2.04; B: 64, T2.06), so SAMPLE counts differ and both are recorded
(steps are the optimizer-step match the D1 lesson names). (3) Task B's
states are the committed 0.1*randn model-space states of T2.06's rig —
task B never touches obs_proj, which is part of why the overlap gate is
measured rather than assumed.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the 5.5-GPU-hour scar). Cost projection borrows
T2.04's measured P100 probe rates (same config, same train-step op for the
A phases: 0.4225 s/step, 0.0157 s/eval-row) — per seed 1200 + 3*1200 train
steps -> ~14400 steps total ~6100 s, + 24000 eval rows ~380 s, + B-step
overhead unprobed. est_hours 2.5, timeout_s 18000 (inside the runner's
21600 s child cap). THE RATES ARE BORROWED, NOT PROBED: this spec is
registered behind T2.01's FAIL and cannot dispatch today; before any real
dispatch a probe at production config is owed (16th audit B1 — a borrowed
rate is not a measured rate, and the B-step op has never been timed).

Pilot modes (out-of-band, NO ledger write — "running a spec writes the
ledger"):
    python -m experiments.tests.t4_04_task_interference smoke
    python -m experiments.tests.t4_04_task_interference fixtures
`fixtures` replays every _check branch on synthetic dicts; `smoke` runs the
real code path at the production SHAPE (d512 brain) with tiny step counts on
CPU — it exercises plumbing and measures the shared-substrate overlap for
real; its gates are NOT the claim's (too few steps to learn).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped brain + shipped training paths.
IMPL_DEPS = ["TrainingPipeline.py", "UnifiedBrain.py"]

SEEDS = [0, 1, 2]

# ── task A (T2.04's task, same literal constants) ────────────────────────
OBS_DIM, ACT_DIM = 348, 17
ACTION_LIMIT = 0.4
EP_CAP = 100
EXPLORE_STD = 0.1
N_TRAIN, N_TEST = 3000, 1000

A_STEPS = 1200                   # phase 1 (T2.04's BC_STEPS)
B_STEPS = 1200                   # phase 2, EVERY branch — the matched budget
A_BATCH = 256                    # T2.04's recipe
B_BATCH = 64                     # T2.06's recipe
A_LR = 3e-4
B_LR = 3e-4

# ── pre-registered gates. FINAL — committed before the first run. ────────
DROP_MAX = 0.10                  # the registry's own ">10%"
A_LEARN_RATIO = 0.8              # mse_A0 must sit >=20% under the mean anchor
Z_B_LEARNED = 3.0                # B's language side vs chance, per seed
CHANCE = 1.0 / 8.0

# The script law: T2.04's committed constants, re-derived from the SAME seed.
_S = np.random.RandomState(240814)
KP = _S.uniform(1.0, 3.0, size=ACT_DIM)
KD = _S.uniform(0.1, 0.5, size=ACT_DIM)
M = (_S.randn(ACT_DIM, ACT_DIM) / np.sqrt(ACT_DIM))


def script_action(obs: np.ndarray) -> np.ndarray:
    jang = obs[5:22]
    jvel = obs[28:45]
    u = KP * (0.0 - jang) - KD * jvel + 0.5 * np.tanh(M @ jang)
    return (ACTION_LIMIT * np.tanh(u)).astype(np.float64)


def _collect(env, n: int, seed: int, domain: int) -> tuple:
    """T2.04's collection: scripted rollouts + exploration noise, disjoint
    derived-seed domains (0 train, 1 test), all mod 2**32 (T2.03 scar)."""
    X = np.empty((n, OBS_DIM), dtype=np.float64)
    Y = np.empty((n, ACT_DIM), dtype=np.float64)
    i, ep = 0, 0
    while i < n:
        ep_seed = int((seed * 1_000_003 + domain * 500_009 + ep * 9_176) % 2**32)
        ep += 1
        obs, _ = env.reset(seed=ep_seed)
        rng = np.random.RandomState((ep_seed + 1) % 2**32)
        for _t in range(EP_CAP):
            label = script_action(obs)
            X[i], Y[i] = obs, label
            i += 1
            act = np.clip(label + EXPLORE_STD * rng.randn(ACT_DIM),
                          -ACTION_LIMIT, ACTION_LIMIT)
            obs, _r, term, trunc, _info = env.step(act)
            if i >= n or term or trunc:
                break
    return X, Y


# ── task B data: committed synthetic family over the SHIPPED synonyms ────
CHUNK = 16                       # SemanticActionAnchors.action_encoder's 17*16
N_B_TRAIN = 2000
N_B_TEST_PER_CAT = 25
_N_CATS = 8

_SB = np.random.RandomState(41004)
_MASKS = (_SB.rand(_N_CATS, ACT_DIM) < 0.45).astype(np.float64)
for _k in range(_N_CATS):
    if _MASKS[_k].sum() < 3:
        _MASKS[_k, _SB.choice(ACT_DIM, 3, replace=False)] = 1.0
_PATTERNS = _SB.randn(_N_CATS, ACT_DIM) / math.sqrt(ACT_DIM)
_JOINT_PHASE = _SB.uniform(0, 2 * np.pi, size=(_N_CATS, ACT_DIM))
_FREQS = np.array([1.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5])
_T = np.arange(CHUNK, dtype=np.float64)


def gen_action(cat: int, rng: np.random.RandomState) -> np.ndarray:
    """(CHUNK, ACT_DIM) sequence per category — T2.06's envelope shapes under
    THIS spec's own committed constants (seed 41004, drawn once above)."""
    amp = rng.uniform(0.5, 1.0)
    phase = rng.uniform(0, 2 * np.pi)
    m, w = _MASKS[cat], _PATTERNS[cat]
    base = 0.3 * amp * m * w
    if cat in (0, 1, 7):
        t = 2 * np.pi * _FREQS[cat] * _T / CHUNK
        seq = amp * m * w * np.sin(t[:, None] + phase + _JOINT_PHASE[cat])
    elif cat == 2:
        c = rng.uniform(4, 12)
        seq = amp * m * w * np.exp(-0.5 * ((_T[:, None] - c) / 2.0) ** 2)
    elif cat == 3:
        seq = np.zeros((CHUNK, ACT_DIM))
    elif cat in (4, 5):
        sign = 1.0 if cat == 4 else -1.0
        seq = sign * amp * m * np.abs(w) * (_T[:, None] / CHUNK)
    else:
        seq = -amp * m * np.abs(w) * (_T[:, None] >= CHUNK // 2)
    return (base + seq + 0.05 * rng.randn(CHUNK, ACT_DIM)).astype(np.float64)


def _binom_z(acc: float, n: int, p0: float = CHANCE) -> float:
    return (acc - p0) / math.sqrt(p0 * (1 - p0) / n)


# ── everything below runs remotely (or locally for the smoke) ────────────
def remote_run(seeds: list, n_train: int = N_TRAIN, n_test: int = N_TEST,
               a_steps: int = A_STEPS, b_steps: int = B_STEPS,
               a_batch: int = A_BATCH, b_batch: int = B_BATCH,
               n_b_train: int = N_B_TRAIN,
               n_b_test_per_cat: int = N_B_TEST_PER_CAT) -> dict:
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

    def train_a(tp, Xn, Y, steps, gen_seed):
        """T2.04's loop: Adam/MSE on the shipped forward."""
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
        """T2.06's loop: the shipped grounding objective on model params
        (anchors included; obs_proj is not in task B's graph)."""
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
        """(mse, det_ok): T2.04's double-pass bit-identity assertion with the
        running-normalisation state snapshot-restored between passes."""
        snap = (tp.obs_mean.clone(), tp.obs_var.clone(), tp.obs_count)
        p1 = np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])
        tp.obs_mean, tp.obs_var, tp.obs_count = \
            snap[0].clone(), snap[1].clone(), snap[2]
        p2 = np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])
        tp.obs_mean, tp.obs_var, tp.obs_count = snap
        return float(((p1 - Yte) ** 2).mean()), bool(np.array_equal(p1, p2))

    def eval_b_lang(tp, phrases, cats):
        """Language-side retrieval acc — the T2.06 anchors-argmax eval."""
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

    def grad_overlap(tp, Xn, Y, states, acts, phrases):
        """Tensor-level overlap of the two losses' gradient support. One
        backward each, no optimizer step, grads cleared after."""
        import torch as _t
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

        X = _t.tensor(Xn[:a_batch], dtype=_t.float32, device=tp.device)
        T = _t.tensor(Y[:a_batch], dtype=_t.float32, device=tp.device)
        S = _t.tensor(states[:b_batch], dtype=_t.float32, device=tp.device)
        AT = _t.tensor(acts[:b_batch], dtype=_t.float32, device=tp.device)
        tp.model.train()
        tp.obs_proj.train()
        sa = support(lambda: _t.nn.functional.mse_loss(
            tp.policy_mean(tp.model(tp.project_obs(X))), T))
        sb = support(lambda: compute_language_grounding_loss(
            tp.model, S, AT, phrases[:b_batch])[0])
        return len(sa & sb), len(sa - sb), len(sb - sa)

    for seed in seeds:
        # A data (T2.04 domains), B data (committed family, shipped synonyms)
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
        import torch
        torch.manual_seed(int(seed))
        from UnifiedBrain import SemanticActionAnchors as _SAA
        tp = TrainingPipeline(PipelineConfig())
        tp.model.semantic_anchors = _SAA(
            tp.model.config.d_model, num_anchors=8).to(tp.device)
        dims_ok = (obs_env == tp.config.mujoco_obs_dim == OBS_DIM
                   and act_env == tp.config.action_dim == ACT_DIM)

        Xtr_norm = tp.normalize_obs(Xtr)

        # phase 1: task A
        loss_a, fin_a = train_a(tp, Xtr_norm, Ytr, a_steps, seed * 7 + 1)
        mse_a0, det0 = eval_a(tp, Xte, Yte)
        mse_mean = float(((Ytr.mean(0) - Yte) ** 2).mean())

        # shared-substrate measurement on the phase-1 brain (no step taken)
        shared_n, a_only_n, b_only_n = grad_overlap(
            tp, Xtr_norm, Ytr, states_tr, acts_tr, phr_tr)

        snap = snapshot(tp)

        # branch SEQ: task B on the same brain
        lb_first, lb_last, fin_b = train_b(
            tp, states_tr, acts_tr, phr_tr, b_steps, seed * 7 + 2)
        acc_b_lang = eval_b_lang(tp, phr_te, cats_te)
        mse_seq, det1 = eval_a(tp, Xte, Yte)

        # branch ALONE (the null): more task A, matched steps
        restore(tp, snap)
        loss_a2, fin_a2 = train_a(tp, Xtr_norm, Ytr, b_steps, seed * 7 + 3)
        mse_alone, det2 = eval_a(tp, Xte, Yte)

        # branch CTRL: corrupted supervision on A's own inputs, matched steps
        restore(tp, snap)
        perm = np.random.RandomState(int(seed) + 41).permutation(len(Ytr))
        loss_c, fin_c = train_a(tp, Xtr_norm, Ytr[perm], b_steps, seed * 7 + 4)
        mse_ctrl, det3 = eval_a(tp, Xte, Yte)

        denom = max(mse_alone, 1e-12)
        out["seeds"].append({
            "seed": int(seed), "dims_ok": bool(dims_ok),
            "mse_a0": round(mse_a0, 6), "mse_mean": round(mse_mean, 6),
            "mse_seq": round(mse_seq, 6), "mse_alone": round(mse_alone, 6),
            "mse_ctrl": round(mse_ctrl, 6),
            "drop_seq": round((mse_seq - mse_alone) / denom, 6),
            "drop_ctrl": round((mse_ctrl - mse_alone) / denom, 6),
            "retention": round(mse_alone / max(mse_seq, 1e-12), 6),
            "acc_b_lang": round(acc_b_lang, 4),
            "z_b_lang": round(_binom_z(acc_b_lang, n_b_test), 3),
            "n_b_test": int(n_b_test),
            "loss_b_first": round(lb_first, 6), "loss_b_final": round(lb_last, 6),
            "loss_a_final": round(loss_a, 6), "loss_ctrl_final": round(loss_c, 6),
            "shared_grad_tensors": int(shared_n),
            "a_only_grad_tensors": int(a_only_n),
            "b_only_grad_tensors": int(b_only_n),
            "det_ok": bool(det0 and det1 and det2 and det3),
            "steps": {"a_phase1": a_steps, "phase2_each_branch": b_steps,
                      "a_batch": a_batch, "b_batch": b_batch},
            "finite": bool(np.isfinite(
                [mse_a0, mse_seq, mse_alone, mse_ctrl, loss_a, lb_last,
                 loss_a2, loss_c]).all()
                and fin_a and fin_b and fin_a2 and fin_c),
        })
    env.close()
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _sp, sys as _sys, os as _o
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
import json
from experiments.tests.t4_04_task_interference import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t404.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Sizing: borrowed T2.04 probe rates (see docstring — a probe at
    # production config is OWED before any real dispatch; this spec is
    # registered behind T2.01 and cannot dispatch today).
    res = submit(job, prefer="kaggle", est_hours=2.5, timeout_s=18000,
                 fetch=["t404.json"])
    if not res.ok:
        raise RuntimeError(f"T4.04 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t404.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE.get("backend", "local"),
        "task_a_retention": min(r["retention"] for r in rows),
        "retention_all": [r["retention"] for r in rows],
        "drop_seq_all": [r["drop_seq"] for r in rows],
        "drop_seq_max": max(r["drop_seq"] for r in rows),
        "mse_a0_all": [r["mse_a0"] for r in rows],
        "mse_seq_all": [r["mse_seq"] for r in rows],
        "mse_alone_all": [r["mse_alone"] for r in rows],
        "mse_mean_all": [r["mse_mean"] for r in rows],
        "acc_b_lang_all": [r["acc_b_lang"] for r in rows],
        "z_b_lang_all": [r["z_b_lang"] for r in rows],
        "loss_b_first_all": [r["loss_b_first"] for r in rows],
        "loss_b_final_all": [r["loss_b_final"] for r in rows],
        "shared_grad_tensors_all": [r["shared_grad_tensors"] for r in rows],
        "a_only_all": [r["a_only_grad_tensors"] for r in rows],
        "b_only_all": [r["b_only_grad_tensors"] for r in rows],
        "steps": rows[0]["steps"],
        "claim_all_seeds": all(r["drop_seq"] <= DROP_MAX for r in rows),
        "a_learned_all": all(r["mse_a0"] <= A_LEARN_RATIO * r["mse_mean"]
                             for r in rows),
        "b_learned_all": all(r["z_b_lang"] >= Z_B_LEARNED for r in rows),
        "shared_substrate_all": all(r["shared_grad_tensors"] > 0 for r in rows),
        "dims_ok_all": all(r["dims_ok"] for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "drop_ctrl_all": [r["drop_ctrl"] for r in rows],
        "mse_ctrl_all": [r["mse_ctrl"] for r in rows],
        "control_caught_all": all(r["drop_ctrl"] > DROP_MAX for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig lanes first: an invalid run is VOID, not evidence.
    if not m["dims_ok_all"]:
        return Status.VOID          # config contract broken (T0.14 scar)
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval not deterministic (dropout scar)
    if not m["a_learned_all"]:
        return Status.VOID          # A never learned: nothing to retain
    if not m["b_learned_all"]:
        return Status.VOID          # B never learned: "no interference" empty
    if not m["shared_substrate_all"]:
        return Status.VOID          # disjoint parameters: interference
                                    # impossible by construction (two towers)
    if not c["control_caught_all"]:
        return Status.VOID          # planted degradation unseen: dead ruler
    return m["claim_all_seeds"]


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T4.04"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


def _fixture_rows(drop_seq, drop_ctrl, **over):
    m = {
        "dims_ok_all": True, "finite_all": True, "det_ok_all": True,
        "a_learned_all": True, "b_learned_all": True,
        "shared_substrate_all": True,
        "claim_all_seeds": all(d <= DROP_MAX for d in drop_seq),
    }
    m.update({k: v for k, v in over.items() if k in m})
    c = {"control_caught_all": all(d > DROP_MAX for d in drop_ctrl)}
    return m, c


def _fixtures():
    """Replay every _check branch on synthetic dicts. Exit non-zero on any
    branch answering wrongly."""
    cases = [
        # (name, m-overrides, drop_seq, drop_ctrl, expected)
        ("pass", {}, [0.01, 0.02, 0.05], [0.9, 1.2, 0.7], True),
        ("fail_one_seed", {}, [0.01, 0.22, 0.05], [0.9, 1.2, 0.7], False),
        ("fail_boundary", {}, [0.1001, 0.0, 0.0], [0.9, 1.2, 0.7], False),
        ("pass_boundary", {}, [0.10, 0.10, 0.10], [0.9, 1.2, 0.7], True),
        ("void_dims", {"dims_ok_all": False}, [0.0] * 3, [0.9] * 3,
         Status.VOID),
        ("void_nonfinite", {"finite_all": False}, [0.0] * 3, [0.9] * 3,
         Status.VOID),
        ("void_nondet", {"det_ok_all": False}, [0.0] * 3, [0.9] * 3,
         Status.VOID),
        ("void_a_unlearned", {"a_learned_all": False}, [0.0] * 3, [0.9] * 3,
         Status.VOID),
        ("void_b_unlearned", {"b_learned_all": False}, [0.0] * 3, [0.9] * 3,
         Status.VOID),
        ("void_two_towers", {"shared_substrate_all": False}, [0.0] * 3,
         [0.9] * 3, Status.VOID),
        ("void_dead_ruler", {}, [0.0] * 3, [0.05, 0.9, 0.9], Status.VOID),
        ("void_dead_ruler_boundary", {}, [0.0] * 3, [0.10, 0.9, 0.9],
         Status.VOID),
    ]
    bad = 0
    for name, over, ds, dc, want in cases:
        m, c = _fixture_rows(ds, dc, **over)
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
        # SHAPE (default d512 brain — the language/anchors dims only agree at
        # the shipped config, so a thin-model smoke would exercise a brain
        # this spec never trains). Step counts are far too small to learn;
        # the smoke asserts PLUMBING (dims, determinism, finiteness, all
        # four branches run, snapshot/restore isolation) and measures the
        # shared-substrate overlap for real.
        out = remote_run([0], n_train=300, n_test=40, a_steps=25, b_steps=15,
                         a_batch=32, b_batch=16, n_b_train=200,
                         n_b_test_per_cat=5)
        print(json.dumps(out, indent=1))
        r = out["seeds"][0]
        assert r["dims_ok"] and r["det_ok"] and r["finite"], r
        assert r["shared_grad_tensors"] >= 0 and r["steps"], r
        print("SMOKE OK — shared_grad_tensors", r["shared_grad_tensors"],
              "a_only", r["a_only_grad_tensors"],
              "b_only", r["b_only_grad_tensors"])
    else:
        run()
