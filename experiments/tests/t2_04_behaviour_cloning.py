"""T2.04 — Behaviour cloning on scripted trajectories.

HYPOTHESIS (registry). The action head reproduces scripted MuJoCo trajectories
above a nearest-neighbour baseline. Falsified by: fails to beat NN retrieval.
Null: nearest-neighbour lookup in the demo set. Metric: action_mse.

WHAT THIS MEASURES, SAID PLAINLY. Can the SHIPPED action path — obs →
`normalize_obs` → `obs_proj` → `UnifiedBrain` → `policy_mean` (tanh-squashed),
the exact path `act_deterministic` owns — be supervised into imitating a
deterministic controller from (obs, action) pairs, better than looking the
answer up? T1.01 proved each module can memorise ONE batch; this is the first
supervised claim on real MuJoCo trajectories with a held-out set. It needs no
external dataset — the registry note records why (the CMU MoCap URLs 404 and
the old loader fabricated sinusoids; T1.13 lesson).

THE SCRIPT — a fixed deterministic state-feedback law, committed here:

    jang = obs[5:22]   (17 joint angles; Humanoid-v5 layout, qpos[2:] first)
    jvel = obs[28:45]  (17 joint velocities)
    u    = KP*(0 - jang) - KD*jvel + 0.5*tanh(M @ jang)
    a    = ACTION_LIMIT * tanh(u)

KP, KD, M are module constants drawn once from RandomState(240814) — the
"script" is this committed function, not a family. The label is a pure
function of obs, so cloning it is well-posed; the tanh mixing term keeps it
off the purely-linear manifold (the ridge reference arm measures how much).

DEMOS. Episodes reset with gymnasium's own randomisation (derived seeds, all
mod 2**32 — the T2.03 overflow scar). Each step records (obs, a_script(obs))
and EXECUTES clip(a_script + 0.1*eps, ±0.4): the exploration noise widens the
visited state distribution (falls included — they are states too) while the
label stays the script's exact answer at the visited state. Train and test
draw from disjoint derived-seed domains. Episodes end at termination or
EP_CAP steps; collection runs until the quota of pairs is met.

THE ARMS:
  bc        the shipped path, trained end-to-end (obs_proj + trunk + head)
            with Adam/MSE. Evaluated per-row through `act_deterministic` —
            the ONE deterministic path (T0.16); nothing here re-implements the
            forward.
  nn        THE NULL: 1-nearest-neighbour in z-scored (train-stats) obs space,
            answering with the neighbour's stored action.
  ridge     the reference arm simple enough that its failure indicts the task
            (T1.02 lesson): multi-output ridge on z-scored obs, l2 chosen on a
            deterministic train-internal split.
  mean      predict the train-mean action — the scale anchor.

CONTROL (registry, must fail): the same shipped path trained identically on a
SHUFFLED (obs, action) pairing must NOT beat the nearest-neighbour null. If
information-free supervision beats real retrieval, the metric is not measuring
imitation. A control that clears its gate is VOID, not FAIL (bakeoff lesson).

PRE-REGISTERED GATES — all relative or exogenous, no pilot constants (T2.08
anti-lottery lesson):
  CLAIM    mse_bc <= CLONE_RATIO * mse_nn on EVERY seed (CLONE_RATIO = 0.8:
           at least 20% below the null, per seed, reported per seed).
  CONTROL  mse_shuffled >= mse_nn on every seed.
  RIG → VOID, not FAIL (an invalid run is not evidence):
    - obs/action dims must match the config contract (348/17 — T0.14 scar);
    - every recorded loss finite;
    - eval determinism ASSERTED, not hoped (the .eval() scar): the full
      per-row eval is run twice with the running-normalisation state
      snapshot-restored between passes, and the two prediction arrays must be
      bit-identical;
    - mse_nn < mse_mean — a null no better than the mean means the demo set
      is degenerate and the comparison would be against noise;
    - if the claim fails AND ridge also fails to beat nn, the TASK is
      indicted, not the head (T1.02 lesson) → VOID.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the 5.5-GPU-hour scar). Kaggle first: W32's
expiring hours are assigned to this spec (OVERSIGHT B4), and a Kaggle kernel
computes server-side even if the local watcher dies (JACK_REUSE_KERNEL
reattaches at zero quota). Science code lives HERE and the JOB string only
imports it (T0.16: code in strings is invisible to every guard).

Declares no `COVERS:` commitment. "action head" is not a name in
`coverage.COMMITMENTS`, and the registry — the copy `declarations()` reads,
and the authoritative one — carries no marker for T2.04. An earlier revision
wrote `action head (claim)` here; that bought nothing and read as a claim
(26th audit B2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped action path; both files hash into the cert.
IMPL_DEPS = ["TrainingPipeline.py", "UnifiedBrain.py"]

SEEDS = [0, 1, 2]

N_TRAIN, N_TEST = 3000, 1000
EP_CAP = 100                     # steps per episode before a fresh reset
EXPLORE_STD = 0.1                # executed = clip(label + std*eps)
ACTION_LIMIT = 0.4               # Humanoid-v5 actuator range (config contract)
OBS_DIM, ACT_DIM = 348, 17

BC_STEPS = 1200
BC_BATCH = 256
BC_LR = 3e-4
L2_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
VAL_EVERY = 5                    # every 5th train row is the l2-selection split

# Pre-registered gates (relative/exogenous — see docstring). FINAL.
CLONE_RATIO = 0.8

# The script's constants: drawn ONCE, committed by this literal seed.
_S = np.random.RandomState(240814)
KP = _S.uniform(1.0, 3.0, size=ACT_DIM)
KD = _S.uniform(0.1, 0.5, size=ACT_DIM)
M = (_S.randn(ACT_DIM, ACT_DIM) / np.sqrt(ACT_DIM))


def script_action(obs: np.ndarray) -> np.ndarray:
    """The scripted controller: deterministic function of one raw obs (348,)."""
    jang = obs[5:22]
    jvel = obs[28:45]
    u = KP * (0.0 - jang) - KD * jvel + 0.5 * np.tanh(M @ jang)
    return (ACTION_LIMIT * np.tanh(u)).astype(np.float64)


def _collect(env, n: int, seed: int, domain: int) -> tuple:
    """n (obs, label) pairs from scripted rollouts with exploration noise.
    domain 0 = train, 1 = test — disjoint derived-seed spaces, mod 2**32
    (numpy refuses larger; the T2.03 scar met its range on a paid kernel)."""
    X = np.empty((n, OBS_DIM), dtype=np.float64)
    Y = np.empty((n, ACT_DIM), dtype=np.float64)
    i, ep, n_epis, n_falls = 0, 0, 0, 0
    while i < n:
        ep_seed = int((seed * 1_000_003 + domain * 500_009 + ep * 9_176) % 2**32)
        ep += 1
        n_epis += 1
        obs, _ = env.reset(seed=ep_seed)
        rng = np.random.RandomState((ep_seed + 1) % 2**32)
        for _t in range(EP_CAP):
            label = script_action(obs)
            X[i], Y[i] = obs, label
            i += 1
            act = np.clip(label + EXPLORE_STD * rng.randn(ACT_DIM),
                          -ACTION_LIMIT, ACTION_LIMIT)
            obs, _r, terminated, truncated, _info = env.step(act)
            if i >= n or terminated or truncated:
                if terminated:
                    n_falls += 1
                break
    return X, Y, n_epis, n_falls


# ── the learner: the shipped path, trained end-to-end ────────────────────
def _train_bc(tp, Xtr_norm: np.ndarray, Ytr: np.ndarray, seed: int) -> float:
    """Adam/MSE on the shipped forward (obs_proj + trunk + policy_mean).
    Returns the final training loss. Mode discipline: .train() here; every
    evaluation goes through act_deterministic, which guarantees .eval()."""
    import torch
    device = tp.device
    X = torch.tensor(Xtr_norm, dtype=torch.float32, device=device)
    Y = torch.tensor(Ytr, dtype=torch.float32, device=device)
    params = list(tp.model.parameters()) + list(tp.obs_proj.parameters())
    opt = torch.optim.Adam(params, lr=BC_LR)
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 1)
    tp.model.train()
    tp.obs_proj.train()
    loss_val = float("nan")
    for _step in range(BC_STEPS):
        idx = torch.randint(0, len(X), (BC_BATCH,), generator=g).to(device)
        mean = tp.policy_mean(tp.model(tp.project_obs(X[idx])))
        loss = torch.nn.functional.mse_loss(mean, Y[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_val = float(loss.item())
    return loss_val


def _eval_rows(tp, Xte: np.ndarray) -> np.ndarray:
    """Per-row predictions through act_deterministic — the shipped single-obs
    API (it returns mean[0]; batching it would mean re-implementing the
    forward, which is the exact T0.16 anti-pattern)."""
    return np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])


def _eval_deterministic(tp, Xte: np.ndarray) -> tuple:
    """(predictions, det_ok). Runs the full eval twice with the running-
    normalisation state snapshot-restored between passes; bit-identity is the
    dropout-scar assertion. normalize_obs mutates its running stats on every
    call, so without the snapshot the two passes would differ for a reason
    that has nothing to do with eval mode."""
    snap = (tp.obs_mean.clone(), tp.obs_var.clone(), tp.obs_count)
    pred1 = _eval_rows(tp, Xte)
    tp.obs_mean, tp.obs_var, tp.obs_count = snap[0].clone(), snap[1].clone(), snap[2]
    pred2 = _eval_rows(tp, Xte)
    tp.obs_mean, tp.obs_var, tp.obs_count = snap
    return pred1, bool(np.array_equal(pred1, pred2))


# ── the null and the reference arms ──────────────────────────────────────
def _zscore_stats(Xtr: np.ndarray) -> tuple:
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def _nn_mse(Xtr, Ytr, Xte, Yte) -> float:
    """1-NN in z-scored obs space, chunked (1000x3000 fits, but stay flat)."""
    mu, sd = _zscore_stats(Xtr)
    A = ((Xtr - mu) / sd).astype(np.float32)
    B = ((Xte - mu) / sd).astype(np.float32)
    a2 = (A * A).sum(1)
    preds = np.empty_like(Yte)
    for i in range(0, len(B), 256):
        b = B[i:i + 256]
        d = a2[None, :] - 2.0 * (b @ A.T)      # + b2, constant per row
        preds[i:i + 256] = Ytr[d.argmin(1)]
    return float(((preds - Yte) ** 2).mean())


def _ridge_mse(Xtr, Ytr, Xte, Yte) -> tuple:
    """(mse, chosen l2). Multi-output ridge on z-scored obs; l2 picked on the
    deterministic every-5th-row split of TRAIN (test labels never consulted)."""
    mu, sd = _zscore_stats(Xtr)
    A = ((Xtr - mu) / sd).astype(np.float64)
    B = ((Xte - mu) / sd).astype(np.float64)
    val = np.arange(len(A)) % VAL_EVERY == 0
    fit = ~val

    def solve(X, Y, l2):
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + l2 * np.eye(d), X.T @ Y)

    ym = Ytr[fit].mean(0)
    best_l2, best = None, np.inf
    for l2 in L2_GRID:
        W = solve(A[fit], Ytr[fit] - ym, l2)
        err = float(((A[val] @ W + ym - Ytr[val]) ** 2).mean())
        if err < best:
            best, best_l2 = err, l2
    ym = Ytr.mean(0)
    W = solve(A, Ytr - ym, best_l2)
    return float(((B @ W + ym - Yte) ** 2).mean()), best_l2


# ── remote entry point (also the local smoke, scaled down) ───────────────
def remote_run(seeds: list, n_train: int = N_TRAIN, n_test: int = N_TEST,
               bc_steps: int = BC_STEPS, pipe_kwargs: dict | None = None) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally, tiny,
    for the smoke). Returns the JSON-able result dict."""
    import gymnasium as gym
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    env = gym.make("Humanoid-v5")
    obs_env = int(env.observation_space.shape[0])
    act_env = int(env.action_space.shape[0])
    for seed in seeds:
        Xtr, Ytr, ep_tr, falls_tr = _collect(env, n_train, seed, domain=0)
        Xte, Yte, ep_te, _ = _collect(env, n_test, seed, domain=1)

        # bc — the shipped path. Weight init seeded per spec seed.
        torch.manual_seed(int(seed))
        tp = TrainingPipeline(PipelineConfig(**(pipe_kwargs or {})))
        dims_ok = (obs_env == tp.config.mujoco_obs_dim == OBS_DIM
                   and act_env == tp.config.action_dim == ACT_DIM)
        Xtr_norm = tp.normalize_obs(Xtr)         # one batch merge: train stats
        loss_bc = _train_bc(tp, Xtr_norm, Ytr, seed)
        pred_bc, det_ok = _eval_deterministic(tp, Xte)
        mse_bc = float(((pred_bc - Yte) ** 2).mean())

        # the null and the references
        mse_nn = _nn_mse(Xtr, Ytr, Xte, Yte)
        mse_ridge, l2_ridge = _ridge_mse(Xtr, Ytr, Xte, Yte)
        mse_mean = float(((Ytr.mean(0) - Yte) ** 2).mean())

        # control — identical training on a shuffled pairing, fresh weights.
        torch.manual_seed(int(seed) + 7)
        tp_sh = TrainingPipeline(PipelineConfig(**(pipe_kwargs or {})))
        Xtr_norm_sh = tp_sh.normalize_obs(Xtr)
        perm = np.random.RandomState(int(seed) + 41).permutation(len(Ytr))
        loss_sh = _train_bc(tp_sh, Xtr_norm_sh, Ytr[perm], seed)
        pred_sh, det_ok_sh = _eval_deterministic(tp_sh, Xte)
        mse_shuffled = float(((pred_sh - Yte) ** 2).mean())

        out["seeds"].append({
            "seed": int(seed), "dims_ok": bool(dims_ok),
            "obs_env": obs_env, "act_env": act_env,
            "mse_bc": round(mse_bc, 6), "mse_nn": round(mse_nn, 6),
            "mse_ridge": round(mse_ridge, 6), "mse_mean": round(mse_mean, 6),
            "mse_shuffled": round(mse_shuffled, 6),
            "l2_ridge": l2_ridge,
            "loss_bc_final": round(loss_bc, 6),
            "loss_shuffled_final": round(loss_sh, 6),
            "det_ok": bool(det_ok and det_ok_sh),
            "episodes_train": ep_tr, "falls_train": falls_tr,
            "episodes_test": ep_te,
            "finite": bool(np.isfinite([mse_bc, mse_nn, mse_ridge, mse_mean,
                                        mse_shuffled, loss_bc]).all()),
        })
    env.close()
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _sp, sys as _sys, os as _o
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
import json
from experiments.tests.t2_04_behaviour_cloning import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t204.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Kaggle first: W32's expiring hours are assigned to this spec (B4), and
    # the kernel survives a dead local watcher (JACK_REUSE_KERNEL). Timeout
    # sized from a PROBE AT THE PRODUCTION CONFIG on the target P100
    # (overseer 16th audit B1: the earlier declaration extrapolated the d64/
    # 2-layer smoke's cost across a ~256x trunk-size gap — LESSONS: a cost
    # measured on the smoke's configuration is not a cost for the production
    # configuration). Probe 2026-08-14 07:10 UTC, kernel
    # jack-ladder-1786691401, PipelineConfig() defaults d512 x 8 verified in
    # its artifact: 0.4225 s/train-step (25 timed after 5 warmup),
    # 0.0157 s/eval-row (act_deterministic), 0.00063 s/collect-row.
    # Projection: 7200 train steps (1200 x 2 arms x 3 seeds) = 3042 s
    # + 12000 eval rows = 189 s + 12000 collect rows = 8 s + 6 builds = 8 s
    # = 3247 s core; + ~250 s setup (probe's whole billable window incl. pip
    # was 192 s) ~= 3500 s ~= 0.97 h -> est_hours 1.0. timeout_s 7200 is
    # ~2x the projection (queue + variance headroom, fits the runner's
    # 21600 s child timeout); billable charges only what the kernel's own
    # log reports, so the generous cap costs nothing when the run is short.
    res = submit(job, prefer="kaggle",
                 est_hours=1.0,
                 timeout_s=7200,
                 fetch=["t204.json"])
    if not res.ok:
        raise RuntimeError(f"T2.04 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t204.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    ratios = [r["mse_bc"] / r["mse_nn"] for r in rows]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "action_mse": [r["mse_bc"] for r in rows],
        "mse_nn": [r["mse_nn"] for r in rows],
        "mse_ridge": [r["mse_ridge"] for r in rows],
        "mse_mean": [r["mse_mean"] for r in rows],
        "clone_ratio_max": round(max(ratios), 4),
        "clone_ratio_all": [round(x, 4) for x in ratios],
        "all_seeds_beat_null": all(r["mse_bc"] <= CLONE_RATIO * r["mse_nn"]
                                   for r in rows),
        "ridge_beats_null_any": any(r["mse_ridge"] < r["mse_nn"] for r in rows),
        "nn_informative_all": all(r["mse_nn"] < r["mse_mean"] for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "dims_ok_all": all(r["dims_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
        "loss_bc_final_max": max(r["loss_bc_final"] for r in rows),
        "falls_train": [r["falls_train"] for r in rows],
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "mse_shuffled": [r["mse_shuffled"] for r in rows],
        "shuffled_beats_null": any(r["mse_shuffled"] < r["mse_nn"]
                                   for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["dims_ok_all"]:
        return Status.VOID          # config contract broken (T0.14 scar)
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing was measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval path is not deterministic (dropout scar)
    if not m["nn_informative_all"]:
        return Status.VOID          # the null is no better than the mean:
                                    # degenerate demos, comparison meaningless
    # Control: information-free supervision must NOT beat real retrieval.
    if c["shuffled_beats_null"]:
        return Status.VOID          # the metric leaks; it is not measuring imitation
    claim = m["all_seeds_beat_null"]
    if not claim and not m["ridge_beats_null_any"]:
        return Status.VOID          # the reference arm fails too: task indicted,
                                    # not the head (T1.02 lesson)
    return claim


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.04"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, tiny, CPU: exercises the REAL code paths at production-shaped
        # argument extremes (largest derived seed via seed=2 + test domain;
        # termination path; snapshot/restore bit-identity; both training loops)
        # on a small PipelineConfig so it stays inside this box's RAM budget.
        out = remote_run([2], n_train=120, n_test=40, bc_steps=30,
                         pipe_kwargs={"d_model": 64, "n_layers": 2})
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert row["det_ok"] and row["dims_ok"] and row["finite"], row
        print("SMOKE OK")
    else:
        run()
