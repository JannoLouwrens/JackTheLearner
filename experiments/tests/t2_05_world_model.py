"""T2.05 — World model beats constant prediction.

HYPOTHESIS (registry). k-step latent prediction error < a persistence
baseline. Falsified by: predicting 'next state = current state' does as well.
Nulls: persistence and mean-state. Metric: k_step_mse.

WHAT THIS MEASURES, SAID PLAINLY. The repo ships a full TD-MPC2 WorldModel
(`UnifiedBrain.WorldModel`: encoder / action-conditioned dynamics / decoder /
imagine_trajectory) that no pipeline code had ever instantiated — the
`enable_world_model` flag existed in UnifiedBrainConfig and nothing forwarded
it (the passthrough in PipelineConfig was added FOR this spec). This is the
first gradient the component has ever received. The claim: the SHIPPED
imagination path — obs → `normalize_obs` → `obs_proj` → `UnifiedBrain.forward`
→ cls_features → `world_model.encode` → K dynamics steps (`imagine_trajectory`,
the exact path `UnifiedBrain.imagine` owns) → `world_model.decoder` — trained
end-to-end on real Humanoid-v5 trajectories, predicts the observation K steps
ahead better than copying the current observation.

THE RULER, and why it is NOT latent error. The registry says "latent
prediction error", but a latent ruler is owned by the model under test: a
collapsed latent (constant z) scores zero prediction error on everything while
predicting nothing — the same disease as LC.03's life-gain ruler measuring the
world instead of the learner, one level down. So error is measured in z-scored
RAW observation space (train-set statistics, the same affine for every arm and
both nulls), reached from the decoder through a jointly-trained linear readout
(256→348). The readout exists because the shipped decoder targets
`config.obs_dim=256`, a dimension that corresponds to nothing the real
pipeline observes — recorded here as a finding about the component, not
repaired silently. The readout cannot cheat the ruler: the targets are fixed,
and a model that only matches persistence fails the 20% margin.

HORIZON. K = 5 — the shipped `imagination_horizon` default. Deep supervision
at horizons 1..K during training; the gate reads horizon K only.

DEMOS. Same collection machinery shape as T2.04 (scripted state-feedback
controller + exploration noise widens the visited-state distribution; the
T2.03 mod-2**32 seed guard). Referenced, not transcribed: `script_action` is
imported from t2_04 and hashed via IMPL_DEPS. Its purpose there and here is
the same — a diverse, reproducible visited-state distribution — so the
referenced-constant lesson's "same regime?" question is answered yes. The
EXECUTED action (script + noise, the thing the dynamics actually received) is
what conditions the rollout. Windows of K+1 contiguous steps within one
episode; train and test draw from disjoint derived-seed domains.

THE ARMS:
  wm       the shipped imagination path trained end-to-end (obs_proj + trunk +
           encoder + dynamics + decoder + readout), Adam/MSE, deep supervision.
  persist  THE NULL: predict z(obs_{t+K}) = z(obs_t).
  mean     predict the train-target mean — the scale anchor.
  ridge    the reference arm simple enough that its failure indicts the task
           (T1.02 lesson): linear map from [z(obs_t), actions] to the K-step
           RESIDUAL (so l2=inf reproduces persistence exactly), l2 chosen on a
           deterministic train-internal split.

CONTROL (registry, must fail): identical training on a shuffled (window,
target) pairing — inputs and actions keep their true joint distribution, the
future they are asked to predict belongs to another window. Must NOT beat
persistence. A control that clears its gate is VOID, not FAIL.

PRE-REGISTERED GATES — relative/exogenous, no pilot constants:
  CLAIM    mse_wm <= WM_RATIO * mse_persist on EVERY seed (WM_RATIO = 0.8:
           at least 20% below persistence, per seed, reported per seed).
  CONTROL  mse_shuffled >= mse_persist on every seed.
  RIG → VOID, not FAIL (an invalid run is not evidence):
    - obs/action dims match the config contract (348/17 — T0.14 scar) and
      `tp.model.world_model` is not None (the wiring this spec exists to test);
    - every recorded loss finite;
    - eval determinism ASSERTED, not hoped (the .eval() scar): the full eval
      runs twice and the two prediction arrays must be bit-identical;
    - mse_persist < mse_mean — persistence no better than the mean means the
      horizon is degenerate and the comparison would be against noise;
    - if the claim fails AND ridge also fails to beat persistence, the TASK is
      indicted, not the model (T1.02 lesson) → VOID.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the 5.5-GPU-hour scar). Kaggle first: W32's
expiring hours die Sunday, and a Kaggle kernel computes server-side even if
the local watcher dies (JACK_REUSE_KERNEL reattaches at zero quota). Science
code lives HERE and the JOB string only imports it (T0.16).

COVERS: world model (claim).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit
from .t2_04_behaviour_cloning import script_action, ACTION_LIMIT, EXPLORE_STD

# The claim is about the shipped imagination path; all three hash into the
# cert (t2_04 supplies the scripted controller the demos are drawn with).
IMPL_DEPS = ["TrainingPipeline.py", "UnifiedBrain.py",
             "experiments/tests/t2_04_behaviour_cloning.py"]

SEEDS = [0, 1, 2]

K = 5                            # shipped imagination_horizon default
N_TRAIN, N_TEST = 3000, 1000     # windows
EP_CAP = 100                     # steps per episode before a fresh reset
OBS_DIM, ACT_DIM = 348, 17

WM_STEPS = 1200
WM_BATCH = 256
WM_LR = 3e-4
L2_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
VAL_EVERY = 5                    # every 5th train row is the l2-selection split

# Pre-registered gate (relative — see docstring). FINAL.
WM_RATIO = 0.8


def _collect_windows(env, n: int, seed: int, domain: int) -> tuple:
    """n contiguous (obs_t, a_t..a_{t+K-1}, obs_{t+1}..obs_{t+K}) windows from
    scripted rollouts with exploration noise. The stored action is the
    EXECUTED one — the thing the dynamics actually received. domain 0 = train,
    1 = test — disjoint derived-seed spaces, mod 2**32 (the T2.03 scar)."""
    X0 = np.empty((n, OBS_DIM), dtype=np.float64)
    A = np.empty((n, K, ACT_DIM), dtype=np.float64)
    Y = np.empty((n, K, OBS_DIM), dtype=np.float64)
    i, ep, n_epis, n_falls, n_short = 0, 0, 0, 0, 0
    while i < n:
        ep_seed = int((seed * 1_000_003 + domain * 500_009 + ep * 9_176) % 2**32)
        ep += 1
        n_epis += 1
        if n_epis > 50 * (n // (EP_CAP - K) + 1) + 100:
            raise RuntimeError("window collection is not converging; "
                               "episodes too short for K+1 steps?")
        obs, _ = env.reset(seed=ep_seed)
        rng = np.random.RandomState((ep_seed + 1) % 2**32)
        obs_seq, act_seq = [obs], []
        for _t in range(EP_CAP):
            act = np.clip(script_action(obs) + EXPLORE_STD * rng.randn(ACT_DIM),
                          -ACTION_LIMIT, ACTION_LIMIT)
            obs, _r, terminated, truncated, _info = env.step(act)
            act_seq.append(act)
            obs_seq.append(obs)
            if terminated or truncated:
                if terminated:
                    n_falls += 1
                break
        L = len(act_seq)             # transitions in this episode
        if L < K:
            n_short += 1
            continue
        for t in range(L - K + 1):   # windows: obs_t + K executed steps
            X0[i] = obs_seq[t]
            A[i] = np.stack(act_seq[t:t + K])
            Y[i] = np.stack(obs_seq[t + 1:t + K + 1])
            i += 1
            if i >= n:
                break
    return X0, A, Y, n_epis, n_falls, n_short


# ── the ruler: one affine z-transform for every arm and both nulls ────────
def _z_stats(tp) -> tuple:
    """(mu, sd) from the pipeline's merged running stats, as numpy. Frozen at
    read time — applying them mutates nothing (normalize_obs would)."""
    mu = tp.obs_mean.detach().cpu().numpy().astype(np.float64)
    sd = np.sqrt(tp.obs_var.detach().cpu().numpy().astype(np.float64) + 1e-8)
    return mu, sd


def _z(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """The exact normalize_obs transform (z-score, clip ±10), no mutation."""
    return np.clip((X - mu) / sd, -10.0, 10.0).astype(np.float32)


# ── the learner: the shipped imagination path, trained end-to-end ─────────
def _rollout_predictions(tp, readout, X0z, A):
    """z-space predictions at horizons 1..K through the SHIPPED path:
    project → forward → cls_features → wm.encode → imagine_trajectory →
    decoder → readout. Returns (B, K, OBS_DIM) torch tensor on tp.device."""
    import torch
    wm = tp.model.world_model
    out = tp.model(tp.project_obs(X0z))
    lat = wm.encode(out["cls_features"])
    latents, _rewards = wm.imagine_trajectory(lat, A)   # (B, K+1, latent)
    return readout(wm.decoder(latents[:, 1:]))          # (B, K, OBS_DIM)


def _train_wm(tp, readout, X0z, A, Yz, seed: int) -> float:
    """Adam/MSE with deep supervision at horizons 1..K. Mode discipline:
    .train() here, .eval() asserted in _eval_deterministic (dropout scar)."""
    import torch
    device = tp.device
    X = torch.tensor(X0z, dtype=torch.float32, device=device)
    Am = torch.tensor(A, dtype=torch.float32, device=device)
    Ym = torch.tensor(Yz, dtype=torch.float32, device=device)
    params = (list(tp.model.parameters()) + list(tp.obs_proj.parameters())
              + list(readout.parameters()))
    opt = torch.optim.Adam(params, lr=WM_LR)
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 1)
    tp.model.train()
    tp.obs_proj.train()
    readout.train()
    loss_val = float("nan")
    for _step in range(WM_STEPS):
        idx = torch.randint(0, len(X), (WM_BATCH,), generator=g).to(device)
        pred = _rollout_predictions(tp, readout, X[idx], Am[idx])
        loss = torch.nn.functional.mse_loss(pred, Ym[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_val = float(loss.item())
    return loss_val


def _eval_deterministic(tp, readout, X0z, A) -> tuple:
    """(k-step predictions (B, OBS_DIM) at horizon K, det_ok). Full eval run
    twice in eval mode under no_grad; bit-identity is the dropout-scar
    assertion. Nothing here mutates running stats (frozen _z transform)."""
    import torch

    def one_pass():
        tp.model.eval()
        tp.obs_proj.eval()
        readout.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X0z), WM_BATCH):
                X = torch.tensor(X0z[i:i + WM_BATCH], dtype=torch.float32,
                                 device=tp.device)
                Am = torch.tensor(A[i:i + WM_BATCH], dtype=torch.float32,
                                  device=tp.device)
                outs.append(_rollout_predictions(tp, readout, X, Am)[:, -1]
                            .cpu().numpy())
        return np.concatenate(outs)

    pred1 = one_pass()
    pred2 = one_pass()
    return pred1, bool(np.array_equal(pred1, pred2))


# ── the reference arm ─────────────────────────────────────────────────────
def _ridge_mse(X0z_tr, Atr, Yz_tr, X0z_te, Ate, Yz_te) -> tuple:
    """(mse, chosen l2). Linear map [z(obs), actions] -> K-step RESIDUAL, so
    l2=inf reproduces persistence exactly; l2 picked on the deterministic
    every-5th-row split of TRAIN (test labels never consulted)."""
    def feats(X0, A):
        return np.concatenate([X0, A.reshape(len(A), -1)], axis=1)

    Ftr = feats(X0z_tr, Atr).astype(np.float64)
    Fte = feats(X0z_te, Ate).astype(np.float64)
    Rtr = (Yz_tr[:, -1] - X0z_tr).astype(np.float64)     # residual targets
    val = np.arange(len(Ftr)) % VAL_EVERY == 0
    fit = ~val

    def solve(F, R, l2):
        d = F.shape[1]
        return np.linalg.solve(F.T @ F + l2 * np.eye(d), F.T @ R)

    rm = Rtr[fit].mean(0)
    best_l2, best = None, np.inf
    for l2 in L2_GRID:
        W = solve(Ftr[fit], Rtr[fit] - rm, l2)
        pred = X0z_tr[val] + Ftr[val] @ W + rm
        err = float(((pred - Yz_tr[val, -1]) ** 2).mean())
        if err < best:
            best, best_l2 = err, l2
    rm = Rtr.mean(0)
    W = solve(Ftr, Rtr - rm, best_l2)
    pred = X0z_te + Fte @ W + rm
    return float(((pred - Yz_te[:, -1]) ** 2).mean()), best_l2


# ── remote entry point (also the local smoke, scaled down) ────────────────
def remote_run(seeds: list, n_train: int = N_TRAIN, n_test: int = N_TEST,
               wm_steps: int = WM_STEPS, wm_batch: int = WM_BATCH,
               pipe_kwargs: dict | None = None) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally, tiny,
    for the smoke). Returns the JSON-able result dict."""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    global WM_BATCH
    WM_BATCH = wm_batch          # eval chunking follows the training batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    env = gym.make("Humanoid-v5")
    obs_env = int(env.observation_space.shape[0])
    act_env = int(env.action_space.shape[0])
    kw = dict(pipe_kwargs or {})
    kw["enable_world_model"] = True

    for seed in seeds:
        X0tr, Atr, Ytr, ep_tr, falls_tr, short_tr = _collect_windows(
            env, n_train, seed, domain=0)
        X0te, Ate, Yte, ep_te, _, _ = _collect_windows(
            env, n_test, seed, domain=1)

        # wm — the shipped path. Weight init seeded per spec seed.
        torch.manual_seed(int(seed))
        tp = TrainingPipeline(PipelineConfig(**kw))
        readout = nn.Linear(tp.model.config.obs_dim, OBS_DIM).to(tp.device)
        dims_ok = (obs_env == tp.config.mujoco_obs_dim == OBS_DIM
                   and act_env == tp.config.action_dim == ACT_DIM
                   and tp.model.world_model is not None)

        # One merge of train stats; the frozen affine is the ruler everywhere.
        tp.normalize_obs(np.concatenate([X0tr, Ytr.reshape(-1, OBS_DIM)]))
        mu, sd = _z_stats(tp)
        X0z_tr, Yz_tr = _z(X0tr, mu, sd), _z(Ytr, mu, sd)
        X0z_te, Yz_te = _z(X0te, mu, sd), _z(Yte, mu, sd)

        loss_wm = _train_wm(tp, readout, X0z_tr, Atr, Yz_tr, seed)
        pred_wm, det_ok = _eval_deterministic(tp, readout, X0z_te, Ate)
        mse_wm = float(((pred_wm - Yz_te[:, -1]) ** 2).mean())

        # the nulls and the reference
        mse_persist = float(((X0z_te - Yz_te[:, -1]) ** 2).mean())
        mean_tr = Yz_tr[:, -1].mean(0)
        mse_mean = float(((mean_tr - Yz_te[:, -1]) ** 2).mean())
        mse_ridge, l2_ridge = _ridge_mse(X0z_tr, Atr, Yz_tr,
                                         X0z_te, Ate, Yz_te)

        # control — identical training on a shuffled (window, target)
        # pairing, fresh weights.
        torch.manual_seed(int(seed) + 7)
        tp_sh = TrainingPipeline(PipelineConfig(**kw))
        readout_sh = nn.Linear(tp_sh.model.config.obs_dim,
                               OBS_DIM).to(tp_sh.device)
        tp_sh.normalize_obs(np.concatenate([X0tr, Ytr.reshape(-1, OBS_DIM)]))
        perm = np.random.RandomState(int(seed) + 41).permutation(len(Yz_tr))
        loss_sh = _train_wm(tp_sh, readout_sh, X0z_tr, Atr, Yz_tr[perm], seed)
        pred_sh, det_ok_sh = _eval_deterministic(tp_sh, readout_sh, X0z_te, Ate)
        mse_shuffled = float(((pred_sh - Yz_te[:, -1]) ** 2).mean())

        out["seeds"].append({
            "seed": int(seed), "dims_ok": bool(dims_ok),
            "obs_env": obs_env, "act_env": act_env,
            "mse_wm": round(mse_wm, 6), "mse_persist": round(mse_persist, 6),
            "mse_mean": round(mse_mean, 6), "mse_ridge": round(mse_ridge, 6),
            "mse_shuffled": round(mse_shuffled, 6),
            "l2_ridge": l2_ridge,
            "loss_wm_final": round(loss_wm, 6),
            "loss_shuffled_final": round(loss_sh, 6),
            "det_ok": bool(det_ok and det_ok_sh),
            "episodes_train": ep_tr, "falls_train": falls_tr,
            "short_episodes_train": short_tr,
            "episodes_test": ep_te,
            "finite": bool(np.isfinite([mse_wm, mse_persist, mse_mean,
                                        mse_ridge, mse_shuffled,
                                        loss_wm, loss_sh]).all()),
        })
    env.close()
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ───────────
JOB = r'''
import subprocess as _sp, sys as _sys, os as _o
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)
import json
from experiments.tests.t2_05_world_model import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t205.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # SIZED FROM A PROBE AT THE PRODUCTION CONFIG on the target GPU (the
    # T2.04/B1 lesson: a cost measured on the smoke's configuration is not a
    # cost for the production configuration). Probe kernel 1786702211
    # (P100, PipelineConfig() defaults, 2026-08-14): train 0.4276 s/step,
    # collect 0.00072 s/row, build 1.54 s, eval(600 rows, 2 passes) 0.68 s.
    # Per seed: 2 trainings (wm + shuffled) x 1200 steps x 0.4276 = 1026 s,
    # collect 4000 rows ~3 s, 2 builds ~3 s, 2 evals ~3 s, ridge ~10 s
    # -> ~1045 s. 3 seeds ~3135 s + clone/pip setup ~180 s = ~3315 s = 0.92 h.
    # timeout_s 7200 caps the remote run at 7140 s = 2.15x measured;
    # est_hours 1.2 = 1.3x measured (afford() gates on it, charge() bills
    # actual — the declaration errs high on purpose).
    res = submit(job, prefer="kaggle",
                 est_hours=1.2,
                 timeout_s=7200,
                 fetch=["t205.json"])
    if not res.ok:
        raise RuntimeError(f"T2.05 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t205.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    ratios = [r["mse_wm"] / r["mse_persist"] for r in rows]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "k_step_mse": [r["mse_wm"] for r in rows],
        "mse_persist": [r["mse_persist"] for r in rows],
        "mse_mean": [r["mse_mean"] for r in rows],
        "mse_ridge": [r["mse_ridge"] for r in rows],
        "horizon_k": K,
        "wm_ratio_max": round(max(ratios), 4),
        "wm_ratio_all": [round(x, 4) for x in ratios],
        "all_seeds_beat_null": all(r["mse_wm"] <= WM_RATIO * r["mse_persist"]
                                   for r in rows),
        "ridge_beats_null_any": any(r["mse_ridge"] < r["mse_persist"]
                                    for r in rows),
        "persist_informative_all": all(r["mse_persist"] < r["mse_mean"]
                                       for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "dims_ok_all": all(r["dims_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
        "loss_wm_final_max": max(r["loss_wm_final"] for r in rows),
        "falls_train": [r["falls_train"] for r in rows],
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "mse_shuffled": [r["mse_shuffled"] for r in rows],
        "shuffled_beats_null": any(r["mse_shuffled"] < r["mse_persist"]
                                   for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["dims_ok_all"]:
        return Status.VOID          # config contract broken / wm not wired
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing was measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval path not deterministic (dropout scar)
    if not m["persist_informative_all"]:
        return Status.VOID          # persistence no better than the mean:
                                    # degenerate horizon, comparison meaningless
    # Control: information-free supervision must NOT beat persistence.
    if c["shuffled_beats_null"]:
        return Status.VOID          # the metric leaks; not measuring prediction
    claim = m["all_seeds_beat_null"]
    if not claim and not m["ridge_beats_null_any"]:
        return Status.VOID          # the reference arm fails too: task
                                    # indicted, not the model (T1.02 lesson)
    return claim


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.05"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, tiny, CPU: exercises the REAL code paths at production-shaped
        # argument extremes (largest derived seed via seed=2 + test domain;
        # PRODUCTION K; termination path; window building at episode edges;
        # both training loops; eval bit-identity) on a small PipelineConfig
        # and batch so it fits this box.
        out = remote_run([2], n_train=80, n_test=40, wm_steps=6, wm_batch=32,
                         pipe_kwargs={"d_model": 64, "n_layers": 2})
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert row["det_ok"] and row["dims_ok"] and row["finite"], row
        print("SMOKE OK")
    else:
        run()
