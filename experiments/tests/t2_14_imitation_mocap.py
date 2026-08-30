"""T2.14 — Imitation from real motion capture.

HYPOTHESIS (registry). BC on the CMU corpus reaches held-out action error below
mean-action AND below nearest-neighbour retrieval. Falsified by: a lookup table
(NN retrieval) matches the model. Null: mean-action; nearest-neighbour
retrieval. Metric: heldout_vs_nn_ratio. Deps T1.13, T1.08 (both PASS).

WHAT THIS MEASURES, SAID PLAINLY. T2.04 proved the shipped action path can be
supervised into cloning a controller WE WROTE, on trajectories WE GENERATED —
its own docstring says it "needs no external dataset". This is the same path
asked the harder question: can it imitate REAL RECORDED HUMAN MOTION, better
than looking the answer up in the demonstrations? That is the difference
between cloning a function and cloning a creature, and it is the first claim in
this ladder trained on motion nobody in this project authored.

THE DATA IS REAL, AND THAT SENTENCE HAS A HISTORY. `MoCapLoader` used to
fabricate a sinusoid when it found no motion files and pair it with a label
drawn by `np.random.randint`; T1.13 measured what that produced (dataset_len 1,
real_f_ratio 0.0, label_signal_advantage 0.0) — a dataset that could teach
nothing while showing a clean loss curve. `mocap_cmu.py` replaced it with the
actual CMU Graphics Lab corpus (2514 AMC clips, free for all uses), and
`SkeletonRetargeter` mapped it onto the 17 Humanoid-v5 actuators. This spec
consumes 391 of those retargeted clips, 63,905 frames.

Because a GPU VM cannot see `/data/jack-data`, the derived corpus TRAVELS IN
THE REPO (`experiments/data/t214_cmu_corpus.npz`, 5.7 MB) — see
`scripts/build_t214_corpus.py` for why and how to regenerate it. The
consequence worth having: `assert_ref_is_current` already refuses a GPU job
whose HEAD is not published, so pinning the corpus to the same commit makes
that guard cover the DATA as well as the code. A RIG GATE re-checks the blob's
sha256 before any GPU hour is spent on it, so a silently regenerated or
substituted corpus voids the run instead of quietly changing what was claimed.

THE OBSERVATION AND THE LABEL — and the leak that had to be designed out.
At frame t the learner sees the CURRENT state and must produce the action that
takes the body to the NEXT recorded pose:

    obs   = concat(positions[t-1], velocities[t-1])            (34 real numbers)
    label = ActionComputer.compute_actions(...)                (17 actuators)
          = clip(KP*(positions[t] - positions[t-1]) - KD*velocities[t-1], +-0.4)

The label is computed by the SHIPPED `ActionComputer`, imported, not
re-derived — a second copy of that formula is the T0.07 anti-pattern. Note
what it implies: since velocities[t] = (positions[t]-positions[t-1])/dt, the
label equals clip(KP*dt*velocities[t] - KD*velocities[t-1], +-0.4), so ONE of
its two terms is already in the observation and the other requires predicting
where the motion goes next. That residual is the learnable content, and it is
why `positions[t]` is kept OUT of the observation: put it in and the label
becomes exact arithmetic on the input, and every arm scores well while nothing
is measured.

WHY THIS TARGET AND NOT THE OBVIOUS ONE — measured, not argued. Probe
2026-08-30 (`scripts/t214_probe.py`, artifact `/data/t214_probe.json`), same
corpus, same clip-disjoint split, on the three candidate targets the shipped
loader documents:

    target      mse_mean   mse_nn    mse_ridge   nn/mean   ridge/nn
    pd          0.01939    0.01202   0.00807     0.62      0.67      <- chosen
    nextpose    0.04899    0.01700   0.000011    0.35      0.0006
    delta       0.000073   0.000038  0.000011    0.52      0.28

`nextpose` (predict positions[t] directly) is DEGENERATE: a linear map beats
nearest-neighbour retrieval by ~1600x, because next pose is current pose plus
a smooth extrapolation. A spec built on it would have recorded a confident
PASS about arithmetic. The shipped PD action is the only one of the three that
leaves both registry nulls informative AND ordered (mean > nn > ridge) with no
order-of-magnitude collapse. That choice is rig design and it was made before
any gate below was frozen; the numbers are disclosed here so an auditor sees
exactly what was on the screen.

SPLITS ARE CLIP-DISJOINT, and this is load-bearing. Splitting by FRAME would
let 1-NN retrieve frame t+1 of the very clip it is being tested on — leakage
wearing a null's clothes, and it would make the null unbeatable for the wrong
reason. Train and test share no clip; the disjointness is ASSERTED as a rig
gate, not assumed. All arms see the SAME 20,000 training rows: giving the
learner more demonstrations than the retrieval null would flatter the claim.

THE ARMS:
  bc      the shipped path, trained end-to-end (obs_proj + trunk + policy_mean)
          with Adam/MSE and evaluated per row through `act_deterministic` —
          the ONE deterministic path (T0.16). The 34-dim mocap state is placed
          into the Humanoid-v5 observation layout the config contract declares
          (joint angles at obs[5:22], joint velocities at obs[28:45], the rest
          zero), the same slots T2.04's scripted controller reads. Nothing here
          re-implements the forward and nothing reconfigures the obs dim.
  nn      THE NULL: 1-nearest-neighbour in z-scored (train-stats) obs space,
          answering with the neighbour's stored action.
  ridge   the reference arm simple enough that its failure indicts the task
          (T1.02 lesson): multi-output ridge, l2 chosen on a deterministic
          train-internal split, test labels never consulted.
  mean    predict the train-mean action — the registry's other null and the
          scale anchor.

The nulls are given the COMPACT 34-dim observation, not the padded 348-dim
one. The 314 constant-zero dimensions carry no information, and diluting the
nulls with them would hand the claim an advantage the trunk did not earn.

CONTROL (must fail): the same shipped path trained identically on a SHUFFLED
(obs, action) pairing must NOT beat the nearest-neighbour null. If
information-free supervision beats real retrieval, the metric is not measuring
imitation. A control that clears its gate is VOID, not FAIL (bakeoff lesson).

PRE-REGISTERED GATES — frozen before the registered run; every one relative or
exogenous, no constant fitted to this rig's pilot (T2.08 anti-lottery lesson):
  CLAIM 1  mse_bc <= CLONE_RATIO * mse_nn on EVERY seed. CLONE_RATIO = 0.8 is
           INHERITED from T2.04 unchanged, not chosen here.
  CLAIM 2  mse_bc <  mse_mean on EVERY seed (the registry names two nulls and
           this is the second; it is implied by CLAIM 1 whenever the rig gate
           below holds, and is gated anyway because the registry says "AND").
  CONTROL  mse_shuffled >= mse_nn on every seed.
  RIG -> VOID, not FAIL (an invalid run is not evidence about the hypothesis):
    - corpus sha256 matches CORPUS_SHA256, with the declared clip/frame counts;
    - obs/action dims match the config contract (348/17 — the T0.14 scar);
    - train and test share ZERO clips;
    - every recorded loss and metric finite;
    - eval determinism ASSERTED, not hoped (the .eval() scar): the full per-row
      eval runs twice with the running-normalisation state snapshot-restored
      between passes, and the two prediction arrays must be bit-identical;
    - mse_nn < mse_mean — a null no better than the mean means the corpus is
      degenerate and the comparison would be against noise;
    - THE TASK-TRIVIALITY FLOOR (new here, see below): mse_ridge >=
      TRIVIAL_FLOOR * mse_nn. A task a LINEAR map annihilates is not an
      imitation task, whatever a 57M-parameter trunk then scores on it;
    - if the claim fails AND ridge also fails to beat nn, the TASK is indicted,
      not the head (T1.02 lesson) -> VOID.

THE TASK-TRIVIALITY FLOOR, AND WHY IT CARRIES A KNOWN-ANSWER TEST. VO.02 was
written the day before this spec and found a permutation "floor" that was an
INVARIANCE of its own statistic — it could not move, so the gate on it was
unsatisfiable by arithmetic, and that was T3.10's defect one day later on a
different surface. The generalisation in LESSONS.md is about NULLS. This spec
adds the same discipline one level up, on the TASK: the `nextpose` row above is
a real, measured, planted violation of exactly the degeneracy this floor exists
to catch, so the floor is not merely asserted to work — `_task_floor_selftest`
rebuilds the rejected target from the same corpus inside the run and reports
`floor_selftest_fires` (nextpose must be BELOW the floor) alongside
`floor_selftest_passes` (pd must be ABOVE it). Both are rig gates. A floor that
cannot demonstrate a case it rejects is a floor nothing has ever failed.

GPU. One submission for the whole spec (module cache — `run_spec` calls
`_experiment` once per seed; the 5.5-GPU-hour scar). Science code lives HERE and
the JOB string only imports it (T0.16: code in strings is invisible to every
guard).

Declares no `COVERS:` commitment. "imitation" is not a name in
`coverage.COMMITMENTS` and the registry carries no marker for T2.14; claiming
one here would read as a commitment the ladder cannot check (26th audit B2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped action path AND the shipped action definition;
# all three files hash into the certificate.
IMPL_DEPS = ["TrainingPipeline.py", "UnifiedBrain.py", "MoCapLoader.py"]

SEEDS = [0, 1, 2]

# ── the corpus: pinned by content, not by path ───────────────────────────
CORPUS = Path(__file__).resolve().parents[1] / "data" / "t214_cmu_corpus.npz"
CORPUS_SHA256 = "bc181c88a2ae5468d40600f902db121e17302004d7c8787601b91db64c4ac627"
CORPUS_CLIPS = 391
CORPUS_FRAMES = 63905

OBS_DIM, ACT_DIM = 348, 17       # config contract (Humanoid-v5)
JANG_SLICE = slice(5, 22)        # joint angles, the layout T2.04 reads
JVEL_SLICE = slice(28, 45)       # joint velocities
MOCAP_DIM = 2 * ACT_DIM          # the compact 34-dim observation

TRAIN_CLIP_FRAC = 0.75
N_TRAIN, N_TEST = 20000, 2000    # identical for every arm (see docstring)

BC_STEPS = 1200
BC_BATCH = 256
BC_LR = 3e-4
L2_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
VAL_EVERY = 5

# Pre-registered gates. FINAL.
CLONE_RATIO = 0.8                # inherited from T2.04, unchanged
TRIVIAL_FLOOR = 0.05             # ridge may not beat the null by more than 20x


def _load_corpus() -> tuple:
    """(positions, velocities, clip_ids, sha_ok, counts_ok). Content-checked."""
    raw = CORPUS.read_bytes()
    sha_ok = hashlib.sha256(raw).hexdigest() == CORPUS_SHA256
    d = np.load(CORPUS, allow_pickle=False)
    pos = d["positions"].astype(np.float64)
    vel = d["velocities"].astype(np.float64)
    clip = d["clip"].astype(np.int64)
    counts_ok = (len(pos) == CORPUS_FRAMES
                 and int(clip.max()) + 1 == CORPUS_CLIPS
                 and pos.shape[1] == ACT_DIM)
    return pos, vel, clip, sha_ok, counts_ok


def _labels(pos_prev, pos_next, vel_prev) -> np.ndarray:
    """The SHIPPED action definition — imported, never re-derived (T0.07)."""
    from MoCapLoader import ActionComputer, MoCapConfig
    return ActionComputer(MoCapConfig()).compute_actions(
        target_positions=pos_next,
        current_positions=pos_prev,
        current_velocities=vel_prev,
    )


def _build_pairs(pos, vel, clip, target: str = "pd") -> tuple:
    """(obs34, label, clip_id) over every within-clip consecutive frame pair.

    `target` exists ONLY so the triviality floor can rebuild the rejected
    `nextpose` target as its planted violation; the claim always uses "pd".
    """
    same = clip[1:] == clip[:-1]
    p_prev, p_next = pos[:-1][same], pos[1:][same]
    v_prev = vel[:-1][same]
    g = clip[:-1][same]
    X = np.concatenate([p_prev, v_prev], axis=1)
    if target == "pd":
        Y = _labels(p_prev, p_next, v_prev)
    elif target == "nextpose":
        Y = p_next
    else:
        raise ValueError(target)
    return X, np.asarray(Y, dtype=np.float64), g


def _split(X, Y, g, seed: int) -> tuple:
    """Clip-disjoint train/test, subsampled to the declared per-arm budgets."""
    ids = np.unique(g)
    order = np.random.RandomState((int(seed) * 7919 + 13) % 2**32).permutation(len(ids))
    n_tr = int(TRAIN_CLIP_FRAC * len(ids))
    tr_ids, te_ids = set(ids[order[:n_tr]].tolist()), set(ids[order[n_tr:]].tolist())
    m_tr = np.isin(g, list(tr_ids))
    m_te = np.isin(g, list(te_ids))
    rs = np.random.RandomState((int(seed) * 104729 + 7) % 2**32)

    def take(mask, n):
        idx = np.flatnonzero(mask)
        if len(idx) > n:
            idx = rs.choice(idx, n, replace=False)
        return idx

    i_tr, i_te = take(m_tr, N_TRAIN), take(m_te, N_TEST)
    overlap = len(set(g[i_tr].tolist()) & set(g[i_te].tolist()))
    return (X[i_tr], Y[i_tr], X[i_te], Y[i_te], int(overlap),
            int(len(tr_ids)), int(len(te_ids)))


# ── the learner: the shipped path, trained end-to-end ────────────────────
def _to_env_obs(X34: np.ndarray) -> np.ndarray:
    """Place the mocap state into the declared Humanoid-v5 observation layout.
    Not a reconfiguration of the obs dim — the config contract is the thing the
    rig gate checks, and reshaping it to fit the data is how T0.14 happened."""
    out = np.zeros((len(X34), OBS_DIM), dtype=np.float64)
    out[:, JANG_SLICE] = X34[:, :ACT_DIM]
    out[:, JVEL_SLICE] = X34[:, ACT_DIM:]
    return out


def _train_bc(tp, Xtr_norm: np.ndarray, Ytr: np.ndarray, seed: int) -> float:
    """Adam/MSE on the shipped forward (obs_proj + trunk + policy_mean)."""
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
    API (batching it would re-implement the forward: the T0.16 anti-pattern)."""
    return np.stack([tp.act_deterministic(Xte[i]) for i in range(len(Xte))])


def _eval_deterministic(tp, Xte: np.ndarray) -> tuple:
    """(predictions, det_ok). Two full passes with the running-normalisation
    state snapshot-restored between them; bit-identity is the dropout-scar
    assertion (normalize_obs mutates running stats on every call)."""
    snap = (tp.obs_mean.clone(), tp.obs_var.clone(), tp.obs_count)
    pred1 = _eval_rows(tp, Xte)
    tp.obs_mean, tp.obs_var, tp.obs_count = snap[0].clone(), snap[1].clone(), snap[2]
    pred2 = _eval_rows(tp, Xte)
    tp.obs_mean, tp.obs_var, tp.obs_count = snap
    return pred1, bool(np.array_equal(pred1, pred2))


# ── the null and the reference arms ──────────────────────────────────────
def _zscore_stats(Xtr: np.ndarray) -> tuple:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def _nn_mse(Xtr, Ytr, Xte, Yte) -> float:
    """1-NN in z-scored obs space, chunked to stay flat in memory."""
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
    """(mse, chosen l2). l2 picked on the deterministic every-5th-row split of
    TRAIN; test labels are never consulted."""
    mu, sd = _zscore_stats(Xtr)
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    val = np.arange(len(A)) % VAL_EVERY == 0
    fit = ~val

    def solve(X, Y, l2):
        return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ Y)

    ym = Ytr[fit].mean(0)
    best, best_l2 = np.inf, None
    for l2 in L2_GRID:
        W = solve(A[fit], Ytr[fit] - ym, l2)
        err = float(((A[val] @ W + ym - Ytr[val]) ** 2).mean())
        if err < best:
            best, best_l2 = err, l2
    ym = Ytr.mean(0)
    W = solve(A, Ytr - ym, best_l2)
    return float(((B @ W + ym - Yte) ** 2).mean()), best_l2


def _task_floor_selftest(pos, vel, clip, seed: int) -> dict:
    """Point the triviality floor at a target it MUST reject and one it must
    accept. `nextpose` is the measured, planted violation (docstring probe);
    `pd` is the claim's own task. A floor that never fires is not a floor."""
    out = {}
    for target in ("pd", "nextpose"):
        X, Y, g = _build_pairs(pos, vel, clip, target=target)
        Xtr, Ytr, Xte, Yte, _ov, _a, _b = _split(X, Y, g, seed)
        nn = _nn_mse(Xtr, Ytr, Xte, Yte)
        rg, _ = _ridge_mse(Xtr, Ytr, Xte, Yte)
        out[target] = round(rg / nn, 6) if nn > 0 else float("inf")
    return {
        "floor_ratio_pd": out["pd"],
        "floor_ratio_nextpose": out["nextpose"],
        "floor_selftest_fires": bool(out["nextpose"] < TRIVIAL_FLOOR),
        "floor_selftest_passes": bool(out["pd"] >= TRIVIAL_FLOOR),
    }


# ── remote entry point (also the local smoke, scaled down) ───────────────
def remote_run(seeds: list, n_train: int = N_TRAIN, n_test: int = N_TEST,
               bc_steps: int = BC_STEPS, pipe_kwargs: dict | None = None,
               selftest: bool = True) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally, tiny,
    for the smoke). Returns the JSON-able result dict."""
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    global N_TRAIN, N_TEST
    n_train_o, n_test_o = N_TRAIN, N_TEST
    N_TRAIN, N_TEST = n_train, n_test
    try:
        pos, vel, clip, sha_ok, counts_ok = _load_corpus()
        X, Y, g = _build_pairs(pos, vel, clip, target="pd")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "corpus_sha_ok": bool(sha_ok), "corpus_counts_ok": bool(counts_ok),
               "n_pairs": int(len(X)), "seeds": []}

        for seed in seeds:
            Xtr, Ytr, Xte, Yte, overlap, n_ctr, n_cte = _split(X, Y, g, seed)

            # bc — the shipped path. Weight init seeded per spec seed.
            torch.manual_seed(int(seed))
            tp = TrainingPipeline(PipelineConfig(**(pipe_kwargs or {})))
            dims_ok = (tp.config.mujoco_obs_dim == OBS_DIM
                       and tp.config.action_dim == ACT_DIM)
            Etr, Ete = _to_env_obs(Xtr), _to_env_obs(Xte)
            Etr_norm = tp.normalize_obs(Etr)      # one batch merge: train stats
            loss_bc = _train_bc(tp, Etr_norm, Ytr, seed)
            pred_bc, det_ok = _eval_deterministic(tp, Ete)
            mse_bc = float(((pred_bc - Yte) ** 2).mean())

            # the null and the references — on the COMPACT obs (docstring)
            mse_nn = _nn_mse(Xtr, Ytr, Xte, Yte)
            mse_ridge, l2_ridge = _ridge_mse(Xtr, Ytr, Xte, Yte)
            mse_mean = float(((Ytr.mean(0) - Yte) ** 2).mean())

            # control — identical training on a shuffled pairing, fresh weights.
            torch.manual_seed(int(seed) + 7)
            tp_sh = TrainingPipeline(PipelineConfig(**(pipe_kwargs or {})))
            Etr_norm_sh = tp_sh.normalize_obs(Etr)
            perm = np.random.RandomState((int(seed) + 41) % 2**32).permutation(len(Ytr))
            loss_sh = _train_bc(tp_sh, Etr_norm_sh, Ytr[perm], seed)
            pred_sh, det_ok_sh = _eval_deterministic(tp_sh, Ete)
            mse_shuffled = float(((pred_sh - Yte) ** 2).mean())

            row = {
                "seed": int(seed), "dims_ok": bool(dims_ok),
                "clip_overlap": overlap,
                "clips_train": n_ctr, "clips_test": n_cte,
                "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "mse_bc": round(mse_bc, 8), "mse_nn": round(mse_nn, 8),
                "mse_ridge": round(mse_ridge, 8), "mse_mean": round(mse_mean, 8),
                "mse_shuffled": round(mse_shuffled, 8),
                "l2_ridge": l2_ridge,
                "loss_bc_final": round(loss_bc, 8),
                "loss_shuffled_final": round(loss_sh, 8),
                "det_ok": bool(det_ok and det_ok_sh),
                "finite": bool(np.isfinite([mse_bc, mse_nn, mse_ridge, mse_mean,
                                            mse_shuffled, loss_bc]).all()),
            }
            if selftest:
                row.update(_task_floor_selftest(pos, vel, clip, seed))
            out["seeds"].append(row)
        return out
    finally:
        N_TRAIN, N_TEST = n_train_o, n_test_o


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import json, os as _o
from experiments.tests.t2_14_imitation_mocap import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t214.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Sizing, from T2.04's PROBE AT THE PRODUCTION CONFIG on the target P100
    # (kernel jack-ladder-1786691401, PipelineConfig() defaults d512 x 8):
    # 0.4225 s/train-step, 0.0157 s/eval-row. This spec needs no environment
    # and no rollouts — the corpus arrives with the clone — so collection cost
    # is zero and the only additions are numpy nulls and the floor self-test.
    # Projection: 7200 train steps (1200 x 2 arms x 3 seeds) = 3042 s
    # + 12000 eval rows (2000 x 2 arms x 3 seeds) = 188 s
    # + nulls & self-test (2 targets x 3 seeds, 20000x2000 1-NN) ~= 120 s
    # + ~120 s setup = ~3470 s ~= 0.96 h -> est_hours 1.0. timeout_s 7200 is
    # ~2x the projection (queue + variance headroom, inside the runner's
    # 21600 s child timeout); billing charges what the kernel's log reports,
    # so a generous cap costs nothing when the run is short.
    res = submit(job, prefer="kaggle", est_hours=1.0, timeout_s=7200,
                 fetch=["t214.json"])
    if not res.ok:
        raise RuntimeError(f"T2.14 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t214.json"]).read_text())
    out["backend"] = res.backend
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    ratios = [r["mse_bc"] / r["mse_nn"] for r in rows]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "corpus_ok": bool(_CACHE["corpus_sha_ok"] and _CACHE["corpus_counts_ok"]),
        "n_pairs": _CACHE["n_pairs"],
        "action_mse": [r["mse_bc"] for r in rows],
        "mse_nn": [r["mse_nn"] for r in rows],
        "mse_ridge": [r["mse_ridge"] for r in rows],
        "mse_mean": [r["mse_mean"] for r in rows],
        "heldout_vs_nn_ratio": round(max(ratios), 4),
        "clone_ratio_all": [round(x, 4) for x in ratios],
        "all_seeds_beat_null": all(r["mse_bc"] <= CLONE_RATIO * r["mse_nn"]
                                   for r in rows),
        "all_seeds_beat_mean": all(r["mse_bc"] < r["mse_mean"] for r in rows),
        "ridge_beats_null_any": any(r["mse_ridge"] < r["mse_nn"] for r in rows),
        "nn_informative_all": all(r["mse_nn"] < r["mse_mean"] for r in rows),
        "task_not_trivial_all": all(r["mse_ridge"] >= TRIVIAL_FLOOR * r["mse_nn"]
                                    for r in rows),
        "floor_selftest_ok": all(r.get("floor_selftest_fires")
                                 and r.get("floor_selftest_passes")
                                 for r in rows),
        "floor_ratio_nextpose": [r.get("floor_ratio_nextpose") for r in rows],
        "floor_ratio_pd": [r.get("floor_ratio_pd") for r in rows],
        "clip_overlap_max": max(r["clip_overlap"] for r in rows),
        "det_ok_all": all(r["det_ok"] for r in rows),
        "dims_ok_all": all(r["dims_ok"] for r in rows),
        "finite_all": all(r["finite"] for r in rows),
        "bc_beats_ridge_all": all(r["mse_bc"] <= r["mse_ridge"] for r in rows),
        "loss_bc_final_max": max(r["loss_bc_final"] for r in rows),
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
    if not m["corpus_ok"]:
        return Status.VOID          # not the corpus this spec froze (T1.13 scar)
    if not m["dims_ok_all"]:
        return Status.VOID          # config contract broken (T0.14 scar)
    if m["clip_overlap_max"] > 0:
        return Status.VOID          # train/test share a clip: the null leaks
    if not m["finite_all"]:
        return Status.VOID          # training diverged; nothing was measured
    if not m["det_ok_all"]:
        return Status.VOID          # eval path not deterministic (dropout scar)
    if not m["nn_informative_all"]:
        return Status.VOID          # null no better than the mean: degenerate
    if not m["floor_selftest_ok"]:
        return Status.VOID          # the triviality floor cannot demonstrate
                                    # a case it rejects — it is not a floor
    if not m["task_not_trivial_all"]:
        return Status.VOID          # a linear map annihilates the task; a good
                                    # score here would be arithmetic, not imitation
    # Control: information-free supervision must NOT beat real retrieval.
    if c["shuffled_beats_null"]:
        return Status.VOID          # the metric leaks; not measuring imitation
    claim = bool(m["all_seeds_beat_null"] and m["all_seeds_beat_mean"])
    if not claim and not m["ridge_beats_null_any"]:
        return Status.VOID          # the reference arm fails too: task indicted,
                                    # not the head (T1.02 lesson)
    return claim


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.14"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, tiny, CPU: exercises the REAL code paths (corpus load and sha
        # check, shipped label computation, clip-disjoint split, both training
        # loops, snapshot/restore bit-identity, every null, and the floor
        # self-test) on the smallest PipelineConfig that still runs them.
        #
        # SIZED FROM A MEASUREMENT, and the measurement is the point (builder,
        # 2026-08-30). `tp.model(...)` is the FULL UnifiedBrain forward — world
        # model, 512-sample MPC, planner, the lot — so one training step at
        # d64 x 2 costs ~50 s on this box's 4 ARM cores. An earlier draft of
        # this smoke used bc_steps=30 / n_test=200 (T2.04's shape) and was
        # still in its first training loop after 20 minutes, holding 1.8 GB,
        # looking exactly like a hang. It was neither hung nor OOM: it was
        # ~2 orders of magnitude too big for the hardware. Keep these numbers
        # tiny — this smoke exists to prove the paths EXECUTE, never to
        # measure anything. Every number the claim rests on comes off the GPU.
        out = remote_run([2], n_train=400, n_test=4, bc_steps=1,
                         pipe_kwargs={"d_model": 32, "n_layers": 1})
        print(json.dumps(out, indent=1))
        row = out["seeds"][0]
        assert out["corpus_sha_ok"] and out["corpus_counts_ok"], "corpus"
        assert row["det_ok"] and row["dims_ok"] and row["finite"], row
        assert row["clip_overlap"] == 0, "clip leak"
        assert row["floor_selftest_fires"], "triviality floor never fires"
        assert row["floor_selftest_passes"], "triviality floor rejects its own task"
        print("SMOKE OK")
    else:
        run()
