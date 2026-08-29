"""T2.19 — Flow head handles multimodal actions.

HYPOTHESIS (registry). On a bimodal task (pass obstacle left OR right) the flow
head succeeds where MSE regression collapses to the mean. Falsified by: L1/MSE
regression matches the flow head — "OFT found this on some benchmarks; genuine
falsification risk, and if it happens the flow head loses its justification".
Null: deterministic regression head, same params. Metric:
bimodal_success_ratio. Control: on a unimodal task the two heads must tie.

WHY THIS SPEC IS THE ONE THAT JUSTIFIES THE FLOW PATH. T1.11 proved gradient
reaches ActionExpert; T1.12 proved the SAMPLER the robot runs reconstructs a
fixed target. Neither asks the only question that makes flow matching worth its
4.6M parameters: a deterministic head can reconstruct a fixed target too. Flow
matching earns its place if and only if the action distribution is genuinely
MULTIMODAL — where two different actions are both correct and their average is
wrong. If regression ties here, the honest move is a direct head, and T1.12's
`kills` line already names it.

WHAT IS MEASURED, SAID PLAINLY. Both arms are the SHIPPED UnifiedBrain and the
SHIPPED ActionExpert (T1.11 runtime-parity path). They differ in exactly two
places, the objective and the sampler:

  FLOW  train `action_training_loss` (the one loss the robot uses: conditional
        flow matching on ActionExpert + 0.1 * action_head aux)
        sample `generate_actions_flow_matching` (@no_grad, 10 Euler steps)
  REG   train MSE( action_expert(x=0, vlm_features, t=1), target )
        sample the same deterministic forward

The regression arm is therefore not a smaller stand-in — it is the same module
with a constant zero query instead of a noise query, so "same params" in the
registry's null is exact, not approximate: `n_params_flow == n_params_reg` is
gated as an equality and a mismatch is VOID. Under `flow_parameterisation="x1"`
ActionExpert already predicts the clean action rather than a velocity, so the
regression arm is its natural deterministic reading, in-distribution for the
convention it is trained under.

The flow arm carries the aux MSE term because that is the production loss and
this spec refuses to measure a path the robot does not run. Note which way that
cuts: the aux term pulls the backbone toward the conditional MEAN, which is the
regression arm's failure mode. The flow arm is being handicapped by its own
shipped objective, so the comparison is conservative in the direction that
matters.

THE FIXTURE — an obstacle, and the average of the two ways round it is a
collision. `S` scenarios; each scenario `s` carries a fixed random observation
`o_s` in the 256-d state and a fixed random "content" pattern over action dims
1..16 that BOTH modes share. Action dim 0 is lateral: the target carries
`sigma * AMP * p(t)` with the bump `p(t) = sin(pi*(t+0.5)/chunk)` peaking
mid-chunk — swerve out and come back. The two modes are `sigma = +1` (left) and
`sigma = -1` (right); their mean is `sigma = 0`, which is the straight line
through the obstacle. That is the whole point and it is a property of the
construction, not of any measurement: nothing a mean-seeking objective can
output is a valid action.

  BIMODAL   sigma drawn fresh +/-1 at every draw -> the conditional p(a|o_s)
            has two equal modes. Fresh batches every step; no memorisation.
  UNIMODAL  sigma FIXED per scenario (half the bank +, half -) -> the same
            marginal variance on dim 0, the same content, the same scenario
            count, the same steps and batch. The ONLY difference between the
            two tasks is whether the conditional has one mode or two, which is
            what makes the unimodal leg a control rather than a second task.

SUCCESS IS A CONJUNCTION, AND THE SECOND HALF IS NOT DECORATION. For a
generated chunk `a`, the lateral projection is
`d = sum_t a[t,0] p(t) / sum_t p(t)^2`, so a perfect mode reads `d = +/-AMP`
and mean-collapse reads `d = 0`. Then

  committed   |d| >= COMMIT_FRAC * AMP        (it chose a side)
  conditioned mean sq err on dims 1..16 <= SHARED_TOL   (it used the percept)
  success     BOTH

The conjunction exists because the commitment half ALONE is won vacuously by an
arm that ignores its conditioning and emits a large lateral in a random
direction — the T2.09 scouting note's warning ("an arm that ignores percepts
wins the trap test vacuously") applies verbatim to this rig, and an untrained
flow sampler starting from pure noise is exactly such an arm. SHARED_TOL is
exogenous, fixed by the construction rather than by any pilot: the content
pattern is unit-variance, so an arm that ignores the percept and emits the
global mean scores ~1.0 and the bar of 0.25 demands 75% of the content variance
explained. COMMIT_FRAC = 0.5 is geometric — at least halfway from the collision
to a mode.

WHAT IS *NOT* MEASURED. This is not a generalisation test — evaluation is on
the training scenario bank, because the claim is that the head can REPRESENT a
two-mode conditional at all, not that it transfers to unseen phrasings. That is
T2.07, and it is settled FAIL; conflating the two would let a representational
failure hide behind a transfer failure.

CONTROL (must tie) AND THE ALIVE-PROOFS. The registry's declared control is the
unimodal leg, and it does double duty as the regression arm's alive-proof —
without it, "regression fails at bimodal" is indistinguishable from "regression
is a dead arm", which is the 24th audit's at-chance-control lesson pointed at
the null instead of at the claim. Four rig gates, all VOID rather than FAIL:

  1. UNIMODAL TIE. Both arms >= UNI_MIN and |flow_uni - reg_uni| <= TIE_BAND.
     A regression arm that cannot do the single-mode version has not been
     beaten by multimodality; it has not been trained.
  2. REGRESSION IS ALIVE ON THE BIMODAL LEG TOO. Its CONDITIONING half must
     pass at >= SHARED_PASS_MIN there. This is the gate that makes the finding
     specific: the null must be shown to have learned the percept and failed
     only on the commitment half. Its mean |d| is reported beside it, and
     mean-collapse predicts ~0.
  3. UNTRAINED FLOOR. At initialisation both arms on both tasks must score
     <= UNTRAINED_MAX. Success must not be free; this is the proof the metric
     can read a failure at all.
  4. SHUFFLED CONDITIONING. The trained flow arm re-evaluated with the scenario
     observations permuted against their content must get WORSE on the content
     error by a factor >= SHUF_MULT. This fires on positive evidence (a rise),
     per T3.01's fate-(ii) fork, rather than asserting something sits at
     chance.

Plus: finite everywhere, matched params exactly, and both arms' training loss
falling on both legs (a non-learner cannot arbitrate head design — T2.02's
principle).

THE CLAIM, gated on the WORST seed: `bimodal_success_ratio >= RATIO_MIN` AND
`flow_success_bimodal >= FLOW_MIN`. The second conjunct is load-bearing: a ratio
is won trivially when the denominator is near zero, so a run where BOTH arms
collapse must FAIL, not PASS on a large quotient.

GPU. One submission for the whole spec (module cache — run_spec calls
_experiment once per seed; the T2.01/T2.03/T4.02 pattern). No mujoco, no
downloads: torch + numpy + the cloned repo. The science lives in THIS module
and the JOB string only imports it (T0.16 lesson).

GATES ARE PROVISIONAL. `_GATES_FROZEN` is False and `run()` refuses. Five bars
depend on what this rig can reach in its budget and MUST be frozen from the
pilot artifact before any registered run: UNI_MIN, TIE_BAND, UNTRAINED_MAX,
SHARED_PASS_MIN, SHUF_MULT, and the claim pair RATIO_MIN / FLOW_MIN. The
exogenous fixture constants above (AMP, COMMIT_FRAC, SHARED_TOL) are NOT
pilot-derived and do not move. SM.03 is the cautionary case this idiom exists
for: implemented, tracked, gates never frozen, and worth zero for five days.

COVERS: one brain / unison (claim)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the shipped ActionExpert and the shipped sampler; the
# file they live in hashes into the certificate.
IMPL_DEPS = ["UnifiedBrain.py"]

SEEDS = [0, 1, 2]
SMOKE_SEED = 90

# ── fixture (exogenous — construction, never a pilot) ────────────────────
S_SCEN = 16                # scenarios in the bank
AMP = 1.0                  # lateral amplitude of a mode; mean of modes is 0
COMMIT_FRAC = 0.5          # committed = at least halfway from collision to mode
SHARED_TOL = 0.25          # content mse bar; ignore-the-percept scores ~1.0
TARGET_NOISE = 0.05        # keeps the task non-degenerate; floor on content err
LATERAL_DIM = 0            # action dim carrying the left/right choice

# ── training (matched across both arms and both legs) ────────────────────
BATCH = 16
STEPS = 300
LR = 3e-4
N_DRAWS = 8                # samples per scenario at eval (flow is stochastic)
REG_QUERY_T = 1.0          # x1 parameterisation: t=1 is the clean-action end

# ── gates ────────────────────────────────────────────────────────────────
# PROVISIONAL — every one of these must be re-written from the pilot artifact
# before _GATES_FROZEN may flip True. They are placeholders, not measurements.
_GATES_FROZEN = False

UNI_MIN = None             # control: each arm's success on the unimodal leg
TIE_BAND = None            # control: |flow_uni - reg_uni| allowed to still tie
UNTRAINED_MAX = None       # rig: success at initialisation must be at most this
SHARED_PASS_MIN = None     # rig: regression's conditioning half on the bimodal leg
SHUF_MULT = None           # rig: content-error rise under shuffled conditioning
RATIO_MIN = None           # claim: bimodal_success_ratio, worst seed
FLOW_MIN = None            # claim: flow's own bimodal success, worst seed

RATIO_FLOOR = 1.0 / (S_SCEN * N_DRAWS)   # one success; never divide by zero


# ── the world ────────────────────────────────────────────────────────────
def _bank(seed: int, cfg, n_scen: int):
    """The scenario bank: observations, shared content, and the unimodal signs.

    Everything a scenario is, is fixed here from one RandomState so both arms
    and both legs see byte-identical worlds — the arms differ in objective and
    sampler only.
    """
    rs = np.random.RandomState((seed * 9973 + 17) % (2 ** 32))
    chunk, adim = cfg.action_chunk_size, cfg.action_dim
    obs = rs.randn(n_scen, cfg.obs_dim).astype("float32")
    content = rs.randn(n_scen, chunk, adim - 1).astype("float32")
    # Half the bank swerves left, half right: the unimodal leg keeps the
    # bimodal leg's marginal variance on the lateral dim, so the legs differ
    # only in whether the CONDITIONAL has two modes.
    signs = np.array([1.0] * (n_scen // 2) + [-1.0] * (n_scen - n_scen // 2),
                     dtype="float32")
    rs.shuffle(signs)
    t = np.arange(chunk, dtype="float32")
    bump = np.sin(np.pi * (t + 0.5) / chunk)          # swerve out and back
    return {"obs": obs, "content": content, "signs": signs, "bump": bump,
            "rs": rs, "chunk": chunk, "adim": adim}


def _targets(bk, idx, sigma):
    """Assemble [B, chunk, adim] targets for scenarios `idx` with signs `sigma`."""
    chunk, adim = bk["chunk"], bk["adim"]
    out = np.empty((len(idx), chunk, adim), dtype="float32")
    lat = sigma[:, None] * AMP * bk["bump"][None, :]           # [B, chunk]
    out[:, :, LATERAL_DIM] = lat
    rest = [d for d in range(adim) if d != LATERAL_DIM]
    out[:, :, rest] = bk["content"][idx]
    out += bk["rs"].randn(*out.shape).astype("float32") * TARGET_NOISE
    return out


def _draw(bk, batch, bimodal: bool):
    """One training batch. Bimodal: sign is a fresh coin at every draw."""
    idx = bk["rs"].randint(0, len(bk["obs"]), size=batch)
    if bimodal:
        sigma = bk["rs"].choice([-1.0, 1.0], size=batch).astype("float32")
    else:
        sigma = bk["signs"][idx]
    return idx, _targets(bk, idx, sigma)


# ── the two arms ─────────────────────────────────────────────────────────
def _build(seed: int, device):
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    brain = UnifiedBrain(cfg).to(device)
    return brain, cfg


def _reg_forward(brain, cfg, state):
    """The deterministic regression arm: same ActionExpert, constant zero query.

    `generate_actions_flow_matching` feeds the expert a NOISE query and
    integrates; this feeds a ZERO query at t=1 and reads the answer directly.
    Same modules, same parameters, no sampling.
    """
    import torch
    B = state.shape[0]
    vlm = brain.forward(state)["hidden_states"]
    x = torch.zeros(B, cfg.action_dim * cfg.action_chunk_size,
                    device=state.device)
    t = torch.full((B,), REG_QUERY_T, device=state.device)
    pred = brain.action_expert(x, vlm, t)
    return pred.view(B, cfg.action_chunk_size, cfg.action_dim)


def _predict(brain, cfg, state, arm: str):
    import torch
    brain.eval()
    with torch.no_grad():
        if arm == "flow":
            out = brain.generate_actions_flow_matching(state)
        else:
            out = _reg_forward(brain, cfg, state)
    brain.train()
    return out


# ── scoring ──────────────────────────────────────────────────────────────
def _score(pred, bk, idx):
    """Commitment, conditioning, and their conjunction, per sample."""
    a = pred.detach().float().cpu().numpy()
    bump = bk["bump"]
    d = (a[:, :, LATERAL_DIM] * bump[None, :]).sum(1) / (bump ** 2).sum()
    rest = [j for j in range(bk["adim"]) if j != LATERAL_DIM]
    err = ((a[:, :, rest] - bk["content"][idx]) ** 2).mean(axis=(1, 2))
    committed = np.abs(d) >= COMMIT_FRAC * AMP
    conditioned = err <= SHARED_TOL
    return {"d": d, "content_err": err,
            "committed": committed, "conditioned": conditioned,
            "success": committed & conditioned}


def _evaluate(brain, cfg, bk, arm, device, shuffle_obs=False):
    """Every scenario, N_DRAWS times. Flow is stochastic; reg is not, but both
    are drawn the same number of times so the two rates are commensurable."""
    import torch
    n = len(bk["obs"])
    idx = np.tile(np.arange(n), N_DRAWS)
    obs_idx = np.roll(idx, 1) if shuffle_obs else idx     # percept vs content
    rows = []
    for lo in range(0, len(idx), BATCH):
        sl = slice(lo, lo + BATCH)
        state = torch.from_numpy(bk["obs"][obs_idx[sl]]).to(device)
        pred = _predict(brain, cfg, state, arm)
        rows.append(_score(pred, bk, idx[sl]))
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return {
        "success_rate": round(float(cat["success"].mean()), 4),
        "committed_rate": round(float(cat["committed"].mean()), 4),
        "conditioned_rate": round(float(cat["conditioned"].mean()), 4),
        "abs_lateral_mean": round(float(np.abs(cat["d"]).mean()), 4),
        "content_err_mean": round(float(cat["content_err"].mean()), 4),
        "finite": bool(np.isfinite(cat["d"]).all()
                       and np.isfinite(cat["content_err"]).all()),
    }


# ── one arm, one leg ─────────────────────────────────────────────────────
def _train_measure(seed, arm, bimodal, steps, batch, device, n_scen=S_SCEN,
                   curve_every=0):
    import torch
    brain, cfg = _build(seed, device)
    bk = _bank(seed, cfg, n_scen)
    n_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)

    before = _evaluate(brain, cfg, bk, arm, device)      # the untrained floor

    opt = torch.optim.Adam([p for p in brain.parameters() if p.requires_grad],
                           lr=LR)
    losses, curve = [], []
    for step in range(steps):
        idx, tgt = _draw(bk, batch, bimodal)
        state = torch.from_numpy(bk["obs"][idx]).to(device)
        target = torch.from_numpy(tgt).to(device)
        if arm == "flow":
            loss = brain.action_training_loss(state, target)["loss"]
        else:
            loss = torch.nn.functional.mse_loss(
                _reg_forward(brain, cfg, state).float(), target.float())
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        # Pilot only: the curve is how the NEXT iteration sizes STEPS. A single
        # end-point cannot distinguish "this rig cannot do it" from "this rig
        # was not given enough steps", and that distinction is the whole
        # difference between a FAIL and a badly-budgeted VOID.
        if curve_every and (step + 1) % curve_every == 0:
            e = _evaluate(brain, cfg, bk, arm, device)
            curve.append({"step": step + 1, "success": e["success_rate"],
                          "committed": e["committed_rate"],
                          "conditioned": e["conditioned_rate"],
                          "abs_lateral": e["abs_lateral_mean"],
                          "content_err": e["content_err_mean"]})
            print("CURVE", seed, arm, bool(bimodal), json.dumps(curve[-1]),
                  flush=True)

    after = _evaluate(brain, cfg, bk, arm, device)
    shuf = (_evaluate(brain, cfg, bk, arm, device, shuffle_obs=True)
            if arm == "flow" and bimodal else None)

    q = max(1, len(losses) // 4)
    return {
        "arm": arm, "bimodal": bool(bimodal), "n_params": int(n_params),
        "untrained": before, "trained": after, "shuffled": shuf,
        "curve": curve,
        "loss_first": round(float(np.mean(losses[:q])), 5),
        "loss_last": round(float(np.mean(losses[-q:])), 5),
        "finite": bool(np.isfinite(losses).all()
                       and before["finite"] and after["finite"]),
    }


# ── remote entry point ───────────────────────────────────────────────────
def remote_run(seeds: list, steps: int = STEPS, batch: int = BATCH,
               n_scen: int = S_SCEN, curve_every: int = 0) -> dict:
    """Both arms on both legs, for every seed. Runs on the GPU VM, or locally
    at reduced steps/batch/bank for the smoke — argument SHAPES stay
    production (full config, real ActionExpert, real sampler)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "steps": steps, "batch": batch, "n_scen": n_scen, "seeds": []}
    for seed in seeds:
        row = {"seed": seed}
        for arm in ("flow", "reg"):
            for bimodal in (True, False):
                key = f"{arm}_{'bimodal' if bimodal else 'unimodal'}"
                row[key] = _train_measure(seed, arm, bimodal, steps, batch,
                                          device, n_scen, curve_every)
                print("LEG_DONE", seed, key,
                      row[key]["trained"]["success_rate"], flush=True)
        out["seeds"].append(row)
        print("SEED_DONE", json.dumps(row), flush=True)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import json, os as _o
from experiments.tests.t2_19_flow_multimodal import remote_run
out = remote_run(__SEEDS__, steps=__STEPS__, curve_every=__CURVE__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t219.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list, steps: int = STEPS, curve_every: int = 0) -> dict:
    body = (JOB.replace("__SEEDS__", repr(list(seeds)))
               .replace("__STEPS__", str(int(steps)))
               .replace("__CURVE__", str(int(curve_every))))
    job = build_job(body)
    # One submission runs 2 arms x 2 legs x len(seeds) trainings serially —
    # multiply by seeds AND arms before sizing any budget or timeout. The
    # curve adds one full evaluation per checkpoint per leg, so it is priced
    # in rather than assumed free.
    n_legs = 4 * len(seeds)
    evals = n_legs * (2 + (steps // curve_every if curve_every else 0))
    est_hours = round(0.05 + 0.0004 * n_legs * steps + 0.002 * evals, 2)
    # The timeout is DERIVED from the estimate, never sized independently.
    # First cut of this function had est_hours=2.08 against a hand-written
    # `1800 + 1200*len(seeds)` = 50 min, so the watcher was set to give up at
    # 40% of the run's own predicted length — a job killed by the very
    # estimate that said it needed longer. If the estimate is wrong, fix the
    # estimate; the two numbers may not disagree.
    timeout_s = int(est_hours * 3600 * 1.5) + 900
    res = submit(job, prefer="kaggle", est_hours=est_hours,
                 timeout_s=timeout_s, fetch=["t219.json"])
    if not res.ok:
        raise RuntimeError(f"T2.19 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t219.json"]).read_text())
    out["backend"] = res.backend
    return out


# ── the reading ──────────────────────────────────────────────────────────
def _fold(rows: list) -> dict:
    """Per-seed rows -> the numbers the gates read. Worst seed everywhere:
    report per partition, gate the minimum."""
    def leg(key):
        return [r[key] for r in rows]

    fb, rb = leg("flow_bimodal"), leg("reg_bimodal")
    fu, ru = leg("flow_unimodal"), leg("reg_unimodal")
    all_legs = fb + rb + fu + ru

    ratios = [round(f["trained"]["success_rate"]
                    / max(r["trained"]["success_rate"], RATIO_FLOOR), 3)
              for f, r in zip(fb, rb)]
    return {
        "bimodal_success_ratio": min(ratios),
        "ratio_per_seed": ratios,
        "flow_success_bimodal": min(x["trained"]["success_rate"] for x in fb),
        "reg_success_bimodal": max(x["trained"]["success_rate"] for x in rb),
        "flow_bimodal_per_seed": [x["trained"]["success_rate"] for x in fb],
        "reg_bimodal_per_seed": [x["trained"]["success_rate"] for x in rb],
        # the collapse signature: mean-seeking predicts |d| ~ 0
        "reg_abs_lateral_bimodal": max(x["trained"]["abs_lateral_mean"]
                                       for x in rb),
        "flow_abs_lateral_bimodal": min(x["trained"]["abs_lateral_mean"]
                                        for x in fb),
        # rig
        "params_matched": float(len({x["n_params"] for x in all_legs}) == 1),
        "n_params": all_legs[0]["n_params"],
        "finite_all": float(all(x["finite"] for x in all_legs)),
        "loss_fell_all": float(all(x["loss_last"] < x["loss_first"]
                                   for x in all_legs)),
        "untrained_max": max(x["untrained"]["success_rate"] for x in all_legs),
        "reg_shared_pass_bimodal": min(x["trained"]["conditioned_rate"]
                                       for x in rb),
        "shuf_mult": min(
            round(x["shuffled"]["content_err_mean"]
                  / max(x["trained"]["content_err_mean"], 1e-9), 3)
            for x in fb),
        # control (the registry's own): the unimodal tie
        "uni_min": min(min(x["trained"]["success_rate"] for x in fu),
                       min(x["trained"]["success_rate"] for x in ru)),
        "uni_gap": max(abs(f["trained"]["success_rate"]
                           - r["trained"]["success_rate"])
                       for f, r in zip(fu, ru)),
        "flow_unimodal_per_seed": [x["trained"]["success_rate"] for x in fu],
        "reg_unimodal_per_seed": [x["trained"]["success_rate"] for x in ru],
        "loss_first_last": [[x["arm"], x["bimodal"], x["loss_first"],
                             x["loss_last"]] for x in all_legs],
    }


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    m = _fold(_CACHE["seeds"])
    m["gpu"] = _CACHE["gpu"]
    m["backend"] = _CACHE.get("backend", "?")
    return m


def _control(seed: int) -> dict:
    """The registry's declared control: the unimodal leg, where the two heads
    must TIE. It is also the regression arm's alive-proof — see the docstring."""
    m = _fold(_CACHE["seeds"])
    return {k: m[k] for k in ("uni_min", "uni_gap", "flow_unimodal_per_seed",
                              "reg_unimodal_per_seed", "untrained_max",
                              "reg_shared_pass_bimodal", "shuf_mult")}


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["finite_all"]:
        return Status.VOID          # NaN somewhere — measured nothing
    if not m["params_matched"]:
        return Status.VOID          # "same params" is the null's own word
    if not m["loss_fell_all"]:
        return Status.VOID          # a non-learner cannot arbitrate head design
    if m["untrained_max"] > UNTRAINED_MAX:
        return Status.VOID          # success is free — the metric reads nothing
    if c["shuf_mult"] < SHUF_MULT:
        return Status.VOID          # conditioning check is not alive
    if c["reg_shared_pass_bimodal"] < SHARED_PASS_MIN:
        return Status.VOID          # the null is dead, not mean-collapsed
    # The control the registry declares: on a unimodal task the two must tie.
    if c["uni_min"] < UNI_MIN or c["uni_gap"] > TIE_BAND:
        return Status.VOID          # no tie -> the arms are not comparable
    # The claim, on the worst seed. The second conjunct is load-bearing: a
    # ratio is won trivially when its denominator is near zero.
    return (m["bimodal_success_ratio"] >= RATIO_MIN
            and m["flow_success_bimodal"] >= FLOW_MIN)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T2.19 gates are provisional — run the pilot "
            "(`python -m experiments.tests.t2_19_flow_multimodal pilot`), "
            "freeze UNI_MIN/TIE_BAND/UNTRAINED_MAX/SHARED_PASS_MIN/SHUF_MULT/"
            "RATIO_MIN/FLOW_MIN in this file from its artifact, then run "
            "(SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T2.19"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── dry check: every verdict path against fabricated rows ────────────────
def _dry(bars=None):
    """Known-answer table over the gate logic, with NO brain and NO GPU.

    The bars are injected rather than read from the module so this table is
    runnable while `_GATES_FROZEN` is False — the gate LOGIC is falsifiable
    today even though the gate VALUES wait on the pilot.
    """
    b = bars or dict(UNTRAINED_MAX=0.05, SHUF_MULT=2.0, SHARED_PASS_MIN=0.9,
                     UNI_MIN=0.8, TIE_BAND=0.15, RATIO_MIN=3.0, FLOW_MIN=0.6)
    base_m = dict(finite_all=1.0, params_matched=1.0, loss_fell_all=1.0,
                  untrained_max=0.0, bimodal_success_ratio=8.0,
                  flow_success_bimodal=0.85, reg_success_bimodal=0.1)
    base_c = dict(shuf_mult=6.0, reg_shared_pass_bimodal=0.99,
                  uni_min=0.9, uni_gap=0.03, untrained_max=0.0)
    cases = [
        # the claim path returns a bool (run_spec maps it); only rig faults VOID
        ("planted pass",            {},                          {}, True),
        ("nan anywhere",            {"finite_all": 0.0},         {}, Status.VOID),
        ("params not matched",      {"params_matched": 0.0},     {}, Status.VOID),
        ("an arm never learned",    {"loss_fell_all": 0.0},      {}, Status.VOID),
        ("success free at init",    {"untrained_max": 0.4},      {}, Status.VOID),
        ("conditioning check dead", {}, {"shuf_mult": 1.02},         Status.VOID),
        ("null is dead not collapsed", {}, {"reg_shared_pass_bimodal": 0.2},
                                                                     Status.VOID),
        ("no tie: arms too weak",   {}, {"uni_min": 0.3},            Status.VOID),
        ("no tie: flow wins unimodal too", {}, {"uni_gap": 0.5},     Status.VOID),
        # the genuine falsification the registry names
        ("regression matches flow", {"bimodal_success_ratio": 1.05,
                                     "reg_success_bimodal": 0.8},  {}, False),
        # a ratio won by a near-zero denominator must NOT pass
        ("both arms collapse",      {"bimodal_success_ratio": 20.0,
                                     "flow_success_bimodal": 0.04}, {}, False),
    ]
    ok = True
    for name, dm, dc, want in cases:
        m, c = {**base_m, **dm}, {**base_c, **dc}
        got = _check_with(m, c, b)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {name:34} -> {got}  (want {want})")
    return ok


def _geometry():
    """Known-answer certification of the FIXTURE, with no brain and no GPU.

    The gate table above certifies the verdict logic; this certifies the thing
    the verdict logic reads. Five hand-built actions whose scores are known by
    construction — in particular that the mean of the two modes is a COLLISION
    (conditioned, uncommitted) and that a large lateral with the wrong content
    is NOT a success, which is the vacuous-win hole the conjunction exists to
    close. If either of those rows ever moves, the metric has stopped meaning
    what the docstring says it means.
    """
    import types
    import torch
    cfg = types.SimpleNamespace(obs_dim=256, action_dim=17,
                                action_chunk_size=16)
    bk = _bank(0, cfg, 8)
    idx = np.arange(8)
    up = np.ones(8, dtype="float32")
    left, right = _targets(bk, idx, up), _targets(bk, idx, -up)
    big = np.concatenate([np.full((8, 16, 1), 2.0, dtype="float32"),
                          np.zeros((8, 16, 16), dtype="float32")], axis=2)
    # name, action, (committed, conditioned, success) expected
    cases = [
        ("mode LEFT",                 left,            (1.0, 1.0, 1.0)),
        ("mode RIGHT",                right,           (1.0, 1.0, 1.0)),
        ("MEAN of the two modes",     (left + right) / 2, (0.0, 1.0, 0.0)),
        ("content ignored (zeros)",   np.zeros_like(left), (0.0, 0.0, 0.0)),
        ("big lateral, wrong content", big,            (1.0, 0.0, 0.0)),
    ]
    ok = True
    for name, a, want in cases:
        s = _score(torch.from_numpy(np.ascontiguousarray(a)), bk, idx)
        got = (s["committed"].mean(), s["conditioned"].mean(),
               s["success"].mean())
        good = all(abs(g - w) < 1e-9 for g, w in zip(got, want))
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {name:27} "
              f"|d|={np.abs(s['d']).mean():6.3f} "
              f"content_err={s['content_err'].mean():6.3f} "
              f"-> committed/conditioned/success {got} (want {want})")
    return ok


def _check_with(m, c, b):
    """`_check` with the bars injected — one body, so the table cannot drift
    from the gate it certifies."""
    g = globals()
    saved = {k: g[k] for k in b}
    g.update(b)
    try:
        return _check(m, c)
    finally:
        g.update(saved)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    if mode == "dry":
        print("T2.19 fixture geometry (known-answer, no brain):")
        geo = _geometry()
        print("T2.19 gate table (bars injected; _GATES_FROZEN =",
              _GATES_FROZEN, ")")
        raise SystemExit(0 if (_dry() and geo) else 1)
    elif mode == "smoke":
        # Local, CPU, minutes: production SHAPES (full config, real
        # ActionExpert, real flow sampler), reduced ONLY in steps/batch/bank.
        # Proves the whole path runs and the fixture behaves before any quota.
        out = remote_run([SMOKE_SEED], steps=6, batch=4, n_scen=4)
        m = _fold(out["seeds"])
        print(json.dumps(out["seeds"][0], indent=1)[:2000])
        print(json.dumps({k: v for k, v in m.items()
                          if k != "loss_first_last"}, indent=1))
        assert m["params_matched"], "arms are not parameter-identical"
        assert m["finite_all"], "non-finite reading in the smoke"
        print("SMOKE OK")
    elif mode == "pilot":
        # ONE seed, a LONGER budget, and a curve. Not three seeds at STEPS:
        # the pilot's questions are "can either arm reach the task at all" and
        # "how many steps does it need", and a seed spread answers neither.
        # The registered run is the one that spends seeds.
        p_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
        p_every = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        out = _submit([SMOKE_SEED], steps=p_steps, curve_every=p_every)
        out["pilot"] = {"steps": p_steps, "curve_every": p_every,
                        "seed": SMOKE_SEED}
        Path("/data/t219_pilot.json").write_text(json.dumps(out, indent=1))
        for leg in ("flow_bimodal", "reg_bimodal",
                    "flow_unimodal", "reg_unimodal"):
            row = out["seeds"][0][leg]
            print(f"\n{leg}: loss {row['loss_first']} -> {row['loss_last']}")
            for p in row["curve"]:
                print(f"   step {p['step']:5}  success {p['success']:.3f}  "
                      f"committed {p['committed']:.3f}  "
                      f"conditioned {p['conditioned']:.3f}  "
                      f"|d| {p['abs_lateral']:.3f}  "
                      f"content_err {p['content_err']:.3f}")
        print("\nPILOT ARTIFACT /data/t219_pilot.json")
    else:
        run()
