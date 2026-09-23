"""T4.06 — Fusion balancing bakeoff: three arms against the shipped brain.

DESIGNED BY THE REVIEW, 2026-09-23 (docs/REVIEW_QUEUE.md,
`t402-touch-drowns-audio-at-the-fusion-boundary`, DISPOSITIONED). This module
implements that design verbatim; where a number had to be operationalised the
choice is declared here, in source, before the run.

THE INCUMBENT AND THE PREMISE. `T4.02` attempt 4 (2026-08-21T07:26:36, FAIL)
measured worst-seed `max_modality_grad_ratio` 30.1197 with std 3.55e-15
across seeds — architecture, not a seed lottery: touch (~2.9e-3) drowns audio
(~1e-4) at the CrossModalFusion boundary against the registry's exogenous 10x
gate (written 2026-08-04). **That 10x gate does not move in any arm, in
either direction.** Every arm runs T4.02's SHIPPED rig unchanged — same
matched-information fixture, same equal-variance latents, same hooks, same
learning gate, same grad-scale control — imported from
`t4_02_no_modality_collapse` rather than copied, and declared in IMPL_DEPS so
this certificate stales if that rig moves.

THE DISQUALIFICATION, WHICH IS THE DESIGN'S ONE REAL DECISION. Arm (a)
equalises `max_modality_grad_ratio` BY CONSTRUCTION and cannot fail the
stated metric. So the bakeoff carries a SECOND, STRICTLY HARDER conjunct —
`min_modality_latent_r2`: after training, each modality's k=8 latent z_m is
probed by ridge regression from the FUSED representation (the
CrossModalFusion CLS output, the one vector every sense must reach for the
senses to teach each other) on held-out draws; the statistic is the WORST
modality's R2 per seed, gated at the minimum over seeds. **The bar is the
incumbent's own measured value** — T4.02's shipped brain re-run unchanged as
arm zero, executed FIRST in the same submission; the RULE (bar = incumbent's
min_modality_latent_r2, an arm must strictly EXCEED it) is pre-registered
here before any number exists. An arm that clears the ratio while leaving
the worst sense's recovery at or below the incumbent's is REFUTED — it moved
the bookkeeping and not the creature.

STATISTIC_BOUND (the unsaturated-null rule, adopted 2026-09-23):
  - max_modality_grad_ratio: bound 1.0 (max/min >= 1). The anchor (incumbent)
    measured 30.12 — 3x the far side of the 10.0 gate, distance from the
    bound ~29. Not saturated.
  - min_modality_latent_r2: bound 1.0. The anchor is the incumbent's in-run
    measured value; it cannot be quoted here before it exists, so the
    saturation check is a RUNTIME VOID LANE: if the incumbent's
    min_modality_latent_r2 >= 0.99 the exceed-the-anchor conjunct is not
    satisfiable within noise and the gate is not registerable — the run
    returns VOID with `anchor_saturated`, and the repair is a change of
    statistic or venue, never envelope.
  - RECORDED AFTER ATTEMPT 1 (PASS, 2026-09-23; 109th audit RANK 1 — a
    correction of the claim's REACH, the certificate itself stands): the
    anchor ARRIVED at min_modality_latent_r2 = -2.3939, nowhere near the
    1.0 bound the saturation lane guards — but the winner (loss_reweight)
    cleared it by 0.0187 against the incumbent's own seed-to-seed spread of
    0.2699 (margin = 6.9% of the spread), with one of three seeds regressing.
    The saturation lane could not see this failure mode: the risk was never
    the bound, it was a margin small against the anchor's own noise. So read
    conjunct (2) honestly: the RATIO result (29.83x -> 2.45x vs the exogenous
    10x gate) is the demonstrated thing; the latent-recovery conjunct
    certified the winner INSIDE the anchor's noise and must not be quoted as
    demonstrated. Adoption remains the Review's under the t402 row. In every
    arm including the winner, four of five senses read latent R2 in
    [-2.38, -0.17] from the fused representation, and only proprioception is
    positive — whether that is the brain or a ridge probe fitting 513 params
    to 768 rows is not answerable from this run and is now askable.

THE ARMS (mechanisms declared before the run; nothing tuned against a gate):
  incumbent — the shipped brain, T4.02's rig byte-identical. Establishes the
    R2 bar and the paired eval-loss reference. Under the winner rule it MUST
    NOT win (see control).
  (a) grad_norm — per-modality gradient normalisation at the fusion boundary.
    Forward identity; backward rescales each modality's boundary gradient to
    the mean of the five RAW boundary-gradient norms of the PREVIOUS step
    (step 0 passes through unscaled; per-backward-call raw norms are averaged
    per step). Equalises the measured ratio by construction while preserving
    the overall gradient scale — normalising to unit norm would multiply
    ~1e-3 norms by ~1e3 and measure divergence, not balance.
  (b) loss_reweight — per-modality weights on the shipped objective, applied
    as frozen backward scalars at the boundary (forward identity, backward
    x w_m — `_GradScale` with per-modality gains). THE DECLARED RULE, frozen
    off T4.02 attempt 4's recorded numbers and never revisited: w_m
    proportional to 1 / (attempt 4's mean boundary norm for m over its three
    seeds), normalised to geometric mean 1. The frozen inputs are quoted in
    INC_NORMS_A4 below, verbatim from the ledger row. A weight tuned until
    the ratio clears would be threshold-moving in a hat; these were computed
    once, from numbers recorded 33 days before this file existed.
  (c) modality_dropout — each sense's fusion token multiplied by an
    independent Bernoulli(1-DROP_P) mask, drawn ONCE PER STEP (the objective
    forwards twice per step; the two forwards must see the same world), from
    a dedicated generator (seed*7919+3) so the global RNG stream stays
    matched across arms. If all five draw 0 the step redraws. No rescaling;
    masks off at eval/probe. DROP_P = 0.2, declared here. The only arm whose
    mechanism never mentions the measured quantity — the cleanest test of
    whether the metric tracks anything.

THE WINNER RULE, pre-registered: an arm wins iff ALL of
  (1) worst-seed max_modality_grad_ratio <= 10.0  (T4.02's gate, unmoved),
  (2) min-over-seeds min_modality_latent_r2 STRICTLY exceeds the incumbent's,
  (3) mean held-out eval loss <= the incumbent's mean (measured over the SAME
      fixture draws — the per-seed draw sequence is deterministic, so the
      comparison is paired; balance bought by breaking the task is not
      balance),
  (4) every VOID lane green for that arm.
The spec's claim: n_winning_arms >= 1. Arms that clear (1) and fail (2) are
recorded in `refuted_arms` — that outcome is a RESULT (the imbalance is a
symptom), not a failure of the bakeoff.

VOID LANES, inherited from T4.02 and NOT relaxed, binding EVERY arm
including the incumbent: finite losses; fired_ok (each boundary hook fires
exactly 2x/step); fixture share gate (each modality's realised variance
share in [0.10, 0.30]); the learning gate (mean loss over the last quarter
below the first, every seed — a non-learner cannot arbitrate routing); and
the grad-scale control (frozen snapshot, vision token backward x100 planted
INSIDE the arm's mechanism, measured dominance > 10x on every seed — the
detector must see imposed dominance THROUGH the arm's machinery or the arm's
reading is blind). Plus `anchor_saturated` above.

CONTROL (spec-level, must fail). The incumbent evaluated under the winner
rule must NOT win: it cannot strictly exceed its own R2 bar, and its ratio
is a measured 30x. If the harness crowns the unchanged brain, or the
incumbent's worst-seed ratio arrives <= 10 (the premise this whole bakeoff
rests on, measured twice at zero variance), the rig changed underneath the
bakeoff and the run is VOID — control-green maps to VOID, not FAIL.

GPU. One submission for the whole spec (module cache, T2.01 pattern), Kaggle
first: 2026-W38 holds ~29.5 free hours expiring Saturday 2026-09-26 and this
is their one legal buyer (Review, 2026-09-23). Sizing multiplies by seeds AND
arms (LESSONS): T4.02 attempt 4 ran 3 seeds x (300+50 steps) in 514.69 s;
four arms plus probes is ~0.75 h projected.

COVERS: one brain / unison (rule)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about balancing the shipped brain's fusion routing, measured
# by T4.02's own rig: both hash into this certificate.
IMPL_DEPS = ["UnifiedBrain.py",
             "experiments/tests/t4_02_no_modality_collapse.py"]

from .t4_02_no_modality_collapse import (  # noqa: E402 — the shipped rig
    _Fixture, _GradMeter, _GradScale, MODALITIES, BATCH, STEPS, CTRL_STEPS,
    LR, FWD_PER_STEP, RATIO_MAX, SHARE_BAND, PLANT_GAIN, ZERO_SENTINEL)

SEEDS = [0, 1, 2]
SMOKE_SEED = 90

INCUMBENT = "incumbent"
ARMS = (INCUMBENT, "grad_norm", "loss_reweight", "modality_dropout")

# T4.02 attempt 4 (ledger, ran_at 2026-08-21T07:26:36): per-modality boundary
# norms, mean over seeds 0/1/2 — the FROZEN inputs to arm (b)'s declared rule.
INC_NORMS_A4 = {
    "proprio": (0.000792 + 0.000809 + 0.000555) / 3.0,   # 0.000719
    "vision": (0.000919 + 0.000471 + 0.000565) / 3.0,    # 0.000652
    "touch": (0.002933 + 0.000981 + 0.002654) / 3.0,     # 0.002189
    "audio": (0.000104 + 0.000080 + 0.000088) / 3.0,     # 0.0000907
    "language": (0.000840 + 0.000579 + 0.000762) / 3.0,  # 0.000727
}
_inv = {m: 1.0 / v for m, v in INC_NORMS_A4.items()}
_gm = math.exp(sum(math.log(v) for v in _inv.values()) / len(_inv))
ARM_B_WEIGHTS = {m: round(v / _gm, 4) for m, v in _inv.items()}
# -> proprio ~0.81, vision ~0.89, touch ~0.27, audio ~6.43, language ~0.80

DROP_P = 0.2
R2_SATURATION = 0.99      # anchor at the statistic's bound -> VOID, not a gate
PROBE_N = 1152            # held-out draws for the R2 probe (also the paired
PROBE_BATCH = 48          # eval-loss sample); first 2/3 fit, last 1/3 score
RIDGE_LAMBDA = 1e-3

# The five fusion-boundary attributes T4.02 instruments, in MODALITIES order.
BOUNDARY_ATTRS = {"proprio": "proprio_encoder", "vision": "vision_proj",
                  "touch": "touch_proj", "audio": "audio_proj",
                  "language": "language_proj"}


# ── arm mechanisms (forward-identity unless stated; nothing touches the rig) ─
class _NormState:
    """Arm (a)'s shared state: raw per-modality backward norms this step;
    target = mean of the previous step's five per-modality raw norms."""

    def __init__(self):
        self.raw = {m: [] for m in MODALITIES}
        self.target = None          # None -> step 0 passes through

    def step_end(self):
        means = [float(np.mean(v)) for v in self.raw.values() if v]
        if len(means) == len(MODALITIES):
            self.target = float(np.mean(means))
        self.raw = {m: [] for m in MODALITIES}


def _grad_norm_wrap(inner, name: str, state: _NormState):
    import torch

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            y = self.inner(x)
            if torch.is_grad_enabled() and y.requires_grad:
                st, nm = state, name

                class _F(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, t):
                        return t

                    @staticmethod
                    def backward(ctx, g):
                        n = float(g.norm())
                        st.raw[nm].append(n)
                        if st.target is None or n == 0.0:
                            return g
                        return g * (st.target / n)

                return _F.apply(y)
            return y

    return _Wrap()


class _DropState:
    """Arm (c)'s per-step masks, one draw per step so the objective's two
    forwards see the same world; dedicated RNG so arms stay stream-matched."""

    def __init__(self, seed: int):
        self.rng = np.random.RandomState((seed * 7919 + 3) % 2**32)
        self.mask = {m: 1.0 for m in MODALITIES}
        self.active = False

    def draw(self):
        while True:
            m = {k: float(self.rng.rand() >= DROP_P) for k in MODALITIES}
            if any(m.values()):
                self.mask = m
                return


def _dropout_wrap(inner, name: str, state: _DropState):
    import torch

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            y = self.inner(x)
            if state.active:
                return y * state.mask[name]
            return y

    return _Wrap()


# ── one arm-seed: train (or freeze+plant), measure, probe ────────────────────
def _train_measure_arm(seed: int, arm: str, steps: int, batch: int,
                       device: str, plant_gain: float | None = None,
                       probe_n: int = PROBE_N,
                       probe_batch: int = PROBE_BATCH) -> dict:
    """T4.02's `_train_measure` loop with the arm's mechanism stacked OUTSIDE
    the plant (inner module -> plant [control only] -> arm wrapper), the meter
    on the INNER modules exactly as T4.02 hooks them. plant_gain None = the
    experiment (Adam steps, then the R2 probe + paired eval loss); a float =
    the frozen-snapshot control."""
    import torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig(llm_enabled=False)
    brain = UnifiedBrain(cfg).to(device).train()
    fx = _Fixture(seed, cfg)

    norm_state = _NormState() if arm == "grad_norm" else None
    drop_state = _DropState(seed) if arm == "modality_dropout" else None

    inners = {}
    for m, attr in BOUNDARY_ATTRS.items():
        inner = getattr(brain, attr)
        inners[m] = inner
        mod = inner
        if plant_gain is not None and m == "vision":
            mod = _GradScale(mod, plant_gain).to(device)
        if arm == "grad_norm":
            mod = _grad_norm_wrap(mod, m, norm_state).to(device)
        elif arm == "loss_reweight":
            mod = _GradScale(mod, ARM_B_WEIGHTS[m]).to(device)
        elif arm == "modality_dropout":
            mod = _dropout_wrap(mod, m, drop_state).to(device)
        if mod is not inner:
            setattr(brain, attr, mod)

    meter = _GradMeter(inners)
    opt = (torch.optim.Adam(brain.parameters(), lr=LR)
           if plant_gain is None else None)
    if drop_state is not None:
        drop_state.active = True

    losses, norm_rows, fired_ok, finite = [], [], True, True
    for _ in range(steps):
        if drop_state is not None:
            drop_state.draw()
        b = fx.draw(batch)
        out = brain.action_training_loss(
            torch.from_numpy(b["state"]).to(device),
            torch.from_numpy(b["target"]).to(device),
            language=torch.from_numpy(b["tokens"]).long().to(device),
            vision=torch.from_numpy(b["vision"]).to(device),
            touch=torch.from_numpy(b["touch"]).to(device),
            audio=torch.from_numpy(b["audio"]).to(device))
        loss = out["loss"]
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            brain.zero_grad(set_to_none=True)
        loss.backward()
        norms, fired = meter.pop()
        fired_ok = fired_ok and all(fired[m] == FWD_PER_STEP for m in MODALITIES)
        norm_rows.append(norms)
        losses.append(float(loss))
        if norm_state is not None:
            norm_state.step_end()
        if opt is not None:
            opt.step()

    for h in meter.handles:
        h.remove()

    def window_mean(rows):
        return {m: float(np.mean([r[m] for r in rows])) for m in MODALITIES}

    def ratio_of(mean_norms):
        lo, hi = min(mean_norms.values()), max(mean_norms.values())
        return ZERO_SENTINEL if lo == 0.0 else hi / lo

    n = len(norm_rows)
    late = window_mean(norm_rows[n // 2:]) if n else {m: 0.0 for m in MODALITIES}
    q = max(1, len(losses) // 4)
    others_min = min(v for m, v in late.items() if m != "vision") if n else 0.0
    row = {
        "arm": arm, "seed": seed,
        "finite": finite, "fired_ok": bool(fired_ok), "steps_run": n,
        "shares": fx.shares,
        "norms": {m: round(v, 6) for m, v in late.items()},
        "ratio": round(ratio_of(late), 4),
        "vision_dominance": round(
            ZERO_SENTINEL if others_min == 0.0 else late["vision"] / others_min, 4),
        "loss_first": round(float(np.mean(losses[:q])), 4) if losses else float("nan"),
        "loss_last": round(float(np.mean(losses[-q:])), 4) if losses else float("nan"),
    }

    if plant_gain is not None or not finite:
        return row

    # ── the probe: min_modality_latent_r2 + paired eval loss ────────────────
    brain.eval()
    if drop_state is not None:
        drop_state.active = False
    captures: list = []
    hook = brain.cross_modal_fusion.register_forward_hook(
        lambda _m, _a, out: captures.append(
            out[:, -1, :].detach().float().cpu().numpy()))
    X_rows, Z_rows, eval_losses = [], {m: [] for m in MODALITIES}, []
    with torch.no_grad():
        for _ in range(max(1, probe_n // probe_batch)):
            # Draw latents first (the probe targets), then embed them exactly
            # as fx.draw does — same generative path, held-out by freshness.
            z, cls = fx._draw_z(probe_batch)
            rng = fx.rng
            state = z["proprio"] @ fx.E_p + 0.5 * rng.randn(
                probe_batch, fx.obs_dim).astype(np.float32)
            vis = 0.5 + 0.18 * (z["vision"] @ fx.B_v) + 0.08 * rng.randn(
                probe_batch, fx.B_v.shape[1]).astype(np.float32)
            vision = np.clip(vis, 0.0, 1.0).reshape(probe_batch, 3, 224, 224)
            touch = z["touch"] @ fx.E_t + 0.5 * rng.randn(
                probe_batch, 10).astype(np.float32)
            audio = z["audio"] @ fx.A_a + 0.3 * rng.randn(
                probe_batch, fx.audio_n).astype(np.float32)
            tokens = fx.templates[cls]
            fhat = [((z[m] @ fx.W[m]) - fx.mu[m]) / fx.sd[m] for m in MODALITIES]
            target = (sum(fhat) / np.sqrt(len(MODALITIES))).astype(np.float32)
            n_fires = len(captures)
            out = brain.action_training_loss(
                torch.from_numpy(state.astype(np.float32)).to(device),
                torch.from_numpy(target.reshape(probe_batch, *fx.chunk)).to(device),
                language=torch.from_numpy(tokens).long().to(device),
                vision=torch.from_numpy(vision.astype(np.float32)).to(device),
                touch=torch.from_numpy(touch.astype(np.float32)).to(device),
                audio=torch.from_numpy(audio.astype(np.float32)).to(device))
            eval_losses.append(float(out["loss"]))
            X_rows.append(captures[n_fires])   # first fire of this batch
            for m in MODALITIES:
                Z_rows[m].append(z[m])
    hook.remove()
    del captures

    X = np.concatenate(X_rows, 0).astype(np.float64)
    n_fit = (2 * len(X)) // 3
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb[:n_fit].T @ Xb[:n_fit] + RIDGE_LAMBDA * np.eye(Xb.shape[1])
    r2 = {}
    for m in MODALITIES:
        Z = np.concatenate(Z_rows[m], 0).astype(np.float64)
        W = np.linalg.solve(A, Xb[:n_fit].T @ Z[:n_fit])
        pred = Xb[n_fit:] @ W
        resid = ((Z[n_fit:] - pred) ** 2).sum(0)
        tot = ((Z[n_fit:] - Z[n_fit:].mean(0)) ** 2).sum(0)
        tot[tot < 1e-12] = 1e-12
        r2[m] = round(float(np.mean(1.0 - resid / tot)), 4)
    row["latent_r2"] = r2
    row["min_modality_latent_r2"] = min(r2.values())
    row["eval_loss"] = round(float(np.mean(eval_losses)), 4)
    return row


# ── remote entry point: every arm, every seed, incumbent FIRST ──────────────
def remote_run(seeds: list, steps: int = STEPS, ctrl_steps: int = CTRL_STEPS,
               batch: int = BATCH, probe_n: int = PROBE_N,
               probe_batch: int = PROBE_BATCH) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "arm_b_weights": ARM_B_WEIGHTS, "arms": {}}
    for arm in ARMS:                       # incumbent first: the bar exists
        rows = []                          # before any arm's number does
        for seed in seeds:
            rows.append({
                "experiment": _train_measure_arm(
                    seed, arm, steps, batch, device,
                    probe_n=probe_n, probe_batch=probe_batch),
                "control": _train_measure_arm(
                    seed, arm, ctrl_steps, batch, device,
                    plant_gain=PLANT_GAIN),
            })
            print("ARM_SEED_DONE", arm, seed, flush=True)
        out["arms"][arm] = rows
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ─────────────
JOB = r'''
import json, os as _o
from experiments.tests.t4_06_fusion_balancing_bakeoff import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t406.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Sizing multiplies by seeds AND arms (LESSONS): T4.02's 3x(300+50) ran
    # 514.69 s; four arms plus probes projects ~0.75 h. Timeout generous.
    res = submit(job, prefer="kaggle",
                 est_hours=round(len(ARMS) * (0.05 + 0.06 * len(seeds)), 2),
                 timeout_s=2400 + 1200 * len(seeds),
                 fetch=["t406.json"])
    if not res.ok:
        raise RuntimeError(f"T4.06 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t406.json"]).read_text())
    out["backend"] = res.backend
    return out


def _arm_summary(rows: list) -> dict:
    exp = [r["experiment"] for r in rows]
    ctl = [r["control"] for r in rows]
    all_shares = [s for r in exp for s in r["shares"].values()]
    return {
        "ratio_worst": max(r["ratio"] for r in exp),
        "ratio_per_seed": [r["ratio"] for r in exp],
        "min_r2": min(r.get("min_modality_latent_r2", float("nan")) for r in exp),
        "r2_per_seed": [r.get("min_modality_latent_r2") for r in exp],
        "latent_r2_per_seed": [r.get("latent_r2") for r in exp],
        "eval_loss_per_seed": [r.get("eval_loss") for r in exp],
        "eval_loss_mean": round(float(np.mean(
            [r.get("eval_loss", float("nan")) for r in exp])), 4),
        "loss_decreased_all": float(all(
            r["loss_last"] < r["loss_first"] for r in exp)),
        "finite_all": float(all(r["finite"] for r in exp)
                            and all(r["finite"] for r in ctl)),
        "fired_ok_all": float(all(r["fired_ok"] for r in exp)
                              and all(r["fired_ok"] for r in ctl)),
        "share_min": min(all_shares), "share_max": max(all_shares),
        "ctrl_vision_dominance_min": min(r["vision_dominance"] for r in ctl),
        "norms_per_seed": [r["norms"] for r in exp],
    }


def _winner_lanes(s: dict, bar_r2: float, inc_eval: float) -> dict:
    """The pre-registered winner rule, one arm vs the incumbent's numbers."""
    lanes_green = (s["finite_all"] and s["fired_ok_all"]
                   and s["loss_decreased_all"]
                   and SHARE_BAND[0] <= s["share_min"]
                   and s["share_max"] <= SHARE_BAND[1]
                   and s["ctrl_vision_dominance_min"] > RATIO_MAX)
    ratio_ok = s["ratio_worst"] <= RATIO_MAX
    r2_ok = s["min_r2"] > bar_r2
    loss_ok = s["eval_loss_mean"] <= inc_eval
    return {"lanes_green": lanes_green, "ratio_ok": ratio_ok,
            "r2_ok": r2_ok, "loss_ok": loss_ok,
            "wins": bool(lanes_green and ratio_ok and r2_ok and loss_ok)}


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    arms = {a: _arm_summary(_CACHE["arms"][a]) for a in ARMS}
    inc = arms[INCUMBENT]
    bar_r2, inc_eval = inc["min_r2"], inc["eval_loss_mean"]
    verdicts = {a: _winner_lanes(arms[a], bar_r2, inc_eval)
                for a in ARMS if a != INCUMBENT}
    winners = sorted(a for a, v in verdicts.items() if v["wins"])
    refuted = sorted(a for a, v in verdicts.items()
                     if v["lanes_green"] and v["ratio_ok"] and not v["r2_ok"])
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "n_winning_arms": len(winners),
        "winning_arms": winners, "refuted_arms": refuted,
        "arm_verdicts": verdicts,
        "incumbent_min_r2_bar": bar_r2,
        "incumbent_eval_loss_mean": inc_eval,
        "incumbent_ratio_worst": inc["ratio_worst"],
        "anchor_saturated": float(bar_r2 >= R2_SATURATION),
        "arm_b_weights": _CACHE["arm_b_weights"],
        "arms": arms,
        "all_finite": float(all(a["finite_all"] for a in arms.values())),
        "all_fired_ok": float(all(a["fired_ok_all"] for a in arms.values())),
        "all_learned": float(all(a["loss_decreased_all"] for a in arms.values())),
        "all_ctrl_dominance_ok": float(all(
            a["ctrl_vision_dominance_min"] > RATIO_MAX for a in arms.values())),
        "share_min": min(a["share_min"] for a in arms.values()),
        "share_max": max(a["share_max"] for a in arms.values()),
    }


def _control(seed: int) -> dict:
    arms = {a: _arm_summary(_CACHE["arms"][a]) for a in ARMS}
    inc = arms[INCUMBENT]
    v = _winner_lanes(inc, inc["min_r2"], inc["eval_loss_mean"])
    return {
        "ctrl_incumbent_wins": float(v["wins"]),
        "ctrl_incumbent_ratio_worst": inc["ratio_worst"],
        "ctrl_incumbent_still_red": float(inc["ratio_worst"] > RATIO_MAX),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not (m["all_finite"] and m["all_fired_ok"]):
        return Status.VOID          # NaN or a dead hook — measured nothing
    if not (SHARE_BAND[0] <= m["share_min"]
            and m["share_max"] <= SHARE_BAND[1]):
        return Status.VOID          # fixture shares unbalanced — task, not brain
    if not m["all_learned"]:
        return Status.VOID          # a non-learner cannot arbitrate routing
    if not m["all_ctrl_dominance_ok"]:
        return Status.VOID          # some arm's detector cannot see dominance
    if m["anchor_saturated"]:
        return Status.VOID          # R2 anchor at its bound — unregisterable
    if c["ctrl_incumbent_wins"] or not c["ctrl_incumbent_still_red"]:
        return Status.VOID          # rig changed under the bakeoff
    # The claim: at least one arm balances the boundary AND improves the
    # worst sense's recovery, at no cost to the task.
    return m["n_winning_arms"] >= 1


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T4.06"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, minutes: production shapes, reduced ONLY in steps/batch/
        # probe count. Proves all four arms build, hooks attach, arm (a)
        # equalises, the plant inflates through every arm's mechanism, and
        # the probe returns finite R2s — before any quota is spent.
        out = remote_run([SMOKE_SEED], steps=8, ctrl_steps=4, batch=2,
                         probe_n=48, probe_batch=8)
        for arm in ARMS:
            row = out["arms"][arm][0]
            exp, ctl = row["experiment"], row["control"]
            assert exp["fired_ok"] and ctl["fired_ok"], f"{arm}: hooks not 2x"
            assert exp["finite"] and ctl["finite"], f"{arm}: non-finite"
            assert ctl["vision_dominance"] > RATIO_MAX, \
                f"{arm}: plant not seen through the mechanism " \
                f"({ctl['vision_dominance']})"
            assert "min_modality_latent_r2" in exp, f"{arm}: probe missing"
        gn = out["arms"]["grad_norm"][0]["experiment"]["ratio"]
        assert gn < 2.0, f"grad_norm did not equalise (ratio {gn})"
        print(json.dumps({a: out["arms"][a][0]["experiment"] for a in ARMS},
                         indent=1))
        print("SMOKE OK")
    else:
        run()
