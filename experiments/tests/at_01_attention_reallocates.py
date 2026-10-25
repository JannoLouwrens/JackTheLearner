"""AT.01 — attention reallocates when a channel goes dark, or selection is
decoration.

THE QUESTION (anatomy audit 2026-10-25, the Attention seat's ring). GOAL.md
justifies olfaction as "the sense that works when sight fails" — a claim about
REALLOCATION: ten channels onto one finite brain means something selects, and
nothing in the ladder ever tested that the selection responds to a channel
going dark. This spec races input-dependent modality reweighting against the
registry's null — fixed modality weights fitted on unoccluded trials and
frozen — on trials where vision is occluded and the cue arrives by audio and
smell.

THE VENUE. K=8 cue classes (chance 0.125), three channels (vision, audio,
smell), each a D=16 vector reading code_m[c] + sigma*N(0,I) when it carries
the cue. Per-channel codebooks are unit-norm rows drawn once from a fixed
venue generator (seed 424242) — the world is the world; seeds vary training
stochasticity, not the venue.

  CLEAN trial (the training distribution, for BOTH arms): every channel
  carries the cue, with per-trial fidelity drawn per channel — vision
  sigma ~ U[0.10, 0.80], audio/smell sigma ~ U[0.40, 1.20]. Vision is
  usually (not always) the best channel, so a static fit favours it, while
  the overlap means per-trial quality varies — which is the only thing that
  makes a quality->weight rule LEARNABLE from clean data. Redundancy across
  the three carrying channels is deliberate: it compresses any per-trial
  reweighting advantage on clean trials toward zero (control 1's job) while
  leaving the null a live learner when vision dies (the T2.02 lane's job).
  OCCLUDED trial (the claim's eval block, never trained on): vision reads
  pure noise at OCC_SIGMA=1.5 — no cue content, energy above the clean
  range, the fog GOAL.md names — audio and smell carry as usual.
  ALL-EQUAL trial (control 2's block): every channel carries the cue at one
  fixed sigma SIG_EQ=0.30 — selection has nothing to select.

ARMS. Selection can live ONLY in the pooling weights, by construction: both
arms share the identical architecture — per-channel LINEAR encoders
Linear(D,H) and a LINEAR head over the weighted sum of embeddings. A linear
encoder cannot implement input-conditional attenuation internally, so the one
degree of freedom that differs between the arms is the one the claim names.

  fixed   w = softmax(theta), theta a free 3-vector learned on clean trials
          and frozen with everything else — the registry's null verbatim
          (initialised at uniform pooling; it fits static weights, selects
          nothing per-input).
  gate    w = softmax(MLP(stats)) where stats are per-channel [rms, max-abs]
          (6 numbers — class-blind energy summaries), hidden 16. Trained on
          the SAME clean stream, frozen the same way. Occlusion is never
          seen in training: reallocation at test time must come from the
          learned quality rule, not from fitting the occluded block.

Both arms train on IDENTICAL clean batch streams (one seeded generator per
arm, same seed, same draw order), identical optimiser/iters, and are
evaluated on IDENTICAL pre-generated blocks. Parameter counts are reported,
not gated (gate arm +~200 params): the registry's own capacity control is
control 1 — an arm that wins because it is bigger wins on clean trials too,
and that is exactly what control 1 forbids.

THE DECISION RULE, pre-registered before any claim-arm number existed:

  VOID  non-finite training loss; or the null never fit its own training
        distribution (null_clean_acc < FIT_MIN — the rig, not the claim); or
        the null is DEAD on the occluded block (null_occ_acc < NULL_LIVE_MIN
        — a dead null cannot arbitrate, T2.02); or the null is SATURATED
        there (null_occ_acc > 1 - MARGIN_MIN — unsaturated-null rule firing
        at runtime).
  PASS  on EVERY seed: occluded margin = gate_occ_acc - fixed_occ_acc
        >= MARGIN_MIN, AND control 1 holds (unoccluded margin
        <= CTRL_MARGIN_MAX), AND control 2 holds (all-equal margin
        <= CTRL_MARGIN_MAX).
  FAIL  otherwise — including a gate that learned a CONSTANT policy (its
        occluded weights are reported for diagnosis), and including a gate
        that beats the null on clean trials too (capacity, not selection —
        the registry's own words).

WORST-SEED GATE, from authorship (the WM.01 incident, same sitting,
2026-10-25): `run_spec` aggregates per-seed metrics by MEAN and calls
`_check` once, so every deciding conjunct here is a per-seed BOOLEAN whose
mean is 1.0 only if every seed cleared: `margin_ok` (experiment side),
`ctrl_unocc_ok` and `ctrl_allch_ok` (control side), all gated `== 1.0`.

CTRL_MARGIN_MAX implements the registry's "margin <= 0 within seed spread"
as a fixed pre-registered epsilon: at N_EVAL=512 and a clean operating point
near 0.9, the paired binomial SE of an accuracy difference is <= ~0.019, so
0.03 is a 1.5-2 sigma allowance around zero — declared here before any
number was read, and never derived from an observed spread.

STATISTIC_BOUND (the unsaturated-null rule), with the venue trail (the
WM.01 idiom: the VENUE moved on the NULL's own readings, before the gate arm
was ever built; the bars above were frozen first and none has moved).
Null-only pre-checks, sizing seed 90 — never a claim seed, claim arm never
constructed, nothing written to the ledger — recorded 2026-10-25 before the
pre-registration commit:

  v1 (SIG_O_HI 1.20, OCC_SIGMA 1.5, MAX_ITERS 800): null_clean_acc 0.8711,
  null_occ_acc 0.2422 — the null marginally DEAD on the occluded block
  (under NULL_LIVE_MIN 0.25), so the gate was not registerable there: a
  margin over a dead null arbitrates nothing (T2.02).
  v2 (registered: SIG_O_HI 1.00, OCC_SIGMA 1.2, MAX_ITERS 1200):
  null_clean_acc 0.9082 (fit lane clear), null_occ_acc 0.3047 — alive at
  2.4x chance, distance to bound 0.6953 = 7x the required margin,
  null_w_vision 0.4617 (the static fit sits near uniform pooling, the
  registry's own phrase). Registerable on both counts.

CONTROL-ALIVE (LESSONS: a control must carry proof its instrument was
alive): both controls are near-zero-margin statements, and their presence is
demonstrated at the identical operating point by the experiment itself — the
same two frozen models whose occluded margin is the claim produce the
control margins on blocks generated the same way. A dead rig cannot show a
large occluded margin and zero control margins at once.

FORECAST, pre-registered with its reasoning (ps09 idiom): lean PASS — the
gradient on clean trials rewards inverse-variance weighting, the quality
signal (per-channel rms) is linearly separable from the gate's inputs, and
occluded vision reads MORE extreme than any training noise, in the direction
the learned rule generalises. Two live risks, either of which is an honest
FAIL: (i) the gate converges to a near-constant vision-heavy policy (the
redundancy that protects control 1 also weakens the pressure toward input
dependence), reading as occluded margin ~0; (ii) the gate's per-trial
selection buys a real margin on CLEAN trials, tripping control 1 — in which
case this venue cannot separate selection from capacity and the finding
routes to the Review. Either branch is a result: the reallocation claim has
never had a measured number.
"""
from __future__ import annotations

import math

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

K = 8                       # cue classes -> chance 0.125
M = 3                       # channels: vision, audio, smell
D = 16                      # dims per channel
H = 24                      # embedding width (shared by both arms)
GATE_HIDDEN = 16
VENUE_SEED = 424242         # fixed codebooks: the world, not the seed

SIG_V_LO, SIG_V_HI = 0.10, 0.80    # clean vision fidelity range
SIG_O_LO, SIG_O_HI = 0.40, 1.00    # clean audio/smell fidelity range
OCC_SIGMA = 1.2                    # occluded vision: pure noise, no cue
SIG_EQ = 0.30                      # control 2: all channels, equal fidelity

BATCH = 128
MAX_ITERS = 1200
LR = 5e-3
N_EVAL = 512                # per eval block per seed

# Bars — frozen before any number was read (see docstring).
MARGIN_MIN = 0.10           # the claim: occluded margin, per seed
CTRL_MARGIN_MAX = 0.03      # both control margins must sit at/below, per seed
NULL_LIVE_MIN = 0.25        # VOID lane: dead null cannot arbitrate (2x chance)
FIT_MIN = 0.80              # VOID lane: the null must fit its own train dist

_CACHE: dict = {}           # seed -> trained artifacts, shared with _control


def _codebooks():
    import torch
    gen = torch.Generator().manual_seed(VENUE_SEED)
    books = torch.randn(M, K, D, generator=gen)
    return books / books.norm(dim=-1, keepdim=True)


def _trial_batch(n: int, gen, books, block: str):
    """One batch of trials. block: 'clean' (training dist), 'occluded'
    (vision = pure noise at OCC_SIGMA, cue in audio+smell), 'all_equal'
    (every channel carries at SIG_EQ). Returns (x: (n, M, D), y: (n,))."""
    import torch
    y = torch.randint(0, K, (n,), generator=gen)
    x = torch.empty(n, M, D)
    for m in range(M):
        noise = torch.randn(n, D, generator=gen)
        if block == "occluded" and m == 0:
            x[:, m] = OCC_SIGMA * noise
            continue
        if block == "all_equal":
            sig = torch.full((n, 1), SIG_EQ)
        elif m == 0:
            sig = SIG_V_LO + (SIG_V_HI - SIG_V_LO) * torch.rand(
                n, 1, generator=gen)
        else:
            sig = SIG_O_LO + (SIG_O_HI - SIG_O_LO) * torch.rand(
                n, 1, generator=gen)
        x[:, m] = books[m][y] + sig * noise
    return x, y


def _build(arm: str):
    import torch
    import torch.nn as nn

    class _Pooled(nn.Module):
        """Per-channel LINEAR encoders, weighted sum, LINEAR head. The pooling
        weights are the only input-conditional degree of freedom, and only in
        the gate arm (docstring: selection cannot hide in a linear encoder)."""

        def __init__(self, gated: bool):
            super().__init__()
            self.enc = nn.ModuleList(nn.Linear(D, H) for _ in range(M))
            self.head = nn.Linear(H, K)
            self.gated = gated
            if gated:
                self.gate = nn.Sequential(
                    nn.Linear(2 * M, GATE_HIDDEN), nn.ReLU(),
                    nn.Linear(GATE_HIDDEN, M))
            else:
                self.theta = nn.Parameter(torch.zeros(M))  # uniform at init

        def weights(self, x):                    # x: (n, M, D)
            if not self.gated:
                return self.theta.softmax(0).expand(x.shape[0], M)
            stats = torch.cat([x.pow(2).mean(-1).sqrt(),
                               x.abs().amax(-1)], dim=1)   # (n, 2M)
            return self.gate(stats).softmax(-1)            # (n, M)

        def forward(self, x):
            w = self.weights(x)
            emb = torch.stack([self.enc[m](x[:, m]) for m in range(M)], 1)
            return self.head((w.unsqueeze(-1) * emb).sum(1)), w

    return _Pooled(gated=(arm == "gate"))


def _train_arm(arm: str, seed: int, books) -> dict:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = _build(arm)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Identical clean stream per seed for BOTH arms: same generator seed,
    # same draw order, same draw sizes.
    gen = torch.Generator().manual_seed(12345 + seed)
    finite, recent = True, []
    for _ in range(MAX_ITERS):
        x, y = _trial_batch(BATCH, gen, books, "clean")
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        recent = (recent + [
            (logits.argmax(1) == y).float().mean().item()])[-50:]
    return {"model": model.eval(), "n_params": n_params, "finite": finite,
            "train_fit": round(float(np.mean(recent)) if recent else 0.0, 4)}


def _eval_arm(model, seed: int, books, block: str) -> tuple:
    """Accuracy and mean vision weight on the seed's pre-generated block.
    Block generators are disjoint from the training stream by construction
    (different base offsets), identical across arms."""
    import torch
    base = {"clean": 777, "occluded": 888, "all_equal": 999}[block]
    gen = torch.Generator().manual_seed(base + seed)
    x, y = _trial_batch(N_EVAL, gen, books, block)
    with torch.no_grad():
        logits, w = model(x)
    acc = float((logits.argmax(1) == y).float().mean().item())
    return acc, float(w[:, 0].mean().item())


def _pipeline(seed: int) -> dict:
    if seed in _CACHE:
        return _CACHE[seed]
    books = _codebooks()
    _CACHE[seed] = {"fixed": _train_arm("fixed", seed, books),
                    "gate": _train_arm("gate", seed, books),
                    "books": books}
    return _CACHE[seed]


def _experiment(seed: int) -> dict:
    art = _pipeline(seed)
    fx, gt, books = art["fixed"], art["gate"], art["books"]
    m = {
        "fixed_params": fx["n_params"],
        "gate_params": gt["n_params"],
        "fixed_train_fit": fx["train_fit"],
        "gate_train_fit": gt["train_fit"],
    }
    if not (fx["finite"] and gt["finite"]):
        m["void_reason"] = "non-finite training loss — the rig, not the claim"
        return m
    m["fixed_clean_acc"], m["fixed_w_vision"] = (
        round(v, 4) for v in _eval_arm(fx["model"], seed, books, "clean"))
    m["gate_clean_acc"], _ = (
        round(v, 4) for v in _eval_arm(gt["model"], seed, books, "clean"))
    m["fixed_occ_acc"], _ = (
        round(v, 4) for v in _eval_arm(fx["model"], seed, books, "occluded"))
    m["gate_occ_acc"], m["gate_occ_w_vision"] = (
        round(v, 4) for v in _eval_arm(gt["model"], seed, books, "occluded"))
    m["occluded_cue_acc_vs_fixed_null"] = round(
        m["gate_occ_acc"] - m["fixed_occ_acc"], 4)
    # Worst-seed gate: run_spec aggregates by MEAN, so the deciding conjunct
    # is a per-seed boolean — its mean is 1.0 only if EVERY seed cleared.
    m["margin_ok"] = 1.0 if (
        m["occluded_cue_acc_vs_fixed_null"] >= MARGIN_MIN) else 0.0
    if m["fixed_clean_acc"] < FIT_MIN:
        m["void_reason"] = (
            f"the null never fit its own training distribution: "
            f"fixed_clean_acc {m['fixed_clean_acc']} vs FIT_MIN {FIT_MIN} "
            f"— the rig, not the claim")
    elif m["fixed_occ_acc"] < NULL_LIVE_MIN:
        m["void_reason"] = (
            f"the null is not a learner on the occluded block (T2.02 rule — "
            f"a dead null cannot arbitrate): fixed_occ_acc "
            f"{m['fixed_occ_acc']} vs NULL_LIVE_MIN {NULL_LIVE_MIN}")
    elif m["fixed_occ_acc"] > 1.0 - MARGIN_MIN:
        m["void_reason"] = (
            f"the null is saturated on the occluded block (unsaturated-null "
            f"rule): fixed_occ_acc {m['fixed_occ_acc']} leaves less headroom "
            f"than MARGIN_MIN {MARGIN_MIN}")
    return m


def _control(seed: int) -> dict:
    """The registry's two must-fail blocks, on the experiment's own frozen
    models: (1) unoccluded clean trials — the gate must NOT beat the fixed
    null (a win here is capacity, not selection); (2) cue in EVERY channel at
    equal fidelity — selection has nothing to select. Both margins must sit
    at or below CTRL_MARGIN_MAX on every seed."""
    art = _pipeline(seed)
    try:
        c = {}
        fx_clean, _ = _eval_arm(art["fixed"]["model"], seed, art["books"],
                                "clean")
        gt_clean, _ = _eval_arm(art["gate"]["model"], seed, art["books"],
                                "clean")
        fx_eq, _ = _eval_arm(art["fixed"]["model"], seed, art["books"],
                             "all_equal")
        gt_eq, _ = _eval_arm(art["gate"]["model"], seed, art["books"],
                             "all_equal")
        c["unocc_margin"] = round(gt_clean - fx_clean, 4)
        c["allch_margin"] = round(gt_eq - fx_eq, 4)
        c["ctrl_unocc_ok"] = 1.0 if (
            c["unocc_margin"] <= CTRL_MARGIN_MAX) else 0.0
        c["ctrl_allch_ok"] = 1.0 if (
            c["allch_margin"] <= CTRL_MARGIN_MAX) else 0.0
        return c
    finally:
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict):
    if "void_reason" in m:
        return Status.VOID
    # All three conjuncts are worst-seed gates: per-seed booleans aggregated
    # by mean read 1.0 only if every seed cleared (WM.01 incident, docstring).
    return (m["margin_ok"] == 1.0 and c["ctrl_unocc_ok"] == 1.0
            and c["ctrl_allch_ok"] == 1.0)


def _precheck(seed: int = 90) -> dict:
    """REGISTERABILITY ONLY (unsaturated-null rule): train and evaluate the
    NULL arm alone on the sizing seed. Never a claim seed, never the gate
    arm, writes nothing to the ledger. Readings recorded in the docstring's
    STATISTIC_BOUND block."""
    books = _codebooks()
    fx = _train_arm("fixed", seed, books)
    clean, w_v = _eval_arm(fx["model"], seed, books, "clean")
    occ, _ = _eval_arm(fx["model"], seed, books, "occluded")
    return {"null_clean_acc": round(clean, 4),
            "null_occ_acc": round(occ, 4),
            "null_w_vision": round(w_v, 4),
            "distance_to_bound": round(1.0 - occ, 4),
            "registerable": (clean >= FIT_MIN
                             and NULL_LIVE_MIN <= occ <= 1.0 - MARGIN_MIN)}


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["AT.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
