"""UB.10 — Fusion bakeoff: six arms, matched params, matched steps.

THE QUESTION. Does shared computation over tokens buy anything over
"concatenate and pray" at this scale? Six architectures, one certified
battery, matched everything that is not architecture. The survivor is the
trunk Jack ships; the rest are deleted (registry `kills`). A0 tying the best
arm everywhere is a REPORTABLE FAIL, not a reason to re-roll: it would mean
'one brain' buys nothing over bolt-on encoders at this scale and GOAL.md's
architecture claim must be restated (registry `falsified_by`).

THE BATTERY — inside UB.9's certificate, never outside it. All data comes
from `ub_9_heard_not_seen._data_for` (byte-for-byte: PG.7's audio features,
PG.6's eye discipline, the by-quad structure). Three readouts per episode,
trained jointly (equal-weight CE) and each doing a different job:

    slot        which slot fell.      I(audio;Y)=I(vision;Y)=0, joint = 1 bit
                (PG.7 certificate) — the PURE-SYNERGY task; the ranking task.
    large_slot  which slot holds the large sphere — decodable from VISION
                alone (UB.9's `vision_carries_bit` >= 0.90 gate).
    large_fell  whether the large sphere fell — decodable from AUDIO alone
                (UB.9's `audio_carries_bit` >= 0.90 gate).

The two marginal tasks are the LEARNING GATE (T2.02's principle): every arm
must reach MARGINAL_FLOOR on both, else the run is VOID — an arm that cannot
learn a unimodally-decodable bit under this optimiser/budget has not been
trained, and non-learners cannot arbitrate an architecture. They are also the
collapse detector: an arm that solves slot while failing a marginal has eaten
a sense.

MATCHED, CONCRETELY (registry: params +-5%, tokens, steps, data order):
  tokens     every arm consumes the IDENTICAL layout: 36 vision tokens (48x48
             pooled frame -> 6x6 grid of 8x8 patches, dim 64) + 14 audio
             tokens (one per PG.7 feature, dim 1). No arm sees more or fewer.
  params     A1 at D_BASE is the anchor; every other arm's width is scanned
             (d in 48..256 step 8, nhead=4 divides all) to the closest param
             count. Any arm outside +-PARAM_TOL of the anchor -> VOID.
  steps      EPOCHS x ceil(N_train/BATCH) optimisation steps for every arm.
             A3/A4 pay their aux-objective forward on top of the same step
             count — the spec matches optimisation steps, not FLOPs, and the
             aux pass is the objective's own cost, reported via n_params/d.
  data order one permutation sequence per seed, precomputed as numpy arrays
             and handed to every arm. Paired by construction; per-arm
             randomness (init, dropout draws) comes from (seed, arm) streams.

THE ARMS (registry notes, made runnable):
  A0  late concat: per-modality encoders (depth-1 transformer each, own CLS)
      -> one pooled vector per modality -> concat -> shared MLP head. The
      null. Its head is an MLP on purpose: UB.9 proved concat+MLP solves the
      XOR, so this null CAN win — a null that cannot win is a straw man.
  A1  shared token trunk: one depth-2 transformer over all 50 tokens +
      readout CLS, modality-ID + positional embeddings.
  A2  A1 + modality dropout: p=P_DROP per modality per sample (never both),
      dropped block replaced by a learned [MISSING-m] token broadcast.
  A3  A2 + cross-modal masked prediction (cross-signal, not joint): separate
      aux forward per step with ONE modality fully masked by [MASK-m];
      linear aux heads reconstruct the masked tokens' raw content (MSE).
      Reconstruction at a masked position can only route through the OTHER
      modality. loss += LAMBDA_AUX * mse.
  A4  A2 + contrastive alignment, state-proximity positives: pooled vision-
      token and audio-token outputs, projected to 64-d, InfoNCE at T_NCE
      where positives are SAME-QUAD pairs present in the batch (same nuisance
      draw = proximal state) — NOT episode identity, per the registry's
      false-negative warning. Samples with no in-batch positive contribute
      nothing that step. loss += LAMBDA_AUX * nce.
  A5  per-modality experts + learned router: A0's two experts, but a router
      (linear on the mean stem token) softmax-mixes the two expert vectors
      into ONE d-vector before the head. The credible non-trunk alternative;
      if it wins, 'one brain' is the wrong shape and we say so.

THE ENSEMBLE NULL, PER ARM (registry null_baseline). For every arm, two
unimodal variants of the SAME architecture are trained with the other
modality's raw input clamped to its train mean (uninformative but on-shape),
and their softmax outputs averaged. The ensemble remains the null the winner
must beat (PASS clause below) — but it is NOT a leak detector, and the
original VOID gate on it rested on a false theorem. AMENDED 2026-08-20,
before any registered seed ran; the original text said the ensemble "is
structurally incapable of synergy, so on slot it MUST sit at chance", and
gated VOID on ens_slot > NULL_GATE. That premise is mathematically wrong:
the averaged-softmax decision is sign(s(vision) + t(audio)), and an additive
composition of two unimodal scorers reaches up to 0.75 on a balanced XOR
(3 of 4 cells; dually 0.25) through miscalibrated confidences ALONE, with
each scorer's own argmax accuracy pinned at exactly 0.5. The seed-90 pilot
measured precisely this: ens_slot {A0 0.525, A1 0.653, A2 0.513, A3 0.484,
A4 0.250, A5 0.747} — the 0.75 ceiling and a below-chance 0.25 in one rig —
while EVERY unimodal variant of every arm read exactly 0.5000. (CORRECTED
per the 23rd audit, B2: twelve 0.5000 readings do NOT prove the fixture
clean — a constant predictor also reads exactly 0.5000 on a balanced test
set, and four of those twelve came from A2/A3, whose full arms provably
never trained. The fixture's cleanliness becomes established only when a
unimodal variant demonstrably learns its own sense's marginal task AND
still reads 0.5 on slot — the liveness observables below buy exactly
that.) The gate as written fired VOID on what the false theorem cannot
distinguish from a clean fixture, and it is also WEAKER than the correct
detector against an argmax-visible leak (a leaky channel averaged with a
chance channel dilutes below 0.60).
THE CORRECT LEAK DETECTOR, gating VOID from now on: the unimodal variants'
own slot accuracy. Any function of one modality must sit at chance on a
clean XOR (that one IS a theorem), so any (arm, seed, sense) unimodal slot
acc deviating from 0.5 by more than NULL_GATE - 0.5 (same 0.10 = ~3.5 sigma
at n_test = 320, now two-sided) -> VOID, investigate, do not celebrate.
This replacement is strictly stronger against any leak visible in a
unimodal argmax and removes the provably-false firing mode; it is BLIND to
a calibration-only leak (one that shifts confidences without moving any
argmax), which the old ensemble gate could see — the
winner-beats-own-ensemble PASS clause carries that residual exposure.
ens_slot stays recorded per arm. A dead unimodal variant would blind this
detector too, so each variant's own-sense marginal acc and loss curve are
recorded and gated (uni_marginal_ok / uni_learn_ok, added 2026-08-20 per
the 23rd audit B1, before any registered seed ran). (LESSONS.md: a null's
value under H0 is a theorem to prove, not a property to assert.)

CONTROL (registry, must fail): cross-episode SWAP per sense at eval — the
test set's audio (resp. vision) rows rolled by SWAP_ROLL so every episode
reads another quad's stream, everything else intact. For every arm, at least
one sense's swap must cost >= SWAP_HURT on at least one task. An arm
invariant to swapping EVERY sense has learned a marginal, not a
correspondence; its battery score is uninterpretable and the run is VOID.

PRE-REGISTERED VERDICT RULE (all gates exogenous — UB.9's 0.75 and 0.60 are
inherited, the rest are stated here before the first run):
  VOID if any: eye canary moved; any arm's params outside +-5% of anchor;
      any arm below MARGINAL_FLOOR on a marginal task on any seed; any arm's
      loss did not decrease (first-quarter mean -> last-quarter mean) on any
      seed; any unimodal variant's slot acc off chance by > NULL_GATE - 0.5
      (two-sided; the leak detector — amended 2026-08-20, see the ensemble
      section); any unimodal variant below MARGINAL_FLOOR on its own sense's
      marginal task, or whose loss did not decrease (leak-detector liveness
      — added 2026-08-20 per the 23rd audit B1, before any registered seed
      ran; strengthen-only); any arm swap-invariant on all senses on any
      seed; dropped-quads fraction > DROP_MAX.
  Then, winner = argmax over A1..A5 of median-across-seeds slot accuracy.
  PASS iff ALL of:
      winner slot acc > A0 slot acc on EVERY seed (paired, same data order);
      by-(seed,quad) cluster bootstrap 2.5th pct of the pooled per-episode
        paired difference (winner - A0, slot correctness) > 0;
      the argmax arm is the SAME arm on every seed (ranking stability, the
        hypothesis's own clause);
      winner median slot acc >= WINNER_GATE (0.75, UB.9's bar) AND winner
        beats its own unimodal ensemble on slot on every seed.
  Else FAIL. (A0 tying, an unstable ranking, and a sub-0.75 winner are all
  FAILs — different sentences, same verdict; the metrics distinguish them.)

STATS. 3 seeds resolve nothing unpaired (arXiv:2108.13264) — everything
comparative here is paired per-episode on identical data; the bootstrap
clusters by (seed, quad) because episodes within a quad share their nuisance
draw; medians are reported instead of means (IQM at n=3 is the median).

GPU. One submission for the whole spec (module cache, T2.01 pattern). Kaggle
first: the W33 hours are the perishable resource this spec spends. The job
renders on the VM under MUJOCO_GL=egl exactly as T3.01 did; ensure_gl
respects a preset EGL backend, so importing the UB.9 rig remotely is safe.
Science lives in THIS module; the JOB string only imports it (T0.16 lesson).
PILOT: seed 90, disjoint from the registered seeds, prints and records
nothing — it exists to catch rig faults before the registered spend (SM.02
lesson). Gates do not move on its account.

PILOT RECORD, seed 90, kernel jack-ladder-1787246533 (P100, 0.147 h,
2026-08-20). NOT CLEAN, two rig faults, no registered dispatch:
  1. A2 and A3 — the two arms whose only shared novelty is modality
     dropout — never trained: loss 1.60->1.56 and 1.90->1.82 across all
     150 epochs (A1, same trunk, went 1.43->0.00), slot/vslot exactly 0.5
     (constant predictor), afell 1.0. vslot 0.5 is below MARGINAL_FLOOR ->
     the registered run would VOID on the learning gate. A4 carries the
     same dropout and trained fine; its clean-forward NCE pass is the
     visible difference. Fingerprint says audio-only basin: optimiser
     recipe suspected before architecture (T3.01 curves-probe precedent).
  2. The ensemble VOID gate fired on a provably clean fixture — see the
     amended ensemble section above.
Healthy elsewhere: params all within +-2.9%, canary intact, dropped_frac 0,
swap controls fire correctly for every trained arm (vision swap zeroes the
vision-dependent tasks), A0/A1/A4/A5 all at 1.0/1.0/1.0.

PROBE RECORD, seed 90, kernel jack-ladder-1787249890 (P100, 0.229 h,
2026-08-20; artifact /data/ub10_recipe_probe.json). NEITHER RECIPE CLEAN —
the pre-registered both-fail branch fired: NO registered dispatch, no third
recipe, the one-diagnostic cap is spent, and the arm-design question routes
through PROGRESS.
  warmup (LR 1e-3, 10% linear warmup): A2 and A3 still at slot/vslot 0.5
      with flat loss (1.62->1.56, 1.89->1.82); A0/A1/A4/A5 all >= 0.9875.
      NOT CLEAN: A2:marginal, A3:marginal.
  lolr (LR 3e-4, no warmup): A3 FIXED (1.0/1.0/1.0) but A4 BROKEN (slot
      0.5531, vslot 0.5625, loss 3.27->2.37, and its audio swap now
      IMPROVES slot by 0.2156). A2 unchanged at 0.5/flat.
      NOT CLEAN: A2:marginal, A4:marginal.
  THE FINDING (23rd audit B3): lolr fixing A3 while breaking A4 is not a
  stuck arm — it is RECIPE SENSITIVITY of the six-arm design itself. No
  single uniform recipe in {1e-3+warmup, 3e-4} trains all six matched-param
  arms, so "matched training" as specced under-determines the comparison;
  A2 (modality dropout, no aux loss) learned its marginals under NO tested
  recipe at matched budget. That finding is the input the redesign needs,
  and it is worth more than a third LR would have been. Verdict gates did
  not move on the probe's account in any branch.

COVERS: one brain / unison (claim)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim inherits UB.9's certificate; every file it rests on hashes into
# impl_sha so drift goes stale loudly.
IMPL_DEPS = ["experiments/hns_scene.py",
             "experiments/tests/pg_7_hns_leakage.py",
             "experiments/tests/ub_9_heard_not_seen.py",
             "ContactAudio.py"]

SEEDS = [0, 1, 2]
PILOT_SEED = 90

ARMS = ("A0", "A1", "A2", "A3", "A4", "A5")
TRUNK_ARMS = ("A1", "A2", "A3", "A4", "A5")

# tokens: 48x48 pooled frame -> 6x6 patches of 8x8; 14 audio features
PATCH = 8
N_VTOK = 36
VTOK_DIM = PATCH * PATCH
N_ATOK = 14
N_TOK = N_VTOK + N_ATOK

D_BASE = 128                 # A1's width; the param anchor
DEPTH = 2                    # A1-A4 trunk depth; A0/A5 use 2 depth-1 experts
N_HEAD = 4
HEAD_HIDDEN = 2              # head MLP hidden = HEAD_HIDDEN * d
D_SCAN = tuple(range(48, 260, 4))
PARAM_TOL = 0.05

EPOCHS = 150
BATCH = 128
LR = 1e-3
WARMUP_FRAC = 0.0            # linear LR warmup over this fraction of steps
WEIGHT_DECAY = 1e-4
P_DROP = 0.25                # A2+ modality dropout
LAMBDA_AUX = 0.5             # A3 mse / A4 nce weight
T_NCE = 0.2
PROJ_DIM = 64

MARGINAL_FLOOR = 0.80        # learning gate on the unimodally-decodable tasks
NULL_GATE = 0.60             # UB.9's chance + ~3.5 sigma at n_test = 320
WINNER_GATE = 0.75           # UB.9's fused bar, inherited
SWAP_HURT = 0.10             # a sense's swap must cost at least this, somewhere
SWAP_ROLL = 37               # coprime with 4-per-quad blocks: crosses quads
DROP_MAX = 0.02              # UB.9's whole-quad drop budget
N_BOOT = 1000

TASKS = ("slot", "vslot", "afell")


# ── data: UB.9's certified rig, re-tokenised ─────────────────────────────

def _episode_tensors(seed: int):
    """Per-episode arrays from UB.9's cached generator: vision patches
    [N,36,64], audio features [N,14] (standardised by TRAIN stats), the three
    labels, quad ids, and the split. Train/test split is UB.9's own: the last
    N_TEST_QUADS quads are test, split by quad so no nuisance draw straddles
    the boundary."""
    from . import ub_9_heard_not_seen as ub9

    d = ub9._data_for(seed)
    quads = d["quad"]
    n = len(d["y"])
    frames = np.stack([d["frames"][(int(quads[i]), int(d["large_slot"][i]))]
                       for i in range(n)])                     # [N,48,48]
    side = frames.shape[1] // PATCH
    patches = (frames.reshape(n, side, PATCH, side, PATCH)
               .transpose(0, 1, 3, 2, 4).reshape(n, side * side, VTOK_DIM)
               .astype(np.float32))
    test_q = np.unique(quads)[-ub9.N_TEST_QUADS:]
    te = np.isin(quads, test_q)
    tr = ~te

    Xa = d["audio"].astype(np.float32)
    mu, sd = Xa[tr].mean(0), Xa[tr].std(0)
    sd[sd < 1e-9] = 1.0
    Xa = (Xa - mu) / sd

    return {
        "Xv": patches, "Xa": Xa,
        "y": {"slot": d["y"].astype(np.int64),
              "vslot": d["large_slot"].astype(np.int64),
              "afell": d["y_large_fell"].astype(np.int64)},
        "quad": quads, "tr": tr, "te": te,
        "dropped_frac": d["dropped_frac"],
        "vmean_tok": patches[tr].mean(0),      # [36,64] train-mean vision
        "amean_tok": Xa[tr].mean(0),           # [14]    train-mean audio (~0)
    }


# ── the arms ─────────────────────────────────────────────────────────────

def _build_arm(arm: str, d: int, seed_init: int):
    """One arm at width d. Structure is fixed per arm; only d varies for the
    param match. Returns an nn.Module whose forward(v, a, mode, gen) yields
    (task_logits, extras)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed_init)

    def enc_layer(dm):
        return nn.TransformerEncoderLayer(
            d_model=dm, nhead=N_HEAD, dim_feedforward=2 * dm,
            dropout=0.0, batch_first=True, norm_first=True)

    class Stem(nn.Module):
        """Identical token interface for every arm: patch/scalar projections,
        positional + modality-ID embeddings."""

        def __init__(self):
            super().__init__()
            self.pv = nn.Linear(VTOK_DIM, d)
            self.pa = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(N_TOK, d) * 0.02)
            self.mod = nn.Parameter(torch.randn(2, d) * 0.02)

        def forward(self, v, a):
            tv = self.pv(v) + self.pos[:N_VTOK] + self.mod[0]
            ta = (self.pa(a.unsqueeze(-1))
                  + self.pos[N_VTOK:] + self.mod[1])
            return tv, ta

    class Heads(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d_in, HEAD_HIDDEN * d),
                                     nn.ReLU())
            self.task = nn.ModuleDict(
                {t: nn.Linear(HEAD_HIDDEN * d, 2) for t in TASKS})

        def forward(self, z):
            h = self.mlp(z)
            return {t: self.task[t](h) for t in TASKS}

    class Trunk(nn.Module):
        """A1-A4: one shared transformer over all tokens + CLS."""

        def __init__(self):
            super().__init__()
            self.stem = Stem()
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.enc = nn.TransformerEncoder(enc_layer(d), DEPTH)
            self.heads = Heads(d)
            if arm in ("A2", "A3", "A4"):
                self.missing = nn.Parameter(torch.randn(2, d) * 0.02)
            if arm == "A3":
                self.mask_tok = nn.Parameter(torch.randn(2, d) * 0.02)
                self.aux_v = nn.Linear(d, VTOK_DIM)
                self.aux_a = nn.Linear(d, 1)
            if arm == "A4":
                self.proj_v = nn.Linear(d, PROJ_DIM)
                self.proj_a = nn.Linear(d, PROJ_DIM)

        def _run(self, tv, ta):
            B = tv.shape[0]
            x = torch.cat([self.cls.expand(B, 1, d), tv, ta], dim=1)
            out = self.enc(x)
            return out[:, 0], out[:, 1:1 + N_VTOK], out[:, 1 + N_VTOK:]

        def forward(self, v, a, train_mode=False, gen=None):
            tv, ta = self.stem(v, a)
            if train_mode and arm in ("A2", "A3", "A4"):
                B = tv.shape[0]
                # one uniform per sample: drop vision below P_DROP, audio
                # above 1-P_DROP — exact rate each, never both at once.
                u = torch.rand(B, generator=gen).to(tv.device)
                drop_v = u < P_DROP
                drop_a = u > 1.0 - P_DROP
                tv = torch.where(drop_v[:, None, None],
                                 (self.missing[0] + self.stem.pos[:N_VTOK]
                                  ).expand(B, N_VTOK, d), tv)
                ta = torch.where(drop_a[:, None, None],
                                 (self.missing[1] + self.stem.pos[N_VTOK:]
                                  ).expand(B, N_ATOK, d), ta)
            z, ov, oa = self._run(tv, ta)
            return self.heads(z), (z, ov, oa, tv, ta)

        def aux_masked(self, v, a, which: int):
            """A3: modality `which` (0=vision, 1=audio) fully masked; linear
            heads reconstruct its raw content from trunk outputs."""
            tv, ta = self.stem(v, a)
            B = tv.shape[0]
            if which == 0:
                tv = (self.mask_tok[0] + self.stem.pos[:N_VTOK]
                      ).expand(B, N_VTOK, d)
            else:
                ta = (self.mask_tok[1] + self.stem.pos[N_VTOK:]
                      ).expand(B, N_ATOK, d)
            _, ov, oa = self._run(tv, ta)
            if which == 0:
                return ((self.aux_v(ov) - v) ** 2).mean()
            return ((self.aux_a(oa).squeeze(-1) - a) ** 2).mean()

    class TwoExpert(nn.Module):
        """A0 (concat head) and A5 (routed mix): per-modality depth-1
        encoders with private CLS tokens."""

        def __init__(self):
            super().__init__()
            self.stem = Stem()
            self.cls_v = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.cls_a = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.enc_v = nn.TransformerEncoder(enc_layer(d), 1)
            self.enc_a = nn.TransformerEncoder(enc_layer(d), 1)
            if arm == "A5":
                self.router = nn.Linear(d, 2)
                self.heads = Heads(d)
            else:
                self.heads = Heads(2 * d)

        def forward(self, v, a, train_mode=False, gen=None):
            tv, ta = self.stem(v, a)
            B = tv.shape[0]
            zv = self.enc_v(torch.cat([self.cls_v.expand(B, 1, d), tv], 1))[:, 0]
            za = self.enc_a(torch.cat([self.cls_a.expand(B, 1, d), ta], 1))[:, 0]
            if arm == "A5":
                w = torch.softmax(
                    self.router(torch.cat([tv, ta], 1).mean(1)), dim=-1)
                z = w[:, :1] * zv + w[:, 1:] * za
            else:
                z = torch.cat([zv, za], dim=-1)
            return self.heads(z), (z, None, None, tv, ta)

    return (TwoExpert() if arm in ("A0", "A5") else Trunk())


_WIDTH_CACHE: dict = {}


def _matched_width(arm: str) -> tuple:
    """Scan widths for the closest param count to A1@D_BASE. Deterministic,
    cheap (CPU instantiation only), cached per process."""
    if arm not in _WIDTH_CACHE:
        target = _WIDTH_CACHE.get("_target")
        if target is None:
            target = sum(p.numel()
                         for p in _build_arm("A1", D_BASE, 0).parameters())
            _WIDTH_CACHE["_target"] = target
        if arm == "A1":
            _WIDTH_CACHE[arm] = (D_BASE, target)
        else:
            best = None
            for dd in D_SCAN:
                n = sum(p.numel()
                        for p in _build_arm(arm, dd, 0).parameters())
                if best is None or abs(n - target) < abs(best[1] - target):
                    best = (dd, n)
            _WIDTH_CACHE[arm] = best
    return _WIDTH_CACHE[arm]


# ── training: matched steps, matched data order ──────────────────────────

def _train_arm(arm: str, seed: int, data: dict, perms: list,
               device: str, unimodal: str | None = None) -> dict:
    """Train one arm (or its unimodal variant) and evaluate the battery.
    `unimodal='vision'|'audio'` clamps the OTHER modality's raw input to its
    train mean everywhere (train and eval) — the ensemble construction."""
    import torch

    torch.set_num_threads(2)
    d, n_params = _matched_width(arm)
    arm_idx = ARMS.index(arm)
    variant = {"vision": 1, "audio": 2, None: 0}[unimodal]
    net = _build_arm(arm, d, seed * 1009 + arm_idx * 101 + variant * 7)
    net = net.to(device).train()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed * 60013 + arm_idx * 977 + variant * 31)

    tr = np.where(data["tr"])[0]
    te = np.where(data["te"])[0]

    def tensors(idx):
        v = data["Xv"][idx].copy()
        a = data["Xa"][idx].copy()
        if unimodal == "vision":
            a[:] = data["amean_tok"]
        elif unimodal == "audio":
            v[:] = data["vmean_tok"]
        return (torch.tensor(v, device=device),
                torch.tensor(a, device=device))

    Vtr, Atr = tensors(tr)
    ytr = {t: torch.tensor(data["y"][t][tr], device=device) for t in TASKS}
    quad_tr = torch.tensor(data["quad"][tr], device=device)

    opt = torch.optim.Adam(net.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    ce = torch.nn.CrossEntropyLoss()
    total_steps = len(perms) * int(np.ceil(len(perms[0]) / BATCH))
    warm_steps = int(round(WARMUP_FRAC * total_steps))
    step_i = 0
    losses = []
    for perm in perms:
        for i in range(0, len(perm), BATCH):
            if warm_steps:
                lr_now = LR * min(1.0, (step_i + 1) / warm_steps)
                for g in opt.param_groups:
                    g["lr"] = lr_now
            step_i += 1
            b = torch.tensor(perm[i:i + BATCH], device=device)
            v, a = Vtr[b], Atr[b]
            logits, (_, _, _, _, _) = net(v, a, train_mode=True, gen=gen)
            loss = sum(ce(logits[t], ytr[t][b]) for t in TASKS)
            if arm == "A3" and unimodal is None:
                which = int(torch.rand(1, generator=gen).item() < 0.5)
                loss = loss + LAMBDA_AUX * net.aux_masked(v, a, which)
            if arm == "A4" and unimodal is None:
                _, (_, ov, oa, _, _) = net(v, a)
                pv = torch.nn.functional.normalize(net.proj_v(ov.mean(1)), dim=-1)
                pa = torch.nn.functional.normalize(net.proj_a(oa.mean(1)), dim=-1)
                sim = pv @ pa.T / T_NCE
                q = quad_tr[b]
                pos = (q[:, None] == q[None, :])
                has_pos = pos.any(1)
                if bool(has_pos.any()):
                    lsm = torch.log_softmax(sim, dim=1)
                    nce = -(lsm[pos].sum() / pos.sum())
                    loss = loss + LAMBDA_AUX * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))

    net.eval()

    def probs(V, A):
        with torch.no_grad():
            out = {}
            for i in range(0, len(V), 512):
                logits, _ = net(V[i:i + 512], A[i:i + 512])
                for t in TASKS:
                    out.setdefault(t, []).append(
                        torch.softmax(logits[t], -1).cpu().numpy())
            return {t: np.concatenate(out[t]) for t in TASKS}

    Vte, Ate = tensors(te)
    p = probs(Vte, Ate)
    yte = {t: data["y"][t][te] for t in TASKS}
    acc = {t: float((p[t].argmax(1) == yte[t]).mean()) for t in TASKS}

    row = {
        "d": d, "n_params": int(n_params),
        "acc": {t: round(acc[t], 4) for t in TASKS},
        "loss_first": round(float(np.mean(losses[:max(1, len(losses) // 4)])), 4),
        "loss_last": round(float(np.mean(losses[-max(1, len(losses) // 4):])), 4),
        "probs_slot": p["slot"][:, 1].round(4).tolist(),
        "slot_correct": (p["slot"].argmax(1) == yte["slot"]
                         ).astype(int).tolist(),
    }
    if unimodal is None:
        # cross-episode SWAP per sense: roll the sense's raw rows so every
        # episode reads another quad's stream; everything else intact.
        swaps = {}
        for sense in ("vision", "audio"):
            Vs = Vte.roll(SWAP_ROLL, 0) if sense == "vision" else Vte
            As = Ate.roll(SWAP_ROLL, 0) if sense == "audio" else Ate
            ps = probs(Vs, As)
            swaps[sense] = {t: round(acc[t] - float(
                (ps[t].argmax(1) == yte[t]).mean()), 4) for t in TASKS}
        row["swap_drop"] = swaps
    return row


def _run_seed(seed: int, device: str) -> dict:
    """Everything for one seed: data, 6 arms, 12 unimodal variants, swaps."""
    from . import ub_9_heard_not_seen as ub9

    data = _episode_tensors(seed)
    eye = ub9.get_eye()
    n_tr = int(data["tr"].sum())
    order_rng = np.random.RandomState(seed * 77 + 3)
    perms = [order_rng.permutation(n_tr) for _ in range(EPOCHS)]

    arms_out = {}
    for arm in ARMS:
        full = _train_arm(arm, seed, data, perms, device)
        uni_v = _train_arm(arm, seed, data, perms, device, unimodal="vision")
        uni_a = _train_arm(arm, seed, data, perms, device, unimodal="audio")
        pv = np.asarray(full.pop("probs_slot"))
        pv_v = np.asarray(uni_v.pop("probs_slot"))
        pv_a = np.asarray(uni_a.pop("probs_slot"))
        te_y = data["y"]["slot"][data["te"]]
        ens = ((pv_v + pv_a) / 2.0 > 0.5).astype(int)
        full["ens_slot"] = round(float((ens == te_y).mean()), 4)
        full["uni_vision_slot"] = uni_v["acc"]["slot"]
        full["uni_audio_slot"] = uni_a["acc"]["slot"]
        # Liveness of the leak gate's own instruments (23rd audit, B1): a
        # constant predictor also reads exactly 0.5 on slot, so each unimodal
        # variant must demonstrably learn its own sense's marginal task for
        # its slot reading to mean anything.
        full["uni_vision_vslot"] = uni_v["acc"]["vslot"]
        full["uni_audio_afell"] = uni_a["acc"]["afell"]
        full["uni_vision_loss"] = [uni_v["loss_first"], uni_v["loss_last"]]
        full["uni_audio_loss"] = [uni_a["loss_first"], uni_a["loss_last"]]
        _ = pv  # per-episode slot probabilities live in slot_correct
        arms_out[arm] = full
        print("ARM_DONE", seed, arm, json.dumps(full["acc"]), flush=True)

    return {
        "seed": seed,
        "canary_ok": bool(eye.canary() == eye._canary_ref),
        "dropped_frac": round(float(data["dropped_frac"]), 4),
        "quads_test": data["quad"][data["te"]].astype(int).tolist(),
        "arms": arms_out,
    }


def remote_run(seeds: list) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "widths": {a: list(_matched_width(a)) for a in ARMS},
           "seeds": []}
    for seed in seeds:
        row = _run_seed(seed, device)
        out["seeds"].append(row)
        print("SEED_DONE", seed, flush=True)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.tests.ub_10_fusion_bakeoff import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "ub10.json"), "w"))
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    res = submit(job, prefer="kaggle",
                 est_hours=round(0.20 + 0.35 * len(seeds), 2),
                 timeout_s=2400 + 1500 * len(seeds),
                 fetch=["ub10.json"])
    if not res.ok:
        raise RuntimeError(f"UB.10 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["ub10.json"]).read_text())
    out["backend"] = res.backend
    return out


def _seed_row_clean(s: dict) -> tuple:
    """The registered run's VOID checklist applied to one seed row. Returns
    (clean, reasons) so a pilot/probe read is one look, not a spelunk."""
    reasons = []
    if not s["canary_ok"]:
        reasons.append("canary")
    if s["dropped_frac"] > DROP_MAX:
        reasons.append("dropped_frac")
    for a in ARMS:
        r = s["arms"][a]
        if any(r["acc"][t] < MARGINAL_FLOOR for t in ("vslot", "afell")):
            reasons.append(f"{a}:marginal")
        if not r["loss_last"] < r["loss_first"]:
            reasons.append(f"{a}:loss")
        if max(abs(r["uni_vision_slot"] - 0.5),
               abs(r["uni_audio_slot"] - 0.5)) > NULL_GATE - 0.5:
            reasons.append(f"{a}:uni_leak")
        if (r["uni_vision_vslot"] < MARGINAL_FLOOR
                or r["uni_audio_afell"] < MARGINAL_FLOOR):
            reasons.append(f"{a}:uni_marginal")
        if not (r["uni_vision_loss"][1] < r["uni_vision_loss"][0]
                and r["uni_audio_loss"][1] < r["uni_audio_loss"][0]):
            reasons.append(f"{a}:uni_loss")
        if max(r["swap_drop"][sn][t]
               for sn in ("vision", "audio") for t in TASKS) < SWAP_HURT:
            reasons.append(f"{a}:swap")
    return (not reasons, reasons)


def _print_seed_row(s: dict):
    for a in ARMS:
        r = s["arms"][a]
        sw = max(r["swap_drop"][sn][t]
                 for sn in ("vision", "audio") for t in TASKS)
        print(f"  {a} acc={r['acc']} loss={r['loss_first']}->{r['loss_last']}"
              f" ens={r['ens_slot']} uni_v={r['uni_vision_slot']}"
              f" uni_a={r['uni_audio_slot']}"
              f" uni_vslot={r['uni_vision_vslot']}"
              f" uni_afell={r['uni_audio_afell']} max_swap_drop={sw}")
    clean, reasons = _seed_row_clean(s)
    print("  CLEAN" if clean else f"  NOT CLEAN: {reasons}")


def pilot():
    """Seed-90 pilot at production scale, disjoint from the registered seeds.
    Prints, records nothing; numbers go to the docstring/journal by hand. The
    gates are exogenous and do NOT move on its account (SM.02 lesson)."""
    out = _submit([PILOT_SEED])
    print(json.dumps({"widths": out["widths"]}, indent=1))
    for s in out["seeds"]:
        _print_seed_row(s)
    return out


RECIPES = (("warmup", 1e-3, 0.10), ("lolr", 3e-4, 0.0))


def remote_recipe_probe() -> dict:
    """Runs REMOTELY. The A2/A3 repair probe, pre-registered 2026-08-20
    BEFORE dispatch (T3.01 curves-probe pattern; this is UB.10's ONE
    diagnostic under the SM.02/B5 cap).

    FAULT: pilot seed 90 — A2/A3 (the modality-dropout trunk arms) sat at
    constant-predictor chance on slot/vslot with loss flat for 150 epochs,
    while A1 (same trunk, no dropout) and A4 (same dropout + NCE aux) both
    trained to 1.0. Suspect the optimiser recipe (norm_first transformer at
    Adam 1e-3, no warmup -> audio-only basin), not the architecture, per the
    T3.01 precedent.

    ARMS OF THE PROBE, uniform over all six bakeoff arms, full production
    scale, seed 90 only: (1) 'warmup' = LR 1e-3 with linear warmup over the
    first 10% of steps; (2) 'lolr' = LR 3e-4, no warmup. Same EPOCHS, same
    per-seed data order, same param match; each recipe runs the complete
    _run_seed (18 trainings) so the swap/ensemble/leak checks are read under
    the candidate recipe too.

    READINGS (observables, not premises): per recipe, the _seed_row_clean
    checklist — every arm >= 0.80 on both marginals, every loss decreasing,
    no unimodal slot acc off 0.5 by > 0.10, every arm's best swap drop
    >= 0.10, canary intact, dropped_frac <= 0.02.

    DECISION RULE: adopt the FIRST clean recipe in the order (warmup, lolr)
    — warmup preferred because it is the minimal deviation (identical LR for
    90% of steps, and the healthy arms provably train at 1e-3). Set LR /
    WARMUP_FRAC to the adopted values in a follow-up commit and dispatch the
    registered run via scripts/dispatch.sh UB.10; the adopted recipe's probe
    row IS the pilot (seed 90, disjoint). If NEITHER recipe is clean: no
    registered dispatch, no third recipe — the cap fires, record that A2/A3
    as specced cannot learn their marginals at matched budget under either
    standard recipe, and route the arm-design question through PROGRESS.
    Verdict gates do not move on this probe's account in any branch."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda"
                   else "cpu"),
           "widths": {a: list(_matched_width(a)) for a in ARMS},
           "recipes": {}}
    for name, lr, wf in RECIPES:
        globals()["LR"] = lr
        globals()["WARMUP_FRAC"] = wf
        out["recipes"][name] = _run_seed(PILOT_SEED, device)
        print("RECIPE_DONE", name, flush=True)
    return out


JOB_PROBE = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.tests.ub_10_fusion_bakeoff import remote_recipe_probe
out = remote_recipe_probe()
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "ub10_probe.json"),
                    "w"))
print("DONE", out["gpu"], flush=True)
'''


def recipe_probe():
    """Local side of the probe: submit, save the artifact where the next
    iteration expects it, print the decision-rule read per recipe."""
    job = build_job(JOB_PROBE)
    res = submit(job, prefer="kaggle", est_hours=0.5, timeout_s=4500,
                 fetch=["ub10_probe.json"])
    if not res.ok:
        raise RuntimeError(f"UB.10 recipe probe failed on {res.backend}: "
                           f"{res.message}")
    out = json.loads(Path(res.artifacts["ub10_probe.json"]).read_text())
    Path("/data/ub10_recipe_probe.json").write_text(json.dumps(out))
    for name, _, _ in RECIPES:
        print(f"RECIPE {name}:")
        _print_seed_row(out["recipes"][name])
    return out


# ── aggregation + verdict ────────────────────────────────────────────────

def _paired_boot_lo(rows: list, winner: str) -> float:
    """2.5th pct of the pooled per-episode paired difference in slot
    correctness (winner - A0), cluster-bootstrapped by (seed, quad)."""
    diffs, clusters = [], []
    for s in rows:
        cw = np.asarray(s["arms"][winner]["slot_correct"], dtype=float)
        c0 = np.asarray(s["arms"]["A0"]["slot_correct"], dtype=float)
        q = np.asarray(s["quads_test"])
        diffs.append(cw - c0)
        clusters.append(q + 100_000 * s["seed"])
    diff = np.concatenate(diffs)
    clu = np.concatenate(clusters)
    uc = np.unique(clu)
    by_c = {c: diff[clu == c] for c in uc}
    rng = np.random.RandomState(20260820)
    means = []
    for _ in range(N_BOOT):
        pick = rng.choice(uc, size=len(uc), replace=True)
        means.append(float(np.concatenate([by_c[c] for c in pick]).mean()))
    return float(np.percentile(means, 2.5))


def _aggregate() -> dict:
    rows = _CACHE["seeds"]
    target = _CACHE["widths"]["A1"][1]
    params = {a: _CACHE["widths"][a][1] for a in ARMS}
    params_ok = all(abs(params[a] - target) / target <= PARAM_TOL
                    for a in ARMS)

    slot = {a: [s["arms"][a]["acc"]["slot"] for s in rows] for a in ARMS}
    med = {a: float(np.median(slot[a])) for a in ARMS}
    winner = max(TRUNK_ARMS, key=lambda a: med[a])
    top1 = [max(TRUNK_ARMS, key=lambda a: s["arms"][a]["acc"]["slot"])
            for s in rows]

    marginal_ok = all(
        s["arms"][a]["acc"][t] >= MARGINAL_FLOOR
        for s in rows for a in ARMS for t in ("vslot", "afell"))
    learn_ok = all(s["arms"][a]["loss_last"] < s["arms"][a]["loss_first"]
                   for s in rows for a in ARMS)
    ens_max = max(s["arms"][a]["ens_slot"] for s in rows for a in ARMS)
    uni_dev_max = max(
        abs(s["arms"][a][k] - 0.5)
        for s in rows for a in ARMS
        for k in ("uni_vision_slot", "uni_audio_slot"))
    uni_marginal_ok = all(
        s["arms"][a]["uni_vision_vslot"] >= MARGINAL_FLOOR
        and s["arms"][a]["uni_audio_afell"] >= MARGINAL_FLOOR
        for s in rows for a in ARMS)
    uni_learn_ok = all(
        s["arms"][a][k][1] < s["arms"][a][k][0]
        for s in rows for a in ARMS
        for k in ("uni_vision_loss", "uni_audio_loss"))
    swap_ok = all(
        max(s["arms"][a]["swap_drop"][sense][t]
            for sense in ("vision", "audio") for t in TASKS) >= SWAP_HURT
        for s in rows for a in ARMS)

    beats_a0_all = all(s["arms"][winner]["acc"]["slot"]
                       > s["arms"]["A0"]["acc"]["slot"] for s in rows)
    beats_own_ens = all(s["arms"][winner]["acc"]["slot"]
                        > s["arms"][winner]["ens_slot"] for s in rows)
    boot_lo = _paired_boot_lo(rows, winner)

    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "widths": {a: _CACHE["widths"][a][0] for a in ARMS},
        "n_params": params,
        "params_ok": float(params_ok),
        "slot_per_arm_per_seed": slot,
        "slot_median": {a: round(med[a], 4) for a in ARMS},
        "arm_ranking_x_synergy_gap": round(med[winner] - med["A0"], 4),
        "winner": float(ARMS.index(winner)),
        "top1_stable": float(len(set(top1)) == 1),
        "winner_beats_a0_all_seeds": float(beats_a0_all),
        "winner_beats_own_ensemble": float(beats_own_ens),
        "paired_boot_lo": round(boot_lo, 4),
        "winner_slot_median": round(med[winner], 4),
        "a0_slot_median": round(med["A0"], 4),
        "ens_slot_max": round(ens_max, 4),
        "uni_slot_dev_max": round(uni_dev_max, 4),
        "uni_marginal_ok": float(uni_marginal_ok),
        "uni_learn_ok": float(uni_learn_ok),
        "marginal_ok": float(marginal_ok),
        "learn_ok": float(learn_ok),
        "swap_ok_all_arms": float(swap_ok),
        "canary_ok_all": float(all(s["canary_ok"] for s in rows)),
        "dropped_frac_max": max(s["dropped_frac"] for s in rows),
        "synergy_gap_per_arm": {
            a: round(float(np.median(
                [s["arms"][a]["acc"]["slot"] - s["arms"][a]["ens_slot"]
                 for s in rows])), 4) for a in ARMS},
    }


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    return _aggregate()


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    worst = min(
        max(s["arms"][a]["swap_drop"][sense][t]
            for sense in ("vision", "audio") for t in TASKS)
        for s in rows for a in ARMS)
    return {
        "ctrl_min_over_arms_of_max_swap_drop": round(worst, 4),
        "ctrl_swap_ok": float(worst >= SWAP_HURT),
        "ctrl_swap_drops": {
            a: {sense: s["arms"][a]["swap_drop"][sense]
                for sense in ("vision", "audio")}
            for s in rows[:1] for a in ARMS},
    }


def _check(m: dict, c: dict):
    # Rig first — an invalid run is VOID, not evidence about architecture.
    if m["canary_ok_all"] != 1.0:
        return Status.VOID          # the eye degraded mid-run
    if m["dropped_frac_max"] > DROP_MAX:
        return Status.VOID          # class balance no longer by construction
    if m["params_ok"] != 1.0:
        return Status.VOID          # the match failed; ranking measures size
    if m["marginal_ok"] != 1.0 or m["learn_ok"] != 1.0:
        return Status.VOID          # a non-learner cannot arbitrate (T2.02)
    if m["uni_marginal_ok"] != 1.0 or m["uni_learn_ok"] != 1.0:
        return Status.VOID          # a dead unimodal variant blinds the
                                    # leak gate below: a constant predictor
                                    # reads exactly 0.5 whether the fixture
                                    # is clean or not (23rd audit, B1)
    if m["uni_slot_dev_max"] > NULL_GATE - 0.5:
        return Status.VOID          # a unimodal model off chance = fixture
                                    # leak (amended 2026-08-20: the old
                                    # ens_slot gate's premise was false —
                                    # additive ensembles reach 0.75 on clean
                                    # XOR; see docstring. ens stays reported.)
    if c["ctrl_swap_ok"] != 1.0:
        return Status.VOID          # a swap-invariant arm is uninterpretable
    # The claim.
    return (m["winner_beats_a0_all_seeds"] == 1.0
            and m["paired_boot_lo"] > 0.0
            and m["top1_stable"] == 1.0
            and m["winner_slot_median"] >= WINNER_GATE
            and m["winner_beats_own_ensemble"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["UB.10"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, CPU, minutes: every arm, every code path, at toy scale.
        # Patches the UB.9 rig's quad count DOWN (never the gates) so data
        # generation fits one free core; production shapes are preserved.
        from . import ub_9_heard_not_seen as ub9
        ub9.N_QUADS, ub9.N_TEST_QUADS = 24, 6
        globals()["EPOCHS"] = 2
        globals()["D_BASE"] = 48
        globals()["D_SCAN"] = (32, 40, 48, 56, 64)
        globals()["BATCH"] = 32
        out = remote_run([PILOT_SEED])
        s = out["seeds"][0]
        assert s["canary_ok"], "eye canary moved in smoke"
        for a in ARMS:
            r = s["arms"][a]
            assert "swap_drop" in r and "ens_slot" in r, a
            assert "uni_vision_vslot" in r and "uni_audio_afell" in r, a
            assert "uni_vision_loss" in r and "uni_audio_loss" in r, a
            assert len(r["slot_correct"]) == 24, "test-episode count wrong"
        tgt = out["widths"]["A1"][1]
        spread = {a: round(abs(w[1] - tgt) / tgt, 3)
                  for a, w in out["widths"].items()}
        print("param spread vs anchor:", spread)
        print(json.dumps({a: s["arms"][a]["acc"] for a in ARMS}, indent=1))
        print("SMOKE OK")
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "recipe_probe":
        recipe_probe()
    else:
        run()
