"""WM.01 — the working-memory store earns its slots, or it is decoration.

THE QUESTION (anatomy audit 2026-10-25, the Working-memory seat's ring). ME.8
certified that the shipped GRU buffer (`WorkingMemory.py`) SURVIVES a restart;
it never certified that the buffer is a good store — no arm ever had to beat
it. This spec races the mechanism: the shipped recurrent buffer against an arm
with NO separate working memory — attention over the episodic record alone —
at matched parameters and matched training steps.

THE VENUE, AND THE MEASUREMENT THAT CHOSE IT (both pre-check readings
recorded; the venue moved BEFORE the gate was registered, on the NULL's own
reading, never after any claim-arm number existed):

  v1 — last-write LOOKUP (writes are absolute cues, answer = the latest one).
  Pre-check, sizing seed 90, null arm only: nobuffer_acc = 0.9941 — the null
  AT the ceiling, distance to bound 0.0059 << the 0.10 margin. Under the
  UNSATURATED-NULL RULE (2026-09-20) that gate is not registerable, and the
  reading says something real: any venue whose answer is a function of ONE
  stored event is solved exactly by one linear attention over the record —
  it is DIARY territory. Jack already has a diary and ME.1 certifies cued
  lookup; a venue the diary-reader solves is not a working-memory venue.

  v2 — the RELATIVE-UPDATE REGISTER (this spec's venue). What a working
  memory does that a record-reader cannot is COMPOSITION: hold a state that
  events update as a function of what was held (position updated by moves,
  counts updated by events — GOAL.md's "what Jack is holding in mind
  mid-episode"). T=40 steps; step 0 writes an absolute init cue c0 (one-hot,
  8 classes); DEPTH-1 further writes (depth uniform in {1..5}) at distinct
  random steps in [4, T-6] each carry a DELTA d_i (one-hot, same channels —
  step 0 is the only absolute write, declared); the answer is
  (c0 + sum(d_i)) mod 8. Every step carries Gaussian noise on 4 separate
  channels and a time channel t/(T-1); the final step carries a query flag.
  Chance = 1/8. The depth-1 tier (no deltas: pure hold across 39 noise
  steps, ME.8's own regime) is the null's LIVE tier — it keeps the
  no-buffer arm a learner, so the race is an arbitration and not a
  structural blinding.
  Pre-check, sizing seed 90, null arm only: nobuffer_acc = 0.3457 —
  above LEARN_MIN 0.25, distance to bound 0.6543 = 6.5x the required
  margin. Registerable on both counts, and no cushion constant was tuned
  to get there: the depth distribution is the venue, not a knob.

EXPRESSIVITY, DECLARED HONESTLY (it is the capability under test, not a rig
accident): the attention arm expresses the depth-1 tier EXACTLY (linear
write-vs-noise separation + time-channel recency, softmax-sharpened) and
CANNOT express deep composition — its readout is a linear head over a convex
aggregate of per-event embeddings, and (c0 + sum d) mod 8 is not linear in
any event aggregate. That asymmetry is the point: composition over time IS
what holding state buys, and a null that could compose would be a working
memory. The GRU arm must LEARN the gated mod-8 register update — recurrence
can express it and nothing guarantees it trains; a buffer that cannot learn
to compose loses honestly, with its train fit on the row for diagnosis.

ARMS, matched to <= PARAM_TOL relative parameter difference (asserted, VOID
on breach):
  buffer     the shipped `WorkingMemory(OBS_DIM, C, HIDDEN=32)` — GRUCell +
             linear head, 4,872 params. IMPL_DEPS covers it: this spec is a
             claim about that file's mechanism.
  no_buffer  `_EpisodicAttention(d=125)` — the episodic store is the raw
             per-step event sequence (exactly what the diary records: the
             observation plus when it happened); one learned query attends
             over it at readout; linear key/value maps + linear head, 4,883
             params (0.23% over the buffer arm). No recurrent state anywhere.
             (The shipped `EpisodicMemory` class itself is a token log for
             attributed recall; the arm attends over the same INFORMATION —
             the raw event record — in tensor form, which is the mechanism
             the registry's null names.)

Both arms train on IDENTICAL batch streams (one seeded generator per arm,
same seed, same draw order), identical optimiser (Adam, LR), identical
MAX_ITERS and early-stop rule, and are evaluated on the IDENTICAL
pre-generated holdout episodes.

THE DECISION RULE, pre-registered before any claim-arm number existed (the
three bars below were frozen before the v1 pre-check ran and none has moved;
only the VENUE moved, on the null's reading, as the unsaturated-null rule
instructs — statistic and bars unchanged):

  VOID  param match breached; or non-finite training loss; or the NULL arm's
        holdout accuracy < LEARN_MIN — a dead null cannot arbitrate (T2.02
        rule). A buffer below LEARN_MIN against a live null is NOT void: it
        is the claim losing, and the margin records it.
  PASS  on every seed: margin = buffer_acc - nobuffer_acc >= MARGIN_MIN,
        AND the shuffled-buffer control collapses (shuffled_acc <=
        CONTROL_MAX).
  FAIL  otherwise — including a null at the ceiling on claim seeds (the
        registry falsifier "matches or beats" fires) and including an
        inverted margin. Either way the registry's own sentence applies: the
        working-memory parameters are decoration and the finding routes to
        the Review; this spec deletes nothing itself.

HOW "on every seed" IS ENFORCED, and the incident that forced the mechanics
to be written down (2026-10-25, same sitting, before any verdict was
reported): `run_spec` aggregates per-seed metrics by MEAN and calls `_check`
ONCE on the aggregate, so a bare `margin >= MARGIN_MIN` conjunct reads the
MEAN margin — the exact defect the open row `aggregate-hides-worst-seed`
names. Attempt 1 recorded PASS that way: mean margin 0.1927 over per-seed
margins {0.4707, 0.0800, 0.0273} — two of three seeds BELOW the bar this
docstring had pre-registered. The repair is the worst-seed gate the ruling
on that row mandates, encoded as per-seed BOOLEANS whose mean is 1.0 only if
every seed cleared: `margin_ok` and (control-side) `collapsed_ok`, both
gated `== 1.0`. Bars unmoved; this can only turn a PASS into a FAIL, never
the reverse; attempt 1 is superseded by the first run under the strengthened
gate, and the per-seed numbers above reproduce the recorded aggregates
exactly (the run is a pure function of code and seed).

STATISTIC_BOUND (the unsaturated-null rule): statistic = accuracy margin,
bound = 1.0. Null's measured distance from the bound at the registered venue:
0.6543 (seed 90, pre-check above) against a required margin of 0.10. The
saturated v1 reading (0.0059) is retained above as the trail that forced the
venue move.

THE CONTROL (must fail), from the registry verbatim in substance: SHUFFLE the
buffer contents between steps — a fresh random permutation of the hidden
state's entries after every step, size and parameter count untouched — and
accuracy must COLLAPSE (<= CONTROL_MAX). If a shuffled buffer scores like an
ordered one, the slots are not holding state across the step and ME.8's PASS
was bought by something other than memory. CONTROL-ALIVE (LESSONS: an
at-chance control must carry proof its instrument was alive): the presence is
demonstrated at the same operating point by the experiment's own buffer arm —
the identical weights on the identical eval set with the shuffle OFF are what
the margin is read from, so a dead rig cannot manufacture the collapse
without also killing the claim.

FORECAST, pre-registered with its reasoning (ps09 idiom): lean PASS —
composition is recurrence's home ground, the GRU ships with the retain-bias
init that made ME.8's hold trainable, and the null's expressivity ceiling on
deep tiers is arithmetic. The live risk is the other branch: a GRU that
cannot LEARN mod-8 register updates at this budget loses with train_fit on
the row — a recipe-vs-mechanism question the row's own numbers will separate
(fit train + failed holdout vs never fit). Either branch is a result: the
fast/slow commitment has never had a measured number.
"""
from __future__ import annotations

import math

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# The implementation under test: the shipped buffer whose slots are on trial.
IMPL_DEPS = ['WorkingMemory.py']

C = 8                      # register classes -> chance 0.125
N_NOISE = 4                # distractor channels, N(0,1) every step
OBS_DIM = C + N_NOISE + 2  # + query flag + time channel
T = 40                     # steps; query read at T-1
DEPTH_MAX = 5              # writes per episode: 1 init + up to 4 deltas
WRITE_LO, WRITE_HI = 4, T - 6
HIDDEN = 32                # the shipped buffer's size (ME.8's constant)
D_ATTN = 125               # sized so params match the buffer arm (docstring)
BATCH = 64
MAX_ITERS = 1500
STREAK_STOP = 25           # early stop after this many perfect train batches
LR = 5e-3
N_EVAL = 512               # holdout episodes per seed

# Bars — frozen before any number was read (see docstring; none moved when
# the venue did).
PARAM_TOL = 0.02           # VOID lane: relative param-count mismatch
LEARN_MIN = 0.25           # VOID lane: the NULL must be a learner (2x chance)
MARGIN_MIN = 0.10          # the claim: buffer beats no-buffer by this, per seed
CONTROL_MAX = 0.30         # the shuffled buffer must collapse to here or below

_CACHE: dict = {}          # seed -> trained artifacts, shared with _control


def _build_buffer():
    from WorkingMemory import WorkingMemory
    return WorkingMemory(OBS_DIM, C, HIDDEN)


def _build_attention():
    import torch
    import torch.nn as nn

    class _EpisodicAttention(nn.Module):
        """No working memory: one learned query attends over the raw episodic
        record (every step's observation, which already carries its own time
        channel). Linear keys/values, linear head — capacity lives in d,
        sized to match the buffer arm's parameter count."""

        def __init__(self, obs_dim: int, n_out: int, d: int):
            super().__init__()
            self.key = nn.Linear(obs_dim, d)
            self.val = nn.Linear(obs_dim, d)
            self.query = nn.Parameter(torch.randn(d) * 0.1)
            self.head = nn.Linear(d, n_out)
            self.d = d

        def forward(self, store):            # store: (T, B, obs_dim)
            k = self.key(store)              # (T, B, d)
            v = self.val(store)
            scores = (k * self.query).sum(-1) / math.sqrt(self.d)   # (T, B)
            attn = scores.softmax(dim=0)
            out = (attn.unsqueeze(-1) * v).sum(0)                   # (B, d)
            return self.head(out)

    return _EpisodicAttention(OBS_DIM, C, D_ATTN)


def _episode_batch(n: int, gen) -> tuple:
    """A batch of relative-update-register episodes. (obs: (T, n, OBS_DIM),
    target: (n,)). Step 0 writes the absolute init cue; each later write
    carries a delta on the same channels; target = (c0 + sum(deltas)) mod C.
    Deterministic in the generator: both arms and the eval consume identical
    data."""
    import torch
    obs = torch.zeros(T, n, OBS_DIM)
    obs[:, :, C:C + N_NOISE] = torch.randn(T, n, N_NOISE, generator=gen)
    obs[T - 1, :, C + N_NOISE] = 1.0                      # query flag
    tgrid = torch.arange(T, dtype=torch.float32) / (T - 1)
    obs[:, :, C + N_NOISE + 1] = tgrid.unsqueeze(1)       # time channel
    target = torch.zeros(n, dtype=torch.long)
    for b in range(n):
        depth = int(torch.randint(1, DEPTH_MAX + 1, (1,), generator=gen))
        c0 = int(torch.randint(0, C, (1,), generator=gen))
        obs[0, b, c0] = 1.0
        reg = c0
        if depth > 1:
            steps = (torch.randperm(WRITE_HI - WRITE_LO, generator=gen)[
                :depth - 1] + WRITE_LO).sort().values
            deltas = torch.randint(0, C, (depth - 1,), generator=gen)
            for s, d in zip(steps.tolist(), deltas.tolist()):
                obs[s, b, d] = 1.0
                reg = (reg + d) % C
        target[b] = reg
    return obs, target


def _forward(arm: str, model, obs):
    """One episode batch through either arm. The buffer steps through time;
    the attention arm reads the whole stored record at once."""
    if arm == "buffer":
        model.reset(obs.shape[1])
        for t in range(T):
            logits = model.step(obs[t])
        return logits
    return model(obs)


def _train_arm(arm: str, seed: int) -> dict:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = _build_buffer() if arm == "buffer" else _build_attention()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Identical batch stream per seed for BOTH arms: same generator seed,
    # same draw order, same draw sizes.
    gen = torch.Generator().manual_seed(12345 + seed)
    streak, finite, recent = 0, True, []
    for _ in range(MAX_ITERS):
        obs, target = _episode_batch(BATCH, gen)
        logits = _forward(arm, model, obs)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(1) == target).float().mean().item()
        recent = (recent + [acc])[-50:]
        streak = streak + 1 if acc == 1.0 else 0
        if streak >= STREAK_STOP:
            break
    return {"model": model, "n_params": n_params, "finite": finite,
            "train_fit": round(float(np.mean(recent)) if recent else 0.0, 4)}


def _eval_arm(arm: str, model, seed: int, shuffle: bool = False) -> float:
    """Holdout accuracy on the seed's pre-registered eval set. With
    shuffle=True (buffer arm only), the hidden state's entries are permuted
    by a fresh random permutation after every step — the registry's control:
    size and parameters untouched, held state destroyed."""
    import torch
    gen = torch.Generator().manual_seed(777 + seed)
    obs, target = _episode_batch(N_EVAL, gen)
    sgen = torch.Generator().manual_seed(555 + seed)
    with torch.no_grad():
        if arm == "buffer":
            model.reset(N_EVAL)
            for t in range(T):
                logits = model.step(obs[t])
                if shuffle and t < T - 1:
                    perm = torch.randperm(HIDDEN, generator=sgen)
                    model.h = model.h[:, perm]
        else:
            logits = model(obs)
    return float((logits.argmax(1) == target).float().mean().item())


def _pipeline(seed: int) -> dict:
    if seed in _CACHE:
        return _CACHE[seed]
    buf = _train_arm("buffer", seed)
    nob = _train_arm("no_buffer", seed)
    _CACHE[seed] = {"buf": buf, "nob": nob}
    return _CACHE[seed]


def _experiment(seed: int) -> dict:
    art = _pipeline(seed)
    buf, nob = art["buf"], art["nob"]
    m = {
        "buffer_params": buf["n_params"],
        "nobuffer_params": nob["n_params"],
        "param_rel_diff": round(abs(buf["n_params"] - nob["n_params"])
                                / buf["n_params"], 5),
        "buffer_train_fit": buf["train_fit"],
        "nobuffer_train_fit": nob["train_fit"],
    }
    if not (buf["finite"] and nob["finite"]):
        m["void_reason"] = "non-finite training loss — the rig, not the claim"
        return m
    if m["param_rel_diff"] > PARAM_TOL:
        m["void_reason"] = (f"arms not parameter-matched: "
                            f"{buf['n_params']} vs {nob['n_params']}")
        return m
    m["buffer_acc"] = round(_eval_arm("buffer", buf["model"], seed), 4)
    m["nobuffer_acc"] = round(_eval_arm("no_buffer", nob["model"], seed), 4)
    m["buffer_vs_no_buffer_margin"] = round(
        m["buffer_acc"] - m["nobuffer_acc"], 4)
    # Worst-seed gate: run_spec aggregates by MEAN, so the deciding conjunct
    # is a per-seed boolean — its mean is 1.0 only if EVERY seed cleared.
    m["margin_ok"] = 1.0 if (
        m["buffer_vs_no_buffer_margin"] >= MARGIN_MIN) else 0.0
    if m["nobuffer_acc"] < LEARN_MIN:
        m["void_reason"] = (
            f"the null is not a learner here (T2.02 rule — a dead null "
            f"cannot arbitrate): nobuffer_acc {m['nobuffer_acc']} vs "
            f"LEARN_MIN {LEARN_MIN}")
    return m


def _control(seed: int) -> dict:
    """The shipped buffer with its state shuffled between steps MUST collapse.
    Uses the experiment's own trained weights and eval set, so the presence
    (ordered state scoring high) is demonstrated at the identical operating
    point by the experiment itself."""
    art = _pipeline(seed)
    try:
        acc = round(
            _eval_arm("buffer", art["buf"]["model"], seed, shuffle=True), 4)
        return {"shuffled_acc": acc,
                "collapsed_ok": 1.0 if acc <= CONTROL_MAX else 0.0}
    finally:
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict):
    if "void_reason" in m:
        return Status.VOID
    # Both conjuncts are worst-seed gates: per-seed booleans aggregated by
    # mean read 1.0 only if every seed cleared (see docstring — the mean
    # margin alone bought a PASS that the pre-registered rule forbade).
    return m["margin_ok"] == 1.0 and c["collapsed_ok"] == 1.0


def _precheck(seed: int = 90) -> dict:
    """REGISTERABILITY ONLY (unsaturated-null rule): train and evaluate the
    NULL arm alone on the sizing seed. Never a claim seed, never the buffer
    arm, writes nothing to the ledger. Both venue readings (v1 saturated,
    v2 registered) are recorded in the docstring's STATISTIC_BOUND block."""
    nob = _train_arm("no_buffer", seed)
    acc = _eval_arm("no_buffer", nob["model"], seed)
    return {"nobuffer_acc": round(acc, 4),
            "distance_to_bound": round(1.0 - acc, 4),
            "registerable": LEARN_MIN <= acc <= 1.0 - MARGIN_MIN}


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["WM.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
