"""WM.02 — a rotating working memory takes the seat the GRU buffer could not.

THE QUESTION. The Working-memory seat is VACANT: its only arm, the shipped GRU
buffer, FAILs `WM.01`'s worst-seed gate (per-seed margins over the no-buffer
null 0.47 / 0.08 / 0.03 against a 0.10 bar). `WM.01` raced one mechanism
against a null. This spec enters a CHALLENGER — `RingMemory.py`, a store whose
events rotate the held state (the head-direction circuit's move) — on WM.01's
own venue, own recipe and own bars, with the incumbent run on the same ruler.

THE VENUE IS WM.01's, IMPORTED, NOT COPIED: `_episode_batch` (the
relative-update register, T=40, depth 1..5, chance 1/8), BATCH, MAX_ITERS,
STREAK_STOP, LR, N_EVAL and the three bars LEARN_MIN / MARGIN_MIN /
CONTROL_MAX all come from `wm_01_working_memory_earns_slots`. The GRU and the
no-buffer null are trained by that module's own `_train_arm` and scored by its
own `_eval_arm`. No bar moved.

ARMS, per seed, identical batch stream and identical holdout:
  ring       `RingMemory(OBS_DIM, C, 32)` — 1,031 parameters, 32-float state.
  gru        the shipped `WorkingMemory(OBS_DIM, C, 32)` — 4,872 parameters.
  no_buffer  WM.01's `_EpisodicAttention` — 4,883 parameters, the null.
Parameter rule: the challenger may not EXCEED the incumbent (VOID if ring >
gru * (1 + PARAM_TOL)). It runs at 21% of the incumbent's parameters and the
same state size; a win with fewer is not a matching defect, it is the cost
column.

SEEDS. Five claim seeds, 100..104 (`SEED_OFFSET` + the runner's 0..4). The
pilot below was read on seeds 0..4; the registered run reads none of them.

THE DECISION RULE, declared before the registered run:
  VOID  non-finite loss in any arm; the null below LEARN_MIN (a dead null
        cannot arbitrate); the challenger larger than the incumbent.
  PASS  ALL of, on EVERY seed (per-seed booleans, mean == 1.0 — WM.01's
        worst-seed mechanics):
          ring_acc - nobuffer_acc >= MARGIN_MIN            (earns its slots)
          ring_acc - gru_acc      >= -NOT_WORSE            (never loses to the
                                                            incumbent)
        AND the mean of (ring_acc - gru_acc) >= MARGIN_MIN (beats the
        incumbent by the seat's own margin)
        AND both controls fail on every seed.
  FAIL  otherwise. A FAIL kills the entry: the seat stays VACANT and the ring
        memory goes to docs/LAB.md as DEAD with its number.
NOT_WORSE = 0.02 is the only new constant: two holdout episodes in 512 would
otherwise turn a tie at the ceiling into a loss.

THE CONTROLS (each RUN, each must fail):
  shuffled   WM.01's control on the challenger — the trained ring's state
             entries are permuted after every step; accuracy must collapse to
             <= CONTROL_MAX. Liveness is the experiment's own number: the
             same weights on the same holdout with the shuffle off.
  no_rotate  the same module trained identically with the turn disabled and
             frozen (`rotate=False`) — the gated write alone. Its margin over
             the null must stay BELOW MARGIN_MIN. If it clears the bar, the
             rotation is not what earned the slots and the claim is false.

PILOT (2026-11-11, seeds 0..4, never claim seeds; scratch script, no ledger):
ring 1.000 / 1.000 / 1.000 / 0.871 / 1.000; shuffled 0.11-0.14; no_rotate
0.318 / 0.283 (seeds 0, 1); incumbent on the ledger 0.546 +- 0.203, null
0.354. Generality probe on WM.01's v1 venue (absolute last-write, not a group
task): ring 0.998 / 1.000 / 1.000, gru 1.000 / 1.000 — the ring is not a
one-trick rotation. Wall: ~50 s ring, ~70 s gru, ~40 s null, ~125 s no_rotate
per seed -> ~25 min for five seeds, CPU.

WHAT A PASS DOES NOT SAY. This is a 40-step register fixture, not a life. A
PASS seats the ring memory as the working-memory mechanism to carry into
ME.8 / LF.02 / LF.03; whether it helps him live is those specs' question.
The venue's answer is a cyclic group, rotation's home ground — which is why
the v1 probe is on the record and why the next ring for this seat should be a
non-commutative register.

FORECAST: PASS. The live risk is a ring seed that stalls short of the ceiling
(pilot seed 3: 0.871) on a seed where the GRU happens to converge.
"""
from __future__ import annotations

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from . import wm_01_working_memory_earns_slots as W

IMPL_DEPS = ['RingMemory.py', 'WorkingMemory.py']

SEED_OFFSET = 100          # claim seeds 100..104; the pilot read 0..4
NOT_WORSE = 0.02           # per-seed tolerance against the incumbent

_CACHE: dict = {}          # seed -> trained artifacts, shared with _control


def _build_ring(rotate: bool = True):
    from RingMemory import RingMemory
    return RingMemory(W.OBS_DIM, W.C, W.HIDDEN, rotate=rotate)


def _train_ring(seed: int, rotate: bool = True) -> dict:
    """WM.01's `_train_arm` recipe, statement for statement, for a module that
    module's builder switch does not know: same torch seed, same batch stream,
    same optimiser, same iteration cap and early stop."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = _build_ring(rotate)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=W.LR)
    gen = torch.Generator().manual_seed(12345 + seed)
    streak, finite, recent = 0, True, []
    for _ in range(W.MAX_ITERS):
        obs, target = W._episode_batch(W.BATCH, gen)
        logits = W._forward("buffer", model, obs)
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
        if streak >= W.STREAK_STOP:
            break
    return {"model": model, "n_params": n_params, "finite": finite,
            "train_fit": round(float(np.mean(recent)) if recent else 0.0, 4)}


def _pipeline(seed: int) -> dict:
    s = seed + SEED_OFFSET
    if s not in _CACHE:
        _CACHE[s] = {"ring": _train_ring(s),
                     "gru": W._train_arm("buffer", s),
                     "nob": W._train_arm("no_buffer", s)}
    return _CACHE[s]


def _experiment(seed: int) -> dict:
    s = seed + SEED_OFFSET
    art = _pipeline(seed)
    ring, gru, nob = art["ring"], art["gru"], art["nob"]
    m = {"ring_params": ring["n_params"], "gru_params": gru["n_params"],
         "nobuffer_params": nob["n_params"],
         "ring_train_fit": ring["train_fit"],
         "gru_train_fit": gru["train_fit"]}
    if not (ring["finite"] and gru["finite"] and nob["finite"]):
        m["void_reason"] = "non-finite training loss — the rig, not the claim"
        return m
    if ring["n_params"] > gru["n_params"] * (1 + W.PARAM_TOL):
        m["void_reason"] = (f"challenger larger than the incumbent: "
                            f"{ring['n_params']} vs {gru['n_params']}")
        return m
    m["ring_acc"] = round(W._eval_arm("buffer", ring["model"], s), 4)
    m["gru_acc"] = round(W._eval_arm("buffer", gru["model"], s), 4)
    m["nobuffer_acc"] = round(W._eval_arm("no_buffer", nob["model"], s), 4)
    art["nobuffer_acc"] = m["nobuffer_acc"]
    m["ring_vs_no_buffer_margin"] = round(m["ring_acc"] - m["nobuffer_acc"], 4)
    m["ring_vs_gru_margin"] = round(m["ring_acc"] - m["gru_acc"], 4)
    m["gru_vs_no_buffer_margin"] = round(m["gru_acc"] - m["nobuffer_acc"], 4)
    m["earns_ok"] = 1.0 if m["ring_vs_no_buffer_margin"] >= W.MARGIN_MIN else 0.0
    m["not_worse_ok"] = 1.0 if m["ring_vs_gru_margin"] >= -NOT_WORSE else 0.0
    if m["nobuffer_acc"] < W.LEARN_MIN:
        m["void_reason"] = (f"the null is not a learner here: nobuffer_acc "
                            f"{m['nobuffer_acc']} vs LEARN_MIN {W.LEARN_MIN}")
    return m


def _control(seed: int) -> dict:
    s = seed + SEED_OFFSET
    art = _pipeline(seed)
    try:
        shuffled = round(
            W._eval_arm("buffer", art["ring"]["model"], s, shuffle=True), 4)
        norot = _train_ring(s, rotate=False)
        norot_acc = round(W._eval_arm("buffer", norot["model"], s), 4)
        nob_acc = art.get("nobuffer_acc")
        if nob_acc is None:
            nob_acc = round(W._eval_arm("no_buffer", art["nob"]["model"], s), 4)
        margin = round(norot_acc - nob_acc, 4)
        return {"shuffled_acc": shuffled,
                "collapsed_ok": 1.0 if shuffled <= W.CONTROL_MAX else 0.0,
                "norot_acc": norot_acc,
                "norot_vs_no_buffer_margin": margin,
                "norot_fails_ok": 1.0 if margin < W.MARGIN_MIN else 0.0}
    finally:
        _CACHE.pop(s, None)


def _check(m: dict, c: dict):
    if "void_reason" in m:
        return Status.VOID
    return (m["earns_ok"] == 1.0 and m["not_worse_ok"] == 1.0
            and m["ring_vs_gru_margin"] >= W.MARGIN_MIN
            and c["collapsed_ok"] == 1.0 and c["norot_fails_ok"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["WM.02"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
