"""LF.06 — the hand-over: a Jack trained here still learns on another machine.

BUILD STAGE (ORDERS J8). The spec is registered (registry_expansion.py,
2026-11-13); this module holds the pieces its run needs, declared before any
run. It has NO `run()` yet: the claim run waits on S1's kaggle-cpu backend and
on LF.04 (queued behind it). Nothing here writes the ledger.

Pieces, each reusing the creature's existing code paths, never re-deriving:
  checkpoint   LF.03's five-store `_capture` / `_load_checkpoint` / `_apply`
               (world, needs, diary, learner = weights + optimiser + replay
               ring + RNGs). The hand-over file IS that checkpoint.
  strip        CONTROL 1: same file with the diary emptied, the optimiser
               reset to a fresh one and the replay ring emptied — weights only.
  fresh        CONTROL 2: same architecture, NEW init (torch seed salted),
               nothing carried.
  exam         K_EXAM frozen lives: optimiser never stepped, exploration std
               pinned at its schedule's END value (L.k = HORIZON), exam world
               seeds 9000 + 10*seed + i, ONE fixed noise salt; exam = mean
               sim-s. A censored exam life (LIFE_CAP) -> the exam is VOID.
  tol_from_pilot  TOL_S = max(20, 2 * sd_exam * sqrt(2/5)), sd_exam the pooled
               single-life sd of the seed-90 pilot exam over 2 noise salts
               (LF.04's margin rule, in the exam's own units).

All numbers are copied from the registered spec's notes; if this file and the
spec disagree, the spec wins and this file is the bug.
"""
from __future__ import annotations

import json

import numpy as np

from .lf_03_learner_lives import HORIZON, _apply, _calibration, _capture
from .lf_04_living_teaches import LIFE_CAP, _Life4, live_lives

N_HERE = 10
M_THERE = 10
K_EXAM = 5
CLAIM_SEEDS = (110, 111, 112)
PILOT_SEED = 90
TOL_FLOOR_S = 20.0
MARGIN_S = 272.812                    # LF.04's declared margin, same venue
EXAM_SALT = 2_000_029                 # the ONE fixed exam noise salt
PILOT_SALTS = (EXAM_SALT, 3_000_017)  # the pilot's 2 noise salts
FRESH_TORCH_SALT = 77_777
PILOT_FILE = "/data/tmp/lf06/exam_pilot_s90.jsonl"


def exam_seed(seed: int, i: int) -> int:
    return 9000 + 10 * seed + i


# ── the hand-over file and its two controls ────────────────────────────────
def handover(L) -> dict:
    """The checkpoint that travels: LF.03's five stores, unchanged."""
    return _capture(L)


def strip(ckpt: dict, seed: int, j0: float, alpha: float):
    """CONTROL 1: weights only. Load the full file, then empty the diary,
    replace the optimiser with a fresh one and empty the replay ring."""
    from ..cores import make_optimizer
    L = _apply(ckpt, seed, j0, alpha)
    L.diary.events, L.diary._tok = [], []
    L.opt = make_optimizer(L.core)
    L.ring.n = 0
    L.seg = {"obs": [], "act": [], "rew": [], "val": []}
    return L


def fresh(seed: int, j0: float, alpha: float, train: bool = False):
    """CONTROL 2: same architecture, a NEW init; nothing carried."""
    import torch
    L = _Life4(seed, j0, alpha, train=train)
    torch.manual_seed(9_000 + seed + FRESH_TORCH_SALT)
    from ..cores import build_arm, make_optimizer
    from .lf_03_learner_lives import ARM
    L.core = build_arm(ARM)
    L.core.eval()
    L.opt = make_optimizer(L.core) if train else None
    return L


# ── the frozen exam ────────────────────────────────────────────────────────
def exam(core_state: dict, seed: int, j0: float, alpha: float,
         salt: int = EXAM_SALT, out=None, tag: str = "exam") -> dict:
    """K_EXAM frozen lives, one per exam world seed, the given weights.
    Returns {"exam_s": mean sim-s | None, "spans": [...], "void": reason}."""
    spans, void = [], ""
    for i in range(K_EXAM):
        L = _Life4(exam_seed(seed, i), j0, alpha, train=False,
                   explore_salt=salt)
        L.core.load_state_dict(core_state)
        L.k = HORIZON                  # std pinned at the schedule's END
        rows = live_lives(L, 1, out, tag=f"{tag}:s{seed}:e{i}")
        if rows[0]["censored"]:
            void = f"exam life {i} censored at LIFE_CAP={LIFE_CAP}"
            break
        spans.append(rows[0]["sim_s"])
    return {"exam_s": None if void else float(np.mean(spans)),
            "spans": spans, "void": void}


def tol_from_pilot(path: str = PILOT_FILE) -> dict:
    """The declared TOL_S rule, from the seed-90 pilot exam rows only."""
    rows = [json.loads(x) for x in open(path)]
    ex = [r for r in rows if r["tag"].startswith("exam")]
    if len(ex) < 2 * K_EXAM or any(r["censored"] for r in ex):
        return {"tol_s": None, "why": f"pilot incomplete or censored "
                                      f"({len(ex)} rows)"}
    sd = float(np.std([r["sim_s"] for r in ex], ddof=1))
    return {"tol_s": max(TOL_FLOOR_S, 2.0 * sd * np.sqrt(2.0 / K_EXAM)),
            "sd_exam": sd, "n": len(ex)}


def _pilot(out_path: str = PILOT_FILE) -> None:
    """Seed 90 (not a claim seed): N_HERE learning lives, then the frozen
    exam of the end weights under both pilot salts. Streams JSON lines."""
    import os
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    L = _Life4(PILOT_SEED, j0, alpha, train=True)
    with open(out_path, "a") as out:
        live_lives(L, N_HERE, out, tag=f"here:s{PILOT_SEED}")
        state = {k: v.clone() for k, v in L.core.state_dict().items()}
        for salt in PILOT_SALTS:
            exam(state, PILOT_SEED, j0, alpha, salt=salt, out=out,
                 tag=f"exam{salt}")


def _selftest() -> dict:
    """No lives: build a creature, take a few decisions, hand over, and check
    that the full file carries every store, that strip() drops exactly the
    stores it declares, and that fresh() differs in weights."""
    import torch

    from .lf_03_learner_lives import _flat, _Life
    j0, alpha, prov = _calibration()
    if j0 is None:
        return {"ok": False, "why": f"calibration refused: {prov}"}
    L = _Life(PILOT_SEED, j0, alpha)
    while L.opt_steps < 2:             # until the ring + optimiser are live
        L.step()
    ck = handover(L)
    full = _apply(ck, PILOT_SEED, j0, alpha)
    st = strip(ck, PILOT_SEED, j0, alpha)
    fr = fresh(PILOT_SEED, j0, alpha)
    w_full = torch.equal(_flat(full.core), _flat(L.core))
    w_strip = torch.equal(_flat(st.core), _flat(L.core))
    w_fresh = torch.equal(_flat(fr.core), _flat(L.core))
    res = {
        "full_weights_equal": w_full,
        "full_diary_n": len(full.diary.events),
        "full_ring_n": int(full.ring.n),
        "full_opt_state_n": len(full.opt.state_dict()["state"]),
        "strip_weights_equal": w_strip,
        "strip_diary_n": len(st.diary.events),
        "strip_ring_n": int(st.ring.n),
        "strip_opt_state_n": len(st.opt.state_dict()["state"]),
        "fresh_weights_equal": w_fresh,
    }
    res["ok"] = bool(w_full and w_strip and not w_fresh
                     and res["full_diary_n"] > 0 and res["full_ring_n"] > 0
                     and res["full_opt_state_n"] > 0 and res["strip_diary_n"] == 0
                     and res["strip_ring_n"] == 0
                     and res["strip_opt_state_n"] == 0)
    return res


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(_selftest(), indent=1))
    elif sys.argv[1:] == ["pilot"]:
        _pilot()
    else:
        raise SystemExit("usage: python -m experiments.tests.lf_06_handover "
                         "selftest|pilot")
