"""LF.04 — does living teach him?  (ORDERS J1; PILOT STAGE — not yet registered)

LF.03 PASSed the integration question: the seated learner (`wm-latent`)
lives, dies, resumes and its weights move. It says nothing about whether he
gets BETTER. This module asks that on the whole creature, in lethal W0:

    the learner vs HIS OWN FROZEN TWIN — same torch init, same world seed,
    same exploration-noise stream, same exploration schedule — whose weights
    are never updated. N lives each. Claim (to be registered after the pilot):
    late-life survival (mean sim-s over the last third of lives) exceeds the
    twin's late-life survival by a margin, paired, on EVERY seed, and exceeds
    his own first third.

Why the twin and not "late vs early" alone: the exploration std decays over
HORIZON decisions for every arm, so a frozen policy's lives can lengthen (or
shorten) with no learning at all. The twin carries exactly that schedule;
only the optimiser differs.

THE LOOP is LF.03's `_Life` (itself run_survival's core path) imported, not
re-derived. Two hooks are added here, both declared:
  explore_salt   offsets the exploration-noise RNG (twin-vs-twin control: two
                 frozen twins that differ ONLY in noise draws measure how far
                 two learning-free lives drift apart by chance);
  shuffle_rew    permutes the rewards across steps inside each GAE segment
                 before targets are computed (the shuffled-reward learner: the
                 same optimiser steps on credit that carries no information
                 about which action preceded which outcome).

A life that reaches LIFE_CAP decisions STOPS THE ARM: that life is recorded
CENSORED at its length (a lower bound) and no fake death is written into the
world, the diary or the learner's returns. Reported, never hidden.

PILOT (this file's `pilot` role): ONE seed (90 — not a claim seed), arms
learner / twin-A / twin-B, N_PILOT lives each, one per-life JSON line streamed
to OUT as each life ends. Its job is to pick N and the margin from the twins'
own life-to-life and twin-to-twin spread, declared before claim seeds are read.

DECLARED 2026-11-12 08:20 UTC, BEFORE ANY LEARNER LIFE PAST LIFE 0 WAS READ
(pilot launched 08:10; at declaration the file held learner life 0 = 421.2 s
and twin lives 0-1 only):
  N_LIVES = 15 per arm per seed; thirds = lives 0-4 (early), 10-14 (late).
  MARGIN_S = max(MARGIN_FLOOR_S, 2 * sd_twin * sqrt(2 / 5)), where sd_twin is
    the pooled sample sd of single-life sim-s over ALL 30 pilot twin lives
    (twinA + twinB, seed 90) — i.e. two standard errors of a difference of two
    5-life means. Computed by `margin_from_pilot()` from twin rows ONLY; the
    learner's pilot rows play no part. MARGIN_FLOOR_S = 20 sim-s.
  Claim seeds 100, 101, 102 (SEED_OFFSET 100 + the runner's 0..2).
  Gate per seed, the SAME function for experiment and both controls:
    fires  <=>  late(X) - late(twinA) >= MARGIN_S  AND  late(X) - early(X) >= MARGIN_S
  PASS = fires for the learner on EVERY seed, AND for twinB (twin-vs-twin)
    on NO seed, AND for the shuffled-reward learner on NO seed.
  VOID = a twin life censored (a learning-free arm outliving the cap makes the
    venue unable to show gain), or calibration refused. A censored LEARNER
    life counts at its censored length for every remaining life (a lower
    bound: he had not died) — conservative against the claim.
  REFERENCE (reported, never gated): the statue's mean life, so a reader can
    tell foraging from keeping still.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .lf_03_learner_lives import _Life

N_PILOT = 15
N_LIVES = 15
THIRD = 5
MARGIN_FLOOR_S = 20.0
SEED_OFFSET = 100
PILOT_FILE = "/data/tmp/lf04/pilot_s90.jsonl"
LIFE_CAP = 12_000          # decisions; ~5.5x LF.03's mean life (~2,180)
EXPLORE_SALT_B = 1_000_003


class _Life4(_Life):
    def __init__(self, seed, j0, alpha, *, train=True, explore_salt=0,
                 shuffle_rew=False):
        super().__init__(seed, j0, alpha, train=train)
        if explore_salt:
            self.rng = np.random.RandomState(seed * 6553 + 11 + explore_salt)
        self.shuffle_rew = shuffle_rew
        self.shuf_rng = np.random.RandomState(seed * 104_729 + 5)

    def _close_segment(self, boot, terminal):
        if self.shuffle_rew and len(self.seg["rew"]) > 1:
            rew = self.seg["rew"]
            perm = self.shuf_rng.permutation(len(rew))
            self.seg["rew"] = [rew[i] for i in perm]
        super()._close_segment(boot, terminal)


def live_lives(L: _Life4, n_lives: int, out=None, tag: str = "") -> list:
    """Live n_lives lives; return [(sim_s, decisions, censored, ate)]."""
    w = L.w
    lives = []
    k_life = 0
    ate0 = 0
    t0 = time.perf_counter()
    while len(lives) < n_lives:
        L.step()
        k_life += 1
        died = w.died_this_decision
        if died or k_life >= LIFE_CAP:
            span = float(w.life_lengths[-1]) if died else float(
                w.sim_seconds - w._life_started_at)
            ate = int(sum(w.drives.ate_total.values()))
            row = {"tag": tag, "life": len(lives), "sim_s": round(span, 2),
                   "decisions": k_life, "censored": int(not died),
                   "ate": ate - ate0, "k": L.k, "opt_steps": L.opt_steps,
                   "wall_s": round(time.perf_counter() - t0, 1)}
            lives.append(row)
            if out is not None:
                out.write(json.dumps(row) + "\n")
                out.flush()
            if not died:
                break      # he outlived the apparatus: the arm stops here,
                           # its last life CENSORED (no fake death is written)
            ate0 = int(sum(w.drives.ate_total.values()))
            k_life = 0
    return lives


ARMS = {
    "learner": dict(train=True),
    "twinA": dict(train=False),
    "twinB": dict(train=False, explore_salt=EXPLORE_SALT_B),
    "shuffled": dict(train=True, shuffle_rew=True),
}


def _statue_lives(seed, j0, alpha, n_lives, out, tag):
    """REFERENCE, not an arm: zero action every decision (survival.py's
    'statue' policy) in the same world. Tells a reader whether a longer life
    is foraging or merely spending less energy by keeping still."""
    import mujoco

    from ..cores import ACTION_DIM
    from ..w0 import W0
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    a = np.zeros(ACTION_DIM)
    k_life, n = 0, 0
    while n < n_lives:
        w.decide(a)
        mujoco.mj_forward(w.model, w.data)
        mujoco.mj_rnePostConstraint(w.model, w.data)
        k_life += 1
        if w.died_this_decision or k_life >= LIFE_CAP:
            died = w.died_this_decision
            span = (float(w.life_lengths[-1]) if died
                    else float(w.sim_seconds - w._life_started_at))
            out.write(json.dumps({"tag": tag, "life": n, "sim_s": round(span, 2),
                                  "decisions": k_life,
                                  "censored": int(not died)}) + "\n")
            out.flush()
            if not died:
                return
            n, k_life = n + 1, 0


def _pilot(arm: str, seed: int, n_lives: int, out_path: str) -> None:
    from .lf_03_learner_lives import _calibration
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    if arm == "statue":
        with open(out_path, "a") as out:
            _statue_lives(seed, j0, alpha, n_lives, out, f"statue:s{seed}")
        return
    L = _Life4(seed, j0, alpha, **ARMS[arm])
    with open(out_path, "a") as out:
        live_lives(L, n_lives, out, tag=f"{arm}:s{seed}")


def _thirds(spans: list) -> tuple:
    return (float(np.mean(spans[:THIRD])), float(np.mean(spans[-THIRD:])))


def gate_fires(x_spans: list, twin_spans: list, margin: float) -> bool:
    """THE gate — experiment and both controls read this one function."""
    ex, lx = _thirds(x_spans)
    _, lt = _thirds(twin_spans)
    return (lx - lt) >= margin and (lx - ex) >= margin


def margin_from_pilot(path: str = PILOT_FILE) -> dict:
    """The declared margin rule, from the TWIN rows only."""
    rows = [json.loads(x) for x in open(path)]
    tw = [r for r in rows if r["tag"].split(":")[0] in ("twinA", "twinB")]
    spans = [r["sim_s"] for r in tw]
    if (len(spans) < 2 * N_PILOT or any(r["censored"] for r in tw)):
        return {"ready": False, "n_twin_lives": len(spans)}
    sd = float(np.std(spans, ddof=1))
    m = max(MARGIN_FLOOR_S, 2.0 * sd * (2.0 / THIRD) ** 0.5)
    return {"ready": True, "n_twin_lives": len(spans), "sd_twin": sd,
            "margin_s": m}


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "pilot":
        _pilot(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        raise SystemExit(f"unknown role {role!r}")
