"""LAB — exploration in BOUTS: temporally-correlated action noise vs the white
noise every learner here has explored with (no ledger row).

Seat: the learning core's exploration (no seat of its own; the holder by
default is `_Life.step`'s `mean + std * rng.randn(8)`, independent every
0.2-s decision, std 0.5 -> 0.1 over 15k decisions).

THE BET (LAB 2026-11-12). (1) Replace the white draw by noise with memory —
`ou` (one AR(1), tau 20 decisions = 4 sim-s) or `pink` (a 1/f mixture of
AR(1)s, tau 1..64 decisions; Eberhard et al. 2023, "Pink noise is all you
need", streaming form because his life has no episodes) — SAME marginal std,
same clip, nothing else touched. (2) It might beat white because a perfect
map needs a median 41.5 decisions of one heading from birth to meal, and a
white walk's displacement grows as sqrt(n): he jitters in place, meets food
1-2 times in 22 lives, and no credit rule (gamma .95 or .99) can learn from
meals that do not happen. An infant's movement is bouts, not jitter.
(3) Killed by: stage A below.

STAGE A — `probe` (no learner, minutes): the oracle's own ruler (body_probe:
births with a floor food ready, E = 250 decisions, W0's own birth stream),
policy = clip(STD * noise, -1, 1) on all 8 action dims, mean 0 (the actor at
init). DECLARED BEFORE ANY BIRTH IS READ: seeds 90 and 94, 20 births each,
STD 0.5. A coloured arm is ALIVE iff it feeds more births than `white` on
BOTH seeds and by >= 4 of 40 in sum; if neither `ou` nor `pink` does, the
bet is DEAD (exploration's timescale is not what starves him).

STAGE B — `pilot` (the learner, wean_pilot's loop, smell installed): the
same swap inside the living learner; `install_noise` replaces `_Life4.rng`
with an object whose `.randn(n)` is coloured. lf_03/lf_04/survival on disk
are untouched; no certificate moves.

    nice -n 19 $PY -m experiments.explore_bout probe <kind> <seed> [n_ep] [E] [std]
    nice -n 19 $PY -m experiments.explore_bout pilot <kind> <smell-arm> <seed> <total> <out>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

KINDS = ("white", "ou", "pink")
OU_TAU = 20.0                                  # decisions (4 sim-s)
PINK_TAUS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)   # 0 = a white component


class BoutNoise:
    """Unit-marginal-variance noise with memory; `.randn(n)` like RandomState.
    white: iid.  ou: one AR(1).  pink: equal-variance AR(1)s at octave taus,
    summed and renormalised — a streaming 1/f over 1..64 decisions."""

    def __init__(self, kind: str, seed: int):
        if kind not in KINDS:
            raise SystemExit(f"unknown noise {kind!r}; one of {KINDS}")
        self.kind = kind
        self.rng = np.random.RandomState(seed)
        taus = {"white": (0.0,), "ou": (OU_TAU,), "pink": PINK_TAUS}[kind]
        self.rho = np.array([0.0 if t <= 0 else math.exp(-1.0 / t)
                             for t in taus])[:, None]
        self.x = None

    def randn(self, n: int) -> np.ndarray:
        e = self.rng.randn(len(self.rho), n)
        if self.x is None or self.x.shape[1] != n:
            self.x = e                          # stationary start
        else:
            self.x = self.rho * self.x + np.sqrt(1.0 - self.rho ** 2) * e
        return self.x.sum(0) / math.sqrt(len(self.rho))


def selftest() -> dict:
    """Marginal std ~1 for every kind; lag-10 autocorrelation ~0 for white
    and clearly > 0 for the coloured kinds (a check that can fail)."""
    out = {}
    for k in KINDS:
        g = BoutNoise(k, 0)
        x = np.array([g.randn(8) for _ in range(40000)])
        ac = float(np.mean(x[10:] * x[:-10]) / np.var(x))
        out[k] = {"std": round(float(x.std()), 3), "ac10": round(ac, 3)}
    out["ok"] = (all(abs(out[k]["std"] - 1) < 0.08 for k in KINDS)
                 and abs(out["white"]["ac10"]) < 0.02
                 and out["ou"]["ac10"] > 0.4 and out["pink"]["ac10"] > 0.2)
    return out


def probe(kind: str, seed: int, n_ep: int = 20, E: int = 250,
          std: float = 0.5, out_path: str = "") -> list:
    """Stage A: noise-only births on body_probe's protocol."""
    from . import smell_sense as S
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    S.install("smell", seed)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w, dr = L.w, L.w.drives
    d, b = w.data, w.rover_bid
    w.observe()
    floor = {n: g for n, (g, _) in dr._food.items()
             if d.geom_xpos[g][2] <= 0.5}
    idle = np.r_[np.zeros(4), -1.0, -1.0, 0.0, 0.0]
    noise = BoutNoise(kind, seed * 6553 + 11)
    recs = []
    for ep in range(n_ep):
        while not any(dr.t >= dr._respawn_at[n] for n in floor):
            w.decide(idle)
            if w.died_this_decision:
                w.respawn()
        w.respawn()
        p0 = np.array(d.xpos[b][:2])
        m0 = sum(dr.ate_total.values())
        fed, dmin, far, path = None, 1e9, 0.0, 0.0
        born_d, last = None, p0
        for k in range(E):
            p = np.array(d.xpos[b][:2])
            ready = [np.array(d.geom_xpos[g][:2]) for n, g in floor.items()
                     if dr.t >= dr._respawn_at[n]]
            if ready:
                dn = min(float(np.linalg.norm(q - p)) for q in ready)
                dmin = min(dmin, dn)
                born_d = dn if born_d is None else born_d
            far = max(far, float(np.linalg.norm(p - p0)))
            path += float(np.linalg.norm(p - last))
            last = p
            w.decide(np.clip(std * noise.randn(8), -1.0, 1.0))
            if sum(dr.ate_total.values()) > m0:
                fed = k
                break
            if w.died_this_decision:
                break
        rec = {"kind": kind, "seed": seed, "std": std, "ep": ep, "fed": fed,
               "born_d": round(born_d or -1, 2), "d_min": round(dmin, 2),
               "far": round(far, 2), "path": round(path, 2)}
        recs.append(rec)
        print(json.dumps(rec), flush=True)
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
    fed = [r for r in recs if r["fed"] is not None]
    print(json.dumps({"SUMMARY": kind, "seed": seed, "std": std,
                      "fed": len(fed), "n": len(recs),
                      "far_med": float(np.median([r["far"] for r in recs])),
                      "d_min_med": float(np.median([r["d_min"] for r in recs]))}),
          flush=True)
    return recs


def install_noise(kind: str) -> None:
    """Stage B: every _Life4 built after this explores with `kind` noise
    (same seed formula as _Life's own rng). Per process; nothing on disk."""
    from .tests import lf_04_living_teaches as M
    base = M._Life4

    class _BoutLife(base):
        def __init__(self, seed, *a, **kw):
            super().__init__(seed, *a, **kw)
            self.rng = BoutNoise(kind, seed * 6553 + 11)

    M._Life4 = _BoutLife


def pilot(kind: str, smell_arm: str, seed: int, total: int, out: str) -> None:
    from . import smell_sense as S
    from . import wean_pilot
    S.install(smell_arm, seed)
    install_noise(kind)
    wean_pilot.main("null", seed, 0, total, out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "selftest":
        r = selftest()
        print(json.dumps(r))
        raise SystemExit(0 if r["ok"] else 1)
    if a[0] == "probe":
        probe(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 20,
              int(a[4]) if len(a) > 4 else 250,
              float(a[5]) if len(a) > 5 else 0.5,
              a[6] if len(a) > 6 else "")
    elif a[0] == "pilot":
        pilot(a[1], a[2], int(a[3]), int(a[4]), a[5])
    else:
        raise SystemExit(__doc__)
