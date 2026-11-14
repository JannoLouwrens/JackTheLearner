"""LAB — THE DRIVE IN HIS OWN FRAME: his push is a WORLD-frame force, his nose
is a BODY-frame sense, and nothing he senses tells him which way he faces
(no ledger row; a pilot).

Seat: World (BY VERDICT — W0's drive law, `w0.py`: "a world-frame horizontal
force on the torso") with Proprioception & balance (VACANT) as the rival
repair. END-FIT rows served: 1 (a human-like body commands its muscles in its
own frame; a policy that learned world-frame pushes has nothing to hand a
humanoid), 4 (approach from afar is the first link of every chain). Closes
none: no weight is added, frozen or borrowed — only what a push MEANS changes.

NEW REASON (the one `sense_lookahead`'s post-mortem asked for: "heading — the
push is world-frame, the nose body-frame"). Measured before this file existed
(s90, 588 one-second babble bouts, /tmp/frame_diag.py): he lies toppled in
97.6 % of bouts, his heading is uniform over the circle (resultant length
.089) and turns a median .29 rad per bout; proprio carries no heading. The
change in smell over a bout correlates .50 with (L-R nostril difference x
push along his OWN lateral axis) and .18 / .03 with the same product on the
world's x / y. So "which push raises smell" is a fixed, learnable function of
what he senses only in his own frame; in the world frame it rotates under
him unseen. That would explain three kills at once: smell unused by the
reactive learner (x3), weaning, and the lookahead's f.

FRAMES (per process; `w0.py` on disk untouched, no certificate moves)
  world    HOLDER: W0 as shipped — action[6:8] is a world x,y force.
  ego      CHALLENGER: action[6:8] is (forward, left) in the body's own
           heading — the same heading his nose is mounted on; rotated into
           the world once per decision. Same 600 N, same gate, same clip.
  compass  RIVAL REPAIR: world-frame push, plus a 2-float heading sense
           (sin, cos yaw) — the vestibular answer; he must learn the rotation.
  scram    CONTROL THAT MUST FAIL: `ego`, but the rotation angle is drawn
           uniformly at random every decision — a frame tied to nothing.

STAGE 1 — `fcheck` (reward-free, no chooser): BABBLE decisions of
random-repeat pushes (hold 5), sense_lookahead's own forward model `Mind.f`
fitted on the first 80 % of bouts. Read on the held-out 20 %:
  share = 1 - MSE(f, true push) / MSE(f, push permuted across held-out rows)
on the change of total smell (the 8 nostril floats) — the fraction of f's
smell prediction that the PUSH carries. DECLARED BEFORE ANY SEED IS READ
(seeds 90 and 94, 5000 decisions): `ego` passes stage 1 iff share >= .10 on
BOTH seeds AND share(ego) > share(world) on both AND share(scram) <
share(ego) / 2 on both. Else the bet is DEAD (heading is not what broke f).
`compass` is read, not gated; it enters stage 2 only with share >= .10 on both.

STAGE 2 — `run`: sense_lookahead's four arms, verbatim, with the frame
installed (babble / want / told / shuf; explore_bout's ruler; seeds 90 + 94,
20 test births each after 5k babble). RULE, declared now: sense_lookahead's
own — X in {want, told} is ALIVE iff it feeds more births than `babble` on
BOTH seeds and by >= 8 of 40, AND more than `shuf` likewise — all four arms
run IN THE EGO FRAME — AND X in the ego frame beats the SAME arm in the
world frame already on disk (told 6/40, want 4/40) by >= 8 of 40. `told`
alone ALIVE = the frame lets him climb when the want is given; the learned
want is then the next race, not a win. Read, not gated: `toward`.

    nice -n 19 $PY -m experiments.ego_drive fcheck <frame> <seed> [babble]
    nice -n 19 $PY -m experiments.ego_drive run <frame> <arm> <seed> [babble] [n_test]
    $PY -m experiments.ego_drive tally1 | tally2 <frame>
    $PY -m experiments.ego_drive selftest
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

FRAMES = ("world", "ego", "compass", "scram")
OUT = "/data/tmp/ego_drive"
SHARE_MIN = 0.10
WORLD_ON_DISK = {"told": 6, "want": 4}      # sense_lookahead, s90+s94, /40


def rotate(a: np.ndarray, yaw: float) -> np.ndarray:
    """(forward, left) in a body facing `yaw` -> world (x, y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]])


def install(frame: str, seed: int) -> None:
    """Patch W0 for this process. Call AFTER smell_sense.install."""
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; one of {FRAMES}")
    from . import w0 as W
    if getattr(W.W0, "_ego_frame", None) is not None:
        raise RuntimeError("a frame is already installed in this process")
    W.W0._ego_frame = frame
    if frame == "world":
        return
    orig_decide, orig_observe = W.W0.decide, W.W0.observe

    def _yaw(self) -> float:
        m = np.array(self.data.xmat[self.rover_bid]).reshape(3, 3)
        return math.atan2(m[1, 0], m[0, 0])       # the nose's own heading

    if frame == "compass":
        def observe(self):
            obs = orig_observe(self)
            y = _yaw(self)
            obs["heading"] = np.array([math.sin(y), math.cos(y)],
                                      dtype=np.float32)
            return obs
        W.W0.observe = observe
        return

    def decide(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        a = a.copy()
        if frame == "ego":
            y = _yaw(self)
        else:                                      # scram: a frame of nothing
            if not hasattr(self, "_ego_rng"):
                self._ego_rng = np.random.RandomState(4_441 + int(seed))
            y = float(self._ego_rng.uniform(-math.pi, math.pi))
        a[6:8] = rotate(a[6:8], y)
        return orig_decide(self, a)

    W.W0.decide = decide


def _world(frame: str, seed: int):
    import torch

    from . import smell_sense as S
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    torch.set_num_threads(1)
    S.install("smell", seed)
    install(frame, seed)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    return _Life4(seed, j0, alpha, train=False).w


def babble_bouts(frame: str, seed: int, babble: int = 5000):
    """Random-repeat babble in `frame`: (bouts, smell slice, obs width)."""
    from . import sense_lookahead as SL
    from . import smell_sense as S
    w = _world(frame, seed)
    dr = w.drives
    obs = w.observe()
    keys = list(obs)
    off = int(sum(np.asarray(obs[k]).size for k in keys[:keys.index("smell")]))
    smell = slice(off, off + S.SMELL_DIM)
    dim = len(SL._vec(obs))
    rng = np.random.RandomState(seed * 6553 + 11)
    bouts, k, since = [], 0, 0
    a_prev = np.zeros(8)
    w.respawn()
    while k < babble:
        a = np.clip(SL.STD * rng.randn(8), -1.0, 1.0)
        m0 = sum(dr.ate_total.values())
        xs, dead = [], False
        for _ in range(SL.HOLD):
            xs.append(SL._vec(w.observe()))
            w.decide(a)
            k += 1
            since += 1
            if w.died_this_decision:
                dead = True
                break
        ate = sum(dr.ate_total.values()) > m0
        if not dead and not ate:               # a meal removes the smell
            bouts.append((xs, a_prev, a, SL._vec(w.observe())))
        a_prev = a
        if dead or since >= SL.E:              # explore_bout's 250-decision births
            w.respawn()
            a_prev, since = np.zeros(8), 0
    return bouts, smell, dim, k


def fcheck(frame: str, seed: int, babble: int = 5000) -> dict:
    from . import sense_lookahead as SL
    bouts, smell, dim, k = babble_bouts(frame, seed, babble)
    cut = int(0.8 * len(bouts))
    mind = SL.Mind(dim, smell, seed)
    for b in bouts[:cut]:
        mind.add_bout(*b)
    mind.fit(800)
    held = SL.Mind(dim, smell, seed)
    for b in bouts[cut:]:
        held.add_bout(*b)
    X, AP, A, H = (np.asarray(held.X), np.asarray(held.AP),
                   np.asarray(held.A), np.asarray(held.H))
    true = np.asarray(held.D)[:, smell][:, :8].sum(1)

    def pred(acts):
        import torch
        with torch.no_grad():
            o = mind.f(mind._inp(X, AP, acts, H)).numpy() * mind.dsd
        return o[:, smell][:, :8].sum(1)

    prng = np.random.RandomState(17 + seed)
    mse_true = float(np.mean((pred(A) - true) ** 2))
    mse_perm = float(np.mean([np.mean((pred(A[prng.permutation(len(A))])
                                       - true) ** 2) for _ in range(20)]))
    rec = {"frame": frame, "seed": seed, "babble": k, "bouts": len(bouts),
           "held_rows": int(len(true)),
           "share": round(1.0 - mse_true / mse_perm, 4),
           "r2": round(1.0 - mse_true / float(np.var(true)), 4),
           "corr": round(float(np.corrcoef(pred(A), true)[0, 1]), 4)}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fcheck_{frame}_s{seed}.json", "w") as f:
        json.dump(rec, f)
    print(json.dumps(rec), flush=True)
    return rec


def tally1(dirname: str = OUT) -> bool:
    res = {}
    for p in sorted(glob.glob(f"{dirname}/fcheck_*_s*.json")):
        r = json.load(open(p))
        res[(r["frame"], r["seed"])] = r
        print(f"{r['frame']:8s} s{r['seed']}: share {r['share']:+.3f}  "
              f"r2 {r['r2']:+.3f}  corr {r['corr']:+.3f}  "
              f"held rows {r['held_rows']}")
    seeds = sorted({s for _, s in res})
    need = [(f, s) for f in ("world", "ego", "scram") for s in seeds]
    if len(seeds) < 2 or any(x not in res for x in need):
        print("INCOMPLETE: no verdict")
        return False
    sh = lambda f, s: res[(f, s)]["share"]
    ok = all(sh("ego", s) >= SHARE_MIN and sh("ego", s) > sh("world", s)
             and sh("scram", s) < sh("ego", s) / 2 for s in seeds)
    print("STAGE 1: ego " + ("PASSES" if ok else "DEAD"))
    if all(("compass", s) in res for s in seeds):
        c = all(sh("compass", s) >= SHARE_MIN for s in seeds)
        print("compass " + ("enters stage 2" if c else "stays out"))
    return ok


def run(frame: str, arm: str, seed: int, babble: int = 5000,
        n_test: int = 20) -> None:
    from . import sense_lookahead as SL
    from . import smell_sense as S
    S.install("smell", seed)
    install(frame, seed)
    S.install = lambda *a, **k: None           # SL.run installs smell again
    SL.run(arm, seed, babble, n_test, f"{OUT}/{frame}/{arm}_s{seed}.jsonl")


def tally2(frame: str = "ego") -> None:
    from . import sense_lookahead as SL
    res = SL.tally(f"{OUT}/{frame}")
    seeds = sorted({s for _, s in res})
    if len(seeds) < 2 or any((a, s) not in res
                             for a in SL.ARMS for s in seeds):
        return
    for x, base in WORLD_ON_DISK.items():
        tot = sum(res[(x, s)][0] for s in seeds)
        print(f"{x} in {frame} {tot}/40 vs world-frame {base}/40: "
              f"{'clears' if tot - base >= SL.MARGIN else 'does NOT clear'} "
              f"the +{SL.MARGIN} bar")


def selftest() -> None:
    """rotate() is a rotation; ego with yaw 0 is the world frame; a quarter
    turn sends forward to world +y (checks that can fail)."""
    a = np.array([0.3, -0.7])
    assert np.allclose(rotate(a, 0.0), a)
    assert np.allclose(rotate(np.array([1.0, 0.0]), math.pi / 2), [0.0, 1.0])
    for y in (-2.0, 0.4, 3.0):
        assert abs(np.linalg.norm(rotate(a, y)) - np.linalg.norm(a)) < 1e-12
        assert np.allclose(rotate(rotate(a, y), -y), a)
    print("SELFTEST OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "fcheck":
        fcheck(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 5000)
    elif a[0] == "run":
        run(a[1], a[2], int(a[3]), int(a[4]) if len(a) > 4 else 5000,
            int(a[5]) if len(a) > 5 else 20)
    elif a[0] == "tally1":
        raise SystemExit(0 if tally1() else 1)
    elif a[0] == "tally2":
        tally2(a[1] if len(a) > 1 else "ego")
    elif a[0] == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
