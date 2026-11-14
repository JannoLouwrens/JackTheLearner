"""LAB — GAIN-FIELD FORWARD MODEL: he predicts what a push does to his senses
with a BILINEAR map (every sense x every push), not an MLP (no ledger row).

Seat: Deliberation (the slow path) — VACANT. This re-opens `sense_lookahead`
(DEAD 11-13: told 6/40, want 4/40, shuffled 7/40) with the reason its
post-mortem demanded be shown first: f's held-out push->smell error against
a shuffled net. END-FIT rows 4 (thinking ahead, raced), 3 (`told` = advice as
a want), 1; closes none — the map is plastic, learned from his own babbling
from zero, saved like any weight.

WHERE IT CAME FROM (`ego_drive.py`, DEAD at stage 1 the same day). The MLP f
puts .086 / .044 of its held-out smell prediction on the push (s90 / s94).
Post hoc on s90's saved bouts: one hand feature (nostril difference x push
on his lateral axis) correlates .49 with the change in smell on held-out
bouts, and a ridge over ALL (sense x push) products reads the push at share
.31 in the WORLD frame as shipped (.16 in the ego frame, differenced nose
only). What a push does depends on where he is and how he lies — a product
of state and action — and a 2-layer ReLU net fed 130 inputs does not find a
product in 4k correlated rows. Parietal gain fields are the brain's answer
to the same problem (Andersen 1985; Salinas & Abbott 2001).

THE ARM. `GainMind` = sense_lookahead.Mind with f replaced by
    d(senses) = W . [x, a_prev, a*h, x (x) a*h]        (x normalised, |x| <= 5)
fitted by ridge on standardised columns (lambda LAM x rows) — the batch form of recursive least
squares, refitted after every birth. `want`, the chooser, the candidates,
the babble and the ruler are sense_lookahead's, untouched.

STAGE 1 — `fcheck` (reward-free): babble 5000 decisions in W0 as shipped,
fit on the first 80 % of bouts, read the push's share of the held-out
smell prediction (ego_drive.fcheck's reading, same code). DECLARED BEFORE
ANY SEED IS READ WITH THIS MODEL (seeds 91 and 94; 90 was the exploration
seed and is not counted): PASS iff share(gain) >= .10 on BOTH seeds AND
share(gain) > share(mlp) on the same bouts on both AND the control `gainshuf`
(same map fitted on smell changes permuted across rows — must fail) has
share < share(gain) / 2 on both. Else DEAD.

STAGE 2 — `run`: sense_lookahead's arms told / want / shuf with GainMind,
seeds 90 + 94, 20 test births after 5k babble (`babble` never consults f:
its rows on disk stand, 2/3 = 5/40). RULE, declared now: sense_lookahead's
own — X in {want, told} ALIVE iff it feeds more births than `babble` on
BOTH seeds and by >= 8 of 40, AND more than gain-`shuf` likewise — AND X
beats the same arm under the MLP f on disk (told 6/40, want 4/40) by >= 8
of 40. `told` alone ALIVE = the hand works, the learned want is next.

    nice -n 19 $PY -m experiments.gain_field fcheck <mlp|gain|gainshuf> <seed> [babble]
    nice -n 19 $PY -m experiments.gain_field run <arm> <seed> [babble] [n_test]
    $PY -m experiments.gain_field tally1 | tally2 | selftest
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from . import sense_lookahead as SL

OUT = "/data/tmp/gain_field"
LAM = 0.3
XCLIP = 5.0
SHARE_MIN = 0.10
MLP_ON_DISK = {"told": 6, "want": 4}          # sense_lookahead, s90+s94, /40
BABBLE_ON_DISK = {90: 2, 94: 3}
MODELS = ("mlp", "gain", "gainshuf")


class GainMind(SL.Mind):
    """Mind with a bilinear (sense x push) forward model, ridge-fitted."""

    W = None
    dmu = None
    fmu = fsd = None

    def _feat(self, X, AP, A, H):
        xn = np.clip((np.asarray(X, dtype=float) - self.mu) / self.sd,
                     -XCLIP, XCLIP)
        ah = np.asarray(A, dtype=float) * np.asarray(H, dtype=float
                                                     ).reshape(-1, 1)
        prod = (xn[:, :, None] * ah[:, None, :]).reshape(len(xn), -1)
        F = np.concatenate([xn, np.asarray(AP, dtype=float), ah, prod], 1)
        if self.fmu is None:                     # column scales, fixed at first fit
            self.fmu, self.fsd = F.mean(0), F.std(0) + 1e-6
        return (F - self.fmu) / self.fsd

    def fit(self, steps: int):
        if len(self.X) < 64:
            return
        X, D = np.asarray(self.X), np.asarray(self.D)
        if not self.fitted:                      # scales fixed after babble
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
            self.dsd = D.std(0) + 1e-6
            self.fitted = True
        Dn = D / self.dsd
        if self.shuffle:                         # the control: break push->smell
            perm = np.random.RandomState(7).permutation(len(Dn))
            Dn = Dn.copy()
            Dn[:, self.smell] = Dn[perm][:, self.smell]
        F = self._feat(X, self.AP, self.A, self.H)
        self.dmu = Dn.mean(0)
        G = F.T @ F + LAM * len(F) * np.eye(F.shape[1])
        self.W = np.linalg.solve(G, F.T @ (Dn - self.dmu))
        self._fit_want()

    def predict(self, X, AP, A, H):
        return (self._feat(X, AP, A, H) @ self.W + self.dmu) * self.dsd

    def imagine(self, x, a_prev, cands):
        n = len(cands)
        return self.predict(np.repeat(x[None], n, 0),
                            np.repeat(a_prev[None], n, 0), cands, np.ones(n))


def _share(model: str, bouts, smell, dim, seed: int) -> dict:
    """ego_drive.fcheck's reading, for either model, on given bouts."""
    cut = int(0.8 * len(bouts))
    cls = SL.Mind if model == "mlp" else GainMind
    mind = cls(dim, smell, seed, shuffle_smell=(model == "gainshuf"))
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
        if model == "mlp":
            import torch
            with torch.no_grad():
                o = mind.f(mind._inp(X, AP, acts, H)).numpy() * mind.dsd
        else:
            o = mind.predict(X, AP, acts, H)
        return o[:, smell][:, :8].sum(1)

    prng = np.random.RandomState(17 + seed)
    mse_true = float(np.mean((pred(A) - true) ** 2))
    mse_perm = float(np.mean([np.mean((pred(A[prng.permutation(len(A))])
                                       - true) ** 2) for _ in range(20)]))
    return {"model": model, "seed": seed, "bouts": len(bouts),
            "held_rows": int(len(true)),
            "share": round(1.0 - mse_true / mse_perm, 4),
            "r2": round(1.0 - mse_true / float(np.var(true)), 4),
            "corr": round(float(np.corrcoef(pred(A), true)[0, 1]), 4)}


def fcheck(model: str, seed: int, babble: int = 5000) -> dict:
    from . import ego_drive as G
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; one of {MODELS}")
    bouts, smell, dim, _ = G.babble_bouts("world", seed, babble)
    rec = _share(model, bouts, smell, dim, seed)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fcheck_{model}_s{seed}.json", "w") as f:
        json.dump(rec, f)
    print(json.dumps(rec), flush=True)
    return rec


def tally1(seeds=(91, 94)) -> bool:
    res = {}
    for p in sorted(glob.glob(f"{OUT}/fcheck_*_s*.json")):
        r = json.load(open(p))
        res[(r["model"], r["seed"])] = r
        print(f"{r['model']:8s} s{r['seed']}: share {r['share']:+.3f}  "
              f"r2 {r['r2']:+.3f}  corr {r['corr']:+.3f}")
    if any((m, s) not in res for m in MODELS for s in seeds):
        print("INCOMPLETE: no verdict")
        return False
    sh = lambda m, s: res[(m, s)]["share"]
    ok = all(sh("gain", s) >= SHARE_MIN and sh("gain", s) > sh("mlp", s)
             and sh("gainshuf", s) < sh("gain", s) / 2 for s in seeds)
    print("STAGE 1: gain-field " + ("PASSES" if ok else "DEAD"))
    return ok


def run(arm: str, seed: int, babble: int = 5000, n_test: int = 20) -> None:
    SL.Mind = GainMind
    SL.run(arm, seed, babble, n_test, f"{OUT}/{arm}_s{seed}.jsonl")


def tally2() -> None:
    res = {}
    for p in sorted(glob.glob(f"{OUT}/*_s*.jsonl")):
        rows = [json.loads(x) for x in open(p)]
        te = [r for r in rows if r["phase"] == "test"]
        if len(te) >= SL.N_TEST:
            res[(te[0]["arm"], te[0]["seed"])] = (
                sum(r["fed"] is not None for r in te),
                float(np.mean([r["toward"] for r in te])), rows[0])
    for (a, s), v in sorted(res.items()):
        print(f"{a:5s} s{s}: fed {v[0]}/{SL.N_TEST}  toward {v[1]:+.3f}  "
              f"want smell share {v[2]['want_smell_share']:.2f}")
    seeds = (90, 94)
    if any((a, s) not in res for a in ("told", "want", "shuf") for s in seeds):
        print("INCOMPLETE: no verdict")
        return
    tot = {a: sum(res[(a, s)][0] for s in seeds)
           for a in ("told", "want", "shuf")}
    tb = sum(BABBLE_ON_DISK.values())
    print(f"sum told {tot['told']}/40  want {tot['want']}/40  "
          f"shuf {tot['shuf']}/40  babble(on disk) {tb}/40")
    alive = {}
    for x in ("want", "told"):
        alive[x] = (all(res[(x, s)][0] > BABBLE_ON_DISK[s]
                        and res[(x, s)][0] > res[("shuf", s)][0]
                        for s in seeds)
                    and tot[x] - tb >= SL.MARGIN
                    and tot[x] - tot["shuf"] >= SL.MARGIN
                    and tot[x] - MLP_ON_DISK[x] >= SL.MARGIN)
        print(f"{x}: {'ALIVE' if alive[x] else 'DEAD'}")
    print("BET " + ("ALIVE" if any(alive.values()) else "DEAD"))


def selftest() -> None:
    """A world where the push moves sense 0 only THROUGH a product with
    sense 5 (d x0 = x5 * a6): the gain map must put the push to use, its
    shuffled twin must not (share >= .5 vs <= .1)."""
    rng = np.random.RandomState(0)
    bouts = []
    for _ in range(1500):
        x = rng.randn(14)
        a = np.clip(0.5 * rng.randn(8), -1, 1)
        xe = x + 0.05 * rng.randn(14)
        xe[0] = x[0] + x[5] * a[6]
        bouts.append(([x], np.zeros(8), a, xe))
    got = {m: _share(m, bouts, slice(0, 12), 14, 0)["share"]
           for m in ("gain", "gainshuf")}
    assert got["gain"] >= 0.5 and got["gainshuf"] <= 0.1, got
    print(f"SELFTEST OK (push share: gain {got['gain']}, "
          f"shuffled {got['gainshuf']})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "fcheck":
        fcheck(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 5000)
    elif a[0] == "run":
        run(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 5000,
            int(a[4]) if len(a) > 4 else 20)
    elif a[0] == "tally1":
        raise SystemExit(0 if tally1() else 1)
    elif a[0] == "tally2":
        tally2()
    elif a[0] == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
