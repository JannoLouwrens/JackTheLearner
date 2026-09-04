"""LG.03 blind-twin probe — the recorded diagnostic behind attempt 1's VOID.

NOT A SPEC. It writes nothing to the ledger and nothing depends on it. It exists
because `LG.03` attempt 1 (2026-09-04T17:20:27) returned **VOID** on its own
liveness gate — `blind_calib_rate` 0.583 +- 0.312, readings 1.00 / 0.50 / 0.25
across seeds 0/1/2 against `CALIB_MIN` 0.75 — and the file had PRE-REGISTERED
its repair in the open the day before: *"if it fires, the repair is a third
learner in the `max`, never a lower `CALIB_MIN`"* (LOOP_JOURNAL, 2026-09-04).

**This probe falsifies that repair, and the falsification is why it is kept.**
Five cheap deterministic learners on the identical demonstrations of the
identical calibration cell (`approach@block` on every seed, 96 demo rows), each
rolled out from the cell's own four starts:

    seed  planner_own  knn   ridge  knn1  wknn  ridge_lo | max2   max5
    0        1.00      0.25  0.75   0.75  0.50  0.75     | 0.75   0.75
    1        0.75      0.00  0.50   0.50  0.00  0.50     | 0.50   0.50
    2        0.75      0.50  0.75   0.50  0.50  0.50     | 0.75   0.75

`max5 == max2` on **every seed**. A third learner — a k=1 neighbour, a
ridge-coefficient-weighted metric that attacks the exact defect the shipped pair
was chosen for, a second penalty — buys **exactly zero**. The pre-registered
repair was wrong, and it was wrong for a reason no amount of learner-shopping
reaches.

**THE CAUSE IS THE `planner_own` COLUMN.** On seeds 1 and 2 the PRIVILEGED
planner — told the target, reading the object's world coordinates — reaches
`approach@block` on only **3 of its own 4 starts**. The demonstrations the twin
learns from are capped by the teacher, and a clone cannot be asked to exceed
what was demonstrated. The gate then requires 0.75 = 3 of 4 *absolutely*, so on
those seeds it demands the student reproduce **every** success the teacher had,
perfectly, or the run is void. That bar is not a liveness proof of the learner;
it is a joint test of the learner AND of whether the arbitrarily-fixed
calibration cell happened to be one the servo aced. `avoid`'s own predicate in
the parent file already knows this — it is start-relative *"because an absolute
one is satisfied by standing still far away"* — and the liveness bar is the same
mistake one surface over.

**AND THE SECOND HALF, WHICH IS WHY THE REPAIR IS NOT A ONE-LINE RELATIVISATION.**
Even a teacher-relative bar (0.75 x 0.75 = 0.5625) still VOIDs seed 1, whose best
clone reads 0.50. The cheap-deterministic-clone family — no optimiser, no
training schedule, deliberately, so that nothing about the verdict depends on
one — cannot demonstrate itself alive in W0's 80-number observation at this
horizon on all three seeds. Both halves of the repair are therefore live
questions and both are routed: `docs/REVIEW_QUEUE.md`,
`lg03-blind-twin-cannot-prove-itself-alive`.

APPROXIMATION, DECLARED. `_experiment` runs 160 planner rollouts before it
reaches the calibration fit, so W0's interoceptive channels enter these demos in
a slightly different state; the planner's actions depend only on pose and
velocity, so `Y` is unaffected and only a few of `X`'s 80 columns drift. That is
why seed 0 reads 0.75 here and 1.00 in the registered run. The phenomenon does
not move: two of three seeds under the bar, `max5 == max2` on all three.

    /data/venvs/jackthelearner/bin/python -m experiments.tests.lg03_blind_twin_probe
"""
from __future__ import annotations

import numpy as np

from . import lg_03_command_cells_necessary as L

#: every candidate raced, in the order the table above prints them
KINDS = ("knn", "ridge", "knn1", "wknn", "ridge_lo")


class _Cand:
    """All five candidates over one standardisation of one set of demos.

    `knn`/`ridge` are byte-for-byte the shipped `_Blind`'s two learners. The
    three additions are the honest candidates for a third seat in the `max`:
    `knn1` (no neighbour averaging, in case smoothing five actions blurs a
    servo), `wknn` (the Euclidean metric reweighted by ridge-coefficient
    magnitude, which attacks the diagnosed defect — placebo and silent columns
    eating the metric — directly), and `ridge_lo` (the same learner, a
    hundredth of the penalty).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        self.Xn = (X - self.mu) / self.sd
        self.Y = Y
        Z = np.hstack([self.Xn, np.ones((len(self.Xn), 1))])
        self.W = {lam: np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
                  for lam in (1.0, 0.01)}
        w = np.abs(self.W[1.0][:-1]).mean(axis=1)
        self.dw = w / (w.max() + 1e-12)

    def _knn(self, q, k, weights=None):
        Xn = self.Xn if weights is None else self.Xn * weights
        qq = q if weights is None else q * weights
        k = min(k, len(Xn))
        idx = np.argpartition(np.linalg.norm(Xn - qq, axis=1), k - 1)[:k]
        return self.Y[idx].mean(axis=0)

    def _ridge(self, q, lam):
        return np.clip(np.concatenate([q, [1.0]]) @ self.W[lam], -1.0, 1.0)

    def policy(self, kind: str):
        def p(v, xy, vel, t):
            q = (v - self.mu) / self.sd
            if kind == "knn":
                return self._knn(q, L.KNN_K)
            if kind == "knn1":
                return self._knn(q, 1)
            if kind == "wknn":
                return self._knn(q, L.KNN_K, self.dw)
            if kind == "ridge":
                return self._ridge(q, L.RIDGE_LAMBDA)
            if kind == "ridge_lo":
                return self._ridge(q, 0.01)
            raise KeyError(kind)
        return p


def probe(seed: int) -> dict:
    """One seed: rebuild LG.03's calibration fixture and race all five on it."""
    w = L.W0(seed=seed, j0=L.J0, alpha=L.ALPHA)
    rng = np.random.RandomState(seed * 7919 + 101)
    legal = w.legal_spawns()
    objs = L._live_objects(w)
    starts = {}
    for o in objs:              # consume rng exactly as `_experiment` does
        s = L._starts_for(legal, objs[o][0], rng)
        for v in L.VERBS:
            starts[L._cell(v, o)] = s

    cc = L._cell("approach", sorted(objs)[0])
    oxy, ogids = objs[cc.split("@")[1]]
    X, Y, own = [], [], []
    for st in starts[cc]:
        pol = L._planner("approach", oxy, st)
        rec: list = []

        def taped(vv, xy, vel, t, _p=pol, _r=rec):
            a = _p(vv, xy, vel, t)
            _r.append((vv, a))
            return a

        own.append(L._satisfies("approach", L._rollout(w, st, taped, oxy, ogids)))
        X += [vv for vv, _ in rec]
        Y += [a for _, a in rec]

    cand = _Cand(np.array(X), np.array(Y))
    out = {k: float(np.mean([
        L._satisfies("approach", L._rollout(w, st, cand.policy(k), oxy, ogids))
        for st in starts[cc]])) for k in KINDS}
    out["planner_own"] = float(np.mean(own))
    out["cell"], out["rows"] = cc, float(len(Y))
    return out


def main() -> None:
    print(f"{'seed':>4} {'planner':>8} " + " ".join(f"{k:>9}" for k in KINDS)
          + f" {'max2':>6} {'max5':>6}")
    for seed in (0, 1, 2):
        r = probe(seed)
        m2 = max(r["knn"], r["ridge"])
        m5 = max(r[k] for k in KINDS)
        print(f"{seed:>4} {r['planner_own']:>8.2f} "
              + " ".join(f"{r[k]:>9.2f}" for k in KINDS)
              + f" {m2:>6.2f} {m5:>6.2f}"
              + ("   <- max5 == max2" if abs(m5 - m2) < 1e-9 else ""))


if __name__ == "__main__":
    main()
