"""ps02_fire_probe_diag.py — does ruling 12's cure (no fire) fix PS.02? (11-17)

NOT a spec and writes no ledger. Pilot seed 90 only (PS.02's own pilot seed,
disjoint from the registered 0/1/2); the spec file is not edited.

PS.02 v2 FAILED at `3cbbcecf` (probe R^2 0.317 +- 0.420, bar 0.50). Ruling 12
named the cause as the fire (a random walk passing it prolongs a life
unpredictably) and ordered v3 = the probe's 16 cold lives in a world with no
fire. This script measures that cure BEFORE the spec is changed, and asks
where the information is:

  world    fire as v2 (fire_dist drawn) | none (fire_dist = 1e9 m: ambient()
           is exactly t_cold; checked as felt-minus-cold == 0)
  probes   v2's RFF+ridge on all 82 features (the registered instrument);
           the same RFF on the 2 thermal floats only; the same on the 80
           non-thermal (the control's input); thermal.time_to_lethal_s on the
           two felt floats decoded (the closed form — an upper bound);
           a fixed small MLP (82-64-64-1 tanh, Adam 1e-3, wd 1e-4, 2000
           full-batch steps, seed RFF_SEED) on all 82 / on the 80.

RESULT 11-17 21:36, this script (seed 90; 16 cold lives, 0 censored, deaths
6.4-36 s; felt-minus-cold 0 with no fire, 3.51 degC with it):
  world   RFF-82  RFF-therm2  RFF-80   closed-form  MLP-82  MLP-80
  none    0.290   0.998       -0.825   1.000        0.571   -7.15
  fire    0.309   0.994       -0.939   0.994        0.578   -6.92
(MLP-82 moved < 0.01 between two runs: torch CPU nondeterminism.)
Reading: removing the fire does not move the probe (0.29 vs 0.31; MLP 0.571
vs 0.578), so on the pilot the fire is NOT why PS.02 fails. The time to
freezing is in his senses almost perfectly (two floats -> R^2 0.998; the law
-> 1.000), and the generic probe over all 82 inputs recovers 0.29 of it: 80
irrelevant standardized inputs dominate the random projections (RFF scale
1/sqrt(d)). The FAIL is the instrument's sample efficiency (10 training
lives), not the world's legibility. A probe chosen by reading this pilot
until it passes would be a forking path; the fix needs a rule first (INBOX).
"""
from __future__ import annotations

import sys

import numpy as np

SEED = 90


def _mlp(Xtr, ytr, Xte, seed):
    import torch
    torch.set_num_threads(1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    y = torch.tensor((ytr - ym) / ys, dtype=torch.float32)[:, None]
    for _ in range(2000):
        opt.zero_grad()
        ((net(X) - y) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        Z = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
        return net(Z).numpy()[:, 0] * ys + ym


def main(seed: int = SEED) -> None:
    import experiments.tests.ps_02_cold_is_felt as P
    from experiments import thermal
    for label, fd in (("none", 1e9), ("fire", None)):
        P._CACHE.clear()
        orig = P._run_life

        def life(*a, _o=orig, **k):            # cold lives only: no fire_dist given
            if fd is not None and "fire_dist" not in k:
                k["fire_dist"] = fd
            return _o(*a, **k)
        P._run_life = life
        try:
            d = P._collect(seed)
        finally:
            P._run_life = orig
        cold = P._uncensored(d["cold"])
        n = cold[0]["X"].shape[1]
        tr, te = cold[:-P.N_TEST], cold[-P.N_TEST:]
        cat = lambda rs, cols: np.concatenate([r["X"][:, cols] for r in rs])
        ytr = np.concatenate([r["y"] for r in tr])
        yte = np.concatenate([r["y"] for r in te])
        allc, th, nt = list(range(n)), [n - 2, n - 1], list(range(n - 2))
        rff = lambda c: P._r2(yte, P._fit_predict(cat(tr, c), ytr, cat(te, c)))
        mlp = lambda c: P._r2(yte, _mlp(cat(tr, c), ytr, cat(te, c), P.RFF_SEED))
        X2 = cat(te, th).astype(float)
        tb = X2[:, 0] * thermal.CORE_SPAN + thermal.TB_LETHAL
        tf = X2[:, 1] * thermal.SKIN_SPAN + thermal.T_NEUTRAL
        cf = np.array([thermal.time_to_lethal_s(a, b) for a, b in zip(tb, tf)])
        felt = max(abs(r["t_eff_end"] - r["t_cold"]) for r in d["cold"])
        print(f"{label}: censored {16 - len(cold)} felt-minus-cold {felt:.3g} "
              f"RFF82 {rff(allc):.3f} RFFtherm2 {rff(th):.3f} RFF80 {rff(nt):.3f} "
              f"closed {P._r2(yte, cf):.3f} MLP82 {mlp(allc):.3f} MLP80 {mlp(nt):.3f}",
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else SEED)
