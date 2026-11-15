"""LAB — RUN AND TUMBLE: he keeps a push while what he wants is rising and
draws a new one when it is not. No forward model, no map, no heading (no
ledger row; a pilot).

Seat: Deliberation (the slow path) — VACANT, "the reactive actor holds it by
default". This is a challenger of the other kind: a FAST path that needs no
imagination. If it climbs, any slow path entered for W0 must beat it, not
babble. END-FIT rows served: 1 (a fast loop under a slow chooser), 3 (`told`
= advice arriving as a WANT, never as an action), 4 (the want is the hook a
late payoff hangs on); closes none — the only learned part is a linear want
inside him, learned from his own meals from zero, saved like any weight.

WHY IT MIGHT WIN WHERE FOUR ARMS DIED. sense_lookahead, ego_drive and
gain_field all asked him to PREDICT what a push does to his nose before
pushing (held-out r2 < 0 for both models; told 4/40 and 6/40 vs babble
5/40). Biology climbs gradients without predicting: E. coli runs while the
attractant rises and tumbles when it falls; C. elegans pirouettes on dC/dt <
0 (Berg & Brown 1972; Pierce-Shimomura 1999). The rule needs one bit per
bout — did it get better — and that bit is measured, not modelled. His push
is world-frame and holds its direction while repeated (ego_drive), so a kept
push is a kept bearing even though he lies toppled and senses no heading.
explore_bout's pink noise is the same persistence WITHOUT the bit: 6/40.

THE ARM. Every HOLD decisions (one bout): d = want(senses now) - want(senses
at the bout's start). d > 0 -> repeat the push; else draw a fresh one from
the babble distribution. That is all.

ARMS (same seed, same births, same push distribution; only the bit differs)
  babble    NULL: a fresh push every bout (sense_lookahead's babble).
  told      the want is GIVEN: "more smell" (L+R food-nose sum, the same
            reading sense_lookahead's `told` used). Upper bound on the rule.
  yoked     CONTROL THAT MUST FAIL (stage 1): the same rule, but the bit is a
            coin with `told`'s own measured keep-rate on that seed — the same
            persistence, no information.
  want      the want is LEARNED: logistic "a meal within MEAL_H decisions"
            over ALL his senses, from his own meals. A want cannot start
            without a meal (gain_field post-mortem: s94 had none in 5k
            babble), so phase 1 is a nursery: NURSE babble decisions in
            which every other birth is placed by the mother within R_NURSE
            of ready floor food (wean_pilot's adult — a spawn rule, never
            his action). Then she leaves: test births are W0's own.
  wantshuf  CONTROL THAT MUST FAIL (stage 2): `want`, same nursery, same
            births, but the meal labels permuted across decisions before
            every fit — a want that learned nothing about meals.

THE RULER — sense_lookahead's, verbatim: W0's own birth stream, a floor food
ready at birth, E = 250 decisions, fed = first meal; N_TEST births read.
Published on it (s90+s94): babble 5/40, white 4/40, pink 6/40, lookahead
told 6/40 (MLP) and 4/40 (gain field), perfect-map oracle 38/40.

RULE, DECLARED BEFORE ANY BIRTH IS READ WITH THIS RULE ON THESE SEEDS
(fresh seeds 140 and 141, 20 test births each; 90 is the exploration seed
and is never counted):
  STAGE 1: `told` ALIVE iff it feeds more births than `babble` on BOTH seeds
  and by >= 8 of 40 in sum, AND more than `yoked` likewise. Else the bet is
  DEAD: a measured bit does not climb either, and the body or the nose is
  the next suspect, not the learner.
  STAGE 2 (run only if stage 1 is ALIVE): `want` ALIVE iff it feeds more
  test births than `babble` on BOTH seeds and by >= 8 of 40, AND more than
  `wantshuf` likewise. `told` alone ALIVE = the rule works and his meals do
  not yet teach the want at this dose (say so; not a win for the seat).
Read only, not gated: `toward` (mean cosine of each bout's displacement with
the bearing of the nearest ready floor food), keep-rate, median d_min.

RESULT 11-15 11:15 — STAGE 1 DEAD by the rule above: fed told 5/3 = 8/40,
babble 2/4 = 6/40, yoked 3/2 = 5/40 (lead +2 < 8; s141 told < babble).
Read only: `toward` told +.442/+.196 vs babble +.024/+.032, yoked
+.023/+.107 — the first arm in five that points him at food. Post hoc on
s90 (never counted): unfed told births end nearer than born (4.2 -> 2.9 m
median; babble 5.2 -> 5.3) with keep-rate .80, several still closing at
decision 250. ANY bearing within 90 deg of the food raises the smell, so
d > 0 keeps a bad bearing as readily as a good one: he climbs, slowly.

RE-ENTRY WITH THAT REASON — `steer`, DECLARED 11-15 BEFORE ANY BIRTH IS READ
ON SEEDS 142/143 (s90 explored first, never counted). The same bit, used
twice: tumble on d <= 0 as before; while rising, the next push is the best
push so far plus a small perturbation (0.5 x a fresh draw), and a push
becomes the best only if its rise beat the best's (which fades x0.7 per
losing bout). d is read on log smell, so a rise means bearing, not nearness.
  steer       the rule above, want GIVEN (log of the same L+R sum).
  steeryoked  CONTROL THAT MUST FAIL: same machinery, but tumble and accept
              are coins at `steer`'s own measured rates on that seed.
  babble      the null, re-run on the same seeds.
RULE: `steer` ALIVE iff it feeds more births than `babble` on BOTH seeds
(142, 143; 20 births each) and by >= 8 of 40 in sum, AND more than
`steeryoked` likewise. Else DEAD, and a given want with a measured bit is
not enough on this body in 250 decisions. ALIVE = the rule works with the
want GIVEN; the learned want (stage 2, nursery) is then the entry.

RESULT 11-15 11:28 — `steer` ALIVE: fed 8/6 = 14/40 vs babble 3/0 = 3/40,
steeryoked 2/0 = 2/40 (failed, as it must); toward +.275/+.128.

STAGE 2 FOR `steer`, DECLARED 11-15 BEFORE ANY BIRTH IS READ ON SEEDS
144/145: `steerwant` = the steer rule on the LEARNED want (stage 2's Want
and nursery, unchanged; its logit read as is), `steerwantshuf` = CONTROL
THAT MUST FAIL (meal labels permuted before every fit), `babble` re-run.
ALIVE iff `steerwant` feeds more test births than `babble` on BOTH seeds and
by >= 8 of 40, AND more than `steerwantshuf` likewise. Else: the rule works
on a given want and his meals do not yet teach one at this dose.

    nice -n 19 $PY -m experiments.run_tumble run <arm> <seed> [n_test]
    $PY -m experiments.run_tumble tally
    $PY -m experiments.run_tumble selftest
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ARMS = ("babble", "told", "yoked", "want", "wantshuf", "steer", "steeryoked",
        "steerwant", "steerwantshuf")
SEEDS2, SEEDS3 = (142, 143), (144, 145)
HOLD, STD, E, N_TEST, MEAL_H, MARGIN = 5, 0.5, 250, 20, 25, 8   # SL's ruler
NURSE = 5000           # phase-1 decisions (want arms only) = SL's BABBLE
R_NURSE = 0.6          # m — wean_pilot's R0
SEEDS = (140, 141)
OUT = "/data/tmp/run_tumble"


class Want:
    """Linear 'a meal is coming' over all senses, from his own meals."""

    def __init__(self, dim: int, shuffle: bool, seed: int):
        self.w = np.zeros(dim)
        self.mu, self.sd = np.zeros(dim), np.ones(dim)
        self.X, self.Y = [], []
        self.shuffle = shuffle
        self.rng = np.random.RandomState(977 + seed)

    def add_trial(self, xs, fed):
        for i, x in enumerate(xs):
            self.X.append(x)
            self.Y.append(float(fed is not None and fed - i <= MEAL_H))

    def fit(self, steps: int = 200):
        y = np.asarray(self.Y)
        if y.sum() < 1:
            return
        import torch as t
        X = np.asarray(self.X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        if self.shuffle:
            y = y[self.rng.permutation(len(y))]
        xn = t.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        yt = t.from_numpy(y.astype(np.float32))
        w = t.zeros(X.shape[1], requires_grad=True)
        b = t.zeros(1, requires_grad=True)
        pw = t.tensor(min(50.0, float((1 - y).sum() / y.sum())))
        opt = t.optim.Adam([w, b], lr=0.05)
        for _ in range(steps):
            loss = (t.nn.functional.binary_cross_entropy_with_logits(
                xn @ w + b, yt, pos_weight=pw) + 0.01 * (w ** 2).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w = w.detach().numpy().astype(float)

    def __call__(self, x) -> float:
        return float(((x - self.mu) / self.sd) @ self.w)


def keep(arm: str, d: float, coin: float, rate: float) -> bool:
    """The whole rule. `d` = change in the want over the bout just ended."""
    if arm == "babble":
        return False
    if arm == "yoked":
        return coin < rate
    return d > 0.0


def _vec(obs) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=float).ravel()
                           for k in obs])


def _told_rate(seed: int) -> float:
    p = f"{OUT}/told_s{seed}.jsonl"
    if not os.path.exists(p):
        raise SystemExit(f"yoked needs told's keep-rate first: {p} missing")
    te = [json.loads(x) for x in open(p)]
    te = [r for r in te if r.get("phase") == "test"]
    return sum(r["kept"] for r in te) / max(1, sum(r["bouts"] for r in te))


def _steer_rates(seed: int) -> tuple:
    p = f"{OUT}/steer_s{seed}.jsonl"
    if not os.path.exists(p):
        raise SystemExit(f"steeryoked needs steer's rates first: {p} missing")
    te = [json.loads(x) for x in open(p)]
    b = max(1, sum(r["bouts"] for r in te))
    tum = sum(r["tum"] for r in te)
    return tum / b, sum(r["acc"] for r in te) / max(1, b - tum)


def steer_step(yoked: bool, dv: float, rbest, coin: float, coin2: float,
               rates: tuple) -> tuple:
    """(tumble, accept) for one finished bout."""
    if yoked:
        tum = coin < rates[0]
        return tum, (not tum) and coin2 < rates[1]
    tum = dv <= 0.0
    return tum, (not tum) and (rbest is None or dv > rbest)


def run(arm: str, seed: int, n_test: int = N_TEST, out_path: str = "",
        nurse: int = NURSE) -> None:
    import torch

    from . import smell_sense as S
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    torch.set_num_threads(1)
    torch.manual_seed(41_000 + seed)
    rate = _told_rate(seed) if arm == "yoked" else 0.0
    srates = _steer_rates(seed) if arm == "steeryoked" else (0.0, 0.0)
    S.install("smell", seed)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    w = _Life4(seed, j0, alpha, train=False).w
    dr, d, b = w.drives, w.data, w.rover_bid
    obs = w.observe()
    keys = list(obs)
    off = int(sum(np.asarray(obs[k]).size for k in keys[:keys.index("smell")]))
    x0 = _vec(obs)
    learned = arm in ("want", "wantshuf", "steerwant", "steerwantshuf")
    want = Want(len(x0), arm.endswith("wantshuf"), seed)
    floor = {n: g for n, (g, _) in dr._food.items()
             if d.geom_xpos[g][2] <= 0.5}
    legal = w.legal_spawns()
    idle = np.r_[np.zeros(4), -1.0, -1.0, 0.0, 0.0]
    prng = np.random.RandomState(seed * 6553 + 11)     # pushes
    crng = np.random.RandomState(seed * 733 + 5)       # yoked coin
    nrng = np.random.RandomState(seed * 389 + 3)       # the mother
    out_path = out_path or f"{OUT}/{arm}_s{seed}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def value(x) -> float:
        if learned:
            return want(x)
        return float(x[off:off + 8].sum())             # told / yoked / babble

    def ready():
        return [np.array(d.geom_xpos[g][:2]) for n, g in floor.items()
                if dr.t >= dr._respawn_at[n]]

    def trial(rule: str, nursed: bool) -> dict:
        while not ready():
            w.decide(idle)
            if w.died_this_decision:
                w.respawn()
        if nursed:
            r0 = ready()
            t = r0[int(nrng.randint(len(r0)))]
            ok = np.flatnonzero(np.linalg.norm(legal[:, :2] - t[None], axis=1)
                                <= R_NURSE)
            k = (int(ok[nrng.randint(len(ok))]) if len(ok) else int(np.argmin(
                np.linalg.norm(legal[:, :2] - t[None], axis=1))))
            w.respawn(at=(float(legal[k][0]), float(legal[k][1])))
        else:
            w.respawn()
        m0 = sum(dr.ate_total.values())
        a = np.clip(STD * prng.randn(8), -1.0, 1.0)
        xs_trial, cos = [], []
        fed, dmin, born_d, k, bouts, kept = None, 1e9, None, 0, 0, 0
        steer = rule in ("steer", "steeryoked")
        abest, rbest, n_acc, n_tum = None, None, 0, 0
        lg = (lambda v: float(np.log(max(v, 1e-12)))) if steer and not learned else float
        while k < E and fed is None:
            p0 = np.array(d.xpos[b][:2])
            r0 = ready()
            v0 = lg(value(_vec(w.observe())))
            dead = False
            for _ in range(HOLD):
                xs_trial.append(_vec(w.observe()))
                if r0 and born_d is None:
                    born_d = min(float(np.linalg.norm(q - p0)) for q in r0)
                w.decide(a)
                k += 1
                if w.died_this_decision:
                    dead = True
                    break
                r = ready()
                p = np.array(d.xpos[b][:2])
                if r:
                    dmin = min(dmin, min(float(np.linalg.norm(q - p))
                                         for q in r))
                if sum(dr.ate_total.values()) > m0:
                    fed = k
                    break
                if k >= E:
                    break
            if dead or fed is not None:
                break
            if r0:
                t = min(r0, key=lambda q: float(np.linalg.norm(q - p0)))
                mv, to = np.array(d.xpos[b][:2]) - p0, t - p0
                if np.linalg.norm(mv) > 1e-3 and np.linalg.norm(to) > 1e-3:
                    cos.append(float(mv @ to / np.linalg.norm(mv)
                                     / np.linalg.norm(to)))
            bouts += 1
            fresh = np.clip(STD * prng.randn(8), -1.0, 1.0)   # always drawn:
            coin = float(crng.rand())                          # streams align
            dv = lg(value(_vec(w.observe()))) - v0
            if steer:
                tum, acc = steer_step(rule == "steeryoked", dv, rbest, coin,
                                      float(crng.rand()), srates)
                if tum:
                    a, abest, rbest, n_tum = fresh, None, None, n_tum + 1
                else:
                    if acc or abest is None:
                        abest, rbest, n_acc = a.copy(), dv, n_acc + 1
                    else:
                        rbest *= 0.7
                    a = np.clip(abest + 0.5 * fresh, -1.0, 1.0)
            elif keep(rule, dv, coin, rate):
                kept += 1
            else:
                a = fresh
        want.add_trial(xs_trial, fed)
        pe = np.array(d.xpos[b][:2])                 # read only (post hoc)
        d_end = min([float(np.linalg.norm(q - pe)) for q in ready()] or [-1])
        d_high = min([float(np.linalg.norm(np.array(d.geom_xpos[g][:2]) - pe))
                      for n, (g, _) in dr._food.items()
                      if n not in floor and dr.t >= dr._respawn_at[n]] or [-1])
        return {"d_end": round(d_end, 2), "d_high": round(d_high, 2),
                "acc": n_acc, "tum": n_tum,
                "fed": fed, "born_d": round(born_d or -1, 2),
                "d_min": round(dmin, 2), "n": k, "bouts": bouts, "kept": kept,
                "toward": round(float(np.mean(cos)), 3) if cos else 0.0}

    with open(out_path, "w") as out:
        if learned:
            k_n, i, fed_n = 0, 0, 0
            while k_n < nurse:
                r = trial("babble", nursed=(i % 2 == 0))
                k_n, i, fed_n = k_n + r["n"], i + 1, fed_n + (r["fed"]
                                                               is not None)
            want.fit()
            sm = float(np.abs(want.w[off:off + S.SMELL_DIM]).sum()
                       / max(1e-9, np.abs(want.w).sum()))
            out.write(json.dumps({"arm": arm, "seed": seed, "phase": "nurse",
                                  "births": i, "fed": fed_n, "k": k_n,
                                  "want_smell_share": round(sm, 3)}) + "\n")
            out.flush()
        for ep in range(n_test):
            r = trial("steer" if arm.startswith("steerwant") else arm,
                      nursed=False)
            if learned:
                want.fit()                   # he keeps learning, every birth
            out.write(json.dumps(dict(r, arm=arm, seed=seed, phase="test",
                                      ep=ep)) + "\n")
            out.flush()


def _alive(fed: dict, x: str, nulls: tuple, seeds=SEEDS) -> bool:
    return all(all(fed[(x, s)] > fed[(o, s)] for s in seeds)
               and sum(fed[(x, s)] - fed[(o, s)] for s in seeds) >= MARGIN
               for o in nulls)


def tally(dirname: str = OUT, seeds=SEEDS) -> dict:
    fed = {}
    for p in sorted(glob.glob(f"{dirname}/*_s*.jsonl")):
        te = [r for r in map(json.loads, open(p)) if r["phase"] == "test"]
        if len(te) < N_TEST or te[0]["seed"] not in SEEDS + SEEDS2 + SEEDS3:
            continue
        a, s = te[0]["arm"], te[0]["seed"]
        fed[(a, s)] = sum(r["fed"] is not None for r in te)
        print(f"{a:8s} s{s}: fed {fed[(a, s)]}/{len(te)}  toward "
              f"{np.mean([r['toward'] for r in te]):+.3f}  keep "
              f"{sum(r['kept'] for r in te) / max(1, sum(r['bouts'] for r in te)):.2f}"
              f"  d_min med {np.median([r['d_min'] for r in te]):.2f}")
    verdict = {}
    for stage, x, nulls, seeds in ((1, "told", ("babble", "yoked"), SEEDS),
                                   (2, "want", ("babble", "wantshuf"), SEEDS),
                                   ("steer", "steer", ("babble", "steeryoked"),
                                    SEEDS2),
                                   ("steerwant", "steerwant",
                                    ("babble", "steerwantshuf"), SEEDS3)):
        if any((a, s) not in fed for a in (x,) + nulls for s in seeds):
            print(f"STAGE {stage}: INCOMPLETE, no verdict")
            continue
        verdict[x] = _alive(fed, x, nulls, seeds)
        print(f"STAGE {stage}: " + "  ".join(
            f"{a} {sum(fed[(a, s)] for s in seeds)}/{N_TEST * len(seeds)}"
            for a in (x,) + nulls) + f" -> {x} "
            + ("ALIVE" if verdict[x] else "DEAD"))
    return verdict


def selftest() -> None:
    """The rule on a toy plane, and the judge on planted tallies. A point
    climbs exp(-d/2) with world-frame pushes: `told` must reach the source
    far more often than `babble` and than the yoked coin."""
    def reach(arm, n=200, rate=0.5):
        rng, hit = np.random.RandomState(3), 0
        for _ in range(n):
            p = rng.uniform(-4, 4, 2)
            a = rng.randn(2)
            for _ in range(E // HOLD):
                v0 = np.exp(-np.linalg.norm(p) / 2) + 1e-3 * rng.randn()
                p = p + 0.15 * a / max(1e-9, np.linalg.norm(a))
                dv = np.exp(-np.linalg.norm(p) / 2) + 1e-3 * rng.randn() - v0
                if np.linalg.norm(p) < 0.3:
                    hit += 1
                    break
                fresh, coin = rng.randn(2), rng.rand()
                if not keep(arm, dv, coin, rate):
                    a = fresh
        return hit / n
    got = {a: reach(a) for a in ("babble", "told", "yoked")}
    assert got["told"] >= 0.5 and got["babble"] <= 0.1 and got["yoked"] <= 0.1, got
    f = {("told", 140): 9, ("told", 141): 8, ("babble", 140): 3,
         ("babble", 141): 2, ("yoked", 140): 4, ("yoked", 141): 3}
    assert _alive(f, "told", ("babble", "yoked"))
    assert not _alive({**f, ("yoked", 141): 8}, "told", ("babble", "yoked"))
    assert not _alive({**f, ("told", 140): 6, ("told", 141): 6},
                      "told", ("babble", "yoked"))            # +7 < 8
    assert steer_step(False, -0.1, 0.2, 0, 0, (0, 0)) == (True, False)
    assert steer_step(False, 0.3, 0.2, 0, 0, (0, 0)) == (False, True)
    assert steer_step(False, 0.1, 0.2, 0, 0, (0, 0)) == (False, False)
    assert steer_step(True, 9.0, 0.2, 0.9, 0.9, (0.5, 0.5)) == (False, False)
    wt = Want(6, False, 0)
    rng = np.random.RandomState(1)
    for _ in range(40):
        fed = 30 if rng.rand() < 0.5 else None
        xs = [rng.randn(6) + (np.eye(6)[2] * 3 if fed else 0) for _ in range(30)]
        wt.add_trial(xs, fed)
    wt.fit()
    assert int(np.argmax(np.abs(wt.w))) == 2, wt.w
    print(f"SELFTEST OK (toy reach: {got}; judge refuses a passing control "
          f"and a +7 margin; want finds the cue)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        run(a[1], int(a[2]), int(a[3]) if len(a) > 3 else N_TEST)
    elif a[0] == "tally":
        tally()
    elif a[0] == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
