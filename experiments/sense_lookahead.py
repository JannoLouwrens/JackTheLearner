"""LAB — SENSE LOOKAHEAD: act by imagining what each push would do to his own
senses, and take the one his meals taught him to want (no ledger row; a pilot).

Seat: Deliberation (the slow path) — VACANT; the reactive actor holds it by
default. END-FIT rows served: 4 ("thinking ahead, raced"; "no imagination step
in survival.py"), 3 (the `told` arm is advice arriving as a WANT, never as an
action), 1 (a slow chooser over a fast body). Closes none: both parts are small
plastic nets inside him, learned from his own life, saved like any weight.

NEW REASON to re-open a seat `DP.05` closed in W0 (lookahead bought 13 s ± 19):
that W0 gave him no sense of food before contact, so a lookahead had nothing
to look at. He has a nose now (smell_sense, 99.5 % of poses point to food) and
the reactive learner does not use it in 52k decisions (LAB 11-12/13).

THE ARM — two small learners and no actor:
  f     a forward model of his senses: (all senses now, last push, this push,
        decisions left in the bout) -> change in ALL senses by the bout's end.
        Learned from every decision he lives, reward-free (babbling teaches
        it). Dense where meals are sparse.
  want  a linear cue-value over ALL senses: will a meal come within MEAL_H
        decisions? Learned only from his own meals, all of which are kept.
  act   every HOLD decisions imagine K candidate pushes through f and take the
        one that raises `want` most (one-step lookahead; eps-random otherwise).
Both keep learning after every birth. Nothing is told which sense matters.

ARMS (same seed, same babble phase, same births; only the chooser differs)
  babble  NULL: random-repeat pushes for ever (the actor at init, held HOLD).
  want    the challenger above, nothing given.
  told    f learned, but the want is GIVEN ("more smell"): the adult's advice.
          An upper bound on f, not a from-scratch claim.
  shuf    CONTROL THAT MUST FAIL: `told`, with f trained on smell changes
          permuted across rows (same net, same data, same chooser, same
          pushes available — no link from push to smell).

THE RULER — explore_bout's stage A, verbatim (the oracle's ruler): W0's own
birth stream, a floor food ready at birth, E = 250 decisions, fed = first meal.
Phase 1: BABBLE decisions of babble births (all arms identical). Phase 2:
N_TEST births read. Published on this ruler: white noise 4/40, ou 3/40, pink
6/40 (s90+s94); perfect-map oracle 38/40.

RULE, DECLARED BEFORE ANY TEST BIRTH IS READ (seeds 90 and 94, 20 births
each): an arm X in {want, told} is ALIVE iff it feeds more births than
`babble` on BOTH seeds and by >= 8 of 40 in sum AND more than `shuf` on BOTH
seeds and by >= 8 of 40 in sum. The bet is DEAD if neither is ALIVE. `told`
alone ALIVE = the hand works and the want is not learned at this dose (say
so; that is stage B's question, not a win for the seat). Not gated, read
only: `toward` = mean cosine between each bout's displacement and the
bearing of the nearest ready floor food (ground truth he never sees) — tells
climbing from merely pushing harder.

    nice -n 19 $PY -m experiments.sense_lookahead run <arm> <seed> [babble] [n_test] [out]
    $PY -m experiments.sense_lookahead tally [dir]
    $PY -m experiments.sense_lookahead selftest
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ARMS = ("babble", "want", "told", "shuf")
HOLD = 5               # decisions per push (1 sim-s) — random-repeat's HOLD_K
K = 32                 # candidate pushes imagined per choice
STD = 0.5              # candidate / babble action std (the actor at init)
EPS = 0.1              # fraction of choices left to chance while planning
BABBLE = 5000          # phase-1 decisions
N_TEST = 20
E = 250
MEAL_H = 25            # "a meal is coming": within this many decisions
MARGIN = 8             # of 40
OUT = "/data/tmp/sense_lookahead"
HID = 128


class Mind:
    """f and want, with the rows they learn from. Pure numpy in, torch inside."""

    def __init__(self, dim: int, smell: slice, seed: int, shuffle_smell=False):
        import torch
        self.t = torch
        torch.manual_seed(31_000 + seed)
        self.dim, self.smell, self.shuffle = dim, smell, shuffle_smell
        self.f = torch.nn.Sequential(
            torch.nn.Linear(dim + 17, HID), torch.nn.ReLU(),
            torch.nn.Linear(HID, HID), torch.nn.ReLU(),
            torch.nn.Linear(HID, dim))
        self.opt = torch.optim.Adam(self.f.parameters(), lr=1e-3)
        self.w = np.zeros(dim)                  # the want, in x-normalised units
        self.X, self.AP, self.A, self.H, self.D = [], [], [], [], []
        self.PX, self.PY = [], []               # want rows: senses, meal-soon
        self.rng = np.random.RandomState(53 + seed)
        self.mu = np.zeros(dim)
        self.sd = np.ones(dim)
        self.dsd = np.ones(dim)
        self.fitted = False

    # ── rows ────────────────────────────────────────────────────────────
    def add_bout(self, xs, a_prev, a, x_end):
        n = len(xs)
        for i, x in enumerate(xs):
            self.X.append(x)
            self.AP.append(a_prev)
            self.A.append(a)
            self.H.append((n - i) / HOLD)
            self.D.append(x_end - x)

    def add_trial(self, xs, fed):
        for i, x in enumerate(xs):
            self.PX.append(x)
            self.PY.append(float(fed is not None and fed - i <= MEAL_H))

    # ── learning ────────────────────────────────────────────────────────
    def _inp(self, X, AP, A, H):
        t = self.t
        xn = (np.asarray(X) - self.mu) / self.sd
        return t.from_numpy(np.concatenate(
            [xn, np.asarray(AP), np.asarray(A),
             np.asarray(H).reshape(-1, 1)], axis=1).astype(np.float32))

    def fit(self, steps: int):
        t = self.t
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
        inp = self._inp(X, self.AP, self.A, self.H)
        tgt = t.from_numpy(Dn.astype(np.float32))
        for _ in range(steps):
            idx = t.from_numpy(self.rng.randint(0, len(X), 256))
            loss = ((self.f(inp[idx]) - tgt[idx]) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self._fit_want()

    def _fit_want(self):
        t = self.t
        y = np.asarray(self.PY)
        if y.sum() < 1:
            return
        xn = t.from_numpy(((np.asarray(self.PX) - self.mu) / self.sd)
                          .astype(np.float32))
        yt = t.from_numpy(y.astype(np.float32))
        w = t.zeros(self.dim, requires_grad=True)
        b = t.zeros(1, requires_grad=True)
        pw = t.tensor(min(50.0, float((1 - y).sum() / y.sum())))
        opt = t.optim.Adam([w, b], lr=0.05)
        for _ in range(200):
            loss = (t.nn.functional.binary_cross_entropy_with_logits(
                xn @ w + b, yt, pos_weight=pw) + 0.01 * (w ** 2).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w = w.detach().numpy().astype(float)

    # ── imagination ─────────────────────────────────────────────────────
    def imagine(self, x, a_prev, cands):
        """Predicted change of every sense (raw units) for each candidate."""
        n = len(cands)
        with self.t.no_grad():
            out = self.f(self._inp(np.repeat(x[None], n, 0),
                                   np.repeat(a_prev[None], n, 0), cands,
                                   np.ones(n))).numpy()
        return out * self.dsd

    def score(self, arm, x, a_prev, cands):
        d = self.imagine(x, a_prev, cands)
        if arm == "want":
            return (d / self.sd) @ self.w
        s = d[:, self.smell]
        return s[:, :8].sum(1)                   # told / shuf: more smell (L+R)


def _vec(obs) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=float).ravel()
                           for k in obs])


def run(arm: str, seed: int, babble: int = BABBLE, n_test: int = N_TEST,
        out_path: str = "") -> dict:
    import torch

    from . import smell_sense as S
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    torch.set_num_threads(1)
    S.install("smell", seed)
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    w = _Life4(seed, j0, alpha, train=False).w
    dr, d, b = w.drives, w.data, w.rover_bid
    obs = w.observe()
    keys = list(obs)
    off = int(sum(np.asarray(obs[k]).size for k in keys[:keys.index("smell")]))
    smell = slice(off, off + S.SMELL_DIM)
    x = _vec(obs)
    mind = Mind(len(x), smell, seed, shuffle_smell=(arm == "shuf"))
    floor = {n: g for n, (g, _) in dr._food.items()
             if d.geom_xpos[g][2] <= 0.5}
    idle = np.r_[np.zeros(4), -1.0, -1.0, 0.0, 0.0]
    crng = np.random.RandomState(seed * 6553 + 11)     # candidate pushes
    erng = np.random.RandomState(seed * 733 + 5)       # eps draws
    out_path = out_path or f"{OUT}/{arm}_s{seed}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def ready():
        return [np.array(d.geom_xpos[g][:2]) for n, g in floor.items()
                if dr.t >= dr._respawn_at[n]]

    def trial(plan: bool):
        while not ready():
            w.decide(idle)
            if w.died_this_decision:
                w.respawn()
        w.respawn()
        m0 = sum(dr.ate_total.values())
        a_prev = np.zeros(8)
        xs_trial, cos = [], []
        fed, dmin, born_d, k = None, 1e9, None, 0
        while k < E and fed is None:
            x0 = _vec(w.observe())
            cands = np.clip(STD * crng.randn(K, 8), -1.0, 1.0)
            cands[0] = a_prev                        # staying is always offered
            if plan and erng.rand() >= EPS:
                a = cands[int(np.argmax(mind.score(arm, x0, a_prev, cands)))]
            else:
                a = cands[1]
            p0 = np.array(d.xpos[b][:2])
            r0 = ready()
            xs, dead = [], False
            for _ in range(HOLD):
                xs.append(_vec(w.observe()))
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
            xs_trial += xs
            if dead:
                break
            if fed is None:                  # a meal removes the smell: no row
                mind.add_bout(xs, a_prev, a, _vec(w.observe()))
                if r0:
                    t = min(r0, key=lambda q: float(np.linalg.norm(q - p0)))
                    mv, to = np.array(d.xpos[b][:2]) - p0, t - p0
                    if np.linalg.norm(mv) > 1e-3 and np.linalg.norm(to) > 1e-3:
                        cos.append(float(mv @ to / np.linalg.norm(mv)
                                         / np.linalg.norm(to)))
            a_prev = a
        mind.add_trial(xs_trial, fed)
        return {"fed": fed, "born_d": round(born_d or -1, 2),
                "d_min": round(dmin, 2), "n": k,
                "toward": round(float(np.mean(cos)), 3) if cos else 0.0}

    rows, k_b, fed_b = [], 0, 0
    with open(out_path, "w") as out:
        while k_b < babble:
            r = trial(plan=False)
            k_b += r["n"]
            fed_b += int(r["fed"] is not None)
            rows.append(dict(r, phase="babble"))
        mind.fit(800)
        head = {"arm": arm, "seed": seed, "phase": "fit", "babble_k": k_b,
                "babble_births": len(rows), "babble_fed": fed_b,
                "rows": len(mind.X), "want_norm": float(np.abs(mind.w).sum()),
                "want_smell_share": float(np.abs(mind.w[smell]).sum()
                                          / max(1e-9, np.abs(mind.w).sum()))}
        out.write(json.dumps(head) + "\n")
        out.flush()
        for ep in range(n_test):
            r = trial(plan=(arm != "babble"))
            if arm != "babble":
                mind.fit(100)                # he keeps learning, every birth
            row = dict(r, arm=arm, seed=seed, phase="test", ep=ep,
                       meals_known=int(np.sum(mind.PY) > 0))
            out.write(json.dumps(row) + "\n")
            out.flush()
    return head


def tally(dirname: str = OUT) -> dict:
    res = {}
    for p in sorted(glob.glob(f"{dirname}/*_s*.jsonl")):
        rows = [json.loads(x) for x in open(p)]
        te = [r for r in rows if r["phase"] == "test"]
        if not te:
            continue
        h = rows[0]
        res[(te[0]["arm"], te[0]["seed"])] = (
            sum(r["fed"] is not None for r in te), len(te),
            float(np.mean([r["toward"] for r in te])),
            float(np.median([r["d_min"] for r in te])), h)
    seeds = sorted({s for _, s in res})
    for (a, s), v in sorted(res.items()):
        print(f"{a:6s} s{s}: fed {v[0]}/{v[1]}  toward {v[2]:+.3f}  "
              f"d_min med {v[3]:.2f}  babble fed {v[4]['babble_fed']}/"
              f"{v[4]['babble_births']}  want smell share "
              f"{v[4]['want_smell_share']:.2f}")
    full = (all((a, s) in res and res[(a, s)][1] >= N_TEST
                for a in ARMS for s in seeds) and len(seeds) >= 2)
    if not full:
        print("INCOMPLETE: no verdict")
        return res
    tot = {a: sum(res[(a, s)][0] for s in seeds) for a in ARMS}
    print("sum " + "  ".join(f"{a} {tot[a]}/{N_TEST * len(seeds)}"
                             for a in ARMS))
    alive = {}
    for x in ("want", "told"):
        alive[x] = all(
            all(res[(x, s)][0] > res[(o, s)][0] for s in seeds)
            and tot[x] - tot[o] >= MARGIN for o in ("babble", "shuf"))
        print(f"{x}: {'ALIVE' if alive[x] else 'DEAD'}")
    print("BET " + ("ALIVE" if any(alive.values()) else "DEAD"))
    return res


def selftest() -> None:
    """On a toy world where push dim 6 moves sense dim 0 (and nothing else is
    learnable), f must pick the push that raises it; the shuffled control must
    not: mean (chosen push / best push offered) >= .85 vs <= .3."""
    rng = np.random.RandomState(0)
    picks = {}
    for shuffle in (False, True):
        m = Mind(14, slice(0, 12), 0, shuffle_smell=shuffle)
        for _ in range(600):
            x = rng.randn(14)
            a = np.clip(0.5 * rng.randn(8), -1, 1)
            x_end = x + 0.05 * rng.randn(14)
            x_end[0] = x[0] + a[6]
            m.add_bout([x], np.zeros(8), a, x_end)
        m.fit(600)
        got = []
        for _ in range(100):
            c = np.clip(0.5 * rng.randn(K, 8), -1, 1)
            s = m.score("told", rng.randn(14), np.zeros(8), c)
            got.append(c[int(np.argmax(s)), 6] / c[:, 6].max())
        picks[shuffle] = round(float(np.mean(got)), 2)
    assert picks[False] >= 0.85 and picks[True] <= 0.3, picks
    m = Mind(14, slice(0, 12), 1)
    for _ in range(20):
        xs = [rng.randn(14) for _ in range(30)]
        fed = 30 if rng.rand() < 0.5 else None
        if fed:
            for x in xs:
                x[3] += 3.0
        m.add_trial(xs, fed)
        m.add_bout(xs[:5], np.zeros(8), np.zeros(8), xs[5])
    m.X = m.X * 4
    m.AP, m.A, m.H, m.D = m.AP * 4, m.A * 4, m.H * 4, m.D * 4
    m.fit(10)
    assert int(np.argmax(np.abs(m.w))) == 3, np.round(m.w, 2)
    print(f"SELFTEST OK (chosen push / best push: told {picks[False]}, "
          f"shuffled {picks[True]}; want finds the cue)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        run(a[1], int(a[2]), int(a[3]) if len(a) > 3 else BABBLE,
            int(a[4]) if len(a) > 4 else N_TEST, a[5] if len(a) > 5 else "")
    elif a[0] == "tally":
        tally(a[1] if len(a) > 1 else OUT)
    elif a[0] == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
