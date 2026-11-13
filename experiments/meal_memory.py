"""LAB — MEAL MEMORY: what he replays, raced (no ledger row; a pilot).

Seat: Consolidation / the learning core's replay (holder BY DEFAULT: one FIFO
ring of 256 transitions, `survival.REPLAY_CAP`). END-FIT row 4 lists "replay of
256" under credit over a late payoff; row 2 asks for learning without
forgetting.

WHY. Every LAB race of 11-12/13 says the same thing: he has a nose that points
at food (99.5 % of poses), he meets a meal 1-2 times in 22 lives, and approach
never carries to the next birth. Read the replay: 256 slots at 0.25 updates /
decision, batch 16 — a meal's transitions are sampled ~16 times each and are
overwritten 256 decisions (51 s) after they happen. Smell is paired with food a
handful of times per life-time and then the pairing is gone; no critic can make
smell a conditioned reinforcer from that. Biology keeps what mattered: reward
tags the trace and the hippocampus replays it, far out of proportion to how
often it happened (reverse replay at reward, Foster & Wilson 2006; Singer &
Frank 2009; self-imitation, Oh 2018, is the field's version).

THE ARM. A second ring of the same shape (256). When a segment closes with a
meal inside it (he ate: the taste, a felt event), its rows — observations,
actions, GAE targets, advantages, exactly as the FIFO ring got them — are ALSO
copied there. Once it holds anything, each batch of 16 is 8 from the FIFO ring
and 8 from the meal ring. Same batch, same updates, same core, same loss:
matched compute; memory +256 rows. Nothing frozen: it is his own experience.

ARMS
  fifo    the holder: one ring (this module patches nothing)
  meal    the challenger above
  yoked   CONTROL THAT MUST FAIL: the same second ring, filled at the same
          moments (whenever a meal segment closes) but with the last MEAL-FREE
          segment instead. Same extra memory, same replay of old rows, same
          half-and-half batches — no meal in it. If it eats like `meal`, the
          gain was stale replay, not what was kept.

THE RULER (cheap on purpose: lives are ~2000 decisions, births need not be).
Nursery trials: he is born hungry (e 0.5) within BIRTH_R of a floor food that
is ready, with smell; a trial ends at his first meal, at TRIAL decisions, or at
death; he learns online throughout (AWR policy loss, as every LAB race since
11-12). The world idles, unlearned, until a floor food is ready before each
birth. Trials whose first decision is at k >= WARM are read.

RULE, DECLARED BEFORE ANY ROW EXISTS (seeds 91-93, TOTAL decisions/arm-seed):
fed fraction over read trials. `meal` is ALIVE iff it beats `fifo` on >= 2 of
3 seeds AND beats `yoked` on >= 2 of 3 seeds AND its pooled fed fraction
exceeds each of theirs by >= 0.10. Otherwise DEAD. Secondary, not gated:
median decisions-to-meal among fed trials.

PILOT 11-13 11:34 — ALIVE by the rule, thinly: read fed meal 46/95 = .484,
fifo 30/81 = .370, yoked 32/85 = .376; seed wins 2/3 and 3/3. Rows
/data/tmp/meal_memory/*.jsonl. Most meals land within 5-16 decisions of birth,
so the ruler is dominated by birth luck.

CLAIM GATE, DECLARED BEFORE ANY CLAIM ROW (seeds 94-98, never piloted; 13000
decisions/arm-seed; same arms, WARM, BIRTH_R, TRIAL): `meal` wins iff it beats
`fifo` on >= 4 of 5 seeds AND `yoked` on >= 4 of 5 AND pooled fed fraction
exceeds each by >= 0.10 AND, among read trials with born_d >= 0.7 m, its pooled
fed fraction exceeds each of theirs. Anything else: DEAD, and it stays dead.

    nice -n 19 $PY -m experiments.meal_memory run <arm> <seed> [total] [out]
    $PY -m experiments.meal_memory tally [dir]
    $PY -m experiments.meal_memory selftest
"""
from __future__ import annotations

import functools
import glob
import json
import os
import sys

import numpy as np

ARMS = ("fifo", "meal", "yoked")
BIRTH_R = 1.2          # m: born within this of a ready floor food
TRIAL = 250            # decisions per birth (the oracle ruler's 250)
E0 = 0.5               # born hungry: a meal is a felt drive reduction
WARM = 2000            # trials starting before this decision are not read
TOTAL = 6500
MARGIN = 0.10
OUT = "/data/tmp/meal_memory"


class _Mixed:
    """The FIFO ring, plus a second ring; batches are half-and-half once the
    second holds anything. Everything else is the FIFO ring's own."""

    def __init__(self, main, second):
        self.main, self.second = main, second

    def __getattr__(self, name):
        return getattr(self.main, name)

    def sample(self, gen, size=None):
        import torch

        from .cores import LC_BATCH
        size = size or LC_BATCH
        if self.second.n == 0:
            return self.main.sample(gen, size)
        h = size // 2
        b1, t1 = self.main.sample(gen, size - h)
        b2, t2 = self.second.sample(gen, h)
        return ({k: torch.cat([b1[k], b2[k]]) for k in b1},
                {k: torch.cat([t1[k], t2[k]]) for k in t1})


def _rows(ring, n0: int, n1: int):
    """Copies of the rows pushed to `ring` as pushes n0..n1-1."""
    out = []
    for j in range(max(n0, n1 - ring.cap), n1):
        i = j % ring.cap
        out.append(({k: ring.obs[k][i].clone() for k in ring.keys},
                    ring.action[i].clone(), float(ring.value[i, 0]),
                    float(ring.advantage[i])))
    return out


def make_life(arm: str, seed: int, j0: float, alpha: float):
    """A `_Life4` whose replay is `arm`'s. `fifo` is the class untouched."""
    from .survival import _TargetRing
    from .tests.lf_04_living_teaches import _Life4

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    if arm == "fifo":
        return _Life4(seed, j0, alpha, train=True)

    class _MealLife(_Life4):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.second = _TargetRing(self.w.observe())
            self.ring = _Mixed(self.ring, self.second)
            self._ate_seen = 0
            self._plain = None
            self.kept_segments = 0

        def _close_segment(self, boot, terminal):
            main = self.ring.main
            n0 = main.n
            super()._close_segment(boot, terminal)
            if main.n == n0:
                return
            rows = _rows(main, n0, main.n)
            ate = int(sum(self.w.drives.ate_total.values()))
            meal = ate > self._ate_seen
            self._ate_seen = ate
            if not meal:
                self._plain = rows
                return
            keep = rows if arm == "meal" else self._plain
            if keep is None:
                return
            for o, a, v, adv in keep:
                self.second.push(o, a, v, adv)
            self.kept_segments += 1

    return _MealLife(seed, j0, alpha, train=True)


def run(arm: str, seed: int, total: int = TOTAL, out_path: str = "") -> None:
    import torch

    from . import cores, drives
    from . import smell_sense as S
    from .policy_loss import lc_update_with
    from .tests.lf_03_learner_lives import _calibration

    S.install("smell", seed)
    torch.set_num_threads(1)
    cores.lc_update = functools.partial(lc_update_with, policy_loss="awr")
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = make_life(arm, seed, j0, alpha)
    w, dr = L.w, L.w.drives
    d, b = w.data, w.rover_bid
    w.observe()
    floor = {n: g for n, (g, _) in dr._food.items()
             if d.geom_xpos[g][2] <= 0.5}
    legal = w.legal_spawns()
    idle = np.r_[np.zeros(4), -1.0, -1.0, 0.0, 0.0]
    brng = np.random.RandomState(seed * 977 + 13)
    out_path = out_path or f"{OUT}/{arm}_s{seed}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def ready():
        return [np.array(d.geom_xpos[g][:2]) for n, g in floor.items()
                if dr.t >= dr._respawn_at[n]]

    ep = 0
    with open(out_path, "w") as out:
        while L.k < total:
            while not ready():                 # the world waits; he does not learn
                w.decide(idle)
            tg = ready()
            t = tg[int(brng.randint(len(tg)))]
            dist = np.linalg.norm(legal - t[None], axis=1)
            ok = np.flatnonzero(dist <= BIRTH_R)
            i = int(ok[brng.randint(len(ok))]) if len(ok) else int(np.argmin(dist))
            w.respawn(at=(float(legal[i][0]), float(legal[i][1])))
            dr.state = drives.DriveState(e=E0)
            w._prev_drive = drives.DriveState(e=E0)
            k0, m0 = L.k, sum(dr.ate_total.values())
            born_d, dmin, fed, died = float(dist[i]), 1e9, None, False
            for j in range(TRIAL):
                L.step()
                if w.died_this_decision:
                    died = True                # _Life closed the segment itself
                    break
                r = ready()
                if r:
                    p = np.array(d.xpos[b][:2])
                    dmin = min(dmin, min(float(np.linalg.norm(q - p)) for q in r))
                if sum(dr.ate_total.values()) > m0:
                    fed = j + 1
                    break
                if L.k >= total:
                    break
            if not died and L.seg["rew"]:      # a rebirth is a cut, not a death
                L._close_segment(boot=L.seg["val"][-1], terminal=False)
            row = {"arm": arm, "seed": seed, "ep": ep, "k0": k0, "fed": fed,
                   "born_d": round(born_d, 2), "d_min": round(dmin, 2),
                   "died": int(died), "n": L.k - k0,
                   "kept": int(getattr(L, "kept_segments", 0))}
            out.write(json.dumps(row) + "\n")
            out.flush()
            ep += 1


def tally(dirname: str = OUT) -> dict:
    res = {}
    for p in sorted(glob.glob(f"{dirname}/*_s*.jsonl")):
        rows = [json.loads(x) for x in open(p)]
        if not rows:
            continue
        rd = [r for r in rows if r["k0"] >= WARM and r["n"] > 0]
        fed = [r["fed"] for r in rd if r["fed"] is not None]
        res[(rows[0]["arm"], rows[0]["seed"])] = (
            len(fed), len(rd), float(np.median(fed)) if fed else -1.0,
            rows[-1]["kept"], sum(1 for r in rows if r["fed"] is not None),
            len(rows))
    seeds = sorted({s for _, s in res})
    for (a, s), v in sorted(res.items()):
        print(f"{a:6s} s{s}: read fed {v[0]}/{v[1]}  median-to-meal {v[2]:.0f}"
              f"  kept {v[3]}  all fed {v[4]}/{v[5]}")
    pooled = {}
    for a in ARMS:
        f = sum(res[(a, s)][0] for s in seeds if (a, s) in res)
        n = sum(res[(a, s)][1] for s in seeds if (a, s) in res)
        pooled[a] = f / n if n else float("nan")
        print(f"pooled {a}: {f}/{n} = {pooled[a]:.3f}")
    frac = lambda a, s: res[(a, s)][0] / max(1, res[(a, s)][1])
    full = all((a, s) in res for a in ARMS for s in seeds) and len(seeds) >= 3
    if full:
        wins = {o: sum(frac("meal", s) > frac(o, s) for s in seeds)
                for o in ("fifo", "yoked")}
        alive = all(wins[o] >= 2 and pooled["meal"] - pooled[o] >= MARGIN
                    for o in wins)
        print(f"seed wins vs fifo {wins['fifo']}/3, vs yoked {wins['yoked']}/3"
              f" -> {'ALIVE' if alive else 'DEAD'}")
    else:
        print("INCOMPLETE: no verdict")
    return res


def selftest() -> None:
    """The mixed ring must draw half its batch from the second ring, and the
    second ring must stay empty for `fifo`-shaped use (a check that can fail)."""
    import torch

    from .survival import _TargetRing
    obs = {"x": np.zeros(3, dtype=np.float32)}
    main, second = _TargetRing(obs), _TargetRing(obs)
    for i in range(40):
        main.push({"x": torch.zeros(3)}, torch.zeros(8), 0.0, 0.0)
    mx = _Mixed(main, second)
    g = torch.Generator().manual_seed(0)
    _, t = mx.sample(g)
    assert float(t["value"].sum()) == 0.0 and mx.n == 40
    for i in range(5):
        second.push({"x": torch.ones(3)}, torch.zeros(8), 1.0, 0.0)
    b, t = mx.sample(g)
    assert int(t["value"].sum()) == len(t["value"]) // 2, t["value"].sum()
    assert int(b["x"][:, 0].sum()) == len(t["value"]) // 2
    r = _rows(main, 30, 40)
    assert len(r) == 10
    print("SELFTEST OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        run(a[1], int(a[2]), int(a[3]) if len(a) > 3 else TOTAL,
            a[4] if len(a) > 4 else "")
    elif a[0] == "tally":
        tally(a[1] if len(a) > 1 else OUT)
    elif a[0] == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
