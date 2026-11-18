"""LAB — a place key he learns from the passing of time.  (Seats: Death &
persistence, VACANT; Episodic retrieval's key.  No ledger row; a probe.)

PLAN part "Memory and sleep", right column: memory across death and rebirth;
finding himself again after rebirth.  PUBLISHED_PRIORS §6: "Retrieval is
always by learned similarity"; "also open: re-localising after rebirth".

WHERE THIS STARTS.  `find_himself.py` (11-17): on the body-frame eye (ruling
38: no compass) one look re-finds his place .933 — but only by trying all
128 turns of the look against every memory.  Compared once, ray for ray, it
falls to .54.  A hand-made turn-free key (amplitude spectrum) hit .845 and
was VOID.  A brain's store compares a key ONCE, and nothing in him may be
hand-made: the key has to be learned.

THE BET.  He needs no place tags and no compass to learn it.  Two looks a
moment apart in one life are looks from nearly the same place, usually at a
different heading (he turns faster than he travels).  A small network
trained from scratch only to give looks close in time close keys (and looks
from other lives far keys) will discard heading and keep place — the
slowness account of how place cells form.  Then recall is one comparison
per memory and loses little to the turn search.
PRIOR.  Franzius, Sprekeler & Wiskott 2007: slow features of a simulated
rat's visual stream are place cells when the head turns fast relative to
travel (from memory, source not opened).  Time-contrastive keys in agents:
Savinov 2018 reachability, CPC — pixels, 10^6+ frames (from memory).
PRIORS §6: recall worked only when the write was trained by prediction, at
~10^9 steps (MERLIN, HCAM).  Our budget: 2,880 looks per world, three to
five orders below — reachable only because a look is 256 numbers, not an
image.  The record does not say whether that is enough.  Kill below.

DATA.  `find_himself.collect`, unchanged (entry rung, 480 lives of 16
decisions, 8 kept looks each, born facing anywhere, random actions).  The
key is trained on the SAME 360 lives he remembers (2,880 looks), per world,
from scratch, seeded by the world.  Reborn = first kept look of lives
360-479.  Recall = nearest remembered key (cosine), no turning.
ARMS (128 body-frame rays x [distance, material]).
  kt_mlp   plain network on the raw look: must LEARN to ignore heading.
  kt_ring  ring-shaped network (circular convolutions, pooled round the
           ring): ignoring heading is anatomy; time teaches what to keep.
LOSS.  InfoNCE over a batch of lives: two looks of one life at different
moments are a pair; the other lives in the batch are the negatives.
TWINS THAT MUST FAIL (same network, same init seed).
  _un  untrained: the anatomy alone.
  _sh  time-shuffled: trained as long on pairs that are NOT close in time
       (life labels permuted over the looks).
  tags: recall scored against shuffled place tags (the scoring leaks).
HOLDERS on the same worlds: b128 (turn search), b128n (no turning), b128f
(spectrum, hand-made, reported only).

RULE — declared before any look of worlds 730-749 is generated.  Tuning set
690-692 only (network size, steps, temperature; frozen in SETTINGS below
and the PRIMARY arm named there before collection).  Unit = a world.
  VOID if the primary arm's tag control mean HIT > mean CHANCE + 0.05.
  A key is LEARNED FROM TIME iff mean HIT >= 0.50 with HIT - CHANCE >= 0.25
  on >= 16 of 20 worlds, AND mean HIT - max(mean _un, mean _sh) >= 0.10 with
  the key ahead of BOTH its twins on >= 16 of 20 worlds.
  THE BET IS ALIVE iff the PRIMARY arm is LEARNED FROM TIME and mean
  HIT(b128, turn search) - mean HIT(primary) <= 0.10.  Otherwise DEAD.
  Reported, deciding nothing: the other arm; b128n; b128f; HIT against
  lives lived (15 / 60 / 360: key trained AND memory limited to them);
  a STRANGER's key (trained in the next world, used in this one: is the
  key a fact about the room or about looking); recall cost.
  What this cannot say: whether the key survives inside a brain that is
  also learning to eat; a bigger or emptier world; a key trained online
  while the world's food moves (here every life re-lays the food already).

    nice -n 19 $PY -m experiments.place_key fit <first_seed> <n_worlds> [tune|part0|part1]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.find_himself import (HIT_M, MEM_LIVES, OUT, _nearest,  # noqa: E402
                                      _sub)

N = 128
DIAL_LIVES = (15, 60, 360)
# Frozen on worlds 690-692 before 730-749 were generated.
# Tuning read (ring arm only; the mlp arm takes the same settings untuned):
# 690 at temp .1/.2 = .85/.925 and 2,500 steps .79 (longer is worse); with
# the line below 691 = .950, 692 = .958 (turn search .936 over 690-692).
SETTINGS = dict(dim=32, steps=500, batch=128, temp=0.3, lr=5e-3, wd=1e-4)
PRIMARY = "kt_ring"


def _net(kind: str, dim: int):
    import torch.nn as nn

    if kind == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(2 * N, 128), nn.ReLU(),
                             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, dim))

    class Ring(nn.Module):
        def __init__(self):
            super().__init__()
            c = dict(padding_mode="circular")
            self.c1 = nn.Conv1d(2, 16, 5, padding=2, **c)
            self.c2 = nn.Conv1d(16, 32, 5, padding=2, stride=2, **c)
            self.c3 = nn.Conv1d(32, 32, 5, padding=2, stride=2, **c)
            self.out = nn.Linear(64, dim)

        def forward(self, x):                       # (B, n, 2)
            import torch
            h = x.transpose(1, 2)
            h = torch.relu(self.c1(h))
            h = torch.relu(self.c2(h))
            h = torch.relu(self.c3(h))
            return self.out(torch.cat([h.mean(-1), h.amax(-1)], 1))

    return Ring()


def _keys(net, X):
    import torch
    with torch.no_grad():
        k = net(torch.from_numpy(np.ascontiguousarray(X)).float())
        return torch.nn.functional.normalize(k, dim=1).numpy().astype(np.float64)


def train_key(kind: str, X, life, seed: int, mode: str = "time", **kw):
    """X (M, n, 2) remembered looks, life (M,) their life ids.  mode: 'time'
    (pairs from one life), 'sh' (life labels permuted), 'un' (no training)."""
    import torch
    import torch.nn.functional as F

    s = {**SETTINGS, **kw}
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = _net(kind, s["dim"])
    if mode == "un":
        return net
    rng = np.random.RandomState(seed + 31)
    lab = rng.permutation(life) if mode == "sh" else life
    groups = [np.where(lab == b)[0] for b in np.unique(lab)]
    groups = [g for g in groups if len(g) >= 2]
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["wd"])
    B = min(s["batch"], len(groups))
    for step in range(s["steps"]):
        pick = rng.choice(len(groups), B, replace=False)
        ij = np.array([rng.choice(groups[g], 2, replace=False) for g in pick])
        za = F.normalize(net(Xt[ij[:, 0]]), dim=1)
        zb = F.normalize(net(Xt[ij[:, 1]]), dim=1)
        logit = za @ zb.T / s["temp"]
        tgt = torch.arange(B)
        loss = 0.5 * (F.cross_entropy(logit, tgt) + F.cross_entropy(logit.T, tgt))
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def _recall(kq, km):
    return (kq @ km.T).argmax(1)


def _fit_world(seed: int, stranger_seed: int | None, arms, extras: bool, **kw):
    z = np.load(os.path.join(OUT, f"world_{seed}.npz"))
    xy, life = z["xy"].astype(np.float64), z["life"]
    X = _sub(z["xb"], N)
    rng = np.random.RandomState(seed + 37)
    mem = life < MEM_LIVES
    te = (~mem) & np.r_[True, life[1:] != life[:-1]]
    d_true = np.linalg.norm(xy[te][:, None] - xy[mem][None], axis=2)
    ar = np.arange(int(te.sum()))
    hit = lambda k, D=d_true: float((D[ar, k] <= HIT_M).mean())       # noqa: E731
    row = {"seed": seed, "chance": float((d_true <= HIT_M).mean())}
    q, M, lm = X[te], X[mem], life[mem]
    row["b128"] = hit(_nearest(q, M, True))
    row["b128n"] = hit(_nearest(q, M, False))
    spec = lambda x: np.abs(np.fft.rfft(x.astype(np.float64), axis=1))  # noqa: E731
    row["b128f"] = hit(_nearest(spec(q), spec(M), False))
    shuf = rng.permutation(int(mem.sum()))
    for arm in arms:
        kind = arm.split("_")[1]
        t0 = time.time()
        net = train_key(kind, M, lm, seed, "time", **kw)
        row[arm + "_train_s"] = time.time() - t0
        k = _recall(_keys(net, q), _keys(net, M))
        row[arm] = hit(k)
        row[arm + "_tags"] = hit(shuf[k])
        row[arm + "_med_m"] = float(np.median(d_true[ar, k]))
        for mode in ("un", "sh"):
            tw = train_key(kind, M, lm, seed, mode, **kw)
            row[f"{arm}_{mode}"] = hit(_recall(_keys(tw, q), _keys(tw, M)))
        if not extras:
            continue
        for L in DIAL_LIVES[:-1]:
            sel = lm < L
            nl = train_key(kind, M[sel], lm[sel], seed, "time", **kw)
            kl = _recall(_keys(nl, q), _keys(nl, M[sel]))
            row[f"{arm}_L{L}"] = hit(kl, d_true[:, sel])
        row[f"{arm}_L{DIAL_LIVES[-1]}"] = row[arm]
        if stranger_seed is not None:
            zs = np.load(os.path.join(OUT, f"world_{stranger_seed}.npz"))
            sm = zs["life"] < MEM_LIVES
            ns = train_key(kind, _sub(zs["xb"], N)[sm], zs["life"][sm],
                           stranger_seed, "time", **kw)
            row[arm + "_stranger"] = hit(_recall(_keys(ns, q), _keys(ns, M)))
    return row


def _row(seeds, i, tune, **kw):
    """One world's row, cached on disk for the registered run (two workers)."""
    s, n = seeds[i], len(seeds)
    path = os.path.join(OUT, f"place_key_row_{s}_of_{seeds[0]}_{n}.json")
    if not tune and not kw and os.path.exists(path):
        return json.load(open(path))
    row = _fit_world(s, seeds[(i + 1) % n] if n > 1 else None,
                     ["kt_mlp", "kt_ring"], not tune, **kw)
    if not tune and not kw:
        with open(path, "w") as f:
            json.dump(row, f)
    return row


def fit(first: int, n_worlds: int, tune: bool = False, part=None, **kw) -> dict:
    seeds = list(range(first, first + n_worlds))
    arms = ["kt_mlp", "kt_ring"]
    if part is not None:                        # rows only: worker `part` of 2
        for i in range(part, n_worlds, 2):
            _row(seeds, i, tune)
            print(f"  row {seeds[i]} done", flush=True)
        return {}
    rows = []
    for i, s in enumerate(seeds):
        rows.append(_row(seeds, i, tune, **kw))
        print(f"  world {s}: " + " ".join(
            f"{k}={v:.3f}" for k, v in rows[-1].items()
            if k != "seed" and not k.endswith("_s")), flush=True)
    A = lambda k: np.array([r[k] for r in rows])                 # noqa: E731
    ch = A("chance")
    need = min(n_worlds, math.ceil(0.8 * n_worlds))
    out = {"seeds": seeds, "settings": {**SETTINGS, **kw}, "primary": PRIMARY,
           "rows": rows, **{k: float(A(k).mean()) for k in rows[0] if k != "seed"}}
    for a in arms:
        finds = bool(A(a).mean() >= 0.50 and ((A(a) - ch) >= 0.25).sum() >= need)
        lead = float(A(a).mean() - max(A(a + "_un").mean(), A(a + "_sh").mean()))
        ahead = int(((A(a) > A(a + "_un")) & (A(a) > A(a + "_sh"))).sum())
        out[a + "_verdict"] = {
            "finds_himself": finds, "lead_over_twins": lead,
            "ahead_of_both": ahead, "sd": float(A(a).std()),
            "learned_from_time": bool(finds and lead >= 0.10 and ahead >= need),
            "behind_turn_search": float(A("b128").mean() - A(a).mean())}
    p = out[PRIMARY + "_verdict"]
    void = bool(A(PRIMARY + "_tags").mean() > ch.mean() + 0.05)
    out["verdict"] = ("VOID" if void else "ALIVE" if p["learned_from_time"]
                      and p["behind_turn_search"] <= 0.10 else "DEAD")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in out.items() if k != "rows"}, indent=1))
    if not tune:
        path = os.path.join(OUT, f"place_key_{first}_{n_worlds}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(path)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "fit":
        raise SystemExit(__doc__)
    extra = {}
    for a in sys.argv[5:]:
        k, v = a.split("=")
        extra[k] = float(v) if "." in v or "e" in v else int(v)
    mode = sys.argv[4] if len(sys.argv) > 4 else ""
    fit(int(sys.argv[2]), int(sys.argv[3]), tune=mode == "tune",
        part=int(mode[4:]) if mode.startswith("part") else None, **extra)
