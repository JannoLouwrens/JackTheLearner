"""LAB — the place key, carried across death into a new world.  (Seat: Death &
persistence, VACANT.  No ledger row; a probe.  Follows `place_key.py`.)

PLAN part "Memory and sleep", right column: memory across death and rebirth
at small scale.  END-FIT row 2: everything learned travels with him.

WHERE THIS STARTS.  `place_key.py` read ALIVE on worlds 730-749 (11-19): a
ring key learned only from the passing of time re-finds his place .934,
compared once, against .933 for the 128-turn search (untrained .443,
time-shuffled .273, shuffled tags .026 vs chance .021).  Two reported-only
numbers from that run make this bet:
  * a STRANGER's key (trained in the next world) reads .932 here: the key
    looks like a fact about LOOKING, not about the room;
  * a key grown from his first 15 / 60 lives in a world reads .202 / .542,
    while the turn search with the same few memories read .31 / .65 (worlds
    700-719).  Early in a new world the home-grown key is the weak part.

THE BET.  The key is a skill, not a map: learned once in earlier worlds and
never trained again, it serves him from his first lives in a world he has
never seen — as well as the turn search with the same few memories, and
better than a key grown there.  What survives death is then two things with
different lifetimes: the key (carried everywhere) and the looks (per world).
PRIOR.  Learned place-recognition descriptors transfer across places when
trained on many (NetVLAD, Arandjelovic 2016: pixels, ~10^5-10^6 images,
pretrained trunk — from memory, source not opened); slow-feature place
codes are per-environment in Franzius 2007 (from memory).  PRIORS §6 is
silent on a from-scratch key carried between worlds at 10^4 looks.

DATA.  Nothing new is collected.  The carried key is trained ONCE, from
scratch, on the remembered looks (lives < 360) of worlds 720-724 pooled
(14,400 looks; a pair is two looks of one life of one world), with
`place_key.SETTINGS` and the ring network unchanged, torch seed 720.
Judged on worlds 730-749 with his memory limited to his first L lives
there (L = 15, 60, 360); reborn looks as in `place_key`.
ARMS at each L.
  carried      the key above; no gradient step in the judged world.
  carried1     the same from world 720 alone (reported only).
  b128         holder: the 128-turn search over the same L lives.
  home         `kt_ring_L`: a key grown from those L lives (read from the
               cached rows of the 11-19 run; already seen: .202/.542/.934).
TWINS THAT MUST FAIL.
  carried_un   the same network, untrained.
  carried_sh   trained as long on worlds 720-724 with life labels permuted.
  tags         carried recall at L = 60 scored against shuffled place tags.

RULE — written before any number below is computed.  Unit = a world.
  VOID if mean tags HIT > mean CHANCE(L=60) + 0.05.
  ALIVE iff all three:
   (a) at L = 15 AND at L = 60: mean HIT(b128) - mean HIT(carried) <= 0.05;
   (b) at L = 60: mean HIT(carried) - max(mean _un, mean _sh) >= 0.10, with
       carried ahead of BOTH twins on >= 16 of 20 worlds;
   (c) at L = 15: mean HIT(carried) - mean HIT(home) >= 0.05, with carried
       ahead of home on >= 16 of 20 worlds.
  Otherwise DEAD.  Reported, deciding nothing: L = 360 for every arm;
  carried1; the twins at L = 15.
DISCLOSED.  Worlds 730-749 are not fresh: the 11-19 run scored them, and
its stranger number (.932 at L = 360) is why this bet exists.  Unseen until
this rule was committed: every carried number at L = 15 / 60, the turn
search at L = 15 / 60 on these worlds, and both twins.  Worlds 720-724 were
used once (11-17, the VOID spectrum key), never to train anything.
What this cannot say: a key carried into a DIFFERENT kind of world (bigger,
other materials, the humanoid's eye height); a key inside a brain that is
also learning to eat; whether carrying beats continuing to train it there.

    nice -n 19 $PY -m experiments.place_key_carried
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.find_himself import (HIT_M, MEM_LIVES, OUT, _nearest,  # noqa: E402
                                      _sub)
from experiments.place_key import N, _keys, _recall, train_key  # noqa: E402

TRAIN_WORLDS = (720, 721, 722, 723, 724)
JUDGE_FIRST, JUDGE_N = 730, 20
DIAL = (15, 60, 360)


def _pool(worlds):
    X, L = [], []
    for w in worlds:
        z = np.load(os.path.join(OUT, f"world_{w}.npz"))
        m = z["life"] < MEM_LIVES
        X.append(_sub(z["xb"], N)[m])
        L.append(z["life"][m].astype(np.int64) + 1000 * (w - worlds[0]))
    return np.concatenate(X), np.concatenate(L)


def main() -> dict:
    Xp, Lp = _pool(TRAIN_WORLDS)
    nets = {"carried": train_key("ring", Xp, Lp, 720, "time"),
            "carried_un": train_key("ring", Xp, Lp, 720, "un"),
            "carried_sh": train_key("ring", Xp, Lp, 720, "sh"),
            "carried1": train_key("ring", *_pool(TRAIN_WORLDS[:1]), 720, "time")}
    print(f"keys trained on {len(Xp)} looks of worlds {TRAIN_WORLDS}", flush=True)
    rows = []
    for seed in range(JUDGE_FIRST, JUDGE_FIRST + JUDGE_N):
        z = np.load(os.path.join(OUT, f"world_{seed}.npz"))
        xy, life = z["xy"].astype(np.float64), z["life"]
        X = _sub(z["xb"], N)
        mem = life < MEM_LIVES
        te = (~mem) & np.r_[True, life[1:] != life[:-1]]
        q, M, lm = X[te], X[mem], life[mem]
        D = np.linalg.norm(xy[te][:, None] - xy[mem][None], axis=2)
        ar = np.arange(int(te.sum()))
        home = json.load(open(os.path.join(
            OUT, f"place_key_row_{seed}_of_{JUDGE_FIRST}_{JUDGE_N}.json")))
        kq = {a: _keys(n, q) for a, n in nets.items()}
        km = {a: _keys(n, M) for a, n in nets.items()}
        row = {"seed": seed}
        for L in DIAL:
            sel = lm < L
            Ds = D[:, sel]
            hit = lambda k: float((Ds[ar, k] <= HIT_M).mean())      # noqa: E731
            row[f"chance_L{L}"] = float((Ds <= HIT_M).mean())
            row[f"b128_L{L}"] = hit(_nearest(q, M[sel], True))
            row[f"home_L{L}"] = home[f"kt_ring_L{L}"]
            for a in nets:
                k = _recall(kq[a], km[a][sel])
                row[f"{a}_L{L}"] = hit(k)
                if a == "carried" and L == 60:
                    sh = np.random.RandomState(seed + 41).permutation(int(sel.sum()))
                    row["tags_L60"] = hit(sh[k])
        rows.append(row)
        print("  " + " ".join(f"{k}={v:.3f}" if k != "seed" else f"world {v}:"
                              for k, v in row.items()), flush=True)
    A = lambda k: np.array([r[k] for r in rows])                     # noqa: E731
    m = {k: float(A(k).mean()) for k in rows[0] if k != "seed"}
    a_gap = {L: m[f"b128_L{L}"] - m[f"carried_L{L}"] for L in (15, 60)}
    b_lead = m["carried_L60"] - max(m["carried_un_L60"], m["carried_sh_L60"])
    b_ahead = int(((A("carried_L60") > A("carried_un_L60"))
                   & (A("carried_L60") > A("carried_sh_L60"))).sum())
    c_lead = m["carried_L15"] - m["home_L15"]
    c_ahead = int((A("carried_L15") > A("home_L15")).sum())
    void = m["tags_L60"] > m["chance_L60"] + 0.05
    a_ok = all(g <= 0.05 for g in a_gap.values())
    b_ok = b_lead >= 0.10 and b_ahead >= 16
    c_ok = c_lead >= 0.05 and c_ahead >= 16
    out = {"train_worlds": TRAIN_WORLDS, "means": m,
           "a_behind_turn_search": a_gap, "a_ok": a_ok,
           "b_lead_over_twins_L60": b_lead, "b_ahead_of_both": b_ahead, "b_ok": b_ok,
           "c_lead_over_home_L15": c_lead, "c_ahead_of_home": c_ahead, "c_ok": c_ok,
           "verdict": "VOID" if void else "ALIVE" if a_ok and b_ok and c_ok else "DEAD",
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1,
                     default=lambda o: round(o, 3)))
    path = os.path.join(OUT, f"place_key_carried_{JUDGE_FIRST}_{JUDGE_N}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)
    return out


if __name__ == "__main__":
    main()
