"""LAB — memory across rebirth: can he tell WHERE he is from one look, and
does that survive his eye turning with his head?  (Seats: Death & persistence,
VACANT; Vision encoder, BY DEFAULT.  No ledger row; a probe, not a spec.)

PLAN part "Memory and sleep", right column: *"Finding himself again after
rebirth with 16 rays."*  PUBLISHED_PRIORS §6 "also open: re-localising after
rebirth"; long form row L4 ("Unknown.  Expect poor localisation in symmetric
arenas").

THE BET.  A map he keeps across lives is worth nothing unless a reborn body
can say where on it he stands.  Today's eye casts its rays at WORLD azimuths
(`w0.py:219`, `fast_sim.apply_row_vision`): ray k always points the same
compass way, so he owns a free compass no animal has and the human-like body
(END-FIT row 1, an eye that turns with the head) will not have.  Bet: with
the compass, one look is enough to find himself, even at 16 rays; WITHOUT it,
16 rays cannot (22.5 deg between rays is too coarse to re-align a look) and
the 128-ray row can, by trying every turn of the look against memory.
PRIOR.  Nothing published on re-localising from range-only senses in a
from-scratch agent (long form, "Gaps").  Published successes had vision
(MERLIN) or GPS + compass (Wijmans); robots re-localise from range rings by
rotation-invariant matching, but with hundreds of beams (Scan Context; from
memory, source not opened).  Budget: minutes of CPU — this is a look-up in
his own remembered looks, no training steps at all.

WORLD.  `jack_env` at the entry rung (difficulty 0.75, random_layout: foods
re-placed at every birth, so food is clutter here, never a landmark).  One
world = one seed.  N_BOUTS lives; each a fresh body at a uniform legal spawn,
random actions held HOLD decisions; looks kept from decision KEEP_FROM on
(the body has settled to its living height).
EYES (all [distance/8 m, material] per ray, one level row, cast from the
seated eye point).  Collected once at 256 azimuths; 128 and 16 are exact
sub-rows (every 2nd / 16th ray).
  w16, w128   world frame: ray k at azimuth 2*pi*k/n          (today's eye)
  b16, b128   body frame: the same row turned by the body's own yaw plus a
              heading drawn afresh at each birth (he is born facing anywhere
              and the eye turns with him)                      (the eye to come)
MEMORY (his own, nothing taught).  Every kept look of the first MEM_LIVES
lives.  Reborn (a later life, its FIRST kept look) he recalls the remembered
look nearest to what he sees now: squared distance over all rays and both
channels.  World-frame eyes compare ray k with ray k.  Body-frame eyes take
the best of all n turns of the look (circular shift) — the challenger's
mechanism.  `b128n` = body frame, NO turning: reported, the naive holder.
SCORE (primary).  HIT = the recalled look was seen within HIT_M = 1.0 m of
where he stands now.  The place tags are the adult's ruler, never his input.
CHANCE = the share of remembered looks within 1.0 m of him (a random recall).
CONTROLS THAT MUST FAIL.  (1) tags: the same recall scored against memory
places shuffled over the looks; (2) scrambled eye: the reborn look's rays in
one fixed random order per world, memory untouched.

RULE — declared before worlds 700-719 are generated (tuning set 690-692):
  20 worlds, N_BOUTS 480 (MEM_LIVES 360, 120 rebirths each); unit = a world.
  VOID if, for any of the four arms, the tag control's mean HIT exceeds mean
  CHANCE + 0.05, or the scrambled eye's mean HIT exceeds mean CHANCE + 0.10.
  An arm FINDS HIMSELF iff mean HIT >= 0.50 AND HIT - CHANCE >= 0.25 on
  >= 16 of 20 worlds.
  THE BET IS ALIVE iff b128 FINDS HIMSELF, and mean HIT(b128) - mean
  HIT(b16) >= 0.10 with b128 ahead on >= 16 of 20 worlds.  Otherwise DEAD.
  PLAN's own question ("with 16 rays") is answered by w16 on the same rule.
  Reported, deciding nothing: w128; b128n; the compass's price (w - b at
  each width); distance-only looks (no material); median error in metres;
  HIT against the number of lives remembered (15 / 60 / 360); and a
  STRANGER's memory — recall from the NEXT world's memory (700 <- 701, ...,
  719 <- 700): how much of finding himself is the room every world shares.
  What this cannot say: whether a BRAIN learns to do this (raw looks as
  memory keys are the simplest store there is; the record says retrieval
  works by LEARNED similarity), nor whether one look still suffices in a
  bigger or emptier world.

    nice -n 19 $PY -m experiments.find_himself collect <seed> [<seed> ...]
    nice -n 19 $PY -m experiments.find_himself fit <first_seed> <n_worlds>

FOLLOW-ON (11-17 23:10, after the registered fit on 700-719 read ALIVE) — A
KEY THAT NEEDS NO TURNING.  Trying all 128 turns of a look against every
memory is 128 comparisons per memory; a brain's store compares a key once.
Bet: the amplitude spectrum of the row over azimuth (|rfft|, 65 x 2 numbers,
unchanged by any turn of the head) is such a key and loses little.  Prior:
rotation-invariant ring descriptors re-localise lidar robots (Scan Context,
from memory, source not opened); nothing for a from-scratch agent.
RULE — declared before any look of worlds 720-724 is read (no tuning set:
nothing to tune).  Arm b128f = body frame, 128 rays, nearest remembered
spectrum, no turning.  ALIVE iff b128f FINDS HIMSELF (mean HIT >= 0.50 and
HIT - CHANCE >= 0.25 on >= 4 of 5 worlds) AND mean HIT(b128, turning) - mean
HIT(b128f) <= 0.10.  VOID if its tag control > CHANCE + 0.05 or its
scrambled eye (rays permuted before the spectrum) > CHANCE + 0.10.
Otherwise DEAD: the phases carry the place and the turn search stays.
Reported: b16f; b128n.

    nice -n 19 $PY -m experiments.find_himself fitf <first_seed> <n_worlds>
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402

OUT = os.environ.get("FIND_HIMSELF_OUT", "/data/tmp/find_himself")
N_BOUTS, MEM_LIVES = 480, 360
BOUT, KEEP_FROM, HOLD = 16, 8, 5
N_AZ = 256
HIT_M = 1.0
WIDTHS = (16, 128)
DIAL_LIVES = (15, 60, 360)


def collect(seed: int, n_bouts: int = N_BOUTS) -> str:
    import mujoco
    from experiments import w0 as W0MOD
    from experiments.jack_env import JackEnv
    from experiments.w0 import random_action

    env = JackEnv(seed=seed, difficulty=0.75, random_layout=True)
    w = env.w
    m, d = w.model, w.data
    groups = W0MOD._RAY_GROUPS
    tex = w._tex
    rng = np.random.RandomState(seed + 11)
    az0 = 2 * math.pi * np.arange(N_AZ) / N_AZ

    def cast(pos, offset):
        az = az0 + offset
        vec = np.ascontiguousarray(
            np.stack([np.cos(az), np.sin(az), np.zeros(N_AZ)], -1)).ravel()
        gid = np.full(N_AZ, -1, dtype=np.int32)
        dist = np.zeros(N_AZ, dtype=np.float64)
        mujoco.mj_multiRay(m, d, pos, vec, groups, 1, -1, gid, dist,
                           N_AZ, W0MOD.RAY_MAX_RANGE)
        miss = (gid < 0) | (dist < 0) | (dist > W0MOD.RAY_MAX_RANGE)
        dn = np.where(miss, 1.0, dist / W0MOD.RAY_MAX_RANGE)
        tx = np.where(miss, 0.0, tex[np.maximum(gid, 0)])
        return np.stack([dn, tx], -1).astype(np.float32)

    XW, XB, XY, LIFE, TH = [], [], [], [], []
    t0 = time.time()
    a_act = random_action(rng)
    for b in range(n_bouts):
        w.respawn()
        heading = rng.uniform(0, 2 * math.pi)       # born facing anywhere
        for t in range(BOUT):
            if t % HOLD == 0:
                a_act = random_action(rng)
            if t >= KEEP_FROM:
                mujoco.mj_forward(m, d)
                pos = np.array(d.xpos[w.rover_bid], dtype=float)
                pos[2] += W0MOD.RAY_Z_OFFSET
                R = d.xmat[w.rover_bid].reshape(3, 3)
                th = math.atan2(R[1, 0], R[0, 0]) + heading
                XW.append(cast(pos, 0.0))
                XB.append(cast(pos, th))
                XY.append(pos[:2].copy())
                LIFE.append(b)
                TH.append(th)
            w.decide(a_act)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"world_{seed}.npz")
    np.savez_compressed(path, xw=np.array(XW), xb=np.array(XB),
                        xy=np.array(XY, dtype=np.float32),
                        life=np.array(LIFE), th=np.array(TH, dtype=np.float32))
    xy = np.array(XY)
    print(f"world {seed}: {len(XY)} looks, {n_bouts} lives, x {xy[:,0].min():.1f}"
          f"..{xy[:,0].max():.1f} y {xy[:,1].min():.1f}..{xy[:,1].max():.1f}, "
          f"{time.time() - t0:.0f}s -> {path}", flush=True)
    return path


# ── recall ──────────────────────────────────────────────────────────────
def _nearest(q, mem, turn: bool, chunk: int = 8):
    """Index of the remembered look nearest each reborn look.  q (T, n, c),
    mem (M, n, c).  turn=True: best over all n circular shifts of the look."""
    T = len(q)
    out = np.empty(T, dtype=np.int64)
    m2 = (mem.astype(np.float64) ** 2).sum((1, 2))
    if not turn:
        qf, mf = q.reshape(T, -1).astype(np.float64), mem.reshape(len(mem), -1)
        for i in range(0, T, 64):
            dd = m2[None, :] - 2.0 * qf[i:i + 64] @ mf.T.astype(np.float64)
            out[i:i + 64] = dd.argmin(1)
        return out
    n = q.shape[1]
    Fm = np.fft.rfft(mem.astype(np.float64), axis=1)            # (M, f, c)
    Fq = np.fft.rfft(q.astype(np.float64), axis=1)              # (T, f, c)
    for i in range(0, T, chunk):
        cross = (np.conj(Fq[i:i + chunk, None]) * Fm[None]).sum(-1)   # (t, M, f)
        corr = np.fft.irfft(cross, n=n, axis=-1)                # (t, M, n)
        out[i:i + chunk] = (m2[None, :] - 2.0 * corr.max(-1)).argmin(1)
    return out


def _sub(x, n):
    return x[:, :: N_AZ // n]


def _fit_world(seed: int, stranger_seed: int | None):
    z = np.load(os.path.join(OUT, f"world_{seed}.npz"))
    xy, life = z["xy"].astype(np.float64), z["life"]
    rng = np.random.RandomState(seed + 23)
    mem = life < MEM_LIVES
    first = np.r_[True, life[1:] != life[:-1]]
    te = (~mem) & first
    d_true = np.linalg.norm(xy[te][:, None] - xy[mem][None], axis=2)   # (T, M)
    row = {"seed": seed, "n_mem": int(mem.sum()), "n_test": int(te.sum()),
           "chance": float((d_true <= HIT_M).mean())}
    T = int(te.sum())
    ar = np.arange(T)
    shuf = rng.permutation(int(mem.sum()))
    zs = (np.load(os.path.join(OUT, f"world_{stranger_seed}.npz"))
          if stranger_seed is not None else None)
    for frame, key, turn in (("w", "xw", False), ("b", "xb", True)):
        for n in WIDTHS:
            arm = f"{frame}{n}"
            X = _sub(z[key], n)
            q, M = X[te], X[mem]
            k = _nearest(q, M, turn)
            err = d_true[ar, k]
            row[arm] = float((err <= HIT_M).mean())
            row[arm + "_med_m"] = float(np.median(err))
            row[arm + "_tags"] = float((d_true[ar, shuf[k]] <= HIT_M).mean())
            perm = rng.permutation(n)
            ks = _nearest(q[:, perm], M, turn)
            row[arm + "_scr"] = float((d_true[ar, ks] <= HIT_M).mean())
            kd = _nearest(q[..., :1], M[..., :1], turn)
            row[arm + "_distonly"] = float((d_true[ar, kd] <= HIT_M).mean())
            for L in DIAL_LIVES:
                sel = life[mem] < L
                kl = _nearest(q, M[sel], turn)
                row[f"{arm}_L{L}"] = float((d_true[ar][:, sel][ar, kl] <= HIT_M).mean())
            if zs is not None:
                sm = zs["life"] < MEM_LIVES
                kk = _nearest(q, _sub(zs[key], n)[sm], turn)
                es = np.linalg.norm(xy[te] - zs["xy"][sm][kk], axis=1)
                row[arm + "_stranger"] = float((es <= HIT_M).mean())
            if frame == "b" and n == 128:
                kn = _nearest(q, M, False)
                row["b128n"] = float((d_true[ar, kn] <= HIT_M).mean())
    return row


def fit(first: int, n_worlds: int) -> dict:
    seeds = list(range(first, first + n_worlds))
    rows = [_fit_world(s, seeds[(i + 1) % n_worlds] if n_worlds > 1 else None)
            for i, s in enumerate(seeds)]
    A = lambda k: np.array([r[k] for r in rows])                 # noqa: E731
    arms = [f"{f}{n}" for f in "wb" for n in WIDTHS]
    need = min(n_worlds, math.ceil(0.8 * n_worlds))
    ch = A("chance")
    out = {"seeds": seeds, "rows": rows, "chance": float(ch.mean())}
    void = []
    for a in arms:
        if A(a + "_tags").mean() > ch.mean() + 0.05:
            void.append(f"{a} tag control {A(a + '_tags').mean():.3f}")
        if A(a + "_scr").mean() > ch.mean() + 0.10:
            void.append(f"{a} scrambled eye {A(a + '_scr').mean():.3f}")
        finds = bool(A(a).mean() >= 0.50 and ((A(a) - ch) >= 0.25).sum() >= need)
        out[a] = {"hit": float(A(a).mean()), "sd": float(A(a).std()),
                  "finds_himself": finds,
                  "worlds_clear": int(((A(a) - ch) >= 0.25).sum())}
        for suf in ("_med_m", "_tags", "_scr", "_distonly", "_stranger",
                    *(f"_L{L}" for L in DIAL_LIVES)):
            if a + suf in rows[0]:
                out[a][suf[1:]] = float(A(a + suf).mean())
    out["b128n"] = float(A("b128n").mean())
    lead = float(A("b128").mean() - A("b16").mean())
    ahead = int((A("b128") > A("b16")).sum())
    out["lead_b128_b16"], out["ahead_b128_b16"] = lead, ahead
    out["void"] = void
    out["verdict"] = ("VOID" if void else
                      "ALIVE" if (out["b128"]["finds_himself"] and lead >= 0.10
                                  and ahead >= need) else "DEAD")
    print(f"worlds {seeds[0]}-{seeds[-1]}: chance {ch.mean():.3f}")
    for a in arms + []:
        o = out[a]
        print(f"  {a:5s} HIT {o['hit']:.3f} (sd {o['sd']:.3f}) med {o['med_m']:.2f} m"
              f" | tags {o['tags']:.3f} scr {o['scr']:.3f} | dist-only "
              f"{o['distonly']:.3f} | lives 15/60/360 {o['L15']:.3f}/{o['L60']:.3f}/"
              f"{o['L360']:.3f} | stranger {o.get('stranger', float('nan')):.3f}"
              f" | clear {o['worlds_clear']}/{n_worlds} finds={o['finds_himself']}")
    print(f"  b128n (no turning) HIT {out['b128n']:.3f}")
    print(f"  b128 - b16 = {lead:+.3f}, ahead {ahead}/{n_worlds}  ->  "
          f"{out['verdict']} {void}")
    path = os.path.join(OUT, f"fit_{first}_{n_worlds}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)
    return out


def fitf(first: int, n_worlds: int) -> dict:
    """The follow-on: amplitude-spectrum keys, no turning (rule in docstring)."""
    rows = []
    for seed in range(first, first + n_worlds):
        z = np.load(os.path.join(OUT, f"world_{seed}.npz"))
        xy, life = z["xy"].astype(np.float64), z["life"]
        rng = np.random.RandomState(seed + 29)
        mem = life < MEM_LIVES
        te = (~mem) & np.r_[True, life[1:] != life[:-1]]
        d_true = np.linalg.norm(xy[te][:, None] - xy[mem][None], axis=2)
        ar = np.arange(int(te.sum()))
        hit = lambda k: float((d_true[ar, k] <= HIT_M).mean())      # noqa: E731
        spec = lambda x: np.abs(np.fft.rfft(x.astype(np.float64), axis=1))  # noqa: E731
        row = {"seed": seed, "chance": float((d_true <= HIT_M).mean())}
        for n in WIDTHS:
            X = _sub(z["xb"], n)
            q, M = X[te], X[mem]
            k = _nearest(spec(q), spec(M), False)
            row[f"b{n}f"] = hit(k)
            row[f"b{n}f_tags"] = hit(rng.permutation(int(mem.sum()))[k])
            row[f"b{n}f_scr"] = hit(_nearest(spec(q[:, rng.permutation(n)]),
                                             spec(M), False))
            row[f"b{n}"] = hit(_nearest(q, M, True))
            row[f"b{n}n"] = hit(_nearest(q, M, False))
        rows.append(row)
    A = lambda k: np.array([r[k] for r in rows])                 # noqa: E731
    ch = A("chance")
    need = math.ceil(0.8 * n_worlds)
    void = []
    if A("b128f_tags").mean() > ch.mean() + 0.05:
        void.append("tags")
    if A("b128f_scr").mean() > ch.mean() + 0.10:
        void.append("scrambled")
    finds = bool(A("b128f").mean() >= 0.50
                 and ((A("b128f") - ch) >= 0.25).sum() >= need)
    loss = float(A("b128").mean() - A("b128f").mean())
    verdict = "VOID" if void else "ALIVE" if finds and loss <= 0.10 else "DEAD"
    out = {"rows": rows, "verdict": verdict, "void": void, "loss": loss,
           **{k: float(A(k).mean()) for k in rows[0] if k != "seed"}}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in out.items() if k != "rows"}, indent=1))
    with open(os.path.join(OUT, f"fitf_{first}_{n_worlds}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        for s in sys.argv[2:]:
            collect(int(s))
    elif cmd == "fitf":
        fitf(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "fit":
        fit(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
