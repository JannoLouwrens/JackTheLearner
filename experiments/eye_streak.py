"""LAB — the eye: is it azimuth in the floor band, not elevation, that tells
food from a wall?  (Vision encoder seat, held BY DEFAULT; no ledger row.)

PLAN part "Eye", right column, and PUBLISHED_PRIORS §12 "What the record does
NOT answer": *"Does 16x16 depth+material beat 16 rays at telling food from
wall? 5 seeds, CPU."*  This module answers that question and enters one
challenger of its own beside §12's candidate A.

THE BET.  His food is a 0.2 m thing on the floor.  At 16 azimuths (22.5 deg
apart) a ray touches it on a minority of frames beyond ~1 m WHATEVER the
elevation rows are, so a 16x16 grid spends 256 rays and still misses it.  A
"streak" — the same 256 rays as 128 azimuths x 2 rows (level, and +8 deg,
which clears food beyond ~1 m but still meets a wall) — touches it on nearly
every frame.  Biology: ground foragers carry a horizontal visual streak, not
an even retina.
PRIOR.  Habitat PointNav: depth SPL 0.79 vs RGB 0.46 vs blind 0.42 at 75M
steps (depth is the sense for getting around); "What if Eye" (2025, MuJoCo,
PPO): detection needed two 15x15 eyes, navigation ~10 eyes of 4 receptors —
resolution, not coverage, bought detection.  Neither says which GEOMETRY of a
ray retina finds a small floor object; nothing published at our eye height
(0.12 m).  Budget: this is a supervised probe with a privileged label (the
adult's ground truth around him, never in him) — minutes of CPU; it measures
what an eye CARRIES, not what he learns from it.

ARMS (all [distance/8 m, material] per ray, world-frame azimuths like
`w0.py:219`, cast from the seated eye point = body origin + RAY_Z_OFFSET):
  seated16   16 az x 1 level row                    16 rays   (the holder)
  grid16     16 az x 16 rows, -15..+45 deg          256 rays  (§12 cand. A)
  grid32     32 az x 32 rows, -15..+47 deg          1024 rays (§12 upper end)
  streak     128 az x 2 rows (0, +8 deg)            256 rays  (the challenger)
  row256     256 az x 1 level row                   256 rays  (ablation: is
                                                    the second row needed?)
WORLD.  `jack_env` at the entry rung the pilot lives on: difficulty 0.75,
random_layout (foods re-placed at every birth).  Random actions held 5
decisions; a new body every BOUT = 16 decisions, frames kept from decision 8
on (the body has settled to its living height).  Every arm reads the SAME
frames (paired).
LEARNER (from scratch, <= 81 numbers, one per world and arm).  A table
material -> P(a ray of this material ends on a floor food), counted over the
world's first 75% of bouts (the label is the adult's: ray end point within
0.25 m of a food centre).  On the last 25% of bouts (new food layouts) he
points along the NEAREST ray whose material the table calls food (P >= 0.5);
with none in view, along a random ray that touched anything.
SCORE (primary).  Test frames whose nearest floor food is within 4 m: HIT =
the bearing pointed at is within 20 deg of the bearing to ANY floor food.
CONTROLS THAT MUST FAIL.  (1) blind twin: the same rays cast with every floor
food made transparent — same walls, same non-food objects, no food; (2) the
table taught with its labels shuffled over the training rays.

TUNING SET (worlds 150-159; nothing below was read on 400-419):
  * Across worlds no single frame can tell food: material is a function of
    geom id, ids shift with each world's mutation (food is 0.86/0.43/0.80 in
    world 150, 0.43/0.80/0.37 in 154), and non-food objects share its shapes.
    Decoders trained on 6 worlds, tested on 4 others: HIT .41-.49 for every
    arm against .40-.43 blind.  So food is learned per world — hence the table.
  * Neural decoders per world (2x64 MLP, circular conv; 600 frames) scored
    .26-.68 with no order between arms: they measured the decoder.  Dropped.
  * Table, 10 worlds: seated16 .464, grid16 .466, grid32 .684, streak .874,
    row256 .893 (blind .18-.21, shuffled .19-.22).  Frames with any ray on a
    food 2-4 m away: .21 / .18 / .45 / .85 / .87.  The +8 deg row buys
    nothing over row256 on this ruler.

RULE — declared before worlds 400-419 are generated:
  20 worlds, 1000 frames each; unit of replication = a world.
  VOID if for streak, mean HIT - mean blind HIT < 0.10 (the probe reads the
  room, not the food) or the shuffled twin's mean HIT exceeds the blind
  twin's mean HIT + 0.05.
  STREAK ALIVE iff mean HIT(streak) - mean HIT(grid16) >= 0.05 with streak
  ahead on >= 16 of 20 worlds, AND mean HIT(streak) - mean HIT(seated16) >=
  0.10 with streak ahead on >= 16 of 20.  Otherwise DEAD.
  §12's question is read the same way: grid16 beats seated16 iff the lead is
  >= 0.05 with grid16 ahead on >= 16 of 20 worlds.
  Reported, deciding nothing: grid32, row256, HIT by distance band, and the
  learner-free fraction of frames on which any ray touches a floor food.
  What this cannot say: whether a BRAIN learns to use the eye (a table taught
  by an adult's label is the ceiling of a material look-up, not his learning),
  nor what each geometry costs in his loop (J6 item 3 times that).

RESULT (worlds 400-419, 11-16; rows /data/tmp/eye_streak/fit_400_20.json):
STREAK ALIVE. mean HIT seated16 .528, grid16 .517, grid32 .758, streak
  .900, row256 .907 (sd .04-.09); blind .19-.21, shuffled .20-.22 (both
  fail, as they must).  streak ahead of seated16 20/20, grid16 20/20, grid32
  20/20, row256 6/20.  §12's question: grid16 does NOT beat seated16 (-.012,
  ahead 7/20) - elevation rows buy nothing; azimuth in the floor band does.
  Food 2-4 m away: .43 / .40 / .65 / .89 / .89; 4-8 m: .24 / .23 / .38 /
  .73 / .80.  The +8 deg row buys nothing on this ruler (row256 >= streak).

DIAL — declared before worlds 420-429 are generated (EYE_STREAK_ARMS=dial):
  how many azimuths does the level row need?  row16/32/64/128/256, same
  frames, same table learner, 10 worlds.  Reported: mean HIT each, and the
  KNEE = the smallest row within 0.05 mean HIT of row256.  Decides nothing by
  itself; it sizes the eye the workers build (2 numbers per ray into the
  brain).  VOID unless the dial's row16 and row256 arrays for world 400
  (re-collected) equal the race's seated16 and row256 arrays bit for bit —
  the dial reads the same frames or it reads nothing.  (Replaces, before any
  dial world existed, a first draft that compared means across different
  worlds: 10 worlds cannot hold a 0.03 band.)

DIAL RESULT (worlds 420-429, 11-16; identity on world 400 exact;
  /data/tmp/eye_streak_dial/fit_420_10.json): mean HIT row16 .540, row32
  .696, row64 .856, row128 .909, row256 .925 (blind .17-.20, shuffled
  .18-.20).  KNEE = row128.  Food 4-8 m away: .27 / .38 / .56 / .77 / .86.

    nice -n 19 $PY -m experiments.eye_streak collect <seed> [<seed> ...]
    nice -n 19 $PY -m experiments.eye_streak fit <first_seed> <n_worlds>
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

OUT = os.environ.get("EYE_STREAK_OUT", "/data/tmp/eye_streak")
N_FRAMES = 1000
BOUT, KEEP_FROM, HOLD = 16, 8, 5
FLOOR_Z = 0.5
NEAR_M, HIT_DEG = 4.0, 20.0
MAX_FOODS = 5

# name -> (n_az, elevations in degrees)
ARMS = {
    "seated16": (16, (0.0,)),
    "grid16": (16, tuple(-15.0 + 4.0 * k for k in range(16))),
    "grid32": (32, tuple(-15.0 + 2.0 * k for k in range(32))),
    "streak": (128, (0.0, 8.0)),
    "row256": (256, (0.0,)),
}
# The azimuth dial (rule in the docstring's DIAL section): one level row only.
if os.environ.get("EYE_STREAK_ARMS") == "dial":
    ARMS = {f"row{n}": (n, (0.0,)) for n in (16, 32, 64, 128, 256)}
    OUT = os.environ.get("EYE_STREAK_OUT", "/data/tmp/eye_streak_dial")


def ray_dirs(n_az: int, elevs) -> np.ndarray:
    """(rows, n_az, 3) unit vectors; azimuth k at 2*pi*k/n_az, world frame."""
    out = np.empty((len(elevs), n_az, 3))
    az = 2 * math.pi * np.arange(n_az) / n_az
    for r, e in enumerate(elevs):
        ce, se = math.cos(math.radians(e)), math.sin(math.radians(e))
        out[r, :, 0], out[r, :, 1], out[r, :, 2] = ce * np.cos(az), ce * np.sin(az), se
    return out


def collect(seed: int, n: int = N_FRAMES) -> str:
    import mujoco
    from experiments import w0 as W0MOD
    from experiments.fast_sim import _multi_ray
    from experiments.jack_env import JackEnv
    from experiments.w0 import random_action

    env = JackEnv(seed=seed, difficulty=0.75, random_layout=True)
    w = env.w
    m, d = w.model, w.data
    foods = [gid for name, (gid, _) in w.drives._food.items()
             if name in env.floor_foods]
    assert foods and all(d.geom_xpos[g][2] < FLOOR_Z for g in foods), foods
    dirs = {a: ray_dirs(*spec) for a, spec in ARMS.items()}
    flat = {a: np.ascontiguousarray(v.reshape(-1, 3)).ravel() for a, v in dirs.items()}
    groups = W0MOD._RAY_GROUPS
    tex = w._tex
    rng = np.random.RandomState(seed + 7)

    def cast(arm, pos):
        nray = flat[arm].size // 3
        gid = np.full(nray, -1, dtype=np.int32)
        dist = np.zeros(nray, dtype=np.float64)
        # J20 (4): the eye's own compat (either mj_multiRay signature). Its
        # cutoff is mjMAXVAL, not RAY_MAX_RANGE: a hit past the range is still
        # masked as a miss below, so the features are the same.
        _multi_ray(m, d, pos, flat[arm], groups, gid, dist, nray)
        miss = (gid < 0) | (dist < 0) | (dist > W0MOD.RAY_MAX_RANGE)
        dn = np.where(miss, 1.0, dist / W0MOD.RAY_MAX_RANGE)
        tx = np.where(miss, 0.0, tex[np.maximum(gid, 0)])
        rows, n_az = dirs[arm].shape[:2]
        feat = np.stack([dn, tx], -1).reshape(rows, n_az, 2).astype(np.float32)
        return feat, bool(np.isin(gid[~miss], foods).any())

    X = {a: [] for a in ARMS}
    XB = {a: [] for a in ARMS}
    SEEN = {a: [] for a in ARMS}
    V, Z = [], []
    a_act = random_action(rng)
    t = 0
    t0 = time.time()
    while len(V) < n:
        if t % BOUT == 0:
            w.respawn()
        if t % HOLD == 0:
            a_act = random_action(rng)
        if t % BOUT >= KEEP_FROM:
            mujoco.mj_forward(m, d)
            pos = np.array(d.xpos[w.rover_bid], dtype=float)
            pos[2] += W0MOD.RAY_Z_OFFSET
            vec = np.full((MAX_FOODS, 2), np.nan, dtype=np.float32)
            for i, g in enumerate(foods):
                vec[i] = d.geom_xpos[g][:2] - pos[:2]
            V.append(vec)
            Z.append(pos[2])
            for a in ARMS:
                f, seen = cast(a, pos)
                X[a].append(f)
                SEEN[a].append(seen)
            keep = m.geom_group[foods].copy()
            m.geom_group[foods] = 5            # outside _RAY_GROUPS: transparent
            try:
                for a in ARMS:
                    f, seen = cast(a, pos)
                    assert not seen
                    XB[a].append(f)
            finally:
                m.geom_group[foods] = keep
        w.decide(a_act)
        t += 1
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"world_{seed}.npz")
    np.savez_compressed(
        path, V=np.array(V), Z=np.array(Z),
        **{f"x_{a}": np.array(X[a]) for a in ARMS},
        **{f"b_{a}": np.array(XB[a]) for a in ARMS},
        **{f"s_{a}": np.array(SEEN[a]) for a in ARMS})
    print(f"world {seed}: {len(V)} frames, {t} decisions, {len(foods)} foods, "
          f"eye z median {np.median(Z):.3f}, {time.time() - t0:.0f}s -> {path}",
          flush=True)
    return path


# ── the decoders ────────────────────────────────────────────────────────
def _targets(V):
    """nearest-food unit vector, its distance, and all food bearings."""
    dist = np.linalg.norm(V, axis=2)                      # (n, MAX_FOODS), nan pad
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = dist.argmin(1)
    near = V[np.arange(len(V)), k]
    dn = dist[np.arange(len(V)), k]
    unit = near / np.maximum(dn[:, None], 1e-9)
    bearing = np.arctan2(V[:, :, 1], V[:, :, 0])          # nan where padded
    return unit.astype(np.float32), dn, bearing, dist


def _hit(pred, bearing, dist):
    ang = np.arctan2(pred[:, 1], pred[:, 0])[:, None]
    diff = np.abs((bearing - ang + np.pi) % (2 * np.pi) - np.pi)
    ok = (diff <= math.radians(HIT_DEG)) & (dist <= 8.0)
    return np.nan_to_num(ok, nan=0).any(1)


FOOD_R = 0.25            # m: a ray's end point this near a food centre "is food"
SPLIT = 0.75             # a world's early bouts teach the table, the rest test it


def _ray_table(arm, X, V):
    """Per ray: azimuth, material (2 decimals), distance, and (privileged, for
    teaching only) whether its end point lies on a floor food."""
    n_az, elevs = ARMS[arm]
    az = 2 * math.pi * np.arange(n_az) / n_az
    ce = np.cos(np.radians(np.array(elevs)))[None, :, None]
    dn, tx = X[..., 0], np.round(X[..., 1], 2)
    hit = tx > 0
    r = dn * 8.0 * ce                                       # horizontal reach
    px, py = r * np.cos(az)[None, None, :], r * np.sin(az)[None, None, :]
    food = np.zeros(dn.shape, dtype=bool)
    for i in range(V.shape[1]):
        vx, vy = V[:, i, 0][:, None, None], V[:, i, 1][:, None, None]
        with np.errstate(invalid="ignore"):
            food |= np.hypot(px - vx, py - vy) <= FOOD_R
    food &= hit
    n = len(X)
    return (np.broadcast_to(az, dn.shape).reshape(n, -1), tx.reshape(n, -1),
            dn.reshape(n, -1), hit.reshape(n, -1), food.reshape(n, -1))


def _decode(arm, X, V, tr, te, rng, shuffle=False):
    """The learner: a table material -> P(this ray ends on food), counted over
    the training frames (from scratch, <= 81 numbers). At test he points along
    the NEAREST ray whose material the table calls food (P >= 0.5); with no such
    ray, along a random ray that touched anything."""
    az, tx, dn, hit, food = _ray_table(arm, X, V)
    k = np.round(tx * 100).astype(int)
    lab = food[tr][hit[tr]]
    if shuffle:
        lab = lab[rng.permutation(len(lab))]
    tot = np.bincount(k[tr][hit[tr]], minlength=101).astype(float)
    pos = np.bincount(k[tr][hit[tr]], weights=lab, minlength=101)
    p = pos / (tot + 1.0)
    pred = np.zeros((len(az[te]), 2), dtype=np.float32)
    for j, (a_, k_, d_, h_) in enumerate(zip(az[te], k[te], dn[te], hit[te])):
        if not h_.any():
            ang = rng.uniform(0, 2 * math.pi)
        else:
            sc = np.where(h_, p[k_], -1.0)
            if sc.max() >= 0.5:
                c = np.flatnonzero(sc >= 0.5)
                ang = a_[c[np.argmin(d_[c])]]
            else:
                c = np.flatnonzero(h_)
                ang = a_[c[rng.randint(len(c))]]
        pred[j] = (math.cos(ang), math.sin(ang))
    return pred, int((p >= 0.5).sum())


def fit(first: int, n_worlds: int) -> dict:
    """One table PER WORLD and arm: taught on the world's early bouts, tested on
    its last ones (every bout is a new food layout). Unit of replication: a world."""
    seeds = list(range(first, first + n_worlds))
    res = {"worlds": seeds, "rows": []}
    per = BOUT - KEEP_FROM
    for s in seeds:
        z = np.load(os.path.join(OUT, f"world_{s}.npz"))
        V = z["V"]
        n = len(V)
        a = int(SPLIT * n) // per * per
        tr, te = slice(0, a), slice(a, n)
        Y, dn, bearing, dist = _targets(V)
        d = dn[te]
        row = {"world": s, "n_test_near": int((d <= NEAR_M).sum())}

        def score(pred, lo=0.0, hi=NEAR_M):
            msk = (d > lo) & (d <= hi)
            return float(_hit(pred[msk], bearing[te][msk], dist[te][msk]).mean())

        for arm in ARMS:
            rng = np.random.RandomState(s)
            seen = z[f"s_{arm}"][te]
            pte, nfood = _decode(arm, z[f"x_{arm}"], V, tr, te, rng)
            bte, _ = _decode(arm, z[f"b_{arm}"], V, tr, te, rng)
            ste, _ = _decode(arm, z[f"x_{arm}"], V, tr, te, rng, shuffle=True)
            row[arm] = {
                "food_materials": nfood, "hit": score(pte),
                "blind_hit": score(bte), "shuffled_hit": score(ste),
                "hit_0_2": score(pte, 0.0, 2.0), "hit_2_4": score(pte, 2.0, 4.0),
                "hit_4_8": score(pte, 4.0, 8.0),
                "seen_0_2": float(seen[d <= 2.0].mean()),
                "seen_2_4": float(seen[(d > 2.0) & (d <= 4.0)].mean()),
                "seen_4_8": float(seen[(d > 4.0) & (d <= 8.0)].mean()),
            }
        print(s, {x: round(row[x]["hit"], 3) for x in ARMS}, flush=True)
        res["rows"].append(row)

    def mean(arm, key):
        return float(np.mean([r[arm][key] for r in res["rows"]]))

    def ahead(x, y):
        return int(sum(r[x]["hit"] > r[y]["hit"] for r in res["rows"]))

    keys = ("hit", "blind_hit", "shuffled_hit", "hit_0_2", "hit_2_4", "hit_4_8",
            "seen_0_2", "seen_2_4", "seen_4_8", "food_materials")
    sm = {x: {q: round(mean(x, q), 4) for q in keys} for x in ARMS}
    for x in ARMS:
        sm[x]["sd_hit"] = round(float(np.std([r[x]["hit"] for r in res["rows"]], ddof=1)), 4)
    res["summary"] = sm
    if "streak" not in ARMS:                    # the azimuth dial: no verdict
        top = sm["row256"]["hit"]
        res["knee"] = next(x for x in ARMS if top - sm[x]["hit"] <= 0.05)
        with open(os.path.join(OUT, f"fit_{first}_{n_worlds}.json"), "w") as f:
            json.dump(res, f, indent=1)
        for x in ARMS:
            print(x, json.dumps(sm[x]))
        print("KNEE", res["knee"], flush=True)
        return res
    need = math.ceil(0.8 * n_worlds)
    st = sm["streak"]
    void = (st["hit"] - st["blind_hit"] < 0.10
            or st["shuffled_hit"] > st["blind_hit"] + 0.05)
    alive = (st["hit"] - sm["grid16"]["hit"] >= 0.05
             and ahead("streak", "grid16") >= need
             and st["hit"] - sm["seated16"]["hit"] >= 0.10
             and ahead("streak", "seated16") >= need)
    res["summary"] = sm
    res["streak_ahead_of"] = {x: ahead("streak", x) for x in ARMS if x != "streak"}
    res["grid16_ahead_of_seated16"] = ahead("grid16", "seated16")
    res["grid16_beats_seated16"] = bool(
        sm["grid16"]["hit"] - sm["seated16"]["hit"] >= 0.05
        and ahead("grid16", "seated16") >= need)
    res["verdict"] = "VOID" if void else ("ALIVE" if alive else "DEAD")
    path = os.path.join(OUT, f"fit_{first}_{n_worlds}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    for x in ARMS:
        print(x, json.dumps(sm[x]))
    print("VERDICT", res["verdict"], "streak ahead of", res["streak_ahead_of"],
          "| grid16 beats seated16:", res["grid16_beats_seated16"],
          f"({res['grid16_ahead_of_seated16']}/{n_worlds})", "->", path, flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        for s in sys.argv[2:]:
            collect(int(s))
    elif sys.argv[1] == "fit":
        fit(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
