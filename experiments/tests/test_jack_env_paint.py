"""J24 (0)-(1) proof: jack_env's `paint` option ("geom" default, "kind") and
the `eye="bins16"` computed from the kind-painted row (ruling 111).

    PYTHONPATH=. /data/venvs/jack-std/bin/python -m experiments.tests.test_jack_env_paint [N=150]

The world is J24's: the pilot's (d 0.75, random layout + needs, spawn clear,
food_floor_only, body eye, body motor, lazy audio, native substep), seeds
941/942/943, random actions, life ends crossed.

(a) IDENTITY, paint="geom" is today's world bit for bit: the obs/reward/end
    stream of four cases (16-ray eye seed 153; row128 pilot worlds 941-943)
    with paint="geom" passed == with nothing passed == the sha256 taken from
    HEAD b06d830b's code BEFORE this option was written (box, aarch64,
    N = 150 only). Control that must differ: another action seed.
(b) THE KIND PAINT, against the simulator as judge (`w._row_gid`): same
    seed, same actions, paint="kind" beside paint="geom". Must hold on every
    look: every other sense equal byte for byte; every ray's distance equal;
    a ray that misses reads all-zero channels, a ray that hits exactly one;
    every ray on a FLOOR FOOD lights the food channel and no ray on any
    other geom does; water / wall / panel channels fire only on
    `pool_water` / `wall0-3` / `noise_panel`. The food channel must actually
    be seen lit on >= 2 different food geoms per world (else the row proves
    nothing). The food channel index is the same in all three worlds.
    Control that must FAIL the same "one look for food" test: under
    paint="geom" the foods of one world carry different texture values, and
    the three worlds' food value-sets differ.
(c) NO LEAK: the kind row is a function of (distance, kind of geom hit)
    only — re-derived here from `_row_gid`/`_row_dist` and the geom NAMES,
    without `fast_sim.kind_table`, and compared exactly on every look.
(d) bins16, planted on hand-made rows (no world): an empty row -> 80 zeros;
    food on rays 127, 0, 1 at distance 0.25 of range -> food bin 0 = 0.75
    and the other 79 numbers zero; the same rays painted WALL light wall bin
    0 and no food bin (control); ray 4 belongs to bin 1 and ray 124 to bin
    0; the nearer of two foods in one bin wins.
(e) bins16 in his world: JackEnv(eye="bins16", paint="kind") has vision 80
    wide, equals `bins16` of the kind row of the twin env on every look,
    other senses equal; the set of lit food bins equals the set the judge
    derives from `_row_gid`. PLANTED: obj0 moved to 2 m straight ahead of
    his body (0.4 m in front of whatever stands nearer than 2.2 m) -> food
    bin 0 reads 1 - (nearest food ray of bin 0)/8 and ray 0's geom is obj0. Options that must refuse: paint="kind" on the 16-ray eye;
    bins16 on the geom paint.

RESULT (2026-11-26, box, jack-std, mujoco 3.17.0, nice 19, N = 150, 30
CPU-s): PASS. (a) 4/4 == default == the pre-option sha; control differs.
(b) 941/942/943: all clauses hold on 151 looks each; food in sight on
148/139/144 looks, 3/2/2 of 3 food geoms seen; food channel on a non-food
ray 0 times; control: geom paint's foods read (.43 .49 .86) / (.43 .80 .86)
/ (.37 .43 .80). (c) 3/3. (d) 5/5. (e) 3/3 + plants at 2.00 / 1.30 / 2.00 m.
First run FAILED (e) on 942 alone: the plant at a fixed 2 m sat behind a
thing 1.70 m ahead — an apparatus error (the eye rightly did not see through
it); the plant now goes in front of whatever is ahead. No clause was moved.
"""
import platform
import sys

import mujoco
import numpy as np

from experiments import fast_sim as FS
from experiments.jack_env import JackEnv
from experiments.tests.test_jack_env_paint_cases import (GEOM_CASES,
                                                         geom_stream_sha,
                                                         pilot_kw)
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEEDS = (941, 942, 943)
GOLDEN_N = 150
GOLDEN = {  # HEAD b06d830b, box (aarch64), jack-std, mujoco 3.17.0, N = 150
    "rays16-base-153": "12b7378fbfb1b110d916710ae58181ef82ec0f44e74bd5e7858d9eb08cd18dc0",
    "row128-pilot-941": "ce3036f73732fbb93449bb6de1b0de04ec17744f6749af083622e08512572201",
    "row128-pilot-942": "4b00d2bb8c9c29ca096eb758c809b4513cf9bbbba5d0b634c6f09e4d5c41b49d",
    "row128-pilot-943": "063e722fdf74962baed5fc9a925dffe32782640c1fb32200b715c36c97881028",
}
K = FS.N_KINDS
FOOD, WATER, OBST, WALL, PANEL = (FS.KINDS.index(k) for k in
                                  ("food", "water", "obstacle", "wall", "panel"))


def names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
            for g in range(m.ngeom)]


def food_gids(env):
    return {int(env.w.drives._food[n][0]) for n in env.floor_foods
            if n in env.w.drives._food}


def judge_row(env) -> np.ndarray:
    """(c): the kind row rebuilt from the simulator's own ray results and
    geom names — independent of fast_sim.kind_table."""
    from experiments import w0 as W
    w = env.w
    nm = names(w.model)
    foods = food_gids(env)
    gid, dist = w._row_gid, w._row_dist
    out = np.zeros((len(gid), 1 + K), dtype=np.float32)
    for k in range(len(gid)):
        g, d = int(gid[k]), float(dist[k])
        if d < 0 or d > W.RAY_MAX_RANGE or g < 0:
            out[k, 0] = 1.0
            continue
        out[k, 0] = d / W.RAY_MAX_RANGE
        if g in foods:
            c = FOOD
        elif nm[g] == "pool_water":
            c = WATER
        elif nm[g] in ("wall0", "wall1", "wall2", "wall3"):
            c = WALL
        elif nm[g] == "noise_panel":
            c = PANEL
        else:
            c = OBST
        out[k, 1 + c] = 1.0
    return out.reshape(-1)


def main():
    ok = True
    # (a) ----------------------------------------------------------------
    ref = geom_stream_sha(N)
    got = geom_stream_sha(N, paint="geom")
    for name, seed, kw in GEOM_CASES:
        same = ref[name] == got[name]
        line = f"(a) {name}: paint='geom' == default {same}"
        if N == GOLDEN_N and platform.machine() == "aarch64":
            g = ref[name] == GOLDEN[name]
            line += f" | == HEAD-before-the-option sha {g}"
            same &= g
        else:
            line += " | golden not compared (needs N=150 on the box)"
        print(line)
        ok &= same
    from experiments.tests.test_jack_env_paint_cases import stream_sha
    ctl = stream_sha(JackEnv(seed=941, **pilot_kw(), eye="row128"), 941, N, 5)
    print(f"(a) control, another action seed differs: {ctl != ref['row128-pilot-941']}")
    ok &= ctl != ref["row128-pilot-941"]

    # (b) (c) (e) ---------------------------------------------------------
    food_sets = []
    for seed in SEEDS:
        kw = pilot_kw()
        eg = JackEnv(seed=seed, eye="row128", **kw)
        ek = JackEnv(seed=seed, eye="row128", paint="kind", **kw)
        eb = JackEnv(seed=seed, eye="bins16", paint="kind", **kw)
        nv = 128 * (1 + K)
        shape_ok = (ek.observation_space.shape[0] == eg.observation_space.shape[0] - 256 + nv
                    and eb.observation_space.shape[0] == eg.observation_space.shape[0] - 256 + 80)
        foods = food_gids(ek)
        nm = names(ek.w.model)
        assert foods == food_gids(eg) == food_gids(eb) and len(foods) == 3
        tex = sorted(round(float(eg.w._tex[g]), 4) for g in foods)
        food_sets.append(tuple(tex))
        rngs = [np.random.RandomState(seed + 3000) for _ in range(3)]
        og, _ = eg.reset(seed=seed)
        okd, _ = ek.reset(seed=seed)
        ob, _ = eb.reset(seed=seed)
        rest = dist = onehot = foodch = named = judge = True
        bins_eq = bins_rest = bins_judge = True
        seen_food, lit_other = set(), 0
        food_looks = 0
        for t in range(N + 1):
            vg, rg = og[:256], og[256:]
            vk, rk = okd[:nv].reshape(128, 1 + K), okd[nv:]
            rest &= np.array_equal(rg, rk)
            dist &= np.array_equal(vg[0::2], vk[:, 0])
            gid, dd = ek.w._row_gid, ek.w._row_dist
            miss = (dd < 0) | (dd > 8.0) | (gid < 0)
            ch = vk[:, 1:]
            onehot &= bool(np.all(ch[miss] == 0) and np.all(ch[~miss].sum(1) == 1)
                           and np.all((ch == 0) | (ch == 1))
                           and np.all(vk[miss, 0] == 1.0))
            is_food = np.isin(gid, list(foods)) & ~miss
            foodch &= bool(np.all(ch[is_food, FOOD] == 1))
            lit_other += int(np.sum(ch[~is_food, FOOD]))
            seen_food |= {int(g) for g in gid[is_food]}
            food_looks += bool(is_food.any())
            for c, allowed in ((WATER, {"pool_water"}), (PANEL, {"noise_panel"}),
                               (WALL, {"wall0", "wall1", "wall2", "wall3"})):
                named &= all(nm[int(g)] in allowed for g in gid[(ch[:, c] == 1)])
            judge &= np.array_equal(okd[:nv], judge_row(ek))
            # (e) bins
            bins_eq &= np.array_equal(ob[:80], FS.bins16(okd[:nv]))
            bins_rest &= np.array_equal(ob[80:], rk)
            want = {((k + 4) % 128) // 8 for k in np.flatnonzero(is_food)
                    if dd[k] < 8.0}
            bins_judge &= set(np.flatnonzero(ob[:16] > 0).tolist()) == want
            if t == N:
                break
            a = [random_action(r) for r in rngs][0]
            og, _, te, tr, _ = eg.step(a)
            okd, _, te2, tr2, _ = ek.step(a)
            ob, _, te3, tr3, _ = eb.step(a)
            assert (te, tr) == (te2, tr2) == (te3, tr3)
            if te or tr:
                og, _ = eg.reset()
                okd, _ = ek.reset()
                ob, _ = eb.reset()
        one_look_geom = len(set(tex)) == 1
        b = (shape_ok and rest and dist and onehot and foodch and lit_other == 0
             and named and len(seen_food) >= 2)
        print(f"(b) seed {seed}: widths ok={shape_ok} other senses equal={rest} "
              f"distances equal={dist} miss=zeros/hit=one-hot={onehot} every food "
              f"ray lit={foodch} food channel on non-food rays={lit_other} "
              f"water/wall/panel only on their geoms={named} | looks with food "
              f"in sight {food_looks}/{N + 1}, food geoms seen {len(seen_food)}/3 "
              f"| control, geom paint gives the foods one look: {one_look_geom} "
              f"(values {tex})")
        ok &= b and not one_look_geom
        print(f"(c) seed {seed}: kind row == judge's rebuild from ray results "
              f"and geom names, every look: {judge}")
        ok &= judge
        print(f"(e) seed {seed}: bins16 env == bins16(kind row) {bins_eq}, other "
              f"senses equal {bins_rest}, lit food bins == judge's {bins_judge}")
        ok &= bins_eq and bins_rest and bins_judge

        # (e) planted: obj0 2 m straight ahead of the body
        w, m = eb.w, eb.w.model
        yaw = FS.body_yaw(w)
        x, y = w.data.xpos[w.rover_bid][:2]
        qa, da, z0, quat0 = eb._food_rest["obj0"]
        # park obj0 out of the way, read how far ray 0 sees, then plant it
        # 2 m ahead — or 0.4 m in front of whatever stands nearer than 2.2 m
        # (world 942: a thing at 1.70 m hid a food planted at 2 m).
        w.data.qpos[qa:qa + 3] = (x - 3.0 * np.cos(yaw), y - 3.0 * np.sin(yaw), z0)
        mujoco.mj_forward(m, w.data)
        w.observe()
        free = float(w._row_dist[0])
        r_plant = 2.0 if (free < 0 or free > 2.2) else free - 0.4
        assert r_plant >= 0.6, f"seed {seed}: nothing fits ahead ({free:.2f} m)"
        w.data.qpos[qa:qa + 3] = (x + r_plant * np.cos(yaw), y + r_plant * np.sin(yaw), z0)
        w.data.qpos[qa + 3:qa + 7] = quat0
        w.data.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(m, w.data)
        v = w.observe()["vision"]
        g0, d0 = int(w._row_gid[0]), float(w._row_dist[0])
        hit = g0 == int(w.drives._food["obj0"][0])
        near = min(float(w._row_dist[k]) for k in (list(range(124, 128)) + [0, 1, 2, 3])
                   if int(w._row_gid[k]) in foods)if hit else float("nan")
        val = bool(hit and abs(v[0] - (1.0 - near / 8.0)) < 1e-6
                   and r_plant - 0.3 < d0 < r_plant)
        print(f"(e) seed {seed} planted at {r_plant:.2f} m: ray 0 hits obj0 {hit} at {d0:.3f} m; "
              f"food bin 0 = {v[0]:.4f} (want {1.0 - near / 8.0:.4f}) -> {val}")
        ok &= val
    worlds_differ = len(set(food_sets)) == len(food_sets)
    print(f"(b) control: geom paint's food value-sets differ world to world: "
          f"{worlds_differ} {food_sets}; kind paint: channel {FOOD} in all three")
    ok &= worlds_differ

    # (d) -----------------------------------------------------------------
    def row(hits):          # {ray: (dist_frac, kind)}
        r = np.zeros((128, 1 + K), dtype=np.float32)
        r[:, 0] = 1.0
        for k, (d, c) in hits.items():
            r[k, 0] = d
            r[k, 1 + c] = 1.0
        return r.reshape(-1)

    e = FS.bins16(row({}))
    d1 = e.shape == (80,) and not e.any()
    f = FS.bins16(row({127: (0.25, FOOD), 0: (0.25, FOOD), 1: (0.25, FOOD)}))
    d2 = f[0] == 0.75 and np.count_nonzero(f) == 1
    wl = FS.bins16(row({127: (0.25, WALL), 0: (0.25, WALL), 1: (0.25, WALL)}))
    d3 = not wl[:16].any() and wl[16 * WALL] == 0.75 and np.count_nonzero(wl) == 1
    e4 = FS.bins16(row({4: (0.5, FOOD)}))
    e124 = FS.bins16(row({124: (0.5, FOOD)}))
    e123 = FS.bins16(row({123: (0.5, FOOD)}))
    d4 = (np.flatnonzero(e4).tolist() == [1] and np.flatnonzero(e124).tolist() == [0]
          and np.flatnonzero(e123).tolist() == [15])
    two = FS.bins16(row({8: (0.5, FOOD), 9: (0.1, FOOD)}))
    d5 = abs(two[1] - 0.9) < 1e-6 and np.count_nonzero(two) == 1
    print(f"(d) empty row -> zeros {d1}; food ahead -> food bin 0 only {d2}; "
          f"control, wall ahead lights no food bin {d3}; bin edges (ray 4 -> bin 1, "
          f"124 -> 0, 123 -> 15) {d4}; nearer food wins {d5}")
    ok &= d1 and d2 and d3 and d4 and d5

    # (e) refusals
    ref_ok = True
    for kw in (dict(paint="kind"), dict(eye="bins16"), dict(paint="hue")):
        try:
            JackEnv(seed=153, j0=1.0, alpha=1.0, **kw)
            ref_ok = False
        except ValueError:
            pass
    print(f"(e) refuses kind paint on the 16-ray eye, bins16 on the geom paint, "
          f"an unknown paint: {ref_ok}")
    ok &= ref_ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
