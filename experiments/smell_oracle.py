"""LAB — can a perfect smell-follower eat in W0? (diagnosis for J1; no ledger row)

The smell race (smell_sense.py) was KILLED at 13k: the sense is informative
(bilateral points to food on 99.5 % of poses) yet the AWR learner did not eat
more with it than with a shuffled control. Before blaming the learner, ask
whether the gradient is USABLE through his body at all. A hand-written policy
is NOT a part of him — it is an instrument here, like a probe.

A structural suspect read off the code: `W0.decide` applies the drive
`action[6:8]` as a WORLD-frame xy force (xfrc_applied), while the nose reads
left/right in the BODY frame (+x of the rover body), and proprio carries no
heading (arm slides, velocities, torso z, upright). If his yaw drifts, the map
from "left nostril stronger" to "which drive component to push" changes under
him and nothing he senses says how. So the arms:

  oracle_yaw    bilateral gradient rotated to world by the TRUE yaw (privileged:
                what a body-frame drive or a heading sense would allow)
  blind_fixed   bilateral gradient used as if yaw were 0 (what a yaw-blind
                policy can learn: the best fixed map)
  klinotaxis    yaw-free: keep the drive direction while d/dt food > 0, pick a
                new random direction when it falls (run-and-tumble, E. coli)
  random        a new random drive direction every 10 decisions (null)

Arms 0:6 are held at mid-range (0) for all arms. Reading per arm/seed: meals
(dr.ate_total), lives, mean distance to nearest floor food, and yaw spread
(std of unwrapped yaw over the run) — if yaw barely moves, the mismatch is not
the cause and blind_fixed should match oracle_yaw.

NOT A GATE. It answers "world or learner": if oracle_yaw / klinotaxis eat far
more than random, the world is solved and the learner is the bottleneck.

    nice -n 19 $PY -m experiments.smell_oracle <decisions> <seed> [<seed>...]

RESULT v1 (11-12 14:01, 4k decisions, s91/92/93, meals): oracle_yaw 0/0/0,
blind_fixed 0/0/0, klinotaxis 13/2/3, random 0/1/13. READ AS AN APPARATUS
BUG, not a finding about the world: the v1 "oracle" pushed along body +x and
steered by 4*(cl-cr)/(cl+cr) <= ~0.16 rad, but a WORLD-frame force with no
torque never turns the body, so the steer never corrects anything and both
yaw arms push in a near-fixed line into a wall (d_mean 3.2-4.5 m). Also: 1-2
lives per 4k decisions makes meals a birth lottery (random s93 = 13 from one
lucky birth). v1 can tell nothing apart.

V2 (`episodes` mode) fixes both. Arms that USE the drive's real geometry:
  goto_true     privileged: push straight at the nearest READY floor food
                (reachability ceiling: can this body get to food at all?)
  nose_yaw      nose + TRUE yaw: the lateral component from sign(cl-cr)
                along the body's left axis rotated to world, the forward
                component kept while d/dt C > 0 and flipped when it falls.
                Uses only what a heading sense would add to his senses.
  nose_blind    the same, but yaw assumed 0 (= his birth yaw, since _place
                sets the quaternion to identity): what a yaw-blind map gets
  klinotaxis    yaw-free run-and-tumble on d/dt C (as v1)
  random        new random direction every 10 decisions (null)
Protocol: EPISODES, not lives — respawn uniformly (w.respawn()) every E
decisions or on death; reading per arm/seed = fraction of episodes with a meal
(ate_total rises), and median min-distance to ready floor food. Many births
per arm, so the birth lottery averages out. NOT A GATE; nothing in him moves.

    nice -n 19 $PY -m experiments.smell_oracle episodes <n_ep> <E> <seed>...
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

ARMS = ("oracle_yaw", "blind_fixed", "klinotaxis", "random")
OUT = "/data/tmp/smell_sense/oracle.jsonl"


def run_arm(arm: str, seed: int, n: int) -> dict:
    from . import smell_sense as S
    from . import w0 as W
    if not getattr(W.W0, "_oracle_installed", False):   # wrap observe ONCE:
        S.install("smell", seed)                         # a double wrap reads
        W.W0._oracle_installed = True                    # the nose twice/step
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w, dr = L.w, L.w.drives
    rng = np.random.RandomState(1000 + seed)
    floor = [g for _, (g, _) in dr._food.items()
             if w.data.geom_xpos[g][2] <= 0.5]
    a = np.zeros(8)
    heading = rng.uniform(-math.pi, math.pi)
    yaws, dsum, lives = [], 0.0, 1
    t0 = time.perf_counter()
    for k in range(n):
        o = w.observe()["smell"]
        cl, cr, dd = float(o[0]), float(o[4]), float(o[8])
        mat = np.array(w.data.xmat[w.rover_bid]).reshape(3, 3)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        yaws.append(yaw)
        if arm in ("oracle_yaw", "blind_fixed"):
            # body frame: +x forward, +y left. Push forward, steer to the
            # stronger nostril.
            g = np.array([1.0, 4.0 * (cl - cr) / max(1e-6, cl + cr)])
            th = math.atan2(g[1], g[0]) + (yaw if arm == "oracle_yaw" else 0.0)
            a[6:8] = [math.cos(th), math.sin(th)]
        elif arm == "klinotaxis":
            if dd <= 0.0:
                heading = rng.uniform(-math.pi, math.pi)
            a[6:8] = [math.cos(heading), math.sin(heading)]
        else:
            if k % 10 == 0:
                heading = rng.uniform(-math.pi, math.pi)
            a[6:8] = [math.cos(heading), math.sin(heading)]
        w.decide(a)
        if w.died_this_decision:
            lives += 1
        p = w.data.xpos[w.rover_bid]
        dsum += min(float(np.hypot(*(w.data.geom_xpos[g][:2] - p[:2])))
                    for g in floor)
    rec = {"arm": arm, "seed": seed, "decisions": n,
           "meals": int(sum(dr.ate_total.values())), "lives": lives,
           "d_mean": round(dsum / n, 2),
           "yaw_std": round(float(np.std(np.unwrap(yaws))), 3),
           "wall_s": round(time.perf_counter() - t0, 1)}
    return rec


ARMS2 = ("goto_true", "nose_yaw", "nose_blind", "klinotaxis", "random")
OUT2 = "/data/tmp/smell_sense/oracle2.jsonl"


def run_episodes(arm: str, seed: int, n_ep: int, E: int) -> dict:
    from . import smell_sense as S
    from . import w0 as W
    if not getattr(W.W0, "_oracle_installed", False):
        S.install("smell", seed)
        W.W0._oracle_installed = True
    from .tests.lf_03_learner_lives import _calibration
    from .tests.lf_04_living_teaches import _Life4
    j0, alpha, prov = _calibration()
    if j0 is None:
        raise SystemExit(f"PS.01 calibration refused: {prov}")
    L = _Life4(seed, j0, alpha, train=False)
    w, dr = L.w, L.w.drives
    w.observe()
    rng = np.random.RandomState(2000 + seed)
    floor = {n: g for n, (g, _) in dr._food.items()
             if w.data.geom_xpos[g][2] <= 0.5}
    a = np.zeros(8)
    fed, dmins, firsts, deaths = 0, [], [], 0
    t0 = time.perf_counter()
    for ep in range(n_ep):
        w.respawn()
        w._sm_nose.reset()
        heading = rng.uniform(-math.pi, math.pi)
        fwd = 1.0
        m0 = sum(dr.ate_total.values())
        dmin, first = 1e9, None
        for k in range(E):
            o = w.observe()["smell"]
            cl, cr, dd = float(o[0]), float(o[4]), float(o[8])
            p = np.array(w.data.xpos[w.rover_bid][:2])
            ready = [np.array(w.data.geom_xpos[g][:2]) for n, g in
                     floor.items() if dr.t >= dr._respawn_at[n]]
            if ready:
                dmin = min(dmin, min(float(np.linalg.norm(q - p))
                                     for q in ready))
            if arm == "goto_true":
                if ready:
                    q = min(ready, key=lambda q: np.linalg.norm(q - p))
                    v = q - p
                    a[6:8] = v / max(1e-6, float(np.linalg.norm(v)))
                else:
                    a[6:8] = 0.0
            elif arm in ("nose_yaw", "nose_blind"):
                mat = np.array(w.data.xmat[w.rover_bid]).reshape(3, 3)
                yaw = (math.atan2(mat[1, 0], mat[0, 0])
                       if arm == "nose_yaw" else 0.0)
                if dd < 0.0:
                    fwd = -fwd
                lat = 1.0 if cl > cr else -1.0
                fx, fy = fwd, lat           # body frame: +x fwd, +y left
                c, s_ = math.cos(yaw), math.sin(yaw)
                v = np.array([c * fx - s_ * fy, s_ * fx + c * fy])
                a[6:8] = v / np.linalg.norm(v)
            elif arm == "klinotaxis":
                if dd <= 0.0:
                    heading = rng.uniform(-math.pi, math.pi)
                a[6:8] = [math.cos(heading), math.sin(heading)]
            else:
                if k % 10 == 0:
                    heading = rng.uniform(-math.pi, math.pi)
                a[6:8] = [math.cos(heading), math.sin(heading)]
            w.decide(a)
            if first is None and sum(dr.ate_total.values()) > m0:
                first = k
            if w.died_this_decision:
                deaths += 1
                break
        fed += int(first is not None)
        dmins.append(dmin if dmin < 1e9 else float("nan"))
        if first is not None:
            firsts.append(first)
    rec = {"arm": arm, "seed": seed, "episodes": n_ep, "E": E,
           "fed_frac": round(fed / n_ep, 3), "fed": fed,
           "dmin_median": round(float(np.nanmedian(dmins)), 2),
           "first_meal_median": (int(np.median(firsts)) if firsts else None),
           "deaths": deaths,
           "wall_s": round(time.perf_counter() - t0, 1)}
    return rec


if __name__ == "__main__" and sys.argv[1:2] == ["episodes"]:
    import torch
    torch.set_num_threads(1)
    n_ep, E = int(sys.argv[2]), int(sys.argv[3])
    for seed in [int(x) for x in sys.argv[4:]]:
        for arm in ARMS2:
            rec = run_episodes(arm, seed, n_ep, E)
            print(json.dumps(rec), flush=True)
            with open(OUT2, "a") as f:
                f.write(json.dumps(rec) + "\n")
    sys.exit(0)

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    n = int(sys.argv[1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for seed in [int(s) for s in sys.argv[2:]]:
        for arm in ARMS:
            rec = run_arm(arm, seed, n)
            print(json.dumps(rec), flush=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
