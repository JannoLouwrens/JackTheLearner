"""J12 (0), ruling 38, proof: jack_env's `eye_frame` option ("world" default,
"body" = ray azimuths + the body's yaw), for BOTH eyes (rays16, row128).

    python -m experiments.tests.test_jack_env_eye_frame [N=200]

(a) IDENTITY (default off): JackEnv(eye_frame="world") == JackEnv() byte for
    byte (obs, reward, terminated, truncated), both eyes, seeds 153-155 and
    J8's pilot world (seed 156), random actions, life ends crossed. Control:
    another action seed differs.
(b) AT YAW 0 body == world: every birth is laid upright at quaternion
    (1,0,0,0) (`W0._place`), so the look at birth must be bit-equal in the
    two frames. And the eye moves no body: same seed, same actions -> touch,
    audio, proprio, needs equal on every decision of the body-frame run.
    Control that must fail: once the body has turned (|yaw| > 0.05 rad) the
    two frames' looks differ.
(c) PLANTED, the look turns with him: stop a living body mid-life, read the
    body-frame look, turn the body a quarter turn about the vertical in
    place (root quaternion premultiplied by rot_z(90 deg), mj_forward), read
    again. Ray j now points where ray j + n/4 pointed, so the new look must
    be the old look rolled by a quarter (distances to 1e-5 of the 8 m range:
    the two looks rotate the base rays by different yaws, so directions agree
    to ~1e-16, not bit for bit; material equal off the noise panel). Control
    that must fail: the WORLD-frame look, same turn, does not roll (it equals
    its un-turned self and differs from its own quarter roll).
(d) HOW MUCH HE TURNS (a measurement, no bar): |yaw| over random lives at the
    entry rung — if the body never turned, (c) would be a proof about a
    situation that does not occur.

(e) MOTOR IDENTITY: JackEnv(motor_frame="world") == JackEnv() byte for byte
    (seeds 153, 156; control differs). With motor_frame="body" the first
    decision of a life (yaw exactly 0, so the push is the world push bit for
    bit) leaves the WORLD bit-equal (qpos, qvel) and every sense but the two
    root velocities equal; those two are the world pair turned by -yaw.
(f) PLANTED, the push turns with him: a body on open floor, turned to face
    +y (quarter turn), given "forward" (action[6:8] = (1, 0)) for ONE
    decision (he yaws while pushed, so a longer push curves — by design the
    push follows him): with motor_frame="body" it must travel along WORLD +y
    (dy > 3 |dx|) and proprio must read it as forward (v_fwd > 3 |v_left|);
    control that must fail: motor_frame="world" travels along world +x.

RESULT (2026-11-18, box, nice 19, `200`, 5.2 CPU-min): PASS.
 (a) 8/8 identical, controls differ.  (b) births bit-equal 4/4 runs, other
 senses equal, turned looks differ 194/194, 198/198 (both eyes).
 (c) body frame: rolled by n/4 on 20/20 stops, both eyes, distance error 0;
 world-frame control: rolled 0/20, unchanged by the turn 20/20.
 (d) 800 random decisions at the entry rung: |yaw| median 81.1 deg, p90
 144.9; beyond one 16-ray step (11.25 deg) on .897 of decisions, beyond
 90 deg on .434 — the compass eye was telling him his heading nearly always.
 (e) 2/2 identical, controls differ; first body-motor decision: world
 bit-equal, other senses equal, velocity = the world pair turned by -yaw.
 (f) facing +y, "forward" once: body motor moves dx +0.00 dy +0.37 m and
 reads v = (+2.98 fwd, +0.13 left); world-motor control moves dx +0.38
 dy +0.01.
 First (c) failed 9/10: after mj_step xpos/xmat are one sub-step behind qpos,
 so the turn's mj_forward also moved the pose — apparatus, fixed by levelling
 before the first look. First (f) pushed 5 decisions and curved (he yaws
 while pushed, the push follows him) — the claim is about one decision.
"""
import math
import sys

import numpy as np

from experiments.fast_sim import body_yaw, to_world
from experiments.world_card import I_VX, I_VY, NEUTRAL
from experiments.jack_env import JackEnv
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PARTS = sys.argv[2] if len(sys.argv) > 2 else "abcdef"   # a subset = a debug run
BASE = {"j0": 1.0, "alpha": 1.0, "e0": 0.05, "max_episode_decisions": 40}
PILOT = {"j0": 1.0, "alpha": 1.0, "max_episode_decisions": 40,
         "difficulty": 0.75, "random_layout": True, "random_needs": 1.0,
         "terminal_obs": "dying"}
CASES = [(153, BASE), (154, BASE), (155, BASE), (156, PILOT)]
ENTRY = {"difficulty": 0.75, "random_layout": True}   # shipped calibration
EYES = (("rays16", 16), ("row128", 128))


def roll(env, seed, act_seed, n):
    rng = np.random.RandomState(act_seed)
    o, _ = env.reset(seed=seed)
    out = [o.copy()]
    for _ in range(n):
        o, r, te, tr, _ = env.step(random_action(rng))
        out.append((o.copy(), r, te, tr))
        if te or tr:
            o, _ = env.reset()
            out.append(o.copy())
    return out


def same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                return False
        elif not (np.array_equal(x[0], y[0]) and x[1:] == y[1:]):
            return False
    return True


def look(env, n):
    """One look through the env's own eye, WITHOUT moving the world's noise
    stream (so a second look sees the same panel draw order start)."""
    st = env.w._rng.get_state()
    v = env.w.observe()["vision"].copy()
    env.w._rng.set_state(st)
    assert v.shape[0] == 2 * n
    return v[0::2], v[1::2]


def hit_gids(env, n):
    """Geom hit by each ray of the last look (row: kept; rays16: recast)."""
    w = env.w
    if "_row_gid" in w.__dict__:
        return w._row_gid.copy()
    import mujoco
    pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
    pos[2] += 0.05
    g = np.zeros(1, dtype=np.int32)
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        mujoco.mj_ray(w.model, w.data, pos, w._ray_dirs[k],
                      np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8), 1, -1, g)
        out[k] = g[0]
    return out


def turn_quarter(env):
    w = env.w
    qa = w.ix["root_qposadr"]
    q = np.array(w.data.qpos[qa + 3:qa + 7], dtype=float)
    h = math.sqrt(0.5)                       # rot_z(90 deg) = (h, 0, 0, h)
    a0, a3 = h, h
    w.data.qpos[qa + 3:qa + 7] = (a0 * q[0] - a3 * q[3], a0 * q[1] - a3 * q[2],
                                  a0 * q[2] + a3 * q[1], a0 * q[3] + a3 * q[0])
    w.mujoco.mj_forward(w.model, w.data)


def main():
    ok = True
    # (a)
    for eye, _ in (EYES if 'a' in PARTS else ()):
        for seed, kw in CASES:
            ref = roll(JackEnv(seed=seed, eye=eye, **kw), seed, seed + 1000, N)
            got = roll(JackEnv(seed=seed, eye=eye, eye_frame="world", **kw),
                       seed, seed + 1000, N)
            ctl = roll(JackEnv(seed=seed, eye=eye, **kw), seed, seed + 2000, N)
            ida, ctla = same(ref, got), not same(ref, ctl)
            print(f"(a) {eye} seed {seed}: identical={ida} control differs={ctla}")
            ok &= ida and ctla
    # (b)
    for eye, n in (EYES if 'b' in PARTS else ()):
        for seed in (157, 158):
            ew = JackEnv(seed=seed, eye=eye, **ENTRY)
            eb = JackEnv(seed=seed, eye=eye, eye_frame="body", **ENTRY)
            rw, rb = (np.random.RandomState(seed + 3000) for _ in range(2))
            ow, _ = ew.reset(seed=seed)
            ob, _ = eb.reset(seed=seed)
            birth = True
            births = births_eq = turned = turned_differs = 0
            rest_eq = True
            for _ in range(N):
                y = body_yaw(eb.w)
                vw, vb = ow[:2 * n], ob[:2 * n]
                rest_eq &= np.array_equal(ow[2 * n:], ob[2 * n:])
                if birth:
                    births += 1
                    births_eq += (y == 0.0) and np.array_equal(vw, vb)
                elif abs(y) > 0.05:
                    turned += 1
                    turned_differs += not np.array_equal(vw[0::2], vb[0::2])
                a = random_action(rw)
                random_action(rb)
                ow, _, te, tr, _ = ew.step(a)
                ob, _, te2, tr2, _ = eb.step(a)
                assert (te, tr) == (te2, tr2)
                birth = False
                # a death respawns inside the step: the returned look is a birth
                if eb.w.died_this_decision and eb.terminal_obs == "reborn":
                    birth = True
                if te or tr:
                    ow, _ = ew.reset()
                    ob, _ = eb.reset()
                    birth = eb.w.died_this_decision
            good = (births_eq == births and births > 0 and rest_eq
                    and turned > 0 and turned_differs > 0)
            print(f"(b) {eye} seed {seed}: births bit-equal {births_eq}/{births} "
                  f"| other senses equal={rest_eq} | control: turned decisions "
                  f"whose looks differ {turned_differs}/{turned}")
            ok &= good
    # (c)
    for eye, n in (EYES if 'c' in PARTS else ()):
        q = n // 4
        for frame in ("body", "world"):
            rolls = unrolled = stays = checks = 0
            worst = 0.0
            for seed in (157, 158):
                env = JackEnv(seed=seed, eye=eye, eye_frame=frame, **ENTRY)
                rng = np.random.RandomState(seed + 5000)
                env.reset(seed=seed)
                for t in range(N):
                    _, _, te, tr, _ = env.step(random_action(rng))
                    if te or tr:
                        env.reset()
                    if t % 20 != 19:
                        continue
                    # after mj_step, xpos/xmat are one sub-step behind qpos
                    # (as shipped, for every sense); bring them level so the
                    # turn below is the ONLY thing that changes the pose.
                    env.w.mujoco.mj_forward(env.w.model, env.w.data)
                    d0, t0 = look(env, n)
                    g0 = hit_gids(env, n)
                    turn_quarter(env)
                    d1, t1 = look(env, n)
                    g1 = hit_gids(env, n)
                    checks += 1
                    exp_d, exp_t, exp_g = (np.roll(x, -q) for x in (d0, t0, g0))
                    err = float(np.max(np.abs(d1 - exp_d)))
                    offp = (g1 != env.w.panel_gid) & (exp_g != env.w.panel_gid)
                    is_roll = (err <= 1e-5 and np.array_equal(g1, exp_g)
                               and np.array_equal(t1[offp], exp_t[offp]))
                    rolls += is_roll
                    unrolled += not is_roll
                    stays += float(np.max(np.abs(d1 - d0))) <= 1e-5
                    if is_roll:
                        worst = max(worst, err)
            if frame == "body":
                good = rolls == checks and checks > 0
                print(f"(c) {eye} body frame: quarter turn -> look is the old "
                      f"look rolled by n/4 on {rolls}/{checks} stops (worst "
                      f"distance error {worst:.1e} of range)")
            else:
                good = unrolled == checks and stays == checks and checks > 0
                print(f"(c) {eye} CONTROL world frame: rolled on "
                      f"{checks - unrolled}/{checks} stops, look unchanged by "
                      f"the turn on {stays}/{checks}")
            ok &= good
    # (d)
    ys = []
    for seed in ((157, 158) if 'd' in PARTS else ()):
        env = JackEnv(seed=seed, eye="row128", eye_frame="body", **ENTRY)
        rng = np.random.RandomState(seed + 6000)
        env.reset(seed=seed)
        for _ in range(2 * N):
            _, _, te, tr, _ = env.step(random_action(rng))
            ys.append(abs(body_yaw(env.w)))
            if te or tr:
                env.reset()
    ys = np.degrees(np.array(ys if ys else [0.0]))
    print(f"(d) entry rung, random actions, {len(ys)} decisions: |yaw| median "
          f"{np.median(ys):.1f} deg, p90 {np.percentile(ys, 90):.1f}, max "
          f"{ys.max():.1f}; share beyond 11.25 deg (one of 16 rays) "
          f"{np.mean(ys > 11.25):.3f}, beyond 90 deg {np.mean(ys > 90):.3f}")
    # (e)
    for seed, kw in (((153, BASE), (156, PILOT)) if 'e' in PARTS else ()):
        ref = roll(JackEnv(seed=seed, **kw), seed, seed + 1000, N)
        got = roll(JackEnv(seed=seed, motor_frame="world", **kw), seed, seed + 1000, N)
        ctl = roll(JackEnv(seed=seed, **kw), seed, seed + 2000, N)
        ida, ctla = same(ref, got), not same(ref, ctl)
        ew, eb = JackEnv(seed=seed, **kw), JackEnv(seed=seed, motor_frame="body", **kw)
        ew.reset(seed=seed)
        eb.reset(seed=seed)
        a = random_action(np.random.RandomState(seed + 7000))
        o_w, o_b = ew.step(a)[0], eb.step(a)[0]
        world_eq = (np.array_equal(ew.w.data.qpos, eb.w.data.qpos)
                    and np.array_equal(ew.w.data.qvel, eb.w.data.qvel))
        vel = np.zeros(o_w.shape[0], dtype=bool)
        vel[[I_VX, I_VY]] = True
        rest_eq = np.array_equal(o_w[~vel], o_b[~vel])
        y = body_yaw(eb.w)
        exp = to_world((float(o_w[I_VX]), float(o_w[I_VY])), -y)
        vel_ok = np.allclose(o_b[vel], exp, atol=1e-6) and y != 0.0
        first = world_eq and rest_eq and vel_ok
        print(f"(e) seed {seed}: motor world identical={ida} control "
              f"differs={ctla} | body motor, first decision: world "
              f"bit-equal={world_eq} other senses equal={rest_eq} velocity = "
              f"world pair turned by -yaw={vel_ok}")
        ok &= ida and ctla and first
    # (f)
    for frame in (("body", "world") if 'f' in PARTS else ()):
        along_y = fwd = n_f = 0
        for seed in (157, 158):
            env = JackEnv(seed=seed, motor_frame=frame)
            env.reset(seed=seed)
            for _ in range(3):                    # settle on the floor
                env.step(NEUTRAL.copy())
            turn_quarter(env)
            w = env.w
            p0 = np.array(w.data.xpos[w.rover_bid][:2])
            a = NEUTRAL.copy()
            a[6:8] = (1.0, 0.0)
            o = env.step(a)[0]
            dx, dy = np.array(w.data.xpos[w.rover_bid][:2]) - p0
            vf, vl = float(o[I_VX]), float(o[I_VY])
            n_f += 1
            along_y += dy > 3 * abs(dx)
            fwd += vf > 3 * abs(vl)
            print(f"(f) {frame} motor seed {seed}: facing +y, 'forward' x1 -> "
                  f"moved dx {dx:+.2f} dy {dy:+.2f} m; proprio v = "
                  f"({vf:+.2f}, {vl:+.2f})")
        if frame == "body":
            ok &= along_y == n_f and fwd == n_f
        else:
            ok &= along_y == 0
    if PARTS != "abcdef":
        print(f"PARTIAL ({PARTS}) — not a proof:", "ok" if ok else "FAIL")
        return 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
