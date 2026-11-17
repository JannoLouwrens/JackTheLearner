"""J11 (1) proof: jack_env's `move_cost_dial` and `spawn_clear` options.

    python -m experiments.tests.test_jack_env_cures [N=200] [LIVES=24]

IDENTITY (off = today's world, byte for byte), seeds 153-155, N random
decisions, observations + rewards + deaths compared:
 (a) JackEnv(move_cost_dial=False, spawn_clear=None) == JackEnv();
     JackEnv(move_cost_dial=True) with difficulty None == JackEnv() (the dial
     charges the full KAPPA at d = 1: nothing is patched);
     the pilot world with both options spelled off == the pilot world.
     Control: another action seed differs.
PLANTED, move_cost_dial (the effect, read from the body's own energy):
 (b) pilot world (d 0.75), dial ON: on every decision without a meal, death
     or clip, energy falls by exactly BASAL_B*dt (movement costs nothing
     extra at d0), while the published mean power is > 0 (he DID work).
     At d 0.875 the fall is BASAL_B*dt + 0.5*KAPPA*work.
     Control that must fail (b)'s equality: the same world, dial OFF — the
     fall exceeds basal by KAPPA*work on the same check.
PLANTED, spawn_clear=R:
 (c) pilot world and the dial at d = 0 (which wants him within 0.5 m of a
     food), LIVES forced rebirths each: every birth and rebirth, measured
     from MuJoCo's body positions (not from the sampler), is > R from every
     floor food. Control that must fail: d = 0 without spawn_clear puts
     every spawn inside R. Also spawn_clear alone on the default world.
 (d) the statue (cause a of J8's VOID): a do-nothing body, dial d = 0 so it
     is born as near food as the rule allows, NOOP decisions per life.
     spawn_clear ON: 0 meals. Control that must fail: OFF, it eats.

RESULT (2026-11-17, box, nice 19, `200 24`, 9.1 CPU-min): PASS.
 (a) 3/3 seeds identical on all three, control differs.
 (b) ON at d 0.75: extra drain max 3.7e-17 on 198-199 clean decisions a seed
     while the work done would have cost 5.2-5.9e-4 a decision at full
     KAPPA; d 0.875 = half KAPPA exactly; control OFF = full KAPPA (extra
     drain mean 5.2-6.3e-4 a decision): fails the equality as it must.
 (c) R = 1.55 m, 25 spawns x 3 worlds x 3 seeds: nearest food 1.58-8.07 m
     (pilot), 1.58 m (d = 0), 1.77-7.34 m (default world), 0 failures;
     control: 75/75 uncleared d = 0 spawns inside R (0.44 m).
 (d) 72 do-nothing lives x 150 decisions at d = 0: 0 meals with spawn_clear,
     2,545 without (instant regrowth at d = 0: a body lying on a food eats
     every decision).
 Regressions re-run: test_jack_env_spawn 300 PASS; test_jack_env_layout 300
 (a)-(d) True (its >= 3-respawn guard needs the registered 600, not re-paid).
"""
import sys
import numpy as np

from experiments import drives
from experiments.jack_env import (JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R,
                                  default_cal, dial_move_cost)
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LIVES = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NOOP = 150
SEEDS = (153, 154, 155)
PILOT = dict(difficulty=0.75, reward="dr100-eat", random_needs=NEEDS_S_TRP,
             random_layout=True, lazy_contact_audio=True)


def stream(seed, n, act_seed=None, **kw):
    env = JackEnv(seed=seed, **kw)
    o, _ = env.reset(seed=seed)
    rng = np.random.RandomState((seed if act_seed is None else act_seed) * 6553 + 11)
    out = [o.tobytes()]
    for _ in range(n):
        o, r, term, trunc, info = env.step(random_action(rng))
        out.append((o.tobytes(), float(r), bool(term)))
        if term:
            o, _ = env.reset()
            out.append(o.tobytes())
    return out


def energy_check(seed, n, d, dial):
    """Per decision: (fall - basal*dt, KAPPA*work) on clean decisions."""
    kw = dict(PILOT, difficulty=d, move_cost_dial=dial)
    env = JackEnv(seed=seed, **kw)
    env.reset(seed=seed)
    rng = np.random.RandomState(seed * 6553 + 11)
    dl = env.w.drives
    rows = []
    for _ in range(n):
        e0, m0 = dl.state.e, env._meals()
        _, _, term, _, _ = env.step(random_action(rng))
        if term:
            env.reset()
            continue
        e1 = dl.state.e
        if env._meals() != m0 or e1 <= 0.0 or e1 >= 1.0:
            continue
        work = dl.last_power_w * dl.last_dt
        rows.append((e0 - e1 - drives.BASAL_B * dl.last_dt, drives.KAPPA * work))
    return np.array(rows)


def spawn_dists(seed, lives, **kw):
    env = JackEnv(seed=seed, **kw)
    env.reset(seed=seed)
    w = env.w

    def measure():
        xy = w.data.xpos[w.rover_bid][:2]
        f = np.array([w.data.xpos[w.model.body(n).id][:2]
                      for n in env.floor_foods if n in w.drives._food])
        return float(np.min(np.linalg.norm(f - xy, axis=1)))

    out = [measure()]
    for _ in range(lives):
        w.respawn()
        out.append(measure())
    return out, env.spawn_clear_failed


def statue(seed, lives, clear):
    kw = dict(PILOT, difficulty=0.0, spawn_clear=clear)
    env = JackEnv(seed=seed, **kw)
    env.reset(seed=seed)
    a = np.zeros(8)
    meals0 = env._meals()
    for _ in range(lives):
        env.w.respawn()
        env._draw_needs()
        env._pending_obs = None
        env.reset()
        for _ in range(NOOP):
            _, _, term, _, _ = env.step(a)
            if term:
                env.reset()
    return env._meals() - meals0


def main():
    cal = default_cal()
    ok = True

    # (a) identity
    for s in SEEDS:
        base = stream(s, N, **cal)
        same = [stream(s, N, **cal, move_cost_dial=False, spawn_clear=None) == base,
                stream(s, N, **cal, move_cost_dial=True) == base]
        ctrl = stream(s, N, act_seed=s + 1, **cal) != base
        pb = stream(s, N, **PILOT)
        same.append(stream(s, N, **PILOT, move_cost_dial=False, spawn_clear=None) == pb)
        good = all(same) and ctrl
        ok &= good
        print(f"(a) seed {s}: defaults/dial@d1/pilot-off identical {same}, "
              f"control differs {ctrl} -> {'PASS' if good else 'FAIL'}")

    # (b) move_cost_dial
    assert dial_move_cost(0.75) == 0.0 and dial_move_cost(1.0) == 1.0
    assert abs(dial_move_cost(0.875) - 0.5) < 1e-12 and dial_move_cost(0.3) == 0.0
    for s in SEEDS:
        on = energy_check(s, N, 0.75, True)
        half = energy_check(s, N, 0.875, True)
        off = energy_check(s, N, 0.75, False)
        g_on = len(on) > N // 2 and np.max(np.abs(on[:, 0])) < 1e-12 and on[:, 1].mean() > 0
        g_half = np.max(np.abs(half[:, 0] - 0.5 * half[:, 1])) < 1e-12
        g_off_fails = np.max(np.abs(off[:, 0])) > 1e-6          # control must fail
        g_off_full = np.max(np.abs(off[:, 0] - off[:, 1])) < 1e-12
        good = g_on and g_half and g_off_fails and g_off_full
        ok &= bool(good)
        print(f"(b) seed {s}: ON d0.75 {len(on)} decisions, extra drain max "
              f"{np.max(np.abs(on[:, 0])):.1e} (work priced at full KAPPA would be "
              f"{on[:, 1].mean():.2e}/decision); d0.875 half-KAPPA {bool(g_half)}; "
              f"control OFF extra drain mean {off[:, 0].mean():.2e} = full KAPPA "
              f"{bool(g_off_full)} -> {'PASS' if good else 'FAIL'}")

    # (c) spawn_clear distances
    R = SPAWN_CLEAR_R
    n_ctrl_in = n_ctrl = 0
    for s in SEEDS:
        res = {}
        for tag, kw in (("pilot", dict(PILOT, spawn_clear=R)),
                        ("d0", dict(PILOT, difficulty=0.0, spawn_clear=R)),
                        ("default", dict(cal, spawn_clear=R))):
            ds, failed = spawn_dists(s, LIVES, **kw)
            res[tag] = (min(ds), max(ds), failed)
            ok &= min(ds) > R and failed == 0
        ctrl, _ = spawn_dists(s, LIVES, **dict(PILOT, difficulty=0.0))
        n_ctrl_in += sum(x <= R for x in ctrl)
        n_ctrl += len(ctrl)
        print(f"(c) seed {s}: R={R} m, {LIVES + 1} spawns each, nearest food "
              + ", ".join(f"{k} {v[0]:.2f}-{v[1]:.2f} m (failed {v[2]})"
                          for k, v in res.items())
              + f"; control d0 no clear {min(ctrl):.2f}-{max(ctrl):.2f} m")
    g = n_ctrl_in >= n_ctrl // 2
    ok &= g
    print(f"(c) control: {n_ctrl_in}/{n_ctrl} uncleared d=0 spawns inside R "
          f"-> {'fails as it must' if g else 'CONTROL DID NOT FAIL'}")

    # (d) the statue
    on = sum(statue(s, LIVES, R) for s in SEEDS)
    off = sum(statue(s, LIVES, None) for s in SEEDS)
    g = on == 0 and off > 0
    ok &= g
    print(f"(d) do-nothing body, d=0, {3 * LIVES} lives x {NOOP} decisions: "
          f"meals with spawn_clear {on}, without {off} "
          f"-> {'PASS' if g else 'FAIL'}")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
