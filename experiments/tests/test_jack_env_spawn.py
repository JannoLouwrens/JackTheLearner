"""J2 item 1 (spawn distance) proof: jack_env's difficulty dial places birth
and every respawn near floor food.

Independent measurement: after birth and after every death, the rover body's
x, y (read from MuJoCo, not from the sampler) against the floor foods' bodies.
(a) d=0: every spawn is within max(0.5 m, nearest legal cell) of the nearest
    floor food (+ 1e-6).
(b) d=0.25: every spawn within 0.5 + 17*0.25 = 4.75 m (or nearest legal cell).
(c) Control that must fail: today's world (difficulty=None) breaks (a)'s
    bound on most spawns (else the dial changes nothing).
Guard: >= 1 RESPAWN (not birth) per d, pooled over seeds.
(d) d=1.0 == None byte for byte is test_jack_env_dial's (a); re-run it.
e0=0.05 so lives are short (deaths ~every 100 decisions, as test_jack_env).
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, default_cal, FLOOR_FOODS, dial_spawn_radius
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEEDS = (153, 154, 155)


def spawn_dists(seed, cal, d):
    env = JackEnv(seed=seed, **cal, e0=0.05, difficulty=d)
    env.reset(seed=seed)
    w = env.w
    legal = w.legal_spawns()
    rng = np.random.RandomState(seed * 6553 + 11)

    def measure():
        xy = w.data.xpos[w.rover_bid][:2]
        f = np.array([w.data.xpos[w.model.body(n).id][:2] for n in FLOOR_FOODS])
        nearest_legal = np.min(np.linalg.norm(legal[:, None] - f[None], axis=2))
        return float(np.min(np.linalg.norm(f - xy, axis=1))), float(nearest_legal)

    out = [measure()]
    for _ in range(N):
        o, r, term, trunc, info = env.step(random_action(rng))
        if term:
            out.append(measure())
            env.reset()
    return out


def main():
    cal = default_cal()
    ok = True
    viol_ctrl = n_ctrl = 0
    respawns = {0.0: 0, 0.25: 0}
    for s in SEEDS:
        for d in (0.0, 0.25):
            ds = spawn_dists(s, cal, d)
            R = dial_spawn_radius(d)
            good = all(x <= max(R, nl) + 1e-6 for x, nl in ds)
            print(f"seed {s} d={d}: {len(ds)} spawns, dist to food "
                  f"{['%.2f' % x for x, _ in ds]} bound {R:.2f} "
                  f"{'PASS' if good else 'FAIL'}")
            ok &= good
            respawns[d] += len(ds) - 1
        ds = spawn_dists(s, cal, None)
        v = [x > max(0.5, nl) + 1e-6 for x, nl in ds]
        viol_ctrl += sum(v); n_ctrl += len(v)
        print(f"seed {s} today's world: dist {['%.2f' % x for x, _ in ds]}")
    ctrl = viol_ctrl > n_ctrl / 2
    # guard, pooled over seeds: at d=0 a full-bar meal near food can keep
    # him alive all N decisions, so a seed may show its birth only.
    print("respawns seen per d (guard: >= 1 each):", respawns)
    ok &= all(v >= 1 for v in respawns.values())
    print(f"(c) control: today's world breaks the d=0 bound on "
          f"{viol_ctrl}/{n_ctrl} spawns -> {'fails as it must' if ctrl else 'DID NOT FAIL'}")
    print("PROOF:", "PASS" if ok and ctrl else "FAIL")


if __name__ == "__main__":
    main()
