"""J2 items 3-4 proof: jack_env's reward and time-limit options.

(a) OFF IS BYTE-IDENTICAL: JackEnv with every option at its default, passed
    explicitly, equals JackEnv() decision for decision (observation bytes,
    reward, terminated, truncated) — 3 seeds, e0=0.05 so deaths are inside.
(b) dr100-eat: reward == 100*homeo-dr + eat_pay*(eat events) on every
    decision, and at least one eat event occurs across the seeds (else (b)
    tests nothing; pooled because seed 154 eats nothing in 300 random
    decisions — the formula is still checked on every decision of all 3).
(c) dr100-eat-pos: every alive decision >= 0, every death < 0, and the
    plain dr100-eat run has at least one negative alive decision (else the
    clip changed nothing and (c) tests nothing).
(d) max_episode_decisions=K: truncated exactly at K, 2K, ... between deaths,
    never on a death step; the world is untouched — observations and
    homeo-dr rewards equal the uncapped run's; meals_this_life carries over
    a truncation. Control that must fail: a DIFFERENT action stream does not
    reproduce the uncapped run.
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, default_cal
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = (153, 154, 155)


def run(seed, e0, cal, n, rng_seed, **opt):
    env = JackEnv(seed=seed, e0=e0, **cal, **opt)
    env.reset(seed=seed)
    rng = np.random.RandomState(rng_seed)
    out = []
    for _ in range(n):
        o, r, term, trunc, info = env.step(random_action(rng))
        out.append((o.tobytes(), r, term, trunc, info["meals"],
                    info["meals_this_life"], info["life"]))
        if term or trunc:
            env.reset()
    return out


def main():
    cal = default_cal()
    ok, eats_any = True, False
    for s in SEEDS:
        rs = s * 6553 + 11
        base = run(s, 0.05, cal, N, rs)
        expl = run(s, 0.05, cal, N, rs, reward="homeo-dr", eat_pay=100.0,
                   max_episode_decisions=None)
        a = base == expl
        # (b)/(c) at e0=1.0 (eats happen) and e0=0.05 (deaths happen)
        b = c = True
        eats_seen = neg_alive_seen = False
        for e0 in (1.0, 0.05):
            h = run(s, e0, cal, N, rs)
            d = run(s, e0, cal, N, rs, reward="dr100-eat")
            p = run(s, e0, cal, N, rs, reward="dr100-eat-pos")
            prev = 0
            for x, y, z in zip(h, d, p):
                eats = x[4] - prev
                prev = x[4]
                eats_seen |= eats > 0
                b &= abs(y[1] - (100 * x[1] + 100.0 * eats)) < 1e-9
                b &= x[0] == y[0] == z[0]
                if x[2]:
                    c &= z[1] < 0
                else:
                    c &= z[1] >= 0
                    neg_alive_seen |= y[1] < 0
        eats_any |= eats_seen
        c &= neg_alive_seen
        # (d)
        K = 40
        cap = run(s, 0.05, cal, N, rs, max_episode_decisions=K)
        dd = all(x[0] == y[0] and x[1] == y[1] and x[2] == y[2]
                 for x, y in zip(base, cap))
        since, trunc_ok, carry_ok = 0, True, True
        for i, y in enumerate(cap):
            since += 1
            want = (not y[2]) and since == K
            trunc_ok &= (y[3] == want)
            if y[2] or y[3]:
                since = 0
            if y[3] and i + 1 < len(cap):
                nxt = cap[i + 1]
                # meals_this_life never drops across a truncation
                carry_ok &= nxt[5] >= y[5] and nxt[6] == y[6]
        n_trunc = sum(y[3] for y in cap)
        ctrl = run(s, 0.05, cal, N, rs + 1)
        ctrl_same = all(x[0] == y[0] for x, y in zip(base, ctrl))
        d_ok = dd and trunc_ok and carry_ok and n_trunc > 0 and not ctrl_same
        print(f"seed {s}: (a) off==default {'PASS' if a else 'FAIL'}; "
              f"(b) dr100-eat {'PASS' if b else 'FAIL'} (eats seen {eats_seen}); "
              f"(c) pos {'PASS' if c else 'FAIL'}; (d) cap K={K} "
              f"{'PASS' if d_ok else 'FAIL'} ({n_trunc} truncations, deaths "
              f"{sum(y[2] for y in cap)}, world same {dd}, control "
              f"{'MATCHES (BAD)' if ctrl_same else 'differs'})")
        ok &= a and b and c and d_ok
    print("eat events seen (pooled guard for (b)):", eats_any)
    print("PROOF:", "PASS" if ok and eats_any else "FAIL")


if __name__ == "__main__":
    main()
