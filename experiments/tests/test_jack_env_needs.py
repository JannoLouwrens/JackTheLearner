"""J2 item 2 (part) proof: jack_env's random_needs — random starting needs.

(a) OFF IS BYTE-IDENTICAL: random_needs=None equals JackEnv() decision for
    decision (observation bytes, reward, terminated, truncated, meals) — 3 seeds.
(b) ON DOES WHAT IT SAYS: at s = 1/6 (trp_env) and s = 0.5, at birth and at
    every respawn the needs are e, i in [1-s, 1], w = 0, and the starts of
    one run are all different (a new draw each life). APPARATUS: a life
    lasts ~1,500 random decisions, so the test starves him (sets e = 1e-4)
    at decisions STARVE in every run alike, to put respawns in the window.
    Guard: >= 3 respawns per run, else respawn was not measured.
(c) WORLD UNTOUCHED: the needs come from the env's own stream — right after
    construction every numpy RandomState W0 holds is in the same state as in
    the default env (the draw consumed none of W0's streams).
(d) CONTROL THAT MUST FAIL: the s=0.5 run's start needs are NOT all full,
    and its observation stream differs from the default (else the option
    changes nothing).
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, default_cal, NEEDS_S_TRP
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
STARVE = (150, 300, 450)
SEEDS = (153, 154, 155)


def run(seed, cal, n, rng_seed, s):
    env = JackEnv(seed=seed, **cal, random_needs=s)
    env.reset(seed=seed)
    st = env.w.drives.state
    starts = [(st.e, st.i, st.w)]
    rng = np.random.RandomState(rng_seed)
    out = []
    for k in range(n):
        if k in STARVE:
            env.w.drives.state.e = 1e-4
        o, r, term, trunc, info = env.step(random_action(rng))
        out.append((o.tobytes(), r, term, trunc, info["meals"]))
        if term:
            st = env.w.drives.state
            starts.append((st.e, st.i, st.w))
        if term or trunc:
            env.reset()
    return out, starts


def rng_states(seed, cal, s):
    w = JackEnv(seed=seed, **cal, random_needs=s).w
    return {k: (v.get_state()[1].tobytes(), v.get_state()[2])
            for k, v in vars(w).items() if isinstance(v, np.random.RandomState)}


def main():
    cal = default_cal()
    ok_a = ok_b = ok_c = guard = True
    not_full = differs = False
    for seed in SEEDS:
        rs = seed * 7919 + 3
        base, st0 = run(seed, cal, N, rs, None)
        r0 = rng_states(seed, cal, None)
        a = base == run(seed, cal, N, rs, None)[0]
        ok_a &= a
        for s in (NEEDS_S_TRP, 0.5):
            out, st = run(seed, cal, N, rs, s)
            b = all(1 - s - 1e-12 <= e <= 1 and 1 - s - 1e-12 <= i <= 1 and w == 0
                    for e, i, w in st) and len(set(st)) == len(st)
            c = rng_states(seed, cal, s) == r0 and len(r0) > 0
            guard &= len(st) >= 4
            ok_b &= b
            ok_c &= c
            if s == 0.5:
                not_full |= any(e < 1 or i < 1 for e, i, _ in st)
                differs |= out != base
            print(f"seed {seed} s={s:.3f}: lives started {len(st)} "
                  f"(default {len(st0)}), e range "
                  f"{min(x[0] for x in st):.3f}-{max(x[0] for x in st):.3f}, "
                  f"i range {min(x[1] for x in st):.3f}-{max(x[1] for x in st):.3f}, "
                  f"W0 RNG states ({len(r0)}) match default: {c}; b={b}")
        print(f"seed {seed}: (a) None == default: {a}")
    d = not_full and differs
    ok = ok_a and ok_b and ok_c and guard and d
    print(f"(a) {ok_a} (b) {ok_b} (c) {ok_c} guard>=3 respawns {guard} "
          f"(d) control differs {d}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
