"""J1 step 1 proof: jack_env is W0, unchanged, behind gymnasium.

(a) gymnasium's check_env passes.
(b) fixed-seed random-action runs through the wrapper equal the same runs
    through survival.run_survival(policy="random"): same life ends (decision
    index of every death) and same eat events, 3 seeds, at default e0 and at
    e0=0.05 (so deaths and respawns are inside the window).
Also equal: the summed reward (homeo-dr) to 1e-9.
Control that must fail: the wrapper driven by a DIFFERENT action stream
(rng seed+1) must not match — otherwise the comparison measures nothing.
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, default_cal
from experiments.survival import run_survival
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = (153, 154, 155)


def env_run(seed, e0, cal, n, rng_seed):
    env = JackEnv(seed=seed, e0=e0, **cal)
    env.reset(seed=seed)
    rng = np.random.RandomState(rng_seed)
    ends, eats_at_death, rsum = [], [], 0.0
    for k in range(n):
        _, r, term, _, info = env.step(random_action(rng))  # float64, as the old loop
        rsum += r
        if term:
            ends.append(info["decisions"])
            eats_at_death.append(info["meals"])
            env.reset()
    return ends, eats_at_death, info["ate_total"], rsum


def main():
    from gymnasium.utils.env_checker import check_env
    cal = default_cal()
    check_env(JackEnv(seed=153, **cal), skip_render_check=True)
    print("check_env: PASS")
    ok, ctrl_any = True, False
    for e0 in (1.0, 0.05):
        for s in SEEDS:
            ref = run_survival(s, policy="random", n_decisions=N, e0=e0,
                               train=False, **cal)
            ref_ends = [int(le[0]) for le in ref.get("life_ends", [])]
            ends, ead, ate, rs = env_run(s, e0, cal, N, s * 6553 + 11)
            same = (ends == ref_ends and ate == ref["ate_total"]
                    and ead == list(ref["eats_at_death"])
                    and abs(rs - ref["reward_sum"]) < 1e-9)
            c_ends, c_ead, c_ate, c_rs = env_run(s, e0, cal, N, s * 6553 + 12)
            ctrl_same = (c_ends == ref_ends and c_ate == ref["ate_total"]
                         and abs(c_rs - ref["reward_sum"]) < 1e-9)
            print(f"e0={e0} seed={s}: env deaths@{ends} ref@{ref_ends} "
                  f"ate env={ate} ref={ref['ate_total']} -> "
                  f"reward_sum env={rs:.9f} ref={ref['reward_sum']:.9f} "
                  f"{'MATCH' if same else 'DIFFER'}; control "
                  f"{'MATCHES (BAD)' if ctrl_same else 'differs'}")
            ok &= same
            ctrl_any |= ctrl_same
    print("PROOF:", "PASS" if ok and not ctrl_any else "FAIL")


if __name__ == "__main__":
    main()
