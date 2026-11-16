"""J9 proofs: `terminal_obs="dying"` — the transition that ends a life carries
the body that died, and the world's random stream is not touched.

(1) J1 identity, option OFF and ON: fixed-seed random-action runs through the
    wrapper equal survival.run_survival(policy="random") — same death
    decisions, same eat events, reward sum to 1e-9 — 3 seeds x e0 in
    {1.0, 0.05}. Control: a different action stream must NOT match.
(2) Stream untouched: the full observation/reward/termination stream with the
    option ON equals the option-OFF stream byte for byte on every
    non-terminal step and every reset(); at each death the ON run's reset()
    observation equals the OFF run's terminal one (the reborn body moved, not
    lost).
(3) Planted: at every death the ON terminal observation equals the dying
    body's deterministic senses (proprio, needs) read by an independent,
    draw-free hook on `W0.respawn` at the instant before the body moves;
    the dead need is at 0 (energy deaths: e <= 0) while the reborn body's is
    e0. Control that must FAIL: the same check against the OFF terminal
    observation (today's: the reborn body).
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, SENSES, default_cal
from experiments.survival import run_survival
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = (153, 154, 155)


def offsets(env):
    from experiments.cores import MODALITIES
    out, k = {}, 0
    for s in SENSES:
        out[s] = slice(k, k + MODALITIES[s])
        k += MODALITIES[s]
    assert k == env.observation_space.shape[0]
    return out


def env_run(seed, e0, cal, n, rng_seed, terminal_obs, hook=False):
    env = JackEnv(seed=seed, e0=e0, terminal_obs=terminal_obs, **cal)
    o0, _ = env.reset(seed=seed)
    seen = []
    if hook:
        w = env.w
        orig = w.respawn

        def spy(*a, **kw):          # the instant before the body is moved;
            # proprio/needs only: _vision may draw from the world's stream
            seen.append({"proprio": w._proprio().copy(),
                         "needs": w._needs().copy(),
                         "cause": w.last_death_cause})
            return orig(*a, **kw)
        w.respawn = spy
    rng = np.random.RandomState(rng_seed)
    stream = [("reset", o0)]
    ends, ead, rsum = [], [], 0.0
    for _ in range(n):
        o, r, term, trunc, info = env.step(random_action(rng))
        rsum += r
        stream.append(("step", o, r, term, trunc))
        if term:
            ends.append(info["decisions"])
            ead.append(info["meals"])
            ro, _ = env.reset()
            stream.append(("reset", ro))
    return dict(ends=ends, ead=ead, ate=info["ate_total"], rsum=rsum,
                stream=stream, seen=seen, env=env)


def identity(cal):
    ok, ctrl_any = True, False
    for mode in ("reborn", "dying"):
        for e0 in (1.0, 0.05):
            for s in SEEDS:
                ref = run_survival(s, policy="random", n_decisions=N, e0=e0,
                                   train=False, **cal)
                ref_ends = [int(le[0]) for le in ref.get("life_ends", [])]
                a = env_run(s, e0, cal, N, s * 6553 + 11, mode)
                same = (a["ends"] == ref_ends and a["ate"] == ref["ate_total"]
                        and a["ead"] == list(ref["eats_at_death"])
                        and abs(a["rsum"] - ref["reward_sum"]) < 1e-9)
                c = env_run(s, e0, cal, N, s * 6553 + 12, mode)
                csame = (c["ends"] == ref_ends and c["ate"] == ref["ate_total"]
                         and abs(c["rsum"] - ref["reward_sum"]) < 1e-9)
                print(f"(1) {mode:6s} e0={e0} seed={s}: deaths@{a['ends']} "
                      f"ref@{ref_ends} {'MATCH' if same else 'DIFFER'}; "
                      f"control {'MATCHES (BAD)' if csame else 'differs'}")
                ok &= same
                ctrl_any |= csame
    return ok and not ctrl_any


def stream_and_planted(cal):
    ok2, ok3, ctrl_any, n_deaths = True, True, False, 0
    for s in SEEDS:
        off = env_run(s, 0.05, cal, N, s * 6553 + 11, "reborn", hook=True)
        on = env_run(s, 0.05, cal, N, s * 6553 + 11, "dying", hook=True)
        sl = offsets(on["env"])
        A, B = off["stream"], on["stream"]
        same_len = len(A) == len(B)
        diffs_nonterm, moved = 0, True
        for i, (x, y) in enumerate(zip(A, B)):
            if x[0] == "step" and x[3]:          # a death step
                if not (x[2] == y[2] and x[3] == y[3] and x[4] == y[4]):
                    diffs_nonterm += 1
                # the OFF terminal (reborn) is the ON run's next reset()
                moved &= np.array_equal(x[1], B[i + 1][1])
                continue
            if x[0] != y[0] or not np.array_equal(x[1], y[1]) or \
                    (x[0] == "step" and x[2:] != y[2:]):
                diffs_nonterm += 1
        s2 = same_len and diffs_nonterm == 0 and moved and len(on["ends"]) > 0
        # (3) planted: every death's ON terminal obs vs the spy's dying body
        term_on = [y[1] for y in B if y[0] == "step" and y[3]]
        term_off = [x[1] for x in A if x[0] == "step" and x[3]]
        spy = on["seen"]
        e = sl["needs"].start           # needs = [e, i, w, d, edot, idot]

        def matches(obs_list):
            if len(obs_list) != len(spy) or not obs_list:
                return False
            good = True
            for o, sp in zip(obs_list, spy):
                good &= np.array_equal(o[sl["proprio"]], sp["proprio"])
                good &= np.array_equal(o[sl["needs"]], sp["needs"])
                if sp["cause"] == "energy":
                    good &= bool(o[e] <= 0.0)
                elif sp["cause"] == "integrity":
                    good &= bool(o[e + 1] <= 0.0)
            return bool(good)
        s3 = matches(term_on)
        reborn_e = all(abs(float(B[i + 1][1][e]) - 0.05) < 1e-6
                       for i, y in enumerate(B) if y[0] == "step" and y[3])
        s3 &= reborn_e
        ctrl = matches(term_off)
        n_deaths += len(term_on)
        print(f"(2) seed={s}: {len(on['ends'])} deaths, non-terminal diffs "
              f"{diffs_nonterm}, reborn moved to reset() {moved} -> "
              f"{'PASS' if s2 else 'FAIL'}")
        print(f"(3) seed={s}: dying e at death "
              f"{[round(float(o[e]), 4) for o in term_on]}, reborn e "
              f"{[round(float(o[e]), 4) for o in term_off]}, causes "
              f"{[sp['cause'] for sp in spy]} -> {'PASS' if s3 else 'FAIL'}; "
              f"control (OFF terminal) {'MATCHES (BAD)' if ctrl else 'fails'}")
        ok2 &= s2
        ok3 &= s3
        ctrl_any |= ctrl
    return ok2, ok3 and not ctrl_any and n_deaths > 0


def main():
    # argv[2] = "j0,alpha": the calibration given by hand, for the window in
    # which the w0.py edit has staled LC.02 and default_cal() refuses (J9
    # step 5; values printed by default_cal() on the pre-edit tree).
    if len(sys.argv) > 2:
        j0, alpha = (float(v) for v in sys.argv[2].split(","))
        cal = {"j0": j0, "alpha": alpha}
    else:
        cal = default_cal()
    print("cal:", cal)
    p1 = identity(cal)
    p2, p3 = stream_and_planted(cal)
    print(f"PROOF (1) identity off+on: {'PASS' if p1 else 'FAIL'}; "
          f"(2) stream untouched: {'PASS' if p2 else 'FAIL'}; "
          f"(3) planted dying body: {'PASS' if p3 else 'FAIL'}")
    print("PROOF:", "PASS" if p1 and p2 and p3 else "FAIL")


if __name__ == "__main__":
    main()
