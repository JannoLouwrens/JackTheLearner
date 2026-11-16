"""J6(4) measurement: `JackEnv(integrator="implicitfast")` — a WORLD CHANGE.

The world ships MuJoCo's RK4 (integrator 1); the profile (sim_profile.py,
11-16 14:15) put mj_step x40 under RK4 at 54-58% of a decision and
implicitfast physics 3.9x cheaper. This is not an identity: the order asks
for the measured difference, not a pass.

(1) Default untouched: `integrator="shipped"` vs no argument, same seed and
    action stream -> every observation/reward/flag exactly equal (must PASS).
(2) The difference, per seed, shipped vs implicitfast on the same actions
    (life ends reset as a learner would): decision of first divergence,
    max |obs diff| over the run, life ends (count, how many on the same
    decision), summed reward, MuJoCo warnings and any non-finite observation
    (the stability check: implicitfast at the shipped timestep must not blow
    up).
(3) Speed: process CPU-seconds per decision, shipped / implicitfast /
    implicitfast + lazy_contact_audio, same seed and actions, back to back.

argv: [N steps] [j0,alpha]. Hand calibration (cal-independent, as J6(1)/(2a)).
Exit 0 iff (1) passes and (2) shows no non-finite observation.

Box results 11-16 16:40 (300 steps x 3 seeds, e0 0.05, cap 40, hand cal
j0=alpha=1, load ~0.5): (1) default untouched 3/3 PASS. (2) trajectories
part at the FIRST decision on every seed (contact-event features, e.g. obs
72, swing floor<->value, max|d| 4.8-8.4); no MuJoCo warning, all finite.
Life ends RK4 vs IF: s153 66 vs 71 (42 on the same decision), s154 34 vs 34
(15), s155 44 vs 78 (27). Deaths are almost all INTEGRITY (injury), not
energy: s155 mean life 6.32 -> 3.44 decisions, s153 4.36 -> 4.07 — the
integrator changes how hard contacts hurt (cf. integrity .879 -> .972 in one
life, 11-15). Summed reward s155 -4.25 vs -7.65.
(3) speed: shipped 41.0 dec/CPU-s, implicitfast 70.1 (1.71x), implicitfast
+ lazy_contact_audio 108.5 (2.65x). (30-step smoke: 37.7 / 59.8 / 85.2.)
Re-certification if the WORLD default flipped (playground.py:527
integrator="RK4" -> "implicitfast", `run stale` on a scratch worktree minus
today's baseline): 22 new stale certificates — PASS: BA.01 HR.7 LC.02 LF.02
PG.1 PG.2 PG.3 PG.5 PG.8 PS.01 PS.03 SM.01 SO.02 T2.08 VO.01 W0.PL; FAIL:
HR.5 PS.02 PS.05 T0.32 W.2; VOID: PS.09. The jack_env option (default OFF)
stales nothing (`run stale`: 0 jack_env lines).
"""
import sys
import time

import numpy as np

from experiments.jack_env import JackEnv
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEEDS = (153, 154, 155)


def actions(rng_seed, n):
    rng = np.random.RandomState(rng_seed)
    return [random_action(rng) for _ in range(n)]


def run(seed, acts, kw):
    env = JackEnv(seed=seed, **kw)
    o, _ = env.reset(seed=seed)
    obs, rew, ends, warn = [o], [], [], 0
    t0 = time.process_time()
    for i, a in enumerate(acts):
        o, r, te, tr, _ = env.step(a)
        obs.append(o)
        rew.append(r)
        if te or tr:
            ends.append((i, "death" if te else "cap"))
            warn += int(sum(w.number for w in env.w.data.warning))
            o, _ = env.reset()
            obs.append(o)
    warn += int(sum(w.number for w in env.w.data.warning))
    return dict(obs=obs, rew=rew, ends=ends, warn=warn,
                cpu=time.process_time() - t0)


def main():
    j0, alpha = (float(x) for x in (sys.argv[2] if len(sys.argv) > 2
                                    else "1.0,1.0").split(","))
    kw = {"j0": j0, "alpha": alpha, "e0": 0.05, "max_episode_decisions": 40}
    ok = True
    cpu = {"shipped": 0.0, "implicitfast": 0.0, "implicitfast+lazy": 0.0}
    for s in SEEDS:
        acts = actions(s * 6553 + 11, N)
        ref = run(s, acts, kw)
        dflt = run(s, acts, dict(kw, integrator="shipped"))
        same = (len(ref["obs"]) == len(dflt["obs"])
                and all(np.array_equal(x, y) for x, y in zip(ref["obs"], dflt["obs"]))
                and ref["rew"] == dflt["rew"] and ref["ends"] == dflt["ends"])
        imp = run(s, acts, dict(kw, integrator="implicitfast"))
        both = run(s, acts, dict(kw, integrator="implicitfast",
                                 lazy_contact_audio=True))
        cpu["shipped"] += ref["cpu"]
        cpu["implicitfast"] += imp["cpu"]
        cpu["implicitfast+lazy"] += both["cpu"]
        n = min(len(ref["obs"]), len(imp["obs"]))
        diffs = [float(np.max(np.abs(ref["obs"][i] - imp["obs"][i])))
                 for i in range(n)]
        first = next((i for i, d in enumerate(diffs) if d > 0), None)
        finite = all(np.all(np.isfinite(o)) for o in imp["obs"] + both["obs"])
        same_end = len(set(ref["ends"]) & set(imp["ends"]))
        print(f"seed {s}: default-untouched {'PASS' if same else 'FAIL'}; "
              f"implicitfast first diff at obs #{first}, max|d| "
              f"{max(diffs):.4g}, median|d| {float(np.median(diffs)):.4g}; "
              f"ends RK4 {len(ref['ends'])} "
              f"({sum(e[1] == 'death' for e in ref['ends'])} deaths) vs IF "
              f"{len(imp['ends'])} ({sum(e[1] == 'death' for e in imp['ends'])}"
              f" deaths), {same_end} identical; reward RK4 "
              f"{sum(ref['rew']):.3f} vs IF {sum(imp['rew']):.3f}; warnings "
              f"RK4 {ref['warn']} IF {imp['warn']}; IF finite {finite}",
              flush=True)
        ok &= same and finite
    d = len(SEEDS) * N
    base = d / cpu["shipped"]
    print("speed (dec/CPU-s): " + ", ".join(
        f"{k} {d / v:.1f} ({(d / v) / base:.2f}x)" for k, v in cpu.items()))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
