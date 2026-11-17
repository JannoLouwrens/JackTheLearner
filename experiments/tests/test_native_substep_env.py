"""J6 (2b) proof: jack_env's `native_substep` option (default OFF).

    python -m experiments.tests.test_native_substep_env [N=300]

The option runs the 40 sub-steps' Python pieces (drive gate, drives.substep,
water.apply, contact-audio onset filter) as C twins
(experiments/native_substep.py). Each piece was proven alone (18,000-54,000
sub-steps); this test proves the WHOLE ENV, including what the shadow runs
never exercised: deaths, rebirths, the learner's reset() after a death, and
a reset(seed=...) that rebuilds the world (the option must survive it).

Seeds 153-155, N random decisions (actions from rng(seed)), the same action
stream fed to JackEnv(lazy_contact_audio=True) and the same + native_substep.
Compared EXACTLY every decision: observation, reward, terminated, truncated,
info, qpos, qvel. reset() is called after every terminated/truncated step,
as a learner does; the pending observation it returns is compared too.
 (A) DYING world: e0 = 0.05 (born and reborn starving) — >= 1 death a seed
     is REQUIRED, or the case is void.
 (B) J11's pilot world with both cures: d 0.75, dr100-eat, random_needs,
     random_layout, move_cost_dial, spawn_clear = SPAWN_CLEAR_R,
     max_episode_decisions 100 (truncations exercised too).
 (D) case (A) under integrator="implicitfast" (the twins must compose
     with the other speed option; the box speed table's fastest row).
 (C) rebuild: case (A), and at N/2 both envs reset(seed=seed + 1000), which
     builds a fresh W0; the second half must still be native (asserted: the
     rebuilt world's decide is the patched one) and identical.
CONTROL that must fail: case (A) with the C drives fed a sub-step dt one ulp
larger (np.nextafter) — a 1-ulp change inside the C path; must differ on
>= 1 decision a seed (proves the comparison sees ulp-scale drift).
REPORTED, not gated: the twins built with fma_norm=False (the plain
summation order that the piece tests' controls used). Guard: native_substep without lazy_contact_audio raises.
Speed: dec/CPU-s, shipped(lazy) vs native, every case, printed.

RESULT (2026-11-17 ~12:40 UTC, box, nice 19, N=300, 181 CPU-s): PASS.
 guard raises. 0 differing decisions in every case and seed:
 (A) 2/2/2 deaths, 2 rebirths each; 42.5/44.9/66.7 -> 57.3/57.9/100.0 dec/CPU-s
 (B) 3 truncations a seed (random_needs gave no death in 300); 45.0/46.0/71.1
     -> 59.9/59.8/102.7
 (C) rebuild at 150 kept native (asserted), 2 deaths a seed; 1.32-1.34x
 (D) implicitfast: 2/2/1 deaths; 86.6/97.2/129.3 -> 174.6/177.6/281.9 (2.0-2.2x)
 control dt+1ulp: differs on 273/286/287 of 300 (first at decision 27/7/13).
 FINDING (reported): the plain-order norm (fma_norm=False), the piece tests'
 control (370-775 differing drive fields a seed there), changes NO decision in
 300 x 3 here: its ulp drift in rest_dt/prev_speed never crosses a threshold
 that reaches the body. So it was too weak to be this test's control; the
 FMA order is still kept (exact field identity is what the piece tests prove).
 Identity is a property of THIS box's BLAS order: a Kaggle run must re-prove
 it there before the option is used there.
"""
import sys
import time
import numpy as np

from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
from experiments.native_substep import _native_decide, apply_native_substep

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = (153, 154, 155)
CASES = {
    "A": dict(e0=0.05, lazy_contact_audio=True),
    "B": dict(difficulty=0.75, reward="dr100-eat", random_needs=NEEDS_S_TRP,
              random_layout=True, move_cost_dial=True,
              spawn_clear=SPAWN_CLEAR_R, max_episode_decisions=100,
              lazy_contact_audio=True),
    "C": dict(e0=0.05, lazy_contact_audio=True),
    "D": dict(e0=0.05, lazy_contact_audio=True, integrator="implicitfast"),
}


def same(x, y) -> bool:
    if isinstance(x, dict):
        return (isinstance(y, dict) and x.keys() == y.keys()
                and all(same(x[k], y[k]) for k in x))
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return np.array_equal(np.asarray(x), np.asarray(y))
    return x == y


def pair(seed, case, control=None):
    kw = dict(CASES[case], seed=seed)
    a = JackEnv(**kw)
    b = JackEnv(**kw, native_substep=True)
    if control == "fma":             # rebuild the twins in the plain order
        apply_native_substep(b.w, fma_norm=False)
    elif control == "ulp":           # C drives see dt + 1 ulp
        nd = b.w._native["drives"]
        nd.substep = lambda data, dt, nd=nd: nd.lib.jn_drives_substep(
            data._address, nd.cfg, nd.s, float(np.nextafter(dt, 1.0)))
    oa, ia = a.reset(seed=seed)
    ob, ib = b.reset(seed=seed)
    rng = np.random.default_rng(seed)
    bad = deaths = truncs = 0
    ta = tb = 0.0
    first_bad = None
    for t in range(N):
        if case == "C" and t == N // 2:
            oa, ia = a.reset(seed=seed + 1000)
            ob, ib = b.reset(seed=seed + 1000)
            assert b.w.decide.__func__ is _native_decide, "rebuild lost native"
            assert a.w.decide.__func__ is not _native_decide
            bad += int(not (same(oa, ob) and same(ia, ib)))
        act = rng.uniform(-1.0, 1.0, a.action_space.shape)
        t0 = time.process_time()
        ra = a.step(act)
        ta += time.process_time() - t0
        t0 = time.process_time()
        rb = b.step(act)
        tb += time.process_time() - t0
        ok = (same(ra, rb) and np.array_equal(a.w.data.qpos, b.w.data.qpos)
              and np.array_equal(a.w.data.qvel, b.w.data.qvel))
        if ra[2] or ra[3]:
            deaths += int(ra[2])
            truncs += int(ra[3] and not ra[2])
            oa, ia = a.reset()
            ob, ib = b.reset()
            ok = ok and same(oa, ob) and same(ia, ib)
        if not ok:
            bad += 1
            first_bad = t if first_bad is None else first_bad
    return dict(bad=bad, first_bad=first_bad, deaths=deaths, truncs=truncs,
                rebirths=int(b.w.deaths), shipped=N / ta, native=N / tb)


def main() -> int:
    t0 = time.process_time()
    passed = True
    try:
        JackEnv(native_substep=True)
        print("guard: FAIL (no error without lazy_contact_audio)")
        passed = False
    except ValueError:
        print("guard: native_substep without lazy_contact_audio raises")
    for case in CASES:
        for seed in SEEDS:
            r = pair(seed, case)
            ok = r["bad"] == 0 and (case == "B" or r["deaths"] >= 1)
            passed &= ok
            print(f"({case}) seed {seed}: differing {r['bad']}/{N}, deaths "
                  f"{r['deaths']} truncations {r['truncs']} (world rebirths "
                  f"{r['rebirths']}), {r['shipped']:.1f} -> {r['native']:.1f} "
                  f"dec/CPU-s  {'ok' if ok else 'FAIL'}", flush=True)
    for seed in SEEDS:
        r = pair(seed, "A", control="ulp")
        ok = r["bad"] >= 1
        passed &= ok
        print(f"(control dt+1ulp) seed {seed}: differing {r['bad']}/{N}"
              f" (first at {r['first_bad']})  "
              f"{'fails as it must' if ok else 'FAIL: control identical'}",
              flush=True)
    for seed in SEEDS:
        r = pair(seed, "A", control="fma")
        print(f"(reported: fma_norm=False) seed {seed}: differing "
              f"{r['bad']}/{N} (first at {r['first_bad']})", flush=True)
    print(f"{'PASS' if passed else 'FAIL'}  ({time.process_time() - t0:.0f} CPU-s)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
