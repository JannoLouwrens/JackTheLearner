"""PLAN stage-1 ready-list: "the interface correct at death (the dying body's
senses, not the reborn one's)" — ORDERS Next-up 3, planted box test on the
wrappers J30 and the forge actually run.

Worlds, built by the runs' own factory (`pilot_b2.pilot_world` with
`PILOT_RULE`'s frames) and wrapped in `pilot_b.PilotLives` (what run_b /
r1_run / j24_chain put around every world):
  J30    arm "K1-bins16"  (kind paint, bins16 eye, body frames)
  forge  arm "E-food"     (r1_run.WORLD_ARM: + HandedSense 'food' channel)

At every death a draw-free spy on `W0.respawn` reads the dying body's own
needs and proprio (and, for E-food, the handed channel the wrapper computes
for the body at that instant). The observation PilotLives returns on the
step with terminated=True must equal the spy's, block by block.

  mode "reborn" (JackEnv's default — what J30 and r1L were RUN with): the
      check must FAIL (control: the test can fail).
  mode "dying"  (J9's flag, set on the unwrapped JackEnv): JackEnv's blocks
      must PASS through every wrapper; each wrapper block is judged apart.

A death is hastened by setting energy to 0.02 after each birth (test-only;
the world's stream otherwise untouched). Box-seconds, no learner.

Run: /data/venvs/jack-std/bin/python -m tests.test_death_interface
"""
import dataclasses
import sys

import numpy as np

ARMS = ("K1-bins16", "E-food")
SEED = 402
DEATHS = 2
MAX_STEPS = 3000


def _world(arm: str, mode: str):
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import PILOT_RULE, pilot_world
    env = pilot_world(arm, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                      PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])(SEED)
    je = env.unwrapped
    je.terminal_obs = mode
    je.w.capture_dying_obs = mode == "dying"
    return pb.PilotLives(env), env, je


def _starve(je):
    je.w.drives.state = dataclasses.replace(je.w.drives.state, e=0.02)


def run(arm: str, mode: str) -> dict:
    lives, env, je = _world(arm, mode)
    w = je.w
    handed = hasattr(env, "extra") and hasattr(env, "n_extra")
    n_extra = int(env.n_extra) if handed else 0
    seen = []
    orig = w.respawn

    def spy(*a, **kw):
        seen.append({"needs": w._needs().copy(), "proprio": w._proprio().copy(),
                     "extra": env.extra().copy() if handed else None})
        return orig(*a, **kw)
    w.respawn = spy
    rng = np.random.RandomState(7)
    lives.reset(seed=SEED)
    _starve(je)
    got, after, steps = [], [], 0
    while len(got) < DEATHS and steps < MAX_STEPS:
        o, r, term, trunc, info = lives.step(rng.uniform(-1, 1, lives.action_space.shape))
        steps += 1
        if term:
            got.append(np.asarray(o, dtype=np.float64))
            after.append(env.extra().copy() if handed else None)
            ro, _ = lives.reset()
            _starve(je)
        elif trunc:
            lives.reset()
    if len(got) < DEATHS or len(seen) != len(got):
        return {"ok": None, "why": f"{len(got)} deaths in {steps} steps, spy {len(seen)}"}
    n_p = len(seen[0]["proprio"])
    res = {"needs": True, "proprio": True, "handed": None if not handed else True,
           "handed_is_reborn": None if not handed else True}
    for o, sp, af in zip(got, seen, after):
        end = len(o) - n_extra
        res["needs"] &= np.allclose(o[end - 6:end], sp["needs"], atol=1e-6)
        res["proprio"] &= np.allclose(o[end - 6 - n_p:end - 6], sp["proprio"], atol=1e-6)
        if handed:
            res["handed"] &= np.allclose(o[end:], sp["extra"], atol=1e-6)
            # the block equals the channel read AFTER the respawn (so the
            # handed check can match a body; it matches the reborn one)
            res["handed_is_reborn"] &= np.allclose(o[end:], af, atol=1e-6)
    res["e_dying"] = [round(float(sp["needs"][0]), 4) for sp in seen]
    res["lives_logged"] = len(lives.lives)
    return res


def main() -> int:
    fails = 0
    for arm in ARMS:
        for mode in ("reborn", "dying"):
            r = run(arm, mode)
            print(f"{arm:10s} {mode:6s} {r}")
            if r.get("ok", 1) is None:
                fails += 1
                continue
            core = r["needs"] and r["proprio"]
            if mode == "reborn" and core:
                print("  FAIL: control passed — the check cannot tell bodies apart")
                fails += 1
            if mode == "dying" and not core:
                print("  FAIL: JackEnv's dying senses lost through the wrappers")
                fails += 1
            if mode == "dying" and r["handed"] is False and r["handed_is_reborn"]:
                print("  FINDING: HandedSense's channel on the death step is the "
                      "REBORN body's (extra() reads the world after respawn)")
    print("RESULT", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
