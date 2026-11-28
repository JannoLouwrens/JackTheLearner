"""Ruling 138 / ORDERS Next-up 2: HandedSense's channel at death.

`tests/test_death_interface.py` found (11-27) that under terminal_obs="dying"
the forge's HandedSense 'food' channel on the death step is the REBORN body's
(extra() reads the world after W0 respawned it). `HandedSense(death_extra=)`
is the fix as an option; its default "reborn" is today's behaviour, so r1L's
locked runs are untouched. No run uses "dying" yet.

Same world, same spy, same loop as test_death_interface (arm E-food, JackEnv
terminal_obs="dying", seed 402, 2 deaths):
  death_extra "reborn" (default) : the channel on the death step must NOT be
                                   the dying body's (control: the check can fail)
  death_extra "dying"            : it MUST equal the spy's, read at respawn
and the default must be "reborn". Box-seconds, no learner.

Run: /data/venvs/jack-std/bin/python tests/test_handed_death.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.handed_sense import HandedSense  # noqa: E402
from tests import test_death_interface as tdi  # noqa: E402


def _find_handed(env):
    e = env
    while e is not None:
        if isinstance(e, HandedSense):
            return e
        e = e.__dict__.get("env")
    raise AssertionError("no HandedSense in the E-food world")


def run(death_extra):
    orig = tdi._world

    def world(arm, mode):
        lives, env, je = orig(arm, mode)
        hs = _find_handed(env)
        assert hs.death_extra == "reborn", "default must be today's"
        if death_extra is not None:
            hs.death_extra = death_extra
        return lives, env, je
    tdi._world = world
    try:
        return tdi.run("E-food", "dying")
    finally:
        tdi._world = orig


def main() -> int:
    fails = 0
    for de in ("reborn", "dying"):
        r = run(de)
        print(f"death_extra={de:6s} {r}")
        if r.get("ok", 1) is None:
            print("  FAIL: no deaths"); fails += 1; continue
        if not (r["needs"] and r["proprio"]):
            print("  FAIL: JackEnv's dying blocks lost"); fails += 1
        if de == "reborn" and r["handed"]:
            print("  FAIL: control passed — default already gives the dying channel?"); fails += 1
        if de == "dying" and not r["handed"]:
            print("  FAIL: death_extra='dying' does not carry the dying body's channel"); fails += 1
    try:
        HandedSense.__init__.__defaults__
        assert HandedSense.__init__.__defaults__[-1] == "reborn"
    except AssertionError:
        print("FAIL: default is not 'reborn'"); fails += 1
    print("RESULT", "PASS" if fails == 0 else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
