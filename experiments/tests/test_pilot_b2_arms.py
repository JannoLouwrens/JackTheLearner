"""J11 item 3 prep: PILOT_ARMS map to the settings the order names, and the
B0 world (no cure) is the card's world byte for byte. No learner runs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import pilot_b2 as P  # noqa: E402
from experiments.pilot_b import REWARD, _world_stamp  # noqa: E402


def main():
    ok = True
    want = {"B0": (False, REWARD), "K": (True, REWARD), "P": (False, "dr100-eat-pos"),
            "G": (False, REWARD), "KP": (True, "dr100-eat-pos"),
            "noop": (False, REWARD), "held8": (False, REWARD), "K-lr0": (True, REWARD)}
    for ff in (False, True):
        for arm, (k, rew) in want.items():
            e = P.pilot_world(arm, ff)(402)
            st = _world_stamp(e)
            good = (st["move_cost_dial"] == k and st["reward"] == rew
                    and st["food_floor_only"] == ff and st["lazy_contact_audio"] is True
                    and st["spawn_clear"] == P.make_world(402).spawn_clear)
            ok &= good
            print(f"ff={ff} {arm:6s} {st['move_cost_dial']!s:5s} {st['reward']:14s} "
                  f"ff={st['food_floor_only']} {'ok' if good else 'BAD'}")
    ok &= P.PILOT_ARMS["G"]["gamma"] == 0.999 and P.PILOT_ARMS["K-lr0"]["lr"] == 0.0
    ok &= all(P.PILOT_ARMS[a]["gamma"] is None for a in ("B0", "K", "P", "KP"))
    # identity: B0 world == card world (pilot_b2.make_world defaults), 50 decisions
    for ff in (False, True):
        a, b = P.pilot_world("B0", ff)(402), P.make_world(402, food_floor_only=ff)
        oa, _ = a.reset(seed=402); ob, _ = b.reset(seed=402)
        rng = np.random.default_rng(0)
        same = np.array_equal(oa, ob)
        for _ in range(50):
            act = rng.uniform(-1, 1, a.action_space.shape).astype(np.float32)
            ra, rb = a.step(act), b.step(act)
            same &= np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]
        print(f"identity B0 == card world (ff={ff}): {same}")
        ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
