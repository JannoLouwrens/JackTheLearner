"""J32 DRAFT (prereg/pilot_b20.md): the pilot's two arms.

Planted: B20 arms are J30's K1 dicts with ONLY the reset schedule (to 1M) and
terminal_obs="dying" changed; J30's K1 dicts carry no terminal_obs (their world
stays "reborn"); the world built for a B20 arm really is "dying". Seeds and size
are the order's. Run under /data/venvs/jack-std."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pilot_b2 as p

PAIRS = {"B20-bins16": "K1-bins16", "B20-bins16-blind": "K1-bins16-blind"}


def test_b20_is_j30_cell_plus_dying_and_longer_resets():
    assert set(p.PILOT_B20_ARMS) == set(PAIRS)
    for b, k in PAIRS.items():
        diff = {key for key in set(p.PILOT_B20_ARMS[b]) | set(p.J24K1_ARMS[k])
                if p.PILOT_B20_ARMS[b].get(key) != p.J24K1_ARMS[k].get(key)}
        assert diff == {"param_resets", "terminal_obs"}, (b, diff)
        assert p.PILOT_B20_ARMS[b]["terminal_obs"] == "dying"
        assert p.PILOT_B20_ARMS[b]["param_resets"] == list(range(50_000, 1_000_000, 50_000))
        assert p.PILOT_B20_ARMS[b]["ent_coef"] == "auto"


def test_j30_arms_untouched():
    for k in PAIRS.values():
        assert "terminal_obs" not in p.J24K1_ARMS[k]
        assert p.J24K1_ARMS[k]["param_resets"] == list(range(50_000, 500_000, 50_000))


def test_order_seeds_and_size():
    assert p.B20_T == 1_000_000
    assert p.B20_SEEDS == tuple(range(2001, 2021)) and len(p.B20_SEEDS) == 20
    assert p.B20_BLIND_SEEDS == (2001, 2002, 2003, 2004, 2005)


def test_world_mode_reaches_the_env():
    r = p.PILOT_RULE
    for arm, mode in (("B20-bins16", "dying"), ("B20-bins16-blind", "dying"),
                      ("K1-bins16", "reborn")):
        env = p.pilot_world(arm, r["food_floor_only"], r["eye_frame"], r["motor_frame"],
                            r["native_substep"])(2001)
        je = env.unwrapped
        assert je.terminal_obs == mode, (arm, je.terminal_obs)
