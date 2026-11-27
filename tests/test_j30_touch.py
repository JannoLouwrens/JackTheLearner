"""J30 (1): the life row's BESIDE `touch` columns (pilot_b.PilotLives).

  known-positive  the hand forager on the food bins (j24_bins_forager, an
                  adult of his world, not a learner) in J30's own world: when
                  it has eaten a floor meal, the row says it touched food.
  must-not        a do-nothing body that ate nothing shows no food touch in
                  that life (the counter is not a constant); a world that is
                  not kind-painted carries no `touch` key (row unchanged).

That the counting changes nothing in the run is tests/test_j24_chain.py (the
identities) plus the before/after actor hash in the commit message.

    taskset -c 3 nice -n 19 /data/venvs/jack-std/bin/python tests/test_j30_touch.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED, MAX_DECISIONS = 951, 6000        # a smoke seed, never 941-943


def _lives(arm, policy, n):
    from experiments import pilot_b as pb
    from experiments.pilot_b2 import PILOT_RULE, pilot_world
    w = pb.PilotLives(pilot_world(arm, PILOT_RULE["food_floor_only"], PILOT_RULE["eye_frame"],
                                  PILOT_RULE["motor_frame"], PILOT_RULE["native_substep"])(SEED))
    o, info = w.reset(seed=SEED)
    for _ in range(n):
        o, r, term, trunc, info = w.step(policy(o))
        if term or trunc:
            o, info = w.reset()
        if any(l["floor_meals"] > 0 for l in w.lives):
            break
    return w.lives + [w.unfinished({"ate_total": dict(w.env.w.drives.ate_total)})]


def test_forager_touch_says_food():
    from experiments.fast_sim import KINDS
    from experiments.j24_bins_forager import BinsForager
    pol = BinsForager(SEED)
    lives = _lives("K1-bins16", pol.act, MAX_DECISIONS)
    fed = [l for l in lives if l["floor_meals"] > 0]
    assert fed, "the hand forager ate nothing: the known-positive did not happen"
    for l in lives:
        assert tuple(l["touch"]) == KINDS, l
        assert all(0 <= v <= l["decisions"] for v in l["touch"].values()), l
    assert all(l["touch"]["food"] >= 1 for l in fed), fed
    print(f"forager: {len(lives)} lives, fed {[(l['floor_meals'], l['touch']) for l in fed]}")


def test_still_body_and_unpainted_world():
    from experiments import world_card as WC
    still = lambda o: WC.NEUTRAL.copy()
    lives = _lives("K1-bins16", still, 300)
    unfed = [l for l in lives if l["floor_meals"] == 0]
    assert unfed and all(l["touch"]["food"] == 0 for l in unfed), lives
    plain = _lives("K", still, 50)
    assert all("touch" not in l for l in plain), plain
    print(f"still body: food touch 0 in {len(unfed)} unfed lives; unpainted world: no key")


if __name__ == "__main__":
    test_forager_touch_says_food()
    test_still_body_and_unpainted_world()
    print("PASS")
