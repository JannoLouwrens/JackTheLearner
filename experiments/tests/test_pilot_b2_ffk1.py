"""J11 option (a)'s gap (11-18 06:15): groups f4..f7 hold exactly the 16
held8-k1-ff card cells (seeds 601..616, 4 a kernel); f0..f3 and c0..c7 are
unchanged from before the addition (the f0..f2 kernels in flight ran those);
a held8-k1-ff cell builds the pilot's held8 floor world (movement charged,
food_floor_only on) and one life runs. Control: held8-k0-ff builds dial ON."""
from experiments import pilot_b2 as P

# f0 as submitted 11-18 01:28 (c9908359): seeds 601..604 x CARD_FF_ARMS.
F0_SUBMITTED = [(a, s) for s in range(601, 605) for a in ("noop-k1-ff", "held8-k0-ff")]


def test_groups():
    g = P.card_groups()
    assert g["f0"] == F0_SUBMITTED
    assert all(len(g[f"c{i}"]) == 8 for i in range(8))
    assert all(len(g[f"f{i}"]) == 8 for i in range(4))
    new = [c for i in range(4, 8) for c in g[f"f{i}"]]
    assert new == [("held8-k1-ff", s) for s in range(601, 617)]
    assert all(len(g[f"f{i}"]) == 4 for i in range(4, 8))
    assert "f8" not in g
    # the pilot's held8 floor is k1: the card twin must be too
    assert P.PILOT_ARMS["held8"].get("move_cost_dial", False) is False


def test_world_and_one_life(tmp_path):
    seen = {}
    real = P.make_world

    def spy(seed, move_cost_dial=False, food_floor_only=False, **kw):
        seen[(move_cost_dial, food_floor_only)] = True
        return real(seed, move_cost_dial=move_cost_dial, food_floor_only=food_floor_only, **kw)

    P.make_world = spy
    try:
        rec = P.run_card_cell("held8-k1-ff", 601, lives=1, out=tmp_path / "x.json")
        assert seen == {(False, True): True}
        assert rec["done"] and len(rec["lives"]) == 1
        seen.clear()
        P.run_card_cell("held8-k0-ff", 601, lives=1)  # control: dial on
        assert seen == {(True, True): True}
    finally:
        P.make_world = real


def main():
    import tempfile
    from pathlib import Path
    test_groups()
    with tempfile.TemporaryDirectory(dir="/data") as d:
        test_world_and_one_life(Path(d))
    print("test_pilot_b2_ffk1 PASS")


if __name__ == "__main__":
    main()
