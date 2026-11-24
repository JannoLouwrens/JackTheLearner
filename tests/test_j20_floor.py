"""J20 (2): the floor gate and the 3.2.3 pairing can fire (planted rows)."""
import json

from experiments import j20_reread as J
from experiments import world_card as WC


def _rows(row, meals_by_seed, life_s=300.0):
    return [{"row": row, "seed": s, "life": i, "floor_meals": meals_by_seed.get(s, 0),
             "lifetime_s": life_s} for s in WC.REGISTERED_SEEDS for i in range(WC.LIVES)]


def test_gate_holds_on_a_starving_floor():
    g = J.floor_gate(_rows("a", {}), _rows("b", {s: 1 for s in WC.REGISTERED_SEEDS}))
    assert g["holds"] and g["total"] == 0


def test_gate_fires_when_a_floor_arm_eats():
    seeds = WC.REGISTERED_SEEDS
    g = J.floor_gate(_rows("a", {seeds[0]: 6, seeds[1]: 6}), _rows("b", {}))
    assert g["a_seeds"] == 2 and not g["holds"]


def test_b_reaching_five_is_printed_not_hidden():
    g = J.floor_gate(_rows("a", {}), _rows("b", {WC.REGISTERED_SEEDS[0]: 5}))
    assert g["seeds_ge5_alone"]["b"] == 1


def test_pairing_sees_a_shift(tmp_path):
    old = _rows("b", {s: 0 for s in WC.REGISTERED_SEEDS})
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in old))
    up = J.floor_vs_old("b", _rows("b", {s: 2 for s in WC.REGISTERED_SEEDS}), tmp_path)
    assert up["vs_323_floor_meals"]["up"] == 20 and up["vs_323_floor_meals"]["wilcoxon_p"] < 0.01
    same = J.floor_vs_old("b", old, tmp_path)
    assert same["vs_323_floor_meals"]["tie"] == 20
