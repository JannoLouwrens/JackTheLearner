"""J3 (5) repaired entry (11-19): the @repair arm changes exactly two settings.

identity: _gae_g at survival.GAMMA equals survival._gae on random segments;
planted:  at gamma 0.99 the targets differ (the knob is live);
seeds:    @repair cells use fresh seeds 5-9, plain arms keep 0-4;
refusal:  a recurrent arm with @repair raises (not silently unrepaired).
The byte-identity of a plain cell before/after the edit was checked by hand
at commit time (wm-latent rows 1 and 3, 400 steps: only cpu_s differs).
"""
import numpy as np

from experiments import inhouse_rows as ir
from experiments.survival import GAMMA, _gae


def test_gae_identity_and_live_knob():
    rng = np.random.RandomState(0)
    for n in (1, 2, 17, 64):
        r = list(rng.randn(n)); v = list(rng.randn(n)); b = float(rng.randn())
        for term in (False, True):
            a0, t0 = _gae(r, v, b, term)
            a1, t1 = ir._gae_g(r, v, b, term, GAMMA)
            assert np.array_equal(a0, a1) and np.array_equal(t0, t1)
    r = list(rng.randn(40)); v = list(rng.randn(40))
    assert not np.allclose(_gae(r, v, 1.0, False)[1],
                           ir._gae_g(r, v, 1.0, False, ir.REPAIR_GAMMA)[1])


def test_fresh_seeds_and_cells():
    assert ir.seeds_for("wm-latent") == (0, 1, 2, 3, 4)
    assert ir.seeds_for("wm-latent@repair") == (5, 6, 7, 8, 9)
    c = ir.cells(["wm-latent@repair"])
    assert len(c) == 20 and {s for _, _, s, _ in c} == {5, 6, 7, 8, 9}


def test_recurrent_repair_refuses():
    try:
        ir.run_cell("wm-dyn@repair", 1, 5, steps=10)
    except ValueError:
        return
    raise AssertionError("recurrent @repair did not refuse")


def test_diag_arms_settings_rows_seeds():
    """J3 (5'): each diag arm changes exactly one setting (or none), row 1 only."""
    assert ir._settings("wm-latent@repair") == ("wm-latent", ir.REPAIR_HOLD, ir.REPAIR_GAMMA)
    assert ir._settings("wm-latent@gamma-only") == ("wm-latent", 1, ir.REPAIR_GAMMA)
    assert ir._settings("wm-latent@hold-only") == ("wm-latent", ir.REPAIR_HOLD, None)
    assert ir._settings("wm-latent@plain59") == ("wm-latent", 1, None)
    assert ir._settings("wm-latent") == ("wm-latent", 1, None)
    c = ir.cells(["wm-latent" + a for a in ir.DIAG])
    assert len(c) == 30 and {r for _, r, _, _ in c} == {1}
    assert {s for _, _, s, _ in c} == {5, 6, 7, 8, 9}
    assert ir.rows_for("wm-latent@repair") == (1, 3)


def test_diag_plain59_is_plain_and_reading_fires(tmp_path=None):
    """@plain59 seed 5 == the plain loop at seed 5 (only cpu_s/arm differ);
    the reading names the planted cause and voids on a non-learning plain."""
    import json, tempfile
    from pathlib import Path
    a = ir.run_cell("wm-latent@plain59", 1, 5, steps=300)
    b = ir.run_cell("wm-latent", 1, 5, steps=300)
    assert a["returns"] == b["returns"] and a["opt_steps"] == b["opt_steps"]
    gc = ir.run_cell("wm-latent@gamma-only", 1, 5, train=False, steps=300)
    bc = ir.run_cell("wm-latent", 1, 5, train=False, steps=300)
    assert gc["returns"] == bc["returns"]        # same white noise, same draws
    g = ir.run_cell("wm-latent@gamma-only", 1, 5, steps=300)
    assert g["returns"] != b["returns"]          # gamma reaches the update
    h = ir.run_cell("wm-latent@hold-only", 1, 5, steps=300)
    assert h["returns"] != b["returns"]          # the held-noise knob is live
    d = Path(tempfile.mkdtemp())
    def plant(arm, ms):
        for s, m in zip(ir.REPAIR_SEEDS, ms):
            (d / ir._name("wm-latent" + arm, 1, s, False)).write_text(json.dumps({"metric": m}))
    assert ir.diag_reading(d)["reading"] == "INCOMPLETE"
    plant("@plain59", [-350] * 5); plant("@hold-only", [-400] * 4 + [-1200])
    plant("@gamma-only", [-1250] * 5)
    assert ir.diag_reading(d)["reading"] == "GAMMA 0.99 kills row 1"
    plant("@plain59", [-1200] * 5)
    assert ir.diag_reading(d)["reading"].startswith("VOID")


def test_hold8_one_setting_both_rows():
    """J3 (6), ruling 59: @hold8 = @repair with hold 8, rows 1 AND 3, seeds 10-14;
    the hold knob is live (differs from white-noise gamma-only and from hold 32
    at the same seed), and @repair/diag/plain settings are untouched."""
    assert ir._settings("wm-latent@hold8") == ("wm-latent", 8, ir.REPAIR_GAMMA)
    assert ir.rows_for("wm-latent@hold8") == (1, 3)
    assert ir.seeds_for("wm-latent@hold8") == (10, 11, 12, 13, 14)
    assert ir.seeds_for("wm-latent@repair") == (5, 6, 7, 8, 9)
    assert ir.REPAIR_HOLD == 32 and ir.REPAIR_GAMMA == 0.99
    c = ir.cells(["wm-latent@hold8"])
    assert len(c) == 20 and {s for _, _, s, _ in c} == set(ir.HOLD8_SEEDS)
    assert {r for _, r, _, _ in c} == {1, 3} and sum(ct for *_, ct in c) == 10
    h8 = ir.run_cell("wm-latent@hold8", 1, 10, train=False, steps=300)
    g = ir.run_cell("wm-latent@gamma-only", 1, 10, train=False, steps=300)
    h32 = ir.run_cell("wm-latent@repair", 1, 10, train=False, steps=300)
    assert h8["returns"] != g["returns"] and h8["returns"] != h32["returns"]
    assert h8["arm"] == "wm-latent@hold8" and h8["ctrl"] is True


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_"):
            _f(); print("PASS", _n)
