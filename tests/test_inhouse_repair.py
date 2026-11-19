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


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_"):
            _f(); print("PASS", _n)
