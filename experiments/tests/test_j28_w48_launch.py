"""J28 W48 launcher: waits for the week AND (ear's twin or Sunday); cell 1's failure does not stop it."""
import datetime as dt
import importlib.util

_s = importlib.util.spec_from_file_location("j28l", "/home/opc/jackthelearner/scripts/j28_w48_launch.py")
L = importlib.util.module_from_spec(_s)
_s.loader.exec_module(L)
U = dt.timezone.utc


def test_fire_rule():
    assert L.fire_reason(dt.datetime(2026, 11, 27, 23, 0, tzinfo=U), True) is None     # week not open
    assert L.fire_reason(dt.datetime(2026, 11, 28, 5, 0, tzinfo=U), False) is None     # no twin, before Sun
    assert "twin" in L.fire_reason(dt.datetime(2026, 11, 28, 5, 0, tzinfo=U), True)
    assert "fallback" in L.fire_reason(dt.datetime(2026, 11, 29, 0, 2, tzinfo=U), False)


def test_chain_stop_rule():
    seen = []
    def run(rcs):
        def f(c):
            seen.append(c[5]); return rcs.get(c[5], 0)
        return f
    assert L.run_chain(run({"experiments.body.mjx_timing": 1}), False) == 0 and len(seen) == 5   # cell 1 fails: goes on
    seen.clear()
    assert L.run_chain(run({"experiments.brain_a.fp16_timing": 1}), False) == 1 and len(seen) == 2  # refusal stops
