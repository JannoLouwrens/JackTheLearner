"""J14 (b): the W48 ear poller fires only on A's real seed-10 launch (+grace) or the fallback."""
import datetime as dt
import importlib.util

spec = importlib.util.spec_from_file_location("kws_w48_launch", "/home/opc/jackthelearner/scripts/kws_w48_launch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
U = dt.timezone.utc
HEADER = ("LAUNCH 2026-11-26T10:12:22Z cwd=/home/opc/jackthelearner cmd: /data/venvs/jackthelearner/bin/python "
          "scripts/a2_enc_sat_launch.py\n2026-11-26T10:12:22Z a2_enc_sat_launch now=2026-11-26T10:12:22Z\n")
OPEN = "2026-11-28T00:01:00Z W48 open\n"
A10 = ("2026-11-28T00:01:30Z LAUNCH /data/venvs/jackthelearner/bin/python -m experiments.brain_a.rows "
       "enc-kernel 10 --control cwd=/data/wt/jack-a2enc\n")


def t(h, mi):
    return dt.datetime(2026, 11, 28, h, mi, tzinfo=U)


def test_before_opens_never_fires():
    assert m.fire_reason(dt.datetime(2026, 11, 27, 23, 59, tzinfo=U), HEADER + OPEN + A10) is None


def test_header_alone_does_not_count_as_a_launch():   # control: the unanchored-grep scar
    assert m.a_launched_at(HEADER) is None
    assert m.fire_reason(t(0, 30), HEADER) is None


def test_a10_before_w48_open_does_not_count():
    assert m.a_launched_at(HEADER + A10) is None


def test_fires_after_a10_plus_grace_only():
    txt = HEADER + OPEN + A10
    assert m.a_launched_at(txt) == dt.datetime(2026, 11, 28, 0, 1, 30, tzinfo=U)
    assert m.fire_reason(t(0, 10), txt) is None
    assert "grace" in m.fire_reason(t(0, 17), txt)


def test_fallback_fires_without_a():
    assert m.fire_reason(t(2, 0), HEADER) is None
    assert "fallback" in m.fire_reason(t(2, 1), HEADER)
