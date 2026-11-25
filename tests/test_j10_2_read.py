"""j10_2_read gathers the right segment files and hands them to the frozen verdict; it can read PASS and FAIL."""
import json

from experiments import j10_2_read as r


def _write(root, seed, seg, R, M, hit=True):
    (root / f"s{seed}_seg{seg}.json").write_text(json.dumps(
        {"seed": seed, "segment": seg, "hit_target": hit, "eval": {"R": R, "M": M}}))


def _row(tmp_path, s7_2m):
    for s, (r1, r2, m2) in {3: (4500, 5649, 12.1), 4: (5420, 6254, 17.2), 5: (5220, 6108, 17.3),
                            6: (4541, 5993, 17.9)}.items():
        _write(tmp_path, s, 1, r1, 5.0)
        _write(tmp_path, s, 3, r2, m2)
    _write(tmp_path, 7, 1, 5283, 6.3)
    if s7_2m:
        _write(tmp_path, 7, 3, *s7_2m)
    r1m, r2m, m2m, _ = r.gather(str(tmp_path))
    return r.verdict_j10_2(r2m, m2m, r1m, r.J10_1_CONTROL_R)


def test_pass_when_s7_meets(tmp_path):
    assert _row(tmp_path, (6000, 13.0))["reading"] == "PASS"


def test_stands_when_s7_misses_only_on_metres(tmp_path):
    assert _row(tmp_path, (6000, 11.9))["reading"] == "STANDS"


def test_fail_when_s7_misses_on_return(tmp_path):
    assert _row(tmp_path, (5900, 13.0))["reading"] == "FAIL"


def test_missing_seed_counts_as_miss(tmp_path):
    v = _row(tmp_path, None)
    assert v["reading"] == "FAIL" and v["1m"] == "IN BAND"


def test_unfinished_segment_not_read(tmp_path):
    _write(tmp_path, 7, 3, 9000, 20.0, hit=False)
    assert 7 not in r.gather(str(tmp_path))[1]
