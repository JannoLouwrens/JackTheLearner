"""Planted reads for the ruling-91 twin's pure comparison (no world run)."""
from experiments.lf03_resume_twin import ARMS, compare, verdict

R = [{"g": "a", "x": 1.0}, {"g": "b", "x": 2.0}]


def test_identical_rows_read_identical():
    assert compare(R, [dict(r) for r in R])["all_identical"]


def test_one_feature_off_is_named():
    c = compare(R, [R[0], {"g": "b", "x": 2.5}])
    assert not c["all_identical"] and c["first_diff"] == 1 and c["diff_keys"] == ["x"]


def test_short_resume_is_not_identical():
    assert not compare(R, R[:1])["all_identical"]


def test_verdicts():
    same, diff = compare(R, R), compare(R, R[::-1])
    assert verdict({a: (same if a == "RC" else diff) for a in ARMS}).startswith("RESUMES; every")
    assert verdict({a: same for a in ARMS}).startswith("RESUMES; drop NOT seen")
    assert verdict({a: diff for a in ARMS}).startswith("RESUME DIFFERS")
