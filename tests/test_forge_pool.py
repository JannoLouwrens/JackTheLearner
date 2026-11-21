"""The forge's pool file (F2): the real file is well formed, and every charter
clause the checker enforces is caught when a planted pool breaks it."""
import re
import tempfile
from pathlib import Path

from experiments.forge import SOURCES
from experiments.forge import pool as P


def line(i="K1", source="published", wall="keep", change="resets every 200k",
         prior="§21 — Nikishin 2022 (2205.07802)", ready="knob", extra=""):
    parts = [f"- **{i}**"]
    for k, v in (("source", source), ("wall", wall), ("change", change),
                 ("prior", prior), ("ready", ready)):
        if v is not None:
            parts.append(f"{k}: {v}")
    return " · ".join(parts) + extra


CONTROL = line("C1", source="control", change="plain variation", prior="FORGE.md machine 5")


def refused(text, match):
    """The planted pool must be REFUSED, and for the stated reason."""
    try:
        P.check(text)
    except P.PoolError as e:
        assert re.search(match, str(e)), f"refused for the wrong reason: {e}"
        return
    raise AssertionError(f"a pool that should be refused ({match}) passed")


def pool(*lines):
    return "# pool\n\nprose\n\n" + "\n".join(lines) + "\n"


def test_real_pool_is_well_formed():
    cands = P.check(P.POOL.read_text())
    assert 1 <= len(cands) <= P.MAX_CANDIDATES
    assert P.main([str(P.POOL)]) == 0
    # the page's own count line must agree with its candidate lines
    n = P.counts(cands)
    text = P.POOL.read_text()
    for s in ("published", "recombine", "inventor", "machine", "control"):
        assert f"{s} {n[s]}" in text, f"'Sources, counted' disagrees for {s}: file has {n[s]}"


def test_planted_good_pool_passes():
    cands = P.check(pool(line(), CONTROL))
    assert [c["id"] for c in cands] == ["K1", "C1"]
    assert cands[0]["prior"].startswith("§21")
    assert P.counts(cands)["control"] == 1 and set(P.counts(cands)) == set(SOURCES)


def test_no_prior_is_refused():
    refused(pool(line(prior=None), CONTROL), "law 3a")
    refused(pool(line(prior=""), CONTROL), "law 3a")


def test_prior_that_cites_nothing_is_refused():
    refused(pool(line(prior="it seems like a good idea"), CONTROL), "cites nothing")


def test_unknown_source_is_refused():
    refused(pool(line(source="hunch"), CONTROL), "source")


def test_unknown_wall_or_ready_is_refused():
    refused(pool(line(wall="both"), CONTROL), "wall")
    refused(pool(line(ready="soon"), CONTROL), "ready")


def test_repeated_id_is_refused():
    refused(pool(line(), line(), CONTROL), "twice")


def test_more_than_twenty_is_refused():
    many = [line(f"K{i}") for i in range(20)]
    refused(pool(*many, CONTROL), "at most 20")
    assert len(P.check(pool(*many[:19], CONTROL))) == 20


def test_no_control_is_refused():
    refused(pool(line(), line("K2")), "control")


def test_malformed_field_is_an_error_not_a_skip():
    refused(pool(line(extra=" · just some words"), CONTROL), "key: value")


def test_ordinary_bold_bullets_are_not_candidates():
    text = pool(line(), CONTROL, "- **Already lost here:** pink noise; γ 0.99 · and more")
    assert len(P.check(text)) == 2


def test_missing_file_is_refused():
    with tempfile.TemporaryDirectory() as d:
        assert P.main([str(Path(d) / "nope.md")]) == 2


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e))
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
