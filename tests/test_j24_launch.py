"""J24 (5): the kernel job for one seed's three carried cells (pure parts).

  freeze   no job for a pin whose prereg is a DRAFT (today's HEAD is one);
           a planted FROZEN prereg builds a job that pins its sha.
  refusals short pin, segment out of range, a seed not 941–943.
  job      compiles; holds the seed's three J24 arms; segment 0 starts fresh,
           segment n resumes from /kaggle/input.
  carry    find_carry: exactly one WHOLE carry or refused (none, two, partial).

    /data/venvs/jack-std/bin/python tests/test_j24_launch.py
"""
import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import j24_chain as jc  # noqa: E402
from experiments import j24_launch as jl  # noqa: E402
from experiments.forge import r1_run as rr  # noqa: E402

HEAD = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
FROZEN = b"# eye_bins16\n\n**STATUS: FROZEN 2026-11-26, ruling 999.**\n"
DRAFT = b"# eye_bins16\n\n**STATUS: DRAFT, NOT FROZEN (planted).**\n"
# Real history, not a plant: eye_bins16 was frozen at 03e6285d; its parent
# holds the DRAFT text. (Until 11-26 these tests read HEAD as the draft;
# that premise expired at the freeze, so the guard is now checked on the
# real pre-freeze commit and on a planted draft.)
FREEZE_SHA = "03e6285d"
PRE_FREEZE = subprocess.run(["git", "-C", str(REPO), "rev-parse", FREEZE_SHA + "^"],
                            capture_output=True, text=True).stdout.strip()


def refused(f):
    try:
        f()
    except rr.Refused:
        return True
    return False


def with_prereg(text, f):
    real = jl.prereg_at
    jl.prereg_at = lambda pin: text
    try:
        return f()
    finally:
        jl.prereg_at = real


def test_frozen_reads_the_status_line():
    assert not jl.frozen(None) and not jl.frozen(b"")
    assert not jl.frozen(jl.prereg_at(PRE_FREEZE)), "pre-freeze prereg is a draft"
    assert jl.frozen(jl.prereg_at(HEAD)), "eye_bins16 is frozen since 03e6285d"
    assert not jl.frozen(DRAFT)
    assert jl.frozen(FROZEN)
    assert not jl.frozen(b"**STATUS: DRAFT, NOT FROZEN**\n**STATUS: FROZEN x**\n"), \
        "only the first STATUS line counts"


def test_no_job_on_a_draft():
    assert refused(lambda: jl.job_body(941, PRE_FREEZE)), "the pre-freeze prereg is a draft"
    assert refused(lambda: with_prereg(DRAFT, lambda: jl.job_body(941, HEAD))), \
        "a planted draft at any pin gets no job"


def test_refusals():
    ok = lambda **k: with_prereg(FROZEN, lambda: jl.job_body(**k))  # noqa: E731
    assert refused(lambda: ok(seed=941, pin=HEAD[:10]))
    assert refused(lambda: ok(seed=941, pin=HEAD, segment=rr.MAX_SEGMENTS))
    assert refused(lambda: ok(seed=941, pin=HEAD, segment=-1))
    assert refused(lambda: ok(seed=951, pin=HEAD)), "a smoke seed never gets a job"


def test_job_compiles_and_holds_the_seeds_three_cells():
    b0 = with_prereg(FROZEN, lambda: jl.job_body(942, HEAD))
    b1 = with_prereg(FROZEN, lambda: jl.job_body(942, HEAD, segment=1))
    for b in (b0, b1):
        ast.parse(b)
        assert repr([(a, 942) for a in jc.J24_ONLY]) in b
        assert repr(hashlib.sha256(FROZEN).hexdigest()) in b
        assert "J24_T" not in b and "500000" in b
    assert "resume_root=None" in b0 and "resume_root='/kaggle/input'" in b1
    assert "'segment': 1" not in b0 and '"segment": 1' in b1


def _whole(d, partial=False):
    d.mkdir(parents=True)
    for n in rr.CARRY_NEEDS[:-1] if partial else rr.CARRY_NEEDS:
        (d / n).write_bytes(b"x")


def test_find_carry_one_whole_or_refused():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        name = jc.carry_name("K-bins16", 941)
        assert refused(lambda: jl.find_carry(root, "K-bins16", 941)), "none"
        _whole(root / "a" / "partial" / name, partial=True)
        assert refused(lambda: jl.find_carry(root, "K-bins16", 941)), "a partial is none"
        _whole(root / "a" / "b" / "c" / name)
        assert jl.find_carry(root, "K-bins16", 941) == root / "a" / "b" / "c" / name
        assert refused(lambda: jl.find_carry(root, "K-bins16-blind", 941)), "other arm"
        _whole(root / "z" / name)
        assert refused(lambda: jl.find_carry(root, "K-bins16", 941)), "two"


def test_cell_refuses_without_a_carry_dir():
    assert refused(lambda: jl.main(["cell", "--arm", "K-bins16", "--seed", "951",
                                    "--T", "576", "--out", "/dev/null"]))


def _dirs(t):
    jl.OUT, jl.SEGS = Path(t), Path(t) / "segments"


def test_kernel_refuses_before_any_submit():
    real = (jl.OUT, jl.SEGS)
    with tempfile.TemporaryDirectory() as t:
        _dirs(t)
        try:
            assert refused(lambda: jl.kernel(951, HEAD)), "smoke seed"
            assert refused(lambda: jl.kernel(941, HEAD, segment=1)), "needs seg 0 home"
            jl.raw_file(941).write_text("{}")
            assert refused(lambda: jl.kernel(941, HEAD)), "a seed runs once"
            jl.SEGS.mkdir()
            jl.seg_file(942, 0).write_text('{"pin": "%s", "cells": {}}' % ("f" * 40))
            assert refused(lambda: jl.kernel(942, HEAD, segment=1)), "one pin a run"
            jl.seg_file(943, 0).write_text('{"pin": "%s", "cells": {}}' % HEAD)
            assert refused(lambda: jl.kernel(943, HEAD, segment=1)), "carried nothing"
            assert refused(lambda: jl.kernel(943, HEAD)), "segment 0 already home"
        finally:
            jl.OUT, jl.SEGS = real


def test_chain_carries_waits_out_the_guard_and_goes_on():
    calls, slept = [], []

    def fake(seed, pin, seg):
        calls.append((seed, seg))
        if seed == 941 and seg < 2:
            return {"next_segment": seg + 1}
        if seed == 942 and len([c for c in calls if c[0] == 942]) == 1:
            raise RuntimeError("Uncommitted changes in experiments/x.py")
        if seed == 943:
            raise RuntimeError("kaggle said no")
        return {}
    done = jl.chain([941, 942, 943], HEAD, sleep=slept.append, kernel_fn=fake)
    assert calls == [(941, 0), (941, 1), (941, 2), (942, 0), (942, 0), (943, 0)], calls
    assert done[941] == "home" and done[942] == "home" and done[943].startswith("failed")
    assert slept == [rr.GUARD_WAIT_S]


def test_timing_body_runs_only_the_smoke_seed():
    from experiments import j24_launch as j
    b = j.timing_body("a" * 40)
    compile(b, "timing", "exec")
    assert "951" in b and not any(str(s) in b for s in j.SEEDS)
    assert "__" not in b.replace("__name__", "")
    assert j.TIMING_T == j.jc.CHUNK and f", {j.jc.CHUNK}," in b
    j.jc.rr.chunk_rule(j.TIMING_T, j.jc.CHUNK, 32, 5000)   # the rule that sank the 5k kernel
    for bad in (("abc", 20000), ("a" * 40, 0), ("a" * 40, 5000), ("a" * 40, 20001)):
        try:
            j.timing_body(*bad)
        except j.rr.Refused:
            continue
        raise AssertionError(f"timing_body took {bad}")


def test_k1_variant_own_prereg_files_and_cells():
    """J24 follow-up DRAFT: --variant k1 reads ITS prereg (a draft today, so
    no job), holds the K1 arms, writes under its own folder/label; switching
    back restores J24's exact names."""
    j24 = (jl.PREREG, jl.ARMS, jl.OUT, jl.SEGS, jl.TAG, jl.TIMING_FILE)
    try:
        jl.set_variant("k1")
        assert jl.PREREG == "docs/research/prereg/eye_bins16_k1.md"
        assert refused(lambda: with_prereg(DRAFT, lambda: jl.job_body(941, HEAD))), \
            "no K1 job on a draft (planted, so this survives the K1 freeze)"
        assert jl.cells(941) == [(a, 941) for a in jc.J24K1_ONLY]
        b = with_prereg(FROZEN, lambda: jl.job_body(941, HEAD))
        ast.parse(b)
        assert repr([(a, 941) for a in jc.J24K1_ONLY]) in b and "'j24k1'" in b
        assert "K-bins16" not in b.replace("K1-bins16", "")
        assert jl.raw_file(941) == Path("/data/jack-logs/j24k1/chains/j24k1_s941.json")
        assert jl.seg_file(941, 1).parent == Path("/data/jack-logs/j24k1/chains/segments")
        assert refused(lambda: jl.set_variant("nope"))
    finally:
        jl.set_variant("j24")
    assert (jl.PREREG, jl.ARMS, jl.OUT, jl.SEGS, jl.TAG, jl.TIMING_FILE) == j24
    assert jl.raw_file(941) == Path("/data/jack-logs/j24/chains/j24_s941.json")
    b = with_prereg(FROZEN, lambda: jl.job_body(941, HEAD))
    assert "'j24'" in b and "K1-" not in b


B20_DRAFT_SHA = "e797d367"   # real history: pilot_b20.md drafted, NOT frozen


def test_b20_variant_packs_25_cells_at_1m_and_refuses_a_draft():
    """J32: --variant b20 reads pilot_b20.md (DRAFT at its own drafting commit
    and as a plant -> no job), packs sighted 2001-2020 + blind 2001-2005 three a
    kernel into 9 units, runs 1M decisions, own folder; back to j24 restores."""
    from experiments.pilot_b2 import PILOT_B20_ARMS
    j24 = (jl.PREREG, jl.ARMS, jl.OUT, jl.SEGS, jl.TAG, jl.SEEDS, jl.PACK, jl.RUN_T)
    try:
        jl.set_variant("b20")
        assert jl.PREREG == "docs/research/prereg/pilot_b20.md"
        assert set(jc.B20_ONLY) == set(PILOT_B20_ARMS) and set(jc.B20_ONLY) <= set(jc.CARRIED_ARMS)
        draft_pin = subprocess.run(["git", "-C", str(REPO), "rev-parse", B20_DRAFT_SHA],
                                   capture_output=True, text=True).stdout.strip()
        assert not jl.frozen(jl.prereg_at(draft_pin)), "pilot_b20 was drafted as a DRAFT"
        assert refused(lambda: jl.job_body(1, draft_pin)), "no b20 job on the real draft"
        assert refused(lambda: with_prereg(DRAFT, lambda: jl.job_body(1, HEAD))), \
            "no b20 job on a planted draft"
        assert jl.SEEDS == tuple(range(1, 10))
        flat = [c for u in jl.SEEDS for c in jl.cells(u)]
        assert [len(jl.cells(u)) for u in jl.SEEDS] == [3] * 8 + [1]
        assert sorted(s for a, s in flat if a == "B20-bins16") == list(range(2001, 2021))
        assert sorted(s for a, s in flat if a == "B20-bins16-blind") == list(range(2001, 2006))
        assert len(set(flat)) == 25, "no cell twice"
        assert refused(lambda: jl.cells(10)) and refused(lambda: jl.cells(2001))
        b = with_prereg(FROZEN, lambda: jl.job_body(7, HEAD))
        ast.parse(b)
        assert repr(jl.cells(7)) in b and "'b20'" in b and "1000000" in b and "500000" not in b
        assert jl.raw_file(9) == Path("/data/jack-logs/pilot_b20/chains/b20_s9.json")
        assert jl.seg_file(9, 1) == Path("/data/jack-logs/pilot_b20/chains/segments/b20_s9_seg1.json")
    finally:
        jl.set_variant("j24")
    assert (jl.PREREG, jl.ARMS, jl.OUT, jl.SEGS, jl.TAG, jl.SEEDS, jl.PACK, jl.RUN_T) == j24
    assert jl.cells(941) == [(a, 941) for a in jc.J24_ONLY]


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
    print("ALL PASS")
