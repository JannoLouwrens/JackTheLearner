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
    assert not jl.frozen(jl.prereg_at(HEAD)), "today's prereg is a draft"
    assert jl.frozen(FROZEN)
    assert not jl.frozen(b"**STATUS: DRAFT, NOT FROZEN**\n**STATUS: FROZEN x**\n"), \
        "only the first STATUS line counts"


def test_no_job_on_a_draft():
    assert refused(lambda: jl.job_body(941, HEAD)), "HEAD's prereg is not frozen"


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


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
    print("ALL PASS")
