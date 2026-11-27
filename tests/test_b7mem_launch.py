"""J29 (5): B7mem's Kaggle chain launcher (pure parts; no kernel is submitted).

  freeze    no job at c848d888^ (row B-7 still a proposal); HEAD (frozen) builds one
            that pins the row's sha.
  refusals  short pin, seed outside 0-3, segment out of range.
  job       compiles; segment 0 starts fresh, segment n copies the ONE mounted carry.
  kernel    fake submit: a cut segment carries on and the next one mounts it; a
            complete segment writes the raw file once; a stalled or crashed segment
            ends the seed as LOST (never re-bought); a second launch of a home seed
            refuses.
  chain     segments back to back; a guard error is waited out, a lost seed reported.

    /data/venvs/jack-std/bin/python tests/test_b7mem_launch.py
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import b7mem_launch as bl  # noqa: E402


def sha(ref):
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", ref],
                          capture_output=True, text=True).stdout.strip()


HEAD, PRE_FREEZE = sha("HEAD"), sha("c848d888^")


def refused(f):
    try:
        f()
    except bl.Refused:
        return True
    return False


def test_freeze():
    assert not bl.frozen(bl.row_at(PRE_FREEZE)), "B-7 was a proposal before c848d888"
    assert bl.row_at(PRE_FREEZE) is not None, "the row existed (as a draft) before the freeze"
    assert refused(lambda: bl.job_body(0, PRE_FREEZE))
    assert bl.frozen(bl.row_at(HEAD))
    import hashlib
    body = bl.job_body(0, HEAD)
    assert hashlib.sha256(bl.row_at(HEAD).encode()).hexdigest() in body


def test_refusals():
    assert refused(lambda: bl.job_body(0, HEAD[:10]))
    assert refused(lambda: bl.job_body(4, HEAD))
    assert refused(lambda: bl.job_body(0, HEAD, segment=bl.MAX_SEGMENTS))
    assert refused(lambda: bl.job_body(0, HEAD, segment=-1))


def test_job():
    b0, b1 = bl.job_body(2, HEAD, 0), bl.job_body(2, HEAD, 1)
    ast.parse(b0), ast.parse(b1)
    assert "if 0:" in b0 and "if 1:" in b1          # carry copy only in a continuation
    for b in (b0, b1):
        assert "'b7mem_s2.carry'" in b and '"--threads", str(4)' in b
        assert repr(HEAD) in b and "--deadline" in b and "'gymnasium==1.4.0'" in b


def _fake(steps, complete=False, rc=0, job="u/jack-cpu-1"):
    """A submit that 'ran' a segment ending at `steps`."""
    calls = []

    def submit(body, **k):
        calls.append(k)
        d = Path(tempfile.mkdtemp())
        rec = {"arm": "B7mem", "seed": 0, "complete": complete,
               "evals": [{"step": s, "ret": -1000.0} for s in range(2000, steps + 1, 1000)]}
        (d / bl.SUMMARY).write_text(json.dumps(
            {"seed": 0, "segment": 0, "pin": HEAD, "row_sha256": "x", "rc": rc, "record": rec,
             "versions": {"torch": "t"}}))
        return SimpleNamespace(artifacts={bl.SUMMARY: str(d / bl.SUMMARY)}, job_id=job,
                               duration_s=27000.0, message="ok")
    return submit, calls


def _tmp_out():
    d = Path(tempfile.mkdtemp())
    bl.OUT, bl.SEGS = d, d / "segments"
    return d


def test_kernel_carry_then_complete():
    _tmp_out()
    s0, c0 = _fake(20000, job="u/k0")
    r0 = bl.kernel(0, HEAD, 0, submit_fn=s0)
    assert r0["next_segment"] == 1 and c0[0]["kernel_sources"] is None
    assert c0[0]["prefer"] == "kaggle-cpu" and not bl.raw_file(0).exists()
    assert refused(lambda: bl.kernel(0, HEAD, 2, submit_fn=s0))        # needs seg 1 first
    assert refused(lambda: bl.kernel(0, "f" * 40, 1, submit_fn=s0))    # other pin
    s1, c1 = _fake(51000, complete=True, job="u/k1")
    raw = bl.kernel(0, HEAD, 1, submit_fn=s1)
    assert c1[0]["kernel_sources"] == ["u/k0"]
    assert raw["complete"] and raw["segments"] == 2 and "lost" not in raw
    assert raw["kernel"]["job_id"] == ["u/k0", "u/k1"]
    assert json.loads(bl.raw_file(0).read_text())["evals"][-1]["step"] == 51000
    assert refused(lambda: bl.kernel(0, HEAD, 2, submit_fn=s1))        # a seed runs once


def test_kernel_stall_and_crash_are_lost():
    _tmp_out()
    bl.kernel(0, HEAD, 0, submit_fn=_fake(20000, job="u/k0")[0])
    raw = bl.kernel(0, HEAD, 1, submit_fn=_fake(20000, job="u/k1")[0])  # did not move
    assert not raw["complete"] and raw["lost"].startswith("stalled")
    _tmp_out()
    raw = bl.kernel(0, HEAD, 0, submit_fn=_fake(20000, rc=1)[0])        # crashed
    assert raw.get("lost") and "next_segment" not in raw


def test_kernel_last_segment_incomplete_is_lost():
    _tmp_out()
    for i in range(bl.MAX_SEGMENTS - 1):
        r = bl.kernel(0, HEAD, i, submit_fn=_fake(10000 * (i + 1), job=f"u/k{i}")[0])
        assert r["next_segment"] == i + 1
    raw = bl.kernel(0, HEAD, bl.MAX_SEGMENTS - 1,
                    submit_fn=_fake(10000 * bl.MAX_SEGMENTS, job="u/kl")[0])
    assert raw["lost"].startswith("not complete")


def test_chain():
    seen, slept = [], []

    def kfn(seed, pin, seg):
        seen.append((seed, seg))
        if seed == 1 and seg == 0 and slept == []:
            raise RuntimeError("Uncommitted changes in tree")
        if seed == 2:
            return {"lost": "stalled"}
        return {"next_segment": 1} if seg == 0 else {"complete": True}
    done = bl.chain([0, 1, 2], HEAD, sleep=slept.append, kernel_fn=kfn)
    assert done == {0: "home", 1: "home", 2: "lost: stalled"}
    assert seen == [(0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (2, 0)] and slept == [bl.GUARD_WAIT_S]


if __name__ == "__main__":
    fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            try:
                f()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    print(f"{sum(n.startswith('test_') for n in globals()) - fails}/"
          f"{sum(n.startswith('test_') for n in globals())}")
    sys.exit(1 if fails else 0)
