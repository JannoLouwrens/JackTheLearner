"""Ruling 135: a second chain may take round 1-L groups that a RUNNING chain
also lists, only if no group can ever run twice (check #13: one run a seed).

The claim is the group's segment-0 file, written O_EXCL before any kernel.
Chains A/B were started at e99c1b78 and are not restarted; their own code
refuses a group whose segment-0 file exists, so they honour a claim as is.
Planted: a claimed name is skipped by A/B's own code (loaded from git) and by
an unowned chain; an unclaimed one is not; claims are all-or-none; an owner
runs only what it claimed; a claim is never taken for a home segment 0.

    /data/venvs/jackthelearner/bin/python tests/test_forge_r1_claim.py
"""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.forge import r1_run as rr  # noqa: E402

PIN = "4" * 40
A_B_CODE = "e99c1b78"          # the commit chains A/B run (started 11-26 20:30)


def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp(dir="/data/tmp" if Path("/data/tmp").is_dir() else None))


def refused(f, mod=rr):
    try:
        f()
    except mod.Refused:
        return True
    return False


def _fake_gpu(script):
    calls = []

    class Sub:
        ok, message, charge_seconds = True, "", 0

    def submit(job, **kw):
        calls.append(kw)
        sub = Sub()
        sub.job_id, sub.duration_s = f"u/k{len(calls) - 1}", 1000.0
        sub.artifacts = {"forge_r1L_cells.json": json.dumps(script.pop(0))}
        return sub
    return types.SimpleNamespace(build_job=lambda b: b, submit=submit,
                                 result_json=lambda s, n: json.loads(s.artifacts[n])), calls


def _payload(name, seg, cut):
    cells = {}
    r1L = rr.ROUNDS / "r1L" / "spec.json"
    for a, sd in rr.groups(rr.load_spec(r1L), "L")[name]:
        cells[f"{a}_s{sd}"] = {"arm": a, "seed": sd, "lives": [1],
                               **({"cut": True, "resumable": True} if cut else {})}
    return {"round": "r1L", "rung": "L", "pin": PIN, "spec_sha256": rr.spec_sha256(r1L),
            "segment": seg, "versions": {}, "cells": cells}


def in_r1L(f, mod=rr, script=()):
    """Run f(dir, calls) with `mod` on round r1L, raw paths in a temp dir, a
    fake launcher and every pin accepted; restore everything after."""
    keep = (mod.ROUND, mod.SPEC_PATH, mod.OUT, mod.SEGS, mod.pin_has_rule)
    real_gpu = sys.modules.get("experiments.gpu")
    fake, calls = _fake_gpu(list(script))
    sys.modules["experiments.gpu"] = fake
    d = _tmp()
    try:
        mod.set_round("r1L")
        mod.OUT, mod.SEGS, mod.pin_has_rule = d, d / "segments", lambda pin: True
        return f(d, calls)
    finally:
        mod.ROUND, mod.SPEC_PATH, mod.OUT, mod.SEGS, mod.pin_has_rule = keep
        if real_gpu is not None:
            sys.modules["experiments.gpu"] = real_gpu
        else:
            sys.modules.pop("experiments.gpu", None)


def _old_module():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{A_B_CODE}:experiments/forge/r1_run.py"],
                         capture_output=True, text=True, check=True).stdout
    p = _tmp() / "r1_run_ab.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("r1_run_ab", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROUNDS = rr.ROUNDS           # the frozen rules live in the repo, not the temp copy
    return m


def test_running_chains_code_skips_a_claimed_name_and_runs_an_unclaimed_one():
    old = _old_module()

    def go(d, calls):
        rr.SEGS, rr.OUT = old.SEGS, old.OUT
        rr.claim(["g1-4", "g1-6"], "L", "C", PIN)
        done = old.chain(["g1-4", "g1-6", "g1-7"], "L", PIN, sleep=lambda s: None)
        assert done["g1-4"].startswith("refused") and done["g1-6"].startswith("refused")
        assert done["g1-7"] == "home", "an unclaimed name still runs"
        assert len(calls) == 1, "A/B submit nothing for a claimed name"
        assert json.loads(rr.seg_file("L", "g1-4", 0).read_text())["claimed_by"] == "C"
    keep = (rr.ROUND, rr.SPEC_PATH, rr.OUT, rr.SEGS)
    try:
        rr.set_round("r1L")
        in_r1L(go, old, script=[_payload("g1-7", 0, cut=False)])
    finally:
        rr.ROUND, rr.SPEC_PATH, rr.OUT, rr.SEGS = keep


def test_an_unowned_chain_skips_a_claim_too():
    def go(d, calls):
        rr.claim(["g1-5"], "L", "D", PIN)
        assert refused(lambda: rr.kernel("g1-5", "L", PIN))
        assert refused(lambda: rr.kernel("g1-5", "L", PIN, owner="C")), "another owner"
        assert calls == []
    in_r1L(go)


def test_claims_are_all_or_none_and_never_over_a_home_or_running_group():
    def go(d, calls):
        rr.SEGS.mkdir(parents=True)
        rr.seg_file("L", "g1-6", 0).write_text(json.dumps(_payload("g1-6", 0, True)))
        assert refused(lambda: rr.claim(["g1-4", "g1-6"], "L", "C", PIN))
        assert not rr.seg_file("L", "g1-4", 0).exists(), "a refused claim writes nothing"
        (d / "L_g1-5.json").write_text("{}")
        assert refused(lambda: rr.claim(["g1-5"], "L", "D", PIN)), "raw file home"
        rr.claim(["g1-4"], "L", "C", PIN)
        assert refused(lambda: rr.claim(["g1-4"], "L", "C", PIN)), "claimed once"
        assert refused(lambda: rr.claim(["g1-9"], "L", "C", PIN)), "not a group"
        assert refused(lambda: rr.claim(["g1-7", "g1-7"], "L", "C", PIN))
        assert refused(lambda: rr.claim(["g1-7"], "L", "../x", PIN))
        assert not rr.seg_file("L", "g1-7", 0).exists()
    in_r1L(go)


def test_a_lost_race_unclaims_what_it_made():
    import os
    real = os.open

    def go(d, calls):
        def racing(path, flags, mode=0o777):
            if str(path).endswith("L_g1-6_seg0.json") and not Path(path).exists():
                # the other claimer writes g1-6 between our check and our create
                Path(path).write_text(json.dumps({"claimed_by": "D", "name": "g1-6"}))
            return real(path, flags, mode)
        os.open = racing
        try:
            assert refused(lambda: rr.claim(["g1-4", "g1-6"], "L", "C", PIN))
        finally:
            os.open = real
        assert not rr.seg_file("L", "g1-4", 0).exists()
        assert json.loads(rr.seg_file("L", "g1-6", 0).read_text())["claimed_by"] == "D", \
            "the winner's claim is never overwritten"
    in_r1L(go)


def test_owner_runs_only_its_claim_and_its_payload_replaces_the_claim():
    def go(d, calls):
        assert refused(lambda: rr.kernel("g1-4", "L", PIN, owner="C")), "unclaimed"
        rr.claim(["g1-4"], "L", "C", PIN)
        assert refused(lambda: rr.kernel("g1-4", "L", PIN, segment=1, owner="C")), \
            "segment 1 never mounts a claim"
        r0 = rr.kernel("g1-4", "L", PIN, owner="C")
        assert r0["next_segment"] == 1 and calls[0]["kernel_sources"] is None
        assert not rr.is_claim(json.loads(rr.seg_file("L", "g1-4", 0).read_text()))
        assert refused(lambda: rr.kernel("g1-4", "L", PIN, owner="C")), "segment 0 home now"
        raw = rr.kernel("g1-4", "L", PIN, segment=1, owner="C")
        assert calls[1]["kernel_sources"] == ["u/k0"]
        assert [p.name for p in d.glob("L_g*.json")] == ["L_g1-4.json"]
        assert len(raw["cells"]) == 4 and "claimed_by" not in raw
    in_r1L(go, script=[_payload("g1-4", 0, True), _payload("g1-4", 1, False)])


def test_chain_passes_its_owner_and_a_rung_without_carries_cannot_be_claimed():
    seen = []
    real = rr.kernel

    def fake(name, rung, pin, climbers=None, segment=0, owner=None):
        seen.append((name, segment, owner))
        return {"cells": {}}
    rr.kernel = fake
    try:
        rr.chain(["g1-4", "g1-6"], "L", PIN, sleep=lambda s: None, owner="C")
        rr.chain(["g1-5"], "L", PIN, sleep=lambda s: None)
    finally:
        rr.kernel = real
    assert seen == [("g1-4", 0, "C"), ("g1-6", 0, "C"), ("g1-5", 0, None)]
    assert refused(lambda: rr.claim(["g0"], "S", "C", PIN)), "rung S never carries"


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n, flush=True)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e), flush=True)
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
