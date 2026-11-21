"""J14 (a): the ear row's judge, on planted runs with known answers (11-18, before any seed).

Planted: seeds 96.5/96.9/97.1 (mean 96.83, in band) + twin 9.0 -> PASS.
Controls that must NOT pass: the same seeds with the twin at 15.5 -> VOID; seeds at
95.0/95.5/95.9 -> MISS with extension_due; a seed that never printed `test acc` or
saw the wrong data counts -> NOT JUDGED (never a zero); a twin whose labels were
not actually permuted -> not complete. Band pinned to the paper: 96.9 +- 2 x 0.30."""
from experiments import kws_row as K


def run(seed, test, twin=False, **kw):
    r = {"name": K.run_name(seed, twin), "seed": seed, "twin": twin, "test": test, "rc": 0,
         "cut": False, "stopped": None, "counts": list(K.EXPECT_COUNTS), "valid": [90.0] * K.EPOCHS,
         "shuffled": (33000, 36923) if twin else None}
    r.update(kw)
    return r


def test_band_and_pass():
    assert K.BAND == (96.3, 97.5)
    j = K.judge([run(0, 96.5), run(1, 96.9), run(2, 97.1), run(0, 9.0, twin=True)])
    assert j["verdict"].startswith("PASS"), j


def test_controls_fail():
    j = K.judge([run(0, 96.5), run(1, 96.9), run(2, 97.1), run(0, 15.5, twin=True)])
    assert j["verdict"].startswith("VOID"), j
    j = K.judge([run(0, 95.0), run(1, 95.5), run(2, 95.9), run(0, 9.0, twin=True)])
    assert j["verdict"] == "MISS" and j["extension_due"], j
    j = K.judge([run(0, 95.0), run(1, 95.5), run(2, 95.9), run(3, 99.0), run(4, 99.0), run(0, 9.0, twin=True)])
    assert j["n"] == 5 and not j["extension_due"], j          # judged once more at 5, nothing after
    j = K.judge([run(0, 96.5), run(1, None, rc=1), run(2, 97.1), run(0, 9.0, twin=True)])
    assert j["verdict"] == "NOT JUDGED" and j["missing"] == [1], j
    j = K.judge([run(0, 96.5), run(1, 96.9, counts=[84843, 9981, 11005]), run(2, 97.1), run(0, 9.0, twin=True)])
    assert j["verdict"] == "NOT JUDGED", j
    j = K.judge([run(0, 96.5), run(1, 96.9), run(2, 97.1), run(0, 9.0, twin=True, shuffled=(0, 36923))])
    assert j["verdict"] == "NOT JUDGED" and "twin" in j["missing"], j
    j = K.judge([run(0, 96.5), run(1, 96.9), run(2, 97.1), run(0, 9.0, twin=True, valid=[1.0] * 120)])
    assert j["verdict"] == "NOT JUDGED", j                    # a run short of 200 epochs never scores


def test_plan_and_body():
    assert K.plan_from_args(["s0", "s1", "s2", "twin"]) == [(0, False), (1, False), (2, False), (0, True)]
    try:
        K.plan_from_args(["s0", "bogus"])
        raise AssertionError("bad plan accepted")
    except ValueError:
        pass
    b = K.kernel_body(["s0", "twin"], "kws_row_s0_twin")
    assert "['s0', 'twin']" in b and "kws_row_s0_twin" in b and "run_kernel" in b
    compile(b, "kernel", "exec")


def _torchaudio_block():
    b = K._KERNEL_BODY
    return b[b.index("import importlib.metadata as _md"):b.index("import importlib.metadata as md, platform")]


def _exec_block(versions, probe_ok_before, probe_ok_after, installs_to=None, repin_ok=False):
    """Run the kernel's torchaudio guard with pip, the import probe and package metadata faked."""
    import importlib.metadata as real_md, subprocess as real_sp, types
    calls, state, calls_env = [], {"installed": False}, []
    def version(name):
        if name == "torchaudio" and state["installed"] and installs_to:
            return installs_to
        if versions.get(name) is None:
            raise real_md.PackageNotFoundError(name)
        return versions[name]
    def run(cmd, **kw):
        calls.append(cmd)
        if "pip" in cmd and "torch==2.6.0" in cmd:
            calls_env.append(kw.get("env"))
            if repin_ok:
                versions.update(torch="2.6.0+cu126", torchaudio="2.6.0+cu126")
                state["installed"], state["repinned"] = False, True
            return types.SimpleNamespace(returncode=0 if repin_ok else 1, stderr="" if repin_ok else "no wheel", stdout="")
        if "pip" in cmd:
            state["installed"] = True
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        if any("get_arch_list" in str(x) for x in cmd):       # the re-pin's arch check
            return types.SimpleNamespace(returncode=0, stderr="", stdout="EAR_TORCH 2.6.0 ['sm_60'] (6, 0)")
        ok = True if state.get("repinned") else probe_ok_after if state["installed"] else probe_ok_before
        return types.SimpleNamespace(returncode=0 if ok else 1, stderr="" if ok else "OSError: undefined symbol", stdout="")
    old = real_md.version
    real_md.version = version
    try:
        g = {"subprocess": types.SimpleNamespace(run=run), "sys": __import__("sys")}
        try:
            exec(_torchaudio_block(), g)
            return calls, None
        except SystemExit as e:
            return calls, str(e)
    finally:
        real_md.version = old


def test_torchaudio_matches_the_torch_that_runs():
    pips = lambda calls: [c for c in calls if "pip" in c]
    # matching pair already present: no install, no stop
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.5.1+cu121"}, True, True)
    assert err is None and not pips(calls)
    # PLANTED: the image's 2.8.0 torchaudio beside the preamble's 2.5.1 (imports OR not) -> reinstall 2.5.1 from cu121, no deps
    for ok in (True, False):
        calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.8.0+cu126"}, ok, True, installs_to="2.5.1+cu121")
        assert err is None, err
        (c,) = pips(calls)
        assert "torchaudio==2.5.1" in c and "--no-deps" in c and "https://download.pytorch.org/whl/cu121" in c
        assert not any(x.startswith("torch==") for x in c)
    # missing torchaudio -> installed
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": None}, False, True, installs_to="2.5.1+cu121")
    assert err is None and len(pips(calls)) == 1
    # PLANTED: the install does not take -> the kernel STOPS on the box's first line, never trains on a mismatch
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.8.0"}, False, False, installs_to="2.8.0")
    assert err and err.startswith(("TORCHAUDIO_MISMATCH", "TORCH_REPIN_FAILED")), err
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.8.0"}, True, False, installs_to="2.5.1")
    assert err and err.startswith(("TORCHAUDIO_MISMATCH", "TORCH_REPIN_FAILED")), err
    # every failed first install falls back to the 2.6.0+cu126 trio, outside PIP_CONSTRAINT
    assert any("torch==2.6.0" in c and "torchaudio==2.6.0" in c and "https://download.pytorch.org/whl/cu126" in c
               for c in pips(calls))
    # PLANTED 11-21 (kernel jack-ladder-1795228077): no torchaudio 2.5.1 for cp313 -> re-pin the trio, run on 2.6.0
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.11.0"}, False, False, installs_to="2.11.0",
                             repin_ok=True)
    assert err is None, err
    assert len(pips(calls)) == 2 and "torch==2.6.0" in pips(calls)[1]
    # PLANTED: the re-pin itself fails -> stop, never train
    calls, err = _exec_block({"torch": "2.5.1+cu121", "torchaudio": "2.11.0"}, False, False, installs_to="2.11.0")
    assert err and (err.startswith("TORCH_REPIN_FAILED") or err.startswith("TORCHAUDIO_MISMATCH")), err


def main():
    for t in (test_band_and_pass, test_controls_fail, test_plan_and_body, test_torchaudio_matches_the_torch_that_runs):
        t()
        print(f"{t.__name__}: PASS")
    print("test_kws_row PASS")


if __name__ == "__main__":
    main()
