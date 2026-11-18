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


def main():
    for t in (test_band_and_pass, test_controls_fail, test_plan_and_body):
        t()
        print(f"{t.__name__}: PASS")
    print("test_kws_row PASS")


if __name__ == "__main__":
    main()
