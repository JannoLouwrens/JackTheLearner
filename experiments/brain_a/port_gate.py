"""MLX port, the gradient gate (PORT report rung 10) — compared PER TENSOR, against each tensor's own size.

Source of the rule: PUBLISHED_EVIDENCE_2026-11-15_PORT.md, ladder rung 10: "Gradients per loss term —
relative 1e-4 (1e-3 for recurrent parts); cosine >= 0.9999". The report does not say relative TO WHAT.
0.4d iii (README) found why it must be each tensor's own norm: on the official `debug` agent the
actor-mean kernel gradient is 3.5e-11 while the largest gradient is 6.6e+1 (ratio 5e-13, below float32
precision). Measured against the whole gradient vector, a port whose logp-by-mean backward is WRONG —
even sign-flipped — moves the global error by ~1e-12 and passes. This file is the gate that cannot be
fooled that way, with the planted cases that prove it (`--selftest`).

Per tensor g (golden) and p (port), flattened, float64:
    rel = ||p - g|| / ||g||,   cos = <p, g> / (||p|| ||g||)
    PASS iff rel <= rtol (1e-3 for names under `dyn/` = the recurrent model, 1e-4 otherwise) AND cos >= 0.9999.
    ||g|| == 0: PASS only if p is exactly 0 too, and the tensor is counted UNCHECKED (it carries no signal;
    `g1_probe`'s zero guard refuses a golden file where a gradient is zero in every dumped call).
Keys compared: every `grad/` array in the golden dump (multi-call keys `c<N>/grad/...`); a key missing
from the port dump FAILS. `outs/` arrays are not this rung's (rungs 7-9) and are not compared here.
Not yet this rung as written: per LOSS TERM gradients (the dump holds the total loss's gradient; per-term
needs the loss split inside `Agent.loss` — listed for the port, not built).
The thresholds are the report's (themselves inference from MLX's per-op float32 tests, loosened for
depth); calibrating them against a real MLX run is the port's job, not this file's.

    python -m experiments.brain_a.port_gate GOLDEN.npz PORT.npz      # rc 0 PASS, 1 FAIL
    python -m experiments.brain_a.port_gate --selftest GOLDEN.npz [CONTROL.npz]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

RTOL, RTOL_REC, COS_MIN = 1e-4, 1e-3, 0.9999


def _name(key):
    """`c8/grad/dyn/x` or `grad/dyn/x` -> `dyn/x`; None if not a gradient."""
    parts = key.split("/")
    if parts[0].startswith("c") and parts[0][1:].isdigit():
        parts = parts[1:]
    return "/".join(parts[1:]) if parts and parts[0] == "grad" else None


def gate(golden, port, verbose=False):
    """golden/port: mapping key -> array. Returns (passed, rows, unchecked) with one row per failing key."""
    fails, unchecked, n = [], [], 0
    for k in sorted(golden):
        name = _name(k)
        if name is None:
            continue
        n += 1
        g = np.asarray(golden[k], np.float64).ravel()
        if k not in port:
            fails.append((k, "missing", float("nan"), float("nan")))
            continue
        p = np.asarray(port[k], np.float64).ravel()
        if p.shape != g.shape:
            fails.append((k, "shape", float("nan"), float("nan")))
            continue
        ng = np.linalg.norm(g)
        if ng == 0.0:
            unchecked.append(k)
            if np.any(p):
                fails.append((k, "golden-zero", float("inf"), float("nan")))
            continue
        npn = np.linalg.norm(p)
        rel = np.linalg.norm(p - g) / ng
        cos = float(p @ g / (npn * ng)) if npn > 0 else 0.0
        tol = RTOL_REC if name.startswith("dyn/") else RTOL
        if not (rel <= tol and cos >= COS_MIN):
            fails.append((k, f"rel>{tol:g}" if rel > tol else "cos", float(rel), cos))
    return (not fails), fails, unchecked, n


def global_rel(golden, port):
    """The fooled comparison, for contrast: one relative error over all gradients concatenated."""
    ks = [k for k in sorted(golden) if _name(k) is not None and k in port]
    g = np.concatenate([np.asarray(golden[k], np.float64).ravel() for k in ks])
    p = np.concatenate([np.asarray(port[k], np.float64).ravel() for k in ks])
    return float(np.linalg.norm(p - g) / np.linalg.norm(g))


def report(label, golden, port):
    ok, fails, unchecked, n = gate(golden, port)
    print(f"PORT_GATE {label}: {'PASS' if ok else 'FAIL'} tensors {n} failing {len(fails)} "
          f"unchecked(golden zero) {len(unchecked)} | global-rel {global_rel(golden, port):.3e} "
          f"(a global 1e-4 gate would say {'PASS' if global_rel(golden, port) <= RTOL else 'FAIL'})")
    for k, why, rel, cos in fails[:8]:
        print(f"   FAIL {k} {why} rel {rel:.3e} cos {cos:.6f}")
    if len(fails) > 8:
        print(f"   ... {len(fails) - 8} more")
    return ok


def selftest(golden_path, control_path=None, seed=0):
    """Known-positives through the real gate. Each case states the verdict it MUST give; rc 0 iff all do."""
    G = dict(np.load(golden_path))
    rng = np.random.default_rng(seed)
    grads = [k for k in G if _name(k) is not None]
    means = [k for k in grads if _name(k).startswith("pol/head/action/mean/")]
    cases = []
    cases.append(("identical", G, True))
    # float32 round-off: every element x(1 + 2^-23 * N(0,1)) — an honest port must not fail on this
    cases.append(("float32-roundoff", {k: (v * (1 + rng.standard_normal(v.shape) * 2.0 ** -23)).astype(v.dtype)
                                       if k in grads else v for k, v in G.items()}, True))
    # the 0.4d iii hole: the actor-mean backward sign-flipped in every call, nothing else touched
    cases.append(("actor-mean-sign-flip", {k: (-v if k in means else v) for k, v in G.items()}, False))
    # a 1% scale error on the same tensors (a wrong constant in logp's mean term)
    cases.append(("actor-mean-scale-1.01", {k: (v * 1.01 if k in means else v) for k, v in G.items()}, False))
    # a recurrent tensor 0.5% off: inside 1e-2, outside dyn's 1e-3 -> must FAIL
    rec = sorted(k for k in grads if _name(k) == "dyn/dyngru/kernel")
    cases.append(("dyngru-0.5pct", {k: (v * 1.005 if k in rec else v) for k, v in G.items()}, False))
    # a recurrent tensor 0.02% off: inside dyn's 1e-3 -> must PASS (the looser band is real)
    cases.append(("dyngru-0.02pct", {k: (v * 1.0002 if k in rec else v) for k, v in G.items()}, True))
    # one tensor dropped from the port dump
    cases.append(("missing-key", {k: v for k, v in G.items() if k != grads[0]}, False))
    if control_path:
        cases.append(("control-other-noise", dict(np.load(control_path)), False))
    bad = 0
    for label, P, want in cases:
        got = report(label, G, P)
        flag = "as required" if got == want else "WRONG"
        bad += got != want
        print(f"SELFTEST {label}: must {'PASS' if want else 'FAIL'} -> {'PASS' if got else 'FAIL'} {flag}")
    print(f"SELFTEST {'OK' if not bad else 'BROKEN'} {len(cases) - bad}/{len(cases)} "
          f"(actor-mean tensors {len(means)}, dyngru tensors {len(rec)})")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("port", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.golden, a.port)
    return 0 if report("compare", dict(np.load(a.golden)), dict(np.load(a.port))) else 1


if __name__ == "__main__":
    sys.exit(main())
