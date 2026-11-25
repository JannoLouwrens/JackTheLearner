"""MLX port of DreamerV3 — rung 12, first part (ORDERS J25): the return normaliser and advantage,
references only (the actor heads' logp / entropy are rung 12's second part).

Official code `danijar/dreamerv3` at e01491f:
  `configs.yaml` retnorm {impl perc, rate 0.01, limit 1.0, perclo 5, perchi 95, debias False};
      advnorm {impl none}.
  `embodied/jax/utils.py` Normalize: __call__(x, update) UPDATES FIRST, then returns stats();
      perc: lo <- (1-rate) lo + rate * percentile(x, 5); hi likewise at 95 (jnp.percentile, linear);
      state starts at 0; stats = (lo, max(limit, hi - lo)); debias off -> corr 1.
  `agent.py imag_loss` L407-411: roffset, rscale = retnorm(ret, update);
      adv = (ret - tarval[:, :-1]) / rscale; advnorm none -> adv_normed = adv. (roffset is used only
      for a metric.)

What is compared: NumPy float64 vs the official Normalize run for real (f32), over 6 successive
updates on [32, 15] returns whose spread goes 1 -> 1000 -> 5 (so the limit branch is taken both
ways: early steps sit on limit 1.0, later ones on hi - lo); and adv on each step. Planted wrong
references (debias on, nearest percentile, stats read before the update, rate 0.1, 10/90
percentiles, offset subtracted in adv) must each be CAUGHT.
"Ours beside theirs": f64 vs f32; one device (no all_gather); returns synthetic seeded, not
imagined (rung 11 already pins ret against the official lambda_return).
Tolerance as rung 11: |d|/(|ref|+1) <= 1e-4.

  nice -n 19 /data/venvs/jack-dv3/bin/python -I experiments/brain_a/rung12a.py --vs-official
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL  # noqa: E402

RATE, LIMIT, LO, HI = 0.01, 1.0, 5.0, 95.0
SCALES = (1.0, 50.0, 300.0, 300.0, 1000.0, 5.0)
N, T = 32, 15
TOL = 1e-4
GOLDEN12A = os.path.join(HERE, "rung12a_retnorm_golden_e01491f.npz")


def inputs(seed=12):
    r = np.random.default_rng(seed)
    rets = [np.float32(s * r.standard_normal((N, T)) + 0.3 * s) for s in SCALES]
    tars = [np.float32(s * r.standard_normal((N, T + 1))) for s in SCALES]
    return rets, tars


def REF_retnorm(rets, tars, rate=RATE, limit=LIMIT, lo_q=LO, hi_q=HI, debias=False, method="linear",
                stats_first=False, adv_offset=False):
    """Float64 perc Normalize over successive updates -> per step (offset, scale, adv)."""
    lo = hi = corr = 0.0
    out = []
    for ret, tar in zip(rets, tars):
        x = np.float64(ret)

        def stats():
            c = 1.0 / max(rate, corr) if debias else 1.0
            return lo * c, max(limit, hi * c - lo * c)
        if stats_first:
            off, sc = stats()
        lo = (1 - rate) * lo + rate * np.percentile(x, lo_q, method=method)
        hi = (1 - rate) * hi + rate * np.percentile(x, hi_q, method=method)
        corr = (1 - rate) * corr + rate * 1.0
        if not stats_first:
            off, sc = stats()
        adv = (x - np.float64(tar)[:, :-1] - (off if adv_offset else 0.0)) / sc
        out.append((off, sc, adv))
    return out


PLANTED = {
    "debias on": dict(debias=True),
    "nearest percentile": dict(method="nearest"),
    "stats read before the update": dict(stats_first=True),
    "rate 0.1": dict(rate=0.1),
    "10/90 percentiles": dict(lo_q=10.0, hi_q=90.0),
    "offset subtracted in adv": dict(adv_offset=True),
}


def match(got, ref):
    worst = 0.0
    for (go, gs, ga), (ro, rs, ra) in zip(got, ref):
        for g, r in ((go, ro), (gs, rs), (ga, ra)):
            g, r = np.float64(g), np.float64(r)
            worst = max(worst, float((np.abs(g - r) / (np.abs(r) + 1)).max()))
    return worst <= TOL, worst


def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax
    rets, tars = inputs()
    norm = embodied.jax.Normalize(impl="perc", rate=RATE, limit=LIMIT, perclo=LO, perchi=HI,
                                  debias=False, name="retnorm")

    def step(ret, tar):
        off, sc = norm(ret, True)
        return off, sc, (ret - tar[:, :-1]) / sc

    state, got = {}, []
    with jax.disable_jit():
        for i, (ret, tar) in enumerate(zip(rets, tars)):
            state, o = nj.pure(step)(state, jnp.asarray(ret), jnp.asarray(tar), create=(i == 0),
                                     seed=jax.random.PRNGKey(0))
            got.append(tuple(np.asarray(v) for v in o))
    rows = []

    def put(ok, d):
        rows.append(ok); print(("  ok  " if ok else "  XX  ") + d)
    put(sorted(state) == ["retnorm/hi/value", "retnorm/lo/value"], f"state keys {sorted(state)} (debias off: no corr)")
    ref = REF_retnorm(rets, tars)
    for i, ((go, gs, _), (ro, rs, _)) in enumerate(zip(got, ref)):
        print(f"        step {i} spread {SCALES[i]:g}: offset official {float(go):.6g} f64 {ro:.6g}; "
              f"scale official {float(gs):.6g} f64 {rs:.6g}")
    ok, w = match(got, ref)
    put(ok, f"6 updates: offset, scale, adv f32 vs f64 worst {w:.3g} x(|ref|+1)")
    scs = [r[1] for r in ref]
    both = any(s == LIMIT for s in scs) and any(s > LIMIT for s in scs)
    put(both, f"limit branch taken both ways (scales {', '.join(f'{s:.4g}' for s in scs)}) — not blind")
    for name, kw in PLANTED.items():
        c = not match(got, REF_retnorm(rets, tars, **kw))[0]
        put(c, f"planted '{name}': {'CAUGHT' if c else 'MISSED'}")
    if "--dump-golden" in sys.argv:
        np.savez_compressed(GOLDEN12A, rets=np.stack(rets), tars=np.stack(tars),
                            **{f"step{i}/{k}": np.asarray(v) for i, o in enumerate(got)
                               for k, v in zip(("offset", "scale", "adv"), o)})
        print("        dumped", GOLDEN12A)
    n = len(rows); p = sum(rows)
    print(f"rung12a references vs official e01491f: {p}/{n} {'HOLD' if p == n else 'BROKEN'}")
    return 0 if p == n else 1


# ---------------------------------------------------------------- port tests (MLX)
# `mlx_maths.retnorm_perc(lo, hi, ret, rate, limit, perclo, perchi)` -> (lo', hi', offset, scale):
#   one Normalize('perc', debias False).__call__(ret, update=True) — update first, then stats.
# `mlx_maths.advantage(ret, tarval, scale)` -> (ret - tarval[:, :-1]) / scale (advnorm 'none').
# Each test eager AND under mx.compile, over the 6 steps carried from lo = hi = 0:
#   retnorm_ref     port vs REF_retnorm (float64)
#   retnorm_golden  port vs the official f32 dump


def _port(M, mx, compiled, rets, tars):
    f = lambda lo, hi, r, t: (*M.retnorm_perc(lo, hi, r, RATE, LIMIT, LO, HI), )  # noqa: E731
    f = mx.compile(f) if compiled else f
    a = mx.compile(M.advantage) if compiled else M.advantage
    lo = hi = mx.array(np.float32(0))
    out = []
    for r, t in zip(rets, tars):
        lo, hi, off, sc = f(lo, hi, mx.array(r), mx.array(t))
        out.append((np.asarray(off), np.asarray(sc), np.asarray(a(mx.array(r), mx.array(t), sc))))
    return out


def t_ref(M, mx, compiled):
    rets, tars = inputs()
    ok, w = match(_port(M, mx, compiled, rets, tars), REF_retnorm(rets, tars))
    return [(ok, f"6 steps offset/scale/adv vs REF: worst {w:.3g} x(|ref|+1)")]


def t_golden(M, mx, compiled):
    G = np.load(GOLDEN12A)
    rets, tars = list(G["rets"]), list(G["tars"])
    gold = [tuple(G[f"step{i}/{k}"] for k in ("offset", "scale", "adv")) for i in range(len(rets))]
    ok, w = match(_port(M, mx, compiled, rets, tars), gold)
    return [(ok, f"6 steps offset/scale/adv vs GOLDEN: worst {w:.3g} x(|ref|+1)")]


TESTS = [("retnorm_ref", t_ref), ("retnorm_golden", t_golden)]


def run_tests(M, mx, tests=TESTS, quiet=False):
    n = 0
    for name, t in tests:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c)
                r[mode] = all(o for o, _ in rows)
                det = [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if not quiet and (mode == "eager" or not r[mode]):
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        if not quiet:
            print(f"{name:15s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    return n


def main_mlx():
    import rung6
    mx = rung6._mlx()
    import mlx_maths as M
    n = run_tests(M, mx)
    print(f"RUNG 12a: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


def main_planted():
    """Each planted bug is patched into the real port; the tests must FAIL."""
    import rung6
    mx = rung6._mlx()
    import mlx_maths as M
    keep = {k: getattr(M, k) for k in ("retnorm_perc", "advantage", "_percentile")}
    rp, adv, pc = keep["retnorm_perc"], keep["advantage"], keep["_percentile"]
    bugs = {
        "nearest percentile": dict(_percentile=lambda x, q: mx.sort(x.reshape(-1))[round(q / 100 * (x.size - 1))]),
        "rate 0.1": dict(retnorm_perc=lambda lo, hi, r, rate, *a: rp(lo, hi, r, 0.1, *a)),
        "limit dropped": dict(retnorm_perc=lambda lo, hi, r, rate, limit, *a: rp(lo, hi, r, rate, 0.0, *a)),
        "10/90 percentiles": dict(retnorm_perc=lambda lo, hi, r, rate, limit, a, b: rp(lo, hi, r, rate, limit, 10.0, 90.0)),
        "adv against tarval[:, 1:]": dict(advantage=lambda r, t, s: (r - t[:, 1:]) / s),
    }
    caught = 0
    for name, patch in bugs.items():
        for k, v in patch.items():
            setattr(M, k, v)
        try:
            bad = run_tests(M, mx, quiet=True) < len(TESTS)
        finally:
            for k, v in keep.items():
                setattr(M, k, v)
        caught += bad
        print(f"planted [{name}] -> {'CAUGHT' if bad else 'MISSED'}")
    ctl = run_tests(M, mx, quiet=True) == len(TESTS)
    print(f"PLANTED: {caught}/{len(bugs)} caught, unpatched port {'passes' if ctl else 'FAILS'} -> "
          f"{'OK' if caught == len(bugs) and ctl else 'HARNESS BLIND'}")
    return 0 if caught == len(bugs) and ctl else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    if "--planted" in sys.argv:
        sys.exit(main_planted())
    sys.exit(main_mlx())
