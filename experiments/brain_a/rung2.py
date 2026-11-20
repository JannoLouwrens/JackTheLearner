"""MLX port of DreamerV3 — rung 2 tests (PORT report step 1.4, first half).

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table row
1.4: "Rung 2 tests, then returns, percentile normaliser, slow-critic average —
Rung 2 passes". MLX has no percentile function: "write it as sort plus linear
interpolation, tested against NumPy". Official code `danijar/dreamerv3` at
e01491f (`/data/src/dreamerv3_e01491f`):

  returns      dreamerv3/agent.py L482-490 `lambda_return(last, term, rew,
               val, boot, disc, lam)`; config horizon 333 -> disc 1 - 1/333
               (L464, repl loss; imag loss with contdisc uses disc 1, L401),
               lam 0.95 (configs.yaml L108-109).
  normaliser   embodied/jax/utils.py L16-91 `Normalize(impl='perc')` with
               the config's retnorm (configs.yaml L111): rate 0.01, limit 1.0,
               perclo 5, perchi 95, debias False. `jnp.percentile` default =
               method 'linear' (NumPy's default too). Also debias=True is
               tested (the class default), since `corr` is part of the class.
  slow critic  embodied/jax/utils.py L94-124 `SlowModel.update`: mix = rate if
               count % every == 0 else 0; dst <- mix*src + (1-mix)*dst;
               count += 1. Config slowvalue {rate: 0.02, every: 1} (L110);
               every = 3 is tested too (the branch that must skip).

No gradients: in the official losses returns, normaliser stats and the slow
critic all sit under stop-gradient (agent.py L397-422), so rung 2 is forward
only — stated, and the test that `stats` carries no gradient is the port's
`mx.stop_gradient` on the outputs (checked: d stats / d x == 0).

Tolerance and "1e-6" exactly as rung 1 (`rung1.close`): per element
|port - ref| <= 1e-6 * (|ref| + s), s = 1 (returns, normaliser state and slow
weights are sums of O(1) terms). References are NumPy float64 from the same
float32 inputs. The percentile index arithmetic is exact (no tolerance on
which two elements are interpolated: a wrong rank misses by far more).

  # references against the official JAX code (float32 CPU):
  nice -n 19 /data/venvs/jack-dv3/bin/python -I experiments/brain_a/rung2.py --vs-official
  # the port, eager and mx.compile'd (exit 0 = rung 2 PASS):
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung2.py
  # each planted wrong implementation must FAIL its test:
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung2.py --planted

STATE 11-20 (J16 (e)): tests committed failing vs stubs (d5d840d1), then the
module filled: RUNG 2 4/4 PASS eager and compiled (rung2_pass.log).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL, close  # noqa: E402

HORIZON = 333
DISC = 1 - 1 / HORIZON
LAM = 0.95
RETNORM = dict(rate=0.01, limit=1.0, perclo=5.0, perchi=95.0)
SLOW_RATE = 0.02

# ---------------------------------------------------------------- references


def REF_lambda_return(last, term, rew, val, boot, disc, lam):
    """agent.py L482-490 line for line, float64."""
    last, term, rew, val, boot = (np.float64(a) for a in (last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def REF_percentile(x, q):
    """NumPy's 'linear' method (= jnp.percentile's default) on the flat array."""
    return np.percentile(np.float64(x).ravel(), q, method="linear")


def REF_normalize(batches, rate, limit, perclo, perchi, debias):
    """utils.py L16-91, impl 'perc': state after each update, then stats."""
    lo = hi = corr = 0.0
    out = []
    for x in batches:
        lo = (1 - rate) * lo + rate * REF_percentile(x, perclo)
        hi = (1 - rate) * hi + rate * REF_percentile(x, perchi)
        if debias:
            corr = (1 - rate) * corr + rate * 1.0
        c = 1.0 / max(rate, corr) if debias else 1.0
        out.append((lo, hi, lo * c, max(limit, hi * c - lo * c)))
    return np.array(out)          # [K, 4]: lo, hi, offset, scale


def REF_slow(src_seq, dst0, rate, every):
    """utils.py L113-119 over a sequence of source weights; returns dst after each."""
    dst, out = np.float64(dst0), []
    for count, src in enumerate(src_seq):
        mix = rate if count % every == 0 else 0.0
        dst = mix * np.float64(src) + (1 - mix) * dst
        out.append(dst.copy())
    return np.stack(out)


def inputs():
    r = np.random.RandomState(20261120)
    B, T = 4, 16
    I = {
        "last": (r.rand(B, T) < 0.08).astype(np.float32),
        "term": (r.rand(B, T) < 0.05).astype(np.float32),
        "rew": r.randn(B, T).astype(np.float32),
        "val": (3 * r.randn(B, T)).astype(np.float32),
        "boot": (3 * r.randn(B, T)).astype(np.float32),
    }
    I["last"][0, 5] = 1; I["term"][1, 7] = 1; I["term"][2, -1] = 1   # each branch hit
    # normaliser: 12 batches of [16, 15] returns, scale drifting, one outlier;
    # sizes chosen so the 5th/95th ranks are fractional (240 elements: 0.05*239 = 11.95)
    I["norm"] = [(r.randn(16, 15) * (1 + k) + 2 * k).astype(np.float32) for k in range(12)]
    I["norm"][7][3, 4] = 400.0
    I["ties"] = np.repeat(np.float32([1, 2, 3]), 7).reshape(3, 7)   # tied values
    I["slow_src"] = [(r.randn(32, 8) * 0.1 + 0.01 * k).astype(np.float32) for k in range(10)]
    I["slow_dst0"] = (r.randn(32, 8) * 0.1).astype(np.float32)
    return I


# ---------------------------------------------------------------- the tests
# Port API (mlx_maths.py):
#   lambda_return(last, term, rew, val, boot, disc, lam) -> [B, T-1]
#   percentile(x, q)                                   -> scalar, linear method
#   perc_update(lo, hi, corr, x, rate, perclo, perchi, debias) -> (lo, hi, corr)
#   perc_stats(lo, hi, corr, rate, limit, debias)      -> (offset, scale), stop-gradient
#   slow_update(src, dst, count, rate, every)          -> (dst', count + 1)

def _wrap(mx, fn, compiled):
    return mx.compile(fn) if compiled else fn


def t_returns(M, mx, c, I):
    a = [mx.array(I[k]) for k in ("last", "term", "rew", "val", "boot")]
    rows = []
    for disc in (DISC, 1.0):
        f = _wrap(mx, lambda *z, d=disc: M.lambda_return(*z, d, LAM), c)
        rows.append(close(f"returns disc={disc:.6f}", f(*a),
                          REF_lambda_return(*(I[k] for k in ("last", "term", "rew", "val", "boot")), disc, LAM), 1.0))
    return rows


def t_percentile(M, mx, c, I):
    rows = []
    for nm, x in (("norm[7]", I["norm"][7]), ("norm[0]", I["norm"][0]), ("ties", I["ties"])):
        for q in (0.0, 5.0, 50.0, 95.0, 100.0):
            f = _wrap(mx, lambda z, q=q: M.percentile(z, q), c)
            rows.append(close(f"perc {nm} q={q:g}", f(mx.array(x)), REF_percentile(x, q), 1.0))
    return rows


def t_normalize(M, mx, c, I):
    rows = []
    for debias in (False, True):
        ref = REF_normalize(I["norm"], debias=debias, **RETNORM)
        lo = hi = corr = mx.array(0.0)
        up = _wrap(mx, lambda lo, hi, corr, x: M.perc_update(
            lo, hi, corr, x, RETNORM["rate"], RETNORM["perclo"], RETNORM["perchi"], debias), c)
        st = _wrap(mx, lambda lo, hi, corr: M.perc_stats(
            lo, hi, corr, RETNORM["rate"], RETNORM["limit"], debias), c)
        got = []
        for x in I["norm"]:
            lo, hi, corr = up(lo, hi, corr, mx.array(x))
            off, sc = st(lo, hi, corr)
            got.append([float(lo), float(hi), float(off), float(sc)])
        rows.append(close(f"normalize debias={debias} (lo, hi, offset, scale) x12",
                          np.array(got), ref, 1.0))
    # stats carry no gradient (official: sg(lo), sg(max(limit, hi - lo)))
    g = mx.grad(lambda lo, hi: sum(M.perc_stats(lo, hi, mx.array(0.0), 0.01, 1e-8, False)),
                argnums=(0, 1))(mx.array(1.0), mx.array(5.0))
    z = all(float(v) == 0.0 for v in g)
    rows.append((z, f"d stats / d (lo, hi) == 0: {[float(v) for v in g]}"))
    return rows


def t_slow(M, mx, c, I):
    rows = []
    for every in (1, 3):
        ref = REF_slow(I["slow_src"], I["slow_dst0"], SLOW_RATE, every)
        f = _wrap(mx, lambda s, d, n: M.slow_update(s, d, n, SLOW_RATE, every), c)
        dst, n, got = mx.array(I["slow_dst0"]), mx.array(0), []
        for s in I["slow_src"]:
            dst, n = f(mx.array(s), dst, n)
            got.append(np.asarray(dst))
        rows.append(close(f"slow every={every} x10", np.stack(got), ref, 1.0))
        rows.append((int(n) == 10, f"slow count after 10 updates == 10: {int(n)}"))
    return rows


TESTS = [("returns", t_returns), ("percentile", t_percentile),
         ("normalize", t_normalize), ("slow_critic", t_slow)]


def run_tests(M, mx, I, verbose=True):
    res = {}
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c, I)
                ok, det = all(o for o, _ in rows), [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                ok, det = False, [f"{type(e).__name__}: {e}"[:200]]
            r[mode], r[mode + "_detail"] = ok, det
        res[name] = r
        if verbose:
            print(f"{name:12s} eager {'PASS' if r['eager'] else 'FAIL'}  "
                  f"compiled {'PASS' if r['compiled'] else 'FAIL'}")
            for d in r["eager_detail"]:
                print("    e:", d)
            if r["compiled_detail"] != r["eager_detail"]:
                for d in r["compiled_detail"]:
                    print("    c:", d)
    return res


# ---------------------------------------------------------------- planted
# Standalone WRONG implementations, each a named bug class; its test must FAIL.

def planted(mx):
    import types

    def ret_noboot(last, term, rew, val, boot, disc, lam):
        # classic TD(lambda) that drops the (1 - cont) * live * boot term
        T = rew.shape[1]
        rets = [boot[:, -1]]
        for t in reversed(range(T - 1)):
            live = (1 - term[:, t + 1]) * disc
            cont = (1 - last[:, t + 1]) * lam
            rets.append(rew[:, t + 1] + live * cont * rets[-1])
        return mx.stack(list(reversed(rets))[:-1], 1)

    def perc_nearest(x, q):          # rank rounded, no interpolation
        s = mx.sort(x.reshape(-1))
        return s[int(round(q / 100 * (s.size - 1)))]

    def perc_update_ok(lo, hi, corr, x, rate, plo, phi, debias):
        p = lambda q: mx.array(np.percentile(np.asarray(x, dtype=np.float64), q).astype(np.float32))  # noqa: E731
        lo = (1 - rate) * lo + rate * p(plo); hi = (1 - rate) * hi + rate * p(phi)
        return lo, hi, (1 - rate) * corr + rate if debias else corr

    def stats_nolimit(lo, hi, corr, rate, limit, debias):   # forgets max(limit, .) and debias
        return mx.stop_gradient(lo), mx.stop_gradient(hi - lo)

    def slow_always(src, dst, count, rate, every):          # ignores `every`
        return rate * src + (1 - rate) * dst, count + 1

    P = {
        "returns": ("TD(lambda) without the (1-cont)*live*boot term",
                    dict(lambda_return=ret_noboot)),
        "percentile": ("nearest rank, no linear interpolation", dict(percentile=perc_nearest)),
        "normalize": ("scale without max(limit, .) and without debias",
                      dict(perc_update=perc_update_ok, perc_stats=stats_nolimit)),
        "slow_critic": ("mix every update, `every` ignored", dict(slow_update=slow_always)),
    }
    return {k: (why, types.SimpleNamespace(**f)) for k, (why, f) in P.items()}


def _mlx():
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"   # mlx_smoke.py: mx.compile needs it
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    import mlx.core as mx
    return mx


def main_mlx(planted_mode):
    mx = _mlx()
    I = inputs()
    if not planted_mode:
        import mlx_maths as M
        res = run_tests(M, mx, I)
        n = sum(r["eager"] and r["compiled"] for r in res.values())
        print(f"RUNG 2: {n}/{len(res)} tests pass eager AND compiled "
              f"(mlx {mx.__version__}) -> {'PASS' if n == len(res) else 'FAIL'}")
        return 0 if n == len(res) else 1
    caught, P = 0, planted(mx)
    for test, (why, M) in P.items():
        t, fails = dict(TESTS)[test], []
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                bad = [d for o, d in t(M, mx, c, I) if not o]
            except Exception as e:  # noqa: BLE001
                bad = [f"{type(e).__name__}: {e}"[:160]]
            if bad:
                fails.append(f"{mode}: {bad[0]}")
        caught += bool(fails)
        print(f"planted {test:12s} [{why}] -> {'CAUGHT' if fails else 'MISSED'}")
        for f in fails:
            print("    ", f)
    print(f"PLANTED: {caught}/{len(P)} caught -> {'OK' if caught == len(P) else 'HARNESS BLIND'}")
    return 0 if caught == len(P) else 1


# ---------------------------------------------------------------- vs official

def main_official():
    """Each REF_* against the official JAX code at e01491f, float32 CPU."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.agent import lambda_return
    from embodied.jax.utils import Normalize, SlowModel  # noqa: F401
    I = inputs()
    rows = []

    def chk(name, got, ref, s):
        ok, d = close(name, np.asarray(got), ref, s)
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    a = [jnp.array(I[k]) for k in ("last", "term", "rew", "val", "boot")]
    for disc in (DISC, 1.0):
        chk(f"returns disc={disc:.6f} (jit)", jax.jit(lambda *z, d=disc: lambda_return(*z, d, LAM))(*a),
            REF_lambda_return(*(I[k] for k in ("last", "term", "rew", "val", "boot")), disc, LAM), 1.0)
    for nm, x in (("norm[7]", I["norm"][7]), ("norm[0]", I["norm"][0]), ("ties", I["ties"])):
        for q in (0.0, 5.0, 50.0, 95.0, 100.0):
            # the training step is jitted (agent.train under jax.jit), so the
            # official value is the COMPILED one (rung 1's lesson)
            chk(f"perc {nm} q={q:g} (jit)", jax.jit(lambda z, q=q: jnp.percentile(z, q))(jnp.array(x)),
                REF_percentile(x, q), 1.0)
    # report-only: EAGER jnp.percentile at q=100 lands one float32 ulp below the
    # top rank (measured 11-20: 399.99448 for a max of 400.0, w = 1.5e-5 on the
    # rank below); jit gives 400.0. The agent uses q = 5 and 95, compiled.
    e = float(jnp.percentile(jnp.array(I["norm"][7]), 100.0))
    rows.append((True, f"eager jnp.percentile(norm[7], 100) = {e!r} vs max {float(I['norm'][7].max())!r} (report-only)"))
    print("  ..  " + rows[-1][1])
    for debias in (False, True):
        norm = Normalize("perc", name="retnorm", debias=debias, **RETNORM)

        def step(x):
            norm.update(x)
            lo, sc = norm.stats()
            return jnp.stack([norm.lo.read(), norm.hi.read(), lo, sc])
        state, got = {}, []
        fn = nj.pure(step)
        for x in I["norm"]:
            state, out = fn(state, jnp.array(x), create=True)
            got.append(np.asarray(out))
        chk(f"normalize debias={debias} x12", np.stack(got),
            REF_normalize(I["norm"], debias=debias, **RETNORM), 1.0)
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    print("slow critic: SlowModel.update needs a tracked nj.Module; its three lines "
          "(L115-119) are restated in REF_slow and not run here — stated, not hidden.")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--vs-official" in a:
        sys.exit(main_official())
    sys.exit(main_mlx("--planted" in a))
