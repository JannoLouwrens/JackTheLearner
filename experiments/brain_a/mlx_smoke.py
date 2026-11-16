"""MLX port, stage 0 step 0.1 — the environment and its smoke test.

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step 0.1
("Container recipe (Debian 12, Python 3.12, mlx[cpu]==0.32.4, NumPy), capped at
1 core and 2 GB; smoke script — done when the smoke script prints versions;
gradient, compile, 64-bit all pass").

Ours beside theirs (the recipe, not a learner):

  | setting            | report (step 0.1)                 | ours                                   |
  |--------------------|-----------------------------------|----------------------------------------|
  | isolation          | Debian 12 container (glibc 2.36)  | conda-forge env `/data/venvs/jack-mlx` |
  | MLX build          | PyPI wheel `mlx[cpu]==0.32.4`     | conda-forge `mlx 0.32.4` (same tag)    |
  | Python             | 3.12                              | 3.12 (conda-forge)                     |
  | cap                | 1 core, 2 GB                      | `taskset -c 0`, peak RSS asserted <2 GB|

Why not the report's recipe: the PyPI wheels are `manylinux_2_35`; this box has
glibc 2.34. `objdump -T` on the 0.32.4 aarch64 wheel shows exactly two symbols
above it — `hypotf@GLIBC_2.35` (libm) and
`condition_variable::wait@GLIBCXX_3.4.30` (box libstdc++ tops at 3.4.29) — so
it cannot load. Docker on this box is refused to the worker lane. conda-forge
builds MLX against its own older sysroot and ships its own libstdc++ (>=15), so
it loads natively. The build is the same upstream tag but a DIFFERENT binary
(compiler, BLAS/LAPACK): before any golden-file parity rung is trusted on the
box, the same rung must also pass with the PyPI wheel on Kaggle (Ubuntu,
glibc >= 2.35), where step 0.2 installs it anyway.

Rebuild:
    curl -sL https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
      | tar -xj -C /data/tools/micromamba bin/micromamba
    MAMBA_ROOT_PREFIX=/data/tools/micromamba/root TMPDIR=/data/tmp/mlxtmp \
      /data/tools/micromamba/bin/micromamba create -y -p /data/venvs/jack-mlx \
      -c conda-forge --override-channels python=3.12 mlx=0.32.4 numpy
Run (1 core):
    taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I \
      experiments/brain_a/mlx_smoke.py

Findings of the first run (11-16): MLX's CPU `mx.compile` JIT-builds C++ with
the `g++` on PATH and the box's g++ 11 cannot build it — gcc-toolset-15 can
(set below); `mx.fast.layer_norm`/`rms_norm` silently compute float64 input at
float32 precision, and so do float64 `exp`/`sigmoid`/`softmax`/`logsumexp`
(rel ~3e-7) — traps for 64-bit references, both measured each run. The
"64-bit passes" item of step 0.1 therefore holds for exp-free ops only.

Every check carries a control that MUST fail; the script exits 1 if any check
fails or any control passes. Pure MLX + NumPy; touches nothing in the repo.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time

import numpy as np

# MLX's CPU compiler JIT-builds C++ with whatever `g++` is first on PATH. The
# box's g++ 11 (and gcc-toolset-14) reject the `bf16` literal MLX emits
# (`unable to find numeric literal operator 'operator""bf16'`); gcc-toolset-15
# compiles it. Every MLX process on this box must see it first.
GXX = "/opt/rh/gcc-toolset-15/root/usr/bin"
if os.path.isdir(GXX) and not os.environ.get("PATH", "").startswith(GXX):
    os.environ["PATH"] = GXX + os.pathsep + os.environ.get("PATH", "")

import mlx.core as mx
import mlx.nn as nn

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str, control_ok: bool | None = None) -> None:
    RESULTS.append({"check": name, "pass": bool(ok), "control_failed": control_ok,
                    "detail": detail})
    tag = "PASS" if ok else "FAIL"
    ctl = "" if control_ok is None else (" | control fails: yes" if control_ok
                                         else " | control fails: NO")
    print(f"[{tag}] {name}: {detail}{ctl}")


def check_versions() -> None:
    import importlib.metadata as md
    v = mx.__version__
    print(f"python {platform.python_version()} | mlx {v} | numpy {np.__version__} | "
          f"machine {platform.machine()} | libc {platform.libc_ver()} | "
          f"device {mx.default_device()}")
    try:
        src = md.distribution("mlx").read_text("INSTALLER") or "?"
    except Exception:  # conda installs carry no INSTALLER file
        src = "conda"
    record("versions", v == "0.32.4", f"mlx {v} (pinned 0.32.4), installer {src.strip()}")


def _manual_ln(h, g, b, eps):
    mu = mx.mean(h, axis=-1, keepdims=True)
    v = mx.mean((h - mu) ** 2, axis=-1, keepdims=True)
    return (h - mu) * mx.rsqrt(v + eps) * g + b


def check_gradient() -> None:
    """Autograd vs float64 central differences, through a layer-normed MLP.

    The layer norm here is written from primitives: `mx.fast.layer_norm` cannot
    be finite-differenced in float64 because it computes in float32 inside (see
    `check_fast_norm_precision`). Control: the same comparison against a
    gradient with one entry moved by 1e-3 must fail the 1e-6 tolerance.
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 5))
    x = rng.standard_normal((4, 6))
    g = rng.standard_normal(5) * 0.5 + 1.0
    b = rng.standard_normal(5) * 0.1

    with mx.stream(mx.cpu):
        x_m, g_m, b_m = (mx.array(a, dtype=mx.float64) for a in (x, g, b))

        def f_mx(w_):
            return mx.sum(mx.tanh(_manual_ln(x_m @ w_, g_m, b_m, 1e-4)) ** 2)

        grad = np.array(mx.grad(f_mx)(mx.array(w, dtype=mx.float64)))
        eps = 1e-6
        num = np.zeros_like(w)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                wp, wm = w.copy(), w.copy()
                wp[i, j] += eps
                wm[i, j] -= eps
                num[i, j] = (f_mx(mx.array(wp, dtype=mx.float64)).item()
                             - f_mx(mx.array(wm, dtype=mx.float64)).item()) / (2 * eps)
    err = float(np.max(np.abs(grad - num)))
    bad = grad.copy()
    bad[2, 3] += 1e-3
    ctl_err = float(np.max(np.abs(bad - num)))
    record("gradient (layer-normed MLP, f64 central diff)", err < 1e-6,
           f"max |analytic - numeric| = {err:.2e} (tol 1e-6)", ctl_err >= 1e-6)


def check_fast_layer_norm_grad() -> None:
    """`mx.fast.layer_norm` gradient (fixed in 0.32.0) vs the primitive one, float32.

    Pass: x, weight and bias gradients agree to 1e-5 relative. Control: the
    primitive layer norm with eps 1e-5 (MLX's default, not Dreamer's 1e-4) on a
    small-variance input must differ by more than that.
    """
    rng = np.random.default_rng(3)
    h = rng.standard_normal((16, 32)) * 0.01
    g = rng.standard_normal(32) * 0.5 + 1.0
    b = rng.standard_normal(32) * 0.1
    c = rng.standard_normal((16, 32))
    with mx.stream(mx.cpu):
        H, G, B, C = (mx.array(a, dtype=mx.float32) for a in (h, g, b, c))
        fa = lambda h_, g_, b_: mx.sum(mx.fast.layer_norm(h_, g_, b_, 1e-4) * C)
        fm = lambda h_, g_, b_: mx.sum(_manual_ln(h_, g_, b_, 1e-4) * C)
        fw = lambda h_, g_, b_: mx.sum(_manual_ln(h_, g_, b_, 1e-5) * C)
        ga = mx.grad(fa, argnums=[0, 1, 2])(H, G, B)
        gm = mx.grad(fm, argnums=[0, 1, 2])(H, G, B)
        gw = mx.grad(fw, argnums=[0, 1, 2])(H, G, B)

    def rel(p, q):
        return max(float(np.max(np.abs(np.array(x) - np.array(y))) /
                         (np.max(np.abs(np.array(y))) + 1e-30)) for x, y in zip(p, q))
    r, rc = rel(ga, gm), rel(gw, gm)
    record("fast.layer_norm grad == primitive (f32, x/w/b)", r < 1e-5,
           f"max rel {r:.1e} (tol 1e-5); eps 1e-5 control rel {rc:.1e}", rc >= 1e-5)


def check_fast_norm_precision() -> None:
    """TRAP, measured: `mx.fast.layer_norm`/`rms_norm` on float64 input return a
    float64 array computed at float32 precision.

    Pass = the trap is still exactly as documented (forward error vs the
    primitive float64 norm between 1e-9 and 1e-5); if MLX fixes it this check
    flips and the note must be revisited. Consequence for the port: 64-bit NumPy
    references (rung 1) and any float64 finite-difference test must use the
    primitive norm, never `mx.fast.*`.
    """
    rng = np.random.default_rng(4)
    h = rng.standard_normal((8, 64))
    with mx.stream(mx.cpu):
        H = mx.array(h, dtype=mx.float64)
        one, zero = mx.ones((64,), dtype=mx.float64), mx.zeros((64,), dtype=mx.float64)
        fast = mx.fast.layer_norm(H, one, zero, 1e-4)
        prim = _manual_ln(H, one, zero, 1e-4)
        mx.eval(fast, prim)
    err = float(np.max(np.abs(np.array(fast) - np.array(prim))))
    record("TRAP fast.layer_norm f64 runs at f32 precision", 1e-9 < err < 1e-5,
           f"out dtype {fast.dtype}, |fast - primitive| f64 = {err:.1e}")


def check_compile() -> None:
    """A 64-step unrolled recurrence with gradients, compiled vs eager.

    Shape is a toy (the real compile cost is step 0.3's question). Pass: the
    compiled loss and gradient match eager to 1e-5 relative, and a second call
    of the compiled function is faster than its first (a trace was cached).
    Control: compiled output vs eager output of a DIFFERENT input must differ.
    """
    rng = np.random.default_rng(1)
    H, T = 32, 64
    Wh = mx.array(rng.standard_normal((H, H)) * 0.2, dtype=mx.float32)
    Wx = mx.array(rng.standard_normal((8, H)) * 0.2, dtype=mx.float32)
    xs = mx.array(rng.standard_normal((T, 16, 8)), dtype=mx.float32)
    xs2 = xs + 0.5

    def loss(wh, xs_):
        h = mx.zeros((16, H))
        tot = mx.array(0.0)
        for t in range(T):
            h = mx.tanh(h @ wh + xs_[t] @ Wx)
            tot = tot + mx.mean(h ** 2)
        return tot

    vg = mx.value_and_grad(loss)
    cvg = mx.compile(vg)
    with mx.stream(mx.cpu):
        le, ge = vg(Wh, xs)
        mx.eval(le, ge)
        t0 = time.perf_counter()
        lc, gc = cvg(Wh, xs)
        mx.eval(lc, gc)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        lc2, gc2 = cvg(Wh, xs)
        mx.eval(lc2, gc2)
        t_second = time.perf_counter() - t0
        lo, _ = cvg(Wh, xs2)
        mx.eval(lo)
    rel_l = abs(lc.item() - le.item()) / abs(le.item())
    rel_g = float(np.max(np.abs(np.array(gc) - np.array(ge))) /
                  np.max(np.abs(np.array(ge))))
    ctl = abs(lo.item() - le.item()) / abs(le.item()) > 1e-5
    ok = rel_l < 1e-5 and rel_g < 1e-5 and t_second < t_first
    record("compile (64-step unrolled recurrence + grad)", ok,
           f"loss rel {rel_l:.1e}, grad rel {rel_g:.1e} (tol 1e-5); "
           f"first call {t_first:.3f} s, cached {t_second:.4f} s", ctl)


def check_float64() -> None:
    """float64 on the CPU stream, exp-free ops, vs NumPy float64 to 1e-12 relative.

    First run (11-16) used matmul+exp+log1p and FAILED at 3.5e-8: per-op
    isolation showed matmul/add/mul/log/log1p/sqrt exact, tanh 4e-16, sum 3e-15,
    and the exp family at float32 accuracy (`check_exp_family_precision`). This
    check is the exp-free half; the exp half is a measured trap, not a pass.
    Control: the same ops in float32 must MISS 1e-12.
    """
    rng = np.random.default_rng(2)
    a = rng.standard_normal((64, 64))
    bm = rng.standard_normal((64, 64))
    ref = np.tanh(np.log1p(np.abs(a @ bm)) + np.sqrt(np.abs(a)) * bm).sum(axis=1)
    def run(dt):
        A, B = mx.array(a, dtype=dt), mx.array(bm, dtype=dt)
        o = mx.sum(mx.tanh(mx.log1p(mx.abs(A @ B)) + mx.sqrt(mx.abs(A)) * B), axis=1)
        mx.eval(o)
        return o
    with mx.stream(mx.cpu):
        out, out32 = run(mx.float64), run(mx.float32)
    rel = lambda o: float(np.max(np.abs(np.array(o).astype(np.float64) - ref) /
                                 np.maximum(np.abs(ref), 1e-300)))
    err, err32 = rel(out), rel(out32)
    record("float64 exp-free ops (matmul,log1p,sqrt,tanh,sum vs numpy)",
           out.dtype == mx.float64 and err < 1e-12,
           f"f64 max rel {err:.1e} (tol 1e-12); f32 control {err32:.1e}",
           err32 >= 1e-12)


def check_exp_family_precision() -> None:
    """TRAP, measured: float64 `exp`, `sigmoid`, `softmax`, `logsumexp` are only
    float32-accurate (relative ~1e-7 vs NumPy), while the output dtype says f64.

    Pass = the trap is still as documented (exp rel error in 1e-9..1e-5).
    Consequence for the port: every 64-bit reference (rung 1: symlog/symexp,
    two-hot decode, distributions) is computed in NumPy, never in MLX float64.
    Measured on the conda-forge build; step 0.2 repeats it on the PyPI wheel.
    """
    rng = np.random.default_rng(5)
    a = rng.standard_normal((64, 64)) * 3
    with mx.stream(mx.cpu):
        A = mx.array(a, dtype=mx.float64)
        e = mx.exp(A)
        sm = mx.softmax(A, axis=-1)
        mx.eval(e, sm)
    rel_e = float(np.max(np.abs(np.array(e) - np.exp(a)) / np.exp(a)))
    ref_sm = np.exp(a - a.max(-1, keepdims=True))
    ref_sm /= ref_sm.sum(-1, keepdims=True)
    rel_sm = float(np.max(np.abs(np.array(sm) - ref_sm) / ref_sm))
    record("TRAP float64 exp/softmax at f32 accuracy", 1e-9 < rel_e < 1e-5,
           f"exp rel {rel_e:.1e}, softmax rel {rel_sm:.1e} (dtype {e.dtype})")


def check_keyed_random() -> None:
    """JAX-style keys: same key -> same bits; split keys -> different bits.

    The port reads its noise from golden files and never relies on this, but the
    report lists keyed random numbers among the container checks.
    """
    k = mx.random.key(7)
    a = np.array(mx.random.normal((1000,), key=k))
    b = np.array(mx.random.normal((1000,), key=k))
    k1, k2 = mx.random.split(k)
    c = np.array(mx.random.normal((1000,), key=k1))
    record("keyed random (same key repeats)", bool(np.array_equal(a, b)),
           f"identical {np.array_equal(a, b)}; split key differs "
           f"{not np.array_equal(a, c)}", not np.array_equal(a, c))


def main() -> int:
    t0 = time.perf_counter()
    check_versions()
    check_gradient()
    check_fast_layer_norm_grad()
    check_fast_norm_precision()
    check_compile()
    check_float64()
    check_exp_family_precision()
    check_keyed_random()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        import os
        cores = len(os.sched_getaffinity(0))
    except Exception:
        cores = -1
    record("cap (peak RSS < 2 GB)", peak_mb < 2048,
           f"peak RSS {peak_mb:.0f} MB, cores visible {cores}, "
           f"wall {time.perf_counter() - t0:.2f} s")
    bad = [r for r in RESULTS if not r["pass"] or r["control_failed"] is False]
    print(json.dumps({"mlx": mx.__version__, "all_pass": not bad,
                      "results": RESULTS}, indent=None))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
