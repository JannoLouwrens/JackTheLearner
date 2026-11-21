"""MLX port of DreamerV3 — rung 3 tests (PORT report step 1.5): the optimiser.

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table row
1.5: "Optimiser golden (warm-up values, 5 steps) from the generator; then
clipping, scaling, momentum, warm-up — Rung 3 passes; first update has
learning rate 0". Official code `danijar/dreamerv3` at e01491f
(`/data/src/dreamerv3_e01491f`):

  chain     dreamerv3/agent.py L342-380 `_make_opt`: clip_by_agc(agc) ->
            scale_by_rms(beta2, eps) -> scale_by_momentum(beta1) -> (wd 0:
            none) -> optax.scale_by_learning_rate(join(linear 0->lr over
            warmup, const lr)).
  agc       embodied/jax/opt.py `clip_by_agc(clip=0.3, pmin=1e-3)`: per
            tensor, u * 1/max(1, |u| / (clip * max(pmin, |p|))).
  rms       `scale_by_rms`: step += 1; nu = b2 nu + (1-b2) u^2;
            u / (sqrt(nu / (1 - b2^step)) + eps).
  momentum  `scale_by_momentum` (nesterov False): mu = b1 mu + (1-b1) u;
            out = mu / (1 - b1^step).
  lr        optax: params += -sched(count) * u, count from 0, so update 1
            uses lr * 0 / warmup = 0 (the row's "first update has lr 0").
  config    configs.yaml L87: lr 4e-5, agc 0.3, eps 1e-20, beta1 0.9,
            beta2 0.999, warmup 1000, schedule const, wd 0.

The "golden from the generator" half of row 1.5 is the official chain run
here (`--vs-official`, optax at the jack-dv3 venv), 5 steps from the same
float32 inputs; the NumPy float64 reference must agree with it before the
port is read against the reference.

Two configs, both 5 steps: CFG (configs.yaml exactly; deltas are ~1e-10 so
they are compared as delta / sum of lr so far, O(1)) and VIS (warmup 3, lr 0.1: steps
past the ramp, deltas visible on the parameters themselves). Inputs hit
each branch: one tensor whose agc clip binds, one all-zero tensor (the pmin
branch), one where it does not bind. Tolerance as rung 1: |port - ref| <=
1e-6 * (|ref| + 1) on delta / (lr summed so far) and on the optimiser state (nu, mu).

  nice -n 19 /data/venvs/jack-dv3/bin/python -I experiments/brain_a/rung3.py --vs-official
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung3.py
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung3.py --planted

STATE 11-21 10:10 (J16 row 1.5): references HOLD, `--vs-official` 14/14
(was 10/14). The CFG-under-jit miss is NAMED and in the reference: XLA:CPU
runs optax's ramp as lr_t = fma(-lr, fma(-count, f32(1/warmup), 1), lr) —
the reciprocal rewrite + two fused multiply-adds, the same rewrite as rung
1's linspace; bit-exact vs jax.jit at every count of four configs (`ramp`).
Golden = the jitted official chain, 5 steps x 2 configs, params/nu/mu/update:
`rung3_opt_golden_e01491f.json` (`--vs-official --dump-golden`); the port
is read against the reference AND the golden. Stubs committed FAILING (0/2).
`--planted` 5/5 caught + positive control (the reference chain as a port)
PASSES. Margin note: the eager-ramp planted bug misses by 1.28e-6 against
the 1e-6 tolerance — caught, but narrowly; the port must build the fma form.
Next: fill opt_init / opt_update until 2/2 eager AND compiled.

Port API (mlx_maths.py):
  opt_init(params: dict) -> state  (state = (step int, nu dict, mu dict))
  opt_update(params, grads, state, lr, warmup, agc, eps, beta1, beta2)
      -> (params', state', updates)   (updates = the lr-scaled step, pre-store)
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL, close  # noqa: E402

PMIN = 1e-3
CFG = dict(lr=4e-5, warmup=1000, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999)
VIS = dict(lr=0.1, warmup=3, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999)
NSTEP = 5
GOLDEN = os.path.join(HERE, "rung3_opt_golden_e01491f.json")   # --vs-official --dump-golden


# ---------------------------------------------------------------- reference

def ramp(lr, warmup, count, jit=True):
    """optax.linear_schedule(0, lr, warmup) joined to const lr, AS COMPUTED.

    jit=True (default; the agent trains jitted): what XLA:CPU runs, measured
    11-21 by its HLO + an exhaustive probe — count/warmup becomes
    count * f32(1/warmup), and LLVM contracts BOTH multiply-adds:
    frac = fma(-count, r, 1), lr_t = fma(-lr, frac, lr), each one rounding.
    Matches jax.jit bit for bit at every count 0..warmup+2 on (4e-5, 1000)
    1003/1003, (3e-4, 1000) 1003/1003, (1e-4, 777) 780/780, (0.1, 3) 6/6;
    the eager form matches 390, 380, 401 and 6 of those. Same rewrite as
    rung 1's linspace (mlx_maths._jnp_linspace).
    jit=False: optax eager, (init - end) * (1 - count/steps) + end in float32,
    which cancels (count 1: 4e-5 * 9.99987e-4, 1.3e-5 off the exact 4e-8)."""
    f32, F = np.float32, np.float64
    if not warmup or count >= warmup:
        return float(f32(lr))
    L = f32(lr)
    if jit:   # an f32 x f32 product is exact in float64: one rounding = fma
        frac = f32(F(1) - F(f32(count)) * F(f32(1) / f32(warmup)))
        return float(f32(F(-L) * F(frac) + F(L)))
    frac = f32(1) - f32(count) / f32(warmup)
    return float((f32(0) - L) * frac + L)


def REF_opt(p0, grads_seq, lr, warmup, agc, eps, beta1, beta2):
    """The official chain, float64. Returns per step: (params, nu, mu, update)."""
    # theirs, measured 11-21 (the whole 7e-6 miss of a plain float64 chain):
    # the moment update uses (1 - beta) formed in Python float64 then cast to
    # f32, the bias correction 1 - f32(f32(beta) ** step) in float32. The two
    # disagree (1 - f32(.999) = 9.99987e-4 vs f32(1e-3)), so the official
    # step-1 rms update is NOT u/|u|: it is inflated by ~6.6e-6. Kept as theirs.
    f32 = np.float32
    m1, m2 = float(f32(beta1)), float(f32(beta2))          # decay, cast
    a1, a2 = float(f32(1 - beta1)), float(f32(1 - beta2))  # (1 - decay), cast
    bc = lambda b, n: 1.0 - float(f32(f32(b) ** np.int32(n)))  # noqa: E731
    p = {k: np.float64(v) for k, v in p0.items()}
    nu = {k: np.zeros_like(v) for k, v in p.items()}
    mu = {k: np.zeros_like(v) for k, v in p.items()}
    out = []
    for count, g in enumerate(grads_seq):
        step = count + 1
        upd = {}
        lr_t = ramp(lr, warmup, count)
        for k in p:
            u = np.float64(g[k])
            un, pn = np.linalg.norm(u.ravel()), np.linalg.norm(p[k].ravel())
            u = u / max(1.0, un / (agc * max(PMIN, pn)))
            nu[k] = m2 * nu[k] + a2 * u * u
            u = u / (np.sqrt(nu[k] / bc(beta2, step)) + eps)
            mu[k] = m1 * mu[k] + a1 * u
            u = mu[k] / bc(beta1, step)
            # params are STORED float32 (official and port alike); the
            # update is float64, the stored result rounded as theirs is
            upd[k] = -lr_t * u
            p[k] = np.float64(np.float32(p[k] + upd[k]))
        out.append(({k: v.copy() for k, v in p.items()},
                    {k: v.copy() for k, v in nu.items()},
                    {k: v.copy() for k, v in mu.items()}, upd))
    return out


def inputs():
    r = np.random.RandomState(20261121)
    p0 = {"big": (r.randn(8, 4) * 0.5).astype(np.float32),     # clip does not bind
          "clip": (r.randn(16) * 0.01).astype(np.float32),     # clip binds (grad >> param)
          "zero": np.zeros((3, 3), np.float32)}                 # pmin branch
    grads = []
    for k in range(NSTEP):
        grads.append({"big": (r.randn(8, 4) * 0.05).astype(np.float32),
                      "clip": (r.randn(16) * 5.0 * (1 + k)).astype(np.float32),
                      "zero": (r.randn(3, 3) * 1e-2).astype(np.float32)})
    return p0, grads


# ---------------------------------------------------------------- tests

def lrsum(cfg, i):
    """Sum of the learning rates applied in steps 1..i+1 (0 after step 1)."""
    lr, w = cfg["lr"], cfg["warmup"]
    return sum(lr * min(c, w) / w for c in range(i + 1))


def _flat(d):
    return np.concatenate([np.float64(np.asarray(d[k])).ravel() for k in sorted(d)])


def _run_port(M, mx, c, p0, grads, cfg):
    f = M.opt_update
    if c:
        f = mx.compile(lambda p, g, s: M.opt_update(p, g, s, **cfg))
    else:
        f0 = f
        f = lambda p, g, s: f0(p, g, s, **cfg)   # noqa: E731
    p = {k: mx.array(v) for k, v in p0.items()}
    s = M.opt_init(p)
    out = []
    for g in grads:
        p, s, u = f(p, {k: mx.array(v) for k, v in g.items()}, s)
        out.append((p, s, u))
    return out


def t_opt(M, mx, c, I, cfgname, cfg):
    p0, grads = I
    ref = REF_opt(p0, grads, **cfg)
    got = _run_port(M, mx, c, p0, grads, cfg)
    rows = []
    base = _flat(p0)
    for i, ((gp, gs, gu), (rp, rnu, rmu, ru)) in enumerate(zip(got, ref)):
        if i:   # the update / its lr (O(1)); stored params only where visible
            L = ramp(cfg["lr"], cfg["warmup"], i)
            rows.append(close(f"{cfgname} step {i+1} update/lr_t", _flat(gu) / L, _flat(ru) / L, 1.0))
            if cfgname == "VIS":
                L = lrsum(cfg, i)
                rows.append(close(f"{cfgname} step {i+1} stored delta/sum(lr)", (_flat(gp) - base) / L,
                                  (_flat(rp) - base) / L, 1.0))
        rows.append(close(f"{cfgname} step {i+1} nu", _flat(gs[1]), _flat(rnu), 1.0))
        rows.append(close(f"{cfgname} step {i+1} mu", _flat(gs[2]), _flat(rmu), 1.0))
    gold = _golden()[cfgname]
    for i, (gp, gs, gu) in enumerate(got):   # the jitted official itself, every step
        G = {n: {k: np.array(v) for k, v in gold[i][n].items()} for n in gold[i]}
        if i:
            L = ramp(cfg["lr"], cfg["warmup"], i)
            rows.append(close(f"{cfgname} step {i+1} update/lr_t vs GOLDEN", _flat(gu) / L, _flat(G["update"]) / L, 1.0))
        rows.append(close(f"{cfgname} step {i+1} nu vs GOLDEN", _flat(gs[1]), _flat(G["nu"]), 1.0))
        rows.append(close(f"{cfgname} step {i+1} mu vs GOLDEN", _flat(gs[2]), _flat(G["mu"]), 1.0))
    d1 = _flat(got[0][0]) - base
    rows.append((bool(np.all(d1 == 0.0)), f"{cfgname} first update moves nothing (lr 0): max|d| {np.abs(d1).max():.3g}"))
    rows.append((int(np.asarray(got[-1][1][0])) == NSTEP, f"{cfgname} step counter == {NSTEP}"))
    return rows


_GOLD = {}


def _golden():
    if not _GOLD:
        import json
        with open(GOLDEN) as fh:
            _GOLD.update(json.load(fh))
    return _GOLD


TESTS = [("opt_cfg", lambda M, mx, c, I: t_opt(M, mx, c, I, "CFG", CFG)),
         ("opt_vis", lambda M, mx, c, I: t_opt(M, mx, c, I, "VIS", VIS))]


def run_tests(M, mx, I, verbose=True):
    res = {}
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c, I)
                ok, det = all(o for o, _ in rows), [d for o, d in rows if not o][:4] or ["all rows ok"]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                ok, det = False, [f"{type(e).__name__}: {e}"[:200]]
            r[mode], r[mode + "_detail"] = ok, det
        res[name] = r
        if verbose:
            print(f"{name:10s} eager {'PASS' if r['eager'] else 'FAIL'}  "
                  f"compiled {'PASS' if r['compiled'] else 'FAIL'}")
            for d in r["eager_detail"]:
                print("    e:", d)
    return res


# ---------------------------------------------------------------- planted
# Each a wrong chain built from the REFERENCE with one bug, wrapped as a port.

def _planted_port(mx, bug):
    import types

    def opt_init(p):
        return (0, {k: np.zeros(v.shape) for k, v in p.items()},
                {k: np.zeros(v.shape) for k, v in p.items()})

    def opt_update(p, g, s, lr, warmup, agc, eps, beta1, beta2):
        # bug "none" = the reference chain itself (theirs: jitted ramp, cast
        # (1 - beta), float32 bias correction) — the positive control, must PASS
        count, nu, mu = s
        step = count + 1
        lr_t = ramp(lr, warmup, count + (bug == "warmup+1"), jit=bug != "eager_ramp")
        f32 = np.float32
        m1, m2 = float(f32(beta1)), float(f32(beta2))
        a1, a2 = float(f32(1 - beta1)), float(f32(1 - beta2))
        bc = lambda b, n: 1.0 - float(f32(f32(b) ** np.int32(n)))  # noqa: E731
        newp, nnu, nmu, upd = {}, {}, {}, {}
        for k in p:
            pk, u = np.float64(np.asarray(p[k])), np.float64(np.asarray(g[k]))
            if bug != "no_agc":
                un, pn = np.linalg.norm(u.ravel()), np.linalg.norm(pk.ravel())
                lim = agc * (pn if bug == "no_pmin" else max(PMIN, pn))
                u = u / max(1.0, un / lim) if lim > 0 else u * 0
            nnu[k] = m2 * nu[k] + a2 * u * u
            u = u / (np.sqrt(nnu[k] / (1.0 if bug == "no_bias" else bc(beta2, step))) + eps)
            nmu[k] = m1 * mu[k] + a1 * u
            u = nmu[k] / (1.0 if bug == "no_bias" else bc(beta1, step))
            upd[k] = mx.array(-lr_t * u)
            newp[k] = mx.array((pk - lr_t * u).astype(np.float32))
        return newp, (step, nnu, nmu), upd
    return types.SimpleNamespace(opt_init=opt_init, opt_update=opt_update)


PLANTED = {"warmup+1": "ramp one step early: update 1 has lr > 0",
           "no_agc": "clipping skipped",
           "no_pmin": "agc without max(pmin, |p|): zero tensor never moves",
           "no_bias": "no bias correction in rms or momentum",
           "eager_ramp": "optax's EAGER warm-up ramp, not the jitted fma form the agent runs"}


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
        print(f"RUNG 3: {n}/{len(res)} tests pass eager AND compiled "
              f"(mlx {mx.__version__}) -> {'PASS' if n == len(res) else 'FAIL'}")
        return 0 if n == len(res) else 1
    M = _planted_port(mx, "none")                  # positive control: the chain as theirs
    bad = [d for name, t in TESTS for o, d in t(M, mx, False, I) if not o]
    ctl = not bad
    print(f"control   none      [the reference chain, no bug] -> {'PASSES' if ctl else 'FAILS: ' + bad[0][:90]}")
    caught = 0
    for bug, why in PLANTED.items():
        M = _planted_port(mx, bug)
        bad = []
        for name, t in TESTS:                      # eager only: planted ports are NumPy inside
            bad += [d for o, d in t(M, mx, False, I) if not o]
        caught += bool(bad)
        print(f"planted {bug:9s} [{why}] -> {'CAUGHT' if bad else 'MISSED'}"
              + (f"  ({bad[0][:90]})" if bad else ""))
    ok = caught == len(PLANTED) and ctl
    print(f"PLANTED: {caught}/{len(PLANTED)} caught, control {'passes' if ctl else 'FAILS'} -> "
          f"{'OK' if ok else ('HARNESS BLIND' if ctl else 'HARNESS REJECTS THE RIGHT CHAIN')}")
    return 0 if ok else 1


# ---------------------------------------------------------------- vs official

def main_official():
    """REF_opt against the official optax chain at e01491f (float32 CPU, jit)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import jax
    import jax.numpy as jnp
    import optax
    from dreamerv3.agent import Agent
    p0, grads = inputs()
    rows, gold = [], {}
    for cfgname, cfg in (("CFG", CFG), ("VIS", VIS)):
        kw = dict(cfg)
        opt = Agent._make_opt(None, lr=kw["lr"], agc=kw["agc"], eps=kw["eps"], beta1=kw["beta1"],
                              beta2=kw["beta2"], warmup=kw["warmup"])
        p = {k: jnp.array(v) for k, v in p0.items()}
        st = opt.init(p)

        @jax.jit
        def step(p, g, st):
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, u
        ref = REF_opt(p0, grads, **cfg)
        base = _flat(p0)
        gold[cfgname] = []
        for i, g in enumerate(grads):
            p, st, u = step(p, {k: jnp.array(v) for k, v in g.items()}, st)
            nu, mu = st[1][1], st[2][1]   # embodied opt.py: rms/momentum states are (step, tree)
            gold[cfgname].append({n: {k: np.asarray(v, np.float32).tolist() for k, v in d.items()}
                                  for n, d in (("params", p), ("nu", nu), ("mu", mu), ("update", u))})
            if i:
                L = ramp(cfg["lr"], cfg["warmup"], i)
                chk = [close(f"{cfgname} step {i+1} update/lr_t", _flat(u) / L, _flat(ref[i][3]) / L, 1.0)]
                if cfgname == "VIS":
                    L = lrsum(cfg, i)
                    chk.append(close(f"{cfgname} step {i+1} stored delta/sum(lr)", (_flat(p) - base) / L,
                                     (_flat(ref[i][0]) - base) / L, 1.0))
                for ok, d in chk:
                    rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)
            if i == 0:
                z = bool(np.all(_flat(p) == base))
                rows.append((z, f"{cfgname} official first update moves nothing: {z}"))
                print(("  ok  " if z else "  XX  ") + rows[-1][1])
    if "--dump-golden" in sys.argv:
        import json
        gold["meta"] = {"source": "danijar/dreamerv3 e01491f Agent._make_opt under jax.jit (CPU, float32)",
                        "jax": jax.__version__, "optax": optax.__version__, "cfg": CFG, "vis": VIS,
                        "inputs": "rung3.inputs() seed 20261121", "steps": NSTEP}
        with open(GOLDEN, "w") as fh:
            json.dump(gold, fh)
        print(f"golden written: {GOLDEN}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, optax {optax.__version__}, e01491f): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--vs-official" in a:
        sys.exit(main_official())
    sys.exit(main_mlx("--planted" in a))
