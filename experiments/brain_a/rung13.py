"""MLX port of DreamerV3 — rung 13, first half (ORDERS J25): ONE WHOLE optimiser step of the official
optimiser on one golden batch, over the agent's REAL parameter tree — REFERENCES ONLY in this commit
(the port's tests + module are the next slot; rung 13b, the port's own whole-loss gradients, after).

Split, said before any run: rung 13 = (13a) the official optimiser + slow critic, applied by the agent's
own `train()` to all of its variables, reproduced from the official gradients; (13b) the port's composed
loss giving those gradients. Rungs 3 / 11 checked the optimiser chain and the slow update on toy tensors;
what only the whole step can show: which variables the optimiser owns (all 7 modules, one chain),
clipping PER TENSOR on the real tree, the slow critic mixing the POST-step critic (agent.py L139-142:
`self.opt(...)` then `self.slowval.update()`), the carried state (mu, nu, counts) across calls, and that
nothing else moves.

Official code `danijar/dreamerv3` at e01491f:
  agent.py L74-78  modules [dyn, enc, dec, rew, con, pol, val]; ONE Optimizer over all of them.
  agent.py L137-142 train(): metrics = self.opt(self.loss, ...); self.slowval.update().
  embodied/jax/opt.py Optimizer.__call__: grads = nj.grad(loss, modules); updates = opt.update(grads,
                   state, params); apply_updates; step += 1.  chain = rung 3's (`_make_opt`).
  embodied/jax/utils.py SlowModel.update: mix = rate if count % every == 0 else 0;
                   dst = mix * src + (1 - mix) * dst; count += 1.  configs.yaml slowvalue {0.02, 1}.

The golden (one Kaggle CPU kernel — an eager size1m call peaks 3.4 GB, over the box's budget): the
row-2.1 golden's run A EXACTLY (`golden_kernel.py`: seed 0, file noise seed 0, 8 calls, batch 16,
episode start col 33, warmup 0, f32, eager) plus `g1_probe --dump-params`: every agent variable
before and after calls 1, 2, 6, 8. Listed differences as row 2.1's (warmup 0 so call 1 moves the
weights; the ramp is rung 3's, PASS).

THE LINE — fixed 11-25 ~17:10 UTC in `6ad44c87`, before the kernel (`jack-cpu-1795626717`, 17:11) ran. REFERENCES HOLD iff, at every dumped call:
  (R) REPEAT: every gradient equals the frozen row-2.1 golden's (`golden_size1m_A.npz`) bit for bit
      (same config -> the same step; else the new dump is not the golden's step: NO-READING).
  (S) STEP: for every variable with a gradient, REF(pre weight, pre nu, pre mu, pre count, grad) ->
      post weight within 1 ulp32(|ref|) + 1e-5 |delta_ref|; post nu, mu within 1e-5 (|ref|) + 1e-30;
      the optimiser counts +1. And the call moved something: >= 1% of the weights changed.
  (W) SLOW: slowval post = rate * val POST + (1 - rate) * slowval pre, within 1 ulp32 + 1e-6 |ref|;
      slowval_count +1.
  (U) UNTOUCHED: every other variable post == pre bit for bit, except the listed bookkeeping
      (opt/step, the opt/state counts, retnorm/*: rung 12a's, its inputs are not dumped).
  (P) PLANTED wrong references, each must be CAUGHT by (S) or (W) on at least one call: global-norm
      clip; slow update from the PRE-step critic; no bias correction; momentum before rms; state not
      carried (mu = nu = 0 each call); slow mix 0.98 (rate and 1-rate swapped); lr x 10.
RE-REGISTERED 11-25 ~17:22 (after the read), the REFERENCE (checks and planted set unchanged; one moment tolerance, "Third" below): REF is
`REF_step32` (the official per-op float32 rounding) — the first read with REF_step's float64 arithmetic
held call 1 on every weight and missed from call 2 on, where momentum cancels (REF_step32's docstring).
Third (11-25 ~17:23, the moment TOLERANCE, the weight line untouched): nu, mu are bounded by 1e-5 x
(beta |state0| + |term|), not 1e-5 |result| — after cancellation the official f32 result carries 1-2
ulp of its 0.09-size operands (= 128-1024 ulp of a 1e-4 result: 12 / 197 / 122 of 647,705 momentum
elements at calls 2 / 6 / 8; 0 miss against the operands); the old form is unreachable by any f32
implementation, the port included. Equal to the old form wherever nothing cancels. The slow update's
reference rounds per op in f32 (as the eager golden).
Second named cause, same sitting (reference only again): eager JAX's `b ** int32(t)` in the bias
correction is square-and-multiply in f32, not NumPy's power — see `_pow_eager`.
Anything missing from the harvest -> NO-READING. A miss is a FAIL of the reference, found and fixed,
never a loosened line.

  python -m experiments.brain_a.rung13 kernel            (submits; writes OUT/rung13_golden.json + .npz)
  /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13.py --vs-official   (box, NumPy only)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = Path("/data/jack-logs/mlx/rung13")
# JACK_GOLDEN_A / JACK_R13 (11-26): a wheel kernel that regenerates the goldens points the tests at its
# copies; unset = the box paths, byte-for-byte the old behaviour.
GOLDEN_A = Path(os.environ.get("JACK_GOLDEN_A", "/data/jack-logs/mlx/g1v1/golden_size1m_A.npz"))
NPZ = Path(os.environ.get("JACK_R13", str(OUT / "R13.npz")))
CALLS = (1, 2, 6, 8)
LR, AGC, EPS, B1, B2, PMIN = 4e-5, 0.3, 1e-20, 0.9, 0.999, 1e-3
RATE = 0.02
BOOKKEEPING = ("opt/step", "retnorm/")


# ---------------------------------------------------------------- the kernel

def job_body() -> str:
    sys.path.insert(0, os.path.dirname(HERE))
    from experiments.brain_a import golden_kernel as gk
    rep = {"__PINS__": repr(gk.PINS), "__COMMIT__": repr(gk.COMMIT),
           "__RUNS__": repr([("A13", ["--seed", "0", "--dump-params"])]),
           "__STEPS__": repr(gk.STEPS), "__NOISE_N__": repr(gk.NOISE_N), "__DUMP__": repr(gk.DUMP),
           "__BATCH__": repr(gk.BATCH), "__EP_COL__": repr(gk.EP_COL)}
    s = gk._JOB
    for k, v in rep.items():
        s = s.replace(k, v)
    # keep only what rung 13 reads (grads, pre/post variables, loss): the full dump is ~320 MB
    s = s.replace('print("DONE", flush=True)', '''z = np.load(os.path.join(out, "A13.npz"))
keep = {k: z[k] for k in z if "/grad/" in k or "/pre/" in k or "/post/" in k or k.endswith("loss/total_scaled")}
np.savez(os.path.join(out, "R13.npz"), **keep)
res["r13_keys"] = len(keep)
json.dump(res, open(os.path.join(out, "golden_runs.json"), "w"))
print("DONE", flush=True)''')
    return s


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle-cpu", est_hours=1.0,
                 timeout_s=int(4 * 3600), fetch=["golden_runs.json", "R13.npz", "A13.json"])
    if not sub.ok:
        raise RuntimeError(f"rung13 kernel: {sub.message}")
    r = result_json(sub, "golden_runs.json")
    OUT.mkdir(parents=True, exist_ok=True)
    npz = sub.artifacts.get("R13.npz")
    if npz:
        import shutil
        shutil.copy(npz, NPZ)
    r["npz"] = str(NPZ) if npz else None
    (OUT / "rung13_golden.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: r.get(k) for k in ("npz", "r13_keys", "install_s")}), flush=True)
    return r


# ---------------------------------------------------------------- reference (float64)

def _ulp(x):
    return np.spacing(np.abs(x).astype(np.float32)).astype(np.float64)


def _opt_names(pre, gnames):
    """official optax state: tuple chain -> 'opt/state/<i>/<j>/<param name>'; find nu (rms) and mu
    (momentum) by the param name suffix; counts are the scalar leaves."""
    by = {}
    for k in pre:
        if k.startswith("opt/state/"):
            for g in gnames:
                if k.endswith("/" + g):
                    by.setdefault(g, []).append(k)
    return by


def REF_step(p, nu, mu, count, g, lr=LR, agc=AGC, eps=EPS, beta1=B1, beta2=B2, glob=None,
             bias=True, order="rms_mom"):
    """rung 3's chain (rung3.REF_opt) from a GIVEN state, one step, float64; theirs' f32 casts kept."""
    f32 = np.float32
    m1, m2 = float(f32(beta1)), float(f32(beta2))
    a1, a2 = float(f32(1 - beta1)), float(f32(1 - beta2))
    step = count + 1
    bc = (lambda b: 1.0 - float(f32(f32(b) ** np.int32(step)))) if bias else (lambda b: 1.0)
    u = np.asarray(g, np.float64)
    un = glob if glob is not None else np.linalg.norm(u.ravel())
    pn = np.linalg.norm(np.asarray(p, np.float64).ravel())
    u = u / max(1.0, un / (agc * max(PMIN, pn)))
    if order == "rms_mom":
        nu2 = m2 * nu + a2 * u * u
        u = u / (np.sqrt(nu2 / bc(beta2)) + eps)
        mu2 = m1 * mu + a1 * u
        u = mu2 / bc(beta1)
    else:   # planted: momentum first
        mu2 = m1 * mu + a1 * u
        u = mu2 / bc(beta1)
        nu2 = m2 * nu + a2 * u * u
        u = u / (np.sqrt(nu2 / bc(beta2)) + eps)
    return np.asarray(p, np.float64) - lr * u, nu2, mu2


def _pow_eager(b, t):
    """`b ** jnp.int32(t)` as EAGER JAX computes it (the golden runs with jit disabled): binary
    exponentiation in float32 — 78/78 exact for b in (0.999, 0.9), t 1..39 (box, jax 0.4.33, 11-25);
    exp(t log b) 26/78, repeated multiply 18/78. NumPy's float32 power (correctly rounded) differs at
    t = 3, 6, 7, 8 for 0.999 and t = 7 for 0.9, moving 1/(1 - b^t) by up to 2e-5 — and that is what
    optax gives UNDER jit (the box: `optax.bias_correction` jitted == NumPy 16/16, eager != at 5/16).
    So the port must follow the JITTED value when it trains; this golden is read with the eager one."""
    f = np.float32
    r, x, n = f(1), f(b), t
    while n:
        if n & 1:
            r = f(r * x)
        x = f(x * x)
        n >>= 1
    return r


def REF_step32(p, nu, mu, count, g, lr=LR, agc=AGC, eps=EPS, beta1=B1, beta2=B2, glob=None,
               bias=True, order="rms_mom"):
    """The same chain with the OFFICIAL per-op float32 rounding (the golden is eager: every jnp op
    rounds). Why (measured 11-25 on `jack-cpu-1795626717`): from call 2 on, momentum cancels
    (0.9 * 0.09999 + 0.1 * -0.0891 = 8.6e-4) and the official f32 sum is off the f64 one by ~1 ulp of
    the 0.09 TERMS = 1.4e-5 of the result; REF_step (f64 arithmetic) then misses the line on 286 /
    26,080 / 16,067 moment elements at calls 2 / 6 / 8 and on 413 / 429 weights through them. Call 1
    (no cancellation) held on all 647,705 weights with either reference. Op order, opt.py at e01491f:
    agc: u * (1 / max(1, |u| / (clip max(pmin, |p|))));  rms: nu = b2 v + (1 - b2)(u u),
    nu_hat = nu / (1 - b2^t), u / (sqrt(nu_hat) + eps);  momentum (optax.update_moment):
    mu = (1 - b1) u + b1 mu, mu_hat = mu / (1 - b1^t);  scale_by_learning_rate: -lr * u;
    apply_updates: p + u. Norms in float64 then cast (jnp.linalg.norm's order is not NumPy's)."""
    f = np.float32
    p32, u = np.asarray(p, f), np.asarray(g, f)
    nu, mu = np.asarray(nu, f), np.asarray(mu, f)
    step = np.int32(count + 1)
    bc = (lambda b: f(1) - _pow_eager(b, int(step))) if bias else (lambda b: f(1))
    un = glob if glob is not None else np.linalg.norm(np.asarray(g, np.float64).ravel())
    pn = np.linalg.norm(np.asarray(p, np.float64).ravel())
    upper = f(agc) * f(max(PMIN, f(pn)))
    u = u * (f(1) / f(max(f(1), f(f(un) / upper))))
    b1, b2, a1, a2 = f(beta1), f(beta2), f(1 - beta1), f(1 - beta2)

    def rms(u, nu):
        nu2 = b2 * nu + a2 * (u * u)
        return u / (np.sqrt(nu2 / bc(beta2)) + f(eps)), nu2

    def mom(u, mu):
        mu2 = a1 * u + b1 * mu
        return mu2 / bc(beta1), mu2
    if order == "rms_mom":
        u, nu2 = rms(u, nu)
        u, mu2 = mom(u, mu)
    else:
        u, mu2 = mom(u, mu)
        u, nu2 = rms(u, nu)
    upd = f(-lr) * u
    return (np.asarray(p32 + upd, np.float64), np.asarray(nu2, np.float64),
            np.asarray(mu2, np.float64))


def load(npz=NPZ):
    z = np.load(npz)
    calls = {}
    for c in CALLS:
        pre = {k[len(f"c{c}/pre/"):]: np.asarray(z[k], np.float64) if z[k].dtype.kind == "f" else z[k]
               for k in z if k.startswith(f"c{c}/pre/")}
        post = {k[len(f"c{c}/post/"):]: np.asarray(z[k], np.float64) if z[k].dtype.kind == "f" else z[k]
                for k in z if k.startswith(f"c{c}/post/")}
        grad = {k[len(f"c{c}/grad/"):]: z[k] for k in z if k.startswith(f"c{c}/grad/")}
        calls[c] = (pre, post, grad)
    return calls


def check_call(pre, post, grad, ref=None, slow_from="post", slow_rate=RATE, **kw):
    """-> dict of failures (empty = holds) and stats. `ref`/kw select a planted variant."""
    fails, moved, total = {}, 0, 0
    gn = sorted(grad)
    on = _opt_names(pre, gn)
    gl = np.sqrt(sum(np.sum(np.asarray(grad[k], np.float64) ** 2) for k in gn)) if kw.pop("glob", False) else None
    carried = kw.pop("carried", True)
    counts = sorted(k for k in pre if k.startswith("opt/state/") and pre[k].shape == ())
    count = int(pre[counts[0]]) if counts else None
    for k in gn:
        if k not in pre or len(on.get(k, [])) != 2:
            fails[k] = f"missing pre or opt state ({on.get(k)})"
            continue
        # chain order is agc(0) -> rms(1) -> momentum(2): the lower state index is nu
        kn, km = sorted(on[k])
        nu0, mu0 = (pre[kn], pre[km]) if carried else (0 * pre[kn], 0 * pre[km])
        p1, nu1, mu1 = (ref or REF_step32)(pre[k], nu0, mu0, count, grad[k], glob=gl, **kw)
        d = np.abs(post[k] - p1)
        if not np.all(d <= _ulp(p1) + 1e-5 * np.abs(p1 - pre[k])):
            fails[k] = float(np.max(d / (_ulp(p1) + 1e-30)))
        # moment = beta * state0 + term; the bound is on the OPERANDS (a cancelled f32 sum carries
        # ~1 ulp of its terms, any implementation) — re-registered 11-25, see THE LINE
        for nm, got, want, s0, b in ((kn, post[kn], nu1, nu0, B2), (km, post[km], mu1, mu0, B1)):
            scale = b * np.abs(s0) + np.abs(want - b * s0)
            if not np.all(np.abs(got - want) <= 1e-5 * scale + 1e-30):
                fails[nm] = float(np.max(np.abs(got - want) / (np.abs(want) + 1e-30)))
        moved += int(np.sum(post[k] != pre[k]))
        total += pre[k].size
    for k in counts:
        if int(post[k]) != int(pre[k]) + 1:
            fails[k] = f"count {pre[k]} -> {post[k]}"
    # slow critic
    sv = sorted(k for k in pre if k.startswith("slowval/"))
    for k in sv:
        src = (post if slow_from == "post" else pre)["val/" + k[len("slowval/"):]]
        f = np.float32   # SlowModel.update, eager: mix * src + (1 - mix) * dst, each op in f32
        want = np.asarray(f(slow_rate) * src.astype(f) + (f(1) - f(slow_rate)) * pre[k].astype(f),
                          np.float64)
        if not np.all(np.abs(post[k] - want) <= _ulp(want) + 1e-6 * np.abs(want)):
            fails[k] = float(np.max(np.abs(post[k] - want)))
    sc = [k for k in pre if k.startswith("slowval_count")]
    for k in sc:
        if int(post[k]) != int(pre[k]) + 1:
            fails[k] = "slowval count"
    owned = set(gn) | set(sv) | set(sc) | {x for v in on.values() for x in v} | set(counts)
    for k in pre:
        if k in owned or any(k.startswith(b) for b in BOOKKEEPING):
            continue
        if k not in post or not np.array_equal(pre[k], post[k]):
            fails[k] = "UNLISTED variable changed"
    return dict(fails=fails, moved=moved / max(total, 1), n_grad=len(gn), n_slow=len(sv),
                count=count)


PLANTED = {
    "global_norm_clip": dict(glob=True),
    "slow_from_pre_step_critic": dict(slow_from="pre"),
    "no_bias_correction": dict(bias=False),
    "momentum_before_rms": dict(order="mom_rms"),
    "state_not_carried": dict(carried=False),
    "slow_mix_swapped": dict(slow_rate=1 - RATE),
    "lr_x10": dict(lr=10 * LR),
}


def main_official() -> int:
    if not NPZ.exists() or not GOLDEN_A.exists():
        print("NO-READING: missing", NPZ if not NPZ.exists() else GOLDEN_A)
        return 2
    calls = load()
    G = np.load(GOLDEN_A)
    ok = True
    for c in CALLS:
        pre, post, grad = calls[c]
        if not pre or not post or not grad:
            print(f"NO-READING: call {c} not in the dump")
            return 2
        gk = [k for k in G if k.startswith(f"c{c}/grad/")]
        rep = len(gk) == len(grad) and all(np.array_equal(G[k], grad[k[len(f"c{c}/grad/"):]]) for k in gk)
        print(f"(R) call {c}: grads {len(grad)} vs golden {len(gk)} bit-equal={rep}")
        if not rep:
            print("NO-READING: the dump is not the frozen golden's step")
            return 2
        r64 = check_call(pre, post, grad, ref=REF_step)
        print(f"    f64-arithmetic reference (REF_step, not the line's): {len(r64['fails'])} tensors miss")
        r = check_call(pre, post, grad)
        hold = not r["fails"] and r["moved"] >= 0.01
        ok &= hold
        print(f"(S/W/U) call {c}: count {r['count']}, {r['n_grad']} trained, {r['n_slow']} slow, "
              f"moved {r['moved']:.3f} -> {'HOLD' if hold else 'MISS'}",
              "" if hold else json.dumps(dict(list(r["fails"].items())[:12])))
    caught = 0
    for name, kw in PLANTED.items():
        hit = [c for c in CALLS if check_call(*calls[c], **dict(kw))["fails"]]
        caught += bool(hit)
        print(f"(P) planted {name}: {'CAUGHT at calls ' + str(hit) if hit else 'MISSED'}")
    print(f"PLANTED {caught}/{len(PLANTED)}")
    ok &= caught == len(PLANTED)
    print("RUNG 13a REFERENCES", "HOLD" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- the PORT vs the golden (13a, port side)

def port_check(M, mx, compiled, calls, eager_pow=True):
    """The port's `opt_update` + `slow_update` (mlx_maths, rung 3's functions, unchanged) from each dumped
    call's PRE tree + the official grads -> compared with the official POST by THE LINE's (S)/(W) clauses
    (weights 1 ulp32 + 1e-5 |delta|, moments on the operands, slow 1 ulp32 + 1e-6 |ref|). Ours beside
    theirs: the port's bias correction is mx.power (= the JITTED optax value the agent trains with), the
    golden ran EAGER (square-and-multiply, `_pow_eager`) — the difference is named, not hidden: rows
    report it per call. -> {call: dict(fails, worst)}."""
    f32 = np.float32
    out = {}
    for c in CALLS:
        pre, post, grad = calls[c]
        gn = sorted(grad)
        on = _opt_names(pre, gn)
        cnt = sorted(k for k in pre if k.startswith("opt/state/") and pre[k].shape == ())
        count = int(pre[cnt[0]])
        P = {k: mx.array(pre[k].astype(f32)) for k in gn}
        G = {k: mx.array(np.asarray(grad[k], f32)) for k in gn}
        NU = {k: mx.array(pre[sorted(on[k])[0]].astype(f32)) for k in gn}
        MU = {k: mx.array(pre[sorted(on[k])[1]].astype(f32)) for k in gn}
        st = (mx.array(count, mx.int32), NU, MU)
        bias = ((mx.array(_pow_eager(B1, count + 1)), mx.array(_pow_eager(B2, count + 1)))
                if eager_pow else None)
        fn = lambda p, g, s: M.opt_update(p, g, s, lr=LR, warmup=0, agc=AGC, eps=EPS,  # noqa: E731
                                          beta1=B1, beta2=B2, bias=bias)
        if compiled:
            fn = mx.compile(fn)
        newp, (step, nnu, nmu), _ = fn(P, G, st)
        mx.eval(newp, nnu, nmu)
        fails, worst = {}, 0.0
        if int(step) != count + 1:
            fails["step"] = f"{count} -> {int(step)}"
        for k in gn:
            got = np.asarray(newp[k], np.float64)
            d = np.abs(got - post[k])
            lim = _ulp(post[k]) + 1e-5 * np.abs(post[k] - pre[k])
            r = float(np.max(d / (lim + 1e-30)))
            worst = max(worst, r)
            if not np.all(d <= lim):
                fails[k] = r
            kn, km = sorted(on[k])
            for nm, g2, s0, b in ((kn, nnu[k], pre[kn], B2), (km, nmu[k], pre[km], B1)):
                want = post[nm]
                scale = b * np.abs(s0) + np.abs(want - b * s0)
                if not np.all(np.abs(np.asarray(g2, np.float64) - want) <= 1e-5 * scale + 1e-30):
                    fails[nm] = "moment"
        sv = sorted(k for k in pre if k.startswith("slowval/"))
        sc = [k for k in pre if k.startswith("slowval_count")]
        scount = mx.array(int(pre[sc[0]]), mx.int32) if sc else mx.array(0, mx.int32)
        for k in sv:
            vk = "val/" + k[len("slowval/"):]
            src = newp[vk] if vk in newp else mx.array(post[vk].astype(f32))
            new, _ = M.slow_update(src, mx.array(pre[k].astype(f32)), scount, RATE, 1)
            got = np.asarray(new, np.float64)
            want = post[k]
            if not np.all(np.abs(got - want) <= _ulp(want) + 1e-6 * np.abs(want)):
                fails[k] = float(np.max(np.abs(got - want)))
        out[c] = dict(fails=fails, worst=worst, n=len(gn), n_slow=len(sv))
    return out


def main_port() -> int:
    if not NPZ.exists():
        print("NO-READING: missing", NPZ)
        return 2
    sys.path.insert(0, HERE)
    from rung3 import _mlx
    mx = _mlx()
    import mlx_maths as M
    calls = load()
    ok = True
    for mode in (False, True):
        res = port_check(M, mx, mode, calls)
        for c, r in res.items():
            print(f"{'compiled' if mode else 'eager   '} call {c}: {len(r['fails'])} fail / {r['n']} grads "
                  f"+ {r['n_slow']} slow; worst weight d/line {r['worst']:.3g}"
                  + (f"  e.g. {list(r['fails'].items())[:3]}" if r['fails'] else ""))
            ok &= not r["fails"]
    # the port's OWN bias correction (mx.power) must equal the JITTED official = NumPy's f32 power
    # (rung 3's golden is jitted and holds); here it is shown against NumPy and, for the record, the
    # eager golden read with it (expected to miss only where eager != jitted: t = 3, 6, 7, 8)
    pw = [(b, t) for b in (B1, B2) for t in range(1, 40)
          if float(mx.power(mx.array(b, mx.float32), mx.array(float(t), mx.float32)))
          != float(np.float32(np.float64(np.float32(b)) ** t))]   # correctly rounded f32 power
    print(f"port mx.power == NumPy/jitted f32 power for t 1..39: {'yes' if not pw else 'NO ' + str(pw[:4])}")
    ok &= not pw
    res = port_check(M, mx, False, calls, eager_pow=False)
    print("record, port's own (jitted) bias correction vs the eager golden: "
          + ", ".join(f"call {c} {len(r['fails'])} miss" for c, r in res.items()))
    print(f"RUNG 13a PORT (mlx {mx.__version__})", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif "--vs-official" in sys.argv:
        sys.exit(main_official())
    elif "--port" in sys.argv:
        sys.exit(main_port())
    else:
        raise SystemExit(__doc__)
