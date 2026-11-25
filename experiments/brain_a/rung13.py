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

THE LINE — fixed 11-25 17:40 UTC, before the kernel runs. REFERENCES HOLD iff, at every dumped call:
  (R) REPEAT: every gradient equals the frozen row-2.1 golden's (`golden_size1m_A.npz`) bit for bit
      (same config -> the same step; else the new dump is not the golden's step: NO-READING).
  (S) STEP: for every variable with a gradient, REF64(pre weight, pre nu, pre mu, pre count, grad) ->
      post weight within 1 ulp32(|ref|) + 1e-5 |delta_ref|; post nu, mu within 1e-5 (|ref|) + 1e-30;
      the optimiser counts +1. And the call moved something: >= 1% of the weights changed.
  (W) SLOW: slowval post = rate * val POST + (1 - rate) * slowval pre, within 1 ulp32 + 1e-6 |ref|;
      slowval_count +1.
  (U) UNTOUCHED: every other variable post == pre bit for bit, except the listed bookkeeping
      (opt/step, the opt/state counts, retnorm/*: rung 12a's, its inputs are not dumped).
  (P) PLANTED wrong references, each must be CAUGHT by (S) or (W) on at least one call: global-norm
      clip; slow update from the PRE-step critic; no bias correction; momentum before rms; state not
      carried (mu = nu = 0 each call); slow mix 0.98 (rate and 1-rate swapped); lr x 10.
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
GOLDEN_A = Path("/data/jack-logs/mlx/g1v1/golden_size1m_A.npz")
NPZ = OUT / "R13.npz"
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
    u = np.float64(g)
    un = glob if glob is not None else np.linalg.norm(u.ravel())
    pn = np.linalg.norm(np.float64(p).ravel())
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
    return np.float64(p) - lr * u, nu2, mu2


def load(npz=NPZ):
    z = np.load(npz)
    calls = {}
    for c in CALLS:
        pre = {k[len(f"c{c}/pre/"):]: np.float64(z[k]) if z[k].dtype.kind == "f" else z[k]
               for k in z if k.startswith(f"c{c}/pre/")}
        post = {k[len(f"c{c}/post/"):]: np.float64(z[k]) if z[k].dtype.kind == "f" else z[k]
                for k in z if k.startswith(f"c{c}/post/")}
        grad = {k[len(f"c{c}/grad/"):]: z[k] for k in z if k.startswith(f"c{c}/grad/")}
        calls[c] = (pre, post, grad)
    return calls


def check_call(pre, post, grad, ref=None, slow_from="post", slow_rate=RATE, **kw):
    """-> dict of failures (empty = holds) and stats. `ref`/kw select a planted variant."""
    fails, moved, total = {}, 0, 0
    gn = sorted(grad)
    on = _opt_names(pre, gn)
    gl = np.sqrt(sum(np.sum(np.float64(grad[k]) ** 2) for k in gn)) if kw.pop("glob", False) else None
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
        p1, nu1, mu1 = REF_step(pre[k], nu0, mu0, count, grad[k], glob=gl, **kw)
        d = np.abs(post[k] - p1)
        if not np.all(d <= _ulp(p1) + 1e-5 * np.abs(p1 - pre[k])):
            fails[k] = float(np.max(d / (_ulp(p1) + 1e-30)))
        for nm, got, want in ((kn, post[kn], nu1), (km, post[km], mu1)):
            if not np.all(np.abs(got - want) <= 1e-5 * np.abs(want) + 1e-30):
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
        want = slow_rate * src + (1 - slow_rate) * pre[k]
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


if __name__ == "__main__":
    if sys.argv[1:2] == ["kernel"]:
        kernel()
    elif "--vs-official" in sys.argv:
        sys.exit(main_official())
    else:
        raise SystemExit(__doc__)
