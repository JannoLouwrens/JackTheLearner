"""MLX port of DreamerV3 — rung 13b, first piece (ORDERS J25): the golden's INPUT batch rebuilt on the
box, and the ENCODER — the one module of `Agent.loss` no earlier rung ported.

Why this piece first: rung 13b is the port's own whole-loss gradients against R13's 81 official ones.
The whole loss starts from `obs` through `self.enc` (agent.py L164-165), and both were missing: the
encoder is unported (rung 9's docstring: "encoder + imagination unported"), and the golden stores the
encoder's OUTPUT (`outs/tokens`) but not its input batch. This file shows the batch can be rebuilt
exactly from `g1_probe`'s recipe and that the port's encoder reproduces the official tokens.

Official code `danijar/dreamerv3` at e01491f:
  agent.py L37-40   enc_space = obs_space minus (is_first, is_last, is_terminal, reward) -> {vector}.
  agent.py L312-340 _apply_replay_context: obs = data[k][:, K:] with K = replay_context = 1.
  rssm.py L215-224  Encoder vector branch: DictConcat(sorted keys, squish=symlog) -> reshape(-1, F);
                    `layers` x [Linear(units) -> Norm(rms) -> act]; outs reshaped to (B, T, units).
  configs.yaml      enc.simple {layers 3, units 1024 -> 64 under size1m, act silu, norm rms, symlog True}.
                    (The class default act is gelu; the CONFIG's silu is what runs — read below.)
The batch (`g1_probe.py` L128-150): np.random.default_rng(1234), one draw per key of `agent.spaces` in
its dict order = obs (vector FIRST), then act, then ext; floats uniform(lo, 1) with lo = -1 for
vector/action; then rows 0::2 get is_first at col 33 (row 2.1's `--episode-start 33`). `vector` is the
first key drawn, so its values do not depend on the order of anything after it.

THE LINE — fixed before the first read: at each of calls 1, 2, 6, 8 (encoder params = R13's `c<N>/pre`,
which (R) of rung 13a proved bit-equal to the frozen row-2.1 golden's step):
  (E1) REF_enc (float64, this file) vs golden `c<N>/outs/tokens`: every element |d| <= 1e-5 (|ref| + 1).
       A miss here means the batch or the reference is wrong, not the port.
  (E2) port `mlx_maths.encoder` vs the same golden, eager AND under mx.compile, same bound.
  (C)  control: the golden tokens of call 2 differ from call 1's (the params moved; else E1 is blind).
  (P)  planted, each must miss (E1's bound) on every call: no symlog; gelu for silu; the context step
       kept (data[:, :-1] instead of [:, 1:]); rms eps 1e-6; the bias dropped.
RE-REGISTERED after the first read (the planted CLAUSE only; E1/E2/C and the bound untouched): "every
call" becomes "every call where the plant changes the function". The first read caught no_bias on 3/4:
at call 1 every encoder bias is exactly 0 (published zero init, R13 `c1/pre/enc/mlp*/bias` max 0.0), so
dropping it IS the true encoder there — no test can catch an identical function. A plant whose
output equals the true REF's bit for bit on a call is counted NO-OP on that call; it must still be
caught on every other call, and a plant that is a no-op on every call fails the clause.
Ours beside theirs: f32 throughout (published bf16 compute); size1m widths; the toy Dummy74 batch.

  /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13b.py            (ref + port + planted)

SECOND PIECE (11-25, `--losses`, NumPy only): what the port's composed loss must reproduce, read off the
golden BEFORE any composition is written. agent.py L240: loss = sum_k mean(losses[k]) * scales[k], with
scales = configs.yaml loss_scales {rec->vector 1, rew 1, con 1, dyn 1, rep 0.1, policy 1, value 1,
repval 0.3} (rung 9 reads the same line from the pin).
THE LINE, fixed before the read:
  (L1) at calls 1/2/6/8 the golden's 8 dumped terms `c<N>/outs/losses/*` (float64 means) composed by L240
       equal R13's `c<N>/loss/total_scaled` to relative 1e-5 — i.e. the dump holds EVERY term of the loss
       the 81 gradients were taken of, and the scales are the ones above.
  (L2) `outs/losses/{policy,value}[b, t]` = the mean over the 15-step horizon of `imagloss/loss/*` row
       b*64 + t (relative 1e-5 of the term's max): the imagination losses rungs 10-12 checked are the
       ones the total carries, laid out [B*T, H] -> [B, T].
  (P)  planted, each must miss L1 on every call: rep scale 1 (for 0.1); repval dropped; repval scale 1;
       policy+value summed instead of meaned over the horizon is a layout plant on L2.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_A = "/data/jack-logs/mlx/g1v1/golden_size1m_A.npz"
R13 = "/data/jack-logs/mlx/rung13/R13.npz"
CALLS = (1, 2, 6, 8)
B, T, F, EPISODE_START, BATCH_SEED = 16, 65, 74, 33, 1234
LAYERS, ACT, EPS = 3, "silu", 1e-4
TOL = 1e-5


def rebuild_vector():
    """g1_probe's batch draw, `vector` only (the first key drawn): [B, T, 74] float32."""
    rng = np.random.default_rng(BATCH_SEED)
    return rng.uniform(-1.0, 1.0, (B, T, F)).astype(np.float32)


def obs_slice(x, keep_context=False):
    return x[:, :-1] if keep_context else x[:, 1:]          # agent.py L326/L331: rhs = x[:, K:]


def _symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _act(name, x):
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    c = np.sqrt(2.0 / np.pi)                                 # jax.nn.gelu approximate=True
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def REF_enc(P, vector, layers=LAYERS, act=ACT, eps=EPS, symlog=True, bias=True):
    x = np.asarray(vector, np.float64)
    x = _symlog(x) if symlog else x
    lead = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    for i in range(layers):
        x = x @ P[f"enc/mlp{i}/kernel"].astype(np.float64)
        if bias:
            x = x + P[f"enc/mlp{i}/bias"].astype(np.float64)
        s = P[f"enc/mlp{i}norm/scale"].astype(np.float64)
        x = x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * s
        x = _act(act, x)
    return x.reshape(*lead, -1)


def load():
    G = np.load(GOLDEN_A)
    R = np.load(R13)
    out = {}
    for c in CALLS:
        P = {k[len(f"c{c}/pre/"):]: R[k] for k in R.files if k.startswith(f"c{c}/pre/enc/")}
        out[c] = (P, G[f"c{c}/outs/tokens"])
    return out


def miss(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        return got.size or 1, float("inf")
    r = np.abs(got - ref) / (np.abs(ref) + 1.0)
    return int((r > TOL).sum()), float(r.max())


PLANTED = {
    "no_symlog": dict(symlog=False),
    "gelu_for_silu": dict(act="gelu"),
    "context_kept": "context",
    "rms_eps_1e-6": dict(eps=1e-6),
    "no_bias": dict(bias=False),
}


def main_ref(data, vec):
    ok = True
    for c, (P, gold) in data.items():
        n, w = miss(REF_enc(P, obs_slice(vec)), gold)
        print(f"E1 call {c}: REF_enc vs official tokens {gold.shape}: miss {n}, worst {w:.2e} of {TOL:g}")
        ok &= n == 0
    n, w = miss(data[2][1], data[1][1])
    print(f"C  call 2 vs call 1 golden tokens: {n} elements differ beyond the bound (must be > 0)")
    ok &= n > 0
    caught = 0
    for name, kw in PLANTED.items():
        hits, live = [], []
        for c, (P, gold) in data.items():
            got = REF_enc(P, obs_slice(vec, keep_context=True)) if kw == "context" else \
                REF_enc(P, obs_slice(vec), **kw)
            if np.array_equal(got, REF_enc(P, obs_slice(vec))):
                continue                                     # NO-OP on this call (re-registered)
            live.append(c)
            hits.append(miss(got, gold)[0] > 0)
        caught += bool(live) and all(hits)
        noop = [c for c in data if c not in live]
        print(f"P  planted {name:14s}: caught on {sum(hits)}/{len(live)} calls where it changes the "
              f"function" + (f" (NO-OP at call {noop})" if noop else ""))
    print(f"P  planted caught {caught}/{len(PLANTED)}")
    return ok and caught == len(PLANTED)


def main_port(data, vec):
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"   # mx.compile needs it (rung1.main_mlx, mlx_smoke)
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    sys.path.insert(0, HERE)
    import mlx.core as mx
    import mlx_maths as M
    if not hasattr(M, "encoder"):
        print("E2 port: mlx_maths.encoder MISSING -> FAIL")
        return False
    ok = True
    for c, (P, gold) in data.items():
        Pm = {k: mx.array(v) for k, v in P.items()}
        x = mx.array(obs_slice(vec))
        for mode in ("eager", "compiled"):
            f = (lambda p, v: M.encoder(p, v, LAYERS, ACT))
            f = mx.compile(f) if mode == "compiled" else f
            got = np.array(f(Pm, x))
            n, w = miss(got, gold)
            print(f"E2 call {c} {mode:8s}: port vs official tokens: miss {n}, worst {w:.2e} of {TOL:g}")
            ok &= n == 0
    return ok


SCALES = dict(vector=1.0, rew=1.0, con=1.0, dyn=1.0, rep=0.1, policy=1.0, value=1.0, repval=0.3)
LOSS_TOL = 1e-5


def compose(terms, scales):
    return sum(float(np.mean(terms[k])) * s for k, s in scales.items())


def main_losses():
    G, R = np.load(GOLDEN_A), np.load(R13)
    ok, plants = True, {
        "rep_scale_1": dict(SCALES, rep=1.0),
        "repval_dropped": {k: v for k, v in SCALES.items() if k != "repval"},
        "repval_scale_1": dict(SCALES, repval=1.0),
    }
    caught = {n: 0 for n in plants}
    for c in CALLS:
        terms = {k: G[f"c{c}/outs/losses/{k}"].astype(np.float64) for k in SCALES}
        tot = float(R[f"c{c}/loss/total_scaled"])
        r = abs(compose(terms, SCALES) - tot) / abs(tot)
        print(f"L1 call {c}: L240 over 8 dumped terms {compose(terms, SCALES):.6f} vs total_scaled "
              f"{tot:.6f}: rel {r:.2e} of {LOSS_TOL:g}")
        ok &= r <= LOSS_TOL
        for n, sc in plants.items():
            caught[n] += abs(compose(terms, sc) - tot) / abs(tot) > LOSS_TOL
        for k in ("policy", "value"):
            a = G[f"c{c}/imagloss/loss/{k}"].astype(np.float64)
            o = terms[k]
            scale = np.abs(o).max()
            rm = np.abs(a.mean(1).reshape(o.shape) - o).max() / scale
            rs = np.abs(a.sum(1).reshape(o.shape) - o).max() / scale
            print(f"L2 call {c} {k:6s}: horizon-mean of imagloss vs outs/losses: rel {rm:.2e}; "
                  f"planted horizon-SUM: rel {rs:.2e}")
            ok &= rm <= LOSS_TOL
            caught.setdefault(f"sum_{k}", 0)
            caught[f"sum_{k}"] += rs > LOSS_TOL
    for n, k in caught.items():
        print(f"P  planted {n:15s}: caught {k}/{len(CALLS)}")
    allp = all(k == len(CALLS) for k in caught.values())
    print(f"RUNG 13b-losses: target {'HOLDS' if ok and allp else 'FAILS'} "
          f"(planted {sum(k == len(CALLS) for k in caught.values())}/{len(caught)})")
    return 0 if ok and allp else 1


# THIRD PIECE (11-25, `--obs`, jack-mlx): the RSSM posterior step on the golden's own per-step inputs,
# teacher-forced (no noise needed). THE LINE, fixed before the read: at calls 1/2/6/8 and every t of 64,
# port `rssm_step` on `c<N>/rssm/t<t>/in_*` (dyn params = R13 `c<N>/pre/dyn/`) vs `out/deter` and
# `out/logit` (the posterior), |d| <= 1e-5 (|ref|+1), eager AND compiled. (K) the golden chains:
# out/{deter,stoch}[t] == in_{deter,stoch}[t+1] exactly. (C) control: call 2's out/deter differs from
# call 1's. (P) planted reset-ignored must miss on every call (rows 0::2 start an episode at t 32).
# Not yet: the SAMPLES (needs the noise cursor) and the gradient — both in the composed loss.
def main_obs():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mlx.core as mx
    import mlx_maths as M
    import rung10  # noqa: F401  (TOY <- SIZE1M)
    from rung6 import TOY
    G, R = np.load(GOLDEN_A), np.load(R13)
    spec = {"action": 0}
    ok, caught, prev = True, 0, None
    for c in CALLS:
        P = {k[len(f"c{c}/pre/dyn/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/dyn/")}
        def step(P, d, s, tok, a, r):
            o = M.rssm_step(P, dict(deter=d, stoch=s), tok, {"action": a}, r, TOY, spec)
            return o["deter"], o["post"]
        for mode in ("eager", "compiled"):
            f = mx.compile(step) if mode == "compiled" else step
            n = w = 0
            for t in range(64):
                q = f"c{c}/rssm/t{t:03d}/"
                d, lg = f(P, mx.array(G[q + "in_deter"]), mx.array(G[q + "in_stoch"]), mx.array(G[q + "in_tokens"]),
                          mx.array(G[q + "in_action/action"]), mx.array(G[q + "in_reset"]))
                for got, k in ((d, "out/deter"), (lg, "out/logit")):
                    a, b = miss(np.array(got), G[q + k]); n += a; w = max(w, b)
            print(f"O  call {c} {mode:8s}: 64 steps deter+post vs official: miss {n}, worst {w:.2e} of {TOL:g}")
            ok &= n == 0
        kn = sum(int((G[f"c{c}/rssm/t{t:03d}/out/{k}"] != G[f"c{c}/rssm/t{t + 1:03d}/in_{k}"]).sum())
                 for t in range(63) for k in ("deter", "stoch"))
        print(f"K  call {c}: golden chains out[t] -> in[t+1]: {kn} unequal elements")
        ok &= kn == 0
        pm = 0
        for t in range(64):
            q = f"c{c}/rssm/t{t:03d}/"
            r = mx.zeros((16,), dtype=mx.bool_)
            d, lg = step(P, mx.array(G[q + "in_deter"]), mx.array(G[q + "in_stoch"]), mx.array(G[q + "in_tokens"]),
                         mx.array(G[q + "in_action/action"]), r)
            pm += miss(np.array(lg), G[q + "out/logit"])[0]
        print(f"P  call {c}: planted reset-ignored: miss {pm}")
        caught += pm > 0
        cur = G[f"c{c}/rssm/t010/out/deter"]
        if prev is not None and c == 2:
            diff = int((cur != prev).sum()); print(f"C  call 2 vs 1 out/deter t010: {diff} differ"); ok &= diff > 0
        prev = cur if c == 1 else prev
    allp = caught == len(CALLS)
    print(f"RUNG 13b-obs: port {'PASS' if ok and allp else 'FAIL'} (planted caught {caught}/{len(CALLS)})")
    return 0 if ok and allp else 1


# FOURTH PIECE (11-25, `--scan`, jack-mlx): the posterior NOISE CURSOR, and the port's free-running
# sampled pass from its OWN encoder. Found by search over the regenerated stream (file_noise seed 0:
# `default_rng(0).random(n, float32)`, a prefix of the kernel's 64M draw): the step-t posterior sampler of
# train call N reads uniforms [cur, cur + 2048) with
#     cur(N, t) = 2,789,888 + (N - 1) * 4,460,544 + 2,048 * t      (2,048 = B 16 x stoch 32 x classes 4)
# i.e. contiguous over the 64 real steps, the ninjax prerun's 2,048 already behind it, and the SAME stride
# of 4,460,544 uniforms a train call (observe 65 x 2,048 incl. prerun + the imagination's draws).
# The rest of the batch rebuilds too: after `vector` the recipe draws ONE [B, T] float (reward,
# uniform(0, 1)), then `action` uniform(-1, 1) [16, 65, 8]; the step-t input action is action[:, t]
# (t 0..63, the PREVIOUS action) and reset = is_first[:, 1:] (rows 0::2 at t 32).
# THE LINE, fixed before the port's read — at calls 1/2/6/8:
#   (N) NumPy sampler (log(0.99 softmax + 0.01/4) + Gumbel(u), f32) on the golden's `out/logit` with the
#       uniforms at cur(N, t) == golden `out/stoch`: 0 flips of 32,768 latents a call.
#   (A) rebuilt action[:, :-1] and is_first[:, 1:] == the golden's 64 dumped `in_action` / `in_reset`, bit for bit.
#   (S) port, FREE-RUNNING: tokens = `mlx_maths.encoder`(R13 pre enc, rebuilt obs); carry = golden t000 in_*;
#       `mlx_maths.rssm_observe` with those uniforms vs golden `outs/repfeat`: deter and logit
#       |d| <= 1e-5 (|ref|+1), stoch 0 flips; eager AND compiled.
#   (C) control: the same port pass on constant noise (u = 0.5) flips > 10% of the latents.
#   (P) planted, each must miss (S) on every call where it changes the function: noise one step late
#       (+2,048); prerun not skipped (-2,048); call 1's cursor at every call (NO-OP at call 1 by
#       construction); action drawn where the reward is (reward draw skipped); resets ignored.
# Not yet: gradients — the composed loss.
CUR0, CUR_CALL, CUR_STEP = 2_789_888, 4_460_544, 2_048
ACT_DIM = 8


def cursor(c, t=0):
    return CUR0 + (c - 1) * CUR_CALL + CUR_STEP * t


def rebuild_batch(skip_reward=False):
    rng = np.random.default_rng(BATCH_SEED)
    vec = rng.uniform(-1.0, 1.0, (B, T, F)).astype(np.float32)
    rew = None if skip_reward else rng.uniform(0.0, 1.0, (B, T)).astype(np.float32)
    act = rng.uniform(-1.0, 1.0, (B, T, ACT_DIM)).astype(np.float32)
    first = np.zeros((B, T), bool)
    first[:, 0] = True
    first[0::2, EPISODE_START] = True
    return dict(vector=vec, reward=rew, action=act, is_first=first)


def uniforms(stream, c, shift=0):
    """[T=64, B, S, C]: the uniforms the 64 real posterior sampler calls of train call c read."""
    a = cursor(c) + shift
    return stream[a:a + 64 * CUR_STEP].reshape(64, B, 32, 4)


def _np_sample(lg, u, mix=0.01):
    lg = lg.astype(np.float32)
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    p = ((1.0 - mix) * p + mix / lg.shape[-1]).astype(np.float32)
    g = -np.log(-np.log(np.clip(u, 1e-7, 1 - 1e-7))).astype(np.float32)
    return (np.log(p) + g).argmax(-1)


def main_scan():
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    sys.path.insert(0, HERE)
    import mlx.core as mx
    import mlx_maths as M
    import rung10  # noqa: F401  (TOY <- SIZE1M)
    from rung6 import TOY
    G, R = np.load(GOLDEN_A), np.load(R13)
    stream = np.random.default_rng(0).random(cursor(CALLS[-1], 64) + 2 * CUR_STEP, dtype=np.float32)
    bt = rebuild_batch()
    spec = {"action": 0}
    ok = True
    # (A)
    for c in CALLS:
        ia = np.stack([G[f"c{c}/rssm/t{t:03d}/in_action/action"] for t in range(64)], 1)
        ir = np.stack([G[f"c{c}/rssm/t{t:03d}/in_reset"] for t in range(64)], 1)
        na, nr = int((ia != bt["action"][:, :-1]).sum()), int((ir != bt["is_first"][:, 1:]).sum())
        print(f"A  call {c}: rebuilt action / reset vs the 64 dumped inputs: {na} / {nr} unequal")
        ok &= na == 0 and nr == 0
    # (N)
    for c in CALLS:
        U = uniforms(stream, c)
        fl = sum(int((_np_sample(G[f"c{c}/rssm/t{t:03d}/out/logit"], U[t])
                      != G[f"c{c}/rssm/t{t:03d}/out/stoch"].argmax(-1)).sum()) for t in range(64))
        print(f"N  call {c}: cursor {cursor(c):,}: sampler on golden logits vs golden stoch: {fl} flips of {64 * B * 32}")
        ok &= fl == 0

    def run(c, U, batch=bt, reset=True, f=None):
        Pe = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/enc/")}
        Pd = {k[len(f"c{c}/pre/dyn/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/dyn/")}
        q = f"c{c}/rssm/t000/"
        rs = batch["is_first"][:, 1:] if reset else np.zeros((B, 64), bool)
        args = (Pe, Pd, mx.array(G[q + "in_deter"]), mx.array(G[q + "in_stoch"]), mx.array(obs_slice(batch["vector"])),
                mx.array(batch["action"][:, :-1]), mx.array(rs), mx.array(U))
        o = (f or scan)(*args)
        return [np.array(x) for x in o]

    def scan(Pe, Pd, d, s, obs, act, rs, U):
        tok = M.encoder(Pe, obs, LAYERS, ACT)
        o = M.rssm_observe(Pd, dict(deter=d, stoch=s), tok, {"action": act}, rs, U, TOY, spec)
        return o["deter"], o["post"], o["stoch"]

    def score(c, out):
        d, lg, st = out
        q = f"c{c}/outs/repfeat/"
        n1, w1 = miss(d, G[q + "deter"])
        n2, w2 = miss(lg, G[q + "logit"])
        fl = int((st.argmax(-1) != G[q + "stoch"].argmax(-1)).sum())
        return n1 + n2, max(w1, w2), fl

    # (S)
    for c in CALLS:
        for mode in ("eager", "compiled"):
            n, w, fl = score(c, run(c, uniforms(stream, c), f=mx.compile(scan) if mode == "compiled" else None))
            print(f"S  call {c} {mode:8s}: port enc -> 64-step sampled pass vs official repfeat: "
                  f"miss {n}, worst {w:.2e} of {TOL:g}, flips {fl}")
            ok &= n == 0 and fl == 0
    # (C)
    n, w, fl = score(1, run(1, np.full((64, B, 32, 4), 0.5, np.float32)))
    tot = 64 * B * 32
    print(f"C  call 1 constant noise: flips {fl}/{tot} (must be > 10%)")
    ok &= fl > tot // 10
    # (P)
    plants = {
        "noise_one_step_late": lambda c: run(c, uniforms(stream, c, CUR_STEP)),
        "prerun_not_skipped": lambda c: run(c, uniforms(stream, c, -CUR_STEP)),
        "call1_cursor_always": lambda c: run(c, uniforms(stream, 1)),
        "reward_draw_skipped": lambda c: run(c, uniforms(stream, c), batch=rebuild_batch(skip_reward=True)),
        "resets_ignored": lambda c: run(c, uniforms(stream, c), reset=False),
    }
    caught = 0
    for name, fn in plants.items():
        hits, live = [], []
        for c in CALLS:
            if name == "call1_cursor_always" and c == 1:
                continue                                     # identical function at call 1
            n, w, fl = score(c, fn(c))
            live.append(c)
            hits.append(n > 0 or fl > 0)
        caught += bool(live) and all(hits)
        print(f"P  planted {name:20s}: caught on {sum(hits)}/{len(live)} calls where it changes the function")
    print(f"RUNG 13b-scan: port {'PASS' if ok and caught == len(plants) else 'FAIL'} "
          f"(planted caught {caught}/{len(plants)})")
    return 0 if ok and caught == len(plants) else 1


def main():
    if "--scan" in sys.argv:
        return main_scan()
    if "--obs" in sys.argv:
        return main_obs()
    if "--losses" in sys.argv:
        return main_losses()
    vec = rebuild_vector()
    data = load()
    r = main_ref(data, vec)
    p = main_port(data, vec)
    print(f"RUNG 13b-enc: references {'HOLD' if r else 'FAIL'}; port {'PASS' if p else 'FAIL'}")
    return 0 if (r and p) else 1


if __name__ == "__main__":
    sys.exit(main())
