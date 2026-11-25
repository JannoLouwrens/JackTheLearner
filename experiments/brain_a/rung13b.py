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


# FIFTH PIECE (11-25, `--wm`): the five world-model loss terms on the port's own free-running pass.
# Official agent.py L164-181 + L51-58: dyn/rep = rssm.loss over the observe pass (free 1.0, unimix 0.01);
# vector = dec(repfeat) symlog-MSE vs obs; rew = TwoHot(255 symexp bins) vs reward[:, 1:] (the rebuilt
# SECOND draw); con = Binary vs (1 - is_terminal[:, 1:]) * (1 - 1/333), is_terminal all False here.
# THE LINE, fixed before the first read: at calls 1/2/6/8, eager AND compiled, each of dyn, rep, vector,
#   rew, con [B=16, T=64] from the port vs golden `c<N>/outs/losses/<k>`: every element |d| <= 1e-5 (|ref|+1)
#   (W). (C) control: rew on the FIRST draw (vector's slot) instead of the reward draw misses rew on
#   every call. (P) planted, each must miss its term on every call: free bits 0 (dyn/rep), unimix 0
#   (dyn/rep), reward not sliced [:, :-1] (rew), con target without the horizon discount (con), vector
#   target obs[:, :-1] (the context step kept, one step early) (vector).
# RE-REGISTERED after the FIRST READ (which FAILED: C 3/4, planted 3/5; W untouched and 0 miss): (C) and (P)
#   read "every call where the plant changes the function" — a call where the planted terms equal the
#   true port's terms bit for bit is NO-OP there, must be caught on every other call, and a plant NO-OP on
#   every call fails. Cause, measured: at call 1 the reward head is the published zero init, so every rew
#   loss is log 255 = 5.541263 whatever the target (C and reward_unsliced: no test can see it at call 1);
#   and the golden's KL never reaches the 1.0 floor (dyn min 2.50/3.18/2.54/2.15 at calls 1/2/6/8), so
#   free_bits_0 is the identical function on EVERY call — replaced by free_bits_3 (binds at calls 1/6/8; at
#   call 2 the KL min 3.18 > 3, NO-OP there). Said plainly: this golden cannot certify the free-bits max
#   itself; rung 8a's split RSSM does. A rew plant is NO-OP where the rew head's kernel
#   and bias are all exactly 0 (logits constant -> loss target-free; f32 rounding differs, so this is
#   read off the weights, not off bit-equality); others NO-OP where the terms are bit-equal.
FREE, HORIZON = 1.0, 333
WM_TERMS = ("dyn", "rep", "vector", "rew", "con")


def main_wm():
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

    def terms(c, mix=None, free=FREE, rew_src=None, rew_slice=slice(1, None), horizon=HORIZON,
              vec_slice=slice(1, None), f=None):
        pre = f"c{c}/pre/"
        P = {k[len(pre):]: mx.array(R[k]) for k in R.files if k.startswith(pre)}
        Pd = {k[len("dyn/"):]: v for k, v in P.items() if k.startswith("dyn/")}
        cfg = dict(TOY) if mix is None else {**TOY, "unimix": mix}
        q = f"c{c}/rssm/t000/"
        rew = (bt["reward"] if rew_src is None else rew_src)[:, rew_slice]
        vec = bt["vector"][:, vec_slice]
        term = np.zeros((B, 64), bool)

        def body(d0, s0, obs, act, rs, U, rw, tm, tgt):
            tok = M.encoder(P, obs, LAYERS, ACT)
            o = M.rssm_loss(Pd, dict(deter=d0, stoch=s0), tok, {"action": act}, rs, U, cfg, spec, free)
            rc = M.rewcon_loss(P, o["deter"], o["stoch"], rw, tm, horizon)
            dv = M.dec_vec_loss(P, o["deter"], o["stoch"], {"vector": tgt}, {"vector": (F,)}, LAYERS)
            return o["dyn"], o["rep"], dv["vector"], rc["rew"], rc["con"]
        fn = mx.compile(body) if f == "compiled" else body
        out = fn(mx.array(G[q + "in_deter"]), mx.array(G[q + "in_stoch"]), mx.array(obs_slice(bt["vector"])),
                 mx.array(bt["action"][:, :-1]), mx.array(bt["is_first"][:, 1:]), mx.array(uniforms(stream, c)),
                 mx.array(rew), mx.array(term), mx.array(vec))
        return dict(zip(WM_TERMS, [np.array(x) for x in out]))

    def gold(c, k):
        return G[f"c{c}/outs/losses/{k}"]

    ok = True
    for c in CALLS:
        for mode in ("eager", "compiled"):
            got = terms(c, f=mode)
            parts = []
            for k in WM_TERMS:
                n, w = miss(got[k], gold(c, k))
                parts.append(f"{k} {n}/{w:.1e}")
                ok &= n == 0
            print(f"W  call {c} {mode:8s}: miss/worst  " + "  ".join(parts))
    first = np.random.default_rng(BATCH_SEED).uniform(-1.0, 1.0, (B, T)).astype(np.float32)
    true = {c: terms(c) for c in CALLS}

    def live_hits(kw, ks):
        hits, live = [], []
        for c in CALLS:
            got = terms(c, **kw)
            zero_head = all(not R[f"c{c}/pre/rew/head/logits/{w}"].any() for w in ("kernel", "bias"))
            if all(np.array_equal(got[k], true[c][k]) for k in ks) or (ks == ("rew",) and zero_head):
                continue                                     # identical function on this call
            live.append(c)
            hits.append(any(miss(got[k], gold(c, k))[0] > 0 for k in ks))
        return hits, live

    hits, live = live_hits(dict(rew_src=first), ("rew",))
    print(f"C  reward from the wrong draw: rew missed on {sum(hits)}/{len(live)} calls where it changes the "
          f"function (no-op at {[c for c in CALLS if c not in live]})")
    ok &= bool(live) and all(hits)
    plants = {
        "free_bits_3": (dict(free=3.0), ("dyn", "rep")),
        "unimix_0": (dict(mix=0.0), ("dyn", "rep")),
        "reward_unsliced": (dict(rew_slice=slice(0, -1)), ("rew",)),
        "con_no_discount": (dict(horizon=1e30), ("con",)),
        "vector_context_kept": (dict(vec_slice=slice(0, -1)), ("vector",)),
    }
    caught = 0
    for name, (kw, ks) in plants.items():
        hits, live = live_hits(kw, ks)
        caught += bool(live) and all(hits)
        print(f"P  planted {name:20s}: caught on {sum(hits)}/{len(live)} calls where it changes the function")
    good = ok and caught == len(plants)
    print(f"RUNG 13b-wm: port {'PASS' if good else 'FAIL'} (planted caught {caught}/{len(plants)})")
    return 0 if good else 1

# SIXTH PIECE (11-25, `--imag`): the IMAGINATION's two noise cursors, and the port's free-running
# 15-step imagination with its own policy. agent.py L194-198 + rssm.py `imagine`: per step the policy
# (bounded_normal: mean tanh(m), std 0.9 sigmoid(s + 2) + 0.1) SAMPLES `normal * std + mean` (file
# normals), then img_step samples the prior stoch (file uniforms, OneHot unimix 0.01).
# The cursors were FOUND by search over the regenerated noise file (uniforms = default_rng(0).random(64M,
# f32), normals = the same rng's standard_normal after them, golden_kernel NOISE_N 64M), BEFORE this line:
#   uniforms, prior step k of call N:  icur(N, k) = cur(N, 0) + (2 + k) * 131,072     (= 1024 x 32 x 4)
#     i.e. the posterior's 64 x 2,048 = 131,072, then ONE unidentified block of 131,072 (not a resample of
#     repfeat: 22,6xx flips), then 15 x 131,072; the stride's other 2,232,320 lie after the imagination.
#   normals, policy step k of call N: ncur(N, k) = 182,272 + (N - 1) * 278,528 + 8,192 k   (= 1024 x 8)
# The NumPy f64 reference already read 0 flips (15 steps x 4 calls) and |a - golden| <= 4.7e-7 with them.
# THE LINE, fixed before the port's first read: at calls 1/2/6/8 (dyn + pol params = R13 `c<N>/pre/`):
#   (K) golden imag/start == outs/repfeat flattened [B*T] (deter, stoch) exactly.
#   (U) NumPy sampler on golden imag logits at icur: 0 flips of 15 x 32,768.
#   (N) NumPy f64 policy on the golden carry + normals at ncur vs imag/act/action: |d| <= 1e-5 (|ref|+1).
#   (I) port FREE-RUNNING: golden start -> `imag_rollout` with the port's own policy (`_mlp_head_n` pol,
#       3 layers) sampling the file normals -> feat deter/logit [:, 1:] and action [:, :15] vs golden
#       imag/feat + imag/act: 0 miss (1e-5), stoch 0 flips, eager AND compiled.
#   (C) control: zero normals (the policy's mean action) misses action on every call.
#   (P) planted, each must miss (I) on every call where it changes the function: the unidentified block
#       not skipped (icur - 131,072); normals one step late (+8,192); std without the +2 shift; prior
#       sampled without unimix; the posterior's own cursor (call N's cur(N, 0)) for the prior.
# Ours beside theirs: f32 (published bf16 compute); the 16th policy call (agent.py L196-198, the action at
# the last feat) is fed the next 8,192 normals and not compared — the golden dumps 15 actions.
IMAG_H, ICUR_BLOCK, NCUR0, NCUR_CALL, NCUR_STEP, NOISE_N = 15, 131_072, 182_272, 278_528, 8_192, 64_000_000


def icur(c, k=0):
    return cursor(c) + (2 + k) * ICUR_BLOCK


def ncur(c, k=0):
    return NCUR0 + (c - 1) * NCUR_CALL + NCUR_STEP * k


def normal_stream(n):
    rng = np.random.default_rng(0)
    for _ in range(NOISE_N // 1_000_000):
        rng.random(1_000_000, dtype=np.float32)
    return rng.standard_normal(n, dtype=np.float32)


def _np_policy(R, c, deter, stoch, shift=2.0):
    P = lambda k: R[f"c{c}/pre/pol/{k}"].astype(np.float64)  # noqa: E731
    x = np.concatenate([deter, stoch.reshape(len(stoch), -1)], -1).astype(np.float64)
    for i in range(3):
        x = x @ P(f"mlp/linear{i}/kernel") + P(f"mlp/linear{i}/bias")
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-4) * P(f"mlp/norm{i}/scale")
        x = x / (1 + np.exp(-x))
    m = x @ P("head/action/mean/kernel") + P("head/action/mean/bias")
    s = x @ P("head/action/stddev/kernel") + P("head/action/stddev/bias")
    return np.tanh(m), 0.9 / (1 + np.exp(-(s + shift))) + 0.1


def main_imag():
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
    U_all = np.random.default_rng(0).random(icur(CALLS[-1], IMAG_H) + ICUR_BLOCK, dtype=np.float32)
    N_all = normal_stream(ncur(CALLS[-1], IMAG_H + 2))
    N1 = B * 64
    ok = True

    def U(c, base=None):
        a = icur(c) if base is None else base
        return U_all[a:a + IMAG_H * ICUR_BLOCK].reshape(IMAG_H, N1, 32, 4)

    def NZ(c, shift=0):
        a = ncur(c) + shift
        return N_all[a:a + (IMAG_H + 1) * NCUR_STEP].reshape(IMAG_H + 1, N1, 8)

    for c in CALLS:
        g = f"c{c}/imag/"
        nk = int((G[g + "start/deter"] != G[f"c{c}/outs/repfeat/deter"].reshape(N1, -1)).sum()
                 + (G[g + "start/stoch"] != G[f"c{c}/outs/repfeat/stoch"].reshape(N1, 32, 4)).sum())
        uu = U(c)
        fl = sum(int((_np_sample(G[g + "feat/logit"][:, k], uu[k]) != G[g + "feat/stoch"][:, k].argmax(-1)).sum())
                 for k in range(IMAG_H))
        D = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"][:, :-1]], 1)
        S = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"][:, :-1]], 1)
        nz, wn = 0, 0.0
        for k in range(IMAG_H):
            mean, std = _np_policy(R, c, D[:, k], S[:, k])
            n, w = miss(mean + std * NZ(c)[k], G[g + "act/action"][:, k])
            nz, wn = nz + n, max(wn, w)
        print(f"K  call {c}: imag/start vs repfeat flattened: {nk} unequal")
        print(f"U  call {c}: icur {icur(c):,}: NumPy sampler on golden imag logits: {fl} flips of {IMAG_H * N1 * 32}")
        print(f"N  call {c}: ncur {ncur(c):,}: NumPy policy + normals vs imag/act: miss {nz}, worst {wn:.2e} of {TOL:g}")
        ok &= nk == 0 and fl == 0 and nz == 0
    spec = {"action": 0}

    def make(shift=2.0, mix=0.01):
        cfg = dict(TOY, unimix=mix)

        def f(Pd, Pp, d, s, u, nz):
            i = [0]

            def pol(carry):
                x = M.feat2tensor(carry["deter"], carry["stoch"])
                m = M._mlp_head_n(Pp, "pol", x, "action/mean", 3)
                sd = M._mlp_head_n(Pp, "pol", x, "action/stddev", 3)
                std = 0.9 * mx.sigmoid(sd + shift) + 0.1
                a = mx.tanh(m) + std * nz[i[0]]
                i[0] += 1
                return {"action": a}
            o = M.imag_rollout(Pd, dict(deter=d, stoch=s), pol, u, cfg, spec, IMAG_H)
            return o["feat"]["deter"], o["feat"]["logit"], o["feat"]["stoch"], o["act"]["action"]
        return f

    def run(c, f, u, nz, compiled=False):
        Pd = {k[len(f"c{c}/pre/dyn/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/dyn/")}
        Pp = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/pol/")}
        g = f"c{c}/imag/"
        f = mx.compile(f) if compiled else f
        o = f(Pd, Pp, mx.array(G[g + "start/deter"]), mx.array(G[g + "start/stoch"]), mx.array(u), mx.array(nz))
        return [np.array(x) for x in o]

    def score(c, out):
        d, lg, st, a = out
        g = f"c{c}/imag/"
        n1, w1 = miss(d[:, 1:], G[g + "feat/deter"])
        n2, w2 = miss(lg[:, 1:], G[g + "feat/logit"])
        n3, w3 = miss(a[:, :IMAG_H], G[g + "act/action"])
        fl = int((st[:, 1:].argmax(-1) != G[g + "feat/stoch"].argmax(-1)).sum())
        return n1 + n2 + n3, max(w1, w2, w3), fl, n3

    base = make()
    for c in CALLS:
        for mode in ("eager", "compiled"):
            n, w, fl, _ = score(c, run(c, base, U(c), NZ(c), mode == "compiled"))
            print(f"I  call {c} {mode:8s}: port free-running imagination vs golden imag: miss {n}, "
                  f"worst {w:.2e} of {TOL:g}, flips {fl}")
            ok &= n == 0 and fl == 0
    for c in CALLS:
        _, _, _, na = score(c, run(c, base, U(c), np.zeros_like(NZ(c))))
        print(f"C  call {c}: zero normals: action miss {na}")
        ok &= na > 0
    plants = {
        "block_not_skipped": lambda c: run(c, base, U(c, icur(c) - ICUR_BLOCK), NZ(c)),
        "normals_one_step_late": lambda c: run(c, base, U(c), NZ(c, NCUR_STEP)),
        "std_no_shift": lambda c: run(c, make(shift=0.0), U(c), NZ(c)),
        "prior_no_unimix": lambda c: run(c, make(mix=0.0), U(c), NZ(c)),
        "posterior_cursor": lambda c: run(c, base, U(c, cursor(c)), NZ(c)),
    }
    caught = 0
    for name, fn in plants.items():
        hits = []
        for c in CALLS:
            n, w, fl, _ = score(c, fn(c))
            hits.append(n > 0 or fl > 0)
        caught += all(hits)
        print(f"P  planted {name:22s}: caught on {sum(hits)}/{len(CALLS)} calls")
    good = ok and caught == len(plants)
    print(f"RUNG 13b-imag: port {'PASS' if good else 'FAIL'} (planted caught {caught}/{len(plants)})")
    return 0 if good else 1


# SEVENTH PIECE (11-25, `--imagloss`): imag_loss's heads, returns, value and policy terms on the golden's
# imagined trajectory. agent.py L196-206 + imag_loss L396-416 (published: contdisc True -> disc 1, slowtar
# False -> tarval = val.pred(), lam 0.95, slowreg 1.0, actent 3e-4, retnorm perc 5/95 rate 0.01 limit 1,
# valnorm/advnorm 'none'). feat16 = [imag/start ; imag/feat] (TEACHER-FORCED on the golden's own feats; the
# port's free-running imagination already matches them to 2.9e-6, `--imag`). Params = R13 `c<N>/pre/`.
# THE LINE, fixed before the first read — at calls 1/2/6/8, eager AND compiled, |d| <= 1e-5 (|ref|+1):
#   (H) port rew head twohot pred and con head sigmoid on feat16 vs imagloss/in_rew, in_con.
#   (R) ret = lambda_return(0, 1 - con, rew, val, val, 1, 0.95), rew/con = the golden's in_rew/in_con, val =
#       the port's val head pred on feat16, vs imagloss/outs/ret.
#   (V) value = (cumprod(con) * (twohot(val logits, sg[ret; 0]) + 1.0 * twohot(val logits, slowval pred)))[:, :-1]
#       vs imagloss/loss/value.
#   (Q) policy = `mlx_maths.policy_loss` (continuous key; a 1-class discrete key adds exactly 0) with adv =
#       (ret - val[:, :-1]) / scale, scale from `retnorm_perc` on PRE lo/hi updated by this ret, actions =
#       imag/act/action (the 16th, sliced off by the loss, zero) vs imagloss/loss/policy.
#   (C) control: rew zeroed misses ret on every call.
#   (P) planted, each must miss its term on every call where it changes the function (bit-unequal to the
#       true port's term; NO-OP on every call fails): lam 1.0 (ret); disc 1 - 1/333 (ret); slowreg 0 (value);
#       actent 0 (policy); retnorm not updated (policy); weight without the cumprod (value).
# FIRST READ (11-25 20:33) FAILED, recorded as is: calls 1/2 0 miss on H/R/V/Q (worst 6.3e-7), eager+compiled;
#   calls 6/8 miss H rew 24/46 of 16,384 (con 0), and R/V/Q downstream; C 3/4 (call 1 NO-OP: rew head zero-init);
#   plants 4/6 (retnorm_not_updated NO-OP on every call — scale sits at the limit 1 both ways; disc_horizon 2/3).
#   CAUSE, measured (`rung13b_decode_probe.py`): from call 6 the rew head's top bin holds p ~ 0.0039 at bin
#   symexp(20) = 4.85e8, so one f32 ulp of p moves the twohot decode by 0.226; the misses are +-0.58 / 0.375 /
#   0.625 jumps on EITHER side. Against an f64 decode of the same logits, port and official f32 are equally far
#   (> 1e-5 on 4,517 vs 4,524 elements at call 6; 5,180 vs 5,191 at call 8). The element-wise 1e-5 line vs an f32
#   golden cannot be met by any f32 decode not summing in the official order. The port is not shown wrong; the
#   line is not met. Next: re-register H/R/Q against the f64 truth (port no farther than the official f32 is),
#   and replace the NO-OP plant (retnorm limit 0).
def main_imagloss():
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    sys.path.insert(0, HERE)
    import mlx.core as mx
    import mlx_maths as M
    G, R = np.load(GOLDEN_A), np.load(R13)
    b = M.bins(255)

    def make(lam=0.95, disc=1.0, slowreg=1.0, actent=3e-4, update=True, cum=True, zero_rew=False):
        def f(Pr, Pc, Pv, Ps, Pp, d, s, act, rew, con, lo, hi):
            x = M.feat2tensor(d, s)
            hr = M.twohot_decode(M._mlp_head(Pr, "rew", x, "logits"), b)
            hc = mx.sigmoid(M._mlp_head(Pc, "con", x, "logit")[..., 0])
            lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
            val = M.twohot_decode(lv, b)
            slow = mx.stop_gradient(M.twohot_decode(M._mlp_head_n(Ps, "slowval", x, "logits", 3), b))
            r = mx.zeros_like(rew) if zero_rew else rew
            ret = M.lambda_return(mx.zeros_like(con), 1.0 - con, r, val, val, disc, lam)
            w = mx.cumprod(disc * con, 1) / disc if cum else con
            w = mx.stop_gradient(w)
            tar = mx.stop_gradient(mx.concatenate([ret, 0 * ret[:, -1:]], 1))
            value = (w * (M.twohot_loss(lv, tar, b) + slowreg * M.twohot_loss(lv, slow, b)))[:, :-1]
            if update:
                _, _, off, sc = M.retnorm_perc(lo, hi, ret, 0.01, 1.0, 5.0, 95.0)
            else:
                sc = mx.stop_gradient(mx.maximum(mx.array(1.0, mx.float32), hi - lo))
            adv = M.advantage(ret, val, sc)
            m = M._mlp_head_n(Pp, "pol", x, "action/mean", 3)
            sd = M._mlp_head_n(Pp, "pol", x, "action/stddev", 3)
            z = mx.zeros((*m.shape[:-1], 1), mx.float32)
            pol, _ = M.policy_loss(m, sd, z, act, mx.zeros(m.shape[:-1], mx.int32), adv, w, 0.1, 1.0, actent)
            return hr, hc, ret, value, pol
        return f

    def run(c, f, compiled=False):
        P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa: E731
        g = f"c{c}/imag/"
        d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
        s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
        a = np.concatenate([G[g + "act/action"], np.zeros_like(G[g + "act/action"][:, :1])], 1)
        A = mx.array
        lo, hi = A(R[f"c{c}/pre/retnorm/lo/value"]).astype(mx.float32), A(R[f"c{c}/pre/retnorm/hi/value"]).astype(mx.float32)
        f = mx.compile(f) if compiled else f
        o = f(P("rew"), P("con"), P("val"), P("slowval"), P("pol"), A(d), A(s), A(a),
              A(G[f"c{c}/imagloss/in_rew"]), A(G[f"c{c}/imagloss/in_con"]), lo, hi)
        return [np.array(x) for x in o]

    def score(c, out):
        g = f"c{c}/imagloss/"
        hr, hc, ret, value, pol = out
        r = dict(H=[miss(hr, G[g + "in_rew"]), miss(hc, G[g + "in_con"])], R=[miss(ret, G[g + "outs/ret"])],
                 V=[miss(value, G[g + "loss/value"])], Q=[miss(pol, G[g + "loss/policy"])])
        return {k: (sum(n for n, _ in v), max(w for _, w in v)) for k, v in r.items()}

    ok = True
    base = make()
    true = {}
    for c in CALLS:
        for mode in ("eager", "compiled"):
            out = run(c, base, mode == "compiled")
            true.setdefault(c, out)
            sc = score(c, out)
            print(f"   call {c} {mode:8s}: " + "  ".join(f"{k} miss {n} worst {w:.2e}" for k, (n, w) in sc.items()))
            ok &= all(n == 0 for n, _ in sc.values())
    for c in CALLS:
        n, _ = score(c, run(c, make(zero_rew=True)))["R"]
        print(f"C  call {c}: rew zeroed: ret miss {n}")
        ok &= n > 0
    plants = {"lam_1": (dict(lam=1.0), "R", 2), "disc_horizon": (dict(disc=1 - 1 / 333), "R", 2),
              "slowreg_0": (dict(slowreg=0.0), "V", 3), "actent_0": (dict(actent=0.0), "Q", 4),
              "retnorm_not_updated": (dict(update=False), "Q", 4), "weight_no_cumprod": (dict(cum=False), "V", 3)}
    caught = 0
    for name, (kw, term, idx) in plants.items():
        hits, live = [], []
        for c in CALLS:
            out = run(c, make(**kw))
            changed = not np.array_equal(out[idx], true[c][idx])
            live.append(changed)
            if changed:
                hits.append(score(c, out)[term][0] > 0)
        good = any(live) and all(hits)
        caught += good
        print(f"P  planted {name:20s}: live on {sum(live)}/{len(CALLS)}, caught on {sum(hits)}/{sum(live)} live calls")
    good = ok and caught == len(plants)
    print(f"RUNG 13b-imagloss: port {'PASS' if good else 'FAIL'} (planted caught {caught}/{len(plants)})")
    return 0 if good else 1


# EIGHTH PIECE (11-25 21:30, `--imagloss-sym`): H rew and R RE-REGISTERED after the FIRST READ above failed and
# a kappa line was measured and rejected (`rung13b_cond_probe.py`, `8b2ed8e7`: bound 16x too loose). The 255-bin
# twohot decode is ill-conditioned (one f32 ulp of the top bin's p moves it 0.23), so no f32 decode is the truth
# and the official f32 golden is not either. THE LINE, fixed before its read, at calls 1/2/6/8, eager AND
# compiled. TRUTH = the variant's own function in float64 (numpy) from the port's own f32 logits: rew64 =
# decode64(rew logits); val64 = decode64(val logits); ret64 = lambda_return64(0, 1-con, in_rew, val64, val64,
# disc, lam) with the golden's in_rew/in_con. For each of rew and ret: n_port = #elems |port_f32 - truth| >
# 1e-5(|truth|+1), n_off = the same for the official golden (in_rew / outs/ret). (S) PASS iff |n_port - n_off|
# <= 0.05 * max(n_port, n_off) — the port is as far from the f64 truth as the official f32 code is, both ways
# (a port bug moves the truth away from the golden, so n_off grows; a port decode bug moves n_port).
# (H-con) con sigmoid stays element-wise 1e-5 vs in_con. (C) control: rew zeroed must fail (S) on ret every call.
# (P) planted, each must fail its (S) on every call: rew top-bin logit +1e-3 (H rew); val top-bin logit +1e-3
# (R); lam 1.0 (R); disc 1 - 1/333 (R). V and Q are NOT re-registered here (no golden val/slowval to teacher-
# force them) — the next piece; the NO-OP plant retnorm_not_updated is replaced there by retnorm limit 0.
# FIRST READ (11-25 21:25) FAILED, recorded as is — the PORT meets (S) on every call eager+compiled (rew
#   n_port/n_off 0/0, 737/737, 4517/4524, 5180/5191; ret 0/0, 708/708, 10023/10045, 9155/9188; con 0 miss), but
#   the LINE has no teeth: C 3/4 (call 1 NO-OP — rew head zero-init, in_rew all 0; the old "where it changes the
#   function" exemption was not written into this line), plants 0/4. Causes, measured: (i) COUNT SATURATION — under
#   the top-logit plants the port's own f32 decode is > 1e-5 from its own f64 truth on ~all elements (rew plant:
#   n_port 16384/16384/13417/14057 vs n_off 16384 every call), so a count of "far" elements cannot tell a 0.23
#   rounding from a 1,900 shift; (ii) disc 1-1/333 moves ret below the f32 floor on most elements (n_port 700 vs
#   n_off 718 at call 2; 10002 vs 10000 at call 6). Also SEEN (a lead, not a verdict): the port's WORST ret
#   deviation from the f64 truth exceeds the official's (call 8: 0.382 vs 0.0287; call 6: 0.358 vs 0.225) while
#   rew's is smaller (0.58 vs 0.75) — a magnitude line must explain that before it is written. Next: diagnose the
#   call-8 worst ret element (which t, which val decodes), THEN register a magnitude (not count) line.
# LEAD EXPLAINED (11-25 21:31, `rung13b_retworst_probe.py`): the port's worst ret element at calls 6 AND 8 is
#   t=14, whose bootstrap val decode at t=15 takes ONE f32 top-bin rounding jump (+0.579, the same quantum as
#   rew's); the recursion adds 2e-9. The official golden did not jump at that element (its own max 0.225 at call 6
#   is the same kind of event). The error DISTRIBUTIONS match: |port-truth| p50 2.45e-5 / p99 1.46e-3 vs official
#   2.47e-5 / 1.50e-3 (call 6); 2.40e-5 / 2.73e-3 vs 2.42e-5 / 2.70e-3 (call 8). A max line is a lottery of
#   single rounding events; not a port bug. These quantiles are now SEEN — any quantile line registered next must
#   say so and earn its teeth from the plants (and the NO-OP exemption), not from these figures.
# NINTH PIECE (11-25 21:35, `--imagloss-q`): the (S) line RE-REGISTERED as a MAGNITUDE line, written AFTER the
# quantiles above were SEEN (ratios 0.98-1.01) — so its pass by the true port is expected and proves little; its
# worth is its TEETH, which are not seen: the same control and four plants must now break it. (Q-S) for rew and
# ret, with e = |x - truth| and q_k = the k-th percentile + 1e-7: for k in 50, 90, 99, q_k(port) and q_k(official)
# within a factor 1.25 of each other, both ways. Max is not used (a lottery of single rounding jumps). NO-OP
# EXEMPTION, as in 13b-enc/wm/imag: a control or plant is judged only on calls where its port output is not
# bit-equal to the true port's; NO-OP on every call fails. H-con element-wise 1e-5 unchanged.
# FIRST READ (11-25 21:28) FAILED on ONE plant, recorded as is: port meets (Q-S) on every call eager+compiled
#   (ret q50/q90/q99 2.4e-5/1.7e-4/2.7e-3 both, call 8); con 0 miss; C fails it on 3/3 live calls (call 1 NO-OP);
#   rew top-logit 4/4 (official q50 1.9e3 vs port 7e-2), val top-logit 4/4, lam 1.0 3/3 live — but disc 1-1/333
#   breaks it on 1/3 live calls: at calls 6/8 official q's vs the planted truth are IDENTICAL to the unplanted
#   (2.4e-5/1.3e-4/1.5e-3) — |ret| here is ~5e-3, so a 1/333-per-step discount moves it under the f32 decode
#   floor. The plant cannot be seen at this precision on this golden by ANY ret line; it is not a port finding
#   (the official disc is exactly 1.0, contdisc True). Next: rule that plant's fate (a NO-OP-at-precision class
#   like retnorm's, or a disc certified where ret is large), then V and Q.
# RULING (11-25 21:45, `--imagloss-q2`), written BEFORE its read; the (Q-S) line, control, H-con and the three
# seeing plants are UNCHANGED. The disc 1-1/333 plant is moved to the NO-OP-AT-PRECISION class (as retnorm's
# was), on two measured grounds: (i) at calls 6/8 the official golden's quantiles against the PLANTED truth are
# identical to those against the unplanted one (2.4e-5/1.3e-4/1.5e-3) — |ret|~5e-3 here, so a 1/333 discount is
# under the f32 decode floor for ANY ret line on this golden, port-independent; (ii) the same plant is CAUGHT at
# rung 10 ("disc 1-1/333 under contdisc", PASS 11-25, wheel `cdd0db67`), where ret is large. It is still run and
# printed, not counted. In its place a clause that CAN fail is added, so the disc is not left unchecked here:
# (D) the port's certified module `M.imag_return(rew, con, val, contdisc=True, horizon=333, lam=0.95)` on the
# golden's in_rew/in_con and the port's own val gives ret BIT-EQUAL to this rung's true path at every call, eager
# AND compiled (so the path read above is the one rung 10 certified, with disc exactly 1.0 as the official config);
# (D-control) contdisc=False through the same module must NOT be bit-equal on every live call (calls where ret is
# 0 are exempt). Plants needed: 3/3 seeing plants + D + D-control.
def main_imagloss_sym(qline=False, ruled=False):
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    sys.path.insert(0, HERE)
    import mlx.core as mx
    import mlx_maths as M
    G, R = np.load(GOLDEN_A), np.load(R13)
    b = M.bins(255)
    b64 = np.array(b).astype(np.float64)

    def dec64(L):
        L = np.asarray(L, np.float64)
        p = np.exp(L - L.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        m = 127
        return p[..., m] * b64[m] + ((p[..., m + 1:] - p[..., :m][..., ::-1]) * b64[m + 1:]).sum(-1)

    def lret64(con, rew, val, disc, lam):
        live = con[:, 1:] * disc
        interm = rew[:, 1:] + (1 - lam) * live * val[:, 1:]
        rets = [val[:, -1]]
        for t in reversed(range(live.shape[1])):
            rets.append(interm[:, t] + live[:, t] * lam * rets[-1])
        return np.stack(list(reversed(rets))[:-1], 1)

    def make(lam=0.95, disc=1.0, rew_plant=0.0, val_plant=0.0, zero_rew=False):
        def f(Pr, Pc, Pv, d, s, rew, con):
            x = M.feat2tensor(d, s)
            Lr = M._mlp_head(Pr, "rew", x, "logits")
            Lr = Lr + mx.concatenate([mx.zeros(Lr.shape[:-1] + (254,), mx.float32),
                                      mx.full(Lr.shape[:-1] + (1,), rew_plant, mx.float32)], -1)
            Lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
            Lv = Lv + mx.concatenate([mx.zeros(Lv.shape[:-1] + (254,), mx.float32),
                                      mx.full(Lv.shape[:-1] + (1,), val_plant, mx.float32)], -1)
            hr = M.twohot_decode(Lr, b)
            hc = mx.sigmoid(M._mlp_head(Pc, "con", x, "logit")[..., 0])
            val = M.twohot_decode(Lv, b)
            r = mx.zeros_like(rew) if zero_rew else rew
            ret = M.lambda_return(mx.zeros_like(con), 1.0 - con, r, val, val, disc, lam)
            return Lr, Lv, hr, hc, ret
        return f, dict(lam=lam, disc=disc, zero_rew=zero_rew)

    def run(c, fk, compiled=False):
        f, kw = fk
        P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa: E731
        g = f"c{c}/imag/"
        d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
        s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
        gr, gc = G[f"c{c}/imagloss/in_rew"], G[f"c{c}/imagloss/in_con"]
        f = mx.compile(f) if compiled else f
        Lr, Lv, hr, hc, ret = [np.array(x) for x in f(P("rew"), P("con"), P("val"), mx.array(d), mx.array(s),
                                                       mx.array(gr), mx.array(gc))]
        rew64 = dec64(Lr)
        r64in = np.zeros_like(gr, np.float64) if kw["zero_rew"] else gr.astype(np.float64)
        ret64 = lret64(gc.astype(np.float64), r64in, dec64(Lv), kw["disc"], kw["lam"])
        return hr, hc, ret, rew64, ret64

    def far(a, t):
        a = np.asarray(a, np.float64)
        e = np.abs(a - t)
        return int((e > TOL * (np.abs(t) + 1)).sum()), float(e.max())

    def qs(a, t):
        e = np.abs(np.asarray(a, np.float64) - t)
        return [float(np.percentile(e, k)) + 1e-7 for k in (50, 90, 99)]

    def sym(c, out):
        hr, hc, ret, rew64, ret64 = out
        g = f"c{c}/imagloss/"
        res = {}
        for k, port, off, t in (("rew", hr, G[g + "in_rew"], rew64), ("ret", ret, G[g + "outs/ret"], ret64)):
            (npo, wp), (nof, wo) = far(port, t), far(off, t)
            if qline:
                qp, qo = qs(port, t), qs(off, t)
                good = all(max(a / b, b / a) <= 1.25 for a, b in zip(qp, qo))
                res[k] = (good, "/".join(f"{x:.1e}" for x in qp), "/".join(f"{x:.1e}" for x in qo), wp, wo)
            else:
                res[k] = (abs(npo - nof) <= 0.05 * max(npo, nof), npo, nof, wp, wo)
        return res

    ok = True
    for c in CALLS:
        for mode in ("eager", "compiled"):
            out = run(c, make(), mode == "compiled")
            r = sym(c, out)
            nc, wc = miss(out[1], G[f"c{c}/imagloss/in_con"])
            print(f"   call {c} {mode:8s}: " + "  ".join(
                f"{k} {'ok' if v[0] else 'FAIL'} n_port {v[1]} n_off {v[2]} (worst {v[3]:.2e} / {v[4]:.2e})"
                for k, v in r.items()) + f"  con miss {nc} worst {wc:.2e}")
            ok &= all(v[0] for v in r.values()) and nc == 0
    true = {c: run(c, make()) for c in CALLS}
    live_of = lambda c, out, term: not np.array_equal(out[0 if term == "rew" else 2], true[c][0 if term == "rew" else 2])  # noqa: E731
    for c in CALLS:
        out = run(c, make(zero_rew=True))
        r = sym(c, out)["ret"]
        if qline and not live_of(c, out, "ret"):
            print(f"C  call {c}: rew zeroed: NO-OP (port ret bit-equal), exempt")
            continue
        print(f"C  call {c}: rew zeroed: ret {'fails (S)' if not r[0] else 'PASSES (S) — control dead'} n_port {r[1]} n_off {r[2]}")
        ok &= not r[0]
    plants = {"rew_toplogit_1e-3": (dict(rew_plant=1e-3), "rew"), "val_toplogit_1e-3": (dict(val_plant=1e-3), "ret"),
              "lam_1": (dict(lam=1.0), "ret"), "disc_horizon": (dict(disc=1 - 1 / 333), "ret")}
    caught = 0
    for name, (kw, term) in plants.items():
        outs = [run(c, make(**kw)) for c in CALLS]
        lv = [live_of(c, o, term) if qline else True for c, o in zip(CALLS, outs)]
        rs = [sym(c, o)[term] for c, o in zip(CALLS, outs)]
        hits = [not r[0] for r, l in zip(rs, lv) if l]
        hits = hits if any(lv) else [False]
        print("   " + name + ": " + "  ".join(f"c{c} n_port {r[1]} n_off {r[2]}" for c, r in zip(CALLS, rs)))
        caught += all(hits)
        if name == "disc_horizon":
            disc_hit = all(hits)
            if ruled:
                print("   (disc_horizon: NO-OP-AT-PRECISION by the 21:45 ruling - printed, not counted; certified at rung 10)")
        print(f"P  planted {name:18s}: breaks (S) on {sum(hits)}/{len(CALLS)} calls")
    plants_needed = len(plants)
    if ruled:
        def dpath(contdisc):
            def g(Pv, d, s, rew, con):
                x = M.feat2tensor(d, s)
                val = M.twohot_decode(M._mlp_head_n(Pv, "val", x, "logits", 3), b)
                return M.imag_return(rew, con, val, contdisc, 333, 0.95)["ret"]
            return g
        for c in CALLS:
            P = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/val/")}
            gp = f"c{c}/imag/"
            d = np.concatenate([G[gp + "start/deter"][:, None], G[gp + "feat/deter"]], 1)
            s = np.concatenate([G[gp + "start/stoch"][:, None], G[gp + "feat/stoch"]], 1)
            args = (P, mx.array(d), mx.array(s), mx.array(G[f"c{c}/imagloss/in_rew"]),
                    mx.array(G[f"c{c}/imagloss/in_con"]))
            for mode in ("eager", "compiled"):
                g1 = dpath(True)
                g1 = mx.compile(g1) if mode == "compiled" else g1
                truth = run(c, make(), mode == "compiled")[2]
                eq = bool(np.array_equal(np.array(g1(*args)), truth))
                print(f"D  call {c} {mode:8s}: imag_return(contdisc) ret bit-equal to this path: {eq}")
                ok &= eq
            r0 = np.array(dpath(False)(*args))
            live = bool(np.any(true[c][2] != 0))
            dif = not np.array_equal(r0, true[c][2])
            print(f"DC call {c}: contdisc False {'differs' if dif else 'BIT-EQUAL - control dead'}{'' if live else ' (ret all 0, exempt)'}")
            ok &= dif or not live
        caught -= int(disc_hit)
        plants_needed -= 1
    good = ok and caught == plants_needed
    print(f"RUNG 13b-imagloss-{('q2' if ruled else 'q') if qline else 'sym'} (H rew, H con, R): port {'PASS' if good else 'FAIL'} (planted caught {caught}/{plants_needed})")
    return 0 if good else 1


# TENTH PIECE (11-25 22:20, `--imagloss-vq`): V (value loss) and Q (policy loss) RE-REGISTERED against an f64 truth,
# the way rew/ret were in `--imagloss-q2`; WRITTEN BEFORE ITS READ (V/Q quantiles vs an f64 truth never computed
# before this commit; only the first read's element-wise misses are seen: calls 1/2 0 miss, calls 6/8 V 31/82,
# Q 376/438 — downstream of the val-decode rounding jumps). TEACHER-FORCED: the golden's imagined feats, actions,
# in_rew and in_con; the port's own f32 head logits (val, slowval, pol mean/std pre-activations). TRUTH = the
# variant's own function in float64 (numpy) from those logits: val64/slow64 = decode64; ret64 = lambda_return64
# (disc 1.0, lam 0.95); w64 = cumprod(in_con); value64[t] = w64[t]·(twohot_nll64(Lv[t], ret64[t]) + slowreg·
# twohot_nll64(Lv[t], slow64[t])), t < 15; retnorm64: lo'/hi' = 0.99·pre + 0.01·percentile_linear(ret64, 5/95),
# sc = max(limit, hi'-lo'); adv64 = (ret64 - val64[:, :-1])/sc; mean = tanh, std = 0.9·sigmoid(s+2)+0.1, Normal
# logp/entropy summed over 8 dims; policy64 = w64[:, :-1]·-(logp·adv64 + actent·ent). THE LINE, for V and Q at calls
# 1/2/6/8, eager AND compiled — the unchanged (Q-S): e = |x - truth|, q_k = k-th percentile + 1e-7, k = 50/90/99,
# q_k(port) and q_k(official golden loss/value, loss/policy) within a factor 1.25 both ways. A port bug moves the
# port off the truth; a reference that is not the official function moves the official off it. (C) control: rew
# zeroed must break (Q-S) on V AND Q on every live call (NO-OP exemption as in q2). (P) planted, each must break
# its term on every live call; NO-OP on every call fails the plant: slowreg 0 (V); weight without cumprod (V);
# actent 0 (Q); retnorm LIMIT 0 (Q) — replaces retnorm_not_updated, which was NO-OP everywhere (|ret|~5e-3 keeps
# hi'-lo' under the limit 1.0, so update or not, sc = 1; with limit 0, sc = hi'-lo' and adv is rescaled).
# PASS iff the port meets (Q-S) on V and Q at every call/mode, C holds, plants 4/4. Thresholds as q2's; none moved.
def main_imagloss_vq():
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    sys.path.insert(0, HERE)
    import mlx.core as mx
    import mlx_maths as M
    G, R = np.load(GOLDEN_A), np.load(R13)
    b = M.bins(255)
    b64 = np.array(b).astype(np.float64)
    n = b64.size

    def lsm64(L):
        L = np.asarray(L, np.float64)
        mx_ = L.max(-1, keepdims=True)
        return L - mx_ - np.log(np.exp(L - mx_).sum(-1, keepdims=True))

    def dec64(L):
        p = np.exp(lsm64(L))
        m = 127
        return p[..., m] * b64[m] + ((p[..., m + 1:] - p[..., :m][..., ::-1]) * b64[m + 1:]).sum(-1)

    def nll64(L, tgt):
        t = np.asarray(tgt, np.float64)[..., None]
        below = np.clip((b64 <= t).sum(-1) - 1, 0, n - 1)
        above = np.clip(n - (b64 > t).sum(-1), 0, n - 1)
        eq = below == above
        tt = t[..., 0]
        db = np.where(eq, 1.0, np.abs(b64[below] - tt))
        da = np.where(eq, 1.0, np.abs(b64[above] - tt))
        wb, wa = da / (db + da), db / (db + da)
        lp = lsm64(L)
        return -(wb * np.take_along_axis(lp, below[..., None], -1)[..., 0]
                 + wa * np.take_along_axis(lp, above[..., None], -1)[..., 0])

    def lret64(con, rew, val, lam=0.95):
        live = con[:, 1:]
        interm = rew[:, 1:] + (1 - lam) * live * val[:, 1:]
        rets = [val[:, -1]]
        for t in reversed(range(live.shape[1])):
            rets.append(interm[:, t] + live[:, t] * lam * rets[-1])
        return np.stack(list(reversed(rets))[:-1], 1)

    def perc64(x, q):
        v = np.sort(x.reshape(-1))
        pos = q / 100.0 * (v.size - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, v.size - 1)
        fr = pos - i0
        return v[i0] * (1 - fr) + v[i1] * fr

    def make(slowreg=1.0, actent=3e-4, cum=True, limit=1.0, zero_rew=False):
        def f(Pv, Ps, Pp, d, s, act, rew, con, lo, hi):
            x = M.feat2tensor(d, s)
            lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
            ls = M._mlp_head_n(Ps, "slowval", x, "logits", 3)
            val = M.twohot_decode(lv, b)
            slow = mx.stop_gradient(M.twohot_decode(ls, b))
            r = mx.zeros_like(rew) if zero_rew else rew
            ret = M.lambda_return(mx.zeros_like(con), 1.0 - con, r, val, val, 1.0, 0.95)
            w = mx.stop_gradient(mx.cumprod(con, 1) if cum else con)
            tar = mx.stop_gradient(mx.concatenate([ret, 0 * ret[:, -1:]], 1))
            value = (w * (M.twohot_loss(lv, tar, b) + slowreg * M.twohot_loss(lv, slow, b)))[:, :-1]
            _, _, _, sc = M.retnorm_perc(lo, hi, ret, 0.01, limit, 5.0, 95.0)
            adv = M.advantage(ret, val, sc)
            m = M._mlp_head_n(Pp, "pol", x, "action/mean", 3)
            sd = M._mlp_head_n(Pp, "pol", x, "action/stddev", 3)
            z = mx.zeros((*m.shape[:-1], 1), mx.float32)
            pol, _ = M.policy_loss(m, sd, z, act, mx.zeros(m.shape[:-1], mx.int32), adv, w, 0.1, 1.0, actent)
            return lv, ls, m, sd, value, pol
        return f, dict(slowreg=slowreg, actent=actent, cum=cum, limit=limit, zero_rew=zero_rew)

    def truth(c, kw, lv, ls, m, sd, act, rew, con):
        lo0 = float(R[f"c{c}/pre/retnorm/lo/value"]); hi0 = float(R[f"c{c}/pre/retnorm/hi/value"])
        con, act = con.astype(np.float64), act.astype(np.float64)
        rew = np.zeros_like(con) if kw["zero_rew"] else rew.astype(np.float64)
        val, slow = dec64(lv), dec64(ls)
        ret = lret64(con, rew, val)
        w = np.cumprod(con, 1) if kw["cum"] else con
        value = (w[:, :-1] * (nll64(lv[:, :-1], ret) + kw["slowreg"] * nll64(lv[:, :-1], slow[:, :-1])))
        lo = 0.99 * lo0 + 0.01 * perc64(ret, 5.0)
        hi = 0.99 * hi0 + 0.01 * perc64(ret, 95.0)
        adv = (ret - val[:, :-1]) / max(kw["limit"], hi - lo)
        mean = np.tanh(np.asarray(m, np.float64))
        std = 0.9 / (1 + np.exp(-(np.asarray(sd, np.float64) + 2.0))) + 0.1
        lp = (-0.5 * ((act - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)).sum(-1)
        ent = (0.5 * np.log(2 * np.pi * std ** 2) + 0.5).sum(-1)
        pol = w[:, :-1] * -(lp[:, :-1] * adv + kw["actent"] * ent[:, :-1])
        return value, pol

    def run(c, fk, compiled=False):
        f, kw = fk
        P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa: E731
        g = f"c{c}/imag/"
        d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
        s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
        a = np.concatenate([G[g + "act/action"], np.zeros_like(G[g + "act/action"][:, :1])], 1)
        gr, gc = G[f"c{c}/imagloss/in_rew"], G[f"c{c}/imagloss/in_con"]
        A = mx.array
        lo = A(R[f"c{c}/pre/retnorm/lo/value"]).astype(mx.float32)
        hi = A(R[f"c{c}/pre/retnorm/hi/value"]).astype(mx.float32)
        f = mx.compile(f) if compiled else f
        lv, ls, m, sd, value, pol = [np.array(x) for x in f(P("val"), P("slowval"), P("pol"), A(d), A(s), A(a),
                                                             A(gr), A(gc), lo, hi)]
        v64, p64 = truth(c, kw, lv, ls, m, sd, a, gr, gc)
        return value, pol, v64, p64

    def qs(x, t):
        e = np.abs(np.asarray(x, np.float64) - t)
        return [float(np.percentile(e, k)) + 1e-7 for k in (50, 90, 99)]

    def line(c, out):
        value, pol, v64, p64 = out
        g = f"c{c}/imagloss/"
        res = {}
        for k, port, off, t in (("V", value, G[g + "loss/value"], v64), ("Q", pol, G[g + "loss/policy"], p64)):
            qp, qo = qs(port, t), qs(off, t)
            good = bool(np.all(np.isfinite(qp + qo))) and all(max(x / y, y / x) <= 1.25 for x, y in zip(qp, qo))
            res[k] = (good, "/".join(f"{x:.1e}" for x in qp), "/".join(f"{x:.1e}" for x in qo))
        return res

    ok = True
    for c in CALLS:
        for mode in ("eager", "compiled"):
            r = line(c, run(c, make(), mode == "compiled"))
            print(f"   call {c} {mode:8s}: " + "  ".join(
                f"{k} {'ok' if v[0] else 'FAIL'} q port {v[1]} off {v[2]}" for k, v in r.items()))
            ok &= all(v[0] for v in r.values())
    true = {c: run(c, make()) for c in CALLS}
    idx = {"V": 0, "Q": 1}
    live_of = lambda c, out, term: not np.array_equal(out[idx[term]], true[c][idx[term]])  # noqa: E731
    for c in CALLS:
        out = run(c, make(zero_rew=True))
        r = line(c, out)
        for term in ("V", "Q"):
            if not live_of(c, out, term):
                print(f"C  call {c}: rew zeroed: {term} NO-OP, exempt")
                continue
            print(f"C  call {c}: rew zeroed: {term} {'breaks (Q-S)' if not r[term][0] else 'MEETS (Q-S) - control dead'}"
                  f" q port {r[term][1]} off {r[term][2]}")
            ok &= not r[term][0]
    plants = {"slowreg_0": (dict(slowreg=0.0), "V"), "weight_no_cumprod": (dict(cum=False), "V"),
              "actent_0": (dict(actent=0.0), "Q"), "retnorm_limit_0": (dict(limit=0.0), "Q")}
    caught = 0
    for name, (kw, term) in plants.items():
        outs = [run(c, make(**kw)) for c in CALLS]
        lv = [live_of(c, o, term) for c, o in zip(CALLS, outs)]
        rs = [line(c, o)[term] for c, o in zip(CALLS, outs)]
        hits = [not r[0] for r, l in zip(rs, lv) if l]
        hits = hits if any(lv) else [False]
        print("   " + name + ": " + "  ".join(f"c{c}{'' if l else '(no-op)'} port {r[1]} off {r[2]}"
                                             for c, r, l in zip(CALLS, rs, lv)))
        caught += all(hits)
        print(f"P  planted {name:18s}: live on {sum(lv)}/{len(CALLS)}, breaks (Q-S) on {sum(hits)}/{sum(lv)} live calls")
    good = ok and caught == len(plants)
    print(f"RUNG 13b-imagloss-vq (V, Q): port {'PASS' if good else 'FAIL'} (planted caught {caught}/{len(plants)})")
    return 0 if good else 1


def main():
    if "--imagloss-vq" in sys.argv:
        return main_imagloss_vq()
    if "--imagloss-q2" in sys.argv:
        return main_imagloss_sym(qline=True, ruled=True)
    if "--imagloss-q" in sys.argv:
        return main_imagloss_sym(qline=True)
    if "--imagloss-sym" in sys.argv:
        return main_imagloss_sym()
    if "--imagloss" in sys.argv:
        return main_imagloss()
    if "--imag" in sys.argv:
        return main_imag()
    if "--wm" in sys.argv:
        return main_wm()
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
