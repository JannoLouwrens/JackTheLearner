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


def main():
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
