"""MLX port, stage 0 step 0.3 — throwaway compile spike (server half: `size1m`).

PORT report (docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md, "A one-day
spike"): the one risk no source answers is how long MLX takes to compile
Dreamer's UNROLLED training step and how much memory it needs. This script
builds a SHAPE-FAITHFUL dummy of the official `size1m` train step with random
weights and measures exactly that. It is not the port: no parity is claimed,
weights are not the official init, losses are structurally right but not
numerically checked. It is thrown away once G0 is decided.

Shapes copied from the numerical contract
(/data/jack-research/research_notes/Porting Dreamer to Apple MLX/dreamer_port_spec.md §2):
  size1m: deter 512, hidden 64, stoch 32 x classes 4, units 64, blocks 8;
  Jack's world: 74 observations in, 8 continuous actions out;
  batch 16 x 64 trained steps (+1 context), imagination 15 steps from all
  1,024 posterior starts, 255 symexp bins for reward and critic;
  encoder/decoder 3 x (Linear -> RMSNorm -> SiLU), reward/cont 1 hidden,
  actor/critic 3 hidden; core = 3 input embeddings repeated per block +
  BlockLinear(deter) -> RMSNorm -> SiLU -> BlockLinear(3 deter), gates split
  inside each block (reset, cand, update), update bias -1; posterior 1 layer,
  prior 2 layers; unimix 1 %; straight-through categorical samples; KL dyn/rep
  with free bits 1; lambda-returns (0.95, gamma 0.997); slow critic; full
  gradients; one LaProp-like update with global-norm clipping inside the step.
Randomness is passed IN as arrays (the "noise from file" design the golden
files will use), so the compiled function is pure.

Settings that differ from the official step (all irrelevant to compile cost,
all relevant to parity — this is not the port): random-normal init (official:
truncated normal, fan-in, outscale 0 on heads); actor = tanh-mean Gaussian
(official bounded normal with minstd/maxstd); no return normaliser state, no
replay-value loss, no latent write-back; advantage uses a fixed scale 1.

Gate G0 (report's thresholds, not published): `size12m` compiles < 10 min and
peaks < 8 GB on CPU -> proceed. The server measures `size1m` only (the report
assigns `size12m` to Kaggle); `--size 12m` exists for the Kaggle half.

Run (1 core, gcc-toolset-15 for MLX's CPU JIT, set below):
    taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I \
      experiments/brain_a/mlx_spike.py --size 1m --steps 10 --out /data/jack-logs/mlx/spike_1m.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time

import numpy as np

GXX = "/opt/rh/gcc-toolset-15/root/usr/bin"
if os.path.isdir(GXX) and not os.environ.get("PATH", "").startswith(GXX):
    os.environ["PATH"] = GXX + os.pathsep + os.environ.get("PATH", "")

import mlx.core as mx

SIZES = {
    "1m": dict(deter=512, hidden=64, classes=4, units=64),
    "12m": dict(deter=2048, hidden=256, classes=16, units=256),
}
OBS, ACT, B, T, H, STOCH, BLOCKS, BINS = 74, 8, 16, 64, 15, 32, 8, 255


def symlog(x):
    return mx.sign(x) * mx.log1p(mx.abs(x))


def bins_np():
    half = np.sign(np.linspace(-20, 0, 128)) * np.expm1(np.abs(np.linspace(-20, 0, 128)))
    return np.concatenate([half, -half[:-1][::-1]]).astype(np.float32)


def init_params(cfg, rng):
    d, h, c, u = cfg["deter"], cfg["hidden"], cfg["classes"], cfg["units"]
    S = STOCH * c
    P = {}

    def dense(name, i, o):
        P[name] = {"w": mx.array(rng.normal(0, 1 / np.sqrt(i), (i, o)).astype(np.float32)),
                   "b": mx.zeros((o,)), "g": mx.ones((o,))}

    def head(name, i, o):
        P[name] = {"w": mx.array(rng.normal(0, 0.01 / np.sqrt(i), (i, o)).astype(np.float32)),
                   "b": mx.zeros((o,))}

    def mlp(name, i, n):
        for k in range(n):
            dense(f"{name}{k}", i if k == 0 else u, u)

    mlp("enc", OBS, 3)
    dense("in_deter", d, h); dense("in_stoch", S, h); dense("in_act", ACT, h)
    bi = d // BLOCKS + 3 * h
    P["blk1"] = {"w": mx.array(rng.normal(0, 1 / np.sqrt(d + 3 * h * BLOCKS),
                                          (BLOCKS, bi, d // BLOCKS)).astype(np.float32)),
                 "b": mx.zeros((d,)), "g": mx.ones((d,))}
    P["blk2"] = {"w": mx.array(rng.normal(0, 1 / np.sqrt(d), (BLOCKS, d // BLOCKS, 3 * d // BLOCKS)).astype(np.float32)),
                 "b": mx.zeros((3 * d,))}
    dense("post0", d + u, h); head("post_out", h, S)
    dense("prior0", d, h); dense("prior1", h, h); head("prior_out", h, S)
    feat = d + S
    mlp("dec", feat, 3); head("dec_out", u, OBS)
    mlp("rew", feat, 1); head("rew_out", u, BINS)
    mlp("con", feat, 1); head("con_out", u, 1)
    mlp("act", feat, 3); head("act_out", u, 2 * ACT)
    mlp("cri", feat, 3); head("cri_out", u, BINS)
    return P


def lin(p, x):
    y = x @ p["w"] + p["b"]
    if "g" in p:
        y = mx.fast.rms_norm(y, p["g"], 1e-4)
        y = y * mx.sigmoid(y)  # SiLU
    return y


def mlp(P, name, n, x):
    for k in range(n):
        x = lin(P[f"{name}{k}"], x)
    return x


def unimix_logits(logits, c):
    logits = logits.reshape(*logits.shape[:-1], STOCH, c)
    probs = 0.99 * mx.softmax(logits, axis=-1) + 0.01 / c
    return mx.log(probs)


def st_sample(logl, gumbel):
    """Straight-through one-hot sample; gumbel noise comes from the caller."""
    idx = mx.argmax(logl + gumbel, axis=-1)
    one = (mx.arange(logl.shape[-1]) == idx[..., None]).astype(mx.float32)
    p = mx.exp(logl)
    s = one + p - mx.stop_gradient(p)
    return s.reshape(*s.shape[:-2], -1)


def core(P, cfg, deter, stoch, act):
    d, h = cfg["deter"], cfg["hidden"]
    x = mx.concatenate([lin(P["in_deter"], deter), lin(P["in_stoch"], stoch),
                        lin(P["in_act"], act)], -1)
    n = deter.shape[0]
    x = mx.broadcast_to(x[:, None, :], (n, BLOCKS, 3 * h))
    x = mx.concatenate([deter.reshape(n, BLOCKS, d // BLOCKS), x], -1)
    y = mx.einsum("nbi,bio->nbo", x, P["blk1"]["w"]).reshape(n, d) + P["blk1"]["b"]
    y = mx.fast.rms_norm(y, P["blk1"]["g"], 1e-4)
    y = y * mx.sigmoid(y)
    y = mx.einsum("nbi,bio->nbo", y.reshape(n, BLOCKS, d // BLOCKS), P["blk2"]["w"])
    y = y.reshape(n, 3 * d) + P["blk2"]["b"]
    y = y.reshape(n, BLOCKS, 3, d // BLOCKS)  # gates split INSIDE each block
    r, cand, upd = y[:, :, 0], y[:, :, 1], y[:, :, 2]
    r = mx.sigmoid(r)
    cand = mx.tanh(r * cand)
    u = mx.sigmoid(upd - 1)
    return (u * cand + (1 - u) * deter.reshape(n, BLOCKS, d // BLOCKS)).reshape(n, d)


def prior_logits(P, cfg, deter):
    x = lin(P["prior1"], lin(P["prior0"], deter))
    return unimix_logits(lin(P["prior_out"], x), cfg["classes"])


def twohot_loss(logits, y, bins):
    y = mx.clip(y, bins[0], bins[-1])
    below = mx.sum((bins[None] <= y[:, None]).astype(mx.int32), -1) - 1
    below = mx.clip(below, 0, BINS - 2)
    lo, hi = bins[below], bins[below + 1]
    w_hi = mx.clip((y - lo) / (hi - lo), 0, 1)
    ar = mx.arange(BINS)[None]
    tgt = (ar == below[:, None]) * (1 - w_hi)[:, None] + (ar == (below + 1)[:, None]) * w_hi[:, None]
    ls = logits - mx.logsumexp(logits, -1, keepdims=True)
    return -mx.sum(mx.stop_gradient(tgt) * ls, -1)


def twohot_mean(logits, bins):
    return mx.sum(mx.softmax(logits, -1) * bins, -1)


def make_loss(cfg, bins):
    c = cfg["classes"]

    def loss(P, slow, batch, noise):
        obs, act, rew, con, deter0, stoch0 = (batch[k] for k in
                                              ("obs", "act", "rew", "con", "deter0", "stoch0"))
        tok = mlp(P, "enc", 3, obs)                      # (B,T,u)
        deter, stoch = deter0, stoch0
        deters, posts, stochs = [], [], []
        for t in range(T):                               # 64-step observe pass, unrolled
            deter = core(P, cfg, deter, stoch, act[:, t])
            pl = unimix_logits(lin(P["post_out"], lin(P["post0"],
                                mx.concatenate([deter, tok[:, t]], -1))), c)
            stoch = st_sample(pl, noise["obs_g"][:, t])
            deters.append(deter); posts.append(pl); stochs.append(stoch)
        deter_s = mx.stack(deters, 1); post = mx.stack(posts, 1); stoch_s = mx.stack(stochs, 1)
        prior = prior_logits(P, cfg, deter_s)
        feat = mx.concatenate([deter_s, stoch_s], -1)
        pp, qp = mx.exp(post), mx.exp(prior)
        kl_dyn = mx.sum(mx.stop_gradient(pp) * (mx.stop_gradient(post) - prior), (-1, -2))
        kl_rep = mx.sum(pp * (post - mx.stop_gradient(prior)), (-1, -2))
        l_dyn = mx.maximum(kl_dyn, 1.0).mean()
        l_rep = mx.maximum(kl_rep, 1.0).mean()
        dec = lin(P["dec_out"], mlp(P, "dec", 3, mx.concatenate([stoch_s, deter_s], -1)))
        l_dec = mx.sum((dec - symlog(obs)) ** 2, -1).mean()
        rl = lin(P["rew_out"], mlp(P, "rew", 1, feat)).reshape(B * T, BINS)
        l_rew = twohot_loss(rl, rew.reshape(-1), bins).mean()
        cl = lin(P["con_out"], mlp(P, "con", 1, feat))[..., 0]
        l_con = (mx.logaddexp(0, cl) - con * cl).mean()
        # imagination: 15 steps from all B*T starts, gradient-stopped starts
        d = mx.stop_gradient(deter_s.reshape(B * T, -1))
        s = mx.stop_gradient(stoch_s.reshape(B * T, -1))
        feats, logps, ents = [], [], []
        for k in range(H + 1):
            f = mx.concatenate([d, s], -1)
            ao = lin(P["act_out"], mlp(P, "act", 3, mx.stop_gradient(f)))
            mean, std = mx.tanh(ao[:, :ACT]), 0.1 + 0.9 * mx.sigmoid(ao[:, ACT:])
            a = mx.stop_gradient(mean + std * noise["act_n"][k])
            logps.append(mx.sum(-0.5 * ((a - mean) / std) ** 2 - mx.log(std), -1))
            ents.append(mx.sum(mx.log(std), -1))
            feats.append(f)
            if k < H:
                d = core(P, cfg, d, s, a)
                s = st_sample(prior_logits(P, cfg, d), noise["img_g"][k])
        F = mx.stack(feats, 0)                            # (H+1, N, feat)
        rew_i = twohot_mean(lin(P["rew_out"], mlp(P, "rew", 1, F)), bins)
        con_i = mx.sigmoid(lin(P["con_out"], mlp(P, "con", 1, F))[..., 0])
        Fs = mx.stop_gradient(F)
        val_logits = lin(P["cri_out"], mlp(P, "cri", 3, Fs))
        slow_v = twohot_mean(lin(slow["cri_out"], mlp(slow, "cri", 3, Fs)), bins)
        disc = 0.997 * con_i
        ret = [slow_v[-1]]
        for k in range(H - 1, -1, -1):                   # lambda-returns, unrolled
            ret.append(rew_i[k + 1] + disc[k + 1] * (0.05 * slow_v[k + 1] + 0.95 * ret[-1]))
        ret = mx.stop_gradient(mx.stack(ret[::-1], 0))   # (H+1, N)
        adv = mx.stop_gradient(ret[:-1] - slow_v[:-1])
        l_pol = -(mx.stack(logps[:-1], 0) * adv + 3e-4 * mx.stack(ents[:-1], 0)).mean()
        l_val = twohot_loss(val_logits[:-1].reshape(-1, BINS), ret[:-1].reshape(-1), bins).mean()
        total = l_dec + l_rew + l_con + 0.5 * l_dyn + 0.1 * l_rep + l_pol + l_val
        return total

    return loss


def make_step(cfg, bins):
    vg = mx.value_and_grad(make_loss(cfg, bins))

    def step(P, slow, mom, batch, noise):
        l, g = vg(P, slow, batch, noise)
        leaves = [x for _, x in _flat(g)]
        gn = mx.sqrt(sum(mx.sum(x * x) for x in leaves))
        sc = mx.minimum(1.0, 1000.0 / (gn + 1e-6))
        newP, newM = {}, {}
        for k in P:
            newP[k], newM[k] = {}, {}
            for j in P[k]:
                gg = g[k][j] * sc
                v = 0.999 * mom[k][j] + 0.001 * gg * gg   # LaProp-like: rms first, then momentum
                newM[k][j] = v
                newP[k][j] = P[k][j] - 4e-5 * gg / (mx.sqrt(v) + 1e-20)
        newS = {k: {j: 0.98 * slow[k][j] + 0.02 * newP[k][j] for j in slow[k]} for k in slow}
        return l, gn, newP, newS, newM

    return step


def _flat(tree, pre=""):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _flat(v, pre + k + ".")
        else:
            yield pre + k, v


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1m", choices=SIZES)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--mode", default="compiled", choices=["compiled", "eager"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = SIZES[a.size]
    rng = np.random.default_rng(0)
    P = init_params(cfg, rng)
    slow = {k: dict(v) for k, v in P.items() if k.startswith("cri")}
    mom = {k: {j: mx.zeros(x.shape) for j, x in v.items()} for k, v in P.items()}
    nparams = sum(int(np.prod(x.shape)) for _, x in _flat(P))
    S = STOCH * cfg["classes"]
    bins = mx.array(bins_np())
    step = make_step(cfg, bins)
    if a.mode == "compiled":
        step = mx.compile(step)

    def batch_noise():
        f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32))
        g = lambda *s: mx.array(rng.gumbel(size=s).astype(np.float32))
        batch = {"obs": f(B, T, OBS), "act": mx.tanh(f(B, T, ACT)), "rew": f(B, T),
                 "con": mx.ones((B, T)), "deter0": mx.zeros((B, cfg["deter"])),
                 "stoch0": mx.zeros((B, S))}
        noise = {"obs_g": g(B, T, STOCH, cfg["classes"]), "act_n": f(H + 1, B * T, ACT),
                 "img_g": g(H, B * T, STOCH, cfg["classes"])}
        return batch, noise

    rows = []
    mx.reset_peak_memory()
    for i in range(a.steps):
        batch, noise = batch_noise()
        mx.eval(batch, noise)
        t0 = time.perf_counter()
        l, gn, P, slow, mom = step(P, slow, mom, batch, noise)
        mx.eval(l, gn, P, slow, mom)
        dt = time.perf_counter() - t0
        rows.append({"step": i, "s": round(dt, 3), "loss": float(l), "gnorm": float(gn),
                     "peak_mb": round(mx.get_peak_memory() / 2**20, 1),
                     "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)})
        print(json.dumps(rows[-1]), flush=True)
    steady = [r["s"] for r in rows[2:]] or [rows[-1]["s"]]
    out = {"size": a.size, "mode": a.mode, "params": nparams, "cfg": cfg,
           "B": B, "T": T, "H": H, "obs": OBS, "act": ACT,
           "first_step_s": rows[0]["s"], "steady_median_s": float(np.median(steady)),
           "peak_mb": max(r["peak_mb"] for r in rows),
           "maxrss_mb": max(r["maxrss_mb"] for r in rows),
           "finite": all(np.isfinite(r["loss"]) and np.isfinite(r["gnorm"]) for r in rows),
           "mlx": mx.__version__, "python": platform.python_version(),
           "machine": platform.machine(), "affinity": sorted(os.sched_getaffinity(0)),
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0 if out["finite"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
