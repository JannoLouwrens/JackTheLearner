"""MLX port of DreamerV3 — rung 10 references (ORDERS J25, PORT report step 2.5 first half):
the imagination rollout and the lambda-returns, size1m.

Official code `danijar/dreamerv3` at e01491f:
  (I) `dreamerv3/rssm.py` `RSSM.imagine` L94-118 — per step: action = policy(sg(carry)) (or the
      given action array); deter = _core(deter, stoch, DictConcat(action)); logit = _prior(deter);
      stoch = OneHot(logit, unimix).sample. `agent.py` L190-198 then builds the [N, H+1] rollout:
      feat = concat([start, imagined]), act = concat([imagine's actions, policy(last feat)]) — so
      act[:, t] is the action TAKEN at feat[:, t]; `starts` (rssm.py L56) = the last K replay
      entries flattened to B*K rows.
  (II) `agent.py` `lambda_return` L482-490, as `imag_loss` (L382-405) and `repl_loss` (L448-480)
      call it: imag: last = 0, term = 1 - con, val = boot = tarval (slowtar False -> the value),
      disc = 1 (contdisc True) else 1 - 1/horizon, weight = cumprod(disc * con, 1) / disc;
      repl: real is_last / is_terminal / reward, boot = the imagined return at each start,
      disc = 1 - 1/horizon, weight = ~last.
The official `imag_loss` / `repl_loss` are run for real, with stand-ins for what is NOT this rung:
value / slowvalue heads (`pred()` returns a fixed array, `loss()` returns ones so the value loss
reads back 2 x weight), the policy (logp / entropy zero) and the three normalisers (offset 0,
scale 1) — the critic (rung 11) and actor (rung 12) and the percentile normaliser come later.

"Ours beside theirs": NumPy float64 vs official float32 (published bf16); starts = the rung 7 size1m
golden's deter/stoch at its last K = 8 steps (N = 32 rows; published imag_last 0 -> K = T = 64,
the maths is per row); H = 15 (published imag_length); the toy action space of rungs 6-9 (a 3-d
continuous key drawn in [-3, 3] so the squash binds, a 4-way discrete key); two policies:
(a) a fixed action array (imagine's non-callable path), (b) a deterministic function of the carry
(the callable path, as the agent's — its actor is rung 12). Sampler = file noise (rung 7's patch);
near-tie rule as rung 7. Tolerance: ladder row 9 "relative 1e-4" — element |d|/(|ref|+1).

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung10.py --vs-official [--dump-golden]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL  # noqa: E402
import rung6  # noqa: E402
import rung7  # noqa: E402
from rung6 import TOY, CONT, DISC, CONT_DIM, DISC_N  # noqa: E402

TOY.update(rung7.SIZE1M)          # same dict object rung6/rung7's references read
GOLDEN7 = os.path.join(HERE, "rung7_scan_golden_size1m_e01491f.npz")
GOLDEN10 = os.path.join(HERE, "rung10_imag_golden_size1m_e01491f.npz")
NOISE = "/data/jack-logs/mlx/rung10_noise_size1m.npz"
NOISE_N = 200_000
K, H = 8, 15
HORIZON, LAM = 333, 0.95
TOL = 1e-4


def close(name, got, ref, tol=TOL):
    got, ref = np.float64(np.asarray(got)), np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    el = float((np.abs(got - ref) / (np.abs(ref) + 1)).max())
    ok = bool(np.all(np.isfinite(got)) and el <= tol)
    return ok, f"{name}: element {el:.3g} x(|ref|+1) {'ok' if ok else f'OVER {tol:g}'}"


# ---------------------------------------------------------------- (I) imagination
def starts():
    G = np.load(GOLDEN7)
    P = {k[len("param/"):]: G[k] for k in G.files if k.startswith("param/")}
    d, s = G["out/deter"][:, -K:], G["out/stoch"][:, -K:]
    return P, dict(deter=d.reshape(-1, d.shape[-1]), stoch=s.reshape(-1, *s.shape[2:]))


def action_array(n, seed=20261125):
    r = np.random.default_rng(seed)
    return {CONT: r.uniform(-3, 3, (n, H, CONT_DIM)).astype(np.float32),
            DISC: r.integers(0, DISC_N, (n, H)).astype(np.int32)}


def policy_np(deter, stoch):
    """The callable path's stand-in actor: a fixed function of the CARRY (both halves read)."""
    s = stoch.reshape(stoch.shape[0], -1)
    cont = 2.5 * np.tanh(4 * deter[:, :CONT_DIM] + s[:, :CONT_DIM])
    disc = (deter[:, CONT_DIM:CONT_DIM + DISC_N] + s[:, -DISC_N:]).argmax(-1).astype(np.int32)
    return {CONT: cont, DISC: disc}


def REF_imagine(P, start, U, actions=None, golden=None, squash=True, unimix=TOY["unimix"],
                lag=False, prepend=True):
    """H imagine steps from `start`; returns feat [N, H+1] (start first) and act [N, H+1]
    (act[:, t] taken at feat[:, t]; the last is policy(feat[:, H]) or, for the array path, absent)."""
    deter, stoch = np.float64(start["deter"]), np.float64(start["stoch"])
    F = {k: [] for k in ("deter", "stoch", "logit")}
    A = {CONT: [], DISC: []}
    prev, ties, dflips = None, 0, 0
    for h in range(H):
        if actions is not None:
            a = {CONT: np.float64(actions[CONT][:, h]), DISC: actions[DISC][:, h]}
        else:
            a = policy_np(deter, stoch) if not (lag and prev is not None) else prev
            if lag:
                prev = policy_np(deter, stoch)
            if golden is not None:            # f32/f64 argmax near-tie on the discrete key
                ga = golden["act"][DISC][:, h]
                dflips += int((ga != a[DISC]).sum())
        for k in A:
            A[k].append(a[k])
        deter = rung6.REF_core(P, deter, stoch, rung6.REF_actemb(a), TOY, squash=squash)
        logit = rung6.REF_prior(P, deter, TOY)
        gi = None if golden is None else golden["feat"]["stoch"][:, h + 1].argmax(-1)
        stoch, n = rung7.REF_sample(logit, U[h], gi, unimix)
        ties += n
        for k, v in (("deter", deter), ("stoch", stoch), ("logit", logit)):
            F[k].append(v)
    feat = {k: np.stack(v, 1) for k, v in F.items()}
    if prepend:
        st = dict(deter=np.float64(start["deter"]), stoch=np.float64(start["stoch"]),
                  logit=np.zeros_like(feat["logit"][:, 0]))
        feat = {k: np.concatenate([st[k][:, None], feat[k]], 1) for k in feat}
    act = {k: np.stack(v, 1) for k, v in A.items()}
    if actions is None:
        last = policy_np(deter, stoch)
        act = {k: np.concatenate([act[k], last[k][:, None]], 1) for k in act}
    return dict(feat=feat, act=act, ties=ties, dflips=dflips)


# ---------------------------------------------------------------- (II) lambda-returns
def REF_lambda_return(last, term, rew, val, boot, disc, lam):
    """R_t = r_{t+1} + disc (1-term_{t+1}) [(1 - lam (1-last_{t+1})) boot_{t+1} + lam (1-last_{t+1}) R_{t+1}],
    R_{T-1} = boot_{T-1}; returns R_0..R_{T-2} (float64, written from the definition)."""
    last, term, rew, val, boot = (np.float64(x) for x in (last, term, rew, val, boot))
    T = rew.shape[1]
    R = np.zeros_like(rew)
    R[:, T - 1] = boot[:, T - 1]
    for t in range(T - 2, -1, -1):
        g = disc * (1 - term[:, t + 1])
        c = lam * (1 - last[:, t + 1])
        R[:, t] = rew[:, t + 1] + g * ((1 - c) * boot[:, t + 1] + c * R[:, t + 1])
    return R[:, :-1]


def REF_imag(rew, con, val, contdisc=True, lam=LAM, horizon=HORIZON):
    disc = 1.0 if contdisc else 1 - 1 / horizon
    con = np.float64(con)
    weight = np.cumprod(disc * con, 1) / disc
    ret = REF_lambda_return(np.zeros_like(con), 1 - con, rew, val, val, disc, lam)
    return dict(ret=ret, weight=weight)


def REF_repl(last, term, rew, val, boot, lam=LAM, horizon=HORIZON):
    ret = REF_lambda_return(last, term, rew, val, boot, 1 - 1 / horizon, lam)
    return dict(ret=ret, weight=np.float64(~last))


def ret_inputs(n=4 * K, seed=20261126):
    r = np.random.default_rng(seed)
    S = H + 1
    rew = (r.choice([0.0, 0.0, 1.0, -1.0, 0.03, 7.5, -250.0], (n, S)) * r.uniform(0.5, 1.5, (n, S)))
    con = np.where(r.uniform(size=(n, S)) < 0.1, r.uniform(0, 0.02, (n, S)), r.uniform(0.9, 1.0, (n, S)))
    val = r.normal(0, 30, (n, S))
    slow = r.normal(0, 30, (n, S))
    Br, Tr = 4, K                      # replay part: [B, K] with real episode flags
    last = r.uniform(size=(Br, Tr)) < 0.15
    term = last & (r.uniform(size=(Br, Tr)) < 0.5)
    rrew = r.normal(0, 3, (Br, Tr))
    rval = r.normal(0, 30, (Br, Tr))
    rboot = r.normal(0, 30, (Br, Tr))
    f = np.float32
    return dict(rew=f(rew), con=f(con), val=f(val), slow=f(slow), last=last, term=term,
                rrew=f(rrew), rval=f(rval), rslow=f(r.normal(0, 30, (Br, Tr))), rboot=f(rboot))


# ---------------------------------------------------------------- official
def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.nets as nets
    import embodied.jax.outs as outs
    nets.COMPUTE_DTYPE = jnp.float32
    from dreamerv3.rssm import RSSM
    from dreamerv3 import agent as A
    from file_noise import FileNoise
    os.makedirs(os.path.dirname(NOISE), exist_ok=True)
    noise = FileNoise(NOISE, seed=10, n=NOISE_N).install(outs)
    log = []
    inner = outs.Categorical.sample

    def logged(self, seed, shape=()):
        c0 = noise.cur["uniform"]
        y = inner(self, seed, shape)
        log.append((c0, noise.cur["uniform"] - c0, bool(nj.context().ignore) or nj.creating()))
        return y
    outs.Categorical.sample = logged
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    act_space = {CONT: elements.Space(np.float32, (CONT_DIM,), -1, 1),
                 DISC: elements.Space(np.int32, (), 0, DISC_N)}
    rssm = RSSM(act_space, name="rssm", **TOY)
    P, start = starts()
    N = start["deter"].shape[0]
    acts = action_array(N)

    def pol_j(c):
        d, s = c["deter"], c["stoch"].reshape(c["stoch"].shape[0], -1)
        return {CONT: 2.5 * jnp.tanh(4 * d[:, :CONT_DIM] + s[:, :CONT_DIM]),
                DISC: jnp.argmax(d[:, CONT_DIM:CONT_DIM + DISC_N] + s[:, -DISC_N:], -1).astype(jnp.int32)}

    def run(policy):
        def f(deter, stoch, *a):
            carry = dict(deter=deter, stoch=stoch)
            _, feat, act = rssm.imagine(carry, policy if a == () else a[0], H, False)
            # agent.py L194-198 (first = the start; lastact = policy(last feat)):
            first = dict(deter=deter[:, None], stoch=stoch[:, None], logit=jnp.zeros_like(feat["logit"][:, :1]))
            feat = {k: jnp.concatenate([first[k], feat[k]], 1) for k in feat}
            if a == ():
                la = policy({k: feat[k][:, -1] for k in ("deter", "stoch")})
                act = {k: jnp.concatenate([act[k], la[k][:, None]], 1) for k in act}
            return dict(feat=feat, act=act)
        return f

    state = {f"rssm/{k}": jnp.asarray(v) for k, v in P.items()}
    J = lambda d: jax.tree.map(jnp.asarray, d)  # noqa: E731
    sargs = (jnp.asarray(start["deter"]), jnp.asarray(start["stoch"]))
    per = N * TOY["stoch"] * TOY["classes"]
    out, U = {}, {}
    with jax.disable_jit():
        for tag, pol, extra in (("array", None, (J(acts),)), ("policy", pol_j, ())):
            log.clear()
            _, o = nj.pure(run(pol))(state, *sargs, *extra, seed=jax.random.PRNGKey(1))
            real = [(c, n) for c, n, pre in log if not pre]
            pre = [(c, n) for c, n, p in log if p]
            print(f"        [{tag}] sampler calls {len(log)} = {len(pre)} prerun (+{sum(n for _, n in pre)} "
                  f"uniforms) + {len(real)} real")
            put(len(real) == H and all(n == per for _, n in real)
                and all(real[i + 1][0] == real[i][0] + per for i in range(H - 1)),
                f"[{tag}] real sampler calls: {len(real)} == H, each {per} uniforms, contiguous")
            U[tag] = np.stack([noise.uniform[c:c + per].reshape(N, TOY["stoch"], TOY["classes"]) for c, _ in real])
            out[tag] = jax.tree.map(np.asarray, o)
    nlat = N * H * TOY["stoch"]
    refs = {}
    for tag in ("array", "policy"):
        o = out[tag]
        ref = REF_imagine(P, start, U[tag], acts if tag == "array" else None, golden=o)
        refs[tag] = ref
        put(o["feat"]["deter"].shape == (N, H + 1, TOY["deter"]) and o["act"][CONT].shape[1] == H + (tag == "policy"),
            f"[{tag}] shapes feat {o['feat']['deter'].shape}, act {o['act'][CONT].shape}")
        for k in ("deter", "logit"):
            put(*close(f"[{tag}] {k} [N,H+1] f32 vs f64 ({H} steps)", o["feat"][k], ref["feat"][k]))
        flips = int((o["feat"]["stoch"].argmax(-1) != ref["feat"]["stoch"].argmax(-1)).sum())
        put(flips == 0 and ref["ties"] <= nlat // 100,
            f"[{tag}] samples: {flips} flips of {nlat}; near-ties fed golden {ref['ties']} (<= 1%)")
        drift = [float(np.abs(o["feat"]["deter"][:, t] - ref["feat"]["deter"][:, t]).max()) for t in (1, 8, H)]
        print(f"        [{tag}] deter |d| at t=1,8,{H}: " + ", ".join(f"{d:.2g}" for d in drift))
        if tag == "policy":
            put(*close("[policy] continuous actions vs f64 (H+1, incl. the last)", o["act"][CONT], ref["act"][CONT]))
            put(ref["dflips"] == 0 and bool(np.array_equal(o["act"][DISC], ref["act"][DISC])),
                f"[policy] discrete actions equal (argmax flips {ref['dflips']})")
            sq = float((np.abs(o["act"][CONT]) > 1).mean())
            put(sq > 0.2, f"[policy] squash binds: {sq:.0%} of continuous actions |a| > 1 (need > 20%)")
    # imagined futures must depend on the actions: a pass with other actions must differ
    other = REF_imagine(P, start, U["array"], action_array(N, seed=5))
    d = float(np.abs(other["feat"]["deter"][:, -1] - out["array"]["feat"]["deter"][:, -1]).max())
    put(d > 1e-2, f"control: other actions -> other imagined deter at H ({d:.3g} > 1e-2)")
    imut = {
        "no action squash": dict(squash=False),
        "no unimix in the prior sampler": dict(unimix=0.0),
        "noise read one call late": dict(U=np.concatenate([U["policy"][1:], U["policy"][:1]])),
        "policy reads the carry one step late": dict(lag=True),
        "start not prepended (feat one step early)": dict(prepend=False),
    }
    for name, kw in imut.items():
        UU = kw.pop("U", U["policy"])
        m = REF_imagine(P, start, UU, None, **kw)
        if not kw.get("prepend", True):
            m["feat"] = {k: np.concatenate([v, v[:, -1:]], 1) for k, v in m["feat"].items()}
        o = out["policy"]
        bad = not (close("m", o["feat"]["deter"], m["feat"]["deter"])[0]
                   and close("m", o["act"][CONT], m["act"][CONT])[0]
                   and np.array_equal(o["feat"]["stoch"].argmax(-1), m["feat"]["stoch"].argmax(-1)))
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")

    # ---- (II) returns through the official imag_loss / repl_loss
    R = ret_inputs()

    class Head:
        def __init__(self, p): self.p = jnp.asarray(p)
        def pred(self): return self.p
        def loss(self, x): return jnp.ones_like(self.p)

    class Pol:
        def __init__(self, n): self.n = n
        def logp(self, x): return jnp.zeros(self.n)
        def entropy(self): return jnp.zeros(self.n)

    class Norm:
        def stats(self): return jnp.float32(0), jnp.float32(1)
        def __call__(self, x, update): return jnp.float32(0), jnp.float32(1)

    nm = Norm()
    S = R["rew"].shape
    iact = {"a": jnp.zeros(S)}

    def imag(contdisc, lam=LAM):
        los, o, _ = A.imag_loss(iact, jnp.asarray(R["rew"]), jnp.asarray(R["con"]), {"a": Pol(S)},
                                Head(R["val"]), Head(R["slow"]), nm, nm, nm, update=False,
                                contdisc=contdisc, horizon=HORIZON, slowtar=False, lam=lam,
                                actent=3e-4, slowreg=1.0)
        return dict(ret=np.asarray(o["ret"]), weight=np.asarray(los["value"]) / 2.0)

    def repl(lam=LAM):
        los, o, _ = A.repl_loss(jnp.asarray(R["last"]), jnp.asarray(R["term"]), jnp.asarray(R["rrew"]),
                                jnp.asarray(R["rboot"]), Head(R["rval"]), Head(R["rslow"]), nm, update=False,
                                slowreg=1.0, slowtar=False, horizon=HORIZON, lam=lam)
        return dict(ret=np.asarray(o["ret"]), weight=np.asarray(los["repval"]) / 2.0)

    with jax.disable_jit():
        oi, oiD, orp = imag(True), imag(False), repl()
        oi1, oi0 = imag(True, 1.0), imag(True, 0.0)
    ri, riD, rr = REF_imag(R["rew"], R["con"], R["val"]), REF_imag(R["rew"], R["con"], R["val"], False), \
        REF_repl(R["last"], R["term"], R["rrew"], R["rval"], R["rboot"])
    for tag, o, r in (("imag contdisc", oi, ri), ("imag disc 1-1/333", oiD, riD), ("repl", orp, rr)):
        put(*close(f"{tag} ret f32 vs f64", o["ret"], r["ret"]))
        put(*close(f"{tag} weight f32 vs f64", o["weight"], r["weight"][:, :-1]))
    # the reference itself against two closed forms (lam 1: Monte Carlo + bootstrap; lam 0: one-step TD)
    con, rew, val = np.float64(R["con"]), np.float64(R["rew"]), np.float64(R["val"])
    mc = np.zeros((len(rew), H))
    for t in range(H):
        g, acc = np.ones(len(rew)), np.zeros(len(rew))
        for j in range(t + 1, H + 1):
            acc += g * rew[:, j]
            g = g * con[:, j]
        mc[:, t] = acc + g * val[:, H]
    put(*close("lam 1 official == Monte Carlo + bootstrap (closed form)", oi1["ret"], mc))
    td = rew[:, 1:] + con[:, 1:] * val[:, 1:]
    put(*close("lam 0 official == one-step TD (closed form)", oi0["ret"], td))
    rl = float(orp["weight"].min()), float((R["last"][:, :-1]).mean())
    put(rl[0] == 0.0 and 0 < rl[1] < 0.5, f"repl: is_last rows weighted 0 ({rl[1]:.0%} of steps)")
    m_r = lambda **kw: REF_lambda_return(**{**dict(last=np.zeros_like(con), term=1 - con, rew=rew, val=val,  # noqa: E731
                                                     boot=val, disc=1.0, lam=LAM), **kw})
    rmut = {
        "reward read one step early (rew[:, :-1])": m_r(rew=np.concatenate([rew[:, :1], rew[:, :-1]], 1)),
        "lam on the bootstrap, not the continuation": m_r(lam=1 - LAM),
        "disc 1-1/333 under contdisc": m_r(disc=1 - 1 / HORIZON),
        "term from con not applied (live = disc)": m_r(term=np.zeros_like(con)),
        "boot = last value only (no per-step bootstrap)": m_r(boot=np.repeat(val[:, -1:], H + 1, 1)),
    }
    for name, v in rmut.items():
        bad = not close("m", oi["ret"], v)[0]
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")
    wmut = {"weight without the /disc": np.cumprod((1 - 1 / HORIZON) * con, 1),
            "weight exclusive (weight_0 = 1)": np.concatenate([np.ones((len(con), 1)), np.cumprod(con, 1)[:, :-1]], 1) / (1 - 1 / HORIZON)}
    for name, w in wmut.items():
        bad = not close("m", oiD["weight"], w[:, :-1])[0]
        put(bad, f"planted '{name}' (disc 1-1/333 pass): {'CAUGHT' if bad else 'MISSED'}")
    rrm = REF_lambda_return(np.zeros_like(R["last"]), R["term"], R["rrew"], R["rval"], R["rboot"], 1 - 1 / HORIZON, LAM)
    put(not close("m", orp["ret"], rrm)[0], "planted 'repl ignores is_last': "
        + ("CAUGHT" if not close("m", orp["ret"], rrm)[0] else "MISSED"))

    if "--dump-golden" in sys.argv:
        g = {}
        for tag in ("array", "policy"):
            for k, v in out[tag]["feat"].items():
                g[f"{tag}/feat/{k}"] = v
            for k, v in out[tag]["act"].items():
                g[f"{tag}/act/{k}"] = v
            g[f"{tag}/uniform"] = U[tag].astype(np.float32)
        for tag, o in (("imag", oi), ("imagD", oiD), ("repl", orp)):
            g[f"{tag}/ret"], g[f"{tag}/weight"] = o["ret"], o["weight"]
        np.savez_compressed(GOLDEN10, **g, **{f"ret_in/{k}": v for k, v in R.items()})
        print(f"golden written: {GOLDEN10}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, size1m, N {N}, H {H}): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    print("port tests not written yet (next slot: committed failing, then mlx_maths)")
    sys.exit(2)
