"""MLX port of DreamerV3 — rung 9 (PORT report step 2.4, last world-model loss row): the FULL
world-model loss on one golden batch. References `eb440755`; port tests below (module next).

Official code `danijar/dreamerv3` at e01491f:
  agent.py L80-83   scales = config.loss_scales.copy(); rec = scales.pop('rec');
                    scales.update({k: rec for k in dec_space})          <- rec is PER obs key
  agent.py L163-185 losses: dyn, rep (rssm.loss, rung 8a), rew, con (rung 8b), one per dec key (8c)
  agent.py L240     loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
  configs.yaml      loss_scales {rec 1, rew 1, con 1, dyn 1, rep 0.1, policy 1, value 1, repval 0.3}
The world-model part is the sum over {dyn, rep, rew, con, <dec keys>}; policy / value / repval
(imagination + replay value losses) are NOT in this rung — said now; they are a later row.

What is checked: (a) the scales are READ from the pinned configs.yaml and the three official source
lines above are present verbatim (the pin has not drifted); (b) REF_wm (float64, recomputing every term
from rungs 8a/8b/8c's references on the same [4,64] batch) vs the total composed from the official f32
goldens of 8a/8b/8c by L240's formula, at both param sets (published init; seeded N(0,0.3) noise).
The goldens are each official module's own output (8a 24/24, 8b, 8c 24/24 vs official); the
composition here is L240's arithmetic, not a fresh official Agent.loss call (that needs the encoder
and imagination — not ported). Ours beside theirs: f32 (published bf16), size1m widths, toy inputs.
Tolerance as 8a-8c: relative 1e-5 on the total.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \\
      experiments/brain_a/rung9.py --vs-official [--dump-golden]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rung8a  # noqa: E402  (appends --preset size1m; imports rung7)
import rung8b  # noqa: E402
import rung8c  # noqa: E402
from rung1 import OFFICIAL  # noqa: E402

GOLDEN9 = os.path.join(HERE, "rung9_wm_golden_size1m_e01491f.npz")
WM_KEYS = ("dyn", "rep", "rew", "con")
SRC_LINES = ("    rec = scales.pop('rec')",
             "    scales.update({k: rec for k in dec_space})",
             "    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])")


def official_scales(dec_keys):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(OFFICIAL, "dreamerv3", "configs.yaml")))
    s = dict(cfg["defaults"]["agent"]["loss_scales"])
    rec = s.pop("rec")
    s.update({k: rec for k in dec_keys})
    return {k: float(v) for k, v in s.items() if k in WM_KEYS or k in dec_keys}


def wm_total(terms, scales, mean=np.mean):
    return sum(mean(terms[k]) * scales[k] for k in scales)


def golden_terms(pset):
    """Official f32 [B,T] losses from the three goldens; pset 'init' or 'noised'."""
    A, Bg, C = np.load(rung8a.GOLDEN8), np.load(rung8b.GOLDEN8), np.load(rung8c.GOLDEN8C)
    t = dict(dyn=A["dyn"], rep=A["rep"])
    t.update(rew=Bg["rew0"], con=Bg["con0"]) if pset == "init" else t.update(rew=Bg["rew"], con=Bg["con"])
    t.update({k: C[f"{'loss0' if pset == 'init' else 'lossN'}/{k}"] for k in rung8c.VEC})
    return t


def REF_terms(pset):
    _, _, _, g7 = rung7_golden()
    t = {k: v for k, v in rung8a.REF_dynrep(g7["post"], g7["prior"]).items() if k in ("dyn", "rep")}
    feat, rew, term = rung8b.inputs()
    P0b = rung8b._golden_params()
    Pb = P0b[1] if pset == "init" else P0b[2]
    t.update({k: v for k, v in rung8b.REF_losses(Pb, feat, rew, term).items() if k in ("rew", "con")})
    G, P0c, PNc = rung8c._golden()
    deter, stoch = rung8c.feats()
    d = rung8c.REF_dec(P0c if pset == "init" else PNc, deter, stoch, rung8c.targets(*deter.shape[:2]))
    t.update({k: d[k] for k in rung8c.VEC})
    return t


def rung7_golden():
    return rung8a.rung7._load_golden()


def rel(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1)


def main_official():
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    src = open(os.path.join(OFFICIAL, "dreamerv3", "agent.py")).read().splitlines()
    for ln in SRC_LINES:
        put(ln in src, f"official agent.py has: {ln.strip()}")
    S = official_scales(tuple(rung8c.VEC))
    put(S == {"dyn": 1.0, "rep": 0.1, "rew": 1.0, "con": 1.0, **{k: 1.0 for k in rung8c.VEC}},
        f"scales read from configs.yaml: {S}")
    out = {}
    for pset in ("init", "noised"):
        g, r = golden_terms(pset), REF_terms(pset)
        for k in S:
            put(*rung8a.close(f"{pset}: term {k} REF vs official golden", g[k], r[k]))
        gt = sum(np.float32(g[k].mean(dtype=np.float32)) * np.float32(S[k]) for k in S)  # L240 in f32
        rt = wm_total(r, S)
        put(rel(gt, rt) <= 1e-5, f"{pset}: WM total official-composed {float(gt):.7g} vs REF {rt:.7g} "
                                 f"(rel {rel(gt, rt):.3g} <= 1e-5)")
        out[pset] = (g, r, float(gt), rt)
        muts = {
            "rep scale 1.0 for 0.1": {**S, "rep": 1.0},
            "rec once on the summed keys / 3": {**S, **{k: 1 / 3 for k in rung8c.VEC}},
            "con dropped": {k: v for k, v in S.items() if k != "con"},
        }
        # dyn and rep carry the SAME forward value (rung 8a: the stop-gradients differ only in the
        # backward pass), so swapping their scales is INVISIBLE to any forward check. Printed, not
        # counted: the port tests owe a GRADIENT check (prior side gets 1.0, posterior side 0.1).
        sw = wm_total(r, {**S, "dyn": 0.1, "rep": 1.0})
        print(f"  --  {pset}: 'dyn 0.1 / rep 1.0 swapped' is forward-BLIND by construction "
              f"(rel {rel(sw, gt):.2g}) -> gradient check owed")
        for name, m in muts.items():
            bad = rel(wm_total(r, m), gt) > 1e-5
            put(bad, f"{pset}: planted '{name}': {'CAUGHT' if bad else 'MISSED'} ({wm_total(r, m):.6g})")
        bad = rel(sum(r[k].mean(1).sum() * S[k] for k in S), gt) > 1e-5
        put(bad, f"{pset}: planted 'sum over batch, mean over time': {'CAUGHT' if bad else 'MISSED'}")
    _, _, ti, _ = out["init"]
    _, _, tn, _ = out["noised"]
    put(rel(ti, tn) > 1e-3, f"the two param sets' totals differ ({ti:.6g} vs {tn:.6g}) — not blind")
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN9, total_init=np.float32(ti), total_noised=np.float32(tn),
                 scale_keys=np.array(list(S)), scale_vals=np.array(list(S.values()), np.float32))
        print(f"golden written: {GOLDEN9}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (e01491f, size1m, WM loss composition): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


# ---------------------------------------------------------------- port tests (MLX)
# `mlx_maths.wm_loss(losses, scales)` -> scalar: L240, sum over scales' keys of mean(losses[k]) *
# scales[k] (a key in scales but not in losses is an error, not a zero). Each test eager AND under
# mx.compile:
#   wm_total   port on the official f32 [B,T] terms (8a/8b/8c goldens) vs rung 9's golden total,
#              both param sets, rel 1e-5 (the --vs-official tolerance)
#   wm_grad    GRADIENT ROUTING, the check the forward is blind to (dyn and rep carry one value):
#              total = wm_loss(kl_dyn_rep(post, prior) + the other golden terms) on a [1,2] slice of
#              rung 7's size1m post/prior logits (256 per side); mx.grad wrt prior must equal
#              scale_dyn (1.0) x d mean(KL)/d prior, wrt post scale_rep (0.1) x d mean(KL)/d post,
#              both from rung1.REF_kl_grads (float64 central differences). The FD gradients are
#              non-zero (not blind) and the two expected sides differ by 10x, so a swap cannot pass.
# Patch points the planted harness uses: wm_loss's _wm_mean, and kl_dyn_rep itself.
SLICE = (slice(0, 1), slice(0, 2))


def _kl_inputs():
    _, _, _, g7 = rung7_golden()
    return np.float32(g7["post"][SLICE]), np.float32(g7["prior"][SLICE])


def golden_scales():
    """The scales --vs-official read from configs.yaml and wrote into the golden (the port side has
    no yaml / official checkout — on the box's jack-mlx env nor on the Kaggle wheel)."""
    G = np.load(GOLDEN9)
    return dict(zip([str(k) for k in G["scale_keys"]], [float(v) for v in G["scale_vals"]]))


def t_total(M, mx, compiled):
    G = np.load(GOLDEN9)
    S = golden_scales()
    pub = {"dyn": 1.0, "rep": 0.1, "rew": 1.0, "con": 1.0, **{k: 1.0 for k in rung8c.VEC}}
    rows = [(set(S) == set(pub) and all(np.float32(S[k]) == np.float32(pub[k]) for k in pub),
             f"golden's scales are the published ones (f32): {S}")]
    f = (lambda t: M.wm_loss(t, S))
    if compiled:
        f = mx.compile(f)
    for pset, gk in (("init", "total_init"), ("noised", "total_noised")):
        g = golden_terms(pset)
        got = float(np.asarray(f({k: mx.array(v) for k, v in g.items()})))
        rows.append((rel(got, G[gk]) <= 1e-5, f"{pset}: port total {got:.7g} vs golden {float(G[gk]):.7g} "
                                                f"(rel {rel(got, G[gk]):.3g} <= 1e-5)"))
    nc = float(np.asarray(M.wm_loss({k: mx.array(v) for k, v in golden_terms("init").items()},
                                    {k: v for k, v in S.items() if k != "con"})))
    rows.append((rel(nc, G["total_init"]) > 1e-5, f"control: con's scale dropped moves the total ({nc:.6g})"))
    return rows   # dyn vs rep cannot be told apart here (one value): wm_grad is that check


def t_grad(M, mx, compiled):
    import rung1
    post, prior = _kl_inputs()
    mix, free = rung8a.TOY["unimix"], rung8a.FREE
    S = golden_scales()
    g = golden_terms("noised")
    rest = {k: mx.array(v) for k, v in g.items() if k not in ("dyn", "rep")}
    n = float(np.prod(post.shape[:2]))

    def total(p, q):
        dyn, rep = M.kl_dyn_rep(p, q, mix, free)
        return M.wm_loss({**rest, "dyn": dyn, "rep": rep}, S)

    gf = mx.grad(total, argnums=(0, 1))
    if compiled:
        gf = mx.compile(gf)
    gp, gq = (np.asarray(x) for x in gf(mx.array(post), mx.array(prior)))
    d_prior, d_post = rung1.REF_kl_grads(np.float64(post), np.float64(prior), mix, free)
    d_prior, d_post = d_prior / n, d_post / n                       # mean over [B,T], not sum
    rows = [(float(np.abs(d_prior).max()) > 1e-4 and float(np.abs(d_post).max()) > 1e-4,
             f"FD gradients non-zero (|d prior| {np.abs(d_prior).max():.3g}, |d post| {np.abs(d_post).max():.3g})"),
            (float(np.abs(rung1.REF_kl(np.float64(post), np.float64(prior), mix, 0.0)[1]).min()) > free,
             "the free-bits floor does not bind on the slice (else the gradient is zero by design)")]
    rows.append(rung8a.close(f"d total/d prior == scale_dyn {S['dyn']} x FD", gq, S["dyn"] * d_prior))
    rows.append(rung8a.close(f"d total/d post == scale_rep {S['rep']} x FD", gp, S["rep"] * d_post))
    sw = float(np.abs(gp - S["dyn"] * d_post).max())
    rows.append((sw > 1e-4, f"control: the swapped routing (post x {S['dyn']}) differs ({sw:.3g} > 1e-4)"))
    return rows


TESTS = [("wm_total", t_total), ("wm_grad", t_grad)]


def _mlx():
    import rung6
    return rung6._mlx()


def main_mlx():
    mx = _mlx()
    import mlx_maths as M
    n = 0
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c)
                r[mode] = all(o for o, _ in rows)
                det = [d for o, d in rows if not o][:4] if not r[mode] else [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if mode == "eager" or not r[mode]:
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        print(f"{name:14s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    print(f"RUNG 9: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


def main_planted():
    """Each planted bug is patched into the real port; the tests must FAIL eager."""
    mx = _mlx()
    import mlx_maths as M
    keep = {k: getattr(M, k) for k in ("_wm_mean", "kl_dyn_rep", "wm_loss")}

    def swapped(p, q, mix, free):                   # dyn <-> rep: same values, routing swapped
        d, r = keep["kl_dyn_rep"](p, q, mix, free)
        return r, d

    def no_sg(p, q, mix, free):                     # both sides get gradient from both terms
        d = mx.maximum(M._cat_kl(M.unimix_logits(p, mix), M.unimix_logits(q, mix)).sum(-1), free)
        return d, d

    bugs = {
        "sum over [B,T], not mean": dict(_wm_mean=lambda x: mx.sum(x)),
        "mean over time only, then sum over batch": dict(_wm_mean=lambda x: mx.sum(mx.mean(x, 1))),
        "dyn and rep swapped (forward-blind)": dict(kl_dyn_rep=swapped),
        "no stop-gradients in the KL": dict(kl_dyn_rep=no_sg),
        "rep's 0.1 ignored (every scale 1)": dict(wm_loss=lambda L, S: keep["wm_loss"](L, {k: 1.0 for k in S})),
    }
    caught = 0
    for name, patch in bugs.items():
        for k, v in patch.items():
            setattr(M, k, v)
        try:
            bad = [d for _, t in TESTS for o, d in t(M, mx, False) if not o]
        except Exception as e:  # noqa: BLE001
            bad = [f"crash {type(e).__name__}: {e}"]
        finally:
            for k, v in keep.items():
                setattr(M, k, v)
        caught += bool(bad)
        print(f"planted [{name}] -> {'CAUGHT' if bad else 'MISSED'}" + (f"  ({bad[0][:80]})" if bad else ""))
    ctl = all(o for _, t in TESTS for o, _ in t(M, mx, False))
    print(f"PLANTED: {caught}/{len(bugs)} caught, unpatched port {'passes' if ctl else 'FAILS'} -> "
          f"{'OK' if caught == len(bugs) and ctl else 'HARNESS BLIND'}")
    return 0 if caught == len(bugs) and ctl else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    if "--planted" in sys.argv[1:]:
        sys.exit(main_planted())
    sys.exit(main_mlx())
