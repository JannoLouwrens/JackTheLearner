"""MLX port of DreamerV3 — rung 9 (PORT report step 2.4, last world-model loss row): the FULL
world-model loss on one golden batch — REFERENCES ONLY in this commit (port tests + module next).

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


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    print("rung 9: port tests not written yet (next commit); run --vs-official")
    sys.exit(2)
