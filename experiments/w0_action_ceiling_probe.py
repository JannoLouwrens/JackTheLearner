"""w0_action_ceiling_probe.py — how much do his actions move W0's observation
K decisions later at all? (ORDERS J3, 2026-11-13; records nothing.)

Why. ada42ca: training wm-dyn makes Delta_5 (his actions' effect on his own
5-step roll) FALL on 3/3 seeds. Either the arm washes real action information
out, or W0 under a random policy carries little to learn. This measures the
world side — the oracle ceiling — from the world itself, no model involved.

Method. A random-policy trajectory (the LC.09 probe's policy: i.i.d.
uniform actions, one per decision) lives in W0. At N start states mid-life the
world is DEEP-COPIED and rolled K decisions under: the real actions (must
reproduce the recorded obs bit-for-bit: the determinism check), zeros, and M
fresh random action sequences; plus M HELD sequences (one random action held
K decisions, as survival.py's explorer holds HOLD_K). Branches that die are
dropped. For each modality (per-dim standardised by the pooled spread):

  frac_action = mean_s var_branches(obs_{t+K} | s) / var_all(obs_{t+K})

the share of obs_{t+K}'s variance that the choice of action sequence explains
given the state. 0 = actions are invisible K decisions later; ~1 = state is
irrelevant. The placebo channel is drawn from the world RNG, which the copy
carries, so it is identical across branches by construction (its frac reads
0: a built-in known-negative). Run:
    python -m experiments.w0_action_ceiling_probe [seed] [K]

MEASURED 2026-11-13 19:15 (n_dec 600, 40 starts, M 8, ~90 CPU-s per seed;
determinism failures 0/120 starts: the clone replays the recording exactly):
  K=5, i.i.d. actions, whole obs: 0.656 / 0.690 / 0.622 (seeds 0/1/2);
       held actions: 0.711 / 0.718 / 0.690.
       per modality (iid, s0/1/2): vision .56/.62/.54, audio .82/.79/.76,
       touch .71/.71/.65, proprio .87/.89/.88, needs .44/.52/.36.
  K=1 (seed 0): whole obs 0.427 (vision .28, proprio .71, needs .27).
  placebo .021/.034/.000 (K=5), .003 (K=1): not exactly 0 because the world
  RNG is consumed action-dependently somewhere in decide(), so placebo
  diverges in a few branches — read ~.03 as the measure's floor.
READING: the world is NOT the limit. Two-thirds of what he senses 5 decisions
on is set by what he does (43% one decision on), so ada42ca's falling Delta_5
is the wm-dyn ARM (or Delta_5's cosine-on-state metric) washing action
information out, not W0 failing to carry it.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np

KEYS = ("vision", "audio", "touch", "proprio", "needs")


def _clone(w):
    """Deep copy of a live world; modules (W0 keeps `mujoco` as an attribute)
    are shared, not copied."""
    import types
    memo = {id(v): v for v in vars(w).values() if isinstance(v, types.ModuleType)}
    return copy.deepcopy(w, memo)


def probe(seed: int = 0, K: int = 5, n_dec: int = 600, n_starts: int = 40,
          M: int = 8) -> dict:
    from .w0 import W0, random_action
    from .tests.lc_03_survival_screening import _borrow
    from .cores import PLACEBO_KEY
    t0 = time.process_time()
    cal, prov = _borrow()
    assert cal is not None, prov
    w = W0(seed=seed, j0=cal["j0"], alpha=cal["alpha"], lethal=True)
    rng = np.random.RandomState(seed)
    brng = np.random.RandomState(10_000 + seed)
    starts = set(int(i) for i in np.linspace(20, n_dec - K - 2, n_starts))
    rows, det_fail = [], 0
    pending = {}   # start index -> (clone, real actions so far)
    life = 0
    hist_obs, hist_a, hist_life = [], [], []
    for i in range(n_dec):
        if i in starts:
            pending[i] = _clone(w)
        obs = w.observe()
        a = random_action(rng).astype(np.float32)
        w.decide(a)
        hist_obs.append(obs); hist_a.append(a); hist_life.append(life)
        if w.died_this_decision:
            life += 1
    # recorded obs at t+K is hist_obs[t+K] (observed before decision t+K)
    for s, clone in sorted(pending.items()):
        if s + K >= n_dec or hist_life[s] != hist_life[s + K]:
            continue

        def run(acts):
            c = _clone(clone)
            c.observe()   # keep the world RNG in step with the recording
            for k, a in enumerate(acts):
                c.decide(a)
                if c.died_this_decision:
                    return None
                if k < K - 1:
                    c.observe()
            return c.observe()
        real = run([hist_a[s + k] for k in range(K)])
        if real is None:
            continue
        same = all(np.array_equal(real[k], hist_obs[s + K][k])
                   for k in KEYS + (PLACEBO_KEY,))
        det_fail += int(not same)
        br = {"real": real, "zero": run([np.zeros(8, np.float32)] * K)}
        iid = [run([random_action(brng).astype(np.float32) for _ in range(K)])
               for _ in range(M)]
        held = []
        for _ in range(M):
            a = random_action(brng).astype(np.float32)
            held.append(run([a] * K))
        rows.append({"s": s, "real": real, "zero": br["zero"],
                     "iid": [o for o in iid if o is not None],
                     "held": [o for o in held if o is not None]})
    out = {"seed": seed, "K": K, "decisions": n_dec, "lives": life + 1,
           "starts_used": len(rows), "determinism_failures": det_fail}

    def frac(kind, key):
        allv = np.stack([o[key] for r in rows for o in r[kind] + [r["real"]]])
        tot = allv.var(0)
        keep = tot > 1e-12
        if not keep.any():
            return 0.0, 0
        within = np.mean([np.stack([o[key] for o in r[kind] + [r["real"]]])
                          .var(0)[keep] for r in rows if len(r[kind]) >= 1], 0)
        return float((within / tot[keep]).mean()), int(keep.sum())
    for kind in ("iid", "held"):
        out[f"frac_action_{kind}"] = {}
        for key in KEYS + (PLACEBO_KEY,):
            f, nd = frac(kind, key)
            out[f"frac_action_{kind}"][key] = {"frac": round(f, 4), "dims_varying": nd}
    # whole observation (sensory, placebo/language excluded), dims pooled
    for kind in ("iid", "held"):
        num = den = 0.0
        for key in KEYS:
            allv = np.stack([o[key] for r in rows for o in r[kind] + [r["real"]]])
            tot = allv.var(0); keep = tot > 1e-12
            if keep.any():
                within = np.mean([np.stack([o[key] for o in r[kind] + [r["real"]]])
                                  .var(0)[keep] for r in rows], 0)
                num += float((within / tot[keep]).sum()); den += int(keep.sum())
        out[f"frac_action_{kind}_all"] = round(num / max(den, 1), 4)
    out["cpu_s"] = round(time.process_time() - t0, 1)
    return out


if __name__ == "__main__":
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(json.dumps(probe(seed=sd, K=K), indent=1))
