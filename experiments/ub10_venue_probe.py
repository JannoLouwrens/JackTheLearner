"""UB.10 venue bakeoff — AUDIO-SIDE CPU PRE-PILOT (no renderer, no ledger row).

Ordered by the Review's 2026-09-29 ruling on
`ub10-part1-premise-false-marginals-are-what-saturate`: a bakeoff over
(a) shrink the audio observation window so `afell` is genuinely uncertain, and
(c) per-sense nuisance noise at a declared, matched level. Pre-registered success
criterion (the ruling's, not this file's): at least one marginal at <= 0.95 WITH
`slot = XNOR(vslot, afell)` intact, re-verified after the change. Not owed: any
edit to `hns_scene.py`, `A0_HEADROOM` or `_assert_venue_not_foreclosed()`.

What this file can and cannot say. This box has no GL (EGL and OSMesa both fail),
so the VISION half of (c) cannot be measured here. Arm (a) touches only audio and
(c)'s audio half is the noise on the audio stream alone, so the AUDIO marginal
`afell` can be measured for both arms on CPU in seconds. The ruling's arithmetic
says one marginal at <= 0.95 is enough (vslot at 1.0 and afell at q leaves A0's
union-bound floor at q). This is a SIZING measurement for the GPU bakeoff, not
the bakeoff itself: probes here are a ridge logreg (PG.7's `_logreg_cv`) and a
small torch MLP, NOT the A0 arm on tokens.

PILOT seed 90 only, through ub9._generate's own rng stream
(`seed * 7919 + 13`), so the episodes are the ones UB.10's selection may read.
Never run on a registered seed (0, 1, 2).

Run: python -m experiments.ub10_venue_probe [n_quads=400]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

PILOT_SEED = 90
N_TEST_QUADS = 80                 # ub9's split: the last 80 quads are test
EPISODE_S = 0.90                  # ub9.EPISODE_S (importing ub9 needs GL)
WINDOWS = (0.10, 0.02, 0.01, 0.005, 0.0025, 0.00125)   # arm (a); 0.10 = today
NOISE = (0.0, 0.5, 1.0, 2.0, 3.0)  # arm (c), in units of train-standardised sd
MARGINAL_CEIL = 0.95               # = 1 - UB.10's A0_HEADROOM
MARGINAL_FLOOR = 0.80              # = UB.10's learning gate on the marginals


def _features_at(stereo, sr, win: float):
    from .tests import pg_7_hns_leakage as pg7
    old = pg7.WIN_S
    pg7.WIN_S = win
    try:
        return pg7._features(stereo, sr)
    finally:
        pg7.WIN_S = old


def generate(n_quads: int, seed: int = PILOT_SEED) -> dict:
    """Audio features at every arm-(a) window from ONE render per episode."""
    sys.path.insert(0, str(REPO))
    import mujoco
    from ContactAudio import ContactAudioSynth
    from .hns_scene import build, draw_quad, listener_pose

    rng = np.random.RandomState(seed * 7919 + 13)
    F = {w: [] for w in WINDOWS}
    ya, yv, ys, qid = [], [], [], []
    q = dropped = 0
    while q < n_quads:
        quad = draw_quad(rng)
        rows, ok = [], True
        for ep in quad:
            model, data = build(ep)
            lpos, lyaw = listener_pose(ep)
            synth = ContactAudioSynth(model)
            synth.set_listener(lpos, lyaw)
            for _ in range(int(EPISODE_S / model.opt.timestep)):
                mujoco.mj_step(model, data)
                synth.step(data)
            if not synth.events:
                ok = False
                break
            audio = synth.render(duration=synth.events[0].t + 0.10 + 0.05)
            fs = {}
            for w in WINDOWS:
                f = _features_at(audio, synth.sr, w)
                if f is None:
                    ok = False
                    break
                b, lev, pan = f
                fs[w] = np.concatenate([b, [lev, pan]])
            if not ok:
                break
            rows.append((fs, int(ep.faller_radius > 0.18), int(ep.large_slot),
                         int(ep.faller_slot)))
        if not ok:
            dropped += 1          # whole quad, as ub9 does
            continue
        for fs, a, v, s in rows:
            for w in WINDOWS:
                F[w].append(fs[w])
            ya.append(a); yv.append(v); ys.append(s); qid.append(q)
        q += 1
    ya, yv, ys, qid = map(np.asarray, (ya, yv, ys, qid))
    return {"F": {w: np.asarray(F[w], np.float64) for w in WINDOWS},
            "afell": ya, "vslot": yv, "slot": ys, "quad": qid,
            "dropped": dropped,
            "xnor_mismatches": int((ys != np.where(ya == 1, yv, 1 - yv)).sum())}


def _mlp_acc(Xtr, ytr, Xte, yte, seed: int = 0, epochs: int = 150) -> float:
    """A0-like capacity on the audio vector alone: 14 -> 128 -> 128 -> 2."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(),
                        nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == yte).mean())


def measure(d: dict) -> dict:
    from .tests import pg_7_hns_leakage as pg7
    quads = d["quad"]
    te = np.isin(quads, np.unique(quads)[-N_TEST_QUADS:])
    tr = ~te
    nr = np.random.RandomState(1)
    out = {}
    for w in WINDOWS:
        X = d["F"][w]
        X = (X - X[tr].mean(0)) / np.maximum(X[tr].std(0), 1e-9)
        for s in NOISE:
            # noise is a property of the OBSERVATION: one fixed draw per episode
            Xn = X + s * nr.randn(*X.shape)
            out[f"w{w}_n{s}"] = {
                "logreg_cv": round(float(pg7._logreg_cv(Xn, d["afell"].astype(float),
                                                         quads)), 4),
                "mlp_test": round(_mlp_acc(Xn[tr], d["afell"][tr], Xn[te],
                                           d["afell"][te]), 4)}
    return out


def main(n_quads: int = 400) -> dict:
    t0 = time.time()
    d = generate(n_quads)
    acc = measure(d)
    res = {"seed": PILOT_SEED, "n_quads": n_quads, "n_ep": int(len(d["afell"])),
           "dropped": d["dropped"], "xnor_mismatches": d["xnor_mismatches"],
           "seconds": round(time.time() - t0, 1), "afell_acc": acc}
    # The ruling's criterion on the audio marginal, with both probes; the
    # harder (higher) of the two is the one that counts against the ceiling.
    res["arm_a_clears"] = sorted(
        k for k, v in acc.items() if k.endswith("_n0.0")
        and max(v.values()) <= MARGINAL_CEIL)
    res["arm_c_clears_full_window"] = sorted(
        k for k, v in acc.items() if k.startswith("w0.1_")
        and MARGINAL_FLOOR <= min(v.values()) and max(v.values()) <= MARGINAL_CEIL)
    return res


if __name__ == "__main__":
    os.chdir(REPO)
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 400), indent=1))
