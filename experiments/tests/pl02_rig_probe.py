"""PL.02 rig probe — decompose the seed-90 smoke VOID before re-smoking.

NOT A SPEC. Diagnostic only, per the PL.00 lesson (a cost/failure decomposes
before the substrate gets blamed). The smoke (2026-09-07, VOID,
/data/pl02_smoke_seed90.log) fired two rig gates with one suspected root:
r2_ua -0.0039 vs the 0.80 eye bar while loss_drop_ua 0.0083 aced the pretext
— the encoder solves masked reconstruction without encoding the object, which
at RES=64 subtends only ~2-9 px (0.1-1.5% of the frame; PG.6's own pilot
chose RES=96). The plastic arm's loss rise shares the root: predicting audio
from a radius-blind visual latent is unlearnable, so its loss floor is the
audio marginal.

QUESTIONS, each a measurement:
  raw64 / raw96   Do RAW pixels at the coarse (PL.00-adopted) quality carry
                  radius at PG.6's 0.80 bar? Splits eye-blindness from
                  encoder loss. PG.6's certificate is raw pixels at 96 px at
                  the OLD quality; nothing has measured the coarse eye.
  ua64m60         U_A at RES=64, MASK_FRAC=0.60 — does masking pressure
                  alone rescue the 64 px operating point?
  ua96m35         U_A at RES=96, smoke's 0.35 masking — does resolution
                  alone rescue it?
  ua96m60         Both.

Feature R^2 >= 0.80 on any config makes that config the smoke's next
operating point (rig aliveness tuning on seed 90, disjoint from registered
seeds, no verdict threshold involved). All raw probes < 0.80 would be an
eye-quality finding for the two-eyes-one-certified queue row instead.

Artifact: /data/pl02_rig_probe.json (written incrementally, crash-safe).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from ..render import ensure_gl

ensure_gl()

from ..tests import pl_02_reshaping_gain as pl02  # noqa: E402
from .pg_6_playground_eyes import (IN_FOV_MAX, _r2, _Ridge,  # noqa: E402
                                   _sample_unoccluded)

SEED = 90
N_PRETEXT = 1200
N_TR, N_TE = 600, 400
STEPS, BATCH, LR = 1200, 64, 1e-3
ART = "/data/pl02_rig_probe.json"

_EYES: dict = {}   # held for process lifetime — a GC'd Renderer poisons GLX


def eye_at(res: int) -> pl02._CoarseEye:
    if res not in _EYES:
        _EYES[res] = pl02._CoarseEye(SEED, res=res)
    return _EYES[res]


def episodes(res: int, tag: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(grey frames in [0,1], radii). Same sampler as the spec, no audio."""
    e = eye_at(res)
    rng = np.random.RandomState(SEED * 271 + 3 + tag)
    frames = np.empty((n, res, res), dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    for i in range(n):
        b, d, r, _ = _sample_unoccluded(e, rng, 0.0, IN_FOV_MAX, signed=True)
        frames[i] = e.frame(b, d, r).mean(axis=2)
        radii[i] = r
    return frames, radii


def probe_r2(Xtr, ytr, Xte, yte) -> float:
    fit = _Ridge(Xtr, l2=pl02.L2)
    return float(_r2(yte, fit.predict(ytr, Xte)))


def build_ua(torch, res: int, ch: int = 1):
    nn = torch.nn
    torch.manual_seed(SEED * 1009 + 7)
    feat = 64 * (res // 16) ** 2
    return nn.Sequential(
        nn.Conv2d(ch, 16, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(feat, pl02.Z_A)), feat


def train_ua(torch, res: int, mask_frac: float, frames: np.ndarray,
             ch: int = 1) -> dict:
    """Masked-AE U_A arm at (res, mask_frac); returns encoder + loss drop.

    `frames` is (n, res, res) grey for ch=1, (n, res, res, 3) for ch=3."""
    nn = torch.nn
    enc, feat = build_ua(torch, res, ch)
    dec = nn.Sequential(
        nn.Linear(pl02.Z_A, feat), nn.ReLU(),
        nn.Unflatten(1, (64, res // 16, res // 16)),
        nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(16, ch, 4, 2, 1))
    if ch == 3:
        x_all = torch.from_numpy(
            np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))
    else:
        x_all = torch.from_numpy(frames).unsqueeze(1)
    n = x_all.shape[0]
    g = res // pl02.PATCH
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=LR)
    rng = np.random.RandomState(SEED * 31)
    torch.manual_seed(SEED * 7919 + 11)
    first, last = None, None
    for _ in range(STEPS):
        idx = torch.from_numpy(rng.randint(0, n, BATCH))
        x = x_all[idx]
        keep = torch.from_numpy(
            (rng.rand(BATCH, g, g) >= mask_frac).astype(np.float32))
        m = keep.repeat_interleave(pl02.PATCH, 1).repeat_interleave(
            pl02.PATCH, 2)
        xm, hole = x * m.unsqueeze(1), (1.0 - m).unsqueeze(1)
        recon = dec(enc(xm))
        loss = ((recon - x) ** 2 * hole).sum() / hole.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    return {"enc": enc, "loss_drop": round(last / max(first, 1e-12), 4)}


def features(torch, enc, frames: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        if frames.ndim == 4:                    # RGB: (n, res, res, 3)
            x = torch.from_numpy(
                np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))
        else:
            x = torch.from_numpy(frames).unsqueeze(1)
        out = [enc(x[i:i + 256]).numpy() for i in range(0, x.shape[0], 256)]
    return np.concatenate(out).astype(np.float64)


def main():
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    torch = pl02._torch()
    out: dict = {"seed": SEED, "steps": STEPS, "n_pretext": N_PRETEXT,
                 "n_probe": [N_TR, N_TE]}

    def flush():
        with open(ART, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)

    data: dict = {}
    for res in (64, 96):
        t0 = time.process_time()
        pre, _ = episodes(res, 0, N_PRETEXT)
        xtr, ytr = episodes(res, 1, N_TR)
        xte, yte = episodes(res, 2, N_TE)
        data[res] = (pre, xtr, ytr, xte, yte)
        raw = probe_r2(xtr.reshape(N_TR, -1).astype(np.float64), ytr,
                       xte.reshape(N_TE, -1).astype(np.float64), yte)
        out[f"raw{res}_radius_r2"] = round(raw, 4)
        out[f"raw{res}_cpu_s"] = round(time.process_time() - t0, 1)
        flush()
        print(f"raw{res}: R2={raw:.4f}", flush=True)

    for res, mf in ((64, 0.60), (96, 0.35), (96, 0.60)):
        key = f"ua{res}m{int(mf * 100)}"
        t0 = time.process_time()
        pre, xtr, ytr, xte, yte = data[res]
        r = train_ua(torch, res, mf, pre)
        Xtr = features(torch, r["enc"], xtr)
        Xte = features(torch, r["enc"], xte)
        fr2 = probe_r2(Xtr, ytr.astype(np.float64), Xte, yte.astype(np.float64))
        out[key] = {"feat_radius_r2": round(fr2, 4),
                    "loss_drop": r["loss_drop"],
                    "feat_std_min": round(float(Xte.std(0).min()), 6),
                    "cpu_s": round(time.process_time() - t0, 1)}
        flush()
        print(f"{key}: feat R2={fr2:.4f} loss_drop={r['loss_drop']}",
              flush=True)

    out["done"] = 1
    flush()
    print("DONE", out, flush=True)


def episodes_rgb(res: int, tag: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    e = eye_at(res)
    rng = np.random.RandomState(SEED * 271 + 3 + tag)
    frames = np.empty((n, res, res, 3), dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    for i in range(n):
        b, d, r, _ = _sample_unoccluded(e, rng, 0.0, IN_FOV_MAX, signed=True)
        frames[i] = e.frame(b, d, r)
        radii[i] = r
    return frames, radii


def main_rgb():
    """Second pass: PG.6's certificate is RGB raw pixels; the spec's pipeline
    is grey. Measure the raw RGB ceiling under the coarse quality at the
    spec's own probe split (1000/600) — the number the grey pass cannot give."""
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    ntr, nte = 1000, 600
    out = {"seed": SEED, "n_probe": [ntr, nte]}
    for res in (64, 96):
        t0 = time.process_time()
        xtr, ytr = episodes_rgb(res, 11, ntr)
        xte, yte = episodes_rgb(res, 12, nte)
        raw = probe_r2(xtr.reshape(ntr, -1).astype(np.float64), ytr,
                       xte.reshape(nte, -1).astype(np.float64), yte)
        out[f"rawrgb{res}_radius_r2"] = round(raw, 4)
        out[f"rawrgb{res}_cpu_s"] = round(time.process_time() - t0, 1)
        with open("/data/pl02_rgb_probe.json", "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"rawrgb{res}: R2={raw:.4f}", flush=True)
    print("DONE", out, flush=True)


def main_uargb():
    """Third pass: raw RGB@64 clears 0.80 (0.9327) where grey read 0.5614 —
    the grey conversion was the ceiling. Does the masked-AE ENCODER now
    capture it? U_A on RGB@64 at the smoke's 0.35 masking and at 0.60."""
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    torch = pl02._torch()
    out = {"seed": SEED, "steps": STEPS}
    pre, _ = episodes_rgb(64, 0, N_PRETEXT)
    xtr, ytr = episodes_rgb(64, 1, N_TR)
    xte, yte = episodes_rgb(64, 2, N_TE)
    for mf in (0.35, 0.60):
        key = f"uargb64m{int(mf * 100)}"
        t0 = time.process_time()
        r = train_ua(torch, 64, mf, pre, ch=3)
        Xtr = features(torch, r["enc"], xtr)
        Xte = features(torch, r["enc"], xte)
        fr2 = probe_r2(Xtr, ytr.astype(np.float64), Xte,
                       yte.astype(np.float64))
        out[key] = {"feat_radius_r2": round(fr2, 4),
                    "loss_drop": r["loss_drop"],
                    "feat_std_min": round(float(Xte.std(0).min()), 6),
                    "cpu_s": round(time.process_time() - t0, 1)}
        with open("/data/pl02_uargb_probe.json", "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"{key}: feat R2={fr2:.4f} loss_drop={r['loss_drop']}",
              flush=True)
    print("DONE", out, flush=True)


def main_steps():
    """Fourth pass: RGB@64 raw reads 0.9327 but U_A features read 0.0192 at
    1200 steps (loss_drop 0.0034 — pretext aced by memorising the CONSTANT
    background; the object is the only episode-varying content, so the
    residual loss late in training is entirely the object). Hypothesis: the
    encoder starts encoding the object only once the background is fully
    amortised — feature R^2 should climb with steps. Probe the curve; it
    decides whether STEPS is the operating-point repair or whether U_A
    saturates far below the B4 gate's 0.80 at every honest budget (which
    would be a gate-referent finding to route, not an op point)."""
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    torch = pl02._torch()
    nn = torch.nn
    out: dict = {"seed": SEED, "mask_frac": 0.35, "res": 64, "ch": 3,
                 "curve": {}}
    pre, _ = episodes_rgb(64, 0, N_PRETEXT)
    xtr, ytr = episodes_rgb(64, 1, N_TR)
    xte, yte = episodes_rgb(64, 2, N_TE)
    enc, feat = build_ua(torch, 64, 3)
    dec = nn.Sequential(
        nn.Linear(pl02.Z_A, feat), nn.ReLU(), nn.Unflatten(1, (64, 4, 4)),
        nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 4, 2, 1))
    x_all = torch.from_numpy(np.ascontiguousarray(pre.transpose(0, 3, 1, 2)))
    n = x_all.shape[0]
    g = 64 // pl02.PATCH
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=LR)
    rng = np.random.RandomState(SEED * 31)
    torch.manual_seed(SEED * 7919 + 11)
    t0 = time.process_time()
    for step in range(1, 6001):
        idx = torch.from_numpy(rng.randint(0, n, BATCH))
        x = x_all[idx]
        keep = torch.from_numpy(
            (rng.rand(BATCH, g, g) >= 0.35).astype(np.float32))
        m = keep.repeat_interleave(pl02.PATCH, 1).repeat_interleave(
            pl02.PATCH, 2)
        xm, hole = x * m.unsqueeze(1), (1.0 - m).unsqueeze(1)
        recon = dec(enc(xm))
        loss = ((recon - x) ** 2 * hole).sum() / hole.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1200, 2400, 3600, 4800, 6000):
            Xtr = features(torch, enc, xtr)
            Xte = features(torch, enc, xte)
            fr2 = probe_r2(Xtr, ytr.astype(np.float64), Xte,
                           yte.astype(np.float64))
            out["curve"][str(step)] = {
                "feat_radius_r2": round(fr2, 4),
                "loss": round(float(loss.detach()), 6),
                "cpu_s": round(time.process_time() - t0, 1)}
            with open("/data/pl02_steps_probe.json", "w") as f:
                json.dump(out, f, indent=1, sort_keys=True)
            print(f"step {step}: feat R2={fr2:.4f} "
                  f"loss={float(loss.detach()):.6f}", flush=True)
    out["done"] = 1
    with open("/data/pl02_steps_probe.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "rgb":
        main_rgb()
    elif mode == "uargb":
        main_uargb()
    elif mode == "steps":
        main_steps()
    else:
        main()
