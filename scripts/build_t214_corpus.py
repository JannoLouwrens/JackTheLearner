#!/usr/bin/env python
"""Regenerate T2.14's committed CMU corpus from the retarget cache on this box.

WHY A COMMITTED BLOB. `experiments/gpu.py:build_job` ships work to Kaggle/Colab
by CLONING THIS REPO at a pinned ref. That clone is the ONLY channel to the VM:
`/data/jack-data` (3.1 GB of CMU ASF/AMC, 391 retargeted clips) does not exist
there and `gpu.py` has no dataset-attach support. So a GPU spec whose claim is
about real recorded motion can only be run if the derived corpus travels IN the
repo. 5.7 MB compressed, force-added past `.gitignore`'s `*.npz`.

That is a deliberate precedent and it buys the property the ladder cares about:
the corpus a GPU certificate was earned on is pinned by the same commit sha as
the code, so `assert_ref_is_current` covers the DATA as well as the science.
The alternative — regenerating on the VM — would need the 1.0 GB CMU zip
downloaded and 2747 AMC files parsed inside the billed GPU window.

PROVENANCE, so the blob is checkable and not magic: this script reads only
`/data/jack-data/retarget-cache/cmu_*.npz`, which `MoCapLoader._load_cmu_sequences`
wrote by retargeting real CMU ASF/AMC through `SkeletonRetargeter`. Nothing here
synthesises motion. `mocap_cmu.py`'s docstring records why that matters: the
loader this replaced fabricated sinusoids and T1.13 measured the result
(real_f_ratio 0.0). The spec re-checks this blob's sha256 against a frozen
constant before it will spend a GPU hour on it.

Run:  /data/venvs/jackthelearner/bin/python scripts/build_t214_corpus.py
"""
import hashlib
import os
import sys
from pathlib import Path

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

CACHE = Path("/data/jack-data/retarget-cache")
OUT = Path(REPO) / "experiments/data/t214_cmu_corpus.npz"

MIN_FRAMES = 4          # need t-1, t and something to hold out
N_JOINTS = 17           # Humanoid-v5 actuator count (config contract)


def build() -> dict:
    if not CACHE.is_dir():
        raise FileNotFoundError(
            f"{CACHE} missing — this script only runs on the box that holds the "
            "retargeted CMU cache. It REFUSES to synthesise motion (T1.13)."
        )
    P, V, G, names = [], [], [], []
    skipped = 0
    for f in sorted(CACHE.glob("cmu_*.npz")):
        d = np.load(f)
        if not {"positions", "velocities"} <= set(d.files):
            skipped += 1
            continue
        p, v = d["positions"], d["velocities"]
        if p.ndim != 2 or p.shape[1] != N_JOINTS or len(p) < MIN_FRAMES:
            skipped += 1
            continue
        if p.shape != v.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            skipped += 1
            continue
        P.append(p.astype(np.float32))
        V.append(v.astype(np.float32))
        G.append(np.full(len(p), len(names), dtype=np.int16))
        names.append(f.stem)

    if not names:
        raise RuntimeError("no usable clips — refusing to write an empty corpus")

    pos = np.concatenate(P)
    vel = np.concatenate(V)
    clip = np.concatenate(G)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic bytes: fixed key order, no timestamps in npz payloads.
    np.savez_compressed(OUT, positions=pos, velocities=vel, clip=clip,
                        names=np.array(names))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    return {"clips": len(names), "frames": int(len(pos)), "skipped": skipped,
            "bytes": OUT.stat().st_size, "sha256": sha}


if __name__ == "__main__":
    info = build()
    for k, v in info.items():
        print(f"{k:10s} {v}")
    print(f"\nwrote {OUT}")
    print("Paste sha256 into t2_14_imitation_mocap.py:CORPUS_SHA256")
