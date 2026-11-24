"""Ruling 91 (JACK Next-up #1): the resumed-against-unbroken twin on the
SURVIVAL stack's own checkpoint path — LF.03's `_capture` / `_load_checkpoint`
/ `_apply`, unchanged, on the mujoco the box runs today. No ledger.

LF.03 already gates "resumed == unbroken for 200 decisions" but it has no
planted broken save, so nothing shows its comparison COULD fail on the items
that failed on brain B's path this week (the world's RNG, the physics).
This twin adds them. Each arm is a FRESH process resuming from the SAME
checkpoint file; a planted arm swaps ONE carried item for what a save that
forgot it would give (the value a freshly built life holds):

  RC          the checkpoint as LF.03 writes it              -> must be identical
  no_physics  mj integration state from a fresh world        -> must differ
  no_warm     qacc_warmstart zeroed                          -> must differ (or be named)
  no_wrng     the world's RNG from a fresh world             -> must differ
  no_lrng     the learner's np RNG from a fresh life         -> must differ
  no_torch    torch global RNG re-seeded                     -> must differ

Read: RC identical on all RESUME_CHECK rows AND every planted arm differs on
>= 1 row -> the survival stack resumes, and the test can see a drop. A
planted arm that does NOT differ is named: that item is not carried by the
row's comparison (or not used in RESUME_CHECK decisions) — one INBOX line.

    PY -m experiments.lf03_resume_twin [seed] [k_save]   # box, nice 19, ~5 min
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/data/venvs/jackthelearner/bin/python"
ARMS = ("RC", "no_physics", "no_warm", "no_wrng", "no_lrng", "no_torch")
OUT = REPO / "docs" / "research" / "world_card_data" / "lf03_resume_twin.json"


def _first(tmp, seed, j0, alpha, k_save):
    """Unbroken life: save mid-life at >= k_save (LF.03's own rule), go on."""
    from experiments.tests import lf_03_learner_lives as T
    L = T._Life(seed, j0, alpha)
    saved = None
    while L.k < T.DECISION_CAP:
        L.step()
        if (L.k >= k_save and not L.w.died_this_decision
                and (L.w.sim_seconds - L.w._life_started_at)
                >= T.SAVE_MIN_INTO_LIFE * (L.w.sim_seconds / L.w.decisions)):
            T._save_atomic(T._capture(L), f"{tmp}/ckpt.pkl")
            saved = {"decision": L.k, "life": int(L.w.life),
                     "into_life_s": float(L.w.sim_seconds - L.w._life_started_at)}
            break
    rows = []
    if saved is not None:
        for _ in range(T.RESUME_CHECK):
            L.step()
            rows.append(L.feature())
    T._dump(f"{tmp}/first.json", {"saved": saved, "rows": rows})


def _plant(ckpt, arm, seed, j0, alpha):
    """Swap ONE carried item for a fresh life's value (what a save lacking it gets)."""
    import io

    import numpy as np
    import torch

    from experiments.tests import lf_03_learner_lives as T
    if arm == "RC":
        return ckpt
    fresh = T._Life(seed, j0, alpha)
    world = pickle.loads(ckpt["stores"]["world"])
    if arm == "no_physics":
        world["mjstate"] = T._mj_state(fresh.w).tobytes()
    elif arm == "no_warm":
        world["warmstart"] = np.zeros_like(world["warmstart"])
    elif arm == "no_wrng":
        world["rng"] = fresh.w._rng.get_state()
    elif arm in ("no_lrng", "no_torch"):
        sn = torch.load(io.BytesIO(ckpt["stores"]["learner"]), weights_only=False)
        if arm == "no_lrng":
            sn["np_rng"] = fresh.rng.get_state()
        else:
            g = torch.Generator()
            g.manual_seed(12345)
            sn["torch_rng"] = g.get_state()
        buf = io.BytesIO()
        torch.save(sn, buf)
        ckpt["stores"]["learner"] = buf.getvalue()
    else:
        raise ValueError(arm)
    ckpt["stores"]["world"] = pickle.dumps(world)
    return ckpt


def _second(tmp, seed, j0, alpha, arm):
    from experiments.tests import lf_03_learner_lives as T
    ckpt = T._load_checkpoint(f"{tmp}/ckpt.pkl")     # integrity-checked, as LF.03
    L = T._apply(_plant(ckpt, arm, seed, j0, alpha), seed, j0, alpha)
    rows = []
    for _ in range(T.RESUME_CHECK):
        L.step()
        rows.append(L.feature())
    T._dump(f"{tmp}/second_{arm}.json", {"rows": rows})


def compare(ref: list, res: list) -> dict:
    """Pure: identical rows, first differing row and the features that differ there."""
    same = [a == b for a, b in zip(ref, res)]
    first = next((i for i, s in enumerate(same) if not s), None)
    keys = [] if first is None else sorted(k for k in ref[first] if ref[first][k] != res[first].get(k))
    return {"rows": len(ref), "identical": sum(same), "first_diff": first, "diff_keys": keys,
            "all_identical": len(ref) > 0 and all(same) and len(ref) == len(res)}


def verdict(cmp: dict) -> str:
    """RC must be identical; every planted arm must differ."""
    if not cmp.get("RC", {}).get("all_identical"):
        return "RESUME DIFFERS (RC): " + json.dumps(cmp.get("RC"))
    blind = [a for a in ARMS[1:] if cmp.get(a, {}).get("all_identical")]
    return "RESUMES; every planted drop seen" if not blind else f"RESUMES; drop NOT seen: {blind}"


def run(seed: int = 0, k_save: int = 1200) -> dict:
    import mujoco

    from experiments.tests import lf_03_learner_lives as T
    j0, alpha, prov = T._calibration()
    if j0 is None:
        raise RuntimeError(f"PS.01 calibration refused: {prov}")
    tmp = tempfile.mkdtemp(prefix=f"lf03twin_s{seed}_", dir="/data/tmp")
    t0 = time.perf_counter()

    def child(*a):
        p = subprocess.run([PY, "-m", "experiments.lf03_resume_twin", *map(str, a)], cwd=REPO,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
        if p.returncode:
            raise RuntimeError(f"{a[0]} rc={p.returncode}: {p.stderr.decode()[-800:]}")

    child("first", tmp, seed, repr(j0), repr(alpha), k_save)
    first = json.load(open(f"{tmp}/first.json"))
    if first["saved"] is None:
        raise RuntimeError("no mid-life save before the cap")
    cmp = {}
    for arm in ARMS:
        child("second", tmp, seed, repr(j0), repr(alpha), arm)
        cmp[arm] = compare(first["rows"], json.load(open(f"{tmp}/second_{arm}.json"))["rows"])
        print(arm, cmp[arm], flush=True)
    out = {"seed": seed, "k_save": k_save, "saved": first["saved"], "mujoco": mujoco.__version__,
           "j0": j0, "alpha": alpha, "arms": cmp, "verdict": verdict(cmp),
           "wall_s": round(time.perf_counter() - t0, 1), "tmp": tmp}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(out["verdict"], flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "first":
        _first(a[1], int(a[2]), float(a[3]), float(a[4]), int(a[5]))
    elif a and a[0] == "second":
        _second(a[1], int(a[2]), float(a[3]), float(a[4]), a[5])
    else:
        run(int(a[0]) if a else 0, int(a[1]) if len(a) > 1 else 1200)
