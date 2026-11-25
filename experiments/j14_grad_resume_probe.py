"""J14 (a) probe, box CPU: does brain A's sheeprl resume stay bit-continuous ACROSS gradient steps?

bbd54926's resume-test covered the replay only (learning_starts 1,024 > both runs: no gradient
step). Here learning_starts = LS < SAVE, so gradient steps run before AND after the save point.

  U  : unbroken run to TOTAL, checkpoint every SAVE (so ckpt_SAVE and ckpt_TOTAL exist)
  U2 : the same run again — the DETERMINISM control: if U2's ckpt_TOTAL != U's, the runner is not
       deterministic on this box and no resume can be judged bit-continuous (VOID, not PASS)
  R  : resume from U's ckpt_SAVE to TOTAL
PASS = R's ckpt_TOTAL == U's ckpt_TOTAL on every tensor (model + optimiser state), U2 == U, and
the SAVE->TOTAL leg really took a gradient step (U's world-model weights moved between ckpts —
else the probe is the old buffer-only test again: VOID).

Usage (sheeprl's venv): /data/venvs/jack-a-probe/bin/python -m experiments.j14_grad_resume_probe OUT_DIR
"""
import json
import os
import shutil
import sys

SAVE, TOTAL, LS = 400, 480, 256


def _flat(obj, prefix=""):
    import numpy as np
    import torch
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flat(v, f"{prefix}{i}."))
    elif isinstance(obj, torch.Tensor):
        out[prefix[:-1]] = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        out[prefix[:-1]] = obj
    elif isinstance(obj, (int, float)):
        out[prefix[:-1]] = np.array(obj)
    return out


def load(path):
    import torch
    return _flat(torch.load(path, map_location="cpu", weights_only=False))


def diff(a, b, only=None):
    """(n keys compared, n differing, max abs diff, first 5 differing keys)."""
    import numpy as np
    keys = sorted(k for k in set(a) & set(b) if only is None or k.startswith(only))
    bad, mx = [], 0.0
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            bad.append(k)
            continue
        if x.dtype.kind in "fc":
            d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        else:
            d = float(np.any(x != y))
        if d > 0:
            bad.append(k)
            mx = max(mx, d)
    return {"compared": len(keys), "differ": len(bad), "max_abs": mx, "first": bad[:5],
            "only_in_one": sorted(set(a) ^ set(b))[:5]}


def _run(out, name, total, resume_from=None):
    from experiments.brain_a import in_world as iw
    real = iw.overrides

    def ovr(*a, **k):
        o = [x for x in real(*a, **k) if not x.startswith("algo.learning_starts=")]
        return o + [f"algo.learning_starts={LS}"]
    iw.overrides = ovr
    try:
        r = iw.run(out, total=total, name=name, device="cpu", compile_on=False, deadline_s=1800,
                   ckpt_every=SAVE, buffer_size=4 * TOTAL, resume_from=resume_from)
    finally:
        iw.overrides = real
    return r


def _ckpt(logdir, step):
    import glob
    hits = glob.glob(os.path.join(logdir, "**", "checkpoint", f"ckpt_{step}_0.ckpt"), recursive=True)
    return hits[0] if hits else None


def main(out):
    shutil.rmtree(out, ignore_errors=True)
    rec = {"save": SAVE, "total": TOTAL, "learning_starts": LS}
    u = _run(out, "gu", TOTAL)
    u2 = _run(out, "gu2", TOTAL)
    u_s, u_t, u2_t = _ckpt(u["logdir"], SAVE), _ckpt(u["logdir"], TOTAL), _ckpt(u2["logdir"], TOTAL)
    rec["U"] = {"rc": u["rc"], "wall_s": u["wall_s"], "ckpt_save": u_s, "ckpt_total": u_t,
                "wm_loss_steps": sorted({s for s, _ in u["losses"].get("Loss/world_model_loss", [])})}
    rec["U2"] = {"rc": u2["rc"], "wall_s": u2["wall_s"], "ckpt_total": u2_t}
    if not (u_s and u_t and u2_t):
        rec["verdict"] = "VOID: a checkpoint is missing"
        rec["tails"] = [u.get("log_tail"), u2.get("log_tail")]
        return rec
    # A resume logs into the SAVED run's log_root (bbd54926's finding) and may overwrite its
    # ckpt_TOTAL: read U and U2 into memory FIRST, then resume from U2's save (== U's if U2 == U).
    A, A2, S = load(u_t), load(u2_t), load(u_s)
    rec["moved_save_to_total"] = diff(S, A, "world_model")
    rec["determinism_U_vs_U2"] = diff(A, A2)
    u2_s = _ckpt(u2["logdir"], SAVE)
    rec["determinism_save_U_vs_U2"] = diff(S, load(u2_s)) if u2_s else None
    import glob
    import time
    t0 = time.time()
    r = _run(out, "gr", TOTAL, resume_from=u2_s)
    cands = [c for d in (r["logdir"], u2["logdir"]) for c in
             glob.glob(os.path.join(d, "**", "checkpoint", f"ckpt_{TOTAL}_0.ckpt"), recursive=True)
             if os.path.getmtime(c) >= t0]
    r_t = cands[0] if cands else None
    rec["R"] = {"rc": r["rc"], "wall_s": r["wall_s"], "resumed_from": u2_s, "ckpt_total": r_t,
                "log_tail": r.get("log_tail")}
    if not r_t:
        rec["verdict"] = "VOID: the resumed run left no ckpt_TOTAL"
        return rec
    B = load(r_t)
    rec["resume_R_vs_U"] = diff(A, B)
    rec["resume_R_vs_U_model_only"] = {p: diff(A, B, p) for p in
                                       ("world_model", "actor", "critic", "target_critic")}
    if rec["moved_save_to_total"]["differ"] == 0:
        rec["verdict"] = "VOID: no gradient step between save and total"
    elif rec["determinism_U_vs_U2"]["differ"]:
        rec["verdict"] = "VOID: the unbroken runner is not deterministic on this box"
    else:
        rec["verdict"] = "PASS" if rec["resume_R_vs_U"]["differ"] == 0 else "FAIL"
    return rec


if __name__ == "__main__":
    out = sys.argv[1]
    res = main(out)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "grad_resume_probe.json"), "w"), indent=1, default=str)
    print("PROBE", res.get("verdict"), flush=True)
