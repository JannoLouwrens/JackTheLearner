"""J14 (a) follow-up, box CPU, no training: WHERE does the resumed run R first leave the unbroken run U?

The probe (`j14_grad_resume_probe.py`, receipt `results/j14_grad_resume_probe.json`) read FAIL:
same counters, different weights, and guessed "RNG not checkpointed". This reads the two saved
replay buffers at TOTAL (U's and R's, both written by sheeprl's own checkpoint callback) and finds,
per env and per key, the first row where they differ. The buffer is the input to every gradient
step after the save: if it differs, restoring the RNG alone cannot make R == U.

sheeprl 0.8.5's design, read in its source (dreamer_v3.py:739 and utils/callback.py:61-72):
the checkpoint does NOT save the env state, so (i) the saved buffer's last row is marked
truncated, and (ii) a resume calls `envs.reset(seed=cfg.seed ...)` — a fresh episode. Prediction
of that design: R == U on rows < SAVE_ROW except `truncated` at row SAVE_ROW-1; obs differ from
row SAVE_ROW on (a reset, not the continuation).

Known-positive (planted): first_diff must find a single flipped element at a known row.

Usage (sheeprl's venv): /data/venvs/jack-a-probe/bin/python -m experiments.j14_resume_divergence RECEIPT_JSON
"""
import glob
import json
import os
import sys


def first_diff(a, b):
    """First row index where arrays a and b differ (rows = axis 0), over the common length; None if equal."""
    import numpy as np
    n = min(len(a), len(b))
    if n == 0:
        return None
    x, y = np.asarray(a[:n]), np.asarray(b[:n])
    rows = np.any((x != y).reshape(n, -1), axis=1)
    hit = np.flatnonzero(rows)
    return int(hit[0]) if hit.size else None


def _planted():
    import numpy as np
    a = np.zeros((10, 3))
    b = a.copy()
    b[7, 2] = 1e-9
    assert first_diff(a, b) == 7
    assert first_diff(a, a.copy()) is None
    return True


def _buf_of(ckpt):
    d = os.path.dirname(os.path.dirname(ckpt))
    step = os.path.basename(ckpt).split("_")[1]
    hits = glob.glob(os.path.join(d, "checkpoint_buffers", f"ckpt_{step}_buffer_rank_0.pt"))
    return hits[0] if hits else None


def main(receipt):
    import numpy as np
    from experiments.brain_a.in_world import buffer_rows
    rec = json.load(open(receipt))
    paths = {"U_total": _buf_of(rec["U"]["ckpt_total"]), "R_total": _buf_of(rec["R"]["ckpt_total"]),
             "U_save": _buf_of(rec["U"]["ckpt_save"])}
    out = {"planted": _planted(), "paths": paths}
    U, R, S = (buffer_rows(paths[k]) for k in ("U_total", "R_total", "U_save"))
    envs = []
    for e, ((nu, bu), (nr, br), (ns, bs)) in enumerate(zip(U, R, S)):
        keys = sorted(set(bu) & set(br))
        per = {k: first_diff(bu[k], br[k]) for k in keys}
        row = {"rows_U": nu, "rows_R": nr, "rows_saved": ns, "first_diff_by_key": per}
        # what the saved buffer itself says at its last row, vs what U's live buffer kept there
        if "truncated" in bs:
            row["saved_last_truncated"] = float(np.asarray(bs["truncated"][ns - 1]).ravel()[0])
            row["U_truncated_at_that_row"] = float(np.asarray(bu["truncated"][ns - 1]).ravel()[0])
            row["R_truncated_at_that_row"] = float(np.asarray(br["truncated"][ns - 1]).ravel()[0])
        if "is_first" in br:
            row["R_is_first_at_resume_row"] = float(np.asarray(br["is_first"][ns]).ravel()[0])
            row["U_is_first_at_resume_row"] = float(np.asarray(bu["is_first"][ns]).ravel()[0])
        envs.append(row)
    out["per_env"] = envs
    firsts = [v for e in envs for v in e["first_diff_by_key"].values() if v is not None]
    out["earliest_diff_row"] = min(firsts) if firsts else None
    out["saved_rows"] = [e["rows_saved"] for e in envs]
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    print(json.dumps(res, indent=1, default=str))
