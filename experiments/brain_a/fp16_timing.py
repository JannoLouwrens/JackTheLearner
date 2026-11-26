"""J28 cell 2: brain A size S, mixed precision ON vs OFF — seconds per 1k decisions, both.

A TIMING CELL, NOT A LEARNING RESULT. No score is read, no seat moves, one seed (10, the A2-v1S-enc
prereg's timing seed). The arm is A2-v1S-enc (`rows.enc_base()`, prereg a2_size_s_enc.md, frozen) —
the brain A size S J14 runs on W48 — timed twice in ONE kernel, same GPU, same image:

  OFF: `fabric.precision=32-true`  (what A1-A3 and A2-v1S-enc run; pinned in rows.BASE_V1S)
  ON : `fabric.precision=16-mixed` (the official DreamerV3's float16; a T4 has no native bf16,
       so the v1 exp file's bf16-mixed is not the published choice on this GPU)

Exactly one key moves between the two cells. Both use rows.timing_cell (TIMING_POLICY_STEPS policy
steps, the runner's own Time/* scalars, first window dropped). A cell that crashes is a reading
(rc + log tail), not retried. Seconds per 1k decisions = 1000 / sps_env + 1000 * replay_ratio / sps_train
(the same rates rows.project_seed_hours uses for the seed bill).
"""
import json as _json
import os as _os
import sys as _sys

from experiments.brain_a import rows as R

TAG = "A2v1Senc"
SEED = 10
FETCH = "brain_a_fp16_timing.json"
EST_HOURS = 0.5          # two ~10-min cells + install; the order says 20 minutes
PRECISIONS = {"off": "32-true", "on": "16-mixed"}


def arm(precision: str) -> list:
    base = R.enc_base()
    out = [("fabric.precision=" + precision) if o.startswith("fabric.precision=") else o for o in base]
    assert sum(o.startswith("fabric.precision=") for o in out) == 1
    return out


def sec_per_1k(projection, replay_ratio: float = 0.5):
    """None when the cell gave no readable rates (an unreadable meter is not a zero)."""
    if not projection:
        return None
    return round(1000 / projection["sps_env"] + 1000 * replay_ratio / projection["sps_train"], 2)


def summarise(cells: dict) -> dict:
    out = {"cell": "J28-2 fp16 timing", "timing_only": True, "seed": SEED, "arms": {}}
    for k, res in cells.items():
        p = (res or {}).get("projection")
        out["arms"][k] = {"precision": PRECISIONS[k], "rc": (res or {}).get("rc"),
                          "sec_per_1k_decisions": sec_per_1k(p), "projection": p}
    a, b = out["arms"].get("off", {}), out["arms"].get("on", {})
    if a.get("sec_per_1k_decisions") and b.get("sec_per_1k_decisions"):
        out["speedup_on_vs_off"] = round(a["sec_per_1k_decisions"] / b["sec_per_1k_decisions"], 3)
    return out


_TAIL = r"""
import os
""" + R.ENC_COMPILE_ENV_LINE + r"""
from experiments.brain_a.rows import timing_cell
from experiments.brain_a.fp16_timing import arm, summarise, PRECISIONS, TAG, SEED, FETCH
_cells = {}
for _k in ("off", "on"):
    try:
        _cells[_k] = timing_cell(WORK, "cuda", base=arm(PRECISIONS[_k]), tag=TAG + "_" + _k,
                                 fname="fp16_cell_" + _k + ".json", seed=SEED)
    except Exception as _e:
        _cells[_k] = {"rc": None, "error": repr(_e)[:400]}
_s = summarise(_cells)
_s["cells"] = _cells
json.dump(_s, open(os.path.join(WORK, FETCH), "w"))
print("RESULT " + json.dumps({k: _s[k] for k in _s if k != "cells"}), flush=True)
print("DONE", flush=True)
"""


def kernel_body() -> str:
    """Kaggle's torch pair pinned and compile in-process, exactly as every A2-v1S-enc kernel."""
    return R.ENC_TORCH_PIN_LINES + R._kernel_head() + _TAIL


def main(argv) -> int:
    """kernel   (Kaggle GPU: both cells in one session; writes /data/jack-logs/brain_a/rows/fp16_timing.json)"""
    if (argv[1] if len(argv) > 1 else "") != "kernel":
        print(main.__doc__)
        return 2
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(kernel_body()), prefer="kaggle", est_hours=EST_HOURS,
                 timeout_s=int(2 * 3600), fetch=[FETCH])
    print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
    if not sub.ok:
        return 1
    _os.makedirs(R.OUT, exist_ok=True)
    dst = _os.path.join(R.OUT, FETCH.replace("brain_a_", ""))
    open(dst, "w").write(_json.dumps(result_json(sub, FETCH)))
    print("WROTE", dst, flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
