"""J22 (2) timing cell, BOX half: how fast does arm h4 (conv eye + shift) learn?

Read written before it runs: the sighted cell `H-conv-row128`, s923,
PILOT_RULE's world (pilot_b2.pilot_world, food floor only, as the hunt), run
three times from scratch to T = 5,000 (warmup only: no gradient step), 5,300
and 5,600. Learning decisions/s = 300 / (wall(5,600) - wall(5,300)) -- the
first learning stretch carries the JIT compile, so it is excluded. Then
100k at that rate (95k learning + the warmup's own rate) is the box's price
for ONE cell; the pilot is three cells. Venue rule (ORDERS J22 (2)): the venue
that fits 100k a cell in one session -- box (detached, 2 workers), Kaggle CPU
(12 h cap, 2 cells a kernel), or Colab T4 (the GPU half of this cell).
No row, no ledger: a timing.
Run: nice -n 19 /data/venvs/jack-std/bin/python -m experiments.j22_timing
"""
import json
import sys
import time
from pathlib import Path

OUT = Path("/data/jack-logs/j22/j22_timing_box.json")


def main(arm="H-conv-row128", seed=923):
    from experiments.pilot_b2 import run_pilot_cell
    out = {"arm": arm, "seed": seed, "machine": "box (opc, jack-std, nice 19, 1 thread)", "runs": {}}
    for T in (5_000, 5_300, 5_600):
        t0 = time.time()
        rec = run_pilot_cell(arm, seed, True, T=T)
        out["runs"][T] = {"wall_s": round(time.time() - t0, 2), "cpu_s": rec["cpu_s"],
                          "decisions": rec["decisions"]}
        print(T, out["runs"][T], flush=True)
    r = out["runs"]
    warm = 5_000 / r[5_000]["wall_s"]
    learn = 300 / (r[5_600]["wall_s"] - r[5_300]["wall_s"])
    out["warmup_decisions_per_s"] = round(warm, 2)
    out["learning_decisions_per_s"] = round(learn, 2)
    out["hours_100k_one_cell"] = round((5_000 / warm + 95_000 / learn) / 3600, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
