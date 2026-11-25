"""J10-2 reader: applies the FROZEN `humanoid_sac_row.verdict_j10_2` to the segment files on disk.

No rule lives here. It only gathers, per seed 3-7, the 1M eval (segment 1's `eval`) and the 2M eval
(segment 3's `eval`) from `/data/jack-logs/j10/sac_row2/s<seed>_seg<n>.json`, and hands them to the
frozen verdict. Control: RULE_J10_2 says "J10-1's reading stands" — the late-start control R 68.0
recorded in KNOWN_ANSWERS row J10-1 (same recipe and eval code); it is passed in, not re-run.

    python -m experiments.j10_2_read            # prints per-seed table + verdict (PROVISIONAL if a seed is missing)
    python -m experiments.j10_2_read --check    # exit 0 only when all five 2M files are home
"""
from __future__ import annotations

import json
import os
import sys

from experiments.humanoid_sac_row import J10_2_SEEDS, verdict_j10_2

ROOT = "/data/jack-logs/j10/sac_row2"
J10_1_CONTROL_R = 68.0   # KNOWN_ANSWERS row J10-1, "control (late start, 50k) R 68.0" — the rule's control


def _eval(seed: int, seg: int, root: str = ROOT):
    p = os.path.join(root, f"s{seed}_seg{seg}.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    e = d.get("eval")
    if not isinstance(e, dict) or not d.get("hit_target"):
        return None
    return e


def gather(root: str = ROOT):
    r1m, r2m, m2m, rows = {}, {}, {}, []
    for s in J10_2_SEEDS:
        e1, e2 = _eval(s, 1, root), _eval(s, 3, root)
        if e1:
            r1m[s] = float(e1["R"])
        if e2:
            r2m[s], m2m[s] = float(e2["R"]), float(e2["M"])
        rows.append((s, e1, e2))
    return r1m, r2m, m2m, rows


def main(argv):
    r1m, r2m, m2m, rows = gather()
    print("seed | 1M R | 1M M | 2M R | 2M M")
    for s, e1, e2 in rows:
        f = lambda e, k: f"{e[k]:,.2f}" if e else "—"
        print(f"{s} | {f(e1,'R')} | {f(e1,'M')} | {f(e2,'R')} | {f(e2,'M')}")
    missing = [s for s in J10_2_SEEDS if s not in r2m]
    v = verdict_j10_2(r2m, m2m, r1m, J10_1_CONTROL_R)
    tag = "FINAL" if not missing else f"PROVISIONAL (2M missing for seeds {missing})"
    print(tag, json.dumps(v))
    if "--check" in argv:
        return 0 if not missing else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
