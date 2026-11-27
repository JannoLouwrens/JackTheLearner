"""B-7 reader — the frozen rule of KNOWN_ANSWERS row B-7 (c848d888), applied ONCE.

Built 11-27 before any B7mem seed came home (J29 (5) launched 14:26), so the
read is code sealed before the evidence, not arithmetic done after it.

Rule, verbatim from the frozen row (nothing here is a choice):
  * seed score = mean eval return over eval points with step in [40k, 50k]
    (11 points) — `brain_b_mem.b7_seed_score`, the frozen function;
  * arm score  = mean of the 4 seed scores, seeds 0, 1, 2, 3 exactly;
  * VOID ("not a memory test as built") if the memoryless twin B7flat's arm
    score is >= -190.9;
  * VOID if any B7mem seed is lost / incomplete (never replaced);
  * REPRODUCES iff the B7mem arm score lies in [-190.9, -154.5]; else MISS.

Refuses (never reads) unless every B7mem raw file carries one pin and row B-7
is FROZEN at that pin (`b7mem_launch.row_at` / `frozen`, the launcher's own
guard), and unless all eight files are present.

    python -m experiments.b7_read [--mem DIR] [--flat DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.b7mem_launch import OUT as MEM_DIR, SEEDS, frozen, row_at
from experiments.brain_b_mem import b7_seed_score

FLAT_DIR = Path("/data/jack-logs/j29")
BAND = (-190.9, -154.5)
VOID_AT = -190.9


class Refused(Exception):
    pass


def _load(path: Path) -> dict:
    if not path.exists():
        raise Refused(f"b7_read: {path} is not home — the rule reads all four seeds once")
    return json.loads(path.read_text())


def read(mem_dir: Path = MEM_DIR, flat_dir: Path = FLAT_DIR, row_fn=row_at) -> dict:
    mem = {s: _load(Path(mem_dir) / f"b7mem_s{s}.json") for s in SEEDS}
    flat = {s: _load(Path(flat_dir) / f"b7flat_s{s}.json") for s in SEEDS}
    pins = {r.get("pin") for r in mem.values()}
    if len(pins) != 1 or None in pins:
        raise Refused(f"b7_read: B7mem raw files carry pins {sorted(map(str, pins))}, need one")
    pin = pins.pop()
    if not frozen(row_fn(pin)):
        raise Refused(f"b7_read: row B-7 is not FROZEN at pin {pin}")
    for s, r in flat.items():
        if r.get("arm") != "B7flat":
            raise Refused(f"b7_read: b7flat_s{s} is arm {r.get('arm')!r}, not the twin B7flat")

    out = {"rule": "KNOWN_ANSWERS B-7 FROZEN c848d888", "pin": pin, "band": list(BAND)}
    lost = [s for s, r in mem.items() if r.get("lost") or not r.get("complete")]
    flat_lost = [s for s, r in flat.items() if not r.get("complete")]
    out["twin_seed_scores"] = {s: (None if s in flat_lost else b7_seed_score(r["evals"]))
                               for s, r in flat.items()}
    out["mem_seed_scores"] = {s: (None if s in lost else b7_seed_score(r["evals"]))
                              for s, r in mem.items()}
    if lost or flat_lost:
        out["verdict"] = "VOID"
        out["why"] = f"seed(s) lost/incomplete: B7mem {lost}, B7flat {flat_lost} — never replaced"
        return out
    twin = sum(out["twin_seed_scores"].values()) / len(SEEDS)
    arm = sum(out["mem_seed_scores"].values()) / len(SEEDS)
    out["twin_score"], out["arm_score"] = twin, arm
    if twin >= VOID_AT:
        out["verdict"], out["why"] = "VOID", f"twin {twin:.1f} >= {VOID_AT}: not a memory test as built"
    elif BAND[0] <= arm <= BAND[1]:
        out["verdict"], out["why"] = "REPRODUCES", f"arm {arm:.1f} in band, twin {twin:.1f} fails"
    else:
        out["verdict"], out["why"] = "MISS", f"arm {arm:.1f} outside {BAND}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", default=str(MEM_DIR)); ap.add_argument("--flat", default=str(FLAT_DIR))
    ap.add_argument("--out", default=str(MEM_DIR / "b7_read.json"))
    a = ap.parse_args(argv)
    try:
        r = read(Path(a.mem), Path(a.flat))
    except Refused as e:
        print(f"REFUSED {e}")
        return 2
    print(json.dumps(r, indent=1))
    Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
