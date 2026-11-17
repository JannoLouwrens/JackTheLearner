"""pilot_b.kernel without a summary file still returns every cell it can.

ORDERS J11 prep (11-17 20:50). f76f1edb made a killed CELL keep its lives via
`.partial.json`; but `pilot_b.kernel` only read `pilot_b_cells.json`, which the
job writes after ALL cells end. A kernel cut by Kaggle's cap (or reattached
with JACK_REUSE_KERNEL after its watcher timed out) has no summary, and kernel()
raised — the partial files came back in the output and were thrown away.

Not a pilot run. Real files: one held8 floor run (seed 381, J8 world, 3,000
decisions, PARTIAL_EVERY = 1) written as its final file and as its partial,
laid out under Kaggle's filename keys.

  final     cell with a final file -> that file's lives, row for row.
  partial   cell with only a partial -> `cut`, lives == partial file's lives.
  missing   cell with neither -> `not_found`, 0 lives (never invented).
  none      no cell file at all -> None (kernel() raises, as before).
  control   the old path: summary absent -> no cells (what kernel() did).
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    import experiments.pilot_b as pb
    pb.PARTIAL_EVERY = 1
    tmp = Path(tempfile.mkdtemp(dir="/data/jack-logs"))
    final = tmp / "pilot_held8_s381.json"
    run = pb.run_floor("held8", 381, 3000, partial_path=pb.partial_file(final))
    final.write_text(json.dumps(run) + "\n")
    part = pb.partial_file(final)
    # a second "cell" that only got as far as its partial (copy, renamed)
    part2 = tmp / "pilot_noop_s382.partial.json"
    part2.write_text(part.read_text())
    plives = json.loads(part.read_text())["lives"]
    cells = [("held8", 381), ("noop", 382), ("noop", 383)]
    art = {final.name: str(final), part2.name: str(part2)}
    r = pb.cells_from_artifacts(art, cells)["cells"]
    out = {"lives_run": len(run["lives"]), "lives_partial": len(plives)}
    out["final"] = r["held8_s381"]["lives"] == run["lives"] and not r["held8_s381"].get("cut")
    out["partial"] = (r["noop_s382"].get("cut") is True
                      and r["noop_s382"]["lives"] == plives and len(plives) > 0)
    out["missing"] = r["noop_s383"].get("not_found") is True and r["noop_s383"]["lives"] == []
    out["none"] = pb.cells_from_artifacts({}, cells) is None
    out["control_old_path_empty"] = "pilot_b_cells.json" not in art
    ok = all(out[k] for k in ("final", "partial", "missing", "none", "control_old_path_empty"))
    ok &= out["lives_run"] > 0
    out["PASS"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
