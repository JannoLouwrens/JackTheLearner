"""J11 11-18: the body-motor floor twins (`-bm`, groups m0..m7).

If the pilot's rule names motor_frame="body" (INBOX 12:26 / 07:40; gate 2:
body eye + world motor 0/20, body motor 20/20), held8's f-card floors ran the
WORLD motor and are not that creature's floors. `-bm` = the same floor with
only motor_frame changed.

Checks (real code paths):
  1. identity: every pre-existing group (c0..c7, f0..f7) is cell-for-cell what
     HEAD~ (the source before this change, loaded from git) produced.
  2. m0..m7: 8 groups x 4 cells; each group = 2 worlds x both bm arms; all 16
     CARD_SEEDS x 2 arms covered exactly once.
  3. end to end: one life of held8-k0-ff-bm (seed 901, outside every
     registered range) records motor_frame "body", read off the env.
     Control: held8-k0-ff on the same seed records "world".
  4. must-fail: "noop-k1-ff-bm" and "held8-k0-bm" are refused (a zero push
     needs no twin; bm only rides the repaired world).
"""
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import pilot_b2 as P  # noqa: E402

BASE = "858ba11c"   # last commit before the -bm arms


def main():
    ok = True
    src = subprocess.run(["git", "show", f"{BASE}:experiments/pilot_b2.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    old = types.ModuleType("pilot_b2_old")
    old.__file__ = str(ROOT / "experiments" / "pilot_b2.py")
    exec(compile(src, "pilot_b2_old", "exec"), old.__dict__)
    og, ng = old.card_groups(), P.card_groups()
    g1 = all(ng[k] == v for k, v in og.items()) and len(og) == 16
    print(f"1 identity: {len(og)} old groups unchanged: {g1}")
    ok &= g1

    m = [ng[f"m{i}"] for i in range(8)]
    cells = [c for g in m for c in g]
    g2 = (len(ng) == 24 and all(len(g) == 4 for g in m)
          and all(sorted({a for a, _ in g}) == sorted(P.CARD_FF_BM_ARMS) and len({s for _, s in g}) == 2 for g in m)
          and sorted(cells) == sorted((a, s) for s in P.CARD_SEEDS for a in P.CARD_FF_BM_ARMS))
    print(f"2 m-groups: 8 x 4, 2 worlds x both arms each, 32 cells once: {g2}")
    ok &= g2

    rb = P.run_card_cell("held8-k0-ff-bm", 901, lives=1)
    rw = P.run_card_cell("held8-k0-ff", 901, lives=1)
    g3 = rb["motor_frame"] == "body" and rw["motor_frame"] == "world" and len(rb["lives"]) == 1
    print(f"3 stamp off env: bm={rb['motor_frame']} ({rb['decisions']} dec), "
          f"control={rw['motor_frame']} ({rw['decisions']} dec): {g3}")
    ok &= g3

    refused = 0
    for arm in ("noop-k1-ff-bm", "held8-k0-bm"):
        try:
            P.run_card_cell(arm, 901, lives=1)
        except ValueError:
            refused += 1
    g4 = refused == 2
    print(f"4 must-fail arms refused {refused}/2: {g4}")
    ok &= g4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
