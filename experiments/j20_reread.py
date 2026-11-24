"""J20 (1): gate 2 (the hand forager on the body-frame row128 eye) re-read on
mujoco 3.17.0, on the box, beside the 3.2.3-era Kaggle rows in WORLD_CARD.md.

Nothing in the ruler changes: same rows, seeds 261-280, 10 lives each, same
`eye_forager.run_seed` life loop, same `score_gate2` bar. Only the machine
(box, jack-std venv, mujoco 3.17.0) and the runner differ: this one is
resumable — a cell whose file already holds all its lives is skipped, a
partial cell file (a killed run) is removed and the cell re-run from its seed.
h4 (reported only in gate 2) is not re-read unless asked for by name.

    python -m experiments.j20_reread run <d> <row> [row ...]   # box, detached
    python -m experiments.j20_reread score <d>                 # collect + bar
    python -m experiments.j20_reread floor                     # box, detached
    python -m experiments.j20_reread floor-score               # the three rows

J20 (2), the floor card: WORLD_CARD.md's three learner-free rows a
(do-nothing), b (white random), d8 (held-8 random) re-run with the SAME code
and seeds as the registered 3.2.3-era rows (`world_card cell`, seeds 201-220
x 10 lives = 200 lives a row, d = 1 default world, cap 9,000 decisions) on
mujoco 3.17.0. Same resumable rule: a complete cell file is skipped, a
partial one removed and re-run from its seed.
"""
import json
import sys
from pathlib import Path

from experiments import eye_forager as EF
from experiments import world_card as WC

OUT = Path("/data/jack-logs/j20")
WORKERS = 2          # the box: 4 shared cores, customers first


def _cell_file(row: str, seed: int, d: float) -> Path:
    return OUT / f"{row}_d{int(round(100 * d)):03d}_s{seed}.jsonl"


def _lives(f: Path) -> list:
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def todo(d: float, rows) -> list:
    out = []
    for row in rows:
        if row not in EF.GATE2_ROWS:
            raise ValueError(f"row {row!r} not in gate 2's {EF.GATE2_ROWS}")
        for s in EF.GATE2_SEEDS:
            f = _cell_file(row, s, d)
            if len(_lives(f)) >= WC.LIVES:
                continue
            if f.exists():
                f.unlink()                       # partial: re-run from the seed
            out.append((row, s, d))
    return out


def run(d: float, rows) -> dict:
    import mujoco
    cells = todo(d, rows)
    print(f"mujoco {mujoco.__version__}; {len(cells)} cells to run", flush=True)
    return EF.run_pool(cells, OUT, workers=WORKERS)


def score(d: float) -> dict:
    rows = {}
    for row in EF.GATE2_ROWS:
        for s in EF.GATE2_SEEDS:
            rows.setdefault(row, []).extend(_lives(_cell_file(row, s, d)))
    have = {r: len(v) for r, v in rows.items()}
    b = rows["b4"]
    out = {"d": d, "lives": have}
    for r in ("f4", "g4", "h4"):
        if have[r] == len(EF.GATE2_SEEDS) * WC.LIVES and have["b4"] == have[r]:
            out[r] = WC.score(rows[r], b, EF.GATE2_SEEDS)
            out[r]["seed_means"] = [round(WC.seed_mean(rows[r], s), 2) for s in EF.GATE2_SEEDS]
    return out


FLOOR_ROWS = ("a", "d8", "b")              # longest lives first


def _floor_file(row: str, seed: int) -> Path:
    return OUT / "floor" / f"{row}_s{seed}.jsonl"


def floor_todo() -> list:
    out = []
    for row in FLOOR_ROWS:
        for s in WC.REGISTERED_SEEDS:
            f = _floor_file(row, s)
            if len(_lives(f)) >= WC.LIVES:
                continue
            if f.exists():
                f.unlink()
            out.append((row, s))
    return out


def floor_run(workers: int = WORKERS) -> int:
    import subprocess
    import time
    import mujoco
    cells = floor_todo()
    (OUT / "floor").mkdir(parents=True, exist_ok=True)
    print(f"mujoco {mujoco.__version__}; {len(cells)} floor cells to run", flush=True)
    live, bad = [], 0
    while cells or live:
        while cells and len(live) < workers:
            row, s = cells.pop(0)
            f = _floor_file(row, s)
            err = open(str(f) + ".err", "w")
            live.append((row, s, f, err, subprocess.Popen(
                [sys.executable, "-m", "experiments.world_card", "cell", row,
                 str(s), str(f)], cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.DEVNULL, stderr=err)))
        time.sleep(2)
        for item in list(live):
            row, s, f, err, p = item
            if p.poll() is None:
                continue
            live.remove(item)
            err.close()
            bad += p.returncode != 0
            print(f"{row} s{s}: rc {p.returncode}, {len(_lives(f))} lives", flush=True)
    return bad


def floor_score() -> dict:
    import numpy as np
    out = {}
    for row in FLOOR_ROWS:
        L = [x for s in WC.REGISTERED_SEEDS for x in _lives(_floor_file(row, s))]
        if not L:
            continue
        m = np.array([x["floor_meals"] for x in L], float)
        t = np.array([x["lifetime_s"] for x in L], float)
        causes = {}
        for x in L:
            causes[x["cause"]] = causes.get(x["cause"], 0) + 1
        sm = [WC.seed_mean(L, s) for s in WC.REGISTERED_SEEDS]
        out[row] = {"lives": len(L),
                    "lifetime_s": [round(t.mean(), 1), round(t.std(ddof=1), 1) if len(t) > 1 else 0,
                                   float(t.min()), float(t.max())],
                    "floor_meals": [round(m.mean(), 2), round(m.std(ddof=1), 2) if len(m) > 1 else 0,
                                    int(m.max())],
                    "lives_ge1": int((m >= 1).sum()), "lives_ge5": int((m >= 5).sum()),
                    "seed_means": [round(v, 2) for v in sm],
                    "apples": int(sum(x["apples"] for x in L)), "causes": causes}
    return out


def main(argv):
    if len(argv) >= 2 and argv[1] == "floor":
        return 1 if floor_run() else 0
    if len(argv) >= 2 and argv[1] == "floor-score":
        print(json.dumps(floor_score(), indent=1))
        return 0
    if len(argv) >= 4 and argv[1] == "run":
        r = run(float(argv[2]), argv[3:])
        print(json.dumps({k: len(v["lives"]) for k, v in r.items()}), flush=True)
        return 0
    if len(argv) >= 3 and argv[1] == "score":
        print(json.dumps(score(float(argv[2])), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
