"""J6(d) DRINK race — cells on Kaggle CPU (no box CPU), per the declaration in
docs/research/NEEDS_CLOCK_RACE.md ("J6(d) step 2 — DRINK race DECLARED",
commit 4c316fa). The rule, bars and seeds live there, not here.

Why a new runner, not water_clock_race.run_cell on the box: an NE.01 cell at
k72 is ~31 CPU-min (life_cap 39158 dec x 5 lives), and ruling 3's detached cap
refuses box pollers today. Kaggle CPU takes 2 sessions in parallel.

STALENESS (measured 11-14 18:50, mtimes vs git): the reusable cells
/data/jack-logs/heat_race/shipped_NE.01_s* (00:27) and
/data/jack-logs/water_race/swp_k72_NE.01_s* (03:50) PREDATE NE.01's
re-registration (e054e0a 10:13, 33a1acc 06:40), so they cannot be the
comparison arms. Every cell written here therefore carries `code`, the sha256
of each file on its path, and the tally must use only cells whose `code`
equals the current tree's (unknown is not a match).

One kernel = up to 3 cells side by side (Kaggle CPU has 4 cores).

Usage:
  python -m experiments.drink_race_kcpu run ARM SPEC SEED [OUTDIR]   # one cell, here
  python -m experiments.drink_race_kcpu kernel ARM:SPEC:SEED[,...]    # one Kaggle CPU kernel
      (JACK_REUSE_KERNEL=<user>/<slug> reattaches at zero quota, as CR.01)
  python -m experiments.drink_race_kcpu status                         # cells current / stale / missing
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/data/jack-logs/drink_race")
ARMS = ("shipped", "swp_k72", "dB")          # dB0 is apparatus-only (probe level)
SPECS = ("NE.01", "W.1", "W.2")
SEEDS = range(5)
CODE_FILES = ("experiments/needs.py", "playground.py",
              "experiments/heat_race.py", "experiments/water_clock_race.py",
              "experiments/drink_race.py", "experiments/drink_race_kcpu.py")
CHILD_TIMEOUT_S = 10 * 3600
REMOTE_TIMEOUT_S = 11 * 3600


def _spec_file(spec: str) -> str:
    from experiments import heat_race as H
    return H.SPECS[spec].replace(".", "/") + ".py"


def code_shas(spec: str) -> dict:
    out = {}
    for f in CODE_FILES + (_spec_file(spec),):
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


def _apply(arm: str):
    if arm == "shipped":
        from experiments import heat_race as H
        return H._apply("shipped")
    if arm == "swp_k72":
        from experiments import water_clock_race as WR
        return WR._apply("swp_k72")
    from experiments import drink_race as D
    return D._apply(arm)


def run_cell(arm: str, spec: str, seed: int, outdir: Path = OUT) -> Path:
    from experiments import heat_race as H
    code = code_shas(spec)
    _, mods = _apply(arm)
    mod = mods[spec]
    m = mod._experiment(int(seed))
    rec = {"arm": arm, "spec": spec, "seed": int(seed), "metrics": m, "code": code}
    if spec == "W.1":
        c = mod._control(int(seed))
        rec["control"] = c
        rec["check"] = bool(mod._check(m, c))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{arm}_{spec}_s{int(seed)}.json"
    p.write_text(json.dumps(rec, default=H._jsonable))
    return p


def _parse(cells: str):
    out = []
    for c in cells.split(","):
        arm, spec, seed = c.split(":")
        if arm not in ARMS + ("dB0",) or spec not in SPECS:
            raise KeyError(c)
        out.append((arm, spec, int(seed)))
    if not 1 <= len(out) <= 3:
        raise ValueError("one kernel runs 1-3 cells")
    return out


def run_parallel(cells, outdir, local=True) -> dict:
    """Cells side by side as child processes; returns {name: record}."""
    procs = []
    for arm, spec, seed in cells:
        cmd = [sys.executable, "-m", "experiments.drink_race_kcpu", "run",
               arm, spec, str(seed), str(outdir)]
        procs.append((f"{arm}_{spec}_s{seed}", subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, p in procs:
        _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        if p.returncode != 0:
            res[name] = {"error": f"rc={p.returncode}: {err.decode()[-800:]}"}
            continue
        res[name] = json.loads((Path(outdir) / f"{name}.json").read_text())
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.drink_race_kcpu import run_parallel, _parse
r = run_parallel(_parse(__CELLS__), os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "drink_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(cells_arg: str) -> dict:
    cells = _parse(cells_arg)
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(_JOB.replace("__CELLS__", repr(cells_arg))),
                 prefer="kaggle-cpu", est_hours=1.5,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["drink_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"drink_race kcpu {cells_arg}: {sub.message}")
    r = result_json(sub, "drink_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rec in r.items():
        if "error" in rec:
            print(f"{name}: ERROR {rec['error']}", flush=True)
            continue
        (OUT / f"{name}.json").write_text(json.dumps(rec))
        print(f"{name}: written", flush=True)
    return r


def status() -> dict:
    cur = {s: code_shas(s) for s in SPECS}
    st = {"current": [], "stale": [], "missing": []}
    for arm in ARMS:
        for spec in SPECS:
            for s in SEEDS:
                p = OUT / f"{arm}_{spec}_s{s}.json"
                name = p.stem
                if not p.exists():
                    st["missing"].append(name)
                elif json.loads(p.read_text()).get("code") != cur[spec]:
                    st["stale"].append(name)
                else:
                    st["current"].append(name)
    return st


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        od = Path(sys.argv[5]) if len(sys.argv) > 5 else OUT
        print(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4]), od))
    elif cmd == "kernel":
        print(json.dumps({k: ("error" in v) for k, v in kernel(sys.argv[2]).items()}))
    else:
        print(json.dumps({k: len(v) if k != "missing" else v[:6] for k, v in status().items()}))
