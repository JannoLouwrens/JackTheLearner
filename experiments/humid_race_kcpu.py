"""J6 humidity race: NE.01 (and any W.1/W.2) cells for hum30/hum34 on Kaggle CPU.

Why a separate runner: adding these arms to drink_race_kcpu.py would change
that file's sha. Every drink cell carries its sha in `code`, so all of them
would go stale, k3's in-flight dB cells included. This file only REUSES
drink_race_kcpu (code_shas, _JOB pattern, gpu.submit). It runs cells through
humid_race._apply, the same overlay humid_race.run_cell uses. That overlay
includes the 11-15 W.1(d) flux fix.

Each cell's `code` = drink_race_kcpu.code_shas(spec), plus the shas of the
humidity files. The humid tally must compare against shipped cells whose code
is CURRENT, meaning /data/jack-logs/drink_race/shipped_NE.01_s*. It must not use
heat_race's pre-re-registration ones, which humid_race.tally still reads.

Usage:
  python -m experiments.humid_race_kcpu run ARM SPEC SEED [OUTDIR]
  python -m experiments.humid_race_kcpu kernel ARM:SPEC:SEED[,...]   (1-3 cells)
  python -m experiments.humid_race_kcpu status
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments import drink_race_kcpu as DK

REPO = DK.REPO
OUT = Path("/data/jack-logs/humid_race")
ARMS = ("hum30", "hum34")
SPECS = DK.SPECS
EXTRA_FILES = ("experiments/humid_race.py", "experiments/heat_hazard_table.py",
               "experiments/humid_race_kcpu.py")


def code_shas(spec: str) -> dict:
    out = DK.code_shas(spec)
    for f in EXTRA_FILES:
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


def run_cell(arm: str, spec: str, seed: int, outdir: Path = OUT) -> Path:
    from experiments import heat_race as H
    from experiments import humid_race as HR
    code = code_shas(spec)
    _, mods = HR._apply(arm)
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
        if arm not in ARMS or spec not in SPECS:
            raise KeyError(c)
        out.append((arm, spec, int(seed)))
    if not 1 <= len(out) <= 3:
        raise ValueError("one kernel runs 1-3 cells")
    return out


def run_parallel(cells, outdir, local=True) -> dict:
    procs = []
    for arm, spec, seed in cells:
        cmd = [sys.executable, "-m", "experiments.humid_race_kcpu", "run",
               arm, spec, str(seed), str(outdir)]
        procs.append((f"{arm}_{spec}_s{seed}", subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, p in procs:
        _, err = p.communicate(timeout=DK.CHILD_TIMEOUT_S)
        if p.returncode != 0:
            res[name] = {"error": f"rc={p.returncode}: {err.decode()[-800:]}"}
            continue
        res[name] = json.loads((Path(outdir) / f"{name}.json").read_text())
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.humid_race_kcpu import run_parallel, _parse
r = run_parallel(_parse(__CELLS__), os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "humid_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(cells_arg: str) -> dict:
    _parse(cells_arg)
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(_JOB.replace("__CELLS__", repr(cells_arg))),
                 prefer="kaggle-cpu", est_hours=1.5,
                 timeout_s=DK.REMOTE_TIMEOUT_S, fetch=["humid_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"humid_race kcpu {cells_arg}: {sub.message}")
    r = result_json(sub, "humid_cells.json")
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
    st = {"current": [], "stale_or_uncoded": [], "missing": []}
    for arm in ARMS:
        for spec in SPECS:
            for s in DK.SEEDS:
                p = OUT / f"{arm}_{spec}_s{s}.json"
                if not p.exists():
                    st["missing"].append(p.stem)
                elif json.loads(p.read_text()).get("code") != cur[spec]:
                    st["stale_or_uncoded"].append(p.stem)
                else:
                    st["current"].append(p.stem)
    return st


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        od = Path(sys.argv[5]) if len(sys.argv) > 5 else OUT
        print(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4]), od))
    elif cmd == "kernel":
        print(json.dumps({k: ("error" in v) for k, v in kernel(sys.argv[2]).items()}))
    else:
        print(json.dumps(status(), indent=1))
