"""J5 step 2 — MEAL MEMORY claim grind on Kaggle CPU, and its CLAIM GATE.

The arms, ruler and the claim gate are declared in `experiments/meal_memory.py`
(docstring, "CLAIM GATE, DECLARED BEFORE ANY CLAIM ROW": seeds 94-98, 13000
decisions/arm-seed). Nothing here changes them. Two gaps closed:

1. `meal_memory.tally` implements the PILOT rule (>= 2 of 3 seeds, margin .10)
   and says nothing of the claim gate's 4-of-5 or its born_d >= 0.7 m clause.
   Running the claim rows through it would grade them by the wrong, weaker
   rule. `claim_gate` below is the declared gate, verbatim, with fixtures that
   show each clause can fail (`selftest`).
2. A runner: ruling 3 sends learner-side grinds to Kaggle CPU (no box CPU).
   Pilot price 11-13: 6500 decisions ~ 9 CPU-min, so a 13000-decision cell is
   ~18-20 min; one kernel runs 3 cells side by side (4 cores), 5 kernels total.

Each cell row-file is accompanied by `<arm>_s<seed>.code.json`, the sha256 of
every file on its code path; the gate counts only cells whose stamp matches
the current tree (unknown is not a match). The gate lives in THIS module, which
is not on the stamped path, so editing it stales no cell.

Usage:
  python -m experiments.meal_memory_kcpu kernel ARM:SEED[,ARM:SEED[,ARM:SEED]]
      (JACK_REUSE_KERNEL=<user>/<slug> reattaches at zero quota, as CR.01)
  python -m experiments.meal_memory_kcpu status     # current / stale / missing
  python -m experiments.meal_memory_kcpu gate       # the declared claim gate
  python -m experiments.meal_memory_kcpu selftest
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/data/jack-logs/meal_memory_claim")
ARMS = ("fifo", "meal", "yoked")
SEEDS = (94, 95, 96, 97, 98)
TOTAL = 13000
MARGIN = 0.10
FAR_D = 0.7
WIN_SEEDS = 4
CODE_FILES = ("experiments/meal_memory.py", "experiments/cores.py",
              "experiments/survival.py", "experiments/drives.py",
              "experiments/smell_sense.py", "experiments/policy_loss.py",
              "experiments/tests/lf_03_learner_lives.py", "experiments/w0.py",
              "playground.py")
CHILD_TIMEOUT_S = 6 * 3600
REMOTE_TIMEOUT_S = 8 * 3600


def code_shas() -> dict:
    out = {}
    for f in CODE_FILES:
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


def _parse(cells: str):
    out = []
    for c in cells.split(","):
        arm, seed = c.split(":")
        if arm not in ARMS or int(seed) not in SEEDS:
            raise KeyError(c)
        out.append((arm, int(seed)))
    if not 1 <= len(out) <= 3:
        raise ValueError("one kernel runs 1-3 cells")
    return out


def run_parallel(cells, outdir, local=True) -> dict:
    """Cells side by side as children; returns {name: {"rows": [...], "code": {...}}}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    code = code_shas()
    procs = []
    for arm, seed in cells:
        name = f"{arm}_s{seed}"
        cmd = [sys.executable, "-m", "experiments.meal_memory", "run", arm,
               str(seed), str(TOTAL), str(outdir / f"{name}.jsonl")]
        procs.append((name, subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, p in procs:
        _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        if p.returncode != 0:
            res[name] = {"error": f"rc={p.returncode}: {err.decode()[-800:]}"}
            continue
        rows = [json.loads(x) for x in open(outdir / f"{name}.jsonl")]
        res[name] = {"rows": rows, "code": code}
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.meal_memory_kcpu import run_parallel, _parse
r = run_parallel(_parse(__CELLS__), os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "meal_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(cells_arg: str) -> dict:
    _parse(cells_arg)
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(_JOB.replace("__CELLS__", repr(cells_arg))),
                 prefer="kaggle-cpu", est_hours=1.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["meal_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"meal_memory kcpu {cells_arg}: {sub.message}")
    r = result_json(sub, "meal_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rec in r.items():
        if "error" in rec:
            print(f"{name}: ERROR {rec['error']}", flush=True)
            continue
        with open(OUT / f"{name}.jsonl", "w") as f:
            for row in rec["rows"]:
                f.write(json.dumps(row) + "\n")
        (OUT / f"{name}.code.json").write_text(json.dumps(rec["code"]))
        print(f"{name}: written ({len(rec['rows'])} rows)", flush=True)
    return r


def status(outdir: Path = OUT) -> dict:
    cur = code_shas()
    st = {"current": [], "stale": [], "missing": []}
    for arm in ARMS:
        for s in SEEDS:
            name = f"{arm}_s{s}"
            rp, cp = Path(outdir) / f"{name}.jsonl", Path(outdir) / f"{name}.code.json"
            if not rp.exists() or not cp.exists():
                st["missing"].append(name)
            elif json.loads(cp.read_text()) != cur:
                st["stale"].append(name)
            else:
                st["current"].append(name)
    return st


def _read(rows):
    """Read trials as meal_memory.tally reads them: k0 >= WARM, n > 0."""
    from experiments.meal_memory import WARM
    return [r for r in rows if r["k0"] >= WARM and r["n"] > 0]


def _frac(rows):
    return (sum(r["fed"] is not None for r in rows), len(rows))


def claim_gate(cells: dict) -> dict:
    """cells: {(arm, seed): rows}. The declared CLAIM GATE, verbatim:
    `meal` wins iff it beats `fifo` on >= 4 of 5 seeds AND `yoked` on >= 4 of 5
    AND pooled fed fraction exceeds each by >= 0.10 AND, among read trials with
    born_d >= 0.7 m, its pooled fed fraction exceeds each of theirs.
    Anything else: DEAD. Returns verdict None if any of the 15 cells is absent."""
    if any((a, s) not in cells for a in ARMS for s in SEEDS):
        return {"verdict": None, "why": "INCOMPLETE"}
    rd = {k: _read(v) for k, v in cells.items()}
    per = {k: (lambda f, n: f / n if n else 0.0)(*_frac(v)) for k, v in rd.items()}

    def pooled(arm, far=False):
        f = n = 0
        for s in SEEDS:
            rows = [r for r in rd[(arm, s)] if not far or r["born_d"] >= FAR_D]
            a, b = _frac(rows)
            f, n = f + a, n + b
        return f / n if n else float("nan")

    P = {a: pooled(a) for a in ARMS}
    F = {a: pooled(a, far=True) for a in ARMS}
    wins = {o: sum(per[("meal", s)] > per[(o, s)] for s in SEEDS)
            for o in ("fifo", "yoked")}
    clauses = {}
    for o in ("fifo", "yoked"):
        clauses[f"seeds_vs_{o}"] = wins[o] >= WIN_SEEDS
        clauses[f"margin_vs_{o}"] = P["meal"] - P[o] >= MARGIN
        clauses[f"far_vs_{o}"] = bool(F["meal"] > F[o])   # nan compares False
    return {"verdict": "WIN" if all(clauses.values()) else "DEAD",
            "clauses": clauses, "seed_wins": wins, "pooled": P, "pooled_far": F}


def gate(outdir: Path = OUT) -> dict:
    st = status(outdir)
    if st["missing"] or st["stale"]:
        return {"verdict": None, "why": "need 15 current cells", "status":
                {k: len(v) for k, v in st.items()}}
    cells = {}
    for name in st["current"]:
        arm, s = name.split("_s")
        cells[(arm, int(s))] = [json.loads(x) for x in open(Path(outdir) / f"{name}.jsonl")]
    return claim_gate(cells)


def selftest() -> None:
    """Fixtures: a clean win, and each clause lost alone must be DEAD."""
    from experiments.meal_memory import WARM

    def rows(fed, n, far_fed=None, far_n=0):
        out = [{"k0": WARM, "n": 10, "born_d": 0.3,
                "fed": 5 if i < fed else None} for i in range(n)]
        if far_n:
            out += [{"k0": WARM, "n": 10, "born_d": 0.9,
                     "fed": 5 if i < far_fed else None} for i in range(far_n)]
        out.append({"k0": 0, "n": 10, "born_d": 0.3, "fed": 5})   # unread warm-up
        return out

    def cells(meal, fifo, yoked):
        return {(a, s): rows(*spec[i]) for a, spec in
                (("meal", meal), ("fifo", fifo), ("yoked", yoked))
                for i, s in enumerate(SEEDS)}

    W = [(6, 10, 3, 5)] * 5
    L = [(4, 10, 1, 5)] * 5
    assert claim_gate(cells(W, L, L))["verdict"] == "WIN"
    # only 3 of 5 seed wins vs fifo (two ties), margin still large
    fifo3 = L[:3] + [(9, 10, 1, 5)] * 2
    g = claim_gate(cells(W, fifo3, L))
    assert g["verdict"] == "DEAD" and not g["clauses"]["seeds_vs_fifo"], g
    # wins every seed, margin < .10 vs yoked
    near = [(7, 10, 1, 5)] * 5                       # 8/15 vs meal 9/15 -> .067
    g = claim_gate(cells(W, L, near))
    assert g["verdict"] == "DEAD" and not g["clauses"]["margin_vs_yoked"], g
    # wins seeds + margin but loses the far-born clause vs fifo
    farL = [(3, 10, 4, 5)] * 5
    g = claim_gate(cells(W, farL, L))
    assert g["verdict"] == "DEAD" and not g["clauses"]["far_vs_fifo"] \
        and g["clauses"]["margin_vs_fifo"], g
    # one cell missing -> no verdict
    c = cells(W, L, L)
    del c[("yoked", 98)]
    assert claim_gate(c)["verdict"] is None
    print("SELFTEST OK (win, seeds, margin, far, incomplete)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "kernel":
        print(json.dumps({k: ("error" in v) for k, v in kernel(sys.argv[2]).items()}))
    elif cmd == "gate":
        print(json.dumps(gate(), default=str))
    elif cmd == "selftest":
        selftest()
    else:
        print(json.dumps({k: len(v) if k != "missing" else v[:6]
                          for k, v in status().items()}))
