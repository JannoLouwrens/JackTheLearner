"""J1 step 1 — the POLICY-LOSS RACE on fresh seeds, on Kaggle CPU.

RULE, DECLARED AND COMMITTED BEFORE ANY CELL (ORDERS J1, 11-15):

  Arms     base / tanh / awr / tanh-awr — exactly `lf04_collapse_diag.VARIANTS`,
           the 11-12 s90 race's arms, run by the same `lf04_collapse_diag.main`
           (same _Life4, same world, core, critic, cadence, exploration; each
           arm changes ONLY the policy term).
  Seeds    150, 151, 152 (fresh: never run by any arm). 12 lives per cell.
  Readout  per cell: life spans (sim_s) and act_sat per life. Thirds of 12
           lives: early = lives 0-3, late = lives 8-11. A CENSORED life (it
           hit LIFE_CAP; the diag stops there) fills every remaining life at
           its censored length — LF.04's own rule (a lower bound on him).
  Qualify  an arm qualifies iff BOTH
           (q1) its saturated-action fraction <= 0.05, read as the mean act_sat
                over ALL late-third lives pooled over the 3 seeds (a filled
                censored life repeats its own act_sat), and
           (q2) NO seed whose late-third mean span < 0.5 x its early-third
                mean span.
  Winner   highest mean late-third span (pooled over seeds) among qualifying
           arms. No arm qualifies -> "no winner; base stays". A missing or
           errored cell makes the tally INCOMPLETE: no winner is named.
  Ties     exact ties are not broken by argument: both named, base preferred
           if base is one of them (the seat does not move on a tie).

What the winner buys: the seat in `cores.py` BY this tally, then LC.02, LC.08,
LF.03 re-bought (LF.03 with its ruling-9 statue twin first). This module never
edits `cores.py`; it only names the winner.

Cell files: /data/jack-logs/policy_loss_race/<arm>_s<seed>.jsonl (the diag's
per-life rows) + <arm>_s<seed>.code.json (sha256 of every file on the path).
The tally counts only cells whose stamp matches the current tree. This file is
not on the stamped path, so editing the tally stales no cell.

Usage:
  python -m experiments.policy_loss_race run ARM SEED [OUTDIR]     (one cell)
  python -m experiments.policy_loss_race kernel SEED   (4 arms of one seed, one
      Kaggle CPU kernel; JACK_REUSE_KERNEL=<user>/<slug> reattaches)
  python -m experiments.policy_loss_race status
  python -m experiments.policy_loss_race tally
  python -m experiments.policy_loss_race selftest   (each clause can fail)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/data/jack-logs/policy_loss_race")
ARMS = ("base", "tanh", "awr", "tanh-awr")
SEEDS = (150, 151, 152)
N_LIVES = 12
SAT_MAX = 0.05
DROP_FRAC = 0.5
CODE_FILES = ("experiments/lf04_collapse_diag.py", "experiments/cores.py",
              "experiments/tests/lf_04_living_teaches.py",
              "experiments/tests/lf_03_learner_lives.py",
              "experiments/needs.py", "playground.py")
CHILD_TIMEOUT_S = 10 * 3600
REMOTE_TIMEOUT_S = 11 * 3600


def code_shas() -> dict:
    out = {}
    for f in CODE_FILES:
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


def run_cell(arm: str, seed: int, outdir: Path = OUT) -> Path:
    from experiments import lf04_collapse_diag as D
    if arm not in ARMS:
        raise KeyError(arm)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{arm}_s{int(seed)}.jsonl"
    code = code_shas()
    D.main(int(seed), N_LIVES, out_path=str(p), variant=arm)
    (outdir / f"{arm}_s{int(seed)}.code.json").write_text(json.dumps(code))
    return p


def run_parallel(seed: int, outdir, local=True) -> dict:
    procs = []
    for arm in ARMS:
        cmd = [sys.executable, "-m", "experiments.policy_loss_race", "run",
               arm, str(seed), str(outdir)]
        procs.append((f"{arm}_s{seed}", subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, p in procs:
        _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        if p.returncode != 0:
            res[name] = {"error": f"rc={p.returncode}: {err.decode()[-800:]}"}
            continue
        d = Path(outdir)
        res[name] = {"rows": [json.loads(l) for l in
                              (d / f"{name}.jsonl").read_text().splitlines() if l.strip()],
                     "code": json.loads((d / f"{name}.code.json").read_text())}
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.policy_loss_race import run_parallel
r = run_parallel(__SEED__, os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "plr_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(seed: int) -> dict:
    if int(seed) not in SEEDS:
        raise ValueError(f"seed {seed} not declared {SEEDS}")
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(_JOB.replace("__SEED__", str(int(seed)))),
                 prefer="kaggle-cpu", est_hours=1.5,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["plr_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"policy_loss_race kcpu s{seed}: {sub.message}")
    r = result_json(sub, "plr_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rec in r.items():
        if "error" in rec:
            print(f"{name}: ERROR {rec['error']}", flush=True)
            continue
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in rec["rows"]))
        (OUT / f"{name}.code.json").write_text(json.dumps(rec["code"]))
        print(f"{name}: written ({len(rec['rows'])} rows)", flush=True)
    return r


def _filled(rows: list) -> list:
    """Per-life (span, act_sat), N_LIVES long; a censored last life fills."""
    out = [(float(r["sim_s"]), float(r["act_sat"])) for r in rows[:N_LIVES]]
    if rows and int(rows[-1].get("censored", 0)) and len(out) < N_LIVES:
        out += [out[-1]] * (N_LIVES - len(out))
    return out


def tally_cells(cells: dict) -> dict:
    """cells: {(arm, seed): rows or None}. Pure — the selftest drives it."""
    third = N_LIVES // 3
    arms, missing = {}, []
    for arm in ARMS:
        per_seed, late_spans, late_sat = {}, [], []
        for s in SEEDS:
            rows = cells.get((arm, s))
            lives = _filled(rows) if rows else []
            if len(lives) < N_LIVES:
                missing.append(f"{arm}_s{s}")
                continue
            early = [x[0] for x in lives[:third]]
            late = lives[-third:]
            e = sum(early) / third
            l_ = sum(x[0] for x in late) / third
            per_seed[s] = {"early": round(e, 1), "late": round(l_, 1),
                           "late_sat": round(sum(x[1] for x in late) / third, 4),
                           "dropped": l_ < DROP_FRAC * e}
            late_spans += [x[0] for x in late]
            late_sat += [x[1] for x in late]
        if len(per_seed) < len(SEEDS):
            arms[arm] = {"per_seed": per_seed, "complete": False}
            continue
        sat = sum(late_sat) / len(late_sat)
        q1 = sat <= SAT_MAX
        q2 = not any(v["dropped"] for v in per_seed.values())
        arms[arm] = {"per_seed": per_seed, "complete": True,
                     "late_mean": round(sum(late_spans) / len(late_spans), 1),
                     "late_sat": round(sat, 4), "q1_sat": q1, "q2_no_drop": q2,
                     "qualifies": q1 and q2}
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing, "arms": arms}
    q = {a: v["late_mean"] for a, v in arms.items() if v["qualifies"]}
    if not q:
        return {"verdict": "NO WINNER; base stays", "arms": arms}
    best = max(q.values())
    top = [a for a, v in q.items() if v == best]
    win = "base" if "base" in top else top[0]
    return {"verdict": f"WINNER {win}", "winner": win, "tied": top, "arms": arms}


def _load(cur: dict) -> dict:
    cells = {}
    for arm in ARMS:
        for s in SEEDS:
            p, c = OUT / f"{arm}_s{s}.jsonl", OUT / f"{arm}_s{s}.code.json"
            if p.exists() and c.exists() and json.loads(c.read_text()) == cur:
                cells[(arm, s)] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            else:
                cells[(arm, s)] = None
    return cells


def status() -> dict:
    cur = code_shas()
    st = {"current": [], "stale_or_uncoded": [], "missing": []}
    for arm in ARMS:
        for s in SEEDS:
            p, c = OUT / f"{arm}_s{s}.jsonl", OUT / f"{arm}_s{s}.code.json"
            if not p.exists():
                st["missing"].append(p.stem)
            elif not c.exists() or json.loads(c.read_text()) != cur:
                st["stale_or_uncoded"].append(p.stem)
            else:
                st["current"].append(p.stem)
    return st


def selftest() -> dict:
    """Each clause must be seen to fire on a planted cell."""
    def life(span, sat=0.0, cens=0):
        return {"sim_s": span, "act_sat": sat, "censored": cens}

    good = [life(400)] * 4 + [life(450)] * 4 + [life(500)] * 4
    cells = {(a, s): list(good) for a in ARMS for s in SEEDS}
    cells[("awr", 150)] = [life(400)] * 4 + [life(450)] * 4 + [life(600)] * 4
    r_win = tally_cells(cells)
    sat = dict(cells); sat[("awr", 150)] = [life(400)] * 8 + [life(900, 0.9)] * 4
    r_sat = tally_cells(sat)
    drop = dict(cells); drop[("awr", 150)] = [life(400)] * 8 + [life(150)] * 4
    r_drop = tally_cells(drop)
    miss = dict(cells); miss[("tanh", 151)] = None
    r_miss = tally_cells(miss)
    cens = dict(cells); cens[("awr", 151)] = [life(400)] * 4 + [life(5000, 0, 1)]
    r_cens = tally_cells(cens)
    checks = {
        "winner_named": r_win.get("winner") == "awr",
        "sat_clause_fires": r_sat["arms"]["awr"]["q1_sat"] is False,
        "drop_clause_fires": r_drop["arms"]["awr"]["q2_no_drop"] is False,
        "missing_is_incomplete": r_miss["verdict"] == "INCOMPLETE",
        "censored_fills": r_cens["arms"]["awr"]["per_seed"][151]["late"] == 5000.0,
        "tie_keeps_base": tally_cells({(a, s): list(good) for a in ARMS
                                       for s in SEEDS}).get("winner") == "base",
    }
    return {"ok": all(checks.values()), **checks}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        od = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
        print(run_cell(sys.argv[2], int(sys.argv[3]), od))
    elif cmd == "kernel":
        print(json.dumps({k: ("error" in v) for k, v in kernel(int(sys.argv[2])).items()}))
    elif cmd == "tally":
        print(json.dumps(tally_cells(_load(code_shas())), indent=1))
    elif cmd == "selftest":
        r = selftest()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    else:
        print(json.dumps(status(), indent=1))
