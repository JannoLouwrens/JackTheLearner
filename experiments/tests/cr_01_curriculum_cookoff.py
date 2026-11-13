"""CR.01 — the curriculum cook-off: does staging his world teach him more than no staging?

Runner for the spec registered 2026-11-13 (registry_expansion.py, ORDERS J5).
Every number below is copied from the registered spec; if this file and the
spec disagree, the spec wins and this file is the bug.

THE LIVES are `experiments.cr00_pilot.main` — the CR.00 pilot harness, the
same arms, STAGES, W, EPS and BAND declared there before any claim seed —
called with TRAIN = 40,000 training decisions, then an exam that ends when
N_EXAM = 47 exam lives have ENDED, capped at EXAM_CAP = 200,000 exam decisions
(cr00_pilot `exam_lives` mode: the counted lives are never censored).

    experiment(seed)  candidate arms null, staged, lp, frontier
    control(seed)     the two controls that must fail, shuffled and matched,
                      plus the per-seed comparisons that need both sides
    check             per-seed 0/1 flags aggregate as means (1.0 = every seed)

Scored on exam lives only, by FED-LIFE FRACTION (exam lives with >= 1 meal /
47). X in {staged, lp, frontier} WINS iff on EVERY claim seed (130, 131, 132)
fed(X) - fed(null) >= .20 AND fed(X) - fed(matched) >= .20, for X = staged
also fed(staged) - fed(shuffled) >= .20, and pooled over the three seeds X's
median exam span >= null's. VOID if any arm-seed ends < 47 exam lives inside
EXAM_CAP, or null's exam fed is 0/47 or 47/47 on any seed.

ROUTE. Budget KAGGLE_CPU: the arm lives run in Kaggle CPU kernels (S1's
route, as LF.04), two arm-seeds per kernel, side by side. CR01_BACKEND=local
forces local children (debugging only; ~2.2 CPU-h per arm-seed is far over
this box's day). The rows come back; the gate and the ledger stay here.
Queued second on S1, behind LF.04.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TRAIN = 40_000
N_EXAM = 47
EXAM_CAP = 200_000
MARGIN = 0.20
SEED_OFFSET = 130                       # claim seeds 130, 131, 132
CANDIDATES = ("staged", "lp", "frontier")
EXPERIMENT_ARMS = ("null",) + CANDIDATES
CONTROL_ARMS = ("shuffled", "matched")
CACHE = "/data/tmp/cr01"
REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"
CHILD_TIMEOUT_S = 6 * 3600
REMOTE_TIMEOUT_S = 11 * 3600            # under Kaggle's 12 h session
_ENV_BACKEND = os.environ.get("CR01_BACKEND", "")


def _cache(arm: str, s: int) -> str:
    return f"{CACHE}/claim_{arm}_s{s}.jsonl"


# ── scoring: pure functions of an arm-seed's rows ─────────────────────────
def exam_rows(rows: list) -> list:
    """The first N_EXAM ENDED exam lives (born after TRAIN decisions)."""
    return [r for r in rows if r["phase"] == "exam" and not r["censored"]][:N_EXAM]


def fed_fraction(rows: list):
    ex = exam_rows(rows)
    if len(ex) < N_EXAM:
        return None                       # VOID: too few exam lives
    return sum(1 for r in ex if r["ate"] > 0) / N_EXAM


def spans(rows: list) -> list:
    return [float(r["sim_s"]) for r in exam_rows(rows)]


def beats(a, b) -> float:
    return float(a is not None and b is not None and a - b >= MARGIN - 1e-12)


# ── the lives, wherever the backend says ──────────────────────────────────
def _backend() -> str:
    if _ENV_BACKEND:
        return "" if _ENV_BACKEND == "local" else _ENV_BACKEND
    from ..registry import BY_ID
    return ("kaggle-cpu" if BY_ID["CR.01"].budget.value == "kcpu<12h" else "")


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
from experiments.tests.cr_01_curriculum_cookoff import _arms_parallel
r = _arms_parallel(__ARMS__, __SEED__, os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "cr01_arms.json"), "w"))
print("DONE", flush=True)
'''


def _arms_parallel(arms, s, tmp, local=True):
    """Up to two arm-seeds side by side as child processes (<= 2 workers)."""
    procs = {}
    for arm in arms:
        out = f"{tmp}/{arm}_s{s}.jsonl"
        cmd = [PY if local else sys.executable, "-m", "experiments.cr00_pilot",
               arm, str(s), str(TRAIN), str(EXAM_CAP), out, str(N_EXAM)]
        procs[arm] = (subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), out)
    res = {}
    for arm, (p, out) in procs.items():
        _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        if p.returncode != 0:
            raise RuntimeError(f"{arm} s{s} child rc={p.returncode}: "
                               f"{err.decode()[-800:]}")
        res[arm] = [json.loads(x) for x in open(out) if x.strip()]
    return res


def _arms(arms, s) -> dict:
    """Run `arms` at seed s in pairs; cache every arm-seed's rows."""
    os.makedirs(CACHE, exist_ok=True)
    res = {}
    for i in range(0, len(arms), 2):
        pair = list(arms[i:i + 2])
        if _backend() != "kaggle-cpu":
            r = _arms_parallel(pair, s, CACHE)
        else:
            from ..gpu import build_job, result_json, submit
            body = _JOB.replace("__ARMS__", repr(pair)).replace(
                "__SEED__", str(int(s)))
            sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=5.0,
                         timeout_s=REMOTE_TIMEOUT_S, fetch=["cr01_arms.json"])
            if not sub.ok:
                raise RuntimeError(f"CR.01 kaggle-cpu {pair} s{s}: {sub.message}")
            r = result_json(sub, "cr01_arms.json")
        for arm, rows in r.items():
            with open(_cache(arm, s), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in rows)
        res.update(r)
    return res


def _load(arm: str, s: int):
    p = _cache(arm, s)
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else None


# ── the registered run ────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    s = seed + SEED_OFFSET
    r = _arms(EXPERIMENT_ARMS, s)
    f = {a: fed_fraction(r[a]) for a in EXPERIMENT_ARMS}
    m = {"seed": float(s),
         "void_short": float(any(v is None for v in f.values())),
         "null_no_room": float(f["null"] is not None
                               and f["null"] in (0.0, 1.0))}
    for a in EXPERIMENT_ARMS:
        m[f"fed_{a}"] = -1.0 if f[a] is None else f[a]
        m[f"exam_lives_{a}"] = float(len(exam_rows(r[a])))
        m[f"meals_exam_{a}"] = float(sum(x["ate"] for x in exam_rows(r[a])))
    for x in CANDIDATES:
        m[f"{x}_beats_null"] = beats(f[x], f["null"])
    return m


def _control(seed: int) -> dict:
    s = seed + SEED_OFFSET
    r = _arms(CONTROL_ARMS, s)
    f = {a: fed_fraction(r[a]) for a in CONTROL_ARMS}
    for a in EXPERIMENT_ARMS:             # the experiment's cached rows
        rows = _load(a, s)
        f[a] = fed_fraction(rows) if rows is not None else None
    c = {"void_short": float(any(v is None for v in f.values()))}
    for a in CONTROL_ARMS:
        c[f"fed_{a}"] = -1.0 if f[a] is None else f[a]
    for x in CANDIDATES:
        c[f"{x}_beats_matched"] = beats(f[x], f["matched"])
    c["staged_beats_shuffled"] = beats(f["staged"], f["shuffled"])
    # REPORTED, never seated: a control that beats null by the margin
    c["shuffled_beats_null"] = beats(f["shuffled"], f["null"])
    c["matched_beats_null"] = beats(f["matched"], f["null"])
    return c


def pooled_span_ok(x: str, seeds=(0, 1, 2)):
    """Pooled over the claim seeds, X's median exam span >= null's.
    None if any cache is missing (the run cannot be scored -> VOID)."""
    sx, sn = [], []
    for seed in seeds:
        rx, rn = _load(x, seed + SEED_OFFSET), _load("null", seed + SEED_OFFSET)
        if rx is None or rn is None:
            return None
        sx += spans(rx)
        sn += spans(rn)
    return bool(np.median(sx) >= np.median(sn))


def winners(m: dict, c: dict, span_ok=pooled_span_ok) -> list:
    """Every clause on every seed (aggregated flag means == 1.0)."""
    out = []
    for x in CANDIDATES:
        ok = (m.get(f"{x}_beats_null") == 1.0
              and c.get(f"{x}_beats_matched") == 1.0
              and (x != "staged" or c.get("staged_beats_shuffled") == 1.0))
        if ok and span_ok(x):
            out.append(x)
    return out


def _check(m: dict, c: dict, span_ok=pooled_span_ok):
    from ..protocol import Status
    if m.get("void_short", 1.0) != 0.0 or c.get("void_short", 1.0) != 0.0:
        return Status.VOID                # an arm-seed ended < 47 exam lives
    if m.get("null_no_room", 1.0) != 0.0:
        return Status.VOID                # null 0/47 or 47/47 on some seed
    if any(span_ok(x) is None for x in CANDIDATES):
        return Status.VOID
    return bool(winners(m, c, span_ok))


def run(ledger=None):
    from ..protocol import run_spec
    from ..registry import BY_ID
    return run_spec(BY_ID["CR.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


# ── selftest: the gate on planted rows, no lives ──────────────────────────
def _rows(fed: int, n: int = N_EXAM, span: float = 100.0, train: int = 3):
    r = [{"phase": "train", "censored": 0, "ate": 1, "sim_s": 1.0}] * train
    r += [{"phase": "exam", "censored": 0, "ate": int(i < fed),
           "sim_s": span} for i in range(n)]
    return r + [{"phase": "exam", "censored": 1, "ate": 5, "sim_s": 9e9}]


def _selftest() -> dict:
    """Plant fed counts, run the real scoring + gate, check every branch."""
    from ..protocol import Status
    res = {}
    # censored trailing life and train lives are never counted
    res["fed_counts_exam_only"] = fed_fraction(_rows(10)) == 10 / N_EXAM
    res["short_is_void"] = fed_fraction(_rows(10, n=46)) is None
    # margin boundary: 10 lives of 47 = .2128 >= .20; 9 = .1915 < .20
    res["margin_10_fires"] = beats(20 / 47, 10 / 47) == 1.0
    res["margin_9_holds"] = beats(19 / 47, 10 / 47) == 0.0

    def flags(win: bool, ctrl_close: bool):
        m = {"void_short": 0.0, "null_no_room": 0.0}
        c = {"void_short": 0.0}
        for x in CANDIDATES:
            m[f"{x}_beats_null"] = 1.0 if (win and x == "lp") else 0.0
            c[f"{x}_beats_matched"] = (0.0 if ctrl_close else 1.0) if x == "lp" else 0.0
        c["staged_beats_shuffled"] = 0.0
        return m, c
    yes = lambda x: True                                   # noqa: E731
    m, c = flags(True, False)
    res["lp_wins"] = _check(m, c, yes) is True and winners(m, c, yes) == ["lp"]
    m, c = flags(True, True)
    res["matched_close_blocks"] = _check(m, c, yes) is False
    m, c = flags(False, False)
    res["no_gain_fails"] = _check(m, c, yes) is False
    m, c = flags(True, False)
    m["lp_beats_null"] = 2 / 3                             # 2 of 3 seeds only
    res["two_of_three_fails"] = _check(m, c, yes) is False
    m, c = flags(True, False)
    res["span_clause_blocks"] = _check(m, c, lambda x: False) is False
    m, c = flags(True, False)
    m["null_no_room"] = 1 / 3
    res["null_no_room_void"] = _check(m, c, yes) == Status.VOID
    m, c = flags(True, False)
    c["void_short"] = 1 / 3
    res["short_arm_void"] = _check(m, c, yes) == Status.VOID
    # staged needs the shuffled clause too
    m, c = flags(False, False)
    m["staged_beats_null"], c["staged_beats_matched"] = 1.0, 1.0
    res["staged_without_shuffled_fails"] = _check(m, c, yes) is False
    c["staged_beats_shuffled"] = 1.0
    res["staged_with_shuffled_wins"] = winners(m, c, yes) == ["staged"]
    res["ok"] = all(res.values())
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(_selftest(), indent=1))
    else:
        raise SystemExit("usage: python -m experiments.tests.cr_01_curriculum_cookoff selftest")
