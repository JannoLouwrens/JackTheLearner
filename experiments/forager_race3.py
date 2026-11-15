"""J1 step 3: no arm cleared the forager race. Is that slow learning or none?

RULE, DECLARED AND COMMITTED BEFORE ANY CELL (ORDERS J1 step 3, 11-15):

  Why a separate module: `forager_race.py` stamps itself, so editing it
  would stale step 2's 16 cells. This file imports step 2's hook and arm
  wiring unchanged.

  Which arms (the TIE-BREAK). Step 2's tally (`forager_race tally`) ranked
  ppo-needs first (mean late_nu .024). Second place was a three-way tie at
  .016: ppo-lp, dreamer-xs and wm-latent+ring. `tally_cells` broke the tie by
  ARMS order, which is a silent tie-break nobody declared. Picking a
  secondary key now, with step 2's per-life data already on disk, would be a
  choice made after looking, so none is picked. ALL THREE tied arms run:
      STEP3_ARMS = ppo-needs, ppo-lp, dreamer-xs, wm-latent+ring
  plus `random` as the control that must NOT read as learning. The world's
  diary crosses deaths, so a drifting world could fake a trend, and random
  is the arm that catches that.
  Seeds 153, 154: the same seeds as step 2, so the lives here extend the
  same curves. 60 lives per cell (4x step 2's 15). LIFE_CAP 12,000 and
  censor-fill work as in step 2.

  The 12 h cap. Step 2 ran 4 cells per 4-vCPU kernel, each cell on torch's
  default thread count. Its ppo-lp and ring cells took ~11.1 ks per 15
  lives, which is ~12.3 h for 60, over Kaggle's 12 h session. Two changes:
    (a) ONE learner cell per kernel, so the cells do not share cores (random:
        both seeds in one kernel, ~1.3 h each).
    (b) a WALL DEADLINE of 11.0 h per cell. At the deadline the cell stops at
        the current decision, and the life in progress is DROPPED: a partial
        life is not a life. The cell is written with the lives it finished
        and `ended: "deadline"`. A cut-off cell keeps its curve. It does not
        return nothing.
  No checkpointing: that would mean editing survival.py and survival_rec.py,
  which would stale every certificate that stamps them.

  Readout per cell: per-life nu = 0.08 x floor + 0.5 x apple (step 2's energy
  units). Q1 = the first 15 lives and Q4 = the last 15 the cell finished.
  rho = Spearman(life index, nu) with average ranks for ties. p = a one-sided
  permutation p over 10,000 shuffles of life order (numpy RandomState(0)),
  counting shuffles with rho >= observed.

  Verdict, per arm. An arm is LEARNING iff on BOTH seeds:
      (L1) rho > 0 and p < 0.05,
      (L2) Q4 mean nu > Q1 mean nu,
      (L3) Q4 mean nu > random's mean nu over all of its lives, same seed.
  Otherwise the arm is NOT LEARNING.
  A cell with fewer than MIN_LIVES = 45 finished lives makes that arm
  INCOMPLETE. That covers a deadline stop, a missing cell or a stale stamp.
  CONTROL: random is put through L1 and L2. If random passes both on either
  seed, the curve is measuring drift and not learning, and the whole step is
  VOID. A random cell under 45 lives makes the step INCOMPLETE.
  Finding: any arm LEARNING means SLOW LEARNING (the lever is lives, so order
  the longer runs). Every arm complete and NOT LEARNING means NO LEARNING
  (the lever is the learner or its reward, not time). Each arm's Q1..Q4 curve
  goes in docs/research/FORAGER_RACE.md either way.

  On record before any step-3 cell: over step 2's 15 lives, random ate 6
  and 7 floor meals (s153/s154), and no learner ate more than 5. Whole-life,
  the learners ate no more than random.

Usage:
  python -m experiments.forager_race3 run ARM SEED OUTDIR   (one cell)
  python -m experiments.forager_race3 kernel ARM [SEED]     (one Kaggle CPU
        kernel: a learner arm with one seed, or `random` for both seeds)
  python -m experiments.forager_race3 status | tally | selftest
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import forager_race as F

REPO = F.REPO
OUT = F.OUT / "step3"
STEP3_ARMS = ("ppo-needs", "ppo-lp", "dreamer-xs", "wm-latent+ring")
FLOOR = F.FLOOR
SEEDS = F.SEEDS
N_LIVES = 60
QUARTER = 15
MIN_LIVES = 45
DEADLINE_S = 11.0 * 3600
CHILD_TIMEOUT_S = int(11.4 * 3600)
REMOTE_TIMEOUT_S = int(12.2 * 3600)
N_PERM = 10_000
P_BAR = 0.05
CELLS = [(a, s) for a in STEP3_ARMS + (FLOOR,) for s in SEEDS]   # 10 cells
CODE_FILES = F.CODE_FILES + ("experiments/forager_race3.py",)


def code_shas() -> dict:
    out = {}
    for f in CODE_FILES:
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


class DeadlineHook(F.ForageHook):
    """Step 2's hook, plus a wall deadline that drops the life in progress."""

    def __init__(self, inner=None, n_lives=N_LIVES, life_cap=F.LIFE_CAP,
                 deadline_s=DEADLINE_S, clock=time.time):
        super().__init__(inner=inner, n_lives=n_lives, life_cap=life_cap)
        self.clock = clock
        self.t_end = clock() + deadline_s
        self.deadlined = False

    def __call__(self, r_h, w, obs, core):
        r = super().__call__(r_h, w, obs, core)
        if self.clock() >= self.t_end:
            self.deadlined = True          # the open life is not appended
            raise F._Stop()
        return r


def run_cell(arm: str, seed: int, cal: dict, deadline_s: float = DEADLINE_S,
             n_lives: int = N_LIVES) -> dict:
    """forager_race.run_cell's wiring with DeadlineHook and 60 lives."""
    from experiments.tests import lc_09_learning_core_3x as L
    if arm not in STEP3_ARMS + (FLOOR,):
        raise KeyError(arm)
    n = n_lives * F.LIFE_CAP
    t0 = time.time()
    rec = None
    if arm == FLOOR:
        from experiments.survival import run_survival
        hook = DeadlineHook(n_lives=n_lives, deadline_s=deadline_s)
        fn = lambda: run_survival(seed, policy="random", j0=cal["j0"],
                                  alpha=cal["alpha"], n_decisions=n,
                                  e0=L.E0_LC09, reward_fn=hook)
    else:
        if arm not in cal["ratios"]:
            raise L.UnpricedArm(f"{arm}: no LC.02 committed train_ratio")
        hooks = L._arm_hooks(arm, seed)
        inner = hooks.pop("reward_fn", None)
        if arm in L.LATENT_ARMS:
            rec = L.GuardRecorder(inner=inner)
            inner = rec
        hook = DeadlineHook(inner=inner, n_lives=n_lives, deadline_s=deadline_s)
        common = dict(j0=cal["j0"], alpha=cal["alpha"], n_decisions=n,
                      train=True, train_ratio=cal["ratios"][arm],
                      e0=L.E0_LC09, explore_std=L.EXPLORE_STD_LC09,
                      reward_fn=hook)
        if arm == "wm-latent+ring":
            from experiments.survival_rec import run_survival_rec
            fn = lambda: run_survival_rec(seed, arm=arm, **common)
        else:
            from experiments.survival import run_survival
            fn = lambda: run_survival(seed, policy="core", arm=arm,
                                      **common, **hooks)
    try:
        fn()
        ended = "budget"
    except F._Stop:
        ended = "deadline" if hook.deadlined else "stop"
    out = {"arm": arm, "seed": int(seed), "lives": hook.lives, "ended": ended,
           "wall_s": round(time.time() - t0, 1), "code": code_shas()}
    if rec is not None:
        dim = rec.readings[0]["dim"] if rec.readings else 0
        out["guard_n"] = len(rec.readings)
        out["guard_void"] = (L.guard_void(rec.readings, dim) if rec.readings
                             else "no guard reading")
    return out


def run_parallel(cells: list, cal: dict, outdir) -> dict:
    import subprocess
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cal.json").write_text(json.dumps(cal))
    procs = [(f"{a}_s{s}", subprocess.Popen(
        [sys.executable, "-m", "experiments.forager_race3", "run", a, str(s),
         str(outdir)], cwd=REPO, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)) for a, s in cells]
    res = {}
    for name, p in procs:
        try:
            _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        f = outdir / f"{name}.json"
        if p.returncode != 0 or not f.exists():
            res[name] = {"error": f"rc={p.returncode}: {err.decode()[-800:]}"}
        else:
            res[name] = json.loads(f.read_text())
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.forager_race3 import run_parallel
r = run_parallel(__CELLS__, __CAL__, os.environ["JACK_OUT"])
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "forager3_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel_cells(arm: str, seed=None) -> list:
    if arm == FLOOR:
        return [(FLOOR, s) for s in SEEDS]
    if arm not in STEP3_ARMS or seed is None or int(seed) not in SEEDS:
        raise ValueError(f"kernel: learner arm needs one seed in {SEEDS}")
    return [(arm, int(seed))]


def kernel(arm: str, seed=None) -> dict:
    from experiments.gpu import build_job, result_json, submit
    cells = kernel_cells(arm, seed)
    body = _JOB.replace("__CELLS__", repr(cells)).replace("__CAL__", repr(F._cal()))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=11.5,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["forager3_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"forager_race3 kcpu {cells}: {sub.message}")
    r = result_json(sub, "forager3_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rec in r.items():
        if "error" in rec:
            print(f"{name}: ERROR {rec['error']}", flush=True)
            continue
        (OUT / f"{name}.json").write_text(json.dumps(rec))
        print(f"{name}: written ({len(rec['lives'])} lives, {rec['ended']})",
              flush=True)
    return r


# ── the tally: pure ──────────────────────────────────────────────────────
def _filled(lives: list) -> list:
    out = list(lives[:N_LIVES])
    if out and int(out[-1].get("censored", 0)) and len(out) < N_LIVES:
        out += [out[-1]] * (N_LIVES - len(out))
    return out


def _nu(lives):
    return np.array([F.NU_FLOOR * x["floor"] + F.NU_APPLE * x["apple"]
                     for x in lives], dtype=float)


def _ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                      # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def trend(nu) -> dict:
    """Spearman rho of nu vs life index + one-sided permutation p."""
    nu = np.asarray(nu, float)
    n = len(nu)
    ri = _ranks(np.arange(n, dtype=float))
    rv = _ranks(nu)
    ri_c = ri - ri.mean()
    den = np.sqrt((ri_c ** 2).sum() * ((rv - rv.mean()) ** 2).sum())
    if den == 0:
        return {"rho": 0.0, "p": 1.0}
    rho = float((ri_c * (rv - rv.mean())).sum() / den)
    rng = np.random.RandomState(0)
    perm = np.array([rng.permutation(rv) for _ in range(N_PERM)])
    rhos = ((perm - rv.mean()) @ ri_c) / den
    p = float((np.sum(rhos >= rho - 1e-12) + 1) / (N_PERM + 1))
    return {"rho": round(rho, 4), "p": round(p, 5)}


def cell_readout(lives):
    f = _filled(lives or [])
    if len(f) < MIN_LIVES:
        return None
    nu = _nu(f)
    q = [round(float(nu[i:i + QUARTER].mean()), 4)
         for i in range(0, len(nu), QUARTER)]
    return {"n_lives": len(f), "q1": float(nu[:QUARTER].mean()),
            "q4": float(nu[-QUARTER:].mean()), "mean_nu": float(nu.mean()),
            "curve": q, "meals_floor": int(sum(x["floor"] for x in f)),
            "meals_apple": int(sum(x["apple"] for x in f)), **trend(nu)}


def tally_cells(cells: dict) -> dict:
    """cells: {(arm, seed): lives-list or None}."""
    read = {k: cell_readout(cells.get(k)) for k in CELLS}
    ctrl = {}
    for s in SEEDS:
        r = read[(FLOOR, s)]
        ctrl[s] = None if r is None else {
            **r, "reads_learning": r["rho"] > 0 and r["p"] < P_BAR
                                   and r["q4"] > r["q1"]}
    arms = {}
    for a in STEP3_ARMS:
        per = {}
        for s in SEEDS:
            r, c = read[(a, s)], ctrl[s]
            if r is None or c is None:
                continue
            per[s] = {**r, "L1": r["rho"] > 0 and r["p"] < P_BAR,
                      "L2": r["q4"] > r["q1"], "L3": r["q4"] > c["mean_nu"]}
        done = len(per) == len(SEEDS)
        arms[a] = {"per_seed": per, "complete": done,
                   "learning": done and all(v["L1"] and v["L2"] and v["L3"]
                                            for v in per.values())}
    if any(c is None for c in ctrl.values()):
        return {"verdict": "INCOMPLETE (control)", "arms": arms, "control": ctrl}
    if any(c["reads_learning"] for c in ctrl.values()):
        return {"verdict": "VOID: random reads as learning (drift)",
                "arms": arms, "control": ctrl}
    learning = [a for a, v in arms.items() if v["learning"]]
    if learning:
        return {"verdict": f"SLOW LEARNING: {', '.join(learning)}",
                "learning": learning, "arms": arms, "control": ctrl}
    inc = [a for a, v in arms.items() if not v["complete"]]
    if inc:
        return {"verdict": "INCOMPLETE", "incomplete": inc, "arms": arms,
                "control": ctrl}
    return {"verdict": "NO LEARNING", "arms": arms, "control": ctrl}


def _load(cur: dict) -> dict:
    cells = {}
    for a, s in CELLS:
        p = OUT / f"{a}_s{s}.json"
        d = json.loads(p.read_text()) if p.exists() else None
        cells[(a, s)] = d["lives"] if d and d.get("code") == cur else None
    return cells


def status() -> dict:
    cur = code_shas()
    st = {"current": [], "stale": [], "missing": []}
    for a, s in CELLS:
        p = OUT / f"{a}_s{s}.json"
        if not p.exists():
            st["missing"].append(p.stem)
        elif json.loads(p.read_text()).get("code") != cur:
            st["stale"].append(p.stem)
        else:
            d = json.loads(p.read_text())
            st["current"].append(f"{p.stem} ({len(d['lives'])} lives, {d['ended']})")
    return st


def selftest() -> dict:
    """Every verdict branch must be seen to fire on planted cells, and the
    deadline must drop the open life on a fake world."""
    def L(fl=0, ap=0, c=0):
        return {"floor": fl, "apple": ap, "sim_s": 300.0, "censored": c}

    rs = np.random.RandomState(7)
    flat = lambda: [L(int(rs.rand() < 0.3)) for _ in range(N_LIVES)]
    rising = lambda: [L(int(rs.rand() < 0.05 + i / 50)) for i in range(N_LIVES)]
    base = {k: flat() for k in CELLS}
    grow = dict(base)
    for s in SEEDS:
        grow[("ppo-lp", s)] = rising()
    one = dict(grow); one[("ppo-lp", 154)] = flat()
    drift = dict(grow); drift[(FLOOR, 153)] = rising()
    hi_floor = dict(grow)
    for s in SEEDS:
        hi_floor[(FLOOR, s)] = [L(9)] * N_LIVES
    short = dict(base); short[("dreamer-xs", 153)] = flat()[:44]
    cut = dict(base); cut[("dreamer-xs", 153)] = flat()[:45]
    nofl = dict(grow); nofl[(FLOOR, 154)] = None
    cens = dict(base); cens[("ppo-needs", 153)] = flat()[:30] + [L(3, 0, 1)]

    class _D:
        def __init__(self):
            self.ate_total = {"apple": 0, "obj0": 0, "obj1": 0}

    class _W:
        def __init__(self):
            self.drives, self.died_this_decision = _D(), False
            self.life_lengths, self.sim_seconds, self._life_started_at = [], 0.0, 0.0
    tick = [0.0]
    w, h = _W(), DeadlineHook(n_lives=60, life_cap=10_000, deadline_s=25.0,
                              clock=lambda: tick[0])
    try:
        for t in range(1000):
            tick[0] += 1.0
            w.sim_seconds += 0.2
            w.died_this_decision = t in (9, 19)
            if t == 22:
                w.drives.ate_total["obj0"] += 1
            if w.died_this_decision:
                w.life_lengths.append(w.sim_seconds - w._life_started_at)
                w._life_started_at = w.sim_seconds
            h(0.0, w, None, None)
    except F._Stop:
        pass
    t_rise = trend([0] * 30 + [1] * 30)
    t_null = trend(_nu(flat()))
    checks = {
        "trend_detects_rise": t_rise["rho"] > 0 and t_rise["p"] < P_BAR,
        "trend_flat_quiet": t_null["p"] > P_BAR,
        "rising_arm_learns": tally_cells(grow).get("learning") == ["ppo-lp"],
        "one_seed_flat_not": "ppo-lp" not in tally_cells(one).get("learning", []),
        "drifting_control_voids": tally_cells(drift)["verdict"].startswith("VOID"),
        "L3_floor_clause_fires": tally_cells(hi_floor)["arms"]["ppo-lp"]["per_seed"][153]["L3"] is False,
        "short_cell_incomplete": tally_cells(short)["verdict"] == "INCOMPLETE",
        "45_lives_counts": tally_cells(cut)["arms"]["dreamer-xs"]["complete"],
        "missing_control_incomplete": tally_cells(nofl)["verdict"].startswith("INCOMPLETE"),
        "censored_fills": tally_cells(cens)["arms"]["ppo-needs"]["per_seed"][153]["n_lives"] == 60,
        "flat_world_no_learning": tally_cells(base)["verdict"] == "NO LEARNING",
        "deadline_drops_open_life": h.deadlined and len(h.lives) == 2
                                    and [x["floor"] for x in h.lives] == [0, 0],
        "cells_cover": sorted([c for a in STEP3_ARMS for s in SEEDS
                               for c in kernel_cells(a, s)]
                              + kernel_cells(FLOOR)) == sorted(CELLS),
    }
    return {"ok": all(checks.values()), **checks}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        od = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
        od.mkdir(parents=True, exist_ok=True)
        cf = od / "cal.json"
        cal = json.loads(cf.read_text()) if cf.exists() else F._cal()
        r = run_cell(sys.argv[2], int(sys.argv[3]), cal)
        (od / f"{sys.argv[2]}_s{int(sys.argv[3])}.json").write_text(json.dumps(r))
        print(len(r["lives"]), r["ended"], r["wall_s"])
    elif cmd == "kernel":
        sd = int(sys.argv[3]) if len(sys.argv) > 3 else None
        print(json.dumps({k: ("error" in v) for k, v in kernel(sys.argv[2], sd).items()}))
    elif cmd == "tally":
        print(json.dumps(tally_cells(_load(code_shas())), indent=1, default=str))
    elif cmd == "selftest":
        r = selftest()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    else:
        print(json.dumps(status(), indent=1))
