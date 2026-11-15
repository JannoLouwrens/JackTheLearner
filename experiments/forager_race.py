"""J1 step 2 — the FORAGER RACE: does any learning core learn to eat?

RULE, DECLARED AND COMMITTED BEFORE ANY CELL (ORDERS J1, 11-15):

  Arms     the 7 LC.09 arms (`lc_09_learning_core_3x.ARMS`: ppo-needs, ppo-lp,
           dreamer-xs, wm-efe, wm-latent, wm-latent+ring, wm-dyn), each run
           exactly as LC.09's `run_arm` runs it (same reward hooks, LC.02
           committed train_ratio per arm, e0 1.0, constant explore_std 0.3,
           survival.py / survival_rec.py loops), collapse GUARD ON; plus
           `random` (survival.py policy="random", same world, same e0) as the
           FLOOR. The guard's verdict is REPORTED per cell, not gated on: the
           bar below is LF_04_FAIL.md's, fixed, and ruling 8 makes a VOID the
           guard's finding, not a refusal.
  Seeds    153, 154 (fresh: never run by any arm). 15 lives per cell.
  Lives    a life ends at a death (W0 lethal). A life that reaches LIFE_CAP =
           12,000 decisions (LF.04's cap) is CENSORED: the cell stops and the
           censored life fills every remaining life (LF.04's rule; its meals
           are a lower bound). The cell's decision budget is 15 x LIFE_CAP.
  Readout  per arm-seed, per life: floor meals (obj0 + obj1) and apples, read
           from `w.drives.ate_total` deltas between deaths, through the
           reward_fn hook (runner-side; no world file changes). Late third =
           lives 10-14. late_floor / late_apple = their late-third means;
           late_nu = 0.08 x late_floor + 0.5 x late_apple (energy eaten per
           late life, drives.py's nu values).
  Bar      an arm CLEARS iff on BOTH seeds
           (b1) late_floor >= 5.0 OR late_apple >= 1.0   (LF_04_FAIL.md §2), and
           (b2) late_nu > the random floor's late_nu on the same seed
                (strictly above the floor).
           A missing or errored cell (arm or floor) makes the tally
           INCOMPLETE: nothing clears. Cells count only if their code stamp
           matches the current tree.
  Then     none clears -> J1 step 3 (top two by mean late_nu over both seeds,
           4x the lives, meals-per-life curve). One or more clear -> written
           to ORDERS J1; the Director orders LF.04 + the seat contest.

  Cells run only AFTER step 1 seats the policy loss in cores.py: cores.py is
  stamped, so a cell bought before the seat stales itself.

Cell files: /data/jack-logs/forager_race/<arm>_s<seed>.json + stamp inside.

Usage:
  python -m experiments.forager_race run ARM SEED [OUTDIR]   (one cell, local)
  python -m experiments.forager_race kernel K      (K in 0-3: 4 cells, one
                                                    Kaggle CPU kernel)
  python -m experiments.forager_race status | tally | selftest
  python -m experiments.forager_race smoke ARM N   (N decisions, no file)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/data/jack-logs/forager_race")
LC09_ARMS = ("ppo-needs", "ppo-lp", "dreamer-xs", "wm-efe", "wm-latent",
             "wm-latent+ring", "wm-dyn")
FLOOR = "random"
ARMS = LC09_ARMS + (FLOOR,)
SEEDS = (153, 154)
N_LIVES = 15
LIFE_CAP = 12_000
FLOOR_MEALS_BAR = 5.0
APPLE_BAR = 1.0
NU_FLOOR, NU_APPLE = 0.08, 0.5
FLOOR_FOODS = ("obj0", "obj1")
N_KERNELS = 4
CELLS = [(a, s) for s in SEEDS for a in ARMS]          # 16 cells
CODE_FILES = ("experiments/forager_race.py", "experiments/survival.py",
              "experiments/survival_rec.py", "experiments/cores.py",
              "experiments/cores_dyn.py", "experiments/cores_ring.py",
              "experiments/w0.py", "experiments/drives.py",
              "experiments/tests/lc_09_learning_core_3x.py",
              "experiments/tests/lc_03_survival_screening.py",
              "playground.py")
CHILD_TIMEOUT_S = 10 * 3600
REMOTE_TIMEOUT_S = 11 * 3600


def code_shas() -> dict:
    out = {}
    for f in CODE_FILES:
        p = REPO / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    return out


def kernel_cells(k: int) -> list:
    """Kernel k's 4 cells: arms k and k+4, both seeds."""
    return CELLS[k::N_KERNELS]


class _Stop(Exception):
    pass


class ForageHook:
    """Rides the reward_fn hook (after w.decide, once per decision). Records
    per-life meals by kind and span; raises _Stop at N_LIVES or LIFE_CAP."""

    def __init__(self, inner=None, n_lives: int = N_LIVES,
                 life_cap: int = LIFE_CAP):
        self.inner, self.n_lives, self.life_cap = inner, n_lives, life_cap
        self.lives: list = []
        self.k_life = 0
        self.ate0 = None

    def _ate(self, w):
        t = w.drives.ate_total
        return (sum(int(t.get(f, 0)) for f in FLOOR_FOODS), int(t.get("apple", 0)))

    def __call__(self, r_h, w, obs, core):
        r = self.inner(r_h, w, obs, core) if self.inner else r_h
        if self.ate0 is None:
            self.ate0 = (0, 0)
        self.k_life += 1
        fl, ap = self._ate(w)
        if w.died_this_decision:
            self.lives.append({"sim_s": round(float(w.life_lengths[-1]), 2),
                               "floor": fl - self.ate0[0],
                               "apple": ap - self.ate0[1],
                               "decisions": self.k_life, "censored": 0})
            self.ate0, self.k_life = (fl, ap), 0
            if len(self.lives) >= self.n_lives:
                raise _Stop()
        elif self.k_life >= self.life_cap:
            self.lives.append({"sim_s": round(float(w.sim_seconds
                                                    - w._life_started_at), 2),
                               "floor": fl - self.ate0[0],
                               "apple": ap - self.ate0[1],
                               "decisions": self.k_life, "censored": 1})
            raise _Stop()
        return r


def run_cell(arm: str, seed: int, cal: dict, n_decisions: int = None) -> dict:
    """One arm-seed. Mirrors lc_09.run_arm (guard on) with ForageHook outermost."""
    import time
    from experiments.tests import lc_09_learning_core_3x as L
    if arm not in ARMS:
        raise KeyError(arm)
    n = n_decisions or N_LIVES * LIFE_CAP
    t0 = time.time()
    rec = None
    if arm == FLOOR:
        from experiments.survival import run_survival
        hook = ForageHook()
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
        hook = ForageHook(inner=inner)
        common = dict(j0=cal["j0"], alpha=cal["alpha"], n_decisions=n,
                      train=True, train_ratio=cal["ratios"][arm],
                      e0=L.E0_LC09, explore_std=L.EXPLORE_STD_LC09,
                      reward_fn=hook)
        if arm in ("wm-dyn", "wm-latent+ring"):
            from experiments.survival_rec import run_survival_rec
            fn = lambda: run_survival_rec(seed, arm=arm, **common)
        else:
            from experiments.survival import run_survival
            fn = lambda: run_survival(seed, policy="core", arm=arm,
                                      **common, **hooks)
    try:
        fn()
        ended = "budget"
    except _Stop:
        ended = "stop"
    out = {"arm": arm, "seed": int(seed), "lives": hook.lives, "ended": ended,
           "wall_s": round(time.time() - t0, 1), "code": code_shas()}
    if rec is not None:
        dim = rec.readings[0]["dim"] if rec.readings else 0
        out["guard_n"] = len(rec.readings)
        out["guard_void"] = (L.guard_void(rec.readings, dim) if rec.readings
                             else "no guard reading")
    return out


def _cal() -> dict:
    from experiments.tests.lc_09_learning_core_3x import _borrow_lc09
    cal, prov = _borrow_lc09()
    if cal is None:
        raise RuntimeError(f"calibration unborrowable: {prov}")
    return {"j0": cal["j0"], "alpha": cal["alpha"], "ratios": dict(cal["ratios"])}


def run_parallel(cells: list, cal: dict, outdir, local=True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cal.json").write_text(json.dumps(cal))
    procs = []
    for arm, s in cells:
        cmd = [sys.executable, "-m", "experiments.forager_race", "run", arm,
               str(s), str(outdir)]
        procs.append((f"{arm}_s{s}", subprocess.Popen(
            (["nice", "-n", "19"] if local else []) + cmd, cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res = {}
    for name, p in procs:
        _, err = p.communicate(timeout=CHILD_TIMEOUT_S)
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
from experiments.forager_race import run_parallel
r = run_parallel(__CELLS__, __CAL__, os.environ["JACK_OUT"], local=False)
json.dump(r, open(os.path.join(os.environ["JACK_OUT"], "forager_cells.json"), "w"))
print("DONE", flush=True)
'''


def kernel(k: int) -> dict:
    if not 0 <= int(k) < N_KERNELS:
        raise ValueError(f"kernel {k} not in 0-{N_KERNELS - 1}")
    from experiments.gpu import build_job, result_json, submit
    cells = kernel_cells(int(k))
    body = _JOB.replace("__CELLS__", repr(cells)).replace("__CAL__", repr(_cal()))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=3.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["forager_cells.json"])
    if not sub.ok:
        raise RuntimeError(f"forager_race kcpu k{k}: {sub.message}")
    r = result_json(sub, "forager_cells.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rec in r.items():
        if "error" in rec:
            print(f"{name}: ERROR {rec['error']}", flush=True)
            continue
        (OUT / f"{name}.json").write_text(json.dumps(rec))
        print(f"{name}: written ({len(rec['lives'])} lives)", flush=True)
    return r


# ── the tally: pure ──────────────────────────────────────────────────────
def _filled(lives: list) -> list:
    out = list(lives[:N_LIVES])
    if out and int(out[-1].get("censored", 0)) and len(out) < N_LIVES:
        out += [out[-1]] * (N_LIVES - len(out))
    return out


def late_readout(lives: list):
    """None if the cell is short; else late-third means."""
    f = _filled(lives)
    if len(f) < N_LIVES:
        return None
    late = f[-(N_LIVES // 3):]
    fl = sum(x["floor"] for x in late) / len(late)
    ap = sum(x["apple"] for x in late) / len(late)
    return {"late_floor": round(fl, 3), "late_apple": round(ap, 3),
            "late_nu": round(NU_FLOOR * fl + NU_APPLE * ap, 4),
            "late_span": round(sum(x["sim_s"] for x in late) / len(late), 1)}


def tally_cells(cells: dict) -> dict:
    """cells: {(arm, seed): lives-list or None}."""
    read, missing = {}, []
    for a in ARMS:
        for s in SEEDS:
            lv = cells.get((a, s))
            r = late_readout(lv) if lv else None
            if r is None:
                missing.append(f"{a}_s{s}")
            read[(a, s)] = r
    arms = {}
    for a in LC09_ARMS:
        per = {}
        for s in SEEDS:
            r, fl = read[(a, s)], read[(FLOOR, s)]
            if r is None or fl is None:
                continue
            b1 = r["late_floor"] >= FLOOR_MEALS_BAR or r["late_apple"] >= APPLE_BAR
            b2 = r["late_nu"] > fl["late_nu"]
            per[s] = {**r, "b1_meals": b1, "b2_above_floor": b2}
        done = len(per) == len(SEEDS)
        arms[a] = {"per_seed": per, "complete": done,
                   "clears": done and all(v["b1_meals"] and v["b2_above_floor"]
                                          for v in per.values())}
        if done:
            arms[a]["mean_late_nu"] = round(
                sum(v["late_nu"] for v in per.values()) / len(per), 4)
    floor = {s: read[(FLOOR, s)] for s in SEEDS}
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing, "arms": arms,
                "floor": floor}
    clear = [a for a, v in arms.items() if v["clears"]]
    if clear:
        return {"verdict": f"CLEARS: {', '.join(clear)}", "clears": clear,
                "arms": arms, "floor": floor}
    top2 = sorted(arms, key=lambda a: -arms[a]["mean_late_nu"])[:2]
    return {"verdict": "NONE CLEARS -> step 3", "step3_top2": top2,
            "arms": arms, "floor": floor}


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
            st["current"].append(p.stem)
    return st


def selftest() -> dict:
    """Each clause must be seen to fire on a planted cell, and the hook must
    count meals and stop on a fake world."""
    def L(fl=0, ap=0, span=300.0, c=0):
        return {"floor": fl, "apple": ap, "sim_s": span, "censored": c}

    flo = [L(1)] * 15
    base = {(a, s): [L(1)] * 15 for a in ARMS for s in SEEDS}
    good = dict(base)
    for s in SEEDS:
        good[("wm-dyn", s)] = [L(1)] * 10 + [L(6)] * 5
    r_good = tally_cells(good)
    one = dict(good); one[("wm-dyn", 154)] = [L(1)] * 10 + [L(4)] * 5
    apple = dict(base)
    for s in SEEDS:
        apple[("ppo-lp", s)] = [L()] * 10 + [L(0, 1)] * 5
    floor_hi = dict(good)
    for s in SEEDS:
        floor_hi[(FLOOR, s)] = [L(9)] * 15
    miss = dict(good); miss[(FLOOR, 153)] = None
    cens = dict(base)
    for s in SEEDS:
        cens[("wm-efe", s)] = [L(1)] * 10 + [L(7, 0, 9000.0, 1)]
    short = dict(base); short[("ppo-needs", 153)] = [L(1)] * 14

    class _D:
        def __init__(self):
            self.ate_total = {"apple": 0, "obj0": 0, "obj1": 0}

    class _W:
        def __init__(self):
            self.drives, self.died_this_decision = _D(), False
            self.life_lengths, self.sim_seconds, self._life_started_at = [], 0.0, 0.0
    w, h = _W(), ForageHook(n_lives=2, life_cap=50)
    stopped = False
    try:
        for t in range(1000):
            w.sim_seconds += 0.2
            w.died_this_decision = t in (9, 19)
            if t == 3:
                w.drives.ate_total["obj0"] += 1
            if t == 12:
                w.drives.ate_total["apple"] += 1
                w.drives.ate_total["obj1"] += 2
            if w.died_this_decision:
                w.life_lengths.append(w.sim_seconds - w._life_started_at)
                w._life_started_at = w.sim_seconds
            assert h(0.5, w, None, None) == 0.5
    except _Stop:
        stopped = True
    w2, h2 = _W(), ForageHook(n_lives=5, life_cap=50)
    try:
        for t in range(1000):
            w2.sim_seconds += 0.2
            h2(0.0, w2, None, None)
    except _Stop:
        pass
    checks = {
        "clearer_named": r_good.get("clears") == ["wm-dyn"],
        "one_seed_short_fails": "wm-dyn" not in tally_cells(one).get("clears", []),
        "apple_path_clears": tally_cells(apple).get("clears") == ["ppo-lp"],
        "floor_clause_fires": tally_cells(floor_hi)["arms"]["wm-dyn"]["per_seed"][153]["b2_above_floor"] is False,
        "missing_floor_incomplete": tally_cells(miss)["verdict"] == "INCOMPLETE",
        "short_cell_incomplete": tally_cells(short)["verdict"] == "INCOMPLETE",
        "censored_fills": tally_cells(cens)["arms"]["wm-efe"]["per_seed"][153]["late_floor"] == 7.0,
        "none_goes_step3": tally_cells(base)["verdict"].startswith("NONE"),
        "hook_counts_per_life": stopped and [(x["floor"], x["apple"]) for x in h.lives] == [(1, 0), (2, 1)],
        "hook_censors_at_cap": len(h2.lives) == 1 and h2.lives[0]["censored"] == 1
                               and h2.lives[0]["decisions"] == 50,
        "kernels_cover_cells": sorted(c for k in range(N_KERNELS)
                                      for c in kernel_cells(k)) == sorted(CELLS),
    }
    return {"ok": all(checks.values()), **checks}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        od = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
        od.mkdir(parents=True, exist_ok=True)
        cf = od / "cal.json"
        cal = json.loads(cf.read_text()) if cf.exists() else _cal()
        r = run_cell(sys.argv[2], int(sys.argv[3]), cal)
        (od / f"{sys.argv[2]}_s{int(sys.argv[3])}.json").write_text(json.dumps(r))
        print(len(r["lives"]), r["ended"], r["wall_s"])
    elif cmd == "kernel":
        print(json.dumps({k: ("error" in v) for k, v in kernel(int(sys.argv[2])).items()}))
    elif cmd == "smoke":
        r = run_cell(sys.argv[2], 153, _cal(), n_decisions=int(sys.argv[3]))
        r.pop("code")
        print(json.dumps(r, indent=1))
    elif cmd == "tally":
        print(json.dumps(tally_cells(_load(code_shas())), indent=1, default=str))
    elif cmd == "selftest":
        r = selftest()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    else:
        print(json.dumps(status(), indent=1))
