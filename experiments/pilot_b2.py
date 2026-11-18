"""pilot_b2.py — Pilot 2 at the entry rung: the FLOOR CARD first (ORDERS J11).

PLAN stage 1 (ready-list item 4), parts "Learning to eat" + "His world",
step 2 (combine the record's cures as options) -> 4 (pilot). Prior:
PUBLISHED_PRIORS.md §13 (the lie-still trap), §1, §8; ruling 30: the next
rule is written from the record and from a LEARNER-FREE floor measurement,
never from brain B's 35 J8 runs. Nothing of J8 is re-run.

THIS FILE, TODAY: only the floor card (J11 item 2). No learner is imported
here, no rule exists yet, and none may be written until the card is in
`docs/research/PILOT_B2.md` (item 3). The learner arms are added to this file
by the commit that freezes the rule.

THE WORLD. J8's pilot world (`pilot_b.make_world`: d 0.75, dr100-eat,
random_needs 1/6, random_layout, life cap 9,000 decisions) with
  spawn_clear = jack_env.SPAWN_CLEAR_R (1.55 m)   on in EVERY cell: J8's VOID
      came from a do-nothing body born touching a food (cause a);
  lazy_contact_audio = True                       ruling 34 (identical world);
  move_cost_dial     off / on                     the card's two columns.
Proofs of the two new options: experiments/tests/test_jack_env_cures.py.

THE FLOOR CARD (learner-free; fixed before any cell ran, 11-17):
  arms    noop   the zero action (world_card.DoNothing)
          held8  uniform(-1,1)^8 held 8 decisions (world_card.Held(seed, 8))
          each with move_cost_dial off ("k1") and on ("k0": movement free
          at d0) -> 4 arms.
  size    16 worlds per arm, seeds 601..616 (the SAME worlds in all four
          arms; never 301-345, and the pilot's own 401+ stay unseen), 125
          consecutive finished lives per world in one persisting world ->
          2,000 lives per arm, 8,000 in all.
  per life  pilot_b's row (decisions, seconds, floor meals, apples, cause,
          intake, basal drain, sustained = capped AND intake >= basal
          drain) + the spawn's distance to the nearest floor food.
  reads   per arm: sustained lives / lives (rate, Poisson 95% upper bound);
          runs of W = 1, 2, 3 consecutive sustained lives inside a world;
          life length (median, 5-95%); meals per life (mean, share of lives
          with >= 1, max). These fix W and the floor clause of the rule
          (J11 item 3): W = the smallest W at which the card predicts <= 0.5
          floor crossings over the pilot's planned floor lives
          (`pilot_b2 rule-inputs` computes it; written before the card
          was complete).
  cost    ~19.5 M decisions (noop lives ~2,750 decisions, held8 ~1,500 with
          the movement drain, ~2,750 without); at Kaggle's measured 60.7
          decisions per CPU-s with lazy audio (J6) ~ 89 CPU-h: 8 kernels of
          8 cells, 4 at a time, ~3 h each. > 60 CPU-min -> Kaggle (ruling 32).

PILOT COST, for the rule's kernel plan (11-17 19:40; read off J8's own Kaggle
returns /data/jack-logs/pilot_b/{k,e,f}*.json, no new run). 28 learner runs of
300k packed 4 to a kernel: each got 0.99-1.01 cores (process CPU / wall), so
Kaggle's 4x4-thread oversubscription (J11 item 4) costs no visible CPU there;
decisions per CPU-s 11.29 / median 12.9 / 18.45 — the HOST sets the speed
(kernel medians k0 17.96 vs k3 11.61; 1- and 2-learner kernels 13.0-15.0, no
faster). Wall per run 4.58-7.37 h against CHILD_TIMEOUT_S 8.5 h: the slowest
host seen leaves 13% margin. J11's arms (B0 K P G KP + K-lr0 = 30 learners,
noop/held8 10 cheap) -> 8 kernels at 4 learners each (+ floors riding along),
~4.6-7.4 h each, 4 sessions at a time = 2 waves. `native_substep=True`
(identical world, J6 2b) is the one margin lever; under a learner it is
measured on the box in the J11 state line. Since 11-17 20:20 `pilot_b.run_cell(
..., partial_path=)` writes each run's finished lives every PARTIAL_EVERY (10)
lives and `run_parallel` falls back to that file (`from_partial`, marked `cut`
with `decisions_reached`) — a learner killed at the 8.5 h cut or by Kaggle's
cap returns its lives, not none (tests/test_pilot_b_partial.py). How a cut
run counts is the rule's to say.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.pilot_b import DIFFICULTY, LIFE_CAP, REWARD, PilotLives  # noqa: E402

OUT = Path("/data/jack-logs/pilot_b2")
CARD_SEEDS = tuple(range(601, 617))
CARD_LIVES = 125
CARD_ARMS = ("noop-k1", "noop-k0", "held8-k1", "held8-k0")
# J11 option (a), 11-18 01:40: the card re-run in the world WITH
# jack_env(food_floor_only=True) (adf457ca: no food laid on the ramp/stairs,
# the statue leak's cause). An arm name ending "-ff" runs that world; nothing
# else about a cell changes (same seeds 601..616, 125 lives, cap, policies).
# Only the two arms the order names: noop (the statue leak lives there) and
# held8-k0 (the highest floor, arm K's judge). Groups f0..f3 = 4 worlds x 2
# arms; kernels write OUT/f*.json, which the c*.json card never globs.
CARD_FF_ARMS = ("noop-k1-ff", "held8-k0-ff")
# J11 option (a)'s gap, 11-18 06:15: the pilot's `held8` floor arm runs WITH
# movement charged (PILOT_ARMS['held8'] move_cost_dial=False = k1) and judges
# B0/P/G, but CARD_FF_ARMS has no held8-k1-ff, so under (a) that floor clause
# had no same-world card. Added under NEW group names f4..f7 (f0..f2 were
# running; their cells are unchanged): 16 cells, same seeds 601..616 and 125
# lives, 4 cells a kernel = one wave of 4 workers (held8-k1 card cells ran
# 3.4 h mean, 4.1 h max wall), so no kernel nears POOL_DEADLINE_S.
# `card --glob 'f*.json'` reads them with no change.
CARD_FF_K1_ARMS = ("held8-k1-ff",)
KERNEL_PINS = ["gymnasium==1.4.0", "mujoco==3.17.0"]
WORKERS = 4
REMOTE_TIMEOUT_S = int(9.5 * 3600)
POOL_DEADLINE_S = int(8.5 * 3600)   # in-kernel stop, leaves pip + queue + fetch margin under REMOTE_TIMEOUT_S


def make_world(seed: int, move_cost_dial: bool = False,
               food_floor_only: bool = False, reward: str = REWARD):
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    return JackEnv(seed=seed, difficulty=DIFFICULTY, reward=reward,
                   random_needs=NEEDS_S_TRP, random_layout=True,
                   lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                   move_cost_dial=move_cost_dial, food_floor_only=food_floor_only)


def run_card_cell(arm: str, seed: int, lives: int = CARD_LIVES,
                  out: Path | None = None) -> dict:
    """`lives` finished lives (death or cap) of one floor arm in one world.
    Written to `out` every 10 lives, so a killed cell keeps what it lived."""
    from experiments.world_card import DoNothing, Held
    pol_name, k, *rest = arm.split("-")
    if rest not in ([], ["ff"]):
        raise ValueError(f"unknown arm {arm}")
    pol = DoNothing(seed) if pol_name == "noop" else Held(seed, 8)
    w = PilotLives(make_world(seed, move_cost_dial=(k == "k0"),
                              food_floor_only=(rest == ["ff"])), cap=LIFE_CAP)
    o, info = w.reset(seed=seed)
    pol.new_life()
    spawn = [float(getattr(w.env, "last_spawn_food_dist", float("nan")))]
    t_wall, t_cpu, n = time.time(), time.process_time(), 0

    def record(done: bool) -> dict:
        rows = [dict(l, spawn_dist=round(spawn[i], 3)) for i, l in enumerate(w.lives)]
        rec = {"arm": arm, "seed": seed, "lives_asked": lives, "done": done,
               "decisions": n, "wall_s": round(time.time() - t_wall, 1),
               "cpu_s": round(time.process_time() - t_cpu, 1), "lives": rows,
               "spawn_clear_failed": int(w.env.spawn_clear_failed),
               "layout_failed": int(w.env.layout_failed)}
        if out is not None:
            tmp = Path(str(out) + ".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.replace(out)
        return rec

    while len(w.lives) < lives:
        o, r, term, trunc, info = w.step(pol.act(o))
        n += 1
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            spawn.append(float(getattr(w.env, "last_spawn_food_dist", float("nan"))))
            if len(w.lives) % 10 == 0:
                record(False)
    return record(True)


def card_cells() -> list:
    return [(a, s) for s in CARD_SEEDS for a in CARD_ARMS]


def card_groups() -> dict:
    """c0..c7: 8 cells each = two worlds x the four arms. f0..f3: 8 cells
    each = four worlds x CARD_FF_ARMS. f4..f7: 4 cells each = four worlds x
    held8-k1-ff."""
    c = card_cells()
    g = {f"c{i}": c[8 * i:8 * i + 8] for i in range(8)}
    f = [(a, s) for s in CARD_SEEDS for a in CARD_FF_ARMS]
    g.update({f"f{i}": f[8 * i:8 * i + 8] for i in range(4)})
    h = [(a, s) for s in CARD_SEEDS for a in CARD_FF_K1_ARMS]
    g.update({f"f{4 + i}": h[4 * i:4 * i + 4] for i in range(4)})
    return g


def run_pool(cells: list, outdir, lives: int = CARD_LIVES,
             workers: int = WORKERS, deadline_s: float | None = None) -> dict:
    """Cells as child processes, `workers` at a time; each keeps its own file.

    `deadline_s` (from the call): past it, live cells are terminated and kept
    with what they wrote (every 10 lives), unstarted cells are recorded empty,
    and the pool returns — so a slow kernel still hands back its summary file
    before the watcher's REMOTE_TIMEOUT_S gives up on it (added 11-17 12:50
    for c4..c7 after c0..c3 ran 5 h vs the ~3 h estimate: held8-k0 smoke lives
    were ~3,900-4,080 decisions, not ~1,500). Changes no cell's measurement."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    todo, live, res = list(cells), [], {}
    t_end = None if deadline_s is None else time.monotonic() + deadline_s

    def harvest(arm, seed, f, p):
        err = p.stderr.read().decode()[-800:] if p.stderr else ""
        rec = (json.loads(f.read_text()) if f.exists()
               else {"arm": arm, "seed": seed, "lives": []})
        rec["rc"] = p.returncode
        if p.returncode != 0:
            rec["error"] = f"rc={p.returncode}: {err}"
        res[f"{arm}_s{seed}"] = rec
        print(f"{arm} s{seed}: rc {p.returncode}, {len(rec['lives'])} lives", flush=True)

    while todo or live:
        while todo and len(live) < workers:
            arm, seed = todo.pop(0)
            f = outdir / f"card_{arm}_s{seed}.json"
            live.append((arm, seed, f, subprocess.Popen(
                [sys.executable, "-m", "experiments.pilot_b2", "cell", "--arm", arm,
                 "--seed", str(seed), "--lives", str(lives), "--out", str(f)],
                cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
        time.sleep(2)
        for item in list(live):
            if item[3].poll() is not None:
                live.remove(item)
                harvest(*item)
        if t_end is not None and time.monotonic() > t_end and (todo or live):
            for item in live:
                item[3].terminate()
            for item in live:
                item[3].wait()
                harvest(*item)
                res[f"{item[0]}_s{item[1]}"]["deadline_cut"] = True
            for arm, seed in todo:
                res[f"{arm}_s{seed}"] = {"arm": arm, "seed": seed, "lives": [],
                                         "deadline_cut": True, "not_started": True}
                print(f"{arm} s{seed}: not started (deadline)", flush=True)
            todo, live = [], []
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.pilot_b2 import run_pool
r = run_pool(__CELLS__, os.environ["JACK_OUT"], __LIVES__, deadline_s=__DEADLINE__)
json.dump({"cells": r}, open(os.path.join(os.environ["JACK_OUT"], "pilot_b2_card.json"), "w"))
print("DONE", flush=True)
'''


def job_body(cells: list, lives: int = CARD_LIVES) -> str:
    return (_JOB.replace("__CELLS__", repr(list(cells)))
            .replace("__PINS__", repr(KERNEL_PINS)).replace("__LIVES__", repr(int(lives)))
            .replace("__DEADLINE__", repr(POOL_DEADLINE_S)))


def kernel(name: str) -> dict:
    """Submit one card group (c0..c7) as one Kaggle CPU kernel; write the
    returned cells to OUT/<name>.json (failed cells kept with their error)."""
    from experiments.gpu import build_job, result_json, submit
    cells = card_groups()[name]
    sub = submit(build_job(job_body(cells)), prefer="kaggle-cpu", est_hours=4.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["pilot_b2_card.json"])
    if "pilot_b2_card.json" in sub.artifacts and sub.ok:
        r = result_json(sub, "pilot_b2_card.json")
    else:
        r = cells_from_artifacts(sub.artifacts, cells)
        if r is None:
            raise RuntimeError(f"pilot_b2 kernel {name}: {sub.message}")
        print(f"pilot_b2 kernel {name}: no summary ({sub.message or 'missing'}); "
              f"assembled from per-cell files", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives"
              + (" (cut)" if rec.get("deadline_cut") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r


def cells_from_artifacts(artifacts: dict, cells: list) -> dict | None:
    """RECOVERY (11-17 13:20): a kernel that outlived its watcher (c0..c3 have
    no in-kernel stop) or hit Kaggle's own 12 h cap leaves no summary file, but
    every cell wrote `card_<arm>_s<seed>.json` to /kaggle/working every 10
    lives. Reattach with JACK_REUSE_KERNEL=<user>/<slug> (zero quota; fetches
    output for any terminal status) and this assembles the card from those.
    A cell is `done` only if its own file says so; a missing file is recorded
    `not_found`, never invented. None when no cell file came back at all."""
    res, found = {}, 0
    for arm, seed in cells:
        key, fn = f"{arm}_s{seed}", f"card_{arm}_s{seed}.json"
        if fn in artifacts:
            rec = json.loads(Path(artifacts[fn]).read_text())
            found += 1
            if not rec.get("done"):
                rec["deadline_cut"] = True
        else:
            rec = {"arm": arm, "seed": seed, "lives": [], "not_found": True}
        rec["from_cell_file"] = True
        res[key] = rec
    return {"cells": res, "assembled": True} if found else None


def _arm_order(by: dict) -> dict:
    """Rows in CARD_ARMS, CARD_FF_ARMS, CARD_FF_K1_ARMS order (others last)."""
    known = CARD_ARMS + CARD_FF_ARMS + CARD_FF_K1_ARMS
    return dict(sorted(by.items(), key=lambda kv: (known.index(kv[0])
                                                   if kv[0] in known else len(known))))


def card_table(paths) -> str:
    """The floor card from harvested kernel files (OUT/c*.json)."""
    by = {}
    for p in paths:
        for rec in json.loads(Path(p).read_text())["cells"].values():
            by.setdefault(rec["arm"], []).append(rec)
    by = _arm_order(by)
    lines = ["| arm | worlds (cut) | lives | sustained | rate (95% upper) | runs W=1/2/3 "
             "| life s median (5-95%) | meals/life mean | lives fed | max meals |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, recs in by.items():
        L = [l for r in recs for l in r["lives"]]
        if not L:
            continue
        s = sum(l["sustained"] for l in L)
        up = _poisson_upper(s) / len(L)
        runs = [0, 0, 0]
        for r in recs:
            streak = 0
            for l in r["lives"]:
                streak = streak + 1 if l["sustained"] else 0
                for W in (1, 2, 3):
                    runs[W - 1] += streak >= W
        secs = np.array([l["seconds"] for l in L])
        meals = np.array([l["floor_meals"] + l["apples"] for l in L])
        lines.append(
            f"| {arm} | {sum(bool(r['lives']) for r in recs)} "
            f"({sum(bool(r.get('deadline_cut')) for r in recs)}) | {len(L)} | {s} | {s / len(L):.5f} ({up:.5f}) "
            f"| {runs[0]}/{runs[1]}/{runs[2]} | {np.median(secs):.0f} "
            f"({np.percentile(secs, 5):.0f}-{np.percentile(secs, 95):.0f}) "
            f"| {meals.mean():.3f} | {(meals > 0).mean():.3f} | {int(meals.max())} |")
    return "\n".join(lines)


PILOT_T = 300_000      # J11 item 3: decisions per pilot run
PILOT_SEEDS = 5        # seeds per floor arm in the pilot


def rule_inputs(paths, T: int = PILOT_T, seeds: int = PILOT_SEEDS) -> str:
    """The numbers J11 item 3's rule is written from, computed the same way
    whatever the card turns out to be (written 11-17 18:45, with 4 of 8
    kernels harvested and BEFORE the card is complete, so the derivation
    cannot be tuned to the full card).

    Per card arm: planned floor lives = seeds * T / mean decisions per life;
    per W in 1..3, the card's rate of W-runs (lives ending a run of >= W
    consecutive sustained lives inside one world, divided by lives) times the
    planned lives = predicted floor crossings in the pilot; and the Poisson
    95% upper bound of that count (the floor clause: crossings <= it). The
    smallest W with predicted <= 0.5 is printed per arm. Which pilot floor
    reads which card arm (do-nothing -> noop-k1; held-8 -> held8-k1; arms
    with move_cost_dial -> the k0 twin) is the rule writer's call, stated
    in the rule; this prints all four.
    """
    by = {}
    for p in paths:
        for rec in json.loads(Path(p).read_text())["cells"].values():
            by.setdefault(rec["arm"], []).append(rec)
    by = _arm_order(by)
    lines = ["| arm | card lives | decisions/life | planned lives | "
             "pred. crossings W=1/2/3 | 95% upper W=1/2/3 | smallest W <= 0.5 |",
             "|---|---|---|---|---|---|---|"]
    for arm, recs in by.items():
        L = [l for r in recs for l in r["lives"]]
        if not L:
            continue
        runs = [0, 0, 0]
        for r in recs:
            streak = 0
            for l in r["lives"]:
                streak = streak + 1 if l["sustained"] else 0
                for W in (1, 2, 3):
                    runs[W - 1] += streak >= W
        dpl = float(np.mean([l["decisions"] for l in L]))
        planned = seeds * T / dpl
        pred = [k / len(L) * planned for k in runs]
        upper = [_poisson_upper(k) / len(L) * planned for k in runs]
        w = next((W for W in (1, 2, 3) if pred[W - 1] <= 0.5), None)
        lines.append(
            f"| {arm} | {len(L)} | {dpl:.0f} | {planned:.0f} | "
            + "/".join(f"{x:.2f}" for x in pred) + " | "
            + "/".join(f"{x:.2f}" for x in upper) + f" | {w if w else '> 3'} |")
    return "\n".join(lines)


# ---- J11 item 3: the pilot's arms as settings (11-18 01:50, NO rule yet) ---
# Only the mapping the order already names (arms B0 K P G KP; floors
# do-nothing, held-8, K with learning rate 0), so the rule writer picks the
# world and seeds and the kernels need no new code. Nothing here runs a
# learner; W, seeds, the floor clause and the world choice ((a) food_floor_only
# or (b) today's + disclosure) belong to the rule. Every arm has spawn_clear
# and lazy_contact_audio (make_world). kind: "b" = brain B (run_b), "floor" =
# run_floor. lr None = run_b's default (3e-4); gamma None = make_b's default.
PILOT_ARMS = {
    "B0":   dict(kind="b", move_cost_dial=False, reward=REWARD, gamma=None, lr=None),
    "K":    dict(kind="b", move_cost_dial=True, reward=REWARD, gamma=None, lr=None),
    "P":    dict(kind="b", move_cost_dial=False, reward="dr100-eat-pos", gamma=None, lr=None),
    "G":    dict(kind="b", move_cost_dial=False, reward=REWARD, gamma=0.999, lr=None),
    "KP":   dict(kind="b", move_cost_dial=True, reward="dr100-eat-pos", gamma=None, lr=None),
    "noop": dict(kind="floor", policy="noop", move_cost_dial=False, reward=REWARD),
    "held8": dict(kind="floor", policy="held8", move_cost_dial=False, reward=REWARD),
    "K-lr0": dict(kind="b", move_cost_dial=True, reward=REWARD, gamma=None, lr=0.0),
}


def pilot_world(arm: str, food_floor_only: bool):
    """seed -> JackEnv for one pilot arm (the world half of its settings)."""
    c = PILOT_ARMS[arm]
    return lambda seed: make_world(seed, move_cost_dial=c["move_cost_dial"],
                                   food_floor_only=food_floor_only,
                                   reward=c["reward"])


def run_pilot_cell(arm: str, seed: int, food_floor_only: bool,
                   T: int = PILOT_T, partial_path=None) -> dict:
    """One pilot run of `arm` (PILOT_ARMS) for T decisions."""
    from experiments.pilot_b import run_b, run_floor
    c = PILOT_ARMS[arm]
    world = pilot_world(arm, food_floor_only)
    if c["kind"] == "floor":
        rec = run_floor(c["policy"], seed, T, world=world, partial_path=partial_path)
    else:
        kw = {} if c["lr"] is None else {"lr": c["lr"]}
        rec = run_b(seed, T, world=world, gamma=c["gamma"],
                    partial_path=partial_path, **kw)
    rec["pilot_arm"] = arm
    rec["food_floor_only"] = bool(food_floor_only)
    return rec



# ---- J11 item 5: the pilot's kernels (11-18 02:20, STILL NO RULE) ----------
# Packing only. Seeds and the world ((a) food_floor_only / (b) today's) are the
# rule's; until the rule's commit sets PILOT_RULE, `pilot_kernel` REFUSES, so no
# learner can be launched before the rule exists (philosophy 4). The order's
# plan: 40 runs (B0 K P G KP K-lr0 x5 learners = 30; noop held8 x5 = 10), 4
# runs a kernel -> 10 kernels p0..p9. Learners are dealt seed-major across arms
# (p0 = B0/K/P/G of the first seed, ...) so no arm sits wholly on one slow host
# (J8 kernels: host sets the speed, medians 11.6 vs 18.0 dec/CPU-s); floors go
# last and share no kernel with more than two learners. Each run is pilot_b's
# child-process cell: own file, partial file every 10 lives, CHILD_TIMEOUT_S
# cut -> `cut` with its lives (pilot_b.from_partial), Kaggle's cap -> per-cell
# files reassembled by pilot_b.cells_from_artifacts. How a cut run counts is
# the rule's to say.
PILOT_LEARNERS = ("B0", "K", "P", "G", "KP", "K-lr0")
PILOT_FLOORS = ("noop", "held8")
PILOT_PER_KERNEL = 4
PILOT_RULE = None   # the rule commit sets e.g. {"seeds": (402, ..., 406), "food_floor_only": True, "rule": "<doc ref>"}


def pilot_cells(seeds) -> list:
    """All runs, learners seed-major across arms, then floors."""
    seeds = tuple(int(s) for s in seeds)
    return ([(a, s) for s in seeds for a in PILOT_LEARNERS]
            + [(a, s) for s in seeds for a in PILOT_FLOORS])


def pilot_groups(seeds) -> dict:
    c = pilot_cells(seeds)
    n = PILOT_PER_KERNEL
    return {f"p{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def run_pilot_parallel(cells: list, outdir, food_floor_only: bool,
                       T: int = PILOT_T) -> dict:
    """pilot_b.run_parallel for PILOT_ARMS: all cells at once as children."""
    from experiments.pilot_b import CHILD_TIMEOUT_S, from_partial, partial_file
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for arm, seed in cells:
        f = outdir / f"pilot_{arm}_s{seed}.json"
        procs.append((arm, seed, f, subprocess.Popen(
            [sys.executable, "-m", "experiments.pilot_b2", "pilot-cell", "--arm", arm,
             "--seed", str(seed), "--decisions", str(T), "--out", str(f),
             "--partial", str(partial_file(f))]
            + (["--food-floor-only"] if food_floor_only else []),
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    res, t0 = {}, time.time()
    for arm, seed, f, p in procs:
        try:
            _, err = p.communicate(timeout=max(1, CHILD_TIMEOUT_S - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        rec = (json.loads(f.read_text()) if f.exists()
               else from_partial(arm, seed, partial_file(f)))
        rec["rc"] = p.returncode
        if p.returncode != 0:
            rec["error"] = f"rc={p.returncode}: {err.decode()[-800:]}"
        res[f"{arm}_s{seed}"] = rec
        print(f"{arm} s{seed}: rc {p.returncode}, {len(rec['lives'])} lives", flush=True)
    return res


_PILOT_JOB = r"""
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.pilot_b2 import run_pilot_parallel
from experiments.brain_b import versions
r = run_pilot_parallel(__CELLS__, os.environ["JACK_OUT"], __FF__, __T__)
json.dump({"cells": r, "versions": versions(), "rule": __RULE__},
          open(os.path.join(os.environ["JACK_OUT"], "pilot_b2_cells.json"), "w"))
print("DONE", flush=True)
"""


def pilot_job_body(cells: list, food_floor_only: bool, T: int = PILOT_T,
                   rule: str = "") -> str:
    from experiments.pilot_b import KERNEL_PINS as B_PINS
    return (_PILOT_JOB.replace("__CELLS__", repr(list(cells)))
            .replace("__PINS__", repr(list(B_PINS)))
            .replace("__FF__", repr(bool(food_floor_only)))
            .replace("__T__", repr(int(T))).replace("__RULE__", repr(str(rule))))


def pilot_kernel(name: str, rule: dict | None = None) -> dict:
    """Submit pilot group p0..p9 as one Kaggle CPU kernel -> OUT/<name>.json.
    REFUSES without a rule (PILOT_RULE, set only by the rule's commit)."""
    rule = PILOT_RULE if rule is None else rule
    if not rule or not rule.get("seeds") or "food_floor_only" not in rule:
        raise RuntimeError("pilot_kernel: no frozen rule (PILOT_RULE is None) — "
                           "J11 item 3 must be committed before any learner runs")
    from experiments.gpu import build_job, result_json, submit
    from experiments.pilot_b import cells_from_artifacts
    cells = pilot_groups(rule["seeds"])[name]
    body = pilot_job_body(cells, rule["food_floor_only"], rule.get("T", PILOT_T),
                          rule.get("rule", ""))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=7.5,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["pilot_b2_cells.json"])
    if sub.ok and "pilot_b2_cells.json" in sub.artifacts:
        r = result_json(sub, "pilot_b2_cells.json")
    else:
        r = cells_from_artifacts(sub.artifacts, cells)
        if r is None:
            raise RuntimeError(f"pilot_b2 pilot kernel {name}: {sub.message}")
        print(f"pilot_b2 pilot kernel {name}: no summary ({sub.message or 'missing'}); "
              f"assembled from per-cell files", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pilot_{name}.json").write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives"
              + (" (cut)" if rec.get("cut") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r

def _poisson_upper(k: int, conf: float = 0.95) -> float:
    """One-sided upper bound on a Poisson mean given k events (exact)."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (k + 1)) / 2.0)


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "kernel", "card", "rule-inputs", "pilot-cell", "pilot-kernel"])
    p.add_argument("--arm", default="noop-k1")
    p.add_argument("--seed", type=int, default=601)
    p.add_argument("--lives", type=int, default=CARD_LIVES)
    p.add_argument("--out", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--decisions", type=int, default=PILOT_T)
    p.add_argument("--partial", default=None)
    p.add_argument("--food-floor-only", action="store_true")
    p.add_argument("--glob", default="c*.json",
                   help="card / rule-inputs: which kernel files (f*.json = the -ff re-run)")
    a = p.parse_args(argv)
    if a.cmd == "cell":
        rec = run_card_cell(a.arm, a.seed, a.lives, Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in rec.items() if k != "lives"}))
    elif a.cmd == "kernel":
        kernel(a.name)
    elif a.cmd == "pilot-cell":
        rec = run_pilot_cell(a.arm, a.seed, a.food_floor_only, a.decisions,
                             partial_path=a.partial)
        Path(a.out).write_text(json.dumps(rec) + "\n")
    elif a.cmd == "pilot-kernel":
        pilot_kernel(a.name)
    elif a.cmd == "rule-inputs":
        print(rule_inputs(sorted(OUT.glob(a.glob))))
    else:
        print(card_table(sorted(OUT.glob(a.glob))))


if __name__ == "__main__":
    main(sys.argv[1:])
