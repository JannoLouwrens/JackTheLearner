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
# J11 11-18 13:45, the motor-frame INBOX (12:26, 07:40): gate 2 measured body
# eye + WORLD motor 0/20 vs body motor 20/20, so the pilot's rule will likely
# name motor_frame="body" — and then held8's floors above (world motor: the
# held push points along a world axis) are not the floors of that creature.
# These are the same two held8 floors with ONLY motor_frame="body" changed
# ("-bm"; the held push turns with his heading). noop needs no twin: a zero
# push is zero in either frame. Floors never read the eye, so eye_frame is not
# varied. Built ahead of the ruling so it can launch the slot a session frees;
# if the ruling is "world", these groups are simply never run.
CARD_FF_BM_ARMS = ("held8-k0-ff-bm", "held8-k1-ff-bm")
KERNEL_PINS = ["gymnasium==1.4.0", "mujoco==3.17.0"]
WORKERS = 4
REMOTE_TIMEOUT_S = int(9.5 * 3600)
POOL_DEADLINE_S = int(8.5 * 3600)   # in-kernel stop, leaves pip + queue + fetch margin under REMOTE_TIMEOUT_S


def make_world(seed: int, move_cost_dial: bool = False,
               food_floor_only: bool = False, reward: str = REWARD,
               eye_frame: str = "world", motor_frame: str = "world",
               native_substep: bool = False, eye: str | None = None):
    """The pilot's world. `eye` (J17): None = JackEnv's own default eye, the
    call below unchanged; only J17's arms name one. eye_frame / motor_frame default "world" = the f-card's
    world byte for byte (the floor cards never pass them); the pilot's runs get
    them from the frozen rule (ruling 39; J12 (0), (2')). native_substep
    (11-19): ORDERS — every new run in his world passes it (identical world,
    J6); default off keeps the card cells' code path unchanged."""
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    kw = {} if eye is None else {"eye": eye}
    return JackEnv(seed=seed, difficulty=DIFFICULTY, reward=reward,
                   random_needs=NEEDS_S_TRP, random_layout=True,
                   lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                   move_cost_dial=move_cost_dial, food_floor_only=food_floor_only,
                   eye_frame=eye_frame, motor_frame=motor_frame,
                   native_substep=native_substep, **kw)


def run_card_cell(arm: str, seed: int, lives: int = CARD_LIVES,
                  out: Path | None = None) -> dict:
    """`lives` finished lives (death or cap) of one floor arm in one world.
    Written to `out` every 10 lives, so a killed cell keeps what it lived."""
    from experiments.world_card import DoNothing, Held
    pol_name, k, *rest = arm.split("-")
    if rest not in ([], ["ff"], ["ff", "bm"]) or (pol_name, rest) == ("noop", ["ff", "bm"]):
        raise ValueError(f"unknown arm {arm}")
    pol = DoNothing(seed) if pol_name == "noop" else Held(seed, 8)
    w = PilotLives(make_world(seed, move_cost_dial=(k == "k0"),
                              food_floor_only=("ff" in rest),
                              motor_frame="body" if "bm" in rest else "world"),
                   cap=LIFE_CAP)
    o, info = w.reset(seed=seed)
    pol.new_life()
    spawn = [float(getattr(w.env, "last_spawn_food_dist", float("nan")))]
    t_wall, t_cpu, n = time.time(), time.process_time(), 0

    def record(done: bool) -> dict:
        rows = [dict(l, spawn_dist=round(spawn[i], 3)) for i, l in enumerate(w.lives)]
        rec = {"arm": arm, "seed": seed, "lives_asked": lives, "done": done,
               "decisions": n, "wall_s": round(time.time() - t_wall, 1),
               "cpu_s": round(time.process_time() - t_cpu, 1), "lives": rows,
               "motor_frame": w.env.motor_frame,
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
    held8-k1-ff. m0..m7: 4 cells each = two worlds x CARD_FF_BM_ARMS (both
    body-motor floors in every kernel, so any number of m-kernels home gives
    the two arms the same worlds)."""
    c = card_cells()
    g = {f"c{i}": c[8 * i:8 * i + 8] for i in range(8)}
    f = [(a, s) for s in CARD_SEEDS for a in CARD_FF_ARMS]
    g.update({f"f{i}": f[8 * i:8 * i + 8] for i in range(4)})
    h = [(a, s) for s in CARD_SEEDS for a in CARD_FF_K1_ARMS]
    g.update({f"f{4 + i}": h[4 * i:4 * i + 4] for i in range(4)})
    m = [(a, s) for s in CARD_SEEDS for a in CARD_FF_BM_ARMS]
    g.update({f"m{i}": m[4 * i:4 * i + 4] for i in range(8)})
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
    known = CARD_ARMS + CARD_FF_ARMS + CARD_FF_K1_ARMS + CARD_FF_BM_ARMS
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

# ---- J15 (4) E2c: brain B on arm K's world, HANDED the direction to food ---
# Rule: forager_clone.py "RULE E2c" + its 11-19 AMENDMENT (cossin; zero is the
# primary control, random_point reported only; a 300k miss is weak evidence).
# Each arm is arm K exactly (same world, same B0 SAC, lr, gamma) with the
# observation extended by handed_sense.HandedSense(mode, "cossin") -- a sense
# computed from the simulator: scored-and-ineligible, never his. Kept out of
# PILOT_ARMS so the pilot's 40-run packing is untouched. Known difference:
# PilotLives' cap respawn is not a JackEnv truncation, so random_point
# re-draws there too (a new body = a new life for the control).
E2C_ENCODING = "cossin"
# trace=True (11-19, before seed 801): run_b logs the action spread and SAC's
# losses every 5,000 decisions -- reads only, lives identical (test_pilot_b_trace);
# J11's pilot arms do not carry it (their kernels stay as frozen).
E2C_ARMS = {f"E-{m}": dict(PILOT_ARMS["K"], handed=m, trace=True)
            for m in ("food", "zero", "random_point")}
E2C_SEEDS = (801, 802, 803, 804, 805)
# Ruling 57 (11-20, before any of their seeds): the auto twins -- E-food and
# E-zero exactly, but SAC's entropy coefficient LEARNED (SB3/SBX ent_coef
# "auto", target entropy -dim(action) = -8). §19: a fixed 0.1 on 8 action dims
# has no published precedent and pays ~ one meal's value per living decision.
# A SEPARATE dict and separate kernels a0..a2, so e2c_groups() (e0..e3) and the
# fixed arms' lives stay byte-identical (tests/test_e2c_auto.py).
E2C_AUTO_ARMS = {f"{a}-auto": dict(E2C_ARMS[a], ent_coef="auto")
                 for a in ("E-food", "E-zero")}


# ---- J17: the eye race, row128 vs 16 rays under brain B ---------------------
# Rule: docs/research/prereg/row128_vs_rays16.md, FROZEN by ruling 66 (freeze
# commit 98e22036). Every arm is pilot arm K exactly (K-lr0 for the floor) on
# PILOT_RULE's world with setting S = ent_coef "auto" (ruling 64); ONLY the
# eye differs. No handed sense on any arm. A SEPARATE dict and separate
# kernels r0..r4, so pilot 2's and E2c's groups and arms stay byte-identical
# (tests/test_j17_eye_arms.py). trace=True as E2c: reads only.
#   K-row128-blind  MUST FAIL: the row eye's WIDTH (256 numbers) with every
#                   vision number zero -- the wider input with no content.
J17_ARMS = {
    "K-rays16": dict(PILOT_ARMS["K"], ent_coef="auto", trace=True, eye="rays16"),
    "K-row128": dict(PILOT_ARMS["K"], ent_coef="auto", trace=True, eye="row128"),
    "K-row128-blind": dict(PILOT_ARMS["K"], ent_coef="auto", trace=True,
                           eye="row128", blind=True),
    "K-lr0-row128": dict(PILOT_ARMS["K-lr0"], ent_coef="auto", trace=True,
                         eye="row128"),
}
J17_SEEDS = (921, 922, 923, 924, 925)

# ---- J17 critic hunt (11-22): the cheapest PUBLISHED cure, SimBa ----------
# PUBLISHED_PRIORS §24: sbx 0.28.0 ships SimBa (Lee et al. 2024, ICLR 2025;
# SonyResearch/simba) as SAC("SimbaPolicy"): residual LN blocks, AdamW, LN
# before the head. A SEPARATE dict -- J17_ARMS, PILOT_ARMS and the E2c dicts
# are untouched (tests/test_j17_hunt_arms.py). Each sighted hunt arm carries
# its own blind twin (j17_onset box-twin: box runs are another draw).
# Settings that differ from SimBa's published SAC recipe (stated, not cured):
#   * use_sde False here -- sbx's Simba actor has no gSDE (get_std() = 0), and
#     SimBa samples white Gaussian noise; J17's MlpPolicy arms used gSDE.
#   * sizes: actor 1 block x 128, critic 2 blocks x 512 = SimBa's SAC table;
#   * obs normaliser: our VecNormalize(clip 10) for SimBa's running-stat norm;
#   * gamma 0.995 (J8), 1 update a decision, batch 256, ent_coef auto
#     (ruling 64), learning_starts = WARMUP, his world -- all as J17.
# A PILOT (1 seed per arm, <= 100k decisions, ruling 33 not engaged): it can
# show the runaway is or is not still there, not that SimBa eats.
SIMBA_KW = dict(net_arch=dict(pi=[128], qf=[512, 512]))
J17_HUNT_ARMS = {
    "H-simba-row128": dict(PILOT_ARMS["K"], ent_coef="auto", trace=True, eye="row128",
                           policy="SimbaPolicy", policy_kwargs=SIMBA_KW, use_sde=False),
    "H-simba-row128-blind": dict(PILOT_ARMS["K"], ent_coef="auto", trace=True,
                                 eye="row128", blind=True, policy="SimbaPolicy",
                                 policy_kwargs=SIMBA_KW, use_sde=False),
}
# ---- hunt h1 (11-22, built while h0 is on Kaggle; PUBLISHED_PRIORS §24 arm
# (b), Lyle et al. 2024 "layer normalization and weight decay"): J17's OWN
# K-row128 learner -- MlpPolicy, LN, 2x256, gSDE, everything -- with ONE
# change: optax.adamw in place of optax.adam, at optax's default weight decay
# 1e-4, which is exactly what h0's SimbaPolicy runs. So h0 and h1 split
# SimBa into its two halves: h0 = SimBa's architecture + AdamW, h1 = AdamW
# alone on our architecture. Differences from Lyle's recipe (stated, not
# cured): their weight-decay value was not read (abstract only); sbx hands
# ONE optimiser class to actor and critic, so the actor decays too; the
# entropy coefficient keeps sbx's own adam; optax decays EVERY parameter
# (mask None: biases and LN gains too) -- as in h0. Same pilot terms as h0.
J17_HUNT_ARMS.update({
    "H-wd-row128": dict(J17_ARMS["K-row128"], optimizer="adamw"),
    "H-wd-row128-blind": dict(J17_ARMS["K-row128-blind"], optimizer="adamw"),
})
# ---- hunt h2 (11-22 04:30, built while h0 runs, before any h2 run;
# PUBLISHED_PRIORS §24 arm (c), Hussing 2024 OFN, read from the e-print):
# J17's OWN K-row128 pair with ONE key, critic="ofn" = the critic's last
# hidden features projected onto the unit sphere before its output layer
# (experiments/ofn_critic.py lists every difference from their recipe: our
# LN kept, after-ReLU placement, an eps guard, UTD 1). Optimiser stays adam,
# so h1 vs h2 splits the two non-SimBa cures. Same pilot terms as h0.
J17_HUNT_ARMS.update({
    "H-ofn-row128": dict(J17_ARMS["K-row128"], critic="ofn"),
    "H-ofn-row128-blind": dict(J17_ARMS["K-row128-blind"], critic="ofn"),
})
# ---- hunt h3 (11-22 06:30, built while h2 runs, before any h3 run;
# PUBLISHED_PRIORS §24 arm (d), Lee 2018 / DreamerV3 "a separate encoder per
# sense before fusion"): J17's OWN K-row128 pair with ONE key, critic="split"
# = the critic's first layer split into vision -> Dense(128) -> LN and the
# rest + action -> Dense(128) -> LN, concat -> ReLU, then sbx's own layers
# (experiments/split_critic.py lists every difference). Adam kept. Same terms.
J17_HUNT_ARMS.update({
    "H-split-row128": dict(J17_ARMS["K-row128"], critic="split"),
    "H-split-row128-blind": dict(J17_ARMS["K-row128-blind"], critic="split"),
})
# ---- J22 h4 (11-24, built before any h4 run; ORDERS J22 (1), PUBLISHED_PRIORS
# §12/§24, DrQ): J17's OWN K-row128 pair with ONE key, critic="conv" = the
# ray row as a 1-D image -> 4 x Conv1d(32, k3) -> 50 -> LN -> tanh, shared by
# both Q heads and read DETACHED by the actor, random roll +-6 rays drawn
# separately for obs and next-obs (experiments/conv_eye.py lists every
# difference from DrQ). Third cell: the SHUFFLED twin (blind_eye.ShuffledEye,
# each step's rays permuted), the width-matched control J17 never had.
J17_HUNT_ARMS.update({
    "H-conv-row128": dict(J17_ARMS["K-row128"], critic="conv"),
    "H-conv-row128-blind": dict(J17_ARMS["K-row128-blind"], critic="conv"),
    "H-conv-row128-shuffled": dict(J17_ARMS["K-row128"], critic="conv", shuffled=True),
})
J17_HUNT_GROUPS = {"h0": ("H-simba-row128", "H-simba-row128-blind"),
                   "h1": ("H-wd-row128", "H-wd-row128-blind"),
                   "h2": ("H-ofn-row128", "H-ofn-row128-blind"),
                   "h3": ("H-split-row128", "H-split-row128-blind"),
                   "h4": ("H-conv-row128", "H-conv-row128-blind")}
J17_HUNT_SEED = 923          # the box twin's seed (j17_onset TWIN_SEED)
J17_HUNT_T = 100_000         # sighted/blind separate first at 100k (ac684c09)
J17_T = 300_000
J17_LATE_FROM = J17_T // 2     # ruling 65: the late half = lives STARTING at decision >= 150,000


def arm_cfg(arm: str) -> dict:
    for d in (PILOT_ARMS, E2C_ARMS, E2C_AUTO_ARMS, J17_ARMS, J17_HUNT_ARMS):
        if arm in d:
            return d[arm]
    raise KeyError(arm)


def pilot_world(arm: str, food_floor_only: bool, eye_frame: str = "world",
                motor_frame: str = "world", native_substep: bool = False):
    """seed -> JackEnv for one pilot arm (the world half of its settings).
    The frames apply to EVERY arm, floors included: held8 holds a push, and in
    the body motor frame that push turns with him, so a floor run in another
    motor frame than the learners' is not their floor."""
    c = arm_cfg(arm)

    def world(seed):
        env = make_world(seed, move_cost_dial=c["move_cost_dial"],
                         food_floor_only=food_floor_only,
                         reward=c["reward"], eye_frame=eye_frame,
                         motor_frame=motor_frame,
                         native_substep=native_substep, eye=c.get("eye"))
        if c.get("shuffled"):
            from experiments.blind_eye import ShuffledEye    # J22's width-matched twin
            return ShuffledEye(env, seed)
        if c.get("blind"):
            from experiments.blind_eye import BlindEye       # J17's must-fail arm only
            return BlindEye(env)
        if "handed" not in c:
            return env
        from experiments.handed_sense import HandedSense   # E2c only
        return HandedSense(env, c["handed"], E2C_ENCODING, seed=seed)
    return world


def run_pilot_cell(arm: str, seed: int, food_floor_only: bool,
                   T: int = PILOT_T, partial_path=None, eye_frame: str = "world",
                   motor_frame: str = "world", native_substep: bool = False) -> dict:
    """One pilot run of `arm` (PILOT_ARMS) for T decisions."""
    from experiments.pilot_b import run_b, run_floor
    c = arm_cfg(arm)
    world = pilot_world(arm, food_floor_only, eye_frame, motor_frame, native_substep)
    if c["kind"] == "floor":
        rec = run_floor(c["policy"], seed, T, world=world, partial_path=partial_path)
    else:
        kw = {} if c["lr"] is None else {"lr": c["lr"]}
        if c.get("trace"):
            kw["trace"] = True
        if "ent_coef" in c:
            kw["ent_coef"] = c["ent_coef"]
        for k in ("policy", "policy_kwargs", "use_sde", "optimizer", "critic"):  # J17_HUNT_ARMS only
            if k in c:
                kw[k] = c[k]
        rec = run_b(seed, T, world=world, gamma=c["gamma"],
                    partial_path=partial_path, **kw)
    rec["pilot_arm"] = arm
    if "handed" in c:
        rec["handed"], rec["handed_encoding"] = c["handed"], E2C_ENCODING
    rec["food_floor_only"] = bool(food_floor_only)
    rec["eye_frame"], rec["motor_frame"] = eye_frame, motor_frame
    if "eye" in c:                       # J17 arms only; older records carry no key
        rec["eye"], rec["blind"] = c["eye"], bool(c.get("blind"))
        if c.get("shuffled"):            # J22 only; other records unchanged
            rec["shuffled"] = True
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
# FROZEN 11-19 04:35 (J11 item 3): docs/research/PILOT_B2.md §3. W = 2; floors
# held8-k0-ff-bm (U 0.64) / held8-k1-ff-bm (U 0.80) / noop-k1-ff; motor frame
# body by the INBOX 11-18 07:40 default (no ruling landed; a `world` ruling
# re-drafts and runs on seeds 407+).
PILOT_RULE = {"seeds": (402, 403, 404, 405, 406), "food_floor_only": True,
              "eye_frame": "body", "motor_frame": "body", "native_substep": True,
              "rule": "PILOT_B2.md §3 frozen 2026-11-19 04:35; W=2; "
                      "floors held8-k0-ff-bm U0.64, held8-k1-ff-bm U0.80, noop-k1-ff"}


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
                       T: int = PILOT_T, eye_frame: str = "world",
                       motor_frame: str = "world", native_substep: bool = False) -> dict:
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
            + (["--food-floor-only"] if food_floor_only else [])
            + ["--eye-frame", eye_frame, "--motor-frame", motor_frame]
            + (["--native-substep"] if native_substep else []),
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
r = run_pilot_parallel(__CELLS__, os.environ["JACK_OUT"], __FF__, __T__, __EYE__, __MOTOR__,
                       native_substep=__NATIVE__)
json.dump({"cells": r, "versions": versions(), "rule": __RULE__},
          open(os.path.join(os.environ["JACK_OUT"], "pilot_b2_cells.json"), "w"))
print("DONE", flush=True)
"""


def pilot_job_body(cells: list, food_floor_only: bool, T: int = PILOT_T,
                   rule: str = "", eye_frame: str = "world",
                   motor_frame: str = "world", native_substep: bool = False) -> str:
    from experiments.pilot_b import KERNEL_PINS as B_PINS
    return (_PILOT_JOB.replace("__CELLS__", repr(list(cells)))
            .replace("__PINS__", repr(list(B_PINS)))
            .replace("__FF__", repr(bool(food_floor_only)))
            .replace("__T__", repr(int(T))).replace("__RULE__", repr(str(rule)))
            .replace("__EYE__", repr(str(eye_frame)))
            .replace("__MOTOR__", repr(str(motor_frame)))
            .replace("__NATIVE__", repr(bool(native_substep))))


def pilot_kernel(name: str, rule: dict | None = None, cells: list | None = None,
                 out_name: str | None = None) -> dict:
    """Submit pilot group p0..p9 as one Kaggle CPU kernel -> OUT/<name>.json.
    REFUSES without a rule (PILOT_RULE, set only by the rule's commit)."""
    rule = PILOT_RULE if rule is None else rule
    if not rule or not rule.get("seeds") or "food_floor_only" not in rule:
        raise RuntimeError("pilot_kernel: no frozen rule (PILOT_RULE is None) — "
                           "J11 item 3 must be committed before any learner runs")
    # Ruling 39: every learner arm sees the body-frame eye. The motor frame is
    # the rule's to name (INBOX 11-18 07:40; gate 2's h4: body eye + world
    # motor 0/20 vs body + body 20/20) -- no silent "world" default for either.
    if rule.get("eye_frame") != "body":
        raise RuntimeError("pilot_kernel: the rule must set eye_frame='body' "
                           "(ruling 39)")
    if rule.get("motor_frame") not in ("world", "body"):
        raise RuntimeError("pilot_kernel: the rule must name motor_frame "
                           "('world' or 'body'); if 'body', held8's floor is "
                           "not the f-card's (the f-card ran the world frame)")
    # ORDERS (11-18 sitting): every new run in his world passes
    # native_substep=True (identical world, J6). Found missing 11-19: the
    # pilot's make_world never passed it. No silent default.
    if rule.get("native_substep") is not True:
        raise RuntimeError("pilot_kernel: the rule must set native_substep=True "
                           "(ORDERS: every new run in his world)")
    from experiments.gpu import build_job, result_json, submit
    from experiments.pilot_b import cells_from_artifacts
    if cells is None:
        cells = pilot_groups(rule["seeds"])[name]
    body = pilot_job_body(cells, rule["food_floor_only"], rule.get("T", PILOT_T),
                          rule.get("rule", ""), rule["eye_frame"], rule["motor_frame"],
                          rule["native_substep"])
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
    (OUT / (out_name or f"pilot_{name}.json")).write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives"
              + (" (cut)" if rec.get("cut") else "")
              + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)
    return r

def e2c_cells() -> list:
    """15 runs, seed-major across the three E2c arms."""
    return [(a, s) for s in E2C_SEEDS for a in E2C_ARMS]


def e2c_groups() -> dict:
    c, n = e2c_cells(), PILOT_PER_KERNEL
    return {f"e{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def e2c_auto_cells() -> list:
    """Ruling 57's 10 runs, seed-major across the two auto twins."""
    return [(a, s) for s in E2C_SEEDS for a in E2C_AUTO_ARMS]


def e2c_auto_groups() -> dict:
    c, n = e2c_auto_cells(), PILOT_PER_KERNEL
    return {f"a{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def j17_cells() -> list:
    """20 runs, seed-major: every kernel r0..r4 carries one seed's four arms,
    so a slow host costs every arm the same."""
    return [(a, s) for s in J17_SEEDS for a in J17_ARMS]


def j17_groups() -> dict:
    c, n = j17_cells(), PILOT_PER_KERNEL
    return {f"r{j}": c[n * j:n * (j + 1)] for j in range((len(c) + n - 1) // n)}


def j17_kernel(name: str) -> dict:
    """Submit J17 group r0..r4 -> OUT/j17_<name>.json on PILOT_RULE's world
    (body eye + body motor, food_floor_only, native_substep), T = 300k."""
    if PILOT_T != J17_T:
        raise RuntimeError("j17_kernel: the rule says 300,000 decisions")
    rule = dict(PILOT_RULE, rule="prereg/row128_vs_rays16.md FROZEN ruling 66 (98e22036); "
                                 "S = ent_coef auto; world = " + PILOT_RULE["rule"])
    return pilot_kernel(name, rule, cells=j17_groups()[name], out_name=f"j17_{name}.json")


def j17_hunt_kernel(name: str = "h0") -> dict:
    """J17 critic-hunt PILOT (11-22): J17_HUNT_ARMS (SimBa sighted + its blind
    twin) on seed J17_HUNT_SEED, J17_HUNT_T decisions, in ONE Kaggle CPU kernel
    on PILOT_RULE's world -> OUT/j17_hunt_<name>.json. Not a registered row:
    it reads whether the sighted critic still runs away from its twin by 100k
    (j17_onset's signature), nothing about eating."""
    rule = dict(PILOT_RULE, T=J17_HUNT_T,
                rule="J17 critic hunt PILOT 11-22, SimBa (PUBLISHED_PRIORS §24), "
                     "1 seed, no verdict; world = " + PILOT_RULE["rule"])
    cells = [(a, J17_HUNT_SEED) for a in J17_HUNT_GROUPS[name]]
    return pilot_kernel(name, rule, cells=cells, out_name=f"j17_hunt_{name}.json")


def e2c_ready() -> list:
    """What still blocks E2c: ORDERS J15 (4) runs it only after J11's forty
    runs are home (it inherits their readings). [] = ready."""
    paths = sorted(OUT.glob("pilot_p*.json"))
    _md, missing = pilot_table(paths, PILOT_RULE["seeds"])
    return list(missing)


def e2c_kernel(name: str) -> dict:
    """Submit E2c group e0..e3 -> OUT/e2c_<name>.json, on J11's frozen world
    (PILOT_RULE's frames, food_floor_only, native_substep, T). REFUSES while
    any of J11's forty runs is missing."""
    missing = e2c_ready()
    if missing:
        raise RuntimeError(f"e2c_kernel: J11's pilot is not home ({len(missing)} "
                           "runs missing) -- ORDERS J15 (4) runs E2c after it")
    rule = dict(PILOT_RULE, rule="forager_clone.py RULE E2c + AMENDED 2026-11-19; "
                                 "world = " + PILOT_RULE["rule"])
    groups = e2c_auto_groups() if name.startswith("a") else e2c_groups()
    if name.startswith("a"):
        rule["rule"] += "; auto twins (ruling 57, ent_coef='auto')"
    return pilot_kernel(name, rule, cells=groups[name],
                        out_name=f"e2c_{name}.json")


# ---- J11 item 5: the pilot's TABLE, descriptive only (11-18 06:40, NO rule yet)
# Written before the rule and before any pilot run, so it decides nothing: no
# W is chosen, no floor clause is applied, no reading (sustained / eats but
# starves / slides) is printed -- those are the rule's. It transcribes, per
# arm, the quantities the order names: sustained-mark crossings at W = 1/2/3
# by the frozen harness (pilot_b_report.score; PILOT_T == pilot_b.T_RUN, so
# its scorer applies unchanged), and the J8 early/late table (first 5 lives
# vs last quarter; unfed life length). A cut or crashed run is flagged, its
# lives kept ("crash" appended after its last life, ruling 19); how it COUNTS
# is the rule's. Missing registered runs -> PARTIAL.
def load_pilot(paths) -> tuple:
    """{run_id: rec}, {run_id: file} from harvested pilot kernel files; a run id
    in two files is an error, not a choice."""
    runs, src = {}, {}
    for p in paths:
        for k, rec in json.loads(Path(p).read_text())["cells"].items():
            if k in runs:
                raise ValueError(f"{k} appears in {src[k]} and {Path(p).name}")
            runs[k], src[k] = rec, Path(p).name
    return runs, src


def pilot_table(paths, seeds, arms=None, cells=None) -> tuple:
    """(markdown, missing run ids). Arms in PILOT_LEARNERS + PILOT_FLOORS order
    unless `arms` + `cells` are given (E2c's table, `e2c_table`)."""
    from experiments import pilot_b_report as R
    if PILOT_T != R.T:
        raise RuntimeError("pilot_table: pilot_b_report scores at T_RUN != PILOT_T")
    runs, _ = load_pilot(paths)
    reg = [R.rid(a, s) for a, s in (cells if cells is not None else pilot_cells(seeds))]
    missing = [r for r in reg if r not in runs]
    extra = sorted(set(runs) - set(reg))
    out = [("**PARTIAL** -- " if missing else "") + f"{len(reg) - len(missing)} of "
           f"{len(reg)} registered runs in" + (f"; missing {', '.join(missing)}" if missing else "")
           + (f"; NOT registered (ignored): {', '.join(extra)}" if extra else ""), "",
           "| arm | runs in | cut/crash | runs crossing W=1/2/3 | lives | sustained lives "
           "| meals/life early -> late | % fed early -> late | unfed life s early -> late |",
           "|---|---|---|---|---|---|---|---|---|"]
    arms = tuple(arms) if arms is not None else PILOT_LEARNERS + PILOT_FLOORS
    for arm in arms:
        recs = [runs[R.rid(arm, s)] for s in seeds if R.rid(arm, s) in runs]
        if not recs:
            out.append(f"| {arm} | 0 of {len(seeds)} | | | | | | | |")
            continue
        xs = [R.run_stats(r) for r in recs]
        sc = [R.score(r) for r in recs]
        cr = [sum(sc_["sustained", W].event for sc_ in sc) for W in R.WS]

        def m(key, f, scale=1.0, nd=2):
            """Mean over runs of a run's early/late block field (NaN blocks skipped)."""
            v = [x[key][f] for x in xs if x[key] and x[key][f] == x[key][f]]
            return R._f(scale * st_mean(v) if v else None, nd)
        out.append(
            f"| {arm} | {len(recs)} of {len(seeds)} | {sum(x['crashed'] for x in xs)} "
            f"| {'/'.join(map(str, cr))} | {sum(x['lives'] for x in xs)} "
            f"| {sum(x['sustained_lives'] for x in xs)} "
            f"| {m('early', 'meals')} -> {m('late', 'meals')} "
            f"| {m('early', 'fed', 100.0, 0)} -> {m('late', 'fed', 100.0, 0)} "
            f"| {m('early', 'unfed_s', nd=0)} -> {m('late', 'unfed_s', nd=0)} |")
    out += ["", "Per run (pilot_b_report's columns):", "", R.RUN_HEAD]
    for arm in arms:
        out += R._arm_rows(runs, arm, seeds)
    return "\n".join(out), missing


def e2c_table(paths=None) -> tuple:
    """E2c's 25 registered runs (3 fixed-alpha arms + ruling 57's 2 auto
    twins, seeds 801..805), DESCRIPTIVE ONLY: pilot_table's columns unchanged
    plus, per run, the decision marks (k) at which a life reached the cap, so
    a skill learned and then lost is visible (J15, 11-20). No reading and no
    branch of the rule's table is applied here; a missing run -> PARTIAL."""
    from experiments import pilot_b_report as R
    paths = sorted(paths if paths is not None else OUT.glob("e2c_*.json"))
    arms = tuple(E2C_ARMS) + tuple(E2C_AUTO_ARMS)
    md, missing = pilot_table(paths, E2C_SEEDS, arms=arms,
                              cells=e2c_cells() + e2c_auto_cells())
    runs, _ = load_pilot(paths)
    out = [md, "", "Capped lives, at k decisions (cumulative at the life's end):", ""]
    for a in arms:
        for s_ in E2C_SEEDS:
            r = runs.get(R.rid(a, s_))
            if r is None:
                continue
            t, caps = 0, []
            for life in r["lives"]:
                t += life["decisions"]
                if life["cause"] == "capped":
                    caps.append(t // 1000)
            out.append(f"- {R.rid(a, s_)}: {caps if caps else 'none'}")
    out += ["", *E2C_Q_HEAD]
    for a in arms:
        for s_ in E2C_SEEDS:
            r = runs.get(R.rid(a, s_))
            if r is not None:
                out.append(e2c_q_row(R.rid(a, s_), r))
    out += ["", *E2C_H_HEAD]
    for a in arms:
        for s_ in E2C_SEEDS:
            r = runs.get(R.rid(a, s_))
            if r is not None:
                out.append(e2c_h_row(R.rid(a, s_), r))
    return "\n".join(out), missing


# ---- J15 11-20 18:30: the critic's value estimate beside what the run's own
# eating can support. DESCRIPTIVE, no reading, the frozen rule is untouched.
# SBX SAC's actor loss is mean(alpha*log pi - min(Q1, Q2)), so -actor_loss =
# min-critic Q + alpha*H at the policy's own actions. SUPPORTED is a ceiling on
# the true soft value of a policy that eats at the run's last-50k rate r
# (meals per decision), in dr100-eat (eat_pay 100, jack_env):
#   meals    100 * r / (1 - gamma)          (rate-averaged, not a hard bound)
#   drive    100 * d_max: sum_t g^t (d_t - d_{t+1}) <= d_0 <= d_max, d >= 0
#   entropy  alpha_max * 8 ln 2 / (1 - gamma): a squashed 8-dim action lives
#            in [-1, 1]^8, whose largest differential entropy is 8 ln 2
#   death    charged negative -> omitted (the ceiling only gets looser)
# A Q many times SUPPORTED, in a run that does not eat, is a critic that
# overestimates; it says nothing yet about WHY the eating was lost.
E2C_Q_AT = (50_000, 100_000, 200_000, 300_000)
E2C_Q_HEAD = (
    "Critic's value estimate (-actor_loss, min of twin critics, at k decisions) "
    "beside SUPPORTED = 100·r/(1-γ) + 100·d_max + α_max·8ln2/(1-γ), r = meals per "
    "decision over the last 50k (see e2c_q_row). DESCRIPTIVE, no reading:",
    "",
    "| run | Q 50k | Q 100k | Q 200k | Q 300k | meals/1k last 50k | α max | SUPPORTED | Q 300k ÷ SUPPORTED |",
    "|---|---|---|---|---|---|---|---|---|")


def e2c_q_row(rid: str, rec: dict) -> str:
    import math
    from experiments import pilot_b_report as R
    from experiments.drives import drive
    from experiments.pilot_b import GAMMA as gamma   # E2c arms = arm K: gamma None -> make_b default
    d_max = max(drive(e, i, w) for e in (0.0, 1.0) for i in (0.0, 1.0) for w in (0.0, 1.0))
    trace = [w for w in rec.get("trace", []) if "actor_loss" in w]
    q = {}
    for k in E2C_Q_AT:
        before = [w for w in trace if w["t"] <= k]
        q[k] = -before[-1]["actor_loss"] if before else None
    alpha = max([w["ent_coef"] for w in trace if "ent_coef" in w] or [float("nan")])
    rate = R.run_stats(rec)["meals_per_1k_last50k"] / 1000.0
    sup = (100.0 * rate / (1 - gamma) + 100.0 * d_max
           + alpha * 8 * math.log(2) / (1 - gamma))
    q300 = q[E2C_Q_AT[-1]]
    ratio = (q300 / sup) if (q300 is not None and sup == sup and sup > 0) else None
    f = lambda v, nd=0: "–" if v is None or v != v else f"{v:.{nd}f}"
    return (f"| {rid} | " + " | ".join(f(q[k]) for k in E2C_Q_AT)
            + f" | {f(1000 * rate, 3)} | {f(alpha, 3)} | {f(sup)} | {f(ratio, 1)} |")


# ---- J15 11-20 19:30: the policy's own entropy, recovered from the trace.
# The trace's per-dim action `std` is lost after the first learning window
# (SB3's `train/std` overwrites it; INBOX 18:35), so PILOT_B2.md §3b's
# "std in the band / std falls" reading has no input. But SBX 0.28.0 logs, in
# BOTH modes (sac.py:359; a fixed coefficient still computes it, sac.py:36),
#   ent_coef_loss = log(alpha) * (H - target),  H = -mean(log pi) of the batch
# so H = target + ent_coef_loss / log(alpha), target = -dim(A) = -8.
# Anchor (real runs, 18 home): at the first learning window (10k, ~5k updates
# in) the ten auto runs read H 4.7-5.4 and no run exceeds 5.41 -- at or just
# under 8 x 0.670 = 5.36, PUBLISHED_PRIORS §19's computed entropy of the
# untrained tanh-Gaussian (ceiling 8 ln 2 = 5.55). H is one 256-row
# minibatch's mean at the window's last update. alpha = 1 gives no reading (log 0). DESCRIPTIVE, no reading: H is
# the policy's spread at replayed states, NOT §3b's pooled action std, so the
# frozen 0.51-0.65 band is not applied to it.
E2C_H_AT = (10_000, 50_000, 100_000, 200_000, 300_000)
E2C_H_TARGET = -8.0
E2C_H_HEAD = (
    "Policy entropy H (nats, 8 action dims; from the logged ent_coef_loss = "
    "log α·(H − target), target −8; untrained = 5.36, uniform ceiling 5.55) at "
    "k decisions. §19 cause 1 (entropy bonus dominates) predicts H staying "
    "near 5; a falling H is a sharpening policy. DESCRIPTIVE, no reading:",
    "",
    "| run | H 10k | H 50k | H 100k | H 200k | H 300k | H min | first H ≤ 0 at k | α 50k | α 300k | sat 50k | sat 300k |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|")


def e2c_entropy(w: dict):
    """H of one trace window, or None where the window cannot say (no
    ent_coef_loss / ent_coef logged, alpha <= 0, or alpha == 1)."""
    import math
    a, l = w.get("ent_coef"), w.get("ent_coef_loss")
    if a is None or l is None or a <= 0 or abs(math.log(a)) < 1e-9:
        return None
    return E2C_H_TARGET + l / math.log(a)


def e2c_h_row(rid: str, rec: dict) -> str:
    trace = [w for w in rec.get("trace", []) if "t" in w]
    hs = [(w["t"], e2c_entropy(w)) for w in trace]
    hs = [(t, h) for t, h in hs if h is not None]

    def at(k, seq):
        before = [v for t, v in seq if t <= k]
        return before[-1] if before else None
    al = [(w["t"], w["ent_coef"]) for w in trace if "ent_coef" in w]
    sat = [(w["t"], w["sat"]) for w in trace if "sat" in w]
    first0 = next((t // 1000 for t, h in hs if h <= 0), None)
    f = lambda v, nd=1: "–" if v is None or v != v else f"{v:.{nd}f}"
    return (f"| {rid} | " + " | ".join(f(at(k, hs)) for k in E2C_H_AT)
            + f" | {f(min((h for _, h in hs), default=None))}"
            + f" | {first0 if first0 is not None else '–'}"
            + f" | {f(at(50_000, al), 3)} | {f(at(300_000, al), 3)}"
            + f" | {f(at(50_000, sat), 3)} | {f(at(300_000, sat), 3)} |")


# ---- J11 item 4: THE RULE's readings, transcribed (11-19 05:20, before any
# pilot run is home). PILOT_B2.md §3 frozen `e2c3c378`, clause by clause; the
# table above stays verdict-free. Transcription choices where the frozen text
# leaves a seam (each the stricter reading; none can make an arm look better):
#  (a) crash vs cut: rc = -9 (our own CHILD_TIMEOUT kill) or a record marked
#      `cut` with no rc (Kaggle's cap, reassembled) is a CUT; any other rc != 0
#      or an `error` key is a CRASH (the "rc != 0 / non-finite loss" clause —
#      a non-finite loss is not logged separately; SB3 raises on it -> rc 1).
#  (b) noop's floor(U) is 0 like held8's (card noop sustained 2/2000 lives,
#      U at W=2 < 1); a noop crossing at any W is the broken clause anyway.
#  (c) the EATS-BUT-STARVES floor for K/KP (held8-k0 has no in-pilot twin):
#      the MORE crossing of K-lr0 (its in-pilot bound) and held8.
#  (d) SLIDES' "lives unfed >= 0.9 x noop's median life": the late quarter's
#      median unfed-life seconds (run_stats) vs the median seconds over every
#      finished noop life pooled; a late quarter with no unfed life is not
#      the signature.
PILOT_W = 2
PILOT_FLOOR_U = {"held8-k0-ff-bm": 0.64, "held8-k1-ff-bm": 0.80}
PILOT_FLOOR_MAX = 0                      # floor(U) for U < 1, every floor
PILOT_READ_ARMS = ("B0", "K", "P", "G", "KP")
PILOT_NAMED = {"B0": "held8", "P": "held8", "G": "held8",   # in-pilot k1 twin
               "K": "K-lr0", "KP": "K-lr0"}                # k0: in-pilot bound
PILOT_NEED = 3
PILOT_MIN_DEC = 150_000


def _decisions(rec: dict) -> int:
    return int(rec.get("decisions_reached")
               or sum(l["decisions"] for l in rec["lives"]))


def _crash(rec: dict) -> bool:
    rc = rec.get("rc")
    if rc == -9 or (rc is None and rec.get("cut")):
        return False
    return rc not in (0, None) or "error" in rec


def pilot_readings(runs: dict, seeds) -> dict:
    """THE RULE (§3) on the 40 registered runs; anything missing raises (no
    reading on partial books). Returns {"verdict", "arms": {arm: reading},
    "why": [...], counts}."""
    import statistics as st
    from experiments import pilot_b_report as R
    if PILOT_T != R.T:
        raise RuntimeError("pilot_readings: pilot_b_report scores at T_RUN != PILOT_T")
    seeds = tuple(seeds)
    reg = [R.rid(a, s) for a, s in pilot_cells(seeds)]
    miss = [r for r in reg if r not in runs]
    if miss:
        raise ValueError(f"no reading: missing {miss}")
    by = {a: [runs[R.rid(a, s)] for s in seeds] for a in PILOT_LEARNERS + PILOT_FLOORS}
    sc = {a: [R.score(r) for r in v] for a, v in by.items()}
    n_sus = {a: {W: sum(s["sustained", W].event for s in v) for W in R.WS} for a, v in sc.items()}
    n_th3 = {a: sum(s["theta", 3].event for s in v) for a, v in sc.items()}
    W = PILOT_W
    res = {"W": W, "n_sustained": n_sus, "n_theta3": n_th3, "why": [], "arms": {}}
    # -- our code is broken if
    broken = []
    if n_sus["noop"][1] > 0:
        broken.append(f"noop crosses at W=1 in {n_sus['noop'][1]} run(s) (card 0/2000)")
    if n_sus["K-lr0"][W] >= 2:
        broken.append(f"K-lr0 crosses at W={W} in {n_sus['K-lr0'][W]} of {len(seeds)}")
    for a in PILOT_LEARNERS:
        c = sum(_crash(r) for r in by[a])
        if c >= 2:
            broken.append(f"{a}: {c} runs crashed (rc != 0, not a cut)")
    if broken:
        res.update(verdict="VOID — broken", why=broken)
        return res
    # -- floor clause: in-pilot floors at W must show <= floor(U)
    moved = {f: n_sus[f][W] for f in ("noop", "held8", "K-lr0") if n_sus[f][W] > PILOT_FLOOR_MAX}
    for f, n in moved.items():
        res["why"].append(f"floor moved: {f} crosses at W={W} in {n} > floor(U)={PILOT_FLOOR_MAX}")
    # -- noop's median life (SLIDES)
    noop_s = [l["seconds"] for r in by["noop"] for l in r["lives"] if l["cause"] != "unfinished"]
    noop_med = st.median(noop_s) if noop_s else float("nan")
    res["noop_median_life_s"] = noop_med
    for a in PILOT_READ_ARMS:
        floor = PILOT_NAMED[a]
        if "noop" in moved or floor in moved:
            res["arms"][a] = "VOID — floor moved"
            continue
        if sum(_decisions(r) < PILOT_MIN_DEC for r in by[a]) >= 2:
            res["arms"][a] = "INCOMPLETE"
            continue
        if n_sus[a][W] >= PILOT_NEED:
            res["arms"][a] = "SUSTAINED"
            continue
        fth = n_th3[floor] if a not in ("K", "KP") else max(n_th3["K-lr0"], n_th3["held8"])
        if n_th3[a] >= PILOT_NEED:
            res["arms"][a] = ("EATS BUT STARVES" if fth <= 1
                              else "VOID — θ too easy (floor crosses θ in "
                                   f"{fth} of {len(seeds)})")
            continue
        slid = 0
        for r in by[a]:
            x = R.run_stats(r)
            e, l_ = x["early"], x["late"]
            if (e and l_ and l_["fed"] < e["fed"] and l_["unfed_s"] == l_["unfed_s"]
                    and noop_med == noop_med and l_["unfed_s"] >= 0.9 * noop_med):
                slid += 1
        res.setdefault("slides_runs", {})[a] = slid
        res["arms"][a] = "SLIDES" if slid >= PILOT_NEED else "NEAR RANDOM"
    res["verdict"] = "READ"
    return res


def readings_md(res: dict) -> str:
    out = [f"**Readings by §3 (frozen `e2c3c378`, W = {res['W']}): {res['verdict']}**", ""]
    out += [f"- {w}" for w in res["why"]]
    for a, v in res["arms"].items():
        out.append(f"- {a}: **{v}** — sustained runs W1/2/3 "
                   f"{'/'.join(str(res['n_sustained'][a][w]) for w in (1, 2, 3))}, "
                   f"θ runs at W3 {res['n_theta3'][a]}"
                   + (f", slide-signature runs {res['slides_runs'][a]}"
                      if a in res.get("slides_runs", {}) else ""))
    out.append("- K-lr0 is a bound for K/KP, not read (§3 Named floors).")
    return "\n".join(out)


def st_mean(v) -> float:
    return float(sum(v) / len(v))


def _poisson_upper(k: int, conf: float = 0.95) -> float:
    """One-sided upper bound on a Poisson mean given k events (exact)."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (k + 1)) / 2.0)


# ---- J15 11-20 19:45: E2c's READING, transcribed before the book is complete
# (18/25 home; e2, e3 = E-random_point + the rest of E-food/E-zero still out).
# Rule E2c "Reading" + ruling 57 items 4-5 (forager_clone.py), clause by clause.
# Seams in the frozen text, each printed rather than chosen:
#  (s1) "reading SUSTAINED, or late meals/life > 0.82, on >= 4/5 seeds": does
#       "on >= 4/5" bind SUSTAINED too? Both readings are computed (STRICT =
#       sustained on >= 4/5, J11 = sustained on >= 3/5); if they differ on any
#       arm the branch is printed under both and marked SEAM.
#  (s2) "a zero-control arm that eats as well as its food arm": read as the
#       zero arm passing the SAME eats test (the stricter for crediting the
#       sense: any zero arm that eats withholds the credit).
#  (s3) none of the three branches fires (fixed E-food eats) -> printed
#       "NO BRANCH" (INBOX 17:15): the rule names no consequence; the
#       Director rules it, this code does not.
# VOID / INCOMPLETE as J11's readings: >= 2 crashed runs in an arm -> VOID;
# >= 2 runs short of PILOT_MIN_DEC -> INCOMPLETE.
E2C_NEED = 3            # Reading: "sustained on >= 3 of 5, control < 3"
E2C_LATE_BAR = 0.82     # ruling 57 item 5: K-lr0's late meals/life (pilot 2)
E2C_LATE_NEED = 4       # "... on >= 4/5 seeds"


def e2c_branch(c: dict) -> dict:
    """c = {arm: {"sus": n sustained at W=2, "late": n runs late meals/life >
    0.82, "crash": n, "short": n}} for the five E2c arms -> the reading. Pure."""
    out = {"arms": {}, "why": []}
    for a, v in c.items():
        if v["crash"] >= 2:
            out["arms"][a] = "VOID — crashed"
        elif v["short"] >= 2:
            out["arms"][a] = "INCOMPLETE"
    if out["arms"]:
        out["verdict"] = "NO READING — " + ", ".join(f"{a} {r}" for a, r in out["arms"].items())
        return out
    sus_ok = c["E-food"]["sus"] >= E2C_NEED
    ctl_ok = c["E-zero"]["sus"] < E2C_NEED
    out["pass_clause"] = ("PASS" if sus_ok and ctl_ok else
                          "VOID — control sustains" if sus_ok else "MISS")
    out["random_point"] = c["E-random_point"]["sus"]          # reported only

    def eats(v, need_sus):
        return v["sus"] >= need_sus or v["late"] >= E2C_LATE_NEED

    def branch(need_sus):
        e = {a: eats(v, need_sus) for a, v in c.items()}
        fired = []
        if e["E-food-auto"] and not e["E-food"]:
            fired.append("THE FIXED 0.1 WAS THE WALL — later B runs in his world use auto")
        if not e["E-food-auto"] and not e["E-food"]:
            fired.append("BOTH FOOD ARMS MISS — weak evidence at 300k; next question run length (forge round 1)")
        for f, z in (("E-food", "E-zero"), ("E-food-auto", "E-zero-auto")):
            if e[z]:
                fired.append(f"{z} EATS — the handed sense carries nothing; {f}'s eating not credited to it")
        return e, fired or ["NO BRANCH — the table names no consequence (INBOX 17:15); Director rules"]
    e_j11, b_j11 = branch(E2C_NEED)
    e_str, b_str = branch(E2C_LATE_NEED)
    out["eats"] = e_j11
    out["branch"] = b_j11
    if e_j11 != e_str:
        out["seam"] = {"strict_eats": e_str, "strict_branch": b_str}
    out["verdict"] = "READ"
    return out


def e2c_readings(paths=None) -> tuple:
    """The 25 registered runs -> (markdown, e2c_branch result). Raises on a
    missing run: no reading on a partial book."""
    from experiments import pilot_b_report as R
    paths = sorted(paths if paths is not None else OUT.glob("e2c_*.json"))
    runs, _ = load_pilot(paths)
    cells = e2c_cells() + e2c_auto_cells()
    miss = [R.rid(a, s) for a, s in cells if R.rid(a, s) not in runs]
    if miss:
        raise ValueError(f"no reading: {len(miss)} of {len(cells)} missing {miss}")
    c, late_v = {}, {}
    for a in tuple(E2C_ARMS) + tuple(E2C_AUTO_ARMS):
        rs = [runs[R.rid(a, s)] for s in E2C_SEEDS]
        lm = [(R.run_stats(r)["late"] or {}).get("meals", float("nan")) for r in rs]
        late_v[a] = lm
        c[a] = {"sus": sum(R.score(r)["sustained", PILOT_W].event for r in rs),
                "late": sum(x > E2C_LATE_BAR for x in lm if x == x),
                "crash": sum(_crash(r) for r in rs),
                "short": sum(_decisions(r) < PILOT_MIN_DEC for r in rs)}
    res = e2c_branch(c)
    md = [f"**E2c reading (rule E2c + ruling 57 items 4-5, W = {PILOT_W}): {res['verdict']}**", ""]
    for a, v in c.items():
        md.append(f"- {a}: sustained {v['sus']}/5, late meals/life > {E2C_LATE_BAR} on {v['late']}/5 "
                  f"({', '.join(f'{x:.2f}' for x in late_v[a])}), crashed {v['crash']}, short {v['short']}"
                  + (f" -> EATS {res['eats'][a]}" if "eats" in res else ""))
    if res["verdict"] == "READ":
        md += ["", f"- Reading clause (E-food vs E-zero): **{res['pass_clause']}**; "
               f"E-random_point sustained {res['random_point']}/5 (reported only)"]
        md += [f"- Table branch: **{b}**" for b in res["branch"]]
        if "seam" in res:
            md.append("- SEAM (s1): under STRICT (sustained on >= 4/5) -> "
                      + "; ".join(res["seam"]["strict_branch"]))
    return "\n".join(md), res


# ---- J17 (3) NEXT: THE RULE's reading, transcribed 11-21 08:30, BEFORE any
# J17 kernel is home. Rule: prereg/row128_vs_rays16.md FROZEN `98e22036`.
# Transcription seams (each the stricter reading; none can help row128):
#  (a) "late" = ruling 65's late half: FINISHED lives whose first decision is
#      >= J17_LATE_FROM (150,000); an unfinished last life is not counted (as
#      run_stats). No late life -> NaN, which wins no MWU pair and no seed pair
#      for row128 in either direction (the life cap 9,000 makes it near-
#      impossible; NaN can only cost row128). The draft's
#      last-quarter figure (run_stats "late") is printed beside, NOT read.
#  (b) MWU one-sided U = #(row128_i > other_j) + 0.5 #(ties) over 25 pairs;
#      seed pairs = same seed, strict >.
#  (c) an arm EATS iff its per-arm reading is SUSTAINED or EATS BUT STARVES
#      (§3's classes; the order's verdict row128 eats / neither / both is read
#      off the two eyed arms). K's named floor for EATS BUT STARVES is the
#      in-race bound K-lr0-row128 (held8-k0-ff-bm has no in-race twin; its
#      card U 0.64 -> floor(U) 0 is the same bar). SLIDES vs NEAR RANDOM needs
#      noop's median life, which J17 does not run -> both print NOT EATING.
#  (d) VOID clauses as drafted: blind SUSTAINED >= 3/5; K-lr0-row128 crosses
#      W=2 on >= 2/5; >= 2 crashed runs in any arm. >= 2/5 runs of an arm below
#      150k decisions -> INCOMPLETE (re-run on 926+).
J17_U_NEED = 21
J17_PAIRS_NEED = 4


def j17_late_meals(rec: dict, late_from: int = J17_LATE_FROM) -> float:
    from experiments import pilot_b_report as R
    total = sum(life["decisions"] for life in rec["lives"])
    if late_from >= total:  # no life can start there: a mis-set mark, not "no meals" (nan)
        raise ValueError(f"late_from {late_from} >= run length {total} decisions")
    t, ms = 0, []
    for life in rec["lives"]:
        if life["cause"] != "unfinished" and t >= late_from:
            ms.append(R._meals(life))
        t += life["decisions"]
    return sum(ms) / len(ms) if ms else float("nan")


def j17_capped_marks(rec: dict) -> tuple:
    """(first, last) capped life, in decisions at the life's end; None if none."""
    t, caps = 0, []
    for life in rec["lives"]:
        t += life["decisions"]
        if life["cause"] == "capped":
            caps.append(t)
    return (caps[0], caps[-1]) if caps else (None, None)


def mwu_u(xs, ys) -> float:
    return sum((x > y) + 0.5 * (x == y) for x in xs for y in ys)


def j17_readings(runs: dict) -> dict:
    """THE RULE on the 20 registered J17 runs; anything missing raises."""
    from experiments import pilot_b_report as R
    if J17_T != R.T:
        raise RuntimeError("j17_readings: pilot_b_report scores at T_RUN != J17_T")
    miss = [R.rid(a, s) for a, s in j17_cells() if R.rid(a, s) not in runs]
    if miss:
        raise ValueError(f"no reading: {len(miss)} of 20 missing {miss}")
    W, seeds = PILOT_W, J17_SEEDS
    by = {a: [runs[R.rid(a, s)] for s in seeds] for a in J17_ARMS}
    sc = {a: [R.score(r) for r in v] for a, v in by.items()}
    n_sus = {a: sum(x["sustained", W].event for x in v) for a, v in sc.items()}
    n_th3 = {a: sum(x["theta", 3].event for x in v) for a, v in sc.items()}
    late = {a: [j17_late_meals(r) for r in v] for a, v in by.items()}
    res = {"W": W, "n_sustained": n_sus, "n_theta3": n_th3, "late": late,
           "late_quarter": {a: [(R.run_stats(r)["late"] or {}).get("meals", float("nan"))
                                for r in v] for a, v in by.items()},
           "capped": {a: [j17_capped_marks(r) for r in v] for a, v in by.items()},
           "arms": {}, "why": []}
    broken = []
    if n_sus["K-row128-blind"] >= 3:
        broken.append(f"K-row128-blind SUSTAINED on {n_sus['K-row128-blind']}/5 (must fail)")
    if n_sus["K-lr0-row128"] >= 2:
        broken.append(f"K-lr0-row128 crosses W={W} on {n_sus['K-lr0-row128']}/5")
    for a, v in by.items():
        c = sum(_crash(r) for r in v)
        if c >= 2:
            broken.append(f"{a}: {c} runs crashed (rc != 0, not a cut)")
    if broken:
        res.update(verdict="VOID — broken", why=broken)
        return res
    floor_th = n_th3["K-lr0-row128"]
    for a in ("K-rays16", "K-row128", "K-row128-blind"):
        if sum(_decisions(r) < PILOT_MIN_DEC for r in by[a]) >= 2:
            res["arms"][a] = "INCOMPLETE"
        elif n_sus[a] >= PILOT_NEED:
            res["arms"][a] = "SUSTAINED"
        elif n_th3[a] >= PILOT_NEED:
            res["arms"][a] = ("EATS BUT STARVES" if floor_th <= 1 else
                              f"VOID — θ too easy (K-lr0-row128 crosses θ in {floor_th}/5)")
        else:
            res["arms"][a] = "NOT EATING (SLIDES / NEAR RANDOM: no noop arm in J17)"
    if any(v == "INCOMPLETE" for v in res["arms"].values()):
        res["verdict"] = "INCOMPLETE — re-run on seeds 926+"
        return res
    row, rays, blind = late["K-row128"], late["K-rays16"], late["K-row128-blind"]
    res["U_rays16"], res["U_blind"] = mwu_u(row, rays), mwu_u(row, blind)
    res["pairs_rays16"] = sum(x > y for x, y in zip(row, rays))
    res["pairs_blind"] = sum(x > y for x, y in zip(row, blind))
    beats_rays = res["U_rays16"] >= J17_U_NEED and res["pairs_rays16"] >= J17_PAIRS_NEED
    beats_blind = res["U_blind"] >= J17_U_NEED and res["pairs_blind"] >= J17_PAIRS_NEED
    res["clause"] = ("row128 HELPS" if beats_rays and beats_blind else
                     "width, not sight" if beats_rays else
                     "no difference — no eye effect at 300k under S")
    eats = {a: res["arms"][a] in ("SUSTAINED", "EATS BUT STARVES") for a in ("K-row128", "K-rays16")}
    res["verdict"] = ("both eat" if all(eats.values()) else
                      "row128 eats" if eats["K-row128"] else
                      "rays16 eats (row128 does not)" if eats["K-rays16"] else
                      "neither eats")
    return res


def j17_md(res: dict) -> str:
    def f(v):
        return ", ".join("nan" if x != x else f"{x:.2f}" for x in v)

    def k(m):
        return "—" if m is None else f"{m // 1000}k"
    out = [f"**J17 reading (row128_vs_rays16.md FROZEN `98e22036`, W = {res['W']}): "
           f"{res['verdict']}**", ""]
    out += [f"- {w}" for w in res["why"]]
    if "clause" in res:
        out.append(f"- Primary (late-half meals/life, row128 over): rays16 U {res['U_rays16']:g}/25, "
                   f"pairs {res['pairs_rays16']}/5; blind U {res['U_blind']:g}/25, pairs "
                   f"{res['pairs_blind']}/5 (need U >= {J17_U_NEED} AND >= {J17_PAIRS_NEED}/5) "
                   f"-> **{res['clause']}**")
    out += ["", "| arm | reading | sustained W2 | θ W3 | late-half meals/life (read) | "
            "last-quarter (beside, not read) | first / last capped life |", "|---|---|---|---|---|---|---|"]
    for a in J17_ARMS:
        caps = "; ".join(f"{k(x)}/{k(y)}" for x, y in res["capped"][a])
        out.append(f"| {a} | {res['arms'].get(a, 'bound (not read)')} | {res['n_sustained'][a]}/5 "
                   f"| {res['n_theta3'][a]}/5 | {f(res['late'][a])} | {f(res['late_quarter'][a])} | {caps} |")
    return "\n".join(out)


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cell", "kernel", "card", "rule-inputs", "pilot-cell", "pilot-kernel", "table", "readings", "e2c-kernel", "e2c-table", "e2c-readings", "j17-kernel", "j17-table", "j17-hunt-kernel"])
    p.add_argument("--arm", default="noop-k1")
    p.add_argument("--seed", type=int, default=601)
    p.add_argument("--lives", type=int, default=CARD_LIVES)
    p.add_argument("--out", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--decisions", type=int, default=PILOT_T)
    p.add_argument("--partial", default=None)
    p.add_argument("--food-floor-only", action="store_true")
    p.add_argument("--eye-frame", choices=["world", "body"], default="world")
    p.add_argument("--motor-frame", choices=["world", "body"], default="world")
    p.add_argument("--native-substep", action="store_true")
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
                             partial_path=a.partial, eye_frame=a.eye_frame,
                             motor_frame=a.motor_frame,
                             native_substep=a.native_substep)
        Path(a.out).write_text(json.dumps(rec) + "\n")
    elif a.cmd == "pilot-kernel":
        pilot_kernel(a.name)
    elif a.cmd == "e2c-kernel":
        e2c_kernel(a.name)
    elif a.cmd == "j17-kernel":
        j17_kernel(a.name)
    elif a.cmd == "j17-hunt-kernel":
        j17_hunt_kernel(a.name or "h0")
    elif a.cmd == "j17-table":
        runs, _ = load_pilot(sorted(OUT.glob("j17_r*.json")))
        try:
            print(j17_md(j17_readings(runs)))
        except ValueError as e:
            raise SystemExit(f"j17-table: PARTIAL — {e}")
    elif a.cmd == "e2c-table":
        md, missing = e2c_table()
        print(md)
        raise SystemExit(3 if missing else 0)
    elif a.cmd == "e2c-readings":
        try:
            md, _ = e2c_readings()
        except ValueError as e:
            raise SystemExit(f"e2c-readings: PARTIAL — {e}")
        print(md)
    elif a.cmd == "table":
        seeds = (PILOT_RULE or {}).get("seeds")
        if not seeds:
            raise SystemExit("table: no frozen rule (PILOT_RULE) -> no registered seeds")
        md, missing = pilot_table(sorted(OUT.glob("pilot_p*.json")), seeds)
        print(md)
        raise SystemExit(3 if missing else 0)
    elif a.cmd == "readings":
        md, missing = pilot_table(sorted(OUT.glob("pilot_p*.json")), PILOT_RULE["seeds"])
        if missing:
            raise SystemExit(f"readings: PARTIAL, {len(missing)} runs missing -> no reading")
        runs, _ = load_pilot(sorted(OUT.glob("pilot_p*.json")))
        print(readings_md(pilot_readings(runs, PILOT_RULE["seeds"])))
    elif a.cmd == "rule-inputs":
        print(rule_inputs(sorted(OUT.glob(a.glob))))
    else:
        print(card_table(sorted(OUT.glob(a.glob))))


if __name__ == "__main__":
    main(sys.argv[1:])
