"""LAB — FED ONCE, THEN WEANED: ONE MEAL AT A PARENT'S HAND NAMES FOOD FOR A
NEWBORN, AND THE NAME STILL SERVES HIM WHEN HE IS SET DOWN FAR AWAY (no ledger
row; a pilot. Seat: Curriculum, VACANT (`CR.01` parked, ruling 20); PLAN part
"His world, needs and lessons", right-hand column "the entry settings at which
his random movement finds food", and PRIORS §3 last-10% "provisioning that
fades in a persistent life". Follow-up to `live_eye.py`, ALIVE 11-22.)

WHY. `live_eye` showed the hindsight table lights itself from EMPTY after his
first meal — but that meal comes from babble: decision 176 to 6,424 on eight
fresh worlds (median ~950), and in two of eight only after he had starved
once. "The wait is the world's, not the eye's." Every mammal's first meal is
GIVEN. The lab's two weaning bets (11-12/13) died because a learner fed near
his birthplace learned eating-WHERE-born. The table learns WHAT, not where —
that is the new reason to try weaning again.

RESULT (2026-11-24) — **DEAD AT THE PILOT; the judge worlds 1501-1508 were
NOT lived** (a dev read this far from the bar does not buy a 45 CPU-minute
judge). Dev worlds 1101-1106, the arms as below, 3,000 decisions each:
  FIRST own meal   born 2720/76/3000/404/703/3000   (median 1,712)
                   fed  3000/619/3000/507/1348/729  (median 1,038)
                   fed-misnamed median 3,000; nursery0 median 1,812
  fed ahead of born on 1 of 6 worlds (bar 5); median not half; meals 35 vs
  32; deaths 1 vs 3. The control behaved (misnamed: 16 meals, ahead 1/6).
  WHY IT DIED — the NAME, not the weaning: of 15 meals taken at the parent's
  hand the nearest ray wore a food's texture at 8 (in the wild: 74%). Set
  down 0.44 m beside a food he babbles into it side-on, and the nearest ray
  is something else. One wrong name with the same count as a right one wins
  whenever it is nearer (a wall always is) and costs him PATIENCE = 400
  decisions. Where the hand named nothing right (1101) he never ate.
  The dial's own nursery (d = 0) is no better: fed in 4 of 6, first imprint
  on food 1 of 4, ahead of born 3 of 6.
  First version (one food, one meal, no give-up rule), same worlds: fed
  ahead 1 of 6, median 2,352 vs 1,712, meals 28 vs 36; and one fed life
  steered 2,983 decisions straight at a real food it never reached (wedged,
  never "near", so never unlearned) — hence the give-up rule below, which
  fired 2-3 times an arm and did not change a FIRST for `born`.
  Unrun: a label that does not depend on how he arrived (the texture that
  LOOMED over the last second, not the nearest ray at the bite); the hand
  feeding him the way he will later eat (walking him INTO the food).

THE BET. A newborn set down beside a food once (the parent's hand), allowed
his one meal, then picked up and set down anywhere in the same world with no
further help, finds his first food BY HIMSELF in less than half the decisions
a newborn born anywhere needs — feeding time charged to him.

PRIOR (SYSTEM.md 3a; PRIORS §3 read today). Avalon: food 0.5 m in front at
the easiest setting, success-gated dial 0.716 vs 0.001 at 5x10^7 steps;
Ecological RL: helpful starting conditions that wear off beat reward shaping
(0% -> 70%, an 8x8 grid, minutes on a CPU, a learner); "an adult's help is a
world setting tied to d, absent part of every life". All with learners over
10^5-10^7 steps; none at ONE provisioned meal, and "no ablation of food
distance alone exists in any survival world" (§3, not verified). Our budget
reaches the published idea only because the thing taught is a table, not a
policy (Rescorla & Wagner 1972 sizes; from memory). Social transmission of
food identity to the young at one or a few exposures is Galef's rat work
(from memory, not re-read).

WORLD. `taste_eye.make_env` unchanged (world K, entry rung d = 0.75, the
128-ray body-frame eye, mujoco 3.17.0 in `/data/venvs/jack-std`). NOT a
learner (ruling 58): `live_eye`'s reflex and table, unlearning ON, as the
ruler. No world file is touched: the parent's hand is `W0.respawn(at=...)`,
the world's own call.

ARMS (same world, same seed, same babble stream, 3,000 decisions each — about
one unfed lifetime)
  born          the holder: `live_eye`'s `live` — born where the entry rung
                puts him, table empty, first meal by babble.
  fed           decision 0: set at the legal cell nearest a floor food (the
                foods where the world laid them; 0.44 m on dev) and left to
                babble (no steering in the hand). When he has eaten it, or
                after OFFER decisions unfed, the parent sets him beside the
                next food he has not eaten. When he has eaten each of the
                world's floor foods once (or at FEED_CAP decisions) he is
                WEANED: the world's own respawn (a new uniform spot, the
                foods laid out anew, needs drawn anew). Table kept.
  fed-misnamed  MUST FAIL: the same, but each FED meal imprints the texture
                on a random ray that does not wear the nearest ray's texture
                (he was fed; the mouth did not name it). Later meals honest.
  nursery0      beside, gating nothing: the dial's own easy end — the same
                seed's world at d = 0 (5 foods, full-bar meals) until his
                first meal or FEED_CAP, then the d = 0.75 world, table kept.
                Two of the five nursery foods are not food at d = 0.75.

THE REFLEX (every arm, so `born` is `live_eye`'s `live` plus this): steering
at a tabled texture whose ray has not shortened by STUCK_EPS in STUCK
decisions, and is not near, he gives up and babbles UNSTICK decisions.

MEASURE, per world and arm: FIRST = the decision (of his existence; feeding
time charged) of his first meal that nobody set him beside — for `born` any
meal, for the others the first meal AFTER weaning. Censored at 3,000.

RULE — DECLARED BEFORE ANY JUDGE WORLD IS LIVED (judge worlds 1501..1508,
never lived by anything; pins below set on dev worlds 1101-1106 only).
  VOID if `fed` is fed (eats before FEED_CAP) in fewer than 6 of 8 worlds.
  ALIVE iff ALL of: (a) fed's FIRST < born's on >= 6 of 8 worlds;
  (b) median FIRST of fed <= half of born's median; (c) fed's FIRST <
  fed-misnamed's on >= 6 of 8 worlds; (d) fed's meals after weaning, in
  total, >= born's meals in total. Else DEAD. No extension.
  The control is valid iff fed-misnamed itself does NOT meet (a) and (b)
  against born; if it does, the read is "a full belly and a near start, not
  the name" and the bet is DEAD whatever fed scored.
  Reported beside, gating nothing: nursery0's FIRST and meals; deaths;
  whether the fed imprint was on food (simulator truth); decisions spent
  being fed.
  Cannot say, whatever happens: that a LEARNER is helped (J17 / ruling 84);
  anything in a world whose foods differ between nursery and range beyond
  the nursery0 row; a camera eye.

    python -m experiments.fed_once dev [workers] [seeds...]
    python -m experiments.fed_once judge [workers]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import taste_eye as te  # noqa: E402
from experiments.live_eye import NEAR, PATIENCE  # noqa: E402

N = te.N
STUCK = 50         # steering decisions (10 s) with the target no nearer -> give up
STUCK_EPS = 0.01   # "nearer" = the target ray shortens by 8 cm (of the 8 m range)
UNSTICK = 16       # decisions of babble (two Held-8 bouts) after giving up
OUT = Path("/data/jack-logs/fed_once")
ARMS = ("born", "fed", "fed-misnamed", "nursery0")
JUDGE_SEEDS = tuple(range(1501, 1509))
DEV_SEEDS = (1101, 1102, 1103, 1104, 1105, 1106)
BUDGET = 3000      # decisions of existence per cell (600 s ~ one unfed life)
FEED_CAP = 450     # decisions at the parent's hand before he is weaned unfed
OFFER = 50         # unfed in the hand this long, he is set beside the food again;
#                    dev 11-24, worlds 1101-1106, feeding phase only: set down once
#                    and left to babble he is fed by 300 in 4 of 6 (1 / 10 / 44 / 59 /
#                    498 / never); offered again every 50 he is fed in 6 of 6
#                    (1 / 10 / 44 / 51 / 52 / 101).


def _nursery_env(seed: int):
    """The same seed's world at the dial's easy end (d = 0)."""
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b import LIFE_CAP, REWARD, PilotLives
    from experiments.sim_speed_kcpu import _compat_multiray
    _compat_multiray()
    env = JackEnv(seed=seed, difficulty=0.0, reward=REWARD,
                  random_needs=NEEDS_S_TRP, random_layout=True,
                  lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                  move_cost_dial=True, food_floor_only=True, eye="row128",
                  eye_frame="body", motor_frame="body", native_substep=True)
    return PilotLives(env, cap=LIFE_CAP)


def _food_names(e) -> list:
    return [n for n in e.floor_foods if n in e.w.drives._food]


def _hand_cell(e, name: str) -> tuple:
    """The legal cell nearest floor food `name` from which his body touches
    nothing but ground (tested as `JackEnv._layout_spawn` tests it)."""
    w = e.w
    legal = w.legal_spawns()
    food = np.array(w.data.xpos[w.model.body(name).id][:2])
    d = np.linalg.norm(legal - food[None, :], axis=1)
    for k in np.argsort(d):
        x, y = float(legal[k][0]), float(legal[k][1])
        w._place(x, y)
        w.mujoco.mj_forward(w.model, w.data)
        if not w._penetrating():
            return x, y, float(d[k])
    raise RuntimeError("no legal cell beside a food")


def _set_down(e, at=None):
    """A new body in the same world (the world's own respawn; `at` = the
    parent's hand), needs drawn as at any birth. Returns his first look."""
    e.w.respawn(at=at)
    if e.random_needs is not None:
        e._draw_needs()
    e._pending_obs = None
    e._truncated = False
    return e._obs(e.w.observe())


def cell(arm: str, seed: int, budget: int = BUDGET) -> dict:
    from experiments.world_card import Held
    ft = te.food_tex(seed)
    ft = ft[~np.isnan(ft)]

    def is_food(v) -> bool:             # simulator truth: a diagnostic only
        return bool((np.abs(ft - v) < 0.006).any())
    pol = Held(seed, 8)
    rng = np.random.RandomState(seed * 7919 + 13)
    w = _nursery_env(seed) if arm == "nursery0" else te.make_env(seed)
    o, info = w.reset(seed=seed)
    pol.new_life()
    hand_d, shown, offer_n, fed_n = None, [], 0, 0
    if arm in ("fed", "fed-misnamed"):
        shown = _food_names(w.env)      # the foods he has not been fed yet
        x, y, hand_d = _hand_cell(w.env, shown[0])
        o = _set_down(w.env, (x, y))
    weaned = arm == "born"              # born is never in anyone's hand
    fr_tex, fr_best, fr_n, babble_left, gave_up = None, math.inf, 0, 0, 0
    table: dict[float, int] = {}
    unfed: dict[float, int] = {}
    last = int(w.env._meals())
    ate0 = dict(w.env._info()["ate_total"])
    first = fed_at = fed_on_food = None
    meals = deaths = steered = ext_n = 0
    t0 = time.process_time()
    for i in range(budget):
        done = (fed_at is not None) if arm == "nursery0" else (hand_d is not None and not shown)
        if not weaned and (done or i >= FEED_CAP):
            if arm == "nursery0":
                w = te.make_env(seed)
                o, info = w.reset(seed=seed)
            else:
                o = _set_down(w.env)
            pol.new_life()
            last, weaned = int(w.env._meals()), True
            unfed = {}
        elif not weaned and hand_d is not None and offer_n >= OFFER:
            shown.append(shown.pop(0))          # the parent offers the next food
            x, y, _d = _hand_cell(w.env, shown[0])
            o = _set_down(w.env, (x, y))
            pol.new_life()
            last, offer_n = int(w.env._meals()), 0
        offer_n += 1
        dist = o[0:2 * N:2].astype(np.float32)
        tex = o[1:2 * N:2].astype(np.float32)
        a = pol.act(o)                  # always drawn: one babble stream
        target, near = None, False
        if babble_left > 0:
            babble_left -= 1
        elif table and weaned:
            sc = np.zeros(N, np.float32)
            for v, c_ in table.items():
                sc = np.where(np.abs(tex - v) < 0.006, float(c_), sc)
            if sc.max() > 0:
                k = int((sc - dist * 0.5).argmax())
                a = np.zeros(8, np.float32)
                a[6], a[7] = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
                steered += 1
                target = min(table, key=lambda v: abs(float(tex[k]) - v))
                near = bool(dist[k] < NEAR)
                if near or target != fr_tex or dist[k] < fr_best - STUCK_EPS:
                    fr_tex, fr_best, fr_n = target, float(dist[k]), 0
                else:
                    fr_n += 1
                    if fr_n >= STUCK:   # it gets no nearer: give up for a while
                        babble_left, fr_tex, fr_best, fr_n = UNSTICK, None, math.inf, 0
                        gave_up += 1
        c = float(tex[dist.argmin()])
        o, _r, term, trunc, info = w.step(a)
        ate = int(info["meals"]) > last
        last = int(info["meals"])
        if ate:
            if not weaned:
                fed_at = i if fed_at is None else fed_at
                fed_n += 1
                if hand_d is not None:
                    ate_now = [n for n in shown if info["ate_total"][n] > ate0.get(n, 0)]
                    shown = [n for n in shown if n not in ate_now]
                    ate0 = dict(info["ate_total"])
                    if shown:
                        x, y, _d = _hand_cell(w.env, shown[0])
                        o = _set_down(w.env, (x, y))
                        pol.new_life()
                        offer_n = 0
                if arm == "fed-misnamed":
                    other = tex[np.abs(tex - c) > 0.006]
                    c = float(other[rng.randint(len(other))]) if len(other) else c
                fed_on_food = (fed_on_food or 0) + int(is_food(round(c, 2)))
            else:
                meals += 1
                if first is None:
                    first = i
            v = round(c, 2)
            table[v] = table.get(v, 0) + 1
            unfed[v] = 0
        elif target is not None and near:
            unfed[target] = unfed.get(target, 0) + 1
            if unfed[target] > PATIENCE:
                del table[target]
                unfed[target] = 0
                ext_n += 1
        if term or trunc:
            o, info = w.reset()
            pol.new_life()
            deaths += 1
            last = int(info["meals"])
    return {"arm": arm, "seed": seed, "first": budget if first is None else first,
            "meals": meals, "fed_at": fed_at, "fed_meals": fed_n,
            "fed_on_food": fed_on_food, "gave_up": gave_up,
            "hand_d": None if hand_d is None else round(hand_d, 2),
            "deaths": deaths, "steered": steered, "extinctions": ext_n,
            "table": {str(k): v for k, v in sorted(table.items())},
            "cpu_s": round(time.process_time() - t0, 1)}


def _cell(args):
    r = cell(*args)
    print(json.dumps(r), file=sys.stderr, flush=True)
    return r


def judge_rows(rows, seeds, budget=BUDGET) -> dict:
    R = {(r["arm"], r["seed"]): r for r in rows}
    F = {a: [R[(a, s)]["first"] for s in seeds] for a in ARMS}
    M = {a: sum(R[(a, s)]["meals"] for s in seeds) for a in ARMS}
    n, need = len(seeds), math.ceil(0.75 * len(seeds))
    fed_n = sum(R[("fed", s)]["fed_at"] is not None for s in seeds)
    med = {a: float(np.median(F[a])) for a in ARMS}

    def beats(a, b):
        return sum(x < y for x, y in zip(F[a], F[b]))
    cl = {f"(a) fed first < born on >= {need}": beats("fed", "born") >= need,
          "(b) median fed <= half born": med["fed"] <= 0.5 * med["born"],
          f"(c) fed first < misnamed on >= {need}": beats("fed", "fed-misnamed") >= need,
          "(d) fed meals >= born meals": M["fed"] >= M["born"]}
    ctl_passes = (beats("fed-misnamed", "born") >= need
                  and med["fed-misnamed"] <= 0.5 * med["born"])
    void = [] if fed_n >= need else [f"fed in {fed_n} of {n} worlds (< {need})"]
    verdict = ("VOID" if void else "DEAD (control passes: belly and place, not the name)"
               if ctl_passes else "ALIVE" if all(cl.values()) else "DEAD")
    return {"verdict": verdict, "void": void, "clauses": cl,
            "control_passes": ctl_passes, "first": F, "median_first": med,
            "meals": M, "ahead_of_born": {a: beats(a, "born") for a in ARMS[1:]},
            "fed_in": fed_n,
            "fed_at": {a: [R[(a, s)]["fed_at"] for s in seeds] for a in ARMS[1:]},
            "fed_on_food": {a: [R[(a, s)]["fed_on_food"] for s in seeds] for a in ARMS[1:]},
            "deaths": {a: sum(R[(a, s)]["deaths"] for s in seeds) for a in ARMS},
            "fed_meals": {a: [R[(a, s)]["fed_meals"] for s in seeds] for a in ARMS[1:]},
            "gave_up": {a: sum(R[(a, s)]["gave_up"] for s in seeds) for a in ARMS},
            "pins": {"BUDGET": budget, "FEED_CAP": FEED_CAP, "OFFER": OFFER,
                     "STUCK": STUCK, "STUCK_EPS": STUCK_EPS, "UNSTICK": UNSTICK, "NEAR": NEAR,
                     "PATIENCE": PATIENCE},
            "cpu_s": round(sum(r["cpu_s"] for r in rows), 1)}


def run(seeds, workers=1, budget=BUDGET) -> dict:
    jobs = [(a, s, budget) for s in seeds for a in ARMS]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as p:
            rows = p.map(_cell, jobs, chunksize=1)
    else:
        rows = [_cell(j) for j in jobs]
    r = judge_rows(rows, seeds, budget)
    r["rows"] = rows
    return r


def main(argv):
    cmd = argv[0] if argv else "dev"
    workers = int(argv[1]) if len(argv) > 1 else 1
    if cmd == "dev":
        seeds = tuple(int(s) for s in argv[2:]) or DEV_SEEDS[:2]
        r = run(seeds, workers)
    elif cmd == "judge":
        r = run(JUDGE_SEEDS, workers)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "judge.json").write_text(json.dumps(r, indent=1))
    else:
        raise SystemExit(__doc__)
    r.pop("rows", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
