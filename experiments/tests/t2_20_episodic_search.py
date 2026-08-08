"""T2.20 — the episodic store helps the NEXT episode, in the body, in the world.

GOAL.md: "He remembers the ladder." ME.1-ME.10 proved the diary works as a
data structure; this spec asks whether it changes BEHAVIOUR: an agent that
lived episode 1 and wrote down what it saw must find a hidden object in later
episodes faster than an agent searching from scratch.

The task, end to end in the playground (same room, walls and fixtures as
PG.1-PG.5; same collision-free rover as PG.4):

  Episode 1 (shared)   The rover explores by random waypoints. Four uniquely
                       coloured objects sit at random, well-separated spots.
                       Passing within SIGHT_R of one records a "saw" event in
                       EpisodicMemory — plain text plus the perceived position
                       in event.meta — exactly the record/recall contract
                       every ME spec tested. The store also carries 300
                       distractor events (templated chatter, 200 older and
                       100 NEWER than the sightings) so retrieval has to find
                       the right sighting, not the only event.

  Episodes 2..N        Rover resets to the origin and is asked for a specific
                       object ("where did i see the crimson apple"). The
                       MEMORY agent recalls top-1, drives to the remembered
                       position, and only falls back to waypoint exploration
                       if nothing is there. The MEMORYLESS null explores from
                       scratch. Identical drive dynamics, identical caps — the
                       ONLY difference is what the store contributes.

  search_time_ratio = mean(memory steps) / mean(memoryless steps), per seed.

NULL (must lose): the memoryless explorer — same waypoint policy the memory
agent falls back to, so the comparison is store vs no store, not policy vs
policy.

CONTROLS (must fail):
  shuffled   sighting positions deranged across events: retrieval returns the
             RIGHT text with the WRONG place. If the ratio does not climb
             back toward the null, the speedup never depended on the store's
             content and the experiment is measuring something else.
  recency    "goldfish" retrieval: always the most recent sighting. Targets
             are deliberately never the last-sighted object, so a memory that
             cannot address by content must drive to the wrong spot first.

Honesty notes, pre-registered:
  - Chatter vocabulary is DISJOINT from the hidden-object vocabulary, so
    similarity separates cleanly; retrieval-under-collision is ME.5's claim,
    not this one. This spec's claim is the INTEGRATION: diary -> behaviour.
  - Sighting stores the object's position at the moment it is within SIGHT_R
    (1 m) of the rover — perception is abstracted to proximity, as in PG.4's
    dwell radius. Front-back audio, vision noise etc. are other specs.
  - Objects do not move between episodes; "the world is stable, the agent
    forgets" is exactly the regime where episodic memory should pay.
  - Timeouts count as CAP_Q steps, so a memory agent chasing a wrong
    position cannot look good by failing fast.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# ── pre-registered constants ─────────────────────────────────────────────
K_OBJ = 4                   # hidden objects in the room
SIGHT_R = 1.0               # m; seeing == being within this of an object
FOUND_R = 1.0               # m; finding == same radius, no double standard
CAP_EP1 = 3000              # decisions; episode-1 exploration budget
CAP_Q = 1500                # decisions; per-query search budget
N_QUERIES = 3               # distinct targets asked about per seed
R_REPS = 8                  # search episodes per target per arm: search time
                            # is heavy-tailed, 3 episodes/arm let one lucky
                            # find swing a mean 2x (seen in the seed-0 pilot)
SUBSTEPS = 40               # sim dt 0.005 x 40 = 0.2 s per decision
SPEED = 2.0                 # m/s, actuator ctrlrange limit
WAYPOINT_DONE = 0.4         # m; switch waypoint when this close
WAYPOINT_MAX_AGE = 100      # decisions; or when stuck this long

RATIO_MAX = 0.5             # memory must at least HALVE search time
RETRIEVAL_MIN = 0.8         # top-1 recall must be the true target's sighting
NULL_SUCCESS_MIN = 0.8      # the null must actually find objects within CAP_Q
CONTROL_RATIO_MIN = 0.75    # shuffled and recency must climb back toward null

N_CHATTER_PRE, N_CHATTER_POST = 200, 100

# Hidden-object vocabulary — kept disjoint from the chatter pools below.
OBJ_WORDS = [("crimson", "apple"), ("teal", "lantern"),
             ("amber", "compass"), ("violet", "whistle")]

CH_OBJECTS = ["kettle", "hammer", "bucket", "rope", "mirror", "anchor",
              "basket", "kite", "shovel", "candle", "bell", "net"]
CH_PLACES = ["pond", "ramp", "platform", "meadow", "shed", "gate",
             "bridge", "cellar", "orchard", "quarry"]
CH_COLOURS = ["copper", "olive", "ivory", "slate", "coral", "bronze"]
CH_ACTIONS = ["carried", "dropped", "painted", "repaired", "buried",
              "balanced", "measured", "cleaned", "stacked", "traded"]
CH_SPEAKERS = ["ada", "bruno", "chika", "jack"]

T0 = 1_000_000.0

# Shared per-seed world + episode-1 life, filled by _experiment and reused by
# _control so both judge the SAME store (run_spec runs experiments first).
_CACHE: dict = {}


def _build(obj_pos):
    sys.path.insert(0, str(REPO))
    import mujoco
    from playground import PlaygroundParams, build_mjcf

    xml = build_mjcf(PlaygroundParams(seed=0, n_objects=0))
    hidden = "".join(
        f'<geom name="hidden_{i}" type="sphere" pos="{x} {y} 0.1" size="0.12" '
        f'contype="0" conaffinity="0" rgba="0.8 0.2 0.2 1"/>'
        for i, (x, y) in enumerate(obj_pos))
    rover = (
        '<body name="rover" pos="0 0 0.15">'
        '<joint name="rover_x" type="slide" axis="1 0 0" range="-5.5 5.5" damping="1"/>'
        '<joint name="rover_y" type="slide" axis="0 1 0" range="-5.5 5.5" damping="1"/>'
        '<geom name="rover" type="sphere" size="0.15" mass="1" contype="0" '
        'conaffinity="0" rgba="0.9 0.6 0.1 1"/></body>')
    actuators = (
        '<actuator>'
        '<velocity name="vx" joint="rover_x" kv="200" ctrlrange="-2 2" forcerange="-400 400"/>'
        '<velocity name="vy" joint="rover_y" kv="200" ctrlrange="-2 2" forcerange="-400 400"/>'
        '</actuator>')
    xml = xml.replace("</worldbody>", hidden + rover + "\n  </worldbody>")
    xml = xml.replace("</mujoco>", actuators + "\n</mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, (model.actuator("vx").id, model.actuator("vy").id)


def _place_objects(rng):
    """Uniform in the room, pairwise >= 3 m apart, >= 1.5 m from the start."""
    pos = []
    while len(pos) < K_OBJ:
        x, y = rng.uniform(-4.3, 4.3), rng.uniform(-4.3, 4.3)
        if math.hypot(x, y) < 1.5:
            continue
        if all(math.hypot(x - px, y - py) >= 3.0 for px, py in pos):
            pos.append((x, y))
    return pos


class _Explorer:
    """Random-waypoint wanderer — the search policy of EVERY arm.

    The memory agent uses it only as fallback; the null lives on it. Keeping
    the policy identical is what makes the comparison store vs no-store.
    """

    def __init__(self, rng):
        self.rng, self.wp, self.age = rng, None, 0

    def target(self, x, y):
        if (self.wp is None or self.age >= WAYPOINT_MAX_AGE
                or math.hypot(x - self.wp[0], y - self.wp[1]) < WAYPOINT_DONE):
            self.wp = (self.rng.uniform(-4.8, 4.8), self.rng.uniform(-4.8, 4.8))
            self.age = 0
        self.age += 1
        return self.wp


def _drive(data, ax, ay, tx, ty):
    x, y = float(data.qpos[-2]), float(data.qpos[-1])
    data.ctrl[ax] = max(-SPEED, min(SPEED, 3.0 * (tx - x)))
    data.ctrl[ay] = max(-SPEED, min(SPEED, 3.0 * (ty - y)))


def _reset(model, data):
    import mujoco
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def _search(model, data, acts, target_pos, guess_pos, rng, cap=CAP_Q):
    """One search episode. Returns (steps, found). Timeout counts full cap."""
    import mujoco
    ax, ay = acts
    _reset(model, data)
    explorer = _Explorer(rng)
    guess = tuple(guess_pos) if guess_pos is not None else None
    for t in range(cap):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        if math.hypot(x - target_pos[0], y - target_pos[1]) < FOUND_R:
            return t, True
        if guess is not None:
            if math.hypot(x - guess[0], y - guess[1]) < WAYPOINT_DONE:
                guess = None            # arrived, nothing here -> explore
            else:
                _drive(data, ax, ay, *guess)
        if guess is None:
            _drive(data, ax, ay, *explorer.target(x, y))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
    return cap, False


def _quadrant(x, y):
    return ("north" if y >= 0 else "south") + ("east" if x >= 0 else "west")


def _chatter(mem, rng, n, t_start):
    for i in range(n):
        sp = rng.choice(CH_SPEAKERS)
        ch = ("said" if sp == "jack" and rng.random() < 0.5
              else "did" if sp == "jack" else "heard")
        mem.record(ch, sp,
                   f"{sp} {rng.choice(CH_ACTIONS)} the {rng.choice(CH_COLOURS)} "
                   f"{rng.choice(CH_OBJECTS)} near the {rng.choice(CH_PLACES)}",
                   importance=rng.uniform(0.5, 5.0), t=t_start + i * 60.0)
    return t_start + n * 60.0


def _live_episode_one(seed: int):
    """Build the world, explore, write the diary. Cached per seed."""
    import random
    import numpy as np
    import mujoco
    from EpisodicMemory import EpisodicMemory

    py_rng = random.Random(seed * 1009 + 1)
    obj_pos = _place_objects(py_rng)
    model, data, acts = _build(obj_pos)
    ax, ay = acts

    mem = EpisodicMemory(path=Path(tempfile.mkdtemp()) / "t220_life.jsonl")
    t = _chatter(mem, py_rng, N_CHATTER_PRE, T0)

    _reset(model, data)
    explorer = _Explorer(np.random.RandomState(seed * 7919 + 3))
    sighted = {}                        # obj index -> sighting Event
    for step in range(CAP_EP1):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        for i, (ox, oy) in enumerate(obj_pos):
            if i not in sighted and math.hypot(x - ox, y - oy) < SIGHT_R:
                colour, obj = OBJ_WORDS[i]
                sighted[i] = mem.record(
                    "saw", "jack",
                    f"jack saw the {colour} {obj} in the {_quadrant(ox, oy)} "
                    "part of the room",
                    importance=3.0, t=t + step * 0.2,
                    meta={"pos": [ox, oy]})
        if len(sighted) == K_OBJ:
            break
        _drive(data, ax, ay, *explorer.target(x, y))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    now = _chatter(mem, py_rng, N_CHATTER_POST, t + CAP_EP1 * 0.2) + 60.0

    if len(sighted) < 2:
        raise RuntimeError(f"episode 1 sighted only {len(sighted)}/{K_OBJ} "
                           "objects — exploration budget too small to test memory")

    # Targets: never the LAST-sighted object, so recency-only retrieval is
    # wrong by construction, not by luck.
    last = max(sighted, key=lambda i: sighted[i].t)
    candidates = [i for i in sighted if i != last]
    targets = py_rng.sample(candidates, min(N_QUERIES, len(candidates)))

    return {"obj_pos": obj_pos, "mem": mem, "sighted": sighted, "now": now,
            "targets": targets, "last": last,
            "ep1_sighted_frac": len(sighted) / K_OBJ}


def _arm_steps(seed: int, world, model, data, acts, salt: int,
               guess_for) -> list:
    """R_REPS search episodes per target; guess_for(i) -> pos hint or None."""
    import numpy as np
    out = []
    for k, i in enumerate(world["targets"]):
        guess = guess_for(i)
        for r in range(R_REPS):
            rng = np.random.RandomState(seed * 33301 + salt + k * 101 + r)
            steps, _ = _search(model, data, acts, world["obj_pos"][i], guess, rng)
            out.append(steps)
    return out


def _experiment(seed: int) -> dict:
    world = _live_episode_one(seed)
    _CACHE[seed] = world
    model, data, acts = _build(world["obj_pos"])
    mem, now = world["mem"], world["now"]

    null_steps = _arm_steps(seed, world, model, data, acts, 500, lambda i: None)
    world["null_mean"] = sum(null_steps) / len(null_steps)

    recalled, hits = {}, 0
    for i in world["targets"]:
        colour, obj = OBJ_WORDS[i]
        res = mem.recall(f"where did i see the {colour} {obj}", top_k=1, now=now)
        if res:
            hits += res[0].event.eid == world["sighted"][i].eid
            recalled[i] = res[0].event.meta.get("pos")
    mem_steps = _arm_steps(seed, world, model, data, acts, 20000,
                           lambda i: recalled.get(i))

    n, ne = len(world["targets"]), len(null_steps)
    return {
        "search_time_ratio": round(sum(mem_steps) / ne / world["null_mean"], 4),
        "memory_mean_steps": round(sum(mem_steps) / ne, 1),
        "null_mean_steps": round(world["null_mean"], 1),
        "retrieval_hit_rate": round(hits / n, 4),
        "null_success_rate": round(sum(s < CAP_Q for s in null_steps) / ne, 4),
        "ep1_sighted_frac": world["ep1_sighted_frac"],
        "n_queries": n,
    }


def _control(seed: int) -> dict:
    """Shuffled positions and recency-only retrieval — both must lose the speedup."""
    world = _CACHE[seed]
    model, data, acts = _build(world["obj_pos"])
    mem, now = world["mem"], world["now"]
    sighted = world["sighted"]

    # Derange sighting positions by rotation: with >= 2 sightings, no event
    # keeps its own place, so a correct retrieval yields a wrong destination.
    order = sorted(sighted)
    shuffled_pos = {order[j]: sighted[order[(j + 1) % len(order)]].meta["pos"]
                    for j in range(len(order))}

    latest = max(sighted.values(), key=lambda e: e.t)
    recalled_ok = {}
    for i in world["targets"]:
        colour, obj = OBJ_WORDS[i]
        recalled_ok[i] = bool(
            mem.recall(f"where did i see the {colour} {obj}", top_k=1, now=now))

    shuf_steps = _arm_steps(seed, world, model, data, acts, 40000,
                            lambda i: shuffled_pos[i] if recalled_ok[i] else None)
    rec_steps = _arm_steps(seed, world, model, data, acts, 60000,
                           lambda i: latest.meta["pos"])

    ne = len(shuf_steps)
    return {
        "search_time_ratio_shuffled": round(sum(shuf_steps) / ne / world["null_mean"], 4),
        "search_time_ratio_recency": round(sum(rec_steps) / ne / world["null_mean"], 4),
    }


def _check(m: dict, c: dict) -> bool:
    return (m["search_time_ratio"] <= RATIO_MAX
            and m["retrieval_hit_rate"] >= RETRIEVAL_MIN
            and m["null_success_rate"] >= NULL_SUCCESS_MIN
            and c["search_time_ratio_shuffled"] >= CONTROL_RATIO_MIN
            and c["search_time_ratio_recency"] >= CONTROL_RATIO_MIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.20"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
