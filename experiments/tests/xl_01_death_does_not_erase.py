"""XL.01 — death does not erase what he learned.

GOAL.md: "Life N+1 must be measurably better than life N *because of* what
life N recorded." XL.00 certified the MECHANISM — death ends a life, the
respawn is uniform and blind, the diary crosses. This spec asks whether the
crossing MATTERS: a life that follows earlier lives must reach a survival
criterion (eating) faster than a life whose carried memory was wiped at death.

The task, end to end in World Zero (`w0.py`, the certified death/respawn/diary
substrate), lethal, uniform legal respawns, PS.01-calibrated drives:

  Every life      One policy, identical across all arms (T2.20's rule: the
                  comparison is store vs no-store, never policy vs policy).
                  It reads the diary for food sightings at life start, drives
                  to the nearest remembered position, and falls back to
                  random-waypoint exploration when memory is empty or wrong.
                  Passing within SIGHT_R of a food geom records a "saw" row
                  with the perceived position in event.meta — the ME contract.
                  Eating is the WORLD's event (`drives` contact + nu), never
                  the harness's.

  CARRIED (claim) the diary persists across deaths, exactly as W0-3 ships it.
  WIPED (null)    the diary is cleared at every death: every life is a first
                  life. This IS the first-life learning curve, sampled N times
                  rather than once per seed.
  ALIEN (control) the diary is REPLACED at every death with another Jack's
                  lived store — sightings recorded by the same policy
                  exploring a DIFFERENT world (chosen so its food lies
                  >= ALIEN_MIN_DIST from every true food here, measured, VOID
                  if no such world is found in the scan). ME.3's precedent: a
                  memory that helps regardless of whose life it came from is
                  not memory, it is a prior. It must NOT beat wiped; the hurt
                  direction is reported.
  REFERENCE       the same controller handed the TRUE nearest food position —
                  the must-succeed arm, piloted FIRST (the SH.01 lesson:
                  reference, then tripwires, then claim arms — its failure
                  makes every other number moot). If it cannot feed, the rig
                  cannot produce the behaviour under ANY memory and the run is
                  VOID, not FAIL.

  metric          lives_to_criterion_vs_wiped. A life "meets criterion" when
                  it feeds within CRIT_S of spawn; lives_to_criterion is the
                  first life k where lives k AND k+1 both meet it (consecutive
                  pair, because a lucky spawn next to food happens ~1 life in
                  6 and does not repeat; memory does). Plus the time ratio:
                  median time-to-first-feed over lives 2..N, carried / wiped,
                  timeouts counted at the full cap so chasing a wrong memory
                  cannot look good by failing fast (T2.20's rule).

DECLARED FIXTURES, all external to `w0.py`, all symmetric across arms:
  * SHORT LIVES: every life starts at LIFE_E0 (XL.00's fixture, larger here
    because a life must be long enough for exploration to find food — the
    probe measured sighting at 0-620 decisions from random spawns and two
    lives of 890 decisions that never sighted). After the first feed (or the
    cap) the measurement is over and the body is drained to end the life
    through the world's own death path.
  * REGROWTH CLOCK RESET at each death. W0 deliberately does not reset food
    timers on death ("death must not hand him a freshly reset world") — but
    that rule prices CONTINUOUS time, and the drain fixture compresses ~600 s
    of dead time out of every life. Without the reset, the food a life just
    ate is still on its 66.9 s regrowth timer when the next life spawns
    ~10 s later, so the carried arm's remembered food would be blocked BY THE
    FIXTURE's time compression, not by the world. The reset restores the
    timing a full-length life would have seen. It is applied to every arm
    identically and the policy never reads timers.
  * The spawn SEQUENCE is identical across arms of a seed (same seed, same
    `_spawn_rng`, one draw per death), so carried and wiped face the same
    spawn positions life for life — a paired comparison.

WHAT IS REPORTED SEPARATELY, per the spec's notes. The registered notes ask
for two ablations: store carried / weights wiped, and weights carried / store
wiped. The policy here is parameter-free — there are no weights, by
construction, because no learner exists at this envelope (LC.03's pilot
measured every claim arm at 0.0 on a richer rig, and SH.01's oracle-learner
could not execute the behaviour PPO was supposed to learn). So the
store-carried path is the claim, `weights_path_exists` is recorded as 0.0
rather than silently omitted, and the weights-carried half of the
complementary-learning-systems prediction is exactly LC.03/LC.04's
`life_gain` — the bakeoff in flight — not something this spec can fake with
a learner that cannot learn here. That deviation is declared, not hidden.

Honesty notes, pre-registered:
  - Perception is abstracted to proximity (SIGHT_R), as T2.20 declared and
    PG.4 established; vision-under-noise is other specs' claim.
  - The reference arm is privileged BY DESIGN (it reads true positions) and
    is never compared to the claim arms; it licenses reading a wiped/carried
    contrast at all.
  - Food positions are the world's own (`playground` per-seed placement);
    objects are free bodies and may be pushed — a sighting records where the
    food WAS, and a memory pointing at a moved object costs the carried arm,
    never helps it.
  - Natural deaths during measurement (drowning, starving mid-search) are the
    world being the world: the life scores the full cap, unfed, and the loop
    moves on.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..w0 import POOL_XY, W0, SIM_S_PER_DECISION
# After `..w0`, deliberately: importing it puts the repo root on sys.path.
from EpisodicMemory import EpisodicMemory   # noqa: E402

# The world, the drive layer and the store are what this claim rides on.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "EpisodicMemory.py"]

# ── pre-registered constants ───────────────────────────────────────────────
LIFE_E0 = 0.55            # 330 s basal runway > the cap: the measurement
                          # window is never cut short by the fixture itself
CAP_LIFE = 1500           # decisions; 300 s measurement window per life
N_LIVES = 8               # lives per arm per seed
SIGHT_R = 0.5             # m; seeing food == being within this. The claim
                          # pilot at T2.20's 1.0 measured the wiped null at
                          # 33.8 s mean - exploration in this small arena finds
                          # central food nearly as fast as memory drives to it
                          # (ratio 0.59, no headroom). Noticing food only when
                          # nearly upon it is the regime where remembering
                          # where it grows pays; symmetric across every arm.
CAND_DEDUPE_R = 0.3       # m; two sightings closer than this are one memory
KP, KD = 2.0, 0.8         # the PD drive — apparatus, identical in every arm
WAYPOINT_DONE = 0.4       # m
WAYPOINT_MAX_AGE = 100    # decisions
PATIENCE = 60             # decisions parked at a remembered position with no
                          # feed before the policy gives up on that memory
STUCK_N = 50              # decisions without closing on the target before the
                          # policy assumes an obstacle and detours (the ref
                          # pilot measured 2/8 lives wedged on world geometry
                          # driving a straight line — walls and the ladder are
                          # real, and every arm gets the same escape)
DETOUR_N = 60             # decisions of exploration per detour
POOL_MARGIN = 0.8         # m the router keeps clear of the pool footprint.
                          # THE POOL IS A ONE-WAY TRAP FOR THIS BODY, measured
                          # in the pilot: the drive force is gated on
                          # floor/ramp/stair contact and the pit floor is none
                          # of them, so a rover that slides in has no traction
                          # forever (it stalled at (2.35, -3.11) for 1050
                          # decisions under full drive). Routing around a
                          # fixed, known world feature is a declared rig
                          # convenience like the drive itself — identical in
                          # every arm, orthogonal to the store-vs-no-store
                          # contrast. Escaping the pit needs the ARMS, which
                          # is a climbing claim, not a memory claim.
DRAIN_E = 5e-4            # the post-measurement drain charge
CRIT_S = 20.0             # s; a life meeting criterion fed within this.
                          # Piloted: memory-driven lives measure 2-10 s and
                          # travel tops out ~15 s, while a lucky exploration
                          # life landed at exactly 30.0 s under the first
                          # draft's 30 - the criterion must sit between the
                          # two regimes, not on one's tail.
FOOD_NAMES = ("obj0", "obj1")   # floor food; the apple is on the platform and
                                # a driving rover cannot reach it

RATIO_MAX = 0.5           # carried must at least halve wiped's search time
REF_MIN_FED = 0.8         # reference feeds on >= this fraction of lives, else
                          # the rig is broken -> VOID (SH.01's routing)
WIPED_MIN_FED = 0.5       # the null must feed on >= this fraction of lives
                          # 2..N or the cap saturates the denominator -> VOID
CONTROL_RATIO_MIN = 0.75  # alien must NOT recover the speedup (T2.20's value)
ALIEN_MIN_DIST = 1.5      # m; every alien food position must sit at least
                          # this far from every true food position — 3x
                          # SIGHT_R, so a rover parked on an alien memory
                          # cannot sight true food from there. 2.0 was the
                          # first draft and is UNSATISFIABLE: food spawns in a
                          # 4 x 2.5 m box, and the measured best-achievable
                          # separation across 30 candidate worlds is
                          # 1.85/1.79/1.97 m for seeds 0/1/2.
ALIEN_SCAN = range(101, 131)   # candidate alien world seeds
ALIEN_BUILD_CAP = 3000    # decisions the alien Jack gets to live its life

# Shared per-seed state: _experiment fills it, _control reuses it so the
# control judges the SAME wiped baseline and alien store (run_spec runs all
# experiment seeds first — T2.20's pattern).
_CACHE: dict = {}


def _calibration():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _quadrant(x: float, y: float) -> str:
    return ("north" if y >= 0 else "south") + ("east" if x >= 0 else "west")


def _food_positions(w) -> dict:
    out = {}
    for name in FOOD_NAMES:
        try:
            gid = int(w.model.geom(name).id)
        except (KeyError, ValueError):
            continue
        out[name] = (gid, np.array(w.data.geom_xpos[gid][:2], dtype=float))
    return out


def _pool_half(w) -> float:
    return float(w.params.pool_size) + POOL_MARGIN


def _in_pool(xy, half: float) -> bool:
    return (abs(float(xy[0]) - POOL_XY[0]) < half
            and abs(float(xy[1]) - POOL_XY[1]) < half)


def _crosses_pool(a, b, half: float) -> bool:
    for s in np.linspace(0.0, 1.0, 24):
        if _in_pool(a + s * (b - a), half):
            return True
    return False


def _route(xy, target, half: float):
    """The next waypoint toward `target` that stays off the pool footprint."""
    if _in_pool(xy, half):                     # in the margin band: flee first
        away = xy - np.array(POOL_XY)
        n = float(np.linalg.norm(away))
        return xy + (away / n if n > 1e-6 else np.array([1.0, 0.0])) * 1.5
    if not _crosses_pool(xy, target, half):
        return target
    corners = [np.array([POOL_XY[0] + sx * half, POOL_XY[1] + sy * half])
               for sx in (-1.2, 1.2) for sy in (-1.2, 1.2)]
    # A corner the rover is already standing on is not a next leg — returning
    # it parks the body there forever (measured: 300 s at (5.36, -5.16)).
    return min((c for c in corners
                if not _crosses_pool(xy, c, half)
                and float(np.linalg.norm(c - xy)) >= 0.6),
               key=lambda c: float(np.linalg.norm(c - xy))
               + float(np.linalg.norm(target - c)),
               default=target)


class _Explorer:
    """Random-waypoint wanderer — every arm's fallback (T2.20's, plus the
    pool exclusion: a waypoint in the trap is not exploration, it is death)."""

    def __init__(self, rng, half: float):
        self.rng, self.wp, self.age, self.half = rng, None, 0, half

    def target(self, xy):
        if (self.wp is None or self.age >= WAYPOINT_MAX_AGE
                or float(np.linalg.norm(xy - self.wp)) < WAYPOINT_DONE):
            while True:
                wp = np.array([self.rng.uniform(-4.5, 4.5),
                               self.rng.uniform(-4.5, 4.5)])
                if not _in_pool(wp, self.half):
                    break
            self.wp = wp
            self.age = 0
        self.age += 1
        return self.wp


def _recall_positions(diary: EpisodicMemory, now: float) -> list:
    """Remembered food positions, via the ME retrieval contract, deduped."""
    out = []
    for r in diary.recall("saw food room", top_k=8, channel="saw", now=now):
        pos = r.event.meta.get("pos")
        if pos is None:
            continue
        p = np.array(pos, dtype=float)
        if all(float(np.linalg.norm(p - q)) >= CAND_DEDUPE_R for q in out):
            out.append(p)
    return out


def _live_one_life(w, diary, mode: str, rng, ref_targets=None,
                   record_sightings: bool = True) -> dict:
    """Run one life to first feed or CAP_LIFE. Returns ttf and diagnostics.

    The policy: remembered (or, for "ref", privileged) food positions are
    candidates; drive to the nearest un-blacklisted one; PATIENCE decisions
    parked there without a feed blacklists it; no candidates -> explore.
    """
    root = w.ix["root_dofadr"]
    foods = _food_positions(w)
    half = _pool_half(w)
    explorer = _Explorer(rng, half)
    if mode == "ref":
        candidates = [np.array(p, dtype=float) for p in ref_targets]
    else:
        candidates = _recall_positions(diary, now=w.sim_seconds)
    blacklist: list = []
    parked_since = None
    best_dist, best_at = float("inf"), 0
    detour_until = -1
    ate0 = sum(w.drives.ate_total.values())
    sighted_this_life = list(candidates)

    for d in range(CAP_LIFE):
        xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        if sum(w.drives.ate_total.values()) > ate0:
            return {"ttf_s": d * SIM_S_PER_DECISION, "fed": True,
                    "died": False, "n_candidates": len(candidates)}
        # sighting: proximity-abstracted perception, recorded through the store
        for name, (gid, _) in foods.items():
            fp = np.array(w.data.geom_xpos[gid][:2], dtype=float)
            if float(np.linalg.norm(fp - xy)) < SIGHT_R and all(
                    float(np.linalg.norm(fp - q)) >= CAND_DEDUPE_R
                    for q in sighted_this_life):
                sighted_this_life.append(fp.copy())
                candidates.append(fp.copy())
                if record_sightings:
                    diary.record(
                        "saw", "jack",
                        f"jack saw food in the {_quadrant(*fp)} part of the room",
                        importance=3.0, t=w.sim_seconds,
                        meta={"pos": [float(fp[0]), float(fp[1])],
                              "life": w.life})
        live = [c for c in candidates
                if not any(c is b for b in blacklist)]
        if live and d >= detour_until:
            target = min(live, key=lambda c: float(np.linalg.norm(c - xy)))
            dist = float(np.linalg.norm(target - xy))
            if dist < WAYPOINT_DONE:
                parked_since = d if parked_since is None else parked_since
                if d - parked_since >= PATIENCE:
                    blacklist.append(target)
                    parked_since = None
            else:
                parked_since = None
                # wedged on world geometry: no progress toward the target for
                # STUCK_N decisions -> detour, then try again. Symmetric
                # across arms; costs the memory arms time, never buys it.
                if dist < best_dist - 0.1:
                    best_dist, best_at = dist, d
                elif d - best_at >= STUCK_N:
                    detour_until = d + DETOUR_N
                    explorer.wp = None
                    best_dist, best_at = float("inf"), d
        else:
            target = explorer.target(xy)
        wp = _route(xy, target, half)
        v = np.array([w.data.qvel[root], w.data.qvel[root + 1]], dtype=float)
        a = np.zeros(w.action_dim)
        a[4:6] = -1.0        # adhesion OFF: action 0 maps to MID-range, and
        #                      450 N of glue anchors a toppled body to the
        #                      floor for good (measured: v=0 under full drive)
        a[6:8] = np.clip(KP * (wp - xy) - KD * v, -1.0, 1.0)
        w.decide(a)
        if w.died_this_decision:
            return {"ttf_s": CAP_LIFE * SIM_S_PER_DECISION, "fed": False,
                    "died": True, "n_candidates": len(candidates)}
    return {"ttf_s": CAP_LIFE * SIM_S_PER_DECISION, "fed": False,
            "died": False, "n_candidates": len(candidates)}


def _food_home(w) -> list:
    """(qposadr, dofadr, home qpos) for each food body's free joint."""
    out = []
    for name in FOOD_NAMES:
        try:
            gid = int(w.model.geom(name).id)
        except (KeyError, ValueError):
            continue
        bid = int(w.model.geom_bodyid[gid])
        jadr = int(w.model.body_jntadr[bid])
        qadr = int(w.model.jnt_qposadr[jadr])
        dadr = int(w.model.jnt_dofadr[jadr])
        out.append((qadr, dadr, w.data.qpos[qadr:qadr + 7].copy()))
    return out


def _reset_food(w, home: list) -> None:
    """Food regrows where it grew — the death-boundary half of the regrowth
    fixture. Within a life the objects are ordinary pushable bodies (and the
    toppled rover measurably bulldozes them metres while feeding); across a
    death, an item that stayed wherever the LAST body shoved it would make
    every remembered position wrong through the fixture's own dynamics, and
    the claim would be about pushing physics rather than memory."""
    for qadr, dadr, q in home:
        w.data.qpos[qadr:qadr + 7] = q
        w.data.qvel[dadr:dadr + 6] = 0.0
    w.mujoco.mj_forward(w.model, w.data)


def _drain(w) -> None:
    """End the life through the world's own death path (energy -> 0)."""
    w.drives.state = drives.DriveState(e=DRAIN_E)
    a = np.zeros(w.action_dim)
    for _ in range(10):
        w.decide(a)
        if w.died_this_decision:
            return
    raise RuntimeError("drain fixture failed to end the life in 10 decisions")


def _wipe(diary: EpisodicMemory) -> None:
    diary.events.clear()
    diary._tok.clear()


def _refill(diary: EpisodicMemory, rows: list, now: float) -> None:
    _wipe(diary)
    for text, meta in rows:
        diary.record("saw", "jack", text, importance=3.0, t=now, meta=meta)


def _run_arm(seed: int, mode: str, j0: float, alpha: float,
             alien_rows: list = None) -> list:
    """N_LIVES lives under one arm. Returns the per-life dicts."""
    diary = EpisodicMemory()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True, diary=diary)
    home = _food_home(w)
    rng = np.random.RandomState(seed * 45007 + {"ref": 1, "carried": 2,
                                                "wiped": 3, "alien": 4}[mode])
    lives = []
    for life in range(N_LIVES):
        if life > 0:                     # a death has just happened
            if mode == "wiped":
                _wipe(diary)
            elif mode == "alien":
                _refill(diary, alien_rows, now=w.sim_seconds)
        # the declared fixtures: short life, regrowth clock + position reset
        w.drives.state = drives.DriveState(e=LIFE_E0)
        w.drives._respawn_at = {n: 0.0 for n in w.drives._respawn_at}
        _reset_food(w, home)
        ref_targets = None
        if mode == "ref":
            ref_targets = [p for _, (_, p) in
                           sorted(_food_positions(w).items())]
        r = _live_one_life(w, diary, mode, rng, ref_targets=ref_targets,
                           record_sightings=(mode != "ref"))
        lives.append(r)
        if not r["died"]:
            _drain(w)
    return lives


def _build_alien_store(seed: int, j0: float, alpha: float) -> tuple:
    """A different Jack's lived sightings, from a world whose food is far.

    Scans ALIEN_SCAN for the first world seed whose every food position sits
    >= ALIEN_MIN_DIST from every true food position of world `seed`, then
    lets the same policy live one exploration window there and keeps its
    "saw" rows. Returns (rows, fixture_metrics); rows is None when no
    adequate world exists in the scan — the caller VOIDs.
    """
    true_w = W0(seed=seed, j0=j0, alpha=alpha)
    true_pos = [p for _, (_, p) in _food_positions(true_w).items()]
    for cand in ALIEN_SCAN:
        aw = W0(seed=cand, j0=j0, alpha=alpha)
        apos = [p for _, (_, p) in _food_positions(aw).items()]
        if not apos:
            continue
        dmin = min(float(np.linalg.norm(a - t)) for a in apos for t in true_pos)
        if dmin < ALIEN_MIN_DIST:
            continue
        diary = EpisodicMemory()
        rng = np.random.RandomState(cand * 90001 + 11)
        root = aw.ix["root_dofadr"]
        foods = _food_positions(aw)
        half = _pool_half(aw)
        explorer = _Explorer(rng, half)
        known: list = []
        for d in range(ALIEN_BUILD_CAP):
            xy = np.array(aw.data.xpos[aw.rover_bid][:2], dtype=float)
            for name, (gid, _) in foods.items():
                fp = np.array(aw.data.geom_xpos[gid][:2], dtype=float)
                if float(np.linalg.norm(fp - xy)) < SIGHT_R and all(
                        float(np.linalg.norm(fp - q)) >= CAND_DEDUPE_R
                        for q in known):
                    known.append(fp.copy())
                    diary.record(
                        "saw", "jack",
                        f"jack saw food in the {_quadrant(*fp)} part of the room",
                        importance=3.0, t=aw.sim_seconds,
                        meta={"pos": [float(fp[0]), float(fp[1])]})
            if len(known) == len(foods):
                break
            wp = _route(xy, explorer.target(xy), half)
            v = np.array([aw.data.qvel[root], aw.data.qvel[root + 1]])
            a = np.zeros(aw.action_dim)
            a[4:6] = -1.0
            a[6:8] = np.clip(KP * (wp - xy) - KD * v, -1, 1)
            aw.decide(a)
        if not diary.events:
            continue                    # a blind alien is no control
        rows = [(e.text, dict(e.meta)) for e in diary.events]
        return rows, {"alien_seed": float(cand), "alien_min_dist": dmin,
                      "alien_rows": float(len(rows))}
    return None, {"alien_seed": -1.0, "alien_min_dist": 0.0,
                  "alien_rows": 0.0}


def _ttf2(lives: list) -> float:
    """MEDIAN time-to-first-feed over lives 2..N, timeouts at full cap.

    Median, not mean, and the pilot is why: search time under geometry
    wedging is heavy-tailed in EVERY arm (a single carried life measured
    188 s in a world whose other six carried lives measured 1-28 s), and at
    n=7 one blown life owns a mean. The tail is reported separately via the
    fed fractions; the central tendency is what the claim is about."""
    return float(np.median([r["ttf_s"] for r in lives[1:]]))


def _fed_frac2(lives: list) -> float:
    return float(np.mean([r["fed"] for r in lives[1:]]))


def _lives_to_criterion(lives: list) -> float:
    """First 1-based k where lives k and k+1 both fed within CRIT_S."""
    ok = [r["fed"] and r["ttf_s"] <= CRIT_S for r in lives]
    for k in range(len(ok) - 1):
        if ok[k] and ok[k + 1]:
            return float(k + 1)
    return float(len(ok) + 1)


def _experiment(seed: int) -> dict:
    j0, alpha, prov = _calibration()
    m: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return m

    ref = _run_arm(seed, "ref", j0, alpha)
    m["ref_fed_frac"] = float(np.mean([r["fed"] for r in ref]))
    m["ref_mean_ttf_s"] = float(np.mean(
        [r["ttf_s"] for r in ref if r["fed"]] or [float("nan")]))
    m["ok_ref"] = float(m["ref_fed_frac"] >= REF_MIN_FED)

    carried = _run_arm(seed, "carried", j0, alpha)
    wiped = _run_arm(seed, "wiped", j0, alpha)

    alien_rows, alien_fix = _build_alien_store(seed, j0, alpha)
    _CACHE[seed] = {"j0": j0, "alpha": alpha, "alien_rows": alien_rows,
                    "alien_fix": alien_fix, "wiped_ttf2": _ttf2(wiped)}

    m["carried_life1_ttf_s"] = float(carried[0]["ttf_s"])
    m["carried_ttf2_s"] = _ttf2(carried)
    m["carried_fed_frac2"] = _fed_frac2(carried)
    m["carried_ltc"] = _lives_to_criterion(carried)
    m["wiped_ttf2_s"] = _ttf2(wiped)
    m["wiped_fed_frac2"] = _fed_frac2(wiped)
    m["wiped_ltc"] = _lives_to_criterion(wiped)
    m["search_time_ratio"] = (m["carried_ttf2_s"] / m["wiped_ttf2_s"]
                              if m["wiped_ttf2_s"] > 0 else float("nan"))
    m["carried_deaths_natural"] = float(sum(r["died"] for r in carried))
    m["wiped_deaths_natural"] = float(sum(r["died"] for r in wiped))
    m["n_lives"] = float(N_LIVES)
    m["crit_s"] = CRIT_S
    m["weights_path_exists"] = 0.0      # declared: parameter-free policy; the
    #                                     weights-carried ablation is LC.03/04

    m["ok_null_informative"] = float(m["wiped_fed_frac2"] >= WIPED_MIN_FED)
    m["ok_claim"] = float(m["search_time_ratio"] <= RATIO_MAX
                          and m["carried_ltc"] < m["wiped_ltc"])
    return m


def _control(seed: int) -> dict:
    """ANOTHER JACK'S MEMORIES — must not help, should hurt."""
    cache = _CACHE.get(seed)
    if cache is None or cache["alien_rows"] is None:
        fix = (cache or {}).get("alien_fix", {})
        return {"c_fixture_ok": 0.0, **{f"c_{k}": v for k, v in fix.items()}}
    alien = _run_arm(seed, "alien", cache["j0"], cache["alpha"],
                     alien_rows=cache["alien_rows"])
    ttf2 = _ttf2(alien)
    wiped_ttf2 = cache["wiped_ttf2"]
    ratio = ttf2 / wiped_ttf2 if wiped_ttf2 > 0 else float("nan")
    return {
        "c_fixture_ok": 1.0,
        **{f"c_{k}": v for k, v in cache["alien_fix"].items()},
        "c_alien_ttf2_s": ttf2,
        "c_alien_fed_frac2": _fed_frac2(alien),
        "c_alien_ltc": _lives_to_criterion(alien),
        "c_alien_vs_wiped_ratio": ratio,
        "c_alien_ok": float(ratio >= CONTROL_RATIO_MIN),
    }


def _check(m: dict, c: dict):
    # ── the instrument ──────────────────────────────────────────────────
    if m.get("calibrated", 0.0) != 1.0:
        return Status.VOID              # PS.01 supplied no usable j0/alpha
    if m.get("ok_ref", 0.0) != 1.0:
        return Status.VOID              # the rig cannot produce the behaviour
        # under ANY memory; a wiped/carried contrast in it measures nothing
    if m.get("ok_null_informative", 0.0) != 1.0:
        return Status.VOID              # the null saturated at the cap; the
        # ratio's denominator is the fixture, not the world
    if c.get("c_fixture_ok", 0.0) != 1.0:
        return Status.VOID              # no adequately-distant alien world:
        # the control could help by accident and its verdict would be noise
    # ── the control, on its declared side, EVERY seed ───────────────────
    if c.get("c_alien_ok", 0.0) != 1.0:
        return False                    # a foreign store recovered the
        # speedup: the "memory" is a prior, and the test measures nothing
    # ── the claim ───────────────────────────────────────────────────────
    return bool(m.get("ok_claim", 0.0) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["XL.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())


if __name__ == "__main__":
    # Pilot modes, in decisiveness-per-second order (the SH.01 lesson):
    #   ref      the must-succeed reference arm, one seed
    #   claim    carried vs wiped, one seed
    #   alien    the control fixture + arm, one seed
    mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    j0, alpha, prov = _calibration()
    print(f"calibration j0={j0} alpha={alpha}")
    if mode == "ref":
        lives = _run_arm(seed, "ref", j0, alpha)
        for i, r in enumerate(lives):
            print(f"  ref life {i}: ttf {r['ttf_s']:.1f}s fed {r['fed']} "
                  f"died {r['died']}")
        print(f"fed_frac {np.mean([r['fed'] for r in lives]):.2f}")
    elif mode == "claim":
        carried = _run_arm(seed, "carried", j0, alpha)
        wiped = _run_arm(seed, "wiped", j0, alpha)
        for name, lv in (("carried", carried), ("wiped", wiped)):
            print(f"  {name}: " + " ".join(
                f"{r['ttf_s']:.0f}{'F' if r['fed'] else ''}" for r in lv))
            print(f"    ttf2 {_ttf2(lv):.1f}s fed2 {_fed_frac2(lv):.2f} "
                  f"ltc {_lives_to_criterion(lv):.0f}")
        print(f"ratio {_ttf2(carried) / _ttf2(wiped):.3f}")
    elif mode == "alien":
        rows, fix = _build_alien_store(seed, j0, alpha)
        print(f"fixture: {fix}")
        if rows:
            wiped = _run_arm(seed, "wiped", j0, alpha)
            alien = _run_arm(seed, "alien", j0, alpha, alien_rows=rows)
            print(f"  alien ttf2 {_ttf2(alien):.1f}s vs wiped "
                  f"{_ttf2(wiped):.1f}s ratio "
                  f"{_ttf2(alien) / _ttf2(wiped):.3f}")
