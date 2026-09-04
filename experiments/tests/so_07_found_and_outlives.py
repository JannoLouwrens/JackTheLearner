"""SO.07 — What the hands leave is FOUND, and what he learns outlives them.

GOAL.md, the owner: *"their hands may leave things in his world for him to
find — food where he might look... Never puppeteering: what is left must
still be found, learned, and chosen by him."* SO.06 certified the CHANNEL
(a hand reaches only through the world); SO.09 audits the BOOKS (every
placement logged and reconciled). This spec is the CLAIM: provisioning helps,
and what it buys is HIS — it survives the hand's withdrawal.

THE LEARNER, identical in every arm (T2.20's rule: the comparison is
hand-policy vs hand-policy, never learner vs learner). XL.01's certified
diary-reading forager, with ONE addition that is the same in every arm:
foraging is NEED-GATED. He forages (recalled food positions, else
random-waypoint exploration) while e < FORAGE_E and RESTS (zero action,
adhesion off) while sated. The gate is what makes C-GIVE a real puppeteer
probe: a hand that makes searching unnecessary removes the practice — de
Haan, Jayaraman & Levine 2019's helper-in-the-observation result, instantiated
as a policy that never needs to look. Learning = the diary (sightings of food
within SIGHT_R, recorded through the ME contract, carried across deaths —
XL.01's certified store). No weights, no optimiser: "matched optimiser steps"
is discharged as matched decision caps, and matched wall-clock as identical
phase structure per arm.

THE PHASES, per arm, one diary carried through all of them:
  TRAIN  N_TRAIN lives at E0_TRAIN = 0.35 (basal runway ~210 s < the 300 s
         cap: an unfed life DIES inside the window — the hand can matter).
         Hand live per the arm's policy.
  ON     N_ON scoring lives, hand still live, E0_SCORE = 0.35 (hungry from
         t=0, so the gift trigger is armed immediately and C_on measures
         competence WITH the scaffold, not trigger latency).
  OFF    N_OFF scoring lives, hand withdrawn AND the apple PARKED off-world
         (40 m — SO.06's parking idiom) for EVERY arm, so every arm's
         withdrawal world is identical and any C_off difference is carried by
         the diary plus the lived training, never by leftover gifts.

COMPETENCE is 1 / median(time-to-first-feed) over a phase's scoring lives,
timeouts and pre-feed deaths counted at the full cap (T2.20's rule: chasing a
wrong memory cannot look good by failing fast). Deliberately NOT energy
eaten: the gift apple's nu (0.50) would mechanically inflate C_on for every
gift arm and manufacture the low-R puppeteering verdict out of bookkeeping —
found while designing, recorded so nobody "fixes" the metric into that trap.
Time-to-feed is nu-blind: help shows up as C_on, learning as C_off.

    C_on  = competence, hand live       C_off = competence, hand withdrawn
    C0    = A0's C_off (the null's)
    R = C_off / C_on   (does it survive removal?)
    B = C_off / C0     (was it worth having?)     [PURPOSE_AND_SCAFFOLDING
                                                   §3.6, reused verbatim]

THE ARMS. The gift is the APPLE — deliberately: it sits on the platform where
a driving rover cannot reach it (XL.01's measured exclusion), so in A0 it is
inert, in gift arms every apple eat IS a gift eat (attribution is exact, no
new matter, no model edit), and it is NOT one of the floor-food homes the
diary must learn — the hand never moves the curriculum.
  A0        no hand. The null.
  A1        CONTINGENT (the claim arm): drop at DROP_RANGE_M = 3 m in a
            hand-rng direction, legal and off-pool, when e < E_GIFT, apple
            edible, refractory passed. He must still walk to it and find it.
  A2        GRADED (the meerkat arm, Thornton & McAuliffe 2006): as A1, but
            early lives get floor-food-value gifts (obj0), the apple only
            from life GRADE_K on. Reported, NOT load-bearing for the verdict:
            the registered hypothesis is A1's need-contingent hand; A2's
            numbers go to the Review as bakeoff information. Pre-registered
            here so a post-hoc argmax over arms cannot pick seed noise.
  C-MISPLACE (control, MUST fail): same item, same per-life drop COUNTS as
            the measured A1 run at this seed, times and positions drawn from
            the hand's own rng independent of Jack and of e. Separates "the
            hand helped him" from "the world contained more reachable food".
  C-GIVE    (control, calibrates failure): the apple placed IN CONTACT with
            his body at the trigger — consumption still goes through
            drives.DriveLayer's contact rule (NE.03's energy identity holds;
            the direct-e variant was struck at registration). MUST show high
            C_on and R below R_FLOOR, else the instrument cannot see
            puppeteering and the run is VOID, not a verdict.

GATES, all pre-registered before any number was seen (worst-seed over
SEEDS, the LG.02 idiom — per-seed keys are in the row):
  PASS               A1: R >= 0.8 AND B >= 1.5 on every seed — the §3.7
                     outcome table's "owner right" cell, thresholds reused
                     verbatim, unweakened.
  FAIL(puppeteering) A1: R < 0.5 with B > 1 in >= 2 of 3 seeds (§3.7's
                     falsification shape). Reported as a violation of a
                     GOAL.md ENDS-class commitment — the row carries
                     puppeteering_seeds; the response is to STOP provisioning,
                     never to keep it because B looks good. Written before
                     the run, because after it "load-bearing" and
                     "puppeteering" are the same numbers read conveniently.
  FAIL(dead channel) B < 1.5 anywhere else: the hand bought nothing (B ~ 1),
                     or hurt (B < 1), or retention is partial (R 0.5-0.8:
                     report, do not round up).
  VOID lanes         instrument death, never refutation: PS.01 calibration
                     refused; REF arm (privileged positions) fed_frac <
                     REF_MIN_FED (the rig cannot produce the behaviour —
                     SH.01's lesson); A0 never feeds in OFF (C0 degenerate);
                     A1 drops == 0 (trigger never armed) or gift eats == 0
                     (channel never exercised — the claim was not tested);
                     C-GIVE calibration miss (above); C-MISPLACE clearing the
                     claim conjunction (law 2: a passing control means the
                     test measures nothing).

WHY THE DIARY CAN CARRY THE CLAIM (the mechanism, stated before running):
an A0 train life that starves dies ~decision 1050 and stops seeing; an A1
life kept alive by a found gift explores to the cap. More lived decisions →
more true floor-food sightings → a richer carried diary → faster hungry
feeding after withdrawal. The gift's own sightings are recorded too (his
diary is his); in OFF lives those rows point at a parked apple and cost
PATIENCE-bounded time — a bias AGAINST the claim, accepted as conservative.

SO.09 REPLAY RE-BUY: every hand log (A1/A2/C-MISPLACE/C-GIVE) plus end-of-life
food positions per life is written to /data/so07_hand_logs_s{seed}.json.
SO.09's registered note says a measured C-GIVE log replayed through the
accountant supersedes its synthesised one; this file is that log's source.

PILOT LANE (T3.09's discipline): `python -m experiments.tests.
so_07_found_and_outlives aliveness <world>` runs TRAIN-phase lives on pilot
worlds 0-2 (the recording runs on WORLD_BASE+seed = 3..5) and prints COUNTS
ONLY — deaths, drops, gift eats, sightings, wall seconds. No competence, no
R, no B: fixture constants may be re-frozen from counts, never from a peek at
the claim metric.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .. import drives
from ..w0 import W0, SIM_S_PER_DECISION
# After `..w0`, deliberately: importing it puts the repo root on sys.path.
from EpisodicMemory import EpisodicMemory   # noqa: E402
from .xl_01_death_does_not_erase import (   # the certified forager machinery
    _Explorer, _calibration, _drain, _food_home, _food_positions, _in_pool,
    _pool_half, _quadrant, _recall_positions, _reset_food, _route, _widen,
    CAND_DEDUPE_R, KD, KP, PATIENCE, SIGHT_R, STUCK_N, DETOUR_N,
    WAYPOINT_DONE)
from .so_06_hand_reaches_through_world import Hand   # the only actuator

# The claim rides on the world, the drives, the store, the forager and the
# certified hand channel: a change to any of them must read as drift.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py",
             "EpisodicMemory.py",
             "experiments/tests/xl_01_death_does_not_erase.py",
             "experiments/tests/so_06_hand_reaches_through_world.py"]

# ── pre-registered constants ────────────────────────────────────────────────
SEEDS = (0, 1, 2)
WORLD_BASE = 3            # recording worlds 3..5; pilots use 0..2 (XL.01's
                          # convention: pilot worlds are design data)
CAP_LIFE = 1500           # decisions; 300 s per life (XL.01's window)
N_TRAIN, N_ON, N_OFF = 6, 3, 3
E0_TRAIN = 0.35           # ~210 s basal runway < the cap: an unfed train
                          # life dies INSIDE the window, so the hand can
                          # matter (deliberately unlike XL.01's 0.55)
E0_SCORE = 0.35           # scoring lives start hungry: the forage gate and
                          # the gift trigger are both armed at t=0
FORAGE_E = 0.60           # forage below, rest at or above (identical in
                          # every arm — the gate can favour no arm)
E_GIFT = 0.50             # the hand's need trigger ("drops when e < floor")
GIFTS_MAX = 2             # drops per life; the apple respawns at 129.6 s so
                          # a 300 s life supports at most 2 distinct gifts
GIFT_REFRACTORY = 150     # decisions (30 s) between drops: time to find one
GRADE_K = 3               # A2: lives 0..K-1 get obj0-value gifts, apple after
DROP_RANGE_M = 3.0        # SO.06's certified unseen-drop distance
DROP_TRIES = 16           # legal-placement attempts before skipping a drop
PARK_XY = (40.0, 0.0)     # SO.06's parking idiom: the withdrawn apple
TTF_FLOOR_S = SIM_S_PER_DECISION   # competence denominator floor (XL.01's)
CAP_S = CAP_LIFE * SIM_S_PER_DECISION

# gates — PURPOSE_AND_SCAFFOLDING.md §3.7's table, reused verbatim
R_PASS = 0.8              # retention: "owner right" row
B_PASS = 1.5              # benefit: "worth having" column
R_FLOOR = 0.5             # below this with B > 1: the puppeteering cell
PUPPET_SEEDS_MIN = 2      # §3.7: falsified in >= 2 of 3 seeds
GIVE_CON_RATIO_MIN = 2.0  # C-GIVE's C_on must be >= 2x A0's C_on, else the
                          # calibrator never showed on-hands competence
REF_MIN_FED = 0.8         # XL.01's reference bar, reused

LOG_PATH = "/data/so07_hand_logs_s{seed}.json"

_MODE_SALT = {"ref": 11, "a0": 12, "a1": 13, "a2": 14, "mis": 15, "give": 16}


# ── the hand's policies ─────────────────────────────────────────────────────
class _HandCtl:
    """One arm's hand. All placement goes through SO.06's certified `Hand`;
    all randomness through the hand's PRIVATE rng (the honest hand never
    draws from the action stream — SO.06's contract, and its control 1 is
    the detector that would catch a violation)."""

    def __init__(self, w: W0, mode: str, hrng: np.random.RandomState):
        self.w, self.mode, self.rng = w, mode, hrng
        self.half = _pool_half(w)
        self.hands: dict = {}
        self.life_drops = 0
        self.last_drop_d = -(10 ** 9)
        self.schedule: list = []          # mis: this life's drop decisions
        self.life_idx = 0
        self.drops_per_life: list = []    # measured, feeds C-MISPLACE
        self.n_drops = 0
        self.n_skipped = 0                # no legal position found

    def _hand(self, item: str) -> Hand:
        if item not in self.hands:
            self.hands[item] = Hand("owner", self.w, item, self.rng)
        return self.hands[item]

    def _edible(self, item: str) -> bool:
        dl = self.w.drives
        return dl.t >= dl._respawn_at[item]

    def _gift_item(self) -> str:
        if self.mode == "a2" and self.life_idx < GRADE_K:
            return "obj0"
        return "apple"

    def new_life(self, life_idx: int, mis_count: int = 0) -> None:
        self.life_idx = life_idx
        self.life_drops = 0
        self.last_drop_d = -(10 ** 9)
        if self.mode == "mis":
            self.schedule = sorted(
                int(v) for v in self.rng.choice(CAP_LIFE, size=mis_count,
                                                replace=False))

    def end_life(self) -> None:
        self.drops_per_life.append(self.life_drops)

    def _legal_drop_target(self):
        p = np.asarray(self.w.data.xpos[self.w.rover_bid][:2], dtype=float)
        for _ in range(DROP_TRIES):
            th = self.rng.uniform(0.0, 2.0 * np.pi)
            xy = p + DROP_RANGE_M * np.array([np.cos(th), np.sin(th)])
            if abs(xy[0]) <= 4.5 and abs(xy[1]) <= 4.5 and \
                    not _in_pool(xy, self.half):
                return xy
        return None

    def _drop(self, item: str, xy, contact: bool) -> None:
        h = self._hand(item)
        if contact:
            p = np.asarray(self.w.data.xpos[self.w.rover_bid][:2],
                           dtype=float)
            xy = p
        h.place((float(xy[0]), float(xy[1]), h.radius))
        self.life_drops += 1
        self.n_drops += 1
        self.last_drop_d = self._d

    def step(self, d: int) -> None:
        self._d = d
        if self.mode in ("a0", "ref"):
            return
        if self.mode == "mis":
            item = "apple"
            while self.schedule and d >= self.schedule[0]:
                if not self._edible(item):
                    return              # deferred: retried next decision
                self.schedule.pop(0)
                # position independent of Jack: uniform over the arena
                xy = None
                for _ in range(DROP_TRIES):
                    cand = self.rng.uniform(-4.5, 4.5, size=2)
                    if not _in_pool(cand, self.half):
                        xy = cand
                        break
                if xy is None:
                    self.n_skipped += 1
                    continue
                self._drop(item, xy, contact=False)
            return
        # need-contingent modes: a1, a2, give
        e = float(self.w.drives.state.e)
        if (e < E_GIFT and self.life_drops < GIFTS_MAX
                and d - self.last_drop_d >= GIFT_REFRACTORY):
            item = self._gift_item() if self.mode == "a2" else "apple"
            if not self._edible(item):
                return
            if self.mode == "give":
                self._drop(item, None, contact=True)
                return
            xy = self._legal_drop_target()
            if xy is None:
                self.n_skipped += 1
                return
            self._drop(item, xy, contact=False)

    def end_of_life_positions(self) -> dict:
        # SO.06's rule: log positions at each life's end, or the hand cannot
        # be separated from the drift w0._place deliberately never undoes.
        return self._hand("apple").end_of_life_positions()

    def logs(self) -> list:
        out = []
        for h in self.hands.values():
            out.extend(h.log)
        return sorted(out, key=lambda r: r["t"])


# ── the need-gated life ─────────────────────────────────────────────────────
def _sight_gids(w: W0) -> dict:
    """Floor food AND the apple: he may remember a gift he saw. XL.01's
    exclusion of the apple was about reachability of the platform home, not
    about perception — a dropped apple is on the floor and findable."""
    out = dict(_food_positions(w))
    try:
        out["apple"] = (int(w.model.geom("apple").id), None)
    except (KeyError, ValueError):
        pass
    return out


def _live(w: W0, diary, rng, hand: _HandCtl | None,
          ref_targets=None, record_sightings: bool = True) -> dict:
    """One life to CAP_LIFE or death — NOT to first feed (unlike XL.01):
    living past the first meal is where training happens. Need-gated:
    forage while e < FORAGE_E, rest while sated."""
    root = w.ix["root_dofadr"]
    foods = _sight_gids(w)
    half = _pool_half(w)
    explorer = _Explorer(rng, half)
    if ref_targets is not None:
        candidates = [np.array(p, dtype=float) for p in ref_targets]
    else:
        candidates = _recall_positions(diary, now=w.sim_seconds)
    blacklist: list = []
    parked_since = None
    best_dist, best_at = float("inf"), 0
    detour_until = -1
    ate0 = sum(w.drives.ate_total.values())
    apple0 = w.drives.ate_total.get("apple", 0)
    sighted_this_life = list(candidates)
    ttf_s, fed, died = CAP_S, False, False
    n_dec = 0

    for d in range(CAP_LIFE):
        if hand is not None:
            hand.step(d)
        xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        if not fed and sum(w.drives.ate_total.values()) > ate0:
            ttf_s, fed = d * SIM_S_PER_DECISION, True
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
        a = np.zeros(w.action_dim)
        a[4:6] = -1.0                      # adhesion OFF in both branches
        if float(w.drives.state.e) < FORAGE_E:
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
                    if dist < best_dist - 0.1:
                        best_dist, best_at = dist, d
                    elif d - best_at >= STUCK_N:
                        detour_until = d + DETOUR_N
                        explorer.wp = None
                        best_dist, best_at = float("inf"), d
            else:
                target = explorer.target(xy)
            wp = _route(xy, target, half)
            v = np.array([w.data.qvel[root], w.data.qvel[root + 1]],
                         dtype=float)
            a[6:8] = np.clip(KP * (wp - xy) - KD * v, -1.0, 1.0)
        # else: rest — sated, no goal, the world drains him back to hunger
        w.decide(a)
        n_dec = d + 1
        if w.died_this_decision:
            died = True
            break
    return {"ttf_s": float(ttf_s), "fed": fed, "died": died,
            "n_dec": n_dec,
            "apple_eats": int(w.drives.ate_total.get("apple", 0) - apple0),
            "n_sighted": len(sighted_this_life)}


# ── one arm, all three phases, one carried diary ────────────────────────────
def _apple_home(w: W0):
    m = w.model
    bid = int(m.body("apple").id)
    jadr = int(m.body_jntadr[bid])
    qadr = int(m.jnt_qposadr[jadr])
    dadr = int(m.jnt_dofadr[jadr])
    return qadr, dadr, w.data.qpos[qadr:qadr + 7].copy()


def _set_apple(w: W0, home, parked: bool) -> None:
    qadr, dadr, q = home
    if parked:
        w.data.qpos[qadr:qadr + 3] = (PARK_XY[0], PARK_XY[1], float(q[2]))
        w.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    else:
        w.data.qpos[qadr:qadr + 7] = q
    w.data.qvel[dadr:dadr + 6] = 0.0
    w.mujoco.mj_forward(w.model, w.data)


def _run_arm(wseed: int, mode: str, j0: float, alpha: float,
             mis_counts: list = None) -> dict:
    diary = EpisodicMemory()
    w = W0(seed=wseed, j0=j0, alpha=alpha, lethal=True, diary=diary)
    floor_home = _widen(_food_home(w), wseed)
    apple_home = _apple_home(w)
    hrng = np.random.RandomState(wseed * 60013 + _MODE_SALT[mode])
    rng = np.random.RandomState(wseed * 45007 + _MODE_SALT[mode])
    ctl = _HandCtl(w, mode, hrng)
    ref_targets = None
    if mode == "ref":
        ref_targets = [p for _, (_, p) in sorted(_food_positions(w).items())]
    lives = {"train": [], "on": [], "off": []}
    eol_positions = []
    li = 0
    phases = (("train", N_TRAIN, E0_TRAIN, True),
              ("on", N_ON, E0_SCORE, True),
              ("off", N_OFF, E0_SCORE, False))
    if mode == "ref":                      # reference: scoring shape only
        phases = (("off", N_OFF, E0_SCORE, False),)
    for phase, n, e0, hand_on in phases:
        for _ in range(n):
            w.drives.state = drives.DriveState(e=e0)
            w.drives._respawn_at = {k: 0.0 for k in w.drives._respawn_at}
            _reset_food(w, floor_home)
            _set_apple(w, apple_home, parked=not hand_on)
            ctl.new_life(li, mis_counts[li] if (mode == "mis" and mis_counts
                                                and li < len(mis_counts))
                         else 0)
            r = _live(w, diary, rng, ctl if hand_on else None,
                      ref_targets=ref_targets,
                      record_sightings=(mode != "ref"))
            r["phase"] = phase
            lives[phase].append(r)
            if hand_on:
                ctl.end_life()
            eol_positions.append(ctl.end_of_life_positions())
            if not r["died"]:
                _drain(w)
            li += 1
    return {"lives": lives, "ctl": ctl, "eol": eol_positions,
            "log": ctl.logs()}


# ── metrics ─────────────────────────────────────────────────────────────────
def _C(lives: list) -> float:
    ttfs = [r["ttf_s"] for r in lives]
    return 1.0 / max(float(np.median(ttfs)), TTF_FLOOR_S)


def _arm_metrics(arm: dict, prefix: str) -> dict:
    tr, on, off = arm["lives"]["train"], arm["lives"]["on"], arm["lives"]["off"]
    c_on, c_off = _C(on), _C(off)
    out = {
        f"{prefix}_c_on": c_on,
        f"{prefix}_c_off": c_off,
        f"{prefix}_r": c_off / c_on if c_on > 0 else float("nan"),
        f"{prefix}_train_died": float(np.mean([r["died"] for r in tr])),
        f"{prefix}_train_dec": float(np.sum([r["n_dec"] for r in tr])),
        f"{prefix}_off_fed_lives": float(sum(r["fed"] for r in off)),
        f"{prefix}_gift_drops": float(arm["ctl"].n_drops),
        f"{prefix}_gift_skipped": float(arm["ctl"].n_skipped),
        f"{prefix}_gift_eats": float(sum(
            r["apple_eats"] for ph in ("train", "on")
            for r in arm["lives"][ph])),
    }
    return out


_MEMO: dict = {}
_CTL_MEMO: dict = {}


def _flat(memo: dict, keys=SEEDS) -> dict:
    out = dict(memo[keys[0]]) if len(keys) == 1 else {}
    for s in keys:
        for k, v in memo[s].items():
            out[f"{k}_s{s}"] = v
    if len(keys) > 1:
        out.update(memo[keys[0]])
    return out


def _measure(seed: int) -> dict:
    j0, alpha, prov = _calibration()
    m: dict = {"calibrated": float(j0 is not None), **prov}
    if j0 is None:
        return m
    wseed = seed + WORLD_BASE
    m["world_seed"] = float(wseed)

    ref = _run_arm(wseed, "ref", j0, alpha)
    m["ref_fed_frac"] = float(np.mean(
        [r["fed"] for r in ref["lives"]["off"]]))

    a0 = _run_arm(wseed, "a0", j0, alpha)
    a1 = _run_arm(wseed, "a1", j0, alpha)
    a2 = _run_arm(wseed, "a2", j0, alpha)
    m.update(_arm_metrics(a0, "a0"))
    m.update(_arm_metrics(a1, "a1"))
    m.update(_arm_metrics(a2, "a2"))
    c0 = m["a0_c_off"]
    for p in ("a1", "a2"):
        m[f"{p}_b"] = m[f"{p}_c_off"] / c0 if c0 > 0 else float("nan")
    _CTL_MEMO[seed] = {"j0": j0, "alpha": alpha,
                       "mis_counts": list(a1["ctl"].drops_per_life),
                       "a0_c_on": m["a0_c_on"], "c0": c0}
    _dump_logs(seed, {"a1": a1, "a2": a2})
    return m


def _measure_control(seed: int) -> dict:
    cache = _CTL_MEMO[seed]
    wseed = seed + WORLD_BASE
    mis = _run_arm(wseed, "mis", cache["j0"], cache["alpha"],
                   mis_counts=cache["mis_counts"])
    give = _run_arm(wseed, "give", cache["j0"], cache["alpha"])
    c = {}
    c.update({f"c_{k}": v for k, v in _arm_metrics(mis, "mis").items()})
    c.update({f"c_{k}": v for k, v in _arm_metrics(give, "give").items()})
    c0 = cache["c0"]
    c["c_mis_b"] = c["c_mis_c_off"] / c0 if c0 > 0 else float("nan")
    c["c_give_b"] = c["c_give_c_off"] / c0 if c0 > 0 else float("nan")
    c["c_give_con_ratio"] = (c["c_give_c_on"] / cache["a0_c_on"]
                             if cache["a0_c_on"] > 0 else float("nan"))
    _dump_logs(seed, {"mis": mis, "give": give}, merge=True)
    return c


def _dump_logs(seed: int, arms: dict, merge: bool = False) -> None:
    """The measured hand logs — SO.09's replay re-buy reads these."""
    path = LOG_PATH.format(seed=seed)
    data = {}
    if merge:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    for name, arm in arms.items():
        data[name] = {"log": arm["log"], "end_of_life_positions": arm["eol"]}
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass                                # /data absent: logs are a receipt,
    #                                         never a gate


def _experiment(seed: int) -> dict:
    for s in SEEDS:
        if s not in _MEMO:
            _MEMO[s] = _measure(s)
    out = _flat(_MEMO)
    out.update(_MEMO[seed])
    # the puppeteering count, computed here so the row carries it explicitly
    pup = sum(1 for s in SEEDS
              if _MEMO[s].get("a1_r", 9.9) < R_FLOOR
              and _MEMO[s].get("a1_b", 0.0) > 1.0)
    out["puppeteering_seeds"] = float(pup)
    return out


def _control(seed: int) -> dict:
    for s in SEEDS:
        if s not in _MEMO:
            _MEMO[s] = _measure(s)
    memo: dict = {}
    for s in SEEDS:
        if s not in memo:
            memo[s] = _measure_control(s) if s in _CTL_MEMO else {}
    out = _flat(memo)
    out.update(memo[seed])
    return out


def _check(m: dict, c: dict):
    if m.get("calibrated", 0.0) != 1.0:
        return Status.VOID              # PS.01 supplied no usable j0/alpha
    for s in SEEDS:
        if m.get(f"ref_fed_frac_s{s}", 0.0) < REF_MIN_FED:
            return Status.VOID          # the rig cannot produce the behaviour
        if m.get(f"a0_off_fed_lives_s{s}", 0.0) <= 0.0:
            return Status.VOID          # C0 degenerate: the null never fed
        if m.get(f"a1_gift_drops_s{s}", 0.0) <= 0.0:
            return Status.VOID          # the trigger never armed
        if m.get(f"a1_gift_eats_s{s}", 0.0) <= 0.0:
            return Status.VOID          # the channel was never exercised:
            # nothing the hand left was ever found, so the claim went untested
        # C-GIVE must calibrate what failure looks like, on every seed
        if not (c.get(f"c_give_r_s{s}", 9.9) < R_FLOOR
                and c.get(f"c_give_con_ratio_s{s}", 0.0)
                >= GIVE_CON_RATIO_MIN):
            return Status.VOID          # the instrument cannot see
            # puppeteering; no other number here means anything
        # law 2: C-MISPLACE clearing the claim's own bar → measured nothing
        if (c.get(f"c_mis_r_s{s}", 0.0) >= R_PASS
                and c.get(f"c_mis_b_s{s}", 0.0) >= B_PASS):
            return Status.VOID
    # the claim: worst-seed, thresholds verbatim from §3.7's table
    return bool(all(m.get(f"a1_r_s{s}", 0.0) >= R_PASS
                    and m.get(f"a1_b_s{s}", 0.0) >= B_PASS for s in SEEDS))


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.07"], _experiment, _check,
                    control_fn=_control, ledger=ledger or Ledger())


if __name__ == "__main__":
    # Pilot lanes (SH.01: reference first; T3.09: counts only, no eval
    # metric). The seed argument is the WORLD seed directly — worlds 0-2 are
    # design data, the recording runs on 3-5.
    mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
    wseed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    j0, alpha, _prov = _calibration()
    print(f"calibration j0={j0} alpha={alpha}")
    t0 = time.time()
    if mode == "ref":
        arm = _run_arm(wseed, "ref", j0, alpha)
        for i, r in enumerate(arm["lives"]["off"]):
            print(f"  ref life {i}: fed {r['fed']} died {r['died']} "
                  f"dec {r['n_dec']}")
    elif mode == "aliveness":
        # TRAIN-phase counts for the null, the claim arm and the puppeteer:
        # does A0 die in-window, does A1's trigger fire, are gifts FOUND,
        # does C-GIVE eat by contact? Counts only.
        for md in ("a0", "a1", "give"):
            arm = _run_arm(wseed, md, j0, alpha)
            tr = arm["lives"]["train"]
            print(f"  {md}: died {sum(r['died'] for r in tr)}/{len(tr)} "
                  f"dec {sum(r['n_dec'] for r in tr)} "
                  f"drops {arm['ctl'].n_drops} "
                  f"skipped {arm['ctl'].n_skipped} "
                  f"gift_eats {sum(r['apple_eats'] for r in tr)} "
                  f"sighted {[r['n_sighted'] for r in tr]} "
                  f"fed {[int(r['fed']) for r in tr]}")
    elif mode == "mis":
        arm1 = _run_arm(wseed, "a1", j0, alpha)
        counts = list(arm1["ctl"].drops_per_life)
        arm = _run_arm(wseed, "mis", j0, alpha, mis_counts=counts)
        print(f"  a1 drops/life {counts}")
        print(f"  mis drops {arm['ctl'].n_drops} "
              f"gift_eats {sum(r['apple_eats'] for ph in ('train', 'on') for r in arm['lives'][ph])}")
    print(f"wall {time.time() - t0:.1f}s")
