"""eye_forager.py — J12 (2): the forager gate for the NEW eye (ORDERS J12, ruling 37).

PLAN stage 1, parts "Eye" + "Learning to eat" (the learnability proof), step 2
(combine). Prior: PUBLISHED_PRIORS.md §12 (a first eye of distance-and-material
rays); LAB `experiments/eye_streak.py` (`bddffac4`): at d0 one level row of
128 azimuths has floor food in sight .909 of decisions against .528 for the
16 rays. `jack_env(eye="row128")` built and proven in `9e4031bb`.

NOT A LEARNER, NOT A LEDGER SPEC. A fixed, hand-written ruler held against the
world, scored-and-ineligible, never a part of him. Pass or fail it is an
adult-taught ceiling: it says floor food is FINDABLE from this sense, not that
a brain learns to find it. Results go to `docs/research/WORLD_CARD.md`.

=============================================================================
THE GATE — frozen in this commit, before any gate run (11-17).
=============================================================================
Ruler   row f2's forager (`world_card.HandForager2`, F2_BAN_S = inf, frozen
        11-15 20:12), changed ONLY in that it reads the new eye:
          * N_RAYS 16 -> 128, ray k at azimuth 2*pi*k/128 in the world frame
            (row ray 8j IS old ray j, `test_jack_env_eye`);
          * observation indices after vision shift by +224 (vision 32 -> 256);
          * the blob rule keeps its ANGLES, not its ray counts: "a small
            object" = a run of <= 2 x 22.5 deg of one texture (<= 2*s rays,
            s = 128/16 = 8) whose both neighbours differ; the run-growth cap
            4 rays -> 4*s rays (90 deg). At 16 rays a 2-ray run cap would
            let a food that fills 3 of 128 rays (8.4 deg) read as a wall —
            that would change the ruler, not the eye.
        Nothing else: same targets, bans, cooldowns, wander, constants, rng.
        Proof of "nothing else": `tests/test_eye_forager.py` — at s = 1 on
        the 16-ray eye this class's actions equal HandForager2's bit for bit
        over whole lives (and a must-fail control: a 1-ray change differs).
World   `jack_env.JackEnv(seed, difficulty=d, eye="row128",
        lazy_contact_audio=True)` — row f2's world (default options) at
        d = 0.75 (d0, ruling 22) and d = 1 (difficulty None, today's world).
        lazy_contact_audio = identical world (ruling 34), for speed. The
        shipped calibration (the scar of 11-17: a hand cal kills the body
        before it reaches eye height).
Seeds   FRESH 241..260 (20 worlds; never run by any row, pilot or LAB;
        153/154 smoke only). 10 consecutive lives a seed in one persisting
        world; life cap 9,000 decisions, censored lives respawn in the same
        world — `world_card.run_seed`'s bookkeeping, unchanged.
Rows    at each d in {0.75, 1}:
          f3  the ruler above
          g3  CONTROL that must fail: f3 with its 128 (distance, texture)
              pairs permuted by a fresh permutation every decision
          b3  white random, uniform(-1,1)^8 per decision (row b's policy,
              rng seed*6553+11) — the per-seed floor the bar compares with
Bar     UNMOVED (LF_04_FAIL.md, world_card.score): on a seed, mean floor
        meals per life over its 10 lives >= 5.0 AND > b3's mean on the same
        seed; f3 PASSES at d iff that holds on >= 18 of 20 seeds. g3 is
        scored by the same function and must hold on < 18; if g3 passes,
        f3's pass at that d is VOID (food came from something not sight).
        Judged SEPARATELY at d0 and at d = 1; both are reported.
Extension  NONE. One run of 20 seeds per row per d; a miss is a miss.
Where   2 Kaggle CPU kernels (one per d: f3, g3, b3 x 20 seeds = 60 cells,
        4 workers), rank 2 in ORDERS' Kaggle ranking (after J11's card).
        Only the 1-life smoke on 153 runs on the box.
Cost    smoke (box, 11-17 08:3x, seed 153, d 0.75, 1 life each): f3 9,000
        decisions (cap) 19 meals 204 CPU-s; g3 9,000 (cap) 10 meals 214
        CPU-s; b3 2,227 decisions 3 meals 50 CPU-s (~44 dec/CPU-s). Worst
        case at d0 (every f3/g3 life capped): 4.0 M decisions ~ 18.7
        CPU-h at Kaggle's ~60/s = ~4.7 h on 4 workers; d = 1 less.
Disclosed before the run: at d0 the blind twin ate 10 meals in its one
        smoke life (tuning seed). A g3 pass at d0 — f3 VOID there — is a
        live outcome the bar was not moved to avoid; d = 1 is judged anyway.

=============================================================================
GATE 2 — THE SAME GATE ON THE BODY-FRAME EYE (ruling 38, ORDERS J12 (2)).
Declared 2026-11-18 07:4x UTC, in this commit, before any gate-2 cell runs.
=============================================================================
Why     Both gate-1 rows were earned on an eye cast at WORLD azimuths — a free
        compass. Ruling 38: nothing is believed on the compass eye.
Does the ruler assume the world frame?  YES (ORDERS J12 (1)), three places in
        `EyeForager._act`: (i) a blob at ray angle `ang` is placed at
        rel = (cos ang, sin ang) * dist (the two `rel =` lines) and the push
        is a[6:8] = rel/|rel| — but W0's drive is a force along WORLD x, y
        (`w0.py` decide: xfrc_applied[rover, :2]); (ii) the wander push
        a[6:8] = (cos, sin) of ray_ang[heading], same assumption; (iii) the
        target is dead-reckoned with rel -= v * DT, v = proprio's root
        velocity, WORLD x, y. With a body-frame eye and none of these changed
        the ruler pushes along a compass it cannot see: the body is born at
        yaw 0 and under random actions sits a median 81 deg from it
        (`tests/test_jack_env_eye_frame.py` (d)).
The ONE change, said before the run (ORDERS: "change ONLY the action mapping
        needed to steer from body-frame rays"): the mapping is put in the
        WORLD'S INTERFACE, not in the ruler — `JackEnv(motor_frame="body")`:
        action[6:8] is a push (forward, left) along the body's own heading and
        proprio's two root velocities are (forward, left). The ruler's code is
        gate 1's line for line (`EyeForager`, same class, same constants, same
        rng); read in the body frame its three assumptions hold again. It is
        NOT given yaw: a ruler that read the heading from the simulator would
        put the compass back by another door. Known and left as is: `rel` is
        not turned when the body yaws between decisions (he has no turn-rate
        sense); the ruler's own re-sighting rule (same texture within 1.5 m)
        is what it has.
World   `JackEnv(seed, difficulty=d, eye="row128", eye_frame="body",
        motor_frame=<row>, lazy_contact_audio=True, native_substep=True)` —
        gate 1's world (default options, shipped calibration) through the
        two identical-world speed options (ORDERS 11-18), d = 0.75 and d = 1.
Seeds   FRESH 261..280 (ORDERS J12 (2)); 10 lives a seed, cap 9,000 — gate 1's.
Rows    at each d:
          f4  the ruler, body-frame eye, body-frame motor      <- THE ROW
          g4  CONTROL that must fail: f4 with its rays permuted each decision
          b4  white random (same policy and rng as b3) in f4's world — the
              per-seed floor
          h4  REPORTED ONLY, no bar clause, decides nothing: the ruler with
              the body-frame eye and the WORLD-frame motor (what an env with
              only `eye_frame="body"` gives). Predicted before the run: it
              misses. It is here so that "does the motor frame matter?" is a
              measurement, not an argument (scored by the same function
              against b4, labelled as such).
Bar     UNMOVED: mean floor meals/life >= 5.0 AND > b4's mean on the same
        seed, on >= 18 of 20 seeds; g4 must hold on < 18, else f4 is VOID at
        that d. d0 and d = 1 judged separately, both reported.
Extension  NONE. A miss is a miss — and then J15 becomes "say why" (ORDERS).
Where   2 Kaggle CPU kernels (one per d, 80 cells, 4 workers), rank 2.
Cost    gate 1 at d = 1 took one session under its 4.7 h worst case with 60
        cells and no native_substep (2.4x); 80 cells with it: est 5 h.

=============================================================================
GATE 3 — f5, A RULER THE CURRENT ROW DETERMINES (ruling 50, ORDERS J15 (1)-(2)).
Declared 2026-11-19 07:2x UTC, in this commit, before any gate-3 cell runs.
=============================================================================
Why     E2b asks a memoryless net to copy a teacher. f4 is not copyable from
        the row: it holds a target on 0.551 of decisions and otherwise steers
        along a random ray held 25 decisions (PRIORS §17, imitation gap 0.88
        -> <= 0.11). Ruling 50: first a ruler whose every action IS a function
        of the current body-frame row128, then gate 2's bar on it.
Ruler   `f5_act(row)` below — no remembered target, no draw, no counter, no
        energy, no velocity. Blobs by gate 1's rule (`blobs_n`, unchanged),
        kept only if (i) their apparent width, rays x ray pitch x distance,
        is >= F5_THIN_M = 0.11 m and (ii) their nearest ray is >= F5_FRONT_M
        = 0.10 m nearer than BOTH flanking rays; push at the nearest kept
        blob; none -> push at the middle of the widest open gap (rays >= 0.70
        of the row's farthest). Proof it is a function of the row:
        `experiments/tests/test_f5.py` (same row twice; a shuffled-order
        replay of a life's rows; f4 under the same replay must differ).
What it cannot do, said before the run: tell food from any other small
        thing (food's texture differs from world to world — geom ids shift
        with the layout, `w0.py:225` — and f4 learned it by memory), or give
        a thing up. It parks on what it reaches; a parked food regrows
        under him, so a seed can pass by SITTING. The bar counts meals and is
        not moved for that; WORLD_CARD says per row how many meals came from
        one food (floor_bouts and floor_eats_all are logged per life).
How it was shaped (tuning seeds 153/154 ONLY, d0, one life cut at 3,000
        decisions, box): v1 nearest blob of any size: seed 153 0 meals, dead
        of hunger at 2,985 (2,323 decisions on the two ladder rails). v2
        + (i): 153 1 meal, 154 2 (parked on a strip of far wall seen between
        two things 2,283 decisions; on a non-food ball 1,597). v3 + (ii),
        THE RULER: 153 9 meals (5 in a row from one parked food), 154 2
        (parked on the welded block 1,743). Three variants, no fourth.
        Also disclosed: seed 281's world was BUILT once on the box to print
        its geom table (no policy stepped in it; what was read: ngeom 36 vs
        39/40, so food's texture is not a constant).
World   gate 2's f4 world exactly: `JackEnv(seed, difficulty=0.75,
        eye="row128", eye_frame="body", motor_frame="body",
        lazy_contact_audio=True, native_substep=True)`. d0 ONLY (ruling 51).
Seeds   FRESH 281..300 (ORDERS J15 (2)); 10 lives a seed, cap 9,000.
Rows    f5  the ruler                                           <- THE ROW
        g5  CONTROL that must fail: f5 fed its row with the 128 (distance,
            texture) pairs permuted by a fresh permutation every decision
        b5  white random (b3's policy and rng) in the same world — the floor
Bar     UNMOVED: mean floor meals/life >= 5.0 AND > b5's mean on the same
        seed, on >= 18 of 20 seeds; g5 must hold on < 18, else f5 is VOID.
Extension  NONE. A miss is a miss, and it is a miss OF THIS HAND RULER: it
        says a simple memoryless rule does not meet the bar, not that none
        can (a lower bound, ruling 51).
Where   one Kaggle CPU kernel (60 cells, 4 workers), rank 2. Est 5 h (gate
        2's 80 cells took one session; f5 lives run to the cap more often).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import world_card as WC  # noqa: E402

GATE_SEEDS = tuple(range(241, 261))
GATE_DS = (0.75, 1.0)
GATE_ROWS = ("f3", "g3", "b3")
# gate 2 (body-frame eye), declared in the docstring above
GATE2_SEEDS = tuple(range(261, 281))
GATE2_ROWS = ("f4", "g4", "h4", "b4")        # longest first
GATE2_MOTOR = {"f4": "body", "g4": "body", "b4": "body", "h4": "world",
               "f5": "body", "g5": "body", "b5": "body"}   # gate 3: f4's world
# gate 3 (f5, a ruler the current row determines), declared in the docstring
GATE3_SEEDS = tuple(range(281, 301))
GATE3_ROWS = ("f5", "g5", "b5")
GATE3_D = 0.75
EYE = "row128"
N_ROW = 128
SMOKE_SEED = 153
OUT = Path("/data/jack-logs/eye_forager")
KERNEL_PINS = ["gymnasium==1.4.0", "mujoco==3.17.0"]
WORKERS = 4
REMOTE_TIMEOUT_S = int(9.5 * 3600)


def blobs_n(vis: np.ndarray, n: int, s: int, ray_ang: np.ndarray,
            with_len: bool = False):
    """world_card.blobs for n rays with run limits scaled by s (angles kept).
    At n = 16, s = 1 this is world_card.blobs line for line. `with_len`
    (f5 only): each blob also carries its run length in rays; the default
    output is unchanged."""
    d, t = vis[0::2], vis[1::2]
    hit = d < 1.0
    same = lambda a, b: abs(a - b) < 1e-4  # noqa: E731
    out = []
    seen = np.zeros(n, bool)
    for k in range(n):
        if not hit[k] or seen[k]:
            continue
        run = [k]
        j = (k + 1) % n
        while hit[j] and same(t[j], t[k]) and j != k and len(run) < 4 * s:
            run.append(j); j = (j + 1) % n
        i = (k - 1) % n
        while hit[i] and same(t[i], t[k]) and i not in run and len(run) < 4 * s:
            run.insert(0, i); i = (i - 1) % n
        for r in run:
            seen[r] = True
        if len(run) > 2 * s:
            continue
        left, right = (run[0] - 1) % n, (run[-1] + 1) % n
        if (hit[left] and same(t[left], t[k])) or (hit[right] and same(t[right], t[k])):
            continue
        ang = math.atan2(np.mean([math.sin(ray_ang[r]) for r in run]),
                         np.mean([math.cos(ray_ang[r]) for r in run]))
        out.append((ang, float(min(d[r] for r in run)) * WC.RAY_RANGE, float(t[k]))
                   + ((len(run), float(min(d[left], d[right])) * WC.RAY_RANGE)
                      if with_len else ()))
    return out


class EyeForager(WC.HandForager2):
    """HandForager2 reading an n-ray world-frame eye. Only `act` is rewritten,
    with world_card's module constants made per-instance; every learning rule
    is inherited or copied unchanged (identity proven at n = 16)."""

    def __init__(self, seed, shuffle_rays: bool = False, n_rays: int = N_ROW):
        self.n = int(n_rays)
        if self.n % WC.N_RAYS:
            raise ValueError("n_rays must be a multiple of 16")
        self.s = self.n // WC.N_RAYS
        self.ray_ang = np.array([2 * math.pi * k / self.n for k in range(self.n)])
        sh = 2 * self.n - 2 * WC.N_RAYS
        self.i_v = [WC.I_VX + sh, WC.I_VY + sh]
        self.i_e = WC.I_E + sh
        super().__init__(seed, shuffle_rays)

    def act(self, o):
        for k in list(self.banned):        # HandForager2.act, verbatim
            self.ban_until[k] = self.clock + WC.F2_BAN_S
            self.miss.pop(k, None)
            self.food_miss.pop(k, None)
        self.banned.clear()
        return self._act(o)

    def _act(self, o):                     # HandForager.act with n, s, indices
        n = self.n
        o = np.asarray(o, dtype=float)
        vis = o[:2 * n].copy()
        if self.shuffle:
            p = self.rng.permutation(n)
            vis = vis.reshape(n, 2)[p].reshape(-1)
        v = o[self.i_v]
        e = o[self.i_e]
        self.clock += WC.DT
        ate = self.prev_e is not None and e - self.prev_e > 0.03
        self.prev_e = e

        if self.target is not None:
            k = self._key(self.target["tex"])
            if ate and np.linalg.norm(self.target["rel"]) < 0.6:
                self.food.add(k)
                self.food_miss[k] = 0
                self.cool[k] = self.clock + WC.RESPAWN_FLOORFOOD_S
                self.target = None
            elif ate:
                pass
            else:
                self.target["rel"] = self.target["rel"] - v * WC.DT
                self.target["age"] += 1
                if np.linalg.norm(self.target["rel"]) < 0.15:
                    self.contact_wait += 1
                if self.contact_wait > 10:
                    if k in self.food:
                        self._food_missed(k)
                        self.cool[k] = self.clock + 30.0
                    else:
                        self.miss[k] = self.miss.get(k, 0) + 1
                        if self.miss[k] >= 2:
                            self.banned.add(k)
                        else:
                            self.cool[k] = self.clock + WC.RESPAWN_FLOORFOOD_S
                    self.target = None
                elif self.target["age"] > 150:
                    if k in self.food:
                        self._food_missed(k)
                        self.cool[k] = self.clock + 20.0
                    else:
                        self.miss[k] = self.miss.get(k, 0) + 1
                        if self.miss[k] >= 2:
                            self.banned.add(k)
                        else:
                            self.cool[k] = self.clock + WC.RESPAWN_FLOORFOOD_S
                    self.target = None

        seen = blobs_n(vis, n, self.s, self.ray_ang)
        if self.target is not None:
            best = None
            for ang, dist, tex in seen:
                if self._key(tex) != self._key(self.target["tex"]):
                    continue
                rel = np.array([math.cos(ang), math.sin(ang)]) * dist
                err = np.linalg.norm(rel - self.target["rel"])
                if best is None or err < best[0]:
                    best = (err, rel)
            if best is not None and best[0] < 1.5:
                self.target["rel"] = best[1]
        if self.target is None:
            cand = [(self._prio(tex) / max(dist, 0.2), ang, dist, tex)
                    for ang, dist, tex in seen if self._prio(tex) > 0]
            if cand:
                _, ang, dist, tex = max(cand)
                self.target = {"tex": tex, "age": 0,
                               "rel": np.array([math.cos(ang), math.sin(ang)]) * dist}
                self.contact_wait = 0

        a = WC.NEUTRAL.copy()
        if self.target is not None:
            r = self.target["rel"]
            nr = np.linalg.norm(r)
            a[6:8] = r / nr if nr > 1e-6 else 0.0
            self.heading, self.hold = None, 0
            return a
        d = vis[0::2] * WC.RAY_RANGE
        if self.heading is None or self.hold <= 0 or d[self.heading] < 0.8:
            open_ = np.flatnonzero(d >= np.percentile(d, 70))
            self.heading = int(self.rng.choice(open_))
            self.hold = 25
        self.hold -= 1
        a[6:8] = [math.cos(self.ray_ang[self.heading]), math.sin(self.ray_ang[self.heading])]
        return a


# ── J15 (1): f5 — a ruler the CURRENT row determines (ruling 50) ─────────────
F5_GAP_FRAC = 0.70      # "open" = a ray reading >= 70 % of the row's farthest
F5_THIN_M = 0.11        # a blob narrower than this (rays x ray pitch x distance)
#                         is a post, not a meal: every floor food is >= 0.14 m
#                         across, a ladder rail 0.08 m (w0's geom table)
F5_FRONT_M = 0.10       # an object stands IN FRONT of what flanks it: its nearest
#                         ray is this much nearer than both neighbours' (a strip
#                         of far wall seen between two things is not an object)


def f5_act(row, s: int = N_ROW // WC.N_RAYS, thin_m: float = F5_THIN_M,
           front_m: float = F5_FRONT_M) -> np.ndarray:
    """Action as a pure function of ONE body-frame row (2n floats: distance,
    texture per ray). No state, no draw, no clock, no energy, no velocity.
      * small objects in the row (gate 1's blob rule, `blobs_n`, unchanged),
        less those whose apparent width is under `thin_m`: push straight at
        the NEAREST (ties: the lowest bearing);
      * none left: push at the middle of the WIDEST open gap — the longest
        circular run of rays reading >= F5_GAP_FRAC of the row's farthest ray
        (ties: the run starting at the lowest ray).
    It cannot know which texture is food (f4 learned that by memory, and the
    food's texture differs from world to world) and never gives a blob up."""
    vis = np.asarray(row, dtype=float)
    n = vis.shape[0] // 2
    pitch = 2 * math.pi / n
    ray_ang = pitch * np.arange(n)
    a = WC.NEUTRAL.copy()
    seen = [b for b in blobs_n(vis, n, s, ray_ang, with_len=True)
            if b[3] * pitch * b[1] >= thin_m and b[4] - b[1] >= front_m]
    if seen:
        ang = min(seen, key=lambda b: (b[1], b[0] % (2 * math.pi)))[0]
        return a_set(a, ang)
    d = vis[0::2]
    open_ = d >= F5_GAP_FRAC * d.max()
    if open_.all():
        return a_set(a, 0.0)
    best_len, best_start = 0, 0
    for k in range(n):
        if open_[k] and not open_[k - 1]:          # a run starts here
            L = 1
            while open_[(k + L) % n]:
                L += 1
            if L > best_len:
                best_len, best_start = L, k
    return a_set(a, ray_ang[best_start] + 0.5 * (best_len - 1) * pitch)


def a_set(a: np.ndarray, ang: float) -> np.ndarray:
    a[6:8] = [math.cos(ang), math.sin(ang)]
    return a


class RowForager:
    """f5 as a policy object. Holds NOTHING between decisions: `act` hands the
    row to `f5_act`. The shuffled twin (g5) permutes the row's (distance,
    texture) pairs with its own rng first — the draw belongs to the control's
    blindfold, not to the ruler."""

    def __init__(self, seed, shuffle_rays: bool = False, n_rays: int = N_ROW):
        self.n = int(n_rays)
        self.s = self.n // WC.N_RAYS
        self.rng = np.random.RandomState(seed * 6553 + 23) if shuffle_rays else None

    def new_life(self):
        pass

    def act(self, o):
        vis = np.asarray(o, dtype=float)[:2 * self.n]
        if self.rng is not None:
            vis = vis.reshape(self.n, 2)[self.rng.permutation(self.n)].reshape(-1)
        return f5_act(vis, self.s)


def make_policy(row: str, seed: int):
    if row in ("f3", "f4", "h4"):
        return EyeForager(seed)
    if row in ("g3", "g4"):
        return EyeForager(seed, shuffle_rays=True)
    if row == "f5":
        return RowForager(seed)
    if row == "g5":
        return RowForager(seed, shuffle_rays=True)
    if row in ("b3", "b4", "b5"):
        return WC.WhiteRandom(seed)
    raise ValueError(f"row {row!r} unknown")


def make_env(seed: int, d: float, eye: str = EYE, row: str = "f3"):
    from experiments.jack_env import JackEnv
    if row in GATE2_MOTOR:                         # gate 2: body-frame eye
        return JackEnv(seed=seed, difficulty=(None if d == 1.0 else d), eye=eye,
                       eye_frame="body", motor_frame=GATE2_MOTOR[row],
                       lazy_contact_audio=True, native_substep=True)
    return JackEnv(seed=seed, difficulty=(None if d == 1.0 else d), eye=eye,
                   lazy_contact_audio=True)


def run_seed(row: str, seed: int, d: float, lives: int = WC.LIVES,
             cap: int = WC.LIFE_CAP_DECISIONS, policy=None) -> list:
    """world_card.run_seed's life loop, unchanged, on the row128 world.
    `policy` (J15 E2b): drive `row`'s world with another policy object
    (new_life() + act(o)); None = the row's own policy, as before."""
    env = make_env(seed, d, row=row)
    o, info = env.reset(seed=seed)
    pol = make_policy(row, seed) if policy is None else policy
    out = []
    for life in range(lives):
        pol.new_life()
        ate0 = dict(info["ate_total"])
        rew, n, t0 = 0.0, 0, env.w.sim_seconds
        cause = "censored"
        foods = tuple(env.floor_foods)
        prev = {k: info["ate_total"].get(k, 0) for k in foods}
        last = {k: -10 ** 9 for k in foods}
        eats_all = bouts = 0
        while n < cap:
            o, r, term, trunc, info = env.step(pol.act(o))
            rew += r
            n += 1
            for k in foods:
                c = info["ate_total"].get(k, 0)
                if c > prev[k]:
                    eats_all += c - prev[k]
                    if n - last[k] > WC.BOUT_GAP:
                        bouts += 1
                    last[k] = n
                prev[k] = c
            if term:
                cause = info["death_cause"]
                break
        ate = {k: info["ate_total"][k] - ate0.get(k, 0) for k in info["ate_total"]}
        out.append({"row": row, "d": d, "seed": seed, "life": life,
                    "lifetime_s": round(env.w.sim_seconds - t0, 1),
                    "floor_meals": int(ate.get("obj0", 0) + ate.get("obj1", 0)),
                    "apples": int(ate.get("apple", 0)),
                    "cause": cause, "reward": round(rew, 5), "decisions": n,
                    "floor_eats_all": int(eats_all), "floor_bouts": int(bouts)})
        if not term:
            env.w.respawn()
            env._pending_obs = None
        o, info = env.reset()
    return out


def gate_cells(d: float, gate: int = 1) -> list:
    # longest first (f3, then g3, then b3) so the pool's tail is short
    if gate == 3:
        if d != GATE3_D:
            raise ValueError("gate 3 is judged at d0 = 0.75 only (ruling 51)")
        return [(row, s, d) for row in GATE3_ROWS for s in GATE3_SEEDS]
    if gate == 2:
        return [(row, s, d) for row in GATE2_ROWS for s in GATE2_SEEDS]
    return [(row, s, d) for row in GATE_ROWS for s in GATE_SEEDS]


def run_pool(cells: list, outdir, workers: int = WORKERS) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    todo, live, res = list(cells), [], {}
    while todo or live:
        while todo and len(live) < workers:
            row, seed, d = todo.pop(0)
            f = outdir / f"{row}_d{int(round(100 * d)):03d}_s{seed}.jsonl"
            err = open(str(f) + ".err", "w")
            live.append((row, seed, d, f, err, subprocess.Popen(
                [sys.executable, "-m", "experiments.eye_forager", "cell", row,
                 str(seed), str(d), str(f)],
                cwd=REPO, stdout=subprocess.DEVNULL, stderr=err)))
        time.sleep(2)
        for item in list(live):
            row, seed, d, f, err, p = item
            if p.poll() is None:
                continue
            live.remove(item)
            err.close()
            lives = ([json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                     if f.exists() else [])
            rec = {"row": row, "seed": seed, "d": d, "lives": lives, "rc": p.returncode}
            if p.returncode != 0:
                rec["error"] = Path(str(f) + ".err").read_text()[-800:]
            res[f"{row}_d{d}_s{seed}"] = rec
            print(f"{row} d{d} s{seed}: rc {p.returncode}, {len(lives)} lives", flush=True)
    return res


_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.eye_forager import run_pool
r = run_pool(__CELLS__, os.environ["JACK_OUT"])
json.dump({"cells": r}, open(os.path.join(os.environ["JACK_OUT"], "eye_gate.json"), "w"))
print("DONE", flush=True)
'''


def kernel(d: float, gate: int = 1) -> dict:
    """Submit the gate at one d (60 cells; gate 2: 80) as one Kaggle CPU
    kernel; write the returned cells to OUT/gate[2]_dXXX.json (failed cells
    kept with their error)."""
    from experiments.gpu import build_job, result_json, submit
    if d not in GATE_DS:
        raise ValueError(f"d {d!r} not in {GATE_DS}")
    body = (_JOB.replace("__CELLS__", repr(gate_cells(d, gate)))
            .replace("__PINS__", repr(KERNEL_PINS)))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=5.0,
                 timeout_s=REMOTE_TIMEOUT_S, fetch=["eye_gate.json"])
    if not sub.ok:
        raise RuntimeError(f"eye_forager gate d={d}: {sub.message}")
    r = result_json(sub, "eye_gate.json")
    OUT.mkdir(parents=True, exist_ok=True)
    name = {2: "gate2", 3: "gate3"}.get(gate, "gate")
    (OUT / f"{name}_d{int(round(100 * d)):03d}.json").write_text(json.dumps(r, indent=1) + "\n")
    return r


def score_gate(paths) -> dict:
    """The frozen bar at each d, from harvested gate files."""
    rows = {}
    for p in paths:
        for rec in json.loads(Path(p).read_text())["cells"].values():
            rows.setdefault((rec["row"], rec["d"]), []).extend(rec["lives"])
    out = {}
    for d in GATE_DS:
        b = rows.get(("b3", d), [])
        out[d] = {r: WC.score(rows.get((r, d), []), b, GATE_SEEDS) for r in ("f3", "g3")}
    return out


def score_gate2(paths) -> dict:
    """Gate 2's frozen bar at each d. h4 is scored by the same function but
    is REPORTED ONLY (no bar clause)."""
    rows = {}
    for p in paths:
        for rec in json.loads(Path(p).read_text())["cells"].values():
            rows.setdefault((rec["row"], rec["d"]), []).extend(rec["lives"])
    out = {}
    for d in GATE_DS:
        b = rows.get(("b4", d), [])
        out[d] = {r: WC.score(rows.get((r, d), []), b, GATE2_SEEDS)
                  for r in ("f4", "g4", "h4")}
        out[d]["h4"]["reported_only"] = True
    return out


def score_gate3(paths) -> dict:
    """Gate 3's frozen bar (d0 only)."""
    rows = {}
    for p in paths:
        for rec in json.loads(Path(p).read_text())["cells"].values():
            rows.setdefault((rec["row"], rec["d"]), []).extend(rec["lives"])
    b = rows.get(("b5", GATE3_D), [])
    return {GATE3_D: {r: WC.score(rows.get((r, GATE3_D), []), b, GATE3_SEEDS)
                      for r in ("f5", "g5")}}


def main(argv):
    if len(argv) >= 6 and argv[1] == "cell":       # cell <row> <seed> <d> <out>
        from experiments.sim_speed_kcpu import _compat_multiray
        _compat_multiray()                         # mujoco >= 3.3 on Kaggle
        res = run_seed(argv[2], int(argv[3]), float(argv[4]))
        with open(argv[5], "a") as fh:
            fh.write("\n".join(json.dumps(r) for r in res) + "\n")
        return 0
    if len(argv) >= 2 and argv[1] == "kernel3":    # kernel3 — gate 3 (f5), d0
        r = kernel(GATE3_D, gate=3)
        print(json.dumps({k: len(v["lives"]) for k, v in r["cells"].items()}))
        return 0
    if len(argv) >= 2 and argv[1] == "score3":     # score3 <gate3 files...>
        print(json.dumps(score_gate3(argv[2:]), indent=1))
        return 0
    if len(argv) >= 3 and argv[1] == "kernel2":    # kernel2 <d> — gate 2
        r = kernel(float(argv[2]), gate=2)
        print(json.dumps({k: len(v["lives"]) for k, v in r["cells"].items()}))
        return 0
    if len(argv) >= 2 and argv[1] == "smoke2":     # smoke2 [d] [cap] — 1 life, seed 153
        d = float(argv[2]) if len(argv) > 2 else 0.75
        cap = int(argv[3]) if len(argv) > 3 else WC.LIFE_CAP_DECISIONS
        for row in GATE2_ROWS:
            t = time.process_time()
            res = run_seed(row, SMOKE_SEED, d, lives=1, cap=cap)
            print(row, d, res, f"{time.process_time() - t:.0f} CPU-s", flush=True)
        return 0
    if len(argv) >= 2 and argv[1] == "score2":     # score2 <gate2 files...>
        print(json.dumps(score_gate2(argv[2:]), indent=1))
        return 0
    if len(argv) >= 3 and argv[1] == "kernel":     # kernel <d>
        r = kernel(float(argv[2]))
        print(json.dumps({k: len(v["lives"]) for k, v in r["cells"].items()}))
        return 0
    if len(argv) >= 2 and argv[1] == "smoke":      # smoke [d] — 1 life, seed 153
        d = float(argv[2]) if len(argv) > 2 else 0.75
        for row in GATE_ROWS:
            t = time.process_time()
            res = run_seed(row, SMOKE_SEED, d, lives=1)
            print(row, d, res, f"{time.process_time() - t:.0f} CPU-s", flush=True)
        return 0
    if len(argv) >= 2 and argv[1] == "score":      # score <gate files...>
        print(json.dumps(score_gate(argv[2:]), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
