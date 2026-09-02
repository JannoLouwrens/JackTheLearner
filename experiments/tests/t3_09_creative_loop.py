"""T3.09 — the creative loop earns its existence, or is deleted.

`AlphaGeometryLoop.py` (559 lines, "THIS IS WHERE AGI HAPPENS!") has sat in
the repo since the pre-ledger era. Its one nominal call path —
`TaskManager._try_creative_reasoning`, the declared "last resort when trained
skills AND LLM replanning both fail" — hands it `torch.randn(256)` for both
state and goal ("Would use actual state from last tick"), so nothing it has
ever computed has depended on the world. The registry's verdict options are
pre-registered in the spec: wire it or delete it.

THE WIRING, faithful to the module's own declaration. The one decision path in
this project that is certified and actually runs is XL.01's forager (W0,
privileged-target "ref" mode: PD waypoint driving, pool routing, and a
stuck→detour branch measured necessary — 2/8 ref lives wedge on world
geometry). The stuck branch IS TaskManager's declared role transplanted into a
real path: progress toward the target has stalled for STUCK_N decisions and
the shipped recovery is a random detour waypoint. This test consults an
advisor at exactly that point, for exactly the shipped detour duration; only
the SOURCE of the detour direction differs by arm:

  off   (null, the spec's verbatim null_baseline)  loop disabled: the shipped
        random detour. Nothing else in the path changes.
  twin  (reference + attribution control)  direction = unit(goal - xy). Three
        lines, no torch, no sympy — the exact information the loop is handed,
        minus its 559 lines. The SH.01/XL.01 reference lesson, pointed at a
        module: if the loop only matches this, its contribution IS the
        subtraction.
  loop  (claim)  `AlphaGeometryLoop.solve(state, goal)` under the module's own
        conventions, best shot honestly given:
          - state per its docstring ("joint angles, velocities"; predict reads
            [:3]=position, [3:6]=velocity): rover xyz + world-frame velocity,
            zero-padded to its state_dim 256.
          - goal[:3] = the current target food geom's live xyz.
          - config = the repo's sole declared wiring, TaskManager.py:194:
            LoopConfig(max_iterations=5, timeout_seconds=2.0).
          - the proposer is TRAINED first by the module's own
            `train_proposer` (its defaults: 100 episodes, T=1.0, REINFORCE),
            on (state, goal) pairs harvested at real stuck events in a pilot
            world (WORLD_TRAIN) disjoint from the eval world. Skipping its
            declared training step would invite "you tested an untrained
            net"; its acceptance_rate is reported.
          - a returned action's [:2] is a force direction (its convention);
            the detour waypoint is xy + unit(f)*DETOUR_DIST, clipped to the
            arena, held by the SAME explorer machinery for the SAME DETOUR_N.
          - `solve` returning None falls back to the shipped random detour —
            TaskManager's own Strategy-3 fallback semantics — and is counted.
  shuf  (registry control, must fail)  the identical trained loop handed
        goal[:2] = 2*xy - goal_xy: reflected through the rover, exactly as
        much information, exactly wrong. If wrong-goal advice buys the same
        improvement, the site rewards any perturbation and the test measures
        nothing.

PAIRING. Spawns are precomputed once per life and shared by every arm;
per-life RNG is seeded by (world, life) identically across arms and
torch RNG likewise, so lives are byte-identical across arms until the first
consult fires. Spawns are rejection-sampled (OBSTRUCT_TRIES cap) so the
straight segment spawn→nearest-food crosses a non-pool feature box — the
module's own declared operating regime ("robot sees stairs, gets stuck"),
symmetric across arms, and the reason MIN_AFFECTED is reachable at N_LIVES.
"Affected" lives are those whose OFF arm fired the stuck branch at least once
— a property of the shared prefix, never of a compared outcome.

PRE-REGISTERED GATES, from world arithmetic, before any recording run (the
LG.02 discipline; smoke is crash-freedom + consult-aliveness counts ONLY, no
performance metric previewed):

  One stuck cycle costs (STUCK_N + DETOUR_N) = 110 decisions = 22.0 s of life.
  An advisor that genuinely helps must save at least half a cycle on the
  lives where it was consulted: MARGIN_AFF = 11.0 s, on mean time-to-feed
  over AFFECTED lives, timeouts counted at the full cap (T2.20's rule).

  creative_contribution = min(off_aff - loop_aff, twin_aff - loop_aff):
  seconds saved by the wired loop beyond BOTH the loop-less path and the
  three-line twin. The min is the point — beating `off` alone is information
  (the goal direction), not machinery, and would be a rigged pass.

  PASS  iff creative_contribution >= MARGIN_AFF and the shuf control stays
        under the same margin (off_aff - shuf_aff < MARGIN_AFF).
  FAIL  iff creative_contribution < MARGIN_AFF with the rig alive — including
        the case where every consult returns None and the arms are
        byte-identical: "no measurable difference" is the spec's own
        pre-declared falsification branch, measured rather than asserted.
  VOID  (run did not test the claim), any of:
        - PS.01 calibration unborrowable (the drive layer is uncertified);
        - off arm fed_frac < OFF_MIN_FED (cap saturation: the base path
          cannot feed on obstructed approaches, denominators are fixture);
        - n_affected < MIN_AFFECTED or loop consults < MIN_AFFECTED (the
          call site never got its chance);
        - zero training pairs harvested (the module's declared recipe could
          not execute);
        - shuf consults < MIN_AFFECTED (its non-improvement would be vacuous);
        - a PASS whose shuf control ALSO cleared the margin (attribution
          impossible; law 2).

WHAT A FAIL EXECUTES. The spec's kills clause: delete `AlphaGeometryLoop.py`
from the root. `archive/AlphaGeometryLoop.py` is byte-identical (verified
2026-09-02) and stays; TaskManager/UnifiedBrain import it inside try/except
and self-disable. DIRECTION_AUDIT (line 1126) predicted delete; the ledger,
not the audit, decides.

Honesty notes, pre-registered:
  - Whether `solve` answers from its direct branch or its creative branch is
    the module's own behavior under real state (its height/force/velocity
    safety constants are untouched); both branch counts are reported, neither
    is tuned.
  - The twin doubles as the site-leverage diagnostic: twin_gain (off - twin)
    is REPORTED either way, so a FAIL also says whether ANY directional
    advice could have helped at this site. No threshold attaches to it.
  - WORLD_EVAL = 7: XL.01 designed on worlds 0-2, recorded on 3-5; 6 is this
    test's training world. Nothing here certifies on draws that shaped it.
  - Advised waypoints may land on the pool footprint; `_route` treats them
    exactly as it treats random ones (declared apparatus, identical across
    arms); the fraction is reported.
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..w0 import W0, SIM_S_PER_DECISION, uniform_legal_spawn
# The certified decision path and its fixtures — this claim rides on them.
from . import xl_01_death_does_not_erase as xl
# xl's `..w0` import puts the repo root on sys.path.
import torch  # noqa: E402
from AlphaGeometryLoop import AlphaGeometryLoop, LoopConfig  # noqa: E402

IMPL_DEPS = ["AlphaGeometryLoop.py", "SymbolicCalculator.py",
             "experiments/w0.py", "experiments/drives.py", "playground.py",
             "experiments/tests/xl_01_death_does_not_erase.py"]

# ── pre-registered constants ───────────────────────────────────────────────
WORLD_EVAL = 7            # fresh world: XL.01 designed on 0-2, recorded 3-5
WORLD_TRAIN = 6           # proposer training + harvest world, never evaluated
N_LIVES = 16              # per arm; every arm sees the identical spawn list
N_PILOT = 8               # harvest lives on WORLD_TRAIN (off-mode)
CAP_LIFE = xl.CAP_LIFE    # 1500 decisions = 300 s measurement window
DETOUR_DIST = 1.5         # m; advised waypoint distance at a consult
MARGIN_AFF = 11.0         # s; half a stuck cycle — see docstring arithmetic
MIN_AFFECTED = 8          # of N_LIVES; below this the site was under-exercised
OFF_MIN_FED = 0.5         # off-arm fed fraction floor, else cap saturation
OBSTRUCT_TRIES = 200      # spawn rejection-sampling cap per life
CORE_SHRINK = 0.5         # obstruction predicate uses the feature CORE (the
                          # keep-clear boxes at half size): a segment through
                          # the outer margin usually clips past. Frozen from
                          # the TRAIN-world aliveness pilot (2026-09-02,
                          # counts only, no eval metric): full boxes -> 3/8
                          # lives fired the stuck branch; core boxes -> 10/16
                          # affected, 10/16 fed. The eval world was untouched.
STATE_DIM = 256           # the module's own default state_dim
TORCH_INIT_SEED = 90      # proposer weight init (house pilot seed)
TRAIN_LR = 1e-3           # Adam, handed to the module's own train_proposer
ARENA_CLIP = 4.5          # the explorer's own waypoint bounds

_MEMO: dict = {}          # seed -> experiment side, reused by _control


def _crosses_box(a: np.ndarray, b: np.ndarray, box: tuple) -> bool:
    cx, cy, hx, hy = box
    for s in np.linspace(0.0, 1.0, 60):
        p = a + s * (b - a)
        if abs(float(p[0]) - cx) < hx and abs(float(p[1]) - cy) < hy:
            return True
    return False


def _obstructed(spawn: np.ndarray, food_xys: list) -> bool:
    """Straight approach to the nearest food crosses a non-pool feature CORE.

    The pool box (index 0) is excluded: `_route` detours it by design, so a
    pool-crossing approach exercises the router, not the stuck branch. The
    remaining boxes are shrunk by CORE_SHRINK — see that constant's pilot."""
    near = min(food_xys, key=lambda f: float(np.linalg.norm(f - spawn)))
    return any(_crosses_box(spawn, near, (cx, cy, hx * CORE_SHRINK,
                                          hy * CORE_SHRINK))
               for cx, cy, hx, hy in xl.FEATURE_CLEAR[1:])


def _plan_spawns(w: W0, wseed: int, n: int) -> tuple:
    """n paired spawns, rejection-sampled toward obstructed approaches."""
    legal = w.legal_spawns()
    food_xys = [np.array(p, dtype=float) for p in xl._wide_homes(wseed)]
    rng = np.random.RandomState(4177 + wseed)
    spawns, hits = [], 0
    for _ in range(n):
        pick, ok = None, False
        for _ in range(OBSTRUCT_TRIES):
            pick = np.array(uniform_legal_spawn(legal, rng, (0.0, 0.0)))
            if _obstructed(pick, food_xys):
                ok = True
                break
        spawns.append(pick)
        hits += int(ok)
    return spawns, hits / max(n, 1)


def _target_xyz(w: W0, target_xy: np.ndarray) -> np.ndarray:
    """The live 3-D position of the food geom nearest the current target."""
    best, best_d = None, float("inf")
    for _, (gid, _) in sorted(xl._food_positions(w).items()):
        p = np.array(w.data.geom_xpos[gid], dtype=float)
        d = float(np.linalg.norm(p[:2] - target_xy))
        if d < best_d:
            best, best_d = p, d
    return best


def _advise(arm: str, loop_obj, w: W0, xy: np.ndarray,
            target_xy: np.ndarray, stats: dict):
    """The detour waypoint for this consult, or None -> shipped random."""
    if arm == "off":
        return None
    if arm == "twin":
        d = target_xy - xy
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None
        return np.clip(xy + d / n * DETOUR_DIST, -ARENA_CLIP, ARENA_CLIP)
    # loop / shuf — the module's own I/O conventions, see docstring
    root = w.ix["root_dofadr"]
    pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
    vel = np.array([w.data.qvel[root], w.data.qvel[root + 1],
                    w.data.qvel[root + 2]], dtype=float)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:3], state[3:6] = pos, vel
    gxyz = _target_xyz(w, target_xy).copy()
    if arm == "shuf":
        gxyz[:2] = 2.0 * xy - gxyz[:2]      # reflected through the rover
    goal = np.zeros(STATE_DIM, dtype=np.float32)
    goal[:3] = gxyz
    action, meta = loop_obj.solve(state, goal)
    if action is None:
        stats["none"] += 1
        return None
    stats[meta.get("mode", "other")] = stats.get(meta.get("mode", "other"),
                                                 0) + 1
    f = np.asarray(action[:2], dtype=float)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        stats["zero"] += 1
        return None
    wp = np.clip(xy + f / n * DETOUR_DIST, -ARENA_CLIP, ARENA_CLIP)
    if xl._in_pool(wp, stats["_half"]):
        stats["in_pool"] += 1
    return wp


def _live(w: W0, targets: list, rng, arm: str, loop_obj, half: float,
          stats: dict) -> dict:
    """One life on XL.01's ref-mode path; only the stuck branch's detour
    direction differs by arm. Mirrors `xl._live_one_life` with ONE declared
    omission: the sighting block, which in ref mode (both true positions
    pre-seeded as candidates) only re-appends a food position after the body
    bulldozes the item > CAND_DEDUPE_R from home. Candidates here stay at
    the widened homes for the whole life — identical across every arm, so it
    cannot touch the arm contrast."""
    explorer = xl._Explorer(rng, half)
    candidates = [np.array(p, dtype=float) for p in targets]
    blacklist: list = []
    parked_since = None
    best_dist, best_at = float("inf"), 0
    detour_until = -1
    n_stuck = 0
    ate0 = sum(w.drives.ate_total.values())

    for d in range(CAP_LIFE):
        xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        if sum(w.drives.ate_total.values()) > ate0:
            return {"ttf_s": d * SIM_S_PER_DECISION, "fed": True,
                    "died": False, "stuck": n_stuck}
        live = [c for c in candidates if not any(c is b for b in blacklist)]
        if live and d >= detour_until:
            target = min(live, key=lambda c: float(np.linalg.norm(c - xy)))
            dist = float(np.linalg.norm(target - xy))
            if dist < xl.WAYPOINT_DONE:
                parked_since = d if parked_since is None else parked_since
                if d - parked_since >= xl.PATIENCE:
                    blacklist.append(target)
                    parked_since = None
            else:
                parked_since = None
                if dist < best_dist - 0.1:
                    best_dist, best_at = dist, d
                elif d - best_at >= xl.STUCK_N:
                    # THE CALL SITE — TaskManager's declared role, real path
                    n_stuck += 1
                    stats["consults"] += 1
                    adv = _advise(arm, loop_obj, w, xy, target, stats)
                    detour_until = d + xl.DETOUR_N
                    if adv is not None:
                        explorer.wp, explorer.age = adv, 0
                    else:
                        explorer.wp = None      # shipped: fresh random draw
                    best_dist, best_at = float("inf"), d
        else:
            target = explorer.target(xy)
        wp = xl._route(xy, target, half)
        root = w.ix["root_dofadr"]
        v = np.array([w.data.qvel[root], w.data.qvel[root + 1]], dtype=float)
        a = np.zeros(w.action_dim)
        a[4:6] = -1.0                           # adhesion OFF (XL.01's note)
        a[6:8] = np.clip(xl.KP * (wp - xy) - xl.KD * v, -1.0, 1.0)
        w.decide(a)
        if w.died_this_decision:
            return {"ttf_s": CAP_LIFE * SIM_S_PER_DECISION, "fed": False,
                    "died": True, "stuck": n_stuck}
    return {"ttf_s": CAP_LIFE * SIM_S_PER_DECISION, "fed": False,
            "died": False, "stuck": n_stuck}


def _run_arm(arm: str, wseed: int, spawns: list, j0: float, alpha: float,
             loop_obj=None, harvest: list = None) -> tuple:
    """N lives under one arm. Paired RNG: seeded by (world, life) only."""
    from EpisodicMemory import EpisodicMemory
    w = W0(seed=wseed, j0=j0, alpha=alpha, lethal=True,
           diary=EpisodicMemory())
    home = xl._widen(xl._food_home(w), wseed)
    half = xl._pool_half(w)
    stats = {"consults": 0, "none": 0, "zero": 0, "in_pool": 0,
             "_half": half}
    lives = []
    for life, spawn in enumerate(spawns):
        w.respawn(at=(float(spawn[0]), float(spawn[1])))
        w.drives.state = drives.DriveState(e=xl.LIFE_E0)
        w.drives._respawn_at = {n: 0.0 for n in w.drives._respawn_at}
        xl._reset_food(w, home)
        rng = np.random.RandomState(900_001 * (life + 1) + wseed)
        torch.manual_seed(770_003 * (life + 1) + wseed)
        targets = [p for _, (_, p) in sorted(xl._food_positions(w).items())]
        if harvest is not None:
            # off-mode pilot that records what an advisor WOULD have seen
            r = _live_harvest(w, targets, rng, half, harvest)
        else:
            r = _live(w, targets, rng, arm, loop_obj, half, stats)
        lives.append(r)
        if not r["died"]:
            xl._drain(w)
    return lives, stats


def _live_harvest(w, targets, rng, half, harvest: list) -> dict:
    """Off-arm life that appends (state, goal) at each stuck event."""
    class _Tap:
        def solve(self, state, goal):
            harvest.append((np.array(state, dtype=np.float32),
                            np.array(goal, dtype=np.float32)))
            return None, {}
    stats = {"consults": 0, "none": 0, "zero": 0, "in_pool": 0, "_half": half}
    return _live(w, targets, rng, "harvest_tap", _Tap(), half, stats)


def _train_loop(j0: float, alpha: float) -> tuple:
    """The module's own recipe: harvest real stuck pairs, REINFORCE 100 eps."""
    torch.manual_seed(TORCH_INIT_SEED)
    loop = AlphaGeometryLoop(LoopConfig(max_iterations=5, timeout_seconds=2.0))
    from EpisodicMemory import EpisodicMemory
    w = W0(seed=WORLD_TRAIN, j0=j0, alpha=alpha, lethal=True,
           diary=EpisodicMemory())
    spawns, _ = _plan_spawns(w, WORLD_TRAIN, N_PILOT)
    del w
    harvest: list = []
    _run_arm("harvest", WORLD_TRAIN, spawns, j0, alpha, harvest=harvest)
    if not harvest:
        return loop, {"acceptance_rate": -1.0}, 0
    states = torch.stack([torch.from_numpy(s) for s, _ in harvest])
    goals = torch.stack([torch.from_numpy(g) for _, g in harvest])
    opt = torch.optim.Adam(loop.proposer.parameters(), lr=TRAIN_LR)
    rep = loop.train_proposer(states, goals, opt)   # its own defaults
    return loop, rep, len(harvest)


def _mean_ttf(lives: list, idx: list) -> float:
    return float(np.mean([lives[i]["ttf_s"] for i in idx])) if idx else -1.0


def _experiment(seed: int) -> dict:
    j0, alpha, prov = xl._calibration()
    if j0 is None:
        return {"calibrated": 0.0, **prov}
    from EpisodicMemory import EpisodicMemory
    w = W0(seed=WORLD_EVAL, j0=j0, alpha=alpha, lethal=True,
           diary=EpisodicMemory())
    spawns, obstructed_frac = _plan_spawns(w, WORLD_EVAL, N_LIVES)
    del w

    loop, train_rep, n_pairs = _train_loop(j0, alpha)

    off, _ = _run_arm("off", WORLD_EVAL, spawns, j0, alpha)
    twin, _ = _run_arm("twin", WORLD_EVAL, spawns, j0, alpha)
    lp, lp_stats = _run_arm("loop", WORLD_EVAL, spawns, j0, alpha,
                            loop_obj=copy.deepcopy(loop))

    aff = [i for i, r in enumerate(off) if r["stuck"] > 0]
    all_i = list(range(N_LIVES))
    off_aff, twin_aff, loop_aff = (_mean_ttf(off, aff), _mean_ttf(twin, aff),
                                   _mean_ttf(lp, aff))
    cc = min(off_aff - loop_aff, twin_aff - loop_aff) if aff else 0.0

    m = {
        "calibrated": 1.0, **prov,
        "n_affected": float(len(aff)),
        "obstructed_frac": float(obstructed_frac),
        "off_fed_frac": float(np.mean([r["fed"] for r in off])),
        "twin_fed_frac": float(np.mean([r["fed"] for r in twin])),
        "loop_fed_frac": float(np.mean([r["fed"] for r in lp])),
        "off_ttf_all": _mean_ttf(off, all_i),
        "twin_ttf_all": _mean_ttf(twin, all_i),
        "loop_ttf_all": _mean_ttf(lp, all_i),
        "off_ttf_aff": off_aff, "twin_ttf_aff": twin_aff,
        "loop_ttf_aff": loop_aff,
        "twin_gain": (off_aff - twin_aff) if aff else 0.0,
        "creative_contribution": float(cc),
        "loop_consults": float(lp_stats["consults"]),
        "loop_none": float(lp_stats["none"]),
        "loop_direct": float(lp_stats.get("direct", 0)),
        "loop_creative": float(lp_stats.get("creative", 0)),
        "loop_wp_in_pool": float(lp_stats["in_pool"]),
        "train_pairs": float(n_pairs),
        "train_accept_rate": float(train_rep.get("acceptance_rate", -1.0)),
    }
    _MEMO[seed] = {"spawns": spawns, "loop": loop, "off_aff": off_aff,
                   "aff": aff, "j0": j0, "alpha": alpha}
    return m


def _control(seed: int) -> dict:
    side = _MEMO.get(seed)
    if side is None:
        return {"shuf_ran": 0.0}
    sh, sh_stats = _run_arm("shuf", WORLD_EVAL, side["spawns"], side["j0"],
                            side["alpha"],
                            loop_obj=copy.deepcopy(side["loop"]))
    aff = side["aff"]
    sh_aff = _mean_ttf(sh, aff)
    return {
        "shuf_ran": 1.0,
        "shuf_ttf_aff": sh_aff,
        "shuf_fed_frac": float(np.mean([r["fed"] for r in sh])),
        "shuf_gain": (side["off_aff"] - sh_aff) if aff else 0.0,
        "shuf_consults": float(sh_stats["consults"]),
        "shuf_none": float(sh_stats["none"]),
        "shuf_direct": float(sh_stats.get("direct", 0)),
        "shuf_creative": float(sh_stats.get("creative", 0)),
    }


def _check(m: dict, c: dict):
    # ── rig gates: VOID, not FAIL — a run that could not ask the question ──
    if m.get("calibrated", 0.0) != 1.0:
        return Status.VOID          # drive layer uncertified
    if m["off_fed_frac"] < OFF_MIN_FED:
        return Status.VOID          # cap saturation: base path cannot feed
    if m["n_affected"] < MIN_AFFECTED:
        return Status.VOID          # site under-exercised
    if m["loop_consults"] < MIN_AFFECTED:
        return Status.VOID          # the wiring never got its chance
    if m["train_pairs"] < 1:
        return Status.VOID          # the module's own recipe could not run
    if c.get("shuf_ran", 0.0) != 1.0 or c["shuf_consults"] < MIN_AFFECTED:
        return Status.VOID          # control vacuous
    # ── the claim ──
    if m["creative_contribution"] < MARGIN_AFF:
        return False
    # ── a PASS must survive its control (law 2) ──
    if c["shuf_gain"] >= MARGIN_AFF:
        return Status.VOID          # site rewards any perturbation
    return True


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T3.09"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        # Crash-freedom + consult-aliveness ONLY. No performance metric is
        # printed — the gates above were registered from arithmetic.
        globals().update(N_LIVES=2, N_PILOT=2, CAP_LIFE=300,
                         OBSTRUCT_TRIES=40)
        xl.CAP_LIFE = 300
        j0, alpha, _ = xl._calibration()
        if j0 is None:
            raise SystemExit("smoke: PS.01 calibration unborrowable")
        loop, rep, n_pairs = _train_loop(j0, alpha)
        from EpisodicMemory import EpisodicMemory
        w = W0(seed=WORLD_EVAL, j0=j0, alpha=alpha, lethal=True,
               diary=EpisodicMemory())
        spawns, ofrac = _plan_spawns(w, WORLD_EVAL, N_LIVES)
        del w
        counts = {}
        for arm, lo in (("off", None), ("twin", None),
                        ("loop", copy.deepcopy(loop)),
                        ("shuf", copy.deepcopy(loop))):
            lives, st = _run_arm(arm, WORLD_EVAL, spawns, j0, alpha,
                                 loop_obj=lo)
            counts[arm] = {"consults": st["consults"], "none": st["none"],
                           "direct": st.get("direct", 0),
                           "creative": st.get("creative", 0)}
        print("smoke: 4 arms x 2 lives + train(%d pairs) completed "
              "without error" % n_pairs)
        print("smoke consult-aliveness:", json.dumps(counts))
    else:
        m = _experiment(0)
        c = _control(0)
        print(json.dumps({"experiment": m, "control": c}, indent=2))
