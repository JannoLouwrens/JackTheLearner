"""LG.03 — the command cells are language-necessary, certified before any arm.

`LG.04`/`LG.05`/`LG.06` all `depends_on` this spec, so `protocol.blocked_by()`
structurally prevents a grounding arm from ever being scored on an uncertified
cell set. That ordering is the whole design, and it is ME.11.0's, transported
from Q&A to BEHAVIOUR.

WHY IT EXISTS, in one citation. CAST (2508.13446, Glossop et al. 2025) names
the mechanism that kills language-conditioned policies: *"the future action
distribution typically collapses given any single observation (e.g. given an
observation of a chest of drawers, the only probable task for a robot is 'open
the drawer'). Thus, even powerful models have little incentive to pay attention
to the language command."* That is a property of the DATA, not of the model —
no architecture and no scale repairs it — so it is checkable before an arm is
trained, in minutes, and it is checked here.

THE TWO LEGS, both PER CELL and never on average (LANGUAGE_GROUNDING.md §2.3):

  1. NECESSITY.  A language-blind twin, trained on the identical observation
     stream with no instruction channel, must sit inside the pre-registered
     chance band on that cell — and the cell's target must not be FREE either
     (a random body must not achieve it). One clause says the act cannot be
     read off the observation; the other says it is an act at all.
  2. PLURALITY.  From the cell's own start states a privileged planner must
     reach at least two DISTINCT cell targets — the cell's own, and one other
     verb on the same object. More than one thing is achievable from what he
     can see, so the words are what choose between them.

A cell that fails either leg is EXCLUDED and the exclusion is logged with the
reading that excluded it. The author does not get to say which commands are
language-necessary; the blind twin and the random body do.

THE VENUE IS W0, AND THE INSTRUCTION CHANNEL IS ALREADY ABSENT. `w0.W0.DROPPED`
is `("language",)`: W0 has no talker in it, the key is present-and-dropped
rather than zero-filled, and every policy in this file sees exactly the same 80
numbers. So "instruction channel zeroed" is not something this file arranges —
it is the world's own condition, and the blind twin is simply a policy trained
on W0's observation.

THE CELLS. Four verbs x five objects, the objects read LIVE off the model
(`_OBJECTS`) so a mutated world cannot silently rename one:

    approach X   he comes within NEAR_M of it
    touch X      a rover geom contacts one of X's geoms
    avoid X      he never gets meaningfully closer than he started AND ends
                 RETREAT_M further out — a start-relative predicate, because an
                 absolute one is satisfied by standing still far away
    round X      he gets inside ROUND_M and swings ROUND_DEG of bearing around
                 it — being NEAR a thing and going AROUND it are different acts

    ladder · ramp · stairs · block · pool

STARTS ARE DRAWN PER CELL, from `legal_spawns()` filtered to START_R_LO..
START_R_HI metres from that cell's object. This is venue definition, not
curation: "avoid the ladder" from the far corner is a vacuous command, and a
cell whose situations do not contain its object is not a situation about the
object. What the starts may never do is decide the outcome — the necessity leg
does that, and it is run on exactly these starts.

WHY THE BLIND TWIN IS EVALUATED IN-DISTRIBUTION, deliberately. It is trained on
the planner's demonstrations from these very starts and rolled out from them.
That is maximally GENEROUS to the null, which is the strict direction here: a
null given every advantage and still unable to perform the act is the only null
whose failure certifies anything. A held-out blind twin would fail more often
and retain more cells, which is the weak direction.

THE CALIBRATION, AND IT IS THE REASON THIS SPEC CANNOT BE PASSED BY A DEAD
LEARNER. The necessity leg is permissive in exactly one direction: a blind twin
that can do nothing excludes nothing and every cell is retained. A broken
feature vector, a mis-shaped action, a k-NN with no neighbours — each looks
IDENTICAL to "these commands are beautifully language-necessary". This is
LG.01's letter-readout scar and the 24th audit's B3 rule ("an at-chance control
must carry proof its instrument was alive"), and the reading the author wants
here is the null's at-chance reading. So the same learner is re-fit on ONE
cell's demonstrations alone and must reproduce that cell from its own starts at
>= CALIB_MIN, or the run is VOID rather than PASS. The calibration is
in-distribution on purpose: it is a LIVENESS proof, not a generalisation claim.

AND THE PLANNER CARRIES THE SAME BURDEN, PER VERB. A verb its privileged
planner cannot perform is an INSTRUMENT fault, not a venue verdict — its cells
would fail achievability, the verb spread would collapse and the file would
report "this world admits no language-necessary commands" when what happened is
that a servo could not do the thing. `verb_alive_min` is the minimum over verbs
of the best per-object planner reach rate; below VERB_ALIVE the run is VOID and
names the verb.

CONTROL (declared in the registry; must fail): THE PLANNER, STRIPPED. The
plurality leg claims that from a start the planner reaches target A *and*
target B. The stripped planner runs the identical machinery with the target
identity WITHHELD — it is handed some other cell's target, drawn uniformly —
and must not satisfy BOTH of the demanded pair. If it does, "achievable" was
being read out of the world state rather than chosen, and the plurality
certificate is void. This control can genuinely fail: in a world where any
motion satisfies every predicate it reads 1.0, and it is measured on every
candidate cell, not only on the survivors.

VERDICTS.
  PASS  — every seed independently retains >= MIN_CELLS cells spanning
          >= MIN_VERBS verbs and >= MIN_OBJECTS objects with every verb and
          every object represented >= 2 times, the cross-seed intersection
          meets the same bars, and the stripped control fails.
  FAIL  — fewer than that survive. This world does not admit language-necessary
          commands at this horizon. Per the registry, the LG bakeoff is then
          VENUE-BLOCKED rather than model-blocked and the reading is routed to
          `w0-too-shallow` as an instrument — which is exactly why this is a
          registered fixture spec and not a pilot with its verdict in a
          docstring (LANGUAGE_GROUNDING.md §11(b): the one venue certification
          written as a spec, ME.11.0, is quotable by every downstream arm
          because it is on the scoreboard; the five written as pilots are not).
  VOID  — the blind twin is not demonstrably alive, a verb's planner cannot
          perform it, or an observation is not finite. None of those is a
          refutation of the claim.

ONE PLACE THIS IMPLEMENTATION IS STRICTER THAN THE REGISTRY TEXT, declared
because strengthening silently is indistinguishable from drifting. The
hypothesis says plurality needs "at least two DISTINCT cell targets" reachable
and does not say one of them must be the cell's own. Here the cell's own target
must be reachable too (ACH_FRAC of its starts) — a cell nobody can perform is
not a task, and LG.04 would train an arm on it. That adds a requirement; it
removes none.

WHERE THE CROSS-SEED INTERSECTION IS COMPUTED, and why it is in the control.
`run_spec` runs every experiment seed before it runs the control, and
`_aggregate` cannot see across seeds. `_control` is therefore the only hook
from which the intersection of the per-seed retained sets is knowable, so it
carries `common_n` / `common_ok` / `common_cells` alongside the stripped-planner
reading. The keys are named so no auditor mistakes them for control readings.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Tuple

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..w0 import W0, SIM_S_PER_DECISION, random_action

# ── the venue ───────────────────────────────────────────────────────────────
J0, ALPHA = 0.5, 0.02          # W0's drive parameters, LC.02's and LF.01's
N_STARTS = 4                   # start states per cell
T_DECISIONS = 24               # 4.8 simulated seconds per rollout
START_R_LO, START_R_HI = 1.5, 4.0     # m from the cell's object

#: object -> (anchor geom for the xy, prefixes of every geom that IS the object)
_OBJECTS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "ladder": ("rung0", ("ladder_rail", "rung", "platform")),
    "ramp":   ("ramp", ("ramp",)),
    "stairs": ("stair0", ("stair",)),
    "block":  ("welded_block", ("welded_block",)),
    "pool":   ("pool_water", ("pool_", "poolwall")),
}
VERBS = ("approach", "touch", "avoid", "round")

# ── the predicates, pre-registered ──────────────────────────────────────────
NEAR_M = 0.9                   # approach
AVOID_TOL = 0.25               # avoid: never closer than d0 - this
RETREAT_M = 1.0                # avoid: ends this much further out than d0
ROUND_M = 2.5                  # round: must be inside this while swinging
ROUND_DEG = 90.0               # round: bearing swing about the object

# ── the bars, pre-registered ────────────────────────────────────────────────
# CHANCE_HI is 1 / len(VERBS). The blind twin is CREDITED WITH THE OBJECT —
# the object is genuinely in the observation, its rays reach it — so the only
# thing the instruction supplies is the verb, and a twin that cannot hear the
# instruction can do no better than a uniform guess among the four acts on it.
# That is the lenient of the two available chance definitions (the strict one
# is 1 / n_cells = 0.05) and it is chosen because it is the one that is TRUE
# of the observation, not because it is the one that retains more cells.
CHANCE_HI = 1.0 / len(VERBS)
ACH_FRAC = 0.75                # planner reaches the cell's own target
PLUR_FRAC = 0.75               # >= 2 distinct targets reachable from the start
CTRL_MAX = 0.10                # stripped planner satisfying BOTH demanded
CALIB_MIN = 0.75               # blind learner liveness (in-distribution)
VERB_ALIVE = 0.5               # a verb whose planner cannot do it -> VOID
MIN_CELLS, MIN_VERBS, MIN_OBJECTS = 12, 4, 4
MIN_PER_VERB, MIN_PER_OBJECT = 2, 2

# ── the blind twin ──────────────────────────────────────────────────────────
KNN_K = 5                      # neighbours averaged; deterministic, no fitting
KP, KD = 2.0, 0.6              # the servo gains, LF.01's forager's

#: per-seed retained cell sets, filled by `_experiment`, read by `_control`
#: (see the docstring's last paragraph — this is the only cross-seed channel
#: `run_spec` leaves open, and it is read only after every seed has run).
_RETAINED: Dict[int, List[str]] = {}


def _cell(verb: str, obj: str) -> str:
    return f"{verb}@{obj}"


def _alt_verb(verb: str) -> str:
    """The plurality alternate: the NEXT verb on the SAME object.

    Fixed rather than drawn, so `_control` reproduces the demanded pair
    without replaying `_experiment`'s RNG — a control that must re-derive its
    subject from another function's random stream is a control one refactor
    away from measuring a different question.
    """
    return VERBS[(VERBS.index(verb) + 1) % len(VERBS)]


# ── geometry helpers ────────────────────────────────────────────────────────
def _live_objects(w: W0) -> Dict[str, Tuple[np.ndarray, set]]:
    """Anchor xy and geom-id set per object, read off the LIVE model.

    A mutated world (W0 mutates `PlaygroundParams` for every seed but 0) may
    move or drop scenery. Reading it live means the cells are about what is
    actually there; an object whose anchor geom is absent is dropped and the
    drop shows up in `n_objects`.
    """
    names = [w.model.geom(i).name for i in range(w.model.ngeom)]
    out: Dict[str, Tuple[np.ndarray, set]] = {}
    for obj, (anchor, prefixes) in _OBJECTS.items():
        if anchor not in names:
            continue
        gid = int(w.model.geom(anchor).id)
        xy = np.array(w.data.geom_xpos[gid][:2], dtype=float)
        gids = {i for i, n in enumerate(names)
                if any(n.startswith(p) for p in prefixes)}
        out[obj] = (xy, gids)
    return out


def _xy(w: W0) -> np.ndarray:
    return np.array(w.data.xpos[w.rover_bid][:2], dtype=float)


def _obs_vec(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """The 80 numbers a policy in W0 may see. `language` is DROPPED, not zeroed
    — including it would hand every policy 32 constant zeros and quietly change
    what 'language-blind' means."""
    keys = [k for k in obs if k not in W0.DROPPED]
    return np.concatenate([np.asarray(obs[k], dtype=np.float64)
                           for k in sorted(keys)])


def _contacts(w: W0, gids: set) -> bool:
    """Is any rover geom touching one of this object's geoms right now?"""
    for i in range(int(w.data.ncon)):
        c = w.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in w.body_gids and g2 in gids) or (g2 in w.body_gids and g1 in gids):
            return True
    return False


# ── one rollout ─────────────────────────────────────────────────────────────
def _rollout(w: W0, start: np.ndarray, policy, obj_xy: np.ndarray,
             obj_gids: set, T: int = T_DECISIONS) -> dict:
    """Place the body at `start`, run `policy` for T decisions, return the
    trace every predicate is evaluated from. One rollout, one trace, all four
    verbs read off it — so no verb costs its own simulation."""
    w.respawn(at=(float(start[0]), float(start[1])))
    prev = _xy(w)
    d0 = float(np.linalg.norm(prev - obj_xy))
    b0 = math.atan2(prev[1] - obj_xy[1], prev[0] - obj_xy[0])
    dmin, dend, touched, swing_in = d0, d0, False, 0.0
    finite = True
    for t in range(T):
        obs = w.observe()
        v = _obs_vec(obs)
        if not np.all(np.isfinite(v)):
            finite = False
        xy = _xy(w)
        vel = (xy - prev) / SIM_S_PER_DECISION
        prev = xy
        w.decide(policy(v, xy, vel, t))
        xy = _xy(w)
        d = float(np.linalg.norm(xy - obj_xy))
        dmin, dend = min(dmin, d), d
        touched = touched or _contacts(w, obj_gids)
        if d <= ROUND_M:
            b = math.atan2(xy[1] - obj_xy[1], xy[0] - obj_xy[0])
            swing_in = max(swing_in, abs(math.degrees(
                math.atan2(math.sin(b - b0), math.cos(b - b0)))))
    return {"d0": d0, "dmin": dmin, "dend": dend, "touched": touched,
            "swing_in": swing_in, "finite": finite}


def _satisfies(verb: str, tr: dict) -> bool:
    if verb == "approach":
        return tr["dmin"] <= NEAR_M
    if verb == "touch":
        return bool(tr["touched"])
    if verb == "avoid":
        return (tr["dmin"] >= tr["d0"] - AVOID_TOL
                and tr["dend"] >= tr["d0"] + RETREAT_M)
    if verb == "round":
        return tr["swing_in"] >= ROUND_DEG
    raise KeyError(verb)


# ── the privileged planner ──────────────────────────────────────────────────
def _planner(verb: str, obj_xy: np.ndarray, start: np.ndarray):
    """A scripted servo that is TOLD the target. Privileged by construction:
    it reads the object's world coordinates, which no policy in this file's
    necessity leg may see. Each verb gets its own waypoint plan, so `approach`
    and `touch` are not the same trajectory wearing two names."""
    b0 = math.atan2(start[1] - obj_xy[1], start[0] - obj_xy[0])

    def wp(bearing_deg: float, r: float) -> np.ndarray:
        a = b0 + math.radians(bearing_deg)
        return obj_xy + r * np.array([math.cos(a), math.sin(a)])

    if verb == "approach":
        plan = [wp(0.0, NEAR_M * 0.7)]
    elif verb == "touch":
        plan = [obj_xy.copy()]
    elif verb == "avoid":
        plan = [wp(0.0, float(np.linalg.norm(start - obj_xy)) + RETREAT_M + 1.0)]
    elif verb == "round":
        r = ROUND_M * 0.65
        plan = [wp(60.0, r), wp(120.0, r), wp(180.0, r), wp(240.0, r)]
    else:
        raise KeyError(verb)

    state = {"i": 0}

    def policy(v, xy, vel, t):
        i = state["i"]
        tgt = plan[i]
        if i + 1 < len(plan) and float(np.linalg.norm(tgt - xy)) < 0.6:
            state["i"] = i + 1
            tgt = plan[state["i"]]
        a = np.zeros(8)
        a[4:6] = -1.0                       # adhesion off, as LF.01's forager
        a[6:8] = np.clip(KP * (tgt - xy) - KD * vel, -1.0, 1.0)
        return a

    return policy


def _random_policy(rng: np.random.RandomState):
    def policy(v, xy, vel, t):
        return random_action(rng)
    return policy


# ── the language-blind twin ─────────────────────────────────────────────────
class _Blind:
    """k-nearest-neighbour behaviour cloning over W0's observation.

    Chosen over a linear fit deliberately: a stronger null excludes MORE cells
    and retains fewer, so it moves this spec's own bar in the harder direction
    (LG.01's v2 repair, same reasoning). It is deterministic and needs no
    optimiser, so nothing about the verdict depends on a training schedule.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        self.Xn = (X - self.mu) / self.sd
        self.Y = Y
        self.k = min(KNN_K, len(Y))

    def policy(self):
        def policy(v, xy, vel, t):
            q = (v - self.mu) / self.sd
            d = np.linalg.norm(self.Xn - q, axis=1)
            idx = np.argpartition(d, self.k - 1)[:self.k]
            return self.Y[idx].mean(axis=0)
        return policy


# ── the experiment ──────────────────────────────────────────────────────────
def _starts_for(legal: np.ndarray, obj_xy: np.ndarray,
                rng: np.random.RandomState) -> List[np.ndarray]:
    d = np.linalg.norm(legal[:, :2] - obj_xy[None, :], axis=1)
    ok = np.where((d >= START_R_LO) & (d <= START_R_HI))[0]
    if len(ok) < N_STARTS:
        return []
    pick = rng.choice(ok, size=N_STARTS, replace=False)
    return [np.array(legal[i][:2], dtype=float) for i in pick]


def _experiment(seed: int) -> dict:
    w = W0(seed=seed, j0=J0, alpha=ALPHA)
    rng = np.random.RandomState(seed * 7919 + 101)
    legal = w.legal_spawns()
    objs = _live_objects(w)
    cells = [(v, o) for o in objs for v in VERBS]

    starts: Dict[str, List[np.ndarray]] = {}
    for o in objs:
        s = _starts_for(legal, objs[o][0], rng)
        for v in VERBS:
            starts[_cell(v, o)] = s

    # ── the privileged planner: own target, and one alternate verb ─────────
    demo_X: List[np.ndarray] = []
    demo_Y: List[np.ndarray] = []
    calib_X: List[np.ndarray] = []
    calib_Y: List[np.ndarray] = []
    calib_cell = _cell("approach", sorted(objs)[0])
    own_hit: Dict[str, List[bool]] = {}
    plur_hit: Dict[str, List[bool]] = {}
    finite_ok = True

    for v, o in cells:
        c = _cell(v, o)
        oxy, ogids = objs[o]
        alt = _alt_verb(v)
        own_hit[c], plur_hit[c] = [], []
        for st in starts[c]:
            for who, verb in (("own", v), ("alt", alt)):
                pol = _planner(verb, oxy, st)
                rec: List[Tuple[np.ndarray, np.ndarray]] = []

                def taped(vv, xy, vel, t, _p=pol, _r=rec):
                    a = _p(vv, xy, vel, t)
                    _r.append((vv, a))
                    return a

                tr = _rollout(w, st, taped, oxy, ogids)
                finite_ok = finite_ok and tr["finite"]
                hit = _satisfies(verb, tr)
                if who == "own":
                    own_hit[c].append(hit)
                    for vv, a in rec:
                        demo_X.append(vv)
                        demo_Y.append(a)
                    if c == calib_cell:
                        for vv, a in rec:
                            calib_X.append(vv)
                            calib_Y.append(a)
                else:
                    plur_hit[c].append(hit and own_hit[c][-1])

    # ── the blind twin, and its liveness proof ─────────────────────────────
    blind = _Blind(np.array(demo_X), np.array(demo_Y))
    calib = _Blind(np.array(calib_X), np.array(calib_Y))
    oxy_c, ogids_c = objs[calib_cell.split("@")[1]]
    calib_hits = [
        _satisfies("approach", _rollout(w, st, calib.policy(), oxy_c, ogids_c))
        for st in starts[calib_cell]]
    calib_rate = float(np.mean(calib_hits)) if calib_hits else 0.0

    # ── necessity: the blind twin, and the free-target check ───────────────
    blind_rate: Dict[str, float] = {}
    rand_rate: Dict[str, float] = {}
    for v, o in cells:
        c = _cell(v, o)
        oxy, ogids = objs[o]
        bh, rh = [], []
        for j, st in enumerate(starts[c]):
            bh.append(_satisfies(v, _rollout(w, st, blind.policy(), oxy, ogids)))
            rrng = np.random.RandomState(seed * 611953
                                         + zlib.crc32(c.encode()) % 9973 + j)
            rh.append(_satisfies(v, _rollout(w, st, _random_policy(rrng),
                                             oxy, ogids)))
        blind_rate[c] = float(np.mean(bh)) if bh else 1.0
        rand_rate[c] = float(np.mean(rh)) if rh else 1.0

    # ── retention ──────────────────────────────────────────────────────────
    retained, excl = [], {}
    for v, o in cells:
        c = _cell(v, o)
        if not starts[c]:
            excl[c] = "no starts in the annulus"
            continue
        ach = float(np.mean(own_hit[c]))
        plu = float(np.mean(plur_hit[c]))
        if ach < ACH_FRAC:
            excl[c] = f"unachievable (planner {ach:.2f})"
        elif plu < PLUR_FRAC:
            excl[c] = f"no plurality (both-reached {plu:.2f})"
        elif blind_rate[c] > CHANCE_HI:
            excl[c] = f"blind twin above chance ({blind_rate[c]:.2f})"
        elif rand_rate[c] > CHANCE_HI:
            excl[c] = f"free target (random {rand_rate[c]:.2f})"
        else:
            retained.append(c)

    verbs = {c.split("@")[0] for c in retained}
    objects = {c.split("@")[1] for c in retained}
    per_verb = min([sum(1 for c in retained if c.startswith(v + "@"))
                    for v in verbs], default=0)
    per_obj = min([sum(1 for c in retained if c.endswith("@" + o))
                   for o in objects], default=0)
    cellset_ok = float(
        len(retained) >= MIN_CELLS and len(verbs) >= MIN_VERBS
        and len(objects) >= MIN_OBJECTS and per_verb >= MIN_PER_VERB
        and per_obj >= MIN_PER_OBJECT)

    verb_alive = 1.0
    for v in VERBS:
        best = max([float(np.mean(own_hit[_cell(v, o)]))
                    for o in objs if own_hit.get(_cell(v, o))], default=0.0)
        verb_alive = min(verb_alive, best)

    _RETAINED[seed] = sorted(retained)

    return {
        "retained_cells": len(retained),
        "n_candidates": len(cells),
        "n_objects": len(objs),
        "n_verbs_retained": len(verbs),
        "n_objects_retained": len(objects),
        "min_per_verb": per_verb,
        "min_per_object": per_obj,
        "cellset_ok": cellset_ok,
        "blind_calib_rate": round(calib_rate, 4),
        "blind_rate_max": round(max(blind_rate.values(), default=1.0), 4),
        "blind_rate_retained_max": round(
            max([blind_rate[c] for c in retained], default=0.0), 4),
        "rand_rate_retained_max": round(
            max([rand_rate[c] for c in retained], default=0.0), 4),
        "planner_reach_mean": round(
            float(np.mean([np.mean(own_hit[c]) for c in own_hit if own_hit[c]])), 4),
        "verb_alive_min": round(verb_alive, 4),
        "plurality_mean": round(
            float(np.mean([np.mean(plur_hit[c]) for c in plur_hit if plur_hit[c]])), 4),
        "obs_finite": 1.0 if finite_ok else 0.0,
        "excluded_seed0_only": "; ".join(f"{k}: {v}" for k, v in sorted(excl.items())),
    }


def _control(seed: int) -> dict:
    """THE PLANNER, STRIPPED — and the cross-seed intersection.

    The stripped planner is handed some OTHER cell's target, drawn uniformly
    from the candidates, and runs the identical servo. It must not satisfy BOTH
    the demanded cell's predicate and its plurality alternate; if it does,
    'achievable' was being read out of the world state rather than chosen.

    Measured on every CANDIDATE cell, not only the survivors — a control scored
    only where the experiment already succeeded is a control that cannot fail.
    """
    w = W0(seed=seed, j0=J0, alpha=ALPHA)
    rng = np.random.RandomState(seed * 15485863 + 7)
    legal = w.legal_spawns()
    objs = _live_objects(w)
    cells = [(v, o) for o in objs for v in VERBS]
    srng = np.random.RandomState(seed * 7919 + 101)
    starts: Dict[str, List[np.ndarray]] = {}
    for o in objs:
        s = _starts_for(legal, objs[o][0], srng)
        for v in VERBS:
            starts[_cell(v, o)] = s

    both = []
    for v, o in cells:
        c = _cell(v, o)
        if not starts[c]:
            continue
        alt = _alt_verb(v)
        oxy, ogids = objs[o]
        for st in starts[c]:
            wrong = [x for x in cells if x != (v, o)]
            wv, wo = wrong[int(rng.randint(len(wrong)))]
            tr = _rollout(w, st, _planner(wv, objs[wo][0], st), oxy, ogids)
            both.append(_satisfies(v, tr) and _satisfies(alt, tr))

    seeds_done = sorted(_RETAINED)
    common = set(_RETAINED[seeds_done[0]]) if seeds_done else set()
    for s in seeds_done[1:]:
        common &= set(_RETAINED[s])
    cv = {c.split("@")[0] for c in common}
    co = {c.split("@")[1] for c in common}
    cpv = min([sum(1 for c in common if c.startswith(v + "@")) for v in cv],
              default=0)
    cpo = min([sum(1 for c in common if c.endswith("@" + o)) for o in co],
              default=0)
    return {
        "stripped_both_rate": round(float(np.mean(both)) if both else 1.0, 4),
        "n_stripped": float(len(both)),
        "common_n": float(len(common)),
        "common_ok": float(len(common) >= MIN_CELLS and len(cv) >= MIN_VERBS
                           and len(co) >= MIN_OBJECTS and cpv >= MIN_PER_VERB
                           and cpo >= MIN_PER_OBJECT),
        "common_cells": ", ".join(sorted(common)),
    }


def _check(m: dict, c: dict):
    # VOID first: none of these is a refutation of the claim.
    if m["obs_finite"] < 1.0:
        return Status.VOID          # the world handed a policy a NaN
    if m["blind_calib_rate"] < CALIB_MIN:
        return Status.VOID          # the blind twin is not demonstrably alive
    if m["verb_alive_min"] < VERB_ALIVE:
        return Status.VOID          # a verb the privileged planner cannot do
    return (m["cellset_ok"] >= 1.0                 # EVERY seed, not the mean
            and m["blind_rate_retained_max"] <= CHANCE_HI
            and m["rand_rate_retained_max"] <= CHANCE_HI
            and c["common_ok"] >= 1.0
            and c["stripped_both_rate"] <= CTRL_MAX)


def _dry():
    """The check's own truth table. Every branch of the verdict, exercised."""
    def base(**kw):
        d = {"obs_finite": 1.0, "blind_calib_rate": 0.9, "verb_alive_min": 0.9,
             "cellset_ok": 1.0, "retained_cells": 14.0,
             "blind_rate_retained_max": 0.25, "rand_rate_retained_max": 0.0,
             "n_verbs_retained": 4.0, "min_per_verb": 2.0}
        d.update(kw)
        return d
    ctl = {"stripped_both_rate": 0.0, "common_ok": 1.0, "common_n": 13.0}
    rows = [
        ("all green", base(), ctl, True),
        ("a NaN reached a policy -> VOID", base(obs_finite=0.0), ctl, Status.VOID),
        ("blind twin cannot reproduce its own demo -> VOID",
         base(blind_calib_rate=0.5), ctl, Status.VOID),
        ("a verb the planner cannot perform -> VOID",
         base(verb_alive_min=0.25), ctl, Status.VOID),
        ("VOID outranks a failing control",
         base(blind_calib_rate=0.0), {**ctl, "stripped_both_rate": 1.0},
         Status.VOID),
        ("one seed short of the spread -> FAIL",
         base(cellset_ok=2.0 / 3.0), ctl, False),
        ("a retained cell the blind twin can do -> FAIL",
         base(blind_rate_retained_max=0.5), ctl, False),
        ("a retained cell a random body achieves -> FAIL",
         base(rand_rate_retained_max=0.5), ctl, False),
        ("the seeds certify different worlds -> FAIL",
         base(), {**ctl, "common_ok": 0.0}, False),
        ("the stripped planner reaches both -> FAIL",
         base(), {**ctl, "stripped_both_rate": 0.4}, False),
        ("chance band exactly on the bar is inside it",
         base(blind_rate_retained_max=CHANCE_HI,
              rand_rate_retained_max=CHANCE_HI), ctl, True),
        ("control exactly on its bar is inside it",
         base(), {**ctl, "stripped_both_rate": CTRL_MAX}, True),
    ]
    return [(name, _check(mm, cc), want, _check(mm, cc) == want)
            for name, mm, cc, want in rows]


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["LG.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    ok = True
    for name, got, want, good in _dry():
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name}: got={got} want={want}")
    print("dry table:", "PASS" if ok else "FAIL")
