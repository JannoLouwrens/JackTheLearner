"""SO.06 — A hand can reach into a running life, and it reaches ONLY through the world.

GOAL.md, the owner: *"their hands may leave things in his world for him to
find — food where he might look, a tool he has not made yet. Never
puppeteering: what is left must still be found, learned, and chosen by him."*
This is the FIXTURE half. It certifies the CHANNEL and claims nothing about
learning: SO.07 (retention), SO.08 (whose hands) and SO.09 (the accountant)
are all scored against it and all wait on it.

THE VENUE, and why no file outside `experiments/tests/` was touched to build
it. `OWNERS_HANDS.md` §4 proposed a new static gift body, which needs a
`gifts=` contract in `playground.py`. That file is declared in `IMPL_DEPS` by
**54 specs** and `experiments/w0.py` by **19**, so adding the parameter would
have marked 73 certificates stale to certify one channel — a mass re-buy
bought for a kwarg. It is also unnecessary: **W0 already contains the food this
sentence is about.** `drives.FOOD_GEOMS` is `{apple, obj0, obj1}`, all three
free bodies, and a hand that moves one of them is doing exactly what the owner
described — putting food where he might look. So the hand here does not create
matter; it RELOCATES an existing food item by writing its free-joint qpos.
That is a strictly stronger venue for the invariance claim than a new geom
would have been: the idle arm and the no-hand arm are the SAME MODEL, so
bit-identity tests the hand's code path rather than MJCF compilation.

THE FOUR THINGS MEASURED, per seed.

1. INERT — the hand present and dropping nothing is bit-identical to no hand.
   A `Hand` is attached and called at every one of `LIFE_DECISIONS` decisions
   with `drop=False`; per-decision blake2b digests of (qpos, qvel, ctrl),
   the six-modality observation, and the drive state must all match the
   no-hand arm exactly, and the action RNG's terminal state must be identical.
   This is the registry's *"changes NOTHING else"* clause and it is the leg
   that would catch a hand that reaches around the world instead of through it.

2. LEGAL / UNOCCUPIED / UNSEEN — a placement exists that is all three at once.
   The target is `body_xy + DROP_RANGE_M * u(theta)` where `theta` BISECTS two
   adjacent rays: at 16 rays the half-gap is 11.25 deg, so at 3.0 m the nearest
   ray passes 0.585 m from the target — an order of magnitude outside any food
   geom's radius, so unseen-ness is geometric rather than lucky. Legality is
   the arena bounds and the floor; unoccupancy is `ncon` over the moved geom
   after `mj_forward`. `falsified_by`'s first clause is exactly this leg.

3. PERCEPTIBLE — the same object placed ON ray `k` at `SEEN_RANGE_M` is read
   BY THAT RAY, at the right distance. This discharges *"visible in Jack's own
   senses within a declared time once he looks"* as a KNOWN-ANSWER probe:
   the ray IS his looking, and a fixture whose ground truth is computed beats
   one that waits for a random policy to wander into the answer (the
   `probe_objects` precedent — "where the ground truth must be KNOWN rather
   than sampled"). Whether he in fact walks over and eats it is SO.07's
   question and is scored there, not asserted here.

4. LOGGED — every placement carries (t, agent, object, position, need-state),
   and every food geom's position is recorded at each life's end. The second
   half is not bureaucracy: `w0.py:_place` deliberately omits `mj_resetData`,
   so food geoms DRIFT under traffic and the world never re-places them. A
   spec that logs only drops cannot separate the hand from that drift — which
   is the "food ratchet" `lc03_food_probe.py` was written to chase.

THE CONTROLS, both from the registry, both of which must fire.

  (1) POSITIVE — a hand that draws one value from the action RNG and nudges
      the body by `NUDGE_M`. The invariance detector MUST catch it: divergence
      located at or after the hand's first call, and an RNG mismatch. SO.04's
      rule: *a detector that cannot see its own positive control has measured
      nothing.*
  (2) NEGATIVE — the object placed OUTSIDE every ray's reach AND behind
      occlusion (30 m along ray `k`, beyond the perimeter wall). The
      observation must NOT change. An observation that moves for an object he
      cannot see is a side-channel, not a sense. An occlusion-only leg (the
      object placed just BEHIND whatever ray `k` already hits, still inside the
      arena) is reported beside it, because far-and-occluded alone cannot tell
      "too far" from "hidden".

COMPARING VISION: distances only (`_vision()[0::2]`), never textures. W0's
`_vision` draws from `self._rng` when a ray lands on the noisy-TV panel, so a
texture comparison across two calls at one world state is not reproducible by
construction. Distances are RNG-free.

THE REFERENCE VIEW, and why this fixture needs no lucky world. Every
placement leg is scored against `ref` — the ray distances read with the donor
PARKED off-world at 40 m, i.e. what he sees with the gift absent. A first
version instead searched for a food item that happened to be invisible where
it already lay, and at seed 0 no such item existed: all three are visible, or
occlude something that is. That search made the verdict depend on the world's
random layout rather than on the hand. Parking makes "unseen" mean *the same
as if there were no gift*, which is what the sentence meant all along.

WHAT IS SEARCHED AND WHAT IS DECLARED. `falsified_by`'s first clause — *no
placement exists that is simultaneously legal, reachable and initially
unseen* — is an EXISTENCE claim about the venue, so all 16 bisectors are
tried and `n_unseen_candidates` is reported; "the one I tried failed" would
not have refuted it. Ray `k` is likewise taken from the live world (the first
ray clear out to `SEEN_RANGE_M + radius + CLEAR_MARGIN_M`), never hard-coded:
a fixed ray index is a constant a mutated world can silently invalidate,
which is the T0.14 mistake.

SMOKE RUN (2026-09-04, this box, seed 0, at LIFE_DECISIONS=20 /
DROP_DECISION=10, before the registered run): inert bit-identical over all 20
decisions with obs, needs and RNG all matching; 14 of 16 bisectors legal,
unoccupied and unseen; the seen probe read 2.8698 m against the known answer
3.0 - 0.131 = 2.869 (error 0.8 mm) on exactly one ray; the nudge control
caught at decision 0 with an RNG mismatch; far-occluded and occluded-only both
unchanged. NOTHING WAS SIZED FROM IT — every gate here is an exact invariant, a
declared geometric margin, or a known answer computed from the model.

TWO THINGS THE SMOKE RUN CORRECTED, recorded because both were wrong in a way
that would have read as a world finding rather than a fixture bug. (a) The
first version floated the drop at RAY HEIGHT; a food geom's radius (0.131 m)
exceeds the rover's ray plane (0.117 m), so every placement penetrated the
floor and `ncon` was never zero — 0 of 16 candidates, which would have read as
*the venue cannot host provisioning* when it was the fixture holding the object
in the ground. Drops now REST on the floor, which is also what a dropped thing
does one step later. (b) "Unoccupied" was contact with anything at all, which a
resting object can never satisfy; it now means contact with anything that is
NOT ground, which is the line W0's own `_penetrating()` already draws.

VOID LANES, all instrument-death rather than refutation: fewer than
`MIN_LIVE_RAYS` rays returning a finite distance (occlusion untestable), or no
open ray at all (the known-answer visibility probe is not well posed). A world
that offers no legal unseen placement is NOT a VOID lane — that is exactly
`falsified_by`'s first clause and it reads FAIL.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .. import drives
from ..w0 import N_RAYS, RAY_MAX_RANGE, RAY_Z_OFFSET, W0

# The verdict composes over the world, the body and the drive layer: a change
# to any of them must read as drift, not as a standing certificate.
IMPL_DEPS = ["playground.py", "experiments/w0.py", "experiments/drives.py"]

# --- the life ---------------------------------------------------------------
LIFE_DECISIONS = 200          # 40 simulated seconds at 0.2 s per decision
DROP_DECISION = 100           # mid-life, so there is life on both sides of it
J0, ALPHA = 1.0, 0.02         # PS.01's calibrated impact channel, as LC.02 uses

# --- the hand's declared reach ---------------------------------------------
DROP_RANGE_M = 3.0            # where an unseen drop goes
SEEN_RANGE_M = 3.0            # where the known-answer visibility probe goes
FAR_RANGE_M = 30.0            # control 2: outside the arena and every ray
OCCLUDED_MARGIN_M = 0.5       # control 2b: this far BEHIND what ray k hits
CLEAR_MARGIN_M = 0.5          # a probe ray must be clear this far BEYOND
                              # the target, or "it appeared" is unreadable
NUDGE_M = 1e-3                # control 1's deliberate body perturbation

# --- pre-registered instrument constants ------------------------------------
DIGEST_BYTES = 16
RAY_DIST_TOL = 0.20           # m, on a ray reading of a sphere's near surface
MIN_LIVE_RAYS = 1             # a world where no ray hits anything cannot host
                              # an occlusion control
EPS = 0.0                     # bit-identity is EXACT; no tolerance is offered


# ── the hand ────────────────────────────────────────────────────────────────
class Hand:
    """A declared external agent whose entire action surface is one write to
    an existing food item's free-joint qpos, plus a log line.

    It is given the life's action RNG deliberately: the honest hand must never
    draw from it, and control 1 proves the detector notices when one does.
    """

    def __init__(self, agent: str, world: W0, item: str,
                 rng: np.random.RandomState, sabotage: bool = False):
        self.agent = agent
        self.w = world
        self.item = item
        self.rng = rng
        self.sabotage = sabotage
        self.log: list = []
        m = world.model
        bid = int(m.body(item).id)
        jid = int(m.body_jntadr[bid])
        if jid < 0 or int(m.jnt_type[jid]) != int(world.mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"{item} is not a free body; the hand cannot "
                               f"relocate it without a model edit")
        self.bid, self.qadr = bid, int(m.jnt_qposadr[jid])
        self.dadr = int(m.jnt_dofadr[jid])
        self.gid = int(m.geom(item).id)
        self.radius = float(m.geom_size[self.gid][0])

    # -- the only thing it may do to the world ---------------------------
    def place(self, xyz, log: bool = True) -> None:
        d = self.w.data
        d.qpos[self.qadr:self.qadr + 3] = xyz
        d.qpos[self.qadr + 3:self.qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[self.dadr:self.dadr + 6] = 0.0
        self.w.mujoco.mj_forward(self.w.model, d)
        if not log:
            return
        self.log.append({
            "t": float(self.w.sim_seconds),
            "agent": self.agent,
            "object": self.item,
            "position": [float(v) for v in xyz],
            "needs": [float(self.w.drives.state.e),
                      float(self.w.drives.state.i),
                      float(self.w.drives.state.w)],
        })

    def step(self, drop: bool = False, target=None) -> None:
        """Called every decision. Dropping nothing must cost the world nothing."""
        if self.sabotage:
            self.rng.uniform()                       # THE deliberate draw
            self.w.data.qpos[self.w.ix["root_qposadr"]] += NUDGE_M
        if drop and target is not None:
            self.place(target)

    def end_of_life_positions(self) -> dict:
        """Every food geom's position, so the hand can be separated from the
        drift `w0.py:_place` deliberately never undoes."""
        d = self.w.data
        out = {}
        for name in drives.FOOD_GEOMS:
            try:
                gid = int(self.w.model.geom(name).id)
            except (KeyError, ValueError):
                continue
            out[name] = [float(v) for v in d.geom_xpos[gid]]
        return out


# ── instruments ─────────────────────────────────────────────────────────────
def _state_digest(data) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(data.qpos.tobytes())
    h.update(data.qvel.tobytes())
    h.update(data.ctrl.tobytes())
    return h.digest()


def _obs_digest(obs: dict) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    for k in sorted(obs):
        h.update(k.encode())
        h.update(np.asarray(obs[k], dtype=np.float32).tobytes())
    return h.digest()


def _rng_digest(rng: np.random.RandomState) -> bytes:
    _kind, keys, pos, has_gauss, cached = rng.get_state()
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(keys.tobytes())
    h.update(struct.pack("<iid", pos, has_gauss, cached))
    return h.digest()


def _vis_d(w: W0) -> np.ndarray:
    """The 16 ray DISTANCES. Textures are excluded on purpose: `_vision` draws
    from `w._rng` when a ray lands on the noise panel, so texture is not
    reproducible across two reads of one world state."""
    return np.asarray(w._vision()[0::2], dtype=np.float64).copy()


def _world(seed: int) -> W0:
    return W0(seed=seed, j0=J0, alpha=ALPHA, lethal=False)


def _ray_angle(k: int) -> float:
    return 2.0 * math.pi * k / N_RAYS


def _target(w: W0, angle: float, dist: float, radius: float) -> np.ndarray:
    """A point at `dist` along `angle` from the body, RESTING ON THE FLOOR.

    Resting rather than floating at ray height, because a dropped thing rests:
    a floating placement would fall the moment the life resumed and the
    logged position would be a lie one step later. The ray still reads it —
    the rover's ray plane sits ~0.117 m up and a food geom's radius is
    0.131 m, so the centre is inside the ray's own radius of the sphere.
    """
    p = np.asarray(w.data.xpos[w.rover_bid], dtype=float)
    return np.array([p[0] + dist * math.cos(angle),
                     p[1] + dist * math.sin(angle),
                     radius])


def _life(seed: int, hand: bool, sabotage: bool = False) -> dict:
    """One life of `LIFE_DECISIONS` decisions. `hand` attaches an idle Hand
    that is called every decision and drops nothing."""
    w = _world(seed)
    rng = np.random.RandomState(seed * 7717 + 3)
    h = Hand("owner", w, "apple", rng, sabotage=sabotage) if hand else None
    st, ob, nd = [], [], []
    for _ in range(LIFE_DECISIONS):
        if h is not None:
            h.step(drop=False)
        w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
        st.append(_state_digest(w.data))
        ob.append(_obs_digest(w.observe()))
        nd.append((w.drives.state.e, w.drives.state.i, w.drives.state.w))
    return {"state": st, "obs": ob, "needs": nd,
            "rng": _rng_digest(rng), "report": w.report()}


def _first_divergence(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


DONOR = "obj0"                # a floor food item: "food where he might look"
PARK_XY = 40.0                # the reference park — off-world, behind the
                              # perimeter wall, beyond every ray


def _park(hand) -> None:
    """Put the donor out of the scene. `_vis_d` read here is the REFERENCE:
    what he sees with the gift absent. Every placement leg is scored against
    it, which is why this fixture needs no search for a food item that happens
    to be hidden where it lies — a question about the world's random layout,
    not about hands."""
    hand.place(np.array([PARK_XY, PARK_XY, 1.0]), log=False)


def _obstructed(w: W0, gid: int) -> int:
    """Contacts between the placed item and anything that is NOT ground.

    "Unoccupied" cannot mean "touching nothing": a thing resting on the floor
    touches the floor. W0's own legality predicate draws the line in exactly
    this place — `_penetrating()` asks whether the body is inside something
    that is not ground — and this reuses its mask rather than writing a second
    definition of the same idea.
    """
    n = 0
    for c in range(int(w.data.ncon)):
        con = w.data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if gid not in (g1, g2):
            continue
        other = g2 if g1 == gid else g1
        if not bool(w._ground_mask[other]):
            n += 1
    return n


def _setup(seed: int):
    """A world stepped to the drop decision, a hand on the donor, the donor
    PARKED, and the reference view read with the gift absent."""
    w = _world(seed)
    rng = np.random.RandomState(seed * 7717 + 3)
    for _ in range(DROP_DECISION):
        w.decide(rng.uniform(-1.0, 1.0, w.action_dim))
    hand = Hand("owner", w, DONOR, rng)
    _park(hand)
    return w, hand, _vis_d(w)


def _open_rays(ref: np.ndarray, radius: float) -> list:
    """Rays with nothing in the way out to SEEN_RANGE_M plus clearance — the
    only rays on which a known-answer visibility probe is well posed. Derived
    from the live world; a hard-coded ray index would be a constant a mutated
    world could silently invalidate (the T0.14 mistake)."""
    need = SEEN_RANGE_M + radius + CLEAR_MARGIN_M
    return [k for k in range(N_RAYS) if ref[k] * RAY_MAX_RANGE > need]


def _has_geom(w: W0, name: str) -> bool:
    try:
        w.model.geom(name)
        return True
    except (KeyError, ValueError):
        return False


def _probe(seed: int) -> dict:
    """The placement and perception legs, run in their own world so that no
    extra `observe()` or `_vision()` call can perturb the invariance arms."""
    w, hand, ref = _setup(seed)
    arena = float(w.params.arena_size)
    out = {"n_rays_hitting": float(np.sum(ref < 1.0))}

    opens = _open_rays(ref, hand.radius)
    out["n_open_rays"] = float(len(opens))
    if not opens:
        return out                       # VOID lane: no clear line of sight
    k = opens[0]
    out["ray_k"] = float(k)

    # -- leg 2: does a legal, unoccupied, currently-unseen placement EXIST? --
    # Searched over all 16 ray BISECTORS, because `falsified_by`'s first
    # clause is an existence claim about the venue and "the one I tried
    # failed" is not its refutation. The bisector is where unseen-ness is
    # geometric: half-gap 11.25 deg puts the nearest ray 0.585 m off a 3.0 m
    # target, an order of magnitude outside any food geom's radius.
    cands = []
    for j in range(N_RAYS):
        t = _target(w, _ray_angle(j) + math.pi / N_RAYS, DROP_RANGE_M,
                    hand.radius)
        if not (abs(t[0]) < arena and abs(t[1]) < arena and t[2] > 0.0):
            continue
        hand.place(t, log=False)
        if _obstructed(w, hand.gid) == 0 and np.all(_vis_d(w) == ref):
            cands.append(t)
    out["n_unseen_candidates"] = float(len(cands))
    out["placement_exists"] = float(bool(cands))
    out["unseen_margin_m"] = round(
        DROP_RANGE_M * math.sin(math.pi / N_RAYS) - hand.radius, 4)
    if not cands:
        _park(hand)
        return out                       # the venue cannot host provisioning

    t_unseen = cands[0]
    hand.place(t_unseen)                                  # LOGGED placement 1
    out["placement_legal"] = float(abs(t_unseen[0]) < arena
                                   and abs(t_unseen[1]) < arena
                                   and t_unseen[2] > 0.0)
    out["placement_unoccupied"] = float(_obstructed(w, hand.gid) == 0)
    out["placement_unseen"] = float(np.all(_vis_d(w) == ref))

    # -- leg 3: perceptible — the same object ON ray k ----------------------
    t_seen = _target(w, _ray_angle(k), SEEN_RANGE_M, hand.radius)
    hand.place(t_seen)                                    # LOGGED placement 2
    v = _vis_d(w)
    changed = np.flatnonzero(v != ref)
    out["seen_n_rays_changed"] = float(changed.size)
    out["seen_ray_is_k"] = float(changed.size == 1 and int(changed[0]) == k)
    read_m = float(v[k] * RAY_MAX_RANGE)
    expect_m = SEEN_RANGE_M - hand.radius
    out["seen_dist_m"] = round(read_m, 4)
    out["seen_dist_err_m"] = round(abs(read_m - expect_m), 4)
    out["seen_dist_ok"] = float(abs(read_m - expect_m) <= RAY_DIST_TOL)

    # -- leg 4: the log ------------------------------------------------------
    fields = ("t", "agent", "object", "position", "needs")
    out["log_rows"] = float(len(hand.log))
    out["log_rows_ok"] = float(len(hand.log) == 2)
    out["log_fields_ok"] = float(bool(hand.log) and all(
        all(f in row for f in fields) for row in hand.log))
    eol = hand.end_of_life_positions()
    want = {n for n in drives.FOOD_GEOMS if _has_geom(w, n)}
    out["endpos_logged_ok"] = float(set(eol) == want
                                    and all(len(q) == 3 for q in eol.values()))
    return out


def _experiment(seed: int) -> dict:
    none = _life(seed, hand=False)
    idle = _life(seed, hand=True)

    first_div = _first_divergence(none["state"], idle["state"])
    obs_div = _first_divergence(none["obs"], idle["obs"])
    needs_max = max(
        (max(abs(a - b) for a, b in zip(x, y))
         for x, y in zip(none["needs"], idle["needs"])), default=0.0)

    m = {
        "inert_bit_identical": float(first_div == -1),
        "inert_first_divergence": float(first_div),
        "inert_obs_identical": float(obs_div == -1),
        "inert_rng_match": float(none["rng"] == idle["rng"]),
        "inert_needs_max_diff": float(needs_max),
        "decisions": float(none["report"]["decisions"]),
        "decisions_match": float(none["report"]["decisions"]
                                 == idle["report"]["decisions"]),
    }
    m.update(_probe(seed))

    gates = ("inert_bit_identical", "inert_obs_identical", "inert_rng_match",
             "decisions_match", "placement_exists", "placement_legal",
             "placement_unoccupied", "placement_unseen", "seen_ray_is_k",
             "seen_dist_ok", "log_rows_ok", "log_fields_ok",
             "endpos_logged_ok")
    m["provision_channel_ok"] = float(
        all(m.get(g, 0.0) == 1.0 for g in gates)
        and m["inert_needs_max_diff"] <= EPS)
    return m


def _control(seed: int) -> dict:
    """Both registry controls.

    (1) POSITIVE: a hand that draws one RNG value and nudges the body. The
        detector must catch it AT DECISION 0 — the hand acts before the first
        decision, so a divergence appearing later would mean the nudge did not
        take, and this control could not claim the detection.
    (2) NEGATIVE: the object placed outside every ray's reach AND behind
        occlusion. The observation must NOT change.
    """
    none = _life(seed, hand=False)
    bad = _life(seed, hand=True, sabotage=True)
    first_div = _first_divergence(none["state"], bad["state"])
    detected = first_div != -1
    c = {
        "c_nudge_first_divergence": float(first_div),
        "c_nudge_detected": float(detected),
        "c_nudge_in_hand_path": float(detected and first_div == 0),
        "c_nudge_rng_mismatch": float(none["rng"] != bad["rng"]),
        "c_nudge_obs_detected": float(
            _first_divergence(none["obs"], bad["obs"]) != -1),
    }
    c["c_detector_catches"] = float(
        c["c_nudge_detected"] == 1.0 and c["c_nudge_in_hand_path"] == 1.0
        and c["c_nudge_rng_mismatch"] == 1.0)

    w, hand, ref = _setup(seed)
    c["c_n_rays_hitting"] = float(np.sum(ref < 1.0))
    opens = _open_rays(ref, hand.radius)
    c["c_n_open_rays"] = float(len(opens))
    if not opens:
        return c
    k = opens[0]

    # THE REGISTRY CONTROL: outside every ray's reach AND behind occlusion.
    hand.place(_target(w, _ray_angle(k), FAR_RANGE_M, hand.radius), log=False)
    c["c_far_occluded_unchanged"] = float(np.all(_vis_d(w) == ref))

    # Reported beside it: occlusion ALONE, inside the arena, just behind
    # whatever a blocked ray already hits. Far-and-occluded together cannot
    # tell "too far" from "hidden", and only this leg can.
    blocked = [j for j in range(N_RAYS)
               if ref[j] * RAY_MAX_RANGE < RAY_MAX_RANGE - OCCLUDED_MARGIN_M]
    c["c_n_blocked_rays"] = float(len(blocked))
    if blocked:
        j = blocked[0]
        behind = float(ref[j] * RAY_MAX_RANGE) + OCCLUDED_MARGIN_M
        hand.place(_target(w, _ray_angle(j), behind, hand.radius), log=False)
        c["c_occluded_only_unchanged"] = float(np.all(_vis_d(w) == ref))
        c["c_occluder_dist_m"] = round(float(ref[j] * RAY_MAX_RANGE), 4)
    c["c_log_empty"] = float(len(hand.log) == 0)
    return c


def _check(m: dict, c: dict):
    # VOID, never FAIL: instrument death. A world that hides no food, or whose
    # rays hit nothing, refutes nothing about hands.
    if (m.get("n_rays_hitting", 0.0) < MIN_LIVE_RAYS
            or c.get("c_n_rays_hitting", 0.0) < MIN_LIVE_RAYS):
        return Status.VOID
    if m.get("n_open_rays", 0.0) < 1.0 or c.get("c_n_open_rays", 0.0) < 1.0:
        return Status.VOID
    return (m.get("provision_channel_ok", 0.0) == 1.0
            and c.get("c_detector_catches", 0.0) == 1.0
            and c.get("c_far_occluded_unchanged", 0.0) == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["SO.06"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
