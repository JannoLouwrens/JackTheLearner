"""BA.03 envelope probe — does a wall give this body DIRECTIONAL brace authority?

NOT A SPEC. This file writes nothing to the ledger and declares no claim. It is
the measurement that must exist BEFORE `ba_03_braces_against_a_surface.py` is
written, and it exists because BA.02 VOIDed its rig three times, at ~46 min a
run, before anyone measured whether the contrast it gated on had any headroom
at all (`docs/DECISIONS_NEEDED.md` D8). The precedent for a committed probe
file is `t301_shuffle_probe.py`.

WHAT BA.03 NEEDS FROM A PROBE, in its own registry words:

  * *"a learner ... PLACES ITS SUPPORT ON THE LEAN SIDE"* — so the body must
    HAVE a lean side it can place support on. That is an action-space question,
    and §1 below answers it with arithmetic before any simulation.
  * *"the null D8's probes prove is the binding one — THE BEST FIXED BLIND
    POSTURE"* — so the probe must find that posture, not assume `random`.
  * *"SIZING IS PRE-REGISTERED AS A REQUIREMENT ... size k_fit and N_EVAL
    against MEASURED noise in the pilot and amend the TIER, never the
    thresholds"* — so the probe must report a paired SE, not a mean.

Every policy here is HAND-WRITTEN and every keyed policy is an ORACLE (it is
handed the true lean side). They are envelope UPPER BOUNDS on what a trained
arm of the same action shape could reach, exactly as D8's four scratch probes
were for BA.02 — never a claim arm, never a gate, never a ledger row.

## 1. THE ARITHMETIC THAT COMES FIRST — which of the body's actuators can
##    express "on the lean side" AT ALL

`playground._rover_fragments` gives each arm a `reach` slide on axis `0 1 0`
and a `lift` slide on axis `0 0 1`, and pins the two arm bodies at body
x = -0.10 (L) and +0.10 (R). So the arm-pair centre of mass sits at

    x_com = (0.4 * -0.10 + 0.4 * +0.10) / 0.8 = 0   for ALL reach, lift

**Arm POSITION has identically zero body-x authority.** No posture — symmetric
or differential, reach or lift — moves the CoM laterally. A hypothesis about
placing support on the lean side is unrealisable through the slides, and a
probe that varies only postures will return a confident negative that is about
the probe.

ADHESION is different, and it is the whole reason this spec is askable: 900 N
at ONE hand and not the other is a force applied 0.10 m off the body axis, i.e.
a moment about body-y — which is precisely "which side". That is the actuator
BA.03's hypothesis names, and v1/v2 below never varied it.

## 2. WHAT WAS MEASURED, in order, on seed 90 against the arena wall

The braceable surface is `wall1` (`pos 0 -6 1.25`, half-thickness 0.05, so an
inner face at y = -5.95, spanning z 0 to 2.5). It is the only large flat
vertical surface a hand can reach: the hands sweep z 0.45-1.20 at rest, which
puts `welded_block` (top 0.30) and `fulcrum` (top 0.18) out of range, and the
ladder rails sit at x = -+0.25 against hands pinned at x = -+0.10. Standoff D
is measured from that face. Note the site is OUTSIDE `w.legal_spawns()`, whose
grid stops at arena_size - SPAWN_MARGIN = -+5.25: `SPAWN_MARGIN` sizes the
uniform-spawn probe, it is not a legality law, and every site used here is
checked against the world's own predicate `w._penetrating()`.

**v1 (D sweep, /data/ba03_probe.log).** Two facts that reshaped everything
after it. (a) At D = 0.15 a fall AIMED AT THE WALL never topples under any
policy including plain `hold` — 12.000 s of a 12.0 s horizon, with 285 torso
contacts. The wall is a passive BACKSTOP there, and leaning on it is available
to every arm including the blind twin; that is the confound `ba_02`'s own
docstring predicted ("a body that topples against a wall and rests on it would
score upright longer by leaning, which is shelter, not catching"). (b) Floor
adhesion held on constantly is a dominant BLIND strategy — `blind_out_grip`
9.05 s on the away side against `hold`'s 0.725 s. Any null that is not allowed
to grip is not the best fixed blind posture.

**v2 (fore/aft falls, adhesion OFF, /data/ba03_probe2.log).** With the grip
disabled and falls aimed -+y, keying on the channel is WORSE than the best
fixed blind posture at both standoffs — paired, over the same episodes:

    D=0.28   keyed_grav -1.540 +- 0.910 s   keyed_oracle -1.500 +- 0.901 s
    D=0.40   keyed_grav -2.960 +- 1.050 s   keyed_oracle -2.920 +- 1.039 s

The oracle and the channel-reading arm agree to within 0.04 s at both
standoffs, so the deficit is not a decoding failure. This is §1 showing up in
simulation: with adhesion off, the reach slide is the only lever and it has no
sides, so "extend toward the wall" is simply a worse fixed posture than "hands
up" — on BOTH sides of the fall.

**v3 (lateral falls, single-hand adhesion) — THE DECISIVE ONE, and it reverses
the reading v2 alone would have supported.** Falls aimed -+x; every policy
holds the SAME extended posture so both hands are at the wall; the only thing
that differs is WHICH HAND GRIPS. Recorded at N=10 (5/side, D=0.28):

    hold            0.840 +- 0.058
    out_nogrip      0.860 +- 0.067      extending without grip buys nothing
    out_gripboth    7.660 +- 0.685      <- the best fixed BLIND posture
    out_keyed       2.220 +- 0.351      grip the hand on the LOW side
    out_anti        9.460 +- 0.538      grip the hand on the HIGH side

    paired out_anti - out_gripboth:  +1.800 +- 0.538 s   (3.3 sigma)
    paired out_keyed - out_gripboth: -5.440 +- 0.341 s

A single-hand grip keyed on the lean side spans **7.2 s** between its two
labellings, and the better labelling beats the best symmetric blind posture.
So the directional authority BA.02's D8 could not find on open ground EXISTS
at a wall, and it lives in the adhesion channel, not the slides.

Two things the spec author must carry from this, both of which are traps:

  * **The winning hand is the HIGH-side one, not the low-side one** — grip
    where the body is rising and pull, rather than where it is falling and
    push. The probe's first labelling was the intuitive one and it was 5.4 s
    wrong. THE SPEC MUST NOT ENCODE EITHER LABEL. Which hand is correct is
    what the learner is being asked to discover; writing it into the rig would
    make the claim a readout of the author's guess. `brace_side_accuracy` is
    reported against the lean side as BA.03 registers it, and its SIGN is a
    finding, not a gate.
  * **`out_gripboth` at 7.66 s is the null.** Against `random` or `hold`
    (0.84 s) almost anything looks like balance. BA.03's registry already says
    this; the number is here so nobody has to take it on faith.

**Channel legibility, same run:** `sign(grav_body[0])` at the first decision
after the kick separated the two lean sides on 10 of 10 episodes (left leans
-0.080 to -0.278, right leans +0.110 to +0.368). The information the claim
needs is in BA.01's channel and is linearly available at decision one.

## 3. WHAT THIS PROBE DOES NOT ESTABLISH

It does not say BA.03 passes. Its keyed arms are oracles, its N is small, it
runs one seed and one standoff, and a CEM learner on the registered budget is
a different question from an envelope ceiling — D8's second bullet
(k_fit ~ 119, N_EVAL's SE ~0.22 s) is the standing warning and is unrepealed.
What it establishes is that the contrast BA.03 gates on has a measured ceiling
ABOVE its floor, which is exactly the thing BA.02 never had.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from ..protocol import borrow_metrics                          # noqa: E402
from ..w0 import W0, SIM_S_PER_DECISION                        # noqa: E402
from .ba_01_feels_the_fall import (KICK_JIT, KICK_OMEGA_P,     # noqa: E402
                                   TOPPLE_UP, _tilt_quat)

# wall1's inner face: pos (0, -6, 1.25), box half-size (6, 0.05, 1.25).
# Derived in `_wall_face` from the model rather than trusted from this line.
WALL_GEOM = "wall1"
STANDOFF = 0.28          # m from the face to the body axis; see §2
HORIZON = 60             # decisions = 12.0 s
T_SETTLE = 3             # hold decisions before the tilt+kick (BA.01/BA.02)
TILT_LOG10_DEG = (0.6, 1.4)   # BA.02 V3's draw: 4-25 deg, where slides matter
N_PACKS = 20             # 10 per side; v3's recorded run used 10


def _wall_face(w: W0) -> float:
    """The y of wall1's inner face, from the model (never the constant above).

    `w.ix["geom"]` maps only the rover's own geoms (it feeds `touch_gids`), so
    the arena wall is resolved through MuJoCo's name table."""
    gid = w.mujoco.mj_name2id(w.model, w.mujoco.mjtObj.mjOBJ_GEOM, WALL_GEOM)
    if gid < 0:
        raise RuntimeError(f"no geom named {WALL_GEOM!r} in this world")
    return float(w.model.geom_pos[gid][1]) + float(w.model.geom_size[gid][1])


class _Rig:
    def __init__(self, seed: int = 90):
        b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
        if not b.ok:
            raise RuntimeError(f"PS.01 calibration unavailable: {b.refusal}")
        self.w = W0(seed=seed, j0=b.values["j0_ms"], alpha=b.values["alpha"],
                    lethal=False)
        self.lo = np.asarray(self.w.model.actuator_ctrlrange[:4, 0], float)
        self.hi = np.asarray(self.w.model.actuator_ctrlrange[:4, 1], float)
        self.site = (0.0, _wall_face(self.w) + STANDOFF)
        self.hold = self.act(0.0, 0.0)

    def act(self, reach: float, lift: float,
            adh_l: float = -1.0, adh_r: float = -1.0) -> np.ndarray:
        """Joint-space targets -> W0's 8-vector. PS.03's phantom-servo scar:
        a = 0 is MID-range, so every target is mapped through the live
        ctrlrange rather than written as a raw action value."""
        a = np.zeros(8)
        t = np.array([reach, lift, reach, lift])
        a[:4] = 2.0 * (t - self.lo) / (self.hi - self.lo) - 1.0
        a[4], a[5] = adh_l, adh_r
        return np.clip(a, -1.0, 1.0)

    def site_is_legal(self) -> bool:
        """The world's OWN predicate, at a site its coarse spawn grid omits."""
        self.w._place(*self.site)
        self.w.mujoco.mj_forward(self.w.model, self.w.data)
        return not self.w._penetrating()

    def pack(self, rng: np.random.RandomState, left: bool) -> dict:
        """One LATERAL fall. `aim` is -+x so the fall has a side at all; the
        tilt and kick draws are BA.02 V3's, unchanged."""
        theta = math.radians(10.0 ** rng.uniform(*TILT_LOG10_DEG))
        mag = theta * KICK_OMEGA_P * 10.0 ** rng.uniform(*KICK_JIT)
        u = rng.randn(3)
        u /= max(float(np.linalg.norm(u)), 1e-12)
        return {"site": self.site, "aim": math.pi if left else 0.0,
                "theta": theta, "kick": u * mag, "left": left}

    def episode(self, pack: dict, act_fn) -> tuple:
        """Returns (decisions upright, grav_body[0] at the first decision)."""
        w, mj = self.w, self.w.mujoco
        qa, da = w.ix["root_qposadr"], w.ix["root_dofadr"]
        w.respawn(at=pack["site"])
        for _ in range(T_SETTLE):
            w.decide(self.hold)
        q0 = w.data.qpos[qa + 3:qa + 7].copy()
        qt = _tilt_quat(pack["theta"], pack["aim"])
        out = np.zeros(4)
        mj.mju_mulQuat(out, qt, q0)
        w.data.qpos[qa + 3:qa + 7] = out
        w.data.qvel[da:da + 6] = 0.0
        w.data.qvel[da + 3:da + 6] = pack["kick"]
        mj.mj_forward(w.model, w.data)
        gx0 = None
        for t in range(HORIZON):
            R = np.asarray(w.data.xmat[w.rover_bid], float).reshape(3, 3)
            if float(w.data.xmat[w.rover_bid][8]) < TOPPLE_UP:
                return t, gx0
            gb = -R.T @ np.array([0.0, 0.0, 1.0])
            if gx0 is None:
                gx0 = float(gb[0])
            w.decide(act_fn(gb, pack))
        return HORIZON, gx0


def _policies(rig: _Rig) -> dict:
    """Hand-written envelope policies. The two keyed ones are ORACLES: they
    read `pack["left"]`, not the channel, so they bound what ANY policy of this
    action shape could reach. `out_anti` grips the HIGH-side hand — the
    labelling is deliberately NOT named for its outcome, because which side
    wins is the finding (§2)."""
    out = dict(reach=-0.25, lift=0.10)
    return {
        "hold":         lambda g, p: rig.hold,
        "out_nogrip":   lambda g, p: rig.act(**out),
        "out_gripboth": lambda g, p: rig.act(**out, adh_l=1.0, adh_r=1.0),
        # handL sits at body x = -0.10, so a LEFT (-x) lean is handL's side.
        "out_keyed":    lambda g, p: rig.act(**out, adh_l=1.0, adh_r=-1.0)
                        if p["left"] else rig.act(**out, adh_l=-1.0, adh_r=1.0),
        "out_anti":     lambda g, p: rig.act(**out, adh_l=-1.0, adh_r=1.0)
                        if p["left"] else rig.act(**out, adh_l=1.0, adh_r=-1.0),
    }


def probe(seed: int = 90, n_packs: int = N_PACKS) -> dict:
    t0 = time.time()
    rig = _Rig(seed)
    legal = rig.site_is_legal()
    rng = np.random.RandomState(seed)
    packs = [rig.pack(rng, left=(i % 2 == 0)) for i in range(n_packs)]
    pols = _policies(rig)
    ups = {k: {"left": [], "right": []} for k in pols}
    signs = []
    for p in packs:
        side = "left" if p["left"] else "right"
        for name, fn in pols.items():
            up, gx0 = rig.episode(p, fn)
            ups[name][side].append(up * SIM_S_PER_DECISION)
            if name == "hold":
                signs.append((bool(p["left"]), gx0))

    def _flat(k):
        return np.asarray(ups[k]["left"] + ups[k]["right"], float)

    def _se(a):
        return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0

    res = {"seed": seed, "n_packs": n_packs, "standoff_m": STANDOFF,
           "site": list(rig.site), "site_legal": bool(legal),
           "horizon_s": HORIZON * SIM_S_PER_DECISION, "arms": {}}
    for name in pols:
        a = _flat(name)
        res["arms"][name] = {
            "mean_s": float(a.mean()), "se_s": _se(a),
            "left_s": float(np.mean(ups[name]["left"])),
            "right_s": float(np.mean(ups[name]["right"]))}
    base = _flat("out_gripboth")
    res["paired_vs_best_blind"] = {
        k: {"delta_s": float((_flat(k) - base).mean()),
            "se_s": _se(_flat(k) - base)}
        for k in ("out_keyed", "out_anti", "hold", "out_nogrip")}
    # Channel legibility: does sign(grav_body[0]) separate the lean sides?
    ok = sum(1 for lf, g in signs if g is not None and ((g < 0) == lf))
    res["channel"] = {"sign_agrees": ok, "n": len(signs),
                      "first_gx": [None if g is None else round(g, 4)
                                   for _lf, g in signs]}
    res["wall_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PACKS
    print(json.dumps(probe(n_packs=n), indent=1, default=float))
