"""PS.01's blocker, decided by bakeoff: what is the impact channel J_t?

WHY THIS EXISTS. `PURPOSE_AND_SCAFFOLDING.md` §2.2 defines the integrity cost of
an impact as `alpha * max(0, J_t - J_0)` with

    J_t = sum_substeps ||cfrc_ext[torso]|| * dt          (a 6-vector norm)

and PS.01 must measure `J_0` as "the 95th percentile of impact impulse under
normal walking contact". On 2026-08-09 an iteration tried, and stood down
honestly rather than force a threshold: with no locomotion controller anywhere
in this repo, "normal walking contact" and "a fall" are not distinguishable
regimes under that formula — platform-fall J (15-27) landed INSIDE the range of
ordinary lying-on-the-ground J (7-49). The handoff (LOOP_JOURNAL, 22:35) named
two candidate repairs and one escalation, and said explicitly: *worth a bakeoff
rather than an argument*.

This is that bakeoff. It does NOT tune PS.01's threshold — it decides which
OBSERVABLE the threshold will be set on, before any threshold exists. Choosing
the number would be tuning on the metric (LC.02's anti-gaming rule); choosing
the channel is what has to happen first, and it is decided by arithmetic here.

THE METRIC — `fall_vs_ground_auc`. Two labelled regimes, both from the real
playground, both under the same random policy:

  FALL   : the humanoid released at platform height (`ladder_height + SPAWN_Z`)
           beside the platform, free-falling ~1.8 m further than usual before
           it hits the floor. This is §2.2's "a fall from the ladder platform".
  GROUND : the humanoid at its ordinary spawn, which collapses to the floor
           within ~1 s (PG.8: "he falls over"). This is the ordinary contact a
           fall must be told apart FROM, and it is the honest hard case — it is
           the very signal that swamped the §2.2 formula.

Each run scores `max over decisions` of the arm's per-decision channel, and the
metric is the AUC of FALL scores over GROUND scores (the probability that a
random fall outranks a random collapse; ties count 0.5). 1.0 = perfectly
separable, 0.5 = the channel cannot tell them apart. Higher is better.

NULL: the same scores with their labels shuffled — AUC 0.5 by construction, and
measured rather than assumed, so the gate is against a number this fixture
actually produced.

CONTROLS, both of which MUST fail the gate:
  constant : every run scores 1.0. A channel that reads nothing must not
             separate; if it does, the AUC estimator is broken.
  noise    : an rng draw per run, ignoring the physics entirely. This is the
             one that matters — with 10 runs a side, a lucky permutation can
             reach a high AUC, so `noise` measures how much AUC pure chance
             buys at THIS sample size.

WHAT IT CANNOT DO. It cannot say the winning channel is the right physics; it
says which of the four candidate channels carries the fall/no-fall bit at all.
An arm that wins here still has to survive PS.01's own pre-registered gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bakeoff import Arm, run_bakeoff                    # noqa: E402
from experiments.protocol import Budget, Spec                       # noqa: E402

SIM_S_PER_DECISION = 0.2        # w0.py's accounting unit
N_DECISIONS = 60                # 12 sim-s: a 3.2 m fall lands in <1 s, the rest
                                # is the settle that GROUND is made of
N_RUNS = 10                     # per regime, per seed
CTRL_SCALE = 0.4

IMPACT_WINDOW_S = 0.30          # round 2: how long after contact onset an
                                # "impact" lasts. Pre-registered, and NOT swept
                                # — sweeping it would be tuning on the metric.

# ROUND 1's four candidate channels, scored as `max over the run's decisions`.
# Each maps one substep of physics to one scalar; the per-decision value is the
# reducer applied over that decision's substeps.
ROUND1_CHANNELS = ("integral6", "peak6", "peak_force", "peak_dvel")

# ROUND 2 adds six, all anchored to the EVENT rather than the episode. Round 1
# measured that `max over 60 decisions` reports EXPOSURE, not impact: GROUND is
# in contact for nearly all 12 sim-seconds and FALL lands once, so GROUND
# eventually throws the bigger spike and two channels scored BELOW chance.
# These score the landing itself — the substeps from the first humanoid/world
# contact through IMPACT_WINDOW_S after it — or express a rate instead of an
# extremum. Round 1's four stay in, unchanged and uncomputed differently; a
# second round that drops its embarrassing arms is the forbidden move.
#
# Three of them keep §2.2's TORSO sensor and three replace it with the whole
# humanoid subtree, because a diagnostic run while writing round 2 found the
# two disagree completely: a 3.2 m drop lands on the feet, so `cfrc_ext[torso]`
# is *identically zero* for the first 0.30 s after contact onset and only
# spikes at 0.3-0.5 s when the torso itself reaches the floor. §2.2's sensor and
# a whole-body contact event are on different bodies. Both readings compete.
ROUND2_CHANNELS = ("evt_int6", "evt6", "evt_force", "evt_dvel",
                   "impact_speed", "mean_dvel",
                   "evt_bodyf", "evt_body6", "evt_bodyint")

CHANNELS = ROUND1_CHANNELS + ROUND2_CHANNELS


def _build(seed: int, fall: bool):
    """The playground with Jack either at platform height or at his usual spawn."""
    import mujoco
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from playground import (LADDER_X, LADDER_Y, SPAWN_OFFSET_Y, SPAWN_Z,
                            PlaygroundParams, make_playground, humanoid_index)

    p = PlaygroundParams(seed=seed)
    if fall:
        # Beside the platform (the platform sits at LADDER_Y + 0.45), at its
        # height, so he free-falls the full ladder height onto the floor.
        spawn = (LADDER_X, LADDER_Y + 0.45 + 0.9, p.ladder_height + SPAWN_Z)
    else:
        spawn = (LADDER_X, LADDER_Y + SPAWN_OFFSET_Y, SPAWN_Z)
    p = PlaygroundParams(seed=seed, humanoid_spawn=spawn)
    model, data, _ = make_playground(p, with_water=True, with_humanoid=True)
    idx = humanoid_index(model)
    torso = int(model.body("torso").id)
    hgeoms = {g for g in range(model.ngeom)
              if int(model.geom_bodyid[g]) in set(idx["bodies"])}
    return mujoco, model, data, idx, torso, hgeoms


def _world_contact(data, hgeoms) -> bool:
    """Is Jack touching something that is not Jack, right now?

    Exactly one geom of the pair inside the humanoid subtree: two humanoid geoms
    is him folding onto himself (which happens constantly under a random policy
    and is not a landing), zero is the apple rolling into a wall. Label-free by
    construction — the same predicate runs in both regimes and never sees which
    one it is in."""
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if (int(c.geom1) in hgeoms) != (int(c.geom2) in hgeoms):
            return True
    return False


def _run_scores(seed: int, fall: bool, run_i: int) -> dict:
    """One life; returns {channel: score}.

    Round-1 channels reduce with `max over the run's decisions`; round-2
    channels are anchored to the first humanoid/world contact. The PHYSICS here
    is byte-identical to round 1 — same builds, same rng draws, same mj_step
    sequence — so the round-1 arms must reproduce their round-1 numbers exactly.
    `main()` asserts that they do; if they ever drift, the extra bookkeeping has
    perturbed the simulation and no cross-round comparison is legitimate."""
    mujoco, model, data, idx, torso, hgeoms = _build(seed * 101 + run_i, fall)
    rng = np.random.RandomState(seed * 9973 + run_i * 17 + (1 if fall else 0))
    dt = float(model.opt.timestep)
    frame_skip = max(1, int(round(SIM_S_PER_DECISION / dt)))
    window = max(1, int(round(IMPACT_WINDOW_S / dt)))
    dadr = idx["dofadr"]
    nu = model.nu

    best = {c: 0.0 for c in CHANNELS}
    prev_v = np.array(data.qvel[dadr:dadr + 3], dtype=float)
    onset = None            # substep index of first humanoid/world contact
    substep = 0
    evt_i6 = evt6 = evt_f = evt_dv = 0.0
    evt_bf = evt_b6 = evt_bint = 0.0
    sum_dv = 0.0
    bodies = list(idx["bodies"])
    for _ in range(N_DECISIONS):
        acc_i6 = 0.0
        pk6 = pkf = pkdv = 0.0
        ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, nu)
        for _ in range(frame_skip):
            data.ctrl[:nu] = ctrl
            v_pre = prev_v
            mujoco.mj_step(model, data)
            # NOT optional: mj_step does not populate cfrc_ext (the PG.8 lesson).
            # This loop is PS.01's own stepping loop precisely because w0.step()
            # calls it once per decision for throughput — see DriveLayer's
            # docstring.
            mujoco.mj_rnePostConstraint(model, data)
            f6 = np.asarray(data.cfrc_ext[torso], dtype=float)
            n6 = float(np.linalg.norm(f6))
            nf = float(np.linalg.norm(f6[3:6]))     # cfrc_ext = [torque, force]
            v = np.array(data.qvel[dadr:dadr + 3], dtype=float)
            dv = float(np.linalg.norm(v - prev_v))
            prev_v = v
            acc_i6 += n6 * dt
            pk6 = max(pk6, n6)
            pkf = max(pkf, nf)
            pkdv = max(pkdv, dv)
            sum_dv += dv
            if onset is None and _world_contact(data, hgeoms):
                onset = substep
                # The speed he arrived at, read one substep BEFORE the collision
                # resolved it away. Bounded by the drop, which is the property
                # `peak_dvel` survived on in round 1.
                best["impact_speed"] = float(np.linalg.norm(v_pre))
            if onset is not None and substep - onset < window:
                evt_i6 += n6 * dt
                evt6 = max(evt6, n6)
                evt_f = max(evt_f, nf)
                evt_dv = max(evt_dv, dv)
                # The same three statistics with the sensor moved from the
                # torso to the whole subtree — what the CREATURE took, not what
                # one link took. Computed only inside the window, so it costs
                # nothing on the ~97% of substeps outside it.
                bf6 = np.asarray(data.cfrc_ext[bodies], dtype=float)
                b_lin = float(np.linalg.norm(bf6[:, 3:6], axis=1).sum())
                b_all = float(np.linalg.norm(bf6, axis=1).sum())
                evt_bf = max(evt_bf, b_lin)
                evt_b6 = max(evt_b6, b_all)
                evt_bint += b_all * dt
            substep += 1
        for c, val in (("integral6", acc_i6), ("peak6", pk6),
                       ("peak_force", pkf), ("peak_dvel", pkdv)):
            best[c] = max(best[c], val)
    best["evt_int6"] = evt_i6
    best["evt6"] = evt6
    best["evt_force"] = evt_f
    best["evt_dvel"] = evt_dv
    best["evt_bodyf"] = evt_bf
    best["evt_body6"] = evt_b6
    best["evt_bodyint"] = evt_bint
    best["mean_dvel"] = sum_dv / max(1, N_DECISIONS)
    best["_onset"] = float(-1 if onset is None else onset)
    return best


_CACHE: dict = {}


def _scores(seed: int) -> dict:
    """{channel: (fall_list, ground_list)} for one seed. Cached: every arm and
    every control reads the SAME physics, so the comparison is between channels
    and never between rollouts."""
    if seed in _CACHE:
        return _CACHE[seed]
    fall = [_run_scores(seed, True, i) for i in range(N_RUNS)]
    ground = [_run_scores(seed, False, i) for i in range(N_RUNS)]
    out = {c: ([r[c] for r in fall], [r[c] for r in ground]) for c in CHANNELS}
    out["_onset"] = ([r["_onset"] for r in fall], [r["_onset"] for r in ground])
    _CACHE[seed] = out
    return out


# The round-1 result, recorded 2026-08-10 (DECISIONS_RESOLVED.md, PS.01/J).
# Round 2 must reproduce it to the digit or the rollout has been perturbed and
# nothing may be compared across rounds.
ROUND1_REFERENCE = {"integral6": 0.520, "peak6": 0.340,
                    "peak_force": 0.337, "peak_dvel": 0.827}


def check_round1_reproduction(seeds) -> dict:
    """Re-derive round 1's four AUCs from the current rollout code.

    A round-2 file that quietly changed the physics would show its new arms
    winning against numbers that no longer exist. This is the cheapest possible
    guard against that: the old arms are still computed by the same loop, so
    they must still read what they read."""
    import statistics as st
    got = {}
    for c in ROUND1_REFERENCE:
        got[c] = st.mean(_arm(c)(s) for s in seeds)
    bad = {c: (got[c], ROUND1_REFERENCE[c]) for c in got
           if abs(got[c] - ROUND1_REFERENCE[c]) > 5e-4}
    if bad:
        raise AssertionError(
            "round-1 arms no longer reproduce their recorded AUCs — the "
            f"rollout has changed, so no cross-round comparison is valid: {bad}")
    return got


def auc(pos, neg) -> float:
    """P(random pos > random neg), ties at 0.5. No sklearn on this box."""
    n = 0.0
    for a in pos:
        for b in neg:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / (len(pos) * len(neg))


def _arm(channel: str):
    def run(seed: int) -> float:
        pos, neg = _scores(seed)[channel]
        return auc(pos, neg)
    return run


def _null(seed: int) -> float:
    """Labels shuffled, averaged over 200 permutations of the SAME numbers.
    Measured, not assumed to be 0.5."""
    rng = np.random.RandomState(4242 + seed)
    pos, neg = _scores(seed)["peak_force"]
    pool = np.array(pos + neg, dtype=float)
    vals = []
    for _ in range(200):
        p = rng.permutation(pool)
        vals.append(auc(list(p[:len(pos)]), list(p[len(pos):])))
    return float(np.mean(vals))


def _control_constant(seed: int) -> float:
    return auc([1.0] * N_RUNS, [1.0] * N_RUNS)


def _control_noise(seed: int) -> float:
    rng = np.random.RandomState(777 + seed)
    return auc(list(rng.rand(N_RUNS)), list(rng.rand(N_RUNS)))


SPEC = Spec(
    "PS.01/J", 2,
    "Which impact channel separates a fall from ordinary ground contact?",
    hypothesis="At least one candidate impact channel gives FALL runs a higher "
               "peak than GROUND runs with AUC >= 3 sigma above the "
               "label-shuffled null, over 3 seeds.",
    falsified_by="No channel clears the null by 3 sigma. Then no formulation "
                 "of J_t discriminates a fall on this body without a "
                 "locomotion controller, and §2.2's calibration must be "
                 "escalated to the owner rather than re-specified here.",
    null_baseline="The same per-run scores with FALL/GROUND labels shuffled, "
                  "200 permutations per seed: AUC 0.5 by construction.",
    metric="fall_vs_ground_auc", budget=Budget.CPU, seeds=3,
    depends_on=["PG.8"],
    control="constant (every run scores 1.0 — a channel that reads nothing "
            "must not separate) and noise (an rng draw per run — it measures "
            "how much AUC chance buys at 10 runs a side). Both must FAIL the "
            "3-sigma gate.",
    kills="Any impact channel that cannot tell a 3.2 m drop from a collapse. "
          "It cannot kill PS.01 — it decides which observable PS.01 measures "
          "J_0 on, and deliberately reads no threshold.",
    notes="Not registered in the ladder: it is a DECISION, not a capability "
          "claim, so it writes docs/DECISIONS_RESOLVED.md and never the "
          "ledger. cost = substeps of state a channel must carry (1 for a "
          "running sum or a running max; 2 for peak_dvel, which also holds the "
          "previous root velocity).")

ARMS = [
    Arm("integral6", _arm("integral6"), cost=1.0,
        description="§2.2 as written: sum ||cfrc_ext[torso]||*dt over the "
                    "decision. Mixes torque (N·m) and force (N) in one 6-norm."),
    Arm("peak6", _arm("peak6"), cost=1.0,
        description="Handoff option (a): max over substeps of the same 6-norm. "
                    "A sharp landing spike instead of accumulated contact load."),
    Arm("peak_force", _arm("peak_force"), cost=1.0,
        description="max over substeps of the LINEAR force only, cfrc_ext[3:6]. "
                    "Dimensionally coherent where the 6-norm is not."),
    Arm("peak_dvel", _arm("peak_dvel"), cost=2.0,
        description="max over substeps of the root linear velocity jump — the "
                    "deceleration the body actually experienced."),
]

CONTROLS = [
    Arm("constant", _control_constant, cost=0.0,
        description="every run scores 1.0"),
    Arm("noise", _control_noise, cost=0.0,
        description="rng per run; chance AUC at this sample size"),
]


def main():
    res = run_bakeoff(SPEC, ARMS, _null, seeds=[0, 1, 2],
                      learning_gate_sigma=3.0, margin_sigma=1.5,
                      higher_is_better=True, controls=CONTROLS, ledger=None)
    print(f"\n{res.verdict}  winner={res.winner}")
    print(res.reason)
    print(f"null {res.null_mean:.4f} +- {res.null_std:.4f}")
    for a in sorted(res.arms, key=lambda x: x.mean, reverse=True):
        print(f"  {a.name:20s} mean {a.mean:.3f}  sigma {a.sigma_over_null:6.2f}  "
              f"gate {'pass' if a.passed_gate else 'FAIL'}  scores "
              f"{[round(s, 3) for s in a.scores]}")
    return res


if __name__ == "__main__":
    main()
