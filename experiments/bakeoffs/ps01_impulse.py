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

# The four candidate channels. Each maps one substep of physics to one scalar;
# the per-decision value is the reducer applied over the decision's substeps.
CHANNELS = ("integral6", "peak6", "peak_force", "peak_dvel")


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
    return mujoco, model, data, idx, torso


def _run_scores(seed: int, fall: bool, run_i: int) -> dict:
    """One life; returns {channel: peak-over-decisions}."""
    mujoco, model, data, idx, torso = _build(seed * 101 + run_i, fall)
    rng = np.random.RandomState(seed * 9973 + run_i * 17 + (1 if fall else 0))
    dt = float(model.opt.timestep)
    frame_skip = max(1, int(round(SIM_S_PER_DECISION / dt)))
    dadr = idx["dofadr"]
    nu = model.nu

    best = {c: 0.0 for c in CHANNELS}
    prev_v = np.array(data.qvel[dadr:dadr + 3], dtype=float)
    for _ in range(N_DECISIONS):
        acc_i6 = 0.0
        pk6 = pkf = pkdv = 0.0
        ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, nu)
        for _ in range(frame_skip):
            data.ctrl[:nu] = ctrl
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
        for c, val in (("integral6", acc_i6), ("peak6", pk6),
                       ("peak_force", pkf), ("peak_dvel", pkdv)):
            best[c] = max(best[c], val)
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
    _CACHE[seed] = out
    return out


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
