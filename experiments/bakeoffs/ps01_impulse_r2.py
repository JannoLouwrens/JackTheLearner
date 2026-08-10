"""PS.01's impact channel, ROUND 2: anchor the measurement to the event.

ROUND 1 (`ps01_impulse.py`, 2026-08-10) returned **VOID** and the verdict was
correct. Four candidate channels, scored `max over the run's 60 decisions`, AUC
of a 3.2 m platform fall over an ordinary collapse to the floor, against a
label-shuffled null measured at 0.4966 +- 0.0122:

    peak_dvel     0.827   +5.99 sigma   the only finisher
    integral6     0.520   +0.44 sigma   §2.2 as written — at chance
    peak6         0.340   -1.96 sigma   BELOW chance
    peak_force    0.337   -2.62 sigma   BELOW chance

Two channels scoring *below* chance is not noise; it is a reducer bug. `max over
the episode` is an extreme-value statistic, and FALL spends most of its 12
simulated seconds in free flight while GROUND is in contact almost throughout
under a random policy that keeps driving the actuators. Over 60 decisions GROUND
simply gets more draws, so it out-peaks the single landing. The channel that
survived, `peak_dvel`, survived because a velocity jump is bounded by how fast
you were going: lying on the floor cannot manufacture one however long it lies.

ROUND 2 attacks exactly that confound, and takes the repair the round-1 handoff
named: **score the landing, not the life.** Nine new arms —

  evt_int6, evt6, evt_force, evt_dvel : round 1's four statistics restricted to
      the impact window, the substeps from the first humanoid/world contact
      through `IMPACT_WINDOW_S` (0.30 s) after it. Contact onset is detected by
      a label-free predicate (exactly one geom of a contact pair inside the
      humanoid subtree) that runs identically in both regimes. Both regimes have
      exactly one such window, so the number of draws is now matched — which is
      the whole disease.
  evt_bodyf, evt_body6, evt_bodyint : the same three force statistics with the
      sensor moved off §2.2's torso onto the whole 13-body subtree. A diagnostic
      written alongside these arms found the two disagree completely: a 3.2 m
      drop lands on the FEET, so `cfrc_ext[torso]` is identically zero for the
      first 0.30 s after contact onset and only spikes at 0.3-0.5 s when the
      torso reaches the floor. §2.2's sensor and a whole-body contact event are
      on different bodies. Both readings compete rather than one being fixed.
  impact_speed : the root's linear speed one substep BEFORE onset. The arrival
      velocity, which is what a drop height actually buys.
  mean_dvel : total velocity change divided by decisions — a RATE rather than an
      extremum, the other half of the round-1 rule.

THIRTEEN arms on three seeds is enough comparisons that a lucky one could clear
3 sigma, and the `noise` control is in precisely to price that: it measures what
chance buys at this sample size (round 1: 0.570). The margin rule then requires
the winner to beat the RUNNER-UP by 1.5 sigma, so a single lucky arm cannot win
by itself.

**NOTHING IS DROPPED.** All four round-1 arms compete again, including the two
that scored below chance. Dropping an embarrassing arm is what `bakeoff.py`
explicitly forbids, and the only thing separating a legitimate second round from
that is that both rounds are in `docs/DECISIONS_RESOLVED.md` with every arm in
them. They are also recomputed by the same rollout loop, and `main()` asserts
they reproduce their round-1 AUCs to within 5e-4 before any new arm is read: if
the physics moved, the cross-round comparison is void and the run says so.

GATE MODE. This spec declares `gate_mode="screen"` (see `Spec.gate_mode` and
`bakeoff.py` property 4). The arms are OBSERVABLES — thirteen deterministic functions
of the same shared rollouts — not learners, so an arm below the gate is
eliminated rather than invalidating. The gate itself is unchanged at 3 sigma,
two arms must still clear it, and the controls still invert the verdict. The
mode was added AFTER round 1 VOIDed, and the check that it is not
reverse-engineered is that it does not rescue round 1: round 1 had exactly one
finisher and stays VOID under both readings.

WHAT IT STILL CANNOT DO. It says which observable carries the fall/no-fall bit.
It does not set `J_0`, and deliberately reads no threshold — that is PS.01's
job, on the channel this decides.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bakeoff import Arm, run_bakeoff                    # noqa: E402
from experiments.protocol import Budget, Spec                       # noqa: E402
from experiments.bakeoffs.ps01_impulse import (                     # noqa: E402
    IMPACT_WINDOW_S, N_RUNS, _arm, _control_constant, _control_noise, _null,
    _scores, check_round1_reproduction)

SEEDS = [0, 1, 2]

SPEC = Spec(
    "PS.01/J2", 2,
    "Which impact channel separates a fall from ordinary ground contact? "
    "(round 2: event-anchored)",
    hypothesis="Anchoring the impact statistic to the landing rather than to "
               "the episode rescues at least one of round 1's below-chance "
               "channels, and at least two of the thirteen candidates separate "
               "FALL "
               "from GROUND by >= 3 sigma over the label-shuffled null.",
    falsified_by="Fewer than two channels clear the null by 3 sigma. Then the "
                 "event anchor is not the missing ingredient either, and "
                 "§2.2's formulation goes to the owner with round 1's and "
                 "round 2's numbers attached rather than being re-specified "
                 "here a third time.",
    null_baseline="The same per-run scores with FALL/GROUND labels shuffled, "
                  "200 permutations per seed. Round 1 measured 0.4966 +- "
                  "0.0122; it is re-measured here, not carried over.",
    metric="fall_vs_ground_auc", budget=Budget.CPU, seeds=3,
    depends_on=["PG.8"],
    control="constant (every run scores 1.0 — a channel that reads nothing "
            "must not separate) and noise (an rng draw per run — it measures "
            "how much AUC chance buys at 10 runs a side; round 1 measured "
            "0.570). Both must FAIL the 3-sigma gate.",
    kills="Any impact channel that cannot tell a 3.2 m drop from a collapse, "
          "including the four carried from round 1. It cannot kill PS.01 — it "
          "decides which observable PS.01 measures J_0 on, and reads no "
          "threshold.",
    gate_mode="screen",
    screen_rationale=(
        "The arms are observables, not learners: each is a deterministic "
        "reduction of the SAME cached rollouts (`_scores` is memoised per seed, "
        "so every arm and every control reads identical physics). There is no "
        "training that could have failed, so a low score cannot be a broken run "
        "— it is the arm's own property, which is precisely the finding this "
        "bakeoff exists to produce. The T2.02 ambiguity the validity gate "
        "protects against (broken run or worse architecture?) does not exist "
        "here."),
    notes=f"Round 2 of PS.01/J. Impact window {IMPACT_WINDOW_S}s, "
          f"pre-registered and NOT swept — sweeping it would be tuning on the "
          f"metric. cost = substeps of state a channel must carry to be "
          f"computed online (an onset flag and a window counter cost 1 each; "
          f"holding the previous root velocity costs 1).")

ARMS = [
    # --- round 1, carried unchanged. Two of these scored below chance. ---
    Arm("integral6", _arm("integral6"), cost=1.0,
        description="R1. §2.2 as written: max over decisions of "
                    "sum ||cfrc_ext[torso]||*dt."),
    Arm("peak6", _arm("peak6"), cost=1.0,
        description="R1. max over the episode of the 6-norm."),
    Arm("peak_force", _arm("peak_force"), cost=1.0,
        description="R1. max over the episode of the linear force only."),
    Arm("peak_dvel", _arm("peak_dvel"), cost=2.0,
        description="R1. max over the episode of the root velocity jump. The "
                    "only round-1 finisher (0.827, +5.99 sigma)."),
    # --- round 2: the same statistics, anchored to the landing ---
    Arm("evt_int6", _arm("evt_int6"), cost=3.0,
        description="§2.2's integral over the impact window only. The direct "
                    "test of whether §2.2's formula was wrong or merely "
                    "wrongly reduced."),
    Arm("evt6", _arm("evt6"), cost=3.0,
        description="max 6-norm within the impact window."),
    Arm("evt_force", _arm("evt_force"), cost=3.0,
        description="max linear force within the impact window. The "
                    "dimensionally coherent channel, given its one landing."),
    Arm("evt_dvel", _arm("evt_dvel"), cost=4.0,
        description="max root velocity jump within the impact window."),
    # --- round 2: the sensor moved off the torso onto the whole subtree ---
    Arm("evt_bodyf", _arm("evt_bodyf"), cost=4.0,
        description="max over the impact window of the summed LINEAR "
                    "cfrc_ext over all 13 humanoid bodies."),
    Arm("evt_body6", _arm("evt_body6"), cost=4.0,
        description="same, 6-norm."),
    Arm("evt_bodyint", _arm("evt_bodyint"), cost=4.0,
        description="§2.2's integral with the whole-body sensor, over the "
                    "impact window."),
    Arm("impact_speed", _arm("impact_speed"), cost=3.0,
        description="root linear speed one substep before contact onset — the "
                    "arrival velocity, read before the collision removes it."),
    Arm("mean_dvel", _arm("mean_dvel"), cost=2.0,
        description="total root velocity change / decisions. A rate, not an "
                    "extremum: the other repair the round-1 rule names."),
]

CONTROLS = [
    Arm("constant", _control_constant, cost=0.0,
        description="every run scores 1.0"),
    Arm("noise", _control_noise, cost=0.0,
        description="rng per run; chance AUC at this sample size"),
]


def main():
    # Guard first: if the rollout moved, every comparison below is meaningless.
    got = check_round1_reproduction(SEEDS)
    print("round-1 arms reproduce: "
          + "  ".join(f"{k} {v:.3f}" for k, v in got.items()))

    # Diagnostic, not a gate: an event-anchored arm is only meaningful if the
    # event was found. A run with no world contact scores 0 on all six.
    missed = 0
    for s in SEEDS:
        f_on, g_on = _scores(s)["_onset"]
        missed += sum(1 for o in f_on + g_on if o < 0)
    print(f"contact onset found in {len(SEEDS) * 2 * N_RUNS - missed}/"
          f"{len(SEEDS) * 2 * N_RUNS} runs")

    res = run_bakeoff(SPEC, ARMS, _null, seeds=SEEDS,
                      learning_gate_sigma=3.0, margin_sigma=1.5,
                      higher_is_better=True, controls=CONTROLS, ledger=None)
    print(f"\n{res.verdict}  winner={res.winner}  gate_mode={res.gate_mode}")
    print(res.reason)
    print(f"null {res.null_mean:.4f} +- {res.null_std:.4f}")
    for a in sorted(res.arms, key=lambda x: x.mean, reverse=True):
        print(f"  {a.name:20s} mean {a.mean:.3f}  sigma {a.sigma_over_null:6.2f}  "
              f"gate {'pass' if a.passed_gate else 'FAIL'}  scores "
              f"{[round(s, 3) for s in a.scores]}")
    return res


if __name__ == "__main__":
    main()
