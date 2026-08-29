"""T2.11 — skills are distinguishable, measured by someone who did not train them.

*** PARKED 2026-08-29. DO NOT DISPATCH, DO NOT RE-PILOT, DO NOT WRITE A THIRD
*** RIG. Two pre-registered mechanism repairs against one outcome; the
*** pre-registered both-fail branch fired. The label-permuted control passed
*** BOTH pilots — on v2's seed 90 it beat the claim arm, with every rig gate
*** green. `_GATES_FROZEN` stays False and `run()` refuses. The finding, the
*** mechanism and the routed redesign question are in PILOT RECORD v2 below;
*** read it before touching anything here.

`UnifiedBrain.SkillDiscovery` (DIAYN, arXiv:1802.06070) has been in this
repository since before the ladder existed and has never received a gradient in
a registered experiment. Its docstring says "the robot learns walking, jumping,
turning just from wanting to be distinguishable!" — the exact shape of the
README-says-Working disease SYSTEM.md was written to kill. This spec is the test
that could have failed, and its `kills` field is that class, by name.

THE VACUITY THIS SPEC EXISTS TO AVOID, stated before the design because the
design is downstream of it. DIAYN trains a discriminator q(z|s) to recover the
skill from the state and pays the policy log q(z|s) - log p(z) for making that
easy. So "the discriminator recovers the skill" is TRAINING FIT and proves
nothing: it is the objective, read back. Three separate ways this test could
score high while measuring nothing, and what each forces in the design:

  (1) SCORING WITH THE DISCRIMINATOR. Circular. -> The verdict is read by an
      INDEPENDENT classifier: different architecture, different init, different
      optimiser, different input representation (see FEATURES), trained on
      rollouts the DIAYN policy never trained on and evaluated on a further
      disjoint set.
  (2) A DETERMINISTIC POLICY. If eval rollouts are greedy, MuJoCo is
      deterministic and every rollout of skill z is byte-identical, so the
      classifier's "held-out" set is its training set and 100% is arithmetic.
      -> Eval rollouts carry EVAL_EPS exploration on private RNG streams, and
      `no_leak` gates the actual feature-hash overlap at zero rather than
      trusting that argument.
  (3) ANY PER-SKILL LOTTERY. n private policies with n private random seeds
      produce n systematically different walks whether or not a mutual
      information was ever maximised. This is the one that matters, and it is
      why the claim is NOT "above chance": chance is the registered null and it
      is kept, but the binding gate is a MARGIN over two twins that have the
      identical machinery and no MI (see THE ARMS).
      **THE FIRST PILOT PROVED (3) IS NOT HYPOTHETICAL: it happened, at full
      scale, and the margin gate caught it.** See PILOT RECORD v1 and THE
      REPAIR below — the rig now carries ONE SHARED policy for all skills, so
      there are no private parameters left for a lottery to ride on.

THE ARMS — matched world, matched learner, matched budget, matched number of
discriminator steps AND matched number of policy steps, over the identical
shared parameters; they differ in one thing, the reward the policy is paid:
  diayn     the claim. SkillDiscovery's own `get_discriminator_loss` on
            (state, true skill), its own `compute_diayn_reward` paid to the
            policy. The mechanism under test, imported from UnifiedBrain.py —
            not reimplemented, so a FAIL falls on the shipped component.
  shuffled  THE CONTROL, and it must fail. Identical in every respect except
            that the skill labels are permuted inside each discriminator batch.
            Same nets, same optimiser, same number of steps, same reward
            pathway, same reward magnitudes — the MI is destroyed and nothing
            else is. If skills still come out distinguishable here, this rig
            measures the lottery of (2)/(3) and not DIAYN.
  zero      r = 0.0 to the identical learner: the FLOOR instrument. With a
            zero-initialised head every TD error is identically zero, so the
            shared net never moves and its action choice stays a uniform
            tie-break — a random walk, per skill, from an identical start. It
            bounds what "distinguishable" costs when nothing is learned at all.
  oracle    THE LEARNER'S POSITIVE CONTROL, and it must PASS. r = 1 when the
            rover is inside skill z's angular sector of the arena and 0
            otherwise — a reward that certainly depends on z, delivered by the
            identical pathway, to the identical shared net, for the identical
            number of steps. It is a rig gate, never a claim gate: it asks
            "can THIS policy class, on THIS budget, make skills legible at
            all?" Without it, an under-trained shared network is
            indistinguishable from a refuted DIAYN — and this spec's `kills`
            field deletes a shipped component, so that confusion is not
            affordable. `oracle` failing means VOID (the learner is dead), not
            FAIL. Nothing in the claim gates mentions it.

SAID OUT LOUD, because a control that cannot fail is decoration: `zero`'s
failure is close to ENTAILED (exchangeable random walks cannot carry a label
across disjoint rollouts, and here it is entailed twice over — the net is
provably stationary). `shuffled` is NOT entailed — its policy is trained on
shared parameters, it chases a live reward signal of the same magnitude as
the claim arm's, and it could still differentiate if the noise-trained
discriminator leaves systematic (s, z) structure to exploit. That is where
the discriminating work is done, and `_check` reads `shuffled` as the declared
control for exactly that reason. `zero` and `oracle` bracket it: they are
reported and gated only as a floor and a ceiling on the LEARNER, never on the
claim.

WHY Q_INIT = 0 HERE, against PG.4's optimistic 3.0. PG.4/T2.09 want a frontier
sweep; this spec wants specialisation, and an optimistic init pays every policy
to leave wherever it is — it fights the thing being measured. Both twins take
the identical init, so the comparison is matched; the constant is not inherited
from PG.4 and is not claimed to be.

FEATURES — what the independent classifier reads, and why it is not the retina.
The discriminator reads the 68-d retina (position, velocity, 32 rays x
[distance, texture]) as its `state_latent`. The classifier reads a purely
KINEMATIC trajectory summary: the normalised visitation histogram over PG.4's
121 floor cells, plus the (x, y) at 8 evenly-spaced checkpoints. Two reasons.
First, independence: a classifier reading the same features as the objective
inherits its blind spots. Second, it is what DIAYN's own docstring claims —
"skills are distinguishable by the STATES THEY VISIT" — so the verdict is
scored against the component's advertised claim rather than a friendlier one.

THE WORLD is PG.4's, imported not copied: same MJCF (n_objects=0), same
non-colliding velocity rover, same 32-ray retina, same 1 m cell grid. The panel
is STATIC (`noisy=False`) throughout — an irreducibly stochastic percept is
T2.09's subject, and it is settled there; letting it in here would confound a
skill's identity with what it happened to see on the TV.

PRE-REGISTERED GATES — PROVISIONAL. `_GATES_FROZEN = False` and `run()` REFUSES
until a pilot on seeds disjoint from the registered ones freezes every bar
below. The numbers written here are placeholders anchored on mechanism, not
measurements; the pilot may move them once, in the open, before the first
registered seed is drawn.

  RIG (violated -> VOID, not FAIL: the apparatus did not ask the question)
    instrument_alive   the label-SHUFFLED classifier's TRAIN accuracy >=
                       SHUFFLE_FIT_FLOOR. LESSONS, "An at-chance control must
                       carry proof its instrument was alive": a classifier with
                       no capacity reads at chance on everything, which would
                       make every at-chance control below pass for the worst
                       possible reason. This is that proof, and it is a rig
                       gate rather than a claim gate because a dead classifier
                       is a broken instrument, not a refuted hypothesis.
    instrument_honest  the label-shuffled classifier's HELD-OUT accuracy <=
                       CHANCE + SHUFFLE_BAND. If a classifier trained on
                       permuted labels still scores on true held-out labels,
                       something outside the label is carrying the signal.
    no_leak            zero feature-hash collisions between the classifier's
                       train and held-out rollout sets, on every arm. See
                       vacuity (2); T3.01's structural leak gate, same idiom.
    body_moved         every arm's mean rollout coverage > 0 and `zero`'s mean
                       coverage >= FLOOR_COVERAGE. The rover moves in this
                       world under the null policy — without it, "skills are
                       indistinguishable" could mean "nothing went anywhere".
    learner_alive      `oracle` held-out accuracy >= ORACLE_MIN. The shared
                       policy CAN make skills legible when the reward really
                       depends on the skill. See THE ARMS: this is the gate
                       that keeps an under-trained network from reading as a
                       refuted DIAYN and firing `kills`.
    floor_is_uniform   `zero`'s max |Q| over the materialised table == 0.0
                       EXACTLY. Not a bar — arithmetic. r = 0 and a zero head
                       give a zero TD error and a zero gradient, so the net is
                       stationary and `zero` really is a uniform random walk.
                       If this is ever nonzero, something moved the policy
                       without paying it, and the floor is not a floor.

  CLAIM (WORST registered seed, never the mean; all four must hold)
    above_chance   diayn held-out accuracy >= CHANCE + ABOVE_CHANCE_MIN. The
                   registered null (`Chance = 1/n_skills`), kept and not
                   weakened — but it is the weakest of the four.
    beats_shuffled diayn - shuffled held-out accuracy >= MARGIN_MIN. THE GATE
                   THAT DECIDES THIS SPEC. It is the one gate applied to the
                   run only: a control cannot be a margin ahead of itself, and
                   wiring it to a constant for the control would make the
                   control unable to fail (`_fold_control` says this in code).
    beats_zero     diayn - zero held-out accuracy >= MARGIN_MIN.
    per_class      min per-skill recall >= PER_CLASS_MIN. Without it, two
                   legible skills out of eight and six indistinguishable ones
                   clear an aggregate bar — "the MI objective collapsed" (the
                   registered `falsified_by`) is exactly what partial collapse
                   looks like, and an aggregate cannot see it.

  CONTROL (must fail the CLAIM gates): `shuffled`, scored on the identical
  claim gates in the same run.

PILOT RECORD v1 — THE RIG BELOW IS NOT THE RIG THAT PRODUCED THESE NUMBERS.
It is kept in full because it is the evidence for THE REPAIR that follows, and
because a diagnosis with its data deleted is an argument.
2026-08-29, this box, CPU, full registered scale, seeds 7 and 90
(artifacts /data/t2_11_pilot_seed{7,90}.json, 355.1 s and 355.2 s wall, two
concurrent on 4 shared ARM cores). THE CONTROL PASSED. Gates stay PROVISIONAL
and this spec MUST NOT BE DISPATCHED.

  arm       held-out acc s7 / s90   per-class min   coverage   disc loss first->last
  diayn     0.9688 / 0.9844         0.875 / 0.9375  0.064/0.054  2.120 -> 0.562
  shuffled  0.9766 / 0.9766         0.9375 / 0.875  0.080/0.056  2.040 -> 2.056
  zero      0.1484 / 0.1328         0.0625 / 0.0    0.133/0.131  2.104 -> 1.900
  margin_vs_shuffled: -0.0078 (s7), +0.0078 (s90).  chance = 0.125.
  instruments, both seeds: shuffled-CLASSIFIER train 0.80-1.00 / held-out
  0.109-0.180 (alive and honest), hash_overlap 0 on every arm.

READ IT IN THE RIGHT ORDER, because the tempting reading is the wrong one.
  1. Every INSTRUMENT worked. The classifier fits 8 labels (train 1.0) and
     reads chance when its labels are permuted (0.11-0.18 against 0.125). The
     splits do not leak (overlap 0). The floor is real: `zero`, a per-skill
     uniform random walk, is AT CHANCE — so "distinguishable" is not free for
     any policy, and the feature is not carrying an artefact.
  2. THE MECHANISM ALSO WORKED. DIAYN's discriminator loss fell 2.12 -> 0.56
     while `shuffled`'s sat at ln(8) = 2.079 exactly as a permuted-label
     discriminator must. The registered `falsified_by` — "the MI objective
     collapsed" — DID NOT HAPPEN. Whatever this pilot shows, it is not that.
  3. And the control still scored 0.977. A rig in which the label-permuted
     twin is as distinguishable as the trained one is a rig that measures
     nothing (SYSTEM.md law 2), so the correct verdict is about the
     APPARATUS, not about SkillDiscovery.

THE DIAGNOSIS, and it is vacuity (3) arriving at full scale in the one place
the design left the door open. This rig gives each skill its OWN tabular Q
table, so the arms are eight independent policies sharing a discriminator. Any
reward that is non-zero and not identical across tables — including pure label
noise, whose mean |r| here is 0.395 against DIAYN's 1.647 — drives each table
into its own idiosyncratic attractor, and eight idiosyncratic attractors are
trivially separable by where they sit (centroid separation 3.91 m for
`shuffled` against DIAYN's 3.43 m). `zero` escapes only because r = 0 with
Q_INIT = 0 leaves every table exactly equal, i.e. no policy at all. So the
measured quantity is "did each private table receive ANY signal", not "did
maximising I(S;Z) make the skills distinguishable".

THE REPAIR — WRITTEN INTO THE CODE 2026-08-29, AND PRE-REGISTERED HERE BEFORE
THE SECOND PILOT DREW A NUMBER (T3.06's protocol, one level up). The repair is
determined by the diagnosis, so it is not a choice between arms and does not
go to a bakeoff. Four decisions, each with the reason that forced it:

  1. THE POLICY IS ONE SHARED FUNCTION CONDITIONED ON THE SKILL, not n private
     tables — a single network taking (one-hot cell, `SkillDiscovery.
     get_skill_embedding(z)`) and emitting action values (`_make_policy`).
     Under permuted labels a shared policy receives a reward that is
     uninformative about z, so it cannot systematically differentiate and
     should collapse toward the `zero` floor; under true labels it can. That
     is what turns `beats_shuffled` from an identity into a measurement.
  2. THE SKILL EMBEDDING IS FROZEN AT ITS INIT. `nn.Embedding` is n private
     parameter vectors — trainable, it would reinstate vacuity (3) one level
     down, where it is harder to see. Frozen, it is the skill's identity code
     (which any conditioned policy must have), not its learned specialisation.
  3. THE HEAD IS INITIALISED TO EXACTLY ZERO, which is how `Q_INIT = 0.0`
     survives the move off tables and is what keeps `zero` a genuine floor: a
     randomly-initialised head would hand each skill a distinct deterministic
     policy for free, at which point even the r = 0 arm would be
     distinguishable and the floor instrument would be destroyed. Gated as
     `floor_is_uniform`.
  4. A FOURTH ARM, `oracle`, IS ADDED AS THE LEARNER'S POSITIVE CONTROL, and
     it is a RIG gate (`learner_alive`). The first rig could not fail for lack
     of learning; a shared network can, and an under-trained one reads exactly
     like a refuted DIAYN. Since a FAIL here fires `kills: SkillDiscovery`,
     the rig must carry proof its learner was alive before any FAIL is
     honest. Same lesson as "an at-chance control must carry proof its
     instrument was alive", moved from the instrument to the learner.

  NO BAR MOVES. `ABOVE_CHANCE_MIN`, `MARGIN_MIN`, `PER_CLASS_MIN`,
  `SHUFFLE_FIT_FLOOR`, `SHUFFLE_BAND` and `FLOOR_COVERAGE` stand exactly as
  they were written before the first pilot. `ORACLE_MIN` is new because the
  arm is new, and it is PROVISIONAL like the rest. What changed is the arm,
  and the reason is the CONTROL's score, never the claim arm's.

  IF THE REDESIGNED RIG ALSO SHOWS THE CONTROL PASSING, that is two mechanism
  repairs against one outcome and SM.02's decision tree applies: park the
  spec, record the finding, do not write a third rig.

PILOT RECORD v2 — THAT BRANCH FIRED. T2.11 IS PARKED (2026-08-29 ~23:2x UTC).
2026-08-29, this box, CPU, full registered scale, the SAME seeds 7 and 90 —
deliberately the same two, so no seed-shopping was possible across the rig
change (artifacts /data/t2_11_pilot2_seed{7,90}.json, 527.6 s and 531.1 s wall,
two concurrent on 4 shared ARM cores; `_check` replayed offline against the
recorded rows).

  arm       held-out acc s7 / s90   per-class min    q_absmax      mean|r|
  diayn     0.9141 / 0.7812         0.625  / 0.4375  78.7 / 39.9   1.40 / 1.50
  shuffled  0.9141 / 0.8984         0.625  / 0.4375  24.8 / 21.6   0.35 / 0.29
  zero      0.1484 / 0.1328         0.0625 / 0.0      0.0 /  0.0   0.00 / 0.00
  oracle    0.9766 / 1.0000         0.875  / 1.0     32.7 / 26.9   0.47 / 0.49
  margin_vs_shuffled: 0.0000 (s7), -0.1172 (s90).  chance = 0.125.

THE REPAIR WORKED AND THE OUTCOME DID NOT MOVE — that conjunction is the
finding, and both halves have to be read.

  1. THE REPAIR DID WHAT IT WAS DESIGNED TO DO. Every private parameter is
     gone, and the two new gates prove the new rig can fail in both
     directions where the old one could not. `learner_alive`: `oracle` reads
     0.9766 / 1.0000 — the shared conditioned policy CAN make skills legible
     on this budget, so a low claim score would have meant something.
     `floor_is_uniform`: `zero` reads `q_absmax == 0.0` EXACTLY on both seeds
     — arithmetic, not a bar — so the floor is a genuine uniform random walk
     and it lands at chance (0.148 / 0.133). The instrument is bracketed
     top and bottom, which is exactly what v1 could not claim.
  2. AND THE CONTROL STILL PASSED. `_claim_holds(control)` is True on both
     seeds, and on seed 90 the label-permuted twin BEATS the claim arm
     (0.8984 vs 0.7812). Seed 90 is the one to read: EVERY rig gate is green
     there (shuffle_clf_fit 0.7109, held-out 0.125, overlap 0, oracle 1.0,
     zero_q_absmax 0.0), so there is no apparatus escape hatch. Offline
     verdicts: seed 7 VOID (its shuffle_clf_fit 0.5625 misses the 0.60 rig
     floor), seed 90 False, registered worst-seed fold VOID.

THE MECHANISM, and it is a NEW vacuity, not one of the three listed at the top
of this file. `shuffled`'s discriminator is at chance and provably learns
nothing — loss 2.040 -> 2.058 (s7), 2.119 -> 2.068 (s90) against ln(8) =
2.0794. But `compute_diayn_reward` reads log q(z|s) off that network, and a
network carrying ZERO information about z still emits outputs that vary with
(s, z). So the permuted arm is paid a fixed, state-dependent, skill-dependent
RANDOM REWARD FIELD — mean |r| 0.29-0.35 against DIAYN's 1.40-1.50 — and a
shared skill-conditioned policy chasing a random field separates its skills
just as well as one chasing mutual information (centroid sep 4.18-5.42 m
against DIAYN's 5.43-6.69 m; q_absmax 21.6-24.8, decisively off the 0.0 floor).

WHICH MEANS THE METRIC, NOT THE RIG, IS WHAT FAILS HERE. Held-out skill
classification accuracy does not discriminate DIAYN from a chance-level
discriminator, because it measures the POLICY's response to any structured
reward rather than the OBJECTIVE's information content. Both mechanism
repairs were real and correctly diagnosed; neither could move an outcome that
the choice of measurement had already fixed. Generalised into
docs/LESSONS.md as "a control at chance on its own objective can still be at
ceiling on the downstream metric".

PARKED, per the pre-registration above and SM.02's precedent: gates stay
PROVISIONAL, `_GATES_FROZEN` stays False, `run()` keeps refusing, NO third rig
is written and NO dispatch is made. The open question — what measurement WOULD
separate "skills differ because I(S;Z) was maximised" from "skills differ
because they chased different noise" — is a spec-design question and is routed
to the Review as `t211-diayn-metric-cannot-separate-mi-from-noise`
(docs/DECISIONS_NEEDED.md, 2026-08-29). It is not an argument to be settled
here and it is not compute to be re-rolled. Do NOT relaunch these pilots as
cheap work: they are spent evidence, four arms x two seeds, ~18 core-minutes.

SAID PLAINLY SO NO LATER READER HAS TO INFER IT: NEITHER PILOT IS EVIDENCE
AGAINST SkillDiscovery, and this file may not be dispatched. A FAIL from
either rig would have fired `kills: SkillDiscovery` — deleting a shipped
component — off a run whose own label-permuted control scored 0.977 (v1) and
0.898 (v2). That is the outcome `_GATES_FROZEN` exists to prevent, and it is
why the flag is still False.

WHAT THIS SPEC DOES AND DOES NOT TEST, so its `kills` is scoped honestly.
Exercised, and therefore what a FAIL falls on: DIAYN's objective as this
repository ships it — `discriminator`, `get_discriminator_loss`,
`compute_diayn_reward`, and (since the repair) `get_skill_embedding` as the
policy's conditioning path. NOT exercised: a TRAINED skill embedding, and any
continuous-control policy. A FAIL says the shipped objective does not produce
distinguishable behaviour in this world with a skill-conditioned value policy
over PG.4's cell grid; it does not say no policy class could. That
distinction belongs in the record before the run, not after it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .pg_4_noisy_tv import (
    _ACTIONS, _Retina, _build, _cell, GAMMA, N_RAYS, Q_LR, SPEED, SUBSTEPS,
)

# The world, PG.4's rig, and the component under test all hash in: this spec's
# verdict is about SkillDiscovery in THIS arena, and a change to either must
# stale the certificate loudly.
IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py",
             "UnifiedBrain.py"]

_GATES_FROZEN = False           # PARKED, not merely un-piloted. Two pilots ran
                                # (v1 tabular, v2 shared-policy); the control
                                # passed both. Flipping this without a NEW
                                # MEASUREMENT — not a new rig — would dispatch a
                                # `kills: SkillDiscovery` off a run whose own
                                # label-permuted twin outscored it. See the
                                # PARKED banner and PILOT RECORD v2.

SEEDS = [0, 1, 2]               # registered; the registry declares 3 seeds
PILOT_SEEDS = (7, 90)           # disjoint from the registered set, and spent
                                # once used — the same two T2.09 piloted on.

# ── the rig ──────────────────────────────────────────────────────────────
N_SKILLS = 8                    # chance = 0.125. Eight is what an 11x11 arena
                                # can plausibly partition; SkillDiscovery's own
                                # default of 50 would put chance at 0.02 and
                                # make every bar below a coin-flip on 121 cells.
N_CELLS = 121                   # PG.4's floor grid
OBS_DIM = 4 + 2 * N_RAYS        # 68 — what the discriminator sees
EPISODE_LEN = 200               # decisions; 0.3 m/decision at full speed, so
                                # ~60 m of travel across an 11 m arena
TRAIN_EPISODES = 200            # 25 per skill, z sampled uniformly
EVAL_ROLLOUTS = 16              # per skill, per split (train / held-out)
EVAL_EPS = 0.10                 # PG.4's EPS_LO: eval is not deterministic, see
                                # vacuity (2)
EPS_HI, EPS_LO = 1.0, 0.10      # training exploration, decayed over the first
                                # third of the episodes
Q_INIT = 0.0                    # NOT PG.4's 3.0 — see WHY Q_INIT = 0 above.
                                # Realised as a ZERO-INITIALISED policy head,
                                # see `_make_policy`.
DISC_STEPS_PER_EP = 8           # discriminator gradient steps per episode
DISC_BATCH = 128
DISC_LR = 1e-3
CHECKPOINTS = 8                 # (x, y) samples in the trajectory feature
FEAT_DIM = N_CELLS + 2 * CHECKPOINTS

# The independent classifier
CLF_HIDDEN = 64
CLF_LR = 3e-3
CLF_EPOCHS = 300

# The SHARED skill-conditioned policy — the repair (see THE REPAIR above).
POLICY_EMB = 16                 # width `skill_embedding(z)`'s d_model=512 code
                                # is projected to, so the skill code does not
                                # outnumber the 121-d state 4:1 at the input
POLICY_HIDDEN = 128
POLICY_LR = 1e-3                # Adam on the shared net. NOT PG.4's Q_LR=0.2,
                                # which is a tabular step size and is a
                                # different quantity — GAMMA is shared, the
                                # learning rate cannot be.
POLICY_STEPS_PER_EP = 16        # gradient steps per episode, over replay
POLICY_BATCH = 256

_ARMS = ("diayn", "shuffled", "zero", "oracle")
_CLAIM_ARM = "diayn"
_CONTROL_ARM = "shuffled"


def _sector(cell: int) -> int:
    """PG.4's 121 floor cells -> one of `N_SKILLS` angular sectors about the
    arena centre. The `oracle` arm's fixed region assignment; nothing else in
    this spec reads it, and no gate on the claim arm mentions it."""
    cx, cy = cell % 11 - 5, cell // 11 - 5
    a = math.atan2(cy, cx)
    return min(N_SKILLS - 1, int((a + math.pi) / (2 * math.pi) * N_SKILLS))


_SECTOR = tuple(_sector(c) for c in range(N_CELLS))

# ── PROVISIONAL bars ─────────────────────────────────────────────────────
CHANCE = 1.0 / N_SKILLS         # 0.125 — the registered null baseline
ABOVE_CHANCE_MIN = 0.15         # placeholder: ~2x chance
MARGIN_MIN = 0.15               # placeholder: the deciding gate
PER_CLASS_MIN = 0.20            # placeholder: > chance for EVERY skill
SHUFFLE_FIT_FLOOR = 0.60        # placeholder: the classifier can fit 8 labels
SHUFFLE_BAND = 0.10             # placeholder: shuffled held-out <= 0.225
FLOOR_COVERAGE = 0.05           # placeholder: the random-walk twin moves
ORACLE_MIN = 0.60               # placeholder: the LEARNER's positive control


# ── the shared skill-conditioned policy ──────────────────────────────────
def _make_policy(disc, dev):
    """ONE action-value function for all `N_SKILLS`, conditioned on the skill.

    Every parameter is shared. The only per-skill quantity is the FROZEN code
    `SkillDiscovery.get_skill_embedding(z)` read once at construction — and it
    is frozen deliberately: `nn.Embedding` is literally n private parameter
    vectors, so a trainable embedding would reintroduce vacuity (3) one level
    down, where it would be far harder to see. Frozen, it is the skill's
    IDENTITY (an input code every conditioned policy must have), never its
    learned specialisation. Whatever differentiates the skills has to be
    learned by the shared body from a reward that depends on z.

    THE HEAD IS INITIALISED TO EXACTLY ZERO, and that is load-bearing rather
    than cosmetic. It is how `Q_INIT = 0.0` survives the move off tables:
      * at init Q(s, z) == 0 for every (s, z), so `_rollout`'s tie-break is
        uniform over all 9 actions and no skill starts with a private policy
        handed to it by a random init — which a nonzero head WOULD do, and
        would silently make `zero` distinguishable and destroy the floor;
      * for the `zero` arm r == 0, so every TD target is 0, the TD error is
        identically 0, and the gradient is identically 0. The net never moves.
        `zero` is therefore still an exact per-skill uniform random walk, and
        `q_absmax == 0.0` in its output is the arithmetic receipt (gated).
    """
    import torch
    from torch import nn

    with torch.no_grad():                      # the shipped conditioning path
        codes = disc.get_skill_embedding(
            torch.arange(N_SKILLS, device=dev)).detach().clone()

    class _SharedQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("codes", codes)
            self.proj = nn.Linear(codes.shape[1], POLICY_EMB)
            self.body = nn.Sequential(
                nn.Linear(N_CELLS + POLICY_EMB, POLICY_HIDDEN), nn.ReLU(),
                nn.Linear(POLICY_HIDDEN, len(_ACTIONS)))
            nn.init.zeros_(self.body[-1].weight)
            nn.init.zeros_(self.body[-1].bias)

        def forward(self, cells, z):
            oh = torch.zeros(cells.shape[0], N_CELLS, device=cells.device)
            oh[torch.arange(cells.shape[0], device=cells.device), cells] = 1.0
            return self.body(torch.cat([oh, self.proj(self.codes[z])], 1))

    return _SharedQ().to(dev)


# ── one arm, one seed ────────────────────────────────────────────────────
def _feature(cells: list, xy: list) -> "object":
    """Trajectory -> the classifier's input. Kinematic only; see FEATURES."""
    import numpy as np
    f = np.zeros(FEAT_DIM, dtype="float32")
    for c in cells:
        f[c] += 1.0
    f[:N_CELLS] /= max(1, len(cells))
    step = max(1, len(xy) // CHECKPOINTS)
    for i in range(CHECKPOINTS):
        px, py = xy[min(len(xy) - 1, i * step)]
        f[N_CELLS + 2 * i] = px / 6.0
        f[N_CELLS + 2 * i + 1] = py / 6.0
    return f


def _rollout(model, data, retina, q, z: int, rng, eps: float) -> dict:
    """One episode under skill z's table. Returns the transitions and the
    kinematic trace. Always starts from the same reset state, so no skill can
    be identified by where it began."""
    import mujoco
    import numpy as np

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ax, ay = retina.act_ids
    obs, _ = retina.observe(data)
    cells, xy, states, trans = [], [], [], []
    for _ in range(EPISODE_LEN):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        s = _cell(x, y)
        cells.append(s)
        xy.append((x, y))
        states.append(obs)
        if rng.uniform() < eps:
            a = int(rng.randint(len(_ACTIONS)))
        else:
            best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
            a = int(best[rng.randint(len(best))])
        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        obs2, _ = retina.observe(data)
        x2, y2 = float(data.qpos[-2]), float(data.qpos[-1])
        trans.append((s, a, _cell(x2, y2)))
        obs = obs2
    return {"cells": cells, "xy": xy, "states": states, "trans": trans,
            "z": z, "coverage": len(set(cells)) / N_CELLS}


def _train_arm(seed: int, arm: str) -> dict:
    """Train one arm to convergence-of-budget and return its frozen tables.

    Every arm runs the identical loop, the identical number of discriminator
    gradient steps and the identical number of POLICY gradient steps. `arm`
    selects ONLY (a) whether the discriminator sees true or permuted labels
    and (b) what reward reaches the shared policy.

    The policy is frozen WITHIN an episode and updated at the episode
    boundary from replay, so the acting rule for skill z can be materialised
    once as a (N_CELLS, |A|) table and handed to `_rollout` unchanged. That is
    an exact identity, not an approximation, and it keeps the world, the
    rollout code and the eval path byte-for-byte the ones the tabular rig
    used — the arm is the only thing that changed.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from UnifiedBrain import SkillDiscovery, UnifiedBrainConfig

    # `_ARMS.index`, never `hash(arm)`: Python randomises string hashing per
    # process, so a hash-seeded net would make this rig non-reproducible across
    # runs while every other seed in it stayed fixed — the quietest possible
    # determinism bug (T0.02's subject).
    torch.manual_seed(seed * 31 + _ARMS.index(arm))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UnifiedBrainConfig(latent_dim=OBS_DIM)
    disc = SkillDiscovery(cfg, num_skills=N_SKILLS).to(dev)
    opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR)

    model, data, panel_gid, rover_bid, act_ids = _build()
    retina = _Retina(model, panel_gid, rover_bid, False,
                     np.random.RandomState(seed * 7919 + 13))
    retina.act_ids = act_ids
    rng = np.random.RandomState(seed * 104729 + 7)

    policy = _make_policy(disc, dev)
    popt = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
    all_cells = torch.arange(N_CELLS, device=dev)

    def _table(z: int):
        """The shared policy's acting rule for skill z, as a lookup table."""
        with torch.no_grad():
            zt = torch.full((N_CELLS,), z, dtype=torch.long, device=dev)
            return policy(all_cells, zt).cpu().numpy().astype("float64")

    cap = TRAIN_EPISODES * EPISODE_LEN
    rp_i = np.zeros((cap, 4), dtype="int64")        # s, a, z, s2
    rp_r = np.zeros(cap, dtype="float32")
    n_rp = 0

    buf_s, buf_z = [], []
    loss_first = loss_last = None
    pol_first = pol_last = None
    r_abs_sum, r_n = 0.0, 0

    for ep in range(TRAIN_EPISODES):
        z = int(rng.randint(N_SKILLS))
        eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO)
                  * ep / max(1, TRAIN_EPISODES // 3))
        roll = _rollout(model, data, retina, _table(z), z, rng, eps)

        st = torch.as_tensor(np.stack(roll["states"]), device=dev)
        zt = torch.full((len(roll["states"]),), z, dtype=torch.long, device=dev)
        if arm == "zero":
            rewards = np.zeros(len(roll["trans"]), dtype="float32")
        elif arm == "oracle":
            # The LEARNER's positive control: a reward that certainly depends
            # on z, paid by a fixed partition the policy must discover the
            # consequences of. It reaches the shared net by the identical
            # pathway the DIAYN reward does.
            rewards = np.array([1.0 if _SECTOR[c] == z else 0.0
                                for c in roll["cells"]], dtype="float32")
        else:
            with torch.no_grad():
                r, _info = disc.compute_diayn_reward(st, zt)
            rewards = r.cpu().numpy().astype("float32")
        r_abs_sum += float(np.abs(rewards).sum()); r_n += len(rewards)
        # DIAYN pays for the state ARRIVED IN, so transition t is paid
        # r(states[t+1]); the final transition reuses the last state it has.
        arrive = np.concatenate([rewards[1:], rewards[-1:]])

        tr = np.asarray(roll["trans"], dtype="int64")
        k = len(tr)
        rp_i[n_rp:n_rp + k, 0] = tr[:, 0]
        rp_i[n_rp:n_rp + k, 1] = tr[:, 1]
        rp_i[n_rp:n_rp + k, 2] = z
        rp_i[n_rp:n_rp + k, 3] = tr[:, 2]
        rp_r[n_rp:n_rp + k] = arrive
        n_rp += k

        # Policy: the SAME number of gradient steps on every arm, over the
        # SAME shared parameters. Only `arrive` differs between arms.
        for _ in range(POLICY_STEPS_PER_EP):
            idx = rng.randint(n_rp, size=min(POLICY_BATCH, n_rp))
            bs = torch.as_tensor(rp_i[idx, 0], device=dev)
            ba = torch.as_tensor(rp_i[idx, 1], device=dev)
            bz = torch.as_tensor(rp_i[idx, 2], device=dev)
            b2 = torch.as_tensor(rp_i[idx, 3], device=dev)
            br = torch.as_tensor(rp_r[idx], device=dev)
            with torch.no_grad():
                tgt = br + GAMMA * policy(b2, bz).max(1).values
            pred = policy(bs, bz).gather(1, ba[:, None]).squeeze(1)
            ploss = torch.nn.functional.mse_loss(pred, tgt)
            popt.zero_grad(); ploss.backward(); popt.step()
            pol_last = float(ploss.detach())
            if pol_first is None:
                pol_first = pol_last

        buf_s.extend(roll["states"])
        buf_z.extend([z] * len(roll["states"]))

        # Discriminator: identical step count on every arm, including `zero`.
        # `shuffled` permutes the labels INSIDE the batch — same gradients
        # flowing, same magnitudes, no mutual information.
        for _ in range(DISC_STEPS_PER_EP):
            idx = rng.randint(len(buf_s), size=min(DISC_BATCH, len(buf_s)))
            bs = torch.as_tensor(np.stack([buf_s[i] for i in idx]), device=dev)
            bz = np.array([buf_z[i] for i in idx])
            if arm == "shuffled":
                bz = bz[rng.permutation(len(bz))]
            bzt = torch.as_tensor(bz, dtype=torch.long, device=dev)
            loss = disc.get_discriminator_loss(bs, bzt)
            opt.zero_grad(); loss.backward(); opt.step()
            if loss_first is None:
                loss_first = float(loss.detach())
            loss_last = float(loss.detach())

    q = np.stack([_table(z) for z in range(N_SKILLS)])
    return {"q": q, "model": model, "data": data, "retina": retina,
            "disc_loss_first": loss_first, "disc_loss_last": loss_last,
            "pol_loss_first": pol_first, "pol_loss_last": pol_last,
            "q_absmax": float(np.abs(q).max()),
            "mean_abs_reward": r_abs_sum / max(1, r_n)}


def _eval_arm(seed: int, arm: str, trained: dict) -> dict:
    """Freeze the tables, draw two DISJOINT rollout sets per skill, and hand
    them to a classifier that had no part in training anything."""
    import numpy as np
    import torch

    q, model, data = trained["q"], trained["model"], trained["data"]
    retina = trained["retina"]
    feats = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    covs = []
    cent = np.zeros((N_SKILLS, 2))
    for z in range(N_SKILLS):
        for split, base in (("train", 500_000), ("test", 900_000)):
            for k in range(EVAL_ROLLOUTS):
                rr = np.random.RandomState(base + seed * 10_000 + z * 100 + k)
                roll = _rollout(model, data, retina, q[z], z, rr, EVAL_EPS)
                feats[split].append(_feature(roll["cells"], roll["xy"]))
                labels[split].append(z)
                covs.append(roll["coverage"])
                if split == "test":
                    cent[z] += np.mean(roll["xy"], axis=0) / EVAL_ROLLOUTS

    xtr = np.stack(feats["train"]); ytr = np.array(labels["train"])
    xte = np.stack(feats["test"]); yte = np.array(labels["test"])

    # Structural leak gate: an identical feature in both splits would make
    # "held-out" a lie. Hashed on the exact bytes, T3.01's idiom.
    htr = {hash(v.tobytes()) for v in xtr}
    hte = {hash(v.tobytes()) for v in xte}
    overlap = len(htr & hte)

    def _fit(x, y, y_eval_true, seed_off: int) -> tuple:
        torch.manual_seed(seed * 977 + seed_off)
        clf = torch.nn.Sequential(
            torch.nn.Linear(FEAT_DIM, CLF_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CLF_HIDDEN, N_SKILLS))
        o = torch.optim.Adam(clf.parameters(), lr=CLF_LR)
        xt = torch.as_tensor(x); yt = torch.as_tensor(y, dtype=torch.long)
        for _ in range(CLF_EPOCHS):
            loss = torch.nn.functional.cross_entropy(clf(xt), yt)
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            tr_acc = float((clf(xt).argmax(1) == yt).float().mean())
            pred = clf(torch.as_tensor(xte)).argmax(1).numpy()
        acc = float((pred == y_eval_true).mean())
        per = [float((pred[y_eval_true == z] == z).mean())
               for z in range(N_SKILLS)]
        return tr_acc, acc, per

    tr_acc, acc, per = _fit(xtr, ytr, yte, 0)
    # The at-chance control for the INSTRUMENT: same classifier, permuted
    # training labels, scored against the true held-out labels.
    perm = np.random.RandomState(seed * 13 + 5).permutation(len(ytr))
    sh_tr_acc, sh_acc, _ = _fit(xtr, ytr[perm], yte, 1)

    d = np.array([[np.hypot(*(cent[i] - cent[j])) for j in range(N_SKILLS)]
                  for i in range(N_SKILLS)])
    off = d[~np.eye(N_SKILLS, dtype=bool)]

    return {
        "arm": arm,
        "heldout_acc": round(acc, 4),
        "train_acc": round(tr_acc, 4),
        "per_class_min": round(min(per), 4),
        "per_class": [round(p, 4) for p in per],
        "shuffled_clf_train_acc": round(sh_tr_acc, 4),
        "shuffled_clf_heldout_acc": round(sh_acc, 4),
        "hash_overlap": overlap,
        "mean_coverage": round(float(np.mean(covs)), 4),
        "centroid_sep_mean": round(float(off.mean()), 3),
        "disc_loss_first": round(trained["disc_loss_first"] or 0.0, 4),
        "disc_loss_last": round(trained["disc_loss_last"] or 0.0, 4),
        "pol_loss_first": round(trained["pol_loss_first"] or 0.0, 4),
        "pol_loss_last": round(trained["pol_loss_last"] or 0.0, 4),
        "q_absmax": round(trained["q_absmax"], 6),
        "mean_abs_reward": round(trained["mean_abs_reward"], 4),
    }


_ARM_CACHE: dict = {}


def _arm(seed: int, arm: str) -> dict:
    key = (seed, arm)
    if key not in _ARM_CACHE:
        _ARM_CACHE[key] = _eval_arm(seed, arm, _train_arm(seed, arm))
    return _ARM_CACHE[key]


def remote_run(seeds: list) -> dict:
    """Every arm this spec reads, for every seed. Runs on the GPU VM."""
    out = {"gpu": "cpu", "n_skills": N_SKILLS, "seeds": []}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for seed in seeds:
        row = {"seed": seed}
        for arm in _ARMS:
            row[arm] = _arm(seed, arm)
        out["seeds"].append(row)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import subprocess as _s, sys as _y, os as _o
_s.run([_y.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
import json
from experiments.tests.t2_11_skills_distinguishable import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t211.json"), "w"),
          indent=1)
print("DONE", out["gpu"], flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # PLACEHOLDER until the pilot measures it. `est_hours` feeds the GPU budget
    # ledger and the watcher timeout, so it is deliberately left as a number
    # this file cannot honour silently: `_GATES_FROZEN` is False and `run()`
    # refuses, so no submission can be made against a guessed cost. The pilot
    # writes the measured seconds-per-seed here in the same commit that flips
    # the freeze (T2.19's rule: calibrate, never guess).
    est_hours = round(0.10 + _SEC_PER_SEED / 3600.0 * len(seeds), 3)
    timeout_s = int(est_hours * 3600 * 1.5) + 900
    res = submit(job, prefer="kaggle", est_hours=est_hours,
                 timeout_s=timeout_s, fetch=["t211.json"])
    if not res.ok:
        raise RuntimeError(f"T2.11 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t211.json"]).read_text())
    out["backend"] = res.backend
    return out


_SEC_PER_SEED = 355.0           # MEASURED, 2026-08-29 pilot: 355.1 s (seed 7)
                                # and 355.2 s (seed 90), full registered scale,
                                # two concurrent on this box's 4 shared ARM
                                # cores — so it errs long for a solo x86 run,
                                # which is the direction a number feeding a
                                # watcher timeout must err in. It is NOT the
                                # cost of the redesigned rig (see PILOT
                                # RECORD): a shared conditioned policy replaces
                                # 8 tabular tables with a network forward per
                                # decision, and must be re-measured before any
                                # submission. `_GATES_FROZEN` is False, so no
                                # submission can be made against this number.


# ── the reading ──────────────────────────────────────────────────────────
def _seed_view(row: dict) -> dict:
    cl, ctl, zero = row[_CLAIM_ARM], row[_CONTROL_ARM], row["zero"]
    orc = row["oracle"]
    return {
        "seed": row["seed"],
        "claim_acc": cl["heldout_acc"],
        "claim_per_class_min": cl["per_class_min"],
        "margin_vs_shuffled": round(cl["heldout_acc"] - ctl["heldout_acc"], 4),
        "margin_vs_zero": round(cl["heldout_acc"] - zero["heldout_acc"], 4),
        # rig
        "shuffle_clf_fit": min(row[a]["shuffled_clf_train_acc"] for a in _ARMS),
        "shuffle_clf_heldout": max(row[a]["shuffled_clf_heldout_acc"]
                                   for a in _ARMS),
        "hash_overlap_max": max(row[a]["hash_overlap"] for a in _ARMS),
        "zero_coverage": zero["mean_coverage"],
        "min_coverage": min(row[a]["mean_coverage"] for a in _ARMS),
        "oracle_acc": orc["heldout_acc"],
        "zero_q_absmax": zero["q_absmax"],
        # reported, ungated
        "ctrl_acc": ctl["heldout_acc"],
        "ctrl_per_class_min": ctl["per_class_min"],
        "zero_acc": zero["heldout_acc"],
        "claim_centroid_sep": cl["centroid_sep_mean"],
        "ctrl_centroid_sep": ctl["centroid_sep_mean"],
        "claim_disc_loss_first": cl["disc_loss_first"],
        "claim_disc_loss_last": cl["disc_loss_last"],
        "ctrl_disc_loss_last": ctl["disc_loss_last"],
        "claim_mean_abs_reward": cl["mean_abs_reward"],
        "ctrl_mean_abs_reward": ctl["mean_abs_reward"],
        "oracle_per_class_min": orc["per_class_min"],
        "claim_q_absmax": cl["q_absmax"],
        "ctrl_q_absmax": ctl["q_absmax"],
        "claim_pol_loss_last": cl["pol_loss_last"],
        "ctrl_pol_loss_last": ctl["pol_loss_last"],
    }


def _fold(rows: list) -> dict:
    """Per-seed rows -> the numbers the gates read: WORST registered seed.

    Never a mean. `run_spec._aggregate` means whatever it is handed, so a spec
    whose gates are worst-case must fold before it returns — T2.09's scar,
    where a docstring saying "worst of N" was scored on the mean until `_fold`
    was written.
    """
    v = [_seed_view(r) for r in rows]

    def w(key, hi: bool):
        return (max if hi else min)(x[key] for x in v)

    return {
        "n_seeds": float(len(v)),
        "chance": CHANCE,
        # claim, worst seed
        "claim_acc": w("claim_acc", False),
        "claim_per_class_min": w("claim_per_class_min", False),
        "margin_vs_shuffled": w("margin_vs_shuffled", False),
        "margin_vs_zero": w("margin_vs_zero", False),
        # rig, worst seed
        "shuffle_clf_fit": w("shuffle_clf_fit", False),
        "shuffle_clf_heldout": w("shuffle_clf_heldout", True),
        "hash_overlap_max": w("hash_overlap_max", True),
        "zero_coverage": w("zero_coverage", False),
        "min_coverage": w("min_coverage", False),
        "oracle_acc": w("oracle_acc", False),
        "zero_q_absmax": w("zero_q_absmax", True),
        # reported
        "ctrl_acc": w("ctrl_acc", True),
        "zero_acc": w("zero_acc", True),
        "oracle_per_class_min": w("oracle_per_class_min", False),
        "claim_q_absmax": w("claim_q_absmax", False),
        "ctrl_q_absmax": w("ctrl_q_absmax", True),
        "claim_pol_loss_last": w("claim_pol_loss_last", True),
        "ctrl_pol_loss_last": w("ctrl_pol_loss_last", False),
        "claim_centroid_sep": w("claim_centroid_sep", False),
        "ctrl_centroid_sep": w("ctrl_centroid_sep", True),
        "claim_disc_loss_last": w("claim_disc_loss_last", True),
        "ctrl_disc_loss_last": w("ctrl_disc_loss_last", False),
        "claim_mean_abs_reward": w("claim_mean_abs_reward", False),
        "ctrl_mean_abs_reward": w("ctrl_mean_abs_reward", False),
        "per_seed": [[x["seed"], x["claim_acc"], x["ctrl_acc"], x["zero_acc"],
                      x["oracle_acc"], x["claim_per_class_min"],
                      x["margin_vs_shuffled"], x["hash_overlap_max"]]
                     for x in v],
        "per_seed_cols": ("seed claim_acc ctrl_acc zero_acc oracle_acc"
                          " claim_per_class_min margin_vs_shuffled"
                          " hash_overlap"),
    }


def _fold_control(rows: list) -> dict:
    """`shuffled` scored on the SHARED claim gates, over the same seeds.

    "Shared" is the load-bearing word, and it is why `_claim_holds` is stated
    over three gates rather than four. `beats_shuffled` cannot be asked of
    `shuffled` — a control cannot be a margin ahead of itself, and scoring it
    with that gate wired to 0.0 would make `not _claim_holds(c)` a tautology,
    i.e. a control that cannot fail. So the fourth gate is applied to the run
    ONLY, in `_check`, and the control faces the three that are satisfiable by
    it: it clears them if its skills really are legible, above chance, in every
    class, by a margin over the same random-walk floor.
    """
    v = [_seed_view(r) for r in rows]
    return {
        "claim_acc": min(x["ctrl_acc"] for x in v),
        "claim_per_class_min": min(x["ctrl_per_class_min"] for x in v),
        "margin_vs_zero": min(round(x["ctrl_acc"] - x["zero_acc"], 4)
                              for x in v),
        "ctrl_per_seed_acc": [x["ctrl_acc"] for x in v],
    }


def _experiment(seed: int) -> dict:
    """`seed` is ignored: one submission runs every seed and `_fold` reduces
    them to the worst. run_spec calls this once per registered seed and means
    identical dicts, so the recorded numbers are the fold."""
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    m = _fold(_CACHE["seeds"])
    m["gpu"] = _CACHE["gpu"]
    m["backend"] = _CACHE.get("backend", "?")
    return m


def _control(seed: int) -> dict:
    return _fold_control(_CACHE["seeds"])


def _claim_holds(m: dict) -> bool:
    """The three claim gates BOTH the run and the control can be scored on.

    `beats_shuffled` is deliberately absent — see `_fold_control` for why a
    control asked to beat itself is a control that cannot fail.
    """
    return (m["claim_acc"] >= CHANCE + ABOVE_CHANCE_MIN
            and m["claim_per_class_min"] >= PER_CLASS_MIN
            and m["margin_vs_zero"] >= MARGIN_MIN)


def _check(m: dict, c: dict):
    # A dead instrument or a leaking split is an APPARATUS outcome, not a
    # refutation: FAIL would fire this spec's `kills` field — deleting a
    # shipped component — off a run that never asked the question.
    rig = (m["shuffle_clf_fit"] >= SHUFFLE_FIT_FLOOR
           and m["shuffle_clf_heldout"] <= CHANCE + SHUFFLE_BAND
           and m["hash_overlap_max"] == 0
           and m["min_coverage"] > 0.0
           and m["zero_coverage"] >= FLOOR_COVERAGE
           # `learner_alive`: the shared policy CAN make skills legible when
           # the reward really depends on z. Without it, an under-trained net
           # reads exactly like a refuted DIAYN and fires this spec's `kills`.
           and m["oracle_acc"] >= ORACLE_MIN
           # `floor_is_uniform`: r == 0 with a zero head leaves the net
           # untouched, so `zero` is an exact uniform random walk rather than
           # a policy handed out by a random init. Arithmetic, not a bar.
           and m["zero_q_absmax"] == 0.0)
    if not rig:
        return Status.VOID
    return bool(_claim_holds(m)
                and m["margin_vs_shuffled"] >= MARGIN_MIN
                and not _claim_holds(c))


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "T2.11 gates are provisional — pilot first, freeze the bars in "
            "this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["T2.11"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":  # pilot: python -m experiments.tests.t2_11_... SEED OUT
    import time
    _seed = int(sys.argv[1])
    _out = sys.argv[2]
    _t0 = time.time()
    _res = remote_run([_seed])
    _res["wall_s"] = round(time.time() - _t0, 1)
    Path(_out).write_text(json.dumps(_res, indent=1))
    print("PILOT DONE", _out, _res["wall_s"], "s", flush=True)
