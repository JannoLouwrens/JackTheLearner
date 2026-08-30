"""DP.04 — The slow path may be verbal, and that is a claim, not a design.

*** GATES ARE PROVISIONAL. `_GATES_FROZEN = False` and `run()` REFUSES the
*** registered run until a pilot on seeds disjoint from 0/1/2 freezes every
*** bar below. Implementation-only, 2026-08-30. See PILOT PROTOCOL at the end
*** of this docstring.

THE QUESTION, and why it is a claim rather than a design decision. DP.03 and
most of the fast/slow literature quietly ASSUME the slow path is verbal — that
deliberation happens in something like inner speech. In language models the
gain from a reasoning trace is known to be partly the extra COMPUTE rather than
the CONTENT of the trace: filler tokens buy some of what a chain of thought
buys. So a transcript that looks like reasoning is not evidence that reasoning
happened in it, and this spec is built around exactly one comparison:

    the SAME extra internal steps, carrying HIS OWN tokens
    versus
    the SAME extra internal steps, carrying a content-free constant

If those two are equal, the words are decoration on extra computation and Jack
is not thinking in language, whatever the transcript looks like. That null —
`matched-compute filler` — is the null the registry names and it is the null
the metric is defined against.

WHAT THE CHANNEL IS, said plainly so nobody has to infer it. The emission is a
DISCRETE TOKEN from a small learned vocabulary, sampled by the policy, embedded
and fed back in as the input of the next internal step. It is a self-loop with
a categorical bottleneck: he emits, and the next step re-hears what he emitted
and nothing else about how he chose it. **The acoustic path is NOT modelled
here.** VO.01 certified that an emission arrives at an ear as sound — quieter
with distance, muffled through a solid — and that spec is where the sound
lives. What this spec would otherwise be is the failure VO.01 names, "a wire
between two brains wearing the word voice", pointed at his own ear. It is
declared rather than hidden because the claim being tested does not need the
acoustics: whether inner speech buys deliberation is a question about a
bottlenecked recurrent channel, and routing it through a room would add
attenuation, not evidence. A PASS here is a claim about the BOTTLENECK, and any
later spec that wants the sound must re-run it through VO.01's pipeline.

THE WORLD, and the reason it is DP.00's. The tasks are DP.00's certified
survival gridworld and DP.00's certified flat beacon world, imported rather
than re-typed. This matters twice over:

  1. DP.00 already measured, with the simulator itself as the planner's model,
     that lookahead pays in the survival world and provably cannot pay in the
     flat one. So the DELIBERATION DEMAND of each task is not asserted here —
     it is a number DP.00's own instrument produces, and this spec re-measures
     it per task variant with DP.00's code.
  2. The `base` variant (4 food, 4 water) IS `lc_00._World(seed)`, byte for
     byte, and `_check` VOIDs if it ever stops being — `base_world_identical`.
     A dose-response curve whose anchor point drifted from the world it claims
     to inherit is measuring two worlds.

TASKS AND THEIR DEMAND. Five task points per seed, four survival variants that
differ ONLY in how many resource cells the layout draws (2, 3, 4, 8 of each)
and DP.00's flat beacon world. Demand is measured, never assumed:

    demand = oracle_score(H=8) - reactive_score(H=1, strengthened)

in each task's own units, then normalised by that task's declared attainable
range so the five points share an axis. Sparser resources do not automatically
mean more demand — at 2 cells the resource is often further than an 8-step
horizon can see, so the planner loses its advantage too. That is why the axis
is the MEASURED gap and not the cell count, and why `demand_spread` is a rig
gate: if the five tasks turn out to have the same demand, the dose-response
axis does not exist and there is nothing to correlate against.

TWO PROPERTIES OF THAT AXIS, MEASURED AT THE SMOKE ENVELOPE BEFORE ANY GATE WAS
SET, said out loud because both compress it:

  - The H=8 oracle CENSORS at `LIFE_CAP` on every survival variant (it reaches
    200 and stops), so the demand differences between survival variants are
    carried almost entirely by the REACTIVE floor moving with resource density,
    not by the ceiling moving. DP.00 already declared this censoring and
    declared its direction: it understates the advantage, so it can only make
    the dose-response harder to see, never easier to manufacture.
  - The flat world's reactive arm IS its oracle — that is exactly what DP.00
    certified about it — so an "arm must beat the reactive baseline" liveness
    gate would be unsatisfiable there by construction. The alive-instrument
    floor is therefore the UNIFORM RANDOM WALKER in every task
    (`above_random_floor`), and the reactive comparison is kept where it means
    something: gate (e), on the survival variants only.

ARMS. One architecture, one training procedure, one dataset per (task, seed).
The only difference is what flows around the internal loop:

  verbal    K=4 internal steps. At each, a token is sampled from a learned
            head over a 12-symbol vocabulary (Gumbel-softmax straight-through
            in training, argmax at evaluation — eval is deterministic), embedded
            and fed to the recurrent cell.
  filler    K=4 internal steps, token forced to symbol 0 every time. The
            emission head is still EVALUATED and its output discarded, so the
            FLOPs are identical to `verbal` down to the matrix multiply; only
            the content of the channel differs. This is the null.
  mute      K=0. One forward pass, no channel, no extra compute. Not the null —
            a reference point, and the subject of a control gate below.
  scrambled THE CONTROL. `verbal`'s trained weights, evaluated with a fixed
            random permutation applied to the emitted symbol before it is
            re-heard. Identical statistics, identical bandwidth, identical
            compute, learned meaning destroyed. It must NOT help.

Every arm is trained by BEHAVIOUR CLONING on the H=8 oracle's optimal action
SET (a uniform soft target over ties, because an argmax over a tie is label
noise). Learning is removed as a confound the way DP.00 removed it: the teacher
is the simulator-as-model planner, identical for every arm, so the only thing
an arm can differ in is how much of that teacher its forward pass can express.

THE METRIC. `lookahead_gain_over_matched_compute_filler` = mean over the four
SURVIVAL variants of (score_verbal - score_filler), in steps of lifespan. The
flat variant is excluded from the headline number by construction — it is the
zero-demand point, and folding a task where the gain must be zero into the mean
would dilute the very statistic the claim is about. It appears in the
dose-response gate and in `gain_flat`, which is where it belongs.

WHY LIFESPAN. DP.00's argument, inherited: episodic return TELESCOPES in this
world (the sum of d(h)-d(h') over a life is -d(h_T)), so an agent that dies at
step 100 and one that dies at step 400 score the same. Lifespan is the
consequential quantity and it does not telescope. In the flat world the
consequential quantity is steps-to-beacon, so the score there is its NEGATION —
higher is better in both, stated because a sign error would invert a control.

GATES — VOID BEFORE FAIL. Six instrument gates run before the hypothesis is
allowed to be judged, because each one describes a world in which the
comparison is vacuous rather than lost:

  base_world_identical  the `base` variant is `lc_00._World(seed)` exactly, and
                        the shared-memo oracle used here agrees with DP.00's
                        `_action_scores` to 1e-12 on 200 probes per seed. The
                        memo is a 22x speedup and a 22x speedup is exactly the
                        kind of optimisation that silently changes an answer.
  arms_learned          every arm's training loss fell AND every arm's eval
                        score beats its task's UNIFORM-RANDOM floor. An
                        at-chance control must carry proof its instrument was
                        alive (LESSONS.md); a `verbal` arm compared against a
                        `filler` arm that never learned is measuring a dead arm.
  emit_entropy          the verbal arm's emitted-symbol entropy at evaluation,
                        averaged over internal positions, is at least
                        ENT_MIN nats. If the emission collapsed to a constant
                        then `verbal` IS `filler` and the comparison is between
                        an arm and itself — VOID, and specifically NOT the FAIL
                        that a lazy reading would record.
  headroom              on the survival variants the filler arm must sit at
                        least HEADROOM_MIN steps BELOW the measurement CEILING
                        (`LIFE_CAP`). If the null is already at the ceiling the
                        statistic cannot move upward and a zero gain says
                        nothing about the channel. `headroom_to_oracle_*` is
                        reported beside it and is NOT the gate — the H=8 oracle
                        is itself censored at the cap, so a null at the cap
                        "ties the teacher" for a reason that is about the cap.
  demand_spread         measured demand must vary across the five tasks by at
                        least SPREAD_MIN of the normalised range, or the
                        dose-response axis does not exist.
  flat_demand_zero      DP.00's flat world must re-measure as zero-demand here
                        (planner gain <= CTRL_TOL steps). This is DP.00's own
                        certified control, re-run on this spec's code path; if
                        it fires, the task set is not what this spec says it is.

THE HYPOTHESIS, judged only after all six pass. PASS requires ALL of:

  (a) gain over the matched-compute filler >= MIN_GAIN steps, and >= SIGMA_GATE
      sigma across the three seeds. Both, because a margin without a sigma is a
      seed lottery and a sigma without a margin is a rounding error.
  (b) THE CONTROL FAILS: the scrambled-vocabulary arm's gain is at most
      SCRAM_FRAC of the verbal arm's, and at most SCRAM_ABS steps outright. If
      permuting the symbols leaves the gain intact, the channel was bandwidth
      and compute, not meaning — and the claim is refuted, not supported.
  (c) ZERO DEMAND, ZERO GAIN: |gain_flat| <= FLAT_TOL. The registry names this
      as a falsifier in its own right ("equal gain on tasks with zero planning
      demand"), because a channel that helps everywhere equally is helping with
      something other than lookahead.
  (d) DOSE-RESPONSE: Pearson correlation between normalised demand and
      normalised gain across the five task points >= RHO_MIN. Five points is a
      weak correlation instrument on its own and this file says so; it is gated
      in conjunction with (c), which pins the intercept at the one task where
      the answer is known in advance.
  (e) THE MUTE ARM STILL DELIBERATES: mute must beat its tasks' reactive floor
      by at least MUTE_FLOOR_MIN steps. The registry requires this and the
      reason is constitutional rather than statistical — if removing the verbal
      channel destroys lookahead entirely, then language became load-bearing
      for thought, which contradicts one brain with all senses and a Jack who
      could think before he could speak. Its failure is a FAIL that must be
      read as a finding about the architecture, not as a rig fault: the arm ran,
      the measurement stands, and what it refutes is a premise this project
      holds. Say so in the journal if it ever fires.

WHAT A FAIL WOULD MEAN. That the extra internal steps are what buy the gain and
the symbols riding on them are decoration — the filler-token result, reproduced
in a creature rather than in a chat model. That is a real finding and it kills
the reading of DP.03 in which the slow path is assumed verbal. It does not kill
the verbal channel as an EFFECTOR (VO.01, VO.02) and it says nothing about
whether he can be talked TO (LG.00, LG.01).

WHAT THIS SPEC DOES NOT CLAIM. Not that the symbols are words — they are twelve
uninterpreted indices and this file never calls them language. Not that inner
speech is internalised from the parent's speech, which is the Vygotskian
prediction the registry records and which needs an LG-family successor to test
ordering and meaning-attachment. Not that the result transfers to a world with
traps, delays or irreversibility: DP.05 FAILED on 2026-08-24 and its
pre-registered routing binds, so a PASS here is a claim about DP.00's gridworld
and inherits DP.00's own scope, no wider.

DETERMINISM. Evaluation takes the argmax symbol and the argmax action with ties
broken by lowest index; no Gumbel noise, no sampling, no dropout. Training is
seeded per (task, arm, seed) from a string key. Torch runs on CPU with a
declared thread count; the box has 2 cores and this spec is sized to them.

COST. Measured on the 2-core box before any gate was written: the shared-memo
H=8 oracle costs 0.32 ms per decision against 7.0 ms for DP.00's per-call memo,
which is what makes 12,000 supervised labels per (task, seed) affordable at
all. Projected ~90 s per seed; the pilot MEASURES it and the freezing commit
records the number. `est_hours` is not guessed here — GPU_SHORT is the
registry's budget and `_GATES_FROZEN` refuses any submission before the pilot.

PILOT RECORD v1 — BOTH SEEDS VOID ON RIG GATES. THE ENVELOPE CANNOT MEASURE A
FIVE-STEP EFFECT AND IT SATURATES AT THE TOP. `_GATES_FROZEN` STAYS FALSE, NO
DISPATCH, NO CLAIM. (builder, 2026-08-30 19:21-19:25 UTC, head 466a2cf,
artifacts /data/dp04_pilot_seed{90,91}.json, 135.0 s and 130.8 s wall on the
2-core box — the compute was never the problem.)

    seed 90                 rand   react  oracle |  mute filler verbal | ent
      res2                 133.75  112.50  200.00| 191.67 200.00 183.33 | 0.897
      res3                 134.08  112.50  200.00| 200.00 191.67 191.67 | 1.029
      res4                 145.08  112.50  200.00| 191.67 200.00 200.00 | 1.282
      res8                 158.17  120.83  200.00| 200.00 191.67 191.67 | 0.743
      flat                 -57.50   -9.30   -9.30|  -9.30  -9.30  -9.30 | 1.003
    seed 91
      res2                 102.08  112.50  168.75| 175.00 168.75 175.00 | 1.286
      res3                 118.00  108.33  200.00| 200.00 128.92 132.83 | 0.876
      res4                 119.42  108.33  200.00| 137.58 183.33 112.25 | 0.866
      res8                 175.08  139.58  200.00| 191.67 191.67 200.00 | 0.000
      flat                 -59.20   -9.50   -9.50|  -9.50  -9.50  -9.50 | 1.051

WHAT PASSED, and it is most of the instrument. `base_world_identical` 1.0 and
`scorer_mismatch` 0.0 on both seeds — the base variant IS `lc_00._World(seed)`
and the 22x shared-memo oracle agrees with DP.00's scorer to 1e-12 on 200
probes. `demand_flat_steps` 0.000 on both: DP.00's flat world re-measures as
exactly zero-demand on this spec's code path, its reactive arm reaching -9.30
and -9.50 against an identical oracle, which is DP.00's own certification
reproduced here. `demand_spread` 0.875 / 0.917, `losses_fell` 1.0 on both.

THREE FAULTS, each caught by a different pre-registered gate, each about the
ENVELOPE and none about the hypothesis:

  1. SATURATION AT THE CENSORING CAP (seed 90, `headroom`). The filler arm sits
     AT `LIFE_CAP` on res2 and res4 (200.00), and 8.33 below it on the other
     two. The statistic cannot move upward, so a zero gain measures the cap.
     This is also what corrected the gate itself: it originally read
     `oracle - filler`, and since the H=8 oracle is ALSO censored at 200 that
     difference reads 0.00 for a reason that is about the cap rather than about
     the null. It now measures the distance to the ceiling and reports the
     oracle version beside it.
  2. TRAINING AND EVAL VARIANCE LARGER THAN THE EFFECT (seed 91,
     `above_random_floor`). The verbal arm scores 112.25 on res4 — BELOW that
     task's uniform random walker at 119.42 — and 200.00 on res8, from one
     training run per (task, arm) and 12 evaluation lives whose per-life range
     is [100, 200]. `MIN_GAIN` is 5.0 steps. Nothing this noisy can resolve it.
  3. EMISSION COLLAPSE (seed 91, res8, `emit_entropy` 0.000). The verbal arm's
     symbol went constant on that task, which makes it the filler arm wearing
     another name. The gate fired and called it VOID, which is the whole reason
     that gate exists and is NOT a FAIL.

READ THIS BEFORE QUOTING ANY NUMBER ABOVE. The verbal arm LOST to the filler on
both pilot seeds (mean gain -4.17 and -13.15). **That is not evidence about the
claim and must never be reported as any.** Every seed VOIDed on a rig gate, a
VOID is not a FAIL (protocol.py:208), and an arm that fell below a random
walker and an arm whose channel went constant are not arms that tested
anything. The pilot seeds 90 and 91 are now SPENT.

THE REPAIRS ARE SIZING, AND EACH IS A MEASUREMENT RATHER THAN A PREFERENCE —
pre-registered here so the next iteration does not improvise them:

  (a) The ceiling. `LIFE_CAP = 200` is DP.00's halved value against LC.00's
      400, halved for a CPU budget this rig does not have a problem with (135 s
      per seed). Raising it moves the ceiling the statistic hit; the oracle's
      cost scales with it and must be re-measured, not assumed.
  (b) The noise. `N_EVAL_LIVES` and the number of training restarts must be
      SIZED AGAINST THE MEASURED per-arm spread, BA.03's precedent: run one
      arm repeatedly, read its sigma, and set the counts so that `MIN_GAIN`
      sits at the stated sigma. That is arithmetic. It is not a licence to move
      `MIN_GAIN`, which is a claim bar and stays where it is.
  (c) The collapse. One training run per (task, arm) makes a single unlucky
      initialisation the arm. Reporting the median of R restarts is the
      standard repair and it changes no gate.

NOTHING ABOVE SIZES A CLAIM BAR, and that is deliberate: the pilot produced no
valid measurement of the claim statistic, so there is nothing to size
`MIN_GAIN` / `SCRAM_FRAC` / `RHO_MIN` / `FLAT_TOL` against. They are unchanged
from the values committed before the pilot ran. Freezing them off a VOID run
would be fitting a gate to noise.

PILOT PROTOCOL, pre-registered. `python -m experiments.tests.dp_04_slow_path_verbal
pilot` runs seeds 90 and 91 — disjoint from the registered 0/1/2 and SPENT once
used — writes /data/dp04_pilot_seed{90,91}.json, and reports every quantity the
gates below read. The freezing commit must (1) paste the pilot table into this
docstring, (2) set each provisional bar from it in the open, (3) set
`_GATES_FROZEN = True`, and (4) state the measured wall time. A gate fitted to
the run it judges is not a gate. If the pilot shows the verbal arm winning on
the pilot seeds, that is NOT evidence for the claim and must not be reported as
any — it is a sizing measurement, and the registered seeds decide.

SIZING RECORD v1 — THE PRE-REGISTERED REPAIRS ARE REFUTED BY MEASUREMENT. THE
FAULT IS NOT THE ENVELOPE'S SIZE, IT IS THAT THE CLAIM STATISTIC HAS NO
RESOLUTION IN THIS WORLD. `_GATES_FROZEN` STAYS FALSE, AND **DO NOT RE-PILOT
DP.04 UNTIL A REDESIGN LANDS** — seeds 92/93 are NOT to be spent on this
envelope. (builder, 2026-08-30 21:0x-21:2x UTC, head 393881b, artifact
/data/dp04_sizing_seed94.json, 1312.2 s wall on the 2-core box, seed 94 SPENT.)

Protocol as pre-registered above: seed 94, 4 survival variants, 8 restarts per
(task, arm) for verbal and filler, 48 lives each, `LIFE_CAP` raised to 400.

REPAIR (a) — THE CEILING — IS REFUTED, and by the cleanest possible reading.
Raising the cap from 200 to 400 un-censored **exactly zero lives**:

    3072 lives recorded (4 tasks x 2 arms x 8 restarts x 48 lives)
      lifespans ending strictly between 200 and 400 ....    0
      lifespans == 400 (censored at the new cap) ....... 2356   (76.7%)
      lifespans <= 100 ................................   550   (17.9%)
      lifespans in (100, 200) .........................   166    (5.4%)
      DISTINCT lifespan values in the whole run .......    21
    sat_frac_200 == sat_frac_400 for all 8 (task, arm) pairs, to the digit.

A trained clone in this world either dies in its first ~130 steps or finds a
stable food/water cycle and survives to ANY cap you choose. Lifespan is not a
graded quality measure here; it is a survive/die indicator wearing a continuous
type. The cap was never the binding constraint, so moving it bought nothing —
and the H=8 oracle is itself censored at 400 on res2/res3/res8
(`oracle_at_cap` 1), which is why `headroom` cannot be opened by a taller
ceiling either.

REPAIR (b) — THE NOISE — IS REFUTED AS SUFFICIENT. The target is derived, not
chosen: `MIN_GAIN * sqrt(2) / SIGMA_GATE` = **2.357 steps**. No design in the
pre-registered grid reaches it, and raising the cap makes it worse, because
censoring at 200 was compressing the spread rather than distorting it:

    cap  E   R=1     R=3     R=5     R=7      (bootstrap sd of the gain, steps)
    200  12  10.96    9.05    7.93    7.14
    200  24  10.81    8.15    6.87    6.10
    200  48   9.16    7.02    5.85    5.18   <- best in grid, still 2.2x over
    400  12  36.40   28.26   24.11   22.65
    400  48  31.58   22.65   18.63   16.57

AND THE ARITHMETIC SAYS WHY, so this is not a "needs more seeds" result. With
76.7% of lives at the cap the mean lifespan is ~100 + 300p for a Bernoulli p,
so at E lives the statistic is QUANTISED at 300/E steps and its paired-gain sd
is 300*sqrt(2p(1-p)/E):

    E=12  quantum 25.00 steps   gain sd 51.8
    E=24  quantum 12.50 steps   gain sd 36.6
    E=48  quantum  6.25 steps   gain sd 25.9

**`MIN_GAIN` is 5.0 and the finest difference the statistic can express at 48
lives is 6.25.** The gate asks for a difference smaller than the instrument's
smallest step. Reaching a 2.357-step sd from the Bernoulli term alone needs
**E >= 5791 lives per arm per task** — ~120x the eval budget, before a single
restart and before the world-to-world term the sizing run deliberately could
not measure. `MIN_GAIN` is a claim bar and does not move; the metric must.

WHAT IS NOT CONCLUDED. Every arm learned (`losses_fell_all` 1.0 on all eight
task/arm pairs), so this is not a dead-arm result, and NO number here is
evidence about the hypothesis — the gain means in the table above are sizing
quantities on a spent seed and must never be reported as a claim. The verbal
arm is neither vindicated nor refuted by this run.

THE FINDING, and it is about the WORLD, not about DP.04's hypothesis: W0's
survival task is near-binary at every cap, so a mean-lifespan statistic cannot
resolve a 5-step effect at any affordable envelope. That is the FIFTH
independent instrument in this project to land on the world as the bottleneck
(LC.03's darkroom control, LC.03 v2's one-learner-in-five, DP.05's FAIL,
SH.01's ORACLE_CANNOT, and now this). The repair is a DESIGN change — a graded
outcome measure, or a world whose difficulty is tuned so survival is not
almost-free — and both candidate families are runnable arms, so under law 3 it
is a bakeoff somebody has to write, not an argument. Routed to the Review and
to `docs/DECISIONS_NEEDED.md` as `dp04-lifespan-has-no-resolution`.
"""

# WHY A PILOT IS NOT THE REPAIR HERE, declared so the QUEUE-DEPTH instrument
# stops advertising one. `_GATES_FROZEN = False` alone reads "PILOT OWED
# (cheapest repair)" in `run coverage` — true for most provisional specs and
# false for this one, because SIZING RECORD v1 measured the pilot's own
# precondition and refuted it. Spending seeds 92/93 now would buy a third VOID.
_PILOT_BLOCKED = (
    "SIZING RECORD v1 (seed 94, 2026-08-30): mean censored lifespan has no "
    "resolution in W0 — 0 of 3072 lives ended between the old cap and the new "
    "one, 21 distinct lifespans, quantum 6.25 steps at 48 lives against "
    "MIN_GAIN 5.0, and E>=5791 lives/arm/task would be needed for the derived "
    "2.357-step sd. The repair is a world/metric redesign (Review + "
    "DECISIONS_NEEDED `dp04-lifespan-has-no-resolution`), not a pilot."
)
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .dp_00_lookahead_pays import (CTRL_TOL, FLAT_EPISODES, FLAT_MIN_D,
                                   FLAT_SHAPE, FLAT_T, LIFE_CAP, _action_scores,
                                   _flat_scores, _flat_sim, _pick, _sim)
from .lc_00_gridworld_decidable import (ACTIONS, DEPLETE, GAMMA, N_FOOD, SIZE,
                                        _World)

IMPL_DEPS = ["experiments/tests/lc_00_gridworld_decidable.py",
             "experiments/tests/dp_00_lookahead_pays.py"]

# ── the rig ──────────────────────────────────────────────────────────────
H_ORACLE = 8                   # DP.00's H_MAX; the teacher's planning depth
RES_COUNTS = (2, 3, 4, 8)      # food cells == water cells, per survival variant
N_EVAL_LIVES = 12              # lives per arm per task per seed
LIFE_FLOOR = int(1.0 / DEPLETE[0])          # nothing dies before this step
N_LABEL = 12000                # supervised states per task per seed
MEMO_CAP = 300_000             # oracle memo entries before it is cleared (RAM)
EPS_COLLECT = 0.30             # off-oracle action rate while collecting states
N_SCORER_PROBES = 200          # shared-memo vs DP.00 `_action_scores` fidelity

# ── the agent ────────────────────────────────────────────────────────────
HID = 96
VOCAB = 12
EMB = 16
K_STEPS = 4                    # internal steps for `verbal` and `filler`
TAU = 1.0                      # Gumbel-softmax temperature (training only)
EPOCHS = 12
BATCH = 256
LR = 2e-3
THREADS = 2                    # this box has 2 cores; declared, not inherited

OBS_DIM = 3 * SIZE * SIZE + 2  # food / water / self planes, plus two needs
N_ACT = len(ACTIONS)

# ── PROVISIONAL GATES — every one of these is set by the pilot ───────────
_GATES_FROZEN = False

ENT_MIN = 0.30            # nats; below this the emission collapsed to a constant
HEADROOM_MIN = 8.0        # steps the filler arm must sit below the oracle
SPREAD_MIN = 0.10         # normalised demand range across the five tasks
MIN_GAIN = 5.0            # steps of lifespan, verbal over filler
SIGMA_GATE = 3.0          # bakeoff.py:159 / LC.00 / DP.00's ruler, unpaired
SCRAM_FRAC = 0.40         # scrambled gain as a fraction of verbal's
SCRAM_ABS = 4.0           # ...and an absolute ceiling in steps
FLAT_TOL = 0.04           # |normalised gain| allowed at zero demand
RHO_MIN = 0.60            # Pearson r, normalised demand vs normalised gain
MUTE_FLOOR_MIN = 5.0      # steps the mute arm must clear its reactive floor by

_PILOT_SEEDS = (90, 91)
_PILOT_ARTIFACT = "/data/dp04_pilot_seed%d.json"

# ── SIZING RUN — repairs (b) and (c) of PILOT RECORD v1 ──────────────────
# Seed 94 is a SIZING seed: disjoint from the registered 0/1/2 and from the
# pilot seeds 90/91 (spent) and 92/93 (reserved for pilot v2). It measures
# noise, never the claim, and it is SPENT for claims once used.
_SIZE_SEED = 94
_SIZE_ARTIFACT = "/data/dp04_sizing_seed94.json"
_SIZE_R = 8               # independent training restarts per (task, arm)
_SIZE_E = 48              # eval lives per restart; prefix-scored at 12/24/48
_SIZE_CAP = 400           # LC.00's original ceiling; repair (a)'s candidate
_SIZE_ES = (12, 24, 48)   # eval counts reported (prefixes of the same spawns)
_SIZE_CAPS = (200, 400)   # censoring caps reported, from the SAME raw spans
_SIZE_RS = (1, 3, 5, 7)   # restart counts the sizing arithmetic is solved for

# The attainable range each task's score is normalised by, asserted so the
# normalisation cannot rot underneath the dose-response gate.
assert LIFE_CAP > LIFE_FLOOR, "lifespan has no room above its own floor"
assert MIN_GAIN < (LIFE_CAP - LIFE_FLOOR), "the gain gate exceeds the range"
assert K_STEPS >= 1, "the channel needs at least one internal step"
assert 0 <= 0 < VOCAB, "the filler symbol must be in the vocabulary"


# ── worlds ───────────────────────────────────────────────────────────────

class _VariantWorld:
    """LC.00's layout law with the resource count as a parameter.

    At `n == N_FOOD` this reproduces `lc_00._World(seed)` EXACTLY — same rng
    key, same `sample` call, same slicing — which `_rig` asserts per seed. The
    base variant is therefore DP.00's world and not a look-alike.
    """

    def __init__(self, seed: int, n: int):
        rng = random.Random(f"lc00-world-{seed}")
        cells = [(x, y) for x in range(SIZE) for y in range(SIZE)]
        picks = rng.sample(cells, 2 * n)
        self.food = frozenset(picks[:n])
        self.water = frozenset(picks[n:])


def _scores_shared(world, x, y, h0, h1, horizon, memo) -> list:
    """DP.00's `_action_scores` with the memo hoisted out of the call.

    The memo key is state-only, so sharing it across the whole dataset for a
    fixed world is exact, not approximate — and `_rig` proves that per seed
    rather than asserting it. Measured 0.32 ms/decision against 7.0 ms.
    """
    def val(x, y, h0, h1, d):
        if d == 0:
            return 0.0
        key = (x, y, round(h0, 6), round(h1, 6), d)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = -1e18
        for a in range(N_ACT):
            nx, ny, n0, n1, r, dead = _sim(world, x, y, h0, h1, a)
            v = r if dead else r + GAMMA * val(nx, ny, n0, n1, d - 1)
            if v > best:
                best = v
        memo[key] = best
        return best

    out = []
    for a in range(N_ACT):
        nx, ny, n0, n1, r, dead = _sim(world, x, y, h0, h1, a)
        out.append(r if dead else r + GAMMA * val(nx, ny, n0, n1, horizon - 1))
    return out


def _flat_scores_shared(bx, by, x, y, horizon, memo) -> list:
    """The same hoist for DP.00's flat world; key carries the beacon."""
    def val(x, y, d):
        if d == 0:
            return 0.0
        key = (bx, by, x, y, d)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = -1e18
        for a in range(N_ACT):
            nx, ny, r, done = _flat_sim(bx, by, x, y, a)
            v = r if done else r + GAMMA * val(nx, ny, d - 1)
            if v > best:
                best = v
        memo[key] = best
        return best

    out = []
    for a in range(N_ACT):
        nx, ny, r, done = _flat_sim(bx, by, x, y, a)
        out.append(r if done else r + GAMMA * val(nx, ny, horizon - 1))
    return out


def _optimal_set(scores) -> list:
    top = max(scores)
    return [a for a, v in enumerate(scores) if v >= top - 1e-12]


def _seed_of(key: str) -> int:
    """A STABLE seed from a string. `hash()` is salted per process, so seeding
    torch from it makes every run a different experiment and T0.02's
    determinism claim silently false — found by review before the pilot, not
    by a run that disagreed with itself."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ── tasks ────────────────────────────────────────────────────────────────

class _Survival:
    """One survival variant: obs, teacher, rollout, floors and ceilings."""

    kind = "survival"

    def __init__(self, seed: int, n_res: int):
        self.seed = seed
        self.n_res = n_res
        self.name = f"res{n_res}"
        self.world = _VariantWorld(seed, n_res)
        self.memo: dict = {}
        self.floor = float(LIFE_FLOOR)
        self.ceil = float(LIFE_CAP)
        self._planes = torch.zeros(2, SIZE, SIZE)
        for (x, y) in self.world.food:
            self._planes[0, x, y] = 1.0
        for (x, y) in self.world.water:
            self._planes[1, x, y] = 1.0

    def _memo_guard(self):
        if len(self.memo) > MEMO_CAP:
            self.memo.clear()

    def obs(self, st) -> torch.Tensor:
        x, y, h0, h1 = st
        v = torch.zeros(3, SIZE, SIZE)
        v[:2] = self._planes
        v[2, x, y] = 1.0
        return torch.cat([v.reshape(-1), torch.tensor([h0, h1])])

    def teacher(self, st) -> list:
        self._memo_guard()
        return _optimal_set(_scores_shared(self.world, *st, H_ORACLE, self.memo))

    def collect(self, n: int, rng: random.Random):
        """States along an epsilon-greedy-on-the-oracle trajectory.

        The oracle's own path plus a declared EPS_COLLECT of off-oracle actions:
        a policy cloned only from states the teacher visits inherits the
        teacher's distribution and falls apart one step off it.
        """
        obs, tgt = [], []
        x = y = 0
        h0 = h1 = 0.0
        steps = LIFE_CAP
        while len(obs) < n:
            if steps >= LIFE_CAP:
                x, y = rng.randrange(SIZE), rng.randrange(SIZE)
                h0 = h1 = 1.0
                steps = 0
            best = self.teacher((x, y, h0, h1))
            obs.append(self.obs((x, y, h0, h1)))
            t = torch.zeros(N_ACT)
            for a in best:
                t[a] = 1.0 / len(best)
            tgt.append(t)
            a = (rng.randrange(N_ACT) if rng.random() < EPS_COLLECT
                 else best[rng.randrange(len(best))])
            x, y, h0, h1, _r, dead = _sim(self.world, x, y, h0, h1, a)
            steps += 1
            if dead:
                steps = LIFE_CAP
        return torch.stack(obs), torch.stack(tgt)

    def rollout_spans(self, act_fn, n: int | None = None) -> list:
        """The raw per-life lifespans, censored at LIFE_CAP.

        Split out of `rollout` for the sizing run, which needs the spans rather
        than their mean: the spawn sequence is drawn from a fixed key, so the
        first `N_EVAL_LIVES` entries of an `n > N_EVAL_LIVES` call are exactly
        the lives the registered envelope would have scored. That prefix
        property is what lets one sizing run report every eval count without
        re-running anything, and it is why `n` extends the sequence rather than
        re-seeding it.
        """
        spawn = random.Random(f"dp04-spawn-{self.seed}-{self.name}")
        spans = []
        for _ in range(N_EVAL_LIVES if n is None else n):
            x, y = spawn.randrange(SIZE), spawn.randrange(SIZE)
            h0 = h1 = 1.0
            steps = 0
            while steps < LIFE_CAP:
                a = act_fn(self.obs((x, y, h0, h1)))
                x, y, h0, h1, _r, dead = _sim(self.world, x, y, h0, h1, a)
                steps += 1
                if dead:
                    break
            spans.append(steps)
        return spans

    def rollout(self, act_fn, tag: str) -> float:
        """Mean lifespan, censored at LIFE_CAP. Spawns are identical per arm."""
        spans = self.rollout_spans(act_fn)
        return sum(spans) / len(spans)

    def reference(self) -> tuple:
        """(reactive floor, oracle ceiling) — DP.00's two arms, this variant.

        The reactive arm is DP.00's STRENGTHENED null: the per-variant maximum
        of uniform tie-break and persistence.
        """
        oracle = self._planner_score(H_ORACLE)
        react = max(self._planner_score(1, persist=False),
                    self._planner_score(1, persist=True))
        return react, oracle

    def random_score(self) -> float:
        """The ALIVE-INSTRUMENT floor: a uniform random walker, same spawns.

        Not the same thing as the reactive null. In this world DP.00 showed the
        H=1 arm is already a serious baseline, so requiring a trained arm to
        beat REACTIVE is a claim about quality; requiring it to beat RANDOM is
        the check that the arm learned anything at all, which is what a VOID
        gate is for."""
        rng = random.Random(f"dp04-rand-{self.seed}-{self.name}")
        spawn = random.Random(f"dp04-spawn-{self.seed}-{self.name}")
        spans = []
        for _ in range(N_EVAL_LIVES):
            x, y = spawn.randrange(SIZE), spawn.randrange(SIZE)
            h0 = h1 = 1.0
            steps = 0
            while steps < LIFE_CAP:
                x, y, h0, h1, _r, dead = _sim(self.world, x, y, h0, h1,
                                              rng.randrange(N_ACT))
                steps += 1
                if dead:
                    break
            spans.append(steps)
        return sum(spans) / len(spans)

    def _planner_score(self, horizon: int, persist: bool = False) -> float:
        rng = random.Random(f"dp04-arm-{self.seed}-{self.name}-{horizon}-{persist}")
        spawn = random.Random(f"dp04-spawn-{self.seed}-{self.name}")
        spans = []
        for _ in range(N_EVAL_LIVES):
            x, y = spawn.randrange(SIZE), spawn.randrange(SIZE)
            h0 = h1 = 1.0
            prev, steps = None, 0
            while steps < LIFE_CAP:
                self._memo_guard()
                a = _pick(_scores_shared(self.world, x, y, h0, h1, horizon,
                                         self.memo), rng, prev, persist)
                prev = a
                x, y, h0, h1, _r, dead = _sim(self.world, x, y, h0, h1, a)
                steps += 1
                if dead:
                    break
            spans.append(steps)
        return sum(spans) / len(spans)


class _Flat:
    """DP.00's flat beacon world — provably reactive-solvable, demand zero.

    Score is the NEGATION of steps-to-beacon so that higher is better in every
    task; the ceiling is the live shortest path (DP.00's own rule: derived from
    the episode, never from a constant) and the floor is the episode cap.
    """

    kind = "flat"
    name = "flat"

    def __init__(self, seed: int):
        self.seed = seed
        self.memo: dict = {}
        setup = random.Random(f"dp00-flatsetup-{seed}")
        self.episodes = []
        for _ in range(FLAT_EPISODES):
            while True:
                bx, by = setup.randrange(SIZE), setup.randrange(SIZE)
                x, y = setup.randrange(SIZE), setup.randrange(SIZE)
                if abs(x - bx) + abs(y - by) >= FLAT_MIN_D:
                    break
            self.episodes.append((bx, by, x, y))
        self.shortest = sum(abs(x - bx) + abs(y - by)
                            for bx, by, x, y in self.episodes) / FLAT_EPISODES
        self.floor = -float(FLAT_T)
        self.ceil = -self.shortest

    def obs(self, st) -> torch.Tensor:
        bx, by, x, y = st
        v = torch.zeros(3, SIZE, SIZE)
        v[0, bx, by] = 1.0            # the beacon rides the food plane
        v[2, x, y] = 1.0
        return torch.cat([v.reshape(-1), torch.tensor([1.0, 1.0])])

    def teacher(self, st) -> list:
        bx, by, x, y = st
        if len(self.memo) > MEMO_CAP:
            self.memo.clear()
        return _optimal_set(_flat_scores_shared(bx, by, x, y, H_ORACLE,
                                                self.memo))

    def collect(self, n: int, rng: random.Random):
        obs, tgt = [], []
        while len(obs) < n:
            bx, by, x, y = self.episodes[rng.randrange(FLAT_EPISODES)]
            steps = 0
            while steps < FLAT_T and len(obs) < n:
                best = self.teacher((bx, by, x, y))
                obs.append(self.obs((bx, by, x, y)))
                t = torch.zeros(N_ACT)
                for a in best:
                    t[a] = 1.0 / len(best)
                tgt.append(t)
                a = (rng.randrange(N_ACT) if rng.random() < EPS_COLLECT
                     else best[rng.randrange(len(best))])
                x, y, _r, done = _flat_sim(bx, by, x, y, a)
                steps += 1
                if done:
                    break
        return torch.stack(obs), torch.stack(tgt)

    def rollout(self, act_fn, tag: str) -> float:
        taken = []
        for bx, by, x, y in self.episodes:
            steps = 0
            while steps < FLAT_T:
                a = act_fn(self.obs((bx, by, x, y)))
                x, y, _r, done = _flat_sim(bx, by, x, y, a)
                steps += 1
                if done:
                    break
            taken.append(steps)
        return -sum(taken) / len(taken)

    def reference(self) -> tuple:
        return -self._planner_steps(1), -self._planner_steps(H_ORACLE)

    def random_score(self) -> float:
        """DP.00's `broken` arm: the uniform walker, as the alive floor."""
        rng = random.Random(f"dp04-flatrand-{self.seed}")
        taken = []
        for bx, by, x, y in self.episodes:
            steps = 0
            while steps < FLAT_T:
                x, y, _r, done = _flat_sim(bx, by, x, y, rng.randrange(N_ACT))
                steps += 1
                if done:
                    break
            taken.append(steps)
        return -sum(taken) / len(taken)

    def _planner_steps(self, horizon: int) -> float:
        rng = random.Random(f"dp04-flat-{self.seed}-{horizon}")
        taken = []
        for bx, by, x, y in self.episodes:
            steps = 0
            while steps < FLAT_T:
                a = _pick(_flat_scores_shared(bx, by, x, y, horizon, self.memo),
                          rng, None, False)
                x, y, _r, done = _flat_sim(bx, by, x, y, a)
                steps += 1
                if done:
                    break
            taken.append(steps)
        return sum(taken) / len(taken)


# ── the agent ────────────────────────────────────────────────────────────

class _Agent(nn.Module):
    """One architecture; `mode` selects what rides the internal loop.

    `filler` evaluates the emission head and DISCARDS it, so the arm pays the
    same matrix multiply as `verbal` and differs only in the content of the
    symbol it re-hears. That is what makes the null matched-compute rather than
    matched-in-spirit.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.k = 0 if mode == "mute" else K_STEPS
        self.enc = nn.Sequential(nn.Linear(OBS_DIM, HID), nn.ReLU(),
                                 nn.Linear(HID, HID), nn.Tanh())
        self.cell = nn.GRUCell(EMB, HID)
        self.emit = nn.Linear(HID, VOCAB)
        self.sym = nn.Embedding(VOCAB, EMB)
        self.act = nn.Linear(HID, N_ACT)

    def forward(self, obs, train: bool, perm=None):
        h = self.enc(obs)
        toks = []
        for _ in range(self.k):
            logits = self.emit(h)                    # evaluated in EVERY arm
            if self.mode == "filler":
                one = torch.zeros_like(logits)
                one[:, 0] = 1.0                      # content-free constant
            elif train:
                one = F.gumbel_softmax(logits, tau=TAU, hard=True)
            else:
                idx = logits.argmax(dim=-1)
                one = F.one_hot(idx, VOCAB).float()
            if perm is not None:
                one = one[:, perm]                   # the scrambled control
            toks.append(one.argmax(dim=-1))
            h = self.cell(one @ self.sym.weight, h)
        return self.act(h), toks


def _train(task, arm: str, seed: int, obs, tgt, restart: int = 0) -> dict:
    """One supervised fit. `restart` selects an independent initialisation.

    Restart 0 keeps the ORIGINAL key, byte for byte, so adding this parameter
    cannot move any number the pilot already recorded — the sizing run reads
    restarts 0..R-1 and restart 0 is the run the current envelope makes.
    """
    tag = f"dp04-{seed}-{task.name}-{arm}" + (f"-r{restart}" if restart else "")
    btag = f"dp04-b-{seed}-{task.name}-{arm}" + (f"-r{restart}" if restart else "")
    torch.manual_seed(_seed_of(tag))
    net = _Agent(arm)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = obs.shape[0]
    g = torch.Generator().manual_seed(_seed_of(btag))
    first = last = None
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits, _ = net(obs[idx], train=True)
            loss = -(tgt[idx] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        if ep == 0:
            first = tot / n
        last = tot / n
    net.eval()
    return {"net": net, "loss_first": first, "loss_last": last}


def _act_fn(net: _Agent, perm=None):
    """Deterministic greedy action; ties break to the lowest index."""
    def f(obs_vec):
        with torch.no_grad():
            logits, _ = net(obs_vec.unsqueeze(0), train=False, perm=perm)
        return int(torch.argmax(logits, dim=-1).item())
    return f


def _emit_entropy(net: _Agent, task, n: int = 400) -> float:
    """Mean marginal entropy (nats) of the emitted symbol, per internal step.

    Measured on the states the arm actually visits, not on a synthetic sample:
    a channel can look busy on random states and be constant on its own
    trajectory, and it is the trajectory that carries the claim.
    """
    if net.k == 0:
        return 0.0
    counts = [[0] * VOCAB for _ in range(net.k)]
    seen = 0
    def step(st):
        """One forward pass serves both the symbol tally and the action, so
        the entropy is measured on the SAME decisions the arm actually made."""
        nonlocal seen
        with torch.no_grad():
            logits, toks = net(task.obs(st).unsqueeze(0), train=False)
        for k, t in enumerate(toks):
            counts[k][int(t.item())] += 1
        seen += 1
        return int(torch.argmax(logits, dim=-1).item())

    if task.kind == "flat":
        for bx, by, x, y in task.episodes:
            steps = 0
            while steps < FLAT_T and seen < n:
                a = step((bx, by, x, y))
                x, y, _r, done = _flat_sim(bx, by, x, y, a)
                steps += 1
                if done:
                    break
    else:
        spawn = random.Random(f"dp04-ent-{task.seed}-{task.name}")
        while seen < n:
            x, y = spawn.randrange(SIZE), spawn.randrange(SIZE)
            h0 = h1 = 1.0
            steps = 0
            while steps < LIFE_CAP and seen < n:
                a = step((x, y, h0, h1))
                x, y, h0, h1, _r, dead = _sim(task.world, x, y, h0, h1, a)
                steps += 1
                if dead:
                    break
    ents = []
    for row in counts:
        tot = sum(row) or 1
        ents.append(-sum((c / tot) * math.log(c / tot) for c in row if c > 0))
    return sum(ents) / len(ents)


# ── the experiment ───────────────────────────────────────────────────────

_CACHE: dict = {}          # seed -> the trained arms, so _control does not retrain


def _rig(seed: int) -> dict:
    """The instrument, measured before the hypothesis is allowed to speak."""
    base = _VariantWorld(seed, N_FOOD)
    ref = _World(seed)
    identical = float(base.food == ref.food and base.water == ref.water)
    rng = random.Random(f"dp04-probe-{seed}")
    memo: dict = {}
    mism = 0
    for _ in range(N_SCORER_PROBES):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        h0, h1 = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)
        a = _scores_shared(base, x, y, h0, h1, H_ORACLE, memo)
        b = _action_scores(base, x, y, h0, h1, H_ORACLE)
        if max(abs(p - q) for p, q in zip(a, b)) > 1e-12:
            mism += 1
    return {"base_world_identical": identical,
            "scorer_mismatch": float(mism)}


def _fit_all(seed: int) -> dict:
    """Train every arm on every task once; cached so `_control` is free."""
    hit = _CACHE.get(seed)
    if hit is not None:
        return hit
    _CACHE.clear()                        # one seed of nets at a time, by design
    torch.set_num_threads(THREADS)
    tasks = [_Survival(seed, n) for n in RES_COUNTS] + [_Flat(seed)]
    out = {"tasks": {}, "order": [t.name for t in tasks]}
    for task in tasks:
        rng = random.Random(f"dp04-collect-{seed}-{task.name}")
        obs, tgt = task.collect(N_LABEL, rng)
        react, oracle = task.reference()
        entry = {"react": react, "oracle": oracle, "rand": task.random_score(),
                 "floor": task.floor, "ceil": task.ceil,
                 "kind": task.kind, "task": task, "arms": {}}
        for arm in ("verbal", "filler", "mute"):
            fit = _train(task, arm, seed, obs, tgt)
            net = fit["net"]
            entry["arms"][arm] = {
                "score": task.rollout(_act_fn(net), arm),
                "loss_first": fit["loss_first"], "loss_last": fit["loss_last"],
                "net": net}
        # THE CONTROL: verbal's own trained weights, symbols permuted at eval.
        # A DERANGEMENT, constructed rather than hoped for. A uniform
        # `randperm` has a fixed point with probability 1 - 1/e ~ 0.63, so
        # rejecting on the gate would have VOIDed most seeds and a symbol that
        # maps to itself is a symbol the scramble did not scramble. Walking a
        # random single cycle gives a permutation with no fixed point by
        # construction; `ctrl_perm_is_derangement` stays as the guard.
        g = torch.Generator().manual_seed(_seed_of(f"dp04-perm-{seed}-{task.name}"))
        cycle = torch.randperm(VOCAB, generator=g).tolist()
        pl = [0] * VOCAB
        for i, s in enumerate(cycle):
            pl[s] = cycle[(i + 1) % VOCAB]
        perm = torch.tensor(pl)
        vnet = entry["arms"]["verbal"]["net"]
        entry["arms"]["scrambled"] = {
            "score": task.rollout(_act_fn(vnet, perm=perm), "scrambled"),
            "loss_first": entry["arms"]["verbal"]["loss_first"],
            "loss_last": entry["arms"]["verbal"]["loss_last"],
            "net": vnet}
        entry["emit_entropy"] = _emit_entropy(vnet, task)
        entry["perm_is_derangement"] = float(
            all(int(perm[i]) != i for i in range(VOCAB)))
        # The oracle memo has done its work (labels, reference arms); nothing
        # downstream reads it and four of them at MEMO_CAP would be most of
        # this box's declared 1.5 GB ceiling.
        task.memo.clear()
        out["tasks"][task.name] = entry
    _CACHE[seed] = out
    return out


def _norm(entry, score: float) -> float:
    rng = max(entry["ceil"] - entry["floor"], 1e-9)
    return (score - entry["floor"]) / rng


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / max(dx * dy, 1e-12)


def _experiment(seed: int) -> dict:
    t0 = time.time()
    m = _rig(seed)
    fit = _fit_all(seed)
    surv = [n for n in fit["order"] if fit["tasks"][n]["kind"] == "survival"]

    gains, demands, gains_norm = [], [], []
    losses_fell, above_floor = 1.0, 1.0
    headroom = 1e9
    for name in fit["order"]:
        e = fit["tasks"][name]
        v = e["arms"]["verbal"]["score"]
        f = e["arms"]["filler"]["score"]
        gains.append(v - f)
        demands.append(_norm(e, e["oracle"]) - _norm(e, e["react"]))
        gains_norm.append(_norm(e, v) - _norm(e, f))
        m[f"score_verbal_{name}"] = v
        m[f"score_filler_{name}"] = f
        m[f"score_mute_{name}"] = e["arms"]["mute"]["score"]
        m[f"score_react_{name}"] = e["react"]
        m[f"score_rand_{name}"] = e["rand"]
        m[f"score_oracle_{name}"] = e["oracle"]
        m[f"demand_{name}"] = demands[-1]
        m[f"entropy_{name}"] = e["emit_entropy"]
        for arm in ("verbal", "filler", "mute"):
            a = e["arms"][arm]
            if not a["loss_last"] < a["loss_first"]:
                losses_fell = 0.0
            if not a["score"] > e["rand"]:
                above_floor = 0.0
        if e["kind"] == "survival":
            # TWO readings, and the pilot showed why they differ. `oracle - f`
            # is what this gate measured first, and it is the wrong quantity:
            # the H=8 oracle is CENSORED at LIFE_CAP on every survival variant,
            # so that difference reads 0 exactly when the null reaches the cap
            # and says nothing about the true gap. What actually decides whether
            # the statistic CAN move is the distance from the null to the
            # measurement CEILING. Both are recorded; the ceiling one gates.
            headroom = min(headroom, e["ceil"] - f)
            m[f"headroom_to_oracle_{name}"] = e["oracle"] - f

    m["lookahead_gain_over_matched_compute_filler"] = (
        sum(gains[i] for i, n in enumerate(fit["order"]) if n in surv)
        / max(len(surv), 1))
    m["gain_flat"] = gains_norm[fit["order"].index("flat")]
    m["gain_min_survival"] = min(gains[i] for i, n in enumerate(fit["order"])
                                 if n in surv)
    m["demand_spread"] = max(demands) - min(demands)
    m["demand_flat"] = demands[fit["order"].index("flat")]
    # DP.00's own control, in DP.00's own units: steps the H=8 planner saves
    # over greedy where greedy is provably optimal. Gated against DP.00's
    # CTRL_TOL directly rather than against a normalised approximation of it.
    _flat = fit["tasks"]["flat"]
    m["demand_flat_steps"] = _flat["oracle"] - _flat["react"]
    m["above_random_floor"] = above_floor
    m["rho_demand_gain"] = _pearson(demands, gains_norm)
    m["emit_entropy_min"] = min(fit["tasks"][n]["emit_entropy"] for n in surv)
    m["headroom"] = headroom
    m["losses_fell"] = losses_fell
    m["mute_over_floor_min"] = min(
        fit["tasks"][n]["arms"]["mute"]["score"] - fit["tasks"][n]["react"]
        for n in surv)
    m["wall_s"] = time.time() - t0
    return m


def _control(seed: int) -> dict:
    """The SCRAMBLED-VOCABULARY arm, and the MUTE arm's floor reading.

    Both read from `_fit_all`'s cache — the same trained weights the experiment
    scored, never a retrain, so the control is the same network with its symbols
    permuted and nothing else moved.
    """
    fit = _fit_all(seed)
    surv = [n for n in fit["order"] if fit["tasks"][n]["kind"] == "survival"]
    sg = []
    derange = 1.0
    for name in surv:
        e = fit["tasks"][name]
        sg.append(e["arms"]["scrambled"]["score"] - e["arms"]["filler"]["score"])
        derange = min(derange, e["perm_is_derangement"])
    c = {"ctrl_scrambled_gain": sum(sg) / max(len(sg), 1),
         "ctrl_scrambled_gain_max": max(sg),
         "ctrl_perm_is_derangement": derange}
    for name in surv:
        e = fit["tasks"][name]
        c[f"ctrl_score_scrambled_{name}"] = e["arms"]["scrambled"]["score"]
    return c


def _check(m: dict, c: dict):
    # ── the instrument, before the hypothesis ────────────────────────────
    if m.get("base_world_identical", 0.0) != 1.0:
        return Status.VOID          # the base variant is not DP.00's world
    if m.get("scorer_mismatch", 1.0) != 0.0:
        return Status.VOID          # the shared memo changed the teacher
    if m.get("losses_fell", 0.0) != 1.0 or m.get("above_random_floor", 0.0) != 1.0:
        return Status.VOID          # an arm never learned; the compare is dead
    if m.get("emit_entropy_min", 0.0) < ENT_MIN:
        return Status.VOID          # the emission collapsed: verbal IS filler
    if m.get("headroom", 0.0) < HEADROOM_MIN:
        return Status.VOID          # the null already matches the teacher
    if m.get("demand_spread", 0.0) < SPREAD_MIN:
        return Status.VOID          # there is no dose axis to respond to
    if m.get("demand_flat_steps", 1e9) > CTRL_TOL:
        return Status.VOID          # DP.00's zero-demand world is not zero here
    if c.get("ctrl_perm_is_derangement", 0.0) != 1.0:
        return Status.VOID          # a permutation with fixed points is weaker

    # ── the hypothesis ───────────────────────────────────────────────────
    gain = m.get("lookahead_gain_over_matched_compute_filler", 0.0)
    sigma = gain * math.sqrt(2.0) / max(
        m.get("lookahead_gain_over_matched_compute_filler_std", 0.0), 1e-9)
    ok = (gain >= MIN_GAIN
          and sigma >= SIGMA_GATE
          # (b) the control must not help
          and c.get("ctrl_scrambled_gain", 1e9) <= max(SCRAM_FRAC * gain, 0.0)
          and c.get("ctrl_scrambled_gain", 1e9) <= SCRAM_ABS
          # (c) zero demand, zero gain
          and abs(m.get("gain_flat", 1e9)) <= FLAT_TOL
          # (d) dose-response across the five task points
          and m.get("rho_demand_gain", -1.0) >= RHO_MIN
          # (e) the mute arm must still deliberate
          and m.get("mute_over_floor_min", -1e9) >= MUTE_FLOOR_MIN)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "DP.04 gates are PROVISIONAL. Run the pilot "
            "(`python -m experiments.tests.dp_04_slow_path_verbal pilot`), read "
            f"{_PILOT_ARTIFACT % 90} and {_PILOT_ARTIFACT % 91}, freeze "
            "ENT_MIN / HEADROOM_MIN / SPREAD_MIN / MIN_GAIN / SCRAM_FRAC / "
            "SCRAM_ABS / FLAT_TOL / RHO_MIN / MUTE_FLOOR_MIN against the "
            "measured table in a commit that pastes the table into this "
            "docstring, then set _GATES_FROZEN = True. A gate fitted to the "
            "run it judges is not a gate.")
    return run_spec(BY_ID["DP.04"], _experiment, _check, control_fn=_control,
                    ledger=ledger or Ledger())


# ── smoke and pilot ──────────────────────────────────────────────────────

def _shrink(**kw):
    """Temporarily shrink the envelope; returns the previous values."""
    old = {k: globals()[k] for k in kw}
    globals().update(kw)
    return old


def _smoke():
    """Tiny envelope, every entry point once — including `_check`, which is
    where a shape error would otherwise wait for the registered run."""
    old = _shrink(N_LABEL=300, EPOCHS=2, N_EVAL_LIVES=3, RES_COUNTS=(3, 8),
                  N_SCORER_PROBES=20, FLAT_EPISODES=4)
    try:
        _CACHE.clear()
        t0 = time.time()
        m = _experiment(0)
        c = _control(0)
        print("smoke experiment:", json.dumps(m, indent=1, default=float))
        print("smoke control:", json.dumps(c, indent=1, default=float))
        print("smoke check path:", _check(
            {**m, "lookahead_gain_over_matched_compute_filler_std": 1.0}, c))
        print("smoke wall_s: %.1f" % (time.time() - t0))
    finally:
        globals().update(old)
        _CACHE.clear()


def _pilot():
    """Seeds 90 and 91 — disjoint from the registered 0/1/2, and SPENT once
    used. Full envelope, JSON to stdout AND to `_PILOT_ARTIFACT`. No ledger
    row is written and none may be."""
    for seed in _PILOT_SEEDS:
        _CACHE.clear()
        t0 = time.time()
        m = _experiment(seed)
        c = _control(seed)
        out = {"seed": seed, "experiment": m, "control": c,
               "constants": {"RES_COUNTS": list(RES_COUNTS), "N_LABEL": N_LABEL,
                             "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
                             "HID": HID, "VOCAB": VOCAB, "EMB": EMB,
                             "K_STEPS": K_STEPS, "TAU": TAU,
                             "N_EVAL_LIVES": N_EVAL_LIVES,
                             "H_ORACLE": H_ORACLE, "LIFE_CAP": LIFE_CAP},
               "pilot_wall_s": time.time() - t0}
        txt = json.dumps(out, default=float, indent=1)
        try:
            with open(_PILOT_ARTIFACT % seed, "w") as fh:
                fh.write(txt)
        except OSError as exc:                       # pragma: no cover
            print(f"WARN: could not write {_PILOT_ARTIFACT % seed}: {exc}")
        print(txt, flush=True)


def _median(v: list) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _stdev(v: list) -> float:
    n = len(v)
    if n < 2:
        return 0.0
    mu = sum(v) / n
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))


_SIZE_B = 2000            # bootstrap draws per candidate design


def _size_derive(out: dict) -> dict:
    """Turn the raw spans into the sizing arithmetic. Pure post-processing.

    THE TARGET IS NOT A CHOICE. `_check` computes
    `sigma = gain * sqrt(2) / std` and requires `gain >= MIN_GAIN` and
    `sigma >= SIGMA_GATE`, so a MINIMALLY-sized true effect clears only if the
    per-seed gain's standard deviation is at most
    `MIN_GAIN * sqrt(2) / SIGMA_GATE`. That number is derived from two bars
    that do not move; sizing chooses the counts that reach it, never the bar.

    WHAT THIS MEASURES AND WHAT IT CANNOT. The world is HELD FIXED at one
    seed, so every design's spread here is the REDUCIBLE component — training
    initialisation, propagated through the median-of-R rule and the eval
    count. The registered run's `std` is across three DIFFERENT worlds and
    also carries a world-to-world component that no restart count can remove.
    So a design that meets the target here is NECESSARY, not sufficient, and
    the pilot on 92/93 is what tests it.
    """
    target = MIN_GAIN * math.sqrt(2.0) / SIGMA_GATE
    names = list(out["tasks"])
    d = {"target_gain_std": target, "per_arm": {}, "designs": [],
         "target_derivation": "MIN_GAIN * sqrt(2) / SIGMA_GATE",
         "reducible_only": True}

    for n in names:
        e = out["tasks"][n]
        for arm in ("verbal", "filler"):
            runs = e["arms"][arm]
            spans = [s for r in runs for s in r["spans"]]
            row = {"span_min": min(spans), "span_max": max(spans),
                   "span_mean": sum(spans) / len(spans),
                   "losses_fell_all": float(all(r["loss_last"] < r["loss_first"]
                                                for r in runs))}
            for cap in _SIZE_CAPS:
                row[f"sat_frac_{cap}"] = (
                    sum(1 for s in spans if s >= cap) / len(spans))
                for E in _SIZE_ES:
                    sc = [sum(min(s, cap) for s in r["spans"][:E]) / E
                          for r in runs]
                    row[f"score_cap{cap}_E{E}_mean"] = sum(sc) / len(sc)
                    # THE HEADLINE OF REPAIR (b): one arm, run repeatedly,
                    # its sigma. Everything below is arithmetic on this.
                    row[f"score_cap{cap}_E{E}_sd_restart"] = _stdev(sc)
            d["per_arm"][f"{n}/{arm}"] = row

    for cap in _SIZE_CAPS:
        for E in _SIZE_ES:
            sc = {(n, a): [sum(min(s, cap) for s in r["spans"][:E]) / E
                           for r in out["tasks"][n]["arms"][a]]
                  for n in names for a in ("verbal", "filler")}
            for R in _SIZE_RS:
                # Non-parametric: resample restarts with replacement and apply
                # repair (c)'s median-of-R exactly as the envelope would. No
                # normality assumed — 8 restarts is too few to assume it.
                bs = random.Random(f"dp04-size-bs-{cap}-{E}-{R}")
                draws = []
                for _ in range(_SIZE_B):
                    tot = 0.0
                    for n in names:
                        mv = _median([bs.choice(sc[(n, "verbal")])
                                      for _ in range(R)])
                        mf = _median([bs.choice(sc[(n, "filler")])
                                      for _ in range(R)])
                        tot += mv - mf
                    draws.append(tot / len(names))
                sd = _stdev(draws)
                d["designs"].append({
                    "cap": cap, "E": E, "R": R,
                    "gain_mean": sum(draws) / len(draws),
                    "gain_sd_reducible": sd,
                    "meets_target": bool(sd <= target),
                    # trainings per seed = R * len(RES_COUNTS+flat) * n_arms
                    "trainings_rel": R})
    ok = [x for x in d["designs"] if x["meets_target"]]
    d["cheapest_meeting_target"] = (
        min(ok, key=lambda x: (x["R"], x["E"], x["cap"])) if ok else None)
    return d


def _size():
    """SIZING RUN — repairs (b) and (c) of PILOT RECORD v1, pre-registered.

    Seed 94, world HELD FIXED, `LIFE_CAP` raised to `_SIZE_CAP` so repair (a)
    is MEASURED rather than assumed. Raw lifespans are recorded, never means:
    a life censored at 400 is also a life censored at 200, so one run reports
    both ceilings, and the spawn key is fixed so the first 12 lives are the
    lives the registered envelope scores. Nothing here touches a claim bar and
    no ledger row is written.

    The four survival variants only. `flat` is excluded because it does not
    enter `lookahead_gain_over_matched_compute_filler`, which is the statistic
    being sized; its own gate (`demand_flat_steps`) passed on both pilot seeds.
    """
    torch.set_num_threads(THREADS)
    old = _shrink(LIFE_CAP=_SIZE_CAP)
    try:
        t0 = time.time()
        out = {"seed": _SIZE_SEED, "R": _SIZE_R, "E": _SIZE_E,
               "cap_run": _SIZE_CAP, "N_LABEL": N_LABEL, "EPOCHS": EPOCHS,
               "H_ORACLE": H_ORACLE, "N_EVAL_LIVES_registered": N_EVAL_LIVES,
               "tasks": {}}
        for n_res in RES_COUNTS:
            task = _Survival(_SIZE_SEED, n_res)
            rng = random.Random(f"dp04-collect-{_SIZE_SEED}-{task.name}")
            obs, tgt = task.collect(N_LABEL, rng)
            tr = time.time()
            react, oracle = task.reference()
            entry = {"ref_wall_s": time.time() - tr, "react": react,
                     "oracle": oracle, "rand": task.random_score(),
                     "oracle_at_cap": float(oracle >= _SIZE_CAP), "arms": {}}
            for arm in ("verbal", "filler"):
                runs = []
                for r in range(_SIZE_R):
                    fit = _train(task, arm, _SIZE_SEED, obs, tgt, restart=r)
                    runs.append({
                        "restart": r,
                        "spans": task.rollout_spans(_act_fn(fit["net"]),
                                                    _SIZE_E),
                        "loss_first": fit["loss_first"],
                        "loss_last": fit["loss_last"]})
                    print("  %s/%s r%d done (%.0f s)"
                          % (task.name, arm, r, time.time() - t0), flush=True)
                entry["arms"][arm] = runs
            task.memo.clear()
            out["tasks"][task.name] = entry
        out["sizing_wall_s"] = time.time() - t0
        out["derived"] = _size_derive(out)
        txt = json.dumps(out, default=float, indent=1)
        try:
            with open(_SIZE_ARTIFACT, "w") as fh:
                fh.write(txt)
        except OSError as exc:                       # pragma: no cover
            print(f"WARN: could not write {_SIZE_ARTIFACT}: {exc}")
        print(json.dumps(out["derived"], default=float, indent=1), flush=True)
        print("sizing_wall_s %.1f" % out["sizing_wall_s"], flush=True)
    finally:
        globals().update(old)
        _CACHE.clear()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        _pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "size":
        _size()
    else:
        print(run().status)
