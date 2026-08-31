"""LC.03 — Screening: which learning cores learn to survive at all.

THE QUESTION. LC.00 proved the metric decidable in a gridworld, LC.01 admitted
five arms on unison grounds, LC.02 fixed each arm's train_ratio on wall-clock
grounds. This spec is the first time the admitted cores LIVE: lethal W0, real
deaths, random respawns, the learner in the loop — and the claim is only that
at least two of them learn to survive AT ALL. Screening declares no winner
(LC.04's job); it can only exclude, and its VOID ("fewer than two learners")
blocks the decision instead of manufacturing one.

THE ENVELOPE (LEARNING_CORE.md §5.7, fixed HERE for the ledger; v2 sizes,
2026-08-21 — see V2 RE-SCREEN below; v1 was 100k / 4,320 and recorded VOID
data-starved). Each arm-seed runs N_STEPS = 400,000 decisions (= 80,000
sim-s) at e0 = 1.0, and keeps living until it has ALSO consumed W_CLOCK =
17,280 core-seconds (4.8 core-h) — whichever comes later — so LC.04 (matched
experience) and LC.05 (matched compute) can score the SAME stored curves. Per-life resource coordinates and
the decimated span curve are written to
`experiments/artifacts/lc03_curves_seed{N}.json`; LC.04/LC.05 read that file
and run nothing. Untrained twins and the designed-to-fail controls run at
HALF_STEPS = 200,000 (§5.7's "controls at half budget"): their quantities are
stationary points, not curves. Wiped twins run the FULL envelope — they are
the S3 pairing, and a paired difference needs matched windows.

THE GATES, from the registry entry verbatim, encoded per seed where it says
per seed and on the house 3-sigma idiom (mean * sqrt(3) / std, the paired
margin t-statistic T2.08 standardised) where it says sigma:

  per arm, ALL of:
    lg_margin_null   life_gain - random null's life_gain, paired per seed;
                     t >= 3.0
    lg_margin_twin   life_gain - own UNTRAINED twin's, paired per seed; t >= 3.0
    n_lives >= 12    every seed
    needs_satisfied_rate rising (final third > first third, on the seed mean)
    cross_life_transfer > 0 (seed mean; = life_gain - wiped twin's life_gain)
    panel_dwell <= 0.15 every seed, else DISQUALIFIED — PG.4's own control
                     threshold, its exact dwell definition (fraction of the
                     LATE HALF of decisions within DWELL_RADIUS = 2.0 m of the
                     noise panel), ported not paraphrased
    chaos_occupancy >= 3.0 AND chaos_reward_ratio >= 2.0 in any seed => that
                     arm is VOID (CURIOSITY_BAKEOFF.md §2.10, ported below)
    wm-efe only: final-third action entropy (std of the PRE-NOISE policy
                     means) < 10% of dreamer-xs's => VOID for wm-efe
                     (arXiv:2303.01618's epistemic collapse)

  PASS iff >= 2 arms clear everything. Fewer => Status.VOID.

FOUR CONTROLS AND ONE WORLD-TRIPWIRE, each with a pre-registered side (a
wrong side is VOID — a control landing wrong means the instrument, not the
hypothesis, failed). Four falsifiers, not five: after the 2026-08-13
amendment, (e) can no longer fail by the mechanism the amendment itself
documents — basal drain 0.00167/s < active 0.0022/s means passivity wins
life LENGTH structurally, not by draw, so "not strongly negative" is the
world's arithmetic, not a test of the arms. (e) is kept as a TRIPWIRE: it
fires only if W0 ever starts punishing anti-curiosity, which means the rig
must be re-derived. A reader counting five falsifiers is counting one that
is not there. And to be explicit about what DOES exclude learned passivity
in the claim conjunction: not the dwell/chaos gates — a statue scores
perfectly on panel_dwell and every chaos_* signal — but `needs_rise > 0`,
the conjunct every pilot arm failed and the one the registered envelope is
betting 8.3x more experience will flip.
  (a) statue      AMENDED 2026-08-13 (see PILOT RESOLUTION): must RIDE THE
                  BASAL CEILING, |mean_life - e0/BASAL_B| <= 10% — the
                  passive path is clean; nothing but basal starvation may
                  kill a body that never acts. life_gain reported, ungated
                  (zero by construction — the saturated-quantity lesson).
  (b) randrew     fixed random stationary reward projection on the ppo-needs
                  core: must MISS the 3-sigma null gate
  (c) frozen      every untrained twin's |life_gain| within noise of zero
                  (t < 3.0 OR |mean| <= NOISE_FLOOR_S)
  (d) wiped-store every wiped twin's |life_gain| within noise of zero — the
                  2026-08-13 registry amendment: no admitted core retrieves
                  the diary, so the permuted-diary form could never fail for
                  the right reason (T0.13); the cross-life store the torch
                  arms use is the learner state, and wiping it is the
                  corruption the control needs
  (e) darkroom    dreamer-xs rewarded with MINUS its own posterior entropy:
                  AMENDED 2026-08-13 (see PILOT RESOLUTION): must NOT be
                  strongly negative (margin vs the half-budget null,
                  t > -3.0) — the measured inversion locked in as the
                  executable record that life_gain carries LEARNING, not
                  curiosity's sign.

PILOT RESOLUTION, 2026-08-13 (seed 90, 12k decisions, e0=0.3, 7667 s —
numbers recorded verbatim; both suspect sides were measured INVERTED and
both controls were amended in the registry and here IN THE SAME COMMIT,
BEFORE the registered run; T1.02 precedent, the old sides stay in git
history). The caution this block replaces predicted it: PS.01's basal drain
0.00167/s against active 0.0022/s means PASSIVITY maximises life LENGTH in
W0. Measured: statue mean life 180.0 s = e0/BASAL_B (0.3 x 600 s) to 0.02%
— the basal-starvation ceiling, an arithmetic fact, LONGEST of every run in
the pilot (arms 109.0-161.5 s, nulls 118.3/126.2 s, darkroom 183.5 s). The
darkroom LEARNED PASSIVITY and prospered on the length ruler: life_gain
+23.6, margin +49.7 s over its paired null. So (a) 'dies soonest' and (e)
'strongly negative' were unattainable as written — T2.08's passivity
inversion, as flagged. What the pilot ALSO showed, and the claim gates keep
unmoved: four of five arms cleared their null margins on life_gain (ppo-lp
+54.6, wm-efe +52.0, wm-latent +47.7, dreamer-xs +45.7; ppo-needs -1.8),
every chaos and dwell gate read clean, and every arm's needs_rise was
NEGATIVE at the compressed pilot envelope — the registered envelope
(N_STEPS=100k, e0=1.0) is 8.3x longer and the claim conjunction stands as
pre-registered. Gates MUST NOT MOVE after registration; this commit IS the
registration.

RIG RE-DERIVATION, 2026-08-20 (attempt 1, 2026-08-14, VOID — control (c)
fired: every untrained twin's life_gain read +158..190 s with per-arm seed
spread as tight as +/-2.0 s, so a non-learner moved the ruler and the run
did not test the claim). THE MECHANISM, found in the rig, not the world:
run_survival's exploration std decays linearly over the RUN (EXPLORE_STD
0.5 -> 0.1, frac = k/n_decisions) and drives.py prices actuator power at up
to 3x basal drain — so every policy="core" run gets structurally quieter,
and therefore longer-lived, as the run proceeds, learning or not. The
controls that SHOULD be stationary confirmed it in the same run: random
null +6.5 +/- 12.0 (constant full-range actions — no trend), statue +0.013.
The confound also contaminated the CLAIM gate: lg_margin_null compared a
decaying-activity arm against a constant-activity null, so part of every
arm's pilot "learning" was the schedule riding the passivity arithmetic.
THE REPAIR (strengthen-only; removing an inflator makes the claim harder):
every policy="core" run — arms, wiped twins, frozen twins, and the randrew
and darkroom controls (whose margins the schedule also inflated: darkroom
read +162 against its stationary null in the VOID run) — now passes
EXPLORE_STD_LC03 = (0.3, 0.3), a CONSTANT std at the old schedule's
time-mean, so exploration total is matched and the policy process is
time-stationary for non-learners. explore_std is a run_survival parameter,
so no other spec's certificate is touched. Gates UNMOVED (SIGMA_GATE 3.0,
NOISE_FLOOR_S 5.0, the claim conjunction verbatim). Side effects accepted
and declared: arms lose annealing (all arms equally; the twin pairing is
what the gates lean on), and the half-budget twin's schedule-timescale
mismatch (its frac reached 0.1 twice as fast) disappears outright.
PRE-REGISTERED CHECK before any relaunch (experiments/lc03_twin_check.py,
bars in its docstring, set before launch): the dreamer-xs twin at the pilot
envelope must REPRODUCE the spurious gain under the old schedule
(life_gain >= +20 s) and the constant std must KILL it (|life_gain| <=
10 s). Reproduce-fails => diagnosis wrong, stop. Fix-fails => a second
nonstationarity exists, find it first. Both-pass => relaunch the registered
run; a second VOID for the schedule reason is then impossible by
construction.

CHECK RESOLUTION, 2026-08-20 (~10:3x UTC; JSONs /data/lc03_twin_check.json
and /data/lc03_food_probe.json — numbers verbatim). REPRODUCE passed:
+112.35 s under the old schedule (bar >= +20). The raw fix bar FAILED:
constant std read +17.90 s against |gain| <= 10 s, so the fix-fails branch
ran. FOUND, by a bit-identical instrumented replay
(experiments/lc03_food_probe.py, readings pre-stated in its docstring
before launch, replay verified span-identical to the checked run): the
ENTIRE residual is TWO obj1 floor-food eats (nu = 0.08 = +48 s of
basal-equivalent life each) landing by draw in lives 11-12 of 14 — the
only eats in the whole 12k-decision run. Food-corrected life_gain
(span - eaten/BASAL_B) reads -6.1 s, INSIDE the bar; apple z constant at
1.89 m all run, so the platform-apple accessibility ratchet is refuted.
VERDICT: there is no second nonstationarity — the policy process under
constant std is time-stationary, and the check's 10 s bar was smaller than
ONE quantum of the world's food channel on a 14-life ruler (a single eat
moves a 4-life third-mean by +12 s). The constant-std repair STANDS and
the registered relaunch proceeds. Food shot noise remains the failable
territory of controls (c)/(d), whose two-sided branch (|t| >= 3 AND
|mean| > NOISE_FLOOR_S) is sized for symmetric quanta at the registered
envelope; run_survival now exports ate_total / eats_at_death so any future
twin life_gain anomaly is attributable in one read instead of a 385 s
replay probe.

V2 RE-SCREEN, 2026-08-21 (this commit is the registration; gates UNMOVED,
only the envelope grows). CORRECTED DIAGNOSIS of attempt 2 (v1 envelope,
100k / 4,320, recorded 2026-08-21T02:11), found by replaying _check
against the row's own recorded metrics before this redesign: the VOID did
NOT fire at "fewer than two learners" as the 08-21 harvest and commit
eec7d86 narrated — it fired at CONTROL (c), first in the loop:
ppo-needs/twin_life_gain -7.71 s, t = -3.16 vs the 3.0 gate, |mean| 7.71
vs NOISE_FLOOR_S 5.0. The claim loop never ran. The harvest read a 10 s
floor that does not exist (it is 5.0) and back-filled the generic VOID
message with the branch it expected. The magnitude is ONE FOOD QUANTUM:
at the v1 twin envelope (HALF_STEPS 50k, ~22 lives, thirds of 7) a single
obj1 eat moves a third-mean by ~48/7 = 6.9 s — the exact "sized for
symmetric quanta" assumption the CHECK RESOLUTION above flagged as the
failable territory of (c)/(d), failing. A one-eat draw in a frozen twin
is not a ruler drift; but the gate was right to fire — the bar was finer
than the channel's quantum at that envelope, and a gate that cannot tell
a draw from a drift did not test the claim.
  What is ALSO true, from the same recorded metrics (the claim gates,
evaluated offline): zero arms at 3 sigma — best margins wm-efe
lg_margin_null +74.5 s (t=1.25, seed std ~103) and dreamer-xs +44.1 s
(t=0.49, std ~156) — and final-half life-span slopes POSITIVE on 4 of 5
arms (per-seed: wm-efe 10.02/11.93/5.11 s-per-life, dreamer-xs
2.95/10.61/5.68, ppo-needs 15.64/7.01/0.50, ppo-lp 17.15/-1.84/1.96;
wm-latent -13.00/5.50/-7.33). Still climbing at cutoff: the owner's
data-starved branch (2026-08-09) applies on the evidence even though the
recorded VOID formally fired upstream of it — re-screen bigger, do not
eliminate. ONE envelope growth answers BOTH faults: more lives per twin
takes the food quantum back under the floor (v2 twin: 200k steps, ~88
lives, thirds of ~29, one eat = ~1.7 s < 5.0 — and real drift resolves
BETTER, so this strengthens (c), never loosens it), and more lives per
arm gives the climbing margins room to clear 3 sigma or flatten out. SIZING, from the recorded curves and nothing else: at the
measured seed stds, t >= 3 needs a margin of ~179 s for wm-efe and ~270 s
for dreamer-xs, and the claim needs TWO learners, so the envelope is sized
to the SECOND arm. A 1x envelope yields ~50 lives; assume an arm holds its
WEAKEST per-seed slope for half the added lives (slope decay priced in):
k=4 gives dreamer-xs 2.95 * 150/2 = +221 s ~= its +226 s requirement, and
wm-efe 5.11 * 75 = +383 s >> its +105 s. k=2 sufficed only for wm-efe —
one learner cannot pass this gate. Hence 4x: N_STEPS 400,000, W_CLOCK
17,280, HALF_STEPS follows. Unpriced and in the claim's favour: seeds
converging toward the span ceiling shrink the std, raising t faster than
the margin model. needs_rise, the conjunct every pilot arm failed, moved
from all-negative (pilot) to -0.011..+0.010 (v1 registered) — the bet that
more experience flips it continues on trend. Cost: ~4x attempt 2's 47
worker core-h = ~190 core-h, ~63 h wall on 3 nice-19 workers, zero GPU.
TWO GAPS CLOSED in the same commit, both machine-readable, not prose:
`{arm}/data_starved`, promised below since registration and computed
nowhere (the 08-21 harvest found grep hits only in this docstring), is now
computed in _check and recorded in the ledger row; and `void_reason` — the
generic "run did not test the claim" message is what let the harvest
mis-attribute the firing branch, so _check now writes WHICH gate voided
into the metrics it returns through. A VOID that names its branch cannot
be back-filled with the story the reader expected.

PORTS, not paraphrases:
  * panel_dwell — PG.4's `_dwell` (strict < DWELL_RADIUS 2.0 m) over the
    late half, threshold 0.15 = PG.4's CONTROL_DWELL_MAX. The harness's
    `panel_near_frac` (1.5 m, whole run) stays a diagnostic.
  * chaos detector — CURIOSITY_BAKEOFF.md §2.10 in full: one pooled forward
    model per seed over the late-half transitions of every arm plus the
    random null, fit twice (half/full data), 5-fold out-of-fold errors,
    theta = the null's 90th percentile of e_full, CHAOTIC iff e_full >= theta
    AND reducibility < 0.1, occupancy / 0.10, reward ratio of means clipped
    at 1e3, thrash_l1 / null's as the model-free second signal. One declared
    deviation: transitions are subsampled every TRACE_EVERY = 8 decisions
    (identically for every run) — §2.10 was written for 50k-decision runs and
    a full 100k+ trace does not fit the box; a uniform subsample is a rig
    constant, not a threshold.
  * ppo-lp's intrinsic — LC.00's `q_lp` (Oudeyer-style absolute learning
    progress, LP_BETA 0.5 * |delta EMA(prediction error)|, EMA lambda 0.1)
    lifted to continuous outcomes over a SAGG-RIAC-style auto-partitioned
    outcome space (median split on the max-variance dim every LP_SPLIT_N
    region visits, cap LP_MAX_REGIONS). Outcome = (torso xyz, z if airborne,
    climb-contact count, horizontal speed) — CURIOSITY_BAKEOFF's lp goal
    space minus nearest-object displacement, which W0 does not expose.
    Hindsight is implicit: regions score REACHED outcomes. The channel is
    r_lp with its own GAE pass into the critic_lp head; advantages add with
    unit weights (PURPOSE_AND_SCAFFOLDING.md §2.8 option 2; satiety gating
    off per F8).
  * wm-efe's actor objective — LC.00's `model_efe` form: r = r_h + KAPPA_EFE
    * (ensemble disagreement / its running mean), ln C = -d(h) so the
    pragmatic term IS drive reduction. KAPPA_EFE = 0.05, LC.00's kappa
    scale. DECLARED COST DEVIATION: the wm-efe and darkroom reward functions
    each spend one extra uncertainty forward per decision that LC.02 did not
    time; the W_CLOCK axis charges it honestly.

DATA-STARVED GUARD (owner, 2026-08-09, DECISIONS_NEEDED): an arm that fails
the gate WITH a positive final-half life-span slope is reported
`{arm}/data_starved = 1.0` — re-screen at a bigger envelope, do not
eliminate. The convergence and scale-transfer guards are LC.04's.

RUNNING IT. The v2 registered run is ~190 core-hours (~63 h wall) and MUST
be detached, via the helper that proves the launch survived its imports:

    scripts/launch_detached.sh /data/lc03_rescreen.log \
        /data/venvs/jackthelearner/bin/python -m \
        experiments.tests.lc_03_survival_screening

It parallelises the three seeds over 3 single-threaded workers (the memoised
run_spec pattern T2.01 established) and writes the ledger itself on
completion. `pilot` runs seed 90 at the compressed envelope (~2 h), `smoke`
is a minutes-long mechanics check that records nothing.

sb3-ppo, LC.04's ineligible reference arm, is NOT run here: it is not
LC.01-admitted and the registry scopes LC.03 to admissible arms. LC.04's
implementer must either push it through this same harness at this same
envelope (one extra run per seed) or record why not.

VOID-FORECLOSED: the v2 re-screen's own pre-registered fork (ii) fired —
    `fewer than two learners (1 cleared)` — after 400k decisions/arm-seed and
    ~190 core-hours at the 4x envelope. Every control landed on its
    pre-registered side, so the CLAIM loop fired and the rig measured. The
    fork priced growth explicitly ("the requirement scales with added lives
    just as the projected gain does"), so a v3 is the ratchet the fork exists
    to prevent. The repair is a REDESIGN of the screen or of W0, on the
    owner's desk since 2026-08-24 (`docs/DECISIONS_NEEDED.md`, D10).

CONCLUDED — v2, attempt 3, 2026-08-23T21:11 UTC, commit `0d9ad54`; harvested
and replayed offline 2026-08-24. Per-arm t_null / t_twin: wm-latent 4.65 / 4.00
(the sole clean learner, every conjunct green), wm-efe 2.05 / 2.07, ppo-needs
1.06 / 0.99, ppo-lp 1.20 / 1.10 with `needs_rise` NEGATIVE, dreamer-xs -0.94 /
-0.99. Controls: statue 599.92 s on the 600 s basal ceiling, randrew t 0.21,
darkroom t -1.08, zero twin/wiped trips. Three `data_starved` flags do not
reopen it, for the reason the fork gives above.

THE FINDING, which is about the world and not about the arms: W0 does not
discriminate these five learning cores at a reachable envelope — one learns to
survive in it, four do not. Recorded here because this file is where the next
reader arrives; the decision it feeds (LC.04's premise that arbitration needs
>= 2 learners) is the owner's, not a dispatch. Curves for all arms are in
`experiments/artifacts/lc03_curves_seed{0,1,2}.json` (gitignored, on this box)
— LC.04/LC.05 were designed to read them and run nothing, which matters to any
redesign discussion.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..cores import ACTION_DIM, CANDIDATE_ARMS
from ..drives import BASAL_B
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..survival import run_survival
from ..w0 import W0
from .lc_02_throughput_floor import TRAIN_RATIOS, committed_ratio

IMPL_DEPS = ["experiments/survival.py", "experiments/cores.py",
             "experiments/w0.py", "experiments/drives.py", "playground.py"]

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "experiments" / "artifacts"

# ── THE ENVELOPE (LEARNING_CORE.md §5.7) — fixed for the registered run ─────
N_STEPS = 400_000               # decisions per arm-seed (80,000 sim-s).
                                # v2 (2026-08-21): 4x v1's 100k, sized in the
                                # V2 RE-SCREEN docstring block from attempt
                                # 2's recorded slopes — the second learner
                                # (dreamer-xs) binds. v1 recorded VOID
                                # data-starved; gates unmoved.
W_CLOCK_CORE_S = 17_280.0       # 4.8 core-h (4x v1); whichever is LATER
HALF_STEPS = N_STEPS // 2       # twins + designed-to-fail controls
E0 = 1.0                        # LC.03's regime (XL.00 compressed with 0.1)
TRACE_EVERY = 8                 # transition subsample for the chaos detector
EXPLORE_STD_LC03 = (0.3, 0.3)   # RIG RE-DERIVATION 2026-08-20: CONSTANT std
                                # (old schedule's time-mean) for every core
                                # run — the decaying default made every core
                                # policy quieter-and-longer-lived over the
                                # run, learning or not (see the docstring)

# ── THE GATES ───────────────────────────────────────────────────────────────
SIGMA_GATE = 3.0                # the house learning gate (T2.02's invention)
N_LIVES_MIN = 12                # per seed, from the registry hypothesis
NOISE_FLOOR_S = 5.0             # 'within noise of zero' absolute clause: a
                                # twin whose seed spread collapses cannot be
                                # tripped by a 2-second effect (~0.3% of a life)
PANEL_DWELL_MAX = 0.15          # PG.4's CONTROL_DWELL_MAX — the ported gate
DWELL_RADIUS = 2.0              # PG.4's dwell zone, verbatim
CHAOS_OCC_VOID = 3.0            # §2.10 conjunction ...
CHAOS_RATIO_VOID = 2.0          # ... occupancy AND ratio
CHAOS_RED_EPS = 0.1             # reducibility threshold, §2.10
CHAOS_NULL_PCT = 90             # theta = null's 90th percentile of e_full
CHAOS_RATIO_CLIP = 1e3          # §2.10: an unbounded ratio cannot be gated
EFE_ENTROPY_FRAC = 0.10         # wm-efe collapse gate vs dreamer-xs

# ── THE PORTED MECHANISM CONSTANTS ─────────────────────────────────────────
LP_LAMBDA = 0.1                 # LC.00's q_lp EMA lambda
LP_BETA = 0.5                   # LC.00's q_lp bonus scale
LP_SPLIT_N = 400                # SAGG-RIAC: split a region every N visits
LP_MAX_REGIONS = 64
LP_LR = 1e-3                    # the outcome forward model's Adam lr
KAPPA_EFE = 0.05                # LC.00's model_efe kappa scale
CHAOS_HIDDEN = 256              # §2.10: 2x256 MLP, Adam 1e-3
CHAOS_EPOCHS = 3
CHAOS_BATCH = 256

PILOT_SEED = 90                 # disjoint from registered seeds 0/1/2
PILOT_STEPS = 12_000
PILOT_E0 = 0.3


# ── borrowed calibration ────────────────────────────────────────────────────
def _borrow():
    """PS.01's j0/alpha and LC.02's committed train_ratios, or a refusal."""
    b1 = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b1.ok:
        return None, dict(b1.provenance, refusal=b1.refusal)
    keys = [f"{a}/clears@{r}" for a in CANDIDATE_ARMS for r in TRAIN_RATIOS]
    b2 = borrow_metrics("LC.02", keys)
    if not b2.ok:
        return None, dict(b2.provenance, refusal=b2.refusal)
    ratios = {a: committed_ratio(b2.values, a) for a in CANDIDATE_ARMS}
    if any(v is None for v in ratios.values()):
        return None, dict(b2.provenance,
                          refusal="LC.02 committed no ratio for some arm")
    return {"j0": b1.values["j0_ms"], "alpha": b1.values["alpha"],
            "ratios": ratios}, dict(b2.provenance)


# ── the ported reward machinery ────────────────────────────────────────────
class _LPRegions:
    """SAGG-RIAC-style recursive partition of the z-scored outcome space."""

    def __init__(self):
        dim = 6
        self.regions: List[dict] = [{
            "lo": np.full(dim, -np.inf), "hi": np.full(dim, np.inf),
            "n": 0, "ema": 1.0, "buf": []}]

    def find(self, oz: np.ndarray) -> dict:
        for reg in self.regions:
            if np.all(oz >= reg["lo"]) and np.all(oz < reg["hi"]):
                return reg
        return self.regions[0]           # numeric edge: fall back to the root

    def update(self, oz: np.ndarray, err: float) -> float:
        """EMA update in oz's region; returns |delta EMA| (the LP signal)."""
        reg = self.find(oz)
        prev = reg["ema"]
        reg["ema"] = prev + LP_LAMBDA * (err - prev)
        reg["n"] += 1
        reg["buf"].append(oz)
        if len(reg["buf"]) > 512:
            reg["buf"] = reg["buf"][-512:]
        if (reg["n"] % LP_SPLIT_N == 0 and len(self.regions) < LP_MAX_REGIONS
                and len(reg["buf"]) >= 32):
            pts = np.stack(reg["buf"])
            d = int(pts.var(axis=0).argmax())
            cut = float(np.median(pts[:, d]))
            if reg["lo"][d] < cut < reg["hi"][d]:
                a = dict(reg, lo=reg["lo"].copy(), hi=reg["hi"].copy(),
                         n=0, buf=[])
                b = dict(reg, lo=reg["lo"].copy(), hi=reg["hi"].copy(),
                         n=0, buf=[])
                a["hi"][d] = cut
                b["lo"][d] = cut
                # remove by IDENTITY: list.remove scans with ==, and dict
                # equality on numpy-array values raises the moment the split
                # region is not at index 0 (the smoke only ever split the root).
                self.regions = [r for r in self.regions if r is not reg]
                self.regions += [a, b]
        return abs(reg["ema"] - prev)


def _lp_intrinsic_factory(seed: int) -> Callable:
    """LC.00's q_lp form on a learned continuous outcome model. See docstring."""
    torch.manual_seed(seed * 811 + 5)
    net = torch.nn.Sequential(
        torch.nn.Linear(6 + ACTION_DIM, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6))
    opt = torch.optim.Adam(net.parameters(), lr=LP_LR)
    regions = _LPRegions()
    state = {"prev_o": None, "prev_xy": None, "life": -1,
             "mu": np.zeros(6), "var": np.ones(6), "n": 0}

    def outcome(w) -> np.ndarray:
        pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
        touch = w._touch()                     # [logF, flag] x 4 geoms
        foot, handL, handR = touch[3], touch[5], touch[7]
        if state["prev_xy"] is None:
            speed = 0.0
        else:
            speed = float(np.hypot(pos[0] - state["prev_xy"][0],
                                   pos[1] - state["prev_xy"][1]) / 0.2)
        state["prev_xy"] = pos[:2].copy()
        return np.array([pos[0], pos[1], pos[2],
                         pos[2] * (1.0 - float(foot)),
                         float(handL) + float(handR), speed])

    def zscore(o: np.ndarray) -> np.ndarray:
        state["n"] += 1
        d = o - state["mu"]
        state["mu"] = state["mu"] + d / state["n"]
        state["var"] = state["var"] + (d * (o - state["mu"]) - state["var"]) / state["n"]
        return (o - state["mu"]) / np.sqrt(np.maximum(state["var"], 1e-6))

    def fn(w, obs, a, core) -> float:
        o_now = outcome(w)
        oz_now = zscore(o_now)
        if w.life != state["life"] or state["prev_o"] is None:
            state["life"], state["prev_o"] = w.life, oz_now
            return 0.0
        x = torch.from_numpy(np.concatenate(
            [state["prev_o"], a]).astype(np.float32))
        y = torch.from_numpy(oz_now.astype(np.float32))
        pred = net(x)
        loss = ((pred - y) ** 2).mean()
        err = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        r_lp = LP_BETA * regions.update(state["prev_o"], err)
        state["prev_o"] = oz_now
        return r_lp

    return fn


def _randrew_factory(seed: int) -> Callable:
    """Control (b): a fixed random stationary projection of the observation."""
    wvec: Dict[str, np.ndarray] = {}

    def fn(r_h, w, obs, core) -> float:
        x = np.concatenate([obs[k] for k in obs])
        if "w" not in wvec:
            rng = np.random.RandomState(seed * 31 + 7)
            wvec["w"] = rng.randn(x.shape[0]) / math.sqrt(x.shape[0])
        return float(wvec["w"] @ x)

    return fn


def _uncertainty_reward_factory(sign: float, kappa: Optional[float]) -> Callable:
    """wm-efe (sign=+1, kappa) and darkroom (sign=-1, kappa=None => pure)."""
    state = {"rm": None}

    def fn(r_h, w, obs, core) -> float:
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            u = float(core.uncertainty(obs_t, dropped=W0.DROPPED))
        state["rm"] = u if state["rm"] is None else (
            0.99 * state["rm"] + 0.01 * u)
        u_norm = u / max(state["rm"], 1e-9)
        if kappa is None:
            return sign * u_norm
        return float(r_h) + sign * kappa * u_norm

    return fn


# ── the ported detectors ───────────────────────────────────────────────────
def _panel_dwell(xy: np.ndarray, panel_xy: Optional[np.ndarray]) -> float:
    """PG.4's dwell share: late half, strict < DWELL_RADIUS."""
    if panel_xy is None or len(xy) < 4:
        return 0.0
    half = len(xy) // 2
    d = np.hypot(xy[half:, 0] - panel_xy[0], xy[half:, 1] - panel_xy[1])
    return float((d < DWELL_RADIUS).mean())


def _chaos_detect(pool: Dict[str, np.ndarray], null_name: str,
                  seed: int, obs_dim: int) -> Dict[str, dict]:
    """CURIOSITY_BAKEOFF.md §2.10, ported. See the module docstring.

    `pool[name]` rows are [s | a | mean_a | s' | r] with s of `obs_dim`.
    Returns per-name {occupancy, ratio, r_in, r_out, frac}.
    """
    late = {}
    for name, rows in pool.items():
        h = len(rows) // 2
        late[name] = rows[h:]
    names = list(late)
    X = np.concatenate([late[n][:, :obs_dim + ACTION_DIM] for n in names])
    Y = np.concatenate([late[n][:, 2 * ACTION_DIM + obs_dim:
                                2 * ACTION_DIM + 2 * obs_dim] for n in names])
    owner = np.concatenate([np.full(len(late[n]), i)
                            for i, n in enumerate(names)])
    n_rows = len(X)
    gen = np.random.RandomState(seed * 977 + 3)
    perm = gen.permutation(n_rows)
    folds = np.array_split(perm, 5)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        torch.manual_seed(seed * 613 + len(train_idx))
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + ACTION_DIM, CHAOS_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(CHAOS_HIDDEN, CHAOS_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(CHAOS_HIDDEN, obs_dim))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(X[train_idx])
        yt = torch.from_numpy(Y[train_idx])
        for _ in range(CHAOS_EPOCHS):
            order = torch.randperm(len(xt))
            for i in range(0, len(xt), CHAOS_BATCH):
                idx = order[i:i + CHAOS_BATCH]
                loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = net(torch.from_numpy(X[test_idx]))
            return ((pred - torch.from_numpy(Y[test_idx])) ** 2
                    ).mean(dim=1).numpy()

    e_full = np.zeros(n_rows)
    e_half = np.zeros(n_rows)
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        e_full[test] = fit_predict(train, test)
        e_half[test] = fit_predict(train[:len(train) // 2], test)

    null_i = names.index(null_name)
    theta = float(np.percentile(e_full[owner == null_i], CHAOS_NULL_PCT))
    reduc = (e_half - e_full) / np.maximum(e_half, 1e-9)
    chaotic = (e_full >= theta) & (reduc < CHAOS_RED_EPS)

    out = {}
    for i, name in enumerate(names):
        m = owner == i
        frac = float(chaotic[m].mean()) if m.any() else 0.0
        r = late[name][:, -1].astype(np.float64)
        c = chaotic[m]
        r_in = float(r[c].mean()) if c.any() else 0.0
        r_out = float(r[~c].mean()) if (~c).any() else 0.0
        ratio = r_in / r_out if abs(r_out) > 1e-12 else math.copysign(
            CHAOS_RATIO_CLIP, r_in) if abs(r_in) > 1e-12 else 0.0
        ratio = float(np.clip(ratio, -CHAOS_RATIO_CLIP, CHAOS_RATIO_CLIP))
        out[name] = {"occupancy": frac / 0.10, "ratio": ratio,
                     "r_in": r_in, "r_out": r_out, "frac": frac}
    return out


def _final_slope(spans: List[float]) -> float:
    """s-per-life linear slope over the final half of lives (data-starved)."""
    half = spans[len(spans) // 2:]
    if len(half) < 3:
        return 0.0
    x = np.arange(len(half), dtype=float)
    return float(np.polyfit(x, np.asarray(half, dtype=float), 1)[0])


def _entropy_of_means(rows: np.ndarray, obs_dim: int) -> float:
    """Final-third spread of the PRE-NOISE policy means (wm-efe's VOID gate)."""
    if len(rows) < 9:
        return 0.0
    means = rows[-(len(rows) // 3):,
                 obs_dim + ACTION_DIM:obs_dim + 2 * ACTION_DIM]
    return float(means.std(axis=0).mean())


# ── the experiment ─────────────────────────────────────────────────────────
def _experiment(seed: int, n_steps: int = N_STEPS, half_steps: int = HALF_STEPS,
                e0: float = E0, w_clock: Optional[float] = W_CLOCK_CORE_S,
                write_artifact: bool = True) -> dict:
    cal, prov = _borrow()
    m: dict = dict(prov)                # borrowed_* provenance, strings kept
    if cal is None:
        m["borrowed_ok"] = 0.0
        return m
    m["borrowed_ok"] = 1.0
    j0, alpha, ratios = cal["j0"], cal["alpha"], cal["ratios"]
    for a in CANDIDATE_ARMS:
        m[f"{a}/train_ratio"] = float(ratios[a])

    def go(**kw) -> dict:
        return run_survival(seed, j0=j0, alpha=alpha, e0=e0, **kw)

    runs: Dict[str, dict] = {}
    runs["null_random"] = go(n_decisions=n_steps, policy="random",
                             record_xy=True, record_transitions=TRACE_EVERY)
    runs["null_repeat"] = go(n_decisions=n_steps, policy="random-repeat")

    def arm_hooks(arm: str) -> dict:
        """Fresh reward machinery per run — closures carry state and a
        normaliser shared across runs would couple them."""
        h = {}
        if arm == "wm-efe":
            h["reward_fn"] = _uncertainty_reward_factory(+1.0, KAPPA_EFE)
        if arm == "ppo-lp":
            h["intrinsic_fn"] = _lp_intrinsic_factory(seed)
        return h

    for arm in CANDIDATE_ARMS:
        runs[arm] = go(policy="core", arm=arm, train=True,
                       train_ratio=ratios[arm], n_decisions=n_steps,
                       record_xy=True, record_transitions=TRACE_EVERY,
                       min_core_s=w_clock, explore_std=EXPLORE_STD_LC03,
                       **arm_hooks(arm))
        runs[f"{arm}/wiped"] = go(policy="core", arm=arm, train=True,
                                  train_ratio=ratios[arm],
                                  n_decisions=n_steps, wipe_at_death=True,
                                  explore_std=EXPLORE_STD_LC03,
                                  **arm_hooks(arm))
        runs[f"{arm}/twin"] = go(policy="core", arm=arm, train=False,
                                 train_ratio=ratios[arm],
                                 n_decisions=half_steps,
                                 explore_std=EXPLORE_STD_LC03,
                                 **arm_hooks(arm))

    # ── the ported detectors, once per seed ────────────────────────────
    w_probe = W0(seed=seed, j0=j0, alpha=alpha)
    obs_dim = int(sum(v.shape[0] for v in w_probe.observe().values()))
    panel_xy = (np.array(w_probe.model.geom_pos[w_probe.panel_gid][:2])
                if w_probe.panel_gid >= 0 else None)
    del w_probe
    pool = {a: runs[a]["transitions"] for a in CANDIDATE_ARMS}
    pool["null_random"] = runs["null_random"]["transitions"]
    chaos = _chaos_detect(pool, "null_random", seed, obs_dim)

    # ── metrics ────────────────────────────────────────────────────────
    lg_null = runs["null_random"]["life_gain"]
    m["null_life_gain"] = lg_null
    m["null_repeat_life_gain"] = runs["null_repeat"]["life_gain"]
    m["null_mean_life_s"] = runs["null_random"]["mean_life_s"]
    m["null_n_lives_ok"] = float(
        runs["null_random"]["n_lives"] >= N_LIVES_MIN
        and runs["null_repeat"]["n_lives"] >= N_LIVES_MIN)
    m["null_thrash"] = runs["null_random"]["thrash_l1"]
    m["null_chaos_occupancy"] = chaos["null_random"]["occupancy"]
    finite = [runs[k]["physics_finite"] for k in runs]
    m["physics_finite_min"] = float(min(finite))

    for arm in CANDIDATE_ARMS:
        r, w_, t = runs[arm], runs[f"{arm}/wiped"], runs[f"{arm}/twin"]
        m[f"{arm}/life_gain"] = r["life_gain"]
        m[f"{arm}/mean_life_s"] = r["mean_life_s"]
        m[f"{arm}/n_lives"] = r["n_lives"]
        m[f"{arm}/lives_ok"] = float(r["n_lives"] >= N_LIVES_MIN)
        m[f"{arm}/lg_margin_null"] = r["life_gain"] - lg_null
        m[f"{arm}/lg_margin_twin"] = r["life_gain"] - t["life_gain"]
        m[f"{arm}/twin_life_gain"] = t["life_gain"]
        m[f"{arm}/wiped_life_gain"] = w_["life_gain"]
        m[f"{arm}/clt"] = r["life_gain"] - w_["life_gain"]
        m[f"{arm}/needs_rise"] = (r["needs_ok_final_third"]
                                  - r["needs_ok_first_third"])
        m[f"{arm}/panel_dwell"] = _panel_dwell(r["xy"], panel_xy)
        m[f"{arm}/dwell_ok"] = float(
            m[f"{arm}/panel_dwell"] <= PANEL_DWELL_MAX)
        m[f"{arm}/chaos_occupancy"] = chaos[arm]["occupancy"]
        m[f"{arm}/chaos_reward_ratio"] = chaos[arm]["ratio"]
        m[f"{arm}/chaos_ok"] = float(not (
            chaos[arm]["occupancy"] >= CHAOS_OCC_VOID
            and chaos[arm]["ratio"] >= CHAOS_RATIO_VOID))
        m[f"{arm}/thrash_ratio"] = (r["thrash_l1"]
                                    / max(m["null_thrash"], 1e-9))
        m[f"{arm}/action_entropy_final"] = _entropy_of_means(
            r["transitions"], obs_dim)
        m[f"{arm}/final_slope"] = _final_slope(r["life_spans"])
        m[f"{arm}/optimiser_steps"] = r["optimiser_steps"]
        m[f"{arm}/decisions"] = r["decisions"]
        m[f"{arm}/core_s"] = r["process_time_s"]
        m[f"{arm}/params"] = r.get("params", 0.0)

    if write_artifact:
        ARTIFACTS.mkdir(exist_ok=True)
        art = {}
        for k, r in runs.items():
            art[k] = {kk: r[kk] for kk in
                      ("life_spans", "life_ends", "deaths_at_decision",
                       "optimiser_steps", "decisions", "sim_seconds",
                       "process_time_s", "reward_sum")}
            art[k]["params"] = r.get("params", 0.0)
            art[k]["grad_flops_est"] = r.get("grad_flops_est", 0.0)
        (ARTIFACTS / f"lc03_curves_seed{seed}.json").write_text(
            json.dumps({"seed": seed, "e0": e0, "n_steps": n_steps,
                        "w_clock_core_s": w_clock, "runs": art}, indent=1))
    return m


def _control(seed: int, n_steps: int = HALF_STEPS, e0: float = E0) -> dict:
    """(a) statue, (b) randrew, (e) darkroom — plus their own paired null."""
    cal, _ = _borrow()
    if cal is None:
        return {"borrowed_ok": 0.0}
    j0, alpha, ratios = cal["j0"], cal["alpha"], cal["ratios"]

    def go(**kw) -> dict:
        return run_survival(seed, j0=j0, alpha=alpha, e0=e0,
                            n_decisions=n_steps, **kw)

    null = go(policy="random")
    statue = go(policy="statue")
    randrew = go(policy="core", arm="ppo-needs", train=True,
                 train_ratio=ratios["ppo-needs"],
                 explore_std=EXPLORE_STD_LC03,
                 reward_fn=_randrew_factory(seed))
    darkroom = go(policy="core", arm="dreamer-xs", train=True,
                  train_ratio=ratios["dreamer-xs"],
                  explore_std=EXPLORE_STD_LC03,
                  reward_fn=_uncertainty_reward_factory(-1.0, None))
    return {
        "borrowed_ok": 1.0,
        "ctrl_null_life_gain": null["life_gain"],
        "ctrl_null_mean_life_s": null["mean_life_s"],
        "statue_mean_life_s": statue["mean_life_s"],
        "statue_life_gain": statue["life_gain"],
        "statue_n_lives": statue["n_lives"],
        "randrew_life_gain": randrew["life_gain"],
        "randrew_margin": randrew["life_gain"] - null["life_gain"],
        "randrew_opt_steps": randrew["optimiser_steps"],
        "darkroom_life_gain": darkroom["life_gain"],
        "darkroom_margin": darkroom["life_gain"] - null["life_gain"],
        "darkroom_mean_life_s": darkroom["mean_life_s"],
    }


def _tstat(m: dict, key: str) -> float:
    """The house paired 3-sigma idiom: mean * sqrt(n_seeds) / seed std."""
    return m.get(key, 0.0) * math.sqrt(3) / max(m.get(f"{key}_std", 0.0), 1e-9)


def _void(m: dict, reason: str):
    """Name the firing branch in the recorded metrics (v2). The generic VOID
    message admits every narrative; attempt 2's harvest proved it by
    attributing a control-(c) trip to the claim loop that never ran."""
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # ── instrument validity ─────────────────────────────────────────────
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")  # refuses, never refutes
    if m.get("physics_finite_min", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("null_n_lives_ok", 0.0) != 1.0:
        return _void(m, "nulls under 12 lives")  # the world cannot produce
        # 12 lives at this envelope — a world problem, not an arm result

    # ── controls, each on its pre-registered side ───────────────────────
    # (a) AMENDED 2026-08-13 from 'dies soonest' (seed-90 pilot, same commit,
    # T1.02 precedent — see PILOT RESOLUTION in the docstring): passivity
    # maximises life length in W0 (statue 180.0 s = e0/BASAL_B to 0.02%,
    # longest of every pilot run), so the statue now certifies the passive
    # path is CLEAN — nothing but basal starvation may kill a body that
    # never acts (a phantom-damage rig fault, PS.03's servo scar, is what
    # this catches; ctrl runs at module E0, so the ceiling is E0/BASAL_B).
    ceiling = E0 / BASAL_B
    if not abs(c.get("statue_mean_life_s", 0.0) - ceiling) <= 0.10 * ceiling:
        return _void(m, "control (a): statue off the basal ceiling")
    # (b) randrew must miss the null gate.
    if _tstat(c, "randrew_margin") >= SIGMA_GATE:
        return _void(m, "control (b): randrew cleared the null gate")
    # (e) AMENDED 2026-08-13 from 't <= -3' (same pilot, same commit): the
    # darkroom learned passivity and prospered on the length ruler (margin
    # +49.7 s, mean life 183.5 s vs null 126.2 s) — anti-curiosity WINS life
    # length in W0, so life_gain cannot carry curiosity's sign. The measured
    # inversion is locked in: if the world ever punishes anti-curiosity
    # strongly, this fires and the rig is re-derived, never silently re-read.
    if _tstat(c, "darkroom_margin") <= -SIGMA_GATE:
        return _void(m, "tripwire (e): darkroom strongly negative")
    # (c) frozen and (d) wiped-store: every twin within noise of zero.
    for arm in CANDIDATE_ARMS:
        for kind in ("twin_life_gain", "wiped_life_gain"):
            k = f"{arm}/{kind}"
            if (abs(_tstat(m, k)) >= SIGMA_GATE
                    and abs(m.get(k, 0.0)) > NOISE_FLOOR_S):
                return _void(m, f"control (c)/(d): {k} = "
                             f"{m.get(k, 0.0):.2f} s, |t| = "
                             f"{abs(_tstat(m, k)):.2f}")
                # lives lengthen (or shorten) without a persistent learner:
                # the metric measures the world (registry (c)/(d))

    # ── the claim ───────────────────────────────────────────────────────
    dreamer_ent = m.get("dreamer-xs/action_entropy_final", 0.0)
    cleared = 0
    for arm in CANDIDATE_ARMS:
        ok = False
        if m.get(f"{arm}/chaos_ok", 0.0) != 1.0:
            pass                        # §2.10: VOID for that arm
        elif (arm == "wm-efe" and dreamer_ent > 0.0
                and m.get(f"{arm}/action_entropy_final", 0.0)
                < EFE_ENTROPY_FRAC * dreamer_ent):
            pass                        # epistemic collapse: VOID for wm-efe
        else:
            ok = (_tstat(m, f"{arm}/lg_margin_null") >= SIGMA_GATE
                  and _tstat(m, f"{arm}/lg_margin_twin") >= SIGMA_GATE
                  and m.get(f"{arm}/lives_ok", 0.0) == 1.0
                  and m.get(f"{arm}/needs_rise", -1.0) > 0.0
                  and m.get(f"{arm}/clt", -1.0) > 0.0
                  and m.get(f"{arm}/dwell_ok", 0.0) == 1.0)
        # The owner's data-starved guard, machine-readable (v2 — the 08-21
        # harvest found this key promised in the docstring and computed
        # nowhere). run_spec records the same dict this mutates, so the
        # flag lands in the ledger row beside the margins it qualifies.
        m[f"{arm}/data_starved"] = float(
            not ok and m.get(f"{arm}/final_slope", 0.0) > 0.0)
        cleared += int(ok)
    if cleared >= 2:
        return True
    return _void(m, f"fewer than two learners ({cleared} cleared)")
    # Blocks the decision instead of manufacturing one. Data-starved arms
    # (positive final_slope) are re-screened at a bigger envelope, not
    # eliminated.


def run(ledger: Ledger | None = None):
    """The registered run: 3 seeds over 3 single-threaded workers, memoised
    into run_spec (T2.01's pattern — run_spec calls fn once per seed and the
    work must not happen twice)."""
    import multiprocessing as mp

    spec = BY_ID["LC.03"]
    seeds = list(range(spec.seeds))
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        exp = dict(zip(seeds, pool.map(_experiment, seeds)))
        ctl = dict(zip(seeds, pool.map(_control, seeds)))
    return run_spec(spec, lambda s: exp[s], _check,
                    control_fn=lambda s: ctl[s], ledger=ledger or Ledger())


def _worker_init():
    import os
    torch.set_num_threads(1)
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


def _pilot():
    """Seed 90, compressed envelope. Prints JSON; records NOTHING."""
    t0 = time.time()
    torch.set_num_threads(1)
    m = _experiment(PILOT_SEED, n_steps=PILOT_STEPS,
                    half_steps=PILOT_STEPS // 2, e0=PILOT_E0, w_clock=None,
                    write_artifact=False)
    c = _control(PILOT_SEED, n_steps=PILOT_STEPS // 2, e0=PILOT_E0)
    print(json.dumps({"pilot_seed": PILOT_SEED, "steps": PILOT_STEPS,
                      "e0": PILOT_E0, "elapsed_s": round(time.time() - t0, 1),
                      "experiment": m, "control": c}, indent=1))


def _smoke():
    """Minutes-long mechanics check. Records nothing; asserts on the product."""
    torch.set_num_threads(2)
    cal, prov = _borrow()
    assert cal is not None, f"borrow refused: {prov}"
    j0, alpha = cal["j0"], cal["alpha"]

    # Force NON-ROOT region splits: list.remove scans with == (identity
    # short-circuit), so the crash only reachable when the splitting region
    # is not at index 0 — the 400-decision runs below never get there.
    _rng = np.random.RandomState(0)
    _regs = _LPRegions()
    for _ in range(LP_SPLIT_N * 40):
        _regs.update(_rng.randn(6) * 3.0, err=float(_rng.rand()))
    assert len(_regs.regions) > 2, "forced splits did not happen"
    for _ in range(200):
        _oz = _rng.randn(6) * 3.0
        assert sum(1 for rg in _regs.regions
                   if np.all(_oz >= rg["lo"]) and np.all(_oz < rg["hi"])) == 1
    print("lp regions ok:", len(_regs.regions), "regions, partition intact")

    lp = _lp_intrinsic_factory(0)
    r = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=400,
                     policy="core", arm="ppo-lp", train=True, train_ratio=0.5,
                     intrinsic_fn=lp, record_xy=True, record_transitions=4)
    assert r["optimiser_steps"] > 0 and np.isfinite(r["reward_sum"])
    assert len(r["xy"]) == 400 and r["transitions"].shape[0] > 10
    obs_dim = (r["transitions"].shape[1] - 2 * ACTION_DIM - 1) // 2
    print("lp arm ok:", {k: round(float(r[k]), 3) for k in
                         ("n_lives", "optimiser_steps", "reward_sum",
                          "thrash_l1")}, "obs_dim", obs_dim)

    null = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=400,
                        policy="random", record_xy=True, record_transitions=4)
    ch = _chaos_detect({"ppo-lp": r["transitions"],
                        "null_random": null["transitions"]},
                       "null_random", 0, obs_dim)
    assert 0.5 < ch["null_random"]["occupancy"] < 2.0, \
        f"null occupancy should be ~1.0 by construction: {ch}"
    print("chaos ok:", {k: {kk: round(vv, 3) for kk, vv in v.items()}
                        for k, v in ch.items()})

    w_probe = W0(seed=0, j0=j0, alpha=alpha)
    pxy = np.array(w_probe.model.geom_pos[w_probe.panel_gid][:2])
    at_panel = np.tile(pxy, (100, 1)).astype(np.float32)
    assert _panel_dwell(at_panel, pxy) == 1.0
    assert _panel_dwell(at_panel + 50.0, pxy) == 0.0
    print("panel_dwell ok; committed ratios:", cal["ratios"])

    efe = _uncertainty_reward_factory(+1.0, KAPPA_EFE)
    dark = _uncertainty_reward_factory(-1.0, None)
    e = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=60,
                     policy="core", arm="wm-efe", train=True, train_ratio=0.25,
                     reward_fn=efe)
    d = run_survival(0, j0=j0, alpha=alpha, e0=0.12, n_decisions=60,
                     policy="core", arm="dreamer-xs", train=True,
                     train_ratio=0.25, reward_fn=dark)
    assert np.isfinite(e["reward_sum"]) and np.isfinite(d["reward_sum"])
    assert d["reward_sum"] < 0, "darkroom's channel must be negative"
    print("efe/darkroom ok:", round(e["reward_sum"], 3),
          round(d["reward_sum"], 3))
    print("SMOKE OK")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "smoke":
        _smoke()
    elif mode == "pilot":
        _pilot()
    else:
        res = run()
        print(res.status, res.message)
