# RULE — the dial's success gate: gated vs uniform (ORDERS Next-up 3)

Written 2026-11-18 20:40 UTC, **before any seed, rule text only. It RUNS ONLY
AFTER J11's pilot has a reading** (the order says so). Nothing in this file is
run yet; no code changed in this commit. Apparatus: `experiments/dial_curriculum.py`
(`d9caa5eb`, Avalon's `CurriculumWrapper` + generator draw, every difference
listed in its docstring; proof `test_dial_curriculum`).

## The prior, and what this run can and cannot show

- Record (PRIORS §3, l. 18/100/116): Avalon eat task, IMPALA, 50M steps:
  **gated 0.716 vs uniform 0.001**; another study 57% wins vs none. Largest
  published effect class of any staging method.
- **This run is a PILOT, said so now.** 300,000 decisions is 1/167 of
  Avalon's budget; the record does not say whether the gap exists at 3×10^5.
  A null here is weak evidence against the dial; a gap is a strong lead, not
  a seat.
- **Structural fact known before data:** `lo` = ceiling₀ = 0.75 (ruling 22),
  so until his first sustained life the gated arm IS a fixed-d0 arm, and the
  uniform arm draws U[0.75, 1] (harder on average). A gated "win" with a
  ceiling that never left 0.75 is the fixed-easy-world effect, not the gate.
  Clause G below separates the two; the sham arm separates gate-information
  from ceiling-wander.

## Bound by J11 (filled in from J11's frozen rule and reading, not re-chosen)

- learner, hyper-parameters, reward, γ, movement-cost dial, eye frame, motor
  frame, `food_floor_only`, `native_substep`, `lazy_contact_audio`: **the J11
  arm with the best reading** (sustained > eats-but-starves > slides; tie →
  the arm listed first in J11's rule). If J11's every arm reads "slides",
  this rule does not run (nothing to stage) — say so in ORDERS and stop.
- success = `dial_curriculum.sustained` (pilot_b's row: capped AND intake ≥
  basal), unchanged.

## Arms × seeds (fresh seeds 521–525, the same 5 in every arm)

| arm | dial | role |
|---|---|---|
| `gated` | `DialCurriculum(mode="gated", step=0.01, lo=0.75)` | treatment |
| `uniform` | `mode="uniform"` (U[0.75, 1] per life) | Avalon's no-curriculum contrast |
| `sham` | gated, but each life's up/down is a coin with P(up) = that seed's `gated` training success rate | **control that must not beat uniform** (a wandering ceiling with no information) |
| floors | J11's named floor policy (held8, no learner) under `gated` and `uniform` | learner-free floor on the SAME eval worlds |

T = 300,000 decisions per run. 15 learner runs + 2 floor runs.

## Measurement — evaluation lives at fixed difficulty

Training-life success is not comparable across arms (the arms train on
different d). So: every 25,000 decisions, **20 eval lives at d = 0.75 and 20
at d = 1.0**, world seeds from one stream `default_rng([seed, 0xE7A1])` in
[40,000, 41,000) — **identical across arms** (common worlds). Learning paused
during eval lives and they are not written to replay (apparatus, said so);
actor deterministic (mean action). Floors run the same eval lives.

Reading per run: **final rate** = sustained eval lives / lives over the last 3
eval points (60 lives per d). Reported per d; the decision uses d = 0.75 and
reports d = 1.0.

## Decision (d = 0.75 final rates)

- **GATE HELPS** iff all of: (1) `gated` > `uniform` on ≥ 4 of 5 paired seeds;
  (2) pooled `gated` rate > the floor's Poisson 95% upper bound on the same
  eval lives (300 lives); (3) `sham` > `uniform` on ≤ 2 of 5 seeds (else the
  effect is ceiling-wander, reading "NOT THE GATE"); (4) clause G holds.
- **Clause G (the gate opened):** `gated`'s ceiling exceeds 0.80 at T on ≥ 3
  of 5 seeds. If it does not, the reading is **"GATE NEVER OPENED"** —
  whatever the rates, it is the fixed-d0 contrast, reported as that, no
  curriculum claim.
- **NO GAP**: (1) fails with G holding. Pilot null, weak (see above).
- **UNIFORM BETTER**: `uniform` > `gated` on ≥ 4 of 5 seeds — reported; the
  record's "a poor schedule can be worse than none" (§4) is the first reading.
- **VOID (our code is broken)**: a floor run's eval rate exceeds its own f-card
  arm's 95% upper bound × 2, or two `gated` seeds with identical
  `history` (the world stream is not per seed), or any eval world differs
  between arms (sha of the eval seed list printed per run).
- A cut run (session limit) counts at its last completed eval point if it
  has ≥ 6 eval points (150k), else it is a missing seed; ≥ 2 missing seeds in
  any arm → the arm is "incomplete", no reading.

**Code this rule needed — WRITTEN 11-19 (`dial_curriculum.py`: `sham_success_fn`, `eval_plan`/`plan_sha`, `eval_lives`, `make_eval_callback`, `final_rate`; proof `tests/test_dial_race_parts.py` PASS, planted leaky eval caught). Was:** an eval-lives hook (paused
learner, common eval seeds, no replay write) in the pilot runner; `sham`'s
`success_fn` (a seeded coin, `default_rng([seed, 0x5A4D])`). `sham` seed s
runs after `gated` seed s is home (its P(up) is that run's measured rate).

No threshold here may be lowered after seed 521 runs. Reported-only: it
picks whether the decisive test stages his worlds; no seat moves.
