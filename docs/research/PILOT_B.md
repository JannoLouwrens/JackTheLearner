# PILOT_B — Brain B lives in his world at the entry rung (ORDERS J8)

PLAN stage 1, ready-list item 4; step 4 (a PILOT of the last 10%). It fixes W
and T for the decisive test and moves no seat. Code: `experiments/pilot_b.py`
(its docstring holds the full "ours beside theirs" table). Harness mark:
`experiments/harness.outcome_sustained` (ruling 22).

## Setup (J8 item 1, 11-16)

- Learner: Brain B = SBX 0.28.0 `SAC` + LayerNorm, 2×256, unmodified (reproduced
  J3 rows 1b and 3b). Venv `/data/venvs/jack-std`.
- World: `JackEnv(difficulty=0.75, reward="dr100-eat", random_needs=1/6,
  random_layout=True)`; one episode = one life; death = terminated; 9,000-decision
  cap = truncated, then a new body in the same world.
- Settings (§1/§5): lr 3e-4, batch 256, replay = whole run, tau 0.005, γ 0.995
  (40 s horizon at 0.2 s), 5,000 random warm-up decisions, 32 updates every 32
  decisions (1 per decision), exploration = row 3b's (SBX `use_sde`, draw held per
  32-decision rollout, ent_coef 0.1 fixed — differs from §5's default "auto"),
  observations normalised by `VecNormalize` (running stats, frozen at evaluation).
- Per life: decisions, seconds, floor meals, apples, cause, intake (meal sizes
  read from the world's own table at the rung), basal drain, sustained, reward.

## Harness: ruling 22's sustained mark (J8 item 3, first part — done 11-16)

`sustained_life(capped, intake, drain)` and `outcome_sustained(lives, W, T)`:
decisions until W CONSECUTIVE lives each reach the cap with intake ≥ basal drain;
one unsustained life resets the streak; crash/NaN censored at T (ruling 19).
`prereg.freeze(..., mark="sustained")` freezes which mark judges. Edge-case tests
(`tests/test_harness.py`, `test_22_*`, 8 tests) include the planted disagreement:
f2-like capped lives eating 2.4 bars against 3.0 drained cross the θ = 5 mark and
do NOT cross the sustained mark; row-e-like fed lives cross both.

## Box smoke (J8 item 2)

11-16 08:10–08:19 UTC, box, nice 19, seed 0, 20,000 decisions WITH learning
(`/data/jack-logs/pilot_b/smoke_s0_20000.json`): **40.2 decisions/s wall, 25.9 per
CPU-second** (JAX ran ~1.55 cores). 300,000 decisions = 3.2 CPU-h per run → 4 runs
fit one 9-hour Kaggle kernel even at half the box's per-core speed; T = 300,000
stands. Lives: 6 energy deaths (0–1 meals each), 1 unfinished, and life 5
(decisions 5,487–14,486, the first after warm-up) capped and sustained — 23 floor
meals, intake 7.13 bars vs basal drain 3.0. Read through the harness: sustained
mark at W = 1 crossed at decision 14,486. **One seed: an anecdote, not a reading**
— held-8 random caps 2 of 200 lives at this rung (J2).

## The rule (J8 item 3) — committed 11-16, before any kernel

Verbatim in `experiments/pilot_b.py` ("THE RULE"). In short: T = 300,000 for every
arm; B 20 runs (seeds 301–320); floor arms 5 runs each (seeds 301–305): do-nothing,
held-8 random, B with learning rate 0; ceiling = J1's registered row e (not re-run).
Floor gate: ≤ 1 of 15 floor runs crosses the sustained mark at W = 1, else VOID.
Readings: SUSTAINED if ≥ 5 of 20 B runs cross the sustained mark at W = 1; EATS BUT
STARVES if not, but ≥ 5 of 20 cross θ = 5 at W = 3; else DOES NOT LEARN AT THIS
BUDGET. Fixes W (largest of 1, 2, 3 with ≥ 5 crossings) and T (300k, or 150k if the
median crossing ≤ 150k) for the decisive test.

## Results

_none yet — kernels not launched (J8 item 4)._
