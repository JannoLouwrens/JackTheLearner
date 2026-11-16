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

## Seen at the f0/f1 harvest (11-16 21:30): the do-nothing body can meet the mark by luck — not a reading

- `noop_s302`, life 33 (of 105): **capped, 36 floor meals, intake 11.16 vs
  basal drain 3.0 → a sustained life, and so a crossing at W 1** — from
  a body that never acts. All 36 of the noop arm's meals so far (322 lives)
  come from that one life. Mechanism, from the arithmetic, not from a
  position log: the jack_env docstring says a body resting on a floor food
  eats by contact; at d 0.75 the food regrows every 0.75 × 66.9 = 50.2 s,
  and 1,800 s ÷ 50.2 s = 35.9 ≈ 36 meals. He was born touching a food
  (the spawn is within 13.25 m of food at this rung) and ate every regrowth.
- Rate: 1 per 322 floor lives (95% CI 0.00008–0.017). The rule allows the
  floor arms ≤ 1 crossing in total before the pilot is VOID; one is used.
  At the point estimate, the remaining ~214 noop lives (s304, s305) cross
  again with probability **0.49** (CI 0.02–0.98). **Whether the pilot is VOID is
  about a coin flip, decided by where the spawner puts the body.** The rule is frozen and
  is not changed; this is written down before f2/f3 come in.
- At W 1, one lucky spawn is a whole run's success: per 107-life run
  P ≈ 0.28 at this rate. At W 2 (two sustained lives in a row) the same
  lottery is ≈ 0.001 per run. **This is the input the pilot exists to give:
  the decisive test's W must make the do-nothing lottery negligible.** W 1 does not.
- B: 0 sustained in 1,878 lives, where a do-nothing body at this rate would
  get ~5.8 (P(0) = 0.003). Suggestive, from one event: B moves off food he
  is born on, so on the primary mark he does worse than standing still.
  Not a reading.

## Results

Generated by `python -m experiments.pilot_b_report` from `/data/jack-logs/pilot_b/` (f0.json, f1.json, k0.json, k1.json, k2.json, k3.json). Registered runs in: **24 of 35**; exploration (reported only): 0 of 10.

### PARTIAL — descriptive only, no reading (the rule reads 20 + 15)

Missing: b_s317, b_s318, b_s319, b_s320, noop_s304, noop_s305, held8_s304, held8_s305, blr0_s303, blr0_s304, blr0_s305.

### Per arm

| arm | runs in | runs crossing (sustained, W 1) | lives | % capped | life, median s | meals/life | meals per 1,000, last 50k (mean of runs) | ends |
|---|---|---|---|---|---|---|---|---|
| b | 16 of 20 | 0 | 1878 (117.4/run) | 0.0 | 496 | 0.42 | 0.095 | energy 1878 |
| noop | 3 of 5 | 1 | 322 (107.3/run) | 0.3 | 555 | 0.11 | 0.000 | capped 1 energy 321 |
| held8 | 3 of 5 | 0 | 290 (96.7/run) | 0.0 | 569 | 1.65 | 0.531 | energy 290 |
| blr0 | 2 of 5 | 0 | 181 (90.5/run) | 0.0 | 551 | 1.50 | 0.391 | energy 178 integrity 3 |

### Per run

Early = a run's first 5 lives (B's random warm-up); late = the last quarter of its finished lives. A motionless body at this rung lasts ~600 s; a random one ~300 s — the floor rows below measure both.

| arm | seed | lives | capped | sustained lives | first sustained at | meals/life first 20 → last 20 | early → late meals/life | % lives fed | unfed life, median s | reward/life | meals per 1,000 decisions, last 50k | cut-off life: decisions / meals / intake ÷ drain | ends |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b | 301 | 119 | 0 | 0 | – | 0.80 → 0.30 | 1.20 → 0.24 | 60 → 21 | 267 → 484 | 21 → -74 | 0.099 | 1,494 / 0 / 0.00 | energy 119 |
| b | 302 | 114 | 0 | 0 | – | 0.80 → 0.20 | 0.20 → 0.21 | 20 → 18 | 283 → 501 | -79 → -77 | 0.080 | 787 / 0 / 0.00 | energy 114 |
| b | 303 | 119 | 0 | 0 | – | 0.40 → 0.15 | 0.40 → 0.17 | 40 → 17 | 282 → 510 | -59 → -81 | 0.058 | 798 / 0 / 0.00 | energy 119 |
| b | 304 | 118 | 0 | 0 | – | 0.80 → 0.20 | 2.00 → 0.21 | 60 → 17 | 269 → 491 | 101 → -77 | 0.080 | 2,237 / 0 / 0.00 | energy 118 |
| b | 305 | 119 | 0 | 0 | – | 0.85 → 0.15 | 0.40 → 0.14 | 20 → 14 | 375 → 500 | -58 → -85 | 0.058 | 1,869 / 0 / 0.00 | energy 119 |
| b | 306 | 118 | 0 | 0 | – | 0.65 → 0.15 | 0.80 → 0.21 | 60 → 14 | 279 → 500 | -19 → -78 | 0.058 | 139 / 0 / 0.00 | energy 118 |
| b | 307 | 119 | 0 | 0 | – | 1.00 → 0.25 | 3.40 → 0.21 | 60 → 21 | 350 → 497 | 241 → -78 | 0.096 | 585 / 0 / 0.00 | energy 119 |
| b | 308 | 119 | 0 | 0 | – | 0.40 → 0.20 | 0.40 → 0.17 | 40 → 14 | 271 → 495 | -58 → -81 | 0.078 | 123 / 0 / 0.00 | energy 119 |
| b | 309 | 114 | 0 | 0 | – | 1.05 → 0.00 | 3.00 → 0.00 | 60 → 0 | 282 → 521 | 202 → -98 | 0.000 | 2,276 / 0 / 0.00 | energy 114 |
| b | 310 | 119 | 0 | 0 | – | 0.65 → 0.10 | 0.80 → 0.17 | 60 → 17 | 283 → 503 | -18 → -81 | 0.039 | 1,494 / 0 / 0.00 | energy 119 |
| b | 311 | 117 | 0 | 0 | – | 0.40 → 0.50 | 0.60 → 0.41 | 20 → 28 | 272 → 496 | -38 → -57 | 0.136 | 528 / 0 / 0.00 | energy 117 |
| b | 312 | 118 | 0 | 0 | – | 0.50 → 0.15 | 1.00 → 0.17 | 80 → 17 | 254 → 494 | 2 → -81 | 0.058 | 667 / 0 / 0.00 | energy 118 |
| b | 313 | 121 | 0 | 0 | – | 0.90 → 0.10 | 1.00 → 0.10 | 60 → 10 | 354 → 494 | 2 → -88 | 0.040 | 700 / 0 / 0.00 | energy 121 |
| b | 314 | 113 | 0 | 0 | – | 1.05 → 0.20 | 0.60 → 0.14 | 40 → 11 | 493 → 500 | -38 → -84 | 0.079 | 1,737 / 0 / 0.00 | energy 113 |
| b | 315 | 115 | 0 | 0 | – | 1.45 → 0.10 | 0.40 → 0.07 | 40 → 7 | 264 → 513 | -59 → -91 | 0.507 | 6,937 / 24 / 3.22 | energy 115 |
| b | 316 | 116 | 0 | 0 | – | 0.70 → 0.15 | 1.40 → 0.10 | 60 → 7 | 409 → 496 | 42 → -88 | 0.060 | 1,084 / 0 / 0.00 | energy 116 |
| noop | 301 | 108 | 0 | 0 | – | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 553 → 553 | -99 → -99 | 0.000 | 2,658 / 0 / 0.00 | energy 108 |
| noop | 302 | 105 | 1 | 1 | 101790 | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 574 → 549 | -99 → -98 | 0.000 | 1,872 / 0 / 0.00 | capped 1 energy 104 |
| noop | 303 | 109 | 0 | 0 | – | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 562 → 564 | -99 → -99 | 0.000 | 60 / 0 / 0.00 | energy 109 |
| held8 | 301 | 96 | 0 | 0 | – | 2.45 → 1.75 | 1.40 → 1.58 | 60 → 62 | 453 → 465 | 41 → 60 | 0.603 | 2,253 / 1 / 0.41 | energy 96 |
| held8 | 302 | 96 | 0 | 0 | – | 1.55 → 1.80 | 2.00 → 1.71 | 100 → 62 | – → 484 | 101 → 72 | 0.537 | 1,094 / 1 / 0.85 | energy 96 |
| held8 | 303 | 98 | 0 | 0 | – | 1.55 → 1.50 | 1.60 → 1.62 | 60 → 62 | 470 → 478 | 61 → 64 | 0.452 | 400 / 0 / 0.00 | energy 98 |
| blr0 | 301 | 92 | 0 | 0 | – | 1.70 → 0.70 | 0.80 → 0.61 | 40 → 26 | 282 → 500 | -19 → -38 | 0.319 | 2,035 / 6 / 2.74 | energy 92 |
| blr0 | 302 | 89 | 0 | 0 | – | 1.70 → 1.65 | 0.00 → 1.50 | 0 → 59 | 286 → 500 | -99 → 51 | 0.463 | 144 / 0 / 0.00 | energy 86 integrity 3 |
