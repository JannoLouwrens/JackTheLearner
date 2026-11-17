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

## Seen at the k4 harvest (11-16 21:40): B's arm is complete — learning made him eat less than his own untrained self

Descriptive, not the reading (7 floor runs still out). All 20 B seeds in:
**0 sustained in 2,350 lives; every death energy.**
- Meals/life, late quarter, B: max 0.41 (s311), median ~0.20. Every moving
  floor run so far sits above every B seed: held-8 random 1.58–1.71 (3/3),
  B with learning rate 0 — the same network, never updated — 0.61 and 1.50
  (2/2). **20/20 B seeds end below the untrained copy of themselves.**
- Direction: early → late meals/life fell in 18/20 seeds (rose: s302, s319);
  first 20 → last 20 lives fell in 18/20 (rose: s311, s317).
- Life median 496 s, between random (~300 s) and motionless (~555 s): the
  lie-still slide §13 predicted under a movement drain, now on all 20 seeds.
  Unfed life lengthened 267–493 s → 484–521 s in 20/20: he learned to
  spend less, not to find food.
- What this already says for the decisive test (not a ruling): at 300k and
  `dr100-eat`, B does not learn to eat in this world; the change the record
  predicts to work is the cost curriculum (KAPPA ramped with d), with
  `dr100-eat-pos` as the cheaper probe. The exploration arms e0–e2 are
  reported-only and cannot change this.

## Seen at the f2 harvest (11-16 23:25): the floor gate has failed — the pilot is VOID whatever f3 brings

f2 (blr0_s303, noop_s304, held8_s304, blr0_s304) landed 22:41. Floor runs in: 12 of 15
(missing only seed 305 of each arm, in f3). **Runs crossing the sustained mark at W 1:
4 (noop_s302, noop_s304, held8_s304, blr0_s304) against the gate's ≤ 1 → VOID.** f3 can
only add crossings, so this is decided, not predicted. The rule is frozen and is not
changed; the 35/35 report still runs and will say VOID.
- The crossing lives: noop_s304 life 48 — capped, **36** floor meals, intake 11.16 vs
  drain 3.0, the same 36 = 1,800 s ÷ 50.2 s regrowth as noop_s302 (born touching a food,
  eats every regrowth by contact). held8_s304 life 59 — 20 meals, 6.20; blr0_s304
  life 82 — 19 meals, 5.89. The moving floor bodies cross too: this is not only the
  do-nothing lottery written at f0/f1, it is any body that spawns near enough to food.
- Rate: 4 sustained lives in 1,198 floor lives = 0.0033 per life (95% CI ~0.0009–0.0085).
  The f0/f1 forecast (noop crosses again on s304/s305 with P 0.49) came in on s304.
- B against that rate: 0 in 2,350 lives where 7.8 are expected; P(0) ≈ 0.0004.
  Descriptive: on the primary mark B does worse than a body that never acts.
- For the decisive test (input, not a ruling): at 0.0033 per life and ~100 lives per run,
  W 1 is crossed by chance in ~28% of floor runs; W 2 (two sustained lives in a row)
  by chance in ~0.1%. **The decisive test needs W ≥ 2** (or a spawn rule that never
  starts him touching food), and the change the record predicts to make B eat at all
  is the cost curriculum (KAPPA ramped with d).

## Seen at the f3 harvest (11-17 02:10): 35 of 35 in — the reading is VOID, as decided at f2

f3 (noop_s305, held8_s305, blr0_s305; 300,000 decisions each) landed 01:51. **noop_s305
crossed too** — life 36: capped, 36 floor meals, intake 11.16 vs drain 3.0, the same
36 = 1,800 s ÷ 50.2 s regrowth as noop_s302 and noop_s304. The do-nothing body crossed
the sustained mark in 3 of 5 runs, every time by being born touching food and eating
each regrowth by contact. Floor total: 5 of 15 runs cross W 1 (gate ≤ 1), 5 sustained
lives in 1,515 floor lives (0.0033/life, unchanged); B 0 in 2,350.
The rule's report (below, `pilot_b_report --write`) reads **VOID**. No W or T is fixed by
this pilot. What the next pilot must change, from this record: W ≥ 2 or a spawn rule that
never starts him touching food (the floor's crossings are all spawn luck), and the cost
curriculum the record predicts makes B eat at all. The exploration arms e0–e2 (reported
only) are still in flight on Kaggle and are appended when harvested.

## Seen at the e0 harvest (11-17 05:07): white noise slides the same way gSDE did

e0 (bwhite_s301–s304: B with plain i.i.d. Gaussian action noise instead of row 3b's gSDE;
300,000 decisions each; reported only, ruling 28) landed 05:02. **0 sustained lives in 544**
(B's gSDE arm: 0 in 2,350). Same slide as B: meals per life first 20 → last 20 fell in 3 of 4
(1.05 → 0.20, 0.30 → 0.15, 0.45 → 0.25; s304 0.40 → 0.45), fed lives 60/20/40/40%
→ 12/12/15/21%, unfed life lengthened 267–282 s → 438–466 s in 4 of 4 (lie-still). Last-50k
meals per 1,000 decisions: bwhite mean 0.103 vs B 0.100 vs untrained twin (blr0) 0.374 and
held-8 random 0.454. White noise lived shorter (median 435 s vs B 496 s; 136 lives/run vs
117.5) — more motion, no more food. On four of five white seeds, the exploration noise is not
what keeps B from eating at this rung; the movement drain (§13) is the predicted reason.
bwhite_s305 and the pink arm (e1, e2) are still RUNNING on Kaggle (status 05:07).

## Results

Generated by `python -m experiments.pilot_b_report` from `/data/jack-logs/pilot_b/` (e0.json, f0.json, f1.json, f2.json, f3.json, k0.json, k1.json, k2.json, k3.json, k4.json). Registered runs in: **35 of 35**; exploration (reported only): 4 of 10.

### Reading by the rule: **VOID**

- B runs crossing the sustained mark by T = 300,000: W = 1: **0 of 20**, W = 2: **0 of 20**, W = 3: **0 of 20** (bar: ≥ 5 at W = 1).
- B runs crossing θ = 5 meals (secondary): W = 1: 10 of 20, W = 2: 3 of 20, W = 3: 1 of 20 (bar for EATS BUT STARVES: ≥ 5 at W = 3).
- Floor gate: 5 of 15 floor runs crossed (noop 3, held8 1, blr0 1); allowed ≤ 1 → **VOID**.
- Descriptive (harness `boschloo_p`, two-sided): B vs pooled floor p = 0.009; B vs lr-0 twin p = 0.201 (harness check #2: the handicapped copy is NOT told apart on the mark).
- Fixes: no W is fixed (rule: next are the exploration arms and the record's longer budgets, not a decisive test at this T).
- `prereg.judge` was not the door: its two-look `compare` needs 20 runs in each arm (floor 15, blr0 5); its three refusals were run here instead (hashes unmoved, pre-registration committed, #13 books balance).

### Per arm

| arm | runs in | runs crossing (sustained, W 1) | lives | % capped | life, median s | meals/life | meals per 1,000, last 50k (mean of runs) | ends |
|---|---|---|---|---|---|---|---|---|
| b | 20 of 20 | 0 | 2350 (117.5/run) | 0.0 | 496 | 0.40 | 0.100 | energy 2350 |
| noop | 5 of 5 | 3 | 537 (107.4/run) | 0.6 | 551 | 0.20 | 0.000 | capped 3 energy 534 |
| held8 | 5 of 5 | 1 | 485 (97.0/run) | 0.2 | 558 | 1.64 | 0.454 | capped 1 energy 483 integrity 1 |
| blr0 | 5 of 5 | 1 | 493 (98.6/run) | 0.2 | 535 | 1.06 | 0.374 | capped 1 energy 487 integrity 5 |
| bwhite | 4 of 5 | 0 | 544 (136.0/run) | 0.0 | 435 | 0.32 | 0.103 | energy 544 | *(reported only, ruling 28)*

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
| b | 317 | 115 | 0 | 0 | – | 0.35 → 0.40 | 0.40 → 0.29 | 20 → 14 | 289 → 492 | -59 → -70 | 0.153 | 1,835 / 0 / 0.00 | energy 115 |
| b | 318 | 117 | 0 | 0 | – | 0.80 → 0.30 | 0.40 → 0.31 | 40 → 21 | 457 → 511 | -59 → -68 | 0.114 | 555 / 0 / 0.00 | energy 117 |
| b | 319 | 118 | 0 | 0 | – | 0.85 → 0.30 | 0.20 → 0.24 | 20 → 21 | 286 → 520 | -79 → -74 | 0.171 | 812 / 3 / 3.44 | energy 118 |
| b | 320 | 122 | 0 | 0 | – | 0.20 → 0.10 | 0.40 → 0.13 | 20 → 13 | 327 → 488 | -59 → -85 | 0.039 | 1,087 / 0 / 0.00 | energy 122 |
| noop | 301 | 108 | 0 | 0 | – | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 553 → 553 | -99 → -99 | 0.000 | 2,658 / 0 / 0.00 | energy 108 |
| noop | 302 | 105 | 1 | 1 | 101790 | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 574 → 549 | -99 → -98 | 0.000 | 1,872 / 0 / 0.00 | capped 1 energy 104 |
| noop | 303 | 109 | 0 | 0 | – | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 562 → 564 | -99 → -99 | 0.000 | 60 / 0 / 0.00 | energy 109 |
| noop | 304 | 107 | 1 | 1 | 139714 | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 537 → 545 | -99 → -98 | 0.000 | 26 / 0 / 0.00 | capped 1 energy 106 |
| noop | 305 | 108 | 1 | 1 | 102822 | 0.00 → 0.00 | 0.00 → 0.00 | 0 → 0 | 522 → 535 | -98 → -98 | 0.000 | 319 / 0 / 0.00 | capped 1 energy 107 |
| held8 | 301 | 96 | 0 | 0 | – | 2.45 → 1.75 | 1.40 → 1.58 | 60 → 62 | 453 → 465 | 41 → 60 | 0.603 | 2,253 / 1 / 0.41 | energy 96 |
| held8 | 302 | 96 | 0 | 0 | – | 1.55 → 1.80 | 2.00 → 1.71 | 100 → 62 | – → 484 | 101 → 72 | 0.537 | 1,094 / 1 / 0.85 | energy 96 |
| held8 | 303 | 98 | 0 | 0 | – | 1.55 → 1.50 | 1.60 → 1.62 | 60 → 62 | 470 → 478 | 61 → 64 | 0.452 | 400 / 0 / 0.00 | energy 98 |
| held8 | 304 | 98 | 1 | 1 | 185780 | 1.85 → 0.65 | 2.40 → 0.58 | 80 → 33 | 482 → 461 | 141 → -40 | 0.313 | 2,393 / 4 / 1.55 | capped 1 energy 97 |
| held8 | 305 | 97 | 0 | 0 | – | 1.70 → 1.10 | 1.20 → 1.12 | 60 → 62 | 441 → 466 | 22 → 14 | 0.364 | 1,601 / 0 / 0.00 | energy 96 integrity 1 |
| blr0 | 301 | 92 | 0 | 0 | – | 1.70 → 0.70 | 0.80 → 0.61 | 40 → 26 | 282 → 500 | -19 → -38 | 0.319 | 2,035 / 6 / 2.74 | energy 92 |
| blr0 | 302 | 89 | 0 | 0 | – | 1.70 → 1.65 | 0.00 → 1.50 | 0 → 59 | 286 → 500 | -99 → 51 | 0.463 | 144 / 0 / 0.00 | energy 86 integrity 3 |
| blr0 | 303 | 108 | 0 | 0 | – | 0.70 → 0.55 | 0.80 → 0.52 | 40 → 30 | 282 → 510 | -19 → -47 | 0.134 | 2,448 / 0 / 0.00 | energy 108 |
| blr0 | 304 | 92 | 1 | 1 | 269274 | 0.90 → 1.85 | 0.60 → 1.65 | 60 → 43 | 269 → 518 | -39 → 71 | 0.637 | 1,290 / 0 / 0.00 | capped 1 energy 89 integrity 2 |
| blr0 | 305 | 112 | 0 | 0 | – | 0.40 → 0.80 | 0.40 → 0.79 | 20 → 39 | 373 → 480 | -58 → -20 | 0.319 | 1,509 / 0 / 0.00 | energy 112 |
| bwhite | 301 | 129 | 0 | 0 | – | 1.05 → 0.20 | 1.60 → 0.12 | 60 → 12 | 267 → 466 | 61 → -86 | 0.079 | 1,874 / 0 / 0.00 | energy 129 |
| bwhite | 302 | 139 | 0 | 0 | – | 0.30 → 0.15 | 0.20 → 0.18 | 20 → 12 | 277 → 465 | -79 → -81 | 0.058 | 1,759 / 0 / 0.00 | energy 139 |
| bwhite | 303 | 137 | 0 | 0 | – | 0.45 → 0.25 | 0.40 → 0.21 | 40 → 15 | 282 → 438 | -59 → -78 | 0.098 | 1,953 / 0 / 0.00 | energy 137 |
| bwhite | 304 | 139 | 0 | 0 | – | 0.40 → 0.45 | 0.40 → 0.29 | 40 → 21 | 272 → 453 | -59 → -69 | 0.177 | 1,603 / 0 / 0.00 | energy 139 |
