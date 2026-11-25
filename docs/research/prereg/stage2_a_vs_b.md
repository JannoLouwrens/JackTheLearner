# Pre-registration DRAFT — stage 2: brain A against brain B in his world (ORDERS Next-up 4)

**Status: DRAFT, text only, binds nothing.** Written 2026-11-20 by a JACK slot
(Kaggle 5/5, no order could launch). Revised 2026-11-21 12:45 (ORDERS Next-up 1):
ruling 65 written in (mark, gate, futility and decision table read the LATE
half; "reached then lost" beside), ruling 68 written in (B's gate lists row 5
as a named miss), E2c's reading and ruling 64 put into gate 1. No seed of any arm below exists. The
Director freezes it; every `OPEN:` line is a choice left to that sitting.
Sources: `docs/research/STAGE2_PREDICTION.md` (the prediction, §n = its
sections), PUBLISHED_PRIORS §8 (test method) and §9 (battery), PLAN stage 2,
`docs/research/PILOT_B2.md` §4 (pilot 2's reading). Nothing new was read or run.

## The question (PLAN stage 2)

In his own world, at 1,000,000 decisions, which brain learns to live — A
(Dreamer-style world model, `experiments/brain_a/`) or B (SAC, the reproduced
replay learner) — and does each show the advantage the record predicts for it?
A home-made arm enters only if it passed both rows with one setting (J3,
ruling 59); none has today.

## Gate — this rule does not run until ALL hold (each a ready-list line)

1. **B eats somewhere in his world — in the LATE half of its run (ruling
   65).** Pilot 2 (16 rays, fixed α 0.1, 300k): every learner arm SLIDES /
   NEAR RANDOM, 0 sustained lives, late meals/life 0.08–0.16 vs
   never-learning K-lr0 0.82. E2c (25/25 read, ORDERS 11-21): handed the
   direction to food, B reaches whole sustained lives (4/5 fixed α, 3/5
   `auto`; a sense carrying nothing 0/5) — **and in 10 of 10 food runs has
   stopped eating by the end.** Under ruling 65 that is "reached then lost",
   NOT "eats": **gate 1 is unmet today.** STAGE2_PREDICTION §2: "If J11
   finds no setup where B eats, … stage 2 is premature." It waits on:
   - **the eye** (J17, `row128_vs_rays16.md`): a setup where B reaches food
     without being handed the direction, read on the late half; and
   - **keeping** (forge round 1, J18): a B setting whose late-half lives
     stay sustained on E-food.
   Both met → stage 2's world = that eye + that setting, B at
   `ent_coef="auto"` (ruling 64) unless round 1 seats another keep setting.
   Either missing → stage 2 does NOT start: a 1M race where no brain is seen
   still eating at the end compares two floors (and two forgettings).
2. **A reproduced** its three rows (J14: A1–A3 in band) and smoke-passed in
   his world (`in_world.py` DONE clauses (1)–(4) at ≥ 20k on the GPU). A
   miss on any row → fix A; stage 2 does not run with an unreproduced A.
   Today (11-21): A1 PASS, A3 PASS, A2 a named MISS at 100K (393.3 vs
   438–518, in band at 200K) — open until J14 (b)'s A2-v1S (size S) reads.
2b. **B reproduced (ruling 68).** Rows 1b, 2b, 3, 3b stand. **Row 5 is a
   named miss** (rule `0bf84989`: library arm 833.8 vs 841.7–864.3 at
   k = 15, one seed of 15 at 643.6; ours 862.6, not counted). Nothing of ours
   is in the library arm, so it orders no fix and does not block this gate;
   it is printed beside B's rows in every stage-2 report. **Row 6 has read
   (11-22): 6a swingup_sparse a named MISS (library 541.1 vs 726.9–826.1),
   6c reacher-hard a named MISS (824.9 vs 905.7–1000), 6b reacher-easy
   REPRODUCES (library 953.7, ours 958.1).** **Row B-6c-2 (11-25, ruling 70's
   one re-registration, fresh s5–9, stream-100): lib REPRODUCES reacher-hard,
   933.8 in 905.7–1000, ctrl ≤ 7.6** — 6c stays a named miss beside it.
   Rows 5, 6a, 6c are listed here
   as named misses and their extensions are not run (ruling 81). With three
   of the library recipe's last four rows missing on our apparatus, **B does
   not enter stage 2 until J19's reading is written** (`KNOWN_ANSWERS.md`
   B-6, addendum 11-22 07:25: OUR EVAL / AN EXPLORATION DRAW / WIN-THEN-LOSE
   / UNCLASSIFIED, per task, from the cause twin's saved models).
   **Gates 2 and 2b, one line a row (11-25, from `KNOWN_ANSWERS.md` § "Where
   each brain stands"; that table is the source — this block copies it, rules nothing):**
   - A1 cartpole_balance 1m — in band (782.3 / 984.1) → holds
   - A2 cartpole_swingup 1m — named miss at 100K (393.3 vs 438–518) → OPEN
   - A2-v1S size S — named miss ABOVE (610.0 / 835.3) → does not close A2
   - A2-v1S-enc matched size S — frozen, not run (5.0 T4-h a seed; Kaggle GPU from 11-28) → gate 2's open line
   - A3 ball_in_cup_catch 1m — in band (956.1) → holds
   - A smoke in his world — (1)–(4) hold → holds (not a record row)
   - B 1b Pendulum — in band (lib −309.3, ours −302.0) → holds
   - B 3 / 3b MountainCarContinuous — in band (lib 5/5, ours 5/5) → holds
   - B B-5 cartpole-swingup — named miss (lib 833.8) → listed, not blocking (ruling 68)
   - B B-6a swingup_sparse — named miss, the learner's (lib 541.1) → listed, not blocking
   - B B-6b reacher-easy — in band (lib 953.7, ours 958.1) → holds
   - B B-6c reacher-hard 10-ep — named miss (824.9) → listed beside B-6c-2
   - B B-6c-2 reacher-hard, re-registered — in band (lib 933.8) → holds
   **Gate 2 today: OPEN on A2 at the published size (A2-v1S-enc). Gate 2b: the reproduction rows
   hold; B's stage-2 entry still waits on gate 1 (J22 closed B on the wide eye for stage 1, 11-24).**
3. **The harness's thirteen self-checks** (PRIORS §8 "Reproduce first"
   #1–#13) pass through the real pipeline, each with its named test.
4. **The 20-run pilot** (PLAN: "the number is set by a 20-run pilot") is home
   and has frozen: T, θ, W, between-arm correlation (pair or not), and n.

## Arms

| arm | role | runs |
|---|---|---|
| `A` | candidate | n (pilot sets it; ≥ 20 unless the pilot proves fewer suffice — "fewer runs is not" a shortcut, PLAN) |
| `B` | candidate | n, same world seeds |
| `B@A-ratio` | B at A's update ratio — the matched-update control (§8 "World model vs model-free") | 5, reserve, only if A wins |
| `random`, `statue` (noop), `A-lr0`, `B-lr0` (never-trains twins) | floors — **must fail**: ≤ 1 success in total across them, else VOID | 3 each |
| `B-shuffled` | shuffled-input learner — must fail (content, not width) | 5 |
| `forager-f4` (hand forager, body eye + body motor; WORLD_CARD J12 (2′) 20/20, 7.53 meals/life at d 0.75 — on the row128 eye; no 16-ray body-frame forager row exists, `OPEN:` a ceiling on the race's own eye) | ceiling — **must pass**, else VOID | 3 |

Fresh seeds, never used in pilot or tuning (§8 Selection: 4–5× worse on
fresh seeds). No seed replaced; NaN and crashes are outcomes. Runs analysed
= runs launched.

## Primary outcome and test (ruling 65 over PRIORS §8)

- **Primary (ruling 65): the LATE half.** Per run, the share of lives
  STARTING at decision ≥ T/2 (= 500,000) that are SUSTAINED (ruling 22's
  mark, as J17 and forge round 1 read it); tiebreak late meals/life. A run
  "eats" iff that share clears θ_late, set by the pilot with the floor
  arms' late-half rate computed BEFORE (as pilot_b2 `rule-inputs` did).
  `OPEN:` θ_late, and whether θ is §8's 5 meals/life or SUSTAINED.
- **Reported beside, NOT read ("reached then lost"):** decisions until W
  consecutive SUSTAINED lives (§8's time-to-event, censored at T =
  1,000,000), first and last capped life, and the drop from the run's best
  window to its late half. A brain that eats at 50k and starves at 1M does
  not win on this column.
- Test: A vs B on the per-run late-half share, two looks (n/2 and n per
  arm), each at α 0.025; `OPEN:` permutation test on the share or Boschloo
  on runs that eat (§8 copies log-rank for time-to-event, which ruling 65
  demotes to the beside column; its permutation log-rank is still printed).
  Futility at look 1: ≤ 1 run eating in its late half in both arms → "not
  learnable to the end at this budget" (said so, no brain seated on
  eating). Floors: late-half eating only (≤ 1 across all floor runs).
- "Not distinguished" is written now to mean: **the eating race does not
  choose; the battery and the added sense do** (PLAN), then cost.

## The battery (STAGE2_PREDICTION §3; each with a must-fail control)

Run by both brains at the published budget, with the published random arm
as the control that must fail. Predicted ordering written now:

| task | prediction | counts toward A's advantage? |
|---|---|---|
| bsuite catch / cartpole | tie; random 0.00/0.04 must fail | harness only |
| deep_sea | both fail | no — no separation expected |
| mountain_car | tie | no |
| umbrella_length / discounting_chain | A ahead (0.78/0.40 vs 0.39/0.25) | **yes** |
| memory_len / memory_size | A ahead (0.65/0.59 vs 0.04/0.00) | **yes** |
| umbrella_distract | A ahead (0.74 vs 0.30) | yes |
| DMC swingup @100k | B ahead (835 vs 478) | B's advantage |
| an added sense (mid-life) | no published number — inference only | the tie-breaker; `OPEN:` its task and must-fail control (needs the eye, PLAN ready-list last line) |

"Shows its predicted advantage" = beats the other brain on the bsuite score
of that task with the seed count §8's power table needs for the published gap.
`OPEN:` seeds per battery task (the memory gap 0.65 vs 0.04 is large; the
distraction gap needs more).

## Decision table (PLAN, verbatim in effect)

"Eats" below = eats in the late half (ruling 65) everywhere.

- A passes known answers + eats within budget + shows memory/delayed-reward
  advantage; B does not eat → **A**.
- B eats, A does not, or A shows no predicted advantage → **B** until
  evidence says otherwise.
- Both eat, eating race not distinguished → battery + added sense decide;
  still tied → cost and simplicity (B).
- Neither eats by T → no seat moves; "on this world, at this budget, as
  configured".
- Any floor arm succeeds more than once, or forager-f4 fails → VOID.
- Either brain reaches and loses (beside column crosses, late half does
  not) → it does NOT eat; the loss is named in the report as the forge's
  KEEP problem, not hidden in a mean.

## Settings that differ from the published recipes (to be filled at freeze)

- A: `size1m`-class net, 10–100× below any published Dreamer score (§2);
  replay ratio as J14's rows ran it. B: `ent_coef` per J15's table (fixed 0.1
  has no published precedent, §19). World: movement drain 2–3× basal, no
  published eater lived under it (§13). Each is a reason a miss is weak.

## Cost (PLAN figures)

20 full A runs ≈ 160–280 T4-h (5–9 weeks of the free quota) or the Mac. B's
20 runs: under a week on CPU/Kaggle. `OPEN:` which of PLAN's four shortenings
(shorter T for all, lower A update ratio, free-running learner, Mac) — never
fewer runs.
