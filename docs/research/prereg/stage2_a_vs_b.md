# Pre-registration DRAFT — stage 2: brain A against brain B in his world (ORDERS Next-up 4)

**Status: DRAFT, text only, binds nothing.** Written 2026-11-20 by a JACK slot
(Kaggle 5/5, no order could launch). No seed of any arm below exists. The
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

1. **B eats somewhere in his world.** Pilot 2 (16 rays, fixed α 0.1, 300k):
   every learner arm SLIDES / NEAR RANDOM, 0 sustained lives, late
   meals/life 0.08–0.16 vs never-learning K-lr0 0.82. STAGE2_PREDICTION §2:
   "If J11 finds no setup where B eats, the 100k–200k row is void for his
   world, and stage 2 is premature." So stage 2 waits on J15's E2c table:
   - auto-food eats → stage 2's world = K's world, B at `ent_coef="auto"`.
   - both food arms miss at 300k → stage 2 does NOT start; the forge's
     run-length round (F3) answers first. Running a 1M race where no brain
     has been seen eating at any length compares two floors.
2. **A reproduced** its three rows (J14: A1–A3 in band) and smoke-passed in
   his world (`in_world.py` DONE clauses (1)–(4) at ≥ 20k on the GPU). A
   miss on any row → fix A; stage 2 does not run with an unreproduced A.
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

## Primary outcome and test (PRIORS §8, copied, not invented)

- Outcome: decisions until W consecutive lives each SUSTAINED (pilot 2's
  reading, ruling 22's mark); unfinished runs censored at T = 1,000,000.
  `OPEN:` §8 states θ = 5 meals/life; pilot 2 used SUSTAINED at W = 2. One
  must be chosen, with the floor arms' crossing rate at that W computed
  BEFORE (as pilot_b2 `rule-inputs` did).
- Test: A vs B, permutation log-rank, two looks (n/2 and n per arm), each at
  α 0.025. Sensitivity: Boschloo on runs crossed by T. Futility at look 1:
  ≤ 1 run past threshold in both arms → "not learnable at this budget"
  (said so, no brain seated on eating).
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

- A passes known answers + eats within budget + shows memory/delayed-reward
  advantage; B does not eat → **A**.
- B eats, A does not, or A shows no predicted advantage → **B** until
  evidence says otherwise.
- Both eat, eating race not distinguished → battery + added sense decide;
  still tied → cost and simplicity (B).
- Neither eats by T → no seat moves; "on this world, at this budget, as
  configured".
- Any floor arm succeeds more than once, or forager-f4 fails → VOID.

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
