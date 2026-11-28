# Pre-registration DRAFT — stage 2: brain A against brain B in his world (ORDERS Next-up 4)

**Status: DRAFT, text only, binds nothing.** Written 2026-11-20 by a JACK slot
(Kaggle 5/5, no order could launch). Revised 2026-11-28 13:15 (J31 (2)+(5): the race world's memory condition — a dark-eye probe with a keep-heading floor memoryless B must not beat; every OPEN listed with its one row). Revised 2026-11-28 (J31 (1)+(3): ruling 137's verdict clause written into the decision table — no seat on the visible-food race alone, the battery's memory and delayed-reward rows and B-mem read beside it; ruling 138's death observation written into the arms). Revised 2026-11-26 (J27: B = forge keep setting, B-mem arm, free-cloud cost table, OPEN rows; the Mac struck). Revised 2026-11-21 12:45 (ORDERS Next-up 1):
ruling 65 written in (mark, gate, futility and decision table read the LATE
half; "reached then lost" beside), ruling 68 written in (B's gate lists row 5
as a named miss), E2c's reading and ruling 64 put into gate 1. No seed of any arm below exists. The
Director freezes it; every `OPEN:` line is a choice left to that sitting.
Sources: `docs/research/STAGE2_PREDICTION.md` (the prediction, §n = its
sections), PUBLISHED_PRIORS §8 (test method) and §9 (battery), PLAN stage 2,
`docs/research/PILOT_B2.md` §4 (pilot 2's reading). Nothing new was read or run.

## The question (PLAN stage 2)

In his own world, at 1,000,000 decisions, which brain learns to live — A
(Dreamer-style world model, `experiments/brain_a/`) or B (the forge's keep setting on the reproduced
replay learner, J27 (1); B-mem, the same with memory, beside it) — and does each show the advantage the record predicts for it?
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
| `B` | candidate — **the forge's KEEP setting once round 1-L is read** (J27 (1); the BRO / SimBa / MR.Q class, PRIORS §1 "B+"), never plain SAC: on 40 tasks plain SAC reads IQM 0.007 / 0.316 at 100k / 1M vs BRO 0.254 / 0.945 (§1, 2405.16158), and at rung M plain was the ~6% floor vs resets 84% / TQC 82%. `OPEN:` which keep setting, until 1-L's read | n, same world seeds |
| `B-mem` | candidate — **B's keep setting + a recurrent core** (J27 (2)): A has a GRU and B has none, so A vs B alone confounds memory with the world model. The record's form: recurrent SAC/TD3 (Ni 2021, 2110.05038: LSTM-128, separate actor/critic encoders, whole-episode sequences; PRIORS §5 row 7, §28 "smallest: recurrent TD3/SAC, 64 units"); on Craftax-classic a GRU added +8.1 points at no extra time and beat a 201M DreamerV3 at 1M (55.5 vs 47.2, §1, 2502.01591). Replay stores the recurrent state, burn-in, never crosses a death (§6, R2D2/R2I). **Its known-answer row, named now: `B-7` velocity-only Pendulum, recurrent SAC, Ni's configuration, 51,000 steps** — band from the figure (separate arm ≈ −170 at 50k; the "−168.6 ± 3.5" is UNVERIFIED, §5 row 7 correction), shared-encoder arm the must-be-worse comparison. Row `B-7` in `KNOWN_ANSWERS.md` **REPRODUCES** (11-28: arm −172.6, twin −1490.2) — the recurrent core is `experiments/brain_b_mem.py` | n, same world seeds |
| `B@A-ratio` | B at A's update ratio — the matched-update control (§8 "World model vs model-free") | 5, reserve, only if A wins |
| `random`, `statue` (noop), `A-lr0`, `B-lr0`, `B-mem-lr0` (never-trains twins) | floors — **must fail**: ≤ 1 success in total across them, else VOID | 3 each |
| `B-shuffled` | shuffled-input learner — must fail (content, not width) | 5 |
| `forager-f4` (hand forager, body eye + body motor; WORLD_CARD J12 (2′) 20/20, 7.53 meals/life at d 0.75 — on the row128 eye; no 16-ray body-frame forager row exists, `OPEN:` a ceiling on the race's own eye) | ceiling — **must pass**, else VOID | 3 |

**Death observation, every arm (ruling 138).** Every arm's world passes
`terminal_obs="dying"`: the death step carries the dying body's needs and
proprio, not the reborn body's (`tests/test_death_interface.py`; the control
`reborn` fails). Where the world hands a sense (`pilot_b2`'s `handed`
cells), `HandedSense(death_extra="dying")` too (`tests/test_handed_death.py`).
J30 and round 1-L ran `reborn` (harmless to SAC's terminal-masked target, not
to a world model that learns from the death step); no arm here does.

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

## The race world's memory condition (ruling 137 (b), J31 (2) — DRAFT; nothing built or run)

**Three forms weighed against the record (PRIORS §6, §9; long form
`$N/capability_battery.md` §3, `$N/memory_sleep_continual.md`):**
- *Food that goes behind a wall* — REJECTED: on POPGym "the MLP received
  the highest score on almost all navigation tasks, beating all memory
  models", because wall-following covers a maze without memory (2303.01859);
  the memoryless arm is expected to PASS by search, so no control would fail.
  It also needs new walls in the locked world files.
- *A remembered place across a life* — REJECTED: with places fixed, weights
  trained from replay carry "where the tree was" and a recurrent agent with no
  store reaches human level (Mirowski 2017, 1611.03673) — a memoryless arm
  passes by weights; with `random_layout=True` (every race run) the layout is
  re-drawn every life, so nothing across lives is rememberable, and with 16
  rays and no heading re-localising after rebirth has no published example.
- *A cue, a delay, then the meal* — CHOSEN, in the in-world form the record
  names: Animal-AI's level 7, "acting through lights-out periods"
  (2312.11414: Dreamer-v3 above chance on it, PPO "at or near to chance",
  25M pixel steps, text only, no number), built on the passive T-Maze's
  single axis (memory length D with credit length 1; best memoryless return
  0.5 by construction, Ni 2023, 2307.03864).

**The form.** During the race's own training and evaluation, at random
decisions (rate ρ a decision, `OPEN:` ρ; inference 1/500) the eye reads
DARK — every eye channel set to its no-hit value — for D decisions;
needs, proprio, touch and the mouth stay on. The cue is the food seen at
onset. D is a difficulty axis, never one pass mark (the battery rule,
`$N/capability_battery.md` l.705/726): D ∈ {8, 16, 32, 64}, drawn per
blackout. World option, NAMED NOT BUILT: `jack_env.py`
`eye_blackout: Optional[tuple[float, tuple[int, ...]]] = None` (ρ, D set);
`None` = today's world byte for byte (the option rule; a planted test that
the default equals today's stream comes with it).

**Score (per blackout begun with a floor food inside ray range).** Distance
closed to THAT food over the dark window, divided by its onset distance
(1 = reached it in the dark; ≤ 0 = moved away). Per run, the mean over
blackouts in the LATE half (ruling 65), by D.

**The floor that must not be beaten by sight-free motion (the must-fail
control).** `keep-heading`: a script with no memory that, from the dark
step on, repeats the last lit action. Memoryless `B` must NOT beat it at
D ≥ 16 (per-run means, B's late half vs the script on the same frozen
onsets): if it does, motion state carries the cue and the condition is
VOID — it then measures momentum, not memory. A second floor: each brain's
`-lr0` twin.

**Published number for each class (memory length it should hold):**
| arm | record | predicted here |
|---|---|---|
| `A` (GRU-RSSM, 64-step training chunks) | bsuite memory_len 0.65 = success to ~25 steps; R2I: DreamerV3 learns dependencies "up to 30 environment steps"; chunks shorter than the dependency give zero (R2I, 2403.04253) | beats keep-heading at D 8–32, not at 64 |
| `B` (no core) | memory_len 0.04, memory_size 0.00 | ties keep-heading at every D ≥ 16 (that tie IS the control) |
| `B-mem` (LSTM-128, Ni) | recurrent A2C memory_len 0.70 (30-step window); LSTM falters at T-Maze 250; recurrent SAC Key-to-Door 0.996 at 4M (2110.05038); ours B-7 REPRODUCES | as A at D ≤ 32, IF its replay chunk ≥ D — `OPEN:` B-mem's port trains on whole 200-step Pendulum episodes; his lives run 10^3–10^4 decisions, so chunked stored-state replay with burn-in (§6, R2D2 40 of 80) must be built and pass B-7 again first |

Read beside the battery's memory rows, in the same verdict (ruling 137):
an arm "holds the dark" at D iff its late-half score beats keep-heading's
in ≥ the share §8's power table needs for the gap the pilot measures
(`OPEN:` that share, from the validation row below).

**Cost.** The probe rides the race's own runs: no extra training run; the
eye is dark ≈ ρ·E[D] ≈ 6% of decisions at ρ 1/500 (inference), which lowers
every arm's eating alike. Before the race, ONE validation row that can
fail: J30's sighted cell + `eye_blackout`, memoryless B vs keep-heading, 3
seeds × 500k ≈ 3 kernels × 6.5 h ≈ 20 kernel-h (J30's timing), and
B-mem the same 3 seeds once its chunked replay passes B-7 (timing cell
first, ruling 86). If memoryless B beats keep-heading there, the form is
re-drawn before any race seed (VOID before evidence, not after).

## Decision table (PLAN, verbatim in effect)

**Ruling 137 (owner, 11-28) — binds every line below.** No brain takes the
seat on the visible-food race alone. The verdict reads, in the SAME report
and the same sitting: the eating race (primary above), the battery's memory
rows (memory_len / memory_size) and delayed-reward rows (umbrella_length /
discounting_chain), and the race world's own memory condition (§ "The
race world's memory condition" above — the dark-eye probe, with its
must-fail floor). `B-mem` is read beside `B` on every one of them, so a win by A
on memory is not credited to the world model when a recurrent B matches it.
A brain that wins the eating race and loses every memory row is NOT seated
on eating; the line below that applies is the one the memory rows select.

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
  replay ratio as J14's rows ran it (512, `rows.py:42`); compute `32-true`
  where the official is float16. B: the forge's keep setting (1-L), fixed
  0.1 entropy has no published precedent (§19). B-mem: Ni's LSTM-128 vs
  Craftax's GRU-256 — `OPEN:` which, raced on `B-7` first. World: movement
  drain 2–3× basal, no published eater lived under it (§13). Each is a
  reason a miss is weak.

## Cost — FREE CLOUD ONLY (J27 (3), ruling 113: the Mac is parked; every Mac hour struck)

Free compute = Kaggle GPU, **30 T4-h a week, expiring weekly**, one
session ≤ 12 h (a longer seed is a carried chain, `save_life`). B and
B-mem run on Kaggle CPU kernels and the box (forge rung M: 300k × 5 seeds
inside round 1's 720 core-h; 1M ≈ 3.3× a seed (inf)) — they spend no GPU
quota. A's per-seed hours are §1's table, inference from two of our
timings (5.0 T4-h per 100k at size S; `rows.py:46` size1m 25.9 steps/s):
**1M decisions ≈ 50 T4-h a seed at size S, ≈ 11 T4-h at `size1m`.**

| A arm, 1M decisions | runs | T4-h a run | T4-h total | weeks of the 30 h quota |
|---|---|---|---|---|
| size S, ratio 512 — the race (n = 20) | 20 | 50 | 1,000 | **33.3** |
| size S — `A-lr0` floor (no updates; full cost an upper bound) | 3 | ≤ 50 | ≤ 150 | ≤ 5.0 |
| size S — A's half of the 20-run pilot | 10 | 50 | 500 | 16.7 |
| **`size1m`, ratio 512 — the race (n = 20)** | 20 | 11 | 220 | **7.3** |
| `size1m` — `A-lr0` floor | 3 | ≤ 11 | ≤ 33 | ≤ 1.1 |
| `size1m` — A's half of the pilot | 10 | 11 | 110 | 3.7 |
| ~~`size1m` at ratio 64~~ STRUCK — OPEN 1 MISSED 11-28 (only if OPEN 1 holds; ~8× fewer updates, inference) | 20 | ~2–3 | ~40–60 | ~1.5–2 |

**What the quota affords:** size S does NOT — pilot + race + floor ≈ 55
weeks of the whole quota, and one seed (50 h) is more than a Kaggle week.
**`size1m` does, barely:** pilot + race + floor ≈ 12 weeks if A takes the
whole quota and nothing else (ear, A2 rows, forge) spends a GPU hour; at
half the quota ≈ 24 weeks. The catch: `size1m` has NO published score
anywhere (§1 "does not answer" 3) — our A1–A3 are the only rows, and A2 at
the published size S is still open (gate 2). The affordable A is the
unreproduced A. PLAN's shortenings that survive without the Mac: shorter T
for all arms, a lower A update ratio (OPEN 1), a free-running learner —
never fewer runs (§8).

## OPEN — each with the one cheap row that answers it (J27 (4); nothing run)

1. **Does ratio 64 hold A's score on proprio?** No ablation exists (§1
   "does not answer" 2; Crafter 64 → 16 ≈ −20%, read off a figure). Row:
   one A2 (cartpole_swingup) row at ratio 64, ~1 T4-h a seed (inf), 5
   seeds vs the A2-v1S-enc band. Prediction (§1): in band at 500k, a miss
   at 100k. If it holds, A's race cost falls ~5–8× (table's last line).
   **READ 11-28 (J28, `prereg/a2_ratio64.md` § READ, frozen judge): MISSES.**
   A1 cartpole at ratio 64, 200K: s20/21/22 282.3 / 275.9 / 316.1 vs band 932;
   never-trains twin 288.2 — no measurable learning in ≈ 6k updates. The cost
   table stays at ratio 512; its `size1m` ratio-64 line is struck (its
   condition failed).
2. **Mixed precision for A on a T4.** sheeprl's maintainers read 114 → 32
   ms a gradient step on an A100 (issue 336); fp16 on a T4 is unmeasured.
   Row: a 20-minute timing cell — `rows.py` size S `16-mixed` vs `32-true`,
   same seed, steps/s and the first 5k losses side by side.
   **READ 11-28 (J28 cell 2, Kaggle `jack-ladder-1795836553`, raw
   `/data/jack-logs/brain_a/rows/fp16_timing.json`):** `32-true` A2-v1S-enc
   s10 = **165.3 s per 1k decisions** (sps_train 3.04; 4.55 h a seed
   projected). `16-mixed` **crashed at the first update**: Lightning
   `amp.py unscale_gradients` → `NotImplementedError: Gradient clipping is
   not implemented for optimizers handling the unscaling` (sheeprl's
   world-model `clip_gradients`). fp16 is unmeasured, not slower; the table
   stays at fp32. Billed 0.66 h vs 0.5 declared.
3. **B-mem reproduces** (`B-7` above): one Kaggle CPU kernel, 5 seeds ×
   51k steps, before `B-mem` is a candidate.
   **DONE 11-28 (J29):** B-7 REPRODUCES — `experiments/brain_b_mem.py`
   (port of `pomdp-baselines@e7c19c32`, separate LSTM-128 SAC), 4 seeds,
   arm −172.6 in −190.9…−154.5; memoryless twin −1490.2. `B-mem`'s
   recurrent core is that file; it enters as a candidate (draft — a sitting
   freezes this page).
4. **Which B keep setting** — round 1-L's read (J18), not this page.

### Every `OPEN:` still on this page, with the one row that answers it (J31 (5), 11-28)

| # | open line (where) | the one cheap row | cost | waits on |
|---|---|---|---|---|
| 4 | B's keep setting (Arms `B`) | round 1-L look 1, the frozen keep rule (J18) | 0 new (running) | g1-5/6/7, ≈ 11-29 |
| 5 | gate 2: A at the published size (Gate 2) | A2-v1S-enc seeds 10–12, scored once (J14) | 0 new (seed 12 queued) | seed 12 |
| 6 | the ceiling on the race's own eye (Arms `forager-f4`) | the hand forager on `bins16`/kind paint, body frames, 3 seeds × 1 life-set — a box row, no learner | < 15 CPU-min (inf) | nothing |
| 7 | θ_late, and SUSTAINED vs 5 meals/life (Primary) | `pilot_b20` (J32) gives the late-meal spread and the blind-twin floor; floors' late-half rate computed BEFORE, as `pilot_b2 rule-inputs` did | in J32 | J32's freeze + run |
| 8 | permutation on the share vs Boschloo on runs that eat (Test) | none to run: a choice for the sitting; `pilot_b20_read` already carries an exact Boschloo equal to SciPy's | text | sitting |
| 9 | the added sense: task and must-fail control (Battery) | none on record (PRIORS §9 "last 10%" design is inference): one draft row, the switch-on design with a same-shape noise channel as the must-fail twin | text | the eye's stage-1 seat (J30, held) |
| 10 | seeds per battery task (Battery) | §8's power table on each published gap (memory 0.65 vs 0.04 large; distraction 0.74 vs 0.30) — arithmetic, no run | text | nothing |
| 11 | B-mem core: Ni's LSTM-128 vs GRU-256 (Settings) | race both on `B-7` (velocity-only Pendulum, 4 seeds, 51k) — the GRU arm only; LSTM is home | 1 Kaggle CPU kernel | a free session |
| 12 | B-mem in his world: chunked stored-state replay + burn-in (Memory condition) | build it, re-pass `B-7` with chunk 64 / burn-in 32 on the same frozen band | 1 kernel after build | the build |
| 13 | ρ and the "holds the dark" share (Memory condition) | the validation row: J30's cell + `eye_blackout`, memoryless B vs keep-heading, 3 seeds × 500k | ≈ 20 kernel-h | `eye_blackout` built + its default test |

Rows 1–3 above are closed (1 MISSES, 2 unmeasured — fp16 crashed, fp32
stands; 3 DONE). Cheapest first: 10, 8 (text), 6 (box), then 11, 12, 13.
