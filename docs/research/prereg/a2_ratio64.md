# FROZEN — Row A1-r64: brain A on A1's row at train ratio 64 instead of 512 (J28 cell 3, ruling 126)

Status: **FROZEN 2026-11-26 by this commit, before any seed or timing of this arm has run.** Nothing here
edits `experiments/brain_a/rows.py`'s frozen rule for A1–A3, `a2_size_s.md` or `a2_size_s_enc.md`.
The file name says `a2_` because J28 named it so; the row is **A1's** task and band (J28: "task and band
are A1's (the row A already passes)").

## Why this row exists

PUBLISHED_PRIORS §2, last-10% item 2: "Does A at train ratio 64 match 512 on proprio? No ablation exists
(Crafter, 64 → 16: about −20%, read off a figure)." §2 also records that cutting 512 → 32 on Walker Walk
cost about two thirds of the score (one user report). Ratio sets brain A's GPU bill almost linearly
(gradient steps = decisions × ratio / (batch × length)); `stage2_a_vs_b.md`'s free-cloud cost table is
costed at 512. If A holds a passing row at 64, A at stage 2 costs about an eighth of the gradient work.
This is a **test of the last 10%** the record leaves open, at the size of the row it is compared with
(3 seeds, the A1 instrument) — it is NOT a size the record says can show a small ratio effect; a hold
says "no large loss on an easy row", nothing finer.

Note against §2's wording: §2's item 2 proposed "one **A2** row at ratio 64 … in band at 500k, a miss at
100k". J28 (Director, 11-26 19:14) chose **A1** instead — the row A passes at 512 — so a miss reads as the
ratio, not as A2's open 1m miss. That choice is the order's and is kept.

## THE FROZEN RULE

- **Arm:** `rows.BASE` (size `1m`, the arm A1 passed on, byte for byte) with ONE override changed:
  `algo.replay_ratio=0.0625` (= train ratio 64 / (batch 16 × length 64); A1's is 0.5 = 512).
  Everything else as A1: 4 envs, action repeat 2, compile on, `fabric.precision` as A1 ran (32-true),
  learning_starts as A1 ran. The kernel prints its composed config; a printed `replay_ratio` other than
  0.0625, or any other key differing from A1's composed config, makes the seed VOID.
- **Task:** DMC `cartpole_balance`, proprio, 200K env steps (100K policy steps) — A1's.
- **Seeds 20, 21, 22.** **No extension** (J28). Nothing runs after the three seeds whatever they read.
- **Bands: A1's, unchanged** (`rows.py` ROWS["A1"]): 100K **690–898**, 200K **932–1000**.
- **"HOLDS" = every one of the 3 seeds inside A1's band at EVERY point** (J28: "in A1's band on 3/3").
  Anything else is **MISSES**, with the count of seeds in band at each point reported.
- **Score window — ruling 78's: the window the band was computed on.** Score at P = mean return of the
  training episodes that END in env steps [P − 30K, P) (`band_recompute.py`: the band is mean ± 2 pop SD of
  the official per-seed mean over the file's points {P−20K, P−10K, P}, bins [lo, hi) labelled by the right
  edge). **Printed beside, NOT read:** rows.py's frozen window (0.9 P, P]. A seed with no episode in the
  read window is VOID, not zero; a row with a VOID seed is NOT JUDGED and says so.
- **Control:** the never-trains twin — this arm's config, seed 20, `algo.learning_starts = 2 × total`
  (rows.py's control form: no gradient step). It MUST land below the band's lower edge at every point on
  the read window, else the row is VOID. (A1's twin read 312.5 / 296.9.)
- **Printed beside, deciding nothing:** each seed's score at both points next to A1's ratio-512 seeds
  (s0–s2: 826.3 / 843.5 / 677.0 at 100K; 973.8 / 994.1 / 984.3 at 200K, rows.py window), the 3-seed
  mean, gradient steps actually taken (from the log) and wall-seconds a seed.
- **Venue:** Kaggle GPU (T4 asked; the GPU given is printed), W48, behind the ear's s2 + twin (J28's
  launcher). ≈ ≤ 1 T4-h a seed (A1 at 512: 1,970–2,665 s a seed on a T4; 64 takes 1/8 of the gradient
  steps). A quota refusal stops the row and is written (ruling 124); nothing retries.

## READING — written before seed 20

| A1-r64 | reading |
|---|---|
| **HOLDS** (3/3 seeds in band at 100K and 200K) | brain A loses nothing measurable on an easy row at 1/8 the gradient work. `stage2_a_vs_b.md`'s cost table is re-costed at ratio 64 (draft, J28's branch); whether 64 holds on a HARD row (A2, his world) is not answered by this. |
| **MISSES at 100K only** | ratio 64 learns slower early on an easy task, as §2's own prediction for A2 said; the 200K reading is reported. The cost table stays at 512. |
| **MISSES at 200K** | ratio 64 costs score on the easiest passing row; the table stays at 512 and ratio 64 is not a stage-2 setting without a new ruling. |
| VOID / NOT JUDGED | said as that; no reading. |

## Settings that differ from the published recipe (ours beside theirs)

| setting | theirs (official v1, size S, `dmc_proprio`) | ours (this row) |
|---|---|---|
| train ratio | 512 | **64 — the thing tested** |
| implementation | official JAX v1 | sheeprl 0.8.5 `dreamer_v3_5` (v2 port), compile on — as A1 |
| size | S (18M) | `1m` (~0.64M params) — as A1; §2: no score exists for 1m |
| precision | float16 | 32-true — as A1 |
| everything else | as `rows.py`'s rule lists | as A1 |

## Debt carried — said before any seed

- **The 100K point is near-certain to miss on noise alone.** A1 itself, at ratio 512, had seed 2 at
  677.0 < 690 on rows.py's window, and its 3-seed mean was in band by only **1.2** on the 30K window this
  row reads (KNOWN_ANSWERS addendum 11-21 18:30). A per-seed 3/3 rule at 100K is stricter than the rule A1
  passed under (the seed mean). So a 100K miss here does NOT separate "ratio 64 is slower" from "A1's
  100K point is a coin toss for any single seed". The 200K point (A1's seeds 973.8–994.1 against a lower
  edge of 932) is the one that can tell. The rule is kept as J28 wrote it — stricter, never looser —
  and the READING table above reads the two points separately for this reason.
- n = 3 against a band built from 5 file seeds is coarse (as `a2_size_s_enc.md`'s debt).
- The run code: `rows.py` takes `base=` for another arm's settings, so the arm is
  `[o if not o.startswith("algo.replay_ratio=") else "algo.replay_ratio=0.0625" for o in rows.BASE]`
  passed as `base`; the kernel body is J28's launcher's to build. `rows.py`'s frozen A1–A3 rule is not
  edited.

## READ 11-28 09:35 — MISSES (`python -m experiments.brain_a.ratio64 score`, the frozen judge; raw `/data/jack-logs/brain_a/rows/a1r64_s2{0_c,1,2}.json`, output `/data/jack-logs/j28/r64_score_1128.json`)

| point | band | s20 | s21 | s22 | in band | twin (s20, never trains) | A1 at 512 (beside) |
|---|---|---|---|---|---|---|---|
| 100K | 690–898 | 297.7 | 288.7 | 287.3 | 0/3 | 313.0 | 826.3 / 843.5 / 677.0 |
| 200K | 932–1000 | 282.3 | 275.9 | 316.1 | 0/3 | 288.2 | 973.8 / 994.1 / 984.3 |

- **Ratio 64 does not hold A1's band — a miss, written as a miss; the stage-2 table stays at ratio 512.**
- Beside, deciding nothing: all three seeds sit at the never-trains twin's level (≈ 275–316 vs 288–313) — not "slower", **no measurable learning in ≈ 6k gradient steps** ((100K − starts) × 0.0625; A1 at 512 ≈ 49k). The raw keeps only the FIRST losses (step 5000: world-model 17.6, `Params/replay_ratio` 0.0496), so whether the world model was still fitting or the actor never moved cannot be told from these files. Kaggle wall: s21 1,045 s.
