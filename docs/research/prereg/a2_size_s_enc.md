# FROZEN — Row A2-v1S-enc: A2-v1S with the published encoder/decoder size (J14 (a), ruling 78)

Status: **FROZEN 2026-11-24 by this commit, on ruling 78's pins (`docs/DECISIONS_RESOLVED.md` §78). No seed
has run and no timing cell has run.** The only thing allowed before seed 10 is the timing cell (rulings
86, 96, 99): a few thousand policy steps, gradient steps/s printed, **no score read**. Nothing here edits
`rows.py`'s frozen rule for A1–A3 or `a2_size_s.md` (A2-v1S stays a named miss, ABOVE).

## Why this row exists

A2-v1S was meant to be "the published agent at the published size" and was not: the official v1
`dmc_proprio` block shrinks layers with `.*\.layers: 2` / `.*\.units: 512`, which the official
`Config.update` `re.match`es against flat keys — `encoder.mlp_layers` / `encoder.mlp_units` (and the
decoder's) carry no literal `.layers` / `.units`, so they keep the defaults **5 × 1,024**. sheeprl's
`dreamer_v3_S` shrinks them to 2 × 512. A2-v1S read **610.0 @100K, 835.3 @200K, ABOVE both bands** (3
seeds, KNOWN_ANSWERS 11-21 23:41). This row asks the one question left: at the published size, does our
build land in the published band?

**Correction of the frozen A2-v1S file** (stated here, not edited there): `a2_size_s.md`'s "A2-v1S size —
… Matches; nothing to name" was WRONG for the encoder and decoder. The heads, actor, critic and RSSM do
match (2 × 512, deter 512); the encoder and decoder do not (published 5 × 1,024, ours 2 × 512).
Evidence: KNOWN_ANSWERS "Row A2-v1S — ADDENDUM 11-21 19:15", `experiments/brain_a/v1s_diff.py`.

## THE FROZEN RULE

- **Arm:** `rows.BASE_V1S` + `experiments.brain_a.v1s_diff.MATCHED_ENC`, the four targeted keys and
  nothing else:
  `algo.world_model.encoder.mlp_layers=5`, `algo.world_model.encoder.dense_units=1024`,
  `algo.world_model.observation_model.mlp_layers=5`, `algo.world_model.observation_model.dense_units=1024`.
  Re-checked at freeze (11-24 15:30, probe venv, configs only): `v1s_diff --matched-enc` → **BLAST 4
  keys moved by 4 overrides; 53/56 SAME; DIFFERS: precision, replay size, pretrain steps.** The global
  `algo.dense_units=1024` is forbidden (it moves 8 keys and breaks 6 matching rows — addendum 19:35).
- **Task:** DMC `cartpole_swingup`, proprio, the same as A2 / A2-v1S.
- **Pins, as A2-v1S (ruling 67):** `fabric.precision=32-true`; `algo.learning_starts=1024` (official
  prefill count); 4 envs, action repeat 2, `replay_ratio` 0.5, compile on. The kernel prints its
  composed config; a printed value differing from this list makes the seed VOID.
- **Seeds 10, 11, 12.** ONE extension, stated now: seeds 13, 14 (fresh) if the row misses at 3; judged
  once more at 5, same bands. Nothing after that: a miss at 5 is a named miss.
- **Bands unchanged** (`rows.py` ROWS["A2"]): 100K **438–518**, 200K **740–816**. The row is IN BAND when
  its seed mean is inside the band at EVERY point (rows.py's rule).
- **Score window — ruling 78: the window the band was computed on.** The band is mean ± 2 pop SD of the
  official file's per-seed mean over its points {P−20K, P−10K, P}; the file's points are np.histogram
  bins `[lo, hi)` labelled by the right edge (`band_recompute.py`), so those three points cover env
  steps **[P − 30K, P)**. Our score at P = mean return of the training episodes that END in
  [P − 30K, P). (cartpole episodes are fixed-length, so the episode mean equals the mean of the three
  10K-bin means up to one episode at a bin edge.) **Printed beside, NOT read:** the frozen rows.py
  window (0.9 P, P] and the half-open twin (P − 30K, P]. A seed with no episode in [P − 30K, P) is VOID,
  not zero; a row with a VOID seed is not judged and says so.
- **Control:** the never-trains twin — this arm's config, seed 10, `algo.learning_starts = 2 × total`
  (no gradient step; in the v1 runner it plays uniform-random actions). Seed 10 rather than A2-v1S's
  seed 0 so the control is the twin of THIS row's first seed. It MUST land below the band's lower edge
  at every point on the scored window, else the row is VOID.
- **Order:** timing cell → seed 10 → 11 → 12, one seed per session; the control rides with seed 10.

## Venue (J14 (b), ruling 96) — decided by the timing cell, not here

The world model is 7.49M → **16.43M** parameters (×2.19; `v1s_params`), so A2-v1S's ~4.3–5.1 T4-h a
seed is NOT this arm's cost; nobody has timed it.
1. **Timing cell on Colab T4** (≤ 0.5 h, seed 10, a few thousand policy steps): prints gradient
   steps/s and the projected T4-h for 100K policy steps (= 200K env steps). No score read.
2. Projection **≤ 3 h a seed** → the three seeds go on Colab, one at a time, this week.
   **3–9 h** → each seed saves and resumes through `bbd54926`'s checkpoint (replay carried) across
   Colab sessions, or waits for Kaggle's week from Sat 11-28.
   **> 9 h a seed anywhere** → one INBOX line, no seed.
3. A seed is never shortened: the 200K point needs the full 200K env steps.

## READING — written before seed 10

| A2-v1S-enc (seed mean, both points) | reading |
|---|---|
| **in band** | the build is right at the published size. A2's 1m miss @100K is SIZE (a known limit of 1m, not a build fault), and A2-v1S's excess was its smaller encoder/decoder. `prereg/pilot_a.md` proceeds on ruling 62's terms. |
| **miss (above or below)** | a BUILD fault remains with only size matched: diff ours against official `ec526b9` setting by setting — the three named DIFFERS first (precision float16 vs 32-true; replay 1e6 vs 5e5; pretrain 1 vs 0), then the code items the 11-21 addendum did NOT check (weight init, replay sampling order, reward/action row alignment). No A pilot claims the build before it is found. |
| VOID / NOT JUDGED | said as that; no reading. |

The reading is the same whichever side of the band a miss lands on; the direction is reported.

## Settings that differ from the published recipe (ours beside theirs)

| setting | theirs (danijar/dreamerv3 `ec526b9`, `defaults` + `dmc_proprio`) | ours (this row) |
|---|---|---|
| implementation | official JAX v1 | sheeprl 0.8.5 PyTorch port of v1, compile on |
| precision | float16 | 32-true (ruling 67: same as A1–A3 and A2-v1S) |
| replay size | 1e6 | 500,000 (cannot bind: a 200K-env-step run stores at most 200K transitions) |
| pretrain steps | 1 | 0 (one gradient step at the start) |
| encoder / decoder | 5 × 1,024 | **5 × 1,024 — matched by this row** |
| everything else in `v1s_diff`'s 56-row table | — | SAME (53/56) |

## Debt carried

- n = 3 seeds against a band of mean ± 2 SD from 5 file seeds is a coarse instrument (11-21 19:15
  addendum: the 95% one-new-seed interval is wider than the band). The rule is the rows.py rule
  unchanged; the one-seed interval and "above k of 5 file seeds" are printed beside, not read.

## Addendum 11-24 19:30 — timing cell READ (rule 2 above; no score read)

Colab T4, seed 10, 6,000 policy steps, rc 0, wall 1,710 s (`/data/jack-logs/brain_a/rows/v1s_enc_timing.json`;
torch 2.6.0+cu126 — the `bcfbd0e6` re-pin held on its first Colab run; Colab's mujoco 4.0.1, sheeprl 0.8.5).
First window dropped (compile). Gradient steps/s 2.76 (4 windows, 2.755–2.770), env steps/s 1,035.
Projection: (100,000 − 1,024) × 0.5 / 2.76 + 100,000 / 1,035 = 17,930 + 97 s = **5.0 T4-h a seed**
(A2-v1S's cell: 3.24 g/s, 4.26 h — 2.19× the world model costs only 15% more time).
**Branch, by rule 2: 3–9 h.** Each seed either resumes across Colab sessions through `bbd54926`
(which has never crossed a gradient step — it must first be shown bit-continuous mid-training) or
runs whole in one Kaggle GPU session from Sat 11-28 (5 h of a 12-h session; 3 seeds ≈ 15 of W48's 30 h).
