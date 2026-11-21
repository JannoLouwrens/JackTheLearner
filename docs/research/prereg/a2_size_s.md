# FROZEN — "Fix A": row A2 at the published size, arm A2-v1S only (J14 (b), ruling 67)

Status: **FROZEN 2026-11-21 by ruling 67 (`docs/DECISIONS_RESOLVED.md`) — this commit is the freeze. No
seed has run; only a timing cell (no score read) runs before seed 0.** Drafted 05:40 UTC by a JACK
slot; the pins below are the Director's. Nothing here edits `rows.py`'s frozen rule for A1–A3.

## THE FROZEN RULE (ruling 67's pins; where the draft below differs, this section binds)

- **Arm: A2-v1S ONLY.** sheeprl `exp=dreamer_v3_dmc_walker_walk` + `algo=dreamer_v3_S` (the v1 agent
  at size S), DMC `cartpole_swingup`, proprio. **The 12m arm is STRUCK for this week** (it costs more
  than the week holds and 12m is not his size).
- **Pins:** `fabric.precision=32-true` (as A1–A3 ran, so ONLY agent+size move against our rows; the
  official float16 is written under "ours beside theirs"); `algo.learning_starts=1024` (the official
  prefill count). Everything else as the frozen row: 4 envs, repeat 2, `replay_ratio` 0.5, compile on.
- **Seeds 0, 1, 2.** ONE extension, stated now: seeds 3–4 if the row misses at 3; judged once more at
  5, same bands. Nothing after that.
- **Bands unchanged** (`rows.py` ROWS["A2"]): 100K 438–518, 200K 740–816; score window (0.9 P, P]
  training-episode returns; a seed with no episode in a window is VOID, not zero.
- **Control unchanged:** the never-trains twin, seed 0, `learning_starts = 2 × total`; must land
  below the band's lower edge at every point, else VOID. (Named: in the v1 runner a run that never
  reaches `learning_starts` plays uniform-random actions; the v2 runner of A1–A3 played its untrained
  policy. Neither takes a gradient step.)
- **Order:** timing cell first (seed 0, a few thousand policy steps, prints gradient steps/s; NO
  score is read from it), then one seed per kernel.
- **READING — written before seed 0:**
  - **in band @100K** (and the row's verdict by the rule above) → the build is right and the 1m miss
    is SIZE, a known limit. Then `prereg/pilot_a.md` is drafted on ruling 62's terms (text only).
  - **miss** → a BUILD fault: diff ours against official `ec526b9` setting by setting. No A pilot
    before it is found.
  - VOID / NOT JUDGED → said as that; no reading.

## Why

Row A2 (DMC cartpole_swingup) is a **named miss @100K at 5 seeds** (mean 393.3 vs band 438–518) and
in band @200K (788.2). A1 and A3 pass. `rows.py` names exactly two settings where our build differs
from the published protocol: (1) the agent — sheeprl `dreamer_v3_5` (v2/Nature port) vs the v1 agent
of the 2023 score files; (2) the size — `size1m` (~0.65M parameters) vs size S. Its rule said a miss
"can mean a broken build OR that 1m is too small". This draft separates the two. It is not a re-try of
the row at a looser band.

## What was checked on the box, 05:35 (hydra compose only, no training)

Both arms compose with the row's overrides, and nothing else needs changing (proprio `state` key,
4 envs, repeat 2, `replay_ratio` 0.5 = train ratio 512 at batch 16 × 64):

| arm | exp | algo | agent | dense | deter | discrete | learning_starts |
|---|---|---|---|---|---|---|---|
| (row as run) | `dreamer_v3_5_dmc_walker_walk` | `dreamer_v3_5_1M` | v2 | 64 | 512 | 4 | 1024 |
| **A2-v1S** | `dreamer_v3_dmc_walker_walk` | `dreamer_v3_S` | v1 | 512 | 512 | 32 | 1300 (exp file) |
| **A2-12m** | `dreamer_v3_5_dmc_walker_walk` | `dreamer_v3_5_12M` | v2 | 256 | 2048 | 16 | 1024 |

sheeprl's own v1 DMC exp file defaults to `dreamer_v3_S`, so A2-v1S is the port's own reading of the
published protocol. v2 has no "S" size; 12m is the smallest v2 size with published scores (§2).

## Arms (as drafted — A2-12m STRUCK by ruling 67)

- **A2-v1S** — the published agent AND size. It tests the BUILD: the harness, the env wrapper, the
  scoring window, the unit (env steps = 2 × policy steps).
- **A2-12m** — Jack's agent (v2, the one that goes in him) at the nearest scored size. It tests SIZE.
- Seeds 0, 1, 2; same bands, same score window ((0.9 P, P] training-episode returns), same VOID rule.
  The never-trains control (seed 0, `learning_starts = 2 × total`) runs once per arm and must land
  below 438 @100K, else that arm is VOID. One extension (seeds 3–4) per arm only if it misses at 3,
  the same as the frozen row. Nothing after that.

## Reading (stated before any seed)

| A2-v1S | A2-12m | reading |
|---|---|---|
| in band | in band | the build is sound; the 1m miss is SIZE. A's pilot runs at 12m (cost below), or the Director rules the 1m early-curve miss acceptable for a creature that keeps learning — that is a ruling, not a pass |
| in band | miss | the build is sound; the v2 port at small size learns swingup slower early. Name it on A; v1 vs v2 becomes a raced choice, not an assumption |
| miss | any | the BUILD is suspect (harness, wrapper, unit, scoring), not the size: fix the apparatus before any A pilot; the A1/A3 passes are re-read with that in mind |

## Settings that differ from the published recipe (ours beside theirs)

- A2-v1S: sheeprl's v1 port, not the official JAX code; compile on.
- A2-v1S prefill — READ 11-21 06:20 from the official v1 code (danijar/dreamerv3 `ec526b9`, the
  initial release, 2023-01-14): `run.train_fill: 0`, `batch_steps = batch_size × batch_length` =
  16 × 64 = 1,024, and `train_fill = max(batch_steps, train_fill − len(replay))` → **1,024
  random-policy transitions summed over the 4 envs** (`embodied/core/driver.py` adds 1 per env per
  step; `RandomAgent`). sheeprl's v1 exp file says `learning_starts: 1300`, floored to whole
  iterations (1,300 // 4 = 325 × 4 = 1,300 random transitions, `dreamer_v3.py:759`). **Proposed pin:
  `algo.learning_starts=1024`** = the official count; else 1,300 is named as a difference.
- A2-v1S size — READ from the same file: `dmc_proprio` sets `rssm.deter 512, .*.units 512,
  .*.layers 2` (= its `small`), which is what sheeprl `dreamer_v3_S` composes (dense 512,
  mlp_layers 2, recurrent 512). Matches; nothing to name.
- A2-12m: v2 agent vs the v1 agent of the score files (the same difference A1–A3 already carry).
- Both: precision — READ, three different values: (1) the official v1 computes in **float16**
  (`configs.yaml` line 8 `jax.precision: float16`; `dmc_proprio` does not override it;
  `jaxagent.py:72` sets `COMPUTE_DTYPE`; params cast per call); (2) sheeprl's v1 exp file says
  `bf16-mixed`, which a T4 (sm_75) has no tensor cores for; (3) **the frozen 1m rows A1–A3 ran
  `32-true`** (fabric default, the v3_5 exp sets none; printed in
  `rows/smoke/A2_s0_control.log`). bf16 matches neither the record nor our rows → strike it.
  **OPEN for the Director: pin `32-true` (same as A1–A3, the arms differ from the rows by the arm
  only; named difference vs the record's float16) or `16-mixed` (nearest the record, native on a
  T4, but a second difference vs the 1m rows).** Default proposed: `32-true`. The kernel prints it.

## Cost (GUESS until measured)

The 1m A2 runs took 2,085–2,495 s each on a T4 (rows_A2.json, rows_A2_ext.json). Per §2's arithmetic,
12m runs at 2–5 gradient steps/s against 1m's 20–50, so **one 12m seed ≈ 3.5–7 T4-h**. Size S is
similar or more (dense 512). Three seeds × two arms + controls ≈ 20–40 T4-h, which is MORE THAN A
WEEK'S 30. So:
1. **Timing cell first** (one kernel, < 0.5 T4-h): each arm runs 5,000 policy steps on seed 0 and
   prints gradient steps/s. No score is read from it.
2. Then each seed goes in **its own kernel** (Kaggle's 12-h cap), A2-v1S first (it decides whether the
   build is sound, and every other reading depends on it).
3. If the timing gives > 6 T4-h a seed, the Director picks one arm for this week. The other arm waits
   for next week's hours. It is not shortened: a 100K point needs the full 100K steps.

## OPEN for the Director — CLOSED by ruling 67 (A2-v1S frozen, 12m struck, `32-true`, prefill 1,024)

As drafted: 
Freeze or strike each arm; the precision pin (default `32-true`; bf16 struck); the prefill pin (default `learning_starts=1024`, the official count); whether the 12m arm waits on A2-v1S's verdict.
