# DRAFT — "Fix A": row A2 at the published size (J14, ruling 62)

Status: **DRAFT, text only, no seed has run.** Written 2026-11-21 05:40 UTC by a JACK slot. The
Director freezes it (or strikes arms) before any kernel. Nothing here edits `rows.py`'s frozen rule.

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

## Arms (proposed)

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

- A2-v1S: sheeprl's v1 port, not the official JAX code; `learning_starts` 1300 (sheeprl exp file) vs
  the official prefill (to be read from the official v1 config before freezing, not assumed); compile on.
- A2-12m: v2 agent vs the v1 agent of the score files (the same difference A1–A3 already carry).
- Both: `fabric.precision`. The v1 exp file says `bf16-mixed`, and a T4 has no native bf16. The
  kernel must print the precision it ran at. **OPEN for the Director: pin `32-true` for both arms
  (the same as the 1m rows?), named as a difference.**

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

## OPEN for the Director

Freeze or strike each arm; the precision pin; whether the 12m arm waits on A2-v1S's verdict.
