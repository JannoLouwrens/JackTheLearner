# DRAFT — Row W-1: the body's known-answer row, banded on Playground `WalkerWalk` Brax PPO (ruling 133)

**STATUS: DRAFT 2026-11-27 15:45 UTC — NOT FROZEN, NOTHING RUN.** Written by JACK (ORDERS Next-up 1). A sitting freezes it
(ruling 133) after J28's cell 1 (`experiments/body/mjx_timing.py`) shows JAX/MJX runs on the GPU
Kaggle gives. No seed of this row has run. Nothing here edits §29's W-1 text, J28's cells, or any
frozen file. Ruling 135's worker-freeze does NOT apply: this row has no existing J-order's arms/bar.

## Why this row exists

PLAN, Body and movement, stage 3 (REPRODUCE): before any of his own body's walking is trusted, our GPU
build (MJX + Brax PPO on a free Kaggle GPU) must hit a published number on a small standard task.
§29 proposed HumanoidStand (paper 795.5); §30 found nobody outside the authors reproduces it on the
public script (issue #171: ~50 at 105M) — a miss there cannot tell a broken build from a hard task.
§30 / ruling 133: two independent programs agree on **WalkerWalk** PPO, so the band sits there.

## The published record (§30, sources)

- **Playground paper** ([2502.08844](https://arxiv.org/abs/2502.08844) v1, Fig. 9, vector vertices,
  rough pass `/data/jack-research/scratch_sec30_extra/panels_final.txt` l. 21 — axis frame not
  cleaned): WalkerWalk Brax PPO **mean 968.5, seed lines end 965.9 / 967.2 / 967.6 / 969.4 / 972.3**
  (5 seeds); SAC 974.0. Config: Table XII p. 37 — 60M nominal, 2,048 envs, batch 1024, unroll 30,
  32 minibatches, 16 updates/batch, γ 0.995, lr 1e-3, entropy 1e-2, reward_scaling 10, num_evals 10.
- **SLM-Lab** `docs/BENCHMARKS.md` (run 2026-03-14, `scratch_sec30B/notes.md`): a different PPO
  (1 unroll/update, clip 0.2, other nets) scores WalkerWalk **945.31** at 100M frames, training-rollout
  returns, mean of 4 sessions. Same code: HumanoidStand 114.86 — so the walker is the robust one.
- The freezer re-reads the five WalkerWalk vertices from the PDF stream with the axis fit cleaned
  (§30 did this only for HumanoidStand) and writes them here before freezing.

## THE RULE (proposed — every number open until the freeze)

- **Package:** `playground==0.2.0` (imports as `mujoco_playground`; `mujoco_playground` itself is a
  PyPI 404; 1.1.0 is Warp+torch with no Brax configs). Env `registry.load("WalkerWalk",
  config_overrides={"impl": "jax"})`; the kernel prints package VERSIONS and the IMPL used (as
  `mjx_timing.py` does). Any other impl printed → the seed is VOID.
- **Trainer:** `dm_control_suite_params.brax_ppo_config("WalkerWalk")` **unchanged**, through
  `brax.training.agents.ppo.train` + `wrapper.wrap_for_brax_training` (the public script's path,
  `learning/train_jax_ppo.py`). **True step count logged** (§30: from v0.1.0 the config carries
  `num_resets_per_eval=10`, which makes Brax round "60M" up to ~88.5M on HumanoidStand; whether
  WalkerWalk's 0.2.0 entry has it is NOT checked — the kernel prints it). The
  kernel prints the composed config; any key differing from 0.2.0's file other than those listed
  under "Differs" → VOID.
- **Seeds:** 0, 1, 2. **One extension, decided now:** if exactly one seed lands below 900 and the
  other two at or above it, seeds 3 and 4 run and the row reads on all five (still "every seed").
  No other extension; nothing runs after that whatever it reads.
- **Score:** the trainer's own final eval reward (Brax `eval/episode_reward`, deterministic policy,
  the config's eval episodes) at the end of training — the quantity the paper's Fig. 9 plots.
  Printed beside, not read: every eval point with its true step count, wall-seconds, GPU kind.
- **Band:** **REPRODUCES iff every seed ≥ 900** (§30 proposal; paper's lowest seed 965.9, SLM-Lab's
  mean 945.3; 900 leaves ~65 under the paper's spread for a T4/P100 and a different trainer path).
  Anything else **MISSES**, with each seed's number. A seed that crashes or is cut by a Kaggle limit is
  LOST, entered as such, never re-bought; a row with a LOST seed is NOT JUDGED.
- **Control (must fail):** the never-trains twin — same config, seed 0, `learning_rate = 0`. MUST end
  **< 300** (an untrained walker in the DMC walk task scores tens-to-low-hundreds — the freezer checks
  this against the paper's first eval point and sets the number from it before freezing), else the
  row is VOID (the score would not need learning).
- **Beside, unbanded, deciding nothing:** HumanoidStand, same package/impl/trainer, config unchanged,
  seed 0, true step count logged. §30's prediction written now: **< 100 at the end** (#171's path). A
  ~800 would be news for §30, not a pass of anything.
- **Venue:** Kaggle GPU (T4 asked; the GPU given is printed), one session a seed. Cost **(inf)**:
  ≈ 30–45 T4-min a seed at 60–88M steps (A100 ~92k sps on the humanoid ÷ ~2.5). J28's cell 1 replaces
  this guess with a measured steps/s before freezing; if a seed would exceed one session's limit, the
  freezer adds a carry or cuts nothing — the band is never read at a shorter run.

## Differs from the published recipe (every one listed)

| Setting | Published (paper) | This row | Why |
|---|---|---|---|
| Trainer script | `brax/training/learner.py` (maintainer, #171) | `ppo.train` + Playground wrapper (`train_jax_ppo.py`'s path) | the public path; `learner.py` not read (§30 "not verified") |
| Hardware | A100 | Kaggle T4/P100 | free compute only |
| Seeds | 5 | 3 (+2 on the one extension) | budget; band is "every seed", so fewer seeds is a weaker, not looser, claim |
| Package | paper-era repo | `playground==0.2.0`, `impl=jax` | last JAX/Brax release |
| Score source | Fig. 9 eval curve | same quantity, our run's own final eval | — |

## What a result means

- REPRODUCES → our MJX+PPO GPU path is trusted for stage 3's body work at this scale; the humanoid
  beside it says whether the public path learns HumanoidStand at all.
- MISSES with the twin < 300 → our build or venue is wrong (WalkerWalk is the task two programs hit);
  find why before any body run is trusted. It is a FAIL to fix, not to file.
- VOID → the apparatus is wrong; fix and re-run under a re-registration that says why.

## Open before freezing (the sitting)

1. J28 cell 1's measured steps/s and GPU kind (sizes the seed cost; is JAX on that GPU at all).
2. Clean re-read of the five WalkerWalk vertices and the first eval point (sets the twin's line).
3. Whether `wrap_for_brax_training` on WalkerWalk needs any key the config lacks (check on 0.2.0).
