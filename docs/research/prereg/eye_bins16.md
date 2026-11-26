# eye_bins16 — brain B on the kind paint's binned eye (ORDERS J24 (3))

**STATUS: DRAFT, NOT FROZEN (written 2026-11-26 13:30 UTC).** It binds nothing
and no seed runs on it. J24 (2) read toy K HOLDS, toy C MISS, forager PASS; the
order says "a miss → INBOX, no learner seed", and the Director has not yet ruled
whether (3) proceeds on K + the forager (INBOX 10:45, 11:40). If the ruling says
proceed, the freeze is ONE commit: change STATUS to `FROZEN <date>, ruling <n>`,
nothing else. If the ruling changes a term, the term is changed here BEFORE the
freeze, in the same commit, and said so.

## Question
Does brain B (the library's flat-MLP SAC) learn to eat FROM HIS EYE when the eye
shows food the way every published forager's eye does — one look per KIND, the
same in every world (§28) — at 500k decisions?

## Prior (PUBLISHED_PRIORS §28, §27, §12)
- 0 of 12 published foragers' rays return a per-object code; all return a class.
  FoodCollector reward 10 at 2M; SAC PointGoal1 27.0 at 3e6; a TD3 navigator 13%
  at 100k. The record's need is 2–3e6 steps; **500k is below it — a NOT SHOWN here
  is a row, not a door.**
- Ours, read on this world: a hand ruler on the 16 food bins alone eats 20/20
  seeds (mean 25.5 meals a life), its shuffled-bins control 0/20 (2.6), white
  random 0.43 (J24 (2), `/data/jack-logs/j24/bins_forager/score.json`). The look is
  learnable by a plain rule. Toy (J24 (2)): kind look read at 99% recall by step 250.

## Arms (code: `experiments/pilot_b2.py` `J24_ARMS`, built `test_j24_arms.py`)
| cell | fed | role |
|---|---|---|
| `K-bins16` | the world's observation with `eye="bins16"`, `paint="kind"`: vision = 5 kinds × 16 azimuth closeness bins (80) + the 42 non-eye numbers = 122 | candidate |
| `K-bins16-blind` | the same, every bin 0 (`blind_eye.BlindEye`, J17's wrapper unchanged) | twin, must not beat sighted |
| `K-bins16-shuffled` | the same, the 16 bin COLUMNS permuted by one fresh permutation each decision, all kinds together (`blind_eye.ShuffledBins`, own RNG) | twin, must not beat sighted |

Disclosed: the shuffled twin keeps "food (and water, rock…) is visible somewhere,
this near"; it loses only WHERE. In J22 the shuffled twin ate most. A sighted cell
that does not beat it has not shown he reads where food is.

## Learner and world — ours beside the record
Learner = J17's `K-row128` setting exactly, only the eye changed: sbx 0.28.0 SAC,
`MlpPolicy` layer-norm 2×256 (flat MLP, the library arm), lr 3e-4, batch 256,
tau 0.005, γ 0.995, warmup 5,000, train_freq 32 / gradient_steps 32, gSDE on,
`ent_coef="auto"` (ruling 64), buffer = T. Kernel pins `pilot_b.KERNEL_PINS`.
"The J22 pilot's settings" (the order) = J22's world and learner; J22's h4 conv
encoder + random shift is NOT carried — the order names a flat MLP.
World = `PILOT_RULE` (mujoco 3.17.0, ruling 76; eye_frame body, motor_frame body,
food_floor_only, lazy_contact_audio, native_substep; arm K's reward `dr100-eat`,
move-cost dial, random needs, d 0.75, life cap 9,000) + `paint="kind"`, `eye="bins16"`.
Settings that differ from the published foragers: one environment, not many;
500k, not 1–3e6; 3 seeds; our own body and world (no published number exists here).

## Seeds, size, score
- Seeds **941, 942, 943**; **500,000 decisions** a cell; nine cells.
- Score: **late meals a life** = meals per life over lives STARTING at decision
  ≥ 250,000 (ruling 65), read with `j17_late_meals`; `j17_onset` printed beside;
  critic rulers printed beside and decide nothing (ruling 102).

## Rule (the order's, written before any seed)
- **EATS** = sighted above BOTH twins on 3/3 seeds AND sighted mean ≥ 2 × the
  higher twin mean.
- **One extension, stated now:** sighted above both twins on ≥ 2/3 but not EATS →
  the same nine cells carried to 1.5M (through `save_life`, whole carry or refuse),
  read by the same rule at its own late half (lives starting ≥ 750,000).
- Else **NOT SHOWN AT 500k** — a row, not a door.
- A cell that does not come home whole (cut, refused carry) is re-run from its
  last whole carry; it is never read short.

## Compute (ORDERS J24 (4)–(5), ruling 99)
Timing cell first: box 300 decisions, then one Kaggle CPU kernel 5k a cell. Then
three self-advancing chains, one seed's three cells a kernel, carried through
`brain_b.save_life` (`forge/r1_run.py`'s carry). Raw files land under
`/data/jack-logs/j24/b/`.

## After the read
EATS → `pilot_a.md` takes the same row for A; the lab's hindsight eye is raced
next sitting. NOT SHOWN → the next sitting rules 1.5M / the per-ray one-hot arm
(§28 A).
