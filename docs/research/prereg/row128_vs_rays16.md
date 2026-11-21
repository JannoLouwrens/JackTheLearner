# Pre-registration — `row128` vs 16 rays under brain B (ORDERS J17) — FROZEN

**Status: FROZEN 2026-11-21 by ruling 66 (`docs/DECISIONS_RESOLVED.md`) = the
draft as written 11-20, with the pins below. This commit is the freeze: no
seed of any arm exists (grep at freeze: seeds 921–925 appear in no run, spec,
kernel or artifact in the repo).** Where a pin and the draft text differ, the
pin binds.

## THE PINS (rulings 64, 65, 66; ORDERS J17)

- **Setting S = `ent_coef="auto"`** (ruling 64; the Gate below is read: E2c's
  food arm ate under both settings, the tie-break is auto).
- **Arms:** `K-rays16` / `K-row128` / `K-row128-blind` (MUST FAIL) /
  `K-lr0-row128`. World `pilot_b2.PILOT_RULE`, `eye=` the only difference.
- **Seeds 921–925, 300,000 decisions, every arm. NO extension.**
- **Primary = LATE meals/life, row128 over rays16: Mann–Whitney one-sided
  U ≥ 21 of 25 AND ≥ 4 of 5 seed pairs.** "Late" is ruling 65's, which names
  J17: **the late HALF of the run — the lives that START at decision ≥
  150,000.** The draft's "last quarter of lives" number (`pilot_b2 table`) is
  printed beside it and NOT read. (Pinned here before any seed because the
  draft and ruling 65 name two different windows; one INBOX line says so.)
  The same window is used for the row128-over-blind clause.
- **Reported beside it, not read:** first capped life, last capped life
  ("reached then lost").
- Per-arm readings (SUSTAINED / EATS BUT STARVES / SLIDES / NEAR RANDOM) and
  the VOID clauses stand as drafted. The verdict is named as one of: **row128
  eats / neither eats / both eat**, with the HELPS / width-not-sight / no-
  difference clause beside it.

## The question

Does a plain learner (brain B, SAC) in his world eat more when its eye is the
body-frame 128-azimuth level row (`jack_env(eye="row128", eye_frame="body")`)
than with today's body-frame 16 rays — and is any gain the eye's CONTENT, not
the wider input?

## What the record and our own rows already say (so this asks only the rest)

- Published: nothing. PUBLISHED_PRIORS §12: "Nothing in print says if 16
  undifferentiated rays can tell food from a wall"; Safety Gym / Animal-AI rays
  vs pixels: none found. Nearest: "What if Eye" (2025, MuJoCo, PPO) — detection
  needed two 15×15 eyes, navigation ~10 eyes of 4 receptors.
- Ours, by a hand reader (not a learner): floor food in sight .909 (row128)
  vs .528 (16 rays) at the entry rung (lab, ruling 37); in-sight .905 vs .287
  at d 0.75 (J12 (1) `9e4031bb`).
- Ours, by a hand forager in the body frame (WORLD_CARD J12 (2′), d 0.75,
  seeds 261–280): f4 row128 + body motor **20/20, 7.53 meals/life**; g4 the
  same with shuffled rays 2.17 (must fail, failed); b4 random 0.45. So the
  task IS solvable from the row128 eye by a rule that never knows its heading.
  No equivalent 16-ray forager row exists in the body frame.
- Ours, by a learner: pilot 2 (PILOT_B2 §4, 16 rays, fixed α 0.1, 300k) — every
  B arm SLIDES or NEAR RANDOM; meals/life late 0.08–0.16, below K-lr0's 0.82.
  An eye race run under a setting where B does not eat would compare two
  floors; that is why the Gate exists.

## Gate (a run that cannot discriminate is not run)

Runs only after J15's E2c table is read (forager_clone.py RULE E2c item 5):
- auto-food eats, fixed-food does not → setting S = `ent_coef="auto"`.
- fixed-food eats (with or without auto) → S = the setting whose food arm ate
  on more seeds; tie → `auto` (it has the published precedent, §19).
- BOTH food arms miss → **this race does NOT run** at 300k: a handed-direction
  learner that cannot eat leaves nothing for an eye to show. It waits for the
  forge's run-length answer (F3, round 1) and is re-drafted at that length.
- a zero-control arm that eats as well as its food arm → also does not run as
  written: the handed sense carried nothing, so "B eats" in E2c is not a
  property an eye could improve; re-drafted from the zero arm's reading.

## Arms (all = pilot arm K exactly, PILOT_RULE world, setting S; only the eye differs)

| arm | eye | role |
|---|---|---|
| `K-rays16` | rays16, body frame (vision 32 wide) | today's eye |
| `K-row128` | row128, body frame (vision 256 wide) | the candidate |
| `K-row128-blind` | row128 width, every vision number 0 (same wrapper style as E2c's `zero`) | **MUST FAIL**: the wider input with no content |
| `K-lr0-row128` | row128, lr 0 (never trains) | floor/bound, as K-lr0 in pilot 2 |

World: `pilot_b2.PILOT_RULE` unchanged (food_floor_only, spawn_clear 1.55,
d 0.75, drain ramp K, body eye + body motor, native_substep, life cap 9,000),
`eye=` the only difference. No handed sense on any arm.
T = 300,000 decisions (pilot 2's size; a miss at 300k is weak evidence,
"not learned in 300k", as E2c item 3). Seeds **921–925**, every arm (fresh;
the freezing commit greps the repo to confirm none was stepped). 20 runs,
4 a kernel → 5 Kaggle CPU sessions, est. ≈ 7.5 h each (E2c's kernels, plus the
row eye's +0.03 ms a look; the wider SAC input is a measured cost the kernels
log, not assumed).

## Readings (fixed now)

Per arm, `pilot_b2 readings` unchanged (W = 2; SUSTAINED / EATS BUT STARVES /
SLIDES / NEAR RANDOM; floor clause and broken clauses as PILOT_B2 §3, K's named
floor held8-k0-ff-bm). Primary number per run: late meals/life (last quarter of
lives, `pilot_b2 table`).

- **row128 HELPS**: `K-row128` beats `K-rays16` on late meals/life with
  Mann–Whitney one-sided U ≥ 21 of 25 (n 5 vs 5, p ≈ 0.048) AND on ≥ 4 of 5
  seed pairs, AND `K-row128` beats `K-row128-blind` by the same test.
- **width, not sight**: `K-row128` beats `K-rays16` but NOT `K-row128-blind`
  → the gain is not credited to the eye; said so.
- **no difference**: neither clause → "no eye effect at 300k under S" (weak,
  same reason as E2c item 3). The record has no learner answer to fall back
  on; the hand rows above stay what they are (hand rows).
- **VOID — broken**: `K-row128-blind` reads SUSTAINED on ≥ 3/5, or
  `K-lr0-row128` crosses W = 2 on ≥ 2/5, or ≥ 2 runs of any arm rc ≠ 0 / non-
  finite loss. Cut runs as PILOT_B2 §3 (counted, never dropped; ≥ 2/5 below
  150k → INCOMPLETE, re-run on 926+).
- Reported only: meals per 1,000 decisions over the last 50k, % capped,
  causes of death, dec/s per arm (the width cost), SAC trace (§3b).

What it decides: the eye of every later B run in his world (row128 if HELPS,
rays16 otherwise — the cheaper eye holds its seat on a tie). No champion seat
moves; brain A is not judged by it.

## Differences from any published recipe

None to list beyond pilot 2's (PILOT_B2 §3, B0 = J8's SAC) and setting S: the
eyes are ours (no published ray-eye learner row exists to copy).

## To build at freeze (not before)

`pilot_b2`: `eye=` passed through `make_world`/`pilot_world` (default rays16,
pilot and E2c kernels byte-identical — planted test as `test_e2c_auto`); a
`blind` vision wrapper (planted test: obs width 256, vision all 0, every other
obs number and the world's response to a fixed action sequence identical to
`K-row128` over 2k decisions); kernels `r0..r4`.
