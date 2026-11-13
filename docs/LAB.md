# LAB — the inventor's notebook (≤ 10 KB)

One entry per idea: date · seat · the bet · the number · ALIVE / DEAD / ENTERED.
A dead idea with a number is a result. Do not retry one without a new reason.
Folded 2026-11-12 23:40 (was 20 KB): full text of every entry is in git
(`git log -p docs/LAB.md`, last long form at `0275671`).

## Already known from the ladder (so nobody re-invents them)

- 08-23 · learning core · five cores screened in W0 · only `wm-latent` learned to
  survive (4.65σ vs null); `wm-efe` 2.05σ, `ppo-lp` 1.20σ, `ppo-needs` 1.06σ,
  `dreamer-xs` −0.94σ.
- 09-07 · control path · small plastic head 506, MLP 415, 57M-trunk arms 344–350,
  SB3 518 · VOID on the wrong null; re-run as `D1.1`.
- 08-24 · deliberation · lookahead gains 13 s ± 19 in W0 — too shallow a world.
- 08-13 · mood → movement · 0.23–0.38 vs chance 0.25 · DEAD as built.

## Ideas

- 11-11 · **Working memory — ring (rotating) store** · **SEATED, `WM.02` PASS**
  (`2103829`, seeds 100–104): ring 0.902 ± 0.196 vs GRU 0.579, null 0.347; shuffled
  0.127 and no-rotation −0.047 both failed as required. 1,031 params vs 4,872.
  Dead variants: angle init ×4 (0.106), no write gate (0.119). Open: a
  non-commutative register; ME.8 / LF.02 with the ring in place. Philosophy 3, 4.

- 11-12 · **Curriculum — weaning** (`experiments/wean_pilot.py`: the mother bears
  him within r of ready food, r ×1.5 per fed life, then leaves; control = decoy
  points) · **DEAD at first dose (s90, 13k): weaned-window meals nursed 0, null 0,
  decoy 0** — but killed in a world where he could not sense food and one food
  sat in a drive trap (both below). **RE-ENTERED 11-12 23:30 with a new reason —
  see the last entry.**

- 11-12 · **World/body findings the weaning kill uncovered** (J1 + LAB; each has
  its module and numbers in git):
  - *He could not perceive floor food:* rays leave at z 0.70 m, food tops at 0.33 m —
    0 of 584 poses see it; no smell, no x,y.
  - *Eye height* (`food_sense.py`): split rays see food on 76/611 poses; meals
    s91–93 level 1/2/3, split 1/2/3, split_blind 2/2/3 · DEAD by declared rule.
  - *Smell* (`smell_sense.py`, a 12-wide learned modality; bilateral difference
    points to food on 99.5 % of poses): 13k meals smell 8 vs shuffled 10 vs level
    6 · DEAD; 52k (trapped world) second-half smell 1/5/0 vs shuffled 4/17/27 ·
    DEAD; 52k on the REPAIRED world (dose2) **smell 2/0/3 vs shuffled 20/0/3** ·
    DEAD (0 of 3 seeds). A sense that works goes unused by this learner.
  - *Body* (`body_probe.py`): the pole topples in ~19/20 births by decision 2–5
    and is dragged lying; righting is a < 1 % needle. Plinth stands (0–1/20
    topple). After the water repair neither body is the cap (next line); `BD.01`
    stays registered, expect an honest FAIL.
  - *The pool was a DRIVE TRAP* (`pool_lip_contact.py`): `WaterVolume.apply`
    overwrote the drive and "wet" reached onto dry floor; food obj1 sat inside the
    band. Fixed (`W0.PL` PASS `3f15c7f`). **Perfect-map oracle fed per birth:
    23/40 → 38/40 (pole), 30/40 → 36/40 (plinth path).** Body and world are not
    the gap; the learner is.
  - *Credit horizon* (J1, reading of files): GAMMA .95 ≈ 20 decisions; a perfect
    map needs median 41.5 decisions birth → meal (21 % within 20); meal weight at
    birth .12 (value), .0015 (GAE), and per-segment normalisation turns meal-free
    segments into unit-variance noise. **GAMMA .99 race (J1) DEAD by its rule,
    11-13:** 2nd-half meals smell@.99 22/0/1 vs smell@.95 2/0/3 vs
    shuffled@.99 20/23/0 → for on s91 only (1 of 3). Horizon alone is not it.
    *Shape of the meals* (both races, 169 lives): 88 of gamma99's 120 meals are
    in 5 "jackpot" lives (12–20 meals, 970–1400 s, born 1.6–8 m out, so not
    birth luck); only 1 of 5 jackpots is followed by a fed life. He can camp on
    food he stumbled onto; approach from afar does not carry to the next birth.
    For `LF.04`: a meal-count margin is jackpot-dominated, heavy-tailed noise.
  Philosophy 3 (each raced with a control), 5 (his world as much as he).

- 11-12 · **World: shelter law** (J6, `NE.01` `ok_night_shelter`) · ALIVE, not
  shipped. Shipped 70 % k-cut is non-monotone (occ .8 → .029, sealed → 1.000,
  worse than open .498). Bet: the roof hides the cold night sky (Swinbank), k
  unchanged: sealed → 0.110 on s0 (`shelter_law_night.py`). s1/s2 are a fixture
  fault (probe reads a sliding pose, `shelter_pose_bed.py`). Killed by: re-stale
  cost > gain, or a graded-cone arm doing better.
  **11-13 OCCLUSION-LAW RACE (`experiments/occ_law_race.py`, rule declared
  first): NO WINNER.** HOLDS = NE.01 ok_night_shelter AND ok_forager, verbatim,
  seeds 0-4 (3,4 fresh). shipped 4/5 (s0: forager hyperthermia at 17707/18000,
  sleep occ .946); graded cone 2/5 (s1/s2/s4 night cost .47/.20/.93 > .20 - the
  cone picks worse beds); damped sleeper x3 2/5 (s0/s1/s3 forager dies of
  INJURY at 5005/4015/11942 - stiff sleeping joints hurt him); open control 0/5
  (fails, as it must). Shipped is not falsified - it holds on both fresh seeds;
  its one red is s0's knife-edge. By the rule the band-geometry arm (a
  playground edit) is the next raced. Rows /data/tmp/occrace/*.jsonl.

- 11-12 · **Exploration in bouts** (white vs OU vs pink action noise, noise-only,
  `explore_bout.py`, rule first `141e84f`) · **DEAD at stage A:** fed births
  white 4/40, ou 3/40, pink 6/40 (lead 2 < 4). Pink roams 1.7× wider (reach
  6.3 m vs 2.9) and finds food no more often: range is not what starves him,
  direction is. Stage B (bout noise inside the learner) never run. Philosophy 3, 4.

- 11-12/13 · **Curriculum — weaning re-raced WITH smell on the repaired world**
  (`wean_smell_race.sh`, rule first) · **DEAD 11-13 03:35** (null clause 1 of 3):
  weaned meals nursed 1/9/1, decoy 0/0/·, null 2/0/3; s92's 9 are one jackpot life
  born 0.70 m. Nursing feeds him (nurse meals 18/5/6 vs decoy 0) but r never holds:
  up to 0.9–1.69 m, back to 0.5, vs uniform births median 3.1–3.9 m. He learns
  eating-where-born, not walking up a gradient. Staged widening (`CR.00`, J5) is
  the remaining world-side arm. Philosophy 4, 5.

- 11-13 11:20 · **Meal memory — what he replays** (Consolidation / the core's
  replay; holder by default = one FIFO ring of 256) · **ALIVE (thin) 11-13 11:34** ·
  `experiments/meal_memory.py`, rule committed first (`dabd2bf`).
  **Bet:** (1) a second ring of 256 keeps every segment that had a meal in it;
  batches are 8 FIFO + 8 meal — same batch, updates, core and loss. (2) Today a
  meal is sampled ~16× and overwritten 51 s later, so smell is never paired with
  food often enough to become a conditioned reinforcer; brains replay what was
  rewarded far out of proportion (reverse replay; self-imitation, Oh 2018).
  (3) Killed by: on nursery trials (born hungry ≤ 1.2 m from ready food, 250
  decisions, s91–93, 6.5k, read k ≥ 2000) fed fraction not above BOTH `fifo` and
  `yoked` (same ring, filled with the last meal-FREE segment — must fail) on
  ≥ 2 of 3 seeds and by ≥ .10 pooled. END-FIT rows 4 (late payoff: "replay of
  256") and 2 (no forgetting); closes none. Philosophy 3, 4.
  **Number** (read fed trials, s91/92/93): meal **17/33, 13/29, 16/33 = .484**;
  fifo 9/26, 17/33, 4/22 = .370; yoked 7/25, 10/28, 15/32 = .376 (the control
  failed, as it must). Seed wins 2/3 vs fifo, 3/3 vs yoked; pooled margins
  .114 / .108 against a .10 bar — ALIVE by the rule, by a hair. Post hoc, not
  gated: meals that took > 20 decisions (a walk, not a birth on the food) meal
  16, fifo 6, yoked 10; born ≥ 0.7 m fed .39 vs .32 / .33. Most meals land in
  5–16 decisions, so this ruler is mostly birth luck: the pilot shows a lean,
  not approach from afar.
  **Claim gate, declared now, before any claim row** (module docstring): fresh
  seeds 94–98, 13k decisions, same three arms, same rule (≥ 4 of 5 seeds vs
  each, pooled ≥ .10) AND, among trials born ≥ 0.7 m, meal above both pooled.
  Then the dose that matters: uniform births, 52k, vs dose2's ruler. Handed to
  the workers (ORDERS INBOX); not yet a registered bake-off.
