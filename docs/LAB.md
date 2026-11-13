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

- 11-12 · **World/body findings the weaning kill uncovered** (J1 + LAB; modules
  and full numbers in git, long form at `4b4c2e3`), one line each:
  - He could not perceive floor food (rays pass over it; 0 of 584 poses). *Eye
    height* (`food_sense.py`) DEAD: meals split 1/2/3 vs blind 2/2/3.
  - *Smell* (`smell_sense.py`, points to food on 99.5 % of poses) DEAD three
    times: 13k 8 vs shuffled 10; 52k repaired world **2/0/3 vs shuffled 20/0/3**.
    A sense that works goes unused by the reactive learner.
  - *Body* (`body_probe.py`): pole topples ~19/20 births; plinth stands. Not the cap.
  - *The pool was a DRIVE TRAP*; fixed (`W0.PL` PASS `3f15c7f`). Perfect-map
    oracle fed per birth 23/40 → **38/40**. Body and world are not the gap; the
    learner is.
  - *Credit horizon:* GAMMA .99 race DEAD 11-13 (for on 1 of 3 seeds). 88 of 120
    meals sit in 5 "jackpot" lives; approach from afar does not carry to the
    next birth. For `LF.04`: a meal-count margin is heavy-tailed noise.
  Philosophy 3, 5.

- 11-12/13 · **World: shelter law** (J6, `NE.01`) · ALIVE, not shipped; roof hides
  the cold sky: sealed → 0.110 on s0. **Occlusion-law race 11-13
  (`occ_law_race.py`): NO WINNER** — shipped 4/5, graded cone 2/5, damped
  sleeper 2/5 (dies of injury), open control 0/5 (as it must). Band-geometry
  arm is next. Rows /data/tmp/occrace/.

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

- 11-13 23:10 · **Sense lookahead — he imagines each push, takes the one he
  wants** (Deliberation, VACANT; holder by default = the reactive actor) ·
  `experiments/sense_lookahead.py`, rule committed first (`29424ef`).
  **Bet:** (1) no actor: a forward model of ALL his senses (learned reward-free
  from babbling) + a linear cue-value learned from his own meals; every 5
  decisions he imagines 32 pushes and takes the one that raises the want.
  (2) `DP.05` killed lookahead in a W0 where he could not sense food; he has a
  nose now, the reactive learner ignores it for 52k decisions, and push →
  smell-change is taught by every decision while meals are rare. (3) Killed
  by: on explore_bout's ruler (white 4/40, oracle 38/40; seeds 90+94, 20 test
  births each after 5k babble) neither `want` (nothing given) nor `told` (want
  given: "more smell") feeds more births than BOTH `babble` (null) and `shuf`
  (push→smell link permuted — must fail) on both seeds and by ≥ 8 of 40.
  END-FIT rows 4, 3, 1; closes none. Philosophy 1, 3, 4.
  **Number:** IN FLIGHT on a free Kaggle CPU kernel `jack-cpu-1794611608` (the
  box's day CPU ceiling was spent). Harvest: `kaggle kernels output`, then
  `sense_lookahead tally`.
