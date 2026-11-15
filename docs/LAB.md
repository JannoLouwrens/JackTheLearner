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

- 11-13 · **Meal memory — what he replays** (Consolidation; `meal_memory.py`,
  rule first `dabd2bf`) · **DEAD at the claim gate (11-15, s94–98, 13k, Kaggle
  CPU).** Pooled fed: meal .450 vs fifo .277 vs yoked .352; seed wins 5/5, 4/5;
  far (born ≥ 0.7 m) .335 vs .230/.332. Margin vs yoked **.0985 < .10**, the only red
  clause. Stays dead as declared. Pilot below. A second ring
  keeps every segment with a meal; batches 8 FIFO + 8 meal. Nursery trials
  s91–93: fed **.484** vs fifo .370 vs yoked control .376; wins 2/3 and 3/3,
  margins .114/.108 vs a .10 bar. Mostly birth luck (meals land in 5–16
  decisions): a lean, not approach from afar. Claim gate in the module
  docstring (s94–98, 13k, born ≥ 0.7 m clause). END-FIT 4, 2. Philosophy 3, 4.

- 11-13/14 · **Predict-then-push, three arms** (Deliberation, VACANT; long form
  at `1a18168`), all DEAD on the fed-births ruler (babble 5/40):
  *sense lookahead* (`sense_lookahead.py`, MLP forward model + linear want)
  told 6/40, want 4, shuffled 7; *drive in his own frame* (`ego_drive.py`)
  push's share of held-out smell prediction ego .017/.000 vs world .086/.044
  (bar .10); *gain-field model* (`gain_field.py`, bilinear ridge) stage 1
  PASSED (.291/.235 vs MLP .207/.044) but fed told 4/40, want 9/40, shuffled
  5/40. Held-out r² < 0 throughout: he cannot yet predict what a push does to
  his nose. A want cannot start without a meal (s94 had none). Philosophy 3, 4.

- 11-14/15 · **Run and tumble — climb without predicting** (Deliberation,
  VACANT; `run_tumble.py`, each rule committed before its seeds: `c0db7e6`,
  `f61386f`, `1a18168`). Bet: keep a push while what he wants rises, redraw
  when it falls — one measured bit, no model. Killed by: no lead of +8/40
  over babble AND over a coin-yoked twin on both fresh seeds.
  - *told* (keep iff smell rose; s140/141) **DEAD: 8/40** vs babble 6, yoked 5
    — but `toward` +.44/+.20 vs +.02/+.03: first arm to point him at food.
    Any bearing within 90° raises smell, so he climbs slowly (4.2 → 2.9 m).
  - *steer* (tumble on a fall; while rising, best push + small perturbation,
    accepted only if its rise on log smell beat the best's; s142/143)
    **ALIVE with the want GIVEN: fed 8/6 = 14/40 vs babble 3/40, coin-yoked
    2/40** (failed, as it must). First arm of nine to beat babble on this ruler
    (oracle 38/40).
  - *steerwant* (same rule on a want LEARNED from his meals: logistic over
    all senses, nursery of ~31 births / 14 fed; s144/145) **DEAD: 7/40** vs
    babble 5, label-shuffled 7. The want puts .08–.11 of its weight on smell,
    the shuffled twin .10–.14: 14 meals do not teach a 1-of-all-senses cue.
  **Where the gap is now:** not direction (a measured bit steers him) but the
  WANT — what to climb. Next bets, unrun: a want learned from the CHANGE in
  each sense over the bout before a meal (smell rises before every meal;
  levels do not separate); the critic's value as the want; `steer` as the
  exploration inside the seated learner (J1). Not entered: only the given-want
  arm lives, and a given want is advice, not his. END-FIT 1, 3, 4.
  Philosophy 1, 3, 4.
