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
    segments into unit-variance noise. **Race in flight (J1's, launched 22:38):**
    dose2 unchanged but GAMMA .99, `/data/tmp/smell_sense/gamma99/`, tally
    `gamma99_tally.py`. Rule: ALIVE iff smell@.99 out-eats smell@.95 (2/0/3) AND
    smell_shuffled@.99 on ≥ 2 of 3 seeds (second-half meals); both arms up but
    smell ≤ shuffled = "horizon helps, smell unused", not a pass.
  Philosophy 3 (each raced with a control), 5 (his world as much as he).

- 11-12 · **World: shelter law** (J6, `NE.01` `ok_night_shelter`) · ALIVE, not
  shipped. Shipped 70 % k-cut is non-monotone (occ .8 → .029, sealed → 1.000,
  worse than open .498). Bet: the roof hides the cold night sky (Swinbank), k
  unchanged: sealed → 0.110 on s0 (`shelter_law_night.py`). s1/s2 are a fixture
  fault (probe reads a sliding pose, `shelter_pose_bed.py`). Killed by: re-stale
  cost > gain, or a graded-cone arm doing better.

- 11-12 23:10 · **Exploration in bouts** (learning core's exploration; holder by
  default = white noise, an independent draw every 0.2-s decision) · **DEAD at
  stage A** · `experiments/explore_bout.py`, rule committed first (`141e84f`).
  **Bet:** (1) give the action noise memory — `ou` (τ 20 decisions) or `pink`
  (1/f mixture, τ 1–64; Eberhard 2023), same marginal std, nothing else touched.
  (2) A white walk jitters in place, so he meets food 1–2 times in 22 lives and
  no credit rule can learn from meals that do not happen; infants move in bouts.
  (3) Killed by: a coloured arm not out-feeding white on both seeds and by ≥ 4 of
  40 births, noise-only, on the oracle's ruler (food ready at birth, 250
  decisions, std 0.5, s90/s94 × 20 births).
  **Number:** fed births white 3+1 = **4/40**, ou 2+1 = 3/40, pink 3+3 = **6/40**
  (pink ties s90, lead 2 < 4). The premise held and the conclusion did not:
  median farthest reach white 2.9 / 3.4 m, ou 3.8 / 4.9 m, pink **6.3 / 4.5 m** —
  pink roams ~1.7× wider (white walks 30 m of path to get 3 m away) and still
  finds two 0.28-m foods no more often. Nearest approach is ~2.8 m for all.
  **What it says:** blind roaming of ANY timescale feeds ~10–15 % of births; the
  perfect map feeds 95 %. Range is not what starves him — direction is, and
  direction can only come from using smell. Not ruled out: bout noise as a
  credit aid inside the learner (stage B, never run — no reason to, on this
  number). Philosophy 3 (a default raced), 4 (rule first, dead by it).

- 11-12 23:30 · **Curriculum — weaning, re-raced WITH smell on the repaired
  world** (curriculum, no seat; J5) · **ENTERED, not yet run** (CPU day budget
  spent: `launch_detached` REFUSED at 57,600 s; handed to the workers, ORDERS
  INBOX) · `experiments/wean_smell_race.sh`, tally `experiments/wean_smell_tally.py`.
  **Bet:** (1) the 11-12 weaning arms unchanged (nursed / decoy), but now he has
  a nose and no trap; nurse 26k of 52k decisions, s91–93. (2) Every result above
  says the same thing from a different side: with uniform births a meal is 41
  decisions away, behind a 20-decision horizon, 1–2 times per 22 lives. Born at
  0.6 m the meal is a few decisions from a steep smell gradient — inside the
  horizon he has — and r widens only as he succeeds, so the walk he must credit
  grows one step at a time. The curriculum attacks the horizon from the world's
  side, where gamma .99 attacks it from the learner's; they can be stacked.
  (3) **Rule, declared before any row exists:** meals in lives born uniform
  (`phase == "uniform"`) ending at k > 26000. ALIVE iff nursed out-eats decoy on
  ≥ 2 of 3 seeds AND out-eats the null (dose2 smell, same process, already on
  disk: **2/0/3**, reproduced by the tally) on ≥ 2 of 3; else DEAD. Secondary,
  not gated: does r climb in the nursing phase (nursed vs decoy nurse-meals).
  If ALIVE: register it as the curriculum's first bake-off arm (seat + decoy
  control + claim seeds never piloted) and stack it on the gamma winner.
  Philosophy 5 (the curriculum raced, as the owner ranks it) and 1 (he learns
  to forage by living, from where a mammal starts).
