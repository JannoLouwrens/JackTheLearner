# WORLD CARD — can anything live in today's world? (ORDERS J1)

PLAN stage 1, part "His world, needs and lessons", step 2. Prior:
`PUBLISHED_PRIORS.md` §3 (our floor meal is 8% of the bar — the smallest of any
world reviewed — and "the entry settings at which a ray-sensing rover's random
policy eats must be measured here"), §8 "Gates" (floor arms ≤ 1 success; a
scripted forager must pass; else the venue cannot judge learners).

**The rule is in `experiments/world_card.py`'s docstring, committed before any
registered run.** Registered seeds 201–220, 10 lives each, through `jack_env`
with today's world. Smoke seeds 153/154 only were used to write and tune the
hand forager (row f). No learner, no ledger row: these are rulers.

## Rows (registered run: Kaggle CPU; b harvested 11-15 18:31, raw `world_card_data/b.jsonl` sha256 4dc9ef93…; f harvested 19:04, `f.jsonl` sha256 8076f639…; a harvested 19:25, `a.jsonl` sha256 65ceb0cb…; g harvested 20:10, `g.jsonl` sha256 bb1dc832…; c harvested 20:35, `c.jsonl` sha256 c9491ddb…; d8 harvested 21:25, `d8.jsonl` sha256 f4a24125…; d32 harvested 22:10, `d32.jsonl` sha256 7a371c53…; e harvested 11-16 (landed 11-15 23:58), `e.jsonl` sha256 ba26eed3…; b2 harvested 22:50 on 221–240, `b2.jsonl` sha256 2ce802b8…; f2 harvested 22:40 on 221–240, `f2.jsonl` sha256 f11fad34…; g2 harvested 22:40, `g2.jsonl` sha256 32e0cfa2…)

| row | policy | lifetime s (mean ± sd) | floor meals / life | apples | drinks | cause of death | summed reward / life |
|---|---|---|---|---|---|---|---|
| a | do-nothing | 599.9 ± 0.2 (599.4–600.0) | 0 (0/200 lives ≥1) | 0 | n/a | energy 200/200 | −1.00 (all) |
| b | random, white per decision | 301.7 ± 16.4 (278–376) | 0.52 ± 0.76 (max 4; 76/200 lives ≥1; per-seed means 0.1–0.9) | 0 | n/a | energy 200/200 | −1.00 (all) |
| c | pink noise per life | 491.5 ± 30.1 (443–677) | 0.44 ± 0.86 (max 6; 58/200 lives ≥1; 1/200 ≥5; per-seed means 0.0–1.8) | 0 | n/a | energy 200/200 | −1.00 (all) |
| d8 | action held 8 | 539.4 ± 53.6 (493–775) | 1.16 ± 1.46 (max 7; 110/200 lives ≥1; 9/200 ≥5; per-seed means 0.4–2.1) | 0 | n/a | energy 200/200 | −1.00 (all) |
| d32 | action held 32 | 606.9 ± 78.5 (560–1121) | 1.01 ± 1.84 (max 13; 87/200 lives ≥1; 9/200 ≥5; per-seed means 0.0–1.9) | 0 | n/a | energy 200/200 | −1.00 (all) |
| e | privileged NE.01 forager | 1800.0 ± 0 (200/200 at the cap) | 53.49 ± 0.50 (53–54; 200/200 ≥5; per-seed means 53.3–53.6) | 0 | n/a | censored 200/200 | −0.001 (−0.003 to 0.000) |
| f | hand forager, observation only | 742.1 ± 278.9 (6–1800; median 696) | 4.13 ± 4.98 (max 28; 168/200 lives ≥1; 59/200 ≥5; per-seed means 1.6–7.7, median 4.05) | 0 | n/a | energy 189, integrity 10, censored 1 | −1.00 (199 lives); −0.88 (the 1 censored) |
| g | control: (f) with rays shuffled | 701.5 ± 155.7 (7–1037; median 715) | 2.97 ± 2.05 (max 10; 180/200 lives ≥1; 41/200 ≥5; per-seed means 1.5–3.9) | 0 | n/a | energy 193, integrity 7 | −1.00 (all) |
| f2 | hand forager, F2_BAN_S = inf, seeds 221–240 | 903.7 ± 326.9 (504–1800; median 773) | 6.91 ± 7.47 (max 33; 183/200 lives ≥1; 88/200 ≥5; per-seed means 3.0–13.5, median 6.55) | 0 | n/a | energy 188, censored 8, integrity 4 | −1.00 (192 lives); −0.71 to −1.03 (the 8 censored) |
| g2 | control: f2 with rays shuffled, seeds 221–240 | 668.6 ± 76.5 (592–987; median 644) | 1.69 ± 1.72 (max 9; 146/200 lives ≥1; 13/200 ≥5; per-seed means 0.8–2.9) | 0 | n/a | energy 200/200 | −1.00 (all) |
| b2 | random, white per decision, seeds 221–240 (f2's baseline) | 302.1 ± 19.6 (278–421) | 0.47 ± 0.83 (max 4; 61/200 lives ≥1; 0/200 ≥5; per-seed means 0.0–1.0) | 0 | n/a | energy 200/200 | −1.00 (all) |

Bar for (f), unmoved from `LF_04_FAIL.md`: per seed, mean floor meals per life
≥ 5 and above row b on the same seed, on ≥ 18 of 20 seeds. (g) must miss it.

**Row f against its bar: FAIL — 6 of 20 seeds (needs 18).** `score(f, b)`:
every seed is above random (20/20; pooled 4.13 against 0.52 floor meals per
life, 7.9×), but only seeds 204, 205, 207, 208, 211, 215 reach 5.0 per life.
The bar is not moved and f is not retuned on these seeds. Rows a and b against the
same bar: 0 of 20 each — **the §8 floor-arm gate holds** (a + b = 0 seeds ≤ 1).

**Row g (the control that must fail) fails: 0 of 20 seeds** — the control
works. But it shows how little of f's score is sight. Blind, the same wander
eats 2.97 floor meals per life (5.7× random, above b on 20/20 seeds), and f
beats g on only 14 of 20 seeds. Split by life (paired over the 20 seeds,
post-hoc reading):
- **life 0:** f 12.35, g 2.75 — f ahead on 19/20 seeds, paired t = 5.6. Sight works.
- **lives 1–9:** f 3.22, g 2.99 — f ahead on 11/20, paired t = 0.57. **After its
  first life, f forages no better than blind.** This is cause 1 below, now on
  the registered seeds: once its permanent bans cover the true food textures,
  f stops chasing and is the wander alone.
- **Integrity deaths are the wander's, not sight's:** g has 7 (7–67 s, lives
  2–9), f had 10. Straight full-drive runs kill him in a used world with no
  vision involved — a world/body fact for answer (3) and J2.

**Row a (statue) outlives row b (random) 2.0×: 600 s against 302 s.** Random
flailing draws 2.42× the idle drain (answer 1's table) and its 0.52 meals
cannot repay it. So under any reward that pays for time alive, the statue is
the better of the two floor arms, and a learner's first easy optimum is to
keep still — J2 (3)'s reward option must be judged against that.

What the 200 lives show, and why f missed (the diagnosis is on smoke seed 154,
box, `/data/tmp/wc_diag` — tuning seed, NOT evidence):
- **He eats in his first life and not after.** Mean floor meals by life index:
  12.35, 6.45, 4.70, 3.00, 2.75, 2.70, 2.15, 2.55, 2.15, 2.55. Life 0 alone:
  12.35 ± 7.77 meals, 1,147 s, ≥ 5 meals on 17 of 20 seeds (post-hoc reading,
  not the registered bar). Lives 1–9: 3.22 meals, 697 s. The smoke that tuned
  f ran 2 lives per seed, so it never saw this.
- **Cause 1, the ruler:** f's memory carries across lives and its bans are
  permanent. On seed 154 it learned both true food textures (0.43, 0.80) in
  life 0, and by the end of life 1 had demoted and banned both (3 misses in a
  row each). In life 2 it chased something on 4% of decisions (63% in life 0)
  and ate 3 meals, all by bumping into food while wandering.
- **Cause 2, the world:** floor food is a loose body that regrows where it
  lies, and he eats it by driving into it — so every meal shoves it. On seed
  154 food 0 went from (−1.0, −0.4) to (−5.8, −3.4), against a wall, in life 0
  and never left; food 1 ended against another wall (y = 5.75) in life 1. A
  forager given a blank memory each life (diagnostic arm) still ate only 6 and
  4 meals in lives 1 and 2 against 23 in life 0, chasing 99% of the time, and
  mislearned which texture pays (credited 0.25 / 0.24, banned 0.80).
- **Even the best case does not sustain him.** Life 0 pooled intake is
  0.86e-3 /s = 0.52× the idle drain; all 200 lives pooled 0.45e-3 /s = 0.27×.
  One life in 200 reached the 1,800 s cap (seed 215, 28 meals).
- **Ten lives ended in 6–57 s by integrity (injury), not hunger** (9 seeds),
  all at lives 1–9, none in row b's 200. A full-drive forager respawning into
  a used world can kill himself in seconds; cause not yet traced.
- **Next, by rule (a FAIL is fixed, not filed):** f stays FAILED on 201–220.
  A repaired ruler (bans that expire; food memory re-earned each life) is a
  new row `f2`, tuned on 153/154 over all 10 lives, its rule committed first,
  run on fresh seeds 221–240 against the same unmoved bar with its own
  shuffled control. The wall-pinned food is a world fact for J2 (1): the
  record's "instant re-placement at a new random place" removes it.


### f2 tuning smoke (box, seeds 153/154, 10 lives each — tuning seeds, NOT evidence)

2026-11-15 19:25–19:45, `world_card row f2 <seed>`, F2_BAN_S = 200 s, raw
`world_card_data/smoke_f2_ban200.jsonl`. (lifetime s, floor meals) per life; all energy deaths.

| seed | lives 0–9 floor meals | mean | lifetimes s |
|---|---|---|---|
| 153 | 2, 5, 8, 7, 4, 5, 1, 1, 2, 3 | 3.8 | 604–978 |
| 154 | 9, 6, 5, 5, 4, 4, 0, 5, 3, 2 | 4.3 | 597–1011 |

- **The first-life collapse is gone** (f on 201–220: 12.35 then ~2.5): f2 eats
  about the same in every life. Re-earning memory each life works.
- **But it eats less in life 0:** on 154, f ate 23 in life 0 and f2 ate 9. f2
  differs from f only in the timed ban, so inside a life a 200 s ban costs
  more than it saves: f2 returns to decoy textures every 200 s.
- Both seeds stay under 5.0. Next tuning arm: ban for the rest of the life
  (F2_BAN_S = inf; still cleared at new_life) — the diagnostic "fresh" arm,
  which on 154 ate 23 → 6 → 4, the decline then being the wall-pinned food
  (cause 2). If that arm also stays under 5 on lives 1–9, the ruler is
  limited by the world, not by its rules, and J2 (1)'s re-placement is the fix.

**In-life-ban arm (F2_BAN_S = inf), 2026-11-15 19:46–20:09**, box, raw
`world_card_data/smoke_f2_baninf.jsonl`. All 20 lives energy deaths.

| seed | lives 0–9 floor meals | mean | lives 1–9 | lifetimes s |
|---|---|---|---|---|
| 153 | 9, 1, 7, 4, 11, 8, 4, 20, 2, 10 | 7.6 | 7.4 | 646–1527 |
| 154 | 23, 5, 0, 12, 8, 2, 5, 7, 2, 13 | 7.7 | 6.0 | 599–1673 |

- Life 0 is f's again (154: 23; 153: 9, both identical to f's smoke lives —
  the same code path until the first new_life), and later lives no longer
  collapse: lives 1–9 eat 6.0–7.4, against f's ~2.5 on 201–220 and the
  200 s arm's 3.5. **The collapse was the ruler's rules, not the world.**
- Wall-pinned food still shows as spread, not as decline: lives of 0–2 meals
  sit beside lives of 12–20. J2 (1)'s re-placement may tighten it; it is no
  longer the limit.
- **Frozen: F2_BAN_S = inf** (commit with this table). f2, g2, b2 launch on
  fresh seeds 221–240 against the unmoved bar. Two tuning seeds at ~7.6 are
  not a prediction of 18/20 — the registered run decides.

**Row b2 (22:50):** random on f2's fresh seeds 221–240 reproduces row b on 201–220 — 302 s vs 302 s, 0.47 vs 0.52 floor meals/life, per 100 s alive 0.156 vs 0.170: the new seeds are not an easier world. 0/20 seeds reach 5 per life, so b2 is the per-seed floor f2 is scored against; f2 and g2 still running.

**Row f2 against the unmoved bar (22:40): FAIL — 13 of 20 seeds (needs 18); control g2 misses, 0 of 20.** `score(f2, b2, F2_SEEDS)`. Short of 5.0 per life: 223 (3.0), 225 (4.9), 226 (4.2), 228 (3.4), 234 (4.0), 236 (4.5), 237 (3.4). Every seed is above b2 (20/20, paired t 9.31; pooled 6.91 vs 0.47, 14.7×).
- **Sight now carries every life, not just the first.** f2 beats its blind control on 20/20 seeds (t 8.02). Life 0: f2 11.25 vs g2 2.30 (17/20, t 4.28). Lives 1–9: f2 6.43 vs g2 1.62 (20/20, t 6.67) — on f/g it was 3.22 vs 2.99 (t 0.57). Removing the bans fixed the collapse on fresh seeds, as the tuning smoke said. By life: 11.25 10.15 8.05 7.60 4.20 6.10 5.75 6.60 4.20 5.20 — still a slow decline, now from a much higher floor.
- **[CORRECTED 11-16 under answer (1): these 8 lives ate below basal drain and were ended by the cap, not sustained.]** **Floor food CAN sustain a full life: 8 of 200 lives reach the 1800 s cap alive** (27–33 floor meals each, seeds 221/222/224×2/227/233/235/236). That is a measurement for answer (1): the arithmetic says intake can beat drain, and an observation-only ruler shows it does, sometimes. Typical: 0.77 floor meals per 100 s alive (b2 0.16, g2 0.25).
- **Blind wander got worse, not better, on these seeds:** g2 1.69/life vs g's 2.97 — the f/g gap was partly that g's wander ate more on 201–220. 4 integrity deaths in f2 (504–915 s, lives 0–9), none in g2.
- **Answer (3), provisional until e lands:** 16 rays DO tell food from walls and decoys well enough to beat blind 4× per life — but not reliably enough for 5 meals/life on 7 of 20 seeds. Spread is per-world (3.0–13.5 per-seed means), so the misses are layouts, which J2 (2)'s random layout each life will mix.
- Bar stays; no retune on 221–240. A further ruler would need new fresh seeds and its rule first — not this order's step.

## Smoke (box, seeds 153/154 — tuning seeds, NOT evidence)

2026-11-15, box, nice 19, `python -m experiments.world_card smoke 2 f|g`,
2 lives per seed, forager as committed with this page. Lives: (lifetime s,
floor meals). Every life died of energy; no apples.

| row | seed 153 | seed 154 |
|---|---|---|
| f hand forager | (1022.8 s, 9), (690.8 s, 3) | (1672.6 s, 23), (1299.2 s, 17) |
| g rays shuffled | (862.2 s, 6), (695.4 s, 3) | (647.8 s, 3), (641.4 s, 2) |

What the smoke already shows (tuning seeds, not evidence; the registered rows
decide):
- **Sight is used:** on 154, f eats ~8x the shuffled control. On 153 the gap
  is small (6.0 vs 4.5 per life) — the registered 20 seeds say which is typical.
- **The control is not starving:** shuffled rays still eat 2–6 meals per life,
  because the forager's wander is straight-line runs across the room. Steady
  motion with no sight finds food far more often than random's ~0.4 floor
  meals per life (step 2's 15 lives). That is the record's correlated-noise
  prediction showing in the control; rows c/d measure it properly.
- **Floor food nearly sustains a good forager:** seed 154's first life ran
  1,673 s (2.8x the statue's 600 s) on 23 floor meals; a 6,000-decision
  diagnostic on 153 held energy between 0.27 and 0.37 for its last ~500 s (pre-v2 forager).
- **Answer (2) is already fixed by arithmetic:** every life that ended in death
  summed to exactly −1.000 reward, whatever its length or meals. The default
  reward (`d(h_t) − d(h_{t+1})`, death charging to `D_DEATH`) telescopes: a
  life's undiscounted sum is `d(h_birth) − D_DEATH`, the same for a 600 s
  starving life and a 1,673 s fed one. Undiscounted, eating is worth nothing
  and dying early costs nothing; any preference comes only from the learner's
  discount. J2 (3)'s reward option must break this.
- Tuning history on 153/154 (all before any registered seed was run): v1
  chased wall-corner single-ray "blobs" forever → timed-out pursuits now count
  as misses; v2 credited meals bumped en route to the chased texture and
  starved chasing 3 false "foods" on 154 → credit only within 0.6 m, demote a
  food texture after 3 straight misses.

## Row c against b (harvested 20:35; same seeds 201–220)

Pink noise per life outlives white noise by 63% (491.5 vs 301.7 s) but does
not eat more: 0.44 vs 0.52 floor meals/life, per-seed paired difference
−0.07 (t = −0.70), c above b on 7/20 seeds; 0/20 seeds reach f's bar.
The §1 prediction (correlated noise finds 2.5–4× more food) does NOT hold for
pink noise on this world. The longer life is presumably a cheaper action
stream (less arm power; NOT measured) — it is not food found. d8/d32 owed.

## Row d8 against b and c (harvested 21:25; same seeds 201–220)

Holding each random action for 8 decisions eats **1.16 vs 0.52** floor
meals/life (b): per-seed paired difference +0.64, t = 7.29 (p 6e-7), d8 above
b on 19/20 seeds, ratio **2.24×**; against pink c 2.62× (t 4.93, 18/20).
Most of it is a longer life (539 vs 302 s): per 100 s alive d8 eats 0.212 vs
b 0.170 (1.25×, paired t 2.59, 15/20). The §1 prediction (2.5–4× more food
for correlated noise) holds roughly for held actions per life (2.24×, just
under the band) and not for pink noise (row c). Why held arms live longer is
NOT measured (guess: slides saturate and stop drawing power). 0/20 seeds reach
f's bar; d8 is now the best random row, so J2's d0 is measured against it.

## Row d32 against b, c and d8 (harvested 22:10; same seeds 201–220)

Holding each random action for 32 decisions eats **1.01** floor meals/life:
1.96× b (paired t 3.96, 15/20 seeds), 2.30× c (t 4.08, 18/20), and **0.87×
d8** (t −1.12, above d8 on 6/20 — not distinguishable from d8 per life). Its
life is the longest of the random rows (607 s, min 560) but per 100 s alive
it eats 0.164 — the same as white b (0.170; t −0.25, 9/20) and below d8
(0.212; t −2.29, 6/20). So the whole d32 gain over b is a longer life, none of
it food found faster; holding longer than 8 buys life, not foraging. Meals are
lumpier (sd 1.84, max 13; 113/200 lives eat nothing vs d8's 90). **d8 stays the
best random row** (higher per life point estimate and per 100 s), so J2's d0
rule keeps pointing at d8. Why held arms live longer stays NOT measured.

## Answers (written 2026-11-16 after all registered rows)

1. Intake against drain, from `drives.py`, beside row e's lifetime.
   **Arithmetic, now on the rover's own body** (2026-11-15, box probe, seeds
   153/201, 550 decisions each after a 50-decision settle, `last_power_w`):
   | policy | mean power W | drain /s | × basal |
   |---|---|---|---|
   | zero action (row a) | 0.0 | 1.667e-3 | 1.000 |
   | duty 0.10, ±0.4 (row e's idle) | 30.3 / 23.4 | 1.737e-3 / 1.721e-3 | 1.04 |
   | uniform(−1,1) every decision (row b) | 1018.6 / 1029.0 | 4.03e-3 / 4.06e-3 | 2.42 |
   Supply if BOTH floor foods are eaten the instant they regrow:
   2 × 0.08 / 66.9 s = **2.392e-3 /s** — 1.44× an idle rover, 1.38× row e's
   idler, 0.59× a full random flailer (break-even: random arms ≤ 30% of
   decisions). The horizontal drive is an applied force (`xfrc_applied`),
   not an actuator, so **driving costs nothing**: only the four arm slides
   draw `KAPPA` power. So floor food CAN sustain a rover that drives with its
   arms still — but only at near-perfect harvest: one meal every 33 s. Smoke
   f's best life ate one per 73 s (1.1e-3 /s, 0.66× basal) and starved.
   Row e (food served when due) measures whether the ceiling holds in the
   physics: probe, seeds 153/201, 1 life × 1,500 decisions: 9 serves → 9
   meals, energy held at 1.00, censored.
   **Registered row e (seeds 201–220, 200 lives): every life reaches the
   1,800 s cap**, 53.49 floor meals each = 2.377e-3 /s, **99.4% of the
   2.392e-3 ceiling** — the physics delivers the arithmetic: no meal is lost
   to contact, clocks or the serve. Energy stays near full (summed reward
   −0.001). As a ruler it passes the bar on 20/20 seeds.
   **Beside it, the observation-only ruler: no f2 life was sustained.** The 8
   f2 lives that reached the cap ate 27–33 meals = 1.20–1.47e-3 /s, **0.72–0.88×
   an idle rover's drain** (f2 drives by force, so its drain is at least
   basal). Each was in deficit the whole way and lived on its birth energy
   (1.0 + 2.4 eaten − ≥3.0 drained ≈ ≤0.4 left at 1,800 s); with no cap they
   would starve later. So the 22:40 line "floor food CAN sustain a full life"
   is corrected: the cap, not the food, ended those lives.
   **Answer (1): floor food alone CAN sustain him, with a margin of 1.44×
   — but only at ≥ 70% of perfect harvest (one meal per 48 s, arms still).**
   Measured harvest: e 99%; f2's best life 61%, typical f2 0.77 meals/100 s
   alive = 26%; best random row d8 7%. No non-privileged policy has yet
   sustained a life. The gap e → f2 (2.97 vs 0.77 meals per 100 s, 3.9×) is
   the cost of finding and reaching food, which J2's dial exists to shrink.
2. Does dying early pay (summed reward against lifetime)?
   **Yes — under today's reward, dying never costs more than living hungry.**
   All 1,991 deaths on this card (rows a–g2, b2; checked to ±0.005) sum to −1.00 whatever its
   length or meals (the telescoping sum, smoke section). A life that reaches
   the cap sums `d(h_birth) − d(h_end)`: row e, fed, ≈ 0; but f2's eight
   capped lives sum −0.71 to −1.03, and **seed 221 life 2 (27 meals, 1,800 s)
   sums −1.026, below every death.** Undiscounted, eating is worth at most
   +1 over a whole life and a hungry life can be worth less than dying; with
   any discount the drain's early cost dominates. This is the §3 setting in
   which published agents learned to end their lives. J2 (3)'s
   `dr100-eat` / `dr100-eat-pos` options (`14e637c4`) pay on the eating event
   and keep "alive and hungry" non-negative; a learner is judged under one of
   them, never under today's default.
3. Can 16 rays tell food from a wall; if not, which sense is missing?
   **Yes — the rays tell food from walls and decoys; no missing sense is
   shown.** With sight, f2 eats 6.91 meals/life against 1.69 with its rays
   shuffled (20/20 seeds, t 8.02; lives 1–9 6.43 vs 1.62, t 6.67), and the
   control's collapse is the measured value of the 16 rays. What they do not
   yet buy is RATE: f2 harvests 26% of the ceiling against the ≥ 70% a
   sustained life needs, and per-seed means spread 3.0–13.5 by layout. That
   gap is search and locomotion (food shoved against walls by eating, cause
   2; straight full-drive runs that injure him, 4 integrity deaths in f2),
   not discrimination. Not measured here and so not claimed: whether a
   richer sense (smell gradient, wider field) would close it — a question
   for a learner at J2's `d0`, not a further ruler. The foragers stay
   rulers, never parts of him.

**J1 verdict.** The venue can tell a forager from a non-forager (floor arms
a, b, c, d: 0/20 seeds; e 20/20; f2 13/20 with its control 0/20), and the food
can sustain a life only near perfect harvest. No random row survives a life
(0/1,000 random lives reach the cap). **Flag for J2 (5):** its rule as
written — the hardest d at which the best random row eats ≥ 1 meal in ≥ 50%
of lives and do-nothing still dies — is ALREADY met at d = 1: d8 eats ≥ 1 in
110/200 lives (55%) and a dies 200/200. Read literally, `d0` = 1 = today's
world, the one this card shows no non-privileged policy can live in. The
rule is not changed here (it is the Director's); INBOX carries the question.

## J2 item 5 — the entry rung `d0` (rule `8af5f2e1`, registered 2026-11-16 00:2x; d0 and d_sus named 04:40, d=0.5 harvested 05:21, d=0.25 05:39, d=0 06:10 — complete)

Rule in `experiments/world_card.py` docstring ("THE ENTRY-RUNG RULE"), committed
before any run. Dial only (`JackEnv(difficulty=d)`, fixed layout, needs
default), seeds 201–220 × 10 lives, cap 9,000 decisions, rows a (do-nothing)
and d8 (held-8 random, J1's best random row; same policy streams as J1).
d = 1 is J1's rows a/d8 (at d = 1 regrowth is 66.9 s, so every eat is its own
bout: d8 110/200 lives ≥ 1, a 0/200 capped — the rule holds there). Per life,
new: `floor_eats_all` (every food the dial made) and `floor_bouts` (eats more
than 5 decisions after the last eat of the same food — d = 0's resting-body
eats counted once). Reported: `d0` as written, and beside it `d_sus` (largest
d with ≥ 1 capped d8 life) with the capped share per d.
Kaggle CPU, one kernel per d: d=0.75 `jack-cpu-1794788783`, d=0.5
`jack-cpu-1794788798`, d=0.25 and d=0 launched 00:28 (logs
`/data/jack-logs/world_card/entry_dXX.log`, lives → `entry_dXX.jsonl`).

**Harvest 1 of 4 (2026-11-16 04:33, d = 0.75).** Table so far (200 lives per cell):

| d | a: lives capped | a: mean life s | d8: lives ≥ 1 bout | d8: capped | d8: bouts/life | d8: eats(all)/life | d8: mean life s |
|---|---|---|---|---|---|---|---|
| 1 (J1) | 0/200 | 600 | 110/200 (55%) | 0/200 | 1.16 (meals) | 1.16 | 539 |
| 0.75 | 0/200 | 600 | 138/200 (69%) | 2/200 (seeds 204, 218: 17 and 11 bouts) | 2.40 | 2.40 | 694 |
| 0.5 | 1/200 (seed 214 life 1: 54 bouts, food re-placed onto/next to the resting body) | 606 | 137/200 (68%) | 7/200 (seeds 201 205 207 209 213 216 218: 11–23 bouts) | 3.32 | 3.32 | 759 |
| 0.25 | 6/200 (seeds 205 206×2 211 215 218; 5–108 bouts) — **over the ≤ 1 bound** | 636 | 142/200 (71%) | 13/200 (11–36 bouts) | 5.04 | 5.04 | 823 |
| 0 | 24/200 (14 of 20 seeds; most on 1–4 bouts — food re-placed onto the resting body, one 'bout' of up to 8,990 eats) — **over the ≤ 1 bound** | 750 | 169/200 (85%) | 11/200 (seeds 201 204×2 208×3 211 212×2 215 218: 10–35 bouts) | 6.70 | 177.1 | 834 |

**Harvest 2 of 4 (2026-11-16 05:21, d = 0.5, `jack-cpu-1794788798`).** d8
holds the ≥ 50% clause (68%) and a's capped count is 1/200 — exactly AT the
rule's "≤ 1 of 200" floor-arm bound, so "a still dies" holds as written, with
no margin: at d = 0.5 the dial's random re-placement alone fed one motionless
body for 1,800 s. Below d = 0.5 the do-nothing arm is expected to cross the
bound (d = 0 re-places food instantly); that is what the last two rows test.
d8's capped share rises 0/200 → 2/200 → 7/200 (d = 1, 0.75, 0.5) while
"≥ 1 bout" stays flat (55% → 69% → 68%): easing the dial buys longer lives
for the lucky, not a first meal for more of them.

**Harvest 3 of 4 (2026-11-16 05:39, d = 0.25, `jack-cpu-1794788920`).** The
do-nothing arm crosses the bound: 6/200 lives reach the 1,800 s cap without
moving (four of them on 108 bouts — food re-placed onto the resting body), so
"a still dies" FAILS at d = 0.25, as the d = 0.5 note predicted. d8: 142/200
(71%) eat ≥ 1 bout, 13/200 capped, 5.04 bouts/life, mean life 823 s. The
first-meal share stays flat (55 → 69 → 68 → 71%) while capped lives climb
(0 → 2 → 7 → 13 of 200). Below 0.5 the dial feeds a statue, so a rung there
cannot tell a learner from luck. d0 and d_sus unchanged; d = 0 owed (`…935`).

**Harvest 4 of 4 (2026-11-16 06:10, d = 0, `jack-cpu-1794788935`).** The
do-nothing arm is capped in 24/200 lives — 4× d = 0.25 and 12% of all lives:
instant re-placement drops food onto a motionless body often enough that
standing still is capped TWICE as often as d8's random walking (11/200). d8
eats ≥ 1 bout in 169/200 (85%, the first real rise in the first-meal share:
55 → 69 → 68 → 71 → 85%), 6.70 bouts/life, but its capped share falls
(13 → 11). At d = 0 `floor_eats_all` parts from `floor_bouts` (a 815.6 eats
vs 1.74 bouts per life): a resting body eats every decision. The bottom of
the dial rewards stillness over search — a rung that pays a statue cannot
teach foraging. All four rows in; d0 = 1 and d_sus = 0.75 as fixed below.

**`d0` = 1 (today's world), as the rule is written.** The rule takes the
LARGEST grid d where both clauses hold; d = 1 holds (55% ≥ 50%, a 0/200), so
no harvest still out can move it. The rung it names is the world in which no
non-privileged policy lives (J1) — the INBOX flag stands: the rule's
lower clause (random eats once) is too weak to name a teachable entry.
**`d_sus` = 0.75**, also fixed now: d = 1 has 0/200 capped d8 lives, d = 0.75
has 2/200, and 0.75 is the next grid point down. Two capped lives in 200 is a
first sign, not a sustaining world: d8 at 0.75 still dies of hunger 198/200,
mean life 694 s (+29% on d = 1), bouts/life 2.1× d = 1; every seed (20/20) has
≥ 5 of 10 lives that eat. The three outstanding rows fill the shares only.

## J12 (2) — the forager gate for the NEW eye, `row128` (FROZEN 2026-11-17, before any gate run; ruling 37)

**Ruler.** Row f2's forager (`HandForager2`, F2_BAN_S = inf), changed only in
that it reads `jack_env(eye="row128")`: 128 world-frame azimuths (row ray 8j
is old ray j), indices after vision shifted +224, and the "small object" blob
rule kept in ANGLES (≤ 2 × 22.5°, growth cap 90° — ≤ 16 / 32 of 128 rays), not
in ray counts. Code `experiments/eye_forager.py`; proof
`tests/test_eye_forager.py` PASS (11-17): at 16 rays its actions equal
HandForager2's bit for bit, 0 of 12,000 decisions differ (seeds 153/154 ×
d 1 / 0.75 × plain / shuffled, meals inside), control (ray 0 nudged 1e-3)
differs on 26; at 128 rays the angle-kept rule reports a food blob on 486/500
decisions vs 317/500 for the unscaled rule.
**World.** Row f2's (default options, shipped calibration), `lazy_contact_audio`
(identical world, ruling 34), at **d = 0.75 (d0)** and **d = 1**.
**Seeds.** Fresh **241–260**, 10 lives each, life cap 9,000 decisions.
**Rows** at each d: **f3** the ruler; **g3** f3 with its 128 rays permuted
fresh every decision (must fail); **b3** white random (the per-seed floor).
**Bar, unmoved:** a seed counts if f3's mean floor meals/life ≥ 5.0 AND > b3's
on that seed; f3 PASSES at d iff ≥ 18 of 20. g3 scored the same must hold on
< 18, else f3's pass at that d is VOID. Judged at d0 and at d = 1 separately.
**Extension clause: none.** A miss is a miss.
**Where.** 2 Kaggle CPU kernels (`eye_forager kernel 0.75`, `… kernel 1.0`),
rank 2 behind J11's card. Smoke (tuning seed 153, d0, 1 life, box): f3 capped
(1,800 s) 19 meals, g3 capped 10 meals, b3 445 s 3 meals; ~44 dec/CPU-s;
worst case ~4.7 h a kernel. Disclosed before the run: the blind twin's 10
meals in one d0 life make "g3 passes at d0 → f3 VOID there" a live outcome.

### J12 (3) — the gate's reading at d = 1 (harvested 2026-11-18 05:02; raw `world_card_data/eye_gate_d100.json` sha256 ee2003a4…; kernel `jack-cpu-1794960628`; scored by `eye_forager score`)

| row (d = 1, seeds 241–260 × 10 lives) | seeds meeting the bar | mean of seed means, floor meals/life (sd; min–max) | mean life, decisions | capped lives |
|---|---|---|---|---|
| **f3** the forager on `row128` | **20 / 20 → PASSES** | 8.83 (1.76; 5.60–12.20) | 4,995 | 10 / 200 |
| g3 shuffled rays (must fail) | 0 / 20 → fails, as it must | 1.68 (0.58; 0.90–2.80) | 3,343 | 0 |
| b3 white random (floor) | — | 0.52 (0.31; 0.00–1.30) | 1,527 | 0 |

Every f3 seed beat its own b3 seed and cleared 5.0 (lowest: seed 246, 5.6).
All 190 uncensored f3 lives still ended by hunger — a forager that eats ~9
floor meals a life is not yet a forager that lives on them. **At d = 1 the
food is findable from the 128-ray row** (the old 16-ray eye: 6 and 13 of 20
on the same bar, ruling 23). Adult-taught ceiling only: it says the sense
carries the food, not that a brain learns to use it. d = 0.75 (d0): next section.

### J12 (3) — the gate's reading at d = 0.75, the entry rung (harvested 2026-11-18 07:10; raw `world_card_data/eye_gate_d075.json` sha256 417e2c69…; kernel `jack-cpu-1794960563`; scored by `eye_forager score`)

| row (d = 0.75, seeds 241–260 × 10 lives) | seeds meeting the bar | mean of seed means, floor meals/life (sd; min–max) | mean life, decisions | capped lives |
|---|---|---|---|---|
| **f3** the forager on `row128` | **20 / 20 → PASSES** | 13.72 (1.57; 10.10–17.20) | 8,422 | 156 / 200 |
| g3 shuffled rays (must fail) | 8 / 20 → fails, as it must | 4.84 (1.05; 3.50–6.50) | 6,418 | 93 / 200 |
| b3 white random (floor) | — | 0.46 (0.27; 0.00–1.00) | 1,648 | 0 |

At the entry rung food is so near that even the blind twin eats 4.8 meals a
life and reaches the bar on 8 seeds (disclosed before the run) — sight adds
~9 meals a life on top. 156 of 200 forager lives reached the 9,000-decision
cap alive; the 44 that ended: 38 hunger, 6 integrity.

**BOTH J12 (3) rows above (d = 1 and d = 0.75) are "compass eye — ceiling
only" (ruling 38).** They were earned on an eye cast at WORLD azimuths by a
forager that pushes along WORLD axes. Nothing a brain is asked to learn is
sized from them; the body-frame gate below is the row that counts.

## J12 (0) — the body-frame eye and motor (2026-11-18; `jack_env(eye_frame=, motor_frame=)`, proof `experiments/tests/test_jack_env_eye_frame.py` PASS)

- `eye_frame="body"`: ray k of either eye is cast at 2πk/n + the body's yaw
  (the yaw his ears already use); default `"world"` is byte-identical to
  before (8/8 runs). At birth (yaw exactly 0) the two frames' looks are
  bit-equal; a quarter turn in place rolls the body-frame look by n/4 on
  20/20 stops for both eyes, and leaves the world-frame look unchanged 20/20.
- **How much compass the old eye gave:** under random actions at the entry
  rung he sits a median 81° from his birth heading (p90 145°; beyond one
  16-ray step on 90% of decisions).
- **J12 (1): does the hand forager f3 use heading it is not given? YES.**
  `experiments/eye_forager.py`, `EyeForager._act`: the two `rel =
  np.array([math.cos(ang), math.sin(ang)]) * dist` lines place a seen blob
  along WORLD axes, `a[6:8] = r / nr` and the wander line `a[6:8] =
  [math.cos(self.ray_ang[self.heading]), …]` push along them, and
  `self.target["rel"] - v * WC.DT` dead-reckons with a WORLD-frame velocity.
  The cause is in the world, not only the ruler: W0's drive is a force along
  WORLD x, y (`w0.py` `decide`, `xfrc_applied[self.rover_bid, :2] = force`)
  and proprio's root velocities are WORLD x, y — two more compass channels
  that `eye_frame` alone leaves open. **Any learner given the body-frame eye
  and today's motor is asked to aim a compass push at food it sees relative
  to itself, with no sense of heading.**
- `motor_frame="body"` (default `"world"` byte-identical, 2/2): the drive is
  (forward, left) along his heading, turned once per decision; proprio's two
  velocities likewise. Facing +y, "forward" moves him +0.37 m along +y
  (world-motor control: +0.38 m along +x).

## J12 (2′) — GATE 2: the same gate on the body-frame eye (FROZEN 2026-11-18, before any gate-2 cell ran; full text in `experiments/eye_forager.py`)

Same ruler line for line, same bar (≥ 5 floor meals/life and above the
random floor on ≥ 18 of 20 seeds), shuffled twin must miss, **fresh seeds
261–280**, d = 0.75 and d = 1 judged separately, **no extension**. The one
change, said before the run: the mapping lives in the world's interface
(`motor_frame="body"`), not in the ruler, and the ruler is not told its yaw.
Rows: **f4** body eye + body motor (the row), **g4** its shuffled twin
(must fail), **b4** white random (floor), **h4** body eye + WORLD motor —
reported only, predicted to miss, there so "does the motor frame matter" is
measured. World: gate 1's, through `lazy_contact_audio` + `native_substep`.
Box smoke (seed 153, one life capped at 2,000 decisions, d 0.75 — not a
reading): f4 3 meals, g4 2, h4 0, b4 0 (died at 1,454).

### J12 (2′) — gate 2's reading at d = 0.75, the entry rung (harvested 2026-11-18 12:10; raw `world_card_data/eye_gate2_d075.json` sha256 af51fece…; kernel `jack-cpu-1794987421`; scored by `eye_forager score2`)

| row (d = 0.75, seeds 261–280 × 10 lives) | seeds meeting the bar | mean of seed means, floor meals/life (sd; min–max) | mean life, decisions | capped lives |
|---|---|---|---|---|
| **f4** body eye + body motor (the row) | **20 / 20 → PASSES** | 7.53 (1.31; 5.60–11.50) | 7,494 | 88 / 200 |
| g4 shuffled rays (must fail) | 0 / 20 → fails, as it must | 2.17 (1.06; 0.40–4.90) | 4,574 | 29 / 200 |
| b4 white random (floor) | — | 0.45 (0.25; 0.10–1.10) | 1,646 | 0 |
| h4 body eye + WORLD motor (reported only) | 0 / 20 → misses, as predicted | 0.81 (0.49; 0.00–2.10) | 3,480 | 0 |

**At the entry rung, food is findable from the body-frame 128-ray row by a
ruler that is never told its heading.** Every f4 seed beat its own b4 seed
and cleared 5.0 (lowest: seed 275, 5.6). All 112 uncensored f4 lives ended by
hunger. Beside the compass row (f3 13.72 on seeds 241–260): taking the
compass away costs ~6 meals a life — the frozen ruler's dead-reckoning and
wander were written for world axes and were not re-tuned (said before the
run). **The motor frame matters:** the same body eye with the world-axis
push (h4) eats 0.81, barely above random — a body eye needs a body motor.
The shuffled twin fell from 4.84 (gate 1) to 2.17, so the margin between
sight and no sight is wider here than on the compass row.
Adult-taught ceiling only. d = 1: next section.

### J12 (2′) — gate 2's reading at d = 1 (harvested 2026-11-18 18:10; raw `world_card_data/eye_gate2_d100.json` sha256 f69b032f…; kernel `jack-cpu-1795012722`; scored by `eye_forager score2`)

| row (d = 1, seeds 261–280 × 10 lives) | seeds meeting the bar | mean of seed means, floor meals/life (sd; min–max) | mean life, decisions | capped lives |
|---|---|---|---|---|
| **f4** body eye + body motor (the row) | **3 / 20 → FAILS** (needs 18) | 3.92 (1.01; 2.60–6.00) | 3,877 | 0 / 200 |
| g4 shuffled rays (must fail) | 0 / 20 → fails, as it must | 1.24 (0.40; 0.20–1.90) | 3,232 | 0 / 200 |
| b4 white random (floor) | — | 0.47 (0.16; 0.20–0.70) | 1,519 | 0 / 200 |
| h4 body eye + WORLD motor (reported only) | 0 / 20 → misses, as predicted | 0.74 (0.33; 0.20–1.70) | 3,133 | 0 / 200 |

**At d = 1 the frozen gate FAILS on the body-frame eye.** Seeds 262, 264,
280 cleared 5.0; 70 of 200 f4 lives ate ≥ 5. No extension (frozen). Sight
still carries food — f4 beat its own b4 AND its own g4 seed on 20/20 — but
not enough of it for this ruler at the harder rung. All 800 lives ended by
hunger. Beside the compass row at d = 1 (f3 8.83, 20/20): taking the
compass away costs ~4.9 meals a life (d0.75: ~6.2). What the reading does
NOT say: that the body-frame eye cannot carry d = 1 — the ruler's
dead-reckoning and wander were written for world axes and not re-tuned
(said before the run), so this is the frozen ruler's miss, a lower bound on
the eye. What it does say: the stage-1 proof stands at the entry rung only
(d0 = 0.75, 20/20); any claim at d = 1 on the body-frame eye is unproven.
Adult-taught ceiling only.

### J15 (2) — gate 3's reading at d = 0.75: f5, the ruler a memoryless net could copy (harvested 2026-11-19 12:10; raw `world_card_data/eye_gate3_d075.json` sha256 54c77266…; kernel `jack-cpu-1795073402` at `9882df51`; scored by `eye_forager score3`)

| row (d = 0.75, seeds 281–300 × 10 lives) | seeds meeting the bar | mean of seed means, floor meals/life (sd; min–max) | mean life, decisions | capped lives | reported only: all floor foods, meals/life (seeds ≥ 5 and > b5) |
|---|---|---|---|---|---|
| **f5** body-frame row128 → action, no memory, no draw | **5 / 20 → MISSES** (needs 18) | 3.57 (2.89; 0.30–10.00) | 3,893 | 21 / 200 | 5.22 (8 / 20) |
| g5 shuffled rays (must fail) | 0 / 20 → fails, as it must | 2.00 (0.76; 0.80–3.60) | 4,523 | 7 / 200 | 3.03 (1 / 20) |
| b5 white random (floor) | — | 0.50 (0.29; 0.10–1.00) | 1,656 | 0 / 200 | 0.77 |

**THIS memoryless hand rule (f5) does not meet the bar at d0: 5 of 20**
(seeds 283, 284, 287, 289, 297). No extension (frozen). It still carries
food: it beat its own b5 seed on 18/20, but its own shuffled twin on only
12/20. **How it eats: by sitting.** 739 of f5's 1,044 eats (71 %) fall in the
21 lives that reached the 9,000-decision cap, nearly all of them at 23–39
eats, parked on food that regrows under it. Every other life ended in
hunger. The "meals from one food" figure promised before the run cannot be
read from these fields: eats = bouts on all three rows (1,044 = 1,044 for
f5), because the 5-decision bout gap (`world_card.BOUT_GAP`) is shorter than
regrowth, so each regrown meal counts as a new bout. The capped lives are
the stand-in. The bar counts `obj0`+`obj1` only, the same as gate 2 where f4
passed 20/20. The dial's extra foods are shown in the last column, reported
only, and do not change the reading. Beside f4 on gate 2 (7.53, 20/20, with
a remembered target and a held random wander): taking memory and the coin
away costs ~4 meals a life and 15 seeds.
What it says: **E2b (cloning a forager into his network) closes as "not
clonable by a memoryless net, by measurement" for the only memoryless
teacher we have.** What it does NOT say: that no function of the row can
find food. f5 is one hand rule that cannot tell food from other small things
(texture shifts per world) and never gives a thing up. A better memoryless
rule might pass. Finding food on his eye at this bar has so far needed
memory, which a recurrent student could learn, but that is not tested here.
Adult-taught ceiling only.

## J15 — E2c, the two `auto` arms (ruling 57) — PARTIAL: rows only, NO branch read (2026-11-20 21:45 UTC)

Both `auto` arms are complete at 5 of 5 seeds (801–805, 300k decisions,
kernels a0–a2). They are written here beside their control now. The frozen
fixed-α arms are 21 of 25: E-random_point s803/s804 and E-food/E-zero s804
are still out, in kernel e2 (`jack-cpu-1795192704`, RUNNING at 21:40).
`e2c-readings` refuses a partial book (`PARTIAL — no reading: 4 of 25
missing`), so **no branch of ruling 57's table is named in this section**.
The numbers come from `$PY -m experiments.pilot_b2 e2c-table`, 21:40.

| arm | seeds | lives | sustained lives (seeds with any) | late meals/life per seed | seeds > K-lr0's 0.82 | late meals/life, arm | % lives fed early → late |
|---|---|---|---|---|---|---|---|
| E-food-auto | 801–805 | 367 | 30 (5/5) | 0.25, 2.28, 1.63, 2.86, 0.84 | 4/5 (s805 by 0.02) | 1.57 | 72 → 47 |
| E-zero-auto (its control) | 801–805 | 499 | 0 (0/5) | 0.00, 0.17, 0.04, 0.00, 0.08 | 0/5 | 0.06 | 60 → 3 |

W crossings (W=1/2/3): food-auto 5/3/2, zero-auto 0/0/0. So "eats" holds on the
late-meals clause (4/5) but not on SUSTAINED at W=2 (3/5). That is the seam
that `e2c-readings` prints rather than choosing. The control fails cleanly:
the handed direction carries the eating. As in every food run, lives are won
early (first capped life 18–38k) and lost late: the last capped life is at
89–189k, and late meals/life is below early in 5/5. This is the same
win-then-lose shape the fixed arms show. Ruling 57's table is read only when
all 25 runs are in.

## J15 — E2c, the full book: 25 of 25, READ (2026-11-20 22:10 UTC)

All five arms are home: seeds 801–805, 300k decisions, kernels e0–e3 and a0–a2. The
last kernel is e2 (`jack-cpu-1795192704`, harvested 21:55). The numbers come from
`$PY -m experiments.pilot_b2 e2c-readings` and `e2c-table`, run at 22:05. No run
crashed and none was short. The two readings, J11 (sustained on ≥ 3/5) and STRICT
(≥ 4/5), give the same verdict on every arm, so `e2c-readings` prints no seam.

| arm | role | lives | sustained lives (seeds W=2) | late meals/life per seed | seeds > 0.82 | late meals/life, arm | EATS |
|---|---|---|---|---|---|---|---|
| E-food (fixed α 0.1) | arm | 376 | 33 (4/5) | 1.00, 3.47, 0.57, 0.40, 0.58 | 2/5 | 1.20 | yes (sustained) |
| E-zero (fixed) | its PRIMARY control | 500 | 1 (0/5) | 0.32, 0.00, 0.16, 0.04, 0.33 | 0/5 | 0.17 | no |
| E-random_point (fixed) | reported only | 496 | 0 (0/5) | 0.04, 0.50, 0.17, 0.16, 0.08 | 0/5 | 0.19 | no |
| E-food-auto | arm | 367 | 30 (3/5) | 0.25, 2.28, 1.63, 2.86, 0.84 | 4/5 | 1.57 | yes (late meals) |
| E-zero-auto | its control | 499 | 0 (0/5) | 0.00, 0.17, 0.04, 0.00, 0.08 | 0/5 | 0.06 | no |

**Rule E2c's own reading clause: PASS.** E-food sustains on 4/5 seeds and its
control E-zero on 0/5, which is below 3. By the rule's frozen text this means "the
task is learnable by B when the direction is given → a miss of B0 in J11 is the
eye". Pilot 2 had no learner eat. Handed the direction to food, the same SAC
learner wins capped lives on 5/5 seeds in both α settings. Its controls, given the
same inputs carrying nothing, win 1 such life in 10 runs between them.

**Ruling 57's table: NO BRANCH.** Both food arms eat, and neither zero control eats.
None of the three written branches fires: "auto eats while fixed does not", "both
miss", and "a zero arm eats". So the fixed 0.1 is **not shown to be the wall**.
Fixed α eats too: 33 sustained lives against auto's 30. Auto's late meals/life are
higher (1.57 vs 1.20; 4/5 vs 2/5 seeds over 0.82), but no rule reads that
difference, and at 5 seeds it is not a finding. The table names no consequence for
this outcome, so the Director rules it (INBOX).

**Named findings beside the rows (descriptive, no reading):**
- *Win-then-lose, every food run, both settings.* At arm level, early meals/life
  far exceeds late meals/life: fixed 9.84 → 1.20, auto 7.72 → 1.57. Per run, meals/life
  over the first 20 lives against the last 20 falls in 10/10 food runs: fixed
  9.9–26.6 → 0.4–3.0, auto 8.9–23.9 → 0.3–3.4. In fixed runs the last capped life
  comes at 81–118k; in auto runs it comes at 89–189k. "Learnable" here means *learned, then lost*. The loss is the
  next question. It is not a capability.
- *Entropy.* The fixed 0.1 does not hold the policy wide. H falls from 5.4 to ≤ 0 by
  15–35k and reaches −10…−18 at 300k in every fixed run. Auto holds H ≈ −8 (its
  target) and α climbs from 0.07–0.10 to 0.13–0.65. §19 cause 1's signature (H stays
  ≈ 5) is contradicted in 15/15 fixed runs.
- *The critic overestimates everywhere.* At 300k, Q is 1.2–8.3× the SUPPORTED bound.
  Fixed runs sit at 1.7–8.3× and auto runs are tighter at 1.2–2.2×. Controls that
  never eat climb as high as food runs, so the overestimate is generic. It is not
  shown to be the loss's mechanism.

## J17 — the eye under brain B: `row128` vs 16 rays (rule FROZEN `98e22036`; harvested 2026-11-21 15:18 UTC; 20 of 20 runs, rc 0)

Raw: `world_card_data/j17/j17_r{0..4}.json` (sha256 d3ced9aa…, 122f0ed3…, c99bfc7b…, 439e9aaf…, 77a0e0cb…); kernels `jack-cpu-1795246505/246570/246635/253059/253082`; scored by `pilot_b2 j17-table` (planted tests `test_j17_reading` 10/10). Seeds 921–925, 300k decisions, `ent_coef="auto"`, PILOT_RULE world, body frame.

**Verdict by the frozen rule: NEITHER EATS.** Primary clause: row128 over rays16 U 22/25, 4/5 pairs (met), row128 over blind U 0/25, 0/5 (not met) → **width, not sight** — the gain over 16 rays is not credited to the eye. No VOID clause fired (blind sustained 0/5; lr0 W=2 0/5).

| arm | reading | sustained W2 | late-half meals/life (read) | last quarter (beside) |
|---|---|---|---|---|
| K-rays16 | NOT EATING | 0/5 | 0.19, 0.15, 0.13, 0.28, 0.34 | 0.08, 0.15, 0.16, 0.29, 0.21 |
| K-row128 | NOT EATING | 0/5 | 0.44, 0.27, 0.35, 0.40, 0.32 | 0.35, 0.36, 0.12, 0.17, 0.25 |
| K-row128-blind (control, must fail) | NOT EATING — failed, as it must | 0/5 | 1.43, 0.87, 2.08, 0.70, 0.98 | 1.52, 1.00, 2.15, 0.86, 1.10 |
| K-lr0-row128 (never trains; floor) | bound (not read) | 0/5 | 1.05, 1.45, 2.58, 1.29, 1.55 | 1.14, 1.95, 2.68, 1.29, 1.10 |

**The finding the verdict does not say (reported, not read): training with real sight makes him eat LESS than never training.** Both sighted arms sit 3–8× below the never-trains floor; the blind learner sits at the floor. In the SAC trace, every sighted run (10/10) shows the critic running away and α climbing, every blind/lr0 run (10/10) does not:

| arm | critic loss mid → end (5 seeds) | auto α mid → end |
|---|---|---|
| K-rays16 | 49→304, 24→63, 49→169, 55→120, 68→133 | .16→.30, .09→.08, .15→.28, .13→.26, .21→.25 |
| K-row128 | 160→1,010, 234→844, 82→408, 53→154, 71→276 | .18→.51, .22→.46, .16→.31, .12→.20, .14→.22 |
| K-row128-blind | 13→17, 19→10, 2→9, 16→48, 13→13 | .017→.013, .05→.03, .013→.015, .07→.04, .04→.02 |
| K-lr0-row128 | 16→14, 16→17, 13→62, 12→14, 15→16 | 1.0 (fixed) |

So the wall is not (only) the eye's information: content in the eye input destabilises B's critic (the wider and richer the content, the worse: row128 > rays16). Trace `std` is SB3's `train/std` (the overwrite fixed in `974f18e6` landed after these kernels left), not the action std — not read.

### J17 addendum (2026-11-21 19:50 UTC, JACK slot): the critic over-values, the entropy does not — DESCRIPTIVE, no reading, no verdict changed
Made with the existing `pilot_b2.e2c_q_row` / `e2c_h_row` (J15's tools, unchanged) over the 20 harvested `j17_r*.json`; no run.
SUPPORTED = 100·r/(1−γ) + 100·d_max + α_max·8ln2/(1−γ), a loose ceiling on the soft value a policy eating at its last-50k rate can have.

| arm | Q 300k ÷ SUPPORTED (5 seeds) | Q 50k → 300k | α 50k → 300k | H 300k (target −8) |
|---|---|---|---|---|
| K-rays16 | 1.8 1.2 1.8 1.9 2.2 (5/5 > 1) | 141–336 → 473–983, rising | .04–.11 → .08–.30 | −7.9 … −8.3 |
| K-row128 | 1.7 2.0 1.7 1.4 1.6 (5/5 > 1) | 184–360 → 559–1,360, rising | .05–.10 → .20–.51 | −7.7 … −8.1 |
| K-row128-blind | 0.4 0.6 0.4 0.8 0.6 (0/5 > 1) | 178–321 → 145–347, flat/falling | .04–.10 → .01–.04 | −7.0 … −8.8 |
| K-lr0-row128 | 0.0 (never trains) | 1–3 | 1.0 fixed | – |

What it says: (1) the entropy hunt is closed for J17 — auto α holds H on its −8 target in all 15 learning runs, sighted and blind alike (§19 cause 1, "the entropy bonus dominates", does not fit). (2) Every sighted run's min-twin critic ends ABOVE the ceiling its own eating can support and keeps climbing while its eating falls; no blind run does. (3) α rises only where Q rises — consistent with α following the critic's scale, not driving it (an ordering read off 5k windows, not a test). So the cause hunt's first suspect is the critic on eye content (input normalisation of the eye / critic lr / target τ), not reward scale + α. One seed, box-minutes, before any 300k row — still the Director's to name.

### J17 addendum 2 (2026-11-21 22:30 UTC, JACK slot): WHEN the critic runs away — sizing the cause hunt, DESCRIPTIVE, no run, no verdict changed
`python -m experiments.j17_onset` over the same 20 harvested runs: per seed, the first 5k window from which the sighted arm's signature stays above 1.5× its own blind twin's for every later window (critic loss and α = median of the last 3 windows; margin and median are the tool's choices, not registered). Control: the same rule on blind(seed+1) vs blind(seed) — no content on either side.

| signature | K-row128 (s921–925) | K-rays16 | CONTROL blind vs blind |
|---|---|---|---|
| critic loss | 25k 25k 10k 140k 70k | 40k 295k 80k 150k 45k | 265k never 200k never never |
| Q (min twin) | 115k 70k 75k 295k 155k | 80k 235k 100k 180k 70k | never never 15k never never |
| auto α | 45k 65k 60k 160k 105k | 45k 120k 75k 150k 45k | 45k never 20k never never |

What it says: critic loss is the only signature whose control stays quiet until ≥ 200k (Q and α each fire on a blind-vs-blind pair at 15–45k, so they cannot time an onset). On row128 it separates at 10–25k on 3 of 5 seeds, 70k and 140k on the other two. So a one-seed hunt read at ~30k decisions sees the runaway on ~3 seeds in 5, and only ≥ 140k covers all five; J17's Kaggle rate is 13.7–17.5 decisions per CPU-s → 30k ≈ 30–36 CPU-min an arm, 140k ≈ 2.2–2.8 CPU-h an arm (box rate for row128 not measured here). "Box-minutes on 1 seed" is a pilot that can miss on 2 seeds in 5; seeds 923, 921, 922 are the early ones. Each arm needs its blind twin beside it.

### J17 hunt (ruling 77; PILOT — one seed 923, 100k decisions, each arm beside its own blind twin on its own machine; no verdict on eating)

Reader `j17_onset hunt`: Q climb = Q(100k) ÷ Q(50k), Q = −actor_loss; cut points are the MLP groups' own edges (sighted 1.32–1.57, blind 0.92–1.20), not registered thresholds. CURES (ORDERS J17 branch) = the reader does not print RUNS AWAY **and** sighted late meals/life ≥ its blind twin's (late = lives starting ≥ 50k, `pilot_b2.j17_late_meals`).

| arm | the one change (§24) | machine | sighted Q climb | twin Q climb | reader | late meals/life, sighted / twin | cures? |
|---|---|---|---|---|---|---|---|
| h0 | SimBa policy + AdamW | Kaggle `jack-cpu-1795311660` | — | — | **VOID** (cut at 8.5 h: 34,365 / 35,510 of 100k) | — (no life began ≥ 50k) | **not read** |
| h1 | AdamW, wd 1e-4 (Lyle 2024) | box | 1.375 | 1.20 | **STILL RUNS AWAY** | 1.29 / 0.67 | no |
| h2 | critic features on the unit sphere (OFN, Hussing 2024) | box | 1.442 | 1.509 | **NO CONTROL** | 0.56 / 0.93 | **no** (sighted eats less than its twin) |
| h3 | critic first layer split by sense (Lee 2018 / DreamerV3) | Kaggle `jack-cpu-1795331300` | 1.512 | 1.262 | **STILL RUNS AWAY** | 0.18 / 4.33 | **no** |

h2 beside, not read (`world_card_data/j17/j17_hunt_h2.json`): OFN bounds the critic as built — Q at 100k is 42.5 sighted / 60.7 blind against h1's 367.7 / 311.9, critic loss 0.76 / 0.84 (h1 33.6 / 63.3), α 0.014 both. So the RATIO the reader uses climbs on both arms from a small base (twin 1.509 ≥ 1.32 → the reader's own NO CONTROL clause), and the sighted critic is no longer ABOVE its twin's — but the sighted learner still eats about half of what its blind twin eats. On this one seed, holding the critic's scale did not give him sight: the runaway and the not-eating came apart. (h1's late meals/life is re-read here with the project's own function, 1.29 / 0.67; the 06:20 journal line said 1.19 / 0.71 from a hand count that kept the unfinished last life.)

h3 beside, not read (`world_card_data/j17/j17_hunt_h3.json`, home 08:05, rc 0 both, ~86 CPU-min a cell): giving the eye its own first layer did not stop the climb — sighted 1.512 sits inside the MLP sighted group (1.32–1.57), its twin 1.262 just above the MLP blind group's top (1.20). Q at 100k 298.6 / 296.4, critic loss 13.5 / 38.8 (reader's 100k figures). The sighted arm ate almost nothing late: 3 meals in 17 late lives, every one ~2,500–3,600 decisions (starved). Its blind twin ate 4.33 a life over 9 late lives (one 13-meal life of 9,000 decisions) — the highest blind late figure on this card (MLP blind .70–2.08); one seed, so a draw, not a finding about split critics.

h0 beside, VOID (`world_card_data/j17/j17_hunt_h0.json` + `h0_cut/*.partial.json`; home 10:11, read 11:40). Our own child timeout (`pilot_b.CHILD_TIMEOUT_S` = 8.5 h) killed both cells (rc −9) at 34,365 sighted and 35,510 blind decisions. That is **1.12 decisions a second** per cell on the same two-cell Kaggle CPU kernel where h3's MLP pair did 100k in 57 min (~29 a second), so SimBa as built (critic 2 residual blocks × 512, actor 1 × 128, AdamW) is **~26× slower** here. 100k would take ~25 h, past both the 8.5 h cut and Kaggle's 12 h cap, so **h0 cannot be read on a Kaggle CPU session at all**. Two faults in the apparatus: (1) nobody priced the arm before the launch (the 02:45 reader was written first, but no timing); (2) the trace (Q, critic loss, α) lives only in memory and the partial file keeps lives only, so a cut hunt cell has no Q at any mark. The kept lives (10 each, all starved, meals 8 sighted / 15 blind, all before 36k) are inside the early stretch where J17's arms do not separate yet: they say nothing about the cure. Fixing it means a GPU kernel (JAX on a T4, which is the hardware SimBa's published runs used) or a cut that saves the trace. W47's Kaggle GPU hours are spent (33.1 h).

### J17 note (2026-11-22 09:35 UTC, JACK slot): what the blind arm was — TEXT, no run, no verdict changed
In plain words: J17's must-fail arm `K-row128-blind` did not see noise or a scrambled picture. It saw **nothing**. `experiments/blind_eye.py` sets every one of the eye's 256 numbers to exactly 0.0 on every step. Arm K's input normaliser (`VecNormalize(norm_obs=True)`, `experiments/pilot_b.py:460`) turns a number that is always zero into zero. So those 256 inputs reach the critic and the actor as constant zeros. A dense layer's weight on an input that is always zero gets a zero gradient, so it never moves. The blind arm therefore had the eye's width **in shape only**: inside the network it was the same learner with no eye at all. The sighted arms, by contrast, fed 256 (row128) or 32 (rays16) inputs that the normaliser scales to about unit variance each. That is the "dominant block of normalised inputs" the §24 prior names. So the J17 row's clause "row128 < blind → **width, not sight**" compared *256 live inputs* against *no live inputs*. It could not tell **too many live inputs** (width or statistics) apart from **what those inputs say** (content), and no arm in J17 or in the hunt was a width-matched control with content removed. The verdict NEITHER EATS stands as frozen. The word "width" in it means "a wide live input", not "a wide input with nothing in it". J17-2's `K-row128-shuffled` arm (each step's ray values permuted: width and statistics kept, content destroyed; ORDERS J17 (3)) is that missing control. Until it runs, "the critic runs away on eye CONTENT" (addendum 1, ORDERS "Where stage 1 stands") is one reading of the trace, and "on 256 live unit-variance inputs whatever they carry" is the other. Every sighted run on this card is consistent with both.
