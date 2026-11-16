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

## J2 item 5 — the entry rung `d0` (rule `8af5f2e1`, registered 2026-11-16 00:2x; d0 and d_sus named 04:40, d=0.5 harvested 05:21, d=0.25 05:39, 1 row owed)

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
| 0 | owed (`…935`, RUNNING) | | | | | | |

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
