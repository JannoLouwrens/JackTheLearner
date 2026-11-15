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

## Rows (registered run: Kaggle CPU; b harvested 11-15 18:31, raw `world_card_data/b.jsonl` sha256 4dc9ef93…; f harvested 19:04, `f.jsonl` sha256 8076f639…; a,g,e running, c launched 19:07, d8,d32 pending a session)

| row | policy | lifetime s (mean ± sd) | floor meals / life | apples | drinks | cause of death | summed reward / life |
|---|---|---|---|---|---|---|---|
| a | do-nothing | | | | n/a | | |
| b | random, white per decision | 301.7 ± 16.4 (278–376) | 0.52 ± 0.76 (max 4; 76/200 lives ≥1; per-seed means 0.1–0.9) | 0 | n/a | energy 200/200 | −1.00 (all) |
| c | pink noise per life | | | | n/a | | |
| d8 | action held 8 | | | | n/a | | |
| d32 | action held 32 | | | | n/a | | |
| e | privileged NE.01 forager | | | | n/a | | |
| f | hand forager, observation only | 742.1 ± 278.9 (6–1800; median 696) | 4.13 ± 4.98 (max 28; 168/200 lives ≥1; 59/200 ≥5; per-seed means 1.6–7.7, median 4.05) | 0 | n/a | energy 189, integrity 10, censored 1 | −1.00 (199 lives); −0.88 (the 1 censored) |
| g | control: (f) with rays shuffled | | | | n/a | | |

Bar for (f), unmoved from `LF_04_FAIL.md`: per seed, mean floor meals per life
≥ 5 and above row b on the same seed, on ≥ 18 of 20 seeds. (g) must miss it.

**Row f against its bar: FAIL — 6 of 20 seeds (needs 18).** `score(f, b)`:
every seed is above random (20/20; pooled 4.13 against 0.52 floor meals per
life, 7.9×), but only seeds 204, 205, 207, 208, 211, 215 reach 5.0 per life.
The bar is not moved and f is not retuned on these seeds. Row b against the
same bar: 0 of 20 (the §8 floor-arm gate holds for b; a is pending).

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

## Answers (owed after the registered rows)

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
   meals, energy held at 1.00, censored. Registered row e: pending (Kaggle).
2. Does dying early pay (summed reward against lifetime)?
3. Can 16 rays tell food from a wall; if not, which sense is missing?
