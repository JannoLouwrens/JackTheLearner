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

## Rows (registered run: pending — Kaggle CPU)

| row | policy | lifetime s (mean ± sd) | floor meals / life | apples | drinks | cause of death | summed reward / life |
|---|---|---|---|---|---|---|---|
| a | do-nothing | | | | n/a | | |
| b | random, white per decision | | | | n/a | | |
| c | pink noise per life | | | | n/a | | |
| d8 | action held 8 | | | | n/a | | |
| d32 | action held 32 | | | | n/a | | |
| e | privileged NE.01 forager | | | | n/a | | |
| f | hand forager, observation only | | | | n/a | | |
| g | control: (f) with rays shuffled | | | | n/a | | |

Bar for (f), unmoved from `LF_04_FAIL.md`: per seed, mean floor meals per life
≥ 5 and above row b on the same seed, on ≥ 18 of 20 seeds. (g) must miss it.

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
