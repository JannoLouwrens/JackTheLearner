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

- Folded 11-22 (long forms in git at `3b2b7845`): 11-12/13 **weaning** DEAD
  twice (weaned meals 1/9/1 vs null 2/0/3: he learns eating-where-born);
  11-12 **world/body**: rays pass over floor food, eye height DEAD, smell DEAD
  ×3 (PRIORS §14), pool drive trap fixed (`W0.PL`), GAMMA .99 DEAD, 88 of 120
  meals in 5 jackpot lives; **shelter law** NO WINNER (ruling 35);
  **exploration in bouts** DEAD (fed births 4/3/6 of 40: direction, not range);
  11-13 **meal replay** DEAD (margin .0985 < .10); 11-13/14
  **predict-then-push** ×3 DEAD (6/40, .017, 4/40 vs babble 5/40).

- 11-14/15 · **Run and tumble — climb without predicting** (Deliberation;
  `run_tumble.py`, long form `837b4c6f`) · *told* DEAD 8/40 vs babble 6;
  *steer* **ALIVE with the want GIVEN: 14/40 vs babble 3, coin-yoked 2** (first
  arm of nine to beat babble; oracle 38/40); *steerwant* (want learned from 14
  meals) DEAD 7/40 vs shuffled 7. The gap is the WANT, not direction. Unrun: a
  want from the CHANGE in each sense before a meal; the critic's value as the
  want. Not entered (a given want is advice, not his). Philosophy 1, 3, 4.

- Folded 11-24 (long forms in git at `aceba2e7`): 11-15 **ear taught by
  his own prediction of sound** DEAD (+1.8 / +2.3 points, bar +5.0);
  11-16 **eye as a streak** ALIVE → J12 (128 azimuths .900 vs 16 rays .528,
  knee at 128); 11-17 **finding himself, one look** ALIVE → ruling 38 (128
  rays .933 vs 16 rays .795, no compass; the turn-free key VOID on its
  control); 11-18/20 **place key learned from the passing of time, carried
  across death** ALIVE ×2 → ruling 63 (.934 vs turn search .933; carried
  to unseen worlds .294 / .680 / .945 at 15 / 60 / 360 lives; `place_key.py`,
  `place_key_carried.py`); 11-20 **taste teaches the eye** DEAD (.080 vs
  untrained .082 — the label; food wears one of 4 colour sets by world);
  11-21 **one parent's voice, pitch-warped** DEAD (+2.2 points, bar +5.0).

- 11-21 · **The mouth names it, the eye remembers backwards** (Vision
  encoder; `hindsight_eye.py`; long form in git at `a04f095d`) · imprint the
  colour of the NEAREST ray at each meal. As a plain count table **.890**
  of looks point at food (simulator-taught net .829, yoked .000, 20/20
  worlds); as a ring NET on the same marks .434 DEAD. Reflex on the carried
  table: 50 meals vs babble 2, yoked 0. **ALIVE, ENTERED** (ruling 84).
  Unrun: the net with the table as its teacher. Philosophy 1–4.

- 11-22 · **The hindsight eye lit from EMPTY, and unlearning a wrong
  imprint** (Vision encoder, BY DEFAULT; PLAN "Learning to eat" / "His world"
  right columns; `live_eye.py`, rule first `f3654d00`). Bet: a newborn's
  table of one meal cannot out-vote a wrong imprint, so (1) from empty it
  still lights itself after his first babbled meal, and (2) only if a colour
  he stands at unfed for 80 s leaves the table. Kill: not above babble and a
  yoked twin (imprint from a look 200–1,000 decisions earlier) on ¾ of
  worlds and 2× in total. Prior: Rescorla-Wagner 1972 (a table, tens of
  trials — our size; from memory); no forager on record taught WHICH thing is
  food by one look at its own mouth. **ALIVE, ENTERED (INBOX).** 8 fresh
  worlds 1401–1408, meals in the 2,500 decisions after the first meal:
  **live 70 (9/9/9/8/8/9/9/9) vs babble 12, yoked 27; ahead 8/8 and 6/8
  (bar 6).** The ceiling is the 50 s regrow clock (~9). **Unlearning NEEDED:**
  without it 2 of 8 births lock onto a wrong first imprint (1 and 0 meals
  where live eats 8 and 8). First meal by babble: decision 176–6,424 (median
  ~950; 2 worlds only after a death) — the wait is the world's, not the
  eye's. Dev: patience 10 s forgot REAL food 8/8 (eaten food stays in sight
  while it regrows). Disclosed: the yoked twin ate 9 in 2 worlds — random
  imprint + unlearning is itself trial-and-error; the "imprint on food"
  diagnostic undercounts (says 1 of 9 where he ate 8) and is not quoted.
  Cannot say: a learner uses it (J17); a camera eye; carrying the table
  between worlds. Unrun: patience learned from his own regrow waits.
  END-FIT 1, 2, 5. Philosophy 1, 2, 3, 4.

- 11-24 · **Fed once, then weaned** (Curriculum, VACANT; `fed_once.py`;
  long form at `a04f095d`) · a parent's hand names food once. **DEAD AT
  THE PILOT**: first own meal median 1,038 vs born 1,712, ahead on 1 of 6
  dev worlds (bar 5). Died on the NAME: only 8 of 15 hand-fed meals had
  food on the nearest ray. Found: the reflex can wedge (2,983 decisions at
  a food never reached). Philosophy 1, 3, 4, 5.

- 11-25 · **The class is his own: closeness bins fed by his mouth** (Vision
  encoder, BY DEFAULT; PLAN "Learning to eat" / "Eye" right columns;
  `own_bins.py`, rule first `4628b97a`). J24 stopped because his rays carry
  a per-object id, no class. Bet: the record's 16 closeness bins with the
  FOOD class filled by his live table (empty at birth), read by a ruler
  that sees only the bins, eats several times babble and a yoked table and
  at least half of the same bins with the class handed by the simulator.
  Kill: not above babble and yoked on 3/4 of worlds, or under half of
  handed. Prior: bins per class under SAC 27.0 at 3e6, class always handed
  (PRIORS §27); a sense built online from outcome is unpublished (§26); a
  table and tens of trials is our size. **DEAD BY ITS RULE** (ahead of
  yoked on 5 of 7 worlds, bar 6). 8 fresh worlds 1601-1608, 7 ignited,
  meals in 2,500 decisions after the first: **own 65 = handed 65**, 128-ray
  table 55, yoked 17, babble 10. Lost on: one world whose FIRST imprint was
  not food (own 0, handed 9 — babble had no time to find a second meal),
  and one where the yoked twin got lucky (10 vs 9). Found: 16 bins cost
  nothing against 128 rays; suppression changed no meal. Unexplained: own
  18 vs handed 9 in world 1603. Not entered. A new reason to return would
  be a first imprint that cannot be wrong (e.g. touch at the mouth names
  it, not the nearest ray). Philosophy 2, 3, 4.
