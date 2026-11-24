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
  encoder, BY DEFAULT; PLAN "Learning to eat" right column; `hindsight_eye.py`,
  rules first `3a8495b6`, `4463800b`). Bet: taste_eye died on its label, so
  make the label at the meal — imprint the texture of the NEAREST ray when he
  eats, walk back 8 s marking that texture. Kill: taste_eye's bars (hit
  < .50, or < .25 over a yoked twin imprinted at random moments / untrained
  / nearest). Prior: hindsight relabelling and track-back labels are
  published at 10^4–10^6 clips (Andrychowicz 2017, Pathak 2017, from
  memory); at ~200 meals: silent. **ALIVE, ENTERED (INBOX proposal).**
  215 meals, 20 judge worlds, 33,402 looks: as a plain MEMORY (a count of
  eaten colours, no net) **.890** — above the simulator-taught net's .829 —
  yoked .000, ahead 20/20; as a ring NET on the same marks **.434, DEAD**
  (yoked .201, untrained .082). The imprint is 74% on food. By colour set
  .97 / .99 / .74 / .61. **Stage 2** (reflex along the peak ray, carried
  table, worlds 1301–1305, 2,400 decisions): **50 meals vs held8 2, yoked
  0 — ALIVE-2**; read it as "reaches a food by ~18 s and camps on it": 10
  meals a world is the 50 s regrow clock, 5/5 worlds. Disclosed: judge worlds
  were read once by taste_eye; told not refitted. Cannot say: a learner uses
  it; the table filled from empty as he lives; a camera eye (W0's colour is
  one clean number per object — there the net form is needed, and it is
  the dead one). Unrun: the net with the table's output as its teacher.
  END-FIT 1, 2, 5. Philosophy 1, 2, 3, 4.

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

- 11-24 · **Fed once, then weaned — a parent's hand names food for the
  newborn** (Curriculum, VACANT; PLAN "His world" right column, PRIORS §3
  "provisioning that fades"; `fed_once.py`). Bet: `live_eye`'s first meal
  waits on babble (decision 176–6,424); the old weaning died because a
  learner learns eating-WHERE-born, and the table learns WHAT — so set him
  beside each food once, then set him down anywhere: first own meal in half
  born's decisions, feeding time charged. Kill: not ahead on ¾ of worlds, or
  a misnamed twin does as well. Prior: Avalon's dial .716 vs .001 at 5×10^7
  steps, Ecological RL 0 → 70% — learners, 10^5+ steps; one provisioned
  meal: silent. **DEAD AT THE PILOT** (dev worlds 1101–1106, 3,000
  decisions; judge worlds 1501–1508 not lived): first own meal median
  **1,038 vs born 1,712, ahead on 1 of 6 (bar 5)**; meals 35 vs 32; deaths
  1 vs 3; misnamed twin median 3,000. **It died on the NAME, not the
  weaning:** only 8 of 15 hand-fed meals had food on the nearest ray (wild:
  74%) — set down beside it he bumps it side-on — and one wrong name of
  equal count wins whenever it is nearer and costs 400 decisions. The
  dial's own nursery (d = 0): first imprint on food 1 of 4, ahead 3 of 6.
  Found for ruling 84's arm: the reflex can wedge — one life steered 2,983
  decisions at a real food it never reached, never "near", so never
  unlearned. Unrun: imprint the texture that LOOMED over the last second,
  not the nearest ray at the bite; a hand that walks him INTO the food.
  END-FIT 2, 5. Philosophy 1, 3, 4, 5.
