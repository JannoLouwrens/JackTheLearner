# LAB — the inventor's notebook (≤ 10 KB)

One entry per idea: date · seat · the bet · the number · ALIVE / DEAD / ENTERED.
A dead idea with a number is a result. Do not retry one without a new reason.

## Already known from the ladder (so nobody re-invents them)

- 2026-08-23 · learning core · five cores screened in W0 · only `wm-latent`
  learned to survive (4.65σ vs null, +92 s carried across deaths); `wm-efe` 2.05σ,
  `ppo-lp` 1.20σ, `ppo-needs` 1.06σ, `dreamer-xs` −0.94σ · seat PENDING.
- 2026-09-07 · control path · small plastic head 506, MLP 415, 57M-trunk arms
  344–350, SB3 reference 518 · VOID on the wrong null; being re-run as `D1.1`.
- 2026-08-24 · deliberation · lookahead gains 13 s ± 19 in W0 — the world is too
  shallow for thinking ahead to pay · needs the deeper world first.
- 2026-08-13 · mood → movement · does not reach behaviour (0.23–0.38 vs chance
  0.25) · DEAD as built.

## Ideas

- 2026-11-11 · **Working memory** (VACANT; incumbent GRU buffer FAILs `WM.01`
  on 2 of 3 seeds) · **SEATED — `WM.02` PASS 2026-11-12** (`2103829`) · `RingMemory.py`.
  **Claim run** (seeds 100–104): ring 0.902 ± 0.196, GRU 0.579, null 0.347; mean lead
  +0.323; shuffled 0.127, no-rotation margin −0.047 (both controls failed as required).
  **Bet:** (1) a store whose events ROTATE the held state — K planes,
  input-dependent angles, a gated absolute write, everything learned — the
  head-direction circuit's move. (2) It might beat the GRU because composing
  events is composing rotations and a rotation cannot forget or blow up, where
  a gated additive cell must discover modular update the hard way. (3) Killed
  by: any seed under +0.10 over the no-buffer null, losing to the GRU, or the
  no-rotation twin doing as well.
  **Pilot** (seeds 0–4, `WM.01`'s venue/recipe/bars, 1,031 params vs the GRU's
  4,872): ring **1.000 / 1.000 / 1.000 / 0.871 / 1.000**; GRU 0.546 ± 0.203
  (ledger); null 0.354; shuffled-state control 0.11–0.14; rotation disabled
  0.318 / 0.283. Not a one-trick: on the absolute last-write venue ring 0.998 /
  1.000 / 1.000 (GRU 1.000). Dead variants: angle init ×4 → 0.106 (chance);
  rotation with no write gate → 0.119.
  **Registered** with gates declared first, claim seeds 100–104 (never
  piloted), ~25 CPU-min; run handed to the workers (ORDERS INBOX).
  **Open after a PASS:** it is a 40-step fixture on a cyclic-group task — next
  ring should be a non-commutative register, then ME.8 / LF.02 with the ring
  in place of the GRU. Serves philosophy 3 (a seat raced, biology as oracle)
  and 4 (two controls that must fail).
- 2026-11-12 · **Curriculum — weaning** (no seat yet; ORDERS J5's hand-staged
  arm; null = "dropped into W0") · **DEAD at this dose** · `experiments/wean_pilot.py`.
  **Bet:** (1) the mother (a spawn rule around him, never his actions) bears
  each body within r of a ready floor food; a meal widens r ×1.5, a hungry life
  narrows it; after 8,000 decisions she leaves (uniform spawn, as W0). (2) It
  might beat the null because J1 found 1 meal in 12 lives — nothing in his
  life says food ends hunger, so AWR teaches stillness. (3) Killed by: no more
  meals than BOTH the null and a decoy-nursed twin once births are uniform.
  **Pilot** (seed 90, wm-latent + AWR, 13,000 decisions/arm, 16 min): nursing
  lives ate **1, 1, 1, 0, 0** at birth distances 0.55 / 0.51 / 0.12 / 1.07 /
  1.64 m; null 0 meals, decoy 0. **Weaned window (≈2,950 decisions, uniform
  births): nursed 0 meals, null 0, decoy 0** — kill clause met. Life spans
  unchanged (nursed 356–461 s, null 353–471 s).
  **Why — measured after the kill (seed-90 world, all 584 legal poses): he
  cannot perceive floor food before he touches it.** His 16 rays leave
  horizontally at z = 0.70 m; floor food tops out at ~0.33 m (centre 0.19,
  half-size 0.14). Rays that hit a floor food: **0 of 584 poses**, at every
  distance band (0–0.75 m: 0/17; 0.75–1.5: 0/50; 1.5–3: 0/171; 3–6: 0/346).
  W0's observation has no smell channel, and proprio carries no x, y. So the
  three meals were contact at birth, the first birth past 1 m missed (closest
  0.70 m), and no curriculum, core or policy loss can teach approach to a
  thing no sense reports. `LF.04`'s margin needs foraging; in W0 as built
  foraging is blind bumping.
  **Next (a World repair, not a curriculum):** give food a distance sense that
  he must learn to use — smell (`experiments/odour.py`, Smell seat VACANT,
  `SM.02`) into `W0.observe`, or rays that can look down — raced, then re-run
  this pilot unchanged: script, decoy control and kill clause stand. Not
  ruled out by one seed and 5 nursing lives: a slower staircase, a longer dose.
  **Repair raced (11-12 J1, `experiments/food_sense.py`, w0.py untouched,
  vision stays 32 wide):** eye heights. Perception over 611 seed-90 legal
  poses — level 0/611; `low` (16 rays at 0.15 m) 129/611; `split` (8 level +
  8 low) 76/611; median nearest hit ~2.5 m. AWR pilot s90, 13k decisions:
  meals level 0, low 0, split 3. Control `split_blind` (rays pass through
  food) checked blind 76/76. Seeds 91–93 × {level, split, split_blind}
  running (/data/tmp/food_sense/run3.log); kill rule declared in the module.
  **KILLED 11-12 12:44 by the declared rule (split must out-eat level AND
  split_blind on >= 2 of 3 seeds):** meals per 13k decisions — s91 level 1,
  split 1, split_blind 2; s92 level 2, split 2, split_blind 2. Split wins
  0 of 2, so at most 1 of 3 whatever s93 reads (s93 left to finish, ~12:55,
  for the record). Every meal sat in a life whose birth was <= 3.4 m
  from food (contact/drift, not approach); split's were all in life 0. Seeing
  food on 12-21 % of poses at 13k decisions does not make the AWR learner
  approach it. Not ruled out: a longer dose; a sense with a gradient (smell)
  instead of a sparse ray hit. Next: race smell (`experiments/odour.py`).
  s93 for the record: level 3, split 3, split_blind 3 (sums 6 / 6 / 7).
  **Smell raced (11-12 13:08 J1, `experiments/smell_sense.py`):** O1 static
  field (SM.01's exp(-d/2 m)) from each READY food, bilateral OdourSensor as a
  new 12-wide `smell` modality, per-process patch. Probe s90, 611 poses: food
  reading 0.056-1.29 (never zero), bilateral difference points to the nearest
  floor food on 99.5 % of poses. Control `smell_shuffled` = same field read at
  a random legal pose each decision. s91-93 x 2 arms running
  (/data/tmp/smell_sense/run.log); null = level above (6). Kill rule in module.
  **KILLED 11-12 13:34 by the declared rule:** meals/13k smell 1/5/2 (8),
  smell_shuffled 2/8/0 (10), level 1/2/3 (6) - smell out-eats both on 0 of 3
  seeds; d_mean no lower than shuffled's. A sense that points at food 99.5 %
  of the time went unused at this dose: the learner, not the world, is now
  the bottleneck (~1 meal per 2 lives is too little signal). Dose test
  declared and launched 13:40: same arms, 52k decisions, second-half meals
  vs shuffled (/data/tmp/smell_sense/dose.log, tally dose_tally.py).
  Serves philosophy 3 (the curriculum raced with a control, not assumed) and
  5 (his curriculum matters as much as he does).
  **Can his body even reach food? (11-12 14:40 J1, `experiments/smell_oracle.py`):**
  v1 oracle was an apparatus bug (world-frame force, no turn). v2 s91 (24 births
  x 250 dec): goto_true (perfect map) 0.46, klinotaxis 0.25, nose_blind 0.13,
  nose_yaw 0.08, random 0.08; s92 goto_true 0.54. Probed s90: v2 under-reads
  because only 2 floor foods exist and births during regrowth have no target;
  with food ready at birth goto_true fed 8/12, and 2/12 (6/12 with adhesion off)
  froze lying on the side, friction-saturated or hands jammed. Even a perfect
  forager reaches food in ~2/3 of births: the body is a ceiling below the learner.
  **Dose test harvested (11-12 15:40 J1): DEAD at 4x dose too.** 52k decisions/arm,
  declared metric second-half meals (dose_tally.py): smell 1/5/0 vs smell_shuffled
  4/17/27 (s91/92/93) - smell loses on 3 of 3. Totals 6/32/2 vs 6/27/27; meals
  come in single lucky lives (23 meals in one s92 smell life, 23 in one s93
  shuffled life), lives fed 4/25, 5/20, 2/23 vs 3/24, 10/20, 3/21. No arm's late
  third beats its first. More dose does not make the sense useful while the body
  feeds a perfect map ~1/2 of the time. Oracle v3 (ready-food births) s91/s92:
  goto_true 0.54/0.54 (v2 .46/.54), random .08/.25 - ready food does NOT lift the
  ceiling; the cap is the body (/data/tmp/smell_sense/oracle3.log, s93 pending).
