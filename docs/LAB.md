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
  **Why the perfect map feeds ~1/2: HE CANNOT STAND (11-12 16:50 J1, `experiments/body_probe.py`).**
  Every birth classified (goto_true, food ready at birth, 250 dec, s90/s94, 20 births each):
  fed 11/12 of 20 at adhesion 0.5, 15/14 adhesion off; **toppled (upright cos < 0.5) in 19/20, 18/20
  births**, usually by decision 2-5; **every one of the 9+8 misses toppled and never righted** (most
  lay stuck: <5 cm in the last 50 dec); fed births mostly fell within reach of food. Races, each
  through a mujoco shim in the probe (w0.py untouched): push at torso-bottom site 9/13, at the foot
  12/13 (topple delayed 2 -> 6-7 dec, not prevented) - DEAD; drive 0.5x 7/5, 0.25x 2/1 - DEAD, worse.
  **With NO drive at all he topples at ~decision 10 (2 sim-s) in 18/20 births** (adhesion 0.5 and
  off): a capsule on a 9-cm ball foot, COM ~0.37 m up, is a passive inverted pole. Righting from
  lying (`/data/tmp/smell_sense/probe/righting.py`, s90, 3 topples): held arm postures 0/81, random
  4-phase arm sequences 0-1/150, with drive 1/150 - possible, a <1 % needle for an explorer.
  So every learner so far has been scored lying on his side, dragged by the cheat drive.
  **Next (a BODY repair, raced, a new playground option with today's body the default so no
  certificate moves):** a base that stands passively (flat foot / tripod / lower COM) vs today's
  pole; read topple rate + goto_true fed per birth, then re-run the smell oracle v3 on the winner.
  Biology's answer is both a stable base and a learned righting reflex - race, don't assume.
  smell_oracle v3 s93 hung in its wait loop (no line 15:22-16:25, peers 3-6 min); killed pid 2230940.
  **Plinth race (11-12 16:47 J1, `PROBE_BODY=plinth` body_probe = W0.BAL arm C, parked by D9 since 09-01 and
  measured upright 1.000 vs 0.002 then).** goto_true, 20 births, adhesion-0.5 arm vs the pole's same arm: toppled
  0/20 and 1/20 (pole 19/20, 18/20) but **fed 5/20 s90, 10/20 s94 (pole 11/20, 12/20) - standing is solved, feeding
  is worse.** Every plinth miss (25/25) is stuck UPRIGHT against world geometry, none closer than 1.49 m to food:
  partners poolwall 17, ramp 6, stair 5, seesaw 3, ladder 1, fulcrum 1. The lying pole was dragged over/around these.
  So under a standing base the ~1/2 cap is a straight-line instrument in a walled world: a perfect map is not a
  perfect path. Next: (a) a detouring oracle (grid path around walls) to price the plinth's true ceiling; (b) a
  base that stands AND climbs a ramp/stair lip (smaller plinth / tripod) - raced, pole stays the default.
  Logs /data/tmp/smell_sense/plinth/s90.log, s94.log.
  **Detouring oracle (11-12 17:25 J1, `PROBE_ORACLE=path`).** Privileged 8-connected Dijkstra on a 0.1 m grid
  (every geom that can touch a body geom, aabb footprint inflated 0.25 m pole / 0.55 m plinth, pool pit blocked,
  replan every 5 decisions, 0.5 m lookahead), goto_true otherwise, 20 births x s90/s94, adhesion 0.5, E 250:
  **plinth 12/20, 18/20 = 30/40 vs pole 7/20, 16/20 = 23/40** (straight line: plinth 15/40, pole 23/40).
  A first run with a planner bug (ladder, contype 4, wrongly free) read plinth 13+15, pole 12+15: the plinth's
  number held (28-30/40), the pole's swung 12->7 on s90 - every pole miss is a topple at decision ~2, the plinth's
  are 0-1 topples. Plinth misses left (10): pool lip 5 (upright at z 0.64 touching poolwall), ladder 2, ramp 2,
  open floor 1. So with a path the standing base is the better body (+7/40, both seeds), and its remaining cap is
  the pool lip/ladder/ramp. 2 seeds, unpaired births: a pilot, not a claim. Logs /data/tmp/smell_sense/path2/.
  **The pool lip is a seam, not a lip (11-12 J1, `/data/tmp/poollip/geo.py`, seed-90 W0 plinth, mj_forward only).**
  `floor` is a MuJoCo plane (infinite for collision) at z 0; poolwall0-3 are boxes centred z -0.326, half-height 0.326,
  so their TOPS ARE AT z 0.000, coplanar with the floor; pool_floor sits at z -0.672 under the plane. There is no pit:
  the floor plane covers the pool, and the walls are buried boxes whose side faces end exactly at floor level. The
  plinth's 0.70 m box foot (bottom z 0.01) sinks mm into the soft plane and its leading edge can meet a wall's vertical
  face - a ghost-edge contact with a horizontal normal. All 5 plinth pool misses read z_end 0.64 (= standing height on
  floor) touching poolwall*; food obj1 (1.51,-0.73) sits 0.10 m from poolwall2 (y -0.835), and misses ep9/ep19 stall
  at d_min 0.54/0.55 against poolwall1/2. So (i) the planner's blocked 'pool pit' is not a pit, and (ii) the catch is a
  collision artefact of the world, not terrain. NOT yet shown: the contact normal at a stall (next: drive the plinth
  across poolwall2 and log poolwall contact frame normals). Repair to race, World-side: walls raised to a real lip
  vs floor cut (no plane over the pool) - PG specs move, so register before touching playground.py.
  **The pool lip is a DRIVE TRAP, not a snag (11-12 ~18:30 J1, `experiments/pool_lip_contact.py`, seed 90, 40 dec).**
  Driving the plinth straight at each pool wall with W0's own full push: it stalls 0.55-1.27 m in (N/W/diag/obj1
  crossings) while the open-floor control runs +7.20 m. Every pool contact has a VERTICAL normal (foot resting on the
  wall's top face, coplanar with the floor) - no edge catch; and the drive gate is ON 100% of substeps at the stall. The
  cause is `playground.WaterVolume.apply` (playground.py:656): `xfrc_applied[bid][:3] = drag` OVERWRITES the drive that
  `W0.decide` set one line earlier (w0.py:362-364), and `in_pool` inflates the pool footprint by the body's sphere
  radius (plinth 0.227, pole 0.099) with `z - r < 0`, so a body standing on DRY floor beside the wall is "in water":
  drive erased, 131 N buoyancy held on. He cannot leave - pushed back north 20 decisions, still at (2.60,-0.62).
  Counterfactuals (in-probe patch, no certificate moved): water forces OFF -> the 4 crossings run 4.03/5.18/5.12/5.89 m
  straight over the "pool" on the floor plane (no pit, confirmed dynamically); water forces ADDED to the drive -> 3.84/
  5.17/5.20/4.89 m. The pole (W0 as built) is trapped too: ends wet on all 4 crossings, 1.14-1.46 m. Food obj1
  (1.51,-0.73) lies INSIDE the plinth's wet band (y < -0.608): one of W0's three foods is a starvation trap. Also
  latent: W0.decide clears only xfrc[:2], so buoyancy z persists after a body leaves the band (moot while it cannot).
  Repair, World-side, ~18 specs read WaterVolume (PG.2, PS.01, NE.01, LF.01, ...): (a) water forces must ADD to
  xfrc and be cleared each decision - a bug, fixed not raced; (b) pit vs no-pool vs raised lip - raced. Register first.
  **Repair landed, ceiling re-priced (11-12 ~19:30 J1, W0.PL PASS 3f15c7f, body_probe unchanged, s90/s94, 20 births,
  adhesion 0.5, E 250, /data/tmp/smell_sense/path3/).** Oracle fed per birth AFTER vs BEFORE the Water.apply repair:
  pole line 19+19 = **38/40** (was 23/40), pole path 18+20 = **38/40** (23/40), plinth path 16+20 = **36/40** (30/40),
  plinth line 11+13 = 24/40 (15/40; 12 of 16 misses stuck UPRIGHT against geometry). **The ~1/2 'body ceiling' was
  mostly the pool drive trap: a perfect map now feeds ~95 % of births with either body.** The pole still topples in
  37/40 and 38/40 births and feeds anyway, dragged on its side by the drive - standing does not raise the ceiling.
  Consequence: BD.01's premise (plinth > pole by a margin, pilot read in the trapped world) is not what the repaired
  world shows; BD.01 stays registered unchanged and its claim seeds decide - expect it to FAIL honestly. And every
  sense race above (split rays, smell x2 doses) was killed in the trapped world where food obj1 sat in the wet band.
  **Smell dose race re-launched UNCHANGED on the repaired world (19:36, at ad398aa):** dose.sh -> dose2.sh, same arms
  smell / smell_shuffled, s91-93, 52k decisions, -P 2. Declared before reading, same rule as 13:40: smell must beat
  smell_shuffled on second-half meals (k > 26000, dose2_tally.py) on >= 2 of 3 seeds, else DEAD again. Log
  /data/tmp/smell_sense/dose2.log; ~3 h. The 15:40 numbers to compare: smell 1/5/0 vs shuffled 4/17/27.
- 2026-11-12 · **World: shelter law** (J6, `NE.01` `ok_night_shelter` red every seed) ·
  ALIVE, not shipped. The shipped law cuts k_eff 70 % at occ 1, so shelter's night cost is
  NON-monotone: occ 0.8 → 0.029, sealed roof occ 1.0 → **1.000** (T 39.5, worse than the open
  0.498) and by day any occ > ~0.43 at rest crosses 40 C. **Bet:** the roof hides the cold
  night SKY (Swinbank T_sky = 0.0552·T_a^1.5; radiative share h_r/(h_r+h_c) = 0.61), k unchanged,
  no day effect; night air 27.79 C keeps the open night exactly at 0.498. Monotone: occ 1 → 0.110.
  NE.01's own fixture (`experiments/shelter_law_night.py`, s0–2): s0 seals (occ 0.997) — cut 1.000
  FAIL, sky 0.110 PASS; s1/s2 pose at the grid edge x −2.35 (probe 0.667) and the sleeper ends
  at occ 0.007 — 0.498 under both laws: a fixture/pose fault the law cannot fix. Killed by: the
  re-stale cost of a needs.py change outweighing it, or a race arm (graded cone) doing better.
