# PILOT_B2 — pilot 2 at the entry rung (ORDERS J11)

Status 11-18 00:40: options built and proven; **floor card complete (16/16 worlds, 8,000 lives)**; no rule, no learner run yet (ruling 30: the rule is written from the record and this card).

## 1. The two world options (`4978e989`, proof `experiments/tests/test_jack_env_cures.py`)

| option | what it does | off = today | on = the effect | control that fails |
|---|---|---|---|---|
| `move_cost_dial` | movement drain = KAPPA × (d − 0.75)/0.25 | identical, 3/3 seeds | d 0.75: extra drain 3.7e-17; d 0.875: half KAPPA exactly | off: 5.2–6.3e-4 per decision |
| `spawn_clear=1.55` | no birth or rebirth within 1.55 m of a floor food | identical, 3/3 seeds | 225 spawns all ≥ 1.58 m; do-nothing body at d = 0 eats 0 in 72 lives | uncleared: 75/75 inside, 2,545 meals |

**R = 1.55 m, measured learner-free** (`/data/jack-logs/j11/reach_probe.py`, seeds 381–382, 180 do-nothing lives of 250 decisions spawned within 1.6 m of food): the body is not a statue — it tips and slides a median 0.71 m, max 1.07 m. Five lives ate by contact, all at decisions 9–10: 4 of 28 from 0.44 m, 1 of 17 from 0.88 m, 0 of 123 from ≥ 0.98 m. R = 1.07 + torso 0.27 + largest food 0.175, rounded up. Not measured: drift over a full 600 s life (the card below reads it as meals).

## 2. The floor card (design fixed before any cell; `experiments/pilot_b2.py`)

J8's pilot world + `spawn_clear` + `lazy_contact_audio`; arms noop and held-8, each with `move_cost_dial` off (k1) and on (k0); worlds 601–616 × 125 lives = 2,000 lives an arm. Kaggle CPU, kernels c0..c7.

**Results, 11-18 00:35** (kernels c0..c7, all 16 worlds, none cut; receipts in `experiments/gpu_submissions.jsonl`; table by `python -m experiments.pilot_b2 card`, raw `/data/jack-logs/pilot_b2/c*.json`):

| arm | worlds (cut) | lives | sustained | rate (95% upper) | runs W=1/2/3 | life s median (5-95%) | meals/life mean | lives fed | max meals |
|---|---|---|---|---|---|---|---|---|---|
| noop-k1 | 16 (0) | 2000 | 2 | 0.00100 (0.00315) | 2/0/0 | 550 (506-595) | 0.036 | 0.001 | 36 |
| noop-k0 | 16 (0) | 2000 | 2 | 0.00100 (0.00315) | 2/0/0 | 550 (506-595) | 0.036 | 0.001 | 36 |
| held8-k1 | 16 (0) | 2000 | 5 | 0.00250 (0.00526) | 5/0/0 | 575 (434-1122) | 1.849 | 0.614 | 21 |
| held8-k0 | 16 (0) | 2000 | 45 | 0.02250 (0.02885) | 45/2/1 | 725 (516-1541) | 2.462 | 0.675 | 22 |

**Rule inputs** (`pilot_b2 rule-inputs`, the method written 11-17 18:45 before the card was complete): smallest W with ≤ 0.5 predicted floor crossings over the pilot's planned floor lives = **W = 2 for all four arms** (held8-k0, the loosest floor, predicts 0.36 at W = 2, 95% upper 1.15; 0.18 / 0.86 at W = 3).

| arm | card lives | decisions/life | planned lives | pred. crossings W=1/2/3 | 95% upper W=1/2/3 | smallest W <= 0.5 |
|---|---|---|---|---|---|---|
| noop-k1 | 2000 | 2758 | 544 | 0.54/0.00/0.00 | 1.71/0.81/0.81 | 2 |
| noop-k0 | 2000 | 2759 | 544 | 0.54/0.00/0.00 | 1.71/0.81/0.81 | 2 |
| held8-k1 | 2000 | 3215 | 467 | 1.17/0.00/0.00 | 2.45/0.70/0.70 | 2 |
| held8-k0 | 2000 | 4120 | 364 | 8.19/0.36/0.18 | 10.50/1.15/0.86 | 2 |

**What the card says, plainly.** Doing nothing, he starves in 506–595 s every life but two. Flailing at random with movement charged (k1) feeds 61% of lives a little and sustains 0.25%. With movement free at the entry rung (k0) the same flailing sustains 2.25% — the move-cost dial alone lifts the random floor ninefold, so arm K must be judged against the k0 floor, not the k1 one.

**The statue leak (R = 1.55 m), measured on the card.** Every do-nothing meal in 2,000 lives came from ONE spawn ring, 1.577 m from food (the nearest legal grid cell past R): 2 of 65 lives there lived fed (seed 601 life 64, seed 612 life 111; 36 meals each, one food every ~250 decisions = its respawn period). 0 meals in the 1,935 do-nothing lives born ≥ 1.6 m away (280 of them at 1.6–2.2 m). By spawn distance:

| arm | 1.577 m: lives / sustained / meals per life | 1.6–2.2 m | 2.2–4 m | ≥ 4 m |
|---|---|---|---|---|
| noop (k1 = k0) | 65 / 2 / 1.11 | 280 / 0 / 0.00 | 824 / 0 / 0.00 | 831 / 0 / 0.00 |
| held8-k1 | 60 / 0 / 2.70 | 278 / 0 / 2.14 | 824 / 3 / 1.99 | 838 / 2 / 1.55 |
| held8-k0 | 76 / 2 / 2.74 | 271 / 12 / 3.12 | 828 / 16 / 2.51 | 825 / 15 / 2.17 |

The leak is 0.1% of lives and makes **no** W = 2 run (a statue life is followed by a fresh spawn). Its cause (body slide over a 9,000-decision life vs a free-body food rolling to him) is the statue-leak probe's question (`experiments/statue_leak_probe.py`, reading rule frozen 11-17 22:45). **Run 2 (11-18 00:43, `/data/jack-logs/j11/statue_leak_601_64_r2.json`) is VOID by that rule**: the box replay drifts 1–2 decisions from the Kaggle card at lives 51 and 54 — the same two lives as run 1, so the drift is box-vs-Kaggle arithmetic, not `native_substep`; life 64 itself replays exactly (9,000 decisions, 36 meals). *Not a verdict, a lead for the next instrument:* in life 64 the body tips 0.71 m sideways (+y) and lies still from decision 20; the eaten food (obj1) moves 1.9 m in −x in the first 10 decisions — past him to 1.25 m beyond — then settles 0.28 m from his root by the first meal (decision 49), and re-spawns there every ~250 decisions. A food that travels to him is something `spawn_clear` (which clears only the body's birth cell) cannot stop.


**The food-travel lead, chased (11-18 01:10–02:00).** The statue's food in 601/64 (obj1, a 0.13 m ball) was laid at (−3.06, 2.62) — on the 13° ramp. Cause in code: `_place_foods` drops each food at its FLOOR rest height and its clash test exempts every ground geom (floor, ramp, stair0–4), so on 32 of 610 legal cells (5.2%) a food starts buried in the ramp or a stair and is thrown out. `experiments/food_travel_probe.py` (rule frozen `25a58628` before any trace): 320 layouts × 3 foods over worlds 601–608, do-nothing body, 50 decisions. **Verdict by the frozen rule: "still"** — 7/706 foods born ≥ 4 m from him moved ≥ 0.5 m (0.99%, just inside the 1% line; "self-moving" needed ≥ 10%), 0/61 born < 2.2 m moved. Read beside the rule, not in it: all 12 movers at any distance had been laid on the ramp/stair footprints (61 of 960 foods laid there, 20% of them moved, up to 3.65 m); 0 of the 899 foods laid on flat floor moved. The body pushed nothing in 50 decisions. **Repair built, default off:** `jack_env(food_floor_only=True)` lays foods only outside the non-floor ground footprints (`adf457ca`; `test_food_floor_only` PASS — identity 2/2 seeds; on: 0 inside, 0/264 far movers; off control: 25 inside, 5/268 far movers). The floor card above was run WITHOUT it; switching it on is a world change and means re-running the card's noop arm at least.

## 2b. The f-card — the floors re-measured on the repaired world (ruling 39; `food_floor_only=True`, world motor, compass-free floors need no eye)

**Final, 8 of 8 groups home** (f7 = held8-k1-ff's last 4 worlds, `jack-cpu-1795015395`, home 19:02; it moved held8-k1-ff from 7/1500 to 8/2000). `pilot_b2 card --glob 'f*.json'`, 20:25 UTC 11-18:

| arm | worlds (cut) | lives | sustained | rate (95% upper) | runs W=1/2/3 | life s median (5–95%) | meals/life | lives fed | max meals |
|---|---|---|---|---|---|---|---|---|---|
| noop-k1-ff | 16 (0) | 2000 | 0 | 0.00000 (0.00150) | 0/0/0 | 550 (506–595) | 0.000 | 0.000 | 0 |
| held8-k0-ff | 16 (2) | 1970 | 37 | 0.01878 (0.02471) | 37/0/0 | 735 (514–1483) | 2.466 | 0.688 | 20 |
| held8-k1-ff | 16 (0) | 2000 | 8 | 0.00400 (0.00722) | 8/0/0 | 580 (434–1158) | 1.900 | 0.617 | 21 |

`rule-inputs` (T = 300,000, 5 seeds): predicted floor crossings W=1/2/3 — noop 0/0/0 (upper 0.82 each; smallest W ≤ 0.5 → 1); held8-k0-ff 6.82/0/0 (upper 8.97/0.55/0.55 → W = 2); held8-k1-ff 1.85/0/0 (upper 3.33/0.69/0.69 → W = 2; interim 7/8 read 2.13, upper 0.91). Read: no floor ever strings two sustained lives together, so **W = 2 is where the floors go quiet** on point estimates; on 95% uppers no W reaches ≤ 0.5 for held8 (the upper is set by the zero count, not by W). Cut cells: held8-k0-ff seeds 608 (f1) and 616 (f3), 110 of 125 lives each (in-kernel deadline; 1970/2000). **Motor frame:** these held8 floors pushed along WORLD axes. If the pilot's arms run `motor_frame="body"` (INBOX 07:40 default; gate 2: body eye + world motor 0/20), the matching floors are `held8-k0-ff-bm`/`held8-k1-ff-bm`: m0, m1, m2 LAUNCHED 18:09–18:11 (`jack-cpu-1795025355`, `…5407`, `…5443`; 6 of 16 worlds), m3–m7 to follow. noop is frame-free.

**m-card, interim 6/8 (11-19 01:25; m4 home 01:14 + m0–m3, m5, `pilot_b2 card --glob 'm*.json'`; m6, m7 RUNNING):** held8-k0-ff-bm 8/1500 sustained (0.53%, upper 0.96%), W=1/2/3 runs 8/0/0, meals/life 1.45, life median 599 s; held8-k1-ff-bm 0/1500 (upper 0.20%), meals/life 1.02, life median 494 s. m4's two worlds added no sustained life on either arm. Beside the world-motor f-card at the same count: k0 expects 28.2 at 37/1970 — seeing ≤ 8 has Poisson P = 8e-6; k1 expects 6.0 at 8/2000, P(0) = 0.0025. Twelve of 16 worlds, six independent kernel groups: the body-motor held8 floor is lower than the world-motor one on both arms. Still interim (no W set from it); W=2 quiet on all twelve worlds.

**m-card, interim 5/8 (11-19 01:10; m5 home 00:50 + m0–m3, `pilot_b2 card --glob 'm*.json'`; m4, m6, m7 RUNNING):** held8-k0-ff-bm 8/1250 sustained (0.64%, upper 1.16%), W=1/2/3 runs 8/0/0, meals/life 1.41, life median 601 s; held8-k1-ff-bm 0/1250 (upper 0.24%), meals/life 0.98, life median 495 s. Beside the world-motor f-card at the same count: k0 expects 23.5 at 37/1970 — seeing ≤ 8 has Poisson P = 0.0002; k1 expects 5.0 at 8/2000, P(0) = 0.0067. Ten of 16 worlds, five independent kernel groups: the body-motor held8 floor is lower than the world-motor one on both arms. Still interim (no W set from it); W=2 quiet on all ten worlds.

**m-card, interim 4/8 (11-19 00:15; m3 home 23:59 + m0, m1, m2, `pilot_b2 card --glob 'm*.json'`; m4, m5, m6 RUNNING, m7 LAUNCHED 00:08 `jack-cpu-1795046888`):** held8-k0-ff-bm 6/1000 sustained (0.60%, upper 1.18%), W=1/2/3 runs 6/0/0, meals/life 1.42, life median 606 s; held8-k1-ff-bm 0/1000 (upper 0.30%), meals/life 0.98, life median 496 s. Beside the world-motor f-card at the same count: k0 expects 18.8 at 37/1970 — seeing ≤ 6 has Poisson P = 0.0006; k1 expects 4.0 at 8/2000, P(0) = 0.018. Eight of 16 worlds, four independent kernel groups: the body-motor held8 floor is lower than the world-motor one on both arms. Still interim (no W set from it); W=2 quiet on all eight worlds.

**m-card, interim 3/8 (11-18 22:25; m1 home 21:54 + m0 + m2, `pilot_b2 card --glob 'm*.json'`; m3, m4, m5 RUNNING, m6 launching, m7 to follow):** held8-k0-ff-bm 4/750 sustained (0.53%, upper 1.22%), W=1/2/3 runs 4/0/0, meals/life 1.40, life median 603 s; held8-k1-ff-bm 0/750 (upper 0.40%), meals/life 0.96, life median 497 s. Beside the world-motor f-card at the same count: k0 expects 14.1 at 37/1970 — seeing ≤ 4 has Poisson P = 0.0017; k1 expects 3.0, P(0) = 0.050. Six of 16 worlds, still interim, no W set from it; but the gap has now held across three independent kernel groups: the body-motor held8 floor is LOWER than the world-motor one, so borrowing the f-card for body-motor arms would set the bar too high (a floor they need not actually clear). Still no floor strings two sustained lives (W=2 quiet on all six worlds).

**m-card, interim 2/8 (11-18 21:40; m0 home 21:32 + m2, `pilot_b2 card --glob 'm*.json'`; m1, m3, m4 RUNNING, m5–m7 to launch):** held8-k0-ff-bm 3/500 sustained (0.60%, upper 1.55%), W=1/2/3 runs 3/0/0, meals/life 1.42, life median 605 s; held8-k1-ff-bm 0/500 (upper 0.60%), meals/life 1.01. Beside the world-motor f-card at the same count: k0 expects 9.4 at 37/1970 — seeing ≤ 3 has Poisson P = 0.016; k1 expects 2.0, P(0) = 0.14. Four of 16 worlds: still NOT a reading, no W is set from it; if it holds, the body-motor held8 floor is LOWER than the world-motor one (an easier floor for body-motor arms to beat — the reason floors must match the arms' frame).

**m-card, interim 1/8 (11-18 21:25; m2 home 20:46, `pilot_b2 card --glob 'm*.json'`; m0, m1, m3, m4 RUNNING, m5–m7 to launch):** held8-k0-ff-bm 1/250 sustained (0.40%, upper 1.90%), W=1/2/3 runs 1/0/0, meals/life 1.42; held8-k1-ff-bm 0/250 (upper 1.20%), meals/life 1.12. Beside the world-motor f-card at the same count: k0 expects 4.7 at 37/1970 — seeing ≤ 1 has P = 0.052; k1 expects 1.0, P(0) = 0.37. Two worlds only: NOT a reading, no W is set from it.

## 3. The rule — DRAFT (11-19 02:30), NOT FROZEN. 4. Runs — none.

**Status.** Written while m6/m7 run, so the slot that lands the m-card can freeze it in one commit. It binds nothing until the commit that sets `PILOT_RULE` in `experiments/pilot_b2.py`; `pilot_kernel` refuses until then. Every bar below is fixed NOW, before the final card and before any learner seed; the freezing commit may only (i) fill the two marked numbers from `rule-inputs` by the formula given, and (ii) set the motor frame per the Director's ruling (INBOX 12:26 / 07:40). Anything else changed = a new draft, said so.

| clause | text |
|---|---|
| World | J8's pilot world (`pilot_b.make_world`: d 0.75, dr100-eat, random_needs 1/6, random_layout, life cap 9,000) **except**: `food_floor_only=True` (ruling 39), `spawn_clear` 1.55 m, `lazy_contact_audio=True`, `native_substep=True` (ORDERS; identical world, J6 — found missing from `make_world` 11-19 and wired, `pilot_kernel` refuses without it), `eye_frame="body"` 16 rays (ruling 38/39), `motor_frame` = **[the ruling: body by the 07:40 default]**. Every arm, floors included, runs the same frames. B0 = J8's setup except exactly these. |
| T | 300,000 decisions per run, warm-up counted (as J8). |
| Seeds | 402–406, every arm (seed 401 was stepped once in a smoke). |
| Arms | learners B0, K, P, G, KP (5 each) + K-lr0 (5); floors noop, held8 (5 each): 40 runs, kernels p0–p9 (`pilot_groups`). |
| Named floors | B0, P, G judged against held8 **k1** (movement charged); K, KP against held8 **k0** (movement free at d0); K-lr0 against held8-k0 **as a bound only** — K-lr0 has no card twin (a learner with lr 0; SBX `use_sde` holds one noise draw 32 decisions, held8 holds a uniform draw 8) — said so in the table. Card arm = `held8-k{0,1}-ff-bm` if motor body, `held8-k{0,1}-ff` if world; noop-k1-ff either way (frame-free). |
| W | **[fill]** = the largest, over the named card arms in use, of `rule-inputs`' "smallest W ≤ 0.5" on the matching card (f-card final: noop 1, k0 2, k1 2 → W = 2; m-card 6/8 has 0 W=2 runs on both arms). If any named arm prints "> 3", W = 3 and the floor clause below decides. |
| Mark | primary `harness.outcome_sustained(lives, W, T)` (capped AND intake ≥ basal drain, W in a row in one world); secondary `outcome(lives, 3, T, θ = 5 meals)`. All W = 1/2/3 reported; only W decides. |
| Floor clause | per named card arm, U = the card's Poisson 95% upper bound on W-run crossings over the pilot's 5 floor runs (**[fill]** from `rule-inputs`; f-card at W=2: 0.55 k0 / 0.69 k1). The in-pilot floor arm (noop; held8, which runs k1 — held8-k0 has no in-pilot twin, K-lr0 is its in-pilot bound) must show crossings ≤ floor(U) (= 0 for U < 1); else the comparisons judged against that floor read **VOID — floor moved** (the pilot's world is not the card's) and the cause is found before any reading. |
| Readings, per learner arm (n = runs of 5 crossing) | **SUSTAINED**: n_sustained(W) ≥ 3 of 5 (at the upper rate U ≤ 0.69, P(≥ 3 crossings) ≤ 0.033; a pilot's bar, not the decisive test's). **EATS BUT STARVES**: not sustained, and n_θ(W=3) ≥ 3 of 5 while the in-pilot named floor crosses θ in ≤ 1 of 5 (if that floor crosses θ in ≥ 2 of 5, this reading is VOID for the arm: θ too easy). **SLIDES**: neither, and in ≥ 3 of 5 runs the late quarter is both less fed than the early 5 lives (% fed late < early) and lives unfed at least 0.9 × the in-pilot noop's median life (the lie-still signature: no meals, noop-length lives). **NEAR RANDOM**: none of the three (P's prediction). |
| Cut / crashed run | counted with its lives and "crash" appended (ruling 19; frozen harness) — never crossed after the cut, never dropped, never re-seeded. An arm with ≥ 2 of 5 runs reaching < 150,000 decisions reads **INCOMPLETE** (not a FAIL, not a reading) and is re-run on seeds 407+ under this rule, said so. |
| Our code is broken if | any noop run crosses at W = 1 (card 0/2000 lives); or K-lr0 crosses at W in ≥ 2 of 5 runs (a frozen net beating held8-k0's U); or ≥ 2 runs of any learner arm end rc ≠ 0 / non-finite loss. Then the pilot is **VOID — broken**, no reading is written, and the next slot finds the cause. |
| Reported only | per arm: meals per 1,000 decisions over the last 50k, % lives capped, causes of death, first sustained life's decision, early → late table (`pilot_b2 table`). Reported-only: it picks the SETUP for the decisive test; no seat moves. |

Predicted before any seed (§13): K reads SUSTAINED inside 300k; P stops the slide with meals near random (NEAR RANDOM); G unknown; B0 SLIDES (J8's slide, now in the repaired world with the body-frame eye).

