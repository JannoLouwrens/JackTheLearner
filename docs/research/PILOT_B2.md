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

## 3. The rule — not written. 4. Runs — none.
