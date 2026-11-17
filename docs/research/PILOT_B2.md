# PILOT_B2 — pilot 2 at the entry rung (ORDERS J11)

Status 11-17 07:32: options built and proven; floor card running; **no rule, no learner run yet** (ruling 30: the rule is written from the record and this card).

## 1. The two world options (`4978e989`, proof `experiments/tests/test_jack_env_cures.py`)

| option | what it does | off = today | on = the effect | control that fails |
|---|---|---|---|---|
| `move_cost_dial` | movement drain = KAPPA × (d − 0.75)/0.25 | identical, 3/3 seeds | d 0.75: extra drain 3.7e-17; d 0.875: half KAPPA exactly | off: 5.2–6.3e-4 per decision |
| `spawn_clear=1.55` | no birth or rebirth within 1.55 m of a floor food | identical, 3/3 seeds | 225 spawns all ≥ 1.58 m; do-nothing body at d = 0 eats 0 in 72 lives | uncleared: 75/75 inside, 2,545 meals |

**R = 1.55 m, measured learner-free** (`/data/jack-logs/j11/reach_probe.py`, seeds 381–382, 180 do-nothing lives of 250 decisions spawned within 1.6 m of food): the body is not a statue — it tips and slides a median 0.71 m, max 1.07 m. Five lives ate by contact, all at decisions 9–10: 4 of 28 from 0.44 m, 1 of 17 from 0.88 m, 0 of 123 from ≥ 0.98 m. R = 1.07 + torso 0.27 + largest food 0.175, rounded up. Not measured: drift over a full 600 s life (the card below reads it as meals).

## 2. The floor card (design fixed before any cell; `experiments/pilot_b2.py`)

J8's pilot world + `spawn_clear` + `lazy_contact_audio`; arms noop and held-8, each with `move_cost_dial` off (k1) and on (k0); worlds 601–616 × 125 lives = 2,000 lives an arm. Kaggle CPU, kernels c0..c7.

*Results: pending (c0, c1 launched 11-17 07:30). Table by `python -m experiments.pilot_b2 card`.*

## 3. The rule — not written. 4. Runs — none.
