# FORGE SCOREBOARD — what the lab's rounds found, and what they cost

Written by `scripts/forge_scoreboard.py`; do not edit by hand. Rungs S and M only decide who climbs. Only rung L, through `experiments/harness`, says anything about Jack.

## The one number

The current baseline's sustained-life rate at the entry rung, beside its floor.

| date | baseline | sustained-life rate | floor | source |
|---|---|---|---|---|
| 2026-11-20 | brain B (SAC), arm K world, fixed entropy bonus | 0 (0 sustained lives in any of 25 learner runs, 300k decisions) | not measured as a rate; the same brain never learning eats 0.82 meals a life, the learners 0.08-0.16 | pilot 2 (J11, b4aa10f3), read by its frozen rule |

## Rounds

### r1 — KEEP on E-food: which published setting keeps brain B eating in the LATE half of a run (the baseline reaches 4/5 and loses it 10/10)?

Baseline `E-food` · outcome `sustained_rate` (requires eating) · cost 71.16 of 270 core-h, 18.66 of 68 kernel-h

**Rung S** (150,000 decisions × 3 seeds): 8 in, 3 climb. Baseline 0.03509; plain variation (control mean) 0.1008.

| arm | source | mean | tiebreak | runs lost | vs baseline | result |
|---|---|---|---|---|---|---|
| `E-food` | baseline | 0.03509 | 4.219 | 0 | +0 | the thing to beat |
| `E-food-lr0` | never trains | 0 | 1.128 | 0 | -0.03509 | the twin to beat |
| `N3` | published | 0.8519 | 54.84 | 0 | +0.8168 | CLIMBS |
| `K1` | published | 0.8056 | 53.01 | 0 | +0.7705 | CLIMBS |
| `K2` | published | 0.5139 | 35.2 | 0 | +0.4788 | CLIMBS |
| `K3` | published | 0.1898 | 18.12 | 0 | +0.1547 | not in the top third (first below the cut, 0.1898) |
| `C2` | control | 0.1429 | 11.15 | 0 | +0.1078 | not in the top third (cut 0.1898) |
| `K5` | published | 0.1161 | 5.427 | 0 | +0.08105 | not in the top third (cut 0.1898) |
| `C1` | control | 0.05882 | 6.49 | 0 | +0.02374 | not in the top third (cut 0.1898) |
| `K6` | published | 0 | 0.4914 | 0 | -0.03509 | not in the top third (cut 0.1898) |

Mean = the outcome over the rung's seeds; a lost run (crash, cut, missing) counts 0. Tiebreak: late meals per life (mean over the same late lives), used only to order arms equal on the score.

| source | in | reached M | reached L | paying (beats plain variation's yield)? |
|---|---|---|---|---|
| published | 6 | 3 | 0 | yes |
| control | 2 | 0 | 0 | the yardstick |
