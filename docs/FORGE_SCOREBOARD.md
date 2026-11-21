# FORGE SCOREBOARD — what the lab's rounds found, and what they cost

Written by `scripts/forge_scoreboard.py`; do not edit by hand. Rungs S and M only decide who climbs. Only rung L, through `experiments/harness`, says anything about Jack.

## The one number

The current baseline's sustained-life rate at the entry rung, beside its floor.

| date | baseline | sustained-life rate | floor | source |
|---|---|---|---|---|
| 2026-11-20 | brain B (SAC), arm K world, fixed entropy bonus | 0 (0 sustained lives in any of 25 learner runs, 300k decisions) | not measured as a rate; the same brain never learning eats 0.82 meals a life, the learners 0.08-0.16 | pilot 2 (J11, b4aa10f3), read by its frozen rule |

## Rounds

### r1 — KEEP on E-food: which published setting keeps brain B eating in the LATE half of a run (the baseline reaches 4/5 and loses it 10/10)?

Baseline `E-food` · outcome `sustained_rate` (requires eating) · cost 0.00 of 270 core-h, 0.00 of 68 kernel-h

Rule frozen; no rung read yet.

| source | in | reached M | reached L | paying (beats plain variation's yield)? |
|---|---|---|---|---|
| published | 6 | 0 | 0 | not read yet |
| control | 2 | 0 | 0 | the yardstick |
