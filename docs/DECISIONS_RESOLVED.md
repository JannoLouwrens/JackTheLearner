# Decisions resolved by bakeoff

Written by experiments/bakeoff.py. Losing arms are recorded on purpose: a decision whose alternatives were discarded cannot be re-opened when the evidence changes, and the alternatives get silently reinvented later.

## TEST — TIE — mid
mid leads good by only 0.38 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (mid, cost 0).

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| mid | 392.872 | 32.88 | pass | 0 |
| good | 389.605 | 77.56 | pass | 50 |

## TEST — TIE — mid2
mid2 leads good by only 0.38 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (mid2, cost 2).

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| mid2 | 392.872 | 32.88 | pass | 2 |
| good | 389.605 | 77.56 | pass | 50 |

## TEST — VOID
mid and good are within 0.38 sigma so the decision falls to cost, but mid declared none. Declare Arm(cost=...) in the units the spec named (params, latency, GPU-hours) and re-run.

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| mid | 392.872 | 32.88 | pass | — |
| good | 389.605 | 77.56 | pass | 50 |

## TEST — TIE — mid2
mid2 leads good by only 0.38 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (mid2, cost 2).

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| mid2 | 392.872 | 32.88 | pass | 2 |
| good | 389.605 | 77.56 | pass | 50 |

## TEST — WINNER — good
good beats low by 21.66 sigma and clears the null by 77.56 sigma.

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| good | 389.605 | 77.56 | pass | 50 |
| low | 247.256 | 21.25 | pass | 9 |

## TEST — VOID
arms below the 3.0-sigma learning gate: weak. An arm that has not demonstrably learned cannot arbitrate the decision.

metric: `m`  ·  null 107.613 ± 3.636

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| good | 389.605 | 77.56 | pass | 50 |
| weak | 103.261 | -1.20 | FAIL | 1 |
