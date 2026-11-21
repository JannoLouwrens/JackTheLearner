# NH-2 — the needs-hidden row, re-registered (DRAFT, text only, no seed)

Drafted 2026-11-21 06:45 UTC by the JACK lane (J14), **before the μ-inside
pilot (`needs_grid_row pilot 2|5`, launched 06:26) has read anything.** The
Director freezes it; until then no cell of it runs. NH-1 (`5464c5b8`) stays
frozen and FAILED as recorded (KNOWN_ANSWERS Row NH-1); this rule does not
re-score it.

## Why a new rule (KNOWN_ANSWERS NH-1 + addendum)

NH-1 reproduced the paper's DIRECTION (GmQ < DQN; 1-step annealing hurts DQN
only) but missed the MAGNITUDE 10–20× in every learner cell, mostly by
starvation (resources never eaten). The paper's text settles Δ = summed
(ours). Its figure `res_map.png` puts each Gaussian's peak ON the walkable
corner; NH-1 put it on the wall cell, so the best walkable cell reads 37 % of
the peak. That is the ONE change this rule makes to the apparatus.

## The one change, and everything else held

| setting | NH-1 | NH-2 |
|---|---|---|
| Gaussian centres μ | wall corners (10,10),(10,0),(0,0),(0,10) | walkable corners (9,9),(9,1),(1,1),(1,9) = `MUS_INSIDE` (figure reading) |
| σ | Σ = I (text) | Σ = I (text) — the figure's slower fall-off (σ ≈ 1.4 by our colour reading) is NOT adopted: a colour reading is weaker than the text |
| stat bounds | none | none (cause (b), unbounded h, is held so one change is tested) |
| dark cell at walkable (1,1) in the figure | absent | absent (the text names no obstacle; with μ inside it would sit ON res 3's peak — unreadable, left out and said so) |
| all else (DQN/GmQ nets, γ 0.5, lr, buffer, batch, ε schedule, 30k-step life, Δ window [15k,30k), seeds 0–9) | as NH-1 | identical |

## Cells, bands, verdict — the NH-1 clauses unchanged

Same six cells × seeds 0–9 (dqn@5000, gmq@5000, dqn@1, gmq@1, hidden@5000,
frozen@5000), ~1.9 T4-h, one Kaggle kernel. Same bands (the figure's IQR
widened by the 0.1 reading error), read on medians:

- CONTROL frozen@5000 median > 5.0, else VOID.
- R1 PASS iff dqn@5000 ∈ [0.65, 1.45] AND gmq@5000 ∈ [0.32, 0.70] AND gmq < dqn.
- R2 PASS iff dqn@1 ≥ 1.8 AND gmq@1 ≤ 0.80.
- R3 (TEST, EVAAA's prediction carried here) MET iff hidden@5000 > 1.45 AND
  hidden > dqn@5000 on ≥ 8/10 seed pairs; read only if R1 PASSES.
- No extension seeds; no band moves after the first cell.

## The gate — fixed now, before the pilot reads

The pilot is 2 seeds of dqn@5000 with μ inside (NOT a row; frozen-μ NH-1
read 17.2 (s2) and 25.0 (s5) on the same seeds).

- **Launch NH-2** iff BOTH pilot seeds read Δ < 5.0 (below a third of their
  frozen-μ value and inside the range where the band is reachable).
- **Do not launch** if either pilot seed reads Δ ≥ 5.0: moving μ does not
  explain the miss; the next candidate is (b), unbounded stats, as its own
  pilot, then its own rule. A pilot is evidence for WHICH row to run, never a
  row.

## OPEN for the Director

1. σ: keep Σ = I (default, the text) or add σ = 1.4 as a second R1 pair
   (+20 runs, ~0.8 T4-h) — the figure reading vs the text.
2. Whether NH-2 runs on this week's hours behind the ear and the A2 fix.

Code: `experiments/needs_grid_row.py` needs a `kernel --mu-inside` switch
(the frozen NH-1 path byte-identical, a planted test as for the pilot) and a
separate output tag so `score` does not mix NH-1 and NH-2 runs. Not built in
this draft.
