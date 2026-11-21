# NH-2 — the needs-hidden row, re-registered ONCE (FROZEN by ruling 70)

**Status: FROZEN 2026-11-21 by ruling 70 (`docs/DECISIONS_RESOLVED.md`) = the
draft `5e043f6c` as written BEFORE the μ-inside pilot read anything, with ONE
amendment: seeds 10–19 (never 2, 5 or NH-1's 0–9 — the pilot looked at two of
them). This commit is the freeze; no NH-2 cell has run and neither pilot
seed has read at freeze time.** No extension; own output tag (`nh2_grid_*`);
σ = I as the text says (the σ 1.4 pair is not added). The gate below stands
as written and is enforced in code (`needs_grid_row.nh2_gate`: `kernel
--mu-inside` refuses unless both pilot files read Δ < 5.0). NH-1
(`5464c5b8`) stays frozen and FAILED as recorded either way; a second miss
is a miss. Where the text below says "seeds 0–9", read 10–19.

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

## OPEN for the Director — CLOSED by ruling 70 (σ = I; runs this week behind the ear and the A2 fix)

1. σ: keep Σ = I (default, the text) or add σ = 1.4 as a second R1 pair
   (+20 runs, ~0.8 T4-h) — the figure reading vs the text.
2. Whether NH-2 runs on this week's hours behind the ear and the A2 fix.

Code (built 11-21 07:20, with this freeze): `needs_grid_row kernel
--mu-inside` (all six cells, seeds 10–19, tag `nh2_grid_all6_s10_19`),
`gate-nh2`, `score-nh2`. NH-1's kernel body is tested byte-identical by hash;
NH-1's seeds cannot score under NH-2 nor NH-2's under NH-1.
