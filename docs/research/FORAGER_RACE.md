# FORAGER RACE — J1 step 2 (rules: `experiments/forager_race.py` docstring, committed before any cell)

Harvested 2026-11-15 13:40 UTC. All 16 cells are in (7 LC.09 arms + random floor × seeds 153/154 × 15 lives,
Kaggle CPU kernels K0–K3, tanh-AWR policy loss seated). None of the 240 learner lives was censored at LIFE_CAP.
K1–K3 were reattached with `JACK_REATTACH_ACCEPT_MISMATCH=1`. Only the pinned script sha differed,
and the mismatch is recorded in each cell's receipt.

Bar (LF_04_FAIL.md §2, fixed): on BOTH seeds, (b1) late floor meals ≥ 5 or late apples ≥ 1, and (b2) late
energy eaten (nu) strictly above the random floor on the same seed. Late = lives 10–14.

| arm | late floor meals s153 / s154 | late apples | late span s153 / s154 (s) | b1 | b2 | clears | cell wall (s) |
|---|---|---|---|---|---|---|---|
| random (floor) | 0.0 / 0.2 | 0 / 0 | 291.2 / 292.4 | — | — | — | ~1,200 |
| ppo-needs | 0.4 / 0.2 | 0 / 0 | 415.4 / 376.1 | no | yes / yes | **no** | ~7,100 |
| ppo-lp | 0.4 / 0.0 | 0 / 0 | 381.1 / 378.6 | no | yes / no | **no** | ~11,100 |
| dreamer-xs | 0.2 / 0.2 | 0 / 0 | 382.8 / 372.5 | no | yes / no | **no** | ~8,100 |
| wm-efe | 0.0 / 0.0 | 0 / 0 | 384.4 / 395.4 | no | no | **no** | ~2,250 |
| wm-latent | 0.0 / 0.0 | 0 / 0 | 449.0 / 456.9 | no | no | **no** | ~7,150 |
| wm-latent+ring | 0.2 / 0.2 | 0 / 0 | 420.4 / 451.2 | no | yes / no | **no** | ~11,100 |
| wm-dyn | 0.0 / 0.0 | 0 / 0 | 364.8 / 367.0 | no | no | **no** | ~8,100 |

**Verdict: NONE CLEARS.** The best arm eats 0.4 floor meals in a late life, against a bar of 5. No arm ate an apple
in any late life. Every arm outlives random by 73–165 s, but the late lives are no better fed. The longer life
comes from slower energy spending or avoiding hazards, not from foraging. The sight probe (`sight_probe_1115.json`)
rules out blindness as the cause: random rays hit floor food on 19–24% of decisions.

## Step 3 — what the tally picks, and two problems found before any cell

`tally` names top two = **ppo-needs** (mean late nu 0.024) and **ppo-lp** (0.016). Two problems:

1. **A three-way tie.** ppo-lp, dreamer-xs and wm-latent+ring all have mean late floor 0.2 (nu 0.016), and the
   tally chose ppo-lp only by `ARMS` order. The rules declare no tie-break. One must be declared and committed
   before any step-3 cell (options: run all three tied arms, or break the tie on a pre-stated secondary such as
   late span).
2. **4× lives does not fit one Kaggle session.** The cells ran in parallel inside each kernel, so wall time
   per cell is the cost. At 60 lives, ppo-needs needs about 7.9 h, but ppo-lp and wm-latent+ring need about
   12.3 h, which is over Kaggle's 12 h cap. (Lives get longer as learning works, so these figures are lower bounds.)
   Step 3 needs one cell per kernel and a learner-state checkpoint across sessions, or a cheaper arm choice.
   The life count stays at 4×.
