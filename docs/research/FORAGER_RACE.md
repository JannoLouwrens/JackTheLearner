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

## PILOT — learners not known-answer-checked, below every published foraging budget

**Step 3 harvest (J7 (a), 11-16).** Every step-3 kernel that was launched has exited, and no more will be launched.
7 of 10 cells came back with 60 lives each (`/data/jack-logs/forager_race/step3/`). Missing: `dreamer-xs` s154 was
never launched, and both `wm-latent+ring` launches died before the push (their logs have no push line). `forager_race3 tally`
against today's tree reads every cell as stale: they all share one stamp, and the only file that has moved since is
`experiments/w0.py` (J9 `cfb49f25`, with a proof that its default path is byte-identical). So the table below is the
frozen `tally_cells` run on the cells under their own stamp. It is not a verdict on today's code.

| arm | seed | floor meals / 60 lives | apples | nu curve Q1→Q4 | rho (p) | L1 L2 L3 |
|---|---|---|---|---|---|---|
| ppo-needs | 153 | 14 | 0 | .011 .021 .037 .005 | +.02 (.45) | no no no |
| ppo-needs | 154 | 13 | 0 | .016 .011 .037 .005 | −.01 (.52) | no no no |
| ppo-lp | 153 | 13 | 0 | .021 .011 .016 .021 | −.01 (.52) | no no no |
| ppo-lp | 154 | 17 | 0 | .011 .021 .021 .037 | +.25 (.025) | yes yes yes |
| dreamer-xs | 153 | 14 | 0 | .021 .027 .011 .016 | −.15 (.87) | no no no |
| random (control) | 153 | **27** | 0 | .032 .048 .037 .027 | −.02 (.55) | reads learning: no |
| random (control) | 154 | **22** | 0 | .037 .016 .043 .021 | −.02 (.55) | reads learning: no |

**Rule's reading: INCOMPLETE** (dreamer-xs and wm-latent+ring have under 45 lives on a seed). The control holds, so
the step is not VOID. On the complete arms: ppo-needs is NOT LEARNING on both seeds. ppo-lp is NOT LEARNING (it passes
on s154 only, and the rule needs both seeds). dreamer-xs s153 is flat. Over 60 lives, every learner ate about half the
floor meals that random ate (13–17 against 22–27), and nothing ate an apple. This is step 2's finding again at 4× the lives,
and it fits J8's lie-still reading (`PUBLISHED_PRIORS.md` §13): the learners eat less than random, not more. No seat moves.
