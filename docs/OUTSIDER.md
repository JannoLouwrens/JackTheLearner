# OUTSIDER — 2026-11-15

Read cold, plus one 14 s read-only probe (seed 153). Not read: END-FIT,
`needs.py`.

## 1. The one thing most likely wrong: he cannot see food

The team assumes he fails to forage because the learner is wrong, and is racing
seven brains to find one that eats. The seated senses carry no information
about where food is, so no brain can win that race.

- `w0.py:219-221` builds 16 rays with z = 0; `w0.py:257-258` fires them from the
  body origin + 0.05 m, which is 0.70 m above the floor.
- Floor food is 0.06–0.18 m in radius (`playground.py:124`, `:447`), so its top
  is at most 0.36 m. The apple sits at 1.89 m.
- Probe: from **0 of 615** legal spawns does any ray hit `obj0`, `obj1` or the
  apple. Every ray passes over the floor food and under the apple.
- Touch is a force and a contact flag on four body parts (`w0.py:280-295`); it
  does not say what was touched. The first sign of food is energy rising after
  he has already bumped into it.
- The scripted forager's 24–27 meals come from reading `geom_xpos` (`lf_01`):
  they prove the food is there, not that he can find it.

Median spawn-to-food distance is 3.8 m. PROGRESS says a smell sense died in the lab; if so, it
died on top of the four faults below, so that death does not acquit the eyes.

Four faults that would keep a sighted Jack hungry too:

- **Eating at birth pays nothing.** d = (1−e)² (`drives.py:192-197`). A floor
  meal pays 0 at e = 1.0 (clipped), 0.0025 at e = 0.95, 0.074 at e = 0.5. Food
  pays only once he is weak (`drives.py:200`).
- **The reward is a pure difference of d** (`survival.py:290-294`), so every
  life that starves sums to exactly −1 whether he ate 0 meals or 20. Only
  γ = 0.95 (`survival.py:63`, a 4-second horizon) makes eating preferable. Death
  by hunger is charged d_before − D_DEATH, which is about 0.
- **Exploration is white noise redrawn every 0.2 s** (`survival.py:288`). Probe:
  std 0.5 for 50 s ends 1.3 m from the start and burns 2.2× basal; a held full
  drive covers 2.2 m in 3 s at 1.2× basal. Noise is costly and goes nowhere, so
  the first thing learned is stillness (the AWR arms' 563 s against the
  statue's 600 s).
- **A meal is forgotten in 51 seconds.** The replay holds 256 decisions
  (`survival.py:68`), batch 16 (`cores.py:515`), one step per 4 decisions: each
  meal is sampled about 4 times. Advantages are normalised per 64-decision
  segment (`survival.py:129-130`), so a segment with no meal has its critic
  noise stretched to the same size as a segment with one.

Also: `cores.py:362-364` calls the GRU with zero state and zero action on every
call. `dreamer-xs`, `wm-efe` and `wm-latent` have no memory, no action input and
no imagination. Five of the seven arms are one reactive actor-critic with
different side losses; only `wm-dyn` and the ring arm carry anything.

## 2. Is the first job the right job?

Foraging first: yes. A seven-brain race: no. A strong group would first show
the task is learnable at all — one plain learner, food visible, a million
decisions — and race brains only after one eats.
"Learnable within 15 lives" (LF_04_FAIL §b) is a second untested assumption:
from-scratch agents on comparable tasks are given 10⁵–10⁶ steps, and a 12-hour
free kernel is 1.2 M decisions.

## 3. Three experiments

**E1 — Can his senses locate food?** Log 20,000 random-repeat decisions with
the observation and the true direction to the nearest ready food; fit a small
net; test on a held-out seed. Repeat with the ray origin lowered to 0.12 m.
Cost: 2 × 20k decisions, about 25 minutes. Prediction: direction R² near 0
today, above 0.5 with low rays. If so, stop the brain race and fix the eyes. If
today's senses already give R² > 0.3, section 1 is wrong and the race stands.

**E2 — Sighted twin.** The seated arm (ppo-needs, tanh-awr), forager-race rules
unchanged, seeds 153/154, current rays against rays at 0.12 m. Cost: 4 cells of
at most 180k decisions, one kernel, about 2 hours. Prediction: sighted beats
random on late meals but stays under 5 per life. If it clears the bar, sight
was the whole problem; re-run LF.04. If it does not move at all, run E3.

**E3 — Is it learnable in a million?** ppo-needs, sighted, 1 M decisions, a
2×2×2: γ 0.95 / 0.995; white noise / action held 5 decisions (`HOLD_K` already
exists); d as now / d linear in (1−e) so a meal pays ν at any hunger. Cost:
8 cells × 10 h, two kernels. Prediction: only γ 0.995 + held noise + linear d
reaches 5 meals per life, between 300k and 600k decisions; today's corner stays
at 0. A winner fixes the learner settings before any brain race and prices the
15-life rule. If no cell eats, the brain is the problem and the race is right.

## 4. What the field already knows

- **Potential-based shaping** (Ng, Harada, Russell 1999): a reward that is only
  Φ(s) − Φ(s′) ranks policies by where they end, not by what they did.
  Keramati & Gutkin (2014) rely on the discount; a 4 s horizon is too short.
- **Correlated exploration**: Eberhard et al., "Pink Noise Is All You Need"
  (ICLR 2023); Hollenstein et al. (AAAI 2024) for on-policy; gSDE (Raffin 2021).
- **DreamerV3** (Hafner 2023): the RSSM carries state and action between steps,
  returns are scaled by a running percentile range and never per segment, the
  replay holds 10⁶ steps and is reused tens to hundreds of times per step.
  Prioritised replay (Schaul 2016) keeps rare events.
- **Deep homeostatic RL** (Yoshida et al., PNAS Nexus 2024): foraging from
  drive reduction, by agents that can see the food.
- **Biology**: no mammal forages at birth. It is fed, learns what food tastes
  and looks like, then searches. An adult that carries food to him is a teacher
  around him, allowed by rule 2. The weaning pilot fed him at birth, when a meal
  pays 0.

## 5. What they are doing right

Rules fixed before the run, controls that must fail, a statue baseline: these
are why LF.04's failure can be believed. LF_04_FAIL's arithmetic (5 meals or
1 apple) is a bar most groups never write down.
