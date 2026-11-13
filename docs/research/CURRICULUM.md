# CURRICULUM — how a creature is staged, and how we will race it

J5 step (1). Written 2026-11-13 (JACK lane, short slot). Owner, 2026-11-11: the
curriculum is *"as important as jack himself, put as much research into that."*
Cap 15 KB. Citations are to papers I can name with authors and year; fetch
verbatim abstracts through the arXiv `id_list` API before quoting any number
from them (memory: `arxiv-api-returns-verbatim-abstracts`). Nothing below is a
finding about Jack. It is the menu of arms and the measurement plan.

## 0. What we have today, measured

He has no curriculum. He is born into lethal W0 at a uniform pose and scored.
What the lab has measured about that (LAB.md, J1 state, 11-12/11-13):

- The oracle feeds ~95 % of births. The learner eats 1–2 meals in ~22 lives
  (smell dose2, GAMMA .95), so he gets almost no reward to learn from.
- A perfect map needs a median of 41.5 decisions from a uniform birth to the
  first meal. GAMMA .95 discounts a meal to .12 at birth (LAB, 11-12 22:45).
  GAMMA .99 was raced and is DEAD by its declared rule (1 of 3 seeds).
- 88 of 120 meals sat in 5 "jackpot" lives (12–20 meals each). Only 1 of those
  5 was followed by a fed life. He camps on food he found by luck, and the
  approach does not carry over to the next birth.
- Weaning without smell (born near food, then moved further away): born
  ≤ 0.55 m he ate 3 lives of 3; born at 1.07/1.64 m he missed; after uniform
  births resumed, nursed ate 0 = null 0. He was blind to floor food then.
  Smell has since been added, and wean-WITH-smell is running (pid 2432985).

So a curriculum is not decoration here. The current world gives a from-scratch
learner almost no reward to learn from. Staging the world is a candidate repair,
and it has to be RACED against the null, not assumed.

## 1. How children and animals are staged (the oracle)

- **Starting small (Elman 1993, *Cognition* 48).** A recurrent net learned
  grammar only when its memory window or input complexity was grown over
  training. Later replications disputed how general this is (Rohde & Plaut
  1999). Lesson: staging can make a problem learnable, and it can also be
  unnecessary. That is why the null arm is mandatory.
- **Altricial young are protected by their world, not puppeted.** Nursing,
  nests and parental provisioning shrink the distance between the young animal
  and reward. The parent changes the world around the young (food is near, it
  is warm). The young animal's own actions stay its own. This is exactly
  philosophy line 2, and it is the rule for every arm below: **arms stage the
  WORLD and what the adults say, never his actions.**
- **Zone of proximal development (Vygotsky); scaffolding (Wood, Bruner & Ross
  1976).** The tutor takes on the parts the learner cannot yet do and withdraws
  as competence grows. Computationally this means the difficulty tracks
  competence, and the support is withdrawn on a measured signal, not a clock.
- **Weaning is gradual and driven by the young animal.** Provisioning tapers
  while the young animal's own foraging rises. The `wean_pilot` already copies
  this (nurse phase then uniform births). Its failure mode (fed while nursed,
  starving after) is what a scaffolding curriculum has to beat.
- **Play and self-chosen difficulty.** Infants pick tasks of intermediate
  novelty, neither the known nor the incomprehensible (Kidd, Piantadosi & Aslin
  2012, the "Goldilocks effect", *PLoS ONE*). This is the biological case for
  the learning-progress arm.

## 2. What the field has (the machine arms)

- **Curriculum learning (Bengio, Louradour, Collobert & Weston 2009, ICML).**
  Easy-to-hard ordering of training examples, hand-designed. Our
  "hand-staged" arm.
- **Learning progress as the intrinsic signal (Oudeyer, Kaplan & Hafner 2007,
  IAC, *IEEE TEC*; Baranes & Oudeyer 2013, SAGG-RIAC, *RAS*).** Sample tasks or
  goal regions where competence is CHANGING fastest, absolute LP, so a forgotten
  skill draws practice back. Goal babbling: choose goals, not motor commands.
  The Curiosity seat is already held BY ANALYSIS by learning-progress (LT.03).
  A curriculum arm built on LP is consistent with that seat but is a different
  claim: it acts on the WORLD, not on his intrinsic reward.
- **Teacher–student curriculum (Matiisen, Oliver, Cohen & Schulman 2017/2019,
  *IEEE TNNLS*); automated curricula as a bandit (Graves, Bellemare, Menick,
  Munos & Kavukcuoglu 2017, ICML).** A non-stationary bandit picks the next
  task by the slope of the student's score. This is the cheapest
  implementation of "self-chosen by LP" when the task set is discrete
  (birth-distance bins).
- **ALP-GMM (Portelas, Colas, Hofmann & Oudeyer 2019, CoRL).** Absolute LP over
  a continuous task space (here birth distance, food density, night length).
  Survey: Portelas, Colas, Weng, Hofmann & Oudeyer 2020, "Automatic curriculum
  learning for deep RL" (IJCAI).
- **Goal generation at the frontier (Florensa, Held, Geng & Abbeel 2018, GoalGAN,
  ICML).** Train on goals of intermediate difficulty: success between ~0.1 and
  ~0.9. The simplest version for us: keep birth distances where his fed rate is
  inside that band.
- **Reverse curriculum (Florensa, Held, Wulfmeier, Zhang & Abbeel 2017, CoRL).**
  Start near the goal state and move the start outward as success rises. This
  is weaning, formalised. It is the closest published analogue to what LAB
  already tried, and it adds the criterion we lacked: move outward only when
  success at the current radius passes a bar.
- **Unsupervised environment design (Dennis et al. 2020, PAIRED, NeurIPS;
  Jiang, Grefenstette & Rocktäschel 2021, Prioritized Level Replay, ICML;
  Parker-Holder et al. 2022, ACCEL, ICML).** An adversary or a regret-ranked
  replay picks world parameters. PLR is the cheapest: replay the levels with
  the highest value-loss or regret. The "world mutates between deaths" seat
  that CHAMPIONS names as a future seat is this family.
- **Open-ended co-evolution (Wang, Lehman, Clune & Stanley 2019, POET).** Worlds
  and agents evolve together. Too expensive for a free 4-core box at first, so
  it is named and parked.
- **Teacher-proposed goals from a pretrained model (Du et al. 2023, ELLM, ICML;
  Zhang, Lehman, Stanley & Clune 2023, OMNI).** An LLM proposes "interesting"
  next tasks, and LP or a learned model filters them. In our philosophy an LLM
  is an adult of his world, so this is a legal TEACHER arm. Its proposals change
  the world or what is said to him, never his weights or actions. Before any
  LLM score counts it must pass the Judge admission gate (ORDERS "Next up" #4:
  accept a real success, reject the same episode with the deciding event
  removed, and a yes-to-all judge must fail).

## 3. The levers our world actually has

These are the world knobs a curriculum arm may turn. Each must already exist or
be a pure world-parameter change. No arm may reach into the learner.

| lever | where | note |
|---|---|---|
| birth distance to food | W0 births (`wean_pilot` uses it) | the measured bottleneck |
| food density / regrowth | W0 food spawner | a richer world means more meals per life |
| night length / thermal severity | W0 clock, TAU_T | J6 shows thermal is knife-edge; keep fixed in v1 |
| smell plume strength | `smell_sense` | sense calibration, NOT a curriculum lever (it is his body) |
| what the parent says | SmolLM2 parent | the teacher arm's channel |

v1 turns only **birth distance** and **food density**. Every other knob stays at
the W0 default, so the arms differ in exactly one thing: who chooses the
schedule.

## 4. The cook-off (J5 step 3, to be registered next — not yet)

Same core (`wm-latent`, with the ring memory now seated as working memory if the
harness carries it), same seeds, same number of lives and decisions per arm.
Every arm ends in an identical **exam block** of plain uniform-birth W0 lives
with no staging, and only the exam is scored.

Arms:
1. **null**: uniform W0 throughout (today's regime).
2. **hand-staged**: a fixed schedule of birth radius growing over the training
   lives (reverse-curriculum shape, schedule declared in the spec).
3. **self-chosen by LP**: an ALP bandit over birth-radius × food-density bins,
   picking the bin with the highest absolute change in fed rate.
4. **frontier**: GoalGAN-style band. Keep the radius where the fed rate is in
   [0.2, 0.8] and move it outward when above.
5. **teacher-proposed**: SmolLM2-parent proposes the next world setting from a
   text summary of the last lives. Runs only once the Judge admission gate
   exists; until then it is listed and scored-ineligible-by-absence, not dropped.

Control that must FAIL: **shuffled-stage**, the hand-staged arm's exact multiset
of stages in random order. It must not beat hand-staged. If it does, ordering
carries nothing and the "curriculum" is just extra easy lives.

Second control, needed by the jackpot finding: **matched-meal null**, a null arm
given the SAME number of free easy-birth lives as the staged arms, but
interleaved uniformly. This separates "a curriculum" from "more reward".

Metric: `LF.04`'s survival gain, but read on the exam block as **fed-life
fraction plus median span**, NOT a meal count. The 11-13 J1 reading showed meal
counts are dominated by 5 jackpot lives out of ~60. A margin on means of
heavy-tailed counts certifies luck. The pilot (one seed ≠ claim seeds) sets N
lives and the margin from the null's own life-to-life spread. Both are declared
before the claim seeds are read.

Cost (rough, unpriced): the wean pilot ran 52k decisions per arm-seed in ~1.3 h
CPU at ~20 dec/s. 5 arms + 2 controls × 3 seeds ≈ 21 × 1.3 h ≈ 27 CPU-h. That
is over a day of this box's budget, so it is a candidate for the S1 Kaggle CPU
lane (two learners per session). Pilot first on one seed, three arms (null,
hand-staged, shuffled): ~4 CPU-h, detached.

## 5. Order of operations

1. Harvest wean-WITH-smell (J1). If ALIVE, the hand-staged arm inherits its
   schedule. If DEAD, record why in LAB before the cook-off is registered,
   because then the gap is the learner, and staging may not rescue it.
2. Register `CU`-free ids (the `CU.*` namespace is CURIOSITY, see
   `registry_expansion.py:1854`): proposed prefix `CR.` (curriculum race),
   `CR.00` pilot and `CR.01` claim.
3. Pilot `CR.00` detached, declare N and the margin, then run `CR.01`.
4. The Curriculum seat (CHAMPIONS, VACANT, `ARENA: NONE` until `CR.01` exists)
   is decided by `CR.01`'s verdict only.
