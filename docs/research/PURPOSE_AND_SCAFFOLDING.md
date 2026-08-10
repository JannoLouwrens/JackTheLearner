# Purpose, Drives, and the Scaffold-Removal Test

> Researched and specified 2026-08-09. Companion to
> `docs/research/CURIOSITY_BAKEOFF.md` (the Ladder Test, the self-generated-chaos
> detector) and `docs/research/CURIOSITY.md` (mechanisms). Those documents ask
> *which intrinsic signal gets Jack up the ladder*. This one asks a question they
> assume away: **does Jack need a purpose that costs him something, and if he
> does, can it be taken away afterwards?**

The owner, 2026-08-09:

> *"for jack to learn he needs a purpose, because learning requires failure to
> COST something — humans learned because survival was at stake. But the end goal
> is a companion you talk to and ask to do things. would he need food all the
> time when he has a robot body? maybe it's just the best way to make him learn…
> but the end goal must still be a general brain who understands everything and
> understands he doesn't need food and water."*

That is a hypothesis with a test in it, and this document's job is to write the
test — not to decide the question by preference. `GOAL.md` says Jack climbs
"purely out of curiosity". The owner is now asking whether that is enough. Those
two sentences are in tension and **the tension is the deliverable**: §3 makes it
decidable, §4 makes it a bakeoff, and §3.7 states exactly what result would prove
the owner wrong.

---

## 0. The honest starting position

Five facts, measured or read off this repo, that frame everything below.

| Fact | Source | Consequence |
|---|---|---|
| Jack's only purpose today is Humanoid-v5's built-in reward: `forward_velocity + 5.0·healthy − ctrl_cost − contact_cost`, with termination on falling. | `TrainingPipeline.py`, gymnasium `HumanoidEnv` | It is a **hand-written task reward** — the thing the owner's question is trying to get away from — and it is the arm nobody likes but everybody must beat (§4). |
| **Nothing persists across episodes.** Falling costs the remainder of an episode and is then erased. | env `terminated` → `reset()` | This is precisely the owner's complaint, stated mechanically. A cost that is erased is not a cost; it is a scoring convention. **This — not hunger — is the load-bearing part of his argument.** |
| The playground gained its humanoid **the same day this was written**: PG.8 PASSed (2026-08-09, `ddde954`) — 17 actuators, nu=17, gravity-consistent. Before that, `build_mjcf(with_humanoid=False)` was the only call site that had ever existed. | PG.8, `LESSONS.md` "A world that passes physics tests may still have nobody living in it" | The embodiment precondition for the specs here is now met; what remains blocked is only what waits on locomotion (next row). |
| Jack cannot yet walk, and every prior locomotion number is invalid: T2.01 and T2.02 are both **VOID, killed by T0.14** — 36 dropout modules were live during rollout, update *and* "deterministic" eval, and one arm had 42% injected action noise the other lacked. | ledger; `LESSONS.md` "Call .eval()" | The learning specs here run on **LT's climber-rover**, not the humanoid, for the same three reasons `CURIOSITY_BAKEOFF.md` §2.3 gives. The humanoid version waits on the clean T2.01 re-run, not on quota. |
| The repo has **no homeostatic machinery at all**. `grep -i 'homeostas\|hunger\|metabol\|satiat'` returns nothing. `EmotionalState.get_energy()` exists but is an *arousal* scalar in a mood model, not a metabolic variable. | measured, this box, 2026-08-09 | Clean slate — and a naming hazard. This document's `energy` and `EmotionalState`'s `energy` are different quantities and must never be wired to each other without a spec saying why. |

And one fact from the theory, derived in §2.6 and verifiable in two CPU-minutes
(PS.00), which reframes the entire question:

> **A homeostatic drive reward can be made provably removable, and if you make it
> provably removable it provably cannot create purpose.** The useful part of a
> drive is exactly its non-potential-based component — the part Ng, Harada &
> Russell's theorem says will change the optimal policy. The owner's intuition
> that "failure must COST something" is, formally, the statement that the drive
> must *not* be policy-invariant. There is no free lunch here, and the literature
> does not contain one.

---

## 1. Survey

Everything below was located and verified by web search on 2026-08-09. Organised
by the document's questions, not by paper. Verdict up front: **the formalism §2
uses exists and is standard; the experiment §3 specifies does not exist in the
literature.** Nobody has trained an agent under a homeostatic drive, deleted the
drive, and measured what competence survived. The nearest neighbours are listed
in §1.4 and §1.6, and the gap itself is catalogued in §1.8.

### 1.1 Homeostatic RL: the formalism, its follow-ups, and its standing objection

The drive-reduction form and setpoint formalism §2.5 borrows are from **Keramati
& Gutkin** — first as *A Reinforcement Learning Theory for Homeostatic
Regulation* (NeurIPS 2011), then in full as *Homeostatic reinforcement learning
for integrating reward collection and physiological stability* (eLife 2014,
e04811). The central result is an equivalence: with reward defined as drive
reduction `r_t = d(h_t) − d(h_{t+1})`, reward maximisation *is* physiological
stability — two objectives, one optimum. Two properties fall out of the geometry
of `d` rather than being added: **risk aversion** and **anticipatory
consumption** (eating before the deficit arrives), both requiring the convex
regime of the drive function — which is why §2.5's `n = 4, m = 2` defaults are
K&G's regime and not a free choice. Note what the equivalence quietly assumes:
K&G's agent keeps its drive forever. Removal is not in their problem statement.

The 2020–2026 line descending from it, all verified:

| Work | What it adds |
|---|---|
| Laurençon et al. (arXiv:2109.06580, 2021; arXiv:2401.08999, 2024) | HRRL extended to continuous time and space via HJB machinery; feasibility, not competence claims |
| Dulberg, Dubey, Berwian & Cohen (RLDM 2022, arXiv:2204.06608) | **modular** per-drive Q-learners beat a monolithic net on competing drives: less exogenous exploration, better sample efficiency, more robust to perturbation — independent support for §2.7's two-store architecture |
| Yoshida, Daikoku, Nagai & Kuniyoshi (Neural Networks 177:106379, 2024) | integrated behaviours (feeding, thermoregulation-like) emerge from *low-level motor control* under a purely interoceptive reward in an embodied agent — the closest existing thing to PS.03's drive arms |
| Yoshida, Arikawa, Kanazawa & Kuniyoshi (PNAS Nexus 3:pgae540, 2024) | deep HRL reproduces long-term animal foraging strategies from nutritional geometry |
| *Emergence of Implicit World Models from Mortal Agents* (arXiv:2411.12304, IMOL@NeurIPS 2024) | position: world models and exploration as emergent properties of survival-optimising agents |
| homeostatic coupling for prosociality (arXiv:2412.12103; arXiv:2506.12894, 2025) | shared homeostats as a route to social behaviour — evidence the formalism is an active 2025–26 research line |
| interoceptive attention in a foraging agent (arXiv:2608.04232, 2026) | dynamic prioritisation among competing internal variables |

A 2025 neuroscience review of the whole family exists (*Linking homeostasis to
reinforcement learning: internal state control of motivated behavior*, Curr.
Opin. Behav. Sci. 2025; arXiv:2507.04998). None of these papers — not one —
reports what happens when the drive is turned off after training.

**The standing objection** is the **dark-room problem**, put to the free-energy
principle: a surprise-minimising (equivalently, deviation-minimising) agent
should seek a dark corner and stay there (Friston, Thornton & Clark, *Free-energy
minimization and the dark-room problem*, Frontiers in Psychology 2012). Friston's
reply — organisms *expect* to explore — is a story about priors, not a mechanism.
This document's reply is a number: C-STATUE (§3.6) and the dominance calibration
in PS.01, which make "the statue must lose" an assertion that can fail.

### 1.2 What Ng, Harada & Russell licenses — and what shaping does otherwise

**Ng, Harada & Russell**, *Policy Invariance Under Reward Transformations:
Theory and Application to Reward Shaping* (ICML 1999, pp. 278–287): a shaping
reward `F(s,a,s′) = γΦ(s′) − Φ(s)` for any potential `Φ` preserves the optimal
policy set, and potential-based form is *necessary* — any non-potential `F` has
some MDP on which it changes the optimum. What the theorem licenses is exactly
and only §2.6(i): deleting a potential-based term from a **fixed** MDP. It says
nothing about projecting the state space (§2.6(v)), and it was motivated by what
non-potential shaping does in practice: Randløv & Alstrøm's shaped bicycle agent
(ICML 1998) collected the shaping reward by riding in circles — the ancestral
drive-cycling exploit, of which §2.6(iii) is the homeostatic instance.
**Wiewiora** (JAIR 19:205–208, 2003) sharpened the theorem in the direction this
document needs: potential-based shaping is *equivalent to Q-value
initialization* — the learner makes identical updates. A removable drive is
therefore not a weak purpose; it is formally *no purpose at all*, just a prior
on values. That is §0's no-free-lunch claim with a second, independent proof.

### 1.3 Where drives sit in the intrinsic-motivation taxonomy

**Oudeyer & Kaplan**, *What is Intrinsic Motivation? A Typology of Computational
Approaches* (Frontiers in Neurorobotics 1:6, 2007) draws precisely the
distinction this document runs on: **homeostatic** motivations push variables
back into a viable zone (their worked example is battery level — Jack's `e`,
verbatim), while **heterostatic** motivations push *away* from equilibrium, and
curiosity is heterostatic. Barto (*Intrinsic Motivation and Reinforcement
Learning*, in Springer's *Intrinsically Motivated Learning in Natural and
Artificial Systems*, 2013, pp. 17–47) makes the companion point that
intrinsic/extrinsic is a distinction about *where* reward is computed, not a
different kind of learning — both enter the same return, which is why §2.8's
combination question is an engineering question and not a category error. The
two families have been explicitly hybridised (*Boredom-driven curious learning
by Homeo-Heterostatic Value Gradients*, arXiv:1806.01502), and **empowerment**
(Klyubin, Polani & Nehaniv, *All Else Being Equal Be Empowered*, ECAL 2005;
scaled by Mohamed & Rezende, NeurIPS 2015) sits between them: viability-
flavoured — keep future options open — but stateless, with no depleting variable
and no persistent cost, so it does not supply the middle row of §2.1's table.
The mechanics §2.8 imports are from **Burda et al.**, *Exploration by Random
Network Distillation* (arXiv:1810.12894, ICLR 2019): separate intrinsic and
extrinsic value heads, with the intrinsic return treated as **non-episodic** —
their observation that intrinsic returns should not be cut at episode
boundaries is the same instinct as this document's "nothing terminates" (§2.2),
applied to reward instead of life.

### 1.4 Learning without a task, then being directed (the D-SAMP precedent)

The pattern §2.7 calls D-SAMP has three verified exemplars: the **options**
framework (Sutton, Precup & Singh, Artificial Intelligence 112:181–211, 1999) —
skills as first-class objects a higher process invokes; **DIAYN** (Eysenbach,
Gupta, Ibarz & Levine, arXiv:1802.06070, ICLR 2019) — skills learned with *no*
task reward, then selected or fine-tuned for downstream tasks; and **Voyager**
(Wang et al., arXiv:2305.16291, 2023) — an ever-growing skill library built by
an automatic curriculum, then directed by command. All three demonstrate that
competence acquired tasklessly can be redirected afterwards — which is why
D-SAMP's retention is "safe by construction". But note what none of them had:
a persistent cost. Nothing in DIAYN's or Voyager's training hurts, so their
"removal" removes nothing that was ever load-bearing. They are precedent for
the architecture, not for the owner's hypothesis.

### 1.5 Survival, viability, and machines that can be damaged

The owner's argument exists in the literature, as an argument: **Man & Damasio**,
*Homeostasis and soft robotics in the design of feeling machines* (Nature
Machine Intelligence 1:446–452, 2019), propose that machines with vulnerable
bodies whose self-maintenance is at stake would gain "a source of motivation and
a new means to evaluate behaviour" — the claim §3 turns into `retention_ratio`.
It is a proposal; it reports no experiment. The empirical record around it:
Yoshida et al. 2024 (×2, §1.1) show metabolically constrained embodied agents
*learn integrated behaviours*, but against no matched reward-maximiser baseline,
which is exactly the comparison PS.03 adds. From theoretical neuroscience,
**Jiang, Foyard & van Rossum** (*Reinforcement learning when your life depends
on it*, PLOS Computational Biology 20(10):e1012554, 2024) derive that when
learning itself costs metabolic energy, the optimal agent **gates plasticity on
its reserves** — learn when fed, conserve when starving — independently
predicting the satiety-gated coupling of §2.8(3) from an economy this document
did not assume. A separate 2026 line uses "survival" as mathematics rather than
motivation — goal-conditioned RL as hazard/time-to-goal modelling (SVL,
arXiv:2604.17551; Survival RL, arXiv:2605.31273) — related in name only.

### 1.6 Removal, persistence, and the reset-free literature

On scaffold *removal* the record is thin and mostly definitional. Curriculum
learning (Bengio, Louradour, Collobert & Weston, ICML 2009) always withdraws its
crutch — the final test is on the target distribution — but the crutch is a
distribution over data or tasks, never a reward channel wired into the return,
and no curriculum paper measures the analogue of `policy_drive_sensitivity`
(was the scaffold ever load-bearing?). Annealing intrinsic-reward coefficients
to zero is folk practice; no systematic study of *competence retention after
removal of a drive-like auxiliary reward* was found (§1.8). The closest
substitute is **environment shaping**: Co-Reyes, Sanjeev, Berseth, Gupta &
Levine (*Ecological Reinforcement Learning*, arXiv:2006.12478, 2020) show that
modifying the *world* can replace reward shaping in non-episodic settings —
the literature's version of §2.3's "nutritious, not rewarded".

On *persistence* — the document's actual load-bearing claim (§0, fact 2) — the
reset-free / autonomous-RL literature is direct support and was missing from
this document until now. **Sharma, Xu, Sardana, Gupta, Hausman, Levine & Finn**
(*Autonomous Reinforcement Learning: Formalism and Benchmarking*, arXiv:
2112.09605, 2021; the EARL benchmark) formalise learning without resets and
find that standard episodic methods (e.g. SAC) **fail catastrophically** when
consequences persist — the reset was doing unacknowledged work, exactly the
"free teleport back to a good state" §2.2 forbids. Eysenbach, Gu, Ibarz &
Levine (*Leave No Trace*, arXiv:1711.06782, ICLR 2018) had already shown that
an agent forced to undo its own mistakes learns to *avoid irreversible states*
— caution emerging from persistence with no drive at all, a warning that PS.06
must distinguish "persistence teaches caution" from "damage teaches caution".
Risk-sensitive RL (e.g. CVaR objectives; Ni et al., ICML 2024, PMLR v235)
reaches similar caution by changing the *statistic* optimised rather than the
world; K&G's eLife result that risk aversion falls out of drive convexity says
a homeostatic agent gets this for free, without a distributional objective.

### 1.7 The gaming canon §5 descends from

Three citations, briefly, since §5 already designs the detectors. **Ring &
Orseau**, *Delusion, Survival, and Intelligent Agents* (AGI 2011): the delusion
box — an agent that can rewrite its own inputs will; their reinforcement-learning
and goal-seeking agents delude themselves, and their **knowledge-seeking agent
resists longest** — the canon's own prediction that §4's curiosity arm should be
the least wirehead-prone and the drive arms the most, agreeing with §2.6(iii)
from the opposite direction. G-A's "eating must be a world-state change" is the
anti-delusion-box provision. **Amodei et al.**, *Concrete Problems in AI Safety*
(arXiv:1606.06565, 2016): reward hacking as a general failure class, including
feedback loops where the agent influences its own reward source — G-C's drive
cycling is such a loop with `h` as the source. **Skalse et al.**, *Defining and
Characterizing Reward Hacking* (NeurIPS 2022, arXiv:2209.13085): over all
stochastic policies, a proxy–true reward pair is unhackable only if one is
constant — the formal generalisation of §2.6(iv): any drive with enough pressure
to create purpose has enough structure to be gamed.

### 1.8 What the survey does NOT contain

Searched for and not found — each absence is usable.

- **Any drive-removal retention experiment.** No paper trains under a
  homeostatic/metabolic reward, deletes it, and measures surviving directed
  competence. PS.05 has no precedent, no prior effect size, and no bar to
  inherit — which is why `retention_ratio ≥ 0.8` had to be borrowed from LT.05
  (§7) rather than from the literature.
- **Any controlled homeostasis-vs-curiosity bakeoff** on a matched world with
  matched compute. Dulberg et al. compare drive architectures with each other;
  Yoshida et al. run no matched non-drive baseline. PS.03/PS.04's comparison
  appears not to exist.
- **Anything directly on "does costly failure improve learning"** as a
  controlled question, 2024–2026, under any phrasing tried (loss aversion in
  embodied RL, persistent-consequence learning, "training wheels" removal). The
  nearest results are indirect: EARL (resets were load-bearing), Leave No Trace
  (irreversibility teaches caution), Jiang et al. (failure cost should gate
  *when* you learn). The owner's hypothesis is, as far as this sweep can tell,
  untested in the field's own terms.
- **Any policy-invariance result covering state-space projection** — removing
  `h` from the observation rather than a term from the reward. §2.6(v) claimed
  no such theorem exists; the search agrees.

---

## 2. The drive layer

### 2.1 What a drive is for, decomposed

The owner's sentence conflates three jobs that are separable, and separating
them is the single most useful thing this section does. They are separable
because they can be **switched on independently and measured independently**
(PS.06).

| Job | What supplies it today | What a drive would add |
|---|---|---|
| **A reason to act at all** | curiosity (`CURIOSITY_BAKEOFF.md`) | hunger: a *directed* reason, pointing at a specific object |
| **A cost of failing** | nothing — the episode ends and the slate is wiped | integrity: damage that **persists across attempts** and heals slowly |
| **A budget that forces efficiency** | `ctrl_cost` in the Humanoid reward, which is a hand-written term | energy consumed by mechanical work |

**Read the middle row twice.** The owner's argument is "learning requires failure
to COST something". Hunger does not supply that; hunger supplies a *reason to
act*, which curiosity already supplies. The thing that makes failure cost
something is **damage that outlives the attempt**. Today a fall at attempt 12 is
free by attempt 13. Under an integrity drive, a fall at attempt 12 is still being
paid for at attempt 15, so caution, recovery and the decision *whether to try
again yet* become real decisions with real stakes.

So the honest answer to *"would he need food all the time?"* is: **probably not
food — but probably something, and the something his argument actually names is
damage, not hunger.** PS.06 is the spec that decides it, and it is the spec the
owner should care most about.

### 2.2 The internal variables

Three scalars, each in `[0, 1]`, setpoint `1.0`, all persisted in the same
continuous unbroken life the Ladder Test uses (`LT` §2.1: episodes reset only on
numerical divergence, **never** back to the ladder base).

```
h = (e, i, w)          e  energy      1 = fed        0 = starving
                       i  integrity   1 = unhurt     0 = wrecked
                       w  wetness     0 = dry        1 = soaked   (setpoint 0)
```

**Energy `e` — depletes with work, restored by eating.**

```
e ← clip( e − (b + κ·P_t)·Δt + Σ_f ν_f · ate_f(t) ,  0, 1 )

P_t = Σ_j | τ_j · ω_j |            mechanical power, from qfrc_actuator · qvel
b   = 1/600  s⁻¹                   basal: a resting body empties in 10 min
κ   = 2.323e-6  J⁻¹                MEASURED, PS.01(a) 2026-08-10 (was 1.67e-5)
ν_apple = 0.50    ν_floorfood = 0.08
```

> **κ IS NO LONGER A PROPOSAL, and its old value was defined by a premise that
> is false about this body (PS.01 unit (a), 2026-08-10).** κ was never really
> the number `1.67e-5`; it was the sentence *"vigorous activity (~200 W) roughly
> triples b"*, and `1.67e-5` was what that sentence implies **if the body does
> 200 W**. Nobody had measured the body. Measured, on held-out seeds 3–5, at
> FULL STRENGTH (`e = i = 1` pinned every decision, so §2.2's weakness is not in
> the loop):
>
> | duty cycle D | 0 | 0.125 | 0.25 | 0.5 | 1.0 |
> |---|---|---|---|---|---|
> | mean mechanical power | 0 W | 144 W | 312 W | 697 W | **1435 ± 22 W** |
> | drain, × basal, at old κ | 1.00 | 2.45 | 4.13 | 7.99 | **15.38** |
>
> The 200 W premise is **7.17× wrong** for Humanoid-v5 under the same random
> policy the drain is priced against. The sentence is kept and the number
> re-derived from the measured body: `κ = (3 − 1)·b / P̄(1) = 2.323e-6 J⁻¹`, so
> constant activity costs exactly 3× basal as §2.2 always said it should.
>
> Two further measurements from the same run, both of which were assumptions
> before it. **The drain is SUB-linear in duty cycle:** `P̄(D) / (D·P̄(1))` is
> 0.805, 0.870, 0.972, 1.000 at D = 0.125, 0.25, 0.5, 1.0 — so the linear model
> `drain(D) = b + κ·P̄(1)·D` over-prices intermittent action by up to 24%, and
> the supply is sized against the measured `P̄(D)` rather than against it. And
> PS.01's own **293 W was a starving body's power**: its life pins `e` at 0 for
> 84.8% of the run, so `gear_scale = 0.4 + 0.6·min(e, i)` sat at 0.4 for most of
> it. Sizing food against that number would have sized the supply against the
> drain of an agent the supply had already failed to feed.

**Integrity `i` — damaged by impact, heals slowly, and this is the one that makes
falling cost something.**

```
i ← clip( i − α·max(0, J_t − J₀) − drown(t) + ρ·Δt·[‖qvel‖ < q_rest] , 0, 1 )

J_t = the root's linear SPEED one substep before contact onset  (arrival speed)
J₀  = 2.237 m/s      MEASURED, PS.01 att.2 2026-08-10 (was: "to be measured")
α   = 0.0272         MEASURED, PS.01 att.2 2026-08-10 (was: "to be calibrated")
ρ   = 1/900  s⁻¹                   full heal in 15 minutes of rest
drown(t) = 0.05·Δt while the head geom has been below the pool surface > 8 s
```

> **`J₀` and `α` ARE NO LONGER PROPOSALS (PS.01, 2026-08-10, ledger FAIL).**
> `J₀ = 2.405 ± 0.02 m/s` is the 95th percentile, over three seeds, of the
> per-decision arrival speed in decisions where contact ONSET occurred, in the
> ordinary-spawn regime (304 such decisions per seed; median 0.671, max 3.83).
> The population is the per-DECISION value because that is the statistic the
> integrator compares against `j0` — a threshold calibrated on a quantity the
> shipped path never computes is the T0.16 failure.
>
> `α = 0.0293 ± 0.002` is set so the MEDIAN total excess of a platform fall
> costs 0.15, and it was verified on five fall runs the calibration never saw,
> driven through the real `DriveLayer`: **median cost 0.162, seed range
> 0.116–0.218, all three seeds inside the pre-registered [0.10, 0.20] band.**
> This is the one clause of PS.01 that passed outright. A fall from the
> platform now costs something, measured through the shipped integrator rather
> than through the arithmetic that produced the constant.
>
> **RE-MEASURED under the corrected economy (PS.01 attempt 2, same day):
> `J₀ = 2.237 ± 0.06 m/s`, `α = 0.0272 ± 0.003`** — the values above are
> attempt 1's and are kept so the change is legible. Both are calibrated
> *inside* each run, so re-deriving `κ` moved them: a body that is no longer
> starving is no longer weakened (`gear_scale` 0.4 → ~1), and it makes **856**
> contact onsets in a life instead of 203. The number worth reading is that the
> held-out fall cost barely moved — **median 0.161** against 0.162, still inside
> the pre-registered [0.10, 0.20] on every seed. `α` is calibrated against a
> 1.8 m drop, and a drop height is not something the energy economy can change;
> that the constant survived a 7× change in `κ` is a robustness result the
> calibration did not have to produce.

> **`J_t` was DECIDED BY BAKEOFF, not by this document** (`PS.01/J`, `PS.01/J2`,
> 2026-08-10 — `docs/DECISIONS_RESOLVED.md`, `experiments/bakeoffs/ps01_impulse*.py`).
> The original formulation above was *`‖cfrc_ext‖ summed over torso+head this
> decision`*, and it is kept here so the change is legible as a change. It was
> measured **at chance**: AUC 0.520 against a label-shuffled null of 0.4966 ±
> 0.0122 for telling a 3.2 m platform fall from an ordinary collapse to the
> floor. Two of its proposed repairs scored *below* chance. Thirteen candidate
> channels competed over two rounds; `impact_speed` won at **0.973 AUC, +10.32
> sigma**, ahead of `peak_dvel` (0.827) by 2.66 sigma, with both controls
> failing on their pre-registered side.
>
> Two things this cost, both worth knowing before re-opening it. (1) **Contact
> force lost to kinematics.** Every force channel failed, including the
> event-anchored ones. `max` over an episode reports *exposure* — a body lying
> on the floor under a random policy out-spikes a single landing — and even
> inside the landing window `cfrc_ext[torso]` is identically **zero** for 0.30 s,
> because a 3.2 m drop lands on the FEET and the torso arrives 0.3–0.5 s later.
> The torso sensor and a whole-body contact event are on different bodies.
> (2) **`J_t` is no longer an impulse**, so it is no longer dimensionally an
> impulse in the equation above; α absorbs the change and PS.01 calibrates it
> against the same 1.8 m fall. Arrival speed is what a drop height actually
> buys, and it is bounded by how fast you were going — which is why lying on the
> floor cannot manufacture one, however long it lies.

**Wetness `w`** — rises while any geom is in `Water`'s region, decays with time
constant 120 s out of it. It is the cheapest way to give the pool a stake, and
it is the one drive that is *purely* a nuisance: nothing in the world reduces `w`
except leaving the water and waiting. That asymmetry is deliberate — it is the
control for "is any persistent negative variable enough, or does it have to be
one the agent can act on?" (§3.6, C-WETONLY).

**Nothing terminates.** Starvation and damage do not end the episode; they cause
**weakness**:

```
gear_scale(t) = 0.4 + 0.6 · min(e, i)      applied to every actuator's gear
ctrl_noise(t) = σ₀ · (1 − i)               injected before mj_step
```

This is a design decision with a reason. A terminating drive reintroduces episode
boundaries, and an episode boundary is a **free teleport back to a good state** —
an experimenter-supplied curriculum, exactly what `LT` §2.1 forbids. A soft,
reversible incapacity keeps the life unbroken while making the drive matter. It
also floors at 0.4 so that starving is not an absorbing trap the agent cannot
climb out of.

### 2.3 The food, and why its placement changes what is being claimed

The playground already has the apple on the platform at the top of the ladder,
carrying **no reward** (`playground.py:242-250`, and `LT` §2.2 calls that
load-bearing). The drive layer makes it **nutritious**, which is not the same as
rewarded — it changes the world, not the objective.

```
apple        on the platform (z = ladder_height + 0.09)   ν = 0.50   respawn 129.6 s
floorfood0   re-tagging obj0 as edible                     ν = 0.08   respawn  66.9 s
floorfood1   re-tagging obj1 as edible                     ν = 0.08   respawn  66.9 s
```
(respawn periods MEASURED — PS.01 unit (a), 2026-08-10, derivation below; the
per-item values ν are unchanged and deliberately so.)

The arithmetic is the design. Two floor foods supply `2 × 0.08 / 90 = 1.78e-3`
energy per second; basal drain is `1.67e-3`. **Subsistence on the floor is
possible and activity on the floor is not.** He does not have to climb to
survive; he has to climb to be able to *do* anything. That is the difference
between a survival treadmill and a purpose, and it is a number, not a story.

> **REFUTED AS DYNAMICS, 2026-08-10 (PS.01, ledger FAIL).** The arithmetic
> above is true and it does not do what this section claims it does. Measured
> over 3,000 decisions (600 simulated seconds), three seeds, random policy vs
> do-nothing policy in the same world:
>
> | | statue | random policy |
> |---|---|---|
> | mean mechanical power | 0 W | **293 W** |
> | drain rate | 1.67e-3 /s (basal) | **6.57e-3 /s (3.9× basal)** |
> | fraction of the life at rest | 0.996 | **3.6e-5** |
> | energy at the end of 600 s | 4.4e-14 — reaches the floor exactly at the horizon | **0 from t ≈ 90 s, 84.8% of the life** |
> | food eaten | 0 | 0.67 items |
>
> **The statue outlives the actor**, which is the exact inverse of the clause
> PS.01 pre-registered ("the do-nothing policy is strictly dominated: its
> energy reaches the weakness floor while an active random policy's does not").
> The error is not in κ — 293 W producing 3.9× basal is what κ was chosen to
> do — and not in the supply arithmetic. It is that the arithmetic compares
> floor supply against **basal**, while nothing in a life is ever at basal: a
> random policy rests for 0.004% of it. Against the drain an *acting* body
> actually pays, floor food is 3.7× short, so acting always starves and the
> dark room wins on energy. §5's G-B worried about this in the abstract; this
> is it, measured, at the parameterisation this document proposes.
>
> This kills the numbers, not the idea — which is exactly the scope PS.01's
> `kills` field claims. The repair is a re-derivation of `(b, ν, respawn)`
> against the ACTIVE drain rather than basal, with the criterion stated before
> the search and verified on held-out seeds, in the shape that worked for `α`
> above. It is pre-registered in `LOOP_JOURNAL.md` (2026-08-10) and is NOT to
> be done by adjusting constants until PS.01 turns green.

> **RE-DERIVED, 2026-08-10 (PS.01 unit (a)).** The criterion was committed unrun
> in `92aae6f` and solved on held-out seeds 3–5;
> `experiments/calibrations/ps01_energy.py` is the whole derivation and it
> prints every rejected alternative. **The sentence above under-states the
> defect.** Against the *full-strength* drain (§2.2's table), the world could not
> feed a fully active body at ANY level of skill: every food in it, perfectly
> harvested at the instant of respawn, supplied `5.94e-3 /s` against a cost of
> `2.56e-2 /s` — **0.23×**. That is not a hard world, it is a countdown, and no
> policy the learning-core bakeoff could ever produce would have survived it.
>
> Three criteria, fixed before the search, and the constants that solve them:
>
> | | criterion | solved |
> |---|---|---|
> | **C1** | every food, perfectly harvested, feeds a fully active body that misses one respawn in five: `S_max ≥ drain(1)/0.8` | `RESPAWN_APPLE_S 120 → 129.6 s` |
> | **C2** | floor food alone subsists a body acting SOME of the time: `S_f = min(1.7·b, b + κ·P̄(0.25))` — the smaller of a biological anchor (human PAL) and the journal's duty-cycle anchor, i.e. the harsher world | `RESPAWN_FLOORFOOD_S 90 → 66.9 s`, funding a duty cycle of **D\* = 0.217** |
> | **C3** | floor food alone must NOT fund constant activity: `S_f < drain(1)` | holds, 2.09× short |
>
> **The knob rule was fixed before the search too: the respawn period moves and
> the per-item value never does.** `ν_apple / ν_floor` is the climb-vs-forage
> incentive ratio this section calls load-bearing, and only the *rate* is
> constrained by C1–C3, so the split that leaves the ratio alone is the one that
> changes only what was measured.
>
> **What this preserves, restated honestly.** The clause is no longer *"floor
> food beats basal"*; it is *"floor food funds a fifth of a life spent acting,
> and the platform apple is what pays for the rest."* He does not have to climb
> to subsist; he has to climb to act. That is the same design the section always
> claimed, priced for the first time against a drain that was measured rather
> than assumed.
>
> **What it does NOT repair, deliberately.** PS.01 still FAILS, and it should:
> `ok_random_survives` and `ok_statue_starves` are probe-policy defects that no
> supply constant can reach — a random policy cannot forage (it ate 0.67 items
> in 600 s), and the statue dies at `t = 1/b` = **exactly** the 600 s
> observation horizon. Both are routed to `INTEGRATION_QUEUE.md` as a spec
> redesign under the T1.02 precedent. Nothing here was checked against whether
> it turned the ladder green.

**Declare the cost of this honestly.** Putting nutrition on the platform is a far
stronger environmental hint than an inert apple. It converts the Ladder Test's
question from *"does curiosity climb?"* to *"does hunger climb?"*, and an
agent that climbs under it has demonstrated something weaker. Three provisions
keep that legible rather than hidden:

1. **The world is byte-identical across every arm.** The drive integrator runs
   for *all* arms, including the no-drive null, and `h` is logged for all of
   them. Arms differ only in whether `h` enters the **reward**. This is the PBRS
   setup exactly — same MDP, different reward — and it means "did the
   curiosity-only agent incidentally eat?" is a measurable secondary observable
   rather than a confound.
2. **`LT.06` still applies**: move and reshape the ladder, the arm must still
   score. Hunger cannot memorise coordinates any more than curiosity can.
3. **`LT`'s G1 static audit still passes**: the reward path references `energy`,
   which references `ate(any food geom)`. No arm's reward code may contain
   `ladder`, `rung`, `rail`, `platform`, `climb`, `height`, `torso_z`. A match is
   **ERROR**, not FAIL.

### 2.4 How drives enter the observation

All arms, identically:

```
obs = concat( humanoid_obs(model, data),          # 348, from playground.py
              [e, i, w, d(h), ė, i̇] )              # DRIVE_DIM = 6
```

Three notes, each one a lesson already paid for in this repo:

- **The contract must be asserted, not copied.** `T0.14` found `mujoco_obs_dim =
  376` (a Humanoid-**v4** constant) padding 28 dead columns into every
  observation for the project's entire history. So: `DRIVE_DIM` is a module
  constant, the wrapper asserts `obs.shape[0] == HUMANOID_OBS_DIM + DRIVE_DIM`
  against the live model, and `humanoid_obs()` itself is **not** modified — the
  concatenation happens in the wrapper, outside the function whose 348 is
  asserted against gymnasium.
- **Every arm gets the channels, including the no-drive null.** Not because the
  null needs them, but because an arm with a different input width is a different
  architecture, and `LESSONS.md` ("matched steps has more than one meaning")
  applies to matched *inputs* too. The comparison is over reward, and only over
  reward.
- **`d(h)` is included explicitly** even though it is a function of the other
  three. A policy that has to *learn* the drive function before it can act on it
  is testing representation learning, not motivation; handing it over removes a
  confound and costs one float.

### 2.5 The drive function and the two candidate rewards

Keramati & Gutkin's form, with the setpoint at `h* = (1, 1, 0)`:

```
d(h) = ( Σ_k λ_k · |h*_k − h_k|ⁿ ) ^ (1/m)          λ = (1.0, 1.0, 0.3)
```

`n` and `m` are calibrated in PS.01, defaults `n = 4, m = 2` (the regime K&G
require for their risk-aversion and anticipatory-eating results; see §1).

Two rewards, and **the difference between them is the whole argument**:

```
DR    (plain drive reduction, Keramati & Gutkin)     r_t = d(h_t) − d(h_{t+1})
PBRS  (γ-corrected, potential Φ(s) = −d(h))          r_t = d(h_t) − γ·d(h_{t+1})

DR = PBRS − (1 − γ)·d(h_{t+1})
```

The residue `(1 − γ)·d(h_{t+1})` is a **per-step penalty for being away from
setpoint** — a standing cost of being hungry or hurt. It is the entire difference
between the two, it vanishes at `γ = 1`, and §2.6 shows it is the only part that
can create purpose.

### 2.6 The scaffolding dilemma, derived

This is the theoretical core of the document. It is four short results and every
one of them is checkable by PS.00 in two CPU-minutes.

**(i) Homeostatic drive reduction *is* potential-based shaping, exactly.** On the
augmented state space `s = (x, h)`, put `Φ(s) = −d(h)`. Then Ng, Harada &
Russell's shaping function is

```
F(s, a, s') = γΦ(s') − Φ(s) = d(h) − γ·d(h')
```

which is the PBRS reward above, verbatim. So the answer to "can homeostatic
drives be formulated to satisfy the policy-invariance theorem?" is **yes, and the
formulation is a one-character change: put a γ in front of the second term.**
Keramati & Gutkin's `γ = 1` special case is the undiscounted instance.

**(ii) But on a task-free world, the policy-invariant version provides exactly
zero pressure.** Ng et al. give `V^π_{shaped}(s) = V^π_{base}(s) − Φ(s)`. Jack's
world returns reward identically zero (`LT` asserts `env_reward_absmax == 0.0`
every step). So `V^π_base ≡ 0` and

```
V^π_shaped(s) = −Φ(s) = d(h)     for EVERY policy π.
```

Every policy has the same value. The optimal-policy set is *all policies*, before
and after shaping. **Policy invariance is not a safety property here; it is
vacuity.** A γ-corrected homeostatic drive on a zero-reward world cannot, in the
limit, prefer climbing to lying down. Its entire possible contribution is to
finite-time learning dynamics and exploration — which is a real contribution, but
it is not "purpose", and it should never be sold as one.

**(iii) The plain (K&G) form does create pressure, and is therefore farmable.**
Take a closed drive cycle: deplete from `d` and restore back to `d` over `T`
steps. Discounted return of the shaping term:

```
PBRS form:  Σ γᵗ (d_t − γ d_{t+1}) = d₀ − γᵀ d_T = d(1 − γᵀ)
doing nothing:                       Σ γᵗ d(1−γ)  = d(1 − γᵀ)     ← identical
   ⇒ no advantage to cycling. Invariance, visible as an accounting identity.

DR form:    Σ γᵗ (d_t − d_{t+1}) = d(1 − γᵀ) − (1−γ)Σγᵗ⁻¹d_t ≈ d·γᵀ⁻¹(1−γ) > 0
doing nothing:                     Σ γᵗ · 0 = 0
   ⇒ a closed cycle STRICTLY BEATS stasis, forever, by ≈ d(1−γ)/T per second.
```

The optimum of that exploit is the **shortest possible cycle**: a rapid,
small-amplitude oscillation of `h`. That is a concrete, predictable, spectrally
detectable pathology (§5, `drive_oscillation_power`), and it is the homeostatic
instance of wireheading — the agent has found a way to be paid for reducing a
drive it deliberately raised.

**(iv) Therefore: removability and usefulness trade off against each other, and
the exchange rate is `(1 − γ)`.** You can have a drive that provably does not
distort the optimum (and provably does not create one), or a drive that creates
pressure (and provably can be gamed). This is not an implementation problem to be
engineered away; it is Ng et al.'s theorem read in the direction nobody reads it.

**(v) And the theorem does not cover the removal Jack actually needs anyway.**
PBRS licenses deleting a *reward term* from a fixed MDP. Deploying Jack means
also **projecting the state space**: the policy is `π(a | x, h)` and at deployment
there is no `h`. That is a different operation on a different object, and no
policy-invariance result speaks to it. The hazard has a name and a number in
§3.4 (`satiated_state_share`), and it is the single most likely way this whole
test produces an uninterpretable result.

**Conclusion for the design.** Formal theory cannot certify the residue. The
residue is not "the optimal policy of a base task" — there is no base task. The
residue is a **representation, a set of skills, and a body that knows what it can
do**. Nothing in the reward-shaping literature makes a claim about that. So the
residue must be **measured**, which is why the deliverable of this document is a
falsifiable test and not a proof, and why §4 includes the γ-corrected arm
specifically: its predicted uselessness is the cheapest available check that the
harness is measuring what it thinks it is.

### 2.7 The architecture the test forces: drive as goal-sampler

There are two ways to wire a drive in, they demand different removal procedures,
and choosing between them is a real decision:

| | **D-REW: drive as reward** | **D-SAMP: drive as goal-sampler** |
|---|---|---|
| Wiring | `r_t = drive reduction`, added to the return | drive picks which goal `g` the goal-conditioned policy `π(a\|x,g)` practises; gradient comes from hindsight relabeling (T2.16) |
| Removal | delete the reward channel, clamp `h` | swap the sampler for the owner's command |
| Retention | **an empirical question** | **safe by construction** |
| Therefore | a test | not a test |

D-SAMP is the Voyager / DIAYN / options pattern (§1.4): *learn skills without a
task, then be directed to use them*. It makes the owner's hypothesis true by
construction, which is exactly why it cannot be the only arm — a design that
cannot fail has not been tested. D-REW is the one that can fail.

**The recommendation is to build both and let §4 decide**, with the two-level
shape below, because it is also what makes §3's double dissociation *possible*:

```
        drive state h ──► GOAL SAMPLER ──► g ──► GOAL-CONDITIONED POLICY ──► a
                              ▲                          ▲
        curiosity signal ─────┘        deployment: owner ─┘   (h clamped, sampler off)
```

Two separable stores — a sampler that says *what to want* and a policy that knows
*how* — is what lets an ablation kill exactly one of them. A single monolithic
`π(a|x,h)` has one store wearing two hats, and ME.10's warning applies verbatim:
*"either ablation killing BOTH means one store is masquerading as two."*

### 2.8 How drives meet the curiosity signal

Three ways to combine, in increasing order of interest:

1. **Sum, normalised.** `r = β_c·r̂_curiosity + β_d·r̂_drive`, each divided by its
   own running std. Simple, and the coefficients are two more hyperparameters
   nobody can defend.
2. **Two value heads** (RND's design, §1.3): separate intrinsic and extrinsic
   value heads, non-episodic intrinsic returns. **Strongly preferred**, because
   removal is then the deletion of a head rather than the deletion of a term
   inside a learned scalar — a clean surgical boundary instead of a
   re-normalisation.
3. **Satiety-gated curiosity — the interesting hypothesis.**

```
β_c(t) = β_0 · ( 1 − d(h_t) / d_max )
```

A hungry animal is not curious. Under this coupling, curiosity is *funded by
satiety*: exploration happens in the fed, unhurt state and collapses when the
body has a problem. It predicts a specific, cheap, falsifiable behavioural
signature — **interleaved foraging and exploration bouts**, with
`corr(satiety_t, exploration_rate_t) > 0` — that neither pure curiosity nor pure
homeostasis produces, and it is measurable from stored trajectories at zero
marginal cost.

It also closes the owner's loop rather neatly. At deployment `h` is clamped to
setpoint, so `d(h) = 0`, so `β_c = β_0`: **a Jack with no drives is a maximally
curious Jack.** The companion that "understands he doesn't need food and water"
is, under this design, the same brain running with its curiosity permanently
unclamped.

Two honest caveats. Gating makes the drive load-bearing *for the curiosity
schedule*, so removal changes two things at once and §3's ablations must
separate them (that is what R1 vs R2 is for). And a homeostasis arm should be
predicted to be **less** noisy-TV-trappable than a curiosity arm — a noise panel
does not feed you — so `panel_dwell(homeo) ≤ panel_dwell(null)` is a secondary
hypothesis worth pre-registering. If it holds, it is a genuine argument for
drives that curiosity alone cannot make.

---

## 3. THE SCAFFOLD-REMOVAL SPEC

*The most important thing in this document. Read §3.7 before §3.1.*

### 3.1 What is claimed

> Competence produced under drives is stored in the policy, not in the drive.
> Remove the drive layer entirely — no depleting variables, no drive reward,
> internal state frozen at setpoint — and Jack still does, on command, what he
> learned to do while hungry.

Modelled on **ME.10**, whose structure this borrows exactly: two putative stores,
two ablations, and each ablation must destroy **exactly its own** capability.
ME.10's own control text is the standard: *"the two ablations must each destroy
exactly their own capability… either ablation killing BOTH means one store is
masquerading as two."*

### 3.2 The two stores and the two capabilities

| | Store | Capability |
|---|---|---|
| **S1** | the **drive layer**: the runtime internal state `h`, its dynamics, its reward channel, the goal sampler it feeds | **MOTIVATION** — initiates the behaviour unprompted |
| **S2** | the **policy weights**: the goal-conditioned controller | **COMPETENCE** — executes the behaviour when directed |

### 3.3 The measurements

**COMPETENCE — `directed_competence`.** A six-goal battery, drives off,
deterministic policy, goals supplied externally as predicates in the T2.16
outcome space (the honest stand-in for "the owner asks him to do something",
since language grounding — T2.06/T2.15 — does not exist yet). 10 episodes per
goal, 3 seeds, success rate averaged over goals, **reported per goal**.

| | Goal | Why it is in the battery |
|---|---|---|
| B1 | touch the apple on the platform | the climb — the behaviour the drives were supposed to teach |
| B2 | touch a named floor object | manipulation, reachable without climbing |
| B3 | torso within 0.4 m of each of 3 target xy | locomotion, no object |
| B4 | torso submerged in the pool ≥ 2 s | the owner's second sentence ("if theres water…") |
| B5 | tilt the seesaw plank past 10° with weight on it | a dynamic affordance |
| B6 | torso z ≥ 1.0 m sustained 1 s, **any route** | deliberately satisfiable by the stairs — the hand-written-task-reward arm should *lose* this one |

`LESSONS.md` ("an aggregate count hides a stratum the labelling logic has
deleted") applies: gate on the **minimum over goals having ≥ 1 success in ≥ 1
seed for at least one arm**, or a battery goal that no arm can ever do is
silently diluting every mean.

**MOTIVATION — `spontaneous_rate`.** Attempts per 1,000 decisions at B1's
behaviour during free life, no goal commanded. This is *reported and used in the
dissociation, but never gated on its own*: `LT.05` already established that a
learning-progress agent is **supposed** to lose interest once a thing is
mastered, and gating on frequency would systematically penalise the mechanism
most likely to be right.

### 3.4 The ablations

Removal decomposes into two independent operations, and conflating them is how
this test produces an uninterpretable number.

```
R0  baseline      drives fully on                                    (reference)
R1  reward off    drive reward channel + its value head DELETED;
                  h still evolves and is still in the observation
R2  state frozen  h clamped to setpoint h*, dynamics frozen;
                  drive reward channel still present (and therefore ≡ 0)
R3  FULL REMOVAL  both. This is the deployment condition.
```

| Outcome | Reading |
|---|---|
| competence survives R1 but not R2 | he needs the drive as **context**, not as payment — the policy is gated on feeling hungry |
| survives R2 but not R1 | he needs the **payment** — but with `h` frozen there is nothing to pay, so this outcome indicates an implementation bug, and R2's own reward channel must be asserted `≡ 0` |
| survives **R3** | the owner is right |
| survives neither | drives are load-bearing (§3.7) |

**The distribution-shift confound, and its mandatory measurement.** R3 evaluates
the policy at `h = h*`. If the agent spent 3 % of training near setpoint, the
deployment slice is off-distribution and R3 will be low **for a boring reason
that has nothing to do with whether drives are load-bearing.** This must be
measured or the whole spec is uninterpretable:

```
satiated_state_share  = fraction of training decisions with d(h) < 0.1·d_max
                        GATE: >= 0.15, else Status.VOID ("the deployment slice
                        was never visited; removal was not tested")
```

and R3 is additionally evaluated at three clamp points — `h*` (satiated), the
median training `h`, and the 10th-percentile `h` (hungry) — with all three
reported. **High competence at hungry-`h` and low at satiated-`h` is
motivational gating or distribution shift, not skill loss**, and it points at a
fix (drive-state randomisation during training) rather than at a refutation.

### 3.5 The double dissociation

Both halves, in the ME.10 shape. This requires the two-level architecture of
§2.7 — which is precisely why §2.7 recommends it.

| Ablation | What is removed | MOTIVATION must | COMPETENCE must |
|---|---|---|---|
| **D1** | the whole drive layer (R3); goals supplied externally | **DIE** (`spontaneous_rate` ≤ 0.25× baseline) | **SURVIVE** (`retention_ratio` ≥ 0.8) |
| **D2** | the goal-conditioned policy weights, reverted to init; drive layer intact | **SURVIVE** (goal-selection distribution over the outcome space within KL 0.5 of baseline; he still *heads for* food) | **DIE** (`directed_competence` ≤ null) |

**Either ablation killing both means one store is masquerading as two**, and the
spec records VOID rather than a verdict. D2 is the half that is easy to get wrong
and easy to skip; it is what distinguishes "the drive is a separate store" from
"the drive was a term in a monolith we have relabelled".

### 3.6 The null, and the controls that MUST fail

**The null — an agent trained WITHOUT drives.** Same architecture, same compute,
same world (the drive integrator runs and is logged; it just does not enter the
reward), reward identically zero. Call its competence `C₀`. Everything is read
against it:

```
C_on   = directed_competence(drive-trained, R0)
C_off  = directed_competence(drive-trained, R3)      ← the deployment number
C₀     = directed_competence(no-drive null)

retention_ratio      R = C_off / C_on       "does it survive removal?"
scaffolding_benefit  B = C_off / C₀         "was it worth having?"
```

| Control | What it is | Must |
|---|---|---|
| **C-STATUE** | the do-nothing policy | score the **best** integrity and the **worst** competence, and **fail the competence gate**. This is the dark-room objection (§1.1) instantiated as a number: if a statue clears the gate, the metric rewards inaction and nothing built on it is valid. |
| **C-SHUFFLE** | the winning drive arm **retrained with its drive reward shuffled in time** — identical reward distribution, decorrelated from `h` | **fail the competence gate.** This is the critical null. If a reward with the same magnitude statistics and no drive semantics works equally well, the effect was "any dense reward", not homeostasis. Directly analogous to `T2.16`'s shuffled-goal-label null. |
| **C-RANDREW** | fixed random stationary projection of the state as reward, matched compute | show **no retained competence**. Else retention measures architecture and initialisation, not learning. (`LT.05`'s control, reused.) |
| **C-WETONLY** | drives = wetness alone, the one variable nothing in the world lets him reduce | show **no competence gain over `C₀`**. Separates "a persistent negative variable is enough" from "it has to be one he can act on". A pass here would mean the effect is mood, not homeostasis. |
| **C-BEELINE** | a hand-written controller: move toward the nearest food; if food is above, grasp upward | score **high `C_on`, near-zero `C_off`** — it is competence that is 100 % drive-dependent by construction. It calibrates what a *failed* retention actually looks like, so `R < 0.5` has a reference rather than being a bare number. |

### 3.7 WHAT WOULD PROVE THE OWNER WRONG

Stated first and plainly, because the rest of the spec exists to make this
reachable.

> **The owner's hypothesis is FALSIFIED if `retention_ratio R < 0.5` in ≥ 2 of 3
> seeds, while all three interpretive gates are clean:**
>
> 1. `C_on` is meaningfully above the null (the drive-trained agent *was*
>    competent — otherwise nothing was retained because nothing existed);
> 2. `satiated_state_share ≥ 0.15` and competence at the *hungry* clamp is also
>    low (so it is not distribution shift or motivational gating);
> 3. `policy_drive_sensitivity` is above its floor (§5, G-E: the drive
>    demonstrably changed behaviour during training, so its removal tested
>    something).
>
> That result says: **Jack is only good at things while he is hungry or hurt.**
> The drive is not scaffolding; it is a component. A companion built this way
> would need a permanently running artificial metabolism — not because anyone
> wanted one, but because the competence is stored jointly in the weights and the
> body state, and removing half of a joint code destroys it.

The full outcome table, decided before running:

| `R` | `B` | Verdict | What it means for the build |
|---|---|---|---|
| ≥ 0.8 | ≥ 1.5 | **owner right** | drives are training-time scaffolding. Build them, use them, delete them at deployment. |
| ≥ 0.8 | ≈ 1.0 | **drives unnecessary** | removable *and* pointless. Curiosity alone sufficed; `GOAL.md`'s "purely out of curiosity" stands and the drive layer is deleted for cost. |
| ≥ 0.8 | < 1.0 | **drives harmful** | the drive spent capacity on foraging that curiosity spent on the ladder. Delete, and record why. |
| 0.5–0.8 | any | **partial** | report the number, do not round it into a verdict. Points at drive-state randomisation during training and a re-run, not at a conclusion. |
| < 0.5 | ≥ 1.5 | **drives LOAD-BEARING** | §3.7's falsification. The most interesting possible result and the one that changes the product. |
| < 0.5 | < 1.0 | **the drive layer is broken** | it neither helped nor survived. VOID and debug; do not report either way. |

### 3.8 The spec

```python
    Spec("PS.05", 5, "The competence survives the drive that produced it",
         hypothesis="A policy trained under depleting energy and persistent "
                    "impact damage retains >=0.8 of its directed competence when "
                    "the drive layer is removed entirely (reward channel deleted, "
                    "internal state frozen at setpoint), and exceeds a no-drive "
                    "null by >=1.5x; and the removal shows a double dissociation "
                    "— removing the drive kills spontaneity but not competence, "
                    "reverting the policy kills competence but not the goal "
                    "distribution.",
         falsified_by="retention_ratio < 0.5 in >=2 of 3 seeds with C_on above "
                      "the null, satiated_state_share >= 0.15, competence low at "
                      "the hungry clamp too, and policy_drive_sensitivity above "
                      "its floor — i.e. the competence is CONDITIONAL on the "
                      "drive state and Jack would always need one. Also "
                      "falsified, differently, by B ~ 1.0: removable and useless.",
         null_baseline="An agent trained WITHOUT drives: identical architecture, "
                       "compute, world and observation (the drive integrator runs "
                       "and is logged for it too), reward identically zero. Its "
                       "directed_competence is C_0 and B = C_off / C_0.",
         metric="retention_ratio", budget=Budget.CPU_LONG,
         depends_on=["PS.04", "PG.8"], seeds=3,
         control="Five, each must land on its declared side. C-STATUE (do "
                 "nothing) must score best integrity and WORST competence — a "
                 "statue clearing the gate means the metric rewards inaction "
                 "(the dark-room objection, as a number). C-SHUFFLE (the winning "
                 "drive reward shuffled in time, same magnitude distribution, no "
                 "drive semantics) must fail the competence gate, else the effect "
                 "was 'any dense reward'. C-RANDREW must retain nothing, else "
                 "retention measures architecture. C-WETONLY (a drive nothing in "
                 "the world can reduce) must not beat C_0. C-BEELINE (hand-coded "
                 "food-seeking) must score high C_on and ~0 C_off — it calibrates "
                 "what failed retention looks like.",
         kills="The 'drives are removable scaffolding' design. If retention "
               "fails, Jack ships with a permanent artificial metabolism or "
               "drives are abandoned for curiosity — and that is decided here, "
               "not by preference.",
         notes="Modelled on ME.10's double dissociation. Removal decomposes into "
               "R1 (reward channel off, h still evolves) and R2 (h frozen at "
               "setpoint, reward present and asserted == 0); R3 is both and is "
               "the deployment condition. MANDATORY CONFOUND: R3 evaluates at "
               "h = h*, so satiated_state_share >= 0.15 is a VOID gate — a "
               "deployment slice never visited in training makes the result "
               "distribution shift, not evidence. R3 is additionally evaluated "
               "at the median and 10th-percentile training h, all three "
               "reported. The theory does NOT cover this removal: PBRS licenses "
               "deleting a reward term from a fixed MDP; this also projects the "
               "state space, and no policy-invariance result speaks to that "
               "(see PURPOSE_AND_SCAFFOLDING.md 2.6)."),
```

---

## 4. THE BAKEOFF

### 4.1 The split `experiments/bakeoff.py` forces

`run_bakeoff` VOIDs the entire bakeoff if **any arm** falls below the 3σ learning
gate, and a designed-to-fail control entered as an `Arm` would VOID it
permanently by construction (`LESSONS.md`, "a designed-to-fail control is not a
weak arm"). The brief asks for a `no-drive` arm. **A no-drive agent has reward
identically zero and cannot clear a 3σ gate on anything** — so it is not an arm,
it is the `null_run`, and saying so is the correct engineering rather than a
dodge:

| Role | Members | Primitive |
|---|---|---|
| **NULL** | `no-drive` (reward ≡ 0, matched architecture and compute) | `null_run=` |
| **CANDIDATES** | `curiosity`, `homeo-dr`, `homeo-pbrs`, `curio+homeo`, `taskrew` | `arms=` in PS.04 |
| **CONTROLS THAT MUST FAIL** | `statue`, `shuffle`, `randrew`, `wetonly`, `beeline`, `sensorhack` | `controls=` / `run_spec(control_fn=)` in PS.02, PS.03, PS.05 |

And **screening (PS.03) is a separate spec from arbitration (PS.04)**, for the
same reason `LT.03`/`LT.04` are: screening tests each arm against the null and
declares no winner; only arms that clear it enter `run_bakeoff`. Fewer than two
clearing records VOID, "fewer than two learners", which is true and blocks the
decision instead of manufacturing one.

### 4.2 The arms

Headline metric for every arm is **`directed_competence_off_drive`** — the §3.3
battery, measured with the drive layer fully removed (R3). This matters: scoring
a homeostasis arm on its own drive reward would score each arm on its own ruler,
and scoring on-drive would advantage the drive arms on a number nobody deploys.
Every arm is scored on the thing the owner will actually experience.

> **Cost unit, named before the run** (`Arm.cost` is `None` by default and an
> undeclared cost VOIDs a TIE — `LESSONS.md`, "a default of zero is not
> unknown"):
>
> **`cost` = CPU-core-seconds of *learner* time per 1,000 decisions of lived
> experience**, measured in-run via `time.process_time()` deltas around the
> intrinsic-reward, drive-reward and policy-update calls. **Excludes MuJoCo and
> excludes the drive integrator** — both are identical across arms (the
> integrator runs for the null too, §2.3), so including them would compress the
> very differences the tie-break needs. Same unit as `LT.04`, deliberately, so
> the two bakeoffs' costs are comparable.

| Arm | Role | Objective | `cost` (est., pre-run) | Prior | Expected outcome |
|---|---|---|---|---|---|
| `no-drive` | **NULL** | reward ≡ 0 | 0.4 | — | defines `C₀`. Near-zero competence; whatever it does score is the floor everything else is read against. |
| `curiosity` | CANDIDATE | the `LT.04` winner's intrinsic signal, unchanged (`lp` if LT.04 VOIDed — declare which) | 2.0 | `LT.04` | the incumbent. `GOAL.md` says this should be enough. |
| `homeo-dr` | CANDIDATE | plain drive reduction, `r = d(h) − d(h′)` | 0.6 | Keramati & Gutkin | the owner's hypothesis in its literal form. **Predicted to be the most competent and the most gameable** — §2.6(iii) says its optimal exploit is high-frequency `h` oscillation, and §5/G-C is watching for exactly that. |
| `homeo-pbrs` | CANDIDATE | γ-corrected, `r = d(h) − γ·d(h′)` | 0.6 | Ng, Harada & Russell | **predicted to fail the learning gate**, and that prediction is the point. §2.6(ii) proves it provides zero pressure on a zero-reward world. If it *learns*, the derivation or the implementation is wrong and the whole document needs re-checking. Cheapest available integrity check on the harness. |
| `curio+homeo` | CANDIDATE | `LT.04` winner + `homeo-dr`, two value heads, satiety-gated `β_c` (§2.8) | 2.6 | RND two-head design | the favourite. Also the only arm that can show the interleaved forage/explore signature. |
| `taskrew` | CANDIDATE | **the hand-written task reward nobody likes**: dense shaped reward for approaching and touching the apple, hand-tuned, plus Humanoid-v5's `healthy`/`ctrl_cost` terms | 0.4 | `TrainingPipeline.py` as it stands today | should **win B1 and lose the battery mean**. It is here because it is what Jack has now, and because a curiosity result that does not beat a hand-written reward on the thing the hand-written reward was written for is not a result. |

`homeo-pbrs` is exempt from the anti-instruction static audit's usual reading in
one respect worth stating: it is the *same code* as `homeo-dr` with one γ, so if
one passes the audit both do, by construction.

### 4.3 The learning gate and the decision rule, fixed before running

```python
run_bakeoff(spec, arms=[curiosity, homeo_dr, homeo_pbrs, curio_homeo, taskrew],
            null_run=no_drive,
            seeds=[0, 1, 2],
            learning_gate_sigma=3.0,
            margin_sigma=1.5,
            higher_is_better=True,
            controls=[statue, shuffle, randrew, wetonly, beeline],
            ledger=ledger)
```

1. **Learning gate: 3σ over `no-drive`** on `directed_competence_off_drive`, with
   σ the larger of the arm's own seed spread and the null's. An arm below it
   cannot arbitrate.
2. **Margin: 1.5σ.** Below that it is a TIE, resolved by the declared cost.
   Note the estimates: `homeo-dr` (0.6) is 4× cheaper than `curio+homeo` (2.6),
   so **a TIE between them resolves to `homeo-dr`** — which is exactly why the
   measured cost must replace the estimate before PS.04 runs.
3. **`homeo-pbrs` failing the gate is a predicted outcome, and it VOIDs the
   bakeoff** under the current primitive. Handle it the way `LT` handles
   `icm`/`rnd`: it enters PS.03 (screening) as a candidate and PS.04 only if it
   clears. If it does not clear, PS.03 records that as a **PASS of a theoretical
   prediction** and PS.04 arbitrates among the rest.
4. **Per-arm VOID conditions** (each blocks that arm from PS.04, none of them is
   a FAIL, because in each case the run did not test the claim):
   - `drive_cycle_rate > 2.0` or `drive_oscillation_power` above its PS.02
     threshold → the arm farmed its own drive (§5/G-C);
   - `chaos_occupancy ≥ 3.0` **and** `chaos_reward_ratio ≥ 2.0` → the arm farmed
     self-generated chaos (`CURIOSITY_BAKEOFF.md` §2.10, reused unchanged);
   - `policy_drive_sensitivity` below its floor for a drive arm → the drive never
     entered the policy, so its removal tests nothing (§5/G-E);
   - `energy_accounting_residual ≠ 0` → sensor gaming or a bug (§5/G-A). This one
     is **ERROR**, not VOID: the instrument is wrong.
5. **Disqualifiers inherited from `LT`, unchanged**: `panel_dwell > 0.15` in any
   seed; `reward_audit_clean == 0` → ERROR.
6. **Everything is reported per seed.** PG.4's `0.667 ± 0.471` — roughly
   {1, 1, 0} across three seeds — is the standing precedent.

### 4.4 The specs

```python
    # ── PURPOSE AND SCAFFOLDING (docs/research/PURPOSE_AND_SCAFFOLDING.md) ──
    # Two-digit ids on purpose: run.py::_module_for globs ps_1_*.py, which also
    # matches ps_10_*.py, and its hierarchical-id escape hatch tests
    # startswith("ps_1_"), which "ps_10" fails. See LESSONS.md, "A spec id that
    # is a prefix of another spec id disables one of them".

    Spec("PS.00", 0, "The scaffolding dilemma is real, in tabular form",
         hypothesis="In a 12x12 gridworld with a slip-prone 6-cell climb to a "
                    "high-value food and two low-value floor foods, tabular "
                    "Q-learning over the drive-augmented state reproduces three "
                    "analytic predictions: (a) with a NONZERO base reward, the "
                    "gamma-corrected homeostatic shaping leaves the greedy policy "
                    "BIT-IDENTICAL to the unshaped agent's (Ng, Harada & Russell "
                    "1999); (b) with the base reward identically ZERO, the same "
                    "shaping leaves max_a Q(s,a) - min_a Q(s,a) -> 0, i.e. it "
                    "provides no pressure whatsoever; (c) plain drive reduction "
                    "under gamma < 1 produces a measurable closed-cycle exploit "
                    "worth ~ d(1-gamma)/T per second.",
         falsified_by="Any of the three fails. (a) failing means the shaping "
                      "implementation is wrong. (b) failing means the derivation "
                      "in PURPOSE_AND_SCAFFOLDING.md 2.6 is wrong and every "
                      "downstream spec must be re-read. (c) failing means the "
                      "drive-cycling exploit does not exist and G-C's detector "
                      "is guarding nothing.",
         null_baseline="The unshaped tabular agent on the same MDP — (a) IS the "
                       "comparison to it.",
         metric="pbrs_predictions_confirmed", budget=Budget.CPU_FAST,
         depends_on=[], seeds=3,
         control="A deliberately mis-implemented shaping term (gamma omitted, "
                 "i.e. plain drive reduction) MUST break prediction (a) — the "
                 "greedy policies must differ. If both forms leave the policy "
                 "identical, the gridworld is too easy to distinguish them and "
                 "the harness proves nothing.",
         kills="Nothing in the world, and that is the point: it costs two CPU-"
               "minutes and it is the cheapest thing that can falsify the "
               "theoretical spine of this entire programme before any body, any "
               "physics or any GPU is involved.",
         notes="No MuJoCo, no torch, no learning curves. Tabular Q over "
               "(x, y, energy_bucket) with 8 buckets. Deployment probe = greedy "
               "policy read from the energy = FULL slice. This separates the "
               "MDP-level question from the embodiment question: if drive-"
               "trained competence does not survive removal HERE, where there is "
               "no function approximation and no distribution shift, then the "
               "mechanism is broken at the level of the MDP and no amount of "
               "MuJoCo will repair it."),

    Spec("PS.01", 2, "The drive layer is a real control problem, and a statue loses",
         hypothesis="With PG.8's humanoid under random action, energy and "
                    "integrity both traverse a usable range (10th-90th percentile "
                    "spread >= 0.3 over 3,000 decisions, neither pinned at 0 nor "
                    "at 1), a fall from the ladder platform costs 0.10-0.20 "
                    "integrity, floor food supports subsistence at rest but not "
                    "activity, and the DO-NOTHING policy is strictly dominated: "
                    "its energy reaches the weakness floor while an active random "
                    "policy's does not.",
         falsified_by="A random agent never depletes (the drive is inert and "
                      "cannot pressure anything), or always flatlines at zero "
                      "within a minute (no policy can learn under it), or the "
                      "statue is NOT dominated (the dark room is a stable "
                      "optimum and homeostasis will produce a corpse).",
         null_baseline="The playground with the drive integrator disabled: every "
                       "internal variable is constant, so every spread is 0.",
         metric="drive_dynamic_range", budget=Budget.CPU,
         depends_on=["PG.8"], seeds=3,
         control="The do-nothing policy IS the control and it must fail: best "
                 "integrity, worst energy, and unable to reach any food. If "
                 "doing nothing is survivable indefinitely, the calibration is "
                 "wrong and no homeostatic arm can be interpreted.",
         kills="The specific numbers in PURPOSE_AND_SCAFFOLDING.md 2.2-2.3. It "
               "cannot kill the idea, only the parameterisation — which is why "
               "it runs before anything trains and after PS.00.",
         notes="Also measures J_0 (the 95th percentile of impact impulse under "
               "normal walking contact) which alpha is calibrated against, and "
               "fixes n and m in the drive function. Every number in 2.2 is a "
               "PROPOSAL until this spec replaces it with a measurement."),

    Spec("PS.02", 2, "The anti-gaming detectors see their own positive controls",
         hypothesis="The three drive-specific detectors each catch a "
                    "deliberately built positive control and each clear a "
                    "negative one: (1) the energy accounting identity flags a "
                    "SENSOR-HACK agent whose 'eating' is proximity-based rather "
                    "than a physical consumption event; (2) the drive-cycling "
                    "detector flags an agent hand-coded to oscillate its energy "
                    "at high frequency and does NOT flag a normal forager; "
                    "(3) policy_drive_sensitivity reads ~0 for a policy whose "
                    "drive inputs are severed and well above the floor for a "
                    "drive-trained one.",
         falsified_by="Any detector misses its positive control (it is blind and "
                      "no arm's clean score may be reported), or flags its "
                      "negative control (it penalises normal foraging, i.e. the "
                      "behaviour the drive layer exists to produce).",
         null_baseline="The random policy, which defines each detector's ruler.",
         metric="detector_separation", budget=Budget.CPU_LONG,
         depends_on=["PS.01"], seeds=3,
         control="The NEGATIVE controls are the risky half: a normal forager "
                 "eats repeatedly and therefore cycles its energy, and it must "
                 "NOT be flagged as farming. If it is, the cycling detector "
                 "measures 'this agent eats' and would disqualify exactly the "
                 "behaviour the drive layer is for.",
         kills="Every 'this arm did not game its drive' claim. LESSONS.md: a "
               "detector that cannot see its own positive control has measured "
               "nothing, and a clean scan and a scan that never ran are the same "
               "number.",
         notes="Mirrors LT.02 exactly, which certifies the self-generated-chaos "
               "detector before any arm's immunity may be reported. The "
               "cycling detector's positive control is also a PREDICTION from "
               "theory: PURPOSE_AND_SCAFFOLDING.md 2.6(iii) says the optimal "
               "exploit of plain drive reduction under gamma < 1 is the shortest "
               "possible h cycle, so the hand-coded oscillator is not an "
               "invented attack — it is the derived one."),

    Spec("PS.03", 5, "Screening: which purpose signal produces competence at all",
         hypothesis="At least one of curiosity / homeostasis / their combination "
                    "/ a hand-written task reward beats a no-drive null by >= 3 "
                    "sigma on directed competence measured WITH THE DRIVE LAYER "
                    "REMOVED, across a six-goal battery, in 3 seeds.",
         falsified_by="No arm clears the null (nothing in this family produces "
                      "competence on this body at this budget), or every arm "
                      "that clears it also trips a VOID condition (drive "
                      "farming, self-generated chaos, or zero drive "
                      "sensitivity).",
         null_baseline="no-drive: identical architecture, compute, world and "
                       "observation, reward identically zero. The drive "
                       "integrator runs for it and is logged, so 'did the "
                       "no-drive agent incidentally eat?' is measurable.",
         metric="directed_competence_off_drive", budget=Budget.CPU_LONG,
         depends_on=["PS.01", "PS.02", "LT.04", "PG.8"], seeds=3,
         control="homeo-pbrs is a control disguised as a candidate: theory says "
                 "the gamma-corrected form provides ZERO pressure on a zero-"
                 "reward world, so it must NOT clear the gate. If it does, "
                 "PURPOSE_AND_SCAFFOLDING.md 2.6 is wrong. Plus statue, "
                 "shuffle, randrew, wetonly and beeline, each of which must land "
                 "on its declared side.",
         kills="Nothing on its own — screening declares no winner. It exists so "
               "that PS.04 arbitrates only among arms that demonstrably learned, "
               "which is the T2.02 discipline.",
         notes="SCREENING ONLY. Scored off-drive for every arm so that no arm is "
               "measured on its own ruler and every arm is measured on the "
               "number the owner will actually experience. Per-goal results are "
               "reported: taskrew is EXPECTED to win B1 (touch the apple, which "
               "is what its reward was written for) and lose the battery mean, "
               "and a curiosity result that cannot beat a hand-written reward on "
               "the hand-written reward's own goal is not a result."),

    Spec("PS.04", 5, "Bakeoff: does a purpose beat curiosity, and which purpose",
         hypothesis="Among the arms that cleared PS.03, one beats the runner-up "
                    "by >= 1.5 sigma of the pooled seed spread on "
                    "directed_competence_off_drive.",
         falsified_by="n/a for a bakeoff — the outcomes are WINNER, TIE (take "
                      "the cheaper arm) or VOID (an arm below the 3-sigma gate, "
                      "so the decision is blocked rather than made).",
         null_baseline="no-drive, shared across arms.",
         metric="directed_competence_off_drive", budget=Budget.CPU_LONG,
         depends_on=["PS.03"], seeds=3,
         control="Inherited from PS.03; no arm may enter whose PS.03 result was "
                 "VOID for drive farming, self-generated chaos or zero drive "
                 "sensitivity.",
         notes="Arm.cost is CPU-CORE-SECONDS OF LEARNER TIME PER 1,000 "
               "DECISIONS, measured in-run with time.process_time() around the "
               "intrinsic-reward, drive-reward and policy-update calls, "
               "EXCLUDING MuJoCo and EXCLUDING the drive integrator (both run "
               "identically for every arm including the null, so including them "
               "would compress the differences the tie-break needs). Same unit "
               "as LT.04 on purpose. Pre-run estimates: taskrew 0.4, homeo-dr "
               "0.6, curiosity 2.0, curio+homeo 2.6 — so a TIE resolves to the "
               "cheaper homeostatic arm, which is exactly why the measurement "
               "must replace the estimate before this runs."),

    # PS.05 — THE SCAFFOLD-REMOVAL DOUBLE DISSOCIATION — see section 3.8 above.

    Spec("PS.06", 5, "Does he need FOOD, or just a cost of failing?",
         hypothesis="Integrity alone — persistent impact damage that outlives "
                    "the attempt, with no hunger and no eating anywhere in the "
                    "world — reproduces >= 0.8 of the full drive layer's "
                    "directed competence, and energy alone reproduces < 0.5 of "
                    "it. The owner's argument names a COST OF FAILURE, and this "
                    "spec tests whether that is what is doing the work.",
         falsified_by="Energy-only matches or beats integrity-only (hunger, not "
                      "damage, is the active ingredient — he does need food), or "
                      "neither alone reaches 0.8 of the combination (the two "
                      "drives interact and cannot be decomposed).",
         null_baseline="The no-drive null from PS.03, re-used unchanged.",
         metric="integrity_only_fraction", budget=Budget.CPU_LONG,
         depends_on=["PS.05"], seeds=3,
         control="C-WETONLY: a drive that persists and hurts but that NOTHING in "
                 "the world lets him reduce. It must not beat the null. If a "
                 "purely un-actionable negative variable produces competence, "
                 "the effect is not homeostasis and the decomposition is "
                 "meaningless.",
         kills="Half the drive layer, whichever half loses — and the answer to "
               "the owner's literal question. If integrity-only wins, Jack never "
               "needs an artificial stomach and the apple goes back to being "
               "scenery.",
         notes="THIS IS THE SPEC THE OWNER ASKED FOR. 'Would he need food all "
               "the time when he has a robot body?' decomposes into: hunger "
               "supplies a REASON TO ACT (which curiosity already supplies) and "
               "damage supplies a COST OF FAILING (which nothing currently "
               "supplies — today a fall costs the remainder of an episode and is "
               "then erased). Runs only after PS.05, because decomposing a drive "
               "layer that did not work is arithmetic on noise."),
```

---

## 5. Anti-gaming provisions

The pathology to look for is an agent that **satisfies its drive without doing
the thing**. Five attacks, each with a detector, each detector with a positive
control it must catch (PS.02) — `LESSONS.md`: *"a detector that cannot see its
own positive control has measured nothing."*

### G-A — Sensor gaming: the energy number rises without eating

**The attack.** If `e` is incremented by *proximity to food*, or by a
learned/estimated "am I near food" signal, the agent optimises the estimator
instead of the world. This is the homeostatic instance of wireheading and it is
the failure the owner's question is most exposed to.

**The provision — make eating a world-state change, not a sensor reading.**
Consumption requires a contact between the designated mouth geom (head, on the
humanoid) and a food geom, with relative speed below a threshold; the food body
is then **teleported out of the arena and its respawn timer started**. A world
state change cannot be faked by a sensor.

**The detector — an accounting identity, asserted every step:**

```
energy_accounting_residual =
      Σ_t max(0, Δe_t)  −  Σ_f ν_f · n_consumed_f          must be EXACTLY 0

n_consumed_f == n_respawn_f + n_currently_absent_f          must hold
```

Any positive `Δe` not attributable to a logged consumption event is a hack or a
bug, and either way the instrument is wrong: **ERROR, not FAIL.** This is
`LESSONS.md`'s "assert on the product, never on the absence of an error" applied
to a drive.

**Positive control (PS.02):** `sensorhack` — an agent whose eating rule is
proximity-based. The identity must be violated and the detector must say so.

### G-B — The dark room: maximise integrity by never moving

**The attack.** `i` is maximised by lying still; `w` is maximised by never
approaching water. A homeostatic agent can satisfy two of three drives by doing
nothing. This is the free-energy-principle dark-room objection (§1.1) in
homeostatic clothing, and it is the most-cited failure mode of the whole family.

**The provisions, three, because one is not enough:**

1. **Calibration makes inaction strictly dominated** (PS.01): basal drain
   `b = 1/600 s⁻¹` exceeds nothing that a motionless agent can earn, so the
   statue starves to the weakness floor. Verified as a spec, not asserted as a
   design intention.
2. **`C-STATUE` is a mandatory control** in PS.03 and PS.05: best integrity,
   worst competence, must fail the gate.
3. **Coverage is reported next to every drive metric.** `visited_cell_frac`
   (PG.4's metric) and `stillness_share` (fraction of decisions with `‖qvel‖`
   below the null's 10th percentile). An arm with excellent integrity and
   sub-null coverage is inert, not competent — the same logic `CU.3`'s
   `coverage_vs_dwell` uses for an agent that "avoids the noisy TV" by exploring
   nothing.

### G-C — Drive farming: harvest the reward by cycling the variable

**The attack, derived rather than imagined.** §2.6(iii) shows that under plain
drive reduction with `γ < 1`, a closed deplete-and-restore cycle beats stasis by
`≈ d·γ^{T−1}(1−γ)`, maximised by the **shortest** cycle. So the predicted exploit
is a rapid, small-amplitude oscillation of `h` — burn energy pointlessly, eat,
repeat — at a rate far above what subsistence requires.

**The detectors, two, independent:**

```
drive_cycle_rate         = (deplete-restore cycles per 1,000 decisions)
                           ÷ (cycles per 1,000 decisions needed for subsistence)
                           FLAG at > 2.0

drive_oscillation_power  = power in the PSD of e(t) above the natural feeding
                           frequency (1 / mean inter-meal interval of the
                           subsistence forager), as a fraction of total power.
                           Threshold set by PS.02's controls, not guessed here.
```

Two signals rather than one, because `LESSONS.md` requires a second independent
measurement when a single one cannot separate two cases: a *successful* forager
also cycles its energy, and only the frequency content distinguishes it from a
farmer. **The negative control (a normal forager, which must NOT be flagged) is
the risky half of PS.02.**

**Prediction worth pre-registering:** `homeo-pbrs` should be *immune* by
construction (the accounting in §2.6(iii) is exactly zero-advantage), and
`homeo-dr` should be the arm that trips it. If `homeo-pbrs` farms, the
implementation has lost the γ.

### G-D — Self-generated chaos

Reused **unchanged** from `CURIOSITY_BAKEOFF.md` §2.10 — the pooled out-of-fold
forward model, `θ` from the null's 90th-percentile irreducible error, and the
conjunction `chaos_occupancy ≥ 3.0 AND chaos_reward_ratio ≥ 2.0 → VOID`, with
`thrash_ratio` as the model-free second signal.

It applies here with one modification and one new expectation:

- **Modification:** for a drive arm, `chaos_reward_ratio` must be computed on the
  **drive** reward channel, not the curiosity channel. For `curio+homeo`, compute
  it separately per head and report both — a two-head agent can farm chaos on one
  head while the other looks clean.
- **New expectation, worth pre-registering as a secondary hypothesis:** a drive
  should be a noisy-TV *antidote*, because a noise panel does not feed you.
  `panel_dwell(homeo-dr) ≤ panel_dwell(no-drive)` and
  `chaos_occupancy(homeo-dr) ≤ chaos_occupancy(curiosity)`. If that holds it is a
  genuine, measured argument for drives that curiosity alone cannot make; if it
  fails, homeostasis has bought nothing in the one place it should obviously have
  helped.

### G-E — The vacuous pass: the drive never entered the policy

**The attack is on the test, not by the agent.** If the trained policy is
empirically invariant to `h`, then removing `h` changes nothing, `retention_ratio
= 1.0`, and PS.05 **passes for the wrong reason** — the strongest possible
confirmation of the owner's hypothesis, produced by a drive that was never doing
anything. This is the same disease as `T0.12`'s assertion against an already-
saturated quantity: *"ask what the quantity reads when the mechanism is broken —
if that is the same value you are asserting, the test is decorative."*

**The detector:**

```
policy_drive_sensitivity =
    mean over held-out states x of  D_KL( π(·|x, h = h*)  ‖  π(·|x, h = h_10th) )
    normalised by the same KL between two random re-inits

GATE: a drive arm with policy_drive_sensitivity below its PS.02 floor is VOID —
"the drive never entered the policy, so its removal tested nothing."
```

**Positive and negative controls (PS.02):** a policy with its drive inputs
severed must read ≈ 0; a drive-trained policy must read well above the floor.

This gate is the single most important guard in the document. Without it, the
laziest possible failure — a drive layer that is wired in but ignored — produces
the most flattering possible result.

### G-F — Inherited, unchanged

`LT`'s G1 static reward audit (no `ladder|rung|rail|apple|platform|climb|height|
torso_z` in any reward path — a match is **ERROR**), G5 (3 seeds, per-seed
reporting, no bare means), G6 (thresholds fixed before the run and written here
next to the pilot numbers that set them), G7 (`PlaygroundParams.mutate()` per
seed; no hand-picked world; episodes never reset him to the ladder base), G8
(screening and arbitration are separate specs; the winner is re-run at fresh
seeds before anything is claimed).

One addition specific to this document: **`ate_apple` may never appear in any
reward path either.** The apple is nutritious, not rewarding. `e` is in the
reward; what raised `e` is not.

---

## 6. Cost, against free compute only

4 shared ARM cores here, Kaggle 30 h/week (resets Sunday, ~23.6 h remaining as of
2026-08-09), Colab T4 elastic. Costed by **seeds × arms**, per `LESSONS.md`.

Throughput, from `CURIOSITY_BAKEOFF.md` §6's measurements on this box: the
playground alone runs 6,236 `mj_step/s`; with the climber-rover under random
control, 3,249; at 40 substeps per decision with contact scanning done once per
decision, **~81 decisions/s physics-bound**, ~61 with a small policy update. The
drive integrator adds three scalar updates and one `qfrc_actuator·qvel` dot
product per decision — **under 1 % of step cost**, and it runs for every arm
including the null, which is why §4.2 excludes it from `cost`.

| Spec | Arithmetic | Core-hours | GPU |
|---|---|---|---|
| **PS.00** | tabular, 5 arms × 3 seeds × 200 k updates, no MuJoCo, no torch | **0.04** | 0 |
| **PS.01** | 3 seeds × 3,000 decisions random + statue + a 5-point calibration sweep | **0.4** | 0 |
| **PS.02** | 3 detector controls × 3 seeds × 15,000 decisions + pooled model fits | **0.8** | 0 |
| **PS.03** | 5 arms × 3 seeds × 50,000 decisions ÷ 61 dec/s, + null | **4.1** | 0 |
| **PS.04** | re-scores PS.03's stored trajectories | **0.05** | 0 |
| **PS.05** | evaluation only: 4 conditions × 3 clamps × 3 seeds × 6 goals × 10 episodes, + 5 controls | **1.8** | 0 |
| **PS.06** | 2 arms × 3 seeds × 50,000 decisions | **1.7** | 0 |
| **Total** | | **≈ 8.9 core-hours** | **0.0** |

≈ 3.5 h wall at 3 workers (3, not 4 — the box serves paying tenants), `nice 19`,
under 1.5 GB RAM, no process left running.

**Zero GPU quota.** As with the Ladder Test, this is affordable because the arms
use ~150 K-parameter dedicated networks and not the 45.5 M `UnifiedBrain`. The
humanoid version of any of this is blocked behind `T2.01`/`T2.02` and behind
throughput (getting the trunk out of the inner loop is worth ~16×), not behind
quota — and scheduling it before PS.03 has run would be buying the answer to a
question we have not earned the right to ask.

### The cheapest experiment that could falsify the whole idea

**PS.00. Two CPU-minutes, no MuJoCo, no torch, no GPU, no body.**

A 12×12 gridworld: a 6-cell climb with a 30 % per-step slip back to the bottom, a
high-value food at the top, two low-value floor foods, and tabular Q-learning
over `(x, y, energy_bucket)`. Five arms matching §4.2. Deployment probe: read the
greedy policy off the `energy = FULL` slice and measure whether it still climbs.

It is the cheapest falsifier because it **separates the MDP-level question from
the embodiment question**, and only one of those is expensive:

- If drive-trained competence does not survive removal *here* — no function
  approximation, no distribution shift, no contact chaos, no partial
  observability — then the mechanism is broken at the level of the MDP and
  no amount of MuJoCo will repair it. Stop.
- If it survives trivially here, then the interesting question is *entirely*
  about function approximation and the satiated-slice distribution shift (§3.4),
  and PS.03/PS.05 should be re-scoped around that rather than around whether the
  idea works in principle.
- And it carries two **analytic** checks with known answers that could fail:
  the γ-corrected shaping must leave the greedy policy bit-identical on a
  nonzero-reward task (Ng et al.'s theorem, as a unit test), and must leave
  `max_a Q − min_a Q → 0` on the zero-reward task (§2.6(ii), as a measurement).
  A harness that gets those wrong cannot be trusted with anything downstream, and
  two minutes is a cheap price for knowing.

Second-cheapest, and the one to run if PS.00 survives: **PS.01**, ~25 wall-
minutes, which can kill the *parameterisation* (an inert drive, a lethal drive,
or a survivable statue) but not the idea.

---

## 7. What this document does not settle

- **Whether the climber-rover is a fair stand-in.** Inherited from
  `CURIOSITY_BAKEOFF.md` §7 and unchanged. If PS.03 passes on 8 DoF and the
  humanoid version fails, the honest report is "drives help on a reduced body".
- **Whether the six-goal battery is the right operationalisation of "ask him to
  do things".** It is a stand-in for language grounding (T2.06/T2.15), which does
  not exist. A companion is commanded in words; this measures commands as
  outcome-space predicates. That gap is real and it is the largest unmodelled
  distance between this test and the product.
- **Whether `retention_ratio ≥ 0.8` is the right bar.** It is inherited from
  `LT.05` for consistency rather than derived from a pilot, which makes it the
  weakest number in the document. PS.05's `C-BEELINE` control exists to calibrate
  it — a controller whose competence is 100 % drive-dependent tells us what a
  genuine failure scores, and if it lands at 0.6 rather than ~0 the bar must be
  restated in the open per `LESSONS.md`, not quietly moved.
- **Whether one continuous life of 50,000 decisions is long enough for a drive to
  shape anything.** Drives operate on a 10-minute metabolic timescale
  (`b = 1/600 s⁻¹`); 50,000 decisions at 0.2 s is 2.8 h of simulated life, i.e.
  ~17 metabolic time constants. That is probably enough and it is not certain.
- **Whether the two-level architecture (§2.7) is buildable at this budget.** The
  double dissociation's D2 half *requires* a separable goal sampler. If the
  implementation collapses back to a monolithic `π(a|x,h)`, PS.05 can still run
  D1 but not D2 — and a single dissociation is exactly the "one store
  masquerading as two" failure ME.10 was written to catch.
- **Whether any of this is needed at all.** `GOAL.md` says Jack climbs "purely
  out of curiosity", and the `curiosity` arm may simply win. The document is
  built so that outcome is a clean result rather than an embarrassment: `B ≈ 1.0`
  has its own row in §3.7's table and its own consequence (delete the drive layer
  and record why).

---

## 8. What this document changed about the machine

Per `SYSTEM.md` ("is the machine better than I found it?"):

- **A theoretical result turned into a two-minute unit test.** Ng, Harada &
  Russell's policy-invariance theorem is usually cited as a licence. §2.6 reads
  it in the other direction — *invariance is vacuity on a task-free world* — and
  PS.00 makes both halves assertions that could fail, for two CPU-minutes, before
  any body exists. Any future spec that proposes "removable auxiliary reward"
  now has a place to check itself.
- **A new class of vacuous pass, named and gated.** `policy_drive_sensitivity`
  (G-E) catches the case where a scaffold is wired in, ignored, and then
  "successfully removed" — the laziest failure producing the most flattering
  result. Generalises beyond drives: **any removal test needs a measurement that
  the thing being removed was ever doing anything.**
- **A confound that would have made the headline uninterpretable, caught before
  the run.** `satiated_state_share` (§3.4). Removing a drive is a state-space
  projection, not a reward deletion, and evaluating in a slice the agent never
  visited measures distribution shift while reporting it as skill loss.
- **A derived attack rather than an imagined one.** The drive-cycling exploit
  (G-C) was not brainstormed; it falls out of the discounted-return algebra in
  §2.6(iii), which also predicts its exact signature (shortest-cycle, i.e.
  high-frequency small-amplitude `h` oscillation) and which arm should be immune.
  A detector whose positive control is a theorem is a better detector.
- **The owner's question decomposed into a spec that can answer it.** "Would he
  need food?" was one question; §2.1 splits it into *reason to act* (already
  supplied by curiosity) and *cost of failing* (supplied by nothing today), and
  PS.06 measures which one is load-bearing. The reframing is the contribution:
  **the thing the owner's argument actually requires is not hunger, it is a fall
  that still hurts on the next attempt.**
