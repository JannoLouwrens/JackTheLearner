# FIELD_WATCH.md — the scout's current-state report

> **Rewritten weekly. This is a state, not a log.** Every entry here is either
> live (nominated, awaiting the builder/owner) or on the watchlist. Superseded
> entries are deleted, not archived — the one-line history lives in
> `docs/FIELD_WATCH_LOG.md`.
>
> **What this file may and may not do.** It NOMINATES arms. It adopts nothing,
> changes no spec, no threshold, no decision. Every nomination below is a
> candidate for a bakeoff that the builder and the owner decide to run.
> `SYSTEM.md` law 3: decisions are made by bakeoff, never by argument — and a
> field watch that argued would be making decisions.

**Sweep date:** 2026-08-24 · **Window:** ~2026-02 → 2026-08 (6 months)
**Scout:** field watch, week 4. **Twelve days since week 3** (2026-08-12) —
the first sweep on the intended weekly-or-slower cadence rather than a daily
one, and week 3's own embargo (*"fronts 1–3 should not be re-swept before
~2026-08-19"*) is spent.

---

## 0. WHY THIS SWEEP IS SHAPED DIFFERENTLY — two measurements landed that
## re-point the whole search

A scout that reads the same literature the same way every week is a subscription,
not an instrument. Two things happened in the interval, and both of them move
where the useful papers are.

**(1) `LC.03` CONCLUDED, and it named exactly one learner: `wm-latent` (A4).**
The v2 re-screen (4× envelope, 400 K decisions / 17,280 core-s per arm-seed,
~190 core-h, 08-21 → 08-23) recorded VOID `"fewer than two learners
(1 cleared)"` on a clean rig. Per arm, t vs null / t vs twin:

| arm | t vs null | t vs twin | note |
|---|---|---|---|
| **`wm-latent` (A4)** | **4.65** | **4.00** | needs rising +0.022, cross-life transfer +92.2 s — every conjunct green |
| `wm-efe` (A3) | 2.05 | 2.07 | below the 3σ gate |
| `ppo-lp` (A1) | 1.20 | 1.10 | needs FALLING |
| `ppo-needs` (A0) | 1.06 | 0.99 | |
| `dreamer-xs` (A2) | **−0.94** | −0.99 | +46 s → **−48.5 s** at 4× the budget |

The consequence for this desk is blunt and it costs me two nominations.
**Week 3's N1 (CIG) entered on `A3`, which read 2.05. Week 3's N2 (Optimistic
World Models) entered on `A2`, which read −0.94 and got worse with more data.**
Both are demoted in §4 with that stated, not buried. Conversely the arm that
*did* clear is the **decoder-free, latent-predictive, EMA-target** arm — which
means the JEPA-family literature that four consecutive sweeps have filed under
"pixels-only, regime objection, watchlist" is now the literature aimed at our
only measured learner. Fronts were re-weighted accordingly.

**(2) `T4.02` FAILed twice — front 2 finally has a scar.** Attempt 4
(2026-08-21, P100, 506 s): `max_modality_grad_ratio` **30.12** against the
registry's 10× gate, per-seed 28.08 / 12.30 / 30.12, every rig gate green
(`fired_ok_all` 1.0, loss fell on all seeds 0.89→0.51, the vision-dominance
plant control detected at ~12,000×). `SYSTEM.md`'s "no new organ without a
scar" has been the standing objection to every fusion nomination for three
weeks. It no longer applies. **And the scar is not the one the spec was written
for** — the registry note says *"the documented failure where vision drowns
proprioception"*, and the measured main-arm fingerprint is the opposite:

```
seed 0 boundary grad norms:  touch 2.93e-3  >  vision 9.19e-4  ≈  language 8.4e-4
                             ≈  proprio 7.92e-4  >>  audio 1.04e-4
```

**Touch drowns audio; vision sits mid-pack.** That detail decides front 2's
verdict in §5, and it is the single most useful thing this sweep found for the
builder even though it produced no nomination.

---

## 1. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | decoder-free / latent-predictive world models at STATE-VECTOR scale (the new priority); Dreamer successors; JEPA-family control; identifiability theory; scaling-vs-size | **full HTML, losses, result tables, targeted re-fetches** for both nominations + NE-Dreamer + the scale probe; abstracts for 4 siblings |
| **2 · MULTIMODAL FUSION** | modality-imbalance / gradient-modulation family (OGM-GE lineage and its 2026 successors); whether ANY of it leaves supervised classification | search-level across the family + **full abstract for PDMP (2604.05773)**; cross-checked against `UNIFIED_BRAIN_BAKEOFF.md` §1.5's existing citations |
| **3 · MEMORY** | agent memory; extractive vs generative retrieval; provenance; abstention/refusal reporting | abstracts; **Eywa (2605.30771) abstract fetched and interrogated on the generative question** |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation 2026; autotelic/lifelong curricula; exploration under mortality and homeostatic drive | two search passes, no fetch warranted — see §5 |
| **5 · WORLDS & EMBODIMENT** | homeostatic/survival sims; needs-with-death environments; cheap embodied benchmarks | **full HTML environment spec** for ForageWorld (2506.06981) |
| Biology-as-oracle | infant motor development → exploration; developmental reward schedules | **full HTML, formulas and baseline table** for the nomination (2606.16590) |
| Small-model end | sub-1M-parameter embodied control | search-level, nothing found |
| Queued #1 (TD-JEPA full text) | **DONE — and it produced a correction to our own log.** See §4 and §6 | full HTML + a second targeted re-fetch |
| Queued #2 (CIG appendix A.6) | **DONE — the number does not exist in the paper.** See §4 | full HTML, appendix located |
| Queued #3 (scale probe param axis) | **DONE — resolved and dropped.** See §3 | full HTML |
| Queued #4 (2607.22430 identifiability) | **DONE — answered, and it points somewhere unexpected.** See §3 | abstract |
| Queued #5 (conference proceedings) | **DROPPED with cause, not re-queued.** See §5 | decision, no fetch |
| Queued #6 (`SM.02`/`TA.02`/`VO.01`) | **ledger read.** `TA.02` PASS · `VO.01` **PASS at attempt 8** · `SM.02` still never run | ledger |

**Known gaps, stated so nobody assumes coverage:**
- **Three of this sweep's four nomination-grade papers report NO hardware and NO
  wall-clock, and two report no parameter count.** This is a pattern, not an
  accident, and §6 treats it as a structural finding: `B4` is a throughput gate,
  so the literature is now systematically un-pre-priceable against it.
- Koopman Dreamer's **DMC seed count is not stated anywhere in the paper** —
  the experimental section says only *"following the public protocol, we average
  the last recorded score over available seeds"*. Its headline table has no CIs.
- PSG-JEPA and Koopman Dreamer both have **unconfirmed code**; PSG-JEPA has a
  project page only.
- Front 4 got **two search passes and no fetch.** If a curiosity result landed
  that our vocabulary does not name, this sweep would not have seen it.
- No non-English sources. No conference main-track enumeration (§5, now dropped
  rather than pending).

---

## 2. NOMINATIONS

Three. All three enter on or beside **`A4` / `wm-latent`** or its exploration
path — i.e. against the *one arm this project has measured as a learner* — which
is a deliberate consequence of §0 and not a coincidence. Each states its arXiv
**primary category** (week 2's convention), what is **[V]**erified (fetched and
read), **[c]**laimed, or **[C]**omputed here, its cost on **our** substrate, and
— steelmanned both ways — why it might win and why it might lose.

**A jurisdictional note that applies to all three, stated once.** `LC.04` and
`LC.05` are BLOCKED by D10; none of these can enter an arbitration this week.
They are candidates for D10's option **(c) "judge the arms"** — the branch the
builder recorded as *"design work with no current owner of record"* — and for
the owner's scale-transfer guard that option (a) requires before adoption.
Nominating into a blocked bakeoff would be theatre; nominating into the branch
that has no owner is the useful move, and the owner decides whether that branch
runs at all.

---

### N1 — The spectral-radius constraint, extracted from Koopman Dreamer and nominated ALONE: the first world-model result this project has seen measured entirely on PROPRIOCEPTIVE state

**Source:** *Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable
World-Model Imagination* — [arXiv:2607.19719](https://arxiv.org/abs/2607.19719),
**2026-08-01**, primary category **cs.LG**. Li, Zhang, Xie, Jiang, Lan, Pan, Xu.
**Full HTML read, plus a targeted re-fetch to settle the decoder question.**

**Not cited anywhere in this repo.**

**Why this clears the objection that has killed every world-model nomination for
four weeks.** `LEARNING_CORE.md` §5.4 records A4's prior as *"unknown at 2M
parameters on a ray retina; reconstruction is a much stronger learning signal
when the observation is 96-dimensional"* — the regime argument. Every JEPA-family
nomination since week 1 has been argued from pixels and has died on it.
**All nine of Koopman Dreamer's DMC tasks are proprioceptive/state-vector
observations** [V] — low-dimensional continuous vectors, which is W0's regime
exactly.

**The verified claim [V].**

| | |
|---|---|
| DMC, **all proprioceptive** | best final score on **6/9**, exceeds DreamerV3 on **8/9** |
| Acrobot | **131.7 → 292.3** (+121.3 %) |
| Hopper Stand | 650.4 → 859.1 (+32.1 %) · Cheetah 596.3 → 692.8 (+16.2 %) |
| **Walker Run** | **655.5 → 572.0 (−12.7 %) — it LOSES the hardest locomotion task** |
| the constraint | 2×2 rotation–scaling blocks: `ρᵢ = ρ_min + (ρ_max−ρ_min)·σ(αᵢ)`, `θᵢ = π·tanh(ωᵢ)`, giving `‖A_K‖₂ = ρ(A_K) ≤ ρ_max` |
| where it acts | the **deterministic** latent transition `ϕ_{t+1} = clip(A_K ϕ_t + B_a ā_t + H_θ(ϕ_t,ā_t) + B_z z_t)` — confirmed on re-fetch to be *"independent of decoder existence — it structures only the deterministic state evolution mechanism"* |
| **the mechanism, measured** | horizon-64 latent MSE **0.0518 at learned radius 0.943 → 2.2675 at radius 1.026** (their Fig. 8). Crossing 1.0 costs 44× the rollout error |
| seeds / CIs | **NOT STATED** for Table I. 95 % CIs appear only in the UAV-LiDAR ablation |
| params / hardware / wall-clock | **none reported** |
| code | **none found** |

**What I am nominating, and what I am explicitly NOT.** The full Koopman Dreamer
stack is an `A2` derivative: its loss is
`ℒ_base = λ_o ℒ_o + λ_r ℒ_r + λ_c ℒ_c + λ_dyn ℒ_dyn + λ_rep ℒ_rep` — **including
`ℒ_o`, observation reconstruction** — plus four more terms (`ℒ_koop`, `ℒ_roll`,
`ℒ_pred`, `ℒ_opreg`), each with its own `λ` *and* its own annealing schedule
`α(n)`. That is roughly ten new hyperparameters bolted onto the arm that
measured **−0.94**. Nominating it whole would be nominating a decorated corpse.

**The nomination is the spectral parameterisation of `A_K` alone, as a variant of
`A4`.** `A4` is `A2` minus the decoder, so it carries the same deterministic
recurrent transition, and the paper's own structure says the constraint is
decoder-independent. Two hyperparameters (`ρ_min`, `ρ_max`), no new network, no
new loss term — a reparameterisation of an existing weight matrix.

**Cost on our substrate.** Near zero at training time: the block-diagonal
rotation–scaling form has *fewer* free parameters than a dense transition of the
same width, and the per-step cost is the same matrix product. **`A4`'s ≈1.37 M
stays ≈1.37 M or falls.** Against that: their DMC config is `deterministic state
2048, hidden 256` — **8× the `deter 256` in `A2`/`A4`'s RSSM** — so every number
above was measured on a model far larger than ours, and `LESSONS.md`'s
hardware-unlike-ours flag applies to the *width*, not just the silicon.

**Why it might WIN (falsifiable).** `A4` carries a mandatory collapse
diagnostic — effective rank and per-dimension latent variance every 1,000
decisions, with a rank floor that VOIDs the arm — because collapse is its named
silent failure. **A bounded spectral radius is a constraint on the opposite
failure**, divergence, and the two interact: a transition whose spectrum sits
just under 1 neither collapses to a point nor blows up over an imagined horizon.
`A4` won LC.03 while *imagining* rollouts in a survival world where a wrong
imagined future is a death. If `A4`+spectral beats `A4` at ≥1.5σ over 3 seeds on
`life_gain` while both clear the learning gate, then the winning arm's imagination
was leaking accuracy at horizon and the fix was two hyperparameters. If they tie,
we learn that W0's horizons are too short for spectral drift to matter — cheap,
and it lets `A4` keep the plain transition with a reason instead of by default.

**Why it might LOSE (steelmanned). Five.**
1. **No seed count and no CIs on the headline table.** `UNIFIED_BRAIN_BAKEOFF.md`
   §1.8 adopts Agarwal et al.'s standard — with few runs, point estimates are
   unreliable and only interval estimates count. An 8/9 win rate assembled from
   "available seeds" under a public protocol is exactly the statistic that
   standard exists to distrust. **This is the strongest objection.**
2. **It loses Walker Run by 12.7 % — the task closest to a body that must move.**
   Its wins concentrate on swing-up and stand tasks. Jack's binding open problem
   (`T2.01`, `D1`, `D9`) is locomotion.
3. **The paper is an `A2` paper.** Everything measured was measured *with*
   reconstruction present, at 8× our recurrent width. Porting one term of it to
   `A4` at `deter 256` is our inference, not their result — and this project has
   been burned three times by transfers that looked this safe.
4. **`B2` counts hyperparameters and this adds two** (`ρ_min`, `ρ_max`), with the
   paper's own Fig. 8 showing the outcome is *sharply* sensitive to where the
   radius lands. A knob that matters that much is a knob that needs tuning we
   cannot afford.
5. **No code, no params, no hardware, no wall-clock.** Nothing here can be
   pre-priced against `B4`'s 5.0 sim-s/real-s floor before it is written.

---

### N2 — PSG-JEPA: predict the body's future from the latent, and our own LC.03 numbers predict which half will work

**Source:** *Is Forward Prediction Enough? Physical State Grounding for JEPA
World Models* — [arXiv:2608.06799](https://arxiv.org/abs/2608.06799),
**2026-08-07**, primary category **cs.RO**. Yan, Zhu, Jia, Yin, He, Zhong, Li,
Lu, Li, Zhang, Chen, Song, Chen, Gao, Li (HKUST-GZ et al.).
**Full HTML read: losses, tables, ablations.**

**Not cited anywhere in this repo.**

**The method.** Two auxiliary losses on a latent-predictive world model, applied
**only during training**:

```
ℒ_PSG = ℒ_JEPA + λ_g · ( ℒ_static + ℒ_dynamic )          λ_g = 0.1, ONE shared hyperparameter

ℒ_static  = (1/T) Σ MSE( H_s(z_{t+i}),  s_{t+i} )         individual latent → proprioceptive state
ℒ_dynamic = (1/|𝒦|) Σ_k (1/|ℐ_k|) Σ_i MSE( Δq̂_{t+i,k}, Δq_{t+i,k} )
                                                           latent PAIRS → multi-horizon joint-angle CHANGE
```

*"Grounding heads are discarded after training, so PSG-JEPA introduces no
inference-time overhead."* [V]

**The verified claim [V].**

| | |
|---|---|
| identifiability (probing) | end-effector yaw **Pearson r = 0.98** (MLP probe) vs LeWM's **r = 0.08** |
| planning, OGBench-Cube | **95.0 ± 0.7 %** vs LeWM **80.7 ± 1.9 %** (full data, 5 epochs) |
| policy, LIBERO-Goal | **85.3 ± 3.9 %** vs LeWM **77.7 ± 0.5 %** |
| real robot (dual-arm Cobot) | **79.3 % vs 60.0 %** mean over three tasks, 50 trials/task |
| seeds | **3** (planner seeds; policy seeds), ±std reported |
| inputs | **pixels/RGB only.** Proprioception is used *"only as grounding targets during training, never as encoder inputs"* |
| params / hardware / wall-clock | **none reported** |
| code | **not stated**; project page `haodong-yan.github.io/psg-jepa-project-page/` |

**Which spec it enters — and the fork is the interesting part.** `LEARNING_CORE.md`
§5.4 `A4` as a variant, or two variants. And here the two halves of the loss are
in *opposite* positions relative to our own evidence, which makes this the rare
nomination that arrives with a prediction we can pre-register:

- **`ℒ_static` is partial reconstruction in our regime, and LC.03 says
  reconstruction loses.** In PSG-JEPA the encoder sees only pixels, so predicting
  proprioception from the latent is genuine grounding. In W0 the observation dict
  **already contains** proprio, so `ℒ_static` is a decoder on a slice of the
  input — which is the thing `A4` deleted, and `A2`, which keeps the full version,
  measured −0.94 and *fell further* with more data. **Prediction: `ℒ_static`
  alone should not help, and may hurt.**
- **`ℒ_dynamic` is not reconstruction and survives the objection.** Multi-horizon
  *change* in joint angles across latent PAIRS is a temporal quantity that appears
  nowhere in the observation at any single step. It is a constraint on what the
  transition must have encoded, not a copy of the input.

That asymmetry means the two terms **are each other's control**: run `A4`+dynamic
against `A4`+static against plain `A4`, and whichever way it lands, the run
distinguishes "grounding helps" from "any auxiliary reconstruction helps",
which is precisely the ambiguity a single combined arm would leave open.

**Cost on our substrate.** One MLP head per loss, discarded after training. **Zero
inference-time parameters, zero inference-time compute** — the arm's `Arm.cost`
under §5.5's declared unit (trainable params in the learning core) rises only by
the heads during training. One shared hyperparameter (`λ_g = 0.1`), against N1's
two and Optimistic World Models' two. This is the cheapest of the three.

**Why it might WIN (falsifiable).** Two independent reasons, and the second is
the one I would bet on.
1. `A4`'s named silent failure is latent collapse. A latent forced to predict a
   6-DOF physical quantity at multiple horizons **cannot be collapsed** — the
   grounding loss is an anti-collapse mechanism with a *measurable* readout
   (their probing r = 0.98 vs 0.08) rather than a regulariser with a hyperparameter.
   If `A4`+dynamic raises the effective-rank diagnostic `A4` already reports
   *and* raises `life_gain`, that is one mechanism explaining two numbers.
2. **It converges with week 1's N1 from a different direction.** Week 1 nominated
   arXiv:2607.27017 on the finding that *stiffness enters the latent only when
   touch is a prediction TARGET (R² 0.50), not an INPUT (−0.02)*. PSG-JEPA is the
   same shape for proprioception: r = 0.98 as a target, 0.08 in a model that has
   it only as context. **Two 2026 papers, different labs, different modalities,
   different benchmarks, same structural claim** — a sense is only in the latent
   if the loss asks for it. That is a claim about `UNIFIED_BRAIN_BAKEOFF.md`'s
   binding objective, not just about `A4`, and it is now doubly sourced.

**Why it might LOSE (steelmanned). Five.**
1. **Pixels, again.** Every number is from an RGB encoder. The whole argument for
   grounding is *"the pixel encoder does not know where the arm is"* — and W0's
   encoder is handed the joint angles. The nomination survives only via
   `ℒ_dynamic`, and `ℒ_dynamic` is the half with no isolated ablation in the paper.
2. **Its baseline is LeWM, not DreamerV3 or a reconstruction model.** Beating one
   latent world model by 14 points says little about beating *ours*, and nothing
   about the reconstruction-vs-latent axis LC.03 actually measured.
3. **Three seeds, ±std, no CIs, no paired comparison.** Better than N1 (which
   states no seed count) and below §1.8's bar.
4. **Jack's "joints" are a rover body under D9 review.** `ℒ_dynamic`'s target is
   multi-horizon joint-angle change; three independent measurements
   (`T2.01` 2.67σ, `D8`'s no-catch-authority, `W0.BAL`'s `upright_cos` −0.041)
   say the body is the binding constraint. **Grounding a latent in the kinematics
   of a body that falls over may be grounding it in noise** — and unlike the other
   objections, this one gets *worse* if the owner adopts W0.BAL arm C, because the
   joint set changes underneath the loss.
5. **No params, no hardware, no wall-clock, no confirmed code.**

---

### N3 — Infant motor noise: temporally-correlated exploration on a developmental schedule, from the shelf GOAL.md explicitly named

**Source:** *Infant Spontaneous Movement Noise Improves Exploration in Deep RL* —
[arXiv:2606.16590](https://arxiv.org/abs/2606.16590), **2026-06-15** (rev 06-16),
primary category **cs.LG**. López, Ernst, Cruz, Hoffmann, Triesch (FIAS /
Goethe-Frankfurt / UNSW / CTU). **Full HTML read: formula, baseline table.**
**Code released:** `https://github.com/trieschlab/baby-noise-rl`.

**Why it is here.** `GOAL.md`'s biology-as-oracle section lists what is *"still
unmined and on the shelf: **motor babbling**, innate reflex priors, pain as a
fast signal distinct from reward, critical periods, play as safe rehearsal."*
This is a measured motor-babbling result with released code, and it arrives in
the same fortnight that LC.03 found the two PPO arms unable to learn
(`ppo-needs` 1.06, `ppo-lp` 1.20) and `SH.01`'s pilots concluded that *"perception
is not the limit… execution is not the limit… the acquisition rate of the reactive
core is."* Exploration noise is an acquisition-rate intervention costing zero
parameters.

**The method.** Infants' end-effector velocities are **colored** noise, not white,
with a spectral exponent that *rises* developmentally. The agent's exploration
noise is generated by spectral shaping (FFT, PSD `S(f) ∝ f^(−β)`, block length
10,000) with β annealed over training:

```
β(t) = 0.7 + 0.2 · (t / T)            β = 0 white · β = 1 pink · β = 2 red
```

**The verified claim [V] — and it is a modest one, quoted rather than paraphrased.**

| | |
|---|---|
| algorithms | **TD3 and SAC** |
| environments | **12**: MountainCarContinuous, Pendulum, InvertedPendulum, InvertedDoublePendulum, **Hopper, Swimmer, HalfCheetah, Ant**, PointMaze ×4 |
| baselines | white (β=0), blush (0.5), rose (0.75), **pink (1.0)**, red (2.0), **OU** — i.e. the obvious competitors are all run |
| headline | highest normalised AUC; *"the only one with a win rate significantly above chance level 56 % (p<0.05)"* vs white |
| honest caveat, theirs | *"pink noise and other colored baselines show comparable or better performance on specific environments"* (their Table I) |
| seeds | **10 per {algorithm, environment, noise} triplet** — the best seed discipline in this sweep |
| hyperparameters | **two** (β₀ = 0.7, slope 0.2), *"derived from empirical infant data rather than requiring additional tuning"* |
| hardware | UNSW Compute Cluster Katana; **no wall-clock, no parameter counts** |

**Which spec it enters — and the misfit, stated up front the way week 3's CIG
misfit was.** The clean home is an **exploration-noise arm on an off-policy core**,
because TD3/SAC add noise to a deterministic or reparameterised action and
temporal correlation is a free change there. **`A0`/`A1` are PPO**, whose
exploration *is* the policy distribution: injecting autocorrelated noise into an
on-policy actor breaks the likelihood ratio the update is built on unless the
noise process is folded into the policy itself, which is a design decision with
its own cost. **`CURIOSITY_BAKEOFF.md`'s model-free arms have the same problem.**
The honest statement is that this nomination needs a home decision the paper does
not make for us, and I am not making it either. What it can do cheaply and
immediately is serve as a **W0 diagnostic**: run the null (`random`) and
`random-repeat` — which LC.03 already defines and which is literally a crude
colored-noise process — against a β-scheduled random policy, and read whether
temporally-structured *random* action changes `life_gain` in W0 at all. That is
a CPU-minutes measurement about the *world*, and it bears directly on D10's
option (b).

**Cost on our substrate.** Trivial. An FFT over a 10,000-sample block per
episode, amortised to nothing; **zero parameters**; two constants taken from
infant data rather than tuned. Runs on 4 ARM cores.

**Why it might WIN (falsifiable).** W0's food is sparse and the body must travel;
white action noise integrates to a random walk that stays near the spawn point,
while β≈0.9 noise produces sustained directed excursions. If a β-scheduled null
policy beats plain `random` on `life_gain` in W0, then part of what LC.03 measured
as "arms cannot learn" is "the exploration process never reaches the food" — a
finding about the world, which D10's fork (b) is exactly the branch for, and one
that costs CPU-minutes rather than 190 core-hours.

**Why it might LOSE (steelmanned). Four.**
1. **56 % is a weak effect and the paper says so.** "Significantly above chance"
   at 56 % win rate against *white noise* — and their own table concedes that
   plain **pink noise** matches or beats it on specific environments. The
   developmental *schedule* may be buying nothing over a fixed β = 1, and pink
   noise for RL exploration is prior art, not a 2026 result. **The novel part of
   this paper may be the part that does not pay.**
2. **On-policy misfit**, above, unresolved.
3. **Nothing here has needs, death, or more than one sense.** MuJoCo locomotion
   with a dense task reward is the regime; W0 is homeostatic with `r_h` and a
   600 s basal ceiling.
4. **It is an intervention on the arms that LC.03 could not distinguish from
   noise.** Improving `ppo-needs` from t = 1.06 to t = 1.4 changes nothing about
   D10; only crossing 3σ would, and no exploration-noise paper claims a jump of
   that size.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**Resolved this sweep and now DELETED from the watchlist** (recorded here so a
fifth sweep does not re-open them):

| item | resolution |
|---|---|
| **TD-JEPA** (2607.25337) | **Fetched, and it DEMOTES.** Primary category is **cs.CL** for a latent-MPC control paper. 15 M ViT-tiny, patch 14, image 224 — **pixels only**, 10 seeds on primary tables, code released (`HKBU-KnowComp/Temporal-Distance-JEPA`), Two-Room 100.0 ± 0.0, OGB-Cube 82.2 ± 2.9. **No hardware, no wall-clock, no GPU mentioned anywhere.** And a correction we owe our own log — see §6. Not an `A4` arm: 15 M pixels is 11× `A4` in a regime we do not have. |
| **Scale probe** (2605.08578, cs.LG) | **Fetched — the parameter axis is `L ∈ {2,4,8,12,24,48,96}` at embedding 512, "approximately 6 million to 300 million parameters".** 8 seeds per environment, 16 trials for policy learning, **no hardware stated**. It does corroborate the 54K-beat-57M lesson with a real number — *"for certain tasks, such as Amidar, larger models generalize worse, with L=8 outperforming L=96"* — but its **smallest model is 6 M, above `A2`'s 1.9 M and `A4`'s 1.37 M.** It never enters our regime. Dropped: corroboration is not news. |
| **Var-JEPA** (2603.20111, cs.LG) | Dropped. Tabular-only, no dynamics, no RL — and there are now four better-evidenced anti-collapse routes for `A4` (week 1's SMWM and SIGReg/LeJEPA, PSG-JEPA's grounding, NE-Dreamer's Barlow-Twins-on-prediction). It has been superseded on every axis. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **On the Identifiability of Controlled World Models** ([arXiv:2607.22430](https://arxiv.org/abs/2607.22430), 2026-07-24) | cs.LG | **Queued #4 — ANSWERED, and it points somewhere I did not expect.** Zhang, Guan, Zhang, Li, Zhang, Li. It proves that **minimising a LeJEPA-style objective recovers latent states and dynamics up to orthogonal transformation**, under a joint condition (representation identifiability via a *spectral separation* property + transition identifiability via non-degenerate conditional action variation), bounding transition error inversely to the spectral margin. Theory + experiments in four nonlinear observation settings. **It does NOT bear on week 1's N1 the way I expected** — no post-hoc computable certificate, so it supplies nothing to `UB.11`'s missing positive control. **What it does do is give week 1's `A4c` (SIGReg/LeJEPA) a theoretical guarantee for the ACTION-CONDITIONED case** — which is `A4`'s case, and `A4` is the only arm that learned. It also makes N1's spectral quantity and this theory's spectral margin the same family of object, which is either a convergence or a coincidence and I cannot tell which from the abstract. **Promote on:** a full-text read establishing whether the spectral separation margin is computable on a trained model. That is now the highest-value fetch outstanding. |
| **Simulus** ([arXiv:2502.11537](https://arxiv.org/abs/2502.11537)) | cs.LG | Oldest open item, and it gained relevance rather than losing it: its benchmark set includes **DMC Proprioception 500K** — our regime — which four sweeps of notes never recorded. Still blocked on **exactly the same two numbers: a parameter count and a per-step wall-clock**, which `B4` needs and the paper does not report. Its prioritised-replay component remains separately nominated (week 2 N2 → `NE.05`, never run). |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, and now *more* relevant: it measures **rollout-horizon deterioration in the fully observable state space** across RSSM / Transformer / Diffusion / Neural-ODE — which is the exact quantity N1's spectral constraint claims to fix, in our exact regime. Still blocked on: no compute cost, no hardware, no environment size, no code statement. **Promote on: any statement that a domain runs on CPU.** Dated 2025-11-28, outside the window, kept on relevance. |
| **Survival RL** ([arXiv:2605.31273](https://arxiv.org/abs/2605.31273)) | cs.LG | Unchanged disambiguation — "survival" = dwell time at goals, not homeostatic needs. Kept so a fifth sweep does not chase the title. |

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **NE-Dreamer** ([arXiv:2603.02765](https://arxiv.org/abs/2603.02765), 2026-03-03) | cs.LG | Bredis, Balagansky, Gavrilov, Rakhimov (T-Tech). Decoder-free MBRL: a causal temporal transformer predicts the next *encoder embedding*, aligned by a **Barlow Twins redundancy-reduction** loss on the prediction target — a fifth anti-collapse route for `A4`, and the only one that puts the anti-collapse term on the *predicted future* rather than the *current* representation. 12 M params, **5 seeds**, all methods at matched budget. Substantially beats DreamerV3, R2-Dreamer and DreamerPro on DMLab memory/navigation; on DMC it only *"matches or slightly exceeds"*. | **The DMC numbers.** [V] on structure, **[c] on magnitude** — the paper shows learning curves and states no per-task DMC values I could extract. Both benchmarks are 64×64 RGB; no hardware, no wall-clock, no code. A state-vector result or an extractable DMC table would make it an `A4` arm; as it stands its win is on a memory task in pixels. |
| **PDMP** ([arXiv:2604.05773](https://arxiv.org/abs/2604.05773), 2026-04-07) | cs.CV | Wei, Luo, Zhu, Luo. *Rethinking Balanced Multimodal Learning via Performance-Dominant Modality Prioritization.* Argues the field's premise is wrong: **imbalanced learning that prioritises the strongest unimodal modality beats balancing it away.** Rank modalities by isolated performance, then modulate gradients with *asymmetric* coefficients favouring the dominant one. | Nothing, for adoption — see §5, it is the wrong objective for us. It is on the watchlist as **evidence about the field's confidence**, not as an arm: the balancing literature `UNIFIED_BRAIN_BAKEOFF.md` §1.5 cites is now internally contested at the level of its goal. Promote only if someone runs it on a control task. No seeds, hardware, or code stated. |
| **Eywa** ([arXiv:2605.30771](https://arxiv.org/abs/2605.30771), 2026-05-29) | cs.CL | Resham Joshi. The **first front-3 item in three sweeps that is not generative at read time**: *"stores immutable source evidence before deriving canonical facts… retrieves bounded memory context through a deterministic multi-route read path with **zero LLM calls inside retrieval**."* LoCoMo C1–C4 90.19 %, LongMemEval-S 88.2 %, BEAM 81.45 %. | **Whether the WRITE path is generative.** *"Deriving canonical facts"* and *"validates extracted memories"* read as LLM-extracted prose, which `MEMORY_RETRIEVAL_BAKEOFF.md` §5.1 forbids as surely as generative reading. Also: **no hardware, no latency** (so it cannot inform §1.9's CPU table), single author, no code, and metrics are LLM-as-judge. A search snippet claimed it reports refusal rates; **the abstract does not, and I am recording the discrepancy rather than the flattering half.** |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981), 2025-06-08, rev 2025-11-30) | cs.AI | Simmons-Edler, Badman, Berg, Chua, Vastola, Lunger, Qian, Rajan (Kempner/Harvard; NeurIPS 2025). **Not an arm — a design reference for D10's fork (b).** 96×96 procedurally generated arena, 9×11 egocentric view, **health/hunger/thirst/fatigue**, death at health 0, **food that diffuses from spawn points and depletes**, fixed unlimited water, **predators that pursue when in view**, and **sleep that is gated on energy < 50 % and immobilises the agent while restoring energy**. PPO-RNN, 512-unit GRU. Code: `github.com/RileySE/Craftax-Foraging`. | Nothing — it should not be adopted. It is **Craftax-based and GPU-accelerated**, the axis `SURVIVAL_WORLD.md` §2.2 already ruled out, and a discrete gridworld where W0 is a MuJoCo body. It is on the watchlist as **the closest published existence proof of a world with the discriminating features D10(b) would add** — depletion, pursuit, and a sleep action that trades safety for recovery — together with the finding that model-free RNN agents show *"structured, planning-like behavior purely through emergent dynamics"* there, which bears on `DP.00`. Outside the window; kept on relevance. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS — including the two this sweep's evidence demotes

No prior nomination has been run. Rather than re-litigate seven live items, this
table states where each stands against what the ledger now says. **Only the owner
and builder retire a nomination; the "demoted" rows below are my assessment with
its reason, not a withdrawal.**

| nomination | entered | status now |
|---|---|---|
| wk1 · certificate-gated identifiability (2607.27017) → `UB.11` pre-gate | UB.11 | **LIVE, unrun.** `UB.11` has never run; `UB.10`, which it depends on, is parked pending arm redesign. 2607.22430 (§3) does **not** supply the certificate it needs. |
| wk1 · anti-collapse regularisers → `A4b` (SMWM 2606.20104) / `A4c` (SIGReg/LeJEPA 2511.08544) | `A4` variants | **LIVE and PROMOTED by LC.03.** These were nominated onto `A4` when `A4` was a conditional arm with an unknown prior. `A4` is now the project's only measured learner, and `A4c` additionally gained an identifiability theorem for the action-conditioned case (§3). Of everything on this desk, these two have moved up the most and neither needed a new paper to do it. |
| wk1 · interoceptive precision allocation (2608.04232) → `NEEDS_AND_DEATH` §2.4b | NE | **LIVE, unrun.** Still the cheapest nomination on the desk (runs on 4 ARM cores, code released, 2.08× survival over the uniform-precision baseline that IS our current design). |
| wk1 · entity-collision protocol (2605.29630) → `MEMORY_RETRIEVAL` §2 eval design | MR | **LIVE, unrun.** |
| wk2 · the whiff clock → `SM.02` | SM.02 | **LIVE, unrun — and `SM.02` has still never run** (ledger checked; `SM.01` PASS, `SM.02` absent). Third consecutive sweep reporting this. |
| wk2 · RPE-prioritised replay → `NE.05` S1 | NE.05 | **LIVE, unrun.** |
| **wk3 · CIG (2605.20878) → `A3`'s epistemic term** | `A3` | **DEMOTED, on our own measurement.** Its home arm `wm-efe` read **t = 2.05 / 2.07**, below the 3σ gate — CIG refines the epistemic term of an arm that did not demonstrate learning. It also loses its cheapest selling point: **queued #2 is now closed with a null result** — appendix A.6 contains only the words *"both costs are negligible relative to the 𝒪(TMd) ensemble forward passes… (T=15)"*, with **no GPU type, no wall-clock, no measured overhead, and no code section reachable.** The "negligible" was never a measurement. It stays on the desk as a candidate under D10(c) if `A3` is ever repaired, and nowhere else. |
| **wk3 · Optimistic World Models (2602.10044) → `A2` variant** | `A2` | **DEMOTED, hard.** Its home arm `dreamer-xs` read **t = −0.94**, and went from +46 s to −48.5 s when the envelope quadrupled — the arm got *worse* with more data. A +20 % wall-clock tax and two new hyperparameters on that arm is not a nomination. Week 3's lead objection was constitutional (an optimistic dynamics corrupts the representation the UB gates measure); the empirical objection is now stronger than the constitutional one. |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

Stated plainly. An empty front honestly reported beats a padded one.

**FRONT 2 · MULTIMODAL FUSION — no nomination, and this is the sweep's most
considered refusal, because for the first time there IS a scar and I still say no.**

`T4.02` gives front 2 exactly what `SYSTEM.md` demanded: a named, measured,
twice-reproduced failure (30.12 vs the 10× gate). The literature that names this
disease is the **modality-imbalance / gradient-modulation** family — OGM-GE
(2203.15332) and its successors MMPareto, AGM (Shapley-based), PMR, IGDM, G2D,
CGGM — and `UNIFIED_BRAIN_BAKEOFF.md` §1.5 already cites its survey
(BalanceBenchmark, 2502.10816). **Three checkable reasons it does not enter:**

1. **Installing a balancer to pass `T4.02` would Goodhart the gate, and the gate
   is the point.** `T4.02` is a *rule* spec: it asks whether the fusion Jack
   actually has is balanced. A method whose mechanism is "rescale the gradients
   until they are balanced" makes that metric read 1.0 while proving nothing
   about whether every sense is load-bearing — which is what `GOAL.md` wants and
   what `T4.03`/`UB.11` measure. If any of this family ever enters, it must enter
   **`UB.10` as an arm judged on the HNS binding metric, with
   `max_modality_grad_ratio` as a reported secondary** — never as a `T4.02` fix.
   I am flagging this so the builder does not discover it after implementing.
2. **The entire family is supervised classification, and the transfer argument is
   weaker than it looks.** Every method above optimises accuracy on audio-visual
   event/sentiment datasets, where the dominant modality is *always vision* and
   the objective is a label. **Our measurement says touch (2.93e-3) drowns audio
   (1.04e-4) with vision (9.19e-4) mid-pack** — the opposite ordering, on a
   world-model objective, with five heterogeneous senses of wildly different
   dimensionality. Not one paper in this family has been run outside
   classification.
3. **The field is now split on whether balance is even the goal.** PDMP
   (2604.05773, §3) argues the premise is backwards and that prioritising the
   *dominant* modality wins. That does **not** undercut `T4.02` for us — PDMP
   optimises accuracy, and Jack's requirement that every sense be load-bearing is
   constitutional rather than performance-derived — but it does mean the
   balancing literature cannot be imported on authority. Its own community is
   arguing about its objective.

**FRONT 3 · MEMORY — nothing, for the third consecutive sweep, and the
constitutional reason has not moved.** The 2026 agent-memory literature that
surfaced (REMem 2602.13530, Memanto 2604.22085, MaRS, and the two survey papers
2602.06052 / 2603.07670) is generative recall end to end — LLM-extracted memories,
agentic retrievers, LLM-as-judge scoring. `MEMORY_RETRIEVAL_BAKEOFF.md` §5.1 makes
that structurally inadmissible on purpose. **Eywa (§3) is the first item in three
sweeps that even partially escapes** — zero LLM calls *inside retrieval* — and it
is on the watchlist rather than nominated because its write path derives
"canonical facts" from source evidence, which is generation moved one step
upstream, and because it reports no hardware and no latency at all.

**FRONT 4 · CURIOSITY & OPEN-ENDEDNESS — nothing.** Two search passes. What
surfaced in-window was either already ours (2608.04232, week 1's nomination),
generic max-entropy exploration (2603.18965) that adds nothing to
`disagree`/`lp`/`metra`/`vlm-lp`, or LLM-agent autotelic work (MAGELLAN and
descendants) operating in language goal spaces with no body. Nothing here would
add an arm to `CURIOSITY_BAKEOFF.md` §3.1. Front 4 was also the shallowest front
this sweep — two searches, no fetch — and §1 says so.

**FRONT 5 · WORLDS & EMBODIMENT — nothing that enters the fidelity ladder.** The
one genuinely relevant find, ForageWorld, is on the watchlist as a **design
reference for D10's fork (b)** and explicitly not as an arm: it is Craftax-based
and GPU-accelerated (the axis `SURVIVAL_WORLD.md` §2.2 ruled out) and a discrete
gridworld where W0 is a MuJoCo body. The 2026 embodied-benchmark output that
surfaced is overwhelmingly LLM-agent safety and governance benchmarks
(EmbodiedGovBench, AGENTSAFE, SafeAgentBench, RescueBench) — a different field
sharing our vocabulary, which is week 2's discipline finding still earning its
keep.

**SMALL-MODEL END — nothing.** No sub-1M-parameter embodied-control result
surfaced in-window. The nearest hits were 70–500 M language-model agents. The
scale probe (§3) does supply a real *"larger models generalize worse"* datapoint
but bottoms out at 6 M.

**BIOLOGY-AS-ORACLE — one item, and it is nominated** (N3) rather than sitting
here. The other in-window candidate, *Evolutionary Discovery of Developmental
Reward Schedules in Deep RL* (2606.20858), is not nominated: evolutionary search
over reward schedules is unaffordable on free compute, and a *discovered*
developmental schedule is scaffolding, which `PURPOSE_AND_SCAFFOLDING.md` treats
as a live question rather than a settled design.

**CONFERENCE PROCEEDINGS — DROPPED, not re-queued.** Week 3 established the
OpenReview API returns `HTTP 403 ChallengeRequiredError` and `/venues` returns
empty, and set the rule: *"either re-plan it as targeted per-venue searches, or
drop the item with the reason recorded. A third 'still pending' would be a lie by
deferral."* **I am dropping it.** The reason is not only the 403: per-venue
targeted search is what this sweep already does on every front, so re-planning it
that way would be renaming existing work rather than adding coverage. What
enumeration would have bought — catching a strong paper whose vocabulary does not
match our search terms — is real, and is now an acknowledged permanent gap rather
than a pending task. **It will not appear in a future queue.**

---

## 6. A DISCIPLINE FINDING — the second consecutive week the unverified claim was
## mine, and this time it is sitting in our permanent log

Week 1: *an abstract is a claim about a table.* Week 2: *a title is a claim about
a field.* Week 3: *a diagnosis of our own failure must carry the arithmetic, not
the literature.* This week's is narrower and more embarrassing, because the error
is already committed to a file that is append-only.

**`FIELD_WATCH_LOG.md`'s 2026-08-12 entries and week 3's §3 attribute to
arXiv:2607.25337 the claims that it "plans 48× faster than world models on frozen
foundation encoders" and runs "end-to-end from raw pixels on a single GPU in
hours". I fetched the full text twice. Neither claim is in the paper.** The
second fetch asked specifically for any sentence mentioning speed, efficiency,
planning time, a multiplier, a GPU, an A100, an RTX, or training hours, and
returned **NONE FOUND** for every category; the only compute statement in the
paper is *"10 epochs… AdamW with learning rate 5×10⁻⁵ and weight decay 10⁻³, bf16
mixed precision"* (their Appendix C). The OGB-Cube figure was also carried as
"+14.2 points over LeWM"; the table reads **82.2 ± 2.9**.

Week 3 marked that entry **[c — search-level, abstract not fetched]** and named
the fetch as queued #1, so the process worked exactly as designed and the flag
was honest. **The failure is the attribution, not the confidence level.** The
numbers were not the authors' claims *at any confidence*; they were a search
engine's paraphrase of a paper it had summarised, and a paraphrase can invent a
number that appears nowhere in the source. Marked `[c]`, they read as "the
authors claim this and I have not checked the table" — which is week 1's failure
mode — when the true state was "a third party asserts the authors claim this".

**The rule this suggests, offered for `LESSONS.md` and not written by me:
a claim taken from a search result is the SEARCH ENGINE's claim, not the paper's,
and must be attributed to the snippet until the paper is opened.** The cheap
mechanical form is a distinct marker — `[s]` for search-level, kept separate from
`[c]` for claimed-by-the-authors-but-unverified — so that a reader of the log can
tell "unchecked" from "possibly nobody said this". Week 2's primary-category
convention was adopted at zero cost in this file's §3; this one costs one
character.

**A second, structural finding, offered as an observation rather than a rule.**
Of this sweep's four nomination-grade papers, **three report no hardware and no
wall-clock, and two report no parameter count** (N1: none of the three; N2: none
of the three; N3: no wall-clock, no params; TD-JEPA: params only). `B4` is a
throughput gate that has already killed one design, and `LESSONS.md`'s standing
instruction is to flag any nomination whose numbers come from hardware unlike
ours. **We increasingly cannot tell what hardware the numbers come from at all.**
That is not a reason to nominate less; it is a reason that every nomination in
this file now carries "must be re-measured here before it is believed" as a
structural property rather than a caveat.

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware. Nothing in this file is evidence about
  Jack. The LC.03 and T4.02 numbers in §0 are ours and are quoted from the
  ledger; they are context for the sweep, not findings of it.
- **No nomination is a recommendation to adopt.** `SYSTEM.md` law 3 stands.
- **Nothing here changes a spec, a threshold, a decision, or a line of code** —
  including §5's statement about `T4.02`, which describes how a *future* arm
  should be scored and moves nothing.
- **All three nominations enter a bakeoff that is currently BLOCKED.** D10 blocks
  `LC.04`/`LC.05`. They are candidates for D10's option (c), the branch with no
  owner of record. **If the owner takes option (a) or (b), N1 and N2 wait; if the
  owner takes (b), N3's W0-diagnostic form becomes the immediately useful one.**
- **Verification is uneven and marked as such:** N1 full HTML + a targeted
  re-fetch that changed the nomination's shape (decoder question), **seed count
  absent, no CIs on the headline table, no params, no hardware, no code**; N2
  full HTML with losses and tables, **no params, no hardware, code unconfirmed**;
  N3 full HTML with the formula and the full baseline table, **code released, 10
  seeds, no wall-clock**; NE-Dreamer full HTML but **DMC magnitudes not
  extractable**; 2607.22430 and Eywa **abstract-level**; ForageWorld full HTML on
  the environment, **training scale not extractable**.
- **Two of my own live nominations are demoted in §4 on our own measurements**,
  and one of my logged numbers is retracted in §6. Neither is a withdrawal — the
  owner and builder retire nominations, not the scout.
- **Front 4 was swept less deeply than fronts 1, 2, 3 and 5**, and §1 says so.

---

## 8. Queued for next sweep (**not before ~2026-08-31**)

1. **2607.22430 full text** — is the *spectral separation margin* computable on a
   trained model? If yes, it is a diagnostic for `A4` in the same family as the
   effective-rank floor `A4` already reports, it connects to N1's ρ_max, and it
   may be the certificate `UB.11` has been missing since week 1. Highest-value
   fetch outstanding on any front.
2. **NE-Dreamer's DMC table** ([arXiv:2603.02765](https://arxiv.org/abs/2603.02765))
   — the fifth anti-collapse route for `A4`, blocked on magnitudes I could not
   extract from the figures.
3. **Dreamer-CDP** ([arXiv:2603.07083](https://arxiv.org/abs/2603.07083)) — a
   sixth decoder-free route, found at search level and not opened this sweep. One
   fetch decides whether the `A4`-neighbourhood watchlist is saturated.
4. **Front 4 properly.** It got two searches and no fetch this sweep, on a front
   that owns `LT.02`–`LT.04` and `PG.4`. Next sweep it goes first, not last.
5. **Watch `SM.02` (never run, third sweep), `NE.05`, `UB.10`/`UB.11`, and D10.**
   If the owner answers D10, this file's §2 needs re-pointing the same week:
   option (a) sends N1/N2 to the scale-transfer guard, option (b) promotes N3's
   diagnostic form and ForageWorld's design list, option (c) is the branch all
   three were nominated into.
6. **NOT queued, deliberately:** conference proceedings (dropped, §5), CIG's
   wall-clock (closed with a null result, §4), the scale probe's parameter axis
   (resolved, §3), TD-JEPA (resolved and demoted, §3).
