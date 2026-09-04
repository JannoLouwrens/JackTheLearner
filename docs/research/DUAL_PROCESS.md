# DUAL_PROCESS.md — fast and slow, in one brain: the research the specs were owed

> Written 2026-09-04 (builder iteration, 05:xx UTC) to discharge the
> INTEGRATION_QUEUE entry "OWNER DIRECTIVE, 2026-08-10 — FAST AND SLOW",
> which registered DP.00–DP.03 immediately and left the literature sweep
> outstanding for 25 days. Four research agents ran in parallel on the four
> questions the queue entry pre-registered; their findings are §2–§5.
> Citations marked [verified-web] were checked against live sources this
> session; [from-memory] claims are from model knowledge and carry the
> canonical identifiers so a later pass can verify them.

**The owner's words, constitutional:** *"but Will he have fast and slow brain
like a human?"* → *"all I know is we must figure that out right and it must
still be connected but slightly different purposes? it must be in the research
and tests"*

**The bet this project registered before reading the literature:** one shared
substrate operating at two depths, where habit is practice-compressed
deliberation — not two towers. GOAL.md carries it as a named section with
three axes (acting / learning / specialised one-shot) that must not be
conflated. This document is the literature contact for the ACTING axis only.

---

## §1 — Where the family actually stands on the ledger (2026-09-04)

The research below must be read against what has already been measured,
because two of the four registered specs are already answered-in-part and the
other two are unreachable:

| spec | status | what the row says |
|---|---|---|
| DP.00 | **PASS** (12×12 gridworld) | oracle planning beats reactive: plan_h8 197.5 vs reactive-side probe, gap_clear 1.0 all seeds. VENUE-RESTRICTED by the Review 08-31: *not evidence about W0*. |
| DP.05 | **FAIL** (W0, 2026-08-24) | lookahead buys 13.3±18.8 s, σ 0.70 vs a 3.0 gate; H10 pays LESS than H4; every VOID gate green — the world measured, not the rig. Pre-registered routing fired: BO.01 does not run until W0 has traps, delays, irreversibility. |
| DP.01 | registered, unreachable | depends_on LC.04, which is unsatisfiable as written (LC.03 v2 returned one learner; D10's default seated wm-latent BY VERDICT; LC.04 runs only if a redesigned screen yields ≥2 learners). |
| DP.02, DP.03 | registered, unreachable | behind DP.01. Both sit in `GOAL_UNRUNNABLE_BASELINE` — GOAL.md cites them and they cannot run today. |
| DP.04 | PILOT-BLOCKED | mean censored lifespan has no resolution in W0 (0 of 3072 lives ended between the caps; E≥5791 lives/arm/task needed). Routed `dp04-lifespan-has-no-resolution`. |

Three repo facts the literature sections must not contradict and may lean on:

1. **The only learning core that cleared the survival screen is a world-model
   core** (LC.03 v2: wm-latent, t_null 4.65, sole 3σ learner of five). The
   substrate that would carry a deliberative mode is the one that learned.
2. **W0 does not reward lookahead** (DP.05) and does not resolve lifespan
   differences (DP.04 pilot). Every DP claim-spec is therefore blocked on the
   `w0-too-shallow` world decision (Review, DUE 2026-09-06), not on design or
   compute. This document is design INPUT to that decision: §6 states what
   the DP family needs from W1.
3. **The plastic-only decree constrains the alternative**: a separate frozen
   deliberative tower is not a registrable arm; a separate *plastic* tower is
   — and DP.02's two-tower control already is one, deliberately.

## §2 — Question 1: is planner-ablation the right habitisation instrument? (DP.01)

**Answer: it is half the instrument, and the queue entry's suspicion was
right — devaluation is the stronger probe and it is not in the spec.** The
two measure different things: **planner-ablation measures fast-path
COMPETENCE** ("could the reflex path do it alone?"), **devaluation measures
control ALLOCATION** ("which path is actually steering right now?").
Habitisation in the animal-learning sense is a claim about allocation, and
outcome devaluation *tested in extinction* is its defining operationalisation
(Adams & Dickinson 1981; Adams 1982 [verified-web]: overtraining →
devaluation-insensitivity — the dose-response result DP.01's early/late
contrast descends from; Dickinson 1985 [verified-web]; Balleine & Dickinson
1998). The extinction condition is load-bearing: no outcome is delivered at
test, so any behavioural adjustment must come from internally combining the
stored action→outcome contingency with the outcome's NEW value — a cached
policy cannot have been updated. In silico the paradigm goes back to the
founding arbitration model (Daw, Niv & Dayan 2005: rewrite the terminal
outcome's utility, freeze the cache, test; Keramati et al. 2011 simulated
the moderate-vs-overtraining devaluation result directly) [paper identities
verified-web, implementation details from-memory].

**The ablation arm has a confound the spec does not name: distillation by
design.** In Dreamer/MuZero-class agents the fast path is *trained from* the
slow path's outputs, so falling acting-time ablation cost is partially
guaranteed by the training objective — Hamrick et al. 2021 (ICLR,
arXiv:2011.04021) [verified-web] is the reference: MuZero with search
ablated at evaluation loses little on most environments; planning mattered
mostly for *learning*. An agent could show falling ablation cost while still
routing every decision through rollouts — redundancy, not habit. Devaluation
catches that; ablation cannot. Second unnamed confound: **planner decay** —
late in training the slow path itself may have degraded (stale targets,
entropy collapse), so falling ablation cost reflects a weakening planner,
not a strengthening habit; the fresh-task arm controls this only if probed
at the same late checkpoint.

**DP.01's control is probably broken as written.** A goal re-randomised
every episode is cacheable by any goal-conditioned reactive policy if the
goal is observable (universal value functions; Han et al. 2024's
habit-priors are context/goal-conditioned) — its ablation cost may
legitimately fall, which would falsely read as a failed control. To be
genuinely uncacheable the optimal response must depend on information only
obtainable by inference-time computation (hide the goal from the fast path,
or per-episode procedurally novel structure), and the spec should verify
late-checkpoint fast-path-only performance on the control stays near chance.

**A gap worth knowing: nobody has run the devaluation battery on a
world-model deep-RL agent.** The 2024–2025 ML "habitization" line (Han,
Doya, Li & Tani 2024, Nat Commun 15:4461, habit-prior/goal-posterior via
variational Bayes; Lu et al. 2025 "Habi", arXiv:2502.06401, distilling
diffusion planners into 800 Hz reactive policies) measures habitisation
purely as amortisation speed — no flexibility-loss probes at all
[verified-web]. A DP.01 with a devaluation endpoint would be measuring
something this literature conspicuously does not. Also binding: the SR
result (Momennejad et al. 2017 [verified-web]) — reward devaluation alone
cannot detect *transition* caching, so a full battery needs a transition-
revaluation (or contingency-degradation) probe too. And the human-lab
caveats say expect small, finicky effects: habit induction failed five times
in de Wit et al. 2018, and weak devaluation manipulations misclassify
goal-directed actions as habits (Behav Res Methods 2026) [verified-web].

Recommendations carried to §7: devaluation probe as primary endpoint
(STRENGTHEN); fix or demote the cacheable control (INVALIDATE as specified);
planner-health check per checkpoint (STRENGTHEN); ≥3 practice levels
dose-response instead of early/late binary, per Adams 1982's actual shape
(STRENGTHEN); transition-revaluation second axis (STRENGTHEN); keep
rollout-disabling over weight-zeroing, adding Hamrick's acting-time vs
learning-time distinction (KEEP, amended).

## §3 — Question 2: can DP.02's lesion instrument discriminate, and what is the second instrument?

**Answer: as written, no — and the failure is worse than "thin". A single
full-trunk lesion cannot pass its own discrimination requirement.** If the
"shared trunk" is actually one parameter blob containing two functionally
disjoint circuits, lesioning the whole trunk destroys both circuits, both
modes degrade together, and DP.02 PASSES despite zero sharing. The two-tower
architectural control does not cover this case — there the towers are
separated by construction, so it only proves the instrument can see
*architectural* separation, not *functional* separation inside one blob.
The literature backs this: Csordás et al. (ICLR 2021, arXiv:2010.02066
[verified-web]) built differentiable per-task weight masks precisely because
standard networks often FAIL to reuse subnetworks across related subtasks —
"two towers in one blob" is an observed phenomenon, not a hypothetical.

**The correlational measures the spec's notes asked about cannot carry the
claim either.** CKA (Kornblith 2019) is manipulable to arbitrary values
without functional change (Davari et al., ICLR 2023, arXiv:2210.16156) and
dominated by high-variance components (Ding et al., NeurIPS 2021,
arXiv:2108.01661); SVCCA/PWCCA registers large dissimilarity across mere
seed changes; MI estimators are bounded at O(ln N) (McAllester & Stratos
2020) — all [verified-web]. The load-bearing distinction: activation-
statistics measures are blind to which parameters *produce and consume* the
activations; only intervention- or parameter-support-based measures can
discriminate shared substrate from co-located private circuits.

**The lesioning methodology itself needs three repairs, each cited:**

1. **Many small random partial lesions, not one big one** — e.g. ~100 draws
   of 10–20% of trunk units, testing the CROSS-MODE CORRELATION of
   degradation across draws. Shared substrate ⇒ per-draw fast-mode and
   slow-mode damage strongly positively correlated; two-towers-in-a-blob ⇒
   ~uncorrelated. The two-tower control must show the uncorrelated
   signature. This converts the blunt instrument into one that can actually
   discriminate. (Precedent: Morcos et al.'s cumulative random-ablation
   curves, arXiv:1803.06959 [verified-web].)
2. **Mean/resample ablation, not zeroing** — zeroing pushes downstream
   layers off-distribution, so "both modes degrade" may be generic
   distribution shift (Zhang & Nanda, arXiv:2309.16042 [verified-web]).
3. **Match the null on FUNCTIONAL perturbation, not weight magnitude** —
   equal weight-norm is not equal function-space damage; match on induced
   next-layer activation change or output KL. (No source formalises
   "equal-magnitude random lesion" as *the* accepted null — the spec should
   not claim the literature's blessing for it.)

**The corroborating second instrument, recommended: per-mode gradient/Fisher
support overlap on the trunk.** Compute per-parameter squared-gradient mass
under the fast-mode loss and the slow-mode loss separately (a few hundred
batches, no training, no lesioning — minutes on CPU at our <10M scale);
take top-k parameter sets per mode; report IoU against (a) the
hypergeometric chance-overlap null and (b) the two-tower control, whose IoU
must collapse to chance — the proof it can detect a genuinely disconnected
system, which the spec's own control philosophy demands. One step up in
rigour if wanted later: Csordás-style learned weight masks (public code)
anchored by same-mode/different-seed overlap. Third cheap causal option:
cross-mode activation patching with resample corruption — a patch into the
trunk must move BOTH modes' outputs; on the two-tower control, a patch into
tower A must move only mode A.

**Prior art worth citing in the spec's notes, because the question is
genuinely under-measured:** in model-based RL, trunk sharing between the
policy and the model is nearly always *assumed*; the few measurements found
it real but imperfect and contested — PPG measured policy/value objective
interference on a shared trunk and decoupled them (Cobbe et al.,
arXiv:2009.04416); de Vries et al. (arXiv:2102.12924) found MuZero's
observation embeddings and dynamics-unrolled states diverge inside the
nominally shared latent; Hamrick et al. (arXiv:2011.04021) found planning's
contribution is mostly to drive policy learning. An ICLR-2025 paper
(arXiv:2504.01871) gives an end-to-end probe+intervention template at
exactly our model scale for locating planning computation in shared
representations. [all verified-web] No paper was found that measures
world-model/policy encoder sharing with mask/patching-class instruments —
a failure-to-find, not a proven negative, but DP.02 appears to be measuring
something the field assumes.

## §4 — Question 3: does the matched-rate random gate null exist in the adaptive-computation literature? (DP.03)

**Answer: DP.03's sentence is half right and half wrong, and the registry
ordered us to verify or delete it — the verdict is REWRITE.** "Routinely
omitted" survives in softened form: the ponder/early-exit families (ACT,
Graves 2016 arXiv:1603.08983; PonderNet, Banino 2021 arXiv:2107.05407;
BranchyNet; DeeBERT; CALM, Schuster 2022 arXiv:2207.07061) and most
2025-era think/no-think LLM-RL work report NO matched-rate random gate.
But "THE ONLY NULL THAT MATTERS" is factually indefensible on two counts
[verified-web]:

1. **Matched-rate/matched-compute random controls DO exist**: SkipNet's
   random-skip control (arXiv:1711.09485); Mixture-of-Depths' explicit
   stochastic-routing ablation at identical top-k capacity — "performs
   drastically worse" (arXiv:2404.02258, quoted); MoE hash/random routing
   at identical FLOPs-per-token, which is sometimes COMPETITIVE OR BETTER
   than the learned router (Hash Layers arXiv:2106.04426; THOR
   arXiv:2110.04260) — exactly why the control matters; and 2026
   planning-budget RL (arXiv:2606.26463) carries a uniform-random budget
   baseline.
2. **The null that has historically KILLED adaptivity claims is the
   matched-rate CONSTANT gate, not the random one.** Repeat-RNN
   (arXiv:1803.08165) matched ACT with a *fixed* repeat count; in
   arXiv:2606.26463 the random baseline fell below EVERY fixed budget while
   the adaptive gate beat the best fixed budget by +10–65%. A random gate
   at matched rate is a weak null in RL settings; DP.03 needs BOTH nulls —
   constant-at-matched-rate (catches "adaptivity ≈ just average compute")
   and random-at-matched-rate (catches "any stochastic firing suffices").
   DP.03's null_baseline already lists always/never/random; constant-at-
   matched-average-rate is the missing fourth arm.

**Compute-accounting conventions to adopt** [verified-web]: the field
standard is average per-input FLOPs/MACs (dynamic-NN survey,
arXiv:2102.04906), FLOP-matched dense-vs-sparse (Switch/MoE), isoFLOP
curves (MoD), and for RL planner-calls/MCTS-simulations per decision
(Hamrick 2020, arXiv:2011.04021; arXiv:2606.26463 calibrates 32 sims =
1 env frame). The accepted matching method is NOT single-point: sweep the
gate's cost knob and the nulls' rates and compare Pareto curves of return
vs average deliberation compute; for RL match expected planner calls PER
EPISODE and verify realised rates post-hoc (gating shifts state
visitation). Report FLOPs and wall-clock both — dynamic-sparsity FLOPs
savings often do not realise on batched hardware.

**Known gate pathologies, confirming DP.03's control design and adding
one requirement**: collapse to effectively fixed depth (the Repeat-RNN
result is its signature); extreme ponder-cost hyperparameter sensitivity
(ACT's τ-fragility is PonderNet's founding motivation — so DP.03 should
pre-register reporting across ≥3 deliberation-cost values plus the gate's
firing-rate distribution); gates reading global statistics rather than the
input. On that last one: **no surveyed paper runs DP.03's frozen-world
control** (gate fed a stretch where nothing is novel or dangerous, must
not rise) or an input-shuffled gate at matched rate — the spec's control
is a genuine contribution and should say so instead of claiming the
literature omits a null it half-runs.

**Closest published analogue to DP.03's whole claim**: "Finding the Time
to Think" (arXiv:2606.26463) — a PPO-trained gate over a frozen AlphaZero
planner choosing per-state simulation budget, beating the best fixed
budget on 5 games. Also AOP (arXiv:1912.01188) and Hamrick's metacontrol
(arXiv:1705.02670). Hamrick 2020's caution binds our expectations:
planning returns saturate quickly with budget in most domains, so the
adaptive gate's headroom concentrates in hard/time-pressured regimes —
another reason the WORLD must have heterogeneous stakes (§6.4) before
DP.03 can measure anything.

## §5 — Question 4: what does biology actually say about "connected"?

**Verdict, stated plainly per the queue entry's order (a research doc that only
confirms the decree is worthless): biology supports NEITHER literal
one-substrate-two-depths NOR two independent towers. It supports a third
shape: parallel loops through shared machinery — shared cortical state input,
shared action output — with habit implemented as a physically distinct,
CHEAPER shortcut pathway that is trained by the deliberative loop and gated by
a separate prefrontal arbitrator.**

**The strongest honest counterargument to our shared-trunk bet, with the
citation that carries it:** the Yin–Knowlton–Balleine double dissociation
[verified-web]. Lesioning dorsolateral striatum makes behaviour *more*
goal-directed (Yin, Knowlton & Balleine 2004, Eur J Neurosci); lesioning
posterior dorsomedial striatum makes it *more* habitual (Yin, Ostlund,
Knowlton & Balleine 2005). In a single shared network running at two depths,
ablating "the habit part" should degrade the shared computation — instead it
cleanly hands control back to an intact planner, and vice versa. Each
controller demonstrably functions without the other. Reinforcing it: Lee,
Shimojo & O'Doherty 2014 (Neuron) localise arbitration to vlPFC/frontopolar
cortex as an *external* gate that down-regulates posterior putamen — gating
between systems, not depth-selection within one; and Miller, Shenhav & Ludvig
2019 (Psych Rev) argue some habits are value-free S-R repetition — a
*different learning rule*, which a shared trunk with a shallow read-out
cannot express. [all verified-web]

**What survives of the bet — and it is the functional half, which is the half
GOAL.md actually states:** habit content originating as compressed/cached
deliberation has strong, current support:

- **Plan-until-habit** (Keramati, Smittenaar, Dolan & Dayan 2016, PNAS):
  humans plan to a limited depth then splice in cached habitual values for
  the deeper future; time pressure shortens the planned prefix. Goal-directed
  and habitual control are a continuum *inside a single decision*. The single
  strongest empirical result for "one process, variable depth". [verified-web]
- **Successor representation** as middle ground (Momennejad et al. 2017, Nat
  Hum Behav): much human behaviour is cached multi-step prediction — neither
  pure planning nor pure caching. [verified-web]
- **Replay trains the fast path** (Mattar & Daw, Annu Rev Neurosci 2026):
  hippocampal replay "likely trains downstream circuits rather than directly
  guiding choice" — the brain's own planner amortises simulated experience
  into the fast policy, the DYNA/ExIt pattern. [verified-web, abstract]
- **Habits as action chunks called BY the planner** (Dezfouli & Balleine
  2012/2013/2014): hierarchy, not rivalry. [verified-web]
- The 2004-era strict dichotomy is softening: DMS and DLS engage
  concurrently from early training (Kupferschmidt 2017; Vandaele 2019,
  eLife), and the Oct-2025 Trends in Neurosciences synthesis proposes habit
  as trained *shortcut connections between the loops* — "from a strict
  dichotomy toward a continuous, integrated network". [verified-web, abstract]
- In ML the pattern is mainstream and tested: Expert Iteration (Anthony,
  Tian & Barber 2017), AlphaZero's policy head as distilled search, Dreamer's
  policy amortised from imagination, and the amortised-vs-iterative-inference
  formalism (Millidge et al. 2020, arXiv:2006.10524; Marino et al., NeurIPS
  2021) where hybrids beat either alone — with the measured failure mode that
  **the amortised head generalises worse off the distribution the slow
  process visited**. [verified-web]

**Engineering translation for Jack** (carried into §7): a shared state trunk
feeding TWO differently-wired heads — a cheap amortised policy head and a
deliberative planner over the same world model — with ExIt-style distillation
from planner to policy head and an explicit reliability/cost arbitrator.
That keeps the owner's "connected but slightly different purposes" verbatim,
matches the lesion data (heads ablatable independently), matches the ML
evidence (amortised heads fail off-distribution, so the arbitrator must
exist), and keeps the safety property that the slow system can override a
corrupted fast one. Both heads plastic — nothing here needs a frozen tower,
so the plastic-only decree is untouched.

The arbitration literature the DP.03 design should cite: Daw, Niv & Dayan
2005 (uncertainty-based competition); Keramati, Dezfouli & Piray 2011
(speed/accuracy, value-of-information vs deliberation cost); Lee et al. 2014
(reliability-based); Kool, Cushman & Gershman 2016/2017 (cost-benefit
metacontrol — deploy model-based control only when stakes × accuracy
advantage exceed effort cost). O'Doherty et al. 2021 frames the whole family
as a mixture-of-experts with reliability-weighted gating. [all verified-web]

Unverified residue, disclosed: TiNS-2025 and Mattar & Daw 2026 were checked
at abstract level only (paywalls); Everitt & Robbins' addiction mechanism is
[from-memory] and its habit-persistence pillar is contested anyway (Hogarth
2020, Neuropsychopharmacology: human drug-seeking stays largely
goal-directed [verified-web]); no single experiment directly refutes
"habit = shallow inference in the same network" — the case is assembled from
lesion dissociations.

## §6 — What the DP family needs from the world (input to `w0-too-shallow`, Review 2026-09-06)

DP.05's pre-registered falsified_by already named traps, delays and
irreversibility. The literature sweep sharpens that list into five concrete
requirements, each traceable to a paradigm that needs it. This section is
design input to the W1 decision, not a demand — but a W1 that lacks these
will foreclose the DP family a second time, and it should do so knowingly:

1. **Perceivable-in-advance hazards** (DP.00/DP.05): lookahead pays only if
   a trap is visible *before* entry. W0's integrity death (routed
   `w0-kills-a-forager-by-integrity-at-25-minutes`) is the opposite shape —
   damage that accrues invisibly rewards no planning.
2. **Revaluable outcomes — and the project already owns the mechanism.**
   Outcome devaluation (§2's primary probe) needs an outcome whose value can
   change *without new instrumental experience*. Biology's manipulation is
   conditioned taste aversion — which is exactly the TA family
   (GOAL.md's one-trial, long-delay learning; TA.02 passed 08-19). A food
   type that can turn noxious gives the devaluation probe an in-world
   manipulandum for free: no experimenter surgery on the reward function,
   just a world event the slow path can know about and the cached path
   cannot. This is the strongest single synergy the sweep found: **the taste
   machinery is the habitisation instrument.**
3. **Degradable contingencies** (§2, second probe axis): a place/action
   whose P(outcome|response) can be degraded — e.g. food that sometimes
   appears non-contingently — so the SR-style middle ground (Momennejad
   2017) is distinguishable from true goal-directedness.
4. **Heterogeneous stakes and novelty** (DP.03): the gate needs a world
   where some states are dangerous or novel and others safe and familiar —
   the cost-benefit arbitration literature (Kool et al.) is entirely about
   deploying deliberation where stakes justify effort. A uniform world
   gives a gate nothing to allocate; W0 is measured uniform (DP.04:
   lifespan has no resolution; DP.05: H10 pays less than H4).
5. **Stable, repeatable task structure with occasional revaluation events**
   (DP.01): habitisation needs practice to pay (consistent world rules —
   already a GOAL.md constitutional property) punctuated by rare value
   changes that reveal which controller is steering. A world that changes
   constantly never caches; a world that never changes never shows the
   cache.

## §7 — Spec revision proposals (NOT registered by this pass)

Nothing below is registered by this document. DP.01–DP.03 are unreachable
today (behind LC.04's unsatisfiable premise and the W1 decision), so there
is no urgency that would justify skipping the queue protocol; the revisions
should be registered under the T1.02 precedent (strengthen only, old
versions stay in history) when the family becomes reachable, and the
world-facing items route through the Review's `w0-too-shallow` design. Per
SYSTEM.md law 4: every change below is a strengthen or an addition; no
threshold moves in the loosening direction.

**DP.01 (habitisation):**
- ADD a devaluation-in-extinction probe as the primary endpoint: at each
  practice checkpoint, rewrite the outcome's value through a channel only
  the deliberative path reads (§6.2: the TA machinery is the natural
  in-world channel), freeze all learning, measure adaptation with rollouts
  still ENABLED. Falling devaluation sensitivity with practice = habit took
  control. (STRENGTHEN)
- FIX the control: a goal re-randomised every episode is cacheable by a
  goal-conditioned reactive policy. Hide the goal from the fast path or use
  procedurally novel structure, and verify late-checkpoint fast-path-only
  performance stays near chance. (the current control is INVALID as
  specified — this is a control repair, not a loosening)
- ADD a planner-health check at every checkpoint (planner-enabled
  performance on a matched held-out task) so planner decay cannot
  masquerade as habitisation. (STRENGTHEN)
- REPLACE the early/late binary with ≥3 practice levels and a
  pre-registered monotone-decline prediction (Adams 1982's actual shape).
  (STRENGTHEN)
- ADD a transition-revaluation / contingency-degradation second axis so an
  SR-style cache cannot pass as goal-directed (Momennejad 2017).
  (STRENGTHEN)
- KEEP rollout-disabling over weight-zeroing; note the distillation-by-
  design confound (Hamrick 2021) and distinguish acting-time from
  learning-time ablation in the spec text. (KEEP, amended)

**DP.02 (connectedness):**
- REPLACE the single trunk lesion with ~100 small random partial lesions
  and a cross-mode degradation-correlation metric; the two-tower control
  must show the uncorrelated signature. As written the spec can pass on a
  two-towers-in-one-blob network, which its own kills-line says it must
  detect. (control repair / STRENGTHEN)
- Mean/resample ablation instead of zeroing; match the random-lesion null
  on functional perturbation (output KL), not weight magnitude.
  (STRENGTHEN)
- ADD the second instrument: per-mode gradient/Fisher support overlap on
  trunk parameters, IoU vs hypergeometric chance, two-tower control must
  collapse to chance. Minutes of CPU at our scale. (STRENGTHEN)
- KEEP the two-tower must-dissociate control and dose-response reporting.
  (KEEP)

**DP.03 (deliberation gating):**
- REWRITE the notes sentence per §4 (the literature half-runs the null; the
  constant-rate null is the historically lethal one); ADD the constant-at-
  matched-average-rate arm to null_baseline alongside the existing
  always/never/random. (STRENGTHEN — a fourth null is added, none removed)
- Pareto-curve matching over ≥3 deliberation-cost values, firing-rate
  distribution reported, planner calls matched per episode with post-hoc
  rate verification. (STRENGTHEN)
- KEEP the frozen-world control and label it as a contribution — no
  surveyed paper runs it. (KEEP)

**Architecture (for the Review/owner, NOT a spec edit — this is the §5
verdict landing):** the biologically faithful reading of "connected but
slightly different purposes" is a shared plastic state trunk feeding two
differently-wired plastic heads (cheap amortised policy; deliberative
planner over the same world model), ExIt-style distillation from planner to
policy head, and an explicit reliability/cost arbitrator. DP.02's
connectedness claim should be read as binding at the TRUNK (shared state
representation, which biology supports) and NOT at the heads (where biology
shows independent ablatability and ML shows amortised heads failing
off-distribution). GOAL.md's own fast/slow section already says
"differentiated function, shared substrate" — the literature says the
differentiation is real wiring, not just depth. This is a CHAMPIONS-seat
question and an owner conversation, not a decree this doc can issue; the
counterargument (Yin/Knowlton/Balleine) is recorded in §5 as owner
directives require. Note that wm-latent — the seated learning core — is
already a world-model architecture, so the two-head shape is buildable on
the current champion without unseating it.
