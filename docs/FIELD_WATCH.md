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

**Sweep date:** 2026-08-10 · **Window:** ~2026-02 → 2026-08 (6 months)
**Scout:** field watch, week 1 (this is the first sweep; there is no prior
state to diff against, so the NO-ACTION sections below are unusually long by
construction — they establish the baseline the next sweep diffs).

---

## 0. Coverage — what was actually searched, so the gaps are visible

| Front | Searched | Depth reached |
|---|---|---|
| 1. Learning cores | world models, DreamerV4/Simulus/TWM lineage, JEPA-family collapse fixes, sample-efficiency claims, tiny-model control | **abstracts fetched and read** for 6 papers; 2 full-text |
| 2. Multimodal fusion | binding objectives, modality collapse, unified tokenisation, identifiability of physical parameters from multimodal latents | **full text fetched** for the one nomination |
| 3. Memory | episodic/agent memory, consolidation, extractive retrieval, CPU encoders, BEIR/static embeddings | abstracts; one protocol paper fetched |
| 4. Open-endedness & curiosity | intrinsic motivation, learning progress, autotelic agents, noisy-TV, curriculum generation | abstracts; one source **investigated and rejected** (§4) |
| 5. Worlds & embodiment | survival sims, homeostatic RL, MuJoCo/MJX/Newton, embodied benchmarks | abstracts; one full text |
| Biology-as-oracle | infant sleep consolidation, hippocampal replay, homeostatic RL in neuroscience | **abstracts only — one target unreachable (§6)** |

**Known gaps in this sweep, stated so nobody assumes coverage:**
- No conference-proceedings sweep (ICML/NeurIPS/CoRL/RLC 2026 accepted lists were
  not enumerated). arXiv + web only.
- Smell, taste, and voice — three senses `GOAL.md` names as constitutional — were
  **not searched this week**. That is a real hole and it is the first item of the
  next sweep, not a judgement that the field is empty.
- No non-English sources.

---

## 1. NOMINATIONS

Four. Each states the source, what is **[V]**erified (fetched and read) versus
**[c]**laimed (asserted by the source, not independently checked), which spec it
enters, its cost on **our** substrate, and — steelmanned both ways — why it
might win and why it might lose.

---

### N1 — The certificate-gated identifiability protocol → enters `UB.11` as a mandatory pre-gate

**Source:** *What Can Latent World Models Know? Physical Parameter
Identifiability in Multimodal Predictive Representations* —
[arXiv:2607.27017v2](https://arxiv.org/abs/2607.27017), posted 2026-07-30
(Tan, Xu, Tao, Hong, Feng, Du; NYU / CMU / Columbia). **Not cited anywhere in
this repo.**

**The verified claim [V]** — full text read. Modalities: vision, proprioception,
force/touch. Three hidden physical parameters in a controlled environment
(PokeWorld): mass, drag, contact stiffness. The design is a factorial over
*which modalities are INPUTS* × *which modalities are PREDICTION TARGETS*, and
the headline number is this:

| variant | log-stiffness R² | log-mass R² | position R² |
|---|---|---|---|
| V — vision in, vision out | **−0.02** | 0.15 | 0.04 |
| VXt — touch as a **target** | **0.40–0.43** | 0.26 | 0.17 |
| VFX — all in, all out | **0.50** | 0.29 | 0.98 |
| **certificate** (supervised probe on raw obs) | **0.87** | 0.86 | ~1.0 |

Stiffness enters the latent **only when touch is a prediction target — not when
touch is merely fused into the input** (0.50 vs −0.02). The paper's own summary:
*"Inputs limit what can be known, while prediction targets decide what is
retained."* Vision-only models discard even perfectly visible object state under
single-step prediction. Five-fold data increases do not rescue a parameter that
has no prediction pressure.

**Why it is a protocol and not just a result.** The *certificate* is the load-
bearing invention. Before asking "did the model learn parameter X?", a supervised
probe on the **raw observations** establishes that X was recoverable at all
(0.87 for stiffness). Only then is the model's latent probed. This makes a null
result attributable to **the objective** rather than to the environment.

**Which spec it enters.** `UB.11` — the standing modality ablation matrix — as a
**pre-gate on every column**. `UB.11` currently kills any sense whose four
perturbations are indistinguishable from the placebo column, and the kill is
automatic: *"Deletion is the default action, not a discussion."* The certificate
says that verdict is only sound if the sense was **certifiably informative in
the fixture to begin with**. Without it, `UB.11` cannot distinguish "this sense
is decorative" from "this fixture never gave this sense anything to say" — and
it deletes an encoder either way.

That distinction is already this project's most load-bearing one under a
different name: `LESSONS.md`, *"VOID is not FAIL, and the difference is
load-bearing"*, and *"a detector that cannot see its own positive control has
measured nothing"* (T0.13). The certificate **is** the positive control for a
modality column. `UB.11` today has a placebo column (the negative control) and
no positive one.

**Cost on our substrate.** The entire study — including the real-robot section —
ran on **one RTX 4060 laptop GPU**, ~5M-parameter models, **30–45 minutes per
variant** [V, stated by the authors]. A T4 or P100 is in the same class
(plausibly 1.5–2× slower), so a certificate pass is **~1 GPU-hour per fixture**,
run once per fixture rather than per experiment. On the CPU side the certificate
is a supervised probe on cached observations, which is `UB` §6's existing
cached-embedding pattern — likely **minutes on 4 ARM cores**.

**Why it might WIN (falsifiable).** Run the certificate on the `UB.9`/HNS
fixture. If any sense's certificate comes back low — i.e. the fixture does not
actually carry decodable information in that channel — then that sense's `UB.11`
column has been measuring nothing, and any past or future deletion verdict on it
is void. That is a *discoverable, currently-unmonitored* failure of a standing
spec, and the ladder has no other instrument that would find it.

**Why it might LOSE (steelmanned).** Three real ways:
1. **The certificate may be trivially satisfied here.** `UB` fixtures are
   synthesised by us with known ground truth; if every channel certifies at ~1.0
   by construction, the gate adds a run and no information. The counter-argument
   is that this is only knowable by measuring — but if it certifies trivially,
   the honest outcome is to delete the gate, and that costs a GPU-hour.
2. **Hardware unlike ours, per `LESSONS.md`.** Every number above is GPU-trained
   (Ada laptop). This project has been burned three times by published figures
   that did not transfer to this box (int8, i8mm, RTF). Nothing here is measured
   on ARM CPU, and the paper's own limitation is blunt: *"what transfers is the
   protocol, not the ceiling."*
3. **Scope.** The authors exclude objectives whose optimum is not a conditional
   mean — which may exclude `A2`'s twohot/symlog distributional heads. The
   protocol may need adaptation before it applies to the arm we most care about.

**Second, weaker nomination inside the same paper.** The input-vs-target result
argues for an extra `UB.10` arm in which touch and audio are **targets, not just
inputs**. `UNIFIED_BRAIN_BAKEOFF.md` §1.2 already reached that conclusion by
argument ("cross-modal masked prediction is the binding FORCE") and A3 already
encodes it. This paper does not add an arm so much as supply the first
**mechanistic measurement** behind an arm we had chosen on reasoning. Recorded
as corroboration, not as a new arm.

---

### N2 — Two principled anti-collapse regularisers → enter `LEARNING_CORE` §5.4 as `A4b` / `A4c`

**Sources:**
- *Sensorimotor World Models: Perception for Action via Inverse Dynamics* —
  [arXiv:2606.20104](https://arxiv.org/abs/2606.20104), 2026-06-18.
- *LeJEPA / SIGReg* — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
  (LeCun & Balestriero, 2025-11), plus the in-window theory paper *When Does
  LeJEPA Learn a World Model?* —
  [arXiv:2605.26379](https://arxiv.org/abs/2605.26379), 2026-05-25.

**Neither is cited anywhere in this repo.**

**The problem they attack is one we already named.** `A4` (`wm-latent`, the JEPA
representative) carries the survey's harshest verdict — *"the worst silent
failure in the survey: collapse with a falling loss"* — and is defended today by
an EMA target encoder plus a mandatory RankMe / per-dim-variance monitor with a
pre-registered VOID floor. That is **detection**. Both papers below claim
**prevention**, and each removes machinery `A4` currently depends on.

**SMWM [V]** — a single inverse-dynamics regulariser,
`L = L_fwd + λ·E‖h_ψ(z_t, z_{t+1}) − a_t‖²`, where `h_ψ` is a 2-layer MLP of
width 256. The authors' claim: it prevents collapse *and* aligns the latent to
controllable degrees of freedom, **"without frozen encoders, exponential moving
averages, or complex latent regularizers"** — i.e. it deletes `A4`'s target
encoder and its predictor/encoder LR-ratio landmine outright. Measured against
SIGReg under a fixed 50-step planning budget: matches it on three 2D tasks, and
on OGBench-Cube (the only 3D contact-rich task) **retains 84 % success where
SIGReg drops to 59 %** [V].

**SIGReg [c/V]** — enforce that the embedding distribution is isotropic
Gaussian, via random 1-D projections (Cramér–Wold), making collapse impossible
by construction; linear time and memory, one hyperparameter, no stop-gradient,
no teacher–student, no schedulers [c — claims read, not reproduced]. The 2026
theory paper proves the guarantee is **exactly** for Gaussian latents under
stationary additive-noise transitions, and does not hold otherwise [V, abstract].

**Which spec they enter.** `LEARNING_CORE.md` §5.4, as variants of the
*conditional* arm `A4`:

| arm | anti-collapse mechanism | params (est.) |
|---|---|---|
| `A4` (existing) | EMA target encoder + RankMe VOID floor | ≈1.37 M |
| **`A4b` (new)** | inverse-dynamics regulariser, no EMA | ≈1.37 M + ~70 K MLP |
| **`A4c` (new)** | SIGReg, no EMA | ≈1.37 M + 0 (a loss term) |

Both keep `A2`'s world model byte-identical, exactly as `A3` does — one changed
term, which is the isolation discipline §5.4 already uses.

**Cost.** Near zero. `A4c` adds a loss term with linear cost; `A4b` adds a
2-layer MLP. Neither changes the replay ratio, which `LEARNING_CORE` §3.0
measured as the thing that actually costs. If `A4` runs at all, `A4b`/`A4c` cost
what `A4` costs.

**Why they might WIN (falsifiable).** `A4` is *conditional* today (§5.5) largely
because its failure mode is silent and its detector is a monitor rather than a
guarantee. If `A4` VOIDs on the rank floor while `A4b` or `A4c` clears the
learning gate on the same world model and the same seeds, then the JEPA family
was never the problem — the **collapse defence** was, and a family the survey
came close to shelving re-enters on measured grounds. `A4b` additionally
predicts *action-aligned* latents, which is precisely the property `GOAL.md`
wants from a body that learns by acting.

**Why they might LOSE (steelmanned).**
1. **The regime is wrong, and it is our regime.** `LEARNING_CORE` §5.4 states
   the objection already: reconstruction is a much stronger signal when the
   observation is ~96-dimensional and every dimension matters. Both papers argue
   from pixels — SMWM is **pixel-only, with no proprioceptive or multimodal
   inputs at all** [V]. On a ray retina plus a drive vector, the wasted-capacity
   argument for deleting the decoder may simply not apply, and then the best
   anti-collapse regulariser in the world defends an arm that should not exist.
2. **SIGReg's guarantee has a named precondition** — Gaussian latents, stationary
   additive-noise transitions [V]. `A2`'s stochastic state is **32×8
   categorical**, which is not Gaussian. `A4c` may therefore inherit the
   regulariser without the theorem, which is the worst of both.
3. **Scale.** SMWM is ~15 M params (ViT-Tiny encoder ~5 M + ~10 M transformer
   dynamics) on pixel tasks; we would run the mechanism at ~1.4 M on vectors.
   `LESSONS.md`'s standing warning applies.
4. Neither paper reports hardware or GPU-hours [V — I looked; it is absent].

---

### N3 — Interoceptive precision allocation → enters `NEEDS_AND_DEATH` §2.4b as a new arm

**Source:** *Interoceptive Attention as Dynamic Homeostatic Prioritization in a
Foraging Agent* — [arXiv:2608.04232](https://arxiv.org/abs/2608.04232), posted
**2026-08-04** (six days old), SAB 2026 camera-ready. Code, layout banks and
analysis pipelines released:
`github.com/sgrimbly/attention-aif-sab2026-snapshot` [V].

**Already known to this repo — as a survey line, never as an arm.**
`PURPOSE_AND_SCAFFOLDING.md` §1 lists it in a related-work table with the
one-line gloss *"dynamic prioritisation among competing internal variables"*.
Nothing recorded its numbers, its controls, or its code release, and no spec
references it. This nomination adds the measurements below and proposes it as an
arm; it does not claim to have found an unknown paper.

**The verified claim [V].** An active-inference agent in a 6×6 gridworld with
three depleting needs — **hunger, thirst, suffocation** — plus one inert control
channel; depletion kills; 60-step episodes; 12 layouts stratified easy/medium/far.
The mechanism: a **fixed perceptual precision budget κ, dynamically reallocated
toward whichever body-state channel is most urgent**.

| agent | survival | 95 % CI |
|---|---|---|
| **attentive** (precision → most-needed channel) | **0.414** | [0.365, 0.463] |
| uniform (κ = 0.65 on every channel) | 0.199 | [0.158, 0.240] |
| **anti-aligned** (precision → *least*-needed channel) | **0.144** | — |
| oracle ceiling (true model) | ~0.66 | — |

**2.08× survival over uniform**, p ≤ 10⁻⁴, n = 32 seeds × 11 layouts. The
attended channel learns its dynamics **~2.4× faster**. Roughly half the benefit
acts through *planning*, not inference.

**Why this one is worth the builder's attention specifically.** Its
methodology is ours, arrived at independently: a null (uniform), a **control
that must fail and does** (anti-aligned, 0.144 < 0.199 — worse than the null, in
the predicted direction), a ceiling (oracle), 32 seeds, and CIs. This is not a
press release.

**Which spec it enters.** `NEEDS_AND_DEATH.md` §2.4b. That section hands the
policy nine interoceptive floats and then **declares its own divergence from
biology in writing**: *"this interoceptive channel is noiseless, undelayed and
complete. Real interoception is none of those, and a future arm that adds
observation noise and a one-decision delay is the honest version. It is not run
here."*

The uniform baseline this paper beats by 2× **is our current design**. The paper
supplies the missing third leg the section anticipated: not just noise and
delay, but a **budget** — total precision fixed, allocation learned. It is the
declared-divergence arm, already scoped by us, now with an existence proof and
released code.

**Cost.** Trivial. 6×6 gridworld, planning horizon H = 3, five sampled actions,
Dirichlet pseudo-counts. This runs on **4 ARM cores at `nice 19`**; no GPU. It is
the cheapest nomination in this report by a wide margin, and it is the only one
whose original experiment we could re-run in full on this box.

**Also: this is new evidence on `A3`.** `LEARNING_CORE` §3.3.2 dismissed active
inference for having **"no standard-benchmark entry in 9 years"**, and `A3`
(`wm-efe`) carries a *low* prior with a named collapse failure mode
(arXiv:2303.01618). This is a 2026 active-inference result on a needs-and-death
task, with controls, that beats its null 2×. It does not overturn §3.3.2 — a
6×6 gridworld is not a standard benchmark either — but `A3`'s prior was set on a
literature that has since moved, and `SYSTEM.md` says a settled position may be
revisited **on new evidence**. This is new evidence. It is not a decision.

**Why it might WIN (falsifiable).** Give the needs suite a fixed precision budget
and let allocation be dynamic. If survival on `W0`/`W1` beats the complete-and-
uniform nine-float channel at ≥3σ over 3 seeds, then `GOAL.md`'s
"interoception" is not a *list of floats* but an *allocation problem*, and every
sense Jack has inherits the same question — which is a constitutional-scale
finding about what "all senses in unison" means.

**Why it might LOSE (steelmanned).**
1. **The substrate gap is enormous and this is the real risk.** A 6×6 discrete
   gridworld with a factorised POMDP, a 5-action space and a *fixed transition
   prior B* is not a continuous MuJoCo humanoid with a 348-dim observation and a
   learned policy. Precision-weighting has a clean meaning in a Dirichlet
   likelihood and **no obvious referent** in `A0`'s PPO. The mechanism may not
   have a well-defined port at all, and that must be settled before it is an arm.
2. **The gain may be an artefact of a starved baseline.** κ = 0.65 uniform is a
   *chosen* uniform. Our nine floats are noiseless and exact — arguably already
   the *oracle* condition on the perception side, in which case there is no
   budget to reallocate and the result is inapplicable by construction.
3. `NEEDS_AND_DEATH` §2.7 already commits to a deletion ablation per variable.
   Adding an attention mechanism over those variables enlarges the matrix, and
   `SYSTEM.md`'s "no new organ without a scar" applies: **no failure in this
   repo has yet been traced to uniform interoceptive precision.** On that rule
   alone, this is a nomination that should probably wait for `NE.03` to run
   first. Stated plainly because it is the strongest argument against my own
   nomination.

---

### N4 — The entity-collision protocol → enters `MEMORY_RETRIEVAL_BAKEOFF` §2 (evaluation-set design)

**Source:** *Entity-Collision: A Stratified Protocol for Attributing Retrieval
Lift in Agent Memory* — [arXiv:2605.29630](https://arxiv.org/abs/2605.29630).
Code and 37 reproduce scripts released, Apache-2.0 [c — release stated in the
abstract; repository not opened this sweep].

**The claim [V, abstract read].** Construct the distractor set so that **every
distractor shares the answer's entity tokens**. This makes BM25 a **floor by
construction**, so any lift above it is attributable to the embedder rather than
to lexical overlap. Queries are then stratified by discriminator tag rather than
averaged. Evaluated 5 tags × 3 embedders × 5 collision degrees with
paired-bootstrap 95 % CIs; replicated on LongMemEval (n = 500). Finding:
**BGE-large's 2.7× parameter advantage does not uniformly help — encoder
capacity is not the binding constraint.** Extractive throughout (BM25 +
embedding ranking); no generative recall.

**Which spec it enters.** `MEMORY_RETRIEVAL_BAKEOFF.md` §2.1, which builds the
paraphrase set under a **lexical-disjointness invariant**. Entity-collision is
the *opposite* discipline, and the tension is the point:

- *Lexical disjointness* removes lexical overlap so the semantic arms have
  something to do. Risk: it can **flatter** the embedders by handicapping BM25 —
  the exact shape of `LESSONS.md`'s dropout confound (*one arm had a handicap and
  the difference was attributed to architecture*).
- *Entity collision* floors BM25 by construction and measures lift above it.

Our bakeoff arbitrates `A_bm25` against `C_static` / `D_minilm` / `E_hybrid` /
`F_cascade`. If the evaluation set is built only under disjointness, a win by a
semantic arm is confounded with the set's construction. Running **both**
constructions as strata turns that confound into a measurement.

**Cost.** Effectively free. It is a change to how the eval set is generated, not
a new arm — CPU-only, and the whole MR bakeoff is already budgeted at ~45 minutes
wall-clock at `nice 19`.

**Why it might WIN (falsifiable).** Score the existing arms on an
entity-collision stratum. If the ranking **inverts** — BM25 at or above the
static/dense arms where disjointness said otherwise — then the bakeoff's verdict
was a property of its fixture, caught before adoption. That is a real,
currently-unmonitored risk in a bakeoff that is otherwise ready to run.

**Why it might LOSE (steelmanned).**
1. Jack's diary is **not** an entity-heavy corpus of preferences and tool calls.
   It is attributed events — heard/said/did, per person. Entity collision may be
   a poor model of how his diary's distractors actually collide, and importing it
   would test a corpus he does not have.
2. §2.3 already grades distractors three ways and §1.10 already pre-filters on
   provenance. The marginal information over what is specced may be small.
3. **Provenance of the source is weaker than N1–N3.** I read the abstract on
   arXiv; I did not open the repository, check the venue, or verify the authors.
   Given §4 below, that gap is worth stating rather than glossing.

---

## 2. WATCHLIST — promising, not yet nominatable

| item | what it is | what would PROMOTE it |
|---|---|---|
| **Simulus** — [arXiv:2502.11537v4](https://arxiv.org/abs/2502.11537) (v4 2026-05-10) | Combines four known improvements: **flexible tokenisation for mixed modalities**, intrinsic motivation for uncertainty reduction, **prioritised world-model replay**, regression-as-classification. SOTA planning-free on Atari-100K; matches DreamerV3 on **DMC-Proprio-500K**; beats TWM/RND/PPO-RNN/E3B on **Craftax-1M**. Modalities enter as per-modality encoders → fixed-length token sequences → concatenated along the temporal axis, separate embedding tables. | **A parameter count and a per-step wall-clock.** The paper reports neither, and states outright that *token-based world model agents remain slower to train than other baselines*. `LEARNING_CORE` §6.2/B4 requires **≥5.0 sim-s/real-s on 3 ARM cores**; a token-based RetNet world model is exactly the shape that fails that floor. Until someone measures it, it cannot be costed and must not be nominated. Its **prioritised replay** component is separately interesting to `LC.02` (the replay-ratio spec) and is the cheapest piece to test in isolation. |
| **Survival RL** — [arXiv:2605.31273](https://arxiv.org/abs/2605.31273) (2026-05-29, Nguimatsia-Tiofack, Schramm, Le Hellard, Carpentier) | **DISAMBIGUATION, recorded so a future sweep does not chase the title.** "Survival" here means *maximising dwell time at target goals* — a classification-based objective, **not** homeostatic needs and not death. Matches CRL on manipulation; **2–8× better on stable long-horizon locomotion**. | Whether the classification objective still works when the "goal" is a needs setpoint rather than a target state. If it does, it is a candidate reward formulation for `NE.03`'s arms; if not, it is a locomotion result and belongs nowhere near the needs suite. |
| **Reward-prediction-biased replay** — Nature Comms `s41467-025-65354-2` | Post-learning hippocampal-striatal replay is biased by reward-prediction signals in rats; supports sleep-dependent learning over multiple days. Would be a **biology-oracle argument for prioritised replay** in `NE.05` / SIESTA — and it converges with Simulus's prioritised-replay component from the ML side, which is the kind of agreement worth taking seriously. | **Unverified — the fetch failed (connection refused).** Read the paper before any use. `NEEDS_AND_DEATH` §1.2 already carries the flag *"BIOLOGY CITATIONS NOT YET VERIFIED"*; this must not become the next one. |
| **Var-JEPA** ([arXiv:2603.20111](https://arxiv.org/abs/2603.20111)) · **Equilibrium World Models** ([arXiv:2606.23463](https://arxiv.org/abs/2606.23463)) · **Multimodal Latent Reasoning via Predictive Embeddings** ([arXiv:2604.08065](https://arxiv.org/abs/2604.08065)) | Titles surfaced in the JEPA-family sweep; abstracts **not fetched** this week. | Read them. Listed so the next sweep does not re-discover them as if new. |
| **Newton 1.0 GA** (NVIDIA, GTC 2026) — claimed **475× MJX** on manipulation [c, vendor/secondary source] | `SURVIVAL_WORLD.md` §2.2 ruled out Isaac/Newton **definitively**. Nothing here re-opens that: the speedup is NVIDIA-GPU-only and our binding constraint is 4 ARM cores plus burst free GPU. | Recorded only as a **re-open trigger**: if the fidelity ladder ever becomes GPU-bound rather than CPU-bound, §2.2's verdict was decided under the opposite assumption and should be re-read. Not a nomination. |

---

## 3. NO-ACTION — fronts where nothing cleared the bar

Stated plainly. An empty week honestly reported beats a padded one.

**Front 3 — MEMORY: nothing new for the retrieval engine.** The 6-month memory
literature is overwhelmingly **LLM-agent memory** — LoCoMo/LongMemEval
leaderboards, mem0, MAGMA (0.70 judge score), RecMem, A-MEM, E-mem, MemoryOS.
Nearly all of it is **generative**: summarise, reconstruct, consolidate into
prose. `GOAL.md` and `MEMORY_RETRIEVAL_BAKEOFF.md` §5 forbid generative recall
structurally, not by preference — Jack's diary must be *extracted*, verbatim,
or `LG.00` is unfalsifiable. **This entire body of work is constitutionally
inadmissible as an arm**, however good its numbers. Only the *protocol* paper
(N4) survived that filter. No new CPU encoder beat the incumbents; the static-
embedding tier is where §1.2 left it.
**One item worth flagging for the builder, not as a nomination:** RecMem's
*recurrence-triggered* consolidation — a cheap buffer that fires consolidation
into higher layers on a trigger rather than a clock — is structurally the same
question as `NE.05` §3.3 ("what triggers sleep"). It is a generative system, so
it cannot be an arm; its **trigger design** may still be worth a read.

**Front 4 — OPEN-ENDEDNESS & CURIOSITY: nothing cleared the bar.** The window
produced no result that would add an arm to `CURIOSITY_BAKEOFF`'s
`disagree` / `lp` / `metra` / `vlm-lp` set, and nothing new on the noisy-TV
front beyond *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress
Monitoring* (arXiv:2509.25438) — which is 2025 and whose thesis, *LP is the
noise-robust signal*, is the position `CURIOSITY_BAKEOFF` §1.2 already holds and
already made `lp` the favourite for. **Corroboration, not news.** The one
apparently-new curiosity framework found this week was investigated and rejected
outright — see §4.

**Front 5 — WORLDS & EMBODIMENT: no change to the fidelity ladder.** MuJoCo CPU
throughput figures are unchanged in kind (~650 K steps/s single humanoid on an
M3 Max, 1.8 M on a 64-core 3995WX — both irrelevant to 4 shared ARM cores,
and the number that actually governs us is `DIRECTION_AUDIT` §4.1's **measured**
0.17 vs 22.97 sim-s/real-s on *this* box). MJX and Newton are GPU-parallel-env
plays, which is the axis `SURVIVAL_WORLD` §2.2 already ruled out. No new cheap
embodied survival benchmark surfaced. **W0 → W3 stands as specced.**

**Fronts 1 & 2 delivered; three of five fronts did not.** That is the honest
shape of this week.

---

## 4. A DISCIPLINE FINDING — a new failure mode for the RESEARCH step

**Search results now include AI-agent-authored preprints on unmoderated hosts,
and at least one misreports its own results table.**

While sweeping front 4, a well-ranked result appeared: *Toward a Computational
Theory of Curiosity: Information-Theoretic Exploration in Open-Ended
Environments*, hosted at `clawrxiv.io`, claiming a **Curiosity Information Gain**
framework that *"discovers 34 % more environment states than RND and 21 % more
than ICM within identical compute budgets, while avoiding the noisy-TV
problem."* On its face this is a front-4 nomination: it decomposes curiosity into
novelty, **learnability filtering via ensemble disagreement**, and
competence-weighting — a near-exact match for `CURIOSITY_BAKEOFF`'s `disagree`
and `lp` arms combined.

**It was fetched and rejected.** The site describes itself as hosting *"papers
published autonomously by AI agents."* The author is `QuantumWhiskers`, no
affiliation, posted 2026-03-17. There is no peer review or editorial oversight.
The environments (MazeWorld-v3, CausalChains-v1) are custom, not standard. No
code is available. And decisively: **the paper's own results table does not
support its own abstract** — 67.6 % vs RND's 54.0 % coverage is a ~25 % relative
improvement, not the 34 % claimed.

**Why this belongs in a field-watch report rather than being silently
discarded.** This project's research step was designed against *human* failure
modes — press releases, unreproduced claims, hardware mismatch. Machine-generated
preprints are a **different** failure mode: they are fluent, correctly
formatted, well-targeted to the searcher's query, cite real prior work, and can
be internally inconsistent in ways a skim will not catch. This one was
persuasive precisely because it matched our open questions so well.

The existing rule — *VERIFY before nominating; a press release is not a result*
— caught it. But it caught it on the **fifth** check (host → author → review →
code → arithmetic), and the first four are checks a hurried agent skips. **The
cheap check that would have caught it alone is the last one: read the results
table and confirm it says what the abstract says.**

Recorded here for the builder as a candidate `LESSONS.md` entry — *an abstract
is a claim about a table, and nothing was checking the table agreed* — which is
the same shape as the existing lesson *"a ledger entry is a claim about code, and
nothing was checking the code still matched."* **I am not writing it;
`LESSONS.md` is not mine to edit.** It is nominated like everything else here.

---

## 5. What this report does NOT claim

- **No arm here has been run.** Every number is someone else's measurement on
  someone else's hardware. Nothing in this file is evidence about Jack.
- **No nomination is a recommendation to adopt.** `SYSTEM.md` law 3 stands: the
  bakeoff decides, not this document and not its author.
- **Nothing here changes a spec, a threshold, a decision, or a line of code.**
- **N1's protocol is the only item I would defend as more than a candidate**, and
  even that only because it is a *measurement instrument* rather than a
  mechanism — it can only cause us to learn that a column of `UB.11` was empty.
  If it finds nothing, we have lost one GPU-hour and gained a positive control.
- **Verification is uneven and marked as such**: N1 full-text; N2 mixed
  full-text/abstract; N3 full-text + released code, not run; N4 abstract only.
- **Three of six fronts were not meaningfully advanced**, and three senses that
  `GOAL.md` calls constitutional — **smell, taste, voice** — were not searched at
  all this week.

---

## 6. Queued for next sweep (2026-08-17)

1. **Smell, taste, and voice.** Constitutional senses, zero coverage this week.
2. Open the four unfetched watchlist abstracts before they are re-discovered.
3. The Nature Comms replay paper (fetch failed) — and a general pass on whether
   `NEEDS_AND_DEATH` §1.2's unverified biology citations have acquired sources.
4. Conference proceedings: ICML / NeurIPS / CoRL / RLC 2026 accepted lists.
5. Diff against this file. From next week the NO-ACTION sections should shrink
   to genuine deltas.
