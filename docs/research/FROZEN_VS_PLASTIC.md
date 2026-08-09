# FROZEN VS PLASTIC — is a welded-shut tower a ceiling on Jack?

> Researched 2026-08-09, in answer to the owner's challenge of the same day:
> *"he must be able to talk hear smell see feel — he needs all data points of
> humans. Consider what will hold us back, and whether encoding everything and
> FROZEN WEIGHTS are best towards AGI, like LLM and vision. Think of the whole
> system this way."*
>
> **Status: DRAFT IN PROGRESS.** Sections fill in as the evidence lands.
> Provenance tags follow the house convention:
> **[V]** verified against the primary source during this pass ·
> **[c]** carried from another document in this repo and NOT re-verified ·
> **[L]** read from `experiments/ledger.json` ·
> **[C]** derived by arithmetic from committed code ·
> **[M]** measured on this box during this pass ·
> **[?]** claimed in the brief and not yet confirmed.

## Contents

0. The one-paragraph answer
1. The tension, stated exactly
2. Survey — is frozen a ceiling? (demonstrated / against / argued)
3. The middle ground: adapters, layer-wise LRs, **critical periods**
4. Catastrophic forgetting: what protects the inheritance, priced on our box
5. **The unison question — the decisive one**, and the three gates it needs
6. **RECOMMENDATION**, the ranking, and its strongest counterargument
7. **The `PL.*` bakeoff** — arms, costs, gates, decision rule, specs
8. The missing senses: **SMELL, TASTE, VOICE** — design, specs, and the
   measured risk that adding a sense *hurts*
9. The whole-system view: does "encode everything into one latent" scale?
10. Language, the voice he does not have, and the **talkative parent**
11. Is from-scratch vision actually infeasible?
12. Cost — free compute only
13. What we refuse to claim
14. What this makes the machine better at

**The four things a reader short of time should take:** §6.1 (the constitutional
test applied, and the answer), §5.3 (the one measurement a frozen tower cannot
pass), §8.8 (adding a sense is measured to *hurt* unless it is physically
privileged), and §10.4 (what language purity actually costs, in core-hours).

One extra tag, because this document deliberately re-uses a survey rather than
repeating it: **[V*]** means *verified against the primary source during
`D1_CONTROL_ARCHITECTURE.md`'s research pass (2026-08-09) and re-used here
without an independent re-fetch.* Everything tagged [V*] can be re-checked in
that document's §2, which names the paper, the authors and the table. And
**[k]** means *stated from background knowledge and NOT fetched during this
pass* — it is a lead, not a citation, and no spec may quote a [k] number until
someone verifies it.

---

## 0. The one-paragraph answer

**The owner's worry is half right, and the half that is right is the important
half.** Freezing is not a ceiling on *knowledge* — every measured result in the
field says a pretrained tower with a small head is the right way to import
knowledge you cannot afford to train, and Jack's LLM seat is settled correctly
by decree. But freezing **is** a ceiling on *binding*, and binding is the one
thing `SYSTEM.md` makes constitutional. The strongest measured evidence in this
survey is not the frozen-vs-finetuned accuracy gap (OpenVLA: **47.0 % frozen vs
69.7 % finetuned** [V*]); it is M3L's finding that co-training over vision and
touch **improved the vision-only policy at test time** [c] — a representation
that was reshaped by another sense. A welded-shut tower cannot be reshaped by
another sense, by definition, and therefore cannot produce that signature. So
the resolution is not "frozen or plastic" but **where the seam goes**: freeze
the *knowledge*, and make the *interface between senses* the thing that learns.
Concretely, the recommendation in §6 is `frozen tower + trainable per-modality
adapters + a plastic fusion core`, with a **critical-period schedule** as the
challenger arm rather than the default — and, decisively, with the unison gates
(UB.9 "Heard, Not Seen", UB.11's placebo-calibrated ablation matrix) run as an
**admission criterion** on the frozen arm. If the fully-frozen arm cannot pass
UB.9, the fully-frozen architecture is unconstitutional under `SYSTEM.md`'s
"no learning core without unison" clause, and that is a result the ladder must
be able to produce. **It currently cannot: audited 2026-08-09, of 136 registered
specs, exactly three touch plasticity (T5.03, T5.04, T3.10) and NONE of them
tests whether freezing caps a capability.** §7 fixes that.

---

## 1. The tension, stated exactly

`GOAL.md` says two things that are in genuine, not rhetorical, conflict:

> *"All senses, one brain, trained together. … Not bolt-on encoders that
> coexist: a genuinely unified brain where every sense is load-bearing."*

> *"Flexible above all. Frozen pretrained trunks that swap as better models
> ship; a small trained core."*

**Trained together** and **frozen** cannot both be true of the same parameters.
The conflict is not a drafting error; it is a real engineering trade whose two
halves were written for different reasons — the first for the science, the
second for the budget. `docs/CHAMPIONS.md` records the frozen LLM as held
**BY DECREE**, and the vision encoder as **DEFAULT, never defended**. The
decree is the owner's and stands. The default is not a decision at all.

Three sharpenings before any evidence, because they change what the evidence
has to answer:

**(a) "Frozen" is at least four different things, and the literature conflates
them.** They have different costs and different failure modes:

| # | mechanism | what is protected | what can still adapt |
|---|---|---|---|
| F1 | `requires_grad = False` | the weights, absolutely | nothing in the tower |
| F2 | **stop-gradient at the read interface** (2505.23705 [V*]) | the tower from *this* objective's gradients | the tower, by *its own* objective |
| F3 | adapters / LoRA in a frozen tower | the base weights | a low-rank residual per layer |
| F4 | very low layer-wise LR | nothing absolutely | everything, slowly |

`GOAL.md`'s swappability argument only requires **F1 or F2 on the borrowed
tower**. It does not require F1 on Jack's own trunk, and — this is the part the
project has been sloppy about — **F2 is not freezing**. Knowledge Insulation's
whole contribution is that the backbone *keeps training*, just not from the
action head's gradients. `D1_CONTROL_ARCHITECTURE.md` §2.5 already made this
point and it is worth restating as a rule: *the insulation boundary is a
stop-gradient, not `requires_grad=False`.*

**(b) The knowledge argument does not transfer to a tower with no knowledge.**
Three of the four causes the frozen-backbone literature establishes
(random-init gradient shock, objective mismatch, catastrophic forgetting of
pretraining) presuppose pretrained weights worth protecting. Jack's 57M trunk
has none — it is randomly initialised. Freezing it makes a **random-feature
encoder**, and CortexBench measured exactly that case: **random frozen ViT-B
20.4 % vs 47.4 % from scratch** [V*]. So "should we freeze?" has *different
answers for different seats*, and any document that answers it once is wrong:

| seat | pretrained? | freezing means |
|---|---|---|
| Language model (SmolLM2-360M) | yes, massively | protecting knowledge we cannot afford to train — **correct, and settled by decree** |
| Vision encoder (if DINOv2/SigLIP) | yes | protecting knowledge; the live question is whether it *also* blocks binding |
| Vision encoder (current: from-scratch 0.24M) | no | freezing would be a random-feature encoder — never correct |
| Audio encoder (mel favourite) | no (a mel filterbank has no weights to freeze) | not applicable |
| Sensory fusion core | no | freezing is meaningless; this is the thing that must learn |
| The 57M trunk | no | random features — CortexBench's 20.4 % case |

**(c) The unison claim is a claim about the *fusion*, not about the towers.**
This is the reframing that makes the question tractable on our budget. "Trained
together" could mean (i) every parameter in every encoder receives gradient
from every other modality's loss — the maximal reading — or (ii) there exists a
shared representation into which every modality enters, and gradients from each
modality reach the *shared* parameters and the *entry path* of every other
modality. Reading (ii) is what `LEARNING_CORE.md`'s ADMISSION-1 already
operationalises as **U2**: *"there is a named loss term by which modality A's
gradient reaches modality B's encoder."* Note carefully: **U2 as written is
FAILED by a fully frozen tower.** A gradient cannot reach a `requires_grad =
False` encoder. So the constitutional test already on the books excludes the
fully-frozen arm — and nobody has noticed, because U2 has never been run against
a frozen arm. That observation is the single most consequential thing in this
document and §5 turns it into a measurement rather than a syllogism.

---

## 2. Survey — is frozen a ceiling?

Structured as the brief asks: **demonstrated** (a benchmark, a metric, a number
that could have come out the other way) separated from **argued**.

### 2.1 DEMONSTRATED — for frozen / insulated backbones

| result | numbers | what it actually shows |
|---|---|---|
| **Knowledge Insulation**, arXiv:2505.23705 [V*] | π₀ with flow matching alone needs **7.5× as many training steps** as the insulated variant; insulation overhead ≈ **20 % of training time**; control **10 Hz** vs **1.3 Hz** autoregressive; LIBERO 98.0 / 97.8 / 95.6 / 85.8 / 96.0 | **Gradient isolation, not weight freezing.** The mechanism is `sg()` on the backbone's K and V inside the action expert's cross-attention. The backbone *still trains*, from next-token prediction on discrete FAST action tokens (2501.09747 [V*]). The stated cause of damage is that *the action expert is randomly initialised* and its early gradients corrupt pretrained semantics. |
| **RT-2**, arXiv:2307.15818 [V*] | co-fine-tuning vs robot-only: PaLI-X **5B 42 → 44 %**, **55B 52 → 63 %**; **from scratch at 5B: 9 %** | Robot-only finetuning *destroys* web knowledge, and **the penalty grows with scale** (2 points at 5B, 11 at 55B). This is the strongest case for protecting a pretrained tower — and note it is made by *co-training*, not by freezing. |
| **R3M**, arXiv:2203.12601 [V*] | frozen R3M beats from-scratch by **>20 %** across 12 manipulation tasks | A frozen pretrained visual representation genuinely beats training pixels from scratch at low data. |
| **MVP**, arXiv:2210.03109 [V*] | frozen 307M MAE ViT, up to **81 % relative** improvement over from-scratch | Same conclusion, larger tower. |
| **Octo**, arXiv:2405.12213 [V*] | Octo-Small 27M / Octo-Base 93M; head ablation on WidowX **diffusion 83 % vs MSE 35 % vs discrete 18 %** | The **readout-token** interface: readouts *"attend to observation and task tokens … but are not attended to by any observation or task token"*, so a new sense or head attaches while *"wholly retaining the pretrained weights"*. The head's **form** matters far more than its size — 48 points between diffusion and MSE at identical capacity. |
| **Gato**, arXiv:2205.06175 [V*] | 1.2B, 604 tasks; a **1.18B Atari-only specialist beat human level on 44 games to Gato's 23**; a **79M Meta-World specialist hit 96.6 %** across all 50 tasks | A frozen generalist trunk loses to a small specialist on any single task. Relevant to Jack because Jack *is* a single agent in a single world. |
| **LiT — Locked-image text Tuning**, arXiv:2111.07991 [V] | *"locked pre-trained image models with unlocked text models work best"*; **85.2 % zero-shot ImageNet**, **82.5 % ObjectNet** (ViT-g/14) | The strongest for-frozen result in the survey — **and read its setting carefully.** The task is *contrastive alignment of text to an already-correct visual basis*. Freezing wins because the image tower's basis is exactly right for the objective and the text side is what has to move. That is the opposite of Jack's situation, where the downstream objective (survival control in a MuJoCo jungle) is nothing like the tower's pretraining. **LiT is evidence that freezing wins when the frozen basis matches the task, which is the same conditional CortexBench found from the other side.** |

### 2.2 DEMONSTRATED — against frozen

| result | numbers | what it actually shows |
|---|---|---|
| **OpenVLA**, arXiv:2406.09246 [V*] | **frozen vision encoder 47.0 ± 6.9 % vs 69.7 ± 7.2 % fine-tuned** — a ~23-point loss; **last-layer-only 30.3 %**; LoRA rank 32 (97.6M trainable = 1.4 %) **68.2 ± 7.5 %** | The single cleanest frozen-vs-adapted ablation in embodied AI, and it goes **against freezing**. The authors state finetuning the vision encoder is *crucial*, explicitly contrary to prior VLM practice. **And LoRA recovers essentially all of it at 1.4 % of the parameters** — which is the middle ground, measured. |
| **CortexBench / VC-1**, arXiv:2303.18240 [V*] | **random frozen ViT-B 20.4 % vs from-scratch 47.4 %**; adaptation on top of a frozen PVR adds Adroit **59.3 → 72.0**, MetaWorld **88.8 → 96.0**, DMControl **66.9 → 80.9**, ImageNav **70.3 → 81.6**; VC-1 mean 68.7 %, **dominant nowhere** | Three separate findings, all load-bearing here. (i) The brief's 20.4 / 47.4 pair is real and is about a **random** frozen tower, not a pretrained one — it condemns freezing an untrained trunk, not freezing DINOv2. (ii) **Frozen is not enough even when pretrained**: adaptation buys 8–14 points on every suite. (iii) **No universal winner** — R3M best on Adroit/MetaWorld/DMControl, MVP-L on TriFinger/ImageNav/Mobile-Pick, CLIP on ObjectNav. A frozen tower is a *bet on which pretraining matches your world*, and Jack's world (a jungle he will build) matches none of them. |
| **Mechanistic study of VLAs**, arXiv:2603.19233 [c] | six models 80M–7B, 394,000+ rollouts: the **visual pathway dominates**; in all three multi-pathway architectures the expert and VLM pathways occupied **separable activation subspaces**; causal ablation showed **28–92 % zero-effect rates** | Production VLAs built on frozen/insulated towers exhibit **functional dissociation, not integration**. This is the closest thing in the literature to a direct measurement that insulated towers *do not bind*, and it is why §5's design measures binding by intervention rather than by probe. |
| **M3L**, arXiv:2311.00924 [c] | masked autoencoding over vision + touch learned **jointly with the policy** improves sample efficiency and generalisation; and *"representations learned in a multimodal setting also benefit vision-only policies at test time"* | **The single most important result in this document.** The touch channel *reshaped the vision representation*, and the reshaping survived touch's removal. That is the signature of genuine binding — and it is a signature a frozen vision tower cannot produce, because its representation cannot be reshaped by anything. |
| **Cross-modal objectives beat contrastive**, arXiv:2410.16424 [c] | cross-modal *reconstruction* objectives are what force integration; joint masking alone is weaker; input-space modality dropout helps; contrastive is less effective; verified by showing **more cross-modal attention** after pretraining | Binding is produced by a *loss that crosses the modality boundary*, i.e. by gradients flowing between senses. A frozen encoder cannot receive them. |
| **Modality collapse mechanism**, arXiv:2505.22483 [c] (ICML 2025 Spotlight) | collapse occurs when noisy features from one modality entangle, via shared fusion-head neurons, with predictive features from another; fixes work by *"freeing up rank bottlenecks in the student encoder"* | The fix operates **inside the encoder**. If the encoder is frozen, that repair is unavailable and the only remaining lever is the fusion head. |

### 2.3 The synthesis, and the part that is genuinely new here

Reading the two tables together, the field's actual finding is **not** "frozen
wins" and **not** "end-to-end wins". It is three claims that are all true at
once, and they were run together in the brief:

1. **Pretrained beats from-scratch, frozen or not.** R3M +20 %, MVP +81 %
   relative, CortexBench's 20.4 vs 47.4 for random-vs-trained features. This is
   the strongest and least contested result in the survey. It says *borrow
   knowledge* — it says nothing about whether to weld the door shut.
2. **Adapted beats frozen, everywhere it has been measured cleanly.** OpenVLA
   +22.7 points, CortexBench +8 to +14 across four suites. Every clean
   frozen-vs-adapted ablation in embodied AI goes the same way.
3. **Naïve unfreezing destroys the pretraining, and the damage scales.** RT-2's
   11-point gap at 55B, Knowledge Insulation's account of random-init gradient
   shock. This is why insulation exists.

(2) and (3) look contradictory and are not. They are resolved by the mechanism
Knowledge Insulation identified: the damage comes from **a randomly-initialised
head's early gradients**, not from adaptation as such. Fix the gradient path —
stop-gradient, low-rank residual, warmup, low layer-wise LR — and you get
adaptation's gains without forgetting's costs. **OpenVLA's LoRA row is that
statement as a number: 68.2 % at 1.4 % of the parameters against 69.7 % full
and 47.0 % frozen** [V*]. The middle ground is not a compromise; on the only
benchmark that measured all three, it is *within noise of the best arm and 21
points above the frozen one*.

### 2.4 ARGUED, not demonstrated

Stated separately because this project's disease is treating an argument as a
result:

- **"A frozen tower cannot participate in binding."** This is an *argument from
  the definition of a gradient*, supported by circumstantial evidence
  (2603.19233's separable subspaces [c]; M3L's reshaping result [c]; 2410.16424's
  cross-modal-reconstruction finding [c]). **Nobody has run the clean
  experiment** — frozen vs adapted encoders as arms on a pure-synergy binding
  task. §5 and §7 are that experiment. Until it runs, the claim is argued.
- **"Frozen features bottleneck downstream control."** Demonstrated for
  *task accuracy* (OpenVLA, CortexBench). **Not** demonstrated for *binding
  specifically*, and not demonstrated at Jack's scale (~1–6M trunk, not 7B).
- **"Small-scale ranking predicts large-scale ranking."** False in general, and
  `UNIFIED_BRAIN_BAKEOFF.md` §6 already carved the one exception we rely on: a
  **pure-synergy one-bit task is a necessary-condition filter, not a ranking** —
  an arm with no cross-modal pathway does not grow one at scale.
- **"Frozen keeps components swappable."** True, and it is the real reason the
  decree exists. But note what it costs and where: swappability is a property of
  the **interface**, not of the weights. Octo's read-only readout tokens
  [V*] give swappability *with* a trainable trunk. Adapters give swappability by
  keeping the base checkpoint pristine and shipping a small delta. **Freezing is
  one way to buy swappability, and the most expensive one in capability terms.**

### 2.5 The one number that matters most for Jack, and it is not in any paper

Every result above was measured on a tower pretrained on **web-scale data of the
same kind the downstream task uses** (internet images → manipulation from
images). Jack's situation is different in a way that cuts *both* directions and
should be said plainly:

- **Against frozen:** his world is a MuJoCo jungle rendered at low resolution
  with synthesised contact audio. The distribution gap from DINOv2's pretraining
  is larger than any gap in CortexBench, and CortexBench's headline finding was
  that **no PVR dominates and the right one depends on the task**. There is no
  reason to believe any public tower's features are the right basis for
  `ContactAudio`'s modal spectra or for a needs-driven survival policy.
- **For frozen:** he has *no data to adapt with*. OpenVLA adapted on 970k
  demonstrations; we have 2,747 mocap clips [c] and whatever he lives. At our
  data scale, adaptation's 23-point win may simply not be available, and an
  under-trained adapted encoder can be worse than a frozen one. **This is an
  empirical question with a cheap answer, and it is the reason the PL bakeoff's
  CPU stage exists.**

---

## 3. The middle ground

Three families sit between "welded shut" and "everything moves". Only one of
them has a biological story, and — this is the surprise of this section — the
biological story turns out to be about **multisensory integration
specifically**, which puts it on a collision course with §5.

### 3.1 The parameter-efficient family — demonstrated, cheap, boring, and it works

| method | cost | the measured result that matters here |
|---|---|---|
| **LoRA**, arXiv:2106.09685 [V] | a low-rank residual per adapted layer; base weights untouched, so swappability is preserved exactly. GPT-3 175B: **4.7M trainable vs 175,255.8M full FT**, WikiSQL 73.4 vs 73.8, MNLI-m **91.7 vs 89.5**; training VRAM 1.2 TB → 350 GB, checkpoint 350 GB → **35 MB (10,000×)**, **zero added inference latency** (B·A merges into W) | **OpenVLA: 68.2 ± 7.5 % at 1.4 % trainable parameters, vs 69.7 ± 7.2 % full finetune and 47.0 ± 6.9 % frozen** [V*]. On the only embodied benchmark that ran all three arms, LoRA is within noise of the best and **21 points above frozen**. |
| **Adapters**, Houlsby et al. arXiv:1902.00751 [V] | bottleneck MLPs between layers; **3.6 % of parameters trained** | GLUE **80.0 vs 80.4** full fine-tuning — within 0.4 points at 17× fewer trainable parameters |
| **Discriminative LRs**, ULMFiT arXiv:1801.06146 [V] | free — a parameter-group in the optimiser; η_{l−1} = η_l / 2.6 | **18–24 % error reduction** on six datasets; with **100 labelled examples it matches from-scratch training on 100× more data**. `D1_CONTROL_ARCHITECTURE.md` already carries this shape as arm A3 (trunk 3e-6 vs head 3e-4) [c] |
| **Gradual unfreezing** | free | **⚠ direct contrary evidence — see the next row. Do not adopt it as the default schedule.** |
| **Surgical fine-tuning**, Lee et al. arXiv:2210.11466 (ICLR 2023) [V] | free — a choice of *which block* to unfreeze | **Tuning one block beats tuning everything.** Best-block vs full FT: CIFAR-C **82.8 vs 79.9**, Entity-30 **81.2 vs 79.3**, CelebA **86.2 vs 82.2**, CIFAR-Flip **93.8 vs 85.9**. **The rule: input-level shift → FIRST block; feature-level shift → MIDDLE; output-level shift → LAST.** And the finding that reshapes §3.3: **gradual unfreezing ranks 4.71 (first→last) and 4.00 (last→first) against full fine-tuning's 2.71 — it is WORSE than just tuning everything.** **Auto-RGN** (pick the block by relative gradient norm) reaches **rank 1.29 in a single run**, no cross-validation. |
| **Stop-gradient insulation**, arXiv:2505.23705 [V*] | ~20 % of training time | the backbone keeps training from *its own* objective while the action head cannot corrupt it. **This is the mechanism `D1`'s A4/A5 already adopt, and it is not freezing.** |
| **Plasticity injection**, Nikishin et al. arXiv:2305.15555 [V] | **zero** — trainable-parameter count unchanged, predictions unchanged at the moment of injection | freeze the current network, add a zero-initialised trainable residual head pair. **+20 % aggregate score across 57 Atari games** vs other plasticity-loss methods, and it doubles as a *diagnostic*: if injection helps, plasticity loss was the bottleneck. **This is the cleanest available implementation of "open a window on demand".** |

The honest summary of this family: **it is a solved problem with a measured
price, and the price is small.** If the only question were "can we adapt a
pretrained tower cheaply without wrecking it", the answer has been yes since
2019 and Jack should just do it. The reason this document is long is that the
*binding* question (§5) is not answered by any of them.

**Two things in that table should change project practice immediately,
independent of everything else here.** (i) Jack's distribution shift from any
public tower is **input-level** — MuJoCo renders, synthesised contact audio — so
surgical fine-tuning's measured rule says *tune the FIRST block*, i.e. the stem.
That is the cheapest possible adaptation and it is the one the evidence points
at. (ii) **Auto-RGN gets most of the benefit in one run instead of one run per
block** — a ~5× saving in tuning compute, which on 30 Kaggle-hours/week is the
difference between running the experiment and not.

### 3.2 Critical periods — the biology-aligned option

#### 3.2a THE DECISIVE PAPER — and it says the owner is right

**Kleinman, Achille & Soatto, "Critical Learning Periods for Multisensory
Integration in Deep Networks", CVPR 2023, arXiv:2210.04643 [V].** This is the
single most important citation in this document, because it measures exactly
the thing the owner intuited and it measures it *against the frozen-separate-
towers architecture by name.*

Setup: **Split-ResNet-18** — left and right halves of a CIFAR-10 image go to two
separate early pathways whose representations are additively combined. The
deficit blurs one pathway for the first t₀ epochs, followed by **180 more
epochs** of clean, perfectly-paired data.

| finding | number / statement |
|---|---|
| **New metric: Relative Source Variance (RSV) ∈ [−1, +1]** | per-unit measure of whether a unit's variance is driven by source A or B. 0 = uses both; ±1 = single-source |
| **Early blur on one pathway ⇒ permanent single-source polarization** | RSV concentrates at **−1**: units encode only the initially-good source, **permanently, despite 180 subsequent epochs of clean paired data with both sensors working perfectly.** The network never learns to use the recovered sensor |
| **"Dissociation" deficit** (both sensors sharp, but the two crops drawn from *different images* — only the *correlation* is destroyed) | RSV **polarizes to ±1**; each unit becomes single-view, so **no unit can extract synergistic information.** Compared by the authors to strabismic monkeys |
| **Depth is the vulnerability** | *"even deep linear networks exhibit critical learning periods for multi-source integration, while shallow networks do not"* |
| **IT FAILS SILENTLY — the trap** | CIFAR-10 accuracy stays in the **93–94 %** band either way, because the network compensates by leaning on the one good source. **Your loss curve looks fine while fusion is dead.** |
| **It only shows in accuracy when the task genuinely needs synergy** | Multi-View Transformer on **Kinetics-400** (two frames a multiple of 0.33 s apart, so motion is synergistic): **up to 20 % absolute test-accuracy loss** depending on deficit onset |
| **THE FIX, and it is already an arm in our bakeoff** | a **masking-based cross-sensor reconstruction** objective (reconstruct masked patches of one view from the other) + 20 epochs of fine-tuning ⇒ **no critical period at all** — a flat sensitivity curve across every deficit onset, against up to −20 % for the supervised model |

And the authors' own recommendation, quoted because it is as close to a direct
verdict on Jack's settled architecture as the literature contains:

> *"Pre-training different backbones separately on each modality, as advocated
> in some foundational models, may yield representations that ultimately fail to
> encode synergistic information. Instead, training should be performed across
> modalities at the outset."*
>
> *"Analysis 'at convergence' of the learning dynamics of a network are
> irrelevant for sensor fusion, as their fate is sealed during the initial
> transient."*

**Four consequences for Jack, and they are large.**

1. **A permanently frozen encoder is a permanent multisensory deficit in exactly
   the sense this paper measures.** Not by analogy: the deficit is *"this
   source's statistics were wrong or uncorrelated during the window"*, and a
   frozen tower's statistics are someone else's, forever. **The owner's worry is
   not speculation; it has a CVPR paper and a metric.**
2. **The failure is silent, and our current gates would not catch it.** 93–94 %
   accuracy with dead fusion is precisely the regime `UB.11`'s ablation matrix
   was built for — but only if the tasks in the matrix *require synergy*.
   Kinetics needed motion before the damage appeared in accuracy at all. **This
   is independent confirmation of `UNIFIED_BRAIN_BAKEOFF.md`'s design decision
   to build a pure-synergy task (HNS) rather than measure fusion on ordinary
   tasks.**
3. **RSV should be adopted as a first-class metric in this project, today.** It
   is cheap (a variance decomposition over units), it is the direct measurement
   of modality collapse, and it catches what accuracy hides. It belongs in
   `UB.11` alongside cross-modal attention mass and the learned modality mask.
4. **The order in which senses come online matters, and a sense added LATE may
   never integrate.** Direct, testable risk for §8: bolting smell onto a trained
   Jack could produce a sense that is encoded and never used — 2603.19233's
   28–92 % zero-effect regime [c]. **Mitigation, and it is nearly free: wire
   every channel in from step 0, even when it carries nothing** — which is
   exactly the placebo modality `UB.11` already requires, now with a second
   justification.

#### 3.2b The single-source version, and the knob it hands us

**Achille, Rovere & Soatto, arXiv:1711.08856 (ICLR 2019) [V].** All-CNN on
CIFAR-10; the "cataract" deficit is downsample-then-upsample (blur), removed at
epoch t₀ and followed by 160 clean epochs.

- **The window:** blur not removed within the **first 40–60 epochs** ⇒ permanent
  degradation. Persisting 140 epochs takes test error from ~6.4 % to **>18 %**
  (⚠ the 6.4 % figure is the ResNet baseline; the All-CNN curve starts nearer
  8 % — treat the magnitude as ~2–3× rather than exact).
- **Sliding window:** a fixed 40-epoch deficit at varying onset peaks in damage
  around **epoch 30** and falls to ~zero for late onsets — an in-silico replica
  of Olson & Freeman's kitten experiment.
- **Only low-level statistical corruption matters.** Vertical flips and label
  permutation produce **no critical period at all.**
- **Deprivation is milder than corruption.** White-noise inputs damage *less*
  than blur, because blur teaches the false fact *"there is no fine structure"*.
  (Matches dark-rearing lengthening the biological CP.)
- **Fisher Information** rises steeply, peaks as test accuracy plateaus, then
  *decreases* — and deficit sensitivity, fit as exp(−S₄₀), tracks tr(F) closely.
  Layer-wise: remove the deficit at epoch 40 and the network **reorganises**;
  remove it at epoch 100 and layer-wise information is **frozen**. That
  frozen-ness is the operational definition of lost Information Plasticity.
- **The scheduling knob:** **weight decay tunes the window** — none gives a
  *shorter, sharper* critical period; more gives a *longer, softer* one.
- **A direct warning against Phase 0 of any naive schedule:**
  *"pre-training on blurred data can have the opposite effect; i.e., it can
  severely decrease the final performance."*

#### 3.2c The biology, and the one place it says something we cannot copy

*(Hensch 2005 *Nat Rev Neurosci* 6:877–888; Takesian & Hensch 2013 *Prog Brain
Res* 207:3–34; all items below verified by the literature pass.)*

- **What OPENS a window:** maturation of **PV+ GABAergic perisomatic
  inhibition** — a specific circuit, not GABA in general; manipulating it
  triggers premature onset or delays it, bidirectionally. **Otx2** transfers
  non-cell-autonomously into PV cells (Sugiyama et al. 2008, *Cell*
  134:508–520); exogenous Otx2 **accelerates** onset. Elegantly, Otx2 binds the
  perineuronal nets that later close the window.
- **What CLOSES it:** **structural brakes** — perineuronal nets condensing round
  PV cells; myelin-associated **NgR1** (McGee et al. 2005, *Science* 309:2222 —
  NgR1⁻/⁻ mice have normal plasticity that then *fails to close*, persisting at
  P45 and P120); **PirB**. Plus a **functional brake**, **Lynx1** (Morishita et
  al. 2010, *Science* 330:1238) — *Lynx1*⁻/⁻ adults show juvenile-like V1
  plasticity. **Closure is active suppression, not decay**, and opening and
  closing are mechanistically distinct systems.
- **Windows are staggered and modality-specific.** Mouse auditory: tonotopic
  CP **P12–P15** precedes the FM-sweep CP **P31–P38** — same modality, two
  windows. Human (Werker & Hensch 2015, *Annu Rev Psychol* 66:173–196): native
  phoneme sensitivity at **6–7 months**, consonant plasticity closing at
  **10–11 months**, morphosyntax declining sharply after **~age 7**. The
  cleanest human dose–response anywhere: cochlear implantation (Sharma et al.
  2002, n = 104) — deprivation **< 3.5 yr** ⇒ cortical P1 latency normalises
  within 6 months; **> 7 yr** ⇒ **never** normalises; **3.5–7 yr** ⇒ **~50 %**.
- **Binding requires co-presence during the plastic transient — the biological
  statement of §3.2a.** Sadato et al. 1996 (*Nature* 380:526) found blind
  subjects *activate* V1/V2 during tactile discrimination while sighted controls
  *deactivate* them; Cohen et al. 1997 (*Nature* 389:180) showed TMS over
  occipital cortex **induces Braille reading errors** — causal recruitment —
  **and that this does not occur in late-onset blind.** Same deprivation,
  functional reassignment inside the window and mere co-activation outside it.
- **And the thing we cannot copy:** every demonstrated *reopening* in biology is
  **global and unselective** — chondroitinase ABC digesting PNNs (Pizzorusso et
  al. 2002, *Science* 298:1248), fluoxetine (Maya Vetencourt et al. 2008,
  *Science* 320:385), dark exposure (He et al. 2006, *J Neurosci* 26:2951). The
  famous human result, **valproate restoring absolute-pitch learning (Gervain et
  al. 2013, *Front Syst Neurosci* 7:102), is weak and should stop being cited as
  established**: n = 23 in arm 1, VPA 5.09/18 vs placebo 3.50 against chance 3,
  p = 0.02 — and **the crossover arm did not replicate and reversed direction
  (VPA 2.75 vs placebo 3.33, both at chance)**.

> **The asymmetry that decides §3.3: biology has no targeted, reversible,
> per-module plasticity window. Artificial networks do.** Weight decay tunes
> window length; shrink-and-perturb reopens on demand at a chosen λ; plasticity
> injection reopens with unchanged parameter count and *unchanged predictions at
> the moment of injection*; ReDo and continual backprop reopen continuously at
> the unit level; surgical fine-tuning opens a chosen depth band. **Here the
> analogy stops being a metaphor and becomes engineering** — which is exactly
> the "biology is the oracle, not the blueprint" clause doing its job.

#### 3.2d Plasticity dies on its own — the mechanisms and their measured remedies

`T5.04` already names this. The numbers, verified:

- **Dohare et al., *Nature* 632:768–774 (2024) / arXiv:2306.13812 [V].**
  Continual ImageNet, 2,000 sequential binary tasks: accuracy **89 % on early
  tasks → 77 % by task 2000 — down to the level of a linear network**, and poor
  at task 2000 *for every hyperparameter tried*. Online Permuted MNIST:
  **up to 25 % dead units after 800 tasks**, monotonic weight-magnitude growth,
  decaying effective rank. **Adam and dropout significantly EXACERBATE it**
  (Adam causes a dramatic effective-rank collapse); **L2 and shrink-and-perturb
  substantially ease it**; **online normalization surprisingly develops dead
  units after ~200 tasks.** Only **continual backprop** (utility-weighted
  reinitialisation of a small fraction of low-utility units, replacement rate
  ρ ≈ 1e-4) is non-degrading over the full run.
  ⚠ The Slippery-Ant PPO and class-incremental CIFAR-100 magnitudes are
  paywalled and **unverified**.
- **ReDo / dormant neurons, Sokar et al. arXiv:2302.12902 [V].** Dormant
  fraction rises monotonically through training in DQN, DrQ(ε) and SAC, reaching
  **30–50 %** at high replay ratios. Cause isolated as **target
  non-stationarity** (fixed-target CIFAR-10 → dormancy *decreases*; labels
  reshuffled every 20 epochs → dormancy *increases*). **Over-parameterisation
  does not fix it** — 2× and 4× width have "at most a mild positive effect".
  *Good news for a small budget; bad news for "just make it bigger".*
- **Primacy bias, Nikishin et al. arXiv:2205.07802 [V].** Atari-100k, 26 games,
  20 seeds: **SPR + periodic resets IQM 0.478** (CI 0.46–0.51) vs **SPR 0.380**
  (0.36–0.39) — a ~26 % relative IQM gain **from periodically throwing weights
  away.**
- **Shrink-and-perturb, Ash & Adams arXiv:1910.08475 [V].** Warm-starting costs
  generalisation at *identical training accuracy* (ResNet-18/Adam CIFAR-10
  **78.0 → 74.4**, CIFAR-100 **41.4 → 35.0**); logistic regression shows **no
  gap**, so it is a non-convex dynamics effect. Fix: `θ ← λθ + N(0, σ²)`;
  **λ = 0.6, σ = 0.01 performs identically to random init while keeping
  warm-start speed.**
- **Lyle et al. arXiv:2303.01486 [V].** Plasticity loss is tied to loss-landscape
  curvature and **occurs independently of dead units**; parameter norm
  correlates but is not causal. Their intervention bake-off found
  **LayerNorm the single most effective intervention** — in some cases beating
  resetting the final layer — and adding LayerNorm after each hidden layer of
  double DQN improved results **across all 57 ALE games with no retuning.**
  Companion arXiv:2204.09560 defines capacity loss and proposes InFeR.

**Three of these are free and should be adopted regardless of how the PL bakeoff
resolves: LayerNorm everywhere (already `LEARNING_CORE.md` F9), shrink-and-
perturb at task/life boundaries instead of warm-starting, and measuring
dead-unit fraction + effective rank + RSV as first-class metrics.** All three
literatures independently found that **accuracy hides the damage**.

### 3.3 Would a scheduled plasticity window work for Jack — and what opens and closes it?

The design, stated concretely enough to be wrong:

```
PHASE 0  INHERIT      towers frozen (F1) or insulated (F2).
                      Adapters present but zero-initialised, so the network is
                      exactly the frozen network at step 0 and there is no
                      random-init gradient shock (arXiv:2505.23705 [V*]).
                      Every modality channel — including ones with no content
                      yet — is WIRED IN from step 0 (arXiv:2210.04643 [V]).

  OPEN when:  (i) the fusion core has itself converged on the frozen basis
                  (its loss plateau, measured, not guessed) — so the head is no
                  longer random and cannot shock the tower; AND
             (ii) N_frames of Jack's own distribution have been collected
                  (>= the number PL.00 says SSL needs to be well-conditioned).
              The trigger is a MEASUREMENT, not an epoch count.

PHASE 1  ADAPT        adapters/LoRA unlocked; layer-wise LR ramp (stem fastest,
                      deep layers slowest); cross-modal masked prediction ON
                      (the resilience mechanism, arXiv:2210.04643 [V]);
                      replay of the pretraining proxy ON (section 4).

  CLOSE when: the reshaping gain R (section 5.3) stops increasing across a
              measured window, OR the retention metric (section 4) crosses its
              pre-registered floor. Whichever fires first.

PHASE 2  CONSOLIDATE  adapters merged into a frozen checkpoint; plasticity
                      retained ONLY in the fusion core and the heads.
                      Diary and episodic memory continue to be the fast store
                      (hippocampus-vs-cortex, GOAL.md).
```

**What opens it, in one sentence:** *the head has stopped being random and the
data has started being his.* Both are measurable and neither is a guess.

**What closes it, in one sentence:** *adaptation has stopped buying reshaping,
or has started costing retention.* Also measurable.

**The four honest objections, and the second and third are serious enough that
this option enters §7 as a challenger rather than as the recommendation.**

1. **It adds two hyperparameters to a project whose whole method is refusing
   unjustified choices.** Mitigated by making both triggers *measurements* with
   pre-registered thresholds, but not eliminated.
2. **⚠ Gradual unfreezing is measured to be WORSE than just unfreezing
   everything.** Surgical fine-tuning [V] ranks gradual unfreezing at **4.71 and
   4.00 against full fine-tuning's 2.71** across seven distribution-shift
   benchmarks. The naive schedule — thaw layer by layer over time — is not a
   neutral middle ground; it is measurably the *wrong* thing under distribution
   shift. **If a critical-period arm is built, it must open a *chosen block*
   (Auto-RGN, rank 1.29) rather than sweep a thaw-front through the network.**
   This is a direct correction to the obvious design and it was found by
   evidence, not by taste.
3. **⚠ Achille's result cuts both ways, and hard.** Two of his findings attack
   Phase 0 directly: *"pre-training on blurred data can have the opposite
   effect; i.e., it can severely decrease the final performance"*, and
   sensitivity to a deficit **falls to ~zero for late onsets** because Fisher
   Information — Information Plasticity — has already declined. So a window that
   opens late may open **after the critical period has closed** and adapt
   nothing. Worse, Kleinman's dissociation result says a Phase 0 in which one
   modality's statistics are *someone else's* is itself the deficit. **The
   schedule may be self-defeating, and the experiment must be able to say so:
   that is exactly what the reshaping gain R (§5.3) measures.**
4. **The biology is about the same tissue learning, not a borrowed brain being
   thawed.** Nature never freezes a cortex and unfreezes it; an animal's
   critical period opens on a network that has been developing on its *own*
   input throughout. **Phase 0 has no biological analogue**, and the appeal to
   nature that makes this option attractive does not actually cover it. What
   biology *does* license is the opposite lesson — Kleinman's *"training should
   be performed across modalities at the outset"* and Cohen's early-vs-late
   blind dissociation both say: **be co-present early, or never bind.**

**The honest reading of objections 2–4 together is uncomfortable and should be
stated plainly: the evidence supports "adapt from the start, with cross-sensor
reconstruction on" more strongly than it supports "freeze first, thaw later".**
The critical-period arm survives into the bakeoff because it is cheap to run and
because its *opening trigger* (wait until the head is no longer random) targets
a real, measured failure — random-init gradient shock [V*]. But it goes in as a
challenger, and if it loses, the reason will have been visible here first.

---

## 4. Catastrophic forgetting: what protects the inheritance if we unfreeze

Freezing exists because unfreezing forgets. If §6 recommends any plasticity, it
owes an answer to *what protects the borrowed knowledge*, priced in **our** units:
4 shared ARM cores at `nice 19` under ~1.5 GB, Kaggle 30 h/week P100, Colab T4.

### 4.1 The methods, priced

| method | memory cost | compute cost | measured benefit | verdict for Jack |
|---|---|---|---|---|
| **EWC** (Kirkpatrick et al., *PNAS* 114:3521, arXiv:1612.00796 [V]) | **2× parameters** (diagonal Fisher + the anchor θ*). Does *not* grow with task count — sums of quadratics are a quadratic | one Fisher pass per task boundary | **⚠ near zero in our regime.** A-GEM's Appendix F, run explicitly: *"EWC and similar methods perform only slightly better than VAN [vanilla sequential training]"*, and with a ResNet-18 at **3× fewer feature maps** and **a single epoch per task**, **EWC ≈ vanilla on Split CIFAR**. Their conclusion: regularisation approaches *"require over-parameterized architectures and multiple passes over the samples of each task in order to perform well"* [V] | **DO NOT BUILD.** Jack is small and single-pass — the exact regime where EWC is measured to buy nothing. `TrainingPipeline.py` already contains an `EWC.compute_fisher` that the RL path never calls [c]; this is the evidence that it should be deleted rather than wired up. |
| **A-GEM** (Chaudhry et al., arXiv:1812.00420 [V]) | small per-task exemplar buffer | **~100× faster than GEM, ~10× less memory** — i.e. EWC-class overhead | **greatly outperforms EWC / RWALK / vanilla** on Permuted MNIST and Split CIFAR; lowest forgetting among fixed-capacity methods | **the best measured cost/benefit point in the literature for our constraints.** And Jack *already has the buffer*: the diary. |
| **Plain replay / rehearsal** | buffer only | one extra forward/backward per replayed batch | the workhorse behind A-GEM's win | **already constitutional in another guise** — SIESTA wake/sleep consolidation is the Consolidation seat [c], and `GOAL.md` calls sleep replay one of biology's answers. Forgetting protection is therefore **free in architecture, paid only in sample budget.** |
| **L2-SP** (Li, Grandvalet & Davoine, arXiv:1802.01483 [V]) | 1× parameters (the pretrained anchor, which we keep anyway for swappability) | one extra penalty term | **always improves over plain L2 fine-tuning** across Caltech-256, Stanford Dogs, MIT Indoor, from both ImageNet and Places-365 sources — **and the improvement is larger when there is less target data**, which is Jack's situation exactly. L2-SP ≈ L2-SP-Fisher, i.e. **the Fisher weighting adds nothing measurable** | **BUILD THIS.** It is EWC's benefit without EWC's Fisher, one line in the loss, and its measured advantage grows precisely as data shrinks. |
| **Stop-gradient insulation** (2505.23705 [V*]) | 0 | ~20 % of training time | the backbone is protected from the head's gradients while still trainable by its own objective | **already adopted** by `D1`'s A4/A5 and by the PL arms below. Note: it is a *design pattern* realised by adapters/LoRA/plasticity-injection, not a separately benchmarked method — the literature pass found no independent evaluation of it under that name. |
| **Generative replay** | ~2× model + generator training | high | quality degrades over long sequences as the generator replays its own drift [k] | **no.** Wrong trade on 4 ARM cores. |
| **Dynamic architectures** (PROG-NN) | grows **super-linearly** in task count | high | zero forgetting by construction, but **OOMs during training** on Split CUB and Split AWA [V] | **no.** |

### 4.2 The finding that decides §4, and it is a saving

> **At Jack's scale, the expensive forgetting-protection machinery is measured to
> buy nothing, and the cheap machinery is measured to buy most of it.**

EWC — the method everyone reaches for, and the one already half-implemented in
this repo — is *specifically* ineffective for small networks trained with few
passes. The things that work in our regime are **L2-SP** (one penalty term
against a checkpoint we already keep), **replay** (which we already have as the
diary and as sleep consolidation), and **adapters/LoRA** (which insulate the
base weights by construction — there is nothing to forget because nothing moves).

**Adapters are simultaneously the adaptation mechanism and the forgetting
protection.** That is not a coincidence and it is the strongest practical
argument in this whole document for the middle ground: with a frozen base and a
trainable low-rank residual, the inherited knowledge is protected *by the same
structure* that provides the plasticity, at 1.4 % of the parameters and — per
LoRA's own measurements [V] — a **35 MB** checkpoint delta and **zero added
inference latency**.

### 4.3 What we must measure, since we are choosing not to build EWC

Refusing a mechanism obliges us to detect the failure it would have prevented.
`T5.03` (backward transfer) and `T5.04` (plasticity does not die) already exist
and already have the right shape. Three additions, all free:

1. **RSV / source-sensitivity** (§3.2a) — because accuracy hides fusion damage.
2. **Dead-unit fraction and effective rank**, logged every consolidation cycle —
   `T5.04`'s notes already require this; the point here is that Dohare's data
   says **Adam and dropout make it worse** [V], and `LEARNING_CORE.md` F2 already
   bans dropout, so the remaining exposure is the optimiser.
3. **A retention floor as a pre-registered gate**, not a report: the frozen
   tower's *original* capability (a linear probe on ImageNet-style classes, or
   for the LLM a fixed general-knowledge battery) measured before and after the
   plasticity window. `T3.10` — *"Trunk knowledge survives action training"* —
   is already exactly this spec and is currently the only one of the three
   plasticity specs that could catch the failure. **It should be promoted to a
   standing gate on any arm that unfreezes anything.**

---

## 5. The unison question — the decisive one

*(Written before §3 and §4 because it is constitutional and therefore outranks
them: if frozen fails here, no cost argument rescues it.)*

### 5.1 The syllogism, and why it is not enough

The argument is short. `SYSTEM.md`: *"No learning core without unison … its
adoption is VOID until the standing unison gates pass under it."*
`LEARNING_CORE.md` ADMISSION-1 turns that into four checkable requirements, of
which **U2** reads:

> *"There is a **named loss term by which modality A's gradient reaches
> modality B's encoder.** … `LC.01` asserts the gradient is nonzero by finite
> difference: perturb modality A's input, require a nonzero gradient at
> modality B's encoder."*

A `requires_grad = False` encoder returns **exactly zero** on that finite
difference. Not "small" — zero, to machine precision. **So the fully-frozen
architecture fails an admission criterion that is already on the books, and it
fails it by arithmetic rather than by experiment.**

That is a real finding and it should be recorded. But taken alone it is a
lawyer's argument, and this project does not decide by argument. Three honest
objections to it, each of which changes the conclusion:

1. **U2 may be pointing at the wrong parameter.** If the frozen tower is
   preceded/followed by a trainable **stem or adapter**, then "modality B's
   encoder" — the *path by which B enters the shared representation* — is
   trainable, and the finite difference is nonzero. Under this reading
   `frozen tower + adapter` passes U2 and only `frozen tower + no adapter`
   fails. U2's text does not say which it means. **It must be amended to say,
   and §6.4 proposes the amendment.** Note what this reading implies: the
   configuration that fails is *the cheap default we would drift into* — frozen
   tower, concatenate the outputs, train a head.
2. **A nonzero gradient is not binding.** `UNIFIED_BRAIN_BAKEOFF.md` §1.1 is
   emphatic: *encoded is not used*, and 2603.19233 [c] measured 28–92 %
   zero-effect ablation rates in production VLAs whose architectures all
   permitted cross-modal gradient flow. U2 is necessary, never sufficient.
3. **Binding might not require reshaping the encoders at all.** This is the
   serious objection and it deserves its own subsection, because it is the one
   that could vindicate freezing.

### 5.2 Why "Heard, Not Seen" alone CANNOT settle this — a design finding

The brief asks for HNS run with frozen vs adapted encoders as arms. It should
be run, but **it will not discriminate them**, and this needs saying before
anyone spends the CPU hours.

HNS-A is a pure-synergy 2AFC: audio carries object *identity* through the modal
fundamental (`f0 = clip(180/char_size, 80, 4000)`, a deterministic bijection
from radius to pitch); vision carries *which radius is at which position*; the
one bit lives only in the conjunction. Solving it requires computing
`f0 → radius` ∧ `radius → position`. **But both of those are readouts of
features the towers already have.** If a frozen audio front-end exposes band
energies (a mel filterbank exposes f0 trivially) and a frozen vision tower
exposes object scale, then a *trainable fusion head over frozen features* can
compute the conjunction. Nothing needs to be reshaped.

So the expected result is: **frozen encoders pass HNS.** And that is correct
behaviour, not a leak — HNS tests whether a cross-modal pathway *exists*, which
is exactly the necessary-condition filter `UNIFIED_BRAIN_BAKEOFF.md` §6 designed
it to be. It is admission, not arbitration.

The consequence for the bakeoff design is sharp:

> **HNS is the ADMISSION gate for every PL arm — including the frozen one — and
> it is not the discriminator. Two further measurements are needed, and they are
> new.**

### 5.3 Gate 2 — the RESHAPING test (the M3L signature). The measurement a frozen tower cannot pass, by construction.

M3L's result [c] is the cleanest positive evidence that binding is a real
physical process rather than a diagram: training over vision **and** touch
produced representations that *"also benefit vision-only policies at test
time"*. The touch channel changed what the vision encoder computed, and the
change outlived touch's removal.

Turn it into a metric. For modality A (say vision) and partner B (say audio):

```
U_A   = encoder(A) trained alone on the pretext + task objective
M_AB  = encoder(A) trained jointly with B, cross-modal masked prediction
        (arXiv:2311.00924, 2410.16424 [c])
eval both A-ONLY at test time, B replaced by its learned [MISSING-B] token
        (arXiv:2410.03010 [c]; never zeros — zeros measure off-manifold shock)

RESHAPING GAIN  R_A  =  perf(M_AB | A only)  −  perf(U_A)      paired by seed
```

Properties that make this the right test:

- **The frozen arm scores R_A = 0 exactly.** Not "poorly" — zero, because
  `M_AB`'s A-encoder *is* `U_A`'s A-encoder, the same frozen tensor. This makes
  the frozen arm the **analytic null** for the whole measurement. There is no
  cheaper or more honest null in this document.
- **It is a `LESSONS.md`-shaped control.** A control that also passes means the
  test measures nothing: if an *adapted* arm also scores R_A ≈ 0, then binding
  produced no reshaping in our setup and the constitutional worry is empirically
  empty — a result that would **vindicate freezing** and must be equally
  reportable.
- **It has a downstream consequence, so it is not a metric about metrics.**
  `perf(M_AB | A only)` is literally *how well Jack copes when a sense fails* —
  night removes vision, rain removes audio, and in a survival world both happen.
  The reshaping gain is the operational value of unison, expressed as
  robustness. Tie the claim to that and it survives the question "so what?".
- **It is CPU-cheap.** It needs no policy and no controller: `UNIFIED_BRAIN_BAKEOFF.md`'s
  whole point is that binding is a perception claim. Two small encoders per
  modality pair per seed.

**Calibrate the expected effect size down, hard.** Kepler-Encoder's fused-vs-
unimodal force-prediction R² was **0.049 / −0.001 / 0.187 across three robots,
one of them negative, p ≤ 0.012** [c]. Design for a small paired effect (paired
seeds, IQM, bootstrap CI on the paired difference, 2108.13264 [c]) or the
experiment cannot see the thing it is looking for.

### 5.4 Gate 3 — the OUT-OF-BASIS test. Where frozen genuinely caps capability, if it does.

Gate 2 asks whether adaptation *changes* the representation. Gate 3 asks the
harder question: is there information Jack needs that a frozen tower **cannot
expose at all**? This is the "ceiling" in the owner's question, made falsifiable.

Construction, in three pre-registered steps:

1. **Certify the basis.** Train a linear probe on the *frozen* features to
   recover a target attribute (PG.6 already does exactly this for object radius
   and bearing, with pre-registered thresholds R² ≥ 0.8 and ≤ 5° median error).
   Call the attribute **in-basis** if the probe succeeds and **out-of-basis** if
   it sits at chance.
2. **Find an out-of-basis attribute that the world makes load-bearing.**
   Candidates in Jack's actual world, cheapest first: contact-audio *material*
   identity from modal spectra (no public audio tower is trained on
   `ContactAudio`'s synthesis); the 10-scalar touch vector's left/right load
   asymmetry; and — if smell ships (§8) — odour concentration, which no
   pretrained tower has ever seen a single example of.
3. **Run the binding task on that attribute, frozen vs adapted.** If the adapted
   arm learns it and the frozen arm stays at chance, **the frozen basis is a
   measured ceiling** and the number is the size of it.

The reason this is fair to freezing rather than rigged against it: step 1 is a
*pre-registered certification*, so "the frozen tower can't see it" is
established before the binding task runs, not inferred from the binding task's
failure. And there is a **control that must fail**: an attribute certified
**in-basis** must show *no* frozen-vs-adapted gap. If frozen loses on in-basis
attributes too, the experiment is measuring adapter capacity, not basis
coverage, and every other cell is uninterpretable.

### 5.5 The three gates as one admission table

| gate | question | frozen arm's expected result | if frozen fails it |
|---|---|---|---|
| **G1 — UB.9 HNS-A** (existing) | does a cross-modal pathway exist at all? | **passes** — the conjunction is a readout of features both towers already have | the arm has no fusion pathway; excluded, not scored (`UNIFIED_BRAIN_BAKEOFF.md` §6's necessary-condition filter) |
| **G1b — UB.11 ablation matrix + placebo** (existing, standing) | is every sense load-bearing above the placebo column, under the cross-episode SWAP perturbation? | unknown — this is a genuine open question for frozen features | the sense is decorative under that arm; Tier-3 deletion applies to the *arm's wiring*, not to the sense |
| **G2 — RESHAPING (new, PL.4)** | does training with B improve A's own representation? | **R = 0 by construction** — it is the analytic null | frozen fails U2 in substance, not just in text: the senses coexist, they do not co-adapt |
| **G3 — OUT-OF-BASIS (new, PL.5)** | is there load-bearing information the frozen basis cannot expose? | unknown; this is the ceiling question | frozen is a measured capability ceiling and the frozen architecture must be amended |

### 5.6 The verdict rule, fixed in advance

Written now so no number can move it later:

- **Frozen passes G1 and G1b, and G2 ≈ 0 for *every* arm including adapted
  ones** ⇒ *binding does not reshape encoders at our scale.* The owner's worry
  is empirically unfounded; freeze and save the compute. **Record it loudly** —
  it would be the cheapest good news in the project.
- **Frozen passes G1 and G1b; adapted arms show G2 > 0 with a paired CI
  excluding zero; G3 shows no frozen ceiling** ⇒ *frozen is constitutionally
  deficient but practically adequate.* Escalate to the owner
  (`DECISIONS_NEEDED.md`) with both numbers: this is a values call about what
  "trained together" is worth, not a metric call.
- **Frozen fails G1b or G3** ⇒ **frozen is unconstitutional AND capped.** The
  settled decision is overturned on evidence; adapters or a critical period
  become the default and `docs/CHAMPIONS.md`'s vision-encoder seat changes hands
  by verdict.
- **Any arm fails G1** ⇒ excluded from the bakeoff, not scored (a
  designed-to-fail control is not a weak arm — `LESSONS.md`).
- **The LLM seat is out of scope of every clause above.** It is held BY DECREE,
  it is out-of-process, and `GOAL.md`'s LG.00 (*strip the diary and the learned
  core and his answers about his own life must COLLAPSE*) depends on it staying
  frozen. **Nothing in this document proposes unfreezing the language model.**
  If anything, the evidence strengthens the decree: RT-2 measured an 11-point
  loss of web knowledge from robot-only finetuning at 55B [V*], and the whole
  point of the mouth-not-mind principle is that nothing Jack learns may hide in
  there.

---

## 6. RECOMMENDATION

### 6.1 The constitutional test, applied directly

`GOAL.md`, top of file, owner 2026-08-09:

> *"capable of LEARNING EVERYTHING with one interconnected brain … a system
> whose limit is what its world asks of it, not what its architecture allows.
> **Wherever a design choice would make some class of learning permanently
> impossible, that choice is suspect no matter how well it scores.**"*

Apply it literally. **Does a frozen perception tower make some class of learning
permanently impossible?**

**Yes. Exactly one class, and it is not hypothetical — it is arithmetic.**

> **A frozen encoder's representation cannot be reshaped by another sense.**
> The reshaping gain `R` of §5.3 is **identically zero** for a frozen tower, not
> small. There is no data, no schedule and no amount of training that changes
> this, because the parameter that would have to move is not in the graph.

And that class of learning is *demonstrated to be real and useful*, not merely
conceivable:

- **M3L [c]:** vision+touch co-training produced representations that
  *"also benefit vision-only policies at test time"* — the touch channel
  reshaped the vision encoder and the reshaping outlived touch's removal.
- **Kleinman, Achille & Soatto, CVPR 2023 [V]:** whether units encode one source
  or two is decided in the early transient and is **permanent**; separately
  pretrained per-modality backbones *"may yield representations that ultimately
  fail to encode synergistic information"*; and the damage is **invisible in
  accuracy** (93–94 % either way) until the task genuinely requires synergy, at
  which point it costs **up to 20 points**.
- **2410.16424 [c]:** cross-modal *reconstruction* — a gradient that crosses the
  modality boundary — is what forces integration.

And a **third, independent** line arrived from the multimodal-optimisation
literature after this section was drafted (§8.8): Wang, Tran & Feiszli
(CVPR 2020 [V]) measured that **unimodal pretraining "fails to offer
improvements"** to modality imbalance — it lifts every stream ~3 points without
fixing which stream the network actually uses — and that **separate encoders
with late fusion IS the failing baseline**, worse than the best single stream on
**all four** Kinetics combinations. Three literatures, three methods, same
direction.

So under the constitutional test as the owner wrote it, **the fully-frozen
perception architecture is SUSPECT.** Not condemned: suspect is exactly the
right word, and the correct response to suspicion in this project is a bakeoff,
not a rewrite. §7 is that bakeoff. But three things follow immediately and do
not wait for it:

1. **The suspicion is evidence-backed, not a feeling.** The owner's worry has a
   CVPR paper, a metric (RSV), and a measured 20-point effect behind it.
2. **The current default is the worst-placed option**, because "frozen tower →
   concatenate → train a head" is precisely the configuration Kleinman warns
   about, and it is also the one that fails `LEARNING_CORE.md`'s U2 most
   directly.
3. **The mitigation is already an arm in our own bakeoff.** Cross-sensor masked
   reconstruction *abolished* the multisensory critical period in their
   experiments — a flat sensitivity curve where the supervised model lost 20
   points. `UB.10`'s arm A3 is that objective. It should be **on by default in
   every arm**, not one arm among six.

### 6.2 The ranking asked for — fully pure / critical-period / adapters / frozen

| rank | option | the evidence for it | the evidence against it |
|---|---|---|---|
| **1** | **Frozen base + trainable per-modality adapters + plastic fusion core, cross-sensor reconstruction ON from step 0** | OpenVLA: **68.2 % at 1.4 % trainable params vs 47.0 % frozen** [V*]; adapters are *simultaneously* the adaptation mechanism and the forgetting protection (§4.2); LoRA keeps the base checkpoint pristine, so `GOAL.md`'s swappability is fully preserved (**35 MB delta, zero added inference latency** [V]); surgical fine-tuning says Jack's shift is **input-level ⇒ tune the first block** [V], the cheapest possible intervention; L2-SP protects the inheritance at one penalty term and helps *more* when target data is scarce [V] | it is a hedge, and §6.3 is the case that it is the *wrong* hedge on this box |
| **2** | **Fully pure — no pretrained weights in perception; everything learned from his own stream** | **the owner's lean, and the evidence is friendlier to it than expected.** It is the only option that satisfies the constitutional test with nothing left over. It is *co-adapted from step 0*, the one configuration Kleinman shows avoids the multisensory critical period. Data is free and perfectly matched: **1M frames ≈ 19 core-h at 128², ≈ 6 core-h at 64²** (§11.2). **It may be free outright** if a Dreamer-class core wins `LC` — next-frame prediction *is* representation learning, and `dreamer-xs` (1.9M params [c]) already contains the encoder. And §11.3 suggests the frozen tower is the **expensive** option at runtime here | CortexBench: pretrained-frozen beats from-scratch on real-image benchmarks (R3M **+20 %**, MVP **up to +81 % relative** [V*]); the cold-start problem is real — he must bootstrap perception and control simultaneously; and *"whether that transfers to a low-res MuJoCo jungle with a few dozen object types"* is exactly the unmeasured question |
| **3** | **Critical period — start frozen, open a window, consolidate** | biologically motivated; the opening trigger targets a real measured failure (random-init gradient shock [V*]); and **artificial networks genuinely have the targeted, reversible, per-module window control biology lacks** (plasticity injection: zero parameter change, unchanged predictions at injection, **+20 % across 57 Atari games** [V]) | **⚠ three measured objections** (§3.3): gradual unfreezing is *worse* than full fine-tuning under shift (**rank 4.71 / 4.00 vs 2.71** [V]); Achille shows deficit sensitivity **falls to ~zero at late onsets**, so a late window may adapt nothing, and warns *"pre-training on blurred data … can severely decrease the final performance"*; and Phase 0 has **no biological analogue** — nature never freezes a cortex and thaws it |
| **4** | **The current default: fully frozen, no adapters** | cheapest in trainable parameters; maximal swappability; LiT shows freezing wins **when the frozen basis already matches the task** [V] | fails the constitutional test by arithmetic (§6.1); fails `LEARNING_CORE.md` U2 as literally written (§5.1); is the exact configuration Kleinman names; **and its runtime cost on 4 ARM cores may be 7–20× the render cost per frame** (§11.3, to be measured by PL.00) |

**On the owner's lean: the evidence does not contradict it, and on this box it
partially supports it.** Pure is ranked second rather than first only because of
cold-start risk, and **that risk is measurable in a CPU afternoon.** If `PL.00`
finds a frozen ViT costs ~1 s/frame on ARM against a 68 ms render, the frozen
arms are dead on throughput before accuracy is ever discussed and **pure becomes
the recommendation by default.** That is not a rhetorical concession; it is the
pre-registered condition under which the ranking flips, written down before the
measurement.

### 6.3 The strongest counterargument to the recommendation

Stated as forcefully as I can make it, because `SYSTEM.md` requires the price
tag next to the call:

> **Adapters are a hedge that may get the worst of both ends.** You pay the
> frozen tower's *runtime* cost on every frame, forever, on a box with four
> shared cores and paying tenants. You pay adaptation's *training* cost anyway.
> You inherit a feature basis chosen for internet photographs and then spend
> compute bending it toward a MuJoCo jungle. And — this is the sharp part —
> **Kleinman's result says the co-adaptation you are buying with adapters is the
> thing that had to happen at step 0, and adapters bolted onto a tower whose
> statistics were fixed by someone else may be exactly the "pre-trained
> separately per modality" configuration the paper warns produces
> non-synergistic representations.** Meanwhile a 1–2M-parameter encoder trained
> from scratch on his own stream is cheaper at inference, constitutionally
> clean, co-adapted from the first gradient, and — if the world-model core wins
> `LC` — arrives free inside a component already budgeted.

I do not think this counterargument wins, but I think it is close, and the
distance is one measurement wide. **Two numbers decide it, both cheap:**

- **`PL.00`** — ms/frame for each candidate encoder on one ARM core. If the
  frozen tower cannot clear `LEARNING_CORE.md`'s **≥ 5.0 sim-s per real-s**
  throughput floor, it is inadmissible and the argument is over.
- **`PL.02`** — the reshaping gain `R`. If adapters recover most of the
  reshaping that pure gets, the hedge works. If `R_adapters ≈ R_frozen ≈ 0` and
  only `R_pure > 0`, the counterargument was right and the hedge bought nothing
  that mattered.

### 6.4 Three amendments the system should adopt regardless of the verdict

Per `SYSTEM.md` — *is the machine better than I found it?*

1. **Amend `LEARNING_CORE.md`'s U2** so it says which parameter it means:
   *"a named loss term by which modality A's gradient reaches modality B's
   **entry path into the shared representation** (encoder, stem, or adapter),
   verified by finite difference."* As written, U2 excludes every frozen tower by
   arithmetic and nobody noticed, because U2 has never been evaluated against a
   frozen arm. The amendment keeps the criterion meaningful and makes the
   configuration it excludes the *right* one: frozen tower with **no** trainable
   entry path.
2. **Adopt RSV (relative source variance) as a standing metric in `UB.11`**,
   alongside cross-modal attention mass and the learned modality mask. All three
   literatures surveyed here independently found that **accuracy hides fusion
   damage**; RSV is the cheapest direct measurement of it that exists.
3. **Turn on the three free plasticity hygiene measures everywhere**: LayerNorm
   after each hidden layer (measured as the single most effective plasticity
   intervention, improving double DQN across **all 57 ALE games** with no
   retuning [V]); **shrink-and-perturb (λ = 0.6, σ = 0.01)** at life boundaries
   instead of warm-starting [V]; and logging dead-unit fraction + effective rank
   every consolidation cycle (`T5.04` already asks for this — the addition is
   that Dohare measured **Adam and dropout to make it worse** [V]).

And one **deletion** the evidence licenses: `TrainingPipeline.py`'s
`EWC.compute_fisher`, which the RL path never calls [c], should be deleted
rather than wired up. At Jack's scale and single-pass regime, EWC is measured to
be **indistinguishable from vanilla sequential training** [V]. Building it would
cost 2× parameter memory for nothing.

---

## 7. THE PL BAKEOFF — how the freeze/adapt question gets decided

### 7.1 Id hygiene

Checked against the live registry (`from experiments.registry import BY_ID`,
136 ids) **and** against every `Spec(...)` proposed anywhere in `docs/` and
`experiments/`. Prefixes in use: `CU D1 HR LC LF LG LT ME NE PG PS SO T0–T6 UB
W WP X`. **`PL`, `SM`, `TA` and `VO` are all free.** Ids are zero-padded
(`PL.00`, not `PL.0`) so that **no id is a string prefix of another** — the
`ME.11` / `ME.11.0` scar that disabled a spec's module glob (`LESSONS.md`).

### 7.2 What is FIXED across all arms — repairs and invariants, not variables

Inherits `D1_CONTROL_ARCHITECTURE.md` R1–R6 and `LEARNING_CORE.md` F1–F11 in
full. Named additions specific to this bakeoff:

| P | invariant | why |
|---|---|---|
| **P1** | **Cross-sensor masked reconstruction is ON in every arm.** Not an arm; an invariant. | It is the *only* intervention measured to abolish the multisensory critical period [V]. Leaving it off in some arms would mean the bakeoff measures it instead of the freeze question. |
| **P2** | **Every modality channel is wired in from step 0**, including ones with no content yet, and including the **placebo** channel. | arXiv:2210.04643 [V]: a source absent during the early transient may never integrate. This is also what protects smell/taste (§8) from arriving too late to bind. |
| **P3** | **Token budget per modality is equalised across arms.** | `UNIFIED_BRAIN_BAKEOFF.md`: otherwise this compares token counts (arXiv:2601.16667 [c]). |
| **P4** | **Ablation uses the learned `[MISSING-m]` token, never zeros.** | arXiv:2410.03010 [c]; zeros measure off-manifold shock. |
| **P5** | **Paired evaluation** — identical seeds, identical data order, identical eval episodes across arms; IQM and bootstrap CIs on the *paired difference*. | arXiv:2108.13264 [c]. Unpaired 3-seed architecture comparisons resolve nothing at this budget. |
| **P6** | **Every arm logs RSV, dead-unit fraction, effective rank, and cross-modal attention mass.** | Accuracy hides the damage — the one finding all three literatures share. |

### 7.3 The specs

Exact `experiments/registry_expansion.py` format.

```python
    # ── PL: is a frozen tower a ceiling? ────────────────────────────────

    Spec("PL.00", 2, "What each encoder actually costs on THIS box",
         hypothesis="Every candidate perception encoder is measured on one ARM "
                    "core for ms/frame at its native resolution, and at least "
                    "one candidate lets the full loop clear the 5.0 sim-s per "
                    "real-s throughput floor with vision live at 5 Hz.",
         falsified_by="No candidate clears the floor with vision live — then "
                      "vision at 5 Hz is unaffordable on this box at ANY "
                      "freeze setting, and the whole PL question is premature.",
         null_baseline="The measured render cost itself: 68 ms/frame at 128x128, "
                       "185 ms at 320x320 (DIRECTION_AUDIT.md, xvfb+llvmpipe; "
                       "osmesa and egl DO NOT EXIST on this box). An encoder "
                       "cheaper than its own render is free; one costing 10x "
                       "the render is the dominant cost of having eyes.",
         metric="ms_per_frame_x_sim_seconds_per_real_second",
         budget=Budget.CPU, seeds=3, depends_on=["T0.07"],
         control="A no-op identity encoder must sit at ~0 ms and must NOT change "
                 "the measured loop throughput. If it does, the harness is "
                 "measuring something other than the encoder.",
         kills="Any PL arm whose encoder cannot clear the throughput floor - "
               "INADMISSIBLE, not scored (LEARNING_CORE.md ADMISSION-2). This "
               "spec can eliminate the frozen-tower arms in MINUTES, before any "
               "accuracy question is asked, and it is the pre-registered "
               "condition under which the section-6 ranking flips to "
               "fully-from-scratch.",
         notes="Candidates: DINOv2 ViT-S/14 @224 (~4.6 GFLOP, estimated "
               "0.5-1.5 s/frame on ARM - THE NUMBER THAT DECIDES THE BAKEOFF), "
               "SigLIP-base, a dreamer-xs-class CNN @64 (~100 MFLOP), the "
               "current from-scratch 0.24M encoder @64, and a mel filterbank "
               "for audio. Report resident RAM too: SYSTEM.md caps this box at "
               "~1.5 GB and a ViT-B's weights alone are a meaningful fraction. "
               "One frame at 128x128 already costs ~104 env-steps of compute."),

    Spec("PL.01", 4, "ADMISSION: can each arm bind at all? (unison gates)",
         hypothesis="Under each arm, all four ADMISSION-1 criteria hold (U1 "
                    "route audit; U2 finite-difference gradient from modality A "
                    "to modality B's ENTRY PATH; U3 modality dropout supported; "
                    "U4 no modality below 1/|M| of the loss), AND the arm "
                    "solves UB.9 HNS-A above chance with its SWAP-FLIP control "
                    "passing.",
         falsified_by="An arm with a zero U2 finite difference, or at chance on "
                      "HNS-A. Either way it has no cross-modal pathway and is "
                      "EXCLUDED from the bakeoff rather than scored - a "
                      "designed-to-fail control is not a weak arm (LESSONS.md).",
         null_baseline="The FULLY FROZEN arm with NO trainable per-modality "
                       "entry path: its U2 finite difference is EXACTLY ZERO by "
                       "arithmetic, not by measurement. It is the analytic null "
                       "for this gate and it is expected to fail it.",
         metric="min_u2_gradient_x_hns_accuracy", budget=Budget.CPU_LONG,
         seeds=3, depends_on=["PG.6", "PG.7", "UB.9"],
         control="The PLACEBO modality must pass U1/U3 and FAIL to contribute "
                 "any HNS accuracy. A placebo that helps means the fixture "
                 "leaks and every cell of this gate is uninterpretable.",
         kills="The fully-frozen-no-adapter architecture, if it fails U2 - which "
               "would make it UNCONSTITUTIONAL under SYSTEM.md's 'no learning "
               "core without unison' clause. That is a major overturn and it is "
               "decided here, on CPU, for free.",
         notes="EXPECTATION, recorded before the run so it cannot be claimed "
               "afterwards: frozen encoders are expected to PASS HNS-A, because "
               "the f0->radius and radius->position conjunction is a READOUT of "
               "features both towers already expose - nothing needs reshaping. "
               "HNS is therefore the admission filter, NOT the discriminator "
               "(docs/research/FROZEN_VS_PLASTIC.md section 5.2). PL.02 and "
               "PL.03 are the discriminators. Amend LEARNING_CORE.md U2 to say "
               "'entry path' before running this, or the criterion excludes "
               "every frozen arm by wording rather than by evidence."),

    Spec("PL.02", 4, "The RESHAPING test: does another sense change what an encoder computes?",
         hypothesis="For each modality pair (A,B), an encoder for A trained "
                    "JOINTLY with B by cross-modal masked prediction "
                    "outperforms an A-only encoder of matched capacity WHEN "
                    "BOTH ARE EVALUATED ON A ALONE at test time. The reshaping "
                    "gain R = perf(M_AB | A only) - perf(U_A) is positive, "
                    "paired by seed, bootstrap CI excluding zero.",
         falsified_by="R indistinguishable from zero for EVERY arm including "
                      "the adapted ones. Then binding does not reshape encoders "
                      "at our scale, the constitutional worry is empirically "
                      "empty, and FREEZING IS VINDICATED. This outcome must be "
                      "reported as loudly as the other one.",
         null_baseline="The FULLY FROZEN arm, whose R = 0 EXACTLY, by "
                       "construction: M_AB's A-encoder IS U_A's A-encoder, the "
                       "same frozen tensor. The analytic null - no cheaper or "
                       "more honest null exists in this project.",
         metric="reshaping_gain_R", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.1", "PL.00"],
         control="SHUFFLED-PARTNER: train M_AB with B's stream drawn from a "
                 "DIFFERENT episode - correspondence destroyed, marginals and "
                 "temporal statistics preserved. R must collapse to ~0. If "
                 "shuffled-B reshapes A just as well, the gain is capacity or "
                 "regularisation, not binding, and the whole test is void.",
         kills="The claim that a frozen tower can participate in genuine "
               "cross-modal binding. R = 0 by construction is not a poor score; "
               "it is a class of learning made permanently impossible, which is "
               "GOAL.md's own definition of a suspect design choice.",
         notes="This is the M3L signature made into a metric (arXiv:2311.00924: "
               "representations learned with touch 'also benefit vision-only "
               "policies at test time'). It has a downstream meaning, which is "
               "what stops it being a metric about metrics: perf(M_AB | A only) "
               "IS how well Jack copes when a sense fails - night removes "
               "vision, rain masks audio, and both happen in the survival "
               "world. CALIBRATE FOR A SMALL EFFECT: Kepler-Encoder's closest "
               "analogue was R^2 0.049/-0.001/0.187 across three robots with "
               "one NEGATIVE, p<=0.012. Paired seeds, IQM, bootstrap CI on the "
               "paired difference (arXiv:2108.13264), or the experiment cannot "
               "see what it is looking for."),

    Spec("PL.03", 4, "The OUT-OF-BASIS test: is there information a frozen tower cannot expose?",
         hypothesis="There exists a load-bearing attribute of Jack's world that "
                    "a linear probe CANNOT recover from frozen features "
                    "(at chance) and CAN recover from adapted or from-scratch "
                    "features (above a pre-registered threshold), and the "
                    "frozen-vs-adapted gap on a binding task over that "
                    "attribute is significant.",
         falsified_by="Every attribute we can name is either in-basis for the "
                      "frozen tower, or out-of-basis for EVERY arm. Then the "
                      "frozen basis is not a measured ceiling on anything Jack "
                      "needs, and the ceiling worry is unfounded.",
         null_baseline="An attribute certified IN-BASIS by the same probe "
                       "protocol (PG.6 already does exactly this for object "
                       "radius R^2>=0.8 and bearing <=5 deg).",
         metric="out_of_basis_gap", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.6", "PL.00"],
         control="THE CONTROL THAT MUST FAIL: on an IN-BASIS attribute there "
                 "must be NO frozen-vs-adapted gap. If frozen loses on in-basis "
                 "attributes too, this measures adapter capacity rather than "
                 "basis coverage and every cell is uninterpretable.",
         kills="The frozen architecture, if the gap is real: an attribute the "
               "frozen tower cannot expose is a capability ceiling, measured. "
               "Symmetrically, a null result here is the strongest evidence "
               "FOR freezing that this document's design can produce.",
         notes="Candidate out-of-basis attributes, cheapest first: contact-audio "
               "MATERIAL identity from ContactAudio's modal spectra (no public "
               "audio tower has seen this synthesis); left/right load asymmetry "
               "in the 10-scalar touch vector; and - once SM.* ships - odour "
               "concentration, which NO pretrained tower has ever seen a single "
               "example of. Step 1 (certify the basis with a probe) runs BEFORE "
               "the binding task, so 'the frozen tower cannot see it' is "
               "established independently rather than inferred from failure."),

    Spec("PL.04", 4, "FREEZE OR ADAPT: the bakeoff (Score A - binding)",
         hypothesis="At matched data, matched optimiser steps, matched tokens "
                    "per modality and paired seeds, the arms differ in synergy "
                    "gap over the unimodal late ensemble on the 3-task binding "
                    "battery, and the ranking is stable across 3 paired seeds.",
         falsified_by="All admitted arms tie within 1.5 sigma. Then freezing "
                      "costs nothing measurable at our scale and the cheapest "
                      "arm ships - which, given the cost unit below, is the "
                      "frozen one. Report it; do not re-run until it looks "
                      "better.",
         null_baseline="The UNIMODAL LATE ENSEMBLE (independently trained "
                       "per-sense models, predictions averaged) - structurally "
                       "incapable of synergy, so every point above it is joint "
                       "computation by construction. PLUS a random-init FROZEN "
                       "tower as a scored CONTROL (CortexBench measured random "
                       "frozen ViT-B at 20.4% against 47.4% from-scratch).",
         metric="synergy_gap_over_ensemble", budget=Budget.GPU, seeds=3,
         depends_on=["PL.00", "PL.01", "PL.02", "UB.10"],
         control="TWO. (a) The RANDOM-FROZEN control must FAIL the learning "
                 "gate; if random frozen features score as well as pretrained "
                 "frozen ones, the pretraining is contributing nothing and the "
                 "frozen arm's premise is void. (b) Every admitted arm must be "
                 "HURT by the cross-episode SWAP ablation on at least one "
                 "sense; an arm invariant to swapping every sense learned "
                 "marginals, not correspondences, and its score means nothing.",
         kills="Three of four architectures, and one branch of GOAL.md's "
               "'frozen pretrained trunks' sentence. The losers are recorded in "
               "docs/DECISIONS_RESOLVED.md with their numbers and deleted.",
         notes="ARMS, cost declared in TRAINABLE PARAMETERS (Arm.cost; "
               "bakeoff.py returns VOID if a tied arm leaves it undeclared, and "
               "each arm asserts its MEASURED requires_grad parameter count "
               "against the declared value at +-5% or returns VOID). "
               "A0 frozen_entry_only: frozen tower + trainable per-modality "
               "linear entry projection + trainable fusion core (the realistic "
               "cheap default; the no-entry-path variant is excluded at PL.01). "
               "A1 frozen_plus_adapters: A0 + LoRA/adapters on the FIRST block "
               "per surgical fine-tuning's measured rule for input-level shift "
               "(arXiv:2210.11466), block chosen by Auto-RGN in ONE run. "
               "A2 critical_period: A1's structure, adapters zero-initialised "
               "and LOCKED until the opening trigger fires (fusion-core loss "
               "plateau AND N_frames of Jack's own stream collected), then "
               "opened on the Auto-RGN block, then merged and re-frozen when "
               "the reshaping gain stops rising or the retention floor is "
               "crossed. NOT gradual unfreezing - that ranks WORSE than full "
               "fine-tuning (4.71/4.00 vs 2.71, arXiv:2210.11466). "
               "A3 pure_from_scratch: no pretrained weights anywhere in "
               "perception; encoders trained from step 0 on Jack's own stream. "
               "INVARIANTS P1-P6 (see FROZEN_VS_PLASTIC.md 7.2) - in "
               "particular cross-sensor masked reconstruction is ON IN EVERY "
               "ARM, because it is the only measured intervention that "
               "abolishes the multisensory critical period (arXiv:2210.04643), "
               "and every channel including the placebo is wired in from step "
               "0. GATES: (i) PL.00 throughput floor >=5.0 sim-s/real-s, "
               "INADMISSIBLE below it; (ii) PL.01 unison admission; (iii) "
               "bakeoff.py's 3-sigma learning gate; (iv) Gate B - each arm "
               "beats ITS OWN UNTRAINED TWIN by 3 sigma (T2.02's untrained MLP "
               "cleared random by 2.74 sigma against a 3.0 gate, so the plain "
               "gate is nearly clearable by a network that never learned); "
               "(v) matched optimiser steps, max/min <=2.5. Any failure => "
               "VOID, and the question stays open."),

    Spec("PL.05", 4, "FREEZE OR ADAPT: Score B - does it matter downstream?",
         hypothesis="The PL.04 ranking is reproduced when the same stored "
                    "checkpoints are scored on lived outcome - survival time "
                    "gain across lives - at matched WALL-CLOCK rather than "
                    "matched data.",
         falsified_by="The Score A and Score B rankings disagree. That is a "
                      "SPLIT, not a winner: 'binding quality and lived "
                      "competence point different ways at 30 GPU-h/week' is the "
                      "honest report, and it must be written that way.",
         null_baseline="Same nulls as PL.04, re-scored; plus a statue policy "
                       "and a random policy on the same lives.",
         metric="life_gain_at_matched_wallclock", budget=Budget.GPU, seeds=3,
         depends_on=["PL.04"],
         control="A FROZEN-WEIGHTS control (no learning at all) must fail the "
                 "learning gate. If lives get longer without learning, the "
                 "world is doing the work and every arm's score is void.",
         kills="Nothing on its own. It exists so that a PL.04 winner cannot be "
               "adopted on a perception metric alone - CHAMPIONS.md rule 2: "
               "winning the seat's metric while breaking the whole is not "
               "winning.",
         notes="Re-runs run_bakeoff on the SAME stored curves at a wall-clock "
               "budget, exactly as LC.05 does to LC.04. The dual-budget rule is "
               "what stops a sample-efficient-but-slow arm from winning a "
               "decision it cannot afford to cash."),
```

### 7.4 Declared costs, and why this unit

> **Unit: trainable parameters** — `sum(p.numel() for p in ... if
> p.requires_grad)`, asserted against the declared value at **±5 %** with
> `Status.VOID` on mismatch. Consistent with `D1` §5 and `LEARNING_CORE.md`
> §5.5, so the three bakeoffs remain comparable.
>
> **Mandatory reported secondaries, not tie-breaks:** resident parameters, peak
> RSS, and **measured ms/frame on one ARM core** from `PL.00`.

| arm | trainable (**cost**) | resident | derivation |
|---|---|---|---|
| A0 `frozen_entry_only` | **≈ 0.35 M** | tower + 0.35 M | per-modality linear entries (~0.15 M) + fusion core (~0.2 M) [C, order] |
| A1 `frozen_plus_adapters` | **≈ 0.45 M** | tower + 0.45 M | A0 + LoRA r=8 on the first block (~0.1 M) [C, order] |
| A2 `critical_period` | **≈ 0.45 M** | tower + 0.45 M | identical to A1 by construction — only the *schedule* differs |
| A3 `pure_from_scratch` | **≈ 2.0 M** | 2.0 M | `dreamer-xs`-class encoder stack (1.9 M measured [c]) + fusion core |

**Two tie-resolution rules, declared before any number exists**, because
`bakeoff.py` breaks ties by cost and the cheapest arm here is also the
constitutionally weakest:

1. **A tie whose tied set is exactly {A1, A2} is reported as *"the schedule does
   not matter; adaptation does"*.** They have identical cost by construction, so
   the tie cannot be broken by cost and pretending otherwise is the `Arm.cost`
   bug in a new costume (`D1` precedent).
2. **Cost may only break a tie among arms that PASSED PL.01 ADMISSION.** This is
   not special pleading — it is `SYSTEM.md`'s constitutional layer working as
   designed: an arm that cannot bind never reaches the scoring stage, so the
   cheapest *admissible* arm wins, and if A0 is admissible then it genuinely
   deserves to win on cost. **The constitution gates entry; cost breaks ties;
   neither overrides the other.**

### 7.5 The decision rule, mapped before the numbers exist

| outcome | what it means, and what changes |
|---|---|
| **A0 fails PL.00's throughput floor** | the frozen tower is unaffordable on this box. Ranking flips: **A3 pure becomes the recommendation**, and the owner's lean is vindicated on cost rather than on principle. |
| **A0 fails PL.01 (U2 or HNS)** | **the fully-frozen architecture is UNCONSTITUTIONAL** under "no learning core without unison". Escalate to `DECISIONS_NEEDED.md`; `CHAMPIONS.md`'s vision-encoder seat changes hands. A major overturn, decided on CPU. |
| **R ≈ 0 for every arm in PL.02** | binding does not reshape encoders at our scale. **Freezing is vindicated**, the owner's worry is empirically unfounded, and this is the cheapest good news available. Report it loudly. |
| **R > 0 for A1/A2/A3 and = 0 for A0 (expected)** | frozen is constitutionally deficient. Whether it is *practically* deficient is then PL.03's and PL.04's question. |
| **PL.03 finds no out-of-basis gap** | frozen is deficient but not capped. This is a **values call, not a metric call** — escalate to the owner with both numbers. |
| **PL.03 finds a gap** | **frozen is a measured capability ceiling.** Adapters or pure become the default by verdict. |
| **A1 beats A3 by ≥1.5σ on both scores** | the hedge works; borrow the tower, adapt the stem. |
| **A3 beats A1 by ≥1.5σ on either score** | §6.3's counterargument was right; **the tower was never worth carrying.** |
| **A2 beats A1 by ≥1.5σ** | the schedule earns its two hyperparameters, and critical periods enter the project as a real mechanism rather than a metaphor. |
| **any gate fails** | **VOID.** Fix the arm; do not decide. `kills` does not fire. |

**One reading forbidden in advance, whatever the numbers say:** no outcome of
`PL.*` may be used to unfreeze or otherwise modify the **language model**. Its
role is settled by owner decision (§10.2) and it is not in the perception path
this bakeoff arbitrates.

### 7.6 REACHABILITY AUDIT — run before publishing, and it caught something

`LESSONS.md`: *"A dependency graph can quietly make your most important claim
unreachable."* That scar cost the entire unison ladder once
(`UNIFIED_BRAIN_BAKEOFF.md` Finding 1), so the specs above were checked against
the **live** ledger rather than assumed. Results, 2026-08-09 [M]:

| dependency | ledger status |
|---|---|
| `T0.07`, `PG.1`, `PG.5` | **PASS** |
| `PG.6` (camera + radius/bearing probe), `PG.7` (HNS leak certification) | **NOT_RUN** — registered, never executed |
| `UB.9` → `UB.10` → `UB.11` | **NOT_RUN**, chained behind `PG.6`/`PG.7` |

**The first draft of this section had the same disease it was written to avoid.**
`PL.02` — the reshaping test, the single spec that decides the constitutional
question in §6.1 — was parented on `PL.01`, which is parented on `UB.9`, which is
blocked behind two unrun fixtures. **The most important measurement in this
document was unreachable, for reasons that have nothing to do with what it
measures.**

Re-parented, on the same principle `UNIFIED_BRAIN_BAKEOFF.md` used for UB.1–UB.8
— *a spec should depend on what it actually needs*:

- **`PL.02` → `["PG.1", "PL.00"]`.** The reshaping gain is an encoder-pair
  experiment: train A alone, train A jointly with B, evaluate both on A alone.
  It needs no binding fixture, no HNS, no policy and no controller.
- **`PL.03` → `["PG.6", "PL.00"]`** (the probe protocol genuinely needs a camera;
  HNS admission does not gate it).
- **`SM.02` → `["SM.01", "PG.6"]`**, **`TA.02` → `["TA.01"]`** — neither needs
  the unison admission gate to measure its own value claim.

`PL.01` keeps its dependency on `UB.9` because it *is* the HNS admission gate and
cannot exist without it. That is a real dependency, not an accident.

**Resulting reachability — four specs are runnable on this box today:**

| runnable today | blocked, and on what |
|---|---|
| **`PL.00`** (encoder cost — the 20 minutes that can kill the frozen arms) | `PL.01`, `PL.04`, `PL.05` on `PG.6`/`PG.7`/`UB.9`/`UB.10` |
| **`PL.02`** (the reshaping test — **the constitutional question**) | `PL.03`, `SM.02`, `TA.01`, `TA.02` on `PG.6` |
| **`SM.01`** (odour field fixture) | `TA.03` on `UB.11` |
| **`VO.01`** (voice channel fixture) | `VO.02` on `GEN.02` — a second Jack, not a spec |

> **So the owner's question can begin to be answered this week without touching
> Kaggle, and without waiting for the unison ladder to unblock.** `PL.00` +
> `PL.02` together cost **≈ 3.3 CPU-hours** and between them decide (a) whether
> the frozen tower is affordable on this box at all and (b) whether freezing
> costs any measurable binding. **Running `PG.6` and `PG.7` — two registered,
> never-executed, CPU-only fixtures — unblocks nine of the remaining specs**, and
> is the highest-leverage unrelated work available.

---

## 8. The missing senses: SMELL, TASTE, and VOICE

> **OWNER DECISION, 2026-08-09:** every sense a human has is now constitutional
> — sight, hearing, touch, proprioception/balance, **smell**, **taste**, pain,
> temperature, interoception, and **voice** (he must MAKE sound, not only
> receive it). So this section is not "are they worth it". It is *design and
> cost*.

### 8.1 The audit

| sense | status in the repo | gap |
|---|---|---|
| sight, hearing, touch, proprioception | seats in `CHAMPIONS.md`, specs in `PG.*`/`HR.*`/`UB.*` | — |
| temperature | designed: `Thermal` overlay, W0, with an analytic gate (**thermoneutral point 27.55 °C computed from first principles**) [c] | designed, not registered |
| interoception | designed: the seven-variable drive vector in `NEEDS_AND_DEATH.md` [c] | designed, not registered; `DIRECTION_AUDIT.md` already asks that `UB.11` give it **its own row and its own placebo** |
| pain | designed: phasic rectified `−Δi` as a **separate channel** from tonic integrity `i` [c] | designed, not registered, and the split is explicitly *"a live question, not a settled design"* |
| **smell** | **NOTHING. Zero specs of 136.** | everything |
| **taste** | **NOTHING. Zero specs of 136.** | everything |
| **voice** | **NOTHING. Zero specs of 136.** He can hear and (via the LLM) emit text, but he cannot make a sound in his world. | everything |

### 8.2 The scheduling constraint that governs all three, and it comes from §3.2a

Smell and taste have **no referents at W0** — there is no food to rot, no fire
to smell, no plant to poison him. Their content arrives at W1/W2. But
arXiv:2210.04643 [V] says a source absent during the early transient may
**never integrate**, and 2603.19233 [c] measured exactly that outcome in
production systems (28–92 % zero-effect ablation rates).

> **Therefore: wire the channel at W0, add the content at W1.** The odour, taste
> and voice channels are present from step 0 of every training run, carrying
> near-zero signal, with their own tokens and their own stem — indistinguishable
> in shape from `UB.11`'s placebo modality. This costs almost nothing and it is
> the difference between a sense that binds and a sense that is decorative. It
> is invariant **P2** of the PL bakeoff (§7.2) for exactly this reason.

### 8.3 SMELL — what it is for, and why it is not a weak second vision

Olfaction earns its place on three properties vision does not have, and each
maps to a measurable claim:

1. **It passes occlusion.** Odour flows around obstacles; light does not. This
   is the one property that makes smell non-redundant, and it is the basis of
   the value test.
2. **It integrates over time and distance.** A plume carries information about a
   source far outside line of sight, and about *the past* (a decaying trail).
3. **It is intermittent and non-gradient.** Real plumes are turbulent: local
   concentration gradients point the wrong way most of the time. **Infotaxis**
   (Vergassola, Villermaux & Shraiman, *Nature* 445:406–409, 2007 [V]) exists
   because of it — search by maximising expected information gain rather than
   climbing a gradient. Its key measured property: **mean search time scales
   LINEARLY with initial distance**, against the **exponential** dependence
   needed to average out concentration noise for a gradient-follower; and the
   search-time PDF **decays exponentially** rather than with the random walker's
   **1/T²** heavy tail. On real turbulent dye data the search was only **twice
   as long** as under the idealised model. With wind, infotaxis spontaneously
   produces **casting and zigzagging like a moth**.

Property 3 is the interesting one, because it means **a smooth inverse-square
field is not a cheap approximation of smell — it is a different sense.** Three
independent pieces of evidence, and together they settle the fixture design:

- **FlyGym's own shipped demo is the counter-proof.** NeuroMechFly v2's
  `OdorArena` is a static inverse-square field, and the bundled
  `simple_odor_taxis.py` solves it with a **hand-written bilateral-difference
  controller and no learning at all** [V, read from source]. A smooth field is
  not a learning problem.
- **Biology stops working in a smooth plume.** Celani, Villermaux & Vergassola
  (*Phys. Rev. X* 4:041015, arXiv:1411.3507 [V]): moths *"exposed to steady,
  uniform stimuli briefly move upwind, arrest their flight toward the source and
  begin crosswind casting"*, and resume upwind flight only when the stimulus is
  **pulsed**. The intermittency *is* the message.
- **Real plumes are 83–90 % blank.** Farrell et al. (*Environ. Fluid Mech.*
  2:143–169, 2002 [V]) against Jones' field data: intermittency **85.2 % at 2 m,
  90.1 % at 5 m, 83.7 % at 10 m**, with peak/mean concentration ratios of
  **36 / 78 / 112**. An instantaneous gradient in that signal is mostly noise.

**And the learnability ablation this implies has never been published.** Nobody
has trained the same RL agent on (a) a smooth analytic field and (b) an
intermittent plume and reported the delta [V by absence, ~8 targeted searches].
`SM.01`/`SM.02` as designed below would be the first, which is a pleasant
side-effect of doing the fixture honestly rather than a reason to do it.

**What it senses in MuJoCo.** A site on the head (two sites, left and right —
mammals do use bilateral sampling), reading `C` odour channels. Caveman
realism: `C = 4` is enough (**food / decay / smoke / water**), tagged per source,
never chemistry. Sampled at **5 Hz** — which is both the affordable rate and,
pleasingly, inside the 4–12 Hz mammalian sniff band [k]. Per sample Jack
receives `2 × C` concentrations plus their temporal derivative: **12 floats**.

**Three field models, and they are the arms of `SM.01`:**

**Measured on this box (Oracle ARM free tier, one core, NumPy), µs per step [M]:**

| model | config | µs/step | share of a 30 Hz frame |
|---|---|---:|---:|
| static inverse-square, 4 sensors | 8 sources | **12.7** | 0.04 % |
| " | 512 sources | 113.6 | 0.34 % |
| **puff advect + diffuse + sample, 4 sensors** | **500 puffs** | **124** | **0.4 %** |
| " | 5,000 puffs | 1,364 | 4 % |
| " | 50,000 puffs | 13,167 | 40 % — the first config that hurts |
| explicit grid advection-diffusion | 128² | 149 | 0.45 % |
| " | 256² | 615 | 1.9 % |

| arm | model | cost | what it can teach |
|---|---|---|---|
| **O1 `static`** | Σ over sources of `A·exp(−d/λ)`, no wind, no occlusion | **12.7 µs/step [M]** | gradient ascent only. **This is the control that must be beaten**: if O1 is as good as O2/O3, smell is a distance sensor. FlyGym's static field is solved by a hand-written controller with no learning [V]. |
| **O2 `puffs + occlusion`** — **RECOMMENDED** | Poisson puff emission + wind advection + crosswind Gaussian noise (Farrell/Singh); occlusion by **GADEN's exact trick**: a 3σ distance cutoff, then a one-cell-granularity ray-cast line-of-sight per candidate filament | **124 µs/step at 500 puffs [M]** — **0.4 % of a 30 Hz frame** | intermittency, whiffs, blanks, *time-since-last-encounter*, casting/surging — **and occlusion, which is the property that decides whether smell helps or hurts (§8.8)** |
| **O3 `baked CFD`** | offline Navier–Stokes, dumped to HDF5, replayed as an array lookup (FlyGym's `OdorPlumeArena` does exactly this) | ~0 at runtime; hours offline; **zero generalisation to a changing world** | full turbulence — but a jungle Jack rebuilds is exactly the world a baked plume cannot follow |

**The recommendation is O2, and the cost argument is now measured rather than
assumed: 124 µs/step is 0.4 % of a frame on this box.** Singh, van Breugel, Rao
& Brunton (*Nat Mach Intell* 5:58–70, 2023, arXiv:2109.12434 [V]) got
insect-like emergent behaviour — surge, cast, recover, plus learned internal
estimates of head direction and **time since last odour encounter** — from a
**3-number observation** (egocentric wind x, y + local concentration) and a
**64-unit RNN**, at **5M PPO steps ≈ 16 h per seed on 1–4 cores** of a
workstation. Their memory ablation is the relevant design lesson: **memory buys
almost nothing on a constant plume and a great deal on a switching or sparse
one** — i.e. the sense only becomes interesting when the world moves.

**How it enters the brain.** As a first-class modality with a multi-token stem
and a modality-ID embedding, into the same shared trunk as everything else
(never a scalar appended to proprioception — that is the 18-vs-1 token imbalance
`UNIFIED_BRAIN_BAKEOFF.md` already identified as the collapse mechanism). It
gets its own row in `UB.11`'s ablation matrix and its own placebo comparison.

**The value test, and the prediction that makes it falsifiable.** A Jack with
smell must find **occluded** food faster than a no-smell twin — *and must show
little or no advantage when the food is in plain sight.* That conditional is
the whole claim: it is the same shape as the audio literature's finding that
*"audio pays when vision is occluded or ambiguous, and approximately nothing
otherwise"* (ManiWAV, Audio-VLA [c]). A test that only measures the occluded
condition cannot tell "smell works" from "the extra channel helped".

### 8.4 TASTE — the one-trial learner, and the reason it needs a fast path

Taste is not a weak flavour channel. It is the only place in Jack's design where
**one exposure must produce a permanent change**, and standard RL cannot do it.

**The biology — and two corrections to the standard retelling that this
document would otherwise have repeated.**

Garcia & Koelling, *"Relation of cue to consequence in avoidance learning"*,
*Psychonomic Science* 4:123–124 (1966). Rats licked a spout delivering
**saccharin-flavoured** water where each lick also triggered a flashing light
and a clicking relay ("bright-noisy-**saccharin**"). 2 × 2: cue tested
(audiovisual vs gustatory) × US (**illness** — LiCl or X-irradiation — vs
**footshock**). Shocked rats avoided the audiovisual cue; sick rats avoided the
taste. **A double dissociation**, which is why it survived the objection that
illness is merely a stronger US.

> **⚠ CORRECTION 1 — it was NOT one trial.** Per Domjan's 2015 retrospective
> (*Int. J. Comp. Psychol.* 28, doi:10.46867/ijcp.2015.28.01.08 [V]), as of 1981
> the Garcia–Koelling effect had been reported in only three papers **and every
> one used multiple conditioning trials**. The first *one-trial* demonstration
> of the selective-association effect is **Miller & Domjan (1981b)**, *Animal
> Learning & Behavior* 9:339–345.
>
> **⚠ CORRECTION 2 — it had no control groups.** Both the 1966 study and the
> 1968 *Science* paper *"included only groups of rats that receive paired
> presentations"*. Saline controls came with Domjan & Wilson (1972); the
> selective-sensitization objection (Rescorla & Holland 1976) was answered by
> Miller & Domjan (1981a), who showed sensitisation is real but **short-lived**.
>
> The primary 1966 text is paywalled and was **not obtained**; the above is from
> Domjan 2015, who replicated the design himself. **This document had asserted
> "single trial" from memory and was wrong. Recorded rather than quietly fixed.**

**What IS pinned down, and it is enough:**

- **One-trial CTA is real** — Garcia, Kimeldorf & Koelling (1955): aversion after
  a *single* saccharin–radiation pairing. Modern standard: 0.1 % saccharin, one
  pairing with 0.15 M LiCl at 20 ml/kg.
- **The delay tolerance:** **1–6 h reliably**, 6–12 h marginal (Smith & Roll
  1967: 12 h only marginally significant), 24 h reported once (Etscorn &
  Stephens 1973). Riley, Hempel & Clasen (*Psychon. Bull. Rev.* 25:429–441,
  2018): 1–6 h in animals, **up to 7 h in humans**.
- **The contrast that makes it a privileged channel, not slow memory:**
  non-taste conditioning typically fails past **~1 minute**, and sensory
  preconditioning *between two tastes* works only at delays of a few **seconds**.
  The long delay is specific to the taste↔illness pairing.
- **Humans, one trial** — Bernstein (*Science* 200:1302–1303, 1978): 41
  paediatric chemotherapy patients given novel "Mapletoff" ice cream before
  treatment; later choice test **21 %** chose it (ice-cream + chemo) vs **67 %**
  (chemo only) and **73 %** (ice cream only). **And the children knew the drug
  caused the nausea and said so — the aversion formed anyway.** The mechanism is
  not inferential, which settles whether to implement it as an association or as
  reasoning.
- **Retention is months, and decay is driven by extinction, not by a clock**
  (Rosenblum et al., *Learn. Mem.* 10:16, 2003). **Design consequence: aversion
  must be un-learned by *safe re-exposure*, never by a timer.**
- **Latent inhibition** — prior familiarity weakens CTA as an increasing
  function of **frequency × duration** of pre-exposure, and still tracks
  **amount consumed** at fixed time (De la Casa & Lubow 1995). So the learning
  rate should be gated by familiarity, not fixed.
- **Concurrent interference** — Kwok & Boakes: inserting a *second* novel taste
  into the delay window **overshadows** the first aversion in a single trial,
  more strongly when it comes late in the window. **An agent that eats several
  novel things before falling ill should smear credit. That is correct
  behaviour, not a bug.**
- **Neophobia, measured** (Lin, Arthurs & Reilly, *Physiol. Behav.*
  106:515–519, 2012): first exposure to 0.5 % saccharin ~3.5 mL rising to ~13 mL
  by trial 4 — **~3.7×, asymptoting in 2–3 exposures**. But note 30 % Polycose
  showed **no intake neophobia at all**, and a wild-colony field study (Modlinska
  et al., *PLoS ONE* 2016) explicitly reported *"no characteristic symptoms of
  food neophobia… such as sampling the novel food"*. **The "rats take a tiny
  test bite" story is folklore-adjacent; model neophobia as reduced *portion*,
  not as a ritual sampling behaviour.**

**Why this cannot be an ordinary reward channel.** With γ < 1 and a poisoning
delay of thousands of simulated steps, the discounted return carries essentially
no signal back to the eating decision — and even if it did, one sample is not a
gradient. **Taste aversion therefore requires a dedicated mechanism, and it is
one of the two "named, ablatable fast paths" §9.4 budgets for.**

**The design, minimal:**

```
taste vector   t ∈ R^5      sweet · bitter · sour · salt · umami   (caveman, not chemistry)
               emitted ONLY on an ingestion event, into the shared trunk as a modality

taste trace    a bounded FIFO of the last K distinct (t, sim_time) ingestions,
               retained for a LONG window (hours of sim time, not seconds)

illness event  a delayed interoceptive insult (a hit to integrity i / a nausea
               channel) fired D seconds after ingesting a toxic item

the fast path  on an illness event, ONE aversion update binds the illness to the
               TASTE TRACE ONLY — not to what he saw or heard at the time.
               Stored in the diary (attributed, extractive) AND as an aversion
               value on the taste vector, so it crosses death like every other
               diary entry.
```

**The algorithm to build it on, and it already exists.** Not eligibility traces:
TD(λ) weights an event *n* steps back by λⁿ, so bridging thousands of steps
needs λ → 1, which collapses to Monte Carlo and its variance. The right
mechanism is **Temporal Value Transport** (Hung et al., arXiv:1810.06721,
*Nat. Commun.* 10:5223, 2019 [V]): a memory-based agent identifies significant
memory-**read** events and splices the value prediction at the read back onto
the past timepoint that was read from, editing the reward at that past step.
**The structural match is exact: the taste trace is the memory key, and the
malaise is the read event.** RUDDER (arXiv:1806.07857 [V]) is the alternative
via return decomposition.

Calibration for what episodic memory alone buys: MFEC (arXiv:1606.04460 [V]) on
Labyrinth **"Forage and Avoid"** — apples +1, **lemons −1** — reached in
**under 3M frames** what A3C needed **over 40M** for, a ~13× data-efficiency
gain on precisely this shape of task. But note the limit: episodic control gives
one-shot *value assignment* from the same episode's rewards; **it does not give
one-shot credit assignment across an hours-long delay**, which is CTA's actual
difficulty. That gap is why the fast path is a fast path.

**And the honest novelty note:** no published RL agent implements a
taste-specific one-trial associator with an hours-long eligibility window and
latent inhibition [V by absence, ~8 targeted searches across the RL and ALife
literatures]. The nearest relatives are NAGI (arXiv:2207.13583 — food colour
flips mid-life, **88.4 % mean / 95.9 % end-of-sample accuracy**, but feedback is
*immediate*, so it is reversal learning) and Stanley/Bryant/Miikkulainen 2003's
"dangerous foraging". **If `TA.02` runs, it is first — which is a reason to
design it carefully, not a reason to claim anything before it passes.**

**Neophobia is not decoration — it is what makes the mechanism survivable.**
If one bite kills, there is nothing to learn from. So the world must make the
first dose sub-lethal (a pre-registered dose–response curve is part of the
fixture), and Jack should carry a small innate prior toward *sampling* an
unfamiliar taste rather than consuming it. That is one of `GOAL.md`'s
*"innate reflex priors"*, still on the shelf, finally earning a use.

**The value test:** after **ONE** exposure, avoidance of the poisonous item on
the next encounter above a pre-registered rate, **and persistence across a
death**. **The controls that must fail** — and the first is Garcia & Koelling's
own, which is why this design is worth building rather than improvising:

1. **CUE–CONSEQUENCE SWAP.** Pair the illness with an *audiovisual* cue instead
   of a taste. Aversion must **not** form (or must form far more weakly). This
   proves the mechanism is a *selective* prior and not a generic one-shot
   memoriser bolted to the loss.
2. **SHUFFLED TASTE.** Illness paired with a random taste vector must not
   produce avoidance of the actual poison.
3. **PLACEBO TASTE CHANNEL.** A matched-dimension noise channel must not support
   aversion.
4. **And the symmetric control that must PASS:** pairing an audiovisual cue with
   *shock* (a fast, external insult) **must** produce avoidance. If neither
   pairing works, the harness cannot learn one-shot anything and the taste
   result would have been an artifact.

**Ablation:** delete taste and he must eat the poison repeatedly — a
vision-only Jack cannot learn poison identity in one trial, because nothing in
the visual channel distinguishes the toxic plant from its safe twin **by
construction** (the fixture makes them visually identical, which is what makes
this a clean test rather than a colour-discrimination task).

### 8.5 VOICE — the missing effector

Design in §10.5: a small vector of synthesis parameters (fundamental,
brightness, amplitude, duration) driven by the policy and rendered by
`ContactAudio` into the same stereo stream he hears. **Not a symbolic
side-channel.** `ContactAudio` synthesises in microseconds per event [c], so the
cost is the policy's extra action dimensions and nothing else.

Two specs, and the second is gated on `GEN.02`:
**VO.01** — the channel exists and is *audible to a listener* at distance, with
occlusion and attenuation (fixture; a muted emitter must be inaudible, and a
listener behind a wall must hear it attenuated by the amount the fixture
declares). **VO.02** — mutual information between emission and referent
**estimated at the listener's ear**, with a shuffled-channel control that must
destroy coordination. VO.02 is the emergent-signalling test and is also,
conveniently, the same estimator §10.7 uses for the parent's words.

### 8.6 The specs

```python
    # ── SM: smell ───────────────────────────────────────────────────────

    Spec("SM.01", 2, "The odour field obeys its own pre-registered rules",
         hypothesis="An Odour overlay in the Water pattern produces "
                    "concentrations that match the declared field model to "
                    "within 1%: inverse-exponential falloff with distance for "
                    "O1, downwind displacement of the peak proportional to wind "
                    "speed for O2, and non-zero concentration at a receiver "
                    "with NO line of sight to the source (odour passes "
                    "occlusion; light does not).",
         falsified_by="Concentration at an occluded receiver is zero, or the "
                       "wind term does not move the peak - then the field is a "
                       "distance sensor wearing the word 'smell' and no value "
                       "test built on it means anything.",
         null_baseline="A receiver at the same distance with the source "
                       "DISABLED must read the noise floor.",
         metric="field_rule_max_deviation", budget=Budget.CPU, seeds=3,
         depends_on=["PG.1"],
         control="A DELIBERATELY BROKEN variant (wind term dropped) must be "
                 "CAUGHT by the same assertions - else the fixture checker is "
                 "blind and its pass means nothing (the PG.5 precedent).",
         kills="SM.02 and SM.03. A value test on a leaky or trivial field "
               "measures the field.",
         notes="ARMS for the field model, decided by cost since all three can "
               "satisfy the rules above: O1 static exponential (free, O(sources) "
               "per sample); O2 + analytic drifting plume + one mj_ray per "
               "source per sample for occlusion; O3 Farrell-style filaments for "
               "TURBULENT INTERMITTENCY. O1 is the control that must be beaten "
               "in SM.02: if O1 is as good as O2/O3, smell is a distance sensor "
               "and the intermittency literature does not apply to us. Sampled "
               "at 5 Hz (inside the 4-12 Hz mammalian sniff band). C=4 channels "
               "- food, decay, smoke, water - tagged per source, never "
               "chemistry (the caveman standard). Two receiver sites, left and "
               "right of the head, so bilateral comparison is available. "
               "MEASURE the O3 cost before adopting it; O1/O2 are expected to "
               "sit near the fire CA's measured 0.06% of one core."),

    Spec("SM.02", 4, "Smell finds what vision cannot see",
         hypothesis="A Jack with the odour modality reaches OCCLUDED food in "
                    "fewer simulated seconds than an identical no-smell twin, "
                    "AND shows no significant advantage when the same food is "
                    "in plain sight.",
         falsified_by="No advantage when occluded (smell is decorative), OR an "
                      "EQUAL advantage when visible (the channel is helping for "
                      "some reason other than occlusion - extra capacity, a "
                      "distance cue, or a leak).",
         null_baseline="The no-smell twin, identical in every other respect "
                       "including token count; PLUS a PLACEBO odour channel of "
                       "matched dimension carrying noise.",
         metric="occluded_minus_visible_advantage", budget=Budget.GPU, seeds=3,
         depends_on=["SM.01", "PG.6"],
         control="TWO that must fail. (a) SHUFFLED FIELD: odour concentrations "
                 "drawn from a different episode's source layout must give NO "
                 "advantage. (b) The PLACEBO channel must give no advantage. "
                 "And ONE that must pass: with the occluder removed the smell "
                 "and no-smell twins must be statistically indistinguishable.",
         kills="The odour modality. A sense whose ablation column is "
               "placebo-indistinguishable loses its parameters (Tier-3), and "
               "this document carves no exception for a constitutional sense - "
               "constitutional means it EXISTS, not that it is exempt from "
               "earning its wiring.",
         notes="The conditional IS the claim, and it mirrors the measured shape "
               "of the audio result (ManiWAV, Audio-VLA: audio pays when vision "
               "is occluded or ambiguous and approximately nothing otherwise). "
               "A test that only measures the occluded condition cannot "
               "distinguish 'smell works' from 'an extra channel helped'."),

    # ── TA: taste ───────────────────────────────────────────────────────

    Spec("TA.01", 2, "The poison fixture: sub-lethal first dose, visually identical twin",
         hypothesis="Two plant types are IDENTICAL to a visual probe (a "
                    "classifier on rendered frames is at chance) and DISTINCT "
                    "to the taste vector; the toxic one produces a delayed, "
                    "SURVIVABLE interoceptive insult on a first small dose, "
                    "following a declared dose-response curve.",
         falsified_by="A visual probe distinguishes them above chance (then "
                      "TA.02 is a colour-discrimination task), or the first "
                      "dose is lethal (then there is nothing to learn from - "
                      "one-trial learning requires surviving trial one).",
         null_baseline="Chance for the visual probe over the declared "
                       "candidate set.",
         metric="visual_probe_accuracy_x_first_dose_survival",
         budget=Budget.CPU, seeds=3, depends_on=["PG.6"],
         control="A DELIBERATELY COLOUR-CODED variant must be classified WELL "
                 "above chance by the same probe - else the probe is blind and "
                 "its null result is worthless (PG.7's precedent exactly).",
         kills="TA.02.",
         notes="Neophobia rides here: the world must make sampling cheaper than "
               "consuming, and Jack carries a small innate prior toward small "
               "first bites - one of GOAL.md's 'innate reflex priors', finally "
               "used. The delay D between ingestion and illness is declared in "
               "this spec and is the quantity TA.02's difficulty scales with."),

    Spec("TA.02", 5, "Conditioned taste aversion: learning from ONE exposure",
         hypothesis="After exactly ONE ingestion of the toxic plant followed by "
                    "delayed illness, Jack avoids that plant on the next "
                    "encounter above a pre-registered rate, and the aversion "
                    "PERSISTS ACROSS A DEATH via the diary.",
         falsified_by="Avoidance at the base rate after one exposure, or "
                       "aversion that does not survive the life boundary. "
                       "Either way the fastest learning in biology has no "
                       "analogue in this system.",
         null_baseline="Base encounter/consumption rate for the SAFE twin; and "
                       "a standard-RL agent with the same reward and no taste "
                       "trace, which is expected to require many exposures - "
                       "the whole point is that a discounted return cannot "
                       "bridge the delay D.",
         metric="one_trial_avoidance_rate", budget=Budget.GPU, seeds=3,
         depends_on=["TA.01"],
         control="FOUR. Three MUST FAIL: (a) CUE-CONSEQUENCE SWAP - pairing the "
                 "illness with an AUDIOVISUAL cue instead of a taste must "
                 "produce no aversion, or far weaker aversion (Garcia & "
                 "Koelling 1966); (b) SHUFFLED TASTE - illness paired with a "
                 "random taste vector must not produce avoidance of the actual "
                 "poison; (c) the PLACEBO taste channel must not support "
                 "aversion. One MUST PASS: pairing an audiovisual cue with a "
                 "FAST external insult (shock-analogue) MUST produce avoidance "
                 "- if nothing one-shot works, (a) failing proves nothing.",
         kills="The taste fast path. If aversion forms equally to any cue, the "
               "mechanism is a generic one-shot memoriser and the "
               "cue-consequence prior - the thing that makes it BIOLOGICAL "
               "rather than a hack - is not there.",
         notes="Control (a) is the single most beautiful control available to "
               "this project: Garcia & Koelling's 1966 design is ALREADY a "
               "control-that-must-fail, published sixty years before this "
               "ladder existed. Standard RL cannot do this task - with gamma<1 "
               "and D of thousands of steps the credit does not arrive - so a "
               "dedicated fast path is REQUIRED, and it is one of the two such "
               "paths this project budgets for (FROZEN_VS_PLASTIC.md 9.4). "
               "VERIFY Garcia & Koelling 1966 and the CTA delay tolerance "
               "against the primary sources before running; both are currently "
               "carried as [k]."),

    Spec("TA.03", 3, "Taste earns its parameters",
         hypothesis="Ablating the taste modality degrades survival in a world "
                    "containing the visually-identical toxic twin, "
                    "significantly above the PLACEBO column of UB.11.",
         falsified_by="No degradation - taste is decorative and loses its "
                      "wiring (not its constitutional existence: the owner "
                      "ruled the sense EXISTS; this spec decides whether the "
                      "current implementation of it is load-bearing).",
         null_baseline="UB.11's placebo modality column, re-estimated under "
                       "this architecture.",
         metric="taste_ablation_margin_over_placebo", budget=Budget.GPU,
         seeds=3, depends_on=["TA.02", "UB.11"],
         kills="The current WIRING of taste - its tokens, its stem, its fast "
               "path - if the column is placebo-indistinguishable. Not the "
               "sense itself: the owner ruled that it exists.",
         control="In a world with NO toxic plants, the taste ablation must "
                 "produce NO degradation. If removing taste hurts in a world "
                 "where taste is uninformative, the matrix is measuring "
                 "capacity rather than information.",
         notes="Registered so that a constitutional sense still has to earn its "
               "IMPLEMENTATION. GOAL.md's Tier-3 rule and the owner's decree do "
               "not conflict: the decree says he HAS taste; this says our "
               "wiring of it must do something measurable or be rebuilt."),

    # ── VO: voice ───────────────────────────────────────────────────────

    Spec("VO.01", 2, "He can make a sound, and it is heard as a sound in the world",
         hypothesis="A policy-driven emission (f0, brightness, amplitude, "
                    "duration) is rendered by ContactAudio into the shared "
                    "stereo stream, is recoverable by a probe on a LISTENER's "
                    "audio input, and attenuates with distance and occlusion by "
                    "the amounts the fixture declares.",
         falsified_by="The emission is not recoverable at the listener, or does "
                      "not attenuate - then it is a wire between two brains "
                      "wearing the word 'voice'.",
         null_baseline="A MUTED emitter: the listener's probe must be at "
                       "chance.",
         metric="listener_recovery_x_attenuation_error", budget=Budget.CPU,
         seeds=3, depends_on=["PG.5"],
         control="A listener BEHIND A WALL must hear it attenuated by the "
                 "declared amount, and a listener with the emitter muted must "
                 "hear nothing above the noise floor.",
         kills="Any emergent-signalling claim, and the two-way half of the "
               "talkative-parent design (FROZEN_VS_PLASTIC.md 10.5, 10.7).",
         notes="Cheapest constitutional gap in the audit: ContactAudio "
               "synthesises in microseconds per event and the path already "
               "exists. The action space grows by 4 dimensions. Deliberately "
               "NOT a symbolic channel - an emergent protocol must survive "
               "distance, occlusion and the listener's own encoder, and its "
               "information content must be measurable AT THE EAR."),

    Spec("VO.02", 5, "Do two Jacks invent a signal? (gated on a second Jack)",
         hypothesis="With two Jacks in one world and a coordination problem "
                    "that pays only if they act differently, the mutual "
                    "information between an emitter's acoustic output and the "
                    "referent, ESTIMATED AT THE LISTENER'S EAR, rises above the "
                    "shuffled-channel floor, and coordination success rises "
                    "with it.",
         falsified_by="Coordination rises while I(signal;referent) at the ear "
                      "stays at the floor - the pair coordinated through "
                      "something other than the signal (position, timing, turn "
                      "count), which is this field's most common false "
                      "positive.",
         null_baseline="THREE, all cheap and all mandatory (arXiv:1903.05168): "
                       "(i) SCRAMBLED MESSAGES - permute the emission before "
                       "delivery; (ii) UNTRAINED COMMUNICATION PARAMETERS - "
                       "never train the emission head; (iii) a MUTED pair. "
                       "Lowe et al. measured speaker consistency essentially "
                       "UNCHANGED under (i) and (ii): 0.202 default vs 0.198 "
                       "scrambled vs 0.171 untrained on the 2x2 game. Any "
                       "metric that cannot separate those three is measuring "
                       "the shared trunk, not communication.",
         metric="ear_mutual_information_over_scrambled", budget=Budget.GPU,
         seeds=3, depends_on=["VO.01"],
         control="POSITIVE LISTENING, not merely positive signalling: the "
                 "causal influence of communication must exceed its floor. In "
                 "Lowe et al., 89.3% (2x2), 97.9% (4x4) and 99.9% (8x8) of "
                 "games sat within 1.02x of the CIC minimum while LOOKING like "
                 "they communicated. Report the floor and the measured value, "
                 "never the value alone. Diagnostic: with SEPARATE emission and "
                 "action networks their speaker consistency collapses from "
                 "0.510 to 0.124 (4x4), which localises the artifact.",
         kills="Every claim that Jack invented a language.",
         notes="BLOCKED ON GEN.02 (a second Jack), and that is the point: a "
               "lone agent has no reason to signal. STAGE IT CHEAPLY - the "
               "floor of this literature is TABULAR: 2 agents, ZERO "
               "parameters, 2 states/2 signals/2 acts, four Polya urns, "
               "Roth-Erev reinforcement, convergence to a signalling system "
               "with probability 1 (Argiento, Pemantle, Skyrms & Volkov 2009), "
               "measured at ~0.2 s of one CPU core for 10^5 plays. Run that "
               "as the harness check first. The 3x3 game converges only ~90.4% "
               "of the time under basic reinforcement (Barrett 2009); "
               "Roth-Erev WITH FORGETTING fixes it to 100% up to 32 symbols at "
               "no extra cost. EXPECT A HOLISTIC PROTOCOL: compositionality "
               "requires a re-learning bottleneck plus an expressivity "
               "constraint (FROZEN_VS_PLASTIC.md 10.6b), not a bigger "
               "vocabulary."),
```

### 8.7 Cost of the three senses, honestly

| item | where | estimate |
|---|---|---|
| `Odour` overlay O1/O2, 4 channels, 2 sites, 5 Hz | CPU | **expected near-free** — the same shape as the fire CA measured at **0.06 % of one core** [c]. **Measure, do not assume.** |
| `Odour` O3 filaments | CPU | **unknown — the only real cost here. Measure before adopting.** |
| Taste vector + trace + fast path | CPU | free (a 5-vector on ingestion events, a bounded FIFO) |
| Voice emission + render | CPU | microseconds/event [c]; +4 action dimensions |
| SM.01 / TA.01 / VO.01 fixtures | 4 ARM cores | **≈ 1.5 CPU-h total** |
| SM.02 occluded-food value test, 3 seeds | T4/P100 | **≈ 2–3 GPU-h** |
| TA.02 one-trial aversion, 3 seeds + 4 controls | T4/P100 | **≈ 3–4 GPU-h** |
| TA.03 ablation | T4 | **≈ 1 GPU-h** (eval-only on trained arms) |
| **total** | | **≈ 1.5 CPU-h + 6–8 GPU-h** |

The real cost is **not** the senses; it is the **world content** they need —
poisonous plants, occluded caches, spoilage, smoke — and that is already
budgeted as W1/W2 work in `SURVIVAL_WORLD.md`. **The senses are cheap; the
things worth smelling are the project.**

### 8.8 THE RISK NOBODY BUDGETED: adding a sense is measured to HURT

This is the finding that most changes what §8 should do, and it arrived from the
multimodal-learning literature rather than the olfaction one.

**Wang, Tran & Feiszli, "What Makes Training Multi-modal Classification Networks
Hard?", CVPR 2020, arXiv:1905.12681 [V].** On Kinetics, **every one of four
modality combinations was worse than the best single stream**:

| combination | multimodal V@1 | best unimodal | Δ |
|---|---|---|---|
| Audio + RGB | 71.4 | RGB 72.6 | **−1.2** |
| RGB + Optical Flow | 71.3 | RGB 72.6 | **−1.3** |
| Audio + OF | 58.3 | OF 62.1 | **−3.8** |
| Audio + RGB + OF | 70.0 | RGB 72.6 | **−2.6** |

*"In every case, the validation accuracy of naive joint training is
significantly worse than the best single stream model."* And the cleanest
"low-bandwidth channel poisons a rich one" datapoint anywhere: on mini-Sports a
**22.1 %** audio stream dragged a **62.7 %** RGB stream down to **60.2 %**.
Diagnosis: the joint model has *lower* train error and *higher* validation error
— strictly more information, strictly worse generalisation. Corroborated by
OGM-GE (Peng et al., CVPR 2022 oral, arXiv:2203.15332 [V]): on CREMA-D **every**
naive fusion sits *below* audio-only (52.5) — concat 51.7, sum 51.5, FiLM 50.6.
And by Wu et al. (arXiv:2202.05306 [V]), who measured the conditional
utilisation rate directly: on NVGesture **u(RGB | depth) = 0.01**.

**Which mitigations actually work, measured:**

| mitigation | best measured win |
|---|---|
| **Gradient-Blending** (Wang) | **+8.0** (Kinetics OF+A 58.3 → 66.3); turns a −1.2 deficit into a +2.1 surplus |
| **OGM-GE** (Peng) | **+10.2** (CREMA-D 51.7 → 61.9) |
| modality dropout | +2.7 CREMA-D, +1.5 KS — about **a quarter** of OGM-GE's win |
| **separate encoders + late fusion** | **this IS the failing baseline** |
| **unimodal pretrain, then fuse** | *"pre-training fails to offer improvements"* — it lifts everything ~3 points **without fixing the imbalance** |
| plain dropout | **+0.3** — nothing |
| early stopping | **fails** — under-fits the strong stream |

Four consequences, and they touch every part of this document:

1. **This is a THIRD independent line of evidence against separately-pretrained
   frozen per-modality towers.** Kleinman said such backbones *"may fail to
   encode synergistic information"* [V]; Wang measured that unimodal
   pretraining *"fails to offer improvements"* to the imbalance [V]; 2603.19233
   found separable pathway subspaces in production VLAs [c]. **Three
   literatures, three methods, same direction.** §6.1's constitutional
   suspicion is not resting on one paper.
2. **The harm is an optimisation pathology, not an information deficit.** Theory
   says more modalities lower population risk (arXiv:2106.04538 [V, abstract]);
   Huang et al.'s modality-competition result (ICML 2022, arXiv:2203.12221 [V,
   abstract]) proves that jointly-trained late-fusion encoders *"will learn only
   a subset of modalities"* as an optimisation fixed point. **Every mitigation
   that works modifies the gradient balance. Every architectural or generic
   regulariser measured (+0.2, +0.3, ~0, negative) failed.**
3. **Smell helps only if it is *physically privileged*.** The regime split is
   clean: a low-bandwidth channel that is redundant-but-easy **dominates and
   suppresses** the rich stream (CREMA-D); one that is weak-and-noisy **gets
   ignored or poisons** it (Kinetics, mini-Sports); one that carries information
   the rich stream **physically cannot access** wins big — Lee et al.
   (arXiv:1810.10191, ICRA 2019 [V*, secondary]): peg insertion image-only
   **49 %** vs vision+force+proprioception **77 %**, **+28 points**.
   **Therefore: if odour sources are always visible, adding smell is predicted
   to be neutral-to-harmful. The occlusion mechanism is not a nice-to-have; it
   is the thing that decides the sign of the effect.** `SM.02`'s
   occluded-minus-visible design was chosen for a different reason and turns out
   to be the exact right test.
4. **Two additions to the specs, pre-registered now.** (a) `SM.02` and `TA.03`
   must each report a **pre-flight unimodal baseline pair** — train both
   unimodal policies first and record the strength gap, because harm magnitude
   tracks *imbalance*, not modality count. (b) **Gradient-Blending or OGM-GE is
   budgeted from the start**, not added after the first disappointment; modality
   dropout alone buys about a quarter of what they do. And per BalanceBenchmark
   (arXiv:2502.10816 [V, secondary]), **do not over-correct** — driving toward
   *absolute* balance measured worse than moderate balancing.

---

## 9. The whole-system view: does "encode everything into one latent" scale?

The owner asked to *"think of the whole system this way"*, so this section takes
the organising principle itself as the object, not just the freeze/adapt knob.

### 9.1 What the principle actually claims

"Encode everything into one latent" is three separable claims that get shipped
as one:

| # | claim | status |
|---|---|---|
| **P1** | Every sense enters *one shared representation* | constitutional (`SYSTEM.md`), not up for bakeoff |
| **P2** | That representation is a *single fixed-width vector* `z` | an engineering choice — `UNIFIED_BRAIN_BAKEOFF.md`'s contract sets `z ∈ R^32..64` from k readout tokens |
| **P3** | *One* such representation suffices for all downstream use (control, memory, language, planning) | **untested, and the weakest of the three** |

P1 is Jack's constitution. P2 and P3 are the parts that could cap generality,
and they are not the same as freezing — which is why this section exists
separately from §2.

### 9.2 The honest case against P2/P3

- **A bottleneck is a prior about what matters, imposed before you know.**
  `z ∈ R^64` at 5–10 Hz is ~640 bits/s of everything Jack can know about the
  world outside his own joints. It is a strong claim that survival, language
  routing, object identity, thermal context, social presence and odour all fit
  there simultaneously. `UNIFIED_BRAIN_BAKEOFF.md` already flagged the token
  version of this problem (18 proprio tokens against 1 per other sense) and
  2505.22483's collapse mechanism [c] is precisely *entanglement in a shared
  fusion head with no room for modality-specific subspaces*.
- **The one credible non-trunk result is in the survey and is not an arm here.**
  2509.23468 [c] factors the policy into per-representation experts plus a
  learned router, reports beating feature-concatenation on tasks needing
  multimodal reasoning, and is robust to sensor corruption. `UB.10` already
  carries it as arm A5, with the correct pre-commitment: *"if A5 wins, 'one
  brain' as a single trunk is the wrong shape and we should say so."*
- **Biology, the project's own oracle, does not use one latent.** The nervous
  system has many partially-overlapping maps with heavy cross-talk, several
  parallel routes from the same receptor (the retina's ~20 ganglion cell types;
  the "what" and "where" streams), and — the load-bearing case for §8 —
  **olfaction bypasses the thalamus entirely**, projecting to piriform cortex
  and amygdala directly. Nature's answer to "one latent?" is *no, many, wired
  together*. Biology is the oracle, not the blueprint, so this is a nomination
  for a bakeoff arm and not an argument that wins by itself.

### 9.3 The honest case for P2/P3, which is stronger than it looks

- **A bottleneck is what makes ablation meaningful.** UB.16's asymmetry test —
  *zero `z` and perception-dependent tasks must degrade while flat walking must
  not* — is only possible because there is exactly one channel to zero. Replace
  `z` with a dozen private routes and the ablation matrix becomes a combinatorial
  problem nobody on this budget can run. **The single latent is not only an
  architecture; it is the measurement apparatus.** Discard it and the project
  loses the ability to prove its own central claim.
- **A bottleneck is a known driver of abstraction.** Compression under a
  predictive objective is what makes representations abstract rather than
  photographic; this is the whole rationale of the JEPA family and of
  `LEARNING_CORE.md`'s `wm-latent` arm [c].
- **It is swappable and cheap**, and on a free-tier box that is not a small
  virtue.

### 9.4 The synthesis, and the alternative organising principle worth naming

The failure mode of "one latent" is not insufficient *size*; it is insufficient
**structure**. The alternative that best fits both the evidence and Jack's
budget is:

> **ONE SHARED SPACE, MANY ROUTES, ONE MEASURABLE INTERFACE.**
> Keep the single shared representation (P1 — constitutional). Give each
> modality *multiple* tokens with modality-ID embeddings so it can own a
> subspace (already required by `UNIFIED_BRAIN_BAKEOFF.md` §5 as the collapse
> mitigation). Allow a small number of *named, ablatable* fast paths that bypass
> the trunk where biology and latency both demand it — pain, and (§8) taste
> aversion, which must act in one trial and cannot wait for a slow consolidation
> loop. And keep exactly one metered interface, `z`, so UB.11 and UB.16 remain
> runnable.

The fast paths are the interesting part, and they are not a hedge: they are
the same structure that makes one-trial aversion possible in animals (§8.3), and
each one is a named channel with its own ablation column, so it cannot become an
unmeasured shortcut. **Two is the budget.** A third requires a scar.

**What this means for generality.** Per `docs/GENERALITY.md`, five of the twelve
barriers (GEN.01, 02, 05, 09, 11) are about the world and who is in it, not
about the brain. Nothing in this section changes that ordering: **the latent's
shape is not what is between Jack and generality.** The honest ranking of
ceilings, from this document's evidence, is:

1. **the world** (GEN.01 — nothing asks him to be general),
2. **being alone** (GEN.02 — and §10 shows this is also what gates language),
3. **experience efficiency** (GEN.12),
4. …then, some distance below, the freeze/adapt question this document is about.

Saying that plainly is part of the job: **the frozen-weights question is real,
is worth the bakeoff in §7, and is not the main thing holding Jack back.**

---

## 10. Language, the voice he does not have, and the three roles of the LLM

> Owner, 2026-08-09: *"Would he learn new visuals as he lives, and language etc?
> I'm leaning towards giving him a body and ALL senses and having him figure out
> EVERYTHING himself."* And, separately: *"could and should I let Jack make
> sounds and see whether he creates language?"*
>
> And the distinction the owner drew that this section is built on:
> **borrowing words is not borrowing meanings.**

### 10.1 The distinction is correct, and it dissolves most of the apparent conflict

A child does not invent English. They take the word forms from the people
around them and attach **their own** meanings, built from their own experience.
That child is not a puppet. So *"figure out everything himself"* does not
require inventing a language from nothing; it requires that **the meanings
originate in his life rather than in someone else's statistics.**

This is exactly `GOAL.md`'s LG.00 criterion restated: *strip the diary and the
learned core, and his answers about his own life must COLLAPSE, while his
general knowledge survives untouched.* Word forms are general knowledge. Meaning
is lived. The frozen LLM is only a puppet if it supplies the second.

### 10.2 Three roles for the LLM, and only one of them is in the current design

| role | what the LLM does | puppet risk | in the design today? |
|---|---|---|---|
| **(a) MIND** | decides what is true and what to say about Jack's life | **fatal** — this is precisely the failure LG.00 exists to detect | no, and forbidden |
| **(b) MOUTH** | Jack's core fixes the content; the LLM finds words for it | **low but real**: the LLM's priors can leak into *how* the content is framed, and a fluent sentence can smuggle a fact the core never had | **yes — the current design, held BY DECREE** |
| **(c) TALKATIVE PARENT** | the LLM is a **voice in his world**. It speaks *to* him, through his ears, while things happen. He learns which sounds go with which events. | **zero for meaning, by construction** — the words arrive as sensory input like everything else, and their meanings are built by him from co-occurrence | **DECIDED BY THE OWNER, 2026-08-09 — this is now the design.** (b) is demoted to fallback. |

> **OWNER DECISION, 2026-08-09, carved into `GOAL.md`:** the language model is
> **not inside Jack**. It lives in his **world**, as a voice that speaks to him;
> he learns words by hearing them used while things happen, and builds meanings
> from his own life. Rationale: *borrowing words is not borrowing meanings — a
> child does not invent English and is not a puppet for it.* Zero puppet risk by
> construction, and he can be spoken to from day one so his childhood is not
> silent. **The "mouth" framing is now the fallback.**
>
> The rest of this section therefore does **not** argue for or against the frozen
> LLM as an internal component. It asks what the **parent role requires, what it
> costs, and how it fails** — and §10.7 is new, added for that purpose.

**(c) is not an alternative to (b); it is the missing half.** (b) is Jack's
*output* path; (c) is his *input* path. A creature that can speak but has never
been spoken to was the design until today, and stated that way it is obviously
incomplete. `GOAL.md` already says *"their words are teaching — one sentence can
spare him a thousand falls"* — but the only speaker was the owner, who is
intermittent. (c) makes teaching a **continuous property of the world** rather
than an occasional visit.

**What (c) costs architecturally, and this is the important constraint:** the
words must enter **through his existing audio modality**, not through a text
side-channel that bypasses his senses. Otherwise it is not a parent, it is a
label feed, and it violates the one-brain constraint the same way a symbolic
side-channel would. The pipeline already exists: the ASR seat (whisper.cpp, BY
ANALYSIS, 3.8–8.3× measured on this box [c]) is the ear; `ContactAudio` is the
mixing point. **The parent's utterance is rendered into the same stereo stream
Jack hears rocks fall in.** That single decision is what makes (c) constitutional
rather than a cheat, and it costs nothing extra because the path is built.

**The new risk (c) introduces, named before anyone adopts it.** A talkative
parent generated by a model that knows things Jack does not can *narrate the
answer*: "you are hungry", "that berry is poisonous", "the fire is hot". Some of
that is legitimate teaching (GOAL.md permits it explicitly, and the diary
attributes it so trust can be earned and checked). But for **measurement** it is
a confound of the first order: any spec that appears to test Jack's
understanding may instead be testing the parent's narration. The controls are
obvious once named and must be mandatory in every spec that runs with a parent:

- a **MUTE-PARENT twin** (the parent is present but silent), and
- a **SHUFFLED-PARENT twin** — utterances paired with the *wrong* events, same
  words, same rate, same acoustics. **This control MUST fail.** If Jack's
  grounding survives shuffling, he learned the words' distributional statistics
  and not their referents, and the whole claim collapses.

### 10.3 The existence proof, and it is startlingly cheap

**Vong, Wang, Orhan & Lake, "Grounded language acquisition through the eyes and
ears of a single child", *Science* 383(6682):504–511, 2024,
doi:10.1126/science.adi1374 [V].** Head-mounted camera on one child, weekly from
6 to 25 months, **>60 hours of footage** containing **~250,000 word instances**
paired with the video frames of what the child saw as those words were spoken. A
two-module network (vision encoder + language encoder) trained on that —
**and only that** — *"acquires many word-referent mappings present in the
child's everyday experience, enables zero-shot generalization to new visual
referents, and aligns its visual and linguistic conceptual systems."*

Read the number again: **a quarter of a million words, from one child's actual
day, was enough to ground a vocabulary and generalise it.** Not a billion. Not
even a million. This is the single most encouraging result in this document for
the owner's lean, and it is the direct existence proof for role (c).

Two calibrating results on the pure-language side, so the encouragement stays
honest:

- **BabyLM** (the sample-efficient pretraining challenge) sets its tracks at
  **10M words (strict-small)** and **100M words**; models at those budgets
  *lag* RoBERTa-base (125M params, ~30B words) on most tasks [V, from the
  challenge's own framing].
- **BabyBERTa**: a **5M-parameter** model trained on **5M words of
  child-directed speech** matches RoBERTa-base's accuracy on **Zorro**, a
  *vocabulary-limited* minimal-pair benchmark [V via the BabyLM findings; the
  primary paper was not fetched — **verify before quoting in a spec**].

Synthesis: **grounding a vocabulary is a ~10⁵-word problem. Learning English is
a ~10⁷-word problem.** Those are four orders of magnitude apart, and Jack only
needs the first — *provided he can borrow the word forms.*

### 10.4 The cost of full purity, quantified

Assumptions, stated so they can be attacked: one parent utterance per **10
simulated seconds** while co-present, ~10 words each ⇒ **1,200 words per
Jack-day** (a Jack-day is 1,200 sim-s at k = 72, costing **85 core-s** of
physics [c] and, at 64² / 5 Hz, an estimated **138 core-s** of render, §11.2).

| target | words | Jack-days | physics | + 64² vision @5 Hz | verdict |
|---|---:|---:|---:|---:|---|
| **CVCL-scale grounding** (a real vocabulary) | 250K | **208** | **4.9 core-h** | **≈ 13 core-h** | **affordable tonight** |
| BabyBERTa-scale (limited-vocab syntax) | 5M | 4,167 | 98 core-h | ≈ 258 core-h | ~2.7 days on 4 cores — a campaign, not a run |
| BabyLM strict-small | 10M | 8,333 | 197 core-h | ≈ 516 core-h | a week of the whole box; competes with everything else |
| BabyLM 100M / conversational English | 100M | 83,333 | **1,970 core-h** | **≈ 5,160 core-h** | **not affordable. 7 months of one core, on a box with paying tenants.** |
| a child's first four years (~45M words [k]) | 45M | 37,500 | 886 core-h | ≈ 2,320 core-h | not affordable |

Add the cost of *producing* the parent's speech: 250K words ≈ 330K tokens from
SmolLM2-360M on ARM CPU is on the order of **5–9 core-hours** at a plausible
10–20 tok/s [C, order — measure it]. It is a **one-time, cacheable** cost: build
a templated + LLM-varied utterance corpus once, sample it forever. Do not
generate live in the loop.

**So the plain answer to the owner's question about the cost of purity:**

> **A vocabulary grounded entirely in his own life costs about 13 core-hours and
> is available now. English costs somewhere between 250 and 5,000 core-hours and
> is not available on this box at all. Until it works, he cannot talk to you —
> and on the pure path, "until it works" is measured in months of box-time for a
> result that would still be worse than the 360M model already sitting on disk.**

That is the honest price tag the owner asked for.

### 10.5 Jack has no voice — and if he is to have one, it must be acoustic

Audited 2026-08-09: **zero specs for vocalisation.** He has ears (the HR family,
whisper.cpp, `ContactAudio`) and a mouth in the sense of *text out through the
LLM*, but no channel by which he **makes a sound in his world**. That is a real
gap independent of the emergent-language question — a creature that cannot make
a noise cannot call, warn, or be located by another creature.

The design constraint, if it is built: **a continuous acoustic emission into the
same audio modality he already hears** — a small vector of synthesis parameters
(fundamental, formant/brightness, amplitude, duration) driven by the policy,
rendered by `ContactAudio`'s existing synth into the shared stereo stream. **Not
a symbolic side-channel.** Two reasons, both hard:

1. **Constitutional.** A discrete token channel between agents that bypasses the
   ear is a private wire between two brains, and it is exactly the "bolt-on
   encoders that coexist" shape `GOAL.md` forbids.
2. **Scientific.** If the signal goes through the air, it is subject to
   distance, occlusion, noise and the listener's own ASR/audio encoder — which
   means an emergent protocol has to be **robust**, and its mutual information
   with the referent is measurable at the *ear*, where it matters, rather than
   at the *wire*, where it is trivially perfect.

### 10.6 Emergent language — is "let him invent it" as strong as it looks?

*(The literature is in §10.6b, which is deliberately weakly sourced and says so.
This subsection is the architectural assessment, which does not depend on it.)*

The owner's architectural argument is: **a symbol two Jacks invent through use
is grounded by construction — it means something because it did something.
Mapping it to English afterwards is translation, not grounding.** And LG.00
becomes trivially decidable, because an invented symbol cannot have come from
the LLM.

**Is it as strong as it looks? Partly — and the part that is weak is decisive.**

**What is genuinely strong:**
- **The grounding inversion is real.** It does invert the hard problem. The LG
  family currently tries to attach borrowed English to lived experience; an
  invented protocol starts from lived experience and attaches a form to it. That
  is the easier direction, and it is not a trick.
- **The anti-puppet proof becomes free.** An invented signal has provably zero
  probability of having come from SmolLM2. LG.00 stops needing an ablation to
  prove the negative.
- **It is measurable with a clean falsifier**, which is what this project cares
  about most. `I(signal ; referent)` estimated at the *listener's ear*, with a
  **shuffled-channel control that must destroy coordination** — same signal
  statistics, wrong pairing. If coordination survives the shuffle, the agents
  were not communicating.

**What is decisively weak:**
- **It requires at least two agents with a coordination problem, and Jack is
  alone.** This is `GEN.02` — *"of 136 specs, exactly ONE touches other minds"*.
  A lone Jack has **no reason to signal**, so there is nothing to invent.
  Emergent language is therefore not an alternative to the LLM question; it is
  **downstream of solving GEN.02**, which `GENERALITY.md` already rates the
  *cheapest high-value item on its list* ("a second process, not a second GPU").
- **It does not let the owner talk to Jack.** An invented protocol is between
  Jacks. Translating it to English (Andreas, Dragan & Klein, "Translating
  Neuralese", ACL 2017, arXiv:1704.06960 [k]) is a further research problem with
  its own failure modes. The end goal — the owner speaks to Jack and Jack
  answers about his life — is **not served** by an invented language, and is
  served immediately by roles (b) + (c).
- **Emergent protocols are usually holistic, degenerate, or not protocols at
  all** — and this is now measured, not feared (§10.6b): 99 % success on **two
  symbols**; 95 % success communicating about **Gaussian noise**; 100 % on seen
  instances and **25.6 %** on unseen; and causal influence at its floor in
  **89–99.9 %** of games that looked like communication. So even after GEN.02 is
  solved, the default outcome is a handful of unanalysable holistic signals —
  and the *default measurement* of them is a false positive.

**Verdict, revised after the literature came back: worth building, and cheaper
than expected — but still not a language strategy.**

- **The compute objection is dead.** The floor of this literature is *two
  tabular agents with zero parameters*, converging in **~0.2 s of one CPU
  core** (§10.6b(v)). The urn game should be run **now**, as a harness check,
  before any Jack is involved.
- **The GEN.02 objection stands and is the whole thing.** A lone Jack has no
  reason to signal. This is not a caveat; it is the gate.
- **The "it does not let the owner talk to him" objection stands.** An invented
  protocol is between Jacks, and translating it is a further research problem
  whose own literature reports **BLEU 6.08–9.21** — about as translatable as a
  distant natural language, which is to say barely (arXiv:2502.07552 [V]).
- **But the architectural claim came out STRONGER than expected**, and from an
  unexpected direction: §10.6b(iv) shows the owner's "same channel, no dedicated
  wire" instinct is a named position (Quinn 2001) that the field has **not
  revisited in 25 years**, and that the acoustic-emission line and the
  no-dedicated-channel line have **never been joined**. Jack's constitution
  forces him into that intersection for unrelated reasons.

So: **the voice channel (§10.5, `VO.01`) is built now** — it is a missing
sense-organ regardless. `VO.02` is registered and **BLOCKED on GEN.02**, framed
as *"do two Jacks invent signals?"* — a result about social intelligence — and
never as *"is this how Jack gets language?"*, which it is not.

### 10.6b The emergent-communication literature — verified

Three findings, and each one changes a design decision.

**(i) Protocols emerge whenever coordination pays, and this fact is nearly
uninformative.** Success rates of 99–100 % coexist with protocols that are not
languages in any sense:

| result | number |
|---|---|
| Lazaridou, Peysakhovich & Baroni (ICLR 2017, arXiv:1612.07182) [V] | an *agnostic* sender reaches **99 % communication success using 2 symbols**; the best informed sender reaches 100 % success at only **46 % semantic purity** |
| **Bouchacourt & Baroni** (EMNLP 2018, arXiv:1808.10696) [V] | agents trained on ImageNet photos communicate about **pure Gaussian noise vectors at 95 % / 87 %**. Sender–receiver alignment ρ = **0.98**; alignment with the input ρ = **0.33**. And the reliability nobody quotes: the more conceptual "different-image" game succeeded in **19 of 100 seeds** |
| **Kottur et al.** (EMNLP 2017, arXiv:1706.08502) [V] | **100 % on seen instances, 25.6 % on unseen** — the agents built a per-instance codebook across two "dialog" rounds. Compositionality appeared only under **minimal vocabulary AND an agent whose memory was reset each round** |
| Chaabouni et al. (NeurIPS 2019, arXiv:1905.12561) [V] | emergent codes are significantly **anti-Zipfian** — mean message length **26.7–29.4 against an optimal 1.9–3.6** — because length helps the *listener* discriminate. And: **25 of 48 configurations converged at all** |
| Havrylov & Titov (NIPS 2017, arXiv:1705.11192) [V] | forcing the protocol toward natural language costs **43 points of success (95.65 % → 52.51 %)**; the unregularised symbols behave like syllables, not words |

**(ii) The measurement problem is worse than the emergence problem, and there is
a published fix.** **Lowe, Foerster, Boureau, Pineau & Dauphin, "On the Pitfalls
of Measuring Emergent Communication", AAMAS 2019, arXiv:1903.05168 [V]** —
speaker consistency (mutual information between an agent's message and its own
next action) is **essentially unchanged when the messages are scrambled before
delivery, or when the communication parameters are never trained at all**:

| 2×2 game | default | scrambled | comm. params never trained | separate action/message nets |
|---|---|---|---|---|
| speaker consistency | 0.202 ± 0.040 | **0.198 ± 0.038** | **0.171 ± 0.033** | **0.028 ± 0.002** |

and the *causal influence of communication* sits at its floor in **89.3 %
(2×2), 97.9 % (4×4) and 99.9 % (8×8)** of games. The artifact is the shared
trunk: a linear message head on shared features separates by intended action for
free. **Positive signalling does not imply positive listening.** This is why
`VO.02` (§8.6) carries all three of Lowe's nulls as mandatory rather than
optional. *(And note the shape of that finding: it is the same disease as
`UNIFIED_BRAIN_BAKEOFF.md`'s "encoded is not used", in a different field.)*

**(iii) Compositionality has one reliable cause, and it is not an environmental
knob.** It is **a re-learning bottleneck: someone who does not know the language
must learn it, abruptly, under an expressivity constraint.**

- **Kirby, Cornish & Smith, *PNAS* 105:10681–10686 (2008) [V]** — the cleanest
  statement, in humans. Bottleneck **alone** produces *underspecification*: a
  chain collapses from **27 distinct words to 2** ("everything moving
  horizontally = *tuge*"). Bottleneck **plus an ambiguity filter** produces real
  compositional morphology (27 → 12–23 words, structure z = +6.805, p < 0.05).
  **Two competing pressures, not one.**
- **Ren et al., Neural Iterated Learning (ICLR 2020, arXiv:2002.01365) [V]** —
  topsim rises **0.575 → 0.935** over 80 generations; zero-shot generalisation
  **0.136 (no reset) → 0.598 (speaker reset) → 0.847 (both reset)**. Resetting
  the *speaker* is what matters.
- **Li & Bowling (NeurIPS 2019, arXiv:1906.02403) [V]** — the active ingredient
  is **abruptness, not diversity**: reset topsim 0.59 rising (best 0.97) vs
  no-reset 0.51 falling; simultaneous reset beats staggered.
- **Rita et al. (NeurIPS 2022, arXiv:2209.15342) [V]** reframes it as an
  *optimisation* property rather than an environmental one: controlling how far
  the listener is allowed to converge takes generalisation **0.58 → 0.95** and
  topsim **0.22 → 0.42**, with *no* channel bottleneck, no under-parametrisation
  and no population dynamics.
- **Population size does NOT deliver it on its own.** "Emergent Communication at
  Scale" (ICLR 2022, OpenReview `AUGBfDIV9rL` — **⚠ this paper has no arXiv ID;
  any citation giving one is wrong**) found **50 pairs worse than 10 pairs** on
  both generalisation and robustness, and the `best 1 pair` baseline beating 50
  pairs. Rita et al. (ICLR 2022, arXiv:2204.12982) [V] showed the null is an
  **artifact of assuming homogeneous agents** — with log-normal heterogeneity,
  compositionality rises ~22 % with population size and across-seed variance
  falls 2–15× — but even then a heterogeneous population **never exceeds the
  best hand-swept 2-agent pair.**
- **And compositionality is not what generalisation needs anyway.** Chaabouni et
  al. (ACL 2020, arXiv:2004.09124) [V]: generalisation correlates with **input
  space size, ρ = 0.86**, while across **141 settings only 4** show a
  significant correlation between test accuracy and any compositionality metric.
  Worse, the metrics disagree with each other (topsim/posdis **0.08**), and
  topsim is nearly useless outside small symbolic worlds — Yao et al.
  (arXiv:2203.13344) [V] measured its correlation with downstream utility at
  **0.030 / 0.003**, against **0.757 / 0.829** for *translatability*.

**(iv) And the finding that speaks directly to the owner's channel design.** The
owner's instinct — *emit into the same sense you hear, not a symbolic
side-channel* — is a named position in this literature with a 2001 answer and
**no modern successor**:

> **Quinn, "Evolving Communication without Dedicated Communication Channels",
> ECAL 2001 [V]:** *"Artificial Life models have consistently implemented
> communication as an exchange of signals over dedicated and functionally
> isolated channels. I argue that such a feature prevents models from providing
> a satisfactory account of the origins of communication."* Two Khepera robots
> with **8 IR proximity sensors and 2 wheels and nothing else** — no message
> channel of any kind. **27 of 30 runs** evolved a working protocol in which the
> *signal is a movement*, perceived through ordinary proximity sensors, and the
> evolutionary trajectory shows alignment appearing first as a non-communicative
> behaviour and only later being recruited as a signal.

A dedicated channel presupposes what it is supposed to explain. Floreano et al.
(*Current Biology* 17:514–519, 2007) [V] make the same move with an LED ring and
a camera; Grupen et al. (AAMAS 2022, arXiv:2106.11156) [V] with position alone,
reaching **0.375 bits/timestep** of instantaneous coordination against a
baseline maximum of 0.15.

Meanwhile the only modern *acoustic* emergent-communication work — Eloff et al.
(arXiv:2111.02827) [V], which pipes phone sequences through TTS, room impulse
responses and 10 dB background noise — found something Jack should want: **the
acoustic speaker invents redundancy that the discrete speaker never does**
(repeated bigrams per utterance **2.680 vs 1.623**, trigrams **0.935 vs 0.277**)
— an error-correcting code — while the discrete agent is *hurt* by extra channel
capacity under noise (**0.651 → 0.564**). But its speaker does **not hear
itself**, and its channel is still dedicated.

> **The intersection — acoustic/vocal emission, into the same channel the agent
> perceives the world with, in a multi-agent setting with emergent semantics —
> is EMPTY.** Nobody occupies it. Oudeyer 2005 is closest on the vocal axis,
> Quinn 2001 on the no-dedicated-channel axis, and the two lines have never been
> joined. **This is the strongest novelty claim available anywhere in this
> document, and Jack's design walks into it by accident** — because `GOAL.md`
> forbids bolt-on channels for reasons that have nothing to do with linguistics.

**(v) And it is nearly free.** The cheapest demonstrated emergent protocol is
**2 agents, ZERO parameters**: four Pólya urns under Roth–Erev reinforcement, a
2-state/2-signal/2-act Lewis game, **proven to converge with probability 1**
(Argiento, Pemantle, Skyrms & Volkov, *Stoch. Proc. Appl.* 119:373–390, 2009)
[V] — **~0.2 s of one CPU core for 10⁵ plays**, measured in plain Python on this
box [M]. The first hard case is 3×3, which basic reinforcement solves only
**90.4 %** of the time (Barrett 2009), and **Roth–Erev with forgetting fixes it
to 100 % up to 32 symbols at no extra cost.** The cheapest *neural* protocol is
~**700 parameters**. So the honest cost objection to emergent communication is
not compute — it is that **Jack has nobody to talk to.**

### 10.7 What the PARENT role actually requires — and the one property that makes or breaks it

The decision is made; this is the engineering. **The whole mechanism is
CONTINGENCY: a parent narrating "you're cold" *when he actually is cold* is what
teaches. A parent narrating it at random teaches nothing — worse, it teaches
that the word is noise.** Everything below follows from that one sentence.

**R1 — The parent must be STATE-CONDITIONED, not free-running.** Its utterances
are a function of the live world state at that instant: his need vector
(hunger/thirst/temperature/sleep), what he is touching, what is in view, what
just made a sound, what he just did. Concretely, an utterance is generated from
a **structured event record** — `(need_deltas, contact_events, visible_objects,
last_action, audio_events)` — sampled from the sim, not from a language prior.
The LLM's job is to render that record into varied natural English, which is the
one thing it is uniquely good at and the one thing templates do badly.

**R2 — It must be SPARSE and EVENT-TRIGGERED.** A parent that talks constantly
destroys the contingency it is supposed to create: if the word "cold" occurs in
40 % of all windows, its mutual information with being cold collapses. The rate
is a designed parameter, and it should be **triggered by salient events**
(a need crossing a threshold, a first contact with a novel object, a death, a
success) rather than by a clock. Pre-registered default: **≤ 1 utterance per 10
simulated seconds, and never two about the same event.**

**R3 — It must arrive through his EARS.** Rendered into the same stereo stream
as everything else, via the ASR seat. Non-negotiable: a text side-channel is a
label feed, and a label feed is the "bolt-on encoder" shape `GOAL.md` forbids.
It also buys a free and desirable difficulty — **the parent has a position**, so
words get quieter with distance and are masked by rain. Presence becomes
physical, which is exactly what the *company* need requires.

**R4 — It must be ATTRIBUTED in the diary.** `GOAL.md` already requires that
*"his diary records whose advice proved true, so trust in a person can be earned
and checked."* The parent is a person in that sense. A parent that is sometimes
wrong is not a bug — it is the only way trust can be *learned* rather than
assumed, and it is cheap to implement (a controlled error rate).

**R5 — It must be CHEAP and PRE-GENERATED.** ~330K tokens for CVCL scale at
10–20 tok/s on ARM is **5–9 core-hours** [C, order]. Generate a corpus of
utterance templates keyed by event type **once, offline**, sample and slot-fill
at runtime. **Never call the LLM inside the loop** — `SYSTEM.md`'s throughput
floor (≥ 5 sim-s per real-s) forbids it, and T0.07 already measured what a
model in the control path does to throughput [L].

**The five failure modes, each with the control that detects it:**

| failure | what it looks like | control |
|---|---|---|
| **Non-contingent narration** | words uncorrelated with state; he learns distributional co-occurrence, not reference | **SHUFFLED-PARENT twin** (same words, same rate, wrong events) — **must destroy grounding.** If it does not, nothing was grounded |
| **Over-talking** | high utterance rate; every word's MI with its referent falls | pre-register the rate; report **I(word ; referent)** per word, and the rate at which it was produced |
| **Answer leakage** | the parent narrates a discovery Jack never made ("that berry is poisonous"), and a downstream spec scores *his* understanding when it measured *its* narration | every spec that runs with a parent MUST also run a **MUTE-PARENT twin**; a capability that only exists with the parent talking is the parent's, not Jack's |
| **Vocabulary the world cannot ground** | the LLM emits words for things with no referent in a MuJoCo jungle ("Tuesday", "expensive") | restrict generation to a **closed event ontology**; log the fraction of emitted word types that have no referent — it is a fixture-health metric |
| **A parent who is never wrong** | trust becomes a constant, and `GOAL.md`'s "advice proved true" machinery has nothing to measure | inject a declared error rate; require the diary's trust estimate to track it |

**And the measurement that makes the whole thing a claim rather than a hope:**
`I(word ; referent)` estimated **at his ear**, after ASR, with the
shuffled-parent control. That is the same estimator the emergent-language work
uses (§10.6), which is a pleasing economy: **one metric covers both the words he
is given and the signals he might invent.**

### 10.8 The verdict on language *(revised after the owner's decision)*

1. **The LLM is a voice in his world, not a component in his head.** Decided.
   Everything above is the implementation and its controls.
2. **The MOUTH role (b) is the fallback**, retained because Jack still has to
   answer his owner and §10.5's raw voice channel cannot produce English. Note
   the evidence still supports keeping any such model frozen: RT-2 measured an
   11-point loss of general knowledge from task-only finetuning at 55B [V*],
   which is what LG.00's *"general knowledge survives untouched"* clause needs.
3. **Do not attempt from-scratch English.** 250–5,000 core-hours for a result
   worse than the model already on disk, during which Jack cannot speak to his
   owner. This is the one place where "figure out everything himself" is
   quantifiably the wrong call, and the reason is arithmetic, not taste.
4. **Build the voice (§10.5).** Now constitutional: he must MAKE sound, not only
   receive it. It is a sense-organ, not a language feature, and it is the
   prerequisite for two-way parent interaction and for emergent signalling.
5. **Emergent language after GEN.02, framed as a social result**, not as a
   language strategy.

The synthesis, in one line: **let him borrow the forms and forbid him from
borrowing the meanings — and the way to enforce that is to make every word he
learns arrive through his ears, attached to something that was happening to him
at that moment.**

---

## 11. Is from-scratch vision actually infeasible? — the owner's second question

> Owner, 2026-08-09: is training vision himself *"too big a task"*?

**Short answer: no, and the reason the intuition says yes is that it is
benchmarked against the wrong thing.** The relevant baseline is not DINOv2. It
is the visual-RL literature, which has trained pixel encoders from scratch on
**one** environment for a decade. Below, the arithmetic on this box.

### 11.1 The wrong baseline and the right one

DINOv2 (arXiv:2304.07193 [V] for the paper and for *"a ViT model with 1B
parameters … distilled into a series of smaller models"*; **[k]** for the
dataset size, which the abstract does not state) was trained on the curated
**LVD-142M — ~142 million images** — because its objective is to be good at
*every* visual domain at once:
satellite, medical, ImageNet, depth, segmentation. Jack needs **one world, a few
dozen object types, one camera, one lighting model, and a distribution that is
his own by construction.** Those are different problems by roughly two orders of
magnitude of required data, and the field has already measured the smaller one:

- Pixel-based DMControl / Atari agents (DrQ-v2, DreamerV3, EfficientZero) train
  their convolutional encoders **from scratch**, from **10⁵–10⁷ environment
  frames of a single environment**, and reach expert or human-level performance.
  The Atari-100k benchmark is literally *100,000 frames*. **[k — background
  knowledge, not re-fetched this pass; the specific numbers must be verified
  before they are quoted in a spec.]**
- `LEARNING_CORE.md`'s own `dreamer-xs` arm is **1,896,047 parameters** [c] and
  is already budgeted. A Dreamer-class core *contains* a from-scratch visual
  encoder; there is no separate vision project to fund.

**So the honest framing is: from-scratch vision for Jack is not a research
programme, it is a line item — and it may already be inside the learning-core
budget.**

### 11.2 The real constraint is rendering, and it is measured

`DIRECTION_AUDIT.md` §6.2, measured on this box 2026-08-09 [c]:

| resolution | fps | ms/frame | cost in env-steps |
|---|---|---|---|
| 128×128 | 14.6 | **68** | ~104 |
| 320×320 | 5.4 | 185 | — |

and `MUJOCO_GL=osmesa` / `egl` **do not exist on this box** — the working path
is `xvfb-run -a -s "-screen 0 640x480x24" MUJOCO_GL=glfw` over swrast/llvmpipe.
(PG.6's note that it can render via osmesa is false and should be corrected.)

Derived, with the W0 clock (k = 72, so one Jack-day = 1,200 sim-seconds =
240,000 physics steps ≈ **85 s** at the measured 2,826 steps/s with overlays
[c]):

| quantity | at 128², 5 Hz vision | derivation |
|---|---|---|
| frames per Jack-day | **6,000** | 1,200 sim-s × 5 Hz |
| render cost per Jack-day | **408 core-s (6.8 min)** | 6,000 × 68 ms [C] |
| render : physics ratio | **4.8×** | 408 / 85 [C] |
| Jack-days to reach 1M frames | **167** | [C] |
| **core-hours to render 1M frames** | **18.9** | 1M × 68 ms [C] |
| same, on 3 usable ARM cores | **≈ 6.3 wall-hours** | one overnight run [C] |

**1M frames of perfectly distribution-matched, label-free, self-generated
visual data costs one night on this box.** That is the whole feasibility
question, and the answer is a number rather than a wall.

Three mitigations, each with its expected effect and each requiring a
measurement rather than an assumption:

1. **Render at 64² instead of 128².** Pixel count drops 4×; llvmpipe is largely
   fill-rate-bound, so expect ~3× (≈ 23 ms/frame), taking the render:physics
   ratio to ≈ 1.6×. **Estimated, not measured — measure it.** 64² is what the
   Dreamer family uses, so this costs nothing in method.
2. **Render at 5 Hz, not at the 50 Hz control rate.** Already assumed above;
   it is a 10× saving and it is free, because vision at 5 Hz is what
   `DIRECTION_AUDIT.md` already found affordable.
3. **Cache for fixed-layout work.** `UNIFIED_BRAIN_BAKEOFF.md` §6 relies on
   ~500 distinct HNS layouts being rendered once. This works for fixtures and
   **does not work for a living Jack**, whose viewpoint is never twice the same.
   Say so explicitly, because the caching trick is quietly load-bearing in every
   cost estimate in this repo and it evaporates the moment he walks.

### 11.3 The finding that inverts the cost argument

Here is the part that was not in the brief and changes the recommendation.

**Rendering is a sunk cost of having vision at all.** A frozen tower does not
save one millisecond of it — Jack still has to render the frame before anyone
encodes it. What freezing saves is the **backward pass**. What it *costs* is the
**forward pass through a big tower**, every frame, forever, on 4 ARM cores.

Order-of-magnitude, and it must be measured before it is quoted:

| encoder | ~FLOPs / frame | plausible ARM-core time | vs the 68 ms render |
|---|---|---|---|
| DINOv2 ViT-S/14 @ 224² | ~4.6 GFLOP [k] | **~0.5–1.5 s** | **7–20× the render** |
| current from-scratch encoder, 0.24M params @ 64² | ~10 MFLOP [C, order] | **< 1 ms** | negligible |
| `dreamer-xs`-class CNN @ 64² | ~100 MFLOP [C, order] | ~10–30 ms | ~0.3× the render |

If those numbers survive measurement, then **on this box the frozen pretrained
tower is the EXPENSIVE option at runtime, not the cheap one** — by roughly an
order of magnitude — and the free-compute constraint that was assumed to argue
*for* freezing argues *against* it. The frozen tower is cheap only in the
regime where its embeddings can be cached, which is exactly the regime a living
Jack is not in (mitigation 3 above).

This is important enough to be a spec rather than a paragraph, and it is the
cheapest spec in this document: **PL.0** (§7.3) measures the four numbers in
that table on this box, on CPU, in minutes, and it can invalidate the frozen
arm's entire cost case before any bakeoff runs.

### 11.4 At what scale does from-scratch vision stop being affordable?

Pre-registered, so the answer is not adjustable after the fact:

| total frames needed | render cost @64² (est. 23 ms) | verdict |
|---|---|---|
| ≤ 1M | ≈ 6.4 core-h | **affordable** — one overnight run |
| 1M – 5M | 6.4 – 32 core-h | **affordable if amortised** across the lives Jack lives anyway; the frames are a by-product of living, not a separate job |
| 5M – 20M | 32 – 128 core-h | **marginal**; needs Kaggle for the training and a scheduled render campaign for the data |
| > 20M | > 128 core-h | **not affordable on this box.** Borrow a tower. |

And the crucial observation about the middle rows: **Jack generates these frames
by living whether or not anyone trains on them.** If the render is happening for
the policy's sake, the SSL objective on those same frames is nearly free — it is
backward-pass compute on a small encoder, not new data collection. The marginal
cost of from-scratch vision, given that Jack has vision at all, is **the
backward pass and nothing else.**

### 11.5 Does the critical-period option dominate both extremes?

Stated as the hypothesis it is, not as a conclusion:

> **Start frozen** (borrow DINOv2/SigLIP features so that early learning has a
> usable visual basis from step one and does not have to bootstrap perception
> and control simultaneously), **open plasticity on his own stream** once he has
> enough lived frames for SSL to be well-conditioned, **then consolidate.**

It dominates *on paper* because it takes each extreme's strength: the frozen
phase avoids the cold-start problem CortexBench's 20.4 %-random-features result
describes [V*], and the plastic phase captures the 8–23 points that adaptation
is measured to buy [V*]. It also has a *specific* biological rationale rather
than a vague one (§3), and — the part that matters here — the runtime cost
argument in §11.3 pushes the same way: the expensive frozen forward pass is only
paid during the early window, after which Jack runs on a small encoder distilled
or adapted from it.

**But "dominates on paper" is exactly the kind of sentence this project
distrusts**, and there are two specific ways it could be wrong: the schedule
adds a hyperparameter (when to open, when to close) that nothing yet tells us
how to set, and a window that opens too early reproduces the random-init
gradient shock that Knowledge Insulation exists to prevent [V*]. That is why it
enters §7 as an **arm**, not as the recommendation's default.

---

## 12. Cost — free compute only

Constraints (`SYSTEM.md`): 4 shared ARM cores here at `nice 19`, under ~1.5 GB,
leaving no process running; Kaggle 30 h/week P100 (**~23 h remaining this
week** per `gpu_budget.json` [c]); Colab T4, elastic. **No paid compute is
proposed anywhere in this document.**

### Stage 0 — CPU only, and it can kill the most expensive arm for free

| item | estimate | what it can kill |
|---|---|---|
| **PL.00** encoder ms/frame + loop throughput | **0.3 CPU-h** | **every frozen-tower arm**, if a ViT costs ~1 s/frame against a 68 ms render. This is the highest-leverage 20 minutes in the document. |
| **PL.01** unison admission (U1–U4 + HNS-A at d=128) × 4 arms × 3 seeds | **2.0 CPU-h** | the fully-frozen architecture, if U2 is zero — a constitutional overturn decided on CPU |
| **PL.02** reshaping gain R, 3 modality pairs × 4 arms × 3 seeds | **3.0 CPU-h** | the whole "frozen cannot bind" claim, in either direction |
| **PL.03** out-of-basis probes + binding task | **2.0 CPU-h** | the ceiling claim |
| **SM.01** odour field rules (measured: 124 µs/step at 500 puffs) | **0.5 CPU-h** | SM.02 |
| **TA.01** poison fixture (visual probe at chance, dose–response) | **0.5 CPU-h** | TA.02 |
| **VO.01** voice audible at a listener, attenuates, occludes | **0.5 CPU-h** | every signalling claim |
| **Stage 0 total** | **≈ 9 CPU-h** | **four arms, three claims and three fixtures — for zero GPU quota** |

**Ordered by what is actually reachable today (§7.6), not by spec number:**
`PL.00` (0.3 h) → `PL.02` (3.0 h) → `SM.01` (0.5 h) → `VO.01` (0.5 h) = **4.3
CPU-h, all runnable now.** The remaining Stage-0 items (`PL.01`, `PL.03`,
`TA.01`) are blocked on `PG.6`/`PG.7`, two **registered but never-executed**
CPU-only fixtures; running those first is ~40 minutes of work that unblocks nine
specs here and the whole unison ladder besides.

Run overnight at `nice 19` across 3 cores ⇒ ~3 wall-hours. **If PL.00 or PL.01
kills the frozen arms, the GPU stages shrink by roughly half.**

### Stage 1–3 — GPU

| item | backend | estimate | note |
|---|---|---|---|
| **PL.04** bakeoff, 4 arms × 3 seeds, binding battery at d=384 | P100 | **10–14 GPU-h** | the frozen arms are cheap (embeddings cacheable *only* for fixed-layout fixtures — not for a living Jack, §11.2); A3 pure pays a full backward pass |
| **PL.05** Score B, survival at matched wall-clock | T4 | **4–6 GPU-h** | re-scores stored curves + fresh lives |
| **SM.02** occluded-food value test | T4/P100 | **2–3 GPU-h** | includes the visible-condition arm, which is half the claim |
| **TA.02** one-trial aversion + 4 controls | T4/P100 | **3–4 GPU-h** | the cue–consequence swap doubles the run count |
| **TA.03** taste ablation | T4 | **1 GPU-h** | eval-only |
| subtotal | | **20–28 GPU-h** | |
| **+25 % slack** (preemption, resume, one re-run) | | **≈ 25–35 GPU-h** | **one to one-and-a-half Kaggle weeks** |

### One-time CPU campaigns, amortised

| item | estimate | note |
|---|---|---|
| Talkative-parent utterance corpus, ~330K tokens from SmolLM2-360M | **5–9 core-h** [C, order — measure] | **generate once, sample forever. Never call the LLM in the loop** (`SYSTEM.md` throughput floor; T0.07 measured what a model in the path does [L]) |
| 1M frames of his own visual stream @64², 5 Hz | **≈ 6.4 core-h** (est.; 128² is **18.9 core-h** measured-per-frame) | a by-product of lives he lives anyway; the *marginal* cost of from-scratch vision is the backward pass alone (§11.4) |
| 250K grounded words of parent speech (CVCL scale) | **≈ 4.9 core-h** physics, **≈ 13 core-h** with 64² vision | 208 Jack-days (§10.4) |

### Scheduling against the real quota

- **This week (~23 Kaggle-h left):** Stage 0 on CPU (no quota), then **PL.04**
  (10–14 GPU-h). That alone answers the owner's question — *does freezing cap
  binding?* — and leaves ~9 h.
- **Next week:** PL.05 + the SM/TA value tests (10–13 GPU-h).
- **Cheaper if the budget tightens:** drop `PL.05` (−4–6 GPU-h). Its question is
  conditional on PL.04 producing a winner at all, so sequencing PL.04 first
  makes PL.05 optional rather than planned.

Everything checkpoints every ~15 min and resumes (T0.04/T0.05 PASS [c]), and
**one GPU submission per spec** guarded by a module-level cache — the 5.5-GPU-h
scar of 2026-08-07 (`SYSTEM.md`).

---

## 13. What we refuse to claim

- **That frozen is a ceiling.** It is *suspect* under `GOAL.md`'s own
  capability test, on one specific and arithmetically certain ground (the
  reshaping gain is identically zero), with three independent literatures
  pointing the same way. **Suspect is not measured.** `PL.02` and `PL.03` are
  what would make it a finding, and both can come out the other way.
- **That frozen is fine.** Equally untested here. The one clean embodied
  ablation that exists (OpenVLA, **47.0 % frozen vs 69.7 % fine-tuned** [V*])
  goes against it, and CortexBench's adaptation gains (+8 to +14 across four
  suites [V*]) go the same way — but neither was measured at Jack's scale, on
  Jack's distribution, or on a binding task.
- **That the critical-period schedule will work.** Three measured objections
  stand against it (§3.3): gradual unfreezing ranks *worse* than full
  fine-tuning under shift; deficit sensitivity falls to ~zero at late onsets;
  and pre-training on the wrong statistics *"can severely decrease final
  performance"*. It is an arm, not a plan.
- **That "one latent" is the right organising principle.** §9 argues it is
  defensible and that its main virtue is that it makes the ablation matrix
  runnable at all. `UB.10`'s arm A5 (per-modality experts + router) is the
  standing challenger, and if it wins, "one brain as one trunk" is the wrong
  shape and we say so.
- **That smell or taste will help.** The multimodal literature's default outcome
  is that adding a low-bandwidth modality **hurts** (Kinetics: −1.2 to −3.8
  points on every combination [V]). They are constitutional, so they get built;
  whether our *wiring* of them is load-bearing is `SM.02`/`TA.03`'s question,
  and the honest prior is that it depends entirely on whether the occlusion
  mechanism makes them physically privileged.
- **That Jack will invent language.** He is alone; a lone agent has no reason to
  signal. Emergent communication is downstream of `GEN.02`, and its default
  outcome in the literature is a holistic, non-compositional protocol.
- **Anything from the primary sources we did not open.** Specifically, and these
  must be verified before any spec quotes them: Garcia & Koelling 1966's raw
  numbers and trial count (paywalled — and note this document already
  **corrected its own earlier claim** that it was a single-trial study); the
  DINOv2 LVD-142M dataset size; BabyBERTa's 5M-word result; the Dohare *Nature*
  Slippery-Ant and class-incremental CIFAR-100 magnitudes; NeuroMechFly v2's
  real-time factor; and Lee et al.'s 49 %/77 % peg-insertion figures.
- **A large effect anywhere.** The closest measured analogue to Jack's binding
  gain is Kepler-Encoder's R² of **0.049 / −0.001 / 0.187 across three robots,
  one negative** [c]. Every statistic in the PL family is designed paired,
  IQM-aggregated, with bootstrap CIs on the paired difference — or the
  experiment cannot see what it is looking for.

---

## 14. What this makes the machine better at

Per `SYSTEM.md`'s closing question. Five items, each a guard rather than a fix:

1. **`LEARNING_CORE.md`'s U2 is ambiguous in a way that silently excludes every
   frozen arm** (§6.4). Amend it to say *"entry path"*, and the criterion starts
   testing evidence instead of wording.
2. **RSV (relative source variance) becomes a standing metric in `UB.11`.** All
   three literatures surveyed here independently found that **accuracy hides
   fusion damage** — 93–94 % either way while fusion is dead [V]. RSV is the
   cheapest direct measurement of it in existence.
3. **A "wire the channel at W0, add the content at W1" rule** for every future
   sense (§8.2), derived from a measured failure mode rather than invented: a
   source absent during the early transient may never integrate [V].
4. **A deletion the evidence licenses**: `TrainingPipeline.py`'s
   `EWC.compute_fisher`, never called by the RL path [c], for a method measured
   to be **indistinguishable from vanilla** at our scale and pass count [V].
5. **A reachability audit as a publishing step, not an afterthought** (§7.6).
   This document's own first draft parented its most important spec behind two
   unrun fixtures — the exact `LESSONS.md` failure that once made the whole
   unison ladder untestable. **The check takes one script and it caught a live
   instance.** It should run against every proposed spec block before the block
   is committed, and it is a three-line addition to the integration queue's
   cross-check protocol.
6. **A correction to this document's own record**, kept visible: it asserted
   Garcia & Koelling 1966 as a single-trial study with the standard retelling,
   and the primary-source check said otherwise (§8.4). The correction is left
   in the text rather than edited away, because a document that quietly fixes
   itself cannot be audited — the same reason the ledger keeps history.
