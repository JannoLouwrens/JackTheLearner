# How Jack's brain should connect everything

## 1. The answer in five sentences

No — the current design does not bind the senses, and it could not even if it were trained, because every modality is compressed to a single pooled vector before fusion (`UnifiedBrain.py:4145-4198`, six `.unsqueeze(1)` calls), language is never passed at runtime at all (`VirtualWorld.py:626-639`), and `UnifiedBrain.py:4088` (`self.apply(self._init_weights)`) re-randomizes every pretrained encoder the moment it loads, so the "frozen LLM backbone" has never once existed in memory. A transformer will deliver *part* of it: it is the right substrate for the frozen language model you did not train and for a small learned-query fusion block over a real token pool, and it is the wrong substrate for low-level control, for second-to-second memory, and for hierarchical planning — all three of which 2025-26 results say should be an MLP, a short context window, and a frozen LLM respectively. The single architectural change that matters most is this: **binding comes from the training objective, not from the fusion layer** — you must give each sense many tokens instead of one, then train a masked cross-modal prediction loss that forces one modality to be predictable from the others, because arXiv 2603.19233 showed π0.5 encodes its language prompt at 99.3% linear-probe accuracy while its behavior is completely invariant to that prompt. Everything else in this document is downstream of that one sentence. The second-most important fact is that Jack's real blocker is not architecture at all: he has zero training data, all ten MoCap download URLs 404, and the loader silently substitutes random sinusoids paired with **randomly drawn** language labels (`MoCapLoader.py:701-706`) — which is not merely uninformative, it is anti-training.

---

## 2. What Jack actually is today

I am separating three states: **IMPLEMENTED** (runs on a live path and does what it says), **PARTIAL** (real code, wrong shape or wrong scale), **NAMED-ONLY** (constructed but unreachable, or the name is the only thing that matches).

| Component | Claim | Reality | Evidence |
|---|---|---|---|
| **Fusion** — `CrossModalFusion` | "3 layers, 512-dim, sensors attend to each other" | **PARTIAL.** Genuine pre-norm multi-head attention, 9,457,152 params. But it is `attn(x,x,x)` — **self**-attention, not cross-attention — over a **7-token** sequence (one token per sense + CLS), and its output is immediately re-attended by the 8-layer backbone over the same tokens. A 9.5M-param redundant pre-pass learning a 7×7 mixing matrix. | `UnifiedBrain.py:1498-1526`, `:1521`, `:4207-4213`, `:4227-4239` |
| **Vision** | "DINOv2 (1024, spatial) + SigLIP (768, semantic), OpenVLA-style" | **NAMED-ONLY.** `use_pretrained_vision` defaults False (`:105`) and `VirtualWorld.py:1877` forces it False again. Live path is a 4-conv CNN → `AdaptiveAvgPool2d((1,1))` → 128 numbers → Linear: **244,960 random params, 0.2% of the model, all spatial structure destroyed.** The pretrained branch loads **CLIP**, not SigLIP (`:600`), takes CLS only (`:631`, so not spatial), and its projector is `Linear(1024+768)` fed a 2048-d concat — it would crash on the first image and has therefore never executed. | `UnifiedBrain.py:105, 581-618, 600, 631-633` |
| **Memory** — `TemporalMemory` | "4-layer TransformerEncoder, 50 timesteps" | **NAMED-ONLY.** 12,635,136 params (10.7% of the model). Fires only under `if memory is not None` (`:4218`); `memory=` is passed at **zero** call sites repo-wide and `update_memory()` is called **zero** times. Effective temporal context: **1 frame**. Jack is a reflex agent. | `UnifiedBrain.py:1530-1571, 4218-4220` |
| **Action generation** | "SOTA pi0/GR00T flow matching, diffusion-denoised chunks, 48 actions" | **PARTIAL.** The flow-matching math is correct (`x_t = t·x1+(1-t)·x0`, velocity target, 10-step Euler). But the chunk is **16**, not 48 (`:84`; "48" exists only in README prose), it is 4.6M params vs π0's 300M, and — fatally — the entire 16×17 chunk is embedded as **one token** (`:2342-2343`), so the 4 self-attention layers run at sequence length 1 and are expensive MLPs. It cannot represent temporal structure inside a chunk, which is the only thing the architecture exists to do. Two mutually contradictory loss implementations exist; one down-weights t≈1 while simultaneously training the velocity head to emit clean actions at t=1 (`:5623-5626`, `:5637-5646`). `FlowMatchingScheduler` is constructed and never called. | `UnifiedBrain.py:2241-2410, 2342-2343, 3935, 5545-5648` |
| **Dual system** | "S2 2-5 Hz → S1 chunks 10-20 Hz → S0 torques 500 Hz" | **PARTIAL and inverted.** The wall-clock gate is real. Measured on this ARM host: S2 = **6.5 Hz**, S1 = **14.1 Hz**. S1 regenerates a full 10-step ODE every tick (~77-96 ms) and `get_action_from_chunk` then **discards it 15 times out of 16** (`:2474-2476`) — worst-case compute *and* worst-case smoothness, a guaranteed jerk at every boundary. `System0Controller` is never constructed (`system0_enabled=False`, `:164`); MuJoCo holds one `ctrl` vector across 5 substeps. **There is no 500 Hz loop anywhere in this repo.** | `UnifiedBrain.py:2413-2492, 2470-2484, 4552-4577`; `VirtualWorld.py:135, 890` |
| **LLM connection** | "Frozen SmolLM2-1.7B backbone + trainable projector" | **BROKEN, three ways.** (1) `self.apply(self._init_weights)` at `:4088` recurses into `self.language_encoder.llm` and overwrites every `nn.Linear`/`nn.Embedding` with `normal_(std=0.02)` — `requires_grad_(False)` does not protect data from in-place init. Reproduced: pretrained `q_proj` std 0.1010 → 0.0196, embedding std 1.0013 → 0.0197. (2) 3.42 GB fp16 exceeds this host's 3 GiB container cap. (3) Load failures are swallowed (`:1264-1268`), falling back to `ord(c) % 1000` over `text[:20]` into an untrained LSTM. And the only live trainer builds the brain with `llm_enabled=False, vision_enabled=False, audio_enabled=False` (`TrainingPipeline.py:228-230`) while `strict=False` everywhere hides the state-dict mismatch at load. | `UnifiedBrain.py:4088, 4096-4102, 1264-1268, 4168-4180`; `TrainingPipeline.py:228-230, 338` |
| **Grounding** — `SemanticActionAnchors` | "LLM-agnostic contrastive grounding" | **NAMED-ONLY.** Constructed at `:3865`, referenced only inside `compute_language_grounding_loss`, which is imported only by `archive/RobustTrainer.py`. `forward()` never touches it. `get_anchor_for_label` returns `0` (= walk) on any miss: "sit down"→walk, "please stop"→walk, "walk backward"→walk, and the dataset's own default label "move naturally"→walk. | `UnifiedBrain.py:3865, 5367-5518, 5440-5449` |
| **Planner** — `HierarchicalPlanner` | "HAC, 3-level hierarchy" | **NAMED-ONLY.** 37,166,972 params — the single largest block, 31.5% of the model. Gated on `use_hierarchy and task is not None`; nothing sets either. Receives no gradient, written to every checkpoint. | `UnifiedBrain.py:1875-2096, 4290` |
| **World model** | "TD-MPC2" | **NAMED-ONLY as TD-MPC2.** Four MLPs (2.97M). `plan_action_mpc` is random Gaussian shooting with `argmax(sum(rewards))` over horizon 5 — no MPPI/CEM refinement, no policy prior, no Q-ensemble, no terminal value, no discount. The decoder predicts a **256-d proprioception vector only**, so the loss is fully satisfiable by ignoring vision, audio and language. `compute_world_model_loss` has zero callers. | `UnifiedBrain.py:1692-1868, 1746, 5733` |
| **Dialogue** — `ResponseGenerator` | "Generates responses using the LLM" | **PARTIAL.** Three tiers; only tier 3 fires. API disabled by default, local generation runs on the weights `:4088` destroyed. Verified live: `generate('task_done', task='walking')` → `"Completed walking."`, `answer_question('What is 1+1?')` → `"I can't answer questions without my language model."` Jack's conversation is a 6-key dict of canned strings. | `UnifiedBrain.py:2577-2789, 3846-3853` |
| **Backbone** | "LLaMA-style, RMSNorm + SwiGLU + RoPE, 8 layers, d 512" | **IMPLEMENTED.** Correct pre-norm RMSNorm, SwiGLU, correct `rotate_half`. 36,710,400 params, sequence length 41. This and `Persistence.py` are the two things that are exactly what the docs say. | `UnifiedBrain.py:234-287, 357-385, 3891` |
| **Data** | "MoCap → retarget → imitation" | **NAMED-ONLY.** All 10 download URLs 404 (wrong path segment; the repo layout is `data/007/07_01.bvh`, not `subjects/07/`). `MoCapDataset.__len__` returns `max(1, ...)` so an empty dataset iterates, and `__getitem__` returns a random sinusoid with a label drawn from `np.random.randint(6)` — **uncorrelated with the motion**. `get_root_position()` exists and is called zero times, so root translation is discarded and *"forward" vs "backward" is not distinguishable in the training targets at all*. The entire language dataset is 19 hand-typed dict entries with 13 unique strings. | `MoCapLoader.py:265, 690, 701-706, 786, 805-833` |

**Parameter ledger.** Constructed on this box: **117,888,028** trainable params, not 105M. Of that, ~100.6M (85%) is attention that has never seen a gradient, and **51.68M (43.8%) is on no live path at all**. The actual senses total ~1.1M. Nothing has ever been trained; there is no `.pt`/`.pth` in the repo; `checkpoints/` is empty. The README's "verified" metrics (1.4 m/s walking, 850+ episodes, 73% push recovery) cannot have been measured and must be deleted from a public repo today.

---

## 3. Binding the senses: what 2026 says

**The fusion verdict.** The field landed on **unified token space** — every modality becomes tokens in one sequence, one transformer attends over all of them (π0, GR00T N1.5, the unified-action-tokenization line). But that is a *compute* story: it works because there are 500-800 tokens and 2-3B parameters to spend on them. At Jack's scale the correct 2026 choice is the compute-poor equivalent: a **Perceiver-style learned-query latent bottleneck over a unified token pool**, published as Kepler-Encoder-v0.1 (arXiv 2607.13522, Jul 2026). Tokenize each modality natively, concatenate to ~200 tokens, let 8-16 learned queries cross-attend through 2 blocks into one latent. ~2M trainable params on frozen backbones, O(NM) not O(N²), 40 epochs, entirely T4-feasible. That is the template. Copy it.

**But — and this is the answer to the owner's actual question — no fusion architecture binds modalities on its own.** The decisive 2026 result is "Not All Features Are Created Equal" (arXiv 2603.19233): π0.5's layer 17 predicts the language prompt category at **99.3% linear-probe accuracy while the model's behavior is completely invariant to that prompt**. Injecting visual activations under a *null* prompt recovers 0.999 cosine similarity to baseline behavior and 73-77% task success. Language only becomes causally load-bearing when the scene is ambiguous (libero_goal: wrong prompt drops success 97% → 0-10%; libero_object: 50-100% regardless). Multi-pathway models show functional *dissociation*, not integration.

Read that carefully: **encoded is not used.** A stack of attention layers over multimodal tokens produces separately-readable, modality-siloed features by default. Genuine integration comes from the **objective** — masked cross-modal latent prediction, cross-modal distillation, contrastive alignment. Kepler's ablation is the proof in the other direction: its fused latent predicts end-effector force at R² 0.049 / −0.001 / 0.187 across three robots versus 0.010 / −0.019 / 0.067 for a compute-matched **vision-only** control, paired t-test p ≤ 0.012; distance correlation between latent geometry and true state 0.221 vs 0.050. Those are modest absolute numbers — deliberately quoting them so you calibrate. The effect is real, statistically clean, and small. That is what "genuine binding" looks like when someone actually measures it rather than asserting it from an architecture diagram.

So: **the fusion layer is plumbing; the loss is the mechanism.** Jack has zero binding losses. He has never been trained. The question "does the brain integrate or concatenate" currently has a trivial answer — it does neither, it applies random weights to a 128-number image summary.

**On the 512-dim bottleneck: it is not the bottleneck. Do not widen it.** DINOv2-family features have measured effective rank in the ~50-80 range against a nominal width of 1024 (IdEst, arXiv 2606.03338), so a rank-512 linear map from a single pooled vector loses essentially nothing. What Jack is actually losing is **spatial**: `dinov2-large` at 224px emits 256 patch tokens and `:628` keeps index `[:, 0]`. That is a ~256× reduction, and it is why object-finding and navigation can never work regardless of how the fusion is arranged.

**The real imbalance runs against vision, and it is hard-wired.** Every modality gets exactly one token, while `JointTokenizer` emits 17 joint tokens plus a body token. The backbone sequence is [CLS 1][modality 6][joints 17][body 1][actions 16] = 41 tokens — **vision is 2.4%, proprioception is 44%.** π0 and GR00T are the exact inverse: hundreds of vision patch tokens against one state token. Attention mass follows token count. ReViP (arXiv 2601.16667) documents this state-dominant bias as the cause of "false completion" — the robot reports success while visibly failing — and recovers +26% over π0 by rebalancing it.

**Modality collapse is now mechanistically understood.** "A Closer Look at Multimodal Representation Collapse" (arXiv 2505.22483, ICML 2025) shows collapse happens when noisy features from one modality become entangled, *through a shared set of neurons in the fusion head*, with predictive features from another — masking the second modality's positive contribution. Cross-modal distillation works because it implicitly disentangles these by freeing rank bottlenecks. With 6 tokens and a shared 512-d head, Jack's entanglement is maximal: there is no room for modality-specific subspaces.

**Detection before mitigation.** BalanceBenchmark (arXiv 2502.10816) formalizes the metrics: modality contribution disparity, ablation gap, Shapley attribution. No mitigation method wins uniformly, and OGM-GE's α is dataset-specific (0.1 VGGSound vs 0.8 CREMA-D) — so you cannot pick a balancing hyperparameter without measuring first. For Jack the cheap version is a 30-line harness: zero each modality in turn, report Δ action-L1 and Δ task success. Expect it to reveal that zeroing vision changes nothing.

**Concretely, for Jack:**
1. Delete `CrossModalFusion` (`:1498-1523`, `:3861`). It is self-attention over 7 tokens whose output the backbone re-attends anyway. Deleting it is functionally free and recovers 9.46M params.
2. Emit **16 vision tokens** (`last_hidden_state[:, 1:]` → 16×16 grid → `AdaptiveAvgPool2d(4,4)` → project), not 1.
3. Add `PerceiverFusion`: 16 learned queries, 2 blocks of cross-attention + SwiGLU over the full token pool. ~3M params.
4. Add **masked cross-modal latent prediction**: each step, randomly hold out one modality; predict its EMA-target embedding from the fused latent via a small per-modality MLP head; smooth-L1. Plus VICReg-style variance/covariance terms — a latent-only loss with an EMA target collapses without them. No reconstruction decoder.
5. Add per-modality dropout p=0.15 and the ablation harness.
6. **Delete the mood token** (`:4191-4198`). `.expand(B, 1, -1)` makes it identical across the batch — it carries literally zero per-example information and can only be absorbed as a bias the CLS token already supplies. Mood is interoception derived from Jack's own event history, not an independent sense, and giving it peer status steals 1/7 of fusion attention. Keep `MovementMoodCoupling.modulate_action` (`:4620`), which is FiLM-style post-hoc modulation and is already exactly right, and keep injecting mood as prompt text.
7. **Pass `language=` on every tick**, not only inside a decomposed subtask (`VirtualWorld.py:632-639` vs `TaskManager.py:292`). A fusion module that never sees language during unprompted exploration cannot bind words to sensation.

---

## 4. Where the LLM belongs

**Commit: frozen LLM on the side, small learned adapter, gradients never reach it.**

Jack's instinct is the correct 2026 instinct. The frontier moved *away* from end-to-end VLM trunks precisely as the data got bigger. π0.5 uses AdaRMS "knowledge insulation" — the action expert attends to VLM layers at inference but its gradients are **stopped** from reaching pretrained weights (arXiv 2505.23705). GR00T N1/N1.5 **freeze** the Eagle VLM in both pretraining and finetuning, explicitly to preserve language understanding. The empirical case is stark: naive behavioral cloning through a VLM destroys **94% of GQA accuracy within 10k steps**. Figure's Helix is the only counterexample that backprops S1→S2, and Figure has thousands of hours of proprietary teleop.

The VLA-as-trunk paradigm is a way of *spending* 970k robot episodes (OpenVLA), 22.9k episodes / 10.6M frames (SmolVLA), or 20,854 hours of egocentric video (GR00T N1.7). Jack has zero. Training a bespoke multimodal brain against those is not a resource question, it is a category error.

There is also a decisive solo-dev property nobody writes papers about: **a frozen encoder's outputs are cacheable.** Precompute HumanML3D/BABEL text embeddings once — 45k captions × 960 floats ≈ 170 MB — and every subsequent adapter experiment trains on the Oracle box's CPU in minutes with **no GPU resident at all**. That single property is what makes this plan fit a 16 GB T4 and a burst schedule, and it is destroyed the instant the LLM becomes the trunk.

**Which model, what size, running where:**

- **Dialogue:** `HuggingFaceTB/SmolLM2-360M-Instruct`, frozen, fp16 ≈ 0.7 GB, running **out-of-process on CPU** next to MuJoCo. Never a submodule of `nn.Module` — that is what let `:4088` destroy it. Demote from 1.7B: 3.42 GB does not fit the 3 GiB container cap on a box shared with paying tenants, and it is too slow for a real-time companion on a no-GPU host. For demo-quality conversation, set `llm_api_enabled=True` and use an API model; it costs cents per session and is the only path to dialogue that is not canned strings.
- **Grounding:** a **separate** small frozen sentence/text tower (SigLIP2 text tower, or a ~110M sentence encoder like bge-m3). Do not reuse the chat model's mean-pooled hidden states — mean-pooling a causal decoder is a known-weak sentence representation, anisotropic, dominated by surface overlap, and it destroys word order ("walk to the chair then sit" ≈ "sit then walk to the chair"). Decoupling also means you can swap the chat model freely, which is the *actual* correct solution to the LLM-swap fragility `SemanticActionAnchors` was invented for.
- **Vision:** frozen `google/siglip2-base-patch16-224` (~200M), keep the patch tokens, one small trainable projection. Do **not** restore the DINOv2+CLIP fusion — that buys millimetre grasp precision Jack does not need; he needs to know there is a chair over there.
- **Interface:** the LLM sequence is cross-attended by the adapter (never mean-pooled), and the adapter emits a **32-d motion latent** plus a **subtask string**. Gradients stop at the LLM boundary by construction.

Total frozen: ~565M, all CPU-runnable. Total trained: ~22M.

**Do not fine-tune SmolVLA-450M or any VLA.** It is mechanically affordable (LoRA fits a T4, ~20-30 T4-hours), so cost is not the objection. The objection is that the only thing finetuning buys over freezing is SmolVLA's pretrained action expert, and that expert was trained on 6-DoF SO-100 arm data with **zero locomotion**. It transfers nothing to a 17-DoF whole-body humanoid. You would pay 30 GPU-hours to reinitialize the one component you were paying for. OpenVLA-7B is simply out of reach — LoRA needs ~24 GB, a T4 has 16.

---

## 5. Will a transformer help?

Directly: **a transformer is the right answer for two of the five jobs, and the wrong answer for three.** No single transformer will "connect everything" — the connective tissue is a shared latent plus a predictive loss plus an external store, and a transformer is only one ingredient.

**Fusion — YES, but tiny.** Cross-attention over a small multimodal token set is the correct and standard mechanism, it has no temporal axis so the quadratic cost is irrelevant, and it is what SmolVLA does at 450M on a consumer GPU. Two blocks, 16 learned queries, ~3M params. Nine and a half million params to mix seven tokens is absurd over-provisioning at zero data.

**Short-horizon memory (0.1-10 s) — NO as a separate module.** Ni et al. (arXiv 2307.03864) is the load-bearing result: transformers in RL substantially improve **memory** (recall at 1500 steps) but do **not** improve **long-term credit assignment**. Jack's unsolved problem is credit assignment — which joint commands produce forward locomotion — and a transformer contributes nothing to that while multiplying every gradient step by ~1000×. Delete `TemporalMemory` (12.6M dead params) and instead keep a deque of the last ~16 fused latents prepended as tokens to the one trunk you keep. If partial observability still hurts after that, add **one GRU(512)** (~1.6M) — not Mamba, not xLSTM; those win at 200M+ params over long sequences, which is not where Jack lives.

**Long-horizon memory (hours to months) — NO neural sequence model at all.** See §7.

**High-level policy — YES, and it is the one you did not train.** The frozen LLM already decomposes tasks in natural language (π0.5-style subtask strings, arXiv 2504.16054), and `TaskManager.py` is ~80% of the way there. Delete `HierarchicalPlanner` (37.17M, 31.5% of the model). HAC is among the least reproducible things in RL and does not train from scratch without an already-working low-level policy, which Jack does not have. Two untrained planners competing for one job is worse than one working pathway.

**Low-level control — NO. Use a residual MLP.** SimBa (arXiv 2410.09754) and SimbaV2 (arXiv 2502.15280, ICML 2025) hold SOTA across 57 continuous-control tasks including HumanoidBench with running observation normalization + residual feedforward blocks + LayerNorm and **no attention anywhere**. TD-MPC2 (arXiv 2310.16828) — which Jack's `WorldModel` already correctly copies as pure MLP — scales the same recipe to 317M params over 80 tasks. The MuJoCo-humanoid-from-scratch problem has been solved repeatedly and never with a transformer policy. The measured numbers on this box are the argument: full `brain.forward` **107 ms** (9.3 Hz ceiling), backbone alone **76 ms**, flow-matching ODE another **77-96 ms** — versus **0.09 ms** for a 376→256→256→17 MLP. That is an **840-1200× compute tax for a policy that has learned nothing**, and it makes both the documented 50 Hz S1 and 500 Hz S0 arithmetically impossible on any CPU Jack will run on. It also makes the training plan impossible: PPO on Humanoid needs O(10M) steps; at 840× per-forward cost with 5 optimization epochs that is hundreds of GPU-hours against 30 free hours a week. The same policy as a SimBa MLP with an off-policy learner is 6-15 GPU-hours — one or two Kaggle sessions.

**Action decoding — NO denoising.** OpenVLA-OFT (arXiv 2502.19645) replaced diffusion with bidirectional parallel decoding over empty action-query embeddings plus plain L1 and got **higher** success at **~26× throughput** (LIBERO 76.5% → 97.1%). Helix S1 — the highest-frequency production humanoid VLA, 200 Hz over 35 DoF — is direct regression, no denoising. Flow matching exists in π0 to model **multimodal demonstration distributions** across 10k+ hours of teleop; Jack has kinematic mocap, where tracking has one correct answer. He is paying the entire cost of a generative model for a property his data does not have. L1 also converges to the median trajectory, which filters noisy retargeting — exactly what you want.

Note the pleasant consequence: the regression path is **already built and Jack routes around it**. `JointTokenizer` already appends 16 learned action query tokens (`:1593`, `:1650-1655`), the mask is already all-ones/bidirectional (`:1661`), and `forward()` already returns `action_head(action_feat)` → `[B,16,17]` in one pass (`:4243`). Change the loss to `F.l1_loss` and make `act_dual_system` call it.

**The small-data regime, honestly.** MaIL (arXiv 2406.08234): "Transformers struggle with smaller datasets, often leading to overfitting or suboptimal representation learning" — Mamba beats a Transformer on **every** LIBERO task at limited data and only ties at full data. LRAM (arXiv 2410.22391): recurrent xLSTM backbones beat Transformers across 432 tasks in 6 domains at **every** model size, with linear-time inference. X-IL (arXiv 2502.12330) reproduces it. Jack's data regime is not "limited" — it is **zero**. 105M params with no data is not ambition; choosing an architecture whose only advantage appears at 1000× your data budget is a design error. And transformers have *not* been displaced at the frontier — Dreamer 4 (arXiv 2509.24527) is a 2B block-causal transformer needing an H100 — but "at scale" is a regime Jack will never enter on a T4.

---

## 6. Grounding language to action

**This is the blocker. Treat everything above as prerequisite plumbing.**

**Verdict on `SemanticActionAnchors`: the instinct is correct and worth preserving; the implementation must be deleted entirely.**

The docstring's insight is genuinely good and the big labs get it for free by never swapping backbones: an action-side embedding space that language *selects into* survives replacing SmolLM2 with Llama. Keep that idea. Now the reasons the code cannot survive:

1. **It is unreachable.** Constructed at `:3865`, used only inside `compute_language_grounding_loss`, imported only by `archive/RobustTrainer.py`. `forward()` never calls it. Trained to perfection it would change nothing about what Jack does.
2. **The label function poisons its own supervision.** `get_anchor_for_label` (`:5440-5449`) does exact-synonym then substring matching and returns `0` (walk) on failure. I ran it against 22 realistic commands: 19 failed. "please stop"→walk, "stop!"→walk (punctuation breaks it), "sit down"→walk, "dance"→walk, "follow me"→walk, "walk backward"→walk (opposite direction, same anchor). And the dataset's own default label, `"move naturally"` — applied to every clip outside the 19-entry table, i.e. ~2,590 of ~2,600 CMU clips — also resolves to walk. The contrastive loss would teach Jack that essentially all human motion is walking.
3. **The InfoNCE term fights the anchor terms.** Loss 3 (`:5490-5494`) uses `targets = arange(B)`, so every off-diagonal pair is a negative — but there are only 8 categories. At B=32 balanced, 96 of 992 off-diagonal cells (9.7%) are same-category false negatives; under the real label distribution it approaches 100%. Losses 1 and 2 pull same-category samples together while Loss 3 pushes them apart, and Loss 3 dominates numerically. TMR (arXiv 2305.00976) identified exactly this and fixes it by filtering negatives above a text-similarity threshold.
4. **The temperature has no floor.** `nn.Parameter(0.07)` used as `logits / temperature.abs()`. CLIP parameterizes *log*-temperature and multiplies by a `logit_scale` clamped at 100. Jack divides by a raw parameter, so gradient descent reduces loss by shrinking it toward zero. Measured drift: 0.07000 → 0.03912 over 3000 Adam steps, still falling. `.abs()` is also non-differentiable at 0 and permits a sign flip.
5. **Eight discrete buckets is not a latent space.** "walk to the door then turn left" resolves to one integer. There is no compositionality, no object referents, no spatial grounding. SCRIPT (arXiv 2605.22894) names this directly: approaches "relying on compact policies or latent skill spaces limit their ability to scale with increasingly diverse motion data and complex language instructions," and specifically criticizes compressing sentence semantics "into one global vector" — which is exactly `LLMEncoder.encode_batch`'s mean-pool at `:1303`. The anchors are a rediscovery of PADL/ASE/CALM (2022-23); the 2026 line has moved on.
6. **The premise is false.** The projector it protects is `Linear(2048→1024) + Linear(1024→512) + LayerNorm` ≈ 2.62M params. Retraining that on a fixed paired dataset is **~20 minutes of T4 time.** The entire contortion exists to avoid a 20-minute job.

**Also delete Phase 3.3 outright.** `archive/RobustTrainer.py:4744-4771`: success sets `loss = loss * 0.5`, failure sets `loss = loss * 1.5`, then `backward()`. Multiplying a scalar loss by a positive constant scales gradient **magnitude only**; direction is identical — so failure takes a *larger step in the same direction* as success. There is no policy gradient, no log-prob, no advantage. And the underlying objective (`:4753-4761`) is cosine similarity between a 17-DoF joint-torque vector and the unit vector `[1,0,...,0]`, a quantity with no relationship to forward locomotion. Running this would actively damage the model. It is checked `[x]` in `TRAINING_PIPELINE_PLAN.md:649`.

### How the bridge should actually be trained

**Representation:** a continuous **32-d motion latent** with PULSE geometry (unit-normalized, VIB-regularized, sampled from a learned prior), not 8 one-hot anchors. PULSE covers 99.8% of AMASS in 32 dimensions.

**Order of operations — this is not stylistic.** PULSE measured **97.1% imitation success when the latent is distilled from a pretrained tracker, versus 32.6% when the same latent space is trained directly with RL.** That 3× gap is the single most actionable number in this entire report. Track first → distill second → attach language third. Do not attempt language conditioning before Jack can stand; a language interface on a policy that cannot walk is a demo of nothing.

**Conditioning:** stop mean-pooling. Return the full token sequence from the text encoder and cross-attend to it. Pooling "obscures token-level cues such as action verbs, body-part references, and modifiers" (SCRIPT) — it is why Jack cannot distinguish "walk forward" from "walk backward" from "walk slowly." π0, π0.5, OpenVLA and GR00T all cross-attend to token sequences; none of them pool.

**Loss:** SigLIP sigmoid pairwise loss (arXiv 2303.15343), `s_ij = t·x_i·y_j + b`, with **learnable bias b initialized to −10 and t' = log 10**. The bias init is load-bearing: it matches the |B|²−|B| negatives vs |B| positives prior. Sigmoid significantly outperforms softmax InfoNCE below batch 16k and your batch on one T4 is 32-64 — squarely the regime where it wins. Add TMR-style negative filtering (drop off-diagonal pairs whose text-text similarity exceeds ~0.8).

**Compositionality:** π0.5-style two-stage inference. The LLM emits a subtask *string* ("walk to the door" → "turn left"); the low-level policy conditions on that string's tokens and executes one leg. Language stays the interface between levels, so it stays open-vocabulary and human-readable.

**Reward, later and only later:** once the supervised base works, use the frozen text-motion retrieval tower as a **dense** reward — `r = cos(text_emb, motion_emb)` over the executed trajectory — plus balance terms. SCRIPT does exactly this ("trajectory-level contrastive rewards using a frozen state-text contrastive model"); MotionRFT (arXiv 2603.27185), ReAlign (arXiv 2511.19217) and MVR (arXiv 2603.01694, HumanoidBench locomotion) confirm the pattern. Nobody learns grounding from a binary "did it move forward" — one bit per 100-step episode.

### Where the paired data comes from — the blocker does not exist

| Source | Size | What it gives | Cost |
|---|---|---|---|
| **CMU mocap BVH** (`una-dinosauria/cmu-mocap`) | 1.04 GB, ~2,600 clips | Motion, **unrestricted license** — safe for MIT-licensed shipped weights | free, `git clone --depth 1` |
| **HumanML3D** | tens of MB (texts + index.csv only) | **44,970 free-form captions** over 14,616 clips, 3 per clip, 5,371-word vocab — free paraphrase supervision, which is the synonym problem the anchors were invented for | free, research-only |
| **BABEL v1.0** | ~100 MB JSON | 28k sequence + **63k frame-level** labels over 250+ categories on 43 h of AMASS; keyed by `feat_p` (e.g. `CMU/CMU/07/07_01_poses.npz`) which **joins directly to the BVH filename** | free, research-only |
| **LocoMuJoCo** (`robfiras/loco-mujoco-datasets`) | **1.97 GB total** | 22,000+ clips **already retargeted** across 12 humanoids, plus a validated robot-to-robot retargeter | free, auto-caches |

That is ~45,000 real (sentence, motion) pairs for ~200 MB and zero annotators. **Skip MoCapAct** — delete it from `README.md:296`. It is 50 GB minimum (600 GB large), HDF5, and its experts are for dm_control's **56-actuator** CMU humanoid against Jack's `action_dim=17`; the value it adds is embodiment-specific and does not transfer. `/data` has 34 GB free.

### The blocker inside the blocker

**Root translation is discarded.** `BVHParser.get_root_position()` exists at `MoCapLoader.py:265` and is called **zero times**. `_build_observation` concatenates 17 joint positions + 17 joint velocities and nothing else — verified: 34 non-zero columns out of 256. So "walk forward" vs "walk backward," "turn left" vs "turn right" are **provably identical under the observation**. The contrastive loss would be asked to separate anchors that carry no distinguishing signal. No architecture and no quantity of data fixes this. Add root linear velocity (3) and root yaw rate (1) by finite-differencing `get_root_position` and the root rotation channels, into both the observation and the action conditioning, **before training any grounding at all.**

And verify the retarget visually before spending a GPU-hour. `retarget_frame` (`:404-411`) is a direct axis-by-axis Euler copy — no rotation-order handling, no BVH-Y-up → MuJoCo-Z-up conversion, no parent-frame composition, no per-joint sign calibration. MuJoCo's `right_knee` range is (−2.62, −0.03), strictly negative. Measured: BVH knee +30° → −0.0300 (clipped), +60° → −0.0300 (clipped). **If CMU encodes knee flexion as positive rotation, every knee in every clip is pinned straight and nothing would catch it.** Retarget 10 clips, replay through `mj_data.qpos`, render, and look at it. Or skip the file and use LocoMuJoCo's validated retargeter, which is the better call.

---

## 7. Memory and what holds experience together

Three timescales, three different mechanisms. They are not substitutes.

**0.1-10 s — learned, in the trunk.** This is what turns a stream of independent frames into "I have tried this three times and it is not working." Jack has *none*: `TemporalMemory` is unreachable, so his effective context is one frame and he is a reflex agent with a chat log. Fix: keep a deque of the last ~16 fused latents in the brain and prepend them as tokens to the trunk sequence. Delete the separate 12.6M encoder — the temporal axis belongs in the one trunk you keep, not in a duplicate module.

**Minutes-hours — task state, not memory.** `TaskManager` already handles this. Leave it.

**Hours-to-months — external, symbolic, on disk.** This is the only representation that survives a weight update, is auditable, is editable, and cannot be catastrophically forgotten. The 2026 consensus is unambiguous: modular/external memory as the default (arXiv 2603.01761), sparse memory finetuning (arXiv 2510.15103), physically isolated per-category LoRA for embodied continual learning (arXiv 2605.27762), with ICLR 2026 still listing hippocampal-neocortical consolidation as **open**. Empirically the gap between "has memory" and "no memory" exceeds the gap between LLM backbones, and Mem0's 92.5% LoCoMo / 94.4% LongMemEval come entirely from external stores.

`CompanionMemory` (`:2791`) is the right *shape* and is genuinely wired (`VirtualWorld.py:1296-1297, 1614-1616`; `TaskManager.py:244, 559, 722`). Four fixes:

1. **The eviction rule is inverted and it deletes the user's name first.** `:2840` scores `importance * (1 - (now - t)/86400)`. After 24 h the bracket goes **negative**, so multiplying by importance makes important memories rank **lower**. Verified: a 3-day-old "name is Janno" at importance 2.0 scores **−4.0**; a 3-day-old "[Thought] bored" at importance 0.3 scores **−0.6**. The idle thought survives; the name is evicted. With `memory_size=1000` and two entries per conversational turn, this fires after ~500 exchanges. Replace with the Generative Agents composite: `0.5·recency + 3.0·relevance + 2.0·importance`, `recency = 0.995^age_hours` (exponential, never negative), all terms normalized. Use the same composite in `recall()`, which currently ranks on raw cosine only.
2. **Embeddings are mean-pooled SmolLM2 hidden states** (`:2826`, `:2853`) — anisotropic, poor for retrieval without contrastive finetuning, so "What does Janno like?" will not reliably retrieve "Janno likes chess," and the keyword fallback (`:2869-2878`) is what actually runs whenever the LLM is off. Swap in a purpose-built small embedding model (bge-m3 / Nomic Embed v2) on the ARM CPU inside the 3 GiB cap. Dimensionality drops 2048 → 768 as a bonus.
3. **Move to SQLite + sqlite-vec.** Currently 1000 × 2048 float32 (~8 MB) is re-serialized in full via `torch.save` every 300 s (`Persistence.py:791`). SQLite gives incremental writes, real crash safety, and a store you can grep and hand-edit.
4. **Add reflection.** At session end or every 100 memories, have the LLM emit 3-5 higher-level statements ("Janno prefers short conversations in the evening") with pointers to source rows, plus a session summary; add a `level` column so retrieval prefers reflections. Have the LLM score importance 1-10 on write instead of the hardcoded constants at every call site. Generative Agents (arXiv 2304.03442) ablated this: removing reflection degraded agents to repetitive, context-free behavior within 48 simulated hours. Reflection is also what makes the store *shrink* rather than grow.

`Persistence.py` itself is genuinely good — atomic tmp+rename, `max_saves=10`, round-trip self-test, correct handling of embeddings, PAD baselines, monologue history. **Keep the pipe, fix the cargo.**

**The world model.** Delete the MPC path (`plan_action_mpc` `:1815-1863`, `reward_predictor`, the obs decoder, `use_mpc` at `:4285`). It is not TD-MPC2: random Gaussian shooting, no MPPI/CEM refinement, no policy prior, no terminal value, no discount, horizon 5. `argmax` over 512 random rollouts of an **untrained** reward head is an adversarial search for the largest model error. TD-M(PC)² (L4DC 2026) shows even the correct algorithm over-estimates value in high-DoF control without a policy-constraint term, and fixing that alone more than doubles 61-DoF humanoid performance.

**But keep the encoder/dynamics/EMA-target (~2.3M) and repurpose them as a JEPA-style auxiliary loss** predicting the target-encoder embedding of the **full fused multimodal latent** at t+k for k ∈ {1, 4, 16}, weighted ~0.1, with VICReg terms. This is the only honest version of "prediction forces binding": V-JEPA 2 (arXiv 2506.09985) reached 80% zero-shot Franka pick-and-place from <62 h of unlabeled robot video by predicting in latent space — but **it binds what it predicts**. Jack's current decoder targets a 256-d proprioception vector (`:147`, `:1746`), so the loss is fully satisfiable by ignoring vision, audio and language.

**Do not build a Dreamer/TD-MPC training loop.** Latent imagination buys sample efficiency in *environment steps* — the one resource Jack does not pay for. DreamerV3 needs ~15 h on a 3090 for 1M steps on one visual control task (≈40-60 h on a T4 with Colab's 2 vCPU), more than Kaggle's entire weekly budget for a single skill. MuJoCo Playground (arXiv 2502.08844) trains most DM Control tasks in **under 10 minutes on one GPU**, entirely on-device with thousands of parallel envs; FastTD3 (arXiv 2505.22642) solves HumanoidBench tasks in under 3 h on an A100, ≈9-12 h on a T4, one Kaggle session. Trading cheap parallel GPU sim steps for expensive world-model gradient steps is a losing trade here.

**The hard rule, written down and asserted in code:** episodic memory **never** enters weights online. `VirtualWorld` stays `brain.eval()` + `torch.no_grad()` — add a guard assert. Weight updates happen only in scheduled offline "sleep" runs on Kaggle, replaying the persisted buffer. For a solo dev with no eval harness, online consolidation is unrecoverable — you cannot detect the forgetting, bisect it, or roll it back. An SQLite file you can diff and restore is the only safe substrate. This is one of the few things Jack already gets right; do not let the continual-learning goal talk you out of it.

---

## 8. The revised architecture

```
                        ┌──────────────── FROZEN, CPU, OUT-OF-PROCESS ────────────────┐
   speech ──ASR──►  faster-whisper-tiny (39M)                                          │
                        │                                                              │
   chat ────────────►  SmolLM2-360M-Instruct (or API)  ──► dialogue text ──► TTS       │
                        │        └─► subtask string ("walk to the door")               │
                        │                                                              │
   text ────────────►  text tower (~110M, e.g. SigLIP2-text / bge-m3)                  │
                        │        └─► T token embeddings  [CACHED TO DISK]              │
   camera ──────────►  SigLIP2-base-p16-224 (~200M)                                    │
                        │        └─► 196 patches → pool 4x4 → 16 tokens [CACHED]       │
                        └──────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
        ┌────────────────────── TRAINED  (~22M total) ──────────────────────┐
        │                                                                    │
        │  TOKEN POOL  (~40-60 tokens, d=384)                                │
        │    17 joint tokens   ← sin/cos(angle) ⊕ symlog(vel) ⊕ symlog(τ)   │
        │     1 root token     ← lin.vel (3) + yaw rate (1)   ★ NEW          │
        │     1 touch token    ← 10 contact channels                         │
        │    16 vision tokens  ← projected patches                           │
        │     T text tokens    ← projected text tower                        │
        │                            [1.5M trainable adapters]               │
        │                              │                                     │
        │                              ▼                                     │
        │  PERCEIVER FUSION  — 16 learned queries × 2 blocks                 │
        │    cross-attn(q=latents, kv=pool) → SwiGLU → 16 latents @384       │
        │                            [3M]  O(NM), not O(N²)                  │
        │                              │                                     │
        │            ┌─────────────────┼──────────────────┐                  │
        │            ▼                 ▼                  ▼                  │
        │   BINDING HEADS      TEXT→LATENT ADAPTER   JEPA PREDICTOR          │
        │   per-modality MLPs  → 32-d motion latent  → fused latent @ t+k    │
        │   predict held-out      (PULSE geometry,      k∈{1,4,16}           │
        │   modality's EMA         VIB, unit-norm)      + VICReg             │
        │   embedding [1M]         [2M]                 [2.3M, aux w=0.1]    │
        │   ↑ DISCARDED after                                                │
        │     pretraining          └────────┬─────────┘                      │
        │                                   ▼                                │
        │                     ┌─────────────────────────────┐                │
        │                     │  32-d MOTION LATENT SLOT    │  ← written at  │
        │                     │  (written 2-5 Hz, bg thread)│    S2 rate     │
        │                     └──────────────┬──────────────┘                │
        │                                    ▼                               │
        │   S1 @ 50 Hz:  SimBa RESIDUAL MLP  (no attention)                  │
        │     obs-norm → Linear(→512) → 3×[LN→512→2048→ReLU→512→+res] → LN   │
        │     inputs: proprio ⊕ 16 fused latents ⊕ 32-d motion latent        │
        │     output: 16×17 chunk of JOINT POSITION TARGETS, L1 loss,        │
        │             parallel decoded in ONE forward, exec 8 of 16,         │
        │             exponentially-weighted overlap ensembling  [~12M]      │
        └────────────────────────────────────┬───────────────────────────────┘
                                             ▼
   S0 @ 500 Hz:  MuJoCo <position kp/kv> actuators — ZERO learned params,
                 zero inference latency, closes the PD loop every 2 ms
                                             ▼
                                        MuJoCo room

   ─────────────────── SIDECARS (no gradients at runtime) ───────────────────
   MEMORY:  SQLite + sqlite-vec.  score = 0.5·rec + 3.0·rel + 2.0·imp,
            rec = 0.995^age_hours.  Tiers: core / archival / raw.
            Session-end LLM reflection → summaries + importance scores.
            Survives restarts. Never touched by SGD online.
   CURIOSITY: RND novelty over the FROZEN vision embedding → drives
            unprompted "go look at the thing I haven't seen."  [~1M]
   OFFLINE:  VirtualWorld logs (obs, action, reward, next_obs) + dialogue
            to disk → nightly/weekly Kaggle "sleep run" refits the
            text→latent adapter and (rarely) the S1 policy from replay.
```

**Component table.**

| Component | Size | Frozen/Trained | Trained by | Data |
|---|---|---|---|---|
| faster-whisper-tiny | 39M | frozen | — | — |
| SmolLM2-360M-Instruct (or API) | 360M | frozen | — | — |
| Text tower (SigLIP2-text / bge-m3) | ~110M | frozen | — | — |
| SigLIP2-base vision | ~200M | frozen | — | — |
| Token adapters (joint/root/touch/vision/text) | 1.5M | trained | Stage 1 + 3 | retargeted CMU + rendered frames |
| PerceiverFusion (16q × 2 blocks) | 3M | trained | Stage 1 | paired text↔motion↔vision |
| Binding heads (masked cross-modal) | 1M | trained, then discarded | Stage 1 | same |
| JEPA predictor + EMA target | 2.3M | trained (aux) | Stages 1-3 | MuJoCo rollouts |
| Text→32-d motion latent adapter | 2M | trained | Stage 4 | HumanML3D/BABEL ∩ CMU |
| Motion tracker (teacher) | ~8M | trained, then distilled away | Stage 2 | ~400 retargeted CMU clips, MJX PPO |
| S1 SimBa policy + L1 chunk head | ~12M | trained | Stages 2-3 | MJX rollouts + distillation |
| RND curiosity | ~1M | trained online-free | Stage 6 | frozen vision embeddings |
| **Total trained (shipped)** | **~22M** | | | |
| **Total frozen** | **~710M** | all CPU-runnable | | |

**Rates.** S2 = fusion + LLM on a background thread at 2-5 Hz writing a 32-d slot (an S2 tick that takes 300 ms is harmless — the legs keep walking). S1 = SimBa MLP at 50 Hz emitting position targets (0.09 ms per forward measured, so 50 Hz is trivially met). S0 = MuJoCo's own position actuator at 500 Hz, free. This is the honest version of GR00T/Helix at Jack's scale, and no gradients flow from S1 into S2 — knowledge insulation by construction, which is also what makes the offline-cached adapter training possible.

**One XML change is the highest-leverage line in the whole plan.** In `assets/humanoid_full.xml` (and `jack_room.xml`, `humanoid_terrain.xml`), replace every `<motor gear=100..300>` with `<position kp=... kv=... ctrlrange="<joint limits>">`. The action becomes a target joint angle (or delta from default pose) instead of a torque. This (a) gives you the 500 Hz S0 for free, (b) makes the action signal smooth and low-frequency, which is what makes chunking and mocap supervision work at all, and (c) matches what every 2025-26 humanoid sim-to-real paper and Helix do. Torque targets are near-white noise; position targets are what mocap actually gives you.

---

## 9. What to delete from Jack

Deleting is progress. This removes roughly **100M of the 118M parameters** and ~6,000 lines.

**Fatal bugs — fix or delete before anything else:**
- `UnifiedBrain.py:4088` `self.apply(self._init_weights)` — move it **before** any pretrained encoder is constructed, or guard it to skip modules whose params are all `requires_grad=False`. **Until this is fixed, no LLM or vision result from this codebase means anything.**
- Every `strict=False` load (`Persistence.py:568`, `TrainingPipeline.py:338/340`) → print missing/unexpected keys and **refuse to start** if any encoder is uninitialized.
- `VirtualWorld.py:661` `logger.debug` on brain exceptions → `logger.exception` + a consecutive-failure counter that halts. Right now if the brain crashes every frame, Jack stands motionless and the console is silent.
- `TrainingPipeline.py:228-230` vs `VirtualWorld.py:1874-1878` — one shared config object. There is currently **no path by which a trained Jack ever sees or hears anything.**

**Delete outright (params recovered):**
| What | Lines | Params |
|---|---|---|
| `HierarchicalPlanner` + `HighLevelPlanner` + `MidLevelController` + `compute_hierarchical_loss` | `:1875-2096`, `:5760` | 37.17M |
| 8-layer bespoke trunk out of the control path (`TransformerBlock`/`MultiHeadAttention`/`CrossAttention`) | `:291-386`, `:3888` | 36.71M |
| `TemporalMemory` | `:1530-1571`, `:4218-4220` | 12.64M |
| `CrossModalFusion` | `:1498-1523`, `:3861` | 9.46M |
| `ActionExpert` + `FlowMatchingScheduler` + `generate_actions_flow_matching` + `train_flow_matching_step` + `compute_flow_matching_loss` + config keys | `:2241-2410`, `:4376-4501`, `:5545-5648`, `:150,166-174` | 4.62M |
| `LanguageEncoder` LSTM + the `ord(c) % 1000` hash + both inline vocab dicts | `:1144-1153`, `:4168-4180`, `:4674-4681` | 4.71M |
| `ObjectDetector` (untrained DETR head; `find_object` needs >0.5 from a random 22-way softmax) + `NavigationPlanner` (`plan_path` is literal `start*(1-t)+goal*t`, `update_map` ignores its vision arg) | `:641-884` | 1.47M |
| `SemanticActionAnchors` + `compute_language_grounding_loss` | `:5367-5518`, `:5652-5730` | 0.41M |
| `PhysicsRuleBank` + `get_active_rules` (25,600 random numbers with 25 physics-law names typed next to them; nothing binds row *i* to name *i*, and `get_active_rules` reports "F=ma: 0.03" about a random vector) | `:1669-1689`, `:4319-4325` | 0.03M |
| `AudioEncoder` (`transcribe()` literally returns the string `"[Audio transcription requires Whisper model]"`) — replace with faster-whisper-tiny | `:915-1141` | 0.42M |
| `System0Controller` + `system0_hz`/`system0_enabled` | `:2494-2551`, `:163-164` | — |
| `AMPDiscriminator` (never constructed anywhere) | `:388` | — |
| mood token block | `:4191-4198` | — |
| `plan_action_mpc` + `reward_predictor` + obs decoder + `use_mpc` branch | `:1815-1863`, `:4285-4312` | ~0.7M |
| `Empowerment`, `SkillDiscovery`/DIAYN, `Metacognition`, `AutotelicGoalGenerator` (three answers to one question, none trained) — keep only RND over frozen vision embeddings | `:3173-3525` | ~5M |
| `_execute_command` (returns hardcoded 3-vectors for a 17-joint action space), `make_coffee` | `:5160-5232` | — |
| `AlphaGeometryLoop.py`, `SymbolicCalculator.py` | whole files | — |

**Delete from the data path:**
- `MoCapLoader._get_synthetic_sample` (`:748-772`) and the empty-data fallback (`:701-706`). `__len__` must return the real length so an empty dataset raises. A hard crash is the correct behavior.
- `MoCapDownloader.MOTION_LABELS` (`:805-833`) — 19 entries, 13 unique strings, replaced by 45k real captions.
- `ActionComputer` (`:496-536`) and `kp`/`kd` config (`:81-82`). It manufactures pseudo-torques (kp=10, kd=1, clipped ±0.4) from mocap frames **that were never simulated**, fed to actuators with gear 100-300. BC on that produces a humanoid that falls over on frame one — and that failure would have been blamed on the action head.
- `tests/test_all_fixes.py:138-156`, which explicitly **asserts** that random synthetic noise is produced when data is missing, and passes. This is why none of it surfaced.

**Delete from docs and repo root:**
- `README.md:230-235` fabricated metrics — this is a **public MIT repo**; do this today.
- README's "48 actions" (`:48, 67, 178, 271`) and the Diffusion Policy row (Chi et al. predict 16, execute 8, never 48).
- The "500 Hz" / "10-20 Hz" / "TD-MPC2" / "HAC" / "AlphaGeometry" / "GMR-style recursive IK" / "quaternion rotations" claims. (`config.use_ik` and `scale_factor` are read **zero** times; the only match for "quaternion" in `MoCapLoader.py` is the docstring asserting it.)
- `RUN_ON_COLAB.ipynb`, `COLAB_WITH_DRIVE.ipynb`, `TRAIN_ON_COLAB.ipynb` (the last clones `github.com/YOUR_USERNAME/JackTheWalker.git`, an unedited template placeholder). README Quick Start invokes `Phase0_Physics.py`/`Phase1_Locomotion.py`/`Phase2_Imitation.py`, all of which are in `archive/` and would crash on a fresh clone.
- `archive/RobustTrainer.py` entirely, including `_train_phase3_3_llm_projector_with_feedback` and `_train_phase3_4_language_vision_grounding` (`:4662-4906`), and Phases 3.3/3.4 from `TRAINING_PIPELINE_PLAN.md`.
- The MoCapAct row (`README.md:296`).

**Keep, unchanged:** `Persistence.py`, `VirtualWorld.py`'s pure-inference stance, `JointTokenizer`'s per-joint tokenization (better than most VLAs — just add sin/cos + symlog at `:1584`), `MovementMoodCoupling.modulate_action`, the backbone's RMSNorm/SwiGLU/RoPE implementation (correct, and worth salvaging as a reference).

---

## 10. Staged path

Total to a first honest multimodal companion: **~90-110 T4-hours**, roughly 4 weeks at Kaggle's 30 h/week. Every script must checkpoint every ~15 min and be resumable — a run that cannot resume is a run you will never finish on a 12 h cap with Colab teardown.

### Stage 0 — Surgery and data. 0 GPU-hours. 4-6 sessions.
Fix `:4088`. Delete everything in §9. Switch the XML to `<position>` actuators. Fix `CMU_MOCAP_URL` → `https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/` with `f'{int(subj):03d}/{subj}_{trial}.bvh'`; assert `len(files) > 0`. `git clone --depth 1` the full 1.04 GB corpus to `/data`. Download BABEL JSONs (~100 MB) and HumanML3D texts + index.csv (tens of MB); join to CMU by `feat_p`. Add root linear velocity + yaw rate to the observation. Retarget ~400 clean clips (locomotion, turns, sits, gestures, waves), **replay them through `mj_data.qpos` in the sim and watch the render** — if the knees are stiff, flip the sign in `JOINT_MAPPING` (`:324, 328`). Precompute and cache text-tower embeddings for all ~45k captions (~170 MB).
**Testable outcome:** 400 retargeted clips replay in MuJoCo without immediate termination, ≥12 of 17 actuators show non-zero variance, and no label in the corpus falls through to a default. Repo builds with ~18M trainable params.

### Stage 1 — Prove multimodal binding. ~6 T4-hours. 1 Kaggle session.
**This is the smallest thing that proves binding works at all, and it needs no policy, no RL, and no MuJoCo rollouts.** Train only the token adapters + PerceiverFusion + binding heads on (retargeted motion window, cached caption embedding) pairs. Two losses: SigLIP sigmoid pairwise (t'=log 10, b=−10, TMR negative filtering) and masked cross-modal latent prediction (hold out one modality, predict its EMA-target embedding, + VICReg). Modality dropout p=0.15.
**Testable outcome — three numbers, all against controls:**
1. Text→motion **R@10 on held-out clips** substantially above chance, *and* above an identical model trained on **shuffled** captions. (Shuffled control is non-negotiable given what `__getitem__` used to do.)
2. **Kepler's binding test:** R² of predicting held-out proprioception/root-velocity from the fused latent beats a **compute-matched proprio-only** control, paired t-test across seeds. If it does not, the fusion is decorative and no downstream stage will save it.
3. **Ablation harness:** zero each modality at test time, report Δ retrieval and Δ prediction. If zeroing text changes nothing, you have reproduced π0.5's failure mode (arXiv 2603.19233) and must strengthen the masking schedule before proceeding.

### Stage 2 — Make him move. 40-60 T4-hours. 5-8 sessions.
MJX / MuJoCo Playground. (a) Velocity-command flat-ground walking with a SimBa residual MLP, PPO or FastTD3-style off-policy — Playground quotes <10 min on a modern GPU; budget 2-4 T4-hours plus 3-5 tuning runs. (b) DeepMimic-style motion tracking against the ~400 retargeted clips, π(a | s, future target poses), with an AMP-style style reward. **This is the largest single item and it is unavoidable.** Use `gym.vector` or MJX parallel envs — the current `collect_rollout` steps one non-vectorized env with a full forward per step (`TrainingPipeline.py:517-567`), which is why no checkpoint has ever materialized.
**Testable outcome:** Jack walks on command at a commanded velocity for >60 s without falling, and tracks ≥70% of held-out clips to within a joint-position error threshold.

### Stage 3 — Distill to a 32-d latent. 10-15 T4-hours. 2 sessions.
Distill the Stage 2 tracker into a 32-d VIB latent + decoder (PULSE recipe). **Distillation, not RL on the latent** — PULSE measured 97.1% vs 32.6%.
**Testable outcome:** ≥90% of the tracker's clip-tracking success recovered through the 32-d bottleneck; sampling from the prior produces recognizable, non-degenerate motion.

### Stage 4 — Attach language. 10-15 T4-hours. 2-3 sessions.
Train the text→motion-latent adapter on cached caption embeddings, supervised, with the Stage 1 alignment loss as an auxiliary term (weight 0.1-0.5). Then wire π0.5-style hierarchical inference: LLM emits subtask strings, adapter converts each to a latent, decoder executes.
**Testable outcome:** on 50 held-out held-out commands including synonyms, Jack executes the correct motion class ≥70% of the time — and critically, "walk backward" ≠ "walk forward" and "please stop" ≠ walk. Then, and only then, add dense contrastive-reward RL refinement (10-15 T4-h) if the supervised base plateaus.

### Stage 5 — Vision into the loop. ~5 T4-hours. 2 sessions.
Render `jack_room.xml`'s eye camera at 128×128 while replaying retargeted motions with randomized camera and object placement. **Labels come from `mj_data` for free** — which of the 6 named objects are in FOV, egocentric bearing and range, contact flags. No annotator, no LLM. 100k frames = 490 MB uint8 or 360 MB as cached frozen SigLIP embeddings. Render with **EGL on the ephemeral GPU** (~500 fps, minutes) — **not** on the Oracle box (osmesa/llvmpipe ~10 fps, ~3 h, competing with paying tenants). Train only the vision projection into the existing fusion.
**Testable outcome:** the ablation harness from Stage 1 now shows a **non-zero** Δ when vision is zeroed, and "look at the plant" produces a correct head/torso orientation.

### Stage 6 — Companion behavior. 0 GPU-hours. 3-4 sessions.
SQLite + sqlite-vec memory with the corrected scoring, tiers, and session-end reflection. RND curiosity over frozen vision embeddings driving unprompted exploration. The transition logger that writes `(obs, action, reward, next_obs)` and dialogue to disk for the next sleep run — **this is the actual bridge between the product and the training loop, and it does not exist today.** Note that no VLA in the landscape — RT-2, OpenVLA, Octo, π0/π0.5, GR00T, Helix, SmolVLA — does unprompted exploration or continual learning; they are all instruction-conditioned feedforward policies. There is no prior art to copy, and it must be built at the memory/behavior layer, not the gradient layer.
**Testable outcome:** across a restart, Jack recalls the owner's name and three preferences; over a 10-minute idle session he visits ≥3 distinct novel locations unprompted; the memory store is under 5 MB after 1,000 exchanges (i.e. reflection is compressing, not accumulating).

---

## 11. Open questions for the owner

1. **Position actuators, yes or no?** Switching `<motor>` → `<position>` changes the meaning of every action in the repo. It is the single highest-leverage line in this document, but it invalidates any tuning that assumed torque. **Recommend: yes, do it in Stage 0 before anything else is written.** The free 500 Hz PD loop and the smooth low-frequency action signal are worth more than anything you would preserve.

2. **Is the 17-DoF skeleton fixed, or will you adopt a validated retarget target?** `MoCapLoader`'s retargeter has no rotation-order handling, no Y-up→Z-up conversion, and a plausible knee sign inversion that would silently pin every knee straight. **Recommend: adopt LocoMuJoCo's skeleton and validated retargeter (1.97 GB, 22k pre-retargeted clips) rather than debugging your own IK.** It costs you a one-time XML change and saves you the hardest-to-detect class of bug in the project.

3. **How strict is MIT-license purity for the shipped weights?** CMU mocap is unrestricted; AMASS/BABEL/HumanML3D annotations are research-only. **Recommend: train shipped weights on the CMU subset only** (BABEL and HumanML3D both index CMU clips, so you keep ~45k real captions restricted to the safe motion subset), and keep any full-AMASS experiment in a clearly-marked non-shipped branch.

4. **Is demo-quality dialogue a near-term requirement?** Local SmolLM2-360M on ARM CPU next to MuJoCo will be slow and shallow; the current fallback is a 6-key dict of canned strings. **Recommend: API LLM for dialogue** (cents per session, and it is the only tier that ever produces real conversation), frozen local text tower for grounding only. Make templates an explicit offline-degraded mode with a visible banner.

5. **Are you willing to delete ~90% of the parameter count and ~6,000 lines?** Everything in this plan assumes yes. If the answer is no, the honest alternative is to keep the current repo as an architecture sketch and start the trainable system in a clean directory that imports nothing from it. **Recommend: delete in place, on a branch, with the current HEAD tagged.** A 22M-param system that walks is worth more than a 118M-param system that has never run a gradient step.

6. **Fix the public README today, independent of everything else.** `README.md:230-235` states verified metrics — 1.4 m/s walking, 850+ episodes before falling, 73% push recovery — for a system with no checkpoint that has never been trained. **Recommend: delete that table and the "48 actions" / "TD-MPC2" / "HAC" / "AlphaGeometry" / "500 Hz" claims before the next person reads the repo.** This is the one item with a real external cost.