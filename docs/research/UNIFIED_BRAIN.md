# Unified Multimodal Brain — fusion that actually fuses (researched 2026-08-06)

Serves GOAL.md ("All senses, one brain, trained together"). Tests named for the
registry; each has a null and a control that must fail.

## 1. Fusion architectures, ranked by evidence of real cross-modal computation
- **Token concat in ONE self-attention stream** (Gato 2205.06175, Unified-IO 2
  2312.17172): the only pattern where cross-modal interaction is STRUCTURAL —
  inspectable attention. Needs QK-norm etc. for mixed-modality stability.
- Cross-attention injection (Flamingo 2204.14198): asymmetric; gates can close
  to zero — "coexists".
- **Perceiver latent bottleneck** (2103.03206/2107.14795): cheapest true
  fusion; right tool to compress the 732M frozen vision features to few tokens.
- Q-Former (BLIP-2 2301.12597): at small scale a linear projection + token
  pruning matches it (SmolVLA) — don't spend params.
- Late fusion (concat before head): the NULL hypothesis. See-Hear-Feel
  2212.03858: attention fusion > concat on contact-rich tasks; flat concat
  invites collapse.
**Verdict at 58M: stems → one shared 6-8 layer d=512 trunk over ~45 modality
tokens (with modality-ID embeddings) → readout tokens (Octo).**
- **FUSE-1**: shared trunk vs equal-param late fusion. K: tie everywhere = "one
  brain" adds nothing, report honestly. C: cross-modal TIME-SHUFFLE at eval
  must hurt, else attention never crossed modalities.

## 2. Binding spaces: retrieval, not action
ImageBind 2305.05665, LanguageBind 2310.01852, EBind 2511.14229 (single-GPU
rebuild), RLBind 2509.14383. **No 2022-2026 evidence contrastive binding alone
improves closed-loop control** — it builds retrieval geometry (modality gap:
2412.07909); policies need time-local complementary info that InfoNCE on
global embeddings discards. Use only as light auxiliary + diagnostic.
- **BIND-1**: alignment on/off vs action success on hearing-tasks. C: the
  aligned model must at least RETRIEVE (audio→vision clip) ≫ chance, else the
  null result is uninformative.

## 3. Cross-modal masked prediction — THE tie that binds
MultiMAE 2204.01678; **M3L 2311.00924 (masked vision+touch improves POLICY
sample-efficiency — the direct evidence)**; cross-signal prediction beats joint
masking (2410.16424); MMP 2410.03010 (project missing modalities → robustness);
Sparsh 2410.24090.
For Jack: predict masked touch from vision+proprio (contact is visible), audio
-event from vision+proprio (impacts follow dynamics), masked proprio from all.
Each is a physics constraint tying senses to one latent body-state. ~1-2M params.
- **MASK-1**: ±cross-modal masking at equal steps. C: touch-from-SHUFFLED-vision
  must collapse to the unconditional mean, else the head ignores vision.

## 4. Audio (events + localization, not speech)
SoundSpaces 1912.11474/2206.08312: two-channel mel-STFT + small CNN suffices
for localization. ManiWAV 2406.19464 & Audio-VLA 2511.09958: contact audio
helps exactly when vision is occluded/ambiguous (2602.13640: gains gate on
acoustic task-relevance).
**Sim audio, cheap**: procedural MODAL SYNTHESIS on MuJoCo contact events —
damped resonator bank scaled by impulse + material, noise for scraping
(differentiable: 2210.15306); stereo pan by bearing = free localization labels.
Microseconds per contact, CPU-side. Pipeline: 16kHz → 64-mel × 2ch → conv stem
→ 4 tokens. Skip pretrained audio towers — sim audio is narrow.
- **AUDIO-1**: out-of-view fall → turn toward it; occluded contact timing.
  C: L/R swap must invert turning; 500ms lag must break timing. Muted-audio
  eval unchanged = modality collapse.

## 5. Touch: what 10 dims can/cannot do
Tactile-VLA 2507.09160 (force-aware 90% vs 25-40%), VLA-Touch 2507.17294,
OmniVTLA 2508.08706, Octopi 2405.02794 — all use tactile IMAGES or 6-axis F/T.
10 scalars CAN: contact detection/timing (foot-strike!), L/R load asymmetry,
grasp force. CANNOT: texture, slip, geometry, in-hand pose — never write tests
assuming them (DreamTacVLA 2512.23864's "low-dim force tier").
- **TOUCH-1**: blind push-recovery. HONEST possible outcome: touch redundant
  with proprioception (foot force partly inferable from torques) — a finding
  worth having. C: channel permutation must cause stumbles if load-bearing.

## 6. Modality collapse — the DEFAULT failure
Mechanism: greedy learning of the easiest modality (2505.22483 cleanest
analysis; 2506.11550). For Jack the dominant sense is PROPRIOCEPTION (348 clean
dims) — vision/audio/touch all at risk, the inverse of VLM text-dominance.
**Cure: modality dropout** (ModDrop 1501.00102; AV-HuBERT 2201.02184 — dropout
is WHY it uses lip video; MMP 2410.03010) + per-sense unimodal auxiliaries +
§3 (a modality that must be predicted cannot die).
**The standing audit — MODALITY-ABLATION MATRIX** (tasks × senses): for each
sense report Δsuccess under (a) zeroed (b) noise (c) time-shuffled (d) swapped
from another episode. Load-bearing iff all four hurt. "Unison" = no all-zero
column. Runs at every eval, forever.
- **COLLAPSE-1**: twin runs ± dropout. C: with-dropout model with proprio
  zeroed must still briefly stand from vision (vestibular substitution).

## 7. Does unison beat separate encoders? Honest evidence
FOR: PaLM-E 2303.03378 (positive transfer, grows with scale), RT-2, HPT
2409.20537 (shared trunk over stems, +20% over specialists — trunk-sharing IS
the transfer vehicle), See-Hear-Feel, ManiWAV.
AGAINST: Gato showed little cross-domain transfer; JAT 2402.09844 reproduced
the mixed picture at small scale.
**Implication at 58M: don't expect PaLM-E knowledge transfer (that's scale +
web data, which live in Jack's FROZEN towers). Jack's achievable unison =
state-estimation redundancy + representation shaping from cross-modal
prediction. Claim only what the ablation matrix shows.**
- **UNISON-1 (headline)**: shared trunk vs (i) per-sense specialists and
  (ii) frozen-separate-encoders→concat, matched params/steps, on a battery
  where each sense is load-bearing somewhere. C: leave-one-task-family-out
  must SHIFT other tasks (zero shift = covert late fusion inside the trunk).
  Until this passes, "the senses work in unison" stays OUT of the capability
  list.

## 8. The 58M recipe (what tiny VLAs do)
**SmolVLA 2506.01844 — closest blueprint**: frozen compact VLM, ≤64 visual
tokens, SKIP TOP HALF of VLM layers (mid-depth features better for control,
half the cost), ~100M flow-matching expert with INTERLEAVED cross+self
attention (their ablation: interleaved > either alone). TinyVLA 2409.12514,
π0 2410.24164, Octo 2405.12213 converge on: frozen towers as feature
extractors (mid-layer, few tokens), small joint transformer, separate small
action expert attending to trunk tokens, flow > autoregressive at small scale,
token thrift everywhere.
- **SCALE-1** (run last): interleaved vs cross-only vs self-only in the flow
  head; no difference → simplify and bank the params.

## THE RECIPE FOR JACK
1. **Stems (~4M)**: vision = frozen DINOv2+CLIP mid-layer → Perceiver resampler
   → 16 tokens; proprio 348→linear→4; touch 10→linear→1; audio 2ch-mel→conv→4;
   language = SmolLM2 embed → ≤16. Every token: modality-ID + time embedding.
2. **Trunk (~35M)**: 6-8 layers, d=512, QK-norm, one self-attention stream over
   ~45 tokens × short history + 2 readout tokens.
3. **Flow expert (~15M)**: 4 layers, interleaved cross(trunk)+self(chunk)
   attention, 8-16-step chunks.
4. **Aux heads (~2M)**: touch-hat(vision,proprio), audio-event-hat(v,p),
   masked-proprio-hat(all), optional audio↔vision InfoNCE.

**One loss, everything together (the owner's directive, operationalized):**
L = L_flow + 0.3·L_mask-crossmodal + 0.1·L_contrastive(pending BIND-1)
  + 0.1·L_unimodal-aux (anti-laziness).

**Dropout schedule**: per-sense independent — vision .2, audio .3, touch .3,
language .2, proprio .1; never vision+proprio together; dropped sense = learned
[MISSING-m] token (MMP), not zeros; 10% warmup at 0.

**Test order: COLLAPSE-1 → FUSE-1 → MASK-1 → AUDIO-1 → TOUCH-1 → BIND-1 →
UNISON-1 (headline) → SCALE-1.** All ledger-gated.

**Refuse to claim**: PaLM-E-grade transfer; texture/slip from 10-dim touch;
any sense not evidenced by its ablation-matrix column.
