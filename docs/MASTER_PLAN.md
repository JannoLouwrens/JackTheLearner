# MASTER PLAN — the complete checklist and how everything connects

**The goal (GOAL.md): one brain, all senses in unison, learning its world by
living in it.** This document is the map from that sentence to every test.
CHECKLIST.md (generated from the ledger) is the scoreboard; this is the plan.
Four research pillars feed it: `research/CAPABILITIES.md`, `research/MEMORY.md`,
`research/CURIOSITY.md`, `research/UNIFIED_BRAIN.md`.

## The one-paragraph theory of Jack

A small trained brain (stems → one shared trunk → flow-matching action expert,
~58M) sits on frozen swappable towers (DINOv2+CLIP vision 732M, SmolLM2
language). It learns by **goal-conditioned hindsight regression** (every failure
is a success at what it did achieve — no RL machinery needed for the flow head),
practices what **learning progress** says is learnable (noisy-TV-proof,
saturation-proof, purely data-side), in a **playground** it explores because a
VLM proposes interesting goals ("climb the ladder") and LP disposes. All senses
enter one trunk and are tied by **cross-modal masked prediction** (predict touch
from vision, audio events from dynamics), kept alive by **modality dropout**,
audited by the **modality-ablation matrix**. Mastered goals compile into
**ledger-verified named skills**; everything he learns and remembers lives in
**plain files on this box** and survives restarts. The environment itself
mutates toward his frontier (ACCEL), so learning never saturates.

## Dependency graph (what unlocks what)

```
Tier 0 harness ─► Tier 1 primitives ─► T2.01/02 locomotion ─┐
                                       T2.03 vision-vs-random├─► Tier 3 ablations ─► Tier 4 unison ─► Tier 5 claims ─► Tier 6 living Jack
CMU+HumanML3D data (T1.13 ✓) ────────► T2.04/06/07 grounding ┘
PLAYGROUND (PG.*) ───────────────────► CURIOSITY (CU.*) ─► OPEN-ENDED (OE.*)
MEMORY files (ME.*) ─────────────────► Tier 5/6 persistence claims
UNIFIED BRAIN (UB.*) = Tier 4, expanded
```

Cross-cutting rules that never expire: X3 seed discipline (≥3 seeds for
learning claims; CI must exclude the null — T1.08's floor: min detectable
effect 0.047); the noisy-TV dwell-time report on every curiosity claim; the
modality-ablation matrix on every multimodal eval; the wipe-the-file test on
every memory claim.

## PHASE MAP (each line = a spec or spec family; ✓ = already in ladder)

### Phase A — foundations (DONE or in flight)
- ✓ T0.01–T0.12 harness · ✓ T1.01–T1.13 primitives (3-seed reverify queued)
- ✓ T2.01 locomotion vs random (RUNNING) → T2.02 vs honest MLP

### Phase B — the playground (new: PG.*) — CPU, ~1-2 weeks, unlocks everything
- PG.1 Procedural room generates & is physically sound (ragdoll floats at
  ρ-ratio depth ±10%; boxes slide iff tanθ>μ; energy bounded)
- PG.2 Water works: buoyancy callback; C: disable buoyancy → ragdoll sinks,
  swim metric → 0 (else it measures floor contact)
- PG.3 Ladder + adhesion hand actuators; falling produces clean data
- PG.4 Noisy-TV panel fixture + dwell-time metric (mandatory reporting fixture)
- PG.5 Procedural contact audio (modal synthesis, stereo pan) + free
  localization labels

### Phase C — capabilities vs null (Tier 2, expanded)
- ✓ T2.03 pretrained vision beats random · T2.04 BC · ✓ T2.05 world model
- ✓ T2.06/T2.07 grounding · ✓ T2.08/T2.09 curiosity+noisy-TV · ✓ T2.10 memory
  retrieval · ✓ T2.11 skills · ✓ T2.12 emotion · ✓ T2.13 convergence
- NEW T2.14 imitation from real mocap beats nearest-neighbour retrieval
- NEW T2.15 free-form paraphrase routes to the right task (LLM→task, L2 grid
  designed BEFORE training)
- NEW T2.16 hindsight flow goal-reaching; N: shuffled-goal-label training
- NEW T2.17 progress/success estimation (PL3 — gates all RL/TTA; null:
  linear-in-timestep; reversed video → reversed progress)
- NEW T2.18 action chunking tradeoff (C2) + latency reactivity (C3)
- NEW T2.19 bimodal action distributions: flow vs MSE-collapse (C4 — genuine
  falsification risk, OFT found L1 ties sometimes)
- NEW T2.20 episodic memory: find hidden object faster in episode N+1; wipe
  store → null (M2 — near-empty niche in SOTA, Jack can be novel)

### Phase D — earn your parameters (Tier 3, expanded)
- ✓ T3.01–T3.08 ablations (vision, proprio, world model, planner, memory,
  curiosity, mood, LLM)
- NEW T3.09 creative loop earns its existence or is deleted (currently ZERO
  call sites — wire-or-delete)
- NEW T3.10 trunk-knowledge preservation (LE8 — linear probes hold through
  action training; unfreezing must reproduce drift; cheapest D1 evidence)

### Phase E — UNISON (Tier 4 = UB.*, the headline)
Order per UNIFIED_BRAIN.md: UB.1 COLLAPSE-1 (±dropout twins; ablation matrix
has no all-zero column) → UB.2 FUSE-1 (shared trunk vs late fusion;
time-shuffle control) → UB.3 MASK-1 (cross-modal masking helps the POLICY) →
UB.4 AUDIO-1 (turn toward unseen fall; L/R swap inverts) → UB.5 TOUCH-1 (blind
push-recovery; honest risk: redundant with proprio) → UB.6 BIND-1 → UB.7
UNISON-1 (vs specialists AND vs bolt-on at matched params — until this passes,
"senses in unison" stays unclaimed) → UB.8 SCALE-1 (flow-head attention
ablation). ✓ T4.01–T4.05 remain as the composition gate.

### Phase F — CURIOSITY: he teaches himself (CU.*)
Per CURIOSITY.md stack (LP primary; VLM proposes, LP disposes; no raw
ICM/RND rewards ever):
- CU.1 goal babbling in outcome space produces broader coverage than action
  babbling
- CU.2 LP goal-sampling → EMERGENT CURRICULUM (time-ordered mastery onsets
  stand→walk→push→ramp; unlearnable goals decay to ε) — the first falsifiable
  "Jack teaches himself"
- CU.3 noisy-TV immunity: near-zero panel dwell while coverage grows (ICM
  control arm MUST get trapped — proves the fixture works)
- CU.4 METRA skill space: z decodable >90%, beats random-repeat null; distilled
  into the flow head
- CU.5 VLM proposer: earlier ladder/pool engagement + higher blind-rated
  interestingness; scrambled-caption control must not beat LP-only
- CU.6 affordance archive predicts pushability/liftability of held-out objects;
  welded object classifies un-pushable
- CU.7 lessons-from-failure (Reflexion): retry improves vs pure-resampling null

### Phase G — MEMORY: it's him (ME.*)
Per MEMORY.md build order, all plain files in /data/jack/memory/:
- ME.1 events.db day-one: cued QA ≥80% @1k events; fabricated-event query must
  ABSTAIN
- ME.2 profile.json + facts.db: preference honoured next session; newer
  contradiction overrides; WIPE-THE-FILE → base rate (memory lives on disk)
- ME.3 reflections beat raw top-k at equal tokens
- ME.4 forgetting: decay+supersede beats FIFO; stale-answer control
- ME.5 degradation curve at every store decade (10²→10⁵), gap-to-oracle
  reported (LongMemEval rubric incl. abstention)
- ME.6 skill library (Voyager): composite task faster; corrupted-skill control
  must fail
- ME.7 SIESTA wake/sleep: old concepts held ≤2pt after sleep; emptied-buffer
  control must forget
- ME.8 working memory survives restart (GRU state checkpointed to wm.state;
  zeroing it mid-episode must hurt)

### Phase H — the claims & the living Jack (Tiers 5-6, expanded)
- ✓ T5.01–T5.07 (thesis, forgetting, plasticity, sleep, unprompted exploration)
- NEW T5.08 open-ended non-saturation: distinct mastered outcome clusters grow
  8 weeks without plateau vs fixed-scene null; unfiltered-mutation control must
  degenerate (OE loop: ACCEL + OMNI-EPIC-lite)
- NEW T5.09 cross-embodiment transfer (LE2): morphology-variant pretraining
  speeds new body; white-noise pretraining gives nothing
- ✓ T6.01–T6.04 + NEW T6.05 companion battery: contingent responsiveness (S1),
  intent inference (S3), safety-zone <1/1000 across reward scales (S4),
  persistent identity vs re-seeded twin (S5)

### Written scope exclusions (an honest ladder states what it cannot test)
No hands → no dexterous manipulation (C8). No robot → no sim-to-real. No
resident GPU → no always-on world-model planning. Texture/slip → impossible
with 10-dim touch.

## The single sentence to keep

**VLM proposes, learning progress disposes, hindsight regression learns, the
ledger decides, the files remember, and the playground grows with him.**
