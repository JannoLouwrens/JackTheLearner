# JackTheLearner Architecture Deep Dive

## Model Overview: UnifiedBrain (105M Parameters)

```
┌─────────────────────────────────────────────────────────────────────┐
│                       UNIFIED BRAIN (105M params)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      ENCODERS                                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────┐  │   │
│  │  │  Proprio   │  │   Vision   │  │  Language  │  │ Touch  │  │   │
│  │  │  256→512   │  │  DINOv2+   │  │  Embed→512 │  │ 10→64  │  │   │
│  │  │            │  │  SigLIP    │  │            │  │        │  │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └───┬────┘  │   │
│  └────────┼───────────────┼───────────────┼─────────────┼───────┘   │
│           │               │               │             │           │
│           └───────────────┴───────────────┴─────────────┘           │
│                                   │                                  │
│                    ┌──────────────▼──────────────┐                  │
│                    │    CROSS-MODAL FUSION       │                  │
│                    │    (3 layers, 512 dim)      │                  │
│                    └──────────────┬──────────────┘                  │
│                                   │                                  │
│                    ┌──────────────▼──────────────┐                  │
│                    │   TEMPORAL MEMORY (50 ts)   │                  │
│                    └──────────────┬──────────────┘                  │
│                                   │                                  │
│  ┌────────────────────────────────▼────────────────────────────┐   │
│  │              TRANSFORMER BACKBONE (LLaMA-style)              │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  8 Layers × [RMSNorm → Attn → RMSNorm → SwiGLU FFN]   │  │   │
│  │  │  d_model=512, n_heads=8, d_ff=2048, RoPE positions     │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│           ┌───────────────────────┼───────────────────────┐         │
│           │                       │                       │         │
│  ┌────────▼────────┐  ┌──────────▼──────────┐  ┌────────▼────────┐ │
│  │   WORLD MODEL   │  │  PHYSICS RULE BANK  │  │   HIERARCHICAL  │ │
│  │    (TD-MPC2)    │  │   (100 rules)       │  │    PLANNER      │ │
│  │  latent=256     │  │   SymPy-grounded    │  │   (20 skills)   │ │
│  └────────┬────────┘  └──────────┬──────────┘  └────────┬────────┘ │
│           │                       │                       │         │
│           └───────────────────────┴───────────────────────┘         │
│                                   │                                  │
│                    ┌──────────────▼──────────────┐                  │
│                    │       OUTPUT HEADS          │                  │
│                    │  Action (17), Physics (10)  │                  │
│                    │  Value, Next State (256)    │                  │
│                    └─────────────────────────────┘                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## File Structure & Connections

```
JackTheLearner/
├── UnifiedBrain.py          # The brain (105M params)
│   ├── UnifiedBrainConfig   # All hyperparameters
│   ├── RMSNorm, SwiGLU      # LLaMA components
│   ├── TransformerBlock     # Main backbone
│   ├── WorldModel           # TD-MPC2 imagination
│   ├── HierarchicalPlanner  # HAC skills
│   ├── PhysicsRuleBank      # Neuro-symbolic
│   └── compute_*_loss()     # Loss functions
│
├── RobustTrainer.py         # Training system
│   ├── ReplayBuffer         # Anti-forgetting
│   ├── EWC                  # Elastic Weight Consolidation
│   ├── PhysicsConsistency   # Constraint checking
│   ├── obs_projection       # MuJoCo 376→256
│   └── train_phase0/1/2/2.5 # Training phases
│
├── SymbolicCalculator.py    # Ground truth physics
│   └── SymbolicPhysicsCalculator
│       ├── F=ma, torque, energy
│       └── predict_robot_state()
│
└── archive/                 # Full implementations
    ├── Phase1_Locomotion.py # Complete PPO+MuJoCo
    └── ScalableRobotBrain.py
```

## Training Pipeline

```
PHASE 0: Physics (15-30 min)
├── Input: Random states (256 dim)
├── Supervision: SymPy ground truth
├── Output: Physics predictions (10 values)
├── Saves: EWC Fisher info, Replay buffer
└── Result: Model learns F=ma, energy, momentum

         │
         │ EWC protects physics weights
         ▼

PHASE 1: Walking (4-12 hours)
├── Input: MuJoCo Humanoid-v5 (376→256)
├── Method: RL (PPO-style)
├── Safeguards: 20% replay, EWC penalty
├── Output: Walking policy
└── Result: Model can walk

         │
         │ EWC protects locomotion
         ▼

PHASE 2: Imitation (2-4 hours)
├── Input: Demo trajectories
├── Method: Flow Matching (pi0 style)
├── Safeguards: All active
├── Output: Natural movements
└── Result: Human-like motion

         │
         │ EWC protects skills
         ▼

PHASE 2.5: Language (1-2 hours)
├── Input: Commands + states
├── Commands: "walk forward", "stop", etc.
├── Method: Language-conditioned policy
├── Safeguards: All active
└── Result: Follows verbal instructions
```

## Comparison to SOTA

| Model | Params | Our Equivalent | Notes |
|-------|--------|----------------|-------|
| Octo (Berkeley) | 93M | ≈ JackTheLearner | Similar size, diffusion |
| OpenVLA | 7B | Need scaling | Pretrained VLM |
| RT-2 (Google) | 55B | Future goal | Massive VLM |
| π0 (Physical Int.) | ~3B | **ActionExpert + FlowMatching** | ✅ Architecture implemented |
| GR00T N1 (NVIDIA) | ~2B | **DualSystemController** | ✅ S0/S1/S2 implemented |
| Figure Helix | ~1B | **Dual System** | ✅ VLM+Visuomotor separation |

**JackTheLearner at 110M implements the SAME architecture patterns as π0, GR00T N1, and Figure Helix!**
The difference is scale (we use 110M vs their billions) and training data.

## Is This Architecture Good Enough?

### STRENGTHS:
1. **Unified backbone** - All modalities share same transformer
2. **TD-MPC2 WorldModel** - SOTA for model-based RL
3. **Neuro-symbolic physics** - SymPy grounds learning
4. **Anti-forgetting** - EWC + Replay prevents catastrophic forgetting
5. **Flow matching** - Better than diffusion for robotics (π0 style)
6. **Hierarchical planning** - Can compose skills
7. **Audio encoder** - Whisper + wav2vec2 for speech understanding
8. **Domain randomization** - DORAEMON/Humanoid-Gym style sim-to-real
9. **Dual System Architecture** - GR00T N1/Figure Helix style (NEW!)
10. **Action Expert** - Separate transformer for fast action generation (NEW!)
11. **LLM Integration** - Frozen backbone + trainable projector (NEW!)

### CURRENT GAPS:
1. **Vision not trained** - Encoder exists but unused
2. ~~**Language encoder simple**~~ - **FIXED: LLM integration added!**
3. ~~**No dual system**~~ - **FIXED: S0/S1/S2 implemented!**
4. **Skills not learned** - Planner exists but untrained
5. **Model could be bigger** - 256M-500M for more capacity

## LLM Encoder (NEW - Phase 2.5 Ready)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM LANGUAGE ENCODER                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Architecture (following OpenVLA, pi0, RT-2):                       │
│                                                                      │
│  "Walk to the red cup"                                              │
│          │                                                           │
│          ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  FROZEN LLM      │  ← NOT part of 105M brain weights             │
│  │                  │  ← Weights NEVER change                        │
│  │  Options:        │                                                │
│  │  • SmolLM2 1.7B  │  (default, best quality/size)                 │
│  │  • TinyLlama 1.1B│  (smaller, faster)                            │
│  │  • Gemma 2B      │  (Google's efficient model)                   │
│  │  • Phi-2 2.7B    │  (Microsoft's capable model)                  │
│  └────────┬─────────┘                                               │
│           │ hidden states (2048 dim)                                │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │  PROJECTOR       │  ← TRAINABLE (part of 105M brain)             │
│  │  2048 -> 1024    │  ← Learns to translate for robot              │
│  │  GELU + Dropout  │                                                │
│  │  1024 -> 512     │                                                │
│  │  + LayerNorm     │                                                │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              UNIFIED BRAIN (105M params)                      │   │
│  │  Cross-modal fusion → Transformer → Action heads              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Fallback (no HuggingFace):                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Token IDs  │ -> │  Embedding  │ -> │    LSTM     │ -> d_model  │
│  │             │    │  (vocab=1K) │    │  (2 layers) │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
│  Research:                                                           │
│  - OpenVLA (2024): Frozen Llama-2 + trainable action heads          │
│  - pi0 (Physical Intelligence 2024): Frozen PaliGemma + flow match  │
│  - RT-2 (Google 2023): Frozen PaLM-E + action tokens                │
│  - SmolVLA (HuggingFace 2024): Democratized VLA at 450M             │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Frozen LLM?**
1. LLM already understands language perfectly - don't retrain
2. Brain focuses on physics + motor control
3. Prevents catastrophic forgetting of language understanding
4. Can swap LLMs anytime (upgrade SmolLM → Llama 3.2 → GPT-4)
5. Enables local (on-device) vs cloud (API) flexibility

**Usage:**
```python
# With LLM (Colab/GPU with HuggingFace)
config = UnifiedBrainConfig()
config.llm_enabled = True
config.llm_backend = "smollm"  # or "tinyllama", "gemma"
brain = UnifiedBrain(config)

# Natural language commands
action = brain.act_with_language(state, "walk forward slowly")
action = brain.act_with_language(state, "pick up the red cup")
```

## Audio Encoder (NEW)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AUDIO ENCODER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option 1: Pretrained (HuggingFace)                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Whisper   │    │  wav2vec2   │    │  Projector  │             │
│  │   (tiny)    │ +  │   (base)    │ -> │  768->512   │ -> d_model  │
│  │  Speech2Txt │    │  Embeddings │    │             │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
│  Option 2: CNN Fallback (no HuggingFace)                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Waveform   │ -> │ Mel Spect.  │ -> │  CNN + MLP  │ -> d_model  │
│  │  (16kHz)    │    │  (80 mels)  │    │  (128->512) │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
│  Research: Whisper (OpenAI 2022), wav2vec2 (Meta 2020),            │
│            ES3 (CVPR 2024), WavTokenizer (ICLR 2025)               │
└─────────────────────────────────────────────────────────────────────┘
```

## Domain Randomization (NEW)

```
┌─────────────────────────────────────────────────────────────────────┐
│               DOMAIN RANDOMIZATION (Sim-to-Real)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Applied per-episode during Phase 1 RL training:                   │
│                                                                     │
│  PHYSICS PARAMETERS (DORAEMON style):                              │
│  ├── body_mass:      ±20% (default)                                │
│  ├── body_inertia:   ±20%                                          │
│  ├── geom_friction:  ±30%                                          │
│  ├── dof_damping:    ±20%                                          │
│  ├── actuator_gain:  ±10%                                          │
│  └── joint_friction: ±50%                                          │
│                                                                     │
│  SENSOR & ACTUATOR NOISE:                                          │
│  ├── sensor_noise:   Gaussian, std=0.01                            │
│  └── action_delay:   0-2 steps (motor latency)                     │
│                                                                     │
│  Research: DORAEMON (ICLR 2024), Humanoid-Gym (ICRA 2024)          │
│  Expected: 84-93% sim-to-real success rate                         │
└─────────────────────────────────────────────────────────────────────┘
```

### SCALING ROADMAP:
```
JackTheLearner-Small: 105M (current)
  └── 8 layers, 512 dim

JackTheLearner-Base: 256M (next)
  └── 12 layers, 768 dim

JackTheLearner-Large: 500M (future)
  └── 16 layers, 1024 dim
  └── Pretrained language encoder (DistilBERT)
  └── Pretrained vision (DINOv2)
```

## Data Flow Example

```python
# Phase 2.5: Language-conditioned action

state = torch.randn(B, 256)           # Robot state
language = tokenize("walk forward")   # Command tokens

# 1. Encode inputs
proprio_emb = proprio_encoder(state)  # (B, 512)
lang_emb = language_encoder(language) # (B, 512)

# 2. Fuse modalities
tokens = [proprio_emb, lang_emb, cls_token]
fused = cross_modal_fusion(tokens)    # (B, seq, 512)

# 3. Transformer backbone
for layer in transformer_layers:
    fused = layer(fused)              # (B, seq, 512)

# 4. Output heads
action = action_head(fused)           # (B, 16, 17)
physics = physics_head(fused)         # (B, 10)

# 5. WorldModel can imagine future
future_states = world_model.imagine(fused, action)
```

## Anti-Forgetting System

```
┌────────────────────────────────────────────────┐
│         CATASTROPHIC FORGETTING PREVENTION      │
├────────────────────────────────────────────────┤
│                                                 │
│  1. REPLAY BUFFER                              │
│     └── Mix 20% Phase 0 data into Phase 1/2   │
│                                                 │
│  2. EWC (Elastic Weight Consolidation)         │
│     └── Fisher info protects important weights │
│     └── penalty = λ × Σ(Fisher × Δweight²)    │
│                                                 │
│  3. MULTI-RATE LEARNING                        │
│     └── Backbone: 0.1x learning rate          │
│     └── Heads: 1x learning rate               │
│                                                 │
│  4. PHYSICS CONSISTENCY                        │
│     └── Penalize impossible predictions        │
│     └── Energy conservation check              │
│                                                 │
└────────────────────────────────────────────────┘
```

## Research Papers Implemented

| Paper | Component | Location |
|-------|-----------|----------|
| LLaMA (2023) | RMSNorm, SwiGLU, RoPE | UnifiedBrain.py:99-200 |
| TD-MPC2 (ICLR 2024) | WorldModel | UnifiedBrain.py:700-900 |
| HAC (2019) | HierarchicalPlanner | UnifiedBrain.py:900-1000 |
| OpenVLA (2024) | PrismaticVisionEncoder, LLM architecture | UnifiedBrain.py:253-306, LLMEncoder |
| EWC (2017) | Elastic Weight Consolidation | RobustTrainer.py:119-235 |
| pi0 (Physical Intelligence 2024) | Flow Matching, ActionExpert, Frozen LLM | UnifiedBrain.py:ActionExpert, FlowMatchingScheduler |
| RT-2 (Google 2023) | VLA paradigm, Frozen backbone | LLMEncoder architecture |
| SmolVLA (HuggingFace 2024) | Democratized VLA | LLMEncoder default backend |
| **GR00T N1 (NVIDIA 2025)** | **Dual System (S0/S1/S2), Action Expert** | **UnifiedBrain.py:DualSystemController** |
| **Figure Helix (2025)** | **Dual System, VLM+Visuomotor** | **UnifiedBrain.py:DualSystemController** |
| Flow Matching (Lipman 2022) | Conditional flow for diffusion | UnifiedBrain.py:FlowMatchingScheduler |
| DiT (Peebles 2023) | Diffusion Transformer | UnifiedBrain.py:ActionExpert time embedding |
| Whisper (2022) | Speech-to-Text | UnifiedBrain.py:AudioEncoder |
| wav2vec2 (2020) | Audio Embeddings | UnifiedBrain.py:AudioEncoder |
| DORAEMON (ICLR 2024) | Domain Randomization | RobustTrainer.py:DomainRandomization |
| Humanoid-Gym (ICRA 2024) | Zero-shot Sim2Real | RobustTrainer.py:DomainRandomization |
| GMR (ICRA 2026) | Motion Retargeting | MoCapLoader.py:SkeletonRetargeter |
| MoCapAct (NeurIPS 2022) | MoCap Dataset | MoCapLoader.py:MoCapDataset |
| LocoMuJoCo (2024) | Locomotion Benchmark | MoCapLoader.py:retarget_sequence |

## SOTA Action Generation (NEW - π0/GR00T Style)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DUAL SYSTEM ARCHITECTURE                             │
│              (NVIDIA GR00T N1 / Figure Helix / π0 Style)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  SYSTEM 2 - VLM REASONING (9 Hz)                               │     │
│  │  "Slow thinking" - Scene understanding, language comprehension │     │
│  │                                                                 │     │
│  │  Input: Vision + Language + State                              │     │
│  │  Output: Scene features (cached for System 1)                  │     │
│  │  Frequency: ~9 Hz (every 111ms)                                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                │                                         │
│                                │ Cached features (async)                 │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  SYSTEM 1 - ACTION EXPERT (50 Hz)                              │     │
│  │  "Fast actions" - Visuomotor policy with flow matching         │     │
│  │                                                                 │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │  ActionExpert (4 layers, 256 dim)                        │  │     │
│  │  │  - Cross-attention to VLM features                       │  │     │
│  │  │  - Sinusoidal time embeddings                            │  │     │
│  │  │  - Flow matching denoising (10 steps)                    │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  │                                                                 │     │
│  │  Input: Noisy action + VLM features + timestep                 │     │
│  │  Output: Smooth action chunk [16 steps, 17 joints]             │     │
│  │  Frequency: 50 Hz (every 20ms)                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                │                                         │
│                                │ Target joint positions                  │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  SYSTEM 0 - MOTOR CONTROL (1 kHz) [OPTIONAL]                   │     │
│  │  "Reflexes" - PD control for real hardware                     │     │
│  │                                                                 │     │
│  │  τ = Kp*(q_target - q) + Kd*(dq_target - dq)                   │     │
│  │  + Learned residual MLP for model mismatch                     │     │
│  │                                                                 │     │
│  │  Frequency: 1000 Hz (every 1ms) - disabled for sim             │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Research Papers:**

1. **π0 (Physical Intelligence, 2024)**: Flow matching for action generation
   - Paper: "π0: A Vision-Language-Action Flow Model for General Robot Control"
   - Contribution: Flow matching scheduler, action expert, 50Hz control

2. **GR00T N1 (NVIDIA, 2025)**: Dual system architecture
   - Paper: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
   - Contribution: System 0/1/2 hierarchy, cross-embodiment transfer

3. **Figure Helix (Figure AI, 2025)**: VLM + Visuomotor policy
   - Paper: "Helix: A Vision-Language-Action Model for Generalist Robots"
   - Contribution: Dual system separation, scene reasoning

4. **Flow Matching (Lipman et al., 2022)**: Conditional flow matching
   - Paper: "Flow Matching for Generative Modeling"
   - Contribution: Smoother than DDPM, fewer denoising steps

**Usage:**
```python
# Flow matching action generation (π0 style)
actions = brain.generate_actions_flow_matching(state, language="walk forward")

# Dual system (GR00T N1 style) - manages S2/S1 frequencies
result = brain.act_dual_system(state, language="pick up cup", current_time=t)
# result['action'] - single action for current timestep
# result['system2_ran'] - True if VLM was updated this tick

# Flow matching training loss
loss = brain.train_flow_matching_step(state, target_actions, language=cmd)
```

## MoCap Loader & Skeleton Retargeting (NEW)

```
+---------------------------------------------------------------------+
|                    MOCAP PIPELINE (Phase 2)                          |
+---------------------------------------------------------------------+
|                                                                      |
|  CMU MoCap BVH Files                                                |
|  (2500+ motions)                                                    |
|       |                                                              |
|       v                                                              |
|  +-------------------+     +------------------------+               |
|  |   BVH Parser      |---->|  Skeleton Retargeting  |               |
|  |  - Hierarchy      |     |  CMU 31 joints         |               |
|  |  - Rotations      |     |  --> MuJoCo 17 acts    |               |
|  |  - Frame timing   |     |                        |               |
|  +-------------------+     |  Joint Mapping:        |               |
|                            |  - Spine -> abdomen    |               |
|                            |  - UpLeg -> hip_xyz    |               |
|                            |  - Leg -> knee         |               |
|                            |  - Arm -> shoulder     |               |
|                            |  - ForeArm -> elbow    |               |
|                            +------------------------+               |
|                                      |                               |
|                                      v                               |
|  +-------------------+     +------------------------+               |
|  | Velocity Estimator|<----|  Actuator Values       |               |
|  | (finite diff)     |     |  (17 dims, [-0.4,0.4]) |               |
|  +-------------------+     +------------------------+               |
|            |                         |                               |
|            v                         v                               |
|  +--------------------------------------------------+               |
|  |              MoCapDataset                         |               |
|  |  - (obs, actions) pairs for imitation learning   |               |
|  |  - Context window: 10 frames                     |               |
|  |  - Action chunk: 16 steps (diffusion policy)     |               |
|  |  - Caching for fast reload                       |               |
|  +--------------------------------------------------+               |
|                             |                                        |
|                             v                                        |
|  +--------------------------------------------------+               |
|  |         Flow Matching Training (Phase 2)          |               |
|  |  Diffusion policy learns smooth, natural motion  |               |
|  +--------------------------------------------------+               |
|                                                                      |
+---------------------------------------------------------------------+
```

## Conclusion

**Is this the BEST architecture for endless possibilities?**

**At 105M params: YES, it's among the best for this size class.**
- Comparable to Octo (93M), which is SOTA
- Correct components (transformer, world model, planning)
- Proper training curriculum with anti-forgetting

**For truly "endless possibilities": NEEDS SCALING**
- Scale to 500M-1B params
- Add pretrained vision (DINOv2)
- Add pretrained language (DistilBERT)
- Train on diverse robot data

**But the ARCHITECTURE is correct - just needs more compute.**
