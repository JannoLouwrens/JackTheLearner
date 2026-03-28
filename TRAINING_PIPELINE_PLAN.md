# JackTheWalker Training Pipeline - FINAL PLAN

## Core Principles

1. **Imitation First** - MoCap gives prior knowledge, RL refines it
2. **Reinforcement Loops Everywhere** - Every component learns from action outcomes
3. **Vision + LLM Before Manipulation** - See and understand before grasping
4. **Gradients Flow to All** - Single reward updates all contributing components

---

## PHASE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  PHASE 0: Physics Foundation                                        │
│      ↓                                                              │
│  PHASE 1: Imitation (MoCap) ← Learn what movements LOOK like       │
│      ↓                                                              │
│  PHASE 2: Locomotion RL ← Make imitated walking WORK               │
│      ↓                                                              │
│  PHASE 3: Perception ← Vision + Object Detection + LLM             │
│      ↓                                                              │
│  PHASE 4: Manipulation ← Reach/Grasp WITH vision + LLM feedback    │
│      ↓                                                              │
│  PHASE 5: Audio ← Speech commands + responses                       │
│      ↓                                                              │
│  PHASE 6: Planning ← Hierarchical + World Model + Navigation       │
│      ↓                                                              │
│  PHASE 7: Full Integration ← Everything together, Dual System      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 0: PHYSICS FOUNDATION

**Goal**: Understand how the world works (state + action → next state)

**What Trains**:
- State encoder
- Physics predictor

**Reinforcement Loop**:
```
State + Action → Predict next state → Actual next state → MSE loss
```

**No vision, no LLM, no imitation** - pure physics understanding.

**Inputs**: Proprioception only
**Outputs**: Next state prediction

---

## PHASE 1: IMITATION LEARNING (MoCap)

**Goal**: Learn what human movement LOOKS like before trying to do it

### Phase 1.1: Locomotion Imitation

**What**: Walking, running, turning from MoCap data

**Reinforcement Loop**:
```
MoCap motion: [walk forward]
     ↓
Robot tries to copy joint angles
     ↓
Discriminator: Does it look human?
Physics: Did robot fall?
     ↓
REWARD:
  + Matches MoCap pose
  + Stays upright
  + Smooth motion
     ↓
Update policy to be more human-like
```

**Research**: AMP (Adversarial Motion Priors), DeepMimic

### Phase 1.2: Upper Body Imitation

**What**: Reaching, arm movements, head motion from MoCap

**Reinforcement Loop**:
```
MoCap motion: [reach forward]
     ↓
Robot copies arm trajectory
     ↓
Discriminator: Does it look natural?
     ↓
REWARD:
  + Matches MoCap trajectory
  + Smooth arm motion
  + Natural pose
     ↓
Update arm control policy
```

### Phase 1.3: Manipulation Imitation

**What**: Grasping motions, object handling from MoCap

**Reinforcement Loop**:
```
MoCap motion: [grasp object]
     ↓
Robot copies finger/hand motion
     ↓
Discriminator: Does grasp look human?
     ↓
REWARD:
  + Matches MoCap hand shape
  + Natural finger curl
     ↓
Update hand control policy
```

### Phase 1.4: Combined Motion Imitation

**What**: Walking + reaching, carrying objects

**Reinforcement Loop**:
```
MoCap motion: [walk while carrying]
     ↓
Robot copies full body coordination
     ↓
REWARD:
  + Whole body matches MoCap
  + Coordination between legs/arms
     ↓
Update full body policy
```

**After Phase 1**: Robot LOOKS human-like but may not be robust yet.

---

## PHASE 2: LOCOMOTION RL (Refine Imitation)

**Goal**: Make imitated walking actually WORK in varied conditions

### Phase 2.1: Basic Walking RL

**What**: Refine walking with physics feedback

**Reinforcement Loop**:
```
Walk forward (using Phase 1 prior)
     ↓
Physics feedback:
  - Did I fall? (negative)
  - Did I move forward? (positive)
  - Am I efficient? (energy reward)
     ↓
REWARD refines the imitated motion
     ↓
Still looks human (MoCap prior) + Actually works (RL)
```

**Key**: MoCap prior prevents ugly motions, RL makes it robust

### Phase 2.2: Terrain Adaptation

**What**: Walk on stairs, slopes, rough terrain

**Reinforcement Loop**:
```
Terrain: stairs
     ↓
Use imitated walking as starting point
     ↓
Adapt foot placement, timing
     ↓
REWARD:
  + Climb successfully
  + Don't fall
  + Still look somewhat human
     ↓
Update policy for terrain
```

**Curriculum**: Flat → slight slope → stairs → rough → gaps

### Phase 2.3: Domain Randomization

**What**: Adapt to varying physics (mass, friction, delays)

**Reinforcement Loop**:
```
Randomize: mass ±20%, friction ±30%, action delay
     ↓
Walk with randomized physics
     ↓
REWARD: Still walk successfully
     ↓
Policy becomes robust to real-world variation
```

**After Phase 2**: Robot walks robustly AND looks human-like.

---

## PHASE 3: PERCEPTION FOUNDATION

**Goal**: See the world, detect objects, understand language

### Phase 3.1: Vision Training (with Action Feedback!)

**What**: Train DINOv2 + SigLIP to understand visual input

**Reinforcement Loop** (NOT just image labels!):
```
See scene → Extract features
     ↓
Features predict: "graspable object at position X"
     ↓
Robot reaches for position X
     ↓
Touched something?
  YES → Vision features were useful! (positive)
  NO  → Vision features were wrong! (negative)
     ↓
Update vision projector based on ACTION outcome
```

**Key**: Vision learns what features MATTER for action

### Phase 3.2: Object Detection Training

**What**: Train ObjectDetector to find objects

**Reinforcement Loop**:
```
Detect: "cup" at [1.5, 0.1, 0.78]
     ↓
Reach for that position
     ↓
Hand contacts cup?
  YES → Detection correct! (positive)
  NO  → Detection wrong! (negative)
     ↓
Update object detector
```

**Not just position labels** - learns from grasp success!

### Phase 3.3: LLM Projector Training

**What**: Connect language to robot's representation space

**Reinforcement Loop**:
```
Command: "move forward"
     ↓
LLM (frozen) → embeddings → Projector (trainable) → action embedding
     ↓
Motor executes action
     ↓
Did robot move forward?
  YES → Projector understood! (positive)
  NO  → Projector misunderstood! (negative)
     ↓
Update projector
```

**Synonyms get same reward**: "walk forward" = "go ahead" = "move forward"

### Phase 3.4: Language-Vision Grounding

**What**: Connect words to visual objects

**Reinforcement Loop**:
```
Command: "look at the cup"
     ↓
LLM: understand "cup"
Vision: find cup-like object in scene
     ↓
Robot turns head toward detected object
     ↓
Is it actually a cup? (ground truth from MuJoCo)
  YES → Vision-language aligned! (positive)
  NO  → Misalignment! (negative)
     ↓
Update both vision and LLM projector
```

**After Phase 3**: Robot can see, detect objects, and understand basic commands.

---

## PHASE 4: VISION-GUIDED MANIPULATION

**Goal**: Grasp objects using vision + language, with full feedback loops

### Phase 4.1: Vision-Guided Reaching

**What**: Reach for visually detected objects

**Full Reinforcement Loop**:
```
Command: "reach for the cup"
     ↓
┌─────────────────────────────────────────────────────────┐
│ LLM Projector: parse "reach" + "cup"                   │
│      ↓                                                  │
│ Vision: see scene                                       │
│      ↓                                                  │
│ Object Detector: "cup" at [1.5, 0.1, 0.78]            │
│      ↓                                                  │
│ Motor: reach to position (using Phase 1.2 prior!)     │
│      ↓                                                  │
│ Feedback: hand near cup? (distance < 5cm)              │
│      ↓                                                  │
│ REWARD → Updates ALL:                                   │
│   • Motor (reaching accuracy)                           │
│   • Object Detector (position correctness)             │
│   • Vision (feature usefulness)                         │
│   • LLM Projector (command understanding)              │
└─────────────────────────────────────────────────────────┘
```

### Phase 4.2: Vision-Guided Grasping

**What**: Grasp visually detected objects

**Full Reinforcement Loop**:
```
Command: "grasp the bottle"
     ↓
┌─────────────────────────────────────────────────────────┐
│ LLM: parse "grasp" + "bottle"                          │
│      ↓                                                  │
│ Vision: locate bottle                                   │
│      ↓                                                  │
│ Object Detector: bottle at [1.5, -0.15, 0.82]          │
│      ↓                                                  │
│ Motor: reach (Phase 1.2) + close fingers (Phase 1.3)  │
│      ↓                                                  │
│ Feedback: bottle lifted off table?                      │
│      ↓                                                  │
│ REWARD → Updates ALL components                         │
└─────────────────────────────────────────────────────────┘
```

### Phase 4.3: Vision-Guided Loco-Manipulation

**What**: Walk + carry using vision

**Full Reinforcement Loop**:
```
Command: "bring the cup to the counter"
     ↓
┌─────────────────────────────────────────────────────────┐
│ LLM: parse task                                         │
│      ↓                                                  │
│ Vision: find cup, find counter                          │
│      ↓                                                  │
│ Motor: grasp cup + walk + navigate + place             │
│      ↓                                                  │
│ Continuous visual feedback during walk                  │
│      ↓                                                  │
│ Feedback at each step:                                  │
│   • Cup still in hand? (grasp maintained)              │
│   • Moving toward counter? (navigation)                │
│   • Placed on counter? (task complete)                 │
│      ↓                                                  │
│ REWARD flows through entire system                     │
└─────────────────────────────────────────────────────────┘
```

**After Phase 4**: Robot can see objects, understand commands, and manipulate.

---

## PHASE 5: AUDIO INTEGRATION

**Goal**: Understand speech commands, respond verbally

### Phase 5.1: Speech Recognition (Whisper)

**Reinforcement Loop**:
```
Audio: "pick up the cup" (spoken)
     ↓
Whisper → transcription → LLM → action
     ↓
Task executed
     ↓
Cup picked up?
  YES → Transcription was correct!
  NO  → Maybe misheard?
     ↓
Update Whisper projector
```

### Phase 5.2: Speech Response (TTS)

**Reinforcement Loop**:
```
Task: "pick up the cup"
     ↓
Robot attempts task
     ↓
Success → Robot says "Done, I picked up the cup"
Failure → Robot says "I couldn't reach the cup"
     ↓
Human feedback: "Good" / "Try again"
     ↓
Update response generation
```

**After Phase 5**: Robot can hear commands and speak back.

---

## PHASE 6: ADVANCED PLANNING

**Goal**: Complex task decomposition, world modeling, navigation

### Phase 6.1: Hierarchical Planning

**What**: Break complex tasks into subtasks

**Reinforcement Loop**:
```
Task: "make coffee"
     ↓
High-level planner: [go to kitchen, find cup, find machine, ...]
     ↓
Execute each subtask
     ↓
Each subtask success → reward to planner
Task complete → BIG reward
     ↓
Update hierarchical planner
```

### Phase 6.2: World Model (TD-MPC2)

**What**: Predict outcomes before acting

**Reinforcement Loop**:
```
World model predicts: "If I push cup, it will fall"
     ↓
Actually push cup
     ↓
Compare prediction to reality
     ↓
Prediction error → Update world model
```

### Phase 6.3: Navigation Planning

**What**: Path planning with obstacle avoidance

**Reinforcement Loop**:
```
Goal: "go to kitchen"
     ↓
Nav planner: plan path
Vision: detect obstacles
     ↓
Execute path with continuous vision
     ↓
Arrived at kitchen?
     ↓
REWARD → Update navigation planner
```

### Phase 6.4: Memory-Augmented Planning

**What**: Remember past experiences for better planning

**Reinforcement Loop**:
```
Task: "bring coffee like last time"
     ↓
Memory: recall previous coffee-making
     ↓
Use recalled plan
     ↓
Success?
     ↓
REWARD → Update memory retrieval
```

**After Phase 6**: Robot can plan complex tasks, predict outcomes, navigate.

---

## PHASE 7: FULL INTEGRATION + DUAL SYSTEM

**Goal**: All components working together at different timescales

### Phase 7.1: Dual System Training

**What**: Coordinate S0 (motor), S1 (action), S2 (planning)

**Multi-timescale Reinforcement**:
```
┌─────────────────────────────────────────────────────────┐
│ System 2 (2-5 Hz): High-level planning                 │
│   "I need to grasp the cup"                            │
│        ↓                                                │
│ System 1 (10-20 Hz): Action chunk generation           │
│   [reach motion over next 0.5s]                        │
│        ↓                                                │
│ System 0 (500 Hz): Low-level PD control               │
│   [individual joint torques]                           │
│        ↓                                                │
│ Physics execution                                       │
│        ↓                                                │
│ REWARD flows back through ALL systems:                 │
│   • S0 learns optimal PD gains                         │
│   • S1 learns smooth action chunks                     │
│   • S2 learns good plans                               │
└─────────────────────────────────────────────────────────┘
```

### Phase 7.2: End-to-End Complex Tasks

**What**: Full tasks with all systems active

**Reinforcement Loop**:
```
Task: "Make me coffee and bring it here"
     ↓
ALL SYSTEMS ACTIVE:
  • Audio: heard command
  • LLM: understood task
  • Vision: seeing environment
  • Planner: decomposed task
  • Navigator: planning paths
  • Motor: executing actions
  • World Model: predicting
  • Memory: using past experience
     ↓
Continuous execution with feedback
     ↓
Coffee delivered?
     ↓
MASSIVE REWARD → Updates entire system
```

### Phase 7.3: Continuous Learning

**What**: Keep improving from new experiences

**Reinforcement Loop**:
```
New task / New environment
     ↓
Execute with current abilities
     ↓
Learn from success/failure
     ↓
Update relevant components
     ↓
Better next time
```

---

## SUMMARY: GRADIENT FLOW

Every action outcome sends gradients to ALL contributing components:

```
                         REWARD (Task Success)
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
    ┌───────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
    │VISION │  │OBJ DET │  │  LLM   │  │ MOTOR  │  │PLANNER │
    │DINOv2 │  │        │  │PROJECTOR│  │POLICY  │  │        │
    │SigLIP │  │        │  │        │  │        │  │        │
    └───────┘  └────────┘  └────────┘  └────────┘  └────────┘
        │           │           │           │           │
        └───────────┴───────────┼───────────┴───────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │     BACKBONE      │
                    │   (Transformer)   │
                    └───────────────────┘
```

---

## PHASE DEPENDENCY GRAPH

```
Phase 0 (Physics)
    │
    ▼
Phase 1 (Imitation) ──────────────────────────────────┐
    │                                                  │
    ▼                                                  │
Phase 2 (Locomotion RL) ← refines Phase 1 walking    │
    │                                                  │
    ▼                                                  │
Phase 3 (Perception) ← needs robot that can move      │
    │                                                  │
    ▼                                                  │
Phase 4 (Manipulation) ← combines Phase 1.2/1.3 + Phase 3
    │                      (imitated reach/grasp + vision)
    ▼
Phase 5 (Audio) ← adds speech to existing capabilities
    │
    ▼
Phase 6 (Planning) ← needs manipulation + perception
    │
    ▼
Phase 7 (Integration) ← everything together
```

---

## IMPLEMENTATION CHECKLIST

### Phase 0
- [x] train_phase0() - unchanged

### Phase 1 (Imitation)
- [x] train_phase1_1_locomotion_imitation()
- [x] train_phase1_2_upper_body_imitation()
- [x] train_phase1_3_manipulation_imitation()
- [x] train_phase1_4_combined_imitation()
- [x] Add discriminator for AMP-style training

### Phase 2 (Locomotion RL)
- [x] train_phase2_1_walking_rl()
- [x] train_phase2_2_terrain_adaptation()
- [x] train_phase2_3_domain_randomization()

### Phase 3 (Perception)
- [x] train_phase3_1_vision() - with action feedback
- [x] train_phase3_2_object_detection() - with grasp feedback
- [x] train_phase3_3_llm_projector() - with execution feedback
- [x] train_phase3_4_language_vision_grounding()

### Phase 4 (Manipulation)
- [x] train_phase4_1_vision_guided_reaching()
- [x] train_phase4_2_vision_guided_grasping()
- [x] train_phase4_3_vision_guided_loco_manipulation()

### Phase 5 (Audio)
- [x] train_phase5_1_speech_recognition()
- [x] train_phase5_2_speech_response()

### Phase 6 (Planning)
- [x] train_phase6_1_hierarchical_planning()
- [x] train_phase6_2_world_model()
- [x] train_phase6_3_navigation()
- [ ] train_phase6_4_memory() (optional - can be added later)

### Phase 7 (Integration)
- [x] train_phase7_1_dual_system()
- [x] train_phase7_2_end_to_end()
- [ ] train_phase7_3_continuous_learning() (optional - can be added later)

---

## RESEARCH BACKING

| Phase | Technique | Paper/Source |
|-------|-----------|--------------|
| 1 | Imitation + Discriminator | AMP: Adversarial Motion Priors (2021) |
| 1 | MoCap Imitation | DeepMimic (2018) |
| 2 | Terrain Curriculum | Legged Gym (RSS 2022) |
| 2 | Domain Randomization | DORAEMON, Humanoid-Gym (2024) |
| 3 | Vision-Action Learning | RT-2 (2023), PaLM-E (2023) |
| 4 | Vision-Guided Manipulation | RoboFlamingo (2024) |
| 6 | Hierarchical Planning | HAC, SayCan (2022) |
| 6 | World Model | TD-MPC2 (2024) |
| 7 | Dual Process | π₀ (Physical Intelligence, 2024) |

---

## IMPLEMENTATION COMPLETE

This plan ensures:
✅ Imitation comes first (prior knowledge)
✅ RL refines imitation (robustness)
✅ Perception before manipulation (see then act)
✅ Every component gets reinforcement (no dead ends)
✅ Gradients flow everywhere (unified learning)
✅ All existing phases preserved and reorganized

**Implementation Status: COMPLETE**

All phases (0-7) have been implemented in RobustTrainer.py with full reinforcement loops.

Usage:
```bash
# Run full pipeline
python RobustTrainer.py --phase 0 --epochs 50   # Physics
python RobustTrainer.py --phase 1 --epochs 500  # Imitation
python RobustTrainer.py --phase 2 --epochs 100  # Locomotion RL
python RobustTrainer.py --phase 3 --epochs 200  # Perception
python RobustTrainer.py --phase 4 --epochs 300  # Manipulation
python RobustTrainer.py --phase 5 --epochs 150  # Audio
python RobustTrainer.py --phase 6 --epochs 200  # Planning
python RobustTrainer.py --phase 7 --epochs 300  # Integration
```
