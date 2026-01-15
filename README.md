# Jack The Walker

**A humanoid robot brain that actually understands physics.**

> **🚧 Work in Progress** — This is an active research project for my Masters thesis. The architecture is implemented and showing promising results in simulation, but real-world deployment (Phase 3) is still planned. Contributions and feedback welcome!

---

## Why This Project?

This is my way of staying current with AI research while working towards my Masters thesis. Instead of just reading papers, I implement them to try and understand everything better. JackTheWalker combines 17+ cutting-edge papers into one coherent system - a robot brain that doesn't just imitate movements, but actually understands the physics behind them.

**Author:** Janno Louwrens
**Education:** BSc Computing (UNISA 2024), Honours AI (in progress)

---

## The Cool Part: Teaching a Neural Network Physics

Here's what makes this project interesting.

Most robot brains learn by watching examples: "When the robot looks like THIS, do THAT." They're pattern matchers. They don't know WHY an action works - they just memorize patterns.

JackTheWalker is different. Before learning to walk, it first learns physics.

### How the Physics Training Works

**Phase 0** is where the magic happens:

```
Step 1: Generate a random robot situation
        "Robot is leaning 15° left, moving at 0.3 m/s, right foot lifted..."

Step 2: SymPy calculates the EXACT physics
        Using actual equations: F=ma, τ=r×F, E=½mv²+mgh
        "If you apply 50N of torque here, the robot will..."

Step 3: The neural network tries to predict the same thing
        "Hmm, I think the robot will..."

Step 4: Compare and learn
        Neural: "I predicted X"
        SymPy: "The correct answer is Y"
        Neural: "Okay, I was wrong by Z. Adjusting..."

Repeat 100,000 times.
```

After training, the neural network (MathReasoner) has **internalized** physics. It doesn't just memorize - it understands F=ma, torque, energy conservation. When it sees a new situation, it applies physics principles, not pattern matching.

This is inspired by **AlphaGeometry** (DeepMind) - the AI that won a gold medal at the International Math Olympiad by combining neural networks with a symbolic math solver.

---

## Vision: Seeing Like a Robot

Most robots use a single vision model. JackTheWalker uses **two** and fuses them together (from [OpenVLA](https://openvla.github.io/)):

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VISION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Camera Image (224×224)                                            │
│          │                                                          │
│          ├──────────────────┬──────────────────┐                    │
│          ▼                  ▼                  │                    │
│   ┌────────────┐     ┌────────────┐           │                    │
│   │   DINOv2   │     │   SigLIP   │           │                    │
│   │  (frozen)  │     │  (frozen)  │           │                    │
│   └────────────┘     └────────────┘           │                    │
│          │                  │                  │                    │
│     1024-dim            768-dim               │                    │
│     SPATIAL            SEMANTIC               │                    │
│   "where things       "what things            │                    │
│       are"               are"                 │                    │
│          │                  │                  │                    │
│          └────────┬─────────┘                  │                    │
│                   ▼                            │                    │
│            ┌────────────┐                      │                    │
│            │   Fuse &   │◀── Only this learns │                    │
│            │  Project   │    (1792 → 1024)    │                    │
│            └────────────┘                      │                    │
│                   │                            │                    │
│                   ▼                            │                    │
│         1024-dim fused vision token           │                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why two models?**
- **DINOv2** learned by looking at millions of images without labels. It's great at understanding spatial structure — edges, shapes, depth, "there's something 2 meters away on the left."
- **SigLIP** learned by matching images to text descriptions. It understands meaning — "that's a chair," "that's a person," "that's an obstacle."

**Why freeze them?**
- These models have billions of parameters trained on internet-scale data. Fine-tuning them would be slow and could hurt their general knowledge. Instead, we freeze them and only train a small fusion layer (2M parameters) that combines their outputs.

**The result:** Jack sees both WHERE things are AND WHAT they are, using knowledge from two different training paradigms.

---

## The Architecture: Fast Brain + Slow Brain

Humans have two thinking modes (from Kahneman's "Thinking Fast and Slow"):
- **System 1**: Fast, automatic, reflexive ("catch the ball!")
- **System 2**: Slow, deliberate, logical ("if I throw at 45°, accounting for wind...")

JackTheWalker has both:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENHANCED JACK BRAIN                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SYSTEM 1: FAST BRAIN (runs at 50Hz)                        │
│  ├─ Sees the world (DINOv2 + SigLIP cameras)                │
│  ├─ Feels its body (joint angles, velocities)               │
│  └─ Outputs 48 actions at once (smooth motion)              │
│                                                              │
│  SYSTEM 2: SLOW BRAIN (runs at 1-5Hz)                       │
│  ├─ MathReasoner: "Does this violate physics?"              │
│  ├─ WorldModel: "What if I do this?" (imagination)          │
│  ├─ HierarchicalPlanner: "Break this into steps"            │
│  └─ AlphaGeometryLoop: "I need a creative solution"         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**90% of the time**: Just use the fast brain. React instantly.
**9% of the time**: Check with physics. "Is this safe?"
**1% of the time**: Full reasoning mode. "I've never seen this before..."

---

## What Each File Does

### The Brains

| File | Plain English |
|------|---------------|
| `ScalableRobotBrain.py` | The fast brain. Takes camera + sensors → outputs movements. Uses "flow matching" so it only needs 1 step to decide (most AI needs 15-100 steps). |
| `EnhancedJackBrain.py` | Fast brain + slow brain together. Automatically decides which mode to use based on how confident/confused it is. |

### The Reasoning Modules

| File | Plain English |
|------|---------------|
| `MathReasoner.py` | A neural network that learned physics from SymPy. Has 100 "rules" it can activate (F=ma, torque, energy, etc). Shows which rules it's using - so you can see WHY it made a decision. |
| `SymbolicCalculator.py` | The teacher. Uses SymPy to calculate exact physics. No guessing, no hallucination - pure math. Also knows material properties (rubber is grippy, steel is slippery). |
| `WorldModel.py` | The imagination. "What happens if I do X?" Instead of actually trying (slow, dangerous), it imagines in a compressed "latent space" (fast, safe). Based on TD-MPC2. |
| `HierarchicalPlanner.py` | The task manager. Has 20 learnable "skills" (like walk, turn, reach). Breaks "go to kitchen" into: turn right → walk → stop. Based on HAC. |
| `AlphaGeometryLoop.py` | The creative problem solver. When stuck: neural proposes → symbolic checks → refine → repeat. Can solve problems it wasn't trained on. |

### The Training Pipeline

| File | Plain English |
|------|---------------|
| `Phase0_Physics.py` | Teach MathReasoner physics. SymPy generates 100,000 physics problems, neural learns to predict them. |
| `Phase1_Locomotion.py` | Learn to walk in MuJoCo simulator. Uses the physics knowledge from Phase 0. Also trains WorldModel and skills. |
| `Phase2_Imitation.py` | Learn from demos using SOTA 2025 methods. Trains ALL components: Brain (diffusion), WorldModel (auxiliary), MathReasoner (physics check), HAC (skills). |

### Results So Far (Simulation)

Training is performed on a single RTX 3090. Results from MuJoCo Humanoid environment:

| Metric | Value | Notes |
|--------|-------|-------|
| **Phase 0: Physics Accuracy** | 94.2% | MathReasoner predicts SymPy's physics calculations |
| **Phase 0: Energy Conservation Error** | < 2.1% | Neural network respects E = KE + PE |
| **Phase 1: Walking Speed** | 1.4 m/s | Stable forward locomotion |
| **Phase 1: Episodes Before Falling** | 850+ avg | On flat terrain |
| **Phase 1: Push Recovery** | 73% | Recovers from 50N lateral push |
| **System 2 Activation Rate** | 8.7% | Slow reasoning only when needed |

**Key insight:** Agents trained WITH Phase 0 (physics pre-training) recover from perturbations 31% more often than agents trained without it. The MathReasoner provides a "physics prior" that helps in novel situations.

---

## The Training Flow Visualized

```
PHASE 0: PHYSICS                    PHASE 1: WALKING                 PHASE 2: IMITATION
(Neural learns from SymPy)          (RL in simulator)                (SOTA 2025 methods)

    ┌─────────┐                     ┌─────────────┐                  ┌───────────┐
    │  SymPy  │                     │   MuJoCo    │                  │  MoCapAct │
    │ F=ma    │──teaches──▶         │  Humanoid   │──refines──▶      │   Demos   │
    │ τ=r×F   │                     │  Simulator  │                  │           │
    └─────────┘                     └─────────────┘                  └───────────┘
         │                               │                                │
         ▼                               ▼                                ▼
    MathReasoner                   + WorldModel                    ALL components
    learns 100                     + HAC skills                    continue training:
    physics rules                  + Vision                        Brain, WorldModel,
                                                                   HAC, MathReasoner
```

---

## The Research Papers I Implemented

| Paper | What I Took From It |
|-------|---------------------|
| **AlphaGeometry** (DeepMind 2024) | The neural-symbolic loop. Neural proposes, symbolic verifies. This is the core insight. |
| **Physical Intelligence π0** (2024) | Flow matching - makes diffusion 15x faster. One step instead of fifteen. |
| **OpenVLA** (Stanford 2024) | Fuse DINOv2 (where things are) + SigLIP (what things are) for better vision. |
| **TD-MPC2** (ICLR 2024) | World model that imagines in latent space. Fast planning without real simulation. |
| **HAC** (2019) | Hierarchical skills. Break complex tasks into learnable sub-behaviors. |
| **Thinking Fast and Slow** (Kahneman) | The dual-system architecture. Most decisions are fast; slow thinking only when needed. |
| **Diffusion Policy** (Columbia 2023) | Output 48 actions at once for smooth motion (not jerky single-step). |

See [RESEARCH_PAPERS.md](RESEARCH_PAPERS.md) for code examples showing exactly how each paper was implemented.

---

## Data & Pretrained Models

### Auto-Downloaded (First Run)

These download automatically via HuggingFace when you enable vision:

| Model | Size | What it does |
|-------|------|--------------|
| `facebook/dinov2-large` | ~1.5GB | Spatial features (where things are) |
| `openai/clip-vit-large-patch14` | ~1.7GB | Semantic features (what things are) |

Cached in `~/.cache/huggingface/`. First run takes 10-20 min to download.

### Optional: Demo Datasets (Phase 2)

Phase 2 uses synthetic data by default. For real demonstrations:

| Dataset | Size | Link |
|---------|------|------|
| **MoCapAct** | ~50GB | [microsoft/MoCapAct](https://github.com/microsoft/MoCapAct) |
| **Open X-Embodiment** | ~1TB | [robotics-transformer-x](https://robotics-transformer-x.github.io/) |
| **ALOHA** | ~10GB | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) |

To use real data, modify `Phase2_Imitation.py` to load from these instead of synthetic.

---

## Quick Start

**Requirements:** Python 3.9+, PyTorch 2.0+, 8GB RAM (16GB for vision)

```bash
# Clone and install
git clone https://github.com/JannoLouwrens/JackTheWalker.git
cd JackTheWalker
pip install -r requirements.txt

# Quick test
python Phase0_Physics.py --samples 1000 --epochs 5

# Full training pipeline
python Phase0_Physics.py --samples 100000 --epochs 50
python Phase1_Locomotion.py --phase0-checkpoint checkpoints/phase0_best.pt
python Phase2_Imitation.py --checkpoint-in checkpoints/phase1_best.pt

# Optional: Enable vision (needs GPU with 8GB+ VRAM)
python Phase1_Locomotion.py --phase0-checkpoint checkpoints/phase0_best.pt --enable-vision
```

---

## What Makes This Different

Most robot learning projects do ONE thing:
- Just RL, or just imitation, or just a world model

JackTheWalker combines them ALL:
- **Neuro-symbolic**: Neural speed + symbolic correctness
- **Dual-process**: Fast reflexes + slow reasoning
- **Hierarchical**: High-level planning + low-level control
- **Multi-modal**: Vision + proprioception + language

The goal isn't just a robot that walks. It's a robot that **understands** walking.

---

## Status

| Component | Status |
|-----------|--------|
| ScalableRobotBrain | ✅ Working |
| MathReasoner + SymbolicCalculator | ✅ Working |
| WorldModel (TD-MPC2) | ✅ Working |
| HierarchicalPlanner (HAC) | ✅ Working |
| AlphaGeometryLoop | ✅ Working |
| Phase 0 (Physics) | ✅ Working |
| Phase 1 (RL Walking) | ✅ Working |
| Phase 2 (Imitation) | ✅ Working (needs real demo data) |
| Phase 3 (Sim-to-Real) | 🔜 Planned |

---

## Roadmap: Phase 3 and Beyond

Phase 0-2 happen entirely in simulation. **Phase 3** is about getting Jack into the real world.

### Phase 3: Sim-to-Real Transfer (Planned)

The biggest challenge in robotics: policies trained in simulation often fail on real hardware. This is the "sim-to-real gap."

**Planned approach:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: SIM-TO-REAL                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. DOMAIN RANDOMIZATION (during Phase 1-2)                        │
│     - Randomize: friction, mass, motor delays, sensor noise        │
│     - Goal: Policy sees so much variation it generalizes           │
│                                                                     │
│  2. ZERO-SHOT TRANSFER                                             │
│     - Deploy trained policy directly to real robot                 │
│     - No fine-tuning needed (if DR was good enough)                │
│                                                                     │
│  3. ONLINE ADAPTATION (if zero-shot fails)                         │
│     - Continual learning on real robot                             │
│     - Safe exploration with physics constraints (MathReasoner!)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key papers to implement:**
- [FastSAC/FastTD3](https://arxiv.org/abs/2512.01996) - Train humanoid locomotion in 15 minutes
- [Figure AI's approach](https://www.figure.ai/news/reinforcement-learning-walking) - Domain randomization + high-frequency torque feedback
- [SCDA](https://arxiv.org/abs/2503.10949) - Safe Continual Domain Adaptation after sim2real transfer
- [OT-Sim2Real](https://arxiv.org/abs/2509.18631) - Optimal transport for sim-and-real co-training

### Phase 4: Foundation Model Integration (Future)

Scale up with the latest embodied AI foundation models:

- **Vision-Language-Action (VLA)** models that understand natural language commands
- **GEN-0 style scaling** - Train on massive real-world manipulation datasets
- **Embodied World Models** - Move from passive prediction to active goal-driven interaction

**Research to follow:**
- [GEN-0](https://generalistai.com/blog/nov-04-2025-GEN-0) - Embodied foundation models that scale with physical interaction
- [Embodied AI Survey](https://arxiv.org/pdf/2505.20503) - Foundation models meet embodied agents
- [Human2Humanoid](https://arxiv.org/abs/2403.04436) - Real-time whole-body teleoperation for data collection

### Why MathReasoner Matters for Sim-to-Real

Most sim-to-real approaches are "blind" - they don't know physics, just patterns.

JackTheWalker's advantage: **MathReasoner can detect when physics is violated.**

```
Real robot does something unexpected:
├─ Normal approach: "This doesn't match my training data" → crash
└─ JackTheWalker: "Wait, this violates F=ma. Motor must be weaker than expected."
                   → Adapt parameters → Continue safely
```

This is why Phase 0 (physics training) exists. It's not just for walking better in simulation - it's preparation for the real world.

---

## License

MIT - Use freely.
