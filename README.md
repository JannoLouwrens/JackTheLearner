# Jack The Walker

**Neuro-Symbolic AGI for Humanoid Robots**

A research implementation combining 17+ AI papers into one unified brain.

---

## The Idea

Most robot brains either:
- **Learn from data** (neural networks) - flexible but don't understand physics
- **Use physics equations** (symbolic) - accurate but can't generalize

JackTheWalker does **both**: Neural networks learn patterns, SymPy verifies physics.

This is inspired by DeepMind's AlphaGeometry, which combined neural networks with a symbolic geometry solver to achieve IMO gold medal performance.

---

## Architecture

```
         ┌─────────────────────────────────────────────────────┐
         │              ENHANCED JACK BRAIN                    │
         ├─────────────────────────────────────────────────────┤
         │                                                     │
SYSTEM 1 │  ScalableRobotBrain (50Hz - Fast)                   │
  Fast   │  ├─ DINOv2 + SigLIP vision (OpenVLA)               │
         │  ├─ Cross-modal fusion transformer                  │
         │  └─ Flow matching diffusion → 1-step actions        │
         │                                                     │
─────────│─────────────────────────────────────────────────────│
         │                                                     │
SYSTEM 2 │  Reasoning Modules (1-5Hz - Slow)                   │
  Slow   │  ├─ WorldModel: "What if I do this?"                │
         │  ├─ MathReasoner: "Does this violate physics?"      │
         │  ├─ HierarchicalPlanner: "Break goal into skills"   │
         │  └─ AlphaGeometryLoop: "Novel solution needed"      │
         │                                                     │
         └─────────────────────────────────────────────────────┘

THREE MODES:
  REACTIVE (90%):  Just System 1 - fast reflexes
  VERIFIED (9%):   System 1 + physics check - safe
  CREATIVE (1%):   Full reasoning loop - novel problems
```

---

## Files Explained

### Brain (core architecture)

| File | What it does |
|------|--------------|
| `ScalableRobotBrain.py` | **System 1**: Fast reactive brain. VLA transformer with DINOv2+SigLIP vision, flow matching diffusion for 1-step action inference. Runs at 50Hz. |
| `EnhancedJackBrain.py` | **System 1+2 unified**: Combines fast brain with slow reasoning. Automatically selects mode based on confidence/novelty. |

### System 2 Modules (reasoning)

| File | What it does |
|------|--------------|
| `WorldModel.py` | **TD-MPC2 imagination**: Learns latent dynamics to imagine "what happens if I do X?". Used for planning without trial-and-error. |
| `MathReasoner.py` | **Neural physics**: Learns 100 physics rules (F=ma, torque, energy). Gets training data from SymPy. |
| `SymbolicCalculator.py` | **Exact physics**: SymPy equations for rigid body dynamics. Verifies neural predictions. Teacher for MathReasoner. |
| `HierarchicalPlanner.py` | **HAC skills**: 20 learnable skill embeddings. Breaks "walk to kitchen" into subgoals. Like a task manager. |
| `AlphaGeometryLoop.py` | **Creative solving**: Neural proposes action → Symbolic verifies → Refine → Repeat. For novel situations. |

### Training Pipeline

| File | What it does |
|------|--------------|
| `Phase0_Physics.py` | Train MathReasoner on physics. SymPy generates ground truth, neural learns to predict. ~2-3 days. |
| `Phase1_Locomotion.py` | RL walking in MuJoCo Humanoid-v5. PPO + WorldModel imagination + HAC skills. ~3-4 days. |
| `Phase2_Imitation.py` | Behavior cloning from demos. Flow matching diffusion on MoCapAct/RT-1 data. ~2-3 days. |

---

## How Key Components Work

### WorldModel (TD-MPC2)
```
Observation → [Encoder] → Latent state (z)
                              ↓
            [Dynamics] → Next latent (z')  ← predicts future
                              ↓
            [Reward] → Expected reward     ← evaluates plans
```
Instead of simulating real physics (slow), the WorldModel imagines outcomes in latent space. Used during training to improve value estimates and during runtime to evaluate action plans.

### HierarchicalPlanner (HAC)
```
Goal: "Walk to kitchen"
         ↓
    [Planner]
         ↓
    Skill #3: "Navigate"
         ↓
    Subgoal: "Turn right 45°"
         ↓
    Low-level: Joint torques
```
20 learnable skill embeddings act like "verbs" the robot knows. The planner selects which skill to use and generates subgoals. Skills are trained end-to-end with RL.

### AlphaGeometryLoop (Creative Mode)
```
State: "Stuck on obstacle"
Goal: "Reach other side"
         ↓
Neural: "Try jumping" (proposal)
         ↓
Symbolic: "Landing force = 2000N, max = 1500N" (verification)
         ↓
Neural: "Try lower jump" (refinement)
         ↓
Symbolic: "Force = 1200N, OK" (verified)
         ↓
Execute refined action
```
This loop is expensive (1Hz) so only used when System 1 has low confidence or high novelty. It's what makes the system AGI-like - it can solve problems it wasn't trained on.

---

## Quick Start

```bash
# Install
git clone https://github.com/JannoLouwrens/JackTheWalker.git
cd JackTheWalker
pip install -r requirements.txt

# Test physics (5 min)
python Phase0_Physics.py --samples 1000 --epochs 5

# Full training
python Phase0_Physics.py --samples 100000 --epochs 50
python Phase1_Locomotion.py --phase0-checkpoint checkpoints/phase0_best.pt
python Phase2_Imitation.py --checkpoint-in checkpoints/phase1_best.pt
```

---

## Research Foundation

This project implements ideas from 17+ papers. See [RESEARCH_PAPERS.md](RESEARCH_PAPERS.md) for details on HOW each paper was used.

Key influences:
- **AlphaGeometry** (Nature 2024): Neural-symbolic loop architecture
- **Physical Intelligence pi0** (2024): Flow matching for 1-step diffusion
- **OpenVLA** (Stanford 2024): DINOv2 + SigLIP vision fusion
- **TD-MPC2** (ICLR 2024): World model imagination
- **Thinking Fast and Slow** (Kahneman 2011): Dual-process architecture

---

## Status

| Component | Status |
|-----------|--------|
| ScalableRobotBrain (System 1) | Working |
| EnhancedJackBrain (System 1+2) | Working |
| Phase 0 (Physics) | Working |
| Phase 1 (RL + WorldModel + HAC) | Working |
| Phase 2 (Imitation) | Framework ready, needs real data |

---

## Author

**Janno Louwrens**
- BSc Computing (UNISA 2024)
- Honours in AI (UNISA 2026, in progress)

---

## License

MIT
