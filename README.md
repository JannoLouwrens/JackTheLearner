# Jack The Walker

**Neuro-Symbolic AGI for Humanoid Robots**

A research implementation combining 17+ AI papers into one unified brain.

---

## What Is This?

A humanoid robot brain that **understands physics** instead of just imitating behaviors.

| Feature | How |
|---------|-----|
| Understands physics | Neural learns from symbolic (AlphaGeometry approach) |
| Fast + slow thinking | System 1 (50Hz reflexes) + System 2 (physics reasoning) |
| 1-step action inference | Flow matching diffusion (not 15-100 steps) |
| Learnable skills | 20 reusable behaviors via HAC |

---

## Design Rationale

JackTheWalker began as a research exercise in combining modern robotics papers into a single, coherent brain rather than isolated demos. I started with neuro-symbolic physics (Phase 0) to ensure the model could reason about dynamics, then moved to locomotion and skill acquisition so the brain could act, not just predict.

The architecture is deliberately dual-process: a fast reactive system handles immediate control, while a slower planner reasons about physics, goals, and long-horizon decisions. This mirrors cognitive science literature and provides a clear path to scale from simulated locomotion to richer embodied tasks.

## Files

```
JackTheWalker/
│
├── EnhancedJackBrain.py      # THE BRAIN (all architecture in one file)
│
├── Phase0_Physics.py         # Train: Neural learns physics from symbolic
├── Phase1_Locomotion.py      # Train: RL walking in MuJoCo
├── Phase2_Imitation.py       # Train: Behavior cloning from demos
│
├── WorldModel.py             # TD-MPC2 imagination
├── HierarchicalPlanner.py    # HAC skill decomposition
├── MathReasoner.py           # 100 physics rules (neural)
├── AlphaGeometryLoop.py      # Creative problem solving
├── SymbolicCalculator.py     # Exact physics (SymPy)
│
└── requirements.txt
```

**One brain file. Three training phases. Five support modules.**

---

## Training Pipeline

```
Phase 0                    Phase 1                    Phase 2
(Physics)                  (Walking)                  (Skills)
   │                          │                          │
   ▼                          ▼                          ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│SymPy    │──teaches──▶  │ MuJoCo  │──refines──▶  │ Demos   │
│Physics  │              │ Humanoid│              │MoCapAct │
└─────────┘              └─────────┘              └─────────┘
   │                          │                          │
   ▼                          ▼                          ▼
MathReasoner             + WorldModel              + Diffusion
learns 100               + HAC Skills               Policy
physics rules            + Vision                   fine-tune

2-3 days                 3-4 days                  2-3 days
```

### Run Training

```bash
# Phase 0: Physics (neural learns from symbolic calculator)
python Phase0_Physics.py --samples 100000 --epochs 50

# Phase 1: Walking (RL in MuJoCo Humanoid-v5)
python Phase1_Locomotion.py --phase0-checkpoint checkpoints/phase0_best.pt --epochs 1000

# Phase 1 with vision (optional, needs more GPU):
python Phase1_Locomotion.py --phase0-checkpoint checkpoints/phase0_best.pt --enable-vision

# Phase 2: Behavior cloning (currently uses synthetic data)
python Phase2_Imitation.py --checkpoint-in checkpoints/phase1_best.pt
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ENHANCED JACK BRAIN                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  SYSTEM 1: FAST (50Hz)                                       │
│  ├─ DINOv2 + SigLIP vision                                   │
│  ├─ Cross-modal fusion transformer                           │
│  └─ Diffusion policy (flow matching, 1-step)                 │
│                                                               │
│  SYSTEM 2: SLOW (1-5Hz)                                      │
│  ├─ WorldModel (TD-MPC2) - imagination                       │
│  ├─ MathReasoner - 100 physics rules                         │
│  ├─ HierarchicalPlanner - 20 learnable skills                │
│  └─ AlphaGeometryLoop - creative solving                     │
│                                                               │
│  Three Modes: REACTIVE (90%) | VERIFIED (9%) | CREATIVE (1%) │
└──────────────────────────────────────────────────────────────┘
```

---

## Research Papers

| Paper | Year | What We Use |
|-------|------|-------------|
| AlphaGeometry (DeepMind) | 2024 | Neural-symbolic loop |
| Physical Intelligence π0 | 2024 | Flow matching diffusion |
| OpenVLA (Stanford) | 2024 | DINOv2 + SigLIP fusion |
| TD-MPC2 (ICLR) | 2024 | World model imagination |
| Thinking Fast & Slow | 2011 | Dual-process architecture |
| HAC | 2019 | Hierarchical skills |
| Frozen Reps Ineffective | Nov 2024 | Fine-tune, don't freeze |

Full list: [Janno_Research_Papers_Implemented.md](Janno_Research_Papers_Implemented.md)

---

## Quick Test

```bash
# 5-minute physics test
python Phase0_Physics.py --samples 1000 --epochs 5

# Should see MathReasoner learning F=ma, torque, energy conservation
```

---

## Installation

```bash
git clone https://github.com/JannoLouwrens/JackTheWalker.git
cd JackTheWalker

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, 8GB RAM

---

## Status

| Component | Status |
|-----------|--------|
| EnhancedJackBrain | ✅ Complete (all in one file) |
| Phase 0 (Physics) | ✅ Working |
| Phase 1 (RL + WorldModel + HAC + Vision) | ✅ Working |
| Phase 2 (Imitation) | ⚠️ Framework ready, needs real data |
| Real demo datasets (MoCapAct/RT-1) | ❌ Not yet |

---

## Author

**Janno Louwrens**
- BSc Computing (UNISA 2024)
- Honours in AI (UNISA 2026, in progress)

---

## License

MIT - Use freely.
