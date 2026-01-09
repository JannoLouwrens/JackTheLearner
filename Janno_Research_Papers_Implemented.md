# Research Papers & Techniques Implemented by Janno Louwrens

This document provides links to the academic research papers and techniques implemented in my projects, demonstrating deep understanding of cutting-edge AI/ML approaches.

---

## JackTheWalker Project - AGI Robotics System

### Vision-Language Models

**1. DINOv2: Learning Robust Visual Features without Supervision**
- Authors: Oquab et al. (Meta AI)
- Year: 2023
- Link: https://arxiv.org/abs/2304.07193
- Implementation: Used as self-supervised vision backbone (1024-dim features)

**2. SigLIP / CLIP: Learning Transferable Visual Models From Natural Language Supervision**
- Authors: Radford et al. (OpenAI)
- Year: 2021
- Link: https://arxiv.org/abs/2103.00020
- Implementation: Vision-language aligned features (768-dim), fused with DINOv2

**3. OpenVLA: An Open-Source Vision-Language-Action Model**
- Authors: Kim et al. (Stanford, Berkeley)
- Year: 2024
- Link: https://arxiv.org/abs/2406.09246
- Implementation: Prismatic architecture pattern (DINOv2 + SigLIP fusion)

---

### Diffusion Policies & Flow Matching

**4. Physical Intelligence π0: A Vision-Language-Action Flow Model**
- Authors: Physical Intelligence
- Year: 2024
- Link: https://www.physicalintelligence.company/blog/pi0
- Implementation: Flow matching for 1-step action inference

**5. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
- Authors: Chi et al. (Columbia, Toyota Research)
- Year: 2023
- Link: https://arxiv.org/abs/2303.04137
- Implementation: Denoising transformer for action generation

**6. Flow Matching for Generative Modeling**
- Authors: Lipman et al. (Meta AI)
- Year: 2023
- Link: https://arxiv.org/abs/2210.02747
- Implementation: Velocity field prediction for fast inference

---

### World Models & Model-Based RL

**7. TD-MPC2: Scalable, Robust World Models for Continuous Control**
- Authors: Hansen et al.
- Year: 2024 (ICLR)
- Link: https://arxiv.org/abs/2310.16828
- Implementation: Latent dynamics model with MPC planning

**8. DreamerV3: Mastering Diverse Domains through World Models**
- Authors: Hafner et al. (DeepMind)
- Year: 2023 (Nature Machine Intelligence)
- Link: https://arxiv.org/abs/2301.04104
- Implementation: Imagination-based learning architecture

---

### Neuro-Symbolic AI

**9. AlphaGeometry: Solving Olympiad Geometry without Human Demonstrations**
- Authors: Trinh et al. (DeepMind)
- Year: 2024 (Nature)
- Link: https://www.nature.com/articles/s41586-023-06747-5
- Achievement: IMO Silver Medal level performance
- Implementation: Neural proposer + Symbolic verifier loop for creative problem-solving

**10. AlphaProof: AI Achieves Silver Medal Standard in Mathematical Olympiad**
- Authors: DeepMind
- Year: 2024
- Link: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
- Implementation: Neuro-symbolic reasoning for physics (adapted from geometry)

---

### Dual-Process Cognitive Architecture

**11. Thinking, Fast and Slow**
- Author: Daniel Kahneman
- Year: 2011 (Book)
- Implementation: System 1 (50Hz reactive) + System 2 (5Hz deliberative) architecture

**12. Robots Thinking Fast and Slow: On Dual Process Theory for Safe AI in Embodied Systems**
- Authors: Oxford Robotics Institute
- Year: 2024
- Link: https://arxiv.org/abs/2401.08929
- Implementation: Dual-system robot control architecture

---

### Hierarchical Reinforcement Learning

**13. Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL**
- Authors: Sutton, Precup, Singh
- Year: 1999 (Artificial Intelligence)
- Link: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf
- Implementation: Options framework with 20 learnable skills

**14. Hierarchical Actor-Critic (HAC)**
- Authors: Levy et al.
- Year: 2019 (ICLR)
- Link: https://arxiv.org/abs/1712.00948
- Implementation: Three-level hierarchy (high/mid/low)

---

### Critical Research Insight

**15. On the Ineffectiveness of Pre-Trained Representations for Model-Based RL**
- Year: November 2024
- Finding: Frozen pretrained weights hurt RL performance; fine-tuning with 10x lower LR is correct
- Implementation: All pretrained components fine-tuned during RL phase

---

## Vineyard Robot Project - Autonomous Agriculture

### Object Detection

**16. YOLOv8: Ultralytics YOLO**
- Authors: Ultralytics
- Year: 2023
- Link: https://docs.ultralytics.com/
- Implementation: Custom-trained tree trunk detector

### Navigation & SLAM

**17. Probabilistic Robotics (Book)**
- Authors: Thrun, Burgard, Fox
- Year: 2005
- Implementation: GPS waypoint navigation, sensor fusion concepts

---

## Stock Market Predictions Project - Pre-AI Era (2023)

### Sentiment Analysis

**18. VADER: A Parsimonious Rule-based Model for Sentiment Analysis**
- Authors: Hutto & Gilbert
- Year: 2014 (ICWSM)
- Link: https://ojs.aaai.org/index.php/ICWSM/article/view/14550
- Implementation: Rule-based sentiment scoring

**19. Flair: A State-of-the-Art NLP Framework**
- Authors: Akbik et al. (Zalando Research)
- Year: 2018 (NAACL)
- Link: https://aclanthology.org/C18-1139/
- Implementation: Deep learning sentiment classification

---

## Summary Statistics

| Category | Papers/Techniques Implemented |
|----------|------------------------------|
| Vision-Language Models | 3 |
| Diffusion/Flow Matching | 3 |
| World Models/MBRL | 2 |
| Neuro-Symbolic AI | 2 |
| Cognitive Architecture | 2 |
| Hierarchical RL | 2 |
| Object Detection | 1 |
| NLP/Sentiment | 2 |
| **Total** | **17+** |

---

## Why This Matters

Implementing these research papers demonstrates:

1. **Research Literacy** - Ability to read, understand, and implement academic papers
2. **Systems Thinking** - Integrating multiple complex techniques into cohesive systems
3. **Technical Depth** - Going beyond tutorials to state-of-the-art approaches
4. **Self-Directed Learning** - Continuously staying current with 2024-2025 research
5. **Problem-Solving** - Adapting techniques (e.g., AlphaGeometry for physics instead of geometry)

This level of engagement with cutting-edge research is typically seen in graduate students and research engineers, demonstrating capability for advanced technical roles.
