# Research Papers & Techniques Implemented - JackTheWalker

This document lists the academic research papers and techniques implemented in the JackTheWalker project.

---

## Vision-Language Models

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

## Diffusion Policies & Flow Matching

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

## World Models & Model-Based RL

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

## Neuro-Symbolic AI

**9. AlphaGeometry: Solving Olympiad Geometry without Human Demonstrations**
- Authors: Trinh et al. (DeepMind)
- Year: 2024 (Nature)
- Link: https://www.nature.com/articles/s41586-023-06747-5
- Achievement: IMO Silver Medal level performance
- Implementation: Neural proposer + symbolic verifier loop for creative problem-solving

**10. AlphaProof: AI Achieves Silver Medal Standard in Mathematical Olympiad**
- Authors: DeepMind
- Year: 2024
- Link: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
- Implementation: Neuro-symbolic reasoning for physics (adapted from geometry)

---

## Dual-Process Cognitive Architecture

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

## Hierarchical Reinforcement Learning

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

## Critical Research Insight

**15. On the Ineffectiveness of Pre-Trained Representations for Model-Based RL**
- Year: November 2024
- Finding: Frozen pretrained weights hurt RL performance; fine-tuning with 10x lower LR is correct
- Implementation: All pretrained components fine-tuned during RL phase

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
| Critical Insight | 1 |
| **Total** | **15** |
