> ## SUPERSEDED IN PART — 2026-08-09
> **"Freeze a pretrained trunk and learn a small adapter" NO LONGER HOLDS for
> anything INSIDE Jack.** Owner decree: PLASTIC ONLY — his encoders, core and
> fusion all learn. Reason: a frozen tower's reshaping gain is identically
> zero, foreclosing a class of learning, which GOAL.md's capability target
> marks suspect by definition. Evidence, counterargument and re-open trigger:
> docs/CHAMPIONS.md. Still standing from this document: the LLM is frozen and
> swappable — but it is now a PARENT in his world, not a component inside him,
> so it is not governed by the plastic-only rule.

# Settled decisions

Made by the owner 2026-08-04. The loop treats these as fixed and does not
relitigate them without new evidence. Open items live in `DECISIONS_NEEDED.md`.

---

## D1 — Freeze the trunk, delete ~90% of the parameters. **ACCEPTED**

Keep a pretrained frozen trunk plus a ~22M trained adapter. Tag current HEAD,
delete on a branch.

Why it was not close: there is no data to train 118M from scratch — the MoCap
URLs 404, the error is swallowed, and `MoCapLoader.__getitem__` fabricates
sinusoids paired with *randomly drawn* language labels, which is anti-training
rather than merely uninformative. Separately, pretrained trunks barely forget
with trivial replay while small from-scratch policies forget catastrophically
(arXiv:2603.03818), so freezing is what makes the continual-learning goal
reachable at all.

## D2 — The goal is an actual working humanoid, not a thesis argument

The owner was explicit: *"our goal is an actual humanoid it's not for the
thesis we want the actual thing... choose the one that's best proven."*

So the criterion is what demonstrably produces working humanoids, and that
settles it in two parts rather than one:

**Locomotion — drop physics-first as a training method.** Nothing that walks in
2026 got there by pre-training on symbolic physics targets. What works is
massively parallel simulation, domain randomisation, and RMA-style online
adaptation. Physics-first is additionally contradicted by arXiv:2507.06952 and
arXiv:2111.05458. `SymbolicCalculator` survives as a **frozen regression gate**
and a unit-correct action limiter — it stops being a teacher and becomes a check.
T5.01 still runs, cheaply and early, so the decision rests on our own numbers.

**Companion behaviour — continual learning, on top.** It is what makes Jack
improve rather than merely function, it is the repo's biggest genuine hole, and
the wake/sleep pattern (SIESTA, arXiv:2303.10725) turns the ephemeral-GPU
constraint into the design rather than a limitation.

These are not competing answers. Locomotion is solved by RL + domain
randomisation; continual learning is what sits above it.

## D3 — Delete the fabricated results from the public README. **DONE**

The README stated 1.4 m/s walking, 850+ episodes before falling, 73% push
recovery, 94.2% physics accuracy, and a 31% improvement from physics
pre-training — as measured results, for a repository with no checkpoint that has
never run a training step. Removed rather than restated as targets: a target
written in the shape of a result is how the confusion started. Eight "Working"
status cells corrected to "Constructs; untrained". README now points at
CHECKLIST.md, which cannot claim anything without a passing test.

## D4 — Dialogue and action are one *system*, two *paths*

The owner asked the right question: *"surely dialogue must be incorporated with
action? isn't it one brain?"*

Functionally yes. Architecturally no — and the reason is empirical, not
aesthetic. Naive behavioural cloning through a VLM destroys **94% of GQA
accuracy within 10k steps**. Making it literally one network by training end to
end is what *removes* the language ability you wanted. π0.5 uses "knowledge
insulation" (arXiv:2505.23705): the action expert attends to the language model's
layers at inference while gradients are **stopped** from reaching pretrained
weights. GR00T N1/N1.5 freeze their VLM in both pretraining and finetuning, for
exactly this reason.

So the integration lives in the **adapter**, which is trained on language and
action together and is the only thing that learns:

```
  speech ─► dialogue LLM (frozen, swappable, API or local)
                    │
  command ─► text tower (frozen, small, local, free) ──┐
                                                        ├─► ADAPTER (~22M, TRAINED)
  vision ──► SigLIP2 (frozen) ────────────────────────┤        │
  proprioception ─────────────────────────────────────┘        │
                                                    32-d motion latent + subtask
                                                               │
                                                          policy ─► joints
```

Two consequences that answer the rest of the question:

**Grounding is decoupled from chat.** A separate small frozen text tower does
language→action; the dialogue model never touches it. That is what makes the
chat model swappable — replace it with anything, later, and grounding is
unaffected. It is also the correct fix for the LLM-swap fragility that
`SemanticActionAnchors` was invented to work around.

**Frozen means cacheable.** Precompute the text embeddings once (~45k captions ≈
170 MB) and every subsequent adapter experiment trains on this box's CPU in
minutes with no GPU resident. That single property is what makes the plan fit a
16 GB T4 and a burst schedule — and it is destroyed the moment the LLM becomes
the trunk.

**Decision on the chat model: API primary, local fallback behind a flag.** Cents
per session, and the only tier that produces real conversation; SmolLM2 on a
CPU-only ARM box shared with paying tenants is slow and shallow. Grounding stays
local and free regardless, so this costs nothing architecturally and can be
reversed at any time.
