# THE GOAL

> **One brain, all senses in unison, that learns its world by living in it.**

Jack is a virtual humanoid who **learns his environment the way a child does**:
not from a curriculum we write, but from curiosity we give him. This is the
project's north star. Every architecture decision, every spec in the ladder,
every training run is in service of it. If a piece of work does not trace back
to this page, question the work.

## What this means, concretely

**He explores because he wants to.** In his free time — no task, no instruction
— Jack tries to figure out his environment. If there is a ladder with an apple
on top, he must try to climb the ladder, fall, and learn from falling, purely
out of curiosity. If there is water, he must try to swim, struggle, and learn to
swim by struggling. Nobody scripts "ladder lesson" or "swim lesson". The
environment plus intrinsic motivation IS the curriculum.

**All senses, one brain, trained together.** Seeing, hearing, touch,
proprioception, language — processed together in one model, involved together
in training, so that what he hears can teach what he sees, and what he touches
can correct what he predicts. Not bolt-on encoders that coexist: a genuinely
unified brain where every sense is load-bearing (and we PROVE each one is —
ablate a sense, something measurable must degrade).

**Really learning, not appearing to learn.** Learning that survives our
falsification ladder: every capability claimed only by an experiment that could
have failed, against a null baseline, with controls that must fail, at ≥3 seeds
where the claim is about learning. A loss curve is not learning. A README
saying "Working" is not learning. Climbing the ladder on attempt 40 after
falling on attempts 1–39, without anyone telling him to — that is learning.

**Memory makes it him.** What he learned yesterday — about the world and about
his owner — persists on disk, inspectable, across restarts. He remembers the
ladder. He remembers you.

And it is two memories, not one. When people interact with him he remembers
what he **heard**, what he **said**, and what he **did** — attributed, per
person ("what did I tell you" is not "what did you tell me") — while the same
lived experience ALSO distils into general skill. Keeping the record and
learning from it are separate stores with separate failure modes, and both are
ledger-tested: ME.9 (attributed recall of heard/said/did) and ME.10 (wipe the
diary, the skill survives; revert the weights, the diary survives).

**Flexible above all.** Frozen pretrained trunks that swap as better models
ship; a small trained core; capabilities added without retraining the world;
components that must EARN their parameters via ablation or be deleted.

## The path (each stage gated by the ladder — see CHECKLIST.md)

1. **Harness + primitives** (Tiers 0–1, DONE): measurement works, every part can learn.
2. **Capabilities vs null** (Tier 2, in progress): locomotion, grounding, memory,
   curiosity — each must beat a dumb baseline or be replaced by it.
3. **Earn your parameters** (Tier 3): every component ablated; dead weight deleted.
4. **Unison** (Tier 4): senses fused; each proven load-bearing; no modality collapse.
5. **The claims** (Tier 5): continual learning without forgetting, plasticity
   that does not die, curiosity that drives real exploration — the thesis itself.
6. **A living Jack** (Tier 6): runs for hours, remembers across sessions,
   explores unprompted in his playground world.

## Research foundations (docs/research/)

- `CAPABILITIES.md` — the generalist-agent capability taxonomy (π0.5, GR00T,
  Gemini Robotics era) with falsifiable test designs
- `MEMORY.md` — memory architectures: episodic stores, consolidation (SIESTA),
  forgetting, user memory — all as plain files on this box
- `CURIOSITY.md` — open-ended intrinsically-motivated learning (in progress)
- `UNIFIED_BRAIN.md` — multimodal fusion that actually fuses (in progress)

**The ledger (`experiments/ledger.json`) is the only scoreboard.** The goal is
accomplished when Tier 6 passes — not when it feels done.
