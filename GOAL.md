# THE GOAL

> ## THE FIRST PRINCIPLE — the only real goal
> ## **Give him a brain, a body, and a world. Let him naturally become.**
>
> Everything below is that one sentence, unpacked. Every mechanism in this
> repository either builds the brain, builds the body, builds the world — or
> protects the honesty of watching what happens when the three meet. Anything
> that serves none of those is not this project.

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

## The world is the teacher (owner directive, 2026-08-09)

**Jack has the needs of a human.** He must eat, drink, sleep, stay warm — too
cold kills him, too hot kills him — and he needs company. Not as decoration:
the needs ARE the curriculum. Cold nights teach shelter-building the way no
scripted lesson can. Needs do not replace curiosity — they complete it:
curiosity is the explorer, needs are the reason. The "purely out of curiosity"
sentence above predates this section; read the two drives as partners, and
their exact balance as an empirical question the curiosity/needs bakeoffs
decide, not a doctrine either sentence settles. In the owner's words: *"having the needs of a human will
have him learn the most efficient ways... and will allow users to talk to him
while he is there doing stuff and he will be relatable — I want to watch him
figure out the world himself."*

**He lives, he dies, he remembers.** The destination world is as realistic a
survival simulation as buildable — the jungle. And "realistic" means what it
meant to a caveman, not to a chemist (owner, 2026-08-09: *"we don't actually
need to understand chemistry for this — just like cavemen didn't"*). The world
must be **consistent** (same act, same conditions, same result — so rules are
learnable), **discoverable** (rules surface through poking at things — so
curiosity grips), and **consequential** (outcomes couple to his needs — so
learning matters). Fire is a state machine, not combustion. The world-fidelity
gates verify that the world obeys its own pre-registered rules; consistency is
the falsifiable property, realism never was. He gets thrown in, figures life
out or doesn't, dies, and tries again — and what survives death is the point:
the diary and the skills persist across lives (the ME family already proves
the substrate). Life N+1 must be measurably better than life N *because of*
what life N recorded. Death is not a reset; it is a page turn.

**The system is the product, not the model.** In the owner's words: *"at the
end of the day it won't be the most complex model that Jack is. It will be
just a system that can learn and get input from every single sense."* We build
tests, throw him in, get results, build bigger tests, throw him in again — and
the final test in that sequence is the real world. The 57M-vs-54K lesson is
already on the ledger: complexity must earn its place or lose it.

**Biology is the reference implementation (owner, 2026-08-09).** Human and
nature biology is the best model we have — the only working example of general
intelligence built by living. When stuck, ask how nature solved it: needs are
interoception, the diary-vs-weights split is hippocampus-vs-cortex, sleep
consolidation is replay, curiosity-as-learning-progress came from infant
studies, dreaming is training in imagination. But biology is the ORACLE, not
the blueprint — planes do not flap. Nature's solution enters as a bakeoff arm
and must win on our substrate like any other; where it loses, we take the win
and record the divergence. And in one place Jack deliberately surpasses
biology: genes cannot inherit experience, but culture can — the diary crossing
death is Lamarckian inheritance, the caveman's fireside story made structural.
Still unmined and on the shelf: motor babbling, innate reflex priors, pain as
a fast signal distinct from reward, critical periods, play as safe rehearsal.

**His people are part of his world (owner, 2026-08-09).** Their presence is
company — being near him is care. Their words are teaching — one sentence can
spare him a thousand falls, and his diary records whose advice proved true,
so trust in a person can be earned and checked. And their hands may leave
things in his world for him to find — food where he might look, a tool he
has not made yet. Never puppeteering: what is left must still be found,
learned, and chosen by him. His diary records who left it — so gratitude,
like trust, has somewhere real to grow.

**Who he becomes is not written here (owner, 2026-08-09).** We carve what he
IS — the needs, the senses, the honesty, the mortality — and we deliberately
refuse to carve his character. His kindness is not decreed; it is expected to
GROW from his need for company, the way it grew in us. His understanding that
his memories of people are OF those people is not a rule; it comes from the
structure of his memory itself — every event attributed, every voice named.
In the owner's words: *"I would believe his need for socialising will make him
kind... I want to let as much of this naturally develop."* The deal that keeps
this honest: what emerges is OBSERVED, measured, and reported truthfully —
never scripted, and never silently patched. If who he becomes ever troubles
us, that is a conversation for the owner, not a hotfix. The conditions are
ours; the person is his.

**The staging is unchanged and deliberate.** First prove he can see, talk,
walk, and learn in every way (the ladder as it stands). Only then does he go
into the survival world with everything at once — and the testing never stops
there; it moves in with him.

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
