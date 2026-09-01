# GENERALITY.md — every known reason Jack is not AGI, as tasks

> Owner, 2026-08-09: *"every single reason Jack can't be AGI, you must make a
> task — we must solve."* This file is that list. Each barrier gets a
> falsifiable test, so it stops being a feeling and becomes a result.
>
> **Read the honesty clause first.** Most of these are years out and several
> may be unsolvable, unnecessary, or wrong. Naming them is not promising them.
> The value is that the ladder stops being SILENT about the distance between a
> competent creature and a general mind — silence is how a project mistakes
> the edge of its map for the edge of the world. Nothing here is scheduled;
> nothing here competes with the current frontier. It is the map.
>
> Ranked by how much each one blocks the others. The first three are the ones
> I would bet the distance to generality actually lives in.

## GEN.00 — THE FINAL EXAM: he learns something nobody taught
**The direct test of the first principle**, and the one thing 136 specs did not
check. Every other spec asks "did he learn X?" — where X is something we
designed, specced and rewarded. That can all pass while Jack is a COLLECTION
OF TRAINED BEHAVIOURS rather than a learner. The claim is bigger than the
scorecard: not "he learned these things" but "he acquires things".
**Barrier.** Seven existing specs use held-out DATA (unseen phrasings, unseen
states). None uses a held-out CAPABILITY — a challenge the trainer never
anticipated.
**Test.** A SEALED CHALLENGE, designed by someone blind to the training:
  - The challenge is a world feature that is CONSEQUENTIAL to his existing
    needs (so he has a reason to engage) and NOVEL in mechanism (so no trained
    behaviour transfers directly). Example shape, not the actual test: a plant
    that is nourishing only after being heated, where nothing in training ever
    coupled fire to food.
  - It is committed as a HASH before training and revealed only at test time.
    Nobody who wrote a spec, a curriculum, a reward or a research doc may see
    it first. That sealing is what makes it a real exam rather than a
    rehearsal.
  - PASS: competence improves measurably within a life or across lives, with
    NO change to training, reward, or architecture.
**Falsified by.** No improvement. He only ever gets better at things someone
designed for — a trained system, not a learner.
**Null.** The same Jack with learning FROZEN at test time (weights fixed, diary
read-only). It must NOT improve — that separates learning from competence he
already had.
**Control that must fail.** A SHAM-NOVELTY challenge: superficially new,
functionally identical to something trained. Improvement there proves nothing;
he must improve on the genuinely novel one SPECIFICALLY. Without this, "he
adapted" could just be "he recognised".
**Status.** Runnable EARLY and cheaply — in the playground, long before the
jungle, as soon as needs and a learning core exist. It should be run
repeatedly, with a fresh sealed challenge each time, forever: the exam only
means anything while the questions are ones nobody studied for.
**Kills.** The first principle's central claim, if it fails. "Give him a brain,
a body and a world and let him become" is exactly the assertion that he can
acquire what we did not install.

## GEN.01 — The world is the ceiling
**Barrier.** You cannot become more general than your environment demands. A
jungle with fire, water and fruit can produce an excellent forager and can
never produce a mathematician, because nothing in it ever asks. Every other
item on this list is downstream of this one.
**Test.** Capability must track world complexity across the fidelity tiers
(W0→W3). Measure competence-per-tier. **If it plateaus while the world keeps
getting richer, the ceiling is HIM. If it keeps climbing, the ceiling is the
WORLD — and generality is a world-building problem, not a brain problem.**
That is the single most informative measurement in this document.
**Status.** Fidelity ladder designed (SURVIVAL_WORLD.md); measurement not.

## GEN.02 — He is alone
**Barrier.** Human intelligence is overwhelmingly social. The hardest thing in
our ancestors' world was other people: imitation, teaching, competition,
deception, coalition. One Jack plus an occasional human visitor is missing
most of the pressure that made us smart. **Measured 2026-08-09: of 136 specs,
exactly ONE touches other minds, and it is mocap imitation.**
**Test.** Two Jacks, one world. Does one acquire a skill by WATCHING the
other, never having been taught? Control: watching a random-action agent must
not help. **Cheapest high-value item on this list — a second process, not a
second GPU.**
**Status.** Registered 2026-09-01 (`GEN.02`, depends on `VO.02`+`LC.07`;
Review 08-31 item 6).

## GEN.03 — He has no model of other minds
**Barrier.** Theory of mind: knowing that another agent knows something you
do not, or believes something false. Prerequisite for teaching, deception,
cooperation, and most of language's real work.
**Test.** Embodied false-belief. Jack watches another agent see food hidden at
A; the food is moved to B while that agent is absent. Does Jack predict the
agent searches at A? Control: if the other agent WAS present for the move,
the prediction must flip. **Passing this is a landmark; failing it is normal.**
**Status.** Registered 2026-09-01 (`GEN.03`, depends on `GEN.02`).

## GEN.04 — Abstraction and composition are unproven
**Barrier.** Concepts about concepts; hierarchies deep enough for reasoning to
bootstrap. This is where RL genuinely struggles today, and **nobody knows** if
this architecture supports it. The honest unknown.
**Test.** Zero-shot composition: a task solvable only by combining two learned
skills in an order never trained. Control: an agent that has each skill but
never used them jointly must fail. (ME.6's composite-skill spec is the seed.)

## GEN.05 — He cannot make tools
**Barrier.** Cumulative technology is arguably the human difference. Using an
object for a purpose it was never given is the first rung; making one is the
second.
**Test.** Present a need solvable only by repurposing an object (reach the
apple using the plank as a ramp — never demonstrated). Then: assembling two
objects into a thing neither is. Control: an agent given the assembled tool
must succeed, proving the task is solvable and the gap is invention.

## GEN.06 — Transfer across worlds is untested
**Barrier.** A Jack who masters the jungle and is helpless in a desert is not
general — he is fitted. Generality IS transfer.
**Test.** Train in world A, drop into world B: different layout, different
resources, SAME underlying rules. Prior experience must beat a fresh agent.
Control: a world with SHUFFLED rules must NOT transfer — otherwise the gain is
general fitness, not learned structure.
**Status.** Registered 2026-09-01 (`GEN.06`, depends on `LC.07`+`W0.DIAG` —
the validated difficulty instrument certifies A and B comparable first).

## GEN.07 — He does not know what he does not know
**Barrier.** Metacognition. Calibrated uncertainty is what makes an agent seek
information, ask questions, and hedge instead of blunder.
**Test.** Does he explore MORE when his world-model uncertainty is high and
less when it is low — measured against his own model's error, not a proxy?
Does he ASK when asking is cheaper than trying? Control: an agent with
uncertainty shuffled must show no such coupling. (Partial seed: EpisodicMemory
already abstains honestly — knowing what he does not REMEMBER. This extends it
to knowing what he does not UNDERSTAND.)

## GEN.08 — His goals come from his body, not his mind
**Barrier.** Needs and curiosity generate goals. Humans also invent goals no
drive demanded — build a thing to see if it can be built. Open-endedness in
the strong sense.
**Test.** Count self-set objectives that are not need-reducing and not
novelty-seeking, that persist across sessions and are pursued to completion.
Control: the needs-only agent must produce approximately none.

## GEN.09 — One diary is not a culture
**Barrier.** His diary crosses HIS deaths — Lamarckian, already a cheat on
biology. But human intelligence is CUMULATIVE across individuals: each
generation starts where the last stopped.
**Test.** Three generations. Generation 3 must know something generation 1
never knew, transmitted ONLY through diaries and teaching — never through
weights. Control: with transmission severed, generation 3 must fall back to
generation 1's level.
**Status.** Registered 2026-09-01 (`GEN.09`, depends on `ME.9`+`ME.10`+
`GEN.02`; the severed-transmission branch is the null, a shuffled-content
diary is the control).

## GEN.10 — Long-horizon credit assignment
**Barrier.** A human plants a tree for shade in twenty years. Learning that an
act pays off thousands of steps later is unsolved in RL generally.
**Test.** A task whose only reward arrives after a delay far exceeding the
discount horizon. Does he learn it at all? Control: the same task with dense
intermediate reward must be learnable, isolating the horizon as the barrier.

## GEN.11 — Nothing in his world requires symbols
**Barrier.** Counting, measuring, arithmetic, abstract representation. A
creature never asked to count will not learn to. Downstream of GEN.01.
**Test.** Introduce a world rule that only a counting agent can exploit (a
trap safe on exactly the third crossing). Does the capability emerge, or must
it be installed? Either answer is informative.

## GEN.13 — Told knowledge must anchor to lived knowledge
**Barrier.** Jack must know more than a jungle, and almost everything he will
ever know must arrive second-hand — as it does for every human. But borrowed
facts are only knowledge if they CONNECT to something lived. Otherwise the
library sits inert beside the creature and he is a language model wearing a
body after all.
**Test.** Teach two facts of matched complexity: one anchored to his lived
primitives (heat, weight, effort, danger), one entirely foreign (abstract
finance). Measure integration — can he use it, reason with it, apply it to a
novel case? **Grounded facts must integrate measurably better.** Control: if
both integrate identically he is reciting, and the jungle bought nothing.
**Status.** Named 2026-08-09; the falsifier is stated in GOAL.md's expansion
path. Depends on the LG grounding family.

## GEN.12 — Experience efficiency versus a child
**Barrier.** A human gets ~10^9 seconds plus 500My of evolutionary prior.
Jack gets simulated hours on a free box. Even perfect architecture may need
more life than we can afford to give.
**Test.** Standing measurement: lifetime-experience-to-competence, tracked per
capability, compared against published human-developmental milestones where
they exist. It sets the honest expectation for everything above.

---

## What this list is FOR

Three uses, and none of them is a promise:

1. **Anti-drift.** When Jack forages beautifully and we are tempted to call it
   general, this file says what general would actually require.
2. **Cheap-first ordering.** GEN.02 (a second Jack) costs a process. GEN.01
   is measurable the moment the fidelity ladder exists. Most of the rest are
   years away — and knowing WHICH are cheap is the point.
3. **Honest failure.** If several of these prove unreachable, that is a real
   scientific result about this design, and it belongs in the ledger like any
   other. A creature that is not AGI is still the first of its kind.

**The pattern worth noticing:** GEN.01, 02, 05, 09 and 11 are all about the
WORLD and WHO IS IN IT — not the brain. If the distance to generality is
mostly environmental, then the first principle already had it right: give him
a brain, a body, and a world, and keep making the world bigger.
