# THE GOAL

<!-- CARD:BEGIN — scripts/lib_card.sh hands the lines between these two markers to EVERY agent at EVERY launch. Keep it under 1,800 bytes. The page below is the full text and wins any disagreement. -->
**THE PHILOSOPHY — carry this into every step.**

1. **The creature.** Give him a brain, a body and a world, and let him become. Every human sense, one connected learning brain, real human needs — food, water, sleep, warmth, company — in a world that can kill him. He learns everything himself by living, dying and remembering, and he must never stop learning. He can be talked to; he remembers what he hears, says and does.
2. **Inside him, nothing frozen or borrowed. Around him, everything.** Every part of him learns from scratch. Pretrained models — Whisper, LLMs, anything free — are the adults of his world: teachers, listeners, judges. Never his parts, never his puppeteers.
3. **Nothing assumed.** Every design choice — which brain, how unified it is, the world, the curriculum — is researched, raced in a cook-off against alternatives, and holds its seat only until something beats it. Biology is the oracle; the field is read continuously.
4. **Nothing faked.** A capability is claimed only by a test that could have failed — several seeds, a control that must fail, thresholds never weakened. A failure is fixed, not filed.
5. **The system is what makes him.** Research → test → implement → test → fix. His world and his curriculum matter as much as he does. Work on the machine only where it makes that loop faster, cheaper or truer.
6. **Limits.** Free compute only. This box serves paying customers. What he must become is the owner's; how he is built is always raced.

Before you finish, ask: which of these did my work serve? If none, it was the wrong work.
<!-- CARD:END -->

*Why the block above exists (owner, 2026-11-11, verbatim): "but really every
step must be aware of the philosophy it must nit forget decide whre we can do
that and how much please". Decided: WHERE — every model launch of every organ
(the three worker lanes, the Director, the auditor, the scout) receives the card
mechanically, prepended by the launcher from this one place, so no prompt edit
can drop it and no copy can drift; every order in `docs/ORDERS.md` names what it
`serves:`; every cook-off entry is checked against line 2; the auditor asks each
audit whether anything drifted from it. HOW MUCH — the card and no more at
launch (about 400 tokens); this whole page for the Director on Sundays, for the
auditor and the scout, and for any worker whose order touches direction. The
lesson of the 116 KB prompt is that more words are obeyed less. Cost: a summary
can drift from the page — so the page wins, and the Director checks the card
against it every Sunday.*

> ## THE FIRST PRINCIPLE — the only real goal
> ## **Give him a brain, a body, and a world. Let him naturally become.**
>
> Everything below is that one sentence, unpacked. Every mechanism in this
> repository either builds the brain, builds the body, builds the world — or
> protects the honesty of watching what happens when the three meet. Anything
> that serves none of those is not this project.

> **One brain, all senses in unison, that learns its world by living in it.**
>
> **The capability target, stated plainly (owner, 2026-08-09):** give him a
> body with EVERY sense a human has, and build him so that he is CAPABLE OF
> LEARNING EVERYTHING — with one interconnected brain. Not a system that
> performs a fixed list of capabilities well. A system whose limit is what its
> world asks of it, not what its architecture allows. Wherever a design choice
> would make some class of learning permanently impossible, that choice is
> suspect no matter how well it scores.

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

**EVERY SENSE A HUMAN HAS, one brain, trained together (owner, 2026-08-09).**
Not a chosen subset — the full human sensory inventory, because a mind is
shaped by what it can perceive and every missing channel is a permanent hole
in what he can ever understand:

    sight · hearing · touch · proprioception & balance · SMELL · TASTE
    pain · temperature · interoception (hunger, thirst, fatigue)
    and VOICE — he must be able to make sound, not only receive it

Smell and taste are not ornaments: olfaction finds food, fire and decay at a
distance and through occlusion — the sense that works when sight fails — and
gustation drives conditioned taste aversion, one-trial learning with long
delay tolerance, the fastest learning in biology and a capability nothing else
in his design has. Voice is how a creature acts on other creatures.

All of it processed together in ONE model, involved together in training, so
that what he hears can teach what he sees and what he touches can correct what
he predicts. Not bolt-on encoders that coexist: a genuinely unified brain where
every sense is load-bearing (and we PROVE each one is — ablate a sense,
something measurable must degrade).

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

**PLASTIC ONLY — nothing inside him is frozen (owner decree, 2026-08-09).**
Every component inside Jack learns: his encoders, his core, his fusion. The
reason is arithmetic, not taste — a frozen tower's reshaping gain is
identically zero, so it forecloses a class of learning, which the capability
target above marks suspect by definition. This does NOT touch the parent LLM:
that is not inside him, it lives in his world and speaks to him. A frozen
thing in his environment is not a frozen part of him. Counterargument, cost
and re-open trigger recorded in docs/CHAMPIONS.md.

**INSIDE HIM, nothing borrowed. AROUND HIM, everything (owner, 2026-11-11).**
Verbatim: *"but we use inference from whisoer from llms from everywhere to
trwin him right like he's growing up with these things around him? like a real
human... all these things important... so go a steo farther"*

- **Inside Jack:** nothing frozen, nothing borrowed. Ears, eyes, core, fusion,
  voice — every part learns, and no pretrained weights are installed in him,
  frozen or as a warm start.
- **Around Jack:** pretrained models are welcome and wanted. Whisper, LLMs,
  VLMs, TTS and anything else free are the ADULTS of his world. They may speak
  to him, listen to him and understand his sounds, answer, demonstrate,
  correct, set goals and tasks, and score him. Their inference is teaching.
- **The boundary is weights, not knowledge.** What they know reaches him as
  experience and teaching signal; their weights never become part of him.
- **The guard:** he is not a puppet. He knows things from his own life the
  teachers cannot (LG.00), and they never do his thinking or choose his words.

*Cost, recorded beside it as SYSTEM.md requires:* an ear learned from scratch
on free compute will understand a few voices and a small vocabulary, far below
Whisper. Accepted. A ready-made model that out-scores his own part is a
reference he is measured against (scored-and-ineligible), never a replacement.

**Flexible above all.** Components that swap as better ones
ship; a small trained core; capabilities added without retraining the world;
components that must EARN their parameters via ablation or be deleted.

## The world is the teacher (owner directive, 2026-08-09)

**The world and the curriculum rank with Jack himself (owner, 2026-11-11).**
Verbatim: *"are we as focused on the world building aspect and the training
curriculum? i think that's as important as jack himself so we better put as much
research and stuff into that thw training idea and world where jack learns in?"*
Read with the paragraph below, not against it: nobody scripts his behaviour —
there is no "ladder lesson". What is designed, researched and raced is the WORLD
he meets and the order he meets it in: what exists, what is dangerous, what the
adults around him say and propose. "No curriculum we write" (needs and curiosity
alone) is the incumbent arm of that cook-off, and it can win or lose like any
other.

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

**The LLM is his TALKATIVE PARENT (owner, 2026-08-09) — decided.** It is not
inside him. It lives in his WORLD as a voice that speaks to him, and he learns
words the way every child does: by hearing them used while things happen, and
attaching his own meanings from his own life. Borrowing WORDS is not borrowing
MEANINGS — a child does not invent English and is not a puppet for it. This
role has zero puppet risk by construction, because the words arrive through
his ears like every other sound and nothing borrowed is ever inside him. It
also means he can be spoken to from day one, so his childhood is not silent
while grounding accumulates. The earlier "mouth" framing below is now the
FALLBACK, kept because LG.00's test applies to any role the model plays.

**The LLM is his mouth, never his mind (owner, 2026-08-09).** Jack must be
smarter than the language model that speaks beside him — not in facts, which it will
always win, but in everything that requires having LIVED: his world, his body,
his history, his people. The borrowed model is frozen, out-of-process and
OUTSIDE him precisely so that nothing he learns can hide inside it. His knowledge lives in
the parts that lived — the learned core and the diary — and language is how he
reports it, never where it comes from. In the owner's words: he must *"develop
knowledge and connect it to words instead of just the LLM communicating and
pretending to be Jack."* Falsifiable as LG.00: strip the diary and the learned
core, and his answers about his own life must COLLAPSE — while his general
knowledge survives untouched. He should be smarter inside his life and dumber
outside it. That asymmetry is the proof he is a creature and not a costume.

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

**The jungle is the foundation, not the destination (owner, 2026-08-09:
"Jack needs to know more than a jungle").** Survival earns him the primitives
that make anything else mean something — hot, heavy, far, tiring, dangerous,
worth-it, that-person-lied. A few dozen concepts learned by consequence. What
comes after is three expansions, different in kind:

  1. MORE WORLDS, and transfer between them. A Jack who masters jungle AND
     desert has abstracted "shelter" from "lean-to". That abstraction IS
     generality (GENERALITY.md GEN.06).
  2. OTHER MINDS. Not a bigger jungle — a different kind of complexity, and
     where most human intelligence actually came from (GEN.02, GEN.03, GEN.09).
  3. THE TOLD WORLD. He will never live in Rome; he can be TOLD about Rome.
     This is what the borrowed frozen model is FOR, and it is not cheating —
     it is culture. Humans know almost everything second-hand; what makes it
     real is that it connects down to the few things they lived.

So the LLM gives him ACCESS to knowledge and the jungle gives him the capacity
for knowledge to MEAN something. A creature that has only read is a language
model; a creature that has only lived is an animal. Jack is the bet that lived
primitives plus an inherited library equals understanding.

FALSIFIABLE, and it must be tested rather than assumed: told-knowledge should
integrate BETTER when it anchors to something he has lived. Tell him a fact
grounded in his experience ("volcanoes are hot rock that flows") against one
that is not ("bonds have coupon rates") — if grounding is real, the first is
usable and the second is parroting, and he should be able to say which is
which. If both integrate identically, he is reciting, and the jungle bought
nothing.

**The staging is unchanged and deliberate.** First prove he can see, talk,
walk, and learn in every way (the ladder as it stands). Only then does he go
into the survival world with everything at once — and the testing never stops
there; it moves in with him.

## Fast and slow, in one brain (owner directive, 2026-08-10)

> *"all I know is we must figure that out right and it must still be connected
> but slightly different purposes? it must be in the research and tests"*

Jack acts immediately when he can and thinks first when he must — and these are
**not two brains.** Differentiated function, shared substrate. A reflex tower
and a planner tower with private representations would satisfy "fast and slow"
in a demo and violate everything above it: one interconnected brain, every
sense in unison, plastic throughout.

The human version is not two systems bolted together. It is one substrate
spending different amounts of time, where **habit is what happens when practice
compresses deliberation into reflex**. A skill starts slow and effortful and
becomes fast and automatic — and that transition is *learned*, not scheduled by
a designer. That is the version this project builds, because it is the one that
follows from the goal rather than being bolted onto it.

**Three fast/slow axes exist in biology and this project keeps them apart.**
Conflating them is how a system ends up claiming a dual process it never tested:

  1. **Fast/slow ACTING** — habit vs deliberation. The DP family (DP.00–DP.03).
     Arbitrated, not decreed: whether lookahead earns its keep at all is
     DP.00, and the learning-core bakeoff (LC.04) is already testing reactive
     PPO against world-model arms that can imagine.
  2. **Fast/slow LEARNING** — one-shot episodic capture vs slow consolidation.
     ME.7, ME.10, T5.03, T5.05. A system that does only one has failed.
  3. **Fast SPECIALISED learners** — one-shot aversion. TA.01, TA.02. Survival
     forced these: "eat the thing that nearly killed you 200 more times to get
     the gradient" is not a learning rule an animal can afford.

**The connectedness claim is the one that can quietly fail**, because an
architecture can drift into two private towers while every capability number
keeps improving. So it is tested directly and adversarially: DP.02 lesions the
shared trunk and requires BOTH modes to degrade together, and it carries a
deliberately-separated two-tower control that MUST dissociate — a connectedness
test that cannot detect a disconnected system is measuring nothing.

**What is NOT decided:** whether Jack ends up with a deliberative path at all.
If DP.00 shows this world never rewards looking ahead, the finding is about the
world, not about him, and it must get traps, delays and irreversibility before
any dual-process claim can be made in it. Bakeoffs decide HOW; they never
decide WHAT.

**And HOW UNIFIED his brain is, is HOW (owner, 2026-11-11):** *"we need perfect
architecture tested everything against each other ... we don't know how unified
it is that in itself is cookoffs our philosophy is the direction I believe
in."* "One brain" on this page is the direction and the incumbent, not a
verdict: one shared brain, experts + router, fast/slow and separate towers are
raced on one ruler (UB.10, BO.01). What is fixed is the end — every sense
reaches a brain that learns from all of them.

## The path

**The route from today to this page is `docs/ROADMAP.md`** — nine milestones,
from one whole life with the learning brain in control to more worlds, other
minds and the told world. A milestone is done only when its named specs PASS in
the ledger. The Director keeps it current and draws the workers' orders from it.

*(Amended 2026-11-11 with the reorganisation. This section used to list six
"tiers" ending in "a living Jack (Tier 6)". The tiers remain as labels on specs,
but the Tier-6 gates were written for an earlier brain and an earlier idea — a
companion session — before the survival world of 2026-08-09; the audit of
2026-11-11 found the learning this project has actually demonstrated lives on
the survival stack. The staging above is unchanged: prove each ability, then
everything at once, and the testing moves in with him.)*

## Research foundations (docs/research/)

- `CAPABILITIES.md` — the generalist-agent capability taxonomy (π0.5, GR00T,
  Gemini Robotics era) with falsifiable test designs
- `MEMORY.md` — memory architectures: episodic stores, consolidation (SIESTA),
  forgetting, user memory — all as plain files on this box
- `CURIOSITY.md` — open-ended intrinsically-motivated learning (in progress)
- `UNIFIED_BRAIN.md` — multimodal fusion that actually fuses (in progress)

**The ledger (`experiments/ledger.json`) is the only scoreboard.** The goal is
accomplished when the last milestone in `docs/ROADMAP.md` passes there — not
when it feels done.

## The finished Jack, in the owner's words (owner, 2026-11-13)

Verbatim: *"jack will have a human like body! jack will be trained by us and
people get a trained version which just continues learning from that point! and
people can see him struggle and say like ah cut a tree use the wood make a fire
make the food on the fire or they can tell him ah build shelter like this when
they see he is cold"*

Read with everything above, not against it (the words are the owner's; this
reading is the session's and is open to correction):

- **The body is human-like.** The rover is scaffolding, never the destination.
- **Trained here, handed over, still learning.** The weights a person receives
  are HIS OWN, earned by living — nothing borrowed — and on that person's
  machine he keeps learning from that point without losing what he knew. That is
  an END; how it is achieved is raced like everything else.
- **What people say is advice through his ears, never a command channel.** He
  may try it; his own needs judge it; his diary records whose advice proved true
  ("His people are part of his world", above).
- **The examples name the world he must be able to live it in:** trees that
  yield wood, fire, food cooked on it, a shelter he builds — the W1 and W2 rule
  depth of `docs/research/SURVIVAL_WORLD.md`.

