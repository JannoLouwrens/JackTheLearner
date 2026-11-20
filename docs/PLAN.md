# THE PLAN — research first, build from what is known, test only what is new

> The owner, 2026-11-15 (typos cleaned): "We must research before we try our own stuff. Yes, we have problems to solve that have never been solved, but research will show us what we can combine to build our own, and then we test for the last 10%. This is a fact, it must be like this."

`GOAL.md` says what Jack must become. `docs/ROADMAP.md` lists the nine milestones on the way. This page says HOW the work is done. It binds every agent: the Director's orders, the workers, the Inventor, the Researcher and the Auditor.

## Why the plan changed

Until today the rule was "decide by racing". The team wrote its own learners and raced them against each other. The published record shows that could not work. The food-finding race gave each brain about 40,000 decisions (one decision is one action, 0.2 seconds of his life). Published agents on comparable tasks need 100,000 to several million. The learners were not what their names said, and none had ever reproduced a published result. The race measured our mistakes, not brains.

Most of what is needed to build Jack has already been tested by other groups, often at a scale we could never afford. So we read first, and we build from what they proved.

## The method: five steps, for every part of Jack

1. **Research.** What has the world already tested? How does it work, and what should we expect, in numbers? The answer is written into `docs/research/PUBLISHED_PRIORS.md` before anything is built.
2. **Combine.** Build our version from published, proven parts: designs and code, never ready-made weights. Every place where ours differs from the published recipe is written down with its reason.
3. **Reproduce.** Before our version is trusted, it must reproduce a published number on a small standard task. If it cannot, it is broken, and it is fixed before anything else is done with it.
4. **Test the last 10%.** Our own experiments are only for what the record does not answer. Each one is written down first: what the record predicts, the pass mark, and enough repeated runs for the result to be believed.
5. **Put it in him.** What passes goes into the living Jack and is checked again there.

No part skips a step. A part's status is the step it is on. Racing still decides between candidates, but only at step 4, only between candidates the record supports, and only at a size the record says can show something.

## Where each part stands

The Director keeps this table current at every sitting. Step: 0 not started, 1 research, 2 combine, 3 reproduce, 4 test the last 10%, 5 in him.

| Part of Jack | Step | Known: take it from the record | The last 10%: ours to find out |
|---|---|---|---|
| His brain | 3 reproduce (brain B); brain A still 2. 11-20: A's three known-answer runs and its first smoke in his world launch on the free GPU from 11-21 00:00 (J14). The home-made brains' repaired entry SPLIT: it reproduced the sparse standard row (4 of 5, the first in-house row ever reproduced) and failed the dense one (0 of 5); the cause was measured the next day (exploration noise held 32 decisions kills the dense row; the discount change does not). Ruled (59): one last entry with one setting for both rows, then J3 closes either way. The Mac port passed rung 1 (9 of 9 maths checks, eager and compiled); the wheel check and rung 2 are ordered (J16) | A Dreamer-style world model: one brain that learns by predicting every sense, proven from scratch on 150+ tasks; the official code defines it and one maintained port can run it. A simple replay learner (SAC type, from a public library) as the control. Exact scores each build must reproduce. | Does it hold at our small size? No published score exists below about 10 million parameters. How fast it runs on our hardware. Which one wins in his world. |
| Learning to eat | 4 test the last 10% (pilot 2 READ; E2c running). 11-20: pilot 2 is home, 40 of 40 runs, read by its frozen rule: no learner ate. Every learning arm on every seed ends at 0.08–0.16 meals a life, below the same brain never learning (0.82) and below random held pushes (0.95), in the movement-free world too. The record's prediction (the free-movement arm eats inside 300,000 decisions) failed. Priced suspect: a fixed exploration bonus worth 15–18 times the task reward (reference section 19; no published forager used it). The learner handed the direction to food (E2c) is running, and by ruling 57 it gets twins with the published learned setting before its last seeds, so a miss can be told from the setting | Keep everything he lived (a large replay). Look 40 seconds ahead, not 4. Move in sustained runs, not jitter. Pay him when he eats. Food near at first. 100,000 to 1,000,000 decisions. | Can 16 distance rays tell food from a wall? |
| His world, needs and lessons | 2 combine. 11-19: his push and his felt velocity are now in his body's frame, like his eye (ruling 49: the human-like body has no compass; with a compass motor under a body eye the hand forager ate on 0 of 20 worlds). "Learnable from his senses" is shown at the entry rung only (ruling 51); the full rung is the dial's work in stage 3. The heat tests stay parked to stage 3 (rulings 35, 42) | Worlds that taught agents make the first meal big and near, lives long, food sensed, every life a new layout, and difficulty a dial that tightens only as he succeeds. Ours is the outlier: an 8% meal, and a lasting life needs the hardest skill first. | The entry settings at which his random movement finds food. Trees, wood, fire, cooking and shelter for his own needs in a physics world: single steps have been learned, the chain never. |
| Eye | 2 combine. 11-19: on the eye that turns with his body the hand forager passes at the entry rung on 20 of 20 fresh worlds (7.5 meals a life; shuffled rays 0 of 20) and FAILS at the full rung (3 of 20 against 18). Reported as a failure; the bar is unmoved (J12 done, ruling 51). Next for the eye: 128 rays against 16 under a learner, after pilot 2 | World models learn from camera images. | What it costs on our hardware. Small details. |
| Memory and sleep | 1 research: filed 11-15 (reference section 6). 11-20: the Inventor's two probes read alive (a 10,000-weight place key learned from time alone matches a 128-turn search, .934 against .933, and carries to unseen worlds); entered as a last-10% candidate for stage 3, no order now (ruling 63) | A small recurrent memory. Keeping old experience, chosen by reservoir or coverage. Layer-norm and weight decay to stay able to learn. Learning from stored experience while the body is idle. In a world that stays the same, weights plus kept experience carry memory across lives. | Memory across death and rebirth in one lasting world, at small scale. Whether concentrated sleep beats the same learning spread evenly. Finding himself again after rebirth with 16 rays. |
| Body and movement | 1 research → 3 reproduce as a side job. 11-20: the published walk is half-way: two seeds are home at 1,000,000 steps, both inside the published band (returns 4,065 and 5,294 against 3,784–5,866), but covering only 3.5 and 5.1 metres per 1,000 steps where the 2,000,000-step pass mark asks 12; the third seed and the next segments are running (J10). The body itself waits for stage 3 | Walking from scratch needs thousands of parallel copies (minutes on the free GPU) or about a million steps in one world. The network that moves the muscles should be small. A brain deciding 5 times a second needs a fast reflex layer under it. | One human-like body that both moves and forages. Hands on a walking body: unsolved by anyone from scratch. |
| Ear | 2 combine → 3 on 11-21. 11-19: not moved. The published 9,000-weight keyword network's known-answer run (published 96.9%) is frozen and launches on the free GPU after brain A's (J14, rulings 48, 52) | Hear through a spectrum (FFT or mel), never the raw wave. Networks of 9,000 to 100,000 weights learn about 35 spoken words from about 1,000 recordings per word. Reward alone does not train an ear: a label or a prediction loss on the sound does. | An ear trained by his brain's own prediction of sound, with no word labels. One voice that carries over to other voices. |
| Words and advice | 1 research: filed 11-15 (reference section 7) | Words are learned by predicting them, not from reward alone. Sight comes first. A parent who describes what he just did is the most efficient published teacher. Two small agents learned a handful of spoken command words from scratch, at very large budgets. The published word-learning world model is 150M+ parameters. | Spoken words with a world-model brain at our size, in one life, as advice he may ignore. Whom to trust when advice is false: no published test. |
| Other senses, all together | 1 research: filed 11-19 (reference section 18; smell is section 14). Nobody has published more than three senses fused from scratch with each one tested by hiding it, so ours is last-10% work; the record's one ablation with a known answer (hide his needs and he fails) is ordered as a cheap reproduction (ruling 55) | One small encoder per sense, all feeding one shared state. | More than about four senses in one brain. |
| The science lab (how his algorithms improve) | 2 combine: charter `docs/FORGE.md` written 11-20 from the published record (`PUBLISHED_EVIDENCE_2026-11-20_DISCOVERY`); lane `forge` starts 11-20 14h with its instruments (round runner, scoreboard). Placed by ruling 61: its first round is the eating wall, on a rule the Director freezes, after its instruments pass planted tests and E2c has read; the judge's tripwire is system order S13 | The loop is published: candidates, a fixed judge, short runs that buy long runs, winners become the baseline. The judge matters more than the search method; plain sampling is the control to beat; agents tamper with a judge they can write | Whether rounds of cook-offs raise the champion learner in HIS world, which source of candidates pays, and whether machine-written single functions ever beat plain variation here |

## The stages

**Stage 0: research the whole creature (now, about two days).** Review 1 is done: which brain, and how to test brains fairly (`docs/research/PUBLISHED_EVIDENCE_2026-11-15.md`). Review 2 landed the same day in eight parts: building the Dreamer brain correctly; building the control brain correctly; the correct test method; a battery of standard tests for brains; ear, words and advice; memory, sleep and learning for a lifetime; body, movement and simulation speed; the world, needs and lessons. It corrected several statements of review 1 (marked at the top of that report). After that the Researcher works twice a day, one question per sitting, and every order carries what the record says.

**Stage 1: the instrument (about a week for the world, the test harness and brain B; until early December for brain A, which needs the free GPU from 21 November).** Before any brain is judged, the thing that judges must be right.

- His world behind the standard interface that every published learner plugs into, with its speed measured and then raised: the simulator, not the learner, is the bottleneck (about 30 decisions a second today; the known fixes are estimated at 500 to 2,000).
- A first rung he can live on: difficulty as a dial (meal size, amount of food, distance, how fast it reappears), starting easy, with today's world near the hard end.
- Two brains built from published code: A, the Dreamer-style world model; B, the simple replay learner. Our own home-made brains are not ruled out (owner, 2026-11-15: "they need a fair chance"): each takes the same checks in the next line, and none is a candidate until it passes them.
- Each brain reproduces published numbers on small standard tasks.
- The setup the record supports: large replay, long horizon, sustained exploration, payment on eating, hunger varied at birth, food near at first.
- The test harness passes its own thirteen checks first (for example: a learner compared with itself must come out "no difference").
- Proof that the task can be learned with his senses: a hand-written forager that uses only what he senses, a copy of it inside his own network, and a learner given the direction to food. (11-19: shown at the entry rung only, on the eye and motor that turn with his body, by a hand forager that is a lower bound; ruling 51.)
- The prediction for stage 2, written down from the record.

**Stage 2: choose the brain (the simple brain's runs take under a week; Dreamer's runs set the date: on the free GPU alone, a verdict around mid-January to February 2027).** A against B in his own world, 1,000,000 decisions each, with enough repeated runs to believe the result (the number is set by a 20-run pilot). The record predicts that B eats reliably after 100,000 to 200,000 decisions and that A is close behind on today's simple body. Both also take a battery of standard tests: memory, exploration, delayed reward, and adding a new sense, where the record says A should pull ahead. The rule is written before the runs. A brain must pass its known-answer checks, learn to eat within the budget, and show the advantage the record predicts for it. If A does not and B does, B is his brain until evidence says otherwise. If both pass and tie on the rover, the battery and the added sense break the tie, because that is where he is going. Twenty full-length Dreamer runs are an estimated 160 to 280 hours of the free GPU, five to nine weeks of quota. Four things can shorten that, each with a price: a shorter run for every brain, less training per decision for Dreamer, letting its learner run freely beside the simulator, or the Mac mini once its speed is measured. Fewer runs is not one of them: it would throw away the ability to see a difference at all.

**A fair chance for our own brains.** The seven home-made brains failed a race that was too short, in a world whose reward gave no credit for eating. That says nothing final about them. Each gets the same standard-task checks as the published brains, then the same pilot runs in the repaired world: same length, same worlds, the same improved settings available to it, and a pass rule written before the runs. Any that learns goes into the decisive test beside A and B. Any that cannot pass the standard-task checks is set aside with its numbers on record, and may return if it is repaired.

**The decisive test, and what must be ready before the Mac arrives (owner, 2026-11-15).** The verdict on the brain comes from long runs: about twenty full-length runs per brain, a few hundred machine-hours in all, on the Mac mini (expected late December, running day and night) together with the free GPU. Until then, free compute has one job: to make sure every Mac hour is an hour of the real test, not of setup. Ready-list:

- His world: the first rung liveable, the test card passed.
- The simulator fast, re-certified, and running on macOS.
- The interface correct at death (the dying body's senses, not the reborn one's).
- Brain B reproduced, then piloted in his world on free CPUs. This needs no Mac: we should know before the Mac arrives whether anything learns to eat in the repaired world.
- Brain A reproduced at the early points of its published curves on the free GPU, and smoke-tested in his world.
- The harness's thirteen checks passed; the rule of the race written down.
- A first-day speed test for the Mac: the simulator, and each brain's learner on the Mac's processor and on its graphics chip. Nobody has published Dreamer's speed on Apple hardware, and its official code targets other graphics chips, so this is the largest unknown.
- The eye researched and designed, so that "add a sense" can be part of the decisive test.

What the Mac is expected to deliver (reference section 10; every figure is scaled from other machines, none is measured on a Mac yet): his world and the simple brain run fast there, and the simple brain's twenty runs take under a day. Dreamer is the slow one: its official code cannot use the Mac's graphics chip, and no version of it has been reported running on a Mac. For the smallest Dreamer at the published training rate, twenty runs are an estimated 3 to 11 weeks on the Mac alone and 1.6 to 5 weeks with the free GPU running alongside. The Dreamer size that has published scores (about 10 million parameters) is not affordable on any machine we have.

What that test will tell us: which brain learns to survive in his first world, at that length of run, with twenty runs each (large differences only), at the size and training rate he can afford to run. It does not tell us which brain would win with unlimited compute. What it will not: which brain carries many senses and words. That is tested again when the eye goes in, where the record expects Dreamer's advantage.

**Stage 3: grow him on that brain.** One part at a time, each through the five steps, each old bench proof earned again inside him. First he survives on the chosen brain (eat, drink, keep warm). Next he moves into his human-like body, before his eye, because what he sees and hears depends on the body he senses from (owner's question, 2026-11-15; reference section 4). Walking itself is a known recipe and does not wait for the brain: a separate, fast walking layer is built and checked in parallel from now, and when he moves in, the brain says where to go five times a second while that layer does the walking. Then his eye, because the record says words need sight first; then memory and sleep, the ear, words and the other senses, in the order their needs and the reviews set. Hands come in stages (a simple grab first, fingers later): hands on a walking body are unsolved by anyone from scratch.

**Stage 4: the problems nobody has solved.** These are Jack's own work, the right-hand column above: words from a parent's raw voice tied to his own life, memory across lives, many senses in one brain, a body with hands that survives, learning for months without forgetting, knowing whom to trust. Research gives the nearest results and the right experiment. Then we run real experiments, written down first.

## What stops today, and what continues

- Stops: racing home-made learners before the instrument is right; inventing a challenger with no published basis; world-rule races with no published basis.
- The long food-finding runs already on the free machines finish as pilots. No more of them are launched.
- Continues: repairs to the machine, tests of the world's mechanics, harvesting what is running.
- Old results: tests of the world and of the machine stand. Results about his brain that were earned with unchecked learners are marked "to earn again" on the proven brain.

## Who does what

- **Researcher** (twice a day) and the owner's session (the big reviews): step 1.
- **Director**: every order names its part, stage and step. No order skips a step. The morning report says, for each part, the step it is on and what moved.
- **Workers**: steps 2, 3 and 5, and running step 4.
- **Inventor**: invents inside the last 10% only, starting from what the record says is open.
- **Auditor**: reports any order or result that skipped a step.
- **Outsider**: checks the plan against the record and the code.

## Rules for agents

- An order carries `part:`, `stage:`, `step:` and, from step 2 on, a pointer to its section of `docs/research/PUBLISHED_PRIORS.md`.
- Step 2 leaves a design note in the module's docstring: where each part comes from, and an "ours beside theirs" table of every setting that differs from the published recipe.
- Step 3 leaves a known-answer row: task, steps, published score, our score. A learner that fails is a suspect, not a contestant.
- Step 4 is pre-registered: the prediction from the record, one primary measure, the number of runs and how it was chosen, the controls, and what result would mean "our code is broken" as opposed to "the task is hard". A test harness must first report "no difference" when a learner is compared with itself.
- A question the published record already answers is answered by citing it. A seat may be held BY RECORD: on published evidence, cited, and open to any challenger that brings its own prior.
- A result that disagrees with the record is a finding to report, never something to explain away.
- Nothing here changes the ends in `GOAL.md`, loosens a pass mark, or lets ready-made weights inside him.

## Honest limits

- Free compute sizes everything. Today 1,000,000 decisions take about 10 hours on one free machine, so making the simulator faster is part of stage 1.
- Checking brain A the published way needs an estimated 25 to 60 hours of the free GPU. A CPU cannot do it.
- About 100 free machine-sessions buy roughly 20 runs per brain. That proves only large differences (an 80%-against-20% success gap is found 96% of the time, 70%-against-30% only 61%). A modest difference between the brains may not be provable on free compute.
- The published brain that learned words is 150 million parameters or more. Words at our size are uncertain.
- The record is a prediction, not a guarantee. Nobody has published results at our small size or on our hardware.
- Stage 4 has never been done by anyone. That is the point of the project, and it is where our own tests belong.
