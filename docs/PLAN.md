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
| His brain | 1 research: first review done, build review running | A Dreamer-style world model: one brain that learns by predicting every sense, proven from scratch on 150+ tasks. A simple replay learner (SAC type) as the control. | Does it hold at our small size and on free hardware? Which one wins in his world? |
| Learning to eat | 2 combine: the setup is known, not built yet | Keep everything he lived (a large replay). Look 40 seconds ahead, not 4. Move in sustained runs, not jitter. Pay him when he eats. Food near at first. 100,000 to 1,000,000 decisions. | Can 16 distance rays tell food from a wall? |
| His world, needs and lessons | 1 research running | How the survival worlds that did teach agents were built and staged. | Trees, wood, fire, cooking and shelter learned from scratch in a physics world. |
| Eye | 0 not started | World models learn from camera images. | What it costs on our hardware. Small details. |
| Memory and sleep | 1 research running | A small recurrent memory. Keeping old experience. Layer-norm and weight decay to stay able to learn. | Memory across lives beyond weights and replay. What sleep should do. |
| Body and movement | 1 research running | Walking is learned in minutes in parallel simulation. A motor layer under the brain. | One human-like body with hands that also survives. No published learner manages it. |
| Ear | 1 research running | Small networks learn a few dozen spoken words from scratch. | A learned ear inside a living learner. |
| Words and advice | 1 research running | Words are learned by predicting them, not from reward alone. Sight comes first. | Words from a parent's raw voice, tied to his own life. Nobody has done it. |
| Other senses, all together | 0 not started | One small encoder per sense, all feeding one shared state. | More than about four senses in one brain. |

## The stages

**Stage 0: research the whole creature (now, about two days).** Review 1 is done: which brain, and how to test brains fairly (`docs/research/PUBLISHED_EVIDENCE_2026-11-15.md`). Review 2 started on 2026-11-15 in eight parts: building the Dreamer brain correctly; building the control brain correctly; the correct test method; a battery of standard tests for brains; ear, words and advice; memory, sleep and learning for a lifetime; body, movement and simulation speed; the world, needs and lessons. After that the Researcher works twice a day, one question per sitting, and every order carries what the record says.

**Stage 1: the instrument (about a week).** Before any brain is judged, the thing that judges must be right.

- His world behind the standard interface that every published learner plugs into, with its speed measured.
- Two brains built from published code: A, the Dreamer-style world model; B, the simple replay learner. No home-made learner is a candidate until it passes the next line.
- Each brain reproduces published numbers on small standard tasks.
- The setup the record supports: large replay, long horizon, sustained exploration, payment on eating, hunger varied at birth, food near at first.
- Proof that the task can be learned with his senses: a hand-written forager that uses only what he senses, a copy of it inside his own network, and a learner given the direction to food.
- The prediction for stage 2, written down from the record.

**Stage 2: choose the brain (about two weeks of free compute).** A against B in his own world, 1,000,000 decisions each, with enough repeated runs to believe the result (the number is set by a 20-run pilot). The record predicts that B eats reliably after 100,000 to 200,000 decisions and that A is close behind on today's simple body. Both also take a battery of standard tests: memory, exploration, delayed reward, and adding a new sense, where the record says A should pull ahead. The rule is written before the runs. A brain must pass its known-answer checks, learn to eat within the budget, and show the advantage the record predicts for it. If A does not and B does, B is his brain until evidence says otherwise. If both pass and tie on the rover, the battery and the added sense break the tie, because that is where he is going.

**Stage 3: grow him on that brain.** One part at a time, each through the five steps, each old bench proof earned again inside him. First he survives on the chosen brain (eat, drink, keep warm). Then his eye, because the record says words need sight first. The order after that (memory and sleep, the human-like body, the ear, words, the other senses) is set by what each part needs and by review 2. The human-like body with hands is the hardest known part and gets its review before any more compute is spent on it.

**Stage 4: the problems nobody has solved.** These are Jack's own work, the right-hand column above: words from a parent's raw voice tied to his own life, memory across lives, many senses in one brain, a body with hands that survives, learning for months without forgetting, knowing whom to trust. Research gives the nearest results and the right experiment. Then we run real experiments, written down first.

## What stops today, and what continues

- Stops: new races between home-made learners; inventing a challenger with no published basis; world-rule races with no published basis.
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
- The record is a prediction, not a guarantee. Nobody has published results at our small size or on our hardware.
- Stage 4 has never been done by anyone. That is the point of the project, and it is where our own tests belong.
