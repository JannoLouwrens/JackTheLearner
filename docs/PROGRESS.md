# PROGRESS — for the owner, on a phone

Written 2026-11-20 (Friday) 12:35 UTC by the Director, daily sitting (held
late: the machine was paused on the usage limit until about 09:30).
Rewritten every day; git keeps the old ones.

## The short version

**Pilot 2 is home, and the answer is no: no learner ate.** Brain B (the
simple published learner) ran in five setups of his repaired world, forty
runs, 300,000 decisions each (one decision is one action, 0.2 seconds of
his life). The rule was frozen before any run and the reading was done by
code, not by a person. Not one learner had a sustained life. Worse: every
learning run, on every seed, ended eating 0.08–0.16 meals a life — below
the same brain with learning switched off (0.82) and below random pushes
held for a moment (0.95). Learning made him eat less.

**That points at the learner's setting, not his world.** It happened even
in the setup where moving costs no energy, so the "lie still to save
energy" explanation does not cover it. The Researcher priced a suspect: we
ran the learner with a fixed "exploration bonus" that pays him, every
moment he is alive, 15–18 times more than food does. No published forager
used that setting; the published one lets the bonus tune itself. The
published record's prediction (he eats inside 300,000 decisions in the easy
setup) failed, and is reported as failed.

**The test that separates the causes is already running.** A learner that
is simply told the direction to food started at 12:11. Ruled today: before
its last seeds it gets twins with the published self-tuning setting. If the
twin eats and the fixed one does not, the setting was the wall. If neither
eats, the next question is run length (the published forager needed two
million decisions), and that is the science lab's first round.

**Your science lab is placed.** Its charter, lane and schedule (02h and 14h
UTC) exist. Today's rulings: it builds and tests its own instruments first;
its first round is the eating problem, on a rule I freeze before any run;
it may never edit its own judge, and a tripwire that checks the judge's
files is now a system order. The research review's own correction stands:
no machine-written reward search as a first move.

## What Jack can do today

He lives whole lives in his first world, with needs, senses, a diary that
survives death, death and retry. He still does **not** get better by
living. The test ladder reads **111 of 270**, unchanged: this stage builds
the instrument, and its work does not move that number.

## Each part of the plan: the step it is on (0–5)

(1 research, 2 combine from published parts, 3 reproduce a published
number, 4 test what is new, 5 in him.)

- **His brain — 3 for brain B, 2 for brain A (the Dreamer world model).**
  *Moved:* the home-made brains' repaired entry reproduced one standard
  task (4 of 5 — the first time any home-made brain reproduced a published
  number) and failed the other (0 of 5). A one-hour diagnostic found the
  cause: holding exploration noise for 32 decisions helps the sparse task
  and kills the dense one. Ruled: one last entry with one setting for both
  tasks, then that chapter closes either way. The Mac port passed its next
  rung (9 of 9 maths checks). Brain A's three reproduction runs start
  tonight at 00:00 UTC on the free GPU.
- **Learning to eat — 4.** *Moved:* pilot 2 read (above); the
  told-the-direction test launched.
- **His world — 2. Eye — 2.** Not moved.
- **Body and movement — 1, with a reproduce side job.** *Moved:* the
  published walk is half-way. Two seeds at one million steps are inside the
  published score band, but they mostly stand: 3.5 and 5.1 metres per 1,000
  steps, where the two-million-step pass mark asks 12.
- **Ear — 2.** Not moved; its reproduction run follows brain A's this week.
- **Memory and sleep — 1.** *Moved (lab probe only):* a tiny network that
  learns "where am I" from time alone recognised places as well as a slow
  search, and carried to worlds it never saw. Noted for stage 3; no order.
- **Words — 1. Other senses — 1.** Not moved.
- **The science lab — 2.** New today (above).

## What did not move

- No learner has eaten in his world.
- **About 15 worker-hours were lost overnight.** Weekly usage hit the stop
  at 18:07 on the 19th and the machine stayed paused until about 09:30
  today. Runs that came home in the evening sat unharvested and all five
  free CPU machines stood idle for 13–15 hours.
- System clean-up (S9) moved one slot; the steering-file trims and the disk
  prune (S11) were not started and are now behind the TPU door.

## What the workers are on next

1. The told-the-direction test with its self-tuning twins; reading per arm.
2. From 00:00 UTC tonight: brain A's three reproduction runs, A's first
   steps in his world, the ear's run, and one small "hide his needs and he
   must fail" reproduction — about 17 of the week's 30 free GPU-hours. The
   rest is held for brain A's own pilot in his world, if A reproduces.
3. The home-made brains' last entry.
4. The walk's remaining segments; the Mac port's rung 2.
5. Science lab: its round runner and scoreboard, then the candidate pool.

## The machine, in four numbers

- Test points per worker run: 0 over about 25 runs. Landed instead: forty
  pilot runs read, the diagnostic, two walk seeds to one million steps.
- Worker runs lost to budget: about 15 hours of every lane (the usage stop,
  above). Usage now reads 25% of the new week (all models), and the
  attribution tool says 80% of that is not this project.
- Free GPU-hours expired this week: 0 (33.8 of 30 used). **Free TPU-hours
  expired: all 20.** Kaggle also gives 20 hours a week of a second
  accelerator (TPU) and we have never used it; nothing here can send a job
  to it yet. Ordered today (S12): the door, then your gate test — does
  Dreamer run on it, and is it at least as fast as the GPU.
- Oldest unrepaired blocker: the cold-is-felt test, red 99 days, parked to
  stage 3 (the fault is in the probe, not his senses).

## Dead or stuck

- The auditor has not reported since 19 Nov 00:37 (the usage stop refused
  its last sittings); its next slot should run now that usage is 25%.
- The paper scout last swept 19 Nov 06:00; today's sweep was refused too.
- Main disk 4.8 GB free (workers stop under 3), down from 5.3. The prune is
  still unstarted — it is the next system item after the TPU door.
- Kaggle allows five CPU machines, not four; the ranking is redrawn for
  five, and every worker slot now starts by harvesting and refilling.
- The usage meter has given three impossible readings this month
  (reporting only). The new week already reads 25% eight hours after its
  reset, mostly from outside this project — at that pace the stop returns
  before the week ends.
- A machine self-test (`test_lane_guard.sh`) showed 5–6 failures this
  morning; ordered reproduced and fixed by cause (S13).

## FOR THE OWNER

One thing worth your eye: the weekly usage allowance is shared with your
other work, and that share stopped Jack for 15 hours last night. Nothing to
decide unless you want Jack to have a reserved share. The standing question
keeps its date: whether recordings of how people move may ever be a part
inside him (default on 29 Nov: no — outside him only, as a judge).
