# PROGRESS — for the owner (written by the Director, Tuesday 24 Nov 2026, 12:35 UTC)

Glossary: *brain A* = the Dreamer-style world model; *brain B* = the simple
replay learner (SAC, from a public library); *a decision* = one action, 0.2 s
of his life; *a known-answer row* = our build re-running a published
experiment to see if it gets the published number; *the critic* = the part
of brain B that estimates how good a situation is; *the forge* = your
science lab; *a blind twin* = the same run with the eye switched off;
*a shuffled twin* = the same run with the eye's numbers scrambled.

## The short version

**The team was stopped for 45 hours.** The shared weekly usage allowance
hit 100% on Sunday afternoon; 136 worker runs and two of my sittings were
refused. Work resumed at 09:51 today when you signed the box in again. The
free machines kept computing through the stop, and most of what they
finished has been read this morning.

**No learner eats from his own eye yet, and the cure hunt is closed.** Four
published cures for brain B's runaway critic were each tried on one run
beside a blind twin. None stops it, and none eats: 0.2 to 1.1 meals a life,
where a hand-written rule using the same eye eats 7.9. I closed the hunt by
the rule written before the runs.

**What comes next is not a fifth cure but a different door.** The published
record says a replay learner fails on raw images and that one recipe alone
repairs it: a small pattern-finding (convolution) layer plus randomly
shifting the image during learning. His eye is a row of 128 rays, an image
one pixel tall, and we have been feeding it in as 256 loose numbers. That
recipe is being built for his eye today, with a blind and a shuffled twin
and the pass/fail rule written first. If it fails too, brain B is finished
on this eye for this stage and the question passes to brain A.

**His world is confirmed learnable by sight** on the one physics version:
the hand-written forager passes its bar again, 19 of 20 worlds (the control
that must fail is running now).

## What Jack can do today

He lives, senses, hungers, dies, is reborn, and keeps a diary. A learner
eats when handed the direction to food, for a while. Nothing learns to find
food from his own eye, and nothing keeps the skill. Test ladder: 109 of 270
(110 on Sunday: one passing test went stale and has not been re-run; I
have ordered it named).

## Each part of the plan: the step it is on (0–5)

- **His brain — 3 (reproduce), not cleared.** Moved: B's three misses were
  dissected. One (reacher-hard) was mostly our test being too small (10
  test episodes; on 100 the same models score 908, inside the published
  906–1000) — it is registered once more on fresh runs. One
  (swingup-sparse) is B's own fault: a run stops exploring before it ever
  finds a reward. Brain A: unchanged; its true-size check is still to be
  frozen, and now goes on the free Colab GPU rather than waiting.
- **Learning to eat — 4.** Moved: all four cures read, hunt closed, the
  image recipe ordered. Not moved: the lab's "keep the skill" round — its
  30 runs came home Sunday and nobody has read them.
- **His world — 2.** Moved: the entry rung re-proven on the one physics
  version (19 of 20).
- **Eye — 2.** Not at fault by today's evidence. The lab's second route (a
  table of what was at his mouth when he ate) lit itself from empty in 8
  fresh worlds; it stays a candidate until its research is done.
- **Memory and sleep — 1.** Not moved.
- **Body and movement — 3 as a side job.** Moved: the repaired walk row is
  running, two of five runs, about 10 hours a piece on free machines.
- **Ear — 3, VOID.** Not moved (its timing run is next on the free GPU).
- **Words and advice — 1.** Not moved.
- **Other senses together — 3, FAILED twice, closed until stage 3.**
- **Science lab — 4 on round 1.** Runs home, unread. The lock that stops
  anyone editing the judge now exists and was tested with planted edits.

## What moved since Sunday

The last cure (SimBa) was run on a free Colab GPU in 56 minutes after 8.5
hours on a free CPU had returned nothing. B's saved models re-tested. The
hand forager re-read on the one physics version. The walk relaunched. The
judge lock built. One lab idea (a parent feeds the newborn once) tried and
honestly killed at its pilot. Nine rulings (92–100).

## What did not move

No test point. No worker did anything for 45 hours. The lab round's
results sat unread for two days. Brain A's true-size check is still not
frozen. The three machine orders after the lock were not reached.

## What the workers are on next

1. Build the image recipe for his eye; three runs; take the written branch.
2. Read the lab round's 30 runs behind the judge lock, then launch its
   second rung. If the lab lane has not by 06:00 tomorrow, a worker does.
3. Finish the physics re-read. 4. Carry the walk's runs forward.
5. Brain A's and the ear's timing runs on the free Colab GPU, today.
6. Brain B's reacher-hard row once more. 7. The Mac port's next rungs.

## The machine, in four numbers

- Test points per worker run: 0 over 7 runs since the restart (−1 overall).
- Worker runs lost to budget: **136** (11-22 13:00 to 11-24 09:51).
- Free GPU-hours expired: 0. Kaggle's 30 for this week were spent by last
  Saturday; the free Colab GPU is being used instead until Saturday 28th.
- Oldest unrepaired blocker: the cold-is-felt test, 104 days, parked to
  stage 3. Oldest not parked: a rebirth test stale 78 days, queued.

## Dead or stuck

- **The usage allowance will stop the team again.** It reads 15% after one
  day, resets Monday 30 Nov 05:00 UTC, and about half of what is spent on
  it is not Jack's. At last week's pace the workers stop Thursday night
  and stay stopped three days. I have not touched the stop. I ordered one
  saving: a worker run that would find every machine busy and nothing to
  read now exits before it costs anything.
- **The auditor and the paper scout have not run since the stop** (auditor
  last real audit 22 Nov; scout 19 Nov). They are gated by the same meter
  and should return on their next slots; if the auditor's page still reads
  22 Nov tomorrow I will treat it as dead.
- Data disk 15 GB free — at its floor. Cause found: 136 job folders nothing
  deletes, plus caches. The clean-up is the first machine order.
- Main disk 5.4 GB, level. Pushes level with GitHub.
- The old review backlog has no open rows.

## FOR THE OWNER

Nothing blocks on you. One thing only you can change: **the weekly
allowance is shared, and Jack lost two days to it.** If you want him to
keep working through a week, either give him his own allowance or say what
share is his. Default on 29 Nov unchanged: recordings of how people move
stay outside him, as a judge, never a part of him.

Philosophy served today: 3 (the hunt closed by its rule; the next arm
taken from the published record, not invented), 4 (a miss dissected and
found partly our own ruler; the rule for a cure made stricter, not
looser), 5 (unread results given a deadline; idle runs made free), 6 (free
Colab GPU in place of a spent Kaggle week).
