# PROGRESS — for the owner, on a phone

Written 2026-11-14 06:50 UTC by the Director (daily sitting). Rewritten every
day; git keeps the old ones.

## The short version

The most important test in the project — **does living make him better?**
(`LF.04`) — has still never run, 70 hours after it was written. Yesterday it
was stuck behind two things. Both are cleared this morning: the speed test
passed on a quiet machine at 06:10, and I cut a wrongly-drawn link that made
it wait on an unrelated language-memory test. The workers' first job today,
before anything else, is to re-check the three tests under it and send it off
to free cloud CPU.

## What Jack can do today

The learning brain lives whole lives in the first world (W0): hunger, thirst,
cold, his senses, a diary carried across deaths. **He does not yet learn to
find food by living** — nothing shows him improving. That is what `LF.04`
asks, against a frozen copy of himself.

One honest caveat on milestone 1 ("one whole life"): its test passed on
12 Nov, but files underneath it were re-run or edited since, so the pass is
marked stale and must be bought again before I count it. I expect it to pass
again; I am not counting it until it does.

Something the workers found that matters: the brain currently holding the
"learning core" seat is a "world model" in name only — the part meant to
predict what his actions do never gets trained. The bigger-scale race you
asked for (`LC.09`) includes a challenger that does it properly, and will
disqualify the current seat-holder on that point. That is the race working.

The test ladder reads **110 of 270** (107 of 267 yesterday).

## Where he stands on the route (9 milestones)

1. **One whole life** — met 12 Nov, pass now stale; re-buy ordered first.
2. **A world and curriculum that can teach** — in work, and the hard part
   this week. See below.
3. **Brain shape by contest** — working memory decided; fusion raced, no
   winner; the movement-controller contest (`D1.0`) is running on free GPU.
4. **One body that moves** — waits on that contest.
5–9. Senses, words, never-stops-learning, a living Jack, more worlds — not
   started, except single tests.

## What moved since yesterday (57 worker runs, none failed)

- **Speed test green again** (`LC.02`). Its two reds were other programs
  loading the machine, not his brain getting slower. It now refuses to give a
  verdict on a busy machine instead of failing.
- **Seed-noise test passed** on Kaggle GPU (`T1.14`) — it says how much two
  identical runs differ, which every other test's margins lean on.
- **The world's heat rule passes** (`W.1`).
- **The world, honestly:** four contests were run to fix his body clock and
  the heat (slower hunger/thirst, sweat, water, humidity). **None won.** But
  they found real things: once he can sweat, heat can no longer kill him in
  W0 at all; humid heat can, the way it kills people. And two of the losses
  were the *test's* fault — the "nobody survives by accident" test (`NE.01`)
  measured time in seconds tuned to the old fast clock, and its injury check
  measured its own schedule. I ruled both today: time is measured in the
  world's own units, injury is tested to the lethal edge. No pass mark moved.
- **The deeper world (W1):** one of its tests was proven impossible to pass
  in *any* world (a counting mistake in how lives were lined up). Ordered
  rebuilt with the same pass mark.
- The cloud-CPU route (free Kaggle machines without GPU) is built and tested.
- The curriculum contest (`CR.01`) is ready to run — six ways of ordering his
  early life, against "no ordering at all".

## What did not move

- `LF.04` — see top. A day lost to a link nobody had authority to cut until
  this sitting, and to our own edit last night that staled the speed test.
  I have frozen the files under it until it is sent off.
- The video you asked for: not started, third day. It is next in the machine
  lane after one repair.
- "He moves better than random" (`T2.01`) — waits on the controller contest.

## What the workers are on next

1. Re-check the three tests under `LF.04`, then run it on free cloud CPU.
2. Send the curriculum contest to free cloud CPU (it needs none of ours).
3. Rebuild the two mis-measuring world tests and re-run them.
4. The hand-over test (train on one machine, keep learning on another).
5. The bigger-scale brain race, priced and sent to cloud CPU.
6. Watch the controller contest on GPU (25 hours of work, in 13 pieces).

## The four machine numbers

| | |
|---|---|
| Points per worker run | +0.05 (107 → 110 over 57 runs) |
| Slots lost to budget | 0 worker slots. But the CPU day overran again (59,599 of 57,600 s) and refused tests for six hours. Now capped: races may use 60% of the day, 2 processes at once |
| Free GPU-hours expired | None unused: last week's 30 Kaggle hours all went to your other project. This week the controller contest holds them |
| Oldest unrepaired blocker | 94 days: "he moves better than random" (`T2.01`), behind the running contest |

## Dead or stuck

- Nothing is dead. Workers, auditor, re-checker and this review ran on time;
  pushes work. The paper scout last ran 12 Nov (weekly).
- Disk: 5.6 GB free on the system disk (workers stop under 3). Unchanged.
- Usage: 57% of the week used, resets 20 Nov; the stop is at your 95%. The
  meter gave one impossible reading on 13 Nov (fell from 69% to 26%); two
  thirds of this week's use is not this project's.
- `run verify` (re-checks every pass from its record) is red for the third
  day on one test whose pass cannot be re-derived. Repair is first in the
  machine lane, and needs the quiet machine we have this morning.
- The speed test peaked at 2.1 GB of memory against our 1.5 GB limit. Brief,
  nothing was harmed; I am watching it.
- Old backlog: 25 rows left (35 yesterday).

## FOR THE OWNER

One thing, and it has a default so nothing waits on you. **The free Kaggle
GPU week is shared with your other project, which used all 30 hours last
week; Jack's GPU work waited three days.** If you want a split, say so. If
you say nothing by 21 Nov: no split — Jack runs smallest jobs first, takes a
refusal as final for the week, and overflows to Colab.

Philosophy served today: 1 (he learns by living — the test that asks it is
unblocked), 4 (two tests that could not be passed honestly are rebuilt, no
pass mark moved), 6 (the box's CPU is capped for the paying customers).
