# PROGRESS — for the owner, on a phone

Written 2026-11-15 (Sunday) by the Director, full weekly sitting. Rewritten
every day; git keeps the old ones.

## The short version

The most important test in the project finally ran, and **it failed
honestly: living does not yet make him better, because he never learns to
eat.** In 120 lives he died of hunger 120 times. The curriculum contest ran
too and said the same thing from the other side: no way of ordering his
early life beats no ordering at all, because a learner that cannot find food
has nothing to be taught. Both tests had their cheating-controls in place
and no pass mark was touched.

That is real progress — we now know exactly what is wrong — and it changes
the work. **From today the workers' first job, every hour, is: get him to
forage.**

## What Jack can do today

He lives whole lives in the first world (W0): hunger, thirst, heat and cold,
his senses, a diary that survives death, death and retry. That is milestone
1 and its test passes again (re-bought 14 Nov).

What he cannot do: get better at it. Numbers a phone can hold:
- A body that does nothing at all lives 600 seconds. An untrained brain
  lives about 565. His trained brain lives about the same — and in one of
  three runs it wrecked itself, falling from 598 to 158 seconds, because of
  a known flaw in how his actions are trained (it can push without limit).
- To pass, he needs about 5 small meals per life, or one apple. He got 0,
  3 and 0. The world does contain enough food: a scripted forager eats 24–27
  meals in it.

The test ladder reads **110 of 270** (same as yesterday).

## Where he stands on the route (9 milestones)

1. **One whole life** — MET (test `LF.03` passes, current).
2. **A world and curriculum that can teach** — in work. World-rule tests
   still red; curriculum contest lost by every arm (see above).
3. **Brain shape by contest** — working memory decided. The
   movement-controller contest (`D1.0`) must be re-bought (below). Seven
   candidate learning brains are built; they now race on foraging.
4. **One body that moves** — waits on that controller contest.
5–6. Senses, words — not started beyond single tests.
7. **He keeps learning** — first test run (`LF.04`): FAIL, see top.
8–9. A living Jack, more worlds — not started.

## What moved since yesterday (59 worker runs; 3 ran out of time, none lost to budget)

- **"Does living teach him?" (`LF.04`) ran: FAIL.** Described above.
- **Curriculum contest (`CR.01`) ran: FAIL for all arms.** "No curriculum"
  keeps the seat by verdict. It will be re-raced once he can forage.
- **A lab idea died honestly:** "remember where meals were" beat its rivals
  on 5 of 5 runs but missed its pre-set margin by a hair (0.0985 against
  0.10). The bar was set before the run, so it stays dead.
- **The world:** a fifth rule contest (how he drinks) shipped nothing. A
  humid-heat rule is the first to fix one of the two red world tests on all
  5 runs; its last ten runs are in flight on free cloud machines. The
  workers also found the other red test partly measures its own stopwatch
  (how tired he gets "varies" only because lives are short). I ruled that
  check rebuilt as a real exhaustion-and-rest test, pass mark unchanged.
- **Free cloud CPU works well:** Kaggle runs 4 of our CPU jobs at once
  beside the GPU job. Almost all experiments now run there, not on your box.
- **Controller contest (`D1.0`): attempt 9 was void.** Kaggle quietly
  swapped its GPU for a slower model, so three of four contestants got only
  60% of their training. Time limits re-measured (no pass mark moved). The
  full contest now needs ~40 GPU-hours against 30 free a week, so it is
  being bought in slices; I ordered a try on Colab to finish next week
  rather than in three.

## What did not move

- **The video you asked for** — not started, fourth day. No good reason:
  the machine lane spent its slots on repairs. It is now second in that
  lane with "a rough 30-second clip today" as the instruction.
- "He moves better than random" (`T2.01`) — waits on the controller contest.
- The deeper world (W1): one test contradicted itself; ruled today.

## What the workers are on next

1. **Foraging.** First a three-run contest to fix the flaw that wrecks
   him; then all seven candidate brains plus a random baseline raced on
   meals per life. If none eats, the two best get four times longer, to
   tell "slow learner" from "cannot learn this".
2. Harvest the humid-heat runs; rebuild the tiredness check; re-run.
3. The deeper world's contradicting test, then its hunger clock.
4. Two old broken tests nobody owned (cold is felt before it kills: 94 days).
5. Write the missing tests for fire and for "advised, not commanded".
6. Machine lane: make the all-tests re-check green; your video; then close
   a hole that nearly put mislabelled results into the controller verdict
   (a crash saved it — nothing wrong was recorded).

Parked until he can forage, because they would only measure the same
failure again: the hand-over test, the bigger-scale brain race you asked
for (its seven brains are what race in step 1), the curriculum re-race.

## The honest paragraph (Sunday)

Closer, not just busier — for the first time in weeks. The week's biggest
step: the creature test ran with honest controls and told us the truth, and
milestone 1 holds. The worst drift: it took 70 hours to run a test that was
ready, and about 70 of every 119 commits were workers updating notes rather
than changing Jack. And a caution I owe you: a dozen cheap ideas to make
him find food (smell, a food sense, weaning, longer memory of reward) have
already died in the lab. Foraging may need a different brain rather than a
tweak. That is exactly what step 1 measures.

## The four machine numbers

| | |
|---|---|
| Points per worker run | 0.00 (110 → 110 over 59 runs; the two big verdicts were both FAILs, which score nothing and are worth more) |
| Slots lost to budget | 0. The CPU day overran a third time on 14 Nov (64,886 of 57,600 s; idle cloud-watchers were billed as work — fixed at midnight). Today so far: 4,700 s |
| Free GPU-hours expired | None that I can show unused; this week 7.8 h spent + 21 h in flight of 30. The weekly counter starts on the wrong day (known, queued) |
| Oldest unrepaired blocker | 95 days: "he moves better than random" (`T2.01`), behind the controller contest |

## Dead or stuck

- Nothing is dead. Workers hourly, auditor, re-checker, this review all on
  time; pushes work. Paper scout last ran 12 Nov (weekly, due Monday).
- Disk: 5.5 GB free on the system disk (workers stop under 3). Unchanged.
- Usage: 68% of the week used across everything (57% on this model), resets
  20 Nov; the stop is at your 95%. More than half of this week's use is not
  this project's. The meter gave two impossible readings (11 and 13 Nov).
- `run verify` (re-derives every pass from its record) is red a fifth day,
  now on one bookkeeping test about CPU budgets. Cause known; ruled today;
  first item in the machine lane.
- Old backlog: 15 rows left (25 yesterday).
- Steering files are all inside their size caps. I changed no prompts.
- Weekly re-check of 5 passing tests: 3 were too easy (the "one whole
  life" test would pass a body that never moves). Ordered strengthened.

## FOR THE OWNER

Nothing new. The standing one has a default so nothing waits on you: the
free Kaggle GPU week is shared with your other project. If you say nothing
by 21 Nov: no split — Jack takes a refusal as final for the week and
overflows to Colab.

Philosophy served today: 1 (he must learn by living — the work is now aimed
at the one thing stopping it), 3 (the brain that forages is chosen by a
race, not by me), 4 (three tests ruled where they measured themselves; no
pass mark moved; a near-miss stays dead), 7 (the finished-Jack table is
current).
