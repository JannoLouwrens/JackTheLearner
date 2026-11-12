# PROGRESS — for the owner, on a phone

Written 2026-11-12 06:45 UTC by the Director (daily sitting). Rewritten every
day; git keeps the old ones.

## What Jack can do today

**New since yesterday: the learning brain has lived whole lives.** In the
first world (W0) the brain that learns — not a script — drove the body through
five deaths on each of three runs, with hunger, thirst, cold and every W0 sense
on, a diary carried across each death, and a save-and-resume in mid-life that
came back bit-for-bit. His weights changed as he lived (by 22%). Two fakes
were run to prove the test can fail: a frozen copy of him (caught: nothing
changed) and a copy with its diary wiped (caught). This is test `LF.03`, and
it is roadmap milestone 1.

What that does **not** mean, so it is not oversold:

- He is bad at living. He ate about twice in five lives; lives last 5–9
  simulated minutes. Nothing yet shows a later life going better than an
  earlier one.
- He carries his diary but does not consult it — nothing in his decisions
  reads it yet.
- He still cannot move well, hear with his own ear, or use words.

The test ladder reads **109 of 264** demonstrated (108 yesterday).

## Where he stands on the route (9 milestones)

1. **One whole life** — **met** (`LF.03` passed, a day after it was ordered).
2. **A world and a curriculum that can teach** — three world-rule tests still
   red, deeper world half built, no curriculum. Not touched yet.
3. **Brain shape decided by contest** — the "who controls the body" contest
   is running on a free GPU right now; the "how unified is the brain" contest
   is being repaired (see below). A memory contest is ready to run.
4. **A body that moves** — waits on the controller contest in 3.
5. **Every sense in one brain** — unchanged. 6. **Words** — nothing of his own.
7–9. Keeps learning / a living Jack / beyond the jungle — not started.

## What moved since yesterday (9 worker runs, all completed)

- `LF.03` passed — above.
- `T1.14` ("our measurements are bigger than run-to-run noise") passed with a
  new control that is actually run: feed the same random seed three times and
  the test must reject it. It did. 45 tests that hung on this are unblocked.
- The body-controller contest (`D1.0`, four rival controllers, each judged
  against its own untrained twin) was rebuilt as 13 small jobs and started on
  Colab at 05:21. It needs about 25 GPU-hours.
- The one-brain contest (`UB.10`): a worker found its test bed could not tell
  the rivals apart, tried a harder one on a real GPU, and that one failed the
  other way (every rival fell below the "has learned anything" floor). The
  worker did not tune it by eye: four candidate settings and the rule for
  choosing between them were written down before any is run.
- The Kaggle mystery is solved: your Kaggle account's 30 free hours were
  already used up this week by the other project (`next-l1-icsi-*`). Our code
  mistook Kaggle's refusal for success and then billed itself for waiting.
  Fixed; 3 wrongly billed hours refunded on our meter.

## What did not move

- The world repairs, the deeper world and the curriculum: not started. The
  workers spent the day on the creature test and the two GPU contests, which
  is the order I gave them.
- One finished GPU run (1.1 h) was lost because the result file's download
  failed after the computing was done. Third time this has happened. Repair
  ordered today (S5) — the result will also be read from the job's printed
  output.
- One machine test (`T0.28`) turned red overnight as a side effect of my
  settling old decisions: it treats "no decision is waiting" as "the page is
  broken". Repair ordered.

## What the workers are on next

1. `WM.02` — a 25-minute run: a new working-memory design from the lab against
   the current one, for a seat nobody holds.
2. **`LF.04` — does living teach him?** His later lives against his own
   frozen twin's, on every run, not on average. If he does not improve, that
   is the finding and finding out why becomes the work.
3. Watch the controller contest; seat the winner; retry "he can move".
4. Finish choosing the one-brain test bed, then run that contest.
5. The red world rules; the deeper world; the curriculum contest.
Then: make the diary enter his decisions.

Machine work (3 orders): never lose a finished GPU run to a download; stop
guard tools being red by design; six passes must hold on every run, not the
average.

## The four machine numbers

| | |
|---|---|
| Points per worker run | 0.11 (108 → 109 over 9 runs: two first-ever passes, one machine test lost) |
| Slots lost to budget | 0 |
| Free GPU-hours expired | 0 so far. Used this week: 2.9 h Colab. Kaggle's 30 h went to the other project. The controller contest wants ~25 h on Colab alone — Colab may cut it off; if so the rest waits for Kaggle's weekly reset |
| Oldest unrepaired blocker | 73 days (`W.1`, `W.2`, world heat and needs rules) — owned, not started |

## Dead or stuck

- Nothing is dead. All five agents ran on time and finished cleanly. Pushes
  work. Disk: 6.1 GB free on the system disk (workers stop under 3).
- Usage meter: reads 19% of the week (resets 13 Nov 04:00 UTC). Last night's
  76% reading was the false one; 63% of what has been used is not this
  project's.
- Colab allows one job at a time on your account, so the two GPU contests
  queue behind each other.
- Old paperwork: 4 stale process decisions and 63 backlog rows remain (I
  settled 3 and 10 today, none moved a pass mark).

## FOR THE OWNER

Nothing needs a decision. For your information only: Jack's free Kaggle hours
are shared with your other project, and this week it took all 30.
