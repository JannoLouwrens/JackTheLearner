# PROGRESS — for the owner, on a phone

Written 2026-11-13 08:50 UTC by the Director (second sitting today, after
your END-FIT session). Rewritten every day; git keeps the old ones.

## Your picture of the finished Jack — where I put it

Human-like body; trained here, handed to people, still learning there;
advised in words. It now judges every brain design (the END-FIT table). Today:
- **Ordered now (J8): the hand-over test.** Train him here, move the one saved
  file to a different machine, and check he still knows what he knew AND
  learns more there. Two fakes must fail (a stripped file, a fresh brain).
- **On the route map:** the tool chain (tree → wood → fire → cooked food,
  buildable shelter) under milestone 2; hands that hold and strike under 4;
  advice in words under 6. They open one at a time behind the hand-over test —
  the chain needs the deeper world's fire built first, hands need a controller.
- Corrected a stale line that described his language input as a ready-made
  text model. Nothing ready-made goes inside him.
- The "does living teach him" test (`LF.04`) still comes first, as you wrote.

## What Jack can do today

Same as yesterday, with one honest sharpening. The learning brain lives whole
lives in the first world (W0): hunger, thirst, cold, his senses, a diary
carried across deaths (test `LF.03`, milestone 1). **He does not yet learn to
find food by living**, and yesterday the workers found out a lot about why:

- Six cheap explanations were tested by rules written down first (eye height,
  smell, planning horizon, timid wandering, birth distance, nursing then
  weaning). **All six are dead.** He cannot yet connect "I walked this way"
  with "then I ate".
- A second suspect turned up in the world itself: **his body clock is wrong.**
  The world says one of its days is 20 minutes (72× faster than life). Cold
  obeys that. Hunger and thirst run 6–15× faster still: he dies of thirst in
  about 9½ simulated minutes — less than half a day, before a night has ever
  passed. I ruled today that the world has one clock, and ordered the fix
  raced against the current world before it ships.

New: he has a better short-term memory. A "ring" memory invented in the lab
beat the old one (0.90 against 0.58 on five fresh runs, both fakes caught) and
now holds that seat by contest (`WM.02`).

The test ladder reads **107 of 267** (109 of 264 yesterday).

## Where he stands on the route (9 milestones)

1. **One whole life** — met, with a caveat that appeared yesterday: a speed
   test it leans on (`LC.02`, "the brain is fast enough to live a life") went
   red while our own experiments were loading the box. Probably the box, not
   the brain. First job today is to re-run it on a quiet box and know.
2. **A world and a curriculum that can teach** — now really moving. Shelter
   rule fixed; the two 73-day-old red world rules are diagnosed down to one
   cause (the clock above); curriculum research written, six rival curricula
   built and sized; deeper world's two broken tests repaired, not yet run.
3. **Brain shape decided by contest** — memory seat won. The "one brain or
   several experts" contest (`UB.10`), stuck 59 days, finally has a verdict
   (06:43 today): **no design wins.** All six sit within 2 points of each
   other and the best one changes from run to run. The seat stays undecided
   — raced, not assumed — and I ruled we do not hunt for a kinder test. The
   "who controls the body" contest is **running now** on a free Colab GPU
   (attempt 4, about 25 GPU-hours in 13 resumable pieces).
4. **A body that moves** — waits on that controller contest.
5–9. Senses / words / keeps learning / living Jack / beyond — unchanged.

## What moved since yesterday (51 worker runs, none failed)

- `WM.02` passed — the ring memory, above.
- `LF.04` ("does living make him better", him against a frozen copy of
  himself) was written and registered with both fakes. **It has not run** —
  see below.
- The world: shelter clause green on all three runs; the cause of the two
  oldest red tests found; a four-way race of "how the sky chills a sleeper"
  run to an honest no-winner.
- The one-brain contest: test bed chosen by a rule fixed in advance, run
  twice on a real GPU, one defect repaired without moving any pass mark.
- Finished GPU runs can no longer be lost to a failed download (`S5`).

## What did not move

- **`LF.04` — the most important test — sat unrun for 46 hours.** An old rule
  (correctly) forbids 4-hour jobs on this box, which serves paying customers;
  meanwhile unregistered experiments used the whole CPU day (62,000 s against
  a 57,600 s ceiling). That was my order's fault: I ordered a test the box
  could not run. Fixed today: it runs on free Kaggle CPU, building that route
  is the first machine order, and no new side-experiments start until it has
  a verdict.
- The speed test could not be re-run: other projects kept the box busy all
  morning (load 3–7). Workers now try it first in every slot.
- The free Kaggle-CPU route has no code yet. It is `LF.04`'s only road.
- "He can move" (`T2.01`), the biggest single blocker (25 tests behind it):
  still waiting on that contest.

## What the workers are on next

1. Re-run the speed test the minute the box is quiet; otherwise build the
   Kaggle-CPU route.
2. Deeper world: one repaired test is running now (its prerequisite re-passed
   at 07:56); the second follows.
3. Watch the controller contest on the GPU.
4. Race the honest world clock against today's world (both clock tests were
   re-written against the one declared clock this morning).
5. Register the hand-over test. Finish the free Kaggle-CPU route and send
   `LF.04` through it; then the
   curriculum contest (six rivals), then **your scale question**: a new 3×
   contest of all the learning brains plus the ring memory.
6. Your video: a phone-playable clip of his latest life, hunger and thirst
   overlaid, at `/data/jack-video/latest.mp4` once built. Not built yet.

**Bigger scale, honestly:** a first 3× verdict in one to two weeks (needs the
Kaggle-CPU route, then a pricing pilot). The full 10× is weeks after that.

## The four machine numbers

| | |
|---|---|
| Points per worker run | −0.04 (109 → 107 over 51 runs: 2 won, 4 lost, among them the speed test under load and one machine self-check) |
| Slots lost to budget | 0 worker slots. But the CPU day ran out at 22:48 and refused three experiments |
| Free GPU-hours expired | None of ours: Kaggle's 30 h were used by your other project. Colab 5.4 h used this week, contest running |
| Oldest unrepaired blocker | 93 days: "he moves better than random" (`T2.01`). The 73-day world rules are diagnosed and ordered |

## Dead or stuck

- Nothing is dead. Workers, auditor, re-checker and this review all ran on
  time; pushes work. The paper scout last ran 12 Nov 06:00.
- Disk: 5.6 GB free on the system disk (workers stop under 3) — shrinking
  about half a GB a day; I will order a clean-up if it reaches 4.5.
- Usage meter: 5% of the new week (resets 20 Nov). Readable.
- Two tools exit red (`coverage`, `run verify`): one by design, one because a
  new test's pass cannot be re-checked from its record. Repair ordered (S3).
- Old paperwork: the 4 stale process decisions are closed (each by its
  written default, nothing built); the old backlog is 35 rows, down 20 today.
- Our GPU budget file says Kaggle has 30 h left; Kaggle says 0 (your other
  project used them). Workers trust Kaggle. A small fix is queued.

## FOR THE OWNER

Nothing needs you. One thing you should know I decided, because it changes his
world: **hunger and thirst will be slowed to match the world's own clock
(roughly 6–15× slower), if that version wins its race.** Lives get longer and
cost more compute, but he will live through nights and have time to learn. It
replaces an October hold on world edits that fired because nobody answered. If
you disagree, one sentence in `docs/ORDERS.md` under INBOX undoes it.
