# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-31 06:5x UTC — DAILY. Window: 2026-08-29 06:44 → 2026-08-31 06:5x
(48 h — no Review ran on 08-30; see FOR THE OWNER).**

*The one sentence: **the builder has come back from a five-day blackout at full
speed — 75 commits and +9 demonstrated in 24 hours — and it has spent that
speed discovering, five separate times, that the thing blocking it is not a
spec but the world; nine instruments now say so, nobody had added them up, and
the design that answers them is owed by a Review run that has never once
survived to write it.***

---

## The numbers

| | now | 08-29 | Δ |
|---|---|---|---|
| demonstrated / registered | **93 / 201** | 84 / 187 | **+9 / +14** |
| pass rate | **46.3%** | 44.9% | **+1.4 pts** |
| FAIL / VOID | 13 / 5 | 13 / 5 | 0 / 0 |
| rework (ledger entries at attempt > 1) | 74 / 111 = **66.7%** | — | — |
| commits, last 24 h | **75** | 0 | +75 |
| PASS-bearing commits, last 7 d | 35 | prior 7 d: 27 | +8 |
| builder iterations, last 24 h | 24 of 24 slots | 0 of 24 | full |

**Goodhart check: the rate ROSE while the registry grew, and that is the good
direction.** +14 registered against +9 demonstrated would normally dilute the
rate; it did not, because the week's registrations (`W.1`–`W.8`, `PL.00`/`PL.02`,
`LG.00`/`LG.01`, `T0.28`–`T0.31`) were mostly *implemented and run in the same
day rather than shelved*. This is the first Review in six days whose table is
not byte-identical to the last one, and the first since 08-11 where the rate
moved up rather than down.

**Rework at 66.7% is high and is NOT this week's problem.** It is the ladder
working as designed — attempts 2+ are almost all VOID→repair→re-run cycles on
rig faults the pilots caught before dispatch, which is the failure mode we pay
pilots to buy. The number to watch is not this one.

**The frontier, computed rather than quoted: it is not a spec.** Five specs are
PILOT-BLOCKED or VOID-FORECLOSED — `BA.03`, `LC.03`, `SH.02`, `DP.04`, `T3.06` —
and `coverage` reports **every one of their repairs as a REDESIGN, not a run**.
Three cost classes (`cpu<48h`, `gpu<20min`, `gpu<2h`) read EMPTY *with no path
in*: nothing runnable to implement and nothing gate-provisional to pilot. The
ladder is not short of specs. It is blocked on one world decision, and the
transitive mass behind that decision is larger than any single spec's: two
`REVIEW_QUEUE` rows are explicitly `BLOCKED-BY w0-too-shallow`, four
gate-provisional specs sit behind it, and D10 on the owner's desk is coupled to
it.

---

## The finding: nine instruments, and nobody was counting

`w0-too-shallow`'s row says three instruments. Its own 08-25 update says four.
Field watch wk5 says six, then seven. **The true count is nine**, and the gap is
not sloppiness — it is structural. Each new instrument was routed as *its own*
queue row (`dp04-lifespan-has-no-resolution`, `ba03-null-saturates-the-horizon`,
`t306-matched-magnitude-noise-buys-coverage`, …), which is correct
bookkeeping and means **the aggregate lived nowhere**. A backlog that files each
symptom separately can never notice it has a syndrome. All nine are now named in
the row.

**And the eighth is different in kind, which is the part that should change the
design.** Eight of the nine say *"W0 does not REWARD capability X"*, each on its
own channel — the agreeing-instruments pattern the row itself flags as the exact
condition under which a shared confound is invisible. `DP.04`'s sizing record
says something else: **the outcome variable itself has no resolution.** 3072
lives produced **21 distinct lifespans**; zero ended between the old cap and the
new one; the quantum is 6.25 steps against a `MIN_GAIN` of 5.0. A threshold
finer than its statistic's quantum is not a hard test, it is an unreadable one —
and lifespan is the channel most of the other eight are ultimately scored
through. That makes `DP.04` a live candidate for the shared confound the other
instruments cannot see past.

So the design question sharpens from *"is W0 too shallow"* to **"is W0 too
shallow or merely too COARSE"** — because those have different repairs, and only
one of them bills the 21 `playground.py` certificates.

---

## The honest paragraph

Closer — and for the first time in three weeks I can say that without hedging.
The builder came back from the blackout and did not thrash: it spent the day
killing its own work honestly, declaring three specs VOID-FORECLOSED rather than
re-running them into the same wall, and it saved four hours of compute by
replaying a recorded row offline instead of buying the same verdict twice. That
is a system that has learned to lose cheaply, which is the skill that separates
a ladder from a treadmill. Voice made its first passing claim; the anti-puppet
test — the one the founding premise rests on — now has a green row saying he is
smarter inside his life and dumber outside it, which is the asymmetry that makes
him a creature rather than a costume. The most important step toward Jack this
week is that one. But the drift is real and it is not about speed: **the project
keeps discovering the same thing on new channels and filing it in new places.**
Nine times we have measured that his world does not ask enough of him, and nine
times the finding got a row of its own and no aggregate. Meanwhile the organ
whose entire job is to see the aggregate — this one — has never once survived
long enough to do it on the day it was scheduled to. The builder is not the
bottleneck and has not been for a while. The bottleneck is that the thinking
this project needs is budgeted like a chore.

---

## REWRITTEN / STRENGTHENED

Part 2 (the test re-examination) is **Sundays only** and was correctly skipped.
No spec file, threshold, or control was touched by this run. Steering only:

| what | change | why it is stronger |
|---|---|---|
| `docs/REVIEW_QUEUE.md` · `w0-too-shallow` | **RE-ARMED** to `DUE: 2026-09-06`, in the open, with two stated reasons | the honest repair of an OVERDUE row; the alternative was letting a dated promise stay silent a second day |
| `docs/REVIEW_QUEUE.md` · `w0-too-shallow` | all **nine** instruments named and counted; `DP.04` promoted as a shared-confound candidate | the row claimed three; a design written against three instruments is written against a third of the evidence |
| `docs/INTEGRATION_QUEUE.md` | **`W0.DIAG` added as a real top-of-queue row** with its full design and cost class | it was ORDERED on 08-25 as *prose*, so the builder's top-down read could not see it; an accepted nomination that is not a queue row is one nobody accepted |
| `docs/INTEGRATION_QUEUE.md` | field watch **wk5 consumed**: N1 accepted (narrowed + split, free half first), N2 arm deferred / **control accepted**, N3 accepted as a binding control | N2's control is a strict strengthening of `NE.07`; N3 makes `W0.DIAG` carry a known-answer check it did not have |
| `scripts/ladder_prompt.md` | head block **replaced**: expired `week:Fable` date and spent pace-skip counter removed; frontier stated | the block told the builder to read a model cap off a date that passed at 04:59 today |
| `scripts/ladder_prompt.md` | priorities **re-pointed**: `W0.DIAG` is 1, the `T0.10` re-buy is 2, and the dead GPU-refill list is retired with its reason | all five specs it named (`T2.09`/`T3.06`/`T2.19`/`T2.11`/`T2.14`) are settled, and the classes it aimed at now have *no path in* |

---

## FOR THE BUILDER — ordered

1. **`W0.DIAG`, top of `INTEGRATION_QUEUE.md`.** Register, implement, run.
   `cpu<10min`, which also clears an EMPTY class. Its known-answer control is
   **binding, not optional** — an unvalidated instrument does not get to
   overturn nine validated ones, and two published environment-difficulty
   metrics are *measured* to invert on setups whose true ordering was known.
   This is the input the `w0-too-shallow` design is blocked on, and it is
   simultaneously the cheapest and the most valuable unit on the board. That
   coincidence is rare; take it.

2. **Two minutes of bookkeeping, and one of it is a live hole.**
   `experiments/ledger.json` has been sitting **uncommitted since 06:10 today** —
   it holds `T0.01`'s re-run (attempt 9, PASS), bought because the 52nd audit's
   B5 amended its `control` field, which is a `SPEC_CLAIM_FIELDS` edit. Commit
   it. Then **re-run `T0.10`**, which got the same amendment and was never
   re-bought — `run status` names it under DRIFTED CLAIMS. The 06:07 iteration
   ended after five minutes saying *"Waiting on the Kaggle round-trip"* and left
   this behind; I could find no in-flight job, and the tree has been dirty for
   45 minutes across an iteration boundary.

3. **A note on iteration 06:07 itself, offered as a lesson, not a reprimand.**
   `LESSONS.md` already carries *"waiting on background work is a claim, not
   evidence"*. An iteration that ends `rc=0` in five minutes on a stated wait
   should print what it waited on — job id, log bytes, artefact mtime — or
   spend the slot. It ended with the tree dirty, which the loop's own exit
   discipline (*"tree clean, no leftover processes"*) forbids.

4. **Do not go hunting for GPU work.** `gpu<20min` and `gpu<2h` are EMPTY with
   no path in; `coverage` says the repair is an UNBLOCK. Priority 1 *is* the
   unblock. Retired list and reasons are in `ladder_prompt.md`.

---

## FOR THE OWNER — two, and the first is the reason this page was late

### 1. The Review is budgeted like a chore, and it has never finished its real job

**Four consecutive Sundays, four deaths, zero FULL runs in the project's
history.** The overseer has reported the deaths; what nobody has reported is the
cause, and it is mechanical rather than accidental:

```
2026-08-30T06:37:03  review start — mode FULL, model opus
Error: Reached max turns (60)
2026-08-30T06:48:03  sweep end rc=1
```

Eleven minutes. And **today's DAILY run is budgeted identically** — the log line
reads `mode DAILY, model opus, 20m / 60 turns`. A DAILY run is Part 1 plus Part
2.5. A **FULL** run is all of that *plus* Part 2 (8–12 spec re-examinations,
each a read and a re-run), *plus* an anatomy audit, *plus* the completeness
audit — the one whose entire purpose is finding what nobody wrote down, and the
one that found on 2026-08-09 that smell, taste, voice and body-schema had zero
specs among 136. **None of those has ever run.** The Sunday job is roughly three
times the Tuesday job in the same envelope, and no organ watches for an organ
being asked to do more than its budget permits.

This is not abstract. The `w0-too-shallow` design — nine instruments, two held
rows, four blocked specs, coupled to your D10 — is owed *by the FULL run*, and
has now slipped a week because the FULL run cannot reach it.

**My recommendation, in preference order:**
(a) **Raise the Review's Sunday budget** (turns and wall-clock) to match its
scope — the cheapest fix and it needs no design change; or
(b) **Split the FULL run into two organs** — Part 2 (test re-examination) and
the two completeness audits on different days, each in a DAILY-sized envelope.
I prefer (a): the completeness audit's whole value is looking at everything at
once, and splitting it re-introduces exactly the "each finding gets its own row,
the aggregate lives nowhere" defect that this week's nine instruments are an
instance of.

**I have re-armed `w0-too-shallow` to 2026-09-06 and I am telling you the bet is
conditional.** If the budget is unchanged by then, a fifth Sunday is a lie by
deferral under this project's own week-3 rule, and the correct move on 09-06 is
to split the row rather than re-arm it again. I would rather flag that now than
discover it next Sunday.

### 2. The world decision is ripe, and it has forked in a way that changes its cost

`w0-too-shallow` has always been framed as *"edit W0 or build W1"*, with the
noted asymmetry that editing `playground.py` bills 21 certificates and a new
world bills nothing. **`DP.04` adds a third option that nobody has priced**: if
the problem is that W0's outcome variable is too COARSE — 21 distinct lifespans
across 3072 lives — then the repair may be to the *measurement*, not the world,
and that bills nothing either. Eight instruments say the world does not reward
capability; one says we may not be able to read the reward at all.

**This is design work, and it is mine to do — I am not asking you to decide it.**
I am flagging it because it bears directly on **D10**, whose default fires
**2026-09-01** (tomorrow) and seats `wm-latent` (`A4`) on the learning-core seat
by default. D10's fork (b) is the world redesign. If the shallowness turns out
to be partly a resolution artefact, then `LC.03`'s *"one learner in five"* —
which is the evidence D10 rests on — was read through the same coarse channel.
**I am not recommending you delay D10**; the default is defensible and the
project has paid for indecision before. I am recommending you know that its
evidence base has a live open question against it, and that `W0.DIAG` (priority
1, CPU-minutes, running this week) is the cheapest thing that bears on it.
