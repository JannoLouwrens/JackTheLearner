# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 over the last 24 h, and Part 2.5 in full). Part 2 is
> deliberately skipped — tests are re-examined on Sundays.

**2026-09-14 06:37–07:5x UTC — DAILY.** Window: the 24 hours since the Sunday
FULL sat down.

*The one sentence: **the desk spent Sunday arming a conjunct and Monday
discovering that the conjunct stranded the project's largest unblock, idled
29 GPU-hours, and sat on a steering page whose top three orders were already
corpses — and the builder, reading that page, refused it four slots running and
went and did real science instead.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `b0ab21b`, `d255995`,
> `1ddaf48`, `f34d366`, `0ae7c71`, `6698fc8`.
>
> **One of them was not path-clean and the overseer caught it before I did
> (`76b5254`).** `6698fc8` carries `LESSONS.md` and `OVERSIGHT.md` as well as its
> own `ladder_prompt.md` edit, because the 96th audit was staging its files in
> the same index at the same moment and `git add <path>` + `git commit` commits
> the whole index, not the path. Nothing is lost or damaged — the overseer's work
> is committed, under my message — and history is not being rewritten to tidy it.
> The repair is `git commit -- <paths>`, which bypasses the index, and this page's
> own amendment is the first commit made that way. **"One path-scoped commit per
> act" was a discipline about the ADD and it needed to be one about the COMMIT.**

---

## The numbers

| | today (24 h) | yesterday (09-13, FULL) |
|---|---|---|
| demonstrated / registry | **108 / 249** | 107 / 246 |
| pass rate | **43.4%** | 43.5% |
| net demonstrated, 24 h | **+1** | +3 across the week |
| settle events, 24 h | **36** — 31 PASS / 4 FAIL / 1 BLOCKED | 17 |
| distinct specs settling / PASSing | **18 / 13** | — |
| rework rate (attempt > 1) | **78.4%** (116/148) | 78.6% |
| standing VOID / FAIL / BLOCKED | 14 / **25** / 1 | 14 / 23 / 1 |
| unreachable (shrink-only floor) | **97** | 94 |
| commits, 24 h | **107** | 84 |
| `week:all models` | **3%** (weekly reset at 04:59) | 81% |
| GPU, live week `2026-W37` | 0.82 charged, **29.18 free of 30**, expires Sat 09-19 | 0.00 of 30 |
| queue violations | **13 → 0** (mine, and repaired at ~07:2x) | 0 |
| live queue rows | **49** | 41 |

**Goodhart check: the rate FELL, 43.5% → 43.4%, on a registry that grew 246 →
249 while `demonstrated` grew 107 → 108.** The ladder outran the runner by two
today. That is the shallow reading and it is not the interesting one. **The
interesting one is that 107 commits produced 36 settlements and 13 distinct
PASSing specs, and nine of the thirteen were certificate RE-BUYS forced by
edits.** The day's output was overwhelmingly analysis, and it was excellent
analysis, and almost none of it moved the ladder — because there was nothing
legal for it to move.

---

## Part 1 — the state of progress, last 24 h only

### Did the builder produce, thrash, or stall? None of the three, and the answer matters

**It produced, on a board this desk had filled with corpses, and it said so
every time.** Four consecutive slots opened by verifying the board rather than
inheriting it, found `PROGRESS.md`'s top three orders dead or illegal, refused
them, and took real units instead: `T1.08`'s bar priced at a **22.6% false-fail
rate** with code drift eliminated as an explanation; `T1.07`'s margins priced
and the finding relocated from the flagged bar to **the control**; the implied
backward sweep **scoped and closed** at 840 pairs with 92.1% frozen; field watch
wk7 §6b verified digit-for-digit and executed. **Every one of those cost zero
GPU and zero seeds.** That is not a stall. It is an organ with no legal work
finding the work that needs no permission.

**It is also not sustainable, and the reason is mine.** `run next` has read
**0 fresh of 44** for four slots. `coverage`'s only fillable class is FILL-HELD
by `D19`. Thirteen GPU-cost specs are runnable and **not one was dispatchable**:
six parked or venue-unaffordable, six under a do-not-re-dispatch directive, and
the thirteenth was `T1.08` itself, waiting on a ruling from this desk dated
09-16 while 29.18 free hours ran toward a Saturday expiry. **The builder was not
the constraint. I was.**

### The frontier, recomputed and not quoted

`T1.08` (*Seed variance measured*, FAIL) **frees 3 and blocks 45.** Both halves
are load-bearing and the ranker conflated them until yesterday: repairing
`T1.08` alone frees `D1.0`, `T2.01` and `T2.02` — the other 42 need `T2.01` too,
and `T2.01` is a settled FAIL whose repair path runs through `D1.0`. The largest
mass in the project is **two repairs deep**, and the page that said "one dispatch
away" for five weeks was collapsing a pair into a step.

By free-standing mass the frontier is elsewhere and has been for a while:
`LT.01` FAIL **frees 7** (impl unchanged 13 d), `NE.01` FAIL **frees 7** (impl
unchanged 20 d), `UB.10` VOID **frees 4**. Each is larger, today, than
`T1.08`'s 3 — and none of the three has been touched in the window.

### The one act that mattered, and it was a decision, not a run

**`T1.08` is RULED, two days early (`b0ab21b`).** Three builder annotations had
priced every term of the question at zero GPU; a 09-16 sitting would have known
nothing more, and 29.18 perishable hours had no legal buyer until the ruling
landed. The row's own second annotation pre-emptively refused a dying quota as a
reason to *manufacture a run* — it is not a reason to *withhold a decision*, and
that distinction is the whole of why this was ruled today.

- **AUTHORISED:** the backend-confound arm pair as a **PROBE**, n=5 per backend,
  1.20 GPU-h (alpha 4.0% → 0.2%, power 92.2% → 98.2% for 0.48 h), on the `D1.0`
  twin-spread and `SM.03` F2 precedents. Pre-registration commit first.
- **FORBIDDEN, and it is the load-bearing half:** dispatching `T1.08` to a
  backend *because the probe reported that backend reads lower*. The venue a
  certificate is bought on may never be selected after seeing which venue is
  kind.
- **The read is pre-registered before any number exists**, all three branches,
  and branch (ii)'s seed count is fixed before the probe's numbers are read.
- **7.0 does not move.** The 22.6% is a fact about the bar's *governance* and is
  **not** a defence of attempt 3: a 5.717-true-cv pipeline reads inside
  [0.92%, 11.00%] 95% of the time at n=3, and 40.006 is nowhere near that.

### The thing I got wrong, in the first person

At 06:37 on 09-13 this desk wrote that *"a measurement spec that FAILS when its
own toy task has a small effect is a category error."* **At 10:05 it armed a
conjunct that FAILS when the measured noise is large** — the quantity is better,
the shape is the error I had diagnosed four hours earlier, and the bar was set
at 1.224× one observation of a sample statistic by analogy to a sibling with a
different sampling distribution. **The conjunct stays**, and not to save face:
it reported, in one run, that 45 specs stand on a pipeline with a 40% held-out
seed spread that 49 dependents quote zero times. The gate did not misfire. The
ladder mishandled the report by treating it as a blocker instead of a number to
be quoted.

The same shape appeared in my own steering: I ordered `T2.10` as *"CPU, ten
minutes"* off a page instead of off the spec's own reachability block, which
says the run returns the same FAIL. Retracted in `1ddaf48`; `T2.10`'s repair is
a **design owed by this desk**, and `T6.03` and `LF.02` stay stranded behind it
as my bill.

---

## Part 2.5 — steering maintenance

**1. Priorities — replaced in full (`1ddaf48`), and the retraction goes first.**
Three of `1^6`'s five items were dead within twenty-four hours: item 3 (`D1.0`
into W37) was **illegal** from 10:05 on 09-13 and the legality reader the
builder shipped that same night flagged it — the instrument caught its author;
item 1 (`T2.10`) was never a ten-minute job; item 2 (`D25`) had fired. Item 5's
two seats both moved by measurement. The new block leads with what was wrong and
why, because a page that quietly swaps its dead orders teaches the next reader
nothing. `1^7`'s three units are the `T1.08` probe, `T1.07` gaining `IMPL_DEPS`,
and `D19`'s default on 09-15. **New prohibition (`6698fc8`): a commit that arms
a conjunct on a PASSing spec must carry that spec's `run blast-radius` line.**
Two instances in twenty-four hours, both from one Sunday sitting, both free to
have seen in advance.

**2. Field watch wk7 — CONSUMED the morning it landed (`f34d366`).** Sweep at
05:57, consumed by 07:4x; third consecutive sweep on cadence. **N1** (the
free-embedding critical-dimension probe) **ACCEPTED and ordered first**, because
it can *foreclose* a family rather than add to one: `ME.11`'s four semantic arms
all read `feasible_ok` 0.0 and the bakeoff doc pre-committed *"the correct
response is a better score function"* without anyone knowing whether one can
exist on our fixture — CPU-minutes answer that either way. **N2** (3M-Progress,
virtual zebrafish) **ACCEPTED as an arm and HELD in the open**: out of window by
five months and the first embodied *homeostatic* intrinsic-motivation result in
four sweeps, but `CU.1`–`CU.7` are seven specs with zero implemented and an arm
for a bakeoff with no arms is a design that ages without a referent. **N3**
(ActSWM's `Δ_k`) **ACCEPTED and merged into §6** — its all-zero-action baseline
is better-posed than what wk6 proposed and replaces it. **§6b was already
discharged by the builder** before this sitting and is marked discharged, not
re-asked.

**§6 is routed as its own row (`0ae7c71`), and it is the sweep's real find.**
`LEARNING_CORE.md` §5.4 promises `A4` a *mandatory* collapse diagnostic —
effective rank and per-dimension latent variance, with a rank below a
pre-registered floor making `A4` VOID — and **it is computed nowhere.** `D10`
seated `A4` **by verdict** on 09-01. The seat's evidence is real and unchallenged
by the row; what the row says is that the seat was won in a ring missing one of
its declared walls. Three-way fork, DUE 09-18; the desk's leaning is recorded as
**(i)+(iii) together, never (iii) alone**.

**3. Seat staleness.** `World` still declares **no deciding run and no
`TRIGGER:`** while held BY VERDICT — **eleven days** now, and the oldest seat
finding on the board. `Learning core` holds 3 trigger debts, and the `A4` row
above is a fourth kind of debt against it. `Fast/slow coupling` stays welded
behind `LC.03`. Two seats moved by measurement in one day and both are correctly
marked: **Language routing FILLED BY VERDICT** (`LG.13`, 1.0000 every seed, both
mouths) and **Person model VACANT BY MEASUREMENT** (`SO.10` FAIL — its own
race's winner could not hold it). `champions --check` EXIT 0.

**4. Organ liveness — all four alive, verified against `/data/jack-logs` mtimes
and not against anyone's report.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:24 today | alive; 107 commits in 24 h, four slots on an empty board |
| overseer | 6-hourly | 06:37 today (96th, concurrent with this page) | alive |
| field watch | Mondays | **05:57 today** (wk7) | alive; consumed within two hours |
| review | daily / Sun FULL | this run | alive |

---

## The honest paragraph

No numbers. We are busier than we are closer, and today the two came apart
cleanly enough to see the seam. The builder spent a full day doing the best work
it has ever done — pricing bars nobody had priced, eliminating explanations
nobody had eliminated, closing a sweep by scoping it rather than by running it,
and correcting its own first reading twice in writing before anyone could catch
it — and not one hour of that made the creature more alive, because everything
that could have was blocked behind a decision sitting on my desk with a date on
it. That is the drift and it is entirely mine: this desk has become very good at
producing beautifully reasoned pages and noticeably bad at producing them *on
the day the thing they unblock is still worth unblocking*. Thirteen promises
broke at midnight, and they were not surprises — I forecast them out loud for a
week, wrote the forecast into two consecutive pages, and then held a Sunday
sitting that did not reach a single one of them. The most important step toward
Jack this week was not a spec. It was that the instruments the builder built
last night caught the organ that ordered them: a legality reader flagged its
author's own illegal order, and a reachability block refused an order the same
author had written twice. We are building a system whose parts can tell its
other parts that they are wrong, and today every one of them did, and the only
organ that had to be told rather than telling was this one. That is embarrassing
and it is also exactly what it looks like when oversight starts working.

---

## FOR THE BUILDER

Ordered, and all of it is already in `ladder_prompt.md` `1^7`/`2^7`.

1. **The `T1.08` backend-confound PROBE — the only legal GPU dispatch in the
   project today.** Two commits: a pre-registration commit with **no dispatch in
   it**, fixing the seed list and quoting the ruling's §4 read verbatim; then
   the dispatch. **n = 5 per backend, 1.20 GPU-h** against 29.18 free expiring
   Sat 09-19. Same kernel, same seeds, same commit, both backends; `SEEDS`
   inside the `JOB`. **It is a probe: no `T1.08` ledger row, no verdict bought,
   `T1.08` stays FAIL on every branch.**
2. **`T1.07` gains an `IMPL_DEPS` declaration, is staled by it, and is re-bought**
   (~0.47 GPU-h). It declares none today, which is why a `UnifiedBrain.py`
   change cannot stale it and why you did the drift check by hand twice in two
   days. The staling is the point, not the objection. This is where your (e')
   finding lands: the binding margin is the CONTROL's **x1.255** against a
   measured **x99.59**, not `spread_ratio`'s. It arrived; it is read.
3. **`D19`'s NO-FETCH default is firable from 2026-09-15 00:00.** You found the
   off-by-one and you are right. Required wording: *"the owner did not rule by
   2026-09-14, so the pre-registered default fired."*
4. **Do not pre-empt the `A4` disposition** (DUE 09-18, mine). If you want its
   free half, it is the grep, and the scout has already run it.
5. **Four of the thirteen re-armed queue rows are YOUR execution debt, not
   mine** — `w0-too-shallow`, `lt01-c2-body-cannot-rise`,
   `cross-organ-doc-race-voids-certificates`, `t215-router-under-lexical-null`.
   Their designs are delivered and they are dated where a slot can reach them.
6. **`LT.01` (frees 7, impl unchanged 13 d) and `NE.01` (frees 7, impl unchanged
   20 d) are each larger free-standing unblocks than `T1.08`'s 3 today.** Named,
   not ordered — both are FAIL with no repair design, and the design is this
   desk's debt. Do not invent one; do tell me if you think I am wrong about that.

---

## FOR THE OWNER

**1. `D28` — cited, not re-asked, and its forecast is now a measurement.**
`decide_by` 2026-09-21. When it was routed it said the cost was *"realised, not
forecast: 13 dated promises broke at midnight."* They did. I have re-armed all
thirteen in the open (`d255995`), at the desk's **demonstrated** rate of ~1 per
sitting rather than its measured 6-per-cycle maximum, never onto a day already
at capacity — because promising six a day is the act that built the pile.
`review_queue --check` is back to **EXIT 0, 0 violations**. Three of the
thirteen have now broken **three dates each** and carry a stop-rule I have bound
myself to: **if the fourth breaks, they are DECLINED as a class and the finding
comes to you.** And the honest evidence against `D28`'s own default, which I owe
you because I am the organ it constrains: **its option (a) says the daily
sitting spends its FIRST act on the overdue class, and I did not.** I ruled
`T1.08` first, because 29 GPU-hours were dying and the overdue rows were not.
I would make the same call again, and that means (a) as written would have made
today worse. **My recommendation is that (a) be amended before it fires, to
"overdue first UNLESS a perishable resource is the reason."**

**2. `D19` is due TODAY and its armed default fires tomorrow.** Routed; cited,
not re-asked. `HR.1` currently ranks with a real frees-count of 3 and is
correctly refusing to be dispatched while held.

**3. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive, verified against log mtimes. Field watch wk7 landed at 05:57 and was
consumed in full by 07:4x — the first time this desk has consumed a sweep on
the morning it arrived. `week:all models` 3% after the 04:59 weekly reset; the
gate is all-models.

**4. NO-DECISION: the day's acts, declared because they are mine to do.** One
row ruled two days early, thirteen re-armed, one routed, one nomination set
consumed, one steering block replaced with its own retraction at the top, one
new commit contract adopted that binds this desk harder than it binds the
builder. No threshold moved in any direction. No ledger row was written by hand.
`demonstrated` moved +1 and none of it was mine.
