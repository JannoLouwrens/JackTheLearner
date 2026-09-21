# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** — Part 1 (last 24 h) and Part 2.5. Part 2 (the test
> re-examination) is Sunday's and is deliberately not run today.

**2026-09-21 06:37–0x:xx UTC — DAILY.** Window: the 24 hours since
2026-09-20 06:37.

*The one sentence: **the builder has not run a single iteration since 07:07
yesterday — it is not paced, it is dead, `execve` has been refusing it for
twenty-three hours, and the commit that killed it is mine, made by this desk at
06:44 yesterday morning while it was writing that the builder was healthy.***

> This page is the RECEIPT for commits that already exist, every one of them
> made before this page was written: `b8e807b` (the launcher resurrection —
> `ladder_prompt.md` 140331 → 85391 bytes, exec verified), `a7233b2` (`1^10`
> gains `ITEM 0`), `3d06fc2` (both week-8 field-watch findings ROUTED),
> `0056ea5` (week-8 nominations disposed), `b7bbbc8` (`ITEM 0` corrected
> in-sitting after the overseer fired `D27` out from under it).

---

## THE OUTAGE — measured, and the cause is this desk

**Nineteen consecutive slots, zero iterations, `109/253` unchanged throughout.**

```
2026-09-20T06:10:38  iteration end rc=0     <- the last iteration that ever ran
2026-09-20T07:07:24  rc=126   nice: Argument list too long
2026-09-20T08:07 .. 11:07     rc=126  (four more, same)
2026-09-20T12:07 .. 09-21T05:07   18 consecutive PACING skips  <- the mask
2026-09-21T06:07:22  rc=126   (same)
```

**The mechanism, probed on this box against `/bin/true` so nothing was spent.**
`ladder_loop.sh:270` does `PROMPT=$(cat scripts/ladder_prompt.md)` and line 278
passes it as a **single argv** to `claude -p "$PROMPT"`. Linux caps one
argument at `MAX_ARG_STRLEN` = 32 pages = **131072 bytes**. Measured:
131000 OK, 131071 E2BIG, 131072 E2BIG. `git cat-file -s` dates the crossing to
the hour:

| commit | when | bytes | |
|---|---|---|---|
| `c124fad` | 09-19 06:39 | 129855 | under by 1217 |
| **`af21fe0`** | **09-20 06:44** | **139002** | **OVER — mine, the `1^10`/`2^10` rewrite** |
| `87c8f04` | 09-20 06:48 | 140331 | OVER — mine, the item-5 correction |

The 06:07 slot on 09-20 read the file at 129855 and ended normally at 06:10.
The 07:07 slot was the first to read it after 06:44 and is the first `rc=126`.
No gap in the chain, no other candidate.

**THE REPAIR, and the habit that caused it (`b8e807b`).** This file carried
every superseded priority block under the banner *"this file never deletes
history"*. **That is not law** — it appears in no `LESSONS.md`, `SYSTEM.md` or
`GOAL.md` rule — and git holds those blocks perfectly, which is what history is
for. Excised `1^9`, `1^8`, `1^7`, `1^6`, `1''''`, `1'''`, `1''`:
**140331 → 85391 bytes**, now 88072 with `ITEM 0`.

**Nothing operative was deleted, and the hazard was real.** `2^10` did **not**
restate the older prohibitions — it said *"`2^9`, `2^8` and `2^7` below are
carried IN FULL"*, incorporation **by reference**. Deleting them blind would
have silently voided live prohibitions, which is the exact class of damage this
desk is forbidden to do. So `2^10` now **restates the complete set**: its own
additions, then `2^9`, `2^8`, `2^7` VERBATIM under named attribution. All 18
live prohibition bullets verified present by grep. `3''` and `1'`/`2'`/`3'`
untouched — `3''` cites them by name, so they stay.

**Verified, not assumed:** `nice -n 19 /bin/true "$(cat scripts/ladder_prompt.md)"`
execs at HEAD and did not at `87c8f04`.

**AND THE PART THAT IS ABOUT THIS DESK RATHER THAN THE KERNEL.** Yesterday's
FULL reported *"Ladder 06:10 — alive, 7 of 7 hourly slots ran"*. That was true
when written at 06:37 and **false thirty minutes later by my own hand**. The
106th audit, reading the same morning, recorded *"24 slots started, 24 ran"* —
also true when read. **Both organs certified the builder healthy in the hour
they broke it and in the hour it died, and neither was wrong at the time.** A
liveness reading has a shelf life, and nothing in this project stamps one.

---

## Part 1 — the last 24 hours

**Velocity: nothing.** `109/253 demonstrated, 43.1%`, unchanged. PASS 109,
FAIL 30, VOID 14, BLOCKED 1. **0 iterations, 0 spec attempts, 0 ledger
events, 0 verdicts.** Against a demonstrated 7-of-7 slots on the morning it
died. There is no thrash to diagnose and no stall to interpret: there was no
builder.

**The queue moved, and only the clock moved it.** `review_queue_violations`
**12 → 17 (+5), entirely CLOCK** — `review_queue_violation_forms` is
`{'OVERDUE': 17}` and the instrument says so itself: *"the calendar reached a
date; a real event with a real owner, and no commit is to blame for the rise."*
51 live rows, drain **UNBOUNDED**, arrivals 1.29/cycle against disposals
1.00/cycle. `review_queue_net_arrivals` fell 11 → 2, and that movement is the
sliding trailing window only — no act, nothing to investigate.

**Every other ratchet is unchanged and at floor**: `unreachable` 98 (at the
baseline I raised yesterday, with its restoration obligation intact and
undischarged because the builder could not run the `T2.06` re-buy),
`claim_dead` 4, `commitments_uncovered` 0, `fail_unowned` 0,
`goal_unrunnable` 7, `champions_trigger_debt` 3, `champions_unwinnable` 4,
`gpu_hours_no_verdict` 48.42 h. **`decisions --check` exits 0**, ratchet ok
(0/10, 0/3, 0/0, 0/0, 0/0).

**The frontier is exactly where I left it, and that is the point.** The
priority block `1^10` — `BA.03` (c), `T3.06` (b)+(a), the `T2.06` GPU re-buy,
the `T1.08` trigger declaration — **has never been read by any builder slot.**
It was committed at 06:44 and broke the launcher in the same commit. This is
**unread work, not stale work**, and `ITEM 0` now says so in the file, because
a builder that wakes to a day-old board and assumes a week of refusals will
re-derive a frontier that has not moved.

---

## Part 2.5 — steering maintenance

**ORGAN LIVENESS.** Ladder **DEAD 23 h**, resurrected this sitting — see above.
Overseer **06:37 today, alive**, running concurrently and it found the outage
independently (107th audit, `D34`). Field watch **09-21 05:54, alive and on
cadence** — week 8 landed to the day, fourth consecutive sweep on its intended
Monday. Review — this sitting.

**`ladder_prompt.md` — REWRITTEN, then CORRECTED IN-SITTING (`b8e807b`,
`a7233b2`, `b7bbbc8`).** `ITEM 0` prepended to `1^10`: the outage explained as
a dead launcher rather than a refusal, so the builder does not re-derive a board
that never moved. **Then corrected within the hour**: I wrote `ITEM 0` telling
the builder to fire `D27`'s overdue default; by 06:5x the overseer had fired it
itself and stamped the entry RESOLVED. The stamp is **uncommitted** in the
overseer's working tree, so the builder may legitimately read either state —
`ITEM 0` now says **CHECK, do not double-fire**, and pins the half that does not
go away: `D27`'s default is *(i) BUILD THE SCREEN, REPORTING ONLY*, and a
resolution stamp is not a screen. **That is the second consecutive sitting in
which a steering page of mine was falsified inside the hour by another organ**,
and the remedy is the 06:37 collision's standing one: re-read every instrument
immediately before committing.

**`FIELD_WATCH.md` — CONSUMED IN FULL, and the instrument's green was FALSE.**
`run status` prints `week 8: 2 finding section(s); 0 cited, 2 quoted, **0
UNROUTED-FIELD-FINDING**`. The scout's own §6b **measures all five of those
routings as spurious** and states in terms that *"the true state of both
findings below is UNROUTED"*. It is right. A desk reading the counter instead
of the page would have consumed nothing today. Both findings routed
(`3d06fc2`):

- **`lc03-five-controls-never-switch-off-the-term-a4-is-named-for`** (DUE 09-28,
  mine). Five `LC.03` controls, none isolates the `latent_pred` objective the
  seat is NAMED for; the seat's `t = 4.64` / `t = 4.00` are equally consistent
  with 17.3% of the arm being dead weight. Dated **behind**
  `a4-…-computed-nowhere` deliberately.
- **`fieldwatch-quotation-channel-is-0-for-5`** (DUE 09-23, the builder's). Not
  a design question — the scout named three candidate closures.

Three nominations disposed (`0056ea5`): **N1 ADMITTED design-only** behind a
sequencing bar (three open `A4`-seat questions all want the same 14.40 core-h
with no weights on disk — they are marginal on each other, and buying them
separately pays up to three times for one run); **N2 REJECTED as an arm** with
its one line (it nominates no arm; it is a measured negative about the family
`t402` already names, and `t402` falls due tomorrow); **N3 ADMITTED as a
conditional correction** to a week-5 nomination this desk itself accepted.

**SEAT STALENESS.** `champions_trigger_debt = 3` (unchanged since 09-03),
`champions_unwinnable = 4` (at floor since 09-13). Neither moved; both remain
this desk's standing debt and neither is due today. **The new `lc03` row is a
seat finding** and belongs beside the Learning-core seat's existing
`VERDICT-IS-A-VOID` indictment — same seat, third open question.

---

## FOR THE BUILDER

**Read `scripts/ladder_prompt.md` `ITEM 0` first — it is the binding copy.**

1. **`D27` — CHECK, do not double-fire.** The overseer fired it at ~06:5x
   today. If the stamp is at HEAD, say so in the journal and move on. **Either
   way the CODE half is still yours**: `(i) BUILD THE SCREEN, REPORTING ONLY`.
2. **`D28` does NOT fire today.** `decide_by` 2026-09-21, overdue at `> 0`
   days, so earliest legal firing is 2026-09-22. Do not fold it with `D27`.
3. **`1^10` items 1–4 are UNREAD, not stale.** `BA.03` (c), `T3.06` (b) then
   (a), the `T2.06` `gpu<20min` re-buy (**and restore `UNREACHABLE_BASELINE`
   98 → 96 in that same commit, whichever way the run falls**), the `T1.08`
   trigger declaration. Do not re-derive the board as if a week had passed.
4. **A MECHANICAL GUARD ON THE STEERING FILE'S SIZE — new, and it is the honest
   repair for a rule I could only write in prose.** `b8e807b` put a 125000-byte
   ceiling into `ladder_prompt.md` **as a sentence**, which is precisely the
   failure shape this project keeps re-learning: a desk rules correctly, writes
   it down truthfully, commits it, and nothing an instrument can see has
   changed. Print `scripts/ladder_prompt.md`'s byte count in `run status`, with
   the 131072 cliff named and the current headroom in DAYS at the measured
   growth rate. **Reporting-only, unfloored** — a steering page has legitimate
   reasons to grow, and a gate here would let an instrument refuse the Review's
   own act. The judgement stays a human's; the *visibility* is an instrument's.
5. **`fieldwatch-quotation-channel-is-0-for-5` (DUE 09-23).** Measure before
   picking a closure. **The shingle rule is IMPORTED from
   `decisions.owner_asks`** so the readers cannot drift — a naive repair
   silently re-tunes the instrument that audits my own `FOR THE OWNER` section.
   Parameterise at the fieldwatch call site, or measure both channels'
   false-positive rates in the same commit. Do not change the shared rule blind.
6. **Still do not pre-empt** `A4`, `T2.10`'s repair, `SO.07`, `SO.10`,
   `T1.08`'s **pipeline** repair, `HR.1`'s fixture redesign, `UB.10`'s
   successor arm, the **world-edit window**, or the new `lc03` seat row. All
   mine. Naming candidate arms remains welcome.

---

## FOR THE OWNER

**1. NO-DECISION: liveness and the standing `D30` report, and this morning it
is not a formality.** **The dark-slot streak, counted from
`/data/jack-logs/ladder.log` and not from anyone's summary: 19 consecutive
slots with zero iterations** — five `rc=126`, eighteen `PACING`, one both —
against a builder whose cadence is hourly. That is **9.5× the 2× threshold**
`D30`'s default sets, and it is the first blackout since `D30` fired.
**Beside it, as `D30`'s default requires, the perishable price:** `2026-W38`
carries **30 free Kaggle GPU-hours, expiring Saturday 2026-09-26**, with
**0.00 h drawn**. For the first time in three weeks a **legal buyer exists** —
the `T2.06` `gpu<20min` re-buy my own Part 2 strengthening created yesterday —
**and the only organ that may spend it could not start.** W37 expired with
~27.8 h unspent for want of a buyer; W38 is on the same path for the opposite
reason, which is worse. Five days remain. Nothing here needs a ruling; the
launcher is repaired and the buyer is ordered as builder item 3.

**2. `D34` — cited, not re-asked; routed by the overseer this morning, and its
option (ii) is already EXECUTED.** `D34` asks whether to stop passing the
steering file through `argv` and feed it on **stdin** instead. Its option (ii)
— trim the file, which is the Review editing the Review's own page and needs no
ruling — I executed at 06:4x, about ten minutes before the entry ordering it
was written; the loop is alive at the next `:07`. **I support recommendation
(iii) and I want to sharpen the cost estimate in the entry**: `D34` prices a
trim at *"about five days"* from a 110000-byte target at ~3976 bytes/day. The
excision went further than a trim — 140331 → 85391 — so the runway is **~10–11
days, not five**. That is more room to verify option (i) properly and **less
excuse to fire it unverified**, which is the entry's own stated risk: if `claude
-p` does not read stdin in this lane, the builder starts with an **empty
prompt**, and a silent empty prompt is strictly worse than a loud `rc=126`.
**My recommendation: rule (i) IN but require it to land from a running builder
that verifies stdin is actually read, with the `argv` path kept as a fallback
in the same commit.** Not urgent this week; genuinely urgent before 10 days.

**3. NO-DECISION, and it is a CORRECTION to `D34` that cuts in this project's
favour.** `D34` says the outage was caught by *"luck rather than design — the
Review found this by reading the log during a sitting that had no instrument
telling it to."* **That is not right, and the distinction matters because a
capability recorded as luck does not get protected.** `D30`'s armed default —
fired 2026-09-19, two days ago — **orders this desk to count consecutive dark
slots from `ladder.log` itself, every sitting, beside the GPU-expiry
forecast.** I read that log this morning because a standing order told me to.
**`D30` was written for exactly this event and it worked on its first real
occurrence**, two days after arming. The luck was that it was armed in time;
the design was that it was armed at all. `D34`'s three blind instruments are
all real and its finding stands — but the fourth reader, the one that actually
caught it, is a **standing instruction to a desk**, and it is the cheapest
liveness instrument this project owns precisely because it does not depend on
the code being right about what "ran" means.

**4. `D33` — cited, not re-asked (`decide_by` 2026-09-23). The world-edit
design lost a fourth day, and today the reason was not my choosing.** I
recorded yesterday that `D33` is the item I would put in front of you first and
that is unchanged. Today it did not lose to other work — it lost to an outage I
caused, and the three rows queued behind it (`SH.02`'s venue repair, `BA.03`'s
re-routed (b), `w1-world-edit-window` itself) aged another day regardless of
cause. The recommendation on the entry stands verbatim.

**5. `D28` — cited, not re-asked (`decide_by` 2026-09-21, i.e. its default
fires tomorrow), with today's evidence, which is mixed and I will not round it
in my own favour.** `D28` measures this desk's drain as UNBOUNDED. **In its
favour:** this sitting disposed five field-watch items and routed two findings
against a demonstrated ~1.29/cycle, and it resurrected the builder. **Against
it:** `review_queue_violations` went **12 → 17** and every one of those five is
a promise from this desk that a calendar reached while I was doing something
else. The overseer's reclassification notice fires today and I do not object.
**One fact `D28` should be read with:** for 23 of the last 24 hours the
builder's entire capacity was zero, so the queue's arrival/disposal arithmetic
this cycle describes a system with one working organ. That makes the drain look
no better and the *dependency* look considerably worse.
