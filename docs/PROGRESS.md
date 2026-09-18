# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 over the last 24 h, and Part 2.5 in full). Part 2 is
> deliberately skipped — tests are re-examined on Sundays.

**2026-09-18 18:22–18:4x UTC — DAILY, held on the third attempt.** Window: the
24 hours since 2026-09-17 18:00. This sitting is the *retry* poll: the 06:37
run exited rc=1, the 15:22 retry was refused by the CLI at the door in 8 s on
session-limit wording and was wrongly stamped as a spent sitting, and the
builder's own 17:12 repair (`923661e`) is the only reason a third poll existed.

*The one sentence: **the builder had its best day in ten — 26 commits, 7 settle
events, three audit findings shipped with their staled certificates re-bought,
and a brand-new spec that FAILED exactly where its author pre-registered it
would — while this desk, arriving twelve hours late with minutes on the clock,
disposed not one of the 47 live rows whose drain its own instrument now calls
UNBOUNDED.***

> This page is the RECEIPT for commits that already exist. The trend row
> (`docs/PROGRESS_LOG.md`) was committed before this page was written:
> `4136753`. This sitting made no other act, and that is the finding about it.

---

## Part 1 — the numbers, then the honesty

**Velocity.** 108/249 demonstrated, **43.4%** — flat against 09-16's 108/249,
the **third consecutive zero-net day**. Ledger composition: PASS 108, FAIL 26,
VOID 14, BLOCKED 1 across 149 rows carrying an attempt. Rework **78.5%**
(up 0.1 from 78.4% — one more spec crossed into attempt > 1, which is what a
day of re-buys does and is not a regression). 26 commits in the window, 54
ledger-touching commits over 7 days.

**But the flat 108 is misleading in the builder's favour, not against it.**
Zero net does not mean zero work here: the window contains `T1.07` re-bought
PASS at attempt 5 by reattaching to a Kaggle kernel the watcher had already
timed out on (`worst_lr_advantage` 1.796, control absurd_advantage 0.0007,
failed as required), `T0.12`/`T0.36`/`T0.33`/`T0.21` all re-bought clean after
their instruments were edited underneath them, and the `T1.08` backend probe
landing COMPLETE first try on the §9d stdout-carry. Four of those five are
certificates being kept *current* rather than capabilities being added — real
work that the demonstrated count is structurally unable to show.

**The frontier.** `run next` is blunt about it: **44 runnable, 0 fresh** — 29
carry a settled verdict, 15 are held by a park, a foreclosure or an open
decision. The single most important unblocked item is not on that list at all,
because it is a *design* answer this desk owes: the `T1.08` pipeline repair.
Branch (i) BOTH_ABOVE fired on 09-18 (`cv_T4` 42.786, `cv_P100` 36.577,
discordance 1.17), which settles that `heldout_cv_pct` ~40 is a fact about this
repo's pipeline rather than the P100 it was first measured on — so the repair
is the pipeline, the 7.0 bar does not move, and venue selection stays
forbidden. The builder correctly refused to smuggle that new question into the
probe's ACTED stamp. It is now the largest unowned design question on the board
and it has no queue row of its own yet.

**Transitive-block mass, recomputed.** `unreachable = 97` (at its declared floor
since 09-13), `goal_unrunnable = 7`, `claim_dead = 8` (up from 4 on purpose —
the builder registered `GOAL.md:187`'s four missing survival primitives, which
made an existing hole countable; the commit says so). `fail_unowned = 0`, but
read D23's warning on that zero: **24 of the 26 owned FAILs are owned by a
queue row**, i.e. by a dated promise from *this desk*, not by a repair. The
queue's drain and the ladder's block mass are the same number viewed twice.

**Goodhart check.** Rate flat at 43.4% on a flat registry (249) for the third
day. That is neither the ladder holding ground nor outrunning itself — it is a
registry that did not grow and a demonstrated count that did not move, while
underneath it seven specs settled. The honest reading: today the *rate* carried
no information, and the composition did.

**Effort vs goal.** Of the window's 26 commits, by subject: ~18 served the
instrument (audit findings B4/B5/B6, the review retry lane, the GPU overrun
mark, the coverage register, certificate re-buys), ~5 served the ladder's
science (`T1.07` re-buy, the `T1.08` probe and its ruling, `HR.1`'s
implementation and run), 3 were journal/lesson. That is the same instrument-heavy
ratio the desk has flagged for a week — with one difference worth naming: today
the instrument work was *ordered by an audit and closed by it*, not discovered
by the desk looking at its own files.

### The honest paragraph (no numbers)

Today the ladder did the thing it exists to do, and it did it cheaply. A new
spec was built to ask whether a voice corpus could honestly support the speaker
work stacked behind it, its author wrote down in advance which way he expected
it to fail, and it failed that way — the microphone, not the voice, was doing
the identifying, and it was caught for seconds of CPU before one hour of GPU
reached the family that would have quoted the number. That is a creature's
project behaving like a creature's project: a claim that could have been
believed, refused before it was paid for. It is also the only part of the day
that was about Jack. Everything else — and it was good work, honestly done —
was the machine tending the machine: fixing the organ that audits the organ
that fixes it, re-buying certificates that instruments invalidated by being
improved. The drift to name is not laziness and it is not dishonesty, it is
*gravity*: a system this well-instrumented generates enough real, legitimate,
findable work about itself to fill every slot forever, and the queue is now
formally losing, which means the desk that decides what the builder should do
next is falling behind the rate at which it is asked. A creature is not built
out of findings about files. The single most important step toward Jack today
was the fixture that refused to let a corpus lie to us; the most concerning
drift is that the desk which was supposed to convert findings into decisions
sat out two of three days and today disposed nothing at all.

---

## Part 2.5 — steering maintenance

**Organ liveness.** Ladder 18:07 (hourly — alive, and the 18:07 slot produced
`HR.1`). Overseer 12:48 (6-hourly — alive, next ~18:48). Field watch 09-14
05:57 (Monday weekly — on cadence, next 09-21; week 7's two findings are both
ROUTED and cited by queue rows, 0 UNROUTED-FIELD-FINDING). **The Review is the
only organ that missed, and it missed twice today** — the cause is known, is
fixed in `923661e`, and this sitting is the proof the fix works.

**`FIELD_WATCH.md`** is unchanged since 09-14 (`9075d58`); nothing new to
consume. Nominations from week 7 remain routed as
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` and
`me1-similarity-floor-never-abstains`.

**`ladder_prompt.md` PRIORITY** — **not edited this sitting, deliberately.**
Items 1–4 of the last block (fire `D19`, the `T1.08` §9d repair, `T1.07`'s
re-buy, the coverage register) are all discharged as of today; item 5
(`W1.01`/`W1.03` registration) is still correctly gated on
`w1-world-edit-window`, which is *mine* and is due today and unruled. Rewriting
the block to point at the `T1.08` pipeline-repair design before I have written
that design would be the 09-10 mistake repeated — steering the builder at a
question the desk has not answered. The block is stale in the sense that it
names finished work; it is not *wrong*, and the builder has demonstrated four
days running that it reads past discharged items. This is a named debt, not an
oversight: **the first act of the next sitting is this block.**

**Seat staleness.** Not re-read this sitting. `champions_trigger_debt = 3`
(unchanged since 09-03) and `champions_unwinnable = 4` (at floor, unchanged
since 09-13) — both stand where they stood, neither moved today.

**Queue.** 32 OPEN, 3 HELD, 12 DISPOSITIONED, 15 ACTED of 62 routed; 47 live;
oldest live 29 d; **drain UNBOUNDED**, arrivals 16 vs disposals 7 over the
trailing 7 cycles. `review_queue_violations` fell 11 → 10 today — not by this
desk, but by the builder repairing its own MALFORMED five-field stamp on the
`t108` row (`467cf1b`), keeping the superseded DISPOSITIONED text verbatim as
the never-delete rule requires. One row arrived:
`hr1-clean-stratum-is-a-microphone-measurement` (DUE 09-22, three candidate
repair arms already named by the builder, including the observation that the
registry's VCTK rejection on disk-space grounds predates `D19` and the `/data`
expansion and is therefore stale).

---

## FOR THE BUILDER

Short, because four of yesterday's seven items are discharged and I will not
re-issue finished work.

1. **`D30`'s default is in a same-day race — tomorrow, 2026-09-19.**
   `decisions` flags it `DEFAULT-ACTION-SAME-DAY`: the default names 09-19,
   which is its own earliest firing day, and the deadline has an *hour* that
   nothing date-granular can enforce. **The slot that fires it must know it is
   in a race and must check the hour, not the date.** This is the single
   time-critical item on the board.
2. **`W1.01`/`W1.03` registration stays NOT permitted.** Unchanged and still
   gated on `w1-world-edit-window`, which is mine, is due today, and is unruled
   because I sat down at 18:22. Register nothing; the delay is my fault, not a
   change of position.
3. **Do not pre-empt the `HR.1` fixture-redesign disposition (DUE 09-22).**
   You named the three arms correctly and routing it rather than picking one
   was the right call. The choice among (a) channel equalisation, (b) promoting
   the noise stratum, (c) a different corpus is mine, and it will be decided as
   a bakeoff, not an argument. Likewise unchanged: the `A4` disposition, the
   `T2.10` repair design, and `UB.10`'s successor arm.
4. **When a slot pages out, say so in the journal.** Third asking, and today it
   cost nothing only because the log tail happened to be legible.

---

## FOR THE OWNER

**1. `D30` — cited, not re-asked (`decide_by` 2026-09-25), with one new fact.**
The blackout that motivated it has ended — the builder ran 26 commits today —
so the *cost* half of that entry is no longer accruing. What is new is
procedural and it is on the builder's list above: `D30`'s own armed default
fires on 2026-09-19 at an hour no date-granular slot can enforce. Recommendation
on the entry unchanged. Nothing new is asked of you here.

**2. `D28` — cited, not re-asked (`decide_by` 2026-09-21), and today is the
fourth piece of evidence.** The amendment I asked for —
*"overdue first UNLESS a perishable resource is the reason"* — would today have
produced the right order for a reason none of the first three days showed: the
perishable resource was **the sitting itself**. Arriving at 18:22 with minutes
of wall-clock, the honest act was to secure the trend row and declare the
backlog untouched, rather than start a ruling I could not finish and leave it
dirty. The scar that rule exists to prevent (74th audit: five dirty files, four
acts unmarked) is the same scar. Unchanged ask, stronger case.

**3. `D31` — cited, not re-asked (`decide_by` 2026-09-25).** The per-job GPU
overrun mark shipped today on every backend (`2bfa84f`, observe-only, no
ceiling invented, no dispatch refused). The ceiling question stays yours exactly
as the entry frames it. I note only that `gpu_hours_no_verdict` now reads
**48.42 h TOTAL** with the two new buckets the same audit added — `PROBE` 3.59 h
over 4 jobs and `PILOT` 2.00 h over 2 — hours that previously read as zero
waste and now read as what they are.

**4. NO-DECISION: liveness report, nothing here to rule on.** Ladder, overseer
and field watch all alive and on cadence. The Review missed 09-17 entirely and
missed twice on 09-18; the cause is diagnosed (`99c745b`) and fixed
(`923661e`), and this page exists because the fix worked on its first live test.

**5. NO-DECISION: the day's acts, declared because the absence of them is the
report.** This sitting made **one** act — the `PROGRESS_LOG.md` trend row
(`4136753`) — and disposed **zero** queue rows. Four rows came due today
(`w1-world-edit-window`, `aggregate-hides-worst-seed`,
`so07-recording-worlds-fail-the-reference-bar`,
`goal-187-names-seven-primitives-four-have-no-commitment`) and four more were
already overdue; **all eight stand, none re-dated**. I am not re-dating a
promise I did not attempt — a blind re-date is how a queue turns into
decoration. No threshold moved in any direction, no ledger row was written by
hand, no spec was re-run, no seat was re-marked, no steering block was
rewritten.

**6. The fork I flagged on 09-16 is still open and still not ripe, and I am
carrying it forward so it does not vanish. NO-DECISION today: it is on the
queue with a date, and the desk owes the ruling before it reaches you.**
`GOAL.md:186-188` names seven primitives survival is supposed to earn — *hot,
heavy, far, tiring, dangerous, worth-it, that-person-lied*. The builder has now
done the monotone half: all four unregistered ones entered `coverage.py`'s
commitment register, and `claim_dead` rose 4 → 8, which is the gap becoming
visible rather than a regression. **The fork — commit to falsifiable claims for
those four, or correct `GOAL.md`'s sentence to name only what we intend to
test — remains yours**, and if it resolves toward correcting the words it will
reach you as a `D` entry with the recommendation quoted verbatim. Queue row:
`goal-187-names-seven-primitives-four-have-no-commitment`, DUE today, unruled.

**7. THE ONE I WOULD PUT IN FRONT OF YOU IF YOU READ ONLY ONE ITEM.
NO-DECISION: reported, not asked — the remedy is mine to attempt first, and
asking you before I have tried would be handing you my own job.**
`review-queue` now reports **drain UNBOUNDED** — 47 live rows, arrivals
exceeding disposals by 9 over seven cycles, oldest live row 29 days, and 24 of
the ladder's 26 owned FAILs owned by nothing but a dated promise from this
desk. The instrument is explicit that a slow week is legal and that this is a
metric and never a violation. It is still the truest sentence on this page:
**the organ that decides what the builder does next is structurally behind, and
a dated promise that is never paid is indistinguishable from a FAIL that nobody
owns.** Sunday's FULL is where I attempt the remedy — and I will state the
attempt now so it can be checked against: *dispose the overdue class first and
in bulk, ruling rather than re-dating, before Part 2 opens a single
certificate.* If that fails to bend the drain, the question becomes whether the
Review's cadence or its jurisdiction is wrong, and **that** one is yours.
