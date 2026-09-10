# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-10 06:37–07:0x UTC — DAILY.** Window: the last 24 hours
(2026-09-09 06:37 → 2026-09-10 06:5x).

*The one sentence: **the gate is still the story and it got 28 hours worse, but
the day's real find was smaller and sharper — the builder's own steering told it
to wait for a GPU window that has been open since Monday and closes on
Sunday.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `d582acc`, `1a0e413`,
> `e8a8298`, `91024db`, `6e79f7d`, `20ba75f`, `e4edb57`, `cbd6438`, `fd2101d`,
> `18bc889`.

---

## The numbers

| | today | yesterday |
|---|---|---|
| demonstrated / registry | **108 / 245** | 108 / 245 |
| pass rate | **44.1%** | 44.1% |
| net demonstrated | **0** (2nd consecutive) | 0 |
| rework rate | 77.6% | 77.6% |
| unreachable (shrink-only) | 93 of 245 (38%) | 93 |
| ledger settlements in 24 h | **0** | 0 |
| commits in 24 h | 6 — **0 from the builder** | 0 |
| consecutive skipped builder slots | **46** | 22 |
| `week:all models` | **68%** (line 54%) | 59% (line 45%) |
| `week:Fable` | **100%** | 95% |
| W37 free GPU-hours charged | **0.00 of 30** | 0.00 |
| queue violations | 0 (2 → 0) | 0 |
| live queue rows | 39 | 41 |

Third day at 108/245. The six commits are this desk's and the overseer's.

---

## Part 1 — did the builder produce, thrash, or stall?

**Still none of the three, and the number that matters is the forecast, not the
outage.** Yesterday's page told the owner the builder was foreclosed until
**2026-09-10T17:23 — 57 consecutive dark hours.** Re-deriving from the same
arithmetic against this morning's readings, the release is now
**2026-09-11T22:07: 85.7 consecutive dark hours.** *Nothing was decided,
nothing was spent, and the wait grew by 28 hours in 24.*

### The evidence is cleaner than yesterday's, in two independent ways

**(1) An interval with literally nothing of ours in it.** `usage_ledger.jsonl`
records **no organ run of this project whatsoever** between the overseer's `end`
at 09-09T06:50 (61%) and this morning's `start` at 06:37 (68%). Not a builder
slot, not a field watch, nothing. **+7 points, zero attributable, across a full
day.** Yesterday's window at least contained skipped slots to argue about.
Re-summed over the week across all 37 completed organ runs: **26 points ours,
42 not ours, of 68 — 62% external**, the same ratio on a bigger number.

**(2) The reporting-lag explanation is dead on its own evidence.** The ladder log
shows the meter **flat at 67% for thirteen consecutive hourly readings**,
09-09T17:07 → 09-10T06:07, then 68% at 06:37. Quiet all night, climbing in
daylight. That is a person's working day, and nothing of ours ran in either half.

### The arithmetic that should decide `D26`, and it was not in the entry

The 09-11 release assumes the external consumer never draws again:

```
the pace line rises          65 points / 168 h              =  9.3 points/day
external draw measured  09-08T08:23→09-10T06:37  +37/46 h   = 19.3 points/day
                        09-09T06:50→09-10T06:37   +7/24 h   =  7.0 points/day
```

**At the gentler of the two measured rates the line closes on the meter at 2.3
points a day against a 14-point gap: ~6 days.** The week resets 09-14T05:23. So
under the drain this project has actually measured, **`pace_gate` never releases
the builder at all** — the blackout ends because the week rolls over. That is a
**6.0-day loss of the only organ that can move the creature, larger than the
4.3-day August blackout.**

---

## The day's real finding, and it is not the gate

**`scripts/ladder_prompt.md` — the file the builder navigates by — said
`attempt 3 goes to W37 (opens 09-13)`. `2026-W37` is Monday 2026-09-07 through
Sunday 2026-09-13. It is THIS week. It does not open on 09-13; it CLOSES on
09-13, and it opened three days ago.**

One ISO week-boundary off: W36 ended Sunday 09-06, so W37 began the next
**Monday**, not the next Sunday. The error travelled from `LOOP_JOURNAL` into
three `REVIEW_QUEUE` rows and into the builder's live priority block, and every
copy says the same thing — *wait for a window that is already open.*

The bill is measured, not feared. **`gpu_budget.json` has no `2026-W37` key at
all: 0.00 of 30 free Kaggle GPU-hours, with the week 3 days gone.** Predecessors,
from the same file: W32 16.61, W33 7.89, W34 1.62, W35 19.20, W36 17.73.

**It compounds with the blackout rather than sitting beside it.** Released no
earlier than 09-11T22:07, the builder has under ~26 hours against attempt 2's
measured 17.61 GPU-h. A builder that then defers to a phantom 09-13 opening
spends **zero**. Corrected directly in the priority block (`fd2101d`) —
priorities are operational and this desk may edit them. **No gate moved, no
unchanged re-dispatch authorised, no precondition shortened**: the twin-spread
probe and the committed successor gate still bind, in that order, before any
dispatch. A date was corrected and nothing else.

*Why this desk did not find it yesterday:* yesterday I judged the priority block
**unspent rather than stale** and left it untouched on purpose, which was the
right call about staleness and the wrong call about correctness. "The builder has
not read it" is a reason not to *rewrite* a block. It is not a reason not to
*read* it.

---

## Part 2.5 — steering maintenance

**1. Priorities — reconciled today, for the first time this week, and it was a
correction rather than a rewrite.** The `1'''''`/`2'''''` block still stands
unspent (46 skipped slots; the builder has never read it) and its five units are
unchanged. What changed is the W37 date error above, in both places it appears
(items 1 and `2'''''`).

**2. Field watch — nothing to consume.** Last sweep 2026-09-07 (wk6),
`3b68b7d`, consumed in full by the 09-07 Review. Next sweep Monday 09-14.
Unchanged.

**3. Seat staleness — one new finding, and it is a seat that does not exist.**
See `t215` below: `experiments/champions` has **no language-ROUTING seat**.
`Language grounding (word → lived skill)` is UNDECIDED on arena `LG.04/LG.05/
LG.06`, all NOT_RUN; `Language acquisition` and `Language model` are BY DECREE on
`LG.00`. None of those arenas is `T2.15`, `T2.07` or `T2.06`. Separately, the
`World` seat still reads `VERDICT-UNDECLARED, TRIGGER-UNDECLARED` — carried, not
new. D1 (Control architecture) remains VACANT with `D1.0` its whole arena. No
seat's arena context changed in 24 h; nothing ran.

**4. Organ liveness — all four alive.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:07:13 today | alive; **46 consecutive slots fired and refused** |
| overseer | 6-hourly | 06:37 today | alive; 86th audit committed `22154cd` |
| field watch | Mondays | 09-07, consumed | alive |
| review | daily / Sun FULL | this run | alive |

---

## Dispositions committed this morning (each in its own commit, as it was made)

**Seven rows due or overdue. Seven handled. Queue: 2 violations → 0.**

- **The two OVERDUE `d10-*` gate rows → ACTED (`d582acc`), and they were
  bookkeeping, not undone work.** Both were promised 09-09 and owed by a builder
  that has been dark 46 hours, so they read this morning as the blackout's first
  queue casualties. They were not. `8f2990d` (09-06 08:19) executed all three
  adopted conjuncts — G1 the paired-own-twin statistic that stops "noisy" and
  "did not learn" sharing one verdict, G2 the consistency conjunct, G3 the
  verbatim SB3 reference lane run first in its own kernel — verified red-first,
  three days *inside* the clock. The commissioned run landed too (D1.0 attempt 2,
  `3a4ccfd`, VOID 09-07), and its successor is live on
  `d10-successor-rerun-under-adopted-gate` DUE 09-14, so nothing is buried.
  The rows recorded `EXECUTED` in their own prose and never carried the `ACTED`
  marker `review_queue.py` reads.
- **`w0-kills-a-forager-by-integrity-at-25-minutes` → ACTED (`1a0e413`), and the
  reading came back PARTIAL.** The row asked one thing: did the W1 design consume
  its numbers? **It cited them and then wrote a spec that cannot see them.**
  `W1.04` names the 25-minute integrity death by name — and constrains only the
  horizon a *designer declares*. A declared horizon is a free variable; a body
  wrecking at sim_s 1476.9 ± 382.0 is not. **A venue where every life ends at 25
  minutes PASSES `W1.04` as published, by declaring a 20-minute horizon.** None
  of the other four bounds the life either. **STRENGTHENED: `W1.04` gains
  conjunct (c) THE LIFE IS LONGER THAN THE HORIZON** — 5th-percentile *measured*
  survival ≥ declared horizon, per-life termination cause on the ledger row, an
  explicit ban on repairing it by shortening the horizon to fit the deaths, and a
  mechanism-disabled twin as control. Strictly harder; free, because `W1.04` is
  not registered. Named rather than absorbed: this does not diagnose whether the
  25-minute cap is a W0 bug or an honest hostile world. It makes it fail loudly
  instead of passing quietly.
- **`goal-cites-four-specs-that-resolve-to-corpses` + `reparenting-the-welded-
  fifteen` → BUNDLED as ONE question, DUE 09-15 (`20ba75f`).** Same two weld
  roots, same registry surgery; one date so they cannot be answered
  inconsistently on two mornings. **The finding that came from reading them
  together: four of the seven corpse citations (`GEN.02/03/06/09`) are welded
  behind `LC.07`, whose arena this desk declared VENUE-UNAFFORDABLE on 09-06 and
  whose affordability is the owner's open `D24`. Those four are not repairable
  by any surgery this desk performs.** `LC.03`'s three are. 09-15 is chosen
  because the input is `W1.01/03/04` registration — builder work — and 09-13
  already carries 13 rows against a capacity of 6.
- **`told-world-has-no-rung` → sub-question (b) ANSWERED YES (`e4edb57`), and the
  row's own declared bill was wrong.** Its premise was *"LG.00 has no rows"*.
  **`LG.00` PASSED 09-06** — attempt 7, seeds 0/1/2, 1.33 s CPU,
  `grounded_knowledge_advantage` **0.5327 ± 0.0489**, control lane
  `advantage_general` **−0.2000** at std 2.8e−17 on every seed. The apparatus
  exists and already carries the exact control shape `LG.11` needs: GOAL.md's
  *smarter inside his life, dumber outside it* as a live number. **BILL
  CORRECTION: the row's `SEMANTIC bill: none today` is now false** — `LG.00`
  holds a live PASS pinned to `impl_sha e2d9b4d0350951b5`, so editing the strip
  apparatus to serve `LG.11` stales it. Free to assume, not free to implement.
  (a) re-dated to 09-15 with its premise *weakened*, deliberately not declared
  answered: settling it needs a read of how `LG.00` sources its life corpus, and
  a wrong answer licenses a told-world rung built on a corpus nobody lived.
- **`t215-router-under-lexical-null` → DISPOSITIONED (`cbd6438`) on a defect one
  level above its own question.** It asks whether the anchor-argmax router
  *keeps the seat*. **There is no seat.** Meanwhile four PASS certificates hash
  `UnifiedBrain.py` in `IMPL_DEPS` (`T2.03`, `T2.04`, `T2.06`, `T3.01`) and two
  independent FAILs localise a defect in it (`T2.15` at [8,9,5]/16 against a
  12/16 bar, *beaten by both registered bag-of-words nulls on seed 2*; `T2.07` at
  [2,2,2]/5). **A component four certificates depend on, that two specs have
  refuted, and that no seat watches, is the exact thing `CHAMPIONS.md` exists to
  prevent — and it is invisible to every audit we run, because seat-staleness
  checks seats that EXIST.** Routed to Sunday's anatomy audit (seat creation is
  FULL-mode work), taking 09-13 to **14 rows against a capacity of 6**, declared
  rather than buried. Not DECLINED: its own decline-condition required *no Review
  disposition*, and a disposition is what this is.

**The instrument caught me twice more, and I caught it a third time.** My
`ACTED` on `w0-kills-a-forager` named no commit (`ACTED-WITHOUT-A-COMMIT`); my
fix put the hash in a fifth pipe field (`MALFORMED`); both self-caught within a
minute by re-running the tool rather than trusting the edit. With this morning's
two `d10-*` rows that is **four instances in three days of this desk writing a
true thing in a shape `review_queue.py` cannot read** — 09-08 four clocks under
un-indented prose, 09-09 six DUE clauses above the line the tool takes, today an
`ACTED` with no commit and a malformed line. The pattern is not carelessness
about the truth; it is carelessness about the *reader*.

---

## The frontier

**`D1.0`'s twin-spread probe** is still the most important unblocked unit
(forward passes only, both branches pre-registered, DUE 09-14), standing in front
of `T2.01` — frees 35 / blocks 38, still the largest single unblock in the
project. Untouched; nothing ran.

**But the frontier's binding constraint changed shape today.** Yesterday it was a
shell function comparing our line to a number we did not spend. Today it is that
*plus* a date error which would have wasted the release when it finally comes.
The gate cost us the days. The date would have cost us the hours that were left.

---

## The honest paragraph

No numbers. We are not closer to a creature that lives, learns and is known, and
for the second day running we are not busier either — Jack is exactly what he was
on Monday. But today was not yesterday repeated, and the difference is worth
naming. Yesterday's page said every instrument reported health while the creature
sat still, and called that luck with good paperwork. Today the paperwork earned
something: reading rows nobody was forcing me to read turned up a load-bearing
component that no seat has ever chaired, a design that quoted the evidence
against it and then wrote a test that would have certified the defect, and a
one-week date error sitting in the file the builder steers by. None of those was
found by an alarm either. All three were found by asking *so what?* of documents
that were already green. The week's single most important step toward Jack
remains the one made before this window — the constitution's only falsifier
bought back by fixing a renderer rather than editing the edge in its way. The
most concerning drift is now specific enough to state as a sentence: **this
project's errors are migrating out of its instruments and into its prose.** The
ledger is sound, the gates hold, the ratchets are shrink-only — and the things
that hurt us this week were a date in a priority block, a marker missing from a
queue row, a bill line that was true when written, and a seat nobody drew. Not
one of them is a threshold. Every one of them steers.

---

## FOR THE BUILDER

The `1'''''`/`2'''''` block stands and you still have not read it — **start
there.** Three amendments, in order:

1. **W37 IS THIS WEEK AND IT CLOSES SUNDAY 09-13.** The block used to say
   "W37 (opens 09-13)". It was wrong by one ISO week boundary and it is
   corrected in place (`fd2101d`). The quota is live now and stands at 0.00 of
   30 hours. **Do not defer the attempt-3 dispatch to 09-13.** The probe-and-gate
   precondition is unchanged and still binds: twin-spread result on the row,
   successor gate committed in a non-dispatch commit, and only then a dispatch.
   If ~26 hours does not fit 17.61 GPU-h, **say so on the row** rather than
   squeezing it — but decide that against the real deadline.

2. **Your first act on waking is still a measurement, not a spec.** Unchanged
   from yesterday: append one line to `docs/LOOP_JOURNAL.md` recording how many
   consecutive slots you skipped and the `week:all models` reading that released
   you. It is 46 and counting. Nothing in this system counts a dark slot.

3. **`W1.04` gained conjunct (c) before you register it** — see
   `w0-kills-a-forager` and the `W1.04` block on `w0-too-shallow`. Register from
   the amended design, not the 09-06 text. It stales nothing, because `W1.04` is
   not registered.

---

## FOR THE OWNER

**1. `D26` — the builder's blackout is now forecast at 6.0 days, larger than the
4.3-day August one, and the gate may never release it at all.** `decide_by` is
**today**; the evidence addendum is committed (`6e79f7d`) and `decisions --check`
is EXIT 0. My recommendation is **unchanged and quoted verbatim in the addendum
so the instrument still matches it: (i) ATTRIBUTE THE LINE.** What is new since
you last saw this: the release slipped 09-10T17:23 → 09-11T22:07 while nothing
happened; at the *gentler* measured drain rate the line never catches the meter
before the week resets on 09-14; the cleanest interval yet shows **+7 points with
zero organ runs of ours in it**; and the meter sat flat 13 hours overnight,
killing the lag explanation. **The price of (i), stated and worsened:**
`week:Fable` read 95% yesterday and reads **100%** today, so under (i) the
builder clears `pace_gate`, is refused Fable by `model_gate`, and runs on **Opus
at a per-slot cost nobody has measured** until 09-14. That does not change my
recommendation, because the alternative on offer is six dark days and a fourth
GPU allocation dying unspent. The armed default remains **(iv) MEASURE ONLY** and
I say again that it is not the answer I believe in; a default may not loosen a
gate, and (iv) fixes nothing.

**2. NO-DECISION: the W37 date error, reported because it is expensive and
because it is mine to fix, not yours to rule on.** The builder's steering told it
to wait for a GPU window that opened Monday and closes Sunday; `gpu_budget.json`
carries 0.00 of 30 hours with three days left. Corrected directly in the priority
block — priorities are operational. I am telling you rather than asking you
because there is no fork here, only a cost you should know about: **W37 is on
course to be the fourth allocation in six weeks to die largely unspent, and this
time two independent causes were sufficient on their own.**

**3. NO-DECISION: the docket, announced rather than asked about.** Seven rows due
or overdue, seven handled, queue 0 violations. Three ACTED, four dispositioned or
bundled. Sunday 09-13 goes to **14 rows against a measured capacity of 6** — I
added the fourteenth myself (the `t215` seat question) and I am telling you a
week in advance rather than discovering it on the day. Sunday's FULL is
oversubscribed and will not clear.

**4. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive on cadence — builder fired and refused 46 of 46 slots, overseer committed
its 86th audit, field watch swept 09-07 and was consumed, this run. Nothing is
broken.

**5. `D22` remains OVERDUE and unfired, and I withdrew it yesterday.** Its
default is **(i) THE RULE STANDS**, which denies my own ask; its premise is
doubly falsified (it argued design throughput binds the project because *"the
builder has an empty board, 24 slots a day"* — the builder has had 0 slots a day
for 46 hours). Repeated here rather than dropped, because a `VANISHED-OWNER-ASK`
is exactly the scar this section exists to prevent. If nobody fires the default,
it should be fired as (i).
