# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-09 06:37–06:5x UTC — DAILY.** Window: the last 24 hours
(2026-09-08 06:50 → 2026-09-09 06:46).

*The one sentence: **the builder did not produce, thrash or stall — it was
switched off, and the switch was thrown by a meter that is 62% somebody
else's.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `f7d7d5d` (D26
> routed), the six re-dated queue rows, and the `PROGRESS_LOG` row.

---

## The numbers

| | today | yesterday |
|---|---|---|
| demonstrated / registry | **108 / 245** | 108 / 245 |
| pass rate | **44.1%** | 44.1% |
| net demonstrated | **0** | +2 |
| rework rate | 77.6% | 77.6% |
| unreachable (shrink-only) | 93 of 245 (38%) | 93 |
| ledger settlements in 24 h | **0** | 47 |
| commits in 24 h | **0** | 11 |
| builder slots fired / *used* | 22 / **0** | 22 / 22 |
| queue violations | 0 | 0 |
| live queue rows | 41 | 41 |

Every number in the first column is yesterday's number. That is the finding,
not the preamble to one.

---

## Part 1 — did the builder produce, thrash, or stall?

**None of the three.** `pace_gate` (`scripts/lib_usage.sh:74`) skipped **all 22
hourly slots since 2026-09-08T08:23**. The last commit on `main` is `81a815e`,
22 hours old. No spec ran, no attempt was made, no ledger row moved.

### The measurement: 62% of the meter that gates us is not ours

`pace_gate` compares `week:all models` against a line rising from
`PACE_FLOOR=25` at the reset to `PACE_CAP=90` at the week's end. At 06:07 it
read **57% spent into 29% of the week, line 44%** — and skipped.

`usage_ledger.jsonl` writes a `start`/`end` pair carrying a percent reading for
every organ run, so our own spend is directly summable. Since the week reset
(2026-09-07T05:23 UTC, inferred from `--week-elapsed`), across all 36 completed
organ runs:

```
our own attributable spend (builder + overseer + review + field watch)   23 points
rises recorded while NO organ of this project was running                38 points
------------------------------------------------------------------------------
week:all models at 2026-09-09T06:37                                      59%
```

The largest single interval is unambiguous: **`09-08T08:23 → 09-09T06:37`,
+28 points**, during which the ladder log proves all 22 builder slots were
skipped and the overseer log proves three of its four audits paced. Nothing of
ours ran, and the meter moved more in that gap than our four organs moved it
all week.

### This is the gate's own documented failure mode, arriving through its cure

`lib_usage.sh:47-53`, the comment that justifies the pace line, already says it:

> THE CAUSE IS NOT OVERSPENDING. `week:all models` is a SHARED pool [...] the
> loop is stopped by consumption it does not control, and being the only
> consumer with a gate, it is the one that starves.

The line was built to cure that starvation by spreading *our* spend. It has no
attribution, so it reads external drain as our own prodigality and responds by
starving us further. **Under the 90% stop alone the builder would be running
right now, at 57%.**

### The cost is a forecast, not an adjective

`allow = 25 + ceil(65·elapsed/100)` first exceeds 57 at `elapsed = 50%`. The
week began 09-07T05:23, so the builder is foreclosed until **2026-09-10T17:23
UTC — 35.3 further dark hours, 57.0 consecutive.** That is the largest loss of
builder capacity since the 4.3-day August blackout, and it is happening while
the ladder is healthy, the queue is at zero violations, and yesterday's
priority block holds five runnable units nobody has read. W37's free Kaggle
allocation expires 09-13; the builder wakes on the 10th.

Routed as **`D26`** — see FOR THE OWNER.

---

## Part 2.5 — steering maintenance

**1. Priorities — deliberately NOT reconciled, and that is the act.** The
`1'''''`/`2'''''` block was written 2026-09-08 06:50 and **the builder has never
read it**: every slot since 08:23 was skipped. It is unspent, not stale.
Reconciling a block nobody has consumed would destroy the only thing waiting
for the builder when it wakes, in exchange for nothing. Left untouched on
purpose; it is still live and still correct.

**2. Field watch — nothing to consume.** Last sweep 2026-09-07 (wk6), consumed
in full by the 09-07 Review (all three nominations dispositioned, 42 minutes
after landing). Next sweep Monday 09-14. `FIELD_WATCH.md` unchanged since
`3b68b7d`.

**3. Seat staleness.** Unchanged from yesterday: Control architecture (D1) still
VACANT with `D1.0` its whole arena, now carrying a committed design path (the
twin-spread probe, both branches pre-registered, DUE 09-14). Vision encoder's
finding was discharged yesterday. No seat's arena context changed in the last
24 h — nothing ran.

**4. Organ liveness — all four alive, and that is the finding's sharpest edge.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:07:13 today | **alive; fired 22/22 slots and refused 22/22** |
| overseer | 6-hourly | 06:37 today (running) | alive; paced 3 of 4 on `D15` clause c |
| field watch | Mondays | 09-07, consumed | alive |
| review | daily / Sun FULL | this run | alive |

Not one organ is broken. Nothing crashed. No log is stale. **The system is in
perfect health and did nothing for a day** — which is exactly the shape a
liveness check cannot see, because liveness reads mtimes and the skip path
writes a log line every hour.

---

## Dispositions committed this morning (each in its own commit, as it was made)

Six rows came due. **None was disposed; all six were re-dated in the open with
reasons, and three were bundled.** I am not dressing that up: this desk
discharged zero design questions today.

- **`sh02-null-saturation`, `ba03-null-saturates-the-horizon`,
  `t306-matched-magnitude-noise-buys-coverage` → 09-13, BUNDLED as ONE
  question.** They are the same disease wearing three spec ids: a null or
  anchor that saturates, so the gate cannot resolve what it exists to resolve.
  Yesterday's Review named that across four further fronts (`UB.10`'s anchor at
  ceiling, `w0-too-shallow`'s world that separates nothing, `t309`'s venue where
  correct advice hurts, `dp04`'s 76.7% at the cap) and sent it to Sunday as a
  single question about how this project chooses what to measure. Designing
  them separately on a Wednesday is the act that would guarantee three
  incompatible local repairs. **Declared rather than buried: 09-13 goes 10 → 13
  rows against a measured capacity of 6** — but 11 units of design.
- **`pl02-eye-gate-reads-the-encoder-not-the-eye` → 09-11, on a REFUSAL rather
  than capacity.** It asks which reading of the B4 VOID gate binds `PL.02` — the
  encoder letter, or the eye its own title names — measured to diverge by
  **0.93**. `PL.02` is the sole registered falsifier of GOAL.md's PLASTIC-ONLY
  decree. Two days ago this desk refused to re-point that same edge in the week
  it produced an inconvenient result, and ordered an expensive renderer bakeoff
  instead. Ruling on the gate's reading in the last ten minutes of a DAILY, *in
  the direction that would let the falsifier run*, is that same refused act
  wearing a deadline. Nothing is weakened by the delay — the registered run
  stays blocked under either reading.
- **`hr5-fixture-refuted` → 09-12.** My **second consecutive re-date of this one
  row**; 9 days open across three promised dates. Named rather than left to look
  routine. Substantive reason unchanged and still binding (it rides the W1 fork
  and must not be designed twice); proximate reason today is capacity, mine.
- **`w0-kills-a-forager-by-integrity-at-25-minutes` → 09-10.** The cheap one: it
  asks for no design of its own, only that the W1 design consume its numbers,
  and that design now exists (`W1.00`–`W1.04`). A reading owed, not a fork.

**The instrument caught me again, in the same family as yesterday.** My first
attempt wrote all six new `DUE:` clocks **above** the existing ones;
`review_queue.py` takes the **last** DUE, so all six were silently ignored — a
no-op that would have read as six honest re-dates. Found by re-reading the
tool's output instead of trusting my own edit, one day after four clocks landed
below un-indented prose. Queue: **0 violations before and after.**

---

## The frontier

Unchanged, because nothing moved it. The most important unblocked unit is still
**`D1.0`'s twin-spread probe** (forward passes only, both branches
pre-registered, DUE 09-14) — item 1 of the block the builder has not read. The
transitive-block mass behind `T2.01` (38 specs) is untouched.

**The frontier question is different today and it is not about a spec.** For 24
hours the binding constraint on this project was neither design throughput nor
compute nor the ladder: it was a shell function comparing our line to a number
we did not spend. Nothing on the frontier can move until that clears on 09-10.

---

## The honest paragraph

No numbers. We are not closer to a creature that lives, learns and is known —
and today, for the first time in this desk's record, we are not busier either.
Jack is exactly what he was yesterday. Nobody looked at him. The most important
step toward Jack this week was made *before* this window, when the builder
bought back the constitution's only falsifier by fixing a renderer rather than
by editing the edge that was in its way; that was real and it still stands. The
most concerning drift is what today revealed about the shape of this system: we
have built organs that measure the ladder, the queue, the seats, the decisions,
the certificates and each other, and not one of them raised its voice while the
only organ that can actually move the creature sat switched off for a day. The
alarm did not fail. There was no alarm. Every instrument reported health,
truthfully, because every instrument measures whether the parts are *working*
and none measures whether the thing is *going anywhere*. I found this by reading
a log by hand, on a morning I happened to be scheduled. That is not a system
catching itself; that is luck with good paperwork.

---

## FOR THE BUILDER

The board is unchanged and the block `1'''''`/`2'''''` in
`scripts/ladder_prompt.md` **stands untouched — start there.** You have not read
it; it is not stale. Two additions only, both of which assume you are reading
this after 2026-09-10T17:23 UTC:

1. **Your first act on waking is a measurement, not a spec.** Before you take
   item 1, append one line to `docs/LOOP_JOURNAL.md` recording how many
   consecutive slots you skipped and the `week:all models` reading that
   released you. Nothing in this system currently counts a dark slot, which is
   why 22 of them produced a page instead of an alarm.

2. **If `D26` has been ruled or its default has fired, item (iv) is yours to
   build and it is small**: `pace_gate`'s skip line additionally prints our own
   attributed spend beside the shared total, summed from `usage_ledger.jsonl`'s
   existing start/end pairs, and consecutive dark slots become a ratcheted
   metric. **Gate nothing. Change no behaviour.** No spec declares
   `lib_usage.sh` in `IMPL_DEPS`, so no certificate is staled by it. If `D26` is
   still open, do not pre-empt it.

---

## FOR THE OWNER

**1. `D26` — the builder is dark for 57 consecutive hours because the gate that
stopped it reads a meter that is 62% somebody else's.** Routed today,
`class: goal`, `decide_by: 2026-09-10`, `decisions --check` EXIT 0. Four
options; the default armed is **(iv) MEASURE ONLY**, which is the only one of
the four that loosens nothing — and I want it on the record that **the default I
armed is not the answer I believe in**. My recommendation is **(i) ATTRIBUTE THE
LINE**: `pace_gate` compares its line to our own summed spend while
`usage_gate`'s 90% hard stop keeps reading the shared pool exactly as it does
today. That is not a loosening of this project's real ceiling; it is the removal
of a second, unintended ceiling nobody set and no decision ever ratified. I
routed it rather than took it because it is a gate, and gates are yours. **The
price of my own default, stated: (iv) fixes nothing.** It makes the next
occurrence visible within one slot instead of within one Review.

**2. `D22` is OVERDUE and I did not fire it — a loose end I am naming rather
than leaving.** Its `decide_by` of 2026-09-08 passed unanswered; `decisions`
prints `OVERDUE — DEFAULT IS DUE TO FIRE`. The default is **(i) THE RULE
STANDS**, which *denies my own ask*. I left the firing paperwork to the
overseer, which set the precedent on `D17` and was running concurrently with me
on the same file — but silence is never success, so it is here in writing. **Its
premise is now doubly falsified anyway:** `D22` argued design throughput binds
the project because *"the builder has an empty board, 24 slots a day."* Today
the builder has **0 slots a day**. Yesterday I told you the evidence had moved
against my own ask; today it moved further. **I withdraw it.** If nobody fires
the default, it should be fired as (i).

**3. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive on cadence — builder fired 22 of 22 slots and refused 22 of 22; overseer
paced 3 of 4 on `D15` clause c and is running now; field watch swept 09-07 and
was consumed the same day; this run. Nothing is broken. That is the report, and
in this instance it is also the problem, which is item 1's subject and not a
second ask.

**4. NO-DECISION: the docket, announced rather than asked about.** Six rows came
due, none was disposed, all six re-dated with reasons in their own commit; three
bundled onto Sunday as one question, taking 09-13 to 13 rows against a measured
capacity of 6. Sunday's FULL is oversubscribed and I am telling you so a week in
advance rather than discovering it at 06:37 on the day. `hr5` is on its second
consecutive re-date by this desk and `pl02`'s slip is a refusal, not capacity.
