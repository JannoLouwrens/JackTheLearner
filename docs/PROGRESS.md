# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 over the last 24 h, and Part 2.5 in full). Part 2 is
> deliberately skipped — tests are re-examined on Sundays.

**2026-09-15 06:37–07:2x UTC — DAILY.** Window: the 24 hours since yesterday's
DAILY sat down.

*The one sentence: **the builder did its best work of the week in the five hours
it was awake, shipped an instrument that settled a two-cause question on the very
next failure — and then went dark for eighteen consecutive slots on a usage meter
three-quarters of which this project did not spend, leaving that finding unread
for nineteen hours and 26.51 free GPU-hours running toward a Saturday expiry with
an authorised buyer and nobody awake to spend it.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `8c85e71`, `cfd3bf1`,
> `1851448`, `1466035`, `54f117d`.

---

## The numbers

| | today (24 h) | yesterday (09-14, DAILY) |
|---|---|---|
| demonstrated / registry | **108 / 249** | 108 / 249 |
| pass rate | **43.4%** | 43.4% |
| net demonstrated, 24 h | **0** | +1 |
| settle events, 24 h | **1** — 1 PASS (`T0.36`, 08:18) | 36 |
| distinct specs settling / PASSing | **1 / 1** | 18 / 13 |
| rework rate (attempt > 1) | **78.4%** (116/148) | 78.4% |
| standing VOID / FAIL / BLOCKED | 14 / 25 / 1 | 14 / 25 / 1 |
| unreachable (shrink-only floor) | **97** | 97 |
| stale certificates | **12 + 1 pre-`impl_sha`** (`T1.07` is new, and deliberate) | 11 + 1 |
| commits, 24 h | **26**, every one of them before 12:07 | 107 |
| **consecutive dark builder slots** | **18** | 0 |
| `week:all models` | **37%** against a 35% pace line | 3% |
| …of which **NOT THIS PROJECT** | **28 of 37 points — 75%** | — |
| GPU, live week `2026-W37` | 3.4899 charged, **26.51 free of 30**, expires **Sat 09-19** | 0.82 / 29.18 |
| queue violations | **6 → 0** (repaired at ~07:0x) | 13 → 0 |
| live queue rows | **50** (drain UNBOUNDED: 2.00 arrive / 0.86 dispose per cycle) | 49 |

**Goodhart check: there is nothing to check, and that is the finding.** The rate
did not fall and the count did not rise. The registry did not grow and the ladder
did not pass. Every number in the left column that moved, moved because of an
outage or because this desk touched it. **A flat rate on a flat registry is not
the ladder holding its ground; it is the ladder switched off.** One settle event
in twenty-four hours, against thirty-six the day before, on a board where
`run next` still reads **0 fresh of 44**.

---

## Part 1 — the state of progress, last 24 h only

### Produce, thrash, or stall? It produced for five hours and was then switched off

The builder's whole day fits between 06:37 and 12:07. In it: the `T1.08`
backend-confound probe **pre-registered in its own commit with no dispatch in
it** (`1652a62`), a deadline-leak in the detachment boundary found, written up as
a LESSON and fixed before any number landed (`e3eb376`, `4f3df24`), the kaggle arm
dispatched and harvested, the colab arm's download failure diagnosed as a *second
failure surface* and instrumented for it (`c249f82`, `521d33e`), the `FIELD_WATCH`
reader shipped, `T0.36` re-bought clean, and `T1.07` given the `IMPL_DEPS`
declaration that deliberately stales it. **That is a very good five hours.** Then
`2026-09-14T13:07` printed `PACING: ... skipping` and so has every slot since.

### The thing that was unread for nineteen hours, and it is today's real find

**The instrument the builder shipped at 11:17 worked, and it settled its question
on the very next failure.** The 09-14 LESSON named two causes for the colab
arm's lost artifact and deliberately refused to choose — *(1)* the job wrote
somewhere the fetch did not look, *(2)* the kept download session lost the run
VM — and said the distinguishing evidence was the `JACK_OUT` line at the *top* of
stdout, which a tail-only record had thrown away. The head capture went in. The
relaunch failed again at 11:22. Its record opens:

    stdout_head='JACK_OUT /content\nREPO 521d33e...'

**The job wrote to `/content`; `run_on_colab` fetched `/content`. Cause (1) is
eliminated on evidence and cause (2) is the cause.** The LESSON's own standing
instruction — *recover from stdout, never a fetch path* — stops being a judgement
call and becomes binding. A 400-character capture, added at zero cost, closed a
question a third GPU dispatch would not have closed. **And it sat on disk unread
until this sitting, because the organ that would have read it went dark
forty-five minutes later.**

Three things follow, all committed at `8c85e71`:

- **Disclosure, not a reading.** Seeds 2, 3 and 4 of the colab arm survive inside
  the truncated tail (`0.047148`, `0.098334`, `0.035367`), under a
  `mean_baseline` identical to the kaggle arm's — which is the arithmetic proof
  both arms ran the same job. Seeds 0 and 1 are lost, twice. **The pre-registered
  n=5 read may not be taken on a three-seed subset selected by what a truncation
  happened to preserve** — seed selection is the venue-selection prohibition of
  §2 wearing different clothes — **but whoever takes the branch will know three
  of its five numbers and must say so on the record.**
- **The overrun is named.** The probe was authorised at **1.20 GPU-h** and has
  charged **2.6716 h (2.2×)**, with **2.1109 h of it in a lane that returned
  nothing at all**. A third dispatch under the unchanged retrieval mechanism is
  **FORBIDDEN**.
- **What is authorised instead**, ~1.05 h: **(a) zero GPU first** — the job
  prints `JACKRESULT` to stdout and the *whole* stdout is captured, because the
  tail bound is what lost seeds 0 and 1; **(b) then** one dispatch. If (a) cannot
  be made to work, **the colab arm is ABANDONED** and the probe reports
  single-backend saying so. `MAX_HELDOUT_CV_PCT` 7.0 does not move on any branch.

### The frontier, recomputed

Unchanged in shape, one day older in every impl age: **`LT.01` FAIL frees 7**
(impl unchanged **14 d**), **`NE.01` FAIL frees 7** (**21 d**), **`UB.10` VOID
frees 4**, **`T1.08` FAIL frees 3 and blocks 45**. None of the four was touched,
because nothing could be. The one that *was* reachable — `T1.07`'s owed re-buy,
~0.47 GPU-h, the cheapest ladder-moving unit on the board — is unbought, and its
certificate is standing **deliberately staled** as a result.

### Effort vs. goal

Twenty-six commits. Six are this desk's, two are the overseer's corrections, and
the builder's eighteen served: the probe and its mechanics (ten), instruments and
their re-buys (three), certification prose (three), journals (two). **Zero served
the creature directly, and there was nothing legal it could have served.** The
honest accounting is that the week's effort bought *governance of a measurement*,
not a capability — and that was the correct trade on a board with 0 fresh units,
right up until the board stopped being the constraint and the meter became it.

---

## Part 2.5 — steering maintenance

**1. Priorities — replaced (`54f117d`), and the retraction this time is smaller
than yesterday's on purpose.** `1^7`'s four items were **not dead, they were
part-executed**, and calling them dead would have been the wrong correction: item
1's probe dispatched and half landed, item 2's declaration landed with its re-buy
owed, item 3 (`D19`) is now **OVERDUE**, item 4 stands. What changed is the
frame. `1^8` opens by telling the builder it is dark through no fault of its own,
then orders by **perishability rather than importance** — *assume this slot is
your last one this week; zero-GPU work keeps until Monday and 26.51 GPU-hours do
not*. Order: `D19`'s overdue default (minutes) → the `T1.08` stdout-carry repair
(§9d, zero-GPU step first) → `T1.07`'s owed re-buy. Four new prohibitions, three
guarding the colab lane from a third identical charge. `run steering`: **13
orders, 0 naming a spec the runner would refuse.**

**2. Field watch — nothing new.** wk7 landed 09-14 05:57 and was consumed in full
the same morning; `FIELD_WATCH.md` is unchanged since. Both its finding sections
read **ROUTED** in `run status` (`§6` → the `A4` row, `§6b` → the `ME.1` row),
**0 UNROUTED-FIELD-FINDING**. The reader the builder shipped for exactly this is
doing its job on its first full cycle.

**3. Seat staleness.** `champions --check` **EXIT 0**, and every number is
yesterday's plus a day. **`World` still declares no deciding run and no
`TRIGGER:` while held BY VERDICT — twelve days**, still the oldest seat finding
on the board and still nobody's dated row. `Learning core` holds 3 trigger debts
plus the `A4` diagnostic debt (routed, DUE 09-18, mine). `Fast/slow coupling`
stays welded behind `LC.03`. 4 seats remain unwinnable by construction —
**Episodic retrieval, Language grounding, Smell, Body schema** — and three of
those four are capabilities `GOAL.md` names by name.

**4. Organ liveness — three alive, one alive-but-gagged, and the distinction is
the whole of today's page.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | **last real iteration 09-14 12:07** | **process alive, output zero — 18 consecutive PACING skips.** Silence here is not idleness and not death; it is a gate firing correctly on the wrong meter |
| overseer | 6-hourly | 06:37 today (97th) | alive; its 12:37, 18:37 and 00:37 slots also paced out, only the `D15` clause-(c) exempt slot ran |
| field watch | Mondays | 09-14 05:57 (wk7) | alive, on cadence, consumed same morning |
| review | daily / Sun FULL | this run | alive; **not** pace-gated, which is why the queue dispositions below are mine and have no excuse attached |

**5. The queue, and the part I got wrong yesterday.** `review_queue_violations`
went **0 → 6** at midnight. Yesterday's sitting flattened the thirteen rows that
broke on 09-13 and **walked straight past the six standing on the day it was
writing**. All six are repaired (`cfd3bf1`), and for the first time the cause is
*measured* rather than confessed — three are builder-execution debt re-dated to
**after** the 09-21 meter reset, one (`d10-successor-rerun-under-adopted-gate`)
is **unreachable by construction** (`D1.0` BLOCKED ← `T1.08` FAIL: three dates now
set on a run no awake builder could have bought, and its stop-rule **re-parents**
rather than re-dates on a fourth break), and two are this desk's own, re-dated
once at the demonstrated ~1/cycle rate with no excuse offered.

**And two rows due TODAY were re-dated BEFORE they broke** (`1851448`), because
both are builder-execution debt and their owner is provably dark: leaving them on
today's date is *knowingly manufacturing* tomorrow's violation. **The four rows
still standing on today's date are this desk's decision debt and they stay there.
If they break at midnight the break is mine and it will be reported as mine.**
`review_queue --check` **EXIT 0, 0 violations, no amber date.** The drain is
still **UNBOUNDED** — 50 live, 2.00 arriving against 0.86 disposed per cycle —
and that is `D28`'s.

---

## The honest paragraph

No numbers. We are not closer and today we were not even busier; we were switched
off, and the switch was not ours. The thing worth saying is that the hours the
creature did get were spent well — a probe that fixed its own pre-registration
before it spent anything, a failure that was diagnosed instead of retried, an
instrument built to tell two indistinguishable causes apart that told them apart
on its very first opportunity — and then the whole apparatus fell silent with the
answer sitting on disk. That is the shape of this project's real risk, and it is
not the risk we spent August guarding against. We built an organ that cannot lie
to itself, and we hung it on a meter that anyone else's afternoon can empty. The
week's most important step toward Jack was the head capture: a creature is built
out of failures you can tell apart, and yesterday we bought the ability to tell
two of them apart for the price of four hundred characters. The most concerning
drift is that the same twenty-four hours proved the project's throughput is not a
function of its science, its budget, or its ladder — it is a function of a
calendar nobody in this repository can see. Everything else here is a system
correcting itself; that one is a system waiting to be told.

---

## FOR THE BUILDER

Ordered, and all of it is already in `ladder_prompt.md` `1^8`/`2^8`.

1. **Fire `D19`'s NO-FETCH default. It has been overdue since 00:00 today** and
   `run decisions` prints it `OVERDUE — DEFAULT IS DUE TO FIRE`. Minutes, zero
   GPU, lifts the hold on `HR.1` (frees 3). An armed default left unfired past
   its own date is a governance defect, not a backlog item.
2. **The `T1.08` colab repair, §9d, in two steps and step (a) spends nothing.**
   `JACKRESULT` on stdout + capture the **whole** stdout; *then* one dispatch,
   n=5, same commit, same seeds. **No fetch-path edit** — the head capture
   measured that surface innocent. **No third dispatch under the unchanged
   mechanism.** Abandoning the lane and reporting single-backend is a legitimate
   outcome and is cheaper than a fourth charge. Carry the three-seed disclosure.
3. **`T1.07`'s re-buy is owed and unbought** (~0.47 GPU-h, funded by hours that
   expire Saturday). The certificate is standing deliberately staled; that was
   the point of the declaration, and leaving it staled is not.
4. **Do not pre-empt the `A4` disposition (DUE 09-18) or the `T2.10` repair
   design.** Both are mine. Unchanged.
5. **When a slot pages out, say so in the journal.** This desk re-dated five of
   your rows today on measured cause. It can only keep doing that if the cause is
   in writing rather than inferred from a log tail at 06:40.
6. **`LT.01` (frees 7, impl unchanged 14 d) and `NE.01` (frees 7, 21 d) are still
   each larger free-standing unblocks than `T1.08`'s 3.** Named, not ordered —
   both are FAIL with no repair design and the design is this desk's debt.

---

## FOR THE OWNER

**1. `D30` — NEW, routed this morning (`1466035`), `decide_by` 2026-09-18, and it
is the only item on this page that you alone can settle.** The builder has been
dark for 18 consecutive hourly slots on a shared usage meter of which **75% (28
of 37 points) is not this project's spend**. The arithmetic is why this is urgent
rather than annoying: the pace line rises at a fixed **0.3869 points/hour** and
the meter is being consumed at **~1.45 points/hour** (~1.09 with our own share
removed), so **the gap widens on its own and the builder does not come back this
week by waiting**. `26.51 free GPU-hours expire Sat 2026-09-19` with a legal,
authorised buyer already waiting for them. W32 lost 8.82 h and W33 lost 22.11 h
to the identical shape. `D26`'s armed default chose MEASURE-ONLY twelve hours
before this blackout began — correctly, because a default may not loosen a gate —
and the measurement it commissioned has now come back and indicted the gate.
**The desk recommends against its own default, quoted verbatim on the entry:**
*"Rule (i): pace against this project's OWN attributed spend, not the shared
total — the 90% all-models hard stop is untouched by it, because `pace_gate` is
checked only after `usage_gate` has already said yes, so this changes which meter
the smoothing line reads and raises no ceiling anywhere."*

**2. `D28` — cited, not re-asked (`decide_by` 2026-09-21), and its amendment is
now two-for-two.** Yesterday I recommended amending option (a) to *"overdue first
UNLESS a perishable resource is the reason."* **Today I again did not spend my
first act on the overdue class** — I spent it reading the colab failure record,
because 26.51 perishable hours and a settled-but-unread diagnosis outranked six
rows that were already late. I would make that call a third time. (a) as written
would have made today worse, for the second consecutive day, and the amendment is
the only change I am asking for on that entry.

**3. `D19` is OVERDUE and its armed default has not fired**, because the organ
that fires it has been dark since before the date turned. Cited, not re-asked.
This is the first instrument-visible cost of `D30` and it will not be the last.

**4. NO-DECISION: liveness report, nothing here to rule on.** Three organs alive
and on cadence; the builder's process is alive and its output is zero, which is a
distinction no ratchet currently carries and which `D30`'s default would make
visible on day one instead of day two. Field watch wk7 is fully consumed, 0
UNROUTED-FIELD-FINDING. `champions --check` EXIT 0. `review_queue --check` EXIT 0.
`decisions --check` EXIT 0.

**5. NO-DECISION: the day's acts, declared because they are mine to do.** One
ruling addendum authorising a repair and forbidding a repeat spend; six broken
promises repaired with a measured cause; two more re-dated before they could
break and four deliberately left standing on today's date as my own bill; one new
owner decision routed; one steering block replaced. **No threshold moved in any
direction. No ledger row was written by hand. No spec was re-run.** `demonstrated`
moved by zero, and for once none of that is anybody's fault inside this repo.
