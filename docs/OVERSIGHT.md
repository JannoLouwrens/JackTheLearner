# OVERSIGHT — 73rd audit, 2026-09-05 06:37–07:2x UTC (at `4f5257b`)

## VERDICT: DRIFTING — the ledger is clean, and two guards this project built in the last 48 hours to watch its own routing behaviour both read GREEN through the exact act they were built to catch. `piled_on` counted **0 of the 4** rows that landed on one Sunday in one commit, and `fail_unowned` went **4 → 0 in six hours** without a single fact about Jack changing.

Say the clean part first, because it is large and it was checked mechanically.

**Section 1 is clean, for the second consecutive audit.** All **104** PASS rows
resolved. Every one has an implementation reachable through
`protocol.module_path_for` (**0 missing**); every `commit` field resolves in git
(**0 dangling**); every one declares a `control`; **102 of 104** wire it in
source, and the two that do not (`T0.01`, `T0.10`) declare
`"NONE, BY DECISION (52nd audit B5)"` in the registry, so the absence is
pre-registered rather than silent. Every PASS carries non-empty
`control_metrics`. `spec_drift` reports **0 SPEC_CHANGED** — no PASS is a
verdict about words that have since moved — against 11 rows that predate
`spec_sha` and are honestly reported as unknowable rather than back-filled.
Four impl-stale rows (`UB.10`, `T3.09`, `D1.0`, `LF.01`) plus one pre-`impl_sha`
content-stale row (`T2.02`); all five are settled reds held for Sunday's
dispositions, none is a PASS. **No finding.**

**Section 2 is clean — the sixth consecutive week.** Over the trailing 7 days
**139 commits** touched `registry.py`, `registry_expansion.py` or
`experiments/tests/`. I extracted every `CONST = number` that changed value:
**19 constants moved, 18 of them in the strengthening or neutral direction**
(`N_PROPERTIES` +1 seven times, `N_LIVES` 16→32, `N_EVAL` 48→120,
`N_DECISIONS` 3200→4800, `LIVES_PER_ARM` 4→16→48, `COORD_MIN` 0.55→0.70,
`COORD_MARGIN` 0.20→0.35, `STEPS` 300→500, `TEMP` 0.25→1.0). **One moved
downward**: `DECAY_MIN` 1.5 → 1.25 (`44f24c4`, 2026-08-29, `T2.09`) — the same
one the 72nd audit read in full; it is a `PILOT`-marked placeholder being frozen
for the first time from disjoint seeds 7/90, declared in the commit under its
own heading, with seven sibling bars confirmed unmoved. `LG.10`'s `TEMP`
0.25→1.0 is the only other knob I re-read this window: it is argued as
claim-harder on every gate, the variety floor and every threshold are
byte-identical to v1, and the VOID that forced it stays in the ledger's
history. No control was deleted, no `_check` gained an `or`, no seed count was
reduced, no assertion was removed. **No finding.**

**Section 4 is clean.** 25 iterations started in the window, **25 ended rc=0**,
zero `PACING:` skips (last skip 08-29). 62 commits, 12 of them journal records.
No repeated identical failures, no paused loop, no credit pressure
(`week:all models` 8% at the last read). The only dead-process event in the
window was the 21:07 SO.07 pilot, which the builder's own inheritance audit
caught inside one slot and paid for in the open.

Now the findings, ranked by how much they damage the trustworthiness of the
map. **None of them is a false PASS.**

---

## 1. `piled_on` IS BLIND TO BATCH ROUTING. FOUR ROWS LANDED ON ONE SUNDAY IN ONE COMMIT AND THE INSTRUMENT COUNTED ZERO OF THEM — INCLUDING THE `XL.01` ROW THE WHOLE EXERCISE WAS FOR.

The 68th audit built `piled_on` because *"moving between piles is not a repair"*
and the pile line reports a symptom after the fact. Its own comment states the
definition precisely:

> *A row is `piled_on` when at least CAPACITY other live rows were ALREADY
> promised on its date at the moment it was routed.*

**Yesterday it failed that definition on its first live test.** At 01:19,
`52dcf9e` routed four rows — `xl01-death-and-retry-has-no-reachable-repair-path`,
`t205-world-model-loses-to-the-ridge-reference`,
`t402-touch-drowns-audio-at-the-fusion-boundary`,
`t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall` —
all with `DUE: 2026-09-13`, all in one commit, using **one** `next_free_due`
lookup **four times**. When row 2 was written, row 1 was already promised there.
When row 4 was written, three were.

**Measured, not argued** — `experiments.review_queue.audit()` run against the
**committed** `docs/REVIEW_QUEUE.md` at HEAD `4f5257b`, `today=2026-09-05`, so
the numbers are reproducible (the Review's DAILY was writing the live file while
I audited; see finding 7):

| | |
|---|---|
| `due_pile['2026-09-13']` before `52dcf9e` | **0** |
| `due_pile['2026-09-13']` now | **4** (AMBER, against `MEASURED_DISCHARGE_CAPACITY = 1`) |
| of those four, in `piled_on` | **0** — where the tool's own definition gives **3** |
| `piled_on` reported | **17** — the identical number PROGRESS recorded on 09-04 |
| `piled_on` under the tool's own stated definition | **22** |
| rows the strict test misses | **5** (23% of the true class) |

(The fourth 09-13 row, `xl01-*`, correctly does **not** fire: it was written
first, onto a genuinely empty date. It is the three written after it that the
instrument cannot see.)

**The mechanism, at `experiments/review_queue.py:499`:**

```python
prior = sum(1 for o in live
            if o is not r and o["due"] == r["due"]
            and o["routed"] is not None and o["routed"] < r["routed"])
```

`routed` is parsed at **day** granularity and the comparison is **strict**. Any
number of rows routed on the same calendar day onto the same date therefore all
compute `prior = 0` and none of them fires. The docstring says the instrument is
*"CONSERVATIVE BY CONSTRUCTION"* and justifies it for a re-armed row whose `DUE:`
was chosen later than its `routed` date — that reasoning is sound and it does not
cover this case at all. **Same-day co-routing is not conservatism; it is a
blind spot, and it is the only routing pattern that can add N rows to a date in
one motion.**

**This is not a one-off.** Six live due-dates were fed by same-day batches:

| due | day | routed | rows | uncounted |
|---|---|---|---|---|
| 2026-09-06 | **Sun** | 2026-09-01 | 4 | 3 |
| 2026-09-07 | Mon | 2026-08-30 | 3 | 2 |
| 2026-09-08 | Tue | 2026-09-02 | 3 | 2 |
| 2026-09-09 | Wed | 2026-08-31 | 2 | 1 |
| 2026-09-09 | Wed | 2026-09-03 | 2 | 1 |
| 2026-09-13 | **Sun** | 2026-09-05 | 4 | 3 |

The 09-06 Sunday pile — the one the 68th audit hand-staggered twice and the
Review then forbade a third pass on — was itself built by a four-row batch on
09-01 that `piled_on` never counted. **`REVIEW_QUEUE.md`'s own written rule is
*"do not re-pile them onto a Sunday"*, and both Sundays on this board were
loaded by an invisible batch.**

**Why this is rank 1.** Two audits ago this project decided, correctly, that the
honest response to a diverging queue is *a number, not a gate* — and then wrote
a number that under-reports the act by 23% and reports **zero** for the largest
batch ever routed. The 09-13 pile now holds `XL.01` — *"death does not erase
what he learned"*, GOAL.md's own page-turn sentence — on a Sunday already
carrying three siblings, in front of a desk whose own reader prints
`drain UNBOUNDED`. Nobody lied; the builder used the tool exactly as instructed
and the tool told it the date was free.

**The repair is three lines and it is not a gate** (see FOR THE BUILDER B1):
rows are parsed in document order and the file is append-ordered, so the row's
index in `audit()`'s `rows` list is the tie-break the timestamp cannot supply.
Ratchet the corrected number, do not re-date anything to make it smaller.

---

## 2. `fail_unowned` WENT 4 → 0 IN SIX HOURS, AND `run status` NOW PRINTS "AT floor — ok" OVER THREE CLAIMS WHOSE REPAIR PATH IS STILL UNREACHABLE

Yesterday I reported three settled FAILs — `XL.01` (17 d), `T2.05` (16 d),
`T4.02` (15 d) — with no owner, no clock and no queue row. The builder shipped
the class (`6fbac74`), found a fourth I had missed (`T2.15`, honest baseline 4
not 3), routed all four (`52dcf9e`), lowered the baseline to 0, and re-bought
`T0.21` behind it. **Every step was ordered by my own B1/B4 and every step was
done properly.** The finding is not about conduct.

**The finding is that the number is now at floor and nothing measured changed.**

- `FAIL_UNOWNED_BASELINE = 0`; `run status` prints
  `fail_unowned = 0 … vs declared floor 0: AT floor — ok`.
- The detector's own docstring is candid: ownership is *"deliberately
  generous — any appearance of the id in the queue doc counts"*, and
  *"a prose mention is a weak owner… weak ownership is a routing-quality
  question for the Review, not a hole this counter can see."*
- The desk that now owns all four: **37 OPEN, 2 HELD, 39 live of 41 routed;
  trailing 7 days arrived 36, disposed 1; `drain UNBOUNDED`; oldest live row
  12 days.** One row left the live set in a week.
- `XL.01`'s repair path is unchanged from yesterday: `NE.08` is
  `blocked<-NE.01`, and `NE.01` is itself a settled FAIL that `run blocked`
  ranks second in the project (frees 8, impl unchanged 11 d).

So the state today is: **`coverage` prints green on the disposition axis for
four negatives whose repair is behind an unbounded queue.** That is a milder
version of yesterday's disease, not its cure — the difference is that yesterday
nothing named them and today a row names them. That is worth something. It is
not worth a floor.

I am not asking for the baseline to be raised (it may not be, and should not
be). I am asking that the counter stop reading as *repaired* when it means
*routed* — see FOR THE BUILDER B2.

---

## 3. `CITED-BUT-UNRUNNABLE` MORE THAN DOUBLED (3 → 7) UNDER AN EXIT CODE THAT WAS ALREADY RED, AND IT IS THE ONE STANDING RED WITH NO RATCHET COUNTER

`coverage` exits **2** this morning. Two red conditions are live:
`claim_dead` (4: smell, balance, shelter/building, thermal) and
`new_unrunnable_citation` (**4**: `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all
`welded<-LC.07` since `LC.07` went PILOT-BLOCKED on 09-01).
`GOAL_UNRUNNABLE_BASELINE` is `{DP.02, DP.03, LC.04}` — so the class went
**3 → 7**, a 133% growth, and it was widened into the
`goal-cites-four-specs-that-resolve-to-corpses` row on 09-04 (DUE 09-10).
All of that is correct and routed.

**What is not covered:** `claim_dead` has been non-empty continuously, so
`coverage` has been exiting 2 for days and the new red **changed no exit code**.
The 64th audit built the `RATCHET COUNTERS` block in `run status` for exactly
this — *"standing-red tools' numbers, printed here so a blessed red can never
silence them"*. It carries nine counters. **None of them is the citation
class.** So the 3 → 7 growth is visible only in `coverage`'s prose, and a
future 7 → 12 would be equally invisible to every committed reading.

This is cheap: one counter, one recorded reading. It is a line in an existing
block, not another organ — the Review's *"no third increment of the CPU
accountant"* prohibition is about building new machinery, and this is not that.

---

## 4. SECTION 3 — DRIFT: NOTHING TRACED TO NO GOAL SENTENCE, AND NOTHING DEMONSTRATED A CLAIM ABOUT JACK

**Every non-journal commit in the window traces to GOAL.md.** 50 non-journal
commits: 19 are creature or science work (`OWNERS_HANDS.md` research; `SO.06`–
`SO.09` registration, implementation and runs — *"their hands may leave things
in his world for him to find… his diary records who left it"*; `LG.03`
implementation and its VOID — *"the LLM is his TALKATIVE PARENT"*;
`LANGUAGE_GROUNDING.md` §2.2–§11; `SO.08` implementation), 31 are audit items,
certificate re-buys, renders, ratchet readings and lessons. **Maintenance share
62%, down from 73% yesterday and 79% the day before** — three consecutive falls,
and that is a real improvement.

**But the output side is blunt.** `demonstrated` 102 → **104**, and:

| settled in the window | kind | credited by `coverage`? |
|---|---|---|
| `SO.06` PASS — a hand reaches only through the world | `COVERS: social/other agents (fixture)` | **no — support** |
| `SO.09` PASS — a life the hands bought is not evidence | `COVERS: social/other agents (rule)` | **no — support** |
| `LG.03` VOID — its own blind-twin liveness gate | `COVERS: language (parent) (fixture)` | n/a |
| `SO.07` VOID — the reference lane on recording worlds | `COVERS: social/other agents (claim)` | n/a |

**Two PASSes, zero claims.** Both are support furniture — good, necessary
furniture, and `coverage` says so itself without being asked
(*"support passing, not credited: SO.06 (fixture), SO.09 (rule)"*). The two runs
that *were* about Jack both returned VOID. That is not failure — a VOID is a
measurement, `SO.07`'s in particular is a good one (fixture constants frozen on
world 0 did not transfer to recording worlds 3–5, and the builder made that
lesson a rig gate in `SO.08` the same night) — but it means **the last
credited claim about Jack is `LG.02`, 2026-09-02, three days ago**, and the
09-03 trio (`SO.02`, `SO.04`, `LF.02`) remains the last cluster.

**The converse question, and it has not moved.** `T2.01 = FAIL, frees 35,
blocks 38, impl unchanged 26 days.` Curiosity: 12 specs, 2 pass. Fast/slow:
8 specs, **0 pass**, five of them welded behind `LC.03`. All-senses fusion
(`one brain / unison`): 25 specs, **1 pass**. Sleep: 5 specs, 0 pass. Those are
the four things GOAL.md says the project is, and three of them have nothing.

---

## 5. SECTION 5 — COMPUTE HONESTY: THE CPU METER'S SECOND DAY IS THE OPPOSITE OF ITS FIRST, AND 84% OF W35's GPU HOURS BOUGHT A ROW THAT IS NOW STALE

**CPU, and this is good news the Review should have on Sunday.** Yesterday's
page reported the day-meter's first full day billing 5,906.8 s across eight line
items with *"not one second bought a new measurement about Jack"*. Today:

```
2026-09-05   SO.07  9,205.09 s   registered claim run (VOID)
             T0.28     66.61 s   re-buy
             T0.21      9.15 s   re-buy
                    ─────────
                     9,280.85 s of 57,600 s   →  99.2% to one registered claim run
```

The `pace_gate`-shaped worry stands (`D20`, decide_by 09-18), but the second
day's composition refutes the first day's reading rather than confirming it.
Worth one line on the Sunday page.

**GPU, W35 (expires 2026-09-06 00:00): 19.20 h of 30 spent, 10.80 h expiring.**
Attributed against ledger rows:

| hours | job → row | verdict |
|---|---|---|
| **16.17** | `D1.0` ×3 kernels | **VOID** — and the row is **impl-STALE** |
| 1.01 | `T2.14` | PASS |
| 0.30 | `UB.10` | VOID |
| 0.10 | `T1.09`, `T1.10` | PASS (re-buys) |
| 0.94 | `D1.0` throughput pilot (0.50) + `LC.07` pilot (0.44) | named in commits, no row |
| 0.67 | 3 jobs | **no ledger row and no named consumer** |

**16.5 of 19.2 GPU-hours (86%) bought VOIDs.** That is legal — a VOID is a
measurement and both are routed — but note the timing on the big one: `D1.0`
ran at **2026-09-01 18:23**; its test file was edited at **20:22 the same
evening** (`112cf3b`, 59th audit B3–B7). So the project's largest recent single
GPU expenditure now sits behind `STALE CLAIMS — the test changed after the run
that recorded it`, and re-reading it costs 16.17 GPU-hours. The disposition is
owed at Sunday's FULL. **Letting W35's 10.80 h expire is right** — every
runnable GPU spec is a settled FAIL whose re-run is a seed lottery, or parked —
and I endorse the Review's item 5 unchanged. This is inventory, not uptime.

---

## 6. SECTIONS 6 & 7 — DECISIONS AND BAKEOFF HYGIENE: NO FINDING

`decisions --check` is rc 0: **10 armed, 0 UNDECLARED, 0 MEANS-ESCALATED,
0 OVERDUE, 0 VANISHED-OWNER-ASK.** Nothing is on the owner's desk that a
measurement could settle, so there is nothing for me to arm this audit and
nothing to seize under rule 3. `D15`, `D16` and `D21` all carry
`decide_by 2026-09-05` and fire tomorrow; `D21` prints the
`DEFAULT-ACTION-SAME-DAY` reading the 72nd audit's B2 shipped, and the builder's
handoff correctly orders the 00:07 slot to fire all three ahead of the 06:37
Review, with 00:07–05:07 as the fallback window. The one tolerated
`UNROUTED-OWNER-ASK` (`PROGRESS #4`, the organ-liveness paragraph) is the
Review's own to annotate `NO-DECISION:`; the builder was right to leave it.

`DECISIONS_RESOLVED.md`: I re-read the fourteen armed-default resolutions of
09-01 and `D10`. No decision was made without a learning gate; `D10`'s VOID is
recorded **as** a VOID and its single-arm caveat is on the seat's face in
`CHAMPIONS.md` rather than laundered into a verdict; no winner was chosen inside
a noise margin. `champions --check` is rc 0 with **0 ARENA-MISSING** — the eight
phantom arenas in my standing orders, including the whole `W.1`–`W.8` family,
are registered specs now. The two `UNVERIFIED VERDICTS` (`Learning core` =
`LC.03` VOID; `World` = no deciding run named) and the three `TRIGGER DEBT`
seats are unchanged and at their declared baselines. **No finding.**

---

## 7. A SMALL STRUCTURAL ONE: THIS AUDIT SLOT CAN NEVER READ THE DAY'S REVIEW

`crontab`: `37 6 * * * review.sh` and `37 */6 * * * overseer.sh`. The 06:37
overseer slot and the daily Review start **in the same minute** — I confirmed it
live (`review.log` and `overseer.log` both mtime `06:37:04`, review pid 1430355
running while I read). So `docs/PROGRESS.md` at this slot is always **yesterday's
page**; I read the 09-04 DAILY, exactly as the 00:37 audit did six hours ago.
Three of four audit slots see a fresh page and this one structurally cannot.

**It also moved the file I was measuring, mid-audit.** At 06:44 the Review
rewrote `docs/REVIEW_QUEUE.md` underneath me — `champions-language-grounding-
arena` went `OPEN → ACTED` (`ARENA: NONE → LG.04, LG.05, LG.06`, two days
early, `UNFALSIFIABLE` **3 → 2**: good work) and `t027-preserved-failimpl-as-
artifact` was legally re-armed `DUE 09-05 → 09-07` with its reason stated. My
first `review-queue` read at 06:38 gave `piled_on = 17`; a re-read at 06:47 gave
16, for no reason but the concurrent write. **Every queue number in this report
is therefore taken from the committed revision at HEAD `4f5257b`, not from the
working tree**, and I have left the Review's five uncommitted files untouched.
This is the read-side twin of `cross-organ-doc-race-voids-certificates` (DUE
09-06), whose fork (a)/(b)/(c) is on Sunday's docket: option (b), *"serialise
organ commits"*, would fix both halves at once, and the crontab minute below is
the zero-cost approximation of it.

Given that my standing orders open with the story of a Review recommendation
that died unread in 24 hours, that is worth one minute of crontab. It is
distinct from `cross-organ-doc-race-voids-certificates` (DUE 09-06), which is
about concurrent *writes* dirtying the tree, not about read ordering. `crontab`
is outside the repo and outside what I may touch — FOR THE OWNER item 2.

---

## FOR THE BUILDER

**B1 (rank 1). Make `piled_on` see a batch.** In
`experiments/review_queue.py:499`, `prior` compares `o["routed"] < r["routed"]`
on a day-granularity date, so N rows routed on one day onto one date all read
`prior = 0`. `audit()` parses rows in document order and the file is
append-ordered, so the row's **index in `rows`** is the tie-break the timestamp
cannot supply: count `o` as prior when
`(o["routed"], o_index) < (r["routed"], r_index)`. Verified against HEAD
`4f5257b`: this moves the reported number **17 → 22**, adding exactly
`t205-world-model-loses-to-the-ridge-reference`,
`t402-touch-drowns-audio-at-the-fusion-boundary`,
`t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall`,
`sm03-heldout-split-saturated` and
`pl02-dependency-on-pl00-verdict-vs-table` — and correctly leaving `xl01-*` out,
since it was the first row onto an empty date. **Three rules on this repair:**
1. It stays a **METRIC, never a gate** — the 68th audit's discipline is right
   and a gate here would forbid a legal move.
2. **Ratchet the corrected number and record it.** 22 is the honest baseline;
   17 was never a real reading. Same shape as `fail_unowned`'s baseline 4.
3. **Do not re-date any row to make the number smaller.** The Review has
   forbidden a third hand-stagger of 09-06 and I endorse that; 09-13 gets the
   same protection. Moving between piles is not a repair.
Add the falsifier to `T0.31`'s property set (its P4/P5/P6 precedent is exactly
this: assert on the TOTAL, and assert that a same-day batch fires).

**B2 (rank 2). `fail_unowned = 0` must not print as `AT floor — ok` when every
member of the class is owned by a prose mention on an unbounded queue.** Do
**not** raise the baseline — it is shrink-only and 0 is correct for what it
measures. Instead make the strength of ownership readable, since the detector's
own docstring already names the three strong forms and the one weak one: report
alongside the count a breakdown of how each settled FAIL is owned —
`repaired_by` / `FAIL-DISPOSED:` / a queue row **with a `DUE:`** / a bare prose
mention — so `run status` prints e.g. `fail_unowned = 0 (owned: 0 repaired_by,
1 disposed, 4 queue-row, 0 mention-only)`. The number stays at floor; the map
stops implying repair. Cheap, no new organ, and it is the reading the Review
needs on Sunday when it decides whether `xl01-*` can wait until 09-13.

**B3. Give the citation class a ratchet counter.** `new_unrunnable_citation` is
a RED condition in `coverage.exit_code` whose class grew 3 → 7 under an exit
code already held red by `claim_dead`, and `run status`'s `RATCHET COUNTERS`
block — built so a blessed red cannot silence a number — has no line for it.
Add `goal_unrunnable` to the counters and record today's reading (7). One line
in an existing block; this is not a third increment of anything.

**B4. When you consume `next_free_due`, consume it once.** `52dcf9e` used a
single lookup for four rows. The tool answers *"the next date carrying fewer
than CAPACITY live rows"* — after you write the first row, that answer is stale.
Re-read it per row, or say in the commit that you deliberately batched and why.
This costs nothing and it is the behavioural half of B1.

**B5. Standing prohibitions, carried forward unchanged and all still live:** do
not re-dispatch `D1.0` (gate design owed at Sunday's FULL; an unchanged re-run
is a 16.17 GPU-hour seed-lottery redraw); `HR.1`–`HR.4` stay `D19`-held to
09-14, no corpus fetch; `HR.6` stays blocked behind `HR.5`; `LF.01` attempt 2
waits for the 09-09 design; no third increment of the CPU accountant; let
`W35`'s 10.80 Kaggle hours expire tonight. Fire `D21`/`D15`/`D16` at the 00:07
slot before the 06:37 Review, then run `SO.08` before ~3,600 s of other billing
accrues.

---

## FOR THE OWNER

**1. `D22` is the decision everything now routes through, and its default fires
on 2026-09-08 as "the rule stands".** I am not re-arguing the Review's
recommendation — it made the case on 09-04 and you have it. I am adding the
measurement it asked for, taken a day later, because the numbers moved in one
direction only:

| | 09-04 (Review) | 09-05 (this audit) |
|---|---|---|
| live queue rows | 33 | **39** |
| routed / disposed, trailing 7 d | 30 / 1 | **36 / 1** |
| drain | UNBOUNDED | UNBOUNDED |
| startable specs behind the desk | 9 | **10** |
| dates carrying more than the measured capacity of 1/cycle | 6 | **7** |

Six more rows arrived, one left, and `XL.01` — *"death does not erase what he
learned"* — joined the back of it yesterday with a promise dated 09-13, a Sunday
already carrying three siblings. If `D22` defaults on 09-08, that promise is
made against a desk that has kept 1 of the last 36. **The default is the only
legal one** (option (iii) would widen what the builder may do, and a default may
not widen), so silence here is not neutral: it is a choice to let the divergence
continue for at least another five days. That is the whole content of my item —
the fork is yours, the price is now measured twice.

**2. One minute of crontab: move the overseer off the Review's start minute.**
`37 6 * * * review.sh` and `37 */6 * * * overseer.sh` fire in the same minute, so
the 06:37 audit can never read the day's `PROGRESS.md` — I read yesterday's page
this morning while the Review was still running, and so did the 00:37 audit. My
own standing orders exist because a Review recommendation died unread in 24
hours. Changing `overseer.sh` to `47 */6 * * *` (or `review.sh` to `27 6`) fixes
it. `crontab` is outside the repo and outside what any agent here may touch, so
it needs your hand. **No urgency, no risk, and no agent can do it.**

**3. The honest paragraph, and it is the answer to the audit's last question.**
Are we closer to a curious humanoid that climbs the ladder than we were
yesterday? **Marginally, and less than the +2 suggests.** The ladder gained two
PASSes in 24 hours and `coverage` credits neither as a claim about Jack — one is
a fixture proving the owner's hand reaches only through the world, one is an
accountant proving a life the hand bought is not evidence. Both are exactly the
right things to build before the claim they support, and building them in that
order is the discipline working. The two runs that *were* claims about Jack both
came back VOID, and one of them (`SO.07`) taught a real lesson that was inside
the next spec before the night ended. So: an honest day's work, an empty
scoreboard, and the last credited claim about Jack is now three days old while
`T2.01` sits FAILED and unmoved for 26 days in front of 38 specs.

What I want on the record is the *shape*, because it is the third audit running
to find it. This project is extraordinarily good at building instruments to
watch itself, and yesterday it built two — one in the morning at my own order —
and **both of them read green through the exact behaviour they were built to
catch.** `piled_on` counted zero of four. `fail_unowned` went to floor in six
hours. Neither is dishonest and neither cost a certificate. But an instrument
that cannot see the act it was built for is worse than no instrument, because it
converts a visible problem into a green number, and this repository now has
enough green numbers that a real hole can hide behind one. The counterweight is
not more instruments. It is that every new one gets a falsifier that fires on
the *class*, not on the tidy example — which is `T0.31`'s P4/P5/P6 rule, already
written down, and the reason B1 and B2 above ask for falsifiers rather than for
counts.
