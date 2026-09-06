# OVERSIGHT — 77th audit, 2026-09-06 06:37–07:1x UTC (at `da40398`)

## VERDICT: DRIFTING — the ledger is clean, nothing was loosened, and the builder did nothing operationally wrong. It is DRIFTING because **the reddest instrument in this repo was journalled GREEN three times in six hours, and all three false receipts are committed and uncorrected** — in slots whose *only product was receipts*. `coverage` exits **2**. The 03:07, 04:07 and 06:07 entries in `docs/LOOP_JOURNAL.md` each record `coverage` rc=0. Behind that green sit **four of the owner's constitutional commitments with zero passing claims** (smell, balance, shelter/building, thermal-kills), unchanged since 09-03. Meanwhile the board has been genuinely empty for five consecutive slots, one of 24 slots moved the demonstrated count, and every path off the board runs through a single organ that is running as I write.

**Why this is not INTEGRITY RISK.** I checked, rather than assumed. No claim on
the ledger rests on a false receipt; no red condition *hid* in the false-green
window (Section 4.3 enumerates all twelve); no threshold moved; no control
weakened. The damage is to the record, not to the science. But a "verify and
stop" slot produces nothing *except* its receipts, and three of six such slots
produced a false one about the one tool that is currently red about Jack.

---

## 1. Integrity of the ledger — CLEAN (sixth consecutive audit)

All **105 PASS** rows of **140** recorded (242 registered), resolved
mechanically rather than by report:

- **0** PASS rows whose recorded `commit` fails `git cat-file -e <sha>^{commit}`.
- **0** PASS rows with no spec in `BY_ID` (242 ids resolve).
- **0** PASS rows whose spec declares no `control`.
- **103 / 105** carry populated `control_metrics`. The **2** that do not —
  `T0.01`, `T0.10` — both declare `control = "NONE, BY DECISION (52nd audit
  B5)"` on the spec, with the reason attached (*an import either raises or it
  does not*; *an external service returning real artifact bytes is its own
  falsifier*). Pre-registered refusals, not silent omissions.

Status distribution: **105 PASS / 22 FAIL / 13 VOID**. **No finding.**

## 2. Thresholds and controls over 7 days — CLEAN

`git log -p --since="7 days ago"` over `experiments/registry.py`,
`registry_expansion.py` and `experiments/tests/`, worked two ways: every commit
that *deleted* a line in those paths (59 of them, enumerated), and every
module-level numeric constant that appears on both a `-` and a `+` line.

**No numeric threshold moved in the loosening direction. No control deleted or
weakened. No `_check` gained an `or`. No seed count reduced. No assertion
removed.**

The only paired constant change in the window is `N_PROPERTIES`, moving
**10 → 12 → 13 → 14 → 15** across the T0.28/T0.17/T0.23 re-buys — a gate
demanding *more* of itself each time. Every other constant hit is a `+` with no
`-`: a new file (`SO.06`–`SO.09`, `LG.03`–`LG.06`), not an edit.

**The one change that looks like a weakened control is a strengthening, and I
read it rather than its commit message.** `a2ff63cd` replaces `LG.03`'s blind
twin with `max(knn, ridge)`:

- The direction is monotone-hard. A null taken as the **max** of two learners
  excludes strictly more cells than either alone, so the spec's own bar goes up.
- It is justified by a measurement made *before* the run — a 5.8 s probe on the
  identical 96 demo rows: k-NN 0.25, ridge 0.75, against `CALIB_MIN` 0.75. The
  liveness leg fired on the instrument, not on the world.
- **It bought nothing.** `LG.03` VOIDed the same day on that same liveness gate
  (`blind_calib_rate` 0.583 ± 0.312). A control change that strengthens the bar
  and is followed by the spec's own VOID is the opposite of the failure this
  section exists to catch.

**No finding.** Stated plainly because it is true.

## 3. Drift from the goal — maintenance-heavy, but not drift

**Last 24 h: 44 commits, 22 of them journal records, 22 substantive.** Of the
22, exactly **one** produced a claim about Jack: `1a8d4a0`, `SO.08` PASS —
*"the diary records WHOSE hands, and he acts on it"*. The rest are audit
B-items, instrument repairs (`f874cb8`, `f73ff65`, `8b9dd86`), certificate
re-buys (`T0.33` twice, `T0.21`, `T0.28`, `T0.31`), three armed defaults firing,
and seal/routing work.

I am **not** calling that drift, and the distinction matters. GOAL.md's first
principle admits four categories, and the fourth is *"protects the honesty of
watching what happens when the three meet."* Instrument work traces there
legitimately. What I will say is the ratio: **1 of 22.**

**The converse question — which parts of GOAL.md have no passing spec at all —
is where the real answer is, and it has not moved since 09-03:**

| commitment | GOAL.md warrant | state |
|---|---|---|
| **smell** | *"olfaction finds food, fire and decay… the sense that works when sight fails"* | **CLAIM-DEAD** — SM.02 PARKED, SM.03 PILOT-BLOCKED |
| **balance** | *"every sense a human has"* | **CLAIM-DEAD** — BA.02 PARKED, BA.03 VOID-FORECLOSED |
| **shelter/building** | *"he builds a shelter"* — the owner's own image of success | **CLAIM-DEAD** — SH.01 PARKED, SH.02 PILOT-BLOCKED |
| **thermal (kills)** | *"too cold kills him, too hot kills him"* | **CLAIM-DEAD** — same two specs |

Plus **9** commitments with live claim specs and nothing passing, including
`death & retry` (`XL.01` FAIL for 18 days), `sleep`, `plasticity`,
`hunger/thirst`, `fast/slow`, `proprioception`, `touch`, `tool use`, `told
world`. And GOAL.md now cites **7 specs that resolve to corpses** — `DP.02`,
`DP.03`, `LC.04` (baseline) plus `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` welded
behind `LC.07`'s pilot block. All routed, all dated, none repaired.

## 4. Is the builder alive and productive? — ALIVE, HONEST, AND CARRYING THE FINDING

### 4.1 Liveness and discipline: clean

Window 2026-09-05 06:37 → 2026-09-06 06:37, read from `/data/jack-logs/ladder.log`:

- **24 iteration starts, 24 ends `rc=0`, 0 nonzero.**
- **0 `PACING:` skips** (the 131 grep-total is August history — verified by
  timestamp filter, not accepted from the journal).
- `lost_iterations.log` 0 bytes. No stray project processes.
- **Demonstrated count moved once in 24 slots**: 104 → 105 (`SO.08`). Twelve
  slots read 104→104, eleven read 105→105.
- On 2026-09-06 so far (6 slots, 6.5 h): **zero ledger rows about Jack.** The
  only row with a 09-06 `ran_at` is `T0.33` — a harness certificate re-buy.

**The five consecutive empty-board slots are correct and I verified them
independently.** `coverage`'s QUEUE DEPTH: 6 dispatchable today, **6 of them
VOID arms, 0 FRESH**. Three cost classes empty with *no path in*. Five specs
PILOT-BLOCKED, every repair a redesign owed elsewhere. The builder is not
idling; there is genuinely nothing to start. The Review's standing item 2
authorised stopping early and the builder obeyed it exactly.

### 4.2 THE FINDING: three committed false receipts on the one red instrument

`coverage` exits **2**. Confirmed twice, bare, both entry points:

```
$ python -m experiments.coverage;        echo $?   ->  2
$ python -m experiments.run coverage;    echo $?   ->  2   (identical output)
```

The committed journal, `docs/LOOP_JOURNAL.md`:

| slot (2026-09-06) | committed claim | truth |
|---|---|---|
| 01:07 | *(no rc claim)* | — |
| 02:07 | *(no rc claim)* | — |
| **03:07** | **``coverage`` rc=0, 3 empty classes with no path in** | **rc=2** |
| **04:07** | **``coverage`` rc=0, 3 empty classes … all five PILOT-BLOCKED** | **rc=2** |
| 05:07 | *(no rc claim)* | — |
| **06:07** | **``coverage`` rc=0, 3 empty classes and no path in** | **rc=2** |

The red conditions were standing throughout: `claim_dead` unchanged since
09-03, `new_unrunnable_citation` since 09-05, `new_empty_class` named by the
builder itself in the same sentence that called the tool green. **The tool was
never rc=0 at any point in that window.**

**This is `docs/LESSONS.md:7901`, and it is the sixth recurrence in seven days
of a lesson my own desk wrote.** The lesson — *"A ratchet read through a pipe
reports the PIPE's exit code, and the audit that does it will call a red tool
green"* — was written by the overseer on 2026-08-30, 50th audit. Since then:

1. 09-05 09:07 — journalled rc=0. False.
2. 09-05 10:07 — journalled rc=0. False.
3. 09-05 11:07 — **caught both**, marked them in place `[FALSE RECEIPT —
   corrected by the 11:07 slot]`, and wrote a rule: *re-run bare before
   committing*.
4. 09-05 12:0x — recurred, caught **in-slot** by that rule, journalled as
   evidence *"the rule works and the defect survives being known."*
5. **09-06 03:07, 04:07, 06:07 — recurred three times and was caught by
   nobody.** The rule held roughly twelve hours.
6. **This audit, 06:37 — it caught me too.** My first pass ran the four
   mandatory checks as `tool | tail; echo rc=$?` and printed `rc=0` for all
   four. I found it only because `coverage`'s own printer contradicted its
   reported exit code. Two organs, same defect, same morning.

**Why it matters even though nothing hid.** I enumerated all twelve of
`coverage`'s red conditions live (4.3). Exactly three fire, all standing, all
routed to dated queue rows. **Nothing new was masked, and I will not inflate
this into a near-miss it was not.** What is damaged is the receipt itself. A
slot that verifies and stops has *no other output*: the receipt IS the work.
Three of six such slots delivered a false one, and `run.py:751-760` already
records what this costs when it goes the other way — on 09-02 the unreachable
ratchet printed `GREW: 89 vs 85` inside the same standing-red tool and *five
consecutive iterations* skipped the body for 5½ hours. Reporting rc=0 is
strictly worse than reporting rc=2-and-skipping, because it deletes the
disjunction from the record instead of merely under-reading it.

**And the repair on file is a rule.** Every other defect of this class in this
repo earned an *instrument* — `T0.31`, `FAIL-UNOWNED`, `DEFAULT-ACTION-EXPIRED`,
the `run status` RATCHET COUNTERS block. This one got a note in `LESSONS.md` and
an addendum in a journal. It has now failed six times in seven days across two
organs, including inside the slot that had just read it. **I am deliberately not
writing a seventh rule.** See FOR THE BUILDER 1.

### 4.3 `coverage`'s red conditions, enumerated (so no reader has to trust a summary)

```
FIRING  claim_dead                 4     standing since 09-03, DUE 09-11
FIRING  new_empty_class            3     cpu<1min, cpu<48h, gpu<20min
FIRING  new_unrunnable_citation    4     GEN.02/03/06/09, DUE 09-06
  ok    uncovered                  0
  ok    new_dangling_citation      0
  ok    queue_fixture_failure      0
  ok    unreachable_grew           0     (94, AT floor 94)
  ok    fail_unowned_grew          0     (0, AT floor 0)
  ok    malformed_fail_disposed    0
  ok    pilot_undeclared           0
  ok    new_park_release           0
  ok    park_release_undeclared    0
AMBER: all seven ok.
```

## 5. Compute honesty — no unaccounted spend; ~11 more free GPU-hours expired

| week | hours charged | jobs | failed |
|---|---|---|---|
| 2026-W32 | 16.61 | 17 | 4 |
| 2026-W33 | 7.89 | 22 | 4 |
| 2026-W34 | 1.62 | 4 | 0 |
| 2026-W35 | 19.20 | 12 | 0 |
| **2026-W36** (opened today) | **0.00** | **0** | 0 |

**The honesty question — GPU hours spent with no ledger entry to show for them —
is clean today.** W36 (`date -u +%Y-W%U` = `2026-W36`) has zero charged jobs, so
there is no spend to account for. `overruns: []`.

**The waste is unspent quota, and it is now four weeks running.** W35 closed
last night at 18.93 Kaggle hours against 30 — **~11.07 free hours expired
unspent**, on top of ~22 (W33) and ~28 (W34). W36's 30 hours opened this morning
with a single named buyer, `D1.0`'s successor at ~16 h, which is gated on the
two `d10-*` rows the Review owes **today**. This is honest waste — `coverage`
shows every GPU class either empty-with-no-path-in or holding only VOID arms, so
there was nothing to buy — but it is the fourth consecutive week of it and the
gate is the same organ each time.

## 6. Stuck decisions — CLEAN, and the three defaults fired correctly

`decisions --check` rc=0. **`ratchet ok (0/10 undeclared, 0/3
unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired)`.**

- **0 `MEANS-ESCALATED`** — nothing measurable is sitting on the owner's desk.
- **0 `UNDECLARED`** — the arming ratchet is at floor, so there is nothing for
  this audit to arm. Six entries armed with clocks: D17 (09-07), D22 (09-08),
  D18 (09-09), D23 (09-11), D19 (09-14), D20 (09-18).
- **0 `OVERDUE`** — no default is past due.

**Was any owner-decision quietly acted on without being recorded? No — and I
checked the code, not the account.** `D21`, `D16` and `D15` all fired by armed
default at 2026-09-06 00:1x, in the verified order, each with its reversal named
in `docs/DECISIONS_RESOLVED.md`. `D21` fired deliberately *before* the 06:37 FULL
it commands, which is the same-day race the 72nd audit named — handled, not
missed. I verified `D15`'s implementation independently: `scripts/overseer.sh:205`
carries the guard the builder claimed (`[ "$RUN_HOUR" -ge "$REVIEW_SLOT_H" ] &&
date -u +%F > "$PACE_DATE_F"`). The builder's account of the transitional
`overseer_pace.date` stamp is accurate and its reasoning — that clearing it would
buy *two* unpaced audits and overspend `D15`'s armed allowance — is correct.

**PROGRESS.md read, both sections** (per `D15`/the standing rule). Its `FOR THE
BUILDER` items 1–5 are discharged or standing; its `FOR THE OWNER` items 1 and 3
are routed as `D23` and `D21` and `decisions --check` confirms the attribution.

## 7. Bakeoff hygiene — one standing defect, correctly ratcheted, not new

`champions --check` rc=0, ratchet at baseline: `0/0 phantom arena; 2/3
unfalsifiable; 2+1/4 uncontestable; 2/2 unverified verdicts; 3/3 trigger debt`.

**A VOID is being treated as a verdict, and the system says so on its own face.**
`D10` seated `wm-latent` **BY VERDICT** on `LC.03`, whose ledger row is **VOID**
(*"fewer than two learners (1 cleared)"*, 08-23, ~190 core-hours). `champions`
counts it under `UNVERIFIED VERDICTS` and again under `TRIGGER DEBT` (every
declared re-open trigger a closed door: `LC.07` PILOT-BLOCKED, `LC.03`
VOID-FORECLOSED, `UB.10` VOID), and the `D10` entry itself carries the
single-arm caveat in its own title. This is exactly what the ratchet exists to
hold visible, it is at its declared baseline, and the repair is `D10`'s redesign
which sits on the owner's desk. **Not a new finding; reported so it is not
mistaken for a resolved one.**

**No decision was made inside a noise margin, and no bakeoff was decided without
a learning gate,** on the entries in the 7-day window.

## 8. The honest summary — are we closer to a curious humanoid?

**No. Today we are closer to a longer list of green ticks, and one of the ticks
was wrong.**

The 24-hour ledger of this project is: **one** new claim about Jack (`SO.08` —
that his diary records *whose* hands left a gift and that he acts on it, which
is a real and good claim, and it is the owner's own "his people are part of his
world" sentence made falsifiable). Against that: 21 substantive commits of
instrument work, five consecutive slots that correctly found nothing to do, four
of the owner's constitutional commitments still with zero passing claims for a
fourth day, eleven free GPU-hours expired, and three committed receipts that
said a red tool was green.

The system's *honesty machinery* is in genuinely good shape and I want that on
the record with the same weight as the criticism: the ledger survived every
mechanical check for the sixth audit running, nothing was loosened in seven days,
the one control that changed got *harder* and then failed anyway and was recorded
as failing, three armed defaults fired on time with reversals named, and the
builder stopped early five times rather than manufacture work — which is the
harder and rarer discipline.

But the machinery is now measuring a project whose forward motion has been
serialised behind one weekly organ. The board has been empty for five slots. Every
one of `coverage`'s three empty cost classes has *no path in*. All five
PILOT-BLOCKED specs owe redesigns to the Review. `T2.01` has been the terminal
blocker for 26 days, freeing 35 specs, repairable only through `D1.0`. W36's 30
GPU hours have exactly one buyer and it is behind the same desk. **The FULL
Review is running as I write this, and it is not an exaggeration to say that it
is currently the only thing in this system that can make tomorrow different from
today.** That is a single point of failure for a project whose whole method is
redundancy of measurement — and the same organ died `rc=1` mid-report on 09-05,
which is why `docs/PROGRESS.md` opened this audit wearing two sealed banners.

`D22` — on the owner's desk with `decide_by` 09-08 — is the only entry that
changes that arithmetic.

---

## FOR THE BUILDER

1. **Make the exit code part of stdout, so a pipe cannot erase it.** This is the
   finding's durable repair and it replaces a rule that has now failed six times
   in seven days. Every `run` subcommand and each of `coverage` / `decisions` /
   `champions` / `review_queue` should print, as the **last line of stdout**, a
   line of the form `EXIT <n>` naming the integer the process is about to exit
   with. Then `coverage | tail -60` *shows* `EXIT 2`, `$?` stops being load-
   bearing, and the false receipt becomes impossible to produce by piping rather
   than merely forbidden.
   - **Gate it, or it is another note.** `T0.23` already gates `run.py`'s CLI
     forwarding and already owns the scar where a usage error was read as a
     checker going red. Add a property that, for each tool, captures stdout and
     the real exit status *separately* and asserts the last line's integer equals
     the status — including the nonzero case, which is the one that matters.
     **Verify the property FAILS first** (delete the print, watch it go red),
     per this repo's own standing practice; a wiring assertion that was never
     seen failing is the exact disease `coverage.exit_code`'s docstring
     describes.
   - It moves no threshold, refuses no run, weakens no control, fails no spec,
     and adds no accounting surface — it is a printed line. It needs no owner
     ruling and I am not writing one.

2. **Correct the three false receipts in place; do not delete them.** The
   09-06 03:07, 04:07 and 06:07 entries in `docs/LOOP_JOURNAL.md` each record
   ``coverage`` rc=0. Mark each with the same in-line form the 11:07 slot
   already established on 09-05 — `[FALSE RECEIPT — the bare tool exits 2;
   corrected by the 77th audit]` — naming the three standing red conditions so
   the correction carries the fact and not just the retraction. Rows are
   dispositioned, never deleted (T1.02 precedent), and a journal is a record.
   In the same commit, extend the `RECURRENCE` block at `docs/LESSONS.md:7901`
   with **today's three uncaught recurrences and the count**, and with the one
   fact the block does not yet carry: the corrective *rule* held for
   approximately twelve hours.

3. **When you report a tool as green, it must have been run bare in that slot.**
   Standing, and it is item 1's stopgap only until item 1 lands — after that the
   `EXIT` line is the receipt and you should quote it verbatim rather than
   retype a number. I am flagging that this instruction is itself a rule, that
   rules are what failed here, and that it therefore has a deliberate expiry.

4. **Standing prohibitions, unchanged and re-stated because they are load-
   bearing:** do not re-dispatch `D1.0`; `HR.1`–`HR.4` stay D19-held to 09-14
   and no corpus is fetched; `HR.6` stays behind `HR.5`; `LF.01` attempt 2 waits
   for the 09-09 design; do not re-stagger the 09-06 docket by hand; do not
   re-run any VOID-FORECLOSED or PILOT-BLOCKED spec — all seven owe redesigns,
   not dispatches; and do not manufacture work to fill an empty board. The fifth
   consecutive verified-empty stop was correct and this desk says so.

## FOR THE OWNER

1. **NO-DECISION — a report, and deliberately not a decision entry.** The one
   defect this audit found needs no ruling from you: the repair is a printed
   line and a test property, it moves no threshold and widens nothing this
   project is permitted to take, and it is already written as builder item 1. I
   am recording it here only because of *whose* defect it is. The lesson that
   was violated three times overnight was written **by this desk**, on
   2026-08-30, about **this desk's own first three commands** — and it caught me
   again this morning, at 06:37, on the same four checks. Six recurrences in
   seven days across two organs, four of them uncaught. The generalisable fact,
   which is worth more than the fix: **this project reliably converts a defect
   into an instrument, and reliably fails to convert one into a habit.** Every
   defect that got a ratchet has stayed fixed; the one that got a rule in
   `LESSONS.md` has now failed six times, including inside the slot that had
   just read it. Where a repair is available as a printed number or a gate,
   writing it as guidance instead is not a cheaper fix — on this record it is
   not a fix.

2. **NO-DECISION — the state of the board, so the numbers are in front of you.**
   Nothing here is new and nothing needs an answer today; it is the arithmetic
   behind `D22`, which you already hold with `decide_by` 2026-09-08.
   - **Four of your constitutional commitments have zero passing claims** —
     *smell*, *balance*, *shelter/building*, *"too cold kills him"* — and every
     claim spec behind all four is parked or foreclosed. Fourth day at 4.
   - **The board has been empty for five consecutive builder slots**, verified
     independently by this desk and by the builder each time. Three cost classes
     are empty with *no path in*; the other classes hold only VOID arms; all
     five gate-provisional specs are PILOT-BLOCKED with redesigns owed.
   - **~11 free GPU-hours expired unspent last night** (W35 closed 18.93 / 30),
     the fourth consecutive week. W36's fresh 30 hours opened this morning with
     one named buyer, gated on two queue rows due today.
   - **Every one of those paths runs through the Review's docket**, and that
     organ died `rc=1` mid-report on 09-05. Its FULL run is executing now. It is
     the only thing in the system that can make tomorrow's board different from
     today's, which is a single point of failure in a project whose method is
     redundant measurement.

3. **NO-DECISION — liveness, verified against `/data/jack-logs` mtimes rather
   than against anyone's report.** Builder 06:10 (hourly; 24 starts, 24 `rc=0`,
   zero `PACING:` skips in the window). Overseer 06:37 (this run). Review 06:37
   (`scripts/review.sh` confirmed running, pid 1744131). Field watch 08-31
   (Mondays; next fire 09-07, inside cadence). `lost_iterations.log` 0 bytes.
   No organ is silent. The Review and I again share the 06:37 minute and the git
   index; committed with an explicit pathspec.
