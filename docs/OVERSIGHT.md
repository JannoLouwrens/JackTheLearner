# OVERSIGHT.md — the overseer's current-state report

**139th audit — 2026-10-04, launched 06:37:51 UTC.** Instrument readings below
were taken between **06:38 and 06:45 at `HEAD = cbadea2`**. This sitting
**overlaps the Sunday FULL Review**, which launched in the same minute
(`37 6` vs `37 */6`), and `HEAD` moved under me three times while I worked —
`33a41bb`, `7d5069e`, `413c566` (Review acts 1–3) and `e0bcd3d` (regate sweep).
Every number is stamped with the `HEAD` it was read at, and the readings are
**re-taken at commit time** in the closing block. Nothing here is inherited from
my predecessor's page without re-derivation; where I carry its finding I say so.

---

## VERDICT: DRIFTING — and what is **new** is that the six-day-dead current-state page has stopped being merely *unread* and started being **quoted**. Twice in four days the Review re-derived a fact from its own stale `PROGRESS.md` instead of from the file it was editing, and one of those false facts is now the standing justification for a live, FULL-sized queue row. The page is no longer a reporting defect. It is a **source**.

The unchanged and still-largest reading is the one my predecessor has reported
four audits running and I make the fifth: **the builder has produced nothing for
76 hours** (76 dark slots, floor 0; last `rc=0` 2026-10-01T02:17:07;
demonstrated `107 → 107`), the owner-facing page says *"Dark slots 0 … the
blackout stayed closed"*, and **73 % of the usage pool that is silencing this
project was not spent by this project.** I re-derived the release arithmetic from
`lib_usage.sh` rather than carrying it, and it has **moved against us**: the
first legal slot is ≈**2026-10-06 16:07 UTC**, not the ≈14:10 my predecessor
computed yesterday — the meter's one-point rise 82 → 83 cost **two hours**, and
every further point costs ~1.7 h more.

---

## RANK 1 — NEW, and it is the one that damages the board's trustworthiness: **the stale page is being copied into live rows.** Two measured instances in four days, by the same desk, both contradicting that desk's own recent commits.

`docs/PROGRESS.md` was last **rewritten** 2026-09-28 06:52:50 (`9cc0b0b`) and
last **touched** 2026-09-30 06:37:05 (`3fcad58`, the STALE stamp). At my launch
it was **96.0 h** old by git and its content described the **2026-09-28** window
— six days of world unreported — while its own banner says *"last moved **47h**
ago"*. That the banner understates is the 136th audit's lesson
(`LESSONS.md:20696`) and my predecessor's RANK 1; I am not re-litigating it.
**What is new is the consequence.** A current-state page that stops being
rewritten does not go quiet. The only thing it can still do is be read as state —
and it has been, twice:

### 1a. A false fact about `REVIEW_QUEUE.md`, asserted twice, now standing in a live row — and it is the repair the same desk made itself two days earlier

`docs/REVIEW_QUEUE.md:18158`, inside the row routed yesterday morning
(`seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`, `:18134`,
`fdc3522`, 2026-10-03 06:55:47):

> *"Two of them (`ne01`, `water-apply`) have **no `DUE:` at all** — the hold was
> their only clock, so they have been ageing for 40 days against nothing."*

The same sentence is in that commit's message. **Both halves are false, and were
false when written.** Both rows have carried `DUE: 2026-10-09` since
**2026-10-01 06:56:29**, commit `a65fdd7`, whose own title is *"Review DAILY
10-01 act 6/N: the two ageing-EXEMPT holds get a clock —
`ne01-occlusion-knife-edge` and `water-apply-phantom-force`, 38 days old with no
`DUE:` at all"*. Verified at `HEAD`: `grep -n 'A CLOCK, AT LAST'` resolves to
`REVIEW_QUEUE.md:298` and `:325`, and `git log -S 'A CLOCK, AT LAST'` returns
`a65fdd7` alone.

**The provenance is exact and it is the page.** `PROGRESS.md`'s 09-28 section
reads: *"Two of them — `ne01-occlusion-knife-edge` and
`water-apply-phantom-force`, both 35 d `HELD` — carry **no `DUE:` at all**; the
hold was their only clock."* The 10-03 row is that sentence with the day count
bumped **35 → 40** and the repair deleted. The desk re-derived its predecessor's
*reason* — which is correct doctrine and is this project's own standing rule —
from the **five-day-old page that states the reason** rather than from the
**file at `HEAD`**, and so reproduced the page's age along with its words. The
mechanism is exactly the one the 10-01 commit closed.

**Why this is not a typo.** That false premise is load-bearing. It is the first
substantive paragraph of a row that commissions **FULL-sized work on seven rows**
and is deliberately dated onto **2026-10-11, one over that date's measured
capacity of 6**, with the over-booking justified in prose. A sitting that opens
that row on 10-11 will be told two of its seven subjects have no clock, will find
`DUE: 2026-10-09` on both — a date that will have **passed** by then — and will
have to re-derive the row's own premise before it can act. **The desk cannot see
its own acts from four days ago**, and the artifact that blinded it is the page
it is supposed to maintain.

### 1b. The same blindness, measured again: one field finding routed **twice**, three days apart, with two different clocks, and the instrument prints only the first

Both of these are `OPEN` at `HEAD` and both route **the same** `docs/FIELD_WATCH.md`
§6 week-9 claim (`T4.06`'s deciding statistic read against an uncomputed floor):

| row | routed | DUE | ordered unit |
|---|---|---|---|
| `t406-latent-floor-was-never-computed` (`:17662`) | 2026-09-30 | **2026-10-05** | *"CHECK the arithmetic, then rule"* |
| `t406-deciding-statistic-read-against-an-uncomputed-floor` (`:17951`) | 2026-10-02 | **2026-10-13** | *"re-derive the arithmetic from the ledger"* |

The second row's body argues the routing question from first principles
(*"WHY IT IS ROUTED RATHER THAN ACTED ON, and this is the whole point of the
row"*) and **never mentions the first**, which had already answered the same
question — at greater length — three days earlier. It is not a supersession: the
first row is not dispositioned, carries no pointer, and keeps its own earlier
date. **One unit of work, two deadlines eight days apart.**

**And no instrument can say so.** `run status`'s `FIELD-WATCH FINDINGS` reader
prints `§6 ROUTED — quoted by queue-row t406-latent-floor-was-never-computed` —
**the first quoter only**. The duplicate is invisible to the reader whose whole
job is tracking whether a field finding reached a desk. A finding routed twice
reads identically to a finding routed once.

### The repair, and it is three cheap things — see FOR THE BUILDER 1 and 2

The page's staleness is already routed (my predecessor's FTB 5 and 6, both still
unexecuted because the builder is dark). What RANK 1 adds is that **visibility
repairs on the page are not sufficient**, because the page's damage is now
downstream of it, in committed rows. The two readers that would have caught
these — a contradiction check between a row's prose and its own parsed fields,
and a DOUBLE-ROUTED reading over field findings — do not exist and are small.

---

## RANK 2 — §4: the blackout, **fifth day and fifth audit**, now 76 slots. Carried from my predecessor, with its release arithmetic **re-derived and corrected against us by two hours**.

**The measurement.** `run status` ratchet counter at `HEAD`: `dark_slots = 76`,
**MOVED +76 since the committed reading of 0**, and **ABOVE its declared floor of
0** — growth nobody raised the constant for. The loop's own 06:07:13 line reads
*"75 consecutive dark slot(s); 0 failed slots (75.8 h since the last `rc=0`)"*;
the one-unit difference is the counter including the slot in flight, and I report
both rather than picking. Last `rc=0`: **2026-10-01T02:17:07** —
**76.3 h before this line**. `demonstrated 107 → 107` across all of it. Nothing
in `experiments/registry.py`, `registry_expansion.py` or `experiments/tests/`
has changed since **2026-10-01** (`87bc128`); the last three days contain **zero
commits that touch a spec or a test**.

**The cause is not the builder and not a fault.** Every skipped slot logs
`PACING: acting on 'week:all models' 83% at 54% of the week (line 61%) …
skipping, budget held for later in the week`. The loop is awake, correct, and
obeying a gate. **Of this week's 82 shared usage points: builder 16 (19 %),
desks 6 (7 %), both 0, NOT THIS PROJECT 60 (73 %).** The gate reads a *shared*
meter and throttles *one* tenant, so a co-tenant's spend buys this project's
silence. That asymmetry is `D40`'s subject (Review, 2026-10-03, armed,
`decide_by 2026-10-10`, default (v) = status quo) and it is correctly on the
owner's desk.

**The release arithmetic, re-derived from source and corrected.**
`scripts/lib_usage.sh:85` is `allow = PACE_FLOOR + ((PACE_CAP − PACE_FLOOR) ×
elapsed + 99)/100` in integer bash, `PACE_FLOOR 25`, `PACE_CAP 90`. Check against
the log: `elapsed 54 → 25 + (3510+99)/100 = 61`, and the 06:07 line prints
`line 61%` — exact. The usage week's start pins from the log's own `elapsed`
crossing (39 → 40 between 10-03 06:07 and 07:07): **≈2026-09-30 11:55 UTC**,
matching my predecessor's Wednesday. Then:

- **At meter 82** (yesterday): release needs `allow ≥ 82` → `elapsed ≥ 87` →
  **≈2026-10-06 14:07 UTC**. This reproduces my predecessor's ≈14:10 exactly.
- **At meter 83** (today): `elapsed ≥ 88` → 147.84 h after the week start →
  threshold 15:45, **first hourly slot ≈2026-10-06 16:07 UTC**.
- **Hard ceiling, independent of the meter:** at the week reset `elapsed → 0`,
  `allow → 25`, and the weekly percentage resets with it →
  **the blackout cannot outlast ≈2026-10-07 12:07 UTC.**

**So the forecast is not a date; it is a date that recedes ~1.7 h for every point
the other tenants spend.** One point of co-tenant usage cost this project two
hours of its own loop yesterday. I make **no forecast of the 90 % hard stop** —
the 137th audit's withdrawal of the 136th's "six hours" alarm was right and is
carried, not re-opened; the measured external draw spans 0.0–1.4 pts/h and the
honest statement is a range.

---

## RANK 3 — NEW, and visible only because this organ was running at the time: **today's FULL broke its own 24-hour-old FIRST-act promise and then kept the promise that mattered — `T1.08`'s design landed, on its date, at act 4. `D38`, the unruled arbiter of exactly this collision, falls due TODAY on a framing that is now stale in three ways.**

Yesterday's sitting pre-committed, in a commit title: *"`T1.08`'s design
**re-dated onto tomorrow's FULL as its FIRST act**"* (`8a4b55e`, 10-03 06:50).
`t108-pipeline-repair-has-no-design` read `OPEN 9 d … DUE 2026-10-04` — **today**
— and `T1.08` (FAIL) is the root of the creature gate chain
`T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08` and of **both** live routes to spending
the free GPU quota.

**What today's FULL actually did, from the commits that landed while I wrote
this page:** `33a41bb` *"act 1 (**OVERDUE FIRST**)"*, `7d5069e` *"act 2 (OVERDUE
FIRST)"*, `413c566` *"act 3 (OVERDUE FIRST)"* — then **`4fad464` act 4:
*"`T1.08`'s pipeline repair HAS A DESIGN, on its date"***, whose Step 0 found the
premise rotten (*"the recipe the noise floor is measured under is used by NO
pipeline"* — `make_action_optimizer` is Adam/warmup-constant/clip2.0 with four
spec callers and zero pipelines, against `TrainingPipeline.py:493`'s
AdamW/wd1e-4/clip1.0). **That is the most valuable thing to happen to this
project in four days and I am recording it before I record the defect.**

**The defect is narrow and it is about ordering, not delivery.** The promise was
not *"the design lands today"* — that was already the row's `DUE:` — it was *"as
its FIRST act"*, and that is the clause that broke. It matters only because it is
the **third observed instalment** of a collision the project has an open decision
about, and because the desk's own argument for `D28` winning rests on the claim
that the two cannot both fit in one sitting. **Today they both fit.**

**This is `D38` and `D38` is open, `CONDUCT-DESK`, and `decide_by 2026-10-04`.**
It was minted 09-27 to arbitrate *"two armed defaults each claim the FULL
Review's FIRST act, they collide only on Sundays"* — `D28` (dispose OVERDUE
first) against `D33` (W1 world design first). **Its framing is now stale in two
ways that the ruling desk should see before it executes today:**

1. **One of the two claimants is gone.** `D33`'s object —
   `w1-world-edit-window` — was stamped `DECLINED`, and the 10-02 addendum
   established `D33`'s default is **MOOT**, not merely expired. The collision as
   written is half-empty.
2. **A third claimant appeared yesterday and lost the ordering this morning.**
   The 10-03 sitting put `T1.08`'s design on the FULL's first act, which is a new
   instalment of the same collision under a different name — and `D28` took the
   first three acts, un-ruled, for the third observed time.
3. **The counterargument the entry calls "the stronger one against me" has been
   answered by measurement, today.** `D38` recommends `D28` keeps the first act
   on the ground that *"a 45-spec design is a large unit that has now
   demonstrated, five times, that it does not fit beside anything else"*, and its
   counterargument is that anything preceding W1 means W1 never happens. **This
   morning the desk emptied OVERDUE (5 rows), cleared STALE (2 rows), AND
   produced the design — eight acts before 06:51.** That is one sitting doing
   both, which is the evidence `D38` has been waiting for and did not have when
   it was written on 09-27. It does not settle which order is right, but it
   refutes the premise that the question is a forced choice.

**Not escalated to the owner, and the reason is recorded so a successor can audit
it:** `decisions --check` classes `D38` `CONDUCT-DESK` — *"desk-executable, not
the owner's; execute it, report it, do not ask"* — and it is the **Review's**
desk, not mine. `D13` forbids me the Review's prompt and `SYSTEM.md` forbids me
its rulings. This page is read by that desk every morning, which is the correct
channel, and this rank is the delivery.

---

## RANK 4 — §5 Compute honesty: **`2026-W39` is lost — ~28.93 of 30 free GPU-hours expired yesterday — and `W40` opened this morning onto a board where two of three GPU cost classes are EMPTY with no path in.**

From `experiments/gpu_budget.json`'s own `weeks` map, drawn against 30 free
Kaggle hours per week:

| week | kaggle drawn | lost at reset |
|---|---|---|
| W37 | 1.379 h (+3.033 h colab) | ~28.6 h |
| W38 | 0.9176 h | ~29.1 h |
| **W39** (expired Sat 2026-10-03) | **1.0719 h** | **~28.9 h** |

**3.37 h drawn of 90 over three weeks; ~86.6 free GPU-hours gone.** `W40`'s 30 h
opened at today's Sunday reset and carries **no entry yet**. It is on course to be
the fourth, and the reason is structural rather than negligent: `coverage` reads
**`gpu<20min` EMPTY** and **`gpu<8h` EMPTY**, both *"NOT FILLABLE — pilot BLOCKED
on evidence; the repair is a REDESIGN"*, and `gpu<2h` holds **only `UB.10`,
which is VOID** (an arm to repair, not a dispatch). The dispatchable-today queue
is **6 specs, all 6 VOID**. The builder is dark until ≈10-06 regardless.

**No dispatch has been manufactured to spend these hours and none should be** —
that would be the exact Goodhart failure this organ exists to catch. **The one
thing that could create a legal buyer landed during this audit:** `T1.08`'s
repair design (`4fad464`, RANK 3). It is **not** a dispatch and should not be
read as one — its own Step 0 reports the premise rotten, which points at
mechanism work before any seed is spent, and the re-buy it prices is
0 citing / 19 mechanical / 4 semantic. **`W40`'s 30 hours still have no legal
buyer today, and the builder that would build one cannot run until ≈10-06.**

**Spend without a verdict, unchanged:** `gpu_hours_no_verdict` TOTAL **49.49 h**,
of which **`D1.0` alone is 33.78 h across 2 attempts for 0 verdicts**;
`gpu_unattributed_jobs` **21, AT floor**. `D1.0` is `VOID` with a stale
`impl_sha`, so the cause is recorded rather than mysterious — but 33.78 GPU-hours
bought no row and that number has not moved in a month.

---

## RANK 5 — NEW, latent rather than live: **`decisions.py` counts a citation of a RESOLVED decision as "reaching a desk".** The one ask on the owner's page today is attributed to a decision that closed 22 days ago.

`decisions --check` prints:

```
1 owner-ask(s) reached a desk and are NOT reported above — check the attribution:
  PROGRESS #1   matched-by: D22 (cites)
```

**`D22` is `RESOLVED BY ARMED DEFAULT`, fired 2026-09-12** —
`DECISIONS_RESOLVED.md:789`. The ask's live home is **`D33`**, which the ask
names in its own first line (*"Already routed; cite `D33`"*) and which the same
tool reports as **STALE by 11 days** carrying the register's **only broken
ratchet class**.

**The defect, from source.** `_reaches_a_desk()` (`decisions.py:1127`) resolves
against `_entries(needed_text, resolved_text)` (`:1159`, `:1178`) — the entry
universe **deliberately includes the resolved file** — and applies **no
open/closed test**. Ties are broken by `_entry_key`, *lowest decision number
first*, so when an ask cites both an open and a closed decision the **closed one
wins the attribution**. Here the ask cites `D22` only to explain why the thing
asked for **cannot be done** (*"because `D22` is your resolved ruling"*), and
that citation is what credits it as routed.

**Severity, stated honestly rather than inflated: nothing is being lost today.**
`D33` is open, so the ask does reach a live desk and
`decisions_unrouted_owner_ask = 0` is substantively correct. **The hole is
latent and the tool's own docstring names the direction:** *"the silencing is the
dangerous direction: an ask that goes quiet for the wrong reason is invisible
unless the report can say who quieted it."* An ask whose **only** citation is a
resolved decision is silenced today with no class, no default and no
`decide_by`, on a page rewritten every morning — which is precisely the
`D1`-shaped hole `UNROUTED-OWNER-ASK` was built to close. `review_queue.py`
already has a reader for this exact shape (`DISPOSITION-ON-A-CLOSED-DECISION`,
3 rows today). `decisions.py` has no counterpart.

---

## RANK 6 — §1 and §2: **no findings, re-derived from scratch this sitting rather than inherited.** This is a real result and I state it plainly.

**§1 Integrity of the ledger.** Over all **107 PASS** rows in
`experiments/ledger.json` (157 rows with a verdict, 255 specs registered):

- **107/107** resolve a declared implementation that exists on disk.
- **107/107** recorded `commit` values resolve under
  `git cat-file -e <commit>^{commit}` — **zero** orphaned certificates.
- **105/107** declare a `control` **and** carry recorded `control_metrics`. The
  two exceptions are `T0.01` and `T0.10`, and both hold an **explicit falsy
  refusal object** (`NoControlByDecision`, shipped at `eba3e58` on 09-27
  specifically so that a refusal cannot be claimed by typing a sentence a
  detector pattern-matches). **No PASS in this ledger rests on a control that
  was never run.**

The known reds behind that clean result are reported where they belong and are
not new: **2 DIRTY STAMPS** (`T6.03`, `PL.02`), **15 STALE CLAIMS**, **1 STALE
PRE-`impl_sha` CLAIM** (`T2.02`), **5 UNBACKED CERTIFICATES**, and
**`pass_on_dead_dependency = 5` ABOVE its floor of 3** — whose cause is READ not
reasoned (`T0.13`'s honest re-buy to FAIL took its two dependents with it) and
whose repair (`t013-latently-red-28-disarmed-keys`, DUE 2026-10-05) is owed by a
builder that cannot run.

**§2 Thresholds and controls over time.** `git log -p --since="7 days ago"` over
`registry.py`, `registry_expansion.py` and `experiments/tests/`: **no threshold
moved in the loosening direction, no control was deleted or weakened, no `_check`
gained an `or`, no seed count fell, no assertion was removed.** What the window
contains is the opposite — `T0.31` strengthened 22 → 24 properties (`5651fc1`),
`T0.01` made to hash the thirteen modules it certifies (`21c4883`), `T0.18`'s
Probe C taught to distinguish a refusal from a promise **with the hole that
opens armed in the same commit** (`3eddd91`), `DP.04`'s `NEED_MIN_GAIN`
registered at 35.0 by a pre-declared rule (`5b0d4c0`), `PS.05`'s known-answer
control pre-registered **before** it was run once (`6e2493f`), and `LG.14`
registered with both mandatory conjuncts in the pre-registration rather than
promised for later (`87bc128`).

The single candidate I examined closely and cleared: `0ac932b`'s *"ordered
`T2.08` softening"* in `worst_seed_audit` promotes a sibling-std-protected
`WRONG` to its own `PROTECTED` verdict. It is a **reader classification** change,
it is justified in the commit by arithmetic (`margin_floor = mean − 1.5·std > 0`
guarantees per-seed positivity at n=3, and the 0.05 bar is mean-level), it names
the defect as reader ambiguity rather than an admitted violating seed, and it
**corrects the counts downward** for the desk that will price the ruling
(lane A 139/1 WRONG, lane B 17 gates/13 specs). No bar moved. **Not a silent
loosening.**

---

## RANK 7 — §3 Drift: there was **no work to drift**. The converse question has an answer, it is unchanged, and it is bad.

**What the builder worked on in the last day: nothing.** The 24-hour window
contains **15 commits and not one is the builder's** — 7 Review acts + its
INCOMPLETE row, 2 regate sweeps, the overseer's sealed draft and its 138th
audit. Zero lines of Jack. The last builder slot (10-01 02:07–02:17) registered
`LG.14`, which traces to GOAL.md's *"and VOICE — he must be able to make sound,
not only receive it"*, and executed the 134th audit's honesty orders, which trace
to *"protects the honesty of watching what happens when the three meet"*.
**Neither is drift.** That slot's own summary records the honest cost:
*"**Creature gate: NONE — seventeenth consecutive**, recorded as the violation it
is. `LG.14` is voice work … but a registration is machine, not creature."*

**Which parts of GOAL.md have no passing spec at all** — `coverage` at `HEAD`,
`commitments_uncovered = 0` (AT floor, and that is the good news):

- **3 CLAIM-DEAD commitments** — *smell*, *shelter/building*, *thermal (too
  cold/hot KILLS him)*. Every spec that could falsify them is parked or
  foreclosed on honest evidence; the parking was right and leaving the
  commitment claim-dead is the bug. Unchanged **eight days**.
- **14 commitments with live claim specs and nothing passing**, including
  *touch*, *tool use*, *told world*, *proprioception*, *sleep*, *fast/slow*
  (every member welded behind `LC.03`), and **`one brain / unison` — 28 specs,
  1 passing**.
- **6 NO-LIVE-PATH holes** (3 CLAIM-DEAD + 4 unwinnable seats, 1 seen by both);
  the repair for every member is a **registration**, never an unpark.
- **4 NEW unrunnable GOAL.md citations** — `GEN.02`, `GEN.03`, `GEN.06`,
  `GEN.09`: each id resolves, each resolves to a corpse (`welded<-LC.07`), so
  the citation's present tense is false and the dangling count cannot see it.
  Owned by `gen-four-revival-needs-an-affordable-lc07-successor` (DUE 10-11).

**The Goodhart reading, from `SETTLE EVENTS` over 7 days: 145 runs recorded →
4 first-ever verdicts, 137 re-buys, 4 status changes. 125 of 145 (86 %) are
instrument-coupled** — this project's own tool edits staling its own
certificates. Of the 4 first-ever verdicts, **three are `T0.*`** (`T0.21`,
`T0.28`, `T0.31`) and one is `T2.11` VOID. **In a week of 145 runs, the number
that settled something new about Jack is one, and it VOIDed.**

---

## RANK 8 — §4 The routed work: the desk emptied `OVERDUE` to 0 yesterday and it was **5 again within 24 hours**. Drain is still **UNBOUNDED**.

`run review-queue` at `HEAD = cbadea2`: **53 OPEN / 3 HELD / 29 DISPOSITIONED /
37 ACTED / 2 DECLINED of 124 routed; 85 live rows; oldest live 41 d; EXIT 2 with
14 violations — OVERDUE 5, STALE 2, HOLD-ON-A-RESOLVED-BLOCKER 7.**

**The 24-hour round trip, and it is a capacity fact rather than misconduct.**
Yesterday's sitting drove `OVERDUE 8 → 4 → 3 → 0` across acts 2 and 4. All five
rows that are overdue this morning fell due on **2026-10-03 itself** — the day
the desk sat, made seven acts, and discharged none of them:
`w1-cold-is-not-lethal-at-night` (**third** break),
`xl01-death-and-retry-has-no-reachable-repair-path` (**third**),
`cross-organ-doc-race-voids-certificates` (second),
`d35-none-quota-has-no-satisfying-move`,
`ba03-registered-run-foreclosed-by-d20-class-closure`. Four of the five were
re-dated onto 10-03 by this desk on 09-26. **The instrument warned before the
dates passed** — `IMMINENT` printed *13 live dated rows due on or before the
next cycle against a measured capacity of 6, 7 of them undischargeable* — which
is the only time anything can be done about it.

**And then the FULL cleared the whole lot in thirteen minutes while I was writing
this rank.** By 06:51 all five OVERDUE rows and both STALE rows were disposed
(acts 1–8, `33a41bb`…`897fd38`), `review_queue_violations` fell **14 → 7**, and
the only class left is the orphaning the project is deliberately refusing to
launder. **Eight acts against a demonstrated capacity of six, including the
design** — so the honest reading of the 24-hour round trip is **capacity, not
neglect**, and today it exceeded its own measured capacity. What that does not fix
is the arithmetic below: the desk disposed 7 rows this morning and the backlog is
85 live.

**Throughput, measured against git rather than declared dates:** arrived **26
(3.71/cycle)**, disposed **14 (2.00/cycle)**, designed **17 (2.43/cycle, not a
disposal — the row stays live and keeps ageing)**. **Drain UNBOUNDED; arrivals
exceed disposals by 12 over the window; the backlog has no projected end.**

**Two ratchet movements I must say out loud:** `review_queue_piled_on`
**4 → 9** (+5) and `review_queue_net_arrivals` **32 → 12** (−20, of which
clock −16, act −4). **Nine live rows now share 2026-10-13 against a measured
capacity of 6** — that many promises are scheduled to break together — and
2026-10-04, 10-05, 10-09 and 10-11 are each amber too. Of the 9 rows `DATED ONTO
A FULL DAY`, the row from RANK 1a is one: dated onto 10-11 when 6 were already
promised there, disclosed rather than hidden, which is the honest half.

**The 7 `HOLD-ON-A-RESOLVED-BLOCKER` rows are deliberately NOT laundered**, and
that is correct: re-pointing them at a fresh blocker would hide the single
largest structural fact the project has. The instrument's message for the class
still says *"the window it was waiting for has opened"*, which is **false of a
`DECLINED` blocker** — it was abandoned, not opened — and reading it literally
gets the disposition backwards. That defect has been known and recorded, not
repaired, since 09-28.

---

## RANK 9 — §6 and §7: one broken ratchet class on the owner's register, **two conduct entries stale past their own dates**, an armed default **firing tomorrow**, and the ladder's most load-bearing seat held **BY VERDICT off a VOID**.

**§6 Stuck decisions.** `decisions --check` EXIT 1.

- **`RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0`** — `D33`, red since
  **2026-09-23**, 11 days. Its default names a date in the past **and** its
  object went terminal when `w1-world-edit-window` was stamped `DECLINED`, so
  the 10-02 addendum is right that it is **MOOT, not merely expired**: no desk
  can clear this by firing anything. This is the only broken class on the
  register and nothing in the repo can repair it.
- **`D37` is armed and its `decide_by` is TODAY (2026-10-04)** — so the default
  **fires tomorrow** if the owner does not rule. See FOR THE OWNER 5. It costs
  **0 specs**, its default is the only legal one, and the entry says so.
- **`D38` `decide_by` TODAY, `D35` STALE by 10 days, `D39` 10-15, `D40`
  10-10 — all four `CONDUCT-DESK`**, i.e. *"execute it, report it, do not ask"*,
  and all four are other desks'. `D38` is RANK 3.
- **Nothing `MEANS-ESCALATED`** — no fork a measurement could settle is sitting
  on the owner's desk. **Nothing `UNDECLARED`** (`decisions_undeclared = 0`, AT
  floor), so there is nothing for me to arm this audit, and I am not inventing
  something to arm.
- **`STEERING-DATE-MISMATCH ×3`, every one `D33`:** `PROGRESS.md` quotes
  **09-27** and **10-09**, `OVERSIGHT.md` quotes **10-09**, the register says
  `decide_by 2026-09-23`. A stop-rule published to the owner **16 days** past the
  register's own deadline, with the register never moved. A deadline that moves
  in prose but not in the register is the deadlock it replaced.

**§7 Bakeoff hygiene.** `champions --check` **EXIT 0 with 10 violations** —
every class AT its declared floor, which is the floor being the problem, not the
tool being wrong.

- **`Learning core` — `VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE`.** Held **BY
  VERDICT**, the strongest marking in the file, off **`LC.03`, which is VOID** —
  and `SYSTEM.md` says a VOID decided nothing. This is §7's own checklist item
  *"a VOID treated as a verdict"*, standing since 2026-09-01, and **every
  pre-registered re-open trigger is a closed door** (`LC.07` PILOT-BLOCKED,
  `LC.03` VOID-FORECLOSED, `UB.10` VOID). The whole ladder rests on this seat.
- **`World` — `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED`.** Held BY VERDICT
  and names neither the row that bought it nor what could fire a rematch. An
  unwritten promise cannot decay visibly.
- **`Fast/slow coupling` — `ARENA-UNREACHABLE` + `TRIGGER-UNREACHABLE`** (rooted
  at `LC.03`); **2 `NO-ARENA`** seats (ASR, Speaker ID) that nothing could ever
  unseat; **2 `UNCONTESTED`** (Vision encoder, the PLASTIC-ONLY decree), both
  turning on `PL.02`, which is dated.
- **4 seats nobody can ever WIN** (Episodic retrieval, Language grounding,
  Smell, Body schema) — every pending arena member welded, with no unearned
  holder to indict.
- **No winner chosen inside the noise margin that is unowned.** `T4.06`'s
  `loss_reweight` is CERTIFIED at **+0.0187 = +6.9 % of the anchor's own seed
  spread** on `min_modality_latent_r2` (**2 improving / 1 REGRESSING**) and
  **+8.9 %** of spread on `eval_loss_mean` — a margin thin enough to deserve the
  question — and it **is** routed, twice over, which is RANK 1b.

---

## §8 — THE HONEST SUMMARY. Are we closer to a curious humanoid that climbs the ladder than we were yesterday?

**No. We are four days further from it than we were on 2026-10-01, and today is
the first day that standing still has started producing false evidence.**

The ledger has not moved: `107 → 107` for 76 hours. In the last seven days 145
runs bought **four** first-ever verdicts, three of them about the rig's own
instruments and the fourth a VOID; **86 % of all that volume was this project's
tools staling this project's certificates.** Of the owner's own constitutional
commitments, three — smell, shelter, *too cold kills him* — have **no living
falsifiable claim at all**, and that has been true and reported for eight days.
The architectural seat the entire ladder stands on is held by the strongest
marking the file has, off a **VOID**, with every door out of it closed. ~86.6
free GPU-hours have expired across three weeks and the fourth week opened this
morning onto two empty cost classes with no path in. The creature gate has been
NONE for seventeen consecutive builder slots.

**And the honest part is that almost none of that is anyone's fault today.** The
builder is healthy, correct, obeying a gate, and dark because 73 % of a shared
meter was spent by another tenant. The desks are sitting every morning and dying
at a 20-minute wall clock after six to nine real acts. Both design debts between
this project and Jack belong to a desk whose largest unit has never fitted inside
its own sitting. Every instrument is working; several are working better than
last week.

**What is new today, and it is the thing worth taking from this page, is a second-order
failure.** A system this instrumented does not usually go wrong by lying. It goes
wrong when its organs start reading each other's *reports* instead of the
*artifacts*, and today I can measure that happening twice in four days: a desk
asserting a false fact about a file it was editing, copied from its own six-day-old
page, contradicting its own commit from two days before — and the same desk routing
one finding twice without noticing, because the reader that tracks routing prints
only the first quoter. **A stale current-state page is not a quiet page. It is a
source, and it has begun to be cited.** That is a smaller fact than the blackout
and a more dangerous one, because the blackout ends on Tuesday at the latest by
arithmetic nobody can argue with, and a false premise standing in a live row ends
only when somebody checks.

---

## FOR THE BUILDER

**0. EVERY ORDER FROM THE 135th, 136th, 137th AND 138th AUDITS IS STILL OPEN, AND
NONE OF IT IS YOUR FAULT.** Your last slot ended **76 hours** before this line;
all four reports were written after it. I have **not** re-ranked them and I am
**not** restating them in full — my predecessor's page (`cbadea2`) verified items
1–4 of its own list from source and carried the 137th's and 136th's forward, and
that work stands. Read, as live and unexecuted: the **steering-size growth fit**
(make it monotone or say it is not), the **cliff reader pointed at stdin instead
of the argv pages**, the **four prose sites asserting a dead mechanism in the
present tense**, the **re-aimed outage fixture**, the **stale banner's live age**,
**the page not written last**, and **the shared-file commit race reader**. The
136th's *"nothing in `experiments/` reads the usage meter, so the distance to a
stop that pauses every organ reaches no exit code"* is still the right thing to
spend your first legal slot on — and after 76 dark slots that judgement is worth
more, not less.

**My two new items are both small, both reporting-only, and both come from RANK 1
— the class of defect where an organ cannot see its own recent acts.**

**1. GIVE `review_queue.py` A CONTRADICTION READING: a row's PROSE against its
own PARSED FIELDS.** RANK 1a is a committed row (`REVIEW_QUEUE.md:18158`)
asserting *"have **no `DUE:` at all**"* about two rows that have carried
`DUE: 2026-10-09` since `a65fdd7` (2026-10-01 06:56:29) — a repair the same desk
made, and whose commit title says it made. The parser already resolves every row
id and every `DUE:`/`BLOCKED-BY:`/`WAITS-ON:` field, so this costs almost
nothing: when a row body names another row id in the same sentence as a phrase
asserting the **absence** of a field that in fact parses on that row, print
**`PROSE-CONTRADICTS-A-PARSED-FIELD — <row> says <claim> about <row2>, which
carries <field>`**. **Constraints, and they matter more than the feature:** keep
it **reporting-only and unfloored** — a desk writing history in prose is legal
and a gate here would refuse honest work — match on **declared field names
only**, never on free interpretation of intent, and when the match is uncertain
say nothing. A confidently wrong contradiction claim is worse than none. If you
judge the phrase-matching too fragile to be honest, **say so in the commit and
ship the narrower half instead**: print, beside every row id a body cites, that
row's **currently parsed `DUE:`** — so a reader of the prose sees the field
without having to go and look. That narrower version catches RANK 1a outright and
has no heuristic in it at all.

**2. MAKE `FIELD-WATCH FINDINGS` PRINT **EVERY** QUOTER, AND ADD A
`DOUBLE-ROUTED` READING.** RANK 1b: `docs/FIELD_WATCH.md` §6 is routed by **two
live rows** — `t406-latent-floor-was-never-computed` (09-30, DUE 10-05) and
`t406-deciding-statistic-read-against-an-uncomputed-floor` (10-02, DUE 10-13) —
and `run status` prints *"§6 ROUTED — quoted by queue-row
`t406-latent-floor-was-never-computed`"*, the first only. One unit of work, two
deadlines eight days apart, invisible to the reader whose job is tracking whether
a finding reached a desk. Print the **full list** of quoting rows, and when it is
longer than one print **`DOUBLE-ROUTED — §<n> is owned by <N> live rows with <N>
distinct DUE dates`**. **Reporting-only and unfloored**, per `D27`'s own
reasoning: a finding legitimately touched by two rows is possible and a gate here
would forbid a legal move. Do **not** propose deleting either row — rows are
dispositioned, never deleted (T1.02 precedent), and which row survives is the
Review's call, not a tool's.

**3. `decisions.py`: PREFER AN **OPEN** ENTRY WHEN AN ASK CITES BOTH, AND NAME
THE CLOSED CASE (RANK 5).** `_reaches_a_desk` (`:1127`) resolves against
`_entries(needed_text, resolved_text)` with no open/closed test, and `_entry_key`
hands the attribution to the **lowest decision number**, so today's single
silenced ask reads `matched-by: D22 (cites)` — resolved 2026-09-12 — while its
live home `D33` sits 11 days stale with the register's only broken ratchet.
**Two changes, both small:** (a) in the tie-break, prefer an entry from
`DECISIONS_NEEDED.md` over one from `DECISIONS_RESOLVED.md`, so the printed
attribution names the desk that can actually receive the ask; (b) add a soft
reading **`ROUTED-TO-A-CLOSED-DECISION`** for an ask whose **only** home is
resolved — `review_queue.py` already has the analogue
(`DISPOSITION-ON-A-CLOSED-DECISION`) and its docstring explains why. **Keep
`UNROUTED-OWNER-ASK` exactly as it is and do not fold the new class into its
count**: the floor is 0 and AT 0, and growing a ratchet to carry a new class is
the tidy-up `T0.31`'s P4/P5/P6 exist to forbid. The new reading is **unfloored**.
Nothing is being lost today — say that in the commit, because the honest
justification for this change is the latent case, not a live one.

**4. WHEN YOU COME BACK, READ THE FILE AND NOT THE PAGE.** This is conduct, not
code, and it is the generalisation behind items 1 and 2 — a lesson is appended to
`docs/LESSONS.md` this sitting. `docs/PROGRESS.md` has been six days dead with a
banner understating its own age by 49 hours. Four audits and five Review sittings
have been written on top of it. **When you re-derive any predecessor's reason —
which is the standing rule and is right — re-derive it from the artifact at
`HEAD`, not from the page that states it.** Every instance in RANK 1 would have
been caught by one `grep` against the file being edited.

---

## FOR THE OWNER

**1. The 90 % hard stop is still yours alone and still nobody is watching it.**
Unchanged from yesterday and I am not re-asking: read **`D40`** (Review,
2026-10-03, armed, `decide_by 2026-10-10`, default (v) = status quo — pace the
builder against this project's *own* attributed spend instead of
`week:all models`). `lib_usage.sh:121` refuses `ladder_loop.sh`, `overseer.sh`,
`review.sh` and `field_watch.sh` — **every organ except the regate sweep** — at
the stop, and resuming requires a `.usage-resumed` file written **by you**. There
is none on disk. `week:all models` reads **83 %** (up one point in 18 h) and
**73 % of this week's 82 shared points were not this project**. Nothing in the
repo can fire a default here, correctly — a default may not loosen a gate. **The
decision worth making in the quiet rather than at the stop is whether you want a
standing pre-authorised resume ceiling with an expiry, or whether a hard halt
until you look is what you intend.** Either answer is fine. **I attach no alarm
and forecast no time:** the measured external draw spans 0.0–1.4 pts/h, the
honest statement is a range, and the 136th audit's withdrawn "six hours" alarm is
why I am saying so explicitly.

**2. `D37`'s armed default FIRES TOMORROW (2026-10-05) — today is the last day
you can rule on it.** This is the one new time-critical item on your desk, and
**the Review reached it independently at 06:50 this morning** (`5b9fcfb`, act 9:
it refused to reclass the entry to `conduct`, because *"executing is the one thing
no relabel can make legal against `D29`'s resolved (iii)"*, and gave you the same
one-day notice). Two organs arriving separately is the system working;
`decide_by: 2026-10-04`. **It costs 0 specs and the cost of delay is not
perishable** — `A4` holds the Learning-core seat, the arms behind it are
`LC.03` (VOID-FORECLOSED) and `LC.07` (unaffordable at both venues), so nothing
is waiting on it this week, and the entry says so rather than overselling itself.
The question: `LEARNING_CORE.md` §5.4 promises `A4` a **mandatory** collapse
diagnostic that was never computable, `D10` seated `A4` BY VERDICT anyway, and
`D29` already ruled (iii) *record the debt, change no marking* — on a premise its
own author called insufficient, because the deliverable that was supposed to
inform it slipped. **The default is (iii) HOLD `D29` AS IT STANDS, and it is the
only legal one** — a default may not reverse a resolved decision. It is
monotone: it can only leave the debt visible and unguarded, never hide it, never
move a threshold, never spend a GPU-hour. **Its price, stated rather than
buried:** the next latent-prediction arm inherits an uncomputable VOID condition,
and the project keeps a `mandatory` guard it has never once been able to run.
**To reverse it after it fires:** rule on `D37` at any time and the readout gets
built; the firing writes no code and moves no marking, so there is nothing to
unwind.

**3. NO-DECISION: `D30`'s standing report, delivered here because the page it is
supposed to live on still says the opposite.** Builder dark **76 consecutive
slots / 76.3 h**, the longest on record; last `rc=0` 2026-10-01T02:17:07;
demonstrated **107 → 107** for 76 hours; first legal slot ≈**2026-10-06 16:07
UTC**, hard ceiling ≈**2026-10-07 12:07 UTC** at the week reset — both re-derived
from `lib_usage.sh`'s integer arithmetic this sitting, and the first moved two
hours **later** than yesterday's figure purely because the shared meter rose one
point. **`2026-W39` closed with 1.07 h drawn of 30 free Kaggle GPU-hours; ~28.9 h
expired at yesterday's reset — the third consecutive week lost** (W37 1.38 h,
W38 0.92 h; **~86.6 free GPU-hours in three weeks**). **`W40` opened this
morning** and no dispatch has been manufactured to spend it, correctly: two of
three GPU cost classes are EMPTY with **no path in**, and the third holds one
VOID arm. All four organs fired within cadence; none is silent past 2× its
cadence. **`docs/PROGRESS.md`, which `D30`'s armed default made the vehicle for
this report, still reads "Dark slots 0 … the blackout stayed closed", has not
been rewritten in 96 hours, and its own banner understates that by 49 hours** —
which is why you are reading this paragraph here instead of there.

**4. Nothing new is asked on `D33`; this is a pointer, not a re-ask.** **11 days**
past `decide_by`, the sole cause of the one broken ratchet class on your
register, and its default is **MOOT** rather than merely expired — its object
went terminal when `w1-world-edit-window` was stamped `DECLINED`, so no desk can
clear it by firing anything. The Review's published stop-rule fires
**2026-10-09**; **this morning's FULL re-dated a fourth row onto that same class
clock** (`413c566`), so the class is growing while the clock runs. Its
recommendation stays quoted verbatim in the entry and is unchanged.

**5. The standing ask no instrument will ever raise, repeated once because it is
cheap for you and expensive for us.** Three of your own constitutional
commitments are **CLAIM-DEAD**: *smell*, *shelter/building*, and *too cold/hot
kills him*. Every spec that could have falsified them is parked or foreclosed on
honest evidence — the parking was right; leaving the commitment claim-dead is the
bug. Each needs a **successor spec registered**, which is real design work the
ladder cannot generate from inside itself, because a missing spec has no id,
blocks nothing and fails no gate. `coverage` has reported this unchanged for
**nine days**. **If you want these three alive, the cheapest thing you can do is
say which ONE matters most**, so one successor gets designed instead of three
waiting equally. They are, with the jungle in mind: *smell* is the sense that
works when sight fails, *shelter* is your own image of success, and *thermal
death* is the pressure that was supposed to teach shelter. I am not ranking them
for you.

---

## READINGS RE-TAKEN AT COMMIT TIME — and the concurrency disclosed

The Sunday FULL Review launched in the same minute as this audit and was still
running when I wrote the above. **Everything in the body is stamped
`HEAD = cbadea2`, read 06:38–06:45 UTC.** The re-read below is taken immediately
before this file is committed; where a number moved, **the Review's live acts are
the cause** and the movement is to the project's credit, not against it. I commit
with `git add` by name so that nothing of the Review's lands inside my commit.

**RE-READ AT 06:51:24 UTC, `HEAD = 897fd38`** — eight Review acts landed between
my first reading and this one (`33a41bb`, `7d5069e`, `413c566`, `4fad464`,
`ebb6792`, `d2f4228`, `59856a2`, `897fd38`), plus the 06:44 regate sweep.

| instrument | 06:38–06:45 @ `cbadea2` | 06:51 @ `897fd38` | why |
|---|---|---|---|
| `coverage` | EXIT 2 · 0 uncovered · 3 CLAIM-DEAD · unreachable 96/95 · pass-on-dead-dep 5/3 | **unchanged** | no spec moved; the builder is dark |
| `decisions --check` | EXIT 1 · `DEFAULT-ACTION-EXPIRED 1` (floor 0) · `D37` due today | **unchanged** | `D33` is MOOT; nothing can clear it |
| `champions --check` | EXIT 0 · 10 violations, all AT floor | **unchanged** | no arena ran |
| `run review-queue` | EXIT 2 · **14** violations (OVERDUE 5, STALE 2, HOLD×7) · 124 routed / 37 ACTED | EXIT 2 · **7** violations (**HOLD×7 only**) · **126 routed / 40 ACTED** | **the FULL disposed all 5 OVERDUE and both STALE** |
| ratchet SLOT LINE | 4 MOVED | **6 MOVED** — `review_queue_violations` **14 → 7**, `piled_on` **4 → 11**, `net_arrivals` **32 → 11**, `fail_unowned_owned_forms` queue-row **31 → 32**, `dark_slots` **0 → 76** | Review acts; `dark_slots` unchanged |
| floors | 4 ABOVE (`dark_slots`, `decisions_default_action_expired`, `pass_on_dead_dependency`, `unreachable`), 0 BELOW, 0 UNVERIFIED | **unchanged** | — |

**Nothing in the body is withdrawn by the re-read.** RANK 3 and RANK 8 were
rewritten above to credit the acts rather than left standing against them, and
RANK 1's two findings are untouched by any of it: `REVIEW_QUEUE.md:18158` still
asserts *"no `DUE:` at all"* about two rows carrying `DUE: 2026-10-09`, and the
§6 field finding is still owned by two live rows with two clocks. **`dark_slots`
did not move and the page the owner reads has still not been rewritten** — at
this line, `docs/PROGRESS.md` is **96.2 h** old by git with a banner saying 47 h.
The Review is still running; if it reaches its page this sitting, the banner
clears and this paragraph is the record of what the morning looked like before it
did.

**Two further Review acts landed after the table above** — `5b9fcfb` (act 9,
`D37`'s misfiling flag disposed, reclass REFUSED) and `817c3f9` (act 10, `T0.31`
re-bought to PASS at attempt 23 after falling to FAIL at 06:44 **on this
sitting's own act-3 malformation**, repaired at act 8). The desk breaking and
repairing its own instrument inside one sitting, and saying so, is the honest
version of that failure and I record it as such.

I commit **`docs/OVERSIGHT.md` and `docs/LESSONS.md` by name only**.
`experiments/cpu_budget.json` is dirty at this instant (`T0.31` 2.53 → 5.79 s,
`used_s` 103.38 → 106.64) — **that is the Review's act-10 re-buy billing itself,
not mine, and I have deliberately left it for the organ that produced it.**
Nothing else of mine was in the tree, nothing was detached, no container or daemon
was touched, no experiment was re-run, and no spec, test, model file or
`experiments/ledger.json` was edited.
