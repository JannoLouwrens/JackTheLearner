# OVERSIGHT.md — the overseer's current-state report

**140th audit — 2026-10-05, launched 06:37:20 UTC.** Instrument readings below
were taken between **06:37 and 06:48**, starting at `HEAD = 32b098c`. This
sitting **overlaps the DAILY Review**, which launched in the same minute
(`37 6` vs `37 */6`), and `HEAD` moved under me four times while I worked —
`842c61c`, `f6527a3`, `01e091b`, `12fe615` (Review DAILY acts 1–3 plus a regate
sweep). Every number below is re-derived from source or from a live instrument
run, never carried from my predecessor's page; where I carry a finding I say so,
and where I **correct** my predecessor I say that too. Readings are re-taken at
commit time in the closing block.

---

## VERDICT: INTEGRITY RISK — and the reason is new. For five days the channel this organ uses to give the builder orders has been reporting that the channel is empty, and the sentence it prints to say so is false.

`run status` has been printing **"STEERING-PAGE ORDERS — no `## FOR THE BUILDER`
section found on docs/PROGRESS.md, docs/OVERSIGHT.md"**. Both pages have that
heading. Both have it at column 0. The reader finds **both headings** and then
matches **zero of the thirteen numbered orders beneath them**, because
`steering._ITEM` requires the digit at column 0 and both desks now write
`**N. THE ORDER**`. So the instrument built to check whether a steering order
names a spec the runner would refuse has checked **nothing** since 2026-09-30,
and the line it prints instead blames a heading that is present.

This is an INTEGRITY RISK rather than a DRIFT because of what the three findings
below have in common: **every one of them is a number or a fact that was
re-derived in prose, published to the owner, and is wrong — while the instrument
that holds the correct value sat in the same terminal output.** The ledger's
PASS rows are sound (§1 is clean, and so is §2). What is failing is the layer
that reports them.

Ranked by damage to the trustworthiness of what this project publishes:

| | finding | age | reads |
|---|---|---|---|
| **1** | the steering-order reader sees 0 of 13 live builder orders and prints a false reason | 5 d | no gate; ungated by any spec |
| **2** | the owner-facing "first legal slot" is wrong on 3 consecutive days — systematic off-by-one, **~4 h optimistic today** | 3 d | mine, and my predecessor's |
| **3** | the Review's "re-run after the last act" block quotes `pass_on_dead_dependency` **5**; live was already **6** | 1 d | above floor 3 |
| **4** | `dark_slots` **100** / 99.8 h, longest on record; W40's 30 free GPU-h will be the **fourth** consecutive week lost | 4 d | floor 0 |
| **5** | field watch week 10: **3 of 5** finding sections UNROUTED, all three about the field-watch organ itself | 1 d | unfloored by design |

**Fired this audit, per rule 3:** `D37`'s armed default, on its first legal day.
The OVERDUE class in `decisions --check` is now empty. Details in §6.

---

## RANK 1 — The `FOR THE BUILDER` reader has been blind for five days, and it says so in a sentence that is not true

**THE MEASUREMENT, taken live rather than inferred.**

```
$ python -c "from experiments import steering; print(len(steering.read()))"
0

docs/PROGRESS.md    _BUILDER_HEADING hits: 1   builder_items(): 0
docs/OVERSIGHT.md   _BUILDER_HEADING hits: 1   builder_items(): 0
```

Both headings are found. The items are not. `steering.py:113` is

```python
_ITEM = re.compile(r"^(\d{1,2})\.\s+(.*)$")
```

and every live order on both pages is written `**N. THE ORDER IN BOLD.** The
reasoning.` — the `**` precedes the digit, so the line never matches. Counted in
the two live sections: **PROGRESS.md items 0–6 (7 orders)** and
**OVERSIGHT.md items 0–4 (5 orders, my predecessor's)** — 12 numbered orders plus
`T1.08` Steps 0+1 carried as PROGRESS item 1's two-part unit. **Zero are read.**

**WHEN IT BROKE, bisected rather than guessed** (counting plain-form vs
bold-form item lines inside the section at every commit that touched each page):

| page | last commit with readable orders | first commit with none | dark for |
|---|---|---|---|
| `docs/OVERSIGHT.md` | `5a692015` 2026-09-30 06:49 (2 items) | **`94d0fb24` 2026-09-30 12:54** | **5 days, 8 consecutive pages** |
| `docs/PROGRESS.md` | `3fcad583` 2026-09-30 06:37 (3 items) | **`8a42c102` 2026-10-04 07:06** | 1 day |

**The worse half is the middle period, and it is the part no reader could have
caught.** Between 2026-09-30 12:54 and 2026-10-04 07:06, PROGRESS.md still had
plain-form items and OVERSIGHT.md did not. `render()` only emits the
"no section found" line when **both** pages come back empty, so for four days it
printed a confident head line of the form *"N item(s) on **1** page(s)"* — while
`STEERING_PAGES` declares **two**. A reader reporting from one of two declared
pages, with no line saying the other was empty, is precisely the failure the
module's own docstring says it exists to prevent:

> *"a page with no orders at all must be visibly distinguishable from a page
> whose orders are all fine"*

It is not distinguishable. Eight consecutive OVERSIGHT pages carrying 4–8 orders
each went into a channel that reported them as absent, and in the four days when
the defect was still half-hidden it reported them as *not existing on a page it
did not name*.

**AND NOTHING GATES IT.** `grep -rln steering experiments/tests/` returns two
files, both incidental (`t2_19`, `t2_11`); no spec asserts that
`steering.read()` is non-empty, that both declared pages contribute, or that the
reader's own item count is above zero. Compare the three instruments this
project has already paid for shipping with one counted class — `coverage.py`,
`decisions.py`'s `NO-DEFAULT`, `champions.py`'s `ARENA-MISSING` — each of which
got a "repair" that lowered its own number, and all three of which are now
ratcheted by `T0.31`'s P4/P5/P6. **The steering reader has no `T0.31`.** It can
go to zero, print a false reason, and no exit code moves.

**WHY THIS IS THE RANK 1 FINDING AND NOT A TIDY-UP.** `steering.py:3-4` records
its own provenance: *"THE SCAR (overseer, 95th audit, `docs/OVERSIGHT.md` RANK 3
and FOR THE BUILDER item 2 — this module is that item)."* This organ
commissioned this reader so that its orders could not die as prose. The reader
now cannot read this organ's orders. For five days every FOR THE BUILDER item
from both desks has gone out with its legality unchecked — nobody has verified
that an order does not name a spec the runner would refuse, which is the one
question the module computes.

**MY OWN HALF OF THE REPAIR, TAKEN TODAY RATHER THAN ROUTED.** The FOR THE
BUILDER section at the bottom of this page is written in the **plain
`N. **ORDER**` form**, which is the form the reader parses and the form
`builder_items`' own docstring names as house style. I verified
`steering.read()` sees them before committing — the count is in the closing
block. That restores the channel on *this* page today. It does not fix
PROGRESS.md and it does not fix the reader, and I am not touching
`experiments/steering.py`: `D13` forbids me the code.

---

## RANK 2 — The owner-facing release forecast has been wrong three days running, the error is systematic, and today it is ~4 hours optimistic. This is my organ's error.

`docs/PROGRESS.md` FOR THE OWNER item 2 and my predecessor's page both publish
**"first legal slot ≈2026-10-06 16:07 UTC"**, the second one presented as a
correction of the first (*"corrected from 14:10 — the meter's 82 → 83 rise cost
two hours"*). **16:07 is not the answer for 82 %, for 83 %, or for 84 %.**

**THE BUG, at source.** `scripts/lib_usage.sh:85-86`:

```bash
allow=$(( PACE_FLOOR + ((PACE_CAP - PACE_FLOOR) * elapsed + 99) / 100 ))
if [ "$pct" -ge "$allow" ]; then   # ... return 1   (skip)
```

The builder proceeds iff `pct < allow`, so release requires **`allow ≥ pct + 1`**.
My predecessor's derivation (`docs/OVERSIGHT.md:144-147`) used **`allow ≥ pct`**:

> *"At meter 82 (yesterday): release needs `allow ≥ 82` → `elapsed ≥ 87`…
> At meter 83 (today): `elapsed ≥ 88` → … first hourly slot ≈2026-10-06 16:07."*

`allow ≥ 83 → elapsed ≥ 88` is the correct arithmetic **for pct = 82**. So the
off-by-one makes each day's "corrected" forecast reproduce the *previous* day's
meter value. That is why correcting 14:10 → 16:07 still landed on a stale
premise: both numbers came from the same one-point-too-loose comparison, and
neither was ever the answer to the question being asked.

**THE CORRECTED NUMBERS, by simulating the gate hour by hour rather than solving
it in prose** (reset epoch read from the CLI's own field, `resets Oct 7, 12pm
(UTC)` → week start 2026-09-30T12:00:00Z exactly, not the ≈11:55 derived by hand
from log crossings):

| meter | elapsed needed | line | **first legal slot** |
|---|---|---|---|
| 82 % | 88 % | 83 % | 2026-10-06 **15:07** UTC |
| 83 % | 90 % | 84 % | 2026-10-06 **19:07** UTC |
| **84 % (live, 06:07 today)** | **91 %** | **85 %** | **2026-10-06 20:07 UTC** |
| 85 % | 93 % | 86 % | 2026-10-07 **00:07** UTC |
| 86 % | 94 % | 87 % | 2026-10-07 **01:07** UTC |

**So the owner has been told the blackout ends ~4 hours earlier than it does**,
and the meter rose **83 % → 84 %** at 06:07 this morning, which the published
figure predates. The hard ceiling is the one number my predecessor got right and
I confirm it from the CLI rather than from log crossings: at the week reset
`elapsed → 0`, `allow → 25`, and the weekly percentage resets with it, so the
blackout cannot outlast **2026-10-07 12:00 UTC** (first slot 12:07).

**WHY THIS IS A RANK 2 FINDING AND NOT A TRANSCRIPTION SLIP.** Three
consecutive sittings have hand-derived this number, each has published it to the
owner as the headline cost of the blackout, each has presented itself as
correcting its predecessor, and all three were wrong in the same direction by
the same mechanism. It is a pure function of two live readings (`--pct`,
`resets`) and two constants in a file every desk already reads. Prose arithmetic
on a number this load-bearing is the defect; the fix is to compute it. Routed as
FOR THE BUILDER 2.

---

## RANK 3 — The Review's "every one re-run after the last act" block quotes a ratchet value that was already stale, and the missing delta is the one its own act created

`docs/PROGRESS.md` opens its instrument block with an explicit promise:

> *"**INSTRUMENT EXIT CODES, every one re-run after the last act of this sitting
> and not quoted from the top of it** (the 06:37 overseer collision makes a stale
> reading the default failure here)"*

The five exit codes in that block are correct — I re-ran all five and got
`coverage 2`, `decisions 1`, `champions 0`, `status 2`, `review-queue 2`. But the
same sentence lists *"the four ABOVE-floor ratchets"* as `dark_slots 76,
decisions_default_action_expired 1, **pass_on_dead_dependency 5**, unreachable
96`, and **`pass_on_dead_dependency` was already 6.**

**MEASURED, at the sitting's own last act.** I read the committed ledger at
`b61515f` (Review FULL act 19, the last act of that sitting) and at every commit
since:

```
b61515f  T1.11=FAIL T1.12=PASS  T0.13=FAIL T0.18=PASS T0.19=PASS
         T1.08=FAIL T2.03=PASS T2.14=PASS  T6.03=BLOCKED LF.02=PASS
90cd178  (identical)      32b098c  (identical)      HEAD  (identical)
```

All six pairs — `LF.02←T6.03`, `T0.18←T0.13`, `T0.19←T0.13`, **`T1.12←T1.11`**,
`T2.03←T1.08`, `T2.14←T1.08` — existed at `b61515f`. The sixth,
`T1.12←T1.11`, was **created by that sitting's own act 12b** (`c7b4bb9`, 06:59),
the `T1.11` demotion the page is otherwise right to be proud of, eleven minutes
before its last act.

**THE MECHANISM, and it matters more than the one-count.** `run status` prints
this counter as `pass_on_dead_dependency = 6  !! MOVED +1 since 2026-09-26
(was 5)`. **5 is the `(was …)` number — the committed reading from 09-26.** The
block quoted the ratchet's *remembered* value instead of its *live* one. That is
the same defect the sentence was written to prevent, wearing different clothes:
not a reading from the top of the sitting, but a reading from the last time
anyone recorded it. `dark_slots 76` in the same list *is* live (its committed
reading is 0), so the sitting plainly ran the tool — it read the wrong column.

**Damage.** `pass_on_dead_dependency` is **above its declared floor of 3** and is
one of the four counters that drive `status`'s exit 2. A standing PASS resting on
a recorded non-PASS dependency means the board renders a claim that could not be
re-derived today, and `T1.12` ("Flow matching actually denoises") is now one of
them. The page under-reported the size of its own most important finding.

---

## RANK 4 — Builder dark 100 slots / 99.8 h, the longest on record; and W40 is on course to be the fourth consecutive week of free GPU-hours lost

Carried from four previous audits and re-derived, not inherited. From
`/data/jack-logs/ladder.log`, last line 2026-10-05T06:07:12Z:

```
'week:all models' 84% at 68% of the week (line 70%); week:Fable 52% (not the gate);
of this week's 83 shared point(s): builder 16 (19%), desks 7 (8%),
NOT THIS PROJECT 60 (72%); 99 consecutive dark slot(s);
0 failed slots (99.8 h since the last rc=0)
```

`dark_slots` reads **100** against a declared floor of **0** — 50× its `D30`
trigger of 2× the hourly cadence. Last `rc=0` **2026-10-01T02:17:07Z**. **Zero
failed slots**: the loop is healthy and is being paced out, not crashing. There
is no `.usage-resumed` on disk, so `lib_usage.sh`'s 90 % hard stop is armed and
unattended — and at 84 % it is six points away, with the meter rising ~1 pt/day.

**And 72 % of the meter silencing this project was not spent by this project.**
That is the single most important line in this section and it is unchanged across
five audits.

**THE PERISHABLE COST, re-derived from `experiments/gpu_budget.json` rather than
quoted.** Kaggle-only hours drawn, by week:

```
2026-W36  17.724 h      2026-W38   0.918 h
2026-W37   1.379 h      2026-W39   1.072 h      2026-W40   NO ENTRY — 0.000 h
```

W37–W39 lost ≈**83 free GPU-hours** across three weeks. **W40's 30 h expire
Saturday 2026-10-10.** The builder cannot wake before **2026-10-06 20:07** (RANK
2), hard-capped at 2026-10-07 12:07 — leaving ~3.5 days of W40, in which the only
**designed** buyer is `T1.08` Step 1 at ~0.3 h. On the measured record W40 ends
at ≈0.3 of 30 h and the three-week figure becomes a **four-week ≈113 h**. No
dispatch should be manufactured to spend them and I am not asking for one; the
number is the cost of the pacing decision, which is `D40`'s and the owner's.

**Separately, from `gpu_hours_no_verdict`, unchanged and still the largest single
waste on the board:** 49.49 h total with no verdict, of which **`D1.0` is
33.78 h across 2 attempts for 0 verdicts**. `gpu_unattributed_jobs` 21, AT floor.

---

## RANK 5 — Field watch week 10 landed this morning with 3 of its 5 findings routed nowhere, and all three are findings about the field-watch organ itself

`run status`, live:

```
FIELD-WATCH FINDINGS — week 10: 5 finding section(s); 0 cited, 2 quoted,
                                3 UNROUTED-FIELD-FINDING
  §6   ROUTED — quoted by t406-latent-floor-was-never-computed
  §7   UNROUTED — "my enumeration script died silently at import, and a livene…"
  §7b  UNROUTED — "my own verification grep false-positived on English pro…"
  §7c  ROUTED — quoted by field-watch-rc124-page-is-untrustable-and-then-deleted
  §7d  UNROUTED — "a fourth finding, caused by trying to publish the three above…"
```

`docs/FIELD_WATCH.md` is rewritten weekly (`32b098c` replaced 1,599 lines this
morning), so these three expire on **2026-10-12** unless a desk quotes them. The
reading is reporting-only and unfloored by `D27`'s own reasoning — *a finding may
legitimately be discharged in code, which no desk file shows* — which is exactly
why an organ's self-diagnosis needs a row rather than a paragraph. §7 and §7b are
the two that matter: a sweep script that **died silently at import** and a
verification grep that **false-positived on prose** are faults in the instrument
that produced the week's other findings, and they are the same two failure modes
this project has paid for before. Routed as FOR THE BUILDER 3, which is a
reporting change only — I may not write `REVIEW_QUEUE.md`.

---

## The audit, section by section

### §1 — Integrity of the ledger: CLEAN, and that is a real result

Checked, not assumed:

- **106 of 106 PASS rows name a commit that still exists in git.** Zero missing.
- **Zero PASS rows name a spec absent from the registry** (255 registered).
- **Zero PASS rows lack an implementation** — `run status` renders
  `(not implemented)` separately and no such spec carries a PASS.
- **2 PASS specs declare no `control`: `T0.01`** (repo imports clean) **and
  `T0.10`** (Kaggle job round-trip). Both are harness round-trips where a null
  arm is arguably undefined — and this is **already owned with a clock**:
  `t018-explicit-no-control-reads-as-an-unrun-promise`, OPEN, which fell DUE
  **today** and was re-dated by the Review's DAILY sitting in flight. Not a new
  finding; I confirm the class is covered and name it so it is not double-counted.

What the record *does* say against itself, all of it printed by the instrument
and none of it hidden: **2 DIRTY STAMPS** (`T6.03`, `PL.02` — ran from modified
trees), **15 STALE CLAIMS** (a path inside `impl_sha` moved after the run that
recorded it; 11 of the 15 are the spec's own test file), **6 UNBACKED
CERTIFICATES**, and **549 metrics recorded but read by no conjunct** across 58
certificates — the last at a hand-measured **95 % false-positive rate**, so it is
correctly unfloored and cannot be acted on spec-by-spec. None of these is new and
each is reporting-only by a recorded decision.

### §2 — Thresholds and controls over time: NO LOOSENING FOUND

Ten commits touched `experiments/registry*.py` or `experiments/tests/` in seven
days. I read the diff for every hit on `MIN|MAX|_FLOOR|_CEIL|THRESH|_BAR|seeds|
control|_check| or `. **The ratchet moved in the tightening direction every
time**, and in one case cost a certificate:

- **`T1.11` STRENGTHENED** (`04f99d1`, `c7b4bb9`): a third conjunct
  **conjoined**, `SHIPPED_CALLER_MIN = 1`; both original conjuncts and both
  original constants byte-unmoved; control untouched; demotion **pre-registered
  in the diff before the run** and it demoted PASS → FAIL. This is the system
  working exactly as designed and it is the best thing on the board this week.
- **`T0.31` strengthened 22 → 24 properties** (`5651fc1`).
- **`DP.04`: `NEED_MIN_GAIN` None → 35.0** (`5b0d4c0`) — a *registration* of a
  previously-unset gate, `ceil(34.0206765975521)` by a **pre-declared** rule,
  rounded **up**, with companions derived at fixed multiples. Not a loosening;
  and the pre-check then read *below* the new bar, predicting a VOID.
- **`LG.14` registered** (`87bc128`) with every bar quoted from source and
  byte-unmoved, an UNSATURATED-NULL declaration, and 0 billed certificates.
- **One flag word checked and cleared:** `0ac932b`'s message says *"the ordered
  T2.08 softening"*. It is **not** a spec threshold. The change is in
  `worst_seed_audit`, a reporting tool, reclassifying a reader verdict `WRONG` →
  `PROTECTED`; `experiments/tests/t2_08_curiosity_coverage.py` is **untouched in
  seven days** and the 0.05 bar did not move. The commit was ordered by the 134th
  audit's own FTB 2.

No threshold moved down, no control was deleted or weakened, no `_check` gained
an `or`, no seed count was reduced, no assertion was removed, and no FAILING or
VOID spec was rewritten to pass. **Section 2 is clean and saying so is the
result.**

### §3 — Drift from the goal

**The builder worked on nothing in the last day — 100 dark slots — so there is no
builder work to trace to a GOAL.md sentence.** The desks worked: a Sunday FULL
Review, two regate sweeps, a week-10 field watch, and a DAILY Review running
concurrently with this audit.

**Re-derived with my own numbers rather than carried from PROGRESS.md** (7 days
to 2026-10-05 06:45):

```
191 commits total
  0  touched UnifiedBrain.py          0  touched survival.py
  0  touched TrainingPipeline.py      0  touched EpisodicMemory.py
  0  touched playground.py
143  touched docs/ or scripts/       10  touched experiments/tests/
```

My count differs from the page's 249 only by window. **The finding is identical
and it is the one that matters: not one commit in seven days touched the thing
that is supposed to learn.** Every organ is serving *"protects the honesty of
watching what happens when the three meet"* — the fourth clause of GOAL.md's
first principle — and none is serving the first three. That is not drift in the
sense of work serving no sentence; it is the whole project living in one clause.

**The converse, which is the harder question.** From `coverage`: **3 commitments
CLAIM-DEAD** — `smell`, `shelter/building`, `thermal ("too cold kills him")` —
each with every claim spec PARKED or FORECLOSED, unchanged for **9 days**. **14
more have live claim specs and nothing passing**, including `touch/contact`,
`tool use`, `told world`, `proprioception`, `sleep`, `plasticity` and
`fast/slow`. `commitments_uncovered` is **0** and AT floor, so the §FIRST check
passes and no commitment is without a spec — but **curiosity is 2 of 12 passing,
one-brain/unison is 1 of 28, and learning-by-living has `LT.03` VOID and `LT.02`
FAIL.** The three claims GOAL.md says are most likely to be quietly neglected are
exactly the three that are. `claim_dead = 3` is now routed where it can be
answered (`D42`), which is new this week and is the right move.

**`NO-LIVE-PATH`: 6 distinct commitments/seats with no live path at all** (3
CLAIM-DEAD + 4 unwinnable seats, 1 seen by both). The repair for every member is
a **registration**, and no instrument can ask for it, because a missing spec has
no id.

### §4 — Is the builder alive and productive?

**Alive, healthy, and gagged.** 0 iterations in 24 h; 0 `rc=0`; 0 failed slots;
PASS delta from builder work **0**. Every slot for 100 consecutive hours ended
`PACE-SKIP`, not error. No paused loop, no crash, no credit exhaustion, no
aborting on load. The cause is entirely the shared usage meter at 84 %, 72 % of
which is another tenant's — see RANK 4. The `demonstrated` count fell
**107 → 106** and the fall is the Review's own strengthening of `T1.11`, not a
regression.

Three `PACE-SKIP NOTICE` lines have repeated every hour since 10-01 for
`run_spec T0.21`, `T0.28`, `T0.31` — detached dispatches that **EXITED
2026-10-01T02:17:07** and may hold artifacts outside the harvest paths. 100 slots
of notices and no slot able to read them. Harmless today, but it is a queue of
unharvested work the first live slot inherits.

### §5 — Compute honesty

`2026-W40`: **0.0 h of 30 free Kaggle GPU-hours drawn**, expiring Saturday
2026-10-10. W37/W38/W39 lost 1.379 / 0.918 / 1.072 h of ~30 each. **49.49 GPU-h
recorded with no verdict**, `D1.0` alone 33.78 h for 0 verdicts across 2
attempts — found and reported, cause is the VOID-FORECLOSED state of that arm,
not a leak. `gpu_unattributed_jobs` 21, AT its declared floor. No GPU-hour was
spent this week, so no hour was wasted this week; the waste is the **unspent**
quota, which is RANK 4.

### §6 — Stuck decisions, and the one I fired

**`D37` — FIRED, this audit, on its first legal day.** `decisions --check`
printed it as the single `OVERDUE — DEFAULT IS DUE TO FIRE`; `decide_by` was
**2026-10-04**; its own 10-04 addendum states the firing date is 2026-10-05.
**The owner did not rule by 2026-10-04, so the pre-registered default fired** —
option **(iii) HOLD `D29` AS IT STANDS**, recorded in `DECISIONS_NEEDED.md` with
the monotonicity argument, the price quoted from the entry, and the reversal
named. It is the status quo: the debt stays recorded in two places, the `Δ_k`
readout is **not** built, no threshold moved, nothing was spent. Option **(i)
BUILD THE DIAGNOSTIC** — the entry's own recommendation — was **not** taken and
remains the owner's to rule at any time. I verified at source that the firing
orders no work: `effective_rank` still has no computing call site, and the
Learning-core seat's `HELD: BY VERDICT` marking is untouched. **The
transcription onto `DECISIONS_RESOLVED.md` is the Review's**, per `D13`.

**`MEANS-ESCALATED`: none.** No fork that a measurement could settle is sitting
on the owner's desk. The D1 disease is absent this morning and that is worth
stating plainly.

**`UNDECLARED`: none.** `decisions_undeclared = 0`, AT floor. My standing
instruction is to arm at least one per audit; **there is nothing to arm**, and
inventing one would be manufacturing a finding. The ratchet has not grown.

**Still broken, and not mine to clear: `decisions_default_action_expired = 1`,
floor 0, caused solely by `D33`.** Its `decide_by` was 2026-09-23 — now **12
days** past. The 10-02 Review established the default is **MOOT, not merely
expired**: its object went terminal when `w1-world-edit-window` was stamped
`DECLINED`, so **no desk can clear this by firing anything.** I re-derived that
and agree. It is the sole cause of the one broken class on the owner's register.

**`D38` is a desk obligation that is now past its date and it is the Review's,
not mine.** `CONDUCT-DESK`, `decide_by 2026-10-04`, **STALE by 1 day** —
*"desk-executable, not the owner's — execute it, report it, do not ask."* It
arbitrates which of two armed defaults (`D28` OVERDUE-first vs `D33` W1-first)
owns the FULL Review's first act. Its default (i) CHANGE NOTHING is monotone and
self-realising, so nothing is broken today; but the next FULL is **2026-10-11**
and an entry whose deadline has passed reads as an unmade decision. Named here
because the Review reads this page. `D35`, `D39`, `D40` are also `CONDUCT-DESK`;
`D35` is stale by 11 days.

**Quietly acted on without being recorded: nothing found.** I checked the last
7 days of commits against `DECISIONS_RESOLVED.md` and found no owner decision
executed without a record.

**One correction to my own predecessor's page, now fixed by this rewrite.**
`run status`'s `STEERING-DATE-MISMATCH` flagged *"`D33` docs/OVERSIGHT.md says
2026-09-12 — register says `decide_by 2026-09-23`."* That misquote was on the
139th audit's page and is gone from this one.

### §7 — Bakeoff hygiene

No decision was made without a learning gate this week, and no winner was chosen
inside a noise margin. The standing bad news is structural and unchanged, so I am
not re-litigating it — `champions --check` **EXIT 0** with all 10 violations **AT
their declared floors**, `champions_unwinnable` 4 AT floor,
`champions_trigger_debt` 3. **No seat lost a door in this window.**

The two markings that remain indefensible on their own evidence, restated once
because `champions` exits 0 and a reader could mistake that for health:

- **Learning core is held `BY VERDICT` off `LC.03`, which is a VOID** — and
  `SYSTEM.md` says a VOID decided nothing. All three pre-registered re-open
  triggers are closed doors (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED,
  `UB.10` VOID). The strongest marking in the file rests on a non-verdict.
- **World is held `BY VERDICT` and names neither a deciding row nor a rematch
  trigger** — `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED`. An unwritten promise
  cannot decay visibly.

Both are AT floor, both are recorded, and the honest repair for each is a
redesign, a re-parenting or an honest re-marking — never deleting a trigger.
This is `D37`'s substance too, and `D37`'s default has now locked the weakest
option in.

### §8 — The honest summary: are we closer to a curious humanoid, or to a longer list of green ticks?

**Neither, this week — and that is a worse answer than either.** The list of
green ticks got *shorter*, honestly, from 107 to 106, because a desk strengthened
a test and demoted it; the registry did not grow; the Goodhart check came back
clean for the right reason. That is the scoreboard behaving exactly as it should,
and it is real.

But Jack did not move, and he has not moved in seven days — **0 of 191 commits
touched `UnifiedBrain.py`, `TrainingPipeline.py` or `playground.py`.** The builder
has been silent for 100 hours for a reason that is nobody here's fault and
nobody here's to fix: 72 % of the usage meter that silences it belongs to another
tenant on this box.

And the thing I have to report that is mine: **this week the apparatus that
watches Jack started getting its own readings wrong.** Three of my five findings
are not about Jack at all — they are about a reader that reports its own emptiness
in a false sentence, a forecast hand-derived wrong three days running and
published to the owner each time, and a ratchet value quoted from memory inside a
paragraph swearing it had been re-read. Each was caught by reading an instrument's
own output against the prose beside it. None would have been caught by looking
harder at the ledger.

So the honest position is this. The ledger is trustworthy: §1 is clean, §2 found
no loosening, 106 of 106 PASS commits exist, and the one certificate that fell
this week fell because someone made the test harder. What is **not** currently
trustworthy is the reporting layer — and since the reporting layer is how the
owner and the builder learn what the ledger says, that is the more urgent of the
two. A project whose scoreboard is sound and whose dashboard drifts will keep
making correct measurements that nobody acts on, which is indistinguishable from
not measuring. We are not closer to a curious humanoid than we were yesterday.
We are closer to knowing which of our own instruments to stop believing, and
after five days of invisible builder orders that is worth something — but it is
not progress toward the ladder and I will not dress it up as such.

---

## FOR THE BUILDER

0. **EVERY ORDER FROM THE 135th–139th AUDITS IS STILL OPEN, AND SO IS EVERY ITEM OF THE REVIEW'S PROGRESS FTB 1–6.** Your last `rc=0` was 2026-10-01T02:17:07 and all of it was written after that. None of it is your fault and I am not re-ranking it. Read the 139th audit's items as live — **and read them from `git show 8f3337bc:docs/OVERSIGHT.md`, because this page has replaced them and item 1 below explains why you could not have seen them anyway.** Nothing here displaces the Review's FTB 1 (`T1.08` Steps 0+1), which stays the highest-value unit on your board and whose Step 0 is free.

1. **FIX `steering._ITEM` SO IT MATCHES A BOLD-LEADING NUMBERED ITEM, AND GATE THE READER SO IT CANNOT SILENTLY READ ZERO AGAIN.** `experiments/steering.py:113` is `^(\d{1,2})\.\s+(.*)$`; both steering pages write `**N. THE ORDER**`, so `builder_items()` returns 0 on both while `_BUILDER_HEADING` matches both. Live proof: `python -c "from experiments import steering; print(len(steering.read()))"` → `0`. Allow optional leading emphasis (`^\**(\d{1,2})\.\s+`) and **keep the existing plain form matching** — do not swap one exclusive form for another, or you will re-break it the next time a desk changes style. Two more things in the same commit, and the second is the one that matters: (a) `render()`'s empty message says *"no `## FOR THE BUILDER` section found"* when the heading **was** found — make it distinguish *heading absent* from *heading present, zero items parsed*, because the current wording sent five days of readers looking for the wrong defect; (b) make `render()` name **every** declared page that contributed zero items, so *"N item(s) on 1 page(s)"* can never again stand in for *"one of two declared pages is empty"*. **Then ratchet it**: no spec asserts this reader is non-empty, which is why it went to zero unnoticed — `T0.31` is the precedent and the right home is a property that fails when a live `FOR THE BUILDER` section parses to zero items. Report the item count before and after.

2. **COMPUTE THE FIRST-LEGAL-SLOT FORECAST IN AN INSTRUMENT INSTEAD OF LETTING THREE DESKS DERIVE IT WRONG IN PROSE.** The release condition is `pct < allow` (`scripts/lib_usage.sh:86`), so release needs `allow >= pct + 1`; the 139th audit used `allow >= pct` and published ≈2026-10-06 16:07, which is the arithmetic for meter **82 %**, not the 83 % it claimed, and the live meter is now **84 %**. Correct answers, by simulating the gate hour by hour: **82 % → 15:07, 83 % → 19:07, 84 % → 20:07, 85 % → 2026-10-07 00:07.** Add a `pace-forecast` reading that prints the next slot at which `pace_gate` would return 0, derived from `claude_usage.py --pct`, the CLI's own `resets` field (`Oct 7, 12pm (UTC)` → week start 2026-09-30T12:00:00Z — **read it, do not re-derive it from log crossings**), `PACE_FLOOR` and `PACE_CAP`, plus the sensitivity row (what one more meter point costs). **Reporting-only; gate nothing, and do not touch `PACE_FLOOR`, `PACE_CAP` or the 90 % stop** — a forecast that could move the line would be a loosening wearing a convenience.

3. **MAKE `FIELD-WATCH FINDINGS` SURVIVE THE WEEKLY REWRITE, OR AT LEAST SAY WHAT IT IS ABOUT TO LOSE.** Week 10 landed with **3 of 5** sections `UNROUTED-FIELD-FINDING` (§7, §7b, §7d) and `docs/FIELD_WATCH.md` is rewritten weekly — `32b098c` replaced 1,599 lines this morning — so those three expire **2026-10-12** with no desk record. All three are findings about the field-watch organ itself (a sweep script that died silently at import; a verification grep that false-positived on English prose). Reporting-only repair: print, beside the count, **the rewrite date at which each unrouted finding will vanish**, and carry the unrouted set forward from `docs/FIELD_WATCH_LOG.md` so a finding that was never routed is distinguishable from one that was discharged. **Do not auto-route anything** — routing is a desk act and `D27`'s reasoning (a finding may legitimately be discharged in code) is why this class is unfloored.

4. **HARVEST THE THREE EXITED DETACHED DISPATCHES BEFORE ANYTHING ELSE IN YOUR FIRST LIVE SLOT.** `run_spec T0.21` (2630305), `T0.28` (2630448) and `T0.31` (2630920) all **EXITED 2026-10-01T02:17:07** and have printed a `PACE-SKIP NOTICE` every hour for 100 slots saying they may hold artifacts outside the harvest paths. 100 slots of notices and no slot able to act. Read them, record what they bought, and clear the notices — a notice that repeats 100 times and is never consumable is training every future reader to skip it.

5. **WRITE YOUR OWN NUMBERED ORDERS IN THE FORM YOUR READER PARSES, AND PREFER THE INSTRUMENT'S COLUMN TO YOUR OWN ARITHMETIC.** This page's FOR THE BUILDER is deliberately written `N. **ORDER**` rather than `**N. ORDER**`, which is the form `builder_items()`'s docstring names as house style and the only form it matches; I verified `steering.read()` sees these items before committing. Until item 1 lands, **a bold-leading numbered order is an invisible order.** And the discipline behind RANK 3: `run status` prints `pass_on_dead_dependency = 6 !! MOVED +1 since 2026-09-26 (was 5)` — **6 is the live value and 5 is the remembered one.** Quote the live number, never the `(was …)`.

---

## FOR THE OWNER

1. **`D37` fired this morning on its first legal day — the weakest of its four options, locked in on the first day the missing premise could have mattered. One sentence from you reverses it.** The owner did not rule by 2026-10-04, so the pre-registered default fired: **(iii) HOLD `D29` AS IT STANDS.** The `Δ_k` collapse diagnostic is **not** built. Nothing moved, nothing was spent, no threshold changed — it is the status quo, which is why it was the only legal default. Its price in the entry's own words: *"the project keeps a `mandatory` guard it has never once been able to run."* Concretely, `LEARNING_CORE.md` §5.4 promises `A4` a **mandatory** collapse diagnostic that no code in this repo computes, and `A4` holds the Learning-core seat **BY VERDICT off a VOID** with all three re-open triggers closed. Option **(i) BUILD IT** stays the entry's own recommendation and yours to rule at any time; the work is the builder's and small (the readout and a pre-registered floor). Nothing is waiting on it this week — every arm behind that seat is foreclosed or unaffordable — so this is a question about whether the *next* learning-core arm is falsifiable, not about this week's board.

2. **NO-DECISION — the blackout report `D30`'s armed default requires, with one correction you should have: the builder cannot wake until ≈2026-10-06 20:07 UTC, four hours later than you were told yesterday.** Builder dark **100 slots / 99.8 h**, the longest on record, **0 failed slots** — the loop is healthy and paced out, not broken. The published ≈16:07 figure was derived with the release comparison one point too loose and matches no meter value; corrected by simulating the gate: **84 % (live) → 2026-10-06 20:07**, and every further meter point costs ~1.7 h more. Hard ceiling, independent of the meter: **2026-10-07 12:07 UTC**, when the week resets. **`week:all models` reads 84 %** and the 90 % hard stop is six points away, armed, with no `.usage-resumed` on disk. **72 % of this week's shared pool is another tenant's.** The perishable cost: **`2026-W40` has 0.0 h drawn of 30 free Kaggle GPU-hours, expiring Saturday 2026-10-10**; W37–W39 lost ≈83 h; on the measured record W40 becomes the fourth consecutive week and the running total ≈113 h. The only designed buyer is `T1.08` Step 1 at ~0.3 h. `demonstrated` 107 → **106**, and the fall is a desk strengthening a test, not a regression. Nothing here needs a ruling — `D40` (armed, `decide_by 2026-10-10`) is where the pacing question lives and its measurements hold; I re-derived them independently.

3. **The thing I most want you to know this week is not about Jack: for five days, the channel both desks use to give the builder orders has been reporting that it is empty, and nothing could tell.** Twelve numbered orders across two pages — eight consecutive reports from this desk, carrying 4–8 orders each — went into a reader that matched **zero** of them and printed *"no `## FOR THE BUILDER` section found"* about pages that plainly have one. The cause is one regex expecting `1.` where both desks now write `**1.`; no spec gates the reader, so it could fall to zero and no exit code moved. I have half-repaired it inside my own permissions (this page's orders are written in the form the reader parses, verified before commit) and routed the code half as FOR THE BUILDER 1. **No ruling is asked.** I report it because it bears directly on something you are entitled to assume: that when a desk writes an order down, the system can see it. For five days that was false, and the two other findings on this page are the same shape — a number re-derived in prose and published while the instrument holding the correct value sat in the same output. The ledger is sound; the reporting around it is what needs watching.

4. **Nothing new on `D33`, `D41` or `D42` — pointers only, so a re-ask does not become wallpaper.** `D33`: `decide_by` was 2026-09-23, now **12 days** past, and it is the sole cause of the one broken ratchet class on your register (`decisions_default_action_expired = 1` against floor 0). The 10-02 Review established its default is **MOOT, not merely expired** — its object went terminal when `w1-world-edit-window` was stamped `DECLINED` — so **no desk can clear this by firing anything**; I re-derived that and agree. `D41` (which artefact is Jack — the ladder's rig or `TrainingPipeline.py`) and `D42` (which of three claim-dead commitments gets a successor) are both armed with monotone defaults, `decide_by 2026-10-18`, and I am not re-arguing either. On `D42` I will say only that the underlying reading is unchanged and independently confirmed: **`claim_dead` has read 3 for nine days** — **smell**, **shelter/building**, **thermal ("too cold kills him")** — and `coverage` counts **6 distinct commitments or seats with no live path at all.** Every park was right on its evidence; the bug is leaving the commitment claim-dead, and no instrument can ever ask for the fix, because a missing spec has no id, blocks nothing and fails no gate.

5. **NO-DECISION — what this sitting did not do, named rather than omitted.** I did not audit the **cognitive half** of the sensory/capability completeness list — attention, working memory, imagination, self-model, theory of mind, teaching. It is owned with a clock at `completeness-audit-2026-09-13-the-cognitive-half-is-the-hole` (OPEN, DUE 2026-10-11), so it ages in public rather than inside this paragraph. I spent the sitting's clock on RANK 1 instead, on the ground that an unreadable order channel makes every other finding on this page undeliverable. I also did **not** arm a new decision: `decisions_undeclared` is **0** and AT floor, so there was nothing to arm, and manufacturing one to satisfy a quota would be the opposite of the rule's purpose. **And one disclosure about my own conduct:** `scripts/ladder_prompt.md` stands at 121,041 bytes with **3,959 bytes of headroom** against its 125,000 ceiling and 13 days to it at the measured +296 B/day. I added nothing to it this sitting.

---

## CLOSING BLOCK — every instrument re-run AFTER my last act, not quoted from the top of the sitting

Re-run at **2026-10-05 06:5x UTC**, after the `D37` firing and after this page
was written, at a `HEAD` that moved five times under me (last seen `4b76d80`,
Review DAILY act 5). **Not one of these reds is new, and none is mine:**

```
coverage          EXIT 2      decisions --check  EXIT 1      champions --check  EXIT 0
run status        EXIT 2      run review-queue   EXIT 2
```

**SLOT LINE, quoted from the instrument rather than composed** (an exit code is a
LEVEL; a ratchet reading is a DELTA):

```
ratchets vs committed readings (HEAD): 7 MOVED (dark_slots 0 -> 100,
  fail_unowned_owned_forms queue-row 31 -> 32, pass_on_dead_dependency 5 -> 6,
  review_queue_net_arrivals 32 -> -2, review_queue_piled_on 4 -> 9,
  review_queue_violation_forms {HOLD-ON-A-RESOLVED-BLOCKER 8, OVERDUE 6} ->
  {HOLD-ON-A-RESOLVED-BLOCKER 7}, review_queue_violations 14 -> 7);
no counter refused to compute; floors: 4 ABOVE (dark_slots,
  decisions_default_action_expired, pass_on_dead_dependency, unreachable),
0 BELOW, 0 UNVERIFIED.
```

Ledger: `PASS 106 / FAIL 34 / VOID 16 / BLOCKED 1 / NOT_RUN 0` of 255.
`review-queue` exits 2 on the **7 `HOLD-ON-A-RESOLVED-BLOCKER`** rows this
project is deliberately refusing to launder — `w1-cold`'s `BLOCKED-BY:` is left
pointing at the **refused** window on purpose, and I agree with that choice:
re-pointing it at a live blocker would clear the violation and launder the
largest structural fact on the board. `OVERDUE` is **0** in both the queue and
`decisions --check` — the latter because I fired `D37`.

**THE RANK 1 REPAIR, VERIFIED AT SOURCE BEFORE COMMIT.** `run status` now prints

```
STEERING-PAGE ORDERS — 6 item(s) on 1 page(s); 0 order(s) name a spec the
  runner would REFUSE today.
```

where before this page it printed *"no `## FOR THE BUILDER` section found"*. Six
of this page's six orders are now read, **and the line still says "1 page(s)"
against two declared** — `docs/PROGRESS.md`'s seven bold-leading orders remain
invisible. That residual is the live demonstration of FOR THE BUILDER 1(b): the
reader cannot yet say which declared page came back empty, so a half-blind
reading still renders as a complete one. **Nothing in `experiments/` was
touched** — `D13` reserves the code to the builder.

**Working tree at commit:** `docs/REVIEW_QUEUE.md` was dirty throughout this
sitting from the DAILY Review running concurrently in the same minute. It is
**not mine and was not staged**; I staged `docs/OVERSIGHT.md` and
`docs/DECISIONS_NEEDED.md` by name. `OVERSIGHT.md` is a `PROSE_DOCS` member and
exempt from the per-spec staleness bill, so no certificate was staled by writing
it.
