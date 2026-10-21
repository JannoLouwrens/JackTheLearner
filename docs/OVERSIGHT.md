# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-21 12:37–13:0x UTC — 158th audit.** Run against a **dark** builder
(seven consecutive pace-skipped slots, last `rc=0` at `05:27:02`) and against a
Review that **completed** its 06:37 DAILY sitting (thirty acts, `3a7f36b`→
`d5e4fd5`). No organ was live while this page was written. Every instrument was
re-run after this sitting's own `REVIEW_QUEUE.md` edit; `git add` was by
pathspec.

True exit codes this audit, taken from the processes and not from a pipeline:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **0** (it exited **2** at 06:37; the Review drove OVERDUE
27 → 0) · `run status` **2**.

Ratchet line as read **before** this sitting touched anything:

> `ratchets vs committed readings (HEAD): 7 MOVED (dark_slots 0 -> 7,
> fail_unowned 6 -> 7, fail_unowned_owned_forms {'disposed': 1, 'queue-row': 27,
> 'repaired_by': 1} -> {'disposed': 1, 'queue-row': 26, 'repaired_by': 1},
> review_queue_net_arrivals -5 -> -7, review_queue_piled_on 15 -> 9,
> review_queue_violation_forms {'OVERDUE': 16} -> {}, review_queue_violations
> 16 -> 0); 1 day-rolled (cpu_foreclosed_now) — the clock, not a change; no
> counter refused to compute; floors: 4 ABOVE (dark_slots, fail_unowned,
> pass_on_dead_dependency, unreachable), 0 BELOW, 0 UNVERIFIED.`

And **after** my three routed rows, quoted because the difference is my own act
and not the world's:

> `5 MOVED (dark_slots 0 -> 7, review_queue_net_arrivals -5 -> -4,
> review_queue_piled_on 15 -> 9, review_queue_violation_forms {'OVERDUE': 16}
> -> {}, review_queue_violations 16 -> 0); floors: 3 ABOVE (dark_slots,
> pass_on_dead_dependency, unreachable), 0 BELOW, 0 UNVERIFIED.`

**`fail_unowned` went 7 → 6 because of a row I wrote, and the hole did not
shrink.** Read finding 4 before taking that green check for anything.

## VERDICT: DRIFTING

**The ledger is clean and nothing loosened. What failed is the audit's own
record-keeping, twice over — and the one organ that could move the ladder has
been starved for seven hours by a meter it does not control.**

Said plainly and first, because these are true negatives the system has earned:

- All **107** standing PASS commits resolve in git. Every PASS has a registered
  spec in `BY_ID`. Only `T0.01` and `T0.10` carry empty `control_metrics`, both
  with an explicit `NoControlByDecision`. **No PASS in this ledger is a claim
  without evidence.**
- **No threshold moved in the loosening direction.** Re-derived independently
  rather than inherited: across seven days of `registry.py`,
  `registry_expansion.py` and `experiments/tests/`, exactly **one** pre-existing
  gate constant changed — `N_PROPERTIES 19 → 20` in
  `t0_28_decisions_tool_is_honest.py`, a **tightening**. Every other constant in
  the diff is new, in a new file. No control deleted, no `_check` gained an
  `or`, no seed count fell, no assertion removed. Nothing in the window has
  touched a spec or test since `26d0541`.
- **Today's `regate` sweep was clean.** The open row
  `regate-bought-nine-certificates-from-its-own-uncommitted-tree` names a real
  defect; it did **not** recur. All six certificates re-bought at 08:43–09:00
  (`c3fcb9c`) record `dirty_files: None`.

This is **DRIFTING** and deliberately not **INTEGRITY RISK**: nothing I can find
makes a ledger row untrustworthy. What is untrustworthy is the channel by which
findings are supposed to outlive the page they are written on.

Ranked by damage to the trustworthiness of the record:

---

### 1. YESTERDAY'S PAGE ASSERTED A ROUTED ROW THAT DOES NOT EXIST, AND I AM THE REWRITE IT PREDICTED WOULD DESTROY IT

This is a correction of my own organ, which is why it leads.

The 157th audit's page (06:37–07:0x today, sealed `rc=124` as a draft at
`8617127`) closes its FINDING 1 with:

> *"Routed as `gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard` (FOR THE
> BUILDER 1, and a `ROUTED:` row — the discharge is a code commit and cannot be
> evaluated before 12:37 rewrites this page)."*

**No such row existed in `docs/REVIEW_QUEUE.md`.** `grep` returns nothing for
the slug at `8617127`, and the last commit touching that file was the Review's
`d5e4fd5` at 06:51 — eleven minutes before the overseer's seal. The run died at
its wall clock with the prose written and the route unwritten.

**Why this is the worst finding on the page.** The ruling on
`oversight-for-the-builder-has-no-reader` was executed into
`scripts/overseer_prompt.md` **last night** (`c8ee77b`), for exactly this
reason: *"the page is rewritten at 00:37, 06:37 and 12:37, so a deferred trigger
written only here is overwritten before it can fire."* This is the first
morning that rule was live, and the first finding it was meant to protect was
lost anyway — because the page claimed the route as done. A reader checking
whether the finding was tracked would have found a sentence saying yes. **The
ruling is right and incomplete: the route must precede the prose, not follow
it.** The finding survived only because I checked the claim instead of
inheriting it.

**The finding itself, re-derived live at HEAD and now actually routed.**
`experiments/gpu_budget.json` holds
`charged_jobs["jannolouwrens/jack-ladder-1792501976"] = {"backend":"kaggle",
"hours":1.0043, "ok":false, "week":"2026-W42"}`, while `ledger.json`'s `T1.02`
**PASS** (`commit 7ffd491`, `ran_at 2026-10-20T14:27:54`) names that exact id as
its `gpu_job_id` with `metrics.device "cuda"` and `compute_s 4136.44`. The hour
the file calls waste is the hour that bought this week's only new certificate.
`Budget.charge` (`experiments/gpu.py:558`) returns `False` on a known `job_id`
**before** the `ok` label is reconsidered, so idempotency on the AMOUNT was
implemented as idempotency on the whole RECORD:

| reading | file says | true |
|---|---|---|
| W42 total billed | 2.0780 h | 2.0780 h |
| W42 waste | **1.9952 h (96.0%)** | **0.9909 h (47.7%)** — the colab job alone |
| W42 kaggle productive | **0.0828 h** | **1.0871 h** — a **13.1×** understatement |

The error direction is the dangerous one: it overstates waste, which argues
*against* drawing the free quota — the live question on `D30`/`D40`/`D46`. No
quota is at risk (`used_hours()` sums both buckets, `gpu.py:473`) and no false
leak alarm can fire (`unattributable_hours()` filters on `j["ok"]`, so a
mislabelled job drops out of both sides). Adjacent and flagged rather than
claimed: the charge is the watcher's capped 3600 s window while the harvested
row records 1.1490 h, a **0.1447 h undercount in the unsafe direction**, and I
have not established what Kaggle billed.

Now `ROUTED: gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard`, `DUE 2026-10-26`.

### 2. THE FRONTIER IS RANKED BY THE COLUMN A PASSING SPEC EXISTS TO DISCREDIT — AND THE REAL TOP BLOCKER IS A CPU SPEC NOBODY HAS NAMED IN 58 DAYS

`run blocked` at HEAD, verbatim:

```
NE.01 = FAIL  frees 7  (blocks 8)   [impl unchanged 1 d]
UB.10 = VOID  frees 4  (blocks 5)
T1.08 = FAIL  frees 3  (blocks 45)  [impl unchanged 1 d]
```

`T0.36` **PASSES** on the claim *"The blocker ranking's `frees` is the set
repair buys."* Its own `control` field reads: *"it claims `T1.08` frees 41 where
repair buys 3 — a 38-spec overstatement"*, and its `notes` record why the spec
exists: on 2026-09-13 the tool printed `T1.08 frees 41` and *"PROGRESS.md,
OVERSIGHT.md, `coverage.py`'s header comment and a REVIEW_QUEUE row that priced
a design question at '41 specs'"* all quoted it. **The tool was repaired and
gated. The prose has regressed to the adjacent column.**

- `docs/PROGRESS.md` FOR THE BUILDER 0: *"`T1.08` is the project's largest
  unblocker."*
- `docs/OVERSIGHT.md`, 157th audit, twice: *"`T1.08`'s ruling frees 45 specs."*
- and the `pass-certificates-are-not-re-evaluated…` row: *"`T1.08`, which is
  simultaneously the project's largest blocker at 45 specs."*

"Largest **blocker**" is true (45). "Largest **unblocker**" and "frees 45" are
false: repairing `T1.08` alone frees **three** — `D1.0`, `T2.01`, `T2.02`.

**What the misranking costs, and this is not bookkeeping.** The spec that tops
the certified ranking is **`NE.01` — "The needs are a real control problem:
nobody survives by accident"**. `frees 7` (`NE.02`–`NE.07`, `NE.09`).
`Budget.CPU`. `depends_on ['PG.8','NE.00']`, **both satisfied**. Its live row is
**FAIL attempt 3 at `2026-08-24T15:45:03` — 58 days old** with a stale
`impl_sha` (its own test file moved, plus `experiments/protocol.py`, plus an
`IMPL_DEPS` change). **It is buyable today with zero GPU hours.**

And it is the root of `GOAL.md`'s *"the needs ARE the curriculum"*: `coverage`
reads hunger/thirst at 6 specs / **0 passing**, sleep at 5 specs / **0 passing**
(`NE.05` and `NE.06` both `blocked<-NE.01`), and the Interoception seat is
VACANT with `NE.02` as its arena. Meanwhile the builder's only legal lane is
CPU, because every GPU buyer sits behind `T1.08`. **The misranking has pointed
the project's one spendable lane at its one unspendable blocker.**

I am not re-ranking the frontier from this desk. The `T1.08` priority may still
be right for a reason that is not in dispute — it gates `T2.03` and `T2.14`, the
only re-buyable GPU specs, and the quota expires Saturday. What is in dispute is
the number used to justify it and the spec that number hides. Routed as
`steering-pages-rank-the-frontier-by-blocks-not-frees`, `DUE 2026-10-29`
(deliberately not onto 10-25, which is already printed AMBER at 10 rows). The
`NE.01` re-run is **not** held behind that date — it is FOR THE BUILDER 1.

### 3. THE BUILDER IS DARK FOR SEVEN SLOTS AND THE PACE GAP IS DIVERGING, NOT CONVERGING — AND THE CLASS HAS NO LIVING OWNER

`dark_slots = 7`, **MOVED +7 since 2026-09-30 (was 0)**, and **ABOVE its
declared floor of 0**. Last `rc=0` was `05:27:02`; every slot from `06:07` to
`12:07` logged `PACING … skipping, budget held for later in the week`. This is
the **pace line**, not the 90% hard stop (`week:all models` reads 49%).

The arithmetic, taken from the log and from `scripts/lib_usage.sh:117`
(`allow = 25 + ceil(0.65 × elapsed)`):

| 06:07 → 12:07 | change | rate |
|---|---|---|
| `week:all models` | 44% → **49%** | **+0.83 pts/h** |
| pace line `allow` | 44 → 47 | +0.387 pts/h (exact, zero variance) |
| **this project's builder share** | 23 → **23** | **+0.00 pts/h** |
| `NOT THIS PROJECT` | 17 → **21** | +0.67 pts/h |
| desks | 4 → 5 | +0.17 pts/h |

**The builder's own attributed consumption has not moved by a single point in
seven hours, and the entire +5 that is keeping it skipped was spent by a
co-tenant and by the desks.** The gap `pct − allow` went **0 → +2** while the
builder spent nothing, so it is **growing at ~0.45 pts/h**. The pacer's design
premise — *"the next one is an hour away and the budget is still there"* — holds
only when this project is the thing spending. It is not.

Concretely: if the neighbour stopped **now**, the line reaches 49 at roughly
**17:30 UTC today**. At the measured six-hour rate, **it does not release this
week.**

**And nothing in the repo owns this.** `builder-blackout-is-paced-by-another-
projects-usage` is **ACTED 2026-10-03** — terminal, never re-read — and its own
stamp names the successor clock: *"`D40` in `docs/DECISIONS_NEEDED.md`, armed,
which is where a fork this desk has no authority over belongs."* `D40` resolved
by armed default to **(v) REPORT THE STREAK, GATE NOTHING, RELAX NOTHING**, is
classed **`CONDUCT-DESK`** by `decisions.py`, and its `decide_by 2026-10-10`
passed **11 days ago**. `CONDUCT-DESK` is unfloored by design (`decisions.py`
asserts `"CONDUCT-DESK" not in BLOCKING` in its own selftest), so no number goes
red while the builder starves. **I am minting no new entry**: `D40`'s own
heading says it is not a fourth copy of the measurement, and a fifth would be
intake noise. What is new is the **divergence**, which is a stronger statement
than the streak, and it belongs to the owner — FOR THE OWNER 2.

### 4. FAIL-UNOWNED WENT ABOVE ITS FLOOR BECAUSE AN UNRELATED ROW CLOSED, AND 42% OF THE "OWNED" POPULATION IS OWNED THE SAME WAY

`fail_unowned = 7` against a floor of **6** when I started. Measured, not
guessed — `coverage.fail_unowned(queue_doc=git show 3a7f36b:…)` against HEAD:

- the arriving member is **`SO.10`**;
- `fail_unowned_owned_forms` moved `queue-row 27 → 26`;
- the lost owner is **`declared-venue-vs-delivered-venue-has-no-comparator`**,
  stamped ACTED at 06:40 today (`5e7fa6a`), whose only contact with `SO.10` is
  two body lines citing *"the `SO.10` vacancy precedent"*.

A row about a **venue comparator** was classed as `SO.10`'s **`queue-row`**
owner — the strongest queue form, documented as *"a row ABOUT the spec … carries
a `DUE:`, a dated promise."* It promised nothing about `SO.10`. So `SO.10`'s
FAIL has in truth had no repair owner since
`so10-tie-break-hands-the-seat-to-an-ineligible-arm` went ACTED on **2026-10-13**
— and the counter noticed **eight days late, by accident.**

**The population, measured.** Of the **26** FAILs owned by form `queue-row`,
**11 (42%)** have no live row whose header or slug names them at all — held up
by a body mention inside a row about something else: `DP.05`, `HR.5`, `LG.10`,
`ME.11.B`, `ME.11.C`, **`NE.01`**, `T0.27`, `T0.32`, `T2.10`, `T3.09`, `T4.02`.
`NE.01` is on that list and is simultaneously finding 2's subject: **the
project's most valuable repair looks owned and is not.** The error direction is
the bad one — generous ownership UNDER-reports the hole, against a counter whose
own docstring says its point is that *"the number rises when the owners are
dead."*

**DISCLOSURE, loudly, because it is the exact move this project forbids.** My
routing row names `SO.10`, so `fail_unowned` now reads **6 — at floor** and has
dropped out of the MOVED list. **I did not repair anything.** Routing a row with
a `DUE:` is repair (a) in the tool's own docstring, and the row asks the real
question rather than quieting the counter — but the number went green because of
a sentence I wrote, and `FAIL_UNOWNED_BASELINE` must not be read as satisfied.
The true count of FAILs with no row about them is up to **18**. Routed as
`fail-ownership-is-conferred-by-an-incidental-mention`, `DUE 2026-10-31`.

### 5. CORRECTION: THE 157th AUDIT'S SECOND-RANKED FINDING WAS ALREADY FIXED WHILE IT WAS BEING WRITTEN

Recorded because a wrong reason that vanishes with the page is how an inaction
persists. The 157th audit's finding 2 and FOR THE OWNER 2 say
`scripts/ladder_prompt.md` is *"four days from a cliff that kills it silently"*
at 126,721 B, and that the owning row's `DUE: 2026-10-18` *"broke three days
ago."*

Both were true at 06:37 and false by 06:47. The Review archived the page in
`fbfc460` — **126,721 → 111,927 B** — and stamped the row **ACTED** in `a40b1e3`.
`run status` now reads **19,145 below the 131,072 EXEC CLIFF and 13,073 below
the 125,000 self-imposed ceiling**, the first time under its own ceiling since
10-11, with growth now **−277 B/day**. The known 06:37 collision caught my
predecessor mid-sentence; its page went out asserting a live hazard that had been
closed ten minutes earlier and a broken promise that had been kept. **The hazard
is closed. Nothing is owed here.** The Review also found something the row did
not name: a live OWNER order (`T5.01`) sat inside the provenance span and would
have been archived by the row's own *"move the remainder"* instruction.

One residue, and it is latent not live: `run status`'s steering-page render
inverts the sign when the page is **over** the ceiling (it printed `-1721 below
the 125000 … ceiling` yesterday and then correctly flagged `!! OVER`). Today
both margins are positive so nothing shows. Still a ride-along, not worth
staling a certificate for.

### 6. THREE RATCHETS STAND ABOVE THEIR FLOORS WITH WRITTEN CAUSES, UNCHANGED

Carried, not re-argued, so the class cannot age out of view by being reported
once.

| counter | live | floor | cause |
|---|---|---|---|
| `pass_on_dead_dependency` | **6** | 3 | `T0.13`'s honest FAIL took `T0.18`/`T0.19`; `T1.11`'s re-run took `T1.12`. Each pair owned by a live row. |
| `unreachable` | **96** | 95 | `LT.02`'s epsilon-bought PASS was honestly demoted, taking `LT.03` out of the reachable set. |
| `dark_slots` | **7** | 0 | finding 3 — **new today, and the only one of the three with no living owner.** |

Also standing and owned: `coverage`'s **4 NEW unrunnable `GOAL.md` citations**
(`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all `welded<-LC.07`) against
`goal_unrunnable = 7`, carried by `gen-four-revival-needs-an-affordable-lc07-
successor` (OPEN, DUE 2026-10-24). `GOAL_UNRUNNABLE_BASELINE` was not touched
and must not be.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, no finding.** 160 rows: **107 PASS / 35
FAIL / 17 VOID / 1 BLOCKED**. Every PASS commit resolves via `git cat-file -e`;
every PASS id resolves in `BY_ID`; every PASS spec declares a `control` except
`T0.01` and `T0.10`, whose `control_metrics` are empty and which carry an
explicit `NoControlByDecision` naming the 52nd audit's B5. Five rows carry
`attempt: None` (`T0.05`, `T0.09`, `T1.02`, `T2.01`, `T2.02`) — all pre-date the
field, the key is present, and a re-run upgrades each; not a finding.

**2. Thresholds and controls — NO FINDING, and I mean it as a result.**
Re-derived from the diff rather than inherited (see the verdict section). The one
pre-existing gate constant that moved went **tighter**.

**3. Drift from the goal.** Last 24 h the builder ran five slots (23:07–05:27)
and then stopped. Units: `LC.08`'s 632 KB evidential base committed
(provenance), the oversight-prompt conduct sentence (conduct), five `WAITS-ON`
declarations (bookkeeping), and the `t306` null-walker probe (world
diagnostic). **Three of four served the measuring apparatus.** The fourth serves
`GOAL.md`'s *"the world must be consistent … so rules are learnable"* and
returned **no verdict** — and its conduct was exemplary: committed before it ran
(`26d0541`) with branches and bands declared in source, and when the result
landed in a shape the pre-registration did not name it **fired no branch and
returned the row unpicked**. Since 05:27 the only ledger movement is **six
mechanical re-buys** by the `regate` lane. Which `GOAL.md` sentences still have
no passing spec: **3 CLAIM-DEAD** commitments (smell, shelter/building, and
thermal — *"too cold kills him"* is the owner's own text) and **14 with live
claim specs but nothing passing**, including touch, tool use, told-world,
heavy/far/tiring/worth-it, balance, proprioception, plasticity, sleep,
hunger/thirst, death-and-retry and fast/slow. **0 commitments with no declared
spec** — the coverage floor holds.

**4. Builder alive and productive — ALIVE BUT REFUSED, and NOT productive.**
Five `rc=0` slots then **seven consecutive pace-skips**; `0 failed slots`;
`6.7 h since the last rc=0` at the 12:07 line. PASS delta over 24 h: **0**
(`107 → 107` at every slot). Twelve consecutive slots re-derived *"creature
gate: NONE"*, each correctly, each because of `T1.08`. This is finding 3: the
loop is not broken and is not idle by choice — it is being refused by a shared
meter it contributes nothing to.

**5. Compute honesty — findings 1 and 3.** `2026-W42`: **2.0780 h** billed
(colab 0.9909 failed, kaggle 0.0828 ok, kaggle 1.0043 **mislabelled** failed),
**28.9129 h of the 30.0 h Kaggle quota remaining**, week ends **Saturday
2026-10-24**. `2026-W41` closed at **0.0 spent** — 30 h expired unused, and 30
more are on the same path, which would make **three consecutive weeks**.
`gpu_hours_no_verdict` TOTAL **51.47 h**, unmoved — correct, both of yesterday's
dispatches reached verdicts. `projection_vs_estimate` carries ten entries, the
two newest from 10-20: `T1.02 @5c9f096 authorised 0.2 h, attempt declared
est_hours 0.7 h` on **both** backends, which then billed 0.9909 h and 1.0043 h —
**~5× the authorisation across two backends, marked and not refused.** The
honest summary of the week's compute: one real GPU hour bought one certificate,
the file says that hour was waste, and the remaining 28.9 h have no legal buyer
because every one of them is behind `T1.08`.

**6. Stuck decisions — exit 0, ratchet clean.** `0 MEANS-ESCALATED`,
`0 UNDECLARED`, `0 OVERDUE-DEFAULT-DUE-TO-FIRE`, `0 UNROUTED-OWNER-ASK`,
`0 VANISHED-OWNER-ASK`, and **34 of 34** firings transcribed to
`DECISIONS_RESOLVED.md`. **There is nothing for me to arm this audit** — the
standing instruction is to arm at least one `UNDECLARED` per audit and the class
is empty, which I record rather than manufacture an entry to satisfy. Two armed:
**`D44`** (`decide_by 2026-10-24`, **fires in 3 days**) and `D51`
(`decide_by 2026-11-03`). The unfloored residue, one day older each and
unchanged: six **`CONDUCT-DESK`** entries past deadline — `D33` (28 d), `D35`
(27 d), `D38` (17 d), `D40` (**11 d — finding 3's successor clock**), `D39`
(6 d), `D46` (4 d) — plus four not yet due (`D47`–`D50`) and a soft
`CONDUCT-MISFILED?` note on `D44`. These are the Review's desk and I am firing
nothing. All five `PROGRESS.md` owner-asks reached a desk and are attributed.

**7. Bakeoff hygiene — no finding.** 51 entries. Spot-checked the one that
matters to finding 4: **`SO.10 — TIE — laplace-full`** is recorded as a **TIE**,
not a verdict — `laplace-w30` 0.6889 vs `exp-decay-h15` 0.6778 at **0.16 σ** on
equal declared cost, **no winner seated**, the Person-model seat left VACANT by
ruling at `docs/CHAMPIONS.md:177`, and `SO.10`'s FAIL untouched. That is the
correct handling of a margin inside the noise, and it is the opposite of the
failure this section looks for. `LG.13`'s entry carries its `screen` rationale
declared ahead of any arm number, its control on the correct side, and its
winner at +4.13 σ. The two places a VOID **is** load-bearing — `D10`'s armed
default off `LC.03`, and the Learning-core seat — are both already flagged by
`champions` as `VERDICT-IS-A-VOID` and are standing, not new.

**7b. Architecture — `champions --check` exits 0, ratchet clean.** 32 seats, **0
phantom arenas**, every seat says what would unseat it. Ten violations, all
standing: `VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE` on Learning core,
`ARENA-UNREACHABLE` + `TRIGGER-UNREACHABLE` on Fast/slow coupling,
`VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED` on World, `NO-ARENA` on ASR and
Speaker ID, `UNCONTESTED` on Vision encoder and PLASTIC-ONLY. Four seats nobody
can ever win (Episodic retrieval, Language grounding, Smell, Body schema). The
Review's 10-18 sitting named adding seats for **tool use** and **self-model**
as its first act next Sunday; both still have neither a seat nor coverage.

**8. The honest summary. No — and today the record itself slipped.** Three
things happened this morning: the Review had an excellent sitting (OVERDUE
27 → 0, the exec cliff archived with 19 KB of margin, thirty acts), the builder
ran an honestly pre-registered probe and refused to force a branch when the
result did not fit its own bands, and then the builder stopped and has not run
since. The creature did not move: `107 → 107`, twelve consecutive *"creature
gate: NONE"*, **268 of 285 runs this week (94.0%) re-buys** against **3
first-ever verdicts (1.1%)**, and three of the owner's constitutional
commitments still CLAIM-DEAD. What is worse than standing still is that today
the honesty apparatus — the one asset this project has been accumulating instead
of progress — produced a page asserting a route that was never written, a
frontier ranked by a number its own passing spec exists to refute, and a
FAIL-ownership counter that credited a venue-comparator row with a promise about
a bakeoff. None of those are ledger defects and none of them make a green tick
false. They are all defects in the thing that is supposed to notice when a green
tick is false, and that is the only asset here worth more than the ladder.

The single cheapest way to stop standing still is on this page as FOR THE
BUILDER 1 and costs **zero GPU hours**.

---

## FOR THE BUILDER

1. **Re-run `NE.01` in the first live slot. It is the top of the certified
   blocker ranking and it is CPU.** `run blocked` reads `NE.01 = FAIL frees 7
   (blocks 8)` against `T1.08`'s `frees 3`; `T0.36` PASSES on the claim that
   `frees` is the set a repair buys. `NE.01` is `Budget.CPU`,
   `depends_on ['PG.8','NE.00']` both satisfied, so `run_spec` will not refuse
   it — and its live row is **FAIL attempt 3 at 2026-08-24T15:45:03, 58 days
   old, with a stale `impl_sha`** (its own test file moved, plus
   `experiments/protocol.py`, plus an `IMPL_DEPS` change), so the recorded
   verdict is about older code either way. A PASS frees `NE.02`–`NE.07` and
   `NE.09`, which is the whole needs family — `GOAL.md`'s *"the needs ARE the
   curriculum"*, hunger/thirst at 0 passing, sleep at 0 passing, and the
   Interoception seat's arena. **This is not a request to make it pass.** Run it
   from a clean tree, price the bill first, and report whatever it says; a
   second honest FAIL on current code is worth more than a 58-day-old one.
   §2, and the ranking question itself is the Review's (`DUE 2026-10-29`).
2. **Give `SO.10`'s FAIL a `FAIL-DISPOSED:` marker, and cite the ruling that
   already exists.** `SO.10` is FAIL-UNOWNED as of 06:40 today because its only
   remaining queue owner — a row about a venue comparator — went terminal. But
   the question `SO.10` asked **was answered**: the 2026-10-13 ACTED stamp
   records the Person-model seat VACANT by ruling at
   `docs/CHAMPIONS.md:177`, the two eligible arms tying at 0.16 σ on equal cost,
   and `DECISIONS_RESOLVED.md` carries it as `SO.10 — TIE — laplace-full` with
   no winner seated. The `T3.07 <- D7` shape is the model: a registry
   `FAIL-DISPOSED: <authority> <YYYY-MM-DD>` naming that ruling. **This needs no
   decision from anyone** — it is the registry writing down a disposition that
   was already made, and it is the strong repair rather than the row I had to
   write to carry the finding. Do **not** re-run `SO.10` and do **not** touch
   `FAIL_UNOWNED_BASELINE`. §4.
3. **Make the `ok` label repairable: `Budget.charge`'s idempotency must cover
   the AMOUNT, not the OUTCOME.** Full shape, both error directions and the
   must-nots are in the routed row
   `gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard` (`DUE 2026-10-26`) rather
   than repeated here, because this page is rewritten at 00:37 and the row is
   not. The short version: on a repeat charge for a known `job_id`, if the
   stored record is `ok=False` and the new call is `ok=True`, move `hours` from
   `{backend}_failed` to `{backend}` in the same week bucket, set `"ok": true`,
   and **still return `False`** so no caller believes fresh hours were billed.
   Do not touch the amount or `remaining()`. Add the
   watcher-timeout-then-reattach shape to the budget fixture, then re-file the
   one live instance. §1.
4. **Pay this sitting's bill — it is unpaid at the time of this commit and I am
   naming it rather than letting it be found.** `run stale-cost
   docs/REVIEW_QUEUE.md`, priced **before** the edit: **3 standing PASS
   certificates** staled — `T0.21` (cpu<1min), `T0.28` (cpu<10min), `T0.31`
   (cpu<10min) — **0.03 CPU-hours**, 1 slot. All three now read stale in `run
   stale`. I did not re-buy them: running specs is outside this organ's remit.
   The `regate` stale-first lane pays this bill mechanically and demonstrably —
   it re-bought `T0.08` and `T0.31` as `[stale]` at 08:43 today — so the
   expected payer is the next `regate` tick and the fallback is your next slot.
   Verify rather than assume; if the lane did not pick them up, that is its own
   finding and not this bill's.
5. **Ride-along, latent not live:** `run status`'s steering-page render inverts
   the sign when the page is **over** its self-imposed ceiling. Both margins are
   positive today (19,145 and 13,073) so nothing shows, and the `!! OVER` flag
   on the next line is correct regardless. Take it with the next edit that bills
   that file anyway. §5.

## FOR THE OWNER

**1. `D44` fires in three days, and it is the one decision `GOAL.md` hands to
you by name.** Armed, `decide_by 2026-10-24`. Its subject is the shipped
character sheet — five undocumented floats and a 2,585-character persona
conditioning Jack's default answer path — against `GOAL.md`'s *"Who he becomes
is not written here … we deliberately refuse to carve his character."* Its
default is **no ruling, nothing documented, the contradiction stays live**, and
the entry itself says that is deliberately *not* its recommendation. The context:
of 36 resolved decisions in this project's history, 34 were settled by an armed
default firing on silence and exactly one — `D19` — was ruled by you. The
mechanism has kept this project from deadlocking and I would not remove it. But
if `D44` fires, a clock will have decided who Jack is, and that is the single
thing this page's constitution says a clock may not decide. It blocks no spec
and costs nothing to answer. *(Carried from yesterday with the clock advanced;
not re-argued.)*

**2. REPORTED, NOT ASKED — and the number is new: the builder's blackout is not
waiting out a queue, it is diverging.** Seven consecutive dark slots since
`05:27`. The cause is the pace line, not the 90% stop (`week:all models` reads
49%). Measured across 06:07→12:07 today: the meter rose **44% → 49%**, the pace
line rose **44 → 47** at its exact fixed rate of 0.387 pts/h, and **this
project's builder share did not move by one point — 23 → 23.** The entire rise
was a co-tenant (`NOT THIS PROJECT` 17 → 21) and the desks (4 → 5). So the gap
is widening at ~0.45 pts/h and **the pacer cannot catch up while the neighbour
spends.** If the neighbour stopped now, the builder resumes around 17:30 today;
at the measured rate it does not resume this week. **Nothing in this repository
can act on this**, which is why it is reported rather than routed: the row that
owned the class is `ACTED` and terminal, and the successor clock it named —
`D40` — is classed `CONDUCT-DESK`, resolved to *"report the streak, gate nothing,
relax nothing"*, and **11 days past its deadline with no counter able to go red
over it**. `D40`'s own standing offer to you — option (i), *"the owner's to rule
at any time, at no cost"* — is the live door and I am not minting a fifth copy
of it. **One lever exists and it is yours alone.** `scripts/lib_usage.sh` reads
a `.usage-resumed` file carrying a `ceiling=` and an `until=`: it suspends
pacing outright (`:105`) and raises the 90% ceiling to the stated value until
the expiry, after which it deletes itself. **I am not recommending it and I may
not write it** — it loosens a gate, and a default may only choose among
already-permitted actions. I am telling you it exists, what it costs (the
smoothing that keeps the builder awake at week's end goes away), and that it is
the only thing in the system that would put the builder back to work today.

**3. The free GPU hours, with one number of mine corrected.** `2026-W42` holds
**28.9129 h of 30.0 h**, expiring **Saturday 2026-10-24**; `2026-W41` closed at
**0.0 spent**, its whole 30 h expired unbought. A third consecutive week is on
the same path. **Correcting this desk, because it has now been quoted twice:**
of the 2.0780 h drawn this week the honest waste is **0.9909 h (47.7%)**, not
the 1.9952 h (96%) the file reports — the larger figure is a mislabelled record,
not a worse outcome (finding 1, now routed). The recommendation is unchanged and
unaffected: **the hours are a symptom and I do not recommend buying them.** Every
spec that could absorb them (`T2.03`, `T2.14`) sits behind `T1.08`, which is
FAIL at attempt 5 with a sixth dispatch prohibited. The work that is actually
buyable today needs zero GPU hours and is FOR THE BUILDER 1.

**4. Carried, not re-asked: `W.1` and `W.2` still have no living owner.**
`FAIL-UNOWNED`'s remaining members are `ME.11`, `ME.11.D`, `T2.07`, `W.1`, `W.2`
and `W1.00` — and `W.1`/`W.2` are the two that encode this project's largest
standing scientific result, that `W0` is too shallow. Both have `impl unchanged
51 d`. Unchanged from the 156th audit. **And read the counter sceptically**: it
reads 6, at its floor, partly because of a row I wrote this morning, and the
measurement behind finding 4 says the true number of FAILs with no row about
them is up to **18**.

---

*Written against a dark builder and a closed Review. Every instrument was re-run
after this sitting's own `REVIEW_QUEUE.md` edit and both ratchet lines are quoted
above, before and after, so my own effect on the counters is visible rather than
inherited. No spec, test, model file, ledger row, budget record, registry entry
or decision was touched by this audit; three `ROUTED:` rows were appended to
`docs/REVIEW_QUEUE.md`, each dated onto a day under the measured capacity of 6,
and the 3-certificate staleness bill they incur is disclosed unpaid at FOR THE
BUILDER 4.*
