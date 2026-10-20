# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-20 12:37–13:1x UTC — 155th audit.** Second audit slot of the day, and
the first in a week to run against a *live* builder: the usage gate released at
`07:07` and the loop has taken six consecutive `rc=0` slots since. A `regate`
sweep started at `12:43:02` (mid-audit) and a Colab GPU job for `T1.02` has been
in flight since `12:13:28`, so the tree was dirty throughout and every count
below carries the clock at which I read it.

True exit codes this audit, taken from the processes and not from a pipeline:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **2** · `run status` **2**.

Ratchet line, quoted before anything was recorded:
`floors: 2 ABOVE (pass_on_dead_dependency 6 vs 3, unreachable 96 vs 95), 0
BELOW, 0 UNVERIFIED; 4 MOVED (review_queue_violations 8 -> 16,
review_queue_violation_forms {OVERDUE: 8} -> {OVERDUE: 16},
review_queue_piled_on 17 -> 15, review_queue_net_arrivals -6 -> -5)`.

## VERDICT: DRIFTING

**Nothing is lying in the ledger. Three of the four readers that were supposed
to catch things today cannot see their own subject, and two of the three went
blind this morning in repairs committed this morning.** §1 and §2 are clean and
I want both said plainly before the findings: every one of the 107 standing PASS
commits resolves in git, every PASS that declares a control recorded control
metrics, and **not one threshold moved in the loosening direction** — the day's
one threshold question (`PS.09`'s `KA_SIGNAL_MIN` 0.90) was measured
*unreachable* and escalated to the owner rather than lowered, which is the
correct conduct and is worth more than a green tick.

Ranked by damage to the trustworthiness of the record:

1. **`METER-NON-MONOTONE` — the reading built this morning to catch the gate's
   false meter compares the window by its RENDERED STRING, and the meter renders
   one instant two ways.** The CLI printed `Oct 26, 4:59am (UTC)` and
   `Oct 26, 5am (UTC)` for the same reset instant, 7 times and 5 times, in the
   twelve rows written since the field shipped. **5 of the 11 consecutive pairs
   (45%) therefore read as a WINDOW CHANGE and are exempt from the invariant —
   inside a single real window.** A `95% -> 0%` fall landing on one of those
   boundaries is reported as a legitimate weekly reset. §A.
2. **The overseer's own `FOR THE BUILDER` channel is dark at the reader, for the
   second time in ten days, and this organ severed it itself.** `run status`:
   `docs/OVERSIGHT.md: !! heading PRESENT (1), ZERO items parsed`. Measured
   across revisions: 10-17 **7** items, 10-18 **7**, 10-19 **6**, 10-20 **0**.
   Cause is the numbering style `**B1.` — `steering._ITEM` needs `N.` and
   `_BOLD_LEAD` only normalises `**N.`. The 154th audit's six orders got no
   legality check at all. **Repaired on my side in this file**; the parser half
   is B2. §B.
3. **The number that AUTHORISES a GPU dispatch and the number that AUDITS it are
   different numbers, and nothing in this repo compares them.** In flight right
   now: `T1.02` on Colab, authorised at `--projected-hours 0.2` (the receipt
   `dispatch.sh` demands *so that projected-vs-actual is auditable*), while
   `t1_02_shuffled_control.py:139` hardcodes `submit(..., est_hours=0.7)` — the
   only comparator the per-job overrun mark uses. **It may bill 0.875 h, 4.4× the
   projection it was permitted on, and leave no mark.** §C.
4. **A commit written DURING this audit claims three status moves that did not
   happen, in the ME.11 family.** `734378b` (12:50:04) says *"STATUS MOVED on
   ME.11.B, ME.11.C, ME.11.E"*. Ledger pair `5c9f096 -> HEAD`: all three are
   byte-unmoved — same status, same attempt, same `ran_at`. **They were not even
   re-run.** Second live instance of a row routed 3.5 h earlier, and permanent
   history now asserts movement in the one family where a false move would read
   as the paraphrase-retrieval question finally breaking open. §E.
5. **`rc0-slot-ends-mid-wait-with-watchers-armed-and-work-uncommitted` fired a
   FIFTH time, 23 hours after it was routed**, and this instance held a live GPU
   receipt. `dark_slots` reads **0**, exactly as the row predicted. §D — which
   also records the compounding that did **not** happen, checked rather than
   assumed, and the one that then did.
6. **The frontier did not move, and the sharp number is new: 191 recorded runs in
   seven days produced ZERO first-ever verdicts.** §3, §8.

---

## §A — RANK 1: today's repair is blind in the direction of the scar

### A.1 What shipped, and it is good work

The 154th audit's B1/B2 landed at `2b07a1c` (07:19). B1 adds the meter's own
`resets` string beside `pct` in `scripts/lib_usage.sh:usage_ledger()`, parsed as
a pair from one CLI invocation, and emits rows on the refusal and skip paths.
**Verified live on disk**, not from the commit message — `/data/jack-logs/usage_ledger.jsonl`
carries `resets` on all 12 rows written since 07:16. B2 adds
`usage_attribution.py:meter_non_monotone()`, reporting-only and unfloored in
D26's idiom. Live read `[]`, which is correct: the 18 false readings were never
recorded, because a gate-refused slot used to write nothing — that was B1's whole
point. The pair is well built and the division of labour is right.

### A.2 The defect, measured

`meter_non_monotone` (scripts/usage_attribution.py:208):

```python
if prev is not None and resets == prev[2] and pct < prev[1]:
```

`resets` is the **rendered** string. Census of the only data the reading has:

```
B1-era rows carrying (pct, resets):           12
  'Oct 26, 4:59am (UTC)'                       7
  'Oct 26, 5am (UTC)'                          5
consecutive pairs, same window:                6
consecutive pairs, window reads CHANGED:       5   <- 45% exempt
```

There was one window all day — `reset_epoch` puts both renderings inside 60
seconds of each other, and real windows are seven days apart. The flutter is the
CLI rounding `04:59` to `5am` and back. So the invariant *"pct may not fall while
`resets` holds"* is unevaluable across 45% of adjacent readings, and the
exemption branch is the one the docstring names as benign: *"A fall where
`resets` CHANGED is the legitimate weekly reset and is not an event."*

**Why this is the scar's own direction.** The 154th audit's trace is
`42% (15:22) -> 95% (x15 reads) -> 0% (06:37)`. With B1 landed those refusal rows
now exist, so the next firing IS recordable — and whether B2 sees it depends on
which of two renderings the CLI happened to emit either side of the fall. A
detector that is a coin-flip at the exact transition it was built for is worse
than an absent one, because `meter_non_monotone []` reads as *checked and clean*.

### A.3 The repair is one line and the function already exists

`scripts/claude_usage.py:44 reset_epoch()` already parses both renderings
(`Oct 26, 4:59am` -> 04:59 UTC, `Oct 26, 5am` -> 05:00 UTC). Compare the WINDOW
by that epoch within a tolerance — anything from 5 minutes to an hour separates
rendering flutter from a real reset by four orders of magnitude — never by the
string. Carried as B1 below. It loosens nothing: it can only make MORE pairs
subject to the invariant, and the reading gates nothing either way.

### A.4 What today's rows say about the 18 refused slots, stated as the weak
evidence it is

The 12 B1-era rows climb monotonically `4 -> 5 -> 9 -> 10 -> 12 -> 14 -> 15 ->
18 -> 20 -> 22 -> 25 -> 26` in an unchanged `Oct 26` window. That is consistent,
and it is weak evidence that the FALSE readings were the `95%` plateau rather
than today's zeros — i.e. that **the project was never near its budget when the
gate paused it 18 times.** Weak, because the plateau itself was never recorded
and cannot be re-read. I am not upgrading the 154th audit's finding on it; the
proven statement remains *at least one of the pair is false*, and the policy half
is on `D46`.

---

## §B — RANK 2: this organ broke its own order channel, twice

`run status`, STEERING-PAGE ORDERS block, verbatim:

```
docs/OVERSIGHT.md: !! heading PRESENT (1), ZERO items parsed — the orders exist
                   and the reader cannot see them (form drift; the 2026-09-30
                   outage shape)
```

Counted directly off the revisions of this file:

| revision | date | `^N. ` items | `^**B<n>.` items | steering reads |
|---|---|---|---|---|
| `1ba5094` | 10-17 12:52 | 4 | 0 | 4 |
| `d32952b` | 10-18 06:54 | 7 | 0 | 7 |
| `71eea04` | 10-19 12:57 | 6 | 0 | 6 |
| `ba55263` | **10-20 06:55** | **0** | **6** | **0** |
| `417f772` | 10-10 06:54 | 0 | 6 | 0 |

So it has happened twice, ten days apart, and both times the organ that wrote the
orders is the organ that hid them. `steering.py:115-127` predicted exactly this
in its own comment — *"A third style will still go dark — which is why
`page_stats` now makes a section that parses to zero items LOUD instead of
silent."* The loudness worked; it is how I found it. The style did not.

**What it actually cost.** The builder reads the prose and did execute B1/B2/B3/B5
today, so no work was lost. What was lost is the legality check: `steering`'s
`SUBJECT ... BLOCKED` reading is what flagged `PROGRESS` item 1's
`SUBJECT T2.14 BLOCKED — T1.08 (FAIL)` this morning. **The 154th audit's six
orders received no such check**, and one of them (B5, `CHAMPIONS.md`) and the
carried B4 (`T0.22`) both name specs whose runnability a reader would want
verified. Neither was.

**Two honest repairs, and I am taking mine today.** This page numbers its orders
`1.`/`2.`/`3.` in house style, so `steering` parses them. The durable half is the
parser's and is B2 below — a house style that only one of three organs can
remember is a latent outage, not a convention.

---

## §C — RANK 3: the authorising number and the auditing number are not the same number

### C.1 The live instance

| record | value | source |
|---|---|---|
| projection filed against the week | **0.2 h** | `gpu_budget.json:projections`, `at 2026-10-20T12:13:25`, head `5c9f096`, week `2026-W42`, spec `T1.02` |
| estimate the dispatcher declared | **0.7 h** | `gpu_submissions.jsonl` attempt row `1792498408900-796519-colab`, `est_hours: 0.7` |
| source of the 0.7 | hardcoded | `experiments/tests/t1_02_shuffled_control.py:139` — `submit(job, prefer="colab", est_hours=0.7, …)` |
| overrun mark fires above | **0.875 h** | `gpu.py:581` — `billed_h > est_hours * (1.0 + PER_JOB_OVERRUN_MARGIN)`, margin 0.25 |

The job is still running at the time of writing (33 min elapsed, `timeout_s 3540`).

### C.2 Why this is a hole and not a bookkeeping quibble

`scripts/dispatch.sh`'s REFUSAL 3 states the purpose of the projection in its own
words: *"it is checked against the week's remaining quota and recorded beside the
job so projected-vs-actual is auditable."* The projection (0.2) is what
`dispatch_guard` checked and what `record_projection` filed. **The only automated
projected-vs-actual check in the repo reads the other number.** A job authorised
at 0.2 h can bill 4.4× that and leave no `overruns` entry, and nothing anywhere
compares `projections[].hours` to the attempt row's `est_hours`.

It diverges in both directions, so this is a missing reconciliation rather than a
one-sided loosening: the `2026-10-08` `T1.08` dispatch was projected **0.75 h**
and its recorded overrun names `est_hours 0.5` — there, the audit was stricter
than the authorisation. Either way the two numbers are set in two different files
by two different authors and no reader holds them to each other.

**This is not an accusation of gaming.** The builder's own 12:07 slot summary
states its reasoning in the open — *"honest projections (0.2 h / 0.1 h vs T1.09's
prior actual 0.05 h)"* — and 0.2 h against T1.09's measured 0.05 h is a
conservative estimate, not a convenient one. The defect is structural: the honest
number went in one file and the binding number was already in another.

---

## §D — RANK 4: the fifth firing, and the compounding that did not happen

**The firing.** The 12:07 slot ended `iteration end rc=0` at **12:15:55**, 8m47s
in. At that instant `/data/jack-logs/declared_pids` held:

```
796480:1542640441  2026-10-20T12:13:21+00:00  dispatch T1.02
796519:1542640770  2026-10-20T12:13:25+0000   run_spec T1.02
```

…and the tree held `experiments/gpu_budget.json` + `experiments/gpu_submissions.jsonl`
uncommitted. The slot's own words: *"my background waiter will wake me then …
Waiting on the watcher now; no tree edits until the recordings land."* The
session then ended. That is the routed row's shape exactly — rc=0, watcher
armed, tracked work uncommitted — firing **5** in 72 hours, 23 h after
`6952912` routed it OPEN with `DUE 2026-10-26`. `usage_attribution` reads
`0 dark slots`, as the row says it must.

**What is newly at stake, quantified.** The uncommitted pair is the *only* record
of a GPU spend currently in progress. `gpu_unattributed_jobs` is **AT its
declared floor of 21**; if that attempt row is lost, the Colab job becomes the
22nd and a floored ratchet breaks on a lost file rather than on a decision. This
instance is therefore more expensive than the four the row was priced on, which
cost dirty certificates and a slot's work.

**And the dirty-certificate compounding did NOT happen — checked, not assumed.**
The row documents that the 10-19 08:52 regate sweep ran over an uncommitted tree
and bought five `+dirty` certificates. A sweep started again at **12:43:02** over
this dirty tree and bought `T0.13` (12:43:06) and `T2.10` (12:43:14). Both rows
record **`dirty_files: None`**. `e735f36`'s guard keys on
`protocol.gate_precondition`'s CODE-dirt form, and
`gpu_budget.json`/`gpu_submissions.jsonl` are data, so the guard correctly let
the sweep through and the stamp correctly read clean. The guard and the stamp did
not drift. That is the repair working, and I checked it rather than predicting it.

**The compounding that DID happen is a different one, and it closes §D.** That
same sweep committed at **12:50:04** as `734378b`, and its file list is
`cpu_budget.json`, `ledger.json`, **`gpu_budget.json`, `gpu_submissions.jsonl`**.
So the 12:07 slot's GPU receipts are now safely in HEAD — under a commit whose
subject reads *"regate sweep: cheap certificates re-bought mechanically"*. The
floor-breaking risk above is discharged; the provenance is not. This is
`bare-git-commit-writes-the-whole-shared-index` (OVERDUE) firing for the second
time in 24 hours — the 154th audit's B6 was the first, when the Review's `522db77`
swept 137 lines of overseer-authored decision-register prose. Both times nothing
was lost and both times the authorship of the swept content became unreadable
from `git log`. A sweeping lane and an uncommitting lane compose into a provenance
hole neither one has.

---

## §E — A commit claims three status moves that did not happen

`734378b` (12:50:04, written while this audit was running) ends its message:

```
re-bought 8 of 80 cheap certificate(s) (8 stale + 72 rotation);
STATUS MOVED on ME.11.B, ME.11.C, ME.11.E
```

Ground truth, from the ledger pair `5c9f096` -> HEAD:

| spec | status | attempt | `ran_at` |
|---|---|---|---|
| `ME.11.B` | FAIL -> **FAIL** | 4 -> **4** | 2026-09-28T11:14:26, **unchanged** |
| `ME.11.C` | FAIL -> **FAIL** | 3 -> **3** | 2026-09-28T11:15:13, **unchanged** |
| `ME.11.E` | VOID -> **VOID** | 1 -> **1** | 2026-08-31T20:17:03, **unchanged** |

**All three claims are false, and not merely about status: those rows were not
re-run at all** — no attempt increment, no new timestamp, three-week-old
certificates untouched. The sweep's own per-spec lines say what actually
happened: each of the three printed `(no verdict line)`.

**The cause is already routed, by the builder, this morning.** `7fc68f2` (09:26)
routed `regate-status-move-detector-reads-stdout-substrings` (OPEN, DUE
2026-10-26) on exactly this: the lane infers a move from stdout substrings, so a
child that produced no verdict line reads as a move. That routing was priced on
**one** false claim (`3db1ad7`, on `T0.22`). This instance is **three in a single
commit message, 3.5 hours later**, and it is worse in kind than the first: a
false `STATUS MOVED` on a Tier-0 instrument self-test is noise, whereas `ME.11.B`
/`.C`/`.E` are three arms of the episodic-retrieval bakeoff — the family whose
`beat-both-parents` gate `coverage` reports as *arithmetically unreachable*. A
`git log` reader searching for when that family last moved will find a commit
message asserting it moved today.

**Why this ranks where it does.** Nothing in the ledger is wrong — the rows are
correct and `run status` reports them correctly. What is wrong is the
**permanent** record *about* the ledger: a commit message is not re-derivable and
not amendable, and this project's own instruments (`steering`'s metric reader,
the Review's act stamps, `commit-messages-cite-queue-rows-that-do-not-exist`)
exist because commit prose is read as evidence here. Attach this instance to the
existing row with its ledger pair; do not re-route it.

---

## §1 — Integrity of the ledger: CLEAN, with two standing floors above their line

Re-derived this audit against `experiments/ledger.json` (160 rows, 107 PASS):

- **107 of 107** PASS rows name a commit, and **every one resolves** under
  `git cat-file -e <sha>^{commit}`. Zero dangling.
- **0** PASS rows declare a live control with empty `control_metrics`. The only
  two standing PASSes without a control declare it explicitly as
  `NoControlByDecision` (`T0.10`'s carries the 52nd audit's B5 reasoning).
- Every PASS resolves in `BY_ID`; no orphan certificates.

Standing, not new, and both printed by `run status` as **ABOVE floor**:

- `pass_on_dead_dependency = 6` vs floor **3** (unchanged since 10-17). Pairs:
  `LF.02<-T6.03 BLOCKED`, `T0.18<-T0.13 FAIL`, `T0.19<-T0.13 FAIL`,
  `T1.12<-T1.11 FAIL`, `T2.03<-T1.08 FAIL`, `T2.14<-T1.08 FAIL`. Owned by
  `t013-latently-red-28-disarmed-keys` (DUE 10-23) and
  `t112-pass-stands-on-t111s-orthogonal-added-conjunct` (OVERDUE, 10-18). The
  floor was correctly NOT raised and the counter carries its own derivation note.
  `regate` skips all three `T0.18`/`T0.19`/`LF.02` rotations for the right
  reason — *"rotation would record BLOCKED, not a verdict"*.
- `unreachable = 96` vs floor **95** (unchanged since 09-29), caused by `LT.02`'s
  honest demotion taking `LT.03` out of the reachable set. Clears only when
  `LT.02` is honestly re-bought.

Neither grew today. Both are the ratchet doing its job: the number is red, the
floor did not follow it, and the reason is written down.

## §2 — Thresholds and controls: CLEAN, second consecutive day

`git diff ba55263..HEAD -- experiments/registry*.py experiments/tests/` and the
full 7-day log. **Not one numeric threshold moved in the loosening direction; no
control was deleted or weakened; no `_check` gained an `or`; no seed count fell;
no assertion was removed.** What the day's commits did:

| commit | change | direction |
|---|---|---|
| `30dbeb9` | `T0.35` gains P10–P12 with known-positive AND declared-negative fixtures per kind; the exemption itself gated | **STRENGTHEN** |
| `30dbeb9` | `IMPL_DEPS` widened on 11 specs (`LC.02`+`experiments/cores.py`, `T0.33`+`experiments/run.py`, …); `T2.10` de-grandfathered to declare all five edges | **STRENGTHEN** (more ways to stale a certificate) |
| `28be875` / `1e36243` | `PS.09` re-based reference probe, pre-registered with a forecast BEFORE any number was read | **STRENGTHEN** |
| `b6069dc` | `T2.05` ports `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok`; reports `mse_nonlinear` | **STRENGTHEN** (adds a VOID gate) |

**The conduct item worth naming.** `PS.09`'s run measured `KA_SIGNAL_MIN 0.90` to
be at or above the venue's monotone ceiling on 2 of 3 seeds — i.e. the floor is
unclearable by any honest single-channel reference. The arm available to a
builder that wants a green tick is to lower 0.90. It did not move: `KA_SIGNAL_MIN`
0.90, `KA_GAP_MAX` 0.10 and `ACC_MIN` 0.65 are byte-unmoved and the floor was
escalated to the owner as `D51` with an armed default that *keeps* it. The slot
further discloses that its own first-draft default was **refused by
`decisions --check` as SAFETY-CLAIM-DEAD** and that it took the correction. That
is the system working as designed and it is the best thing in this audit.

## §3 — Drift from the goal

Builder units in the last 24 h, each against a `GOAL.md` sentence:

| unit | commits | GOAL.md sentence served |
|---|---|---|
| 154th audit B1/B2/B5, B3 | `2b07a1c`, `16c1515` | *"protects the honesty of watching what happens"* (l.8) — apparatus |
| `review-defers-leave-no-labelled-hole` | `d148c79` | apparatus |
| **`PS.09` re-based reference** | `28be875`, `1e36243` | **"worth-it"** (l.187) — the only unit that touched Jack, and it returned a NEGATIVE |
| `T0.35` domain widened | `30dbeb9` | apparatus |
| `T1.02`/`T1.09` GPU re-buys | in flight | certificate maintenance on a Tier-1 primitive |
| regate sweeps ×8 | 8 commits | mechanical re-buys |

**Nothing is drift** — every unit traces to a sentence, and the apparatus
sentence is a real one in `GOAL.md`. The honest reading is the ratio: **one of
six units was about Jack, and it reported that a registered floor cannot be
met.** The converse question is harsher. `coverage` reports **0 commitments with
no declared spec** (correct, and the floor holds) but **14 commitments with live
claim specs and nothing passing**, and **3 CLAIM-DEAD** — smell, shelter/building
and thermal(kills) have no passing claim and every claim spec parked or
foreclosed. Curiosity: 12 specs, 2 pass, neither the Ladder Test (`LT.03` VOID).
One brain / unison: 28 specs, **1** pass. All-senses fusion and learning-by-living
remain the thinnest parts of the board, as they have been for weeks.

New this reading and worth the builder's eye: **4 NEW unrunnable `GOAL.md`
citations** — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, each `welded<-LC.07`. The
page's present tense is false about four of its own ids; owned by
`gen-four-revival-needs-an-affordable-lc07-successor` (OPEN, DUE 10-24).

## §4 — Is the builder alive and productive?

Live again, after a blackout. In the 24 h to 12:37:

- **12 hourly slots refused** on `STOPPED at 95% weekly usage` (19:07 10-19
  through 06:07 10-20) — the reading the 154th audit falsified.
- **6 slots ran, 6 ended `rc=0`** (07:07, 08:07, 09:07, 10:07, 11:07, 12:07).
  Zero failures, zero `rc=124`/`rc=126`.
- **PASS delta: 107 -> 107. Zero**, at every one of the six slot boundaries.
- `dark_slots` 0, `failed slots` 0, `0.5 h since the last rc=0`.

Two live flags, neither hidden by the loop:

- **`LC.02` peaked 2039.6 MB against the 1536 MB ceiling** at 11:37 (`ru_maxrss`,
  self-reported). The loop named it, did not kill it, and said *"a declaration is
  not a waiver"*. On a box with paying tenants (`CLAUDE.md`) that is the right
  report; whether the ceiling is right is `D` business on the owner's desk.
- **`/data` at 87%, 14 GB free** — already under the 15 GB corpus-fetch floor, so
  any fetch is refused by standing rule. Noted by the 10:07 slot for its
  successor. I confirm it rather than re-discover it.

## §5 — Compute honesty

- **`2026-W42`: 0.0 of 30.0 free Kaggle hours drawn**, four days to expiry
  (Saturday 2026-10-24). **`2026-W41` closed at 0.0** — 30 h expired unbought.
  For scale, W38/W39/W40 drew 0.92/1.07/1.97 h, so even active weeks buy ~5%.
- **The week's first GPU draw is in flight and it is a RE-BUY, not a claim.**
  `T1.02` (Colab) and `T1.09` (queued behind the lock) are stale standing PASSes
  whose certificates the `30dbeb9` `IMPL_DEPS` widening staled — a ruling-forced
  bill, correctly named as such by the 11:07 slot rather than dressed up as
  progress. `PROGRESS` FTB item 0's actual order — spend released slots on
  `T1.08`'s repair, the project's largest unblocker — is **not** discharged by
  this, and `t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign`
  is OPEN with `DUE 2026-10-21` (tomorrow).
- **`gpu_hours_no_verdict` TOTAL 51.47 h.** The dominant entry is unchanged and
  still the worst single line in this section: **`D1.0` — 33.78 h over 2 attempts
  for 0 verdicts.** `UNATTRIBUTED 6.32 h / 21 jobs`, AT its declared floor of 21
  (see §D for how that floor is currently one lost file from breaking).
- CPU: **6955 s of 57600 s** spent today, but `cpu<2h`'s slack is 3600 s, so that
  class is foreclosed until midnight — **37 of 60 cpu specs unaffordable**, 36 of
  which have never run. `coverage` reports 7 dispatchable specs today, **all 7
  VOID** (an arm to repair, not a dispatch), and three cost classes
  (`cpu<1min`, `gpu<20min`, `gpu<8h`) are NEWLY EMPTY with no path in.

## §6 — Stuck decisions

`decisions --check` exits **0** and the ratchet is clean: `0/0 undeclared`,
`0/0 unrouted-owner-ask`, `0/0 vanished-owner-ask`, `0/0 default-action-expired`,
`0/0 firing-diff`, and **34 of 34 firings transcribed** to
`DECISIONS_RESOLVED.md`.

- **No `MEANS-ESCALATED`.** Nothing a measurement could settle is sitting on the
  owner's desk. The `D1` disease is absent.
- **No `UNDECLARED`, so there is nothing for me to arm this audit.** I am saying
  that rather than manufacturing an arming: the standing instruction is to arm at
  least one per audit, and the register currently has none to arm. Two entries are
  armed and running — `D44` (due 10-24) and `D51` (due 11-03, routed this morning
  off `PS.09`).
- **Six `CONDUCT-DESK` entries are stale, unchanged from the 154th audit**:
  `D33` (27 d), `D35` (26 d), `D38` (16 d), `D40` (10 d), `D39` (5 d), `D46` (3 d).
  The tool's words are *"desk-executable, not the owner's — execute it, report it,
  do not ask."* These are the Review's and not mine; I report the staleness
  without re-routing it for a third time.
- All 5 `PROGRESS.md` owner-asks reach a desk and are attributed (`D46`, `D47`,
  `D48`, `D30`). I read both `FOR THE BUILDER` and `FOR THE OWNER` on that page
  this audit; its items 1–6 are live and §B/§5 above answer the two I can.
- **Nothing acted on without a record.** `DECISIONS_RESOLVED.md` has not moved
  since `fadf712` (10-19 13:22, `D41`/`D42` transcribed).

## §7 — Bakeoff hygiene

`champions --check` exits 0, every seat declares what would unseat it, and the
ratchet is at floor on all seven classes. The standing findings are unchanged and
I am not re-routing them:

- **`VERDICT-IS-A-VOID` — Learning core is held BY VERDICT off `LC.03`, which is
  a VOID.** This is the one §7 question the prompt names, and the answer is still
  yes: the file's strongest marking rests on a non-verdict. It is honestly
  disclosed (`D10 RESOLVED BY ARMED DEFAULT 2026-09-01`, single-arm caveat on its
  face) and `LC.03` is VOID-FORECLOSED, so the repair is a redesign, not a re-run.
- **`World` is held BY VERDICT with no deciding run named and no `TRIGGER:`
  declared.** An unwritten re-open promise cannot decay visibly.
- `TRIGGER DEBT` **3/3** (Learning core, Fast/slow coupling, World);
  `UNCONTESTED` on Vision encoder and PLASTIC-ONLY; `NO-ARENA` on ASR and
  Speaker ID; **4 seats no one can ever win** (Episodic retrieval, Language
  grounding, Smell, Body schema). `NO-LIVE-PATH` totals **6 distinct
  commitments/seats** with no live path, and the repair for every member is a
  REGISTRATION.
- One new reading: `LF.02` is a **KINDLESS DISCHARGE** — credited as the Death &
  persistence challenger while declaring no `COVERS:` kind, so the contest cannot
  be verified from the registry. 1/1, at floor, and a one-line declaration clears
  it.

## §8 — Are we closer to a curious humanoid that climbs the ladder?

**No. We are closer to a longer list of green ticks, and this is the first audit
able to put a single number on it.**

`run status`, SETTLE EVENTS, last 7 days: **191 runs recorded = 0 first-ever
verdicts, 183 re-buys, 8 status changes.** Seven days, 191 runs, and **not one
spec rendered a verdict for the first time.** 75 of the 191 are
instrument-coupled — this project's own tool edits staled the certificate the run
then re-bought. `demonstrated` has read 107 at every slot boundary for 24 hours.

What genuinely improved today is real and I will not undersell it: a false meter
reading can no longer black out the box on one bad read; the gate's refusals now
leave a record a program can check; an implementation dependency now means *a
file whose contents can change the verdict*, which immediately found a fourth
latent instance no row had named; and `PS.09`'s floor was proven unreachable and
escalated instead of lowered. Every one of those is an improvement to the
apparatus for watching Jack, bought with slots that could not be spent on Jack —
and three of today's four top findings are apparatus that cannot see itself.

The creature gate has read **NONE** for five consecutive slots and the
re-derivation is the same every time: `T2.01` sits behind `T1.08` (FAIL at
attempt 5, sixth prohibited), `T6.01` is blocked transitively through
`T4.05 <- T4.04 <- T2.01`, `XL.01`'s row is the desk's. The single ruling that
would unlock the most board is the `T1.08` estimator decision, OPEN and DUE
**tomorrow**. Until it lands, the loop is structurally unable to advance the
ladder however awake it is, and the honest description of the last week is a
measuring apparatus being refined around a creature that has not moved.

## FOR THE BUILDER

1. **Compare the usage window by `reset_epoch`, not by its rendered string.**
   `scripts/usage_attribution.py:208` tests `resets == prev[2]`; the CLI renders
   one instant as both `Oct 26, 4:59am (UTC)` and `Oct 26, 5am (UTC)`, and **5 of
   11 consecutive pairs in the live ledger (45%) therefore read as a window
   change and are exempt from the invariant**. Use
   `scripts/claude_usage.py:reset_epoch()` — it already parses both — and treat
   two windows as the same when their epochs are within a few minutes (real
   windows are 7 days apart, flutter is 60 s). This can only ADD pairs to the
   invariant, so it loosens nothing, and the reading stays reporting-only and
   unfloored. Add the two live renderings as a fixture so the regression is
   pinned. **Do not** change `PACE_FLOOR`, `PACE_CAP` or the 90% stop. §A.
2. **Teach `steering._ITEM` the `B<n>.` form, or normalise it like `**N.`.**
   `docs/OVERSIGHT.md` parsed **7 items on 10-18, 6 on 10-19, 0 today**, because
   `**B1.` is neither `N.` nor `**N.`. The 10-10 revision went dark the same way.
   The parser comment at `steering.py:118-127` already names this as the predicted
   third style. Extend `_BOLD_LEAD`/`_ITEM` to accept an optional single-letter
   prefix (`B1.`, `**B1.`) and add both live shapes to the module's own fixture
   pages, so the next organ that drifts is caught by a test rather than by an
   audit noticing a zero. §B.
3. **Reconcile the two GPU numbers.** `gpu_budget.json:projections[].hours` is
   what `dispatch_guard` authorised (`T1.02`: **0.2 h**); the attempt row's
   `est_hours` is what the per-job overrun mark compares against (`T1.02`:
   **0.7 h**, hardcoded at `t1_02_shuffled_control.py:139`). Nothing compares
   them, so a dispatch permitted at 0.2 h can bill 0.875 h with no mark. Add a
   **reporting-only** reading — `PROJECTION-VS-ESTIMATE`, one line per attempt row
   whose `est_hours` differs from the projection filed for the same
   `(spec, head, week)` — in the D26 idiom: it must not refuse a dispatch and must
   not move `PER_JOB_OVERRUN_MARGIN`. Deciding WHICH number should bind is a
   threshold question and is not yours; print the pair first. §C.
4. **Make `regate`'s `STATUS MOVED` claim read the ledger pair, not stdout — the
   row is yours and already routed, so this is the instance to execute it on.**
   `734378b` (12:50:04 today) asserts *"STATUS MOVED on ME.11.B, ME.11.C,
   ME.11.E"*; all three are unchanged in status, attempt AND `ran_at`, and all
   three printed `(no verdict line)`. That is **three** false claims in one
   message, against the **one** (`T0.22`, `3db1ad7`) that
   `regate-status-move-detector-reads-stdout-substrings` (OPEN, DUE 10-26) was
   priced on 3.5 h earlier, and the lane runs ~12 times a day. The repair is the
   one the row already names — compare the `(status, attempt, ran_at)` triple
   before and after per spec and report the diff — and a child that produced no
   verdict line must read as **no move**, never as a move. Add the three live ids
   as the fixture. No threshold moves; this is a reporting repair. §E.
5. **Commit the GPU receipts before the slot ends, or exit non-zero.** The 12:07
   slot ended `rc=0` with `dispatch T1.02` armed in `declared_pids` and
   `gpu_budget.json` + `gpu_submissions.jsonl` uncommitted — the only record of a
   live GPU spend. This is firing 5 of
   `rc0-slot-ends-mid-wait-with-watchers-armed-and-work-uncommitted` (OPEN, DUE
   10-26) and the row's design covers it; attach this instance to it with its
   timestamps rather than re-routing. The marginal fact to add: `gpu_unattributed_jobs`
   is AT its floor of 21, so a lost attempt row breaks a floored ratchet. §D.
6. **Declare `LF.02`'s `COVERS:` kind.** `champions --check` reports it as the
   only KINDLESS DISCHARGE (1/1, at floor): it is credited as the Death &
   persistence challenger and the contest cannot be verified from the registry.
   One declaration line; no verdict moves. §7.
7. **`T0.22`'s guard comment still says "All nine ran" while `N_PROPERTIES = 15`.**
   Carried unchanged from the 154th audit's B4 and the 10-18 Review's FTB 3. No
   effect on the gate. **Do not** stale a certificate for a comment — ride it along
   with the next edit that bills `T0.22` anyway.

**Not for the builder, said so it is not picked up by mistake:** the `T0.13`
per-key adjudication and `T1.11`'s re-run both sit behind their own FAILs and the
two `pass_on_dead_dependency` rows that own them; `LT.02`'s honest re-buy is what
clears `unreachable 96`, and it is a science question, not a tidy-up; the policy
half of §A is `D46`; and the six stale `CONDUCT-DESK` entries are the Review's
desk, not yours.

## FOR THE OWNER

**1. Your project's own self-check broke this morning, in the repair that was
written to fix the last break — and this is the second such case in ten days.**
Yesterday the gate that decides whether anything may run was proven to be acting
on a number that cannot be true. The fix shipped at 07:19 today and is good work.
But it decides *"is this the same week?"* by comparing a piece of printed text,
and your usage meter prints the same moment two different ways (`4:59am` and
`5am`). On the only day's worth of data it has, **45% of the readings it should
be checking are exempt**, and the gap sits exactly where the original fault
occurred. Repair is one line, using a function already in the same folder (B1).
The pattern worth your attention is not the bug: it is that **two of today's
three worst findings are readers keyed on how something was printed rather than
on what it means** — the other being this report's own order list, which a
numbering style silently hid from the instrument that checks it (B2).

**2. Nothing in the ledger is dishonest, and the one chance to flatter a number
today was declined in the open — but a commit message written during this audit
claims three results that did not happen.** Taking the good first. `PS.09`
measured that one of its own registered
floors (`KA_SIGNAL_MIN 0.90`) cannot be met by any honest reading of the channel
at this venue. The floor did not move; it was escalated to you as `D51` with a
default that *keeps* it. Separately, that slot's first-draft default was refused
by the project's own checker and the slot took the correction and disclosed it.
107 of 107 passing certificates re-resolve in git; every one that declares a
control recorded that control's numbers; no threshold anywhere moved in the
loosening direction for the second day running. **That is the part of this
machine that is working.**

The qualifier, and it matters because this project reads its own `git log` as
evidence: the automated re-buy lane committed at 12:50 claiming *"STATUS MOVED on
ME.11.B, ME.11.C, ME.11.E"*, and all three of those tests are byte-unchanged —
same verdict, same attempt number, same timestamp, three weeks old, never re-run.
The lane guesses at what moved by reading its own printed output instead of
comparing the records, and it guessed wrong three times in one commit. The
ledger itself is right and the board reports the truth; it is the permanent
narration *around* the ledger that is now wrong, in the one test family where a
false "it moved" would look like a breakthrough. Your builder found and routed
this exact defect 3.5 hours before it fired again (B4). **Nobody is fabricating
results — a convenience in a reporting line is writing claims nobody made.**

**3. The thing you are being asked to decide has not changed, and the week's
clock is now four days.** `2026-W42` holds 30.0 free Kaggle GPU-hours with 0.0
drawn, expiring Saturday 2026-10-24; `2026-W41` closed at 0.0 and its 30 h
expired unbought. I do **not** recommend buying them for their own sake, and I
agree with your Review that the hours are a symptom. The decision underneath is
the one that has now been open for two weeks: `T1.08` ("Seed variance measured")
is FAIL at attempt 5, a sixth attempt is prohibited, and it is what blocks
`T2.01`, which blocks `T4.04`/`T4.05`, which block `T6.01` — the first end-to-end
run of Jack. The ruling that would release it is OPEN and promised **tomorrow,
2026-10-21**. Everything else in this report is maintenance on the instruments
around a creature that has not moved in a week.

**4. The number I would put in front of you, if you read only one: 191 runs in
seven days, zero first-ever verdicts.** Not one spec in the ladder rendered a
judgement for the first time. 183 of those runs re-bought a certificate the
project's own tool edits had staled. The board has read 107 of 257 demonstrated
at every slot boundary for 24 hours. The machinery for proving things about Jack
is in better shape than it has ever been; what it is proving things about is
standing still. **That is the drift, and it is not fixable by any of the six
orders above.**

**5. No new entry was opened on your desk by this audit, deliberately.** Your
Review routed six in one morning (`D46`–`D51`) and the 154th audit appended an
evidence addendum to `D46`. Every finding above is either executable at a desk or
already owned by a live queue row with a date. Adding a seventh ask would make
the register harder to read, not your decision easier. If you would rather see
§A or §C as their own decisions, say so and they will be routed.

---

*Instruments re-run immediately before this commit, with a `regate` sweep and a
Colab `T1.02` job live throughout; committed by explicit pathspec, never
`git commit -a` (`bare-git-commit-writes-the-whole-shared-index`, OVERDUE).*
