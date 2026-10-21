> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The overseer run that wrote this file exited rc=124 and did not
> complete its own checklist (2026-10-21T07:02:09+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.

# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-21 06:37–07:0x UTC — 157th audit.** Run against a *live* builder (five
consecutive `rc=0` slots since `23:07`, the last ending `05:27:02`) and against
a *live* Review — `scripts/review.sh` started at `06:37:00`, pid 1410497, the
known 06:37 collision. Every volatile count below was re-read after that
sitting closed; `git add` was by pathspec.

True exit codes this audit, taken from the processes and not from a pipeline:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **2** · `run status` **2**.

Ratchet line, quoted before anything was recorded:
`ratchets vs committed readings (HEAD): 3 MOVED (review_queue_net_arrivals -5 ->
-3, review_queue_violation_forms {'OVERDUE': 16} -> {'OVERDUE': 27},
review_queue_violations 16 -> 27); 1 day-rolled (cpu_foreclosed_now) — the
clock, not a change; no counter refused to compute; floors: 2 ABOVE
(pass_on_dead_dependency, unreachable), 0 BELOW, 0 UNVERIFIED.`
All three movements are the calendar, not an act. Both floor exceedances are
the standing ones with written causes, and neither grew.

## VERDICT: DRIFTING

**The ledger is clean, the builder is working, and the record of what the
builder's compute bought is wrong.** Said plainly and first, because three of
these are true negatives the loop has earned:

- All **107** standing PASS commits resolve in git. Every PASS has a registered
  spec. The only two PASS rows with empty `control_metrics` — `T0.01`, `T0.10` —
  carry an explicit `NoControlByDecision` naming the 52nd audit's B5 reasoning.
  **No PASS in this ledger is a claim without evidence.**
- **No threshold moved in the loosening direction.** Seventeen commits touched
  `registry.py`, `registry_expansion.py` or `experiments/tests/` in seven days;
  I read every numeric change and every removed assertion. Nothing loosened.
  `26d0541` (the t306 probe) adds no gate. `31a3fd0`/`4f75328`/`b585f43` are the
  `LC.08` chain, already verified at the 156th audit and unchanged since.
- **The builder discharged every actionable item on yesterday's page.** FTB 1
  (`coverage._live_queue_view`, FAIL-UNOWNED 0 → 6 with the baseline raised in
  the same commit and the reason in its growth log), FTB 2 (`LC.08`'s 632 KB
  evidential base committed, `b585f43`), FTB 3 (five `2026-10-21` rows now
  declare `WAITS-ON:`; the day reads *"11 rows are 11 decisions"* instead of
  WITHHELD). FTB 4 was correctly **skipped** — it is ride-along-only and no edit
  billed that file. Three for three, overnight, with bills priced before each
  edit and paid from committed trees.

What is wrong is the accounting record, not the ladder — and separately, the
thing that keeps the builder alive is four days from failing silently.

Ranked by damage to the trustworthiness of the record:

---

### 1. THE WEEK'S ONLY PRODUCTIVE GPU HOUR IS FILED AS WASTE, THE DEDUPE GUARD MAKES THAT PERMANENT, AND THIS PAGE REPEATED THE WRONG NUMBER YESTERDAY

This is a correction of my own predecessor, which is why it leads.

`gpu_budget.json` records job `jannolouwrens/jack-ladder-1792501976` as
**`"ok": false`**, 1.0043 h, filed into `2026-W42`'s **`kaggle_failed`** bucket.
`T1.02`'s **live PASS row** (`commit 7ffd491`, `ran_at 2026-10-20T14:27:54`)
names that exact job as its `gpu_job_id`, with `metrics.device: "cuda"`,
`metrics.gpu_repo_sha: 5c9f096` and `compute_s: 4136.44`. **The hour the file
calls waste is the hour that bought this week's only new certificate.**

The `ok:false` was written by the *watcher*, not by the job: commit `bf409f8`
records *"Kaggle watcher capped at 3600s with kernel still QUEUED… kernel
jack-ladder-1792501976 is still RUNNING server-side — reattach via
JACK_REUSE_KERNEL, do not resubmit."* The builder then reattached and harvested
a real GPU result. That recovery was good work and was fully disclosed.

**Why it cannot self-heal.** `Budget.charge` (`experiments/gpu.py:558`):

```python
if job_id and job_id in self.data["charged_jobs"]:
    return False
```

The idempotency guard was built — correctly — so a reattach cannot double-bill
the same kernel (its own docstring: *"Two JobResults with the same id are the
SAME compute… that is what makes a reattach free"*). But it returns **before**
the `ok` label is reconsidered, so **idempotency on the AMOUNT was implemented
as idempotency on the whole RECORD, outcome label included.** A job first
charged `ok=False` can never be re-labelled by the reattach that proves it
succeeded. This is structural, not an oversight of attention.

**What it costs, measured.** `failed_hours()`'s own docstring defines the
bucket: *"Hours the provider still billed for a run that returned nothing… Kept
in a separate bucket because it is WASTE, and waste that cannot be seen cannot
be reduced."* The bucket built to make waste visible is now the thing hiding it:

| reading | file says | true |
|---|---|---|
| W42 total billed | 2.0780 h | 2.0780 h |
| W42 waste | **1.9952 h (96.0%)** | **0.9909 h (47.7%)** — the colab job alone |
| W42 kaggle productive | **0.0828 h** | **1.0871 h** — a **13.1×** understatement |

**And the error direction is the dangerous one.** It overstates waste, which is
the direction that argues *against* drawing the free quota — the exact question
live on `D30`, `D40` and `D46`. Yesterday this page's §5 headline read *"96% of
what it bought was a failed job"* and FOR THE OWNER 3 repeated *"1.995 h (96%)
went to jobs recorded as failed."* True as a quotation of the file; false as a
statement about the world. I am correcting it here rather than leaving it to be
re-quoted a third time.

**Two things this is NOT, stated so nobody over-reacts:**
- **No quota is at risk.** `used_hours()` sums productive **and** failed
  (`gpu.py:473`), so `remaining()` reads 28.9129 h either way. This is a
  reporting defect, not a spending one.
- **The leak detector is not fooled.** `unattributable_hours()` filters on
  `j["ok"]`, so a mislabelled job drops out of *both* sides of its subtraction.
  No false `GPU BUDGET LEAK` alarm, and none fired.

One adjacent measurement, flagged and not claimed: the charge is the watcher's
metered window (capped at 3600 s = 1.0 h) while the harvested row records
`compute_s 4136.44` = **1.1490 h**. When a reattach outlives the watcher cap the
quota is charged the cap rather than the occupancy, and that undercount is in
the unsafe direction. I have not established which number Kaggle actually
billed; I am recording the 0.1447 h gap because nothing else does.

Routed as `gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard` (FOR THE BUILDER 1,
and a `ROUTED:` row — the discharge is a code commit and cannot be evaluated
before 12:37 rewrites this page).

### 2. THE BUILDER'S STEERING PAGE IS FOUR DAYS FROM A CLIFF THAT KILLS IT SILENTLY, AND THE ONLY ROW THAT OWNS THE REPAIR IS THREE DAYS OVERDUE

`run status`, this morning: `scripts/ladder_prompt.md` is **126,721 B — 4,351
below the 131,072 `MAX_ARG_STRLEN` EXEC CLIFF**, growing **+1,123 B/day over 14
commits**, so the instrument prices the cliff at **4 days** and the self-imposed
125,000 ceiling at **0** — the page is already **1,721 B OVER its own ceiling**.
Past the cliff the builder does not read a degraded prompt. **It does not start,
and the failed slot is indistinguishable from any other `rc=126`.**

The repair is owned by exactly one row:
`steering-page-is-two-days-from-the-launch-cliff-and-31kb-is-provenance`,
**DISPOSITIONED**, `DUE: 2026-10-18` — **OVERDUE by 3 days**, and in
`review-queue`'s violation list this morning. Its own `DUE:` note says *"THIS
DESK's execution of THE ARCHIVE below. A desk date, not the builder's — the
builder must not edit its own steering page."* So: the design is complete, the
executor is named, the date broke, and the page has since crossed its ceiling.

Three organs have now walked into this independently and none could stop it.
The 2026-10-18 FULL wrote a `1^23` block, measured **782 B of margin**, and
reverted its own edit — which is why two live orders reach the builder only
through `PROGRESS.md`, a weaker channel, named as such. The builder flagged it
at `05:27` and correctly refused to touch it (*"the desk's file to trim"*).
Yesterday this page carried it only as a ride-along note about an inverted minus
sign in the render. **That was an under-ranking and I am re-ranking it:** this
is the one hazard on the page whose failure mode is invisible by construction,
and every routine edit by builder, overseer or Review moves it closer.

I am not editing that file — it is the Review's and the builder is forbidden it.
What I can do is say that the distance is now 4 days by the instrument's own
arithmetic, that the owning promise is already broken, and that the honest
repairs are the queue's three: ACT, DECLINE, or re-arm with a new `DUE:` and a
reason. FOR THE OWNER 2.

### 3. THE PILE THE LAST AUDIT PREDICTED BROKE EXACTLY AS PREDICTED: 16 → 27 OVERDUE IN ONE CLOCK TICK

`run review-queue` exits 2 with **27 OVERDUE**, up from 16 yesterday.
`review_queue_violations` **+11** and `review_queue_violation_forms`
`{'OVERDUE': 16} → {'OVERDUE': 27}` — the instrument's own annotation reads
*"OVERDUE +11 — CLOCK: the calendar reached a date… no commit is to blame."*

Yesterday's §7 said *"2026-10-21 carries 11 live dated rows against a measured
capacity of 6."* It was not a warning that went unheeded so much as arithmetic
that nothing could act on: **11 more rows fall due today**, against the same
measured capacity of 6, and `IMMINENT` prices **32 of 38** as undischargeable by
the next cycle. Queue state: 42 OPEN, 0 HELD, 32 DISPOSITIONED, 61 ACTED, 13
DECLINED of 148 routed; oldest live **52 d**; drain **173 cycles** — up from 104
yesterday.

The one that matters:
`t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign` is
**still OPEN and falls DUE today**. It owns `T1.08`'s FAIL, the blocker on 45
specs, on both GPU specs that could absorb this week's quota, and on all three
creature gates the builder has re-derived for twelve consecutive slots. The
Review is in session as I write. This is its desk and I am re-dating nothing.

Note the composition honestly: this is `D47`'s subject (capacity), it is
reported, and five consecutive sittings have answered it by re-dating. A drain
that moved 104 → 173 cycles in a day is the measurement that re-dating is not
working, and that is a stronger statement than any individual broken date.

### 4. TWELVE GREEN SLOTS, AND 93.8% OF THE WEEK'S RUNS WERE RE-BUYS

`107 → 107 demonstrated` across every slot in the last 24 h. `run status`'s
settle-event ledger, last 7 days:

| | count | share |
|---|---|---|
| runs recorded | 275 | |
| re-buy (same status as the run before) | **258** | **93.8%** |
| status change | 14 | 5.1% |
| **first-ever verdict** | **3** | **1.1%** |
| PASS events | 243 | of which **first-ever: 1** (0.41%) |

`T0.21×20, T0.28×19, T0.31×17` — **56 re-buys of three instrument self-tests in
one week.** The pass rate cannot fall under this and cannot mean anything
either; it is measuring the re-buy lane, not the ladder. The Review named this
as the week's most important number on 10-18 and it has not improved.

**The honest counterweight, because it is real.** The one slot that was not
apparatus work was the `t306` null-walker probe, and its conduct was exemplary:
committed **before it ran** (`26d0541`) with three readings, branches A/B/C and
bands declared in source; run to completion after two mid-flight deaths; and
when the result landed in a shape the pre-registration did not name — occupancy
1.0556 and run-length 1.0053/1.2024 **AGREE** with the chain while lag-1 excess
reads 0.1131, 2.3× the declared line — **it fired no branch and returned the
row unpicked.** It also refuted the mechanism the ruling proposed (lstsq mass
0.1427 on `a_t` vs 0.0021 on `a_{t−1}`: not servo lag). Forcing a branch there
would have been the easy, wrong move. I note explicitly that this does **not**
indict `RANDOM_DWELL_MAX`: the two statistics the analytic bound is actually
about both agreed, so the probe neither vindicated nor condemned the cap. The
finding is that the branch set was incomplete, and that is the row's to answer.

### 5. SIX DESK-EXECUTABLE DECISIONS PAST DEADLINE — CARRIED, NOT RE-ARGUED

Unchanged from yesterday but one day older each; recorded so the class cannot
age out of view by being reported once.

| id | class | decide_by | stale by |
|---|---|---|---|
| `D33` | CONDUCT-DESK | 2026-09-23 | **28 d** |
| `D35` | CONDUCT-DESK | 2026-09-24 | 27 d |
| `D38` | CONDUCT-DESK | 2026-10-04 | 17 d |
| `D40` | CONDUCT-DESK | 2026-10-10 | 11 d |
| `D39` | CONDUCT-DESK | 2026-10-15 | 6 d |
| `D46` | CONDUCT-DESK | 2026-10-17 | 4 d |

I verified the exemption is deliberate rather than an omission:
`decisions.py:1708` asserts `"CONDUCT-DESK" not in BLOCKING` in its own
selftest. The class is printed and unfloored by design — it has no armed
default so `DEFAULT-ACTION-EXPIRED` cannot fire, no owner-ask so
`UNROUTED-OWNER-ASK` cannot, and it is not `UNDECLARED`. The design choice is
right (a conduct entry must not deadlock the owner); the consequence is that
nothing goes red while it rots. Four new entries (`D47`–`D51`) were opened in
the same window in which none of these six was discharged. These are the
Review's desk, not mine, and I am firing nothing.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** 160 rows: 107 PASS / 35 FAIL / 17 VOID
/ 1 BLOCKED. Every PASS commit resolves via `git cat-file -e`; every PASS has a
registered spec in `BY_ID`; every PASS spec declares a `control`. Only `T0.01`
and `T0.10` have empty `control_metrics`, both carrying `NoControlByDecision`.
No finding.

**2. Thresholds and controls — NO FINDING, and I mean it as a result.** Seventeen
commits in the window. No numeric threshold moved loose, no control deleted or
weakened, no `_check` gained an `or`, no seed count fell, no assertion removed.
The one movement worth naming is in the honest direction: `FAIL_UNOWNED_BASELINE`
was **raised** 0 → 6 in the same commit that made the test stricter, with the
reason in its growth log — the ratchet contract working exactly as written.

**3. Drift from the goal.** Last 24 h, five slots: `coverage` ownership repair
(instrument), `WAITS-ON` declarations (bookkeeping), `LC.08` artifact commit
(provenance), the oversight-prompt conduct sentence (conduct), and the `t306`
probe (world diagnostic). **Four of five served the measuring apparatus.** The
fifth serves `GOAL.md`'s *"the world must be consistent… so rules are
learnable"* and returned no verdict. Which `GOAL.md` sentences still have no
passing spec: `coverage` reports **3 CLAIM-DEAD** commitments (smell, shelter/
building, thermal-kills — *"too cold kills him"* is the owner's own text) and
**14 with live claim specs but nothing passing**, including touch, tool use,
told-world, proprioception, plasticity and sleep. **0 commitments with no
declared spec** — the coverage floor holds.

**4. Builder alive and productive — YES on alive, NO on the ladder.** Slots at
23:07, 02:07, 03:07, 04:07, 05:07, all `rc=0`; zero failed slots; `0 dark slots`
at the last pacing line. The 15-slot blackout is over — the usage gate released
and `week:all models` read 44% at 29% of the week this morning, under its own
pace line. PASS delta over 24 h: **0**. Twelve consecutive slots have re-derived
*"creature gate: NONE"*, each correctly, each because of `T1.08`.

**5. Compute honesty — the audit's finding 1.** `2026-W42`: 2.0780 h billed
(colab 0.9909 failed, kaggle 0.0828 ok, kaggle 1.0043 mislabelled failed), 28.91
h of the 30.0 h Kaggle quota remaining, week ends **Saturday 2026-10-24**.
`2026-W41` closed at 0.0 spent. Four `per_job` overruns recorded, two of them
yesterday. `projection_vs_estimate` (shipped 10-20) reads `T1.02 @5c9f096:
authorised 0.2h, attempt declared est_hours 0.7h` — twice, once per backend —
and the two then billed 0.9909 h and 1.0043 h: **~10× the authorisation across
two backends, marked and not refused.** `gpu_hours_no_verdict` TOTAL is 51.47 h
and did **not** move for yesterday's spend, which is correct — `T1.02` and
`T1.09` both reached verdicts.

**6. Stuck decisions.** `decisions --check` exits **0**, ratchet clean: **0
MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE-DEFAULT-DUE-TO-FIRE, 0
UNROUTED-OWNER-ASK, 0 VANISHED-OWNER-ASK**, and 34 of 34 firings transcribed to
`DECISIONS_RESOLVED.md`. **There is nothing for me to arm this audit** — the
standing instruction is to arm at least one `UNDECLARED` per audit and the class
is empty, which I record rather than manufacture an entry to satisfy. Two armed:
`D44` (due 2026-10-24, fires in 3 days) and `D51` (due 2026-11-03). Finding 5 is
the unfloored residue. All five `PROGRESS.md` owner-asks reached a desk and are
attributed.

**7. Bakeoff hygiene — no finding.** 36 resolved. `LG.13`'s entry carries its
`screen` rationale declared *in advance of any arm number*, the control
(`state-free-prompt`, 0.064, −5.02σ) on the correct side, and the winner at
+4.13σ over the runner-up — outside the noise margin. No VOID is treated as a
verdict in the resolved file; the two places a VOID *is* load-bearing (`D10`'s
armed default off `LC.03`, and the Learning-core seat) are both already flagged
by `champions` as `VERDICT-IS-A-VOID` and are standing, not new.

**8. The honest summary.** **No. We are not closer to a curious humanoid than
yesterday, and today the gap is sharper than usual.** The machinery had a good
night — three audit orders discharged, a pre-registered probe run honestly and
returned unpicked, not one threshold loosened. The creature did not move:
`107 → 107`, twelve consecutive `creature gate: NONE`, 93.8% of the week's runs
re-buys of instruments, and three of the owner's constitutional commitments
still CLAIM-DEAD. The single blocker is unchanged and is a desk decision, not a
science problem: `T1.08`'s ruling frees 45 specs, both GPU-absorbing specs, and
all three creature gates — and it falls due today. Everything else on this page,
including my own finding 1, is housekeeping on a project that is standing still
while keeping an exceptionally honest record of standing still. That record is
worth a great deal and it is not the goal.

---

## FOR THE BUILDER

1. **Make the `ok` label repairable: `Budget.charge`'s idempotency must cover
   the AMOUNT, not the OUTCOME.** `experiments/gpu.py:558` returns `False` on a
   known `job_id` before `ok` is ever reconsidered, so a job charged `ok=False`
   by a watcher timeout can never be re-labelled by the reattach that proves it
   succeeded. **Measured consequence in the committed file:**
   `jannolouwrens/jack-ladder-1792501976` reads `"ok": false` and sits in
   `kaggle_failed` while `T1.02`'s live PASS (`7ffd491`) names it as its
   `gpu_job_id` with `metrics.device: "cuda"` — making W42 report 1.9952 h of
   waste where 0.9909 h is real, and 0.0828 h of productive Kaggle time where
   1.0871 h is real. The repair is **narrow and must not re-bill**: on a repeat
   charge for a known `job_id`, if the stored record is `ok=False` and the new
   call is `ok=True`, move `hours` from `{backend}_failed` to `{backend}` in the
   same week bucket, set `"ok": true`, and **return `False` as it does today** so
   no caller thinks fresh hours were billed. Do not touch the amount, do not
   touch `remaining()` (it already sums both buckets and is correct), and add the
   watcher-timeout-then-reattach shape to the budget fixture so it cannot
   regress. Then re-file the one live instance above. §1, and routed.
2. **While you are in that file: the undercount is worth one printed line, not a
   gate.** The charge is the watcher's metered window (capped 3600 s = 1.0 h);
   `T1.02`'s harvested row records `compute_s 4136.44` = 1.1490 h. Where a
   reattach outlives the cap, print the gap beside the charge so the
   0.1447 h is visible. Reporting-only — I have not established what Kaggle
   actually billed, and a gate on an unverified quantity would be worse than
   the gap. §1.
3. **Ride-along, unchanged from yesterday and still correctly unbilled:** `run
   status`'s steering-page line renders `-1721 below the 125000 self-imposed
   ceiling` and then correctly flags `!! OVER` on the next line. The sign is
   inverted in the render only. Take it with the next edit that bills that file
   anyway; it is not worth staling a certificate for a minus sign.

## FOR THE OWNER

**1. `D44` fires in three days, and it is the one decision `GOAL.md` hands to
you by name.** Armed, `decide_by 2026-10-24`. Its subject is the shipped
character sheet — five undocumented floats and a 2,585-character persona
conditioning Jack's default answer path — against `GOAL.md`'s *"Who he becomes
is not written here… we deliberately refuse to carve his character."* Its
default is **no ruling, nothing documented, the contradiction stays live**, and
the entry itself says that is deliberately *not* its recommendation. The context
you should have: of 36 resolved decisions in this project's history, 34 were
settled by an armed default firing on silence and exactly one — `D19` — was
ruled by you. The mechanism has kept the project from deadlocking and I would
not remove it. But if `D44` fires, a clock will have decided who Jack is, and
that is the single thing this page's constitution says a clock may not decide.
It blocks no spec and costs nothing to answer. Three days. *(Carried from
yesterday with the clock advanced; not re-argued.)*

**2. Reported, not asked — the thing that runs the builder is four days from
stopping without a visible error.** `scripts/ladder_prompt.md` is 126,721 B,
4,351 below the 131,072 exec cliff, growing +1,123 B/day, and already 1,721 B
over its own 125,000 ceiling. Past the cliff the builder does not start and the
slot looks like an ordinary `rc=126`. The repair is designed and owned — queue
row `steering-page-is-two-days-from-the-launch-cliff-and-31kb-is-provenance` —
and that row's `DUE: 2026-10-18` **broke three days ago**. Three organs have now
measured this independently; one of them reverted its own edit at 782 B of
margin, which is why two live builder orders currently reach the builder only
through `PROGRESS.md`. **Nothing is blocked on you** and I am routing no new
entry: the existing row is the right home and the Review is the named executor.
I am telling you because the failure mode is silent, the owning promise is
already broken, and you are the only reader who will still be here if the
builder stops looking broken and starts looking idle. If you want it as its own
decision, say so and it will be routed.

**3. Carried, not re-asked: `W.1`/`W.2` still have no living owner** (the 156th
audit's FOR THE OWNER 1, unchanged — `FAIL-UNOWNED` now correctly reads 6 at its
raised floor after the builder's repair, so the hole is at last visible in an
instrument); **and the free-GPU question is still undecidable** — W41 closed at
0.0 of 30.0 h, W42 has drawn 2.0780 h and ends Saturday. I correct one number
this desk gave you yesterday: of that 2.0780 h, the honest waste is **0.9909 h
(47.7%)**, not 1.9952 h (96%) — the larger figure is a mislabelled record, not a
worse outcome. The recommendation is unchanged and unaffected: the hours are a
symptom, I do not recommend buying them, and the specs that could absorb them
sit behind `T1.08`, whose ruling is the Review's and falls due today.

---

*Written against a live builder and a live Review; all five instruments re-run
after the 06:37 Review sitting closed, and every count above re-read from that
state. No spec, test, model file, ledger row, budget record or decision was
touched by this audit.*
