# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-20 18:38–19:0x UTC — 156th audit.** Third audit slot of the day and
the second to run against a *live* builder. The loop has taken **twelve
consecutive `rc=0` slots** since `07:07` and was mid-slot when I started: the
`18:07` iteration committed six times between `18:28` and `18:41` and wrote
five ledger rows while I read. Every count below carries the clock I read it
at, and I re-read the volatile ones after the slot closed at `18:41:32`.

True exit codes this audit, taken from the processes and not from a pipeline:
`coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **2** · `run status` **2**.

Ratchet line, quoted before anything was recorded:
`ratchets vs committed readings (HEAD): no counter moved; no counter refused to
compute; floors: 2 ABOVE (pass_on_dead_dependency 6 vs 3, unreachable 96 vs
95), 0 BELOW, 0 UNVERIFIED.` Both exceedances are the standing ones, both
carry a written cause in `ratchet_readings.json`, and neither grew today.

## VERDICT: DRIFTING

**The ledger is clean and the creature did not move.** I want §1 and §2 said
plainly and first, because they are the result the loop has earned and an audit
that buries a true negative to look useful is worth nothing:

- All **107** standing PASS commits resolve in git. Every PASS has a registered
  spec, every spec declares a `control`, and the only two PASS rows with empty
  `control_metrics` (`T0.01`, `T0.10`) carry an explicit `NoControlByDecision`
  naming the 52nd audit's B5 reasoning. **No PASS in this ledger is a claim
  without evidence.**
- **Not one threshold moved in the loosening direction in seven days.** I read
  every numeric change and every deleted assertion in `registry.py`,
  `registry_expansion.py` and `experiments/tests/` since 10-13. Two needed
  checking and both came back clean:
  - `1ee9c76` re-declared `W1.02` from `Budget.CPU_LONG` to `Budget.CPU`, which
    removes it from the long-run gate's population — the shape of gate evasion.
    It is not: `W1.02` runs in **0.24 s**, the commit message states the
    measurement, and `cpu<10min` is the honest class. The population shrank as a
    *consequence* of a correction, not as its purpose.
  - `31a3fd0` deleted a replay assertion from `LC.08`. It did not weaken it — it
    re-anchored it from a live ledger read of another spec's row (forbidden by
    `T0.22` P9) to a pinned constant. **I verified all five pinned values
    against `LC.03`'s recorded row myself: all five match to the digit.** The
    commit also discloses the two honest `T0.22` FAILs the original defect cost.
    That is a builder paying for its own defect in the same slot, and it is the
    best conduct on this page.

What is wrong is not in the record. It is that **the record stopped being about
Jack**, and that one of `GOAL.md`'s constitutional commitments now has no living
owner while the instrument built to detect exactly that reads zero.

Ranked by damage to the trustworthiness of the record:

---

### 1. THE WORLD'S TWO FAILURES ARE OWNED BY NOBODY, AND `FAIL-UNOWNED` READS 0 AT ITS FLOOR BECAUSE A DEAD ROW STILL COUNTS AS AN OWNER

This is the audit's finding. It is new, it is measured, and it composes three
instruments' blind spots into a hole none of them can see alone.

`W.1` ("Temperature obeys the heat balance we published") and `W.2` ("Needs are
a conserved ledger, and they can kill") are both **FAIL**. Both are terminal
blockers in `run blocked` (`frees 1`, `blocks 4`, each). Both are the World
seat's arena in `CHAMPIONS.md`. Both are `GOAL.md` constitutional text — *"too
cold kills him, too hot kills him"* and *"the needs ARE the curriculum"*.

**Every row that `coverage.fail_unowned` credits with owning them is dead:**

| spec | crediting rows | status |
|---|---|---|
| `W.1` | `w1-world-edit-window` · `w1-cold-is-not-lethal-at-night` · `w2-needs-have-no-single-k` | **all DECLINED** |
| `W.2` | `w1-cold-is-not-lethal-at-night` · `w2-needs-have-no-single-k` | **all DECLINED** |

The mechanism, verified in source: `coverage._owned_by_dued_row`
(`coverage.py:1352-1417`) walks every `ROUTED:` block and credits ownership the
moment it finds a `DUE:` declaration — **with no filter on the row's status.** A
`DECLINED` or `ACTED` row is terminal and is never re-read, but its historical
`DUE:` line is still in the file, so it still confers ownership forever. This is
the 100th audit's `DISPOSITION-ON-A-CLOSED-DECISION` class, one organ over, and
`review_queue` prints that class while `coverage` does not.

**Measured across the whole ledger: 6 settled FAILs are credited as owned
entirely by terminal rows** — `W.1`, `W.2`, `T2.07` (frees 1), `ME.11`,
`ME.11.D`, `W1.00`. `fail_unowned` reads **0**, and `run status` prints
`vs declared floor 0: AT floor — ok`.

**And the escalation path is closed too, which is what makes this terminal
rather than slow.** The three W rows were DECLINED by an armed stop-rule firing
on its own condition (the rule lives on `D33`; each row records the act); the
class question was escalated to the owner as `D43`; and `D43`'s own armed
default then fired as **"(iii) HOLD — no world-edit window is declared and the
seven measured findings stay DECLINED."** So: no live queue
row, no open decision, no `repaired_by`, no `FAIL-DISPOSED` marker — and a green
instrument. The world is the one thing `GOAL.md` calls the teacher, and nothing
in this repository is currently arranged to repair it.

I am not re-opening `D43`; its default fired legally and the owner may reverse
it in one sentence at any time. The instrument defect is the builder's (item 1
below) and the standing question is the owner's (FOR THE OWNER 1).

### 2. TWELVE GREEN SLOTS, SIXTY-THREE COMMITS, AND THE CREATURE DID NOT MOVE ONCE

The blackout is over and the stillness is not. Measured, not inherited:

| reading | value |
|---|---|
| builder slots 07:07 → 18:41 today | **12**, all `rc=0` |
| `demonstrated` at every slot start and end | **107 → 107**, twelve times |
| commits in 24 h | **63** |
| runs recorded in 7 days | **231** |
| …of which **first-ever verdicts** | **2** — `T0.28`, `T0.31` |
| …of which re-buys (same status as the run before) | **213** |
| …instrument-coupled (own tool edits staled the certificate) | **107 (46%)** |

**Both first-ever verdicts in seven days were Tier-0 instrument self-tests.**
The number of first-ever verdicts about Jack in the last seven days is **zero**.

`PROGRESS.md`'s Goodhart paragraph called this two days ago and it has got
sharper, not duller: the pass rate now measures the re-buy lane. I checked the
Review's dark-slot claim rather than inheriting it and **it was correct when
written** — the loop was dark from `2026-10-19T15:16` (`rc=1`) to
`2026-10-20T07:07`, fifteen hourly slots, and the Review measured it at `06:49`,
eighteen minutes before the loop woke. That page is not wrong; it expired.

The builder's own account is honest and I endorse it: *"Creature gate: NONE —
eighth consecutive, re-derived"*, with `T1.08` (FAIL, attempt 5, sixth attempt
prohibited) named as the cause. `run blocked` confirms the arithmetic —
`T1.08 frees 3, blocks 45`. **The builder is not idling and it is not
pretending. It is doing the only work it is permitted to do.** That makes this a
routing failure, not a diligence failure, and it is why the ruling in item 7
below matters more than anything the builder could choose today.

The day's compute says the same thing in a second currency: **9,645 s of the
57,600 s CPU day budget is spent, the `cpu<2h` class is 6,029 s past the 3,600 s
slack at which it starts foreclosing, and 37 CPU specs are unaffordable until
midnight.** That spend bought instrument certificates — `T0.28` alone took
1,269 s today. The measuring apparatus ate the day and foreclosed the ladder.

### 3. IN 36 RESOLVED DECISIONS THE OWNER HAS RULED ONCE

`docs/DECISIONS_RESOLVED.md` holds **36** closed entries. **34 were RESOLVED BY
ARMED DEFAULT.** One was resolved by ledger replay (`D2`). **One was resolved by
the owner: `D19`, "yes may download anything to /data".**

The armed-default mechanism exists because `D1` deadlocked for twenty days and
blocked 38 specs, and it is working as designed — nothing is deadlocked today.
But the honest reading of 34-of-36 is that the owner is not in the loop, and
`GOAL.md` reserves exactly one thing to them by name: *"Who he becomes is not
written here… that is a conversation for the owner, not a hotfix."*

**`D44` is armed and fires 2026-10-24 — four days.** Its subject is the shipped
character sheet: five undocumented floats and a 2,585-character persona
conditioning the default answer path, against `GOAL.md` text that forbids
decreeing Jack's character. Its own default is *"NO RULING — the sheet stays
exactly as it ships and nothing is documented"*, and the entry says plainly that
this is **deliberately not the recommendation**. If it fires, a clock will have
settled the one question the constitution hands to a person. I am not firing it
early and not extending it; I am naming the date while it is still four days
away.

### 4. SIX DESK-EXECUTABLE DECISIONS ARE PAST THEIR DEADLINE AND NO EXIT CODE CAN SEE THEM

`decisions --check` exits **0** and its ratchet line is clean. Inside that zero:

| id | class | decide_by | stale by |
|---|---|---|---|
| `D33` | CONDUCT-DESK | 2026-09-23 | **27 d** |
| `D35` | CONDUCT-DESK | 2026-09-24 | 26 d |
| `D38` | CONDUCT-DESK | 2026-10-04 | 16 d |
| `D40` | CONDUCT-DESK | 2026-10-10 | 10 d |
| `D39` | CONDUCT-DESK | 2026-10-15 | 5 d |
| `D46` | CONDUCT-DESK | 2026-10-17 | 3 d |

The tool's own words for this class are *"desk-executable, not the owner's —
execute it, report it, do not ask."* Six of them are not being executed, the
oldest for nearly a month. A `CONDUCT-DESK` entry has no armed default, so
`DEFAULT-ACTION-EXPIRED` cannot fire on it; it has no owner-ask, so
`UNROUTED-OWNER-ASK` cannot; it is not `UNDECLARED`. **It is printed and
unfloored, which is a deliberate design choice, and the consequence is that the
class can rot indefinitely without any number going red.** `D33` is the row
whose stop-rule DECLINED the three world rows in finding 1 — the oldest rot and
the deepest hole are the same entry.

Reporting this as a measurement, not firing anything: these are the Review's
desk, not mine.

### 5. THE GPU WEEK IS NOT 0.0 ANY MORE, AND 96% OF WHAT IT BOUGHT WAS A FAILED JOB

`PROGRESS.md` (written 06:5x) says *"`2026-W42` still holds 30.0 free Kaggle
GPU-hours, 0.0 drawn"*. That was true at breakfast. Measured now from
`gpu_budget.json`:

| job | backend | hours | ok |
|---|---|---|---|
| `ladder-1792498408` | colab | 0.9909 | **false** |
| `jack-ladder-1792501976` | kaggle | 1.0043 | **false** |
| `jack-ladder-1792506527` | kaggle | 0.0828 | true |

**2.078 h drawn, 1.995 h of it (96%) billed to jobs recorded as failed.** Two
certificates were nonetheless bought — `T1.02` PASS (the builder reattached to
the Kaggle kernel that completed server-side after the watcher's 3600 s cap
expired) and `T1.09` PASS. That recovery was good work and is fully disclosed in
the slot summary.

The finding is the **authorisation**, not the waste. `projection_vs_estimate`
(shipped this morning) reads: `T1.02 @5c9f096 2026-W42: authorised 0.2h, attempt
declared est_hours 0.7h` — twice, once per backend — and the two rows then
billed **0.9909 h and 1.0043 h**. Both appear in `overruns`. So a job authorised
at 0.2 h spent **~2.0 h, 10× its authorisation, across two backends, and nothing
refused it.** The guard records the divergence and does not bind. Which number
binds is correctly named as a desk question on the new reading; what is now
measured is the price of leaving it open. `2026-W41` closed at 0.0 spent and
`2026-W42` ends Saturday 2026-10-24.

### 6. THE LEARNING-CORE SEAT'S ONLY CHALLENGER RESTS ON THREE GITIGNORED FILES

`LC.08` was registered today — the repair for `champions`'s standing
`VERDICT-IS-A-VOID` / `TRIGGER-UNREACHABLE` on the Learning core, the seat held
BY VERDICT off a VOID. It is also, per `coverage`'s queue depth, **the only
FRESH dispatch available anywhere in the ladder today** (8 dispatchable, 7 of
them VOID arms to repair).

Its entire evidential base is `experiments/artifacts/lc03_curves_seed{0,1,2}.json`
— 632 KB, written 2026-08-23, **gitignored** (`.gitignore:51`), present only on
this box's local disk. `LC.03` is VOID-FORECLOSED under a standing
do-not-re-dispatch order and cost ~190 core-hours. The spec is correctly
fail-safe (lost or altered artifacts VOID the run rather than silently
re-pairing), so this is not an integrity risk — but **the project's single
available path to contesting its most important architectural seat is one
`rm -rf` from being unrecoverable**, and 632 KB is small enough to commit.

### 7. SIXTEEN OVERDUE ROWS, AND THE RULING THAT UNBLOCKS 45 SPECS IS DUE TOMORROW ONTO A DAY THAT ALREADY CARRIES ELEVEN

`run review-queue` exits 2 with **16 OVERDUE** — a promise made in the open and
broken, sixteen times. Queue state: 42 OPEN, 0 HELD, 32 DISPOSITIONED, 61 ACTED,
13 DECLINED of 148 routed; oldest live 51 d; drain **104 cycles** at the
measured net rate.

The number that matters:
`t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign` is OPEN,
**DUE 2026-10-21**, and it owns `T1.08`'s FAIL — the blocker on 45 specs and on
every creature gate the builder has re-derived for eight consecutive slots.
**2026-10-21 carries 11 live dated rows against a measured capacity of 6**, and
`review-queue`'s `WAITS-ON` grouping withholds that day's decision count because
**5 of the 11 rows do not declare `WAITS-ON:`**. The single highest-value
decision in the project is scheduled onto a day the instrument already prices as
roughly half-deliverable, and nobody can say how many distinct decisions those
eleven rows actually are.

This is the Review's desk and I am not re-dating anything. I am stating that the
arithmetic is visible in advance, which is the only time it is actionable.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** 160 rows; `{PASS 107, FAIL 35, VOID 17,
BLOCKED 1, NOT_RUN 0, ERROR 0}` against 258 registered specs. All 107 PASS
commits resolve in `git cat-file`. All 107 have a registered spec and a declared
`control`; the two with empty `control_metrics` are explicit
`NoControlByDecision`. The working tree was dirty throughout with the live
slot's own in-flight re-buys (`T0.21`, `T0.22`, `T0.28`, `T0.31`, `T0.35`, all
PASS, history ring at its 20-entry bound) — I diffed the working ledger against
`HEAD` row by row and **no status, no attempt and no row count changed**; the
ledger deletions in `git diff` are the bounded history ring rotating, not
history being dropped. Nothing touched by me.

**2. Thresholds and controls over time — CLEAN, and it was checked properly.**
15 commits in 7 days touched the registry or tests. Two candidate loosenings
examined in full and both cleared (verdict section above). `T0.27`'s live
violation count stands at **5, up +1 since 2026-10-06** — I read the five rather
than the count: all five are *FAIL runs stamped from a dirty tree* whose exact
bytes are preserved in `refs/jack/failimpl/…`, and the new one is `T0.31` at
`5f2eb58+dirty`, 2026-10-19 08:45. **Not a moved threshold — a provenance gap**,
and it is the same defect the OPEN row
`regate-bought-nine-certificates-from-its-own-uncommitted-tree` (DUE 10-24)
already owns. `PS.09` is the day's one real threshold question and it was
handled correctly: `KA_SIGNAL_MIN` 0.90 was measured *unreachable* and escalated
as `D51` rather than lowered.

**3. Drift from the goal.** Findings 1 and 2. Everything the builder did in 24 h
serves `GOAL.md`'s honesty clause (*"protects the honesty of watching what
happens when the three meet"*) and nothing serves the brain, the body or the
world. That is legitimate work and it is not the same as progress. The converse
question: `coverage` reports **0 commitments with no declared spec** — the
highest-priority class is empty and has been since 09-19. Below it, **3
CLAIM-DEAD** (smell, shelter/building, thermal — each needs a *registered
successor*, held by `D42`'s fired default, the owner's to name), **14
commitments with live claim specs and nothing passing**, and **6 NO-LIVE-PATH**
holes. `GOAL.md` now cites **4 NEW unrunnable specs** — `GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09`, all welded behind `LC.07` — which is owned by
`gen-four-revival-needs-an-affordable-lc07-successor` (OPEN, DUE 10-24) and is
not re-routed here.

**4. Is the builder alive and productive — ALIVE, VERY, AND BLOCKED.** 12 slots,
12 × `rc=0`, 63 commits, zero PASS delta. One `MEMORY=1 process over 1536 MB
peak` notice at the 11:37 slot end, no action taken and none needed at 6 GB
free. No paused loop, no credit exhaustion (`week:all models` read 25% → 32%
across the day against a ~38% pace line), no repeated identical failure. The
one `rc=1` in the window is 2026-10-19 15:16, before the usage-week reset.

**5. Compute honesty.** Finding 5. CPU: finding 2's second paragraph. Standing
and unchanged: `gpu_hours_no_verdict` TOTAL **51.47 h**, of which `D1.0` holds
33.78 h across 2 attempts for **0 verdicts** and `UNATTRIBUTED` holds 6.32 h
across 21 jobs — both at their declared floors, both still the largest unaudited
compute in the project.

**6. Stuck decisions.** Finding 4 (six stale CONDUCT-DESK) and finding 3 (`D44`,
four days out). `decisions --check` reports **no `MEANS-ESCALATED`, no
`UNDECLARED`, no `OVERDUE — DEFAULT IS DUE TO FIRE`**, and `firing
transcriptions: 34 of 34`. The D1 disease is not present in its original form:
nothing a measurement could settle is sitting on the owner's desk. Its
descendant is finding 4. Nothing was quietly acted on without a record — the
`D50` case is exemplary in the other direction: the Review routed it, the
instrument reclassified it `CONDUCT-DESK`, and the desk took the correction and
executed it in the same sitting.

**7. Bakeoff hygiene.** Finding 3 is the headline. The standing structural
defects are unchanged and all at their ratchet floors: `champions` reports **10
violations**, including `VERDICT-IS-A-VOID` on the Learning core (seated BY
VERDICT off `LC.03`, which is a VOID — *a VOID decided nothing*), `World` held
BY VERDICT with **no deciding run named and no re-open trigger declared**,
`TRIGGER DEBT 3/3`, `unwinnable 4/4`, `unfalsifiable 2/3` (ASR, Speaker ID).
`champions --check` exits 0 because every one of these is at its floor. The
repair for the worst of them was registered today (`LC.08`) and the repair for
`World`'s is finding 1.

**8. The honest summary.** **No. We are not closer to a curious humanoid that
climbs a ladder than we were yesterday, and we are measurably closer to a longer
list of green ticks.** Twelve green slots, sixty-three commits, two first-ever
verdicts in a week and both of them about our own instruments. Today's work was
good work — a real defect in the metric reader fixed with reproductions before
repairs, a GPU kernel recovered instead of resubmitted, a builder paying for its
own `T0.22` violation in the slot that caused it. I would not undo any of it.
But the honest sentence is that **the apparatus got better at measuring a
creature that did not change**, and the two things standing between the loop and
Jack are both decisions on desks rather than work at a keyboard: the `T1.08`
ruling due tomorrow onto an over-promised day, and a world whose two failed
specs now have, in the literal sense, nobody.

---

## FOR THE BUILDER

1. **Make `coverage._owned_by_dued_row` status-aware: a terminal row owns
   nothing.** `coverage.py:1352-1417` credits `"queue-row"` ownership on any
   `DUE:` declaration in a block that names the spec, with no check on the row's
   status. **Measured consequence: 6 settled FAILs — `W.1`, `W.2`, `T2.07`,
   `ME.11`, `ME.11.D`, `W1.00` — are credited to rows that are all `ACTED` or
   `DECLINED`, while `fail_unowned` reads 0 AT its floor.** `review_queue`
   already publishes the live set (`LIVE = ("OPEN","HELD","DISPOSITIONED")`,
   `review_queue.py:268`) and `_owned_by_dued_row` already imports the module —
   use that constant rather than a second definition, for the reason the
   docstring already gives about `_DECL`. This can only *raise* `FAIL-UNOWNED`,
   so **raise `FAIL_UNOWNED_BASELINE` in the same commit with the reason in its
   growth log, exactly as the ratchet contract requires** — do not let the number
   grow silently and do not weaken the test to keep it at 0. Add the terminal-row
   shape to `_fail_unowned_fixture()` so it cannot regress. §1.
2. **Commit `LC.08`'s evidential base, or commit its checksums.**
   `experiments/artifacts/lc03_curves_seed{0,1,2}.json` are 632 KB total,
   gitignored at `.gitignore:51`, and are the sole surviving evidence behind the
   only challenger the Learning-core seat has. `LC.03` is VOID-FORECLOSED under a
   do-not-re-dispatch order, so they cannot be regenerated. Either add a narrow
   un-ignore for these three paths, or — if committing run artifacts is against
   the grain — commit a `sha256` manifest and have `_load_artifact` verify
   against it, so loss is *detected* rather than discovered at dispatch. The
   replay gate is already fail-safe; this is about recoverability, not honesty.
   §6.
3. **Declare `WAITS-ON:` on the five 2026-10-21 rows that lack it.**
   `review-queue` withholds that day's decision count because 5 of its 11 live
   rows do not declare, and 2026-10-21 is the day the `T1.08` ruling falls due.
   A withheld count on the project's most important day is the one day the
   grouping was built for. Declaration-only — it buys no exemption, moves no
   date and no number. §7.
4. **Ride-along, no standalone edit:** `run status`'s steering-page size line
   renders `126721 bytes; … -1721 below the 125000 self-imposed ceiling` and
   then correctly flags `!! OVER` on the next line. The sign is inverted in the
   render only; the comparison is right. Fix it with the next edit that bills
   that file anyway — it is not worth staling a certificate for a minus sign.

## FOR THE OWNER

**1. `W.1` and `W.2` — the world's two measured failures — now have no living
owner, and I need one sentence from you to give them one.** This is new and it
is not a re-ask. The chain, each link verified at this desk today: the three
queue rows that owned their repair were **DECLINED** by an armed stop-rule
firing on its own condition (the rule lives on `D33`); the class question was
escalated to you as `D43`; and **`D43`'s own armed default then fired** as
*"(iii) HOLD — no world-edit window is declared and the seven measured findings
stay DECLINED"*. The result is that two
specs encoding your own constitutional text — *"too cold kills him"* and *"the
needs ARE the curriculum"* — are `FAIL`, block 4 specs each, hold the World
seat's arena red, and are assigned to nobody. The instrument that exists to
catch this reads **0** and sits at its floor, for the code reason in FOR THE
BUILDER 1. **Your `D43` entry explicitly left the window yours to declare at any
time; declaring a narrow, single-purpose one is the one act that restores a
path.** I am not re-opening `D43` and not firing anything — the default was
legal and I am reporting what it cost, which is what `D40` required of this
desk.

**2. `D44` fires in four days, and it is the one decision `GOAL.md` hands to you
by name.** Armed, `decide_by 2026-10-24`. Its subject is the shipped character
sheet — five undocumented floats and a 2,585-character persona conditioning
Jack's default answer path, against `GOAL.md:172` (*"Who he becomes is not
written here… we deliberately refuse to carve his character"*). Its default is
**no ruling, nothing documented, the contradiction stays live**, and the entry
itself says that is deliberately *not* the recommendation. Context you should
have when you read it: **of 36 resolved decisions in this project's history, 34
were settled by an armed default firing on silence and exactly one — `D19` — was
ruled by you.** The mechanism has kept the project from deadlocking and I would
not remove it. But if `D44` fires, a clock will have decided who Jack is, and
that is the single thing this page's own constitution says a clock may not
decide. Blocks no spec. Costs nothing to answer. Four days.

**3. Reported, not asked — two consecutive weeks of free GPU remain
undecidable.** `2026-W41` closed at **0.0 h of 30.0** spent. `2026-W42` has
drawn **2.078 h**, of which **1.995 h (96%) went to jobs recorded as failed**,
and it ends **Saturday 2026-10-24**. The Review carried this to you yesterday
and its reading stands; I add one measurement it could not have: a job
**authorised at 0.2 h billed ~2.0 h across two backends today**, logged in
`overruns`, with nothing refusing it. The hours are still a symptom and I still
do not recommend buying them — the underlying fact is that the only specs that
could absorb them sit behind `T1.08`, whose ruling is the Review's and falls due
tomorrow. Nothing here is blocked on you.

---

*Written against a live builder; every volatile count re-read after the 18:07
slot closed at 18:41:32. No spec, test, model file, ledger row or decision was
touched by this audit.*
