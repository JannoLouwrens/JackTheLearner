# OVERSIGHT — 83rd audit, 2026-09-07 18:37–19:0x UTC (at `3babc3a`, tree clean)

## VERDICT: ON TRACK — **the ledger is sound, nothing was loosened, and 24 of 24 iterations ran green. The day's findings are all in the COMPUTE-ACCOUNTING lane: the `gpu_hours_no_verdict` reading that shipped this afternoon prints 36.91 h and is structurally blind to 11.86 h more on `T2.01` and `T2.02`, while 17.48 h of charged jobs name no spec in any machine-readable record this project keeps.**

Sections 1, 2 and 4 are clean and I want that stated plainly before the
findings. **108 PASS rows, zero dead commits** (checked across every head row
*and* every history row), every PASS has an implementation on disk, every PASS
spec declares a control, and the only two rows without `control_metrics` are
`T0.01`/`T0.10`, which declare `"NONE, BY DECISION (52nd audit B5)"` in their
own registry text. **Nothing was loosened in seven days**, positively verified:
every deletion the diff shows is a conjunct being *added* beside it
(`ME.9`'s `MIN_DISTRACTOR_EVAL` 9 → 12, `T0.35`'s `control_is_blind` →
`control_is_blind and control_onehop_blind`, `T1.01`'s
`control_did_not_learn and mode_declared`, `D1.0`'s
`sigma_vs_random` → `paired_t` against its own untrained twin with the 3.0 bar
**unmoved and both conjuncts live**). No `_check` gained an `or`. No seed count
fell. **Eleven certificates are STALE and not one of them is a PASS** — every
resident is FAIL or VOID.

The findings are ranked by how much damage they do to the trustworthiness of
what this project reports about itself. **Neither of the top two touches a
claim on the ladder**, and I am saying so rather than dressing them up.

---

## 1. THE FINDING — the instrument shipped today to price compute-without-verdict cannot see the most expensive compute-without-verdict on the board

The 82nd audit's B2 landed at 16:07 (`a5de52f`) and `run status` now prints:

```
gpu_hours_no_verdict = {'TOTAL': '36.91 h',
                        'D1.0': '33.78 h / 2 attempt(s) / 0 verdict(s)', ...}
```

That number is correct for what it joins, and what it joins is
`gpu_budget.json` → every ledger row's `gpu_job_id`. **Five of the 21 remote
(GPU) rows in the ledger carry `gpu_job_id: None`, and two of them are the two
most expensive non-PASS GPU rows this project owns:**

| spec | row | `ran_at` | `duration_s` | `gpu_job_id` |
|---|---|---|---|---|
| `T2.02` | head **VOID** | 2026-08-09T07:30:25 | 22 604.42 s = **6.28 h** | `None` |
| `T2.01` | head **FAIL** | 2026-08-12T12:59:15 | 20 097.42 s = **5.58 h** | `None` |
| `T1.08` | head PASS | 2026-08-12T08:17:36 | 1 983.01 s | `None` |
| `T1.07` | head PASS | 2026-08-14T02:37:02 | 1 606.68 s | `None` |
| `T0.09` | head PASS | 2026-08-30T10:09:39 | 34.16 s | `None` |

`T2.01` is the spec at the top of `run blocked` — **frees 35 / blocks 38**,
settled FAIL, implementation unchanged 29 days. `T2.02` is a `gpu<8h` VOID and
the ladder's one remaining pre-`impl_sha` stale row. Both are exactly the rows
the new reading exists to price, and both read as **zero**. Its true total is
**≥ 48.8 h**, not 36.91 h — a 32% understatement, concentrated on the frontier.

**The join is recoverable and I did it.** `gpu_submissions.jsonl:10` records
job `jannolouwrens/jack-ladder-1786519461` with `duration_s 20093.64` and
`charge_seconds 20087.11`; the result line lands at
`ts 1786539555` = **2026-08-12T12:59:15**, matching `T2.01`'s head row *to the
second*, at 5.5798 h. Job `1786304547` submits ~2026-08-09T19:42 and its
5.5786 h lands at 2026-08-10T01:17 — `T2.01`'s history FAIL is
`2026-08-10T01:17:15`. This is not a new discovery: **the 17th audit already
wrote both ids down in prose**, at `docs/DECISIONS_NEEDED.md:1611` — *"billed
5.58 h on each of the two occasions it has run (`1786304547`,
`1786519461`)"*. **Eleven point one six GPU-hours have been known to belong to
`T2.01` since 2026-08-14 and no machine has been able to read it since.**

### 1b. The other direction of the same hole: 17.48 h charged to no spec at all

Joining `charged_jobs` → ledger → `gpu_submissions.jsonl` (attempt/attribution/
result lines, on `attempt_id`):

```
63 charged jobs, 63.05 h of per-job records
  35 jobs claimed by a ledger gpu_job_id
   5 jobs attributed only via gpu_submissions.jsonl   2.13 h
       SM.02 (pilot) 1.558   LC.07 (pilot) 0.440   T2.04/T2.05/T2.06 probes 0.136
  23 jobs attributed BY NOTHING                      17.48 h   ← 27.7% of all
     per-job records                                            per-job records
       5.5798 h  W32 ok  1786519461   (submission record exists, spec field empty)
       5.5786 h  W32 ok  1786304547   (no submission record at all)
       ...21 more, 0.05–0.99 h each
```

The two 5.58 h jobs at the head of that list are `T2.01`'s, per §1. The
remaining 6.32 h across 21 jobs is genuine bookkeeping debris, most of it W32.

**Why this is a finding and not a chore.** Two of this project's standing
lessons are that a shrink-only counter must count the whole class, and that an
instrument reading clean over a domain it cannot see is worse than no
instrument. `gpu_hours_no_verdict` is one week old, it is honest about its
inputs, and it will now be quoted — the builder's own 18:10 journal entry
already leans on GPU arithmetic to reason about `D1.0` attempt 3. A reader who
takes 36.91 h as the project's compute-without-verdict total is off by at least
a third, in the direction of thinking the ladder is cheaper than it is.

**The repair has a precedent in this repo and it does not touch the ledger.**
`gpu_submissions.jsonl:50` is an *attribution line* appended by the 20th
audit's own B2 backfill — *"the SM.02 pilot was dispatched outside `run_spec`
so its attempt receipt reads `spec:""`; this line names it"* — joined by
`attempt_id`. That is the idiom. Ordered as **B1** below: append attribution
lines, teach the reader to follow them, and print the unattributable remainder
as its own number so it can only ever shrink. **No ledger row is edited, no
threshold moves, nothing is re-run.**

---

## 2. Core-hours and wall-hours are the same number on the owner's desk, and they are not the same number on this box — `D24` decides in 4 days

`experiments/cpu_budget.py` is titled *"CPU-hour accounting"* and bills, by its
own docstring at `:23`, *"the wall clock actually spent"*. `bill_interval`
charges `seg - t0`: pure wall, no core weighting. That is a deliberate,
documented choice and it is self-consistent.

**The planning side is written in core-hours and divides by that wall ceiling.**
`docs/DECISIONS_NEEDED.md:5023`, the `D24` addendum, `decide_by` **2026-09-11**:

> *"~618 core-hours = 38.6 fully-billed 57,600-s days ≈ 5.5 weeks of this box's
> ENTIRE CPU day budget"*

and, in the same paragraph, *"the largest single run (arm, 4.0M decisions) is
**48.0 core-h** … but that lands it exactly in the `cpu<48h` class"* — a class
defined by `spec_child_timeout_seconds`, i.e. wall.

**Measured today, on this box, twice.** The two detached `PL.02` probes:

| probe | LAUNCH → last write | billed | own recorded CPU | ratio |
|---|---|---|---|---|
| `pl02_steps_probe` | 14:23:37Z → 14:48:22Z = **1 485 s wall** | **1 485.39 s** | 2 498.9 cpu-s | **1.68×** |
| `pl02_rig_probe` | 14:13:10Z → 14:37:43Z = **1 473 s wall** | **1 473.33 s** | 2 529.9 cpu-s | **1.72×** |

`scripts/ladder_loop.sh:253` exports `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, so
2.0× is what the environment permits and ~1.7× is what these took, on a
4-core box.

At the measured 1.7×, **618 core-hours is ~363 wall-hours ≈ 22.7 fully-billed
days**, not 38.6; at the 2.0× the env allows, 19.3. And 48.0 core-h is ~28
wall-h, which is not where the `cpu<48h` class placement was argued from —
that class's own foreclosure row, `cpu48h-class-self-forecloses-the-day-meter`,
is **DUE tomorrow**.

**The direction of the error is the part that matters to this desk.** It
*over*-states the venue's cost, and `D24`'s armed default is *"(iii) DECLARE IT
UNAFFORDABLE"*. `champions --check` already reports the Learning-core seat as
`VERDICT-IS-A-VOID` **and** `TRIGGER-UNREACHABLE` — every re-open door closed.
An arithmetic that mixes units in the direction of foreclosing the only arena
of the seat everything else rests on is precisely what this organ is for.

**The honest limit, stated rather than buried: nobody has measured `LC.07`'s
own thread width.** My 1.68/1.72 readings are torch-training probes; `LC.03`'s
survival runs are MuJoCo physics and may well be single-threaded, in which case
core-h ≈ wall-h and 38.6 is right. The artifacts that would settle it —
`lc03_curves_seed*.json` — **are not on disk**; I looked, and their absence is
the same fact field watch wk6 reported as *"no trained `A4` weights exist"*.
**So the repair is a measurement, not a number change** (B2 below). I am not
asking anyone to move 38.6; I am asking that the ratio sit next to it before a
default fires on it.

### 2b. The builder's own reported smell is refuted, and I would rather say so than let it stand

The 14:42 journal entry reports, explicitly *"for the overseer"*: *"detached
billing undercounts a long run's final segment (steps probe billed 1485.39 s vs
2498.9 cpu-s in-probe)"*, diagnosed as *"heartbeat tail unbilled"*.

**Both halves are wrong and the code says so.** `_wrap`'s loop bills
`[last, now]` on the `done` branch as well as on `TimeoutExpired`, so no tail is
lost; and the arithmetic above shows the steps probe billed **1 485.39 s against
1 485 s of wall — exact.** The gap is wall-vs-CPU (§2), not a lost tail. Left
uncorrected this sends the next reader to patch a loop that is already right,
and away from the unit question that is actually live on `D24`. The
*observation* was good and reporting it was correct; only the mechanism named
was wrong.

---

## 3. `pl02-dependency-on-pl00-verdict-vs-table` is DUE today, its ordered work is finished and on the ledger, and it will go OVERDUE at midnight anyway

`run review-queue` prints **0 violations** and one row dated today. That row is
`DISPOSITIONED` — which the tool correctly does *not* count as a disposal, so
it is still live and still ageing.

The disposition was made this morning as **(i)+(iii)**: the `PL.02 → PL.00`
edge STANDS, and a clearing arm was ordered to *"dissolve the edge by
satisfying it"*. Both halves are done:

- `b7324ba` ran the renderer bakeoff, arm (iii) — winner `coarse-shadow512`,
  worst-seed 8.594 against the **unmoved** 5.0 floor;
- **`PL.00` PASS attempt 2 at 2026-09-07T11:22:15**, 128.23 s, on the ledger;
- `run coverage` now lists `PL.02` as **RUNNABLE**.

The row's own order is discharged in substance and it is still wearing a live
`DUE: 2026-09-07`. At 00:00 it becomes `OVERDUE` — *"a dated promise that was
broken"*, the strongest signal in that file — and moves `review_queue_violations`
off a shrink-only floor of 0, for work that was completed inside the day it was
promised. The Review runs at 06:37 and can mark it `ACTED`; it will read this
first. **Not the builder's to touch** — I am naming it, not routing it.

---

## 4. The builder mis-numbers this organ, and two commits now claim orders that will not exist where they point

`02f9df7` and `a5de52f` (15:15 and 16:07 today) execute *"83rd audit B1"* and
*"83rd audit B2"*. Those are the **82nd** audit's B1 and B2 — `2b3e8a6`,
12:49, the only `OVERSIGHT.md` revision between them, and its `FOR THE BUILDER`
section carries exactly the near-miss detector and `gpu_hours_no_verdict`.
Audits run on a 6 h cadence (81st 06:37, 82nd 12:37); **this one is the 83rd**,
and its B-items are below. So the git history now holds two different "83rd
audit B1/B2". Cheap to avoid and not worth a code change: **quote the
`OVERSIGHT.md` commit sha, not the ordinal.**

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** 143 rows: 108 PASS / 22 FAIL / 13 VOID /
0 NOT_RUN / 0 ERROR. Every `commit` on every head row *and* every history row
resolves in git (**0 dead**). Every PASS resolves to an implementation in
`experiments/tests/`. Every PASS spec declares a `control`; 106 of 108 carry
`control_metrics`, the two exceptions arguing `"NONE, BY DECISION"` on their own
specs. **11 STALE claims + 1 pre-`impl_sha` stale, and none is a PASS** —
`T3.07`, `ME.11.B/C/D`, `T3.09`, `XL.01`, `LG.10` (FAIL) and `UB.10`, `D1.0`,
`LF.01`, `SO.07`, `T2.02` (VOID). `T2.10` cleared the lane at 17:12 with an
honest re-bought FAIL. `ME.11.C` and `T3.07` are held deliberately and the
reasons are written down.

**2. Thresholds and controls — NO LOOSENING, positively verified.** See the
header. The one movement in seven days is `ME.9`'s `MIN_DISTRACTOR_EVAL` 9 → 12,
a tightening with its arithmetic at the constant. `SO.08`'s budget re-declared
`cpu<2h` → `cpu<1min` (`7aa9619`) is a **cost class re-sized on a measurement**
(0.74 s/seed at full `N_ROUNDS=240`) — the 75th audit's own F1 repair, and no
verdict threshold. `ME.3` gained `raw_answer_rate >= 0.95` as a *new required*
conjunct; `A5`'s 0.552–0.688 against `A0`'s 0.625 is recorded as a restoration
and not written up as an improvement, exactly as ordered.

**3. Drift — none, and the composition is honest.** Today's committed work:
`PL.02` implementation + rig decomposition (GOAL.md:76, the PLASTIC-ONLY
decree's **sole registered falsifier**), the `PL.00` renderer bakeoff (the eye
— *"every sense a human has"*), `ME.3`'s contract split and `ME.1`'s floor
(*"memory makes it him"*), `T2.10`'s re-buy (episodic retrieval), and two
instrument repairs (`decisions.py` near-miss, `gpu_hours_no_verdict`). Every one
traces. **The converse is the uncomfortable half and it has not moved:** four
commitments remain CLAIM-DEAD (smell, balance, shelter/building, thermal), nine
more have live claim specs and nothing passing, and *curiosity* has 12 specs
and 2 passes while *one brain / unison* has 25 specs and 1. `coverage` EXIT 2 on
that, routed as `five-commitments-are-claim-dead-behind-foreclosures` (DUE
09-11). **0 commitments with no declared spec** — the gate that matters most is
green.

**4. Builder liveness — 24 of 24 iterations `rc=0`** in the trailing 24 h
(2026-09-06 19:14 → 2026-09-07 18:11), demonstrated **106 → 108** (`ME.3` 09:12,
`PL.00`+`T0.35` 11:25). No repeated identical failure, no pause, no abort on
load. Credit meter `week:all models` **16%** at 18:37 — not the constraint.
`lost_iterations.log` still 0 bytes. The last four slots each ended having
verified an empty board against `status`/`next`/`blocked`/`coverage`/`decisions`
and stopped early, which is the standing rule working, not a stall.

**5. Compute honesty — see §1 and §2.** Beyond those: **W36 reads 17.7238 /
30 kaggle-hours, 12.28 h free**, and the week opened yesterday. `D1.0`'s two
attempts measured **16.17 h** (W35) and **17.61 h** (W36); I verified both
against `charged_jobs`. **A third attempt cannot fit W36's remainder** — the
builder's 18:10 arithmetic is right, and it belongs in tomorrow's
`d10-successor-rerun-under-adopted-gate` ruling, which is where it put it.
CPU: 6 227.34 s of 57 600 used today, of which **5 621.75 s (90.3%) is the six
detached `PL.02` probes**. The 51.58 h gap between `sum(weeks)` 114.64 h and
`sum(charged_jobs)` 63.05 h is W31's 45.20 h of pre-per-job records plus the
labelled 6.38 h W32 opening balance — **documented, not a finding.**

**6. Stuck decisions — nothing improperly parked.** `decisions --check` EXIT 0:
**0 UNDECLARED** (so there is nothing for me to arm this audit, and I am saying
that rather than manufacturing an arming), **0 MEANS-ESCALATED, 0 OVERDUE, 0
unrouted or vanished owner-asks.** Both of `PROGRESS.md`'s `FOR THE OWNER` items
are matched to `D25`/`D24` by citation. **`D17` falls due TODAY** and its default
fires tomorrow if unanswered; the 00:37 audit inherits it. Note without
alarm: `D17`'s default text names *"a renderer-cost bakeoff over the arms named
above"* as follow-on builder work, and that bakeoff **already ran today** via
the Review's independent `pl02-…` disposition (`b7324ba`). Convergent, not a
quiet enactment — different route, same act, both recorded.

**7. Bakeoff hygiene — clean this window, with one standing red correctly
flying.** `ME.1`/`ME.3`'s floor was settled by a six-arm bakeoff against four
pre-registered scar shapes with `A5` the sole survivor, after `decisions.py`
*refused* its escalation as `MEANS-ESCALATED` — the law working on the organ
that wrote it. No winner was chosen inside a noise margin. The one VOID-treated-
as-a-verdict is `D10` → the Learning-core seat, **and it is not hidden**:
`champions --check` prints `VERDICT-IS-A-VOID` and `TRIGGER-UNREACHABLE` against
it, and the seat's marking carries its own single-arm caveat. That is the honest
handling of a bad seat, and §2 is about the arithmetic that decides whether it
ever gets contested.

---

## The honest summary — are we closer to a curious humanoid that climbs the ladder?

**Marginally, and today's gain was in his eye rather than his mind.** The
renderer bakeoff found that 40 ms of "eye" was 23 ms of 4096² shadow pass plus
13 ms of 4× MSAA rendering **four thousand pixels** — two full-scene passes
nobody chose, serving a thumbnail — and killing them made the floor reject
encoders instead of eyes for the first time. Then the `PL.02` decomposition
found that `.mean(axis=2)` had been quietly throwing away the colour channel
that carries most of the signal PG.6 certified: raw radius R² **0.5614 in grey,
0.9327 in RGB, at the same resolution and the same render cost**. Both are real
and neither is a scoreboard move. `demonstrated` has not budged since 11:25.

**What I cannot report as progress is that he still cannot smell, cannot
balance, and cannot build a shelter, and has not been able to for three weeks.**
Four constitutional commitments are claim-dead behind foreclosures whose repairs
are all redesigns on one desk. Twelve curiosity specs have bought two passes.
Twenty-five unison specs have bought one. The ladder-and-apple standard is not
closer today than it was yesterday; the instruments around it are.

**And the day's two findings share a shape worth naming.** Both are places where
this project measured something real, wrote it down in prose a person can read
— the 17th audit naming `T2.01`'s two job ids, the `D24` addendum pricing the
CPU venue — and then built an instrument that reads a *different* field and
reports a clean, confident, smaller number. Prose is where this project's
knowledge goes to become unreadable. The repair each time is the same and it is
cheap: **make the join machine-readable, and print what still will not join.**

---

## FOR THE BUILDER

1. **Make `gpu_hours_no_verdict` follow the attribution path, and print what
   still refuses to join.** Two changes, both additive:
   (a) **Append attribution lines to `gpu_submissions.jsonl`** in the
   `:50` idiom — same shape, joined by `attempt_id`, reason stated on the line —
   naming `jannolouwrens/jack-ladder-1786519461` and
   `jannolouwrens/jack-ladder-1786304547` as **`T2.01`**, with the evidence in
   the line itself: the first's result `ts 1786539555` = 2026-08-12T12:59:15 and
   `duration_s 20093.64` against `T2.01`'s head row's `2026-08-12T12:59:15` /
   `20097.42`; the second's 5.5786 h landing at 2026-08-10T01:17 against
   `T2.01`'s history FAIL at `2026-08-10T01:17:15`; and
   `docs/DECISIONS_NEEDED.md:1611`, where the 17th audit named both ids in
   prose on 2026-08-14. `1786304547` has **no submission record at all**, so it
   needs a synthesised `attempt_id` — say so on the line rather than inventing a
   receipt. **Do not touch `experiments/ledger.json`.** Rows are settled; the
   attribution file is where provenance is repaired here, by this project's own
   precedent.
   (b) **`run status`'s reading joins ledger `gpu_job_id` FIRST, then
   `gpu_submissions.jsonl` attribution, and prints a third figure:
   `gpu_hours_unattributed`** — charged hours that join to no spec by either
   path, today **17.48 h across 23 jobs**. Make it shrink-only with a declared
   floor, in the same idiom as `fail_unowned`. **MEASURE AND REPORT, GATE
   NOTHING** — no dispatch refused, no spec failed, no threshold moved, monotone
   by construction. Known-positive for the fixture: a charged job reachable only
   through an attribution line must be counted, and one reachable through
   neither must appear in the unattributed figure and not silently vanish.

2. **Measure the thread width of one `LC.03`-class arm-seed and put the ratio
   next to the 618 core-hours, before 2026-09-11.** `D24`'s addendum equates
   core-hours with 57 600-s wall days (`docs/DECISIONS_NEEDED.md:5023`) and I
   measured 1.68×/1.72× divergence on this box today (§2). One short run, wall
   clock and `process_time` both recorded, is enough; **`OMP_NUM_THREADS` /
   `MKL_NUM_THREADS` and `nproc` must be recorded beside it or the ratio means
   nothing.** Append the reading to the `D24` addendum as a second dated
   addendum. **Change no number in the existing one, arm no new option, move no
   threshold** — if the ratio comes back 1.0 the addendum is vindicated and the
   entry is stronger for having been checked. If it comes back near 1.7, say so
   plainly: the owner is four days from a default that declares an arena
   unaffordable on arithmetic that would then be ~40% high. Cost: minutes.

3. **`experiments/cpu_budget.py`'s module docstring says "CPU-hour accounting"
   and the module bills wall clock.** One sentence in the docstring, at the
   `Scope, stated honestly` list where it belongs: the metered unit is **wall
   seconds of the child**, not core-seconds, so a child permitted 2 threads by
   `ladder_loop.sh:253` can consume up to 2 core-seconds per billed second —
   measured 1.68×/1.72× on 2026-09-07. **Do not change `CPU_DAY_CEILING_S`, do
   not change `bill_interval`, do not add a core weighting.** The meter's
   behaviour is deliberate and its refusal semantics are load-bearing; what is
   missing is that it says so where a planner writing core-hours would read it.
   This is documentation of an existing measurement, not a policy change.

4. **Quote the `OVERSIGHT.md` commit sha, not the audit ordinal, in commit
   messages.** `02f9df7` and `a5de52f` both say *"83rd audit"* for orders that
   are the **82nd**'s (`2b3e8a6`); this file is the 83rd. Free, and it stops the
   history holding two of everything.

5. **Standing prohibitions, unchanged and restated:** no third `D1.0` dispatch
   before tomorrow's `d10-successor-rerun-under-adopted-gate` row answers — and
   note independently of that ruling that **W36's 12.28 free hours cannot hold a
   16–17.6 h attempt**, so authorisation means W37 or a smaller design;
   `PL.02`'s registered run stays blocked behind the 09-09
   `pl02-eye-gate-reads-the-encoder-not-the-eye` ruling and the gate is not to be
   touched meanwhile; `T3.07`'s stale re-buy stays declined; `ME.11.C` stays
   stale under the ACTED `me11` family row; `HR.1`–`HR.4` stay `D19`-held to
   09-14; `LF.01` attempt 2 waits for the 09-09 design.

---

## FOR THE OWNER

1. **`D24` (decide_by 2026-09-11) is priced in a unit its own ceiling does not
   use, and the error runs toward foreclosing the arena.** Nothing for you to
   answer today; a correction is in flight and will reach you before the
   default fires. The addendum reads *"~618 core-hours = 38.6 fully-billed
   57,600-s days"*, and `experiments/cpu_budget.py` bills **wall** seconds, not
   core-seconds — measured today on two live probes at **1.68×** and **1.72×**
   CPU-to-wall, with `ladder_loop.sh` permitting 2.0×. If `LC.07`'s workload
   threads like those probes, 618 core-hours is ~22.7 fully-billed days rather
   than 38.6. **`D24`'s armed default is (iii) DECLARE IT UNAFFORDABLE**, so the
   overstatement argues for the option that leaves the Learning-core seat — held
   `BY VERDICT` off a **VOID**, with every re-open trigger already a closed door
   — with no reachable arena at all. I am **not** recommending a different option
   and I am not asking you to move the 10×; the honest position is that
   `LC.07`'s thread width has never been measured, the artifacts that would
   settle it are not on disk, and a one-run measurement (ordered as B2, cost:
   minutes) tells you whether the number in front of you is right before you
   have to rule on it. **If it comes back 1.0, the entry is unchanged and
   stronger.**

2. NO-DECISION: **compute honesty, reported because the number just became
   quotable.** `run status` began printing `gpu_hours_no_verdict` this afternoon
   — GPU hours bought against verdicts returned — and reads **36.91 h, of which
   `D1.0` is 33.78 h across 2 attempts for 0 verdicts.** That reading is
   understated: **`T2.01` (5.58 h, FAIL) and `T2.02` (6.28 h, VOID) carry no job
   id in the ledger and are invisible to it**, and a further **17.48 h of
   charged jobs are attributable to no spec by any record this project keeps**.
   The honest total for compute that bought no verdict is **≥ 48.8 h**, and the
   honest total for compute nobody can name at all is **17.48 h out of 63.05 h
   of per-job records (27.7%)**. Ordered as B1 — an attribution backfill and a
   third printed counter, gating nothing. Nothing on the ladder is affected: no
   PASS depends on any of it.

3. NO-DECISION: **the desk-capacity report the Review gave you this morning is
   confirmed from here, and it got worse by six rows during the day.**
   `review-queue` now reads **42 live rows**, drain **UNBOUNDED**, 36 arrivals
   against 3 disposals over the trailing week, and **10 rows share
   2026-09-13** against a measured capacity of 1 dated row per cycle. The
   builder ran **24 of 24 iterations green** and spent its last four slots
   verifying an empty board and stopping early, because everything runnable is
   held behind a redesign that desk owes. I read the Review's own words on this
   before writing mine and I have nothing to add to them except a second
   measurement agreeing.

4. NO-DECISION: liveness. All four organs live against `/data/jack-logs` mtimes,
   not anyone's report: builder 18:11 (hourly, 24/24 `rc=0`), review 06:54,
   field watch 05:56 today, overseer 18:37 (this run).
   `lost_iterations.log` still 0 bytes and still never exercised. Credit meter
   `week:all models` 16% — credits are not the constraint this week, and neither
   is the GPU quota until W37 opens.
