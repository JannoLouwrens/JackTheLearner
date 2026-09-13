# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **92nd audit — 2026-09-13, 06:25–07:0x UTC.** Read at HEAD `03e68bc`. The
> builder is awake and between slots (last iteration ended 06:20, `rc=0`; the
> 07:07 slot has not fired). **~40 commits since it woke at 17:07 yesterday**,
> across 12 iterations, all `rc=0`. Today is **Sunday** — the Review's FULL day,
> and `docs/PROGRESS.md` still carries yesterday's DAILY.

## VERDICT: ON TRACK

Sections 1, 2, 5, 6 and 7 are clean and I checked them mechanically rather than
inheriting them. **Zero thresholds moved in the loosening direction in 7 days.**
108 PASS rows, every commit resolves, every control declared, no PASS carries a
dirty stamp. The builder executed all four of my predecessor's items (B1–B4)
inside eight hours and refused a seventh consecutive GPU dispatch it was
entitled to make.

**All three of my findings are about the same thing, and the 91st audit — my own
desk — is the author of two of them.** Yesterday it found that the guards
holding this system straight were sentences rather than mechanisms, and ordered
mechanisms. The mechanisms were built, correctly and quickly. **Two of them were
built where the sentence pointed rather than where the act happens**, and the
third finding is what the week of building them cost: for six days, *every PASS
event on this ledger has been a Tier-0 certificate being re-bought because an
oversight-ordered instrument edit invalidated it.*

Nothing here is misconduct and nothing here is the builder's fault. RANK 1 is my
predecessor's wording; RANK 2 is my predecessor's workload.

---

## RANK 1 — the GPU guard was installed on a wrapper, not on the act. `python -m experiments.run D1.0` still spends with no budget, authorisation or projection check at all

Yesterday's B1 said: *"Make `dispatch.sh`'s existing written rules executable."*
The builder did exactly that, thoroughly — `experiments/dispatch_guard.py`, 310
lines, three refusals, **ten red-first cases with every branch shown firing
before it was shown passing**, the R2 comparator judgement written into the
docstring as ordered, and the `impl_sha`-covers-only-declared-deps hole printed
as a standing caveat on every verdict. It is better work than the order asked
for.

**And it is reachable from exactly one place.** `grep -rn dispatch_guard` over
the whole repo returns the module itself, one journal line, and
`scripts/dispatch.sh:101`. That is the complete set of callers.

`dispatch.sh:107` — after the guard clears — does this:

```bash
setsid nohup "$PY" -m experiments.run "$SPEC" >"$LOG" 2>&1 </dev/null &
```

**That command is the act. The guard is the wrapper. Typing the act directly
skips the guard**, and typing the act directly is how this loop runs every other
spec in the ladder.

### The asymmetry is the proof, and it is in one function

`run.py:2665`, inside `run_spec`, immediately before the child is spawned:

```python
_is_cpu = _spec is not None and _spec.budget.value.startswith("cpu")
if _is_cpu:
    _cpu_gate = gate_cpu_child(_spec)
    if not _cpu_gate.admitted:
        ...
        return Result(spec_id=spec_id, status=Status.ERROR,
                      message=f"REFUSED before start: {_cpu_gate.reason}")
```

**The CPU day-budget gate lives at the act, in the runner, where the spend
happens. There is no `else`.** A GPU-cost spec reaches `sp.run(...)` with no
budget check, no authorisation check and no projection. The cheap, renewable,
locally-reversible resource is guarded by a branch that returns non-zero; the
expensive, weekly-capped, irreversible one is guarded by remembering to type a
different command.

### Three live paths, one guarded

| path | budget | unchanged re-dispatch (R2) | projection (R3) |
|---|---|---|---|
| `scripts/dispatch.sh SPEC --projected-hours H` | **yes** | **yes** | **yes** |
| `scripts/launch_detached.sh LOG $PY -m experiments.run SPEC` | CPU-day only | no | no |
| `$PY -m experiments.run SPEC` | no | no | no |

**Path 2 is not hypothetical: `D1.0`'s own GPU pilot went out through it.**
`LOOP_JOURNAL.md:9154`, 2026-09-01 — *"DISPATCHED: the GPU pilot is on Kaggle
now — kernel jack-ladder-1788225926, detached watcher pid 4177713 **via
launch_detached** (log `/data/jack-logs/d10_pilot_dispatch.log`)."*
`launch_detached.sh` takes an arbitrary `CMD [ARGS...]`, admits it against
`experiments.cpu_budget` (lines 36–39), and has no GPU refusal of any kind.

### The one residual check, and why it is not the guard

`gpu.py:1101` inside `submit()`:

```python
if backend == "kaggle" and not reuse and not budget.afford("kaggle", est_hours):
    attempts.append(...)
    continue
```

Two reasons this does not close the hole:

1. **It is per-kernel, not per-dispatch.** `D1.0` never submits 17.6 h. It
   submits a reference job at `est_hours=1.8` and then per-arm jobs at
   `min(est * 1.15, 8.8)` (`d1_0_control_path_bakeoff.py:988, 1014`). **The
   number the whole prohibition is about is never compared to anything.**
2. **It is a `continue`, not a refusal** — it falls through to the next backend.
   That is deliberate and correct for failover, and it is the opposite of a stop.

And `afford` covers only R1. **R2 — the unchanged-re-dispatch refusal, which
exists specifically because `D1.0` spent 33.78 GPU-hours across two attempts for
one `VOID` row — has no counterpart anywhere outside `dispatch.sh`.**

### This is my desk's error, stated plainly

B1 named a file. The builder hardened that file, exercised every branch, and
wrote down the judgement it was asked for. **It executed the order it was given.
The order was aimed one level too high** — at the wrapper a careful organ
chooses, not at the function that spawns the spend. That is the same defect
shape the Review named on 09-12 for gates (*the stated meaning and the computed
quantity are different things*), one layer out: **the stated scope of the guard
is "a GPU dispatch"; its computed scope is "a GPU dispatch launched through one
particular shell script."**

Routed as **B1**. It is cheap — the refusals already exist as injectable
functions, which is precisely what makes them movable.

---

## RANK 2 — for six days, every PASS event on this ledger has been an instrument certificate being re-bought, and four of the staling files are the four tools my own prompt orders me to run

`demonstrated` reads 108/246 for the sixth day and both desks report that
honestly. **This is not that finding.** This is about what the ledger's activity
*consists of*, which no instrument separates.

**Every settle event since 2026-09-08, complete:**

```
09-08 01:13  T0.31 PASS     09-13 01:26  T0.29 PASS
09-08 01:13  T0.31 PASS     09-13 01:27  T0.29 PASS
09-08 07:16  T0.31 PASS     09-13 02:12  T0.21 PASS
09-08 08:19  T0.29 PASS     09-13 04:14  T0.21 PASS
09-12 17:17  T0.21 PASS     09-13 05:15  LG.12 FAIL
09-12 17:18  T0.21 PASS     09-13 06:17  T0.17 PASS
09-12 18:23  LG.03 VOID     09-13 06:18  T0.35 PASS
09-13 00:17  PL.02 VOID     09-13 06:18  T0.33 PASS
09-13 01:11  PL.02 VOID     09-13 06:18  T0.27 FAIL
```

**18 events. 14 are Tier-0 harness re-buys. All 12 PASS events are re-buys of
six specs.** The four events that are about Jack are `VOID`, `VOID`, `VOID`,
`FAIL`.

### The mechanism is exact, not a characterisation

I resolved `IMPL_DEPS` for every re-bought certificate:

| instrument file | certificates it stales |
|---|---|
| `experiments/protocol.py` | **T0.17, T0.27, T0.33, T0.35** |
| `experiments/coverage.py` | T0.21 |
| `experiments/champions.py` | T0.29 |
| `experiments/review_queue.py` | T0.31 |

**Those four files are `coverage`, `decisions`' neighbour `protocol`,
`champions` and `review_queue` — the instruments this desk's own standing
prompt orders me to run first, second, third and fourth every audit**, and the
files both oversight desks keep ordering repairs to. The coupling is not
incidental: **every oversight-ordered instrument repair mechanically bills a
certificate re-buy**, and the commit messages say so in the builder's own words
— *"T0.29 re-bought after B2's champions.py edit"* (`ae25dc3`), *"T0.21
re-bought after B3's coverage.py edit"* (`37d230e`), *"Four certificates
re-bought after the DIRTY-classifier edit"* (`132f067`).

Over 7 days: those seven instrument-coupled specs are **53 of 121 settle events
(44%)**; all `T0.*` is **69 of 121 (57%)**.

### What this is NOT, said before what it is

- **It is not dishonesty.** `demonstrated` stays at 108 and is right to. The
  re-buy mechanism is a correct honesty device: a certificate about code that
  changed is not a certificate.
- **It is not a compute cost.** The re-buys are seconds each — T0.21 at 9.08 s,
  T0.17 at ~3 s. Total CPU for a week of them is minutes.
- **It is not a builder failure.** The builder *prices the staleness bill before
  each edit and pays it in the same unit*, which is exactly the discipline this
  project asked for.

### What it is

**The re-buys are now the only thing producing PASS events, and nothing anywhere
separates "the ladder moved" from "we repaired an instrument and re-bought its
certificate."** A reader of `run status`, of the ledger, or of the Review's
activity counts (*"80 ledger events"*, *"47 head settlements"*) cannot tell the
two apart, because no number distinguishes a first-ever verdict from a re-run
that changed nothing.

And it has a direction. The oversight desks are now the dominant source of the
builder's work — of the ~40 commits since 17:07, the largest units are B1
(`dispatch_guard`), B2 (`champions` ratchet), B3 (`coverage` union), the DIRTY
repair, and five armed-default firings. **The more instrument repairs we order,
the more re-buy events, and re-buy events are what ledger activity looks like.**
`GOAL.md` has the sentence for it one level up: *"A README saying 'Working' is
not learning."* Twelve green PASS events in six days, not one of them about Jack,
is a README.

Routed as **B2**: one counter, measure-and-report, gate nothing.

---

## RANK 3 — the class that has consumed more of this project than any other has no name, no count and no gate: the spec whose gate could not have discriminated, discovered after the run

`LG.12` was registered at 04:14 this morning, implemented, run, and settled
`FAIL` at 05:15. The builder's own post-run record is the finding:

> `MATCH_MIN` 0.90 needed `m ≥ ln(14/(3·(1/0.9−1))) = 3.74` — **twice the
> largest dominance the mechanism ever produces** (span `[1.383, 1.826]`, sd
> 0.080, twelve of sixteen grid points identical). **Computable from pool sizes
> with zero seeds. The mechanism was foreclosed before it ran.**

That is honest, self-caught, and routed with a lesson. **It is also the eleventh
row of the same shape in `REVIEW_QUEUE.md`, and nothing counts them.** Nine are
live right now:

```
t211-diayn-metric-cannot-separate-mi-from-noise   08-29  OPEN   DUE 09-16
aggregate-hides-worst-seed                        08-30  OPEN   DUE 09-18
sm03-heldout-split-saturated                      08-30  DISPOS DUE 09-15
sh02-null-saturation                              08-30  OPEN   DUE 09-13
dp04-lifespan-has-no-resolution                   08-30  OPEN   DUE 09-22
ba03-null-saturates-the-horizon                   08-31  OPEN   DUE 09-13
t306-matched-magnitude-noise-buys-coverage        08-31  OPEN   DUE 09-13
w100-honest-null-does-not-rescue-pile-a           09-06  OPEN   DUE 09-15
ub10-part1-premise-false-marginals-are-what-      09-13  OPEN   DUE 09-20
lg12-abstention-knob-has-no-resolution            09-13  OPEN   DUE 09-14
```

**11 of the 50 rows ever routed. 22%.** Plus `T2.11` and `SH.02`, parked on the
same finding, which have no ledger row at all and so are invisible to every
count in the repo.

### The cost, with its caveat stated

The ledger records **53.37 h** of run duration across all rows. **33.38 h of it
— 62.5% — settled `VOID`: "the run did not test the claim."** PASS work is
10.24 h (19.2%); FAIL work is 9.75 h (18.3%).

**The honest caveat: 71% of that VOID total is two rows** — `D1.0` at 17.62 h and
`T2.02` at 6.28 h. The distribution is not uniform and I will not claim it is.
The count is the more stable statement: **14 of 145 settled rows are `VOID`, and
`VOID` means the apparatus failed to measure, not that Jack failed to learn.**

### And an instrument steered the builder into the newest instance

`coverage`'s `cpu<10min` class named exactly one fillable spec this morning; the
builder correctly took the only fresh unit on the board and it was foreclosed by
algebra. **The recommender has no reachability check.** The builder's own lesson
today names the two free numbers that would have caught it — *required setting*
and *available range* — and the cheap symptom needs no algebra at all: *"print
the knob's frontier table; byte-identical consecutive grid points mean a dead
control."*

### Why this is not the Review's `gates-that-measure-something-other-than-what-they-say`

They are siblings and both matter. The Review's row (`ca09bd9`, DUE 09-20) is
*the gate computes a different quantity from the one it names*, and its
dangerous case is **PASSING** specs. **This class is the gate computing the right
quantity against a null or a knob that could never have separated the arms**, and
every instance is on a FAIL, VOID or PARKED spec. Neither is counted. The
Review's is dated; mine is not routed anywhere.

Routed as **B3**.

---

## The audit, section by section

### 1. Integrity of the ledger — NO FINDINGS (checked mechanically, not inherited)

All 108 PASS rows, this morning:

- **Commits resolve: 108/108.** Zero unknown, zero missing from git.
- **Dirty stamps on a PASS: 0.**
- **Controls declared: 108/108.** `control_metrics` present on **106**; the two
  exceptions are `T0.01` and `T0.10`, both carrying the literal string
  `"NONE, BY DECISION (52nd audit B5)"` in the spec's own `control` field.
- Implementations exist for every PASS.

Status distribution across 145 settled rows: **108 PASS / 23 FAIL / 14 VOID**.

### 2. Thresholds and controls, over 7 days — NO FINDINGS; the direction is *tightening*

Every named numeric constant that was **changed** (a `-` and a `+` on the same
name) in `registry.py`, `registry_expansion.py` and `experiments/tests/` in
7 days — the complete list:

| constant | change | direction |
|---|---|---|
| `N_PROPERTIES` | 15 → 16 → 17 | **tightening** |
| `MIN_LEARN_SIGMA` | 3.0 → 3.0, comment only | **unmoved** |
| `MIN_DISTRACTOR_EVAL` | 9 → 9, comment only | **unmoved** |

Everything else in the diff is an **addition** — new specs and new conjuncts,
not moved bars. The notable additions are tightenings that cost: `PLANNER_CALIB_MIN
= 1.0` (`1bd42dc`) indicts the teacher before the twin and **took `LG.03` to
VOID-FORECLOSED**; `TWIN_DRIFT_SIGMA = 3.0` and `A0_HEADROOM = 0.05` are new rig
gates; `EYE_RADIUS_R2_MIN` stayed at 0.80 while its referent was re-aimed.

**Zero thresholds moved in the loosening direction. Zero controls deleted or
weakened. No `_check` gained an `or`. No seed count reduced.**

**I looked hardest at `8a97fd9`, the DIRTY-classifier edit**, because an edit
that changes what a standing alarm *says* about a run is the ideal disguise for
softening it. It survives:

- `staleness_of` still emits the kind `DIRTY` in **all four** sub-states —
  `protocol.py:2673` replaces only the detail string, never the kind. Since
  `Ledger.unsatisfied`, `borrow_metrics` and `gate_precondition` filter on the
  kind, every refusal a dirty row earned yesterday it still earns.
- The staleness bill was **priced before the edit and paid immediately**:
  T0.17/T0.35/T0.33 re-bought PASS clean, and **`T0.27` was re-run and stayed
  honestly red at `live_violations = 3`** — an edit that left its own red number
  red is not a tidy-up.
- The counterfactuals include the direction that would *excuse* a dirty run
  (always-COMMITTED) and `run stale` now refuses to report at all if any
  sub-state cannot be exercised.

### 3. Drift from the goal — NO DRIFT in the work; the converse is the whole problem

| work, last 24 h | `GOAL.md` sentence |
|---|---|
| `dispatch_guard` (B1) | *"protects the honesty of watching what happens"* |
| `champions` unwinnable ratchet (B2) | *"ARCHITECTURE always contested"* (SYSTEM.md invariant) |
| `coverage` union of dead paths (B3) | *"every capability claimed only by an experiment that could have failed"* |
| DIRTY-classifier truth repair | same — a standing alarm that lied about its only live row |
| `LG.12` (register → run → FAIL) | *"he learns words the way every child does"* |
| `UB.10` part 1 → foreclosure | *"One brain, all senses in unison"* |
| `PL.02` attempt 2 → VOID, arm-attributed | *"PLASTIC ONLY — nothing inside him is frozen"* — the **sole** registered falsifier |

**The converse, and it has not moved in six days.** One-brain/unison: **1
passing spec of 27**. Sleep: **0 of 5**. Fast/slow: **0 of 8**. Curiosity: 2 of
12. Four constitutional commitments are CLAIM-DEAD — *smell* (owner-named
constitutional), *balance*, *"he builds a shelter"*, *"too cold kills him"* —
and `champions` reports four architectural seats nobody can ever win. The union
is **7 distinct commitments or seats with no live path**, and as of `9da23c6`
(B3, executed last night) that number is finally printed in one place. Printing
it is progress; it did not shrink.

### 4. Is the builder alive and productive? — ALIVE, and the discipline is better than the scoreboard

**12 iterations 17:07 → 06:20, all `rc=0`**, ~40 commits, zero repeated
identical failures, no pause, no load abort. Two slots (19:07, 20:07) were
**pacing skips, correctly taken and correctly logged** — and the skip line now
names who drew the meter (*builder 18, desks 8, NOT THIS PROJECT 50 of 77*),
which is `D26`'s fired default working. It walked itself off Fable
(`week:Fable` 100%) onto Opus via `D14` option (b) on every slot.

**Meter at 06:07: `week:all models` 80%** against the 90% stop. Resets
09-14T04:59.

**PASS delta: 0** — see RANK 2 for what the PASS events actually were.

It executed **all four** of my predecessor's items inside eight hours (B1, B2,
B3, B4), deferred B5 to its authorised 09-20 date rather than touching a passing
certificate early, and **refused a seventh consecutive GPU dispatch** it was
formally entitled to make, with `W37` fresh at 30 h and both preconditions
discharged, because the row that authorises it is the Review's and is dated
09-14. Under `SYSTEM.md`'s standing clause that is a full night's work.

### 5. Compute honesty — CLEAN; the standing yield number is unchanged

- **`2026-W37` has no entry in `gpu_budget.json['weeks']` — 0.00 h of 30 spent.**
  Zero GPU dispatched in the window.
- `2026-W36` closed at **17.72 h of 30**, 12.28 h unspent, and the refusal to
  scrape attempt 3 (17.61 h) out of it was correct under pressure from two desks.
- **`gpu_hours_no_verdict` TOTAL 48.07 h**, of which **`D1.0`: 33.78 h / 2
  attempts / 0 verdicts**. Unchanged since 09-07 because nothing ran. This is
  yield, not waste in the dishonest sense — every charged job maps to an attempt
  with an honest recorded result.
- **`gpu_budget.json['projections']` is still `null`** — correct: the R3 receipt
  ships, and nothing has been dispatched through it yet. Worth noting so the
  first non-null value is read as the guard working, not as a surprise.
- CPU healthy and billed per-spec.

### 6. Stuck decisions — CLEAN; one decides TODAY

`decisions --check` **EXIT 0**. **3 armed, 0 overdue, 0 `MEANS-ESCALATED`, 0
`UNDECLARED`, 0 `UNROUTED-OWNER-ASK`, 0 `VANISHED-OWNER-ASK`.** The five
defaults that were overdue on 09-12 were all fired between 17:09 and 17:25 with
the required wording and a reversal path each; I checked them against
`DECISIONS_RESOLVED.md` and found nothing quietly acted on.

- **`D25` — `decide_by` is TODAY.** Costs 0 specs.
- `D19` — `decide_by` 2026-09-14, **costs 3 specs** (`HR.1`–`HR.4`), and is why
  `coverage`'s `cpu<10min` class reads FILL-HELD.
- `D20` — `decide_by` 2026-09-18, costs 0.

**I have no `UNDECLARED` entry to arm, because there are none (0/10).** The
standing instruction is satisfied vacuously and I say so rather than
manufacturing one — second audit running.

### 7. Bakeoff hygiene — NO FINDINGS

No VOID treated as a verdict. `LG.12`'s FAIL is a true FAIL and not a rig
artefact: every rig gate green, the null **alive and beaten**
(`null_utter_rate` 1.0, `null_match` 0.044/0.083 against a 0.35 bar), and the
tuning rule for the abstention margin was **pre-registered in `bd4cb61` before
any number existed**, leave-one-seed-out, which is the correct guard against
making the utterance floor true by construction. **No code was changed in
`LG.12` after the adverse verdict** — the builder named that explicitly as
amending-after-the-fact and refused it. `PL.02`'s VOID is arm-attributed to
`FROZEN`, the registered null. No winner chosen inside a noise margin.

### 8. The honest summary

**Not closer. Sixth day.**

Jack cannot do one thing today that he could not do on Monday. `demonstrated`
has read 108 for six days, and the twelve PASS events underneath that flat line
are six Tier-0 certificates being bought back from our own edits. Four of his
owner's constitutional commitments have no living claim. One passing spec of
twenty-seven for the unified brain that is the entire thesis. Zero of five for
sleep, zero of eight for fast/slow.

What did improve is real and I will not undersell it: the project can now refuse
a GPU dispatch in code rather than in prose, count a class of permanently dead
architecture, print in one number how much of `GOAL.md` has no live path, and
tell the truth about a dirty ledger row instead of a plausible inference about
one. Four instruments got better in eight hours. **That is exactly what
`SYSTEM.md` says counts as a full night's work, and it is exactly what I am now
worried about.**

Because look at what my own two findings have in common. **A guard was ordered
and built at the place the order named rather than the place the spend happens.
A week of guard-building produced a ledger whose every green mark is a guard's
own certificate being re-bought.** We have become extremely good at the
meta-level — at instruments that watch instruments, ratchets on classes, counters
on counters — and the meta-level has started to be the thing that generates its
own work. Eleven of fifty queue rows say some version of *the measurement could
not have decided anything*, and `VOID` — "the run did not test the claim" — is
the majority of every compute-hour this ledger has ever recorded.

That is not dishonesty and it is not drift; every one of those rows is a true
statement, honestly found, correctly filed. It is something narrower and worse:
**this project has spent four months learning to measure whether it is measuring
honestly, and the thing it is supposed to be measuring has not moved since
Monday.** The ladder has run out of rungs that can be climbed without a redesign,
the redesigns are queued behind a desk whose own drain reads **UNBOUNDED**, and
**14 of those rows are due today against a measured capacity of 6** — on the
Sunday FULL run that `D25` exists because it dies at the wall.

The builder is not the constraint. It had its second-best night of the week and
refused the one expensive thing it was allowed to do. The constraint is that the
only work left on the board is redesign, and redesign is the one thing neither
desk has capacity to dispose of.

---

## FOR THE BUILDER

**B1 — RANK 1. Move the pre-flight from the wrapper to the act, and do it before
any GPU dispatch this week.** Your guard is good; it is in the wrong place.
`run_spec` (`run.py:2665`) gates **CPU** children with `gate_cpu_child` and has
no `else`, so `$PY -m experiments.run <GPU-SPEC>` spends with no budget,
authorisation or projection check. `launch_detached.sh` is the same hole with a
CPU-day admit in front of it, **and `D1.0`'s own pilot went out through it**
(`LOOP_JOURNAL.md:9154`, 2026-09-01).

1. **Put the refusal where the spend is.** `dispatch_guard.preflight` is already
   injectable and already returns `(ok, lines)` — call it from `run_spec` for any
   spec whose `budget` is a GPU class, with the same UNRECORDED-refusal idiom the
   CPU gate uses (`Status.ERROR`, `"REFUSED before start: ..."`), so a refusal
   never supersedes a real result.
2. **The projection has to travel.** `dispatch.sh` should export it (an env var
   in the `JACK_*` idiom) and `run_spec` should refuse a GPU spec that arrives
   without one — that makes R3 the enforcement point rather than the CLI, and it
   closes path 3 with the same branch that closes path 2.
3. **Decide and write down what a *reattach* does on this path.** `dispatch.sh`
   exempts `JACK_REUSE_KERNEL` loudly and for a good reason; `run_spec` must make
   the same carve-out or it will refuse the recovery path, which is the failure
   `gpu.submit` already has a scar for.
4. **Price it first, as you have been.** `experiments/gpu.py` is declared in
   `T0.12`'s `IMPL_DEPS`; `run.py` is declared in none that I could find, but
   check with `grep -l` before you touch anything and state the bill in the
   commit. If the cheapest correct site turns out to be `gpu.py`, a `T0.12`
   re-buy is a price, not a blocker — say so out loud rather than routing around
   it.

**Red-first, as always: show the refusal firing against `$PY -m experiments.run
D1.0` on a constructed budget BEFORE you show it clearing.** Your existing ten
cases are all wrapper-entry; none of them enters the way the loop actually runs a
spec.

**This does not decide the `D1.0` dispatch question.** That is
`d10-successor-rerun-under-adopted-gate`, DUE 09-14, and it is the Review's. Do
not pre-empt it. You have refused seven times; refuse an eighth.

**B2 — RANK 2. One counter, measure-and-report, gate nothing.** Add to
`run status` a trailing-7-day split of ledger settle events into:
(a) **first-ever verdict** for that spec, (b) **re-buy** — same spec, same
resulting status, no change, (c) **status change**. Name the
**instrument-coupled** subset explicitly, because it is computable exactly and
is not a judgement call: resolve `IMPL_DEPS` and flag any spec declaring an
instrument file. The live map, resolved this morning:

```
experiments/protocol.py      -> T0.17, T0.27, T0.33, T0.35
experiments/cpu_budget.py    -> T0.33, T0.34
experiments/coverage.py      -> T0.21
experiments/champions.py     -> T0.29
experiments/review_queue.py  -> T0.31
experiments/gpu.py           -> T0.12
experiments/run.py           -> (none — and B1 will not change that)
```

That is **44% of 7-day settle events and 100% of 6-day PASS events.** Note
`run.py` is declared by no spec, which is why B1's cheapest correct site costs
nothing in staleness — state that in the commit rather than leaving it inferred.

**Do not gate on it and do not ratchet it** — a cap on re-buys would be a cap on
honesty, which is the opposite of the repair. The point is that a reader can
currently see *"12 PASS events"* and cannot see that none of them is about Jack.
`SYSTEM.md` already licenses instrument work as a full night; this makes the
licence legible instead of invisible.

**B3 — RANK 3. Two free numbers at registration, and one table after the run.**
Your own lesson this morning already specifies it; make it structural rather than
a lesson:

1. **A pre-registration reachability statement** on any spec with a tuned knob or
   a threshold scored against a null: the **required setting** for the bar to be
   clearable, and the **reachable range** the mechanism can actually produce. Both
   are usually computable with zero seeds — `LG.12` needed `m ≥ 3.74` against a
   span of `[1.383, 1.826]`, from pool sizes alone.
2. **The cheap symptom needs no algebra**: print the knob's frontier table.
   Byte-identical consecutive grid points mean a dead control. Twelve of
   `LG.12`'s sixteen were identical.
3. **Name the class and join the rows.** Eleven of fifty routed rows are this
   shape (listed in RANK 3 above); `T2.11` and `SH.02` are parked on it with no
   ledger row at all. They are currently eleven unrelated incidents with eleven
   separate dates. **One name, so the twelfth instance is recognised as a
   recurrence instead of discovered again.**

Where it goes is yours to judge — `coverage` is the tool that steered you into
`LG.12` this morning by naming `cpu<10min` as the only fillable class, so a
reachability caveat on that recommendation may be the higher-value half.

**B4 — carry to the Review, do not act on it.** `PROGRESS.md`'s `FOR THE BUILDER`
item 7 (*"`W1.04` still gains conjunct (c) before you register it"*) has been
carried unchanged for three mornings, and **you demonstrated on 09-12 that it is
already discharged** — `8d30cec`: conjunct (c) was the Review's own act on 09-10,
and `W1.04`'s registration is held by `w1-world-edit-window`, not by your
backlog. Flag it so today's FULL stops re-carrying it. **Do not edit that page;
it is the Review's.**

---

## FOR THE OWNER

**1. `D25` decides TODAY (2026-09-13); its default fires tomorrow.** If you do
nothing, **(iii) FIX THE SEAL** fires: `lib_seal.sh` learns to read a dying run's
own committed acts, so a Sunday FULL that committed its whole page stops being
banner-ed *"THIS IS A DRAFT, NOT A FINDING … UNVERIFIED"* identically to one that
committed nothing. **I have no objection and recommend letting it fire** — it is
the only legal default of the three, it moves no threshold, spends nothing, and
is monotone. The cost is stated honestly in the entry: Sunday FULLs keep exiting
`rc=124`. This is the second audit running that I have recommended it; nothing
has changed.

**2. `D19` decides 2026-09-14 and it costs 3 specs.** Its default is **NO FETCH**
— no corpora downloaded outside the repo — which leaves `HR.1`–`HR.4`, the whole
hearing-claim family, *"runnable-on-paper and blocked-on-disk"*, and leaves
`coverage`'s `cpu<10min` class empty. **That is the correct default and I am not
asking you to change it.** You should know that letting it fire keeps one of the
senses you named constitutional unbought, and that the honest alternative is a
spec amendment through the strengthen-only lane, not a default.

**3. NO-DECISION — what I found, and the fact that my own desk wrote both of
them.** Yesterday this desk found that the guards holding this project straight
were sentences rather than mechanisms, and ordered mechanisms. The builder built
them in eight hours, better than the order asked for. **I then found that the
biggest one was built where my order pointed rather than where the money is
spent** — the GPU guard refuses a dispatch launched through one shell script, and
the command that actually spends is `python -m experiments.run <SPEC>`, which
still has no budget, authorisation or projection check at all. The CPU day
budget, by contrast, *is* enforced in the runner. **The cheap resource is guarded
by a branch; the irreversible one by a convention.** Nothing has been overspent —
`W37` sits at 0.00 h of 30 and the builder has now declined seven consecutive
dispatches it was entitled to make. This needs nothing from you; it is routed as
B1. You are seeing it because *"it held because the organ was conscientious"* is
still the safety property we have, one day after we thought we had replaced it.

**4. NO-DECISION — the number I would most like you to see, because it is the
answer to "are we closer".** Every PASS recorded on this ledger in the last six
days — **all twelve** — is a Tier-0 harness certificate being re-bought because
an instrument edit that this desk or the Review ordered invalidated it. Not one
is about Jack. The four runs in that window that *were* about Jack returned
`VOID`, `VOID`, `VOID`, `FAIL`. Across the whole ledger, **62.5% of recorded
compute-hours settled `VOID` — "the run did not test the claim"** (with the
honest caveat that two rows, `D1.0` and `T2.02`, are 71% of that total). And
**eleven of the fifty rows ever routed to the review desk say some version of
"the measurement could not have decided anything."**

None of that is dishonesty — every row is a true statement, honestly found and
correctly filed, and `demonstrated` has stayed flat at 108 rather than being
flattered. It is something narrower: **we have got very good at measuring whether
we are measuring honestly, and the creature has not moved since Monday.**

**5. NO-DECISION — the constraint, unchanged and now dated.** Four constitutional
commitments are claim-dead (*smell*, *balance*, *"he builds a shelter"*, *"too
cold kills him"*); four architectural seats can never be won; the union is
**seven**, and as of last night that union is finally printed in one place. **The
repair for all seven is the same — successor specs — and that is redesign work.**
It sits behind a review desk whose drain reads **UNBOUNDED**, with **14 rows due
today against a measured capacity of 6**, on the Sunday FULL run that `D25`
exists because it dies at the wall. The builder is not the constraint and has not
been for a week.
