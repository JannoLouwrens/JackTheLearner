# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **FULL** (Part 1, Part 2, Part 2.5, the anatomy audit and the
> completeness audit all ran).

**2026-09-13 06:37–07:0x UTC — FULL, the sixth Sunday sitting.** Window: the
week 2026-09-06 → 2026-09-13.

*The one sentence: **the blackout ended by itself and the builder cleared every
board we had, and then Part 2 opened the oldest certificate on the ladder and
found that for thirty-six days it had been certifying the wrong thing.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `d44d21a`, `35ecac2`,
> `e9c1b68`, `470088d`, `91eb539`, `66b2951`, `83132c9`, and the
> `UNREACHABLE_BASELINE` raise inside `470088d`'s predecessor.

---

## The numbers

| | today | last Sunday (09-06) |
|---|---|---|
| demonstrated / registry | **107 / 246** | 104 / 242 |
| pass rate | **43.5%** | 43.0% |
| net demonstrated, week | **+3** (104→107, and it hides today's −1) | +2 |
| distinct specs PASSing in the week | **47** | 92 |
| PASS events in the week | **102** | 193 |
| rework rate (attempt > 1) | 78.6% | 75.0% |
| VOID / FAIL / BLOCKED | 14 / 23 / **1** | 14 / 23 / 0 |
| unreachable (shrink-only) | **94** of 246 — baseline raised today, by me | 93 |
| settlements in 24 h | **17** | 0 |
| commits in 24 h | **84** | 11 |
| `week:all models` | **81%** | 75% |
| GPU, live week `2026-W37` | **0.00 h of 30 — opened today** | W36, 17.72 of 30 |
| queue violations | 0 | 0 |
| live queue rows | 41 (three routed today, all mine) | 38 |

**The week in one comparison.** 47 distinct specs passed this week against 92
last week — almost exactly half — and the reason is not quality: the builder
was **switched off for 4.3 of the seven days** by the shared usage meter, and
then produced ~30 commits and 17 settlements in the single night of 09-12→13.
Read the halving as an outage, not as a slowdown.

**Goodhart check: the rate ROSE, 43.0% → 43.5%, against a registry that grew
242 → 246.** The runner outran the ladder this week. That is the honest
direction — but note the composition before taking any comfort from it: of the
week's net **+3**, today's Part 2 took one back, and it took the *oldest* one.

---

## Part 1 — the state of progress

### The frontier, recomputed and not quoted

`T2.01` (*Locomotion beats a random policy*, FAIL) **blocks 38 specs
transitively — four times the next largest** (`T4.04`, `T3.02`, `LT.01` at 9
each). That has been the largest single unblock in the project for five weeks
and it has not moved. What HAS changed, overnight and for the first time: its
repair path's two-step precondition is **satisfied**. `D1.0`'s twin-spread
probe ran and its successor gate is committed in a non-dispatch commit
(`7cb00ea`), which is exactly the stamp this desk has demanded since 09-08, and
`2026-W37` opened this morning with a full 30 GPU-hours. **The largest unblock
in the project is now one dispatch away from moving, and it was not last
Sunday.**

Effort-vs-goal: the week's commits served the *instrument* heavily again — four
`91st audit` items, a dispatch-guard hardening, a DIRTY-classifier repair, four
certificate re-buys — but the night also produced `LG.12` registered,
implemented, run and routed inside two hours, `PL.02` attempts 1 and 2
harvested and arm-attributed, and `SM.03`'s F2 probe answering a question with
a third branch its ruling did not have. That last one is the shape worth
naming: **the builder returned better than its guardrails**, which is the 91st
audit's own finding and I agree with it.

### The thing that did not move at all

**`GENERALITY.md`: 14 barriers named, 4 registered (`GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09`), 0 RUN, 0 PASS.** Recomputed today; byte-identical to the
09-06 reading. All four registered barriers are reachable on paper and none has
been dispatched in the week. Generality is the project's stated destination and
it received zero seconds of compute this week.

---

## Part 2 — the test re-examination

Twelve PASSing specs sampled, oldest-passed and least-recently-reconsidered
first: `T6.03`, `T0.19`, `T1.02`, `T1.08`, `T1.07`, `T2.06`, `T2.03`, `T2.04`,
`TA.02`, `T3.01`, `T2.19`, `T2.09`. The aim was the row this desk routed to
itself on 09-12 — *gates that measure something other than what they say* — and
it was aimed at PASSES on purpose, because all three prior instances were found
on VOID or FAIL specs where somebody was already hunting for a reason.

### STRENGTHENED — implemented, verified, committed

**`T6.03` — Cross-session persistence (`d44d21a`).** PASS since 2026-08-08,
attempt 1, the oldest certificate on the board, never once reconsidered. It is
the row this project cites for GOAL.md's *"What he learned yesterday — about the
world and about his owner — persists on disk."*

> **All fourteen of its conjuncts were true of a session-1 brain that never
> took an optimiser step.** `weights_match` therefore certified that
> `torch.save`/`torch.load` round-trips a tensor — which is `T0.03`'s claim,
> one tier lower and 300 lines cheaper — and the word *learned* in the sentence
> the spec answers had no referent anywhere in the file.

Nothing was broken. Every gate was sound. The spec was simply not about what it
was cited for. **ADDED** (strictly; no existing conjunct or bar touched, so the
passing set can only shrink): `train_loss_fell` and `train_moved_weights` —
session 1 now trains 24 steps on a fixed rank-8 pool before saving, so there is
something learned to persist and `weights_match` compares a TRAINED digest; and
`probe_dev_postload` / `probe_dev_preload` / `probe_sep_ratio` — the restored
brain must reproduce session 1's held-out probe loss to ≤ 1e-4 with the RNG
matched, while the virgin brain sits ≥ 1e-3 away and ≥ 50× further off. **Bytes
are not behaviour**: those are the first conjuncts in the file that read the
restore as a FUNCTION.

`PROBE_TOL` is calibrated and the file says so, per the LG.12 lesson written at
05:22 this morning: its reachable range is [1.13e-6 float noise at bit-identical
weights, 7.82e-3 virgin signal], and 1e-4 sits ~88× above the floor and ~78×
below the signal. **My first pass set it at 1e-6, which is BELOW the noise floor
and un-clearable by construction — the exact `LG.03` defect found on 09-12,
caught this time before registration and only because the lesson was three hours
old.**

**The certificate is OWED, and the reason is the day's second finding.**

### THE FINDING — a PASS is never re-evaluated when a dependency dies

The re-run did not return PASS. It returned **BLOCKED —
`dependencies not satisfied: T2.10 (FAIL)`.**

`T2.10` fell on **2026-08-31** under the paraphrase conjunct this desk itself
ordered. `T6.03` declares `depends_on: [T2.10, T0.05]`. It went on rendering
`[PASS]` in `run status` **every hour for thirteen days.** Nothing was wrong
with the runner — it refuses a blocked spec correctly and it did. The gap is
that the **board reports a STORED status, and no organ re-evaluates a standing
PASS when a spec beneath it dies.** A certificate is a claim that could be
re-derived today; `T6.03`'s could not be, and the ladder said otherwise.

**Scoped honestly, because the temptation was to report the mechanism at sweep
scale:** over all 246 entries, exactly **one** PASS stood on a non-PASS declared
dependency at 06:37 (`T6.03`), and exactly one does now (`LF.02` ← `T6.03`, the
second-order casualty). This is a narrow class. The finding is the mechanism and
its silence, not a backlog. Routed as
`pass-certificates-are-not-re-evaluated-when-a-dependency-falls` (`35ecac2`,
DUE 09-16), asking for a `pass_on_dead_dependency` counter in `run status`,
shrink-only floored — the `fail_unowned` shape.

`demonstrated` therefore goes **108 → 107**, and `UNREACHABLE` **93 → 94**. I
raised that floor myself and signed the growth log in the first person, because
raising a floor to accommodate one's own act is exactly what that log exists to
make visible.

### PROPOSED, not implemented — both are GPU and the edit would strand them

Two further instances of the same class, hand-verified in the sample. Both are
GPU specs whose certificates would go stale on an un-re-run edit, so they go to
the builder as designs rather than as edits:

- **`T1.08` — the noise-floor spec measures the wrong thing.** Its docstring:
  *"the number this produces should be quoted whenever a later tier claims an
  improvement."* That number is `min_detectable_effect` = 0.005694. **No
  conjunct reads it.** The gate is `snr = effect/noise ≥ 3.0` and it read
  **86.09** — 28× clear — where `effect` is a property of the fixed synthetic
  rank-8 task, not of the noise floor. A measurement spec that FAILS when its
  own toy task has a small effect is a category error: the correct output of a
  noise-floor measurement with large noise is *"the noise floor is large."*
- **`T1.07` — "Not knife-edge on learning rate."** The knife-edge quantity is
  `spread_ratio` = **4.931**: the measured advantage swings 1.38→6.80 across the
  10× LR span. **No conjunct reads it.** The gate requires only that no LR be
  catastrophic (all three clear 1.15×). Stated meaning: robustness. Computed
  quantity: absence of collapse.

### And the mechanisable pattern, including the part that failed

All four instances — `PL.02`'s `r2_ua`, `LG.03`'s `own_hit`, `HR.5`'s
`position_only_acc`, and now `T1.08`/`T1.07` — share one footprint: **the run
measured the quantity that would have indicted it, and then did not look.** I
prototyped a probe for it (scratch, uncommitted, not an instrument) that parses
each `_check` for the metric names its conjuncts read and diffs them against the
ledger row's keys.

**It does not work and I will not dress it up.** It flags 104 of 107 PASS specs
and 1,423 metrics, because a metric summarised into a gated aggregate — seven
per-property booleans behind one `properties_failed`, as in `T0.19` — is read in
substance while unread by name. Twelve hand-checks, three real. A screen at that
rate is worse than nothing. Whether a summarisation-aware version is buildable
is a genuine open question, and it is why this went to the owner as **`D27`**
rather than into a commit.

---

## The anatomy audit — two seats added (`e9c1b68`)

29 seats against GOAL.md. Both additions are VACANT on arrival and both are
justified in `CHAMPIONS.md`.

- **Language routing** (what he says, and which task a command becomes). Owed
  here: the Review of 09-10 found the gap and routed it to this audit. The case
  sharpened overnight without anyone noticing — **`LG.12` was registered at
  04:14 today as *"LG.10's sibling and provably NOT its repair"*, which is a
  challenger declaration in everything but name.** Two competing mouth designs,
  both FAIL, arbitrated by nothing. ARENA `LG.10`, `LG.12`, `T2.15` — all red,
  so the ring is open and undefended.
- **Person model** (trust, attribution, whose advice proved true). GOAL.md gives
  it a section of its own. How a person is represented and how trust updates is
  a mechanism with real arms and the repo picked one by accident in
  `OwnerProfile.py`. ARENA `LG.02`, `SO.08`, `ME.9` — **all three PASS, which
  makes it the only seat any anatomy audit has created that is contestable on
  the day it is created.**

Naming `ME.9` as a challenger broke the KINDLESS-DISCHARGES ratchet 1 → 2.
Repaired the sanctioned way — declaring `ME.9`'s COVERS kind honestly in the
registry — not by narrowing the arena. `champions --check` EXIT 0.

---

## The completeness audit — against the external reference (`91eb539`)

`run senses` reports **10/10** of the sensory inventory spec'd, and it is right.
It audits the SENSORY half against GOAL.md's own list. **Nothing in this
repository audits the cognitive half against anything**, and that is where every
gap below lives. Body schema remains the one zero from 2026-08-09 that has never
moved.

**The finding this audit exists to produce: `emotion` is not a declared `COVERS`
domain at all.** 1,149 lines of `EmotionalState.py`, a `BY DEFAULT` seat, two
specs unchanged since the 08-09 scar named them — and `coverage` cannot report
on it and `run senses` cannot see it, **because GOAL.md's sensory sentence never
listed it.** An organ measuring against a stated standard, and the standard
omits the thing. Missing outright inside it: any spec that affect changes what he
LEARNS or REMEMBERS.

Also named, one at a time: **attention** 0 specs (UB.2/UB.8 are
architecture-attention, not the capability); **working memory** 1 spec, and it
tests that WM survives a restart, not that it holds or manipulates anything;
**imagination** 0 for the capability; **teaching** 0 — `GEN.02` is Jack as
STUDENT, and in 246 specs Jack never teaches, which GOAL.md's culture claim
needs; **tool use** 0 PASS with `GEN.05` unregistered; **symbols** 0.

And **one gap closed by accident**: `LG.12`, registered at 04:14 today, is this
project's first metacognition spec — *he speaks correctly or he is silent* is
knowing what you do not know. Recorded because a gap that closes unnoticed is as
invisible as one that opens unnoticed.

---

## Part 2.5 — steering maintenance

**1. Priorities — replaced in full (`66b2951`).** The 09-08 block's five items
are all spent and its ~120 lines of `pace_gate` forecasting now describe weather
that has passed; a 03:00 iteration would have planned against a dead GPU pot.
Superseded, not deleted. New order: `T2.10` first (CPU, ten minutes, and it
turned out this morning to gate two certificates), then `D25`'s armed default
which is due today and which fixes the instrument that will judge this very
page, then `D1.0` attempt 3 into W37 with `T2.01`'s 38 specs behind it. One new
prohibition, aimed at myself as much as the builder: **do not raise
`UNREACHABLE_BASELINE` to cover your own work** — it moved today by my act and
the growth log says so in the first person.

**2. Field watch — nothing to consume.** Last sweep 2026-09-07 (wk6, `3b68b7d`),
consumed in full by the 09-07 Review, unchanged byte for byte since. Next sweep
Monday 09-14.

**3. Seat staleness.** `Learning core` holds 3 trigger debts (`LC.07`
PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10` VOID); `World` still declares no
deciding run and no `TRIGGER:` while held BY VERDICT, the file's strongest
marking — that is now **ten days** carried and it is the oldest seat finding on
the board; `Fast/slow coupling` welded behind `LC.03`; `D1` VACANT with its
arena about to run. Two seats added today, above.

**4. Organ liveness — all four alive, verified against `/data/jack-logs` mtimes
and not against anyone's report.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:20 today | alive; ~30 commits and 17 settlements overnight |
| overseer | 6-hourly | 06:50 today (92nd, concurrent with this page) | alive |
| field watch | Mondays | 09-07, consumed | alive; next 09-14 |
| review | daily / Sun FULL | this run | alive |

---

## The honest paragraph

No numbers. We are closer, and the week's most important step is one the builder
took while nobody was awake to see it: it came back from a four-day outage and,
instead of doing the easy work waiting for it, it read the code underneath three
arguments and found that all three were being had one level too high. That is
the behaviour of something that has learned what this project is for. And then
this desk opened the oldest certificate on the ladder — the one that says he
remembers yesterday, which is as close to a claim about *him being someone* as
anything we own — and found that it had been proving a file round-trips, in a
brain that had never learned anything, since the day it was written. Nobody
cheated. Every gate was sound. The spec was cited for a sentence it was not
about, and it took thirty-six days and a Sunday sitting for one person to read
it. That is the drift, and it is the most concerning thing here: this project
has spent four months getting very good at asking whether a number cleared its
bar, and it has almost no machinery for asking what the number *is*. Four
instances in four days now. Every one found by a human reading code that a
machine had already called green. The encouraging half, and it is real: the
mistake I made while fixing it — setting a new tolerance below its own noise
floor — is the *identical* mistake I diagnosed in `LG.03` on Friday, and I
caught it in myself within the hour because the lesson had been written down
three hours earlier. The system is starting to teach its own organs faster than
they can repeat themselves. That is a thing a creature does.

---

## FOR THE BUILDER

Ordered. Items 1–3 are in `ladder_prompt.md` as well; 4–6 are new here.

1. **`T2.10` — CPU, ten minutes, and worth more today than yesterday.** It is
   runnable and FAIL, and it now gates **two certificates**: `T6.03` cannot be
   re-bought and `LF.02` is out of the reachable set behind it. Its repair
   returns `UNREACHABLE` to 93. The paraphrase conjunct is the bar and **the bar
   does not move.**
2. **`D25`'s armed default is due TODAY** — option (iii) FIX THE SEAL, BUY
   NOTHING. Required journal wording: *"the owner did not rule by <date>, so the
   pre-registered default fired."* Cheap, monotone, and it fixes the instrument
   that will judge this page.
3. **`D1.0` attempt 3 into W37.** The two-step precondition is satisfied for the
   first time (`7cb00ea`). 3.0σ does not move; each arm scores against its OWN
   untrained twin; random stays in as a reported floor; an unchanged
   re-dispatch is still forbidden. `T2.01`'s **38** transitively-blocked specs
   are what is behind it.
4. **`T1.08`'s gate is a category error — redesign it, and it is HARDER.**
   ADD, do not replace: (a) gate on `heldout_cv_pct` — the seed spread of the
   held-out metric itself — which is the quantity the spec exists to produce
   and which no conjunct reads; (b) require `min_detectable_effect` to be
   RECORDED AND CITED by at least one downstream entry, or the spec is
   producing a number for nobody. Keep `snr ≥ 3.0` untouched. This is strictly
   harder: `snr` read 86.09 against a 3.0 bar, so the existing conjunct is
   carrying no weight and the new ones do.
5. **`T1.07` must gate the quantity in its own title.** ADD
   `spread_ratio ≤ 6.0` as a conjunct — it read 4.931, so the bar binds at the
   observed value with modest headroom and cannot be cleared by the current run
   getting worse. Do not touch `MIN_BEAT_MEAN`. "Not knife-edge" is a claim
   about SPREAD and the spec has never gated spread.
6. **Do not re-run `T6.03` until `T2.10` is PASS.** It will only return BLOCKED
   and burn a slot. Its code is verified out-of-band on seed 0 and its
   certificate buys itself the moment `T2.10` is green.

---

## FOR THE OWNER

**1. `D27` — NEW, routed today (`83132c9`), `class: goal`, `decide_by`
2026-09-20.** *Part 2 re-examines about ten of 107 certificates a week, and
today the oldest one had been certifying the wrong thing for thirty-six days.
Keep hand-sampling, or buy a mechanical screen?* My recommendation is quoted
verbatim on the entry: **(i) BUILD THE SCREEN — and build it to report, not to
gate, until its false-positive rate is measured.** The entry carries my
prototype's failure honestly rather than burying it: it flags 104 of 107 PASS
specs, 3 of 12 hand-checks were real, and a screen that cries wolf at that rate
will be ignored within a week. Whether a summarisation-aware version can be
built is the open question, and it is the reason this is your call and not my
order.

**2. `D25` is due TODAY and its armed default should fire.** Already routed;
cited, not re-asked. It is the seal that has bannered four of the four Sunday
FULLs — including, in all likelihood, this one — as unverified drafts whether or
not they committed their work. I am the organ it defames and I still think (iii)
is right.

**3. NO-DECISION: the week's science, declared because it is mine to do.**
Two seats added, three queue rows routed (all three to my own docket, all three
dated off Sunday's pile), one certificate strengthened, one floor raised by my
own hand and signed as mine. The `demonstrated` count fell by one this morning
and it fell for a true reason: I said on 09-12 that the sweep's foreseeable cost
was a PASS having to be re-bought, and that **"it is the only foreseeable way
`demonstrated` goes *down* by an act of this desk."** It did, within twenty-four
hours, and I would rather you match that sentence to this number yourself than
find it later.

**4. NO-DECISION: the docket, announced rather than asked about.** Queue 0
violations before and after; 41 live rows. Sunday 09-13 carried **14 rows
against a measured capacity of 6** — I said for a week it would not clear, and
it did not; nothing new was dated onto it. Today's three routings went to 09-16,
09-20 and 09-21, taken from `review-queue`'s own `next_free_due` rather than
chosen by hand.

**5. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive, verified against log mtimes. The builder ended a 4.3-day outage by
working and cleared every dated item this desk had put in front of it.
`week:all models` 81%; `week:Fable` 100%, and the gate is all-models.
