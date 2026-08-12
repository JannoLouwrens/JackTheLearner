# OVERSIGHT — 10th audit, 2026-08-12 12:37 UTC

## VERDICT: ON TRACK

First `ON TRACK` in this file's history, and it is not a softening of the
standard. In the 5 h 43 m since the 9th audit the loop produced **24 commits and
+5 PASS (67 → 72)** with **zero silent loosening**, and it spent them on the two
things this project was accused of neglecting: **`thermal` and `voice` — two of
GOAL.md's zero-pass commitments — now have certified sensors.** Both were chosen
by the *standing rule* (a commitment with no passing spec outranks fan-out), not
by `run blocked`. That is the coverage organ changing behaviour, which is what it
was built for.

`ON TRACK` does not mean nothing is wrong. Four findings follow. One is a live
defect in the compute meter that has been masked by an idempotency check, and
one is a carried finding now 24 commits old.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

**Commitments with specs but nothing passing: 12 → 9.** Cleared today:
`thermal (kills)` (PS.02), `voice` (VO.01), `generality` (T1.02's recovered
delivery). Still at zero: **touch/contact, balance, damage/nociception,
shelter/building, tool use, proprioception, sleep, social/other agents,
plasticity.**

Headline: **165 specs · 72 PASS · 1 FAIL · 1 VOID · 91 not implemented.
65 of 165 unreachable. 0/10 of the sensory inventory is LOAD-BEARING.**

---

## RANK 1 — the GPU meter computed a charge 15× too high, and an idempotency check hid it

`experiments/gpu_submissions.jsonl`, the 06:56 reattach that recovered T1.02:

```
attempt_id 1786517770416-2154524-kaggle   job_id jannolouwrens/jack-ladder-1786482462
charge_seconds 35330.34        duration_s 35331.38        ok true
```

**35 330 s is 9.81 hours. The same kernel's own metered window, recorded by the
original poll five hours earlier, was 2 361.88 s (0.656 h).** Same job_id, same
compute, two charge figures differing by **14.96×**.

### Cause, established from the code

`gpu.py:507-517` — on `JACK_REUSE_KERNEL`, `t0` is rewound to the submission
epoch embedded in the slug. `gpu.py:554` — `t_meter_open = t0 if reuse else
time.time()`. `gpu.py:566` — `billable_s = time.time() - t_meter_open`.

So a reattach bills **submission epoch → the moment the local process noticed**,
which includes every hour the kernel sat *already terminated* waiting for someone
to come back. Here the kernel finished around 21:47 on 08-11 and was reattached
at 06:56 — roughly **9.15 idle hours billed as GPU time.**

The rewind's stated rationale (*"Kaggle bills the kernel's full wall time whether
or not anyone local was watching"*) is correct about the **run** window and wrong
about the **reattach** window. The meter opens correctly and never closes.

### Why nothing caught it

`Budget.charge()` (`gpu.py:289`) returns early when the job_id is already in
`charged_jobs`. This kernel had already been billed 0.6561 h by the successful
21:47 poll, so the 9.81 h was computed, written to the evidence log, and
discarded. **`gpu_budget.json` is untouched: Kaggle W32 reads 12.6196 both before
and after the reattach. No hours were actually lost.**

But idempotency only fires when the original poll *already charged* — and the
entire reason `JACK_REUSE_KERNEL` exists is the case where it did **not**
(`gpu.py:500`: *"session restart SIGPIPEd T2.01 v3's waiter at ~80 min in"*). In
that scenario the job_id is absent, the early return does not fire, and the wrong
number lands in the budget. A kernel pushed overnight and reattached in the
morning would bill 9–12 h against a 30 h/week bucket for a 40-minute run.

**The consequence is not hypothetical arithmetic.** 17.3804 Kaggle-hours remain
and T2.01 needs 6.5. A spurious 9.81 h charge makes
`Budget.afford("kaggle", 6.5)` return **False** and silently withdraws the
ladder's #1 blocker from this week's budget.

### The guard is one layer too low

`t0_12_gpu_accounting.py:171-186` tests `submit_reattach_is_free` — it calls
`b.charge(..., 7200.0, job_id="kaggle/u/job-A")` **twice with the same amount**
and asserts the total did not move. That gates the *ledger of charges*. It does
not gate the *meter that produces the amount*, because the test supplies the
amount itself and stubs `JobResult` (`_STUB_KAGGLE`, line 221) rather than
exercising `run_on_kaggle`'s reattach path where the defect lives.

`docs/LESSONS.md:605-640` names this exact failure family — *"Charging on
failure, on retry, on reattach, and on queue time are four different ways to make
a meter read high"* — and its **GUARD** paragraph declares the reattach case
closed by `job_id`. It is not. `job_id` closed *billed twice*; it never closed
*billed wrong*. The lesson's own closing sentence, that nothing reconciles the
meter against Kaggle's own report of what a kernel ran for, is precisely the
missing check. **I have appended the refinement to LESSONS.md rather than a new
entry**, since the family is already named.

---

## RANK 2 — `coverage.py` still credits `curiosity` and `one brain / unison` to specs that do not test them (carried from the 8th and 9th audits, now 24 commits old)

Re-verified programmatically this afternoon, not carried on trust:

| commitment | specs | pass | the single passing spec is… |
|---|---|---|---|
| one brain / unison | 21 | 1 | **LC.01** — *"Every candidate core takes every sense into one latent, or it is not a candidate"* |
| curiosity | 12 | 1 | **PG.4** — *"Noisy-TV panel traps naive curiosity"* |

LC.01 is an **admission rule** about arms that have not been run. PG.4 certifies
that the **playground fixture** contains a working noisy-TV trap. Neither is
evidence that senses fuse or that Jack is curious. All 16 UB specs and all 6 CU
specs remain `NOT_RUN`; `run senses` agrees independently at **0/10
LOAD-BEARING**.

This was the 8th audit's RANK 1 and the 9th audit's RANK 3. It is now the oldest
untaken finding, and unlike the last two audits **the builder has had hours**: 12
iterations in 24 h, 11 of them `rc=0`. The remedy is unchanged and is repeated in
FOR THE BUILDER item 2. It is a reporting defect, not a science defect — but it
is the one number in this system that makes the thesis look covered when it is
not, which is the same shape as the 2026-08-10 miss that created the coverage
organ.

---

## RANK 3 — the two newest PASSes are `+dirty`, including the one that clears a GOAL.md commitment

`run status`:

```
T0.21  recorded PASS; ran from a modified tree at f239118
VO.01  recorded PASS; ran from a modified tree at 41d7b5b
```

Both are **true positives** — the code that ran was committed one commit later
(`21ec7cb` and `f239118` respectively), which is the loop's ordinary
record-then-commit rhythm. Neither is a false stamp of the kind repaired this
morning, and the ladder surfaced both without being asked.

It matters more than usual here because **VO.01 is the spec that just moved
`voice` off the zero-pass list.** GOAL.md's first passing claim for a
constitutional commitment should not be the one that cannot name the code that
produced it. The loop cleared T2.00, T0.25, T0.22 and PS.02 by clean-tree re-run
within the same day, so the pattern is established; these two simply have not had
their turn yet. **~77 s and ~2 s of CPU** clears both.

Related standing exposure, improved: **35 rows predate `impl_sha`** (was 39).
`run stale` reads **0 DIRTY, 0 CHANGED**.

---

## RANK 4 — `is_code_dirt` matches its exclusion list by suffix, not by path

`protocol.py:107`: `return not any(path.endswith(o) for o in NOT_CODE)`.

`NOT_CODE` holds bare basenames (`ledger.json`, `gpu_submissions.jsonl`,
`gpu_budget.json`) alongside repo-relative paths (`docs/LOOP_JOURNAL.md`). Any
future file whose path *ends with* one of those strings — `archive/ledger.json`,
`tools/my_gpu_budget.json` — is silently exempted from the `+dirty` stamp and
from the GPU push guard, which now shares the same predicate.

Severity today is **zero**: no such file exists, and I checked. I am reporting it
because the loop wrote a LESSON this morning (`0555761`, *"a named set is still
hand-maintained, and it was incomplete twice"*) about this precise list, and the
matching rule is the remaining soft edge in it. Anchor the comparison, do not
grow the list.

---

## Section-by-section

### 1. Integrity of the ledger — **CLEAN**

All 72 PASS rows checked programmatically against the registry and git, fresh:

| check | result |
|---|---|
| PASS row has a resolvable implementation via `_module_for` | **72/72** |
| `commit` still resolves in git | **72/72**, 0 missing |
| PASS row with no registered spec | **0** |
| ledger `seeds` count matches `spec.seeds` | **72/72** |
| spec declares a `control` | 70/72 |
| declares a control **and** `control_metrics` non-empty | **70/70** |
| `control_metrics` byte-identical to `metrics` (a fake control) | **0** |
| PASS stamped `+dirty` | **2** (RANK 3) |

The two without a declared control are **T0.01** (imports) and **T0.10** (Kaggle
round-trip) — both long-standing, both correctly reasoned deferrals.

**No PASS in this ledger is a claim without evidence.**

### 2. Thresholds and controls over time — **CLEAN, and two changes deserve naming as good practice**

98 commits touched `registry.py`, `registry_expansion.py` and `experiments/tests/`
in 7 days; 24 of them since the last audit. Mechanical scan re-run, not carried:

- **`_check` gaining an `or`: 0 hits.** Every `or` in the window is inside
  property-fixture assertions (T0.22, T0.24), never in a gate expression.
- **Seed reductions: 0.** The single apparent 3→1 in the raw diff is T1.08, where
  only `budget` changed (`CPU` → `GPU`); `seeds=1` predates the window and is
  documented at `registry.py:360-365` — the spec is *about* seed variance and
  varies seeds [0,1,2] internally, so spec-level `seeds=3` launched three
  identical jobs.
- **Specs removed from the registry: 0.**
- **Numeric constants moved: every move in the window is a tightening.**
  `N_CALIB` 60 → 400, `N_OCC` 160 → 600, `N_PROPERTIES` 12 → 13 → 14 → 15.

Two changes are of the shape that normally means loosening, and both survive
scrutiny:

**VO.01 (`efcc4fa` → `9357573` → `41d7b5b` → `f239118`), FAIL → PASS in one
hour, and it is clean.** The recovery gate `OCC_R2_MIN = 0.50` **never moved** —
verified against every revision in the window. What moved was sample size, and
the criterion for moving it was **pre-registered at 11:23 in
`docs/LOOP_JOURNAL.md:3636-3647`, 54 minutes before the 12:17:55 run that
passed**, in writing, including the sentence *"`OCC_R2_MIN` MUST NOT MOVE; if
duration still misses … the finding is real."* The diagnosis came from a
*control* (a clear-on-clear probe reading 0.63 against set A's 0.80 — a ridge
with 80 examples for 115 features), not from the score under test. `_check`
gained **`and`** clauses, not `or`: a new two-sided ±2 dB gate on achieved SIR.
This is the discipline the system asks for, executed under time pressure.

**PS.02 (`dcc24ec`) replaced a gate in the commit that made it pass, and
disclosed it.** `all_cold_died == 1.0` became `cold_censored <= CENSORED_MAX and
censored_explained == 1.0`. That is a loosening on one axis, and the commit says
so in its own words — *"replaced, not weakened quietly"*, citing law 4 — with the
reason: one life in 48 survived by **walking to the fire**, which the spec's own
"rises near heat" hypothesis requires to be possible. The compensating clause
`_fire_explains` is strictly stronger on the failure that matters (it separates a
rescue from a broken integrator, which `all_cold_died` could not) and is derived
from `thermal.py`'s law with no tunable threshold. **Correctly handled.** Two
residual notes for the builder, neither an integrity problem: `CENSORED_MAX = 2`
*is* a free knob chosen after seeing 1 survivor per 16, and censoring removes the
hardest-to-predict lives from the probe set, so the headline R² carries a small
upward bias that is not stated. See FOR THE BUILDER item 4.

**Nothing was loosened silently. Third audit running.**

### 3. Drift from the goal — **none**

Every unit of work since 06:47 traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| **PS.02** — thermal world, cold is felt before it kills | *"too cold kills him… the needs ARE the curriculum"* |
| **VO.01** — voice emitted, recovered by a listener through a wall | *"and VOICE — he must be able to make sound, not only receive it"* |
| **T1.02** recovery — generalisation vs shuffled control | *"Really learning, not appearing to learn"* |
| **T0.24, T0.25, T0.22, T0.21** — delivery contract, critic-as-baseline, borrowed constants, coverage parser | *"every capability claimed only by an experiment that could have failed"* — an answer that cannot survive delivery was never obtained |
| **T2.00** — trunk gradient health post-GAE-fix | path stage 2, and the gate in front of T2.01 |
| **dirty-stamp repair** (`eeafd2d`, `e25108b`, `71f7f03`) | SYSTEM.md's loop-on-itself: a stamp that lies blocks 47 specs |

**No item serves nothing.** Worth stating plainly: PS.02 and VO.01 were both
picked by the standing rule and **PS.02 frees zero specs** — the loop deliberately
spent an iteration on work no ranking would surface, because a GOAL.md commitment
had nothing behind it. That is the correction the 4th audit asked for, working.

**The converse.** Nine commitments still have no passing spec at all: touch,
balance, damage, shelter, tool use, proprioception, sleep, social, plasticity.
And the two the thesis rests on — **curiosity and unison — are credited a PASS
each that does not test them** (RANK 2). **0 of 10 senses are load-bearing.**

### 4. Is the builder alive and productive? — **yes, and this is its best day**

Window 2026-08-11T12:37 → 2026-08-12T12:37:

| | |
|---|---|
| iteration starts | **12** |
| `rc=0` | **11** |
| `rc=1` | **1** (max turns 120, 20:07 — still earned a PASS) |
| **PASS delta** | **65 → 72 (+7)** |
| since the 9th audit (5 h 43 m) | **+5**, 24 commits, 6 iterations |
| dead time | **9 h 44 m** owner pause (21:03 → 06:47) |
| working tree | clean; `origin/main..HEAD` empty — **all pushed** |

- **The usage grant expired at 12:00 and nothing broke.** The 12:07 iteration ran
  with no `RESUMED BY OWNER` line because weekly usage fell **below 90%** at the
  weekly reset — `usage_gate` (`scripts/lib_usage.sh:27`) returns early on
  `pct < 90` without consulting the override. I checked the expiry branch fails
  **closed**: at the next 90% crossing it will log `owner resume EXPIRED`, delete
  `.usage-resumed`, and stop. Correct behaviour; D5's hard deadline passed
  without incident (§6).
- **Fable is still unusable** — 5 consecutive iterations logged `OUT OF CREDITS on
  fable — falling back to opus`. The fallback works; model selection is
  decorative and has been for three audits.
- Not stalled, not thrashing, no repeated identical failures.

**One thing the builder did right that deserves recording.** At 08:16 it set out
to land `deps_sha`, discovered that adding any field to `Result` would
`TypeError` inside **T2.01's 6.5-GPU-hour poll** (running since 07:24 on
pre-`71f7f03` code), and landed the forward-compatibility prerequisite
(`Result.from_row`) instead — then held `deps_sha` back across three further
iterations while the poll stayed alive. **I verified the interlock empirically:**
the field set of `Result` at `08444b2` and at `HEAD` is identical, and no key in
any current ledger row is outside it. `unknown_keys` is deliberately not a
dataclass field, so `asdict` cannot write it back. **T2.01's in-flight job is
schema-safe.** The builder gave up its planned work to protect a run it could not
see, and was right.

### 5. Compute honesty — **the accounting is honest; the meter that feeds it is not (RANK 1)**

| | |
|---|---|
| Kaggle W32 | 12.6196 used / 30 — **17.3804 h remain**, bucket closes Sun 2026-08-16 |
| Colab W32 | **0.5513 productive**, 0.9914 failed |
| GPU spent since the 9th audit | **0.5498 h** (Colab, T1.08) |
| GPU hours with **no** ledger entry to show | **0** |
| unaccounted hours | **0** |
| in flight | **T2.01**, submitted 07:24, `est_hours=6.5`, timeout 8.9 h — **5 h 13 m elapsed**, process alive |

**This week's productive GPU is 1.2074 h and it bought two PASSes** (T1.02 via the
recovered Kaggle kernel, T1.08 via Colab). The 0.9914 failed Colab hour is
yesterday's RANK 1, already reported, and its payload was recovered at zero extra
cost — the re-delivery the 9th audit recommended was done and cost **0.0 GPU
hours**. That is the finding closed correctly.

Against that, RANK 1: the meter computed a 9.81 h charge for 0.656 h of compute
and only the idempotency key stopped it reaching the budget file.

### 6. Stuck decisions

**D5's hard deadline (2026-08-12T12:00 UTC) has passed, and it passed
harmlessly** — the weekly usage reset dropped consumption under 90% before the
grant lapsed, so the loop is running unthrottled and the 12:00 expiry was never
tested. The standing-policy question is **still open but no longer urgent**: the
next time weekly usage crosses 90%, `usage_gate` will find the expired
`.usage-resumed`, delete it, and stop every agent until the owner acts. I have
appended this to `DECISIONS_NEEDED.md` with the mechanism and the numbers.
Closing it is the owner's, not mine.

Nothing else is blocked that a bakeoff could have settled, and **no owner
decision was quietly acted on** — I checked the day's 24 commits against the open
D-entries specifically.

### 7. Bakeoff hygiene — **CLEAN**

`DECISIONS_RESOLVED.md` holds the same 2 decisions, both from PS.01, both
re-checked this audit:

- **PS.01/J → VOID**, correctly — three arms below the 3.0σ learning gate, with
  the reasoning recorded verbatim. A VOID was not treated as a verdict.
- **PS.01/J2 → WINNER `impact_speed`** — 2.66σ over the runner-up, 10.32σ over
  the null, all 11 gate-eliminated arms named, and the `screen` gate mode
  justified in writing (the arms are deterministic reductions of identical cached
  rollouts, so there is no training that could have failed). Outside the noise
  margin.

No decision was made without a learning gate. SYSTEM.md's third law has still
been exercised on exactly one real question.

### 8. The honest summary — are we closer to a curious humanoid?

**Yes — today, for the first time in the four audits I can compare against, by
something other than arithmetic on green ticks.**

The distinction the 9th audit drew was between *a longer list of green ticks* and
*a creature*. Here is the test applied honestly. Of the +5 PASSes, three
(T1.02, T0.24, T0.25, T2.00 re-deliveries and harness) are the machine
maintaining itself — real work, but not Jack. **Two are not.** Yesterday Jack's
world had no temperature in it at all; today cold exists, it kills him in
5.9–34.9 s, and the approach of death is **legible from his own sensory vector at
R² 0.617 while a probe with the thermal channel deleted reads −0.138.** Yesterday
he could not make a sound; today he emits one, and a listener **behind a wall**
recovers its pitch at R² 0.814 and its duration at 0.599 at a difficulty that is
declared rather than inherited. Two of the ten senses the owner called
constitutional went from paper to instrumented in one day, and both were chosen
*because* they were empty, by a rule this file's predecessors argued for.

That is the ladder-and-apple direction. It is not the ladder yet.

**What has not moved, and I will not let the good day obscure it.** Curiosity has
12 specs and **not one has ever run**. The unified brain has 21 specs and **not
one has ever run**. **Zero of ten senses are load-bearing** — no sense in this
system has yet been shown to change anything Jack *does*, which is the whole
content of "in unison". T2.01 has been FAIL for days, blocks 26 specs including
every curiosity spec and every Tier-5 and Tier-6 claim, and its 6.5-hour job has
been in flight for 5 h 13 m — **that job's result is the most consequential thing
in this project right now**, and it is the next iteration's first read.

And the instrument keeps earning trust. I went looking across 98 commits for a
threshold moved the easy way and found sample sizes raised, gates frozen in
advance and in writing, an `and` where an `or` would have been cheaper, and a
loosening that announced itself in its own commit message. 72 of 72 PASSes have
implementations, resolvable commits, matching seed counts and real controls. The
builder surrendered its planned work to protect a GPU run it could not observe,
and was correct to. **The ledger is worth believing.**

What it still is not being asked to measure is the thesis.

---

## FOR THE BUILDER

Ranked.

1. **Fix the reattach meter, then gate the meter and not the ledger of charges.**
   In `run_on_kaggle`, `t_meter_open = t0` on reuse makes `billable_s` run to
   `time.time()` — the moment the *local process noticed* — so a reattach bills
   every idle hour between the kernel terminating and someone coming back.
   Measured: **35 330 s charged for a kernel whose own metered window was
   2 361.88 s**, 14.96×. Close the window at the kernel's terminal time, from
   Kaggle's own report of what it ran for, not from the local clock. Then extend
   **T0.12** past `submit_reattach_is_free`: that property calls `charge()` twice
   *with the same amount you supplied* and asserts the total held, so it can
   never see a wrong amount. The new property must drive `run_on_kaggle`'s reuse
   path (or a seam over it) with a kernel that finished long before the reattach,
   and require `billable_s` to track the **kernel's** window — with the current
   rewind kept executable as the control that fails it. **The idempotency key is
   not the fix**: it only fires when the original poll already charged, and the
   whole purpose of `JACK_REUSE_KERNEL` is the case where it did not.

2. **Give `COVERS:` a kind, and stop crediting `curiosity` and `unison` to specs
   that do not test them.** Carried verbatim from the 8th and 9th audits; you now
   have had the hours. `(claim)` vs `(fixture)` / `(rule)` / `(sensor)`; count
   only `claim` in `n_pass` and report the rest separately. Re-declare **PG.4 as
   `(fixture)`** and **LC.01 as `(rule)`**. Expected effect:
   commitments-with-nothing-passing goes **9 → 11**, and `curiosity` and
   `one brain / unison` correctly read zero. Add the property to T0.21 in both
   directions — a `(claim)` must count and a `(fixture)` must not.

3. **Clear the two dirty stamps.** `VO.01` (~77 s) and `T0.21` (~2 s), both from
   a clean tree. VO.01 especially: it is the first passing spec for a
   constitutional commitment since the standing rule was written, and it should
   be able to name the code that produced it.

4. **Two residual notes on PS.02, neither urgent.** (a) `CENSORED_MAX = 2` is a
   free knob chosen after observing one survivor in sixteen — `_fire_explains`
   has no tunable threshold but the count around it does; derive it or report the
   observed censoring rate beside it so a drift upward is visible. (b) Censoring
   removes the lives *hardest to predict* from the probe set, which biases the
   headline R² upward by an unstated amount. State the direction in the docstring,
   or report R² on the uncensored-only subset of a fully-lethal world as the
   comparison. The gate is not in question; the number's provenance is.

5. **Anchor `is_code_dirt`'s comparison.** `path.endswith(o)` over a list mixing
   bare basenames with repo-relative paths means any future `*/ledger.json` or
   `*_gpu_budget.json` is silently exempt from both the `+dirty` stamp and the GPU
   push guard, which now share the predicate. Compare full repo-relative paths.
   Zero exposure today — this is closing the soft edge on the list your own
   `0555761` lesson was written about.

6. **UB.9 remains the cheapest route into the unison hole** — `run blocked` ranks
   it 3rd (frees 4, blocks 7), *"Heard, not seen: the task that is impossible
   without fusion"*. Deferred seven consecutive iterations; the oldest untaken
   piece of science in the ladder.

**And when T2.01 lands, read it before anything else.** It has been in flight
5 h 13 m of a 6.5-hour estimate, it frees 26 specs, and every curiosity and
Tier-5/Tier-6 claim sits behind it. Its process is schema-safe — I verified the
`Result` field set is unchanged since `08444b2` — so `deps_sha` stays blocked
until it records, exactly as you left it.

---

## FOR THE OWNER

**1. The best day this project has had, and the ledger still checks out.**
+7 PASS in 24 hours, 24 commits since 06:54, zero thresholds loosened across 98
commits in 7 days, 72 of 72 passing claims with real implementations and real
controls. More to the point: **two of the ten senses you called constitutional
went from paper to instrumented today.** Cold now exists in Jack's world and
kills him, and he can be shown to feel it coming (R² 0.617, against −0.138 when
the thermal channel is deleted). He can now make a sound, and a listener behind a
wall can recover its pitch. Both were picked *because* they had nothing behind
them, by the rule the coverage audit put in place — one of them freed no other
work at all, which is the point.

**2. D5's deadline passed and cost nothing.** The 12:00 UTC grant expiry was
never tested: weekly usage reset below 90% first, so the loop is running
unthrottled. No action is needed today. The standing question is still open for
the next time usage reaches 90% — at that moment every agent stops until you
resume them. Recorded in `DECISIONS_NEEDED.md` with the mechanism.

**3. One thing was quietly broken and I found it: the GPU meter, not the
science.** Recovering yesterday's lost kernel computed a charge of **9.81 hours
for 39 minutes of actual compute** — 15× too high. It never reached the budget
file, because a duplicate-billing check caught it first. But that check only
works when the job was already paid for once, and the recovery mechanism exists
precisely for the case where it was not. With 17.38 free Kaggle-hours left and
T2.01 needing 6.5, one wrong charge of that size would have made the ladder's
biggest blocker unaffordable for the week. Fix specified; no hours were lost.

**4. What today did not change.** Curiosity has 12 specs and **none has ever
run**. The unified brain has 21 specs and **none has ever run**. Zero of ten
senses are yet *load-bearing* — we have shown Jack can sense things, not that any
sense changes what he does, and "all senses in unison" is the claim that needs
the second. Both of those sit behind **T2.01**, which has been computing on
Kaggle for five hours as I write. That result is the most important thing in this
project right now.
