# OVERSIGHT — 9th audit, 2026-08-12 06:54 UTC

## VERDICT: DRIFTING

**Same verdict as the 8th audit, for the same reason, and this time it is not
the builder's fault.** There are **zero commits since the 8th audit** — the loop
was paused by the owner at 21:03 and resumed at 06:47, 9 h 44 m later. Every
finding I filed last night is still open because nobody was permitted to work
on it.

What is new is one measurement: **T1.02's GPU job succeeded, cleared every gate
on all three seeds by 8–25×, and was thrown away by a seven-day-old un-migrated
caller.** 1.6475 GPU-hours bought an `ERROR` row. The payload is still on disk
and I have read it (RANK 1).

Integrity is intact. Sections 1, 2, 5 and 7 are clean and I checked them
independently rather than carrying last night's result.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

**12 of 23 commitments have specs and nothing passing** — unchanged from last
night, and two of the eleven that "have something passing" still do not really
(RANK 3).

Headline: **163 specs · 67 PASS · 2 FAIL · 1 ERROR · 1 VOID · 92 not
implemented. 67 of 163 unreachable. 0/10 of the sensory inventory is
LOAD-BEARING.**

---

## RANK 1 — a completed, gate-clearing GPU measurement was destroyed on delivery, and the result is still recoverable on disk

`experiments/ledger.json` (uncommitted, written 2026-08-11T21:47:10):

```
T1.02  status ERROR  duration_s 5940.64
       "ValueError: dictionary update sequence element #0 has length 3; 2 is required"
```

The Kaggle job it is reporting on **succeeded**:
`gpu_submissions.jsonl` → `jannolouwrens/jack-ladder-1786482462`, `ok: true`,
`charge_seconds: 2361.88` (0.6561 h). The colab arm that failed over to it
burned a further **0.9914 h**. Total **1.6475 receipted GPU-hours for an ERROR
row.**

### The chain, established by reading the code and the artifact

1. `t1_02_shuffled_control.py:129` writes the result to a hardcoded
   `/content/out.json`. The job wrapper defines `JACK_OUT` at its own line 27
   for exactly this reason — *"Colab VMs start in /content; Kaggle kernels must
   write to /kaggle/working or the file is never collected."* T1.02 does not
   use it. **On Kaggle the artifact was therefore never collected.**
2. `t1_02_shuffled_control.py:140` looked it up as
   `r.artifacts.get("/content/out.json")` — a **full-path** key. `gpu.py` keys
   artifacts **by basename on both backends**. That lookup could never hit.
3. The fallback `or next(iter(r.artifacts.values()), None)` then accepted the
   only file Kaggle did return: the console log,
   `/data/tmpf3q6yglx/out/jack-ladder-1786482462.log`.
4. That log is a JSON **array** of `{stream_name, time, data}` records, so
   `json.loads` succeeded and `_CACHE.update(<list of 3-key dicts>)` raised the
   ValueError. Element #0 has length 3 because a 3-key dict does.
5. The `RESULT `-line fallback written for precisely this case (lines 143–146)
   was unreachable — `path` was truthy — and would have failed anyway, because
   `run_on_kaggle` constructed its `JobResult` with `stdout=""`. **Kaggle has no
   stdout pipe, so that branch was dead code on the only backend that runs the
   long jobs.**

### The measurement survived, and it is a clean 3/3

I recovered the `RESULT` line from the log the job left behind. Against the
spec's own gates (`MIN_REFERENCE_GAIN=1.5`, `MIN_STRUCTURE_ADV=1.25`,
`MIN_BEATS_MEAN=1.10`):

| seed | reference_gain | heldout_structure_advantage | beats_mean_baseline |
|---|---|---|---|
| 0 | **8.00** | **22.51** | **11.81** |
| 1 | **7.85** | **17.35** | **9.03** |
| 2 | **8.47** | **24.68** | **13.61** |

Every gate cleared on every seed, by **5.2× to 19.7×** at the tightest margin.
The shuffled control behaved as required — heldout error 0.513–0.562, i.e.
*worse* than predicting the mean (0.280–0.295) — so the control did what a
control must. `reference_gain ≈ 8` is far above the 1.5 VOID floor, so this is
**not** the VOID outcome the last three handoffs warned about.

**I am not verdicting this and I have not touched the ledger.** It has to come
back through the real code path. But it means `generality` — one of the 12
zero-pass commitments — is one honest re-delivery away from its first PASS, and
it does not have to cost another 0.7 GPU-hours.

### The systemic finding, which is worse than the bug

`55a07f4` (**2026-08-05**, *"One job contract, actually: JACK_OUT and basename
artifact keys"*) fixed the organ and left T1.02 behind on **both** halves of the
new contract. Nine other GPU specs key by basename correctly
(`t201.json`, `t108.json`, `t109.json`, `t110.json`, `t107.json`, `t202.json`,
`probe_result.json` ×2, `job_result.json`). T1.02 was the sole survivor, it has
ERRORed on every attempt since (2026-08-08, 2026-08-11), and it burned GPU each
time.

`docs/LESSONS.md:2252` — written **yesterday** — is this lesson already:

> *"There was a third caller and it was not looked for… **grep for the retired
> rule's syntax, not for the new rule's callers.** … Prefer to encode that
> search as a property in the guard spec so the next survivor is found by the
> ladder rather than by an auditor."*

It was found by an auditor. **I am not appending a new lesson for this**, because
the correct one exists and the finding is that it was not applied.

**The builder is already on it, live.** As of 06:54 the working tree contains a
`result_json()` in `gpu.py` that refuses a full-path key outright, a
`_kaggle_log_streams()` that finally populates `JobResult.stdout` on Kaggle, a
`JACK_REUSE_KERNEL` order fix so a reattach cannot pay Colab for a finished
Kaggle job, and a new `experiments/tests/t0_24_result_survives_delivery.py`.
That is the right response and it is the guard-spec half the lesson asked for.
Reported here because it is not committed yet and because the recovered payload
below is the thing that saves the re-spend.

---

## RANK 2 — the loop has roughly five hours of permitted runtime left and the #1 blocker is still untouched

This is the 8th audit's RANK 2, now with a deadline attached.

| resource | state |
|---|---|
| weekly Claude usage | **95%** (log, 06:47), ceiling 100% |
| `.usage-resumed` grant | expires **2026-08-12T12:00 UTC — 5 h 6 m from now** |
| Kaggle W32 | 12.6196 used, **17.3804 h remain**, bucket closes Sun 2026-08-16 |
| T2.01 | `FAIL`, **frees 26 / blocks 36**, `est_hours=6.5`, `prefer="kaggle"`, affordable (`Budget.afford("kaggle", 6.5) → True`) |

`run blocked` is unchanged: T2.01 at frees 26, then LC.03 at 7 (3.7× smaller).
**Every curiosity spec (CU.1–CU.7, T2.08), every Tier-5 claim and every Tier-6
living-Jack spec sits behind T2.01.** Nothing has been submitted for it since the
v4 re-spec.

Both binding resources — permitted hours and free GPU-hours — expire without
being spent, and the GPU-hours can only be spent during permitted hours. The
loop cannot fix this; it is an owner decision (FOR THE OWNER §1).

---

## RANK 3 — `coverage.py`'s `n_pass` is still discharged by specs that do not test the commitment (carried, unfixed, 0 commits available)

Re-verified independently this morning, not carried on trust. The two
commitments GOAL.md calls the thesis are each discharged by exactly one PASS:

| commitment | specs | pass | the passing spec is… |
|---|---|---|---|
| one brain / unison | 21 | 1 | **LC.01** — *"Every candidate core takes every sense into one latent, or it is not a candidate"* |
| curiosity | 12 | 1 | **PG.4** — *"Noisy-TV panel traps naive curiosity"* |

LC.01 is an **admission rule** on arms that have not been run. PG.4 certifies
that the **playground fixture** contains a working noisy-TV trap. Neither is
evidence that senses fuse or that Jack is curious. All 16 UB specs and all 6 CU
specs are `NOT_RUN`; `run senses` agrees independently at **0/10 LOAD-BEARING**.

Unchanged from the 8th audit's RANK 1. The fix is specified there and repeated
below. **The builder is not at fault for the non-fix — it had zero permitted
hours between the two audits.**

---

## RANK 4 — `{"phase":"selftest"}` is still in the production evidence log, still ungated (carried)

`experiments/gpu_submissions.jsonl` line 2, committed in `b52c0eb`. No
`attempt_id`, no `backend`, no `ts`; `submissions()` parses and returns it as a
record. Severity genuinely low — nothing downstream reads the log for decisions
— but no spec gates the log's contents, and the new T0.24 in flight is the
natural place to add the property. Do not delete the line; gate it.

---

## RANK 5 — Colab is the unmetered backend, it is preferred by five science specs, and it is 0-for-1 this week

`Budget.remaining("colab")` returns **`inf`** by construction, so `afford()` can
never refuse a Colab job. In W32 Colab produced **0.0015 productive hours against
0.9914 failed hours** — one attempt, and it is the one that failed over into
RANK 1. Five specs still carry `prefer="colab"`: T1.02, T1.07, T1.08, T1.09,
T1.10. Four of those five already PASS, so live exposure is small and I am not
inflating it — but T1.02, the one that does not pass, is the one that has now
paid Colab twice for nothing.

To the meter's credit: failed hours **do** count against the ceiling
(`used_hours = productive + failed`, `gpu.py:258`). The accounting is honest;
it is the *ceiling itself* that does not exist for Colab.

---

## RANK 6 — 39 ledger rows still cannot be checked for staleness; pain and temperature still have no sensor

- `run status`: **39 entries predate `impl_sha`** (was 40 — one improved).
  `run stale` reads **0 DIRTY, 0 CHANGED**. Standing exposure, not a new fault.
- `run senses`: **`[ABSENT] pain (nociception)`** and **`[ABSENT] temperature
  (thermoception)`** — *"load-bearing: NO SPEC would prove it."* Both are named
  in GOAL.md's inventory and both are consequential to the survival world.

---

## Section-by-section

### 1. Integrity of the ledger — **CLEAN**

Checked all 67 PASS rows programmatically, from the ledger and the registry, not
from last night's report:

| check | result |
|---|---|
| PASS row has a resolvable implementation via `_module_for` | **67/67** |
| `commit` still resolves in git (`cat-file -e`) | **67/67**, 0 missing |
| PASS stamped `+dirty` | **0** |
| PASS row with no registered spec | **0** |
| ledger `seeds` count matches `spec.seeds` | **67/67** |
| spec declares a `control` | 65/67 |
| declares a control **and** `control_metrics` non-empty | **65/65** |
| `control_metrics` byte-identical to `metrics` (a fake control) | **0** |

The two without a declared control are **T0.01** (imports) and **T0.10** (Kaggle
round-trip). T0.10 remains a known, self-reported, correctly-reasoned deferral.

**No PASS in this ledger is a claim without evidence.**

### 2. Thresholds and controls over time — **CLEAN**

87 commits touched `registry.py`, `registry_expansion.py`, `experiments/tests/`
in 7 days; **0 of them since the last audit.** I re-ran the mechanical scan
rather than carrying the result:

- **`_check` gaining an `or`: 0 hits.**
- **Seeds: every `seeds=` edit in the window is an addition or an increase. Zero
  reductions.**
- **Controls: 5 `control=` lines disappear in the diff. I resolved each against
  the live registry — 3 are reflow noise (text still present verbatim), and the
  2 genuinely gone are both T0.12's, replaced by a strictly stronger version:**
  *"Two named broken meters"* → *"**Three** named broken mechanisms… a Budget
  whose weeks deliberately leak must FAIL isolation; the pre-2026-08-09
  `charge()` plus `submit()` loop must FAIL every billing property; and the
  pre-2026-08-11 dispatch loop, run against a HEALTHY meter, must FAIL every
  receipt property."* (`2e8d558`.)
- **Specs removed from the registry: 0.**

**Nothing was loosened. Recording this as a genuine pass for the second audit
running.**

### 3. Drift from the goal

**Nothing was built between the two audits — zero commits.** The last day's work
(15:57–20:38 on 08-11) was audited in full last night and every item traced to a
GOAL.md sentence; that assessment stands and I am not re-litigating it.

The work in flight *right now* (06:47→) is T1.02 delivery repair. It serves
*"Really learning, not appearing to learn… every capability claimed only by an
experiment that could have failed"* — an experiment whose result cannot survive
delivery has not been run. **Not drift.**

**The converse — what still has no passing spec at all:** unison, curiosity,
generality, plasticity, thermal, shelter, tool use, damage, social, sleep,
proprioception, balance, touch, voice. **12 of 23 commitments**, unchanged.

### 4. Is the builder alive and productive?

Window 2026-08-11T06:47 → 2026-08-12T06:47:

| | |
|---|---|
| iteration starts | **6** (5 completed + the one running now) |
| `rc=0` | **4** |
| `rc=1` | **1** (max turns 120, at 20:38 — still earned a PASS) |
| skipped (`previous iteration still running`) | 1 |
| **PASS delta** | **65 → 67 (+2)**: SM.01, T0.23 |
| dead time | **9 h 44 m paused** (21:03 → 06:47) + **9 h 10 m** of the earlier 90%-usage stop |
| actual permitted runtime in 24 h | **≈ 4 h 40 m** |

- **The pause was lifted correctly.** Both `.paused` and `.loop-paused` are gone
  as of 06:47; the deletions are the uncommitted `D` entries in `git status`.
- **Fable remains unusable** — 3 consecutive iterations logged `OUT OF CREDITS
  on fable — falling back to opus`. The fallback works; model selection is
  decorative.
- **Weekly usage is at 95% against a ceiling of 100% that lapses at 12:00.**
  On the last five iterations' burn rate that is roughly **five iterations of
  headroom**, and they run out at about the same time the grant does.

Not stalled, not thrashing, not repeating identical failures. **Rate-limited by
permission, not by capability.**

### 5. Compute honesty — the accounting is honest and it is recording waste correctly

| | |
|---|---|
| Kaggle W32 | 12.6196 used / 30 — **17.3804 h remain**, bucket closes Sun 2026-08-16 |
| Colab W32 | 0.0015 productive, **0.9914 failed** |
| `Budget.remaining("kaggle")` | **17.3804** — verified by calling it |
| `Budget.afford("kaggle", 6.5)` (T2.01) | **True** |
| unaccounted GPU hours | **0** |

Every hour is receipted, and the failed colab hour is correctly booked to
`colab_failed` and correctly counted against the ceiling. The W31 Kaggle overrun
(37.4554 vs 30.0) remains a known, already-instrumented scar.

**But the honesty question this section actually asks is "GPU hours spent with no
ledger entry to show for them", and the answer this morning is 1.6475 — 100% of
the week's dispatches.** The meter is not at fault; RANK 1 is. Flagging that the
receipt organ built on 2026-08-11 did its job perfectly and still could not
prevent this, because it records *that a job ran and what it cost* and nothing
records *what it returned*. The 53 KB log holding the answer survived only
because `/data/tmpf3q6yglx` has not been cleaned up yet.

### 6. Stuck decisions

**D5's load-bearing question has been answered by action and not recorded.** The
8th audit's correction asked the owner: *"Was the 21:03 pause meant to be
temporary?"* The owner deleted both `.paused` and `.loop-paused` at 06:47. **It
was temporary.** D5's original three options are therefore live again, with
**5 h 6 m** until the grant lapses. I have appended this to
`DECISIONS_NEEDED.md` rather than closing it — closing an owner decision is not
mine to do.

Nothing else is blocked that the system could have resolved itself with a
bakeoff, and no owner decision has been quietly acted on by the loop.

### 7. Bakeoff hygiene — **CLEAN**

`DECISIONS_RESOLVED.md` holds the same 2 decisions, both from PS.01, both
re-checked:

- **PS.01/J → VOID**, correctly — three arms below the 3.0σ learning gate, with
  the reasoning recorded verbatim. A VOID was not treated as a verdict.
- **PS.01/J2 → WINNER `impact_speed`** — 2.66σ over the runner-up, 10.32σ over
  the null, all 11 gate-eliminated arms named, `screen` rationale stated. The
  winner is outside the noise margin.

No decision was made without a learning gate.

### 8. The honest summary — are we closer to a curious humanoid?

**No. We are exactly where we were 9 hours 44 minutes ago, and that is the
finding.**

There is nothing to weigh this morning because nothing was built. The ladder
reads 67 PASS, the same 67. Curiosity has 12 specs and none has run. The unified
brain has 21 specs and none has run. Zero of ten senses are load-bearing — **no
sense in this system has yet been shown to change anything Jack does.**

Two things are worth saying plainly anyway.

The first is that **the instrument keeps proving it can be trusted.** I went
looking for silent loosening across 87 commits and found controls strengthened
and seeds increased, never the reverse; 67 of 67 PASSes have implementations,
resolvable commits, matching seed counts and real controls; every GPU hour is
receipted including the ones that bought nothing. When the builder found its own
seven-day-old bug this morning it wrote the scar into the docstring, named the
date, and registered a spec to catch the next one. **That is a system telling the
truth about itself without being asked.**

The second is that **a 0.66-hour GPU run cleared its gates by an order of
magnitude and became an `ERROR` because of a string key.** That is the honest
picture of where the bottleneck now is. It is not rigour — rigour is the one
thing this project has in surplus. It is that the machine is permitted to run
for four and a half hours a day, and in this particular four and a half hours
the one experiment it paid for could not get its answer back through the door.

We are not closer to a curious humanoid. We are closer to a system that will not
lose the next answer.

---

## FOR THE BUILDER

Ranked. **1 is time-critical — the payload is in a temp directory nobody is
protecting.**

1. **T1.02's answer already exists. Do not re-pay for it blind.** The completed
   Kaggle run's `RESULT` line is in
   `/data/tmpf3q6yglx/out/jack-ladder-1786482462.log` — a JSON array of
   `{stream_name, time, data}` records; concatenate `data` and take the line
   starting `RESULT `. **Copy it somewhere durable before anything else**;
   `/data/tmp*` is not protected and losing it converts a free recovery into
   another 0.7 GPU-hours. Whether a hand-carried payload is admissible evidence
   is your call and I am deliberately not making it — carrying data around the
   runner is the shape of provenance laundering this project distrusts. **My
   recommendation: treat these numbers as the pre-registered expectation, then
   deliver the result through the fixed path** (`JACK_REUSE_KERNEL` reattach if
   the kernel is still reachable, otherwise a fresh 0.7 h run out of 17.38 h
   available). If the re-run disagrees materially with the table below, *that* is
   a finding worth more than the PASS:

   | seed | reference_gain | structure_advantage | beats_mean |
   |---|---|---|---|
   | 0 | 8.00 | 22.51 | 11.81 |
   | 1 | 7.85 | 17.35 | 9.03 |
   | 2 | 8.47 | 24.68 | 13.61 |

   Gates are 1.5 / 1.25 / 1.10. `reference_gain ≈ 8` means **this is not the
   VOID the last three handoffs warned about.**

2. **Finish the job-contract migration, not just the delivery half.** `gpu.py`'s
   `result_json` (in your working tree) fixes the *read*. T1.02 still *writes*
   to a hardcoded `/content/out.json` and still passes that full path to
   `fetch=`, so on Kaggle the artifact is still never collected — you would be
   relying on the newly-populated stdout path every time. Use the wrapper's own
   `JACK_OUT`. Then make `t0_24_result_survives_delivery.py` assert the
   **contract**, in both directions and across the caller set: no test may pass a
   path separator to `fetch=`/`result_json`; a Kaggle-shaped `JobResult` carrying
   only a `.log` must RAISE rather than parse; and a job that wrote outside
   `JACK_OUT` must be detectable. A property that only proves `result_json` works
   on a good input is decorative — the defect was a *bad* input being accepted.

3. **Fold RANK 4 into T0.24 while you are there.** Every line in the default
   `SUBMISSION_LOG` must carry `attempt_id`, `backend`, and
   `phase ∈ {attempt, result}`. Check both directions — a log containing
   `{"phase":"selftest"}` must FAIL the property. **Leave the existing line in
   place**; deleting evidence to make a gate pass is the wrong lesson.

4. **T2.01, and it is now this week or not at all.** 17.3804 Kaggle-hours remain
   and the bucket closes Sunday 2026-08-16. T2.01 is `est_hours=6.5`,
   `prefer="kaggle"`, affordable, and frees 26 specs — 3.7× the next blocker,
   with every curiosity spec and every Tier-5 and Tier-6 claim behind it. Do not
   spend the week's GPU on anything cheaper first. If permitted hours run out at
   12:00, **submitting before then is what matters** — the poll survives the
   parent, as T1.02 just demonstrated.

5. **Make `coverage.py` distinguish a commitment-bearing PASS from an adjacent
   one** (carried verbatim from the 8th audit, unactioned only because you had no
   hours). Give `COVERS:` a kind — `(claim)` vs `(fixture)` / `(rule)` /
   `(sensor)` — count only `claim` in `n_pass`, report the rest separately.
   Re-declare **PG.4 as `(fixture)`** and **LC.01 as `(rule)`**. Expected effect:
   commitments-with-nothing-passing goes **12 → 14**, and `curiosity` and
   `one brain / unison` correctly read zero. Add the property to T0.21 in both
   directions.

6. **UB.9 is still the cheapest route into the unison hole** — `run blocked`
   ranks it 3rd (frees 4, blocks 7), *"Heard, not seen: the task that is
   impossible without fusion"*. Deferred five consecutive iterations; now the
   oldest untaken finding.

**One thing not to do:** do not append a new LESSONS entry for the T1.02 caller.
`docs/LESSONS.md:2252` — *"grep for the retired rule's syntax, not for the new
rule's callers… encode that search as a property in the guard spec"* — is
already exactly this lesson, written yesterday, about `run.py:590`. Item 2 above
is the follow-through it asked for. A second entry saying the same thing would
make LESSONS.md longer and the system no wiser.

---

## FOR THE OWNER

**1. D5 is live again and has about five hours left.** You answered its
load-bearing question by action — deleting `.paused` and `.loop-paused` at
06:47 means the 21:03 pause was temporary. That re-opens the original question,
which nothing else can answer:

> The `.usage-resumed` grant (ceiling 100%) **expires 2026-08-12T12:00 UTC**.
> Weekly usage is at **95%**. Renew daily / grant through to the weekly reset /
> accept the stop at 12:00?

Nobody is proposing to weaken the 90% rule; it stays the default under all three
options. Appended to `DECISIONS_NEEDED.md` with today's numbers.

**2. The concrete cost, in one line.** 17.38 free Kaggle GPU-hours expire Sunday.
T2.01 needs 6.5 of them, frees 26 specs, and is the gate in front of *every*
curiosity spec — GOAL.md's north star, currently 12 specs and zero ever run.
Those hours are only spendable during hours the loop is permitted to run, and
right now that is about five. **This is not "a slower week"; it is whether the
curiosity thesis becomes testable this month.**

**3. A 0.66-hour GPU run passed everything and was recorded as an error.** Not a
science failure and not dishonesty — a string key that went stale on 2026-08-05
and was never migrated in one file. The result is recovered and intact
(reference_gain ≈ 8 against a floor of 1.5; the structured model beat its
shuffled control by 17–25×). I am telling you because it is the clearest single
picture of where this project's bottleneck now sits: **not rigour, which is in
surplus, but the number of hours the machine is allowed to run and how much of
each one survives to the ledger.**

**4. Trust the green ticks; keep asking what they measure.** Second audit
running: zero controls weakened, zero thresholds loosened, zero seed counts
reduced, 67 of 67 PASSes with real implementations and real controls, zero GPU
hours unaccounted. The builder found and disclosed its own week-old bug within
ten minutes of being allowed to run again. **The ledger is worth believing.**
What it is being asked to measure is still not the thesis: 0 of 10 senses are
load-bearing, and curiosity and unison are each discharged by a single spec that
does not test them.
