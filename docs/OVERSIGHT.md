# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-10 06:45 UTC (4th audit; previous 2026-08-10 00:45 at `b809b6b`)
**HEAD:** `a40336f` · ladder **58 PASS / 147 registered** (2 FAIL, 1 VOID, 1 ERROR) ·
**20 commits unpushed** · working tree dirty from the Review organ running
concurrently (`PROGRESS.md`, `PROGRESS_LOG.md`, `INTEGRATION_QUEUE.md`,
`ladder_prompt.md`) — not mine, and excluded from my commit.

## VERDICT: DRIFTING

**The ledger is trustworthy and I can say so with independent evidence.** I
re-derived every PASS from the record (`run verify`): 57/57 verdicts still
re-derive from their committed `_check`, 53 controls probed, **0 gates ignore
their control, 0 controls declared but never run, 0 unreplayable**. Every PASS
has an implementation on disk, every recorded `commit` still exists in git,
`run stale` reports no claim that names a test which has since changed. Section
2 is clean for the **fourth** audit running: I read every diff to
`registry.py`, `registry_expansion.py` and `experiments/tests/` over the
project's entire life and found **no threshold moved in the loosening
direction, no control deleted or weakened, no assertion removed, no `_check`
that gained a permissive `or`**. Every `or` in every live `_check` is a VOID
condition or a deliberate redundancy, and the one seed-count reduction (T1.08,
3→1) is justified by a measurement in its own commit. That is a real result and
it is the best news in this report.

The drift is not in the ledger. It is in **what the ledger is allowed to be
about**:

1. **44 of 147 specs cannot be attempted at all**, and the blocked set is
   GOAL.md's headline rather than a side branch: **all 7 curiosity specs, 10 of
   16 unison specs**, and Tiers 3–5 almost entirely. 36 of the 44 sit behind
   `T2.01 = FAIL`, which needs a GPU re-run, which needs a `git push`, which
   needs **one line from the owner (D3)**. That line has been asked for across
   three audits. (§3.2, §6.1)

2. **T2.01 came back weaker, not stronger, and it is the clean number.** 5.58
   GPU-hours, 3 seeds, ~692K steps/seed: `sigma_advantage` **1.19** against a
   5σ bar, down from 2.21σ at 192K steps. More compute lowered the effect size
   because seed spread grew (means 231.9 / 384.5 / 155.3). The builder
   journalled this honestly and told the next iteration to read it before
   planning GPU work. It changes D3's price tag and nobody has restated D3 with
   the new number. (§5.3, §8)

3. **The most expensive entry in the ledger names the wrong commit.** T2.01's
   row is stamped `commit 2cd0289`; the code that actually ran on Kaggle at
   19:42 was `496e951`, six commits earlier. `assert_ref_is_current` exists
   precisely to make a GPU result attributable to a commit, and the record then
   throws that attribution away. Self-disclosed by the builder, still open.
   **RANK 1 — it is the only finding in this audit that makes a ledger entry
   say something untrue.** (§1.2)

4. **Three cheap integrity items are now unactioned at their FOURTH audit** —
   T1.03/T1.05 controls, the 19 undeclared `Spec.control` fields, ME.8 at one
   seed. Each is under an hour of CPU work, none needs the owner, and each has
   lost to a new PASS every cycle. (§1.3–§1.5)

The builder is alive, fast and honest. The last 24 h were its most productive
day: 22 iterations, +16 PASS, and it took the previous audit's RANK-1 finding
(the missing senses) over its own handoff — exactly what the FOR THE BUILDER
section is for. I want to be clear that this verdict is not a complaint about
the builder's judgement. It is that the builder can only reach what CPU can
reach, and what CPU can reach is no longer where the goal is.

---

## 1. Integrity of the ledger

**Mechanical checks — all clean.** For all 62 entries: implementation present in
`experiments/tests/` (62/62), recorded `commit` resolves in git (62/62), spec
present in the live registry (62/62), no duplicate-implementation glob hazards.
`run stale`: *"No stale claims — every verifiable entry names the test as it
stands today."*

**Backward re-judgement — all clean.** `python -m experiments.run verify`, the
organ this audit asked for two cycles ago and which now exists:

    Re-judged 57 PASS entries from the record alone; probed 53 controls.
      verdicts that no longer re-derive      0
      gates that IGNORE their control        0
      controls declared but never run        0
      gates that could not be replayed       0
      entries that could not be audited      0

No PASS in this ladder is unearned. No control is decorative. This is the
strongest statement this system has ever been able to make about itself, and it
is now a command anyone can run rather than an auditor's assertion.

### 1.2 T2.01's row names a commit that did not run — RANK 1, NEW

`experiments/ledger.json` records T2.01 as:

    "commit": "2cd0289",  "ran_at": "2026-08-10T01:17:15",  "duration_s": 20090.47

The Kaggle kernel was submitted at ~19:42 on 08-09 and ran for 5.58 h. HEAD at
19:42 was **`496e951`** (`git log --before="2026-08-09 19:42" -1`), which is
also the ref recorded in `.git/FETCH_HEAD`. `2cd0289` is what HEAD happened to
be when the detached poll wrote the result — **six commits later**.

Why this outranks everything else here: `experiments/gpu.py:assert_ref_is_current`
exists on the stated principle that *"a GPU result is only attributable to a
commit if the commit is what ran"*, and it enforces that at submission. The
record then discards it. Every multi-hour GPU entry the ladder will ever write
is stamped with the wrong code, and the error is silent, plausible and grows
with job duration — the exact shape of failure this project was built to make
impossible. For same-minute CPU specs the stamp is right, which is why nothing
has caught it.

The builder found this itself and wrote it as handoff item 1 at
`LOOP_JOURNAL.md:1645`. It was not picked up by the four iterations since.
Cheapest fix is the one already identified: stamp the commit when `_experiment`
*starts*, not when the result lands. `build_job` already pins and prints the
real ref, so the correct value is available at submission time.

### 1.3 Four PASSes have no control at all — RANK 5, fourth audit

`T0.01`, `T0.10`, `T1.03`, `T1.05`. (Down from five: T0.08 gained a real
control on 08-10.) `run verify`'s Probe B has nothing to say about these —
there is no control to delete. T0.01 and T0.10 are structural preconditions and
I do not think they need one. The two that do:

- **T1.03 (gradient coverage)** — 3 seeds, `params_without_grad`. A parameter
  deliberately detached from the graph *must* be reported as orphaned, or the
  metric has never been shown capable of reporting the bad case.
- **T1.05 (frozen stays frozen)** — 1 seed, no control. An unfrozen sentinel
  *must* move.

Note also that T1.05's hypothesis ("The pretrained trunk/LLM does not change
during policy training") reads against the PLASTIC-ONLY decree. The builder
handled this correctly and explicitly in `814ed89` — annotated in
`registry_expansion.py` as *"still valid as a MECHANISM test... It no longer
implies we ship frozen parts"*, with **no threshold touched**. That is law 4
observed properly. The annotation lives in `registry_expansion.py` while the
spec lives in `registry.py`; a reader of the spec alone will not find it.

### 1.4 19 entries run a control their spec does not declare — RANK 6, fourth audit

    ME.5, ME.8, PG.1, PG.3, PG.4, T0.03, T0.05, T0.06, T0.07, T0.09,
    T0.11, T1.04, T1.06, T1.07, T1.08, T1.09, T1.10, T2.10, T2.12

The science is fine — `run verify` proves every one of these gates reads its
control. The **declaration** is what is wrong, and `Spec.control` is the field
an auditor greps. 19 of 53 false negatives makes that grep useless. The fix
named two audits ago still stands: have `run_spec` raise when `control_fn` is
supplied and `spec.control is None`, then backfill 19 declarations.

### 1.5 ME.8 is a PASS at one seed whose own commit records a seed-2 collapse — RANK 4, fourth audit

`ledger.json` records `ME.8 seeds: [0]`. `registry_expansion.py:536` declares no
`seeds=`, so it defaults to 1. Its own commit message (`663270b`) reads: *"GRU
retain-bias init fixes seed-2 training collapse"* — **the fix was never verified
at the seed that motivated it.** GOAL.md's standard is "at ≥3 seeds where the
claim is about learning", and ME.8's claim ("a recurrent state resumes
mid-episode; zeroing it hurts") rests on a trained recurrent state. This is the
single weakest PASS on the board and it is a CPU spec.

### 1.6 What improved since the last audit

- **T0.18 built and PASS** — exactly the backward record-check item 1 asked for,
  with a known-answer fixture (a planted loosened gate and a planted
  control-blind gate) so the detector itself is falsifiable.
- **The stale-writer bug found and closed.** A detached GPU poll holding an
  hours-old snapshot reverted 6 entries and 5 amendments; the merge is now
  single-key and T0.08 gained a fifth property plus a replay-the-old-merge
  control that fails on exactly the right two properties.
- **The dirty ledger from last audit was committed properly** (`58f07f6`), as a
  duplicate PG.6 re-run with a note, not silently.
- **T2.01's 5.58 GPU-hours were rescued rather than discarded** during that
  repair — the right call, and journalled with the reasoning.

---

## 2. Thresholds and controls, over time — NO FINDINGS

The repository is six days old, so `--since="7 days ago"` is its entire history:
54 commits touching `registry.py`, `registry_expansion.py` or
`experiments/tests/`. I diffed every one at `--unified=0` and filtered for
removed lines containing thresholds, seed counts, gate operators or assertions
with no identical re-added twin. Every survivor was justified in its commit
message with a measurement. The ones worth naming because they *look* like
loosening and are not:

| change | reads as | actually |
|---|---|---|
| T1.08 `seeds=3 → 1` (`beaea27`) | seed reduction | spec-level seeds were launching 3 **identical** jobs; `_experiment` ignores its seed argument and varies [0,1,2] internally. 3× quota for zero information. |
| T2.01 `TRAIN_MINUTES 30 → 110`, `GPU → GPU_LONG` (`90d8b3c`) | budget inflation | **more** compute against an unchanged 5σ bar, on the pre-registered branch for a still-climbing curve. Comment states "The 5-sigma bar, the control, and the all-seeds rule are untouched." |
| T1.04 `moved_frac >= 0.95` → `undeclared_stuck_params == 0` (`bb55c15`) | weaker gate | *stricter*: any stuck module outside an explicitly pre-declared list now fails loudly, at dotted-submodule granularity so `action_head` as a whole cannot hide a regression in the part that matters. The declared list has one edit in its life. |
| T2.00 `max_vf_pg_ratio` → `max_vf_pg_grad_ratio` (`4df2c8c`) | metric swap | the loss-ratio gate tripped at 178.57 with nothing wrong; the gradient ratio reads a healthy 1.9–2.8× at every minibatch size. Loss ratio kept as a diagnostic. |
| T0.04 `loss jump >20%` → `fidelity ratio < 10` (`7255426`) | threshold rewrite | the old metric read 1.326% in **both** arms — it could not discriminate at all. |

Strengthenings in the same window, for balance: T0.09 gained `is_nvidia`, T0.13
gained three conjuncts and a staleness detector, T6.03 promoted its byteflip arm
from info-only to a gated control, T0.12 went from 12 to 24 properties, T0.08
gained a fifth property and a replay control.

**Silent loosening is the failure this section exists to catch, and it has not
happened once in the life of this project.**

---

## 3. Drift from the goal

### 3.1 What the last 24 hours bought

22 builder iterations, 08-09 06:39 → 08-10 06:39. Every unit traces to GOAL.md:

| work | GOAL.md sentence it serves |
|---|---|
| PG.6 PASS (eyes: R² 0.9747, bearing 1.27°) | "sight" in the sensory inventory; unblocks UB.9 |
| PG.7 PASS (audio leaks 0.000 bits about position) | "hearing"; makes the HNS fusion task honest |
| T0.17, T0.18, T0.08 property 5 (ledger provenance) | "protects the honesty of watching what happens" |
| T0.19 + `screen` gate mode | "Decisions are made by bakeoff, never by argument" |
| PS.01/J and J2 bakeoffs → WINNER `impact_speed` | interoception; unblocks LC.03–LC.06 |
| PS.01 FAIL (J₀=2.405, α=0.0293 measured) | "the needs ARE the curriculum" — refuted §2.3's arithmetic with a measurement |
| SM/TA/VO registration + T0.20 + `experiments/senses.py` | "EVERY SENSE A HUMAN HAS" |
| LC.02 re-cert, `run senses`, tmp_reaper, credit detection | the machine, per SYSTEM.md |

**No drift in this list.** Nothing the builder did in 24 hours fails to trace to
a GOAL.md sentence. That is not the usual result and it deserves saying.

### 3.2 The converse question, and it is the whole finding — RANK 2

Which parts of GOAL.md have **no passing spec at all**:

| family | registered | PASS | blocked by a FAIL/VOID/ERROR |
|---|---|---|---|
| **CU** — curiosity | 7 | **0** | **7 of 7** |
| **UB** — all senses in one brain | 16 | **0** | 10 of 16 |
| **SM / TA / VO** — smell, taste, voice | 7 | **0** | 0 (all runnable) |
| **LC** — the learning core | 7 | 3 | 4 of 7 |
| **PS** — drives / interoception | 1 | 0 (FAIL) | — |
| ME — memory | 18 | 10 | 1 |
| PG — the world | 8 | 8 | 0 |
| T0/T1 — harness & primitives | 33 | 30 | 1 |

Recomputed over all 147 specs against the live ledger: **44 specs have a
FAIL/VOID/ERROR in their dependency chain** — 36 behind `T2.01 = FAIL`, 4 behind
`T2.02 = VOID`, 4 behind `PS.01 = FAIL`.

So: the harness is excellent, the world is measured, memory is genuinely
demonstrated — and **the three claims GOAL.md leads with have zero passing
specs between them.** "He explores because he wants to": 0/7, all unreachable.
"All senses in unison, one brain": 0/16, 10 unreachable. "Climbing the ladder on
attempt 40 after falling on 1–39": PG.3 proves the ladder is climbable by a
*scripted* ascent; nothing shows Jack learning to climb it.

**One material change since the last audit:** T2.01 moved VOID → FAIL. That
means the 36 specs behind it are no longer blocked by the VOID-policy question
at all — **D2's price tag drops from 40 specs to 4**, and D3's rises to
everything. Whoever restates these decisions should carry that number.

### 3.3 The new senses organ overstates what it measures — RANK 3, NEW

`python -m experiments.run senses` is a genuinely good organ — an outside
reference that the registry cannot edit, coverage by explicit declaration rather
than grep, gated by T0.20 with a control that fails on exactly the right two
properties. It is the guard `LESSONS.md:783` prescribed and nobody built for 30
hours. I want that on the record before the criticism.

The criticism is its vocabulary. `senses.py:131`:

    return DEMONSTRATED if self.passing else REGISTERED

`DEMONSTRATED` means *"at least one declared spec for this sense is PASS."* It
therefore prints:

    [PASS] sight (vision)     PG.6, T2.03, T3.01   demonstrated: PG.6
    [PASS] hearing (audition) PG.5, PG.7, UB.4     demonstrated: PG.5, PG.7

PG.6 certifies that **the world renders frames a ridge regression can read** —
its own docstring is scrupulous about this: *"A linear read-out on raw pixels...
that would certify the net, not the sensor."* PG.5 certifies that the world
**emits** contact audio with recoverable bearing. PG.7 is a *leakage control*.
None of the three involves Jack's brain. What is demonstrated is that the sense
has a signal to carry, not that Jack has the sense.

GOAL.md's own standard for a sense is explicit and much harder: *"every sense is
load-bearing (and we PROVE each one is — ablate a sense, something measurable
must degrade)."* That is the UB family, and **0 of 16 UB specs have run**. By
GOAL.md's standard, **0 of 10 senses are demonstrated**, not 2.

This matters because "8/10 registered, 2 demonstrated" is a headline number that
will be quoted into the Review, PROGRESS.md and future audits within days. The
fix is one status tier, not a redesign: `ABSENT → REGISTERED → SENSOR (the world
emits it, a probe reads it) → LOAD-BEARING (a UB ablation degrades something)`.
An organ built to stop a capability reading as complete should not be the thing
that makes one read as complete.

### 3.4 Pain and temperature: handled correctly

The builder declined to register them and said why: temperature is
`SURVIVAL_WORLD` W.1/W.3 and arrives with a whole world (a scope call), pain is
an open bakeoff arm in `NEEDS_AND_DEATH` §2.9 that the doc itself calls "a live
question, not a settled design" — registering it as written would decide by
argument a question queued for a bakeoff. **Refusing to register rather than
registering a guess is the right call**, and it was escalated with a narrowed
ask instead of being dropped. No finding.

---

## 4. Is the builder alive and productive? — YES, its best day so far

Window 2026-08-09 06:39 → 2026-08-10 06:39 (`/data/jack-logs/ladder.log`):

    iteration starts                 22
    ended rc=0                       18
    ended rc=1                        4   (07:07, 08:07, 09:07, 10:07 — all credit exhaustion,
                                           all BEFORE the fallback shipped at 12:07)
    aborted on load                   1   (11:07, load 8.37 > 6.0 — correct, tenants first)
    iterations with no end line        0   (both prior cases were before this window)
    PASS delta                      42 -> 58  (+16)
    registry delta                 105 -> 147 (+42)

No thrash, no repeated identical failure, no paused loop, no silent death. The
17:07 and 22:07 missing-end-line cases from the last audit have not recurred,
though `trap ... EXIT` was still not added to `ladder_loop.sh`, so the log still
cannot distinguish "killed having done nothing" from "killed having committed a
PASS" if it happens again.

**Goodhart check:** pass rate 40.0% → 39.5%. The registry grew 40% while PASSes
grew 38%. Rate essentially flat while both counts rose steeply — this is a
ladder being *extended* at roughly the speed it is being *climbed*, which is the
honest reading, not a treadmill. (The Review reached the same number
independently at 06:43.)

### 4.1 The cron still names a model that is always out of credits — third audit

    7 * * * * JACK_LOOP_MODEL=fable /home/opc/jackthelearner/scripts/ladder_loop.sh

**21 of the last 22 iterations opened with `OUT OF CREDITS on fable`.** The
fallback works and is now load-bearing every single hour. Two costs: the
`iteration start` line reports `model fable` when the iteration will run on
opus, so the log records the wrong model for every run in the last 19 hours; and
on 08-09 the same condition killed **six consecutive iterations at rc=1** before
the fallback existed. `JACK_LOOP_MODEL=opus` in cron, one line.

### 4.2 Two organs fire in the same minute, daily

`37 */6 * * * overseer.sh` and `37 6 * * * review.sh` collide every morning at
06:37 — both were running while I wrote this, and the Review committed
`a40336f` mid-audit, moving HEAD underneath me. They touch different files so
nothing has corrupted, but two unattended agents racing `git commit` share one
`index.lock`, and a loser gets a confusing failure rather than a retry. Move one
to a different minute.

---

## 5. Compute honesty

### 5.1 The current week

    2026-W32 (reset Sun 2026-08-09, next reset Sun 2026-08-16)
      kaggle  11.9635 h of 30.0    ->  18.04 h remaining, expiring in 6 days
      colab    0.0015 h

W32 reconciles cleanly against the ledger: T2.02's VOID run (22,604 s = 6.28 h,
08-09 07:30) plus T2.01's FAIL run (20,090 s = 5.58 h, charged with a job id) =
11.86 h against 11.96 h charged. **The meter is now accurate.**

### 5.2 The previous week does not reconcile, and the cause is known and fixed

    2026-W31   kaggle 37.4554 h (of a 30.0 h ceiling)  +  colab 7.7461 h  =  45.2 h

Ledger entries produced in W31 account for roughly **6.5 GPU-hours**
(T0.09/T0.10/T0.11 ≈ 0.13 h, T1.09/T1.10 ≈ 0.12 h, T1.08 0.36 h, T1.12 0.32 h,
T1.02 ERROR 6.6 s, T2.01 v3 VOID 5.58 h). **~38.7 GPU-hours have no surviving
ledger entry.**

I do not report this as fresh waste. The causes were found and fixed on 08-09,
and `496e951` names them: `charge()` billed **failed** jobs as work, re-billed
**reattached** kernels, and billed **this box's wall clock** rather than the
metered window. Plus the documented 5.5 h burned on 08-07 by submitting one GPU
job per declared seed. So an unknown but large fraction of the 38.7 h was never
spent at all — it was over-*counted*, which is how a 30 h ceiling closed at
37.4554 h and **denied T1.02 its 0.7 h**. T1.02 has been `ERROR` ever since
(08-08), and it is a Tier-1 control spec.

What remains genuinely open, in T0.12's own notes: *"Reconciling the meter
against Kaggle's own reported kernel runtime needs network and a live kernel,
and remains OPEN."* Until that runs, the only assurance the meter is right is
that it agrees with itself.

### 5.3 The GPU that was spent bought a weaker number — read before spending more

T2.01, the single most expensive result in the project (5.58 h, 3 seeds,
~692K env-steps/seed, 331 wall-minutes on a Kaggle P100):

    trained_mean        257.2   (seeds: 231.9, 384.5, 155.3)
    random_mean         118.0 ± 52.7
    untrained (control) 153.8   <- an UNTRAINED net already beats random
    sigma_advantage     1.19    against a 5.0 bar
    all_seeds_beat_random  1.0

Every seed beat random, and the effect size is not close. It is **weaker** than
the invalidated v4 it replaced (2.21σ at 192K steps) despite 3.6× the steps —
because the across-seed spread (116.7) grew faster than the mean. And the
control says a meaningful share of the remaining gap is architectural bias
rather than learning: untrained 153.8 vs random 118.0.

The builder recorded all of this and warned the next iteration in writing
(`LOOP_JOURNAL.md:1636`): *"Do not cite it as an architecture verdict on its own
— T2.02 is the spec built to arbitrate this, and it is still VOID."* That is
exactly right and I have nothing to add to it except this: **the case for D3 is
usually made as "one 13 h re-run frees 36 specs." After this result, the honest
case is "we cannot even find out whether it frees them without the push."** Both
arguments end in the same one-line answer, but the owner should be given the
second one.

---

## 6. Stuck decisions

### 6.1 D3 is the whole bottleneck and it has got worse — RANK 2

*May the loop `git push` its own commits to `origin/main`?*

    unpushed commits   9  (last audit, 00:45)  ->  20  (now)
    Kaggle expiring    23.6 h by 08-16         ->  18.04 h by 08-16
    specs behind it    36 (all CU, 10 of 16 UB, Tiers 3-5)

`assert_ref_is_current` refuses to build a GPU job from an unpushed HEAD, and it
is right to — the VM clones from GitHub, and on 08-05 the alternative cost two
GPU runs and produced a wrong diagnosis. Every GPU submission needs a push
first. The repo is already public and already contains every file involved.

Nothing else in `DECISIONS_NEEDED.md` blocks 36 specs. Nothing else is one line.

### 6.2 Two decisions have enough evidence and one has had it for six days

- **The "Kaggle GPU is not being granted" block at the top of the file** —
  asks the owner to choose among three options for a problem the system solved
  itself. It claims to block T0.10 and T0.11, **both PASS since 2026-08-04**,
  and Kaggle is now the primary backend with 11.96 h billed this week. The
  suggested `DECISIONS_RESOLVED.md` entry is already drafted in the file. Fifth
  audit asking. It is the first thing an owner or a fresh agent reads, and it is
  false.
- **"/data is 95% full"** is marked `(OPEN, owner action)` and is now stale:
  `df` reads **18 GB used of 100 GB, 83 GB available**. The condition it
  escalates no longer exists. Strike it or mark it resolved — a decisions file
  where two of the top entries are obsolete trains readers to skim it.

### 6.3 Nothing is blocked that a bakeoff could have settled

I checked the converse. D1 (does the 57M trunk stay in the control path) is
exactly the question T2.02 was built to arbitrate; it is VOID and gated on GPU,
so escalating rather than arguing is correct. D2 is a policy question about what
a VOID *means*, which no experiment can answer. No decision is sitting on the
owner that the system could have decided for itself.

### 6.4 Was an owner-decision acted on without being recorded? — no

I checked the recent directives: PLASTIC-ONLY (recorded in GOAL.md, CHAMPIONS.md
and propagated through nine affected specs with no threshold touched), the
owner's hands (recorded, approved same day), T5.01 scheduled, physics-first
"RUN IT" — all four are recorded where they were acted on. The Kaggle
accelerator resolution is the one that was acted on and never written down, and
that is §6.2's first bullet.

---

## 7. Bakeoff hygiene — one disclosed slip, no violations

Two bakeoffs exist, both from 08-10, and `DECISIONS_RESOLVED.md` is otherwise
empty by design (its nine `TEST` entries were self-test pollution, removed, and
the file made injectable so a test cannot reach it again).

- **PS.01/J — VOID.** Three arms below the 3σ learning gate. VOID recorded as
  VOID, no winner crowned. Correct.
- **PS.01/J2 — WINNER `impact_speed`.** 0.973 AUC, 10.32σ over null, beating
  runner-up `peak_dvel` by 2.66σ against a pre-registered `margin_sigma` of 1.5.
  Both controls (`noise` 1.47σ, `constant` 0.28σ) failed the gate as designed.
  Two arms cleared, meeting `MIN_FINISHERS = 2`. **Winner is outside the noise
  margin; no VOID was treated as a verdict.**

**The `screen` gate mode deserved scrutiny and survives it.** It converts "an
arm missed the learning gate → VOID" into "eliminate that arm and let the
survivors compete", which is structurally a relaxation of the decision
procedure. It is fenced properly: the gate value itself is unchanged; ≥2 arms
must still clear it; controls still invert the verdict; the mode and a written
rationale live on the **committed Spec**, not on the call; and it is gated by
its own spec (T0.19, PASS). The strongest check is the one bakeoff.py makes on
itself — `screen` does **not** change the verdict of the run that motivated it
(round 1 had exactly one finisher and stays VOID under both modes). The
distinction it rests on is sound: a learner that misses the gate is ambiguous
(broken run or worse architecture?), an observable that misses it is not, since
every arm is a deterministic reduction of the same cached rollouts.

**The one slip, self-disclosed:** the builder ran the round-2 bakeoff once
during development *before* committing the spec, which inverts SYSTEM.md's
ordering. It discarded that draft's `DECISIONS_RESOLVED.md` entry, committed the
spec, and re-ran — so the record contains exactly one verdict from the committed
spec. The remediation is right and the disclosure is what a healthy system looks
like. I note it only because the arm set *and* the gate mode for round 2 were
therefore chosen with round-1 results visible; the counter-check above is what
keeps that from mattering here, and it should be run again the next time a spec
introduces a new gate mode.

---

## 8. The honest summary — are we closer to a curious humanoid?

**Yesterday: yes, genuinely, and more than on any previous day. Toward the
ladder-and-apple standard specifically: barely.**

What is real. Jack has eyes that resolve geometry (R² 0.9747) and ears that
localise a contact to 1.27° while leaking 0.000 bits about position through the
wrong channel. He has a diary that survives death, is attributed per person, and
dissociates cleanly from his skills in both directions. The world he lives in
has measured physics. The impulse channel that tells a fall from a collapse was
decided by a bakeoff against controls rather than by argument, and when the
drive layer's energy arithmetic met a real integrator it was **refuted with a
number** — a statue outliving an actor, §5's dark room measured rather than
predicted. Three senses that no command in this repository could see yesterday
morning are schedulable CPU work tonight. The ledger can now re-judge itself
from the record.

What is not. **Nothing in this system has yet wanted anything.** Curiosity is
0 of 7 and all 7 are unreachable. Unison — the namesake claim, "what he hears
teaches what he sees" — is 0 of 16, with 10 unreachable, so no sense has been
shown to be load-bearing in one brain and by GOAL.md's own ablation standard
**zero of ten senses are demonstrated**. PG.3 proves the ladder is climbable by
a script; nothing has climbed it. The one attempt at "a policy that learns to
move" came back at 1.19σ against a 5σ bar with an untrained network already
beating random by a third of the gap.

So the fair answer to "closer to a creature, or just a longer list of green
ticks?" is neither of the offered options. We are building an **instrument** of
unusual quality — and every one of its 58 ticks is honest, which is not nothing
and is more than most projects can say. But an instrument measures; it does not
live. The gap between 58 PASSes and one creature is not 89 more PASSes. It is
the 44 specs sitting behind a FAIL that cannot be re-run, behind a `git push`
that nobody has authorised.

Twenty-two iterations of excellent work yesterday, and **not one of them could
touch the thing the goal is about**. That is the finding. It is not the
builder's fault and it is not fixable by building harder.

---

## FOR THE BUILDER

Ranked by damage to the trustworthiness of the ledger. None needs the owner.
Items 2–4 are carried from **three** prior audits.

1. **Stamp the running commit at submission, not at record time (§1.2). NEW,
   RANK 1.** `Result.env_stamp()` runs when the result lands, so T2.01's row
   claims `commit 2cd0289` while the Kaggle kernel actually ran `496e951`, six
   commits earlier. Every multi-hour GPU entry the ladder writes will name the
   wrong code, silently. `build_job` already pins and prints the true ref —
   capture it when `_experiment` starts and carry it into the `Result`. Then add
   the ordering guard the builder itself scoped out (`LOOP_JOURNAL.md:1650`):
   `record()` should warn or refuse when an update moves `ran_at` backwards.
   Consider a T0.17 property asserting that a GPU entry's `commit` equals the
   ref the job was built from — this is exactly the class T0.17 exists for.

2. **Give T1.03 and T1.05 controls (§1.3). Fourth audit.** T1.03: a parameter
   deliberately detached from the graph that *must* be reported as orphaned.
   T1.05: an unfrozen sentinel that *must* move. Both cheap. Both convert "we
   observed the good thing" into "and the measurement can see the bad thing."

3. **Make `Spec.control` load-bearing (§1.4). Fourth audit.** 19 entries record
   `control_metrics` while declaring `control=None`. Have `run_spec` raise when
   `control_fn` is supplied and `spec.control is None`, then backfill the 19.
   `run verify` already prints the exact list. The declaration is the audit
   surface and it is 19/53 wrong.

4. **Re-run ME.8 at 3 seeds (§1.5). Fourth audit.** PASS at `seeds=[0]` whose
   own commit message records a *seed-2 training collapse* fixed by a GRU
   retain-bias init — the fix was never verified at the seed that motivated it.
   `registry_expansion.py:536` declares no `seeds=`. This is the weakest PASS
   on the board and it is CPU work.

5. **Add a fourth status tier to `experiments/senses.py` (§3.3). NEW.**
   `DEMONSTRATED` currently means "some declared spec is PASS", so PG.6 (a ridge
   probe on rendered pixels, whose own docstring says it certifies *the sensor,
   not the net*) marks sight `[PASS]`. Propose:
   `ABSENT → REGISTERED → SENSOR (the world emits it and a probe reads it) →
   LOAD-BEARING (a UB ablation degrades a measured quantity)`. Under it today:
   sight and hearing read SENSOR, and **0 of 10 read LOAD-BEARING**, which is
   what GOAL.md's "ablate a sense, something measurable must degrade" actually
   asks. Extend T0.20 with the property that a sense cannot reach LOAD-BEARING
   without a passing UB-family spec. The organ is good; the word is too strong,
   and it is about to be quoted everywhere.

6. **`JACK_LOOP_MODEL=opus` in cron (§4.1). Third audit.** 21 of the last 22
   iterations opened `OUT OF CREDITS on fable`. The fallback should not be
   load-bearing every hour, and `iteration start` currently logs a model the
   iteration will not use.

7. **`trap ... EXIT` in `ladder_loop.sh` (§4).** No misses this window, but the
   log still cannot distinguish "killed having done nothing" from "killed having
   committed a PASS."

8. **Move `review.sh` off minute 37 (§4.2).** It collides with `overseer.sh`
   every morning; two unattended agents share one `index.lock`.

9. **Restate D3 with the post-T2.01 number (§5.3).** The entry argues "the
   re-run frees 36 specs." After 1.19σ the honest argument is "we cannot find
   out whether anything frees them without the push." Same ask, truthful price
   tag — and SYSTEM.md requires the cost travel with the directive.

---

## FOR THE OWNER

1. **D3, and it is now the only thing that matters.** *May the loop `git push`
   its own commits to `origin/main`?* **20 commits are unpushed** (9 at the last
   audit). `assert_ref_is_current` refuses to build a GPU job from an unpushed
   HEAD and is right to — the VM clones from GitHub. **18.04 of 30 Kaggle hours
   expire on 2026-08-16 and unspent free quota is not saved.** Behind it: 36
   specs, including **all seven curiosity specs and ten of sixteen unison
   specs** — the two claims GOAL.md leads with. The repo is already public and
   already contains every file involved. Options 1 (standing authorisation) or 2
   (authorised only when a GPU submission needs one) both unblock it today;
   option 3 is a real choice but its price is the quota, and it should be picked
   deliberately rather than by silence.

2. **Read this number before you decide D3, because it cuts both ways.** The
   T2.01 re-run finished — 5.58 GPU-hours, the first clean run with the dropout
   bug fixed at both call sites. It **failed**: 1.19σ against a 5σ bar, *weaker*
   than the 2.21σ it replaced despite 3.6× the training steps, and an
   **untrained** network already scores 153.8 against random's 118.0. Every seed
   beat random; the effect size is not close. This does not make D3 less urgent
   — it makes it more, because the question "is this architecture capable of
   learning to move at all?" is now the live one, and it cannot be asked without
   GPU. It does mean nobody should promise you that one more run turns 36 specs
   green.

3. **Two entries at the top of `DECISIONS_NEEDED.md` are obsolete and should be
   struck.** *"Kaggle GPU is not being granted"* claims to block T0.10 and T0.11
   — **both PASS since 2026-08-04**, and Kaggle has been the primary backend all
   week (11.96 h billed). Fifth audit asking; the replacement
   `DECISIONS_RESOLVED.md` entry is already drafted in the file. *"/data is 95%
   full"* is marked OPEN; `df` now reads 18 GB used of 100 GB. Both are the
   first things a reader sees, and both are false.

4. **D2 got much cheaper while you were not looking.** The last audit priced
   "does a VOID block its dependents?" at 40 specs. T2.01 has since moved
   VOID → **FAIL**, so its 36 dependents are blocked regardless of how you
   answer. **D2 is now worth 4 specs** (UB.15, UB.16, T2.13, T5.09). It is no
   longer urgent; D3 absorbed all of its weight.

5. **A scope question the builder correctly refused to answer for you.** Smell,
   taste and voice are now registered (7 specs, all with resolving dependencies,
   `SM.01`/`TA.01`/`VO.01` all CPU and buildable without you). **Pain and
   temperature were deliberately not registered** — temperature arrives only
   with the whole survival world (W.1/W.3: thermal ODE, shelter, occlusion), and
   pain is an open bakeoff arm that `NEEDS_AND_DEATH.md` §2.9 itself calls "a
   live question, not a settled design". The ask is: **schedule the W family
   (temperature, and with it shelter — the only mechanism in the design that
   teaches construction) now, or after the LC bakeoff?** Both now read `ABSENT`
   in `run senses` every time it runs, so the hole is visible rather than
   invisible; it is merely open.

6. **For information, no action needed.** The ledger is clean, and this time by
   a command rather than by my assertion: `python -m experiments.run verify`
   re-judged all 57 PASSes from the record through their committed gates
   (57/57 re-derive), probed 53 controls (0 ignored, 0 undeclared-and-unrun),
   and found 0 unreplayable. I separately read every spec and test diff in the
   project's history: **no threshold has ever been moved in the loosening
   direction, no control weakened, no assertion removed.** On the integrity
   axis, four audits running, this system is doing exactly what it was built to
   do. What it cannot currently do is spend a GPU hour.

---

*Ledger untouched. No experiment re-run — `run status`, `run next`, `run stale`,
`run senses` and `run verify` are read-only re-judgements of the existing
record, and I confirmed `verify` writes nothing before running it. Nothing
outside `/home/opc/jackthelearner` changed. The Review organ was running
concurrently and its files (`PROGRESS.md`, `PROGRESS_LOG.md`,
`INTEGRATION_QUEUE.md`, `ladder_prompt.md`) are excluded from this commit. This
commit is not pushed; that is D3, and it is not mine to decide.*
