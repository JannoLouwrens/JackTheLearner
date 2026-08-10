# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-10 12:45 UTC (5th audit; previous 2026-08-10 06:45 at `a40336f`)
**HEAD:** `90934f1`, tree clean, nothing unpushed · ladder **61 PASS / 160
registered** (1 FAIL `T2.01`, 1 VOID `T2.02`, 1 ERROR `T1.02`) · Kaggle **18.04
of 30 h remaining, expiring 2026-08-16**

## VERDICT: DRIFTING

**The ledger is clean and I can prove it with a command rather than an
assertion.** `python -m experiments.run verify` re-derived **60 PASS entries
from the record alone through their committed `_check` functions**: 0 verdicts
that no longer re-derive, 0 gates that ignore their control, 0 controls declared
but never run, 0 unreplayable, 0 unauditable. Every PASS has an implementation
in `experiments/tests/`. Every recorded `commit` still resolves in git. The four
integrity items carried across three prior audits are all **closed this window**
— T1.03 and T1.05 have controls, `Spec.control` is now enforced at `run_spec`
(20 undeclared → 0), ME.8 re-ran at 3 seeds, and the commit stamp names the code
that actually ran. Section 2 is clean for the fifth audit running.

The drift is not in the ledger. It is in **the instrument that decides what the
ladder should be about**, and in what the last day of work did and did not buy:

1. **`experiments/coverage.py` — the tool this audit charter orders me to run
   FIRST — reports a passing spec for three commitments that have none.** Not by
   a subtle argument: `shelter/building` is credited with a PASS because the
   regex token `nest` matches the word "ho**nest**". The gap-finder that exists
   because four commitments were invisible is currently making four *other*
   commitments look covered when they are not. (§1a, RANK 1)

2. **XL.00 — the best result of the day — reads its calibration constants out of
   PS.01's ledger entry and never checks that entry is fresh.** PS.01 is on the
   stale list right now. This instance is benign and I verified why; the missing
   guard is not. (§1b, RANK 2)

3. **The push block lifted this morning and no GPU job has been submitted
   since.** D3 was answered YES, `origin/main..HEAD` is empty, 18.04 h expire in
   six days, and the ladder's largest blocker (`T2.01 = FAIL`, frees 26, blocks
   36) is `run next`'s second entry. Eleven CPU-only iterations have passed.
   (§5)

4. **Unison — the constitution's own headline — is 0 of 7, and one of its
   entry points has been CPU-runnable and untaken for 22 hours.** (§3)

---

## 1. Integrity of the ledger

**64 entries, 61 PASS.** Checked mechanically:

| check | result |
|---|---|
| PASS entries with no implementation in `experiments/tests/` | **0** / 61 |
| recorded `commit` that no longer resolves in git | **0** / 64 |
| `run verify`: verdicts that no longer re-derive from the record | **0** / 60 |
| `run verify`: gates that IGNORE their control | **0** |
| `run verify`: controls declared in the spec but never run | **0** |
| `run verify`: controls run but NOT declared (last audit: 19) | **0 / 0** |
| PASS entries with no control at all | **2** — `T0.01`, `T0.10` |

`T0.01` (13 modules import) and `T0.10` (Kaggle job round-trip) remain the only
two claims whose gate has never been shown capable of reporting the bad case.
Carried from audit 4. For `T0.01` I accept it — a failed import is self-evident.
For `T0.10` I do not: "a job round-trips" is exactly the shape that a broken
detector reports as healthy, and the project has already been bitten by it once
(the Kaggle kernel that ran to `COMPLETE` with no GPU attached, 2026-08-04).

**Staleness. 3 flagged, and all three are definitional, not scientific.**
`run stale` names `PG.3`, `PG.8`, `PS.01`. I traced the cause: commit `74f8631`
added `IMPL_DEPS = ["playground.py"]` to nine world-certificate tests, which
widens `impl_sha` — no test logic changed, and I diffed all three to confirm the
only edit is the five-line declaration. `playground.py`'s last modification is
`29d189f` (08-10 08:06), which **predates** PS.01's run (08:32:26), so no world
change is hiding behind these flags. Six of the nine cleared by re-running; three
have not.

**44 of 64 entries (41 of them PASS) predate `impl_sha` and cannot be
staleness-checked at all.** That is 67% of the ledger with no self-knowledge
about whether it still describes live code, and it shrinks only by re-running.
Priced from the entries' own recorded durations: 36 of the 41 are CPU
(27 `CPU`, 6 `CPU_FAST`, 3 `CPU_LONG`), and the whole set of 41 sums to **234
minutes** of previously-measured runtime. Five are GPU (`T0.09`, `T0.10`,
`T0.11`, `T1.09`, `T1.10`), all `GPU_SHORT`.

### 1a. RANK 1 — the gap-finder is reporting coverage that does not exist

`experiments/coverage.py` matches a commitment to a spec by regex over the spec
**title**. Its own docstring records the failure it was born from — BA.01 was
written to close the `balance` hole and the regex could not see it — and its own
lesson (`LESSONS.md:1643`, *"The instrument that finds gaps can have a gap, and
it will flatter you"*) prescribes the fix: an explicit `COVERS:` declaration,
with the regex demoted to a safety net.

**That lesson addressed only the false-negative half.** The false-positive half
is live today. I matched every commitment regex against every one of the 160
spec titles and inspected each hit:

| commitment | tool reports | what is actually there | the accident |
|---|---|---|---|
| **shelter/building** | 4 specs, **1 PASS** | `SH.01` only — **not implemented** | `nest` matches "ho**nest**": `ME.11.0` *"The paraphrase eval set is honest…"* (**PASS**), `T2.02` *"…the honest MLP baseline"* (VOID), `UB.5` *"…or honestly redundant"* |
| **proprioception** | 2 specs, **1 PASS** | `T3.02` "Ablate proprioception" — **not run** | `limb` matches `PG.3` *"Ladder is c**limb**able"* (**PASS**) |
| **touch/contact** | 2 specs, **1 PASS** | `UB.5` "Touch is load-bearing" — **not run** | `contact` matches `PG.5` *"Procedural contact audio"* (**PASS**) — an audio spec |
| **death & retry** | 11 specs, **6 PASS** | ~4 specs, ~2 PASS | `surviv` matches `T0.05` "Preemption survival", `ME.5`, `ME.8`, `T3.10`; `dies` matches `T5.09` "…across bo**dies**" |
| **hearing** | 8 specs, 4 PASS | 7 specs, 3 PASS | `sound` matches `PG.1` *"Playground generates and is physically **sound**"* — "sound" as in *valid* (**PASS**) |
| **hunger/thirst** | 2 specs, 1 PASS | `PS.01` only (correctly PASS) | `drive` matches `T2.08` *"Curiosity **drive**s coverage"* |
| **voice** | 2 specs, 0 PASS | `VO.01`, `VO.02` | **`VO.01` — the primary voice spec — is INVISIBLE** (its title carries no matching token), while `PS.03` *"Damage is a **signal**"* is counted as voice. The count is right by coincidence; every member is wrong |
| **sight** | 6 specs, 2 PASS | 4 specs, 2 PASS | `SM.02` *"Smell finds what **vision** cannot see"*, `TA.01` *"…**visual**ly identical twin"* |
| **memory across lives** | 1 spec, 0 PASS | `XL.00` also belongs here | `XL.00` *"…and the diary crosses"* carries no matching token and no `COVERS:` marker, so today's headline result is not counted against the commitment it was written for |

**Why this is RANK 1 and not housekeeping.** `coverage.py` exits nonzero only
when a commitment has **zero** specs. Today nothing hits zero — but three
commitments that genuinely have zero *passing* specs are being reported green,
and the zero-spec alarm itself is now load-bearing on luck. Concretely: **if
`SH.01` were deleted tomorrow, `shelter/building` would still report 3 specs and
1 PASS, and `coverage.py` would still exit 0.** The owner's own image of success
— *"he builds a shelter"* — would go uncovered and the instrument built to catch
exactly that would say nothing. The same is true of `proprioception` if `T3.02`
went, and `touch/contact` if `UB.5` went.

Nothing in the ledger is wrong. What is wrong is the **map from PASSes to the
owner's commitments**, and that map is the only instrument that answers *"is the
ladder the right ladder?"*

I did not tune the regex and the builder should not either — the file's own
docstring is right that a detector edited until it agrees with you measures the
patience of its maintainer. The repair is in **FOR THE BUILDER §1**.

### 1b. RANK 2 — XL.00 consumes a stale-eligible ledger entry with no freshness check

`experiments/tests/xl_00_death_and_respawn.py:150` reads `j0` and `alpha` out of
PS.01's ledger entry at runtime — deliberately, and the docstring explains why:
*"a calibration pasted into a second file is a constant that drifts from its
measurement (T0.14)"*. That is the right instinct. But `_calibration()` gates on
`entry.status != Status.PASS` **and nothing else**. It does not ask whether the
entry it is reading is stale.

PS.01 is on the stale list at this moment. XL.00 recorded PASS at 12:27:59 using
its numbers.

**This instance is benign and I checked rather than assumed** — PS.01's flag is
the `IMPL_DEPS` widening described above, and `playground.py` has not changed
since PS.01 ran, so the constants are the ones PS.01 actually measured. But the
guard is absent, and the failure class is new: T0.14's lesson stopped constants
being **copied**; reading them live from the ledger does not stop them being
**stale**. A world change tomorrow makes PS.01's entry describe a world that no
longer exists, and XL.00 — plus LC.03/LC.04, which score `life_gain` in that
same world — will keep consuming it silently.

XL.00's own `kills` clause states the stakes better than I can: *"A wrong answer
here is not a wrong answer about the world; it is a wrong answer about every arm
scored in it."*

---

## 2. Thresholds and controls over time — CLEAN (fifth audit running)

I read every diff to `experiments/registry.py`, `experiments/registry_expansion.py`
and `experiments/tests/` over the last 7 days (`223` commits in the window,
`5,782` lines of registry diff and `20,226` lines of test diff), scanning for
removed assertions, threshold constants moved, seed counts reduced, controls
deleted, and `_check` gaining a permissive `or`.

**Two thresholds moved. Both are tightenings, and both are justified with a
measurement in their own commit message.**

**PG.6 `NULL_BEARING_FLOOR` (`6c0fcd1`).** This is the one that deserved
scrutiny, because it was changed *after attempt 2 failed on that gate alone* —
the exact shape of a test being edited until it passes. It is not that. The old
gate demanded both nulls score **worse than 20°** in a **±22°** band; a null
answering 0 scores the band's median |bearing|, measured **8.87 / 8.91 / 8.78°**
on seeds 0/1/2 over 3,000 draws, so 20° required a null *anti-correlated* with
truth — arithmetically unreachable, and the nulls (8.96, 8.20) were sitting
exactly on the constant predictor, i.e. behaving correctly. The number is
correct for the **40–75° control band** and **is still enforced there
unchanged** (58.0° measured). Both registered capability gates (`R²≥0.80`,
`bearing≤5.0°`) were untouched. The replacement adds **two new gates that did
not previously exist**: the probe must beat the constant predictor 2×, and the
grey null must *equal* it to 0.05° (a live check that the ridge solver is
correct, since a grey frame leaves the design matrix rank-0). Net: strictly
more constraint than before. The two FAILs remain in the ledger's history.

**XL.00's permutation gates (`1480126`, `e3c5660`).** `|z| ≤ 3` replaced by
two-sided permutation `p ≥ 0.01`, and the positive controls' `z > 3` replaced by
`p < 0.001`. The old gate was **unreachable**: a permutation z for a linear
statistic is bounded by `sqrt(n−1)`, so at the drift control's n=9 lives the
ceiling was 2.83 and the gate was measuring sample size, not trend. The
replacement is stricter in both directions (`p ≥ 0.01` ↔ `|z| ≤ 2.576`, versus
the old `|z| ≤ 3`), and `N_PERM` went 2,000 → 100,000.

**Everything else in the window moves in the tightening direction**: `T1.03` and
`T1.05` gained controls they never had; `T0.13`'s `_check` gained a
`guard_works` conjunct; `T0.12` went from 12 to 24 properties; `T0.13`'s scan
gained `unreadable_gates` and a positive control that must see an `IMPL_DEPS`
declaration or refuse to report; `T2.02`'s `_check` gained a VOID branch that
can only *reduce* the set of passing runs; `LC.02` gained a calibration check
and an empty-loop control. Commit `9ed2ded` took undeclared controls from 20 to
0 and gave two blind gates eyes.

**No threshold moved in the loosening direction. No control deleted or weakened.
No assertion removed. No `_check` gained a permissive `or`.** I have now checked
this over the project's entire life across five audits and it has never once
been violated. That is the single most valuable fact in this report.

---

## 3. Drift from the goal

**Last 24 hours of builder work, each traced to a GOAL.md sentence:**

| work | GOAL.md sentence it serves |
|---|---|
| `XL.00` PASS (death, respawn independence, diary crossing death) | *"He lives, he dies, he remembers… Death is not a reset; it is a page turn."* |
| `PS.01` PASS (drive calibration; a statue loses) | *"Jack has the needs of a human… the needs ARE the curriculum."* |
| `PG.6`, `PG.9` PASS (the eye resolves; the view is not mostly obstacle) | *"sight"* — every sense a human has |
| `PG.7` PASS (heard-not-seen fixture leaks nothing) | *"what he hears can teach what he sees"* — the unison gate |
| `coverage.py` + PS.02/PS.03/SH.01/XL.01/BA.01 registered | *"too cold kills him"*, *"he builds a shelter"*, damage as learnable, balance |
| SM/TA/VO family registered (7 specs), `T0.20` PASS | *"EVERY SENSE A HUMAN HAS… Smell and taste are not ornaments"* |
| DP family registered | owner directive 2026-08-10, *"Fast and slow, in one brain"* |
| `ME.11` family, `ME.8` re-run at 3 seeds | *"Memory makes it him"* |
| `T0.17`/`T0.18` provenance, commit stamps, `+dirty`, `impl_sha` unification | *"protects the honesty of watching what happens"* |
| `LC.00`–`LC.02` PASS | *"a small trained core"*, the learning-core bakeoff |

**Zero drift.** Every item traces. I looked for work serving no GOAL.md sentence
and found none — the builder is not building the wrong things.

**The converse question is where the problem is.** Which parts of GOAL.md have
no passing spec at all:

- **`one brain / unison` — 0 of 7 passing.** This is not a branch; it is *"one
  interconnected brain"*, the sentence GOAL.md opens with, and the hard
  constraint in SYSTEM.md that *"no bakeoff can trade away"*. UB.2, UB.3, UB.6,
  UB.7 (the headline claim), UB.13, UB.14, DP.02 — none run.
- **`curiosity` — 1 of 6 passing, and the 1 does not say what it looks like it
  says.** The pass is `PG.4`, which certifies that a **noisy-TV trap exists in
  the world** and captures naive ICM. It is a world-fidelity certificate, not a
  claim about Jack wanting anything. `T2.08` (curiosity drives coverage),
  `T3.06` (ablate curiosity), `T5.06` (unprompted exploration is real), `CU.2`,
  `CU.5` — none run. Nothing on the ladder yet shows Jack **choosing** to do
  anything.
- **All-senses fusion** — 0 of 10 senses reach "load-bearing" under GOAL.md's
  own standard (*"ablate a sense, something measurable must degrade"*). Sight
  and hearing have sensor-level certificates only. `experiments/senses.py` still
  calls that `DEMONSTRATED`; the fourth status tier asked for at audit 4 has not
  landed. **Carried, fifth audit.**
- **`fast/slow` — 0 of 3.** `DP.00` ("does this world reward looking ahead at
  all") is CPU and runnable now; the owner set this direction two days ago.
- **`generality` — 0 of 4**, and its only spec with any status is `T1.02 =
  ERROR`.

**One specific neglect worth naming, because the mechanism is generalisable.**
`UB.14` — *"Cross-modal prediction, against the null that usually wins"* — is
`CPU_LONG`, depends on `PG.1` alone, and has been **runnable since 2026-08-09
14:22**: ~22 hours, ~22 builder iterations. `UB.9` joined it at 08:13 today.
Neither has been taken.

The reason is structural, not negligence: the builder takes work in `run
blocked` fan-out order, and **`UB.14` frees 0 specs, so it never ranks**. A
priority function that optimises unblocking power systematically defers claims
that are *leaves* — and GOAL.md's headline claims are leaves by construction,
because they are the destination rather than the road. `ladder_prompt.md` names
`UB.9` at priority `0aa` and `LC.03` at `0`; it names no unison spec that frees
nothing. That is how a constitution gets deferred by a machine that is working
correctly.

---

## 4. Is the builder alive and productive? — YES, and it is doing good work

24 iterations started in the 24 h to 12:37. **22 recorded an end: 21 `rc=0`, 1
`rc=1`** (10:07, `Reached max turns (120)` — it still committed useful repairs).
The two with no end line (08-09 17:07 and 22:07) predate the `EXIT` trap, which
landed this morning and is verified firing under SIGTERM. **PASS delta +18
(43 → 61).**

Quality is high and self-critical. Three examples from the log that are the
system working as designed:

- The 11:07 iteration found that `_check` sees the **mean over seeds**, so none
  of XL.00's five control gates was per-seed — and reconstructed the hidden
  per-seed values `{2e-5, 2e-5, 2.62e-3}` from the recorded mean and std to
  prove two seeds at the permutation floor were carrying a third that was 2.6×
  over its own gate. It also ran **both** candidate repairs on the same data
  instead of choosing one, and reported that the elegant fix (Spearman) **lost**.
- The 12:07 iteration found `impl_sha` had **two implementations that
  disagreed** — writer hashed file + `IMPL_DEPS`, reader hashed the file alone —
  so all twelve specs declaring `IMPL_DEPS` were flagged stale in perpetuity and
  **no re-run could clear the flag**. It fixed by deletion, not by sync, and
  proved it on a real case (LC.02's flag cleared).
- The 11:07 iteration killed its own in-flight re-run rather than let it land
  with a stamp naming code that did not run, disclosed the judgment call, and
  deliberately did not relaunch so the next iteration would start unambiguous.

**Credit exhaustion: 24 of the last 24 iterations opened `OUT OF CREDITS on
fable — falling back to opus`.** `crontab` still reads `JACK_LOOP_MODEL=fable`.
The fallback works and no iteration has been lost to it, so the cost is not
capability — it is that **`iteration start` logs a model the iteration will not
use**. This project spent two commits this week fixing exactly that class of
defect for the ledger (a stamp that names code which did not run). The loop's
own log has the same bug. **Fourth audit asking; one word in one crontab line.**

`review.sh` (`37 6 * * *`) still collides with `overseer.sh` (`37 */6 * * *`)
every morning — two unattended agents sharing one `index.lock`. Third audit.

---

## 5. Compute honesty

```
2026-W32 (Sun 09 – Sat 15, the live week):  kaggle 11.9635 h   colab 0.0015 h
                                            remaining 18.0365 h, expiring 08-16
2026-W31:                                   kaggle 37.4554 h   colab 7.7461 h
overruns recorded: []
```

**The meter is honest and reconciles.** W32's 11.96 h maps to exactly two ledger
entries: `T2.02` (6.28 h P100, 2026-08-09T07:30) and `T2.01` (5.58 h, charged as
job `jack-ladder-1786304547`, 2026-08-10T01:17). No unaccounted hours this week.

**What those 11.96 hours bought: one VOID and one FAIL.** That is not waste —
`T2.02` refusing to arbitrate and `T2.01` measuring 1.19σ against a 5σ bar are
both real measurements, and a FAIL that tells the truth is the point of this
system. But it should be stated plainly rather than averaged away: **the
project's entire GPU spend this week produced zero PASS entries, and every one
of the 64 ledger entries carries `hardware: aarch64/Linux/torch2.8.0+cpu/cpu`.**

W31's 37.46 h against a 30.0 h ceiling with `overruns: []` is the **already-known
and already-fixed** gap — the overrun recorder was written afterwards
(`gpu.py:293`) and the ceiling is documented in the code comment as the
motivating failure. No new finding. It has still never been reconciled against
Kaggle's own reported runtime.

**The live problem is the opposite of overspend.** D3 was answered YES today,
`origin/main..HEAD` is **empty** (everything pushed), `assert_ref_is_current`
will now pass, and **18.04 free hours expire on 2026-08-16 — unspent free quota
is not saved, it is lost.** The last GPU submission landed at 01:17. Eleven
iterations have run since, all CPU. Ten GPU specs are runnable right now,
including:

- **`T1.02`** (`gpu<20min`) — **ERROR since 2026-08-08T22:07**, and the recorded
  cause is infrastructure, not science: `"kaggle: 0.0h left, need 0.7h"` plus a
  Colab session that vanished. It is a Tier-1 spec, it is one of only four specs
  behind `generality`, and it is `run next`'s first entry. **0.7 h clears a
  two-day-old ERROR.**
- **`T2.01`** (`gpu<8h`) — `run blocked`'s #1 terminal blocker: **frees 26,
  blocks 36**, including all seven curiosity specs.

**69 of 160 specs (43%) are unreachable.** Terminal blockers: `T2.01=FAIL`
(frees 26), `UB.9=NOT_RUN` (frees 4), `LC.03=NOT_RUN` (frees 4, unblocked at
12:35 today by XL.00), `T2.08` (frees 3), `T2.06` (frees 3), `T2.03` (frees 2),
`T2.02=VOID` (frees 2).

---

## 6. Stuck decisions

**Nothing was acted on without being recorded.** I checked the converse
specifically: D3 was answered by the owner on 2026-08-10 and is recorded with
the tradeoff (public repo → pushing publishes) *and* the failure mode if the
repo is ever made private (`build_job` clones with no credentials). That is
SYSTEM.md's rule followed properly.

**Open and now decidable:**

- **D1** (does the 57M trunk stay in the control path) is correctly frozen by
  its own 2026-08-09 CORRECTION — the evidence is confounded by live dropout and
  16× fewer optimiser steps, and the file says so in bold. It should **not** be
  decided. But its option set is still stale in a second way already raised at
  audit 4: **option A ("freeze the trunk") contradicts the PLASTIC-ONLY decree**
  the owner issued on 2026-08-09. An owner reading D1 today is being offered a
  recommended option their own constitutional directive forbids.
- **D2** (does a VOID block its dependents) is worth 2 specs now, not 4 and not
  40 — `T2.02=VOID` frees `T2.13` and `T5.09`. It is no longer urgent and the
  file should stop implying it is.
- **Two entries at the top of `DECISIONS_NEEDED.md` are still false.** *"Kaggle
  GPU is not being granted"* claims to block `T0.10`/`T0.11`, both PASS since
  2026-08-04, and Kaggle has been the primary backend all week. *"/data is 95%
  full"* is marked OPEN; `df` reads **21 GB of 100 GB, 21%**. Both corrections
  exist — 700 lines further down, in a HOUSEKEEPING block added at audit 4. The
  corrections are honest; their placement means a reader still meets two false
  statements first. **Fifth audit asking. Only the owner may strike them.**

**Could anything blocked have been resolved by a bakeoff instead?** I checked
D1 and D2. D2 cannot be — there is no metric, only a choice about what the
ladder means, and the file says so correctly. D1 **can** be, and the file
already names the instrument (`T2.21`, ~6.3 GPU-h). It is waiting on GPU, not on
the owner.

---

## 7. Bakeoff hygiene — CLEAN

`docs/DECISIONS_RESOLVED.md` holds two entries, both from PS.01/J.

- **PS.01/J → VOID.** Three arms below the 3.0σ learning gate. Correct: the gate
  is T2.02's invention and it fired exactly as designed. **A VOID was not
  treated as a verdict** — the bakeoff was re-run as J2 after the arms were
  fixed, rather than a winner being read out of a broken field.
- **PS.01/J2 → WINNER `impact_speed`.** 0.973 AUC, 10.32σ over the null, beating
  the runner-up `peak_dvel` by **2.66σ against a pre-registered 1.5 margin** —
  outside the noise band, not inside it. Eleven eliminated arms are recorded by
  name.
- **The `screen` gate mode** is structurally a relaxation and I scrutinised it
  again. It is properly fenced: the gate *value* is unchanged, the rationale is
  on the committed record, and the justification is sound — every arm is a
  deterministic reduction of the **same memoised rollouts**, so there is no
  training that could have failed and a low score is the arm's own property. The
  T2.02 ambiguity the learning gate protects against (broken run, or worse
  architecture?) genuinely cannot arise here. `T0.19` gates the mode.

**No decision made without a learning gate. No VOID treated as a verdict. No
winner chosen inside the noise margin.** The nine `TEST` fixtures that once
polluted this file are gone and the write path is injectable so tests cannot
reach it again.

---

## 8. The honest summary — are we closer to a creature, or to a longer list of ticks?

**Both, and the honest accounting is that the list grew faster than the
creature.** In 24 hours the ladder went **43 → 61 PASS (+18)** while
**registered specs went 123 → 160 (+37)**. The demonstrated fraction improved
slightly (35.0% → 38.1%), and most of the +37 is the deliberate registration of
constitutional holes that had nothing behind them, which is exactly right. But
41 of the 61 PASSes are harness, world-fidelity and memory-substrate
certificates. They are the honest floor this project needs; they are not Jack.

**Genuinely closer, in one specific and important way.** `XL.00` means that
death, a respawn statistically independent of where he died, and a diary that
survives the death are now **measured facts with five positive controls firing**,
not design intent. Thirty-six hours ago *"He lives, he dies, he remembers"* — an
entire GOAL.md section and the owner's central directive — had **zero**
falsifiable claims behind it. It now has one, it holds at 3 seeds, and it
unblocks `LC.03`, the bakeoff that decides **how Jack learns**. That is real
movement toward the creature and not toward the scoreboard.

**Not closer on the axis GOAL.md leads with.** Curiosity's only PASS certifies
that a trap exists in the world, not that Jack wants anything. Unison is 0 of 7.
No sense is load-bearing by GOAL.md's own ablation standard. And the
ladder-and-apple sentence — *"climbing the ladder on attempt 40 after falling on
attempts 1–39, without anyone telling him to"* — has **no spec that attempts
it**. `PG.3` certifies the ladder is climbable by a **scripted** policy with
adhesion hands. That is a fact about the world, not about him. The gap between
"the ladder can be climbed" and "he climbed it because he wanted to" is the
entire project, and nothing on the board currently reaches across it.

**And the instrument that is supposed to notice all of this is currently
flattering us** — three commitments reporting a green PASS from the word
"honest", "climbable" and a contact-audio test. That is why the verdict is
DRIFTING and not ON TRACK: not because the ledger lies (it does not, and
`run verify` proves it), but because the map from the ledger to the owner's
commitments has developed exactly the blind spot it was built to eliminate.

---

## FOR THE BUILDER

Ranked by damage to the trustworthiness of the ledger and its map. None needs
the owner.

1. **Make `COVERS:` load-bearing in `coverage.py` (§1a). NEW, RANK 1.** Count a
   spec toward `n_specs`/`n_pass` **only** when it declares
   `COVERS: <commitment>` in its notes. Print regex hits in a **separate
   NOMINATIONS column** — so an undeclared-but-plausible spec reads as *work to
   do*, never as coverage. Do **not** tune the patterns; the file's own docstring
   is right that a detector edited until it agrees with you measures its
   maintainer's patience. Two supporting changes:
   (a) add `\b` word boundaries as a cheap partial — it kills `honest`→`nest`,
   `climbable`→`limb`, `bodies`→`dies`, but note it does **not** fix
   "physically sound", "curiosity drives", or "damage is a signal", which is
   itself the argument for (1);
   (b) add a **known-answer test**, per `LESSONS.md:1673` (*"feed it the case you
   already know is broken"*): a synthetic spec titled *"The honest baseline"*
   must NOT count toward `shelter/building`, and a synthetic spec declaring
   `COVERS: shelter/building` with an unrelated title MUST count.
   Then backfill the declarations, starting with the ones this audit found:
   `VO.01` → `voice`; `XL.00` → `memory across lives`; `T3.02` →
   `proprioception`; `UB.5` → `touch/contact`; `SH.01` → `shelter/building`.

2. **Give `_calibration()` a freshness check (§1b). NEW, RANK 2.**
   `xl_00_death_and_respawn.py:151` returns PS.01's `j0`/`alpha` whenever the
   entry is PASS. Have it also return `(None, None)` → `Status.VOID` when
   `run.stale_claims` names PS.01, and **record the source entry's `impl_sha`
   in XL.00's own metrics** so the provenance of a consumed constant is in the
   record rather than in a docstring. This generalises: any spec reading another
   spec's ledger numbers needs the same guard, and a `T0.17` property asserting
   it would make the class impossible rather than fixed.

3. **Clear the three definitional stale flags — `PG.3`, `PG.8`, `PS.01`.** All
   CPU, all fast, and `PS.01` is the one XL.00 and the whole LC family consume.
   Their last recorded durations are short; this is cheap and it removes the
   only three entries on the board that a reader cannot distinguish from real
   debt.

4. **Re-run `T1.02` (§5).** ERROR since 2026-08-08T22:07 on `"kaggle: 0.0h left,
   need 0.7h"` — an infrastructure error, not a scientific one, and quota is now
   18.04 h with the push block gone. It is `run next`'s first entry and one of
   only four specs behind `generality`. An ERROR is not a verdict and it has been
   carried for two days.

5. **Take a unison spec that frees nothing (§3).** `UB.14` is `CPU_LONG`,
   depends only on `PG.1`, and has been runnable for 22 hours through ~22
   iterations. It frees 0 specs, which is precisely why the fan-out ranking has
   never surfaced it — and `one brain / unison` is 0 of 7 with SYSTEM.md calling
   it the one thing no bakeoff may trade away. Consider adding to
   `ladder_prompt.md` a standing rule that when a GOAL.md commitment has **zero**
   passing specs, its cheapest runnable member outranks fan-out.

6. **Backfill `impl_sha` on the 41 pre-`impl_sha` PASSes (§1).** 67% of the
   ledger cannot be staleness-checked. 36 of the 41 are CPU and the whole set
   sums to 234 minutes of previously-measured runtime — this is a background
   sweep, not a project. It converts the largest blind spot in the ledger's
   self-knowledge into a checkable property.

7. **Add the fourth status tier to `experiments/senses.py` (§3). Fifth audit.**
   `DEMONSTRATED` currently means "some declared spec is PASS", so `PG.6` — a
   ridge probe whose own docstring says it certifies *the sensor, not the net* —
   marks sight `[PASS]`. Proposed:
   `ABSENT → REGISTERED → SENSOR → LOAD-BEARING (a UB ablation degrades a
   measured quantity)`. Under it today **0 of 10 senses read LOAD-BEARING**,
   which is what GOAL.md actually asks for. Extend `T0.20` so a sense cannot
   reach LOAD-BEARING without a passing UB-family spec.

8. **Give `T0.10` a control (§1).** The last remaining PASS whose gate has never
   been shown able to report the bad case, and the one where it matters: this
   project has already seen a Kaggle kernel run to `COMPLETE` with no GPU
   attached and nothing signal a problem.

9. **`JACK_LOOP_MODEL=opus` in crontab (§4). Fourth audit.** 24 of the last 24
   iterations open `OUT OF CREDITS on fable`. The fallback should not be
   load-bearing every hour, and `iteration start` currently logs a model the
   iteration will not use — the same defect class this project spent two commits
   fixing for the ledger's commit stamp.

10. **Move `review.sh` off minute 37 (§4). Third audit.** It collides with
    `overseer.sh` every morning; two unattended agents share one `index.lock`.

---

## FOR THE OWNER

1. **Two false statements are still the first thing anyone reads in
   `DECISIONS_NEEDED.md`, and only you can strike them. Fifth audit asking.**
   *"Kaggle GPU is not being granted"* says it blocks `T0.10` and `T0.11` —
   **both PASS since 2026-08-04**, and Kaggle has been the primary GPU backend
   all week (11.96 h billed). *"/data is 95% full"* is marked OPEN; `df` now
   reads **21 GB of 100 GB (21%)**. Corrections for both are already drafted 700
   lines down in the same file; the loop is not permitted to strike an
   owner-decision block itself. **One line from you clears both.**

2. **The Kaggle quota is the one resource that expires, and it is now unblocked
   and unspent.** You answered D3 today and it worked: everything is pushed,
   `assert_ref_is_current` passes, and the entire GPU half of the ladder is
   available for the first time. **18.04 of 30 hours expire 2026-08-16.** No
   GPU job has been submitted in the 11 hours since. No action is required from
   you — this is a builder item (**FOR THE BUILDER §4**) and I have logged it
   there. You should simply know that the thing you unblocked has not yet been
   used, and that the deadline is Sunday.

3. **D1's recommended option contradicts your own decree, and it should not be
   decided in its current form.** D1 offers option A — *"freeze the trunk; a
   small dedicated policy head does control"* — as RECOMMENDED. On 2026-08-09
   you issued the PLASTIC-ONLY decree: *"Every component inside Jack learns…
   a frozen tower's reshaping gain is identically zero."* Option A is
   constitutionally unavailable. Separately, D1's evidence is already frozen by
   its own 2026-08-09 correction (live dropout injecting 42% action noise into
   one arm and not the other; 16× fewer optimiser steps on a 457× larger model),
   so the comparison must be re-run before any option is chosen. **The honest
   ask: nothing, yet — but the option set needs rewriting before it reaches you
   again, and the loop should not present you a choice your own directive
   forbids.**

4. **The scope question from audit 4 is still open and still yours.** Smell,
   taste and voice are registered and CPU-runnable (`SM.01`, `TA.01`, `VO.01`).
   Temperature and pain are not, deliberately: temperature arrives only with the
   whole survival world (thermal ODE, shelter, occlusion) and pain is an open
   bakeoff arm that the research itself calls *"a live question, not a settled
   design"*. **Schedule the W family — temperature, and with it shelter, the
   only mechanism in the design that teaches construction — now, or after the LC
   bakeoff?** Both currently read `ABSENT` in `run senses`, so the hole is
   visible rather than invisible; it is merely open.

5. **For information: the integrity guarantee holds, and this time all four
   carried items closed.** `run verify` re-judged 60 PASS entries from the
   record through their committed gates — 60/60 re-derive, 58 controls probed, 0
   ignored, 0 declared-but-unrun, 0 unreplayable. Undeclared controls went 19 →
   **0**. T1.03 and T1.05 got the controls asked for across four audits. ME.8
   re-ran at 3 seeds. Commit stamps now name the code that ran, and flag
   `+dirty` when the tree was not clean. Across the project's entire history:
   **no threshold has ever been moved in the loosening direction, no control
   weakened, no assertion removed.** Five audits running. On the integrity axis
   this system does exactly what it was built to do.

6. **The one thing I would want you to read in this report.** `XL.00` passing
   means Jack can now die, reappear somewhere he did not choose, and keep his
   diary — measured, with five positive controls firing, at 3 seeds. Thirty-six
   hours ago that entire section of GOAL.md had nothing behind it. Against that:
   **nothing on the board yet shows Jack choosing to do anything.** Curiosity's
   only PASS certifies that a trap exists in the world; unison is 0 of 7; and
   the ladder-and-apple sentence you wrote the project around has no spec that
   attempts it. That is not a criticism of the last day's work, which was good
   and honest. It is the answer to *"closer to a creature, or just busier?"* —
   **closer, on mortality and memory; not yet started, on wanting.**

---

*Ledger untouched. No experiment re-run — `run status`, `run next`, `run
blocked`, `run stale`, `run verify` and `python -m experiments.coverage` are
read-only re-judgements of the existing record. Nothing outside
`/home/opc/jackthelearner` changed. No container or daemon touched. Tree was
clean at audit start (`90934f1`) and this commit contains only `OVERSIGHT.md`
and `LESSONS.md`. Nothing was appended to `DECISIONS_NEEDED.md`: every owner
item in this report is already recorded there from audits 3 and 4, and a fifth
restatement would bury the asks rather than sharpen them.*
