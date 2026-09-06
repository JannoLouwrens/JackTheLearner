# OVERSIGHT — 78th audit, 2026-09-06 12:33–12:5x UTC (at `605b273`, tree dirty in 3 files)

## VERDICT: INTEGRITY RISK — **35 of the ledger's 105 PASS certificates cannot go stale when the component they test is rewritten**, and today one of those rewrites happened. `impl_sha` hashes the test file plus whatever the test *declares* in `IMPL_DEPS`; **54 of 105 PASS specs declare nothing at all**, and 35 of those import a repo-root implementation module by name. So `run status`'s STALE-CLAIMS lane printed a clean board this morning — 0 stale PASS rows — while `EpisodicMemory.py`'s similarity scorer was replaced under **eight** PASS certificates and only one of them (the repair target) was re-run. The other seven now assert a result about a scorer that no longer exists, and no instrument in this repo can say so.

**This is not drift and nobody cheated.** Section 2 is clean: I looked hard and found **no loosening anywhere in seven days** — the ME.1 repair moved a module calibration and left the spec's 0.95 bar and its exclusion denominator provably untouched, the D1.0 gate got strictly harder, LG.01's retention band was *tightened* 0.25 → 0.0. The science this week was unusually honest. The risk is structural: a third of the scoreboard is blind to its own subject, the blindness is invisible to every organ, and **the corrective note the builder wrote about it today reproduces the exact blind spot** (§1.3).

Ranked below by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger — ONE STRUCTURAL FINDING, and it fired today

### 1.0 The mechanical sweep is clean (seventh consecutive audit)

All **105 PASS** rows of **142** recorded (**244** registered), resolved mechanically:

- **0** PASS rows whose recorded `commit` fails `git cat-file -e <sha>^{commit}`.
- **0** PASS rows with no spec in `BY_ID`; **0** with no implementation file.
- **0** PASS rows whose spec declares no `control`.
- **103 / 105** carry populated `control_metrics`. The two that do not — `T0.01`,
  `T0.10` — declare `control = "NONE, BY DECISION (52nd audit B5)"` with the
  reason on the spec. Pre-registered refusals, not omissions.

Status distribution: **105 PASS / 23 FAIL / 14 VOID**.

### 1.1 THE FINDING — `impl_sha` covers the test, not the code under test, for a third of the ledger

`protocol.impl_sha_of` is documented as *"sha256 of the test file … PLUS any
files the test module declares in `IMPL_DEPS`, because a test file alone is not
the code under test."* The mechanism is correct. Its **inputs** are not:

| | count |
|---|---|
| PASS specs whose test declares an `IMPL_DEPS` list | **51** |
| PASS specs whose test declares **nothing** | **54** |
| PASS specs importing a repo-root `*.py` implementation module **not** in their `IMPL_DEPS` | **35** |
| PASS rows `run status` reports as mechanically stale | **0** |

The 35, grouped by the module they test and do not declare:

- **`UnifiedBrain.py` — Jack's core — 12 specs:** `T0.03`, `T0.04`, `T0.07`,
  `T1.01`, `T1.02`, `T1.03`, `T1.04`, `T1.05`, `T1.06`, `T1.07`, `T1.08`,
  `T1.09`, `T1.10`, `T1.11`, `T1.12`, `T6.03`. *Rewrite the core and every
  Tier-1 certificate stays green by construction.*
- **`EpisodicMemory.py` — 8 specs:** `ME.1`, `ME.3`, `ME.4`, `ME.5`, `ME.9`,
  `ME.10`, `T2.20`, `XL.00`.
- **`TrainingPipeline.py` — 5:** `T0.14`, `T0.16`, `T0.25`, `T2.00`, `PG.8`.
- Singles: `OwnerProfile.py` (`ME.2`), `WorkingMemory.py` (`ME.8`),
  `Reflections.py`/`Forgetting.py` (`ME.3`/`ME.4`), `ContactAudio.py` (`PG.7`),
  `VirtualWorld.py` (`T0.06`), `EmotionalState.py` (`T2.12`),
  `MoCapLoader.py` (`T1.13`), `Persistence.py` (`T6.03`).

There is **no guard spec for this**. `T0.17` checks ledger provenance, `T0.27`
checks moved thresholds — nothing asks whether a test declares the code it
imports.

### 1.2 It is not hypothetical. It fired at 12:16 today.

Commit `6502d36` replaced `EpisodicMemory.recall`'s scoring rule: `abstain_below`
0.34 → **0.95**, raw containment `|q∩e|/|q|` → **coverage over the cue's KNOWN
words**, plus a new `_vocab` set and an early `return []` for all-unknown cues.
That is a different retrieval function, and the builder knew it — the probe that
chose it (`me1_floor_probe.py`) measured **ME.9's scar and ME.11's scar
explicitly** on candidate arms.

Of the eight PASS specs that import it, **one** (`ME.1`, the repair target) was
re-run. The other seven still carry rows from before the rewrite:

| spec | status | last ran | commit |
|---|---|---|---|
| `ME.3` reflections | PASS | 2026-09-02T18:42 | `c7325c2` |
| `ME.4` forgetting | PASS | 2026-09-02T18:42 | `c7325c2` |
| `ME.5` retrieval at scale | PASS | 2026-09-04T00:18 | `b1df9f6` |
| **`ME.9` attributed recall** | PASS | 2026-09-02T18:44 | `c7325c2` |
| `ME.10` diary vs skill | PASS | 2026-09-02T18:44 | `c7325c2` |
| `ME.11.A` lexical incumbent | PASS | 2026-09-02T18:45 | `c7325c2` |
| `T2.20` episodic search | PASS | 2026-09-04T00:17 | `b1df9f6` |
| `XL.00` death & respawn | PASS | 2026-09-04T03:37 | `0d8a31c` |

**`ME.9` is named by id in `GOAL.md`** (*"ME.9 (attributed recall of heard/said/
did)"*). `ME.5` is *retrieval at scale* — a retrieval claim whose retrieval
function changed four hours ago. Their `impl_sha` values are unmoved because
their test files are unmoved, which is precisely the defect.

The six specs that **did** get re-bought (`LG.00/01/02`, `SO.08`, `LF.02`, and
by name `LG.10`/`SO.07`/`LF.01`) were re-bought because they *declare*
`EpisodicMemory.py`. The machinery worked perfectly for the half of the ladder
that opted in.

### 1.3 The correction written today encodes the same blind spot

The uncommitted edit to `docs/REVIEW_QUEUE.md` (+513, `me1-similarity-floor-
never-abstains`) honourably corrects the row's "Staleness bill: NONE mechanical"
and prescribes the fix:

> `grep -rl IMPL_DEPS experiments/tests/ | xargs grep -l EpisodicMemory.py`

**That grep filters to files which already declare `IMPL_DEPS`.** All seven
missed ME-family files declare none, so they are removed by the first stage of
the pipeline before the second stage can see them. Run today, it returns exactly
the eight the builder already found and none of the seven it missed. The lesson
written to prevent the recurrence cannot detect the recurrence.

This is the 77th audit's own lesson one layer out — *a rule asks a fallible
reader to be careful; an instrument asks nothing*. The instrument here is cheap
and stated in B1.

---

## 2. Thresholds and controls over 7 days — CLEAN. No finding.

I went looking for silent loosening and did not find any. Everything that moved
moved the hard way:

- **`ME.1`'s repair obeyed its explicit prohibition, and this is checkable from
  the ledger rather than the commit message.** The Review's FOR THE BUILDER said
  *"do not repair it by widening ME.1's exclusion filter or lowering 0.95"*.
  Attempt 6 (FAIL, `35b9d51`) and attempt 7 (PASS, `6502d36`) read **identical
  denominators** — `distractor_evaluated` 40.0 ± 4.546061 and
  `distractor_excluded` 20.0 ± 4.546061 in both — so no cue was excluded to buy
  the pass; `cued_recall` is byte-identical at 0.85 ± 0.0136355; the 0.95 bar is
  untouched; `me_1_event_log.py` is not in the commit's diffstat at all. The
  calibration moved inside the module, which is where the FTB said to put it.
- **`D1.0`'s learning gate got strictly harder** (`8f2990d`): vs-random 3.0σ →
  paired-per-seed against each arm's **own untrained twin** at the same unmoved
  3.0σ, plus a consistency conjunct, plus an SB3 reference lane that VOIDs the
  run as a harness fault. Verified red-first — all three adversarial fixtures
  passed the old gate and each now hits its named conjunct.
- **`LG.01` tightened**: `CHANCE_BAND_HI` 0.25 → **0.0** (retain a probe only if
  the LLM null is outright wrong, not merely at chance).
- **`T1.01` gained conjuncts**, none removed: `mode_training` asserted on both
  arms, frozen-control `improvement_ratio < 1.5`.
- The `_worst(m, k)` → `min(_per_seed(k))` conversions across `LG.01`, `LC.03`
  and others are the `aggregate-hides-worst-seed` repair: equal or harder.

No `_check` gained an `or`. No seed count fell. No control was deleted or
weakened. No `falsified_by` was narrowed.

---

## 3. Two ratchets went red in a commit that named neither

`run status` prints both, so nothing is hidden — but the rule is that *the
commit which grows a ratchet names the growth*, and `605b273` named the `LG.00`
VOID and neither of its two consequences.

- **`champions_trigger_debt` 3 → 5** (`champions --check` **EXIT 1**, baseline
  `BASELINE_TRIGGER_UNREACHABLE = 3`). The two new seats are **Language model**
  and **Language acquisition**, both TRIGGER-UNREACHABLE because their sole
  arena `LG.00` went VOID at 12:16.
- **`unreachable` 95 → 96** (`coverage` **EXIT 2**), above its declared floor.
  The floor was legitimately raised 94 → 95 in the ME.1 discharge commit
  `56742cc` with justification; nothing raised it to 96.

**And the 09:07 slot's recorded inference was wrong.** It journalled the 08:07
`97 vs 95` flag as *"reads 95-at-floor at head — most likely T1.01's 08:43 PASS
restored two dependents. I recorded the inference as an inference."* Recording it
as an inference was the right conduct; the inference did not hold. It reads
**96** now, and the arithmetic points at `LG.00`'s VOID blocking `LG.11`
(*told world*), not at `T1.01`.

**Fair mitigation, stated:** the builder declared this pending — *"LG.00 re-runs
when it lands"* — and the detached `--llm-pass` job **completed at 12:32**
(`/data/lg00_llm_verdicts.json`, 4227 verdicts, 1590 pairs). The 13:07 slot can
clear both ratchets. The finding is that a commit turned two ratchets red
without saying so, not that the state is permanent.

---

## 4. `GOAL.md`'s existential claim currently has no passing spec

`LG.00` — *"Jack knows what his LLM cannot — he is not a puppet"* — is **VOID**
as of 12:16 (attempt 5). `GOAL.md` names it by id: *"Falsifiable as LG.00: strip
the diary and the learned core, and his answers about his own life must
COLLAPSE."* It is also the **only** arena of two champion seats.

The VOID is honest and self-inflicted by design — `verdicts_missing` went 0 →
623.7/seed because the verdict cache keys hash the exact prompt, and the
recalibrated recall changed Jack's retrieved context. The spec's own docstring
promises exactly this. The re-run is queued.

**The durable point is not today's VOID.** It is that the project's single
anti-costume claim **self-VOIDs on any change to Jack's memory**, and its repair
costs a ~14-minute out-of-process LLM recompute that (§5) breaches the box's
memory ceiling. That coupling is why both language seats carry trigger debt: the
only thing that could unseat the LLM decree is the one spec that breaks whenever
Jack's memory improves. Worth a design note on the `lg10-mouth-fidelity-vs-
freedom` row; not something I will prescribe.

---

## 5. CONDUCT — the memory ceiling was breached, on a box with paying tenants

`SYSTEM.md` hard constraint: *"Stay at nice 19, under ~1.5 GB RAM… This box
serves paying tenants."* CONDUCT is class 3 — fixed, and not up for measurement.

At **12:25:52** the watchdog named it:

```
MEMORY 1830395 — peak rss 2424 MB (VmHWM) over the 1536 MB ceiling, 856s CPU,
cmd: .../python -m experiments.tests.lg_00_not_a_puppet --llm-pass
```

**2424 MB is 58% over the ceiling**, sustained for ~14 minutes, alongside
`worldtwin` (302 MB), four tenant agents and the Kaggle watcher. Separately,
`T1.01`'s own ledger row records `peak_rss_mb = 1532.2` — **4 MB under** the
same ceiling.

Nothing refused either run, and that is **correct**: `D18`'s armed default is
*measure and report, gate nothing, relax nothing*, and it fired as written. So
this is the instrument working, producing the first real breach it has caught.
It is evidence on the owner's open `D18` (decide_by **2026-09-09**) and I have
appended it there rather than acting on it.

---

## 6. `DP.04`'s module has not compiled for 7 days, and nothing notices

The 11:07 slot flagged this for me; I confirmed it:

```
compile(dp_04_slow_path_verbal.py) →
SyntaxError: from __future__ imports must occur at the beginning of the file (line 391)
```

`ast.parse` succeeds, so any static reader passes it; only `compile`/`import`
fails. `DP.04` is **registered**, is a member of the **Deliberation** seat's
arena, and `coverage` lists it under *fast/slow* as FORECLOSED. `W1.02` had to
read `MIN_GAIN = 5.0` out of its **source text** this morning because the module
cannot be imported. Broken since 2026-08-30.

Nothing catches it because `run.module_path_for(strict=True)` deliberately
avoids importing test modules when listing (a correct design, for a good
reason). The cost is that a registered spec can be uncompilable indefinitely and
appear healthy in every listing. Routed as B6.

---

## 7. Drift from the goal — none in the last day; the standing hole is unmoved

**Every unit the builder shipped today traces to a `GOAL.md` sentence:**

| work | `GOAL.md` sentence it serves |
|---|---|
| `W1.00` (FAIL, informative) + `W1.02` (PASS) | *"The world must be **consistent**, **discoverable** and **consequential**"* — W1.02 replaces a censored lifespan that resolved ~20 of 96 lives with a graded outcome resolving 96/96, at zero new constants |
| `EpisodicMemory` floor repair + `ME.1` attempt 7 | *"Memory makes it him… He remembers the ladder. He remembers you"* |
| adopted `D1.0` gate, executed | law 1 — a capability is claimed only by a test that could have failed |
| `T1.01` re-buy under strengthening | same |
| `LG.00`/`LG.01`/`LG.02`/`SO.08`/`LF.02` re-buys | certificate hygiene; the ledger is the only scoreboard |

**No drift.** `W1.02` in particular is the most goal-serving thing in the log:
it says the *outcome channel itself* was the shallow part, which is a finding
about Jack's world rather than about our instruments.

**The converse is where it hurts, and it did not move.** `coverage` reports:

- **4 CLAIM-DEAD commitments** — *smell*, *balance*, *shelter/building*,
  *thermal (kills)* — every claim spec parked or foreclosed. Unchanged since
  2026-09-03. Three of these are the owner's own words (*"too cold/hot KILLS
  him"*, *"every sense a human has"*, *"owner's own image of success"*).
- **9 commitments with live claim specs and nothing passing**: *touch*,
  *tool use*, *told world*, *proprioception*, *death & retry*, *plasticity*,
  *sleep*, *hunger/thirst*, *fast/slow*.
- **`goal_unrunnable = 7`** — `GOAL.md` cites `DP.02`, `DP.03`, `GEN.02`,
  `GEN.03`, `GEN.06`, `GEN.09`, `LC.04`; every one resolves to a parked,
  foreclosed or welded spec, so the citation's present tense is false. Four are
  newly so. Routed (`goal-cites-four-specs-that-resolve-to-corpses`, DUE 09-10).
- **`GENERALITY.md`'s fourteen barriers**: ten still have no spec, and **zero
  have a passing one.** The Review found this yesterday. Nothing was registered
  against it today, and no instrument in this repo will ever raise it — the same
  shape as the 2026-08-10 miss that `coverage.py` was built for, one document
  over.

Curiosity: 12 specs, **2 passing, 0 runnable today**. All-senses fusion
(*one brain / unison*): 25 specs, **1 passing**. Learning-by-living
(*death & retry*): 4 specs, **0 passing**, `XL.01` FAIL for 18 days. These are
the three the audit brief names as most likely to be quietly neglected, and all
three are the ones standing still.

---

## 8. Builder liveness and productivity — HEALTHY

**13 iterations** 00:07 → 12:07, **13 ended `rc=0`**, `lost_iterations.log` 0
bytes, no undeclared processes, load 0.27.

Demonstrated **105 → 105** net, and the flat number is the least informative
thing about the day. Inside it: `W1.02` PASS (fresh, first-attempt), `ME.1`
FAIL → PASS with the confabulation genuinely repaired, `W1.00` FAIL on its
informative branch, `LG.00` PASS → VOID, `T1.01` re-bought under a
strengthening, five certificates re-bought after the floor edit, three armed
defaults fired at 00:14–00:16, the adopted `D1.0` gate implemented red-first and
attempt 2 dispatched. **Two fresh dispatches — the first since `SO.08` — both
harvested the same day.** Five consecutive verified-empty slots before the FULL,
each of which correctly stopped early rather than inventing a unit.

The Review's Sunday FULL **died `rc=124`** at 07:17 (the fifth Sunday FULL death
of five scheduled), but it had committed every disposition as it made it, so its
design survived its own death — and the seal fired correctly on `PROGRESS.md`.
Field watch last ran 2026-08-31 (Mondays; next fire 09-07, inside cadence).

**Owed and not yet done** (both still inside their day, noted as debt not
breach): the Review's FTB item 3 — add the distractor conjunct to `ME.3`,
`ME.5`, `ME.9`, `ME.10` *in the same run* as the floor repair — was not done
(grep: 0 occurrences of `distractor` in `me_3`/`me_9`/`me_10`, 1 in `me_5`); and
FTB item 8, the `T0.11` re-buy, is untouched at **33 days**, still the oldest
live certificate in the ledger.

---

## 9. Compute honesty — CLEAN

**GPU.** W36 (resets Sunday) charged **1.5412 h** of 30 — the `D1.0` attempt-2
reference kernel `jack-ladder-1788682804`, `ok: true`, and it bought a real
result (it cleared its 450-return floor and released the arm phase). ~28.46 h
remain; the arm phase is in flight (est 6.04 h, pids 1775588/1775608 alive and
declared). **Every W36 GPU-hour has a ledger consequence.** No orphaned kernels.

**CPU.** 2026-09-06 used **4619.41 s of 57600**. The `by_spec` map sums to
`used_s` to the centisecond — `T1.01` 2083.69, `W1.02` 636.03, `W1.00` 548.13,
`detached:lg00_llm_pass.log` 838.72 (declared), `LF.02` 268.84, the rest
instruments. **No unattributed spend.** `overruns: []`.

`cpu<2h` is foreclosed for the remainder of the day (slack 3600 s against 4619 s
spent; 39 of 60 specs unaffordable until midnight). That is `D20`'s wall-vs-core
arithmetic arriving exactly as routed, not a new fault.

---

## 10. Stuck decisions and bakeoff hygiene — no new finding

`decisions --check` **EXIT 0**: 7 armed, 0 `MEANS-ESCALATED`, 0 `UNDECLARED`,
0 overdue, 0 unrouted owner-asks, 0 vanished. `D24` correctly matched to
`PROGRESS` FOR-THE-OWNER item 1. Nothing on the owner's desk that a measurement
could settle; nothing acted on without being recorded (`D21`/`D16`/`D15` each
have a `DECISIONS_RESOLVED.md` entry with its reversal named).

`DECISIONS_RESOLVED.md`: the one entry that treats a VOID as a verdict — `D10`
seating wm-latent **BY VERDICT off `LC.03` (VOID)** — carries its single-arm
caveat on its face and is already the standing `VERDICT-IS-A-VOID` red in
`champions`, routed. No winner chosen inside a noise margin.

**The queue's own arithmetic is the thing to watch, and it is `D22`'s stated
price arriving on schedule:** `review-queue` **EXIT 0, 0 violations**, but
**40 live rows**, arrivals **5.29/cycle** against disposals **0.29/cycle**,
`drain` **UNBOUNDED**, and **10 rows share DUE 2026-09-13 against a measured
capacity of 1 per cycle**. Ten promises are scheduled to break together in seven
days. `D22`'s default (the status quo, the only legal one) fires 2026-09-08 and
priced this explicitly. I am not re-routing it; I am confirming the price is
being paid.

---

## 11. The honest summary — are we closer to a creature?

**Yes, and by more than the scoreboard shows — which is exactly why the finding
in §1 matters so much.**

Today the project measured that its world's outcome channel could not see the
effects it exists to detect (censored lifespan resolves ~20 of 96 lives; the
graded drive-distance integral resolves 96/96, at zero new constants), and it
repaired that. It measured that its own shallowness nulls were too weak, and
then measured that the weakness was *immaterial* — and published the FAIL rather
than the comfortable half of it. It repaired a memory that confabulated on 100%
of the questions it should refuse, chose the repair by a five-arm probe instead
of an argument, and named the hole the winner opens in the docstring instead of
hiding it. That is the loop working at full strength: research, test, implement,
test, fix.

And that is precisely the reason a third of the ledger being unable to notice
when its subject changes is an integrity risk rather than a chore. **The better
this project gets at rewriting Jack's components, the more certificates go
quietly wrong.** Today the memory scorer changed and seven green ticks — one of
them named in `GOAL.md` — stopped describing the code they certify, with every
instrument reporting clean. Tomorrow it could be `UnifiedBrain.py` and twelve
Tier-1 rows. The ledger is the only scoreboard, and a scoreboard that cannot
tell when the game changed is the "Working" README with better paperwork.

The other honest thing to say: **four of the owner's constitutional commitments
still have zero passing claims, and none of them moved today.** Smell, balance,
shelter, and *too cold kills him* have been claim-dead since 09-03. Ten of
`GENERALITY.md`'s fourteen barriers have no spec at all. We are getting sharper
at measuring the ladder we built and no closer at all to the ladder we did not.

---

## FOR THE BUILDER

**B1 — Make undeclared implementation dependencies impossible to have, not
forbidden to write. This is the audit's highest-priority item.**
Ship a `T0`-class guard (the natural home is beside `T0.17`'s provenance
checks) that, for every registered spec with an implementation file, parses the
test module statically, collects every top-level `import X` / `from X import …`
where `X.py` exists at the repo root, and FAILs when any of them is absent from
that module's `IMPL_DEPS`. Static parsing only — do not import test modules in
bulk; `module_path_for(strict=True)` exists for exactly this reason. Verify it
red-first against `me_9_attributed_recall.py` as it stands today. The current
number is **35 of 105 PASS specs** (54 declare nothing at all); ratchet it
shrink-only from 35 in the same commit. Per the 77th audit's own lesson: a
printed number stays repaired, a rule has a ~12-hour half-life.

**B2 — Re-buy the seven certificates the 12:16 floor edit silently staled, and
discharge the Review's FTB 3 in the same run.**
`ME.3`, `ME.4`, `ME.5`, `ME.9`, `ME.10`, `ME.11.A`, `T2.20`, `XL.00` all import
`EpisodicMemory.py` and none declares it. Add the declaration *and* re-run;
declaring alone will make `run status` shout without answering the question.
The Review's FTB item 3 — the distractor conjunct on `ME.3`/`ME.5`/`ME.9`/
`ME.10` — was written to land *in this same run*, and it has not landed
(`distractor` appears 0 times in `me_3`, `me_9`, `me_10`). `ME.9` is named in
`GOAL.md`; it should not be last. **Commit the rows as the runner writes them.**
If the new coverage floor costs `ME.9` its terse-cue recall or `ME.5` its
at-scale recall, that is a finding about the repair and must be recorded as one,
not re-rolled — the floor probe measured those two scars on *candidate* arms
through `ME.1`'s harness, never on `ME.9`'s and `ME.5`'s own harnesses.

**B3 — Correct the correction.** The uncommitted `REVIEW_QUEUE.md` note on
`me1-similarity-floor-never-abstains` prescribes
`grep -rl IMPL_DEPS experiments/tests/ | xargs grep -l EpisodicMemory.py`.
That pipeline's first stage removes every file that declares no `IMPL_DEPS` —
which is all seven files it missed. Replace it with the reverse question
(*"which tests import it?"* → `grep -rl EpisodicMemory experiments/tests/`, then
subtract the declarers), and say in the row that the original grep was the
blind spot rather than the cure. B1 is what actually retires it.

**B4 — Name the two ratchet moves in a commit, or clear them.** `run ratchets
record` in the commit that justifies them: `champions_trigger_debt` 3 → 5
(Language model, Language acquisition — both off `LG.00`'s VOID) and
`unreachable` 95 → 96, which is **above** its declared floor and which no commit
raised. The `--llm-pass` recompute landed at 12:32 (4227 verdicts on disk), so
re-running `LG.00` is the cheapest path to clearing both. Also retire the 09:07
journal's inference explicitly — it read 95-at-floor and self-resolved; it did
not.

**B5 — Re-buy `T0.11`** (Review FTB 8). Backend failover, last run 2026-08-04
attempt 1, **33 days**, the oldest live certificate in the ledger, asserting
something about a dispatch path rewritten twice since. Not a rewrite — just ask
it again.

**B6 — `experiments/tests/dp_04_slow_path_verbal.py` does not compile** and has
not since 2026-08-30: a bare-string banner sits between the docstring and
`from __future__`, so `compile()` raises `SyntaxError: from __future__ imports
must occur at the beginning of the file` at line 391 while `ast.parse` passes.
Move the banner below the future-import. Then consider whether the guard in B1
should also assert *compilability* of every registered spec's module — that is
one more static check on the same walk, and it is the only reason this went
seven days unseen.

**B7 — Standing, unchanged:** the D1.0 arm phase (pid 1775588, declared, est
6.04 h) owns its own ledger row — do not relaunch it; `HR.1`–`HR.4` stay
D19-held to 09-14; the `cross-organ` fork (c) does not land without its mutation
falsifier; and the ten rows dated 2026-09-13 against a capacity of one need
re-dating *before* they go OVERDUE, not after — the next date carrying no
promise is **2026-09-15**.

---

## FOR THE OWNER

**1. DECISION EVIDENCE, appended to `D18` (decide_by 2026-09-09) — the ~1.5 GB
memory ceiling took its first measured breach under your own armed default, and
it was 58% over.**
At 12:25:52 today, `lg_00_not_a_puppet --llm-pass` peaked at **2424 MB VmHWM**
against `SYSTEM.md`'s 1536 MB ceiling, for ~14 minutes, on the box that also
runs WorldTwin and four paying tenant agents. Separately `T1.01` recorded
1532.2 MB — 4 MB under. Nothing refused either run, which is `D18`'s default
working exactly as you armed it (*measure and report, gate nothing, relax
nothing*). You are seeing it because the instrument you approved has now
produced the number it was built to produce, three days before your deadline.
Full evidence is appended to the `D18` entry in `docs/DECISIONS_NEEDED.md`.
**No action requested from me** beyond the ruling you already hold.

**2. NO-DECISION, reported so it is not inferred from a table: `GOAL.md`'s
anti-costume claim has no passing spec as I write.**
`LG.00` — *"Jack knows what his LLM cannot — he is not a puppet"*, which
`GOAL.md` names by id — went **VOID** at 12:16 when the memory repair changed
the prompts its verdict cache keys on. This is the spec's own documented
behaviour, the recompute finished at 12:32, and the re-run is the next slot's
first job, so I expect it back within the hour. Nothing here is for you to rule
on **unless the re-run does not restore it**.
What I would flag as a design fact rather than an incident: this spec self-VOIDs
on **any** improvement to Jack's memory, and its repair costs a ~14-minute LLM
recompute that breaches the ceiling in item 1. It is also the *only* arena for
two champion seats, which is why both now carry trigger debt. The project's
single proof that Jack is a creature and not a costume is the one test that
breaks every time Jack gets better. That is worth a design decision eventually;
it is not one today.

**3. NO-DECISION, a price arriving on schedule.** `D22`'s default fires
2026-09-08 and priced silence at *"approximately 17 further net queue rows"*.
The queue now holds **40 live rows**, disposes **0.29 per cycle** against
**5.29** arriving, reads `drain: UNBOUNDED`, and has **ten rows promising the
same date (2026-09-13) against a measured capacity of one per cycle**. Nothing
is overdue yet and there are **zero violations**. I am not re-routing this and I
am not asking again — `D22` is already on your desk with the arithmetic. I am
confirming, from outside the desk that wrote it, that the forecast is holding
and that ten dated promises are currently scheduled to break in the same hour.

**4. NO-DECISION, the standing hole, restated because no instrument will ever
raise it on its own.** Four of your constitutional commitments — smell, balance,
shelter-building, and *too cold kills him* — have **zero passing claim specs**
and have had since 2026-09-03. `docs/GENERALITY.md` names fourteen barriers
between Jack and generality: **ten have no spec at all and none has a passing
one.** Every organ in this system will keep reporting green while that stays
true, because each measures the ladder we built. The Review said this yesterday
in almost these words; I am saying it again today because nothing was registered
against it in the twenty-four hours since, and a finding that only ever gets
restated is a finding that is being managed rather than fixed.
