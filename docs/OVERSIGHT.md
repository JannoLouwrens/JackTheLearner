# OVERSIGHT — 23rd audit, 2026-08-20 18:45 UTC

## VERDICT: ON TRACK — the ledger is honest and section 2 is clean for a second week, but the leak gate that UB.10 repaired yesterday reads the same number from a healthy null and from a null that never trained, and the false inference behind it is now sitting in LESSONS.md

Every hard integrity check passes. 83 PASS, every one resolving to a live
commit and an implementation; **zero module-level constants moved in any
existing spec in seven days**, verified by parsing the AST at HEAD against the
tree seven days ago rather than by reading diffs; every W33 GPU hour is
attributable to a named job with a recorded outcome. The builder ran 25
iterations in 24 h, all `rc=0`, and delivered sight's first claim-kind PASS
(T3.01) — the eighth constitutional commitment to gain one.

The finding that outranks the rest is in yesterday's repair, not in what it
replaced:

> **UB.10's new leak detector — "any unimodal variant's slot accuracy off 0.5
> by more than 0.10 → VOID" — is read from arms that carry no learning check
> of any kind.** A unimodal variant that converged and correctly found the XOR
> undecodable reads exactly 0.5000. A unimodal variant that never trained at
> all also reads exactly 0.5000. `_run_seed` computes both distinguishing
> observables (the variant's own marginal accuracy, its first→last loss) and
> **discards them** at `ub_10_fusion_bakeoff.py:561-562`, keeping only
> `acc["slot"]`. All **36** unimodal trainings recorded so far (pilot + both
> probe recipes) read exactly 0.5000 — including A2's and A3's, whose *full*
> arms provably never trained under either recipe.

Nothing is laundered: UB.10 has never recorded a ledger row, and per its own
pre-registered rule it is now barred from dispatching. But the commit message,
the docstring and the new LESSONS.md entry all state that the twelve 0.5000
readings *prove the fixture clean*, and that inference does not hold. The same
shape — a must-fail control observed only through its outcome metric — sits
under T3.01's live PASS (§1.2).

---

## 0. Coverage — is the ladder the RIGHT ladder?

`python -m experiments.coverage` exits **0**. **Zero commitments with no
declared spec**; zero malformed declarations. The tool's own guard holds.

**14 of 23 commitments have specs but nothing passing** — down one from the
22nd audit, because **sight** gained its first claim PASS (T3.01, 15:29 UTC).
The nine now carrying a claim-kind PASS: damage/nociception, taste,
memory-across-lives, generality, language (parent), hearing, curiosity,
one-brain/unison, **sight**.

The harder read is *why* the 14 are stuck. Every one of their claim specs, by
dependency status:

| commitment | claim spec(s) | implemented? | blocked by |
|---|---|---|---|
| touch/contact | UB.5 | **no** | UB.1 ← T2.01=FAIL |
| tool use | CU.6 | **no** | CU.1 ← T2.01=FAIL |
| proprioception | T3.02, UB.16 | **no** | T2.01=FAIL / T2.02=VOID |
| sleep | ME.7, T5.05 | **no** | T5.03 ← T4.04 ← T2.01=FAIL |
| plasticity | T5.03, T5.04 | **no** | T4.04 ← T2.01=FAIL |
| social/other agents | T6.05, VO.02 | **no** | T6.01 ← T2.01=FAIL |
| hunger/thirst | PS.04 | **no** | LC.03=VOID (in flight) |
| fast/slow | DP.01–DP.04 | **no** | LC.04 ← LC.03=VOID |
| shelter, thermal | SH.01 | yes | parked to LC.04 |
| smell | SM.02 | yes | PARKED 08-20 (repair 3 negative) |
| balance | BA.02 | yes | D8, escalated |
| death & retry | XL.01 | yes | power-blocked |
| voice | VO.02 | **no** | needs a second Jack |
| *(sight — now passing)* | T3.01 | yes | — |

**Eleven of the fourteen are downstream of `T2.01 = FAIL` or of `LC.03`.** This
is not neglect and it is not drift: the builder cannot reach them. It is the
single most important structural fact in this audit and it belongs in §8.

---

## 1. Integrity of the ledger

### 1.1 Hard checks — CLEAN

83 PASS. All checks run mechanically against the live ledger:

| check | result |
|---|---|
| PASS rows whose `commit` no longer exists in git | **0 / 83** |
| PASS rows with no implementation in `experiments/tests/` | **0 / 83** |
| PASS rows whose spec declares no `control` | **2 / 83** — `T0.01` (repo imports clean), `T0.10` (Kaggle round-trip). Both are harness specs with no hypothesis to control; correct by design, unchanged from the 20th audit. |
| PASS rows with a declared control and no `control_metrics` recorded | **0 / 81** |
| `audit_supersedes_fail` violations on the live ledger | **0** (checked_pairs 1, unauditable 28) |
| stale claims | 3, all VOID/pre-`impl_sha`: `LC.03` (registered re-run in flight), `BA.02` (D8), `T2.02` (pre-`impl_sha` content drift) |

Note on the supersede guard, for the record: it now covers the VOID lane
(22nd-audit B1/B2, shipped in `5a2e8e1`) and reads 0 violations — but its
*actual* coverage is **1 pair of 29**. The other 28 predate `impl_sha` and are
structurally unauditable. That is a true statement of a limit, not a fault; it
means "0 violations" should never be quoted without "checked_pairs 1".

Minor accuracy note: the 17:07 iteration summary says "**Zero stale claims now
remain in the ledger**." Three stale rows are live right now (above). The
builder meant *zero stale PASS/FAIL rows*, which is true and is the meaningful
statement — but the sentence as written is the kind a later reader will quote.

### 1.2 A control that is only observed through its outcome cannot prove it ran

This is the generalisation of the headline finding, and it has a second live
instance.

**T3.01 (PASS, credited to sight today).** Its declared must-fail control is
shuffled-label training, gated by `shuffled_dev_max > 0.10 → VOID`. The control
arm is observed through exactly one number, `acc_shuffled`
(`t3_01_ablate_vision.py:363-367`), and the gate passes when that number is at
chance. A shuffled arm that trained honestly and could not generalise reads
chance. A shuffled arm that silently never trained also reads chance. Nothing
in the recorded row distinguishes them.

I do **not** think T3.01's control is broken — it shares the training function
and the budget with the full arm, and the full arm demonstrably trained
(0.553–0.643, no class collapse). But that is an argument from code-sharing,
and this project's own standard, restated in LESSONS.md yesterday, is that a
premise you did not measure is a second experiment hiding inside your
diagnostic. One extra recorded number per control arm — its train accuracy, or
first→last loss — converts the argument into evidence.

**Ranked #1** because it is the only finding that touches what a credited PASS
can prove from the record, and because the fix is nearly free in both specs:
the numbers already exist and are being thrown away.

---

## 2. Thresholds and controls over time — NO SILENT LOOSENING, second week running

Method (stronger than reading diffs): parse every file in `experiments/tests/`
at `HEAD` and at `88507ab` (the last commit before 2026-08-13 18:45) with
`ast`, extract every module-level `UPPERCASE` constant, and diff the values.

**Result: 0 constants changed, 0 removed, across every spec that existed seven
days ago.**

Two constants moved inside specs *born* this week; both are documented and
neither is a loosening:

| spec | change | verdict |
|---|---|---|
| `T3.01` | `EPOCHS 25 → 62` (`0b2b41b`) | **Justified by measurement, pre-registered.** The change was dictated by a decision rule written into `t3_01_curves_probe.py` *before* the probe launched: "smallest epoch where all seeds clear the band (31), doubled". Every verdict gate (`MIN_FULL`, `MIN_DROP`, `ABL_CEIL`, `TRAIN_TOL`) is byte-identical; the control trains at the same raised budget; the attempt-2 VOID stays in history; a one-diagnostic cap was pre-stated. This is the textbook version of the manoeuvre. |
| `UB.10` | leak gate `ens_slot > 0.60` → `abs(uni_slot − 0.5) > 0.10` (`dcef2fb`) | **Loud, pre-registered, no seed had run — but see §2.1.** |

Registry diffs over the same window are **additions only**: `T2.05`, `T3.01`,
`T2.04`, `T4.02`, `T3.07` each gained a `control=` declaration where none
existed, several explicitly marked "added BEFORE first run". No `_check` gained
an `or`; no seed count fell; no assertion was deleted.

Section 2 is a genuine null result and the second consecutive week of one.

### 2.1 The UB.10 gate amendment — right to make, over-claimed in the record

The amendment itself is defensible and I am not asking for the old gate back.
The old premise really was false mathematics: `sign(s(v)+t(a))` reaches 0.75 on
a balanced XOR through miscalibration alone, and the pilot measured
`ens_slot = 0.747` on A5. Amending it before any registered seed ran, in a
commit that states the reasoning at length, is law 4 working.

Three things in the record are wrong or unsupported, and they matter because
this project treats gate claims as theorems:

1. **"ALL 12 unimodal accs read exactly 0.5000, proving the fixture clean"**
   (commit `dcef2fb`, repeated in the docstring and in the new LESSONS.md
   entry as "which PROVES the fixture clean"). It does not. Exactly 0.5000 on a
   balanced test set is what a *converged* unimodal model returns on a clean
   XOR **and** what a model that never trained returns, because both emit a
   constant. In that very pilot, A2's and A3's full arms sat at loss 1.60→1.56
   and 1.90→1.82 for all 150 epochs; their unimodal variants share the trunk,
   the optimiser and the recipe, so four of the twelve readings quoted as proof
   are the ones least entitled to be believed. The recipe probe reproduced all
   twelve at 0.5000 under both recipes — 36 identical readings, zero variance,
   from arms that include known non-learners. Zero variance across 36
   independent trainings is itself the tell.

2. **"Strictly stronger against every real leak."** Not established. The two
   detectors see different things. The unimodal-accuracy gate is strictly
   stronger against any leak visible in a unimodal argmax — which is most of
   them, and the amendment is right about the dilution problem. But a leak that
   lives in *calibration* while both argmaxes sit at chance is exactly the
   regime the pilot demonstrated is reachable, and the new gate is blind to it
   where the old one was not. The `winner_beats_own_ensemble` PASS clause
   absorbs part of that exposure and should be named as the thing doing that
   work, rather than the claim of strict dominance.

3. The new gate reads a metric from an arm with **no `learn_ok` and no
   `marginal_ok`**. `_check` computes both (`:787-791`) exclusively over the
   full arms. A leak detector that cannot fire on a dead arm cannot protect the
   arms most likely to be broken.

### 2.2 The recipe probe has landed and it says: do not dispatch

`/data/tmp/ub10_recipe_probe.log`, written 18:31 UTC (uncommitted; charge
0.2288 h already in `gpu_budget.json`):

```
RECIPE warmup:  NOT CLEAN: ['A2:marginal', 'A3:marginal']
RECIPE lolr:    NOT CLEAN: ['A2:marginal', 'A4:marginal']
```

Neither recipe is clean. The pre-registered rule in `remote_recipe_probe`'s
docstring is explicit: *"neither clean → no dispatch, arm redesign routes
through PROGRESS."* Note also that `lolr` **fixed A3** (1.84→1.58, slot 1.0)
while **breaking A4** (slot 0.5531, was 1.0) — a recipe-sensitivity signal, not
a single stuck arm. The one-diagnostic cap is spent. A third recipe is not
available. **I flag this in advance rather than after the fact**: the temptation
next iteration will be to try one more LR, and the rule the builder wrote
yesterday forbids it.

---

## 3. Drift from the goal — none

Everything the builder touched in the last 24 h, and the GOAL.md sentence it
serves:

| work | GOAL.md sentence |
|---|---|
| T2.05 harvest (FAIL, predicted in the docstring before the run) | "Fast and slow, in one brain" — prices the world-model arms for LC.04 |
| T3.01 pilot → VOID → curves probe → v2 → **PASS** | "ablate a sense, something measurable must degrade" (sight) |
| T3.07 + T0.17 stale re-runs | "every capability claimed only by an experiment that could have failed" |
| T4.02 re-run (FAIL, predicted) | "no modality collapse" — indicts the fusion for UB.10 |
| UB.10 implement → pilot → gate repair → recipe probe | "one interconnected brain… every sense load-bearing" |
| supersede-guard widening + T0.27 P11/P12 | "protects the honesty of watching what happens" (GOAL.md:8) |
| LC.03 registered run (in flight, ~8 h 30 m worker CPU) | "learns his world by living in it" |

**No drift.** Every item traces to a sentence, and two of the seven are pure
falsification machinery — the ratio a healthy week should have.

The converse question is §0's table, and its answer is uncomfortable but not a
builder fault: **11 of the 14 zero-pass commitments are unreachable behind
T2.01 or LC.03**, so the frontier the builder is allowed to work is narrow by
construction. Curiosity (12 specs, 1 PASS) and one-brain/unison (21 specs, 1
PASS) remain the two commitments with the widest gap between how much ladder
was built for them and how much of it has been climbed.

---

## 4. Is the builder alive and productive? — YES

- **25 iterations** in the 24 h to 18:19 UTC; **25 ended `rc=0`**; zero aborts,
  zero timeouts, zero repeated-identical failures.
- **PASS: 81 → 83** (net +2; the count dipped to 80 at 20:07 when a staleness
  re-run withdrew a claim, then recovered honestly). Render **83/169**.
- No paused loop, no credit exhaustion. The Fable weekly meter reads **75 %
  against a 90 % hard stop** and the builder has been propagating a "stay lean"
  note through the journal since 15:07 — appropriate, and see §6.
- **LC.03 verified alive independently**, not taken on the builder's word: pid
  92854 plus three spawn workers, `etime` 8 h 19 m, per-worker CPU 30 610–30 692 s
  (~8 h 30 m) and climbing, ~470 MB RSS each. This is the correct liveness
  signal for a run that prints nothing until it ends.

Three iterations in the window produced no ledger delta (11:07, 12:07, 17:07)
and each has an honest reason on record: a mid-flight kernel it deliberately
refused to race, an implementation-only unit, and a pilot that caught two rig
faults. Refusing to dispatch is work.

---

## 5. Compute honesty — CLEAN, and the quota is the story

Every one of the **16 W33 kernels is attributable**. I joined
`gpu_budget.json:charged_jobs` to ledger rows and to
`gpu_submissions.jsonl`; the six jobs with no ledger row all resolve:

| kernel | h | what it bought |
|---|---|---|
| `1787185633` | **1.5583** | SM.02 pilot — largest single W33 charge, and the *only* one that would have been anonymous. It is named by an **attribution receipt** the 20th audit asked for (`phase: "attribution"`, 01:19 UTC). The machinery worked. Outcome: SM.02 PARKED (`d7be64c`, repair 3 measured negative). Honest exploratory spend with a recorded decision. |
| `1787235257` | 0.5679 | T3.01 curves probe — produced the pre-registered EPOCHS repair that turned a VOID into today's PASS |
| `1787231872` | 0.1994 | T3.01 attempt-2 VOID (in `history`, not the current row) |
| `1787231324` | 0.1173 | T3.01 seed-90 pilot |
| `1787246533` | 0.1470 | UB.10 pilot — caught two rig faults before a ~1.25 h registered spend |
| `1787249890` | 0.2288 | UB.10 recipe probe (§2.2) |

- **W33 charged: 6.4879 h** (6.283 successful + 0.2049 dead kernels = **3.2 %
  waste**, all three deaths from the mujoco 3.12.0 sdist-before-wheels trap,
  diagnosed and pinned the same day).
- **Bought:** 5 PASS (T2.06, T2.03 re-cert, T2.04 re-cert, TA.02, T3.01), 2
  FAIL (T2.05, T4.02 — both *predicted in the docstring before the run*), 1
  park (SM.02), 1 no-dispatch decision (UB.10). Nothing was spent to get a
  better number.
- **W33 remaining: ~23.51 h of 30, expiring Sunday 2026-08-23** — roughly 2 days.

**And as of 18:31 UTC there is no queued spender.** UB.10's registered run
(~1.25 h) was the designated one and its own pre-registered rule just barred it.
The only large, ready, high-value job in the repo is **T2.01 at 5.58 h — which
fits four times over in what will expire — and it cannot be dispatched without
one line from the owner (D1).** See §8.

---

## 6. Stuck decisions

**D1 (the 57M trunk / PLASTIC-ONLY) — open since 08-10, and it is now the
reason a third consecutive week's GPU quota will expire unused.** Nothing new
in the evidence; the fork is unchanged. What is new is §0's table: `run blocked`
puts T2.01 at **frees 35 / blocks 36**, and I can now name what those 35
contain — the claim specs for **touch, tool use, proprioception, sleep,
plasticity and social**, six constitutional commitments, plus T4.04 (unison).
This is escalated below and appended to `DECISIONS_NEEDED.md`.

**D7 (MovementMoodCoupling: delete, redesign, or accept as cosmetics) — open
7 days, and the evidence is now complete.** Today's local re-run recorded FAIL
**bit-identical** to the 08-13 row (`acc_per_seed [0.225, 0.275, 0.375]` vs
`MIN_ACC 0.45`), on current code, proving the intervening `IMPL_DEPS` drift
never touched the mood→action path. There is no further measurement that would
inform this decision. It is a three-way call the system may not make for itself
(deleting a shipped component). Ready to decide.

**D8 (BA.02 unmeasurable in the rover body)** — unchanged, correctly parked.

**Could the system have resolved any of these itself with a bakeoff?** No. D1
is constitutional (the 15th audit settled that a bakeoff cannot answer what a
decree admits), D7 is a delete-authority question, D8 is a body-design fork.

**Was any owner decision quietly acted on?** No. I grepped seven days of
commits under `jack/`, `models/` and `src/` for `requires_grad`, `.eval()`,
`freeze` and `detach()` — **zero hits**. The PLASTIC-ONLY decree has not been
narrowed in code while D1 waits.

---

## 7. Bakeoff hygiene — nothing new to audit

`docs/DECISIONS_RESOLVED.md` is unchanged since 2026-08-13 and still holds
exactly three entries: `PS.01/J` (VOID, correctly recorded as *not* a verdict
and re-run), `PS.01/J2` (winner `impact_speed`), and `D2` (VOID blocks its
dependents, resolved by ledger replay with the loser's surviving insight and a
re-open trigger recorded). No new decision was taken by bakeoff in the audit
window, no VOID was treated as a verdict, and no winner was declared inside a
noise margin.

The live bakeoffs are LC.04 (blocked on LC.03, in flight) and UB.10 (blocked on
its own recipe, §2.2). Both are behaving: UB.10 in particular has now twice
refused to dispatch on rig grounds, which is the hygiene this section exists to
check.

---

## 8. The honest summary — are we closer to a curious humanoid that climbs the ladder?

**Yes, genuinely, and for the first time in a while it is not only bookkeeping.**

Today sight became load-bearing. `T3.01` is not a fixture and not a sensor
certificate: it trains the vision encoder end-to-end for the first time in its
existence, then removes vision and watches the brain fall to exactly chance
(0.25) on all three seeds while the intact brain reads 0.553–0.643. That is
GOAL.md's own standard — *"ablate a sense, something measurable must degrade"* —
met for a sense, on the record, with both controls at chance. It got there
through a VOID, a pre-registered diagnostic, and a repair the diagnostic
dictated in advance. The whole machine worked exactly as designed.

Two FAILs landed this week that were **written down before the runs that
produced them**. T2.05's docstring predicted its own FAIL three commits early;
T4.02's did the same. A system that can pre-commit to its own bad news is not
fooling itself, and it is the strongest single piece of evidence in this audit.

**And yet.** He still cannot be shown to walk. `T2.01` — *Locomotion beats a
random policy* — has read FAIL since 2026-08-12T12:59, **nine days**, and it
sits under 35 other specs including the claim specs for touch, tool use,
proprioception, sleep, plasticity and social company. The ladder-and-apple
standard in GOAL.md is *"climbing the ladder on attempt 40 after falling on
attempts 1–39."* Climbing requires moving. We are certifying senses one by one
onto a creature whose locomotion spec is red, and 11 of the 14 uncredited
commitments are stacked behind that one row.

So the honest scoring: the green ticks earned this week are real, not
decorative — one of them is a sense proven load-bearing and two of them are
predicted failures. But the *shape* of what remains has not changed in nine
days, and it is now a single sentence from the owner wide. This week that
sentence also costs ~23.5 GPU hours that expire on Sunday with nothing queued
to spend them on.

---

## FOR THE BUILDER

**B1 (RANK 1). Give every null/control arm a liveness observable, and gate on
it.** Two live instances, one fix.

*UB.10* — in `_run_seed` (`ub_10_fusion_bakeoff.py:553-562`) you already train
the unimodal variants and compute their full accuracy dict and their
`loss_first`/`loss_last`, then keep only `acc["slot"]`. Keep three more numbers
per variant: `uni_v["acc"]["vslot"]`, `uni_a["acc"]["afell"]` (each is decodable
from precisely the sense that variant retains, so each is a *must-succeed*
reading), and each variant's `loss_first`/`loss_last`. Then extend the VOID
checklist in `_check` and in `_seed_row_clean`: a unimodal variant that misses
`MARGINAL_FLOOR` on its own sense's marginal task, or whose loss did not
decrease, makes the run VOID. Without this the leak gate at `:869` cannot fire
on a dead arm, and A2/A3 are dead arms right now.

*T3.01* — `_control` (`t3_01_ablate_vision.py:363`) returns only
`acc_shuffled`. Record the shuffled arm's **train** accuracy (or first→last
loss) in the same dict, and add a rig gate: a shuffled arm that did not fit its
own shuffled training set did not run the control. I do not think your control
is broken; I am saying the row cannot prove it, and the number is free. A
re-run is **not** owed — record it the next time T3.01 runs for any reason.

**B2 (RANK 2). Correct the record on "proving the fixture clean" — three
places.** The commit message `dcef2fb`, the amended ensemble section of the
UB.10 docstring, and the new LESSONS.md entry "A null baseline's value under H0
is a theorem to prove" all assert that twelve unimodal readings of exactly
0.5000 *prove* the fixture clean. They do not: a constant predictor returns
exactly 0.5000 on a balanced test set whether the fixture is clean or the model
never trained, and four of those twelve came from A2/A3, whose full arms
provably never trained. LESSONS.md is the file you read before designing every
experiment; a false inference parked there will be cited. Amend the LESSONS
entry (the *rule* it states is right and should stand — it is the supporting
sentence that is wrong) and state the corrected reading in the docstring's
PILOT RECORD. The honest version: *the fixture's cleanliness is not yet
established; it becomes established once the unimodal variants demonstrate they
can learn their own marginal task and still read 0.5 on slot* — which is
exactly what B1 buys you.

While you are there, soften "strictly stronger against every real leak" to what
is true: strictly stronger against any leak visible in a unimodal argmax, blind
to a calibration-only leak that the old ensemble gate could see, with
`winner_beats_own_ensemble` carrying that residual exposure.

**B3. Honour your own no-dispatch rule.** The recipe probe came back NOT CLEAN
on both arms (§2.2). Your pre-registered rule says: no dispatch, arm redesign
routes through PROGRESS. The one-diagnostic cap is spent. Do not run a third
recipe. Note in the harvest that `lolr` **fixed A3 and broke A4** — that is a
recipe-sensitivity finding about the six-arm design, not a stuck arm, and it is
worth more to the redesign than another LR would be.

**B4. Commit the inherited receipts first.** `experiments/gpu_budget.json` and
`experiments/gpu_submissions.jsonl` are modified and uncommitted right now —
the recipe probe's 0.2288 h charge and its result line, written by the watcher
after the 18:19 iteration ended. Commit before anything else, with the probe's
verdict in the message.

**B5 (standing, carried from the 22nd audit).** When LC.03 lands (~01:20 UTC,
verified alive at 8 h 30 m worker CPU), remember its control side (e) is a rig
tripwire, not a must-fail control. If it reads PASS, the dwell/chaos gates carry
the curiosity burden — say so explicitly in the harvest commit.

**B6 (credit).** The attribution-receipt machinery from the 20th audit did real
work this week: the largest single W33 charge (1.5583 h, the SM.02 pilot) was
dispatched outside `run_spec` and would have been an anonymous line in
`gpu_budget.json`. It is named. Keep routing every out-of-band dispatch through
it.

---

## FOR THE OWNER

**1. D1 is now nine days old and it is the reason six of your named senses have
no runnable claim spec.** I am not taking a side and the fork is unchanged: (i)
strike option A (freeze the trunk), since the PLASTIC-ONLY decree of 2026-08-09
stands as written; or (ii) keep option A available and narrow the decree's
scope, saying where. Please do not answer *"do what the measurements say"* —
the 15th audit established this is a constitutional question about what your
decree admits, not one a bakeoff can settle.

What is new is the shape of the cost, which I can now name precisely rather
than count. `T2.01 = FAIL` blocks 36 specs, and those 36 contain the **only**
claim specs for **touch, tool use, proprioception, sleep, plasticity and
social/other agents** — six of the commitments you wrote into GOAL.md yourself.
Eleven of the fourteen commitments with nothing passing are behind it. The
ladder is not slowing down; it is running out of reachable rungs.

And the arithmetic, this week: **~23.5 of 30 Kaggle GPU hours expire Sunday
2026-08-23.** T2.01 costs 5.58 h and fits four times over. As of tonight the
loop's designated spender (UB.10) has barred its own dispatch on rig grounds —
correctly — so there is currently nothing queued for those hours. This is the
third consecutive week D1's openness converts into expired quota, and this time
it is not a scheduling matter the builder could fix.

**2. D7 is ready to decide and has been for a week.** *MovementMoodCoupling
failed its ablation.* Today's re-run on current code reproduced the 08-13
result **bit-identically** (`acc_per_seed [0.225, 0.275, 0.375]` against
`MIN_ACC 0.45`), which proves the intervening code drift never touched the
mood→action path. Mood does not reach behaviour. No further measurement will
change this. The call is yours because it is delete-authority: **delete the
component, redesign it, or accept it on the record as cosmetics.** Any of the
three is fine; leaving it undecided means a component that earns no parameters
stays in the model, which GOAL.md's "earn your parameters or be deleted" rule
does not permit indefinitely.

**3. A note, not a request: the Fable weekly credit meter is at 75 % against a
90 % hard stop**, four days into the week. The builder has been running lean
deliberately since 15:07. If the meter trips, the loop pauses and the Sunday
quota expiry becomes certain rather than likely. No action needed unless you
want to raise the ceiling.
