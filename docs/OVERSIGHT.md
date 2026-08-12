# OVERSIGHT — 11th audit, 2026-08-12 18:50 UTC

## VERDICT: ON TRACK

Since the 10th audit (12:37 UTC) the loop produced **11 commits and +1 PASS
(72 → 73)**, closed **three of that audit's four findings**, and put the first
`(claim)` PASS on `one brain / unison` — the project's namesake commitment,
which had been 0-of-21 for its entire existence. Section 2 is **clean**: I diffed
every threshold and control touched in seven days and found **no loosening**,
silent or otherwise. Section 1 is **clean**: 73 PASS, every commit resolves, zero
`+dirty`, zero controls faked. Section 5 **reconciles to the cent**.

`ON TRACK` is not "nothing is wrong." Three findings follow. The first is the
serious one and it is a new class for this file: **a pre-registered threshold
that was never touched, on a gate that no longer measures what it gates.** Law 4
says never weaken a threshold. It does not yet say anything about changing what
the number means, and that is the gap BA.01 v2 fell through.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

**Commitments with specs but nothing passing: 10** (9 at the 10th audit → 11
after the `COVERS:` kinds fix correctly de-credited PG.4 and LC.01 → 10 after
UB.9). Still at zero:

    touch/contact · balance · damage/nociception · shelter/building
    tool use · proprioception · sleep · social/other agents · plasticity
    CURIOSITY (12 specs, 0 pass, 0 ever run)

Headline: **165 specs · 73 PASS · 2 FAIL · 1 VOID · 89 not implemented.
62 of 165 unreachable.**

The coverage organ is now doing exactly what it was built for. The last two
work units (UB.9, BA.01) were both chosen by the *standing zero-pass rule*, not
by `run blocked` fan-out. That is the right selection pressure and it is
visibly operating.

---

## RANK 1 — BA.01 v2's rig-degeneracy guard is inert: the threshold never moved, but the statistic it gates is now dominated by the rig's own RNG

**Status: no PASS is corrupted — BA.01 has never passed. This is a guard that
will be silently dead on the next run, and the fix is one line.**

### What the spec says the gate is for

`ba_01_feels_the_fall.py`, docstring failure mode **#2** — unedited since v1:

> **The rig makes every episode identical** — if every spawn topples on the same
> schedule […] time-to-topple degenerates into the clock and the control could
> not have failed however honest the probe. […] **TF_SPREAD_MIN gates that the
> spread actually happened.**

`TF_SPREAD_MIN = 2.5` is the detector for degenerate *fall dynamics*. In v2 it
became load-bearing for a **verdict**, not just a warning — `_check` returns
`Status.VOID` when `seed_rig_ok < 1.0`, and `seed_rig_ok` is
`toppled_frac >= 0.60 and tf_spread >= TF_SPREAD_MIN`.

### What v2 changed underneath it

v2 redefined the statistic (`_evaluate`, the commit's own comment):

```python
# tf_spread is the spread of ABSOLUTE topple times (hold + fall)
t_fs = [T_SETTLE + ep["t_r"] + ep["t_f"] for ep in eps if ep["t_f"] is not None]
```

`t_r ~ U{0..HOLD_MAX}` with `HOLD_MAX = 40`. Measured:

```
std of t_r alone                  = 11.849 decisions
TF_SPREAD_MIN                     =  2.5
margin from the injected hold RNG =  4.74x
```

The builder's own v2 pilot confirms it: **`tf_spread 12.79`** (v1 pilot: 5.69).
That number is the hold, not the world. **A world in which every episode fell in
exactly the same number of decisions — zero fall variance, failure mode #2 in
its purest form — would still read `tf_spread ≈ 11.8` and clear the gate by
4.7×.** The guard cannot fail. By this project's own mirror rule, quoted three
times inside this very file, *a control that cannot fail is not a control.*

### Why this is worse than one dead gate

v2 disabled **both** independent defences against failure mode #2 in a single
change, and the second one on purpose:

1. **The elapsed-time null** — defeated *by construction*. The uniform box makes
   `P(y|t)` flat regardless of the world. (I checked the box arithmetic: for
   `t_abs ∈ [11, 40]` the within-window position `t = t_abs - t_r` ranges over
   the full `{0..11}` at every `t_abs`, so the design is sound and the claim is
   true.) That is legitimate confound removal — but a randomisation-defeated
   null becomes a *manipulation check*, not a detector.
2. **`tf_spread`** — the remaining detector, now dominated by that same
   randomisation.

So in a degenerate world the vestibular features at post-release position `t`
become a near-deterministic function of `t`, the headline AUC inflates, the
clock null reads chance (the box guarantees it), the blind control still fails
(arm slides are noise), and `tf_spread` reads 11.8. **Every gate would be green
and every one of them would be measuring the hold RNG.** `MIN_CLASS_ROWS = 25`
is only a partial backstop: identical fast falls still populate both classes
inside a 12-decision window.

### The file now holds two incompatible readings of one constant

- Docstring #2 (unedited): `TF_SPREAD_MIN` gates that the **log-uniform tilt
  draw spread the fall times**.
- v2 code comment: `tf_spread` is the spread **that must be wide for the clock
  null to be able to fail**.

The `VOID` branch and the 18:36 journal pre-registration both invoke the *first*
reading. The code implements the second. Under the second reading the gate is
coherent and trivially satisfied; under the first it is the degeneracy detector
and it is blind.

### The fix is already written

v2 computes the correct statistic and then does not gate it:

```python
# tf_fall_spread reports the fall-dynamics spread alone, for the reader.
out["tf_fall_spread"] = float(np.std(falls)) if falls else 0.0
```

**`tf_fall_spread` is what `TF_SPREAD_MIN` was pre-registered to guard.** Gate
it. See FOR THE BUILDER.

### What I am *not* claiming

I am not claiming v2 is a rescue of a FAIL. I checked this specifically and the
evidence is against it: the v2 design direction — *"random hold-then-release, or
mid-episode kicks for a memoryless hazard"* — was written into the journal at
17:55 **in the FAIL commit itself**, before v2 existed. Every gate is
byte-identical to v1. Three pilot iterations are documented with the measurement
that forced each. The shuffle-null repair (one fixed permutation → mean of 8)
is a genuine strengthening, correctly diagnosed. The FAIL→VOID reclassification
for rig degeneracy matches docstring #3 and the T2.02 lesson, both of which
predate v1, and as a *PASS bar* v2 is equivalent-to-stricter (v1's aggregate
means were implied by its per-seed conjunction). **This is careful work with one
hole in it**, and the hole is of a kind the project's rules do not currently
name.

---

## RANK 2 — curiosity: 12 specs, 0 passing, 0 ever run — the north star is the single least-measured thing in the project

Not a defect. A structural fact that has now survived eleven audits, and it
belongs at RANK 2 because it is the largest gap between what GOAL.md says this
project is and what the ledger can show.

GOAL.md's opening concrete commitment:

> **He explores because he wants to.** […] If there is a ladder with an apple on
> top, he must try to climb the ladder, fall, and learn from falling, purely out
> of curiosity.

`coverage.py`: **curiosity — 12 specs, 0 pass.** The one passing spec ever
credited to it (PG.4) was correctly re-declared `(fixture)` at `60686ac` — it
proved the noisy-TV *trap* works, not that curiosity does.

**Why it is stuck**, from `run blocked`:

```
T2.01 = FAIL  frees 26  (blocks 36)  — Locomotion beats a random policy
    frees: CU.1..CU.7, ME.7, T2.16..T2.18, T3.02, T3.04, T3.05,
           T4.04, T4.05, T5.01..T5.05, T5.07, T6.01, T6.02, T6.04, T6.05
```

Every curiosity spec, every Tier-5 claim, and every Tier-6 living-Jack spec sits
behind one FAIL. T2.01 is at **2.67 σ against a pre-registered 5 σ** (v4: 1.19 σ;
the critic fix more than doubled it — real progress, correctly recorded, gate
untouched). It has consumed **11.16 of this week's 18.20 Kaggle-hours** across
two attempts.

The arithmetic the next iteration is walking into: **11.80 h remain, the bucket
closes Sunday 2026-08-16, and one more T2.01 attempt costs ~6.5 h.** One attempt
fits. Two do not. At 2.67 σ needing 5 σ, a third identical attempt is not
obviously the best use of the last GPU week — and T2.03 (`gpu<20min`, frees 2
directly and 11 in total including UB.1–UB.8) is queued and cheap. The builder
has already ordered them that way in the journal. I am recording the tradeoff so
that it is a decision rather than a default. See FOR THE OWNER.

---

## RANK 3 — `is_code_dirt` matches its exclusion list by suffix, not by path (CARRIED, 2nd audit, severity still zero)

`protocol.py:113` — unchanged since the 10th audit:

```python
return not any(path.endswith(o) for o in NOT_CODE)
```

`NOT_CODE` mixes bare basenames (`ledger.json`, `gpu_budget.json`) with
repo-relative paths (`docs/LOOP_JOURNAL.md`). Any future file whose path merely
*ends with* one of those strings — `archive/ledger.json` — is silently exempted
from the `+dirty` stamp and from the GPU push guard, which share the predicate.

**No such file exists; I checked.** Reported only because the loop wrote a lesson
about this exact list on 2026-08-12 (`0555761`, *"a named set is still
hand-maintained, and it was incomplete twice"*) and the matching rule is the
remaining soft edge in it. Anchor the comparison; do not grow the list.

---

## Section-by-section

### 1. Integrity of the ledger — **CLEAN**

76 entries · **73 PASS · 2 FAIL (T2.01, BA.01) · 1 VOID (T2.02)**.

| check | result |
|---|---|
| PASS whose implementation is missing | **0** |
| PASS whose `commit` no longer resolves in git | **0** |
| PASS carrying a `+dirty` stamp | **0** (was 4 at the 9th audit) |
| PASS with empty `control_metrics` | **2 — both legitimate** |
| PASS where `control_metrics == metrics` (a faked control) | **0** |

The two empty-control entries are **T0.01** (`null_baseline="n/a — structural
precondition"`, no `control=` declared) and **T0.10** (`null_baseline="n/a"`, no
`control=` declared). Neither spec declares a control, so neither ledger row is
missing one. The declare-a-control property is itself gated — T0.19
(`control_blind_specs`) PASS — so this is machine-checked, not my spot-check.

One honest stale row, correctly self-reported by `run status`:

```
BA.01  recorded FAIL; ran on 8c88fe49a96d625f, now 33c7cf6907e730c0
```

That is the staleness detector working as designed on v2's edit, not a defect.
**34 pre-`impl_sha` entries still cannot be checked for staleness** — unchanged,
each fixed by a re-run.

### 2. Thresholds and controls over 7 days — **CLEAN. No loosening found.**

I diffed `registry.py`, `registry_expansion.py` and every file under
`experiments/tests/` since 2026-08-05, filtering for numeric constants,
`seeds=`, `_MIN`/`_MAX`, and `_check` structure. Every directional change moves
the strict way:

| change | commit | direction |
|---|---|---|
| LC.03 gates `\|z\| <= 3` → two-sided permutation `p >= 0.01` | `1480126` | **stricter** — `\|z\|<=3` admits to p≈0.003; both tails now gated |
| `N_PERM` 2 000 → 100 000 | `1480126` | **stricter** — p-floor 2e-5 under a 1e-3 gate |
| `N_CALIB` 60 → 400 (VO.01) | `9357573` | **stricter** — sized from its own tolerance |
| `N_OCC` 160 → `2*N_TRAIN` (600) | `f239118` | **stricter** |
| T0.20/T0.21/T0.22 `N_PROPERTIES` 7→8, 12→13→14→15, 9→12 | various | **stricter** — properties added, none removed |
| `seeds` 1 → 3 (several) | `9ed2ded` | **stricter** |
| BA.01 v2: every gate | `0fce271` | **byte-identical to v1** — verified |

Two changes that *look* like loosening and are not:

- **BA.01 v2 removed `toppled_frac`/`tf_spread` from the FAIL conjunction.**
  They moved to a per-seed `VOID` branch. As a PASS bar this is equivalent-to-
  stricter: the aggregate means were already implied by the per-seed
  conjunction. Correctly argued in the commit. *(The separate problem with what
  `tf_spread` now measures is RANK 1 — that is not a threshold move.)*
- **Budget reclassifications** (`CPU_FAST`→`CPU`, `CPU`→`GPU`) are scheduling,
  not gates.

No `_check` gained an `or`. No control was deleted or weakened. No assertion
was removed. No seed count was reduced. **Law 4 is being honoured.**

### 3. Drift from the goal — **NONE. Every unit traces.**

| last-day work | GOAL.md sentence it serves |
|---|---|
| UB.9 PASS — "heard, not seen" | *"All of it processed together in ONE model […] a genuinely unified brain where every sense is load-bearing"* |
| BA.01 v1 + v2 — vestibular probe | *"proprioception & balance"* in the sensory inventory |
| `COVERS:` kinds (`60686ac`) | *"protects the honesty of watching what happens"* |
| T0.12 P9/P10, `Budget.charge` locking | same — the meter that prices everything else |

Nothing served none. **The converse is RANK 2**, and it is the finding that
matters: `curiosity` (0/12), `plasticity` (0/2), `sleep` (0/2), `social` (0/2),
`shelter` (0/1), `tool use` (0/1), `damage` (0/1) — the parts of GOAL.md about
*living*, as opposed to *sensing*, are where the zeroes have collected. The
project is building a body it can measure faster than it is building a creature
that uses one.

### 4. Is the builder alive and productive? — **YES, most productive day on record**

24 h to 2026-08-12 18:07 UTC: **14 iterations started, 13 ended, 12 at `rc=0`.**
One `rc=1` (`Reached max turns`) at 2026-08-11 20:07 still delivered +1 PASS.
**PASS delta +7 (66 → 73).** No repeated identical failures, no unresumed pause,
no credit exhaustion, no load aborts (load 0.00–0.32 throughout, 13 GB free).

**One hole: a 10-hour gap, 2026-08-11 20:38 → 2026-08-12 06:47** — ten hourly
slots lost overnight to the 90 % usage gate. The loop resumed by itself when
weekly usage fell. This is the live cost of open decision **D5** and the number
is better than the 9th audit's (≈4 h 40 m permitted, now 14 h of 24).

### 5. Compute honesty — **RECONCILES EXACTLY**

`gpu_budget.json`, week 2026-W32:

```
weeks.2026-W32   kaggle 18.1994   colab 0.5513   colab_failed 0.9914
charged_jobs     kaggle 11.8145   colab 0.5498            0.9914
                 unreceipted delta: 6.3849 kaggle-hours
```

**The delta is fully explained and is not a leak.** I traced the file's git
history: at `92931a6` (2026-08-09) W32 kaggle already read 6.3849 with
`charged_jobs` empty — the receipt mechanism did not exist yet. Every charge
since then reconciles to the hour:

```
92931a6  6.3849  ->  0a7540e 11.9635   delta 5.5786 = receipt 5.5786  OK
0a7540e 11.9635  ->  ebd7366 12.6196   delta 0.6561 = receipt 0.6561  OK
ebd7366 12.6196  ->  dd7186b 18.1994   delta 5.5798 = receipt 5.5798  OK
```

`MAX_TRACKED_JOBS = 500` with 5 entries, so no trimming has occurred. **Every
GPU-hour spent since the meter was built has a receipt naming the job that
bought it.** The 10th audit's RANK 1 (the 15× reattach overbill) is **closed** —
the meter now bills the kernel's own console window, gated by T0.12 P9.

**What the hours bought:** 11.16 h → T2.01 v4+v5, both FAIL, but a real
measurement (1.19 σ → 2.67 σ, traced to the decorative-critic fix). 0.9914 h →
the thrown-away colab run, **recovered at zero re-spend** as T1.02 PASS. 0.5498 h
→ T1.08 re-run PASS. **No hours are unaccounted for and none were wasted
without a recorded cause.**

**Remaining: 11.80 of 30 Kaggle-hours. Resets Sunday 2026-08-16 — 4 days.**

### 6. Stuck decisions — 5 open, 1 newly decidable

`docs/DECISIONS_NEEDED.md`: **D1** (57M trunk in the control path — evidence
complete, option set flagged stale against PLASTIC-ONLY), **D2** (does a VOID
dependency BLOCK dependents), **D3** (may the loop `git push`), **D4** (LC
bakeoff cost), **D5** (usage-resume policy — lost its deadline, not its
question).

**Nothing has been quietly acted on without being recorded.** I checked D3
specifically: the loop is pushing (`origin/main` current), and D3's own thread
records that permission — no silent resolution.

**D2 is now resolvable by the system itself and should not be waiting on the
owner.** It asks whether a VOID dependency blocks its dependents. The evidence
arrived today from two directions: T2.02 is VOID and blocks 4; BA.01 v2 has just
made VOID a *routine* verdict (per-seed rig degeneracy) rather than a rare one.
The answer changes how the whole ladder schedules, and it is a property question
with a testable answer, not a values question. See FOR THE BUILDER.

### 7. Bakeoff hygiene — **CLEAN**

`docs/DECISIONS_RESOLVED.md` holds exactly two entries, both PS.01:

- **PS.01/J — VOID.** Three arms below the 3.0 σ learning gate → no verdict
  recorded. **Correct.** A VOID was not treated as a result.
- **PS.01/J2 — WINNER `impact_speed`.** 0.973 at **10.32 σ over null**, beating
  runner-up `peak_dvel` (0.827, 5.99 σ) by **2.66 σ** — outside the noise
  margin, stated explicitly. Both controls fail the gate (`noise` 0.570/1.47 σ,
  `constant` 0.500/0.28 σ). Eleven eliminated arms recorded by name. `screen`
  gate mode carries a written rationale for why these arms are observables
  rather than learners.

No decision made without a learning gate. No winner chosen inside the noise
margin. No VOID promoted to a verdict.

*Standing observation, not a finding:* SYSTEM.md's third law has been exercised
on exactly **one** real question in the project's life. LC.04 (learning core)
and the curiosity/needs bakeoffs GOAL.md names as the arbiters of its central
open questions are all still unrun — LC.04 behind LC.03, which is the largest
non-GPU unblock available (frees 7, CPU).

### 8. The honest summary — are we closer to a curious humanoid?

**Closer to the humanoid. Not measurably closer to the curious one.**

The case for real progress is strong and I want to state it plainly, because
this file has been sceptical for ten audits and the last two days earned
something. `one brain / unison` — the phrase GOAL.md leads with — went from
**0 of 21 to 1 of 21** on a test that could genuinely have failed: UB.9's fused
arm scored 0.993 where audio-only, vision-only and their late ensemble all sat
at chance, and the builder caught a leak in its own swap control *before* the
recorded run. That is not a green tick. That is a fusion claim with the
synergy-incapable null underneath it. Voice, smell, and thermal all got
certified sensors this week by the same standard. The instruments improved too:
the GPU meter stopped lying by 15×, `COVERS:` kinds stopped crediting fixtures
as claims, and every dirty stamp is gone.

But the ladder's own selection pressure is pointing somewhere the ledger cannot
follow. Nine of ten zero-pass commitments are about *living* — shelter, sleep,
damage, tool use, company, plasticity — and **curiosity, the north star, has 12
specs, has never been run once, and is blocked behind a single GPU spec that has
now failed five times.** What this project can currently demonstrate is that
Jack has *senses*. What it cannot yet demonstrate, at all, is that he *wants
anything*. The apple on the ladder is still a sentence in GOAL.md.

And RANK 1 is the shape of the risk that comes with a fast, honest builder:
nobody moved a threshold — the builder is scrupulous about that, and section 2
proves it eleven audits running — but a rig was rebuilt underneath a constant
until the constant stopped meaning what it was registered to mean. **Law 4
protects the number. Nothing yet protects the measurement.** That gap is worth
closing before it lands under a spec that passes.

Still ON TRACK. The machine is better than I found it. But the next iteration
that spends its window on another sensor instead of on curiosity is choosing the
easier green tick, and it should choose it knowingly.

---

## FOR THE BUILDER

**B1 — RANK 1, do this before `run BA.01` (one line, plus a docstring).**
`tf_spread` can no longer detect the condition docstring #2 assigns it. Gate the
statistic you already compute:

```python
TF_FALL_SPREAD_MIN = 2.5     # fall-dynamics spread — the failure-mode-#2 guard
...
out["seed_rig_ok"] = 1.0 if (toppled_frac >= TOPPLED_FRAC_MIN
                             and tf_spread >= TF_SPREAD_MIN
                             and out["tf_fall_spread"] >= TF_FALL_SPREAD_MIN) else 0.0
```

This is a **new gate on a previously ungated statistic**, so it is a
strengthening and law 4 permits it — but it must be **pre-registered in the
journal with its pilot number before the registered run**, exactly as v2's other
gates were. v1's pilot measured fall-time spread at 5.69 and v1's registered run
at 3.68 ± 2.32, so **2.5 is the value already in the file's history**; do not
pick a new one after seeing v2's number. Then reconcile the two readings of
`TF_SPREAD_MIN` in the file: docstring #2 must say which constant now guards
fall-dynamics degeneracy, and `tf_spread`'s role as "the clock null must be able
to fail" should be renamed (`tf_abs_spread`) so one name does not carry two
jobs.

**B2 — generalise it. A guard against RANK 1's whole class.**
This is the second time a gate kept its number and lost its meaning (the first
was `60686ac`: a count that credited fixtures as claims). Spec it: **when a
spec's implementation changes, any pre-registered constant whose gated statistic
changed definition must be re-declared.** A cheap version that would have caught
BA.01 today: for every gate constant, assert in-code that the gated statistic's
value under a *degenerate rig fixture* falls **below** the gate. A gate that a
deliberately-broken world still clears is inert, and that is machine-checkable —
the same executable-control discipline T0.12 already uses. I have appended the
lesson to `docs/LESSONS.md`; the spec is yours to write.

**B3 — D2 is yours, not the owner's; resolve it with a bakeoff.**
"Does a VOID dependency BLOCK its dependents?" has been open on the owner's desk
while the system accumulated the evidence to answer it: T2.02 is VOID and blocks
4, and BA.01 v2 has just made VOID a routine per-seed verdict. This is a
property question with a testable answer, not a values question. Run it, record
it in `DECISIONS_RESOLVED.md`, and take it off the owner's list.

**B4 — carried from the 10th audit (2nd time), severity zero.**
`protocol.py:113` — anchor `is_code_dirt`'s exclusion match to the full
repo-relative path instead of `str.endswith`. Do not grow `NOT_CODE`; fix the
comparison.

**B5 — the standing zero-pass rule now points at `curiosity`, not at another
sensor.** All 7 CU specs are behind T2.01, so the rule cannot pick one. Before
defaulting to the next cheap sensor, spend ten minutes asking whether **any**
curiosity claim can be made independent of T2.01's locomotion gate — a coverage
or learning-progress measurement on a rig that does not need a trained walker.
If none can, say so in the journal explicitly, so that "curiosity is unreachable"
becomes a recorded fact with a reason rather than eleven audits of silence.

---

## FOR THE OWNER

**Nothing is broken and nothing needs you urgently.** The ledger is honest, the
compute meter reconciles exactly, and the loop had its most productive day yet
(+7 PASS in 24 h). Two things are worth your attention.

**1. The GPU week closes Sunday 2026-08-16 with 11.80 of 30 hours left, and the
allocation is a real fork.**

- **T2.01** (locomotion) is the project's #1 blocker: it alone blocks all 7
  curiosity specs, every Tier-5 claim and every Tier-6 living-Jack spec. It is
  at **2.67 σ against a pre-registered 5 σ** — genuine improvement (was 1.19 σ),
  still a long way short. One more attempt costs **~6.5 h**. One fits in the
  remaining budget; two do not.
- **T2.03** (pretrained vision features) costs **~20 minutes** and frees 2 specs
  directly, 11 counting co-requisites (UB.1–UB.8).

The builder's queue already puts T2.03 first, which I think is right. Flagging
it because a third 6.5-hour T2.01 attempt at the same architecture is the
plausible default, and at 2.67 σ it is closer to a coin flip than to a plan.
**If you have a view on whether T2.01 should get the week's last long slot or
whether its 5 σ bar deserves re-examination on the evidence, that view is worth
more than another attempt.** (Its threshold has not been touched and should not
be touched by the loop — that is exactly the call that is yours, not its.)

**2. D5 (the 90 % usage-resume policy) is still open and cost 10 hours
overnight.** The loop lost ten consecutive hourly slots (2026-08-11 20:38 →
2026-08-12 06:47) and resumed by itself when weekly usage fell. The expired
`.usage-resumed` file is still on disk and **fails closed**: the next time weekly
usage crosses 90 %, every agent stops until you resume them. The three options
are unchanged (renew daily / grant through each weekly reset / accept the stop at
90 %), nobody is proposing to weaken the 90 % rule, and there is no longer a
deadline on it. Reported with today's number so the choice is made on current
evidence: **14 of 24 hours permitted, up from ≈4 h 40 m at the 9th audit.**

**3. A note you did not ask for, on where the project actually is.** Jack can
now be shown to *sense* — sight, hearing, smell, voice, thermal, and a unified
brain that fuses two of them better than either alone. He cannot yet be shown to
*want* anything: curiosity has 12 pre-registered specs and has never been run,
because it sits behind the locomotion gate above. That is not a failure of
honesty — the ledger says so plainly, which is the system working. It is a
statement about what the next GPU week buys. The ladder-and-apple standard in
GOAL.md is a claim about wanting, not about sensing, and it remains untested.
