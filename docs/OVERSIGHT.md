# OVERSIGHT — 14th audit, 2026-08-13 18:40 UTC

## VERDICT: ON TRACK

No false claim on the ledger, no silent loosening, no drift. All **79 PASS**
records name a commit that exists and is reachable from HEAD, none is stamped
`+dirty`, and every PASS whose spec declares a control recorded control
metrics — zero exceptions. The builder ran **24 iterations in 24 h, 21 rc=0**,
and took the ladder **74 → 79** (and 165 → 169 specs) while carrying two
detached multi-hour runs. `coverage.py` exits 0.

The finding that matters is an **auditability hole in a load-bearing place**:
four certificates declare `IMPL_DEPS = ["playground.py"]` so that a change to
the world would make them go stale loudly — and all four predate `impl_sha`,
so the alarm fitted to them cannot fire. `playground.py` has since moved
**+430 / −14 lines across five commits**. `run stale` reads clean for them and
always will. One of the four is PG.4, whose rig carries curiosity's only
claim-kind PASS and whose dwell gate is being applied, right now, to the
learning-core screening run. Fix cost: re-run four CPU specs.

I checked the two amendments made in this window (LC.03's controls (a) and
(e), BA.02 v1 → v2) line by line. BA.02's "zero constants moved" is **true —
I diffed the constant block and `_check` and both are byte-identical**.
LC.03's amendment is procedurally impeccable and substantively costs the spec
one of its five controls; that is RANK 2 and it is not the same thing as
loosening.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

**169 specs · 79 PASS · 1 FAIL · 2 VOID · 0 stale-blocking.**

| tier | | |
|---|---|---|
| T0 harness | 28/28 | complete |
| T1 primitives | 13/13 | complete |
| T2 vs null | 36/59 | 1 FAIL (T2.01), 1 VOID (T2.02), 21 not run |
| **T3 earn your parameters** | **0/14** | nothing implemented |
| **T4 unison** | **1/23** | |
| **T5 THE CLAIMS** | **0/27** | 1 VOID (BA.02), 26 not run |
| T6 living Jack | 1/5 | |

**Zero-pass constitutional commitments: 17 of 23** (unchanged since the 13th
audit; T2.03's honest re-kind to `fixture` took sight back to zero and BA.02
VOIDed, so balance did not move). Six commitments have a passing claim:
curiosity (1 of 12 specs), hearing (1 of 6), one-brain/unison (1 of 21),
generality (1 of 4), memory-across-lives (1 of 3), damage (1 of 1).

---

## RANK 1 — four certificates were given a staleness alarm that is structurally inert, and the world underneath them has moved 420 lines

**Nothing on the ledger is shown to be false by this. The damage is that it
cannot be shown to be true, in the four places the project itself decided
needed watching.**

### The evidence

`74f8631` (2026-08-10) added `IMPL_DEPS = ["playground.py"]` to eleven world
certificates with an explicit rationale in the source:

> *"This spec certifies a property of the WORLD, so the world hashes into
> `impl_sha`. Change `playground.py` and this certificate goes stale loudly
> instead of standing over a world it no longer describes."*
> — `experiments/tests/pg_4_noisy_tv.py:56-59`

Four of the eleven have **no `impl_sha` at all**, because their runs predate
the field:

| spec | title | ran_at | commit |
|---|---|---|---|
| PG.4 | Noisy-TV panel traps naive curiosity | 2026-08-08 09:16:27 | `0d438f0` |
| PG.1 | Playground generates and is physically sound | 2026-08-09 14:22:09 | `76ccc6c` |
| PG.2 | Water works: buoyancy + drag | 2026-08-09 14:22:11 | `76ccc6c` |
| T2.20 | Episodic memory helps the next episode | 2026-08-09 14:23:07 | `76ccc6c` |

`playground.py` commits landing **after** those runs:

| commit | when | lines | what it says it did |
|---|---|---|---|
| `ddde954` | 2026-08-09 **14:24:45** | +195 / −6 | PG.8 PASS: Jack is in the playground |
| `910f3d6` | 2026-08-09 20:25 | +34 / −0 | *78 of the playground's 348 observation columns were identically zero* |
| `56fbf38` | 2026-08-09 22:36 | +143 / −3 | tmp_reaper: decide by liveness |
| `170cb52` | 2026-08-09 22:53 | +35 / −3 | PG.6 unblocked: MuJoCo renders under Xvfb |
| `29d189f` | 2026-08-10 08:06 | +23 / −2 | *Eye moved off the ladder axis*; impl_sha covers the world |

PG.1/PG.2/T2.20 were recorded at 14:22–14:23 and the next `playground.py`
commit landed at **14:24:45 — 98 seconds later.** PG.4 predates all five.

`run status` reports *"33 entries predate `impl_sha` and CANNOT be checked for
staleness"* and calls a re-run the fix. That is true and honest. What no
instrument says, and what I checked by hand, is **which of those 33 have a
declared dependency that has actually moved.** Four have. The other 29 are
T0/T1 harness and memory specs with no world dependency; those I am not
worried about.

### Why these four and not the other 29

PG.4 is not an isolated certificate. It is apparatus that two live things
stand on:

- **T2.08** — the *only* claim-kind PASS behind the **curiosity**
  commitment — does `from .pg_4_noisy_tv import _build`
  (`t2_08_curiosity_coverage.py:149`), declares
  `IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py"]`, and
  copies PG.4's `SUBSTEPS`, `EPS_HI/EPS_LO` schedule verbatim. T2.08's own
  staleness is tracked correctly. The certificate saying its rig behaves as
  claimed is not.
- **LC.03's registered run, in flight right now** (pid 2536994, 3 h 20 m
  elapsed of 15–20 h) disqualifies any arm on
  `PANEL_DWELL_MAX = 0.15 # PG.4's CONTROL_DWELL_MAX — the ported gate` and
  `DWELL_RADIUS = 2.0 # PG.4's dwell zone, verbatim`
  (`lc_03_survival_screening.py:182-183`).

To be fair about what is and is not at stake: **copying a pre-registered
threshold constant out of another spec's source is legitimate**, LC.03 says
so explicitly ("PORTS, not paraphrases"), and `borrow_metrics` is for
measurements, not constants — so this is *not* a T0.22 bypass and I am not
calling it one. The exposure is narrower and real: the evidence licensing
0.15 as the number that separates a trapped agent from a free one was
measured in a `playground.py` that has since changed by 430 lines, including
a commit that changed the observation vector PG.4's ICM agent consumes.

### The generalisable shape

A provenance mechanism cannot cover the records that predate it, and those
are **exactly the oldest and least re-run records** — which is to say the
foundations. Attrition does not fix this on its own: cheap T0 specs re-run
constantly and pick up `impl_sha` for free, while the expensive science specs
that most need it (PG.4 at CPU_LONG × 3 seeds, T2.20 at CPU_LONG × 3 seeds)
are precisely the ones nobody re-runs without a reason. Waiting is not a plan.
Lesson appended.

### Cost to close

Four CPU specs: PG.1 (`CPU`, 1 seed), PG.2 (`CPU`, 1 seed), PG.4
(`CPU_LONG`, 3 seeds — 496 s when it last ran), T2.20 (`CPU_LONG`, 3 seeds).
Order of tens of minutes of CPU, and it must wait for or run beside the two
detached jobs. If any of the four now FAILs against the current world, that
is not a regression the re-run caused — it is a finding the alarm was built
to surface and could not.

---

## RANK 2 — LC.03's darkroom control was amended onto the side its own pilot had already measured, so it can no longer fail. The spec has four working controls, not the five it declares.

### What happened

`87590a4` (the registration commit, 2026-08-13 15:23) amended two of LC.03's
five pre-registered controls after the seed-90 pilot:

| control | before | after |
|---|---|---|
| (a) statue | must **die soonest** — shorter mean life than every arm and null | mean life within **10 %** of the basal ceiling `e0/BASAL_B` |
| (e) darkroom | life_gain margin vs its null **`t ≤ −3`** (strongly negative) | margin **`t > −3`** (not strongly negative) |

### What is right about it, stated first and fairly

This is **not** silent loosening and I will not call it that:

- The amendment was made **before** the registered run, in the same commit as
  the registration, on a **disjoint pilot seed (90)**, with the old sides left
  in git history and the T1.02 precedent cited by name.
- It was **predicted in writing before it was needed.** The block it replaces
  (`PILOT CAUTION`) named both suspect sides, named the mechanism (PS.01's
  basal drain 0.00167/s vs active 0.0022/s), named the precedent (T2.08's
  passivity inversion) and committed to measuring rather than assuming. The
  builder called its own shot and then honoured it.
- **The claim gates did not move.** I diffed them.
- Control (a)'s new side is a genuine, falsifiable rig check: `statue mean
  life 180.0 s = e0/BASAL_B to 0.02 %` is arithmetic, and a phantom-damage
  fault (PS.03's servo scar) would break it. That amendment I have no quarrel
  with.

### Why (e) is still a finding

The pilot measured the darkroom's margin at **+49.7 s** — strongly *positive*.
The amended gate fires only at `t ≤ −3`, strongly *negative*. By the builder's
own closing arithmetic — passivity maximises life in W0 because basal drain is
below active drain, a structural property of the world, not a draw — the
amended side is not merely likely to pass, it is **guaranteed by the mechanism
the amendment itself documents.** A control that the author has proven cannot
fail has stopped being a control. It is now a tripwire on the world, which is
a useful thing to have and a different thing from what the docstring calls it:

> *"THE FIVE CONTROLS and their pre-registered sides (a wrong side is VOID — a
> control landing wrong means the instrument, not the hypothesis, failed)"*

Second, on the claim's own headline metric the darkroom is **not** an outlier
in the losing direction — it is inside the winning band:

| run | life_gain margin vs its null |
|---|---|
| ppo-lp | +54.6 |
| wm-efe | +52.0 |
| **darkroom (control)** | **+49.7** |
| wm-latent | +47.7 |
| dreamer-xs | +45.7 |
| ppo-needs | −1.8 |

An anti-curiosity control out-scoring three of the five claim arms on the
metric the spec is named for is the exact shape SYSTEM.md law 2 warns about.
Nothing in `_check` evaluates the darkroom against the **claim conjunction**,
only against `lg_margin` — so the spec cannot currently answer the question
"would my designed-to-fail arm pass my claim?"

### The honest counterweight, and a correction to the amendment's rationale

The claim conjunction is **not** naked. It requires `needs_rise > 0`
(`needs_satisfied_rate` rising, final third over first third) and `clt > 0`,
and a body that never acts cannot raise its needs-satisfied rate. That is a
real anti-passivity guard and it is gated, so a learned-passivity arm should
not be able to clear the claim.

But the commit message credits the wrong guard: *"dwell/chaos gates carry the
curiosity burden."* They do not. `panel_dwell` is PG.4's anti-noisy-TV gate
and `chaos_*` is the anti-chaos-seeking gate; **neither excludes passivity** —
a statue scores perfectly on both. The conjunct actually doing that work is
`needs_rise`. This matters because of what the pilot measured next.

### A forecast the owner should have, not a defect

**Every pilot arm read `needs_rise` NEGATIVE**, and `needs_rise > 0` is a
gated conjunct. LC.03 returns VOID unless ≥ 2 arms clear *everything*. The
builder's disclosed bet is that the registered envelope (8.3× longer, e0 1.0
vs 0.3) flips the sign on at least two arms. That bet is riding **~90
core-hours** already 3 h 20 m into a 15–20 h wall clock, and no intermediate
envelope was measured to test the sign flip first. If it does not flip, LC.03
VOIDs, LC.04/LC.05 stay blocked, and the loss is a day of the box. Disclosed
in the registration commit; I am restating it because the ladder log's
summaries did not carry it forward.

---

## RANK 3 — 11.35 Kaggle GPU-hours expire in ~2.2 days and nothing is dispatched

This is a repeat of the 13th audit's **B4**, unfixed and now closer to the
deadline.

| | |
|---|---|
| W32 (`%U`, Aug 9–15) kaggle used | **18.65 h** of 30 (18.5322 productive + 0.1225 failed) |
| remaining before Sunday 2026-08-16 reset | **11.35 h** |
| last GPU dispatch | 2026-08-13 06:09 — **12 h 30 m ago** |
| GPU specs currently runnable | **11** (T2.04/T2.06/T3.07/T4.02 at `gpu<20min`; T2.05/T2.09/T2.11/T3.01/T3.06 at `gpu<2h`) |
| W31 precedent | closed at **37.4554 h** of a 30 h ceiling |

The builder **correctly declined** B4's specific suggestion (re-running
T2.01), and its reasoning in `a3b12f6` is sound and I am not re-litigating it:
v5 already ran clean post-critic-fix, curves flat ~5.15 from 100 K to 700 K
steps, so a re-run is a seed redraw against a 5σ bar — run-until-pass. That
was the right call.

But declining a use is not finding one. The hand-off has pointed the expiring
hours at T3.07/T4.02 for three consecutive iterations and all three iterations
went to CPU work instead. Both are `gpu<20min` and both **need implementing**,
which is the actual blocker — and implementing a 20-minute GPU spec is one
iteration's work. GPU jobs run remotely and do not compete with LC.03 or BA.02
for this box, so "the box is busy" is not a reason.

Nine of the eleven runnable GPU specs need implementing. That is the real
state: **the ladder has no GPU work ready to dispatch**, which is why free
hours evaporate. T3 is 0/14 and T4 is 1/23 largely for the same reason.

---

## RANK 4 — three builder iterations were lost to credit exhaustion and nothing counted them

```
2026-08-13T10:07:04 iteration start — 78/166 demonstrated, load 0.05
You've hit your session limit · resets 1pm (UTC)
2026-08-13T10:07:07 iteration end rc=1 — 78 -> 78 demonstrated
```
Identical at 11:07 and 12:07. **Three hours of builder capacity, 12.5 % of the
day**, gone in 3–4 s each. First occurrence of this failure mode in the log
(the seven rc=1 iterations on 2026-08-09 were a different cause).

It self-resolved at 13:07 and no work was corrupted. The finding is that
**nothing in the system knows it happened**: the message is a stdout string,
no counter increments, no retry is scheduled, and the 13:07 iteration began
with no idea it was inheriting a three-hour gap. `DECISIONS_NEEDED.md` already
carries *"Claude credits are the binding resource and are unmetered (OPEN,
owner)"* — this is that entry's first measured instance and it should be
attached to it.

---

## 1. Integrity of the ledger — clean

Checked all **79 PASS** records mechanically:

- **Implementation exists** for every one: 0 missing test files.
- **Commit exists and is reachable from HEAD** for every one: 0 missing,
  0 unreachable, 0 stamped `+dirty`.
- **Every PASS is a registered spec**: 0 orphans.
- **Control declared → control metrics recorded**: 0 exceptions.
- **Two PASSes declare no control** — T0.01 (*Repo imports clean*) and T0.10
  (*Kaggle job round-trip*). Both are structural preconditions with
  `null_baseline="n/a"`; a control is meaningless for them. Correct as-is.
- `run stale`: **one** stale claim — BA.02, recorded VOID, code since changed
  to v2. The builder recorded it, diagnosed it and shipped v2 in the open; the
  entry is a VOID, so it asserts nothing. Not a finding.
- **33 PASS records predate `impl_sha`** (8 × T0, 9 × T1, 15 × T2, 1 × T6).
  Four of them are RANK 1; the rest have no declared dependency that has moved.

## 2. Thresholds and controls over time — no silent loosening

53 commits touched the registry files in 7 days. Every candidate I could find:

- **LC.03 (a) and (e)** — analysed at RANK 2. Openly amended, pre-registration,
  disjoint pilot seed, claim gates unmoved, old sides in git history. (e) is
  substantively weaker; it is not silent.
- **BA.02 v1 → v2 (`88507ab`), claimed "zero constants moved" — VERIFIED
  TRUE.** I diffed `ad24b62 → 88507ab`: the `^[A-Z_]+ = ` constant block is
  **byte-identical** and `_check` is **byte-identical**. The v2 changes are
  measurement scheduling only (interleaved CEM and eval, boundary-spawn
  exclusion, a `drift_recheck` metric reported never gated). The VOID was
  recorded first, as-is, before the diagnosis — the correct order.
- **No seed count was reduced anywhere.** The `-seeds=3` diff lines are all
  line-rewrites where the same `seeds=3` reappears on the `+` side.
- **No `_check` gained a disjunctive escape.** I read every `+` line adding
  `or` in `experiments/tests/` over 7 days; all are `None`-guards,
  `dict.get(...) or {}` defaults, VOID-propagation, or prose.
- **T2.08's `COV_MIN` 0.70 → 0.50** was the 13th audit's RANK 1. It now has an
  executable guard: **T0.27** (`90676c6`) stamps `supersedes_fail` onto any
  verdict amending a FAIL and makes "commit the failing implementation before
  re-running" auditable. Closed properly.
- **Budget changes** (T2.08 `GPU→CPU`, LC.03 `CPU_LONG→CPU_DAYS`, T0.09
  `CPU→GPU`) are cost re-estimates, not gates. No finding.

## 3. Drift from the goal — none found

Every unit in the last 24 h traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| LC.03 pilot → registered run | *"one interconnected brain… learns its world by living in it"* — the learning core the whole ladder feeds |
| BA.02 implement / pilot / registered VOID / diagnosis / v2 | *"EVERY SENSE A HUMAN HAS… proprioception & balance"* + *"a test that could have failed"* |
| Overseer B1/B2/B3/B5/B6, T0.27, T0.17, T0.12 | SYSTEM.md *"is the machine better than I found it?"* — all trace to named scars |
| CHAMPIONS refresh, wk3 LESSONS, journal | the honesty apparatus |

**The converse question is where the discomfort is.** T3 is **0/14**, T4 is
**1/23**, T5 is **0/27**. Everything GOAL.md calls the thesis —
*"earn your parameters"*, *"senses fused, each proven load-bearing"*,
*"continual learning without forgetting, curiosity that drives real
exploration"* — has **zero passing specs**. The two T5 attempts of the last
24 h (BA.02 twice) both VOIDed on rig validity. That is the gate working, and
it is also the honest statement of where we are.

Curiosity: 1 passing claim of 12 specs, and RANK 1 puts its rig certificate
out of audit. All-senses fusion: 1 of 21. Learning-by-living: LC.03 is the
attempt and it is mid-flight. These are exactly the three the audit brief
predicted would be quietly neglected; they are not neglected — they are being
worked on and they are hard.

## 4. Builder liveness — healthy

| | |
|---|---|
| iterations since 2026-08-12 19:00 | **24** |
| ended rc=0 | **21** |
| ended rc=1 | **3** (all credit exhaustion — RANK 4) |
| ladder | **74 → 79 PASS**, 165 → 169 specs |
| in flight | LC.03 registered (pid 2536994, 3 workers @ 100 % CPU, 3 h 20 m); BA.02 v2 pilot (pid 2568658, 16 m) |
| loop paused? | **No** — `.paused` and `.loop-paused` both absent |

One PASS went **backwards** in the window (74 → 73 at 20:07) when BA.01 v3
VOIDed and took back v2's PASS. A scoreboard that can go down is working.

## 5. Compute honesty — reconciles

W32 (`%U`, Sunday-start — deliberate, documented at `gpu.py:277-286`):
kaggle **18.5322 h productive + 0.1225 h failed**, colab **0.7616 + 1.053
failed**. The 6.38 h gap between `weeks` and `charged_jobs` is
`MAX_TRACKED_JOBS` pruning older entries, not missing hours.

**Waste: 1.18 h of 19.41 h (6.1 %), all diagnosed.** 0.1225 h in two errored
T2.03 kernels (root-caused to the upstream cudnn pin, fixed in `643f542`);
0.9914 h in a colab job whose artifact fetch failed; 0.0616 h likewise.

**11.16 h bought T2.01's FAIL** (`08444b2`, two 5.58 h kernels). A FAIL is a
result on the ledger; that is not waste.

**I re-checked the "15× scar"** — the 35 330 s reattach charge sitting in
`gpu_submissions.jsonl` for a kernel whose metered window was 2 361.88 s. The
10th audit found it and it is **fixed**: `gpu.py:654-668` now closes the meter
from `_kaggle_log_window`, Kaggle's own report, and prints a loud upper-bound
warning when no log is readable. The stale log line is the historical record
of the fault, and the budget was never wrong (the idempotency key held). No
new finding.

## 6. Stuck decisions

**D1 — "Does the 57M trunk stay in the control path?" — open since 2026-08-04,
evidence marked complete, 9 days.** It is the only thing blocking the
locomotion branch, it holds the ladder's single FAIL (T2.01) and single T2
VOID (T2.02), and the builder's decision to skip the T2.01 re-run turns on it.
The recommendation (A: freeze the trunk, small policy head) has not changed and
no new evidence has arrived to change it. Appended to `DECISIONS_NEEDED.md`
with the current arithmetic.

**Nothing is blocked that the system could resolve itself.** D2 was
correctly taken off the owner's desk by ledger replay (`2eaf2d0`) and
recorded in `DECISIONS_RESOLVED.md` with its loser and a re-open trigger —
which is the pattern the system should keep using.

**No owner-decision was acted on without being recorded** in this window.

## 7. Bakeoff hygiene — clean

Three entries in `DECISIONS_RESOLVED.md`. PS.01/J correctly returned **VOID**
(three arms under the 3σ gate) rather than crowning a winner. PS.01/J2 declares
`screen` mode with a written rationale for why its arms are observables rather
than learners, and its winner (impact_speed, 10.32σ over null) beats the
runner-up by 2.66σ — outside the noise margin. D2 was resolved by ledger
replay rather than `run_bakeoff`, with the method and its justification stated;
its loser and re-open trigger are recorded. No decision made without a learning
gate, no VOID read as a verdict, no winner inside the margin.

## 8. The honest summary — are we closer to a creature, or just to more green ticks?

**Closer to a creature, but the distance closed today was in the instrument,
not in Jack.**

The ledger went 74 → 79 and 4 of those 5 are Tier 0 harness re-stamps and
guards. The two units aimed at Jack himself — BA.02 (*he catches himself*) and
LC.03 (*which core learns to survive*) — produced **one VOID and one run still
in flight.** BA.02 has now VOIDed twice: once on rig validity, once when its
apparent +0.40 s gain turned out to be world-state drift measured by block
order. Balance is still a zero-pass commitment.

**That is not a bad day. It is what an honest day looks like at this tier.**
The BA.02 diagnosis is the best work in the window by a distance: the builder
had a PASS-shaped pilot result, went looking for why the registered run
disagreed, found that its own claimed gain (+0.3958 s) equalled the drift
between its first two eval blocks (−0.40 s) to two decimal places, and threw
away its own result. Measuring your instrument until it takes back your finding
is the whole discipline. Tier 0 being 28/28 is what buys the right to trust
Tier 5 when it eventually goes green.

**The uncomfortable arithmetic.** T3 0/14, T4 1/23, T5 0/27 — every claim
GOAL.md actually makes is unproven, and 17 of 23 constitutional commitments
have nothing passing. The ladder is 79/169 by count and roughly 0/66 by thesis.
The green ticks are real and they are almost all apparatus. Nobody is pretending
otherwise — `coverage.py` says 17 zero-pass to your face every audit, and the
builder re-kinded its own T2.03 sight PASS *downward* to `fixture` when it
noticed the winning arm was constitutionally barred from being inside Jack.
A system that voluntarily reduces its own score is not fooling itself.

**What would change the answer.** LC.03 landing with ≥ 2 arms clearing
everything would be the first real evidence that anything in this design
learns to survive by living. It is the single highest-value thing in flight,
and its `needs_rise` conjunct is a genuine coin-flip that no intermediate
measurement de-risked.

---

# FOR THE BUILDER

**B1 (RANK 1, do this first — it is cheap and it is foundational).** Re-run the
four certificates whose declared `IMPL_DEPS` have moved and whose staleness
alarm cannot fire: **PG.4, PG.1, PG.2, T2.20**. All four are CPU; PG.4 was
496 s × 3 seeds when it last ran. Run them beside or after the detached jobs —
do not disturb pid 2536994.
- PG.4 first: T2.08 (curiosity's only claim-kind PASS) imports its `_build`,
  and LC.03's live run gates every arm on its `0.15` / `2.0` dwell constants.
- If one of them now FAILs against the current `playground.py`, record it as a
  FAIL. That is the finding, not a regression you caused — and it is worth more
  than the PASS it replaces.
- Then close the class rather than the instance, in the T0.22 tradition: add a
  property to an existing T0 spec (T0.17 or T0.21 are the natural homes)
  asserting **no PASS record may declare `IMPL_DEPS` while lacking `impl_sha`
  when any declared dependency has commits after its `ran_at`.** Today that
  property fires on exactly these four; the point is that the next one cannot
  hide. `run stale`'s *"33 predate impl_sha and cannot be checked"* line should
  split into "cannot be checked" and **"cannot be checked AND its declared
  dependency has since moved"** — the second number is the one that matters and
  it is currently invisible.

**B2 (RANK 2, before LC.03 records).** Two edits, neither of which touches a
gate:
- The docstring says *"THE FIVE CONTROLS"*. After the amendment it is four
  controls and one world-tripwire. Say so in the text — control (e) can no
  longer fail by the mechanism the amendment itself documents (basal drain
  0.00167 < active 0.0022 ⇒ passivity wins length, structurally, not by draw).
  A reader counting five falsifiers is counting one that is not there.
- Correct the rationale: *"dwell/chaos gates carry the curiosity burden"* is
  wrong. A statue scores perfectly on `panel_dwell` and `chaos_*`. The conjunct
  that actually excludes learned passivity is **`needs_rise > 0`**, and that is
  the sentence the next reader needs, because it is also the conjunct every
  pilot arm failed.
- Consider, for LC.04 rather than retrofitting LC.03: run the darkroom through
  the **full claim conjunction** and require it to fail there. A control
  evaluated on one conjunct of a six-conjunct claim cannot answer "would my
  designed-to-fail arm pass my claim?" — and this one already out-scores three
  claim arms on the headline metric.

**B3 (RANK 3, time-boxed — 2.2 days).** 11.35 Kaggle hours expire Sunday
2026-08-16. The blocker is not the box and it is not D1: **nine of the eleven
runnable GPU specs are unimplemented.** Implement one `gpu<20min` spec —
T3.07 or T4.02, both of which your own hand-off has named three iterations
running — and dispatch it. T3 is 0/14 and T4 is 1/23; a single ablation
landing would be the first entry in either tier that is about Jack. GPU work
runs remotely and does not compete with LC.03 or BA.02 for this box.

**B4 (RANK 4).** Make credit exhaustion countable. Three iterations
(10:07, 11:07, 12:07 today) died in 3 s on *"You've hit your session limit"*
and nothing in the system registered it: no counter, no retry, no note for the
next iteration. At minimum, have `ladder_loop.sh` detect the string, write a
distinguishable marker, and let the next successful iteration read how many
predecessors it lost. 3 of 24 iterations is 12.5 % of a day.

**B5 (carried, unfixed from the 13th audit — B7).** The `hardware` field still
reads `aarch64/Linux/torch2.8.0+cpu/cpu` on all **8** GPU-run records,
including T2.03's PASS. Nothing is misreported (the truth is in
`metrics.gpu`/`metrics.backend`), but the field a reader trusts for *"where did
this run"* says "cpu" for eight GPU results.

**B6 (carried, acknowledged).** B3 bullets 1–2 from the 13th audit remain open
— the live-receipt property and charge-at-attempt/reconcile-on-result. You
correctly called them one designed unit; they are still owed.

---

# FOR THE OWNER

**1. D1 has been on your desk for 9 days and it is the only thing holding the
locomotion branch.** *Does the 57M trunk stay in the control path?* The
evidence has been complete since 2026-08-09 and nothing since has changed it:
three independent runs at matched env-steps show the 57M trunk at 261/318
return against a 54 K-parameter MLP at 531 and a 125 K net at 530 — the trunk
failing a 3σ learning gate that a 125 K net clears at 7σ. The loop's
recommendation is **A: freeze the trunk for control, small dedicated policy
head, trunk keeps perception/language/memory**. One line settles it; *"do what
the measurements say"* will be read as A. Full entry and options in
`docs/DECISIONS_NEEDED.md`.

**2. Your credits are the binding resource and they bit for the first time
today.** Three builder iterations (10:00, 11:00, 12:00 UTC) died instantly on
a session limit — 12.5 % of the day's capacity, and the system had no way to
notice. This is the `DECISIONS_NEEDED` entry *"Claude credits are the binding
resource and are unmetered"* getting its first measurement. Nothing needs
deciding today; you should know the ceiling is now real rather than
theoretical.

**3. Where the project actually stands, without the scoreboard flattery.**
The ladder reads 79 of 169. The tiers that carry your commitments read **T3
0/14, T4 1/23, T5 0/27**, and **17 of your 23 constitutional commitments have
nothing passing.** Everything green is apparatus — a harness that has become
very good at refusing to lie. Two attempts at a real claim in the last 24 h
both returned VOID, one of them because the builder caught its own +0.40 s
"gain" being an artifact of measurement order and threw the result away. That
is the machine working exactly as you specified it, and it is also the reason
progress on Jack himself looks slow: it is slow, and the alternative was a
README that said "Working".

**4. One thing in flight is worth more than everything else combined.**
LC.03 — which of five learning cores learns to survive at all — has been
running ~3 h of an expected 15–20 h, ~90 core-hours. If ≥ 2 arms clear its
gates it is the first evidence in this project that anything in the design
learns to survive **by living**. Its riskiest gate (`needs_satisfied_rate`
rising) read negative on every arm in the pilot, and the bet is that an 8.3×
longer envelope flips it. It may well come back VOID. You should hear that
from the audit before you hear it from the result.
