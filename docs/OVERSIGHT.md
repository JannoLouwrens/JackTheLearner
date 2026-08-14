# OVERSIGHT — 16th audit, 2026-08-14 06:45 UTC

## VERDICT: ON TRACK

The ledger is sound, and I checked it harder than the summary line suggests.
All **79 PASS** records name a commit that still exists and an implementation
on disk, and — the check that matters — for every PASS whose spec declares a
control, `_check(m, c)` **actually references its control parameter**. I
verified that by AST over the second formal argument, not by the presence of
`control_metrics` in the record. Zero exceptions. Only T0.01 and T0.10 lack
control metrics, and both correctly declare `control=None`.

**No findings in section 2.** Over 7 days and 122 commits touching the
registry and tests, I diffed 724 constant-definition lines. Every in-place
numeric change moved in the *strengthening* direction or resized apparatus.
Not one threshold was loosened, no control was deleted or weakened, no
assertion was removed, no seed count was reduced. Saying that plainly is the
honest result.

The one live finding is not about a claim already on the ledger. It is about
the claim the builder is minutes away from buying: **T2.04's GPU cost estimate
was extrapolated from a model roughly 256× smaller than the one the production
kernel will actually instantiate, and it is queued to be dispatched against
the last expiring Kaggle hours of the week.**

**169 specs · 79 PASS · 3 FAIL · 2 VOID · 0 NOT_RUN.** Builder: **24
iterations in 24 h, 21 rc=0, 3 lost to a session limit.** PASS delta over the
window: **+1**.

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

| tier | pass/total | |
|---|---|---|
| T0 harness | 28/28 | complete |
| T1 primitives | 13/13 | complete |
| T2 vs null | 36/59 | T2.01 FAIL, T2.02 VOID |
| **T3 earn your parameters** | **0/14** | T3.07 FAIL |
| **T4 unison** | **1/23** | T4.02 FAIL |
| **T5 THE CLAIMS** | **0/27** | BA.02 VOID |
| T6 living Jack | 1/5 | |

**Zero-pass constitutional commitments: 17 of 23**, unchanged for four audits.
Six commitments have a passing claim-kind spec: curiosity (1 of 12 specs),
hearing (1 of 6), one-brain/unison (1 of 21), generality (1 of 4),
memory-across-lives (1 of 3), damage (1 of 1). The rest is fixtures and
sensors — real work, correctly not credited as the commitment.

`run stale` reads clean: **one** flagged entry, T2.02, which is VOID and
deliberately left flagged pending D1. The 15th audit's B1 (declaration-free
staleness) is verifiably closed — of 28 pre-`impl_sha` records, 1 stale by
content, 27 verified byte-identical, **0 unanswerable**.

---

## RANK 1 — T2.04's 2.0 h declaration was derived from a 256×-smaller model, and it is about to spend the week's last GPU hours

**Nothing on the ledger is false because of this.** The damage is prospective
and it is on a clock: the dispatch is armed, the quota dies Sunday, and if the
estimate is wrong the failure mode is a timed-out kernel that bills real hours
and records nothing.

**The measurement and the production run do not use the same model.**

`_submit` (`experiments/tests/t2_04_behaviour_cloning.py:331-341`) declares
`est_hours=2.0, timeout_s=18000`, justified in its own comment:

> *"the tiny smoke (d64, 30 steps, batch 256) ran >39 s/step on this box, so
> the production kernel (7200 train steps + 12000 single-row evals) overruns
> the original 3300 s cap even at a 40x GPU speedup."*

That extrapolation runs from the smoke's configuration to the production
kernel's step count while holding cost-per-step fixed. But the two do not
share a configuration:

- The smoke calls `remote_run(..., pipe_kwargs={"d_model": 64, "n_layers": 2})`.
- Production is `JOB` → `remote_run(__SEEDS__)` → `pipe_kwargs=None` →
  `PipelineConfig()` defaults, which are **`d_model=512, n_layers=8`**
  (`TrainingPipeline.py:64-65`).

Trunk cost scales roughly as `n_layers × d_model²`: `(8/2) × (512/64)² =`
**256×**. The comment's per-step figure was measured on the very parameters
that were shrunk to make the smoke cheap. Extrapolating a timing across those
parameters is circular.

**The cited per-step number is also already superseded by the smoke itself.**
The `>39 s/step` figure came from a partial observation at 05:26. The smoke's
first `_train_bc` has since *completed*: `/data/tmp/t204_smoke.log` stamps
`obs normalized` at 2.1 s and `bc trained` at 3098.2 s — **3096 s for 30 steps
= 103 s/step**, 2.6× the number the timeout was sized from. At 103 s/step and
the comment's own 40× GPU factor, 7200 steps = **18,540 s — already past the
18,000 s cap before any scale correction.**

**Why this can't be caught downstream.** The builder's standing instruction is
"dispatch on `SMOKE OK`". The smoke's verdict asserts
`det_ok and dims_ok and finite` — it has no cost gate, so it will report OK and
the dispatch will fire. And `Budget.afford()` gates on the **declared** estimate
while `charge()` bills **actual** elapsed; the code says so itself
(`experiments/gpu.py:381-384`): *"nothing prevents an overrun — but an overrun
that leaves no mark is how week 31 closed at 37.4554 of a 30.0 h ceiling."*
A 2.0 h declaration passes `afford()` against 10.9 h remaining no matter what
the run actually costs.

**In fairness:** a P100 is far better at dense `d512` matmuls than this ARM CPU
under LC.03's three-worker load, so 256× will not appear as 256× wall-clock —
large batches are exactly where a GPU recovers. That is precisely the point.
Nobody has measured it, the direction of the error is unknown, and the number
in the file is a budget commitment against a quota that expires in two days.

**Status of the smoke:** not hung. It is in its final stage (the shuffled-control
`_train_bc`, second pipeline built at 3101 s), running at 120% CPU, and should
report `SMOKE OK` around **07:10 UTC**.

---

## RANK 2 — 6.3849 Kaggle hours in the week counter have no per-job record, and the loop is rationing against them

`weeks["2026-W32"]["kaggle"] = 19.0931`, but `charged_jobs` itemises only
**12.7082 h** of successful W32 Kaggle work. The failed side reconciles
*exactly* (0.1225 h, two jobs), as does all of Colab (0.7616 ok + 1.0530
failed). Only the Kaggle ok-side is short, by **6.3849 h**.

I traced it through the budget file's own git history. The gap opened at
`92931a6` — when `charged_jobs` held **zero** entries — and has been frozen at
exactly 6.3849 through the twelve commits since:

```
92931a6  weeks.kaggle=6.3849    sum(charged_jobs ok)=0.0000    gap=6.3849  njobs=0
0a7540e  weeks.kaggle=11.9635   sum(charged_jobs ok)=5.5786    gap=6.3849  njobs=1
...
15afd8c  weeks.kaggle=19.0931   sum(charged_jobs ok)=12.7082   gap=6.3849  njobs=12
```

**This is a legacy opening balance, not a live leak.** It predates per-job
records. Ring-buffer eviction is ruled out: `MAX_TRACKED_JOBS = 500` against 12
tracked jobs. Every charge since has been attributable.

It also errs in the **safe** direction — it over-states spend, so it cannot
hide waste. But it has an operational cost. The loop computes "hours remaining"
from `weeks`, so it has been reporting ~10.78 h and rationing GPU work against
that figure for three days. The honest range is **10.91 h to 17.17 h**; the
lower bound is being treated as the number while a quota expires Sunday. I am
**not** asking anyone to raise the counter — the conservative direction is the
right one to keep — only to label the balance so a reader knows it is a floor.

**Related, and clean:** the 15× reattach scar is fixed and I verified it on
live data. The 08-12 06:56 receipt reports `charge_seconds=35330` for job
`jack-ladder-1786482462`, but `charged_jobs` bills it **0.6561 h** — the 2361.88 s
window Kaggle's own log reports. The meter now reads the kernel's report, not
the local clock, exactly as designed.

---

## 1. Integrity of the ledger — no findings

All 79 PASS: implementation present, `commit` resolves in git, control declared
where the spec declares one, and `_check`'s control argument is referenced in
every case. `control_metrics` present for all 77 specs that declare a control;
absent only for T0.01 and T0.10, which declare `control=None` correctly
(repo-imports-clean and a Kaggle round-trip have no sabotage condition).

## 2. Thresholds and controls over time — no findings

122 commits over 7 days touched `registry.py`, `registry_expansion.py` and
`experiments/tests/`. Every in-place numeric change:

| constant | change | direction |
|---|---|---|
| `N_PERM` (XL.00) | 2 000 → 100 000 | stronger |
| `N_CALIB` (VO.01) | 60 → 400 | stronger |
| `N_OCC` (VO.01) | 160 → `2 * N_TRAIN` = 600 | stronger |
| `N_DECISIONS` (PS.01 v2) | 3 000 → 4 500 | more data |
| `TRAIN_MINUTES_PER_SEED` (T2.01) | 30 → 110 | more compute |
| `RES` (PG.6) | 64 → 96 px | operating point, set by pilot |
| `N_PROPERTIES` (T0.17/T0.22) | 7→8→9, 12→13→14→15 | more gated properties |
| BA.02 `PILOT-FINAL` set | **values unchanged**, comments annotated with pilot numbers | none |

Controls only grew: `"Two named broken meters"` → `"Three named broken
mechanisms"` → `"Five named broken mechanisms"`. No `control=` field was
deleted or softened; several were newly declared. No `assert` was removed. No
seed count was reduced. The `or`s added this week are in rig/geometry helpers
and VOID-side validity gates (which tighten), not in claim gates.

BA.02's V3 envelope amendment (`5dc0620`) deserves the specific note that it
*passes* this test: it moved one tilt-draw constant, justified by a measured
headroom probe, with every gate, arm, control and eval structure byte-identical
— and the claim itself still came back VOID afterwards, which is what an honest
amendment looks like.

## 3. Drift from the goal — none in what was built; the gap is in what is not

Last 24 h of builder work, each against its GOAL.md sentence:

| unit | GOAL.md sentence it serves |
|---|---|
| T4.02 (fusion-boundary gradient balance) — FAIL | *"a genuinely unified brain where every sense is load-bearing"* |
| T3.07 (ablate mood conditioning) — FAIL | *"components that must EARN their parameters via ablation or be deleted"* |
| BA.02 v3 + diagnosis — VOID | proprioception & balance in the sensory inventory |
| B1 staleness detector, T0.09/T1.07 re-runs | *"protects the honesty of watching what happens"* |
| B3 GPU attributability | same |
| T2.04 (behaviour cloning vs 1-NN null) | Tier 2, *"each must beat a dumb baseline"* |

**No drift.** Every unit traces to a sentence.

The converse is the standing finding, and it is structural rather than new:
**Tier 3 is 0/14 and Tier 5 — THE CLAIMS — is 0/27.** Curiosity has 1 passing
claim of 12 specs; all-senses fusion has 1 of 21. 43 of the 79 passes are Tier
0 and Tier 1 — harness and primitives. The instrument is excellent. The
creature is barely measured.

## 4. Is the builder alive and productive? — alive, honest, and flat

24 iterations in the 24 h to 06:18 UTC. **21 rc=0.** The 3 failures were
consecutive (08-13 10:07, 11:07, 12:07), each ~3 s, each with one line in the
log: `You've hit your session limit · resets 1pm (UTC)`. **Credit exhaustion,
three iterations lost.** The 14th audit's B4 fix (`b0b9506`, detect + retry on
fallbacks) shipped at 19:21 that evening — *after* the incident — and no
session-limit line has appeared since, so the guard is in place but has not yet
been exercised in the wild.

PASS delta over the window: **78 → 79**, the single increment landing at 14:20
on 08-13. The last **16 consecutive iterations moved the count by zero** while
the registry grew 166 → 169. Pass rate 79/169 = **46.7%**.

That is not a stall, and I want to be precise about why. Those 16 iterations
produced T4.02 (FAIL), T3.07 (FAIL), BA.02 v3 (VOID with an arithmetic
diagnosis), the B1 staleness class-closer, B2, B3, and T2.04's implementation.
Three of those are **negative results about Jack** — which is the ladder
working as designed, and worth more than green ticks about the harness.

Two iterations (04:07–04:17, 06:07–06:18) ended within ~11 minutes having done
essentially nothing but confirm the smoke was still running. Two of 24
iterations spent waiting on a CPU job that a GPU spec is blocked behind.

**In flight and healthy:** LC.03's registered run (pid 2536994, ~15 h 20 m of
an expected 15–20 h, three workers at ~100%) lands today. Do not relaunch it.

## 5. Compute honesty

Beyond RANK 2: W32 waste is **1.1755 h of 14.64 h billed (8.0%)** — 1.053 h
Colab and 0.1225 h Kaggle — and every hour of it has a documented cause (the
cudnn/torchvision pin failures, and a Colab session that died with its watcher).
The deadline guard added afterward (`1a01e69`) is why the latter cannot recur.

What W32's 12.71 productive Kaggle hours bought, honestly: T2.03 PASS (0.33 h),
T1.07 re-run PASS (0.44 h), T4.02 FAIL (0.12 h) — **0.89 h**. The remaining
~11.8 h went to the T2.01 family and produced a convergence diagnosis rather
than ledger movement. That is not waste — the diagnosis is precisely why the
builder correctly *declined* to re-submit T2.01 rather than redrawing seeds
against a 5σ bar. But it is the shape of the week: ~93% of GPU hours bought one
argument, correctly made.

**Kaggle W32 remaining: 10.91 h by the counter, up to 17.17 h in truth. Resets
Sunday 2026-08-16.**

## 6. Stuck decisions — no finding

**D1** (does the 57M trunk stay in the control path) is **10 days open** and is
the largest single blocker on the board. The 15th audit's correction — that
the option menu contained a choice the PLASTIC-ONLY decree forbids — was acted
on correctly by the builder: `7ce25c4` bars `frozen-trunk+head` on the
CHAMPIONS D1 seat *pending reconciliation*. Barred, not resolved. That is the
right handling.

**D7** (MovementMoodCoupling failed its ablation: delete / redesign / accept as
cosmetics) and **D8** (BA.02 is unmeasurable in the rover body) are both fresh,
both carry their arithmetic, and both are genuinely owner-level — they turn on
what the owner wants Jack to *be*, not on a measurement the system could take.
Neither could be settled by a bakeoff the loop could run itself.

Nothing was quietly acted on without being recorded. No decision on the board
now has enough evidence to be decided that has not been escalated.

## 7. Bakeoff hygiene — no finding

Two bakeoff records. **PS.01/J is VOID and was not treated as a verdict** — all
arms sat below the 3.0σ learning gate, and the honest consequence (a re-run as
PS.01/J2) is what happened. **PS.01/J2's winner** (`impact_speed`) beats the
runner-up by 2.66σ and the null by 10.32σ — comfortably outside the noise
margin — under an explicit `screen` gate mode with a written rationale for why
these arms are observables rather than learners. **D2** was resolved by ledger
replay with the method stated and the losing reading preserved. The file's own
header discloses the nine `TEST` fixtures removed on 2026-08-09.

## 8. The honest summary

**Are we closer to a curious humanoid that climbs the ladder, or only to a
longer list of green ticks?**

Neither, this window — and that is the most interesting answer the system has
produced in a while. The ladder gained **one** green tick in 24 hours. What it
gained instead was three refutations: mood conditioning does not measurably
change Jack's behaviour (T3.07), the fusion boundary's gradients are *not*
balanced across senses (T4.02, and the imbalance runs touch-over-audio, not the
direction anyone documented), and the rover body has no actuator with
directional catch authority, so a whole claim is unmeasurable until the body
changes (D8).

Those are worth more than ticks. Every one of them is a fact about Jack rather
than about the harness, and every one arrived because a gate was allowed to
fail. A system that wanted to look good could not have produced them.

But the structural picture has not moved for four audits, and I will not soften
it: **Tier 5 — the tier that holds the actual thesis — is 0 of 27.** Tier 3 is
0 of 14. 17 of the owner's 23 constitutional commitments have nothing passing
behind them. 43 of 79 passes are harness and primitives. We have built a
genuinely excellent instrument and pointed it at a creature that is still
mostly hypothetical. The honest reading of this week is that we are getting
much better at *knowing* whether Jack is learning, and not yet much better at
Jack learning.

The single most valuable thing that could happen next is not another T0 spec.
It is D1 being answered, because the learning core is what Tier 5 is waiting on.

---

## FOR THE BUILDER

**B1 (RANK 1, before the T2.04 dispatch — this is time-critical).**
Do **not** dispatch T2.04 on `SMOKE OK` alone. The smoke measured
`pipe_kwargs={"d_model": 64, "n_layers": 2}`; `_submit`'s `JOB` calls
`remote_run(__SEEDS__)` with `pipe_kwargs=None`, so the production kernel
instantiates `PipelineConfig()` defaults `d_model=512, n_layers=8`
(`TrainingPipeline.py:64-65`) — a trunk ~256× more expensive per step
(`n_layers × d_model²`). The `est_hours=2.0, timeout_s=18000` declaration at
`t2_04_behaviour_cloning.py:331-341` is derived from the small config and is
not evidence about the large one. Do one of:

  (a) **Preferred.** Dispatch a short probe kernel that times ~5 `_train_bc`
      steps at the *production* config on the target GPU, then re-derive
      `est_hours` and `timeout_s` from that number and commit the arithmetic in
      the comment. A probe of a few minutes protects up to 5 h of a quota that
      expires Sunday.
  (b) Pass the smoke's `pipe_kwargs` explicitly through `_submit` into
      `remote_run` so the thing measured and the thing run are the same model —
      and say so in the spec, because it changes what the claim is about.

Either way, **correct the comment**: the smoke's first `_train_bc` has now
completed at **3096 s / 30 steps = 103 s/step**, not the `>39 s/step` cited.
At 103 s/step with the comment's own 40× factor, 7200 steps = 18,540 s, which
already exceeds the 18,000 s cap before any scale correction.

Note the asymmetry that makes this worth a probe: `afford()` gates on the
declared estimate and `charge()` bills actual elapsed (`gpu.py:381-384`), so a
wrong declaration is not caught anywhere — it is only observed afterward, which
is how W31 closed at 37.4554 h of a 30 h ceiling.

**B2 (RANK 2, cheap, no urgency).**
Label the unattributable Kaggle balance. `weeks["2026-W32"]["kaggle"]` carries
**6.3849 h** with no `charged_jobs` row — frozen since `92931a6`, when
`charged_jobs` was empty, so it is a pre-per-job-records opening balance rather
than a leak (`MAX_TRACKED_JOBS=500` vs 12 tracked jobs rules out eviction).
**Do not lower the counter** — over-stating spend is the safe direction. Do add
an explicit `opening_balance` field or file comment recording it, and make
whatever prints "hours remaining" report the range (**10.91–17.17 h**) with the
unattributable component named, so the loop stops rationing against a floor it
believes is a fact. This matters this week specifically, with a quota expiring
Sunday and GPU specs queued behind it.

**B3 (housekeeping).**
Two of 24 iterations were spent solely confirming a background smoke was still
running. If a unit's next step is gated on a detached CPU job whose measured
cost exceeds an iteration window, the journal hand-off should carry the
*expected completion time* (here: ~52 min per `_train_bc`, two of them, so
~07:10 UTC) so the next iteration can pick a different unit instead of waking
up to wait.

---

## FOR THE OWNER

**Nothing new needs your decision this audit.** Three things are already on your
desk and I want to be clear about their relative cost.

**D1 — the 57M trunk in the control path. 10 days open, and it is now the
binding constraint on the whole project.** Tier 5 is 0 of 27 and it is waiting
on the learning core; the learning core is waiting on D1. Every other blocker
on the board is downstream of it. The option menu was corrected on 2026-08-14
(one listed option contradicts your own PLASTIC-ONLY decree) and the builder
has barred that option pending your reconciliation rather than choosing for
you. It is the highest-value hour you could spend on this project.

**D7 — mood conditioning failed its ablation.** T3.07 measured mood→action
classification at 0.225/0.275/0.375 against 0.25 chance: action distributions
across moods are statistically identical, while a reference arm proves the map
is learnable. `style_net`/`posture_net` receive no gradient anywhere in the
repo. Delete, redesign, or accept as cosmetics — a values call, not a
measurement call.

**D8 — BA.02 cannot be measured in the current body.** No actuator has
directional catch authority; the claim's contrast tops out at +0.09 ± 0.07 s
against a 0.20 s gate in every envelope tested. The builder's recommendation is
to park it until a humanoid body exists, and I agree — this is a fact about the
rover, not about balance.

**One observation you did not ask for.** Three builder iterations were lost on
2026-08-13 to a Claude session limit. A retry-on-fallback guard shipped that
evening and has not yet been tested in anger. Credits remain the unmetered
binding resource, as flagged in DECISIONS_NEEDED since 2026-08-13.
