# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-09 (first audit — no prior report to compare against)
**HEAD:** `ea5b236` · ladder 43/123 demonstrated · tree clean

## VERDICT: INTEGRITY RISK

Not because any PASS is believed false — I could not demonstrate a single
unearned claim, and section 2 found **no silent loosening at all**. The risk is
that three Tier-0/Tier-2 *gates* do not gate: T0.09's check is bypassed by an
operator-precedence bug, T0.12's headline property cannot fail under any
implementation, and T2.02's `_check` still returns `False` for VOID, so
re-running it recreates the exact ledger corruption that had to be repaired by
hand this morning. Tier 0 is what every other tier rests on. Separately, the
ledger now asserts `attempt: 1` for specs known to have run four times.

Findings are ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger

**Clean on the mechanical checks.** 46 entries: 43 PASS, 1 FAIL (T2.01), 1 VOID
(T2.02), 1 ERROR (T1.02).

- Every entry has an implementation in `experiments/tests/`. No orphan claims.
- Every `commit` field resolves in git (46/46). No claim points at a lost tree.
- Every PASS ran at least its spec's declared seed count. No under-seeding.
- **37 of 43 PASSes run a control AND gate on it** — verified by AST, checking
  that `_check`'s second parameter is actually referenced in the body, not just
  accepted. This is a good number and it is real.

Six PASSes have no control at all (`_check(m, _c)`, zero `control_metrics`):
T0.01 (imports), T0.08 (ledger round-trip), T0.10 (Kaggle round-trip), T0.12
(GPU accounting), T1.03 (gradient coverage), T1.05 (frozen stays frozen). For
four of them a control is arguably meaningless. T1.03 and T1.05 are Tier-1
learning claims and could carry one (a deliberately-detached parameter that
*must* read as orphaned; an unfrozen sentinel that *must* move). Low severity.

### 1.1 — `Spec.control` is decorative; the audit surface it provides is false

25 of 43 PASSing specs declare `control=None` in the registry while their test
runs and gates a control anyway (ME.5, ME.8, PG.1, PG.3, PG.4, T1.04, T1.06,
T2.10, T2.12 and 16 others). Nothing enforces agreement between the declared
field and the implementation.

The consequence is exactly the one this audit hit: the mechanical question
"does the spec declare a control?" returns 25 false alarms and cannot be used.
The practice is better than the paperwork, which is the good direction to fail
in — but the paperwork is what an auditor reads.

### 1.2 — `attempt: 1` and `history: []` are asserted for specs that ran 2–4 times

`Ledger.record` gained `history[]` + `attempt` in `20fdf24`, correctly diagnosed:
overwrite-by-`spec_id` made SYSTEM.md law 4 unenforceable. But the migration
backfilled **every** entry with `attempt: 1, history: []`, including specs whose
re-runs are on the record:

| spec | attempts visible in git/journal | ledger says |
|---|---|---|
| T2.01 | v1, v2, v3, v4 (commits `79af208`, `90d8b3c`, `10e3aef`) | `attempt: 1` |
| T0.05 | PASS 08-04 → FAIL 08-08 15:20 → PASS 08-08 16:08 | `attempt: 1` |
| T2.02 | ERROR `2026-08-09T00:14:49` → VOID `07:30:25` | `attempt: 1` |
| T1.02 | ERROR at least twice (08-07, 08-08 22:07) | `attempt: 1` |

`attempt: 1` on a fourth attempt is not "unknown", it is **wrong**, and it is
machine-readable. This is `docs/LESSONS.md`'s own "A default of zero is not
'unknown'" lesson recurring in a new file: a sentinel that is also a valid value
cannot be detected. `None`/`"unknown"` for un-reconstructed entries would have
been honest. The commit message states the agent *did* reconstruct 132 runs over
45 specs from git — that reconstruction is not in the ledger.

### 1.3 — T2.02's status was hand-edited, in a file whose header forbids it

`experiments/ledger.json` line 2: *"Written by experiments/run.py under an
exclusive lock. Do not hand-edit."* Commit `20fdf24` changed T2.02
`status: FAIL → VOID` and rewrote its `message` without a run.

The edit is **correct on the merits** and was disclosed in the commit message —
T2.02's own metrics read VOID, and FAIL would have fired its `kills` field
against the transformer policy. I am not asking for it to be reverted. But the
ledger now contains one status not produced by a runner, distinguishable from a
real one only by reading English prose in the `message` field, and there is no
`restated_by` / `restated_at` field to make it machine-visible. If restatement
is legitimate, it needs a recorded shape.

---

## 2. Thresholds and controls, over time — **NO FINDINGS**

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/` covers 30 commits.

- **No numeric threshold moved in the loosening direction.** Not one.
- **No control deleted or weakened.** The only control edits move the other way:
  `4796e7c` promoted T6.03's byteflip arm from info-only to gated (`+ and
  c["byteflip_load_raised"] == 1.0`); `aeedaf9` and `fff924e` raised PG.5/PG.3
  from `seeds=1` to `seeds=3` *and* added explicit controls before the run.
- **No seed count reduced.** Every seed change is upward.
- **No assertion removed** except in two justified rewrites, both documented with
  measurements in `LESSONS.md`: `4df2c8c` (T2.00 replaced the vf/pg loss-ratio
  gate with per-term trunk gradient norms — the ratio was a structural artifact)
  and `b481786` (T0.05's kill timing rebuilt after its control went vacuous).
- Four `_check` functions contain `or`. Three are legitimate: T1.09's `absurd_oom
  or absurd_peak_gb > MAX_GB` is a control-detects-excess disjunction, and
  T2.02's two are VOID guards that make the test *harder*. The fourth is a bug —
  see 2.1.

The T2.20 commit (`3921991`) deserves explicit credit: the builder found its
shuffled control reading 0.39 in a seed-0 pilot, raised repetitions 3 → 8 per
target per arm **before** the recorded run, and said so in the message without
touching a gate. That is the behaviour this section exists to protect.

### 2.1 — `t0_09_colab_roundtrip.py` — precedence bug disables three assertions

```python
return (m["ok"] and m["cuda_available"] and m["matmul_finite"]
        and "NVIDIA" in m["gpu"].upper() or "TESLA" in m["gpu"].upper()) \
    and m["artifact_bytes"] > 0 and not c["ok"]
```

`and` binds tighter than `or`, so the parenthesised group evaluates as
`(ok and cuda and matmul and "NVIDIA" in gpu) or ("TESLA" in gpu)`.

**Colab's GPU string is literally `"Tesla T4"`, so the right-hand branch is true
on every real run and `ok`, `cuda_available` and `matmul_finite` are never
consulted.** Verified by evaluation:

```
m = {ok: False, cuda_available: False, matmul_finite: False,
     gpu: "Tesla T4", artifact_bytes: 124};  c = {ok: False}
_check(m, c) -> True
```

A Colab job that reported failure, had no CUDA, and returned non-finite matmul
results would still pass T0.09 provided it emitted a non-empty artifact and the
control failed. This is the certification spec for the backend that ran T1.07,
T1.12 and every T1.02 attempt.

The *recorded* PASS is substantively sound — the 2026-08-04 run has `ok: true`,
`cuda_available: true`, `matmul_finite: true` — so no claim in the ledger is
currently false because of this. The guard is simply off, and it is the exact
guard `LESSONS.md`'s "Silence is not success" lesson was written about ("A Colab
job 'succeeded' while its download silently failed").

### 2.2 — T0.12's `weeks_isolated` cannot fail, and it missed the real bug

`t0_12_gpu_budget.py` claims four independently-checkable properties. The fourth:

```python
b.charge("kaggle", (KAGGLE_WEEKLY_HOURS - 2.0) * 3600)   # used is now 30.0
exhausted = b.remaining("kaggle") == 0.0 and not b.afford("kaggle", 0.1)
other = time.strftime("%G-W%V", time.localtime(time.time() - 14 * 86400))
b.data["weeks"][other] = {"kaggle": 29.0}
weeks_isolated = b.remaining("kaggle") == 0.0
```

`remaining()` is `max(0.0, 30.0 - used_hours(current_week))`. The quota has
already been drained to exactly 30.0 two lines earlier, so `remaining()` is 0.0
**whatever the injection does** — even a total isolation failure that summed all
weeks would give `max(0, 30 - 59) = 0`. The assertion is true under every
possible implementation of the mechanism it claims to test.

Worse, `other` is built with `%G-W%V` — the **retired ISO key format**.
`Budget._week()` has returned `%Y-W%U` since `96aa771`. So the test writes into a
key space the live code no longer produces.

The bug this property exists to catch **actually happened**: on 2026-08-08 the
`%G-W%V → %Y-W%U` migration left a colliding `2026-W32` entry that would have
reported 37.5 of 30 Kaggle hours used for the entire fresh quota week and refused
the T2.02 job the whole plan depended on. It was found by a human reading the
budget file. T0.12 was green throughout and is green now.

Secondary: `_check` is `all(m.values())`, which passes on any truthy value. It
works today because every metric is a bool, but it will not fail loudly if one
ever returns a non-empty string.

### 2.3 — T2.02 still returns `False` for VOID; re-running it recreates the corruption

`20fdf24` added `Status.VOID` and taught `run_spec` to accept a `Status` return
from `check`. `t2_02_mlp_showdown.py` was **not** updated — all three of its VOID
paths still `return False`, and `run_spec` maps a falsy bool to
`Status.FAIL, "pre-registered threshold not met"`.

So the fix landed in the framework but not in the one test that motivated it. The
moment T2.02 is re-run — which the D1 decision will require — it records FAIL
again, its `kills` field ("the transformer policy") reads as fired again, and
someone has to hand-edit the ledger again. The lesson was written; the guard was
not closed.

---

## 3. Drift from the goal

**No drift.** Every unit of work in the last day traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| T2.20 — episodic memory speeds the next episode | "He remembers the ladder." |
| T0.05 re-PASS, surgical mid-write kill | "Really learning, not appearing to learn" — the harness that makes the rest believable |
| T1.02 kernel fix (3 seeds, one submission) | same; plus the compute constraint |
| T2.02 launch + postmortem | "One brain" — whether the 57M trunk can be the control path |
| `a3129b2` — 10 unified-brain specs (UB.9–UB.16, PG.6, PG.7) | "All senses, one brain, trained together" |
| `20fdf24` — Status.VOID, ledger history, TIE-by-cost | SYSTEM.md: the machine that cannot fool itself |
| `0c1ff06`/`ea5b236` — ME.11 bakeoff + ME.11.0 PASS | "He remembers you" — retrieval that survives a paraphrase |

The ME.11.0 result is the single best piece of work in the window and it is worth
naming: the builder built a benchmark, ran the **shipped** retriever against it,
recorded **0.000**, and committed that as a PASS. A system that pre-registers a
benchmark its own incumbent scores zero on is working correctly.

### 3.1 — The converse: what has *no passing spec at all*

| GOAL.md claim | specs | PASS |
|---|---|---|
| Memory ("memory makes it him") | 36 ME + T2.10 + T2.20 + T6.03 | **18 ME + 3** |
| Harness / learning primitives | 25 (T0, T1) | **24** |
| Playground physics | 14 PG | **10** |
| **Curiosity** ("he explores because he wants to") | 14 CU + T2.08, T2.09, T3.06, T5.06 | **0** |
| **All senses in unison** | 32 UB + 5 T4 | **0** |
| **Earn your parameters** (ablation) | 12 T3 | **0** |
| **The claims** (Tier 5) | 11 | **0** |
| **Learning by living** (Tier 6) | 6 | **1** (T6.03, persistence) |
| **Locomotion** | T2.01, T2.02 | **0** — FAIL and VOID |

43 of 123, and the distribution is lopsided in a way the headline number hides.
**Memory and harness account for 45 of the 43 PASSes' subject matter; curiosity,
fusion, ablation and the thesis tier account for zero.** PG.4 ("noisy-TV panel
traps naive curiosity") is the closest thing to a curiosity PASS and it is
honestly scoped: it proves the *environment* can trap a naive curiosity signal,
not that Jack is curious.

This is not drift — it is the correct order (the memory branch was unblocked and
cheap; curiosity and fusion need GPU and a settled trunk). But it should be said
plainly: **the three GOAL.md claims most central to the project's identity are at
0/63 specs, and 34 specs are currently blocked behind T2.01's FAIL, T2.02's VOID,
or an unimplemented parent.**

### 3.2 — The ladder-and-apple standard, measured honestly

GOAL.md's own test is "climbing the ladder on attempt 40 after falling on
attempts 1–39, without anyone telling him to." Current state:

- PG.3 PASS proves the ladder is **climbable in principle** — via a *scripted*
  ascent with *adhesion* hands. The spec title says "in principle"; the ledger
  does not overclaim.
- Nothing claims Jack learns to climb. Nothing claims he tries unprompted.
- Jack cannot walk: T2.01 FAIL (4.06σ against a bar of 5), T2.02 VOID (the 57M
  trunk at 2.46σ, below its own 3σ learning gate, while a 125K MLP clears 7.11σ).

### 3.3 — "the namesake claim is reachable again" — true, but at four removes

`a3129b2`'s claim checks out: UB.1 was dead-ended behind T2.01's FAIL and is now
parented `UB.1 → T4.01 → T3.01 → T2.03`, and T2.03 is NOT_RUN-but-runnable rather
than FAILed. The re-parenting is real and the lesson was correctly drawn.

Worth stating for calibration: UB.1 is still **four unimplemented GPU specs**
away, and all four would train a trunk whose architecture is the open D1
question. The PG.6 → UB.9 → UB.10 branch is the genuinely near-term one.

---

## 4. Is the builder alive and productive?

Window 2026-08-08 12:21 → 2026-08-09 12:21 (`/data/jack-logs/ladder.log`):

- **23 iteration starts, 1 ABORT** (11:07, load 8.37 > 6.0 — correct behaviour,
  the box has paying tenants).
- **14 ended `rc=0`, 9 ended `rc=1`.** All nine `rc=1` are the identical line:
  `You're out of usage credits.`
- **PASS delta: 41 → 43 (+2).** T2.20 at 13:18, ME.11.0 at 12:21. T0.05's
  FAIL→PASS churn accounts for the 42→41→42 wobble at 15:20/16:08.

### 4.1 — Ten consecutive hourly slots burned on credit exhaustion, now fixed

02:07 through 11:07 produced nothing: nine 3-second exits plus one load abort.
That is ~10 hours of the day. `04f528c` ("Loop: detect credit exhaustion, fall
back, and stop burning cron slots") added a fallback chain
(`FALLBACK_MODELS="opus sonnet"`) and a pause file, and it **demonstrably worked**
at 12:07 — the log shows `OUT OF CREDITS on fable — falling back to opus`
followed by a productive iteration. Good fix, verified by its product rather than
its absence of error. No open action.

### 4.2 — T2.02's result sat unprocessed for 5 hours

The detached poller wrote T2.02's outcome to the ledger at 07:30:25Z. Every
iteration from 02:07 to 11:07 was dead. The 6.28-GPU-hour result — the one that
settles D1 — was picked up and journalled at ~12:08, by which point a human had
already written D1 into `DECISIONS_NEEDED.md` manually. The loop did not lose the
result (the poller and ledger did their jobs), but it could not act on it.

### 4.3 — Note on authorship

Five commits landed 10:41–11:19 today (`04f528c`, `7addc20`, `92931a6`,
`a3129b2`, `20fdf24`) while the loop was out of credits and mid-ABORT. They are
manual-session work, not builder-loop output. Worth knowing when reading "the
builder shipped 10 specs today": the loop itself shipped ME.11.0.

---

## 5. Compute honesty

`experiments/gpu_budget.json`:

```
2026-W31:  colab 7.7461   kaggle 37.4554
2026-W32:  colab 0.0015   kaggle  6.3849
```

**Kaggle W32 (started today, Sunday 2026-08-09): 6.38 of 30 h spent, ~23.6 h
remaining until the 2026-08-16 reset.** Spend is T2.02's 6.28 h kernel plus the
~0.1 h first attempt that died at 361 s on the torch pin. The 6.38 h is fully
attributable to a ledger entry, and that entry is a VOID — no PASS, but not
waste: a VOID that refuses to arbitrate is the correct product of that run, and
it is exactly the evidence D1 needs.

### 5.1 — ~32 of W31's 37.46 Kaggle hours have no ledger entry

Summing ledger entries whose metrics declare `backend: kaggle` and which ran in
W31: T1.08 (0.36 h), T1.09 (0.06 h), T1.10 (0.06 h), T2.01 (v4's recorded
duration). Even generously including the long GPU-era CPU-stamped entries
(T1.01 0.60 h, T1.06 1.08 h, T1.12 0.32 h), **under 5 hours of the 37.46 charged
Kaggle hours are attributable to a surviving ledger entry.**

The cause is not fraud and is largely on the record already: `Budget.charge()`
bills `res.duration_s` for *every* job including failures and abandoned kernels
(`gpu.py:404`), and the runs that spent the hours — T2.01 v1/v2/v3, the
triple-submitted 5.5 h kernel documented in SYSTEM.md, T1.02's repeated ERRORs —
were **overwritten by `spec_id`** and no longer exist. The `history[]` fix
prevents recurrence from here on; it does not recover W31.

### 5.2 — W31 exceeded the declared 30 h Kaggle cap by 7.46 h

37.4554 against `KAGGLE_WEEKLY_HOURS = 30.0`. `afford()` gates *before* a job on
an *estimate* and `charge()` bills the *actual* duration afterwards, so overshoot
is structurally possible and the tracker correctly refused work once exhausted
(T1.02's ERROR: `kaggle: 0.0h left, need 0.7h`). Two readings, and I cannot
distinguish them from here: either ~7.5 h of real quota was spent past the cap,
or the ISO→`%U` migration double-filed hours into W31. Both matter — the first is
a real overrun, the second means the tracker's numbers are not trustworthy inputs
to the decision it exists to inform. T0.12 (§2.2) is the spec that should have
been able to tell them apart.

---

## 6. Stuck decisions

### 6.1 — D1 has complete evidence and is genuinely the owner's call. Correct.

`DECISIONS_NEEDED.md` D1 ("Does the 57M trunk stay in the control path?") now
carries three independent runs at matched env-steps:

```
T2.01 v4   57M trunk       261 return   4.06σ (bar 5)   curve PLATEAUED
MLP probe  54,179 params   531 return   ~6.5σ           still climbing
T2.02      124,707 params  530 return   7.11σ
           57M trunk       318 return   2.46σ  <- below its own 3σ learning gate
```

This is an architecture call and SYSTEM.md reserves those for the owner. The
escalation is correct, the evidence is complete, and it is not stale — it was
written today. **It could not have been resolved by a bakeoff**: T2.02 *was* the
bakeoff, and it returned VOID by design, because two arms cannot arbitrate when
one has not learned. The system did the right thing and correctly stopped.

Cost of delay is real: 34 specs sit behind T2.01/T2.02, including all of Tier 3,
Tier 4 and Tier 5.

### 6.2 — An owner-decision *was* quietly acted on without being recorded

The first block in `DECISIONS_NEEDED.md` — "Kaggle GPU is not being granted —
needs your account action" — is stale and now actively misleading:

- It states **"Blocks: ladder specs T0.10 and T0.11."** Both are **PASS** in the
  ledger (2026-08-04T15:50 and 15:53).
- It recommends **"option 3 for now"** (skip Kaggle, use Colab). The system did
  **option 1** instead — install a torch build with sm_60 kernels inside each
  kernel run. `114e8f7` is the postmortem of exactly that: *"job's own sb3
  install clobbered the P100 torch pin; PIP_CONSTRAINT now holds torch==2.5.1
  for all later installs."*
- Kaggle is not merely working, it ran a **6.28-hour P100 job today** and is the
  project's primary GPU backend.

So an option the owner was asked to choose between was implemented, debugged and
put into production, and the decision record still asks the question. Anyone
reading `DECISIONS_NEEDED.md` top-down learns something false about the system's
compute before reaching D1. Appended to `DECISIONS_NEEDED.md` for the owner to
close.

---

## 7. Bakeoff hygiene

### 7.1 — `DECISIONS_RESOLVED.md` contains only the bakeoff's own unit-test output

All six entries are for spec id `TEST`, with synthetic arms named `mid`, `mid2`,
`good`, `low`, `weak` and an invented null of `107.613 ± 3.636`. **Zero real
decisions have ever been resolved by bakeoff.**

Cause: `bakeoff.py:48` hard-codes
`DECISIONS = Path(__file__).parent.parent / "docs" / "DECISIONS_RESOLVED.md"`
and `:229` appends to it, with no path override and no test mode. So exercising
the primitive writes into the project's permanent decision record — the same
class of error as a test writing to the real ledger, which `protocol.py` is
careful to prevent.

On the audit questions themselves, judged against the six fixture entries: the
learning gate fires correctly (the `weak` arm at −1.20σ produces VOID, not a
verdict); VOID is never presented as a winner; the two TIEs are honestly labelled
as within-noise (0.38σ against a 1.5 margin) and resolve by declared cost; and
the undeclared-cost case correctly VOIDs rather than handing victory to a
`None`. **The primitive's logic is sound.** It is the output plumbing that is
wrong — and the pollution means that when the first real bakeoff lands, its
result will appear in a file that a reader has already learned to distrust.

---

## 8. The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday, or
only closer to a longer list of green ticks?**

Genuinely closer on two axes, and honestly further away on the one that matters
most.

**Closer, and it counts.** T2.20 is the day's real capability result: episodic
memory cut embodied search time to 0.046 of a memoryless null in the actual
playground, with two controls restored to null level — the speedup lives in the
store's *content*, not the policy. That is "he remembers the ladder," measured in
a body, in the world, against something that could have beaten it. ME.11.0 is the
day's real *system* result: a benchmark pre-registered before any arm is scored,
on which the shipped retriever scores 0.000. Both are the opposite of a green
tick.

**Closer on the machine, which SYSTEM.md says counts double.** `Status.VOID`
exists, `Arm.cost` no longer defaults to a lie, the loop survives credit
exhaustion, and the overseer you are reading exists. Four classes of bug made
unrepeatable rather than fixed.

**Further away, and this is the finding.** Jack cannot walk. T2.01 FAILed and
T2.02 refused to arbitrate, so the 57M trunk that the whole "one brain"
architecture rests on has now failed a 3σ learning gate that a 125K MLP clears at
7.11σ, in three independent runs. Every downstream claim — ablation, fusion,
continual learning, unprompted exploration, the ladder itself — is parked behind
that. **Curiosity: 0 of 18 specs. All senses in unison: 0 of 37. Earn your
parameters: 0 of 12. The thesis tier: 0 of 11.** The 43/123 headline is carried
by memory and harness, and those are the two branches that are cheap on a free
CPU box. That correlation is not an accident and it should be named: *the ladder
has been climbed where it was climbable, not where it was load-bearing.*

The gap between "43 demonstrated" and "a curious humanoid" is not 80 more ticks.
It is one architecture decision (D1), which is with the owner today with complete
evidence, and after that a Tier-2 locomotion result that does not yet exist.

**What would make tomorrow's answer better:** D1 decided, and the first spec in
the CU or UB branch implemented rather than pre-registered. Ten well-researched
unified-brain specs were written today and zero were run. Pre-registration is a
virtue and this project does it better than most published work — but a
registry entry is not evidence, and the distance between 123 specs and 43 results
is now the largest single number in the system.

---

## FOR THE BUILDER

Ranked. Each is a code change; none require the owner.

1. **Fix `t0_09_colab_roundtrip.py:_check` precedence (§2.1).** Wrap the GPU-name
   disjunction: `... and ("NVIDIA" in m["gpu"].upper() or "TESLA" in
   m["gpu"].upper()) and ...`. As written, `ok`, `cuda_available` and
   `matmul_finite` are unreachable on any real Colab run. Re-run T0.09 after the
   fix — the recorded metrics say it will still PASS, so this costs one Colab
   round-trip and buys back a Tier-0 guarantee. Then grep every `_check` in
   `experiments/tests/` for a bare `or` mixed with `and` at the same level; this
   pattern is not detectable by any current gate.

2. **Make `t0_12_gpu_budget.py`'s `weeks_isolated` falsifiable (§2.2).** Assert
   isolation *before* draining the quota, not after: charge 2 h to the current
   week, inject 29 h into a foreign week, then assert `remaining("kaggle") ==
   28.0` — a value that changes if weeks leak. Build the foreign key with the
   live `Budget._week()` logic (`%Y-W%U`), not the retired `%G-W%V`, and add a
   fifth property that a *stale-format* key (`2026-W32` written as ISO) does not
   alter the current week's remaining — that is the bug that actually occurred on
   08-08 and it is still untested. Also replace `_check = all(m.values())` with
   an explicit conjunction of named booleans.

3. **Return `Status.VOID` from `t2_02_mlp_showdown.py:_check` (§2.3).** All three
   VOID paths still `return False`, which `run_spec` maps to FAIL. Re-running
   T2.02 today recreates the exact corruption that required hand-editing the
   ledger. Then add a permanent guard rather than only fixing this test: in
   `run_spec`, if a `_check` writes the substring `"VOID"` into `m["verdict"]`
   but returned a bare `False`, raise — a metrics/status disagreement should be
   impossible to record, not merely caught by an auditor reading prose.

4. **Stop `bakeoff.py` writing to the real decision record from tests (§7.1).**
   Give `record_decision` an output-path parameter defaulting to `DECISIONS`, and
   have the self-tests pass a temp path. Then delete the six `TEST` entries from
   `docs/DECISIONS_RESOLVED.md` — they are unit-test fixtures, not decisions, and
   the file currently contains nothing else.

5. **Reconcile `attempt`/`history` with reality, or admit ignorance (§1.2).**
   `attempt: 1` is a false statement for T2.01, T0.05, T1.02 and T2.02. Either
   backfill from the git reconstruction the agent already performed (132 runs
   over 45 specs, per `20fdf24`'s message), or set un-reconstructed entries to
   `attempt: null` with `history: null`. A wrong integer is worse than a null,
   and this is `LESSONS.md`'s `Arm.cost` lesson recurring in a second file —
   consider whether that lesson should be generalised to *"never backfill a
   migration with a value that is also a valid measurement."*

6. **Decide whether VOID blocks dependents, and make code and docs agree (§1.3
   context).** `Ledger.blocked_by` (`protocol.py:242`) returns any dependency
   whose status `is not Status.PASS`, so VOID blocks exactly like FAIL — T2.13
   and T5.09 are BLOCKED on T2.02 right now. `Status.VOID`'s own docstring says a
   VOID spec "does not BLOCK its dependents on the grounds that the claim was
   refuted." Blocking may well be the right behaviour (an undemonstrated
   foundation is still undemonstrated), but the docstring and the code cannot
   both stand.

7. **Make `Spec.control` load-bearing (§1.1).** Have `run_spec` raise when
   `control_fn` is supplied and `spec.control is None`, or vice versa. Then
   backfill the 25 specs that run a control without declaring one. This is what
   turns "does the spec declare a control?" from a question with 25 false alarms
   into a real audit surface.

8. **Low priority: give T1.03 and T1.05 controls.** T1.03 could carry a parameter
   deliberately detached from the graph that *must* be reported as orphaned;
   T1.05 an unfrozen sentinel that *must* move. Both are cheap and both convert a
   "we observed the good thing" into "we observed the good thing and the
   measurement can see the bad thing."

9. **Consider re-running ME.8 at 3 seeds.** It is a PASS at `seeds=1` whose own
   commit message records that a **seed-2 training collapse** was fixed by a GRU
   retain-bias init. The fix was never verified at the seed that motivated it,
   and GOAL.md asks for ≥3 seeds where the claim is about learning. T1.07 (LR
   robustness, `seeds=1`) is the same shape at lower stakes.

---

## FOR THE OWNER

1. **D1 is on your desk with complete evidence and is blocking 34 specs.**
   `docs/DECISIONS_NEEDED.md` has the full table. Three independent runs at
   matched env-steps say the 57M trunk underperforms a 125K MLP at motor control
   by roughly 2× return and fails its own 3σ learning gate. The loop's
   recommendation is **option A** (freeze the trunk, small dedicated policy head
   does control) — it keeps "one brain, all senses" intact while stopping the
   trunk from being asked to be a motor controller. Everything in Tiers 3, 4 and
   5 waits on this. Saying *"do what the measurements say"* is enough; it will be
   read as A and journalled.

2. **Close the stale Kaggle block at the top of `DECISIONS_NEEDED.md`.** It asks
   you to choose an option that was implemented four days ago. Kaggle works, runs
   the project's longest jobs, and T0.10/T0.11 both PASS. Appended a note there
   with the evidence; it needs one line from you to be struck.

3. **Credit exhaustion cost ~10 hours yesterday** (02:07–11:07, nine dead
   iterations). The loop now falls back `fable → opus → sonnet` and pauses rather
   than burning cron slots, and that fix is verified working. No action needed
   unless you want the primary model changed.

4. **One number worth your attention:** 123 specs, 43 results. Ten new
   unified-brain specs were pre-registered today and none were run. The registry
   is now growing faster than the ledger, and pre-registration is only a virtue
   while the gap closes.

*Audited by `scripts/overseer.sh` — independent of the builder loop, read-only on
all spec, test and model code.*
