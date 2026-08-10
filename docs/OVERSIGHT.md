# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-10 00:45 UTC (3rd audit; previous 2026-08-09 18:37 at `db9fd7b`)
**HEAD:** `b809b6b` · ladder **55/137 demonstrated** · **9 commits unpushed** ·
working tree **dirty: `experiments/ledger.json`, written by a detached script at
00:39 and committed by nobody** (§1.4)

## VERDICT: DRIFTING

**The ledger is in the best condition it has ever been in, and I can now say so
with evidence rather than with mechanical checks.** I re-evaluated all 55 PASS
entries by feeding their *recorded* metrics back through their *committed*
`_check` functions: **55/55 still return True**. I then re-ran the same 50
checks with `control_metrics` emptied: **0/50 survive** — every control in this
ladder is load-bearing, none is decorative. No PASS is unearned, no threshold
moved in the loosening direction this week, and the one seed-count reduction
(T1.08) is justified by a measurement in its own commit. Section 2 is clean for
the third audit running.

The drift is elsewhere, and it is getting worse, not better:

1. **Five of the senses GOAL.md calls constitutional have ZERO specs among
   137** — smell, taste, voice, pain, temperature. Not blocked. Not failing.
   *Absent.* `run blocked` cannot see them, `run next` cannot see them, and this
   audit's own §3 is the only organ in the system that can. `LESSONS.md:783`
   recorded exactly this failure 30 hours ago and prescribed a guard; the guard
   was never built. (**RANK 1 for drift.**)
2. **The most consequential job in the project has been running unattributed and
   unmonitored for 5 hours.** `experiments.run T2.01` — the spec that alone
   blocks 36 others — has been polling a remote GPU since 2026-08-09 19:42. No
   journal entry launched it, no log line records it, three successive
   iterations treated it purely as a lock to route around, and it dies to its own
   `timeout 34000` at **~05:09 UTC** whether or not the kernel finished. If it
   dies there, nothing is written and the hours are gone. (**RANK 2.**)
3. **A detached script wrote the ledger 25 minutes after the iteration that
   spawned it had exited.** PG.6 is now `attempt: 4` in a dirty working tree with
   no commit and no journal line. It happens to be a harmless duplicate re-run —
   but this is precisely the mechanism by which an unattributed claim enters the
   record. (**RANK 3.**)
4. **Zero of the ~44 metered Kaggle GPU-hours this project has spent has
   produced a single PASS.** Every GPU spec attempted since 2026-08-06 is VOID
   (T2.01, T2.02) or ERROR (T1.02).
5. **Three record-integrity items are now unactioned across three consecutive
   audits** — `Spec.control` (20 false negatives), T1.03/T1.05 controls, ME.8 at
   3 seeds. The builder cleared four of the previous ten items, including both
   hard ones; these three are the cheap ones left behind.

Findings ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger

58 entries: **55 PASS, 2 VOID (T2.01, T2.02), 1 ERROR (T1.02)**. Verified
programmatically, all 58.

**Mechanical checks — all clean:**

- Every entry resolves to **exactly one** implementation in `experiments/tests/`.
  No orphan claims, no glob collisions (`ME.1`/`ME.10` and `ME.11.0`/`ME.11.A`
  both resolve uniquely).
- Every `commit` field resolves in git — 58/58, plus **all 28 history commits**.
  No claim points at a lost tree.
- `run stale`: **no stale claims**. Every entry carrying an `impl_sha` names the
  test file as it stands today.
- Every PASS ran at least its declared seed count.

### 1.1 NEW EVIDENCE — the ledger audits itself, and it passes

Two probes nobody has run before. Both cost zero credits and zero compute, and
both are stronger than any structural check the previous audits could make.

**Probe A — re-evaluate every recorded verdict.** For each PASS, import its test
module and call the *committed* `_check(metrics, control_metrics)` on the
*recorded* numbers:

    re-evaluated 55 PASS entries against their committed _check
      agree (True):  55
      DISAGREE:       0
      unevaluable:    0

This is what catches a check loosened *after* the run that it certified, or a
metric that never satisfied its own gate. Nothing here is inconsistent.

**Probe B — is the control load-bearing?** Re-run the same 50 checks (the 5 with
no control are excluded) with `control_metrics` replaced by `{}`:

    PASS specs whose _check STILL PASSES with the control emptied: none

Every control-declaring check either raises or returns False without its
control. There is no PASS in this ledger whose control could be deleted without
the verdict changing. T0.13 asserts this property forward (no gate is inert);
these two probes assert it *backward*, over the record as it stands. **They
should be a command — see FOR THE BUILDER item 1.**

### 1.2 Five PASSes have no control at all — RANK 6, third audit unactioned

`T0.01` (imports), `T0.08` (ledger round-trip), `T0.10` (Kaggle round-trip),
`T1.03` (gradient coverage), `T1.05` (frozen stays frozen). All five declare
`control=None` in the registry *and* define no `_control` in their test file, so
this is honest rather than concealed — **there is no PASS whose control was
declared and never run.**

Two of the five are not bookkeeping. **T1.03** claims "gradient reaches every
trainable parameter" and **T1.05** claims "frozen stays frozen" — both are
existence claims about a mechanism, and neither has ever been shown capable of
reporting the bad case. T1.05's own docstring records the bug it was written for
(`self.apply(self._init_weights)` silently randomising a loaded LLM); nothing
demonstrates the sentinel would have caught it. Flagged at 12:37 and 18:37;
still open.

### 1.3 `Spec.control` is still not load-bearing — RANK 5, third audit unactioned

**20 entries record `control_metrics` while their spec declares `control=None`:**
ME.5, ME.8, PG.1, PG.3, PG.4, T0.03, T0.05, T0.06, T0.07, T0.09, T0.11, T1.04,
T1.06, T1.07, T1.08, T1.09, T1.10, T2.02, T2.10, T2.12.

The science is fine — the control ran, and probe B above proves each one is
read. The *declaration* is the audit surface, and 20 false negatives make "does
this spec declare a control?" useless as a check. It was useless to me: I had to
verify by importing 55 modules instead of reading 55 fields. (Was 19 at the last
audit; it grew by one.)

### 1.4 A detached script wrote the ledger after its iteration exited — RANK 3

`experiments/ledger.json` is **dirty in the working tree right now**. PG.6 went
`attempt: 3 → 4`, `commit 185cb1c → b809b6b`, `ran_at 2026-08-10T00:39:05`.

The writer was `/data/tmp/run_pg6.sh`, created 22:58 by an earlier iteration:

```bash
for i in $(seq 1 90); do
  if flock -n /tmp/jack-ladder.lock true 2>/dev/null; then break; fi
  sleep 60
done
... timeout 60m python -m experiments.run PG.6
```

It waited out the lock for ~90 minutes, fired at ~00:28, and finished at 00:39 —
**25 minutes after the 00:07 iteration ended at 00:14.** Nothing committed the
result; nothing journalled it; `/data/tmp/pg6_run.out` is the only record it
exists.

**Why this matters more than the harmless outcome.** The run was a duplicate:
same `impl_sha` (`f8b7cf05`), same numbers (`bearing_med_deg` 1.27,
`bearing_med_grey == bearing_med_const == 8.200667`), so no science changed and
the entry is legitimate. But the mechanism is the one this repo has already been
burned by twice — "silence is not success" plus an unattributed write. Had it
returned **FAIL**, the ladder would have dropped 55 → 54 in a dirty tree, and the
first thing to notice would have been the next iteration wondering why. SYSTEM.md
requires "leave no process running"; this iteration left a 90-minute one, and it
outlived its author.

### 1.5 48 of 58 entries cannot be checked for staleness — RANK 7, honest

They predate `impl_sha` (landed 2026-08-09 20:25). `run status` reports the
number unasked and never folds it into "clean", which is the right behaviour. It
resolves one entry at a time, on re-run. Not a defect; a known debt.

### 1.6 The `amend` mechanism landed and was used correctly

`Result.amended` + `run amend` (T0.17, `b809b6b`) closed the previous audit's
RANK 1. Five entries carry amendments: T2.01 and T2.02 record the `9b92d14`
hand-edit as an edit, with author, reason, prior value and commit; T0.05, T0.09,
T1.02 record `attempt: 1 → null`. `Ledger.AMENDABLE = (VOID, SKIP, NOT_RUN)`
means an amendment can never reach a status that asserts a capability. I tried to
fault this and could not: the design is right, the backfills are accurate, and
the control (the `9b92d14` edit replayed on a temp ledger, invisible to the
detector) is the correct one.

---

## 2. Thresholds and controls, over time — NO FINDINGS

80 commits touched `registry.py`, `registry_expansion.py` or `experiments/tests/`
in the last 7 days. I read every deleted line carrying a number or an assertion.
**Nothing was loosened silently.** Four changes deserved scrutiny and all four
survive it:

| change | commit | verdict |
|---|---|---|
| `NULL_BEARING_FLOOR = 20°` scoped to the control band only (PG.6) | `6c0fcd1` | **JUSTIFIED, and net tighter.** See below. |
| `moved_frac >= 0.95` → `undeclared_stuck_params == 0` (T1.04) | `bb55c15` | **JUSTIFIED.** Replaces a percentage bar with a pre-declared exclusion list of 10 modules, each with a written reason; any *undeclared* stuck module now fails loudly, where before it could hide under 5% slack. |
| `MAX_VF_PG_RATIO = 50.0` removed (T2.00) | `4df2c8c` | **JUSTIFIED by measurement.** `pg_loss ≈ 0` by construction at an unmoved policy, so the ratio was unthresholdable; replaced by per-term gradient norm on the shared trunk. Already a LESSONS.md entry. |
| T1.08 `seeds=3 → seeds=1` | registry | **JUSTIFIED.** The only seed-count reduction in the window. T1.08 is the spec *about* seed variance; `_experiment` ignores its seed argument and varies [0,1,2] internally, so spec-level `seeds=3` launched three identical GPU jobs for zero information. Documented in the spec's own `notes`. |

**On PG.6 specifically**, because it is the one case where a spec FAILED on a
constant and the same iteration changed that constant and passed. I checked it
adversarially and it holds:

- The constant was an implementation choice from attempt 1, not a registry
  threshold. The registry pre-registers R² ≥ 0.80 and bearing ≤ 5° and *names*
  the two nulls without numbering them. Both registered gates are untouched.
- The 20° floor is arithmetically unreachable in a ±22° band: a constant
  predictor scores the band's median |bearing|, measured 8.87/8.91/8.78° over
  3000 draws per seed. To exceed 20° a null would have to be anti-correlated with
  truth. The nulls read 8.96 and 8.20 — they were *working*.
- It is still enforced unchanged on the 40–75° control band (measured 58.0°).
- The replacement is tighter in two directions and I verified both are live in
  `_check` (via `seed_gates_ok`, `pg_6_playground_eyes.py:524-543`):
  the probe must beat the constant predictor 2× (`PROBE_BEARING_FRAC = 0.50`,
  measured 1.27 vs 8.20), and the grey null must *equal* it to 0.05°
  (`GREY_MATCHES_CONST_DEG`, landed at exactly 8.200667 == 8.200667).

The same iteration also *declined* a change that would have raised every number
in the file — the strict 3-ray occlusion rule rejects 49.6% of the band against
30.8% for the centre ray, and it took the centre ray with the reasoning written
down: *"choosing the dataset that scores best against a fixed threshold is the
same sin as moving the threshold."* That is the behaviour this section exists to
find the absence of. It is present.

**No control was deleted, no `_check` gained a disjunction, no assertion was
removed.** Two controls were *strengthened*: T6.03's byteflip arm was promoted
from information-only to a gated control (`4796e7c`), and PG.8 gained an
in-contact observation comparison after 78 dead `cfrc_ext` columns were found
(`910f3d6`) — the 1e-9 contact-free gate was kept and a second one added at 0.05,
four orders of magnitude below the broken path's 135.9.

---

## 3. Drift from the goal

### 3.1 What the last 24 hours bought, and what it serves

`git log --since=2026-08-09T00:00` shows 100 commits — most from an interactive
owner session between 16:00 and 21:00, the rest from 13 productive loop
iterations. Every substantial item traces to GOAL.md:

| work | GOAL.md sentence it serves |
|---|---|
| PG.6 PASS — the eye (R² 0.9747, bearing 1.27°) | "EVERY SENSE A HUMAN HAS" — sight |
| PG.7 PASS — heard-not-seen fixture, 0.000 bits position leak | "what he hears can teach what he sees" — the UB.9 substrate |
| PG.8 — 78 dead observation columns found and fixed | "give him a body… a world" |
| LC.00/01/02 PASS — core decidable, unison-admissible, fast enough to live | "ONE model", and SYSTEM.md's "no learning core without unison" |
| T0.12/T0.15/T0.16/T0.17 PASS | "Really learning, not appearing to learn" |
| `run blocked`, `run stale`, `impl_sha`, `amend` | "protects the honesty of watching what happens" |
| `drives.py` / PS.01 attempt | "Jack has the needs of a human" |
| GOAL.md/CHAMPIONS/queue rewrites | owner directives, correctly recorded with counterarguments |

**I found no drift in what was built.** Not one commit in 24 hours fails to trace
to a GOAL.md sentence. PS.01 in particular was *stood down* rather than forced —
the builder measured that its impulse formula could not distinguish a fall from
resting ground contact and stopped, which is the right call and the expensive one.

### 3.2 The converse question, and it is bad — RANK 1 for drift

**Which parts of GOAL.md have no passing spec?**

GOAL.md, in the owner's own words on 2026-08-09, names the sensory inventory:

> sight · hearing · touch · proprioception & balance · **SMELL** · **TASTE**
> **pain** · **temperature** · interoception (hunger, thirst, fatigue)
> and **VOICE** — he must be able to make sound, not only receive it

Grepping all 137 specs for `smell|olfact|taste|gustat|voice|vocal|pain|thermo|
temperature|interocept|hunger|thirst|fatigue` returns **exactly one hit, and it
is the word "voiced" describing a struck geom in PG.5's audio spec.**

    smell        0 specs registered    0 PASS
    taste        0 specs registered    0 PASS
    voice        0 specs registered    0 PASS
    pain         0 specs registered    0 PASS
    temperature  0 specs registered    0 PASS
    interoception  1 spec (PS.01)      0 PASS — NOT_RUN, last attempt stood down

`docs/CHAMPIONS.md:78-80` is honest about this — three seats read **"VACANT —
sense not yet built"**, and voice reads *"needs a spec"* — but nobody wrote the
spec. GOAL.md's justification for these is not decorative: olfaction is "the
sense that works when sight fails", and gustation drives "one-trial learning with
long delay tolerance, the fastest learning in biology and a capability nothing
else in his design has."

**The structural point is worse than the count.** A spec that is registered and
blocked is visible to `run blocked`, to `run next`, to `run status` and to the
Review. A capability that was never registered is invisible to *every organ this
system has*, and reads as completeness in all of them. `LESSONS.md:783` recorded
this exact failure on 2026-08-09 — *"AMBITION blindness: structurally invisible,
because the map is the thing with the hole"* — and prescribed the guard:
*"at least one recurring audit must measure against a reference from OUTSIDE the
project's own documents."* **No such organ was built.** Thirty hours later the
hole is the same size, and the lesson that named it is now itself evidence for
the rule that a lesson prescribing a guard is not a guard.

### 3.3 The three headline claims, counted honestly

| GOAL.md claim | registered | PASS | reachable today |
|---|---|---|---|
| Curiosity (CU.1–CU.7, T2.08/T2.09) | 9 | **0** | 0 — all behind `T2.01=VOID` |
| All senses in unison (UB.1–UB.16, T4.01–T4.05) | 21 | **0** | UB.9 only |
| Learning by living (T6.01–T6.05, T5.*) | 14 | **1** (T6.03 persistence) | 0 |

PG.4 is the nearest thing to a curiosity result and it is a *negative* one: it
certifies that a noisy-TV panel traps naive ICM (dwell 0.667 vs 0.061 random
null). That is a property of the world and a warning about the mechanism — not a
demonstration that Jack is curious.

### 3.4 One thing that improved

The registry-outgrowing-evidence trend from the last audit **reversed**. At 18:37:
136 registered / 51 PASS, gap 85. Now: 137 / 55, gap **82**. In the last 6 hours
the builder registered 1 spec and demonstrated 4. Item 10 of the previous
FOR THE BUILDER was taken.

---

## 4. Is the builder alive and productive?

**Alive and, in its second half, excellent.** 24 hours to 00:14:

| | |
|---|---|
| iterations started | **25** (hourly, 00:07 → 00:07) |
| ended `rc=0` | 14 |
| ended `rc=1` | 9 — **all nine were ~3-second credit-exhaustion deaths**, 02:07–10:07 |
| no `iteration end` line at all | **2** (17:07, 22:07) |
| hour with no iteration at all | **1** (11:07) |
| PASS delta | **42 → 55 (+13)** |

All 13 PASSes came from the 13 hours after 12:07. The nine dead hours are the
already-diagnosed credit exhaustion, and the fallback chain that fixed it works —
it engaged 12 times and every engagement produced work.

### 4.1 The cron still names a model that is always out of credits — NEW

```
7 * * * * JACK_LOOP_MODEL=fable /home/opc/jackthelearner/scripts/ladder_loop.sh
```

**Every single iteration since 12:07 on 2026-08-09 — 12 of 13 consecutive runs —
has begun with `OUT OF CREDITS on fable — falling back to opus`.** The fallback
is doing its job, but it is being asked to do it every hour, forever, on a model
that has not had credits in 13 hours. One iteration (22:07) fell through opus to
sonnet as well. The cost is small (a wasted invocation and ~3 s per hour) and the
risk is not: the chain has one rung left, and the loop's model tier is being
decided by exhaustion rather than by choice. **One-line fix:
`JACK_LOOP_MODEL=opus` in cron.**

### 4.2 The iteration-end trap was not built — item 9, unactioned

52 `iteration start` / **50** `iteration end` across the whole log. `ladder_loop.sh`
writes its end line as the last statement with no `trap ... EXIT`, so an iteration
killed outside `timeout 50m` records nothing. Both missing ends (17:07, 22:07)
*did* land work — which is worse, not better: the log cannot distinguish "killed
having done nothing" from "killed having committed a PASS."

### 4.3 Detached processes are outliving their iterations — RANK 2 and RANK 3

Two right now, both invisible to `ladder.log`:

- **`experiments.run T2.01`, pid 1126493, started 2026-08-09 19:42, up 5h03m.**
  1.75 seconds of CPU consumed in five hours — it is a remote-GPU poller. Under
  `timeout 34000`, so it is **SIGKILLed at ~05:09 UTC** regardless of the kernel's
  state; if that fires first, `run_spec` never records and the hours buy nothing.
  Nothing launched it from the loop (the 19:07 iteration ended at 19:19), nothing
  journalled it, and the three iterations that met it (PG.7, PG.8, PG.6) each
  treated it purely as a lock to route around. `docs/LOOP_JOURNAL.md:931` still
  says the T2.01 re-run "is now blocked on a push" — while it has in fact been
  running for five hours.
- **`run_pg6.sh`**, the 90-minute lock-waiter of §1.4.

The lock-split and the idle-holder overflow slot (`8970638`, `6c0fcd1`) were the
right engineering and they worked — PG.6 ran on the overflow slot with the
holder's pid, core count and age printed. The gap is ownership, not locking:
**no organ knows these processes exist, and neither will report if they die.**

---

## 5. Compute honesty

```json
"2026-W31": { "colab": 7.7461,  "kaggle": 37.4554 }   // Aug 2–8  (%U weeks)
"2026-W32": { "colab": 0.0015,  "kaggle":  6.3849 }   // Aug 9–15, current
```

**Remaining this week: 23.6 of 30.0 Kaggle hours, expiring end of 2026-08-15.**
Unspent free quota is not saved. Week 31 closed **7.4554 h over** the ceiling;
the meter observes the ceiling, it does not enforce it — the `overruns` list and
the stderr shout added in `496e951` are the correct response and are now in place.

### 5.1 ~44 Kaggle GPU-hours have produced zero PASSes — RANK 4

Every ledger entry ever produced on a GPU:

| spec | when | outcome |
|---|---|---|
| T0.09, T0.10, T0.11 | 2026-08-04 | PASS ×3 — `gpu<20min` round-trips |
| T1.09, T1.10 | 2026-08-06 | PASS ×2 — `gpu<20min` |
| T2.01 | 2026-08-07 | **VOID** (killed by T0.14's dropout finding) |
| T1.02 | 2026-08-08 | **ERROR** (`kaggle: 0.0h left, need 0.7h` — denied by the broken meter) |
| T2.02 | 2026-08-09, 6.28 h P100 | **VOID** (two non-learners cannot arbitrate) |

Five short round-trip PASSes on Aug 4–6, then **nothing.** The 43.84 metered
Kaggle hours and 7.75 Colab hours have bought two VOIDs and one ERROR. That is
not fraud and it is not waste in the ordinary sense — T2.01/T2.02 were invalidated
by a *real* bug (dropout live through rollout, update and eval; 42% action noise
on one arm and none on the other), and finding that bug was worth the hours. But
the honest headline is that **no capability of Jack's has ever been demonstrated
on a GPU**, and the record should say so plainly rather than average it away.

### 5.2 The cause is not the meter any more; it is D3

The meter's three defects are fixed and gated (T0.12 at 24 properties, up from
12, with `_PreFixBudget` reproducing the pre-fix billing loop verbatim as a
control). What now stops GPU work is `assert_ref_is_current`: **9 commits are
unpushed**, so no GPU job can be built until someone pushes. That is D3, open
since 2026-08-08, and the last three iterations have each escalated it and
stopped. Explicitly **not** claimed by anyone: nothing has ever reconciled our
meter against Kaggle's own reported kernel runtime — the builder stated that
limitation itself rather than letting T0.12's green tick imply otherwise.

---

## 6. Stuck decisions

### 6.1 D1's recommended option is now forbidden by the owner's own decree — NEW

`docs/DECISIONS_NEEDED.md:73` still asks the owner to choose, with **A**
RECOMMENDED:

> A. Freeze the trunk; small dedicated policy head does control. **RECOMMENDED.**

and line 241, in the 13:45 correction, reaffirms: *"Option A (freeze + small
head) is still the recommendation."*

At **21:16 the same day** the owner decreed (`eea7195`, GOAL.md:76,
CHAMPIONS.md:83): **PLASTIC ONLY — NO FROZEN COMPONENTS IN JACK.** Scope, stated
precisely in CHAMPIONS.md: *"this governs components INSIDE Jack — his encoders,
his core, his fusion. It does NOT touch the parent LLM."* A frozen 57M trunk with
a small head is a frozen component inside Jack. **Option A is unconstitutional
under the decree that postdates it.**

`814ed89` ("Propagate the plastic-only decree everywhere it changes meaning")
swept DECISIONS.md, `ladder_prompt.md` and 9 registry specs — the *answered*
record. It did not touch the *open questions*: `DECISIONS_NEEDED.md` still offers
A as the recommendation, and `CHAMPIONS.md:64` still lists "frozen-trunk+head" as
a live challenger for the vacant D1 seat. **The owner is currently being asked to
pick between four options, one of which he has already ruled out.** Appended to
`DECISIONS_NEEDED.md` with evidence; the reconciliation is his, not mine.

### 6.2 Nothing else is decidable that is not already escalated

- **D3 (may the loop push?)** — the whole bottleneck, one line, unchanged since
  the last audit except that the cost is now dated: 23.6 h expire 2026-08-15.
- **D2 (does VOID block?)** — cost correction landed (40, now **59** blocked); the
  choice itself is genuinely a judgement about what the ladder means, not a
  measurable question. Correctly parked.
- **D1** — correctly *not* decidable: the evidence is confounded (dropout live)
  and needs the re-run. §6.1 is about the option set, not the verdict.

### 6.3 Was an owner-decision acted on without being recorded? — one, still open

The **Kaggle accelerator block** at the top of `DECISIONS_NEEDED.md` still asks
the owner to choose between three options while the system implemented option 1
five days ago (`114e8f7`, in-kernel torch pin with sm_60 kernels), and it still
states that it *"Blocks: T0.10 and T0.11"* — both PASS since 2026-08-04. Flagged
at the last audit; unstruck. It is the first thing anyone reads in that file, and
it is false in three ways. Only the owner may strike it.

Conversely, the **care-verbs** decision was made by the owner on 2026-08-09
(*"Can you also drop stuff in for him… Yes"*) and is correctly marked **DECIDED**
in place, with the anti-puppeteering constraint recorded beside it. Good practice.

---

## 7. Bakeoff hygiene — no violations, because no bakeoff has ever run

`docs/DECISIONS_RESOLVED.md` contains **zero decisions**. The nine `TEST` entries
that were its entire content were unit-test fixtures leaking through a
module-constant path; they are gone and `run_bakeoff(decisions_path=...)` makes
it impossible to repeat.

So: no decision made without a learning gate, no VOID treated as a verdict, no
winner chosen inside the noise margin. **Nothing to find — and that is the
finding.** SYSTEM.md's third law ("decisions are made by bakeoff, never by
argument") has never once been exercised on a real question, while
`docs/CHAMPIONS.md` shows six of Jack's seats filled anyway:

| seat | held by |
|---|---|
| Learning core | PPO — **DEFAULT, never defended** |
| Vision encoder | from-scratch 0.24M — **DEFAULT, never defended** |
| Needs/reward form | drive-reduction — **BY ANALYSIS** |
| Curiosity signal | learning-progress — **BY ANALYSIS** |
| Consolidation | SIESTA — **BY ANALYSIS** |
| Language model / acquisition | LLM-as-parent — **BY DECREE** |

**CHAMPIONS.md labels every one of these honestly**, records the strongest
counterargument beside the decree, and pre-registers a re-open trigger for the
plastic-only call. That is exactly what SYSTEM.md asks for and it is the reason
this is a §7 note rather than a §7 finding. But the gap between "the law" and
"the practice" is now six seats wide, and the cheapest way to close one is
already registered and unrun: **ME.11.B–F**, the memory-retrieval bakeoff, CPU,
all dependencies PASS. It would be the first real entry this file has ever had.

---

## 8. The honest summary — are we closer to a curious humanoid?

**Closer, genuinely — but along one axis only, and it is not the axis the goal
is measured on.**

What the 55 PASSes actually certify:

| | count | what it is |
|---|---|---|
| T0.* | 17 | the measuring apparatus |
| T1.* | 12 | the model's plumbing can receive a gradient |
| PG.* | 8 | the world exists and obeys its own rules |
| ME.* | 10 | memory: diary, retrieval, forgetting, attribution |
| T2.* | 4 | four components beat a null |
| LC.* | 3 | candidate learning cores are admissible |
| T6.03 | 1 | he survives a restart |

**29 of 55 — more than half — are T0 and T1: the ruler and the wiring, not Jack.**
Tiers 3, 4 and 5 read 0/24. Curiosity reads 0/9. Unison reads 0/21.

Against the ladder-and-apple standard GOAL.md sets: PG.3 proves the ladder is
climbable — by a *scripted* adhesion controller, 0.973 of rung spacing. Nothing
has ever climbed it by wanting to. T2.01 says Jack cannot yet be shown to walk
better than a random policy, and that entry is VOID, which means we do not even
know that he *can't*.

So the honest answer to the question this audit exists to ask: **today bought a
better machine, not a better creature.** Jack gained an eye (PG.6, real and hard-
won through two documented failures), a certified fixture for proving fusion
(PG.7), a fixed observation vector, and a record that can now tell a run from an
edit. Those are the right things to have built and they were built well. He did
not gain a single new thing he can *do*.

And there is a specific reason to be uneasy rather than merely patient. The work
that is easy on this box — CPU fixtures, harness integrity, world properties — is
exactly the work that has been getting done, and the work that is hard — GPU
learning, curiosity, unison — is exactly the work that has been VOID or blocked
for four days. That is not laziness; every iteration escalated the blocker
correctly and then did the most valuable available thing. But four days of
correct local decisions have produced a ladder whose green section and whose goal
are drifting apart, and **the five missing senses are the proof that the drift is
not merely about ordering.** Nobody chose to defer smell and voice. They were
never on the map at all.

---

## FOR THE BUILDER

Ranked. None requires the owner. Items 3–5 are carried from **two** prior audits.

1. **NEW — make the ledger audit itself: `python -m experiments.run verify`.**
   Both probes in §1.1, as a command, costing nothing:
   (a) for every PASS, re-evaluate its *committed* `_check` against its
   *recorded* `metrics`/`control_metrics` and report any disagreement — this
   catches a check loosened after the run it certified, which nothing currently
   catches;
   (b) re-evaluate each with `control_metrics = {}` and fail any spec that still
   passes — a control that the check does not read is a control that is not
   there. Today both return clean (55/55 and 0/50), so the known-answer fixture
   is free: plant one loosened check and one control-blind check and assert the
   command catches exactly those two. Register it as **T0.18**; it belongs next
   to T0.13 (gates are live *forward*) as the *backward* check over the record.

2. **NEW — nothing owns a detached process (§4.3).** Two are running as I write:
   a 5-hour `run T2.01` GPU poll that no journal entry launched and that dies to
   its own `timeout 34000` at ~05:09, and a `run_pg6.sh` lock-waiter that wrote
   the ledger 25 minutes after its iteration exited (§1.4). Two concrete asks:
   (a) any process expected to outlive its iteration writes a line to
   `/data/jack-logs/ladder.log` at launch **and** at exit, including the spec id,
   the pid and the deadline, so `tail ladder.log` shows what is in flight;
   (b) `run status` prints in-flight runs — scan `/proc` for
   `experiments.run <SPEC>` and report spec, age and remaining timeout. Right
   now the only way to learn that T2.01 is running is `ps`, and three consecutive
   iterations saw it purely as a lock.

3. **Give T1.03 and T1.05 controls (§1.2).** T1.03: a parameter deliberately
   detached from the graph that *must* be reported as orphaned. T1.05: an
   unfrozen sentinel that *must* move. Both cheap, both convert "we observed the
   good thing" into "and the measurement can see the bad thing." Third audit.

4. **Make `Spec.control` load-bearing (§1.3).** 20 entries record
   `control_metrics` while declaring `control=None`. Have `run_spec` raise when
   `control_fn` is supplied and `spec.control is None`, then backfill the 20
   declarations. The declaration is the audit surface and it is currently 20/50
   wrong. Third audit.

5. **Re-run ME.8 at 3 seeds (§1).** PASS at `seeds=1` whose own commit message
   records a **seed-2 training collapse** fixed by a GRU retain-bias init — the
   fix was never verified at the seed that motivated it. `ME.8` declares no
   `seeds=` at all (`registry_expansion.py:406`), so it defaults to 1. Third audit.

6. **`trap ... EXIT` in `ladder_loop.sh` (§4.2).** 52 starts / 50 ends. Both
   missing ends landed work, so the log cannot tell "killed having done nothing"
   from "killed having committed a PASS."

7. **Register the missing senses (§3.2) — the highest-value item on this list,
   and the only one that changes what Jack can become.** Five of GOAL.md's named
   senses have zero specs among 137: smell, taste, voice, pain, temperature.
   Do not build them; **register** them, so they become visible to `run blocked`,
   `run next` and the Review instead of being invisible to all three. Voice is
   the one to write first — `CHAMPIONS.md:80` already says *"needs a spec"*, it
   is a prerequisite for GEN.02/GEN.03 (other minds) and for emergent language,
   and a first falsifier is cheap on this box: he emits a sound whose parameters
   depend on his state, and a probe recovers the state from the sound above a
   shuffled-pairing null. PG.5 already ships the modal-resonator synthesis and
   PG.7 already ships the leak-control pattern to copy.

8. **Build the outside-reference audit that `LESSONS.md:783` prescribed
   (§3.2).** That lesson is 30 hours old, names this exact hole, and says *"at
   least one recurring audit must measure against a reference from OUTSIDE the
   project's own documents."* No organ does. Cheapest honest version: a monthly
   (not weekly — no new organ without a scar, and this one has one) checklist run
   against the human sensory inventory and the GEN.* taxonomy, reporting
   *registered vs absent* per capability. It costs one agent-hour and it is the
   only check in the system that can see a hole in the map.

9. **`JACK_LOOP_MODEL=opus` in cron (§4.1).** 12 of the last 13 iterations opened
   with `OUT OF CREDITS on fable`. The fallback works; it should not be load-
   bearing every hour.

10. **Commit or discard the dirty ledger (§1.4)** — PG.6 `attempt: 4` is
    uncommitted in the working tree. It is a legitimate duplicate re-run (same
    `impl_sha`, same numbers), so committing it with a note is correct; leaving
    it dirty is how a `git checkout` silently reverts a PASS. **I did not touch
    `experiments/ledger.json` and my commit deliberately excludes it.**

---

## FOR THE OWNER

1. **D3 is still the whole bottleneck, and the deadline is now four days away.**
   *May the loop `git push` its own commits to `origin/main`?* Nine commits are
   unpushed; `assert_ref_is_current` refuses to build a GPU job from an unpushed
   HEAD, and it is right to (the VM clones from GitHub). **23.6 of 30 Kaggle
   hours expire at the end of 2026-08-15 and unspent free quota is not saved.**
   Behind it: T2.01's re-run, which alone would free **26** specs and unblock
   36 — including all 9 curiosity specs and the entire unison ladder. The repo is
   already public and already contains every file involved.

2. **Something launched `run T2.01` at 19:42 last night and it is still going —
   was that you?** It has been polling a remote GPU for five hours, no journal
   entry created it, and it will be killed by its own timeout at ~05:09 UTC. If
   it was you, the loop needs to be told so it can watch the kernel instead of
   routing around the lock; if it was not, something is submitting GPU work
   unattributed and that is worth knowing. Either way the answer belongs in
   `docs/LOOP_JOURNAL.md`. See §4.3.

3. **D1's option set is stale and one option contradicts your own decree.**
   `DECISIONS_NEEDED.md` asks you to pick A/B/C/D with **A — freeze the trunk**
   marked RECOMMENDED. Your PLASTIC-ONLY decree of 2026-08-09 21:16 forbids any
   frozen component inside Jack, which rules A out. `CHAMPIONS.md:64` also still
   lists "frozen-trunk+head" as a live challenger. I have appended the evidence
   to `DECISIONS_NEEDED.md`; reconciling it is yours. Note this does not mean
   decide D1 — the underlying evidence is still confounded and still needs the
   re-run that D3 gates. It means the *menu* should not offer a dish you have
   already sent back.

4. **Five senses you named as constitutional have no spec at all.** Smell, taste,
   voice, pain, temperature: 0 of 137 registered. Not blocked, not failing —
   absent, and therefore invisible to every report this system produces. I have
   asked the builder to *register* them (item 7) rather than build them, because
   registering costs nothing and makes the gap visible; building is a scope call
   that is yours. The one I would ask you to bless first is **voice**: GOAL.md
   calls it "how a creature acts on other creatures", it gates the other-minds
   expansion, and the synthesis machinery already exists from PG.5.

5. **Still waiting on you from two audits ago:** strike the "Kaggle GPU is not
   being granted" block at the top of `DECISIONS_NEEDED.md`. It claims to block
   T0.10 and T0.11 (both PASS since 2026-08-04) and asks you to choose an option
   when option 1 shipped five days ago and Kaggle is now the primary GPU backend.
   One line from you, and the suggested `DECISIONS_RESOLVED.md` entry is already
   drafted in that file.

6. **For information, no action needed:** the ledger is clean. I re-checked every
   one of the 55 PASSes against its own committed gate and every one of the 50
   controls for whether it is actually read. Nothing is unearned, nothing was
   quietly loosened, and the two hand-edits found in the last audit are now
   recorded *as* hand-edits inside the record itself. On the integrity axis this
   system is doing what it was built to do.

---

*Ledger untouched — `experiments/ledger.json` is dirty from a detached script,
not from me, and is excluded from this commit. No experiment re-run. Nothing
outside `/home/opc/jackthelearner` changed. This commit is not pushed; that is
D3, and it is not mine to decide.*
