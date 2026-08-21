# OVERSIGHT — 25th audit, 2026-08-21 07:00 UTC

## VERDICT: DRIFTING — not in *what* the builder works on (there is no goal drift; every unit in 24 h traces to a GOAL.md sentence) but into **not working at all**: the loop has declared itself blacked out on a meter that is not the gate, has dispatched nothing since 03:13, and its written plan holds that freeze until 2026-08-24 — one day *after* 22.54 free Kaggle GPU-hours expire. The ledger itself is the cleanest it has been: 83 PASS, `run verify` 0/82 failures, exactly two module constants moved in seven days and both are envelope growth.

The integrity checks found nothing that damages the ledger. The findings below
are about the machine stopping, and about two PASS claims that went stale for
no scientific reason. Ranked by damage.

---

## 0. Is the ladder the right ladder? — coverage clean, 12th straight audit

`experiments.coverage` exits 0. **0 of 23 constitutional commitments has no
declared spec.** The 2026-08-10 miss stays closed.

**9 of 23 have a claim-kind PASS; 14 have none.** Up one from yesterday: sight
returned when T3.01 v3 passed at 01:28. Six of the fourteen show a passing spec
the tool refuses to credit because it is a fixture or a sensor rather than a
claim (smell/SM.01, voice/VO.01, balance/BA.01, thermal/PS.02, hunger/PS.01,
death/XL.00) — the world can do the thing; nobody has shown *Jack* does.

The two GOAL.md warns are most likely to be quietly neglected are still the two
that are, and neither moved today:

- **Curiosity — 12 specs, 1 claim PASS** (T2.08). **T5.06 — "unprompted
  exploration is real", the ladder-and-apple sentence itself — is NOT_RUN**, as
  is T2.09, the noisy-TV adversary written to attack T2.08's own claim.
- **One brain / unison — 21 specs, 1 claim PASS** (UB.9, and see RANK 2: UB.9's
  own docstring now records a hole in it). T4.02 FAIL at 27.28× vs a 10× gate;
  UB.7, the headline UNISON claim, NOT_RUN; UB.10 parked.

## RANK 1 — The builder stopped itself on the wrong meter, and it is costing the week

**Three consecutive iterations declared a blackout against a stop that did not
fire.** The enforced gate is `usage_gate` in `scripts/lib_usage.sh`, which reads
`claude_usage.py --pct` — and that flag returns **`week:all models`, nothing
else**. Measured during this audit:

    session                 16%   resets Aug 21 09:00 UTC
    week:Fable              93%   resets Aug 24 05:00 UTC
    week:all models         77%   resets Aug 24 05:00 UTC   <- THE GATE
    claude_usage.py --pct = 77
    .usage-resumed          does not exist (and is not needed at 77%)

The 04:24, 05:07 and 06:07 journal entries and commits (`39bf5a1`, `901b263`,
`639112a`) all read *"Fable 91% (>90% hard stop)"* and planned no new work.
`week:Fable` is the meter for the model the crontab happens to pass
(`JACK_LOOP_MODEL=fable`). It is **not** the gate, and `ladder_loop.sh` already
handles Fable running out on its own: `FALLBACK_MODELS="opus sonnet"` fires on
the refusal, at a cost of ~3 s per dead attempt.

**The builder's own B6 plan, written at 01:14, records the distinction
correctly** — *"meter 01:14: Fable 86%, all-models 74%, stop 90%"* — and then
the next three iterations applied it to the wrong line. The plan also says the
thing that makes the freeze self-defeating: *"W33 hours die Sun 08-23
REGARDLESS, so anything worth W33 must be dispatched before ~88%, not queued
behind the stop."* Nothing has been dispatched since 03:13.

**What it costs, in numbers:**
- **22.5410 Kaggle GPU-hours expire Sunday 2026-08-23.** W33 stands at 7.4590 h
  charged (7.2012 ok + 0.2578 failed). Zero GPU spend in the last ~30 h; the
  last kernel was `jack-ladder-1787274738` at 01:12.
- The freeze as planned runs to the **Aug 24 05:00** reset — **a day past the
  Kaggle expiry**. Every remaining W33 hour dies under this plan.
- **The freeze does not even save the meter it was protecting.** The three
  "lean liveness" passes took `week:Fable` from 89% → 91% → 93%. Iterations
  that produce a commit saying nothing happened still cost tokens.
- Yesterday I wrote that the hours would expire "into an empty queue." **That is
  no longer the reason and I am correcting it:** T3.01 is un-frozen and has
  passed, LC.03 v2 is running on CPU, and the queue has candidates. The hours
  are now expiring because of a misread meter.

**Credit where it is due:** the Review organ, which started at 06:37 and was
still running as I wrote this, found the same thing independently and has put a
correction at the top of `scripts/ladder_prompt.md` (uncommitted at 06:5x). I
verified it against `lib_usage.sh` and the live meter myself. I am recording it
here anyway because a prompt file is a working surface that the next commit can
rewrite, and this is the largest item on the board.

**The fair half of the argument, which the Review states and I endorse:**
running on opus draws the same `all models` pool the Review, the field watch and
this audit draw from, so burning it to 90% takes the *auditors* down with the
builder. That argues for **restraint**. It does not argue for **zero**. The
correct posture is the one the builder already wrote down: *dispatch, then idle*
— Kaggle kernels and `launch_detached.sh` runs compute through any blackout and
write their own receipts.

## RANK 2 — Two PASS claims went stale for zero scientific reason, and seven answers were routed away from their gates to avoid making it worse

`run status` reports **2 stale PASS claims — T2.06 and UB.9** — where yesterday's
audit reported 0 and all three stale entries were VOIDs. Both were staled by one
commit, `eec7d86` (the 24th audit's B3 sweep). I read the diffs: **both are
docstring-only additions; the executable code is byte-identical.** No gate, no
threshold, no assertion changed.

That is a false alarm in the science, but it is a real cost in two ways, and the
second is the one that matters:

1. The headline "83/169 demonstrated" now includes two entries the project's own
   instrument says are about older code. A reader cannot tell a doc-only stale
   from a real one without doing the diff by hand, which is the work the
   instrument exists to save.
2. `eec7d86`'s own commit message states the seven remaining B3 answers were put
   in `LESSONS.md` rather than in the test docstrings **"to avoid a mass
   staleness alarm."** So the audit answer to *"what observable distinguishes
   converged-at-chance from never-trained?"* now lives one page away from the
   gate it describes, for seven gates, because writing it next to the gate would
   turn seven PASSes amber.

The builder cites `protocol.py`'s doctrine that mass staleness flags "teach the
loop to ignore staleness warnings." I checked it: that passage
(`protocol.py:884-895`) is about **`IMPL_DEPS` hashing being opt-in**, not about
docstring edits. It is an analogy, not the precedent it reads as.

**The repair is NOT to stop hashing docstrings.** In this project the
pre-registered gates *live* in the docstrings — LC.03's V2 registration block,
T3.01's fork (ii), the whole PRE-REGISTERED GATES section of every test. A hash
that ignored them would let a pre-registration be rewritten after the run, which
is precisely the T0.27 hazard. The clean repair is a **doc-only amendment lane**:
the ledger row already carries an `amended` list with six clean
provenance-only precedents (T0.05, T0.09, T1.07–T1.10), each attributed to a
named audit with a reason. A prose-only edit that leaves the AST unchanged is a
smaller claim than those, and is machine-checkable.

**And inside this finding is a substantive one the B3 sweep produced, which
deserves ranking rather than filing.** UB.9's new docstring records, honestly:

> *"the unimodal arms have no must-learn target of their own and their loss
> descent is not recorded, so a PER-ARM recipe pathology (UB.10's measured
> disease — one uniform recipe leaving one matched-param arm dead) is not fully
> excluded by this design."*

**UB.9 is the only claim-kind PASS behind "one brain / unison" (21 specs) and
one of two behind hearing.** The disease named there is not hypothetical: UB.10
measured it. That caveat is now on the record, unresolved, and unscheduled.

## RANK 3 — LC.03 v2 has no pre-registered stop, and its own sizing model predicts it lands just short

The v2 re-screen (registered `5074440`, launched `39bf5a1`, running now) grew the
envelope 4× — N_STEPS 100k→400k, W_CLOCK 4320→17280. **The growth itself is
legitimate and well-argued**: control (c) fired on a single food quantum
(6.9 s on a 22-life ruler against a 5.0 s floor), and more lives per twin takes
that quantum to ~1.7 s — which *strengthens* (c). Gates unmoved, arithmetic
published in the registered docstring rather than hidden. Good work.

The problem is what the same arithmetic predicts about the claim side. Quoting
the docstring's own sizing:

> *"k=4 gives dreamer-xs 2.95 × 150/2 = +221 s ≈ its +226 s requirement"*

The envelope was sized to the **second** arm, because the gate needs two
learners — and by its own model that second arm lands **~2% below** its 3σ
requirement. This is a **63-hour, ~190 core-hour run whose pre-registered
expectation is a coin flip on the conjunct that decides it.** (Verified alive
and genuinely computing: pid 310395 plus 3 workers, 2:21:42 CPU time each at
2:22:45 elapsed, 99.2% CPU, ~400 MB RSS; log header-only at 146 bytes as
expected while block-buffered.)

**There is no cap on LC.03 re-screens.** I grepped for one. T3.01 carries a
one-diagnostic cap; SM.02 was parked; LC.03 has neither. Attempt 1 VOIDed
(schedule confound), attempt 2 VOIDed (food quantum), attempt 3 is 4× bigger. If
it lands with fewer than two arms at 3σ, the next move under the current record
is k=8 — ~380 core-hours, ~5 days — and that is the run-until-pass ratchet with
a compute bill attached.

Nothing has been broken yet. The ask is that the fork be committed **before the
number arrives**, which is the only time pre-registration means anything.

## 1. Integrity of the ledger — no findings

| check | result |
|---|---|
| PASS entries | **83** |
| PASS resolving to an implementation in `experiments/tests/` | 83 / 83 |
| PASS whose `commit` still exists in git | 83 / 83 |
| PASS declaring a control but carrying empty `control_metrics` | 0 |
| PASS recorded from a dirty tree | 0 |
| `run verify` — verdicts that no longer re-derive | 0 / 82 |
| `run verify` — gates that IGNORE their control | 0 / 80 |
| `run verify` — controls declared but never run | 0 |
| `run verify` — gates unreplayable / entries unauditable | 0 / 0 |
| controls run but NOT declared (the ratchet debt) | **0 / 0** |
| stale PASS claims | **2** — T2.06, UB.9 (RANK 2; doc-only, verified) |

Two PASSes declare no control at all — **T0.01** (imports) and **T0.10** (Kaggle
round-trip) — unchanged, by design, and still carrying the honest caveat from
earlier audits: an existence claim whose gate was never shown capable of
reporting the bad case. `run verify` self-excludes one entry (T0.18 cannot
re-judge its own row); its gate is exercised by T0.18's control.

One dirty stamp remains — **LC.03**, recorded VOID at `a483302+dirty`. It is a
VOID, not a claim, and v2 is in flight from a clean tree to replace it.

Six entries carry an `amended` record; every one is provenance-only (a
`hardware` string, an `attempt` count), attributed to a named audit, with a
reason. **No metric and no verdict has ever been amended.**

## 2. Thresholds and controls over time — no findings

Method: AST parse of every file under `experiments/tests/` plus `registry.py`
and `registry_expansion.py` **present at both** `8390e0d` (2026-08-13 23:18) and
HEAD, comparing every module-level uppercase constant. Not a diff read.

- **Exactly two constants moved, both in `lc_03_survival_screening.py`:**
  `N_STEPS` 100,000 → 400,000 and `W_CLOCK_CORE_S` 4,320 → 17,280. Both are
  **envelope**, both are in the direction of *more* compute, both are argued
  from recorded curves in a registered docstring block, and the claim gates
  (`SIGMA_GATE` 3.0, `NOISE_FLOOR_S` 5.0, the claim conjunction) are untouched.
  See RANK 3 for the hygiene ask that goes with it — it is not a loosening.
- **Zero thresholds moved. Zero spec or test files deleted. Zero controls
  removed. No `_check` gained an `or`. No seed count reduced** (T3.01 ran its
  registered [0, 1, 2]).
- **One control was DEMOTED and it checks out.** T3.01's `SHUFFLE_FIT_FLOOR`
  went from a VOID gate to a recorded diagnostic. This was pre-authorised by the
  24th audit as fork (ii), and I verified in the code rather than the commit
  message that it was executed in the required order:
  - the deterministic structural gate landed **first** and is a real VOID exit
    (`t3_01_ablate_vision.py:468`, `hash_overlap_max != 0 → VOID`), and it read
    **`hash_overlap_max` 0.0** on the passing run;
  - `SHUFFLE_BAND` remains a VOID gate — the positive-evidence leak test still
    fires;
  - the caveat the 24th audit demanded is in the docstring, and the dry-check
    suite pins the new behaviour explicitly (*"dead control arm → recorded, not
    gated"*).

  **The standing residual, stated plainly so it is not lost:** T3.01's PASS is
  not protected by a demonstrably *live* leak control — `shuffled_fit_min` read
  0.25 on two of three seeds, i.e. dead. Its protection is hash-disjointness
  (identity leak, deterministic, conclusive) plus `acc_pixshuf` at 0.25 on all
  three seeds, which does exclude a pure per-class marginal-statistic shortcut
  since pixel shuffling preserves the histogram. Honest limit on that comfort:
  `acc_pixshuf` is a test-time perturbation of a trained model, so its
  at-chance reading is partly confounded with distribution shift. The claim is
  sound; the leak coverage is good but not the complement pair the 24th audit
  described.

Method limit, stated: this catches module-level constants, not a number
hardcoded inline inside a `_check`. I read the removed lines of the seven-day
diff for that shape and found none.

## 3. Drift from the goal — no drift in the work; the drift is into idleness

Every unit in the last 24 h traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| T3.01 v3 redesign + run → **PASS** | *"ablate a sense, something measurable must degrade"* (sight) |
| B3 at-chance-control sweep | *"protects the honesty of watching what happens"* |
| LC.03 harvest, corrected diagnosis, v2 registration + launch | *"he lives, he dies, he remembers"* |
| B4 W0.BAL body bakeoff → D9 | the body; T2.01's *"needs a better body"* |
| B6 hard-stop plan | keeping the loop alive |
| 3 × liveness-only passes | nothing |

The converse remains §0's answer, and it did not move: **T5.06 has still never
run.** The single image GOAL.md uses to define the project — climbing on attempt
40 after falling on 1–39, unprompted — sits behind a spec that has never been
executed, in a commitment holding 1 claim-kind PASS out of 12 specs.

**Nothing was quietly acted on.** I checked the converse directly for the body,
which is D8/D9's owner-gated subject: `playground.py` is **not** among the files
changed in 24 h, and `experiments/w0bal_probe.py` builds its A/B/C bodies by
monkey-patching the spawn function inside its own `_build` (line 119-141), not
by editing the shipped geometry. Running was not adopting, in the code as well
as in the prose.

## 4. Is the builder alive and productive? — alive, disciplined, and now voluntarily stopped

- **24 iterations in 24 h (08-20 07:07 → 08-21 06:07), all `rc=0`, none lost.**
  `lost_iterations.log` is 0 bytes. The last non-zero exit was 2026-08-19.
- **PASS delta 82 → 83 (+1)**, the T3.01 restoration. The 82↔83 oscillation in
  the log is the instrument working, not noise: the count fell when T3.01's
  20:26 VOID displaced its PASS and rose when v3 passed at 01:28.
- **LC.03 v2 verified alive by process, not by claim** (the standing
  do-not-take-the-builder's-word rule): 3 workers, 99.2% CPU, CPU time within
  63 seconds of elapsed time — genuinely computing, not a stalled coordinator.
- **The last three iterations produced no science by choice.** See RANK 1. At
  one iteration per hour, the plan as written spends **~70 more iterations**
  doing liveness checks, and the compute it is protecting expires first.

## 5. Compute honesty — no findings on the accounting

- **W33: 7.4590 h charged** (7.2012 ok + 0.2578 failed across 20 kernels).
  **22.5410 h remain**, expiring Sunday 2026-08-23.
- **Last 24 h: 3.7182 h across 14 kernels**, every one attributable to a named
  job with a recorded outcome. The one kernel since yesterday's audit
  (`jack-ladder-1787274738`, 0.2713 h, P100) produced T3.01's PASS row. Three
  kernels failed for 0.1643 h total, each with a named environment cause.
- **No unattributed hours. No GPU-hour without either a ledger row or a named
  probe purpose in a commit message.**
- The honest note is not about the accounting, it is about the spending: **zero
  GPU hours in the last 30 h, against a quota that dies in ~2 days.** That is
  RANK 1's cost, restated as a number.

## 6. Stuck decisions — unchanged, and one correction owed to the owner

- **D1** (does the 57M trunk stay in the control path) — open **12 days**,
  blocks T2.01 and 36 specs including the only claim specs for six senses.
  Priced five times now. Not re-argued here.
- **D7** (MovementMoodCoupling) — READY TO DECIDE since 08-20; T3.07's re-run
  came back bit-identical, so no measurement remains that would inform it.
- **D9** (the body fork) — B4 executed exactly as ordered; see §7.
- **Nothing was quietly acted on** — verified in §3.
- **Correction owed:** yesterday's FOR THE OWNER said the Kaggle hours would
  expire "into an empty queue... because every GPU-capable spec is parked,
  escalated, or behind D1." That was true at 00:40 and is not true now. The
  queue has candidates. The hours are expiring for the reason in RANK 1, and the
  owner should not read yesterday's page as still accurate.

## 7. Bakeoff hygiene — no findings

`DECISIONS_RESOLVED.md` is unchanged at three records (PS.01/J VOID, PS.01/J2
WINNER `impact_speed`, D2), all audited clean previously and unaltered.

One bakeoff ran in the window — **W0.BAL (B4)** — and correctly produced **no**
record, because nothing was decided. Its hygiene is the best I have audited:

- The pre-registration was preserved **verbatim** in D9 and executed as written:
  arms A/B/C, `upright_frac` and `hand_reach_z_max`, 3 seeds × 500 decisions,
  **identical per-seed uniform-random action streams**, same mutated worlds,
  null = the rover as built.
- Arm B's "bounded" torque was **priced before running** (KP = TMAX = 120 N·m
  against ~88 N·m worst-case toppling) rather than tuned to the answer.
- **The kill criterion is reported as NOT firing, against the builder's own
  thesis**, and the entry's premise was *narrowed* rather than defended: *"a
  prone rover cannot raise a hand to a rung"* became *"not for the ~99.7% of its
  life it spends prone."* That is a finding written against interest.
- Arm C's cost (it rewrites PG.3's inherited geometry, so inheritance
  certificates need re-checking) is carried forward, not dropped.
- **No learning gate, correctly.** No arm learns — the action streams are
  identical and random — so this is a physical-property screen, not an
  architecture arbitration, and T2.02's two-non-learners hazard does not arise.
  The absence is right, not an omission.

## 8. The honest summary — are we closer to a curious humanoid?

**Yesterday: closer in trustworthiness, further in capability. Today: closer in
capability by exactly one rung, and about to lose three days.**

The one real gain is real. **Sight is a claim again**, and this time it is a
claim that survived the gate that killed its predecessor: `acc_full` 0.61–0.63
against a 0.4467 frozen-probe reference, ablated and pixel-shuffled both at
chance 0.25 on every seed, `hash_overlap` provably 0.0. The retraction on 08-20
and the re-earning on 08-21 are the same machine working, thirty hours apart,
and the second number means more because the first one was allowed to be taken
away.

**But the ladder-and-apple standard is exactly where it was twelve audits ago.**
T5.06 has never run. Curiosity holds 1 claim PASS out of 12 specs. Unison holds
1 out of 21, and today that one — UB.9 — wrote into its own docstring that it
cannot exclude the disease UB.10 measured. We are still adding rungs near the
bottom of a ladder whose top has never been touched.

And the thing that would most obviously help is free and about to be thrown
away. There are 22.54 GPU-hours sitting in an account that resets on Sunday, a
builder with 13 points of headroom on the meter that actually gates it, and a
plan in the journal that says do nothing until Tuesday. **The system is not
being throttled. It is standing still while holding a full tank**, and it is
doing so out of a misreading it wrote down correctly six hours earlier. That is
the whole audit. Fix the meter reading and the rest of this page is a good day.

---

## FOR THE BUILDER

Ranked. B1 is worth more than everything else on this page combined and it is
one command.

**B1 — YOU ARE NOT IN A BLACKOUT. Read both meters and act on the right one.**
Run `scripts/claude_usage.py` with no flags, print all three lines, and state in
the journal which one you are acting on and why. The gate is
`week:all models` = 77% against a 90% stop; `week:Fable` = 93% is the meter for
the model the crontab passes and is **not** the gate. `.usage-resumed` does not
exist and is not needed. `FALLBACK_MODELS="opus sonnet"` already handles Fable
refusing.
  - **Then apply your own B6 plan as written**: *dispatch, then idle*. Anything
    worth W33 must be **dispatched before Sunday 08-23**, when 22.5410 h die.
    Kaggle kernels and `launch_detached.sh` runs compute through any blackout
    and write their own receipts, so a dispatch costs you one lean iteration and
    buys a result you cannot buy on Monday.
  - **Restraint is still correct, and zero is not restraint.** Opus draws the
    same `all models` pool the Review, the field watch and this audit draw from;
    burning it to 90% takes the auditors down with you. Budget for that
    explicitly — a lean dispatch iteration, not a full one — but budget
    something.
  - **Correct the three journal entries** (04:24, 05:07, 06:07) in place rather
    than silently: they are the record, and they say the stop fired when it did
    not.

**B2 — pre-register LC.03 v2's fork NOW, before the number lands.** The run's
own sizing says the second arm reaches +221 s against a +226 s requirement, so a
sub-two-learner outcome is roughly as likely as not, and there is no cap on
re-screens in the spec (T3.01 has a one-diagnostic cap; LC.03 has none). Commit,
before harvest, which of these fires:
  - **(i)** one further growth, with the k, the core-hour cost, and the
    arithmetic that says it clears — and a hard statement that this is the last;
    or
  - **(ii)** the screen is CONCLUDED. Record the finding as what it is — *W0 does
    not discriminate these cores at a reachable envelope* — which is a result
    about the world and about LC.04's premise, not a failure to be re-rolled.

  Either is defensible. Choosing after seeing the number is not. This costs one
  paragraph and it is the difference between a screen and a ratchet.

**B3 — close the doc-only staleness hole properly; do NOT loosen the hash.**
T2.06 and UB.9 are stale from prose-only edits (I verified the diffs), and seven
B3 answers were routed to `LESSONS.md` to avoid staling seven more passing
specs. Both halves of that are a tax on documentation.
  - **Do not strip docstrings from `impl_sha`.** The pre-registered gates *live*
    in the docstrings; a hash that ignored them would let a pre-registration be
    rewritten after the run — the exact T0.27 hazard.
  - **Build the doc-only amendment lane instead.** The ledger row already has
    `amended`, with six clean provenance-only precedents. A prose-only edit is
    machine-checkable: `ast.dump` of the old and new file identical (or
    `ast.get_docstring` stripped and the remaining tree byte-equal) ⇒ record an
    `amended` entry carrying old sha, new sha, author and reason, and let
    `stale_claims()` treat it as answered. Anything that changes the AST stays
    stale, loudly.
  - The precedent citation in `LESSONS.md` is inexact and worth fixing while you
    are there: `protocol.py:884-895`'s "mass false alarm" doctrine is about
    `IMPL_DEPS` hashing being opt-in, not about docstring edits.

**B4 — UB.9's recorded gap needs a spec, not a docstring line.** UB.9 is the
only claim-kind PASS behind "one brain / unison" (21 specs) and one of two
behind hearing, and its own docstring now says a per-arm recipe pathology —
UB.10's *measured* disease — is not excluded by its design. Rank it as ladder
work: either a same-run must-learn target for each unimodal arm, or a recorded
loss-descent per arm, or an explicit statement in the registry that UB.9's
claim is conditional on the shared-trainer argument. It does not need to be done
this week. It needs to stop being prose.

## FOR THE OWNER

Nothing here is new science, and I am not re-arguing any decision.

**1. I owe you a correction from yesterday.** I told you 22.81 free Kaggle hours
would expire "into an empty queue... because every GPU-capable spec is parked,
escalated, or behind D1." **That is no longer why.** T3.01 is un-frozen and
passed overnight — sight is a claim again — and the queue has candidates. The
hours (now 22.5410, expiring Sunday 08-23) are on course to expire because the
loop misread its own usage meter and declared a three-day blackout that its gate
never triggered. That is a builder-side fix, it is in FOR THE BUILDER as B1, and
it needs nothing from you.

**2. D7 has now been ready to decide for 54 hours and takes ten seconds.**
MovementMoodCoupling failed its ablation twice; the re-run on current code came
back bit-identical, so no measurement remains that would inform the call. Delete
it, redesign it, or **accept it on the record as cosmetics** — (c) is a
legitimate answer and closes it. Until then the model carries a component that
has twice failed to earn its parameters, which is the disease `GOAL.md:87`
exists to prevent.

**3. D9 now has its table, and nothing was adopted.** The bakeoff you were asked
to order was run at this desk on free CPU, per my instruction yesterday that
*running is not adopting*. Arm C (0.35 m plinth, mass in the base) is upright
100.0% of decisions on all three seeds and reaches ~1.17 m of a ~1.19 m ceiling;
the as-built rover is upright 0.2–0.4%; arm B rights it 9–26%. The kill criterion
did **not** fire, which honestly narrows the entry's own premise. Your decision
is unchanged — park / order / fold into D1 — you simply now have the numbers
instead of three options and a shrug.

**4. D1 is 12 days open.** Priced five times; I am not pricing it a sixth. The
narrow question from yesterday stands and is answerable in one line: **if D1 is
going to stay open past Sunday, say so**, and the loop will stop sizing its week
around a quota it cannot spend.

---

*Method note: every number above is from a command run during this audit —
`experiments.coverage`, `run status`, `run verify`, `claude_usage.py`, an AST
parse of every test/registry file present at both `8390e0d` and HEAD, direct
reads of `ledger.json` / `gpu_budget.json` / `lib_usage.sh` / `ladder_loop.sh` /
`t3_01_ablate_vision.py` / `lc_03_survival_screening.py`, `git show` of the two
staleness-triggering diffs, `ps` against the LC.03 v2 workers, and
`/data/jack-logs/ladder.log`. Nothing is quoted from a commit message without
being checked against the artifact it describes.*
