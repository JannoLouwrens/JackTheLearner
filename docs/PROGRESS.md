> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The review run that wrote this file exited rc=124 and did not
> complete its own checklist (2026-10-22T06:57:09+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.

> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: newest row in docs/PROGRESS_LOG.md is 2026-10-19 (2d old; the schedule allows 1d)
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-10-21T06:37:04+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

> **PARTIALLY REFRESHED 2026-10-20 BY A COMPLETED DAILY SITTING — read this
> before trusting the staleness banner below it.** The DAILY Review of
> 2026-10-20 ran to completion and committed six acts. A DAILY does NOT
> rewrite this page (its remit is `FOR THE BUILDER` plus one `PROGRESS_LOG.md`
> line), so the seal's banner stands for everything it does not touch — but
> taken unqualified it would mark TODAY's most perishable order as a record of
> an older world, which is the opposite of true. **FRESH, measured 2026-10-20:**
> `FOR THE BUILDER` item 0 (the usage gate has released; 30.0 GPU-hours expire
> Saturday 2026-10-24), item 4 (`longrun_unbound` is SIX specs, not ten), and
> `FOR THE OWNER` item 4 (the blackout attributed to the 90% stop, not the
> pacer). **STALE, as the banner says:** Part 1's counts, Part 2's sweep, and
> every "current state" framing — those are the 2026-10-18 FULL's record. In
> particular `consecutive dark builder slots | 14` in Part 1 reads **15** today.
>
> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: docs/PROGRESS.md itself last moved 29h ago against a 25h cadence; PROGRESS_LOG.md's fresh row is the dying run's own B4 disclosure and cannot vouch for the page
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-10-19T12:37:03+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

Written by the Sunday FULL Review of **2026-10-18**. Current-state by design:
anything not carried forward here is either committed elsewhere or routed to a
`D` id. Every act on this page was committed BEFORE the page was written —
the page is the receipt, not the work.

## Part 1 — the state of progress

### The numbers, recomputed this sitting, never quoted

| reading | value | note |
|---|---|---|
| registry | **257** specs | |
| ledger rows | **160** | so **97 registered specs have no row at all** |
| PASS / FAIL / VOID / BLOCKED | **107 / 35 / 17 / 1** | |
| pass rate vs registry | **41.6%** (107/257) | Aug 7–9 read 40.0% → 38.3% |
| pass rate vs rows run | 66.9% (107/160) | |
| rework (attempt > 1) | **79%** (127/160) | |
| specs RUN this week (10-11→10-18) | **5**, of which 4 PASS | T0.02, T0.12, T0.21, T0.36 |
| specs RUN last week (10-04→10-11) | **11**, of which 5 PASS | |
| `ledger.json` commits, this week / last | **15 / 43** | |
| consecutive dark builder slots | **14** (14.7 h) | hourly cadence → 7× the 2× trigger |
| `review-queue` | **0 violations**, OVERDUE **6 → 0** | all six disposed this sitting |

**Goodhart check, and it is the week's most important number.** The pass rate
did NOT fall — 41.6% against August's 38.3%. But **all four of this week's
PASSes are `T0.*`**: Tier-0 instrument self-tests, re-bought mechanically by
`regate`'s rotation lane, which by construction cannot advance `GOAL.md`. Count
rising, rate rising, and **zero frontier movement**. The honest reading is that
the rate is now measuring the re-buy lane, not the ladder. A rate that improves
while the frontier is still is not good news; it is the metric detaching from
the thing it was built to track.

**The frontier.** The single most important unblocked item is `t205`'s ordered
`T2.14 → T2.05` port (DUE 10-25): zero GPU hours, zero world edits, zero new
instruments, and it converts the project's longest-standing unattributable FAIL
into an attributable one. The builder is not working on it, because the builder
is not working at all — 14 dark slots. Effort-vs-goal for the week: of the 15
`ledger.json` commits, **15 are mechanical re-buys**. No commit this week served
`GOAL.md`'s current stage.

### The honest paragraph — no numbers

We are busier, not closer, and this week the machinery that proves it got
better while the creature stood still. The instruments are in excellent health:
the queue's violation class is empty, the certificate lane re-derives itself
twelve times a day, and this morning the oldest cohort of passing tests was
audited and came back cleaner than anyone predicted. That is real and it is not
nothing — a project that cannot trust its own green ticks cannot learn
anything. But every one of those improvements was an improvement to the
measuring apparatus, bought with slots the builder could not spend on Jack. The
single most important step toward him was discovering, by accident, that a guard
we already invented for one test had never reached its sibling, and that this
absence is precisely why our world model's defeat by a straight line sat
unexplained for six weeks: we had the instrument the whole time and never
carried it next door. The most concerning drift is that the thing holding the
builder dark is not a bug anyone can fix, and we have now spent two consecutive
weeks' worth of free compute without a decision about whether to spend it —
the failure is not that the hours expire, it is that nobody is able to choose.

## Part 2 — the test re-examination (the gates sweep, discharged)

Run as the sweep that `gates-that-measure-something-other-than-what-they-say`
part (a) ordered, against its own three named shapes, oldest-passed first:
**T0.19, T1.02, T2.03, T2.04, TA.02, T3.01, T2.19, T2.09, T2.14, T0.18, T0.09,
T0.10, T1.09, T1.06**, plus **T0.22** and **T2.00** for the battery shape.

**The result is not the one the row predicted, and that is the finding.**

- **Shape 3** (a declared control the check never compares against the
  headline): **ZERO instances.** Checked against the declared `Spec.control`
  field, never prose. `T0.10` reads as controlless and is not — its control is
  an explicit `NoControlByDecision` carrying the 52nd audit's B5 reasoning.
- **The battery shape** (`properties_failed == 0` passing a battery that stopped
  early — T0.13's first bug): **already guarded, correctly.** `T0.19` and
  `T0.22` both pin `properties_checked == N_PROPERTIES` on the experiment **and**
  the control, and both require the control to fail on SPECIFICALLY NAMED
  properties. **`T0.19` is the pattern every battery in this repo should be read
  against.**
- **The capped-teacher shape**: `T2.04` and `T2.14` carry **more** guards than
  `LG.03` did — `nn_informative_all`, `clip_overlap_max > 0 → VOID`,
  `floor_selftest_ok`, `task_not_trivial_all`.

**So the class is real but it is not population-wide rot — it is UNEVEN
PROPAGATION.** `T2.14` gates on `task_not_trivial_all` (*a linear map must not
annihilate the task; a good score here would be arithmetic, not imitation*);
`T2.05` measures the same shape against `mse_ridge` and **has no such
conjunct**. That absence is exactly why `T2.05`'s FAIL was unattributable for 43
days and why `t205` broke four dated promises: its two offered repairs were
never two choices, they were one unmeasured quantity.

### STRENGTHENED / REWRITTEN

| spec | what changed | why it is STRONGER |
|---|---|---|
| **LG.03** | `87fc826`. Both LIVE comments that still taught the refuted teacher-caps-twin mechanism replaced with the justification that survives (an un-aceable calibration cell is an invalid LIVENESS VENUE, not a bound on the student). Refutation left visible in place. | The docstring recorded the refutation since 09-12 while the comments at eye level taught the falsified claim — a future desk auditing the load-bearing conjunct ORDER would have inherited it. Comment-only, verified by `git diff -U0 \| grep -v '^[+-]\s*#'` returning nothing. `PLANNER_CALIB_MIN` stays 1.0, `CALIB_MIN` untouched, conjunct order untouched. Bill: 0 certificates (LG.03 is VOID). |
| **T2.05** (ordered, not yet executed) | Port `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok` pair; report `mse_nonlinear` and `nonlinear_headroom`. DUE 10-25. | ADDS a VOID gate with a working self-test and REMOVES an unearned inference. No threshold moves: `WM_RATIO` stays 0.8, `CTRL_TOL` stays 0.98, ridge gate stays ratio-free, claim conjunct byte-untouched. Bill: 0 standing PASS certificates. |

Nothing was weakened. No threshold moved downward anywhere this sitting.

## Part 2.5 — steering maintenance

- **OVERDUE class: 6 → 0**, every row disposed by act, stamp or dated re-arm.
  No row deleted, no `DUE:` dropped, nothing relabelled `HELD`.
  `standing-pass-certificates…` **ACTED** `4a3e646` · `t032…` **ACTED**
  `ac51b29` · `t205` **RULED → DISPOSITIONED** 10-25 · `gates-that-measure…`
  **DISPOSITIONED** 10-26 (part (a) discharged) · `lg03-teacher…` **ACTED**
  `87fc826` · `pl02-void-gate…` **re-dated 10-27 as a holding action** → `D49`.
  All six now declare `WAITS-ON`; **2026-10-17 has dropped out of
  `review-queue`'s WITHHELD list entirely.**
- **ORGAN LIVENESS.** Overseer live (06:37 today, on cadence). Field watch live
  (10-12 Monday, next due 10-19). Review: this sitting — the previous four days
  were gate-deferred and the 10-17 DAILY died `rc=124`. **Builder: fires hourly,
  has not iterated in 14 slots.**
- **Seat staleness / anatomy audit:** 32 declared seats over 257 specs; the
  2026-08-09 scar gaps are all now SEATED (smell, taste, voice, body schema,
  emotion, person model). Standing findings unchanged and NOT re-routed:
  `NO-ARENA` on ASR and Speaker ID, `TRIGGER DEBT` 3/3 (Learning core, Fast/slow
  coupling, World). As in the last five sittings, every match reads "not
  progressing" because nothing is progressing — that is a reading of the
  blackout, not of the arena, and it is not routed as a seat finding again.
- **PRIORITY RECONCILED — AUDITED CLEAN, THEN AN ATTEMPTED EDIT WAS REVERTED,
  AND THE REVERT IS THIS SITTING'S LAST FINDING.** The live `1^22` block was
  audited for the dangerous failure mode (a FALSE live order at the top of the
  builder's page): **none found.** Its cached GPU figures were already excised
  on 10-17, it points at living sources, and its one status claim verifies —
  `t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign` is
  still OPEN, DUE 2026-10-21, exactly as written. **But this sitting's two new
  builder units were not on the page**, so a `1^23` block naming them was
  written — and then **reverted by its own author**, because `steering` priced
  it: the insertion took `scripts/ladder_prompt.md` to **130,290 bytes, 782
  below the 131,072 EXEC CLIFF**, past which the builder does not read a
  degraded prompt — **it does not start, and the slot looks like an ordinary
  `rc=126`.** Reverted; the page is back to 126,721 B, 4,351 below the cliff.
  **The two orders therefore live in FOR THE BUILDER on this page only**, which
  is a weaker channel than the builder's own prompt and is named as such rather
  than papered over. **Also recorded, because the instrument said it and this
  desk did not act on it:** `steering` flags the item-1 order
  `SUBJECT T2.14 BLOCKED — T1.08 (FAIL)`. The ORDER is still legitimate — it is
  a spec-file port, not a run of `T2.14` — but whether the re-run that pays its
  bill can execute today is NOT established by this desk, and the builder should
  derive that before spending a slot.
- **NOT DONE, named rather than left to look done:** `FIELD_WATCH.md` was
  **not** consumed (it has not changed since 10-12 and its week-10 findings were
  all routed by the 10-13 sitting, so nothing is owed there). The sitting spent
  its clock on six OVERDUE disposals, the Part 2 sweep and two committed
  strengthenings.

## FOR THE BUILDER — ordered, cheapest and most perishable first

**REWRITTEN BY THE DAILY OF 2026-10-22.** Every number below was measured at
this sitting; nothing is inherited from the 10-18/10-20 text it replaces.

0. **28.91 FREE GPU-HOURS EXPIRE SATURDAY 2026-10-24 — two days — AND THE
   PREVIOUS VERSION OF THIS ITEM UNDERSTATED THE SPEND, SO RE-READ IT.**
   Measured by `gpu.Budget()`'s own accounting, not by hand: week `2026-W42`
   (Sun 10-18 → Sat 10-24) shows **1.0871 h used** of `KAGGLE_WEEKLY_HOURS =
   30.0`, so **28.91 h remain** — the page previously said `0.0 spent`, which
   is no longer true. `2026-W41` closed at 0.0 and is gone for good.
   **This still does NOT authorise a GPU dispatch** and the reason is
   unchanged: the two specs that could absorb hours (`T2.03`, `T2.14`) both
   read `needs T1.08`, and `T1.08` is FAIL at attempt 5. Nothing in the
   long-run GPU set is buyable, which is exactly why the hours keep expiring.
1. **`LC.08` IS REGISTERED, REACHABLE AND UNBOUGHT — 14.40 core-h of CPU, and
   it is the cheapest real science on your board.** Registered at `da9ece0`
   with both ruling conjuncts verified in source at this sitting (paired on
   identical seeds/budget/venue; scored on the seat's own `lg_margin_null` and
   `lg_margin_twin`). Its five dependencies all PASS, it has **no ledger row**,
   and its staleness bill is **0 certificates** — the cost is CPU and nothing
   else. It answers a live seat question: is 17.3% of the seated `A4` arm dead
   weight? It is CPU, so the GPU gate above does not touch it. Its purchase is
   owned by the row `lc08-comparator-registered-and-unbought` (OPEN, DUE
   10-26), routed this morning precisely so this compute could not be parked
   by its parent closing.
2. **Repair `t306_null_walker_probe.py`'s `V2` self-validation, and the repair
   is CONSTRAINED: round the replayed value to the recorded precision and
   compare exactly. A widened epsilon is REFUSED in advance.** `V2` currently
   reports `"ok": false` on a run that replays CORRECTLY. The guard at
   `:252-253` rejects on `abs(diff) > 5e-5` against ledger columns its own
   docstring (`:59-60`) calls *"rounded as recorded"* at 4 dp — whose maximum
   legal error is exactly 5e-5. All four reported mismatches are that boundary
   (0.00075 recorded as 0.0008; 0.01975 as 0.0198), each computing to
   5.000000000000002e-05; every `cov` difference is well inside tolerance.
   `V1` (1e-12) and `V3` pass, so the measurement itself stands. Loosening the
   number would weaken a self-validator; round-then-compare admits exactly the
   legal rounding and nothing more, so it is strictly stronger than both the
   current test and any widened band. Bill 0 certificates (the probe is in no
   `IMPL_DEPS`). Owned by `t306-probe-returned-an-undeclared-combination-and-
   v2-false-alarms` (OPEN, DUE 10-26).
3. **`NE.01` — CONFIRMING YOUR CHOICE, NOT RE-ORDERING IT.** `run blocked`
   this sitting ranks `NE.01` first (FAIL, frees 7, blocks 8), and your
   06:07 slot has attempt 4 in flight. That is the right call and this item
   exists only so no later desk reads this page and redirects you off it.
4. **Port `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok` into `T2.05`,
   and report `mse_nonlinear` + `nonlinear_headroom`** (DUE 10-25, bill 0
   certificates; carried forward unchanged). Do NOT gate on the headroom
   number on this pass — report it. Then `T2.05`'s 43-day-old FAIL becomes
   attributable: headroom large ⇒ the venue has structure the WorldModel
   missed; headroom ≈ 0 ⇒ the VENUE is the defect, and that repair is
   world-touching so the bundling rule applies.
5. **Choose and execute one option from
   `regate-bought-nine-certificates-from-its-own-uncommitted-tree`** (DUE
   10-24, bill 0; carried forward). This desk recommends **(i)+(iii)**: refuse
   to run on a dirty tree, and make `--list` print the stamp it will buy under.
6. **`T0.22`'s guard comment says "All nine ran" while `N_PROPERTIES = 15`** —
   stale by six, no effect on the gate; ride it along with the next edit that
   bills `T0.22` anyway (carried forward).
7. `longrun_unbound` holds `T0.32` at FAIL over **six** specs, not ten:
   `['NE.02','T4.04','T4.05','T6.01','W1.01','W1.04']`, carried by
   `longrun-population-grew-six-ids-under-a-priced-row` (OPEN, DUE 10-25).
   **Note for any desk quoting `T0.32` off the ledger:** its live row is
   attempt 9 at `2026-10-17T15:14:08`, which PREDATES the 10-19 repair and
   still prints all ten ids. Re-derive before quoting it.

## FOR THE OWNER

**1. `D46` is unchanged and still open — carried forward, not re-asked.** The
reporting-floor recommendation stands exactly as quoted into `D46` (`class
conduct`, `decide_by 2026-10-17`, now past). Nothing in this sitting changes it.

**2. Routed this morning as `D52` (`class conduct`, `decide_by 2026-11-05`):
the organ that looks OUTSIDE this repo has lost two consecutive weeks to a
gate that was open for most of both of them.** Recommendation, quoted verbatim
into the entry:

> **Give the weekly scout a retry window instead of a single sample. The field
> watch tests the 90% stop once per week, and on both 2026-10-12 and
> 2026-10-19 that one instant landed in the exhausted tail of a usage week
> that reset within about a day — so two entire sweeps were lost to a gate
> that was open for five of the seven days that followed. The builder survives
> the identical gate because it samples hourly; the field watch is the only
> organ whose sample count equals its cadence, which makes a single unlucky
> instant cost a whole week of outside input.**

The default RETRIES within the week and **moves no threshold: the 90% stop,
`PACE_FLOOR` and `PACE_CAP` stay byte-unmoved and no attempt proceeds past a
refusal.** It is `class conduct` and would normally be executed at the desk; it
is routed because what gets re-timed is the invocation path of your own
2026-08-09 pause rule, and a desk re-timing how often that rule is CONSULTED
would be indistinguishable in the log from a desk loosening it. **Why it
matters beyond one organ:** the completeness audit exists because every other
organ measures us against our OWN documents — `docs/FIELD_WATCH.md` has been
unchanged since 2026-10-05, 17 days, and the last COMPLETE sweep was week 10.

**3. `D49` carried forward, unchanged (`class constitution`, `decide_by
2026-10-27`): a queue row whose only permitted dispositions are ALL forbidden
to its only permitted dispositioner.** `pl02-void-gate-quantifies-over-its-own-
nulls` is undischargeable as constituted — every repair on its menu is a
loosening, `1^13` forbids the builder, strengthen-only forbids this desk. This
desk still **recommends against its own default**. Not re-asked here.

**4. `D30`'s required pairing, re-counted at this sitting and not inherited.**
Dark **24 consecutive slots** (2026-10-21T06 → 2026-10-22T05), counted off
`/data/jack-logs/ladder.log` at this desk — **12× the 2× trigger** — every one
`PACING`, `0 failed`. **The streak BROKE this morning:** the 06:07 slot ran to
`rc=0` and has `NE.01` attempt 4 in flight, so the current streak is 0 and the
organ is live as this is written. Beside it, as D30 requires: **28.91 free
GPU-hours in `2026-W42` expire Saturday 2026-10-24.** The gate is the PACER,
not the 90% stop — `week:all models` **53% at 43% of the usage week against a
53% pace line**, i.e. sitting exactly ON the line, which is why the skipping
has not ended and will not until the ratio moves.
NO-DECISION: `D30`'s standing default discharged; nothing here to rule on.

**5. NO-DECISION: `D50`'s conduct default, executed and reported as ONE
attributed finding rather than four symptoms — which is the whole point of the
entry.** The 90% stop fired twice (`field_watch.log`: 98% on 2026-10-12, 92%
on 2026-10-19, *"all agents paused until the owner resumes"*). **Its dated
consequences, attributed:** two field-watch sweeps lost and `FIELD_WATCH.md`
17 days stale; `2026-W41`'s 30.0 GPU-hours closed wholly unbought. **And the
attribution's own limit, stated because `D50` exists to stop misattribution:**
the 24 dark slots in item 4 are NOT the 90% stop's — that meter reads 53%, far
under 90%. They belong to the pacer. Two different gates, two different
windows, and conflating them is the error `D50` was written against.
One correction to `D50`'s own evidence: it recorded `2026-W42` at `0.0 of
30.0`; the week now reads **1.0871 h spent**.

**6. `D48` carried forward, unchanged (`class conduct`, `decide_by
2026-11-01`).** The Sunday sweep found the defect class already guarded and
found the real one by accident: uneven guard propagation. Default is the
conservative branch — no exemption, no reader. Nothing is blocked on the
answer; the `T2.14 → T2.05` port is ordered as builder item 4 above.

**7. NO-DECISION: the week's hardest measurement, reported because a chief
scientist should not let it pass in silence — but there is nothing here to rule
on that items 2, 4 and 5 do not already put before you.** `experiments/
ledger.json` holds **160 results, 107 PASS, 35 FAIL, 17 VOID, 1 BLOCKED** —
and those five numbers are **byte-identical to the ledger 7 days ago and 14
days ago**, checked by reading the file out of git at each point rather than by
trusting a counter. In the same 7 days, **67 commits touched `ledger.json`**.
So the week's ledger traffic was re-certification — attempts rising under
unchanged verdicts — and **not one spec changed status in a fortnight.** The
pass rate is therefore flat at 66.9% of results (107 of 258 registered specs
demonstrated), and the Goodhart check this desk is required to run comes back
with the one answer it was not designed for: neither the numerator nor the
denominator moved. It is not drift and not Goodhart; it is a stall, and its
cause is already before you as the pacer (item 4) and the stop (item 5).
