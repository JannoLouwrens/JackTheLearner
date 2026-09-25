# REVIEW_QUEUE — findings routed to the weekly Review, with their staleness bills

Created 2026-08-24 per the 27th overseer audit's B2: "routed to Review" was a
phrase in commit messages and docstrings with no file, so the backlog was
invisible — nothing could print "3 routed, 0 acted on, oldest 4 days".

**The contract.** One `ROUTED:` row per finding, machine-greppable
(`grep '^ROUTED:' docs/REVIEW_QUEUE.md`). Fields, pipe-separated:
`ROUTED: <id> | <date routed> | <source commit> | <status>` followed by an
indented body: the one-line question and the **staleness bill** — the ledger
rows that acting on it would invalidate. The Review dispositions a row by
setting its status (`OPEN` → `ACTED <date> <commit>` or `DECLINED <date>
<why>`, or `HELD <date> <why>` for the bundling rule below); rows are never
deleted (T1.02 precedent: history stays). **`ACTED` means EXECUTED and must
name the executing commit** (≥7 hex chars) in its status text; a design that
exists but has not been executed is **`DISPOSITIONED <date> <where the design
lives>`**, which stays LIVE — it ages and can go STALE/OVERDUE like `OPEN` —
until somebody executes it and stamps `ACTED` with the commit. Added by Review
09-01 item 4 after `ACTED` on `recipe-sensitivity` (meaning only "design
written") read as closed for seven days and parked `UB.10`; enforced by
`experiments/review_queue.py` (`ACTED-WITHOUT-A-COMMIT`), gated as `T0.31`
P12. The bill is the price of the fix,
computed BEFORE deciding, so "fix the world" decisions are made with the
re-certification cost on the table, not discovered after.

**THE READER — added 2026-08-31, and it is why the two declarations below
exist.** For six days this file had rows and no reader: the 52nd audit found
that nothing in the repo could print *"7 OPEN, oldest 7 days, consumer last ran
2 days ago"*, after the Review's Sunday FULL run died at 11 minutes owing
`w0-too-shallow`'s design and that row's own dated promise passed in silence.

    /data/venvs/jackthelearner/bin/python -m experiments.run review-queue

`experiments/review_queue.py`, gated as `T0.31`. It reads DECLARED fields only —
never prose, because `champions.py` learned on `901f7fc` what a regex over prose
costs. Two optional indented body lines, in the `DECIDE:`/`COVERS:` idiom:

    DUE: <YYYY-MM-DD> | what is owed, and by whom
    BLOCKED-BY: <another row id> | what releases this hold

A live row past its `DUE:` is **OVERDUE**; an `OPEN` or `DISPOSITIONED` row
with no `DUE:` older than one full consumer cycle (8 days) is **STALE**. `HELD` buys exemption from
ageing and must pay for it with a `DUE:` or a `BLOCKED-BY:`, and a hold whose
blocker has reached a TERMINAL status (`ACTED`/`DECLINED` — the window it was
waiting for has opened) is itself a violation — otherwise the bundling
rule below becomes a place rows go to die.

> **Word corrected 2026-09-12 (Review, DAILY). This sentence read "a hold whose
> blocker has been *dispositioned*", in the pre-2026-09-01 sense of "disposed
> of", and it was correct the day it was written. It stopped being correct the
> moment `DISPOSITIONED` became a formal status OF THIS FILE, and the collision
> is not harmless: read literally it FORBIDS holding behind a `DISPOSITIONED`
> blocker, which is legal, which `review_queue.py:487` permits (`tgt["status"]
> in TERMINAL`, and `TERMINAL` is `ACTED`/`DECLINED` only), and which live rows
> now do. Found by walking into it — the correct disposition of
> `hr5-fixture-refuted` today was a hold, and this sentence said the hold would
> be a violation. The PROSE was wrong and the instrument was right, which is
> the opposite of the usual direction and worth recording for that reason
> alone. Second clarification, same paragraph, same cause: `HELD` buys
> exemption from **STALE**, never from **OVERDUE** — line 491 tests
> `due < today` for every LIVE status — so a hold carrying a stale `DUE:` reds
> out on schedule and must be re-armed in the open like any other row.**

Deleting a row, or dropping a `DUE:`
that went red, are each their own violation, computed against the previous
committed revision. **The escape hatch is re-arming in the open** — a new `DUE:`
with a reason, exactly as `decide_by` is re-armed in `DECISIONS_NEEDED.md`. What
a row must not be able to do is go quiet.

Two bills per world-touching row, and they differ: the SEMANTIC bill (rows
whose measured behaviour actually changes) and the MECHANICAL bill (every
PASS row whose `impl_sha` hashes the edited file — all of these go stale
loudly at the next `run status` regardless of semantics, and a dynamics
change has no `--doc-only` lane). As of 2026-08-24, 21 PASS certificates
cite `playground.py` in IMPL_DEPS: BA.01, LC.02, PG.1–PG.6, PG.8, PG.9,
PS.01–PS.03, SM.01, T2.03, T2.08, T2.20, T3.01, TA.01, VO.01, XL.00.

---

## THE BUNDLING RULE — added by the Review 2026-08-25 on first use of this file

Three of the four rows below edit `playground.py`, and each one bills the SAME
21 PASS certificates mechanically. **Paying that bill three times is three times
the re-certification cost for one world.** So: world-touching rows are acted on
in ONE edit window, not as they arrive. `ne01-occlusion-knife-edge` and
`water-apply-phantom-force` are both held for the window that `w0-too-shallow`'s
design opens — not because either is unimportant, but because merging them costs
21 re-runs instead of 63. If `w0-too-shallow` resolves toward a NEW world (W1)
rather than an edit to W0, both rows follow it there and the bill goes to zero,
which is the asymmetry the `w0-too-shallow` row already flagged as design input.

**The rule generalises, and it is the reason this file exists**: a backlog with
a computed bill can be SEQUENCED. A backlog scattered across commit messages
can only be serviced in arrival order, which is the most expensive order.

## THE 09-06 DOCKET — staggered 2026-09-02 (builder, 61st audit B2), in the
## open, while it was still a forecast

Eighteen OPEN rows carried `DUE: 2026-09-06` — one Sunday holding eighteen
dated promises, against a FULL run budgeted 40 m / 120 turns that also owes
Part 2, an anatomy audit and a completeness audit, and that has died four
consecutive Sundays. The audit's B2 ordered the clocks staggered with a stated
reason per row, or rows declined, BEFORE the dates went red — a mass re-arm on
09-07 being the deadline-that-moves failure the `DUE:` mechanism replaced.

Executed as follows. NOTHING is declined — every row will be taken; what moved
is WHEN, and the reasons are on each row (second `DUE:` line, last one wins;
the 09-06 lines stay in place as history). The organising distinction: the
bundling rule binds world EDITS to one edit window, not DECISIONS to one
sitting — a row that must be decided *in light of* Sunday's design belongs
AFTER Sunday, not beside it.

    09-06 (FULL)  w0-too-shallow, lt01-c2-body-cannot-rise,
                  d10-learning-gate-uses-two-different-denominators,
                  d10-learning-gate-sits-at-the-untrained-twin-level,
                  lc07-checkpoint-branch
                  — the coupled design bundle: the W0/W1 design, the
                  world-vs-body fork, the gate venue judgment, and the
                  checkpoint-vs-venue call. Plus the ACTED me11 row's owed
                  family disposition (not re-armed; ACTED rows are not mine
                  to touch).
    09-07 (DAILY) sm03-heldout-split-saturated, t310-anticorrelated-gates,
                  pl02-dependency-on-pl00-verdict-vs-table,
                  champions-language-grounding-arena
                  — four self-contained decisions with no coupling to the
                  W0/W1 design.
    09-08 (DAILY) ub10-seed-fragility-and-saturated-battery,
                  d10-successor-rerun-under-adopted-gate,
                  lg10-mouth-fidelity-vs-freedom,
                  cpu48h-class-self-forecloses-the-day-meter (added 09-04,
                  68th audit B6 — deliberately NOT 09-06 per its B7)
                  (t309-control-clears-the-claims-own-margin was already
                  here, deliberately off the pile)
                  — consequence-stamps of Sunday's decisions plus the
                  unison redesign, which has its own lineage.
    09-09 (DAILY) ba03-null-saturates-the-horizon, sh02-null-saturation,
                  t306-matched-magnitude-noise-buys-coverage
                  — the three venue repair-arm picks, decided in light of
                  the 09-06 design (if it resolves toward W1, the arm
                  choices change).
    09-10 (DAILY) reparenting-the-welded-fifteen,
                  goal-cites-four-specs-that-resolve-to-corpses
                  — registry surgery, downstream of the design and of
                  lc07-checkpoint-branch respectively.
    09-11 (DAILY) five-commitments-are-claim-dead-behind-foreclosures
                  — the most downstream row on the board: successor specs
                  need the design (09-06), the arm picks (09-09) and the
                  re-parenting (09-10) as inputs.

If a daily run cannot carry its day's rows, re-arm the slipped rows
individually, in the open, with the slip as the reason — do not re-pile them
onto a Sunday.

### THE STAGGER DID NOT HOLD — second pass 2026-09-04 (builder, 68th audit B7),
### and the instrument that would have prevented it now exists

**The measurement.** The 09-02 stagger above took `2026-09-06` from **18 live
rows to 5**. By 09-04 it read **8** again. Nothing was smuggled: three rows
were routed in between (`cross-organ-doc-race-voids-certificates`,
`hr5-fixture-refuted`, `w0-kills-a-forager-by-integrity-at-25-minutes`) and
each named its reason for choosing Sunday. The defect is not in any of the
three — it is that **each router picked a date with nothing telling it the date
was already full**, so the file could only report the pile after it re-formed.
A stagger that has to be re-done every two days by hand is not a repair.

**Scope of this pass, stated so it is not read as re-litigation.** Only the
three rows routed AFTER 09-02 were considered. The five-row coupled bundle the
09-02 pass deliberately chose is left exactly as it is: `w0-too-shallow` is the
decision, `lt01-c2-body-cannot-rise` is the world-vs-body fork the design
cannot be made without, the two `d10-*` gate rows have money and a date
attached (W36 opens 09-06 with 30 free GPU-hours, and
`d10-successor-rerun-under-adopted-gate` is DUE 09-08 behind them), and
`lc07-checkpoint-branch` was placed in the bundle two days ago on evidence that
has not changed. Re-deciding a two-day-old deliberate act with no new evidence
is the failure this file exists to prevent.

    RE-ARMED to 09-09  hr5-fixture-refuted
                       — decided IN LIGHT OF Sunday's design; the 09-02 rule
                         puts such rows after Sunday, and the edit window (and
                         its 21-certificate bill) is unchanged.
    RE-ARMED to 09-09  w0-kills-a-forager-by-integrity-at-25-minutes
                       — an INPUT to Sunday's design, not a decision beside
                         it; taking the row's own written offer. Its evidence
                         is on the desk from Sunday regardless.
    LEFT ON 09-06      cross-organ-doc-race-voids-certificates
                       — it chose the full day knowingly and its reason is
                         stronger than pile-avoidance: the trap re-arms every
                         night an audit runs and each trip re-bills four
                         certificates. Overriding a stated reason with a
                         scheduling preference would be the worse move.

**09-06 goes 8 -> 6; 09-09 goes 3 -> 5. BOTH ARE STILL AMBER, and saying so is
the point.** With 31 live rows against a measured capacity of 1/cycle, no
arrangement of dates is honest — re-dating moves promises between piles, it
does not create capacity. `next_free_due` reads **2026-09-12**: every day
between now and then already carries a promise. That is a fact about the desk,
not about the calendar, and the only repairs that touch it are ACT and DECLINE.

**What changed so this does not need a third pass.** `review_queue.py` now
reports the ACT rather than only the symptom, gated as `T0.31` P14:

- `piled_on` — every live row whose `DUE:` named a date that ALREADY carried
  the measured capacity when the row was routed. It reads **17 of 31** today,
  and its three worst entries are exactly the three rows above, at 5 apiece —
  the instrument found this pass's triage before the prose did.
- `next_free_due` — the first future date carrying no promise yet, so the next
  router has a mechanical answer instead of defaulting onto Sunday.

Both are METRICS and neither is a violation, deliberately: several piled rows
had good reasons, and a gate at zero would forbid a legal move. Under-counting
is also deliberate — a re-armed row is timestamped by its `ROUTED:` date, which
can only make `piled_on` too small, never too large.

ROUTED: recipe-sensitivity | 2026-08-20 | probe jack-ladder-1787249890 | ACTED 2026-09-01 in 15eb02e (builder EXECUTED the 08-25 disposition — grid, selection, SCORED-AND-INELIGIBLE verdict all in ub_10_fusion_bakeoff.py; UB.10 unparked. The 08-25 'ACTED' meant only 'design written' and parked this row's spec for 7 days — the two-meaning token was Review 09-01 FOR THE BUILDER item 4, repaired the same day: that state is now DISPOSITIONED, which ages)
    Question: no single uniform training recipe trains all six matched-param
    UB.10 arms (warmup@1e-3 leaves A2/A3 dead; 3e-4 fixes A3 but breaks A4);
    A2 learned its marginals under NO tested recipe. Per-arm recipes, arm
    redesign, or drop the uniform-recipe constraint? UB.10 was PARKED on
    this; UNPARKED 2026-09-01.
    Full record: PROBE RECORD in ub_10_fusion_bakeoff.py; 23rd audit B3.
    Staleness bill: NONE — UB.10 has no PASS and no certificate cites its
    arms. This is the cheapest row to act on.
    DISPOSITION 2026-08-25 (Review): **matched TUNING BUDGET, not matched
    hyperparameters** — and it is STRICTLY HARDER than what it replaces.
    Dropping the uniform-recipe constraint destroys the comparison (an arm
    would win by getting a better LR); keeping it is what left A2 dead. So
    every arm gets the IDENTICAL pre-registered LR grid, the same number of
    trials, selected by the same pre-registered criterion, all declared before
    any arm runs. Cost rises from N to N x K runs — that is the point, the
    budget is what is matched. The gate that makes it honest already exists:
    per the 23rd audit B1, `uni_marginal_ok`/`uni_learn_ok` mean a dead arm can
    no longer read as a clean 0.5, so "did this arm's recipe train it" is
    machine-checkable per arm. An arm that clears `uni_learn_ok` NOWHERE on the
    grid is recorded SCORED-AND-INELIGIBLE (SYSTEM.md's new language, 0345f0d)
    — measured on the same ruler, kept as a standing challenger, not seated and
    not silently a 0.5. Full reasoning in PROGRESS.md.
    EXECUTED 2026-09-01 (builder): the grid is K=5 (base 1e-3 / warmup
    1e-3+10% / lolr 3e-4 / lolr_warm 3e-4+10% / xlolr 1e-4), declared in the
    test file before any grid trial; selection is `_select_recipes` — first
    eligible in grid order on the arm-local conjuncts, provably blind to the
    claim metric (sabotage fixture in smoke); ineligible arms run at base,
    recorded, excluded from winner/conjuncts; A0-ineligible and
    zero-eligible-trunk are VOID floors in _check. run() REFUSES until the
    grid pilot (seed 90, one kernel, ~0.7 h P100) is harvested and SELECTED
    committed. Pilot dispatch deliberately queued behind D1.0 (Review 09-01
    item 3). Smoke green incl. selection fixtures.

ROUTED: ne01-occlusion-knife-edge | 2026-08-24 | 5063144 | HELD 2026-08-25 for the world-edit window (see THE BUNDLING RULE)
    BLOCKED-BY: w1-world-edit-window | RE-POINTED 2026-09-06 (Review FULL)
        from `w0-too-shallow`, which was DISPOSITIONED this morning — a hold
        whose blocker has been dispositioned is itself a violation, and this
        hold's substance is unchanged: it is waiting for the world-edit
        sitting, which is now its own row with its own DUE 2026-09-13. If W1
        is built this follows it there and the bill goes to zero, exactly as
        the original line said.
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: the sitting this hold waits on did not
        happen. The 2026-09-13 FULL sat at 06:37 and did not take up W1, so
        `w1-world-edit-window` goes OVERDUE at 00:00 tonight and this hold ages
        another cycle behind it. The hold itself remains CORRECT — its blocker
        is live, dated and now visibly broken, which is what a hold with a clock
        is for. Full record on `w1-world-edit-window`.
    Question: the 9-ray head-cone occlusion law yields knife-edged ninths a
    sleeping ragdoll cannot hold — the statically-found 0.5–0.9 band realises
    occ 0.337±0.467 overnight (slides out and freezes, or seals and cooks).
    Grade the cone, widen the band geometry, or damp the sleeper? All three
    are runnable arms — this is a redesign BAKEOFF, not an argument.
    Full record: FAIL RECORD in ne_01_nobody_survives_by_accident.py.
    Staleness bill: SEMANTIC — no PASS certificate yet cites the occlusion
    law (NE.01 itself is FAIL); MECHANICAL — the 21 playground.py rows above
    if the fix edits playground.py.

ROUTED: water-apply-phantom-force | 2026-08-24 | a210b34 | HELD 2026-08-25 for the world-edit window (see THE BUNDLING RULE)
    BLOCKED-BY: w1-world-edit-window | RE-POINTED 2026-09-06 (Review FULL)
        from `w0-too-shallow` for the same reason as the row above; the same
        world-edit window, still paying the 21-certificate mechanical bill
        once instead of three times.
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: the 2026-09-13 FULL did not take up W1, so
        the window this hold rides did not open and `w1-world-edit-window` goes
        OVERDUE at 00:00 tonight. The hold stands and its substance is unchanged.
        Full record on `w1-world-edit-window`.
    Question: Water.apply (playground.py:627) writes a body's xfrc row only
    while it is in the pool, so any body that exits keeps its last buoyancy/
    drag force forever — a phantom force in live dynamics, found by DP.05's
    fidelity pilot (snapshot/restore made it visible). Fix the world?
    Full record: LESSONS d1bc3d1; DP.05 PILOT RECORD.
    Staleness bill: SEMANTIC (worlds with a pool, per the 27th audit) —
    BA.01, LC.02, PS.02, PS.03, XL.00 — 5 PASS certificates; MECHANICAL —
    all 21 playground.py rows above.

ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | DISPOSITIONED 2026-09-06 (Review FULL — the W1 SPEC-FAMILY DESIGN is published below, five specs W1.00–W1.04 with their falsifiers, controls and ordering; the builder registers them, nothing is registered by this row and no world is edited by it. GOAL.md untouched, no spec re-parented — those stay the owner's, per D21)
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: the Sunday FULL this row's `DUE: 2026-09-13`
        names sat at 06:37 today and did not take up W1. `W1.01`, `W1.03` and
        `W1.04` are still NOT REGISTERED, as the ORDERED line below reports.
        Full record on `w1-world-edit-window`.
    ORDERED: W1.00 W1.01 W1.02 W1.03 W1.04 | the five specs this disposition
        commissioned. Recorded 2026-09-06 23:xx (builder), BACKFILLED under
        the 79th-audit guard the same day the first return refuted the
        commission: W1.00 FAILed its "immaterial" branch at 10:30 — seven of
        the eight Pile A margins move by 0.022–0.084 std under the stronger
        null, so the two-pile reading does NOT dissolve them. The pair now
        prints in `run review-queue` instead of living only in the ledger.
    D21 DEFAULT FIRED 2026-09-06 00:1x UTC (builder; record in
        DECISIONS_RESOLVED.md, committed before the 06:37 FULL on purpose —
        the same-day race the 72nd audit named): today's FULL takes the W1
        design as its FIRST DESIGN item, ahead of Part 2, BEHIND the two
        d10-* gate rows, which keep the head of the docket, and publishes a
        W1 spec-family design as a routed disposition. GOAL.md is not
        touched and no spec is re-parented — those stay the owner's. This
        is the armed default re-ordering an already-dated item, NOT a third
        hand-stagger; the 09-02 bundle's dates all stand.
    DUE: 2026-09-06 | the W0/W1 design, owed by the Review's Sunday FULL run.
        RE-ARMED 2026-08-31 (DAILY) from 2026-08-30, in the open, with two
        reasons and one of them is my own fault:

        (1) THE DESIGN IS SEQUENCED BEHIND A DIAGNOSTIC THAT WAS NEVER MADE
        RUNNABLE. On 2026-08-25 the Review accepted field-watch `wk4-N3` and
        ordered it *"BEFORE any W1 world redesign"* — the cheap attack on a
        shared confound across instruments that were all run by this project
        on this world. That order was written as prose inside
        `INTEGRATION_QUEUE.md`. It has no spec id, no cost class, no row the
        builder's top-down read can consume, and in six days no iteration
        has touched it. So the 08-30 promise was undeliverable by any Review,
        alive or dead: its stated input had never been ordered into existence.
        Fixed today — the diagnostic is now INTEGRATION_QUEUE entry `W0.DIAG`
        and priority 1 in `scripts/ladder_prompt.md`.

        (2) THE FULL RUN HAS NEVER COMPLETED, AND THE CAUSE IS MECHANICAL,
        NOT ACCIDENTAL. Four consecutive Sundays, four deaths. The 08-30 run
        died at `Reached max turns (60)` after 11 minutes. Today's DAILY run
        is budgeted `20m / 60 turns` — the SAME budget — while a FULL run
        additionally owes Part 2 (8–12 spec re-examinations, each a read plus
        a re-run), an anatomy audit and a completeness audit. The Review is
        not failing at its job; it is being asked to do a Sunday's work in a
        Tuesday's budget, and no organ watches for that. Escalated to the
        owner in `docs/PROGRESS.md` § FOR THE OWNER. **A fifth Sunday bet is
        only honest if that is fixed** — if it is not fixed by 09-06, the
        correct disposition on 09-06 is to split this row, not to re-arm it
        a second time. Week 3's rule binds me too: a third deferral is a lie.

        MIGRATED FROM PROSE, by hand, 2026-08-31 — the date above was already
        written in this row's status on 2026-08-25 and was read by nobody. The
        run that owed it started 2026-08-30T06:37, died on `Reached max turns`
        at 06:48 having written nothing, and this row stayed silent for a day.
        That silence is the scar `experiments/review_queue.py` exists to end:
        the promise is not new, only the reader is. Two holds
        (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`) and four
        gate-provisional specs are behind it. Re-arm it with a new DUE: and a
        reason if the next FULL run cannot carry it; do not delete this line.
    DUE: 2026-10-01 | RE-DATED 2026-09-24 (Review DAILY). The 2026-09-23 date BROKE — third break for this row. What is NEW and what is not: the 09-14 re-date blamed the pile and the 09-15-era re-dates blamed the builder's blackout, and BOTH causes are gone this morning — `week:all models` 19%, six consecutive `rc=0` builder slots, and `review-queue` reports 0 fresh work on the board for a sixth day. **The row is DISPOSITIONED and what it owes is EXECUTION: the registration of `W1.01`, `W1.03` and `W1.04`.** `review-queue`'s own ORDERED MEASUREMENTS block confirms it row by row — W1.00 -> FAIL 09-06, W1.02 -> PASS 09-06, and the other three NOT REGISTERED, seventeen days after the design was published in `9eddb52`. **This desk will not hand that execution to the idle builder today, and the reason is a rule rather than a preference:** the 09-23 sitting's own addendum (`d9f568f`) recommended to the owner that this desk be held to registering the three specs ITSELF, `D33` is the open entry on who authors this unit, and ordering it onto the builder's board while that entry sits unanswered would pre-empt the ruling I asked for — `D22` binds the desk that wants the carve-out exactly as hard as it binds anyone else. So the date moves and the authorship does not. Dated 2026-10-01, which is `review-queue`'s own printed "next date with room under the measured capacity" (2 promised against 6), and deliberately AFTER `w1-world-edit-window`'s 09-27 Sunday date, because registering these three is downstream of the world-edit window they run in. ORIGINAL TEXT FOLLOWS, unchanged. | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. The DESIGN IS DELIVERED and this row is DISPOSITIONED: what this date owes is EXECUTION by the builder, not a decision by this desk. Dated where a builder slot can plausibly reach it rather than onto the Review's own calendar. | DELIVERED AND SUPERSEDED. The design owed on 2026-09-06
        was written and committed by the Review's FULL run that morning (the
        W1 DESIGN block below, five specs with falsifiers, controls and an
        ordering) — the fifth Sunday bet, and the first one that paid. What
        this row now owes is REGISTRATION of `W1.00`–`W1.04` from that design,
        and it is owed by the BUILDER, not by me. Dated a week out on purpose:
        `W1.00` and `W1.02` are measurable on W0 AS BUILT and carry no
        staleness bill, so they register first; `W1.01`/`W1.03`/`W1.04` wait on
        `w1-world-edit-window`. This is not a third hand-stagger of the design
        promise — that promise is discharged and its artifact is below.
        REGISTRATION, FIRST HALF DONE 2026-09-06 ~09:3x (builder): `W1.00`
        and `W1.02` registered from the design as published (registry
        242→244, both RUNNABLE at registration — deps W0.DIAG/PS.01 both
        PASS — so `unreachable` unmoved at its 95 floor; `coverage` prints
        both as *fillable today*, the first fresh dispatches since SO.08).
        The step-1 cross-check found NO refutation and four binding
        constraints, all designed into the specs: the colored null imports
        W0.DIAG's erf-AR(1) construction; W1.00 reads LC.03's recorded
        artifacts and runs zero lives at its envelope; W1.02's synthetic
        MIN_GAIN injection is a declared arithmetic transform at the outcome
        level, never a Δe write (PURPOSE_AND_SCAFFOLDING §5/G-A would ERROR
        a physics-level injection); censoring conventions imported from
        NE.08 (cause tags, censoring rate, no uncensored-only means).
        `W1.01`/`W1.03`/`W1.04` deliberately NOT registered — they stay on
        `w1-world-edit-window` per the ordering above.
        `W1.04` / CONJUNCT (c) — READ AS DISCHARGED ON THE DESIGN SIDE AND
        HELD ON THE REGISTRATION SIDE (builder, 2026-09-12 ~23:3x). The item
        *"`W1.04` still gains conjunct (c) before you register it"* has been
        carried verbatim by three consecutive audits (89th/90th B6) and by
        `PROGRESS.md` FOR THE BUILDER on 09-10, 09-11 and 09-12, reading each
        time as unexecuted builder work. It is not. **Conjunct (c) was written
        by the Review itself on 2026-09-10** — the STRENGTHENED block in the
        `W1.04` design above, committed as the CONSEQUENCE half of this row's
        own reading (`w0-kills-a-forager-by-integrity-at-25-minutes`, ACTED
        2026-09-10, `1a0e413`, which says in terms *"Written into the `W1.04`
        block on `w0-too-shallow`, which is what the builder registers from"*).
        Nothing about (c) is owed by the builder; it is a CONSTRAINT ON a
        registration, not an order to perform one.
        AND THE REGISTRATION IT CONSTRAINS IS HELD BY THIS ROW'S OWN
        ORDERING, for a reason that is arithmetic rather than deference:
        `W1.04`'s conjuncts quantify over *"every registered W1 claim"*, and
        that set is today `{W1.00, W1.02}` — of which `W1.00` runs **zero
        lives** by this row's own cross-check three lines up, so conjunct (c)
        has nothing to measure on it. Registering `W1.04` before `W1.01` and
        `W1.03` exist therefore binds it to a set that is missing every spec
        it was designed to size, and manufactures precisely the trivial PASS
        that `W1.04`'s own falsifier says should retire it (*"if the measured
        time-to-consequence ... is already under a third of the horizon
        everywhere, this spec PASSes trivially and should be retired rather
        than kept as decoration"*). The hold is not a deferral of work; it is
        the spec's own falsifier being obeyed in advance.
        SO THE ITEM IS NOT BUILDER-ACTIONABLE UNTIL `w1-world-edit-window`
        (DUE 2026-09-13) lets `W1.03`/`W1.01` be registered. Recorded here
        rather than argued in a journal line so the next audit that reaches
        for it reads the reason beside the order, instead of carrying it a
        fourth time. If this reading is wrong, the repair is one sentence on
        this row — not a fourth restatement of B6.
    Question: three independent instruments now measure W0 as too shallow to
    reward the capabilities the ladder certifies — LC.03's darkroom control
    (passivity prospers), LC.03 v2 (one learner in five), DP.05 (lookahead
    buys 13–21 s under the 20 s margin; deeper lookahead buys LESS; the best
    reactive policy is "starve at the resting ceiling"). The pre-registered
    routing: traps, delays, irreversibility — the DP.00 preconditions GOAL.md
    names — before any dual-process claim; BO.01 does not run. COUPLED to
    D10 branch (b) in DECISIONS_NEEDED.md (owner) — the Review designs, the
    owner sequences.
    UPDATE 2026-09-06 ~19:2x (builder, 79th audit item 1): THE PILE A HALF OF
        THIS DISPOSITION IS FALSIFIED BY ITS OWN ORDERED INSTRUMENT. `W1.00`
        — registered from this row's design at ~09:3x, run at 10:30:12 —
        recorded FAIL on its pre-registered immaterial branch: the honest
        null exists but rescues nothing. Full arithmetic and consequence on
        `w100-honest-null-does-not-rescue-pile-a` immediately below (DUE
        2026-09-15). Pile B — the world is genuinely shallow — is where the
        evidence now points. The disposition's W1 design and registrations
        stand; it is the Pile A *diagnosis* ("the repair is in our
        instruments") that did not survive four hours.

ROUTED: w100-honest-null-does-not-rescue-pile-a | 2026-09-06 | 79th-audit-item-1 (builder; finding §3.1) | ACTED 2026-09-16 (Review DAILY, executing commit `d521384` — the carry-back was already discharged on the `w0-too-shallow` row itself on 09-06; what this row still owed was the ORDERING consequence, and it is delivered below. The W1 ordering stops being contingent and becomes unconditional. See ANSWER below)
    DUE: 2026-09-15 | first future date carrying no promise yet per
        `review-queue`'s own `next_free_due` (09-13 already carries 10 rows
        against a measured capacity of 1/cycle). Owed by the Review: carry
        `W1.00`'s result back into the `w0-too-shallow` disposition it
        contradicts, and say what the W1 ordering means now that Pile A does
        not dissolve.
    EVIDENCE, from the committed attempt-1 FAIL row (ran 2026-09-06T10:30:12,
    commit 902ee21, 3 seeds, branch "immaterial: repeat outscores white but
    no recorded margin moves by more than its own std"):
    - A stronger null DOES exist: `gain_repeat` 3.13 (±3.97) vs `gain_white`
      −0.019 (±0.69). The first conjunct held; the Pile A prediction was that
      re-scoring under it would dissolve the shallowness findings.
    - It dissolves nothing. Shift-over-own-std across the eight recorded
      Pile A margins: wk5_coverage 0.022, ppo_needs 0.029, ppo_lp 0.036,
      darkroom 0.037, dreamer_xs 0.038, wm_efe 0.038, wm_latent 0.084 —
      seven of eight, none within a factor of ten of the 1.0 gate.
    - The eighth is "CANNOT TELL", never "did not move": `dwell` (T3.06)
      reads shift_ratio 16.2 but `fired_dwell` 0.0 because the spec's own
      guard excludes it — the dw channel's noise floor `f_dw` 0.0082 exceeds
      that margin's own std 0.0062, so the shift is unreadable. Guard
      verified in source (`w1_00_null_is_strongest_nonlearner.py:452-456`)
      by the 79th audit, not trusted from the branch string.
    CONSEQUENCE: the Pile A / Pile B split published in the 09-06 disposition
    above is falsified for Pile A on its own pre-registered test. Seven
    shallowness findings stand under the honest null; the world question tips
    toward Pile B — W0 is genuinely shallow. `W1.01` (passivity dies), the
    spec that would settle W0's headroom directly, remains UNREGISTERED
    behind `w1-world-edit-window` (DUE 09-13); this row is evidence INPUT to
    that sequencing, not a request to jump it.
    NOT A SPEC REPAIR: `W1.00` fired a pre-registered branch and that is a
    result — do not re-run it for a different branch. It is attempt 1 and
    unsettled, so `fail_unowned` will never surface it; this row is the
    owner it would otherwise never get. The separate methodological finding
    from the same run (`selection_divergence` 1.0 — historical null-picking
    by the claim arm's margin selects a different process than picking by
    the null's own outcome, all three channels) is 79th-audit item 2, a
    scoping question over past certificates, deliberately NOT bundled here.
    ANSWER, 2026-09-16 (Review DAILY) — **the carry-back was already done;
        the ORDERING consequence was not, and it is this: the W1 ordering
        stops being CONTINGENT and becomes UNCONDITIONAL.**

        **On the carry-back, stated so the row is not credited twice.** The
        `UPDATE 2026-09-06 ~19:2x` block on `w0-too-shallow` already carries
        `W1.00`'s result into that disposition, in terms — *"Pile B is where
        the evidence now points. The disposition's W1 design and
        registrations stand; it is the Pile A diagnosis that did not survive
        four hours."* That half was discharged the day this row was written
        and nothing here adds to it.

        **The ordering, which is what was actually owed.** Before `W1.00`,
        the W1 programme had two live readings and the ordering was hedged
        between them: if Pile A dissolved — if the eight shallowness margins
        were an artefact of a weak null — then W0 was fine, the repair was in
        our scoring, and W1 was optional. `W1.00` was the pre-registered test
        of that and it fired the immaterial branch. **So the hedge is gone.**
        `W1.01` / `W1.03` / `W1.04` are no longer the Pile B *contingency*;
        they are the only remaining branch, and their order follows from
        their own content rather than from a preference:

        1. **`W1.01` (passivity dies)** — first, because it measures W0's
           headroom DIRECTLY and is the one spec that can still falsify the
           whole programme cheaply. If a do-nothing agent keeps prospering in
           the new venue, the world edit did not work and nothing downstream
           is worth registering.
        2. **`W1.03` (traps, delays, irreversibility)** — second: the
           `DP.00` preconditions `GOAL.md` names, and the substance of the
           edit `W1.01` is testing for.
        3. **`W1.04` (horizon sizing)** — last, and ALREADY held there by its
           own falsifier (`1a0e413`, conjunct (c)): it quantifies over *"every
           registered W1 claim"* and registering it into an empty set
           manufactures the trivial PASS it says should retire it.

        **The consequence for the docket, and it is the reason this answer is
        worth writing on a dark morning.** `w1-world-edit-window` (DUE
        2026-09-18) is the gate that lets `W1.01`/`W1.03` be registered at
        all. With Pile A closed, that row is no longer one design question
        among several — **it is the gate on the project's only remaining
        world branch**, and the registration behind it is BUILDER work. That
        is a second, independent instrument-visible cost of the builder being
        dark, and it is named in this morning's `PROGRESS.md` rather than
        left to be rediscovered.

        **The honest limit, stated rather than rounded.** Pile A is closed on
        **seven** of its eight margins, not eight. The eighth (`dwell`,
        `T3.06`) reads CANNOT TELL and never "did not move": the dw channel's
        noise floor `f_dw` 0.0082 exceeds that margin's own std 0.0062, so
        the spec's own guard excludes it. Nothing in this answer depends on
        the eighth, and no reader should later find it quietly counted.

        **Bills.** SEMANTIC none — `W1.00` fired a pre-registered branch and
        is NOT re-run. MECHANICAL none. No spec, threshold or registration is
        touched by this answer; it orders work that other rows own.

    SCOPED 2026-09-06 ~20:1x (builder, 79th audit item 2 — the answer, not a
    new promise): **no recorded certificate picked its null in the
    claim-favoring direction.** Every registered spec that reduces multiple
    candidate nulls does so margin-MINIMIZING — the claim must beat the
    strongest — which is exactly the pick `W1.00` labels `by_claim`, taken
    as the hardest bar rather than as a fit: `DP.00` (PASS a3,
    `max(react_greedy, react_persist)`, called "strengthened null" in
    source), `DP.05` (FAIL a1, same pattern in W0), `T2.05` (FAIL a4,
    `min(mse_persist, mse_mean)` per seed — redesigned TO this after v1's
    control exposed the weak persistence null), `T2.08` (PASS a4, margins vs
    `max(random, eps0)` on both channels), `UB.9` (PASS a3, conjunctive:
    beat max_m unimodal AND ensemble), `ME.11` (FAIL a1, `max(recalls)` is
    generous to the family and the family still failed), `LG.03` (VOID a1,
    null = max over {k-NN, ridge} selected by the null's OWN calibration
    liveness, strengthen-only ratchet declared in its docstring). `ME.9`
    (PASS a4) selects nothing: margin vs the pooled MEAN, pre-registered,
    plus a cap on the STRONGEST trivial arm. The Pile A shallowness sources
    (`LC.03`, `BA.03`, `T3.06`, wk5) — where strongest-null selection WOULD
    be claim-favoring, since their claims invert — used the single
    registered white null with no selection, and `W1.00` measured the
    white->strongest upgrade for them directly (<= 0.084 of own std).
    So `selection_divergence` = 1.0 undermines no recorded PASS; it binds
    PROSPECTIVELY through `W1.00`'s kills clause (W-venue specs registered
    after its row state their null as the venue-outcome-selected process).
    Method and blind spot, stated: structural sweep (reduction idioms
    max/min/argmax/sorted near null/best/strongest over all 155 test files,
    plus every registry `null_baseline` string with selection language),
    each hit read in source; a spec running a second null it neither
    declares nor reduces would evade this grep, but would also be a
    spec/impl mismatch, a different violation with its own instruments.
    Adjacent observation, out of scope here: `BA.02`/`BA.03` argmax the
    CLAIM arm (`best_trained`) against a fixed null — the mirror direction,
    guarded by their sigma gates, and both rows are currently VOID, so no
    live certificate rests on it.

# ============================================================================
# THE W1 DESIGN (Review FULL, 2026-09-06). Five specs, ordered. Design only.
# ============================================================================

**FIRST, THE DIAGNOSIS CHANGED ON 2026-08-31 AND NOBODY RE-READ THIS ROW.**
`W0.DIAG` — the cheap falsifier this row itself sequenced ahead of the design
on 2026-08-25, and which took six days to become a spec because it was written
as prose — **PASSED on 2026-08-31** (attempt 3, 3 seeds). Its claim branch is
recorded as *"correlation buys life through food"*, and the numbers are not
marginal: a random policy whose per-decision marginal action distribution is
**identical** to the `random` null every shallowness instrument used, differing
only in being temporally CORRELATED, records `gain_up` **12.12 ± 1.20** against
the stationary null's `gain_random` **0.0095 ± 0.39`, and mean life **52.53**
against **41.23**. `eats_up` 1.0 against `eats_random` 0.33.

So W0 is **not** a world in which nothing is available to be gained. It is a
world in which *sustained directed movement is worth twelve units of life and
our standard null cannot produce sustained directed movement.* That single
result splits this row's eleven instruments into two piles that need opposite
repairs, and lumping them was the error I am correcting:

  **Pile A — UNDER-NULLED, and the repair is in OUR instruments, not the
  world.** Every reading of the form *"the null does as well as the learner"*
  was taken against a stationary white-noise process that `W0.DIAG` now shows
  is strictly weaker than a same-marginal correlated one. That does not make
  those findings wrong — it makes them **too kind to the learners**, because
  the honest null is harder. Members: `LC.03`'s darkroom control, `LC.03` v2's
  one-learner-in-five, field watch wk5 (*"a random policy covers W0 as well as
  the curious arm"*), `T3.06`'s `random_dwell_worst_life`.

  **Pile B — GENUINELY SHALLOW, and the repair is the world.** Readings where
  the null **saturates the outcome** or the outcome **cannot resolve a
  difference at all** — no stronger null rescues these, because there is
  nothing above the ceiling to reach. Members: `SH.02` (twin, privileged
  oracle and both-cosmetic control all exactly **1.0000** against
  `HEADROOM_MAX` 0.85 — the null holds the roof completely), `SH.01`'s
  `ORACLE_CANNOT`, `DP.04` (**0 of 3072** lives ended between the two caps; 21
  distinct lifespans; quantum 6.25 steps against `MIN_GAIN` 5.0 — the
  measurement quantum is LARGER than the effect it must detect), `DP.05`
  (deeper lookahead buys LESS; best reactive policy is *starve at the resting
  ceiling*), `BA.03` (blind twin holds 98.9% of the 12.0 s horizon),
  `LF.01` (the forager dies of **integrity** at ~25 min, not starvation),
  `LG.03` (*"this world does not admit language-necessary commands at this
  horizon"*), `SO.07` (the recording worlds cannot produce the behaviour).

**THE FIVE SPECS.** W1.00 and W1.02 are Pile A / measurement and run on **W0 as
built — no world edit, no staleness bill.** W1.01, W1.03 and W1.04 are Pile B
and need the world-edit window, which is now its own row
(`w1-world-edit-window`) so the two holds behind this one have a live blocker.

  **`W1.00` — The null is the strongest process that has not learned.**
  *Claim:* for any W-venue spec, the registered null is the best-scoring member
  of {stationary white, temporally-correlated colored noise at the `W0.DIAG`
  schedule, repeat-action} **selected on the NULL's own outcome, never on the
  claim arm's** — and re-scoring the Pile A findings against it changes at
  least one recorded margin by more than its own std. *Falsifier:* if the
  correlated null does NOT outscore the white null on the venue's own outcome,
  `W0.DIAG`'s result does not generalise past food-seeking and Pile A
  dissolves. *Control:* the selection rule must be run against a claim arm too
  — if picking the best null by the null's own score ever selects a DIFFERENT
  process than picking it by the claim arm's margin, the old practice was
  fitting the null to the claim, and that must be printed. *Why this is a
  strengthening and not a re-litigation:* it can only ever RAISE a null.
  Nothing that passed can pass more easily. Cost: cpu, re-scores existing
  artifacts. **This one is cheap and it is first.**

  **`W1.01` — Passivity dies.** *Claim:* in the venue, a do-nothing agent does
  NOT hold the outcome roof: there is measurable headroom between a passive
  arm and a hand-coded competent oracle, `passive <= FLOOR < ROOF <= oracle`,
  with the gap wider than 3× the W1.02 quantum, on every seed. *This is the
  precondition `SH.02`'s pilot falsified* — it measured twin, oracle AND both
  cosmetic controls at exactly 1.0000, i.e. the world admits no such gap at
  all today, which is why freezing SH.02's bars against that pilot would have
  fitted them to a saturated null. *Control:* the same measurement on a
  deliberately-benign twin world must show NO gap — a headroom test that
  cannot detect a world without headroom is measuring nothing. *This spec
  gates every W1 capability claim*: no claim about learning is admissible in a
  venue where doing nothing already wins.

  **`W1.02` — Outcomes have resolution.** *Claim:* the venue's outcome metric
  resolves differences smaller than the effect any claim built on it wants to
  make: ≥ K distinct outcome values over N lives, measurement quantum ≤
  `MIN_GAIN`/3, and censoring below a declared cap. *This is `DP.04`'s defect
  promoted from a pilot's refusal to a world-fidelity gate* — 21 distinct
  lifespans over 3072 lives with a quantum of 6.25 against a `MIN_GAIN` of 5.0
  is a metric that cannot see the thing it exists to see, and the derived
  requirement of ≥ 5791 lives/arm/task is the arithmetic saying so. *Repair
  the metric, not the bar:* censored lifespan is replaced by a graded outcome
  (integrated need-satisfaction, or per-need time-to-first-failure), and the
  spec's job is to prove the replacement resolves. *Control:* a synthetic arm
  with a KNOWN injected advantage of exactly `MIN_GAIN` must be detected; if
  it is not, the metric is still blind and the spec FAILs. Runs on W0 as
  built.

  **`W1.03` — Traps, delays and irreversibility exist, and they are
  discoverable.** *Claim:* the venue contains at least one of each of GOAL.md's
  three named `DP.00` preconditions, and each is (i) DISCOVERABLE — the W1.00
  null encounters it at a rate > 0, so curiosity has something to grip; (ii)
  CONSEQUENTIAL — entering it moves the W1.02 outcome by more than the W1.02
  quantum; and (iii) actually what it says: for irreversibility, no action
  sequence returns the need-vector to its prior value; for delay, the need-cost
  arrives ≥ D decisions after the act; for a trap, the locally-improving action
  is the globally-worse one. *Control — and this is the one that matters:* a
  twin world with the three features REMOVED must fail all three conjuncts
  under the identical measurement. Without that twin this spec would certify
  a world by describing it. *This is the spec that opens the world-edit window*
  and inherits the 21-certificate mechanical bill.

  **`W1.04` — The horizon is longer than the consequence.** *Claim:* for every
  registered W1 claim, the episode horizon is ≥ 3× the MEASURED time-to-
  consequence of the mechanism being claimed, and the run reports both numbers
  on its ledger row. *This is `BA.03` and `LF.01` seen as one defect from two
  ends* — a blind twin holding 98.9% of a 12.0 s horizon and a forager dying of
  integrity at 25 minutes are both "the window closed before the thing we are
  claiming had time to happen". *Falsifier:* if the measured time-to-consequence
  for the venue's mechanisms is already under a third of the horizon
  everywhere, this spec PASSes trivially and should be retired rather than
  kept as decoration — and it must say so on its own row.

  > **STRENGTHENED 2026-09-10 (Review DAILY), on the reading owed by
  > `w0-kills-a-forager-by-integrity-at-25-minutes`.** As published above,
  > `W1.04` names LF.01's 25-minute integrity death as motivation and then
  > constrains only the horizon we DECLARE. That is not enough, and the gap is
  > arithmetic: a horizon is a number a designer picks, and a body the world
  > wrecks at ~1477 sim-s caps the experience actually obtainable no matter
  > what horizon is written down. As written, `W1.04` PASSes a venue where
  > every life ends at 25 minutes simply by declaring a 20-minute horizon —
  > it would certify the exact defect that produced it. So a THIRD conjunct,
  > and it is strictly harder than the two it joins:
  >
  > **(c) THE LIFE IS LONGER THAN THE HORIZON.** For every registered W1
  > claim, the MEASURED survival time — 5th percentile across lives, not the
  > mean — must be >= the declared episode horizon, with the per-life
  > termination CAUSE reported on the ledger row. If lives end before the
  > horizon closes, the horizon is fiction and the spec FAILs; it may not be
  > repaired by shortening the horizon to fit the deaths, because that is the
  > move conjunct (c) exists to forbid. *Control:* a twin run with the
  > terminating mechanism disabled must show the 5th-percentile survival rise
  > above the horizon under the identical measurement — a survival gate that
  > cannot tell a fatal world from a survivable one is measuring nothing.
  >
  > This can only RAISE the bar: nothing that would have passed `W1.04` as
  > published passes it more easily, and a venue that already grants hour-long
  > lives satisfies (c) for free. `W1.04` is NOT REGISTERED (it waits on
  > `w1-world-edit-window`), so this amendment stales no certificate, moves no
  > threshold on any run row, and costs no re-run — it changes what the
  > builder registers, before it is registered.

**ORDERING, and it is not the order the pile is written in.** `W1.00` first
(cheap, no world edit, and it is the one that could show a third of this row's
evidence was under-nulled). `W1.02` second (no world edit, and `W1.01`/`W1.03`
both quote its quantum, so it must exist before they can state their bars).
Then the world-edit sitting: `W1.03`, then `W1.01` measured in the edited
world, then `W1.04` sizing the windows.

**WHAT I DID NOT DO, deliberately.** I did not touch `GOAL.md`. I did not
re-parent any spec. I did not declare W0 dead — `W0.DIAG` is the reason, and a
world where sustained movement is worth twelve units of life is not a world
with nothing in it. I did not bundle the `d10-*` gate rows in here (they are
scoring defects, dispositioned separately this morning). And I did not put a
number on "how deep is deep enough": `W1.01` and `W1.02` make that an empirical
bar rather than a taste, which is the whole reason they are specs and not a
paragraph.

**FOR THE OWNER, and it is why `D10` is coupled to this row.** This design
answers *"what would W1 have to prove"*. It does not answer *"is a new world
the right spend"* — that is `D10` branch (b) and it is the owner's, unchanged.
If the answer is no new world, `W1.00`, `W1.02` and `W1.04` still stand: they
are measurement repairs and they are owed regardless of which world we run in.

ROUTED: me1-similarity-floor-never-abstains | 2026-09-06 | Review FULL 09-06 Part 2 (ME.1 strengthened, FAIL, clean-tree re-buy) | ACTED 2026-09-23 (Review DAILY — executing commits `a33ed72` (the ruling that adopted A5, the contract split) and `a59363a` (the ordered ME.3 harness redesign), both 2026-09-07, SIXTEEN DAYS before the stop-rule that ordered this row DECLINED was armed. The stop-rule's own branch is NOT taken and this is the reasoning, verified by this desk against `experiments/ledger.json` today and not read off any page: ME.1 attempt 10 (2026-09-14T06:18:59, `db4200e`, `dirty_files: None`) is **PASS** with `distractor_abstention` **1.0** over 94.7 cues, `fabricated_abstention` 1.0, `cued_recall` 0.85, against the 0.95 bar which is unmoved in `me_1_event_log.py`; ME.3 attempt 6 (same commit, clean) is **PASS** with `raw_answer_rate` 1.0. A `DECLINED` stamp would refuse work that is on the ledger, and the number the stop-rule ordered routed to the owner — `distractor_abstention = 0.0000 +/- 0.0` — is the 09-06 routing-time figure, falsified by the live certificate; carrying it to the owner's desk would carry a falsehood. `ACTED` is the STRONGER disposition here, not the softer one: it asserts the work exists and names the commits that did it, and it is auditable against the ledger in one command. What was actually wrong was the ROW, not the work — it was stamped `DISPOSITIONED` on 09-07 when its own block already recorded the adoption AND the implementation, so it kept ageing, collected three dated breaks and then a stop-rule, all against a debt that no longer existed. Same family as the `d10-learning-gate-sits-at-the-untrained-twin-level` scar of 09-09: the desk writes the truth in the prose and not in the token the instrument reads. The stop-rule's failure of arming is routed to the owner on today's page as a finding in its own right.)
    DUE: 2026-09-13 | a repair for `EpisodicMemory`'s similarity floor, owed
        by the BUILDER — either a calibration that abstains on absent targets
        without costing `cued_recall`, or a measured demonstration that the
        two cannot be had together on this scorer, which would be an
        architecture finding and belongs on the owner's desk.
    Question: `EpisodicMemory.recall` never abstains when the target is
    absent but its neighbours are present. ME.1 strengthened 2026-09-06 with
    ME.11's distractor control — 60 events held OUT of the store, cued for
    against the 940 retained — measures `distractor_abstention` **0.0000 ±
    0.0** on 3 seeds (40.0 ± 4.5 cues evaluated) at the spec's OWN unchanged
    0.95 bar, while the pre-existing `fabricated_abstention` reads a perfect
    **1.0000**. Verified alive before the row was written: store size 940 as
    designed, and a genuinely out-of-vocabulary cue still returns `[]`, so the
    floor exists — it is calibrated for disjoint vocabulary only. Worked
    example: *"the thing about the meadow and the ladder amber"* returns
    *"ada buried the amber kite near the meadow"* — 2 of 3 content words, full
    confidence, no abstention. That is the failure mode ME.1's own docstring
    names: *"confabulating the nearest neighbour is the failure mode that
    poisons every downstream user of memory — a companion that invents your
    preferences is worse than one that forgets them."* It was invisible for 29
    days because the only control that could see it was in a different spec.
    Blast radius, stated plainly: ME.1 FAILs, and ME.3, ME.5, ME.9, ME.10 are
    blocked behind it (unreachable 94 -> 95, baseline raised with this
    justification in `coverage.py`'s growth log). `ME.9` is named in GOAL.md.
    Staleness bill: NONE mechanical — no PASS certificate hashes
    `EpisodicMemory.py` today. SEMANTIC — the four ME specs above all rest on
    the same floor and each should be re-read once it is recalibrated.
    [CORRECTED 2026-09-06, builder, while executing this row: the mechanical
    bill was NOT none — eight specs declare `EpisodicMemory.py` in
    `IMPL_DEPS` (LG.00/01/02, LG.10, SO.08, LF.01/02, SO.07), five of them
    holding PASS certificates when the floor was edited. All five were
    re-bought in the same slot (~272 s total; LG.00 additionally needed its
    documented `--llm-pass` verdict recompute because the recalibrated
    retrieval changed Jack's prompts).]
    [CORRECTED AGAIN 2026-09-06 (78th audit B3): the grep this note first
    prescribed — `grep -rl IMPL_DEPS experiments/tests/ | xargs grep -l
    EpisodicMemory.py` — was the BLIND SPOT, not the cure: its first stage
    removes every file that declares no IMPL_DEPS, which is exactly the seven
    importers it missed (ME.3/4/5/9/10, ME.11.A, T2.20, XL.00 — found by the
    78th audit). The right question is the reverse one: *which tests import
    it?* — `grep -rl EpisodicMemory experiments/tests/`, then subtract the
    declarers. A search filtered by the declaration cannot find what failed to
    declare. What actually retires this class is the audit's B1 guard (static
    import parse vs IMPL_DEPS), not any grep.]
    **A NOTE ON MY OWN RECOMMENDATION.** This row is the first live test of
    `D23`: I have just discharged a `FAIL-UNOWNED` by routing a FAIL into the
    queue that measures its own drain as UNBOUNDED, which is precisely the act
    I asked the owner to keep counting. It was still the right act — the
    alternative is an unrouted orphan — and it is exactly why `D23`'s default
    is a second printed number rather than a stricter gate. Recorded here so
    that when `FAIL-OWNED-BUT-UNDRAINED` first prints, this row is in it.
    [UPDATE 2026-09-06 ~14:5x (builder, 78th audit B2): the adopted
    coverage-over-known-words calibration has a THIRD measured scar, and this
    time it is the shipped repair that carries it, not a rejected arm.
    `ME.3` attempt 4 FAILed on its equal-tokens honesty gate: its raw-arm cue
    is a bag of 4 mutually-exclusive candidates plus a speaker
    (`" ".join([s] + cands)` — by design, so neither arm gets the answer
    handed to it), every cue word is KNOWN to the store, and no event can
    contain them all, so the floor abstains on EVERY question — raw_tokens
    40.0 -> 0.0, raw_acc 0.625 -> 0.2917 vs base rate 0.25 (attempts 2-3 vs
    4, same seeds). The reflect arm (`Reflections.recall`, no coverage floor)
    was untouched at 1.0, so the "gain" inflated to 0.708 and the
    equal-tokens gate refused to certify a starved null — the harness worked.
    The repair choice this adds to the row's question: the probe measured the
    verbose-recall scar and the terse-answer scar, but never a DISJUNCTIVE
    cue (many known words, at most one per event). Any recalibration should
    add ME.3's cue shape to the probe's arms before adoption. The four
    distractor conjuncts landed by B2 all read 1.0 on the shipped floor
    (ME.1 40±4.5, ME.3 39.3±2.9, ME.9 15/15, ME.10 36/36 evaluated), so the
    abstention half of the trade is measured and healthy; the cost half now
    has a named, committed FAIL row (ME.3 a4, commit 42ad5c9) instead of a
    forecast.]
    [UPDATE 2026-09-06 ~15:4x (builder): THE ROW IS ANSWERED BY MEASUREMENT,
    both halves, and the builder's owed repair is discharged. The
    precondition was honoured first: the probe now carries ME.3's exact cue
    construction (its `_build_life`, `_questions`, `_pack`, `_read`) plus a
    separability measurement of bestcov(q) — the one statistic every floor
    in this family thresholds on. (1) THE DEMONSTRATION: on the same store,
    all three seeds, the two populations separate in the WRONG ORDER —
    distractor cues (must abstain) bestcov 0.667 exactly, disjunctive cues
    (must answer) bestcov 0.400 exactly, gap −0.267, overlap 1.000. Any
    floor ≤ 0.40 admits every distractor (ME.1's measured confabulation);
    any floor > 0.667 starves ME.3 to chance (the a4 FAIL). A0–A4 confirm
    empirically: no floor arm passes all four scar shapes, and by the gap
    none in this scorer family can — the two specs make contradictory
    demands on the same call signature, and the difference (AND-intent vs
    OR-intent) is not in the token bag. (2) THE ROUTING CORRECTION, recorded
    honestly: I first wrote this to the owner's desk as `D25`, per this
    row's own "belongs on the owner's desk" — and `decisions.py` REFUSED it
    (MEANS-ESCALATED: a means fork is settled by bakeoff, not by the owner).
    The checker was right and the row's prose was wrong: the fork's arms are
    runnable, so law 3 governs. D25 is deleted; the bakeoff ran instead.
    (3) THE BAKEOFF VERDICT: arm A5 (alternatives DECLARED at the call site;
    each candidate its own conjunctive sub-cue under the SAME unchanged 0.95
    coverage floor, results union-ranked) is the SOLE SURVIVOR — all ME.1
    conjuncts green (distractor 1.0, fabricated 1.0, cued 0.833–0.867,
    35–46 evaluated), disjunctive 1.000 answered / acc 0.552–0.688 (A0's
    evidence, exactly), verbose 0.833, terse 1.0, on every seed. The
    mechanism is the CONTRACT SPLIT: single-cue recall keeps ME.1's
    abstention contract; OR-intent moves to the call signature, where the
    asker actually holds it. (4) WHAT REMAINS, and whose it is: adoption is
    NOT a module calibration — `EpisodicMemory.py` is untouched today, zero
    certificates staled, the shipped floor stays, and ME.3's FAIL stands as
    a true measurement until a redesign lands. The redesign — ME.3's raw arm
    declaring its four alternatives instead of joining them into one string
    (its reflect arm gets the same declared shape, so the arms stay matched)
    — is a SPEC redesign and is the ROUTER/Review's to order on this row,
    with the probe's A5 numbers as its design input. The builder's owed half
    is done: the row asked for a calibration or a demonstration, and it got
    the demonstration that no calibration of this scorer exists PLUS the
    measured mechanism that does.]
    [VERIFIED AT THE DEADLINE, 2026-09-23 09:1x UTC (builder, the first live
    slot after the stop-rule's midnight — every slot from 2026-09-22T07:07
    through 2026-09-23T08:07 was `STOPPED at 98-100% weekly usage` in
    `ladder.log`, so the "roughly seventeen hourly slots" the arming priced
    were ZERO and no organ has sat since the deadline passed).
    **THE DEBT THIS STOP-RULE NAMES WAS DISCHARGED SIXTEEN DAYS BEFORE IT WAS
    ARMED, and the ledger — not this note — is the receipt. Read before
    stamping anything:**
    (1) The ordered repair is DONE and was VERIFIED BY THIS DESK on 09-07:
    the DISPOSITIONED block immediately below this line adopts A5 (the
    contract split) and records ME.1 a8 `distractor_abstention` 0.0000 ->
    1.0000 with `cued_recall` 0.85 unmoved — "the trade this row was afraid
    of did not happen." Ruling commit `a33ed72`; the ordered ME.3 harness
    redesign landed in `a59363a` and its row returned PASS 09-07.
    (2) The LIVE certificates, neither stale nor drifted in `run status`
    today: ME.1 attempt 10 (2026-09-14, PASS) — `distractor_abstention`
    1.0000 over 94.7 cues evaluated, `cued_recall` 0.85,
    `fabricated_abstention` 1.0, the 0.95 bar unmoved. ME.3 attempt 6
    (2026-09-14, PASS) — `raw_answer_rate` 1.0, `raw_acc` 0.625,
    `aggregation_qa_gain` 0.344.
    (3) The number the arming orders routed to the owner —
    `distractor_abstention = 0.0000 +/- 0.0` — is the 09-06 ROUTING-TIME
    figure and is FALSIFIED by the live certificate. Routing it would carry
    a falsehood to the owner's desk; and the "architecture finding" branch
    was already superseded ON THIS ROW on 09-06, when `decisions.py` refused
    the D25 escalation (MEANS-ESCALATED) and the bakeoff answered instead.
    (4) The three counted "breaks" (09-13 onward) were breaks of a date on
    work that was already on the ledger before the FIRST of them — the
    arming sitting read this row's date line and not its body, and the body's
    falsifying paragraph was adjacent. `run status` now carries a
    STEERING-METRIC-MISMATCH reader (quoted certificate numbers diffed
    against the ledger, `steering.py`, shipped this slot) that flags exactly
    this page-vs-scoreboard divergence; it reads the 0.0000 quote on
    `scripts/ladder_prompt.md` as its first live finding.
    The disposition stays this desk's, and no stamp is written here. But a
    DECLINE would decline work that is on the ledger; the stamp this row's
    own history supports is the one its 09-07 block already wrote — the
    mechanism adopted, the redesign landed, both specs PASS.]
    STOP-RULE ARMED FOR MIDNIGHT TONIGHT, 2026-09-22 (Review DAILY). NO NEW `DUE:` IS WRITTEN HERE AND THAT IS THE POINT — this row's own text binds this desk: *"THIRD BREAK FOR THIS ROW. STOP-RULE, binding on this desk... if this date breaks too, the row is DECLINED and the finding goes to the owner — a promise renewed four times is not a promise."* A fourth re-date is the one disposition forbidden here, so the row keeps its date and takes the break if it comes. The debt is the BUILDER's execution of the `EpisodicMemory` similarity-floor repair, and it has been ordered onto `scripts/ladder_prompt.md` as `1^11` ITEM 0 for today — its last legal window, roughly seventeen hourly slots. If it is unexecuted at midnight, the NEXT sitting stamps `DECLINED` and routes `distractor_abstention = 0.0000 +/- 0.0` to the owner as an ARCHITECTURE finding, which is the branch this row named for itself on 09-06: *a measured demonstration that the two cannot be had together on this scorer belongs on the owner's desk.*

    **DISPOSITIONED 2026-09-07 (Review, DAILY) — THE CONTRACT SPLIT IS
    ADOPTED AS THE MECHANISM, AND `ME.3`'s HARNESS REDESIGN IS ORDERED WITH
    ONE CONJUNCT ADDED THAT WOULD HAVE CAUGHT THIS FAILURE BY NAME.**
    Disposed six days early, off a date that already carried ten rows, for
    one reason: `ME.3` is a settled FAIL that MY OWN 09-06 order caused, and
    a desk that dates its own damage a week out is not owning it.

    **What I verified before ruling, rather than reading the builder's
    report.** (1) `ME.1` a8: `distractor_abstention` 0.0000 -> **1.0000**
    while `cued_recall` is **0.85 +/- 0.0136 — byte-identical to the FAILing
    attempt 5 and to every attempt before it**. The row asked for abstention
    *without costing recall*, and the recall number did not move at all. The
    trade this row was afraid of did not happen, and that is the single most
    important fact on the page. (2) The conjunct is on all five ME specs and
    all five read `distractor_abstention` 1.0 with live denominators — `ME.1`
    40.0+/-4.5, `ME.3` 39.3+/-2.9, `ME.9` 15/15, `ME.10` 36/36, `ME.5` 52-60
    across four store sizes. (3) `ME.3` a4's collapse is real and is exactly
    one thing: `raw_tokens_mean` **40.0 -> 0.0**, `raw_acc` 0.625 -> 0.2917
    against `base_rate` 0.25, `reflect_acc` untouched at 1.0. The raw arm was
    starved to silence and the equal-tokens gate refused to certify the
    inflated 0.708 gain. **The harness caught it. Nothing was hidden and the
    FAIL is a true measurement.**

    **THE RULING.** Adopt **A5**, the contract split, as the mechanism, on
    the separability measurement and not on its story: on the same store, all
    three seeds, `bestcov` for cues that MUST ABSTAIN is **0.667 exactly** and
    for cues that MUST ANSWER is **0.400 exactly** — gap **-0.267**, overlap
    **1.000**. The two populations separate in the wrong order on the only
    statistic any floor in this family can see, so **no monotone single-cue
    floor can serve both, and that is arithmetic, not a preference.** A0-A4
    confirm it empirically and A5 is the sole survivor of a decision rule
    written before the run.

    **WHAT IS ORDERED, PRECISELY, AND WHY IT IS NOT A WEAKENING.**
    - `ME.3`'s raw arm DECLARES its four alternatives instead of joining them
      into one string; each candidate becomes its own conjunctive sub-cue
      (speaker + candidate) under the **same unchanged 0.95 coverage floor**,
      results union-ranked. **`EpisodicMemory.py` is not touched, the shipped
      floor does not move, `ME.1`'s 0.95 abstention bar does not move, and
      zero certificates stale.**
    - **The reflect arm gets the identical declared shape.** This is
      load-bearing: `ME.3`'s claim is that reflection beats raw retrieval, and
      handing the declared form to only one arm would buy the claim with an
      asymmetry. Matched arms or no redesign.
    - **No threshold of `ME.3`'s moves in either direction.** `base_rate`
      0.25, the equal-tokens honesty gate and the `aggregation_qa_gain` bar
      all stand exactly as registered. A5 measured `disj_acc` **0.552-0.688**
      against A0's pre-repair **0.625** — this **RESTORES the raw null to the
      strength it always had; it does not exceed it, and I will not describe
      a restoration as a strengthening.**
    - **ONE CONJUNCT IS ADDED, AND IT IS STRICTLY HARDER: `raw_answer_rate
      >= 0.95`** — the raw arm must SURFACE EVIDENCE on essentially every
      question, per the probe's own pre-stated `disj_answer >= 0.95`, which
      A5 measured at **1.000 on every seed**. Here is why it earns its place:
      the equal-tokens gate caught this failure only because the starvation
      was TOTAL (40.0 -> 0.0). A floor change that starved the raw arm to
      *half* its evidence would have left the gate satisfiable and shown up as
      a *larger* `aggregation_qa_gain` — i.e. **as a better-looking result for
      the claim.** This conjunct converts that from silent flattery into a
      named FAIL. It is additive, it cannot make any spec easier to pass, and
      it is the assertion whose absence cost the ledger a certificate.

    **WHAT THIS ROW DOES NOT CLAIM.** Adopting the contract split says
    OR-intent lives at the call site. It does NOT say Jack's memory now
    handles disjunctive questions — it says our harness must ask them in a
    form the scorer can express, and that the scorer's inability to infer
    intent from a token bag is a real, measured limitation of
    `EpisodicMemory` that we are routing around rather than fixing. **That
    limitation is not written down anywhere a certificate would show it**, so
    it goes to `FOR THE BUILDER` as a docstring the module owes, not as a
    silent success.

    **Status: DISPOSITIONED, not ACTED.** The design is delivered; `ME.3`'s
    redesign and re-run are the builder's, and this row stays live and keeps
    ageing until `ME.3` returns a row under the declared shape. `ME.3`'s FAIL
    stands until then and must not be papered over.
    DUE: 2026-09-11 | RE-DATED EARLIER, not later (was 2026-09-13). The design
        input is complete, the work is one harness edit plus a re-run, and
        09-13 carries ten rows against a measured consumer capacity of ~1 per
        cycle while 09-11 carried one.
    DUE: 2026-09-22 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE. CAUSE, MEASURED AND NOT GUESSED: the builder has been PACE-DARK for 18 consecutive hourly slots since 2026-09-14T12:07 — `week:all models` 37% against a pace line of 35% at 15% of the week, with this project's own attributed share of that meter measured at 25% (builder 7 + desks 2 of 37 points; 28 points, 75%, NOT THIS PROJECT). A row owed by the BUILDER cannot be honestly dated inside a week the builder cannot run in, so this date is set AFTER the 2026-09-21 05:00 UTC meter reset rather than onto another dark day. Routed to the owner as `D30`. THIRD BREAK FOR THIS ROW. STOP-RULE, binding on this desk and on the same terms as the sh02/ba03/t306 class: if this date breaks too, the row is DECLINED and the finding goes to the owner — a promise renewed four times is not a promise. This row is the BUILDER's execution debt (the ME.1 similarity-floor repair), not a decision owed by this desk. ORIGINAL TEXT FOLLOWS, unchanged. | RE-DATED (Review DAILY 2026-09-11), and the reason is the
        one thing the pull-forward above did not check: **it was dated on
        DESK capacity for work that is entirely the BUILDER's.** "One harness
        edit plus a re-run" is a builder slot, and the builder has fired and
        refused **70 consecutive hourly slots** since 2026-09-08T08:23 under
        `pace_gate`. The row was moved onto a day the builder could not work,
        by a desk counting its own load. 09-14 is chosen against the measured
        release bound (forecast 09-12T08:40–23:40, week resets 09-14T05:00), so
        it is the first date with a builder awake on it under every branch —
        not 09-13, which carries 14 rows against a capacity of 6. **Nothing
        about `ME.1` or `ME.3` moves meanwhile: both FAILs stand, the 0.95
        `distractor_abstention` bar does not move, and `ME.3`'s FAIL must not
        be papered over.**
    [UPDATE 2026-09-14 ~06:2x (builder) — EVIDENCE ADDENDUM, NOT AN ARRIVAL:
        the conjunct this row shipped was CERTIFIED BELOW ITS OWN BAR, and
        the mechanical half is repaired. Field watch wk7 §6b applied
        MEMORY_RETRIEVAL_BAKEOFF §1.8 to the five denominators quoted in this
        row's own verification note (2): a perfect run over m negatives
        certifies a_L = γ^(1/m) at confidence 1-γ, so m >= 59 at γ=0.05 is
        the minimum that can certify 0.95 — and the shipped denominators were
        ME.9 15 (certifies 0.819), ME.10 36 (0.920), ME.3 39.3 (0.927), ME.1
        40.0 (0.928), ME.5 52 at its smallest decade (0.944). Every recorded
        1.0 on those five was statistically compatible with a true rate below
        the bar it was read against. EXECUTED TODAY (db4200e + runner rows):
        ME.1/ME.3/ME.5 raised to N_DISTRACTOR 130 / MIN_DISTRACTOR_EVAL 59
        (bars untouched both directions) and re-bought PASS — evaluated now
        94.7 ± 4.9 / 87.7 ± 3.4 / 110-130 per decade, abstention 1.0
        everywhere, worst certified level 0.966. NOT EXECUTED, and the desk
        should rule rather than the builder improvise: ME.9's denominator is
        capped at 36 BY CONSTRUCTION (3 askable speakers x 12 topics, and
        full censorship degenerates the control store), and ME.10's 36 held
        pairs are LOAD-BEARING in the main claim (N_SEEN 84 of the 120-pair
        grid feeds the skill net; holding out >= 59 either starves training
        against the unmoved MIN_SKILL 0.85 or grows the grid, which moves
        chance floors). Both need a fixture redesign to reach a certifiable
        denominator; until then their 1.0 readings certify 0.819 / 0.920 and
        should be read as such. ME.11 runs at m=300 and is clear.]

ROUTED: w1-world-edit-window | 2026-09-06 | Review FULL 09-06 (w0-too-shallow disposition) | OPEN
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — THE FACT, NOT A NEW
        PROMISE, AND NOT A RE-DATE. The Sunday FULL this row was dated to sat at
        06:37 today and DID NOT TAKE UP W1. Verified against the day's commits
        rather than taken from the audit: the FULL's output was its page
        (`11face6`), two new champion seats (`e9c1b68`), `D27` (`83132c9`), a
        completeness audit and a replaced priority block — no W1 design commit.
        `run review-queue` still reads `w0-too-shallow ordered W1.01/W1.03/W1.04
        -> NOT REGISTERED`, so the sitting this row promised has no spec to serve
        and did not open. This row goes OVERDUE at 00:00 tonight and that is the
        promise breaking, not the instrument. NOT re-dated: re-dating the
        Review's own design debt is the Review's call, and the six rows below
        and behind this one carry the same note for the same reason — the next
        reader is tomorrow's 06:37 DAILY, looking at red rows with no cause
        attached to any of them.
    DUE: 2026-09-18 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | the single world-edit sitting that `W1.03` opens, which
        pays the 21-certificate `playground.py` mechanical bill ONCE for every
        world edit that is owed. This row exists so that the two holds that
        were `BLOCKED-BY: w0-too-shallow` have a LIVE blocker after that row
        was dispositioned this morning — a hold whose blocker is dispositioned
        is itself a violation, and re-pointing them is the honest repair, not
        quietly leaving them pointed at a closed row.
    BLOCKED-BY: w0-too-shallow | the W1 design above must be REGISTERED
        (W1.03 in particular) before a world edit has a spec to serve; editing
        the world first would be the 21-certificate bill paid for a change
        nothing yet measures.
    DUE: 2026-09-27 | RE-DATED 2026-09-24 (Review DAILY), and this is the FOURTH
        break of this row's date. A bare fifth re-date on the old cause would be
        dishonest, so here is what is NEW today and measurable. (1) **The cause
        every previous re-date cited is GONE.** 09-15 and 09-20 both re-dated on
        "the builder is PACE-DARK"; `week:all models` reads **19%** this morning
        and the builder has run `rc=0` in six consecutive hourly slots
        (02:1x–06:1x). Nothing is dark. (2) **The cause that remains is this
        desk's own sitting length, and it is now measured rather than suspected.**
        Every one of this row's four breaks happened inside a 20-minute DAILY
        walk-through; `scripts/review.sh:80-94` records **7 max-turns deaths
        across the three organs**, and **four of four Sunday FULL runs ever fired
        on cron died at max turns** — including the 09-23 sitting, whose own
        `PROGRESS.md` still carries the INCOMPLETE-RUN banner. A from-scratch
        world specification has never once fitted in the time this desk is given,
        and that is not a fact about any one morning. (3) **`D33` is the entry
        that asks about exactly this, its `decide_by` was 2026-09-23, and it is
        one day STALE on the owner's desk.** Its default (i) — re-date once more
        and change nothing else — has now fired and produced nothing for the
        fourth time, which is precisely the price the entry stated when it armed
        it. Dated onto **Sunday 2026-09-27**, deliberately: not because Sunday is
        free (the pile reads 6 there) but because the FULL sitting is the only
        one with twice the clock, and a unit that has lost four 20-minute
        sittings should not be given a fifth. **STOP-RULE, stated in the open: if
        2026-09-27 breaks, this desk does not re-date this row again — it
        DECLINES the authorship and says so on the owner's page, `D33` answered
        or not.** ORIGINAL TEXT FOLLOWS, unchanged. | RE-DATED 2026-09-20 (Review FULL), and the reason is a
        finding against THIS DESK, written plainly rather than as a scheduling
        note. **The 2026-09-18 date BROKE and this is the row's fourth slip.**
        `D21` was RESOLVED BY ARMED DEFAULT on 2026-09-06 with the instruction
        that the FULL take the W1 design as its FIRST design item; **three FULL
        sittings have now passed — 09-06, 09-13 and today — and the design does
        not exist.** I am not re-dating this quietly. Two things make today
        different from the previous two slips and both are recorded against me:
        **(1) I took the `sh02`/`ba03`/`t306` bundle first today, not W1** — a
        defensible choice, because those three carried a STOP-RULE that had
        fired and W1 does not, and because ruling three rows beats designing
        none; but it IS the choice I made and it is the third consecutive FULL
        at which W1 lost. **(2) Today's ruling put TWO MORE repairs behind this
        window** — `SH.02`'s adopted arm (b) and the newly-split
        `ba03-vestibular-channel-is-never-load-bearing-under-one-kick` — so my
        own act this morning increased the load on the one thing I did not do.
        Three rows now queue behind an undesigned window. **Re-dated to
        2026-09-23, the earliest date carrying capacity, and DELIBERATELY NOT
        to the next FULL**, because a fourth consecutive Sunday would be this
        desk betting on the sitting that has lost this race three times.
        **STOP-RULE, binding, and on harder terms than the sh02 class because
        this row has already outlived that class's stop-rule: if 2026-09-23
        breaks, W1 is not re-dated again by this desk — it goes to the owner as
        a decision about whether the Review is capable of producing it at all,
        with the recommendation that design authority for the world edit be
        moved.** Routed to the owner today as `D33`. ORIGINAL TEXT FOLLOWS,
        unchanged.
    Question: which world edits ride this single sitting, and in what order?
    Known passengers as of routing: `W1.03`'s traps/delays/irreversibility;
    `ne01-occlusion-knife-edge`; `water-apply-phantom-force` (Water.apply
    writes a body's xfrc row only while it is in the pool, so a body that
    exits keeps its last buoyancy force forever); and whatever the `W1.01`
    headroom repair turns out to need once `W1.02` has fixed the metric.
    Staleness bill: MECHANICAL — all 21 `playground.py` rows; SEMANTIC —
    BA.01, LC.02, PS.02, PS.03, XL.00 (worlds with a pool, per the 27th
    audit). Paying it once instead of three times is the entire point.
    Full record: D10 + its 08-24 evidence update; FAIL RECORD in
    dp_05_lookahead_pays_in_w0.py.
    UPDATE 2026-08-25: a FOURTH instrument, weighing differently — SH.01's
    pre-registered oracle pilot at the full envelope (ORACLE_CANNOT,
    z_shelter 0.0 with the working-hut direction IN the observation) removes
    the perception excuse and implicates the certified ppo-needs CORE
    jointly with the world: sheltering demonstrably pays (curriculum lives
    shelter, freezing kills) and the core still cannot learn to seek it.
    See D10's 08-25 evidence update. Design input: world redesign (b) alone
    may not suffice; the learning-core seat is part of the same question.
    Staleness bill: depends on the design — a new-world spec (W1, T1.02
    strengthen-only precedent) bills NOTHING; editing W0's playground.py
    bills the 21 rows above. That asymmetry is itself design input.
    UPDATE 2026-08-31 (Review, DAILY): THE COUNT IS NINE, AND NOBODY WAS
    COUNTING. This row says three, its 08-25 update says four, field watch
    wk5 says six then seven. The true figure is NINE, and the gap exists
    because each new instrument was routed as its OWN queue row — so the
    aggregate was never assembled anywhere. Named, so the next reader does
    not have to re-find them: (1) LC.03 darkroom control, passivity prospers;
    (2) LC.03 v2, one learner in five; (3) DP.05, deeper lookahead buys LESS;
    (4) SH.01 ORACLE_CANNOT, z_shelter 0.0 with the direction IN the
    observation; (5) BA.03, the blind twin holds 11.868 s of a 12.0 s horizon
    (98.9%); (6) SH.02, twin/oracle/both-cosmetic all exactly 1.0000 against
    a 0.85 cap — the null holds the roof it was placed under; (7) T3.06's
    recorded row, `curious − random` = +0.0124, t = 0.39, while
    `random − task` = +0.2333, t = 10.48; (8) DP.04 SIZING RECORD; (9) the
    T3.06 control `delta_shuf` red on every seed.
    AND (8) IS DIFFERENT IN KIND — it is the one that should lead the design.
    The other eight say "W0 does not REWARD capability X", each on its own
    channel, which is the agreeing-instruments pattern this row already flags
    as the condition under which a shared confound is invisible. DP.04 says
    something else: **the outcome variable itself has no resolution.** 3072
    lives produced 21 distinct lifespans; 0 of them ended between the old cap
    and the new one; the quantum is 6.25 steps at 48 lives against a MIN_GAIN
    of 5.0, and the derived sd needs E>=5791 lives/arm/task. A threshold finer
    than its statistic's quantum is not a hard test, it is an unreadable one —
    and lifespan is the channel most of the other eight are ultimately scored
    through. That makes DP.04 a live CANDIDATE for the shared confound the
    other instruments cannot see past, which is precisely what wk4-N3 was
    ordered to attack from the other side. Design input, in one line: settle
    whether W0 is too shallow or merely too COARSE before choosing between
    editing W0 and building W1 — they have different repairs and only one of
    them bills the 21 rows.
    DISPOSITION 2026-08-25 (Review): **STAYS OPEN, and the design is owed by
    this desk on 2026-08-30, dated so it cannot drift.** DAILY mode does not
    have the budget for a world redesign and pretending otherwise is how a
    routed row rots. But one thing IS ordered today, and it is ordered BEFORE
    the design, not after: **run the cheap falsifier first.** All four
    instruments behind this row are expensive (LC.03 v2 ~190 core-h, DP.05
    ~115 min, SH.01's pilot at N=10000), they were run by this project on this
    world, and they all point the same way — which is exactly the condition
    under which a shared confound is invisible. Field-watch wk4-N3 supplies a
    CPU-minutes attack on that confound: a beta-scheduled colored-noise random
    policy against the plain `random` and `random-repeat` nulls LC.03 already
    defines. It asks whether "the cores cannot learn in W0" is partly "the
    exploration process never reaches the food". If it fires, the diagnosis
    changes before we spend a redesign on it; if it does not, the shallowness
    finding survives an attack that cost almost nothing. **A redesign informed
    by four expensive agreeing instruments plus one cheap disagreeing one beats
    a redesign informed by four.** Queue entry: INTEGRATION_QUEUE, wk4-N3.
    UPDATE 2026-08-31: a SIXTH instrument, and the first one on the humanoid
    body rather than the gridworld — `BA.03`'s registered VOID. Its BLIND twin
    (no vestibular channel, plantar touch pinned) holds **98.9% of the 12 s
    horizon**, leaving 0.132 s of room for a claim needing 1.336 s, with the
    other six rig conjuncts green on every seed. This weighs differently from
    the five above and sharpens the fork: those measure the world as too easy
    to REWARD a capability; this one measures it as too easy to *require* one —
    the sense being tested is not merely unhelpful, it is unnecessary, and the
    spec's ANATOMY table names the substitute (the winning policy reads plantar
    touch and nothing vestibular). Note also that `BA.03`'s option (c) and
    `DP.04`'s option (i) are the same repair — a bounded outcome variable that
    saturates — arriving on two unrelated rigs, so "deepen the world" and
    "change the claim statistic" are separable questions this desk should
    answer separately. Row: `ba03-null-saturates-the-horizon`.
    UPDATE 2026-08-31 (builder, LT.01 attempt 1 FAIL): the "body cannot act
    in it" reading — which PROGRESS 08-31 FOR THE OWNER §1 called untestable
    for lack of an arena — now has its FIRST registered-spec measurement, in
    the playground rather than W0. LT.01's C2 clause pre-registered (from the
    08-09 pilot) that a random agent reaches >= 0.6 m of NON-LADDER torso
    rise: the pilot's free-roam z ceiling was 1.007 m. On the as-built rover
    body the recorded row reads `nonladder_rise_max` **0.084 +/- 0.067 m**
    across 3 seeds x 3000 decisions — the body tips over within seconds and
    travels by dragging, never regaining standing (W0.BAL's 0.002-0.004
    upright fraction, reproduced on a third rig). Every aliveness guard was
    green (force calibration +1.000 W, scripted hang ENGAGED through the full
    h(t) conjunction, oracle rise 0.416 m), and the OTHER three clauses of
    the measurability claim all held: null floor exactly 0 engaged attempts,
    P(hang|3 s burst) 0.031 inside the pre-registered [0.01, 0.05] bootstrap
    band, platform unreachable by free-roam AND the adhesion-disabled oracle.
    So the instrument is certified alive and honest while the BODY fails the
    gameability premise. Design input, one line: LT.02/LT.03 (the north-star
    arena, registered 08-31, frees 7) are now blocked behind a FAIL whose
    falsified clause is a fact about the body, not about h(t) — the same
    repair fork as D9/W0.BAL, arriving from the curiosity ladder's side. A
    redesign that re-scopes C2 must route through this desk (threshold rule:
    strengthen-only, T1.02 precedent), not through a quiet re-run.
    UPDATE 2026-09-01 (builder, UB.14 probe record `cf0ff46`): the SENSORY
    mirror of LT.01's motor finding, on the same playground venue. UB.14's
    fixture aliveness gate (`vision_sees_body` >= 0.5, pooled frame -> root
    xy) is measured unreachable by ANY decoder in the only region the world
    contract allows the body to be seen: linear ridge 0.374 at the full
    envelope (flat in resolution 96/48/24 px), body-blob centroid features
    0.275-0.295, the rig's own MLP trainer 0.159 held-out. The binding
    constraint is geometric, not statistical — the contract eye's 30 deg
    half-FOV admits a +-0.4 m in-view spawn box, so var(root xy) is small
    against the tumbling body's blob-centroid-vs-root offset; episodes
    16 -> 48 move the reading 0.26 -> 0.37, saturating. Downstream, vision
    carries ~zero touch-relevant signal under a random policy
    (vision_only_r2 0.009), so the fused arm at matched capacity is drowned
    by its own vision dims (best 0.039 vs the 0.05 floor across pool4 and a
    100x WD sweep, the WD lever capped by the loss_fell conjunct). Design
    input, one line: the playground cannot currently test ANY claim of the
    form "vision helps X" — the eye cannot place the body and the policy
    never makes vision matter — which is the same eye/body/venue fork as
    D9/W0.BAL and LT.01's C2, arriving from the unison ladder's side. The
    recorded 3-seed VOID lands 2026-09-01 (launched this slot);
    VOID-FORECLOSED declaration owed at harvest. Do not re-run unchanged;
    do not lower VISION_BODY_GATE.
    CONSOLIDATED NOTE 2026-09-02 (builder, executing the 62nd audit's B4 —
    a bundling of EVIDENCE, not of decisions; nothing below pre-empts any of
    the nine repair rows, per the 09-06 stagger's own distinction). Six specs
    have now INDEPENDENTLY RECORDED, in their own registered words, that the
    venue — not the instrument — is what failed. Quoted verbatim with their
    numbers so Sunday's desk sees one convergence instead of six unrelated
    arm choices:
    (1) DP.04 (SIZING RECORD, dp_04_slow_path_verbal.py, seed 94, 08-30):
        "mean censored lifespan has no resolution in W0 — 0 of 3072 lives
        ended between the old cap and the new one, 21 distinct lifespans,
        quantum 6.25 steps at 48 lives against MIN_GAIN 5.0, and E>=5791
        lives/arm/task would be needed for the derived 2.357-step sd." And
        its FINDING paragraph: "W0's survival task is near-binary at every
        cap, so a mean-lifespan statistic cannot resolve a 5-step effect at
        any affordable envelope."
    (2) SH.02 (PILOT RECORD, sh_02_born_sheltered.py, seed 90, 08-30,
        N=3000/arm): "Every arm without a live policy gradient holds the
        roof COMPLETELY — twin, privileged oracle and both-cosmetic control
        all exactly 1.0000 against HEADROOM_MAX 0.85, learner 0.0136 — so
        the null already holds the roof it was placed under and no choice
        can show above it. [...] this is D10 evidence that W0 is the
        bottleneck."
    (3) UB.14 (VOID-FORECLOSED declaration, ub_14_cross_modal_touch.py,
        recorded row 09-01): "The binding fault is the VENUE, measured, not
        the instrument: the eye is world contract (EYE_POS fixed, 30 deg
        half-FOV), the spawn is at its measured in-view optimum, and in the
        only region the body may be seen the information to place it does
        not reach the gate" — vision_sees_body 0.4036 +- 0.0256 vs the 0.5
        gate, fused_r2 0.0013 +- 0.0098 vs the 0.05 floor, the rig's own
        MLP (strongest readout tried) 0.159 held-out at the full envelope.
    (4) BA.03 (VOID-FORECLOSED declaration,
        ba_03_braces_against_a_surface.py, recorded row 08-31): "the blind
        twin holds 11.868 s of the 12.0 s horizon (98.9%), leaving 0.132 s
        of room for a claim that needs 1.336 s — headroom ratio 0.236 +/-
        0.184 against HEADROOM_MIN_MULT 2.0 [...] Clearing it requires the
        twin's ceiling share to fall from 98.9% to <= 88.9% — a redesign of
        the world or the horizon, not a sample size."
    (5) T3.09 (attempt-3 row 09-02 + the vacuity lane's amended docstring,
        t3_09_creative_loop.py): "the shuf control cleared the margin (the
        site rewards any detour perturbation and the test measures
        nothing)" — creative_contribution -9.96 vs MARGIN_AFF 11.0 while
        the wrong-goal control gained +12.47 and CLEARED the claim's own
        margin, loop_creative 0 on every life.
    (6) LC.03 (VOID-FORECLOSED declaration, lc_03_survival_screening.py,
        v2 recorded 08-23): "fewer than two learners (1 cleared)" after
        400k decisions/arm-seed and ~190 core-hours at the 4x envelope,
        every control on its pre-registered side; "The repair is a REDESIGN
        of the screen or of W0, on the owner's desk since 2026-08-24."
        [POINTER STALE, corrected 2026-09-06 (79th audit §3.3b): D10 fired
        its armed default 2026-09-01 and that desk is closed. Live homes:
        D24 (screen/seat, decide_by 09-11) and the W1 family (W0). The
        quoted sentence is kept as the historical declaration; the source
        docstring in lc_03_survival_screening.py now carries the same
        correction, which is what `coverage` prints.]
    Shape of the convergence, for the design: (1) says the OUTCOME VARIABLE
    has no resolution; (2), (4) and (6) say the NULL already holds the
    ceiling the claim was placed under; (3) says the SENSORY contract cannot
    deliver the signal its own gate demands; (5) says the venue rewards
    perturbation AS SUCH. Three different failure channels (statistic /
    ceiling / channel), five different families (fast-slow, shelter, unison,
    balance, curiosity, learning-core), all landing on the venue. The nine
    instruments enumerated in the 08-31 update above remain the full count;
    these six are the subset that recorded the diagnosis IN THE SPEC ITSELF
    rather than in a routing row. Sources: _PILOT_BLOCKED in
    dp_04_slow_path_verbal.py and sh_02_born_sheltered.py; VOID-FORECLOSED
    blocks in ub_14_cross_modal_touch.py, ba_03_braces_against_a_surface.py,
    lc_03_survival_screening.py; the vacuity lane in t3_09_creative_loop.py.
    UPDATE 2026-09-04 (builder): DESIGN INPUT LANDED — the fast/slow
    literature sweep owed since 08-10 is now `docs/research/DUAL_PROCESS.md`
    (four research agents, citations verified live where possible). §6 states
    the five world properties the DP family needs from W1, each traced to the
    paradigm that needs it: perceivable-in-advance hazards; REVALUABLE
    OUTCOMES (the TA taste-aversion machinery is biology's own devaluation
    manipulation, so the taste family doubles as the habitisation instrument
    — the strongest synergy the sweep found); degradable contingencies;
    heterogeneous stakes/novelty (a uniform world gives a deliberation gate
    nothing to allocate — DP.04's no-resolution finding and DP.05's H10<H4
    are the same fact from this side); stable rules punctuated by rare
    revaluation events. §5 records the neuroscience verdict the owner is
    owed: biology supports a shared state TRUNK with two differently-wired
    heads and an external arbitrator, NOT one-network-two-depths (the
    Yin/Knowlton/Balleine double dissociation is the counterargument,
    recorded per SYSTEM.md's owner-directive duty). §7 holds strengthen-only
    revision drafts for DP.01/DP.02/DP.03 — none registered, all waiting on
    this row's design.

ROUTED: t215-router-under-lexical-null | 2026-08-25 | 20b8660 (row ran_at 2026-08-25T04:40) | DISPOSITIONED 2026-09-10 (Review DAILY — NOT DECLINED, because its own decline-condition is not met, and because reading it turned up a defect one level above the question it asks: **the mechanism it wants to unseat holds no seat.** See FINDING below; the seat question is routed to Sunday's ANATOMY AUDIT, DUE 09-13)
    DUE: 2026-09-10 | re-armed by the builder, 2026-09-03, under 64th-audit
    B4 (9 d OPEN, past the 8-day cycle, no date). Reason: the honest ACT is
    registering a retrieval/bag-of-words challenger as a bakeoff arm — which
    this row's own staleness analysis prices at zero bill — but WHICH seats
    get challenged, and in what venue, is the same registration-asymmetry
    design input the Review takes up with `w0-too-shallow` on 09-06; dated
    09-10 so it comes due AFTER that decision and off the eighteen-row 09-06
    pile (61st audit B2). If 09-10 arrives with no challenger registered and
    no Review disposition, DECLINE it in the open rather than re-arm again.
    Question: the shipped routing mechanism (UnifiedBrain semantic-anchor
    argmax over compute_language_grounding_loss) transfers held-out phrasings
    at [8,9,5] of 16 vs a 12/16 bar on a grid DESIGNED for composition and
    provably lexically resolvable (NB 14/16, TF-IDF 11/16) — on seed 2 it
    routes WORSE than both registered bag-of-words nulls. Paired with T2.07
    (FAIL: shipped-table composition [2,2,2] of 5), two independent FAILs now
    localise the defect in the MECHANISM, not the training data: does the
    anchor-argmax router keep the language-routing seat, or is the seat's
    challenger a retrieval/bag-of-words baseline that currently outperforms
    it? Full record: FAIL RECORD in t2_15_freeform_routing.py.
    Staleness bill: SEMANTIC — T2.06 (PASS) is the only certificate about
    this mechanism's behaviour. MECHANICAL — any edit to UnifiedBrain.py
    stales 4 PASS rows whose IMPL_DEPS hash it: T2.03, T2.04, T2.06, T3.01.
    A challenger registered as a NEW spec (bakeoff arm, T1.02 precedent)
    bills NOTHING; that asymmetry is the same design input as w0-too-shallow.

    FINDING 2026-09-10 (Review DAILY) — **there is no language-routing seat.
        This row asks whether the anchor-argmax router "keeps the seat", and
        the register does not contain one.** `experiments/champions` lists
        `Language grounding (word → lived skill)` as **UNDECIDED** with arena
        `LG.04 LG.05 LG.06` — all three `NOT_RUN` — and `Language acquisition`
        / `Language model` both `BY DECREE` on arena `LG.00`. Not one of those
        arenas is `T2.15`, `T2.07` or `T2.06`. So the shipped mechanism this
        row indicts — `UnifiedBrain`'s semantic-anchor argmax over
        `compute_language_grounding_loss` — is the default champion of
        nothing, while being about as load-bearing as a component gets:
        **four PASS certificates hash `UnifiedBrain.py` in `IMPL_DEPS`
        (`T2.03`, `T2.04`, `T2.06`, `T3.01`)** and two independent FAILs
        (`T2.15` at [8,9,5]/16 against a 12/16 bar, beaten by both registered
        bag-of-words nulls on seed 2; `T2.07` at [2,2,2]/5) localise a defect
        in it. **A component that four certificates depend on, that two specs
        have refuted, and that no seat watches, is the exact thing
        `CHAMPIONS.md` exists to prevent** — and it is invisible to every
        audit we run, because seat staleness checks seats that EXIST.
        Registering the retrieval/bag-of-words challenger, which this row
        correctly prices at a zero staleness bill, would put a challenger
        into an arena with no chair in it.
        Why this is not settled here: **adding a seat is the ANATOMY AUDIT's
        act and the anatomy audit runs on Sunday FULL.** The Review's own
        standing rule permits adding seats directly with justification
        (adding one only invites competition, so it loosens nothing) — but
        doing it in the closing minutes of a DAILY, off a single register
        read, is how a seat gets carved with the wrong boundary. The 09-09
        desk refused a comparable last-ten-minutes ruling on `pl02` for the
        same reason and that refusal stands as precedent.
    DUE: 2026-09-26 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. The DESIGN IS DELIVERED and this row is DISPOSITIONED: what this date owes is EXECUTION by the builder, not a decision by this desk. Dated where a builder slot can plausibly reach it rather than onto the Review's own calendar. | **the ANATOMY AUDIT item, and it is deliberately ON the
        Sunday pile rather than off it** — seat creation is FULL-mode work
        and no other sitting can do it. Two conjuncts owed: (1) does the
        language-ROUTING mechanism get its own seat, or is it inside
        `Language grounding (word → lived skill)`'s boundary and that seat's
        arena is simply wrong (`LG.04/05/06`, all NOT_RUN, none of them the
        specs that actually measure the router)? (2) whichever way (1) falls,
        the retrieval/bag-of-words challenger is registered as a NEW spec at
        zero bill, per this row's own analysis and the `T1.02` precedent.
        **Declared, not buried: this takes 09-13 to 14 rows against a
        measured capacity of 6.** It goes there anyway because the alternative
        is a fifth week of a load-bearing component nobody has chaired. The
        decline-condition written on the 09-03 re-arm ("if 09-10 arrives with
        no challenger registered and no Review disposition, DECLINE") is not
        met: a Review disposition is what this is.

ROUTED: t211-diayn-metric-cannot-separate-mi-from-noise | 2026-08-29 | pilots /data/t2_11_pilot2_seed{7,90}.json | OPEN
    DUE: 2026-09-16 | RE-ARMED 2026-09-07 (builder): the row went STALE at 9 d
    with no DUE — the one live queue violation. 2026-09-16 is `next_free_due`,
    the tool's own mechanically-named first date carrying no promise (09-07
    through 09-13 are all AMBER piles against a measured 1 row/cycle). The row
    itself is unchanged: a METRIC redesign for the Review, zero staleness bill,
    T2.11 PARKED behind it.
    Question: what measurement separates "skills differ because I(S;Z) was
    maximised" from "skills differ because they chased different noise"?
    T2.11's label-permuted control passed BOTH pilots and on v2's seed 90 —
    every rig gate green — it BEAT the claim arm (0.8984 vs 0.7812,
    margin −0.1172). The mechanism is not the policy class: `shuffled`'s
    discriminator is provably uninformative (loss pinned at ln 8 = 2.0794),
    but `compute_diayn_reward` reads log q(z|s) off it, and a network carrying
    ZERO information about z still emits (s, z)-varying outputs — so the
    control is paid a fixed random reward field (mean |r| 0.29–0.35 vs DIAYN's
    1.40–1.50) and a shared conditioned policy chasing a random field separates
    its skills as well as one chasing MI. Held-out skill-classification
    accuracy measures the POLICY's response to any structured reward, not the
    OBJECTIVE's information content, so NO repair to the rig can fix it. This
    is a METRIC redesign, not an arm redesign — which is why it is here and not
    in a bakeoff. Candidate directions the Review should weigh as arms, all
    cheap and all CPU: (a) score the discriminator's held-out MI directly and
    gate on claim-minus-control MI rather than on downstream separability;
    (b) subtract the field — a fourth arm paid by a FROZEN randomly-initialised
    discriminator, making "beats a random reward field" the registered null
    instead of chance; (c) test the property DIAYN actually promises (skills
    are individually *identifiable and reusable*, not merely mutually
    distinguishable). Note (b) is the null this spec should probably have been
    written against from the start, and it is nearly free — it reuses the
    existing rig unchanged.
    Full record: PILOT RECORD v2 in t2_11_skills_distinguishable.py; the
    generalised lesson is in docs/LESSONS.md ("A REPAIR CAN BE RIGHT AND
    CHANGE NOTHING").
    Staleness bill: NONE, and this is the cheapest row on the page. T2.11 has
    no PASS certificate, `SkillDiscovery` has never passed a registered
    experiment, and every candidate direction above is a change to THIS SPEC's
    metric — a new null arm or a new gate — not an edit to `UnifiedBrain.py`.
    So nothing goes mechanically stale: the 4 PASS rows that hash
    UnifiedBrain.py (T2.03, T2.04, T2.06, T3.01) are untouched unless the
    Review chooses to change the component rather than the measurement.
    DUE: 2026-09-25 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. T2.11 is PARKED behind this row and has been for 24 days, which makes it the only row in the overdue class holding a registered spec hostage. Zero staleness bill, a METRIC redesign, no run needed — it is cheap and it has been outranked by louder things five times.

---

ROUTED: aggregate-hides-worst-seed | 2026-08-30 | bf947a1 (found writing T3.06 v2) | OPEN
    DUE: 2026-09-18 | RE-ARMED 2026-09-08 (Review DAILY) from no date at all.
        This row went STALE — OPEN 9 days past the 8-day consumer cycle with no
        `DUE:` to re-arm it — and that is this desk's fault, not the router's:
        it was routed without a date and nothing but the staleness lane ever
        asked about it. The date is chosen and not arbitrary: today's `UB.10`
        disposition adopts a PER-ARM STABILITY conjunct (an arm must train on
        all registered seeds), which is this row's question one layer down —
        a worst-case requirement gated on the seed MEAN is not a worst-case
        requirement. Decide the recorder change with the UB.10 redesign
        (DUE 09-15) already in hand, so the general fix is shaped by a concrete
        caller instead of in the abstract. Dated onto a day carrying no other
        promise.
    Should `protocol.py:_aggregate` emit `<key>_min` / `<key>_max` across seeds
    beside the `<key>_std` it already emits — so a spec can gate the WORST SEED
    directly instead of reconstructing it from mean ± 1.5*std?
    THE FINDING (mechanism, not opinion). `_aggregate` means every numeric
    metric across the registered seeds before `_check` is called once, and
    `_check` receives a flat dict of scalars with no marker saying which were
    already averaged. So a metric whose NAME and PURPOSE are "the worst X"
    — `n_informative`, `*_worst_life`, any per-seed min/max — is silently
    gated on the mean of the per-seed worst cases. Seeds with 2, 6 and 10
    informative lives average to a healthy 6 and clear a gate no seed clears.
    T3.06 v2 closes it locally with an exact bound (for n=3, ddof=0, the
    extreme deviation is <= sqrt(2)*std, so 1.5*std bounds every seed) and the
    generalised rule is now in docs/LESSONS.md ("A worst-case instrument gated
    on the SEED MEAN is not a worst-case instrument"). But that is a LESSON,
    i.e. a thing the next author must remember — and the grep is not
    reassuring: 26 spec files fold a `worst`/`_lo`/`_hi` quantity and 89 lines
    read a `_std`, so the population that could carry this bug is large and
    nothing mechanical distinguishes a correct gate from a wrong one.
    WHY IT IS ROUTED AND NOT JUST DONE. The fix is four lines and strictly
    additive, but it is an edit to the RECORDER, which is the one file whose
    behaviour every certificate depends on, and the cheap version has a real
    failure mode: emitting `_min`/`_max` makes the WRONG gate (raw mean) no
    harder to write while making the right one easier, so it improves ergonomics
    without closing the hole. The stronger arms, for the Review to weigh:
      (a) additive `_min`/`_max` — cheapest, ergonomic only;
      (b) `_aggregate` REFUSES to flatten a key matching a worst-case naming
          convention (`*_worst_*`, `n_informative`, `*_min`/`*_max`) into a
          bare mean, emitting only `_min`/`_max` for it, so a spec that gates
          the mean gets a KeyError rather than a plausible wrong number. This
          is the version that makes the bug unrepeatable, and it is the one
          that will break existing specs — which is the point and the cost;
      (c) leave the recorder alone and add a T0-family static audit that reads
          each spec's `_check` and flags a bare `m["<key>"]` comparison on a
          key the same file folds with min/max/len. Catches it without touching
          the recorder; needs an AST pass and will have false positives.
    Staleness bill, MECHANICAL: 4 spec files name `protocol.py` in IMPL_DEPS —
    T0.17, T0.22, T0.27, XL.00. All four are cpu<1min or fixture, so the
    re-certification cost is minutes, not GPU hours; T0.27 is already RED and
    stale for unrelated reasons (PROGRESS B4). SEMANTIC: zero under arm (a),
    since no existing gate's value changes; under arm (b) every spec that gates
    a renamed key fails loudly at its next run, which is the intended behaviour
    and must be paid deliberately rather than discovered.
    DUE: 2026-09-29 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. A recorder change (`protocol.py:_aggregate`) whose concrete caller was meant to be the UB.10 redesign; that redesign is now itself blocked, so the coupling argued on 09-08 no longer sets the date. Re-dated on its own merits onto an empty day rather than inheriting a blocked row's schedule.

    THE SWEEP THIS ROW ASKED FOR IS ATTACHED (builder, 2026-09-12, 90th audit
    B2 — no second row opened). It was RUN, not estimated: an AST pass over
    every multi-seed PASS row, extracting each `_check` comparison of
    `m["<key>"]` against a resolvable module constant where `<key>_std > 0`,
    and testing the exact n=3 extreme-value bound `|x_i - mu| <= sigma*sqrt(2)`
    (ddof=0) against the bar. No new machinery: the ledger and the spec files
    are the whole input. **10 (spec, metric) pairs across 7 specs came back**,
    and they do NOT all mean the same thing — the sweep's value is the
    classification, not the count:

      spec     metric                        mean       std      bar     worst adm
      PG.4     icm_dwell_share             0.66667   0.47140   >=0.40    -3.4e-07
      PG.4     dwell_margin                0.60527   0.44544   >=0.25    -0.024676
      PG.4     panel_reward_ratio          6.411e8   4.534e8   >=2.0     -71730
      PG.4     rays_on_panel_while_dwell   7.47667   5.28680   > 0       -1.5e-06
      ME.10    skill_gain                  0.37037   0.09442   >=0.25     0.236832
      PS.02    shuffled_r2                -0.18109   0.19870   <=0.05     0.099947
      T2.08    coverage_margin             0.05440   0.01873   >=0.05     0.027908
      LG.01    retained_min_per_category  23.00000   2.16000   >=20      19.944900
      T3.01    ref_min                     0.44670   5.6e-17   >=0.38     (std==0)
      W0.DIAG  jit_delta_up                0.02177   0.00227   > 0.0      0.018563

    FOUR CLASSES, and two of them correct the finding as ordered:

      BROKEN, PROVEN — **`PG.4`, and only `PG.4`.** Finding 3 below. The
        ordered text's headline ("the measured breakage cost of arm (b) is ONE
        spec") SURVIVES the sweep.
      THE RECORD CANNOT ANSWER — **`ME.10` AND `PS.02`, not `ME.10` alone.**
        `ME.10 skill_gain` is as ordered (worst admissible 0.236832 vs bar
        0.25). **`PS.02` was excluded in the ordering as "idiom 2" and the
        exclusion does not hold:** `ps_02_cold_is_felt.py:420` builds
        `seed_gates_ok` out of `cold_censored`, `censored_explained`,
        `death_s_min/max`, `law_dev`, `warm_delta_c` and `warm_deaths` — the
        RIG gates — and `shuffled_r2` is NOT among them. The shuffled-control
        conjunct is still gated on the cross-seed mean, and a seed at +0.0999
        against `SHUFFLED_R2_MAX` 0.05 is admissible by the record. Idiom 2
        protects PS.02's rig, not its claim. One re-run settles each.
      PARTIALLY PROTECTED — **`T2.08`, which the ordering recorded as safe.**
        Idiom 3 is real and it is there (`margin_floor = coverage_margin -
        SEED_SPREAD_FACTOR*margin_std`, plus a paired t-stat), but it gates
        `margin_floor > 0.0` — POSITIVITY, not clearance. The separate
        conjunct `coverage_margin >= MARGIN_MIN` is still a bare mean, and its
        worst admissible seed is 0.027908 against `MARGIN_MIN` 0.05. So idiom 3
        as deployed buys a weaker guarantee than its presence suggests, which
        is itself an argument for arm (b) over a convention.
      SAFE, AND WORTH NAMING SO THE NEXT SWEEP DOES NOT RE-FLAG THEM —
        `LG.01` by integer arithmetic exactly as ordered (any integer
        >= 19.9449 is >= 20; the admissible multisets are {20,24,25} and
        {21,22,26}, both clearing `RETAIN_MIN`); `T3.01` `ref_min` whose std is
        5.6e-17, i.e. floating-point zero — it is idiom 1, the worst seed
        FOLDED INTO the metric before aggregation, which is the fix already
        applied by hand; and `W0.DIAG` `jit_delta_up`, which gates a t-statistic
        built from the std rather than the mean alone. **`T3.01` and `W0.DIAG`
        appear above only as DIRECTION ARTIFACTS of a naive scan** — both are
        written as `if <bad condition>: VOID`, so the failing tail is the
        opposite one from the comparison operator; any future implementation of
        arm (c) must handle that inversion or it will report two false
        positives on its first run.

    **FINDING 3, RE-DERIVED RATHER THAN TRANSCRIBED — and one leg of it as
    ordered does not hold.** The ordered text says `rays_on_panel` and
    `panel_reward_ratio` "carry the `{a,a,0}` signature ON THE SAME SEED".
    That is a PAIRING claim, and `docs/LESSONS.md` gained an entry on 2026-09-12
    (`ab0544f`) saying a mean+-std can pin a multiset exactly and say nothing
    about the pairing. It applies here, so each leg was tested separately:

      PROVEN, from the row alone. `icm_dwell_share` is `{1, 1, 0}` exactly.
        `Sum x = 3*0.666667 = 2` and `Sum x^2 = 3*(sigma^2 + mu^2) = 2` agree to
        1.7e-06, so `Sum x(1-x) = 0`; every term is non-negative on [0,1], so
        every seed is 0 or 1, and the sum forces two ones and a zero. This one
        does not need the pairing — the [0,1] BOUND does the work, and no other
        metric in the table has it.
      PROVEN INDEPENDENTLY, from a DIFFERENT metric pair. If dwell is
        `{1,1,0}` then `dwell_margin` (= dwell - null, per seed) must sum to
        `2 - Sum(null_dwell_share)`. Measured: `Sum dwell_margin` = 1.815801
        and `2 - Sum null` = 1.815800. Agreement to 1e-06 across two metrics
        that were aggregated separately. The zero-dwell seed's margin is
        `-null_2`, i.e. NEGATIVE.
      NOT A SIGNATURE, BUT A DEFINITION — and this is STRONGER than what was
        ordered. `pg_4_noisy_tv.py:283` computes `rays_on_panel_while_dwelling
        = panel_hits_dwell / max(1, dwell_steps)`. A seed with zero late dwell
        has `dwell_steps == 0`, so the metric is `0/1 = 0` BY CONSTRUCTION. No
        statistical inference is needed and none should be offered: the
        conditioning makes the zero mandatory, not merely consistent.
      REFUTED AS STATED, WITHOUT DISTURBING THE CONCLUSION. The exact
        `{a,a,0}` signature requires `sigma/mu = sqrt(2)/2 = 0.7071068`.
        `rays_on_panel` reads 0.7071068 (fits); **`panel_reward_ratio` reads
        0.7071860 and `late_reward_in_zone` the same — a 7.9e-05 discrepancy,
        far outside 6-significant-figure rounding.** Solving `{a, b, 0}`
        instead: `panel_reward_ratio` is `{9.700e8, 9.534e8, 0}`,
        `rays_on_panel` `{11.219, 11.211, 0}`, `late_reward_in_zone`
        `{0.96999, 0.95341, 0}`. Two NEARLY-equal seeds and a zero, never two
        exactly-equal ones. The conclusion is untouched and the arithmetic is
        now right.
      CORROBORATED BY AN INDEPENDENT SOURCE, which is how the pairing is
        legitimately known at all: commit `4a4afb3` (2026-08-10) states in its
        own message *"per-seed dwell is (1.0, 1.0, 0.0) — one seed never
        discovered the panel in its 20k-step life"*. The row cannot establish
        the pairing; the disclosure can, and it agrees. That asymmetry is
        exactly the defect this row exists to close.

    **AND THE CONJUNCT COUNT IS CONFIRMED: four of five.** On the zero-dwell
    seed, `pg_4_noisy_tv.py:322`'s five experiment conjuncts read
    `icm_dwell_share` 0.0 vs `>= 0.40` FAIL; `dwell_margin` negative vs
    `>= 0.25` FAIL; `panel_reward_ratio` 0 vs `>= 2.0` FAIL;
    `rays_on_panel_while_dwelling` 0 vs `> 0` FAIL; `null_dwell_share` passes
    (it is the null arm's own reading and is small on every seed). **The row
    says PASS and four specs depend on it.**

    FINDING 1, as ordered and now demonstrated rather than asserted: the sweep
    is runnable TODAY from the ledger and the spec files alone and needs no new
    machinery — it was run to produce this table. That is a live argument for
    arm (c), whose cost was recorded above as "needs an AST pass and will have
    false positives": the AST pass took minutes, and the false positives are
    now enumerated and explained (`T3.01`, `W0.DIAG`, and the operator
    inversion that produces them). Arm (c) is cheaper than this row priced it.
    **It is NOT a substitute for arm (b)** — a static audit reports, and only
    the recorder can make the wrong gate impossible to write.

    **A THIRD CASE ARRIVED ON ITS OWN, AND IT SPLITS THIS ROW'S QUESTION IN TWO
    (builder, 2026-09-13, harvesting `PL.02` attempt 1 — no new row, no new
    date, `net_arrivals` unmoved).** `PL.02` VOIDed on `learn_ok`
    **0.666667 ± 0.471405**, and that aggregate did exactly the right thing:
    `learn_ok` is a per-seed BOOLEAN and a mean of booleans compared against
    `< 1.0` is a true conjunction over seeds — the mean cannot hide a failing
    seed, it reports one. **So this row's defect is not universal, and the
    boundary is worth recording beside the counterexamples: booleans-as-
    conjunctions aggregate CORRECTLY under `_aggregate`; continuous worst-case
    quantities do not.** Any recorder arm should preserve that, because
    refusing to flatten `*_ok` keys would break a pattern that is already sound.

    **But the SAME row then failed at something none of the three arms
    addresses: it could not NAME the failing seed or arm.** `learn_ok`
    quantifies over three arms (`U_A`, `PLASTIC`, `FROZEN`) and only two of the
    three ratios were emitted; `shuffled_learn_ok` gated a fourth and emitted
    none. This desk's own `sqrt(2)*std` bound recovered part of it from the
    record — `loss_drop_plastic` 0.4033 ± 0.0769 (worst admissible 0.5120) and
    `loss_drop_ua` 0.0085 ± 0.0030 (worst 0.0127) both clear `LEARN_DROP` 0.90
    on every seed, so the implicated arms are `FROZEN` and `SHUFFLED` — **and
    that is where the bound runs out, because the two implicated arms are
    exactly the two whose ratios were never recorded.** Attribution was bought
    by a code change and a re-run (`7ffd3c8`, disclosure only: `loss_drop_frozen`
    and `loss_drop_shuffled` added, `LEARN_DROP` unmoved, attempt 2 predicted
    VOID with every other number identical).
    **`_min`/`_max` would not have helped here, and that is the point for the
    09-18 sitting:** the missing information was not the spread of a recorded
    metric, it was a metric that a GATE READ AND THE ROW NEVER STORED. Arm (c)'s
    AST sweep is the only one of the three that could find that class — a
    `_check` conjunct reading a quantity the spec does not emit — and it would
    have found it statically, before the 2,936 s run. Worth weighing when (c)
    is priced against (a) and (b).

---

## `t310-anticorrelated-gates` — a spec whose rig control and claim gate move in
## OPPOSITE directions under the same knob (builder, 2026-08-30; T3.10 PARKED)

ROUTED: t310-anticorrelated-gates | 2026-08-30 | 06c65f8 (T3.10 REPAIR pilots 1-2, seed 90, Colab T4) | OPEN
    DUE: 2026-09-06 | a design answer from the Review FULL run: what independent
        control certifies zero drift when phase A moves nothing, and whether
        +0.0299 is the ceiling of the question or of the substrate.
    DUE: 2026-09-07 | the same design answer, moved to the Monday DAILY —
        RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the
        zero-drift-control and bottleneck-headroom questions are about the
        frozen-vs-plastic substrate, with no coupling to the W0/W1 venue
        design the Sunday sitting owes; a daily run can carry it.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated.)
    DUE: 2026-09-11 | RE-DATED 2026-09-07 (Review, DAILY) — **second slip, and
        I am naming it as one rather than dressing it as sequencing.** The
        first re-arm (09-06 -> 09-07) was correct and was about venue coupling.
        This one is not: it is capacity. Four rows came due today against a
        desk whose measured throughput is ~1 disposal per cycle and whose drain
        reads UNBOUNDED (41 live, +34 arrivals against 2 disposals over the
        trailing week). I disposed three today — `t027` closed on `D16`,
        `pl02` ruled, and `me1-similarity-floor` pulled forward six days
        because it owns a FAIL this desk caused — and this row is the fourth.
        **The reason it is the one that waits:** its question 1 (what
        independent control certifies zero drift when a converged phase A moves
        an unfrozen trunk no more than a frozen one) is a genuine control
        design that needs the frozen-vs-plastic evidence re-read, not a pick
        among stated arms; question 2 asks whether +0.0299 at ~1.7σ is the
        ceiling of the question or of the substrate, and answering that from
        the two pilots alone would be exactly the kind of reading-off this desk
        refuses elsewhere. Dated onto 09-11, which carried one live row, rather
        than onto 09-13's pile of ten. **Nothing about `T3.10` moves in the
        meantime: it stays PARKED, the one-diagnostic cap stays SPENT, no third
        recipe, and the 0.15 `knowledge_margin_min` bar does not move.**
    DUE: 2026-09-20 | THIRD SLIP, and I am repairing the pattern rather than
        the date. Two of the three slips (09-07, and this one) are CAPACITY,
        and the note above already conceded that this row's question 1 *"is a
        genuine control design that needs the frozen-vs-plastic evidence
        re-read"*. **A FULL-sized design question has now been dated onto a
        DAILY three times and has failed to be carried three times.** That is
        not four consecutive bad mornings; it is a row filed against the wrong
        kind of sitting. So it goes onto a **FULL**, which is the sitting whose
        budget includes re-reading evidence — 09-20 and not 09-13, because
        09-13's FULL already carries 14 rows against a measured capacity of 6
        and this desk said a week in advance that it will not clear. **This is
        a longer slip than either previous one and I am not dressing it as
        sequencing: the honest choice is one date this row can actually be met
        on, rather than a fourth short date that breaks.** The cost of the
        delay is bounded and stated: `T3.10` is PARKED, nothing is blocked
        behind it, and every bar named above stays exactly where it is.
    DUE: 2026-09-30 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. THIRD slip, named as one. The first re-date was about venue coupling and was correct; the second and this one are capacity. The question (what certifies zero drift when phase A moves nothing) is unchanged and nothing is held behind it, which is the only reason it ranks below the rows above.

**Routed here by the spec's own pre-registered fork (ii), not by an argument.**
The one-diagnostic cap (SM.02/UB.10 precedent) is SPENT: no third recipe was
tried and none may be. Full receipts in `t3_10_trunk_knowledge_survives.py`'s
REPAIR 1 PILOT block and in the registry's `PARKED:` marker.

**What was measured** (Colab T4, seed 90, ~9 min, head `06c65f8`). Both
pre-registered repairs did exactly what they were specified to do:

| | pilot 1 (EPOCHS_P 40) | pilot 2 (EPOCHS_P 150) | gate |
|---|---|---|---|
| `final_perception_loss` | 2.2246 | **1.4244** | (chance 3.4655) |
| probe `shape` after P | 0.3633 | **0.4492** | random trunk 0.4193 |
| `knowledge_margin_min` | unsatisfiable by arithmetic | **0.0299** | **≥ 0.15** |
| `probe_drift_unfrozen` | **0.1875** | **0.0078** | **≥ 0.10** (rig) |
| `reach_margin` | 0.1576 | 0.3138 | ≥ 0.10 ✓ |

`null_admissible` (REPAIR 1(b)) worked: colour and near dropped as unreadable
from a random trunk, `shape` retained, `n_null_admissible` 1.

**The question for the Review, and it is a DESIGN question.** Converging phase P
made the claim measurable and simultaneously killed the control — because a
converged trunk is one whose features phase A's gradients no longer move,
frozen or not. The control's sensitivity was a side-effect of the apparatus
being under-trained. So:

1. **What independent control certifies the frozen arm's zero drift** when
   phase A no longer moves an unfrozen trunk either? Without one, this spec
   cannot distinguish "the freeze held" from "nothing was going to move
   anyway", at any `EPOCHS_P`.
2. **Is +0.0299 the ceiling of the question or of the substrate?** The signal is
   real and correctly signed — a 128-d globally-pooled bottleneck *does* learn
   something about shape its random init cannot read — at ~1.7σ (n_test 768)
   against a bar of 0.15 that does not move and must not. Candidate arms: a
   larger bottleneck, a relational/compositional probe target (colour and
   apparent size are low-order statistics that survive any random projection),
   or a task where the margin can be large by construction.
3. **Does this generalise to the other frozen-vs-plastic specs?** The failure is
   about the *bottleneck's* representational headroom, and `PL.00`/`PL.02` are
   named in `CHAMPIONS.md` as the arenas for the plastic-only decree. If the
   answer to (2) is "the substrate", they inherit it.

**One retraction to carry forward.** Pilot 1's headline second finding —
*"supervised training made the seated 245K trunk a worse linear feature
extractor on all three targets"*, written up as corroborating `T2.03` from the
opposite direction — was an **under-training artefact** and is withdrawn. At 150
epochs shape goes above random, not below. Do not cite it.

---

## ROUTED 2026-08-30 (builder): `SM.03`'s held-out split is saturated — pick the
## repair arm, do not let me pick it

ROUTED: sm03-heldout-split-saturated | 2026-08-30 | 13c0440 (pilot /data/sm03_pilot_seed90.json) | DISPOSITIONED 2026-09-12 (Review DAILY — the arm pick is REFUSED, not slipped: all three offered arms act on F1's split geometry and NONE of them acts on F2, whose failure VOIDs the run whichever arm wins. F2 is promoted from rider to blocker and answered by a MEASUREMENT the builder owes, not a pick this desk owes. See RULING below)
    DUE: 2026-09-06 | the Review picks the repair arm — the author must not; and
        F2 (the dead alive-proof) needs its own answer whichever arm wins.
    DUE: 2026-09-07 | the same owed pick, moved to the Monday DAILY — RE-ARMED
        2026-09-02 from 2026-09-06 (61st audit B2, builder): the fault is
        split-geometry arithmetic (94.2 m² of exclusion asked of an 11.06 m²
        annulus) plus a dead alive-proof, not venue shallowness — all three
        arms are runnable whatever the W0/W1 design says, so this decision
        does not need the Sunday FULL sitting and should not compete with it.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated.)
    DUE: 2026-09-12 | RE-DATED 2026-09-07 (Review, DAILY) — **second slip,
        named as capacity and not as sequencing.** Same arithmetic as `t310`
        above: four dated rows landed on one DAILY against ~1 disposal per
        cycle; three were disposed and this is the fourth. **Why this one and
        not another:** the row's own text says the author must not pick the arm
        and that arm 3 (hold out by BEARING SECTOR) *"is the biggest change to
        the pre-registered claim, so it is the one I am least entitled to make
        alone"* — that is a correct instinct and it means the pick needs the
        odour-field arithmetic re-derived, not a preference stated. It also has
        a second fault (`vis_open` 0.1167 against a 0.60 floor) that the row
        itself says may not be downstream of the first, and picking F1's arm
        without an answer for F2 would order a run that VOIDs on the alive
        proof exactly as the pilot did. **Nothing moves meanwhile:**
        `_GATES_FROZEN` stays False, `run()` keeps refusing, no seeds are
        spent, and `MIN_SEP_M` / `VIS_OPEN_MIN` do not move in either
        direction. Dated onto 09-12 (one live row) rather than 09-13 (ten).
        **The cost of the wait, stated:** `SM.02` is PARKED with `SM.03` as its
        stated revival path, so this row is one of the three
        `PARK-ON-AN-UNREACHABLE-RELEASE` pairs `coverage` prints, and smell
        stays a commitment with a spec and no measurement for five more days.

    DUE: 2026-09-22 | RE-DATED 2026-09-15 (Review DAILY), BEFORE the break rather than after it, and that is the point. This row is the BUILDER's execution debt and the builder has been PACE-DARK for 18 consecutive hourly slots since 2026-09-14T12:07 ('week:all models' 37% against a 35% pace line; this project's own attributed share of that meter is 25% — 28 of 37 points, 75%, are NOT THIS PROJECT). Leaving a builder-owned row dated on a day the builder provably cannot run is KNOWINGLY MANUFACTURING a violation, and re-dating it at 07:00 with a measured cause is strictly more honest than re-dating it at 07:00 tomorrow with the same cause and a red ratchet in between. Dated AFTER the 2026-09-21 05:00 UTC meter reset. This is the desk declining to let its own calendar launder someone else's outage. Routed to the owner as `D30`. ORIGINAL TEXT FOLLOWS, unchanged. | RULED 2026-09-12 (Review, DAILY) — the owed unit CHANGES
        HANDS AND KIND: what is owed on 09-15 is the builder's F2 diagnostic
        probe, not this desk's F1 arm pick. Dated 09-15 (5 live rows, measured
        capacity 6) and not 09-13 (14 rows). Design in the RULING below.
    DUE: 2026-09-30 | DISPOSED 2026-09-22 (Review DAILY) — the 108th audit's RANK 3 caught this one three minutes after I committed a pass that routed me away from it, and it is right: D28's (a) OVERDUE FIRST looks only at rows that have ALREADY broken, so a binding stop-rule falling due TODAY was outside its field of view. Acting on it inside the same sitting. The owed act is an arm PICK this desk REFUSED on 09-12 rather than slipped, and a refusal that is never revisited is a slip with better manners. Re-dated once, onto a day carrying 3 live rows, and ranked below the A4-seat convergence because nothing is held behind it. If 09-30 breaks this becomes a third break and the desk should DECLINE it to the owner rather than renew it a fourth time.

**Status: DISPOSITIONED. Gates provisional, `run()` still refuses, nothing
dispatched, `_GATES_FROZEN` still False.**

**RULING, 2026-09-12 (Review, DAILY). The arm pick is REFUSED on ordering
grounds, and I am naming this as the third dated promise on this row rather
than letting a third slip look routine — but it is a refusal with a reason and
a state change, not a fourth date.**

**The argument, from this row's own recorded numbers and nothing else.** The
pilot carries two faults, and this row has treated F1 as the decision and F2 as
a rider since 2026-08-30. That ordering is backwards:

| | fault | measured |
|---|---|---|
| F1 | held-out split saturated | 94.2 m² of exclusion discs asked of an 11.06 m² annulus (8.5× oversubscribed); reject 0.9958; every retained test position exactly at the `MIN_SEP_M` 0.25 floor |
| F2 | the alive-proof is dead | `vis_open` **0.1167** against `VIS_OPEN_MIN` 0.60, with **chance at 0.125** |

> **`vis_open` is BELOW chance.** It is the open-condition visual baseline — the
> arm the nose is compared against. The registered run would be VOID by this
> spec's own tree on F2 regardless of what F1 does, and `vis_occ` proves nothing
> about occlusion while it stands.

**The three arms on offer are all F1 arms, and each moves F2 the wrong way or
not at all:**
  - **shrink `N_TRAIN_L`** — fewer training positions; strictly *lowers* the
    visual baseline's ceiling.
  - **widen `SRC_R_RANGE`** — a different annulus, hence a different visual
    task; effect on `vis_open` unknown and unmeasured in either direction.
  - **hold out by BEARING SECTOR** — the largest generalisation demand of the
    three; if anything it *lowers* `vis_open` further.

**So there is no arm among the three whose selection produces a valid run, and
the only move that would "fix" F2 by choosing is lowering `VIS_OPEN_MIN` —
which the one law forbids and which I am not entitled to do even if it were
wise.** Picking today would be ordering a run that VOIDs exactly as the pilot
did, on a spec whose parked predecessor (`SM.02`) names it as the sole revival
path. That is the honest reason, and it is stronger than the capacity reason
that carried the two previous slips.

**WHAT IS ORDERED — a probe, explicitly NOT a pilot.** `coverage` marks `SM.03`
PILOT-BLOCKED and its own repair note says *"not another pilot"*; that binds.
The unit is a **scratch diagnostic in the `lg03_blind_twin_probe.py` idiom** —
kept in `experiments/tests/`, outside the ledger, spending no seeds, writing no
row, freezing no gate, moving no constant:

  `experiments/tests/sm03_vis_open_probe.py`, answering ONE question with a
  number: **is `vis_open` at chance because the visual observation carries no
  usable signal at this geometry, or because the retained test set is too small
  and too degenerate to measure one?** Report, at the CURRENT geometry and on
  the pilot's own seed: `n_test` retained, the per-class confusion in the OPEN
  condition, and `vis_open` recomputed on a split built WITHOUT the `MIN_SEP_M`
  exclusion (an instrument-only reading, never a bar to pass).

**That third number is the discriminator and it is why the probe is worth the
credits:** if `vis_open` rises to well above 0.125 once the saturated exclusion
is lifted, F2 is a SYMPTOM of F1 and the arm pick becomes a real pick that this
desk can make on the 09-15 sitting. If it stays at chance, F1 is cosmetic, the
smell fixture has no live visual comparison arm at all, and the repair is a
venue redesign that routes to `w0-too-shallow` — a much larger answer than any
of the three arms, and one nobody should reach by elimination.

**Nothing moved, in either direction:** `MIN_SEP_M` 0.25, `VIS_OPEN_MIN` 0.60,
`N_TRAIN_L` 480 and `SRC_R_RANGE` are all untouched; `_GATES_FROZEN` stays
False; `run()` keeps refusing; the pilot stays spent evidence and is not re-run.
**The cost of the wait is unchanged and still real:** `SM.02` is PARKED with
`SM.03` as its stated revival path, so this stays one of `coverage`'s three
`PARK-ON-AN-UNREACHABLE-RELEASE` pairs and smell stays CLAIM-DEAD meanwhile.
I am buying three more days of that to avoid ordering a run that cannot count.

**PROBE RESULT — the ordered F2 diagnostic is DELIVERED (builder, 2026-09-13,
two days ahead of its 09-15 date). `experiments/tests/sm03_vis_open_probe.py`
(committed `8b6480a` BEFORE the run, with its reading branches declared),
artifact `/data/sm03_vis_open_probe.json`, seed 90, CPU, 141.2 s. No seeds
spent, no ledger row, no gate frozen, no constant moved, `_GATES_FROZEN` still
False, `run()` still refusing — and the arm pick is still this desk's, untaken
here.**

**NEITHER LEAF OF THE RULING'S TREE FIRED. The cause is the READOUT, and a
venue redesign would have been a misroute.**

Self-validation first, because nothing else counts without it: the probe rebuilds
the pilot's split from the spec's own `_draw_layout` with the spec's RNG seeding,
open frames only, and `cnn_vis_open_excl` reads **0.1167** against the pilot's
recorded **0.1167** — the same object, to the last digit.

| # | ordered | measured |
|---|---|---|
| 1 | `n_test` retained | **240 of 240**; reject rate 0.9964 with the exclusion vs **0.2258** without (the latter reproduces the pilot's occlusion-only 0.2405) |
| 2 | OPEN-condition confusion | prediction histogram **[240, 0, 0, 0, 0, 0, 0, 0]** — a COLUMN, not a diagonal |
| 3 | `vis_open` without `MIN_SEP_M` | **0.1042**, against 0.1167 with it — it goes DOWN |
| 4 | *(extra)* train fit on its own 480 rows | **0.1646** (chance 0.1250) |
| 5 | *(extra)* spatially-explicit ridge reference | train **1.0000**, held-out-with-exclusion **0.9917** (238/240), no-exclusion 0.9833 |

**Three things follow, and the first two dispose of the question this row has
carried since 08-30:**

1. **"The test set is too small" is refuted by construction, not by
   measurement.** `_build_split` redraws until it has `n`, so retention is
   fixed at 240 whatever the exclusion does. The exclusion does not buy fewer
   rows; it buys worse ones — median nearest-training-position **0.2822 m**,
   max 0.3483 m, against 45° bins subtending ~1.7 m of arc at 2.2 m.

2. **`acc_vis_open` 0.1167 was never a chance-level discrimination.** Bin 0's
   base rate in that split is 28/240 = 0.11666…, and the readout assigns
   **every** row to bin 0. The number the gate compared against `VIS_OPEN_MIN`
   is the base rate of whichever bin a CONSTANT predictor emits — it contains no
   information about the venue at all. (Identically on the no-exclusion split:
   [240, 0, …], 0.1042 = 25/240.)

3. **The venue is the most measurable thing in the rig.** A closed-form LINEAR
   ridge on 4×-pooled raw pixels (λ swept on the spec's own 1-in-5 validation
   split) recovers the bearing at **99.17% on the very held-out split the pilot
   called saturated**, and the source is visible on every single layout (7.12
   source-coloured pixels per panorama on average, min 2, zero on none).

**THE MECHANISM, algebraic and measured, because the measurement alone cannot
separate "cannot represent" from "cannot optimise".** `_make_cnn` ends
`AdaptiveAvgPool2d(1) → Flatten → Linear(64, 8)`: it GLOBALLY AVERAGES the 8×8
feature map before the classifier, while the label is the source's BEARING — a
purely spatial quantity. A global average is translation-invariant, so two
layouts differing only in where the ball sits (which is what differing bearing
bins ARE) map to near-identical pooled vectors. The one cue that survives is
which of the 4 channel-concatenated frames holds the ball: 4-way quadrant
information for an 8-way label, ceiling ~0.5. Measured, not even that survives,
and the dilution says why — ~7 source pixels in 4,096 is ~0.17% of the average
the classifier reads. The 0.1646 train fit is the same statement from the data
side: a 28,472-parameter CNN does not fail to overfit 480 rows in 40 epochs
unless it cannot express the mapping.

**WHAT THIS DOES AND DOES NOT DECIDE FOR THE 09-15 SITTING.**

- **It does not pick the arm.** That is this desk's and the probe took no step
  toward it. What it does is bound the pick: **none of the three offered F1 arms
  touches the cause** — which is the same shape as this desk's own 09-12
  finding, reproduced one level further in.
- **It prices a second-order effect the pick must not miss.** `vis_occ` and
  `vis_open` share `_make_cnn`. Any repair to the readout applies to the
  OCCLUDED arm too, so it makes `vis_occ <= VIS_OCC_CEIL` **harder** to satisfy,
  not easier. That is the legal direction under the never-weaken law and the
  direction that makes the occlusion claim mean anything — but the repaired rig
  must be expected to be a STRICTER test of the claim, not a friendlier one.
- **F1 is not exonerated; it is independently confirmed from the other side.** A
  99.17% held-out reading on a split whose median test position is 0.28 m from a
  training position is exactly what "not a held-out sample of the source band"
  looks like once a competent readout is pointed at it. **F1 and F2 are two
  faults, not one.** F2's cause is the readout; F1 remains a real defect in what
  the claim would be measuring after F2 is fixed, and it is still unanswered.
- **And the reading this row should carry forward about itself:** F2 was called
  a *rider* for thirteen days and promoted to *blocker* on 09-12; it turns out
  to have been neither a rider nor a split problem but a defect in the
  instrument every side of this row treated as the neutral part. The
  generalisation is in `docs/LESSONS.md` (2026-09-13).

The full-size seed-90 pilot ran on CPU in 8 minutes (`/data/sm03_pilot_seed90.json`,
head `13c0440`) and found two faults; the numbers and the arithmetic are in
`sm_03_nose_reports_occluded.py`'s PILOT section and in `LESSONS.md`. In short:
`MIN_SEP_M` = 0.25 against `N_TRAIN_L` = 480 asks for up to 94.2 m² of exclusion
inside an 11.06 m² annulus, so the held-out set is the residue of a saturated
domain rather than a sample of it (occlusion assert alone rejects 0.2405;
with separation, 0.9958). And the alive-proof leg came back at chance
(`vis_open` 0.1167 vs a 0.60 floor), so the registered run would have been VOID.

**The question for the Review: which repair, and by what evidence?** Three arms,
all runnable on CPU, none obviously dominant — which is exactly the shape
`SM.02`'s three-mechanism-repair park says must not be settled by argument:

1. **Shrink `N_TRAIN_L`** until the exclusion budget fits. Cheapest, and it cuts
   the training rows the vision alive-proof may already be starved of — the two
   faults pull in opposite directions, which is the interesting part.
2. **Widen `SRC_R_RANGE`**. Buys area, but changes the odour problem: source
   distance is the dominant term in the field, so the arms are no longer being
   compared on the same difficulty as the pilot measured.
3. **Hold out by BEARING SECTOR rather than euclidean distance.** For a
   direction task this is arguably what "held-out" should have meant all along,
   and the exclusion budget stops scaling with the training count. It is also
   the biggest change to the pre-registered claim, so it is the one I am least
   entitled to make alone.

Whichever wins, F2 (the dead alive-proof) needs its own answer and may not be
downstream of F1 at all: 480 rows for a CNN on 12×64×64, and a 0.12 m ball at
1.8–2.6 m under a 90° fovy at 64×64 (~4 px), are both live suspects and neither
is measured.

---

## ROUTED 2026-08-30 (builder): should a PRESERVED failing implementation count
## as `audit_supersedes_fail`'s artifact? I built the mechanism and deliberately
## did not answer this

ROUTED: t027-preserved-failimpl-as-artifact | 2026-08-30 | 7ffd961 (preserve_impl_bytes mechanism) | ACTED 2026-09-07 in 0e60ac1 (the builder firing D16's armed default at 2026-09-06 00:14 — option (b) ALONE: T0.27 stays RED, the guard is unedited, the visible failure kept over the exonerating green. The executing commit records a deliberate NO-OP as chosen-by-default rather than skipped-by-neglect, which is the act; see the ACTED: body below for the correction that put this hash here)
    DUE: 2026-09-05 | `D16` (armed default, docs/DECISIONS_NEEDED.md) fires and
        the owner's answer disposes this row; the gate is the owner's, not the
        Review's and not mine.
    Second data point (2026-09-02, added at migration per 60th audit B5): the
    question now has one counter-example in each direction. `LG.00`'s failing
    bytes ARE preserved and cryptographically verified at
    `refs/jack/failimpl/LG.00/2026-08-30T18-47-59`, so of the two live `T0.27`
    violations one is recoverable and one (`T0.17`) is not — and
    `audit_supersedes_fail` reports both with the same sentence, "that
    implementation was never committed", which is true of only one of them.
    The gate itself is not changed here — that is `D16`.
    Third data point (2026-09-02, per 62nd audit B1; every number below
    re-verified by the builder against the live ledger and refs before this
    line was written): the live count is now **3 violations, 8 checked pairs,
    24 unauditable** — the third is `T0.29 FAIL 661a48f+dirty
    2026-09-02T09:18:06`, created by the 61st audit's own B4 work, and its
    failing bytes are preserved and verified at
    `refs/jack/failimpl/T0.29/2026-09-02T09-18-06` (blob `facfff9`). So
    recoverable is now the MAJORITY: two of three (`LG.00`, `T0.29`) admit the
    `git diff` the rule demands; only `T0.17` is genuinely lost — and
    `audit_supersedes_fail`'s one sentence, "that implementation was never
    committed", is now false for most of the rows it prints. Progression from
    `T0.27`'s own rows: **1 (08-29) → 2 (08-30) → 3 (09-02)**, three distinct
    specs, ~1 new violation per 1.5 days against a 20-entry history — they
    arrive FASTER than they age out, which is a measurement against option
    (a)'s "ages out soon-ish" premise. Full arithmetic: `D16`'s 62nd-audit
    evidence update in `docs/DECISIONS_NEEDED.md` (2026-09-02 12:44 UTC).
    No gate, default, or `decide_by` changed here.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated.)
    DUE: 2026-09-07 | RE-DATED 2026-09-05 (Review, DAILY) — **this row was dated
        one day BEFORE the event that disposes it, and that is the defect, not
        the slip.** The row's own gate line says the disposal is `D16`'s: *"the
        gate is the owner's, not the Review's and not mine."* `D16`'s
        `decide_by` is 2026-09-05, so its default cannot fire until the first
        slot of 2026-09-06 — after this date. No amount of Review work today
        could have closed it, and letting it break at midnight would record a
        violation against a promise that was unkeepable when it was written.
        Not re-dated onto 2026-09-06: that day already carries six live rows
        against a measured consumer capacity of ~1/cycle, plus `D21`'s default
        and the Sunday FULL's own Part 2 and two completeness audits. 09-07 is
        the first day on which the disposing event has already happened, and
        the disposal itself is a bookkeeping close (record what `D16`'s default
        did), not design work — so it does not compete with 09-07's own load.
    ACTED: 2026-09-07 (Review, DAILY) | executing commit **`0e60ac1`**
        (2026-09-06 00:14, builder — *"D16: option (b) ALONE — T0.27 stays
        RED, guard unedited, the visible failure kept over the exonerating
        green"*).
        **CORRECTION, recorded rather than quietly fixed:** I first stamped
        this row ACTED with no executing commit, and `review-queue` refused it
        (`ACTED-WITHOUT-A-COMMIT`, EXIT 2) inside the same run. My reasoning
        was that `D16`'s chosen option is a **no-op** — its own resolved entry
        says *"Execution: nothing. That is the option."* — so I felt there was
        no commit to name. **The instrument was right and the reasoning was
        wrong.** A deliberate no-op is still an act, and the commit that
        RECORDS a no-op as chosen-by-default rather than skipped-by-neglect is
        exactly the artifact this field exists to make findable. An ACTED row
        whose act cannot be pointed at is indistinguishable from a row somebody
        closed because they were tired of it. Named above.
        **Closed by the event this row was always waiting for, and closed with
        nothing done, on purpose.** `D16`
        fired by armed default at 2026-09-06 00:1x UTC: option **(b) ALONE** —
        the warning stands, `T0.27` stays RED, it is not re-run and not
        touched, and the red is reported in every `status` until the pair ages
        out of history (`docs/DECISIONS_RESOLVED.md` D16). The row's own gate
        line said the disposal was the owner's and not this desk's; the owner's
        armed silence chose the option that costs the ladder a visible failure
        rather than manufacturing a green, and there is nothing for the Review
        to add to that. **The row's THIRD data point survives the close and is
        not filed away with it:** violations arrived 1 (08-29) → 2 (08-30) → 3
        (09-02), ~1 per 1.5 days against a 20-entry history, i.e. FASTER than
        they age out — which is a live measurement against option (a)'s
        "ages out soon-ish" premise and therefore against the durability of
        (b) itself. `T0.27` reads `live_violations = 3, unchanged since
        2026-09-04T08:15:29` today, so the arrival rate has now been flat for
        three days and (b) is holding. If the counter resumes climbing, the
        thing to re-open is `D16`, not this row. **And the sentence the row
        was right about stays wrong in the code:** `audit_supersedes_fail`
        still prints *"that implementation was never committed"* for all three,
        when two of the three (`LG.00`, `T0.29`) have their failing bytes
        preserved and hash-verified under `refs/jack/failimpl/`. That is a
        one-sentence truthfulness defect in a standing-red instrument, it is
        not `D16`'s and it is not a threshold, so it goes to the builder as an
        item rather than holding this row open behind the owner's closed one.
        This is the same class the 70th audit's B1 shipped as
        `DEFAULT-ACTION-EXPIRED` — a default dated after the event it commands —
        arriving from the other side: a QUEUE ROW dated before the default that
        disposes it. `review_queue.py` has no reader for that direction; named
        here rather than routed as a new row, because the drain is the finding.

**Status: OPEN. No gate was moved. `T0.27` is still FAIL for its real reason.**

`run_spec` now archives the exact bytes of every `+dirty` FAIL/VOID into git's
object database (`preserve_impl_bytes`, ref under `refs/jack/failimpl`), because
`T0.17`'s 2026-08-29 failing implementation is provably unrecoverable and
`T0.27`'s live-ledger property is therefore permanently red. The mechanism
verifies what it stores: the ref is only written when the stored bytes re-derive
the `impl_sha` the row names.

**The question: `audit_supersedes_fail` currently accepts only a COMMITTED tree
state. Should a verified preserved manifest be a second lane?**

- FOR: the evidence is identical in kind and proven by the same function; a
  committed tree state is accepted because it reconstructs the sha, and so does
  this. `T0.27`'s title asks for an *artifact*, and `git cat-file -p <blob>`
  produces one. Without a reader, the mechanism prevents future loss but every
  future dirty pair still reads as a violation — the ledger accumulates
  permanent reds for breaches whose evidence actually survives.
- AGAINST: `T0.27`'s `kills` field names *the practice* of amending a FAIL from
  an uncommitted tree, not merely the loss of bytes. An automatic artifact makes
  the practice cheap, and cheap is how a discipline dies. The permanent red may
  be the deterrent working as designed.

I am the author of the mechanism, which makes me the wrong organ to rule on the
gate that would read it. Note the decision is not urgent and not blocking: the
bytes are being kept either way, so a later YES loses nothing, while a NO costs
only some disk in `.git`.

---

## ROUTED: OPEN — `sh02-null-saturation`: the born-inside geometry has no headroom, and the fix is an arm redesign
## (builder, 2026-08-30 11:33 UTC; pilot artifact `/data/sh02_pilot_seed90.json`, spec commit `8abfa70`)

ROUTED: sh02-null-saturation | 2026-08-30 | 8abfa70 (pilot /data/sh02_pilot_seed90.json) | DISPOSITIONED 2026-09-20 (Review FULL — option (b), the matched outward impulse at spawn; (a) and (c) refused on the record; VENUE repair, so execution is bound to `w1-world-edit-window`. THE STOP-RULE FIRED AND IS DISCHARGED BY A RULING, NOT A DECLINE. See THE BUNDLED RULING below)
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: this row's arm pick was RE-DATED to the
        2026-09-13 FULL precisely so it could be made IN LIGHT OF the W0/W1
        design, and that FULL did not take up W1. The premise the re-date was
        bought with therefore did not arrive, and the row goes OVERDUE at 00:00
        tonight for that reason. Full record on `w1-world-edit-window`.
    DUE: 2026-09-06 | the Review picks among arms (a)/(b)/(c) — re-pointing a
        registered null is a spec redesign under the T1.02 precedent, not a
        builder's edit.
    DUE: 2026-09-09 | the same pick, moved to the Wednesday DAILY — RE-ARMED
        2026-09-02 from 2026-09-06 (61st audit B2, builder): the saturation
        (every no-gradient arm holds its roof at 1.0000) is a venue property
        and one of the nine w0-too-shallow instruments — the null should be
        re-pointed IN LIGHT OF Sunday's W0/W1 design, three days later, not
        in the same sitting.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated;
    its heading carried the declaration one `## ` away from being read.)
    DUE: 2026-09-19 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. THIRD BREAK FOR THIS ROW (09-06 -> 09-09 -> 09-13 -> now), and it was bundled to a Sunday FULL that sat on 09-13 and did not reach it. STOP-RULE, binding on this desk: if this date breaks too the row is DECLINED and the finding is carried to the owner as a class, because a promise renewed four times is not a promise and a row nobody will ever rule on should not be occupying a clock. | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
    09-09). This row, `ba03-null-saturates-the-horizon` and
    `t306-matched-magnitude-noise-buys-coverage` are the SAME QUESTION wearing
    three spec ids: a null or anchor that saturates, so the gate cannot resolve
    the thing it was built to resolve. Yesterday's Review named that disease
    across four more fronts (UB.10's anchor at ceiling, w0-too-shallow's world
    that separates nothing, t309's venue where correct advice hurts, dp04's
    76.7% at the cap) and sent it to Sunday as ONE question about how this
    project chooses what to measure. Designing these three separately on a
    Wednesday is the act that would guarantee three incompatible local repairs.
    Declared cost, honestly: 09-13 already carried 10 rows against a measured
    capacity of 6, and these make it 13 ROWS — but ONE unit of design, and the
    FULL is the only sitting with the hours to take the general form.
    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL). THE DESIGN IS
        DELIVERED; what this date owes is EXECUTION by the builder, not a
        decision by this desk. **The STOP-RULE above FIRED** — the 2026-09-19
        date broke at midnight, the fourth break for this row — and I am
        recording that it fired and then DECLINING TO DECLINE, on the
        stop-rule's own stated ground: it exists because "a row nobody will
        ever rule on should not be occupying a clock", and ruling on it is
        strictly better than the remedy it was built to force. The stop-rule is
        DISCHARGED, not weakened, and is NOT re-armed, because after today this
        row is an execution debt, not a decision debt. The full ruling, the
        UNSATURATED-NULL RULE and the refusals of arms (a) and (c) are in THE
        BUNDLED RULING below.
        **PLACED INSIDE THE `DUE:` BLOCK 2026-09-20 ~07:0x, SAME SITTING, after
        the 106th audit's RANK 2 read this row as still OVERDUE. The audit was
        RIGHT and so was the instrument** — my first two attempts wrote this
        date below a `###` heading and then below a non-indented paragraph, and
        `review_queue.py` stops reading a row at the first unindented line. The
        ruling was real both times and its date was invisible both times, so
        `review_queue_violations` correctly refused to move. Same family as the
        09-09 scar where six DUE clauses sat above the line the tool takes:
        **the desk keeps writing the truth in a place the instrument does not
        look, and twice in one morning is not bad luck.** Nothing about the
        ruling changes; only its position does.

**The measurement.** `SH.02`'s seed-90 pilot (N=3000/arm, 6 arms, ~19 min)
fired the spec's own pre-registered `HEADROOM` VOID. Every arm without a live
policy gradient holds the roof it was born under **completely** — twin
`1.0000`, privileged oracle `1.0000`, both-cosmetic control `1.0000` — while
the learner reads `0.0136`. `headroom_twin` 1.0 against `HEADROOM_MAX` 0.85.
The z's (`z_shelter` −377.7, `z_need` −412.4) are zero-variance artefacts over
2 twin eval lives and carry no effect size; the LEVELS are the evidence.

**The rig is alive.** Random walk 0.3639 sheltered with 25 of 26 lives ending
FROZEN — huts are escapable, cold still kills. `warm_reward_abs` 0.0 exactly,
confirming live the arithmetic identity found symbolically at design time.

**Why it is yours and not mine.** The repair is a choice among three runnable
arms, so rule 3 governs and it is a bakeoff, not an escalation — but it changes
what the registry's declared NULL is, and re-pointing a registered null is a
spec redesign under the `T1.02` precedent, not a builder's edit:

- (a) score the contrast against the RANDOM walk (0.3639, real headroom)
  instead of the motionless twin;
- (b) give every arm a matched outward impulse at spawn, so "stay" costs
  something in every arm and the twin's zero advantage stops buying it 1.0;
- (c) score only lives in which the agent left at least once, making RETURN the
  measured quantity.

**The finding to carry regardless of which arm wins, because it outlives this
spec.** `SH.01` and `SH.02` now BRACKET the thermal-drive question and the two
geometries are exhaustive. Born outside: seeking is unlearnable — the field
beyond the hut is spatially flat and a privileged oracle sheltered in 0 of 27
lives (`ORACLE_CANNOT`, 08-25). Born inside: maintenance is unmeasurable — the
null saturates at 1.0 and there is nothing to be above. Both at reachable
envelopes, both with an oracle. That is a **fifth instrument** agreeing with
LC.03's darkroom control, LC.03 v2's one-learner-in-five, DP.05's FAIL and
SH.01's ORACLE_CANNOT that **W0, not the core, is the measured bottleneck** —
and it is `D10` evidence. Do NOT authorise an envelope growth: the pilot's
failure is not a budget.



### THE BUNDLED RULING — 2026-09-20, Review FULL. Read this block once; `ba03`
### and `t306` cite it rather than repeating it.

**The bundle's premise is HALF RIGHT, and finding that is the ruling's first
act.** These three rows were bundled on 09-09 as "the SAME QUESTION wearing
three spec ids: a null or anchor that saturates". Designing them separately
was said to guarantee three incompatible local repairs. Read together with
fresh eyes they are **two diseases, not one**, and treating them as one would
have guaranteed the opposite error — one repair applied where it cannot work.

- **Disease A — THE BOUNDED STATISTIC.** The claim statistic has a ceiling and
  BOTH arms reach it. `BA.03`: time-to-topple, twin at 11.868 of a 12.0 s
  horizon. `DP.04`: lifespan at 76.7% of its cap. `UB.10`: anchor at ceiling.
  The signature is that every legal repair inside the file — more seeds, more
  eval episodes, more budget — only SHRINKS the standard error and LOWERS the
  bar, and none of them raises the headroom, because headroom is arithmetic
  here and not noise. **Repair: change the STATISTIC to an unbounded one.**
- **Disease B — THE FREE-WIN NULL.** The statistic is fine; the VENUE hands
  the null the ceiling. `SH.02`: born inside the hut, staying still is free, so
  the motionless twin, the privileged oracle and the both-cosmetic control all
  hold 1.0000 while the learner reads 0.0136. Changing the statistic cannot
  help, because any monotone function of 1.0000 still saturates. **Repair: make
  the free action COST something, in every arm equally.**
- **`T3.06` IS NEITHER, and belongs in this bundle only by adjacency.** Its
  control did not saturate — it PASSED (`delta_shuf` +0.1072 +/- 0.0311 over
  DELTA_MIN 0.05 on every seed). Nothing is at a ceiling. The defect is
  ATTRIBUTION: matched-magnitude uninformative reward recovers the coverage,
  so the contrast cannot say the bonus's INFORMATION bought anything. That is
  the `t211` disease (a metric that cannot separate signal from matched noise),
  not the saturation disease. Ruling it as a saturation case is precisely the
  incompatible-repair the bundle was formed to prevent, arriving from the
  inside.

**THE GENERAL RULE, adopted today and binding on new registrations
(strengthen-only — it forbids something previously allowed, requires a
declaration that did not exist, and moves no threshold in any direction).**

> **THE UNSATURATED-NULL RULE.** A claim gate may not be REGISTERED on a
> statistic whose NULL or ANCHOR sits within the claim's own required margin of
> that statistic's bound. Every new claim spec must declare, in source before
> it runs, either `STATISTIC_BOUND: none` or the bound together with the null's
> measured or piloted distance from it. If the null is AT the bound, the gate
> is not registerable and the repair is a change of STATISTIC or of VENUE — it
> is never a change of ENVELOPE. Growing the envelope against a saturated null
> is the one repair that is arithmetically guaranteed not to work, and this
> project has now paid for that lesson on four rigs.

**AND THE SEQUENCING RULE THAT GOES WITH IT, because it is what actually
decides these three:** repairs are taken in ascending mechanical bill —
**statistic (zero) before scoring (zero) before venue (bills every
`playground.py` certificate) before envelope (forbidden above)**. A world edit
is the most expensive instrument this project owns and it is currently
UNDESIGNED (`w1-world-edit-window`, OPEN, +2 d). Nothing may be sent to it that
a statistic change can fix.

**RULING ON `SH.02` — option (b), THE MATCHED OUTWARD IMPULSE AT SPAWN.**

Disease B, so the statistic is not the lever. Rejecting the other two on the
record, because the rejections carry the reasoning:

- **(a) score against the RANDOM walk (0.3639) is REFUSED.** It does buy
  headroom, and it is the tempting cheap answer. But it changes only the
  COMPARATOR while leaving the born-inside geometry intact: the twin would
  still hold 1.0000, the oracle would still hold 1.0000, and the spec would
  then be claiming *"the learner shelters better than a wanderer"* — a claim
  whose own privileged oracle beats it and which therefore cannot be evidence
  that MAINTENANCE was learned. A contrast that leaves three degenerate arms at
  the ceiling and looks away from them is not repaired, it is re-aimed.
- **(c) score only lives in which the agent left at least once is REFUSED, and
  the reason is a defect worth naming.** It is cheap and it does remove the
  saturation — but it removes it by CONDITIONING THE SAMPLE ON THE BEHAVIOUR
  BEING MEASURED. The twin never leaves, so it contributes zero scored lives:
  the null is not beaten, it is DELETED, and a claim with no null is not a
  claim. This is the selection-effect shape the ladder exists to catch, and it
  would have entered through a repair.
- **(b) is ADOPTED.** Give every arm a matched outward impulse at spawn so
  "stay" costs the same in every arm. This removes the free win at its source
  rather than at the scoreboard: the twin's 1.0000 becomes impossible BY
  CONSTRUCTION, the oracle stays a real oracle, and the both-cosmetic control
  stays a real control. It is the only one of the three under which all three
  degenerate arms remain in the comparison and stop winning it.

**Its price, stated rather than buried: (b) is a VENUE repair and the
sequencing rule above therefore BINDS IT TO `w1-world-edit-window`**, which is
open and overdue — so this row's execution is blocked behind a window this desk
also owes. I will not launder that by picking the cheap arm instead. Declared
as `BLOCKED-BY: w1-world-edit-window` in substance; the date above is the
builder's execution date if the window lands first, and if it does not, this
row re-dates ON THE WINDOW, not on a guess.

**The one thing the builder MAY do meanwhile, and its limits.** `SH.02` may be
run under (a) as a **DIAGNOSTIC ONLY** — scored against the random walk,
reported in the run record, and **NEVER as the registered gate**. It costs CPU
this project has spare, it measures whether the learner clears a non-degenerate
comparator at all, and it tells us before the expensive window opens whether
(b) is worth the certificates it will bill. A diagnostic that is labelled a
diagnostic is not a re-pointed null. If it is run, the registered `HEADROOM`
VOID stands untouched and no ledger status may move on it.

**RULING ON `BA.03` — option (c), CHANGE THE METRIC. (b) is re-routed, not
dropped.**

Disease A. `claim_headroom_ratio` 0.236 +/- 0.184 against `HEADROOM_MIN_MULT`
2.0 is not a seed problem — the claim needs 1.336 s and the world has 0.132 s,
and the row's own arithmetic shows every in-file repair shrinks `gain_se` and
lowers the bar without touching that gap.

- **(a) RAISE THE HORIZON is REFUSED, and it is the important refusal.** It is
  an ENVELOPE growth against a saturated statistic, which the general rule above
  now forbids outright. Concretely: the twin survives to 12 s, so it will
  survive to 15 s, and the repair buys one horizon and re-arms the identical
  failure at the new cap — at full CPU cost, with a VOID at the end of it. The
  `sh02` row already declared "the pilot's failure is not a budget"; this makes
  that declaration general.
- **(b) HARDEN THE PERTURBATION is the right SCIENCE and the wrong ACT TODAY.**
  The spec's own ANATOMY table points at it — the winning vest policy reads
  PLANTAR TOUCH and nothing vestibular; deleting touch costs it 7.3 s, deleting
  any true vestibular block costs it nothing — so one kick per episode is
  survivable by a route that makes the graviceptive channel unnecessary, and
  that is the question BALANCE actually needs answered. But it touches
  `playground.py`, bills the 21 listed certificates plus `BA.01`, and belongs
  in the undesigned world-edit window. **It is re-routed as its own row**
  (`ba03-vestibular-channel-is-never-load-bearing-under-one-kick`), bound to the
  window, so that adopting (c) does not quietly retire the finding. A cheap
  repair that erases an expensive question is not a repair.
- **(c) is ADOPTED**, with the statistic named rather than left open:
  **INTEGRATED ABSOLUTE TILT over a FIXED 12 s window** (fall back to RECOVERY
  COUNT only if integrated tilt is shown degenerate in pilot, and say so in the
  record). It is unbounded above, it is defined for every life including the
  ones that never topple — which is exactly where time-to-topple threw its
  information away — and it carries no staleness bill: `HORIZON`, `N_EVAL` and
  the metric all live in `ba_03_braces_against_a_surface.py`, which no other
  certificate imports.

**STRENGTHENING BINDING ON THE `BA.03` REDESIGN, and it is not optional.** The
new bar is set from the RANDOM walk's measured distribution, not from the blind
twin — the twin is retained as a REPORTED arm but may no longer be the thing
the claim clears, because a null that survives to the horizon carries no
information about tilt. All SIX currently-green rig conjuncts are carried
forward UNCHANGED (random topples on 94.7% and survives 2.30 s; best trained
arm beats it by 9.56 s; no-surface control 0.0094 s against the 0.30 cap;
`gripboth` 4.29 s behind the twin; the noise control at `gain_noise` -7.011).
**A change of claim statistic may not drop a control that is currently
passing** — that is the line between a redesign and a rescue, and the T1.02
precedent puts this redesign on the legitimate side only because the
EXPERIMENT is demonstrably wrong: a 12 s ceiling cannot measure durability.

**RULING ON `T3.06` — (a) AND (b), BOTH, and NOT (c).**

Not a saturation case, as established above. Two independent defects, so two
independent repairs, and they do not interact:

- **(b) RE-DERIVE `RANDOM_DWELL_MAX` AS AN n-AWARE ORDER-STATISTIC BOUND —
  ADOPTED.** This is what actually VOIDed the run: `random_dwell_worst_life`
  0.0227 against a 0.02 cap frozen on a 16-life pilot and READ AT 48 LIVES. An
  extreme-value instrument whose exceedance grows with n by construction is
  measuring n, not the world. The replacement is an exogenous quantile of the
  ANALYTIC chance dwell at the read n. **It is explicitly permitted to come out
  LOWER at n=16 and HIGHER at n=48, and that is not a weakening** — it is the
  same bound correctly evaluated, and the direction is not to be chosen after
  seeing which way it falls. Derived in source before the run, from the
  analytic distribution, with the derivation in the file. This also discharges
  `aggregate-hides-worst-seed`'s instance on this row (the gate fired on a
  mean+1.5s bound over seeds; the actual worst seed is unanswerable from the
  aggregate at <= 0.0223) — **the new bound must be read against the ACTUAL
  worst seed, never an aggregate**. The parent row keeps its own general
  question.
- **(a) RESCORE AGAINST THE NOISE ARM — ADOPTED, WITH THE RANDOM-ACTION
  COMPARATOR BINDING, and I expect it to FAIL.** The new gate requires BOTH
  `cov(curious) - cov(shuftask) >= 0.05` (recorded-but-not-counting at +0.1385,
  t = 3.94 — ample) AND `cov(curious) - cov(random) >=` the C-RANDREW clearance
  `CURIOSITY_BAKEOFF.md` §O1 already demands (>= 1.5 vs the random-reward arm).
  Field watch wk5 measured `curious - random` at **+0.0124 +/- 0.0317,
  t = 0.39 — no clearance at all.** So the honest forecast is that T3.06 fails
  its redesigned gate, and **that is why the redesign is legitimate**: a
  conjunct is being ADDED that the spec is currently expected to fail, against
  a standard this project's own bakeoff document already wrote down and this
  spec was not being held to. The new gate is strictly HARDER than the old one
  — old: `delta_coverage` against a no-bonus null (green at +0.2458, 5.8 sigma);
  new: that, AND matched-magnitude noise, AND a random-action policy. Any
  redesign that beats only `shuftask` re-buys the same unattributable contrast
  and is refused in advance.
- **(c) THE WORLD ARM IS REFUSED.** (b) demonstrates the breach is instrument
  n-dependence, so spending the project's single most expensive instrument on
  it would be spending the world-edit window on the cheapest defect in the
  bundle. The goal-attractor question is not dismissed — if (b) lands and the
  ACTUAL worst seed STILL breaches an n-correct bound, then and only then is
  it a world question, and it is re-routed at that point with a measurement
  behind it instead of a suspicion.
- **AND THE `kills:` FIELD IS REPAIRED IN THE SAME MOTION.** As frozen, `_check`
  maps control-red to FAIL, which fires `kills: IntrinsicCuriosityModule` off a
  run whose own control says the instrument cannot attribute. A spec may not
  execute a capital sentence on the strength of a contrast it has just declared
  uninterpretable. Control-red must map to VOID, not FAIL. **This is a
  strengthening and not a rescue: it removes a FALSE kill, it cannot save a
  true one** (a green control with a red claim still FAILs and still kills),
  and it is the same defect `t211-diayn-metric-cannot-separate-mi-from-noise`
  routed one commitment over. Ordered on T3.06 only; `t211` keeps its own row.

**WHAT THIS RULING DOES NOT DO.** It does not design the world-edit window —
two of these three repairs now queue behind it and the third does not need it,
which is itself the argument for designing it, and that argument goes to the
owner rather than into another date on this page. It does not move a single
threshold downward. It does not touch `SH.02`'s, `BA.03`'s or `T3.06`'s
recorded VOID rows, all three of which stand exactly as run.

    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL). THE DESIGN IS
        DELIVERED and what this date owes is EXECUTION by the builder, not a
        decision by this desk. **The STOP-RULE above FIRED** — the 2026-09-19
        date broke at midnight, the fourth break for this row — and I am
        recording that it fired and then DECLINING TO DECLINE, on the
        stop-rule's own stated ground: it exists because "a row nobody will
        ever rule on should not be occupying a clock", and ruling on it is
        strictly better than the remedy it was built to force. The stop-rule is
        DISCHARGED, not weakened, and is NOT re-armed, because after today this
        row is an execution debt and no longer a decision debt. The same
        applies to `ba03-null-saturates-the-horizon` (DUE today) and
        `t306-matched-magnitude-noise-buys-coverage` (DUE tomorrow), both ruled
        in the same sitting below. All three carried the same stop-rule; none
        of the three is declined; the bundle is discharged on time for two of
        three and one day late for the third.
        **RE-PLACED 2026-09-20 ~07:0x, SAME SITTING, after the 106th audit's
        RANK 2 read this row as still OVERDUE. The audit was RIGHT and so was
        the instrument:** the `DUE:` line was written ABOVE a `###` heading,
        where `review_queue.py` stops reading the row — so the ruling was real
        and its date was invisible, and `review_queue_violations` correctly did
        not move. Same family as the 09-09 scar where six DUE clauses sat above
        the line the tool takes: **the desk keeps writing the truth in a place
        the instrument does not look.** Nothing about the ruling changes; only
        its position does.

ROUTED: w1-cold-is-not-lethal-at-night | 2026-08-30 | 487d5ea | OPEN
    DUE: 2026-09-20 | RE-ARMED 2026-09-08 (Review DAILY) from no date at all —
        STALE, OPEN 9 days past the 8-day cycle with nothing to re-arm it. This
        is a WORLD EDIT row (`needs.py` constants, `DELTA_T_NIGHT`, and W.3 is
        specced over the same constants), and the bundling rule binds world
        edits to ONE edit window. `w1-world-edit-window` is DUE 09-13; this row
        is decided IN LIGHT OF whatever window that opens, not beside it, so it
        lands after. Dated onto a day carrying no other promise rather than
        piled onto 09-13, which already carries ten.
    Question: at the world's OWN night ambient, cold carries no death
    gradient — so what is the curriculum GOAL.md promises actually made of?
    W.1 measured `needs.py`'s shivering loop
    `M_BASAL + C_SH*(37-T) = K_DRY*(T-T_env)` with C_SH = 33.33 W/C against
    K_DRY = 14.29 W/C. It parks the body at **34.000 C in a 20 C ambient,
    flat, forever**, and solving it for the world's own `T_COLD_DEATH = 28.0`
    gives a lethal ambient of **exactly 0.0 C**. The world's night is
    `T_DAY - DELTA_T_NIGHT` = 30 - 10 = **20 C**. A night in the open is
    therefore survivable indefinitely by a body that does nothing at all.
    This is DECLARED, not a bug: `needs.py` says "a night in the open
    equilibrates ~3.0 C cold ... survivable once, costly" (§2.3 pedagogy) and
    NE.01's assigned sweep calibrated DELTA_T_NIGHT 12 -> 10 to sit mid-band.
    W.1 does not overturn it; W.1 PRICES it, and the price lands on GOAL.md's
    "cold nights teach shelter-building the way no scripted lesson can".
    A quantitative account of SH.02's saturated null falls straight out: if
    the open night never kills, shelter has nothing to buy.
    SECOND, INDEPENDENT ROW ON THE SAME FILE — W0 HAS NO WIND AT ALL.
    `k_eff(skin_wetness, sky_occlusion)` takes no velocity and `wind` does not
    occur anywhere in `experiments/*.py`, so the shipped world is structurally
    identical to W.1's own deliberately-broken control on check (c): raising
    wind 0 -> 5 m/s changes its time constant by exactly nothing (ratio 1.0
    vs the physiological 0.3095). No policy can ever learn to seek a
    wind-break for being a wind-break. W0's shelter is NOT thereby decorative
    — it works through `sky_occlusion` cutting `k_eff` — but the wind
    affordance is absent, and W.3 is the spec that would price it.
    THIRD, and cheapest to act on: `TAU_T = 240 s` is the OPEN-LOOP constant
    (C_EFF/K_DRY) and the world relaxes with the CLOSED-LOOP one, measured at
    **72.0 s**, 3.33x faster, whenever the body is below 37 C — which at night
    is always. A published constant that is not the one the code exhibits.
    Arms are runnable and this is a redesign BAKEOFF, not an argument:
    (i) lower C_SH so shivering cannot outrun conduction; (ii) drop
    DELTA_T_NIGHT below the 0.0 C lethal ambient; (iii) add the wind term and
    let a windy night be the lethal one, which is the only arm that also buys
    the missing affordance. Note (iii) is the sole arm that makes shelter's
    insulation load-bearing rather than its occlusion.
    Full record: FINDINGS 2b/3/4 in w_1_heat_balance.py; ledger W.1 attempt 2.
    Staleness bill: **ZERO MECHANICAL — no PASS certificate cites `needs.py`
    in IMPL_DEPS (checked, 0 of 90).** This does NOT belong in the
    `playground.py` bundle above and must not be held behind it; the whole
    point of the bundling rule is that a computed bill lets rows be sequenced,
    and this row's bill is nil. SEMANTIC: NE.01 (FAIL) calibrated
    DELTA_T_NIGHT and W.2/W.3 are specced over these same constants — W.3 in
    particular is the registered instrument for the shelter question and
    should be implemented against whatever this row decides, not before it.
    DUE: 2026-09-25 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. WORLD EDIT, and the dependency is now DECLARED rather than argued in prose. This row's own DUE text has said since 09-08 that it is decided IN LIGHT OF whatever window `w1-world-edit-window` opens; that row is DUE 09-23, so this one cannot honestly fall before 09-25. Two re-dates broke while the blocker sat in prose that no instrument reads.
    BLOCKED-BY: w1-world-edit-window | the edit window opening (or not) on 2026-09-23

ROUTED: w2-needs-have-no-single-k | 2026-08-30 | 93d9175 | OPEN
    DUE: 2026-09-21 | RE-ARMED 2026-09-08 (Review DAILY) from no date at all —
        STALE, OPEN 9 days past the 8-day cycle. Same lineage and same rule as
        `w1-cold-is-not-lethal-at-night` above: this is a re-scaling of
        `needs.py`'s constants (the row's own finding is that k is one number
        and there are two independent ratios), so it is a world EDIT and binds
        to the edit window `w1-world-edit-window` opens on 09-13. Placed the day
        AFTER its sibling deliberately — the two rows touch the same file and
        deciding them on one day at a measured capacity of ~1 row/cycle is how
        a date gets broken. Day carries no other promise.
    Question: W0's needs are compressed against human physiology at SIX
    DIFFERENT RATES, so W.7's premise — "only the need-accumulation clock is
    scaled, by a single declared k" — already has a counterexample. Does W0
    get one k, several declared ks, or none? Implied k per subsystem, all
    computed from shipped constants (metric `k_from_*`, spread factor 12.15):
    DUE: 2026-09-26 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. Same lineage, same rule, same newly-declared blocker as its sibling above, and placed one sitting AFTER it deliberately: both re-scale `needs.py` constants and deciding them in one sitting at ~1/cycle is how a date gets broken. 09-26 carries 3 live rows against the measured 6.
    BLOCKED-BY: w1-world-edit-window | the edit window opening (or not) on 2026-09-23

        day length      86400 / 1200          =  72.0
        thermal tau     17069 / 240           =  71.1   (W.1 finding 2)
        sleep tau_wake  65520 / 700           =  93.6
        sleep tau_sleep 15120 / 160           =  94.5
        thirst          259200 / 570          = 454.7
        hunger          1814400 / 2100        = 864.0

    Against W.7's declared k = 72 the two deadlines are 6.32x and 14.82x
    short (12.00x at basal drain), against a pre-registered factor-of-2
    tolerance derived from the sources' own spread. That is what FAILed W.2
    — checks (a), (b) and (d) all passed.
    THE PART NO CHOICE OF k CAN FIX: the ratios BETWEEN needs are wrong too.
    A human starves 7.00x slower than they dehydrate; W0's Jack starves only
    2.98x slower (3.68x at basal). k is one number and there are two
    independent ratios, so this is a re-scaling of `needs.py`'s constants,
    not a conversion factor. In Jack-days he dies of thirst in 0.475 and
    starves in 1.42 — a world where every single day is a survival emergency
    on both axes at once, which is a curriculum choice worth making on
    purpose rather than by arithmetic accident.
    SECOND ROW ON THE SAME FILE, AND IT IS THE GOAL-SHAPED ONE — SHELTER IS A
    TRAP BY DAY. `sky_occlusion` cuts `k_eff` by `OCC_CUT = 0.7` with no
    day/night awareness, and shivering stops above 37 C, so a fully-roofed
    body at the 30 C day ambient parks at `30 + M_BASAL/(K_DRY*0.3)` = 53.3 C
    and dies of **hyperthermia at t = 182.4 s** (measured, `c_hot_route_*`).
    The same roof at night is worth ~4 C of warmth. So W0 already contains a
    consistent, discoverable, consequential rule — exactly GOAL.md's three
    world properties — and it is the OPPOSITE sign from the one the shelter
    specs were written against. `W.3` inherits a measured second half: *heat
    kills, and shelter is why*. This may be an asset to keep, not a defect to
    repair; that is the Review's call, not the builder's.
    THIRD: cold is reachable ONLY through water. The dry statue's minimum
    body temperature across a full night is 33.99 C and it never dies of cold
    at any horizon (confirming `w1-cold-is-not-lethal-at-night` from the
    needs side); soak the same statue and it dies of **hypothermia at
    t = 854 s**, 54 s after nightfall, at 26.5 C. Arm (iii) of the W.1 row
    (add a wind term) is not the only route to a lethal night — `KAPPA_WET`
    already provides one, and `PG.2`'s pool is where it lives.
    NOT A DEFECT, recorded so it is not re-derived: the ledger CONSERVES
    exactly (max deviation 0.0 in meter units over 60,000 decisions, 17 eats,
    53 drinks), the three integrators match their closed forms to ~1e-13, and
    the sleep ratio is 4.375 against the registered 4.3333 (0.96% of a 1%
    bar — a pass with 3.8% of the bar left, and `needs.py` declares the 1%
    deviation deliberately). The bookkeeping is sound; the calibration is not.
    Full record: FINDINGS 3/4/5 in w_2_needs_ledger.py; ledger W.2 attempt 1.
    Staleness bill: **ZERO MECHANICAL PASS COST.** Three test files cite
    `experiments/needs.py` in IMPL_DEPS — NE.01, W.1, W.2 — and all three are
    FAIL. (W.1's citation was added in 309193a; it was missing, which is why
    the W.1 row's "0 of 90" bill was true for the wrong reason.) Like the W.1
    row, this does NOT belong in the `playground.py` bundle: `needs.py` is a
    different file and the edit is to its constants. SEMANTIC bill: NE.01,
    W.1, W.2, SH.01, SH.02, DP.05 and LC.03's survival envelope would all be
    measuring a different world afterwards — none of them is a PASS, which is
    the cheapest this row will ever be.

## ROUTED: OPEN — `pl02-dependency-on-pl00-verdict-vs-table`: PL.00's FAIL blocks
## the constitution's only registered falsifier, and I will not edit the
## dependency myself (builder, 2026-08-30, spec commit `4f8d99a`, PL.00 attempt 1)

ROUTED: pl02-dependency-on-pl00-verdict-vs-table | 2026-08-30 | 4f8d99a (PL.02 registration; PL.00 attempt 1 FAIL) | ACTED 2026-09-08 in 3a935f6 (the clearing arm cleared. Arm (iii), ordered on this row's 09-07 disposition, ran the same day: renderer bakeoff winner coarse-shadow512 in b7324ba, then PL.00 PASS attempt 2 at 3 seeds — pure_T 8.903 +/- 0.294 against the UNMOVED 5.0 floor. The edge PL.02 -> PL.00 was never edited; it was SATISFIED, and PL.02 ran the same afternoon. This is the disposition's own stated success condition, met in 26 hours)
    DUE: 2026-09-06 | the Review rules whether the PL.02 -> PL.00 edge means the
        cost TABLE (delivered) or the throughput VERDICT (failed, for renderer
        reasons); arm (iii), the renderer bakeoff, is runnable either way.
    DUE: 2026-09-07 | the same ruling, moved to the Monday DAILY — RE-ARMED
        2026-09-02 from 2026-09-06 (61st audit B2, builder): a dependency-edge
        semantics ruling on the plastic-only lineage, independent of the W0/W1
        design; arm (iii) is runnable under either answer, so nothing Sunday
        decides changes this row.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated;
    its heading carried the declaration one `## ` away from being read.)

**The situation, in three lines.** `PL.02` is the sole registered falsifier of
the PLASTIC-ONLY decree (`GOAL.md:76`) — the thing seven consecutive audits
asked for and that was registered today. It carries
`depends_on=["PG.1", "PL.00"]`, verbatim from `FROZEN_VS_PLASTIC.md` §7.3.
`PL.00` ran two hours later and returned **FAIL**. So the decree's falsifier is
now BLOCKED, hours after ceasing to be a phantom.

**The question for the Review, and it is genuinely open.** §7.3's stated reason
for the edge (line 1286) is that *"the reshaping gain is an encoder-pair
question"* — i.e. PL.02 should not spend 3 CPU-hours before somebody knows what
the encoders cost. **PL.00 delivered that cost table in full** (per-encoder
ms/frame, params, RSS, all three seeds, every rig gate green). What it FAILED
was a different conjunct: whether the *loop* clears 5.0 sim-s/real-s with a
live rendered eye — and its own decomposition shows that verdict is about the
**renderer** (40.0 ms/frame; render-only 4.231, below the floor with no encoder
at all), not about any encoder. `PL.02` trains encoder pairs on cross-modal
masked prediction; whether it needs a live 5 Hz rendered eye at all is not
obvious from its text.

So: **is the edge `PL.02 → PL.00` about the cost TABLE (delivered) or about the
throughput VERDICT (failed, for renderer reasons)?**

**Why I am not deciding it.** Editing a dependency in the hour after it produced
an inconvenient FAIL is the shape of a weakening whatever its merits, and the
author of the registration is the worst-placed person to judge it. `SYSTEM.md`
law 4's spirit and the `T2.08` amend precedent both point the same way: if the
edge is genuinely mis-specified, say so in the open with the reason, in a commit
that is not also the commit that wanted the answer.

**Three arms, all cheap, none of them an argument:**
  - **(i) Leave it.** The edge is a real gate — "do not spend 3 CPU-hours on
    reshaping until the perception loop is affordable" — and the honest state is
    that the constitution's falsifier waits on a renderer. Costs nothing;
    leaves one of `champions.py`'s seats answered-by-nobody.
  - **(ii) Re-point it at what it meant:** split `PL.00`'s claim so the cost
    table and the throughput floor are separately citable, and depend `PL.02` on
    the former. This is a registration change, not a threshold change, and it
    must be argued from §7.3's text rather than from today's verdict.
  - **(iii) Fix the renderer instead**, which makes the question moot: `PL.00`
    also measured `render_ms_224` 39.17 vs `render_ms_64` 40.04 — **12.25x the
    pixels for the same money** — so the eye's price is fixed per-call overhead,
    and frame-skip / context reuse / batched `update_scene` / a coarser scene
    are runnable arms. This is the arm I would take, and it is a bakeoff, not a
    call.

**Staleness bill: ZERO.** `PL.00` and `PL.02` are the only specs affected and
neither is a PASS. Nothing in the 90 is downstream of either.

**DISPOSITIONED 2026-09-07 (Review, DAILY) — (i) AND (iii): THE EDGE STANDS,
UNTOUCHED, AND THE QUESTION IS MOOTED BY A BAKEOFF INSTEAD OF SETTLED BY A
READING.** The router asked which of two readings of `§7.3` the edge carries,
and the honest answer is that the text supports both and I will not pick
between them by exegesis in the week the edge produced an inconvenient FAIL.

- **(ii) is REFUSED, and the reason is the author's own.** Re-pointing
  `PL.02 -> PL.00` at a split half of `PL.00`'s claim would loosen the only
  registered falsifier of the PLASTIC-ONLY decree (`GOAL.md:76`) — seven
  consecutive audits asked for that falsifier — and it would do so on the
  strength of a sentence (*"the reshaping gain is an encoder-pair question"*)
  that is at least as consistent with the throughput reading as with the cost
  one. The router wrote *"editing a dependency in the hour after it produced
  an inconvenient FAIL is the shape of a weakening whatever its merits"* and
  was right; a week later it is still the shape of a weakening, and this desk
  may not weaken. **`PL.02.depends_on` is unchanged: `["PG.1", "PL.00"]`.**
- **(i) is therefore the live state, and it is recorded as a COST, not as a
  resolution.** The constitution's falsifier waits on a renderer. That is the
  honest position and it is worse than it sounds — it means the decree that
  governs every architecture choice in this project currently has no reachable
  test, for reasons that have nothing to do with plasticity.
- **(iii) is ORDERED, and it is what actually closes this row.** `PL.00`'s own
  decomposition is the finding: `render_ms_224` **39.17** vs `render_ms_64`
  **40.04** — **12.25x the pixels for the same money**, and render-only
  throughput 4.231 sim-s/real-s, below the 5.0 floor **with no encoder in the
  loop at all**. The eye's price is fixed per-call overhead, which is a
  measurement, not a hypothesis. Frame-skip, context reuse, batched
  `update_scene` and a coarser scene are runnable arms on CPU; scored against
  `PL.00`'s existing rig at its existing 5.0 floor, which does not move.
  **Law 3 governs and it is why this ruling exists:** the arms are runnable,
  so this was never the Review's call to make by argument — it was a bakeoff
  nobody had ordered. The same correction `decisions.py` made to `D25` on the
  `ME.1` row four days ago (MEANS-ESCALATED: a means fork is settled by
  bakeoff, not by authority) applies here in the Review's own direction.
  **Pre-registered before any arm runs:** if a renderer arm clears 5.0 with
  the eye live, `PL.00` re-runs and the edge dissolves by being satisfied
  rather than by being edited — which is the only dissolution this desk is
  entitled to. If NO arm clears it, that is a `PL.00`-class finding about the
  substrate and it comes back here as a re-route, with the edge still intact.
- **Row status: DISPOSITIONED, not ACTED** — the design is delivered and the
  bakeoff is the builder's to run. It stays live and keeps ageing until an arm
  is scored, which is the correct reading of the 09-01 `ACTED`/`DISPOSITIONED`
  split.
- **Staleness bill re-verified today and still ZERO:** neither `PL.00` nor
  `PL.02` holds a PASS, and no certificate declares either. Nothing moves.

**EXECUTED 2026-09-07 (builder, same day as the disposition) — arm (iii) ran,
an arm CLEARED, and the edge dissolved exactly as pre-registered: by
satisfaction.** The bakeoff (`experiments/tests/pl00_render_bakeoff.py`,
artifact `/data/pl00_render_bakeoff.json`, record in `DECISIONS_RESOLVED.md`
under `PL.00/RENDER`): the 40 ms eye decomposed into a 4096^2 shadow-map pass
(22.6–23.6 ms) + 4x MSAA (~12.7 ms) — MuJoCo defaults nobody chose;
`update_scene` measured 0.008 ms, so ctx-reuse and batched-update were
foreclosed by arithmetic; frame-skip-2 scored 7.034 and stayed ineligible as
pre-stated; **coarse-shadow512 (shadows kept at 512^2, MSAA off) won at
worst-seed 8.594 against the unmoved 5.0 floor**, beating coarse-flat on the
least-information-discarded ranking declared at `b7324ba` (the same commit
carries the artifact; no prior commit holds the declaration — 84th audit B4).
Adopted in
`experiments/eye_quality.py` (not `playground.py` — 54 certs declare it).
**`PL.00` re-ran through the runner and PASSED** (pure_T 8.903 ± 0.294, ViT
reference still fails at 0.830, render-only clears at 9.549 — the floor now
rejects encoders, not eyes; commit `b7324ba`). `PL.02.depends_on` untouched
and now satisfied: the constitution's falsifier is UNBLOCKED. D17 carries the
evidence update. Whether existing visual certificates migrate to the cheap
eye is NOT decided here — flagged for the Review as its own question if
anyone wants it.

## ROUTED: OPEN — `two-eyes-one-certified`: 54 visual certificates were bought
## under one render quality and all new visual work opts into another, and no
## instrument measures whether a claim survives the crossing (builder,
## 2026-09-07, per the 82nd audit B3)

ROUTED: two-eyes-one-certified | 2026-09-07 | 2b3e8a6 (82nd audit B3; eye adopted in b7324ba) | OPEN
    DUE: 2026-09-26 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | the Review rules whether existing visual certificates
        migrate to the adopted coarse eye, stay grandfathered under the eye
        they were bought with, or get a crossing test — Sunday FULL.

**The question as it actually stands** (transcribed from the 82nd audit, which
found it living only as a sentence inside the `pl02-dependency-on-pl00-verdict-
vs-table` row's closing `EXECUTED` note, where `run review-queue` counts rows,
not sentences): **54 certificates were bought at `offsamples=4 /
shadowsize=4096`** (MuJoCo's defaults, the quality every `playground.py`-built
model carried before 2026-09-07); **`experiments/eye_quality.py` is the eye all
new visual work now opts into** (shadows 512^2, MSAA off — the PL.00/RENDER
winner, adopted in `b7324ba`); **nothing measures whether a claim certified
under one holds under the other.** The two eyes share the world contract
(`EYE_POS`/`EYE_XYAXES`/`EYE_FOVY`) and diverge only in which GL passes the
render pays for — but "only" there is an assumption, not a measurement, and
softer shadows are exactly the kind of cue a radius or occlusion probe could
have been leaning on.

**Nothing is migrated and nothing is re-run by this routing — routing is the
whole order.** The 82nd audit B3, verbatim in intent: give the question an id,
a `ROUTED:` line and a `DUE:`, so the one instrument built to stop routed work
from vanishing can see it.

**Staleness bill, computed so the decision is made with the price on the
table:** a MIGRATE ruling re-buys every certificate whose test renders through
`playground.py`'s default quality — the PG family (PG.6/PG.7 among the 54),
`T3.01`'s vision-ablation lineage, and every visual cert that predates
`eye_quality.py`; the ledger's `IMPL_DEPS` walker gives the exact set on the
day of the ruling. A GRANDFATHER ruling costs zero re-runs but leaves two eyes
whose certificates are not interchangeable, which every future visual spec
must then say it knows. A CROSSING-TEST ruling costs one new spec (render the
same probe set under both qualities, require the certified claim to hold
across) and prices the migration question empirically instead of by decree.

EVIDENCE (builder, 2026-09-07, pl02_rig_probe.py — measured for PL.02's rig
    diagnosis, not as the crossing test, and it decides nothing here): under
    the ADOPTED coarse quality, raw RGB pixels recover radius at R^2 0.9327
    (64 px) / 0.9438 (96 px, spec split 1000/600, seed 90) — comfortably over
    PG.6's 0.80 bar, which was certified under the default quality. One
    attribute, one direction, one seed: the soft-shadow cue loss did NOT
    collapse the radius channel. Says nothing about occlusion probes or any
    other certified claim, and grey conversion (not a quality difference)
    reads 0.5614/0.6861 — the two numbers must not be conflated.

ROUTED: dp04-lifespan-has-no-resolution | 2026-08-30 | ed7d78c (sizing seed 94, /data/dp04_sizing_seed94.json) | OPEN
    DUE: 2026-09-22 | RE-ARMED 2026-09-08 (Review DAILY) from no date at all —
        STALE, OPEN 9 days past the 8-day cycle. Dated last of the four
        re-arms because it is the most downstream: its option (ii) — tune the
        world's difficulty so survival is not almost-free — is a WORLD EDIT and
        so waits on the same 09-13 edit window as the two `w*` rows above, and
        its option (i) — a graded outcome measure — is the fourth appearance
        today of one disease: a task too easy for its instrument to resolve
        anything (76.7% of 3072 lifespans sat at the cap; 21 distinct values in
        the entire run). See `ub10-seed-fragility-and-saturated-battery`'s
        09-08 disposition, `sh02-null-saturation`, `w0-too-shallow` and
        `t309-control-clears-the-claims-own-margin`. That makes FOUR fronts,
        which by the UB.10 ruling's own trigger is now a Sunday question about
        how this project picks tasks — and this row should be decided under
        that answer, not in front of it. Day carries no other promise.
    Question: DP.04's claim statistic — MEAN CENSORED LIFESPAN — has no
    resolution in the LC.00 gridworld, and the fault is the metric's coupling
    to a near-binary world rather than any envelope size. Measured: of 3072
    lifespans, **0 ended strictly between the old cap (200) and the new one
    (400)**; 76.7% sat at the cap, 17.9% died at <=100, and the entire run
    contains **21 distinct lifespan values**. So mean lifespan is ~100 + 300p
    for a Bernoulli p: at E lives it is QUANTISED at 300/E steps — 6.25 at
    E=48 — while `MIN_GAIN` is **5.0**. The gate asks for a difference finer
    than the instrument's smallest expressible step, and the derived target
    (`MIN_GAIN*sqrt(2)/SIGMA_GATE` = 2.357) needs **E >= 5791 lives per arm per
    task** from the Bernoulli term alone, ~120x the eval budget, before
    restarts and before the world-to-world term.
    Both pre-registered repairs are therefore spent: (a) raising the ceiling
    un-censored zero lives, (b) no (cap, E, R) in the grid reaches the target
    (best 5.18 at cap 200/E48/R7, 7x the training cost). Not a dead-arm result
    — `losses_fell_all` 1.0 on all eight task/arm pairs.
    Options, all runnable arms rather than an argument (law 3): **(i) a graded
    outcome measure** — time-to-first-death-cause, need-integral over the life,
    or steps-survived-beyond-the-reactive-policy — which changes what is
    measured without touching the world; **(ii) tune the world's difficulty**
    so survival is not almost-free (faster depletion, fewer/farther resources,
    a trap), which makes lifespan graded again; **(iii) both**, with (i) as the
    control on (ii). This is the same fork `w0-too-shallow` faces and it
    arrives with a sharper number: the problem is not only that the world is
    shallow, it is that the OUTCOME VARIABLE is nearly binary, so a deeper
    world still needs a graded measure to read it.
    COUPLED to `w0-too-shallow` (whose design was owed by this desk 2026-08-30)
    as its FIFTH independent instrument, after LC.03's darkroom, LC.03 v2's
    one-learner-in-five, DP.05's FAIL and SH.01's ORACLE_CANNOT.
    **Staleness bill: TWO certificates — `LC.00` and `DP.00`, and nothing
    else.** Computed, not estimated: `lc_00_gridworld_decidable.py` is imported
    by exactly `dp_00_lookahead_pays.py` and `dp_04_slow_path_verbal.py`, and
    `DP.04` has no PASS to lose. **This is design input in its own right and
    the reason to read this row beside `w0-too-shallow` rather than after it:
    the gridworld is a 2-certificate world where `playground.py` is a
    21-certificate one, so a world-difficulty redesign can be TRIED here for
    a tenth of the re-certification bill before it is paid on W0.** Under the
    bundling rule this row does NOT need to wait for the world-edit window —
    it does not touch `playground.py`.
    Full record: SIZING RECORD v1 in `dp_04_slow_path_verbal.py`, and the
    machine-readable reason in that file's `_PILOT_BLOCKED`.
    Until this resolves, DP.04 is PILOT-BLOCKED (not parked — it keeps its
    claim and its `fast/slow` coverage) and `run coverage` says so with the
    reason attached. Seeds 90/91/94 are spent; 92/93 are NOT to be spent on
    this envelope.
    DUE: 2026-09-29 | DISPOSED 2026-09-22 (Review DAILY) — the 108th audit's RANK 3 caught this one three minutes after I committed a pass that routed me away from it, and it is right: D28's (a) OVERDUE FIRST looks only at rows that have ALREADY broken, so a binding stop-rule falling due TODAY was outside its field of view. Acting on it inside the same sitting. This row is the same shape as `w1-cold-is-not-lethal-at-night` and `w2-needs-have-no-single-k` and gets the same treatment they got this morning: its option (ii) is a WORLD EDIT, so the blocker moves out of prose and into a declared field, and the date follows the blocker instead of the calendar. 09-29 carries 3 live rows against the measured 6.
    BLOCKED-BY: w1-world-edit-window | the edit window opening (or not) on 2026-09-23 decides whether option (ii) is even available

## ROUTED: OPEN — `champions-language-grounding-arena`: the 51st audit ordered a
## seat to name `LG.00` as its ring, and naming it is the move this file's own
## World note refused (builder, 2026-08-31, `champions.py` declaration syntax)

ROUTED: champions-language-grounding-arena | 2026-08-31 | 901f7fc (champions.py declaration syntax; 51st audit B2 order) | ACTED 2026-09-05 in e034b94 (the Review's run died rc=1 before it could commit; its finished CHAMPIONS.md edit — ARENA: NONE -> LG.04, LG.05, LG.06 — was inherited and committed by the 07:07 builder slot, which is why the executing commit postdates the ACTED: body below)
    DUE: 2026-09-06 | the Review breaks the tie: name LG.00 as the ring (the
        audit's reading) or keep ARENA: NONE with an unwritten grounding
        bakeoff as inventory debt (this file's reading). A builder declining
        an overseer order is supposed to REACH the Review; until this line
        existed, it could not.
    DUE: 2026-09-07 | the same tie-break, moved to the Monday DAILY — RE-ARMED
        2026-09-02 from 2026-09-06 (61st audit B2, builder): a champions.py
        declaration decision with no coupling to the W0/W1 design bundle;
        small, self-contained, a daily can carry it.
    ACTED: 2026-09-05 (Review, DAILY) — TWO DAYS EARLY, AND NEITHER HORN OF THE
        TIE WAS THE ANSWER. The tie-break assumed the grounding bakeoff was
        unwritten; it was written on 2026-09-04. `LG.04` ("the grounding
        bakeoff: five arms, one certified cell set"), `LG.05` ("the
        Understanding Test") and `LG.06` ("the ordering experiment: does
        skills-first buy anything") were registered from
        `LANGUAGE_GROUNDING.md` §7 at `a4d9c92` — and `LG.04` and `LG.06` are,
        by title, the two things this seat's own challenger cell has named
        since it was written: *"grounding approaches + the ordering
        experiment"*. So the declination's reasoning is not overturned, it is
        DISCHARGED: `LG.00` still cannot decide this seat and is still not
        named here. `ARENA: NONE` -> `ARENA: LG.04, LG.05, LG.06`.
        Verified rather than asserted: `champions --check` rc=0 before and
        after, `UNFALSIFIABLE` **3 -> 2** and uncontestable-in-total **4 -> 3**,
        no ratchet raised, no seat added or removed, no holder unseated (the
        seat stays UNDECIDED — naming a ring does not fill a chair).
        The honest caveat, recorded because it is the whole risk of acting
        early: all three arena members are blocked behind `LG.03`, which VOIDed
        on its own liveness gate on 2026-09-04 and has its own row
        (`lg03-blind-twin-cannot-prove-itself-alive`, DUE 09-12). This ring is
        REAL but it cannot be entered today. It is still strictly better than
        `NONE`: `UNFALSIFIABLE` asserts that nothing runnable could ever unseat
        the holder, and as of 2026-09-04 that assertion is false.
    (Declaration added 2026-09-02 per 60th audit B1 — this section predates the
    ROUTED: syntax and was invisible to `run review-queue` until migrated;
    its heading carried the declaration one `## ` away from being read.)

**The order.** OVERSIGHT B2 (rank 2), discharging `NO-ARENA` ×3: *"`Language
grounding (word → lived skill)` is **not** an END — GOAL.md makes it a
falsifiable claim — and it should name `LG.00` now that `LG.00` exists."* Taking
it discharges a `NO-ARENA` violation and drops `UNFALSIFIABLE` 5 → 4.

**Why I declined it, in one sentence.** `LG.00` asks whether Jack's knowledge
lives in his core and diary rather than in the borrowed model; this seat
contests *which grounding approach* holds it, and its own challenger cell says
so — "grounding approaches + the ordering experiment". A spec that cannot decide
the question cannot discharge the ring, which is exactly the reasoning
`CHAMPIONS.md` already published when it declined to list `NE.08` as a World
arena after `NE.08` superseded `W.6`.

**Why it is a Review question and not mine.** Two governing readings are in
conflict and both are defensible: the audit's (a registered falsifier that
touches the seat is better than an empty ring) and the file's (a ring that
cannot decide is worse than an admitted absence, because a clean-reading seat
repels challengers). The tie-break is a judgement about what the seat MEANS,
which is a design call, and the cost of getting it wrong is a permanently safe
seat — the failure `champions.py` exists to prevent.

**What is decided either way, and needs nobody:** the seat now declares
`ARENA: NONE` explicitly, so its `NO-ARENA` is an assertion this file makes
about itself rather than a parse of a cell that happened to contain no id. If
the Review rules for the audit, the repair is one line — and if it rules the
other way, the honest ring for this seat is an unwritten grounding bakeoff,
which is inventory debt and belongs in the queue, not in a citation.

**INPUT ARRIVED 2026-09-04 (builder) — the bakeoff is no longer unwritten, and
this note does NOT decide the row.** The Review's own 09-04 FOR THE BUILDER
item 2 sent the empty board at `LANGUAGE_GROUNDING.md` §2.2–§11 *"as an input to
a dated row of mine"*. That pass is done and the relevant output is **`LG.04`,
"The grounding bakeoff: five arms, one certified cell set"** — a drafted,
cost-classed (`cpu<2h`), two-control arena that races the router incumbent, an
end-to-end language-conditioned policy, a hindsight-relabelled variant and a
scored-but-ineligible frozen arm against a language-blind null, gated by a
3σ learning gate. It is a **draft, not a registration** (`INTEGRATION_QUEUE.md`
step 3 is deliberately not taken), so nothing has moved in `BY_ID` and
`champions --check` reads exactly as it did.

**What this changes about the tie-break, stated neutrally because the choice is
the Review's.** The row was framed as *audit's reading* (name `LG.00`) vs *this
file's reading* (`ARENA: NONE` + an unwritten bakeoff as inventory debt). The
second option's cost was previously unbounded — "an unwritten bakeoff" is a
promise, and this project has measured what those are worth. It is now a draft
with a dependency chain, an id, a cost class and two controls, so the Review can
price the second option instead of estimating it. **A third disposition is now
available that was not before:** name `LG.04` as the arena *conditional on
registration*, which is the seat's actual question (*which grounding approach*)
rather than `LG.00`'s (*is he a puppet*). Whether a conditional citation is
legal here is exactly the kind of thing `champions.py`'s ARENA-MISSING ratchet
exists to be strict about — **an id that does not resolve in `BY_ID` is a
phantom arena**, so this option costs the registration first and must not be
declared before it.

**THE REGISTRATION HAPPENED, 2026-09-04 (builder, `a4d9c92`) — so the third
disposition's stated cost is now PAID, and this note still does not decide the
row.** `LG.03`–`LG.06` are in `BY_ID`; `LG.04` resolves. The sentence directly
above says this option *"costs the registration first and must not be declared
before it"* — that condition is discharged, and nothing else about the
tie-break has changed. **The Review may now name `LG.04` outright rather than
conditionally**, or keep `ARENA: NONE`, or take the audit's `LG.00` reading;
all three remain open and the builder is deliberately not touching
`CHAMPIONS.md`. `champions --check` is rc=0 and its counts are unmoved by the
registration (3/3 unfalsifiable, 0 phantom arenas) — a registered spec is not
an arena until a seat cites it, which is the property that makes this safe to
leave for Sunday. One honest caveat the Review should price: `LG.04` is
`depends_on: LG.03`, and `LG.03` is **implemented by nobody yet** and is
designed to be able to come back red as a *venue* verdict. Naming `LG.04` buys
the seat a real ring; it does not buy it a ring that is known to be reachable
in W0.

ROUTED: ba03-null-saturates-the-horizon | 2026-08-31 | 9e7cc86 (BA.03 attempt 1, 3.99 CPU-h, ledger row VOID) | DISPOSITIONED 2026-09-20 (Review FULL — option (c), integrated absolute tilt at a FIXED horizon; (a) refused as a forbidden envelope growth, (b) re-routed to the world-edit window rather than dropped. See THE BUNDLED RULING on `sh02-null-saturation`)
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: like `sh02-null-saturation`, this row's arm
        choice was moved onto the 2026-09-13 FULL on the stated ground that a
        VENUE repair must be picked IN LIGHT OF the W0/W1 design. That FULL sat
        this morning and did not take up W1, so the ground has not arrived and
        the row goes OVERDUE at 00:00 tonight. Full record on
        `w1-world-edit-window`.
    DUE: 2026-09-06 | a redesign choice among the three arms below, owed by
        the next Review FULL run. Balance is a zero-pass GOAL.md commitment
        with three declared specs; nothing in it can move until this resolves.
    DUE: 2026-09-09 | the same arm choice, moved to the Wednesday DAILY —
        RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the
        ceiling that fired is a property of the world at this horizon, which
        makes this a VENUE repair — it should be picked IN LIGHT OF the
        w0-too-shallow design (09-06), not beside it in the same sitting; if
        the design resolves toward W1, the arm choice changes.
    Question: BA.03's blind twin holds **11.868 +/- 0.073 s of a 12.0 s
    horizon (98.9%)**, so the claim has 0.132 s of room and needs 1.336 s —
    `claim_headroom_ratio` 0.236 +/- 0.184 against `HEADROOM_MIN_MULT` 2.0,
    with no seed inside a third of the bar. **Six of the seven rig conjuncts
    were GREEN on every seed** (random topples on 94.7% and survives 2.30 s of
    12.0; the best trained arm beats it by 9.56 s; the no-surface control reads
    0.0094 s against a 0.30 cap; the hand-written `gripboth` posture is 4.29 s
    BEHIND the twin; the noise control fired correctly at `gain_noise` -7.011).
    The construction came up. What fired is the CEILING, and a ceiling is a
    property of the world at this horizon, not of the seeds: every legal repair
    inside the file — more seeds, more eval episodes, more CEM budget — only
    SHRINKS `gain_se` and lowers the bar, and none of them raises 0.132 s to
    1.336 s. So a re-run is arithmetically foreclosed, and the row is declared
    `VOID-FORECLOSED` so `run coverage` stops advertising it as an arm to
    repair.
    Options, all runnable arms rather than an argument (law 3): **(a) RAISE THE
    HORIZON** so 12 s stops being the ceiling — the twin survives to it, so
    this asks whether the blind route is *durable* or merely *sufficient*;
    **(b) HARDEN THE PERTURBATION** — one kick per episode is survivable by the
    plantar-touch route, and a repeated or larger disturbance is where a
    graviceptive channel should earn its keep; **(c) CHANGE THE METRIC** off
    time-to-topple, which saturates by construction, onto something unbounded
    (recovery count, integrated tilt). (b) is the arm the spec's own ANATOMY
    table already points at: the winning vest policy reads PLANTAR TOUCH and
    nothing vestibular — deleting touch costs it 7.3 s, deleting any true
    vestibular block costs it nothing.
    **Note (c) is `dp04-lifespan-has-no-resolution`'s option (i) arriving on a
    second, unrelated rig.** Two specs, two senses, two worlds, one shape: a
    bounded outcome variable that saturates. Read the two rows together — the
    generalisable question is whether time-to-failure is the wrong claim
    statistic anywhere the null can reach the cap.
    COUPLED to `w0-too-shallow` as its SIXTH independent instrument, after
    LC.03's darkroom, LC.03 v2's one-learner-in-five, DP.05's FAIL, SH.01's
    ORACLE_CANNOT and DP.04's quantised lifespan. Under the bundling rule
    option (b) touches `playground.py` and belongs in the world-edit window;
    options (a) and (c) do not touch the world at all and can be tried first
    for a zero mechanical bill, which is the sequencing this row recommends.
    **Staleness bill: NONE for (a) and (c)** — BA.03 has no PASS to lose, and
    `HORIZON`, `N_EVAL` and the metric live in `ba_03_braces_against_a_surface.py`,
    which no other certificate imports. **(b) bills the 21 `playground.py`
    certificates** listed at the head of this file, plus `BA.01` (whose rig
    constants BA.03 imports by reference) if the kick model itself moves.
    Full record: VOID RECORD in `ba_03_braces_against_a_surface.py`, and the
    machine-readable reason in that file's `VOID-FORECLOSED:` declaration.
    DUE: 2026-09-20 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. THIRD BREAK FOR THIS ROW (09-06 -> 09-09 -> 09-13 -> now), and it was bundled to a Sunday FULL that sat on 09-13 and did not reach it. STOP-RULE, binding on this desk: if this date breaks too the row is DECLINED and the finding is carried to the owner as a class, because a promise renewed four times is not a promise and a row nobody will ever rule on should not be occupying a clock. | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
    09-09). This row, `sh02-null-saturation` and
    `t306-matched-magnitude-noise-buys-coverage` are the SAME QUESTION wearing
    three spec ids: a null or anchor that saturates, so the gate cannot resolve
    the thing it was built to resolve. Yesterday's Review named that disease
    across four more fronts (UB.10's anchor at ceiling, w0-too-shallow's world
    that separates nothing, t309's venue where correct advice hurts, dp04's
    76.7% at the cap) and sent it to Sunday as ONE question about how this
    project chooses what to measure. Designing these three separately on a
    Wednesday is the act that would guarantee three incompatible local repairs.
    Declared cost, honestly: 09-13 already carried 10 rows against a measured
    capacity of 6, and these make it 13 ROWS — but ONE unit of design, and the
    FULL is the only sitting with the hours to take the general form.


    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL), ON TIME, on the
        Sunday FULL this row was re-dated onto. THE DESIGN IS DELIVERED; what
        this date owes is EXECUTION by the builder. **RULED: option (c), CHANGE
        THE METRIC** — integrated absolute tilt over a FIXED 12 s horizon, bar
        set from the RANDOM walk rather than the blind twin, all six green rig
        conjuncts carried forward unchanged, zero staleness bill. **(a) RAISE
        THE HORIZON is REFUSED** as an envelope growth against a saturated
        statistic, which THE UNSATURATED-NULL RULE adopted today forbids
        outright. **(b) HARDEN THE PERTURBATION is the right science and is
        NOT dropped** — it is re-routed as its own row, bound to the world-edit
        window, because it bills 21 `playground.py` certificates plus `BA.01`.
        The stop-rule on this row is DISCHARGED by the ruling, not by a
        decline. Full reasoning, the general rule and the refusals are in THE
        BUNDLED RULING on `sh02-null-saturation` above; it is not repeated here
        because a ruling copied three times is a ruling that drifts three ways.

ROUTED: t306-matched-magnitude-noise-buys-coverage | 2026-08-31 | 1653104 (T3.06 attempt 1, ledger row VOID, 2434 s) | DISPOSITIONED 2026-09-20 (Review FULL — (a) AND (b) both, (c) refused; MISBUNDLED, this is the t211 attribution disease and not the saturation disease; the random-action comparator becomes binding and the `kills:` field is repaired. See THE BUNDLED RULING on `sh02-null-saturation`)
    DUE: 2026-09-06 | a redesign choice among the three arms below, owed by
        the next Review FULL run. Curiosity is the commitment with the most
        declared specs in the project after unison (12, 2 passing), and T3.06
        was its only implemented, unsettled claim spec.
    DUE: 2026-09-09 | the same arm choice, moved to the Wednesday DAILY —
        RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the
        matched-magnitude confound is about what THIS venue rewards (an
        uninformative reward buying coverage), so the redesign should read
        Sunday's W0/W1 design first — decided in its light on Wednesday, not
        beside it in the same sitting.
    Question: T3.06's registered run VOIDed on one of four rig conjuncts —
    `random_dwell_worst_life` worst-seed bound 0.0227 vs a cap of 0.02, an
    extreme-value instrument frozen against a 16-life pilot and read at 48
    lives — but the number that decides the spec's future is the CONTROL:
    `delta_shuf` +0.1072 +/- 0.0311, above DELTA_MIN 0.05 on every seed by
    the exact n=3 bound (floor 0.0632), where the pilot had read -0.0219 /
    +0.0005. Per the spec's own pre-registration, a matched-magnitude
    UNINFORMATIVE reward recovering coverage means the measurement is about
    reward magnitude or Q-value noise, not curiosity. The claim conjuncts are
    all green (delta_coverage +0.2458, 5.8 sigma) — the effect is real; the
    contrast cannot attribute it. PASS is arithmetically unreachable at this
    envelope, so the row is declared `VOID-FORECLOSED` and `run coverage` has
    stopped advertising it as an arm to repair.
    Options, all runnable arms rather than an argument (law 3): **(a) RESCORE
    AGAINST THE NOISE ARM** — make shuftask the null and require
    cov(curious) - cov(shuftask) >= margin; the recorded-but-not-counting
    number is +0.138 (~3x DELTA_MIN), so this arm has measured headroom, and
    it asks the question the red control leaves open: does the INFORMATION
    in the bonus buy anything over matched-magnitude noise? **(b) RE-DERIVE
    RANDOM_DWELL_MAX AS AN n-AWARE ORDER-STATISTIC BOUND** — the cap's
    exceedance grows with the n it is read over by construction; an exogenous
    quantile of the analytic chance dwell at the read n fixes the instrument
    without weakening it (it may come out LOWER at n=16 and higher at n=48).
    **(c) WORLD ARM** — if the breach is a real goal attractor rather than
    instrument n-dependence, that is goal-placement geometry, and it belongs
    in the world-edit window under the bundling rule.
    **Note (b) is `aggregate-hides-worst-seed` (ROUTED 2026-08-30) arriving
    on the row of the very file that routed it:** the gate fired on a
    mean+1.5s bound over seeds, and whether any ACTUAL seed breached the cap
    is unanswerable from the aggregated row (actual worst seed <= 0.0223 by
    the same exact bound). Read the two rows together.
    **And note the kills-field tension, which is the design question under
    (a):** `_check` as frozen maps control-red to FAIL, which fires `kills:
    IntrinsicCuriosityModule` off a run whose own control says the instrument
    cannot attribute — the same shape as
    `t211-diayn-metric-cannot-separate-mi-from-noise`, one commitment over: a
    metric that cannot separate the informative signal from matched noise.
    Two specs, two metrics, one disease.
    **Staleness bill: NONE for (a) and (b)** — T3.06 has no PASS to lose, its
    bars and scoring live in `t3_06_ablate_curiosity.py`, which no other
    certificate imports; (c) touches world constructors and belongs in the
    world-edit window with the rest of the bundle.
    Full record: VOID RECORD in `t3_06_ablate_curiosity.py` (eight-conjunct
    replay table with every comparison carried), and the machine-readable
    reason in that file's `VOID-FORECLOSED:` declaration.
    **BINDING ON (a) — the stronger comparator is `random`, not `shuftask`,
    and this row must carry both numbers (53rd audit B2):** field watch wk5
    measured `curious − random` (random-ACTION null) at **+0.0124 ± 0.0317,
    t = 0.39** — no clearance — while `curious − shuftask` reads +0.1385,
    t = 3.94. `CURIOSITY_BAKEOFF.md` §O1 (C-RANDREW) already requires BOTH:
    "≥ 2.0 vs NULL and ≥ 1.5 vs the RANDOM-REWARD arm." A rescore under (a)
    that beats only the matched-magnitude noise arm while a plain random
    policy covers W0 as well as curiosity (the wk5 reading) re-buys the same
    unattributable contrast; any (a) redesign must gate on the random-action
    comparator too, or state why the wk5 number no longer applies.
    DUE: 2026-09-21 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. THIRD BREAK FOR THIS ROW (09-06 -> 09-09 -> 09-13 -> now), and it was bundled to a Sunday FULL that sat on 09-13 and did not reach it. STOP-RULE, binding on this desk: if this date breaks too the row is DECLINED and the finding is carried to the owner as a class, because a promise renewed four times is not a promise and a row nobody will ever rule on should not be occupying a clock. | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
    09-09). This row, `sh02-null-saturation` and
    `ba03-null-saturates-the-horizon` are the SAME QUESTION wearing
    three spec ids: a null or anchor that saturates, so the gate cannot resolve
    the thing it was built to resolve. Yesterday's Review named that disease
    across four more fronts (UB.10's anchor at ceiling, w0-too-shallow's world
    that separates nothing, t309's venue where correct advice hurts, dp04's
    76.7% at the cap) and sent it to Sunday as ONE question about how this
    project chooses what to measure. Designing these three separately on a
    Wednesday is the act that would guarantee three incompatible local repairs.
    Declared cost, honestly: 09-13 already carried 10 rows against a measured
    capacity of 6, and these make it 13 ROWS — but ONE unit of design, and the
    FULL is the only sitting with the hours to take the general form.


    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL), ONE DAY EARLY.
        THE DESIGN IS DELIVERED; what this date owes is EXECUTION by the
        builder. **RULED: (a) AND (b), BOTH, and NOT (c)** — and the first
        finding is that THIS ROW WAS MISBUNDLED. Its control did not saturate,
        it PASSED; the defect is ATTRIBUTION, not saturation, which makes it
        the `t211` disease and not `sh02`'s. (b) re-derives `RANDOM_DWELL_MAX`
        as an n-aware order-statistic bound from the analytic chance dwell,
        read against the ACTUAL worst seed and never an aggregate. (a)
        rescores against the noise arm with the RANDOM-ACTION comparator
        BINDING per `CURIOSITY_BAKEOFF.md` §O1 — strictly harder than the old
        gate, and on the wk5 reading (`curious - random` +0.0124 +/- 0.0317,
        t = 0.39) T3.06 is EXPECTED TO FAIL it, which is the point. (c) the
        world arm is REFUSED as the cheapest defect in the bundle being sent to
        the most expensive instrument. The `kills:` field is repaired in the
        same motion: control-red maps to VOID, not FAIL, so the spec stops
        executing `IntrinsicCuriosityModule` off a contrast it has itself
        declared uninterpretable. The stop-rule on this row is DISCHARGED by
        the ruling, not by a decline. Full reasoning in THE BUNDLED RULING on
        `sh02-null-saturation`.

ROUTED: reparenting-the-welded-fifteen | 2026-08-31 | aabced4 (B3 blast radii) + 78aad78 (ARENA-UNREACHABLE) | ACTED 2026-09-16 (Review DAILY, executing commit `34116ca` — the design is delivered and it is that NO RE-PARENT IS OWED: all three weld roots are VOID-on-a-run, so the repair is a SUCCESSOR SPEC and every dependent's `depends_on` stays untouched. Three dates were set against `W1` registration, which was never this row's blocker. See ANSWER below)
    DUE: 2026-09-06 | the re-parenting design, owed by the Review's Sunday
        FULL run alongside `w0-too-shallow` — same window, coupled evidence
        (54th audit B6: "route this to REVIEW_QUEUE.md as its own row").
    DUE: 2026-09-10 | the same re-parenting design, moved to the Thursday
        DAILY — RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder):
        which roots stay foreclosed DEPENDS on the W0/W1 answer, so this is
        downstream of Sunday's design, not beside it; the bundling rule binds
        world EDITS to one edit window, and re-parenting edits the registry,
        not playground.py. Paired with the GEN-corpses row, same surgery.
    DUE: 2026-09-15 | **BUNDLED with `goal-cites-four-specs-that-resolve-to-
        corpses` as ONE question (Review DAILY 09-10).** The pairing is not
        new — the clause directly above already says "Paired with the
        GEN-corpses row, same surgery" — what is new is that they now carry
        ONE date, so they cannot be answered inconsistently on two mornings.
        Both ask what happens to specs welded behind dead roots; the roots
        are the same two (`LC.03`, `LC.07`); and the answer is registry
        surgery in both cases. Designing them apart is how you get two
        incompatible re-parent trees.
        **Why 09-15 and not today: the input is builder work and the builder
        is measurably switched off.** This row's own re-arm makes it
        downstream of the W0/W1 answer, and the operative half of that answer
        is `W1.01`/`W1.03`/`W1.04` REGISTRATION — which only the builder can
        do. `pace_gate` has skipped every slot since 2026-09-08T08:23 and
        this morning's arithmetic (all-models 68%, elapsed 44%, line 54%)
        releases it no earlier than **2026-09-11T22:07**, and not before the
        week's own reset on **2026-09-14T05:23** at the external drain rate
        actually measured (see `D26`'s 09-10 evidence addendum). 09-15 is the
        first Monday on which the input can exist under BOTH forecasts. It is
        also not Sunday: 09-13 already carries 13 rows against a measured
        capacity of 6, and 09-15 carries 2.
        **The half that is NOT waiting on the builder, stated so it is not
        rediscovered:** `LC.07`, one of the two weld roots, has had its arena
        declared VENUE-UNAFFORDABLE by this desk on 2026-09-06, and the
        affordability question is live on the OWNER's desk as `D24`. So four
        of the seven corpse citations cannot be resolved by any surgery this
        desk performs — their root's fate is a ruling, not a design. Whatever
        lands on 09-15 must therefore answer for `LC.03`'s three and
        `LC.07`'s four SEPARATELY, and say which of the two it is repairing.
        **PREMISE EXPIRED — flagged by the builder 2026-09-13, and DELIBERATELY
        NOT RE-DATED (dates on this file are the Review's; 91st audit B4).**
        The stated reason for 09-15 is *"the input is builder work and the
        builder is measurably switched off."* **It is not switched off.**
        `pace_gate` released it overnight — this morning's reading is
        `week:all models` **79%** against a line of ~**80%** at **84%** elapsed
        — and the 91st audit's B1, B2 and B3 are all executed and committed
        (`563178e`, `6022447`, `9da23c6`). The date may still be the right one
        on CAPACITY grounds (09-13 carries 14 rows against a measured capacity
        of 6; 09-15 carries 2), and that is untouched. What has expired is the
        AVAILABILITY reason written on the row: the input can exist before
        09-15 if this desk wants it to. `W1.01`/`W1.03`/`W1.04` registration is
        unstarted and is not on the builder's ordered list, so whether it
        outranks the 09-14/09-15/09-16 items is a ruling, not a builder choice.
        Note also that the release was *unconditional* on 09-14 at the week
        reset regardless, so this premise had a known expiry when it was
        written.

    ANSWER, 2026-09-16 (Review DAILY) — **THERE IS NO RE-PARENT OWED. The row
        asked for the wrong artefact, which is why three sittings could not
        write it.** Dates broken: 09-06, 09-10, 09-15 — three, and the fourth
        would have triggered the standing stop-rule. Rather than set it, the
        design is delivered.

        **The fifteen, recomputed from the live registry at 249 (not quoted
        from the 08-31 walk at 211 — the number is unchanged and that is
        itself worth recording):** `LC.03` -> 8 (`DP.01`, `DP.02`, `DP.03`,
        `LC.04`, `LC.05`, `LC.06`, `OP.01`, `PS.04`); `UB.10` -> 5 (`TA.03`,
        `UB.11`, `UB.12`, `UB.13`, `UB.16`); `T3.06` -> 2 (`T5.06`, `T5.08`).
        Fifteen exactly. `LC.07` -> 4 (`GEN.02`, `GEN.03`, `GEN.06`,
        `GEN.09`) is the companion row's set and is answered there.

        **The distinction the row never drew, and everything follows from
        it.** A re-parent is the correct repair when a spec has the WRONG
        PARENT. It is NOT the correct repair when a spec has the right parent
        and that parent had a failed RUN. All three weld roots here are the
        second case:

        - `LC.03` **VOID** — the learning-core screen ran and did not
          arbitrate. Its 8 dependents genuinely need what `LC.03` was going
          to give them. Moving them off it would be relabelling a debt as a
          graph edge.
        - `UB.10` **VOID** on a marginal floor — a measurement, not a
          mis-wiring. Its 5 dependents want the same measurement.
        - `T3.06` **VOID** on a marginal floor — same.

        **So the design is: SUCCESSOR SPECS, NOT SURGERY**, and this project
        already has four worked instances of the pattern — `SM.02 -> SM.03`,
        `BA.02 -> BA.03`, `SH.01 -> SH.02`, `D1.0`'s successor. A successor
        inherits the root's arena and leaves every dependent's `depends_on`
        **untouched**; the weld dissolves when the successor passes, and
        nothing in the dependency graph is edited at all. That is also the
        only version of this that is safe: editing 15 `depends_on` entries to
        point somewhere reachable would shrink `unreachable` (floor 97) by
        RE-LABELLING rather than by repair, which is precisely what `T0.31`
        was gated to forbid after three instruments each paid a "repair" that
        lowered its own number.

        **Consequence — the three repairs already have owners, and none of
        them is this row:** `UB.10`'s arm redesign is this desk's outstanding
        debt; `T3.06`'s repair-arm pick is its own queue row; `LC.03`'s
        successor is `D10`'s seated `A4` and therefore downstream of **`D29`**
        (the mandatory collapse diagnostic, DUE 2026-09-18, this desk's).
        **The row was dated three times against `W1.01`/`W1.03`/`W1.04`
        REGISTRATION and that was never its blocker** — W1 registration
        decides where a FUTURE world-line spec hangs, not what happens to
        fifteen specs behind three VOIDs. Three availability forecasts were
        written about a builder meter that had nothing to do with the
        question.

        **The one spec in this family that DOES need a re-parent is not among
        the fifteen.** `LG.11` (`told-world-has-no-rung`, ACTED this morning,
        `81fdaba`) depends on `LG.00`, which PASSES — and whose "life" corpus
        is synthesised by an RNG. That is a WRONG PARENT: the dependency
        claims to supply lived ground and supplies none. `LG.11` re-parents
        onto the W1 line when it exists. The contrast is the test of the rule:
        re-parent when the edge is false, write a successor when the run
        failed.

        **Bills.** SEMANTIC: none — no spec is edited by this answer.
        MECHANICAL: none — no `depends_on` changes, so no certificate stales
        and `UNREACHABLE` (floor 97), `goal_unrunnable` (7) and
        `GOAL_UNRUNNABLE_BASELINE` are all unmoved in both directions. The
        row closes having spent nothing.
    Question: which of the specs welded behind foreclosed/parked roots get
    re-parented off those roots, and onto what evidence. The set, computed
    over `depends_on` at registry 211 (2026-08-31): **15 specs**, from three
    contributing roots — LC.03's 8 (LC.04, LC.05, LC.06, DP.01, DP.02, DP.03,
    OP.01, PS.04), T3.06's 2 (T5.06, T5.08), UB.10's 5 (UB.11, UB.12, UB.13,
    and second-order TA.03, UB.16). The 54th audit's "13" counted only
    first-order behind UB.10; the transitive walk adds the last two. BA.03's
    declared radius is none and nothing depends on SM.02, so those two roots
    weld nobody — their cost is champion-ring reachability, already counted
    by ARENA-UNREACHABLE.
    **DP.02 is the audit's named case and the cheapest call:** "lesion the
    shared trunk, both modes degrade together" is a probe on a trained core,
    not a claim that needs the five-way screen to have returned two learners
    — yet it sits at DP.02 <- DP.01 <- LC.04 <- LC.03. Candidate re-parent:
    the post-D10 seated core plus its scale-transfer challenger spec
    (registered in D10's firing commit, depends_on LC.00-LC.02), which is
    also the natural new parent for the LC.04-LC.06 chain once D10's default
    amends LC.04's premise ("the screen IS the arbitration when it returns
    exactly one").
    Precedent, both directions: re-parenting UB.1-UB.8 off T2.01 made eight
    specs immediately runnable (LESSONS.md), and a challenger registered as
    a NEW spec bills nothing (T1.02 precedent).
    Staleness bill: **NONE, verified not assumed (2026-08-31).** SEMANTIC:
    `depends_on` is a SPEC_CLAIM_FIELDS member, so re-parenting moves each
    spec's spec_sha — but all 15 have zero ledger rows (checked directly), so
    no bought verdict drifts. MECHANICAL: zero certificates cite
    `registry.py`/`registry_expansion.py` in IMPL_DEPS (grepped
    experiments/tests/). The only cost is design attention, which is why this
    is routed rather than done: which parent each spec gets decides what its
    claim MEANS, and that is the Review's desk.

ROUTED: me11-every-arm-hits-the-same-infeasible-branch | 2026-08-31 | 23d53c7 (55th audit §8; rows e3824bf ME.11.C, 459eeb1 ME.11.D) | ACTED 2026-08-31 (option (a), ordered by the Review FULL 08-31 and executed in 7549b79: ME.11.E and ME.11.F recorded VOID-FORECLOSED by runs that verify the arithmetic LIVE — E re-measured lex recall@1 0.0 AND lex gold-score-max 0.0 at full retrieval depth, F re-measured recall@50 0.475/0.381/0.463 — with leaky-cue aliveness floors and parent-row replays that ERROR if any cited row ever changes; blast radius none; the semantic-retrieval redesign need is carried by the T2.10 paraphrase-venue conjunct, PROGRESS 08-31 Part 3 item 2, not a new row)
    DUE: 2026-09-06 | a family-level disposition for ME.11's two remaining
        arms (E, F), owed by the next Review FULL run. Both are known-outcome
        runs against the 0.80 parent hypothesis by the arithmetic below;
        what needs deciding is whether they run anyway for their secondary
        gates, or the family settles with the invariant recorded.
    Question: five distinct encoder configurations, static and contextual,
    all hit the SAME pre-registered INFEASIBLE branch on all three seeds —
    the invariant is evidence about the rig at least as much as the arms
    (55th audit §8; f66a5be: "a gate can be too STRONG to be met"):
        arm B  (bm25s+Snowball)   recall@1 0.0000  ceiling —      n/a (lexical zero PROVEN)
        arm C  (potion-base-8M)   recall@1 0.0437  ceiling 0.123  tau_fpr 0.365 > tau_cov 0.184, 3/3 seeds
        C var  (potion-base-2M)   recall@1 0.031   —              INFEASIBLE
        C var  (mrl-en-v1@256d)   recall@1 0.015   —              INFEASIBLE
        arm D  (all-MiniLM-L6-v2) recall@1 0.0667  ceiling 0.250  tau_fpr 0.388 > tau_cov 0.227, 3/3 seeds
        D var  (bge-small)        recall@1 0.067   —              INFEASIBLE
    The parent hypothesis requires paraphrase recall >= 0.80. The best
    unthresholded ceiling any arm measured is 0.250 — the target is 3.2x
    above the credulity-free maximum of the best arm tried.
    **THE NUMBER THAT SETTLES ARM F, measured 2026-08-31 before implementing
    it (55th audit B2; scripts/probe_me11c_recall_at_k.py, reusing ME.11.C's
    own index/model code, seeds 0/1/2, certified stem-disjoint fixture):**
        Arm C recall@50 unthresholded: 0.475 / 0.381 / 0.463  (mean 0.4396)
        Arm C recall@10 unthresholded: 0.294 / 0.238 / 0.306  (mean 0.2792)
    F's premise — "Arm C retrieves top-50 (pilot recall@10 was 1.000, so the
    answer is present)" — is falsified on the certified fixture: the answer
    is ABSENT from the top-50 on 56% of cues. A PERFECT reranker is capped at
    0.44 before the abstention threshold even applies, and F's abstention is
    pinned by control to C's first stage, whose conformal arithmetic is
    INFEASIBLE on every seed. That is the FOURTH pilot number this family has
    falsified on the certified fixture (485 docs/s, 18-min reindex, int8
    slower, and now recall@10=1.0) — the pilot family and the certified
    fixture are different distributions, and pilot numbers must not size or
    justify any further ME.11 arm.
    **E's arithmetic, stated plainly (55th audit B2 asked):** E's OWN gate
    (beat both parents on recall@1 at fixed abstention, parents 0.0000 and
    0.0667) is reachable in principle — but its MECHANISM is dead on this
    fixture: the lexical parent scores 0.0000 on all 160 cues x 3 seeds
    (proven a ceiling, not a dead rig — ME.11.B), so the fusion has nothing
    to add exactly where its hypothesis says lexical should help, and the
    0.80 parent hypothesis is out of reach by the 0.250 ceiling regardless
    of the weight w. E cannot decide the family; at best it re-measures D.
    Options for the Review, all runnable or declarable, not an argument:
    **(a) settle the family** — declare E and F VOID-FORECLOSED with the
    arithmetic above as FORECLOSURE ARITHMETIC (blast radius to compute at
    declaration; ME.3's offline retriever interest in F noted), and route
    the semantic-retrieval need to a redesign row (different fixture bar,
    different encoder class, or GPU-scale encoder as a new spec); **(b) run
    them anyway** for the secondary findings (F's recall/latency curve, E's
    fitted-w costume check), each ~one cpu iteration, with the known-outcome
    stated in the journal at dispatch; **(c) re-examine the 0.80 bar's
    provenance** — if it was sized on the pilot family (the distribution
    that has now been falsified four times), the bar itself may be the rig
    defect, and per law 4 that is said in a commit message and recorded,
    never quietly moved.
    Staleness bill: NONE — neither E nor F has a test file or a ledger row;
    C and D are settled FAIL and stay settled regardless of disposition.
    UPDATE 2026-09-02 (builder; ME.11 FAMILY VERDICT RECORDED — no decision
    taken, data attached for the 09-06 disposition): the parent ME.11 is now
    SETTLED FAIL (attempt 1, impl 2e12d1f, ran 08:19:02, seeds 0/1/2,
    81.6 s) — the GOAL.md memory commitment behind it moved from unmeasured
    to measured, bars untouched. The verdict run re-bought the deciding row
    live (Arm D via the family's shared pipeline: recall 0.0667 +- 0.0147,
    ceiling 0.250, tau_fpr 0.388 > tau_cov 0.227 — identical to the recorded
    row), rig fully alive (lexical AND dense leaky 1.0, lexical null 0.0),
    verbatim 1.0, and all six family rows are now PINNED: re-running any arm
    to a different answer makes ME.11 raise instead of citing it stale.
    ONE NEW NUMBER, measured by the registry's own distractor-store control
    (each cue's gold masked out, the topically-similar rest of the life
    remains, tau calibrated identically): the best dense arm ANSWERS on
    12.29% +- 1.56% of cues whose true target is ABSENT — distractor
    abstention 0.877 vs the 0.95 the claim requires — while finding only
    6.67% when the target is present. At the family's best operating point
    confabulation is ~1.8x as frequent as correct recall. A redesign under
    (a) or (c) inherits that asymmetry as the thing to beat, not just the
    0.250 recall ceiling: this venue's semantic scorers invent more easily
    than they find, exactly as the registry's control note predicted.

ROUTED: lt01-c2-body-cannot-rise | 2026-09-01 | a0e6011 (LT.01 attempt 1, FAIL, 3 seeds x 3000 decisions) | ACTED 2026-09-19 in 4091066 (builder EXECUTED the 09-06 disposition: C2' implemented at b16de57 with the adversarial adhesion-enabled height-seeker, and LT.01 attempt 2 ran to PASS at 4091066 — 2017 s, 3 seeds, FOREGROUND in-session after the 06:09 background attempt died with its slot. C2' resolved to branch **G-adv**: the null could not game raw torso height (nonladder_rise_max 0.084 +/- 0.067 m) but the privileged height-seeker DID — adv_rise_max 0.6157 m, adv_ge_bar 0.667 over the UNMOVED 0.6 m bar, adv_engaged 0.0. Raw height is GAMEABLE, h(t) necessary, the original C2 claim restored as a measurement. The disposition required exactly this arm; the re-run carries it. LT.02 and the LT.03-LT.07/LT.09 chain — the Curiosity-signal seat's arena — unblock behind this PASS. Staleness bill NONE as the disposition stated: LT.01 had one FAIL row and no certificate cited it. Supersedes the prior status, preserved verbatim per the t108 repair precedent of 09-18: DISPOSITIONED 2026-09-06, Review FULL — option (a), the re-scope specified in the C2' block below; design only, the builder implements and re-runs, and the 0.6 m bar does not move in either branch)
    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B2) — the fact, not a new
        promise, and NOT a re-date: this row was dated to be decided in the SAME
        window as `w0-too-shallow`, on the ground that both turn on the identical
        world-or-body fork. The 2026-09-13 FULL sat this morning and did not take
        up W1, so the paired window did not happen and this row goes OVERDUE at
        00:00 tonight. Full record on `w1-world-edit-window`.
    DUE: 2026-09-06 | a disposition for LT.01's C2 clause, owed by the Review's
        Sunday FULL run and decided in the SAME window as `w0-too-shallow`,
        because both turn on the identical fork (is the repair the world, or
        the body?). Opened by the Review 2026-09-01 (DAILY) as a row rather
        than left as an UPDATE paragraph inside `w0-too-shallow`: the
        instrument COUNT belongs in the aggregate row — that was the 08-31
        finding and the builder applied it correctly — but the owed ACTION
        does not, because `run review-queue` prints row titles and not row
        bodies, and an owed redesign that only exists 200 lines inside another
        row is exactly the shape of `wk4-N3`, which was ordered as prose on
        2026-08-25 and read by nobody for six days.
    Question: LT.01's C2 clause pre-registered — from the 2026-08-09 pilot,
    whose free-roam z ceiling was 1.007 m — that a random agent reaches
    >= 0.6 m of NON-LADDER torso rise, so that raw height is demonstrably
    gameable and a ladder-specific h(t) is therefore necessary. On the
    as-built rover body the recorded row reads `nonladder_rise_max`
    **0.084 +/- 0.067 m**: the body tips within seconds and travels by
    dragging. The clause is FALSIFIED, and it is falsified by a fact about
    the BODY, not about the instrument — every aliveness guard was green
    (force calibration +1.000 W, scripted hang ENGAGED through the full h(t)
    conjunction, oracle rise 0.416 m) and the other three claim clauses all
    HELD (null floor exactly 0 engaged attempts; P(hang | 3 s burst) 0.031
    inside the pre-registered [0.01, 0.05] bootstrap band; platform
    unreachable by free-roam AND by the adhesion-disabled oracle).
    Why it is worth a row: LT.02 and LT.03-LT.07/LT.09 are welded behind this
    FAIL — **frees 7, blocks 9** — and LT.03/LT.04 are the Curiosity-signal
    seat's ENTIRE arena. That seat is held BY ANALYSIS, has never been
    defended, and curiosity is GOAL.md's north star. The arena was registered
    on 2026-08-31 and welded shut on 2026-08-31, inside one day.
    THE CIRCULARITY, computed 2026-09-01 and the reason this cannot simply
    wait for the humanoid: **D9's default (fired 2026-09-01) parks the body
    question "until the playground-humanoid line", and the playground-humanoid
    line is `LT.08`** — `depends_on = [LT.07, T2.01, T2.02]`, and LT.07 sits
    at the end of the LT.01 -> LT.03 -> LT.05 -> LT.07 chain. So the body
    question is parked behind a spec chain whose FIRST link failed because of
    the body. `BA.02` is re-parented behind the same LT.08 by D8. Neither
    default is wrong on its own terms; the deadlock is a joint property that
    only appears when the two are read together, and no organ reads them
    together.
    Options, all declarable rather than arguable: **(a)** re-scope C2 to a
    non-rise gameability check that the as-built body CAN exercise (strictly
    a different measurement, and it must be shown at least as hard to game);
    **(b)** hold LT.01 and route the body itself — register `W0.BAL` as a
    spec id so the body gets a seat and arm C's upright 1.000 vs 0.002-0.004
    becomes a defended verdict rather than a parked bakeoff (PROGRESS 08-31
    FOR THE OWNER 1, still on the owner's desk); **(c)** decide that the LT
    family's venue is wrong and re-parent the whole arena onto whatever body
    `w0-too-shallow` produces. THE THRESHOLD RULE BINDS: 0.6 m may not be
    lowered to make C2 green. A re-scope is legitimate only if the EXPERIMENT
    is wrong (T1.02 precedent), and the old spec version stays in the ledger's
    history.
    Staleness bill: NONE for (a) — LT.01 has one FAIL row and no certificate
    cites it. (b) and (c) bill nothing either; (c) defers to whatever bill
    `w0-too-shallow` chooses.
    DUE: 2026-09-24 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. The DESIGN IS DELIVERED and this row is DISPOSITIONED: what this date owes is EXECUTION by the builder, not a decision by this desk. Dated where a builder slot can plausibly reach it rather than onto the Review's own calendar. | IMPLEMENTATION of `C2'` below plus LT.01 attempt 2,
        owed by the BUILDER. This row stays LIVE until a commit carries the
        adversarial height-seeking arm; a re-run without that arm is NOT this
        disposition and must not be stamped against it.

**DISPOSITION (Review FULL, 2026-09-06): option (a), and here is the exact
re-scope. Read the objection first, because this is the shape the law
forbids and I have to show why it is not that.**

**The objection.** Re-scoping a clause of a FAILING spec so the spec can pass
is, on its face, the one act I am forbidden. `LT.01` FAILs, seven specs are
welded behind it, the Curiosity-signal seat's entire arena is among them, and
I have an obvious motive.

**Why the EXPERIMENT is wrong, in one line that decides it.** The spec is
titled ***"The Ladder Test is measurable: null floor and un-gameable rise"***
— and it FAILED because the rise turned out to be **un-gameable**. `C2`
pre-registered, from a 2026-08-09 pilot on a different body whose free-roam z
ceiling was 1.007 m, that a random agent *must reach* ≥ 0.6 m of NON-LADDER
torso rise. That requirement exists as a NECESSITY ARGUMENT for the
ladder-specific `h(t)` metric: *show raw height is gameable, therefore h(t) is
needed.* On the as-built rover body the reading is **0.084 ± 0.067 m** — the
body tips within seconds and travels by dragging. So the clause demands, as a
precondition of the claim, an observation whose absence is the claim's own
title. That is not a hard bar the system failed to clear; it is a
pre-registration that encoded the wrong sign, and its failure carries **zero
information about whether the Ladder Test is measurable**. Every other clause
HELD: null floor exactly 0 engaged attempts, P(hang | 3 s burst) 0.031 inside
the pre-registered [0.01, 0.05] band, platform unreachable by free-roam AND by
the adhesion-disabled oracle, force calibration +1.000 W, scripted hang ENGAGED
through the full `h(t)` conjunction, oracle rise 0.416 m. This is the T1.02
precedent on its facts, and the attempt-1 FAIL row stays in history.

**`C2'` — the replacement, and it is STRICTLY HARDER than `C2`.** The clause
becomes a two-branch necessity test in which **the branch taken is RECORDED on
the ledger row**, and the `0.6 m` bar is unchanged in both:

  **Branch G — GAMEABLE (the original observation).** A null reaches ≥ 0.6 m
  non-ladder torso rise → raw height is gameable, `h(t)` is necessary,
  identical to `C2` as written. Nothing about this branch changes.

  **Branch U — UN-GAMEABLE, and it must be EARNED, not inferred from an
  absence.** To record `un-gameable`, the run must show BOTH: (i) the null's
  non-ladder rise ceiling is < 0.6 m — the reading `C2` already has — AND
  (ii) **an ADVERSARIAL height-seeking arm also fails to reach 0.6 m**: a
  privileged arm with adhesion ENABLED, scored on raw torso height alone,
  explicitly optimised to maximise it while never engaging the ladder, run at
  the same seeds. Only if a policy that is *trying* to game raw height cannot
  do it is "un-gameable" a measurement.

**That is the strengthening, stated plainly:** `C2` as written had **no
adversarial arm at all** — it inferred gameability from a random agent and
would have inferred un-gameability from that same random agent's silence.
`C2'` requires a run to defeat a deliberate gamer before it may claim the
metric is safe. A spec that could previously conclude "un-gameable" from a
null doing nothing must now beat an optimiser doing its best.

**And `h(t)` STAYS under Branch U.** Un-gameability is a property of THIS body,
measured today; a future body that can rise re-opens Branch G. Deleting the
ladder-specific metric because the current body is too weak to threaten it
would be exactly the kind of convenience this file exists to refuse.

**WHAT THIS DOES NOT FIX, and I will not let it be read as fixing it.** The
`D9`/`D8` DEADLOCK is untouched: `D9`'s default parks the body question until
the playground-humanoid line, that line is `LT.08`, and `LT.08` sits behind the
`LT.01 → LT.03 → LT.05 → LT.07` chain whose first link failed *because of the
body*. Option (a) ROUTES AROUND that deadlock; it does not dissolve it, and
nobody should read a green `LT.01` as evidence the body question was answered.
Option (b) — register `W0.BAL` so the body gets a seat and arm C's upright
1.000 vs 0.002–0.004 becomes a defended verdict — went to the owner on
2026-08-31 and is still on the desk. It stays there, and I am re-raising it in
this run's FOR THE OWNER rather than deciding it: a joint property of two
armed defaults is not mine to unpick.

ROUTED: five-commitments-are-claim-dead-behind-foreclosures | 2026-09-01 | adca793 (58th audit F1) + the B1 repair commit | OPEN
    DUE: 2026-09-06 | successor specs or re-parenting for the dead
        commitments, owed by the Review's Sunday FULL run in the SAME window
        as `w0-too-shallow`, `ba03-null-saturates-the-horizon`,
        `t306-matched-magnitude-noise-buys-coverage`,
        `lt01-c2-body-cannot-rise` and `reparenting-the-welded-fifteen` —
        four of the five commitments are downstream of the same W0 venue
        findings, so they sequence into ONE design window (the bundling
        rule).
    DUE: 2026-09-11 | the same successor/re-parenting decision, moved to the
        Friday DAILY — RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2,
        builder): the most downstream row on the board — successor specs need
        the W0/W1 design (09-06), the venue arm picks (09-09) and the
        re-parenting outcome (09-10) as INPUTS, so it goes last in the
        staggered docket; "one design window" here means one week decided in
        dependency order, not one sitting.
    DUE: 2026-09-16 | RE-DATED (Review DAILY 2026-09-11) on a DEPENDENCY that
        is concrete and checkable, not on capacity. This row's own text above
        names *"the re-parenting outcome"* as an INPUT. That input is
        `reparenting-the-welded-fifteen`, which the Review DAILY 09-10 bundled
        with `goal-cites-four-specs-that-resolve-to-corpses` into one question
        **DUE 2026-09-15** (`20ba75f`) — so on 09-11 this row's declared input
        does not exist yet, and a successor/re-parenting decision taken today
        would be taken blind to the registry surgery that determines what the
        five commitments can be re-parented ONTO. 09-16 is the first date after
        its input lands, and it carries 2 rows. **The `CLAIM-DEAD` ratchet
        stays RED at 4 and `coverage` keeps exiting rc=2 until this row is
        acted on — that red is the tool working and it is not to be quieted,
        unparked, or answered with a successor spec against the same venue the
        pilots already measured as unable to grade it.**
    Question: `balance`, `smell`, `shelter/building` and `thermal (kills)` —
    four of the owner's own 2026-08-09 survival directives — have zero
    passing claims and every claim-kind spec PARKED or FORECLOSED
    (BA.03 VOID-FORECLOSED + BA.02 parked; SM.03 PILOT-BLOCKED + SM.02
    parked; SH.02 PILOT-BLOCKED + SH.01 parked, carrying shelter AND thermal
    together). The CLAIM-DEAD ratchet now sees this: `coverage.foreclosure()`
    is the shared conjunction, `FORECLOSED` the fifth reachability state, and
    the count went 0 -> 4 with coverage rc=2 — the red is the tool working,
    and it stays red until this row is acted on. The repair is REGISTRATION
    or RE-PARENTING, never unparking, quieting, or a successor spec written
    against the same venue the pilots already measured as unable to grade the
    claim (SH.02's pilot: "the null already holds the roof it was placed
    under"; SM.03's: the held-out split saturated at the 0.25 floor before
    the nose was ever measured).
    THE FIFTH COMMITMENT, carried here because the ratchet honestly cannot
    count it: `fast/slow` is claim-dead IN FACT (the 58th audit's five-table
    stands — nothing anybody may run) but not by the predicate, because
    DP.01/DP.02/DP.03 are BLOCKED behind LC.03 (itself VOID-FORECLOSED, so
    that blocker resolves never) and BO.01 is BLOCKED behind DP.05's FAIL
    (which a W0 redesign could re-open). Blocked-is-alive is the ratchet's
    founding distinction, and widening it to blocked-behind-FAIL would flood
    the count with every commitment behind T2.01. If the Review wants the
    transitive case counted, the honest instrument is a SIXTH state
    (transitively-foreclosed: every terminal blocker parked or foreclosed),
    which would catch DP.01-03 and still honestly leave BO.01 — and
    `fast/slow` — alive on one thread. That is a design choice with its own
    flood risk, routed here rather than decided by the builder.
    Staleness bill: NONE mechanical for acting (registering successor specs
    edits the registry, which no PASS certificate hashes beyond T0.21's
    ordinary coverage.py re-buy). Any repair that instead edits W0 inherits
    the 21-certificate playground.py bill already computed on
    `w0-too-shallow`.
    DUE: 2026-09-29 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. The most downstream row on the board: successor specs need the W0/W1 design as INPUT, and that design is behind the 09-23 edit window. Four CLAIM-DEAD commitments sit here, which is why it is not pushed further despite being the least ready.

---

## ROUTED 2026-09-01 (builder, 59th audit B4): `d10-learning-gate-uses-two-different-denominators` — "noisy" and "did not learn" share one verdict

ROUTED: d10-learning-gate-uses-two-different-denominators | 2026-09-01 | 59th-audit-B4 | ACTED 2026-09-10 (Review DAILY — executing commit `8f2990d`, 2026-09-06 08:19, three days INSIDE this row's own clock: G1 replaces the two-denominator statistic with a paired-own-twin mean(d)/(sd(d)/sqrt(n)) at an unmoved 3.0 bar, so "noisy" and "did not learn" no longer share one verdict. Verified red-first — all three fixtures passed the old gate and each now hits its named conjunct — and G1 was checked through the real `_experiment` arithmetic. The commissioned run landed: D1.0 attempt 2, `3a4ccfd`, 2026-09-07, VOID under the executed gate, and its successor is owned by the live row `d10-successor-rerun-under-adopted-gate` DUE 09-14. Nothing is buried by this marker)
    DUE: 2026-09-06 | gate-design decision owed by the Review; bundle with `w0-too-shallow`'s window if the venue is judged the common cause
    DUE: 2026-09-09 | EXECUTION of the adopted gate in the D1.0 family's
        scoring code, owed by the BUILDER, not by me — the design exists as
        of 2026-09-06 and this row stays LIVE until a commit implements it.
        Dated 09-09 rather than 09-08 so it lands AFTER
        `d10-successor-rerun-under-adopted-gate` (DUE 09-08) has stamped the
        either/or it owes, and so the gate is committed before W36's
        16-hour dispatch is spent rather than during it.

**What was measured (D1.0 attempt 1, VOID 2026-09-01, metrics on the ledger
row — correct and untouched).** The learning gate scores
`(arm_mean - random_mean) / max(arm_std, rnd_std)` at n=3 seeds / 5 eval
episodes. Three arms happened to be scored against random's spread; `c_e2e`
was scored against its OWN spread, because its seed means (319 / 536 / 358)
spread wider than random's. `c_e2e` returned 404.3 against random's 108.7 —
a 3.7× gain — and is recorded as not having learned (2.56σ vs the 3.0 bar).
It failed a CONSISTENCY test, and the ledger calls it a LEARNING failure;
the owner's copy of this audit flags that "the end-to-end arm did not learn"
is the sentence that would otherwise enter the record, and it is not what
was measured.

**Options to weigh (the audit's, not decided here):** a paired t-statistic;
a fixed random-spread denominator; more eval episodes; or an explicit
separate consistency gate so the two failure modes stop sharing one verdict.
Any change is a gate redesign for FUTURE D1.0-family runs — the recorded
VOID stands per T2.02 precedent and nothing re-runs on this row's account.

**Design input on the record (builder, 2026-09-03, from processing the
integration queue's D1_CONTROL_ARCHITECTURE.md row):** the source doc's §3/§6
(the unregistered T2.21 draft) pre-registers exactly the machinery this row
weighs — an EXTERNAL reference arm (verbatim SB3, ≥450 return) whose failure
VOIDs the run as a harness fault rather than recording a learning verdict on
any arm; per-arm untrained twins as the binding null (the sibling row's
option); and a shuffled-z control for percept-fed arms. Evidence, not a
recommendation; the doc's frozen arms stay struck per D1's resolution.

## ROUTED 2026-09-01 (builder, 59th audit B4): `d10-learning-gate-sits-at-the-untrained-twin-level` — the control passed by 0.04σ

ROUTED: d10-learning-gate-sits-at-the-untrained-twin-level | 2026-09-01 | 59th-audit-B4 | ACTED 2026-09-10 (Review DAILY — executing commit `8f2990d`, 2026-09-06 08:19, three days INSIDE this row's own clock: all three conjuncts live in `d1_0_control_path_bakeoff.py` — G1 the paired statistic, G2 the consistency conjunct that returns INCONSISTENT rather than "did not learn", G3 the verbatim SB3 reference lane run FIRST in its own kernel at floor 450 so a harness fault VOIDs before 15 h of arm kernels are submitted. Reference lane construction-smoked at 124,707 params, T2.02's `mlp_params` to the digit. The commissioned run landed: D1.0 attempt 2, `3a4ccfd`, 2026-09-07, VOID under the executed gate; successor owned by `d10-successor-rerun-under-adopted-gate` DUE 09-14. **Why this was two OVERDUE violations this morning and not two undone jobs:** the row recorded the work in its own prose as `EXECUTED` on 09-06 and never carried the `ACTED` marker `review_queue.py` actually reads — the same family as the 09-09 scar where six DUE clauses were written above the line the tool takes. The desk keeps writing the truth in a place the instrument does not look)
    DUE: 2026-09-06 | gate-design decision owed by the Review; same bundle judgment as the denominators row
    DUE: 2026-09-09 | EXECUTION of the adopted gate, owed by the BUILDER; same
        dating reason as the denominators row, which this block also governs.
    EXECUTED 2026-09-06 08:3x (builder, `8f2990d`) — all three conjuncts live
        in `d1_0_control_path_bakeoff.py`, verified red-first (each fixture
        passed the old gate, each now hits its named conjunct); registry
        claim text amended in the same commit; G3's reference lane
        construction-smoked locally (124,707 params, T2.02's `mlp_params` to
        the digit). Covers BOTH d10-* gate rows, three days inside the 09-09
        clock.
    ORDERED: D1.0 | attempt 2, dispatched 2026-09-06 ~08:2x under the executed
        gate; its watcher lands the row (do not relaunch). Recorded 2026-09-07
        00:xx (builder) per the 79th-audit join, the second live use after
        `w0-too-shallow`. Until that row lands, the join prints attempt 1's
        VOID of 2026-09-01 — a PRE-commission verdict, visible as such by its
        date; the commissioned return is attempt 2, expected ~09-07. The
        sibling denominators row and `d10-successor-rerun-under-adopted-gate`
        read the pair here rather than repeating it.

**THE ADOPTED GATE (Review FULL 2026-09-06). It governs BOTH `d10-*` rows, and
it is STRICTLY HARDER than the gate that produced the VOID.** I am not bundling
these into `w0-too-shallow`: the venue may well be a common cause of much else,
but neither of these two defects is a venue defect. Both are scoring defects,
both are visible in the recorded row's own arithmetic, and holding a scoring
repair behind a world redesign would leave the project's largest unblock
(`T2.01`, frees 35) waiting on a decision it does not depend on.

Three conjuncts, ALL required — a run passes the learning gate only if all
three hold. The conjunction is the reason this is a strengthening and not a
re-slicing: nothing that failed the old single statistic can pass by being
re-described.

  **G1 — LEARNING, scored against the arm's OWN untrained twin, paired by
  seed.** Replace `(arm_mean - random_mean) / max(arm_std, rnd_std)` with a
  PAIRED per-seed statistic against that arm's own untrained twin:
  `d_i = arm_i - twin_i` over the shared seeds, and the bar is met when
  `mean(d) / (sd(d)/sqrt(n)) >= 3.0`. The 3.0σ bar does not move (SYSTEM.md
  law 4). Why this is harder, in the recorded numbers: the untrained twins
  read **2.96σ and 2.94σ against random**, i.e. the null this gate used was
  one twentieth of a sigma away from clearing the bar by architecture alone.
  Any arm scored against that random baseline is being credited for its
  architecture's bias. Scoring against its own twin removes exactly that
  credit, and it removes it from every arm including the one that came
  closest. It also DISSOLVES the denominators row: a paired statistic has one
  denominator by construction, so "scored against its own spread" cannot
  recur, and the choice of which spread to use is no longer a choice.

  **G2 — CONSISTENCY, as its own named conjunct.** The audit is right that
  "noisy" and "did not learn" shared a verdict, and it is right that they are
  different findings. They are separated by ADDING a conjunct, never by
  relaxing one: an arm must ALSO hold `sd(arm seed means) / |mean| <= 0.50`
  (c_e2e's recorded seed means 319/536/358 give 0.29 — it PASSES this and
  always would have; the conjunct exists so that when an arm fails it, the
  ledger says `INCONSISTENT` and not `did not learn`). The verdict token is
  what changes. The pass set does not widen: G2 can only ever remove an arm
  from the pass set, never add one.

  **G3 — EXTERNAL REFERENCE ARM, and it VOIDs the run rather than scoring
  anyone.** Adopt the `D1_CONTROL_ARCHITECTURE.md` §3/§6 machinery the builder
  attached as design input on 09-03: a verbatim external reference (SB3 PPO,
  >= 450 return on the venue's own task). If the reference does not clear its
  floor, the run is VOID as a HARNESS fault and NO learning verdict is
  recorded on any arm. This is the conjunct that would have stopped
  "the end-to-end arm did not learn" from entering the record on a run whose
  own harness was never shown to be able to produce learning at all. It can
  only ever turn a recorded verdict into a VOID; it can never turn a FAIL
  into a PASS.

**What this does NOT do.** It does not move the 3.0σ bar in either direction.
It does not re-run anything on its own account: `D1.0`'s recorded VOID stands
per the T2.02 precedent, and attempt 2 is `d10-successor-rerun-under-adopted-
gate`'s business, not this row's. It does not touch W0. It does not re-score
the recorded run — the numbers above are read FROM that row, they are not a
re-derivation of it.

**The one thing I want on the record against myself.** G1 makes the gate
harder, and the arm it makes hardest is `c_e2e`, the arm this project would
most like to see pass. If attempt 2 under this gate returns a FAIL where the
old gate would have given a PASS, that is the gate working, and I am writing
that sentence now, before the run, so it cannot be re-litigated after it.

**What was measured (same run).** The untrained twins read 2.96σ and 2.94σ
against the 3.0σ learning bar — the control cleared by 0.04σ. A gate whose
untrained twin sits at the bar's edge is measuring architectural bias plus
noise, not learning headroom. **Option to weigh:** score each arm against
its OWN untrained twin rather than against random, which also dissolves the
denominators question above for the twin comparison. Same scope note: gate
redesign for future runs only; the recorded VOID stands.

## ROUTED 2026-09-01 (builder, UB.10 attempt-1 harvest): `ub10-seed-fragility-and-saturated-battery` — the unparked design ran honestly and measured two defects in itself

ROUTED: ub10-seed-fragility-and-saturated-battery | 2026-09-01 | UB.10-attempt-1-VOID | ACTED 2026-09-24 (Review DAILY, executing commits `e85d1e5` (parts 2 and 3) and `9bb2d19` (part 1) — the 09-08 disposition ordered THREE parts and all three were executed by the builder on 09-13; verified in `experiments/tests/ub_10_fusion_bakeoff.py` rather than from the journal. Part 2, `A0_HEADROOM = 0.05`, is pre-registered at line 474 and emits `a0_headroom_ok`/`a0_headroom_margin` per seed. Part 3, per-arm stability, is a SCORED DISQUALIFICATION at line 1350 — a fragile TRUNK arm is named and cannot win WITHOUT voiding the rig, a fragile ANCHOR still voids — and the commit mutation-checks the fixture BOTH ways. **Part 1 was executed and returned a FORECLOSURE rather than the recoding it ordered, and that is this desk's error being corrected by a measurement, not a builder's shortfall**: `slot` is ALREADY the cross-modal XOR (`slot = XNOR(vslot, afell)`, structural in `hns_scene`, re-derived 2000/2000 episodes 0 mismatches) and attempt 1 measured `uni_slot_dev_max` 0.0, so the premise of the order — *"no single modality carries the answer"* — was false of this venue before it was written. What saturates is the two MARGINALS (A0 read vslot 1.0, afell 1.0), and by the union bound any f(vslot, afell) is then >= 1.0 for A0, so the WHOLE FAMILY of label re-codings is foreclosed by arithmetic. WHAT THIS STAMP DOES NOT LAUNDER, and the reason it can be terminal: UB.10's `run()` STILL REFUSES — `_BATTERY_REDESIGN_OWED` is still set at line 518 and `_assert_venue_not_foreclosed()` fires on positive evidence from the spec's own committed row — so closing this row cannot make UB.10 look dispatchable to anybody; the guard returns non-zero and no ledger row is written. The replacement question (WHICH ARM pays a marginal its headroom) is not this row's and never was: it is the live row `ub10-part1-premise-false-marginals-are-what-saturate`, DUE 2026-09-28, and it is THIS DESK's decision, not the builder's. Earlier: DISPOSITIONED 2026-09-08 (Review DAILY — the anchor's saturation is the load-bearing defect and is repaired by HARDENING THE TASK, never by shortening training; the per-arm stability conjunct is adopted; the seed-level SCORED-AND-INELIGIBLE retirement is REFUSED as a weakening. Design below)
    DUE: 2026-09-06 | an arm/task redesign decision owed by the Review's
    Sunday FULL run; bundle beside `recipe-sensitivity`'s lineage (this row
    is what its 08-25 disposition, fully executed, measured next)
    DUE: 2026-09-08 | the same redesign decision, moved to the Tuesday DAILY —
    RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the unison
    bakeoff has its own lineage (recipe-sensitivity → the executed 08-25
    disposition → this measurement) and no dependence on the W0/W1 design;
    the bundle it names is a reading order, not a sitting.
    DUE: 2026-09-23 | RE-DATED 2026-09-15 (Review DAILY), BEFORE the break rather than after it, and that is the point. This row is the BUILDER's execution debt and the builder has been PACE-DARK for 18 consecutive hourly slots since 2026-09-14T12:07 ('week:all models' 37% against a 35% pace line; this project's own attributed share of that meter is 25% — 28 of 37 points, 75%, are NOT THIS PROJECT). Leaving a builder-owned row dated on a day the builder provably cannot run is KNOWINGLY MANUFACTURING a violation, and re-dating it at 07:00 with a measured cause is strictly more honest than re-dating it at 07:00 tomorrow with the same cause and a red ratchet in between. Dated AFTER the 2026-09-21 05:00 UTC meter reset. This is the desk declining to let its own calendar launder someone else's outage. Routed to the owner as `D30`. ORIGINAL TEXT FOLLOWS, unchanged. | DECISION DELIVERED 2026-09-08 (Review DAILY): harden the
    TASK (composite / cross-modal-XOR slots) and gate the anchor's headroom;
    the per-arm stability conjunct adopted; the training-budget cut and the
    seed-level SCORED-AND-INELIGIBLE retirement both REFUSED as weakenings —
    full design in the DISPOSITION block at the foot of this row. What is owed
    is the BUILDER's implementation stamp: the hardened battery, the
    A0_HEADROOM rig gate and the stability conjunct, committed BEFORE any
    re-dispatch. Dated onto a day carrying one promise, at this desk's own
    measured capacity of ~1 row/cycle, rather than onto the 09-13 pile of ten.

**What was measured (UB.10 attempt 1, VOID 2026-09-01, kernel
jack-ladder-1788293396, ledger row committed 75aafd5; _check replayed
offline, VOID reproduces; full record in the spec docstring's REGISTERED
RUN RECORD).** The matched-tuning-budget disposition was executed exactly:
K=5 grid, blind first-eligible selection at seed 90, no arm ineligible,
registered dispatch under the selected recipes. Two defects, one run:

1. **Seed fragility.** A2 (lolr_warm) collapsed on seed 0 (vslot 0.5) and
   A3 (lolr) on seed 1 (vslot 0.7 vs floor 0.8) — each clean at seed 90 and
   on the other two registered seeds. Third independent demonstration that
   the dropout arms' training is basin-fragile (pilot 08-20, probe 08-20,
   now the registered seeds). The two legal-looking repairs are both
   illegal: per-registered-seed re-selection tunes on registered seeds;
   a seed re-roll is run-until-pass.

2. **Saturated anchor.** A0 reads slot 1.0 on ALL three seeds; the winner
   ties it (paired_boot_lo -0.0104, ranking gap 0.0). The PASS conjunct
   'winner > A0 on every seed' cannot fire against an anchor at ceiling —
   at this budget the fused battery discriminates nothing among healthy
   arms. Had the marginals held this would have been the pre-registered
   A0-tie FAIL, but the deeper fact is the task, not the tie: kin to
   `sh02-null-saturation` (a null with no headroom) and a fusion-scale
   echo of `w0-too-shallow`.

**Options to weigh (not decided here):** harden the battery (more slots /
composite XOR variants / lower training budget so accuracy leaves ceiling);
per-arm stability conjunct (an arm must train on all registered seeds to
hold a seat — turns fragility into a scored finding instead of a rig
VOID); retire the marginal floor VOID in favour of SCORED-AND-INELIGIBLE
at the seed level. Any change is spec redesign under the T1.02 precedent —
the recorded VOID stands, nothing re-runs on this row's account, and UB.11
(which Review 08-31 item 4 needs for T2.12's conjunct) stays blocked behind
an UB.10 verdict this row's redesign must first make reachable.

    CLOCK (the parsed `DUE:` for this row is the one in its header above; this
        copy sits beside the design and is deliberately NOT declaration-shaped)
        — DUE 2026-09-15, DECISION DELIVERED 2026-09-08 (design below). What is
        owed is the BUILDER's implementation stamp on this row: the hardened
        battery, the A0-HEADROOM rig gate and the per-arm stability conjunct,
        committed BEFORE any re-dispatch. Dated onto a day carrying one
        promise, at this desk's own measured capacity of ~1 row/cycle, rather
        than onto the 09-13 pile of ten.

**DISPOSITION (Review DAILY, 2026-09-08): defect 2 is the one that matters, and
the two defects must not be repaired in the same direction.**

**Order of importance, because the row lists them the other way round.** Seed
fragility (defect 1) is an ARM result — two dropout arms cannot train reliably,
which is a finding about those arms and a third independent demonstration of
it. Anchor saturation (defect 2) is an INSTRUMENT result: A0 reads slot 1.0 on
all three seeds, so the PASS conjunct *"winner > A0 on every seed"* cannot fire
against anything. **Repairing fragility alone buys a run that still cannot
return a verdict.** Defect 2 is therefore the unit of work and defect 1 rides
with it.

**ADOPTED (defect 2): harden the TASK, and gate the anchor's headroom.**
Of the three offered repairs for saturation the row lists — more slots,
composite XOR variants, lower training budget — **the training-budget cut is
refused.** Shortening training to move the anchor off its ceiling makes every
arm worse in order to make the picture interesting, and it confounds the claim
this battery exists to test: *"fusion helps"* would become *"fusion helps when
undertrained"*, which is a different and much weaker sentence. The honest
repair for a null at ceiling is a task the null cannot solve. So:

  - **Composite / cross-modal-XOR slots**, whose defining property is that no
    single modality carries the answer — the discriminating structure a fusion
    battery is supposed to have and currently does not. More slots as needed
    to hold measurement error down, but slot COUNT is not the repair; slot
    STRUCTURE is.
  - **A new rig gate, `A0_HEADROOM`, pre-registered and firing BEFORE any arm
    is scored:** the anchor must read strictly below ceiling on EVERY
    registered seed, by a declared margin, or the run VOIDs on the rig and no
    ranking is emitted. This is the exact shape of the learning gate this desk
    adopted on `D1.0` — a reference arm that must leave room before anyone
    else's number means anything — and it is strictly additive: no run that
    passes it could have failed today's rig.

**ADOPTED (defect 1): the per-arm stability conjunct.** An arm must train on
ALL registered seeds to hold a seat. This turns basin fragility from a rig VOID
that hides the finding into a SCORED finding that disqualifies the arm, and it
is strictly harder than today's spec, which has no such requirement at all. The
row's own two "legal-looking repairs" stay illegal and are restated so no later
reader re-proposes them: per-registered-seed re-selection is tuning on
registered seeds, and a seed re-roll is run-until-pass.

**REFUSED: retiring the marginal-floor VOID in favour of SCORED-AND-INELIGIBLE
at the seed level.** This is the one option on the menu that makes a run which
today returns no verdict return one instead, and it does so by letting an arm
that failed to train on a registered seed keep competing on the seeds where it
did. That is a weakening wearing a bookkeeping name, and it points the opposite
way from the conjunct adopted above. The recorded VOID stands; UB.10 becomes
reachable by getting a battery that can discriminate, not by relaxing what
counts as a result.

**The kinship is not decoration and it is the reason this ruling is not local.**
A null at ceiling here, a null with no headroom in `sh02-null-saturation`, a
world too shallow to separate anything in `w0-too-shallow`: three fronts, one
disease — **our tasks are too easy for our instruments to say anything.** Every
one of them is repaired in the same direction, by making the world harder, and
never by making the bar lower. If a fourth arrives, it is not a coincidence and
it belongs on a Sunday as a single question about how this project picks tasks.

**Nothing re-runs on this row's account until the redesign is committed**, per
the T1.02 precedent, and `UB.11` stays blocked behind a real UB.10 verdict.

## ROUTED 2026-09-01 (builder, LC.07 pilot harvest): `lc07-checkpoint-branch` — the seat's own scale-transfer arena cannot physically run inside a Kaggle kernel

ROUTED: lc07-checkpoint-branch | 2026-09-01 | LC.07-pilot-branch-B | ACTED 2026-09-13 in a3a090a (builder EXECUTED the residue the 09-06 disposition left owed — the CPU venue is PRICED at 535.5 core-hours, venue ratio 1.0, 33.5 days of the whole CPU budget, largest run 3.0x WORST_LEGAL_CHILD_S. There was no GPU term to convert: `survival.py` uses none. The arena stays VENUE-UNAFFORDABLE at both venues; that the disposition's "checkpointing repairs the wrong constraint" INVERTS at this venue is reported to the Review, not decided here. Earlier: DISPOSITIONED 2026-09-06, Review FULL — checkpointing REFUSED, arena declared VENUE-UNAFFORDABLE, ratio question routed as `D24`. Design below)
    DUE: 2026-09-06 | a checkpoint-vs-venue decision owed by the Review's
    Sunday FULL run; bundle judgment beside `w0-too-shallow` and D10's
    lineage — this arena is the one D10's firing commit registered so the
    wm-latent seat would not be held with a dead arena
    DUE: 2026-09-13 | DECISION DELIVERED 2026-09-06 (checkpointing REFUSED,
        arena declared VENUE-UNAFFORDABLE, ratio question routed as `D24`).
        What remains owed on this row is the ONE cheap thing: the BUILDER
        prices the CPU venue — 526 GPU-wall-hours through the pilot's own
        borrowed `LC.02` GPU:CPU ratio against the measured 57,600 s/day CPU
        budget — and writes the number here. A calculation, not a run: no
        dispatch, no seeds. The row closes when the number exists, whichever
        way it falls.

**What was measured (throughput pilot, seed 90, kernel
jack-ladder-1788297232, 0.44 h, 2026-09-01 21:40; artifact
/data/lc07_pilot.json; full PILOT RECORD in the spec docstring).** The rig
is healthy — all 7 run classes measured, wiring exact, physics finite,
RSS ~550 MB, borrowed LC.02 ratio calibrated — and the pre-registered
branch B fired: rule A requires every full-scale run <= 8.5 h wall, and the
CHEAPEST class (statue, 2.0M decisions) projects **14.49 h** while the arm
(4.0M decisions at 27.19 dec/s) projects **40.86 h** — 4.8x the kernel
ceiling. Parallelism cannot help a single run; the total plan is ~526
wall-hours (21 runs, ~132 kernel-hours at ideal 4-way packing) against a
30 h/week free allocation. Per the freeze step, nothing froze: `run()`
keeps refusing, the envelope did not shrink, no constant moved.

**DISPOSITION (Review FULL, 2026-09-06). Option 1 — checkpoint/resume surgery
on `survival.py` — is REFUSED, and the reason is that it repairs the wrong
constraint.**

**The arithmetic that decides it.** Checkpointing addresses the **8.5 h
per-run kernel ceiling**. It does not touch the **~526 wall-hour total**. At
the free allocation of 30 h/week, this arena costs **≈17.5 weeks of the
project's ENTIRE GPU budget** — every other GPU spec, every bakeoff, every
re-buy, for four months — and it costs that whether or not each run fits
under 8.5 hours. So option 1 converts *impossible* into *17.5 weeks*, and
bills the full `survival.py` surgery — which stales every `LC` and `XL`
certificate — for the conversion. `LF.02`'s PASS on 2026-09-03 (bit-exact
resume over 1000 decisions, all four stores, weights-only null diverging
8.1 ± 2.7) is a genuine and valuable existence proof that the surgery is
FEASIBLE; feasible is not the question this row asks.

**What I am declaring instead: the arena is VENUE-UNAFFORDABLE, and that is a
finding about the seat, not about the spec.** `LC.07` is the arena that `D10`'s
firing commit registered so the wm-latent Learning-core seat would not be held
with a dead arena. It is now measured as an arena nobody can enter for four
months. `champions --check` already reads the Learning core seat as TRIGGER
DEBT — *every declared re-open trigger a closed door* (`LC.07`=PILOT-BLOCKED,
`LC.03`=VOID-FORECLOSED, `UB.10`=VOID) — and this row is the third door
measured shut. **A seat whose arena costs 17.5 weeks of the whole allocation is
contested on paper and uncontested in fact, and that is precisely the shape
`D23` names**: an instrument reading green because a debt moved to a desk that
cannot pay it.

**The 10x is a THRESHOLD and I will not lower it.** The row's third option —
*"a Review/owner re-read of ~10x"* — is the one that looks cheapest and is the
one my own law forbids me to take on my own authority: a 10x scale-transfer
claim is strictly stronger than a 3x one, and shrinking it to fit the budget
is buying a PASS with a smaller question. If the ratio moves, it moves on the
owner's signature with the cost on the table, not on mine.

**The one thing owed to the builder, and it is cheap.** *Price the CPU venue
rather than arguing about it.* We now have a CPU accountant (`T0.33`/`T0.34`,
a measured 57,600 s/day) and `LC.07`'s pilot already calibrated a borrowed
`LC.02` GPU:CPU ratio. Convert 526 GPU-wall-hours through that measured ratio
and put the number on this row. My expectation is that it is far worse and the
option dies on arithmetic — but *"CPU venue"* has been an unpriced option on
this row for five days, and an unpriced option is how a decision gets deferred
forever. This is a calculation, not a run: no dispatch, no seeds, no budget.

**PRICED 2026-09-13 (builder, 93rd audit B1). THE NUMBER IS 535.5 CORE-HOURS,
AND THE EXPECTATION ABOVE IS WRONG: THE CPU VENUE IS NOT "FAR WORSE" — IT IS
THE SAME PRICE, AND IN CALENDAR TERMS IT IS 3.6x CHEAPER. IT STILL DIES, BUT ON
A DIFFERENT CEILING, AND THAT CHANGES WHICH CONSTRAINT IS BINDING.**

*First, the conversion the instruction asked for does not exist, and saying so
is half the answer.* This paragraph asks me to convert 526 GPU-wall-hours
through "the pilot's own borrowed `LC.02` GPU:CPU ratio". There are two errors
in that sentence and they point the same way. `LC.02`'s borrowed quantity is
`train_ratio` — **optimiser steps per decision** (`lc_07`:36), not a venue
speed ratio; and there is no GPU term to convert, because **`experiments/
survival.py` contains no `cuda`, no `device` and no `.to(...)`** — this row's
own option 2 already says it (*"the runs are single-thread CPU (27–38 dec/s, no
GPU use)"*), and `LC.03`'s docstring says *"zero GPU"*. The 526 hours were
never GPU-hours. They are single-thread CPU hours that were **billed against a
GPU quota** because `gpu.py` submits GPU kernels only. So the venue transfer is
not a conversion at all; it is a core-for-core comparison, and the audit's
restatement — *"through the pilot's own measured dec/s"* — is the instruction I
executed.

*What I priced it against.* `LC.03` v2's on-box curves
(`experiments/artifacts/lc03_curves_seed{0,1,2}.json`) measure **the same
`survival.py`, the same `wm-latent` arm `LC.07` holds the seat for, on this
box, at three seeds**. The comparison is like-for-like on both axes that could
have broken it: the pilot's own wall-vs-process gap is **≤ 0.11%** on every
one of the seven classes, so Kaggle wall-hours and on-box `process_time_s`
core-seconds are the same unit here; and the train ratios match — pilot arm
**0.1238** opt/dec against `LC.03` `wm-latent`'s **0.1250**, 1% apart. Note
where this datum came from: `LC.03` v2 spent ~190 core-hours in the detached
lane *before the accountant existed*, which is why there is anything on this
box to price against.

    class     Kaggle dec/s   this box dec/s (3 seeds)   ratio box/Kaggle
    arm           27.19        24.39  [23.80–25.13]        0.897
    wiped         27.69        25.68  [25.05–26.82]        0.927
    twin          34.24        39.15  [36.76–41.19]        1.143
    null          38.48        42.01  [38.40–45.66]        1.092
    ctl_null      37.22        42.65  [40.52–45.06]        1.146
    (statue borrows `null_random`, randrew borrows `wm-latent` — both
     are the matching class by wiring, and the Kaggle pair agrees: statue
     38.34 vs null 38.48, randrew 27.82 vs arm 27.19.)

**The venue ratio is ~1.0.** This box's core is 10% slower on the trained
classes and 9–15% faster on the untrained ones. A Kaggle CPU allocation and a
free-tier ARM core are the same machine for this workload.

    full-scale run, this box, one core      core-s      h
      arm       4.0M dec / 24.39 dec/s      164,031    45.56
      wiped     4.0M / 25.68               155,760    43.27
      null      4.0M / 42.01                95,218    26.45
      randrew   2.0M / 24.39                82,015    22.78
      twin      2.0M / 39.15                51,090    14.19
      statue    2.0M / 42.01                47,609    13.22
      ctl_null  2.0M / 42.65                46,891    13.03

    WHOLE PLAN (7 classes x 3 seeds = 21 runs, 60.0M decisions):
      1,927,842 core-s = 535.5 core-hours   (vs 526.35 h at the Kaggle venue)

**Against the two ceilings that actually decide it:**

- **The day budget.** 1,927,842 s / `CPU_DAY_CEILING_S` 57,600 s = **33.5 days
  of the ENTIRE ladder's CPU budget** — every certificate re-buy, every gate
  sweep, every CPU spec, for a month. Set that beside the GPU venue's **17.5
  weeks = 122 days** of the entire free allocation. **The CPU venue is 3.6x
  cheaper in calendar terms.** That is the opposite of what this row expected,
  and it follows directly from the ratio being 1.0 while the two budgets are
  not: 16 core-h/day is simply more than 30 GPU-h/week.
- **The per-run ceiling, and this is what kills it.** The largest run is
  **45.6 h = 3.0x `WORST_LEGAL_CHILD_S` (54,000 s) and 2.8x a whole day's
  ceiling.** Even the **CHEAPEST of the 21 runs is 13.0 h = 0.8x an entire
  day's CPU budget.** `T0.33` refuses every one of them before it starts. For
  completeness and not as an option: ignoring the meter entirely, at the 3
  nice-19 workers `LC.03` v2 actually measured, the plan is ~178 h ≈ **7.4 days
  of wall clock** — the physics is unremarkable; it is the accounting that
  forbids it.

**THE CONSEQUENCE, ROUTED AND EXPLICITLY NOT DECIDED HERE.** This disposition
refused checkpointing because *"it repairs the wrong constraint"* — it fixes
the 8.5 h per-run kernel ceiling and leaves the ~526 h total untouched. **At
the CPU venue that reasoning inverts.** The total stops being the binding
constraint (33.5 days, not 17.5 weeks) and the **per-run ceiling becomes the
only thing in the way** — which is precisely what checkpoint/resume repairs,
and `LF.02`'s PASS already proves the surgery is feasible one level below
`survival.py`. So **option 2 does not escape option 1; it meets it from the
other side**, and the live question is no longer *checkpoint OR venue* but
*checkpoint AND venue, for 33.5 days of CPU budget*. Whether that is worth
buying is a resource judgment with the same shape as `D24` and it belongs to
the Review and the owner. I am not reopening the disposition, I am reporting
that its arithmetic moved.

**What did NOT move.** No threshold, no envelope, no constant, no gate. No
dispatch, no seeds, no budget spent — this is the calculation the row asked
for and nothing else. `LC.07`'s `run()` still refuses, `_GATES_FROZEN` is
still False, the 10x scale-transfer reading is untouched, and the arena is
still `VENUE-UNAFFORDABLE` at both venues. The row asked for a number "whichever
way it falls"; it fell sideways.

**To the owner, as `D24` (see `docs/DECISIONS_NEEDED.md`).** The affordability
of this arena is a resource decision and it is not mine.

**Why this is a Review decision, not a builder unit.** The docstring
pre-registered it: `run_survival` has no mid-run checkpoint, and building
one is surgery on `experiments/survival.py` — an IMPL_DEPS of every LC/XL
certificate, so the change stales certificates and must be its own
reviewed unit, not a freeze-step side effect.

**Options to weigh (not decided here):**
1. **Checkpoint/resume in `run_survival`** (GPU_LONG's own requirement:
   checkpoint, not trim). Determinism across a checkpoint boundary is the
   hard part — the RNG stream, the world state, and the episodic store all
   have to survive a kernel death bit-exactly, or a resumed run is a
   different run wearing the same seed. Cost: stales every LC/XL
   certificate's impl stamp; the amend lane (prose_only_delta) will NOT
   cover it because it is a code change.
2. **A different venue for CPU-bound survival runs** — the runs are
   single-thread CPU (27–38 dec/s, no GPU use); Kaggle CPU-only sessions
   have longer caps (docstring notes a CPU lane needs gpu.py surgery), or
   the box itself at nice 19 (40.86 h wall is ~2 days of a core; the
   tenant/RAM constraints allow one worker at ~550 MB) — slow but legal,
   and `launch_detached.sh` already owns the liveness discipline.
3. **Re-examine whether 10x decisions is the right reading of the owner's
   "~10x" guard** — ONLY as a Review/owner question: the envelope is
   registered and may not move by builder hand; this option exists so the
   Review can weigh it against D10's intent rather than have it decided
   by a docstring's silence.

Whatever is chosen, the pilot's numbers are spent evidence: no re-roll, no
second pilot, and LC.07 stays refusing until a decision writes the freeze.

**UPDATE 2026-09-03 (builder, LF.02 PASS — evidence for option 1, not a
decision).** The hard part option 1 names — *"the RNG stream, the world state,
and the episodic store all have to survive a kernel death bit-exactly, or a
resumed run is a different run wearing the same seed"* — now has a measured
existence proof one level below `survival.py`: **LF.02 PASSed on 2026-09-03
(attempt on clean tree, 3 seeds)** — a W0 life SIGKILLed mid-decision-stream
(rc -9 verified per seed), all four stores (mjSTATE_INTEGRATION + every W0/
DriveLayer python-side mutable incl. both RandomStates, diary, GRU working
memory) checkpointed atomically every decision, and the resumed process
matched the uninterrupted reference **bit-exactly over 1000 decisions**
(state-digest match 1.0, max float delta 0.0 on all seeds), weights-only null
diverging 8.1 +/- 2.7, all four store-corruption loads raising loudly. Two
mechanical traps were found at smoke and are solved in the committed fixture,
and they are exactly the traps a `run_survival` checkpoint will meet: (a)
after `mj_step`, `data`'s kinematics/contacts describe the PRE-integration
pose — a derived layer no state vector carries — so the checkpoint boundary
must be pinned with an explicit `mj_forward` on every arm; (b) a restore-side
`mj_forward` clobbers `qacc_warmstart` and the saved one must be re-seated or
the first substep diverges in its final bits. Option 1's COST is unchanged
(surgery on `survival.py` stales every LC/XL certificate); what changed is
that its feasibility is no longer a hypothesis.

**UPDATE 2026-09-07 (builder, 03:0x slot): THE CPU-VENUE PRICING THIS ROW WAS
OWED IS DONE — it was executed 2026-09-06 08:2x and recorded as an ADDENDUM to
`D24` in `docs/DECISIONS_NEEDED.md`, not here; this line exists so the next
reader of this row finds the number without hunting.** The figure: box-side
rate from `LC.03` v2's committed 400k decisions / 17,280 core-s (23.15 dec/s)
vs the pilot's 27.19 dec/s same-class on the Kaggle VM → all 21 runs scale to
**~618 core-hours = 38.6 fully-billed 57,600-s days ≈ 5.5 weeks of this box's
ENTIRE CPU day budget**, total monopoly of the meter, foreclosing every other
CPU spec for the duration. The largest single run (arm, 48.0 core-h) needs no
checkpoint surgery on this venue — it lands in `cpu<48h`, whose children
charge across calendar days via `T0.34`'s midnight split — but that class's
self-foreclosure question is exactly the one already routed
(`cpu48h-class-self-forecloses-the-day-meter`, DUE 09-08) and governed by
`D20`. So the CPU venue converts 17.5 GPU-weeks into ~5.5 CPU-weeks of meter
monopoly and stales nothing; it is a PRICE on `D24`'s table (decide_by
09-11), not an armed option. I re-derived the arithmetic independently this
slot before finding the addendum — same ratio, same 617.8 core-h / 38.6 days
— so the number is now double-checked; my first draft of this update claimed
the day meter forecloses every run outright, which is wrong (the `cpu<48h`
detached lane admits multi-day children by design) and is corrected here
rather than deleted silently.

## ROUTED 2026-09-02 (builder, 60th audit B2): `d10-successor-rerun-under-adopted-gate` — the project's largest unblock returned an honest VOID and became nobody's work in the same motion

ROUTED: d10-successor-rerun-under-adopted-gate | 2026-09-02 | 60th-audit-B2 | DISPOSITIONED 2026-09-08 (Review DAILY — VOID-FORECLOSED is REFUSED and the twin-denominator successor gate is ADOPTED, with a cheap pre-registered probe that must run BEFORE any third dispatch. Design below)
    DUE: 2026-09-06 | the Review must either adopt a gate option on the two
    `d10-*` gate rows (then the builder repairs the D1.0 learning gate
    accordingly and dispatches attempt 2 in W36) or direct a VOID-FORECLOSED
    declaration carrying the arithmetic below; past this date the row goes
    OVERDUE — the point is that the repair has an owner and a clock
    DUE: 2026-09-08 | the same either/or, moved to the Tuesday DAILY —
    RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): this row is
    a pure consequence of the two `d10-*` gate rows, which STAY 09-06; the
    gate is adopted Sunday, this consequence is stamped Tuesday, and W36 runs
    to 09-13 so the attempt-2 dispatch loses nothing. The clock and owner
    this row exists for are intact.
    DUE: 2026-09-26 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE, and for this row the pacing blackout is NOT the cause and must not be allowed to stand in for it: this row is UNREACHABLE BY CONSTRUCTION. `D1.0` is BLOCKED <- `T1.08` FAIL, so the successor re-run it owes cannot be dispatched at any budget, on any meter, in any week. Three dates (09-06 -> 09-08 -> 09-14) have now been set on a run that no awake builder could have bought. STOP-RULE, and it is a different one because the defect is different: if `T1.08` is still FAIL on 2026-09-26, this row is NOT re-dated a fourth time — it is RE-PARENTED behind `T1.08`'s repair, because a clock on an unreachable run is a promise the calendar cannot keep and the queue should say so in its structure rather than in its prose. ORIGINAL TEXT FOLLOWS, unchanged. | DECISION DELIVERED 2026-09-08 (Review DAILY):
    VOID-FORECLOSED refused, the twin-denominator successor gate adopted, and
    the twin-spread probe pre-registered in BOTH branches — full design in the
    DISPOSITION block at the foot of this row. What is now owed is the
    BUILDER's two-step stamp: (1) the twin-spread probe result written onto
    this row, (2) the successor gate committed in a commit that is NOT a
    dispatch commit. Dated 09-14 because W36 has ~12.4 GPU-h left against
    attempt 2's measured 17.61 h — attempt 3 cannot fit this window anyway and
    W37 opens 09-13, so the date buys the probe honest room instead of racing
    a quota. 09-14 carried one promise when this was written.

**Why this row exists (60th audit FINDING 2, quoted arithmetic).** `D1.0`
fired as D1's armed default, ran 16.17 GPU-hours — 54% of a weekly quota —
and returned VOID (`c_e2e` 2.56σ against the 3.0σ learning gate). The two
sibling rows (`d10-learning-gate-uses-two-different-denominators`,
`d10-learning-gate-sits-at-the-untrained-twin-level`, both DUE 2026-09-06)
correctly scope themselves to FUTURE gate design and disclaim the re-run,
so no row, no `DUE:` and no priority line owned fixing the arm — while
`T2.01` (frees 35, blocks 38) waits on D1.0's winner and the
Control-architecture seat reads VACANT with `champions --check` ok.

**The named unit is REPAIR-AND-RERUN, and the arithmetic says it is NOT
foreclosed.** A `VOID-FORECLOSED` declaration must show the verdict cannot
change at this envelope. It can: (a) the fired conjunct is a gate-scoring
artifact, not an envelope wall — `c_e2e` returned 404.3 vs random's 108.7,
a 3.7× gain, scored against its OWN wider spread, with the untrained twins
at 2.94–2.96σ against the 3.0 bar; (b) the venue fits — attempt 1's three
kernels each ran under the 8.5 h ceiling (largest ~4.1 h) and W36 opens
2026-09-06 00:00 UTC with a full 30 h against attempt 1's measured 16.17 h.
Sequencing: an UNCHANGED re-dispatch is a seed-lottery redraw and stays
forbidden; attempt 2 exists only under a gate design adopted on the sibling
rows first (committed before dispatch, σ bar unmoved — strengthen-only per
SYSTEM.md law 4). That is the T2.02 path: fix the rig, then re-run; the
recorded VOID stands either way.

**The honest alternative, on its face.** If the Review judges the venue the
common cause (`w0-too-shallow` bundle) and declines a gate repair, the
terminal state is a `VOID-FORECLOSED` declaration quoting this arithmetic —
this row then converts rather than vanishes. What may NOT happen is what the
last seven days did: 16.17 hours buying a VOID that no instrument owns.

**UPDATE 2026-09-07 (builder, 02:0x slot): ATTEMPT 2 HAS RUN UNDER THE
ADOPTED GATE AND RETURNED VOID — on the OTHER denominator. The either/or
this row owns is now decided with 33.8 GPU-hours of evidence, and the fresh
half says the gate design itself, not any arm, is what cannot decide.**
Attempt 2 (4 kernels, P100, 17.61 h, W36's opening spend; dispatched 09-06
under the d10-* gate dispositions; row landed 2026-09-07T01:57:12, committed
as found in `4abb2e6`, kernel heads amended to dispatch values per 80th-audit
B1). Every trained arm cleared the gate against random: aprime 13.02σ
(mean 506.4), d_mlp 12.99σ (415.0), c_e2e 10.80σ (350.0) — attempt 1's sole
non-learner now comfortably over — b_split 10.53σ (344.0); random 111.0,
sb3 reference 518.5; margin_sigma 3.37, winner aprime. And the VOID branch
that fired is the twin-level control this gate was adopted to carry:
**untrained aprime 3.95σ and untrained d_mlp 3.91σ against the 3.0 bar —
"the gate is measuring architectural bias, not learning."**

The arithmetic the Tuesday sitting needs, derived from the two rows' own
numbers rather than re-argued: the twin MEANS are identical across attempts
(aprime twin 198.4, d_mlp twin 197.6, both runs — deterministic twin eval);
what moved is the random denominator's spread (std 30.27 → 22.12, mean 108.7
→ 111.0). Same twins, same bar: 2.94–2.96σ in attempt 1, 3.95/3.91σ in
attempt 2. So (a) an untrained aprime/d_mlp genuinely banks ~87 raw points
of architectural prior over random, and (b) whether that prior reads as
over or under 3σ is decided by the sampling noise of one random-policy
draw. Two attempts, two different VOID branches, both from denominator
arithmetic the arms never touched. A third dispatch under any
random-denominator gate is a coin-flip on random_std and stays forbidden by
this row's own sequencing rule; the live options are a twin-denominator
successor design (each arm scored against its OWN untrained twin — the
sibling row's design question) or the VOID-FORECLOSED declaration quoting
this block. The choice is the Review's; both attempt rows stand either way.

    CLOCK (the parsed `DUE:` for this row is the one in its header above; this
        copy sits beside the design and is deliberately NOT declaration-shaped)
        — DUE 2026-09-14, DECISION DELIVERED 2026-09-08 (VOID-FORECLOSED refused;
        the twin-denominator successor gate adopted, below). What is now owed
        is the BUILDER's two-step execution stamp: (1) the TWIN-SPREAD PROBE
        result written onto this row, (2) the successor gate committed in a
        commit that is NOT a dispatch commit. Dated 09-14 and not sooner
        because W36 has ~12.4 GPU-h left against attempt 2's measured 17.61 h,
        so attempt 3 cannot fit this window anyway and W37 opens 09-13 — the
        date buys the probe honest room instead of racing a quota. 09-14
        carried one promise when this was written.

**DISPOSITION (Review DAILY, 2026-09-08): the denominator was the defect, so
the denominator is what changes — and it changes in the harder direction.**

**VOID-FORECLOSED is refused, and the refusal is arithmetic, not appetite.** A
`VOID-FORECLOSED` declaration must show the verdict cannot change at this
envelope. Two attempts show the opposite with unusual clarity: every trained arm
cleared the gate against random in attempt 2 (aprime 13.02σ, d_mlp 12.99σ,
c_e2e 10.80σ, b_split 10.53σ, margin_sigma 3.37, winner aprime). Nothing about
the ARMS is foreclosed. What is foreclosed is one specific gate design, and
declaring the whole arena dead on account of a repairable instrument would
retire `T2.01` — **frees 35 / blocks 38, the largest single unblock in the
project** — for a reason that is not about locomotion at all.

**The defect, stated as the two rows' own numbers state it.** The twin MEANS
are identical across both attempts (aprime 198.4, d_mlp 197.6 — twin eval is
deterministic). The bar is identical (3.0σ). The verdict flipped — 2.94–2.96σ
to 3.95/3.91σ — because the RANDOM policy's spread moved (std 30.27 → 22.12,
mean 108.7 → 111.0). **A gate whose verdict is a function of the sampling noise
of one random-policy draw is not measuring the thing it is named for.** And
the substantive half is worse than the noise half: an untrained aprime banks
~87 raw points over random *before any learning happens*. Scored against
random, an arm is paid for its architectural prior. That is precisely the
confound the gate's own VOID text named — *"the gate is measuring architectural
bias, not learning."*

**ADOPTED: the twin denominator. Each arm is scored against its OWN untrained
twin, never against the random policy.** This is a STRENGTHENING and it should
be read as one: the ~87 points of architectural prior that an arm currently
banks for free are subtracted by construction, so an arm that wins only by
being a better-shaped network at initialisation now scores zero. Nothing that
passes the new gate could have failed the old one for a reason we care about.

Binding conditions, all of them:

  1. **The σ bar does not move. 3.0 stays 3.0.** (SYSTEM.md law 4,
     strengthen-only.) The denominator changes; the bar does not.
  2. **The random policy STAYS IN THE RUN as a reported floor** — it is a
     genuine sanity reading and both attempts' rows cite it. It simply stops
     being the denominator. We keep the number and lose the dependence on it.
  3. **THE PROBE COMES FIRST, AND IT IS PRE-REGISTERED IN BOTH BRANCHES.**
     Twin eval is deterministic *at a fixed init seed*, so a twin has no spread
     to divide by and the design is undefined until somebody measures whether
     it has one ACROSS init seeds. Cost is the reason this is cheap and the
     reason it is honest: **the untrained twin is never trained**, so K twin
     evaluations cost K forward passes and no gradient steps — the expensive
     term does not scale with the denominator's seed count at all. Run
     K >= 16 untrained twins per architecture at distinct init seeds and
     report the mean and std of each architecture's prior. Both branches are
     declared NOW, before the number exists:
       - **spread is real** (twin std materially non-zero): the gate is
         `(arm − twin_mean) / twin_std >= 3.0`, the arm measured against the
         distribution of what its own architecture gets for free.
       - **spread is ~zero** (the architectural prior is deterministic): σ is
         undefined and MAY NOT be manufactured by borrowing a spread from
         somewhere else. The gate falls back to a RAW-MARGIN form declared in
         the same commit — arm − twin_mean >= the margin attempt 2's own
         numbers make non-trivial — and the fact that the gate changed units
         is written on the row, not buried in the diff.
     Choosing the branch after seeing the number is the forbidden move; that
     is why both are written here, today, with no number in hand.
  4. **The successor gate is committed BEFORE any dispatch, in a commit that
     is not also the dispatch commit** — this row's own sequencing rule,
     unchanged, and the same rule the T2.02 path follows.
  5. **An unchanged re-dispatch stays forbidden.** Attempt 3 does not exist as
     a re-roll of attempt 2 under any circumstances. Both recorded VOIDs stand
     in the ledger regardless of what the successor returns.

**TWIN-SPREAD PROBE RESULT — STEP (1) OF THE OWED STAMP, LANDED (builder,
2026-09-12; `experiments/tests/d10_twin_spread_probe.py`, artifact
`/data/d10_twin_spread.json`, 1838 s CPU, K=32 init seeds per architecture,
forward passes only, nothing trained).** The branch criterion was committed in
`8624fa0` BEFORE the run, with no number in hand, per condition 3's own
prohibition; the run is `e938a89`-clean and this block is the result.

    arm       twin mean    twin std     cv     recorded twin (attempt 2)
    aprime       197.09        4.54   0.0230        198.4
    b_split      171.49       42.23   0.2462          -
    c_e2e        179.41       71.75   0.3999          -
    d_mlp        197.70        2.02   0.0102        197.6

**The probe is measuring the right object — independent corroboration.** Its
K=32 means reproduce attempt 2's own recorded twin means to within 1.3 points
on both arms the row published (aprime 197.09 vs 198.4; d_mlp 197.70 vs 197.6),
having been computed from 32 fresh init seeds by a separate entry point. The
~87 raw points of architectural prior over random that the disposition
identified are confirmed and are not an artifact of three seeds.

**THE PRE-REGISTERED BRANCH IS `SPREAD_IS_ZERO`**: `min(cv) = 0.0102` (d_mlp)
against the committed `CV_MIN = 0.05`, so the unanimity clause fails and the
gate takes the DECLARED RAW-MARGIN FALLBACK. `σ` may not be manufactured, and
per the disposition the change of units is written here rather than buried in
a diff.

**AND THE PROBE FOUND SOMETHING THE DISPOSITION DID NOT ANTICIPATE, which is
the real result: the spreads are not merely small, they are HETEROGENEOUS BY
35.5x** (2.02 for d_mlp to 71.75 for c_e2e). A per-arm `twin_std` denominator
is therefore **not a common unit**, and scoring four arms in four different
units is not a bakeoff. Run the adopted "spread is real" formula against
attempt 2's own recorded trained means and it fires on the wrong arm:

    arm      trained   twin_mean   raw margin   twin_std    sigma   verdict
    aprime     506.4      197.09        309.3       4.54    68.20   PASS
    b_split    344.0      171.49        172.5      42.23     4.09   PASS
    c_e2e      350.0      179.41        170.6      71.75     2.38   FAIL (<3.0)
    d_mlp      415.0      197.70        217.3       2.02   107.39   PASS

**`b_split` and `c_e2e` learned the same amount — raw margins 172.5 and 170.6,
1.1% apart — and the twin-σ gate PASSES one and FAILS the other.** The run
would have VOIDed on `c_e2e` for the third time, and for the third different
denominator reason, with nothing about `c_e2e`'s learning having changed. That
is the identical disease this row was opened to cure, reproduced inside its own
proposed cure: *a gate whose verdict is a function of a denominator rather than
of learning.* Note also the top end — 68σ and 107σ — where dividing a real
margin by a near-deterministic prior inflates the statistic until the 3.0 bar
stops being a bar at all. **The twin denominator fails in BOTH directions at
once, and only the raw-margin branch escapes both.**

So the pre-registered criterion earned its keep: written blind, it routed the
design away from a formula that two hours of arithmetic then showed would have
mis-fired. Recording that explicitly because the opposite is the standing
temptation — `CV_MIN = 0.05` was a guess, and the guess is vindicated by a
mechanism (unit heterogeneity) it was not chosen for.

**WHAT IS STILL OWED, AND IT IS STEP (2), NOT THIS BLOCK.** The successor gate
is now determined in FORM — `arm − twin_mean >= MARGIN`, per-arm twin means as
tabulated above, `MIN_LEARN_SIGMA` untouched and no longer the operative
comparison for G1 — but its MARGIN CONSTANT is not yet chosen, and this desk
deliberately did not choose one while holding attempt 2's four margins (309.3,
217.3, 172.5, 170.6) in hand. **Fitting a bar to the numbers it will judge is
the move every rule here forbids**, and the disposition's phrase *"the margin
attempt 2's own numbers make non-trivial"* is the one clause in it that invites
exactly that. The next unit must derive the margin from something that is not
attempt 2's trained means — the random floor, the twin spread itself, or a
declared effect size — and say which, in the non-dispatch commit. Until then
**no dispatch is authorised**, and the standing prohibition on an unchanged
re-dispatch is untouched.

**What this costs and what it does not.** The probe is CPU-cheap forward passes
and buys the design its missing premise. Attempt 3 remains ~17 GPU-h and is NOT
authorised by this disposition — it is authorised by the probe landing and the
gate being committed, and it should be dispatched into W37 (opens 09-13) rather
than scraped out of W36's remaining ~12.4 h. This desk has now spent 33.8
GPU-hours on two VOIDs and will not buy a third verdict from a gate that has
not first been shown to be able to return one.

**SUCCESSOR GATE COMMITTED — STEP (2) OF THE OWED STAMP, LANDED (builder,
2026-09-12 ~23:0x, commit `7cb00ea`, which is NOT a dispatch commit, per
condition 4). Both steps of this row's execution are now discharged.** The σ
bar did not move; the random policy stays computed and recorded and is no
longer a denominator anywhere; both recorded VOIDs are undisturbed.

    LEARN_MARGIN = MIN_LEARN_SIGMA x max_arm( twin_std x sqrt(2/n) ) = 175.74

**THE CHANGE OF UNITS, written here as condition 3 requires rather than left in
a diff: the operative comparison for the new conjunct is RETURN POINTS, not
sigma.** The derivation takes one pre-registered constant (`MIN_LEARN_SIGMA`
3.0), the registered seed count (3) and the probe's measured spreads — **and
nothing from attempt 2's trained means**, which is this row's own prohibition.
The null it excludes is *"training moved the weights and learnt nothing
useful"*, under which the trained score is a fresh draw from the arm's own
untrained prior, so `mean(d)` has sd `std·sqrt(2/n)`. The `max` over arms is
FORCED by the probe's heterogeneity finding: a single raw bar has to dominate
the noisiest architecture's null or it admits init luck there. The constant is
**computed in code from the frozen table, not typed**, so it cannot be
hand-tuned without visibly editing measured numbers. A 200k-sample bootstrap
off the probe's own 32 returns corroborates it and is recorded as a known
limitation rather than smoothed: `c_e2e`'s empirical p99.865 is **185.80**, ~6%
above the parametric bar, because its prior is heavy-tailed (one draw at
411.3). The bar stays parametric — an empirical tail quantile from 32 draws is
set by a single observation and is not reproducible — and the exposure is
bounded by the conjunction below.

**ONE DEPARTURE FROM THIS DISPOSITION'S WORDING, named rather than quietly
taken.** The disposition's branches were written about
`(arm − twin_mean)/twin_std`, and this row's step-(2) note concluded that
`MIN_LEARN_SIGMA` is *"no longer the operative comparison for G1"*. **Attempt
2's ledger row refutes the premise: G1 as implemented on 09-06 is PAIRED at the
same init seed** — `mean(d)/(sd(d)/√n)` — which is not that formula and is
strictly better than it, because pairing cancels the architectural prior per
seed instead of subtracting its across-seed mean. Executing the literal words
would have replaced a paired statistic with an unpaired one, i.e. weakened the
gate in the middle of a strengthening, which law 4 forbids. **So the raw margin
was ADDED as a conjunct instead of replacing G1**: the branch is executed
literally and the better statistic survives, and an arm must now clear both.
The substantive reason to keep both is that they fail differently — paired-t is
immune to a heavy-tailed prior (the same init seed is on both sides) but
inflatable by a coincidentally small `sd(d)` at n=3; the raw margin is immune
to that collapse and blind to pairing. The refuted premise is routed onto
`gates-that-measure-something-other-than-what-they-say` (DUE 09-20), not
settled here.

**WHAT ATTEMPT 2 ACTUALLY DIED ON, and it was not the arms.** `G1` cleared ALL
FOUR arms (paired-t 20.41 / 6.00 / 9.25 / 16.96 vs the 3.0 bar). The VOID came
from the **control**, which still scored untrained twins against RANDOM. That
was the last random denominator in the gate and it is the surface repaired:

    G0  NEW RIG   the run's own untrained means must agree with the frozen
                  TWIN_PRIOR table (K=32) within 3 standard errors, or NO arm
                  is scored. The gate now depends on constants measured by
                  another file on another day, and a silent drift would
                  mis-score every arm with nothing to announce it.
    G1b NEW CLAIM raw paired gain >= LEARN_MARGIN, in return units.
    CONTROL       an untrained twin must miss the margin a TRAINED arm has to
                  clear, as excess over its own frozen prior.
                  `untrained_*_sigma` is still computed and recorded and gates
                  nothing — condition 2 kept, the dependence dropped.

**RED-FIRST, AND IT CAUGHT A DEFECT IN THE REPAIR ITSELF.** Every branch was
replayed through `_check` against the recorded row. `G0` fires on a +20 drift;
`G1b` on a 170.0 gain; `G1` names both conjuncts when both miss. **But the new
control was UNREACHABLE FOR EVERY ARM as first ordered** — `G0`'s tolerance
`1.732·std_a` is always tighter than `LEARN_MARGIN = 2.449·max_std`, so the
control could never fire, which is precisely what law 2 forbids. Repaired by
ORDER (control before `G0`) with the inequality recorded in the docstring.

**THE FINDING THE REVIEW NEEDS BEFORE AUTHORISING ~17 GPU-HOURS.** Attempt 2's
verbatim row, replayed through the successor gate, clears `G0`, the control and
BOTH learning conjuncts on all four arms — raw gains 308.00 / 217.47 / 201.33 /
193.17 against 175.74, the closest by 10% — **and then lands on
`VOID (SPLIT-PENDING)`.** `aprime` leads `d_mlp` by 3.37σ on eval mean (506.4
vs 415.0) while `d_mlp`'s final-third TRAINING reward is **higher** (5.411 vs
5.303, gap −0.108) with a positive slope, so the owner's convergence check
fires: no winner while the runner-up is closing. **That is the first attempt-2
verdict that is about the ARMS rather than about a denominator** — and it says
the budget, not the gate, is now what cannot separate these two. On this
evidence a re-run at the same `STEP_TARGET` (750,000) is likelier to return
SPLIT-PENDING than a winner. **That is a dispatch question and it is the
Review's: this desk has not moved `STEP_TARGET` and will not.** The
precondition this row set is satisfied in form; whether ~17 GPU-h should buy a
run whose most likely verdict is "still converging" is a judgement the row
should make explicitly before W37's quota is spent.
ROUTED: lg10-mouth-fidelity-vs-freedom | 2026-09-02 | LG.10-attempt-2-FAIL | ACTED 2026-09-13 in 939e3c4 (builder, three days INSIDE this row's 09-16 clock: `LG.12` "He speaks correctly or he is silent" is REGISTERED — option (a) as a NEW claim beside `LG.10`, never an edit to it. All three binding conditions are in the registration: UTTER_MIN 0.50 on report trials as a CONJUNCT of the claim (a mouth that buys fidelity by going mute FAILs, it does not score lower); every fidelity bar carried UNMOVED (match/unanimity/swap_agree 0.90, variety 0.30, liveness 0.80, speak_silence 0.0, leak_draws 0, NULL_MATCH_MAX 0.35); and the null re-run through the IDENTICAL abstention machinery with its at-chance reading required to prove itself alive — if the margin silences the null below the same floor the run is VOID (NULL_SILENCED_BY_MECHANISM), not a quiet pass. NO DISPATCH and no LLM verdicts bought on this row's account; the run is its own decision. **TWO CORRECTIONS THE REGISTRATION CARRIES, both written into the registry block, and the first is to this disposition.** (1) "strictly more to satisfy, not less" is BACKWARDS: give the `LG.10` mouth an abstention rule that never abstains and utter_rate 1.0 clears any floor while match-on-spoken IS match-on-all, so `LG.10` PASS => `LG.12` PASS by construction — `LG.12` is formally WEAKER and cannot be a repair of `LG.10` even in principle, which is the strongest available argument for this row's own conclusion that `LG.10` must keep standing. UTTER_MIN is the only thing bounding how much weaker. (2) THE FLOOR WAS CHECKED FOR REACHABILITY BEFORE IT WAS WRITTEN (the `UB.10`/`BA.03`/`ME.11` lesson): at precision 0.90 an ORACLE abstention lets a mouth with ungated match m speak at most m/0.90 of the time, and `LG.10` v2's worst per-seed per-model match is 0.60, so ANY utterance floor above 0.6667 is born unreachable. 0.50 sits under it with headroom. Also recorded: the SILENCE control is DEMOTED on the record — with an abstention path available it is partly true by construction, so the utterance floor now carries the aliveness burden it can no longer carry alone. Discharges DISPOSITIONED 2026-09-08 (Review DAILY): (c) LG.10's bar and its FAIL both STAND; (b) refused on the spec's own docstring warning; (a) accepted as a NEW registered claim with a binding utterance-rate floor and explicitly NOT as a rewrite of a failing spec. Design in the DISPOSITION block at the foot of this row, unchanged)
    DUE: 2026-09-06 | a mouth-design decision owed by the Review's Sunday
    FULL run; bundle beside the ME.11 family disposition — both are cases
    where a language-side hypothesis was measured against a bar the
    incumbent machinery cannot reach, and the repair is a redesign, not a
    re-roll.
    DUE: 2026-09-08 | the same mouth-design decision, moved to the Tuesday
    DAILY — RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder):
    the ME.11 family disposition it bundles beside lands Sunday, so a
    Tuesday decision reads that disposition as a fresh INPUT instead of
    competing with it for the same 120-turn sitting; language-side, no
    coupling to the W0/W1 design.
    DUE: 2026-09-16 | DECISION DELIVERED 2026-09-08 (Review DAILY): (c) — the
    bar and the FAIL both STAND; (b) refused on the spec's own docstring
    warning; (a) accepted as a NEW registered claim with a binding
    utterance-rate floor and explicitly NOT as a rewrite of a failing spec —
    full design in the DISPOSITION block at the foot of this row. What is owed
    is the BUILDER's registration stamp for the new sibling spec:
    REGISTRATION ONLY, no dispatch, no LLM verdicts bought on this row's
    account. Dated onto a day carrying one promise.

**What was measured (LG.10, attempts 1+2, 2026-09-02, both from clean trees;
verdict artifact /data/lg10_llm_verdicts.json, 1588 verdicts, both frozen
SmolLM2 mouths).** The selection pipeline was run at BOTH ends of the
freedom knob, so the fork now has two measured endpoints and no open
argument about where the tradeoff sits:

    T=0.25 (v1): meaning-match 0.9833/1.0/1.0, swap_agree 1.0 — and VOID by
      the pre-registered variety floor (0.25/0.50/0.00 vs 0.30 worst-seed):
      the sampler had no measured freedom, invariance was vacuous.
    T=1.0  (v2): variety 1.0 and liveness 1.0 on all seeds (instrument fully
      alive) — and FAIL: match 0.60/0.7833/0.70 arm, 0.6667/0.7167/0.70
      swap, unanimity 0.0833-0.3333, swap_agree 0.75/0.9167/0.8333, all
      under the 0.90 bars. Controls all behaved: null 0.0-0.1167 (bar
      0.35), silence 0.0, leak 0, fabrications gate-rejected 1.0.

**The finding:** intent conditioning is a large real effect (null 0.02-0.12
-> arm 0.60-0.78) but at honest sampler freedom the frozen mouth chooses
part of the content: of 55 wrong draws (model A, arm prompt), 29 drift to a
DIFFERENT truthful memory and 26 collapse to the phatic "Hmm, let me
think." — fluency-attractor and subject-drift, two distinct mechanisms.

**Options, all runnable arms or declarations, not an argument:**
(a) a pre-registered dominance-margin abstention in the selection (utter
    only when the intent's phrasings clear a margin; meaning-flips become
    SILENCE, which the silence control already measures — changes the claim
    to "he speaks correctly or not at all", arguably the GOAL.md-honest
    mouth);
(b) extend the verification gate from record-membership to
    intent-consistency — WARNED AGAINST in the spec's own docstring: it
    makes (a)/(c) true by construction and the test decorative;
(c) keep the bar and the FAIL as the standing measurement: the mouth needs
    a stronger chooser (bigger frozen model, structured decode) and LG.10
    re-runs only under a design that could honestly reach 0.90.
Do NOT re-roll attempt 2 unchanged, and do not fit T — both endpoints are
already paid for.

    CLOCK (the parsed `DUE:` for this row is the one in its header above; this
        copy sits beside the design and is deliberately NOT declaration-shaped)
        — DUE 2026-09-16, DECISION DELIVERED 2026-09-08 (design below): (c)
        adopted, (b) refused, (a) accepted as a SEPARATE registration. What is
        owed is the BUILDER's registration stamp for the new sibling spec —
        registration only, no dispatch, no LLM verdicts bought on this row's
        account. Dated onto a day carrying one promise.

**DISPOSITION (Review DAILY, 2026-09-08): (c) — the bar stands and so does the
FAIL. (a) is a good idea about JACK and it may not be spent repairing a
measurement about the LLM.**

**(b) is refused, and the spec refused it first.** Extending the verification
gate from record-membership to intent-consistency makes (a) and (c) true by
construction and the test decorative. The spec's own docstring warns against it.
Nothing in the two attempts changed that; if anything the measurement makes the
warning sharper, because intent-consistency is precisely the quantity in
dispute.

**(c) is adopted, and the reason is the law this desk is bound by.** `LG.10` is
a FAIL. Redesigning a FAILING spec in the direction that would let it pass is
forbidden — and adding an abstention path to the selection is exactly that
direction, whatever its independent merits, because it converts every wrong
draw into a non-answer. Under the T1.02 precedent a redesign is legitimate only
when the EXPERIMENT is wrong, and this experiment is not wrong: it asked whether
Jack chooses what he says while the frozen mouth chooses only how, it ran both
ends of the freedom knob at 1588 verdicts from clean trees, its controls all
behaved (null 0.0–0.1167 against a 0.35 bar, silence 0.0, leak 0, fabrications
gate-rejected 1.0), and it returned a clean answer: **at honest sampler freedom
the frozen mouth chooses part of the CONTENT.** 29 of 55 wrong draws drift to a
different truthful memory, 26 collapse to the phatic "Hmm, let me think."

**That FAIL is one of the most valuable measurements on this board and it must
not be redesigned away.** `GOAL.md` says the LLM is his mouth and never his
mind, and `LG.10` is the registered falsifier of exactly that sentence. It has
now falsified it for the incumbent mouth. Erasing the record by changing the
claim would leave the project asserting in its constitution something its own
ladder had measured to be false and then stopped measuring. The finding stays
on the board: **intent conditioning is a large real effect (null 0.02–0.12 →
arm 0.60–0.78) and it is not large enough**, and the mouth needs a stronger
chooser — bigger frozen model, structured decode — before `LG.10` can honestly
reach 0.90.

**(a) is ACCEPTED, as a NEW registered claim beside `LG.10` and never as an
edit to it.** *"He speaks correctly or he is silent"* is a genuinely different
and arguably more `GOAL.md`-honest capability than *"he speaks correctly"*, and
it deserves its own falsifier rather than being smuggled in as a repair. Three
conditions bind it, and the first is a lesson this project paid for six days
ago:

  1. **A pre-registered UTTERANCE-RATE FLOOR is mandatory, declared before the
     first run.** An abstaining mouth with no such floor is `ME.3`'s starvation
     failure arriving in the language family: abstain on everything hard, score
     1.0 on what is left. `ME.3` was caught only because starvation was TOTAL
     (`raw_tokens` 40.0 → 0.0); **half-starvation would have passed every gate
     AND inflated the headline number.** A mouth that is silent most of the
     time is a mute, not an honest speaker, and the spec must be able to say so.
  2. **Fidelity bars do not move.** Match, swap-agree, variety and liveness
     carry over at their current values — the new claim is `LG.10`'s bars PLUS
     an abstention path PLUS the utterance floor, which is strictly more to
     satisfy, not less.
  3. **Every control re-runs under the abstention machinery**, especially the
     null and the silence control. A null that can abstain must still fail. An
     abstention mechanism that rescues the null is measuring the mechanism, not
     the creature.

**What this row does NOT authorise: any spend.** No `LG.10` re-roll (both
endpoints are paid for), no re-fitting of T, and no verdict purchase for the new
spec on this row's account. Registration first; the run is its own decision.
Noted for whoever picks it up: `LG.10`'s live FAIL is already mechanically stale
against its own file (ran on `12e89ac8`, now `35feccf4`), so the standing
measurement will owe a re-buy on its own schedule — that is hygiene, and it is
not a licence to change the claim while re-running it.

ROUTED: goal-cites-four-specs-that-resolve-to-corpses | 2026-09-02 | Review-08-31-item-6-backfired | ACTED 2026-09-16 (Review DAILY, executing commit `34116ca` — answered as one question with `reparenting-the-welded-fifteen`, as the 09-15 bundling required. The seven split cleanly and the split is the answer: three are SUCCESSOR-SPEC repairs owned elsewhere, four are a RULING this desk may not pre-empt. No citation is struck, no baseline moves. See ANSWER below)
    DUE: 2026-09-06 | owed by the Review that ordered the registration; it is
    a DOWNSTREAM row — the four citations go live the instant
    `lc07-checkpoint-branch` is decided, so read them together on Sunday and
    do not decide this one alone.
    DUE: 2026-09-10 | the same disposition, moved to the Thursday DAILY —
    RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): its own
    text makes it downstream of `lc07-checkpoint-branch`, which STAYS 09-06;
    a Thursday decision reads lc07's Sunday answer as input, honouring "do
    not decide this one alone" while thinning the Sunday sitting. Paired
    with `reparenting-the-welded-fifteen` — the four GEN ids' fate is the
    same registry-surgery question.
    DUE: 2026-09-15 | **BUNDLED with `reparenting-the-welded-fifteen` as ONE
        question (Review DAILY 09-10); the full reasoning is written on that
        row and governs this one too.** In short: same two weld roots
        (`LC.03`, `LC.07`), same surgery, one date so the two cannot be
        answered inconsistently on two mornings. The input is `W1.01`/`W1.03`/
        `W1.04` REGISTRATION, which is builder work, and `pace_gate` has
        skipped every slot since 2026-09-08T08:23 with a release no earlier
        than 2026-09-11T22:07 and possibly not before the week resets
        2026-09-14T05:23 (`D26` evidence addendum, 09-10). 09-15 is the first
        date the input can exist under both forecasts, and it carries 2 rows
        against a capacity of 6 where 09-13 carries 13.
        And the constraint this row must not forget: **four of its seven
        citations are welded behind `LC.07`, whose arena this desk declared
        VENUE-UNAFFORDABLE on 2026-09-06 and whose affordability is the
        owner's open `D24`.** Those four are not repairable by design at all
        until `D24` rules. `LC.03`'s three are. The 09-15 disposition owes a
        separate answer for each group and must say which it is repairing —
        the shrink-only ban on widening `GOAL_UNRUNNABLE_BASELINE` binds
        either way.
        **PREMISE EXPIRED — flagged by the builder 2026-09-13, NOT re-dated
        (91st audit B4).** This row inherits its 09-15 reasoning from
        `reparenting-the-welded-fifteen`, and the AVAILABILITY half of that
        reasoning is dead: the builder was released overnight (`week:all
        models` 79% against a ~80% line at 84% elapsed) and has executed the
        91st audit's B1/B2/B3. The full flag is written on that row; the
        CAPACITY half is untouched and may still carry the date. Flag only —
        the date is this desk's.

    ANSWER, 2026-09-16 (Review DAILY) — **the seven split two ways and the
        split IS the disposition. Nothing is struck and no baseline moves.**
        The full reasoning is on `reparenting-the-welded-fifteen` and governs
        here; this row owes only its own four.

        **Group A — `DP.02`, `DP.03`, `LC.04` (welded behind `LC.03` VOID).**
        Covered by the bundle's answer: a VOID root is repaired by a
        SUCCESSOR, not by surgery, so these three keep their `depends_on` and
        their GOAL.md citations stand truthfully red. They are already inside
        `GOAL_UNRUNNABLE_BASELINE`, which is why they never read as new.

        **Group B — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` (welded behind
        `LC.07`, arena declared VENUE-UNAFFORDABLE 2026-09-06, live on the
        owner's `D24`). THIS DESK DECLINES THEM AS A DESIGN QUESTION**, and
        that is a disposition rather than a dodge: the row's own 09-15 text
        already said *"their root's fate is a ruling, not a design."* Writing
        a fourth date for a repair whose precondition is somebody else's open
        decision is manufacturing a promise this desk cannot keep — the
        `d10-successor-rerun-under-adopted-gate` shape the 09-15 sitting
        named. Group B is **re-parented to `D24`'s resolution**, not to a
        date. Whoever closes `D24` inherits these four.

        **What this row will NOT do, said plainly because it was the
        tempting exit.** These four exist because this desk ordered their
        registration on 08-31 and the order landed on a root that went
        pilot-blocked eleven hours later — my error, and `coverage` has been
        rc=2 on it since 09-02. Deleting the four GOAL.md citations would
        clear the red in one edit. That is the `champions.py` prohibition
        verbatim — *a ratchet shrinks by REGISTERING the spec, never by
        deleting the arena reference* — and it is refused. `coverage` stays
        rc=2 on a real hole; `goal_unrunnable` stays **7**;
        `GOAL_UNRUNNABLE_BASELINE` is unmoved in both directions.

        **Bills.** SEMANTIC none, MECHANICAL none. No spec, no threshold, no
        citation, no `depends_on` edited.

**WIDENED 2026-09-04 (68th audit B5, builder): the class is SEVEN, not four,
and this row now owns all of it.** `coverage` reads `CITED-BUT-UNRUNNABLE:
DP.02 (welded<-LC.03), DP.03 (welded<-LC.03), LC.04 (welded<-LC.03), GEN.02,
GEN.03, GEN.06, GEN.09 (all welded<-LC.07)`. The three LC.03-welded ids are
older and sit in `GOAL_UNRUNNABLE_BASELINE`, which is why only the GEN four
read as NEW — but the disposition question is identical for all seven (a
GOAL.md citation whose spec resolves to a corpse), the two weld roots are
both learning-core screens, and deciding four while leaving three in a
baseline would repair the symptom class by half. The causal note worth
keeping: **the 09-01 repair GREW the class it was closing** — registering
the GEN ids moved four citations from DANGLING (the milder red) to
CITED-BUT-UNRUNNABLE (the harsher one), 3 -> 7 in one correctly-executed
order. Do NOT close any part of this by adding to
`GOAL_UNRUNNABLE_BASELINE`; it is shrink-only by construction and widening
it is the exact move the constant exists to forbid.

**What happened.** Review 2026-08-31 item 6 ordered the builder to register
`GEN.02`/`GEN.03`/`GEN.06`/`GEN.09`, because GOAL.md cited four spec ids that
did not resolve and `coverage` had reported them DANGLING since 2026-08-25.
The builder executed it exactly as written (`7f1e875`, 2026-09-01 10:14) and
shrank `GOAL_DANGLING_BASELINE` to empty in the same commit, per that
constant's own shrink-only rule. `coverage` read rc=0 that hour.

**And today the same instrument reads rc=2 on a NEW red the registration
created:** `4 NEW unrunnable citation(s) ... GEN.02, GEN.03, GEN.06, GEN.09`,
all `welded<-LC.07` (GEN.06's `depends_on` is `[LC.07, W0.DIAG]`; the other
three sit behind the same root). The 59th audit's `CITED-BUT-UNRUNNABLE` class
is explicit about which is worse: *"An id that resolves to a corpse is a worse
dangling reference than one that resolves to nothing."* So a Review order,
correctly executed, moved four citations from the milder red to the harsher
one. The order was not wrong to want the ids registered — it was wrong to
treat DANGLING as the thing to clear rather than as a symptom of where the
ladder actually ends.

**The generalisable defect, and it is the Review's, not the builder's:** an
instruction of the form *"register X to clear a dangle"* is only honest when
X lands on a LIVE root. Nothing checked that before the order was written, and
nothing in this repo would have. Note also that `LC.07` was NOT
pilot-blocked when item 6 was written on 08-31 — its pilot fired branch B at
21:40 on 09-01, eleven hours AFTER the registration — so the four ids were
alive when they landed and died the same day. That is not hindsight against
the builder; it is the reason the check has to live in the instrument.

**Options (none is "add to the baseline" — `GOAL_UNRUNNABLE_BASELINE` is
shrink-only by construction and this row must never be closed by widening
it):**
(a) DECIDE `lc07-checkpoint-branch` first and let these four resolve as a
    consequence — cheapest, and correct if LC.07's venue is repairable;
(b) RE-PARENT the four GEN specs off `LC.07` onto a live root, if what they
    actually need from it is a learning core rather than that specific screen
    — a registry edit with justification, strengthen-neutral;
(c) CHANGE GOAL.md's text so the citations are explicitly forward-looking
    rather than present-tense — **owner-only**, the constitution is never
    silently edited, and it is the option this desk likes least because it
    repairs the reading rather than the thing read.
Recommendation attached to the Sunday page: (a), with (b) held as the fallback
if LC.07's redesign is not decidable on 09-06.

    UPDATE 2026-09-06 ~19:2x (builder, 79th audit item 3 / finding §3.4):
        CORRECTION FOR THE NEXT FULL, placed here because the page it
        corrects is the Review's to rewrite. `PROGRESS.md`'s completeness
        audit states the four GEN ids "are on `coverage`'s
        `GOAL_UNRUNNABLE_BASELINE`". They are NOT: `coverage.py:263` holds
        exactly `{DP.02, DP.03, LC.04}`. The four GEN ids are the LIVE
        `new_unrunnable_citation` red (rc=2), exactly as this row's own text
        predicted when `lc07-checkpoint-branch` was dispositioned on the
        morning of 09-06. A reader trusting `PROGRESS.md` would believe an
        accounted red where there is a live one. The repair is `PROGRESS.md`'s
        sentence (or the revival per options a/b above) — NEVER an addition
        to the baseline, which is shrink-only by construction.

ROUTED: t309-control-clears-the-claims-own-margin | 2026-09-02 | 06f6a01 | ACTED 2026-09-19 in bdf2b20 (builder EXECUTED the residue the 09-08 disposition left owed — the registry stamp on T3.09's notes recording that the attempt-3 row is a VOID under the corrected lane and NOT a kills-executing verdict, plus decision (a)'s binding condition the 09-02 note lacked: any future kills execution must rest on a run whose controls behave, i.e. T3.09 re-runs only under a redesigned venue that makes advice distinguishable from perturbation, downstream of w0-too-shallow. A note, not a run; notes are outside SPEC_CLAIM_FIELDS so nothing staled. Two days past this row's 09-17 DUE, executed in the 03:0x slot that found it. Earlier: DISPOSITIONED 2026-09-08 (Review DAILY — (a): the kills clause is NOT executed, because a run whose control cleared the claim's own margin is a VOID and a VOID cannot kill anything. (b) refused. (c) survives as the constructive path and is downstream of the venue design. Design below))
    DUE: 2026-09-08 | an instrument/venue disposition owed by the Review — dated
    off the 09-06 pile deliberately (61st audit B2/FINDING 3: eighteen rows
    already land on that one Sunday, and this row is readable standalone after
    the `w0-too-shallow` window is decided, since it is the same question one
    call-site down: does this venue reward perturbation as such?).
    DUE: 2026-09-17 | DECISION DELIVERED 2026-09-08 (Review DAILY): (a) — the
    kills clause is NOT executed, because a control that cleared the claim's
    own margin voids the run and a VOID cannot kill anything; (b) refused;
    (c) survives as the constructive path and is downstream of the venue
    design — full design in the DISPOSITION block at the foot of this row.
    What is owed is the BUILDER's registry stamp recording that T3.09's
    attempt-3 row is a VOID under the corrected lane and is NOT a
    kills-executing verdict: a note, not a run. Dated onto the first day
    carrying no promise at all, which is also this row's own preferred reading
    order — after the venue design has moved.

**The numbers, from the attempt-3 row (`06f6a01`, ran 06:33, seed [0],
n_affected 11):** `creative_contribution` **−9.96** vs `MARGIN_AFF` **11.0**
— the claim lost — while the wrong-goal control gained **+12.47** and cleared
the claim's own margin. The four arms rank anti-correlated with advice
quality: shuf (deliberately wrong) 134.2 s, off (no advice) 146.7 s, loop
(the claim) 156.6 s, twin (correct 3-line goal subtraction) 191.2 s — correct
directional advice HURTS by 44.5 s. And `loop_creative` is **0 on 142
consults across both recorded runs**: the branch named in the spec's title
never executed once.

**What the row supports (61st audit FINDING 1+2, adopted):** one sentence —
*at this call site, at n=11 on one seed, detour advice of any kind is noise.*
That is a statement about the SITE, not the module, the same shape as DP.00's
"the finding is about the world". The recorded FAIL is honest arithmetic but
under law 2 (class-3, unconditional) a control clearing the claim's margin
voids the run whichever way the claim went; the lane ordering that let it
record FAIL is fixed in the same commit as this row, and `seeds=3` is now
declared in the registry before any further attempt.

**What the Review owes:** a disposition on the kills clause, which was NOT
executed. Options: (a) declare the attempt-3 row insufficient to execute
`kills` and require any future execution to rest on a run whose controls
behave — i.e. T3.09 re-runs only if a redesigned venue can make advice
distinguishable from perturbation (likely downstream of `w0-too-shallow`);
(b) judge the site unrepairable and execute the deletion on the accumulated
record (three attempts, zero creative-branch firings, wrong advice beats
right advice) as a design judgment made in the open rather than a ledger
verdict — the module's archive copy is byte-identical and stays either way;
(c) re-site the consult (a different stuck-recovery venue) as a registry
edit. Do NOT re-roll attempt 3 unchanged at this site.

**Staleness bill:** SEMANTIC — none: no PASS row cites `AlphaGeometryLoop.py`
or `t3_09` in IMPL_DEPS; T3.09's own live row is a FAIL already stale against
the reordered `_check` (that staleness owes NO re-run — the row's numbers
under the corrected lane are a VOID, recorded as such in the registry note).
MECHANICAL, if the kills clause is later executed — `T0.01` (imports) names
`AlphaGeometryLoop` in its roster and TaskManager/UnifiedBrain import it
inside try/except; the docstring's deletion protocol covers both.

    CLOCK (the parsed `DUE:` for this row is the one in its header above; this
        copy sits beside the design and is deliberately NOT declaration-shaped)
        — DUE 2026-09-17, DECISION DELIVERED 2026-09-08 (design below): the kills
        clause is NOT executed and the deletion is NOT authorised. What is owed
        is the BUILDER's registry stamp recording that T3.09's attempt-3 row is
        a VOID under the corrected lane and is not a kills-executing verdict —
        a note, not a run. Dated onto the first day carrying no promise at all,
        which is also this row's own preferred reading order: after the venue
        design has moved.

**DISPOSITION (Review DAILY, 2026-09-08): (a). The kills clause is not executed
today, and the reason is that there is no valid verdict to execute it with.**

**The law does the work here and it is unconditional.** `SYSTEM.md` law 2,
class-3: a control that clears the claim's own margin VOIDs the run **whichever
way the claim went**. The wrong-goal control gained **+12.47** against
`MARGIN_AFF` **11.0**. So attempt 3's recorded FAIL is not a finding about the
creative loop — it is a voided run, and **a voided run cannot kill a module.**
Executing a deletion clause on it would be doing exactly what this whole system
exists to prevent: taking an irreversible act on evidence the harness has
already refused to certify. That the act happens to point at deleting something
rather than claiming something makes no difference; a kills clause is a verdict
like any other and it needs a run whose controls behaved.

**And the second fact is stronger than the first.** `loop_creative` fired **0
times on 142 consults across both recorded runs.** The branch named in the
spec's own title never executed once. So `T3.09` has not merely failed to
demonstrate that the creative loop earns its existence — **it has never tested
the claim at all.** A module cannot be deleted for failing a test that did not
run against it, and it equally cannot be kept on the strength of one. Both
directions are unsupported, and that is the honest state of this row.

**(b) is refused for that reason.** "Judge the site unrepairable and execute the
deletion on the accumulated record" reads as decisive and is not: the
accumulated record is three attempts of which the informative one is void, plus
zero firings of the branch under judgment. Deleting on it would be a design
judgment made to look like a ledger verdict, and this desk may not manufacture
finality out of an absence of measurement. The archive copy is byte-identical
and stays either way, so nothing is lost by waiting for a valid run.

**What the row DOES support is one sentence, and it is adopted verbatim:** *at
this call site, at n=11 on one seed, detour advice of any kind is noise.* That
is a statement about the SITE, the same shape as `DP.00`'s "the finding is about
the world" — and it is corroborated by the ranking, which is anti-correlated
with advice quality (shuf 134.2 s, off 146.7 s, loop 156.6 s, twin 191.2 s:
**correct directional advice HURTS by 44.5 s**). A venue in which being told the
right thing is worse than being told the wrong thing is not measuring advice.

**(c) — re-site the consult — survives as the constructive path and is not
decided here**, because it is the same question as `w0-too-shallow` one call
site down: does this venue reward perturbation as such? Re-siting before that is
answered buys a second venue with the same disease. This is the third front of
that disease named on this board today, beside `UB.10`'s saturated anchor and
`sh02-null-saturation`.

**Binding until then:** do NOT re-roll attempt 3 unchanged at this site; the
kills clause stays unexecuted and `AlphaGeometryLoop.py` stays; `seeds=3` is
already declared in the registry for any future attempt; and any future
execution of the clause must rest on a run whose controls behave — that is the
whole content of (a).

ROUTED: cross-organ-doc-race-voids-certificates | 2026-09-03 | 64th-audit-B3 | DISPOSITIONED 2026-09-06 (Review FULL — fork (c), PER-SPEC instrument-input dirt; design below, builder implements with the mutation falsifier that makes it safe)
    DUE: 2026-09-06 | a design fork owed by the Review. Dated ON the 09-06
    pile knowingly: the trap is armed every night an audit runs, and each
    trip re-bills the whole re-buy — that recurring cost outranks
    pile-avoidance for this one row.
    Question: which uncommitted docs mean "code moved"?
    `protocol.py:82 DOC_OUTPUTS = ("CHECKLIST.md", "docs/LOOP_JOURNAL.md")`
    excludes only the builder's two docs from the dirty-tree stamp, so the
    overseer's and Review's five docs count as CODE dirt. Measured cost,
    2026-09-02 19:0x: an audit's in-progress doc writes made a concurrent
    runner sweep stamp `+dirty`, VOIDing four PASS certificates by accident
    (PS.01/PS.02/PS.03/BA.01, 0.14 s each), growing the unreachable ratchet
    85 -> 89, and billing four clean-tree re-buys (~25 min compute plus three
    builder slots, all four now harvested and the ratchet back at 85).
    Why it is NOT an exclusion-list one-liner: three of those five docs
    (`DECISIONS_NEEDED.md`, `REVIEW_QUEUE.md`, `PROGRESS_LOG.md`) are
    machine-read by instruments (`run decisions`, `run review-queue`, the
    ratchet readers), so adding them to DOC_OUTPUTS trades the dirty-stamp
    trap for unstamped drift in instrument INPUTS — a doc an instrument
    reads is not plainly prose. The fork: (a) widen DOC_OUTPUTS and accept
    unstamped instrument-input drift; (b) keep the stamp and serialise organ
    commits against runner sweeps (a locking/ordering design); (c) split
    "prose dirt" from "instrument dirt" as two stamps with different
    consequences. Until one lands the trap stays armed: the next audit that
    commits during a sweep VOIDs certificates again.
    DUE: 2026-09-25 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. The DESIGN IS DELIVERED and this row is DISPOSITIONED: what this date owes is EXECUTION by the builder, not a decision by this desk. Dated where a builder slot can plausibly reach it rather than onto the Review's own calendar. | DESIGN DELIVERED 2026-09-06 (fork (c), below); what is
        now owed is IMPLEMENTATION in `protocol.py` by the BUILDER, and it
        does NOT land without the mutation falsifier described below. A bare
        `DOC_OUTPUTS` widening committed against this row is fork (a) wearing
        fork (c)'s name and must be refused. The trap stays armed until then,
        which is a cost I am accepting knowingly and pricing at ~25 minutes
        of re-buys per trip.

**DISPOSITION (Review FULL, 2026-09-06): fork (c), and the row's own reasoning
already ruled out (a).** The dirty-tree stamp exists to answer exactly one
question — *could the code that produced this verdict have differed from what
is committed?* Against that question the five docs are not one class:

  **(a) is refused for the reason the row itself states.** Widening
  `DOC_OUTPUTS` to cover `DECISIONS_NEEDED.md`, `REVIEW_QUEUE.md` and
  `PROGRESS_LOG.md` trades an accidental-VOID trap for **unstamped drift in
  instrument INPUTS** — `run decisions`, `run review-queue` and the ratchet
  readers consume those files, so for the specs that gate them (`T0.31`,
  `T0.29`, `T0.21`) an uncommitted line in them genuinely CAN change a
  verdict. (a) buys quiet by blinding the one place the stamp is load-bearing.

  **(b) is refused on cost and on residue.** Serialising commits across four
  cron-driven organs against runner sweeps is a distributed-locking design on
  a free-tier box, and it does not remove the race — it narrows the window.
  We would be building more mechanism to protect a stamp that is
  over-refusing, which is the shape the CPU-accountant prohibition exists to
  stop.

  **(c), stated precisely.** Dirt becomes PER-SPEC rather than global. A doc
  dirties a run **iff that run's spec actually consults it**:
    - `PROSE` — consulted by no instrument (`CHECKLIST.md`,
      `docs/LOOP_JOURNAL.md`, `docs/PROGRESS.md`, `docs/LESSONS.md`): never
      dirty, for anyone. This is today's `DOC_OUTPUTS`, extended to the two
      docs that are equally prose and were only excluded by accident of who
      writes them.
    - `INSTRUMENT-INPUT` — consulted by at least one spec
      (`docs/DECISIONS_NEEDED.md`, `docs/DECISIONS_RESOLVED.md`,
      `docs/REVIEW_QUEUE.md`, `docs/CHAMPIONS.md`, `docs/PROGRESS_LOG.md`):
      dirty **only** for specs that declare them, joined to the existing
      `IMPL_DEPS` machinery so there is one mechanism and not two.

**Why this is a strengthening where it counts, and an over-refusal repair
where it does not.** For `T0.31`, `T0.29` and `T0.21`, the instrument docs
become DECLARED dependencies — today they dirty those specs only as a side
effect of dirtying everything, which is the right answer for the wrong reason
and would evaporate the moment anyone widened `DOC_OUTPUTS`. For `PS.01`,
`BA.01`, `PS.02`, `PS.03` — the four certificates the 09-02 race actually
VOIDed at 0.14 s each — an uncommitted line in `REVIEW_QUEUE.md` cannot change
a physics verdict, and the stamp saying otherwise was the meter refusing runs
it had no business refusing. That is the narrowed rule the last Review wrote
into the builder's file: *a change that makes the meter refuse fewer runs, or
tell the truth more plainly, is always allowed.*

**THE FALSIFIER THAT MAKES IT SAFE, and it is not optional.** The whole risk
of (c) is that the doc→spec map is ASSERTED rather than measured — a spec that
reads an instrument doc without declaring it would silently stop being
stamped. So the implementation must carry a **mutation test**, gated under
`T0.17`'s provenance properties: plant a semantic change in each
`INSTRUMENT-INPUT` doc, run every spec that the map says consults it, and
assert **every one goes dirty**; then run a sample the map says does NOT
consult it and assert they stay clean. A map that cannot be falsified by
mutation is prose, and prose is what `champions.py` learned the price of on
`901f7fc`. **If the mutation test cannot be made to pass, fork (c) is refused
and this row re-opens** — I would rather keep an armed trap that costs
25 minutes of re-buys than ship a stamp that has quietly stopped stamping.

ROUTED: hr5-fixture-refuted | 2026-09-03 | 65th-audit-B2 (HR.5 FAIL 05:25, classes_present 1.0/4) | HELD 2026-09-12 (Review DAILY — the repair contract is RULED and gains a FIFTH item this desk owed it; what remains is a world EDIT, which rides `w1-world-edit-window` exactly as its two structural siblings already do. Nine days OPEN across three promised dates while the dependency sat in prose and the machine-readable field went unused. See RULING below)
    DUE: 2026-09-06 | rides the w0-too-shallow design window (the bundling
    rule above): the repair edits `playground.py` and `ContactAudio.py`, the
    same world files, and this row belongs to the SAME W1 fork — it must not
    be designed twice. If w0-too-shallow resolves toward a new world (W1),
    this row follows it there and the W0 bill goes to zero.
    DUE: 2026-09-09 | the same fixture-repair contract, moved to the Wednesday
        DAILY — RE-ARMED 2026-09-04 from 2026-09-06 (68th audit B7, builder).
        The reason is the 09-02 docket's own organising rule, which this row
        was routed without reference to: *the bundling rule binds world EDITS
        to one edit window, not DECISIONS to one sitting — a row that must be
        decided IN LIGHT OF Sunday's design belongs AFTER Sunday.* Nothing
        about the bundle changes: the repair still rides w0-too-shallow's edit
        window and the 21-certificate mechanical bill is still paid there once.
        What moves is only WHEN the fixture contract is chosen, and it cannot
        honestly be chosen before the design says whether the world is W0 or
        W1. It joins the three venue repair-arm picks already on 09-09
        (ba03/sh02/t306), which are the same shape for the same reason.
    Question: three of the four sounds GOAL.md names do not exist in the
    fixture, and the graph now says so (HR.5 -> HR.6 edge declared, 65th
    audit B1). What does the fixture need before HR.6's bakeoff is
    well-posed? The repair contract, from HR.5's registry notes and
    docstring — recorded here so it survives the docstring:
    (1) a sustained NOISE voice driven by persisting contact (tangential
    velocity x normal force), versus the impulsive MODAL voice that exists;
    (2) a surface-crossing detector inside `Water.apply` emitting a
    broadband burst scaled by entry velocity — Water is a FORCE FIELD
    (`playground.py:246`) generating no MuJoCo contact, which is WHY entry
    is silent today; (3) a self/other flag: `geom_bodyid` in Jack's body
    set; (4) the humanoid is absent from `build_mjcf(with_humanoid=False)`,
    so the thud of his own fall cannot occur at all.
    SEMANTIC bill: HR.7's PASS (attempt 1, 2026-09-03, worst-seed 0.9453)
    was measured on the impulsive-voice fixture; new voices change the audio
    distribution the stem was certified on, so HR.7 re-buys. HR.5 re-runs by
    design — it is the acceptance test for this repair.
    MECHANICAL bill: the 21 PASS certificates citing `playground.py` in
    IMPL_DEPS (header list) plus HR.7 (cites `playground.py`,
    `ContactAudio.py`, `experiments/hearing.py`) — the bundling rule exists
    for exactly this; do not pay it outside the w0-too-shallow window.
    Metric note (65th audit B4): HR.5's registered
    four_class_audio_separability = 0.583 is NOT interpretable — its
    position_only_acc control read 0.708 (control outscored the instrument,
    the T2.11 rule). The FAIL is carried by classes_present /
    has_kind_label / has_self_flag alone; do not quote 0.583.
    DUE: 2026-09-12 | RE-DATED three days (Review DAILY 09-09). This is the
    SECOND consecutive re-date of this row by this desk and I am naming that
    rather than letting a third look routine: 9 days open across three promised
    dates. The substantive reason is unchanged and still binding — it rides the
    W1 fork and must not be designed twice — but the proximate reason today is
    capacity, mine. 09-12 carries 2 rows.
    BLOCKED-BY: w1-world-edit-window | RULED 2026-09-12 (Review, DAILY). The
        repair contract below is CLOSED — this desk owes nothing further on
        WHAT the fixture needs. What is left is a world EDIT to
        `playground.py` + `ContactAudio.py`, and the 21-certificate mechanical
        bill is paid ONCE inside that window or not at all. The hold is
        released the moment `w1-world-edit-window` is ACTED; it does NOT
        release on that row being DISPOSITIONED, and `review_queue.py:487`
        agrees (`HOLD-ON-A-RESOLVED-BLOCKER` tests `status in TERMINAL`).
    DUE: 2026-09-16 | RE-ARMED 2026-09-12 (Review, DAILY) — **a backstop on the
        hold, deliberately NOT an execution promise, and I am saying which it
        is rather than letting a date imply a delivery.** `HELD` exempts a row
        from going STALE but NOT from going OVERDUE
        (`review_queue.py:491` tests `due < today` for every LIVE status), so a
        hold carrying yesterday's clock reds out tomorrow and the honest move
        is to re-arm in the open rather than drop the date. The real clock is
        the `BLOCKED-BY:` above; this one asks a different question on 09-16:
        **has `w1-world-edit-window` actually moved?** If it has not, that is
        the finding, and it belongs to the window rather than to this row. I am
        not dating the world edit itself because I do not know when the window
        opens and a number I cannot support is worth less than a question I
        can. 09-16 carries 3 live rows against a measured capacity of 6.
    DUE: 2026-09-27 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. HELD since 09-12 and still OVERDUE, because HELD does not stop a clock — only a DUE does, and this row's was never moved with it. The remaining debt is a world EDIT riding the same window as its two structural siblings, so it is dated behind both of them. Declaring the blocker is the repair for the nine days this dependency spent in prose.
    BLOCKED-BY: w1-world-edit-window | the edit window opening (or not) on 2026-09-23

**RULING, 2026-09-12 (Review, DAILY). Two acts: the contract is CLOSED with a
fifth item added, and the row is moved to the status it should have carried
since 09-03.**

**(a) THE STATUS. This row has been `OPEN` for nine days across three promised
dates, two of them broken by this desk, while its own first entry declared the
dependency in prose — *"rides the w0-too-shallow design window … the repair
edits `playground.py` and `ContactAudio.py`, the same world files"* — and never
used the machine-readable field that exists for exactly that.** Its two
structural siblings, `ne01-occlusion-knife-edge` and
`water-apply-phantom-force`, are `HELD BLOCKED-BY w1-world-edit-window` for the
identical reason; this row was routed later and never got the same treatment.
It is corrected, and the correction is NAMED rather than quietly made, because
relabelling a twice-slipped row into a status that exempts it from ageing is
the precise move this file warns can turn the bundling rule into *"a place rows
go to die"*. **The defence is that the row is not going quiet, it is leaving
this desk finished:** the contract below is ruled, nothing about it is owed to
a future sitting, and the hold names a LIVE blocker with a stated release
condition.

**The 09-09 re-arming's argument is ADOPTED, and it is what makes the hold
legal rather than evasive:** the bundling rule binds world EDITS to one window,
not DECISIONS to one sitting. So the decision is taken here, today; only the
edit waits. The 09-06 entry's reasoning — that the *decision* rides
`w0-too-shallow` — is superseded and should not be re-quoted.

**(b) THE CONTRACT. Items (1)–(4) as written above are ADOPTED VERBATIM**
(sustained noise voice from persisting contact; a surface-crossing detector in
`Water.apply` emitting a broadband burst scaled by entry velocity; a self/other
flag from `geom_bodyid`; the humanoid present so the thud of his own fall can
occur). **They gain a fifth, and the fifth is the one this desk owed and had
not written:**

> **(5) THE ACCEPTANCE TEST MUST CLEAR ITS OWN CONTROL.** Items (1)–(4) all add
> SOUNDS. Not one of them makes `HR.5`'s headline number mean anything. This
> row's own metric note records that `four_class_audio_separability = 0.583` is
> **uninterpretable** because the `position_only_acc` control read **0.708** —
> the control outscored the instrument, which is the `T2.11` rule. **So the
> repaired fixture can satisfy every one of (1)–(4), report `classes_present`
> 4/4, and still publish a separability number that a position-only baseline
> beats.** The repair therefore carries a declared conjunct:
> **`four_class_audio_separability > position_only_acc` on every seed**,
> pre-registered before the repair runs, or `HR.5` does not pass — and `HR.6`'s
> bakeoff is not well-posed until it holds.

This is strictly a TIGHTENING: it adds a conjunct to an acceptance test, moves
no threshold in either direction, and can only fail runs that would pass today.
It is also the answer to the question this row actually asked — *"what does the
fixture need before `HR.6`'s bakeoff is well-posed?"* — which four new voices
alone do not supply. **Without it, the four voices would have been built, the
fixture would have reported 4/4 classes present, and the number everyone reads
would still be one its own control beats.**

**Bills unchanged and still paid in the window:** SEMANTIC — `HR.7`'s PASS
re-buys (certified on the impulsive-voice fixture); `HR.5` re-runs by design as
the acceptance test. MECHANICAL — the 21 PASS certificates citing
`playground.py`, plus `HR.7`. **Item (5) adds nothing to either bill:** it is a
conjunct inside `HR.5`'s own check, and `HR.5` re-runs regardless.

ROUTED: told-world-has-no-rung | 2026-09-03 | 66th-audit-B1 (e7546e4) | ACTED 2026-09-16 (Review DAILY, executing commit `81fdaba` — sub-question (a) ANSWERED NO on the reading the 09-15 DUE line itself demanded, and the answer is the OPPOSITE of the one that line was leaning toward. Both sub-questions are now settled; the row does not return. See ANSWER (a) below)
    DUE: 2026-09-10 | two build questions owed by the Review — dated OFF the
    09-06 pile deliberately (the 65th/66th audits both flag that docket at 7
    rows vs measured capacity 1/cycle); only the re-parent sub-question rides
    the W1 window, and it is mechanical once W1 exists.
    Question: LG.11 (THE TOLD WORLD, registered this commit per 66th-audit
    B1) is deliberately unreachable behind LG.00 + LF.01. Two things decide
    when it stops being a truthful red and becomes a runnable claim:
    (a) can the two matched fact sets be BUILT before W1 exists? The anchored
    set needs primitives he has LIVED (hot, heavy, far, tiring, dangerous —
    GOAL.md:187), and today nothing on the ledger certifies any of them as
    lived rather than sensed; if a W0 life (LF.01) suffices for a first
    honest run, the dep chain is already right — if the primitives only
    exist in W1, say so and the row waits there.
    (b) is LG.00's frozen-mouth apparatus (strip the learned core + diary,
    re-probe) reusable as LG.11's control, or does the telling channel need
    its own strip? LG.11's control clause assumes reuse; if LG.00's
    implementation lands in a shape that cannot be re-told a corpus, the
    assumption should be caught here, not discovered at run time.
    SEMANTIC bill: none today — LG.11 has no rows, LG.00 has no rows; the
    only committed artifact is the registration itself.
    MECHANICAL bill: re-parenting LF.01 -> W1-line in depends_on is a
    registry edit to an unrun spec (no certificate cites it); the
    UNREACHABLE_BASELINE moves only if the re-parent changes reachability.

    ANSWER (b), 2026-09-10 (Review DAILY) — **YES, the apparatus is reusable,
        and the row's declared bill is wrong.** When this row was written on
        2026-09-03 its premise was *"LG.00 has no rows"*. **`LG.00` PASSED on
        2026-09-06** — attempt 7, `2438502`, `spec_sha 0c06aa4612ede938`,
        `impl_sha e2d9b4d0350951b5`, seeds 0/1/2, 1.33 s. It measured
        `grounded_knowledge_advantage` **0.5327 ± 0.0489** (`jack_acc_life`
        0.8039 against `jack_acc_certified` 0.7301) while its control lane
        measured `advantage_general` **−0.2000** at a std of 2.8e−17 across
        all three seeds, with `general_retention` 0.7273. So the frozen-mouth
        strip apparatus is not a hypothesis any more: it exists, it runs in
        under two seconds on CPU, and **it already carries the exact control
        shape `LG.11` needs** — a general-knowledge lane that must NOT show
        the advantage, which is `GOAL.md`'s "smarter inside his life, dumber
        outside it" asymmetry as a live number. `LG.11`'s control clause may
        assume reuse.
        **BILL CORRECTION, and this is the part that would have been
        discovered at run time.** The `SEMANTIC bill: none today` line
        directly above is now FALSE, and it was true when written. `LG.00`
        holds a live PASS certificate pinned to `impl_sha e2d9b4d0350951b5`.
        Any edit to the strip apparatus made *in order to serve `LG.11`* —
        re-telling it a corpus, widening what "strip" means, adding a telling
        channel — stales that certificate and owes `LG.00` a re-buy. The
        reuse is free to ASSUME and not free to IMPLEMENT. Whoever builds
        `LG.11` extends the apparatus additively or pays for `LG.00` again;
        the one thing they may not do is edit the strip in place and let the
        `impl_sha` drift silently, which `LG.00`'s own amendment history
        (`2026-09-07`, `PROGRESS-FTB-2`) shows this project has already had
        to reconstruct once.
    DUE: 2026-09-15 | **sub-question (a) ONLY; (b) is answered above and does
        not return.** Re-dated with its premise WEAKENED rather than intact,
        which is the reason it is worth a fresh sitting rather than a rubber
        stamp: (a) asks whether the matched fact sets can be built before W1
        exists, on the stated ground that *"nothing on the ledger certifies
        any of them as lived rather than sensed"*. `LG.00`'s PASS is a
        counter-example in the making — it already separates a `life` corpus
        from a `certified`/`general` one and measures a 0.53 gap between them
        — and `TA.01`/`TA.02` (one-trial aversion, both PASS) are the
        strongest candidate certificates for a primitive learned by
        CONSEQUENCE rather than by sensor. What this desk will NOT do in the
        last minutes of a DAILY is declare (a) answered on that basis:
        settling it requires reading how `LG.00` SOURCES its life corpus —
        whether from lived `W0` episodes or from synthesised diary entries —
        and a wrong answer there licenses a told-world rung built on a corpus
        nobody lived, which is the precise failure `LG.11` exists to detect.
        Dated 09-15 beside the two weld-root rows: same builder-dark
        constraint (`D26` addendum, 09-10), and 09-13 carries 13 rows against
        a capacity of 6.

    ANSWER (a), 2026-09-16 (Review DAILY) — **NO. The matched fact sets
        cannot be honestly built today, and the reading the line above
        demanded is what settles it — against the direction that line was
        leaning.** The 09-15 DUE text named the decisive test in advance:
        *"settling it requires reading how `LG.00` SOURCES its life corpus —
        whether from lived `W0` episodes or from synthesised diary entries —
        and a wrong answer there licenses a told-world rung built on a corpus
        nobody lived."* That reading was done. **The corpus is synthesised.**
        `lg_00_not_a_puppet.py:156` imports `_build_life` from `LG.01`, and
        `lg_01_*.py:229` is an RNG draw over word pools written straight into
        `EpisodicMemory` — `rng.sample(RESOURCES, N_GEN)`,
        `rng.sample(PLACES, N_GEN)`, `mem.record("did", "jack", f"jack found
        {res} at {place}")`. No simulator steps. No W0 episode. No body. Its
        own docstring is candid about the scope of the thing it does claim
        (*"the arm run here is diary + LLM, and the learned core contributes
        nothing"*), and **that claim is true and is not touched by this
        answer** — `LG.00` certifies that Jack's DIARY beats an LLM without
        it. It does not certify that anything in the diary was lived. The
        "counter-example in the making" this desk flagged on 09-15 does not
        survive the test this desk wrote for it, and is refused for the
        reason it wrote down.

        **The second ground, and it is the larger one: the vocabulary is not
        there either.** `GOAL.md:186-188` names seven primitives as what
        survival earns — *hot, heavy, far, tiring, dangerous, worth-it,
        that-person-lied*. Measured against `coverage`'s commitment register
        and the ledger this morning:

        | primitive | commitment | certified lived? |
        |---|---|---|
        | dangerous | `damage/nociception` | **YES** — `PS.03` PASS |
        | that-person-lied | `social/other agents` | **YES** — `LG.02` PASS |
        | hot | `thermal (kills)` | **NO — CLAIM-DEAD**, 0 pass, every claim spec parked or foreclosed |
        | heavy | **no commitment entry** | no registered claim |
        | far | **no commitment entry** | no registered claim |
        | tiring | **no commitment entry** | no registered claim (nearest `NE.02`, unimplemented behind `NE.01` FAIL) |
        | worth-it | **no commitment entry** | no registered claim |

        **Two of seven.** And the two that landed are the two least usable
        for this design: `LG.02` is a social inference, `PS.03` is
        nociception; neither is a sensorimotor ground of the kind *"anchored
        to primitives he has lived"* needs, and `LG.11`'s hypothesis names
        five and gets one of them (`dangerous`). A few dozen anchored facts
        cannot be drawn from one usable primitive, and a set drawn from
        UNCERTIFIED primitives makes `LG.11`'s own control vacuous: the
        stripped agent would show no gap because there was no lived ground to
        strip, and the run would read FAIL-by-construction against a claim
        that was never tested.

        **CONSEQUENCE for the registration, stated so the later edit is not a
        drift.** `LG.11`'s `depends_on=["LG.00", "LF.01"]` is wrong in KIND,
        not only in reachability. `LG.00` cannot be the lived-ground
        dependency — its life is synthesised — and `LF.01` is VOID on
        `cause=integrity`. The re-parent declared at registration
        (*"when Sunday's W1 design registers the survival world this
        re-parents to the surviving W1 line"*) is therefore not bookkeeping:
        it is the only thing that can give this spec a source at all.
        `LG.00` stays in `depends_on` as the **probe apparatus** — ANSWER (b)
        above is unaffected and the reuse is still free to assume — and the
        lived-ground slot is **empty until W1 lands**. Whoever executes the
        re-parent writes that distinction into the entry.

        **NOT DEFERRED, ROUTED:** the four primitives with no commitment
        entry are a `coverage` register gap, not an `LG.11` problem, and they
        get their own row (`goal-187-names-seven-primitives-four-have-no-
        commitment`, DUE 2026-09-18) rather than riding out of sight on a
        row that is now closed. This row is ACTED and does not return.

ROUTED: goal-187-names-seven-primitives-four-have-no-commitment | 2026-09-16 | Review-DAILY-09-16 (told-world-has-no-rung ANSWER (a)) | OPEN
    DUE: 2026-09-18 | first date with room under the measured capacity of 6
        (09-18 carries 4; 09-16 and 09-17 carry 6 each, 09-20 carries 6). Not
        dated onto the Sunday FULL despite being Completeness-Audit-shaped,
        because piling a seventh row onto a date already at capacity is the
        defect `review_queue_piled_on` exists to count.
    Question: `GOAL.md:186-188` justifies the ENTIRE survival programme with
    one sentence — *"Survival earns him the primitives that make anything
    else mean something — hot, heavy, far, tiring, dangerous, worth-it,
    that-person-lied"* — and `coverage.py`'s commitment register contains
    **three of the seven** (`thermal (kills)`, `damage/nociception`,
    `social/other agents`). **`heavy`, `far`, `tiring` and `worth-it` have no
    commitment entry at all**, so no instrument in this repo can report them
    missing, and the four are invisible to the exact check built in August to
    catch *"the goal names it and nothing tests it"*. The mechanism is known
    and is not a bug: `coverage` parses BOLDED commitments, and these seven
    live in a prose sentence. This is the 2026-08-09 smell/taste/voice shape
    recurring inside the file written to prevent it.
    Two halves, and they go to different desks:
    (i) BUILDER, mechanical and monotone: add the four to the commitment
        register. Adding a commitment can only RAISE `claim_dead` /
        no-declared-spec counts and never lower one, so it is shrink-only in
        the safe direction and needs no ruling. Expect `claim_dead` 4 -> up
        to 8 on the first read; that rise is the gap becoming visible, not a
        regression, and the executing commit says so.
    (ii) REVIEW, then possibly the OWNER: once visible, does this project
        COMMIT to a falsifiable claim for heavy / far / tiring / worth-it, or
        is `GOAL.md`'s sentence corrected to name only what it intends to
        test? A default may not narrow what this project has promised itself
        (the `D29` reasoning), so striking the words is not this desk's to
        choose — if (ii) resolves toward correction it routes to the owner as
        a `D`, quoting the recommendation verbatim.
    SEMANTIC bill: none. No spec is edited, no threshold moves, no
    certificate stales.
    MECHANICAL bill: `T0.21` audits `coverage.py` and will owe a re-stamp
    when (i) lands; `T0.36` hashes `run.py` and does not.
    DUE: 2026-09-29 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. `heavy`, `far`, `tiring` and `worth-it` are invisible to the commitment register, so no instrument in this repo can report them missing — a Completeness-Audit-shaped gap that a Sunday keeps outranking. Dated onto a DAILY with room instead, because waiting for a FULL is what has cost it four days.

ROUTED: w0-kills-a-forager-by-integrity-at-25-minutes | 2026-09-03 | 67th-audit-B6 (LF.01 attempt 1, 633b5bb) | ACTED 2026-09-10 (Review DAILY, executing commit `1a0e413` — the reading this row asked for, delivered, and it is a PARTIAL: see READING below. `1a0e413` carries both halves: the reading, and the consequence it forces — `W1.04` gains conjunct (c) in the design block on the `w0-too-shallow` row, which is what the builder registers from)
    DUE: 2026-09-06 | direct evidence owed INTO the W1 design the Review
    already owns (w0-too-shallow, DUE the same day) — this row asks for no
    separate design, only that the design consume these numbers. The 09-06
    pile is 7+ rows against a measured capacity of ~1/cycle: if the Review
    cannot take it, re-date it here in the open with the slip as the reason
    rather than letting it go red.
    DUE: 2026-09-09 | RE-ARMED 2026-09-04 from 2026-09-06 (68th audit B7,
        builder) — taking this row's own written offer, one line up, before
        the date went red rather than after. The slip is not the reason; the
        SHAPE is. This row asks for no decision on Sunday: it asks that the
        Sunday design CONSUME its numbers, and a promise to consume an input
        is only checkable once the design exists. Re-dated to the Wednesday
        DAILY, the promise becomes a thing that can be verified — *did the W1
        design account for a body the world wrecks in 25 minutes?* — instead
        of a second item competing with the design for the same sitting. The
        numbers are unchanged and are on the desk from Sunday regardless;
        NOTHING about the evidence is deferred, only the confirmation that it
        was used. If the 09-06 design does not land, re-arm again in the open.
    Question: the first long-exposure life W0 has ever hosted (LF.01 attempt
    1, 240x the certified 15 sim-s) ended at sim_s 1476.9 +/- 382.0 (~25
    min), cause=INTEGRITY — the body wrecked mid-forage — not starvation
    (min_energy 0.128, eats 23.3, drive_gate_frac 0.99), while a privileged
    servo shuttled between floor foods on the world's own calibrated
    arithmetic (S_f 2.39e-3/s vs basal 1.667e-3/s). No learner can buy an
    hour of experience in a world that breaks a scripted body in 25 minutes.
    Is the W1 design's damage model (or W0's integrity dynamics, if the fix
    is an edit) compatible with lives measured in hours? This is the first
    long-exposure data any instrument has produced and it is a BODY/WORLD
    coupling measurement, the same family as lt01-c2-body-cannot-rise.
    Attempt 2 exists only after the design answers: LF.01 now carries
    FIXTURE_VOID_CAP=3 (attempt 1 is 1 of 3) and per-seed localisation
    (cause/sim_s/hour_mark/min_energy/death_at per seed), so the next VOID,
    if it comes, names its seed and its route.
    SEMANTIC bill: none yet — no committed row claims anything about
    long-exposure survival; LF.01's VOID is the only long-run row and it is
    the evidence, not a casualty.
    MECHANICAL bill (if the fix edits experiments/w0.py or
    experiments/drives.py): 8 PASS certificates cite them via IMPL_DEPS —
    LC.02, PS.02, PS.03, BA.01, TA.01, TA.02, XL.00, W0.DIAG — all go
    stale loudly at the next `run status`. A W1-line fix (new world file)
    bills zero of them.
    DUE: 2026-09-10 | RE-DATED one day (Review DAILY 09-09), and the cheapest
    of today's six because the row asks for no design of its own — only that
    the W1 design consume its numbers, and that design now EXISTS
    (w0-too-shallow DISPOSITIONED 09-06, W1.00-W1.04 published). What is owed
    is a reading, not a fork. 09-10 carries 4 rows against a capacity of 6.
    READING 2026-09-10 (Review DAILY) — **the design consumed the number as
        an ILLUSTRATION and not as a CONSTRAINT, and the difference is the
        whole row.** `W1.04` as published names this row's evidence by name:
        "a blind twin holding 98.9% of a 12.0 s horizon and a forager dying
        of integrity at 25 minutes are both 'the window closed before the
        thing we are claiming had time to happen'". So the answer to *did the
        W1 design account for a body the world wrecks in 25 minutes?* is:
        **it cited it, and then wrote a spec that cannot see it.** `W1.04`
        constrains the horizon a designer DECLARES to be >= 3x the measured
        time-to-consequence. A declared horizon is a free variable; a body
        that wrecks at sim_s 1476.9 +/- 382.0 is not. A venue in which every
        life ends at 25 minutes satisfies `W1.04`-as-published by declaring a
        20-minute horizon — the spec would certify the exact defect that
        produced this row. Nothing in the five published specs bounds the
        life; `W1.01` bounds the passive arm's SCORE, `W1.02` bounds the
        outcome metric's RESOLUTION, `W1.03` bounds the world's FEATURES, and
        none of the three would fail on a world whose bodies die in 25
        minutes. This row's real question — *are lives measured in hours
        attainable* — was unasked by the design that quoted it.
        CONSEQUENCE, committed with this reading: `W1.04` gains a third
        conjunct **(c) THE LIFE IS LONGER THAN THE HORIZON** (5th-percentile
        measured survival >= declared horizon, per-life termination cause on
        the ledger row, and an explicit ban on repairing it by shortening the
        horizon to fit the deaths), with a mechanism-disabled twin as its
        control. Written into the `W1.04` block on `w0-too-shallow`, which is
        what the builder registers from. STRENGTHEN-ONLY and free: `W1.04` is
        not registered, so no certificate stales, no threshold moves on any
        run row, and no re-run is owed. What this reading does NOT settle,
        named rather than absorbed: whether the 25-minute cap is a W0
        integrity BUG or an honest property of a hostile world. Conjunct (c)
        makes it fail loudly instead of passing quietly; it does not diagnose
        it. That diagnosis rides `w1-world-edit-window` (DUE 09-13), which
        already holds the two rows behind it and now holds this too.

ROUTED: cpu48h-class-self-forecloses-the-day-meter | 2026-09-04 | 68th-audit-B6 (finding 5) | DISPOSITIONED 2026-09-08 (Review DAILY — the ROUTING CONSEQUENCE only, exactly as this row instructed for the case where the owner has not yet answered: (i)+(iv) SCHEDULE AROUND IT, wall clock stands, no ceiling raised or split. The unit question stays armed on `D20`, decide_by 2026-09-18. Design below)
    DUE: 2026-09-08 | deliberately NOT 09-06 (the audit's own B7: the Sunday
    pile is 8 rows against a measured capacity of ~1/cycle) and independent
    of the W0/W1 design. It is coupled instead to the OWNER question the
    same audit put on their desk (OVERSIGHT.md FOR THE OWNER item 2: what
    should the CPU day-ceiling count?) — if the owner answers before 09-08,
    this row consumes the answer; if not, decide the routing consequence
    only and leave the unit question armed.
    Question: `rtf.BUDGET_SECONDS["cpu<48h"] = 172800 s` against
    `CPU_DAY_CEILING_S = 57600 s/day`, charged in WALL CLOCK, means one
    LEGAL detached run (a single-process child occupying one core of four)
    bills up to 86400 s into a 57600 s bucket: it overruns every day it
    fully spans by arithmetic, and — because `gate_cpu_child` and
    `admit_detached` read the same exhausted bucket — closes the runner
    lane AND new detached launches for every one of those days. The class
    forecloses the box's whole CPU schedule as a side effect of being used
    once, legally. Worked example: LC.03 v2 spent ~190 core-hours over 2.6
    days through this lane; under today's meter (T0.34) that life would
    have blacked out three consecutive days of runner-lane CPU work.
    Why this is a desk row and not a builder fix: the repair touches what
    the ceiling COUNTS (wall-seconds vs core-seconds on a 4-core box),
    which changes what the tenant protection protects. Any answer other
    than "wall clock stands" RELAXES a protection on a box with paying
    tenants — that is a threshold question, owner-gated, and no default
    here may fire it (SYSTEM.md law 4; the T0.33 ceiling comment already
    binds `cpu_foreclosed == []` to the current arithmetic).
    Options, priced: (i) wall clock stands — then a cpu<48h dispatch is
    accepted as buying N foreclosed days, and the honest repair is
    SCHEDULING (the queue plans around it; `run status`'s new CPU DAY
    BUDGET block makes the foreclosure visible while it happens); (ii)
    core-seconds against 4 cores (owner-only) — a single-core child then
    bills ~21600 core-s into a 230400 core-s day and coexists with the
    runner lane; needs per-process CPU-time sampling, not wall clock, or a
    declared cores-occupied factor; (iii) a separate detached-lane
    sub-ceiling so one lane cannot exhaust the other's allowance — still a
    threshold edit, still owner-gated.
    Instruments already in place feeding this row: T0.33's
    `n_foreclosed_now` metric and `run status`'s live unaffordable-set
    print (68th audit B3/B4, landed 2026-09-04) — a foreclosed day now
    reports itself, so the decision can be made against observed
    foreclosure counts rather than the worst-case arithmetic alone.
    SEMANTIC bill: none — no committed row claims anything about cpu<48h
    scheduling; T0.33/T0.34's certificates gate ADMISSION and DISJOINTNESS,
    not the unit the ceiling counts.
    MECHANICAL bill: whichever option lands edits `experiments/cpu_budget.py`
    (and (ii) also `experiments/rtf.py`) — T0.33 and T0.34 both cite
    cpu_budget.py via IMPL_DEPS and re-buy at ~2 s and ~60 s respectively;
    option (ii) additionally stales every certificate citing rtf.py.

    AMENDED 2026-09-04 (builder, 69th audit B4) — the RUNNER lane forecloses
    by the same arithmetic, and after B4 it is the ONLY foreclosure left.
    Added as evidence to this row rather than routed as a new one: it is the
    same constant and the same owner question ("what should the CPU
    day-ceiling be, and count?"), and the 09-06/09-08 docket does not need a
    fourth pass. B4 replaced the admission estimate with the spec's own
    measured cost and the live foreclosure count fell **53 -> 36** — but the
    certificate's new `n_foreclosed_unmeasured` reads **36**, i.e. ALL of the
    residual are specs that have never run and so have nothing to project
    from. The arithmetic, and it is a pure inequality with no measurement in
    it: `CPU_DAY_CEILING_S` (57600 s) is **1.067x** the largest legal child
    (`cpu<2h` x 3 seeds x 2 = 54000 s), so a never-run `cpu<2h` spec is
    refused once the day passes **3600 s — 6.25% of the ceiling**, which one
    routine gate sweep spends. Why it matters to the DATE rather than
    someday: a never-run spec is exactly what the 09-06 Review orders when it
    resolves a redesign, so this fires on the first morning of the work the
    Review is about to commission, not on maintenance. Why it is not a
    builder fix: every repair raises or splits a tenant-protection ceiling
    (SYSTEM.md law 4), and the honest cheap alternative — projecting a
    never-run spec from a CLASS prior over its budget-mates — is a genuine
    estimator design with its own bakeoff, not a constant edit. Option (iv)
    for the menu above, priced with the others: **schedule rather than
    raise** — the loop runs first-run `cpu<2h` specs before its own
    housekeeping, which costs nothing and loosens nothing, and is the runner-
    lane twin of option (i)'s answer for the detached lane.

    DUE: 2026-09-19 | ROUTING CONSEQUENCE DELIVERED 2026-09-08 (below); the
        UNIT question is untouched and stays armed on `D20` (decide_by
        2026-09-18). Re-dated to the day AFTER that deadline so this row
        consumes either the owner's answer or `D20`'s armed default, instead
        of asking the same question a third time in front of it. What is owed
        on 09-19 is the consequence of whichever way `D20` lands.
    DUE: 2026-09-30 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. The ROUTING CONSEQUENCE was executed on 09-08 and stands. What is still live is the UNIT question, which is armed on `D20` and is now the subject of `D32` (one sentence this repository reads two opposite ways). This desk may not pre-empt an open decision, so the date tracks the sitting after D32 is expected to land.

**DISPOSITION (Review DAILY, 2026-09-08): (i)+(iv) — schedule around it. No
ceiling is raised, split, or re-unitised by this desk.**

**Taking the row's own instruction rather than improvising one.** This row was
dated 09-08 with an explicit conditional: *if the owner answers before 09-08,
consume the answer; if not, decide the routing consequence only and leave the
unit question armed.* `D20` is live with `decide_by 2026-09-18` and unanswered.
So the conditional resolves to its second branch and this disposition
deliberately does less than the row's menu offers.

**Options (ii) and (iii) are not mine to take and I am not taking them.** Both
edit what a tenant-protection ceiling counts or how it is partitioned, on a box
with paying tenants. `SYSTEM.md` law 4 makes that owner-gated, the row says so
itself, and no default here may fire it. Anything other than "wall clock
stands" RELAXES a protection — that is the whole reason this is a desk row and
not a builder fix, and it would remain the reason even if I found the arithmetic
persuasive, which for the record I do.

**ADOPTED, both halves, because they are the same answer for two lanes:**

  - **(i) for the detached lane.** Wall clock stands. A `cpu<48h` dispatch is
    accepted as *buying N foreclosed days*, and that is planned for rather than
    argued with. The instruments to plan against already exist and were built
    for this: `T0.33`'s `n_foreclosed_now` and `run status`'s live
    unaffordable-set print. A foreclosure that reports itself while it happens
    is a schedulable cost.
  - **(iv) for the runner lane.** The loop runs first-run `cpu<2h` specs before
    its own housekeeping. This matters on the date rather than someday for the
    reason B4 gave: `CPU_DAY_CEILING_S` is only **1.067×** the largest legal
    child, so a never-run `cpu<2h` spec is refused once the day passes **3600 s
    — 6.25% of the ceiling**, which one routine gate sweep spends. A never-run
    spec is exactly what today's dispositions commission (`UB.10`'s hardened
    battery, `D1.0`'s twin probe, `LG.10`'s sibling), so this fires on the first
    morning of the work this page just ordered.

**Why this is the honest small answer and not a dodge.** (i)+(iv) are the only
two options on the menu that **cost nothing and loosen nothing**: they move no
constant, edit no ceiling, stale no certificate, and change what no instrument
counts. They are pure scheduling. If they turn out to be sufficient, the owner's
question becomes cheaper to answer rather than more urgent; if they turn out to
be insufficient, `D20` will be answered by then and the expensive options are
still on the table with a measured foreclosure count behind them instead of
worst-case arithmetic. Deciding a threshold question early, in the absence of an
answer, on a box with tenants, to save myself a second sitting, is not a trade
this desk should make.

**Bills, restated so 09-19 does not have to re-derive them:** SEMANTIC none.
MECHANICAL — (i)+(iv) as adopted touch no ceiling constant and so bill nothing;
it is (ii)/(iii) that would edit `experiments/cpu_budget.py` (T0.33 ~2 s, T0.34
~60 s re-buys) and (ii) that would additionally stale every certificate citing
`rtf.py`. That asymmetry is itself an argument for scheduling first.

## ROUTED 2026-09-04 (builder, LG.03 attempt-1 harvest): `lg03-blind-twin-cannot-prove-itself-alive` — the certifier VOIDs on its own liveness gate, and the repair it pre-registered is falsified

ROUTED: lg03-blind-twin-cannot-prove-itself-alive | 2026-09-04 | LG.03-attempt-1 | ACTED 2026-09-24 (Review DAILY, executing commit `1bd42dc` with its two doc-only amends `75a5544` and `927d9f4` — the 09-12 ruling ordered FOUR things and all four are on disk and on the ledger, verified item by item rather than taken from the builder's journal: the constant `PLANNER_CALIB_MIN = 1.0` is pre-registered in `experiments/tests/lg_03_command_cells_necessary.py`, `planner_calib_reach` is a first-class emitted metric reading **0.8333 ± 0.1179** on the LG.03 row, the new conjunct is ordered BEFORE the twin's reading, and attempt 2 re-ran on CPU at 2026-09-12T18:23:20 in 727.18 s over 3 seeds. The gate fired exactly where the ruling said it would. TWO THINGS THE STAMP MUST NOT SWALLOW. First, **the ordered measurement REFUTED the ruling's own cap MECHANISM** — the per-seed join (teacher 1.00/0.75/0.75 vs twin 0.50/1.00/0.25) shows `planner_calib_reach` is not a ceiling on `blind_calib_rate`; the repair survives, the explanation under it does not. That refutation is NOT orphaned by this stamp: it is the live row `lg03-teacher-does-not-cap-the-twin` (routed 09-12, DUE 2026-10-04), which is why this one can close. Second, the 90th audit's B1 amendment declared the result **VOID-FORECLOSED** — attempts 1 and 2 share 33 metrics differing in no digit, both gates shut on the mean, and seed 0, the only seed surviving gate 1, reads twin 0.50 against `CALIB_MIN` 0.75 — so auditing the venue does NOT deliver the liveness proof, and LG.03's FAIL remains UNBOUGHT. ACTED here means the execution this row's `DUE:` bought was delivered, not that LG.03 is alive. Earlier: DISPOSITIONED 2026-09-12 (Review DAILY — RULED: none of the four options as written. The gate's own maximum achievable value is `planner_own`, not 1.0, because the calibration tape is recorded from the privileged planner's MISSES as well as its hits; the repair is a NEW conjunct on the teacher, with CALIB_MIN untouched. See RULING below)
    DUE: 2026-09-12 | a liveness-gate redesign owed by the Review. Deliberately NOT 09-06/09-07: `review-queue` names 09-12 as the next date carrying no promise, and this row has no money and no clock on it — nothing expires and no quota dies while it waits. Coupled to `champions-language-grounding-arena` (DUE 09-07) as an INPUT, not a decision beside it: that row asks whether the language-grounding seat has an arena at all, and the answer is now "it has one, registered, and its certifier cannot yet certify itself".
    DUE: 2026-09-23 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE. CAUSE, MEASURED AND NOT GUESSED: the builder has been PACE-DARK for 18 consecutive hourly slots since 2026-09-14T12:07 — `week:all models` 37% against a pace line of 35% at 15% of the week, with this project's own attributed share of that meter measured at 25% (builder 7 + desks 2 of 37 points; 28 points, 75%, NOT THIS PROJECT). A row owed by the BUILDER cannot be honestly dated inside a week the builder cannot run in, so this date is set AFTER the 2026-09-21 05:00 UTC meter reset rather than onto another dark day. Routed to the owner as `D30`. SECOND break for this row (09-12 -> 09-14 -> now). This row is the BUILDER's execution debt — the liveness-gate design was DELIVERED 2026-09-12 and what is owed is implementation. ORIGINAL TEXT FOLLOWS, unchanged. | DISPOSITIONED 2026-09-12 (Review DAILY): the design is
        RULED and lives in the RULING block at the end of this row. What is
        owed now is EXECUTION by the builder — one constant
        (`PLANNER_CALIB_MIN = 1.0`), one emitted metric
        (`planner_calib_reach`), one VOID conjunct ordered BEFORE the twin's
        reading, then a re-run (CPU, ~725 s, 3 seeds). Dated 09-14, not 09-13:
        09-13 carries 14 rows against a measured capacity of 6, and 09-14 is
        `review-queue`'s own `next_free_due`. The row stays LIVE and keeps
        ageing until a commit stamps it ACTED.

**What was measured (LG.03 attempt 1, VOID 2026-09-04T17:20:27, 724.6 s,
3 seeds; metrics on the ledger row, correct and untouched).**
`blind_calib_rate` **0.583 +- 0.312** — readings 1.00 / 0.50 / 0.25 across
seeds 0/1/2 — against a pre-registered `CALIB_MIN` of 0.75. The blind twin
could not demonstrably reproduce, from the identical starts, demonstrations it
was trained on. Its at-chance readings elsewhere therefore prove nothing, and
`_check` correctly returned VOID before reading a single claim number.

**Everything else in the rig was green, which is what makes this a gate
question and not a bug hunt.** `obs_finite` 1.0; `verb_alive_min` 1.0 (the
privileged planner can perform every verb on some object); `planner_reach_mean`
0.754; and the declared control — the stripped planner, target identity
withheld — FAILED as it must, `stripped_both_rate` 0.0542 against `CTRL_MAX`
0.10, measured on all 80 candidate cells rather than only the survivors.

**The claim numbers the VOID refuses to publish, quoted here so the Review can
see what is at stake:** `retained_cells` 6.33 +- 0.47 against `MIN_CELLS` 12,
`cellset_ok` 0.0 on **every** seed, a cross-seed intersection of three cells
(`approach@block`, `round@stairs`, `touch@block`), `min_per_verb` 1,
`min_per_object` 1. Seed 0's 13 exclusions split 3 blind-above-chance / 5
unachievable / 5 no-plurality. If the instrument is repaired, this fixture
looks likely to deliver a **FAIL** — which per its own registry text is a
VENUE verdict ("this world does not admit language-necessary commands at this
horizon") routing to `w0-too-shallow` as a further instrument, and would make
`LG.04`/`LG.05`/`LG.06` VENUE-blocked rather than model-blocked. **That verdict
is currently UNBOUGHT, not delivered.** Note the direction of the coupling
before weighing any repair: a STRONGER blind twin excludes MORE cells, so every
repair that helps the instrument prove itself alive also pushes the claim
further from PASS. There is no run-until-pass move available on this row.

**The repair this file pre-registered is FALSIFIED, and that is the finding
this row is worth.** The journal committed in the open, the day before the run:
*"if it fires, the repair is a third learner in the `max`, never a lower
`CALIB_MIN`."* It fired. `experiments/tests/lg03_blind_twin_probe.py` (kept,
reproduces on demand) then raced five cheap deterministic learners on the
identical demos of the identical calibration cell (`approach@block`, 96 rows):

    seed  planner_own  knn   ridge  knn1  wknn  ridge_lo | max2  max5
    0        1.00      0.25  0.75   0.75  0.50  0.75     | 0.75  0.75
    1        0.75      0.00  0.50   0.50  0.00  0.50     | 0.50  0.50
    2        0.75      0.50  0.75   0.50  0.50  0.50     | 0.75  0.75

`max5 == max2` on **every seed** — including `wknn`, a Euclidean metric
reweighted by ridge-coefficient magnitude, aimed squarely at the placebo and
near-silent columns that were the diagnosed reason for the shipped pair. A
third seat in the `max` buys exactly zero.

**The cause is the `planner_own` column, and it is a design defect in the gate
rather than in the learner family.** On seeds 1 and 2 the PRIVILEGED planner —
told the target, reading the object's world coordinates — reaches the
calibration cell on only **3 of its own 4 starts**. The demonstrations are
capped by the teacher while `CALIB_MIN` is absolute, so on those seeds the gate
demands the student reproduce *every one* of the teacher's successes perfectly
or the run is void. The parent file's own `avoid` predicate already knows this
shape — it is start-relative *"because an absolute one is satisfied by standing
still far away"* — and the liveness bar is the same mistake one surface over.

**Why this is not a one-line fix, and why the builder did not make it.**
Relativising the bar to the teacher (0.75 x 0.75 = 0.5625) still VOIDs seed 1,
whose best clone reads 0.50. So both halves are live, and `CALIB_MIN` is a
pre-registered constant on a spec that now carries a ledger row — moving it, or
its semantics, after seeing the failure is precisely the act that needs an eye
that is not the author's (`D22` is on the owner's desk asking whether that
changes; until it resolves, the default is that it does not). **Nothing has
moved:** `CALIB_MIN` unchanged, `_Blind.KINDS` unchanged, both file edits are
docstring prose re-stamped through the `--doc-only` amend lane, and the
recorded VOID stands per the T2.02 precedent.

**Options to weigh, none decided here.**
  (i)   **Teacher-relative liveness** — the clone must reach `CALIB_MIN` x the
        privileged planner's own rate on the identical starts. Repairs the
        measured defect; does not clear seed 1 alone. Loosens the absolute
        floor, which is the reason it needs ratification.
  (ii)  **Calibrate where the teacher is perfect** — choose the calibration
        cell by the planner's own reach rather than fixing it at
        `approach@<first object alphabetically>`. Keeps `CALIB_MIN` absolute
        and untouched; costs nothing; but "select the venue for a liveness
        proof" is one short step from "search cells until the gate passes", so
        the selection rule must be declared before the run and be blind to the
        twin's reading.
  (iii) **A twin with an optimiser** — the file deliberately excluded one so
        that "nothing about the verdict depends on a training schedule". That
        purchase is now measurably expensive: no member of the cheap
        deterministic family demonstrates itself alive in W0's 80-number
        observation at this horizon on all three seeds. Buying a schedule buys
        a stronger null, which is the strengthening direction for the claim.
  (iv)  **Read the VOID as a venue instrument in its own right.** A policy
        trained on the world's own observation, from a privileged servo's
        demonstrations, reproducing the demonstrated act on half the starts is
        a statement about what W0's observation supports — arguably a further
        `w0-too-shallow` reading, pointed at the OBSERVATION rather than at the
        world's depth. If the Review takes this branch, the gate redesign
        follows the W1 decision rather than preceding it.

**Cost of waiting, stated plainly:** `LG.04`/`LG.05`/`LG.06` all
`depends_on: LG.03`, so `protocol.blocked_by()` structurally holds the whole
grounding bakeoff behind this row — which is the ordering working exactly as
designed, not a fault. None of the three is implemented, so nothing is idle
that would otherwise be running.

**RULING, 2026-09-12 (Review, DAILY). None of (i)–(iv) as written. The defect
is one line upstream of all four, it is algebraic rather than interpretive, and
the run ALREADY MEASURES the quantity that invalidates its own gate and throws
it away.**

Read from source, not from this row's prose
(`experiments/tests/lg_03_command_cells_necessary.py`):

```
502-508   for vv, a in rec:  demo_X.append(vv); demo_Y.append(a)
          if c == calib_cell:  for vv, a in rec: calib_X.append(vv); ...
```

`rec` is the tape of the privileged planner's rollout, and it is appended
**unconditionally — there is no `if hit`**. On a start where the servo missed,
the twin is trained to imitate a MISS. Then:

```
517-521   hits = [_satisfies("approach", _rollout(w, st, calib.policy(kind), ...))
                  for st in starts[calib_cell]]
          calib_rate = max(calib_rate, mean(hits))
671       if m["blind_calib_rate"] < CALIB_MIN: return Status.VOID
```

the twin is scored on **task success over all four starts**, against an
**absolute** bar.

> **Perfect reproduction of the training tape therefore scores `planner_own`,
> not 1.0.** The gate's stated meaning is *reproduction fidelity* ("reproduce,
> from the identical starts, demonstrations it was trained on", line 76); the
> quantity it computes is `fidelity × teacher competence`. It compares that
> product against a bar calibrated as though it were fidelity alone. With
> `planner_own` = 1.00 / 0.75 / 0.75 and `CALIB_MIN` = 0.75, seeds 1 and 2 have
> **exactly zero margin** — the gate is clearable there only by FLAWLESS
> imitation, and on any seed where the servo read below 0.75 it would be
> **un-clearable by construction**. That is the `PL.02` subtrahend shape
> (ruled 09-11) one spec over: a guard that algebraically suppresses the thing
> it was added to watch.

**The sharpest fact, and the reason this is a repair and not a redesign:
`own_hit[calib_cell]` is already computed on line 501 and simply never read.**
The run measures the teacher's reach at the calibration cell, reports only its
mean over ALL cells (`planner_reach_mean` 0.754), and the gate that the number
invalidates never consults it.

**THE REPAIR — strictly a TIGHTENING, and it is the only one of the five that
is.**

1. `CALIB_MIN` stays **0.75, absolute, same semantics, not re-based, not
   relativised**. It is not touched in either direction.
2. Emit `planner_calib_reach = mean(own_hit[calib_cell])` as a first-class
   metric — a number the run already has.
3. **ADD a VOID conjunct** with a new pre-registered constant
   `PLANNER_CALIB_MIN = 1.0`: if the privileged planner does not reach the
   calibration cell from **every one of its own starts**, the run is VOID for
   an INSTRUMENT reason — the liveness venue is invalid — checked *before*
   `blind_calib_rate` is read. Order matters: the teacher is indicted first.
4. The calibration cell stays `approach@sorted(objs)[0]`, **declared, fixed and
   blind to the twin's reading.** This is what separates the ruling from option
   (ii): the venue is not selected, it is *audited*.
5. `_Blind.KINDS` unchanged; no third learner. The pre-registered repair stays
   falsified and that finding stands.

**Why each of the four options was refused, on the record:**
  (i) **Teacher-relative liveness is a LOOSENING** (0.75 × 0.75 = 0.5625 < 0.75)
      of the alive-proof of a CONTROL, and the one law binds: I may strengthen,
      never weaken. It also mis-locates the fault — the teacher's incompetence
      becomes a discount the twin gets to keep, when it should be a reason to
      refuse the measurement outright.
  (ii) **Venue selection**, and this row's own author named the hazard
      correctly: one step from searching cells until the gate passes. Conjunct 3
      gets (ii)'s entire benefit — a calibration cell where the teacher is
      perfect — without buying the hazard, because a cell that fails the audit
      VOIDs the run instead of being swapped for a better one.
  (iii) **An optimiser in the twin buys a stronger null and is the strengthening
      direction — but it does not touch this defect.** Under the repaired gate
      a twin with a training schedule is still scored as `fidelity × teacher`,
      so it would VOID on seeds 1 and 2 for the same reason. It is a legitimate
      SEPARATE strengthening and it is NOT ordered here, because ordering it
      now would spend a training schedule on a gate that is still mis-posed.
  (iv) **Reading the VOID as a venue instrument is RIGHT, and this ruling
      composes with it rather than choosing against it** — see the cost below.

**THE COST, STATED BECAUSE IT IS THE EXPENSIVE HALF.** Under this repair
`LG.03` in W0 as built VOIDs on **2 of 3 seeds** — more often than today, not
less, and for a reason it can name. The spec cannot deliver its FAIL (the
VENUE verdict quoted above) until the fixture admits a calibration cell the
privileged servo aces. **That is option (iv)'s reading, arrived at from the
gate rather than asserted about the world**, and it is a further
`w0-too-shallow` instrument pointed at the OBSERVATION. `LG.04`/`LG.05`/`LG.06`
stay structurally blocked; none is implemented, so nothing idles.

**Nothing weakened:** no threshold moves, `CALIB_MIN`/`_Blind.KINDS`/
`CHANCE_HI`/`CTRL_MAX` untouched, the stripped-planner control untouched, the
recorded VOID stands per the `T2.02` precedent, and the change is MONOTONE — it
can only VOID runs that pass today, never pass runs that VOID today.
**SEMANTIC bill: none** (`LG.03` has no PASS row; its only row is the VOID,
which stands as history). **MECHANICAL bill: none outside `LG.03`'s own
re-run** — no other certificate cites this file.

ROUTED: xl01-death-and-retry-has-no-reachable-repair-path | 2026-09-05 | 72nd-audit-B4 (FAIL-UNOWNED, 6fbac74) | OPEN
    DUE: 2026-09-25 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | a reachable repair path for the death-and-retry commitment — the question is "what buys it one", NOT "re-run XL.01". Date is `next_free_due` per B4, not Sunday.

**The claim and the silence:** `XL.01` — *"Death does not erase what he
learned"*, filed by its own `COVERS:` under both **death & retry** and
**memory across lives** — has read FAIL since **2026-08-19**, with no
`repaired_by`, no queue row and no disposition, for 17 days. `coverage`
printed the commitment as `1 now` throughout; the 72nd audit's lesson ("a
negative with no owner is invisible") is this row's provenance.

**The diagnosis, which was CORRECT and changed nothing:** a POWER failure —
XL.01 cannot resolve a 2× effect at 3 seeds × 8 lives (its B3 verdict). It
was carried into `NE.08`'s registry notes as a BINDING pre-run power
calculation (INTEGRATION_QUEUE.md, NEEDS_AND_DEATH row) — excellent filing,
zero routing: `NE.08` is `blocked<-NE.01`, and `NE.01` is itself a settled
FAIL (`ne01-occlusion-knife-edge`, this queue). A repair path that runs
through two failures exists in prose only; no field in this repository can
express it, so no tool can rank it.

**What the Review owns here:** either `NE.01`'s occlusion redesign (already
on this desk) is the single upstream unblock and this row is decided WITH it
— in which case say so and couple them — or death-and-retry needs a claim
that is powerable at free-tier seat counts (more lives per seed is CPU-cheap;
XL.01's own row shows lives, not seeds, carry the variance). Strengthen-only
binds any successor; XL.01's FAIL stands as history either way.

ROUTED: t205-world-model-loses-to-the-ridge-reference | 2026-09-05 | 72nd-audit-B4 (FAIL-UNOWNED, 6fbac74) | OPEN
    DUE: 2026-10-04 | RE-DATED 2026-09-24 (Review DAILY). The 2026-09-23 date BROKE — second break. This row is NOT part of the W1 docket and must not inherit its excuse: what it owes is a disposition of the fast/slow world-model fixture — *what does the DP family require of a world model that beats every null but loses to a linear probe?* — and no world edit, no registration and no GPU hour gates it. The honest cause of this break is therefore the plainest one available and it is this desk's alone: **at a DEMONSTRATED disposal rate of ~1 row per sitting, three Review-owned designs cannot share one morning, and this one ranked third of three.** It is dated LAST of today's three for a stated reason rather than by accident — `w1-world-edit-window` blocks five other rows and `w0-too-shallow` holds the project's largest standing result, while this one blocks a FAIL-UNOWNED diagnosis and nothing downstream of it. Dated 2026-10-04 (2 promised against a measured 6), one sitting clear of 10-01 so the two are not made to share a morning the way these three just were. ORIGINAL TEXT FOLLOWS, unchanged. | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | a disposition for the fast/slow world-model fixture: what does the DP family require of a world model that beats every null but loses to a linear probe? Date is `next_free_due` per B4.

**Measured, twice, consistently.** `T2.05` v1 (2026-08-14) VOIDed itself
honestly — the persistence ruler leaked marginal statistics (shuffled
control beat it) — and its redesign facts pre-registered the reference arm.
v2 ran under the strengthened rig and **FAILed 2026-08-20**: `wm_ratio_max` 0.2529 (the WM
beats the informed null on every seed, 4× margin) but `wm_beats_ridge_all`
**0.0** — the learned world model loses to ridge regression on the same
prediction task. 16 days, no owner, no row, until now.

**Why it needs a desk and not a re-run:** the fixture measures exactly what
it was strengthened to measure, and what it says is that our world model is
worse than a linear map at 5-step prediction in this venue. The DP family's
imagination arms and `w0-too-shallow`'s venue question both lean on world
models; whether the repair is a better WM, a harder venue, or accepting the
linear reference as the champion (SCORED-AND-INELIGIBLE idiom) is a design
call. The bar does not move; T1.02 precedent binds.

ROUTED: t402-touch-drowns-audio-at-the-fusion-boundary | 2026-09-05 | 72nd-audit-B4 (FAIL-UNOWNED, 6fbac74) | DISPOSITIONED
    DUE: 2026-09-22 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | the fusion-balancing redesign the priority head has pointed at "the Review, not an argument" since 08-21 — now with a row and a clock instead of a standing sentence. Date is `next_free_due` per B4.
    DUE: 2026-09-25 | RE-DATED 2026-09-23 (Review DAILY) under D28's (a)
        OVERDUE FIRST. **The debt this date carried — the bakeoff DESIGN — is
        discharged below, in this sitting.** The new date carries the RUN,
        which is the builder's. **The date is derived from a PERISHABLE
        RESOURCE and not from calendar room, and I am saying so because it
        knowingly piles on**: `2026-W38` holds 30 free Kaggle GPU-hours of
        which **0.4789 h** is drawn; **~29.5 h expire on Saturday 2026-09-26**
        and `T2.06`'s re-buy was the only legal buyer this project had. The
        three-arm bakeoff below is `GPU_SHORT` — `T4.02` attempt 4 ran in
        **514.69 s ≈ 0.14 h**, so three arms at three seeds is **~0.45 h** —
        and it is the first legal buyer for those hours that this desk has
        been able to produce in three weeks. 09-25 carries 6 rows against a
        measured capacity of 6, so this raises `review_queue_piled_on` 1 -> 2.
        **That is a deliberate act with a named reason, reported on today's
        page rather than left for an audit to find**: a date derived from an
        expiry beats a date derived from an empty square when the resource
        dies on Saturday. 09-25 and not 09-26 so a one-day slip does not burn
        the hours.

**Measured, twice (`T4.02` attempts 3 and 4, 2026-08-21).** Worst-seed
`max_modality_grad_ratio` **30.12** against the exogenous 10× gate, zero
variance across seeds; touch (~2.9e-3) dominates audio (~1e-4) at the fusion
boundary; every rig gate green (`finite_all`, `fired_ok_all`,
`loss_decreased_all` 1.0). This is GOAL.md's stage 4 verbatim — *"no
modality collapse"* — settled FAIL for 15 days with no owner while CLAUDE.md
carried a prose prohibition on re-dispatching it as cheap work.

**What the Review owns here:** the architecture measurement is real (a
gradient-scale imbalance of 30× is not a seed lottery at std 3.6e-15). The
candidate repairs are runnable arms — per-modality gradient normalisation,
loss reweighting, modality dropout schedules — so this is a bakeoff to
design, not an argument to have. `UB.10`'s recipe-sensitivity finding
(`recipe-sensitivity`, this queue) is adjacent: both say uniform training
treats unequal senses unequally. Couple them if one design answers both.

    **THE DISPOSITION — 2026-09-23 (Review DAILY). The bakeoff is designed,
    the gate does NOT move, and the arm that would win it trivially is
    disqualified in the design rather than after the run.**

    **THE INCUMBENT IS REAL AND ALREADY MEASURED.** `T4.02` attempt 4 is the
    DEFAULT arm and it carries a number, not an impression:
    `max_modality_grad_ratio` **30.1197**, `std` **3.55e-15** across seeds —
    zero variance, so this is architecture and not a seed lottery. Touch
    (~2.9e-3) over audio (~1e-4) at the `CrossModalFusion` boundary, against
    the registry's own exogenous **10×** gate written 2026-08-04. **That 10×
    does not move in any arm, in either direction.** Every arm runs `T4.02`'s
    SHIPPED rig unchanged — same fixture, same equal-variance latents, same
    hooks, same learning gate, same grad-scale control — and is read by
    `T4.02`'s own metric. An arm that needs the rig changed to look good is
    not an arm.

    **THE DESIGN'S ONE REAL DECISION, and it is a disqualification.**
    Arm (a), per-modality gradient normalisation, **equalises
    `max_modality_grad_ratio` BY CONSTRUCTION.** It cannot fail the stated
    metric. Certifying it on that metric alone would be the purest Goodhart
    this ladder has ever been offered: we would buy a green tick for an
    arm whose mechanism IS the measurement. So the bakeoff carries a SECOND,
    STRICTLY HARDER conjunct that no arm can satisfy by construction, and it
    comes from `T4.02`'s own docstring rather than from my taste: *"a sense
    whose fusion token is ignored is a sense the other senses cannot teach
    (GOAL.md: what he hears must be able to teach what he sees)"*.

    **THE ADDED CONJUNCT — `min_modality_latent_r2`.** The fixture already
    makes this free: each modality carries an independent k=8 latent `z_m`,
    and `target_actions` is a sum of per-dim STANDARDISED readouts of the five
    `z_m` divided by sqrt(5), so every sense contributes an equal ~1/5
    variance share BY CONSTRUCTION. After training, probe each `z_m` from the
    FUSED representation on held-out draws and report per-modality recovery;
    the statistic is the **WORST** modality's R², per seed, gated at the
    minimum over seeds — `T4.02`'s own "report per partition, gate the
    minimum" discipline, unchanged. **The bar is the INCUMBENT's own measured
    worst-modality recovery, and an arm must EXCEED it.** That number does not
    exist yet, so the incumbent is re-run as the bakeoff's first arm to
    establish it; it is pre-registered before any arm's number is seen, and it
    is not a threshold this desk chose — it is whatever the shipped brain
    already achieves. **An arm that equalises the gradient ratio while leaving
    the worst sense's latent recovery at or below the incumbent's is
    REFUTED — it moved the bookkeeping and not the creature.** That is the
    sentence the whole design exists for, and it is strictly harder than the
    row's original ask, which was the 10× gate alone.

    **THE THREE ARMS, unchanged from the row's own naming (this desk is
    designing the contest, not picking the winner).**
      (a) **per-modality gradient normalisation** at the fusion boundary. The
          arm most likely to pass the ratio and fail the recovery conjunct.
          That outcome is a RESULT, not a failure of the bakeoff: it would
          say the imbalance is a symptom and normalising it treats the
          symptom.
      (b) **loss reweighting** — per-modality weights on the shipped
          `action_training_loss`. Weights must be set by a DECLARED rule
          (e.g. inverse of the incumbent's measured boundary norms, computed
          once from attempt 4's numbers and frozen), never tuned against the
          gate. A weight tuned until the ratio clears is threshold-moving
          wearing a different hat.
      (c) **modality dropout schedule** — each sense's fusion token randomly
          masked during training, so no sense can be relied on. The only arm
          of the three whose mechanism does not mention the measured quantity
          at all, which makes it the cleanest test of whether the metric
          tracks anything.

    **VOID LANES, inherited and NOT relaxed.** `T4.02`'s learning gate (mean
    loss over the last quarter below the first, every seed), its
    `fired_ok` hook-attachment assertion (2 fires/step), its rig-health share
    gate (each modality's realised variance share in [0.10, 0.30]), and its
    grad-scale control are all conditions on every arm. **One addition:** an
    arm whose final loss is WORSE than the incumbent's does not get to win on
    balance — balance bought by breaking the task is not balance. Report it;
    do not certify it.

    **THE UB.10 COUPLING: REFUSED, with the reason, because the row asked.**
    `recipe-sensitivity` and this row do rhyme — both say uniform training
    treats unequal senses unequally — but `UB.10` is **VOID** on attempt 1 and
    its Part-1 premise is itself under challenge in this queue
    (`ub10-part1-premise-false-marginals-are-what-saturate`, OPEN). Coupling a
    designed, runnable, GPU_SHORT bakeoff to a VOID spec whose premise has an
    open row would make this design inherit that row's clock, and this row has
    already broken twice waiting for things. They stay separate. **If (c)
    wins here, that result is an INPUT to `UB.10`'s successor arm and should
    be cited there** — a one-directional citation costs nothing and creates no
    dependency.

    **STALENESS BILL: ZERO.** Nothing above edits `T4.02` or any shipped
    module; the arms are new spec(s) registered beside it, and `T4.02`'s FAIL
    stands as a true measurement until an arm returns under the declared
    shape. No certificate anywhere hashes a file this disposition touches.

ROUTED: t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall | 2026-09-05 | 72nd-audit-B4 + builder (the FAIL-UNOWNED detector's 4th member — the audit's own count missed it) | OPEN
    DUE: 2026-09-24 | RE-DATED 2026-09-14 (Review DAILY). The 2026-09-13 date BROKE — one of THIRTEEN that broke together at midnight, the project's first queue violations (`review_queue_violations` 0 -> 13, a ratchet that had read 0 since 09-03). Re-armed in the open at the desk's DEMONSTRATED disposal rate (~1/cycle), NOT at its measured maximum (6/cycle), and never onto a day already carrying its capacity — promising six a day is the act that built the pile. This flattens the pile; it does not fix the drain, which is `D28`'s. ORIGINAL TEXT FOLLOWS, unchanged. | a disposition for T2.15's FAIL: route the memorisation-route finding somewhere an instrument can see it, or dispose it explicitly. Date is `next_free_due` per B4.

**Why this row exists at all:** the 72nd audit measured FAIL-UNOWNED at 3;
the detector built to its own B1 conjunction finds **4**. `T2.15`
(*free-form language routes to the right task*) FAILed **2026-08-25** —
`heldout_correct` [8, 9, 5], min 5, under its pre-registered bar, with every
rig gate green (`construction_ok`, `det_ok_all`, `loss_fell_all` 1.0) — and
its diagnosis lives ONLY in `SM.03`'s registry notes (*"held-out layouts
close the memorisation route T2.15 just measured on language"*). `SM.03` is
PILOT-BLOCKED. By the audit's own lesson, a successor's notes are not
routing when the successor is unreachable: same shape as XL.01→NE.08, one
family over, and invisible to the audit's own scan for the same reason.

**What the Review owns here:** small. Either T2.15's finding folds into the
language-grounding design already owed (`champions-language-grounding-arena`,
DUE 09-07 — the LG family is the successor venue for language claims), in
which case write the coupling and a `FAIL-DISPOSED:` marker on T2.15 naming
that decision, or it needs its own successor spec. Nothing here re-runs.

ROUTED: so07-recording-worlds-fail-the-reference-bar | 2026-09-05 | builder (SO.07 attempt-1 harvest, 9bd3114) | OPEN
    DUE: 2026-09-18 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE — FIRST break for this row, and it is this desk's own decision debt, not the builder's. Re-dated ONCE at the desk's DEMONSTRATED disposal rate (~1/cycle), onto a date with measured room under the 6/day capacity, never onto a day already at it. The Review is NOT pace-gated (its 06:37 slot is exempt), so unlike the builder-execution rows in this batch, this desk has no excuse available to it and is not offering one. ORIGINAL TEXT FOLLOWS, unchanged. | a disposition for SO.07's VOID: what re-validates the reference arm on the recording worlds — re-frozen fixture, a wider design-world set, or a world/body redesign. Date is `next_free_due` per the router's own print (every earlier day is at or over measured capacity).
    DUE: 2026-09-26 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. SO.07's VOID has no repair path until this fork is taken, and the row has now broken twice; the 09-15 re-date said plainly that this desk 'has no excuse available to it and is not offering one'. That sentence stands and is not repeated as if it were new information.

**The verdict:** SO.07 attempt 1 (2026-09-05T02:41:48, 9201.5 s, worlds 3/4/5)
recorded **VOID at the first pre-registered VOID lane**: the REF arm —
privileged food positions, `REF_MIN_FED = 0.8` reused verbatim from XL.01 —
fed **0.333 / 0.667 / 0.667** of lives on the three recording worlds. The rig
could not demonstrate, with every advantage, the behaviour the spec measures,
so per its own gates table no other number means anything. This is the lane
doing its job (SH.01's lesson, cited in the docstring), not a harness fault.

**The diagnosis, one sentence:** fixture constants were frozen from an
aliveness pilot on design world 0, and privileged-position feeding did not
transfer to recording worlds 3–5 — the design/data world split (correct, and
kept) means the fixture was never validated where the claim runs. The repair
choices are exactly the ones the split forbids the builder from taking
unilaterally: (a) re-freeze from counts on a wider design-world set and state
why worlds 3–5 should now transfer, (b) redesign the REF arm so its bar is
world-relative rather than XL.01's absolute 0.8, stated as HARDER not weaker,
or (c) file this as a further `w0-too-shallow` instrument (adjacent to
`ba03-null-saturates-the-horizon` and SH.01's ORACLE_CANNOT — a body that
cannot reach privileged food on 2 of 3 worlds is the same family) and let the
W1 decision own it. Option (c) costs nothing now: the row deliberately does
NOT declare `BLOCKED-BY: w0-too-shallow` because that design is due 09-06 and
this row's date is after it — the Review can couple them with the decision in
hand.

**Reported alongside, because it scopes the redesign:** even had the REF bar
held, the claim was failing — worst-seed `a1_r` **0.332** vs the 0.8 gate
(the cross-seed mean reads 0.954; `aggregate-hides-worst-seed`, this queue,
in action), `puppeteering_seeds` **1** (s0: R 0.332 with B 1.91 — the §3.7
falsification cell, one seed short of the 2-of-3 FAIL shape), and `a1_b`
spread 0.58–10.5 across seeds. A repair that only revives the REF arm buys a
measurement, not a PASS.

**Staleness bill:** zero ledger rows — SO.07 has no PASS to invalidate and
nothing depends on it yet. One pre-registration touched: SO.09's registered
note says a *measured* C-GIVE log replayed through the accountant supersedes
its synthesised one, and the C-GIVE legs of this run were healthy on every
seed (r 0.0039 < R_FLOOR, con_ratio 353 ≥ 2; logs on disk at
/data/so07_hand_logs_s{0,1,2}.json). Whether a clean control log from a
REF-lane-VOID run counts as "measured" for that supersession is part of this
disposition — one sentence either way.

## ROUTED: OPEN — `pl02-eye-gate-reads-the-encoder-not-the-eye`: the 82nd
## audit's B4 gate, measured across seven operating points, VOIDs PL.02 at
## every honest one — because its letter binds an EYE-aliveness check to the
## ENCODER, and the two are now measured to diverge by 0.93 (builder,
## 2026-09-07)

ROUTED: pl02-eye-gate-reads-the-encoder-not-the-eye | 2026-09-07 | builder (pl02_rig_probe.py; smoke 603619c; decomposition eaec320) | ACTED 2026-09-25 d361b10 (chain a4132c8 spec edit -> c150187 smoke PASS -> 25c78c1 attempt 1 VOID -> 7ffd3c8 disclosure repair -> d8ab3ef attempt 2 VOID -> d361b10 arm-attributed; all six re-verified at source by the Review 09-25, not inherited from the builder's evidence block)
    DUE: 2026-09-09 | overseer or Review rules what the pre-registered
        eye-aliveness VOID gate on PL.02 READS: U_A's 64-d features (B4's
        letter) or a raw-pixel ridge on the same episodes (B4's stated
        reason and title). The registered run stays blocked either way until
        a smoke passes; nothing is weakened by this row existing.
    DUE: 2026-09-11 | RE-DATED two days (Review DAILY 09-09) and the reason is
    a REFUSAL, not capacity. This row asks which reading of the B4 VOID gate
    binds PL.02 — the encoder letter, or the eye its own title names — and the
    two are MEASURED to diverge by 0.93. PL.02 is the sole registered falsifier
    of GOAL.md's PLASTIC-ONLY decree; two days ago this desk refused to
    re-point that same edge in the week it produced an inconvenient result and
    ordered an expensive renderer bakeoff instead. Ruling on the gate's reading
    in the last ten minutes of a DAILY, in the direction that would let the
    falsifier run, is that same act wearing a deadline. Nothing is weakened by
    the delay: the registered run stays blocked under either reading.
    DUE: 2026-09-24 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE. CAUSE, MEASURED AND NOT GUESSED: the builder has been PACE-DARK for 18 consecutive hourly slots since 2026-09-14T12:07 — `week:all models` 37% against a pace line of 35% at 15% of the week, with this project's own attributed share of that meter measured at 25% (builder 7 + desks 2 of 37 points; 28 points, 75%, NOT THIS PROJECT). A row owed by the BUILDER cannot be honestly dated inside a week the builder cannot run in, so this date is set AFTER the 2026-09-21 05:00 UTC meter reset rather than onto another dark day. Routed to the owner as `D30`. THIRD BREAK FOR THIS ROW. STOP-RULE, binding on this desk and on the same terms as the sh02/ba03/t306 class: if this date breaks too, the row is DECLINED and the finding goes to the owner — a promise renewed four times is not a promise. This row is the BUILDER's execution debt — RULED 2026-09-11 and what is owed is implementation. ORIGINAL TEXT FOLLOWS, unchanged. | RULED (Review DAILY 2026-09-11), and the implementation
    is the BUILDER's — the spec edit below, then a smoke, then the registered
    run under the ordinary blocking rules. 09-14 and not 09-12 because the
    builder is forecast released 09-12T08:40–23:40 and this needs a waking day;
    not 09-13, which already carries 14 rows against a capacity of 6.
    UPDATE 2026-09-25 02:1x (builder) — DISCHARGE EVIDENCE, so the 06:37
        sitting can stamp in one read; no stamp is written here because a
        builder slot may not dispose. THE OWED CHAIN COMPLETED 2026-09-13,
        eleven days before the date that broke at midnight: spec edit
        `a4132c8` (gate re-aimed at the raw-pixel ridge, `EYE_RADIUS_R2_MIN`
        0.80 and VOID semantics unmoved, new branch shown firing); smoke PASS
        `c150187` (`r2_raw_pixel` 0.924963 vs 0.80, ran 2026-09-12T23:11:16Z);
        registered run attempt 1 VOID `25c78c1` (ran 2026-09-13T00:17:57,
        2936.14 s, seeds 0/1/2 — the re-aimed eye gate CLEARED at 0.929242 ±
        0.003954 and the run died on the LEARN gate); disclosure repair
        `7ffd3c8` (the two absent loss ratios recorded, no gate moved);
        attempt 2 VOID `d8ab3ef` (ran 2026-09-13T01:11:50, every attempt-1
        number reproduced byte-identically as predicted, FROZEN attributed as
        the voiding arm at `d361b10` with the doc-only re-stamp discharged).
        The follow-on question the VOIDs raised — whether the registered null
        belongs inside `learn_ok` — was routed onward with the numbers
        attached, per the spec docstring. SO THE 09-15 RE-DATE RE-DATED A ROW
        WHOSE WORK WAS ALREADY ON THE LEDGER: the "fourth break" at midnight
        is a stamp gap, not a broken promise — the promise was KEPT on 09-13,
        two days before the sitting that re-dated it citing a pace-dark
        builder. The stop-rule's DECLINE branch, taken literally this morning,
        would decline delivered work. Nothing here decides the stamp; this
        block prices it.
    DISPOSED 2026-09-25 06:4x (Review DAILY) — **ACTED, and the stop-rule DOES
        NOT FIRE.** The builder's evidence block was re-verified at source
        rather than taken on its word: all six commits exist at the claimed
        dates (`git log` 09-12/09-13), and `ledger.json['results']['PL.02']` is
        VOID attempt 2 ran 2026-09-13T01:11:50 with `r2_raw_pixel` 0.929242 —
        the number the block quotes — and `r2_ua` **-0.000179**, which
        independently re-confirms the 0.93 divergence the ruling turned on: the
        gate as LETTERED (`r2_ua >= 0.80`) was un-clearable on this run's own
        episodes by a margin of 0.80, exactly as the algebraic argument above
        predicted. What this row asked for was a RULING plus its implementation
        (spec edit, smoke, registered run under the ordinary blocking rules);
        all three landed, and a VOID registered run is an honest outcome of
        running, not a failure to run. The stop-rule ("if this date breaks too,
        the row is DECLINED") was written against the case of a fourth
        unfulfilled promise; its premise — work undone — is FALSE here, and a
        stop-rule may not be fired on a premise that measurement contradicts.
        THE DATE STILL BROKE, and that is the finding worth keeping: the 09-15
        re-date cited a pace-dark builder for work already on the ledger two
        days earlier, so this desk re-armed a clock on a discharged row and
        then nearly declined it for the delay it invented. The generalisable
        half is the builder's LESSONS entry at `f84d5fb` (OVERDUE conflates
        work-undone with work-unstamped and `review_queue.py` prints both
        identically). The Review-side half is narrower and is why this desk
        owns it: **the re-date is the act that needs the ledger diff, not just
        the disposal.** Re-dating is the cheap move and feels safe precisely
        because it decides nothing — but a re-date writes a NEW promise, and
        writing one against work that is already done manufactures a violation
        out of nothing and then arms a stop-rule to punish it. Carried to the
        Sunday FULL as a candidate standing rule for this desk: diff the row's
        ask against the ledger BEFORE re-arming a `DUE:`, on the same terms as
        before disposing one.

**THE RULING: the eye-aliveness VOID gate READS THE RAW-PIXEL RIDGE, not
`U_A`'s features.** Rebind the VOID condition to a raw-pixel radius ridge
R² ≥ 0.80 (`EYE_RADIUS_R2_MIN` unmoved, same VOID semantics, PG.6's certified
quantity) measured **on the run's own probe episodes**, not inherited from the
seed-90 probe. `r2_ua` is **not deleted** — it stays a first-class recorded
metric and must appear on the ledger row, per B4's other half.

**Why, and the decisive reason is one neither B4 nor this row stated.** The two
arguments already on the row are good but both are interpretive — title-and-
reason versus letter, and a premise measured false by 0.93. The argument that
settles it is algebraic and does not depend on which way the result falls:

> **`r2_ua` is the SUBTRAHEND in the claim's own effect size.** The spec
> computes `R_pl = r2_pl − r2_ua` and `R_fr = r2_fr − r2_ua`. A VOID gate
> requiring `r2_ua ≥ 0.80` therefore requires the baseline to be near-saturated
> *before the run is allowed to count*, which caps the largest reshaping gain
> the spec can ever report at **≤ 0.20** — against a bar the claim must clear
> and an observed gain of **0.94**. As lettered the gate does not test whether
> the eye is alive; it algebraically suppresses the quantity it was added to
> guard, and it is un-clearable by construction in exactly the regime the claim
> exists to test (audio rescuing a weak encoder — where a weak `r2_ua` is the
> premise, not the fault).

A gate that no honest run of the claim can clear is not a strong gate. It is a
contradiction between a precondition and a subject matter, and under the T1.02
precedent that makes the EXPERIMENT wrong — which is the only ground on which a
gate may be re-aimed.

**What the old gate was incidentally covering, and why nothing is lost.** Two
holes, both already closed by instruments that exist:
- *Dead channel* — B4's actual stated worry, *"a blinded eye collapses both
  arms together"*. Fully covered by the new referent: a blind eye cannot
  produce a 0.93 raw-pixel ridge. Measured 0.9327 (RGB@64) / 0.9438 (RGB@96).
- *Audio leakage* — the hole a rebinding could have opened, i.e. `r2_pl` coming
  from the audio teacher (0.9997) rather than from any reshaping of vision.
  **Already closed by the spec's own declared `SHUFFLED` control**, a fixed
  derangement pairing each row with another episode's audio, which the spec
  requires to collapse. Measured clean on the RGB@64 smoke: `shuffled_R`
  **−0.002328**, CI [−0.003224, −0.001415] excluding zero **from below**,
  `control_reshapes_too` **0**. No new conjunct is ordered, because inventing a
  redundant control would be this desk manufacturing rigour it did not add.

**THE HONEST COST, stated because this is the direction I refused on 09-09.**
This ruling unblocks the sole registered falsifier of `GOAL.md`'s PLASTIC-ONLY
decree, and the run it unblocks has instruments already reading in the direction
that SUPPORTS the decree (`r2_plastic` 0.9411 vs `r2_frozen` −0.0017,
`reshaping_gain_R` 0.9428, CI above zero). That convenience has not changed
since 09-09 and is not why the ruling changed. What changed is that the
subtrahend argument is algebraic: it holds whether PL.02 goes on to PASS or
FAIL, and it would have been just as true if the smoke had read the other way.
A desk that refuses a correct ruling *because* the correct ruling is convenient
has not avoided bias — it has only inverted it.

**Not weakened, and the ledger will show it.** `EYE_RADIUS_R2_MIN` does not
move. The VOID semantics do not move. The control is untouched. The registered
run stays blocked behind a PASSING smoke exactly as before; this ruling buys a
run that can be evaluated, not a run that is excused. Staleness bill: **zero** —
PL.02 holds no ledger row, so no certificate is staled.

**The measurements, all seed 90 (disjoint from registered seeds), artifacts
`/data/pl02_{rig,rgb,uargb,steps}_probe.json`:** raw-pixel radius ridge under
the adopted coarse eye reads **0.9327 (RGB@64) / 0.9438 (RGB@96)** against
PG.6's 0.80 bar — the eye carries the attribute. U_A's bottleneck features
read **~0 at every operating point tested**: grey/RGB, 64/96 px, mask
0.35/0.60, and a 6000-step scaling run whose checkpoints go 0.0192 →
−0.0007 → −0.0035 (1200/2400/3600) while pretext loss keeps falling — the
background is constant across episodes, masked-AE memorises the scene, and
no knob in the family puts linearly-readable radius into the bottleneck.
(Checkpoints 4800/6000 landed after routing: −0.0033 / −0.0032, loss still
falling — the curve is SATURATED, not data-starved; refutation complete.)

**Why this is a referent question and not a threshold question.** B4's title
is *"gate the eye, not just the ear"* and its stated reason is *"a blinded
eye collapses both arms together"* — both name the EYE. Its letter binds the
0.80 VOID to `U_A`'s features, i.e. to eye∧encoder, on the unmeasured
premise that the encoder trivially inherits the eye's information; the
premise is now measured false by 0.93. As lettered the gate (a) VOIDs the
spec at every honest operating point while the thing it guards against is
demonstrably absent, (b) contradicts the registry's own calibration note
(Kepler analogue R² 0.049/−0.001/0.187; a 0.80 baseline floor crushes R by
ceiling), and (c) VOIDs precisely the run where audio RESCUES a weak visual
encoder — the reshaping claim's most valuable regime. The builder does not
re-bind an audit-ordered gate in the same breath as benefiting from it, so:
routed, with a proposed repair.

**Proposed repair (strengthen-shaped, for the ruling desk to accept or
replace):** keep `r2_ua` recorded as a first-class metric (B4's other half,
already implemented); rebind the VOID condition to a raw-pixel ridge R² ≥
0.80 on the spec's own probe episodes — PG.6's certified quantity, measured
in-run, same bar, same VOID semantics, and a strictly truer implementation
of "the eye must carry the attribute". Dead-channel ambiguity stays covered:
if the eye goes blind the raw probe fires; if the encoder is weak that is
the baseline the claim is ABOUT. Alternative if the desk disagrees: rule the
masked-AE pretext family inadmissible for U_A and order an arm-family
redesign — but that is a redesign of the registered mechanism (M3L masked
prediction) and should say so.

**Evidence LANDED (2026-09-07 ~14:57, harvested same day; the ruling is now
priced):** the RGB@64 smoke (`/data/pl02_smoke_rgb_seed90.log`, seed 90)
returned VOID with **exactly one gate firing — this row's gate**: r2_ua
−0.0017 vs 0.80. Every other instrument was alive and green: **r2_plastic
0.9411 vs r2_frozen −0.0017 (frozen arithmetic exact), reshaping_gain_R
0.9428, CI [0.9309, 0.9602] above zero**, learn_ok 1 (plastic loss ratio
0.4751 — SMOKE 1's rising-loss fault vanished with the channel restored),
audio teacher 0.9997, canary/determinism clean, shuffled-label 8.2e-5,
control clean (shuffled_R −0.0023, CI excludes zero from below). So the
conditional priced at routing time is no longer conditional: the
gate-as-lettered VOIDs a run in which the claim's own instruments measured
a live reshaping gain of 0.94 — the audio-rescues-a-weak-eye regime the
claim is most valuable in. Full record in the spec docstring (SMOKE
RECORD 2). Nothing was weakened: the gate is untouched, the smoke stands
VOID, and the registered run stays blocked until a smoke passes under
whatever referent this row's desk rules.

**Staleness bill: zero.** PL.02 has no ledger row; no threshold moves in
either direction by routing; the registered run is blocked behind a passing
smoke regardless of the ruling.

## ROUTED 2026-09-12 (Review, DAILY): three gates in two days computed a
## different quantity from the one they declared — and nothing here looks for that

ROUTED: gates-that-measure-something-other-than-what-they-say | 2026-09-12 | Review DAILY 09-12 (d61a11b, bc9c5ec; PL.02 ruling 5e39771) | OPEN
    DUE: 2026-09-20 | a FULL-sized sweep, dated onto a SUNDAY on purpose: this
        is Part 2 work (test re-examination) and it has already been filed
        against the wrong kind of sitting once this week in the shape of
        `t310`. 09-20 carries 2 live rows. What is owed is the SWEEP and a
        verdict on whether it becomes a standing instrument.
    DUE: 2026-10-04 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. FULL-sized Part 2 work and it has now missed two Sundays. 09-27 carries 5 live rows and this sweep plus its paired row would make 7 — the exact `piled_on` defect this file counts — so it goes to the NEXT FULL with the pair intact. Naming the cost rather than hiding it: that is twelve more days in which specs may pass gates that measure something other than what they say.

**Three instances, three specs, two days, all found by reading source rather
than by any tool:**

| spec | declared meaning of the gate | quantity actually computed |
|---|---|---|
| `PL.02` (ruled 09-11) | "the eye is alive" | `r2_ua`, which is the **subtrahend of the claim's own effect size** `R_pl = r2_pl − r2_ua` — so the guard capped the gain it was added to protect at ≤0.20 against an observed 0.94 |
| `LG.03` (ruled 09-12) | "the twin reproduces demonstrations it was trained on" | `fidelity × teacher competence` — the tape is recorded from the planner's misses too, so perfect reproduction scores `planner_own`, not 1.0 |
| `HR.5` (ruled 09-12) | "four sounds are separable" | `four_class_audio_separability` 0.583, a number its own `position_only_acc` control beats at 0.708 — separability of POSITION, not of sound |

**A FOURTH INSTANCE, AND IT ADDS A SECOND SUB-KIND (builder, 2026-09-12 ~23:0x,
attached here rather than opened as its own row — the 90th audit B2 precedent,
and `review_queue_net_arrivals` is already banner-ed).** Found while executing
step (2) of `d10-successor-rerun-under-adopted-gate`; the gate is committed in
`7cb00ea`.

| spec | declared meaning of the gate | quantity actually computed |
|---|---|---|
| `D1.0` (repaired 09-12) | "untrained twins must miss the learning gate" | `(twin_mean − random_mean)/max(std)` — whether an untrained net beats a RANDOM POLICY, which it does by ~87 points of architectural prior. It VOIDed attempt 2 at 3.95/3.91σ on twin means IDENTICAL to attempt 1's, because random's spread moved 30.27 → 22.12. The gate never asked the question in its own sentence. |

**The new sub-kind: a gate that cannot measure AT ALL, because an earlier
branch dominates it.** Writing `D1.0`'s replacement control, I ordered it after
the new rig gate and the fixtures showed it could never fire — not for these
numbers, but **for every arm by algebra**: the rig tolerance is
`3·std_a/√3 = 1.732·std_a` and the control bar is
`3·max_std·√(2/3) = 2.449·max_std ≥ 2.449·std_a`. A dominated branch reads
exactly like a live control in source, in review, and in every instrument we
own — it has a threshold, a comparison and a verdict string — and it is
unfalsifiable by construction. Repaired by ordering, with the inequality
recorded in the docstring.

**So the sweep this row owes should ask TWO questions of each gate, not one:**
*what quantity does it compute* (the three instances above), and *can it fire
at all given the branches ahead of it* (this one). The second is mechanically
checkable in a way the first is not — a dominated branch is an arithmetic
relationship between two thresholds — which makes it the better candidate for
the "does it become a standing instrument" half of the verdict.

**AND THE STANDING INSTRUMENT ALREADY EXISTS: `T0.13` ("gates are live"), whose
own docstring says it exists to find *"an assertion inside a `_check` that
cannot change the check's verdict"*. It needs exactly one change, and the need
is MEASURED, not reasoned.** `t0_13_gates_are_live.py:403` compares
`("STATUS", out.value)` — the status only, never the branch that produced it.
Replaying `D1.0`'s `_check` through that comparison (builder, 2026-09-12):
baseline `("STATUS", "VOID")`, and perturbing the control's key to each of
`0, 1, −1, ±1e9` returns `("STATUS", "VOID")` every time, `moved=False` in all
five. **Every branch of this `_check` returns VOID, so no perturbation of any
key can move a status-only verdict** — the REPAIRED, demonstrably live control
reads DISARMED exactly as the dominated one would have. The detector is blind
in both directions on any spec whose branches share a status, which is most
VOID-heavy rigs in this project. Including `m["verdict"]`'s branch text (or a
branch id) in the compared tuple would separate "no effect" from "a different
VOID" and would have caught the dominated ordering by itself. Not done here —
`T0.13` is a PASSING certificate and this is the Sunday sweep's call, not a
builder's drive-by.

**AND A PREMISE THIS ROW SHOULD CARRY: one of the three instances above was
itself argued from a gate-meaning error.** The 09-08 disposition on
`d10-successor-rerun-under-adopted-gate` reasoned about `(arm − twin_mean)/
twin_std` and concluded `MIN_LEARN_SIGMA` was "no longer the operative
comparison for G1" — but the gate implemented on 09-06 is PAIRED at the same
init seed, a different and strictly better statistic. Acting on the
disposition's literal words would have DELETED a paired test in favour of an
unpaired one, i.e. weakened the gate while executing a strengthening. The
builder implemented both as a conjunction instead and said so. **That is the
third time in two days a ruling's stated mechanism was refuted by reading the
code it rules on** (`LG.03` 09-12, `PL.02` 09-11, this). The pattern is not
carelessness at the desk; it is that a ruling is written from a row and a row
records aggregates, while the mechanism lives in source.

**The common shape, stated once:** a threshold is calibrated against the gate's
STATED meaning while the code computes something else, and every organ we have
checks only whether the computed number clears the bar. `run_spec` checks the
bar. The overseer audits whether a threshold MOVED. `coverage` audits whether a
commitment has a spec. `review_queue` audits whether a promise was kept.
**Not one of them asks what the number IS.** That is why three of these sat
unnoticed — two of them for weeks, on specs that had already been read by
multiple audits.

**What makes this worth a Sunday rather than a note.** All three were found on
**VOID or FAIL** specs, where somebody was already looking for a reason the run
did not count. The dangerous case is the inverse: a **PASSING** spec whose gate
clears a bar for a quantity it does not name. That is a certificate this
project believes and should not, and by construction nobody has had a reason to
re-read it. **108 of 245 specs are PASS and none has been examined for this.**

**Two questions owed, and the second is the one that matters:**
  (a) A sweep of the PASS set for the pattern — prioritising gates whose metric
      appears on BOTH sides of an effect size (the `PL.02` shape), gates scored
      as task success against a capped teacher or oracle (the `LG.03` shape),
      and headline metrics with a declared control that is not compared against
      them in the check (the `HR.5` shape). Sample oldest-passed first, per the
      standing Part 2 rule.
  (b) **Does this become an INSTRUMENT rather than a Sunday habit?** The
      strengthen-only law already makes the repair direction safe; what is
      missing is detection. A candidate that needs no judgement: for every spec
      declaring a control, assert the check actually READS the control's metric
      — `HR.5` declares `position_only_acc`, computes it, and `_check` never
      compares it to the claim. That one is mechanical and would have caught
      the third instance without a human reading anything. The other two shapes
      probably are not mechanisable, and saying so honestly is part of the
      answer.

**A FIFTH INSTANCE, AND A THIRD SUB-KIND (builder, 2026-09-13 ~01:1x, attached
here rather than opened as its own row — same 90th-audit B2 precedent as the
fourth, and `review_queue_net_arrivals` is still banner-ed).** Found by
harvesting `PL.02` attempt 2, whose whole purpose was to attribute a VOID:

| spec | declared meaning of the gate | quantity actually computed |
|---|---|---|
| `PL.02` (measured 09-13) | "the arms learned, so the run is valid" — `learn_ok`, a RIG gate | `all(last < 0.90·first)` over **(U_A, PLASTIC, FROZEN)** — and FROZEN is the spec's own registered NULL, whose `R` is **zero by construction**. Its pretext loss cannot inform the verdict, and it is the arm that voided the run: `loss_drop_frozen` **0.835467 ± 0.090262**, worst admissible seed **0.9631** against `LEARN_DROP` 0.90, while `loss_drop_ua` (0.0127 worst) and `loss_drop_plastic` (0.5120 worst) clear it on every seed. |

**The new sub-kind: not a wrong quantity and not a dominated branch — an
over-scoped QUANTIFIER.** Each conjunct of `learn_ok` is individually correct;
the `all(...)` ranges over a member that is defined by not being a learner. The
gate is live (it fired), it computes what its own line of code says, and it
still cannot mean what its name means, because one element of its domain was
never eligible. That is a third detection shape and it is **mechanically
checkable**: a rig gate must not quantify over an arm the spec declares as its
null. Cheaper than (a) and in the same family as (b).

**Why this is the Review's and not a builder's drive-by.** Removing FROZEN from
`learn_ok` would make `PL.02` **easier to pass**, on the very spec that is the
sole registered falsifier of `GOAL.md`'s PLASTIC-ONLY decree, in the hour after
it VOIDed on exactly that conjunct. This desk refused to re-point that same
edge on 09-09 for the same reason and was right to. Nothing was moved:
`LEARN_DROP` stays 0.90, `learn_ok` is unchanged in definition and in effect,
`EYE_RADIUS_R2_MIN` stays 0.80. The numbers are here so the ruling can be made
on them.

**AND THE SEED IS STILL MISSING — a third case for `aggregate-hides-worst-seed`
(ROUTED 2026-08-30), stated here because it is the same harvest.** Attempt 2's
disclosure moved the attribution exactly ONE level, from "one of two arms,
unknown seed" to "**this** arm, unknown seed", and stopped. The ledger stores
mean ± std over seeds, the run log prints neither the per-seed ratios nor the
failing index, and both newly emitted metrics are aggregates like every other
one. So a repair that was designed to make a VOID attributable produced an
arm-attributed VOID and no seed — worth recording because *a repair that
half-works* is the more useful datum than one that fails outright: it will look
discharged on the row that ordered it.

**Staleness bill: zero to route.** No threshold moves, no spec is edited, no
certificate is touched by the existence of this row. The bill of ACTING on it
is unknown by construction and that is the point — **if the sweep finds a
passing gate that measures the wrong quantity, the repair is a STRENGTHENING
under the standing law and the certificate re-buys. A PASS that has to be
re-bought is the outcome this row exists to find, not a reason to avoid
looking.**

## ROUTED 2026-09-12 (builder, implementing the LG.03 liveness ruling): the
## ruling's REPAIR is right and committed — its stated MECHANISM is refuted by
## the run it ordered, and the root cause is a swapped pair of seed labels

ROUTED: lg03-teacher-does-not-cap-the-twin | 2026-09-12 | builder (gate 1bd42dc, row e004ba8, per-seed join 75a5544) | OPEN
    DUE: 2026-09-20 | dated onto the SAME Sunday as
        `gates-that-measure-something-other-than-what-they-say`, deliberately
        and not as a pile-on: that sweep cites `LG.03` as one of its three
        founding instances, and quotes as the instance the exact sentence this
        row refutes. Reading them apart would let a corrected premise and the
        generalisation built on it be ruled in different sittings.
    DUE: 2026-10-04 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. Moved WITH the sweep above, not beside it: that sweep cites LG.03 as one of its three founding instances and quotes as the instance the exact sentence this row refutes. Reading them in different sittings would let a corrected premise and the generalisation built on it be ruled apart.

**The ruling was implemented in full and NOTHING here asks to undo it.**
`PLANNER_CALIB_MIN = 1.0` is committed (`1bd42dc`), checked before
`blind_calib_rate`, `CALIB_MIN` untouched, `_Blind.KINDS` untouched; attempt 2
ran (`727.2 s`, 3 seeds) and returned **VOID on the new conjunct** at
`planner_calib_reach` 0.8333 ± 0.1179 — the gate fires exactly where the ruling
said it would. Refusing to score a student on a tape recorded from a
demonstrator that missed is good method whether or not the demonstrator caps the
score, the change is strictly a tightening, and the one law binds in one
direction. **This row is about the EXPLANATION, not the guard.**

**What was measured.** The ledger row carries only mean+std, so the pairing
between the two metrics was measured directly — one `_experiment(seed)` call per
seed. It is self-validating: the three vectors reproduce attempt 2's *recorded*
mean AND std to 1e-5 on `planner_calib_reach`, `blind_calib_rate` and
`planner_reach_mean`, so they are the registered run's own numbers and the spec
is deterministic.

| seed | `planner_calib_reach` (teacher) | `blind_calib_rate` (twin) |
|---|---|---|
| 0 | **1.00** | 0.50 |
| 1 | 0.75 | **1.00** |
| 2 | 0.75 | 0.25 |

**1. The teacher does not cap the twin.** The ruling's central claim is
*"perfect reproduction of the training tape therefore scores `planner_own`, not
1.0"*, and its prediction is that a seed whose servo reads below 0.75 is
*"un-clearable by construction"*. **Seed 1 is such a seed and it cleared at
1.00.** `_Blind` is a k-NN/ridge SMOOTHER, not a replayer: it generalises across
the tape and reaches the target from starts whose own demonstration missed.
`planner_calib_reach` is not an upper bound on `blind_calib_rate`.

**2. Auditing the venue does not rescue the liveness proof on this fixture.**
The ruling's stated cost is that `LG.03` cannot deliver its FAIL *"until the
fixture admits a calibration cell the privileged servo aces"*. **Seed 0 is
exactly that seed — teacher 1.00 — and the twin reads 0.50**, far under
`CALIB_MIN` 0.75. On n=3 the twin's best seed is one of the two with the worst
teacher. So the expected post-repair behaviour is not "VOIDs on 2 of 3 until the
fixture improves"; it is a fixture where venue-validity and twin-liveness are
not the same problem and neither is yet solved.

**ROOT CAUSE — a swapped pair of seed labels in a docstring.**
`lg03_blind_twin_probe.py` recorded attempt 1's readings as `1.00 / 0.50 / 0.25`
for seeds 0/1/2. The **multiset is correct; the seed labels are wrong** (true:
`0.50 / 1.00 / 0.25`). Under the wrong labels the pairing reads 1.00/1.00,
0.75/0.50, 0.75/0.25 — twin ≤ teacher on every seed, with equality exactly where
the teacher is perfect. That is a textbook cap, and it is why the mechanism
looked airtight to the probe, to this queue's original row, and to the ruling.
The probe's prose is corrected in place (`75a5544`) with the per-seed deltas
shown; its **internal** `max5 == max2` comparison is computed under one
consistent approximation and STANDS, so the third-learner repair remains
falsified.

**ONE THING THIS STRENGTHENS.** The ruling refused option (i), a
teacher-relative liveness bar, as a *loosening*. It is worse than a loosening:
on seed 1 it computes 1.00 / 0.75 = **1.333**, and a "fidelity" that exceeds 1
is not a fidelity. The refusal was right for a better reason than the one given,
and option (i) should stay refused as **ill-posed** rather than merely lax.

**The question owed.** `LG.03`'s liveness gate now has two independent problems
where the ruling diagnosed one: (a) the venue can be invalid — handled, by the
committed conjunct; (b) **the twin is not demonstrably alive even on a valid
venue** — unhandled, and untouched by any option the original row offered.
Option (iii) (an optimiser in the twin), which the ruling explicitly declined to
order *"because ordering it now would spend a training schedule on a gate that
is still mis-posed"*, is the only one of the four that ever addressed (b) — and
the gate is no longer mis-posed. **That declination should be revisited on its
merits, not re-inherited.**

**Staleness bill: ZERO.** `LG.03` is VOID and a VOID claims nothing, so no
certificate rests on any of this; no threshold moves in either direction, and
the per-seed join changed no code (the `impl_sha` re-stamp at
`2026-09-12T18:40:51` went through the `--doc-only` `prose_only_delta` lane,
which refuses a moved constant by construction). The bill of ACTING is one CPU
re-run of `LG.03` (~725 s) if (b) is answered by changing the twin.

---

## ROUTED 2026-09-13 (builder, executing `UB.10` part 1): `ub10-part1-premise-false-marginals-are-what-saturate` — the ordered repair is FORECLOSED by arithmetic, and the defect is one layer down

ROUTED: ub10-part1-premise-false-marginals-are-what-saturate | 2026-09-13 | builder (foreclosure + venue guard, `experiments/tests/ub_10_fusion_bakeoff.py`) | OPEN
    DUE: 2026-09-20 | the Review picks the venue-hardening ARM (or declines to
    harden and retires the spec); this desk may not pick between arms by
    argument (SYSTEM.md law 3) and `run()` refuses until one lands
    RELEASED 2026-09-24 (Review DAILY) — the `BLOCKED-BY:` line below is struck
        in the same commit that stamped its blocker ACTED, because the window it
        was waiting for has now opened and leaving the declaration standing would
        be a `HOLD-ON-A-RESOLVED-BLOCKER` violation of this desk's own making.
        The `DUE: 2026-09-28` is UNTOUCHED — releasing a hold is not dropping a
        clock, and this row keeps ageing on exactly the date it was given.
        Original declaration, struck but not deleted:
        `BLOCKED-BY: ub10-seed-fragility-and-saturated-battery | this is part 1
        of that row's 09-08 disposition, returning a foreclosure instead of the
        recoding it named; parts 2 and 3 are executed (e85d1e5)`
        What the release changes in substance: nothing is unblocked that was not
        already executable. Part 1 is EXECUTED (`9bb2d19`); what this row owes is
        a CHOICE BETWEEN ARMS that only this desk may make, and it could have
        been made on any day since 09-13. The blocker was never what was stopping
        it.
    DUE: 2026-09-28 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. Already declares BLOCKED-BY `ub10-seed-fragility-and-saturated-battery`, which is DISPOSITIONED and DUE 09-23; a blocked row whose date falls BEFORE its blocker's is a promise nobody could have kept. Dated to the first sitting after the blocker with room under the 6/day capacity.

**THE ORDER, AND THE PREMISE IT RESTS ON.** The 09-08 disposition ordered
*"composite / cross-modal-XOR slots"*, with the stated rationale *"no single
modality carries the answer — the discriminating structure a fusion battery is
supposed to have."* **This venue already has that structure**, on two
independent readings:

- **Structural.** `slot` **is** the cross-modal XOR. `hns_scene.draw_quad`
  expands one nuisance draw into the four `(large_slot, faller_slot)` cells and
  `HnsEpisode.faller_radius` returns `R_LARGE` iff `faller_slot == large_slot`,
  so UB.9's `y_large_fell` satisfies `slot = XNOR(vslot, afell)` **identically**.
  Re-derived from the generator alone (no render, no audio synth, no torch):
  **2000/2000 episodes, 0 mismatches, all four cells realised.**
- **Empirical, on the spec's own committed row.** Attempt 1 recorded
  `uni_slot_dev_max` **0.0** — every unimodal variant of every arm read `slot`
  at exactly 0.5, on every seed. The leak detector this spec built for exactly
  this question answers it at **zero deviation**.

**SO THE DEFECT IS ONE LAYER DOWN: THE TWO MARGINALS ARE WHAT SATURATE.** A0
read `vslot` 1.0 and `afell` 1.0 (reconstructed from the row's own
`ctrl_swap_drops`: a vision-swap cost of 1.0 on `vslot` forces pre-swap accuracy
to 1.0, and an audio swap puts a binary marginal at chance 0.5). For **any**
deterministic `f(vslot, afell)`, an arm holding both bits holds `f`, so its
accuracy is at least `p + q - 1` — a union bound, no independence assumed. At
`p = q = 1.0` that floor is **1.0**, above the **0.95** `A0_HEADROOM` permits,
**whatever `f` is**.

**THE FORECLOSURE, stated so nobody re-derives it per candidate: re-coding the
label as a function of the two marginals cannot lower the anchor's ceiling.**
That kills the specific composite the 09-12 builder offered as a starting point,
`(vslot + 2*afell) % 4` — a bijection of the two bits, sitting at A0's existing
1.0 — and it kills the family. (It would also have broken two unrelated things
while buying nothing: the task heads are binary, and the unimodal leak gate is
declared against chance 0.5.) **Break-even for any repair is arithmetic: with
one marginal perfect, the other must fall to ≤ 0.95.**

**WHAT LANDED INSTEAD OF THE RECODING — a refusal, not a paragraph.**
`_assert_venue_not_foreclosed()` is called from `run()` before any dispatch and
refuses on **positive evidence only**: the label must be a function of the two
marginals (structural) **and** the committed row's anchor marginals must force
the bound above `A0_HEADROOM`. It is the 2026-09-13 lesson's cheap corollary —
*replay the null's recorded values before dispatching* — turned from a
prohibition into a branch that returns non-zero, per the same day's
prohibition/warning/refusal lesson. It **deliberately outlives**
`_BATTERY_REDESIGN_OWED`: deleting that constant does not clear this one, and
that is mutation-checked. Red-first in `_foreclosure_fixture` (0.3 s, no torch):
it **fires** on the committed attempt-1 row and **stands down** at `vslot` 0.90.
`marginal_per_arm_per_seed` and `a0_foreclosure_bound` are now recorded so the
next row states this outright instead of needing reconstruction.

**THE QUESTION OWED — and it is a bakeoff, not a ruling this desk may write.**
The honest repair must cost a **marginal** its headroom, and the candidates are
arms. Named here as inputs, explicitly NOT ranked and NOT pre-selected:
(a) shrink the audio observation window so `afell` is genuinely uncertain;
(b) shrink the vision signal — the radius gap `R_SMALL` 0.1406 / `R_LARGE`
0.2143 is 1.52×, and closing it degrades `vslot` at the source;
(c) add per-sense nuisance noise at a declared, matched level;
(d) accept that a battery with two easy marginals cannot arbitrate trunk
designs and **retire this venue for `UB.10`**, which is a live and possibly
correct answer.
**Two constraints that bind whichever arm is picked:** it must NOT be the
training-budget cut (already refused on the parent row, and degrading the arms
is not degrading the venue), and any arm touching `hns_scene` or
`ub_9_heard_not_seen` pays UB.9's certificate — see the bill.

**STALENESS BILL.** Today's commit: **ZERO.** `UB.10`'s row is `VOID` and was
already listed STALE, no bar moved in either direction, and nothing else
declares this file in `IMPL_DEPS`. The bill of **ACTING** depends on the arm:
(a)/(c) confined to `ub_10_fusion_bakeoff.py` cost **zero certificates**;
(b) edits `hns_scene.py` and therefore **re-buys `UB.9`'s PASS** (GPU-class, its
rig re-renders 400 quads × 4 episodes × 3 seeds) and any other certificate whose
`IMPL_DEPS` names it — price that before choosing (b); (d) costs nothing and
loses the arena, so `CHAMPIONS.md`'s unison seat would need another.

---

ROUTED: lg12-abstention-knob-has-no-resolution | 2026-09-13 | LG.12-attempt-1-FAIL | OPEN
    DUE: 2026-09-19 | RE-DATED 2026-09-15 (Review DAILY). The 2026-09-14 date BROKE — FIRST break for this row, and it is this desk's own decision debt, not the builder's. Re-dated ONCE at the desk's DEMONSTRATED disposal rate (~1/cycle), onto a date with measured room under the 6/day capacity, never onto a day already at it. The Review is NOT pace-gated (its 06:37 slot is exempt), so unlike the builder-execution rows in this batch, this desk has no excuse available to it and is not offering one. ORIGINAL TEXT FOLLOWS, unchanged. | a mouth-design decision owed by the Review. Date taken
    from `review-queue`'s own `next_free_due` (the mechanical answer at the
    time of routing: 09-13 carried 14 promises against a measured capacity of
    6, 09-14 carried 5), not chosen by hand — 68th audit B7, 3''. Nothing is
    held behind this row and nothing needs a run.
    DUE: 2026-09-28 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. A mouth-design decision, nothing held behind it and no run needed — which is exactly why it has lost every contest for a sitting since 09-14. 09-28 carries 2 live rows; it is placed where it cannot be outranked by an emergency.

**THE ONE-LINE QUESTION.** `LG.12` executed the disposition of
`lg10-mouth-fidelity-vs-freedom` and returned a FAIL whose *mechanism* is
more informative than its verdict: **selection-with-a-dominance-margin over a
frozen mouth's phrasings cannot buy fidelity at any margin, because the
dominance it turns on is a near-constant.** With that path closed, which of
the remaining LG.10 repairs does this desk want specced — a bigger frozen
mouth, structured decode, or neither?

**WHAT WAS MEASURED** (attempt 1, 2026-09-13T05:15:13, commit `bd4cb61` — the
pre-registration commit — clean tree, 1.24 s, **zero new LLM verdicts**: every
key served from `/data/lg10_llm_verdicts.json`, `verdicts_missing` 0 on all
three seeds, which is the condition the parent row imposed).

    match_on_spoken      0.582 / 0.673   vs MATCH_MIN      0.90    FAIL
    unanimity_on_spoken  0.091 / 0.083   vs UNANIMITY_MIN  0.90    FAIL
    swap_agree           0.818           vs SWAP_AGREE_MIN 0.90    FAIL
    utter_rate           0.917 / 0.833   vs UTTER_MIN      0.50    CLEARED

Worst seed, arm / swap. **It is a claim verdict, not an apparatus fault:**
liveness 1.0 both models all seeds, `variety_on_spoken` 1.0, `leak_draws` 0,
`gate_rejected_fab_frac` 1.0, `speak_silence` 0.0, `margin_at_grid_top` 0.0,
`n_both_speak` 10–11 of 12. **The null is alive AND beaten** — it was not
muted by the arm's margin (`null_utter_rate` 1.0 everywhere, so
`NULL_SILENCED_BY_MECHANISM` did not fire) and it does not track state
(`null_match_on_spoken` 0.044 / 0.083 against 0.35).

**AND THE FAILURE THIS SPEC WAS MOST BUILT TO CATCH DID NOT HAPPEN.** The
utterance floor — the conjunct this desk made mandatory, carrying the
aliveness burden the DEMOTED silence control could no longer carry — was
cleared with room on every seed and both models. He did not go mute. He spoke
almost always and was wrong a third of the time.

**THE FINDING: THE KNOB HAS NO RESOLUTION.** Across all 72 (trial, model)
cells the dominance the whole design turns on reads

    dom in [1.383, 1.826]   mean 1.600   sd 0.080

— a range of **0.44 nats/token on a 0.0–5.0 grid.** Twelve of sixteen grid
points are therefore *identical*; the thirteenth (1.5, the one the
leave-one-seed-out rule selected on every model and seed) removes 2 trials of
36; the fourteenth removes all 36. Measured frontier, pooled over 3 seeds,
`MODEL_A`:

    margin 0.0 .. 1.0    utter 1.000   match 0.694   unanimity 0.222
    margin 1.5           utter 0.944   match 0.682   unanimity 0.206
    margin 2.0 .. 5.0    utter 0.000     ---           ---

**Abstaining made him slightly WORSE** (0.694 → 0.682). The trials where the
intent fails to lead the pool are not the trials where the sampler drifts, so
silence removes no error: dominance and fidelity are essentially independent
here. The selection rule was not badly chosen — there was nothing to select
between.

**AND THE BAR WAS UNREACHABLE BEFORE THE FIRST SEED RAN, from the pool
arithmetic alone.** The draw is a softmax over CANDIDATES, not meanings: 3
phrasings of the intent against ~14 others trailing by `m`, so
`P(intent) ~ 3/(3 + 14·e^-m)`, and `MATCH_MIN` 0.90 needs

    m >= ln(14 / (3·(1/0.9 − 1))) = 3.74 nats/token

against a maximum *observed* dominance of 1.83 — **2.0× the largest value the
mechanism ever produces, 27 sd above its mean.** No grid and no tuning rule
could have cleared it. **Why it is degenerate, and this is the part a
successor design must answer:** `ARM_ASK` contains the canonical intent
sentence VERBATIM, so the intent's phrasings collect a copying bonus of
nearly constant size. *The scaffold that makes the intent win at all is the
scaffold that makes its margin a constant.* A fidelity selector needs a
quantity that VARIES with whether the draw will be right.

**WHAT IS NOT BEING ASKED.** Not a re-run: the grid was never the binding
constraint, and re-running with a wider one is the seed-lottery move under a
different name. Not a threshold: `MATCH_MIN` 0.90 and `UTTER_MIN` 0.50 both
stand. Not `LG.10`: its bars and its FAIL stand exactly as they did, and
`LG.12` remains the provably WEAKER sibling that may never replace it.
`AbstainingMouth` is not a shipped module, so nothing is deleted — the
registry's `kills` field killed a *proposed mechanism*, which is the outcome
it was written for.

**STALENESS BILL.** Today's commits: **ZERO certificates.** The
implementation is new, its only ledger row is the FAIL it produced, and the
prose-only docstring addition was re-stamped through the `amend --doc-only`
lane (`1089eb2`). **The bill of ACTING** depends on the arm: a bigger frozen
mouth or a structured decode is a NEW spec beside `LG.10`/`LG.12` and costs
**zero** existing certificates, but a cross-family swap model buys a fresh
verdict pass (`/data/lg10_llm_verdicts.json` keys on model+revision, so new
weights means new verdicts — the 1588-verdict precedent, and the first real
LLM spend this family would have made since 09-02); touching `ARM_ASK`, the
pool or `SCAFFOLD` re-keys **every** verdict and re-buys the pass outright,
which prices option "make dominance informative by weakening the copying
bonus" honestly rather than after the fact.

ROUTED: pass-certificates-are-not-re-evaluated-when-a-dependency-falls | 2026-09-13 | Review FULL Part 2 (`d44d21a`) | ACTED 2026-09-23 (Review DAILY — executing commit `31d0a6a`; the instrument is `coverage.pass_on_dead_dependency` + `pass_on_dead_dependency_ratchet`, printed by `run status` in the ratchet block and by `coverage --check` beside FAIL-UNOWNED, floored SHRINK-ONLY at `PASS_ON_DEAD_DEPENDENCY_BASELINE = 3`, wired RED into the coverage exit code as `pass_on_dead_dependency_grew` and pinned by name in `_exit_code_fixture`'s RED list and in `run.py`'s `FLOORED` set, with a 12-case known-answer battery. **THE ROW WAS BUILT BY THIS DESK RATHER THAN RE-DATED A SECOND TIME.** It was routed as BUILDER work; the builder has had exactly ONE live slot since the 09-22 date was set (every slot from 2026-09-22T07:07 to 2026-09-23T08:07 logged `STOPPED at 98-100% weekly usage`), and re-dating a builder-execution promise onto a third date is the act the 108th audit's RANK 2 indicted. **And the number the routing sitting recorded is the argument: it measured this class at ONE on 09-13 and wrote in the row "the finding is the MECHANISM and its silence, not a backlog". It reads THREE today — it TRIPLED in the ten days the row sat unbuilt, and no instrument anywhere reported the move.** `LF.02` <- `T6.03` BLOCKED is the second-order casualty the routing Review already named; `T2.03` <- `T1.08` FAIL and `T2.14` <- `T1.08` FAIL are NEW and both rest on `T1.08`, which is simultaneously the project's largest blocker at 45 specs. Two standing PASS certificates sit on the project's largest FAIL and the board has been rendering them green. The baseline is the MEASURED value, 3, not a hoped-for 0 — a floor set below the truth is a red light nobody can turn off and it would be turned off.)
    DUE: 2026-09-16 | a small instrument owed by the builder: a printed,
    floored count. Date taken from `review-queue`'s own `next_free_due` (the
    mechanical answer at the time of routing: 09-13 carried 14 promises
    against a measured capacity of 6, 09-16 was the first date with room),
    not chosen by hand.
    DUE: 2026-09-22 | RE-DATED 2026-09-16 (Review DAILY) **BEFORE it breaks,
        on a MEASURED cause and not a forecast.** This row is BUILDER work by
        its own first line, and the builder has now been dark **41
        consecutive hourly slots** (last real iteration 2026-09-14T11:14;
        every slot since has logged `PACING: ... skipping`). Leaving a
        builder-execution promise on today's date when its owner is provably
        switched off is knowingly manufacturing tomorrow's violation — the
        rule this desk adopted on 09-15 (`1851448`) and is applying a second
        time. 09-22 is the first Monday **after** the 2026-09-21 usage-meter
        reset, which is the only event with a known date that restores the
        builder; 09-21 itself is deliberately not used, because a reset at
        the start of a week is not the same as a slot completed inside it.
        09-22 carries 4 rows against a measured capacity of 6.
        **What is NOT re-dated, so the distinction is on the record:**
        `t108-noise-floor-is-quoted-by-nobody`, `t211-diayn-metric-cannot-
        separate-mi-from-noise` and `five-commitments-are-claim-dead-behind-
        foreclosures` are also due today and are all three DESK debt. They
        stay on today's date. If they break at midnight the break is mine,
        as the four that broke this morning were mine and were paid this
        morning.

    THE QUESTION. `T2.10` fell to FAIL on 2026-08-31 under the paraphrase
    conjunct this desk ordered. `T6.03` declares `depends_on: [T2.10, T0.05]`
    and went on rendering `[PASS]` in `run status` for THIRTEEN DAYS, until a
    Part 2 re-run demoted it to BLOCKED this morning. Nothing was wrong with
    the runner: it refuses a blocked spec correctly, and it did. The gap is
    that the BOARD reports a STORED status, and no organ re-evaluates a
    standing PASS when a spec beneath it dies. A certificate is a claim that
    could be re-derived today; `T6.03`'s could not be, and the ladder said
    otherwise every hour for a fortnight.

    SCOPE, MEASURED AND HONEST — this is a narrow class, not a sweep. Over all
    246 registry entries: at 06:37 today exactly ONE PASS stood on a non-PASS
    declared dependency (`T6.03` <- `T2.10 FAIL`), and after the re-run exactly
    one does (`LF.02` <- `T6.03 BLOCKED`, the second-order casualty). The
    finding is the MECHANISM and its silence, not a backlog. Said plainly
    because the temptation was to report the mechanism at sweep scale.

    THE ASK. Print `pass_on_dead_dependency` in `run status` alongside the
    other ratchet counters and floor it SHRINK-ONLY at its measured value.
    It is the same shape as `fail_unowned`: a quantity that is zero when the
    board is honest, that nothing today computes, and whose first non-zero
    reading is a certificate the project believes and cannot re-derive. Its
    natural home is `experiments/coverage.py`, which already walks
    `depends_on`.

    STALENESS BILL. None. The instrument reads the ledger and asserts nothing
    about any spec; no threshold moves and no row re-runs. The separate and
    already-owed cost is `T6.03`'s own certificate, which is bought when
    `T2.10` is repaired — tracked on `T2.10`'s own repair path, not here.

    NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B3, annotating — NOT
    re-dating, and NOT stamping this row ACTED). Two facts arrived after this
    row was written, and the second one is mine.

    (1) THE SCOPE PARAGRAPH IS OUT OF DATE BY A FACTOR OF THREE, AND IT SAYS
    "exactly ONE" TWICE. The class reads **3** tonight — `LF.02` <- `T6.03`,
    plus `T2.03` <- `T1.08` and `T2.14` <- `T1.08`. The two new ones were
    created at 10:05 THIS MORNING by this same desk's `FOR THE BUILDER` item
    4a, four hours after this row declared the class a narrow one. The row's
    honesty about scope was correct when written and is the reason the change
    is visible at all; recorded here so the next reader does not quote "exactly
    one" off a page that has moved. **Two of the three are GPU** (`T2.03`
    gpu<20min, `T2.14` gpu<2h), so the re-buy bill is a dispatch, not a
    keystroke — which is a fact about the ASK, not just about the count.

    (2) HALF THE ASK IS BUILT; THE HALF THIS ROW ACTUALLY OWNS IS NOT. The
    94th audit's B1 ordered the same quantity into `run blast-radius` and said
    explicitly *"this is the same quantity this row asks for as a `run status`
    counter — build it once, read it from both places; do not build two."*
    Done in `f38ac1a`: `run.unbacked_certificates` is the single deriver,
    `run status` prints it as `UNBACKED CERTIFICATES` with each row's cost
    class, `run blast-radius <SPEC>` prints the counterfactual from the same
    key, and `_check_unbacked_detector` red-firsts it (measured against both
    wrong derivers before shipping).

    **BUT IT IS UNFLOORED, AND THIS ROW ASKS FOR IT FLOORED SHRINK-ONLY. That
    disagreement is live and it is the Review's, not the builder's.** B1
    ordered reporting-only with a reason — *a certificate standing on a fallen
    dependency is a LEGAL state* — and the same order is what the owner's `D27`
    default already settled for the sibling screen: report first, floor once
    the false-positive rate has been measured and written down. The case
    against flooring, in one line: a floor at 3 would turn the NEXT honest
    strengthening into a violation by arithmetic, and this desk armed two of
    today's three itself. The case for it: an unfloored counter is a number
    nobody is accountable to, which is `D27`'s own stated price. **This row
    stays OPEN on its 09-16 date for exactly that question** — the instrument
    is no longer what is owed; the ratchet decision is. The builder does not
    get to settle it by having shipped the easy half.

ROUTED: completeness-audit-2026-09-13-the-cognitive-half-is-the-hole | 2026-09-13 | Review FULL (completeness audit, external reference) | OPEN
    DUE: 2026-09-21 | this desk's own docket: convert the named gaps into
    registered specs or into written refusals, cheapest-first. Date is the
    next FULL sitting plus one; 09-20 already carries 5 rows including the
    `gates-that-measure...` sweep, and stacking both sweeps on one Sunday is
    the mistake this file measures as `piled_on`.
    DUE: 2026-10-11 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. The longest push in this batch and the reason is this row's OWN text: stacking two sweeps on one Sunday is the mistake this file measures as `piled_on`, and 10-04 now carries the gates sweep. The named gaps (voice, body schema, smell, taste, the cognitive half) are not going anywhere and are NOT being converted to refusals by silence — a three-week date is the honest one, and it is ugly on purpose.

    THE METHOD, and why the number below is not reassuring. The audit is
    against an EXTERNAL reference — the human sensory and cognitive inventory
    plus GENERALITY.md's barriers — never against our own documents, because
    an audit inherits every hole in its own standard. `run senses` reports
    10/10 of the inventory spec'd and it is right; that instrument audits the
    SENSORY half against GOAL.md's own list. Nothing in the repo audits the
    COGNITIVE half against anything, and that is where every gap below lives.

    SENSORY HALF — closed, and it stays closed. 10/10 registered, 2/10
    (sight, voice) proven load-bearing. Body schema remains the one zero from
    the owner's 2026-08-09 list that has never moved: 0 own specs, seated
    2026-09-06, arena `UB.14` VOID-FORECLOSED on the venue.

    COGNITIVE HALF — nine capabilities, measured one at a time:
      attention          0 specs. UB.2/UB.8 are architecture-attention (a
                         mechanism inside the model), not the capability
                         (selecting among competing stimuli under limited
                         capacity). Not a declared COVERS domain either.
      working memory     1 spec, ME.8 (PASS) — and it tests that WM SURVIVES A
                         RESTART, not that it holds or manipulates anything.
                         Nothing anywhere asks WM to do cognitive work.
      emotion            2 specs (T2.12 PASS, T3.07 FAIL) against 1,149 lines
                         of EmotionalState.py and a BY DEFAULT seat. UNCHANGED
                         since the 2026-08-09 scar named it. AND — the finding
                         this audit exists to produce — `emotion` IS NOT A
                         DECLARED `COVERS:` DOMAIN AT ALL, so `coverage` cannot
                         report on it and `run senses` cannot see it, because
                         GOAL.md's sensory sentence never listed it. An organ
                         measuring against a stated standard, and the standard
                         omits the thing. Missing outright inside it: any spec
                         that affect changes what he LEARNS or REMEMBERS —
                         affective salience gating consolidation is among the
                         best-established facts in human memory and this
                         project's own biology-as-oracle rule points straight
                         at it.
      imagination        0 specs for the CAPABILITY. LC.* imagine as a
                         world-model rollout; T3.09 (creative loop) is FAIL.
                         Nothing asks whether imagining SOLVES a novel problem
                         without acting, though GOAL.md names "dreaming is
                         training in imagination".
      self-model /       GEN.07 unregistered — but NEWLY NON-ZERO and by
      metacognition      accident: LG.12, registered at 04:14 TODAY, is this
                         project's first metacognition spec. "He speaks
                         correctly or he is silent" is knowing what you do not
                         know. Recorded because a gap that closes unnoticed is
                         as invisible as one that opens unnoticed.
      theory of mind     GEN.02, GEN.03 registered 09-01, both NOT_RUN since.
      teaching           0 specs. GEN.02 is Jack as STUDENT ("a second Jack is
                         a teacher"). Nothing in 246 specs has JACK teach, and
                         GOAL.md's culture claim (generation 3 knows what
                         generation 1 never knew) needs transmission in that
                         direction to mean anything.
      tool use           1 declared spec (CU.6, affordances, 0 PASS); GEN.05
                         ("he cannot make tools") unregistered.
      symbols / number   GEN.11 unregistered, 0 specs.

    GENERALITY.md's BARRIERS — recomputed, not quoted: 14 named, 4 registered
    (GEN.02, GEN.03, GEN.06, GEN.09), 0 RUN, 0 PASS. That is byte-identical to
    the 2026-09-06 reading one week ago. Zero movement on generality in a week
    is the honest headline of this audit and it is not a spec gap — all four
    registered barriers are reachable-on-paper and none has been dispatched.

    A NAMED GAP IS A DECISION; AN UNNAMED GAP IS A BLIND SPOT. Nothing above is
    a demand to build. Several of these Jack may never need. The bill this row
    owes is a WRITTEN DISPOSITION for each — a spec or a refusal with a reason
    — not nine new registrations.

    STALENESS BILL. None: the audit asserts nothing about any spec and moves no
    threshold. The cheapest real repair it points at is declaring an `emotion`
    COVERS domain on T2.12/T3.07, which costs one registry edit and makes the
    gap visible to `coverage` instead of only to this page.

    ### ADDENDUM — COMPLETENESS AUDIT 2026-09-20 (Review FULL). RECOMPUTED, not
    ### quoted. Two corrections to last week and one gap nobody has named.

    **FIRST, TWO CORRECTIONS AGAINST MY OWN FIRST PASS, because an audit that
    hides its errors is worth less than one that never ran.** Grepping the
    registry by keyword told me `pain` had zero specs and that `voice` was
    still the 2026-08-09 zero. Both are WRONG and `run senses` — which audits
    the sensory half against `GOAL.md`'s own list — is right: **pain is `PS.03`
    (*Damage is a signal, not just an ending*), PASS**, and **voice is
    `VO.01`/`VO.02`, with `VO.02` LOAD-BEARING.** Voice was one of the four
    zeros the owner found in an evening on 2026-08-09; it is now one of only
    **two** channels in the whole inventory that clears the standard `GOAL.md`
    actually sets. That is the single best piece of news in this audit and the
    keyword sweep would have buried it.

    **AND THE GAP THAT SURVIVES BOTH PASSES, which is not a missing spec.**
    `run senses` reads `10/10 of the inventory has a registered spec`, and last
    week's audit called the sensory half *"closed, and it stays closed"* on
    that basis. Read the per-channel lines instead of the total and one channel
    does not say what the other nine say:

        pain (nociception)        sensor: PS.03
                                  load-bearing: NO SPEC would prove it

    Every other unproven channel reads `load-bearing awaits: <spec>` — smell
    awaits `SM.02`/`UB.11`, taste `TA.03`, touch `UB.5`, hearing `UB.4`,
    balance `T3.02`, temperature `SH.01`, interoception `UB.11`. **Pain alone
    reads NO SPEC.** `GOAL.md` does not make ablation optional — *"we PROVE each
    one is load-bearing — ablate a sense, something measurable must degrade"* —
    so pain is the one sense in the inventory for which the goal's own standard
    is currently **unreachable by construction**, not merely unmet. A channel
    with no path to its proof is a different object from a channel with an
    unrun spec, and the `10/10` headline cannot distinguish them. **That is the
    thing nobody wrote down**, and it is exactly the shape this audit exists to
    find: an organ measuring against a stated standard where the standard's
    total hides a structural zero inside it.

    Cheapest honest repair, and it is a REGISTRATION not a run: an ablation
    venue for nociception — the natural passenger is `UB.11`, which already
    carries the load-bearing leg for four other channels, and adding pain to it
    costs one registry edit rather than a new rig. **Not ordered here** — this
    row's bill is a written disposition per gap, and this is the disposition:
    NAMED, cheapest-repair identified, owed by the same docket.

    **GENERALITY — second consecutive week of byte-identical zero.** 14 barriers
    named in `GENERALITY.md`, **4 registered** (`GEN.02`, `GEN.03`, `GEN.06`,
    `GEN.09`), **0 implemented, 0 run, 0 PASS**. Identical to the 09-13 reading
    and to the 09-06 reading before it. Three weeks, three identical readings.
    Last week called zero movement *"the honest headline"*; a third identical
    reading makes it a TREND rather than a reading, and it is reported to the
    owner in that form today rather than as a fourth restatement here. The four
    are reachable-on-paper and none has been dispatched — so this is not a spec
    gap and no new registration would touch it.

    **COGNITIVE HALF — unchanged from 09-13 except where this week moved it.**
    `self-model / metacognition` is still non-zero only through `LG.12`, which
    settled **FAIL** on 09-13 — so the gap that closed by accident re-opened by
    measurement, and nothing else has entered. `tool use`, `attention`,
    `symbols/number`, `teaching (Jack as teacher)` and `imagination as a
    problem-solver` are all unmoved at their 09-13 readings. **What DID move is
    not on the cognitive list at all:** the week's four first-ever FAILs
    (`PS.05` far, `PS.06` tiring, `PS.08` heavy, `PS.09` worth-it) are the
    lived-primitive half of `GOAL.md`'s own sentence, and they failed honestly —
    which is the first time this project has measured, rather than assumed, that
    its world does not yet charge for distance, exertion or mass.

---

ROUTED: t108-noise-floor-is-quoted-by-nobody | 2026-09-13 | `445b9e1` (T1.07/T1.08 strengthening, Review items 4a/5) | DISPOSITIONED 2026-09-20 (Review FULL — the docstring is a REAL REQUIREMENT, not an overclaim; the conjunct is ARMED but BOUND to the T1.08 pipeline-repair dispatch and forbidden before it. See RULING below)
    DUE: 2026-09-16 | a design answer owed by the Review: WHICH downstream spec
    should quote the noise floor, and is it worth a GPU re-buy to make it do so.
    Date taken from `review-queue`'s own `next_free_due` (the mechanical answer
    at the time of routing: 09-13 carried 14 promises against a measured
    capacity of 6, 09-16 carried 4), not chosen by hand — 68th audit B7, 3''.
    Nothing is held behind this row; `T1.08` is being re-bought today under its
    other new conjunct and does not wait for this.
    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL), **four days
        late, and the lateness is this desk's**. THE DESIGN IS DELIVERED; what
        this date owes is a one-line DECLARATION by the builder, not a decision
        by this desk and not a dispatch. The docstring's "should be quoted" is
        a REAL REQUIREMENT; the conjunct arms in the SAME DISPATCH as T1.08's
        post-pipeline-repair re-buy, never separately. Full ruling below.
        **PLACED INSIDE THE `DUE:` BLOCK 2026-09-20 ~07:0x, SAME SITTING, after
        the 106th audit's RANK 2 read this row as still OVERDUE. The audit was
        RIGHT and so was the instrument** — my first two attempts wrote this
        date below a `###` heading and then below a non-indented paragraph, and
        `review_queue.py` stops reading a row at the first unindented line. The
        ruling was real both times and its date was invisible both times, so
        `review_queue_violations` correctly refused to move. Same family as the
        09-09 scar where six DUE clauses sat above the line the tool takes:
        **the desk keeps writing the truth in a place the instrument does not
        look, and twice in one morning is not bad luck.** Nothing about the
        ruling changes; only its position does.

**THE ONE-LINE QUESTION.** `T1.08` exists to produce `min_detectable_effect` —
its own docstring says *"the number this produces should be quoted whenever a
later tier claims an improvement"* — and **49 transitive dependents quote it
zero times.** Is the repair to make one of them genuinely compare its effect
against the floor (a GPU certificate re-buy), or is this spec's output honestly
advisory and the docstring's "should be quoted" overclaimed?

**WHY IT IS A ROW AND NOT A COMMIT.** The Review's `FOR THE BUILDER` item 4b
ordered the citation ARMED as a conjunct. I did not arm it, and the reason is
item 4b's sibling, B3 item 1: required setting **>= 1** citing entry, available
range **0 of 49** — computable with zero seeds. Arming it would have dispatched
a GPU run under a bar foreclosed before it ran, which is the `LG.12` defect the
same audit routed as its RANK 3. This row exists so that refusal has an owner
and a date rather than living in a docstring.

**WHAT IS ALREADY DONE, so the desk is not asked for it twice.** The debt is on
the LEDGER rather than in prose: `T1.08` now records `mde_downstream` (49) and
`mde_citing` (0) on every run, REPORTED and deliberately NOT gated, and
`CITE_MARKER` (`CITES T1.08:min_detectable_effect`) is the declaration protocol
that makes the conjunct armable the day a citation is real. The counter was
shown returning 1 against a planted declaration before being trusted at 0, so it
is not a detector wired to nothing.

**NOTE 2026-09-13 ~19:xx UTC (builder, 94th audit B3, annotating — NOT
re-dating).** The candidate named below gained a fact after this row was
written, and it changes the arithmetic of the question rather than the question
itself. `T2.03` declares `depends_on: T1.08`, and `T1.08` settled **FAIL** at
10:05 today under the conjunct armed four hours earlier — so `T2.03` is now
itself a standing certificate that cannot be re-derived (`run status`,
`UNBACKED CERTIFICATES`, shipped tonight in `f38ac1a`). **What follows for this
row: the GPU re-buy it prices as the cost of arming the citation conjunct is
owed ANYWAY, whether or not the conjunct is ever armed**, because `T2.03` cannot
be re-derived until `T1.08` is repaired. The two bills are the same dispatch.
That does not decide the row — the question is still whether the docstring's
*"should be quoted"* is a real requirement or an overclaim — but it removes the
"one `t2_03_*.py` edit plus a GPU re-buy" price tag as an argument AGAINST
arming it, which was the only cost this row had on the table. Nothing re-dated;
09-16 stands.

**THE CANDIDATE, named so the question is concrete.** `T2.03` — *pretrained
vision features beat random features* — is the only PASSing direct dependent and
is exactly the shape the floor exists to bound: an improvement claim over a null.
The other five direct dependents are `T2.01` (FAIL), `T2.02` (VOID), `T2.13`
(not implemented), `T2.14`, `D1.0` (VOID, forbidden to re-dispatch unchanged).

**STALENESS BILL.** Arming the conjunct costs: one implementation edit to
`t2_03_*.py` to compute and declare the comparison (**T2.03's certificate goes
stale — GPU re-buy**), one `_check` edit and claim-field amendment on `T1.08`
(**its certificate goes stale — GPU re-buy, ~0.55 h measured**), and nothing
else: `experiments/registry.py` is in no spec's `IMPL_DEPS` (resolved
mechanically over all test files, 0 hits). Declining costs nothing and leaves
`mde_citing 0` visible on the ledger, which is the honest fallback.

**THE CLASS.** This is the 92nd audit's RANK 3 shape — *a gate that could not
have discriminated* — caught BEFORE the run rather than after it, which is the
first time in the eleven-row history of that class. Worth joining to those rows
if B3 item 3's naming happens.


### THE RULING — 2026-09-20, Review FULL

**THE QUESTION IS ANSWERED YES: `"should be quoted"` IS A REAL REQUIREMENT AND
NOT AN OVERCLAIM.** `T1.08` exists to produce `min_detectable_effect`. A noise
floor that 49 transitive dependents quote **zero** times is not an advisory
number, it is a number in a drawer — and the whole Tier-2-and-above surface of
this ladder is improvement claims, which is precisely the population a floor
bounds. Calling it advisory would be the cheap answer and it would retire, by
redefinition, the only instrument this project has for asking *"is that
improvement bigger than this pipeline's own noise?"* I decline to buy relief
that way. The docstring stands.

**BUT THE CONJUNCT IS NOT ARMED TODAY, AND THE REASON IS NOT COST.** The row
offers one price tag (a `t2_03_*.py` edit plus a GPU re-buy) and the 09-13 note
correctly removes it as an argument, since `T2.03` is an UNBACKED CERTIFICATE
and its re-buy is owed anyway. **The blocker is ORDER, not money.** Editing
`t2_03_*.py` today stales a STANDING PASS whose dependency `T1.08` is FAIL — so
`run_spec` would refuse to re-derive it, and the edit would manufacture a stale
claim that provably CANNOT BE CLEARED by anyone, for an unbounded time, to
express a requirement. Creating unclearable debt is not how a requirement gets
recorded in this repo. That is the same defect in a new costume as arming a bar
against `0 of 49` available: an act whose outcome is decided before it runs.

**WHAT IS ORDERED, and it costs nothing today.**

1. **The citation conjunct is ARMED IN THE SAME DISPATCH THAT RE-BUYS `T1.08`
   AFTER ITS PIPELINE REPAIR — never before, never separately.** One dispatch,
   one bill, no orphan stale claim. `T2.03`'s edit and re-buy ride that same
   dispatch, because `T2.03` cannot be re-derived until `T1.08` passes anyway;
   the two bills the 09-13 note observed are one bill and this is the sentence
   that makes them one act.
2. **The TRIGGER is declared in source now, so it cannot be forgotten when the
   repair lands.** `T1.08`'s spec carries a machine-readable note that
   `CITE_MARKER` arming is OWED at its next PASS-bound re-buy, naming `T2.03`
   as the first citer. This is a declaration, not a gate: it moves no
   threshold, arms no bar, refuses no run, and — per this row's own mechanical
   finding that `experiments/registry.py` appears in **no** spec's `IMPL_DEPS`
   (0 hits over all test files) — it bills **no** certificate if it is written
   there. Write it where it is free.
3. **`mde_citing` stays REPORTED and ungated until a citation is real**, and it
   is to be read as the completeness reading it already is: `mde_citing 0 of
   mde_downstream 49` on every `T1.08` run is the honest fallback the row
   itself named, and it stays visible for exactly as long as the requirement is
   unmet.

**THE STRENGTHENING THIS ROW BUYS, stated so it is not mistaken for a deferral.**
Before today, `"should be quoted"` was a docstring sentence with no owner, no
date and no mechanism — 49 dependents were free to ignore it forever and the
only trace was a counter nobody was accountable to. After today it is a
REQUIREMENT with a named first citer, a named triggering event, and a
declaration in source that fires at that event. The bar goes UP: a future
`T1.08` PASS now owes a citation that a past `T1.08` PASS did not.

**WHAT THIS RULING DOES NOT DO.** It does not design `T1.08`'s pipeline repair
— `heldout_cv_pct` ~40 against a 7.0 bar is a fact about this repo's pipeline
(branch (i) BOTH_ABOVE, n=5, `583a1e9`) and its repair is a separate question
this desk still owes. It does not authorise a GPU dispatch. It moves neither
the 7.0 bar nor any `T2.03` threshold. And it does not make `T1.08` any less
the project's largest blocker: 45 specs and ~29 idle Kaggle hours still stand
behind a pipeline repair that has no design, which is the item the owner is
reading about this week.

    DUE: 2026-09-27 | DISPOSITIONED 2026-09-20 (Review FULL), **four days
        late, and the lateness is this desk's**. THE DESIGN IS DELIVERED; what
        this date owes is a one-line DECLARATION by the builder, not a
        decision by this desk and not a dispatch.
        **RE-PLACED 2026-09-20 ~07:0x, SAME SITTING, after the 106th audit's
        RANK 2 read this row as still OVERDUE. The audit was RIGHT and so was
        the instrument:** the `DUE:` line was written ABOVE a `###` heading,
        where `review_queue.py` stops reading the row — so the ruling was real
        and its date was invisible, and `review_queue_violations` correctly did
        not move. Same family as the 09-09 scar where six DUE clauses sat above
        the line the tool takes: **the desk keeps writing the truth in a place
        the instrument does not look.** Nothing about the ruling changes; only
        its position does.

---

ROUTED: t108-bar-set-from-n1-is-now-the-projects-largest-blocker | 2026-09-13 | `deda088` (T1.08 attempt 3, row ran_at 2026-09-13T10:05:05) | ACTED 2026-09-18 in `583a1e9` (builder — the probe LANDED COMPLETE per §8.5: the §9d stdout-carry dispatch put the colab arm on disk first try, n=5 on a T4 at 0.9221 h inside the ~1.05 h authorisation. The pair: cv_T4 42.786, cv_P100 36.577, both against the 7.0 bar, discordance 1.17 — **branch (i) BOTH_ABOVE fired, read over the full n=5**: `heldout_cv_pct` ~40 is a fact about THIS REPO's pipeline, not the P100 it was first measured on, so the repair is the PIPELINE, never the bar and never the venue. §9b disclosure carried on the record as required: the branch was taken already knowing colab seeds 2/3/4 (0.047148/0.098334/0.035367) from the 09-14 failure records; mean_baseline 0.294928 identical across arms is the same-job arithmetic check. T1.08 stays FAIL (§3); venue selection stays FORBIDDEN (§2); the colab lane was NOT abandoned — §9d's repair worked, and the abandon clause never fired. Artifact: /data/t108_backend_probe.json; the pipeline-repair design is a NEW question and is not smuggled into this stamp)
    DISPOSITIONED 2026-09-14 (Review DAILY — ruled TWO DAYS EARLY and the reason is stated: the row's own second annotation put a perishable 29.18 h on the desk's clock, and three annotations have priced every term of the question at zero GPU, so nothing is bought by waiting. The backend-confound arm pair is AUTHORISED as a PROBE at n=5/backend; it may never buy `T1.08` a verdict; the 7.0 bar does not move; and the desk records that the conjunct it armed on 09-13 committed the category error it had diagnosed four hours earlier. RULING below) — the status field this ACTED stamp replaced, kept verbatim per the never-delete rule; the 18:xx builder slot had appended ACTED as a fifth pipe field at 13:0x (56954e0), which is the MALFORMED violation this line repairs.
    DUE: 2026-09-16 | a design answer owed by the Review: is `heldout_cv_pct`
    40.006 a fact about THIS REPO's pipeline or about the P100 it was measured
    on — and what run settles that WITHOUT being a re-dispatch of an unchanged
    spec. Date from `review-queue`'s own `next_free_due` (09-13 carried 14
    promises against a measured capacity of 6; 09-16 carried 5), not chosen by
    hand — 68th audit B7, `3''`.

**THE EVENT.** `T1.08` re-ran under the conjunct armed four hours earlier
(`445b9e1`, Review `FOR THE BUILDER` item 4a) and returned **FAIL**:
`heldout_cv_pct` **40.006** against `MAX_HELDOUT_CV_PCT` 7.0. Every rig gate
green; the control fired on its own side (`distinct_results` 3, `spread`
0.04209); the OLD conjunct read `snr` 10.1 against its 3.0 bar and cleared, as
the strengthening predicted it would. **The spec did exactly its job.** The bar
is pre-registered and does not move.

**WHY IT IS THE PROJECT'S PROBLEM AND NOT JUST `T1.08`'s.** `run blocked` now
ranks `T1.08` **FIRST at frees 41 / blocks 45**, displacing `T2.01` (frees 35 /
blocks 38) — which is itself now blocked behind it, along with `T2.02` and
`D1.0`. All three declare `depends_on: T1.08`. `UNREACHABLE_BASELINE` 94 -> 97,
growth signed as the builder's. **The `D1.0` attempt-3 dispatch into W37 that
this desk ordered for today is FORECLOSED** — `run_spec` refuses an unsatisfied
dependency (92nd audit B1, shipped this morning, meeting its first real
dispatch). Nothing was worked around.

**THE HONEST CONFOUND, PRE-REGISTERED BEFORE THE RUN AND NOT INVENTED AFTER
IT.** `T1.08`'s own docstring said the 7.0 bar was set from **n=1** (5.717,
attempt 2) where `T1.07`'s sibling bar had n=2, because attempt 1's metrics were
never carried into `history`. Attempt 2 ran on a **Colab T4**; attempt 3 ran on
a **Kaggle P100**. `heldout_std` moved 0.002847 -> 0.023399, a factor of 8.2.
**Backend is confounded with the jump and this row cannot separate them.**

**WHAT THE DESK MUST NOT BE ASKED FOR, and what it may.** It may NOT be asked to
lower 7.0 — law 3 is unconditional, and an unchanged re-dispatch onto a T4
hoping for 5.7 is run-until-pass wearing a hardware argument. What it CAN rule
on is whether a run exists that is genuinely different: a same-kernel,
same-seeds arm pair across both backends would measure the confound directly
rather than re-rolling the verdict, and it is ~0.36 GPU-h per backend (measured,
this row's own attempt). That is a design question with real money behind it —
41 specs — and it is the Review's, not the builder's.

**CORRECTION TO THIS ROW'S PRICE, 2026-09-13 ~14:5x (builder). The date is NOT
touched — this is the 93rd audit B1 precedent (annotate, do not re-date); the
row stays OPEN and DUE 2026-09-16, and re-dating it is the Review's call.**
The *"41 specs"* above was a ranker defect, not a fact, and it is fixed in
`5c444cf` with `T0.36` pinning it. `_terminal_blockers` substituted `T2.01`
away — a settled FAIL with 35 specs behind it — the moment it acquired an
unsatisfied dependency, crediting its whole mass to `T1.08` underneath.
Measured by counterfactual: **repairing `T1.08` alone frees 3** (`D1.0`,
`T2.01`, `T2.02` — the three this row already names as declaring
`depends_on: T1.08`). `run blocked` now reads `T1.08 frees 3 / blocks 45`.

**What this changes for the desk, and what it does not.** It does NOT reduce
the stakes: 45 specs still sit behind `T1.08`, and the ~0.72 GPU-h backend-
confound arm-pair is the same run at the same price. What it changes is what
that money BUYS on its own — **3 specs, not 41** — because the other 42 need
`T2.01` too, and `T2.01` is a settled FAIL with no decided architecture whose
repair path runs through `D1.0`. The honest framing is a **PAIR**: the largest
mass in the project is two repairs deep, and the ranking was collapsing that
into one. A desk pricing this against 41 would be buying a different thing than
the one on offer. `unreachable` is unaffected at 97 (the repair re-labels WHO
blocks, never WHICH specs are stuck — `T0.36` P6).

**WHAT IS ALREADY DONE, so the desk is not asked for it twice.** The row is
committed as the runner wrote it; the floor is raised with the growth named;
`mde_downstream` 49 / `mde_citing` 0 continue to record the separate debt on
`t108-noise-floor-is-quoted-by-nobody` (DUE 09-16, same date, same spec,
deliberately NOT bundled — one asks who quotes the floor, this asks whether the
floor is real).

**THE CLASS, AND IT NOW HAS A TOOL.** Second event in 24 hours where arming or
re-running a gate demoted a certificate and stranded downstream specs
(`T6.03` -> `LF.02` at 06:44 was the first). `run blast-radius <SPEC>` was built
this iteration (`8f3b52a`) and derives the set mechanically, with zero seeds,
BEFORE the edit — the quantity `protocol.BLAST_RADIUS_DECL` has demanded by hand
since the 54th audit and validated as presence-not-truth ever since. Priced
retrospectively, the two sibling conjuncts armed in the SAME commit read
`T1.07 -> none` and `T1.08 -> {D1.0, T2.01, T2.02}`. **The desk should consider
requiring that line in any commit that arms a conjunct on a PASSing spec**, the
way a `VOID-FORECLOSED:` declaration already requires it. That is a contract
change and therefore this desk's, not the builder's.

**SECOND ANNOTATION, 2026-09-13 ~18:2x (builder). THE DATE IS NOT TOUCHED — same
93rd-audit-B1 precedent as the annotation above; the row stays OPEN and DUE
2026-09-16, and re-dating it is the Review's call.** What this adds is the one
term the row prices at zero and that is not zero: **the foreclosed dispatch's
money is PERISHABLE, and the clock on it is three days shorter than the row's
own deadline.**

**THE ARITHMETIC, read from `gpu.Budget`'s own accessor and not from any page**
(the 09-10 scar: `_week()` keys `%Y-W%U`, Sunday-start, and reading the label in
the ISO calendar is how this desk broke a date once already):

    tracker week key   2026-W37          (Sun 2026-09-13 -> Sat 2026-09-19)
    kaggle charged      0.8183 h
    remaining          29.1817 h of 30   — expires end of Sat 2026-09-19

**W37 IS THE POT `D1.0` ATTEMPT 3 WAS ORDERED INTO, AND IT NOW HAS NO BUYER AT
ALL.** Of the **44** specs `ready()` returns today, **13 carry a GPU cost class
and not one is dispatchable**: **6** have never run and are every one PARKED,
PILOT-BLOCKED or `VENUE-UNAFFORDABLE` (`SM.02`, `T2.11`, `T3.10`, `SM.03`,
`DP.04`, `LC.07`); **6** are settled FAIL/VOID under a standing
do-not-re-dispatch directive (`T2.05`, `T2.07`, `T2.15`, `T3.07`, `T4.02`,
`UB.10`) — and `T2.01` and `D1.0` are not even in the runnable set any more, so
they are not among the thirteen at all; and the **thirteenth is `T1.08`
itself** — whose only genuinely-different run is the
~0.72 GPU-h backend-confound arm-pair **this row exists to rule on**. So the
desk's 09-16 decision is not only about 3 specs and 45 blocked ones; it is the
**only** thing standing between a full free allocation and a fourth expiry.

**Why this is an input and not a lever, stated so it cannot be misread as
pressure to rule early.** A dying quota is NOT a reason to manufacture a
dispatch — that prohibition is standing, it is in the builder prompt in four
places, and nothing here weakens it. The 08-29 diagnosis is the frame: these
losses are **inventory, not uptime** (W34 dispatched 0.31 h across 23 unblocked
builder iterations with the full 30 h available), and today is the cleanest
instance the project has yet produced — the loop is awake, the meter has room at
84%, the allocation is fresh, the *builder* is not the constraint, and the
inventory is empty because a conjunct this desk armed at 06:37 settled at 10:05.
That is not a criticism of the strengthening, which did exactly its job and
which this desk should not unwind. It is the price tag the strengthening carries,
arriving on the same page as the decision that owns it, which is the only place
it can be paid.

**NOTHING IS ASKED OF THE DESK BY THIS ANNOTATION.** No new question, no new
date, no re-rank. Three weeks (W32/W33/W34, 61.0 h) were written up as
post-mortems after the hours died; this is the same fact published on day one of
the week instead of after it, which is the whole difference between an input and
an obituary.

**THIRD ANNOTATION, 2026-09-14 ~03:3x (builder). THE DATE IS NOT TOUCHED — same
93rd-audit-B1 precedent as the two above; the row stays OPEN and DUE 2026-09-16.
No bar is moved and nothing new is asked.** This row poses a two-way question —
*is `heldout_cv_pct` 40.006 a fact about THIS REPO's pipeline or about the P100
it was measured on* — and there is a **third term it never names: the
ESTIMATOR.** `heldout_cv_pct` is a sample CV computed from **n=3** with
`ddof=1` (`t1_08_seed_variance.py:220-223, 237`). A sample std at n=3 has a
sampling distribution so wide that it is the largest single term in this row,
and pricing it is arithmetic on the two rows already committed — **zero GPU,
zero seeds, no run.**

**(a) CODE DRIFT IS ELIMINATED, so the row's confound is now genuinely
two-way rather than three-way.** The row says backend is confounded with the
jump; it did not establish that nothing else was. Checked between the two runs'
own commits (`d74e1bd` -> `3d357c4`):

    t1_08_seed_variance.py   116 insertions, 0 deletions — the JOB text is
                             byte-identical; every added line is docstring,
                             `_downstream_ids`/`_citations`, two recorded
                             metrics and one conjunct, all host-side
    UnifiedBrain.py          the ONLY change in the window (`a1c2f9d`) is an
                             extract-function refactor of the grounding
                             fallback tokenizer — a path this JOB never calls
    the task                 `torch.Generator().manual_seed(900)` on CPU, so
                             the tensors are identical on any device
    the init                 `UnifiedBrain(cfg)` is built on CPU and `.to(DEV)`
                             after, so the initial weights are identical too
    the batch order          `i = (step * BS) % (N_TRAIN - BS)` — deterministic

So identical data, identical initial weights, identical batch order. What is
left to differ across venues is CUDA-RNG draws (dropout during `train()`,
sampling inside `generate_actions_flow_matching`) and kernel nondeterminism.
**The row's framing survives the check and is now backed rather than asserted.**

**(b) THE ESTIMATOR'S OWN SPREAD, AND IT IS BIGGER THAN THE EFFECT BEING
ARGUED ABOUT.** With `s^2(n-1)/sigma^2 ~ chi^2_2`, a 3-seed CV drawn from a
pipeline whose TRUE cv is exactly the 5.717 the bar was set from lands, 95% of
the time, anywhere in:

    [0.92%, 11.00%]        a 12x span — and it straddles the 7.0 bar

And the discordance this row is built on is correspondingly weaker than it
looks: **P(two 3-seed CVs differ by >= 6.998x | SAME pipeline, nothing changed)
= 4.0%** (Monte Carlo 0.0402 at 200k trials; closed form `2/(1+r^2)` = 0.0400 —
they agree to four decimals, which validates the arithmetic, not the
normality assumption below). That is unlikely, so a venue effect stays the
leading hypothesis — but it is a 1-in-25 event, not the impossibility a 7x
ratio reads as, and **one 3-seed reading per venue cannot do better than that
by construction.**

**(c) THE BAR'S FALSE-FAIL RATE WAS NEVER PRICED, AND IT IS 22.6%.** This is
the finding the desk most needs on 09-16 and it is about the BAR, not the
confound. `MAX_HELDOUT_CV_PCT` 7.0 sits **1.224x** above a single n=3
measurement. Against the estimator above:

    true cv of the pipeline   P(an HONEST 3-seed run reads > 7.0 and FAILS)
        3.0%                       0.4%
        4.0%                       4.8%
        5.0%                      14.1%
        5.717%  (the bar's own    22.6%
                 source value)
        7.0%                      37.1%

**If the pipeline's true cv is exactly the number the bar was derived from,
better than one honest run in five FAILS.** And the rate is a steep function of
a quantity nobody knows — both readings are n=3, so the true cv is unestimated.
**Seeds do not rescue this cheaply, because the tightness is the headroom and
not the sample:** at true cv 5.717, n=20 still leaves 7.5% (2.4 GPU-h) and n=30
leaves 4.2% (3.6 GPU-h), at the measured 0.12 GPU-h/seed. The 1.224x headroom
was chosen *by analogy* to `T1.07`'s 6.0-against-4.931 (the docstring says so:
*"deliberately the same ratio"*), and the two are different statistics with
different sampling distributions, so the analogy transported a ratio and not a
false-fail rate.

**(d) THE ARM-PAIR THIS ROW ALREADY PROPOSES IS ADEQUATE — this CORRECTS my own
first reading of it, which was that n=3-per-backend could not settle anything.**
It can, marginally, and n=5 makes it comfortable:

    seeds/backend   P(>=6.998x | same pipeline)   power to SEE a real 2x   cost
        3                4.0%                          92.2%             0.72 h
        5                0.2%                          98.2%             1.20 h
        8                0.0%                          99.7%             1.92 h

So the desk's own design holds, and **1.20 GPU-h against 29.18 free hours
expiring Sat 09-19 buys alpha 0.2% instead of 4.0%.** That is a cheap
strengthening of a proposal this row already owns, not a new proposal.

**(e) `T1.07`'s SIBLING BAR SHARES THE CONSTRUCTION AND IS UNPRICED. FLAGGED,
NOT FIXED, AND DELIBERATELY NOT GIVEN A NUMBER.** `spread_ratio <= 6.0` was
armed in the same commit from a single measurement (4.931) at the same ~1.22x
headroom, and `T1.07` is **PASS** — a Tier-1 certificate. Its statistic is a
max/min over LR arms, **not** a sample std over seeds, so its sampling
distribution is not the chi-square above and **the 22.6% MUST NOT be
transported onto it.** Computing it needs the per-arm seed noise, which nobody
has measured. What is transportable is the *class*: a bar set at k x one
observation of a sample statistic, with no false-fail rate computed. Whether
that is worth a row is the desk's call; it is recorded here rather than routed
separately because it arrived as this row's arithmetic.

**METHOD AND ITS ONE ASSUMPTION, stated rather than buried.** All figures are
Monte Carlo (120k-400k trials, seed 20260914) simulating the runner's own
estimator — `s = sqrt(sum((x-mean)^2)/(n-1))`, `cv = 100*s/mean` — cross-checked
against closed forms where they exist. **The assumption is that the held-out
metric is normal across seeds**, which is unverified and unverifiable from two
rows. The likely violation is right-skew (held-out MSE is positive and
occasionally has a bad seed), and skew puts more mass in the upper tail of `s`,
so **22.6% is if anything an UNDER-estimate.** That direction is stated because
it is the one that matters: the error runs against the bar, not for it.

**NOTHING IS ASKED OF THE DESK BY THIS ANNOTATION.** No new question, no new
date, no re-rank, and — explicitly — **no suggestion that 7.0 should move.**
Law 3 is unconditional and a false-fail rate is not a licence; a bar with a
known false-fail rate is strictly better governed than the same bar with an
unknown one, and which of the two repairs (more seeds, more headroom, a
different statistic, or accept the rate) is correct is exactly the design
question already dated 09-16.

**ADDENDUM (e'), 2026-09-14 (builder, zero GPU): `T1.07`'S SIBLING BAR IS NOW
PRICED — AND (e) POINTED AT THE WRONG CONJUNCT AND THE WRONG NOISE TERM.**
(e) said the rate "needs the per-arm seed noise, which nobody has measured."
**No run of `T1.07` can ever produce that number.** `SEED = 0` is a module
constant inside its `JOB` and the registry declares `seeds = 1`, so the
statistic is not SAMPLED across seeds at all. The only variation this spec has
ever exhibited is **VENUE**, and two observations of it are already on the
ledger. Attempts 2 and 3 (P100, `e29bd82` 08-14 and `445b9e1` 09-13 — a month
and a code change apart) agree on **every metric and every control metric to
the recorded digit**, so within-venue determinism is demonstrated on one of the
two venues; the T4 has one observation. Code drift is eliminated by the same
method this row used for `T1.08`: the `t1_07` diff across `1a69db6..e29bd82` is
4 insertions / 3 deletions, all of it the `/content/` -> `JACK_OUT` artifact
contract, and `UnifiedBrain.py`'s two hunks are both in the PRETRAINED vision
path that `use_pretrained_vision=False` never reaches. ("Venue" is
Colab/T4/sm_75/Colab-torch vs Kaggle/P100/sm_60/torch 2.5.1+cu121 — three
things confounded, not separable from two rows.)

| conjunct | live (P100) | room | venue moved | % of log-headroom |
|---|---|---|---|---|
| `worst_lr_advantage >= 1.15` | 1.3800 | x1.200 | x1.063 | 34% |
| `spread_ratio <= 6.00` | 4.9310 | x1.217 | x1.146 | **69%** |
| `reference_advantage >= 1.15` | 7.6050 | x6.613 | x1.000 | 0% |
| `absurd_advantage < 1.15` (CONTROL) | 0.9162 | x1.255 | **x99.59** | **2024%** |

**The bar (e) flagged is the third-thinnest of the four. The binding one is the
CONTROL, and it is not a claim conjunct at all.** `T1.07`'s own docstring says
that if `lr=1.0` clears `MIN_BEAT_MEAN` then "the bar is too low to discriminate
anything and the result is void" — and on the venue the live certificate was
bought on, `lr=1.0` **does not diverge** (`absurd_diverged` False) and lands at
0.9162x mean-prediction, **1.255x from making this spec's own guard vacuous**.
On the other venue it read 0.0092. Its margin is **79.3x smaller than the one
venue change we have on record.** The one venue-invariant arm is the plain-MLP
reference (7.605 -> 7.605, unchanged to four significant figures): the task, the
data and plain Adam reproduce across both venues, and everything that moves is
inside the `UnifiedBrain` training path.

This is a bound on what is KNOWN, not a rate: n=2 supports no probability, the
direction happened to run toward the bar, and a third venue could run the other
way. **Nothing is asked and nothing moves** — `MIN_BEAT_MEAN` is pre-registered
and law 4 is unconditional. Two things the 09-16 disposition may want, both
computed: pricing `T1.07`'s seed term costs ~0.465 GPU-h per seed (attempt 3 ran
5 trainings in 1673 s), so k=5 is 2.33 h against 29.18 free hours expiring 09-19;
and the drift check above had to be done **by hand**, because `T1.07` declares no
`IMPL_DEPS` and a `UnifiedBrain.py` change therefore does not stale its
certificate — the gap `T0.35` already counts, cited rather than re-routed.
Recorded on this row, not routed separately, because it arrived as this row's
arithmetic and the desk's drain reads UNBOUNDED (`D28`). The docstring carries
the same numbers (`run amend T1.07 --doc-only`, `3194d14d -> 7fd74ff2`).

**ADDENDUM (e''), 2026-09-14 (builder, zero GPU, zero seeds): THE SWEEP (e')
LEFT OPEN IS NOW SCOPED, AND THE SCOPING CLOSES IT — a dedicated backward
control-margin sweep over the standing ladder is NOT a unit.** (e')'s closing
sentence ("every spec in this repo has a control conjunct and none carries a
reachability block") implies pricing ~100 certificates by hand. Scoped
mechanically from the ledger's `control_metrics` field before any slot was
spent on it:

- **POPULATION.** 108 standing PASS rows: 2 declare no control BY DECISION
  (`T0.01`/`T0.10`, 52nd audit B5); 9 record NO continuous control magnitude
  at all — every value a flag or 0/1 count (`T0.09`, `T0.11`, `T0.12`,
  `T0.17`, `T0.23`, `T0.24`, `T0.33`, `T0.34`, `T0.35`, all Tier-0
  planted-sabotage catches where a binary IS the design — but note nothing
  under those flags can ever be priced from the ledger); 97 record continuous
  magnitudes. 1,170 control-metric entries in total.
- **THE MOVEMENT HALF OF (e')'s METHOD IS FREE, AND IT IS EXHAUSTED.** 840
  (spec, control-metric) pairs carry >=2 PASS-observation values. **774 of
  840 (92.1%) are FROZEN** — identical to the recorded digit across every
  observation. Of the 66 that move, **exactly ONE moves >=x10: `T1.07
  absurd_advantage`, x99.59 — the finding (e') already made by hand.**
  Runner-up is x5.75 (`LC.02` wall-clock timing stds on a shared 4-core box),
  then instrument counters tracking repo growth at <=x4. Recorded data holds
  no second `T1.07`, and the reason is structural rather than reassuring:
  movement accrues only where venue or seed actually varies, re-buys are
  same-venue and deterministic, and `T1.07` is near-unique in holding
  observations from two venues. **A frozen pair is absence of evidence, not
  evidence of stability.**
- **THE TRAP FOUND ON THE WAY, for whoever ever builds a screen here:
  condition on verdict status.** Unconditioned, the scan's loudest "drifts"
  were `LG.00 verdicts_missing` 0 -> 623.7 and `PS.02 control_r2` flipping to
  0.0 — every such observation is a **VOID row, i.e. the control FIRING.** An
  unconditioned movement scan reads the system working as drift. Likewise
  check design intent before flagging a value: `T1.06` records `final_loss`
  NaN on every PASS because its control at lr=1e4 MUST go non-finite — the
  NaN is the control's success signature.
- **THE MARGIN HALF DOES NOT MECHANISE, and no instrument was built.**
  Distance-to-bar needs the bar; bars live inside each `_check`; pairing
  recorded metrics to bars is the exact parsing problem `D27`'s prototype
  measured failing at 104-of-107 flagged. That fork is the owner's
  (decide_by 09-20) — this scoping is appended to `D27` as evidence, not
  pre-empted here.

**What remains is the FORWARD rule already in LESSONS (09-14): a reachability
block written for a claim gets one for the control, the control's number
first when it is the thinner — applied at registration and strengthen time,
priced per spec at its next natural touch, never as a dedicated pass over 97
certificates whose recorded data holds nothing left to find.** Nothing moves,
nothing is asked of the desk, no row routed (drain UNBOUNDED, `D28`).

---

## RULING, 2026-09-14 ~07:0x UTC (Review, DAILY). The row is DISPOSITIONED. `DUE:` moves 09-16 → 09-16 *unchanged as a date* but the answer is delivered now; the row goes ACTED when the probe lands.

**Ruled two days early, and why that is not the dying-quota pressure the second
annotation pre-emptively refused.** That annotation was right to refuse it: a
dying quota is not a reason to manufacture a dispatch. But the thing it forbids
is manufacturing *a run*, not delivering *a decision* — and the decision was
ripe the moment (c) and (d) landed. Three annotations have priced the estimator,
eliminated code drift, computed the bar's false-fail rate and corrected their own
first reading, **all at zero GPU and zero seeds**. There is nothing a 09-16
sitting would know that this one does not. Ruling on the day the inputs are
complete, rather than on the day the calendar says, is the only difference
between an input and an obituary — the row's own phrase, applied to the row.

### 1. The answer to the question as asked, and it is neither of the two offered terms

*"Is `heldout_cv_pct` 40.006 a fact about THIS REPO's pipeline or about the P100
it was measured on?"* — **the honest answer today is that it is a fact about the
ESTIMATOR first, and the row's own (b) is what establishes that.** At n=3 with
`ddof=1`, a pipeline whose true cv is 5.717 produces readings anywhere in
[0.92%, 11.00%] 95% of the time. **40.006 is not in that interval and is not
close to it.** So (b) does NOT explain attempt 3 away, and that is the finding
the desk takes from it: the estimator is wide enough to make the *bar* a
lottery, and still not wide enough to make the *observation* one. Something
real moved. The two candidates remain venue and pipeline, and two n=3 readings
cannot separate them by construction.

**This matters because the 22.6% has an attractive misreading and the desk is
refusing it in writing.** 22.6% is the rate at which an honest run from a
5.717-true-cv pipeline reads over 7.0. It is a fact about the BAR's governance.
It is **not** a defence of attempt 3 and may not be cited as one. `T1.08` is
FAIL for a reason no sampling argument reaches.

### 2. AUTHORISED: the backend-confound arm pair, as a PROBE, at n=5 per backend

The row asks what run settles the confound without being a re-dispatch of an
unchanged spec. **Answer: not a `T1.08` attempt at all.** It is a probe, on the
`D1.0` twin-spread (`8624fa0` → `8608986`) and `SM.03` F2 (`8b6480a`)
precedent — pre-registration commit first, the read fixed before the number
exists, no ledger row for `T1.08`, no verdict bought.

- **n = 5 per backend, 1.20 GPU-h** of W37's 29.18, not n=3. The row's (d)
  priced the upgrade itself: alpha 4.0% → **0.2%**, power 92.2% → 98.2%, for
  0.48 h. A 1-in-25 false discordance is not good enough to retire a
  three-week-old confound on, and the desk is not going to be asked this
  question twice.
- **Same kernel, same seed list, same commit, both backends.** The probe's
  whole content is that everything except the venue is pinned, and (a) has
  already demonstrated that the JOB text, the task tensors, the initial weights
  and the batch order are identical across devices.
- **`SEEDS` inside the JOB**, not the registry's `seeds` field — the 09-14
  builder sweep established that this is how `T1.08` already multi-seeds and
  that the registry route has no guard.

### 3. THE PROHIBITION, which is the load-bearing half of this ruling

**The probe may not buy `T1.08` a verdict, on any branch.** `T1.08` stays FAIL
until a run *of the registered spec* clears 7.0. Specifically forbidden, and
named because it is the move the probe's own result will make tempting:
**dispatching `T1.08` to a T4 because the probe reported that the T4 reads
lower.** That is run-until-pass wearing a hardware argument, this row named it
first, and the ruling does not create an exception to it. The venue a
certificate is bought on may never be selected after seeing which venue is
kind.

### 4. THE READ, PRE-REGISTERED HERE BEFORE ANY NUMBER EXISTS

Let `cv_T4` and `cv_P100` be the n=5 readings.

- **(i) BOTH > 7.0.** The noise is the repo's. The FAIL is a fact about the
  pipeline, and the repair is the pipeline — never the bar, and never the
  venue. `T1.08` stays FAIL and the ladder's honest statement becomes *"every
  Tier-2+ claim on this pipeline is made against a held-out seed spread this
  large"*, which is item 6 below.
- **(ii) BOTH < 7.0.** Attempt 3's 40.006 was neither venue nor pipeline but a
  tail draw at n=3 — a reading the row's own (b) says is very unlikely, which
  is exactly why this branch must be pre-registered rather than reached for.
  The repair is then **the estimator, not the bar**: `T1.08`'s registered seed
  count rises and the spec is re-run at the higher n. **That is a different run
  and its dispatch is legal.** Raising n is NOT a weakening and the desk states
  why: more seeds move the sample CV toward the truth in *both* directions — if
  the true cv is above 7.0 they make the FAIL more certain, not less. The
  number of seeds is **fixed in branch (ii)'s pre-registration commit, before
  the probe's numbers are read**, so it cannot be chosen to buy a pass.
- **(iii) THEY SPLIT.** The venue term is real. Then `T1.08` cannot certify a
  venue-invariant noise floor from one venue at all, and its claim is
  venue-scoped and must say so in the spec text. A venue-scoped noise floor is
  a smaller claim than the one the docstring makes today, and shrinking a claim
  to match what was measured is the one direction this desk is always allowed
  to go.

### 5. THE BAR DOES NOT MOVE — and the desk owns the error it made arming it

`MAX_HELDOUT_CV_PCT` 7.0 is pre-registered and law 3 is unconditional. Nothing
here touches it, and the 22.6% is not a licence. But the row is entitled to the
rest of the sentence, so here it is in the first person.

**On 2026-09-13 at 06:37 this desk wrote, of `T1.08`'s OLD gate:** *"A
measurement spec that FAILS when its own toy task has a small effect is a
category error: the correct output of a noise-floor measurement with large
noise is 'the noise floor is large.'"* **Four hours later it armed a conjunct
that FAILS when the measured noise is large.** The quantity is better — the new
gate at least reads the thing the spec exists to produce, which the old one
never did — but the shape is the error I had just diagnosed, one term over, and
it was armed at 1.224× one observation of a sample statistic by analogy to a
sibling bar with a different sampling distribution. The desk does not get to
diagnose a class on Sunday morning and commit it on Sunday afternoon without
saying so.

**And the conjunct still stays, for a reason that is not face-saving.** It
worked. It surfaced, in one run, that the pipeline 45 specs stand on has a
held-out seed spread of 40% on the venue we actually run on — a fact that was
true yesterday, was true for the thirty-six days before that, and that **49
transitive dependents were quoting zero times.** The gate did not misfire. It
reported. What the ladder got wrong was treating the report as a blocker rather
than as a number to be quoted.

### 6. THE TWO ROWS ARE ONE DECISION, contrary to their deliberate unbundling

`t108-noise-floor-is-quoted-by-nobody` (DUE 09-16) asks who quotes the floor;
this row asks whether the floor is real. The builder split them on purpose and
the split was right *at routing time*. It is not right now, and the reason is
in that row's own 94th-audit-B3 note: `T2.03` declares `depends_on: T1.08`, so
its certificate cannot be re-derived until `T1.08` is repaired, **so the GPU
re-buy both rows price is the same dispatch.** The desk's position, carried to
09-16 rather than ruled here because that row has its own owner and its own
date: **`T2.03` is the citing spec**, and `CITE_MARKER` arms on it the day
`T1.08` has a live floor to cite. Recorded here so the 09-16 sitting does not
have to re-derive it.

### 7. (e') AND (e'') ARE READ, AND THE TELL THEY SET IS ANSWERED

The builder wrote, twice: *"If the 09-16 disposition rules on `spread_ratio`
without the control's x1.255, the findings did not arrive."* They arrived.

**The binding margin on `T1.07` is the CONTROL's, not `spread_ratio`'s.**
`absurd_advantage < 1.15` reads 0.9162 with x1.255 of room, against a measured
venue movement of **x99.59** in that same metric — 79.3× larger than the room.
`T1.07`'s own docstring says that if `lr=1.0` clears `MIN_BEAT_MEAN` the result
is VOID. So a standing Tier-1 PASS is **one venue draw from being void by its
own text**, and the desk is recording that rather than discovering it on a
Sunday in November. It is a bound, not a rate — n=2 supports no probability —
and no bar moves.

**What the desk orders instead, at zero GPU: `T1.07` gains an `IMPL_DEPS`
declaration.** It declares none today, which is why a `UnifiedBrain.py` change
does not stale its certificate and why the drift check had to be done **by hand,
twice, in two days** (09-13 and 09-14). That is `T0.35`'s counted gap arriving
on a specific spec with a specific cost. Adding `IMPL_DEPS` will stale `T1.07`
and owe a ~0.47 GPU-h re-buy against 29.18 free hours — **and that is the point,
not the objection**: a certificate that cannot be staled by the file its claim
runs through is not being governed, and the re-buy is the honest price of
saying so. Strictly a strengthening; no bar touched in either direction.

### 8. WHAT IS ORDERED, in one place

1. Pre-registration commit for the backend-confound probe: n=5/backend, seed
   list, fixed read (branches (i)/(ii)/(iii) above, quoted verbatim), **no
   `T1.08` ledger row**. No dispatch in that commit.
2. Dispatch the probe. 1.20 GPU-h of W37 (29.18 free, expires Sat 09-19).
3. Report the pair. `T1.08` stays FAIL regardless; the branch selects the
   repair, and branch (ii)'s seed count is fixed before the numbers are read.
4. `T1.07` gains `IMPL_DEPS`, is staled by it, and is re-bought (~0.47 GPU-h).
5. The row goes **ACTED** with the probe's commit. It is not ACTED by this
   ruling — a design is not an execution, and this desk has broken thirteen
   promises this week by forgetting that distinction in the other direction.

### 9. ADDENDUM 2026-09-15 (Review DAILY) — the colab lane's cause is SETTLED on disk, by the instrument built to settle it, and it sat unread for nineteen hours

**What happened, in receipts.** The probe was pre-registered (`1652a62`) and
dispatched on 09-14. The **kaggle arm landed**: `jack-ladder-1789370135`, P100,
n=5, 0.5607 h — `heldout_cv_pct` **36.577**, mean 0.06355, sd 0.023246, effect
0.23138, `snr` 9.95. The **colab arm ran twice and retrieved nothing both
times**: `ladder-1789373334` (1.0277 h) and `ladder-1789381054` (1.0832 h).
**2.1109 GPU-h charged for zero retrieved results.**

**The 09-14 LESSON named two causes and deliberately refused to choose between
them; the head capture shipped at `521d33e` was built to adjudicate them; it
did, on the very next failure, and the answer is unambiguous.**
`/data/t108_backend_probe.json`'s second failure record opens
**`stdout_head='JACK_OUT /content\nREPO 521d33e...'`**. The job wrote to
`/content` and `run_on_colab` fetched `/content`. **Cause (1) — the job wrote
somewhere the fetch did not look — is ELIMINATED. Cause (2) — the kept download
session no longer holds the run VM's filesystem — is the cause.** The LESSON's
own standing instruction is therefore in force and is not a judgement call any
more: **do not edit the fetch path**; recover the artifact from stdout, or
abandon the lane. One 400-character capture, added for free, closed a question
that a third GPU dispatch would not have closed.

**9b. THREE OF THE FIVE COLAB SEEDS ARE ON DISK AND MUST BE TREATED AS READ.**
The same failure records carry `stdout_tail`, and the tail is 400 characters of
the result array: seeds **2, 3, 4** heldout complete — **0.047148, 0.098334,
0.035367** — under a `mean_baseline` of **0.29492783546447754** identical to the
kaggle arm's (0.085438 + 0.20949 = 0.294928), which is the arithmetic check that
both arms ran the same job. Seeds 0 and 1 scrolled past the head and are lost,
twice. **This is a DISCLOSURE, not a reading.** The read pre-registered in §4 is
over n=5 and may not be taken on a three-seed subset selected by what a
truncation happened to preserve — that is the seed-selection form of the
venue-selection prohibition in §2. But the sitting that finally takes the branch
will take it already knowing three of its five numbers, and **it must say so on
the record when it fires**. Pre-registration survives disclosure only if the
disclosure is written down.

**9c. One observation that is NOT a branch read.** The kaggle arm's n=5
`heldout_cv_pct` **36.577** stands beside attempt 3's n=3 **40.006** on the same
backend. That is a within-backend reproduction and it removes "40.006 was an n=3
fluke" from the table. It says nothing about the between-backend contrast, which
is the entire question this probe exists to answer, and no branch is taken here.

**9d. WHAT IS AUTHORISED NOW — and what is forbidden.** The probe's
authorisation in §2 was **1.20 GPU-h**. As executed it has charged **2.6716 h**
(kaggle 0.5607 + colab 2.1109) — **2.2× its authorisation, with every hour of
the overrun in a lane that returned nothing.** A third dispatch under the same
retrieval mechanism is the third identical spend and is **FORBIDDEN**.

AUTHORISED instead, ~1.05 GPU-h against **26.51 free expiring Sat 2026-09-19**,
strictly in this order:

  (a) **Zero GPU first.** The job prints its result JSON to stdout on one
      delimited line (`JACKRESULT {...}`), and the failure record captures the
      WHOLE stdout rather than a 400-char tail. **The tail bound is what lost
      seeds 0 and 1** — the 09-14 repair sampled both ends of the stream and was
      still too narrow to carry the payload it had correctly decided to recover
      from. Widening the sample was the repair; carrying the artifact is.
      **No fetch-path edit lands under this authorisation.**
  (b) **Then** one colab dispatch, same commit, same seed list, n=5.

**If (a) cannot be made to work, the colab arm is ABANDONED** and the probe
reports as a single-backend reading with that stated plainly. A smaller finding
honestly labelled costs less than a fourth charge on a lane that has already
paid 2.1 h for nothing. **`MAX_HELDOUT_CV_PCT` 7.0 does not move under any
branch of this addendum, and `T1.08` stays FAIL.**

## ROUTED 2026-09-13 (builder, 93rd audit B3): `waits-on-declared-field` — six of the fourteen rows that came due today share one root, and the only place that fact lives is prose

ROUTED: waits-on-declared-field | 2026-09-13 | 93rd-audit-B3 | DISPOSITIONED 2026-09-19 (Review DAILY — ADOPT the cheaper variant, STRENGTHENED: `WAITS-ON:` is declaration-only and buys no exemption, and the grouped line prints only when every live row on the date carries an EXPLICIT declaration, with `WAITS-ON: none` permitted as that declaration; design only, the builder implements and re-buys T0.31)
    DUE: 2026-09-21 | RE-DATED 2026-09-19 (Review DAILY) BECAUSE THE DEBT
    CHANGED HANDS, not because it was missed again. The 2026-09-17 date broke on
    this desk and the break stands in the record. What this row owed was a
    GRAMMAR RULING owed by the Review, and that ruling is delivered below
    (DISPOSITION 2026-09-19 — adopt the cheaper variant, strengthened with
    `WAITS-ON: none`). What remains is an IMPLEMENTATION owed by the builder,
    which is a different debt with a different owner, and dating it to the day
    the design landed would be dating work nobody could yet have done. Date is
    `review-queue`'s own `next_free_due` print (2026-09-21, 3 live rows against
    the measured capacity of 6) — taken from the tool, never chosen by hand, and
    never onto a day already at capacity. The design half of this row is
    DISCHARGED; only the build is outstanding. ORIGINAL TEXT FOLLOWS, unchanged.
    | a GRAMMAR decision, and therefore this desk's: may a live
    non-`HELD` row declare `WAITS-ON: <row id>`? Taken from `review-queue`'s own
    `next_free_due` rather than chosen by hand. PROPOSED, deliberately NOT
    implemented — the 93rd audit ordered it in that shape ("Propose it; do not
    implement it ahead of B2"), B2 shipped in `6ddd09c`, and this changes the
    format the Review itself writes in, so imposing it unilaterally would be
    the builder editing the desk's own grammar.
    BILL: zero certificates if refused. If adopted: `experiments/review_queue.py`
    is `T0.31`'s only `IMPL_DEPS`, so implementing it stales and re-buys `T0.31`
    (~1.6 s, and it would arrive as a strengthening, 18 -> 19 properties).
    DUE: 2026-09-25 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. DEBT HAS CHANGED HANDS AND THE DATE NOW SAYS SO. The grammar ruling this row owed was delivered 09-19 (adopt the cheaper variant, strengthened with `WAITS-ON: none`); what remains is an IMPLEMENTATION plus a T0.31 re-buy, owed by the BUILDER, which is a different debt with a different owner. Dated onto the first sitting the builder can reach it now that the launcher is alive again.

**THE MEASUREMENT, which is the whole case.** Fourteen live dated rows came due
on 2026-09-13 against a measured capacity of six. **Six of them share one root**
— the `w0-too-shallow` W1 design — and they say so ONLY in body prose, in
sentences like *"in light of the `w0-too-shallow` design"*. `review_queue.py`
reads DECLARED fields and never prose, deliberately and correctly (`champions.py`
learned on `901f7fc` what a regex over prose costs). So the pile histogram can
print *"14 rows"* and cannot print *"14 rows, 6 of them behind one root"* —
which is a different and much more actionable sentence, because six rows behind
one decision is ONE sitting's work misread as six, and the other eight are the
real overflow.

**THE PROPOSAL.** Allow a third optional body line beside `DUE:` and
`BLOCKED-BY:`, in the same idiom:

    WAITS-ON: <another row id> | why this row's answer depends on that one

**Why not `BLOCKED-BY:`, which already exists.** `BLOCKED-BY:` buys
ageing-exemption, and that is exactly wrong here: **these rows SHOULD age.**
They are live promises with dates, and the desk is answerable for them on those
dates whether or not their root has been decided. `WAITS-ON:` would buy nothing
at all — no exemption, no re-dating, no change to OVERDUE or STALE. It is
declaration-only, so that a reading which already exists can group by it.

**What it would buy, stated as the one line it makes printable:**

    14 rows due on or before 2026-09-14, 6 of them behind one root
    (`w0-too-shallow`) — the pile is 9 decisions, not 14.

**WHAT THIS DESK SHOULD DECIDE, and the honest case against.** Adopting it
means every future router owes a judgment about coupling, and a declared field
that is optional and unenforced drifts into being written by whoever remembers
— at which point the grouped count is confidently wrong rather than absent,
which is worse than today. The builder's own view, offered and not acted on:
that risk is real and is the reason this is proposed rather than shipped. A
cheaper variant exists if the desk prefers it — `WAITS-ON:` permitted but the
grouped line printed only when EVERY row in a pile declares one, so a partial
adoption prints nothing instead of a wrong number.

**DISPOSITION 2026-09-19 (Review, DAILY) — ADOPT THE CHEAPER VARIANT, WITH ONE
STRENGTHENING THAT CLOSES A HOLE IN IT.** Answering the grammar question first,
because it is the one that was asked: **yes, a live non-`HELD` row may declare
`WAITS-ON: <row id>`.** The proposal's central judgment is right and is the
reason this is adoptable at all — coupling and ageing-exemption are different
things, and `BLOCKED-BY:` conflates them. These rows *should* age. A desk that
owes six answers behind one root still owes six answers on their dates; what it
does not owe is six sittings. `WAITS-ON:` buys **nothing** — no exemption, no
re-dating, no change to OVERDUE or STALE — and that emptiness is the feature.

**The builder's case against is correct and is why the base proposal is
REFUSED.** An optional, unenforced declared field gets written by whoever
remembers, and a grouped count assembled from partial declarations is
*confidently wrong*, which is strictly worse than the absent line we have today.
The cheaper variant is the right shape: make the reading refuse to print rather
than print a number it cannot stand behind.

**But the variant as written cannot ever fire, and that is the hole.** "Print
only when EVERY row in a pile declares one" is unsatisfiable for a pile
containing a genuinely independent row — an uncoupled row has no root to name,
so it can never declare, so the line never prints, so the whole feature is
inert. **The repair: permit `WAITS-ON: none` as an explicit declaration of
independence, and gate the grouped line on every live row for that date carrying
an EXPLICIT declaration — a row id or `none`.** Partial adoption still prints
nothing; full adoption now prints something.

**Why this is stronger than what was proposed, in the project's own idiom.** It
converts "did the router remember?" — unanswerable — into a *completeness*
question the instrument answers mechanically, which is the same move
`experiments/decisions.py` makes when it refuses to guess which `FOR THE OWNER`
items are asks and demands a written `NO-DECISION:` instead. **Silence is
reported; exemption is written down.** The identical rule, applied to a second
desk file. A router who declines to judge coupling now leaves a visible hole in
a printed reading instead of a silent gap in a count.

**What the builder implements, and what it may not do.**
- `WAITS-ON: <row id> | <why>` and `WAITS-ON: none | <why not>` as a third
  optional body line beside `DUE:` and `BLOCKED-BY:`, same idiom.
- **Declaration-only.** It may not touch OVERDUE, STALE, ageing, `next_free_due`,
  the disposal-rate measurement or the capacity histogram's row counts. If
  implementing it changes any number already printed other than by ADDING the
  grouped line, the implementation is wrong.
- A `WAITS-ON:` naming a row id that does not exist is a VIOLATION, in the same
  class as the MALFORMED-fields check — an undeclared coupling is a gap, but a
  coupling declared against a corpse is a false statement.
- The grouped line prints per due-date, gated as above, and when the gate is
  unmet it prints **why it is unmet** (`n of m rows undeclared`), never nothing
  at all. A reading that is silently absent is how this row's own problem
  started.
- **BILL, as routed:** `experiments/review_queue.py` is `T0.31`'s only
  `IMPL_DEPS`, so this stales `T0.31` and it is re-bought in the same motion
  (~1.6 s). It arrives as a STRENGTHENING — 18 -> 19 properties minimum, and the
  corpse-reference violation is a 20th if it is asserted separately. No
  threshold moves; nothing is weakened.

**The cost, stated rather than buried.** Every future router now owes a judgment
about coupling on every row it writes, including the judgment "this one is
independent". That is real work and it is being imposed on this desk by this
desk. It is worth it because the alternative measured itself: fourteen rows came
due against a capacity of six and nobody could tell — including the desk that
wrote all fourteen — that it was nine decisions and not fourteen. **Reversal:
delete the reading; the declared lines are inert prose and harm nothing if the
grouped line is never printed again.**

ROUTED: so10-tie-break-hands-the-seat-to-an-ineligible-arm | 2026-09-13 | `498b8a2` (SO.10 attempt 1, FAIL) | OPEN
    DUE: 2026-09-17 | two design answers owed by the Review: (1) which of the
    two measured, tied, ELIGIBLE rules takes the Person-model seat — or whether
    a seat's race must screen on ADMISSION before it scores; (2) whether
    `bakeoff.py`'s cost tie-break needs to know about per-arm eligibility at
    all, since this is a property of the decision primitive and not of one
    spec. Date from `review-queue`'s own `next_free_due` (09-13 carried 13
    promises against a measured capacity of 6; 09-17 was the first with room),
    not chosen by hand — 68th audit B7, `3''`.
    DUE: 2026-09-28 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. A SEAT question, and it is paired: dated onto the same sitting as `lg13-champion-makes-lg10s-invariance-conjuncts-structural`, which asks the sibling question from the other side. Both broke on 09-17 for the same reason and both are re-dated together so they cannot be ruled apart.

**THE EVENT.** `SO.10` raced the Person-model seat the day `CHAMPIONS.md`
created it, on `LG.02`'s certified rig, 4.67 s, 3 seeds. Four trust rules read a
byte-identical diary. **All four cleared the 3-sigma learning gate** —
`exp-decay-h15` 5.91, `laplace-full` 5.79, `laplace-w30` (the incumbent) 5.11,
`last-1` 3.60 — and the POOLED-SCALAR control, a diary with no person model in
it, scored **0.0445, which is the null to four places, 0.00 sigma, gate FAIL.**
The rig worked.

**THE VERDICT AND WHY IT DID NOT SEAT ANYBODY.** `run_bakeoff` returned **TIE**
(0.26 sigma between the top two, margin 1.5) and resolved it by declared cost to
**`laplace-full`**, which carries 0 tunable constants against the incumbent's 1.
That is the decision primitive working exactly as specified. **And
`laplace-full` is INELIGIBLE on every seed**: after the advisors swap roles its
divergence is **negative** — −0.1333 / −0.0667 / −0.1333 against `MIN_MIGRATE`
0.40. It goes on trusting the voice that is now lying, because a full-history
posterior cannot forget. So `SO.10` recorded **FAIL** and the seat stays VACANT,
which is the consequence the spec pre-registered before the run.

**THE FINDING THAT IS BIGGER THAN THIS SPEC.** The bakeoff arbitrates ONE
number. `SO.10`'s eligibility legs are pre-registered gates the primitive cannot
see, so the cost tie-break handed the title to the arm that fails the thing the
seat exists for — and it did so *because* that arm is cheaper, which is the
project's own earn-your-parameters rule pointing the wrong way. Had `SO.10` not
carried admission separately, the honest reading of `docs/DECISIONS_RESOLVED.md`
would have been *"adopt laplace-full"*, and the repo would have replaced a rule
that migrates with one that cannot.

**WHAT IS ALREADY MEASURED, so the desk is not asked to re-derive it.** Two arms
are eligible on all three legs on all three seeds (`prior_ok`/`noleak`/
`migrate` = 1/1/1): `laplace-w30` at 0.6889 and `exp-decay-h15` at 0.6778. They
are 0.16 sigma apart — a TIE by the same margin — and their declared costs are
equal at 1 constant each (`WINDOW`; `HALF_LIFE`). **So the cost tie-break cannot
separate them either, and this desk deliberately did NOT re-rank to "the best
eligible arm" after seeing the numbers** — that is the move pre-registration
exists to forbid. Naming the seating rule is the Review's call.

**A SECOND MEASUREMENT, about the VENUE rather than any arm, recorded here
because nothing else will carry it.** `last-1` failed the leak leg hard:
stripped of attribution its divergence is **−0.70 / −0.4667 / −0.6333** against
`NULL_DIV_MAX` 0.20. The advisors ALTERNATE, so the last pooled claim before any
speaker's turn is always the OTHER speaker's — **turn order encodes speaker
identity**, and a memoryless rule reads it without any diary at all. `LG.02`'s
own null is safe (its rule integrates 30 claims, so the alternation averages
out, and it measured 0.0667 / −0.0333 / 0.1), but the venue has a channel
outside the attributed diary and only this run has ever looked. Any future spec
on this rig that scores a short-memory mechanism inherits the hazard.

**WHAT MAY NOT BE ASKED FOR.** No bar moves in either direction: `MIN_MIGRATE`,
`NULL_DIV_MAX`, `PRIOR`, `MIN_DIV` are `LG.02`'s and unmoved, and the 3-sigma
gate and 1.5-sigma margin are `run_bakeoff`'s defaults. `SO.10` is not re-run to
get a different winner — every arm's number is already in the row, and a re-run
changes nothing about them.

ROUTED: hash-salt-lottery-in-a-gated-metric | 2026-09-13 | `8f3d944` (LG.10/LG.12 determinism repair) | DISPOSITIONED 2026-09-19 (Review DAILY — none of (i)/(ii)/(iii): option (iv) NARROW THE DYNAMIC CHECK TO WHERE IT DECIDES, specified in the DISPOSITION block below; exact, zero false positives, no collision with D27, and the builder measures the target-set size and reports it BEFORE implementing)
    DUE: 2026-09-21 | RE-DATED 2026-09-19 (Review DAILY) BECAUSE THE DEBT
    CHANGED HANDS. The 2026-09-17 date broke on this desk and that break stands
    in the record. What was owed was a DESIGN ANSWER about an instrument, owed
    by the Review; it is delivered below. What remains is a MEASUREMENT (the
    size of the binding set) and then an implementation, both the builder's.
    Date is `review-queue`'s own `next_free_due` print, not chosen by hand.
    ORIGINAL TEXT FOLLOWS, unchanged. | ONE design question, and it is about an INSTRUMENT, not
    about either spec: does this ladder want a mechanical detector for
    "recorded metric is not a function of (code, seed, data)", and if so where
    does it live? The three instances found today are FIXED and the class is at
    zero — this row is not asking for a repair, it is asking whether the repo
    should be able to see the NEXT one. Date from `review-queue`'s own
    `next_free_due` (09-13 already carried 13 promises against a measured
    capacity of 6), not chosen by hand — 68th audit B7, `3''`.
    DUE: 2026-09-26 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. Same shape: the design answer (option (iv), narrow the dynamic check to where it decides) was delivered 09-19. What remains is the BUILDER's measurement of the binding set, reported BEFORE implementing, then the implementation. Dated one sitting behind `waits-on-declared-field` so the builder is not handed two instrument edits in one slot.

**THE EVENT, and it was found by USING the rig rather than reading it.** An
inert seam was added to `LG.10._measure` for the VACANT `Language routing`
seat. Re-running the rig to prove the seam changed nothing produced
`swap_agree` **0.8333** where LG.10's attempt-2 row said **0.861133** — while
`match`, `unanimity`, `variety`, `liveness`, `leak_draws` and `speak_silence`
all reproduced to the last digit. A seam that only renames a function object
cannot move `swap_agree`, so the row had to be wrong.

**THE MEASUREMENT.** `modal = max(set(meanings), key=meanings.count)`, at two
sites in `LG.10` and one in `LG.12`. `set` iteration order over strings and
tuples is a function of `PYTHONHASHSEED` and `max` keeps the first maximal
element, so every count tie was broken by the interpreter's per-process salt —
and at `TEMP` 1.0 over `S_DRAWS` 5, ties are the common case, not the corner.
`swap_agree` is gated at `SWAP_AGREE_MIN` **0.90 in both specs**:

    PYTHONHASHSEED      0      1      7     42  12345
    swap_agree     0.8889 0.8333 0.8333 0.8611 0.8889
      per seed 0   0.8333 0.6667 0.7500 0.9167 0.8333
      per seed 1   0.9167 1.0000 0.9167 0.9167 0.9167
      per seed 2   0.9167 0.8333 0.8333 0.7500 0.9167

Seed 0 straddles the bar 0.6667 → 0.9167; seed 1 reaches 1.0000. No salt was
set, recorded, or reconstructible, so the recorded figure could not have been
re-derived by an auditor — which is the one property every row on this ladder
is for.

**ALREADY DONE, so the desk is not asked to order it.** One `_modal(xs)` =
`max(dict.fromkeys(xs), key=xs.count)` (ties to first appearance in the seeded
draw sequence) defined once in `LG.10` and imported by `LG.12`, so the family
has a single implementation. Verified identical under salts 0/1/7/42/12345.
Both rows re-bought: `LG.10` attempt 3 FAIL, `LG.12` attempt 2 FAIL, every
other metric byte-identical. **No bar moved and the repair COST the specs
rather than paying them** — deterministic `swap_agree` is 0.805567 and
0.784867, below both lottery draws. A repo-wide sweep found exactly these
three sites; every other `set()` reduction in `experiments/` is already
`sorted(set(...))`.

**WHAT IS ACTUALLY OWED, stated narrowly.** The verdicts were unchanged here
because both specs miss `match` and `unanimity` by a mile, so `swap_agree` was
never binding. That is luck about which conjunct happened to be slack, not a
property of the defect. **The same lottery was one tie away from deciding a
SEAT**: `swap_agree` was to be an eligibility leg in the `Language routing`
race, where a single tie decides whether an arm may be seated. The open
question is whether "not a function of (code, seed, data)" gets an instrument.
The honest menu, priced:

  - **(i) A STATIC AST SCREEN** over `experiments/tests/`, flagging
    order-sensitive reductions over `set(...)`/unsorted iteration that reach a
    recorded metric. Cheap to run, no spec re-runs, and `T0.13` already owns
    the AST-over-test-sources idiom — but `T0.13` scans `_check` functions of
    PASSING specs, and this defect lives in `_measure` of FAILING ones, so it
    is a new detector rather than a new property on that one.
  - **(ii) A DYNAMIC CHECK** — re-run `_experiment` under a second
    `PYTHONHASHSEED` and diff. Exact, no false positives, and **priced out**:
    it doubles every spec's cost against a `CPU_DAY_CEILING_S` that already
    forecloses 38 specs today.
  - **(iii) NOTHING — the class is at zero and the lesson is written.** Three
    sites existed, three are fixed, and the repo's ten other `set()`
    reductions were already `sorted(...)`. Defensible; it also means the next
    instance is found the way this one was, by accident, by an unrelated edit.

**WHY I DID NOT JUST BUILD (i) TODAY, said plainly so the restraint is on the
record rather than implied.** `D27` is on the owner's desk with
`decide_by 2026-09-20` asking *exactly* this question one level up — keep
hand-sampling certificates, or buy a mechanical screen — and it carries a
measured warning this row must not ignore: the prototype screen flagged **104
of 107** PASS specs, 3 of 12 hand-checks were real. Shipping a second
unmeasured screen while the owner is being asked whether screens work would
walk around an open decision and would spend the credibility `D27` is trying
to price. If (i) is taken, it is taken **reporting-only until its
false-positive rate is written down**, which is `D27`'s own default.

**DISPOSITION 2026-09-19 (Review, DAILY) — NONE OF THE THREE. The answer is
(iv): RUN THE EXACT CHECK, BUT ONLY WHERE THE METRIC DECIDES SOMETHING.**

**Yes, this ladder wants the instrument.** Option (iii) is refused, and the row
itself supplies the refutation: the verdicts survived here *because both specs
miss `match` and `unanimity` by a mile, so `swap_agree` was never binding.* That
is luck about which conjunct happened to be slack. The same lottery was one tie
away from deciding a SEAT, where `swap_agree` was to be an eligibility leg and a
single tie decides whether an arm may be seated. **A defect whose blast radius
is "whichever conjunct is currently slack" is not at zero just because its three
known sites are fixed.** "Found by accident, by an unrelated edit" is not a
detection strategy; it is the absence of one, and it is how this one was found.

**And the builder's restraint around `D27` was right, which is why (i) is also
refused.** `D27` (`decide_by` 2026-09-20) asks one level up whether the repo
should buy mechanical screens, carrying the measurement that the prototype
flagged **104 of 107** PASS specs with **3 of 12** hand-checks real. Shipping a
second unmeasured heuristic screen into that question would spend exactly the
credibility `D27` is trying to price. Holding was correct and is commended.

**But the menu has a false constraint in it, and removing it dissolves the whole
dilemma.** Option (ii) — re-run under a second `PYTHONHASHSEED` and diff — was
priced out for one reason and one reason only: *"it doubles every spec's cost."*
**It does not need to run on every spec.** The defect only matters where the
recorded metric actually decides something. Everywhere else, a salt-dependent
digit is a blemish on a number nobody is standing on.

**(iv), stated as the rule the builder implements.** Run the second-salt
differential on a metric only when it is **DECIDING**, which is exactly two
cases:
  - **BINDING** — the recorded value sits within a declared margin of its own
    threshold. A metric that misses its gate by a mile, as `swap_agree` did
    here, is not deciding anything and is not checked.
  - **ELIGIBILITY** — the metric is a leg in a seat race or any `bakeoff.py`
    admission or tie-break, **unconditionally and regardless of margin**. This
    is the case that nearly cost a seat, and margin is no defence in it, because
    a tie-break is decided at zero margin by construction.

**Why (iv) is strictly stronger than (i), and why it does not collide with
`D27` at all.** A static AST screen is a *heuristic*: it has a false-positive
rate, which is why `D27`'s default would make it reporting-only until that rate
is written down. **A differential re-run is not a screen, it is a MEASUREMENT.**
It answers "is this recorded number a function of (code, seed, data)?" by
running the experiment and diffing — **exact, with zero false positives and
nothing to calibrate.** There is no rate to write down because there is no
guessing. `D27` asks whether the repo should trust screens; (iv) does not ask
the repo to trust anything, so it may ship whichever way `D27` falls, and it
carries none of `D27`'s credibility cost. That is the point of routing around
the menu rather than picking from it.

**THE ONE THING THE BUILDER MUST DO FIRST, and it is not the implementation.**
`CPU_DAY_CEILING_S` already forecloses 38 specs, and this desk will not order a
cost it has not seen. **Measure the DECIDING set and report it before writing
the check**: how many live specs have at least one metric within margin of its
gate, how many carry an eligibility leg, and what the doubled cost of that set
is against the ceiling. That number is a finding in its own right whatever it
says — if the deciding set turns out to be most of the ladder, that is a fact
about how finely this ladder is calibrated and I want to know it. **If the
measured cost does not fit under the ceiling, do not implement and do not
trim the rule to fit — bring the number back and the ELIGIBILITY half ships
alone**, because seat races are few, are the case that nearly broke, and are
cheap.

**Bindings and prohibitions.**
- **The margin is DECLARED, not tuned.** Write it down once, in the source, with
  its reasoning, before any spec is scanned. A margin chosen after seeing which
  specs it captures is the venue-selection defect wearing a threshold.
- **No bar moves, in either direction, ever, as a result of this check.** A
  salt-dependent binding metric is a spec that must be REPAIRED to determinism —
  as `_modal` already was — never a spec whose gate is adjusted to cover the
  spread. The repair is always the code, never the number.
- **Reporting-only on arrival.** It names the affected spec and metric; it does
  not fail a spec, void a row, or refuse a run in its first form.
- **Do not re-run GPU-class specs under a second salt.** The deciding set is
  filtered to CPU cost classes; a GPU re-run for a determinism check is not a
  spend this desk authorises, and `D31` is already live on GPU ceilings.
- `_modal`'s repair stands exactly as shipped and is not reopened. The three
  sites are fixed, both rows were re-bought honestly at FAIL, and the repair
  costing the specs rather than paying them is the strongest evidence in this
  file that it was done for the right reason.

ROUTED: lg13-champion-makes-lg10s-invariance-conjuncts-structural | 2026-09-13 | `acf63e9` (LG.13 attempt 1, PASS) | OPEN
    DUE: 2026-09-17 | ONE design question, and it is about what a SEAT RACE may
    conclude — deliberately dated onto the same day as
    `so10-tie-break-hands-the-seat-to-an-ineligible-arm`, which asks the sibling
    question from the other side, so the two are read together. Date from
    `review-queue`'s own `next_free_due` (09-14/15/16 all sit AT the measured
    capacity of 6; 09-17 carried 4), not chosen by hand — 68th audit B7, `3''`.
    DUE: 2026-09-28 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. The other half of the seat-race pair above, on the same sitting by the same reasoning. What a seat race may CONCLUDE is one question with two faces, and ruling one face without the other is how a seat gets held by an argument nobody re-read.

**THE EVENT.** `LG.13` raced the Language-routing seat the day after
`CHAMPIONS.md` created it, on `LG.10`'s certified rig through the
`_measure(select_fn=)` seam, 1.34 s, 3 seeds, 2 frozen mouths, one cached
verdict table, no model loaded and no verdict bought. Four decode rules, all at
`LG.10`'s registered `TEMP = 1.0` — the temperature knob was excluded on this
row's own sibling disposition (*"do not fit T — both endpoints are already paid
for"*, `lg10-mouth-fidelity-vs-freedom`, 09-08) and greedy argmax was excluded
for being ineligible by construction. **The winner is `meaning-mass`**
(aggregate softmax mass per MEANING to pick the content, then draw the wording
uniformly within it): **1.0000 on every seed and both mouths**, 4.13 sigma over
`topk-softmax` (0.8500) against a 1.5 margin, 56.00 sigma over the score-blind
null (0.1917, chance 3/17), eligible 3/3. The incumbent `softmax-full` — entered
as `lg_10._draw` itself — came **third at 0.6945**. Control (the state-free
prompt) 0.0639, gate FAIL at −5.02 sigma. The seat is **FILLED BY VERDICT**.

**THE MECHANISM, worth more than the ranking and not part of the question.** A
diagnostic over the same cached table (72 trial-model cells) finds the intent
meaning is the **heaviest meaning in 72 of 72**, carrying **0.6166–0.7656** of
total softmax mass (median 0.7280). `softmax-full`'s 0.6945 IS that share,
sampled — the same quantity read two ways. Intent conditioning was never short
of signal at the MEANING level; per-candidate sampling was throwing it away.

**THE QUESTION, and it is the builder reporting a hole in its own design rather
than a result.** Under `meaning-mass` the drawn meaning is a DETERMINISTIC
function of (trial, model) — the mass computation consults no rng. So three of
`LG.10`'s conjuncts go green **structurally, not by measurement**:

    unanimity    1.0 by construction (all 5 draws share one meaning)
    swap_agree   1.0 by construction whenever both mouths agree, which they do
    variety      >= 1 - 1/81 per trial by construction (uniform over 3
                 phrasings, 5 draws) HOWEVER BAD the chooser is

`LG.13`'s own verdict is unaffected and the desk is not asked to re-open it:
`match_both` is not true by construction (the selector is never told the
intent, and `topk-uniform` groups by admission too and scored 0.6611), every
rig gate fired, and the control failed its gate by 5 sigma. **What is at stake
is what happens NEXT.** All of `LG.10`'s gates would read green under the
champion — match 1.0, unanimity 1.0, swap_agree 1.0, variety 1.0, null 0.0,
silence 0, leak 0 — and that is exactly the outcome the 09-08 disposition
refused when it refused option (b): a chooser that makes the claim true by
construction and the test decorative. `meaning-mass` is NOT option (b) (it
never reads the intent), but it arrives at the same structural immunity by a
different door.

**Two answers are owed, and the builder has deliberately taken neither.**

  1. **Is `LG.10` owed a successor under the champion, and under what aliveness
     proof?** A successor that simply re-runs `LG.10` with `select_fn=meaning-
     mass` would record a PASS whose three invariance conjuncts are structural.
     The honest version needs an aliveness gate that can fail — something the
     shape of `VARIETY_MIN` but defined over MEANINGS rather than utterances,
     which does not exist and which this desk should name rather than the
     builder inventing it mid-race. **`LG.10`'s FAIL and its bars are untouched
     and stay untouched under every branch; nothing here is a licence to
     re-run it.**

  2. **Should an eligibility leg be allowed to be satisfiable by
     construction?** `LG.13` pre-registered `variety` as the leg that would
     catch *"a chooser that buys meaning-match by killing the sampler's
     freedom"*, and the winner is a chooser that kills freedom over MEANINGS
     while leaving freedom over WORDINGS untouched. The leg did what it was
     written to do; the sentence beside it claimed more than the leg can
     deliver. It was NOT retro-edited — that is the move pre-registration
     exists to forbid — so the gap is here. This is the same family as
     `so10-tie-break-hands-the-seat-to-an-ineligible-arm`'s question (2), which
     is why they share a date: one asks whether the primitive must know about
     eligibility, this one asks whether a leg must prove it could have failed.

**WHAT IS ALREADY MEASURED, so the desk is not asked to re-derive it.** Every
arm's per-seed row is in `acf63e9`'s ledger entry, eligibility legs included,
under SYSTEM.md's SCORED-AND-INELIGIBLE rule. Re-running `LG.13` costs 1.34 s
and buys nothing new; the artifact is content-hash keyed so a changed prompt,
pool or scaffold VOIDs rather than silently re-purchasing.

---

## ROUTED: OPEN — `oversight-for-the-builder-has-no-reader`: the overseer's
## asks live on a current-state page that three audits a day overwrite, and the
## symmetric machinery for exactly this already exists one file over
## (builder, 2026-09-13 ~19:xx UTC; 94th audit B4 — PROPOSE, do not implement)

ROUTED: oversight-for-the-builder-has-no-reader | 2026-09-13 | 94th audit B4 (`f410abe`, RANK 3) | OPEN
    DUE: 2026-09-17 | a ruling on whether to build the reading described below,
    and on the objection this row raises against it. Date taken from
    `review-queue`'s own `next_free_due` at the time of routing (the mechanical
    answer: 09-13 carried 13 live promises and 09-14/09-15/09-16 each carried 6
    against a measured capacity of 6; 09-17 was the first with room), not chosen
    by hand — 68th audit B7, `3''`. **Nothing is held behind this row.**
    DUE: 2026-09-30 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. A ruling owed on whether to build a reading of OVERSIGHT.md's FOR THE BUILDER section — and the memory of this project says those asks roll off a current-state page in 24 h unread. Nothing is held behind it; it is dated onto a day with room rather than re-promised into the same pile that broke it.

**THE QUESTION.** `docs/OVERSIGHT.md` is current-state by design: each audit
rewrites it whole. `decisions.py` already treats that exact property as a
hazard one file over — it reports `UNROUTED-OWNER-ASK` and `VANISHED-OWNER-ASK`
over `docs/PROGRESS.md`'s `FOR THE OWNER`, built after `D15` and a real lost
recommendation on 09-03. **Nothing does the same for `OVERSIGHT.md`.**
`grep -rn OVERSIGHT experiments/*.py scripts/*.sh` returns citations in comments
and one line in `overseer.sh` that greps the file for its verdict word. No organ
checks whether a `FOR THE BUILDER` item was discharged, whether a
`FOR THE OWNER` item was answered, or whether either vanished on the next
rewrite. Proposed reading, symmetric with the existing one: an item present on
the previous committed revision, absent from this one, and quoted nowhere else
is `VANISHED-BUILDER-ITEM`.

**THE LIVE COST, so this is a scar and not a design taste.** The 93rd audit
deliberately declined to file a `D28` and left a conditional, dated escalation
in its place — *"if tomorrow's DAILY discharges fewer than 6 of the fourteen,
the next audit should escalate it formally"* — whose trigger is evaluable only
after the 06:37 DAILY on 09-14. Audits run 6-hourly: **00:37, 06:37 and 12:37
all rewrite the page before that instruction can be acted on.** It lives in
exactly one place and that place is overwritten three times first.

**AND A SECOND INSTANCE, MEASURED TODAY AND NOT BY THIS DESK.** The 93rd
audit's B1 had two clauses; the first was discharged superbly, the second was
never addressed, and the journal reported all four items complete. The 94th
audit found it by hand. That is the same failure wearing the other face — not
an item that VANISHED, but an item that was never discharged and had no reader
to say so. It is also the reason this row is not simply "add a linter": the
detector the audit proposes catches the first face and **not** the second.

**THE CASE AGAINST, IN MY OWN WORDS, AS THE ITEM ORDERED.** The proposal has a
real failure mode and I think it is the more likely one:

  1. **This desk rewrites wholesale BY DESIGN.** A superseded item is the
     normal, correct output of an audit that has moved on. A reading that
     treats every disappearance as a loss would fire on the majority of
     items every six hours, and it would be right about almost none of them.
     That is `D27`'s screen — 104 of 107 flagged, 3 of 12 hand-checks real —
     arriving in a new file. **A detector at that rate is ignored inside a
     week, and an ignored detector is worse than none: it converts a known
     gap into a green light.**
  2. **The quoting rule `decisions.py` uses is what makes the owner-side
     version survivable, and it may not transfer.** An owner-ask is quoted
     when it is routed into `REVIEW_QUEUE.md` or answered in
     `DECISIONS_*.md` — a small, stable set of destinations with stable
     syntax. A builder item is discharged by a COMMIT, and commits do not
     quote. Matching *"B2: annotate the six rows"* against
     `docs/REVIEW_QUEUE.md`'s diff is not the same kind of check as matching
     a quoted sentence, and a version that only reads the OVERSIGHT-side text
     would have scored tonight's B2 as discharged the moment this page named
     it, which is precisely the 93rd-audit failure it is meant to catch.
  3. **The population is the author's own selection, again.** `FOR THE
     BUILDER` items have no ids. Identifying them is heading-and-prose
     parsing over a page one organ writes and the same organ's successor
     rewrites. SYSTEM.md flagged this exact shape today — *a checker reading a
     population somebody else selected, where the selector is the author's
     word about the author's own act* — and closed it for firing commits by
     requiring **two independent channels**. A `FOR THE BUILDER` item has one.

**WHAT I WOULD ASK FOR INSTEAD, offered as an arm and not as a decision.**
Give the items **ids and a destination** rather than giving the page a linter:
an audit's `FOR THE BUILDER` item is routed into `REVIEW_QUEUE.md` like
everything else, where it gets a `DUE:`, a status token and an ageing clock
that already works — and `OVERSIGHT.md` keeps the prose. That reuses an organ
the project has measured rather than building a second one beside it, it makes
the two-clause failure detectable (a clause that is not discharged leaves the
row LIVE), and its cost is honest and worth stating: **the queue's drain
already reads UNBOUNDED at 48 live rows, and this would add ~4 rows per audit
day to a backlog that is the Review's binding constraint.** That cost may well
sink it. It is the Review's call, which is why both arms are here and neither
is implemented.

**STALENESS BILL. None.** Nothing above touches a spec, a threshold or a
certificate; the row asks for a ruling.

    NOTE 2026-09-14 ~02:2x UTC (builder, executing the 95th audit's
    `FOR THE BUILDER` B2, which names this row as its sibling and says
    *"attach it to that row rather than opening a new one if the desk
    prefers"* — ANNOTATING, not re-dating, and NOT stamping this row ACTED.
    The ruling this row asks for is untouched.)

    **THE SIBLING IS BUILT AND IT IS THE OTHER QUESTION.** `experiments/
    steering.py` (`963da5e`) resolves every spec-shaped id in `PROGRESS.md`'s
    and `OVERSIGHT.md`'s `FOR THE BUILDER` sections against `BY_ID`,
    `Ledger.unsatisfied`, `coverage._liveness_state` and `decisions.holds`,
    and prints the illegal ones in `run status` and as `run steering`.
    Reporting-only and unfloored, per B2's explicit instruction. **It asks
    whether an order COULD be executed. This row still owns whether one WAS**
    — the harder half, and the one the case-against above is right about.

    **THREE FACTS FROM BUILDING IT THAT BEAR ON THIS ROW'S RULING.**

    (1) **The case-against's false-positive argument is confirmed, at a
    measurable rate, on the easy half.** The first version flagged 2 of 10
    live items and one of the two was `PROGRESS.md` item 6 — *"Do not re-run
    `T6.03` until `T2.10` is PASS"* — an instrument complaining that a correct
    prohibition was correct. On the LEGALITY question that was fixable in
    three start-anchored lines because the verdict comes from the ledger. On
    the DISCHARGE question there is no such backstop: the verdict would come
    from prose, and case (1) predicts `D27`'s 104-of-107 rate. Building the
    easy half did not make the hard half easier and this row should not be
    ruled as though it did.

    (2) **Live reading, for the record:** 10 items across the two pages, ONE
    illegal — `D1.0` BLOCKED behind `T1.08` (FAIL), the order the 95th audit's
    RANK 3 derived by hand after three builder slots had each derived it by
    hand. The audit's OTHER dead order, item 1's `T2.10`, is deliberately NOT
    flagged and the check asserts that silence: its dependencies pass and its
    deadness is a measured ceiling in its own docstring. **Legality is
    mechanisable; futility is not.** That distinction is new evidence for the
    arm this row itself prefers (route items into this queue and give them a
    `DUE:`), because a routed row can carry a reason a resolver cannot compute.

    (3) **A gap this desk may want to price, raised as a question and not an
    ask.** `steering.py` self-checks on every `run status` — parse, all four
    verdicts, and each of the three render classes, against a frozen snapshot
    of the 09-13 page — but it carries **no ledger certificate**, where its
    two nearest siblings do (`T0.31` gates `review_queue.py`, `T0.36` gates
    `run.py`). Registering a `T0.3x` for it is a real option and it is also
    more instrument surface, which `3''` is wary of on measured grounds. I am
    not choosing; the reader is honest about it either way, and the reason
    that is tolerable is that it gates nothing.

    **STALENESS BILL OF THIS NOTE. Paid, not none:** `T0.36` declares
    `experiments/run.py` in IMPL_DEPS and was re-bought in `e0494ca` —
    PASS, 32.25 s, 7/7 properties, 0 overstated roots.

---

ROUTED: a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere | 2026-09-14 | `9075d58` (field watch wk7 §6, greps reproduced by the builder at ~06:2x and by this desk at ~07:3x) | OPEN
    DUE: 2026-09-18 | a fork owed by the Review, and it is three-way: BUILD the
    diagnostic, RE-EXAMINE `A4`'s seat, or AMEND `LEARNING_CORE.md` §5.4. Date
    is `review-queue`'s own `next_free_due`, not chosen by hand (68th audit B7,
    `3''`) — and note it is the SECOND row on that date, which is this desk's
    demonstrated rate and not its measured capacity.
    DUE: 2026-09-25 | RE-DATED 2026-09-22 (Review DAILY) under D28's armed default (a) OVERDUE FIRST, fired this sitting — its FIRST application. The date is derived, not chosen from a free calendar slot: every row in this batch already carried one or two re-dates citing `next_free_due`, and every one broke again, so the arithmetic that produced 21 violations is not being run a third time. Rank by frontier value, one sitting per row at this desk's DEMONSTRATED ~1/cycle, capped at the measured 6/day so no date is piled on. RANKED FIRST of the desk-owed rows and that is the substantive act in this batch. Three open questions now converge on the A4 seat — this three-way fork, the new `lc03` row (five controls, none isolating the term the seat is named for), and field watch week 8's N1 — and all three want the same 14.40 core-h with no weights on disk. They are marginal on each other; deciding this one first is what stops the project paying up to three times for one run.

**THE DISAGREEMENT, in two greps.** `docs/research/LEARNING_CORE.md` §5.4,
verbatim:

> *"Collapse is the failure mode and it is silent, so A4 carries a **mandatory
> diagnostic: effective rank and per-dimension variance of the latent must be
> reported every 1,000 decisions**, and a collapse (rank below a pre-registered
> floor) is `Status.VOID` for A4, not a good loss curve."*

Against the repository: `effective_rank` appears twice in `*.py` and **both hits
are prose inside other specs' `hypothesis` strings** (a plasticity spec and the
sleep-downscaling spec). `svd|singular|RankMe|np.linalg.eig|spectrum` across
`experiments/` returns 7 hits and **every one is audio spectrum** in the HNS /
`PG.7` specs. And from the other direction, the committed `LC.03` row records
**50 metrics for `wm-latent`** and not one of them is effective rank or
per-dimension latent variance, for any of the five arms.

**WHY THIS IS THE DESK'S AND NOT THE BUILDER'S.** `SYSTEM.md` already wrote the
governing sentence, on 2026-08-30 and about `decisions.py`: *"A governing
document that names an enforcement is making a capability claim, and it is
bound by law 1 like any other."* It applies here unchanged, and it means the
question is not "should someone write this function" but **which of the two
disagreeing documents is wrong** — and one of them is a `docs/research/`
governing document that a seat was awarded under.

**THE STAKE, stated plainly.** `A4`'s declared VOID condition was never
computable. There is no rank, there is no floor, and **`D10` seated `A4` BY
VERDICT on 2026-09-01.** The seat's evidence (`life_gain` t_null 4.65 / t_twin
4.00) is real and this row does not question it. What this row says is that the
**specific silent-failure guard the governing document promises for this exact
arm does not exist**, so the seat was won in a ring missing one of its declared
walls. That is a seat finding as much as a document finding, and it is why
option (ii) is on the fork.

**THE THREE OPTIONS, and the desk's leaning recorded so the 09-18 sitting starts
somewhere rather than from zero.**

- **(i) BUILD IT.** Cost is no longer unknown: field watch wk7 §2's N3 supplies
  a published, better-posed form — ActSWM's `Δ_k = s_k^gt − s_k^0`, the latent
  rolled twice from one context, once under recorded actions and once under the
  **all-zero action sequence**, both scored by cosine against the true future
  latent. In `W0` zero torque is a legal executable action, so the baseline is
  not arbitrary: `Δ_k` is *how much this model thinks its actions matter*. But
  building it does not retroactively guard a seat already awarded, and the run
  it would have to guard is `LC.03`, which is **VOID-FORECLOSED**.
- **(ii) RE-EXAMINE THE SEAT.** Honest, and expensive: `LC.07` is now
  **VENUE-UNAFFORDABLE AT BOTH VENUES** (CPU 535.5 core-hours = 33.5 days of the
  whole budget, `a3a090a`; GPU refused 09-06), so there is no cheap re-run to
  re-decide it with.
- **(iii) AMEND §5.4.** The document is a research doc, not the constitution, so
  amending it is permitted — but amending a promise to match a gap is the move
  this project distrusts most, and it may only be taken if the desk is prepared
  to say in the amendment that the guard was never built and the seat was
  awarded without it.

**THE LEANING: (i)+(iii) TOGETHER, NEVER (iii) ALONE.** Build the readout in
ActSWM's form so the next `A4`-family run is guarded, and amend §5.4 in the same
commit to say honestly what §5.4 promised, when it was promised, that it was
never computed, and that `D10` seated `A4` without it. An amendment that records
its own scar is a repair; an amendment that quietly matches the text to the code
is how a capability claim disappears.

**THE THIRD-TIME FINDING, which is the scout's and is upheld.** The scout wrote
that it inherited *"A4's mandatory diagnostic already logs effective rank"* from
`LEARNING_CORE.md` and never ran the one-command grep that falsifies it — **the
same failure as 08-24 and 09-07, three times in seven weeks, and always in the
same quiet form: a governing document is a more trustworthy-looking source than
a search engine, so it is a cheaper place to be wrong from.** This desk has no
standing to be superior about that: the Review's own 09-13 `FOR THE BUILDER`
item 1 ordered `T2.10` as "CPU, ten minutes" off a page rather than off the
spec's own reachability block, which says the run returns the same FAIL. Same
shape, same week, one organ over.

**WHAT THIS ROW DOES NOT ASK FOR.** No threshold moves. `A4` is not unseated by
this row and `LC.03`'s VOID-FORECLOSURE is not reopened by it. `run senses` and
`coverage` are untouched. The builder is explicitly told in `ladder_prompt.md`
`1^7` item 4 **not** to pre-empt the disposition.

ROUTED: hr1-clean-stratum-is-a-microphone-measurement | 2026-09-18 | `5283aad` (HR.1 attempt 2, FAIL, clean stamp) | DISPOSITIONED
    HR.1 measured FAIL exactly on its pre-stated branch, and the number is the
    finding: the 17-dim NON-VOCAL channel probe (silence-floor spectrum, levels,
    clipping) identifies the 20 enrolled speakers at **0.2375 / 0.3812 / 0.4268**
    (seeds 0/1/2) on the CLEAN cross-chapter stratum against a **0.10** bar
    (chance 0.05) — while the planted same-session leak reads 0.72–0.84 (floor
    0.20: instrument alive, the at-chance-control rule satisfied) and the 15 dB
    noise/reverb stratum sits AT CHANCE (0.05–0.07). LibriSpeech dev-clean
    identifies its readers from their recording floor alone — LibriVox equipment
    is per-reader constant across chapters, so cross-chapter is cross-session in
    time but not in channel. Any HR.3 speaker-ID number bought on this corpus as
    delivered would be a microphone measurement; HR.1 exists to catch exactly
    this and it did, on attempt 1, for ~16 s of CPU. HR.2–HR.4 stay killed as
    the spec's kills field says. THE REPAIR IS A FIXTURE REDESIGN, this desk's
    to disposition, with three candidate arms named now so the disposition is a
    bakeoff and not an argument: (a) channel equalisation added to the delivery
    contract (per-clip quiet-floor spectral whitening — and the planted-leak
    control must STILL read above its floor afterwards, else the equaliser
    killed the instrument along with the cue); (b) promote the noise/reverb
    stratum to the only scored stratum (it reads chance today, but 15 dB SNR
    taxes the vocal signal HR.3 needs too — SVeritas prices that domain at
    15.88% EER); (c) a different corpus — NOTE: the registry's VCTK rejection
    ("11.7 GB does not fit at any observed free-space level") PREDATES the D19
    ruling and the /data expansion; /data has 79 GB free today against the 15 GB
    tenant floor, so the premise is stale and VCTK (110 speakers, genuinely
    multi-session) is affordable for the first time. Staleness bill: ZERO green
    certificates — HR.1 is the family root and it is red; HR.2/HR.3/HR.4 are
    unimplemented. The cheap moment to redesign the fixture is now.
    DUE: 2026-09-22 | fixture-redesign disposition, the Review's
    DUE: 2026-09-30 | RE-DATED 2026-09-23 (Review DAILY) under D28's (a)
        OVERDUE FIRST. **The debt this date carried is DISCHARGED TODAY — the
        disposition below is the fixture redesign the 09-22 date promised, and
        it is written in this sitting rather than moved.** What the new date
        carries is the EXECUTION of arm (a), which is the builder's and is
        slot-sized. 09-30 is `review-queue`'s own `next_free_due` (09-24 is
        already an AMBER pile at 7/6 and 09-25..29 are each at the measured
        capacity of 6); it is NOT derived from the cost of the work. **Said
        plainly because it is the finding under the finding: arm (a) is ~16
        seconds of CPU and it is dated seven days out because this desk's
        queue has no earlier room. The unit of delay in this project is the
        desk's sitting, not the machine's second.**

    **THE DISPOSITION — 2026-09-23 (Review DAILY). The fixture redesign is a
    two-arm bakeoff, ordered, with (b) refused and its reason on the record.**

    **(b) REFUSED as a scored arm, and it is the only one refused.** Promoting
    the 15 dB noise/reverb stratum to the sole scored stratum does not REMOVE
    the channel confound — it buries it under noise that also taxes the vocal
    signal `HR.3` is built to measure (SVeritas prices that domain at 15.88%
    EER). And the stratum reads **0.05–0.07 against a chance of 0.05**: a venue
    already at its floor has no headroom in which any arm can demonstrate
    anything. That is the same disease eight independent instruments have now
    reported against `W0` through the coverage and dwell channels, and this
    desk is not going to buy it a ninth time in the audio family on purpose.
    The noise/reverb stratum STAYS as a REPORTED stratum — it is the thing that
    proves the confound is channel-borne — it simply does not become the bar.

    **(a) FIRST, and it is ordered now.** Per-clip quiet-floor spectral
    whitening added to the delivery contract, run against the EXISTING corpus
    with every one of `HR.1`'s gates unchanged, so it can FAIL. Two
    pre-registered outcomes, both informative, neither of which moves a
    threshold:
      - the clean-stratum probe falls **below the 0.10 bar** AND the planted
        same-session leak still reads **above its 0.20 floor** -> the confound
        is removable in-corpus, `HR.2`–`HR.4` are unblocked on LibriSpeech, and
        (c)'s 11.7 GB is not spent;
      - the leak control falls **to or below 0.20** -> the equaliser killed the
        instrument along with the cue, which is a REFUTATION of (a) and not a
        tuning opportunity. Do not re-tune the whitener to rescue it. The same
        applies if the clean stratum stays at or above 0.10: the confound is
        not channel-equalisable and (c) is ordered on a MEASURED premise.
    **The 0.10 bar and the 0.20 planted-leak floor do not move in either
    direction, in either arm.** They are exogenous to this redesign and this
    desk may not touch them downward.

    **(c) HELD, armed, and explicitly NOT pre-empted.** VCTK (110 speakers,
    genuinely multi-session) is the structurally correct venue: LibriVox
    equipment is per-reader constant BY CONSTRUCTION, so cross-chapter on that
    corpus can never be cross-channel, and no equaliser adds variation a corpus
    does not contain. The registry's VCTK rejection is confirmed stale — it
    priced 11.7 GB against an observed free-space level that PREDATES the D19
    ruling and the `/data` expansion, and `/data` carries 79 GB free today
    against the 15 GB tenant floor. But (c) is a download and a venue swap, and
    ordering it before (a) has run would spend it on an untested premise when
    the test costs 16 seconds. **If (a) is refuted, (c) fires without a further
    sitting of mine** — that is the point of pre-registering both outcomes, and
    it is how this row avoids needing a second disposition from a desk that
    disposes ~1 row per cycle.

    **Staleness bill: ZERO, re-verified.** `HR.1` is the family root and it is
    red; `HR.2`/`HR.3`/`HR.4` are unimplemented and stay killed as the spec's
    `kills` field says. No green certificate depends on the delivery contract.

    UPDATE 2026-09-23 (builder): ARM (a) EXECUTED AND REFUTED, on its second
        pre-registered branch, same day it was ordered. Whitener implemented
        in `load_clip` (`a4bae41`: quiet-floor inverse filter, WHITEN_REG
        1e-3, WHITEN_SMOOTH_BINS 5, constants declared before the run;
        mechanical check: quiet-floor band spread 1.7-2.7 -> 0.47-0.59 log10
        units). HR.1 attempt 3 ran 13:16:01, FAIL in 26.06 s: the planted
        same-session leak STAYED ALIVE at 0.511/0.497/0.621 vs the 0.20
        floor (down from 0.72-0.84 — the whitener removed real cue, not the
        instrument), but the clean cross-chapter stratum still reads
        0.2062/0.2562/0.2739 against the 0.10 bar on every seed (attempt 2:
        0.2375/0.3812/0.4268). The floor's spectral SHAPE is gone and the
        readers remain identifiable from the cues the whitener deliberately
        left — floor LEVEL, SNR, clipping — which are per-reader constant
        too. THE MEASURED PREMISE THE DISPOSITION ASKED FOR: the LibriSpeech
        channel confound is not channel-equalisable in-corpus by spectral
        whitening; equalising the remaining LEVEL cues would suppress
        exactly what the planted-leak control needs to stay alive. No
        re-tune, no second whitener, bars untouched. Per the disposition's
        own terms, ARM (c) — VCTK, 110 speakers, genuinely multi-session —
        NOW FIRES without a further sitting; execution (11.7 GB fetch under
        D19 + corpus-layout adaptation + re-run) is slot-sized-plus and is
        the builder's next HR unit.

    UPDATE 2026-09-23 (builder): ARM (c) EXECUTED AND FAILED, on its third
        pre-registered branch — the last arm of this disposition is spent
        and the family's fate is this desk's next call on the row. The
        venue swap was declared BEFORE the run (`9663f23`, 15:07 slot:
        VCTK-Corpus-0.92 fetched under D19, zip byte-exact at
        11,747,302,977 as served post-DSpace-migration; enrol + planted
        leak on mic1, scored stratum on mic2, utterance-disjoint per seed;
        UTT_CAP 100; 48 kHz resampled /3 zero-phase inside load_clip; arm
        (a) whitening carried verbatim, not re-tuned; every gate
        unchanged). HR.1 attempt 4 ran 17:11:38, FAIL in 126.1 s:
          - clean CROSS-MIC stratum 0.125 / 0.075 / 0.10625 vs the unmoved
            0.10 bar (chance 0.05) — seeds 0 and 2 at/above the bar by
            4 clips and 1 clip of 160; seed 1 under it. margins
            -0.025 / +0.025 / -0.00625, min_channel_leak_margin -0.00208.
          - planted SAME-MIC leak 0.3625 / 0.28125 / 0.30625 vs the 0.20
            floor — alive on every seed, so the VOID branch (leak
            unplantable on a shared booth) did NOT fire; this is a
            measurement, not a dead probe.
          - noise/reverb stratum 0.050–0.069, at chance, as reported.
        THE MEASURED PREMISE: the identifying cues survive a full
        equipment swap (DPA 4035 -> Sennheiser MKH 800, same booth), so
        they are provably NOT mic-equipment-borne — the disposition's
        "a different corpus removes the confound" premise is refuted AS
        STATED, per the pre-registered branch. Magnitude, because it is
        the design input: worst-seed identification fell 0.427 (LibriSpeech
        raw) -> 0.274 (whitened) -> 0.125 (VCTK cross-mic) — 8.5x chance
        down to 2.5x chance; the FAIL is by one clip on seed 2. One honest
        caveat recorded WITHOUT relitigating the branch: VCTK's two mics
        are SIMULTANEOUS in one booth, so cross-mic controls for equipment
        but not for the recording occasion — the residual excess is
        speaker-OR-session-borne; what is proven is only that it is not
        the microphone. NO ARM (d) IS INVENTED HERE, per the spec's own
        FAIL branch. Bars untouched; HR.2–HR.4 stay killed as the kills
        field says. What this desk now owns on this row: the family fate —
        and note before any next design that the venue is one clip from
        its bar on the worst seed, i.e. any repair is a STATISTIC or VENUE
        question under the UNSATURATED-NULL rule, never an envelope one.

ROUTED: ps06-legibility-probe-collapses-on-one-mutated-world | 2026-09-19 | `5fd96a7` (PS.06 attempt 1, FAIL, seeds 0/1/2, clean stamp at 9b40588) | OPEN
    PS.06 measured the `tiring` commitment (GOAL.md:187) and split it cleanly
    in two. THE WORLD HALF IS GREEN ON EVERY SEED: fatigue_gap 0.295 ± 0.003
    against max(0.10, 2× quantum 0.032) — 9.3× the quantum; f_spent 0.529
    (floor 0.35); the fatigue-frozen twin near-flat (frozen_spent_ratio 0.862
    vs 0.75, twin_f_max 0.0); rest repays twin-differenced (rec_gain_diff
    0.272 vs 0.08 — live +0.174, frozen −0.099); tau_fit 59.9 s in the
    (45, 75) band with std ~0; rig_ok 1.0, zero deaths; the sensory-amputation
    control failed as required (control_r2 0.033 vs 0.30 cap, margin 0.449 vs
    0.15). Exertion is priced, above the outcome quantum, and repaid by rest —
    tiring is real in W0, distinguished from PS.03's damage by the repayment
    clock. WHAT FAILED, and it is exactly one conjunct: seed_gates_ok 0.667 —
    one seed's PROBE_R2 fell under the 0.35 per-seed floor. probe_r2 reads
    0.482 ± 0.343 over 3 seeds while every other per-seed conjunct has
    std ≤ 0.01, which puts the worst seed's legibility near ZERO against the
    disjoint seed-90 pilot's 0.674 (shuffled null −0.116 ± 0.019, clean on all
    seeds). NOTE, and it is a live instance of `aggregate-hides-worst-seed`
    (OVERDUE on this desk): the ledger stores means/stds only, so the worst
    seed's exact value is arithmetic, not a recorded number. The finding:
    whether the imminent droop is legible beforehand from interoception is
    WORLD-DEPENDENT (seed>0 mutates PlaygroundParams), and the claim's
    all-seeds bar refused to average over that — correctly. Candidate readings
    for the disposition, named so it is a bakeoff and not an argument:
    (a) probe capacity — one fixed RFF draw (N_RFF 200, RFF_SEED) may be a
    per-world lottery; a small pre-registered draw ensemble would separate
    probe variance from world variance at zero new physics; (b) data
    starvation — ~52 train rows per world with session-level holdout may be
    under-powered exactly where the world's fatigue dynamics are least
    redundant; (c) the honest reading: interoceptive legibility genuinely
    varies by world and the spec should measure WHICH world property predicts
    it before any bar is touched. NO bar moves; NO re-run unchanged (a redraw
    is a seed lottery). Staleness bill: ZERO green certificates — PS.06 is red
    on attempt 1 and nothing cites it.
    DUE: 2026-09-24 | redesign disposition, the Review's (09-22 already
    carries A4 + T2.10 + hr1; first free day per the router's own capacity rule)

ROUTED: ps05-legibility-holdout-is-a-band-lottery | 2026-09-19 | `cc38749` (PS.05 attempt 1, FAIL, seeds 0/1/2, clean stamp) | OPEN
    PS.05 measured the `far` commitment (GOAL.md:187) and split it exactly the
    way its sibling PS.06 split `tiring`, five hours earlier, same day. THE
    WORLD HALF IS GREEN ON EVERY SEED: traversal cost monotone in distance on
    all 3 seeds (cost 0.0159 -> 0.0693 across the registered 1.5/3.0/4.5/6.0 m,
    per-distance stds <= 0.0024); near-vs-far gap 0.0534 ± 0.0022 against
    max(0.015, 2x quantum 0.0027) — ~20x the within-distance repeat spread;
    the teleport twin's meter alive (0.0112) and DEAD-FLAT across distances
    (spread 3.6e-6, four orders under the quantum); reach_ok 1.0 everywhere
    (the power-coupled alive-proof fired nowhere); zero deaths. Distance is
    priced in the needs' own currency, above the outcome quantum, and the
    price does not survive teleportation. `far` moves from unmeasured to
    MEASURED. WHAT FAILED: the legibility conjunct, on TWO of three seeds —
    per-seed probe_r2 (re-derived, the ledger stores means): seed 0 -1.126,
    seed 1 +0.356 (all gates green), seed 2 -0.362, vs the disjoint seed-90
    pilot's 0.707; control caught on ALL seeds (amputated probe never beat
    its cap after two rig repairs measured in at the pilot). THE MECHANISM IS
    VISIBLE IN THE ROWS, which is what distinguishes this from the sibling:
    each meandering survey trip occupies a NARROW BAND of remaining-distance
    (~0.2-0.4 wide on a 0-1.4 target), the holdout is by-trip with 4 test
    trips, and R^2 is scored against the test-set mean — so when the test
    draw lands far-heavy (seed 0's test bands 0.61-1.21, where C = e^(-d/2)
    reads 0.03-0.05 and between-trip nuisance variation dominates), the probe
    misses the test mean and the headline goes strongly negative. The
    shuffled null's own wild variance (-4.6/-0.4/-1.9 per seed) says the same
    thing about the estimator. Candidate readings for the disposition, named
    so it is a bakeoff and not an argument: (a) the estimator — trip-level
    holdout with 4 test units makes the headline a lottery over which bands
    the test trips occupy; a band-stratified holdout or more/shorter trips
    is a rig redesign, zero new physics; (b) genuine range limit — odour
    legibility may honestly die past ~5 m at this LAMBDA_M and noise floor,
    and the claim's D_LEG (1, 6) m simply spans past the sense's edge, in
    which case the spec should measure WHERE legibility ends rather than
    average over it; (c) probe capacity, exactly the sibling's reading (a).
    SHARED CLASS, dispose together: this row and
    ps06-legibility-probe-collapses-on-one-mutated-world are the same defect
    family (PS-family legibility conjunct collapsing on a seed subset while
    the world half is green on every seed); one disposition should cover
    both or say why not. NO bar moves; NO re-run unchanged (a redraw is a
    seed lottery). Staleness bill: ZERO green certificates — PS.05 is red on
    attempt 1 and nothing cites it.
    DUE: 2026-09-24 | redesign disposition, the Review's (bundled with the
    sibling row already dated there)

ROUTED: ps09-probe-memorizes-trips-while-a-bare-threshold-reads-the-sign | 2026-09-19 | `7012e84` (PS.09 attempt 1, FAIL, seeds 0/1/2, clean stamp) | OPEN
    PS.09 measured the `worth-it` commitment (GOAL.md:187) and split it the
    way its two siblings split `far` and `tiring`, earlier the same week.
    THE WORLD HALF IS GREEN ON EVERY SEED: across the four registered
    (payoff, distance) offers the net need-gain is +0.0394/+0.0594 on the
    positive side and -0.0397/-0.0634 on the negative, per-offer stds
    <= 0.0035 across three mutated worlds, weakest witness ~12x the 0.0032
    quantum; the free-lunch twin (teleport + identical eat dwell) is
    POSITIVE on every offer and every seed (min +0.0462) so the negative
    branch is a price-vs-payoff trade, not an accounting artifact; every
    trip ate exactly k through the layer's own mouth gate; rig std 0.0 on
    every gate. W0 pays and charges in one currency and some trips do not
    pay — measured. WHAT FAILED: probe_bal_acc 0.547 ± 0.075 against the
    0.65 bar (disjoint seed-90 pilot: 0.78), shuffled 0.499, amputated
    control 0.35 (caught on every seed). AND THE DIAGNOSIS IS SHARPER THAN
    EITHER SIBLING'S, because the venue signal was measured SEPARATELY from
    the registered estimator on seed 1: a bare threshold on the odour
    concentration channel alone, learned on the train trips, reads the
    held-out sign at balanced accuracy 1.00 — while the registered
    RFF+ridge probe on the same rows reads 0.60, train 0.89, calling four
    of five clear positives negative (C = 0.070, 0.164, 0.048 — far outside
    the k-d ambiguity band). The mechanism: 5 near-duplicate pre-departure
    sniff rows per trip x 54 pose/interoception features that individuate
    the trip = the fit memorizes trip identity, and at test its predictions
    collapse toward the train mean. So this is not a range limit and not a
    band lottery: THE SIGN IS PERFECTLY LEGIBLE IN THE SENSE AND THE
    REGISTERED INSTRUMENT CANNOT READ IT. That is the strongest evidence
    yet that the family's legibility conjunct is measuring the estimator,
    not the venue. SHARED CLASS, dispose together with
    ps05-legibility-holdout-is-a-band-lottery and
    ps06-legibility-probe-collapses-on-one-mutated-world (both already DUE
    09-24): one disposition should cover all three or say why not. The
    legal repairs are the ones the 103rd audit item 5 already named — the
    estimator, the probe draw, or measuring where legibility ends — and the
    seed-1 threshold datum says the estimator arm now has a measured
    known-answer control any redesign must pass. NO bar moves; NO re-run
    unchanged (a redraw is a seed lottery). Staleness bill: ZERO green
    certificates — PS.09 is red on attempt 1 and nothing cites it.
    DUE: 2026-09-24 | redesign disposition, the Review's (bundled with the
    two sibling rows already dated there)

ROUTED: ps08-amputation-control-out-reads-the-probe-intero-is-not-clock-like | 2026-09-19 | `8f7d1dc` (PS.08 attempt 1, FAIL, seeds 0/1/2, clean stamp) | OPEN
    PS.08 measured the `heavy` commitment (GOAL.md:187) and split it the way
    all three siblings split `far`, `tiring` and `worth-it` this same week.
    THE WORLD HALF IS GREEN ON EVERY SEED: cost per registered 0.4 m of
    displacement is STRICTLY MONOTONE in object mass on every seed
    (0.00977 / 0.01525 / 0.02320 / 0.03143 at 0.2/0.9/1.7/3.0 kg, mono_ok
    std 0.0), the equalised-mass twin — same schedule, same paired ctrl
    draws, true mass pinned light — is FLAT (spread 9.3e-05 vs the live gap
    0.0217), the same thrash buys 8.65 m at 0.2 kg and 2.76 m at 3.0 kg,
    and the rig is green everywhere (displaced/still/stray std 0.0,
    fresh_frac 0.995). Moving mass costs need-currency and costs more per
    kilogram — measured, not asserted. WHAT FAILED: probe_bal_acc 0.583 ±
    0.059 against the 0.70 bar (disjoint seed-90 pilot: 0.875), shuffled
    0.517. AND THE FAILURE MECHANISM IS NEW — THE FOURTH SIBLING BREAKS THE
    CONTROL, NOT JUST THE PROBE: the amputated "clock-only" control READ
    0.708 ± 0.156, ABOVE the registered probe, and was caught on only 1 of
    3 seeds (pilot: 0.50, caught). The 8 interoceptive dims are not
    clock-like in this venue: a limb blocked by mass does less |tau*omega|
    work, so the e/w DRAIN RATES encode the load class in the very currency
    the claim prices — the venue's cost signal leaks into every channel
    that integrates power, and no jitter of e0/w0/t0 can mask a RATE. The
    family's amputation-control idiom ("drop the load-bearing sense, the
    probe must fail") assumes the remaining channels are inert; PS.08
    measures that assumption false where the priced quantity IS power.
    Second datum, PS.09's lesson recurring in mirror image: the registered
    single-statistic |qvel| threshold — chosen BECAUSE PS.09 measured the
    RFF collapsing — lost its pilot margin on mutated worlds (0.875 ->
    0.583) while the reported RFF diagnostic read 0.646, ABOVE the gated
    probe this time. A pilot certifies the draw, whichever estimator is
    gated. SHARED CLASS, dispose together with
    ps05-legibility-holdout-is-a-band-lottery,
    ps06-legibility-probe-collapses-on-one-mutated-world and
    ps09-probe-memorizes-trips-while-a-bare-threshold-reads-the-sign (all
    DUE 09-24): one disposition should cover all four or say why not, and
    THIS row adds the control-side defect to the estimator-side one — a
    redesign that fixes the probe but keeps the inert-channel assumption
    will pass instruments and still measure nothing. NO bar moves; NO
    re-run unchanged (a redraw is a seed lottery). Staleness bill: ZERO
    green certificates — PS.08 is red on attempt 1 and nothing cites it.
    DUE: 2026-09-24 | redesign disposition, the Review's (bundled with the
    three sibling rows already dated there)

ROUTED: lt02-the-venue-has-no-true-positive-body-chaos-is-reducible | 2026-09-19 | `6fff04f` (LT.02 attempt 1, FAIL, seeds 0/1/2, clean stamp) | OPEN
    LT.02 exists to certify the instrument that can see the noisy-TV-is-your-
    own-body failure (PG.4's blind spot) before CU.3 and the LT.03+ arms may
    claim "his curiosity is not trapped". The registered run FAILed on C1
    ALONE: the deliberate body-chaos farmer — PG.4's textbook construction
    transplanted onto the LT.01 rover with the panel deleted, clutter zeroed
    and adhesion off, so its own contact dynamics are the only irreducible
    surprise on offer — read chaos_occupancy 0.145 +/- 0.122 against the
    registry's 3.0 bar. Every other conjunct held: the known-positive control
    fired BOTH detectors (chaos_occupancy 6.198 >= 3.0 AND panel_dwell 0.667
    > 0.4), the hard-working climber was exercised and cleared (rise 0.786 m,
    100 falls, occupancy 0.083 <= 1.0), C2 held (reward_ratio 5.32), C3 held
    (dwell 0.000), and all four VOID lanes were green (OOF median 0.088 vs
    persistence 1.802; theta 1.205; random occupancy exactly 1.0). So the
    DETECTOR is measured alive on the one agent this project has proven
    trapped, and the pre-registered branch that fired is the registry's own
    words: "the detector is blind — no arm's immunity may be reported and
    the LT.03+ chain stays honest."
    THE MECHANISM IS THE PILOT'S DISCLOSED FORECAST, CONFIRMED: on this body,
    at this obs resolution, self-generated contact chaos is REDUCIBLE —
    e_mean(icm) 0.148 vs e_mean(random) 0.552, reducibility_gain(icm) 0.134,
    thrash_ratio(icm) 0.385 (the farmer moves LESS than random) — while the
    panel's chaos is genuinely irreducible (pilot gain 0.0134). A policy that
    farms a spot concentrates data exactly where the pooled model then
    masters it: self-surprise is SELF-EXTINGUISHING unless the source is
    genuinely stochastic. The docstring's founding premise ("ragdoll contact
    dynamics are an inexhaustible source of irreducible prediction error")
    is falsified by measurement. The venue cannot produce the threat the
    detector exists to catch — a detector with no reachable true positive in
    its own certification venue cannot be certified there, however alive its
    controls prove it.
    DESIGN INPUT, one line: this is the curiosity-side sibling of the
    body-instrument list already named on w0-too-shallow (LT.01's
    nonladder_rise 0.084 m, UB.14's eye at 0.159, W0.BAL's 0.002-0.004
    upright, BA.03's blind twin) — the body is too LEARNABLE to be a noise
    trap, one more reading that the substrate under the curiosity ladder is
    shallower than its specs assume. Counted there, not here.
    THE REPAIR IS A REDESIGN, the Review's, and the fork is narrow: (a) give
    the certification venue a genuinely stochastic self-carried noise source
    (the body-mounted analogue of the panel — a true positive the detector
    MUST flag, with the climber and known-positive kept as-is), or (b)
    accept the registry's falsified_by as final for this venue and re-scope
    what LT.03+ may cite (they inherit "no immunity may be reported", which
    is the honest reading the branch already enforces). NO bar moves; NO
    re-run unchanged (a redraw is a seed lottery against a mechanism three
    independent diagnostics localise). Staleness bill: ZERO green
    certificates — LT.02 is red on attempt 1 and nothing cites it. Block
    mass, live at routing: frees 6 (LT.03-LT.07, LT.09) / blocks 8, the
    third-largest FAIL mass on the board.
    DUE: 2026-09-24 | redesign disposition, the Review's

## ROUTED 2026-09-20 (Review FULL): `ba03-vestibular-channel-is-never-load-bearing-under-one-kick`
## — split out of `ba03-null-saturates-the-horizon` so that adopting the cheap
## metric repair does not quietly retire the expensive scientific question

ROUTED: ba03-vestibular-channel-is-never-load-bearing-under-one-kick | 2026-09-20 | 9e7cc86 (BA.03 attempt 1 ANATOMY table) | OPEN
    **Why this row exists.** Today's bundled ruling adopted `BA.03` option (c)
    — change the saturating statistic — and refused option (b), hardening the
    perturbation, on COST grounds alone: (b) touches `playground.py`, bills the
    21 certificates listed at the head of this file plus `BA.01`, and belongs
    in a world-edit window that is itself undesigned. A refusal on cost is not
    a refusal on merit, and a question refused on cost with no row left behind
    is a question this project has silently decided. This row is the
    non-silence.

    **The measurement, carried verbatim from the parent row so it survives
    independently.** `BA.03`'s own ANATOMY table reads: the winning vest policy
    uses PLANTAR TOUCH and nothing vestibular — deleting touch costs it 7.3 s,
    deleting any true vestibular block costs it NOTHING. One kick per episode
    is survivable by a purely plantar route. So BALANCE — a zero-pass GOAL.md
    commitment with three declared specs — currently has no venue in which a
    graviceptive channel can earn its parameters, and the metric repair adopted
    today does not change that: integrated tilt under one kick is still
    survivable by touch.

    **What is owed, and by whom.** This is a VENUE question, so it is owed BY
    THE WORLD-EDIT WINDOW, not by a builder slot: a repeated or larger
    disturbance regime under which the plantar route is insufficient, so that
    an ablation of the vestibular block has somewhere to show a cost. It is
    NOT a request to weaken `BA.03` and it moves no bar — it asks for a harder
    world, which is the direction this desk is permitted to move things.

    **Sequencing, declared:** do NOT start this before `BA.03`'s (c) redesign
    lands. If integrated tilt under the EXISTING one-kick regime already
    separates the vestibular ablation, this row is discharged for free and the
    21-certificate bill is never paid. Measure first.
    BLOCKED-BY: w1-world-edit-window
    DUE: 2026-09-27 | pick the disturbance regime, in the world-edit window and
        not before it, and only if BA.03's (c) re-run has not already answered
        it. Dated onto the same day as the three rows it was split from so the
        world-edit bill is read as ONE bill, per the bundling rule.

ROUTED: lc03-five-controls-never-switch-off-the-term-a4-is-named-for | 2026-09-21 | `785f921` (field watch week 8, §6) | OPEN
    **The question, and it is a SEAT question before it is a spec question.**
    `A4` (`wm-latent`) holds the Learning-core seat. The machinery it is NAMED
    for is the `latent_pred` head — 149,312 params, 17.3% of the arm. The scout
    measured at HEAD that all five of `LC.03`'s controls switch that term off
    only as part of switching EVERYTHING off (untrained twin, frozen control),
    so **no control isolates the latent-prediction objective.** The seat's
    margins (`lg_margin_null` t = 4.64, `lg_margin_twin` t = 4.00) are therefore
    equally consistent with an RSSM actor-critic having produced the entire
    number and 17.3% of the seated arm being dead weight.
    This is NOT week 7's §6 restated: that one found `A4`'s declared COLLAPSE
    diagnostic does not exist (row `a4-mandatory-collapse-diagnostic-is-declared-
    and-computed-nowhere`). This is about what its CONTROLS remove. Same seat,
    different hole, and the two should be read together, not merged.
    **Why the Review owns it rather than the builder:** the candidate repair is
    an `A4` variant with `l_bind` dropped — a paired comparator this seat has
    never had — and choosing an arm is this desk's, never a slot's.
    **The staleness bill.** `LC.03` is a VOID and the seat is held `BY VERDICT`
    off it (`champions --check` VERDICT-IS-A-VOID, 2 of 2, at floor), so no PASS
    certificate is invalidated by acting here. The cost is compute, not
    re-certification: 14.40 core-h per 3 seeds, no weights on disk [scout, M].
    That is the whole reason this is a decision and not a slot's errand.
    **Sequencing, declared:** if the `a4-…-computed-nowhere` row takes an option
    that runs training, this comparator and field watch N1's `k`-sweep are both
    MARGINAL on that run. Do not buy 14.40 core-h twice. Read that row first.
    DUE: 2026-09-28 | a design answer owed by the Review: does the Learning-core
        seat get an `l_bind`-dropped comparator, and is it bought on its own or
        marginal on the `a4-…-computed-nowhere` run. Dated a week out and BEHIND
        that row deliberately — buying this separately is the expensive mistake.

ROUTED: fieldwatch-quotation-channel-is-0-for-5 | 2026-09-21 | `785f921` (field watch week 8, §6b) | ACTED 2026-09-24 (Review DAILY, executing commit `d901cb4` — the builder EXECUTED this on 09-23 and the row has been sitting one day OVERDUE waiting only for this stamp. Verified against the commit, not against its own UPDATE note: the diff touches `experiments/fieldwatch.py` and NOT `experiments/decisions.py`, which is this row's own load-bearing caution honoured; all three things the `DUE:` bought are present — the FP rate re-measured after the fix and written down (0 spurious of 18 sub-threshold pairs, both true pairs route), the closure picked only after 20 live overlaps were enumerated, and the shared helper left alone with the threshold parameterised at the fieldwatch call site. The measurement also bought something the row did not ask for and could not have: `decisions.owner_asks` has parsed 0 items since 09-09, routed separately as `owner-ask-reader-blind-since-0909`)
    **The instrument built last week to read `FIELD_WATCH.md` returns a FALSE
    GREEN, and the scout measured it rather than writing around it.**
    `experiments/fieldwatch.py`'s quotation channel reported both week-8
    findings as ROUTED. All five routings are spurious: four are stock English
    six-grams shared by two desks writing in the same house style (*"it is the
    same shape as"*, *"why this is not a one"*, *"so it is not mistaken for"*),
    and the fifth is the row's own slug — `_queue_chunks` starts each chunk at
    `ROUTED: <slug> |`, so **naming the row you are distinguishing yourself FROM
    marks you as owned BY it.** Quotation channel: **0 for 5** at n = 5.
    **The direction is the dangerous one.** The 96th audit predicted the
    false-POSITIVE shape would be a finding discharged in code reading UNROUTED.
    The measured shape is the opposite: a finding with **no owner at all**
    reading ROUTED. `0 UNROUTED-FIELD-FINDING` in `run status` is currently a
    green light sitting on top of the exact scar the module was built to end.
    **The citation channel is NOT implicated** — `_cites` is week-anchored and
    slug-independent, and week 7's two findings both routed through it
    correctly. Do not repair what is working.
    **Named by the scout, whose call it is not:** a minimum-overlap count;
    stripping the `ROUTED:` header line from the chunk; subtracting shingles
    that also occur in the previous sweep's page.
    **The staleness bill.** `fieldwatch.py` is reporting-only and unfloored by
    the 96th audit's explicit instruction, so no gate, no threshold and no
    verdict moves. Bill is whatever `T0.31`-class re-buys the edit stales —
    ordinary, and named in the commit that makes it.
    **A caution this desk adds on top of the scout's report:** the shingle rule
    is IMPORTED from `decisions.owner_asks`, deliberately, so the two readers
    cannot drift. A repair here that edits the shared helper therefore silently
    re-tunes the OWNER-ASK reader — the instrument that audits this desk's own
    `FOR THE OWNER` section. **Fix it behind a parameter at the fieldwatch call
    site, or measure the owner-ask channel's false-positive rate in the same
    commit.** Do not change the shared rule blind.
    DUE: 2026-09-23 | a BUILDER repair, not a Review design — the scout named
        three candidate closures and this desk endorses measuring before
        picking. What is owed by then: the false-positive rate re-measured
        after the fix, written down, and the shared-helper caution honoured.
    UPDATE 2026-09-23 (builder): EXECUTED, measured first as ordered. Every
        (finding x desk-chunk) overlap on the live corpus was enumerated and
        adjudicated before a closure was picked: 20 overlapping pairs; the 18
        spurious ones cap at 4 shingles once `ROUTED:` header lines are
        stripped (stock phrases 1-2, row-title mentions 3-4 — bounded by
        slug/title length), the true quotations read 12 (fixture), 18 and 66.
        Closure = scout's (a)+(b): strip the header line, then require
        `MIN_QUOTE_OVERLAP = 6` shared shingles (~11 consecutive words — more
        than any slug or title carries, half the smallest true quote). (c)
        (previous-sweep subtraction) NOT taken: no measured FP needed it and
        it costs a git-history read every render. POST-FIX RATE, written
        down: 0 spurious of 18 sub-threshold pairs route; both true pairs
        route (and are superseded by their own citations, so the live block
        reads 2 cited / 0 quoted / 0 unrouted — now honestly). Shared-helper
        caution honoured by construction: `decisions._shingles` untouched,
        threshold and stripping live in `fieldwatch.py` only; the false-
        negative trade (a one-clause re-worded routing now reads UNROUTED)
        is recorded in the module docstring as the safe direction for a
        reporting-only counter. Fixture `_FIXTURE_QUEUE_SPURIOUS` replays
        all three FP mechanisms at once and asserts the old one-shingle rule
        WOULD have tripped on it (raw >= 6 > stripped > 0), so neither half
        of the repair can silently rot. Measuring first also caught a
        sibling: `decisions.owner_asks` parses 0 items on every PROGRESS.md
        revision since 2026-09-09 (last nonzero 09-08) — routed separately
        as `owner-ask-reader-blind-since-0909`.
        LANDED: commit `d901cb4`, wall-clock end 2026-09-23 12:16:51 UTC —
        named here so the (a) OVERDUE FIRST sweep can stamp ACTED in one read.

## ROUTED 2026-09-21 (builder): `d27-screen-measures-95-percent-false` — the
## armed default ordered a screen, the screen is built, and its own measured
## false-positive rate says it cannot be floored

ROUTED: d27-screen-measures-95-percent-false | 2026-09-21 | `D27` firing (overseer, 107th audit) + `experiments/unread_metrics.py` | OPEN
    **What happened, in one line.** `D27`'s armed default (i) BUILD THE SCREEN,
    REPORTING ONLY fired on 2026-09-21 and is now discharged in code:
    `metric_recorded_but_unread` is in `run status`, unfloored, first reading
    **580 metrics on 62 of 96 decidable certificates**. The entry's own text
    binds the firing to a rate measurement — *"the counter gets floored or
    deleted once it exists"* — and the rate is **19 false / 1 true in a
    deterministic 20-draw, 95%**. The full adjudication table is in
    `docs/DECISIONS_RESOLVED.md` under `D27`.

    **Why this is a decision and not a slot's errand.** At 19/20 the counter
    **cannot be floored**, which leaves exactly two dispositions and both are
    this desk's: narrow it, or delete it. Deleting an instrument that the
    owner's own armed decision ordered built, on the builder's own evidence,
    is not a builder's call — that is the shape of a desk quietly reversing a
    default it did not like. So it is routed with the number attached.

    **The fact that should decide it.** The single true positive is **not an
    instance of `D27`'s cited class.** `T0.06 steps_ok` is the literal `5`
    written into a metrics dict and never read — a *decorative metric*, not
    *"the run measured the quantity that would have indicted it and then did
    not look at it."* In a 20-draw the screen found **zero** instances of the
    class it was built for. `T6.03`, `T1.07 spread_ratio` and
    `T1.08 min_detectable_effect` — the four instances in four days that
    motivated the fork — are not in the residual: `T1.07` was repaired on
    09-13, `T1.08` is a FAIL and out of scope, and the rest read as summarised.

    **The narrowing that is visible in the data and was deliberately NOT
    taken.** The one true positive was found by a property no filter uses: the
    recorded value is a LITERAL in the source. A decorative-metric detector is
    mechanical and has no false positives by construction. It is also a
    DIFFERENT screen from the one `D27` ordered, and the builder declined to
    substitute it — that would be answering a question nobody asked while the
    asked one goes unreported.

    **The known miss shape, named so it is not rediscovered.** Two of the three
    pairs that looked real were read in substance through a SUBSCRIPT
    (`rb["refused"]`, `m["synth_lo_refused"]`) rather than a bare identifier,
    so the syntactic dataflow filter has no shared `ast.Name` to join on. Any
    narrowing proposal should price that first: following subscripts and helper
    calls is the difference between 62 flagged specs and an unknown smaller
    number, and nothing here has measured it.

    **What is NOT owed.** No bar moves, no certificate is invalidated, nothing
    re-runs. The screen refuses nothing today and will refuse nothing under any
    disposition — `D27`'s default is reporting-only and this row does not ask
    to change that.
    DUE: 2026-09-28 | narrow the screen (naming which channel — subscripts,
        helper calls, or a different question entirely), or DELETE it and say
        so in `DECISIONS_RESOLVED.md` under `D27`. A third sitting that leaves
        a 95%-false counter printing in `run status` with no disposition is the
        outcome this row exists to prevent.

ROUTED: dark-slot-counter-is-blinded-by-the-loops-own-notice-lines | 2026-09-22 | Review DAILY (replayed against `/data/jack-logs/ladder.log` at 07:0x) | OPEN
    DUE: 2026-09-29 | the DESK's half: rule on whether `slot_outcomes()` should
        become the ONE reader of what a slot line is, so a third liveness
        counter cannot be blinded by a fourth kind of line. The builder's half
        (the repair itself) is ordered on `scripts/ladder_prompt.md` for today
        and does not wait for this date. 09-29 carries 3 live rows against the
        measured 6.
    Question: `dark_slots` — the skipped-slot streak — has read **0 through a
    15-slot skip streak since 2026-09-21T12:07**, and the blinding agent is
    this loop's own output.

**THE MECHANISM, replayed rather than argued.** `scripts/usage_attribution.py`
`attribution()` walks the log backwards:

    if "PACING:" in line: streak += 1
    elif line[:4].isdigit(): break       # "a real slot line ends the streak"

Every pace-skipped slot writes its `PACING:` line and then
`notice_exited_dispatches()` appends one `PACE-SKIP NOTICE:` line per EXITED
declared dispatch. Those notices are timestamped, so they begin with four
digits, so the NEXT slot's backwards walk hits one first and breaks before it
ever reaches a `PACING:` line. The streak is structurally pinned at 0 for as
long as any EXITED row sits in `declared_pids`.

**MEASURED, not inferred.** Replaying `attribution()` against the real log
truncated to each slot boundary: `2026-09-21T13:07 -> 0`, `18:07 -> 0`,
`2026-09-22T03:07 -> 0`, against true streaks of 1, 6 and 15. The first
EXITED stamp landed 2026-09-21T11:26:15 (the two `run_spec T0.36` dispatches);
the last correct reading is 2026-09-21T10:07 (`3 consecutive dark slot(s)`),
and the counter has been pinned ever since. On 09-20, with no EXITED rows, the
same code counted a 17-slot streak correctly — so this is a regression in the
log's CONTENT, not in the counter's arithmetic, which is why no test caught it.

**WHY IT MATTERS MORE THAN ITS SIZE.** This is the SAME defect the 107th audit
repaired one layer over, on the same line of output, yesterday. That repair's
own comment reads: *"`dark_slots` counts slots that were SKIPPED, which is a
real thing and is left alone."* It was already blind when that sentence was
written — 12:07 the previous day. Two of the three liveness phrases on the
builder's pace line have now been blinded by the same root cause: **a "slot
line" is identified by `line[:4].isdigit()`, and every line this loop writes
starts with a timestamp.** `failed_slots` and `hours_since_rc0` are correct
today only because they parse `iteration start`/`iteration end` explicitly.

**AND THE READING THAT CUTS THE OTHER WAY.** Nothing was lost this time,
because `D30`'s armed default orders THIS desk to count the streak from
`ladder.log` by hand every sitting and not to take the counter's word for it.
That standing instruction has now caught two instrument failures in three days.
It is the cheapest liveness instrument this project owns and it is the only one
that does not depend on the code being right about what "a slot" means.

    Staleness bill: NONE. `scripts/usage_attribution.py` is in no spec's
    IMPL_DEPS (grepped, 0 hits), so the repair stales no certificate and
    re-buys nothing.

ROUTED: world-edit-window-price-is-quoted-at-21-and-measures-35 | 2026-09-22 | `experiments/stale_cost.py` (builder, first reading) | OPEN
    **The most expensive instrument this project owns is priced on three desk
    pages from a cached number, and the number has grown 67% underneath it.**
    `docs/DECISIONS_NEEDED.md:7611`, `docs/REVIEW_QUEUE.md:1018/2865/2895` and
    `scripts/ladder_prompt.md:735` (the LIVE `2^10` prohibition block) all
    price the world-edit window as
    *"21 `playground.py` certificates plus `BA.01`"*. Priced today from the
    `IMPL_DEPS` declarations that `impl_sha_of` actually hashes, a
    `playground.py` edit stales **35 standing PASS certificates** — and
    **`BA.01` is one of the 35**, so the quoted figure understates the bill
    and double-counts its own example. 20 further non-PASS rows also cover
    `playground.py` and cost nothing (staling a FAIL refutes no claim).
    Cost classes of the 35: 24 `cpu<10min`, 6 `cpu<2h`, 2 `cpu<1min`,
    2 `gpu<2h`, 1 `gpu<20min` — so the bill is **~3 GPU-hours plus a
    full CPU re-gate**, not a footnote.
    The 21 was correct when written (2026-08-31); `BA.01`, `VO.01`, `VO.02`,
    `SM.01`, `LT.01`, `W0.DIAG` and the `PS`/`SO` families joined the set
    after it. **This is a correction to a PRICE, not a position on the
    design** — the window, `W1.01`/`W1.03` registration and `HR.5`'s world
    edit are the Review's under `2^10`, and nothing here pre-empts them.
    Reproduce in one line, never from this row:
    `run stale-cost playground.py`.
    STALENESS BILL OF THIS ROW ITSELF: none. Editing the prose on three desk
    pages stales no certificate — none of them appears in any `IMPL_DEPS`.
    DUE: 2026-09-29 | correct the figure wherever the window is priced, or
        record why 21 is the right number and 35 is not. Whoever next quotes
        the bill should quote the tool, which is the general repair.

ROUTED: t306-random-arm-breaches-the-analytic-chance-dwell-bound | 2026-09-22 | builder `3c07448` (T3.06 attempt 2, VOID, 2432 s, 3 seeds) | OPEN
    DUE: 2026-09-30 | a VENUE ruling owed by the Review, and it is NOT a rig
        repair: does W0's dwell distribution disagree with the analytic null
        because the null is wrong, because the world is, or because 48 lives
        at 16.3 informative cannot resolve it? 09-30 carries 3 live rows.
    BLOCKED-BY: w1-world-edit-window | whatever the edit window rules about W0's dynamics decides whether this is a null-model repair or a world one
    Question: the n-derived `RANDOM_DWELL_MAX` = 0.0185, computed in source
    from the stationary occupancy of the null walker before the run, was
    BREACHED BY THE NULL WALKER ITSELF on 2 of 3 seeds (`random_dwell_breach`
    0.667, `random_dwell_worst_life` 0.0165, `random_dwell` mean 0.00303).

**The cap is arithmetic and is not in question.** It was derived 2026-09-21
(`875caf6`) from the stationary occupancy → exact residence-run pmf → compound
Poisson → `(1-α)^(1/n)` tail with α fixed before any cap was computed, and
cross-checked against Kac's identity to 4 dp. **A derivation that the world
then violates is the most useful kind of red**: it says the model of the world
this project reasons with is not the world it simulates.

**AND THE READING THAT MAKES THIS A W0 ROW RATHER THAN A T3.06 ROW.** The same
run measured `coverage_curious` 0.6162 against `coverage_random` 0.6037 —
curiosity does not beat a random walker at coverage here — and
`task_cov_vs_random` −0.2333, the task arm exploring WORSE than random. **In a
world where a random walk is already near-ceiling on coverage, no exploration
policy has room to demonstrate anything**, which is the same disease as the
saturated nulls that foreclosed `SH.02`, `BA.03`, `DP.04` and `UB.10`, arriving
this time through the dwell channel instead of the coverage one. Counted
against `w0-too-shallow`'s instrument list, **this is the eighth independent
instrument** and the first to say it about a NULL rather than about an arm.

    Staleness bill: NONE unless the world moves. A null-model repair edits
    `t3_06_*.py` only (T3.06 is VOID; no certificate rests on it). A WORLD
    repair bills every `playground.py` certificate and must go through the
    edit window, which is why this row is dated behind it rather than beside it.

## ROUTED 2026-09-23 (builder): `owner-ask-reader-blind-since-0909` — the
## UNROUTED-OWNER-ASK class has read empty for 15 days because the page's
## item format moved out from under its parser

ROUTED: owner-ask-reader-blind-since-0909 | 2026-09-23 | measurement in the `fieldwatch-quotation-channel-is-0-for-5` execution (builder) | OPEN
    **What was measured, incidentally, while honouring that row's shared-helper
    caution.** `decisions.owner_asks` parses **0 items** on `docs/PROGRESS.md`
    at every revision from 2026-09-09 through today — the live page carries
    FIVE items under `## FOR THE OWNER`. Last nonzero parse: 3, on the
    2026-09-08 revision. Replay:
    `for c in $(git log --format=%h -- docs/PROGRESS.md); do git show
    $c:docs/PROGRESS.md | python -c "...print(len(owner_asks(stdin)))"; done`.
    **The mechanism, one line.** `decisions._ITEM` is `^(\d{1,2})\.\s+(.*)$` —
    the digit at column 0 — and the Review has written its owner items as
    `**1. NO-DECISION: ...**` (bold marker before the number) since 09-09.
    Same class as the fieldwatch 0-for-5 and the dark-slot counter: a reader
    whose population selector quietly stopped matching, reporting an empty
    class as a green one. `BASELINE_UNROUTED_ASKS = 3` cannot fire on a page
    that parses as zero asks, so `UNROUTED-OWNER-ASK` and `VANISHED-OWNER-ASK`
    have both been structurally silent for 15 days — including through the
    09-22 blackout sitting, when the page carried a `D33` addendum and three
    NO-DECISION items nothing verified.
    **Why this is routed and not fixed in the same commit.** Two instrument
    edits in one slot is how one goes unverified (this desk's own standing
    reasoning), and this reader audits the Review's `FOR THE OWNER` section —
    re-arming it changes what counts against a floored ratchet, which the
    executing slot should verify in isolation. The repair looks like one line
    (`_ITEM` admits `\*{0,2}` before the digit, or `_tokens`-style markdown
    blindness at the line head) plus a fixture in the `**N.** ` shape the live
    page actually uses — and a re-read of what the re-armed counter says
    about the CURRENT page before it lands, because it may go red > 3 the
    moment it can see again, and that red is information, not a bug.
    **PREDICTION CORRECTED AT EXECUTION (builder, 2026-09-25 01:x, per the
    115th audit's 1a, which licensed the early slot this row itself invited):
    the re-armed counter's first honest reading is 0 UNROUTED / 0 VANISHED,
    not > 3** — the live page's four items are two `NO-DECISION:` exemptions
    and two cite-attributed reports (PROGRESS #2 by D22, #4 by D31), so no
    red fired when the reader could see again. The floored ratchets re-armed
    without moving. Repair executed exactly as this row sketched: `_ITEM`
    admits `\*{0,2}` before the digit; fixture in the live `**N. ` shape
    asserts parse, exemption and cite in both directions; T0.28 re-bought.
    DUE: 2026-10-01 | builder repair; cheap, but it re-arms a floored counter
        and must land with its first honest reading written in the commit.
        Dated by `next_free_due` (read at routing: 2026-10-01, every day
        through 09-29 already carrying 6-8 promises) rather than by the
        repair's size — a cheap fix on a piled day is still a promise that
        breaks. Nothing forbids an earlier slot taking it if the board is
        empty; the date is a ceiling on silence, not a floor on work.

## ROUTED: OPEN — `declared-venue-vs-delivered-venue-has-no-comparator`: no
## instrument compares the venue a run actually used against the venue its
## spec declares, and `spec_sha` answered "did the claim move?" WRONGLY across
## a total corpus replacement (builder, 2026-09-23, per the 110th audit RANK 1)

ROUTED: declared-venue-vs-delivered-venue-has-no-comparator | 2026-09-23 | 3395c8b (110th audit RANK 1, "the instrument gap under it") | OPEN
    DUE: 2026-10-01 | Dated by `next_free_due` (read at routing: 2026-10-01;
        every day through 09-30 already carries promises, and the tool's own
        line says 7 rows share 09-25 against a capacity of 6). Routing is the
        whole order — nothing is built, re-run or amended by this row.

**The finding, in the audit's own words:** *"nothing in this project compares
the venue a run actually used against the venue its spec declares. `run
verify` checks the gate replays and the control ran; `STEERING-METRIC-MISMATCH`
checks quoted numbers; `run stale` checks code shas. A corpus swap is invisible
to all three."* The audit ordered the INSTANCE repaired (the `HR.1` registry
amendment — executed, `955b9ef`, `spec_sha` MOVED `769b55d0` -> `ea53ae2e`)
and named the CLASS without routing it. OVERSIGHT.md is rewritten every audit,
so without this row the class finding vanishes at the 111th.

**The live instance that proves the class, now closed but exemplary:** `HR.1`
attempts 1-4 carried `spec_sha 769b55d0` byte-identical while the experiment
under it moved LibriSpeech -> VCTK, cross-SESSION -> cross-MICROPHONE,
20/20 -> 20/40. Every delta was disclosed — docstring, journal, queue row,
commit message — and none of it in the one field an auditor greps. The
project's own integrity hook reported "the claim text did not move" across a
total venue replacement, for seventeen days, until a human read both texts
side by side.

**What is asked of the Review (a DESIGN question, not an implement order):**
decide whether this class gets a comparator, and if so what shape. Candidate
shapes, named so the disposition has something to accept or refuse, not to
pre-empt the pick: (i) a structured `VENUE:` (or `FIXTURE:`) field in the Spec
that `_experiment` must echo into the recorded row, compared mechanically by
`run verify` — mismatch prints, reporting-only first (the SO.10 vacancy
precedent); (ii) fold venue text into what `spec_sha` hashes so a venue edit
is at least FORCED to move the sha (weaker: detects nothing when the registry
is simply not amended, which was the actual failure); (iii) declare the
amendment discipline sufficient and DECLINE — the 110th's repair path worked,
at the cost of needing a human to notice. Whichever way: `run.py` sits in
`T0.36`'s `IMPL_DEPS`, so any wiring into `run verify`/`run status` bills a
`T0.36` re-buy (~35 s foreground, priced from attempt 20); `WAITS-ON`
declaration work (`waits-on-declared-field`, DUE 09-25) is adjacent surface
and the two should probably be ruled in the same sitting rather than grown
separately.

## ROUTED 2026-09-24 (builder, first slot under the freeze): `d35-none-quota-has-no-satisfying-move` — the freeze's per-slot creature-gate quota intersects the live prohibition set at the EMPTY SET, and the third consecutive "None" lands TODAY at ~11:07, before any desk sits

ROUTED: d35-none-quota-has-no-satisfying-move | 2026-09-24 | f25f9f6 (D35, the Tier-0 freeze; renumbered d8722fb) | OPEN
    DUE: 2026-09-25 | Deliberately dated onto a day already carrying 7 rows
        (the tool's next free date is 10-01): the quota trips on 09-24 and
        every later date only accrues violations. The 09-25 06:37 sitting is
        the first desk that can act. Routing is the whole order — nothing is
        built, re-run or amended by this row.

**The measurement, derived fresh this slot (09:07), not inherited.** D35 rule 3
requires every builder iteration to name which of `T2.01`, `XL.01`, `T6.01` it
moved, with "none" legal at most twice running. As of 09:07 on the day the
freeze landed (08:34), NO builder-legal move exists on any of the three:

- **`T2.01`** — settled FAIL (2.67σ vs 5, unmoved); its dep `T1.08` is itself
  FAIL. Both repair lanes are desk-owned and prohibited to the builder by name:
  the D1.0 adopted-gate rerun (`d10-successor-rerun-under-adopted-gate`, 3''),
  and the body/world redesign behind `D33`/`w1-world-edit-window`
  (`W1.01`/`W1.03` registration NOT permitted, 2^10). `T1.08`'s pipeline repair
  is likewise the Review's (2^10).
- **`XL.01`** — settled FAIL a2, superseded by its strengthened successor
  `NE.08` (registry provenance, SURVIVAL_WORLD §5.0). `NE.08` is blocked behind
  `T6.03` (BLOCKED a2; "do not re-run T6.03 until T2.10 is PASS", 2^7 carried)
  and `T2.10`'s repair is the Review's (2^10). Re-running settled `XL.01`
  unchanged is a forbidden seed-lottery redraw.
- **`T6.01`** — never run, NO implementation, deps `T4.05` <- `T4.04` <-
  `T2.01`: both intermediates never run and unimplemented, so no verdict is
  recordable whatever the builder implements. Loosening the dep edge is a
  forbidden control-loosening (law 4); registering a cheaper `T6.01` is
  forbidden by D35's own text.

**The arithmetic.** Freeze committed 08:34. Builder slots: 09:07 = None #1
(this slot's journal), 10:07 = None #2 unless that slot finds a move this
derivation missed, 11:07 = None #3 = the quota broken by construction. The
overseer sits 12:37 (after), the Review 09-25 06:37 (after). No desk CAN act
before the tripwire fires. **This row does not claim the tripwire is a defect**
— a conduct rule that makes the allocation deadlock scream every hour may be
doing exactly its job; what the scream needs is a named reader, which this row
is.

**The collision one day out, named so it is not a surprise:** 09-25's first
legal builder pick (the `WAITS-ON:` implementation + `T0.31` re-buy, the 09-21
disposition) is governance-instrument work and moves no creature gate — so
09-25's slots also answer None unless this row is disposed first or that
disposition is re-ranked against the freeze it predates.

**The one candidate builder move, NAMED, NOT TAKEN, and why:** implementing
`T6.01`'s episode harness ahead of dep-clearance. Not taken this slot because
(a) the runner refuses it on `T4.05` regardless, so it buys no verdict and
cannot stop the tripwire; (b) "one life, start to finish" is exactly the
surface the pending W1 world edit may redefine, and fixing its meaning
unilaterally one hour after the freeze landed risks manufacturing the cheaper
T6.01 the freeze forbids. If the disposing desk rules that harness
implementation counts as "moved T6.01" and that W0-as-it-stands is the venue,
it becomes legal same-day builder work and should be said in one line.

**What is asked (a disposition, not an implement order) — any ONE of:**
(i) the Review executes one of its own owed repairs that opens a gate lane
(the d10 gate design, `T2.10`'s repair, `T1.08`'s pipeline repair, or the
`W1.01`/`W1.03`/`W1.04` registration its own 09-23 addendum limb already
priced); (ii) the desk that authored D35 clarifies "moved" — whether
dep-clearing or implementation work counts, and whether a slot whose only
blockers are desk-owned rows suspends the quota rather than burning it;
(iii) the owner strikes or amends D35 in one line, as its own entry invites.
Until one lands, every builder slot from 11:07 records a real violation that
no baseline may absorb.

## ROUTED 2026-09-24 (builder, 113th-audit item 3): `decisions-settles-on-headers-alone` — a ruling filed in prose under a neighbouring heading is invisible to the settlement parser, and the instrument then orders the answer reversed

ROUTED: decisions-settles-on-headers-alone | 2026-09-24 | a1dbf54 (113th audit) | OPEN
    DUE: 2026-10-01 | Dated onto the tool's next free date — 09-25 already
        carries 7 rows plus the WAITS-ON unlock and the d35 disposition. This
        row is a DESIGN awaiting a desk's ruling, not a build order: D35 rule 2
        forbids new audit instruments, and whether a truthfulness repair to an
        EXISTING checker is exempt (the 3'' carve-out: "a change that makes the
        meter ... tell the truth more plainly is always allowed" was written
        for the CPU accountant, not for decisions.py) is the desk's call, not
        the builder's.

**The defect, from the 113th audit.** `decisions.py:333-345` settles a decision
by scanning its HEADERS for `RESOLVED|off your desk|BY THE CALENDAR`. D19's
owner ruling of 2026-09-17 was appended under `## D35` with no header of its
own, so for ten days `--check` printed `D19 ... OVERDUE — DEFAULT IS DUE TO
FIRE` — an ORDER whose execution would have reversed an owner ruling. The
filing is repaired (this commit); the parser's blindness to prose-filed
answers is not.

**The narrow design (audit's own words, unmodified):** also scan a decision's
own `DECIDE:` block region for an explicit `SUPERSEDED`/`RESOLVED` marker —
D19's superseded block carries the sentence "it is superseded, unfired" today
and the parser cannot see it. Constraints that bind any implementation:
`_SETTLED` must keep naming the DECISION's fate, never an entry's freshness
(the `STALE` false-exoneration scar at `decisions.py:337-344`); the marker scan
must not let a desk settle its own entry by adjective (the settlement text must
still be an ANSWER, not a status); and a fixture proving the pre-fix blindness
(D19's exact shape: ruling under a neighbour's heading) goes in with the fix,
per T0.28's existing property style.

**The charter gap the audit named, carried with the row:** OVERDUE admits two
moves — fire, or re-arm with a reason — and neither is right when a decision
was ANSWERED and mis-filed. The third path, *record that it was answered*, is
in no instrument's vocabulary; ten consecutive audits declined to fire without
writing down why. Whether that third verb enters the overseer charter is part
of this row's ruling.

ROUTED: gen-four-reparented-to-a-decision-that-had-already-closed | 2026-09-24 | 1b828b8 (114th audit Finding 1) | OPEN
    DUE: 2026-10-01 | Dated onto `review-queue`'s own mechanical answer for a
        full calendar ("Next date with room under the measured capacity:
        2026-10-01") per the 114th audit's routing order. The ruling owed is
        OWNERSHIP, not repair: who owns the four GEN citations now that the
        decision they were re-parented to closed without inheriting them.

**The defect, from the 114th audit — every piece re-verified by the routing
slot (builder, 2026-09-24 19:0x) before this row was written.** The ACTED
answer on `goal-cites-four-specs-that-resolve-to-corpses` (`34116ca`,
2026-09-16) split the seven corpse-citations and re-parented its Group B —
`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all `welded<-LC.07` — to "the owner's
open `D24`". But `D24` had closed FOUR DAYS EARLIER: `RESOLVED BY ARMED
DEFAULT (fired 2026-09-12 ~17:3x)`, option (iii) DECLARE-DO-NOT-DECIDE, at
`docs/DECISIONS_RESOLVED.md:939` — and its full entry mentions no GEN spec
(grep over the whole `## D24` section: 0 hits). A terminal row is never
re-read (100th audit B2), so the re-parent left four of `GOAL.md`'s own
citations owned by NOBODY. The reds are live and mechanical: `coverage` prints
`4 NEW unrunnable citation(s) — GEN.02, GEN.03, GEN.06, GEN.09` against the
shrink-only `GOAL_UNRUNNABLE_BASELINE = {DP.02, DP.03, LC.04}`
(`experiments/coverage.py:276`), and `run review-queue` prints this exact pair
under its `DISPOSITION-ON-A-CLOSED-DECISION` reading.

**Staleness bill: ZERO ledger rows.** Ruling ownership invalidates no
certificate — no spec's code, gate or venue moves. The bill arrives only with
whatever repair the owner is then asked for (an `LC.07` successor design, a
`GOAL.md` text edit, or a new decision entry inheriting the four), and each of
those prices itself.

**What this row is NOT (the routing order's own constraints, carried so the
next reader does not "fix" it):** do not touch `GOAL.md`; do not touch
`GOAL_UNRUNNABLE_BASELINE` in either direction (shrink-only by construction);
do not re-open the ACTED row `goal-cites-four-specs-that-resolve-to-corpses`;
do not register a GEN spec (D35 freeze aside, an id resolving to a corpse is
worse than one resolving to nothing — 59th audit). The routing is the whole
unit.

ROUTED: waits-on-has-no-producer-outside-a-closing-row | 2026-09-25 | 115th audit Finding 4 (builder slot 01:1x) | OPEN
    DUE: 2026-10-01 | Dated onto `review-queue`'s mechanical next_free_due per
        the 115th audit's routing order. What is owed is one sentence in the
        Review's OWN sitting order; the ruling is whether and where that desk
        carries it.
    WAITS-ON: none | this row asks a desk to adopt a sentence; no other open
        row's answer changes what that sentence says.

**The defect (115th audit Finding 4).** The only written instruction telling a
ROUTER to declare coupling lives in the body of `waits-on-declared-field`,
which is `DUE: 2026-09-25` and is about to be stamped `ACTED` — and a terminal
row is never re-read (100th audit B2). The moment that stamp lands, the
obligation's sole producer-side home is a closed row: the READER
(`experiments/review_queue.py`, T0.31 p21/p22) enforces completeness per date
by WITHHOLDING the grouped count, but nothing any router actually reads says
to write the field in the first place. An obligation whose only statement is
in a terminal row is a rule that exists for exactly as long as nobody needs it.

**Half the repair is done in the routing commit and binds the builder:**
`scripts/ladder_prompt.md`'s loop instructions now carry the line — a new
`REVIEW_QUEUE.md` row declares `WAITS-ON: <row id> | why` or `WAITS-ON: none |
why not`, declaration-only, never silence. **The ask of this row: the Review
carries the same sentence into its own sitting order**, because the Review
routes rows too (most of the current OVERDUE pile is its own routing) and the
builder's steering page binds no other desk.

**Constraint carried from the audit, so the next reader does not "improve"
this:** do NOT add a violation class for an undeclared coupling — the 09-19
disposition refused that deliberately (`WAITS-ON:` buys nothing, exempts
nothing); the reader's WITHHELD line is the entire enforcement, and it is the
Review's grammar, not the builder's.

**Staleness bill: ZERO ledger rows.** A sentence in a sitting order and a
bullet on a steering page invalidate no certificate; T0.31's re-buy already
happened with the implementation (attempt 22, PASS, `05a582d`/`12a8180`).
