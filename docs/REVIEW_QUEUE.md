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
    Question: Water.apply (playground.py:627) writes a body's xfrc row only
    while it is in the pool, so any body that exits keeps its last buoyancy/
    drag force forever — a phantom force in live dynamics, found by DP.05's
    fidelity pilot (snapshot/restore made it visible). Fix the world?
    Full record: LESSONS d1bc3d1; DP.05 PILOT RECORD.
    Staleness bill: SEMANTIC (worlds with a pool, per the 27th audit) —
    BA.01, LC.02, PS.02, PS.03, XL.00 — 5 PASS certificates; MECHANICAL —
    all 21 playground.py rows above.

ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | DISPOSITIONED 2026-09-06 (Review FULL — the W1 SPEC-FAMILY DESIGN is published below, five specs W1.00–W1.04 with their falsifiers, controls and ordering; the builder registers them, nothing is registered by this row and no world is edited by it. GOAL.md untouched, no spec re-parented — those stay the owner's, per D21)
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
    DUE: 2026-09-13 | DELIVERED AND SUPERSEDED. The design owed on 2026-09-06
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

ROUTED: w100-honest-null-does-not-rescue-pile-a | 2026-09-06 | 79th-audit-item-1 (builder; finding §3.1) | OPEN
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

ROUTED: me1-similarity-floor-never-abstains | 2026-09-06 | Review FULL 09-06 Part 2 (ME.1 strengthened, FAIL, clean-tree re-buy) | DISPOSITIONED
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
    DUE: 2026-09-14 | RE-DATED (Review DAILY 2026-09-11), and the reason is the
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

ROUTED: w1-world-edit-window | 2026-09-06 | Review FULL 09-06 (w0-too-shallow disposition) | OPEN
    DUE: 2026-09-13 | the single world-edit sitting that `W1.03` opens, which
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
    DUE: 2026-09-13 | **the ANATOMY AUDIT item, and it is deliberately ON the
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

    DUE: 2026-09-15 | RULED 2026-09-12 (Review, DAILY) — the owed unit CHANGES
        HANDS AND KIND: what is owed on 09-15 is the builder's F2 diagnostic
        probe, not this desk's F1 arm pick. Dated 09-15 (5 live rows, measured
        capacity 6) and not 09-13 (14 rows). Design in the RULING below.

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

ROUTED: sh02-null-saturation | 2026-08-30 | 8abfa70 (pilot /data/sh02_pilot_seed90.json) | OPEN
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
    DUE: 2026-09-13 | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
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
    DUE: 2026-09-13 | the Review rules whether existing visual certificates
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

ROUTED: ba03-null-saturates-the-horizon | 2026-08-31 | 9e7cc86 (BA.03 attempt 1, 3.99 CPU-h, ledger row VOID) | OPEN
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
    DUE: 2026-09-13 | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
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

ROUTED: t306-matched-magnitude-noise-buys-coverage | 2026-08-31 | 1653104 (T3.06 attempt 1, ledger row VOID, 2434 s) | OPEN
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
    DUE: 2026-09-13 | RE-DATED to the Sunday FULL and BUNDLED (Review DAILY
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

ROUTED: reparenting-the-welded-fifteen | 2026-08-31 | aabced4 (B3 blast radii) + 78aad78 (ARENA-UNREACHABLE) | OPEN
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

ROUTED: lt01-c2-body-cannot-rise | 2026-09-01 | a0e6011 (LT.01 attempt 1, FAIL, 3 seeds x 3000 decisions) | DISPOSITIONED 2026-09-06 (Review FULL — option (a), the re-scope specified in the C2' block below; design only, the builder implements and re-runs, and the 0.6 m bar does not move in either branch)
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
    DUE: 2026-09-13 | IMPLEMENTATION of `C2'` below plus LT.01 attempt 2,
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

ROUTED: ub10-seed-fragility-and-saturated-battery | 2026-09-01 | UB.10-attempt-1-VOID | DISPOSITIONED 2026-09-08 (Review DAILY — the anchor's saturation is the load-bearing defect and is repaired by HARDENING THE TASK, never by shortening training; the per-arm stability conjunct is adopted; the seed-level SCORED-AND-INELIGIBLE retirement is REFUSED as a weakening. Design below)
    DUE: 2026-09-06 | an arm/task redesign decision owed by the Review's
    Sunday FULL run; bundle beside `recipe-sensitivity`'s lineage (this row
    is what its 08-25 disposition, fully executed, measured next)
    DUE: 2026-09-08 | the same redesign decision, moved to the Tuesday DAILY —
    RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the unison
    bakeoff has its own lineage (recipe-sensitivity → the executed 08-25
    disposition → this measurement) and no dependence on the W0/W1 design;
    the bundle it names is a reading order, not a sitting.
    DUE: 2026-09-15 | DECISION DELIVERED 2026-09-08 (Review DAILY): harden the
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

ROUTED: lc07-checkpoint-branch | 2026-09-01 | LC.07-pilot-branch-B | DISPOSITIONED 2026-09-06 (Review FULL — checkpointing is REFUSED because it repairs the wrong constraint; the arena is declared VENUE-UNAFFORDABLE with the arithmetic below, and the affordability question goes to the owner, not to the builder. Design below)
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
    DUE: 2026-09-14 | DECISION DELIVERED 2026-09-08 (Review DAILY):
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

**What this costs and what it does not.** The probe is CPU-cheap forward passes
and buys the design its missing premise. Attempt 3 remains ~17 GPU-h and is NOT
authorised by this disposition — it is authorised by the probe landing and the
gate being committed, and it should be dispatched into W37 (opens 09-13) rather
than scraped out of W36's remaining ~12.4 h. This desk has now spent 33.8
GPU-hours on two VOIDs and will not buy a third verdict from a gate that has
not first been shown to be able to return one.
ROUTED: lg10-mouth-fidelity-vs-freedom | 2026-09-02 | LG.10-attempt-2-FAIL | DISPOSITIONED 2026-09-08 (Review DAILY — (c): LG.10's bar and its FAIL STAND. (b) refused on the spec's own docstring warning. (a) is accepted as a NEW registered claim with a binding utterance-rate floor, and explicitly NOT as a rewrite of a failing spec. Design below)
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

ROUTED: goal-cites-four-specs-that-resolve-to-corpses | 2026-09-02 | Review-08-31-item-6-backfired | OPEN
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

ROUTED: t309-control-clears-the-claims-own-margin | 2026-09-02 | 06f6a01 | DISPOSITIONED 2026-09-08 (Review DAILY — (a): the kills clause is NOT executed, because a run whose control cleared the claim's own margin is a VOID and a VOID cannot kill anything. (b) refused. (c) survives as the constructive path and is downstream of the venue design. Design below)
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
    DUE: 2026-09-13 | DESIGN DELIVERED 2026-09-06 (fork (c), below); what is
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

ROUTED: told-world-has-no-rung | 2026-09-03 | 66th-audit-B1 (e7546e4) | DISPOSITIONED 2026-09-10 (Review DAILY — sub-question (b) ANSWERED YES on evidence that did not exist when this row was written, and the row's own declared bill CORRECTED from zero to real. Sub-question (a) re-dated to 09-15 with its premise materially weakened. See ANSWER (b) below)
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

ROUTED: lg03-blind-twin-cannot-prove-itself-alive | 2026-09-04 | LG.03-attempt-1 | DISPOSITIONED 2026-09-12 (Review DAILY — RULED: none of the four options as written. The gate's own maximum achievable value is `planner_own`, not 1.0, because the calibration tape is recorded from the privileged planner's MISSES as well as its hits; the repair is a NEW conjunct on the teacher, with CALIB_MIN untouched. See RULING below)
    DUE: 2026-09-12 | a liveness-gate redesign owed by the Review. Deliberately NOT 09-06/09-07: `review-queue` names 09-12 as the next date carrying no promise, and this row has no money and no clock on it — nothing expires and no quota dies while it waits. Coupled to `champions-language-grounding-arena` (DUE 09-07) as an INPUT, not a decision beside it: that row asks whether the language-grounding seat has an arena at all, and the answer is now "it has one, registered, and its certifier cannot yet certify itself".
    DUE: 2026-09-14 | DISPOSITIONED 2026-09-12 (Review DAILY): the design is
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
    DUE: 2026-09-13 | a reachable repair path for the death-and-retry commitment — the question is "what buys it one", NOT "re-run XL.01". Date is `next_free_due` per B4, not Sunday.

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
    DUE: 2026-09-13 | a disposition for the fast/slow world-model fixture: what does the DP family require of a world model that beats every null but loses to a linear probe? Date is `next_free_due` per B4.

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

ROUTED: t402-touch-drowns-audio-at-the-fusion-boundary | 2026-09-05 | 72nd-audit-B4 (FAIL-UNOWNED, 6fbac74) | OPEN
    DUE: 2026-09-13 | the fusion-balancing redesign the priority head has pointed at "the Review, not an argument" since 08-21 — now with a row and a clock instead of a standing sentence. Date is `next_free_due` per B4.

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

ROUTED: t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall | 2026-09-05 | 72nd-audit-B4 + builder (the FAIL-UNOWNED detector's 4th member — the audit's own count missed it) | OPEN
    DUE: 2026-09-13 | a disposition for T2.15's FAIL: route the memorisation-route finding somewhere an instrument can see it, or dispose it explicitly. Date is `next_free_due` per B4.

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
    DUE: 2026-09-14 | a disposition for SO.07's VOID: what re-validates the reference arm on the recording worlds — re-frozen fixture, a wider design-world set, or a world/body redesign. Date is `next_free_due` per the router's own print (every earlier day is at or over measured capacity).

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

ROUTED: pl02-eye-gate-reads-the-encoder-not-the-eye | 2026-09-07 | builder (pl02_rig_probe.py; smoke 603619c; decomposition eaec320) | DISPOSITIONED
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
    DUE: 2026-09-14 | RULED (Review DAILY 2026-09-11), and the implementation
    is the BUILDER's — the spec edit below, then a smoke, then the registered
    run under the ordinary blocking rules. 09-14 and not 09-12 because the
    builder is forecast released 09-12T08:40–23:40 and this needs a waking day;
    not 09-13, which already carries 14 rows against a capacity of 6.

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

**Three instances, three specs, two days, all found by reading source rather
than by any tool:**

| spec | declared meaning of the gate | quantity actually computed |
|---|---|---|
| `PL.02` (ruled 09-11) | "the eye is alive" | `r2_ua`, which is the **subtrahend of the claim's own effect size** `R_pl = r2_pl − r2_ua` — so the guard capped the gain it was added to protect at ≤0.20 against an observed 0.94 |
| `LG.03` (ruled 09-12) | "the twin reproduces demonstrations it was trained on" | `fidelity × teacher competence` — the tape is recorded from the planner's misses too, so perfect reproduction scores `planner_own`, not 1.0 |
| `HR.5` (ruled 09-12) | "four sounds are separable" | `four_class_audio_separability` 0.583, a number its own `position_only_acc` control beats at 0.708 — separability of POSITION, not of sound |

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
