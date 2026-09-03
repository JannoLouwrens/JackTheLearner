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
blocker has been dispositioned is itself a violation — otherwise the bundling
rule below becomes a place rows go to die. Deleting a row, or dropping a `DUE:`
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
                  lg10-mouth-fidelity-vs-freedom
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
    BLOCKED-BY: w0-too-shallow | the world-edit window that row opens; if it
        resolves toward W1 this follows it there and the bill goes to zero.
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
    BLOCKED-BY: w0-too-shallow | the same world-edit window; paying the
        21-certificate mechanical bill once instead of three times.
    Question: Water.apply (playground.py:627) writes a body's xfrc row only
    while it is in the pool, so any body that exits keeps its last buoyancy/
    drag force forever — a phantom force in live dynamics, found by DP.05's
    fidelity pilot (snapshot/restore made it visible). Fix the world?
    Full record: LESSONS d1bc3d1; DP.05 PILOT RECORD.
    Staleness bill: SEMANTIC (worlds with a pool, per the 27th audit) —
    BA.01, LC.02, PS.02, PS.03, XL.00 — 5 PASS certificates; MECHANICAL —
    all 21 playground.py rows above.

ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | OPEN — design owed by the Review 2026-09-06 (RE-ARMED 2026-08-31, reason below); one cheap falsifier sequenced first, 2026-08-25
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
    Question: three independent instruments now measure W0 as too shallow to
    reward the capabilities the ladder certifies — LC.03's darkroom control
    (passivity prospers), LC.03 v2 (one learner in five), DP.05 (lookahead
    buys 13–21 s under the 20 s margin; deeper lookahead buys LESS; the best
    reactive policy is "starve at the resting ceiling"). The pre-registered
    routing: traps, delays, irreversibility — the DP.00 preconditions GOAL.md
    names — before any dual-process claim; BO.01 does not run. COUPLED to
    D10 branch (b) in DECISIONS_NEEDED.md (owner) — the Review designs, the
    owner sequences.
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

ROUTED: t215-router-under-lexical-null | 2026-08-25 | 20b8660 (row ran_at 2026-08-25T04:40) | OPEN
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

ROUTED: t211-diayn-metric-cannot-separate-mi-from-noise | 2026-08-29 | pilots /data/t2_11_pilot2_seed{7,90}.json | OPEN
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

ROUTED: sm03-heldout-split-saturated | 2026-08-30 | 13c0440 (pilot /data/sm03_pilot_seed90.json) | OPEN
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

**Status: OPEN. Gates provisional, `run()` still refuses, nothing dispatched.**

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

ROUTED: t027-preserved-failimpl-as-artifact | 2026-08-30 | 7ffd961 (preserve_impl_bytes mechanism) | OPEN
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

ROUTED: pl02-dependency-on-pl00-verdict-vs-table | 2026-08-30 | 4f8d99a (PL.02 registration; PL.00 attempt 1 FAIL) | OPEN
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

ROUTED: dp04-lifespan-has-no-resolution | 2026-08-30 | ed7d78c (sizing seed 94, /data/dp04_sizing_seed94.json) | OPEN
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

ROUTED: champions-language-grounding-arena | 2026-08-31 | 901f7fc (champions.py declaration syntax; 51st audit B2 order) | OPEN
    DUE: 2026-09-06 | the Review breaks the tie: name LG.00 as the ring (the
        audit's reading) or keep ARENA: NONE with an unwritten grounding
        bakeoff as inventory debt (this file's reading). A builder declining
        an overseer order is supposed to REACH the Review; until this line
        existed, it could not.
    DUE: 2026-09-07 | the same tie-break, moved to the Monday DAILY — RE-ARMED
        2026-09-02 from 2026-09-06 (61st audit B2, builder): a champions.py
        declaration decision with no coupling to the W0/W1 design bundle;
        small, self-contained, a daily can carry it.
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

ROUTED: lt01-c2-body-cannot-rise | 2026-09-01 | a0e6011 (LT.01 attempt 1, FAIL, 3 seeds x 3000 decisions) | OPEN
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

ROUTED: d10-learning-gate-uses-two-different-denominators | 2026-09-01 | 59th-audit-B4 | OPEN
    DUE: 2026-09-06 | gate-design decision owed by the Review; bundle with `w0-too-shallow`'s window if the venue is judged the common cause

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

## ROUTED 2026-09-01 (builder, 59th audit B4): `d10-learning-gate-sits-at-the-untrained-twin-level` — the control passed by 0.04σ

ROUTED: d10-learning-gate-sits-at-the-untrained-twin-level | 2026-09-01 | 59th-audit-B4 | OPEN
    DUE: 2026-09-06 | gate-design decision owed by the Review; same bundle judgment as the denominators row

**What was measured (same run).** The untrained twins read 2.96σ and 2.94σ
against the 3.0σ learning bar — the control cleared by 0.04σ. A gate whose
untrained twin sits at the bar's edge is measuring architectural bias plus
noise, not learning headroom. **Option to weigh:** score each arm against
its OWN untrained twin rather than against random, which also dissolves the
denominators question above for the twin comparison. Same scope note: gate
redesign for future runs only; the recorded VOID stands.

## ROUTED 2026-09-01 (builder, UB.10 attempt-1 harvest): `ub10-seed-fragility-and-saturated-battery` — the unparked design ran honestly and measured two defects in itself

ROUTED: ub10-seed-fragility-and-saturated-battery | 2026-09-01 | UB.10-attempt-1-VOID | OPEN
    DUE: 2026-09-06 | an arm/task redesign decision owed by the Review's
    Sunday FULL run; bundle beside `recipe-sensitivity`'s lineage (this row
    is what its 08-25 disposition, fully executed, measured next)
    DUE: 2026-09-08 | the same redesign decision, moved to the Tuesday DAILY —
    RE-ARMED 2026-09-02 from 2026-09-06 (61st audit B2, builder): the unison
    bakeoff has its own lineage (recipe-sensitivity → the executed 08-25
    disposition → this measurement) and no dependence on the W0/W1 design;
    the bundle it names is a reading order, not a sitting.

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

## ROUTED 2026-09-01 (builder, LC.07 pilot harvest): `lc07-checkpoint-branch` — the seat's own scale-transfer arena cannot physically run inside a Kaggle kernel

ROUTED: lc07-checkpoint-branch | 2026-09-01 | LC.07-pilot-branch-B | OPEN
    DUE: 2026-09-06 | a checkpoint-vs-venue decision owed by the Review's
    Sunday FULL run; bundle judgment beside `w0-too-shallow` and D10's
    lineage — this arena is the one D10's firing commit registered so the
    wm-latent seat would not be held with a dead arena

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

## ROUTED 2026-09-02 (builder, 60th audit B2): `d10-successor-rerun-under-adopted-gate` — the project's largest unblock returned an honest VOID and became nobody's work in the same motion

ROUTED: d10-successor-rerun-under-adopted-gate | 2026-09-02 | 60th-audit-B2 | OPEN
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

ROUTED: lg10-mouth-fidelity-vs-freedom | 2026-09-02 | LG.10-attempt-2-FAIL | OPEN
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

ROUTED: t309-control-clears-the-claims-own-margin | 2026-09-02 | 06f6a01 | OPEN
    DUE: 2026-09-08 | an instrument/venue disposition owed by the Review — dated
    off the 09-06 pile deliberately (61st audit B2/FINDING 3: eighteen rows
    already land on that one Sunday, and this row is readable standalone after
    the `w0-too-shallow` window is decided, since it is the same question one
    call-site down: does this venue reward perturbation as such?).

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

ROUTED: cross-organ-doc-race-voids-certificates | 2026-09-03 | 64th-audit-B3 | OPEN
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

ROUTED: hr5-fixture-refuted | 2026-09-03 | 65th-audit-B2 (HR.5 FAIL 05:25, classes_present 1.0/4) | OPEN
    DUE: 2026-09-06 | rides the w0-too-shallow design window (the bundling
    rule above): the repair edits `playground.py` and `ContactAudio.py`, the
    same world files, and this row belongs to the SAME W1 fork — it must not
    be designed twice. If w0-too-shallow resolves toward a new world (W1),
    this row follows it there and the W0 bill goes to zero.
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
