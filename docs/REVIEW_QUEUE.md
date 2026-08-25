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
<why>`); rows are never deleted (T1.02 precedent: history stays). The bill is
the price of the fix, computed BEFORE deciding, so "fix the world" decisions
are made with the re-certification cost on the table, not discovered after.

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

ROUTED: recipe-sensitivity | 2026-08-20 | probe jack-ladder-1787249890 | ACTED 2026-08-25 (design in docs/PROGRESS.md § FOR THE BUILDER item 2)
    Question: no single uniform training recipe trains all six matched-param
    UB.10 arms (warmup@1e-3 leaves A2/A3 dead; 3e-4 fixes A3 but breaks A4);
    A2 learned its marginals under NO tested recipe. Per-arm recipes, arm
    redesign, or drop the uniform-recipe constraint? UB.10 is PARKED on this.
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

ROUTED: ne01-occlusion-knife-edge | 2026-08-24 | 5063144 | HELD 2026-08-25 for the world-edit window (see THE BUNDLING RULE)
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
    Question: Water.apply (playground.py:627) writes a body's xfrc row only
    while it is in the pool, so any body that exits keeps its last buoyancy/
    drag force forever — a phantom force in live dynamics, found by DP.05's
    fidelity pilot (snapshot/restore made it visible). Fix the world?
    Full record: LESSONS d1bc3d1; DP.05 PILOT RECORD.
    Staleness bill: SEMANTIC (worlds with a pool, per the 27th audit) —
    BA.01, LC.02, PS.02, PS.03, XL.00 — 5 PASS certificates; MECHANICAL —
    all 21 playground.py rows above.

ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | OPEN — design owed by the Review 2026-08-30 (Sunday FULL); one cheap falsifier sequenced first, 2026-08-25
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

ROUTED: t215-router-under-lexical-null | 2026-08-25 | 20b8660 (row ran_at 2026-08-25T04:40) | OPEN
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
