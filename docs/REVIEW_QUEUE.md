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
