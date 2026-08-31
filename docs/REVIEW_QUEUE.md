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
deleted (T1.02 precedent: history stays). The bill is the price of the fix,
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

A live row past its `DUE:` is **OVERDUE**; an `OPEN` row with no `DUE:` older
than one full consumer cycle (8 days) is **STALE**. `HELD` buys exemption from
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

ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | OPEN — design owed by the Review 2026-08-30 (Sunday FULL); one cheap falsifier sequenced first, 2026-08-25
    DUE: 2026-08-30 | the W0/W1 design, owed by the Review's Sunday FULL run.
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

---

## ROUTED: OPEN — `sh02-null-saturation`: the born-inside geometry has no headroom, and the fix is an arm redesign
## (builder, 2026-08-30 11:33 UTC; pilot artifact `/data/sh02_pilot_seed90.json`, spec commit `8abfa70`)

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
