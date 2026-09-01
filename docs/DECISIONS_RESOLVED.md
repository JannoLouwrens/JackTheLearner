# Decisions resolved by bakeoff

Written by experiments/bakeoff.py. Losing arms are recorded on purpose: a decision whose alternatives were discarded cannot be re-opened when the evidence changes, and the alternatives get silently reinvented later.

> **2026-08-09 — nine `TEST` entries removed.** They were unit-test
> fixtures, not decisions: `_append_decision` took no path argument, so
> `bakeoff.py`'s own self-tests wrote into the real record. The record has
> since been made injectable (`run_bakeoff(decisions_path=...)`) so a test
> cannot reach this file again. Until a real bakeoff runs, this file is
> EMPTY — and that emptiness is the honest reading: SYSTEM.md's third law
> has never yet been exercised on a real question.

## PS.01/J — VOID
arms below the 3.0-sigma learning gate: integral6, peak6, peak_force. An arm that has not demonstrably learned cannot arbitrate the decision.

metric: `fall_vs_ground_auc`  ·  null 0.497 ± 0.012

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| peak_dvel | 0.827 | 5.99 | pass | 2.0 |
| control:noise | 0.570 | 1.47 | FAIL | 0.0 |
| integral6 | 0.520 | 0.44 | FAIL | 1.0 |
| control:constant | 0.500 | 0.28 | FAIL | 0.0 |
| peak6 | 0.340 | -1.96 | FAIL | 1.0 |
| peak_force | 0.337 | -2.62 | FAIL | 1.0 |

## PS.01/J2 — WINNER — impact_speed
impact_speed beats peak_dvel by 2.66 sigma and clears the null by 10.32 sigma. Eliminated by the gate (not competing): integral6, peak6, peak_force, evt_int6, evt6, evt_force, evt_dvel, evt_bodyf, evt_body6, evt_bodyint, mean_dvel.

metric: `fall_vs_ground_auc`  ·  null 0.497 ± 0.012  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are observables, not learners: each is a deterministic reduction of the SAME cached rollouts (`_scores` is memoised per seed, so every arm and every control reads identical physics). There is no training that could have failed, so a low score cannot be a broken run — it is the arm's own property, which is precisely the finding this bakeoff exists to produce. The T2.02 ambiguity the validity gate protects against (broken run or worse architecture?) does not exist here.

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| impact_speed | 0.973 | 10.32 | pass | 3.0 |
| evt_body6 | 0.840 | 2.55 | FAIL | 4.0 |
| evt_dvel | 0.837 | 2.43 | FAIL | 4.0 |
| evt_bodyf | 0.837 | 2.45 | FAIL | 4.0 |
| peak_dvel | 0.827 | 5.99 | pass | 2.0 |
| evt_bodyint | 0.767 | 1.44 | FAIL | 4.0 |
| mean_dvel | 0.573 | 0.54 | FAIL | 2.0 |
| control:noise | 0.570 | 1.47 | FAIL | 0.0 |
| integral6 | 0.520 | 0.44 | FAIL | 1.0 |
| control:constant | 0.500 | 0.28 | FAIL | 0.0 |
| evt6 | 0.422 | -0.66 | FAIL | 3.0 |
| evt_force | 0.422 | -0.66 | FAIL | 3.0 |
| evt_int6 | 0.415 | -0.74 | FAIL | 3.0 |
| peak6 | 0.340 | -1.96 | FAIL | 1.0 |
| peak_force | 0.337 | -2.62 | FAIL | 1.0 |

## D2 — WINNER — VOID BLOCKS its dependents (resolved by ledger replay, 2026-08-13)

**The question** (open on the owner's desk since 2026-08-09; overseer 11th audit
B3 ruled it the loop's to resolve: "a property question with a testable answer,
not a values question"): `Status.VOID`'s docstring said a VOID spec "does not
BLOCK its dependents" while `Ledger.unsatisfied` blocked on anything that is
not PASS. Which semantics is right?

**Method — not `run_bakeoff`, and why.** The arms are two readings of the
dependency graph, not learners; there is no seed noise, no null, and no
training that could have failed. The pre-stated metric is **retraction
exposure**: replaying the ledger's own recorded history (every entry carries
`history` + `amended` with timestamps, so `status_at(spec, t)` is exactly
reconstructible), how many dependents would each semantics have admitted onto a
foundation whose next honest measurement then refuted or withdrew it?

**M1 — the natural experiment the ledger already ran.** At 2026-08-10T01:00,
T2.01 and T2.02 were both VOID (T2.01 hand-amended FAIL→VOID on 08-09 after
T0.14 found dropout live in its eval). Seventeen minutes later T2.01's clean
re-run recorded **FAIL**.

| semantics | specs admitted at 01:00 | resting on T2.01 (FAILED at 01:17) | resting on T2.02 (still VOID) |
|---|---|---|---|
| BLOCK (shipped code) | 0 | 0 | 0 |
| NO-BLOCK (docstring) | **11** | **9** (T2.16–18, T3.02/04/05, T4.04, T5.01, T5.07) | 2 (T2.13, T5.09) |

Every result those 9 recorded in that window would have rested on a refuted
foundation — unearned green or misattributed red, the repo's original disease.

**M2 — the whole benefit of NO-BLOCK, measured today.** Exactly 3 specs
(T2.13, T5.09, UB.15), all resting on T2.02's VOID — a run that *refused to
arbitrate* — and **none of the three is implemented**, so NO-BLOCK frees zero
immediately runnable specs.

**M3 — the property, now executable (T0.08 property 6).** VOID and NOT_RUN are
the same epistemic state: no verdict on the hypothesis. NOT_RUN blocks. Under
NO-BLOCK, recording a VOID — a run that by definition measured nothing —
*enlarges* the set of runnable specs: a broken rig mints runnability. T0.08
now asserts the invariant (`void_dep_blocks`, `void_why_not_a_refutation`),
recorded PASS 2026-08-13.

**Winner: BLOCK**, at exposure 0 vs 9, benefit 0. **Loser recorded:** NO-BLOCK
("blocking treats a failure to measure as a negative result"). What was right
in the loser survives in the message, not the graph: the asymmetry that
matters is `kills`, which VOID suppresses, and `unsatisfied` now says
"VOID — not demonstrated ... not a refutation" while FAIL stays plain, so a
reader can no longer mistake a withdrawn measurement for a lost hypothesis.
**Re-open trigger** (the quantity this rests on, per LESSONS): if a VOID spec's
implemented dependents ever outnumber its plausible re-run cost — i.e. M2's
benefit side stops reading zero while the VOID is long-lived because its rig
cannot be fixed — the trade should be re-measured, not assumed.

Evidence: `experiments/ledger.json` T2.01/T2.02 `history`; replay over today's
147+ spec registry with historical statuses (method + numbers reproduced in
T0.08's docstring); `protocol.py` Status.VOID docstring + `unsatisfied` fixed
in the same commit. D2's full price history stays in `DECISIONS_NEEDED.md`.

## D10 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): the screen IS the arbitration when it returns exactly one; wm-latent seated BY VERDICT, single-arm caveat on its face

**The question** (armed 2026-08-24, decide_by 2026-08-31, passed unanswered —
the default fired as written, per the executor line of `03f31cf`): LC.03 v2
concluded with exactly ONE 3σ learner (`wm-latent`) out of five arms, in a
clean rig, at a 4× envelope the fork pre-registered as final. Does the
learning-core arbitration premise ("arbitrate among screened learners")
survive a screen that returns one?

**The default that fired, verbatim in effect:** LC.04's premise is amended
from "arbitrate among screened learners" to "the screen IS the arbitration
when it returns exactly one" (recorded in LC.04's registry notes), and
`wm-latent` takes the learning-core seat as measured winner-by-default —
`CHAMPIONS.md` seat marked **BY VERDICT (single-arm)** with the caveat on its
face. The owner's scale-transfer guard binds BEFORE adoption: re-test at ~10×
on Kaggle, which is free. LC.03 stays CONCLUDED in the ledger with its VOID
and history intact — no v3, no envelope growth, no re-roll.

**What the firing commit contains** (one commit, per the 53rd audit's B3
executor line and the 54th audit's B1 condition):
- `CHAMPIONS.md` learning-core cell + seat declaration: `wm-latent` BY
  VERDICT, arena `LC.00–LC.07`.
- LC.04 registry `notes`: the premise amendment, with the caveat that
  LC.04–LC.06 run only if the premise is ever repaired.
- **`LC.07` registered in the SAME commit** — the ~10× Kaggle scale-transfer
  re-test, `depends_on` LC.00–LC.02/PS.01/XL.00 (all PASS), deliberately NOT
  routed through the foreclosed LC.03 — so the seat is seated and contestable
  in the same breath. The ARENA-UNREACHABLE finding on this seat (54th audit)
  is discharged by construction, not by prose.

**Losers recorded:** option (b) (redesign W0 first — alive independently as
the `w0-too-shallow` Review question, DUE 2026-09-06, and W0.DIAG's PASS is
design input to it; nothing here pre-empts it) and option (c) (redesign the
arms — routed to the Review with UB.10's arm-redesign question). Both remain
available ON TOP of this default; neither could fire as a default because
each spends design work the owner may sequence differently.

**Re-open triggers, pre-registered:** LC.07 FAIL (the seat reverts to
contested-VACANT); any repaired screen returning ≥2 learners (LC.04's
original premise revives and the BY VERDICT hold is re-arbitrated); the
unison gates failing under wm-latent (adoption VOID per SYSTEM.md, seat
unchanged but adoption barred).

Evidence: `experiments/ledger.json` LC.03 v2 row (2026-08-23 21:11, VOID
"fewer than two learners (1 cleared)"); per-arm t-stats in the row and
`experiments/artifacts/lc03_curves_seed{0,1,2}.json`; DECISIONS_NEEDED.md D10
entry (armed 2026-08-24) with the full measurement table.

## D12 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder, AFTER D10 per the executor line): TRANSCRIBE, DO NOT DILUTE — the owner's two guards moved from prose to the places where gates bind; closed SUPERSEDED-BY-D10 for its live question

**The question** (owner, 2026-08-09: "are you sure it isn't holding us back
that agents are making CPU tasks and not GPU?" / "PPO might be best... after
20 hours when we stop at 19?"): does the LC bakeoff's verdict survive scale,
and who enforces the two addenda — the data-starved rule and the convergence
check — that the owner wrote against exactly this failure?

**The default that fired:** both guards transcribed VERBATIM into the registry
`notes` of LC.04 and LC.05 (Addendum 1: positive-slope-at-cutoff => DATA-
STARVED, re-screen at ~10x on Kaggle, never eliminate on a rising curve;
Addendum 2: WINNER only if runner-up slope <= 0 or projected crossover beyond
3x the tested budget, else SPLIT-PENDING and extend both). Because D10 fired
first and LC.04 will not run as a two-finalist bakeoff, the convergence check
is ALSO recorded on the `CHAMPIONS.md` learning-core seat as a binding
pre-condition on any FUTURE arbitration that seats a core against a runner-up
— the guard is carried forward, not bypassed by LC.04's retirement. The
scale-transfer check is on the seat as a named pre-condition of ADOPTION and
is now a registered spec: `LC.07` (D10's firing commit).

**Nothing weakened, nothing retired:** no threshold moved, no experiment
deleted; rules the owner already wrote moved from a document that binds
nothing to spec notes and a seat declaration, which the champions/coverage
instruments read. The entry's live question (does the verdict survive scale)
is now `LC.07`'s hypothesis — SUPERSEDED-BY-D10.

Evidence: DECIDE block D12 (armed, decide_by 2026-08-31 passed unanswered);
the addenda verbatim at DECISIONS_NEEDED.md "D12 — Does the LC bakeoff's
verdict survive scale?"; registry_expansion.py LC.04/LC.05 notes; CHAMPIONS.md
learning-core cell.

## D1 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): the PLASTIC-ONLY decree stands verbatim; option A STRUCK as unconstitutional; the four permitted arms go to a registered bakeoff (D1.0)

**The question** (open since 2026-08-04, armed 2026-08-24 under SYSTEM.md rule
3 as amended, decide_by 2026-08-31 passed unanswered): does the 57M trunk stay
in the control path? Only the constitutional half was ever the owner's — does
PLASTIC-ONLY admit a frozen control trunk? — and the default answers it the
only way a default may: by leaving the constitution exactly as written.

**The default that fired:** the PLASTIC-ONLY decree (GOAL.md:76, 2026-08-09)
stands verbatim and unnarrowed. **Option A (freeze the trunk, learn a small
adapter — the 2026-08-04 review's recommendation and this entry's own
"RECOMMENDED" for twenty days) is STRUCK as unconstitutional**: it postdates
nothing and the decree postdates it. The remaining permitted arms go to a
bakeoff at matched experience, multi-seed, one pre-registered metric, learning
gate and margin — registered in this same commit as **`D1.0`**, the exact id
`CHAMPIONS.md` has cited as the Control-architecture seat's arena since
2026-08-10: A-prime (learned control head reading plastic-trunk features), B
(split value/policy trunks), C (end-to-end at more steps — UNTESTED, not
refuted), D (transformer out of the control path). Winner seated by the
recorded margin when it runs.

**Loser recorded:** option A, struck on constitutional grounds, not on
evidence — its empirical content survives inside A-prime (a dedicated control
head reading trunk features), which differs only in that the trunk stays
plastic. The cost note travels with arm D: a D win forecloses DP.02 (private
control representations — the "two brains wearing one wrapper" signature),
recorded with any verdict, not a thumb on the scale.

**What this unblocks:** T2.01/T2.02 stop waiting on an open decision — they
re-run UNDER the D1.0 winner as ordinary ladder work. The 08-13 builder
resolution (T2.01 measures WHETHER the trunk learns; D1 answers WHERE control
belongs) stands and is now discharged by a registered arena rather than a
deadlock. T2.21 remains unregistered by that same decision.

**Re-open trigger:** the owner may narrow the decree at any time (that was
always the branch that needed them); a narrowing reinstates option A as an
arm, and the bakeoff re-runs with five.

Evidence: the three matched-env-step runs in the D1 entry (T2.01 v4 261/4.06σ
plateaued; MLP probe 531/~6.5σ; T2.02 530/7.11σ vs 318/2.46σ VOID); T2.01 v5
2.67σ vs the unmoved 5σ bar; GOAL.md:76; SYSTEM.md rule 3 as amended
2026-08-24.

## D4 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): RATIFY AND CAP — option 1 recorded as TAKEN on 2026-08-13, CPU_DAYS frozen at the envelope already spent

**The question** (escalated 2026-08-09 at ~20-33 core-hours; the spend was
then made without the record ever closing): may the LC screening bakeoff run
on this box's CPU, spread across iterations?

**The default that fired:** option 1 ("run it here, spread across
iterations") is recorded as **TAKEN on 2026-08-13** — retroactively honest,
not retroactively authorised: the decision this entry ratifies was made by the
commit (`7112515`-era) that amended LC.03's budget CPU_LONG → CPU_DAYS, and
the record now says so. The re-costed figure is **~90 core-hours**, 4.5× the
~20-33 the owner was shown. The two runs it paid for, named: **LC.03 v1**
(registered 2026-08-13, ran ~15.8 h, VOID 2026-08-14 07:36, `8ec4be8` — "run
did not test the claim") and **LC.03 v2** (4× envelope, ran 08-21 04:22 →
08-23 21:11, ~190 core-h wall across arms/twins/nulls, VOID "fewer than two
learners (1 cleared)" — the entire evidentiary basis of D10). Both VOIDs are
honest and load-bearing; nothing unsafe happened (nice 19 throughout, load ≤
0.20 sampled, no tenant disturbed, no money, no GPU quota).

**The cap, now standing:** the `CPU_DAYS` tier stays, **frozen at LC.03 v2's
envelope — 400,000 decisions / 17,280 core-seconds per arm-seed.** Any spec
that would exceed it, and any further growth of LC.03's envelope, requires a
fresh escalation with its arithmetic attached BEFORE the run. **Losers
recorded:** option 2 (spend Kaggle quota on CPU arms — trades the one resource
the GPU ladder is scarce for) and option 3 (cut the envelope — buys hours by
weakening a gate, which law 4 forbids outright). Strictly narrowing: nothing
new authorised, no tier added, no certificate touched.

**The lesson this entry carries** (it is the inverse of D1's): D1 was a
decision that blocked work for twenty days; D4 was a decision the work walked
past — the escalated spend was made, grew 4.5× in the making, and the
question sat on the owner's desk looking untouched for nineteen days. An
escalation queue whose entries can be overtaken by action without a record
means nothing in either direction.

Evidence: LC.03 budget amendment comment in registry_expansion.py (CPU_LONG →
CPU_DAYS, 2026-08-13); ledger LC.03 history (VOID 08-14, VOID 08-23);
`5074440` (4× re-registration); the D4 forensic timeline in
DECISIONS_NEEDED.md.

## D8 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): BA.02 PARKED until a body with directional catch authority exists — re-parented behind LT.08, nothing weakened

**The question** (open since 2026-08-14, armed 2026-08-25 with D9): BA.02 ("he
catches himself") is unmeasurable in the rover body — four scratch probes put
the sensing-over-blind contrast ceiling at ~0.0–0.1 s against the spec's own
pre-registered 0.20 s floor, because no actuator's useful effect depends on
fall direction. Park it, give the rover catch authority, or re-scope?

**The default that fired — option 1, PARK:** BA.02 is re-parented in the
registry behind the playground-humanoid line (`depends_on` now includes
LT.08, "The humanoid climbs — same test, real body" — the registered spec on
which a capable body arrives). Claim text, gates and thresholds are UNCHANGED;
the 08-14 VOID and its history stay exactly as recorded; `notes` carries the
`PARKED: 2026-09-01` marker with the measured ceiling. BA.01 stands untouched
— the sense exists and is decoded; only "he ACTS on it" waits for a body that
can act. The commitment `balance` keeps a claim-kind path through BA.03 (the
48th audit's successor, registered 2026-08-30 precisely so this park costs
the ratchet nothing).

**Losers recorded:** option 2 (give the rover catch authority — a
world-contract change that re-runs PG.3/PS/BA certificates and re-opens the
"arms are slides" convenience PG.3 certified) and option 3 (re-scope to a
scenario where direction matters in this body — the probes found none on open
ground; BA.03 is that option done properly, as a NEW spec with new nulls).

**Re-open trigger:** LT.08 PASS un-parks BA.02 mechanically (its deps
satisfy); any new body with directional catch authority before then routes
through the world-contract change process, not through this park.

Evidence: the four probes in BA.02's DIAGNOSIS section (slides +0.09±0.07 s,
adhesion +0.005±0.09, ground drive toward-lean −0.685±0.16, blind headroom
+0.275±0.137); D8's forensic entry of 2026-08-14; BA.03's registration
comment ("registered BEFORE D8's default fires so that parking BA.02 costs
the ratchet nothing").

## D9 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): the rover-body question is PARKED until the playground-humanoid line; W0.BAL stays pre-registered with its numbers attached; nothing adopted

**The question** (raised 2026-08-21 when three independent measurements said
the rover body is the binding constraint; armed 2026-08-25 together with D8
because they are one question): adopt W0.BAL's arm B or C as the body, or
park the fork?

**The default that fired — option (a), PARK:** the W0.BAL bakeoff stays
pre-registered with its numbers attached (arm C upright 1.000 on all three
seeds vs the as-built body's 0.002–0.004; artifact
`experiments/artifacts/w0bal_bakeoff.json`) and runs for adoption the day a
ladder-branch spec becomes unblocked on it. Arms B and C are NOT adopted, so
PG.3's inherited geometry and the BA.01 / PS.02 / PS.03 certificates
downstream of the body are untouched. This is the only branch that adopts
nothing, re-runs nothing, and leaves every recorded certificate valid.

**Recorded beside it, because the Review of 08-31 put it on the owner's desk
and a firing default may not pre-empt an open owner fork:** PROGRESS 08-31
FOR THE OWNER §1 recommends REGISTERING W0.BAL as a spec id and creating a
body seat in CHAMPIONS.md even while this park stands — parking an *adoption*
and having no *chair* are different things. That recommendation is untouched
by this firing (it asks for a seat, not an adoption) and remains open. So
does the evidence attached to the `w0-too-shallow` queue row that some
fraction of "the world is too shallow" may be "the body cannot act in it"
(LT.01's C2 FAIL, 2026-08-31, is the first registered-spec number for that
reading).

**Re-open triggers:** the playground-humanoid line landing (the park's own
terminus); the owner speaking to the PROGRESS fork; or the 09-06 Review's
w0-too-shallow design naming the body as the binding repair — any of these
reopens adoption through the world-contract change process.

Evidence: `experiments/w0bal_probe.py` + artifact; D9's entry of 2026-08-21
(three independent measurements); DECIDE block armed 2026-08-25; W0.BAL table
attached to D9 (commit `e9cc914`, 24th-audit B4, NOTHING adopted).

## D7 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): MovementMoodCoupling ACCEPTED AS COSMETICS, ON THE RECORD

**The question** (T3.07 FAIL, 2026-08-13; ready-to-decide since the 23rd
audit, 2026-08-20; armed with the eleven): MovementMoodCoupling failed its
ablation — mood measurably changes nothing Jack does. Delete it, redesign it,
or accept it as cosmetics?

**The default that fired — option 3:** MovementMoodCoupling is KEPT,
unchanged, for companion UI (idle posture, style text). In exchange the
record is narrowed: **no spec may cite mood as a BEHAVIOURAL channel**,
GOAL.md's interoception claims must route through some other component, and
T3.07's FAIL stands as the registered finding rather than as an open
question. The narrowing lives in T3.07's registry notes (where spec authors
read) and on the CHAMPIONS.md Emotion (affect) cell. No model code written,
no module deleted, no threshold moved, GOAL.md untouched.

**Losers recorded:** option 1 (delete — the component is 1,149 lines of
working UI the companion app uses; deletion is the owner's call and buys
nothing the narrowing does not) and option 2 (redesign until mood moves
behaviour — manufacturing a capability to satisfy a component, backwards by
this project's own laws).

**Context that arrived after arming, recorded not acted on:** the Review of
08-31 (item 4, FOR THE BUILDER) proposes re-aiming T2.12 with a load-bearing
conjunct at the fusion boundary (PAD channel carries gradient in a live UB.11
ablation) — strictly harder, GOAL.md's own ablate-a-sense standard. That
redesign is compatible with this default (it strengthens the seat's OTHER
spec) and remains open builder work.

**Re-open trigger:** new evidence through the Review that some mood-conditioned
pathway moves behaviour — never a bare re-run of T3.07 (its rig was
adjudicated live: reference arm reached speed span 0.30+, both at-chance
controls held).

Evidence: T3.07 FAIL row (2026-08-13, all controls on their sides);
CHAMPIONS.md Emotion (affect) cell (added by the Review 2026-08-31); D7 entry
+ 23rd-audit ready-to-decide note.

## D3 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): the git-push practice is FENCED — recorded, and bounded, exactly as it already happens

**The question** (answered YES by the owner 2026-08-10 for the practice; the
armed remainder was the BOUND — 146 logged pushes were operating under no
stated limit at all): what exactly may the loop push?

**The fence, now standing, verbatim:** the loop may `git push` commits it
authored to `origin/main` on the existing remote, and NOTHING ELSE — no
force-push, no `--force-with-lease`, no push to any branch other than `main`,
no new remote, no tag push, no push of a tree it did not itself commit. This
is a NARROWING of an unbounded observed practice; it widens nothing, edits no
threshold, touches nothing the owner owns, and changes no observable loop
behaviour on the day it fires — the recorded bound IS the artifact, no code
changes.

**Loser recorded:** option 1 (unbounded standing authorisation) — the ratchet
may shrink and may never grow. **To reverse:** the owner states option 3 in
the D3 entry; the loop returns to escalating before each GPU submission, at
the known cost of the weekly Kaggle quota (~8.8 h lost W32, 22.1 h W33, 29.7 h
W34 under exactly that friction).

Evidence: owner's YES of 2026-08-10 (D3 original); `assert_ref_is_current` in
`experiments/gpu.py` (the mechanism that makes an unpushed HEAD invisible to
every GPU job); the 146-push log cited in the armed entry.

## D11 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): the credits posture is ACCEPTED AS-IS, on the record that the cadence meter now governs

**The question** (raised 2026-08-10 when Claude credits were the binding
resource and unmetered): change the loop's cadence, budget, or metering?

**The default that fired — option (a), ACCEPT AS-IS:** the machinery that has
shipped since the entry was raised is the answer. The pace gate
(`lib_usage.sh`, shipped 2026-08-24) reads `week:all models`, names itself as
the gate in every log line, and holds budget across the week; the fallback
chain plus lost-iteration inheritance keep a limited hour from costing a unit
of work. No cadence change, no new budget, nothing widened. If the owner
later wants option (b) or (c), the constants are one line in `lib_usage.sh`
and the schedule is one line in cron.

**Recorded beside it (the standing rule this entry's history bought):** the
meter is driven from off this box — 71–75% of its rise measured in hours with
zero on-box requests, twice, on independent windows. Read the tool, act on
`week:all models`, do not model the meter. Every attempt to price organ-hours
against it (three) was falsified inside a week.

Evidence: `scripts/lib_usage.sh` (pace_gate + 90% stop); the 42-hour join in
CLAUDE.md's meter section; D11's price-history corrections of 08-26/08-27.
