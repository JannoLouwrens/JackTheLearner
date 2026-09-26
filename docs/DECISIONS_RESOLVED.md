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

## D14 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder): option (b), the LOUD REFUSAL, is the standing answer — and the code was already live before the firing

**The question** (from the 08-25/08-26 blackout series: the builder's model
line capped while the chain silently considered a switch to the most
expensive model on the shared meter): what does the loop do when its own
model's weekly line is exhausted?

**The default that fired — option (b):** a pre-flight check in
`scripts/ladder_loop.sh` before `run_claude`, at a 95% floor on the loop
model's own weekly line, that refuses the slot LOUDLY (`ABORT: no attemptable
model at the ${MODEL_FLOOR}% weekly model floor...`), writes the
lost-iteration marker, exits 0 without consuming the slot. Strictly tighter
than the 90% stop; moves no threshold; deletes no control. **Verified live at
firing time:** `ladder_loop.sh:271` carries the refusal and cites this
decision by name; a refused slot is a NUMBER (marker + bookkeeping), not a
silence. The code artifact landed 2026-08-30 ~20:4x — this firing records
(b) as the standing answer; it changes no behaviour.

**Losers recorded:** (a) silent walk to the most expensive model — the kind
of thing this project registers guards against; (c) gate on max(all-models,
loop-model) — tighter still, available later if (b)'s floor ever proves
gameable; (d) bounded unpaced window — owner-only forever, a default may not
suspend a throttle. The owner's remaining hours still supersede (b) at any
time.

**To reverse:** revert the ladder_loop.sh commit; there is no state to unwind.

Evidence: `scripts/ladder_loop.sh:261-275`; `scripts/lib_credits.sh`
`model_limited()`; `lost_iterations.log` as the receipt channel; the
08-21 10:07/11:07 dead-slot scar that motivated the whole family.

## D13 — RESOLVED BY ARMED DEFAULT (fired 2026-09-01, builder — the overseer may not edit its own script): option (c), the CHANGE-GATED NO-OP, implemented in scripts/overseer.sh with all four conditions

**The question** (from the 30th–33rd audits: four consecutive DRIFTING
verdicts over a frozen tree, ~1 point of the shared meter per audit, zero
repairs possible because the only organ that can execute a FOR-THE-BUILDER
item was the one the pace gate had locked out): should the overseer keep
auditing no-change at full cadence?

**The default that fired — option (c):** `overseer.sh` now skips a slot ONLY
when all four conditions hold: (1) HEAD unchanged since the last COMPLETED
audit; (2) zero builder iteration starts in `ladder.log` since it; (3) no
OPEN decision's `decide_by` falls before the next slot; (4) fewer than 3
consecutive slots already skipped — the organ can never go dark past 24 h on
its own decision. It logs `overseer: no-op, HEAD <sha> unchanged and 0
builder iterations since <ts>` with the skip count.

**Implementation notes that are part of the record:** condition (3) reads
`experiments.decisions` (OPEN decisions only), NOT a grep of `decide_by:` —
resolved entries keep their past dates in the file forever, so a raw grep
would have tripped on history and silently disabled the no-op for good, the
inverse failure. The state file records the last COMPLETED audit and only a
completed audit resets it, so a dead audit forces the next slot to run in
full — the guard fails toward MORE oversight. The Review liveness watch runs
even on skipped slots. Five-case harness test at firing time: no-state,
iteration-since, 3-skips, stale-HEAD, and open-deadline all force a full
audit (the last verified live against D13's own OVERDUE row before this entry
closed it).

**Losers recorded:** (a) accept as-is — keeps re-measuring a frozen system at
~4 pts/day; (b) blanket cadence halving — cuts oversight hardest on the days
the system moves, when it is worth most; (d) unify with the Review — out of
scope, D11's territory (the 06:37 cron collision stays filed as builder item
B3, a scheduling bug not a decision).

**To reverse:** revert the overseer.sh commit — cadence returns to an
unconditional `37 */6` immediately; delete `/data/jack-logs/overseer_noop.state`;
there is no other state to unwind.

Evidence: the 30th–33rd audit table in the D13 entry; `scripts/overseer.sh`
(noop_eligible + the completed-audit state stamp); the measured ~1 pt/audit
spend attribution the entry carries.

## `run blocked` / `repaired_by` — RESOLVED AS NOT-THE-OWNER'S (ruled 2026-09-04, overseer 69th audit B3; implemented the same morning, builder `9e847cf`)

**Why this entry exists at all, and it is not the same reason as the entries
above it.** Nothing was deadlocked and no default fired. This is a disposition
that had **nowhere to live**: the Review addressed the question to the owner on
a page that is rewritten daily, the overseer answered it on a page that is
rewritten every six hours, and the builder implemented it — so an ask reached
the owner's desk, was disposed correctly, and left no durable record on any
document either of them reads. `experiments/decisions.py` grew
`VANISHED-OWNER-ASK` on 2026-09-04 and this was its single live positive; this
entry is the prescribed repair, not paperwork about paperwork.

**The question**, verbatim from `docs/PROGRESS.md`, Review 2026-09-03
(`f529ab1`), `FOR THE OWNER` item 3:

> ***"`run blocked` cannot see the project's largest unblock."*** `T2.01`
> blocks 38 specs; its repair runs through `D1.0`; no spec declares
> `depends_on: D1.0`, so the ranker scores that edge at zero and the 60th audit
> had to route the work by hand. **My recommendation: do NOT add the edge to
> the registry** — it would make `T2.01` unreachable until `D1.0` passes and
> would drift its certificate. Instead the ranker should read a declared
> `repaired_by` field that carries mass without carrying blocking semantics.
> *"That is a real design change to `run blocked`, so it is yours to authorise,
> not mine to make."*

**The ruling — it is a MEANS question, so it was never the owner's.** The
overseer's 69th audit, `FOR THE BUILDER` B3: the surviving option *"adds a
**reporting** edge that carries transitive-block mass without blocking
semantics, changes no `depends_on`, no verdict, no gate and no certificate, and
the Review itself already ruled out the variant that would change semantics"*.
`SYSTEM.md`'s third law governs a fork whose dangerous arm is already
eliminated: the loop writes it, it does not ask.

**Implemented and verified at `9e847cf`**, and the authorisation rests entirely
on the field staying reporting-only: `Spec.repaired_by` is read by `cmd_blocked`
alone — never by `Ledger.unsatisfied`, `_terminal_blockers`, `coverage` or any
gate. `run next`, `run status` and `coverage` are byte-identical to HEAD-before
and `unreachable` stays 91, at floor. First declaration `T2.01.repaired_by =
["D1.0"]`, which prints **`D1.0 = VOID  carries frees 35  (blocks 38)`** — a
number that existed nowhere before, with the repair spec's own VOID health
printed beside it so leverage cannot read as permission.

**Losers recorded:** (a) `T2.01.depends_on += ["D1.0"]` — the Review eliminated
it itself; it makes `T2.01` unreachable and drifts its certificate; (b) leave
the edge in prose — the status quo the 60th audit had to work around by hand,
and the shape `HR.5`→`HR.6` and `D19`→`HR.1` both cost an iteration each;
(c) escalate to the owner and wait — the D1 disease, twenty days for a question
no owner input could improve.

**To reverse:** revert `9e847cf`; `repaired_by` defaults to empty and every
ranking returns to its previous output, because nothing else reads the field.

Evidence: `docs/OVERSIGHT.md` 69th audit B3; `experiments/run.py`
`cmd_blocked`/`_check_ranker`/`_check_repair_edges`; `docs/LOOP_JOURNAL.md`
2026-09-04 08:xx; `experiments/decisions.py` `VANISHED-OWNER-ASK`, whose live
reading this entry moves 1 → 0.

## D21 — RESOLVED BY ARMED DEFAULT (fired 2026-09-06 00:1x UTC, builder, deliberately BEFORE the 06:37 FULL it commands — DEFAULT-ACTION-SAME-DAY, the race the 72nd audit named): today's FULL Review takes the W1 design as its FIRST DESIGN item, ahead of Part 2, behind the two d10-* gate rows; a W1 spec-family design is published as a routed disposition. GOAL.md IS NOT TOUCHED.

**The question** (2026-09-04, overseer, lifting the Review's 2026-09-03
recommendation off a page that is rewritten every morning): the Review
recommended that W1 stop being a queue row and become the project's stated
stage — a constitutional staging change — and wrote it in `docs/PROGRESS.md`,
where the next Review's rewrite would have erased it. Should the staging text
in GOAL.md change (options ii/iii), or something narrower?

**The default that fired — NEITHER (ii) NOR (iii).** A default may not edit
the constitution. What fires is the narrowest already-permitted action that
stops the recommendation from ageing in a file nobody reads: the 2026-09-06
FULL Review takes the W1 design as the FIRST DESIGN item on its docket and
ahead of Part 2 — but NOT ahead of the two `d10-*` gate rows, which keep the
head of the docket (narrowed by the 71st audit's amendment: the struck clause
never priced the ~16 h `D1.0` dispatch those rows release into W36's 30 free
hours) — and publishes a W1 spec-family design as a routed disposition. The
Review already owns the ordering of its own docket and `w0-too-shallow` is
already dated 09-06, so this re-orders a scheduled item and creates no new
permission. Explicitly NOT fired: any edit to GOAL.md, any change to the
ladder's stated stage, any re-parenting of registered specs — all three remain
the owner's alone (option ii/iii territory).

**Execution:** the ordering is stamped on the `w0-too-shallow` row in
`docs/REVIEW_QUEUE.md` (the docket the FULL reads), committed and pushed
before 06:37. The 09-02 stagger's published order already had the two `d10-*`
rows at the head and the design bundle before Part 2 — the firing makes that
ordering the DEFAULT'S act rather than a stagger the next router may undo.

**Invariants checked at firing:** no GOAL.md edit, no threshold moved, no
control loosened, no new permission created. The four CLAIM-DEAD commitments
(smell, balance, shelter/building, thermal) stay visibly red rather than
being resolved by fiat — resolving them is what the design this firing
sequences is FOR.

**To reverse:** the Review re-orders its own docket back; nothing else is
written that would need unwinding.

Evidence: `docs/DECISIONS_NEEDED.md` D21 entry + 71st-audit amendment;
`docs/REVIEW_QUEUE.md` 09-02 stagger + the firing stamp this entry names;
`experiments/decisions.py` DEFAULT-ACTION-SAME-DAY note (the one un-armed
entry, for exactly this reason); `docs/OVERSIGHT.md` 72nd audit finding 2.

## D16 — RESOLVED BY ARMED DEFAULT (fired 2026-09-06 00:1x UTC, builder): option (b) ALONE — the warning stands, T0.27 stays RED and is not touched, and the red is reported in every status until the pair ages out of history. A deliberate no-op, chosen because it costs a visible failure rather than manufacturing a green.

**The question** (46th overseer audit, 2026-08-29): the documented
audit-supersedes-FAIL loop manufactures threshold-move pairs that
`T0.27` must refuse forever (live: `live_checked_pairs` 4,
`live_unauditable_pairs` 26, `seeded_violations` 2). Should the guard be
relaxed (c), the row amended, or the red simply stand?

**The default that fired — (b) ALONE.** `T0.27` is NOT touched, not re-run,
not amended. The red row stands in every `run status` until the pair ages out
of history. This default deliberately picks the option that costs the ladder a
visible failure rather than the one that makes it green, because the party
proposing (c) was the party it would exonerate. It weakens nothing and widens
nothing. Nothing blocks on it: `T0.27` has no dependents; the cost is one
honest red row.

**Execution:** nothing. That is the option. This entry is the record that the
no-op was chosen by an armed default rather than by neglect, so the standing
red is a decision with a name on it and not an orphan.

**Losers recorded:** (c) relax the guard — the ratchet being defeated by the
party it caught; amending the row or re-running `T0.17` until the history
scrolls — each available, each the same defeat wearing procedure.

**To reverse:** the owner may take (c) by hand at any time; one entry here
records it and the guard's docstring says how.

Evidence: `docs/DECISIONS_NEEDED.md` D16 entry + 60th/62nd audit evidence
updates; `experiments/tests/t0_27_moved_threshold_leaves_artifact.py`;
commit `be60c3d`.

## D15 — RESOLVED BY ARMED DEFAULT (fired 2026-09-06 00:1x UTC, builder, AFTER D21/D16 per the verified 00:07 ordering): (c) AND (d) together — the overseer takes pace_gate on all but the first completed audit of each UTC day, and every organ script now writes usage attribution to /data/jack-logs/usage_ledger.jsonl at start and end of its run.

**The question** (44th overseer audit, 2026-08-29): the oversight organs are
exempt from the pace gate that stops the builder, and they draw on the same
meter. Should they be paced too, and should the meter's spend be attributed
instead of inferred?

**The default that fired — (c) AND (d), together, and neither alone.**

- **(c)** `scripts/overseer.sh` gains a pace check that EXEMPTS the first
  audit of each UTC day and applies the builder's own `pace_gate` to the
  other three slots. The 90% `usage_gate` is untouched and still runs first.
  `review.sh` and `field_watch.sh` are untouched — at 7/wk and 1/wk they are
  not the term that matters. Implementation detail that is part of the
  record: the exemption stamp is written only at the COMPLETED-audit point
  (beside D13's `NOOP_STATE`), so a first-of-day audit that dies does not
  consume the exemption and the next slot runs exempt again — the guard
  fails toward MORE oversight, the same design D13's no-op state uses.
- **(d)** all four organ scripts (`ladder_loop.sh`, `overseer.sh`,
  `review.sh`, `field_watch.sh`) append
  `{"organ","ts","pct","model_pct","phase"}` to
  `/data/jack-logs/usage_ledger.jsonl` at start and end of each run
  (`usage_ledger()` in `scripts/lib_usage.sh`; one CLI read per append, both
  meters parsed from one invocation, null on unreadable, never blocks). The
  next audit reads attribution instead of inferring it — the inference that
  produced three falsified price models in one week.

Option (b) was STRUCK from the default at arming (outside the repo — it
remains the owner's to take by hand); option (e) was STRUCK (three
consecutive weeks of expired free GPU quota is a measured cost).

**Spend-reducing, authorises nothing new:** no GOAL.md edit, no threshold
moved, no control loosened, no new tier or ceiling; the only behavioural
change is that up to three of four daily audits may skip when the meter is
above the builder's own pace line. The 45th audit's evidence update — that
the meter's rise is mostly off-box — is recorded and does not un-arm the
default: attribution (d) is exactly the instrument that dispute lacked.

**Certificate consequence, paid in the same slot:** `scripts/ladder_loop.sh`
is in `T0.33`'s IMPL_DEPS, so this edit stales that certificate; `T0.33` was
re-bought immediately after the firing commit (row in the ledger, same slot).

**The counterargument, carried from the arming:** the exemption for oversight
was deliberate — the machinery that catches drift kept the plain 90% gate at
full strength, and the 41st–43rd audits each earned it. Three-of-four is the
compromise: a full adversarial pass survives every day, only the redundancy
is paced.

**To reverse:** revert the firing commit — the pace check and the appends are
self-contained; delete `/data/jack-logs/overseer_pace.date`. The usage ledger
is additive and can stay whatever the owner decides.

Evidence: `docs/DECISIONS_NEEDED.md` D15 entry + 45th-audit evidence update;
`scripts/lib_usage.sh` `usage_ledger()`; the firing commit's diff across the
four organ scripts; the self-test line
`{"organ":"selftest","ts":"2026-09-06T00:15:49+00:00","pct":17,"model_pct":31,"phase":"d15-firing-check"}`
in `/data/jack-logs/usage_ledger.jsonl`.

**AMENDED SIX HOURS AFTER THE FIRING (76th audit B1/B2, executed by the
builder 2026-09-06 01:xx) — two repairs of the fired implementation, neither a
reversal, recorded here because the firing record above describes code that
was wrong on day one:**

- **(d)'s model attribution was wrong three ways and mis-attributed a fourth**
  (audit 9.1): `usage_ledger()` read `${JACK_LOOP_MODEL:-opus}`, which only
  the builder sets — the three auditor organs fell to `opus`, matched no meter
  line (the meter prints only `session` / `week:Fable` / `week:all models`),
  and wrote `model_pct: null` permanently; and a builder walked to a fallback
  billed its spend against the PRIMARY's line. The model is now a PARAMETER:
  each organ passes the variable it actually runs on, the builder's end-append
  passes the WALKED model, a floor-refused slot passes `none`, and the JSON
  gains a `"model"` field so a null `model_pct` (a model with no separate
  line) is legible rather than a bug. `pct` — the gate line — was correct
  throughout; no gate was ever affected.
- **(c)'s exempt slot moved** (audit 9.2): as fired, the guaranteed audit was
  00:37 — the one slot that can only read YESTERDAY'S Review page — while the
  slots that could read today's were first to be paced away. The exemption now
  goes to the first COMPLETED audit at or after the Review's 06:37 slot. Same
  spend arithmetic (still exactly one unpaced audit per UTC day), completion
  stamp unchanged (a dead audit does not consume the exemption; a completed
  pre-06:37 audit does not either). The fired default's text said "the first
  completed audit of each UTC day"; this changes WHICH audit, not HOW MANY —
  judged inside the armed text's intent (a full adversarial pass survives
  every day) and done on the audit's explicit order rather than silently.

## PL.00/RENDER — WINNER — coarse-shadow512 (2026-09-07, builder; probe, not run_bakeoff — the arms are loop configurations, not learners, so the 3-sigma learning gate has no referent; the probe carries PL.00's own VOID gates instead)

Ordered by the Review's disposition of `pl02-dependency-on-pl00-verdict-vs-table`
(arm iii). Question: can any renderer configuration put PL.00's loop over
LC.02's unmoved 5.0 sim-s/real-s floor with the eye live at one fresh frame
per decision? Decomposition first (per seed): the 40 ms eye = 4096^2 shadow
map pass 22.55–23.62 ms + 4x MSAA ~12.7 ms + reflection 7.11–7.86 ms +
update_scene 0.008 ms — two full-scene software-GL passes, MuJoCo defaults
nobody chose, serving 4,096 pixels.

metric: worst-seed loop throughput, scratch-cnn (the seat holder) live · floor 5.0 · artifact /data/pl00_render_bakeoff.json

| arm | worst-seed T | verdict |
|---|---|---|
| null (shipped: MSAA 4, shadow 4096) | 4.079 | red, as attempt 1 |
| ctx-reuse | bound 0.008 ms of 39.8 | FORECLOSED by measured arithmetic |
| batched-update | same bound | FORECLOSED by measured arithmetic |
| frame-skip-2 | 7.034 | scored, INELIGIBLE (changes the accounting unit: 1 frame per 2 decisions) |
| **coarse-shadow512** (MSAA off, shadows kept at 512^2) | **8.594** | **CLEARS — WINNER by the least-information-discarded ranking declared at `b7324ba` (the commit that also carries the artifact — no prior commit holds it; 84th audit B4)** |
| coarse-flat (MSAA off, shadow pass deleted) | 11.483 | clears; loses the ranking — shadows are depth information |

Discrimination held under the winner: ViT-S/14 @224 worst-seed 0.836, still
under the floor; render-only 8.949, over it — so the floor now rejects
encoders rather than any live eye, which attempt 1 recorded it could not do.
Adopted in `experiments/eye_quality.py`; NOT in `playground.py` (54 certs
declare it; render quality is the renderer's property, and whether existing
visual certificates migrate is routed, not assumed). PL.00 re-ran through the
runner: **PASS**, pure_T 8.903 ± 0.294, every gate green, commit `b7324ba` —
the PL.02 -> PL.00 edge dissolved by being satisfied, exactly as the
disposition pre-registered, with the edge and the floor untouched.

## D17 — RESOLVED BY ARMED DEFAULT (fired 2026-09-08 01:0x UTC, builder, the first decision this project has ever carried into OVERDUE — deadline passed at 2026-09-08T00:00 UTC, firing slot 01:0x): the PLASTIC-ONLY decree (GOAL.md:76) STANDS verbatim and unnarrowed; the re-open trigger is recorded FIRED and DISCHARGED with its number. GOAL.md IS NOT TOUCHED.

**The question** (2026-08-30, builder, from `PL.00`'s FAIL): `docs/CHAMPIONS.md`
pre-registered on 2026-08-09 that *"if a from-scratch encoder cannot hit the
PL.00 throughput floor on this hardware ... the decision returns to the owner
with that number attached."* `PL.00` FAILed attempt 1 (pure path 4.145
sim-s/real-s vs the 5.0 floor) and the trigger fired, so the decree returned
to the owner with the decomposition attached: render-only, with NO encoder at
all, already read 4.231 — the shortfall was the renderer's, with 2.6%
attributable to any encoder choice.

**The default that fired.** The decree stands verbatim; the trigger is
FIRED and DISCHARGED with its number. No decree narrowed, no threshold moved
(the 5.0 floor is LC.02's and stays at 5.0), GOAL.md untouched, `PL.02` stays
registered and runnable as the decree's falsifier so nothing goes claim-dead.
The default's ordered follow-up was builder work under rule 3 — a
renderer-cost bakeoff over the named arms — and it was ALREADY DONE at firing
time: it ran on the decide_by date itself (2026-09-07, `b7324ba`,
`experiments/tests/pl00_render_bakeoff.py`; the 40 ms eye was a 4096² shadow
map plus 4× MSAA, MuJoCo defaults), after which `PL.00` re-ran through the
runner and **PASSED** — pure_T 8.903 ± 0.294 vs the unmoved 5.0 floor, ViT
reference still fails at 0.830, render-only clears at 9.549 so the floor now
rejects encoders. **The trigger's own premise is FALSE at firing**: the
from-scratch encoder does hit the floor. The firing is therefore pure
paperwork — the alternative, quietly extending a deadline once it goes red,
is the deadlock the armed-default mechanism was built to replace.

**Invariants checked at firing:** no GOAL.md edit, no threshold moved, no
control loosened, no new permission created, nothing re-run. The one live
design question the default named — rays or pixels for Jack's eye in W1 —
remains open and remains routed design work, not resolved by this firing.

**To reverse:** the owner may rule differently at any later date at no cost;
the default wrote nothing to GOAL.md and moved no number.

Evidence: `docs/DECISIONS_NEEDED.md` D17 entry + its EVIDENCE UPDATE
2026-09-07 + the 84th audit's OVERDUE NOTICE (2026-09-08 00:38 UTC);
`docs/OVERSIGHT.md` 84th audit B1 (the routing, with required wording);
`experiments/decisions.py` overdue rule `(today - decide_by).days > 0`;
`b7324ba` (bakeoff + PL.00 PASS).

## D22 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 06:5x UTC by the OVERSEER, 89th audit, `7d0b49c`; transcription completed by the builder 2026-09-12 ~17:1x): design authority over spec design STAYS WITH THE REVIEW, unchanged and unnarrowed. The default writes nothing.

**The owner did not rule by 2026-09-08, so the pre-registered default fired.**

**The question** (2026-09-04, overseer, lifting the Review's own `FOR THE OWNER`
item 1 onto a desk that is not rewritten every morning): the Review reported that
design throughput — not compute, not the ladder — had become the binding
constraint on the project, and asked to hand spec *drafting* to the builder. It
wrote that ask into `docs/PROGRESS.md`, which is regenerated daily, so the ask
would have vanished with the next page. The overseer lifted it to
`DECISIONS_NEEDED.md` precisely so it could not.

**The default that fired: (i) THE RULE STANDS.** Design authority over spec
design stays with the Review. Per the entry's own text, *"Reversal: none needed
— the default writes nothing."* Nothing is re-parented, no threshold moves, no
control is weakened, `GOAL.md` is not touched, no certificate is staled, no spec
is failed, no run is refused, and no commitment goes claim-dead. It was the only
legal default of the three: **(iii)** widens what the builder is permitted to
take, and a default may not widen permissions by silence; **(ii)** spends model
time against `D15`, which has since fired.

**The price, restated rather than buried, because the entry priced it and the
bill came due.** `D22` predicted silence would cost *"approximately 17 further
net queue rows at the measured rate"*. Measured at firing: 46 rows routed, 38
live, oldest live 19 days, drain **UNBOUNDED** (10 arrived against 6 disposed
over the trailing 7 consumer cycles), and 2026-09-13 carrying **14 rows against a
measured capacity of 6**. The divergence the entry named happened. Firing (i)
does not repair it and nobody should read it as a repair; it records that *who
holds design authority* is settled by silence, and leaves the drain standing as
the separate, still-open problem it is.

**FIRST OF ITS KIND, and the reason is itself the finding.** Every one of the
twenty armed defaults fired before this one is stamped `(builder)`. This is the
first fired by a non-builder organ. The overseer's stated reason: `D22`'s default
requires no code, no script, no spec, no ledger row and no file outside its own
brief — and **the builder had refused 94 consecutive paced slots** (104 by the
time this transcription was written), so the mechanism's only executor could not
be reached while the entry sat red for four days. The 88th audit's two grounds
for declining did not survive contact with the entry: `D13`'s parenthetical
(`DECISIONS_RESOLVED.md:459`) is specifically about `scripts/overseer.sh`, which
`D22` does not touch, and a consistent stamp is a convention rather than a rule.

**The permissions tension, declared rather than hidden.** The overseer's brief
says it MAY NOT *"resolve an owner decision"*, and it read firing a pre-armed
default as the opposite of resolving one — executing an already-permitted action
the project armed in advance, substituting no judgement for the owner's. **If the
owner reads it the other way, the revert is one append** and `D22` returns to the
open set with its options, default and `decide_by` untouched. `decisions --check`
drops the entry by `_SETTLED` (`experiments/decisions.py:319`, `:366`), exactly
as `D17`'s addendum did — a firing, not a re-dating. The deadline was **not**
extended.

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved,
no control loosened, no new permission created, nothing re-run, no certificate
staled. Verified against the firing commit `7d0b49c` and its diff.

**To reverse:** the owner may rule (ii) or (iii) at any later date at no cost,
and that ruling is unaffected by this default having fired.

**Why this entry exists at all, stated because it is the point of B1.** The
overseer fired the default but could not write this file or `docs/LOOP_JOURNAL.md`
— neither is in its brief — and it said so in its own append rather than leaving
the gap to be discovered: *"The default has fired; its full transcription has not
been written, and this paragraph exists so that nobody later reads the gap as a
lost record."* That is the record being completed here. An organ that declares
the half of a job it cannot do is the mechanism working; a firing whose
transcription is silently never written is a `VANISHED-OWNER-ASK` wearing a
resolution's clothes.

Evidence: `docs/DECISIONS_NEEDED.md` D22 entry + its OVERDUE NOTICE
(2026-09-09 07:0x, 86th audit) + its `RESOLVED BY ARMED DEFAULT` append
(`:5924`); `7d0b49c` (the firing); `docs/OVERSIGHT.md` 89th audit B1 first
sub-item (the routing of this transcription); `docs/PROGRESS.md` 2026-09-12
`FOR THE OWNER` item 2 + its in-place correction (`a4edb79`).

## D18 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 ~17:2x UTC, builder): MEASURE AND REPORT, GATE NOTHING, RELAX NOTHING. The ~1.5 GB ceiling STANDS VERBATIM. The default's code already existed; what fired was the report, and it reframes the question.

**The owner did not rule by 2026-09-09, so the pre-registered default fired.**

**The question** (2026-09-02, overseer, from a live measurement): `run_spec T2.00`
was sampled at **7.57 GB RSS against `SYSTEM.md`'s ~1.5 GB ceiling — 5.0x** — on
a box with paying tenants, with `nice 19` honoured and only the memory half
breached. `T0.07` carried `policy_peak_rss_mb = 6991.0` on a **PASS** row
re-stamped the same day. No OOM kill ever, swap flat, no tenant harmed — but free
memory reached 808 MB on a 22.9 GB box, and the margin was luck rather than
design. Is the ceiling wrong, or are the specs in breach? Both answers change
what is *permitted* on a shared machine, which is `SYSTEM.md` class 3 (CONDUCT),
so rule 3 does not reach it: the loop may measure, it may not set the bar.

**The default that fired.** The ~1.5 GB figure stands verbatim — not raised, not
narrowed, not annotated with an exception. The ceiling is left **BREACHED AND
VISIBLE** rather than resolved, because both (a) and (b) are the owner's.

**THE PREMISE WAS ALREADY SATISFIED AT FIRING — the second time this project has
fired a default whose ordered work was already done** (`D17` was the first, and
its trigger's premise was likewise false at firing). Both code changes the
default ordered landed **2026-09-03 in `a071d91`**, six days before `decide_by`
and before the entry was armed: `proc_memory_report()` (`lib_procwatch.sh:268`,
wired at `ladder_loop.sh:238`) names every project python over the ceiling from
`VmHWM`, never killing; and `run_spec` records `peak_rss_mb`
(`protocol.py:2910, :3152`). The implementation is **stronger than the default
specified** — `max(RUSAGE_SELF, RUSAGE_CHILDREN)` rather than the ordered
`RUSAGE_CHILDREN` alone, because `run_spec` calls the experiment inline and a
children-only reading would have recorded ~0 MB for the exact `T2.00` scar the
field exists for — and it carries `peak_rss_inherited` so a `--gate` sweep's
inherited high-water mark cannot masquerade as a spec's own peak.

**So the firing wrote no code. What it did was the never-executed second word of
the default: REPORT.** The instrument had been recording for nine days and no
organ had read it in aggregate. Across the ledger at firing — 143 rows, 71
carrying the metric, 69 own-peak:

    own-peak rows over the 1536 MB ceiling : 8 of 69 (12%)
    median own peak                        : 239.7 MB  (6.4x UNDER the ceiling)
    max own peak                           : 7370.0 MB  T1.03, 4.8x, inherited=False
    status of all 8 breaching rows         : PASS

    T1.03 7370.0 (4.8x) · T0.07 6943.4 (4.5x) · T0.04 3539.0 (2.3x)
    T0.16 2632.3 (1.7x) · T1.04 2073.9 (1.4x) · PG.6  2073.4 (1.3x)
    LC.02 2021.4 (1.3x) · T0.14 1783.0 (1.2x)

**THE FINDING, and it is why firing pure paperwork was still worth doing.** The
entry posed a binary — stale ceiling, or ladder in breach — on the evidence of
one live sample. The ledger-wide reading is **neither wholesale**: the ceiling is
right for **88%** of the measured ladder, whose median spec peaks at a *sixth* of
the limit, and the breach is a **heavy tail of eight named specs**. A single
7.57 GB observation generalised to "the specs are in breach" would have been
wrong about 61 of 69 rows. That is a materially easier decision than the one
escalated, and it is now on the owner's desk with names and numbers attached
rather than an anecdote.

**Invariants checked at firing:** no `GOAL.md` edit; no threshold moved (1536 MB
unchanged at `lib_procwatch.sh:62`, ~1.5 GB unchanged in `SYSTEM.md`); no control
loosened; no new permission created; nothing re-run; no certificate staled; no
spec failed; no run refused. The eight breaching specs keep their PASS rows —
**reporting a breach is not failing a spec**, and gating on `peak_rss_mb` is
exactly what the default forbade.

**To reverse:** there is nothing to revert in code — the default's implementation
predates this firing and was not written by it. The owner may rule (a) *the
ceiling is stale, raise it* or (b) *the specs are in breach, fix them* at any
later date at no cost; this default chose neither and the number is now there to
rule on.

Evidence: `docs/DECISIONS_NEEDED.md` D18 entry + its EVIDENCE UPDATE 2026-09-06
12:4x (78th audit) + OVERDUE NOTICE 2026-09-10 07:0x (87th audit) + the
`RESOLVED BY ARMED DEFAULT` append; `a071d91`; `scripts/lib_procwatch.sh:62,
:104-113, :252-285`; `scripts/ladder_loop.sh:238`; `experiments/protocol.py:433,
:2910, :3040, :3143, :3152`; `docs/OVERSIGHT.md` 89th audit B1.

## D24 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 ~17:3x UTC, builder): (iii) DECLARE, DO NOT DECIDE. The Learning-core seat's arena is `VENUE-UNAFFORDABLE`. The "~10x" scale ratio is UNTOUCHED.

**The owner did not rule by 2026-09-11, so the pre-registered default fired.**

**The question** (2026-09-06, Review, FULL): `LC.07` — the scale-transfer arena
registered by `D10`'s own firing commit *specifically so the wm-latent seat would
not be held with a dead arena* — measures **~526 wall-hours** (21 runs, ~132
kernel-hours at ideal 4-way packing) against a **30 h/week** free allocation:
**≈17.5 weeks of every GPU hour this project has, for one seat's arena.** The
pilot rig is healthy (seed 90, kernel `jack-ladder-1788297232`, 0.44 h, all 7 run
classes measured, wiring exact, physics finite, RSS ~550 MB) and its
pre-registered branch B fired on **arithmetic, not a fault**: rule A caps a
full-scale run at 8.5 h wall; the cheapest class projects 14.49 h and the arm
40.86 h — 4.8× the kernel ceiling. Bought, shrunk, or declared unaffordable?

**The default that fired: (iii).** One label, with the arithmetic beside it, on
the seat in `docs/CHAMPIONS.md`. No threshold moves in either direction, no spec
is failed, no run is refused, no certificate is staled. `LC.07` stays
PILOT-BLOCKED with `_GATES_FROZEN` False and `run()` refusing — identical
behaviour before and after.

**It was the only legal default of the three.** (i) commits ~17.5 weeks of the
entire free GPU allocation **by silence**; (ii) SHRINK THE CLAIM re-reads the
"~10x" downward to fit the budget, which is a **threshold move by silence** and
the precise act `SYSTEM.md` law 4 exists to forbid — a 10× scale-transfer claim
is strictly stronger than a 3× one, and shrinking it buys a PASS with a smaller
question. **(ii) did not fire; the 10x is intact at full strength, and nothing in
the silence should be read as choosing it.**

**Why the obvious repair was refused, and by the Review rather than by this
firing.** Checkpoint/resume surgery on `survival.py` is demonstrably feasible —
`LF.02` PASSed on 2026-09-03 with a W0 life SIGKILLed mid-decision-stream and
resumed **bit-exactly** over 1000 decisions across all four stores, weights-only
null diverging 8.1 ± 2.7. But it repairs the **per-run 8.5 h ceiling** and does
not touch the **526-hour total**: it converts *impossible* into *17.5 weeks*, and
bills a surgery that stales every `LC` and `XL` certificate for the conversion.
Refused on `lc07-checkpoint-branch` (DISPOSITIONED 2026-09-06).

**Verified at firing rather than asserted.** `champions --check` exits 0 before
and after, and **every ratchet counter is byte-identical across the edit**: 0/0
phantom arena, 2/3 unfalsifiable, 2+1/4 uncontestable, 2/2 unverified verdicts,
3/3 trigger debt, 1/1 kindless discharges. A declare-only default that moved a
counter would not be declare-only.

**One deliberate narrowing of the default's own words, declared not taken
silently.** The default asked that the label make `champions` "print the
uncontestedness it currently implies". `champions.py`'s declaration grammar
admits only `HELD:`/`ARENA:`/`VERDICT:`/`TRIGGER:`, so an invented `VENUE:` field
would be **silently ignored** — a machine-readable line that reads as declared
and is inert, exactly the trap `DECL-INCOMPLETE` exists to catch. The label went
into the seat's prose cell; the `SEAT:` declaration line is **untouched**. Nothing
is lost: the tool already prints the same fact from the other end, listing the
Learning core in **TRIGGER DEBT** with `LC.07=PILOT-BLOCKED,
LC.03=VOID-FORECLOSED, UB.10=VOID` — every declared re-open trigger a closed
door. The default's own reversal clause ("nothing downstream reads it")
anticipated this.

**The price, restated rather than buried, because the entry priced it:** under
(iii) the wm-latent seat stays **UNDECIDED with no reachable arena**, and the
honest reading of that is that **this project cannot currently contest its own
learning-core choice.** Making it visible is the point. It is not a fix and it is
not being called one.

**Invariants checked at firing:** no `GOAL.md` edit, no threshold moved (the 10x
ratio, rule A's 8.5 h, and every `LC` gate unchanged), no control loosened, no new
permission created, nothing re-run, no certificate staled, no spec failed, no run
refused, no commitment claim-dead.

**To reverse:** change one label. The owner may rule (i) or (ii) at any later
date at no cost, unaffected by this default having fired.

Evidence: `docs/DECISIONS_NEEDED.md` D24 entry + its `RESOLVED BY ARMED DEFAULT`
append; `docs/CHAMPIONS.md` Learning-core row; `experiments/champions.py:551,
:620-700`; `/data/lc07_pilot.json`; `lc07-checkpoint-branch`; `LF.02` PASS
2026-09-03; `docs/OVERSIGHT.md` 89th audit B1.

## D23 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 ~17:4x UTC, builder): (iii) MEASURE THE COMPOSITION, GATE NOTHING, TIGHTEN NOTHING. `FAIL-UNOWNED` keeps its definition and its floor of 0. The first reading is 20, against a drain of UNBOUNDED.

**The owner did not rule by 2026-09-11, so the pre-registered default fired.**

**The question** (2026-09-05, Review, DAILY) — *is routing the same as owning?*
At 01:16 the 72nd audit shipped `FAIL-UNOWNED`, a counted, ratcheted class for *a
settled FAIL with no repair owner*, at an honest baseline of 4. It immediately
found what five other instruments had missed for a fortnight: `XL.01`, *"death
does not erase what he learned"*, had read **FAIL for 17 days** with no owner, no
clock and no queue row, while `run blocked`, `coverage`, `review_queue`,
`champions` and `decisions` each reported it fine — because every one of them is
keyed to a spec's **reachability** and none to its **disposition**. At **01:19,
three minutes later**, the same audit routed the four orphans into
`docs/REVIEW_QUEUE.md` with `DUE 2026-09-13` and the count went **4 -> 0**. The
class read `AT floor — ok`. Nothing in that was misconduct and the entry said so
first: routing is the correct response, the rows are real, dated and reasoned,
and the audit used `next_free_due` rather than piling. **The question was what
the discharge measured** — `coverage`'s own definition makes a queue *mention*
sufficient for ownership, and that same morning the desk which now owned all four
reported its own drain as **UNBOUNDED**.

**The default that fired.** A single printed counter beside `FAIL-UNOWNED`,
computed from data both tools already hold: the count of settled FAILs whose only
repair owner is a `REVIEW_QUEUE` row, printed together with that file's own
`drain` reading.

**THE FIRST READING IS FIVE TIMES THE SIZE OF THE QUESTION THAT ARMED THE
ENTRY.** `D23` was about four orphans. Measured at firing:

    FAIL-UNOWNED                 0   settled FAIL(s) with NO repair owner  (AT floor — ok)
    FAIL-OWNED-BUT-UNDRAINED    20   whose ONLY repair owner is a REVIEW_QUEUE row
                                     ...and that desk's own drain: UNBOUNDED

    DP.05, HR.5, LG.10, LT.01, ME.11, ME.11.B, ME.11.C, ME.11.D, NE.01, T0.27,
    T2.05, T2.07, T2.10, T2.15, T3.09, T4.02, W.1, W.2, W1.00, XL.01

**Both halves are true at once and neither is a fault.** The floor reading of 0
is honest; routing is the correct repair. What nobody could see is the
composition: **twenty** of Jack's settled negatives rest on a warrant from a desk
that measures itself as unable to say when it will pay. `AT floor — ok` is true
and misleading simultaneously, which is the one outcome the entry existed to
prevent, and the sentence is now printed hourly instead of reconstructed once by
a Review.

**WHAT DELIBERATELY DID NOT FIRE, because it is the reason (iii) was the only
legal default.** Option (ii) — *a queue row counts as an owner only while the
drain is bounded* — is a **TIGHTENING**. Firing it by silence would let this
instrument red-light twenty of the builder's specs on a property of a *different*
desk's throughput, with nobody having ruled on it. Option (i), the status quo,
writes the composition off rather than leaving it visible. The new counter
therefore has **no baseline, no `!! MOVED` line, and no effect on any exit
code** — and that absence is **asserted by a fixture arm**, not left to good
intentions, because the number *legitimately rises* when a desk correctly routes
an orphan: a floor here would punish exactly the behaviour the sibling class
rewards. It is MONOTONE — a number appears where there was none — and it is the
same shape `D18`'s default took on the memory ceiling.

**The guard, because the scar demands one.** Four known-answer arms in
`_fail_unowned_fixture` (the `T0.31` P4/P5/P6 shape), each **verified to fail
when the property it guards is broken and to pass when restored** — P1 the
composition (letting `repaired_by` drift into the queue-warranted set fires 3
failures), P2 the load-bearing arm, P3 the no-ratchet guarantee, P4 the healthy
state. **P2 is the one that matters most:** an absent git throughput baseline
must report drain `UNKNOWN` and **never as bounded**. The join of the two numbers
IS the measurement, so defaulting the missing half to a comfortable value would
manufacture precisely the reassurance this entry is about — `Arm.cost`'s lesson,
that a sentinel which is also a valid value cannot be detected.

**Certificate cost, paid rather than deferred:** `T0.21` declares
`experiments/coverage.py` in `IMPL_DEPS`, so this edit staled its certificate and
it was re-bought PASS in the same unit.

**Invariants checked at firing:** no `GOAL.md` edit; no threshold moved
(`FAIL_UNOWNED_BASELINE` still 0, its definition unchanged); no control loosened;
no new permission; nothing re-run for a better number; no spec failed; no run
refused; no commitment claim-dead. `coverage` exits 2 before and after on the
same known residents.

**The price, restated because the entry priced it:** under (iii) the four
orphans stay owned-on-paper and dated 2026-09-13, and the entry said *"if the
drain is still UNBOUNDED then, the new counter is what will say so."* It is, and
it does — a day early, by design.

**To reverse:** delete one printed line; nothing downstream reads it.

Evidence: `docs/DECISIONS_NEEDED.md` D23 entry + its `RESOLVED BY ARMED DEFAULT`
append; `experiments/coverage.py` `FAIL_OWNED_QUEUE_FORMS`,
`fail_owned_but_undrained()`, the block comment and the four fixture arms;
`experiments/review_queue.py:386`, `:851`; `docs/OVERSIGHT.md` 89th audit B1;
the 72nd audit's `6fbac74` (the class) and `52dcf9e` (the routing, three minutes
later).

## D26 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 ~17:5x UTC, builder): (iv) MEASURE ONLY, GATE NOTHING, RELAX NOTHING. The skip line now names who drew the meter. Option (i) was NOT taken and remains the owner's.

**The owner did not rule by 2026-09-10, so the pre-registered default fired.**

**The question** (2026-09-09, Review, DAILY): the builder was dark because
`pace_gate` rations it against `week:all models` — a **shared** pool, two-thirds
of which this project did not draw — while `scripts/review.sh:30` calls
`usage_gate` **without** `pace_gate`, so the two organs that report to the owner
are rationed by nothing. The blackout ran to **104 consecutive skipped slots**
(2026-09-08T08:23 → 2026-09-12T17:07), the largest single loss of builder
capacity since the 4.3-day August blackout, and it happened while the ladder was
healthy with runnable units on the board.

**The default that fired.** `pace_gate`'s skip line additionally prints this
project's own attributed spend beside the shared total, split builder vs desks,
and counts consecutive dark slots. It **gates nothing and changes no behaviour** —
verified on both branches after the edit: at live settings `pace_gate` still
returns 0 and prints nothing on the release path.

**THE INSTRUMENT INDEPENDENTLY REPRODUCES THE HAND COMPUTATION IT WAS BUILT
FROM.** The 89th audit's RANK 1 measured by hand: *"the builder drew 18 of 75
meter points (24%) ... the overseer and Review together drew 7 (9%) ... 50 (67%)
were drawn while no organ of this project was running at all."* Computed from the
`usage_ledger.jsonl` rows: **builder 18 (23%), desks 8 (10%), both 0,
NOT THIS PROJECT 50 (65%) of 76** — the same three figures, one meter point
later. The claim that `pace_gate` rations the builder against a total that is
two-thirds not its own is now reproducible hourly rather than recomputed by hand
each week.

**WHY THE UNION AND NOT THE SUM.** The overseer and the Review run daily and
their sessions overlap; summing per-session deltas double-counts every
overlapping minute and inflates the desks' share. Each span of meter rise is
attributed **once**, to the SET of organs alive during it — a span with a builder
and a desk alive is `both`, not a point to each. Arm P1 plants two **fully
overlapping** desk sessions across a 4-point rise and requires 4; the summing bug
returns 8.

**WHAT DELIBERATELY DID NOT FIRE, and it is the point of the default being
(iv).** Option (i) ATTRIBUTE THE LINE — pace against this project's own spend
rather than the shared total — is what **both** desks recommend, and on the day
it would have released the builder immediately (23% own-spend against a 44%
line). It **widens what the builder may spend**, and `SYSTEM.md` law 4 forbids a
default loosening a gate by silence. It did not fire, and **this firing is not a
step toward it**: the number is printed; what to do about it is untouched and
entirely the owner's. (ii) is the same act with a cruder instrument; (iii) writes
off the blackout and keeps the blind spot.

**The price, restated because the entry priced it — with the correction it could
not have known.** `D26` predicted *"the builder stays dark for the remaining ~35
hours"*. The streak ran to **104 slots**, and what released it was the pace line
rising into a **flat** meter, not the meter falling. **(iv) does not fix that and
is not called a fix.** It makes the next occurrence visible within one slot
instead of within one Review — which is precisely what the 104-slot streak cost:
the one fault this gate cannot report about itself, because the organ that would
report it is the organ being skipped.

**Guards, because a measure-only instrument still gets believed.** Six
known-answer arms in `--selftest`, all green, each planted beside the state it
must not be confused with: **P1** the union; **P2** the split including the span
nobody was awake for (the finding itself); **P3** a builder+desk overlap billed
once to `both`; **P4** the **weekly reset detected as the meter FALLING** rather
than from a hard-coded date — the mistake `CLAUDE.md` has made twice, a cached
reset date going five days stale on the one page that opens by declaring no
number is cached on it; **P5 UNKNOWN IS NOT ZERO** — an unreadable ledger reports
`known=False` with `None` buckets and the printed line says so in words, never a
comfortable 0 (`Arm.cost`'s lesson: a sentinel that is also a valid value cannot
be detected); **P6** the dark streak ends at the last real slot, and 0 is
distinguishable from unknown.

**89th audit B2, landed in the same edit.** The skip line prints the pace line's
**endpoint**: *rising to 90% at week's end = the hard stop, so the line always
converges.* `allow` is a pure function of the clock with zero variance
(`PACE_FLOOR + ((PACE_CAP-PACE_FLOOR)*elapsed + 99)/100`), so `allow(100)` is the
constant 90 — which *is* the 90% stop. Printed so *"pace_gate never releases the
builder"* is unavailable as a sentence to the next reader: that claim was made,
and it was refutable in one substitution from a formula both desks had already
pasted into their own reports.

**Invariants checked at firing:** no `GOAL.md` edit; no threshold moved
(`PACE_FLOOR` 25, `PACE_CAP` 90, `MODEL_FLOOR` 95 all unchanged); no control
loosened; no new permission created; nothing re-run; **no certificate staled** —
no spec declares `scripts/lib_usage.sh` in `IMPL_DEPS`, and `run stale` is
identical before and after; no spec failed; no run refused; nothing spent.

**To reverse:** revert one commit; the gate's behaviour is unchanged by it.

Evidence: `docs/DECISIONS_NEEDED.md` D26 entry + its three EVIDENCE ADDENDA
(2026-09-09 86th audit, 2026-09-10 Review, 2026-09-11 Review premise-correction)
+ its OVERDUE NOTICE (2026-09-11, 88th audit) + the `RESOLVED BY ARMED DEFAULT`
append; `scripts/usage_attribution.py`; `scripts/lib_usage.sh` `pace_gate`;
`/data/jack-logs/usage_ledger.jsonl`; `docs/OVERSIGHT.md` 89th audit B1, B2,
RANK 1.

## SO.10 — TIE — laplace-full
laplace-full leads laplace-w30 by only 0.26 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (laplace-full, cost 0).

metric: `div_lastq`  ·  null 0.044 ± 0.069  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of one already-recorded evidence stream that no arm can perturb (the diary holds the claim and the finding; the follow decision is never recorded, and `rng_agent` draws once per round whatever the rule returns). A low score is therefore a property of the RULE — full-history Laplace cannot migrate, last-claim-only is memoryless — and not evidence that its run was broken, which is exactly the case `validity` mode would mis-VOID. The gate itself is unmoved at 3 sigma and MIN_FINISHERS still applies.

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| laplace-full | 0.722 | 5.79 | pass | 0.0 |
| laplace-w30 | 0.689 | 5.11 | pass | 1.0 |
| exp-decay-h15 | 0.678 | 5.91 | pass | 1.0 |
| last-1 | 0.544 | 3.60 | pass | 2.0 |
| control:pooled-scalar | 0.044 | 0.00 | FAIL | 0.0 |

**THE ARM THIS TIE NAMES DID NOT TAKE THE SEAT, AND THE ROW IS THE AUTHORITY**
(builder, 2026-09-13, appended by hand under the machine-written verdict above
so nobody adopts `laplace-full` off this table). `run_bakeoff` arbitrates the
METRIC. `SO.10` recorded **FAIL**, because the spec pre-registered a second
gate the decision primitive cannot see: the winner must be ELIGIBLE to hold the
seat, and `laplace-full` **cannot migrate** — after the advisors swap roles its
divergence is **negative on every seed** (−0.133 / −0.067 / −0.133 against
`MIN_MIGRATE` 0.40). It goes on trusting the voice that is now lying. The
cheapest arm won the headline number by being unable to forget.

Per-arm eligibility, all three seeds (`prior_ok` / `noleak` / `migrate`):
`laplace-w30` 1/1/1 · `exp-decay-h15` 1/1/1 · `laplace-full` 1/1/**0** ·
`last-1` 1/**0**/1 · control `pooled-scalar` **0**/1/**0**.

**The Person-model seat therefore stays VACANT** — that is the spec's
pre-registered consequence, not a judgement made after seeing the number, and
re-ranking to the best *eligible* arm after the fact is exactly the move
pre-registration exists to forbid. Two eligible candidates are now measured and
tied; which of them the seat goes to, and whether a seat's race should screen on
admission BEFORE it scores, is routed to the Review as
`so10-tie-break-hands-the-seat-to-an-ineligible-arm`.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | 1.0 |

## D25 — RESOLVED BY ARMED DEFAULT (fired 2026-09-14 ~00:2x UTC, builder): (iii) FIX THE SEAL, BUY NOTHING. `lib_seal.sh` learns to tell "died with nothing done" from "died having committed everything". No wall clock was bought; (i) and (ii) were NOT taken.

**The owner did not rule by 2026-09-13, so the pre-registered default fired.**
Transcribed here from the firing record in `docs/DECISIONS_NEEDED.md`, which
carries the full evidence list (95th audit B4 — this was the only one of the
last five fired defaults with no entry on this page).

**The question** (2026-09-07, Review, DAILY): five Sunday FULL runs scheduled,
five deaths — four at `Reached max turns (60)`, and the fifth, on 2026-09-06, at
`timeout(1)`'s 40-minute wall with `rc=124`. But that fifth run died having
already committed everything: both `d10-*` gate adoptions, the `w0-too-shallow`
disposition, `D24`, two `CHAMPIONS` seats, `docs/PROGRESS.md` in full and its
`PROGRESS_LOG` row at 07:12, five minutes before the kill. **It lost its exit,
not its work.** `lib_seal.sh` reads only `rc != 0`, so it gave that finished page
the same *"THIS IS A DRAFT, NOT A FINDING ... UNVERIFIED"* it correctly gives a
run that wrote nothing. The cost was already realised: the builder spent a full
day executing seven of that page's nine `FOR THE BUILDER` items off a document
formally marked unverified, and was right to.

**The default that fired.** `scripts/lib_seal.sh` gained a fourth case: a dying
organ may pass a **tail receipt** — the last item on its own checklist, in a file
the seal can read — and with that receipt present the banner reads *"CHECKLIST
COMPLETE — THE RUN WAS KILLED ON THE TAIL, NOT MID-REPORT"*. `scripts/review.sh`
passes `docs/PROGRESS_LOG.md` and today's row pattern. **A run with no receipt
keeps the old wording BYTE-FOR-BYTE**, and an organ that configures no receipt
at all (overseer, field watch) is untouched. Strictly monotone: it can only put
a truer banner where a false one stood.

**THE DEVIATION, declared at firing rather than taken silently.** The default's
letter required *"committed `docs/PROGRESS.md` AND appended its `PROGRESS_LOG`
row"*. Git says **the first conjunct is false of the very run the decision
cites** — the 09-06 agent left the page dirty and `lib_seal.sh` committed it
(`cf18320`, 07:17:11), which is the act the banner was complaining about.
Implemented literally, the branch **would never have fired on its own scar**. So
the gate is the receipt ALONE and the page's custody is reported rather than
required. This is the source of the 09-13 lesson: *a pre-registered remedy can
name a condition its own motivating case does not satisfy.*

**The trap it does not walk into.** `review.sh` writes its own `PROGRESS_LOG`
row when the agent died before the append (76th audit B4), and that row matches
the date pattern exactly — a receipt an organ's dead-run fallback can satisfy is
the organ certifying itself. `lib_seal.sh` refuses any candidate containing
`INCOMPLETE`, and `test_lib_liveness.sh` asserts it directly.

**THE LOSERS, recorded as this page requires.** **(i) RAISE THE WALL CLOCK** —
refused as an illegal default before it was refused on merit: it spends credits
against the shared `week:all models` meter by silence, and that meter's
exhaustion took every organ dark for 4.3 days and expired three consecutive GPU
allocations. An outlay that size may not fire because nobody answered.
**(ii) CHANGE NOTHING** — writes off a measured falsehood and leaves it standing
every Sunday. Both remain the owner's to rule at any later date, at no cost, and
this firing is not a step toward either.

**The price, restated because the entry priced it.** Sunday FULLs keep exiting
`rc=124` and keep looking unhealthy to anything that reads exit codes alone.
This desk takes a truthful banner over a green exit code on purpose.

**Verified at firing, not asserted:** `scripts/test_lib_liveness.sh` all green
with 14 new assertions across all four cases; `--firing-check WORKTREE` exit 0
(no `GOAL.md` edit, no numeric bar moved in either direction); `decisions
--check` and `champions --check` rc=0; no certificate staled — `lib_seal.sh` is
in no spec's `IMPL_DEPS`. **To reverse:** stop passing arguments 7 and 8 to
`seal_output` in `scripts/review.sh` — one line.

Evidence: `docs/DECISIONS_NEEDED.md` (the firing record in full);
`scripts/lib_seal.sh`; `scripts/review.sh:106-115`;
`scripts/test_lib_liveness.sh`; `cf18320` and `docs/PROGRESS_LOG.md:26`;
commits `696bfcb`, `71eb183`, `958c5ec`.

## D20 — RESOLVED BY ARMED DEFAULT (fired 2026-09-19 ~00:2x UTC, builder): (i) WALL STANDS. The 57600 s wall ceiling is untouched, the detached lane is CLOSED to registered spec work, and no new spec registers in `cpu<48h` until the owner rules. A record, no code.

**The owner did not rule by 2026-09-18, so the pre-registered default fired.**

**The question** (2026-09-04, overseer, from a live reading taken three hours
after the meter shipped): `cpu_budget.CPU_DAY_CEILING_S = 57600.0` (16 h) is
charged in **wall clock**, while `rtf.BUDGET_SECONDS["cpu<48h"] = 172800` (48 h)
is a registered, legal cost class served by `scripts/launch_detached.sh` — so
one legal `cpu<48h` run overruns the day by arithmetic (1.50x; 3.00x under the
since-fixed double-billing), and a single overrun day makes `admit_detached`
refuse every detached launch AND `gate_cpu_child` refuse every runner CPU
child. Every answer except "wall clock stands" increases how much of a shared
machine this project may take — `SYSTEM.md` class 3 (CONDUCT), the owner's
alone, which is why no bakeoff could settle it and why it sat armed.

**What fired, and it is a record with no code.** The ceiling means "this
project may hold the box for 16 h of any day, whoever is running". `cpu<48h`
is not a class this box can serve under that reading; the detached lane is
declared CLOSED to registered spec work; the builder registers no new spec in
the class. `launch_detached.sh` is byte-untouched; nothing is re-run; the
foreclosure stays VISIBLE as the 68th-audit B3 printed number
(`cpu_foreclosed_now`, reading 0 today).

**ENFORCEMENT, stated so nobody discovers it by violating it (101st audit FTB
3):** the no-new-`cpu<48h`-registration constraint is held by NOTHING but this
note — no code in `registry.py`, `registry_expansion.py`, `protocol.py` or
`cpu_budget.py` refuses or flags a new `Budget.CPU_DAYS` registration, and no
instrument reads this file for standing constraints. Read by: whoever
registers a spec, by eye; nothing computes it.

**ONE PREMISE CORRECTED AT FIRING, declared rather than glossed (the D25
precedent: a pre-registered remedy can name a condition its own case does not
satisfy).** The decision text says *"no spec is registered in it today"* and
its `blocks:` line repeats it. Read live at firing, the registry carries **six
specs in `Budget.CPU_DAYS`**: `LC.03` (CONCLUDED — the ~190 core-hour run that
motivated the meter), `BO.01` (blocked behind the DP.05 world gate), `PS.04`
(blocked behind LC.03), `BA.03` (PILOT-BLOCKED), `GEN.06`/`GEN.09`
(unimplemented; the `goal-cites-four-specs-that-resolve-to-corpses` row).
None is dispatchable, so the closure forecloses nothing runnable today and no
commitment goes claim-dead by this firing — but the class is NOT empty, and
the honest consequence the option named (retiring or re-scoping the class) is
therefore a real future decision touching six registered ids, which is exactly
why it stays the owner's and is not taken here.

**THE LOSERS, recorded as this page requires.** **(ii) CORE-SECONDS against 4
cores** — the change that raises what the project may take; refused as an
illegal default before being refused on merit. **(iii) A SEPARATE DETACHED
SUB-CEILING** — invents a budget number on the owner's behalf; a ceiling
picked by the organ it constrains is not a constraint (`D26`'s reasoning).
Both remain the owner's to rule at any later date, at no cost; this firing is
not a step toward either.

**Invariants checked at firing:** no `GOAL.md` edit, no threshold moved in
either direction, no control loosened, no new permission taken, nothing
re-run, no certificate staled (no code file touched at all), no spec failed,
no run refused that was permitted yesterday, no commitment claim-dead.
`decisions --check` rc=0 and `--firing-check WORKTREE` run before the commit.

**To reverse:** the owner rules (ii) or (iii) — one line in this file and the
class re-opens; no code was written that would need unwinding.

Evidence: `docs/DECISIONS_NEEDED.md` (the full entry and its arithmetic table,
read live at `8d623b3`); `experiments/cpu_budget.py` (`CPU_DAY_CEILING_S`);
`experiments/rtf.py` (`BUDGET_SECONDS`); `experiments/protocol.py:254`
(`CPU_DAYS = "cpu<48h"`); the registry's six `Budget.CPU_DAYS` ids, read live
at firing.

## D30 — RESOLVED BY ARMED DEFAULT (fired 2026-09-19 ~00:2x UTC, builder): (v) REPORT THE STREAK, GATE NOTHING, RELAX NOTHING. A builder dark streak past 2x cadence is now a standing FOR THE OWNER finding on the Review's page, printed beside the week's GPU-expiry forecast. The pace line, PACE_FLOOR, PACE_CAP and the 90% stop are byte-identical.

**The owner did not rule by 2026-09-18, so the pre-registered default fired —
inside its own SAME-DAY RACE.** `decisions.py` flagged this entry
DEFAULT-ACTION-SAME-DAY: its firing day (2026-09-19) is also the day of the
06:37 sitting that must read the paragraph, so the deadline had an HOUR, not a
date. Fired at ~00:2x, six hours ahead of the sitting.

**The question** (2026-09-15, Review DAILY): the builder was 18 consecutive
hourly slots dark — later 41 — on `week:all models`, a meter
`usage_attribution.py` measured as **75% not this project** (total 37: builder
7, desks 2, outside 28), while the pace line rises at a fixed 0.3869 pts/h
against consumption arriving at ~1.45 pts/h. The gap widens on its own: the
builder does not come back by waiting, and 26.51 free W37 GPU-hours were
expiring with an authorised buyer (`T1.08` §9d, ~1.05 h) and nobody awake to
dispatch it. The blackout ended on 09-17 when the week reset — the default's
own text said plainly that firing after Saturday could not save the quota and
was not offered as though it could.

**What fired.** One paragraph in `scripts/review_prompt.md` Part 2.5 §4 (the
organ-liveness duty): the Review now counts the builder's consecutive dark
slots (PACING or refusal, no iteration) from `/data/jack-logs/ladder.log`
itself, and whenever the streak exceeds 2x the hourly cadence it prints a FOR
THE OWNER finding BESIDE the week's GPU-expiry forecast (free hours left in
the current `%Y-W%U` week of `experiments/gpu_budget.json` and the Saturday
they expire) — a blackout becomes visible on its first day, with its
perishable cost priced in the same sentence. Report only; no code, no meter,
no gate was touched.

**THE LOSERS, recorded as this page requires.** **(i) pace against this
project's OWN attributed spend** — the desk's explicit RECOMMENDATION, and
still refused as a default: it loosens a gate by silence, the exact reasoning
`D26` used twelve hours before the blackout began, and an organ may not vote
itself more budget. It remains the owner's to rule at any later date at no
cost, and the measurement that would inform it (75% outside share) is now in
this record. **(ii) raise PACE_FLOOR 25 → 45** — same defect, strictly
blunter. **(iii) separate the meters** — spends money and acts outside the
repo; no default may do either. **(iv) accept it** — a third GPU-expiry class
(W32 8.82 h, W33 22.11 h, W37 pending at firing) written off as weather.

**The price, restated because the entry priced it, and it is unusually
high:** (v) fixes nothing. It makes the next blackout visible on the day it
starts instead of on day two. That is all it buys, and it was fired knowing
that.

**Invariants checked at firing:** no `GOAL.md` edit, no threshold moved in
either direction, no control loosened, no new permission taken, nothing
re-run, no certificate staled (`review_prompt.md` is in no spec's
`IMPL_DEPS`), no spec failed, no run refused, no commitment claim-dead.
`decisions --check` rc=0 and `--firing-check WORKTREE` run before the commit.

**To reverse:** delete the one paragraph from `scripts/review_prompt.md`; no
code, no threshold, no ledger row, no re-run.

Evidence: `docs/DECISIONS_NEEDED.md` (the full entry, its measurement block
and its options table); `scripts/review_prompt.md` Part 2.5 §4 (the
paragraph); `scripts/lib_usage.sh` (pace line, untouched);
`scripts/usage_attribution.py` (the 75% reading);
`experiments/gpu_budget.json` (`2026-W37`); `docs/DECISIONS_RESOLVED.md` `D26`
(the reasoning (i) was refused under, twice).

## D27 — RESOLVED BY ARMED DEFAULT (fired 2026-09-21 ~06:5x UTC by the OVERSEER, 107th audit, one day late): (i) BUILD THE SCREEN, REPORTING ONLY. Built, measured, and its measured false-positive rate is 19/20. Options (ii) RAISE THE SAMPLE and (iii) CHANGE NOTHING were NOT taken.

**The owner did not rule by 2026-09-20, so the pre-registered default fired.**
Transcribed here by the builder, 2026-09-21 ~11:3x UTC, per the `D22`
precedent (an overseer may fire, the builder transcribes) and the overseer's
`FOR THE BUILDER` item 2. **The firing itself is the overseer's act, not
mine** — `docs/DECISIONS_NEEDED.md` carries it — and I did not re-fire it.
`ladder_prompt.md` ITEM 0 ordered me to fire it at 06:42; the stamp was
already at HEAD when I read it at ~11:1x, so I checked and did not
double-fire, which is what both pages told me to do.

**WHY IT FIRED LATE, and the lateness has a named cause rather than an
excuse.** The builder handed itself this act in its own journal (*"the first
slot on 09-21 fires `D27`'s default if still armed"*) and **the first slot on
2026-09-21 could not exec**, nor could the five before it back to 2026-09-20
07:07 — `scripts/ladder_prompt.md` had crossed `MAX_ARG_STRLEN` and
`ladder_loop.sh` passes it as one argv. That is `D34`. A default that fires
silently late reads identically in the record to one that fired on time, so
it is written down here.

**WHAT WAS BUILT.** `experiments/unread_metrics.py` — a
`metric_recorded_but_unread` scan over every standing PASS row, printed by
`run status` (and `run unread`), **REPORTING-ONLY and UNFLOORED**. It refuses
nothing, stales no certificate, reddens nothing. First reading: **580 metrics
on 62 of 96 decidable certificates; 34 clean; 13 UNDECIDABLE.**

Only the **name half** is built. Pairing a metric to the *constant that gates
it* — the bar-pairing half, where the Review's prototype's 104-of-107 rate
lives — is deliberately NOT built, per the builder's 2026-09-14 evidence
addendum and the overseer's build order. Four exclusions bring the naive
flag rate from **107/109 specs and 3310 metrics** down to 62/96 and 580:

| stage | specs flagged | metrics |
|---|---|---|
| naive "name not in `_check`" | 107 / 109 | 3310 |
| + recorder-minted `_std` siblings dropped | 107 / 109 | 935 |
| + provenance keys, + once-in-source | 86 / 96 | 678 |
| + syntactic summarisation dataflow | 61 / 96 | 576 |

Only the first exclusion is arithmetic rather than taste: `protocol._aggregate`
mints `f"{k}_std"` for every numeric key at >= 2 seeds, so no spec author ever
wrote those names and they do not exist at one seed. A `_check` that subscripts
with an f-string or a variable reads **UNDECIDABLE and is never flagged** —
unknown is not zero.

**THE RATE, WHICH THE FIRING BINDS TO THE BUILD AND WHICH DOES NOT FLATTER
IT.** `D27`'s text: *"Firing this default therefore also owes the rate
measurement, and the counter gets floored or deleted once it exists."* Twenty
flagged `(spec, metric)` pairs were drawn deterministically
(`unread_metrics.sample()`, seed 27 — reproducible by anyone) and adjudicated
by hand against each spec's `_check`:

| pair | verdict | why |
|---|---|---|
| `LC.01 wm-efe/needs_loss_share` | FALSE | folded into `unison_admission_conjunction`, which the gate reads |
| `LC.01 wm-efe/u4` | FALSE | same conjunction |
| `LC.01 wm-latent/deterministic` | FALSE | same conjunction |
| `LC.01 wm-latent/needs_loss_share` | FALSE | same conjunction |
| `LC.02 wm-latent/clears@1.0` | FALSE | read by `committed_ratio(m, arm)`, a HELPER the scan does not follow |
| `LG.00 grounded_knowledge_advantage_s0` | FALSE | the numerator of `sigma_life`, which IS the gated conjunct |
| `LG.02 follow_liar_lastq` | FALSE | component of the gated `div_lastq` |
| `LG.02 null_abs_div` | FALSE | the gate reads `null_abs_div_s0/_s1/_s2` — the substance |
| `ME.1 events` | FALSE | event-log size; context |
| `ME.10 n_train_episodes` | FALSE | fixture size; context |
| `PG.2 bob_rho0.2` | FALSE | per-density component of the gated error |
| `PG.2 expected_rho0.3` | FALSE | an INPUT, not a measurement |
| `PS.02 death_s_max` | FALSE | descriptive statistic; the claim is `probe_r2` |
| `SM.01 peak_over_mean_2m` | FALSE | per-distance detail under the gated falloff discriminant |
| `SO.06 seen_n_rays_changed` | FALSE | feeds `provision_channel_ok`, which the gate reads |
| `SO.09 synth_lo_refused` | FALSE | consumed at `so_09_hands_accountant.py:400` into the gated `hand_share_audited` |
| `T0.26 deg_tf_abs_spread` | FALSE | folded into the gated `p2_degeneracy_isolated` |
| `T0.33 n_foreclosed_now` | FALSE | the accountant's live count; context |
| `UB.9 n_test` | FALSE | fixture size |
| **`T0.06 steps_ok`** | **TRUE** | the literal `5`, written into the metrics dict at `t0_06_dimension_contract.py:62`, never computed and never read |

**19 FALSE / 1 TRUE = a 95% false-positive rate**, and the single survivor is
**not an instance of `D27`'s cited class**. `T0.06 steps_ok` is a *decorative
metric* — a constant recorded as a measurement — not *"the run measured the
quantity that would have indicted it and then did not look at it."* **In a
20-draw this screen found ZERO instances of the defect class it was built
for.**

Two of the three pairs that looked real were read IN SUBSTANCE and this module
could not see it, both for the same mechanical reason: the value travels
through a **subscript** (`rb["refused"]`, `m["synth_lo_refused"]`) rather than
a bare identifier, so the dataflow filter has no shared `ast.Name` to join on.
That is a known, named miss shape, not a surprise to be discovered later.

**WHAT THE INSTRUMENT DOES ABOUT ITS OWN RATE, and this is the part worth
keeping whatever happens to the counter.** `render()` **cannot print the count
without the rate** — it is one sentence, and the fixture's `P8` asserts
exactly that, so a later edit that quietly drops the rate turns the battery
red. The Review's own warning was *"a screen with that false-positive rate is
worse than nothing — it is a red light nobody can act on, which is how
ratchets die."* At 19/20 that warning applies to this screen, so the number is
shipped welded to its own error bar rather than shipped clean.

**WHAT IS ROUTED RATHER THAN DECIDED HERE.** `D27` says the counter gets
*floored or deleted* once it exists. **At 19/20 it cannot be floored**, and
deleting a default the owner's own armed decision ordered built is not the
builder's call on the builder's own evidence. It goes to the Review as
`d27-screen-measures-95-percent-false`. One narrowing is visible in the data
and is recorded rather than taken: the single true positive was found by a
property no filter here uses — **the recorded value is a LITERAL in the
source.** A decorative-metric detector is mechanical and has no false
positives by construction, but it is a DIFFERENT screen from the one `D27`
ordered, and substituting it would be this desk answering a question it was
not asked.

**Invariants checked at firing:** no `GOAL.md` edit, no threshold moved in
either direction, no control loosened, no seed count changed, no new
permission taken, no spec failed, no run refused, no commitment claim-dead.
No ledger row was written by this work. `experiments/run.py` IS in `T0.36`'s
`IMPL_DEPS`, so its certificate is re-bought in the same commit — the bill is
disclosed, not avoided.

**To reverse:** delete `experiments/unread_metrics.py` and its two call sites
in `experiments/run.py` (`print_unread_metrics_block`, the `unread` entry in
`READ_ONLY_COMMANDS`). No threshold, no ledger row, no re-run. The owner's
option (ii) — raise Part 2's sample — remains theirs to rule at any time and
is unaffected by this default having fired.

Evidence: `docs/DECISIONS_NEEDED.md` `## D27 — RESOLVED BY ARMED DEFAULT` (the
overseer's firing, which is the authority for the act); the superseded entry
below it (the Review's 2026-09-13 fork and its 104-of-107 prototype); the
builder's 2026-09-14 EVIDENCE ADDENDUM (the ledger-only/bar-pairing split that
set the build order); `experiments/unread_metrics.py` (the four exclusions, the
fixture, and `sample()` — the draw is reproducible);
`docs/OVERSIGHT.md` `FOR THE BUILDER` item 2 (the order).

---

## D28 — RESOLVED BY ARMED DEFAULT (fired 2026-09-22 ~07:0x UTC by the REVIEW, DAILY, on the day it came due): **(a) OVERDUE FIRST** — the Review's daily sitting spends its first act disposing the OVERDUE class, ACT / DECLINE / re-date with a reason, before routing anything new.

**The owner did not rule by 2026-09-21, so the pre-registered default fired.**
The entry was reclassified `goal` → `conduct` by the overseer on 2026-09-21
(107th audit) with `decide_by` deliberately untouched, so the owner's window
closed on schedule and the default came due today. Fired by the Review — the
desk the default binds — because `decisions --check` classes it
`[CONDUCT-DESK] D28 — desk-executable, not the owner's: execute it, report it,
do not ask.`

**WHAT WAS BUILT.** One bullet at the head of `scripts/review_prompt.md`'s
DAILY block, matching the armed text clause for clause: first act, OVERDUE
class, three verbs, before routing anything new; and the four things the armed
text promised not to do — no row deleted, no `DUE:` dropped, no row relabelled
HELD to stop a clock, no disposition chosen on anyone's behalf — written into
the bullet so a later sitting cannot quietly widen it. Reversal: delete the
bullet. No code, no threshold, no ledger row, no re-run.

**Options NOT taken, and each for the reason the armed text gave:** (i) more
Review wall clock and (iii) a second consumer organ both spend the shared
all-models meter by silence, and a default may not commit a budget that is
already failing; (iv) builder-drains-its-own-queue reassigns design authority
`D22` placed with the Review, and a default may not widen what an organ may
do; (ii) fewer routings would convert a visible backlog into an invisible one.

**THE PRICE, PAID IN THE SAME SITTING AND NAMED RATHER THAN BURIED.** The
first application of this default disposed all 21 OVERDUE rows and took
`review_queue_violations` **21 → 0 by the desk's own act**. That is not
capacity and must never be read as capacity: arrivals ran 1.57/cycle against
1.00 disposals over the trailing week, 54 rows are live, the drain is still
UNBOUNDED, and `D28`'s underlying capacity fork is still unanswered — this
default was only ever the cheapest of five arms and the one that spends
nothing. The counter that was `D28`'s own evidence is now silent, so the
evidence moves here: **whoever re-opens this question should read the queue's
THROUGHPUT block, not its violation count.**

## D29 — RESOLVED BY ARMED DEFAULT (fired 2026-09-23 ~12:5x UTC by the OVERSEER, 109th audit, one day late — the four intervening overseer slots were `STOPPED at 92–100% weekly usage`, so this was the first audit that could fire it): (iii) RECORD THE DEBT, CHANGE NO MARKING. Options (i) BUILD THE DIAGNOSTIC, (ii) CORRECT `LEARNING_CORE.md` §5.4 and (iv) DOWNGRADE THE SEAT'S MARKING were NOT taken; (iv) remains the owner's to rule at any time.

**WHAT THE FIRING DOES, IN FULL, AND NOTHING ELSE.** The Learning-core cell of
`docs/CHAMPIONS.md` gains a second stated caveat, in the same idiom as the
single-arm caveat and the `VENUE-UNAFFORDABLE` label already on its face:

> *`LEARNING_CORE.md` §5.4 declares a MANDATORY collapse diagnostic — effective
> rank and per-dimension latent variance every 1,000 decisions, a rank below a
> pre-registered floor being `Status.VOID` for A4. It was never implemented.
> `LC.03`'s committed row records 50 `wm-latent` metrics across five arms and not
> one is a rank or a per-dimension variance. This seat was awarded without the
> guard its own governing document calls mandatory against the failure mode that
> document calls silent, and the guard can never now be run on the evidence that
> seated it: the trained A4 weights are not on disk, `LC.03` v2 is
> VOID-FORECLOSED, and `LC.07` is VENUE-UNAFFORDABLE at ~526 wall-hours.*

`LEARNING_CORE.md` §5.4 stands verbatim, unfulfilled and visible. The `HELD: BY
VERDICT` marking is untouched. No threshold moves in either direction. No spec
is failed, no run refused, no certificate staled, no GPU spent. `champions
--check`'s UNVERIFIED-VERDICTS count stays 2/2 — verified before and after the
transcription — because a default that fires by silence may not pay itself a
greener number.

**REVERSAL:** delete the caveat sentence from `CHAMPIONS.md`'s Learning-core
cell. No code, no threshold, no ledger row, no re-run.

**THE LATENESS AND THE PREMISE DEFECT, carried rather than glossed.** The
`decide_by` of 2026-09-22 was placed deliberately AFTER the desk deliverable
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` (DUE
09-18) so the owner would rule with that work in hand; the deliverable slipped
to 2026-09-25 and nothing re-checked the placement, so the default fired on a
premise its own author called insufficient — the third recorded instance of
the "premise dies under a deadline" shape (D20; `me1-similarity-floor-never-
abstains`). The overseer fired anyway, per the charter: a deadline that moves
when it is reached is the deadlock it replaced; a premise defect argues for
firing the WEAKEST option, which (iii) is by construction.

**TRANSCRIPTION:** builder, 2026-09-23, this commit — the overseer fires and
records (firing block: last section of `docs/DECISIONS_NEEDED.md`); the
overseer may not edit `CHAMPIONS.md` or `DECISIONS_RESOLVED.md`.

## D19 — RESOLVED BY THE OWNER 2026-09-17: "yes may download anything to /data"

Asked (2026-09-03): may the builder fetch speech corpora to `/data` for
`HR.1`–`HR.4`, disk that is not this project's to take. The armed default was
NO FETCH. The owner answered a broader question than the one filed — granted:
**anything, to `/data`** — and the default is superseded, unfired. The corpora
were fetched and `HR.1` has since recorded three attempts and two measured
FAILs against them; the builder's self-imposed floor (refuse a fetch that
would take `/data` below 15 GB free, state the size first, delete
intermediates the same iteration) and the tenant-safety constraint stand
unchanged.

**Filing note (builder, 2026-09-24, 113th audit):** the ruling was recorded on
2026-09-17 in `DECISIONS_NEEDED.md` at the tail of the neighbouring `## D35`
section with no `## D19` header of its own, so `decisions --check` kept
printing `D19 … OVERDUE — DEFAULT IS DUE TO FIRE` for ten days over an
answered decision — an order that, if obeyed, would have reversed the ruling,
re-blocked three specs and retroactively de-venued two honest FAILs. This
entry and the matching header in `DECISIONS_NEEDED.md` are the repair; the
superseded `## D19` entry and its `DECIDE:` block are retained in place per
that file's convention.

## D32 — RESOLVED BY ARMED DEFAULT (fired 2026-09-25 ~00:4x UTC by the OVERSEER, 115th audit, on the first legal day; transcription completed by the builder 2026-09-25 ~01:1x): (ii) SEE IT AND SAY IT. The lane declares itself and the code already did — the firing is a RECORDING act and orders no work. Options (i) UNSCOPED, (iii) SCOPED and (iv) DECLINE were NOT taken and remain the owner's to rule at any time.

**Transcription per the `D13` rule and the `D22`/`D29` precedent: the overseer
fires and records; the builder transcribes. The firing block appended to
`docs/DECISIONS_NEEDED.md` (at its `## D32 — RESOLVED BY ARMED DEFAULT` header)
is the firing record and is quoted here in full. Until this entry landed, that
block was the record; this entry completes it and changes nothing in it.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no ratchet counter moved. `_lane_verdict`'s
refuse/permit line untouched — a direct `setsid` launch is refused exactly as
before, a wrapped detached launch is permitted exactly as before. **The scope
question `D32` actually asks — whether `D20`'s closure covers the wrapped
detached lane at every cost class or only at `cpu<48h` — is NOT answered by
this firing and stays on the owner's desk.**

> **THE OWNER DID NOT RULE BY 2026-09-24, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(ii) SEE IT AND SAY IT**. Options (i) UNSCOPED, (iii) SCOPED and
> (iv) DECLINE were **NOT** taken.
>
> **AND THE ONE THING THIS FIRING MUST SAY BEFORE ANYTHING ELSE: THE DEFAULT IS
> ALREADY IMPLEMENTED ON DISK, SO THIS FIRING ORDERS NO WORK.** I verified it
> rather than inferring it from the journal:
>
> ```
> scripts/launch_detached.sh:44   # JACK_DETACHED_LANE: THE LANE DECLARES ITSELF (105th audit item 1)
> scripts/launch_detached.sh:50   # whether D20's closure covers this lane is D32 (the owner's)
> scripts/launch_detached.sh:52   setsid ... env -u JACK_ITER_DEADLINE JACK_DETACHED_LANE="launch_detached.sh $LOG" ...
> experiments/run.py:4574         DETACHED_LANE_ENV = "JACK_DETACHED_LANE"
> experiments/run.py:4590         def _lane_verdict(settle: bool = True) -> tuple:
> experiments/run.py:4656         "this lane is D32's question (the\nowner's); until it rules the ..."
> ```
>
> The builder landed it on 2026-09-20 under the 105th audit's `FOR THE BUILDER`
> item 1, with four new cases in `scripts/test_lane_guard.sh` (declared-lane
> notice present; `lane: launchable` pinned; `sid != pid` proving the WRAPPED
> shape was exercised rather than a hand-rolled `setsid`). **This entry's own
> closing paragraph anticipated exactly that** — *"The one thing that could not
> wait ... is handled by the 105th audit's `FOR THE BUILDER` item 1, which orders
> the detection and the fixture case BEFORE any question of refusal"* — so the
> firing is a RECORDING act, not a build order.
>
> **What the firing does, in full, and nothing else.** It closes the entry on the
> option the code already satisfies, and it states in the open what that option
> bought and what it did not. The refuse/permit line is **untouched in both
> directions**: a direct `setsid` launch is refused exactly as it was yesterday, a
> wrapped detached launch is permitted exactly as it was yesterday. No threshold
> moves. `GOAL.md` is not touched. No spec is failed, no run refused, no
> certificate staled, no GPU spent, no budget committed, and **no ratchet counter
> moves in either direction** — a default that fires by silence may not pay itself
> a greener number.
>
> **WHAT REMAINS OPEN, AND IT IS THE PART THAT MATTERED.** (ii) bought VISIBILITY
> and nothing else, which the entry said in advance. **The scope question — does
> `D20`'s closure cover the wrapped detached lane at every cost class, or only at
> `cpu<48h`? — is NOT answered by this firing and is still yours.** The lane
> stays usable. If nobody reads the mark, the next registered run goes through it
> exactly as the one that prompted this entry did. Options (i) and (iii) are still
> on the table and neither is mine to take: (i) NARROWS what is permitted today
> and (iii) WIDENS what the code permits today, and a default may do neither.
>
> **HOW TO REVERSE IT.** Delete the `DETACHED_LANE_ENV` branch from
> `_lane_verdict` and its fixture case in `scripts/test_lane_guard.sh`, and drop
> the `JACK_DETACHED_LANE=` assignment from `scripts/launch_detached.sh:52`. No
> threshold, no ledger row, no re-run.
>
> **THE PREMISE, RE-CHECKED BEFORE FIRING RATHER THAN AFTER.** `D29`'s firing
> record named "the premise dies under a deadline" as a structural shape, three
> instances deep. `D32`'s premise is *"the control built to enforce the ruling
> reported `launchable`"*. That is **still true today** and I re-derived it:
> `_lane_verdict` prints a lane-specific NOTICE and still returns the wrapped lane
> as permitted, by design and by this entry's own instruction. The premise did not
> die; the entry simply got its cheapest option delivered four days early.
>
> **Evidence:** `experiments/decisions.py --check` at 2026-09-25 00:4x prints
> `D32` under **`OVERDUE — DEFAULT IS DUE TO FIRE`**; `decide_by: 2026-09-24`;
> the file paths and line numbers quoted above, all read at this HEAD; the
> 2026-09-20 01:0x builder journal entry recording the landing and the 22-case
> green fixture run.

## D34 — RESOLVED BY ARMED DEFAULT (fired 2026-09-25 ~00:4x UTC by the OVERSEER, 115th audit, on the first legal day; transcription completed by the builder 2026-09-25 ~01:1x): (iii) BOTH, IN THAT ORDER — the trim was already delivered as desk conduct (90935 B against the 131072 cliff), and the stdin change was ordered onto the builder behind an in-slot verification. THE CODE HALF IS NOW EXECUTED: verification passed and the change landed at `f06afd1`. Options (i)-alone and (ii)-alone were NOT taken and remain the owner's to rule at any time.

**Transcription per the `D13` rule and the `D22`/`D29` precedent, quoting the
firing block the overseer appended to `docs/DECISIONS_NEEDED.md` (at its
`## D34 — RESOLVED BY ARMED DEFAULT` header). Unlike `D32`, this firing carried
one code order, conditional on an in-slot measurement. Both halves are now
done:**

- **The verification, run first as the default requires (builder, 2026-09-25
  01:1x, this slot):** one throwaway prompt fed to `claude -p` on stdin on this
  harness returned a non-empty response (`STDIN-OK`), rc=0. The precondition
  held, so (i) was taken.
- **The change (`f06afd1`, same slot):** `run_claude` in
  `scripts/ladder_loop.sh` now pipes `$PROMPT` into `claude -p` via the
  bash-builtin `printf`, so no `execve` ever sees the steering page's text and
  `MAX_ARG_STRLEN=131072` can never again kill a slot at launch. `bash -n`
  clean; the only remaining `"$PROMPT"` reference is the builtin `printf`.

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved,
no control loosened, no new permission created, nothing re-run, no certificate
staled. The firing is monotone on the thing at issue: the builder can only go
from unable-to-start to able-to-start. **The price the firing states remains
open and is nobody's to close by silence: the steering page keeps growing under
either repair, and whether a 90 KB hourly steering page is a sensible design is
a real question that is NOT in `D34` and is NOT closed by this firing.**

> **THE OWNER DID NOT RULE BY 2026-09-24, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(iii) BOTH, IN THAT ORDER**. Options (i) STDIN ALONE and (ii) TRIM
> ALONE were **NOT** taken.
>
> **HALF OF IT IS ALREADY DELIVERED AND THE OTHER HALF IS NOT. Both halves
> verified at this HEAD, not taken from a page.**
>
> ```
> (ii) the trim   DONE   scripts/ladder_prompt.md = 90935 bytes
>                        (131072 exec cliff; 125000 self-imposed ceiling)
> (i)  stdin      NOT DONE
>                 scripts/ladder_loop.sh:270   PROMPT=$(cat "$REPO/scripts/ladder_prompt.md")
>                 scripts/ladder_loop.sh:282   timeout 50m claude -p "$PROMPT" \
> ```
>
> So `(ii)` stands as executed desk conduct over the Review's own page, exactly as
> the default provides, and `(i)` is what this firing hands forward.
>
> **WHAT IS ORDERED, WITH THE DEFAULT'S OWN PRECONDITION CARRIED VERBATIM.** The
> builder, in one live slot, **first** verifies in that same slot that `claude -p`
> reads stdin on this harness — by launching one throwaway prompt through stdin
> and confirming a non-empty response — and **only then** changes
> `ladder_loop.sh:282` to feed the prompt on stdin instead of argv. **If the
> verification fails, `(i)` is NOT taken**, the change is not made, and the entry
> returns to the owner with the measurement attached. That conditional is the
> default's, not mine, and it exists because an unverified stdin change can fail
> SILENTLY with an empty prompt — a default may not replace a loud failure with a
> quiet one.
>
> **THE COST LINE IN THIS ENTRY IS STALE IN THE BUILDER'S FAVOUR, AND THE FIRING
> SAYS SO RATHER THAN INHERITING IT.** `D34` prices its urgency on *"24 hours, 0
> iterations, 0 ledger events"* and on the steering page growing *"~3976
> bytes/day, monotone"* with *"about ELEVEN DAYS"* of headroom. Measured today:
> the builder has run **45 consecutive `rc=0` iterations** — every slot since the
> last non-zero exit at 2026-09-21T06:07 (`rc=126`), 25 of them in the last 24
> hours, zero dark — and `run status` reads the page at
> **90935 B, +1035 B/day over 27 commits — 39 days to the exec cliff, 33 to the
> self-imposed ceiling.** The growth rate is about a quarter of what the entry
> assumed. **This does not change the firing**: a repair with a computable expiry
> is still a repair with an expiry, `(iii)` is still the right default, and the
> argv path is still the mechanism that took the loop out for 23 hours. It is
> recorded because a default that fires on a premise nobody re-checked is the
> exact defect three prior firings have now named as structural. The premise is
> re-checked, in writing, before firing, and the correct reading is that the
> runway is five weeks rather than eleven days.
>
> **What the firing does, in full, and nothing else.** It picks only
> already-permitted actions: a desk editing its own steering page is what the
> Review does every morning, and repairing the launcher so it can `exec` restores
> a capability rather than widening one. It moves NO threshold in either
> direction, edits no `GOAL.md` text, weakens no gate, refuses no run that is
> permitted today, permits no run that is refused today, spends no GPU, commits no
> budget, fails no spec, and stales no certificate. It is MONOTONE on the thing at
> issue: the builder can only go from unable-to-start to able-to-start.
>
> **THE PRICE, STATED RATHER THAN BURIED — and it is the entry's own words.**
> `(iii)` leaves the growth itself unaddressed. The steering page keeps growing
> under either repair, and **neither option asks the harder question of whether a
> 90 KB steering page is a sensible thing to hand a builder every hour.** That
> question is real, it is NOT in this entry, and it is not closed by this firing.
>
> **HOW TO REVERSE IT.** `git revert` the one-line launcher change and re-grow the
> page. No threshold, no ledger row, no re-run.
>
> **TRANSCRIPTION IS OWED BY THE BUILDER**, per `D13`/`D22`/`D29`: open the
> `DECISIONS_RESOLVED.md` entry quoting this block. Until that lands, **THIS BLOCK
> IS THE FIRING RECORD.** Unlike `D32`, this one DOES carry a code order, and it
> is the conditional stdin change above.
>
> **Evidence:** `experiments/decisions.py --check` at 2026-09-25 00:4x prints
> `D34` under **`OVERDUE — DEFAULT IS DUE TO FIRE`**; `decide_by: 2026-09-24`;
> `wc -c scripts/ladder_prompt.md` = 90935; `scripts/ladder_loop.sh:270,282` as
> quoted; `/data/jack-logs/ladder.log` — 45 `iteration end rc=0` lines after
> 2026-09-21T06:07:22, 45 of 45.

## LG.13 — WINNER — meaning-mass

> **STALENESS RE-BUY, not a second verdict (annotated 2026-09-26 by the
> builder, 119th audit FTB 4).** This record is attempt 2's deterministic
> re-run of the earlier `## LG.13 — WINNER` record above — appended
> 2026-09-25 22:29 by the re-buy owed after the `protocol.py` verdict-gate
> edit staled LG.13's certificate. Byte-identical to attempt 1's because
> every arm is a deterministic function of the frozen verdict table. A
> reader counting bakeoff verdicts should count ONE.

meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | cost |
|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | 1.0 |

## D31 — RESOLVED BY ARMED DEFAULT (fired 2026-09-26 ~00:5x UTC by the OVERSEER, 119th audit, on the first legal day; transcription completed by the builder 2026-09-26 ~01:2x): (i) MARK BUT DO NOT CAP. The default was already implemented on disk on 2026-09-18 (`2bfa84f`), so the firing orders no work and changes no behaviour — it settles the RECORD. Options (ii) GIVE COLAB A CEILING and (iii) DECLINE were NOT taken and remain the owner's to rule at any time.

**Transcription per the `D13` rule and the `D32`/`D34` precedent (`6aaed9b`):
the overseer fires and records; the builder transcribes. The firing block
appended to `docs/DECISIONS_NEEDED.md` (at its `## D31 — RESOLVED BY ARMED
DEFAULT` header) is the firing record and is quoted here in full. Until this
entry landed, that block was the record; this entry completes it and changes
nothing in it.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no ratchet counter moved. **The two things the
firing itself says must survive transcription: the mark shipped `2026-09-18`
(`gpu.py:133` `PER_JOB_OVERRUN_MARGIN = 0.25`, `gpu.py:540` the comparison) so
this firing orders NO work — and option (ii) is still the only option that
changes what the colab lane may spend, and it is still the owner's.**
`remaining('colab')` still returns infinity; no dispatch is refused today that
was permitted yesterday. Reversal: delete one `if` from `experiments/gpu.py`
(line 540).

> **THE OWNER DID NOT RULE BY 2026-09-25, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(i) MARK BUT DO NOT CAP**. Options **(ii) GIVE COLAB A CEILING** and
> **(iii) DECLINE** were **NOT** taken.
>
> **AND THE FIRST THING THIS FIRING MUST SAY: THE DEFAULT WAS ALREADY IMPLEMENTED
> ON DISK EIGHT DAYS AGO, SO THIS FIRING ORDERS NO WORK AND CHANGES NO BEHAVIOUR.**
> Verified at source this audit rather than inherited from the 100th audit's
> addendum or from the journal:
>
> ```
> experiments/gpu.py:133   PER_JOB_OVERRUN_MARGIN = 0.25
> experiments/gpu.py:540   if est_hours > 0 and billed_h > est_hours * (1.0 + PER_JOB_OVERRUN_MARGIN):
> experiments/gpu_budget.json  "overruns": []
> ```
>
> The mark shipped as `2bfa84f` on 2026-09-18 under the 99th audit's `FOR THE
> BUILDER` item 4, as instrument work, and the 100th audit's EVIDENCE ADDENDUM
> above put that on the record without touching `decide_by`. `overruns` reads `[]`
> because no GPU job has been dispatched since the mark landed — the two colab
> overruns it was built from (147% and 155% of declared, 2026-09-14) predate it and
> were never back-filled, which is the addendum's own stated behaviour and not a
> silent failure.
>
> **So what this firing actually settles, stated narrowly.** It closes the QUESTION,
> not a code gap: the entry has presented (i) as one unexecuted option among three
> since 09-15, and an owner reading it cold would have been choosing between an
> option already in effect and two that are not. After this firing the record says
> what the repository does. `remaining('colab')` still returns infinity, the colab
> lane still has no ceiling, and **no dispatch is refused today that was permitted
> yesterday.**
>
> **The price, restated because a firing may not quietly drop it.** (i) buys
> VISIBILITY and nothing else — the entry's own words, and now the measured state:
> *"A marked overrun still spent the hour, and if nobody reads the mark the lane is
> exactly as uncapped tomorrow as it is today."* The realised loss this entry was
> opened on — 2.11 colab-hours across two retrievals that returned nothing, inside a
> probe that spent 223% of its authorised budget with no number turning red — is
> written off by this firing and is not recovered by it. **Option (ii) is still the
> only option that changes what the lane may spend, and it is still yours.** A
> default may record a debt; it may not invent a budget number on the owner's
> behalf (`D26`'s reasoning, cited by this entry when it armed).
>
> **Reversal, unchanged:** delete one `if` from `experiments/gpu.py` (line 540).
> No threshold moves, no ledger row is touched, no re-run is owed, no certificate
> stales.
>
> **Filing debt this firing creates, owed by the builder's next live slot:** the
> `## D31 — RESOLVED BY ARMED DEFAULT` transcription onto `docs/DECISIONS_RESOLVED.md`,
> per the `D13` rule that the overseer stays inside its own file set and the
> `D32`/`D34` precedent (`6aaed9b`, 2026-09-25 01:12). Until that lands,
> `decisions.py`'s second identification channel (`RECORD_PAGE`/`RECORD_MARKER`)
> cannot see this firing.
