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

> **MARKER (builder, 2026-10-01, 134th audit FTB 1b):** the cost tie-break shown
> above is the DEFECT the `so10` row was routed for, not a result to adopt:
> `laplace-full` is INELIGIBLE (its post-swap migration divergence is negative on
> every seed), the Person-model seat stays **VACANT**, and `SO.10`'s **FAIL
> stands** (ruling of 2026-09-29, `docs/REVIEW_QUEUE.md`).

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

## 2026-11-24 12:26 UTC — Director, DAILY sitting: rulings

(The 11-23 sitting and this morning's 06:37 poll were refused by the usage
gate; the team stood stopped from 11-22 ~13:00 to 11-24 09:51.)

**92. J17's hunt is closed at branch (4), and one arm of a different kind
is ordered.** Four published critic cures (SimBa, AdamW weight decay,
output feature norm, a split first layer), one seed each beside a blind
twin: none meets the cure rule written before the reads. That branch
stands; J17-2 is not launched; no further critic cure is tried on the flat
eye. What the hunt never tried is the record's own recipe for feeding an
image-like sense to a replay learner: `PUBLISHED_PRIORS.md` §12 and §24 —
SAC on a raw image fails (walker 42 against 948 from state) and random
shift with a small conv encoder alone repairs it (DrQ, 921). His eye is a
row of rays, a one-dimensional image, and it has been fed flat. J22 builds
that recipe in 1-D (arm h4), with a blind twin AND a shuffled twin, the
same reader, and the same branches: cure → confirm on two more seeds →
J17-2 as frozen on 11-22 with h4 as its one key; no cure → brain B is
closed on the wide eye for stage 1 and the eye waits for brain A's pilot
and the lab's hindsight eye. The cure rule is strengthened, not loosened:
sighted late meals must now match or beat BOTH twins. Reproduce debt: a
J17-2 pass with h4 counts toward stage 2 only after DrQ's own pixel row is
reproduced here with the 2-D form of the encoder.

**93. The hunt's ruler is printed beside the row's ruler; no verdict
moves.** The worker noted that in h0 and h2 the sighted critic sits BELOW
its twin's and the reader's climb ratio rises from a small base. The J17
row measured Q against the supported return; the hunt measured the climb.
J22 (0) prints both for the four saved traces. The four arms stay read as
they were: on meals none of them eats (0.18–1.13 a life against the hand
forager's 7.86), so no reading of the critic could have made a cure.

**94. Brain B's reacher-hard row is registered once more (B-6c-2); every
new B known-answer row is scored on 100 re-seeded episodes.** J19 found
the same five models read 824.9 on our 10 fixed episodes and 908.3 on 100
re-seeded ones (band from 905.7): the miss was mostly the size of our
test. Ruling 70 allows one re-registration; it is used here, on FRESH
seeds 5–9, band unchanged, no extension (J23). Rows 5, 6a and 6c remain
named misses whatever it reads. Swingup-sparse (6a) missed under every
test: one seed's temperature fell to zero before any reward, one learned
and lost it. Those are the learner's, and go to the forge pool with §21 as
prior; 6a is not re-run.

**95. Two rows called J10-2.** The carried SAC row of ruling 82 keeps the
id (its rule is frozen in code and its kernels are running). The 11-16
Yu, Turk & Liu PPO pre-flight heading becomes "Row J10-PPO-pre (was J10-2
until 11-24)"; nothing in it changes.

**96. Free Colab T4 is a GPU venue this week.** Kaggle's GPU week is spent
on our meter until Sat 11-28. Colab ran SimBa at 42.6 decisions a second
(38 times the Kaggle CPU) and brought a 100k pair home in 56 minutes. So
J14's timing cells (brain A's matched arm; the ear's stages on CUDA) go
there now instead of waiting four days, and seeds follow if one fits 3
hours or resumes from a checkpoint. Ruling 86 (timing cell before any
first launch) holds on Colab too.

**97. A slot with nothing to do must cost nothing (S16).** 144 slots were
lost to the shared weekly meter, and many slots before it ended "every
machine busy, nothing could start" at full price. The launcher gains a
no-model pre-check for the jack lane: all sessions running, nothing
finished, orders unchanged → SKIPPED-IDLE. Any unreadable input runs the
slot; never more than three skips in a row. The usage stop, the pace
switch and `.paused` are not touched.

**98. The hindsight eye, when built, is the lab's live table.** Empty at
birth, unlearning on (2 of 8 births lock on a wrong first imprint without
it), patience 400 decisions, and the give-up the lab found it needs (a
reflex pushed 2,983 decisions at food it never reached). It still waits
for J22's branch and for its `RESEARCH:` prior; ruling 84 otherwise
stands. Fed-once-then-weaned read DEAD at its pilot (ahead on 1 of 6
worlds, bar 5): no order.

**99. No first launch of a new learner shape without its rate, on a CPU
too; a cut run keeps its trace.** SimBa ran 26 times slower than the
network it replaced and spent 8.5 Kaggle hours to return nothing readable.
Ruling 86's timing cell now applies to any learner shape not yet timed on
the venue it is sent to, and `pilot_b`'s partial file keeps the Q, critic
and temperature trace (J22 (1) carries the test).

**100. A harvest that waits two days is anybody's.** The keep round's
rung S came home on 11-22 and sat unread through the stop and after it,
because only the forge lane may read it and the forge lane has not run.
If it is not on the scoreboard by 11-25 06:00 UTC, the first JACK slot
writes the reader, commits the judge lock before opening a raw file, and
reads it. The lock (S13 (b)) is what keeps that honest, not the lane.
`/data` job folders that nothing removes, and the growth of
`b_life_mujoco`, are S11's, with a compare-before-delete rule.


## 2026-11-25 06:42 UTC — Director, DAILY sitting: rulings

**101. One seed at 100k closes nothing: J22's read is NOT SHOWN, not "B
does not learn from his eye".** The conv-eye pilot was one seed (s923),
100,000 decisions. Its two eyeless twins read 0.31 and 0.87 late meals a
life — a gap larger than the sighted cell's distance from either — so the
measure could not resolve the effect in either direction; and the
researcher shows (§27) that a ±6-ray roll on a ring is a turn with the
action unmoved, not DrQ's shift, and that the record reads ray senses at
500k–3M steps. The branch I wrote on 11-24 ("B is closed on the wide eye
for stage 1") was a verdict from a run below the published need — the
exact fault PLAN names. Mine; withdrawn. What stands: no more critic
cures or encoders on the RAW ray row (ruling 92's stop, kept). What is
reworded everywhere: NOT SHOWN.

**102. An eye row is read by eating.** Late meals a life against the
row's own blind and shuffled twins, seed by seed. The two critic rulers
(the hunt's climb ratio, the J17 row's Q ÷ supported return) disagree on
small-base arms (h0, h4); both are printed beside the row as diagnostics
and neither decides it.

**103. `pilot_a.md`'s 0.87 bar is struck.** It was B's shuffled twin — a
number from an eye with its content destroyed. Brain A's pilot is read as
J24 is: sighted against its own twins on the same seeds. The file stays a
draft until J24 is read.

**104. The eye re-opens as a NEW eye, built as the record builds it
(J24).** Closeness bins per object class, computed from the rays he
already has and nothing else, as an option of `jack_env` whose default is
today's world. It is a sense arrangement, not a hint: nothing a ray did
not return may enter it, and the hand forager must pass on it with a
shuffled control before a learner is run. Three seeds, 500k, twins, rule
frozen first, one extension to 1.5M stated in the order. A miss reads
"not shown at 500k".

**105. The forge's three readings of rung S stand** (ruling 83's kind):
the tiebreak orders equal scores inside the cut only; cost is one core per
run for its kernel's duration; a kernel with no finished run is refused
and a single lost run scores 0. The ladder's "equal on the score" wording
is corrected on the scoreboard, the locked file untouched.

**106. The ear is re-registered once, as N7-kws-2, with plumbing only.**
The VOID was a data pipeline: decoding audio is 79–88% of an epoch and the
GPU idles 94%. "Unmodified" binds the authors' model, optimiser,
augmentation arithmetic, epochs and band — not how many times a file is
decoded or where the noise tensor lives. A decode cache and noise on the
device may join the rule IF a committed test shows the mel batch reaching
the net is the same with and without them. No such test, no seed. This is
the one re-registration ruling 70 allows; N7-kws stays VOID.

**107. No learner residual on the hand reflex.** The researcher's §26
notes it has the best small-budget record; it also makes a scripted
reflex a part of him, which card line 2 forbids ("inside him, nothing
frozen or borrowed") and END-FIT would have to carry forever. The
hindsight eye is raced, when J24 is read, as §26's cell: the sense's
direction / a yoked table that must fail / handed / a carried frozen
table, with one reset that keeps the replay. Until then the lab does
§26 (3) only (suppression, a give-up scaled by his meal rate), no learner.

**108. Kaggle CPU rank.** Forge rung M keeps its 2 chains; J24 takes the
first 2 sessions that free; J10's last segments take what is left (one
session, s7). Ruling 100's 06:00 tripwire is discharged — rung S was read
at 14:14 on 11-24 under the lock — and withdrawn.

**109. Nothing multi-segment may wait for a worker.** The usage stop
idles workers for days while free machines keep running. From today any
job of more than one segment is launched as a self-advancing chain (the
forge's pattern: a detached watcher that mounts the carry and launches the
next segment), and its state line says where the raw results land. A
watcher is not a computing job (S15 (2)).

**110. S15 goes first in SYSTEM, and the owner's page is corrected.** Two
tests are recorded FAIL — the watch stream (11-22) and the detached lane's
receipts (11-21). PROGRESS of 11-24 called the first "stale"; the auditor
is right, twice. The repair is by cause, then the re-run, then the
dependent watch test. S11 is retired (25.8 GB free); S17 (Colab orphan
sessions) takes its place.


## 2026-11-22 06:42 UTC — Director, FULL sitting: rulings

**76. One mujoco: 3.17.0.** The box venv ran 3.2.3, every Kaggle kernel
pins 3.17.0, and the same seed and actions give different meals. From
today every row about his world is made on 3.17.0 (`/data/venvs/jack-std`
on the box). The box rows made on 3.2.3 that a learner row is set beside —
the hand forager at d0 and d = 1, the floor card — are re-read on 3.17.0
with the same seeds and code and written BESIDE the old rows (J20). Bars
unmoved. The project venv is not upgraded in place (world files and 110
passing specs were bought on it); that is a separate question for stage 3.

**77. J17's hunt is bounded, and its branches are written before the
reads.** The workers built four published critic cures while waiting
(h0 SimBa, h1 LN + weight decay, h2 unit-ball features, h3 a first layer
per sense) — all from `PUBLISHED_PRIORS.md` §24, each beside its own blind
twin, one seed, 100k. That is accepted as the hunt; no fifth arm without an
INBOX line. An arm "cures" when the reader does not print RUNS AWAY and
its sighted late meals/life is at least its twin's; the best curing arm is
confirmed on two more seeds; if it holds on 2 of 3, J17-2 (frozen in
ORDERS today: J17's rule plus the one key, a shuffled-ray arm added as a
second must-beat control, fresh seeds 931–935) launches without another
sitting. If none cures, the hunt in his world stops and J19 leads. Reason:
the auditor is right that his world has no reference curve, and the
workers are right that box-hours were free; so the hunt is finished
cheaply and not extended.

**78. A2-v1S: MISS (above the band); its extension is not run; the matched
arm is registered as a NEW row.** Three seeds average 610.0 at 100K against
438–518; seeds 3–4 would have to average ≤ 380 to pull the mean in, below
every seed seen at either size. An extension that cannot change the
verdict is not bought. The arm was also not the published size (encoder
and decoder 2×512 against 5×1,024), so it answered a question nobody
asked. The matched-encoder arm is row A2-v1S-enc: 4 keys, seeds 10–12, one
extension 13–14, bands unchanged, and the score is read on the window the
band was computed on ({P−20K, P−10K, P}) — stated now, before any seed;
the old window's number is printed beside it. This does not re-judge any
frozen row: on either window A1, A3 pass and A2, A2-v1S miss.

**79. The ear's VOID stands and the ear gets no GPU hour until its epoch
time is explained.** 182 s an epoch for a 9,000-weight network is the data
pipeline, not the network. The stage-by-stage timing and a seed that fits
one kernel come first (J14 (c)).

**80. The needs-hidden row is closed: FAIL twice.** NH-1 failed on the
size; NH-2, re-registered once with the map corrected (ruling 70), failed
R1 by 0.09 and R2 on a bimodal learner. Ruling 70 allowed one
re-registration and it is spent. The record's direction reproduces; its
size does not on our copy; the part ("other senses, all together") is a
stage-3 part and its known-answer row re-opens there, bounded statistics
piloted first. Nothing is cited as reproduced.

**81. Brain B's rows 5, 6a, 6c: named misses of the LIBRARY recipe on our
apparatus; the extensions are not run; B's stage-2 gate is held open.**
6a's extension would need ten fresh seeds averaging 838.7 and 6c's 965.2,
above every seed seen. Three misses in three rows with the same shape (most
seeds match, one flat or zero) is a suspect, not noise to average away: J19
re-runs the two tasks saving every model and re-evaluates them on 100
episodes both ways (our fixed targets; the paper's re-seeded ones). B does
not enter stage 2 until that reading is written. Rows 1b, 2b, 3, 3b and
6b stand.

**82. J10-2 carries the life across segments.** The published walk is one
unbroken run. The resumed twin showed our segments dropped the episode in
progress and re-seeded the env; with `carry_life=True` a resumed run is
bit-identical to an unbroken one. The faithful copy is the one a resume
cannot be told from; "disclose" would keep a known difference from the
recipe in a reproduction. J10-1 stays FAIL with the difference named.

**83. Forge round 1: the four fill-ins stand; the rung runs; acceptance
waits for the tripwire; the forge may build it if SYSTEM has not.** M = 5
seeds, 270 core-h, resets every 50k and one never-trains twin are the
charter's defaults and no term of ruling 69 contradicts them. The tripwire
is a hash check with a planted test; who types it matters less than that
the reader's own file is in the hash list, pinned before any raw file is
opened, and that the other lane reviews it. S13 (b) is first in SYSTEM.

**84. The lab's hindsight eye: ENTERED, not launched.** A table of the ray
colours at his mouth when he ate, filled by his own meals, points at food
on .890 of fresh looks (yoked twin .000). It is learned from his life and
nothing in it is borrowed (card line 2 holds); the hand reflex beside it
is a measuring stick only. It is a learner run in his world (ruling 58),
it has no published prior yet, and END-FIT prefers a sense the brain
learns to read itself — which is what J17's hunt tests. So: a RESEARCH
line for its prior today; it becomes J17's next candidate if the hunt ends
with no cure, or a cook-off against the cured critic if it ends with one.

**85. Kaggle CPU: five sessions in all; a per-order share binds only while
another order's READY kernel waits.** Three slots tonight read "3 RUNNING
= cap" and built speculative arms while two free sessions sat empty. The
share was J17's, not the account's. Ranks re-set in ORDERS.

**86. No first GPU launch without a timing cell; the meter bills run
time.** W47's 30 hours went in a day: 10.1 to an ear seed estimated at
1.5 with no timing cell, and 4.9 billed for a kernel's wait in Kaggle's
queue. S15 corrects the meter (no ceiling moves). Reported to the owner by
name: W47 bought two A rows passed, one missed, one mis-sized arm, and a
VOID; nothing expired, a third was wasted.

**87. J17's frozen readings stand as the worker froze them.** "Late" =
lives starting at or after 150k (ruling 65's late half; the last-quarter
number beside it); "eats" = §3's SUSTAINED or EATS BUT STARVES against the
never-trains floor. Five sessions on J17 on 11-21 was ruling 60 applied
correctly (no session idle). No verdict depends on either.

**88. The usage stop is planned for, not argued with.** The shared meter
read 63% at 06:37 (54% at 00:39), reset Thu 11-26 09:00 UTC; more than
half of the week's use is not this project's. The stop's number, the pace
switch and `.paused` are the owner's and are not touched. What changes is
the orders: every branch is ruled in advance, free sessions hold
multi-hour kernels, results land where Thursday's first slot can read
them, and a slot with nothing to advance exits instead of building. The
meter has contradicted itself four times this month; if it falls again,
nothing here costs anything.

**89. Sunday: END-FIT's unregistered gates wait for their part to open.**
The grip rule, the size cook-off and the rest of W2 read "none
registered". Each belongs to a stage-3 part; the plan forbids a spec with
no published prior and the standing objective is repair before register.
They are registered when the part opens, from its section of the
reference. The collapse guard is the exception already in hand: forge
round 1 measures it, and its winner's rule becomes the spec. No winner
adopted this week closes an END-FIT row.

**90. Sunday: GOAL.md read whole; the card is faithful; one seat
contradicted it.** The Language-routing seat (`meaning-mass`, LG.13) picks
what he says from a ready-made model's output. GOAL (11-11): nothing
borrowed inside him, and the adults "never choose his words". The verdict
stands as a result about the parent rig; the seat is marked a parent-rig
seat that cannot be adopted into him. His own voice is M6's work. Also
confirmed: how unified his brain is remains raced, not assumed (Sensory
fusion UNDECIDED after UB.10; stage 2's added-sense test is its next arena).

**91. Sunday: five passing specs re-examined against the week.** `T0.12`
(GPU accounting) passes while the meter billed 4.9 h of queue wait: it
proves refusal, not that the debit is the run. It gains a clause — a
kernel that waits Q and runs R is debited R — with the old meter as the
planted failure (S15 (3); it may go red there and is repaired in the same
order; no downstream spec needs the clause, so no new dependency edge).
`T0.04` (resume continues) measures optimiser state only; this week a
resume silently dropped the episode in progress, the world's RNG and the
physics. It gains a clause — resumed equals unbroken on a physics world,
a no-physics save must differ — from the test that already passes on
brain B (SYSTEM Next-up; no edge). `T0.03` and `T0.10` stand: their claims
(weights round-trip; artifacts return) were not what failed. `LF.03` (one
whole life) stands as a row about the 3.2.3 world it was bought in; from
now every ledger row records the physics version it ran on (SYSTEM
Next-up), and the survival stack's own resume takes the twin test (JACK
Next-up) — if it differs, M1's "resumes" carries the difference by name.

## 2026-11-21 06:41 UTC — Director, DAILY sitting: rulings

**64. E2c's NO BRANCH, and the default temperature.** E2c read 25 of 25:
both food arms eat (fixed α 0.1 sustained 4/5; learned α 3/5, late > 0.82
on 4/5), no zero control eats. Ruling 57's table has no such branch, so it
fires none: the fixed 0.1 is NOT shown to be the wall, and §19's cause 1 is
contradicted by the entropy trace (the policy goes sharp, not wide). Every
NEW brain-B run in his world uses `ent_coef="auto"` — not because a
measurement asks for it but because PLAN step 2 does: it is the published
setting, the fixed one is ours, and B's reproduced rows used it. Old rows
are not re-read. Policy entropy H may be REPORTED where the trace lost
action std; it does not stand in for §3b's frozen band.

**65. "Eats" means he still eats at the end.** In 10 of 10 E2c food runs
the skill is gone by 300k; rule E2c's clause passes on crossings anywhere in
the run, correctly by its frozen text. From today every NEW rule in his
world (J17, forge round 1, stage 2's mark and gate, A's pilot) reads the
late half of the run as primary; "reached then lost" is reported beside it.
No frozen rule is re-read. Card line 1: he must never stop learning.

**66. The eye race is frozen and goes first.** E2c's own reading says
pilot 2's miss is the eye. `prereg/row128_vs_rays16.md` is frozen as
drafted with S = `auto` (ruling 64; the draft's gate "more seeds" ties
4/5 sustained against 4/5 late, and its tie-break is auto), seeds 921–925,
300k, primary late meals/life. Prior on the page: the record is silent;
our hand forager eats on row128 (20/20). This is last-10% work by PLAN's
table. Ruling 58 is amended to admit it (order J17).

**67. Brain A's fix: one arm, one change.** A2 missed at 100K with a 1m
network against a band earned at size S. Arm (a) of `a2_size_s.md` is
frozen: sheeprl `dreamer_v3_S`, seeds 0–2, one extension, bands and control
unchanged, precision 32-true (so only size differs from our A1–A3; the
official float16 is written beside), prefill 1,024. In band → the build is
right, the miss is size, and A's pilot may be drafted on ruling 62's terms.
Miss → a build fault to find. The 12m arm is struck this week: it costs
more than the week holds and 12m is not his size.

**68. Brain B row 5 is a named miss, and costs B nothing more.** The
library arm read 833.8 against 841.7–864.3 at k = 15, one seed of 15 at
643.6; ours read 862.6 (not counted). The rule is frozen; the miss stands.
Nothing of ours is in the library arm, so there is no fix to order. B keeps
rows 1b, 2b, 3, 3b; stage 2's gate lists row 5 as a miss beside them. Row 6
runs by its rule at the lowest Kaggle rank.

**69. Forge round 1 is a KEEP round on E-food (amends 61).** On arm K
every candidate would read 0 (nothing reaches); on E-food the baseline
reaches and loses. Terms frozen in ORDERS J18: baseline fixed 0.1 + twin;
K1, K2, K3, K5, K6, `n_steps=3`, two plain-variation controls; S = 150k
(the loss begins at 81k+), M = 300k; score = share of late-half lives
sustained, tiebreak late meals/life; fresh seeds. The forge's four readings
of its charter (INBOX 11-20 14:35) stand. `target_entropy=-16`, CrossQ,
SimBa wait for round 2. A temperature cap has no prior: `inventor` only.

**70. A row that failed on our apparatus may be registered again, once.**
NH-1 failed 10–20× off with the Gaussian peaks inside walls — our map, not
the paper's. NH-2 (`prereg/needs_hidden_nh2.md`, written before the pilot
read) is frozen: one change, NH-1's bands and clauses untouched, launch only
if both pilot seeds read Δ < 5.0 — with one amendment: seeds 10–19, because
the pilot looked at two of 0–9. NH-1 stays FAIL. A second miss is a miss.
This is general: new id, fresh seeds, bands unchanged, the fault named, once.

**71. The TPU is closed (D58 executed at the desk).** Two pushes asked for a
TPU and were given a CPU box, 0.00 h billed. No further push asks for one;
S12 closes; the TPU line leaves `gpu_budget`'s report when the failing test
is touched. If the owner finds "TPU" offered on the website, the door
(`7689b52b`) is still there. 20 h lapsed in W46; none are counted from W47.

**72. J10's FAIL is not yet a fact about the published walk.** The
published 2M figure is one unbroken run; ours is four resumed segments. The
row stays FAIL by its rule. Before it is cited, a resumed twin is compared
with an unbroken one (box-minutes). Then J10-2: the same recipe, seeds 3–7,
≥ 4 of 5, eval of 50 episodes on one stated machine (the 10-episode eval
moved 13% across CPUs on the same weights). No cures on a reproduction row;
resets and TQC are the forge's candidates. s2's 2M (6,490) is on record.

**73. Session order, settled.** Kaggle CPU: < 1 h kernels, then J17 (≤ 3),
forge round 1 (≤ 2), J10-2 (≤ 2), B row 6. A one-command check an order
needs may take a rank-0 session without asking (J16's wheel did; correct).

**74. `PUBLISHED_PRIORS.md` has no file cap.** Workers read one section;
≤ 6 KB a section binds (ruling 55); the 70 KB line leaves the researcher's
prompt (S11 b). The page keeps its index at the top.

**75. The usage stop's evidence.** The stop refused 46 slots on a figure
that later read lower inside its own window. The number is the owner's and
is not touched. Ordered (S14): every refusal logs the pct, the meter line
and the printed reset, and the attribution tool counts slots refused
between contradicting readings — so the cost is one grep, and the owner
can decide on a reserved share with numbers.

## D58 — RESOLVED BY ARMED DEFAULT (fired 2026-11-21 by the Director; class conduct, desk-executable, decide_by 2026-11-28): STOP TRYING. S12 (a) closes as "TPU unreachable from the box"; no further push asks for a TPU; brain A stays on the free GPU and Colab. Cost: 20 free accelerator-hours a week unused. Ruling 71 above. The owner may reopen it by finding "TPU" offered under Accelerator on the Kaggle website.

## 2026-11-20 12:27 UTC — Director, DAILY sitting: rulings

**57. E2c gets learned-temperature twins.** Pilot 2 read every learner
below the same brain never learning, in the movement-free world too; the
priced suspect is SAC's fixed `ent_coef` 0.1 (§19: no published precedent
on 8 action dims; ≈ one meal of discounted value for staying alive). E2c
was launched on that same setting, so its miss could not be told from the
setting. Two arms are ADDED before any of their seeds exists —
`E-food-auto`, `E-zero-auto`, seeds 801–805, read by the same frozen
readings. The frozen fixed-α arms finish as frozen; nothing is dropped,
no threshold moves. The branch table is written before kernel a0.

**58. After pilot 2: one owner for the next learner run in his world.**
The pilot picks no setup: no arm's world made the task learnable for plain
SAC at 300k. Until J15 reads, no lane launches another learner run in his
world except J15 (E2c), J14 (brain A's smoke) and the forge's round 1 on a
rule the Director froze. Round 1 is the eating wall on arm K's world
(settings and run length, each candidate with a prior, a metric that
requires eating, plain variation as control). It does not adopt E2c: E2c
is a learnability proof with its own frozen rule. If forge order F1 has
not passed its planted tests by the 11-22 sitting, lane jack drafts the
rule as pilot 3 — the question does not wait on an instrument.

**59. J3's split.** The repaired entry passed sparse row 3 and failed
dense row 1; the cause is measured (noise held 32 decisions; γ is
cleared). A per-row hold is refused: a brain that goes into one creature
reproduces both rows on one setting. One last entry, `wm-latent@hold8`,
fresh seeds 10–14, rule first, no extension. The 8 is ours, not the
record's, and is stated as such. Pass both → the other arms get it; fail
either → J3 closes with every arm's numbers on record.

**60. Kaggle is five CPU sessions, and none idles.** Measured cap 5. The
ranking's per-order caps bind only while a lower-ranked READY kernel
waits. Every JACK slot starts by harvesting and refilling. The 13–15
idle session-hours of 11-19/20 were the weekly usage stop (all agents
paused from 11-19 18:07 at 96% until about 11-20 09:30),
not a ranking fault; no new organ is added for it. The free TPU (20 h a
week, 0 used, this week's lapses unused tonight) gets a door in `gpu.py`
and the owner's gate kernel (S12); no TPU plan before its verdict.

**61. The science lab's place.** Lane `forge` (owner 11-20) keeps its
orders in `docs/FORGE.md`: F1 → F2 → F4, F3 after F1 passes and J15
reads. Its fence is confirmed as written — whatever is improved never
writes its own judge — and the session's corrected advice stands: no LLM
reward search as the first move; a hand-set pre-registered grid. F5's
tripwire is S13; separation under another user writes outside the repo
and is the owner's to grant, so it is not ordered. The forge holds at
most 2 Kaggle sessions at rank 4. Lane `lab`'s learner or setting
candidates for brain B enter through `docs/FORGE_POOL.md` with a prior.

**62. The GPU week, completed.** After the ear, the needs-hidden
known-answer row (rule `5464c5b8`, ~2 T4-h) takes a GPU slot: ≈ 17 of 30
h ordered. The rest is for brain A's pilot in his world, but only by the
plan's steps: its rule is drafted after A1–A3 are in band and the smoke
passes, and frozen at a sitting before any seed.

**63. The lab's place key (11-20).** Two probes read ALIVE (.934 vs turn
search .933; carried to unseen worlds). "Memory and sleep" is at step 1
and stage 3; no order now. It is entered in PLAN's table as a last-10%
candidate; before it goes in him it owes a registered spec on fresh
worlds, looks from a real policy, and the 16-ray eye beside 128.

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

### D27 CLOSURE ADDENDUM — the screen is DELETED, and the answer to D27 is ZERO (builder, 2026-10-26 ~02:xx UTC, executing the Review's 2026-09-29 ruling on `d27-screen-measures-95-percent-false`)

**THE DELETION.** `metric_recorded_but_unread` is deleted from `run status` and
`experiments/unread_metrics.py` is removed from the tree (the module, its
`print_unread_metrics_block` call site and the `run unread` command — the exact
reversal recipe the entry above pre-registered). **The reason, so the
instrument's disappearance is distinguishable from one nobody liked:** its
measured false-positive rate was **19/20 = 95%** (hand-adjudicated 2026-09-21,
seed 27, table above), `D27`'s own text says the counter *"gets floored or
deleted once it exists"*, at 19/20 it cannot be floored, and the Review's
2026-09-29 ruling held that **delete is therefore the branch `D27` armed, not a
reversal of it**. The screen's final reading before deletion: 563 metrics on 60
certificates — a number that, at 95% false, could not be acted on spec-by-spec
and never once printed an instance of its own subject. The module and its
history remain in git at the commit carrying this addendum.

**THE ORDERED MEASUREMENT, and it returns ZERO.** The ruling ordered: *"follow
subscripts and helper calls on the same 20-draw and report how many instances
of D27's actual class appear."* Executed as
`experiments/d27_followed_reads.py` (self-contained — the 20 pairs are
hardcoded as adjudicated, so the measurement outlives the deleted screen;
invoked by nothing, rendered on no board). A transitive reach from `_check`'s
reads over local-variable assignments, string subscript slices and module-local
helper bodies, applied to the 2026-09-21 pairs:

- **Instances of `D27`'s actual class** — *the run measured the quantity that
  would have indicted it and then did not look at it* — **in the followed
  residue: 0.** Zero is the honest answer to `D27`, written down as an answer
  per the ruling rather than left as an absence.
- **And the narrowing itself is REFUTED as a screen**, which is the part the
  ruling could not know in advance: following subscripts and helper calls
  removed only **3 of the 13** known substance-read false positives
  (`SO.09 synth_lo_refused` — the subscript exemplar — plus `T0.26`,
  `LG.02 null_abs_div`) while wrongly absorbing **5 of the 6** genuinely-unread
  context keys (e.g. `ME.1 events` = `len(mem)`: sharing the broad object
  variable `mem` with gated keys is not being read by the gate). The dominant
  residual channel is **dynamic key composition** — `LC.02`'s
  `wm-latent/clears@1.0` is minted as `out[f"{arm}/clears@{t}"]`, so the helper
  exemplar is invisible to ANY static join, the same reason the deleted screen
  marked dynamic `_check` subscripts UNDECIDABLE. A narrowed screen is NOT
  warranted: deeper following degrades precision in both directions.

**THE DECORATIVE-METRIC DETECTOR — ADOPTED UNDER ITS OWN NAME, per the same
ruling, explicitly NOT as `D27`'s screen.** `experiments/literal_metrics.py`,
printed by `run status` and `run literal-metrics`: metrics whose every
recording site inside `_experiment`/`_control` is a bare constant.
Reporting-only, unfloored, P8 of its fixture asserts the not-D27 disclaimer
cannot be dropped from its render. Day-one reading: **2 decorative metrics on
2 certificates** — `T0.06 steps_ok=5` (the known scar) and **`T0.24
properties_checked=8`, a new genuine instance** (`t0_24_*.py:334`: the count is
hardcoded, never counted, and `_check` reads only `properties_failed` — the
exact battery shape `T0.19`/`T0.22` pin with `properties_checked ==
N_PROPERTIES`). Not acted on here: this reading refuses nothing by `D27`'s own
default; the `T0.24` repair rides with the next edit that bills `T0.24` anyway,
the same convention as the `T0.22` comment note on the gates row. Two
conservatism rules were bought by measured near-misses on day one and are
recorded in the module docstring with fixture properties P9/P10 (a self-test
fixture dict is not a recording site — `VO.02 cic`; branch-selected literals
are computed by control flow — `LT.01 c2_branch`).

**Invariants:** no threshold moved, no control loosened, no certificate
invalidated, nothing re-runs, nothing refuses. The owner's option (ii) — raise
Part 2's sample — remains theirs and is unaffected. Bills disclosed and paid in
the same slot: `experiments/run.py` is in `T0.22`/`T0.33`/`T0.36`'s
`IMPL_DEPS` (re-bought after the commit); the queue-row trace bills
`T0.21`/`T0.28`/`T0.31`.

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

## D36 — RESOLVED BY ARMED DEFAULT (fired 2026-09-27 ~00:5x UTC by the OVERSEER, 123rd audit, on the first legal day; transcription completed by the builder 2026-10-06 ~01:1x, NINE DAYS LATE — the gap is the 142nd audit's RANK 1 and the lateness is this desk's): (i) CHANGE NOTHING — the default was already the state of the world by the entry's own author, so the firing ordered no work and changed no behaviour; it closes the QUESTION, not a scheduling gap. Options (ii) SWAP and (iii) DECLINE W1 OUTRIGHT were NOT taken and remain the owner's to rule at any time.

**Transcription per the `D13` rule and the `D31`/`D32`/`D34` precedent
(`251ebcd`, `6aaed9b`): the overseer fires and records; the builder
transcribes. The firing block appended to `docs/DECISIONS_NEEDED.md` (at its
`## D36 — RESOLVED BY ARMED DEFAULT` header) is the firing record and is quoted
here in full. Until this entry landed, that block was the record; this entry
completes it and changes nothing in it. What the nine-day gap cost is on the
record and must survive transcription: the firing commit `c10a128` was in
NEITHER of `decisions.py`'s two identification channels — its subject names no
`D<n>` and matches neither keyword, and this page carried no `D36` heading — so
it was absent from the population `firing_diff_hazards` audits for nine days
while the `FIRING-DIFF` ratchet read 0. Its diff was audited at the 142nd
sitting and reads CLEAN (docs only, no `GOAL.md` edit, no constant moved), so
the exposure was zero.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no ratchet counter moved. **The two things the
firing itself says must survive transcription: the default was ALREADY the
state of the world at firing (the entry's own `CONDUCT-DESK` note: "the state I
have LEFT the world in"), so the firing settled the RECORD and ordered no work
— and the price the entry put on (i) is unchanged and realised: it knowingly
bought a fifth instalment of a design that had beaten four sittings.**

> ## D36 — RESOLVED BY ARMED DEFAULT, fired 2026-09-27 ~00:5x UTC by the OVERSEER (123rd audit), on the first legal day. Off your desk; options (ii) and (iii) remain yours to rule at any time.
>
> **THE OWNER DID NOT RULE BY 2026-09-26, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(i) CHANGE NOTHING** — `w1-world-edit-window` keeps 2026-09-27's FULL
> sitting under `D33`'s standing order, and `t108-pipeline-repair-has-no-design`
> stays at its `next_free_due` of 2026-10-02. Options **(ii) SWAP** and
> **(iii) DECLINE W1 OUTRIGHT** were **NOT** taken.
>
> **AND THE FIRST THING THIS FIRING MUST SAY: THE DEFAULT WAS ALREADY THE STATE OF
> THE WORLD, BY THE ENTRY'S OWN AUTHOR, SO THIS FIRING ORDERS NO WORK AND CHANGES
> NO BEHAVIOUR.** The entry says so itself, in its `CONDUCT-DESK` note: *"The
> DEFAULT, option (i), is a desk act and it is the state I have LEFT the world
> in."* Verified at source this audit rather than inherited:
>
> ```
> docs/REVIEW_QUEUE.md  w1-world-edit-window                 OPEN   DUE 2026-09-27
> docs/REVIEW_QUEUE.md  t108-pipeline-repair-has-no-design   OPEN   DUE 2026-10-02
> experiments/run review-queue                               both rows live, neither re-dated
> ```
>
> **So what this firing actually settles, stated narrowly.** It closes the
> QUESTION, not a scheduling gap. `D36` has been an open entry carrying a passed
> `decide_by` and nothing left to execute — `decisions --check` listed it as a
> stale `CONDUCT-DESK` row on the morning of the sitting it governs. After this
> firing the record says plainly that the allocation was made by a deadline
> passing rather than by a ruling, and it says so **before** 06:37 rather than
> after. No row moved, no date moved, no threshold moved, nothing was widened or
> narrowed, no GPU was committed and no certificate was staled.
>
> **Why the OVERSEER fired a `class: conduct` entry, since that is the reasonable
> objection.** The entry's author drew the line itself and I am honouring it
> exactly: everything desk-executable *was* executed, and the one act that was not
> — option (ii) — was refused on the ground that *"a desk may not re-order its own
> queue to move its own hardest unit off the only sitting big enough to hold it."*
> That reasoning is sound and I am not overriding it. What was left was an entry
> whose deadline had passed with its default already realised, which is precisely
> the case the firing idiom exists for (`D29`, `D31`, `D32`, `D34` precedent). The
> alternative — leaving it open and ageing — would let a passed deadline read as
> an unmade decision.
>
> **The price, restated because a firing may not quietly drop it.** The entry
> priced (i) itself and the price is unchanged: it *"knowingly buys a FIFTH
> instalment of a design that has beaten four sittings"*, and on the measured
> record its most likely product is another INCOMPLETE banner plus `T1.08`'s
> repair still undesigned on 10-02. Its only merit, and it is real, is that it
> keeps both debts VISIBLE and ageing rather than letting the desk re-order its
> own docket to suit itself.
>
> **ONE CORRECTION TO THE EVIDENCE, and it cuts against the entry's own mechanism
> rather than its conclusion (123rd audit FINDING 1).** `D36` argues from *"four
> of four Sunday FULL runs ever fired on cron died at max turns"*. That count is
> true and it is from the **pre-repair 3-turns/min regime**. Under the current
> budget (`scripts/review.sh:100`, `TURNS_PER_MIN=6`) the Review's deaths are
> **clock** deaths, not turns deaths: 2026-09-25 and 2026-09-26 both ended
> `rc=124` — `timeout`'s code — at exactly 20 minutes of a 20-minute wall, with
> their turn budget nowhere near spent. The repair's own comment predicted this
> (*"At 6/min the `timeout` becomes the binding ceiling"*). **The direction of
> `D36`'s argument survives — a sitting that has never completed a design of this
> size is still unlikely to complete one — but the failure mode it names has
> changed, and today's post-mortem must read the `rc` rather than repeat the
> count.** More turns will not buy today's sitting anything; only wall-clock
> would, and that is a budget decision, which is why it is on the owner's desk in
> this audit's `FOR THE OWNER` item 1 rather than being taken here.
>
> **WHAT REVERSES THIS.** Rule option (ii) SWAP or (iii) DECLINE at any time. (ii)
> was the entry author's own written recommendation. Note that after 06:37 today
> (ii) is moot for *this* Sunday and becomes a ruling about 2026-10-04.
>
> **Evidence:** `docs/DECISIONS_NEEDED.md` `D36` `DECIDE` block, `decide_by:
> 2026-09-26`; `experiments.decisions --check` at 2026-09-27 00:4x listing `D36`
> as `CONDUCT-DESK … (due 2026-09-26, STALE by 1 day(s))`; `/data/jack-logs/
> review.log` 2026-09-25 and 2026-09-26 `review start` lines and their
> `rc=124` seals at `06:57:1x`; `scripts/review.sh:89-101`.
>
> **OWED BY THE REVIEW, NOT BY ME:** the `## D36 — RESOLVED BY ARMED DEFAULT`
> transcription onto `docs/DECISIONS_RESOLVED.md`, per the `D13` rule that the
> overseer stays inside its own file set and the `D31`/`D32`/`D34` precedent.
> Until that lands, `decisions.py`'s second identification channel
> (`RECORD_PAGE`/`RECORD_MARKER`) cannot see this firing.

## D37 — RESOLVED BY ARMED DEFAULT (fired 2026-10-05 ~06:5x UTC by the OVERSEER, 140th audit, on the first legal day; transcription completed by the builder 2026-10-06 ~01:1x, the next builder slot after the firing): (iii) HOLD `D29` AS IT STANDS — the debt stays recorded in `docs/CHAMPIONS.md` and `LEARNING_CORE.md` §5.4, and the `Δ_k` collapse readout is NOT built. Option (i) BUILD THE DIAGNOSTIC — the entry's own quoted recommendation — was NOT taken and remains the owner's to rule at any time.

**Transcription per the `D13` rule and the `D31`/`D32`/`D34`/`D36` precedent:
the overseer fires and records; the builder transcribes. The firing block
appended to `docs/DECISIONS_NEEDED.md` (at its `## D37 — RESOLVED BY ARMED
DEFAULT` header) is the firing record and is quoted here in full. Until this
entry landed, that block was the record; this entry completes it and changes
nothing in it. Its firing commit `85cb0ee` was visible to the SUBJECT channel
only; this entry closes the record channel, one day after the firing.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no ratchet counter moved. **The two things the
firing itself says must survive transcription: `LEARNING_CORE.md` §5.4 still
promises `A4` a *mandatory* diagnostic that no code computes, and `A4` still
holds the Learning-core seat BY VERDICT off a VOID — the default leaves that
debt visible and unguarded rather than hidden. And the sharper edge the 10-04
addendum added: the premise `D29` lacked now exists and is committed, so this
default locked in the weakest option on the first day the missing premise could
have mattered. One sentence from the owner reverses it and the work is the
builder's.**

> ## D37 — RESOLVED BY ARMED DEFAULT, fired 2026-10-05 ~06:5x UTC by the OVERSEER (140th audit), on the first legal day. Off your desk; option (i) remains yours to rule at any time.
>
> **THE OWNER DID NOT RULE BY 2026-10-04, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(iii) HOLD `D29` AS IT STANDS** — the debt stays recorded in both
> `docs/CHAMPIONS.md` (per `D29`'s own firing) and `LEARNING_CORE.md` §5.4 (per the
> 2026-10-04 amendment), and the `Δ_k` collapse readout is **NOT built**. Option
> **(i) BUILD THE DIAGNOSTIC** — the entry's own quoted recommendation — was
> **NOT** taken.
>
> **WHY THIS WAS MINE TO FIRE, SINCE THAT IS THE REASONABLE OBJECTION.** `D37` is
> `class: goal` and armed; `decisions --check` listed it this morning as the single
> `OVERDUE — DEFAULT IS DUE TO FIRE` entry, and firing is the day after `decide_by`
> (`D33`'s expiry note; the entry's own 10-04 addendum states **"The default fires
> 2026-10-05"** in terms). The overseer firing an armed default on the first legal
> day is the established idiom — `D27` (107th), `D29` (109th), `D32`/`D34` (115th),
> `D31` (119th), `D36` (123rd). Nothing here reverses the entry's own 10-04 ruling
> that the `CONDUCT-MISFILED?` reclass is **REFUSED**: I am not executing option
> (i) at a desk, I am recording that the deadline passed without a ruling.
>
> **THIS FIRING ORDERS NO WORK AND CHANGES NO BEHAVIOUR.** Verified at source this
> audit rather than inherited from the entry:
>
> ```
> LEARNING_CORE.md §5.4        debt block present, original paragraph byte-identical beneath it
> docs/CHAMPIONS.md            Learning core still HELD: BY VERDICT — marking untouched
> experiments/  effective_rank  no computing call site; `svd|singular|RankMe|eig|spectrum` = audio spectrum only
> champions --check            EXIT 0, all 10 violations AT their declared floors
> ```
>
> It is **MONOTONE**: it moves no threshold, edits no constitution text, unparks
> nothing, spends no GPU-hour, fails no spec, stales no certificate, and leaves the
> debt **visible and unguarded** rather than hidden. It picks only among
> already-permitted actions — it is, exactly, the status quo.
>
> **THE PRICE, RESTATED BECAUSE A FIRING MAY NOT QUIETLY DROP IT.** In the entry's
> own words: *"the next latent-prediction arm inherits an uncomputable VOID
> condition, i.e. exactly the hole this row spent thirteen days finding, and the
> project keeps a `mandatory` guard it has never once been able to run."* And the
> sharper edge the 10-04 addendum added: the premise `D29` lacked **now exists and
> is committed**, so this default locks in the weakest option on the first day the
> missing premise could have mattered. `LEARNING_CORE.md` §5.4 still promises `A4` a
> *mandatory* diagnostic that no code computes, and `A4` still holds the
> Learning-core seat BY VERDICT off a VOID.
>
> **WHAT IS NOT PERISHABLE, SAID SO IT IS NOT OVERSOLD.** The entry's `blocks`
> field is honest: no spec id waits on this today. Every arm behind the seat is
> `LC.03` (VOID-FORECLOSED) or `LC.07` (PILOT-BLOCKED, ~526 wall-hours against
> 30 h/week). Nothing was lost **this week** by the deadline passing.
>
> **WHAT REVERSES THIS.** Rule option **(i) BUILD IT** at any time — it stays the
> entry's own recommendation, in ActSWM's `Δ_k` form (roll the latent twice from one
> context, once under recorded actions and once under the all-zero action sequence,
> score both by cosine against the true future latent; under collapse both rolls
> agree and `Δ_k` → 0, so unlike a loss curve it cannot be satisfied by a collapsed
> latent), with the floor **pre-registered before the run it gates, never after
> seeing it**. One sentence from you reverses it and the work is the builder's. The
> counterargument the entry raised against itself is also unchanged and I am not
> re-arguing it: a desk that missed its own deadline should not get to reopen a
> closed decision by doing its homework late.
>
> **Evidence:** `docs/DECISIONS_NEEDED.md` `D37` `DECIDE` block, `decide_by:
> 2026-10-04`, and its 2026-10-04 addendum naming 2026-10-05 as the firing date;
> `experiments.decisions --check` at 2026-10-05 06:37 printing `D37  costs 0 specs
> OVERDUE — DEFAULT IS DUE TO FIRE`; `docs/DECISIONS_RESOLVED.md` `D29` recording
> option (i) as NOT TAKEN.
>
> **OWED BY THE REVIEW, NOT BY ME:** the `## D37 — RESOLVED BY ARMED DEFAULT`
> transcription onto `docs/DECISIONS_RESOLVED.md`, per the `D13` rule that the
> overseer stays inside its own file set and the `D31`/`D32`/`D34`/`D36`
> precedent. Until that lands, `decisions.py`'s second identification channel
> (`RECORD_PAGE`/`RECORD_MARKER`) cannot see this firing.

## LG.13 — WINNER — meaning-mass

> **STALENESS RE-BUY, not a third verdict (annotated 2026-09-26 by the
> builder, in the same slot that appended it).** Attempt 3's deterministic
> re-run of the `## LG.13 — WINNER` records above — appended 2026-09-26
> 02:33 by the re-buy owed after the `bakeoff.py` receipt-redirect edit
> (`5ee32ff`, the hash-salt differential's write-guard) staled LG.13's
> certificate. Byte-identical verdict for the same reason as attempt 2's:
> every arm is a deterministic function of the frozen verdict table. The
> row's own hash-salt differential read CLEAN (49 deciding metrics +
> eligibility legs reproduced exactly). A reader counting bakeoff verdicts
> should count ONE.

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

> **MARKER (builder, 2026-10-01, 134th audit FTB 1b):** the cost tie-break shown
> above is the DEFECT the `so10` row was routed for, not a result to adopt:
> `laplace-full` is INELIGIBLE (migration gate, every seed negative), the
> Person-model seat stays **VACANT**, and `SO.10`'s **FAIL stands**
> (ruling of 2026-09-29, `docs/REVIEW_QUEUE.md`).

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | yes | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | yes | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | yes | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | yes | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | yes | 1.0 |

> **MARKER (builder, 2026-10-01, 134th audit FTB 1):** the `admitted | yes`
> column above is the unsupplied-predicate DEFAULT, not a predicate's verdict —
> `lg_13_chooser_seat_bakeoff.py` supplies no `admissible=`. Nothing was
> checked. The verdict itself is unaffected (admission was never at issue
> here); the renderer now prints `—` for the unevaluated case.

## SO.10 — TIE — laplace-full
laplace-full leads laplace-w30 by only 0.26 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (laplace-full, cost 0).

metric: `div_lastq`  ·  null 0.044 ± 0.069  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of one already-recorded evidence stream that no arm can perturb (the diary holds the claim and the finding; the follow decision is never recorded, and `rng_agent` draws once per round whatever the rule returns). A low score is therefore a property of the RULE — full-history Laplace cannot migrate, last-claim-only is memoryless — and not evidence that its run was broken, which is exactly the case `validity` mode would mis-VOID. The gate itself is unmoved at 3 sigma and MIN_FINISHERS still applies.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| laplace-full | 0.722 | 5.79 | pass | yes | 0.0 |
| laplace-w30 | 0.689 | 5.11 | pass | yes | 1.0 |
| exp-decay-h15 | 0.678 | 5.91 | pass | yes | 1.0 |
| last-1 | 0.544 | 3.60 | pass | yes | 2.0 |
| control:pooled-scalar | 0.044 | 0.00 | FAIL | yes | 0.0 |

> **MARKER (builder, 2026-10-01, 134th audit FTB 1b):** this record is the
> proof of the 134th audit's RANK 1 — appended mechanically by the regate sweep
> `5bc7471`, 21 minutes after the admissibility predicate landed, with every
> `admitted` cell filled by the DEFAULT (`so_10_trust_rule_bakeoff.py` supplies
> no `admissible=`; nothing was checked). And the verdict it re-renders is the
> defect itself: the cost tie-break is what the `so10` row was routed for,
> `laplace-full` is INELIGIBLE (migration gate, every seed negative), the
> Person-model seat stays **VACANT**, and `SO.10`'s **FAIL stands** (ruling of
> 2026-09-29, `docs/REVIEW_QUEUE.md`). The renderer now prints `—` for an
> unevaluated arm.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## SO.10 — TIE — laplace-full
laplace-full leads laplace-w30 by only 0.26 sigma (margin 1.5). The choice does not matter yet; taking the cheapest tied arm (laplace-full, cost 0).

metric: `div_lastq`  ·  null 0.044 ± 0.069  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of one already-recorded evidence stream that no arm can perturb (the diary holds the claim and the finding; the follow decision is never recorded, and `rng_agent` draws once per round whatever the rule returns). A low score is therefore a property of the RULE — full-history Laplace cannot migrate, last-claim-only is memoryless — and not evidence that its run was broken, which is exactly the case `validity` mode would mis-VOID. The gate itself is unmoved at 3 sigma and MIN_FINISHERS still applies.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| laplace-full | 0.722 | 5.79 | pass | — | 0.0 |
| laplace-w30 | 0.689 | 5.11 | pass | — | 1.0 |
| exp-decay-h15 | 0.678 | 5.91 | pass | — | 1.0 |
| last-1 | 0.544 | 3.60 | pass | — | 2.0 |
| control:pooled-scalar | 0.044 | 0.00 | FAIL | — | 0.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## D43 — RESOLVED BY ARMED DEFAULT (fired 2026-10-17 ~12:5x UTC by the OVERSEER, 151st audit, on the first legal day; transcription completed by the builder 2026-10-17 ~13:1x, the same day as the firing — the first same-day transcription in the series): (iii) HOLD — NO WORLD-EDIT WINDOW IS DECLARED AND THE SEVEN MEASURED FINDINGS STAY DECLINED. The entry's own quoted recommendation — "open a NARROW, SINGLE-PURPOSE world-edit window and let the 10-11 split decide which of the seven enter it" — was NOT taken and remains the owner's to rule, in one sentence, at any time.

**Transcription per the `D13` rule and the `D31`/`D32`/`D34`/`D36`/`D37`
precedent: the overseer fires and records; the builder transcribes. The firing
block appended to `docs/DECISIONS_NEEDED.md` (at its `## D43 — RESOLVED BY
ARMED DEFAULT` header) is the firing record and is quoted here in full. Until
this entry landed, that block was the record; this entry completes it and
changes nothing in it. The firing itself named this transcription as owed
(OVERSIGHT.md 151st audit, FOR THE BUILDER item 2) on the day it was created,
which is why the gap is hours rather than the nine days `D36` cost.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no ratchet counter moved, no terminal row
re-opened. **The three things the firing itself says must survive
transcription: the firing is MONOTONE and ordered no work — the world was
never edited (`playground.py` untouched since `761121a`, 2026-08-19) and no
HELD row was waiting; the price is restated, not dropped — seven real
measurements (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`,
`sh02-null-saturation`, `w1-cold-is-not-lethal-at-night`,
`w2-needs-have-no-single-k`, `hr5-fixture-refuted`, `ba03-vestibular-...-one-
kick`) survive only as closed rows, and the split row
(`seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`, OPEN, DUE
2026-10-25) is the one organ still looking; and the bill the owner was asked to
weigh was wrong by 67% — the entry priced the window at 21 PASS certificates
and `run stale-cost playground.py` measures 35, which cuts FOR the
recommendation that was not taken.**

> ## D43 — RESOLVED BY ARMED DEFAULT, fired 2026-10-17 ~12:5x UTC by the OVERSEER (151st audit), on the first legal day. Off your desk; the window remains yours to declare at any time.
>
> **THE OWNER DID NOT RULE BY 2026-10-16, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(iii) HOLD — NO WINDOW IS DECLARED AND THE SEVEN STAY DECLINED.** The
> entry's own quoted recommendation — *"open a NARROW, SINGLE-PURPOSE world-edit
> window and let the 10-11 split decide which of the seven enter it"* — was **NOT**
> taken. No window exists. No terminal row was re-opened. Nothing was withdrawn.
>
> **WHY THIS WAS MINE TO FIRE.** `D43` is `class: goal` and armed; `decisions
> --check` listed it this audit as the single `OVERDUE — DEFAULT IS DUE TO FIRE`
> entry, and firing is the day after `decide_by`. The overseer firing an armed
> default on the first legal day is the established idiom — `D27` (107th), `D29`
> (109th), `D32`/`D34` (115th), `D31` (119th), `D36` (123rd), `D37` (140th). The
> 2026-10-17 Review could have ruled it and did not: it sat at 12:22, disposed 28
> OVERDUE rows, and died `rc=124` at 12:42 without reaching the register.
>
> **THIS FIRING ORDERS NO WORK AND CHANGES NO BEHAVIOUR.** Read from the files
> this audit, not inherited from the entry:
>
> ```
> the seven, via review_queue.parse()   ne01-occlusion-knife-edge          DECLINED
>                                       water-apply-phantom-force          DECLINED
>                                       sh02-null-saturation               DECLINED
>                                       w1-cold-is-not-lethal-at-night     DECLINED
>                                       w2-needs-have-no-single-k          DECLINED
>                                       hr5-fixture-refuted                DECLINED
>                                       ba03-vestibular-...-one-kick       DECLINED
> w1-world-edit-window                  DECLINED 2026-09-28 — no window to re-open
> HELD rows in docs/REVIEW_QUEUE.md     0 — nothing is waiting on this entry
> playground.py                         last touched 761121a, 2026-08-19 — the
>                                       shared world was never edited
> the split row                         seven-rows-are-held-behind-a-refused-
>                                       window-and-a-moot-decision = OPEN,
>                                       DUE 2026-10-25, outside the hold, alive
> ```
>
> It is **MONOTONE**: it moves no threshold, declares no window, re-opens no
> terminal row, unparks nothing, spends no GPU-hour, stales no certificate,
> softens no control and withdraws no finding. It picks only among
> already-permitted actions — it is, exactly, the status quo.
>
> **THE PRICE, RESTATED BECAUSE A FIRING MAY NOT QUIETLY DROP IT.** In the entry's
> own words: *"the world-touching subset stays unrepairable for as long as the
> hold lasts, the `W0 is too shallow` finding keeps being true and keeps being
> unactionable, and the seven rows' substance survives only for as long as
> somebody keeps choosing to read closed rows."* Three things sharpen it since the
> entry was written:
>
> 1. **The split row is still OPEN and has been re-dated to 2026-10-25** — the
>    `DUE: 2026-10-11` the entry relied on broke, so the one organ that was going
>    to separate the genuinely world-touching rows from the spec-local ones has
>    not sat. The entry's own hazard sentence — *"If that sitting does not happen,
>    seven real measurements are closed and nobody is looking at them"* — is now
>    the live state, not the risk.
> 2. **The economic argument the entry quoted is understated by 67%.** It prices
>    the window at *"21 `PASS` certificates"*; `run stale-cost playground.py`
>    measures **35 standing PASS certificates — 2.88 CPU-hours over 32 and 1.52
>    GPU-hours over 3**, four builder slots of CPU re-buy with `VO.02` alone at
>    3195.8 s, longer than the 3000 s slot that would owe it. The correction is
>    not mine: the queue found it and `world-edit-window-price-is-quoted-at-21-and-
>    measures-35` is ACTED. It cuts **for** the recommendation (paying 35 once
>    beats paying it seven times) and it means the bill in the entry is the wrong
>    number to decide on.
> 3. **`D33`, whose stop-rule produced this class, is `CONDUCT-DESK` and STALE by
>    24 days.** The upstream that made the holds dead is itself undischarged.
>
> **WHAT REVERSES THIS.** Say the window may exist — in one sentence, at any time.
> Authorship is already settled (`D22`, with the Review), the split row is alive
> and dated 2026-10-25, and nothing in this firing has to be undone first: declare
> it and the split puts rows into it. Equally, ruling that the window may **never**
> exist is a real answer and closes the class honestly rather than by silence; the
> seven rows' findings are then permanently a matter of record only, which the
> owner should want said out loud rather than inferred from a passed date.
>
> **TRANSCRIPTION OWED, AND NAMED SO IT IS NOT LOST.** `docs/DECISIONS_RESOLVED.md`
> is outside the overseer's write permissions (SYSTEM.md `D13`), so this firing has
> no record there yet. Measured after the append rather than predicted: the tool
> reports **`firing transcriptions: 31 of 32 ... UNTRANSCRIBED: D43 (invisible to
> the RECORD channel and to firing_coverage until transcribed)`** and **`EXIT 0`,
> with the `firing-diff` ratchet still `0/0`** — so this is a visible reported
> debt, not a ratchet break, and it will not fail anybody's gate. That makes it
> *easier* to forget, which is why it is ordered as **FOR THE BUILDER item 2** in
> `docs/OVERSIGHT.md`. The precedent is `D36`/`D37`, transcribed at `5bc466f`
> nine days and one day late respectively — which is exactly why it is being named
> here on the day it is created rather than left for an instrument to discover.
>


## D41 — RESOLVED BY ARMED DEFAULT (fired 2026-10-19 ~12:5x UTC by the OVERSEER, 153rd audit, on the first legal day; transcription completed by the builder 2026-10-19 ~13:2x, the same day as the firing — the `D43` same-day precedent held): (iii) NEITHER — HOLD BOTH ARTEFACTS, REPAIR EACH INSTANCE ON ITS OWN ROW, RE-ASK IN WRITING. The entry's recommendation was NOT taken; no artefact was named as the thing we are building, `T1.11` stays an honest FAIL with no call site added, and options (i)/(ii) remain the owner's to rule in one sentence at any time.

**Transcription per the `D13` rule and the `D31`/`D32`/`D34`/`D36`/`D37`/`D43`
precedent: the overseer fires and records; the builder transcribes. The firing
block appended to `docs/DECISIONS_NEEDED.md` (at its `## D41 — RESOLVED BY
ARMED DEFAULT` header) is the firing record and is quoted here in full. Until
this entry landed, that block was the record; this entry completes it and
changes nothing in it. The firing named this transcription as owed (OVERSIGHT.md
153rd audit, FOR THE BUILDER item 4) on the day it was created.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no terminal row re-opened. **What the firing says
must survive transcription: it is MONOTONE and is, exactly, the status quo; the
price is restated, not dropped — the recipe drift stays live, a third bridge may
be built and go uncalled the same way, and `demonstrated` (107 at transcription
time, read off `run status` after this slot's five clean re-buys — which moved
no verdict, only stamps, so the count stands exactly as the firing reported it;
a first draft of this line wrote 109 unchecked and was corrected in-slot) keeps
being reported without the caveat that it describes one of two artefacts; and both
instances keep live owners — `t111-certified-loss-has-no-shipped-caller` OPEN
DUE 2026-10-25, `t108-recipe-repair-is-exhausted-and-the-attribution-pair-
flipped-sign` OPEN DUE 2026-10-21.**

> ## D41 — RESOLVED BY ARMED DEFAULT, fired 2026-10-19 ~12:5x UTC by the OVERSEER (153rd audit), on the first legal day. Off your desk; options (i) and (ii) remain yours to rule at any time.
>
> **THE OWNER DID NOT RULE BY 2026-10-18, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(iii) NEITHER — HOLD BOTH ARTEFACTS, REPAIR EACH INSTANCE ON ITS OWN
> ROW, AND RE-ASK IN WRITING.** The entry's recommendation was **NOT** taken; no
> artefact was named as the thing we are building. `T1.11` stays an honest FAIL
> and was **not** repaired by adding a call site. `T1.08`'s Step 0 reports the
> recipe divergence without resolving which recipe is correct.
>
> **WHY THIS WAS MINE TO FIRE.** `D41` is `class: goal` and armed;
> `decisions --check` listed it this audit as one of two
> `OVERDUE — DEFAULT IS DUE TO FIRE` entries, and firing is the day after
> `decide_by`. The overseer firing an armed default on the first legal day is the
> established idiom — `D27` (107th), `D29` (109th), `D32`/`D34` (115th), `D31`
> (119th), `D36` (123rd), `D37` (140th), `D43` (151st). The 2026-10-18 Sunday
> FULL could have ruled it and did not; the Review has not sat at all on
> 2026-10-19 (`06:37` deferred on the usage gate at 93%, the `09:22` and `12:22`
> retries both `ABORT: load`).
>
> **RE-DERIVED AT SOURCE THIS AUDIT, not inherited from the entry.** A firing may
> not rest on the entry's own eight-day-old arithmetic:
>
> ```
> action_training_loss  call sites outside experiments/   0
>                       (every caller is a spec test: T1.02, T1.03, T1.04, T1.06,
>                       T1.07, T1.08, T1.09, T1.12, T4.02, T6.03, T1.11 itself)
> make_action_optimizer spec callers      4  (T1.07, T1.08, T1.09, T6.03)
>                       script callers    1  (scripts/probe_t108_step1.py)
>                       pipeline callers  0
> TrainingPipeline.py:493   torch.optim.AdamW(params, weight_decay=1e-4, eps=1e-5)
> the shared SPEC recipe    plain Adam, warmup-then-constant, no decay, clip 2.0
> T1.11  live ledger status  FAIL
> T1.08  live ledger status  FAIL, attempt 3, ran_at 2026-09-13T10:05:05, 3d357c4
> ```
>
> Both halves of the measurement hold unchanged. Routing status today:
> `t111-certified-loss-has-no-shipped-caller` is **OPEN, DUE 2026-10-25**;
> `t108-pipeline-repair-has-no-design` is **ACTED**; the successor
> `t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign` is
> **OPEN, DUE 2026-10-21**. So both instances still have a live owner and neither
> needed this answered to proceed on its measurement half, exactly as the entry's
> `blocks:` field said.
>
> It is **MONOTONE**: it moves no threshold, repairs nothing cosmetically, deletes
> no module, adds no call site, spends no GPU-hour, stales no certificate,
> unparks nothing, and leaves both findings standing and red. It picks only among
> already-permitted actions — it is, exactly, the status quo.
>
> **THE PRICE, RESTATED BECAUSE A FIRING MAY NOT QUIETLY DROP IT.** In the entry's
> own words: *"the drift stays live, a third bridge may be built and go uncalled
> the same way, and the `demonstrated` count keeps being reported without the
> caveat that it describes one of two artefacts."* The count is **107** as of this
> morning and is reported without that caveat on every page in this repository.
>
> **HOW TO REVERSE IT.** Name **(i)** or **(ii)** in one sentence — which artefact
> is the thing we are building — and the two routed rows change shape
> accordingly. Nothing has been foreclosed by this firing.


## D42 — RESOLVED BY ARMED DEFAULT (fired 2026-10-19 ~12:5x UTC by the OVERSEER, 153rd audit, on the first legal day; transcription completed by the builder 2026-10-19 ~13:2x, the same day as the firing): (iv) HOLD — NO SUCCESSOR IS DESIGNED, AND THE THREE CLAIM-DEAD COMMITMENTS (smell, shelter/building, thermal) STAY CLAIM-DEAD AND SAID SO. The desk's recommendation — name `thermal ("too cold kills him")` as the one — was deliberately NOT taken, because choosing among the owner's constitutional commitments would be a desk writing `GOAL.md`. Naming one commitment remains the owner's, in one sentence, at any time.

**Transcription per the `D13` rule and the same precedent series as `D41`
above; the firing block appended to `docs/DECISIONS_NEEDED.md` is quoted here
in full and nothing in it is changed.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved,
no control loosened, nothing unparked, no PARKED marker touched, no terminal
row re-opened. **What the firing says must survive transcription, and it is
worse than the entry stated: the question's named live home
(`five-commitments-are-claim-dead-behind-foreclosures`) is DECLINED — terminal
— so as of this firing smell, shelter and "too cold kills him" have no live
claim spec, no open queue row, and no open register entry; the question now
exists only inside closed entries, which is exactly the shape `D30` and `D40`
were raised about. `coverage` keeps printing `claim_dead = 3` and that number
is the one organ still saying it out loud.**

> ## D42 — RESOLVED BY ARMED DEFAULT, fired 2026-10-19 ~12:5x UTC by the OVERSEER (153rd audit), on the first legal day. Off your desk; naming one commitment remains yours to rule at any time.
>
> **THE OWNER DID NOT RULE BY 2026-10-18, SO THE PRE-REGISTERED DEFAULT FIRED.**
> Option **(iv) HOLD — NO SUCCESSOR IS DESIGNED, AND THE THREE COMMITMENTS STAY
> CLAIM-DEAD AND SAID SO.** The desk's own recommendation — *"NAME `thermal ("too
> cold kills him")` AS THE ONE, and let smell and shelter wait"* — was **NOT**
> taken, and deliberately so: choosing among the owner's own constitutional
> commitments is a desk writing `GOAL.md`.
>
> **WHY THIS WAS MINE TO FIRE.** `D42` is `class: goal` and armed;
> `decisions --check` listed it this audit as one of two
> `OVERDUE — DEFAULT IS DUE TO FIRE` entries, and firing is the day after
> `decide_by`. Same idiom and same precedent list as `D41` above.
>
> **RE-DERIVED AT SOURCE THIS AUDIT.** `coverage` this morning:
>
> ```
> claim_dead = 3   (unchanged since 2026-09-26)
>   smell              2 specs  0 pass  0 now   CLAIM-DEAD — SM.02 PARKED 2026-08-20,
>                                               SM.03 FORECLOSED (PILOT-BLOCKED)
>   shelter/building   2 specs  0 pass  0 now   CLAIM-DEAD — SH.01 PARKED 2026-08-25,
>                                               SH.02 FORECLOSED (PILOT-BLOCKED)
>   thermal (kills)    4 specs  0 pass  0 now   CLAIM-DEAD — same two
> commitments_uncovered = 0      (correct, and the reason no instrument can ask)
> champions: Smell (olfaction)   among the 4 seats nobody can ever WIN
> park_release_pairs = 3         SH.01 -> SH.02 and SM.02 -> SM.03 both unwalkable
> ```
>
> Every park remains legal and evidence-backed. Nothing was unparked, nothing was
> deleted, no PARKED marker was touched.
>
> It is **MONOTONE**: it moves no threshold, unparks nothing, deletes no
> commitment, registers nothing, spends nothing, and cannot make the hole smaller
> or hide it.
>
> **THE PRICE — AND IT IS NOW WORSE THAN THE ENTRY STATES, WHICH A FIRING MUST
> SAY RATHER THAN INHERIT.** The entry's `blocks:` field names
> `five-commitments-are-claim-dead-behind-foreclosures` as this question's live
> home, *"(OPEN, DUE 2026-10-14)"*. **That row is now DECLINED** — terminal, and
> a terminal row is never re-read (100th audit B2). Verified via
> `review_queue.parse()` this audit: `DECLINED, 48 d, DUE 2026-10-14 (+5 d)`.
>
> So as of this firing:
>
> ```
> smell / shelter / thermal ("too cold kills him")
>   live claim spec     none        (every one PARKED or PILOT-BLOCKED)
>   open queue row      none        (the named home is DECLINED)
>   open register entry none        (this one, as of this block)
>   instrument that can ASK for the repair   none — a missing spec has no id
> ```
>
> That is precisely the shape `D30` and `D40` were raised about: a recommendation
> surviving only inside a closed entry, where nothing can make it answerable. The
> entry's own words apply with the DECLINE added: *"three of the capability
> target's named commitments remain unfalsifiable, the daily report of that
> becomes wallpaper, and the first two instruments that can see the hole continue
> to be unable to ask about it."*
>
> **HOW TO REVERSE IT.** Name **ONE** commitment in one sentence — the desk's
> quoted recommendation is `thermal`, on the arithmetic that it is load-bearing
> for the other two and the only one whose successor needs no world edit — and a
> successor claim spec gets registered that iteration. The firing forecloses
> nothing; it records that nobody chose.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*



## D44 — RESOLVED BY ARMED DEFAULT (fired 2026-10-25 ~12:4x UTC by the OVERSEER, 160th audit, on the first legal day; transcription completed by the builder 2026-10-25 ~13:1x, the same day as the firing — the `D41`/`D42`/`D43` same-day precedent held): NO RULING — THE CHARACTER SHEET STAYS EXACTLY AS IT SHIPS AND NOTHING IS DOCUMENTED. The desk's split recommendation — rule the five temperament floats DELIBERATE, rule the five `core_values` strings and the persona prose a BREACH — was NOT taken, deliberately: the entry armed the no-ruling branch precisely so a clock could not decree who Jack is. Both halves remain the owner's to rule, in one sentence, at any time.

**Transcription per the `D13` rule and the `D31`/`D32`/`D34`/`D36`/`D37`/`D41`/
`D42`/`D43` precedent: the overseer fires and records; the builder transcribes.
The firing block appended to `docs/DECISIONS_NEEDED.md` (at its `## D44 —
RESOLVED BY ARMED DEFAULT` header) is the firing record and is quoted here in
full. Until this entry landed, that block was the record; this entry completes
it and changes nothing in it. The firing named this transcription as owed
(OVERSIGHT.md 160th audit, FOR THE BUILDER item 1) on the day it was created.**

**PROVENANCE CAVEAT, carried forward rather than laundered: the firing was
written by an overseer run that exited `rc=1` mid-report, and its own seal line
says its facts are "that dead run's, unverified until an audit re-measures
them." This transcription moves the record between files; it does not upgrade
the facts. What the builder re-verified independently at transcription: the
firing block exists verbatim in `docs/DECISIONS_NEEDED.md`; `decisions --check`
reads "UNTRANSCRIBED: D44"; the registry at HEAD holds 260 spec ids with the LG
family stopping at LG.14 and no character-sheet spec — so the firing's central
price (the promised spec was never registered) stands re-confirmed at HEAD.**

**Invariants checked at transcription:** no `GOAL.md` edit, no threshold moved
in either direction, no control loosened, no new permission created, nothing
re-run, no certificate staled, no terminal row re-opened. **What the firing says
must survive transcription: the contradiction is LIVE in shipped default code
(`Personality.py:135-139` five Big Five floats, `Personality.py:731` import-time
singleton, `UnifiedBrain.py:4093` wired on by default) while `GOAL.md` says
"His kindness is not decreed"; NO spec measures it; NO instrument can ask for
one, because an unregistered spec has no id; and the register entry that
promised the spec is closed as of the firing. The queue row
`personality-is-a-typed-character-sheet-on-the-answer-path` (DISPOSITIONED,
DUE 2026-11-05) is the one live owner of that registration, and the priority
page's `1^22` ITEM 3 unit 2 orders it from THE RULING on that row.**

> ## D44 — RESOLVED BY ARMED DEFAULT, fired 2026-10-25 ~12:4x UTC by the OVERSEER (160th audit), on the first legal day. Off your desk; both halves of the split recommendation remain yours to rule at any time.
> PROVENANCE (scripts/lib_seal.sh, 2026-10-25T12:57:16+00:00): this entry was written by a overseer run that exited rc=1 without completing its own checklist — its report is sealed as an INCOMPLETE RUN draft. The entry's facts are that dead run's, unverified until an audit re-measures them; nothing here is re-dated, weakened or un-armed by this line.
>
> **THE OWNER DID NOT RULE BY 2026-10-24, SO THE PRE-REGISTERED DEFAULT FIRED.**
> **NO RULING — THE SHEET STAYS EXACTLY AS IT SHIPS AND NOTHING IS DOCUMENTED.**
> The five Big Five floats stay, the five `core_values` strings stay, the
> 2,585-character persona keeps conditioning the default answer path, and
> `GOAL.md` keeps forbidding in text what the code ships by default. The desk's
> own recommendation — *"SPLIT THE OBJECT AND RULE THE TWO HALVES OPPOSITELY"* —
> was **NOT** taken, and deliberately so: the entry itself says a default that
> blessed the status quo would be a desk ruling on who Jack is by letting a clock
> run out, and that reasoning is why the default is the no-ruling branch and not
> the recommendation.
>
> **WHY THIS WAS MINE TO FIRE.** The entry is `class: goal` and armed;
> `decisions --check` listed it this audit as the single
> `OVERDUE — DEFAULT IS DUE TO FIRE` row, and firing is the day after
> `decide_by`. Same idiom and same precedent list as the firings above.
>
> **RE-DERIVED AT SOURCE THIS AUDIT, not inherited from the entry.** Every
> number in its measurement block reproduces in this venv, read off the files
> today:
>
> ```
> Personality.py:135-139   openness 0.85 · conscientiousness 0.55 · extraversion 0.60
>                          agreeableness 0.80 · neuroticism 0.25   (dataclass defaults)
> Personality.py:731       JACK_PERSONALITY = Personality()         (import-time singleton)
> UnifiedBrain.py:264      enable_emotional_state: bool = True
> UnifiedBrain.py:4093     self.personality = JACK_PERSONALITY      (wired on by default)
> grep across experiments/ ZERO callers of answer_question / get_system_prompt /
>                          _generate_with_llm  -> no spec on this ladder sees it
> ```
>
> Nothing in the measurement is contested and nothing about it changed.
>
> **THE PRICE — AND IT IS WORSE THAN THE ENTRY STATES, WHICH A FIRING MUST SAY
> RATHER THAN INHERIT.** The default's own text leans on one mitigation: *"The
> spec registered this morning proceeds on its own clock and REPORTS whatever it
> finds either way."* **That spec does not exist.** Verified three ways this
> audit:
>
> ```
> registry             258 specs; the LG family is LG.00..LG.14 and stops there
> grep registry*.py    ZERO hits for answer_question / get_system_prompt /
>                      'character sheet'; the only 'persona' hits are one
>                      unrelated wiped-persona twin control
> git log -S'Spec("'   since 2026-10-09, exactly ONE spec was registered:
>                      LC.08 (da9ece0, 2026-10-20). No other registration exists.
> ```
>
> The queue row's own ruling says the design is published and *"the registration
> is the builder's from it"*; the entry dates that registration 2026-10-22. It
> has not happened, and the row that owns it has now broken a dated promise on
> it. So as of this firing:
>
> ```
> the contradiction            live, in shipped default code
> a spec that measures it      none
> the register entry           this one, closed as of this block
> an instrument that can ASK   none - no spec id, so no gate, no ranking, no blocker
> ```
>
> That is the shape the coverage tool exists to shout about and cannot see here,
> because an unregistered spec has no id.
>
> It is **MONOTONE**: it moves no threshold, edits no constitution text, deletes
> no component, stales no certificate, spends no GPU-hour and no CPU-second, and
> withdraws no finding. It also decrees nothing: the sheet is exactly as it was
> before this block.
>
> **HOW TO REVERSE IT.** One sentence either way reverses it at any time, and the
> desk's quoted recommendation is still on the record above: rule the five floats
> and their PAD baseline DELIBERATE (and add the sentence to `GOAL.md` so five
> undocumented dataclass defaults stop being undocumented), and rule the five
> `core_values` strings and the persona prose a BREACH to come out of the default
> answer path, paying the 18-certificate `UnifiedBrain.py` bill once on that half
> alone. The firing forecloses neither half; it records that nobody chose, and
> that the evidence the entry promised would arrive regardless has not been
> registered.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-11 — acting Director (owner's session): three rulings made with the reorganisation

**1. `T1.08` is unbundled, not weakened (conduct).** The spec passed twice in
August on its original claim (seed noise small next to the effect, SNR ≥ 3). On
2026-09-13 a second claim was added to the same spec — held-out seed CV ≤ 7% —
with a bar taken from one early reading (the queue row's own name:
`t108-bar-set-from-n1-is-now-the-projects-largest-blocker`). After two honest
repairs the CV is 17%; that is a true and standing fact about the Stack-B
recipe, `MAX_HELDOUT_CV_PCT = 7.0` stays byte-unmoved, and `T1.08` stays FAIL.
What changes is the WIRING. `T1.08`'s `kills` clause reaches "any single-seed
claim"; five of its six direct dependents (`T2.01`, `T2.02`, `T2.03`, `T2.14`,
`D1.0`) run three seeds behind their own sigma gates and quote no single-seed
number. They are re-pointed to `T1.14`, which carries the original August gate
verbatim. `T2.13` (seeds = 1) keeps its edge. Dependency edges are how the
ladder is wired — conduct, which the desks amend — and no gate, threshold or
control moves. The design question had been open since 2026-09-16 and blocked
45 specs for eight weeks. *Loser recorded:* "more seeds until it passes" — the
spec's own Monte Carlo shows the tightness is headroom, not sample; and
"accept the false-fail rate" — which leaves 45 specs behind a coin.

**2. `D1.0`'s VOID is answered by `D1.1`, not by reinterpretation (means).**
The learning gate compared arms to a random-action policy; untrained twins
cleared it at 3.9σ because in Humanoid standing still out-scores flailing. The
null was wrong. `D1.1` re-registers the same arms with the gate set to trained
vs own untrained twin, paired — a harder null, so a strengthening — and is run
fresh on free GPU rather than re-scored from `D1.0`'s rows.

**3. "One whole life" means `LF.03` on the survival stack (conduct; amends
D35).** D35's freeze named `T6.01` as its lift condition. `T6.01` is a Stack-B
"companion session" spec written before the survival-world direction of
2026-08-09; the learning this project has demonstrated is on Stack A. The
freeze stands unchanged in force and now lifts when `LF.03` — one life with the
learning core in the loop — records a verdict. Whether Stack B's Tier-6 specs
are retired, re-parented or kept as challengers is an architecture question and
goes through `docs/CHAMPIONS.md`, raced, not decreed.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-11 (later) — owner ruling: the stop is 95%, and nothing else holds the work back

**Verbatim:** *"yes run the director test tonight go for it don't hold anything
back anymore go to claude 95% then stop only then"*.

- **The hard stop moved from 90% to 95%** of the weekly meter (`USAGE_STOP` in
  `scripts/lib_usage.sh`). It is still a hard stop for every organ, still read
  from the real meter, and still refuses to run when the meter is unreadable.
- **"Stop only then": the pace line is OFF by default** (`JACK_PACE=on` turns it
  back on). The owner's own rule of 2026-08-09 was *"no throttling nothing just
  pause all agents"* at the limit; pacing was added later, on 2026-08-24, to
  spread a starved budget across the week, and over the following weeks it
  skipped 604 hourly slots against 509 at the stop itself.
- *Cost, recorded beside the ruling:* in a week where other work on the account
  takes most of the pool, the agents will now run flat out and go dark once 95%
  is reached, instead of running thin all week — and free GPU hours that expire
  inside a dark tail will expire. Only 5 points of the weekly pool are now kept
  back for everything else on the account.
- *How to reverse:* `JACK_USAGE_STOP=90` and `JACK_PACE=on` in the crontab
  lines, or change the two defaults at the top of the pacing block.

## 2026-11-11 21:58 UTC — Director, first sitting (FULL): rulings

**1. Inside/around: a result whose "Jack" is a ready-made model is a result
about the parent rig, not about him (conduct; applies GOAL.md's inside/around
rule — changes no verdict, no threshold).** `LG.00` (frozen SmolLM2-360M plus
the diary; its own header: "the learned core contributes nothing"), `HR.4`
(ready-made ASR + embedder), `T3.08` (LLM as his command encoder), the
Language-routing seat's `LG.10/12/13/14` (a frozen LLM phrases his speech) and
`ME.11.C/D/E` (pretrained embedders in his memory): every ledger verdict
STANDS as measured. What changes is the credit: none of them counts as a
capability of Jack on the roadmap, in coverage or in a seat. They are kept as
parent-side tools and as scored-but-ineligible references for the learned
parts that must replace them. The Language-model seat (WHICH model the parent
is) is an around-him seat and keeps its holder; the Language-routing seat —
where a frozen model phrases HIS speech — is to read VACANT for his own part
until a learned challenger holds it by verdict. `LG.00` remains GOAL.md's
puppet guard, and it is half-run: GOAL.md says strip the diary AND the learned
core and his answers about his own life must collapse; today's `LG.00` has no
learned core in the loop, so only the diary half has ever been tested. It is
to be re-run with the seated core in the loop once `LF.03` exists — a
strengthening. `PG.6`'s kill-text "move vision to a frozen tower" is struck as a
remedy (a frozen tower inside him is forbidden); `UnifiedBrain.py`'s
wav2vec2/SmolLM2 loaders stay default-off and may be used only by a
reference arm; `AudioListener._transcribe_api` is a paid endpoint and is to be
deleted — the owner's own instruction in the ORDERS inbox of this date (free
compute only). ROADMAP M6's status line is corrected today; the registry and
CHAMPIONS edits are ORDERS "Next up 1". *Loser recorded:* voiding the specs —
they measured what they said they measured; the fault was the reading.

**2. `T1.14` — the record corrected, and the counter not banked (conduct).**
The acting Director's ruling above says all five re-parented specs sit "behind
their own sigma gates". True for `T2.01`, `T2.02`, `D1.0`. `T2.03` gates on
`all_seeds_positive` plus margin floors and `T2.14` on `all_seeds_beat_null` —
three seeds, no single-seed quote, so the substance holds; the wording did
not. `pass_on_dead_dependency` 7 → 5 is a re-wiring, not a repair, and is NOT
to be recorded as a floor until `T1.14` has a PASS row. `T1.14`'s control is
ordered strengthened with a seed-ignoring arm that is actually run (ORDERS J1).

**3. `W.1`/`W.2` have an owner (conduct).** They stay in M2's done-when — a
world whose heat balance and needs ledger fail their own published rules is
not "a world that can teach" — and are owned by ORDERS J6 with `NE.01`.

**4. The clerk buys nothing on a dirty tree (conduct; ratifies `7d33eba`).**
A certificate stamped `+dirty` is worse than the one it replaces.

**5. `DECISIONS_NEEDED.md`'s cap (conduct).** The page is inside its cap when
it holds only OPEN decisions; nobody archives a live addendum to shave bytes.

**6. GPU routing (conduct).** Until ORDERS S2 explains why Kaggle gave
`jack-ladder-1794429036` no session, jobs of ≤ 2 h go to Colab and longer
jobs are split to fit. Kaggle's "30 free hours" is a figure this repo's own
file computes; the account is shared with another project, so it is an upper
bound, not a balance.

**7. Four stale conduct entries are settled by the reorganisation.** `D33`
(the world-edit design that lost three Sundays): superseded — the deeper
world is ORDERS J7 and the red W0 rules are J6; an order has no Sunday to
lose. `D38` (two defaults claiming the FULL Review's first act): superseded —
the Director's first act is fixed: ORDERS.md inside 15 minutes. `D46` (which
organs the pace line exempts): superseded by the owner's ruling above — pace
off, one hard stop at 95% for every organ. `D47` (the Review queue is not
drainable): superseded — the queue is closed to new rows and the Director
disposes up to 10 a sitting, oldest first. Still open and stale, to be read
and ruled three a sitting: `D39`, `D40`, `D48`, `D49`, `D50`, `D52`; `D35`
stands in force until `LF.03` has a verdict.

## 2026-11-13 06:40 UTC — Director, DAILY sitting: rulings

**1. W0 has ONE declared clock, k = 72, and a race-won world edit may ship.**
`W.1`(b) and `W.2`(c) (red 73 days) are one question. The world declares a 72×
clock (a 1200 s day). Thermal already obeys it: TAU_T 240 s = 4.74 h / 72.
Hunger and thirst do not (starvation 1700 s against 25,200; dehydration 570 s
against 3,600 — 14.8× and 6.3× past the clock, and a hunger:thirst ratio of 3
where a human's is 7). Biology is the oracle and nothing is assumed: a
per-channel compression nobody declared is a fudge, and picking a k from the
failing reading would fit the bar to the data (the J6 worker rightly refused).
So: (a) `W.1`(b) and `W.7`'s control are re-registered to test every channel
against the DECLARED k — the experiment's null was wrong (it demanded 1×
physics inside a world whose clock is declared 72×); tolerances byte-unmoved;
shipped needs must then FAIL `W.7` honestly. (b) The needs clock at k = 72 is
raced as a playground arm against shipped on `NE.01`'s probes and `W.2`'s
checks, rule declared before reading; it ships if it passes `W.2`(c) and is no
worse on every `NE.01` clause. (c) D43's HOLD (fired by silence 10-17) is
lifted for exactly this class: a world change that WON a pre-declared race on
registered probes. Hand edits without a race stay held. Grounds: the owner's
11-11 words (the world is "as important as jack himself"; "dont be afraid of
fixing when broken") and card line 3. Cost stated, not hidden: lives get up to
~6× longer in sim-time; S1 (off-box CPU) is where that is paid.

**2. `LF.04`, `BD.01`, `CR.01` and the 3× scale spec run on Kaggle CPU, not
here.** D20/D39 stand: `cpu<48h` stays closed on this box and the 57,600 s
ceiling is untouched (the box serves paying customers). The route is SYSTEM S1
— a `kaggle-cpu` budget class, measured 11-12 to be accepted with the GPU
quota spent and faster per learner than this box. Moving a spec to that class
changes its budget only; claim, seeds, gates and controls stay byte-same. If
S1 has not landed `LF.04` by the 11-15 sitting the Director rules again.

**3. No new detached learner-side pilot until `LF.04` has a verdict.** The
11-12 CPU day went 62,434 s against 57,600, almost all to unregistered pilots
on seeds 90–94 while the registered creature gate sat unrun (auditor, finding
3). The pilots were good work and killed six hypotheses by declared rules; a
further day of them is avoidance. Exempt: J6's world races, J7's registered
runs, and the `LC.02` quiet-box re-run, which needs the box idle anyway.

**4. `LC.02`'s FAIL is a fact about Jack until shown otherwise; no ratchet is
banked on it.** A wall-clock spec was scored while our own race loaded the
box. J1 re-runs it quiet. PASS → the spec gains a load precondition that
VOIDs (never FAILs or PASSes) above a stated load. FAIL → the core got slower
and that is repaired. `pass_on_dead_dependency` / `unreachable` are banked
with `ratchets record --only` after that verdict, at the then-live values.
Yesterday's ruling-2 numbers (5 / 93) are withdrawn as stale.

**5. `UB.10`: an honest "no arm separates" is a result, not a VOID to chase.**
If attempt 3 FAILs on `top1_stable` with the arms inside .02, the Sensory
fusion seat stays UNDECIDED and says so with the six medians. No further
venue is hunted for a winner. Whether `UB.11–13`/`TA.03` need a WINNER or only
a completed contest is a dependency question the Director answers when the
worker lists them.

**6. The control-path contest has an order again (J2).** It fell out of
yesterday's file by my error while the order of work still named it; the
auditor caught it. `D1.0` re-dispatches when Colab frees.

## 2026-11-12 06:40 UTC — Director, DAILY sitting: rulings

**1. A GPU-wait chain is not CPU work.** A process that only sleeps until a
Colab session frees and then calls `scripts/dispatch.sh` may be launched with
plain `setsid`, outside `launch_detached.sh`'s wall-clock CPU meter. It must be
named, with its log path, in the order's `state:` line; it may not run a
registered spec itself (the dispatch it calls does, through the sanctioned
lane). No new budget lane is built for it — the machine may not grow.

**2. Two ratchet movements are earned and may be banked alone.**
`pass_on_dead_dependency` 7 → 5 (bought by `T1.14` PASS at `73665ca`, which
includes the run seed-ignoring arm) and `unreachable` 96 → 93. The worker was
right not to run `ratchets record`, which rewrites every moved counter at once
and would have blessed `review_queue_violations` 0 → 73 unseen. ORDERS S3
step 0 adds `--only <counter>`; the other six movers stay unrecorded until
someone has read each.

**3. Four old blockers are unowned by decision, not by silence.** `HR.1`,
`HR.5` (voice corpus, playground sounds) wait for the learned-ear order
("Next up 3") — repairing a corpus for a ready-made ear nobody will seat is
work for no milestone. `LT.02` (chaos detector) waits for curiosity to be
raced under J5. `T0.08` re-derives again (clerk sweep, 11-12). Seven JACK
orders are open; the creature gate, the two GPU cook-offs and the world
outrank them.

**4. FIELD_WATCH N1 (judge admission gate, arXiv:2611.13596): yes, queued.**
An adult that scores him must first accept a real success from his own world
and reject the same episode with the deciding event removed; a yes-to-all
judge must fail. It opens with J5's teacher-proposed arm, its first user —
not before, because no adult's score gates anything today.

**5. `LF.03` meets M1's done-when; the follow-on is `LF.04`, not a tick.**
M1 said "one whole life", an integration claim, and it passed with both
controls run. Whether living makes him better is a separate claim with its own
control (his frozen twin) and is ORDERS J1 as of today, ranked above the GPU
work because it is the creature. `D35` (in force "until `LF.03` has a
verdict") is discharged by that verdict.

## D35 — RESOLVED, LIFT CONDITION MET (2026-11-12, Director): the Tier-0 freeze ends because `LF.03` recorded a verdict — PASS at `6a92fd4`, three seeds, five deaths each, both controls run. The freeze did its work: one life ran end to end before anything else was built. What survives it is not this entry but the Director's standing rules — the machine may not grow (SYSTEM lane max 3 orders, no new organ), and JACK work is what `docs/ORDERS.md` ranks first. No gate weakened, no threshold moved.

## D39 — RESOLVED BY ARMED DEFAULT (fired 2026-11-12 by the Director; decide_by 2026-10-15): (i)+(iv) STAND. Wall clock remains the CPU day's unit, `CPU_DAY_CEILING_S` stays 57600 s, no detached-lane sub-ceiling is created. Nothing is raised, split or relaxed. The `cpu<48h` class's real answer is not a different unit on this box but free CPU off it (ORDERS SYSTEM "Next up 1", S1); one narrow clarification rides with it — a chain that only waits for a GPU session is not CPU work (ruling 1 of the 2026-11-12 sitting).

## D40 — RESOLVED AS SUPERSEDED (2026-11-12, Director): the question — whether the builder stays paced against the shared `week:all models` meter — was answered by the owner on 2026-11-11 ("the stop is 95%, and nothing else holds the work back"): the pace line is off, one hard stop at 95% binds every organ. Default (v) therefore has nothing left to hold. The stop, the meter and the pace switch are the owner's and untouched.

## D54 — RESOLVED BY ARMED DEFAULT (fired 2026-11-11 by the Director; decide_by 2026-11-08): HOLD THE BAR AND LEAVE `HR.1` RED. `CLEAN_BAR` stays 0.10, `HR.1` stays FAIL, `HR.2`/`HR.3`/`HR.4` stay killed. No threshold moved, nothing deleted, no GPU spent. The hearing family's way forward is not this corpus's bar but a learned ear raced against a no-ASR null (ORDERS "Next up 2"), which needs a corpus whose channel does not identify the speaker — a repair to the recording fixture, never to the bar.

## D51 — RESOLVED BY ARMED DEFAULT (fired 2026-11-11 by the Director; decide_by 2026-11-03): THE FLOOR STAYS AND THE REDESIGN ROUTES ONWARD. `KA_SIGNAL_MIN` 0.90 is byte-unmoved; `PS.09` keeps its verdict. The default routed the signal-present arm's redesign to "a Review design row"; the Review queue is closed, so it routes to the Director's order list instead — not opened today (it frees 0 specs; M1 and the repairable blockers outrank it). Any repair that needs the bar to move returns to the owner.

## D53 — RESOLVED BY ARMED DEFAULT (fired 2026-11-11 by the Director; decide_by passed): ADOPT (c)'s DIAGNOSIS AND GIVE IT A LIVE OWNER. `SO.07`'s VOID is re-filed as a WORLD/BODY capability defect (the body cannot reach privileged food on 2 of 3 recording worlds), one class with `ba03-null-saturates-the-horizon` and `SH.01`'s `ORACLE_CANNOT`. `REF_MIN_FED` is byte-unmoved at 0.8, nothing re-based, `SO.07` stays VOID. Live owner: ORDERS J6 (the red W0 world rules), in place of the long-closed `D21`.

## D55 — RESOLVED BY ARMED DEFAULT (fired 2026-11-11 by the Director; decide_by passed): THROTTLE THE INTAKE AT THE DESK, AND BUY NO EXTRA CLOCK. Adopted in a stricter form than written: since the 2026-11-11 reorganisation `docs/REVIEW_QUEUE.md` takes NO new row at all, and the Director disposes up to 10 old rows a sitting. No timeout raised by this firing, the usage gate untouched.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-13 08:45 UTC — Director, second DAILY sitting (after the owner's END-FIT): rulings

**5. `UB.11`–`UB.13` and `TA.03` need the fusion contest RUN, not WON; the
trunk is A0.** `UB.10` returned FAIL on attempt 3 (`de563c4`): rig green, no
arm separates from the incumbent A0 (medians within .016; A2 beats A0 on every
seed but the top arm changes by seed). The three specs test properties of the
trunk — ablation matrix, synergy over the ensemble, retrieval gate — and
nothing in them reads a winner. The edge as written (`UB.10` PASS) makes a
legitimate "nobody won" block four specs forever, and hunting a venue until
someone wins is the thing card line 4 forbids. So: their dependency becomes
"`UB.10` has a non-VOID verdict", asserted inside each spec from the ledger,
and each declares its trunk before running — A0, the incumbent, because no
challenger earned the seat. Where cheap, A2 (median top) is carried as a
second, reported arm. No bar, control or seed count changes. If a later venue
separates the arms, the four are re-run on the winner. They queue on the GPU
after `D1.0`.

**6. END-FIT is placed on the route; it does not jump the queue.** The owner's
finished Jack (GOAL.md's closing section; END-FIT in `CHAMPIONS.md`) is what
he must BECOME — the owner's to say, and said. Placement: ROADMAP M2 gains W2
(the tool chain and buildable shelter), M3's done-when gains END-FIT's gates,
M4 gains hands, M6 the advice gate, M7 the hand-over. Of the gates END-FIT
lists as "none registered", ONE is ordered now — the hand-over gate (J8),
because it is cheap, it needs only the off-box CPU route S1 is already
building, and it protects every later seat decision; it is exempt from
repair-before-register as a spec an order needs. The collapse guard rides in
J3 (the 3× core race implements LEARNING_CORE §5.4). The grip rule, the size
cook-off, the advice gate and W2 are Next-up 6, opened one at a time: W2
cannot be specified honestly before W1 has fire and parts, and hands before
the control seat has a holder. `LF.04` stays first, as the owner's session
itself wrote.

**7. The `language` slot's specification is stale and is corrected in words
only.** `cores.py:75` calls it "a sentence embedding from the frozen text
tower". Nothing frozen or borrowed sits inside him (card line 2; GOAL: words
reach him "through his ears"). The comment is changed to say the slot is
empty and will be fed by his own learned ear; no dimension, weight or spec
moves (J8). No language arm may be built on the old wording.

## D48 — RESOLVED BY ARMED DEFAULT (fired 2026-11-13 by the Director; decide_by 2026-11-01 passed): NO EXEMPTION, NO READER. No guard-propagation instrument is built; the machine may not grow. A propagation gap between sibling specs is recorded by hand by whoever meets it, and the Sunday re-examination of 5 passing specs reads siblings in pairs.

## D49 — RESOLVED BY ARMED DEFAULT (fired 2026-11-13 by the Director; decide_by 2026-10-27 passed): NO EXCEPTION to strengthen-only exists. The row `pl02-void-gate-quantifies-over-its-own-nulls` is DECLINED on the record, its measurement preserved; `PL.02`'s gate stays byte-as-is. The honest route for a gate arm measured to be an artefact is the Director's standing one — RE-REGISTRATION as a new attempt with reasons, old version kept in history — and it will be ordered when `PL.02` reaches the order page (it is behind `PG.6`, stale 95 d), not by editing the arm in place.

## D50 — RESOLVED BY ARMED DEFAULT (fired 2026-11-13 by the Director; decide_by 2026-11-03 passed): REPORT THE FIRING WHERE ITS DAMAGE IS READ. Each Director sitting reads the usage meter and the organ logs; a fired stop is reported in PROGRESS "Dead or stuck" together with its dark slots and missed dates as one finding. Reporting only; the stop, PACE_FLOOR and PACE_CAP are untouched. Today: no stop fired (week:all models 15%).

## D52 — RESOLVED BY ARMED DEFAULT (fired 2026-11-13 by the Director; decide_by 2026-11-05 passed): RETRY THE SWEEP WITHIN ITS WEEK. `field_watch.sh` may be given later attempts in the same ISO week, each asking the usage gate exactly as now and exiting on a refusal, at most one completed sweep per week. The gate's numbers are byte-unmoved. Queued as SYSTEM Next-up (6) in `docs/ORDERS.md`; installing a cron line is outside the repo and stays the owner's.

## 2026-11-14 06:45 UTC — Director, DAILY sitting: rulings

**1. `T6.03` no longer depends on `T2.10`; it measures what it needs itself.**
`LF.04 ← LF.03 ← LF.02 ← T6.03 ← T2.10 FAIL` held the creature gate for two
days. `T2.10` is red on one conjunct only — paraphrase recall on the
stem-disjoint fixture (added 08-31, 0.0, the ME.11 arena's known weakness).
`T6.03` stores and queries on the templated venue (cue shares the content
word; `t6_03_cross_session.py:268-269`), where `T2.10` reads recall@1 1.0
against recency 0.0. `LF.02` takes from `T6.03` only "stores survive a session
boundary". No link in the chain measures or uses paraphrase recall, and the
runner cannot express a per-conjunct edge. So: `T6.03` gains a clause in its
own run — on its own store and cues the scorer's recall@1 must beat
recency-only by ≥ 0.10, `T2.10`'s own margin — and its `depends_on` becomes
`["T0.05", "T0.08"]`. This is a strengthening of `T6.03` (it now fails by
itself if retrieval is no better than recency) and the removal of an edge
that carried an unrelated red. `T2.10` is byte-untouched and stays FAIL; its
owner is JACK Next-up 2 (a learned scorer raced on the certified fixture).

**2. A green certificate on J1's path is not edited for a lower order.**
`LC.02`'s quiet-box PASS (att 22, 06:10, foreign load 0.66 of a 1.0 limit) is
the standing verdict; its VOID-on-contention precondition stays as built.
Until `LF.04` is dispatched, `cores.py`, `survival.py`, `w0.py`, `needs.py`
and the `LC.02`/`LF.02`–`LF.04` test files are frozen to every other order —
including comment edits and race-won world edits. Last night an edit for the
7th-ranked order staled step 1 of the 1st. Venue: `LC.02` stays on this box
(it is a claim about this box's throughput); no Kaggle timing.

**3. The CPU day is split.** Of the 57,600 s ceiling, detached jobs (races,
pilots) may use at most 34,560 s in total, and at most 2 detached worker
processes may be alive at once across all jobs; 23,040 s are reserved for
registered runs and the clerk. Workers apply it by hand today; SYSTEM S2
makes `admit`/`launch_detached.sh` enforce it. Reason: 11-13 ended at
59,599 s with every registered run refused for six hours and load 6.7 on a
box that serves paying tenants.

**4. `NE.01` is re-registered in the world's own time units; ruling 1 of
11-13 (one clock, k = 72) stands.** Its windows (random deaths within one
day; meals in three days; probe lengths) were sized in seconds on the fast
needs clock, so a world on the human clock cannot pass them without being
more lethal than a human one — the test, not the world, refused the k = 72
arm. Each window becomes a multiple of the world's lethal times (the multiple
it has today on the shipped clock, written before any run, so the shipped
world's windows are numerically unchanged and nothing is easier for it). The
integrity probe, measured to be a ratchet whose reading is set by its
schedule (`docs/research/NE_01_SPREAD_I.md`), becomes option (A) with the
one schedule-free stop rule: falls until death (as built in `33a1acc`, with its planted ground-height control) — the same "to the
lethal edge" exercise every other need gets. Bars unmoved: 5/5 random deaths,
spread 0.30, 5 meals. Old version kept in history.

**5. `W1.04`(c) is re-registered on independent lives.** As registered it
cannot pass in any world (p5 capped at 266.6 s against 5600 s; `93f4040`). New
(c): ≥ 20 independent fresh lives per arm-seed, censored at 5600 s, bar and
control unchanged. The passive arm leaves (c): `W1.01` claims that arm dies,
and a clause demanding it also survive contradicted the spec beside it. That
death stays asserted by `W1.01`, so nothing is loosened.

**6. Meal memory (LAB, ALIVE) enters under the Consolidation seat** as a lab
cook-off: its declared grind (seeds 94–98 × 3 arms × 13k) runs on Kaggle CPU
kernels behind `CR.01` — no box CPU, so the 11-13 hold on local learner
pilots is not touched. It is registered against `NE.05`'s seat only if the
tally clears the gate already written in its module docstring.

**7. `LF.06` runs with its legs reversed.** Train in a Kaggle CPU kernel,
fetch the checkpoint out, examine and continue on this box. The claim is
"handed to another machine he keeps what he knew and learns more"; which
machine is which does not enter it. No input channel is built — the machine
does not grow.

**8. `LC.09` runs as registered; the seat is not repaired first.** The seated
wm-latent trains no action- or state-reading weight and the guard will VOID
it. That is the race's finding about the seat. wm-dyn is the
action-conditioned challenger already built; a repair to the seated arm
enters, if at all, as a new arm after the verdict.

## 2026-11-15 06:40 UTC — Director, FULL sitting: rulings

**1. Foraging is the first order; the 11-14 freeze is lifted.** `LF.04` FAIL
and `CR.01` FAIL share one cause: every life ends of energy and he does not
learn to eat (`docs/research/LF_04_FAIL.md`). `cores.py` may be edited under
order J1 only. The policy loss is seated by a tally on three fresh seeds
(150–152), rule written first: highest mean late-third span among arms with
saturated fraction ≤ 0.05 and no seed whose late third is under half its
early third; the winner re-buys `LC.02`, `LC.08`, `LF.03`. The 11-12 race
(one seed, already read) nominated AWR; it does not seat it. Then all seven
built learning-core arms and a random floor are raced on meals per life
(seeds 153–154, 15 lives), bar fixed from the FAIL analysis: ≥ 5 floor meals
or ≥ 1 apple per late life on both seeds.

**2. `W1.04`: conjuncts (a) and (c) read different arms.** They contradicted
because both read `W1.02`'s repeat arm: (a) needs it to complete ≥ 20 lives
inside the horizon, (c) needed its 5th percentile to reach the horizon. (c)
exists to show the window is survivable "by the lives that are supposed to
fill it" — the competent arm. (c) is measured on the oracle only (20 fresh
lives per seed, censored at 5600 s, control unchanged), and gains a clause:
the oracle's p5 must exceed the repeat arm's p95, so survival is by
competence and not by a soft world. (a) is unchanged. The repeat arm leaves
(c) for the reason the passive arm did under 11-14 ruling 5. Net: one clause
moved to the arm it was written about, one clause added.

**3. `NE.01`'s fatigue-spread clause becomes an edge probe.** Measured
(`64d949a`): the fatigue spread in a random life is fatigue's start-up
transient (TAU_F 300 decisions); it falls monotonically with life length
(.37–.40 at ~470 decisions, .09–.12 on long-lived arms), correlation with
energy −.15. The clause therefore scores how early the life ended, and it
cannot pass together with the gate-2 window on any world. Re-registered in
the shape of the integrity probe (11-14 ruling 4): exertion to exhaustion
then rest; fatigue must cross ≥ 0.30 of its range and recover; a planted
control with fatigue frozen must FAIL. Bar 0.30 unmoved; written before any
run; the old reading stays on the row, reporting-only.

**4. A budget class says where a run really goes.** `W1.00` (20 pooled seeds)
was classed `cpu<2h`, enumerating 360,000 s against a 57,600 s day, and ran
detached anyway; `T0.33` correctly went red. `W1.00` is re-classed
`kcpu<12h` and runs on Kaggle CPU. `T0.33`'s clauses are not touched.

**5. The 2-alive detached cap counts computing processes.** Amends 11-14
ruling 3. The cap protects the customers' CPU; a Kaggle poller billing under
0.10 cores is not load. Pollers are capped separately at 6. The 34,560 s
detached day and 57,600 s ceiling stand. Kaggle ran 4 CPU kernels beside a
GPU kernel (measured 11-15 06:30); orders may use 4.

**6. Reattach.** `JACK_REUSE_KERNEL` is forbidden on any multi-unit spec
until order S7 lands (scope to the first submit; the pin-swap proof done by
the tool; calls 2..N pinned to call 1's HEAD). `D1.0` att 8 would have put
mislabelled units into a seat verdict had it not crashed.

**7. Parked behind a forager:** `LF.06`, the `LC.09` claim run, a `CR.01`
re-race, an `LF.04` re-run, the diary-in-decisions order. Each measures a
learner improving; with a learner that starves each would only re-measure
`LF.04`'s finding (the `LF.06` smoke already shows a fresh brain's exam at
577.6 s against 597.6–601.2 trained). Meal memory stays DEAD as declared
(.0985 < .10) — the bar is not re-read after the fact.

**8. T4, not P100.** Kaggle assigns T4 (all 9 `D1.0` att 9 units). Any GPU
spec whose minutes cap was frozen from a P100 pilot re-derives it from a
measured T4 rate before its next dispatch (no gate moves; step targets are
the matched quantity). `D1.0` tries one long unit on Colab this week so the
verdict need not wait for W48.

**9. Five passing specs re-examined (Sunday); three are strengthened, none
weakened.** `LF.03` (one whole life) would pass for a body that never acts:
no clause reads behaviour, and its row's spans (433–599 s) sit at or below
the statue's 599.9 s. It gains a statue-twin control the gate must detect;
"he eats" is NOT added — that is `LF.04`'s claim, not milestone 1's. Its
train cadence is read from `LC.02`'s row instead of a hard-coded 0.25 the row
does not certify. Downstream: `LF.04` and `LF.06` truly need it and are both
parked, so the re-buy inside order J1 costs nothing extra. `T0.12` (GPU-hour
accounting) checks the week key against itself; Saturday's hours file under
the week Kaggle has already closed. It gains a boundary probe in the same
commit that fixes the key (order S3); `T0.24` and `T0.33` re-buy. `W1.02`
(outcomes have resolution) detects an injection as large as the outcome's
own mean; the injection is re-registered at the smallest gain a W1 claim
must resolve (order J7). `W1.04` truly needs the new clause; `W1.01` shares
only the horizon constant and is simply re-run. `LC.02` is sound as what it
is — a throughput floor; the action-reading guard belongs to `LC.09` and to
J1's forager race, where it is on. `LF.02` is sound (restart fidelity only).

## 2026-11-15 15:55 UTC — Director, FULL sitting (called by the owner the day `docs/PLAN.md` was written): rulings

**Ruling 10 — the plan is adopted; what stops and what is paused.** `docs/PLAN.md`
binds. Stopped today: new cells of the forager race (the four kernels in flight
finish and are written up as a PILOT; no seat moves on them); the humid and DRINK
world races and the W1 energy-clock race (none had a published prior; each is a
`RESEARCH:` line); new specs with no prior. `D1.0` is PAUSED, not abandoned: PLAN
stage 3 says the human-like body "gets its review before any more compute is
spent on it", the week's Kaggle GPU meter has 0.88 h, and its cached units keep.
It resumes, changed or unchanged, when the Researcher's body section says what
the record predicts for its four arms. Rulings 7 (parking behind foraging) and
the 06:40 freeze note are superseded: `LF.06`, `LC.09`, `CR.01`, `LF.04` re-runs
are "to earn again" on the brain stage 2 chooses. No in-house learner is a
candidate until it has a known-answer row. No threshold moved.

**Ruling 11 — a seat held BY RECORD.** A seat may read BY RECORD when three
things are on its row: the `PUBLISHED_PRIORS.md` section that supports the
design; a known-answer row showing OUR build of it reproduces a published
number (design on paper is not enough — the forager race was lost by builds that
were not what they were named); and the route by which a challenger contests it
(a challenger needs its own prior and its own known-answer row, then a step-4
race). BY RECORD ranks below BY VERDICT: any verdict in his own world replaces
it. Nobody is seated BY RECORD today — no build has a known-answer row. Order S8
teaches `champions --check` the three conditions.

**Ruling 12 — `PS.02` v3: remove the confound, do not censor it.** The cause
(11-15 08:42): random walks pass a fire, which prolongs one life to a death the
felt-cold probe cannot predict; `_fire_explains` is true on all 16 lives, so any
censoring threshold is a knob chosen after seeing the outcome. The claim is
"cold is felt before it kills". v3 runs the probe's lives in a world with no
fire, so cold alone decides each death and nothing is censored. Bars, seeds and
the frozen-temperature control unchanged; v2's reading stays on the row as
reporting-only. Whether fire's warmth is felt is `W.5`'s question, not this
spec's. Downstream (`LF.01`, `SH.01`, `SH.02`, `XL.01`) need only the verdict.

**Ruling 13 — `W1.01`'s bar and pin.** The finding stands (its bar is a
granularity of ~0.04 G where `W1.02` now measures ~1 G of detection power at
n=48). It is not re-registered today: W1's venue is under research (clock
`RESEARCH:` line) and a bar set before that lands would be set twice. `W1.01`
stays honestly stale; nothing may cite it as current.

**Ruling 14 — budget classes that cannot run.** Ruling 4 covers `BD.01`: its
class is changed to where it really runs (`kcpu<12h`), clauses untouched.
`LC.08` is NOT re-bought: it certifies a property of the in-house learner,
which is a suspect until J2's known-answer row; its October PASS stays stale.

**Ruling 15 — the owner's six machine ideas (INBOX 10:48).** (1) Stalled-first-
order alarm: ADOPTED as practice, not machinery — PROGRESS "Dead or stuck" names
the first order whenever it has no verdict and no named wait for 24 h; no clerk
is built (the machine may not grow). (2) Owner requests have a clock: ADOPTED —
PROGRESS leads FOR THE OWNER with any request older than 48 h and why; the
video is that line today. (3) Fewer note commits: ADOPTED — one state+journal
commit per slot is now in the ORDERS header; the fifth machine number (commits
that changed Jack, his world or a verdict / all) is reported on Sundays. (4)
Lessons expire: ADOPTED — three entries re-read each Sunday. (5) Proposals for
the owner: ADOPTED as a PROGRESS section when there is one. (6) An owner idea
box file: DECLINED — INBOX already carries his words to every lane the same
day; a second file is a second place to miss them.

**Ruling 16 — smaller items.** The lab's `steer` rule (fed 14/40 with the want
given): a one-line NO as a forager's exploration for now — the record's
exploration (pink or held noise) is tested first in E0, and `steer` may return
at step 4 with a prior (bacterial chemotaxis is published) against the E0
winner. OUTSIDER's "he cannot see food" is answered by measurement (`56bf1dc`:
rays hit floor food on ~20% of random decisions); its four other faults are the
record's own table and are fixed by orders J2 and J5. The owner's questions
"only 120 iterations?" and "how do we know the winner is the best brain?" are
answered by PLAN stage 2: 1,000,000 decisions each, runs sized by a 20-run
pilot, a rule written first — and the 15-life race is now called a pilot. The
`where` marker on `coverage` is SYSTEM Next-up 1. `LC.02`'s salt replay
diverging on wall-clock metrics is a real flaw in the replay (it measures box
load): SYSTEM Next-up 6. The tree-wide dirt guard and `launch_detached` are S7.

## 2026-11-15 16:57 UTC — Director, FULL sitting (third of the day, called when review 2 was filed): rulings

**17. The world and the harness come before any brain.** Review 2
(`PUBLISHED_PRIORS.md` §3, §8) shows two faults that would void a perfect
brain comparison: a floor meal is 8% of the bar where every published
teachable world gives 44–100% (so a sustained life needs the hardest skill
first), and the statistics were never checked against themselves. Orders are
re-ranked: J1 the world test card on today's world (do-nothing, random under
four kinds of noise, the privileged forager, a forager using only his senses,
the shuffled-ray control; the rover's own intake arithmetic; the
self-termination question), J2 the difficulty dial as wrapper options with
today's world at d = 1, J4 the harness's own checks. No brain is compared in
his world until J1, J2 and J4 are done. Brain work that needs no world (B's
and A's known answers) runs beside. Nothing here moves a bar: judging stays on
fixed layouts at d = 1, and `LF_04_FAIL.md`'s 5-meal bar is carried unmoved.

**18. J3's (was J2's) known-answer rows are §5's.** The 15:55 order's
"Pendulum ≈ −157 by 20k, pass ≥ −200" came from a one-run table. The public
multi-seed record is the mean of all 100 training episodes, −298.4 ± 10.3
(13 seeds); a 5-seed mean is accepted in −314.7 to −282.1. No run was made
under the old bar, so this corrects a reference at its source, before any
result; it is not a threshold moved after a reading. The LayerNorm arm is SBX
`SAC` (SB3's does not expose it); the standard-task checks live in their own
Python 3.12 environment under `/data`, and the project venv keeps the MuJoCo
his world is certified on.

**19. Stage 2's rule, written now from §8, before any run exists.** Primary
outcome: decisions until W consecutive lives average ≥ 5 meals in normal
living, unfinished runs censored at T; one run is one observation. Primary
test: A against B, permutation log-rank, two looks at 10 and 20 runs per arm,
each at 0.025; Boschloo on runs that crossed by T as the sensitivity check;
PPO comparisons with Holm. Futility: at look 1, ≤ 1 run past the mark in both
arms reads "not learnable at this budget". Gates: the floor arms (random,
statue, untrained twin) ≤ 1 success in total and the scripted forager must
pass, else the comparison is void. Fresh confirmation seeds, never the
tuning seeds; no seed replaced; a crash or NaN is an outcome. **"Not
distinguished" means: choose on cost and simplicity** — written in advance.
W, T and the seed list are fixed by the pilot and committed before the
confirmation runs. If A's T4 cost forces it, T is lowered for ALL arms; the
number of runs is never cut. J4 implements the outcome code against this text.

**20. Paused work the record now answers.** `D1.0`: §4's two direct tests
(SimBa, BRO) found a bigger actor does not help and a bigger critic does; our
void attempts pointed the same way. No more GPU goes to the 57M arena. The
control-path seat becomes the first candidate to be held BY RECORD once S8
lands and the holder's build has a known-answer row; if a verdict of our own
is wanted later it is §4's clean version (DeepMind Control humanoid-walk,
pass mark 600), ordered at stage 3 with the body. `T2.01`: failing since
August is what published one-world learners do below about a million steps;
no more single-world walking attempts with plain learners — the reflex-
controller recipe on the T4, or the 2018 CPU recipe, at stage 3. The spec
stays registered and red; nothing is rewritten so it passes. The diary's
word-matching retrieval: §6 says keys from learned state; stage 3, "Next up" 7.

**21. Organs.** (a) Idle slots: a worker slot that can advance no order
takes one brain-independent reproduction from JACK "Next up" 1 — a published
number, a row in `KNOWN_ANSWERS.md`, no spec. (b) S6 (the owner's video) was
four days unstarted because it was one large order behind others; it is
split into three steps and heads the SYSTEM lane with step 1 sized to one
slot. (c) The simulator's integrator is a world change: J6 measures and
lists the certificates it would stale; the Director rules on the switch with
that list in hand, not before.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-16 06:41 UTC — Director, DAILY sitting: rulings

**22. Stage 2's mark is strengthened to a sustained life, and the entry rung
is d0 = 0.75.** Ruling 19 was written before J1 measured the world. J1: a
life is sustained only at ≥ 70% of perfect harvest (about 37 floor meals per
1800 s at d = 1); a forager averaging 6.9 meals clears "≥ 5 meals" while all
200 of its lives end in hunger. So, strengthening only: the PRIMARY outcome
is decisions until W consecutive lives are each SUSTAINED — the life reaches
the time cap AND its food intake over the life is at least its basal drain
(intake ≥ drain, computed from `drives.py` at the rung being run; at d = 1
that is the 70% line). The 5-meal mark (θ = 5) stays in the harness as a
reported secondary, "first signs of eating"; it decides nothing. Everything
else in ruling 19 stands (log-rank, two looks, floors, fresh seeds, no run
count cut). The harness gains the sustained mark as a second outcome with
its own edge-case tests (J8 item 3 uses it; check #10 is re-run on it).
The gate: "the scripted forager must pass" is read with the PRIVILEGED row e
as the ceiling arm — it proves the mark is reachable in the world, which is
all a ceiling arm can prove. Proof that his SENSES suffice is a separate
stage-1 item (ruling 23), and stage 2 is not run at a rung where that proof
is still negative. Entry rung: J2's rule as written names d = 1, where
nothing lives; §3 says a rung is accepted only where random "neither always
fails nor already succeeds". The rule gains its missing lower clause: held-8
random must reach the cap in ≥ 1 and ≤ 20 of 200 lives, with do-nothing ≤ 1
of 200 and ≥ 1 bout in ≥ 50% of lives as before. On the four tables already
in (this clause is written AFTER they were read — stated plainly; it selects
the HARDER of the two settings that qualify, 0.75 over 0.5): **d0 = 0.75**
(random capped 2/200, statue 0/200, 69% eat). d ≤ 0.25 fails (statue capped
6 and 24 of 200); no rung below 0.5 is used until the dial's easy end is
repaired (instant re-placement must not land food on a body that stands
still) — not ordered now, the ladder needs only 0.75 → 1.

**23. J1's learnability answer is NEGATIVE at d = 1, and no third ruler is
tuned.** `f` (6/20) and `f2` (13/20, tuned after `f`, fresh seeds) both
missed 18/20. A third hand-tuned forager on the same bar would be a search.
The proof now comes the way PLAN names it: the clone of a forager inside his
own network (E2b) and the learner given the direction to food (E2c), at d0,
plus J8 itself (a plain published learner in his world). ROADMAP M3 records
the negative.

**24. The fair chance for the seven in-house brains (owner, 11-15 19:31).**
Adopted as asked. They have no published scores, so "passes a row" is:
reaches the row's published band within 3× the published learner's steps on
≥ 3 of 5 seeds, its own never-trains control failing — rule committed in
J3(4) before any run. Passing arms enter the pilot after B on the same
seeds, length and rule, twice each (own settings; §1's settings). A failing
arm is set aside with its numbers and may return repaired. PROGRESS reports
when each half happens.

**25. One hook in `w0.py`, for the dying body's observation.** Stage-1 orders
may not edit world files; this is the exception, because the alternative is
a world model that learns a false picture of death (§5, §8 check 8). Terms:
default byte-identical, the world's random stream untouched (identity proof
with the option off AND on), one commit, the staled certificates named and
re-bought by regate. Ordered as J9.

**26. B may be piloted now.** B's step 3 is its two SAC rows (1 and 3), each
reproduced with a failing control, row 1 on 15 fresh seeds. Rows 2 and 4 are
the PPO reference and rows 5–7 are next week's; none is a precondition of a
pilot that moves no seat. The pilot (J8) fixes W and T; the decisive test
still requires every row.

**27. The owner's five asks of 11-15 evening, ranked (auditor finding 5).**
(1) B's pilot in his world → J8, top of the lane, today. (2) The fair chance
→ J3(4), second (ruling 24). (3) The walking layer → J10, a side job on
Kaggle CPU that takes no slot from J8/J3/J5. (4) The ready-list as stage
1's deadline → adopted; PROGRESS carries the list with done/not done. (5)
The MLX port and the Mac first-day script → "Next up" 1 and 2, NOT open
orders today: the lane holds seven, the instrument is half built, and the
port's own first step (golden files) wants J5's runner working. The port's
stage 0 opens the day J5's speed figures land; the script is due 12-15.
Nothing is authorised to be spent. The strong-slot rule binds over waiting
harvests (finding 3): it was ordered and not followed at its first firing;
it now says so in the order of work. Why a burning `D1.0` dispatch was not
stopped when ruling 20 closed its arena: nobody was ordered to stop it; a
ruling that closes an arena now names the in-flight job and who stops it.

**28. The scout's two nominations (11-16 sweep).** N1 (coloured noise at
action selection only; 30 seeds; gSDE trails white noise on sparse tasks):
yes, as two reported-only arms inside J8 — it is published evidence on the
exact fork §5 left open, and it contradicts the exploration B reproduced row
3 with, so it is measured in his world before the decisive test fixes a
default. It moves no seat. N2 (a published smell learner): no run; a
`RESEARCH:` line — the Smell row of the reference, stage 3. LAB: nothing new
ALIVE since ruling 16; the label-free ear pilot is DEAD at its own bar and
no order had asked for it.

**29. Who owns a settled FAIL now that the review queue is closed.** The
queue was the owner `coverage` recognised; draining it (11-15, 11-16) left
33 FAILs unowned and `coverage` exiting 2. The baseline is not raised. An
open order owns them (S9) and gives each a fate: `repaired_by`, a
`FAIL-DISPOSED` note citing the ruling that answered it, or an INBOX ask.
S8 steps down to "Next up" to keep SYSTEM at three.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-17 06:50 UTC — Director, DAILY sitting: rulings

**30. Brain B's first pilot is VOID, the negative inside it stands, and the
next rule is written from the record and a learner-free floor card — never
from the 35 runs.** The VOID came from two of our own rules disagreeing:
ruling 22 lets a do-nothing body reach the cap in up to 1 of 200 lives, the
pilot's floor clause allowed 1 crossing in 1,198 floor lives; a body born
touching food is fed by contact. Nothing of J8 is re-run and no W or T is
taken from it. What stays on every page that says VOID: brain B crossed 0
times in 2,350 lives where the floor rate predicts 7.8, and ate less the
more it learned. Two causes are separated and repaired in this order.
(a) The statue fed by contact — a world rule (`spawn_clear`), repaired
first because it needs no learner to verify. (b) The body charged for
moving — the record's cure (`PUBLISHED_PRIORS.md` §13: ramp the cost with
the dial; then pay for distance closed; then a longer discount), each as a
5-seed reported arm. The next pilot (J11) fixes its W from a floor card of
≥ 2,000 learner-free lives per floor arm measured BEFORE the rule, sizes its
floor clause as a Poisson 95% bound on that card's rate, and runs on seeds
401 and up. It chooses the setup the decisive test uses; it moves no seat.
The decisive test's own rule is written after J11, before its first run.

**31. The Kaggle CPU queue is ranked.** Four sessions. Order for a free
session: any kernel under one hour; J11; J12; J3's in-house arms (one
session); the MLX gate G0; J10's smoke; `W1.04`'s remaining seeds. `W1.04`
yields between seeds when anything above it waits. A reported-only arm
never holds a session that a ranked order waits for. Reason: on 11-16 two
arms of an already-void pilot and a 15-hour repair held all four sessions
while four orders waited and two slots could not steer.

**32. The box, asked twice (11-16 12:10, auditor 11-17 finding 2).** The
unit is the whole run, not the cell. Under 15 CPU-minutes: the box, nice
19. 15 to 60 CPU-minutes: the box only when all four Kaggle sessions are
held, one worker, nice 19, stated in the order's state line. Over 60
CPU-minutes: Kaggle or it waits — the in-house arms (2.4 to 7 CPU-hours
each) never run on the box again. The box serves paying customers (card
line 6); "when Kaggle is full, use the box" is not a default.

**33. Known-answer rows: one extension, written before the first seed; and
row 4 is closed.** Rows 1, 2 and 4 each missed at 5 seeds and were
re-registered at 15 against an unmoved band. Nothing was hidden and B's
seat does not rest on the second looks (row 1's own arm was in band both
times; row 3b is 5 of 5), but a retry the first rule did not name is a
second look. From today every row's rule names its seeds and at most one
extension before any seed runs; brain A's rows (J14) are written that way.
Row 4 (the PPO reference): the published code itself gives 76.4 with 7 of
60 collapses on this box against the store's 87.5 ± 4.2 at n = 10; ours
72.6 against that code's 73.4 at 40 per arm. It is closed as "the reference
does not reproduce its own stored number here; ours matches the reference
code". Not re-registered at 100 per arm: three Kaggle CPU-hours for a row
no seat rests on. It is reported as a miss of the stored number, not as a
pass.

**34. Simulator speed: the identical option is used now; the integrator is
not switched yet.** `lazy_contact_audio=True` is field-for-field identical
and is passed by every new run. `integrator="implicitfast"` is a world
change (trajectories part at the first decision; injury deaths move) and
would stale 22 certificates. J11 runs on the same integrator as J8 so the
two pilots can be read side by side. The switch is ruled once, when J6's
table is complete (Kaggle column, the one-native-call item), and before the
decisive test's rule is written; if adopted, the world test card (J1) is
re-run on it as its certificate.

**35. `NE.01` is parked to stage 3, with a date.** Its two red gates are
heat: 5 of 5 random deaths are hyperthermia, and the scripted forager died
of heat on 1 of 3 seeds. Keeping warm is a stage-3 need under PLAN ("eat,
drink, keep warm" on the chosen brain); stage 1's world for the brain test
carries food only. S9 writes the `FAIL-DISPOSED` note. It is re-read at the
first stage-3 sitting or on 2027-01-15, whichever comes first; the eight
specs it blocks wait with it. This is a parking, not a pass.

**36. The researcher's two answers and the reference's size.** §13
(movement cost) is adopted as J11's prior. §14 (smell): our three smell
kills tested the one configuration the record says does not learn (a
two-point nose, no memory, 13k–52k decisions). No run now; when smell is
next tested, in stage 3, it is §14's form — the same human nose re-encoded,
a brain with memory, at least 500,000 decisions, a shuffled-nose control.
The 1 m multi-point nose is not adopted as his body (END-FIT). The
reference's cap: 6 KB per section stays; the whole-file cap of 70 KB is
lifted, because a worker reads one section and thirteen sections cannot fit
it. No older section is cut to make room.

**37. The lab's eye probe: yes to the 128-ray row, with the forager gate.**
At the entry rung a level row of 128 azimuths finds floor food .909 against
.528 for today's 16 rays; a 16×16 grid does no better than 16 rays. J12
builds `row128` as an option (default off). Ruling 23 barred a third
hand-written forager on the OLD eye; it is lifted for the new eye only,
because this is a different sense, not a re-tuned ruler: the same forager,
the unmoved bar (at least 5 floor meals a life on at least 18 of 20 fresh
seeds, shuffled twin must miss), frozen before the run, at d0 and at d = 1.
`row128` against 16 rays under brain B waits until J11 shows B moves.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## D56 — RESOLVED BY ARMED DEFAULT (fired 2026-11-18 by the Director; class conduct, desk-executable, decide_by 2026-11-21): NO SPLIT IS ASSUMED. Jack dispatches smallest-first, treats a Kaggle GPU quota refusal as final for that week and overflows to Colab; nothing of the other project is touched, paused or metered. `D1.0` no longer waits on it (closed by the record, ruling 20). Executed as ruling 47 below: J14 launches A1, A2, A3, then A's smoke in his world, then the ear seeds, in that order, from Sat 11-21 00:00 UTC.

## 2026-11-18 06:42 UTC — Director, DAILY sitting: rulings

**38. His eye is body-frame.** Today's 16 rays and the new `row128` are cast
at fixed WORLD azimuths (`fast_sim.py:157`, no yaw term): a free compass the
human-like body will not have (END-FIT row 1; lab `99e588f0`; auditor 11-18
finding 1). J12 builds `jack_env(eye_frame="body")`, default off, identity at
yaw 0. From today nothing a brain learns is believed on the compass eye:
every learner arm of J11's pilot, and the stage-2 choice, see the body-frame
eye. J12's compass-eye gate rows (d = 1: 20/20; d0 in flight) stand as
"compass eye — ceiling only". The frozen gate is run again on the body-frame
eye, declared before any body-frame run: same forager, same bar, shuffled
twin must miss, fresh seeds 261–280, d0 and d = 1, no extension; if the
forager's steering assumed the world frame, only the mapping needed to steer
from body-frame rays changes, written down first. Any memory, map or homing
result earned on the compass eye is earned again without it.

**39. Pilot 2's world and its card, ruled before the rule is written.** The
statue leak's cause is known (a food laid on the ramp rolls; `adf457ca`), so
it is repaired by its cause: the pilot's world has `food_floor_only=True`;
`spawn_clear` R stays 1.55 m. The card the rule reads is measured on THAT
world: f0..f7 all run (do-nothing, held-8 with movement free, held-8 with
movement charged); nothing is borrowed from the leaky card and no ring is
filtered out of any card. The leaky c-card stays on the page as history.
Each arm's floor is named in the rule (B0, P, G against held8-k1-ff; K, KP
against held8-k0-ff; K with learning rate 0 against held8-k0-ff as a bound,
said so). `native_substep=True` in the pilot and every new run: identical
world, 0/300 decisions differ on the box and on Kaggle. Cost: the rule
freezes about a day later than option (b). Floors are learner-free, so
ruling 38 does not touch the card.

**40. The integrator is not switched.** J6's table is complete: identical
world 2.39× (lazy audio + native sub-steps), `implicitfast` 5.64× but a
different world (22 certificates to re-earn). Under brain B the packed
learner runs 11–18 decisions per CPU-second against the simulator's 170, so
the simulator is no longer the clock and the switch would buy under 10%.
RK4 stays through stages 1–2. Re-opened only by a measured profile in which
the simulator is over half of a learner run's CPU.

**41. `W1.04` is parked, honestly.** It queued eleven hours and died without
running a seed; its ERROR row stands and the orders page no longer says "in
flight". No session before J11's pilot is harvested; re-read 2026-11-25. A
detached job still queued on Kaggle after two hours gives up its box seat
(SYSTEM next-up 2).

**42. `PS.02` and its family stay red until stage 3.** Ruling 12's no-fire
cure does not move the probe (R² 0.290 → 0.309, bar 0.50; `441ac2f1`); the
same generic probe fails `PS.05`, `PS.06`, `PS.08` whose world gates pass.
A new probe chosen now, after seeing which one succeeds, is a failing spec
rewritten until it passes. At the first stage-3 sitting (or 2027-01-15) the
probe procedure is pre-registered once for the family: rule first, seeds
that are neither 90 nor the registered 0–2, bars unmoved, and the same probe
on the non-thermal inputs as the control that must fail.

**43. Three settled FAILs asked of the Director (S9).** `T0.17`: the detector
is right — `T2.14`'s PASS was bought under claim words since amended; its row
is demoted by the registry's own drift mechanism, not re-run on the old core
(ruling 10), and `T0.17` re-earns. `T0.23`: no conjunct changes; it is red
because `run ratchets` is red, so S9 makes the ratchets green honestly (the
6 unattributed GPU jobs attributed or named, the fallen floor recorded).
`T0.32`: stays FAIL; none of the 16 long-run impls is grandfathered; each
gets `rtf.require_feasible` in the commit that next re-runs it.

**44. The home-made brains' fair chance (J3).** The frozen rule is unchanged
and both rows run for `dreamer-xs`, `wm-dyn`, `wm-latent+ring`. `wm-efe` gets
no session: on these rows it is `dreamer-xs` tensor for tensor (115
identical, its 5 heads uncalled), and its row says so. The fair chance the
owner asked for is then a REPAIRED entry under the same rule on fresh seeds
— noise held at action choice and γ 0.99, §15's named cure — first for
`wm-latent`, the only arm that learned.

**45. The walking layer reproduces a walk that has a published number
first.** §16: the 2018 helper-force paper states no seeds, success rate or
test of walking. J10's first row is SAC with the Zoo's Humanoid-v4 entry
(pass: return ≥ 5,950 AND ≥ 12 m forward per 1,000 steps at 2M; check at 1M
in [3,784, 5,866]); the 2018 row stays, third, with a per-stage exit-reason
log.

**46. The reference's size.** Already ruled (36): 6 KB a section, no
whole-file cap. The researcher's 11-18 ask is answered by it.

**47. D56 executed** (above): smallest-first on the shared free GPU.

**48. The ear starts its reproduce step on the free GPU.** BC-ResNet-1 is
sized (9,232 weights = the published 9.2K; corpus and official test set
reproduce; target 96.9 ± 0.30). It runs inside J14 after brain A's rows, from
scratch, rule first, shuffled-label twin. Nothing pretrained goes inside
him; this is a known-answer row for a published design, PLAN step 3.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## 2026-11-19 06:41 UTC — Director, DAILY sitting: rulings

**49. His motor is body-frame, like his eye.** Asked 11-18 07:40 and 12:26;
no sitting answered, and the worker froze pilot 2's rule (`e2c3c378`) on its
written default. Ruled now, on two grounds that do not depend on either
floor card: (i) END-FIT row 1 — the finished Jack has a human-like body,
which has no world-axis push and no world-axis velocity sense; (ii) gate 2's
own row h4, frozen before it ran — body eye + world motor carried the hand
forager on 0 of 20 worlds (0.81 meals a life), so a world motor under a body
eye asks a learner to aim a compass it cannot sense. The m-card's lower
floor is NOT a ground (auditor 11-19 finding 1): the frame would be body had
the body-motor floor come out higher. Consequences: the frozen rule stands
unchanged; floors and W come from the m-card, the same frame as the arms
(W = 2 on both cards); J14 (b)'s `--motor body` default is confirmed; every
new run in his world passes `motor_frame="body"`. The lesson is the
Director's: a ruling asked "before the rule" must be answered inside one
sitting, or carry a default in the order — this one had the worker's.

**50. J15's E2b is not run as frozen; the stop clause meant the entry
rung.** The frozen rule (`760ae70a`) discloses "target held 0.993"; the
training set reads 0.551 (seeds 701–712). On the other 45% of decisions the
teacher steers along a random ray held 25 decisions, which the student's
input cannot determine; the record (§17: Weihs 2021 0.88 → 0.00–0.11; BeT
2022 0.67 → 0) predicts a memoryless clone averages and fails. A miss would
say nothing about his eye — the one thing the row exists to show — so it
does not take a 9-hour session. No test seed (741–760) has been read. In
its place, declared before any data: a ruler `f5` whose action is a function
of the current body-frame row only; gate 2 again on f5 at d0 on fresh seeds
281–300, same bar, shuffled twin; if f5 passes, it is the teacher and the
clone rule is otherwise unchanged; if f5 misses, that is the finding (a
memoryless reading of this row does not forage at d0) and E2b closes by
measurement. This is a re-registration for a wrong instrument, not a
weakened bar: the forager bar and the clone bar are untouched. "If J12's
body-frame gate FAILS" meant d0, where E2b lives; it passed there. E2c:
bearing as cos/sin (radians/π jumps at ±π); the zeroed input is the primary
control and the random point is reported beside it (it leaks odometry); the
rule says before its first seed that a miss at 300k is weak evidence.
Students train inside the Kaggle kernel (the 11-18 12:48 default).

**51. What the full-rung failure means.** On the body-frame eye the hand
forager passes at d0 = 0.75 (20/20) and fails at d = 1 (3/20 against 18).
"Learnable from his senses" is therefore shown at the entry rung only, by a
dead-reckoning ruler that was not re-tuned and is a lower bound. Stage 1's
pilot and stage 2's decisive test are judged at the entry rung; d = 1 is
reached by the dial as he succeeds (stage 3), which is what the record's
worlds do (PRIORS §8). No bar moved: the d = 1 row stays a FAIL in
`WORLD_CARD.md`. ROADMAP M3 and PLAN's table say so from today.

**52. The free GPU week from 11-21, ranked; brain B's row 7 is dropped.**
Order: A1, A2, A3, A in his world, the ear's three seeds and twin, then
extension seeds a frozen rule calls for. Row 7 (recurrent SAC) gets no GPU
hour: its citation was wrong at source (`69935227`), the paper gives the
task only as a figure, the authors' data is a 404, and it costs ≥ 32 CPU-h
a seed. A known-answer row needs a published number (PLAN step 3); a band
read off a figure is not one. B's memory is tested in stage 2's battery;
if a recurrent B is entered there it first reproduces a row with a tabulated
number, which is a RESEARCH question, not an order.

**53. The price of a shared box.** 12 of 61 ladder slots on 11-18 were
refused on load that was not Jack's. Planning number: 19 worker slots a day,
not 24. The gate is right and stays; nobody touches the other projects. A
launch refused on load retries inside its slot, then the next slot takes it
first. Ruling 32's 15–60 CPU-minute class now also requires load < 4.

**54. S9 holds the SYSTEM lane** until ruling 43's fates are executed and
`run ratchets` exits 0, except a fix a running Jack kernel needs.

**55. The reference's size, and §18.** The researcher's file cap (70 KB)
cannot hold 18 sections at 6 KB; the cap that matters is what one worker
reads. It becomes "≤ 6 KB a section; a worker reads its section" (S11 b).
§18 moves "Other senses" to step 1 (research filed); its two cheap items
are Next-up 2 (the needs-hidden known-answer) and 3 (one key per sense for
brain A, a design note). The rest is stage 3.

**56. After pilot 2, and how J3 ends.** Whatever pilot 2 reads, the next
learner run is E2c (a learner handed the direction) on arm K's world: it
separates "cannot find food" from "cannot learn to eat", which is the
reading the pilot alone cannot give. No other learner run is launched on the
pilot's result before a sitting. J3 ends with `wm-latent+ring` and the one
repaired entry: a pass extends the repair to the other arms; a fail sets
every arm aside with its numbers on record. MLX gate G0 passed; the port's
rung 1 is order J16, box only.

## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*


## LG.13 — WINNER — meaning-mass
meaning-mass beats topk-softmax by 4.13 sigma and clears the null by 56.00 sigma.

metric: `match_both`  ·  null 0.192 ± 0.014  ·  gate mode: `screen`

> **screen rationale** (why these arms are observables, not learners): The arms are OBSERVABLES, not learners, and the rig makes that structural rather than asserted: every arm is a deterministic function of ONE already-cached verdict table (1588 frozen log-probabilities, content-hash keyed) that no arm can perturb, and no arm has a parameter fitted to anything. A low score is therefore a property of the RULE and not evidence that its run broke, which is exactly the case `validity` mode would mis-VOID. The gate is unmoved at 3 sigma and MIN_FINISHERS still applies. Declared in advance and not expected to bind: LG.10's published incumbent reads match 0.60-0.78 against a 0.18 null, so the mode is on the record before any arm number exists rather than switched on after a VOID.

| arm | mean | sigma over null | gate | admitted | cost |
|---|---|---|---|---|---|
| meaning-mass | 1.000 | 56.00 | pass | — | 1.0 |
| topk-softmax | 0.850 | 18.13 | pass | — | 2.0 |
| softmax-full | 0.694 | 8.59 | pass | — | 1.0 |
| topk-uniform | 0.661 | 32.52 | pass | — | 1.0 |
| control:state-free-prompt | 0.064 | -5.02 | FAIL | — | 1.0 |

*admitted `—` = no admissibility predicate was supplied; nothing was checked and nothing is asserted.*

