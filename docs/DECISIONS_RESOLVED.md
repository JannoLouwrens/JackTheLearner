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
