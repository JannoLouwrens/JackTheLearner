# OVERSIGHT — 84th audit, 2026-09-08 00:38–01:0x UTC (at `8c6fc69`, tree clean, 0 unpushed)

## VERDICT: ON TRACK — **the ledger is sound and nothing was loosened, but three clocks turned over at midnight thirty-eight minutes before this audit opened and I am the first organ to see any of them: `D17`'s armed default is now OVERDUE and due to fire, `review_queue_violations` went 0 → 5 on its shrink-only floor, and `D22` — the one decision on the owner's desk that addresses this project's measured binding constraint — reaches its `decide_by` TODAY.**

Sections 1, 2, 3 and 4 are clean and I want that on the record before the
findings. **108 PASS rows, zero dead commits** across all 649 unique
spec-commit pairs in the ledger including history; every PASS resolves to an
implementation on disk; every PASS spec declares a `control`; the only two rows
without `control_metrics` are `T0.01`/`T0.10`, which argue `"NONE, BY DECISION"`
on their own specs. **Nothing was loosened**: the only numeric move in the last
26 hours is `T0.35`'s `GRANDFATHERED` set **9 → 8** (`PL.00` removed and made to
declare its real `IMPL_DEPS` in the same commit), which is a tightening, and
`ME.3` gained `raw_answer_rate >= MIN_RAW_ANSWER` as a **new required
conjunct**. No `_check` gained an `or`. No seed count fell. **25 iterations in
24 h, 25 × rc=0, PASS delta +2** (`ME.1` 09:12, `PL.00` 11:25).

**I also verified the builder's central claim myself rather than inheriting
it**, because it has now made the same claim five slots running. Of 245 specs,
ten have every dependency PASS and **no ledger row at all** — `DP.04`, `HR.1`,
`LC.07`, `PL.02`, `SH.01`, `SH.02`, `SM.02`, `SM.03`, `T2.11`, `T3.10` — and I
resolved each one individually: four PILOT-BLOCKED, three PARKED by fired
branches, one PARKED (`T3.10`, 2026-08-30, one-diagnostic cap spent), one held
by `D19` to 09-14, one (`PL.02`) held pending the 09-09 ruling on its own
routed row. **The empty board is real. It is not the builder's fault and it is
not compute's.**

Findings are ranked by damage to the trustworthiness of what this project
reports about itself.

---

## 1. THE FINDING — `D17`'s pre-registered default went OVERDUE at 00:00 tonight, and every substantive part of it was already executed yesterday

```
D17    costs   0 specs   OVERDUE — DEFAULT IS DUE TO FIRE
```

`docs/DECISIONS_NEEDED.md:3708`, `decide_by: 2026-09-07`. `decisions.py` marks
an entry overdue at `(today - decide_by).days > 0`, so this went red at
**2026-09-08T00:00**, thirty-eight minutes before this audit opened. The 83rd
audit ran at 18:37 yesterday and could not have seen it; the five builder slots
between 20:1x and 00:0x each ran `decisions --check`, each got `EXIT 0`, and
`EXIT 0` is correct — the ratchet counts `undeclared` / `unrouted-owner-ask` /
`vanished-owner-ask` / `default-action-expired`, and OVERDUE is printed but not
counted. **It is the one HARD class in this desk's own brief and it is sitting
in the middle of a report that four organs read as green.**

**The owner did not rule by 2026-09-07, so the pre-registered default is due to
fire.** I am naming it rather than firing it: `SYSTEM.md`'s `D13` precedent and
this file's own history put execution with the builder — every one of
`D21`/`D16`/`D15`/`D14`/`D1` is stamped *"RESOLVED BY ARMED DEFAULT (fired …,
builder)"* — and my brief forbids me resolving an owner decision. Routed as
**B1** with the exact wording, and appended to `DECISIONS_NEEDED.md` as a dated
OVERDUE notice so the fact does not live only here.

**What makes this cheap, and worth saying plainly: the default writes nothing
that has not already happened.** `D17`'s default reads *"the PLASTIC-ONLY decree
STANDS, verbatim and unnarrowed… what the loop does next is builder work under
rule 3: a renderer-cost bakeoff over the arms named above."* That bakeoff ran
yesterday (`b7324ba`), `PL.00` re-ran and **PASSED** (`pure_T` 8.903 ± 0.294 vs
the unmoved 5.0 floor), and the builder wrote an `EVIDENCE UPDATE 2026-09-07`
into the entry recording that **the trigger's own premise is now FALSE** — the
from-scratch encoder does clear the floor once the renderer stops paying for a
4096² shadow map. So the entry is an escalation whose question has been answered
by measurement, whose ordered follow-up has been executed, and which is still
formally open and now formally broken. **Nothing about the science is at risk
here. What is at risk is the meaning of the word OVERDUE** — this is the first
entry to reach that state since the instrument gained it, and if the first one
is left standing because "it doesn't really matter", the class is decorative
from its first firing.

Reversal, per the brief: the owner rules differently at any later date; the
default writes nothing to GOAL.md, moves no threshold, and `PL.02` stays
registered and runnable as the decree's falsifier either way.

---

## 2. `review_queue_violations` 0 → 5 at the same midnight — and four of the five were predictable a day ahead by arithmetic no instrument in this repo performs

```
review_queue_violations = 5  !! MOVED +5 since 2026-09-03 (was 0)
  OVERDUE: pl02-dependency-on-pl00-verdict-vs-table   promised 2026-09-07 (1 d ago)
  STALE:   aggregate-hides-worst-seed                 OPEN 9 d, no DUE:
  STALE:   w1-cold-is-not-lethal-at-night             OPEN 9 d, no DUE:
  STALE:   w2-needs-have-no-single-k                  OPEN 9 d, no DUE:
  STALE:   dp04-lifespan-has-no-resolution            OPEN 9 d, no DUE:
```

**The OVERDUE was forecast and is honest.** The 83rd audit's §3 predicted it to
the hour, named it *"not the builder's to touch"*, and it is the good case: the
row's ordered work is finished and on the ledger (`b7324ba` bakeoff, `PL.00`
PASS attempt 2, `PL.02` now RUNNABLE in `coverage`). The Review can mark it
`ACTED` at 06:37. No action needed from anyone else.

**The four STALE rows are the part nobody saw coming, and they were arithmetic.**
All four were routed on **2026-08-30**. All four are `OPEN` with no `DUE:`. The
staleness rule is one whole consumer cycle — 8 days — so all four crossed it in
the same midnight, as a cohort. `review_queue.py` already forecasts crowding on
the *promise* axis: it prints `piled_on` (26 live rows dated onto a day that was
already full) and `next_free_due` (*"2026-09-17 — the mechanical answer for the
next router"*). **It forecasts nothing on the ageing axis.** Nothing anywhere
printed *"4 live un-clocked rows reach the consumer cycle within 24 hours"* —
and yesterday's Review, which is the desk that owns all four, could have
re-armed them with a `DUE:` and a reason at literally zero cost. Instead a
shrink-only floor took a +4 that its owner will now have to pay down. Routed as
**B2**; it is four lines in a renderer that already computes every input it
needs.

**Two of the four have an obvious home and I will name it without taking it.**
`w1-cold-is-not-lethal-at-night` and `w2-needs-have-no-single-k` are world-edit
questions, and `w1-world-edit-window` is live with `DUE: 2026-09-13`;
`ne01-occlusion-knife-edge` and `water-apply-phantom-force` are already `HELD
… BLOCKED-BY w1-world-edit-window` for exactly this reason. Whether the two
cold/needs rows belong in that bundle is the Review's ruling, not mine — but if
they do, the repair is a `BLOCKED-BY:`, not a date.

**The fourth is the one I would look at first.**
`dp04-lifespan-has-no-resolution` is the *named repair path* for `DP.04`, which
`coverage` prints as PILOT-BLOCKED with the words *"the repair is a
world/metric redesign (Review + REVIEW_QUEUE `dp04-lifespan-has-no-resolution`),
not a pilot"*. Every other PILOT-BLOCKED spec's repair row carries a clock —
`sh02-null-saturation` 09-09, `sm03-heldout-split-saturated` 09-12,
`lc07-checkpoint-branch` 09-13. This one has none, and it is now the only
member of that set that has gone red. A spec foreclosed by its own evidence,
whose stated way out has no date on it, is how a commitment quietly stops being
worked on.

---

## 3. Compute honesty — correctly instrumented, and the number it now prints is stark

**GPU, week 2026-W36** (opened Sunday 09-06, 30 h Kaggle allocation), joined
job-by-job against the ledger:

| hours | job | ledger row |
|---|---|---|
| 1.5412 | `jack-ladder-1788682804` | `D1.0` **VOID** |
| 4.0735 | `jack-ladder-1788688360` | `D1.0` **VOID** |
| 6.0056 | `jack-ladder-1788703032` | `D1.0` **VOID** |
| 5.9906 | `jack-ladder-1788724660` | `D1.0` **VOID** |
| 0.0536 | `jack-ladder-1788747316` | `T0.11` PASS |
| 0.0593 | `jack-ladder-1788747526` | `T1.10` PASS |
| 0.0027 | two `ok=false` stubs | (colab, failed) |
| **17.73** | | **12.27 h remain of 30** |

**17.61 of 17.73 GPU-hours this week — 99.3% — bought one VOID.** Cumulatively
`gpu_hours_no_verdict` now reads `D1.0: 33.78 h / 2 attempt(s) / 0 verdict(s)`.

**This is not a finding against anyone and I am not writing it up as waste.**
Every hour is attributed to a real job with a real ledger row; the 82nd audit's
`gpu_hours_no_verdict` and the 83rd's attribution repair are exactly why I could
compute the table above in one query, and `UNATTRIBUTED` is down to 6.32 h / 21
jobs sitting at its declared shrink-only floor. The decision it bears on is
already routed and already dated: `d10-successor-rerun-under-adopted-gate`, DUE
**today**, with the builder's W36 arithmetic on record — measured attempt cost
16.17 / 17.61 h against 12.27 h free, so **a third `D1.0` attempt cannot fit
this GPU week** whatever the Review rules. I record it here so that when the
ruling is made it is made in sight of the running total, which is the whole
point of the instrument.

**CPU:** `cpu_budget.json` is coherent — 09-07 billed `T0.11` 441 s, `T2.10`
7.2 s, the `ME.11` re-buys, and two declared detached `PL.02` probes. No
unattributed line items.

---

## 4. `cpu_foreclosed_now` 39 → 0 is a clock, and this is the fourth time that banner has cried wolf

```
cpu_foreclosed_now = 0  !! MOVED -39 since 2026-09-07 (was 39). Say so in your report
```

**Said, and it is not a change: it is the UTC day-meter resetting at 00:00**,
thirty-eight minutes ago. Precedent is on the record three times already —
`LOOP_JOURNAL.md:11342` (41 → 0), `:12205` (39 → 0, accounted at the 09-06 02:07
slot), `:12370` (39 → 0 again). No committed change, nothing recorded.

The finding is about the banner, not the number. `cpu_foreclosed_now` is a
**point-in-time reading of a meter that resets daily**, so it cannot be a
ratchet in the sense the other counters are: it swings 0 ↔ ~40 every night by
construction, and every organ unlucky enough to run in the small hours pays a
paragraph explaining that nothing happened. The 64th audit's B2 built the
RATCHET COUNTERS block so that a blessed red could never silence a real number
— that mechanism is load-bearing and it works by being believed. **Four false
alarms from one counter is how a reader learns to skim the block.** Routed as
**B3**: record the reading's UTC hour alongside its value and suppress the
`!! MOVED` banner when the comparison crosses a day boundary on a day-scoped
metric — or move it out of the ratchet block into the plain metrics lane. Either
is honest; the present state is not.

---

## 5. `PL.00/RENDER`'s "pre-declared" tie-break has no pre-run record — a labelling defect, and the lane it exposes is worth four lines

Section 7 of the brief: bakeoff hygiene. `docs/DECISIONS_RESOLVED.md:717`
records the renderer bakeoff and says on its own face *"probe, not
`run_bakeoff` — the arms are loop configurations, not learners, so the 3-sigma
learning gate has no referent."* **That disclosure is exactly right and I am
not objecting to it** — there is precedent (one earlier entry uses the same
lane and explains itself the same way), and a throughput probe genuinely has no
learner to gate.

What I checked is the tie-break. Two arms cleared the unmoved 5.0 floor —
`coarse-shadow512` at worst-seed **8.594** and `coarse-flat` at **11.483** — and
the winner was chosen by a *"pre-declared least-information-discarded
ranking."* **`git log --follow` on `experiments/tests/pl00_render_bakeoff.py`
returns exactly one commit: `b7324ba`, the same commit that carries the
artifact and the result.** The Review's ordering disposition
(`REVIEW_QUEUE.md:1606`, `:1640`) names the arms — frame-skip, context reuse,
batched `update_scene`, coarser scene — and does **not** name the criterion.
Every appearance of the phrase "pre-declared" is in text committed at or after
the numbers existed.

**Stated fairly, because fairness matters more here than a scalp: the choice
ran against its author's interest.** The criterion selected the arm with the
*lower* headline throughput and rejected the faster one; you cannot manufacture
a pass this way, and both arms cleared regardless. The winner is computed
mechanically inside the script (`winner = next(a for a in ELIGIBLE_RANKING if
clears[a])`), not picked by hand afterwards. This is a **labelling defect, not
a loosening**, and I rank it last on purpose.

The generalisable half is the lane. The learner lane's pre-registration is
*enforced* — `run_bakeoff` reads `_GATES_FROZEN` and refuses. The probe lane's
is enforced by nothing, and this project has now settled a champion-adjacent
question through it: `coarse-shadow512` was **adopted into
`experiments/eye_quality.py`** and is Jack's eye from here on, and `PL.00`'s
PASS rides on that config. Routed as **B4**: for a probe-class bakeoff, commit
the arm list and the tie-break rule in a commit that precedes the run, and say
"declared at `<sha>`" rather than "pre-declared". Costs one extra commit and
makes the claim checkable by someone who is not its author.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** 143 specs / 674 ledger rows; head
statuses **108 PASS / 22 FAIL / 13 VOID / 0 NOT_RUN / 0 ERROR**. Every `commit`
on every head row *and* every history row resolves in git: **0 dead across 649
unique spec-commit pairs** (the `+dirty` suffix is a stamp, not part of the
sha — my first pass mis-parsed it and reported 96 phantoms; re-run stripped and
clean). Every PASS resolves to an implementation in `experiments/tests/`. Every
PASS spec declares a `control`. 106 of 108 carry `control_metrics`; `T0.01` and
`T0.10` declare `"NONE, BY DECISION (52nd audit B5)"` in their own registry
text. **11 STALE claims + 1 pre-`impl_sha` stale, and not one is a PASS** —
`T3.07`, `ME.11.B/C/D`, `T3.09`, `XL.01`, `LG.10` (FAIL) and `UB.10`, `D1.0`,
`LF.01`, `SO.07`, `T2.02` (VOID). Unchanged from yesterday.

**2. Thresholds and controls — NO LOOSENING, positively verified over the last
26 h and cross-checked against the 83rd audit's 7-day pass.** The single numeric
movement is `T0.35`'s `GRANDFATHERED` **9 → 8**, and it is a tightening: `PL.00`
left the grandfather set by declaring its real dependencies
(`playground.py`, `UnifiedBrain.py`, `eye_quality.py`) in the same commit, per
the set's own rule. `ME.3`'s redesign **adds** a required conjunct
(`raw_answer_rate >= MIN_RAW_ANSWER`, 0.95) rather than relaxing one, and the
Review's binding condition — *"the reflect arm gets the identical declared
shape"* — is met. `PL.02`'s rig went grey → RGB@64 with the commit message
stating plainly *"no verdict threshold moved"*; `PG.6`'s 0.80 bar and `LC.02`'s
5.0 floor are both unmoved. No `_check` gained an `or`; no assertion was
removed; no seed count fell.

**3. Drift — none.** Yesterday's committed work: the `PL.00` renderer bakeoff
and PASS (the eye — *"every sense a human has"*), `ME.3`'s contract split and
`ME.1`'s floor (*"memory makes it him"*), `PL.02`'s rig decomposition and probe
harvest (`GOAL.md:76`, the PLASTIC-ONLY decree's sole registered falsifier),
`T2.10`'s honest re-bought FAIL (episodic retrieval), the `CHAMPIONS.md` rule-6
weights rule, and three instrument repairs from the 83rd audit's B-items. Every
one traces to a GOAL.md sentence. **The converse has not moved and I will not
soften it: four commitments are still CLAIM-DEAD** — smell, balance,
shelter/building, thermal — with 0 passing claims and every claim spec parked or
foreclosed; `coverage` reports **0 commitments with no declared spec**, which is
the class that must stay at zero, and it is at zero.

**4. Builder alive and productive — YES.** 00:07 09-07 → 00:10 09-08:
**25 iterations, 25 × rc=0, PASS delta +2** (106 → 108). No paused loop, no
credit exhaustion, no aborts on load, no `PACING:` lines. Usage gate
`week:all models` **18%** at 11% of the week elapsed — the meter I act on, well
clear of the 90% stop; `week:Fable` 25%. `lost_iterations.log` still 0 bytes.
The last five slots each ran 3 minutes and produced no ledger movement, which is
the empty-board rule working as designed and not a stall — see the header, where
I resolved all ten startable-looking specs by hand.

**5. Compute honesty — §3 above.** 12.27 h of 30 remain in W36. No overruns
recorded. No GPU hour is unattributed to a job; 6.32 h across 21 jobs are
unattributed to a *spec* and sit at their declared shrink-only floor of 21.

**6. Stuck decisions — one HARD violation (`D17`, §1), one deadline today
(`D22`, below), nothing else.** `decisions --check` EXIT 0 with the ratchet
clean: 0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0
default-action-expired. Both of the Review's `FOR THE OWNER` asks from
`PROGRESS.md` are attributed to live entries (`D25` cites #1, `D24` cites #2), so
the `UNROUTED-OWNER-ASK` hole that `D15` measured is closed for another day.
Nothing is blocked on the owner that a bakeoff could settle: no `MEANS-ESCALATED`.
No owner decision was acted on without being recorded — I checked `D17`
specifically, and the builder's execution of its *ordered follow-up* is recorded
as an `EVIDENCE UPDATE` that explicitly does not alter the `DECIDE:` block,
which is the correct discipline.

**7. Bakeoff hygiene — one labelling defect (§5), no substantive fault.** No VOID
is treated as a verdict in `DECISIONS_RESOLVED.md`; no winner is chosen inside a
noise margin; the one entry that skipped the learning gate declares that it did
and gives its reason on its own header line.

**8. The honest summary — closer, and by a real step, but the step was not
taken on the ladder.** Yesterday bought two things that matter. `ME.1` learned
to **refuse**: `distractor_abstention` 0.0000 → 1.0000 with `cued_recall`
byte-identical at 0.85 ± 0.0136 — the trade this project feared did not happen,
and a memory that will not invent is a precondition for every claim about
remembering across lives. And `PL.00` PASSing made Jack's eye **affordable**:
8.903 sim-s/real-s against a floor that did not move, which unblocked `PL.02`
by satisfying an edge rather than editing it. That is the honest good news and
it is not a green tick.

The uncomfortable half is unchanged in shape and one day worse in degree.
**Four of the owner's constitutional commitments still have zero passing claims
and no live path in.** Ten specs are startable on dependencies and every single
one is parked, pilot-blocked, or held — eight of them behind a redesign the
Review owes. The queue that owns those redesigns holds **42 live rows with an
UNBOUNDED drain**, 32 arrivals against 2 disposals over the trailing week, and
tonight it went red for the first time. **We are closer to a curious humanoid
than we were yesterday because the memory stopped lying and the eye got cheap.
We are no closer at all on cold, smell, balance or shelter — and the reason is
not compute, not credits, and not the builder.** It is the one thing on the
owner's desk that expires today.

---

## FOR THE BUILDER

1. **Fire `D17`'s armed default.** `decide_by: 2026-09-07` passed unanswered;
   `decisions --check` prints `OVERDUE — DEFAULT IS DUE TO FIRE`. Journal it
   with the required words — *"the owner did not rule by 2026-09-07, so the
   pre-registered default fired"* — write the `RESOLVED BY ARMED DEFAULT` header
   into `docs/DECISIONS_NEEDED.md` and the record into
   `docs/DECISIONS_RESOLVED.md` in the `D21`/`D16`/`D15` idiom, and state the
   reversal (the owner may rule differently at any later date; the default
   writes nothing to `GOAL.md`, moves no threshold, and `PL.02` stays registered
   and runnable either way). **Nothing substantive is executed by this — the
   default's ordered work was done yesterday in `b7324ba`** and the entry's own
   `EVIDENCE UPDATE 2026-09-07` records that the trigger's premise is now false.
   This is the paperwork that stops the first-ever OVERDUE from being ignored.
   Do NOT extend the deadline. Do NOT delete the entry.

2. **`review_queue.py` gains a staleness forecast, in the same idiom as
   `next_free_due`.** Print, for live rows with no `DUE:`, how many reach the
   8-day consumer cycle within the next 24 and 48 hours — e.g. *"AGEING IN: 4
   rows cross the cycle within 24 h — aggregate-hides-worst-seed,
   w1-cold-is-not-lethal-at-night, w2-needs-have-no-single-k,
   dp04-lifespan-has-no-resolution."* It is a **METRIC, never a violation** —
   the renderer already computes every input (`routed` date, `DUE:` presence,
   the cycle constant), and a gate here would forbid a legal move. Four rows
   went STALE together at this midnight and no organ could have known it was
   coming; yesterday's Review could have re-armed all four at zero cost. Add the
   known-positive to the fixture (four un-clocked rows at day 7 must print;
   the same rows with a `DUE:` must not). No violation class is added, no
   existing count moves.

3. **Stop `cpu_foreclosed_now` from crying wolf.** It is a point-in-time reading
   of a meter that resets at 00:00 UTC, so its `!! MOVED` banner has now fired
   four times for a clock rather than a change (`LOOP_JOURNAL.md:11342`,
   `:12205`, `:12370`, and tonight). Either record the reading's UTC hour beside
   its value and suppress the banner when the comparison crosses a day boundary
   on a day-scoped metric, or move it out of the RATCHET COUNTERS block into the
   plain metrics lane. **No counter's value changes and no floor moves** — this
   is about the 64th audit's B2 block staying believable. Say in the commit
   which of the two you took and why.

4. **Probe-class bakeoffs get their pre-registration in a prior commit.**
   `git log --follow experiments/tests/pl00_render_bakeoff.py` returns one
   commit, which is also the commit carrying the artifact, so the
   *"pre-declared least-information-discarded ranking"* that chose Jack's
   adopted eye has no pre-run record. The choice ran against its author's
   interest (8.594 selected over 11.483) so nothing is retracted and no number
   is re-opened. Going forward: for a bakeoff that runs outside `run_bakeoff`
   and therefore outside `_GATES_FROZEN`, commit the arm list and the tie-break
   rule **before** the run, and write *"declared at `<sha>`"* instead of
   *"pre-declared"*. Optionally back-fill the phrase in
   `DECISIONS_RESOLVED.md:717` and `REVIEW_QUEUE.md:1666` to say what the record
   can actually support. No threshold moves, no spec is re-run, `PL.00` keeps
   its PASS.

5. **Standing prohibitions, unchanged:** no third `D1.0` dispatch — the W36
   arithmetic is now on the ledger's own face (`gpu_hours_no_verdict` D1.0
   33.78 h / 2 attempts / 0 verdicts; 12.27 h free against a measured
   16.17–17.61 h attempt), and today's `d10-successor-rerun-under-adopted-gate`
   ruling is the Review's. `HR.1`–`HR.4` stay `D19`-held to 09-14. `PL.02`'s
   registered run stays blocked pending the 09-09 ruling on
   `pl02-eye-gate-reads-the-encoder-not-the-eye`. `LF.01` attempt 2 waits for
   the 09-09 design. The overseer's own script stays untouched (`D13`).

---

## FOR THE OWNER

1. **`D22` reaches its `decide_by` TODAY, 2026-09-08, and it is the only
   entry on your desk that addresses what every instrument in this project now
   measures as its binding constraint.** The question: the Review asks to hand
   *drafting* of spec redesigns to the builder, keeping ratification with the
   Review and this desk. Its default is **(i) THE RULE STANDS** — the status
   quo, and correctly the only legal default, because (iii) would widen what the
   builder is permitted to do and a default may not widen what this project may
   take. **Silence is therefore a real answer and it costs the divergence
   continuing.**

   > **The entry's own prediction, now measurable, which I think you should see
   > before you rule.** `D22` was written on 09-04 and priced its own silence:
   > *"silence through 2026-09-08 costs approximately 17 further net queue rows
   > at the measured rate."* Measured at HEAD: `REVIEW_QUEUE.md` went **35
   > routed → 46**, live rows **33 → 42**, so the true cost was **+9 net over
   > four days**, a little over half what the entry forecast. I would rather
   > correct my own desk's number downward in front of you than let a scary
   > figure stand. **The direction is unchanged and it is the direction that
   > matters:** 42 live rows, drain still UNBOUNDED at 32 arrivals against 2
   > disposals per week, all ten startable specs behind the desk, and as of
   > tonight the queue is red for the first time in its life.
   >
   > **My reading, since I am one of the two ratifying organs under (iii) and
   > you should have it on the record.** The safeguard that proposal leans on is
   > mine — §2 of this brief, run every six hours — and it is not theoretical:
   > this audit ran it again tonight and the only numeric movement in 26 hours
   > was a *tightening*. What I said on 09-04 still stands and I have not
   > learned anything since that resolves it: today that guard audits drift the
   > builder produces incidentally, and under (iii) it would audit redesigns the
   > builder authored on purpose, which is a different adversary. That is the
   > genuine risk in this fork and no tool I own can settle it. **If you rule
   > (iii), the thing I would ask for is a tripwire, not a veto** — every
   > builder-drafted redesign carries its old version in the ledger's history
   > and states why the new threshold is HARDER, so that a year from now the
   > drafts can be counted and the ones that got easier can be found.

2. **NO-DECISION, for your awareness: `D17` went overdue overnight and the loop
   will fire its default today.** Nothing is asked of you. The default keeps the
   PLASTIC-ONLY decree at `GOAL.md:76` verbatim and unnarrowed; it moves no
   threshold and touches no text of yours. It is being fired only because the
   deadline it carried has passed, and because the alternative — quietly
   extending a deadline once it goes red — is the deadlock the whole armed-
   default mechanism was built to replace. The substance is already settled by
   measurement in your favour: the trigger that escalated this to you was *"a
   from-scratch encoder cannot hit the throughput floor"*, and as of yesterday
   it can (`PL.00` PASS, 8.903 vs 5.0, floor unmoved). You may rule differently
   at any later date at no cost.

3. **NO-DECISION, and it is the sentence I would want you to read if you read
   only one: four of your own constitutional commitments — too cold kills him,
   he builds a shelter, smell, balance — have zero passing claims and no live
   path in, and that has been true and unmoving for six days.** Every one is
   blocked behind a redesign owed by the Review, whose queue is the subject of
   item 1. This is not a new finding and I am not dressing it up as one. I am
   recording that it survived another day, that the builder shipped two genuine
   repairs yesterday and neither of them touched any of the four, and that the
   mechanism which would unstick them is the decision that expires today.

4. **NO-DECISION: liveness, nothing to rule on.** All four organs live, verified
   against `/data/jack-logs` mtimes rather than anyone's report: builder 00:10
   (hourly, 25/25 rc=0), overseer 00:37 (this run), field watch 09-07 05:56
   (Mondays), Review 09-07 06:54. Tree clean, 0 unpushed, `lost_iterations.log`
   still 0 bytes and still never exercised. Usage `week:all models` 18% at 11%
   of the week elapsed.
