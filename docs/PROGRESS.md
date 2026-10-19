> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: docs/PROGRESS.md itself last moved 29h ago against a 25h cadence; PROGRESS_LOG.md's fresh row is the dying run's own B4 disclosure and cannot vouch for the page
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-10-19T12:37:03+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

Written by the Sunday FULL Review of **2026-10-18**. Current-state by design:
anything not carried forward here is either committed elsewhere or routed to a
`D` id. Every act on this page was committed BEFORE the page was written —
the page is the receipt, not the work.

## Part 1 — the state of progress

### The numbers, recomputed this sitting, never quoted

| reading | value | note |
|---|---|---|
| registry | **257** specs | |
| ledger rows | **160** | so **97 registered specs have no row at all** |
| PASS / FAIL / VOID / BLOCKED | **107 / 35 / 17 / 1** | |
| pass rate vs registry | **41.6%** (107/257) | Aug 7–9 read 40.0% → 38.3% |
| pass rate vs rows run | 66.9% (107/160) | |
| rework (attempt > 1) | **79%** (127/160) | |
| specs RUN this week (10-11→10-18) | **5**, of which 4 PASS | T0.02, T0.12, T0.21, T0.36 |
| specs RUN last week (10-04→10-11) | **11**, of which 5 PASS | |
| `ledger.json` commits, this week / last | **15 / 43** | |
| consecutive dark builder slots | **14** (14.7 h) | hourly cadence → 7× the 2× trigger |
| `review-queue` | **0 violations**, OVERDUE **6 → 0** | all six disposed this sitting |

**Goodhart check, and it is the week's most important number.** The pass rate
did NOT fall — 41.6% against August's 38.3%. But **all four of this week's
PASSes are `T0.*`**: Tier-0 instrument self-tests, re-bought mechanically by
`regate`'s rotation lane, which by construction cannot advance `GOAL.md`. Count
rising, rate rising, and **zero frontier movement**. The honest reading is that
the rate is now measuring the re-buy lane, not the ladder. A rate that improves
while the frontier is still is not good news; it is the metric detaching from
the thing it was built to track.

**The frontier.** The single most important unblocked item is `t205`'s ordered
`T2.14 → T2.05` port (DUE 10-25): zero GPU hours, zero world edits, zero new
instruments, and it converts the project's longest-standing unattributable FAIL
into an attributable one. The builder is not working on it, because the builder
is not working at all — 14 dark slots. Effort-vs-goal for the week: of the 15
`ledger.json` commits, **15 are mechanical re-buys**. No commit this week served
`GOAL.md`'s current stage.

### The honest paragraph — no numbers

We are busier, not closer, and this week the machinery that proves it got
better while the creature stood still. The instruments are in excellent health:
the queue's violation class is empty, the certificate lane re-derives itself
twelve times a day, and this morning the oldest cohort of passing tests was
audited and came back cleaner than anyone predicted. That is real and it is not
nothing — a project that cannot trust its own green ticks cannot learn
anything. But every one of those improvements was an improvement to the
measuring apparatus, bought with slots the builder could not spend on Jack. The
single most important step toward him was discovering, by accident, that a guard
we already invented for one test had never reached its sibling, and that this
absence is precisely why our world model's defeat by a straight line sat
unexplained for six weeks: we had the instrument the whole time and never
carried it next door. The most concerning drift is that the thing holding the
builder dark is not a bug anyone can fix, and we have now spent two consecutive
weeks' worth of free compute without a decision about whether to spend it —
the failure is not that the hours expire, it is that nobody is able to choose.

## Part 2 — the test re-examination (the gates sweep, discharged)

Run as the sweep that `gates-that-measure-something-other-than-what-they-say`
part (a) ordered, against its own three named shapes, oldest-passed first:
**T0.19, T1.02, T2.03, T2.04, TA.02, T3.01, T2.19, T2.09, T2.14, T0.18, T0.09,
T0.10, T1.09, T1.06**, plus **T0.22** and **T2.00** for the battery shape.

**The result is not the one the row predicted, and that is the finding.**

- **Shape 3** (a declared control the check never compares against the
  headline): **ZERO instances.** Checked against the declared `Spec.control`
  field, never prose. `T0.10` reads as controlless and is not — its control is
  an explicit `NoControlByDecision` carrying the 52nd audit's B5 reasoning.
- **The battery shape** (`properties_failed == 0` passing a battery that stopped
  early — T0.13's first bug): **already guarded, correctly.** `T0.19` and
  `T0.22` both pin `properties_checked == N_PROPERTIES` on the experiment **and**
  the control, and both require the control to fail on SPECIFICALLY NAMED
  properties. **`T0.19` is the pattern every battery in this repo should be read
  against.**
- **The capped-teacher shape**: `T2.04` and `T2.14` carry **more** guards than
  `LG.03` did — `nn_informative_all`, `clip_overlap_max > 0 → VOID`,
  `floor_selftest_ok`, `task_not_trivial_all`.

**So the class is real but it is not population-wide rot — it is UNEVEN
PROPAGATION.** `T2.14` gates on `task_not_trivial_all` (*a linear map must not
annihilate the task; a good score here would be arithmetic, not imitation*);
`T2.05` measures the same shape against `mse_ridge` and **has no such
conjunct**. That absence is exactly why `T2.05`'s FAIL was unattributable for 43
days and why `t205` broke four dated promises: its two offered repairs were
never two choices, they were one unmeasured quantity.

### STRENGTHENED / REWRITTEN

| spec | what changed | why it is STRONGER |
|---|---|---|
| **LG.03** | `87fc826`. Both LIVE comments that still taught the refuted teacher-caps-twin mechanism replaced with the justification that survives (an un-aceable calibration cell is an invalid LIVENESS VENUE, not a bound on the student). Refutation left visible in place. | The docstring recorded the refutation since 09-12 while the comments at eye level taught the falsified claim — a future desk auditing the load-bearing conjunct ORDER would have inherited it. Comment-only, verified by `git diff -U0 \| grep -v '^[+-]\s*#'` returning nothing. `PLANNER_CALIB_MIN` stays 1.0, `CALIB_MIN` untouched, conjunct order untouched. Bill: 0 certificates (LG.03 is VOID). |
| **T2.05** (ordered, not yet executed) | Port `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok` pair; report `mse_nonlinear` and `nonlinear_headroom`. DUE 10-25. | ADDS a VOID gate with a working self-test and REMOVES an unearned inference. No threshold moves: `WM_RATIO` stays 0.8, `CTRL_TOL` stays 0.98, ridge gate stays ratio-free, claim conjunct byte-untouched. Bill: 0 standing PASS certificates. |

Nothing was weakened. No threshold moved downward anywhere this sitting.

## Part 2.5 — steering maintenance

- **OVERDUE class: 6 → 0**, every row disposed by act, stamp or dated re-arm.
  No row deleted, no `DUE:` dropped, nothing relabelled `HELD`.
  `standing-pass-certificates…` **ACTED** `4a3e646` · `t032…` **ACTED**
  `ac51b29` · `t205` **RULED → DISPOSITIONED** 10-25 · `gates-that-measure…`
  **DISPOSITIONED** 10-26 (part (a) discharged) · `lg03-teacher…` **ACTED**
  `87fc826` · `pl02-void-gate…` **re-dated 10-27 as a holding action** → `D49`.
  All six now declare `WAITS-ON`; **2026-10-17 has dropped out of
  `review-queue`'s WITHHELD list entirely.**
- **ORGAN LIVENESS.** Overseer live (06:37 today, on cadence). Field watch live
  (10-12 Monday, next due 10-19). Review: this sitting — the previous four days
  were gate-deferred and the 10-17 DAILY died `rc=124`. **Builder: fires hourly,
  has not iterated in 14 slots.**
- **Seat staleness / anatomy audit:** 32 declared seats over 257 specs; the
  2026-08-09 scar gaps are all now SEATED (smell, taste, voice, body schema,
  emotion, person model). Standing findings unchanged and NOT re-routed:
  `NO-ARENA` on ASR and Speaker ID, `TRIGGER DEBT` 3/3 (Learning core, Fast/slow
  coupling, World). As in the last five sittings, every match reads "not
  progressing" because nothing is progressing — that is a reading of the
  blackout, not of the arena, and it is not routed as a seat finding again.
- **PRIORITY RECONCILED — AUDITED CLEAN, THEN AN ATTEMPTED EDIT WAS REVERTED,
  AND THE REVERT IS THIS SITTING'S LAST FINDING.** The live `1^22` block was
  audited for the dangerous failure mode (a FALSE live order at the top of the
  builder's page): **none found.** Its cached GPU figures were already excised
  on 10-17, it points at living sources, and its one status claim verifies —
  `t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign` is
  still OPEN, DUE 2026-10-21, exactly as written. **But this sitting's two new
  builder units were not on the page**, so a `1^23` block naming them was
  written — and then **reverted by its own author**, because `steering` priced
  it: the insertion took `scripts/ladder_prompt.md` to **130,290 bytes, 782
  below the 131,072 EXEC CLIFF**, past which the builder does not read a
  degraded prompt — **it does not start, and the slot looks like an ordinary
  `rc=126`.** Reverted; the page is back to 126,721 B, 4,351 below the cliff.
  **The two orders therefore live in FOR THE BUILDER on this page only**, which
  is a weaker channel than the builder's own prompt and is named as such rather
  than papered over. **Also recorded, because the instrument said it and this
  desk did not act on it:** `steering` flags the item-1 order
  `SUBJECT T2.14 BLOCKED — T1.08 (FAIL)`. The ORDER is still legitimate — it is
  a spec-file port, not a run of `T2.14` — but whether the re-run that pays its
  bill can execute today is NOT established by this desk, and the builder should
  derive that before spending a slot.
- **NOT DONE, named rather than left to look done:** `FIELD_WATCH.md` was
  **not** consumed (it has not changed since 10-12 and its week-10 findings were
  all routed by the 10-13 sitting, so nothing is owed there). The sitting spent
  its clock on six OVERDUE disposals, the Part 2 sweep and two committed
  strengthenings.

## FOR THE BUILDER — ordered, cheapest and most perishable first

1. **Port `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok` into `T2.05`,
   and report `mse_nonlinear` + `nonlinear_headroom`** (DUE 10-25, bill 0
   certificates). Do NOT gate on the headroom number on this pass — report it.
   Then `T2.05`'s 43-day-old FAIL becomes attributable: headroom large ⇒ the
   venue has structure the WorldModel missed and the model is indicted; headroom
   ≈ 0 ⇒ the venue is near-linear and the VENUE is the defect, at which point the
   bundling rule applies because that repair is world-touching.
2. **Choose and execute one option from
   `regate-bought-nine-certificates-from-its-own-uncommitted-tree`** (DUE 10-24,
   bill 0). This desk recommends **(i)+(iii)**: refuse to run on a dirty tree,
   and make `--list` print the stamp it will buy under. Nine certificates were
   bought `+dirty` in one 14-hour window and no instrument floors on DIRTY.
3. **`T0.22`'s guard comment says "All nine ran" while `N_PROPERTIES = 15`** —
   stale by six, no effect on the gate. Deliberately NOT acted on alone (it
   would stale a certificate for a comment); ride it along with the next edit
   that bills T0.22 anyway.
4. `longrun_unbound` still holds `T0.32` at FAIL over ten specs — its own row,
   DUE 10-19, untouched and not pre-empted by this sitting's `t032` stamp.

## FOR THE OWNER

**1. `D46` is unchanged and still open — carried forward, not re-asked.** The
reporting-floor recommendation stands exactly as quoted into `D46` (`class
conduct`, `decide_by 2026-10-17`, now past). Nothing in this sitting changes it.

**2. Routed this morning as `D48` (`class conduct`, `decide_by 2026-11-01`):
the Sunday sweep you ordered found the defect class already guarded, and found
the real one by accident.** An instrument built to the original description
would have found nothing in the oldest fourteen certificates. The defect that
was actually measured is uneven guard propagation. Recommendation quoted
verbatim in the entry; **the default is the conservative branch — no exemption,
no reader** — and this desk will not re-ask it if the default fires. Nothing is
blocked on the answer: the `T2.14 → T2.05` port is already ordered.

**3. Routed this morning as `D49` (`class constitution`, `decide_by
2026-10-27`): a queue row whose only permitted dispositions are ALL forbidden to
its only permitted dispositioner.** `pl02-void-gate-quantifies-over-its-own-nulls`
is not behind schedule — it is undischargeable as constituted. Every repair on
its menu is a loosening; `1^13` forbids the builder; strengthen-only forbids this
desk; the permitted-actor set is empty. **This is not `D47`** (capacity): this
row would still be undischargeable in a sitting with infinite time. This desk
**recommends against its own default** and says so in the entry — but a visible
refusal beats a fourth broken date.

**4. `D30`'s required pairing, counted this sitting and not inherited.** Dark
**14 consecutive slots / 14.7 h**, 0 failed, against an hourly cadence — 7× the
2× trigger. The gate acting is the **PACE LINE**, not the hard stop:
`week:all models` **66%** at 54% of the week against a line of **61%**; the 90%
stop is not engaged. `week:Fable` reads 6% and is **not** the gate.
**`2026-W42` holds 30.0 free Kaggle GPU-hours, 0.000 drawn, all expiring
Saturday 2026-10-24** — and **`2026-W41`'s entire 30.0 h expired unbought
yesterday**, the sixth consecutive week. Said with the honesty `D40` required:
these hours again have no designed buyer, so they would likely have expired
regardless. **The hours are not the argument. Two consecutive weeks of being
unable to decide whether to buy them is.**

**5. NO-DECISION: completeness audit, reported because a named gap is a decision
and an unnamed one is a blind spot.** Audited against the external human
sensory/cognitive inventory, not against our own documents. **The 2026-08-09
scar is closed**: smell (12 specs), taste (5), voice/vocalisation (25) and body
schema (10) all now have specs AND seats. The holes have moved to the cognitive
half, which is already routed as
`completeness-audit-2026-09-13-the-cognitive-half-is-the-hole` (DUE 2026-11-01)
and is deliberately not re-routed here. Thinnest, by spec count: **tool use (1
spec, and NO SEAT)**, **self-model (1 spec, and no seat distinct from Person
model, which models OTHERS)**, theory of mind (3), emotion (3 specs against
1,149 lines of `EmotionalState.py`), attention (4), working memory (4).
**Tool use and self-model are the two capabilities with neither a seat nor
coverage.** I may add seats directly and deliberately did **not** this sitting:
a malformed `CHAMPIONS.md` breaks the one instrument that reads it, and the
remaining minutes of a timed sitting are the wrong place to edit it. It is the
first act I would take next Sunday.

**6. THE BUILDER'S PROMPT IS THREE DAYS FROM AN EXEC CLIFF THAT LOOKS LIKE AN
ORDINARY `rc=126`, AND THIS DESK JUST MEASURED IT BY WALKING INTO IT.**
`scripts/ladder_prompt.md` is **126,721 B — 4,351 below the 131,072
`MAX_ARG_STRLEN` cliff**, growing **+1,358 B/day** over the last 17 commits, so
`steering` prices the cliff at **3 days**. Past it the builder does not read a
degraded prompt: it does not start, and the failed slot is indistinguishable
from any other `rc=126`. This sitting wrote a `1^23` block naming its two new
units, measured the result at **782 B of margin**, and **reverted its own
edit** — which means the orders now reach the builder only through this page.
NO-DECISION is NOT claimed: the archival that fixes this (`1'`/`2'`/`3'`) is on
its own queue row, every desk that edits this page makes it worse, and the next
routine edit by builder, overseer or Review may be the one that silently stops
the builder. **Recommendation: treat the self-imposed 125,000 ceiling as a hard
refusal rather than a reporting line, so the organ that would cross it is
refused in the open instead of discovering the cliff afterwards.** Routed
nowhere new by design — it belongs to the existing archival row, and this desk
is reporting the measured distance rather than opening a seventh entry on your
desk in one morning. If you want it as its own decision, say so and it will be
routed.

---

*Commits this sitting, in order: `4a3e646` (regate rotation lane inherited and
committed) · `56c57e9` (OVERDUE 6→5 + new dirty-stamp row) · `134f92a` (5→4,
t032 ACTED) · `96d046c` (4→3, t205 RULED) · `87fc826` (LG.03 strengthened) ·
`c77e45f` (3→1, the bound pair) · `e960fd5` (1→0, pl02 held) · `cc7d05d` (D48 +
D49 routed). This page was written last.*
