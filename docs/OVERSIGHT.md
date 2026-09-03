# OVERSIGHT — 65th audit, 2026-09-03 06:45 UTC (HEAD `f529ab1`, 1 unpushed, tree clean, no runner alive)

## VERDICT: ON TRACK — and I want to say plainly that this is not the easy call

The two audits before this one read `DRIFTING`. Both were right, and both of their
rank-1 findings are now closed on disk. Today the ledger is mechanically sound, no
threshold moved in the loosening direction anywhere in seven days, the builder ran
**26 of 26 iterations `rc=0`**, and the one live instrument red at the moment this
audit opened was **caught and repaired by a sibling organ eight minutes in**
(§ FINDING 2). Calling that `DRIFTING` to look busy would be manufacturing a
problem, and this desk's credibility is worth more than a third consecutive red word.

There is one finding that matters, and it is a **guard owed before a spend, not
damage already done**. Ranked by damage to the trustworthiness of the ledger:

| # | finding | damage |
|---|---|---|
| 1 | `HR.6` does not `depends_on` `HR.5`, so this morning's fixture refutation is invisible to every instrument — and both `coverage` and the builder's own priority list now point at `HR.6` | **prospective**: ~11 free GPU-h + a builder unit, on a question the registry calls not-well-posed |
| 2 | the ratchet reader built 5½ h ago to be unsilenceable *was* silenced by recording a reading; the Review caught the instance, the structural blindness remains | banked slack, now repaired for the instance only |
| 3 | seven live rows fall due on one Sunday desk with one recorded capacity failure (**independently found by the Review the same morning**) | seven predictable `OVERDUE`s on Monday |
| 4 | the crontab *schedules* the cross-organ doc race the queue is trying to design away; the leftover-process guard fired on the overseer's own read-only tool at 06:37 | a guard learning to cry wolf |

---

## THE FOUR MANDATORY INSTRUMENTS (rc read live, at 06:45, post-`f529ab1`)

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | **0 commitments with NO spec.** Red is the standing, routed one: **4 CLAIM-DEAD** (smell, balance, shelter/building, thermal — DUE 09-11). unreachable **89 of 225, baseline 89** — clean as of `f529ab1`; it was **90 vs 89** when this audit opened (FINDING 2). 8 commitments with live claims and nothing passing. 3 park→release pairs unwalkable, 4 cost classes with no path in. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** Ratchet 0/10 — the list is fully armed, so there is nothing for me to arm this audit. Live: D15/D16 due 09-05, D17 09-07, D18 09-09, D19 09-14. |
| `champions --check` | 0 | 27 seats. `arena_missing` **0**; unfalsifiable 3/3; uncontestable 3+1/4; unverified verdicts 2/2; trigger debt 3/3. **Every ratchet at baseline, none grown.** |
| `run review-queue` | 0 | **0 violations.** 27 OPEN / 2 HELD / 2 ACTED of 31; oldest live 10 d; consumer ran 1 d ago and is running now. See FINDING 3 for what a clean rc is not saying. |

`scripts/lib_liveness.sh` schedule half: all four organs on cadence — builder hourly
at `:07` (26/26), overseer `:37 */6`, Review `06:37` daily (running as I write),
field watch 08-31 → next 09-07.

---

## FINDING 1 (RANK 1) — `HR.5` refuted `HR.6`'s informativeness at 05:25. At 06:45 two organs have read that FAIL three times, written about it twice, and both still point the builder at `HR.6` — because no edge in the machine carries it

### The prose says it. The graph does not.

`HR.5` and `HR.6` were registered in the same commit (`1733280`, 05:25), verbatim
from `docs/research/HEARING_BAKEOFF.md`. That document is unambiguous about the order:

- `HEARING_BAKEOFF.md:1591` — *"**Growing the fixture is a prerequisite for HR.6 being informative**"*
- `HEARING_BAKEOFF.md:1861` — *"**HR.5 goes first because it can invalidate everything downstream** for 15 minutes"*
- `HR.5`'s own registry notes — *"**PREREQUISITE FOR HR.6 BEING INFORMATIVE**: with only impacts, Jack's entire auditory world is (onset, f0, level, pan) — four numbers — and a representation bakeoff on it measures how well each encoder recovers four numbers."*

And the registered edge:

```
HR.5  depends_on = ["PG.5", "PG.2"]              status FAIL   2026-09-03T05:25:38
HR.6  depends_on = ["HR.7", "PG.5", "PG.7"]      status NO ROW  ← HR.5 is absent
HR.8  depends_on = ["HR.5", "HR.6"]                            ← HR.8 carries the edge
```

`HR.8` declares the dependency. `HR.6` does not. So `HR.5`'s FAIL blocks `HR.8` and
frees `HR.6` to be listed as reachable, at a `Budget.GPU` seats-3 cost the source
document prices at **3–6 GPU-hours**.

### What `HR.5` actually measured — three absences, not a close call

```
classes_present     1.0  of 4          water_present  0.0    creak_present  0.0    roll_present  0.0
has_kind_label      0.0  (gate 1.0)    has_self_flag  0.0    (gate 1.0)
four_class_audio_separability 0.583    position-only CONTROL 0.708   ← the control OUTSCORED the claim
alive_two_pitch_acc 1.0                audio_finite   1.0    ← the instrument was alive, so FAIL not VOID
```

The verdict is honest and was pre-stated in the docstring before the run. Three of
the four sounds GOAL.md names — *"the ladder creak, the splash, the thud of his own
fall"* — **do not exist in the fixture**, and the fourth cannot occur because the
humanoid is not in the playground.

### Why that forecloses `HR.6`, in `HR.6`'s own words

`HR.6`'s arm **A5** is the hand-crafted event vector `(t_onset, f0, level, pan)`.
Its registry notes:

> *"A5 IS THE MOST INFORMATIVE ARM AND THE ONE NOBODY WANTS TO RUN… Its SUCCESS —
> matching every learned encoder — would mean the sim's audio is a 3-parameter
> family (f0, amplitude, pan) that a lookup table captures, so **no representation
> experiment run on it can distinguish anything**, and the fixture must grow
> (section 5) before the question is even well-posed."*

`HR.5` has now **measured the antecedent**: `classes_present = 1.0`, kind labels
absent, self flag absent. A5's tie is no longer a hypothesis to be tested; it is
entailed by a committed row. `HR.6`'s own `falsified_by` agrees that this branch
*"indicts the FIXTURE, not the brain, and sends the work to section 5 … rather than
to a bigger model."*

### The part that makes this an oversight finding rather than a builder note

Every instrument in this repo walks the **declared graph**. None of them read prose.
So at 06:45, with `HR.5`'s FAIL committed for eighty minutes:

- **`coverage` line 78, live:** `gpu<2h  1  UB.10  (no FRESH dispatch here) <- fillable today: HR.6`
- **`scripts/ladder_prompt.md:525`, written by the Review at 06:45:** *"**`HR.6`'s CPU staging arms are the cheapest fresh unit on the board**"* — priority item **1'**, and **`HR.5` is not named anywhere in it.**
- The same Review commit message quotes `HR.5`'s FAIL **twice**, in Part 1 (*"HR.5 FAIL exactly as pre-stated — the playground cannot make the sounds GOAL.md names"*) and in FOR THE OWNER (*"HR.5 made it six this morning in a family one day old"*).

Two organs. Three readings of the same FAIL. Both still route to `HR.6`. Nobody was
careless — **the edge that would have made this automatic was never registered.**

### The staging note is a real valve, and it does not cover this branch

Item 1' correctly quotes the registry's safety valve: *"if A2 cannot beat A0b on CPU,
the GPU arms are cancelled for free."* That gate is genuine and cheap. But it fires
on **A2 vs A0b** only. The branch `HR.5` predicts is **A5 ties everything** — and a
run where A2 beats the placebo *and* A5 ties them all would **pass the staging gate
and green-light the 3–6 GPU-hours** on the question the registry itself calls not
well-posed. The valve is aimed at a different failure than the one that was measured
this morning.

### Two smaller things on the same row, recorded so they are not lost

- `HR.5`'s headline registered metric — `four_class_audio_separability = 0.583` — comes
  from a run whose **position-only control read 0.708**. Per `SYSTEM.md` law 2 and the
  `T2.11` precedent (*"flipping `_GATES_FROZEN` … would dispatch a verdict from a run
  its own permuted control outscored"*), that number is uninterpretable and must not be
  quoted downstream. The **verdict is safe** — it is carried independently by
  `classes_present`, `has_kind_label` and `has_self_flag`, three absences the control
  cannot touch — but the row's message is the generic *"pre-registered threshold not
  met"* and says nothing about which conjunct carried it.
- `HR.5`'s FAIL is a **world/fixture refutation of a GOAL.md sentence**, the same shape
  as `DP.04`, `SH.02`, `SM.03`, `T3.06` and `BA.03` — every one of which has a routed
  `REVIEW_QUEUE` row with a `DUE:`. `HR.5` has **none**. Its repair contract (a
  sustained NOISE voice driven by persisting contact; a surface-crossing detector
  inside `Water.apply`; a `geom_bodyid` self flag) exists only in a spec docstring and
  one journal line. No clock, no owner.

---

## FINDING 2 (RANK 2) — the ratchet reader shipped at 01:13 to be unsilenceable was silenced at 06:37 by recording a reading. The Review caught the instance at 06:45; the structural blindness is still there.

### What was live when this audit opened

At **06:37**, `coverage` exited **rc=2** on this line and this line alone:

> *"unreachable count fell to 89 of 225; `UNREACHABLE_BASELINE` still reads 90 and
> must be lowered in the same commit — **the ratchet only ratchets if the floor
> follows the number down**."*

The commit that produced that state was `b8f69f4`, the **last commit of the 06:07
iteration**, whose message reads: *"Ratchets: record unreachable 90 -> 89 … shrink
direction, justified by the harvest commit."* It wrote the new value into
`experiments/ratchet_readings.json` and left `UNREACHABLE_BASELINE = 90` in
`coverage.py`. **Shrinking the reading is not shrinking the floor**, and the commit
that asserted the ratchet was serviced is the commit that left it red.

### Why nothing said so — and this is the durable half

`run status`'s ratchet block was built by the **64th audit's B2** (`210f26f`, 01:13
this morning) and prints its own purpose in its header: *"standing-red tools' numbers,
printed here so **a blessed red can never silence them**."* At 06:37 it printed:

```
RATCHET COUNTERS — … so a blessed red can never silence them (64th audit B2).
      unreachable = 89  (unchanged since 2026-09-03)
```

Quiet. `run.py:861` classifies each counter by comparing the **live value** against
the **last committed reading** in `ratchet_readings.json` — `MOVED`, `UNCHANGED`,
`LOST`, `UNRECORDED`, `VANISHED`. It never compares against the tool's **floor**.
So `run ratchets record` + commit is sufficient to make any counter read `UNCHANGED`
while the instrument it mirrors is red about that exact number. The repair shipped
5½ hours earlier missed its first real case by design, not by accident.

### Credit where it is owed

**The Review found and fixed the instance at 06:45**, eight minutes into this audit,
in `f529ab1`: `UNREACHABLE_BASELINE 90 → 89`, with a new SHRINK entry in the growth
log and the reason written down — *"The reading file is a log; THIS is the floor."*
Verified live after its commit: `89 of 225, baseline 89`, `stale_baseline` clear.
That is the system catching itself without me, which is the point of having four
organs, and it should be read as a good day.

What survives the fix: had the Review not been running, the builder's own reader
would have gone on saying `unchanged` indefinitely. **The instance is closed; the
class is open.**

---

## FINDING 3 (RANK 3) — seven live rows fall due on one Sunday desk that has already failed once at this load, and nothing in the queue reader counts a date pile

Confirmed independently by the Review the same morning — two organs converging on
this without conferring is itself the evidence that it is real.

`run review-queue` exits **0 with 0 violations**, and that clean rc is honest about
what it measures. It reports `OVERDUE` — a promise **already broken**. It has no
reader for a promise **about to break in bulk**. The live due-date distribution:

```
09-05  1     09-06  7  ← Sunday FULL     09-07  4     09-08  4
09-09  3     09-10  3                    09-11  1
```

The seven on **2026-09-06** — `w0-too-shallow` (10 d old, with two `HELD` rows and
three others behind it), both `d10-learning-gate-*` rows, `lc07-checkpoint-branch`,
`cross-organ-doc-race-voids-certificates`, `lt01-c2-body-cannot-rise`, and the
`ACTED` `me11-*` row — all land on **one** 40 min / 120-turn Review FULL run that
also owes Part 2 and both completeness audits.

The precedent is on the record: **the 08-30 Sunday FULL died at eleven minutes owing
`w0-too-shallow`'s design** — one row. It now owes seven. The builder already knows
the pile exists (the 64th audit's B4 dated `t215` *off* it deliberately, and
`REVIEW_QUEUE.md:1658` says so) — but that knowledge lives in row prose, and no
instrument computes it. A queue that only reports broken promises will report seven
of them on Monday morning.

The Review has pre-declared an order for Sunday (`d10-*` first, then
`w0-too-shallow`, then Part 2 at its minimum of 8), which is the right response to
this week's instance. It does not give the reader an eye.

---

## FINDING 4 (RANK 4) — the crontab *schedules* the cross-organ doc race, and the leftover guard fired on the overseer's own read-only tool this morning

```
7 * * * *      ladder_loop.sh      ← builder, hourly at :07, iterations routinely run past :37
37 */6 * * *   overseer.sh         ← lands INSIDE the builder's hour, by construction
37 6 * * *     review.sh           ← and at 06:37, all three organs are live at once
```

The routed row `cross-organ-doc-race-voids-certificates` (DUE 09-06) diagnoses the
mechanism exactly right — `protocol.py:82 DOC_OUTPUTS` excludes only the builder's two
docs, so an audit's in-progress doc writes stamp a concurrent runner sweep `+dirty` —
and offers three arms: (a) widen `DOC_OUTPUTS`, (b) serialise organ commits against
runner sweeps, (c) split prose dirt from instrument dirt. **What the row does not say
is that the collision is scheduled rather than incidental.** The 09-02 incident that
cost four certificates was stamped `e5dcb17` — the 63rd audit's own commit, from the
**18:37** overseer slot, landing inside the builder's 18:0x sweep. That is arm (b)'s
territory and it is cheap evidence for whoever designs it on Sunday.

Two live consequences, both from this morning:

- **06:37:26** — the builder's exit sweep logged
  `LEFTOVER PROCESS 718843 — 0s CPU, cmd: python -m experiments.coverage`. That is
  **this audit's first instrument call**, caught by the builder's boundary. The 61st
  audit's B3 repair made `run_spec` self-declare to procwatch, which covers spec runs;
  the overseer's and Review's read-only tools declare nothing, so they are
  indistinguishable from the 1.26 core-hour `while 1` scar the guard exists for. This
  is the **fourth** `LEFTOVER=1` in four days and the first that is purely an artefact
  of the schedule. A guard that fires on its own auditor teaches the loop to skim it.
- **07:07** — the next builder iteration starts while this report is still uncommitted.
  I am committing immediately and touching nothing else, but the general case is
  exactly the trap the 09-06 row is about.

---

## THE AUDIT, SECTION BY SECTION

### 1. Integrity of the ledger — CLEAN, and this is the most important line in the report

Swept mechanically over all **95 PASS rows**:

- **Implementations exist:** every PASS resolves to a file in `experiments/tests/` — **0 unmatched**.
- **Commits resolve:** every `commit` field passes `git cat-file -e` — **0 dangling, 0 `+dirty` survivors**.
- **Controls declared:** **0** PASS specs lack a declared `control`.
- **Controls recorded:** only `T0.01` and `T0.10` have empty `control_metrics`, and both declare `control = "NONE, BY DECISION (52nd audit B5)"` on their face with the reasoning attached (an ImportError and an external service's own failure *are* the falsifiers).
- **Controls actually implemented:** every PASS spec with a non-`NONE` control has a `def _control` in its implementation file — **0 exceptions**.

No finding.

### 2. Thresholds and controls over time — NO LOOSENING, and I chased every candidate

Across 8 days of commits to `registry.py`, `registry_expansion.py` and
`experiments/tests/`, I extracted every in-place constant change (same name, same
commit, different value). Fourteen hits. **Thirteen move in the strengthening
direction** — `N_LIVES 16→32`, `LIVES_PER_ARM 4→16` and `16→48`, `N_DECISIONS
3200→4800`, `N_EVAL 48→120`, `STEPS 300→500`, `COORD_MIN 0.55→0.70`,
`COORD_MARGIN 0.20→0.35`, `TEMP 0.25→1.0` (more sampler entropy makes every gate
strictly harder), `N_PROPERTIES` +1 three times.

The **one downward move** is `DECAY_MIN 1.5 → 1.25` in `44f24c41` (08-29, T2.09).
It is legitimate and I checked it rather than took it: the constant was a
**placeholder being frozen for the first time** on a spec that had never run and whose
`run()` refused until that commit; the commit message declares it in its own words
(*"ONE BAR MOVED, DOWNWARD, IN THE OPEN"*), gives the measurement that forced it
(seed 90's claim-arm static decay read 1.472, so 1.5 would have discarded a live
decaying signal), and sets the new value **from what the gate is for** — a constant
signal decays by exactly 1.0 — *not* shaved to the observed minimum. The same commit
raised seeds 3→7. That is the disclosure protocol working, not evading.

`_SEC_PER_SEED 1200→355` is a wall-clock estimate, not a gate. No control was deleted
or weakened; no `_check` gained an `or`; no seed count fell; the only `_check` edits
in the window are the `LG.00`/`LG.02` **purity repairs**, which replace live
recomputation with reads of the recorded row and move no bar.

No finding.

### 3. Drift from the goal — none in the work; the gap is in what the work can reach

**What the builder did in 24 h, and the GOAL.md sentence each item serves:**

| work | GOAL.md sentence |
|---|---|
| `HR.1`–`HR.8` registered (217→225) | *"EVERY SENSE A HUMAN HAS … sight · **hearing** · touch …"* |
| `HR.5` implemented + run (FAIL) | *"he must hear the ladder creak, the splash, the thud of his own fall"* — and it is now **measured false of the fixture** |
| `HR.7` implemented + run (PASS) | same; the guard against a stem that averages channels and *"deletes Jack's only directional sense"* |
| 64th-audit B1–B4 closure, 4 certificate re-buys | *"protects the honesty of watching what happens"* |
| `T0.13`/`LG.00`/`LG.02` purity repairs | law 1 — a claim replayable from its own row |

**Zero drift.** Every item traces.

The converse question is the uncomfortable one. **Four constitutional commitments are
CLAIM-DEAD** — smell, balance, shelter/building, thermal ("too cold kills him") — every
claim spec parked or foreclosed, and *not one of them because Jack failed to learn*.
Eight more have live claim specs and nothing passing: touch, tool use, proprioception,
death & retry, plasticity, sleep, hunger/thirst, fast/slow.

And the north star specifically: **curiosity — 12 specs, 2 passing, `0 now`.** There is
no runnable curiosity claim on the board today; `LT.01`'s FAIL holds `LT.02`–`LT.07`
and `LT.09`. The project cannot move its own thesis this week even if every iteration
is perfect.

### 4. Is the builder alive and productive — YES, and genuinely so

**26 iterations in 24 h, 26 `rc=0`, zero paused, zero credit exhaustion, zero aborts
on load.** Demonstrated 94 → 95.

That `+1` understates the day and I want to be fair about it. Of the ~70 ledger rows
written in 24 h, most are the bounded-gate sweep's regression re-stamps and four are
re-buys repairing the 19:08 doc-race. But the **new science** is real:

- **`LG.02` PASS** (first PASS 09-02 04:15) — the owner's liar test. Two advisors at
  0.9/0.1 claim accuracy, trust joined to Jack's own verification *only* through the
  attributed diary; worst-seed divergence 0.60 vs the 0.40 bar, stripped-attribution
  null 0.05, swap control migrates 0.6333, first-encounter trust exactly 0.5. That is
  *"his diary records whose advice proved true, so trust in a person can be earned and
  checked"* — moved from prose to the ledger.
- **`ME.11` FAIL** — a whole retrieval family settled with a verdict rather than left open.
- **`T3.09` FAIL** — the creative loop did not earn its existence, reported as such.
- **`HR.5` FAIL** — hearing's world half moved from *unmeasured* to *measured*, and the
  answer is "the world cannot make these sounds". A refutation bought for 8.2 seconds
  of CPU is one of the best trades on this ledger.
- **`HR.7` PASS** — the stem-deafness guard, with two false-negative modes fought off and
  documented with numbers, plus a generalisable lesson filed.

Three FAILs and one refutation in a day is a healthy loop, not a stalling one.

### 5. Compute honesty — every hour accounted, and every hour of the big spend bought a VOID

**W35 (resets Sunday 2026-09-06): Kaggle 18.93 h of 30 → ~11.07 h remaining. Colab 0.27 h.**

Every charged hour has a job id, a receipt in `gpu_submissions.jsonl`, and a recorded
outcome — the accounting itself is honest, with no orphaned dispatches. What it bought:

| spend | outcome |
|---|---|
| **16.17 h** — `D1.0` ×3 kernels | **VOID**. Owned and clocked (`d10-*` rows, Review 09-06); a standing do-not-re-dispatch order is now priority item 2'. |
| 1.10 h — `UB.10` | **VOID** (routed, DUE 09-08) |
| 0.50 h — `D1.0` pilot | envelope frozen; the escalation branch fired as pre-registered |
| 0.44 h — `LC.07` pilot | **PILOT-BLOCKED**; redesign routed (DUE 09-06) |
| 1.01 h — `T2.14` | **PASS** |
| 0.11 h — `T1.09`/`T1.10` re-buys | **PASS** ×2 |

**~17.7 of 18.9 W35 GPU-hours produced no PASS row.** That is not unaccounted waste —
it is expensive, honestly-recorded negative information, and `D1.0`'s VOID is a real
finding about a gate rather than about Jack. But it is the third consecutive week the
free quota has largely expired unconverted, and it sharpens FINDING 1 into a fork the
owner should see (below).

### 6. Stuck decisions — nothing stuck

`decisions --check`: **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE**, ratchet 0/10.
Every live entry (D15, D16, D17, D18, D19) carries a class, a default and a
`decide_by`. Nothing on the owner's desk is settleable by measurement; nothing was
quietly acted on without being recorded — the eleven 09-01 default firings are each
in `DECISIONS_RESOLVED.md` with their losers. **Nothing for me to arm**, which is the
first audit in some time that can say that.

### 7. Bakeoff hygiene — no finding

`DECISIONS_RESOLVED.md` records eleven armed-default firings and four true bakeoff
settlements, each with losers named. No decision made without a learning gate, no VOID
treated as a verdict, no winner chosen inside the noise margin. The one live concern —
`D10` seating the learning core **BY VERDICT off a VOID** — is already carried in the
open by `champions --check` (`UNVERIFIED VERDICTS 2/2`, `Learning core LC.03=VOID`) and
routed. Standing, known, not fresh.

### 8. The honest summary — are we closer to a creature, or just to a longer list of ticks?

**Closer to a creature, on one axis, genuinely.** `LG.02` is the first spec that makes
*trust in a person* a measured quantity in Jack's own diary rather than a design
intention, and it did it with the null and the swap control both behaving. That is a
piece of the owner's *"his diary records whose advice proved true"* that did not exist
two days ago. `HR.5` and `HR.7` together bought the first honest map of hearing: the
encoder path is safe, and the **world is silent**.

**And that is the shape of the whole week.** Six independent instruments now say the
same thing from six directions — `SH.01`'s `ORACLE_CANNOT`, `SH.02`'s saturated null,
`SM.03`'s oversubscribed split, `DP.04`'s lifespan quantum, `UB.14`'s venue-bound
probe, and now `HR.5`'s single sound class. Four constitutional commitments are
claim-dead behind it, **not one because Jack failed to learn**. The ladder is not
failing; the ladder has nothing to climb. The Review reached this conclusion
independently this morning and recommends W1 stop being a queue row and become the
project's stated stage. **I agree, and I will put it more bluntly: every one of the
last three weeks' expensive negatives has been a fact about the world, and the
project is still spending its scarce GPU quota on the brain.**

So: not just a longer list of green ticks — the day added three refutations and a real
claim. But the list is getting longer in the places where the world already permits
movement, and the places GOAL.md cares most about are stalled behind one thing.

---

## FOR THE BUILDER

**B1 (rank 1, do this before implementing `HR.6`) — carry `HR.5`'s refutation into
the graph.** `HR.6.depends_on` is `["HR.7", "PG.5", "PG.7"]`; `HR.5` is missing, while
`HR.8` declares it. Add **`"HR.5"`** to `HR.6.depends_on`. Expect and accept the
consequences, both of which are the point:

- `HR.6` leaves the runnable set, `coverage`'s `gpu<2h  <- fillable today: HR.6` line
  disappears, and the `unreachable` count rises by 1. **Raise `UNREACHABLE_BASELINE`
  in the same commit with the justification in the growth log** — `f529ab1` has just
  established the shrink half of that discipline, and this is the raise half. A
  truthful red is the deliverable here.
- `ladder_prompt.md` item **1'** must be corrected in the same motion: it names
  `HR.6`'s CPU staging arms as *"the cheapest fresh unit on the board"* without naming
  `HR.5`. Note precisely why the existing staging valve does not cover this: it fires
  on **A2 vs A0b**, and the branch `HR.5` predicts is **A5 ties everything**, which
  passes that valve and green-lights 3–6 GPU-hours on a question `HR.6`'s own notes
  call not well-posed.

If you believe the CPU staging is still worth running as a *confirmation* of `HR.5`'s
finding, that is defensible — but pre-register it as confirming the fixture verdict,
not as arbitrating representations, and say so before the run.

**B2 — route `HR.5`'s FAIL, with a clock.** Every sibling world-refutation (`DP.04`,
`SH.02`, `SM.03`, `T3.06`, `BA.03`) has a `REVIEW_QUEUE` row. `HR.5` has none, and its
repair contract lives only in a docstring: a sustained NOISE voice driven by persisting
contact (tangential velocity × normal force) versus the impulsive MODAL voice that
exists; a surface-crossing detector inside `Water.apply` (Water is a force field,
`playground.py:246`, which is *why* entry is silent); a `geom_bodyid` self flag; and the
humanoid's absence from `build_mjcf(with_humanoid=False)`. Route it — and note on the
row that it belongs to the same W1 fork as `w0-too-shallow`, so it is not designed twice.

**B3 — close the class FINDING 2 only closed the instance for.** `run.py:861`'s ratchet
reader classifies against `ratchet_readings.json` and never against the tool's own
floor, so `run ratchets record` + commit silences it. Add a **`FLOOR`** comparison
alongside `MOVED`/`UNCHANGED`: for each counter that has a declared baseline
(`UNREACHABLE_BASELINE` today), print the live value **against the floor** and mark it
loudly when they disagree in either direction. The reader's own header promises *"a
blessed red can never silence them"*; today it was silenced by a recording, and it
should not be possible to make it quiet by writing a file.

**B4 — annotate `HR.5`'s row so its headline number cannot be quoted.** `HR.5`'s
registered metric `four_class_audio_separability = 0.583` was produced by a run whose
`position_only_acc` control read **0.708**. The verdict is safe (carried by
`classes_present`, `has_kind_label`, `has_self_flag`), but the metric is not
interpretable and the row's message is the generic *"pre-registered threshold not
met"*. Amend the row — or the docstring where a reader will meet the number — to say
which conjuncts carried the FAIL and that the separability figure is void under the
`T2.11` control-outscored rule. Do **not** re-run to get a cleaner number.

**B5 (cheap) — stop the leftover guard firing on its auditors.** `overseer.sh` and
`review.sh` declare nothing to procwatch, so their read-only instrument calls are
indistinguishable from an undeclared builder leftover; this morning's `LEFTOVER=1` was
`python -m experiments.coverage` at 0 s CPU — this audit's own first call. Have both
scripts `proc_declare` their own pid at start (children of a declared pid are already
attributed, per `lib_procwatch.sh:149`). Keeps the 1.26 core-hour scar's guard sharp
instead of routine.

**B6 — give the queue reader an eye for the pile.** `run review-queue` reports
`OVERDUE` — a promise already broken. Have it also print the **due-date histogram and
the worst pile**, and go amber when more rows share one date than a single consumer
cycle has ever discharged. Seven rows sit on 2026-09-06; the desk's one recorded
capacity measurement is that it died at eleven minutes owing **one**. Predicting seven
broken promises on Sunday is worth more than reporting them on Monday.

---

## FOR THE OWNER

**1. A fork you should see this week, because the two halves pull against each other.**
~**11 free GPU-hours** expire Sunday 2026-09-06. `coverage` names exactly one spec that
could spend them: **`HR.6`** — whose informativeness precondition was **measured false
this morning** (FINDING 1). The honest options are (a) let the quota expire, which is
what my B1 recommends and what the third consecutive week of expiry looks like, or
(b) spend it on something that is not `HR.6`. I am not asking you to choose — the
builder can act on either under existing rules — but I want the trade on the record
rather than resolved by whichever instrument happened to speak last. **W36 opens 09-06
00:00 with a full 30 h and, for the first time in three weeks, a named buyer**
(`D1.0` attempt 2 under a repaired gate, 16.17 h measured).

**2. The recommendation I am seconding, not originating.** The Review's FOR THE OWNER
this morning asks that **W1 stop being a queue row and become the project's stated
stage**. Independently, from a different starting point, I reached the same place in
§8. The evidence is now six instruments deep and four constitutional commitments wide
— smell, balance, shelter/building, thermal, all claim-dead, **not one because Jack
failed to learn**. `HR.5` made it six this morning, in a spec family one day old.
Two organs converging without conferring is the strongest signal this system produces,
and it is pointing at the world.

Nothing else needs your ruling. `decisions --check` is 0/0/0 and fully armed;
`champions --check` is at baseline on every counter; the review queue has zero
violations. **The ledger is clean and the machine is watching itself — this morning
it caught itself without me, which is the first time I can write that sentence.**
