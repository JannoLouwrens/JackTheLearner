# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-26 00:37–01:0x UTC — the 119th audit.** Six hours after the 118th
(18:37). The window is the builder's six slots `19:0x`–`00:0x`, which produced
**eight commits and six ledger events: `LT.03` VOID (hand-repaired down from a
false PASS), `T0.17`, `T0.33`, `T0.35` and `LG.13` re-bought.** Demonstrated
`111/254` — **112 → 111**, and the subtraction is the best thing in this report.

---

## VERDICT: DRIFTING — the ledger got more honest and the creature did not move; the one scientific result was a VOID from a venue where nothing reaches the ladder under any policy, and the organ that reports to the owner has now been dark for 42 hours while its own liveness check reads `OK`

There is no integrity risk in this window and I want that said before anything
else: the one integrity event was **found by the builder, in its own
disfavour, disclosed in the same commit, fixed at the right layer, swept
ledger-wide, and routed for ratification.** I re-derived all four of those
claims independently (Section 1) and they hold. The demonstrated count went
DOWN by one because a claim that was never earned was withdrawn. That is the
machine working.

What is drifting is everything the machine was built to serve. `LT.03` — the
constitutional ladder test, `GOAL.md`'s own image — came back **VOID**, and the
number underneath it is worse than the verdict: **every arm read `engaged`
0.0, std 0.0**, across 3 seeds × 10 lives × 2500 decisions. Candidates,
controls, the random walker and the null alike. Not one engaged ladder attempt
by any policy. And because the pre-registered `_check` puts the noise-panel
liveness gate ahead of the claim, **that measurement cannot fire the
pre-committed pivot it was written to fire** (FINDING 2). Meanwhile
`review_liveness` returns `OK` about a `PROGRESS.md` that has not been written
since 2026-09-24 06:45, because the row the dying Review wrote to disclose its
own death is the row that convinces the liveness check it is alive (FINDING 3).

---

## FINDING 1 — a `_check` that could not fail recorded a false PASS on the project's headline Tier-5 claim; the builder caught it, and I have verified the repair rather than taken it on trust (HIGH — reported as EXPOSURE, not as misconduct)

### 1a. What happened

`LT.03` attempt 1 landed at `2026-09-25T22:00:21` (16,580.6 s, seeds 0/1/2,
clean stamp at `c1114ae`) and `run_spec` recorded **PASS with an empty
message**. Its `_check` — alone in 165 check functions in this repository —
returned `(Status.VOID, reason)` and `(False, reason)` **tuples** on every
branch. A non-empty tuple is truthy, `run_spec`'s else-arm mapped truthy to
PASS, and so **this check could not fail for any data at all.** The strongest
pre-registered battery in the ladder was carried by a channel that discarded
its answer.

The armed harvest instruction (the 118th audit's `FOR THE BUILDER` item 2, and
the Review's guidance before it) said: replay `_check` offline against the
recorded row. That instruction is what caught it.

### 1b. The four things I checked rather than inherited

1. **Is the repaired `_check` the same predicate that ran?** A hand-repair
   justified by replaying an *edited* check would be worthless. `git diff
   c1114ae..HEAD -- experiments/tests/lt_03_ladder_test.py` is **purely
   mechanical**: `_void()` moves the reason into `m["void_reason"]` and returns
   a bare `Status.VOID`; three `return False, "…"` become `return False` with
   the reason demoted to a comment; `return True, "…"` becomes `return True`.
   **Not one branch condition changed in any character.** The replay is of the
   predicate that ran.
2. **Is the demotion recorded honestly?** `ledger.json['results']['LT.03']`:
   `status VOID`, `attempt 1`, `commit c1114ae`, `dirty_files null`,
   `duration_s 16580.6`, `impl_sha 8631cb48…`, and the `message` carries the
   HAND-REPAIR disclosure with the replay quoted and the firing branch named
   (`icm_fixates` 0.0 vs the 0.66 floor). **Metrics byte-untouched** —
   `void_reason` is still `None` in the row, which is exactly what a pre-edit
   run must look like. The `T2.02` precedent is the right one and was followed.
3. **Is the sweep's scope claim true?** The commit claims "exactly one
   verdict-inverting row" and "none carries a tuple-idiom `_check` by grep". I
   ran a stricter test than grep — an AST walk of every `experiments/tests/*.py`,
   **165 check-like functions**, looking for any `Return` whose value is a
   `Tuple` or a non-bool constant: **0 hits.** The two float-flag returns the
   commit discloses (`T2.04`, `T2.05`, `return m["all_seeds_beat_null"]`) are
   the only non-bool/non-Status returns in the repository, they are
   verdict-CORRECT, and the new guard coerces exact 0/1 rather than pricing a
   type repair as a certificate re-buy. `T2.14`'s `return claim` looks like a
   third case and is not — `claim = bool(...)` one line above. **The disclosure
   is accurate and complete.**
4. **Was the staleness bill paid?** The `protocol.py` edit stales
   `T0.17`/`T0.33`/`T0.35`. All three re-bought in the next commit (`d0ff118`,
   2.72 s / 1.29 s / 1.3 s), and the re-buy **exposed a real pre-existing gap**:
   `T0.35` FAILed honestly because `LG.13` had an undeclared transitive
   `IMPL_DEP` on `lg_01` via `lg_10`. Declared (`d1cf88d`), re-bought, green.
   A tool finding a real defect during its own re-buy is the tool earning its
   keep.

### 1c. The gate is the right layer, and one note on the detector that should have owned this

`CheckReturnInvalid` raises on anything that is not a bool or a Status, so the
run lands **ERROR — visibly unfinished — rather than as a verdict the check
never gave.** That is `VoidStatusMismatch`'s principle one type over and it is
correct.

Worth recording about `T0.13` ("no gate in the ladder is decorative"), the spec
whose entire purpose is this class: `t0_13_gates_are_live.py:405` collapses any
non-Status return with `("BOOL", bool(out))`. Against a tuple-returning `_check`
every perturbation therefore compares equal to the base verdict, so T0.13 would
have reported **every consulted key as DISARMED** — the right alarm with the
wrong diagnosis, and only while the false PASS stood, because its SENSITIVITY
detector scopes to PASSing specs and that window was **24 minutes wide**. The
type gate, not T0.13, is the durable guard; nothing further is owed here.

### 1d. Why this is still ranked HIGH

For twenty-four minutes this project's ladder read `[PASS] LT.03 THE LADDER
TEST: curiosity alone climbs the ladder`, and the only thing standing between
that and a published rung was a harvest instruction a human had armed by hand
four hours earlier. No instrument watched the verdict channel. One now does.

---

## FINDING 2 — `LT.03`'s pre-committed kills clause is UNREACHABLE in this venue: the ladder half of the claim is gated behind a precondition belonging only to the dwell half (HIGH — new, structural, and it is about `GOAL.md`'s own pivot)

### 2a. The measurement

From the row's own metrics, all three seeds, std 0.0 on every one:

```
lp_engaged 0.0   disagree_engaged 0.0   metra_engaged 0.0   icm_engaged 0.0
randrew_engaged 0.0   null_engaged 0.0   winner_engaged 0.0     (ENGAGED_MIN = 20)
rig_ok 1.0   finite 1.0   ruler_occ 1.0   every *_chaos_void 0.0
icm_fixates 0.0  (CONTROL_DWELL_MIN 0.4, needed in >=2 of 3 seeds -> 0.66)
```

**No policy in this venue — not the three candidates, not the ICM control, not
the random walker, not the null — made a single engaged ladder attempt in
25,000 decisions per arm.** And every rig gate the spec owns read green.

### 2b. Why that cannot fire the pivot

`CHECKLIST.md`'s `dies if:` for `LT.03` reads *"No arm produces a single engaged
attempt (exploration never reaches the ladder)"* → **`GOAL.md`'s ladder image
needs a goal/skill layer, and that pivot is decided by this result, not by
preference.** That is the most consequential pre-committed pivot in the
repository.

`lt_03_ladder_test.py:850-870` orders the gates: `rig_ok`/`finite` → VOID;
`ruler_occ` → VOID; `null_engaged >= ENGAGED_MIN` → VOID; **`icm_fixates < 0.66`
→ VOID**; all-candidates-chaos-VOID → VOID; *then* the claim, whose failure
returns `False` → FAIL → the pivot. The fourth gate fired. The claim block was
**never evaluated**.

That ordering is defensible in isolation — the claim conjoins *climbed* with
*dwelt <= 0.15 at the panel*, and you cannot certify the second half against a
distractor that is not sticky. The structural consequence is the finding:
**a precondition scoped to the dwell conjunct gates the whole verdict, so while
the panel trap is behaviourally dead this spec cannot return FAIL for any ladder
data whatsoever.** And the trap has never been alive in a humanoid body in this
project's history — `LT.02` attempt 1 and attempt 2 (3 seeds each,
`panel_dwell_icm` 0.0, std 0.0), the `LT.03` pilot, and now `LT.03` attempt 1 at
the full envelope: **ten measurements, all 0.0.** The only `> 0.4` reading
anywhere is `LT.02`'s CONTROL block, which records the different key
`panel_dwell` at 0.666667 — and that is PG.4's **rover**, a different
apparatus. The two keys are worth keeping apart by name: quoting the arm's
number as `panel_dwell` is how a rover result reads as a humanoid one.

So the pivot is not merely unfired. It is unreachable at this venue, and
4.61 CPU-hours bought a VOID that no quantity of ladder evidence could have
converted into the FAIL the clause exists to catch.

### 2c. What is already routed, and the gap in it

`lt03-icm-trap-not-live-in-flight` (OPEN, routed `0ce60dd`) is a good row: it
names the rig finding, records that every arm read `engaged` 0.0, states that
the pilot validated the DETECTOR by teleport probe and never the VENUE, flags
the two artifact gaps (per-seed `icm_dwell` not persisted; `metra_chaos_ratio`
−147,365 ± 100,816, five orders beyond every sibling), and correctly says the
kills clause did not fire because a VOID decides nothing.

**What it does not say is that the kills clause cannot fire here at all.** A
redesign that only revives the panel trap would leave the ladder half hostage to
the distractor a second time. Routed below to the Review as an extension of that
row, with the prohibition restated: the repair is **not** lowering
`CONTROL_DWELL_MIN`, which would buy a decidable verdict by weakening the
control that makes the dwell conjunct mean anything.

### 2d. Was the spend foreseeable?

Partly, and the builder said so itself rather than being caught: the pilot
record's own words are that zero engagement at 800 decisions/arm *"is LT.01's
null floor as expected"*, and the teleport probe was added **because the pilot
could not show the trap live.** So `_PILOT_VALIDATED` was flipped on mechanics +
detector with **no positive behavioural control**, which is the same shape that
already cost `SH.02` (null holds the roof at exactly 1.0000), `SM.03` (`vis_open`
0.1167 against a 0.60 floor), `T2.11` (permuted twin beat the claim arm) and
`LT.02` attempt 1. That is five instances and it is now a class. The cost here
was CPU, not GPU, and the builder disclosed the reasoning at the time — so this
is recorded as a pattern for the pilot idiom, not as a charge against this slot.

---

## FINDING 3 — `review_liveness` reports `OK` about a page that is 42 hours stale, because the row the dying Review writes to DISCLOSE its death is the row that satisfies the liveness check (MEDIUM-HIGH — new mechanism; my predecessor found the fact, this is the cause)

### 3a. The fact, worse than yesterday

`git log -1 -- docs/PROGRESS.md` is still **`f7abc08`, 2026-09-24 06:45:13**.
The page is **41 h 52 m old** against a 25 h cadence, carries **no staleness
marking**, and opens `**2026-09-24 06:3x–06:5x UTC — DAILY**` as a finished
report. The 118th audit found it at 36 h. Nothing has moved because the organ
that owns the repair has not sat since; its next cron fire is 06:37 today.

### 3b. The cause, which is not "the exemption is never re-checked"

My predecessor routed this as *"the banner decision is taken once, at the moment
of death, and nothing re-evaluates it"* — true, and it named
`scripts/lib_liveness.sh:review_liveness` as a candidate home for the repair. I
went to check whether that function could host it and found something stronger:
**it already owns the stamping machinery and already runs, and it returns `OK`.**

```
$ REPO=. ; source scripts/lib_liveness.sh; review_liveness say
SAY: review liveness: OK — 2026-09-25 daily, 2026-09-20 FULL
review_liveness rc=0
```

`scripts/lib_liveness.sh:176` is `reason=$(table_liveness docs/PROGRESS_LOG.md 1
FULL 7)`. The predicate reads **`PROGRESS_LOG.md`'s table**, not `PROGRESS.md`'s
own age. And `PROGRESS_LOG.md:45` carries a row for `2026-09-25` — the
**INCOMPLETE row the dying run wrote**, under the 76th audit's B4 guard, so that
the trend would have a labelled hole instead of a silent gap. Its own text says
*"any sealed draft is bannered in docs/PROGRESS.md."*

So: the guard that exists to make the hole VISIBLE is the thing that makes the
liveness check BLIND. `review_liveness`'s `stale_output docs/PROGRESS.md review
…` branch at line 182 — which would stamp exactly the banner that is missing —
is unreachable whenever the Review dies *after* writing its own disclosure row,
which is every time, because B4 writes that row on the death path. Honest
disclosure in one file suppresses the alarm in another.

### 3c. What the owner is being shown right now

The live `PROGRESS.md` is the only current-state page the owner has and it says:

- **Headline:** *"THE BUILDER IS BACK AND HAS NOTHING TO DO."*
- **FOR THE BUILDER 2:** *"Tomorrow (09-25) the `WAITS-ON:` disposition
  unlocks… **Do not start it early**"* — done at 15:13 on 09-25.
- **FOR THE OWNER 1:** *"~29.08 hours expire **Saturday 2026-09-26, two days
  out**"* — that is **today**, ~23 h out.
- **FOR THE OWNER 2:** *"`D33` … is now **one day past** its `decide_by`"* — it
  is **three**.
- **Nowhere on it:** the `LT.03` launch, the `LT.03` VOID, the false PASS, the
  verdict-channel defect, `LT.02`'s PASS, or the five rulings of 09-25 that
  produced it.

### 3d. Re-routing, with the reason stated

The 118th audit gave this to the Review (`FOR THE REVIEW` 6). **I am moving it
to the builder**, and the reason is in the diagnosis: the Review's capacity is
what `D33` and `D36` are about, and handing a three-line shell repair to the
organ measured as the bottleneck is how it ages another week. It is mechanical,
it needs no design, and the staleness bill is **ZERO** — I checked: no spec in
`experiments/registry.py`, `registry_expansion.py` or `experiments/tests/`
declares `scripts/lib_liveness.sh` in `IMPL_DEPS`. Ordered in `FOR THE BUILDER`
1.

---

## FINDING 4 — `docs/DECISIONS_RESOLVED.md` is a file the runner writes DURING a run and is in neither `RUNNER_OUTPUTS` nor `DOC_OUTPUTS`, so a bakeoff poisons the `+dirty` stamp of the next spec to record — fourth occurrence of "the evidence log that invalidates the evidence" (MEDIUM — new, cheap, precisely priced)

`experiments/bakeoff.py:75` writes `docs/DECISIONS_RESOLVED.md` as part of a
registered run. `experiments/protocol.py:52-100` lists `NOT_CODE =
RUNNER_OUTPUTS + DOC_OUTPUTS` and **that file is in neither list**, so an
uncommitted write to it reads as *uncommitted code* to the `+dirty` stamp.

It fired last night, seven seconds wide:

```
LG.13   PASS attempt 2   ran_at 22:29:05   commit d1cf88d       dirty_files null
T0.35   PASS attempt 13  ran_at 22:29:12   commit d1cf88d+dirty dirty_files ['docs/DECISIONS_RESOLVED.md']
```

`LG.13`'s bakeoff appended its winner record; `T0.35`'s re-buy recorded seven
seconds later and was stamped `+dirty` **by the previous spec's own receipt.**
`DIRTY STAMPS` went **2 → 3** in this window (`T6.03`, `PL.02`, now `T0.35`),
and `run status` correctly reports that `T0.35`'s declared implementation
reconstructs byte-identically, so nothing is *unknown* — but a certificate is
carrying a mark that says "the code that ran is in no commit" about a run whose
code was fully committed.

That sentence is copied from `protocol.py`'s own comment block, which documents
**three prior occurrences of this exact class** — `T2.00` (`docs/LOOP_JOURNAL.md`,
998-second re-run), `T0.34` attempt 1 (`cpu_budget.json`, the receipt of the
re-buy beside it), `T2.01` (6.5 Kaggle-hours) — and warned in so many words
about "the exact sibling the paragraph above warned about". This is the fourth,
in a file nobody added.

**The precedent points at `RUNNER_OUTPUTS`, not `DOC_OUTPUTS`.** `ledger.json`
is read by every instrument in the repository and is excluded, because the test
is *"did CODE move"* and a file the runner itself writes during the run it is
stamping is never evidence that code moved. **One caveat the builder must clear
first rather than assume:** `decisions.py` and the instruments DO parse this
page (`decisions.py:315`, `RECORD_PAGE` at 749), so before excluding it, confirm
no spec's verdict reads the live file (`T0.28` is the candidate to check). Bill:
the same three `cpu<10min` re-buys every `protocol.py` edit costs, measured last
night at 2.72 s + 1.29 s + ~1.3 s. Ordered in `FOR THE BUILDER` 2.

---

## FINDING 5 — `decisions --check` is still RED on a ratchet with a floor of 0, and the entry has now been STALE for three days (MEDIUM, inherited twice, ageing)

```
RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0. It may shrink, never grow.
```

Unchanged since the 117th audit. `D33`'s default names **2026-09-23**, its
`decide_by` is **2026-09-23**, earliest legal firing **2026-09-24** — the ordered
action is in the past on the day it fires. The two named repairs (**shorten
`decide_by`**, or declare **`(CLOCK: <whose>)`**) are the Review's and neither
has been taken through three sittings. Ageing: `D33` `CONDUCT-DESK … STALE by 3
day(s)`, `D35` **STALE by 2**, `D36` **due today**. All three are
desk-executable and say so on their own faces — *execute it, report it, do not
ask.*

`run status`'s `STEERING-DATE-MISMATCH` is at **1 arm** and it is the one that
matters: `docs/PROGRESS.md` tells the owner `D33`'s deadline is **2026-09-27**;
the register says **2026-09-23**. My predecessor's arm cleared when its page
committed, exactly as it predicted. This page states `D33`'s deadline once, as
`2026-09-23`, and restates no filing date in any form.

`STEERING-METRIC-MISMATCH` is **1 → 0** with this page, and the way it got there
is worth one sentence because the reader earned it twice in two days. My
predecessor's arm was a legitimate foreign-attempt quote of
`chaos_occupancy_icm`. My own first draft minted a new one by naming `LT.02`'s
ARM dwell with the CONTROL's key — the arm's key is `panel_dwell_icm` (0.0 on
every seed of both attempts) and the rover control's is `panel_dwell`, quoted
correctly at FINDING 2b. **The reader was right both times**, and the second
catch is the more useful one: putting a rover's key beside a humanoid's number is
exactly how a different apparatus's result would launder into this family.
Corrected before commit rather than explained after it.

---

## Section 1 — integrity of the ledger

**Re-derived, not inherited, and it is CLEANER than yesterday by exactly one
withdrawn claim.**

- **111 PASS rows.** Every PASS `commit` still resolves in git — **0 missing**,
  checked with `git cat-file -e` over all 111.
- **Control evidence:** every PASS whose spec declares a `control` carries
  `control_metrics`, with exactly two exceptions and both are the declared-NONE
  form (`T0.01` *"an import either raises or does not"*, `T0.10`). **0 claims
  without evidence.**
- **Implementations exist:** 173 files in `experiments/tests/`; no PASS row
  names a missing one.
- `run status` **EXIT 0**.
- **FINDING 1** is the window's integrity event and it moved the count in the
  honest direction: `112 → 111`.

**Reporting-only debts, with the one that GREW named:** `DIRTY STAMPS` **2 → 3**
(`T0.35`, FINDING 4). `STALE CLAIMS` **17** (was 16; `LT.03` joined, which is the
mechanical consequence of repairing its own `_check` and is owed to the redesign
row, not to a re-run). 1 stale pre-`impl_sha` (`T2.02`), 3 UNBACKED CERTIFICATES
(`LF.02`, `T2.03`, + 1), `pass_on_dead_dependency = 3` **AT floor**,
`unreachable = 95` **AT floor**.

**Ratchets that MOVED, said out loud as the tool demands:**
`review_queue_violations` **0 → 8**, forms `{} → {'OVERDUE': 8}` — the tool's own
derivation is **CLOCK, not act**: eight dated promises crossed midnight on 09-25
together and no commit is to blame. `review_queue_net_arrivals` **12 → 14**,
`+2 by ACT` — and both acts are the builder's two honest routings from the
`LT.03` harvest (`lt03-icm-trap-not-live-in-flight`,
`check-return-type-defect-swept-and-repaired`). Routing a finding is the correct
act; the counter rising is the cost of the desk not draining. Everything else
unchanged and at floor.

## Section 2 — thresholds and controls over time

**No findings. Every numeric move in the window is an addition or a
tightening, and I checked the diffs rather than the messages.**

- `experiments/registry.py` and `registry_expansion.py`: **untouched in this
  window.**
- `lt_03_ladder_test.py`: return-type mechanics only, **no branch condition
  altered in any character** (FINDING 1b.1). No bar moved. `CONTROL_DWELL_MIN`
  0.4, `ENGAGED_MIN` 20, `DWELL_MAX` 0.15 all byte-identical.
- `lg_13_chooser_seat_bakeoff.py`: **+1 `IMPL_DEP`** (`lg_01` via `lg_10`). A
  strengthening — the certificate now stales when the probe pool changes. It
  exists because `T0.35` FAILed on the missing edge.
- `protocol.py`: `CheckReturnInvalid` **added**. Strictly more to satisfy;
  nothing may now record a verdict its check did not give.
- Seven days of `git log -p` over the registry and `experiments/tests/`: **no
  threshold moved in the loosening direction, no control deleted or weakened, no
  `_check` gained an `or`, no seed count reduced, no assertion removed.** The
  window's only `_check` edit made a check able to fail for the first time.

## Section 3 — drift from the goal

**Each unit traced, and the honest total is that this window served the HONESTY
of the ladder and not the creature.**

| unit | `GOAL.md` sentence it serves |
|---|---|
| `LT.03` harvest + VOID | *"he must try to climb the ladder, fall, and learn from falling, purely out of curiosity"* (l.31-33) — attempted directly, and the attempt measured that the venue cannot host it |
| verdict-channel gate + sweep + hand-repair | *"Really learning, not appearing to learn… every capability claimed only by an experiment that could have failed"* (l.57-60). A check that cannot fail is the purest violation of that sentence there is; closing it is load-bearing goal work |
| `T0.17`/`T0.33`/`T0.35`/`LG.13` re-buys | staleness bill for the above. Instrument work, correctly labelled |
| `LG.13` transitive `IMPL_DEP` | ibid. — a real gap the bill exposed |
| hash-salt binding-set measurement | instrument honesty (which gates a hash lottery could decide). Reporting-only, precondition discharged before implementation as the disposition ordered |
| 4 mid-flight verification slots | necessary custody of a 4.6 h detached run; no science, and none claimed |

**Nothing in the window is drift in the sense of serving no sentence.** The drift
is the shape of the total: **zero net PASS, one VOID, five re-buys, and four
slots of watching a process breathe.**

**The converse question, and it is unimproved.** `coverage` EXIT 2:
`commitments_uncovered = 0` **AT floor** — no commitment lacks a declared spec,
which is this organ's founding check and it is green. But:

- **4 CLAIM-DEAD** commitments — smell, balance, thermal-kills, shelter/building
  — where every claim spec is parked or foreclosed. Unchanged since 09-19.
- **13 commitments with live claim specs and nothing passing.**
- **`NO-LIVE-PATH` = 7** distinct commitments/seats with no walkable revival.
- **`goal_unrunnable = 7`, unchanged since 2026-09-05 — twenty-one days.**
  `GEN.02`/`GEN.03`/`GEN.06`/`GEN.09` are the four over the shrink-only
  baseline, and the system found its own orphan here: the row that owned them
  (`goal-cites-four-specs-that-resolve-to-corpses`) is stamped ACTED and
  re-parented to `D24`, which **closed 2026-09-12, before the re-parent** —
  printed by `review-queue`'s `DISPOSITION-ON-A-CLOSED-DECISION` block, and a
  fresh row (`gen-four-reparented-to-a-decision-that-had-already-closed`, DUE
  10-01) was minted four days ago to catch it. **That is the instrument working;
  the 21-day age is still the number.**
- **Curiosity: 12 specs, 2 passing.** `one brain / unison`: **28 specs, 1
  passing** — still the worst ratio on the board and the commitment `GOAL.md`
  argues hardest for. This window did not touch it.
- **Queue depth: 6 dispatchable today, all 6 VOID → 0 FRESH dispatches.** Four
  cost classes NEWLY EMPTY with **no path in** (`cpu<1min`, `cpu<48h`,
  `gpu<20min`, `gpu<8h`) — nothing to implement and nothing to pilot at those
  costs.

## Section 4 — is the builder alive and productive?

**Alive, disciplined, and its conduct in this window is the reason FINDING 1 is a
repair and not a scandal.**

**Iterations since midnight 09-25: 25, all `rc=0`.** Six in my window
(`19:0x`–`00:0x`), all `rc=0`, no aborts, no pause, no credit exhaustion —
`week:all models` **54%** at the 00:0x open (the gate; the Fable line reads 78%
and is not the gate), week 69% elapsed against a pace allow of ~69.8, no skip.
`lost_iterations.log` 0 bytes. `declared_pids` empty. **PASS delta over the
window: 112 → 111.**

| slot | commit(s) | what landed |
|---|---|---|
| 19:0x | `9953754` | mid-flight receipt #1; 118th-audit FTB read; harvest guidance armed in the journal |
| 20:0x | `05bed6d` | receipt #2 |
| 21:0x | `3c7f258` | receipt #3; harvest guidance re-armed verbatim |
| 22:0x | `0ce60dd`, `d1cf88d`, `d0ff118` | **the harvest: false PASS caught, channel fixed, ledger demoted, swept, 4 re-buys** |
| 23:0x | `63263bc` | LT.03 aftermath verified clean on disk; board re-derived empty |
| 00:0x | `19aab39` | hash-salt binding-set measurement, reported before implementation as ordered |

**All five of the 118th audit's `FOR THE BUILDER` items are discharged, verified
against commits.** Item 2 (read a VOID on `icm_fixates` as a rig finding, report
`icm_dwell` per seed, do not touch `CONTROL_DWELL_MIN`) — done, and the queue row
says so in the ordered words; the per-seed `icm_dwell` **could not** be reported
because the run did not persist it, and the builder recorded that as an artifact
gap rather than quietly reporting the mean as if it were the ask. Item 3 (do not
relaunch, replay `_check` offline, commit as found, no re-roll) — this is the
item that caught the false PASS. Item 4 (do not fix `LT.02`'s science) — no bar
moved. Item 5 (do not pre-empt `D31`/`D33`/`D35`/`D36`, the `W1.01/03/04`
registration, `T1.08`'s pipeline repair, `UB.10`'s arm choice, `A4`, `T2.10`,
`SO.07`, `SO.10`) — all untouched, and `D31` was explicitly left for this slot.

**Two pieces of conduct worth naming.** The 22:0x slot could have stamped the
PASS and moved on; the ledger, the ratchets and `CHECKLIST.md` would all have
agreed with it, and the only contrary evidence was an instruction it had written
to itself three slots earlier. It replayed the check instead. And the 00:0x slot
**measured before implementing** on the hash-salt row because the disposition
ordered that sequence, then stopped — including the finding that
`PYTHONHASHSEED` is fixed at interpreter start, so the differential needs a
subprocess and bills `protocol.py`'s certificates. Reporting a design obstacle
found while measuring, instead of quietly routing around it, is the habit.

## Section 5 — compute honesty

**Reconciles, and the one expensive thing in the window bought a VOID honestly.**

- **CPU, 2026-09-25: 18,077.59 s billed**, of which `LT.03` **16,582.33 s =
  4.61 core-hours**. Projected 5.2 h, actual 4.61 h — **under the estimate, and
  the envelope was frozen from the pilot's measurement rather than grown to
  fit.** It returned a VOID. That is not waste: an experiment that ran, recorded
  its metrics and could not test its claim is an outcome of running. What it
  cost is stated in FINDING 2d rather than absorbed.
- **2026-09-26: 0 s billed so far.** The 00:0x hash-salt scan is offline
  `_check` replay against recorded rows — no physics, no meter.
- **GPU: `2026-W38` holds 0.9176 h drawn of 30.** **~29.08 free Kaggle
  GPU-hours expire at the Sunday 00:00 boundary, ~23 hours from now**, and there
  is no legal buyer: `coverage` reports every GPU class EMPTY or NOT FILLABLE,
  with `gpu<20min` and `gpu<8h` having **no path in at all**. The builder has
  refused to manufacture one **34 consecutive times** and should refuse again
  today. An unasked-for GPU run is worse than an expired hour.
- **`overruns: []`** in `gpu_budget.json` — no GPU dispatch since the `D31`
  mark shipped on 09-18, so the mark has had nothing to mark. Consistent, not
  broken.
- **Standing debt unchanged since 09-18:** `gpu_hours_no_verdict` **TOTAL
  48.42 h**, of which `D1.0` alone is **33.78 h / 2 attempts / 0 verdicts** and
  `UNATTRIBUTED` is **6.32 h across 21 jobs** (at its declared floor of 21).

## Section 6 — stuck decisions

**`D31` FIRED THIS AUDIT. The owner did not rule by 2026-09-25, so the
pre-registered default fired.** Option **(i) MARK BUT DO NOT CAP**; options (ii)
GIVE COLAB A CEILING and (iii) DECLINE were NOT taken and remain the owner's.
Recorded at the tail of `docs/DECISIONS_NEEDED.md`. **The firing orders no
work:** option (i) shipped as `2bfa84f` on 2026-09-18, and I verified it at
source (`gpu.py:133` `PER_JOB_OVERRUN_MARGIN = 0.25`, `gpu.py:540` the
comparison) rather than trusting the 100th audit's addendum. What it settles is
the RECORD — the entry has presented (i) as an unexecuted option since 09-15.
`remaining('colab')` still returns infinity, the lane still has no ceiling, no
dispatch is refused that was permitted yesterday, and the 2.11 realised
colab-hours are written off rather than recovered. **Reversal: delete one `if`
from `experiments/gpu.py`.** Transcription onto `DECISIONS_RESOLVED.md` is owed
by the builder (`FOR THE BUILDER` 3) per the `D13` rule and the `D32`/`D34`
precedent (`6aaed9b`).

**Zero `MEANS-ESCALATED`.** Nothing a measurement could settle is on the owner's
desk. **Zero `UNDECLARED`**, so the standing "arm one per audit" duty has no
object today — that is the honest report and not an omission; the ratchet may
shrink and it did, by one, this audit.

Remaining unarmed: three `CONDUCT-DESK` (`D33` stale 3, `D35` stale 2, `D36` due
today — all the Review's, all *execute and report, do not ask*) and `D33`'s
`DEFAULT-ACTION-EXPIRED` (FINDING 5).

**Was anything acted on without being recorded?** Checked both directions. The
window's only `DECISIONS_RESOLVED.md` write is `LG.13`'s bakeoff winner record,
written by `bakeoff.py` during the run — not a decision resolved by the builder.
`D32`/`D34`, fired by the 115th audit, are both transcribed (`6aaed9b`). Nothing
smuggled.

## Section 7 — bakeoff hygiene

**No decision made without a gate, no VOID treated as a verdict, no winner
inside the noise — and one filing duplicate to fix.**

`LG.13` attempt 2 re-ran and appended a winner record: **meaning-mass 1.000,
beating topk-softmax by 4.13 sigma and the null by 56.00 sigma**, metric
`match_both`, null 0.192 ± 0.014, gate mode `screen` with its rationale
**declared in advance** and the reason it is not `validity` argued from the rig's
structure (every arm a deterministic function of one frozen 1588-entry verdict
table no arm can perturb). **The control failed in the required direction:**
`control:state-free-prompt` 0.064, **−5.02 sigma**, marked FAIL. That margin is
well outside noise and the verdict is clean.

**The hygiene item:** the record now appears **twice** in
`docs/DECISIONS_RESOLVED.md` (lines 1237 and 1862), byte-identical because the
re-run is deterministic, with nothing marking the second as the staleness re-buy
of the first. A reader counting bakeoff verdicts counts two. Ordered as a
one-line annotation in `FOR THE BUILDER` 4 — **not a deletion**; the fix is to
say which is the re-buy.

`T4.06` remains the weak one and remains correctly labelled: `run status`'s
`ANCHOR-DECIDED CONJUNCTS` block prints `loss_reweight` at **+0.0187 = 6.9% of
the incumbent's own seed spread 0.2699, with 1 of 3 seeds regressing** and
`grad_norm` **refuted at −56.7%, 0 seeds improving / 3 regressing**. Inside the
noise by any ordinary reading, said so on the certificate's face, adoption routed
to the live `t402` row and **still not taken.** `LT.03` correctly declined
`run_bakeoff` (it VOIDs on a sub-gate arm and its ICM control is REQUIRED to
fail) and defers arbitration to `LT.04`, which stays blocked.

## Section 8 — the honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we
were yesterday. We are closer to a ladder we can trust, which is not the same
thing and is not nothing.**

Yesterday this organ wrote ON TRACK because a registered, pilot-validated,
three-control Tier-5 experiment was in flight whose hypothesis was the sentence
at the top of `GOAL.md`. It landed. The answer is that **no policy in that venue
— not the three curiosity candidates, not the ICM control, not the random
walker, not the null — made a single engaged ladder attempt in 25,000 decisions
per arm, on three seeds, with std 0.0 and every rig gate green.** The ladder is
there. `PG.3` certified it climbable. Nothing went near it.

And the verdict on that is VOID, correctly, because a gate about the noise panel
sits ahead of the claim — which means **the most consequential pre-committed
pivot in this repository cannot be fired by the measurement it was written to
catch** (FINDING 2). Read together with `LT.02` (self-generated chaos is
REDUCIBLE here — the farmer moves LESS than random), `T3.06` (curiosity does not
beat a random walker at coverage, and the null walker breached its own analytic
dwell bound), `SH.01`/`SH.02` (born outside, seeking unlearnable; born inside,
the null holds the roof at exactly 1.0000) and the seven instruments already on
`w0-too-shallow`, the pattern is not that Jack's curiosity is weak. **It is that
the worlds we have built do not produce the behaviours we are gating on.** That
is a finding about the world, it is the most valuable thing this project
currently knows, and the design that would fix it — the W1 world-edit window —
has now lost five consecutive sittings and is the single unit everything else is
queued behind.

Against that: the ledger is more honest tonight than last night, by a measurable
amount. A false PASS on the headline Tier-5 claim stood for twenty-four minutes
and was withdrawn by the organ that would have benefited from keeping it, with
the replay quoted, the sweep run, the metrics untouched, the bill paid in the
same hour and the ratification routed to a desk that has not asked for it yet.
**`112 → 111` is the number I would point at if asked what this window was
worth.** A project that subtracts its own unearned green tick at 22:24 on a
Friday is a project whose remaining 111 mean something.

The thing that did not work, for the second audit running, is the organ that
tells the owner any of this. `PROGRESS.md` is 42 hours old, unmarked, headlined
*"THE BUILDER IS BACK AND HAS NOTHING TO DO"*, and the liveness check that owns
the missing banner reports `OK` — because the Review's honest disclosure of its
own death is what convinces that check it is alive. The owner's only
current-state page does not know the ladder test ran, does not know it VOIDed,
and does not know that for twenty-four minutes it said PASS.

---

## FOR THE BUILDER

1. **Repair `review_liveness` so it reads the page it stamps.** Taken off the
   Review (118th audit `FOR THE REVIEW` 6) and given to you deliberately —
   FINDING 3 is the Review's own capacity problem and this is a three-line shell
   fix. `scripts/lib_liveness.sh:176` asserts liveness from
   `table_liveness docs/PROGRESS_LOG.md 1 FULL 7` — a table the DYING run writes
   its own INCOMPLETE row into (76th audit B4), so the `stale_output
   docs/PROGRESS.md review …` branch at line 182 is unreachable on exactly the
   deaths it exists for. Add a second, independent assertion: **`docs/PROGRESS.md`'s
   own age against its 25 h cadence**, evaluated on every call, stamping the same
   way. Verified for you: **staleness bill ZERO** — no spec declares
   `scripts/lib_liveness.sh` in `IMPL_DEPS`. Do **not** touch the B4 INCOMPLETE
   row; it is honest and the fix is to stop it silencing the other alarm.
2. **Add `docs/DECISIONS_RESOLVED.md` to `RUNNER_OUTPUTS` in
   `experiments/protocol.py`** — FINDING 4, fourth occurrence of a class that
   file's own comment block documents three times. `bakeoff.py:75` writes it
   during a run, so a spec recording seconds after any bakeoff is stamped
   `+dirty` by the previous spec's receipt: `T0.35` attempt 13 is
   `d1cf88d+dirty` on `dirty_files ['docs/DECISIONS_RESOLVED.md']` while
   `LG.13`, seven seconds earlier, is clean. `RUNNER_OUTPUTS` is the right home
   (the `ledger.json` precedent: a file the runner writes is never evidence that
   CODE moved), **not** `DOC_OUTPUTS`. **Clear one thing first rather than
   assuming it:** the instruments parse this page (`decisions.py:315`,
   `RECORD_PAGE` at 749), so confirm no spec's verdict reads the live file —
   `T0.28` is the candidate. Bill: the usual three `cpu<10min` re-buys
   (`T0.17`/`T0.33`/`T0.35`, ~5.3 s total last night). Then re-buy `T0.35` from
   a clean tree so the `+dirty` mark clears.
3. **Transcribe `D31`'s firing onto `docs/DECISIONS_RESOLVED.md`.** I fired the
   armed default this audit and wrote the `## D31 — RESOLVED BY ARMED DEFAULT`
   header into `DECISIONS_NEEDED.md`; the `D13` rule keeps me out of the record
   page and the `D32`/`D34` precedent (`6aaed9b`, 09-25 01:12) is the form.
   Until it lands, `decisions.py`'s second identification channel
   (`RECORD_PAGE`/`RECORD_MARKER`) cannot see the firing. **Quote the two things
   that matter: the default was already implemented on 09-18 so the firing
   orders no work, and options (ii) and (iii) remain the owner's.**
4. **Mark the duplicate `LG.13` winner record.** `docs/DECISIONS_RESOLVED.md`
   lines 1237 and 1862 are byte-identical bakeoff verdicts from attempt 1 and
   the attempt-2 staleness re-buy. **Annotate the second as the re-buy; do not
   delete either.** One line.
5. **Do not "fix" `LT.03`.** No re-roll, no second attempt, and above all **do
   not move `CONTROL_DWELL_MIN`** — lowering it would buy a decidable verdict by
   weakening the control that makes the dwell conjunct mean anything, and FINDING
   2 exists partly to name that temptation before anyone feels it. The redesign
   is the Review's. If you quote `LT.03` anywhere, quote the VOID and the
   `engaged` 0.0 together; the second number is the useful output.
6. **Credit, and it is substantive.** You could have stamped the PASS at 22:00
   and every instrument in this repository would have agreed with you. The only
   contrary evidence was an instruction you had written to yourself three slots
   earlier, and you followed it. Then you fixed the channel rather than the
   symptom, swept the whole ledger, demoted your own row with the replay quoted,
   paid the bill in the same hour, and routed the ratification to a desk that had
   not asked for it. **That is the single most valuable hour of builder work in
   this project's recent record.** Keep re-deriving rather than inheriting.
7. **Still not yours, do not pre-empt:** `D33`/`D35`/`D36`, the
   `W1.01`/`W1.03`/`W1.04` registration, `T1.08`'s pipeline repair, `UB.10`'s arm
   choice, the `t402` adoption, `A4`, `T2.10`, `SO.07`, `SO.10`, the PS siblings
   (held behind the oracle-cut row, DUE 10-03), and `LT.03`'s redesign. **And do
   not manufacture a W38 buyer on its last day** — ~29.08 free GPU-hours expire
   at Sunday 00:00 with every GPU class EMPTY or NOT FILLABLE. Refusal #35 is
   the right act.

## FOR THE REVIEW

8. **Your page has not been written since 2026-09-24 06:45 and your own liveness
   check says you are fine.** FINDING 3 gives the mechanism and I have routed the
   code repair to the builder so it does not wait on your calendar. **What only
   you can do is the page**, and three things on the live copy are now actively
   wrong to the owner: the `D33` deadline (says 09-27, register says 09-23 — the
   sole remaining `STEERING-DATE-MISMATCH`), the W38 expiry (says "two days out",
   it is today), and the headline *"THE BUILDER IS BACK AND HAS NOTHING TO DO"*
   over a window that contained a false PASS, its repair, and the first flight of
   the ladder test. **Whatever else the 06:37 sitting does, `LT.03`'s launch and
   VOID, the verdict-channel defect, and `LT.02`'s PASS belong on the page before
   anything older does.**
9. **`LT.03`'s redesign is yours and it has a second half nobody has stated.**
   `lt03-icm-trap-not-live-in-flight` correctly names the dead trap. It does not
   say that **while the trap is dead the spec cannot return FAIL for any ladder
   data**, because `icm_fixates` is ordered ahead of a claim whose climbing
   conjuncts have nothing to do with the panel (FINDING 2). So a redesign that
   only revives the trap leaves `GOAL.md`'s pivot hostage to a distractor a
   second time. The options I can see, none of them mine to choose: split the
   claim so the climbing half is decidable without the panel; move the panel
   gate behind the claim so a no-engagement run FAILs; or state in the open that
   the pivot is parked until a venue can host it. **`CONTROL_DWELL_MIN` is
   untouchable and no re-roll is legal.** Also carry the two artifact gaps: per-seed
   `icm_dwell` is not persisted, and `metra_chaos_ratio` recorded
   **−147,365 ± 100,816**, five orders beyond every sibling, with `chaos_void` 0.
10. **`D33` still breaks a ratchet with a floor of 0 and only you can repair
    it**, now `STALE by 3 days` with `D35` at 2 and `D36` due today. The named
    repairs are **shorten `decide_by`** or declare **`(CLOCK: <whose>)`**;
    deleting the date is not one. This is the fourth consecutive audit to ask.
11. **8 rows went OVERDUE at midnight and the tool says no commit is to
    blame.** Drain still **UNBOUNDED** — 61 live, arrivals exceed disposals by 14
    over the trailing week, 14 dated rows fall due on or before today against a
    measured capacity of 6, and `2026-10-03` is the next date with room. Your own
    pre-committed stop-rule on `w1-world-edit-window` lands **tomorrow,
    2026-09-27**, the same Sunday `D36` is about. Note also that
    `d35-none-quota-has-no-satisfying-move` — the row that carries a 34-slot
    breach of `D35`'s rule-3 quota — is itself now OVERDUE, so the disclosure
    vehicle for a standing breach has broken its own date.

## FOR THE OWNER

**1. `D31` FIRED THIS MORNING, and it costs you nothing.** You did not rule by
2026-09-25, so the pre-registered default fired at the first legal slot: option
**(i) MARK BUT DO NOT CAP** — a per-job overrun mark and stderr print in
`experiments/gpu.py` when billed hours exceed the declared estimate. **It was
already implemented on 2026-09-18, so the firing changes no behaviour at all**;
what it changes is the record, which had been presenting (i) as one unexecuted
option among three. The colab lane still has **no ceiling**, `remaining('colab')`
still returns infinity, and no dispatch is refused today that was permitted
yesterday. **Option (ii) GIVE COLAB A CEILING is still the only option that
changes what that lane may spend, and it is still yours** — a default may record
a debt but may not invent a budget number on your behalf. Reversal of what fired:
delete one `if` from `experiments/gpu.py`.

**2. NO-DECISION — the ladder test ran, and the answer is about the world.**
`LT.03` landed at 22:00:21 after 4.61 CPU-hours. Verdict **VOID**, and the
number under it is the one to keep: **no policy — the three curiosity
candidates, the naive-curiosity control, a random walker and the null alike —
made a single engaged ladder attempt in 25,000 decisions per arm, on three
seeds, with std 0.0**, while every rig gate the spec owns read green. I told you
yesterday that a VOID was the modal outcome and why; that held. The finding is
**about the venue, not about Jack**, and it is the ninth independent instrument
saying the worlds we have built are too shallow to produce the behaviour we gate
on. Nothing in this is a rung and nothing in it refutes curiosity.

**3. NO-DECISION, and it is the one thing I would want you to read: the ledger
subtracted one of its own green ticks last night, unprompted.** For twenty-four
minutes `LT.03` read **PASS** — the headline Tier-5 claim, the sentence at the
top of `GOAL.md` — because its verdict check returned a data structure the runner
read as "yes" regardless of content. The builder replayed the check against the
run's own recorded numbers because a note it had written to itself hours earlier
said to, found VOID, **fixed the channel rather than the symptom**, swept all 111
certificates for the same defect (exactly one row affected; I re-verified that
with a stricter scan of all 165 check functions), demoted its own row with the
replay quoted in the record, paid the re-run bill in the same hour, and routed
the whole thing for outside ratification. **Demonstrated went 112 → 111.** I
would rather show you that number than any increase this week.

**4. The perishable price, for the last time this week.** `2026-W38` holds
**0.9176 h drawn of 30 — ~29.08 free Kaggle GPU-hours expire at Sunday 00:00,
about 23 hours from now**, and there is no legal buyer: every GPU cost class is
EMPTY or NOT FILLABLE, two of them with no path in at all. The builder has
refused to manufacture one **34 consecutive times** and should refuse a 35th
today. This is the second consecutive week the same hours have idled behind the
same undesigned repair, and as the Review wrote on 09-25: *the hours are not the
scarce resource, the design sitting is.*

**5. NO-DECISION: `D33` is three days past its `decide_by` and the cost line is
the only new thing.** The narrow ask is unchanged — rule that the world EDIT is
IMPLEMENTATION and was never the Review's to hold under `D22`.
`W1.01`/`W1.03`/`W1.04` are now **twenty days unregistered**, `review-queue`
prints them as NOT REGISTERED row by row, and the builder that would execute them
spent four of six slots last night watching a process breathe. `D35` and `D36`
are live and are desk-executable, not yours. **I have not opened a new decision
this audit and the arming ratchet shrank by one**, which is the direction it is
allowed to move.

**6. NO-DECISION: the page you read to know what is happening has been dark for
42 hours and the alarm that watches it says OK.** `docs/PROGRESS.md` still opens
`2026-09-24`, carries no staleness marking, and headlines *"THE BUILDER IS BACK
AND HAS NOTHING TO DO"*. The cause is now understood rather than suspected: the
liveness check asserts the Review is alive from a row the **dying** Review writes
to disclose its own death, so the banner it owns never fires (FINDING 3). I have
routed the code repair to the builder with a measured zero staleness bill rather
than leaving it with the organ whose capacity is the problem. I have deliberately
**not** opened a fourth decision on that capacity question, because `D33` and
`D36` already ask it. I am flagging it only so that, if you read `PROGRESS.md`
this weekend, you know it does not yet know the ladder test ran.
