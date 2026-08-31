# OVERSIGHT — 54th audit, 2026-08-31 13:10 UTC (HEAD `2055cb3`, tree clean)

## VERDICT: INTEGRITY RISK — the ladder is losing rungs to a mechanism nothing counts, and all four ratchets return `rc=0` while it happens

Sections 1, 2, 6 and 7 have **no findings**, checked mechanically. The ledger is
clean, no threshold moved in the loosening direction in seven days, and the
builder is doing what it says it is doing. I say this plainly because it is true
and because it is what makes the rest of this report worth reading: the defect is
not in the ledger, and it is not misconduct.

The defect is that **13 of 211 registered specs can never reach PASS under
today's graph, 10 of them were created in the last three days by three
self-certified `VOID-FORECLOSED` declarations, and no instrument in this repo
prints that number.** Among the 13: *"Connected, not two brains: the substrate is
shared"* (`DP.02` — the test GOAL.md names as the one that can quietly fail),
*"Unprompted exploration is real"* (`T5.06` — the north star), *"Open-endedness:
learning does not saturate"* (`T5.08`), *"He eats because he is hungry"*
(`PS.04`), and the whole of the learning-core arbitration (`LC.04`, `LC.05`,
`LC.06`).

**The four mandatory ratchets.**

| tool | rc | reading |
|---|---|---|
| `coverage` | **2** | **RED**, and honestly so — 4 empty classes with no path in. Its constitutional check is green: **0 commitments with no declared spec.** But its per-class advice is suppressed on `cpu<2h`, hiding six implementable specs (§3.c) |
| `decisions --check` | 0 | 0 UNDECLARED, 0 MEANS-ESCALATED, 0 OVERDUE — **eleven defaults fire tomorrow**, one of them seats an architecture (§3.a) |
| `champions --check` | 0 | `ratchet ok` — **and two seats have no runnable challenger; the tool checks existence, not reachability** (§3.a) |
| `run review-queue` | 0 | 0 violations, 9 OPEN / 2 HELD; `w0-too-shallow` re-armed 08-30→09-06 in the open with reasons — the honest repair, correctly taken |

Every ratchet moved in the right direction yesterday, and `coverage`'s red is the
same honest red it has carried all week. That is the finding: the loss is
happening in the space *between* the instruments.

**And the two red-est instruments hand off to each other across that gap.**
`coverage` ends its empty-class report with *"The repair is an UNBLOCK
(`run blocked`), which is a different unit of work"* — it routes the builder
directly to `run blocked` when the queue is empty, which is exactly now. And
`run blocked` currently ranks `LC.03 = VOID frees 8` **second** on its
what-to-fix list, a spec this project has recorded as un-re-runnable (§3.b).

---

## 1. Integrity of the ledger — NO FINDINGS

Checked mechanically over all **94 PASS** rows of 211 registered specs
(13 FAIL, 5 VOID, 0 NOT_RUN, 0 ERROR — 112 rows total, reconciles exactly):

- **94/94** resolve to an implementation file in `experiments/tests/`.
- **94/94** carry a `commit` that still exists in git (`git cat-file -e`).
- **94/94** have a spec that declares a `control`.
- **92/94** carry populated `control_metrics`. The two that do not are `T0.01`
  and `T0.10`, both of which declare `control="NONE, BY DECISION (52nd audit
  B5)"` in the registry — declared, not silently absent. **Zero claim-kind PASS
  rows have an unrun control.**

`run status` integrity lists are otherwise **empty**: the two live defects the
53rd audit named (`T0.10` DRIFTED, `W0.DIAG` dirty-stamped) were both re-bought
from a clean tree and committed (`dad5f27`). The only remaining flag is `T2.02`,
recorded VOID — no certificate at risk.

Standing declared debt, unchanged: **57 PASS rows predate `spec_sha`**, 18
predate `impl_sha` (17 verified byte-identical, 1 stale). Correctly not
back-filled.

## 2. Thresholds and controls, over time — NO FINDINGS

69 commits touched `registry.py`, `registry_expansion.py` and `tests/` in seven
days. Swept for the five loosening signatures:

- **Numeric thresholds:** no constant moved in the loosening direction. `W0.DIAG`
  states its floors verbatim as *"Floors are UNCHANGED"* and `MIN_LIVES_NULL=12`
  is carried forward as LC.03's, not re-fitted. `T3.06`'s `DECAY_MIN` carries an
  explicit *"does not move; re-fitting it after the fact"* note.
- **Seed counts:** one change, `T2.11` **3 → 7** — a tightening. No reductions.
- **`_check` gaining an `or`:** every added disjunction is on a **VOID branch**
  (`|t(gain_random)| or |t(gain_repeat)| >= 3` — *more* ways to void a run), not
  on a PASS branch. `HEADROOM_MIN_MULT` was **added** and is strengthen-only.
- **Controls deleted or weakened:** none. `W0.DIAG` added a mirror control and an
  `eats` conjunct so a statue-ward drift cannot masquerade as exploration.
- **Assertions removed:** none.

## 3. Drift from the goal — THREE FINDINGS

### 3.a — TWO ARCHITECTURAL SEATS HAVE NO RUNNABLE CHALLENGER, AND ONE IS PROMOTED TO THE FILE'S STRONGEST MARKING TOMORROW

SYSTEM.md's standing rule is:

> **No architectural seat may be held without a REGISTERED, EXISTING challenger
> spec.** Not a named one — an existing one, resolvable in `BY_ID`. A seat whose
> arena does not exist is not contestable, whatever the file says.

`champions.py` enforces exactly that sentence: it resolves arena ids against
`BY_ID`. **The loophole is the word EXISTING.** I walked every seat's arena
through the live dependency graph, treating a spec as reachable only if it can
ever become PASS (not `parked`, not `VOID-FORECLOSED`, and every ancestor
likewise). Two seats have **zero** reachable arena members:

| seat | held | arena | why dead |
|---|---|---|---|
| **Learning core** | BY DEFAULT → **BY VERDICT tomorrow** | LC.03–LC.06 | all four sit behind `LC.03`, declared VOID-FORECLOSED with *"no v3, no envelope growth, no re-roll"* |
| **Fast/slow coupling** | **BY DECREE** | DP.02 (sole member) | `DP.02 ← DP.01 ← LC.04 ← LC.03`, same foreclosure |

`champions --check` reports the first as `ok` and the second as `UNCONTESTED` with
the advice **"Schedule it."** It cannot be scheduled. There is no envelope, no
budget and no seed count at which `DP.02` can run, because its root is declared
un-re-runnable.

**Why this is the top finding and not bookkeeping.** `DP.02` is not an arbitrary
spec. GOAL.md, §*Fast and slow, in one brain*:

> **The connectedness claim is the one that can quietly fail**, because an
> architecture can drift into two private towers while every capability number
> keeps improving. So it is tested directly and adversarially: **DP.02** lesions
> the shared trunk and requires BOTH modes to degrade together…

The project's own written defence against its most-feared silent failure is
currently unreachable, and the instrument built to notice exactly this says `ok`.

**And it gets worse tomorrow.** `D10`'s armed default fires 2026-09-01 and reads:

> *…`wm-latent` takes the learning-core seat as measured winner-by-default
> (CHAMPIONS.md idiom, seat marked **BY VERDICT** with the single-arm caveat on
> its face).*

That promotes the single most load-bearing architecture choice in the project
from `BY DEFAULT` (weakest marking) to `BY VERDICT` (strongest) **on the day its
arena became permanently unrunnable**, and `champions --check` will still print
`ratchet ok`, because `LC.04` resolves in `BY_ID`.

`CHAMPIONS.md` already says this in its own challenger cell, which makes it
harder to defend, not easier:

> *"`LC.04` — the seat's actual match — is blocked behind an `LC.03` PASS the
> fork says will not be manufactured… **`wm-latent` is NOT seated**: that is D10
> option (a), not the harvester's to take."*

D10 option (a) is what fires tomorrow. **I am not asking for the deadline to
move** — a deadline that moves when it is reached is the deadlock it replaced,
and D10's default is otherwise sound and honestly argued. The repair is the one
the constitution already prescribes for `ARENA-MISSING`: **register the
challenger**, in the same commit that fires the default. D10's own text names a
runnable one (§B1).

### 3.b — THE FORECLOSURE BLAST RADIUS IS NEVER PRICED, AND `run blocked` RANKS A CLOSED DOOR SECOND

Three specs were declared `VOID-FORECLOSED` in three days (`BA.03` 08-31 03:16,
`T3.06` 08-31 ~05:07, `LC.03` carried). Each was journalled as a **saving** —
*"four hours saved"*, *"queue depth 3→2"*. Nobody computed the other side of the
ledger. I did:

| foreclosure | downstream specs it alone renders permanently unreachable |
|---|---|
| `BA.03` | **0** |
| `T3.06` | **2** — `T5.06` *"Unprompted exploration is real"*, `T5.08` *"Open-endedness: learning does not saturate"* |
| `LC.03` | **8** — `DP.01`, `DP.02`, `DP.03`, `LC.04`, `LC.05`, `LC.06`, `OP.01` *"A thing behind the rail still exists"*, `PS.04` *"He eats because he is hungry"* |

**Total permanently unreachable today: 13 of 211 (6.2%)**, including two Tier-5
claims — the thesis itself. The 05:07 iteration bought four CPU-hours of savings
and paid for them with *"Unprompted exploration is real"*, which is the sentence
GOAL.md opens with. It did so in good faith, because nothing told it the price.

`run blocked` is the instrument that should catch this and **half-catches it,
which is worse**. It correctly lists `LC.03` and `T3.06` as terminal blockers —
but it prints them as `= VOID`, identically to `T2.02`, which is a genuinely
repairable VOID. So its ranked "what one fix would be worth the most" list reads:

```
  T2.01 = FAIL  frees 35
  LC.03 = VOID  frees  8   <- cannot be fixed. Declared un-re-runnable.
  NE.01 = FAIL  frees  8
```

`experiments/run.py` contains **zero references to `void_foreclosed`** (grepped).
`coverage.py` reads that declaration and excludes foreclosed specs from the
queue; `run.py` does not. **Two instruments disagree about the same three specs**,
and the one the builder consults to pick high-leverage work is the one that is
wrong.

This is the third instance of a lesson this repo already wrote down.
`LESSONS.md:150` — *"A dependency graph can quietly make your most important
claim unreachable"* — records that "periodically ask" failed twice as human
advice before becoming `run blocked`. It has now failed a third time, through a
door `run blocked` cannot see: a spec that is not merely blocked but *closed*.

### 3.c — `coverage.py` HIDES SIX IMPLEMENTABLE SPECS BEHIND ONE VOID ROW

The queue readout prints:

```
      cpu<10min   0   EMPTY   <- fillable today: LG.10, ME.11, ME.11.B, ME.11.C, ME.11.E
      cpu<2h      1   BA.02
```

`cpu<2h` gets **no advice line at all**. But the tool computed one and threw it
away — `queue_depth()['fillable']['cpu<2h']` holds:

```
['LG.02', 'LT.01', 'ME.11.D', 'ME.11.F', 'T3.09', 'UB.14']
```

Six implementable specs, including **`LT.01`** — the builder's own named next
unit — and **`ME.11.D` / `ME.11.F`**, two members of the *Episodic retrieval*
champion arena. Cause, `experiments/coverage.py:1664`: the advice tail is gated on
`if ids: tail = ""`, i.e. on the class being **EMPTY**, not on it having **zero
fresh dispatches**. `BA.02` is a VOID; the headline line correctly says *"of
which 2 VOID → only 0 is a FRESH dispatch"*, and then the per-class lines — the
part actually read when choosing work — present `cpu<2h` as served.

The timing makes it concrete: **D8's default parks `BA.02` tomorrow.** At that
moment `cpu<2h` flips to depth 0 and the hint appears. Six specs will become
visible not because anything was learned but because a park emptied a row.

### 3.d — the converse check: what has no passing spec at all

`coverage` reports **0 commitments with no declared spec** and **0 CLAIM-DEAD**,
but `CLAIM-DEAD` counts *parked*, not *unreachable*. Under the reachability
reading, **fast/slow** is materially claim-dead: `DP.01`/`DP.02`/`DP.03`
permanently unreachable, `DP.04` `PILOT-BLOCKED` (its own record says *"the
repair is a world/metric redesign, not a pilot"*), `BO.01` behind `DP.05 = FAIL`.
Coverage scores it *"1 now"* runnable. Nothing in that family can be run today.

Of the **12 commitments with live claim specs but nothing passing**, the ones
most exposed remain: proprioception, shelter/building, thermal, plasticity,
balance, sleep, hunger/thirst, fast/slow.

### Work in the last 24 h, and which GOAL.md sentence each serves — no drift

| unit | serves |
|---|---|
| `LG.00` PASS (Tier 4) | *"The LLM is his mouth, never his mind… strip the diary and the learned core, and his answers about his own life must COLLAPSE"* — **a real capability, and a central one** |
| `LG.01` PASS | fixture for the above |
| `W0.DIAG` PASS | *"the world must be consistent, discoverable, consequential"* — a negative result on W0, honestly bought |
| `T0.29`, `T0.31` PASS | *"protects the honesty of watching what happens"* — instruments |
| `LT.01`–`LT.09` registered | *"a ladder with an apple on top… purely out of curiosity"* — the north star's arena |
| `BA.03`, `T3.06` foreclosures | queue hygiene — **and §3.b is their unpriced cost** |
| `T0.10`, `T0.01` re-runs | ledger integrity |

Every item traces to a GOAL.md sentence. **No drift.**

## 4. Is the builder alive and productive? — NO FINDINGS

- **24 iterations** in 24 h, **24 ended `rc=0`**. No paused loop, no credit
  exhaustion, no aborts on load, no repeated identical failures.
- **PASS delta: 90 → 94 (+4).** Registry **198 → 211 (+13)**, so the ratio moved
  45.5% → 44.5% — the denominator grew faster, which is correct behaviour when
  a champion arena gets written.
- Model discipline is clean: the 04:07 iteration **refused Fable** at the 95%
  floor and named `week:all models` as the gate it acted on, per D14.
- All six of the 53rd audit's builder items: **B1 done** (LT family registered,
  and it moved the ratchet as predicted, UNFALSIFIABLE 5→4), **B2 done**,
  **B3 done** (all eleven defaults now name an executor and artifact, `03f31cf`),
  **B4 done** (`T0.10` re-bought), **B5 NOT DONE** (§ below), **B6 not taken** —
  deferred by name to `LT.01`, which is a defensible substitution but leaves
  `cpu<10min` empty.

**B5 is the one to notice.** `FORECLOSURE ARITHMETIC:` does not exist anywhere in
the repo (grepped). The 53rd audit asked for it on the grounds that *"the
declaring party is the party it exonerates"*. §3.b is what that ungated
arithmetic bought in the eight hours after the request: `T3.06` was foreclosed
at 05:07, killing two Tier-5 claims, with no adversarial check and no blast
radius attached.

## 5. Compute honesty — NO FINDINGS, but the waste continues

| week (`%U`, Sunday-start, matches Kaggle's reset) | charged | wasted on failures |
|---|---|---|
| 2026-W32 | 16.61 h | 1.18 h |
| 2026-W33 | 7.89 h | 0.26 h |
| 2026-W34 | 1.62 h | 0 h |
| **2026-W35** (current) | **1.28 h** | 0 h |

Every charged hour reconciles to a ledger entry; no unaccounted GPU time.
**~28.7 h of the 30 h weekly Kaggle quota remain and six days to spend them**,
and the honest reason nothing is spendable is unchanged: `gpu<20min` and `gpu<2h`
are both PILOT-BLOCKED pending redesigns. Cumulative expiry across five weeks is
now on the order of **110–120 free GPU-hours**.

## 6. Stuck decisions — NO FINDINGS

`0 MEANS-ESCALATED`, `0 UNDECLARED` (0/10), `0 OVERDUE`. Eleven defaults read
`due 2026-08-31` — **today is the owner's reserved last day**; `overdue` is
`(today − due) > 0`, so they fire 09-01. The builder declined to fire early with
the reason stated in the open (*"today is the owner's reserved last day, so
firing early would have eaten it"*) — correct, and the opposite of the D1
disease. No owner decision was acted on without being recorded: I checked
`DECISIONS_RESOLVED.md` against the 24 h of commits and found no unrecorded
resolution.

## 7. Bakeoff hygiene — NO FINDINGS, with one flag carried to §3.a

`DECISIONS_RESOLVED.md` holds three entries; none is a decision without a
learning gate and none picks a winner inside the noise margin. The `PS.01/J`
VOID is recorded as a VOID, not as a verdict.

The one case that *looks* like "a VOID treated as a verdict" — D10 seating
`wm-latent` off `LC.03`'s VOID — survives scrutiny **as a decision**: the fork
was pre-registered 2.5 days before the number landed, every control landed on its
pre-registered side, `wm-latent` was the sole 3σ learner (`t_null` 4.65,
`t_twin` 4.00), and the single-arm caveat is required on the seat's face. Seating
the only arm that cleared the null is defensible. **What is not defensible is
doing it while the arena goes dark** — that is §3.a, and it is a champions
problem, not a bakeoff-hygiene problem.

## 8. The honest summary — are we closer to a curious humanoid that climbs the ladder?

**Yes, in one real way, and the count flatters it.**

The real way: `LG.00` passed. *"Jack knows what his LLM cannot — he is not a
puppet"* is a Tier-4 claim at the centre of GOAL.md, and it is now on the ledger
with its control. That is one of the genuinely hard ones and it is a good day's
work. And `LT.01`–`LT.09` mean the Ladder Test — the literal ladder, the literal
apple — has an arena for the first time in the project's life.

Where the count flatters: of the +4 PASS in 24 h, **one** is a Jack capability.
Two are instruments about the project, one is a fixture, and `W0.DIAG`'s finding
is *negative* — correlated **random** action buys life in W0 through food. Set
that beside yesterday's t = 0.39 for curious-over-random and the picture is
consistent and uncomfortable: **nothing yet demonstrates that Jack's curiosity
outperforms noise in the world he currently lives in.** The `w0-too-shallow`
design due 09-06 is the correct response and it is properly clocked.

And the thing that makes this an INTEGRITY RISK rather than a slow week: **the
ladder is quietly getting shorter at the top.** *"Unprompted exploration is
real"* and *"Open-endedness: learning does not saturate"* — Tier 5, the thesis —
became unreachable at 05:07 this morning as a side effect of a queue-hygiene
commit that reported itself as saving four hours. *"Connected, not two brains"*
has been unreachable since LC.03 concluded. Tomorrow the learning-core seat gets
the file's strongest marking with nothing left that could unseat it. Every one of
those events passed through four green ratchets.

The system's instruments are excellent at asking *"does the challenger exist?"*
They have never once asked *"could it be run?"* — and 13 specs have now
accumulated in the gap between those two questions.

---

# FOR THE BUILDER

Ranked by damage to the trustworthiness of the ledger. None of these asks you to
move a threshold, delay a deadline, or delete a row.

**B1 — Fire D10's default tomorrow AS WRITTEN, and register a runnable
learning-core challenger in the SAME commit.** Do not delay it; the date is the
owner's and it has been honoured. But the commit that marks the seat `BY VERDICT`
must not leave it with a dead arena. D10's own default text already names the
challenger: *"the owner's scale-transfer guard still binds BEFORE adoption:
re-test at ~10× on Kaggle, which is free."* Register that re-test as a spec whose
`depends_on` **does not route through `LC.03`** — it depends on `LC.00`–`LC.02`
(all PASS) and on the recorded `wm-latent` result, not on the screen. It is
GPU-cheap in a week with ~28.7 h idle, and it converts the project's most
load-bearing seat from unfalsifiable to contested without weakening anything.
The ratchet shrinks; it does not grow.

**B2 — `run blocked` must read `protocol.void_foreclosed`.** `experiments/run.py`
has zero references to it while `coverage.py` reads it, so the two instruments
disagree about `LC.03`, `T3.06` and `BA.03`. Today `run blocked` ranks `LC.03`
**second** on the "what one fix would be worth the most" list — a door declared
closed. Minimum repair: print foreclosed roots as `VOID-FORECLOSED` and rank them
in a separate section headed *"these do not free anything by being re-run — the
repair is a re-parenting or a redesign"*. Cheap, and it puts the two readers back
in agreement.

**B3 — Price the blast radius INSIDE the foreclosure declaration, and discharge
the 53rd audit's B5 in the same edit.** A `VOID-FORECLOSED:` block currently
states why re-running is futile and says nothing about what it kills. It should
be refused unless it carries both:
  - `FORECLOSURE ARITHMETIC:` — the multiplier on N that would clear the bar (the
    53rd audit's B5, still open; the declaring party is the party it exonerates);
  - `BLAST RADIUS:` — the computed list of specs this declaration renders
    permanently unreachable, with their titles. `T3.06`'s would have read
    *"kills T5.06 'Unprompted exploration is real', T5.08 'Open-endedness'"*,
    and I do not believe that commit would have been written the same way.

**B4 — `champions.py` must count REACHABILITY, not just existence — as a NEW
class, added to the ratchet total.** SYSTEM.md says *"A seat whose arena does not
exist is not contestable, whatever the file says."* The sentence needs its second
half: an arena that cannot be **run** is not contestable either. Proposed class
`ARENA-UNREACHABLE` — every arena member is `parked`, `VOID-FORECLOSED`, or
transitively behind one. It fires on exactly two seats today (**Learning core**,
**Fast/slow coupling**) and both are real. Per `T0.31`'s precedent, add it to the
**total** the baseline asserts on — do not let a new class arrive with its own
private zero.

**B5 — `coverage.py:1664`: gate the advice tail on ZERO-FRESH, not on EMPTY.**
One line. `if ids:` should be `if ids and not all(i in q["void"] for i in ids):`.
The list is already computed. Today it hides `LG.02`, `LT.01`, `ME.11.D`,
`ME.11.F`, `T3.09`, `UB.14` behind one VOID row on `cpu<2h`, and two of those are
*Episodic retrieval* arena members. Add a fixture case for it: a class whose only
occupant is VOID must still advertise what would fill it.

**B6 — Re-parenting is the repair for the 13, and nobody has been told it is
owed.** After B2–B4 make the set visible, the specs are not lost — `LESSONS.md`
records the precedent (re-parenting `UB.1`–`UB.8` off `T2.01` made eight
immediately runnable, and a challenger registered as a NEW spec bills nothing,
T1.02 precedent). `DP.02` in particular should not need `LC.03`: *"lesion the
shared trunk, both modes degrade together"* is a probe on a trained core, not a
claim that requires a five-way screen to have returned two winners. Route this to
`REVIEW_QUEUE.md` as its own row with a `DUE:` — it is a design question, and
this desk is where design questions go.

**B7 — `cpu<10min` is still EMPTY and still fillable** (`LG.10`, `ME.11`,
`ME.11.B`, `ME.11.C`, `ME.11.E`). Carried unchanged from the 53rd audit's B6.
`ME.11.*` is the Episodic-retrieval arena, so one implementation clears a
known-empty class and moves a champion contest at the same time.

# FOR THE OWNER

**O1 — Eleven of your decisions default tonight at midnight.** D1 (blocks 38
specs), D10 (8), D4 (8), plus D3, D7, D8, D9, D11, D12, D13, D14. Every one was
armed with a written default and a deadline you were given; I checked the file's
history and none was silently extended. **Today is the last day.** Read D1 and
D10 first.

**O2 — One of those defaults deserves ninety seconds of your attention
specifically.** D10 seats `wm-latent` as Jack's **learning core** — the single
biggest architecture call in the project — and marks the seat `BY VERDICT`, the
strongest marking `CHAMPIONS.md` has. The evidence behind it is honest: it was
the only one of five candidate cores that learned to survive at all, at 3σ, with
a clean rig and a fork pre-registered days in advance. **The problem is not the
choice. It is that after tomorrow there is no experiment left in the ladder that
could ever unseat it** — every challenger sits behind a screen the project has
recorded as un-re-runnable. `CHAMPIONS.md` says so itself, in the seat's own
challenger cell. I have asked the builder to fire your default on time and
register a runnable challenger in the same commit (the ~10× Kaggle scale-transfer
re-test your own guard already requires). If you would rather rule on it
directly, that is your call and today is when you have it.

**O3 — The state of the project, in one paragraph.** The ledger is honest: 94
PASS of 211, every implementation present, every commit resolvable, every control
declared, zero loosening in seven days, 24 of 24 iterations clean. Yesterday
bought a genuinely important result — `LG.00`, *"Jack knows what his LLM
cannot"*, the anti-puppet claim. Against that: **nothing yet shows Jack's
curiosity beating noise in the world he lives in** (curious-over-random t = 0.39;
correlated random action buys life in W0 through food), ~28.7 free GPU-hours will
expire unspent this week for the fifth week running, and **13 specs — including
two Tier-5 claims and the one-brain connectedness test — can no longer be reached
at all.** Nothing here was dishonest. Ten of those thirteen were created by a
tool that reports foreclosure as a saving and has never been asked to report its
cost.

**O4 — The 09-06 Review is now carrying more than the world design.** It owes
`w0-too-shallow` (re-armed 08-30 → 09-06 in the open, with reasons — the correct
repair), plus `ba03-null-saturates-the-horizon` and
`t306-matched-magnitude-noise-buys-coverage`, and after B6 it should also carry
the re-parenting of the 13. It has now been **four Sundays without a completed
FULL run**; the max-turns cause is fixed (`9f4b8da`, 120 turns). The fifth one
matters more than the previous four did.
