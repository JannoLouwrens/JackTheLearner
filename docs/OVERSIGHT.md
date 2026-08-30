# OVERSIGHT — 50th audit, 2026-08-30 12:45 UTC

## VERDICT: ON TRACK — with one assurance gap that fires tomorrow

I checked sections 1, 2 and 7 mechanically rather than by reading commit
messages, and **they have no findings**. All 90 `PASS` rows resolve to an
implementation that calls `run_spec`, to a commit that exists in git, and to
recorded `control_metrics` wherever the spec declares a control. Exactly one
numeric bar moved downward in seven days (`DECAY_MIN` 1.5 → 1.25) and it was
declared in the open, derived from first principles rather than from the data,
and applied to a spec that had never run. No seed count was reduced; the one
that changed went 3 → 7. No control was deleted, no `_check` gained a
disjunction that admits a failure, no assertion was removed.

Of the three mandatory ratchets, two are green and **one is RED**: `decisions`
`rc=0` (0 undeclared, 0 `MEANS-ESCALATED`), `champions` `rc=0` (7/7
unfalsifiable against a baseline of 7, 5/5 phantom arenas) — but **`coverage`
exits 2**. Its *commitment* ratchet is clean (0 uncovered, 0 `CLAIM-DEAD`); what
is red is the queue ratchet, `2 cost class(es) NEWLY EMPTY — gpu<20min, gpu<2h`
(`coverage.py:479-488`, which exits 2 rather than quietly re-baseline an empty
class). That red is §5 of this report and it is the honest one. The builder ran
**24 iterations
in 24 h, 22 at `rc=0`**, two timeouts, and moved the ladder **84 → 90 PASS of
198**.

**What keeps this from being unqualified, and it is carried rather than new:**
the detector for the disease that cost this project twenty days — `MEANS-
ESCALATED` — fires only when the author of an entry types `class: means`, and
**13 of 13 open entries type `goal`**. It has never fired on live data and
cannot. I read all thirteen and I am *not* claiming any is misfiled. This is a
gap in assurance, not evidence of a false escalation. But eleven pre-registered
defaults fire in under twelve hours, `SYSTEM.md` itself records that two of the
three safety clauses governing them are still enforced by nobody, and this is
the instrument standing over that event.

---

## 1. Integrity of the ledger — NO FINDINGS

104 rows: **90 PASS, 10 FAIL, 4 VOID**, of 198 registered specs.

Checked mechanically over every `PASS`:

| check | result |
|---|---|
| `commit` resolves in git (`cat-file -e`) | **0 phantom commits** of 90 |
| an implementation exists that calls `run_spec(BY_ID[...])` | **90 of 90** |
| implementation passes `control_fn=` | **88 of 90** |
| spec declares a `control` | **88 of 90** |
| `control_metrics` recorded when a control is declared | **88 of 88** |

The two exceptions are `T0.01` (repo imports clean) and `T0.10` (Kaggle job
round-trip). Both declare `control=None` and `null_baseline="n/a — structural
precondition."` **in the registry** — the absence is a recorded decision, not an
omission, and neither asserts a capability. This is the 49th audit's B6(a), still
open as a one-sentence docstring edit; it costs nothing and I am not re-ranking it.

Non-PASS rows, all carrying honest verdicts with a `commit` that resolves:
`T0.27`, `T2.01`, `T2.05`, `T2.07`, `T2.15`, `T3.07`, `T4.02`, `NE.01`, `DP.05`,
`XL.01` FAIL; `BA.02`, `LC.03`, `T2.02`, `T3.06` VOID.

## 2. Thresholds and controls over time — NO VIOLATIONS, one wording defect

Every numeric constant that moved in `registry.py`, `registry_expansion.py`,
`experiments/tests/`, `protocol.py` and `bakeoff.py` over seven days, with its
direction:

| spec | constant | move | direction |
|---|---|---|---|
| VO.02 | `COORD_MIN` | 0.55 → **0.70** | tightened |
| VO.02 | `COORD_MARGIN` | 0.20 → **0.35** | tightened |
| T2.19 | `SHUF_MULT` | 2.0 → **10.0** | tightened |
| T2.19 | `RATIO_MIN` | 3.0 → **10.0** | tightened |
| T2.19 | `UNI_MIN` | 0.8 → **0.9** | tightened |
| T2.19 | `TIE_BAND` | 0.15 → **0.10** | tightened |
| T2.19 | `STEPS` | 300 → **500** | more evidence |
| T3.06 | `LIVES_PER_ARM` | 4 → **16** | tightened |
| T2.09 | `seeds` | 3 → **7** | tightened |
| T2.09 | `DECAY_MIN` | 1.5 → **1.25** | **loosened — see below** |

**`DECAY_MIN` is not a violation and I want to be precise about why.** It is a
*rig* bar (seed informativeness), not a claim bar; it was a placeholder on a spec
whose `run()` refused until the freeze commit; the move is justified from what
the gate is *for* — a constant signal decays by exactly 1.0, so any bar above 1.0
excludes a dead one — rather than shaved to the observed minimum (1.472); and
`44f24c4` states it under the heading `ONE BAR MOVED, DOWNWARD, IN THE OPEN`.
That is law 4 obeyed, not evaded.

**The wording defect.** `44f24c4` says the seed-selection formula *"reads only
the null and the rig instruments — never the claim arm's dwell, fed-ratio,
coverage or margin."* The enumeration is true. The summary clause is not:
`t2_09_*.py:583-589` gates informativeness on `claim_static_reward_q1` and
`claim_static_decay`, which are claim-arm measurements (in the static world), and
on `exposure_frac_of_random`, which is the claim arm's exposure. Three of the six
conditions touch the claim arm. **Live effect on `T2.09`'s PASS: zero** — I read
the recorded `per_seed` table and all three exclusions (seeds 2, 4, 5) fired on
`trap_dwell` 0.0/0.0437/0.0 against `TRAP_DWELL_MIN` 0.40, a pure rig instrument;
`static_decay` was ≥ 1.424 on all seven seeds and excluded nobody. The mechanism
is sound and fully auditable from the ledger. It is the *sentence* that promises
more than the code delivers, and a future reader will rely on the sentence.

**One number in that row deserves a guard, not an alarm.** Seed 1 recorded
`trap_ratio` **953,594,661,617.28**. That is a near-zero denominator, not a
spectacular trap — out-zone reward was ~0. It clears `TRAP_RATIO_MIN` 2.0
trivially and in the correct direction, so `T2.09`'s verdict is unaffected. But a
9.5e11 sitting in a certified ledger row is indistinguishable from a bug on
sight, and the next person to read it will have to re-derive what I just did.

**Worth recording as a positive:** `T2.09`'s freeze *found* a live vacuity — the
docstring said "worst of 3 seeds" and `run_spec._aggregate` hands `_check` the
**mean**, so a bimodal trap `[1.0, 1.0, 0.0]` would have cleared a gate on a seed
where the trap never fired. `_fold` now reads the worst informative seed. That is
the second-order loop working exactly as `SYSTEM.md` describes.

## 3. Drift from the goal — none in what was built; the gap is where nothing passes

Everything the builder touched in 24 h traces to a `GOAL.md` sentence:

| unit | GOAL.md sentence |
|---|---|
| `VO.02` PASS | *"and VOICE — he must be able to make sound"* — **voice's first passing claim in the project's life** |
| `SH.02` implemented + piloted | *"too cold kills him"* / *"he builds a shelter"* |
| `SM.03` pilot | *"SMELL … finds food, fire and decay at a distance"* |
| `T2.14` harvested | unison / real human motion |
| `W.1`–`W.8` registered | *"the world must be consistent, discoverable, consequential"* |
| `T0.28`/`T0.29`/`T0.30` | `SYSTEM.md` conduct — *"protects the honesty of watching what happens"* |

**No drift.** The converse question is the live one. `coverage` reports **12
commitments with live claim specs and nothing passing**, and the shape of the
zero-pass list is the thesis itself:

- **fast/slow** — 8 specs, **0 PASS** (`BO.01`, `DP.01`–`DP.04` all blocked)
- **one brain / unison** — 21 specs, **1 PASS**
- **sleep** — 5 specs, **0 PASS** (all four claims blocked on `T2.01`/`NE.01`)
- **hunger/thirst** — 6 specs, **0 PASS**
- **thermal (kills)** — 4 specs, **0 PASS**
- **death & retry** — 3 specs, **0 PASS**
- **curiosity** — 12 specs, **2 PASS**

Curiosity, all-senses fusion and learning-by-living are exactly the families
`GOAL.md` warns are "most likely to be quietly neglected in favour of easy
wins." They are not being neglected — `T2.09` (noisy TV) and `T2.19` (fusion)
both landed this week — but they are where the ladder is thinnest, and most of
them are blocked behind one decision (§6).

## 4. Is the builder alive and productive — YES

- **24 iterations** in the last 24 h, hourly, no gap. Loop alive now (pid 3010066).
- **22 `rc=0`, 2 `rc=124`** (00:57, 05:57 — timeouts, both inherited cleanly by the next unit).
- **PASS 84 → 90** (+6), registry 187 → 198.
- Model: `week:Fable` pinned at 100% since the 08-31 reset window opened; the
  fallback chain walks to Opus in ~3 s every hour and logs it. `week:all models`
  — the gate — reads **81%**. Pacing streak **0**, no blackout. This is working
  as designed and I found no dispatch-then-idle.
- Not one iteration ended in credit exhaustion, an unresumed pause, or an
  identical repeated failure.

**Half the day's PASS growth is the machine testing itself.** `T0.28`, `T0.29`
and `T0.30` are all governance instruments (`decisions.py`, `champions.py`, the
gate precondition), and the `T0` family is now **29 of 90 PASS (32%)**. I am not
calling that drift — `SYSTEM.md` explicitly says a session that makes the machine
better at catching its own errors has done the whole job — and all three were
this auditor's RANK 1 from yesterday. But the hard constraint's corollary applies
now: *"when the machine is sufficient, PROVE it by throughput."* Two of the three
tools every audit opens with are certified as of today. That debt is paid. The
next iteration should not spend itself on a fourth.

## 5. Compute honesty — 59.3 free GPU-hours have expired, and the cause is structural

| week | Kaggle used | of 30 h | **expired** |
|---|---|---|---|
| 2026-W32 | 21.18 | 30 | **8.82** |
| 2026-W33 | 7.89 | 30 | **22.11** |
| 2026-W34 | 1.62 | 30 | **28.38** |
| 2026-W35 | 1.01 | 30 | ~29 remaining, **6 days left** |

**59.31 free GPU-hours expired unspent across three weeks.** Every charged job in
`gpu_budget.json` has a ledger entry or a recorded failure — the accounting is
honest and I found no hours spent with nothing to show. The waste is not
overspending; it is *unspending*.

The cause is measured, not guessed. `coverage`'s queue-depth instrument reports:

```
gpu<20min   0   EMPTY   <- NOT FILLABLE: no runnable spec to implement
gpu<2h      0   EMPTY   <- NOT FILLABLE: no runnable spec to implement
gpu<8h      1   T2.02   (VOID — an arm to repair, not a dispatch)
```

and states that *every unimplemented spec at those costs is blocked upstream*.
The chain is short and it is one link: **`D1` open → `T2.01`/`T2.02` blocked →
the GPU queue cannot be filled → the weekly quota expires.** The builder was
right not to hunt for a GPU spec this week; the repair is an unblock, and the
unblock is a decision that fires tomorrow.

## 6. Stuck decisions

**None overdue.** 13 open, all armed, `ratchet ok (0/10 undeclared)`.

**Eleven defaults fire 2026-08-31 — tomorrow.** Ranked by the computed cost:

| id | costs | blocks | due |
|---|---|---|---|
| **D1** | **38 specs** | `T2.01`, `T2.02` | 2026-08-31 |
| D10 | 8 specs | `LC.03` | 2026-08-31 |
| D4 | 8 specs | `LC.03` | 2026-08-31 |
| D11, D12, D3, D7, D8, D9 | 0 specs | — | 2026-08-31 |
| D15, D16 | 0 specs | — | 2026-09-05 |

**`D1` has been open 21 days and is the single most expensive object in this
project.** Its default is the right shape — the `PLASTIC-ONLY` decree stands
unnarrowed, option A struck as unconstitutional, and the remaining arms
(A-prime / B / C / D) go to a bakeoff. Nothing about it needs my intervention;
it needs tomorrow to arrive and the bakeoff to be written the same day.

**The instrument standing over all eleven cannot see the thing it was built to
see, and there is now a documented instance.** `experiments/decisions.py:160`
reads `CLASSES = ("means", "goal")`. `SYSTEM.md` was amended on 2026-08-24, at
the owner's insistence, to **three** classes — ENDS / ARCHITECTURE / CONDUCT —
because *"a binary means/goal split silently files every architectural bet on the
goal side, where nothing may test it."* The tool still implements the split the
owner rejected. Consequences I verified:

1. **`MEANS-ESCALATED` is keyed on a self-report.** `audit()` reads
   `d.get("class")` and takes the author's word. Nothing resolves it against the
   arms' runnability. All 13 live entries declare `goal`; the violation has never
   fired outside `decisions.py`'s own fixture, where `D91` is a string the same
   author typed. This is author self-certification — the precise pattern
   `SYSTEM.md` had to correct for the *safety* clauses on 2026-08-30, six days
   late, one function further up the same file.
2. **The document has already been bent to fit the instrument.** Commit
   `d461e36` is titled *"D16: class conduct -> goal (decisions.py CLASSES is
   ("means","goal"))"*. The author filed `D16` under `SYSTEM.md`'s correct third
   class, the tool refused it, and the *entry* was changed. The reasoning
   recorded in `DECISIONS_NEEDED.md` for accepting `goal` is sound on its own
   terms — but the sequence is the wrong way round, and it is now precedent.
3. **`decisions.py` resolves no ids.** `champions.py` checks every arena against
   `BY_ID` and drove that ratchet 8 → 6. A `DECIDE:` block may name a phantom
   instrument forever and every organ reports it correctly armed. This is how
   `D13` came to name `SY.01` — a spec that occurs *once* in this repository,
   inside the sentence claiming it would settle the question.

Findings 1 and 3 are **carried**, not new: the 47th audit filed them (B3) and the
49th filed finding 1 as a docstring edit (B6b). Two days on, nothing has landed.
Finding 2 is new evidence that the gap has started to cost something.

**To be fair to the builder:** I read all thirteen entries against
`SYSTEM.md`'s own criterion (*"a measurement may choose among permitted arms; it
may never choose what is permitted"*) and **found no misfiled entry**. `D1`'s
residue is genuinely constitutional — its header narrows it to *"does
PLASTIC-ONLY admit a frozen control trunk?"*, which no experiment can answer.
`D16` reaches `goal` by correct reasoning. Nothing here is a live false
escalation. It is that the guarantee is currently worth exactly the honesty of
whoever types the field.

**Nothing was quietly acted on without being recorded.** `DECISIONS_RESOLVED.md`
carries `D2` (VOID blocks its dependents, resolved by ledger replay) and the
`PS.01/J` family with its losers.

## 7. Bakeoff hygiene — NO FINDINGS

`PS.01/J` returned **VOID** with three arms below the 3.0σ learning gate and was
recorded as VOID, not decided — the gate doing its job. `PS.01/J2` names a winner
(`impact_speed`) with its margin. `D2` was resolved by ledger replay with the
mechanism recorded and a re-open trigger attached. **No decision was made without
a learning gate, no VOID was treated as a verdict, and no winner was chosen
inside a noise margin.**

Two VOIDs fired *today* by their own pre-registered rules and correctly stopped
work rather than producing a number: `SM.03`'s pilot (held-out split saturated;
`vis_open` 0.1167 against a 0.60 floor) and `SH.02`'s (`headroom_twin` 1.0
against `HEADROOM_MAX` 0.85). Neither dispatched. Neither froze a gate. Both cost
CPU minutes instead of GPU hours. That is the machine working.

## 8. The honest summary — are we closer to a creature?

**Yes, and by more than a tick count — but the thing we got closer to today was a
diagnosis, not a capability.**

The real gain of the last 24 h is `VO.02`: two learners sharing no parameters
invented a shared signalling system, coordination 0.9983 against a chance floor
of 0.250, and **all three nulls died in different ways**. Voice is a
constitutional sense that had zero passing claims for this project's entire
life. That is a creature-shaped result and it is on the ledger.

The rest of the day was the system finding out where it is stuck, which is worth
more than it looks. `SH.02` joins `LC.03`'s darkroom, `LC.03 v2`, `DP.05` and
`SH.01`'s `ORACLE_CANNOT` as the **fifth independent instrument saying the same
thing: W0 — the world — is the measured bottleneck, not the learning core.**
`SM.03`'s saturated split is arguably a sixth. Five instruments converging is not
a bad day; it is a finding the project could not have reached by argument.

**And here is what makes it the sharpest item in this report.** The `World` seat
in `CHAMPIONS.md` is held **BY VERDICT** — the strongest marking in the file — on
a measurement that says MuJoCo is *"4–6× faster than Craftax AND goal-aligned"*.
That is a **speed** verdict. The world's *fidelity* has never been measured at
all. `W.1`–`W.8` — the gates that would measure it — were registered this
morning, after five audits asked, and **none has run**. `W.1` (*"temperature
obeys the heat balance we published"*, `Budget.CPU`, deps `PG.1`/`PG.8`, both
PASS) and `W.2` (*"needs are a conserved ledger, and they can kill"*,
`Budget.CPU`, dep `PG.8`, PASS) are both in `coverage`'s **`cpu<10min` fillable
today** list.

So: the loop's own five converging measurements point at the world; the world's
seat is held on the wrong kind of evidence; the specs that would test it are
registered, unblocked, and cost under ten CPU-minutes each; and today's iteration
implemented a governance instrument instead. That is not dishonesty and it is not
drift — every one of those choices was defensible in isolation. It is the
gravitational pull of the thing that is easy to certify over the thing that is
hard to face.

We are closer to a curious humanoid than yesterday. We would be closer still if
tomorrow spent itself on `W.1`.

---

## A note on method, because it nearly changed this report

I first ran `coverage` as `python -m experiments.coverage 2>&1 | tail -60; echo
"EXIT=$?"`. It printed `EXIT=0`. **`$?` after a pipeline is `tail`'s status, not
the tool's** — the tool exits 2. I had drafted the sentence *"the three
mandatory ratchets are green"* and caught it only by re-running the checks bare
before committing. The failure mode is one-directional: `tail`, `head` and
`grep` essentially always succeed, so a masked exit code is essentially always
green. Every organ in this repo shells out and `sh` has no `pipefail` by
default, so `$(cmd | head)` inside `ladder_loop.sh`, `review.sh` or
`field_watch.sh` has the same defect with no `EXIT=` line to expose it.
Recorded in `docs/LESSONS.md`. **When a command's exit code is the evidence, run
it bare and capture the status before filtering.**

---

## FOR THE BUILDER

**B1 — run `W.1` and `W.2`. RANK 1, and it is the cheapest high-value unit on
the board.** Five instruments now say W0 is the bottleneck (`LC.03` darkroom,
`LC.03 v2`, `DP.05`, `SH.01 ORACLE_CANNOT`, `SH.02` headroom). The `World` seat
is held **BY VERDICT** on a *speed* measurement — 4–6× faster than Craftax — and
its fidelity has never been measured. You registered `W.1`–`W.8` this morning;
none has run. `W.1` (`Budget.CPU`, deps `PG.1`/`PG.8` both PASS) and `W.2`
(`Budget.CPU`, dep `PG.8` PASS) are both unblocked and both sit in the
`cpu<10min` **fillable today** class. `W.3` (*"cold kills, and shelter is why it
does not"*) depends on exactly `W.1` + `W.2` and is the registered instrument for
the question `SH.01` and `SH.02` have now failed to reach from two exhaustive
geometries. **Do `W.1` first.** If the heat balance the world publishes is not
the heat balance it obeys, that explains `SH.02`'s saturated null directly, and
you will have converted a fifth VOID into a finding.

**B2 — extend `decisions.py` to `SYSTEM.md`'s three classes, and make `class`
cost something to declare. RANK 2, carried from the 47th (B3) and 49th (B6b).**
Two parts, and the second is the one that matters:
  - (a) `CLASSES = ("means", "goal")` contradicts `SYSTEM.md`'s ENDS /
    ARCHITECTURE / CONDUCT, amended 2026-08-24 at the owner's insistence
    precisely because a binary split *"silently files every architectural bet on
    the goal side."* **Known positive: `d461e36`** — `D16` was written
    `class: conduct`, the tool refused it, and the *entry* was edited. Add
    `conduct` and `architecture`; map `architecture` onto the `MEANS-ESCALATED`
    path (rule 3 governs it) and `conduct` onto the goal path (not measurable).
    Do this **after** the 08-31 cohort fires, not before — twelve armed
    decisions on the enforcement path is not a thing to touch tonight.
  - (b) **`class` is self-declared and `audit()` takes the author's word.** The
    honest ratchet is the one `champions.py` already uses: resolve, don't trust.
    Add `arena:` to the `DECIDE:` block, resolve it against `BY_ID`, and raise
    `NAMED-ARENA-MISSING`. **Known positive: `D13` naming `SY.01`**, which occurs
    exactly once in this repo — inside the paragraph claiming it would settle the
    question. Second known positive: a `goal`-class entry whose own default says
    *"the remaining permitted arms go to a bakeoff"* is describing runnable arms;
    that shape should at minimum print a `CLASS-UNVERIFIED` note. Until one of
    these lands, `MEANS-ESCALATED` is a field, not a check.

**B3 — when `D1` fires tomorrow, write the bakeoff the same day.** RANK 3,
carried from the 49th (B5) and unchanged. `T2.01` frees **35 specs**, has not run
since 2026-08-12, and gates all of Tier 4 and most of Tier 5. This is also the
only thing that will stop W35's ~29 remaining GPU-hours from becoming the fourth
consecutive expired quota (59.31 h lost across W32–W34). `Budget.CPU_DAYS`, capped
at `D4`'s already-spent envelope — it is not itself a GPU fill, but it unblocks
the specs that are. Record the cost the default names: **arm D forecloses
`DP.02`**, because private control representations are the two-towers-in-one-
wrapper signature the connectedness directive forbids. A cost to record, not a
thumb on the scale.

**B4 — two honesty repairs in `T2.09`, neither of which touches a gate.** RANK 4.
  - (a) `44f24c4`'s claim that the seed-selection formula *"reads only the null
    and the rig instruments — never the claim arm's …"* is false as a summary:
    `t2_09_*.py:583-589` gates on `claim_static_reward_q1`, `claim_static_decay`
    and `exposure_frac_of_random`, all claim-arm measurements. **Live effect is
    zero** — I read `per_seed` and all three exclusions fired on `trap_dwell`,
    `static_decay` ≥ 1.424 on all seven seeds — so **do not move `DECAY_MIN`**;
    re-fitting it now would be the real violation. Fix the *sentence*, in the
    docstring and in a `LESSONS.md` line: an enumerated exclusion list and a
    summary clause are different promises, and the reader will believe the
    summary.
  - (b) Seed 1 recorded `trap_ratio` **953,594,661,617.28** — a near-zero
    denominator, not a spectacular trap. It clears `TRAP_RATIO_MIN` in the safe
    direction so the verdict stands. Cap it, or record `out_zone_reward`
    alongside it, so a ledger reader can tell "the trap was fed" from "the
    divisor vanished" without re-deriving it.

**B5 — carried, unchanged from the 49th: register `PL.02`, then `PL.00`, then
`LT.03`/`LT.04`. Seventh audit asking; 22 days.** RANK 5. `CHAMPIONS.md:166`
asserts *"`PL.02` decides it and is runnable today"* about a spec that has never
existed, and it is the sole falsifier of the **PLASTIC-ONLY decree**
(`GOAL.md:76`) — the decree whose constitutional half `D1` is asking the owner
about *tomorrow*. You proved this is one iteration of work when you did
`W.1`–`W.8` this morning. **Do not delete an arena reference to reduce a count** —
that converts `ARENA-MISSING` into `NO-ARENA` and makes the seat permanently
safe. For `ASR`, `Speaker ID` and `Language grounding`, the legitimate discharge
is one sentence in `CHAMPIONS.md` saying the seat is an END, not an architecture.

**B6 — one-line, carried from the 49th (B6a).** `T0.01` and `T0.10` are the only
`PASS` specs with no declared control. Add a sentence to each spec's `control`
field ("a harness liveness check — a control is undefined for it") so the absence
reads as a decision. I re-verified both are honest; this is bookkeeping.

## FOR THE OWNER

**1. Eleven pre-registered defaults fire tomorrow, 2026-08-31. If you want any of
them to go differently, today is the day.** They were armed by earlier audits
precisely so that your silence could not deadlock the project a second time, and
every one of them is reversible from the ledger's history. The one that matters:

> **`D1` — does `PLASTIC-ONLY` admit a frozen control trunk?** Open **21 days**,
> blocking **38 specs** including `T2.01` (which alone frees 35), all of Tier 4
> and most of Tier 5. **Its default changes nothing you decreed**: the
> `PLASTIC-ONLY` decree of 2026-08-09 stands verbatim and unnarrowed, the option
> that would have narrowed it is struck as unconstitutional, and the remaining
> arms go to a bakeoff the loop runs itself. **You only need to act if you want
> the decree narrowed** — that is the branch no experiment may take for you.

The measured cost of the delay, so it is on the record with a number:
**59.31 free Kaggle GPU-hours expired unspent** across W32–W34, because `D1`
blocks `T2.01`/`T2.02`, which is why the GPU queue reports `NOT FILLABLE` at
every cost class. W35 has ~29 h and six days left, and the same thing happens
again unless this clears.

**2. A governance instrument does not yet enforce what `SYSTEM.md` says it
does — and this is the second time in a week.** `SYSTEM.md` now honestly records
that two of the three safety clauses on defaults are enforced by nobody. I found
a second, smaller instance of the same shape: `decisions.py` implements the
**two**-class means/goal split that you personally rejected on 2026-08-24 (*"this
project won't work if you can't let them research and test stuff like that!!!
architectural stuff"*), so `SYSTEM.md` has three classes and the tool has two,
and on `d461e36` a decision entry was edited to fit the tool rather than the tool
extended to fit your ruling.

**Nothing is asked of you here** — it is filed as builder work (B2), and I
verified all 13 live entries are correctly classified on the merits, so no
architecture question is currently hiding on your desk. I am telling you because
the pattern matters more than either instance: **this project's governing
documents keep making capability claims about their own enforcement, and
`SYSTEM.md`'s first law says a capability is claimed only by a test that could
have failed.** Two of the three tools every audit opens with got their first
ledger certificates in the last 48 hours (`T0.28`, `T0.29`). That was the right
call and it is now paid. The remaining exposure is that the documents describing
the machine are not themselves under the ladder.

**3. Nothing is asked of you that the system could have decided itself.** I
checked. Zero `MEANS-ESCALATED`, and every open entry turns on what is
*permitted* rather than on what *works*.
