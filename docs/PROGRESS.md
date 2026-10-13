> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: newest row in docs/PROGRESS_LOG.md is 2026-10-11 (2d old; the schedule allows 1d)
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-10-13T12:37:03+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

**Review FULL, 2026-10-11 (Sunday).** Written LAST, as the receipt for ten
commits that already exist. Every act below was committed as it was made, per
the 74th audit's scar: a max-turns death costs the PAGE, not the WORK.

---

## Part 1 — the state of progress, in numbers

| | reading |
|---|---|
| demonstrated | **107 / 257 = 41.6%** — unmoved for a third week |
| pass rate of *registered* | **107/160 = 66.9%** (`VOID` 17, `FAIL` 35, `BLOCKED` 1) |
| net new PASS, this week | **0** (107 → 107) |
| net new PASS, last week | **−1** (108 → 107) |
| ledger rows | 156 → 157 → **160** over two weeks |
| rework (attempts > 1) | 127/160 = **79.4%** |
| builder iterations, 7 d | **ZERO** since 2026-10-08T12:32 |
| builder dark streak | **66 consecutive slots / 66.3 h**, **0 failed** |
| queue | **OVERDUE 5 → 0**, `review-queue` EXIT 0; 141 routed |
| steering page | **127,327 B** — 3,745 under the launch cliff |

**Velocity, and it is the headline.** Two consecutive weeks of **zero net new
PASSes**, and the earlier of the two was a net **loss** of one. Over the same
fortnight the ledger grew by four rows. The project attempted more and certified
less.

**Goodhart check — rate falling while count rises, which is the exact signature
the DIRECTION_AUDIT warned about.** Pass rate of registered specs: **69.2%
(09-27) → 68.2% (10-04) → 66.9% (today)**, on a ledger growing 156 → 160.
Demonstrated-against-registry sits flat at 41.6% only because the registry did
not move either. This is information and the information is unambiguous: **the
denominator is winning again, and this time without the excuse of productive
weeks behind it.** Three weeks ago that reading could be called the cost of
growth. It cannot now — nothing grew.

**The frontier, recomputed not quoted.** 96 of 257 specs unreachable.
`T1.08` = FAIL **frees 3, blocks 45** — still the largest transitive mass on the
board by a wide margin, and still out of the builder's hands under a standing
prohibition this desk re-affirmed today; the open question is an ESTIMATOR
question and it is mine. `NE.01` = FAIL **frees 7** (blocks 8), impl unchanged
**47 d** — the highest-`frees` blocker, and it sits inside one of the seven
TERMINAL rows declined on 10-09, which is `D43`'s cost paragraph still
concrete. `UB.10` = VOID frees 4, impl unchanged 28 d.

**Is the builder working on the frontier? The builder is not working at all, and
it is not the builder's fault.** `0 failed slots` across all 66 — the loop is
healthy and the pacer is holding budget deliberately. `week:all models` reads
**83%** against a pace line of **61%** at 54% of the week (`week:Fable` 30%, not
the gate). Attribution, taken after this sitting's own ledger mark landed —
which is the only condition the 144th audit called honest — reads **69 shared
points: builder 8 (11%), desks 6 (8%), NOT THIS PROJECT 55 (79%)**. See
`FOR THE OWNER 1`.

**Effort-vs-goal.** Of this week's committed work, essentially none was spent
advancing `GOAL.md`'s current stage, because the organ that advances it did not
run. Ten commits today are desk acts: five queue dispositions, one spec
strengthening, one steering reconciliation, one routing correction, one
completeness-audit increment, one page. That is governance, honestly labelled.

### The honest paragraph — no numbers

Busier, and for the first time in a while I am not sure that is the right word
either, because the thing that was busy was the desks and not the creature. The
instruments are clean, the backlog is empty of violations, the rulings are
careful, and Jack is exactly where he was a fortnight ago. What this week
actually bought was understanding of our own measuring equipment: the single
most important step toward Jack was discovering that a venue too easy to resolve
its own claim produces a confident PASS just as readily as a red one — found in
a passing spec, on the fourth rig to show the same pathology, which means the
disease we have been treating as a property of the rigs that failed is a
property of the ladder. That is a real step and it is a step about us, not him.
The most concerning drift is that it had to be found by a desk reading old
certificates on a Sunday, while the organ that builds him sat dark for most of
three days behind a meter that another tenant is filling. A project whose
reviewers are its only moving parts is not learning about its subject; it is
learning about its own paperwork. I would rather report one new certified fact
about a creature than ten well-reasoned pages about why we have none.

---

## Part 2 — the test re-examination

**Scope, declared honestly: TWO specs examined against the mandated 8–12, and
one strengthening implemented.** The sitting's first act consumed most of its
budget: the OVERDUE class stood at **five rows**, the largest this desk has
faced, and every one needed source verification rather than a stamp. I am not
presenting a reduced sample as a full one. The two examined were chosen
oldest-passed-and-least-reconsidered first, and both returned findings.

### STRENGTHENED — implemented, re-verified, committed

**`DP.04` — GUARD (a2), `need_resolution_band_mass`.** The pre-check measured
its venue twice in fields its own stop-rule never read. Guard (a) asks "does one
VALUE hold too much mass?" — and continuous arithmetic breaks exact ties without
moving mass, so a statistic can satisfy it while remaining as saturated as the
one it replaced. Measured on the spent seed-94 artifact: 616 distinct values and
`max_single_mass` 0.0833 (a comfortable PASS), while the **median sits at 0.9990
of the maximum at BOTH caps** and **74.28% of lives fall inside one
`NEED_MIN_GAIN`-wide band of the top** (71.55% at cap 400) — against the
**76.7%** single-value mass that condemned the retired statistic. Two lives
inside one bar-width are indistinguishable *by the claim's own ruler* whether or
not they are equal, so that is the mass the new guard measures.
**Why it is a strengthening and not a tuning:** it asserts against the
**already-declared** `NEED_MASS_CAP 0.767` — no threshold invented, none
lowered, a refusal condition ADDED where there was none. It reads 0.7428 today
and therefore **PASSES, by 2.4 points**, reported rather than tuned, because a
guard that fired the day it was written would be a guard aimed at a number.
Verified as a pure function against the spent artifact; **`DP.04` was not run**
(it is `_PILOT_BLOCKED`, and running a spec writes the ledger). Bill: zero —
DP.04 holds no PASS.

### RE-EXAMINED — `ME.8` (PASS, attempt 4, 2026-09-06)

**The objection I went looking for is not there, and that is the finding's
first half.** `holdout_acc` **1.0** against `MIN_HOLDOUT` 0.95 and `resume_acc`
**1.0** against `MIN_RESUME` 0.90 both sit flat on the ceiling — the shape today's
`DP.04` ruling convicts. It does not convict `ME.8`: the DECIDING metric is
neither of those (source comments `MIN_GAP = 0.50` as *"`resume_vs_zeroed`, the
registered metric"*), that metric reads **0.8958 against a 0.50 bar** with real
room beneath it, and **its control is genuine** — `zeroed_acc` reads **0.1042
against a chance level of 0.125**, i.e. the null is AT chance, and `MAX_ZEROED`
0.45 is a ~4σ band around it rather than a clearable number. `ME.8` is a better
spec than its one-line summary suggests and its gate is not weak.

**The weakness is real and it is elsewhere: the task is too easy for the
instrument to resolve the capability.** Solved perfectly means the memory demand
was never approached, so `resume_vs_zeroed` measures **WM persistence** and is
structurally incapable of measuring **capacity or manipulation** — it cannot
tell a one-item memory from a seven-item one, because the task never asks for a
second item. **Fourth venue with this pathology (`DP.04`, `SH.02`, `W0`,
`ME.8`), and `ME.8` is the first that is PASSING.** That generalisation is worth
more than the spec: only the red ones have been getting attention.

**Not weakened, not rewritten, nothing re-run.** The proposal is in
`FOR THE BUILDER 4` because it is a redesign, not an added assertion.

---

## FOR THE BUILDER — ordered, cheapest and most perishable first

All three ordered units are **CPU**. None is a buyer for W41's expiring GPU
hours. All are named in `scripts/ladder_prompt.md`'s `1^22` ITEM 2 addendum;
note you have had no `rc=0` since 2026-10-08T12:32, so you have read neither
`1^22` nor its two addenda.

1. **The GPU ref-pin — the cheapest world-touching item on the board.**
   (`gpu-receipt-head-is-push-time-not-kernel-time`, **DUE 10-18**.)
   `repo_preamble`/`build_job` default `ref` to the literal `"main"` and every
   call site takes that default, so the kernel checks out a BRANCH and
   `kernel_sha256` cannot vary with the head it is meant to pin. Do **(i) then
   (ii)**: resolve `HEAD` to a sha and pass THAT as `ref`; then have the job
   print its cloned HEAD and the harvester refuse a mismatch. (ii) is still
   required after (i) because `build_job(verify=False)` bypasses (i) entirely.
   Bill priced: **1 certificate, `T0.12`, `cpu<1min`** — pay it in the same slot.
   Caution: after (i), `kernel_sha256` stops being stable across dispatches by
   design; grep for readers that treat kernel-hash equality as "same kernel".
2. **The two `WAITS-ON` readings.**
   (`prose-sequencing-inverts-when-the-blocker-moves`, **DUE 10-19**.) The corpus
   measurement that row owed **is already done at this desk — do not redo it.**
   Build **class (2) FIRST**: `WAITS-ON-A-CLOSED-WINDOW` (3 live today), then
   `DATE-INVERSION` (1 live today). **Both are READINGS** — printed, no exit
   code, no ageing exemption, no auto-re-date. `WAITS-ON:` must keep buying
   nothing.
3. **`DP.04` option (ii), the gridworld difficulty edit.**
   (`dp04-graded-measure-passed-its-guard-and-did-not-cure-the-venue`, **DUE
   10-22**.) The acceptance test is the **pre-check's resolution reading on a
   spent seed** — band mass well under 0.7428, median/max under 0.9990 — and it
   is **NOT `DP.04`'s claim verdict**, because a world edit accepted on its claim
   verdict is a world tuned to pass a test. Run the pre-check BEFORE any claim
   fold. Bill: **2 certificates, `LC.00` + `DP.00`**; `DP.00`'s re-run is the
   informative one, being a PASS whose venue this makes harder.
4. **PROPOSAL — a load axis on `ME.8`.** A declared `WM_LOAD` parameter (items
   held / interposed distractors), with the registered `resume_vs_zeroed` gap
   required to persist at a load of **at least 2 items** and the bar re-derived
   from chance AT THAT LOAD. **Strictly harder:** `MIN_GAP` 0.50 stays,
   `MAX_ZEROED` 0.45 stays, every conjunct stays binding, a load axis is ADDED
   where there is none, and the variant can FAIL where the current spec passes.
   Falsifier pre-stated: a gap surviving trivially at load 2 retires the load
   axis rather than vindicating it. Bill: **zero certificates.**
5. **PROPOSAL — sweep the PASSING specs for unresolvable venues.** `ME.8` makes
   this a class, not an anecdote. For each PASS, ask whether its claim statistic
   has room between bar and structural ceiling, and whether its task is hard
   enough to approach the capability it names. **This must NOT be done by
   counting metrics at 1.0** — many are boolean conjunct flags, and a count that
   conflated them would be confidently wrong, which is worse than absent. The
   sweep needs a declared rule for distinguishing a claim statistic from a
   conjunct flag before it produces any number. That rule is the deliverable.

---

## FOR THE OWNER

**1. The builder has been dark 66 hours and 79% of the meter rationing it
belongs to another tenant. See `D40`** — already on your desk, `class conduct`,
`decide_by` **2026-10-10** now PASSED with default **(v)** standing. I have
appended an EVIDENCE ADDENDUM rather than opening a second entry, because a
duplicate on the same subject is what this file's discipline forbids.
**What is new and why I trust it:** the 144th audit proved these shares are a
function of *when* they are read — no ledger mark is written while nothing runs,
so during a blackout the instrument under-reports the neighbouring tenant and
over-reports the builder. **I nearly published the blackout reading as my
headline and did not, because that correction exists.** Read after this
sitting's own mark landed — the condition the 144th audit called honest —
**NOT THIS PROJECT has gone 50% → 79% while the builder went 42% → 11% on an
unchanged absolute spend of 8 points.** Sixteen points appeared between the
06:07 `PACING:` line and my reading and **every one was external.**
**This is not the 10-07 story and recycling it would have been wrong:** that
week both halves were true (the builder front-loaded, then paced itself out).
**This week the builder front-loaded nothing** — 8 points across seven days, 66
hours dark — so the "self-inflicted" half has no purchase. The known bias
direction was noted as arguing FOR default (v); corrected and re-read, it argues
against it. My recommendation is unchanged from what `D40` already quotes and I
am not restating it as a new ask.
**`D30`'s required pairing, counted here not inherited:** dark **66 slots /
66.3 h**, `0 failed`; **`2026-W41` opened today with 30.0 free Kaggle GPU-hours,
0.000 used, expiring Saturday 2026-10-17.** Unlike W37–W40 I will **not** call
those the loss: the standing `T1.08` prohibition means they have no designed
buyer, and buying a sixth 3-seed reading of an instrument we have shown
unreliable would be worse than letting them expire. **The loss is the CPU
ladder** — three CPU units ordered today that cannot start while the builder is
dark.

**2. NO-DECISION: liveness and docket report, nothing here to rule on.** All
four organs within cadence: builder hourly (alive, PACING — see item 1),
overseer 6-hourly (last 2026-10-11 06:37), field watch Monday (last 2026-10-05,
week 11 due 10-12), this desk Sunday (now). **And one piece of good news against
a standing concern:** the previous page recorded that 10 of the last 11 DAILY
Reviews died `rc=124` and that four of four Sunday FULLs had died at max turns.
**This FULL completed its checklist and committed ten acts as it made them**,
which is the 74th-audit discipline working as intended rather than a change of
luck. `champions --check` EXIT 0 at every floor (32 seats; 2/3 unfalsifiable,
4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt — all at baseline,
none grown). `review-queue` EXIT 0, **OVERDUE 5 → 0**. `decisions --check`
EXIT 0. Open owner entries unchanged and on their own clocks: `D41`, `D42` (both
10-18), `D43` (10-16), `D44` (10-24).

**3. NO-DECISION: a correction to my own routing, reported because an instrument
caught it and the next desk should not repeat it.** I drafted the steering-page
launch cliff as a new owner decision, `D45`. `run decisions --check` returned
`[UNDECLARED] D45` and broke the `UNDECLARED` ratchet 0 → 1. Reading the classes
properly, `class conduct` means *"desk-executable, not the owner's — execute it,
report it, do not ask"*, and `scripts/ladder_prompt.md` is operational with Part
2.5 duty 1 giving this desk direct edit rights. **I was asking you for
permission to tidy a file I am required to maintain.** `D45` is withdrawn, the
ratchet is back to 0, and the work is routed where it belongs as a desk row
(`steering-page-is-two-days-from-the-launch-cliff-and-31kb-is-provenance`, DUE
**2026-10-18**, with a priority clause putting it ahead of any other act here if
a slot opens sooner). **The fact itself is worth your awareness even though the
act is mine:** the page is **3,745 bytes** from the size at which the builder
stops launching, and past that cliff it does not read a degraded prompt — it
fails with an ordinary-looking `rc=126`, indistinguishable from the launcher
faults we have already spent weeks on, and it would land inside a blackout
currently blamed on the pacer. ~31 KB of the file is text explicitly marked
never to be executed; archiving it takes the page to ~97 KB. I did not do it
this morning because live orders cross-reference `2'` and `3'` *inside* the
superseded block, so the move needs an inline rather than a cut, and half-doing
it at the end of a sitting is a failure this desk has been convicted of before.

**4. NO-DECISION: anatomy and completeness audits, reported.** **Anatomy:** 32
seats against `GOAL.md`; no capability the goal requires is seatless, so **no
seat was added** and none is proposed for merge or removal (both propose-only).
**Completeness, audited against the external reference and not our own
documents:** the sensory half is **closed and stays closed** — `run senses` at
HEAD reports **10/10 of the inventory registered, 2/10 load-bearing (sight,
voice)** — so three of the four zeros the owner found in one evening on
2026-08-09 (voice, smell, taste) are gone. **Body schema remains the one
owner-listed zero that has never moved:** 0 own specs, seated 2026-09-06, arena
`UB.14` VOID-FORECLOSED. **The cognitive half is still the hole, and this
sitting closed one ninth of it:** working memory is converted from a named gap
into a specific harder spec design (`FOR THE BUILDER 4`). Eight capabilities
remain — attention (0 specs), emotion (2 specs against 1,149 lines of
`EmotionalState.py`, and **still not a declared `COVERS:` domain at all**),
imagination, self-model, theory of mind, teaching, tool use, language
production. **They are NOT converted to refusals by silence**; the row is
re-armed `DUE 2026-11-01` with the next sitting pre-committed to `emotion`, the
cheapest real repair and the oldest unmoved item from the 08-09 scar.

---

## Constitution coherence

No new internal contradiction found in `GOAL.md` or `SYSTEM.md` this sitting,
and none was edited — they remain the owner's. The live tension already on your
desk as `D44` (shipped character sheet vs `GOAL.md:172`'s refusal to carve his
character) is unchanged and on its own clock to 10-24.
