# OVERSIGHT — 57th audit, 2026-09-01 06:55 UTC (HEAD `33ea045`, tree dirty: 2 GPU bookkeeping files)

## VERDICT: ON TRACK — the ladder is honest and every threshold that moved this week moved the hard way, but the numerator did not move at all in 24 hours and the newest guard in the repo is trippable by a line-wrap

Sections 1, 2, 3, 6 and 7 have **no violations**, checked mechanically rather
than asserted. Saying that plainly is the point of this audit: it is what makes
FINDING 1 worth acting on instead of reading as noise.

The 56th audit's FINDING 1 (`review_liveness` blind to `**FULL**`) is
**CLOSED** — repaired at `ea0caab`, verified live this audit: `review_liveness`
returns rc=0 and prints no banner.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2 (RED)** | **0 commitments with NO spec, 0 CLAIM-DEAD.** The red term is `new_empty_class`, not `uncovered` — see below |
| `decisions --check` | 0 | 0/10 undeclared, 3 armed (D15/D16 due 09-05, D17 due 09-07). No `MEANS-ESCALATED`, no `OVERDUE` |
| `champions --check` | 0 | 27 seats, ratchet ok. 1 phantom arena, 3 unfalsifiable, 1 arena-unreachable — all baselined |
| `run review-queue` | 0 | 10 OPEN / 2 HELD / 2 ACTED of 14; oldest live 8 d; consumer ran 1 d ago; **0 violations** |

**`coverage` rc=2 is NOT the highest-priority finding this time, and the
distinction matters.** The instruction ranks an uncovered commitment above
everything; there are none. Every one of GOAL.md's commitments has a declared
spec behind it. The red is `new_empty_class` — `cpu<48h`, `gpu<20min` and
`gpu<2h` have nothing dispatchable — which is a queue-depth fact, not a claim
without evidence. It was declared pre-existing at `f34251a` and it is still
true. Four classes have **no path in at all**: nothing to implement, nothing to
pilot. Dispatchable today: **3, of which 1 is a VOID — 2 fresh dispatches.**

**Nothing to arm this audit.** `decisions` reports 0/10 UNDECLARED, so the
standing "arm at least one per audit" duty has nothing to bite on. The ratchet
did not grow.

---

## FINDING 1 (top rank) — a sentence about LC.03 declares `LC.07` VOID-FORECLOSED, and the refusal that is supposed to be loud is printed by nobody

`experiments/tests/lc_07_scale_transfer.py`, docstring lines 12–13:

```
SINGLE-ARM ON PURPOSE. This is not a re-run of the five-arm screen (LC.03 is
VOID-FORECLOSED: no v3, no envelope growth, no re-roll) and it does not route
```

A sentence *about a different spec* wrapped so that `VOID-FORECLOSED:` landed
at column 0. `protocol._margin_declaration` matches on `raw.startswith(keyword)`
and fires. Verified live:

```
protocol.void_foreclosed_refusal('LC.07')
-> "declaration REFUSED — missing required block(s): `FORECLOSURE ARITHMETIC:`,
    `BLAST RADIUS:` (54th audit B3 ...)"
```

It is the **only** refused declaration in the repo — I scanned all 213 specs for
margin-anchored `VOID-FORECLOSED:` / `FORECLOSURE ARITHMETIC:` / `BLAST RADIUS:`
blocks. The other five (BA.03, LC.03, ME.11.E, ME.11.F, T3.06) each carry all
three blocks and are legitimate.

**And nothing prints it.** `void_foreclosed_refusal`'s own docstring states its
reason for existing: *"This exists so the refusal is LOUD ... the readers print
this message beside the spec so the next iteration repairs the declaration."*
`coverage.py` collects refusals only inside `if status == "VOID"`
(`coverage.py:605-616`) — and `LC.07` never reaches that line, because
`gates_frozen(...) is False → continue` at `coverage.py:570` returns it as
GATES-PROVISIONAL first. Confirmed by grepping the full `coverage` output: the
string `LC.07` appears once, in the gate-provisional list, with no refusal
beside it. **A guard's loudness clause that is unreachable for every spec that
has not yet run is a guard nobody will notice is gone** — which is the exact
sentence `coverage.py`'s own `_exit_code_fixture` was written to prevent, one
organ over.

**Damage today: none to the ledger.** `void_foreclosed()` is gated on
`status == "VOID"` too, so no spec can currently be excluded from the queue by a
stray line. This is a latent defect, and I am reporting it as one.

**Damage tomorrow, and why it is top rank anyway.** `LC.07` is the learning-core
seat's *only* live arena member — the single registered thing that keeps
`wm-latent`'s BY VERDICT seat contestable (`champions --check` reads the seat
`ok` because `LC.07` resolves). Its own pre-registered decision tree includes a
VOID branch. On the day it runs and VOIDs, the reader that should say *"this arm
is repairable"* will instead print a refusal message about a declaration nobody
wrote, aimed at the spec that guards the project's most consequential seat.

**And this is the third appearance of one class of defect.** `champions.py` paid
for a regex over prose on `901f7fc`. `review_queue.py` was built reading
DECLARED fields only, and says so in its header, *because* of that. The newest
organ — three days old — went back to matching a keyword at the margin of free
text, and has now been tripped by free text in the wild.

---

## FINDING 2 — 24 slots, 24 rc=0, and the demonstrated count did not move

| window | registered | demonstrated |
|---|---|---|
| 2026-08-31 06:07 | 201 | 93 |
| 2026-09-01 06:26 | 213 | 93 |

24 iteration starts, 24 ends, **all rc=0** — the loop is alive, on cadence, and
not failing. It peaked at 94 and gave one back at 23:14 when `T2.10` was
re-bought **FAIL** under a strictly-harder paraphrase conjunct (`b4805ac`). That
loss is the machine working correctly and I want it recorded as a credit, not a
debit: a seat's founding certificate was re-run against a fixture that did not
exist when it was written, and it came back red and was kept red.

The concern is the ratio. **The ladder grew 12 and climbed 0 in one day.**
44.9% → 43.7%. The day's work was real and lawful — 11 armed defaults fired,
`D1.0` registered and dispatched, `LC.07` implemented, `UB.14` inherited,
probed and dispatched — but not one of it was a capability demonstrated. Two
days at this rate and "the machine is better than I found it" stops being a
defence.

Cause, and it is not the builder: **the venue.** See FINDING 3.

---

## FINDING 3 — 9 specs are welded shut, 11 more sit behind them, and 7 of the 9 weld for the same reason

Computed live from `depends_on` against the ledger, not from prose:

- **VOID-FORECLOSED (5):** `BA.03`, `LC.03`, `ME.11.E`, `ME.11.F`, `T3.06`
- **PILOT-BLOCKED (4):** `DP.04`, `SH.02`, `SM.03`, `T2.11`
- **Transitively unreachable, non-PASS (11):** `DP.01`, `DP.02`, `DP.03`,
  `LC.04`, `LC.05`, `LC.06`, `ME.6`, `OP.01`, `PS.04`, **`T5.06`** and
  **`T5.08`**

The last two are *"Unprompted exploration is real"* and *"Open-endedness"* —
Tier 5, the thesis itself.

Seven of the nine roots foreclose on the same fact wearing different clothes:
`BA.03` (blind twin holds 98.9% of the horizon), `SH.02` (every non-learning arm
holds the roof at exactly 1.0000), `DP.04` (3072 lives, 21 distinct lifespans),
`SM.03` (held-out split saturated, `vis_open` 0.1167 against 0.60), `LC.03` (one
learner in five), `LT.01` C2 (body cannot reach the platform, 0.084 m against a
0.6 m bar), and now `UB.14` (the eye reads root-xy at 0.159 held-out against a
0.5 gate — the sensory mirror of `LT.01`'s motor finding). **W0 does not reward,
or this body cannot perform, the thing being measured.**

This is tracked — `w0-too-shallow` is OPEN with `DUE: 2026-09-06`, and
`reparenting-the-welded-fifteen` is OPEN with the same date. I verified the
consumer will actually be there: `scripts/review.sh:52` sets `MODE=FULL`,
`TMOUT=40m`, `MAXTURNS=120` when `date +%u = 7`; 2026-09-06 is a Sunday; cron
runs `review.sh` at 06:37 daily. The four Sundays of `Reached max turns (60)`
were a budget defect, fixed on 08-31, and the first FULL run in the project's
history completed as a rehearsal that night. **The 09-06 promises land on a run
that can now finish.**

---

## Section 1 — Integrity of the ledger: NO FINDINGS

All **93 PASS rows** checked mechanically:

- **0** with a `commit` that no longer resolves in git (`git cat-file -e`, all 93)
- **0** without a resolvable implementation module (`run._module_for`, all 93)
- **0** whose spec declares no `control`

Two PASS rows carry an empty `control_metrics` block — `T0.01` and `T0.10`.
Both declare, in the spec itself, *"NONE, BY DECISION (52nd audit B5)"* with the
reasoning attached (an import either raises or it does not; a sabotaged upload
fails on the service's side). That is an adjudicated absence, not a control that
was never run.

One stale claim, flagged by the tool itself and not by me: `T2.02` (recorded
VOID; test file content differs from the blob at `ran_at`). 17 entries predate
`impl_sha`, 16 verified byte-identical by git, 0 unanswerable.

## Section 2 — Thresholds and controls over 7 days: NO FINDINGS, one hit examined and cleared

Nine numeric constants moved in `registry*.py` / `tests/` in the last 7 days.
**Eight moved in the tightening direction:** `N_DECISIONS` 3200→4800,
`N_EVAL` 48→120, `COORD_MIN` 0.55→0.7, `COORD_MARGIN` 0.20→0.35,
`LIVES_PER_ARM` 4→16 then 16→48, `STEPS` 300→500, `N_PROPERTIES` 11→12.

**One moved down: `DECAY_MIN` 1.5 → 1.25 (T2.09, `44f24c4`).** Examined in full,
and it is not a loosening:

1. It was a `# PILOT —` placeholder in a spec whose `run()` **refused** until
   that commit. Never a registered bar.
2. It is a seed-**informativeness** bar, not a claim bar.
3. Its direction of effect is to admit *more* seeds into a
   worst-informative-seed fold — which makes the test harder, not easier. Seed
   90, whose `margin_vs_null` is −0.0404, is exactly the seed a lower bar lets in.
4. The commit message declares it under its own heading — *"ONE BAR MOVED,
   DOWNWARD, IN THE OPEN"* — and justifies 1.25 from the mechanism (a dead
   signal decays by exactly 1.0), explicitly not from the observed minimum
   (1.472).

No control was deleted or weakened, no `_check` gained an `or`, no seed count
was reduced (`T2.09` went 3→7), no assertion was removed.

## Section 3 — Drift: NO FINDINGS forward; the backward answer is the same as always

Everything the builder touched in 24 h traces to a GOAL.md sentence:
11 armed defaults (SYSTEM.md rule 3's deadlock clause), `D1.0` (*"one
interconnected brain"* — the control-path question behind 38 specs), `LC.07`
(the owner's own ~10× scale-transfer guard), `UB.14` (*"all senses in unison"* —
vision+proprio predicting touch off the real body). No drift.

The converse, which is the harder question: **12 commitments have live claim
specs and nothing passing** — touch, tool use, smell, balance, proprioception,
shelter, death & retry, thermal, plasticity, sleep, hunger/thirst, fast/slow.
And the two GOAL.md names most likely to be quietly neglected are exactly where
coverage says they are:

    curiosity           12 specs   2 pass
    one brain / unison  22 specs   1 pass

22 specs on the goal's first sentence and one of them passes. That is not a new
finding, but it should not stop being reported until it changes.

## Section 4 — Builder liveness: alive, productive, not idling

24/24 slots, all rc=0. Two `LEFTOVER` undeclared processes flagged by the loop's
own watchdog (19:11 on 08-31, 04:27 today) — the guard fired, printed, and did
not kill, which is what it is written to do. Right now: **two declared live
runs, both verified by `ps`, not by claim** — `D1.0` watcher pid 4187660
(4 h 26 m elapsed, matching the 02:12 dispatch) and `UB.14` pid 49101 (13 m).
No orphans.

## Section 5 — Compute honesty: one bookkeeping flag

W35 Kaggle: **5.58 h of 30**, resetting Sunday 09-06. `D1.0` kernel 1
(`jack-ladder-1788228751`) charged 4.076 h and returned ok at 06:17; kernel 2
dispatched at 06:17:14 at 8.625 h est. Projected total ≈21.7 h — it lands with
days to spare. Prior weeks for contrast: W33 7.6 h, W34 1.6 h against 30 — the
expired-quota scar, now finally being repaid by the largest spend in three weeks.

I checked the week key rather than assuming it: `gpu.py:396` uses
`strftime("%Y-W%U")` — Sunday-start — deliberately, with the comment recording
that ISO `%G-W%V` used to charge Sunday's runs to the previous week. 2026-W35
begins Sunday 2026-08-30, which is Kaggle's actual reset boundary. Correct.

**The flag:** `experiments/gpu_budget.json` and `experiments/gpu_submissions.jsonl`
have been **uncommitted since 06:17**, holding 4.076 GPU-hours of completed
spend and two submission rows. The watcher writes them live, so this is not
concealment — but a clone of `HEAD` understates this week's Kaggle spend by
4.08 h, and it is the same shape as the 08-31 Review's *"`ledger.json`
uncommitted since 06:10"* flag. Committing GPU accounting is the builder's, not
mine; I left both files alone.

## Sections 6 & 7 — Stuck decisions and bakeoff hygiene: NO VIOLATIONS, one disclosed risk worth the owner's eye

Nothing is blocked on the owner that a measurement could settle — 0
`MEANS-ESCALATED`. Nothing is overdue. Three decisions are armed with legal
defaults and live dates. Eleven defaults fired on 2026-09-01; I spot-checked the
one with the most reach.

**D10 seats `wm-latent` BY VERDICT on a screen its own pre-registered fork
declared VOID** — *"fewer than two learners (1 cleared)"*. That is, on its face,
section 7's named disease: a VOID treated as a verdict.

**I judge it not a violation, and here is why the disclosure is what saves it.**
The seat's cell says so itself, in bold, in the strongest available words:
*"BY VERDICT, with the single-arm caveat on its face: the verdict is a
one-learner screen, not a won bakeoff"*, *"Seated ≠ adopted"*, and adoption is
explicitly gated behind `LC.07` PASS **and** the standing unison gates. The seat
carries a registered, existing challenger, so `champions --check` reads it
contestable rather than welded. A VOID converted into a *seating* while adoption
stays gated and the falsifier stays registered is a bookmark, not a claim.

**The residual risk, stated so it is on the record:** the seat is contestable on
paper and not yet in fact. `LC.07`'s gates are PROVISIONAL, its pilot is queued
behind `D1.0`'s ~20-hour GPU lock, and — per FINDING 1 — its docstring currently
trips the foreclosure parser. Three ways for the only challenger to the
project's central architectural seat to stay un-run. That is worth one sentence
of the owner's attention, not an intervention.

---

## Section 8 — The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday? No.
We are closer to knowing exactly why we are not, and that is worth more than
yesterday's green tick would have been.**

Zero capabilities were demonstrated in 24 hours. In the same 24 hours the
project measured, on the record, that its world and its body cannot support
seven of its own claims: the blind twin holds 98.9% of the balance horizon, the
frozen null holds the shelter roof at exactly 1.0, 3072 lives produce 21
distinct lifespans, the body reaches 0.084 m of a 0.6 m platform, and the eye
places the body at 0.159 against a 0.5 gate. Every one of those is an honest
red bought with real compute, and each one was declared rather than re-rolled.

That is the machine doing the harder half of its job. What it cannot do is spend
another week doing only that. Eleven specs — including *"Unprompted exploration
is real"* and *"Open-endedness"* — are parked behind a world design that one
document promises for 2026-09-06. The instrument that owes it will be there and
can now finish. **The single number that decides whether this audit reads as
"healthy pause" or "stall" is whether 09-06 produces a W0/W1 design, and I would
rather name that now than discover it next Sunday.**

The ledger is trustworthy. Nothing was loosened. The falsification machinery
caught a seat's own founding certificate and turned it red. Two of the three
things this project could quietly get wrong — a weakened threshold, a claim
without a control — did not happen this week, and I checked rather than assumed.
The third one, a guard that stops guarding, is FINDING 1.

---

## FOR THE BUILDER

**B1 (top) — repair the `LC.07` foreclosure false-positive, in three parts.**

  (a) **Reflow the sentence.** `experiments/tests/lc_07_scale_transfer.py`
      docstring line 13 must not begin with `VOID-FORECLOSED:`. This is a
      doc-only change — `run amend LC.07 --doc-only` /`prose_only_delta`
      re-stamps, and `LC.07` has no ledger row to invalidate anyway. One line.

  (b) **Make the refusal reachable.** `coverage.py` collects
      `void_foreclosed_refusal` only inside `if status == "VOID"`, after
      `gates_frozen(...) is False → continue` at line 570 has already dropped
      every un-run spec. Collect the refusal for **every** spec regardless of
      status and print it, because a bogus declaration on a spec that has not
      run is exactly the case that has now occurred and is exactly the case
      the current code cannot see. Add a fixture case for it: the existing
      battery covers Q.17 (a VOID spec with an incomplete declaration) and has
      no non-VOID case at all.

  (c) **The durable fix — stop matching a keyword at the margin of free text.**
      This defect has now cost three organs: `champions.py` on `901f7fc`,
      `review_queue.py` was designed around it, and
      `protocol._margin_declaration` walked back into it. The cheapest real
      repair is to reject a candidate whose *preceding* line leaves an unclosed
      `(` — which is precisely how LC.07's fired — and to require the
      declaration to be preceded by a blank line, the way every genuine one in
      the repo already is. State whichever rule you take in
      `_margin_declaration`'s docstring so the next author cannot re-derive the
      loose one.

**B2 — commit the GPU accounting.** `experiments/gpu_budget.json` and
`experiments/gpu_submissions.jsonl` have held 4.076 completed GPU-hours in the
working tree since 06:17. Commit them on the next iteration; `HEAD` should not
understate a week's spend. If there is a reason the watcher's writes are
deliberately left uncommitted until the run merges, say so in the journal — it
is currently indistinguishable from the 08-31 `ledger.json` flag.

**B3 — correct the `T2.21` citation on the Control-architecture seat.**
`champions --check` has one phantom arena that registering cannot fix: `T2.21`
was decided against on 2026-08-13 (`a3b12f6`, choice (b)). The repair is to
correct `docs/CHAMPIONS.md`'s arena cell to the live successor `D1.0`, **not**
to delete the reference — deleting converts ARENA-MISSING into NO-ARENA and
makes the seat permanently safe. The seat's `SEAT:` line at `CHAMPIONS.md:275`
carries the same stale id and must move with it.

**B4 — when `UB.14` harvests, the VOID-FORECLOSED declaration owes its two
companion blocks.** `FORECLOSURE ARITHMETIC:` and `BLAST RADIUS:` are required
or the declaration is refused — and per B1(b) that refusal is currently printed
nowhere, so an unpriced weld would be invisible until somebody re-ran it.
`UB.14`'s blast radius is not "none": check `UB.5`, `UB.11` and `UB.16` before
writing the block.

## FOR THE OWNER

**1. Nothing is waiting on you that a measurement could answer.** Three armed
decisions (D15, D16 due 09-05; D17 due 09-07) are on your desk with defaults
that fire if you say nothing, and every one of those defaults picks among
already-permitted actions. You can ignore all three safely; that is what arming
them was for.

**2. The project spent 24 hours and demonstrated nothing, and I do not think
that is a failure — yet.** The ladder grew 12 specs and climbed 0. What it did
instead was measure, seven separate ways, that the practice world `W0` and the
current body cannot support the claims being asked of them. Eleven further
specs sit behind those seven, including two Tier-5 claims that are the thesis:
*"Unprompted exploration is real"* and *"Open-endedness"*.

**3. The one date that matters is Sunday 2026-09-06.** The W0/W1 design is owed
by the Review's FULL run that morning. I verified it will actually fire and now
has the budget to finish — four earlier FULL runs died on a turn limit that was
fixed on 08-31. If 09-06 produces a design, this audit was a healthy pause. If
it does not, the same eleven specs will still be welded shut next week and the
verdict changes.

**4. One thing to keep half an eye on.** `wm-latent` now holds the learning-core
seat, seated from a screen that returned exactly one learner — labelled as such,
in bold, on its own cell, with adoption still gated behind a 10× re-test
(`LC.07`) and the unison gates. That disclosure is honest and I am not
contesting it. What I want on the record is that `LC.07` — the only registered
thing that could unseat it — is gate-provisional, is queued behind a 20-hour GPU
lock, and until B1(a) lands has a docstring that trips the repo's own
foreclosure parser. Three independent ways for the challenger to stay un-run is
more than one seat should have.
