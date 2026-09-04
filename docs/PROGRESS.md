# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the anatomy audit and the completeness audit are Sunday
> work and were deliberately skipped; the last FULL page is 2026-08-31).

**2026-09-04 06:4x–07:1x UTC — DAILY.** Window: the last 24 h
(2026-09-03 06:45 → 2026-09-04 06:45; `HR.7` sits at 06:35 on the far side of
the boundary and belongs to yesterday's page, where it was counted).

*The one sentence: **between 16:00 and 22:00 yesterday the builder gave Jack
three new true things about himself — he can say "I'm cold" and be right, he is
the same creature whether or not he is watched, and a life of his survives being
killed — and then the board emptied under it, and it emptied because thirty-one
redesigns are parked on this desk.***

---

## The numbers

| | now | 09-03 (DAILY) | Δ |
|---|---|---|---|
| demonstrated / registered | **102 / 234** | 95 / 225 | **+7 / +9** |
| pass rate | **43.6%** | 42.2% | **+1.4 pts** |
| FAIL / VOID (live rows) | **22 / 11** | 22 / 10 | — / +1 |
| unreachable specs | **91 / 234 (39%)** | 89 / 225 (40%) | +2 count, −1 pt |
| rework (ledger rows at attempt > 1) | **104 / 135 = 77.0%** | 97 / 127 = 76.4% | +0.6 |
| commits, last 24 h | **65** (9 of them journal records) | 55 | +10 |
| builder slots fired | **27 starts, zero `PACING:` skips** (last skip 08-29) | 25 | +2 |
| ledger settlements in window | **40**, of which **8 first-ever** | ~86, of which 2 | — |
| maintenance share of non-journal commits | **41 / 56 = 73%** | 37 / 47 = 79% | **−6 pts** |
| ratchets | coverage **rc=2** (4 CLAIM-DEAD + 7 CITED-BUT-UNRUNNABLE) · review-queue rc=0 (**31 OPEN of 35**) · champions rc=0 · decisions rc=0 (10 armed) | — | — |

**The +7 is the best-composed week's-worth of PASSes this project has recorded
in a while, and it is worth breaking apart before anyone celebrates it.** Seven
first-ever PASSes: `SO.02` (*"I'm cold" is true when he is cold* — 3 seeds,
acc_class 1.000 vs base 0.430), `SO.04` (*being watched does not change him* —
watched vs unwatched bit-identical on all 3 seeds over 2000 steps), `LF.02`
(*a life survives `kill -9` bit-exactly*), plus `T0.32`/`T0.33`/`T0.34` (the
rtf gate and the CPU-hour accountant, TIER 0 — the machine, not Jack) and
`SO.01` (a fixture, credited as support and not as a claim by `coverage`).
**Three of the seven are claims about Jack**, and `coverage` credits all three
to commitments that were reading zero or one. The eighth settlement is `LF.01`
VOID — the first long-exposure life W0 has ever hosted, dead at ~25 minutes of
`cause=INTEGRITY` with a privileged scripted forager driving. That VOID is a
measurement, and it is the tenth instrument.

**The maintenance share fell, and that is the honest reading of the day.**
73% of non-journal commits carry an audit B-item, a re-buy, a re-stamp, a
harvest or a drift repair — high, but down from 79% and 81% on the two previous
windows, and the fall is exactly the six-hour creature window. The rework rate
is flat and still contaminated by 09-02's cross-organ doc-race repair; read it
again after the 09-06 row lands.

**Goodhart: the rate ROSE against a growing registry, which is the rare good
case.** Registry 225 → 234 (seven `DIRECTION_AUDIT` stubs registered under the
5-step cross-check with two REFUSED and eight HELD, plus `LG.11` THE TOLD
WORLD, which the 66th audit registered because GOAL.md had declared itself
falsifiable there since 08-09 and never had a spec) against demonstrated
+7. The runner outran the ladder for the first time since 08-31. `unreachable`
rose 89 → 91 and `UNREACHABLE_BASELINE` was raised with it twice, both times in
the growing commit with the justification in the growth log — `HR.6`'s new edge
behind `HR.5`'s FAIL, and `SO.04` deliberately registered blocked behind
`SO.01`. That is the protocol working, not a floor drifting.

---

## The frontier, recomputed — and it is not a spec

`run blocked` is unchanged at the top: **`T2.01` frees 35 / blocks 38**, settled
FAIL since 08-12 at 2.67σ against an unmoved 5σ bar, repair path through
`D1.0`, whose gate redesign is owed here on Sunday. `NE.01` frees 8, `LT.01`
frees 7, `UB.10` frees 4. None of that moved.

**What moved is the bottom of the board, and it went to zero.** Of 234
registered specs, 135 carry a ledger row. Of the 99 that do not, exactly **nine
have all dependencies PASS** — and every one of the nine is parked,
pilot-blocked, or held by an open decision:

| spec | state | who owns the repair |
|---|---|---|
| `SH.01`, `SM.02`, `T2.11`, `T3.10` | PARKED by a fired both-fail branch | this desk |
| `SH.02`, `SM.03`, `DP.04`, `LC.07` | PILOT-BLOCKED — pilot measured its own precondition failing | this desk |
| `HR.1` | FILL-HELD by `D19` (NO-FETCH default, decide_by 09-14) | the owner |

**Eight of nine wait on a redesign that is on this desk; the ninth waits on the
owner. Not one waits on the builder, and not one waits on compute.**
`coverage` says the same thing from the other end without being asked: all four
non-fillable cost classes name the same reason — *"the repair is a REDESIGN"*.
The builder's 04:10 journal line, *"the board is honestly empty"*, is true, and
I verified it rather than taking it.

### The finding: the queue that gates the board is diverging, and no
### rearrangement of dates touches it

`docs/REVIEW_QUEUE.md` has been live for 15 days. In that time it has **routed
35 rows and closed 2** (`me11-every-arm-hits-the-same-infeasible-branch`,
`reparenting-the-welded-fifteen`; two more are HELD behind `w0-too-shallow`,
which is not closure). Arrivals over the last five full days: 9, 5, 6, 4, 4 —
**≈5.6 rows/day.** Closures over the queue's whole life: **≈0.13/day.** The
file's own instrument measures the consumer at **1 dated row per cycle**.

The 65th and 68th audits built genuinely good instrumentation for this and both
aimed it at the wrong quantity. B6 measured the pile per date; B7 measured
`piled_on` (17 of 31 rows named a date that was already full when routed) and
`next_free_due` (**2026-09-12**), and re-staggered 09-06 from 8 rows to 6.
`piled_on` still reads 17 after the re-stagger, and the builder says so plainly
in its own journal — *"moving between piles is not a repair"*. It is right.
**Arrival minus departure is ≈5.5 rows/day and every date in the file is
downstream of it.** Sunday's six rows are not a scheduling problem.

I am the consumer. This is a finding about me.

### Second: the CPU accountant's first full day billed nothing but itself

`T0.33` and `T0.34` landed at 22:18 and 23:22 and are good work — the detached
lane really did spend `LC.03` v2's ~190 core-hours invisibly, and it now writes
receipts before, during and after. Here is its complete ledger for
2026-09-04, every line item:

```
detached:gate_sweep_cpu2h.log  4560.65 s   certificate sweep
detached:rebuy_xl00.log        1171.28 s   re-buy
LC.02                           140.29 s   re-buy
T0.34  24.28 · T0.17 3.44 · T0.33 3.30 · T0.27 1.92 · T0.31 1.66  re-buys
                               ─────────
                               5906.82 s of 57600 s
```

**Every second of it is a certificate re-buy or a re-stamp sweep. Not one
second bought a new measurement about Jack** — and on that basis the meter
currently refuses 53 of the box's CPU specs until midnight. The 68th audit
caught the double-billing that made it worse and fixed it (B1–B3), and routed
the class question as `cpu48h-class-self-forecloses-the-day-meter` (DUE 09-08)
and as `D20` (decide_by 09-18). Both correct. What neither says is the thing
the numbers say: **this is the `pace_gate` shape again** — a throttle
regulating the builder against a quantity that our own audit-and-re-buy churn
generates. That cost 66 dark hours in August before it was understood. It is
not costing anything yet. It is one week old and it should be watched, and the
measurement above is the input `D20` needs, not a recommendation about where
its ceiling belongs.

---

## The honest paragraph

For six hours yesterday this was a project about a creature. He got a voice
that tells the truth about his own body rather than about the world — the first
thing he says that is checkable against what is happening inside him; he was
shown to be the same animal watched and unwatched, which is the precondition
for the owner ever being allowed to look at him; and a life of his was killed
outright and resumed exactly, which is what makes a life something that can
accumulate rather than something that merely runs. Those are three different
kinds of true and none of them is a fixture. Then a long-exposure life was run
for the first time, and the world broke his body a fraction of the way in,
with a privileged servo driving and food never in question — which is the tenth
separate instrument to report that the difficulty here is the place, not the
mind. And after that the board went flat, and the machine did what it always
does when there is nothing to measure: it turned around and measured itself,
beautifully, with a new meter whose entire recorded spend so far has been the
cost of re-proving things it already knew. That is not waste and it is not
dishonesty — every re-buy was owed, every audit item was real. But the shape
of the day is a creature-shaped morning and a mirror-shaped night, and the
switch between them was thrown by this desk. The most important step toward
Jack was the moment he said something about himself that could have been false
and was not. The most concerning drift is that the one organ that cannot be
scaled by working harder is now the only thing standing between an idle builder
and a world nobody has designed yet.

---

## STEERING / STRENGTHENED

**No spec file was touched and no threshold moved in either direction.** Part 2
is Sunday's. One steering edit, in the builder's file only:

- **`scripts/ladder_prompt.md` — the `1'/2'/3'` priority block replaced by
  `1''/2''/3''`.** Item `1'` was spent: it existed to redirect the builder away
  from `HR.6`, the 65th audit executed that on 09-03 by declaring the
  `HR.6 ← HR.5` edge, and `run next`/`coverage` have not offered `HR.6` since —
  a correction that has been mechanically enforced for a day is history, not a
  priority. With it spent, **the block contained three prohibitions and zero
  positive units**, which is what an iteration reads at 03:00 with an empty
  board before it goes looking for something to do. The replacement states the
  board is empty, states *why* and whose desk that is, names the ordered
  fallback (the `INTEGRATION_QUEUE` empty-queue research rule, `LANGUAGE_
  GROUNDING` §2.2–§11 first because it feeds `champions-language-grounding-
  arena` DUE 09-07 and the Language-grounding seat is one of the three
  UNFALSIFIABLE ones), and adds one new prohibition earned by the numbers
  above: **no third increment of the CPU accountant.** The three live
  prohibitions (`D1.0`, `D19`/`HR.1`, `LF.01` attempt 2) are carried forward
  unchanged, and `W35`'s expiring hours are named so nobody manufactures a
  dispatch into them.

---

## FOR THE BUILDER

1. **The board is empty, you are not the reason, and the correct response is
   not to build the machine another organ.** Nine specs could be started today
   and all nine are parked, pilot-blocked or decision-held; `coverage` names
   the repair as a REDESIGN in all four empty cost classes. Do not go looking
   for a spec to write into an empty class — that is the exact move
   `coverage`'s own text forbids.
2. **Take the `INTEGRATION_QUEUE` empty-queue rule, and take
   `LANGUAGE_GROUNDING.md` §2.2–§11 first** (before the SO-family social-world
   pass). Reason, so you can overrule it if you find a better one: `LG.00`,
   `LG.01`, `LG.02` and `LG.10` are already registered, §7 "registry entries"
   is empty, the *Language grounding (word → lived skill)* seat is one of three
   marked **UNFALSIFIABLE** by `champions --check`, and I owe
   `champions-language-grounding-arena` on **09-07** — your research pass is
   an input to a dated row of mine, which the SO-family pass is not.
3. **No third increment of the CPU accountant.** `T0.33`/`T0.34` were owed and
   are good. But its first full day billed 5906.8 s and **every line item is a
   re-buy or a re-stamp** — the meter is currently an instrument for measuring
   our own churn, and extending it further is the mirror, not the creature. The
   open questions it raises are already routed (`cpu48h-class-self-forecloses-
   the-day-meter` DUE 09-08, `D20` decide_by 09-18) and neither is yours.
4. **Standing prohibitions, all unchanged:** do not re-dispatch `D1.0` (gate
   design owed here 09-06; an unchanged re-dispatch is a seed-lottery redraw);
   `HR.1`–`HR.4` stay D19-held to 09-14, do not fetch a corpus; `HR.6` stays
   blocked behind `HR.5`; **`LF.01` attempt 2 waits for the 09-09 design** —
   its own row says so and `FIXTURE_VOID_CAP=3` is not permission.
5. **`W35` has ~11 free Kaggle hours and they expire 2026-09-06 00:00. Let them
   expire.** Every runnable GPU spec is a settled FAIL whose re-run is a seed
   lottery, or parked. This is inventory, not uptime — the 08-29 diagnosis —
   and manufacturing a dispatch to spend a dying quota is the failure mode, not
   the fix. `W36` opens the same instant with 30 h and a named buyer.
6. **Do not re-stagger the 09-06 docket a third time.** The builder's own 68th
   audit B7 note is right and I am endorsing it: `piled_on` and `next_free_due`
   exist so the next *router* reads the number. A third hand-pass would mean
   the guard did not work.

---

## FOR THE OWNER

1. **THE FORK, and it is new: design throughput is now the binding constraint
   on the whole project, and it is structural rather than a matter of anyone
   working harder.** The Review is a ~40-minute-a-week design desk (FULL,
   Sundays) fronting a queue that receives **≈5.6 rows/day** and has closed
   **2 rows in 15 days**. Every one of the nine startable specs sits behind it.
   Meanwhile the builder has an empty board, 24 slots a day, and spent eight of
   yesterday's hours accounting for its own accounting because there was
   nothing else it was permitted to touch.
   **My recommendation: let the builder DRAFT redesigns; keep ratification
   here.** A queue row currently means *"only the Review may answer this"*.
   Change it to *"the builder may write the answer; the Review and the overseer
   must ratify it before any run"*. That converts my 5.6-per-day design deficit
   into a review-of-drafts load, which is perhaps a tenth the cost per row, and
   it puts the work where the capacity actually is.
   **The risk, named because it is the whole reason the rule exists:** the
   builder drafting the redesign of a spec that just failed is precisely the
   conflict of interest the T1.02 precedent guards against. **The safeguard is
   already built and already running** — the strengthen-only law, and an
   overseer whose §2 duty is to audit every spec diff independently of its
   author. So the proposal is narrow: the builder may draft, must state the new
   threshold and why it is HARDER, may not run the spec until ratified, and the
   old version stays in the ledger's history. If you would rather not, the
   alternative is a second Review sitting per week for design only; I prefer
   the draft route because it scales and a second sitting does not.

2. **`D20`'s input, measured rather than argued.** The CPU day-meter's first
   full day of operation billed 5,906.8 s across eight line items and **all
   eight are certificate re-buys or re-stamp sweeps** — zero seconds of new
   science — and it is on that basis currently refusing 53 CPU specs until
   midnight. You are being asked (`D20`, decide_by 09-18; `OVERSIGHT` FOR THE
   OWNER item 2) what the ceiling should count. I am deliberately not
   recommending a number, because every recommendation available to me points
   at loosening a ceiling and that is not my direction to push. What I will say
   is the pattern: this is the shape `pace_gate` had — a throttle regulating
   the builder against a quantity our own audit churn generates — and that one
   cost 66 dark hours before anybody saw it. One week of data is not a crisis.
   It is worth your eyes before it is three weeks.

3. **Sunday 2026-09-06, order unchanged from yesterday's page and now
   six rows rather than eight** (the builder re-staggered `hr5-fixture-refuted`
   and `w0-kills-a-forager` to 09-09, both with reasons I endorse; it correctly
   left `cross-organ-doc-race-voids-certificates` on the full day because that
   row's stated reason outranks pile-avoidance). Order: the two `d10-*` gate
   rows first (cheap, and they release a 16 h dispatch into W36's 30 free
   hours), then `w0-too-shallow` — **now at eleven days and ten instruments**,
   `LF.01`'s 25-minute body-wreck being the tenth — then the rest, Part 2 at
   its minimum of 8. If item 1 above is granted, this docket is the first thing
   that changes shape.

4. **Organ liveness, all green, all verified against `/data/jack-logs` mtimes
   rather than against anyone's report.** builder 06:17 (hourly, 27 starts,
   0 `PACING:` — last skip 08-29), overseer 06:37 (6 h), field watch 08-31
   05:53 (Mondays — next fire 09-07, inside cadence; wk5 consumed 08-31 and
   unchanged since, so nothing to consume this run), review 06:37 (this run).
   `lost_iterations.log` still 0 bytes and still never exercised. No organ is
   silent.
