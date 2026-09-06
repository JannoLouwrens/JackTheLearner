# OVERSIGHT — 76th audit, 2026-09-06 00:37–01:0x UTC (at `f03c2fe`)

## VERDICT: DRIFTING — not because the day went badly. It went well: **+1 credited claim about Jack** (`SO.08`), 25 of 25 builder slots `rc=0`, three armed defaults fired on time and in the verified order, the ledger clean on every mechanical check, and nothing loosened anywhere. It is DRIFTING because of what that good day did *not* touch: **four of the owner's own constitutional commitments still have zero passing claims and every claim spec behind them is parked or foreclosed**, GOAL.md now cites **four specs that resolve to corpses** (`GEN.02/03/06/09`, all NEW this window), and the single artefact that would move any of it — the W1 design owed by `w0-too-shallow` — is **13 days old, past its own `DUE`, and due again in under six hours**. The ladder keeps getting better at measuring itself. The world its commitments need has not been built.

Two live defects found in an instrument that shipped **six hours ago**, and one
provenance gap on the owner's desk. Neither touches the ledger; both are in
Section 9.

---

## 1. Integrity of the ledger — CLEAN (fifth consecutive audit)

All **105 PASS** rows of **140** (242 registered) resolved mechanically, not by
report:

- **0** with no implementation in `experiments/tests/`.
- **0** whose recorded `commit` fails `git cat-file -e <sha>^{commit}`.
- **0** with no `control` declared in the registry.
- **2** carry empty `control_metrics` — `T0.01` and `T0.10` — and both declare
  `"NONE, BY DECISION (52nd audit B5)"` with the reason on the spec (*an import
  either raises or it does not*; *an external service returning real artifact
  bytes is its own falsifier*). Pre-registered refusals, not silent omissions.

Status distribution: 105 PASS / 22 FAIL / 13 VOID. **No finding.**

## 2. Thresholds and controls over 7 days — CLEAN

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`
and `experiments/tests/`. **No numeric threshold moved in the loosening
direction, no control deleted or weakened, no `_check` gained an `or`, no seed
count reduced, no assertion removed.**

The one change that *looks* like a loosening is not one, and I checked it line
by line rather than accepting its commit message:

- **`7aa9619` — `SO.08` re-declared `Budget.CPU_LONG` → `Budget.CPU_FAST`.**
  This is a **cost declaration, not a threshold**. Admission loosens; the
  **child-kill window TIGHTENS, 54,000 s → 1,800 s**. It is justified by a
  measurement that is recorded in the spec's own notes as a SIZING RECORD
  (`_measure(0)` 0.36 s, `_measure_controls(0)` 0.38 s, 0.74 s/seed at full
  `N_ROUNDS=240`). The struck class was typed at `b6518dd` **nine hours before
  the implementation existed** and enumerated 54,000 s — ~28,000× the measured
  cost. No gate constant moved in that diff; I diffed it in full. This is the
  75th audit's B1 executed correctly.

Movement in the window is in the strengthening direction: `T3.09` `N_LIVES`
16→32, `LG.03`'s blind twin raised to `max(knn, ridge)`, `LG.10` `TEMP`→1.0,
`T0.17`/`T0.23`/`T0.28` gaining properties (`N_PROPERTIES` 14→15).
**No finding.**

## 3. Drift from the goal — THE FINDING OF THIS AUDIT

**What the builder built in the last 24 h, and what each serves:**

| work | GOAL.md sentence |
|---|---|
| `SO.08` implemented + PASS | *"his diary records who left it — so gratitude, like trust, has somewhere real to grow"* — **direct, credited** |
| `SO.07` harvest → VOID + routed | same commitment; honest negative |
| `D21`/`D16`/`D15` defaults fired | conduct (SYSTEM.md rule 3) |
| 75th B1/B3, 74th B1–B4, 73rd B1–B3 | honesty infrastructure |
| 4 × `T0.21`/`T0.28`/`T0.31`/`T0.33` re-buys | certificate maintenance |

**Nothing here is drift in the "serves no GOAL.md sentence" sense.** `SO.08` is
the real thing: *the diary records WHOSE hands, and he acts on it* — worst-seed
last-quarter divergence 0.667 against a 0.40 bar pre-registered from binomial
arithmetic before any number existed, with null, donor-shuffle and equal-donors
legs all collapsing as declared. It is the first **credited** claim about Jack
in three days (`social/other agents` 2 pass → 3 pass), and it PASSed at 19:12,
six hours *before* the midnight window the Review had staged four slots to
protect. The builder beat its own plan.

**The drift is the converse question, and it is measured, not asserted:**

- **4 commitments are CLAIM-DEAD** — *smell*, *balance*, *shelter/building*,
  *thermal (kills)*. Zero passing claims each; every claim spec behind them
  PARKED or FORECLOSED (`SM.02`→`SM.03` PILOT-BLOCKED; `BA.02`→`BA.03`
  VOID-FORECLOSED; `SH.01`→`SH.02` PILOT-BLOCKED). Three of these are the
  owner's own words — *"too cold kills him"*, *"he builds a shelter"*, and the
  full sensory inventory. **Not one died because Jack failed to learn.** Every
  one died on the venue.
- **9 more commitments have live claim specs and nothing passing** — including
  *sleep* (5 specs), *fast/slow* (8 specs), *plasticity*, *hunger/thirst*,
  *proprioception*, *touch*, *tool use*, *told world*.
- **`one brain / unison`: 25 specs, 1 pass.** The thesis itself.
- **4 NEW CITED-BUT-UNRUNNABLE**: GOAL.md's three-expansions section cites
  `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09` in the present tense; all four are now
  `welded<-LC.07`, which went PILOT-BLOCKED on 09-01. The whole published
  roadmap past the jungle resolves to corpses. Total CITED-BUT-UNRUNNABLE: 7.
- **`coverage` exits rc=2.** Highest-priority class is clean —
  **0 commitments with NO declared spec** — so the 08-10 miss has not recurred.
  The red is 4 CLAIM-DEAD + 7 CITED-BUT-UNRUNNABLE + 3 empty cost classes.

Every one of these is routed and dated. All of them are downstream of one row.

## 4. Is the builder alive and productive? — YES, and this is the strongest section

`/data/jack-logs/ladder.log`, 2026-09-05 00:37 → 2026-09-06 00:37:

- **25 iteration starts, 25 ended `rc=0`.** Zero aborts, zero load refusals.
- **Zero `PACING:` skips** (last skip 2026-08-29).
- **PASS delta +1: 104 → 105**, and it is `SO.08` — a credited creature claim,
  not a fixture and not a Tier 0 re-buy.
- `lost_iterations.log` **0 bytes**. No undeclared processes. Tree clean,
  everything pushed.
- The empty-board stops at 20:07/21:07/22:07/23:07 were each **independently
  re-derived** rather than inherited, and the 21:07 slot explicitly declined an
  `amend --doc-only` sweep after diffing four stale certificates and finding
  them deliberate holds. That is the correct behaviour and it was journalled
  with its reasoning.

**No finding.** The builder is not the problem and has not been for some time.

## 5. Compute honesty — CLEAN, with one standing cost

**CPU (`cpu_budget.json`).** 09-05 billed **9,364.77 s**, of which **9,205.09 s
(98.3%) went to `SO.07`** — a single honest VOID at its pre-registered
reference lane. Tier-0 certificate re-buys took **155.81 s (1.7%)**. This
inverts the 09-04 pattern the Review flagged (5,906.8 s, *every line item* a
re-buy or re-stamp). Today's meter: 1.75 s spent, `cpu<2h` slack 3,600 s.

**GPU (`gpu_budget.json`).** W35 closed at **18.93 Kaggle h + 0.27 Colab h of
30** — **~11 h expired unspent, the third consecutive week.** Every charged W35
job resolves to a ledger artefact: the three large jobs (4.08 + 6.03 + 6.06 =
**16.17 h**) are `D1.0`'s VOID, which has a row; `1788297232` (0.44 h) is
`LC.07`'s pilot, which has a PILOT RECORD. **No GPU hours were spent without
something to show for them.** The builder declined to manufacture a dispatch to
burn the expiring hours and said so — correct: all 6 dispatchable specs are
VOID-arms behind routed dispositions.

**W36 opened today (`date -u +%Y-W%U` = `2026-W36`) with 30 fresh hours.** Its
named buyer is `D1.0`'s successor at ~16 h, gated on the two `d10-*` rows owed
by today's FULL. **No finding on honesty; the cost is Section 6's.**

## 6. Stuck decisions — nothing overdue, nothing escalated, nothing to arm

`decisions --check` **rc=0**:

- **0 `MEANS-ESCALATED`** — no measurable fork is sitting on the owner's desk.
  The D1 disease is absent.
- **0 `UNDECLARED`** (0/10 ratchet) — **there is nothing for me to arm this
  audit.** I am stating that rather than manufacturing an entry to satisfy the
  standing instruction.
- **0 `OVERDUE`.** All three defaults whose `decide_by` was 2026-09-05
  (`D21`, `D16`, `D15`) fired at 00:14–00:16 today, in the pre-verified order,
  each journalled with the words *"the owner did not rule by <date>"* and each
  with its reversal named in `DECISIONS_RESOLVED.md`. `D21`'s same-day race
  against the 06:37 FULL was won by **six hours**. I checked the firings for
  ratchet violations: `D16` was a deliberate **no-op** (`T0.27` stays RED, the
  guard unedited — the visible failure kept over the exonerating green), `D21`
  touched only row ordering, and `D15` is reviewed in Section 9. **None widened
  what the project is permitted to take.**
- Live open: `D17` (09-07), `D22` (09-08), `D18` (09-09), `D23` (09-11),
  `D19` (09-14). `0/3` unrouted owner-asks, `0/0` vanished.
- **`D22` is the one that matters** and it is the owner's: `REVIEW_QUEUE` reads
  **36 OPEN / 2 HELD / 38 live of 41 routed, oldest 13 d, arrivals 5.00/cycle,
  disposals 0.29/cycle, drain UNBOUNDED.** **Six rows are due today** against a
  measured capacity of 1/cycle. That is six promises scheduled to break
  together, in one 40-minute run that also owes Part 2 and both completeness
  audits.

## 7. Bakeoff hygiene — CLEAN

`DECISIONS_RESOLVED.md` reviewed for the three firings. Each records the
question, the options, which was struck at arming and why, the price of the
default stated rather than buried, and an explicit reversal. `D15`'s entry
carries its counterargument verbatim from the arming. No VOID was treated as a
verdict, no winner was chosen inside a noise margin, no decision was made
without a learning gate. `champions --check` **rc=0**, ratchet unmoved
(0/0 phantom arena, 2/3 unfalsifiable, 2/2 unverified verdicts, 3/3 trigger
debt). **No finding.**

## 8. Review-queue liveness — the *work* half is red, the *schedule* half fired correctly

`review_queue` reports **0 violations** — no `OVERDUE`, no `STALE`, no
`HOLD-WITHOUT-A-CLOCK`, no `VANISHED`, no `CLOCK-REMOVED`. Rows are being
dispositioned honestly.

The schedule half caught what the work half cannot: **the 09-05 DAILY Review
died at `Reached max turns (60)`**, was sealed `INCOMPLETE RUN` at 06:52, and
**never appended its row to `docs/PROGRESS_LOG.md`** — so the trend file that
exists *"so trends survive any single Review"* has a hole at 09-05 (newest row:
2026-09-04). `lib_seal.sh` detected the miss at 00:37:04 today and stamped
`docs/PROGRESS.md` **STALE** (`f03c2fe`). Both guards worked. **Second
consecutive incomplete Review run.**

## 9. What I found that no instrument reported

### 9.1 `D15` clause (d) shipped six hours ago and its model attribution is wrong for 3 of its 4 organs — and mis-attributes the 4th

`usage_ledger()` (`scripts/lib_usage.sh:246`) resolves the model as
`mdl="${JACK_LOOP_MODEL:-opus}"` and greps `week:${mdl}`. Verified against the
meter's actual output, which prints exactly three lines: `session`,
**`week:Fable`**, `week:all models`. **There is no `week:opus` line, ever.**

- `overseer.sh:119` uses `JACK_OVERSEER_MODEL`, `review.sh:37` uses
  `JACK_REVIEW_MODEL`, `field_watch.sh:38` uses `JACK_FIELDWATCH_MODEL`.
  **None of the three sets `JACK_LOOP_MODEL`**, so all three fall to `opus`,
  match nothing, and write `model_pct: null` **permanently**.
- **Confirmed on the first real append.** The ledger has two lines: the hand
  self-test at 00:15:49 (`"model_pct":31`) and the first live one at 00:37:07
  (`"organ":"overseer" … "model_pct":null`) — same box, 22 minutes apart,
  `pct` read fine both times.
- **The 4th organ is worse than null.** `usage_ledger` takes only
  `(organ, phase)` and reads the env var, so it can never see the model
  `model_chain` actually walked to. When `week:Fable` pins at 100% the loop
  runs on **Opus** by design (documented, and it did so all day on 09-04) —
  and the ledger will record that spend against the **Fable** line.

`D15`'s own record says (d) exists so *"the next meter dispute reads
attribution instead of the co-occurrence inference that produced three
falsified price models in one week."* **Attribution that names the wrong model is that
same disease with a JSON wrapper.** Mitigating and stated plainly: **`pct`
(`week:all models`) is correct for every organ**, and that is the gate line —
the load-bearing number is fine. No gate is affected. This is a new instrument
wrong on day one, before anyone has relied on it.

### 9.2 `D15` clause (c) exempts the one audit that cannot read the day's Review

The default fired legally and is spend-reducing; I am not disputing it. But
neither the arming text nor the firing record names this consequence:

The overseer runs **00:37, 06:37, 12:37, 18:37**. The Review runs **06:37**.
Clause (c) exempts *the first completed audit of each UTC day* — **00:37** —
and paces the other three behind `pace_gate` (`allow = 25 + ceil(0.65 ×
week_elapsed)`). So under usage pressure the audit **guaranteed** to survive is
the one that reads **yesterday's** Review page, and the two that could read
**today's** (12:37, 18:37) are the first to be skipped.

This desk's own prompt was amended precisely because *"the Review reads you
every morning and until now you did not read it back"* — and it cost a real
recommendation on 09-03. **Clause (c) makes that reading the first thing to
go.** Today is the demonstration: I am the exempt slot, and the page I am
required to read back is stamped both **STALE** and **INCOMPLETE**.

Not hypothetical: on 09-04 the meter read 60–61% against a pace line of ~67.
The repair in B2 preserves D15's spend arithmetic exactly — still one
guaranteed audit per day — and does not reverse the fired default.

### 9.3 An `INCOMPLETE`-sealed page's owner-ask is now a live armed decision, and nothing says so

`D23` — currently on the owner's desk, `class: goal`, `decide_by` **2026-09-11**,
with a default that **fires by silence** — was written by the 09-05 Review run
that died at max turns. It reached `DECISIONS_NEEDED.md` through `e034b94`,
where the builder committed the dead run's uncommitted edits.

The builder did this correctly: the death is named in that commit message, and
it verified `decisions --check` before committing. And **`D23`'s facts are
true** — I re-measured them today: `FAIL-UNOWNED` is at floor 0 with the four
orphans routed to 09-13, and the queue's drain does read UNBOUNDED at
0.29 disposals/cycle. **This is not a claim that `D23` is wrong.**

The gap is provenance. `lib_seal.sh` banners **the page**; it does not mark
**what the page emitted into other desks**. `D23`'s header reads
`(2026-09-05, Review, DAILY)` with nothing to say its authoring run never
completed its own checklist — and `decisions --check` reports it as an ordinary
armed entry. The 08-30 lesson (*"a report that can be read without its status
is a report whose status does not exist"*) stops at the report. **A dead run's
routed artefacts outlive the banner that was supposed to qualify them**, and
one of them is on a clock to change an instrument by default in five days.

---

## FOR THE BUILDER

1. **Fix `usage_ledger()`'s model resolution (9.1) — it is one line and it is
   wrong three ways.** Resolve the per-organ model from the variable that organ
   actually uses, and for the builder pass the **walked** model (the one
   `model_chain`/`chain_reading` returned) rather than the configured primary —
   the function currently cannot see it, so it needs the model as a parameter,
   not an env read. Verify by asserting a non-null `model_pct` on an append from
   each of the four organs, and by asserting that a forced chain-walk records
   the model that ran. Additive to a log nothing gates on; no threshold, no
   control, no new accounting surface — so the standing CPU-accountant
   prohibition is not touched.
2. **Move `D15` clause (c)'s exemption to the first completed audit AT OR AFTER
   the Review's daily slot (9.2).** This does **not** reverse the fired default
   and **does not change its spend**: still exactly one unpaced audit per UTC
   day. It changes only *which* one, so that the guaranteed adversarial pass is
   one that can read the day's Review rather than yesterday's. The completion-
   point stamp beside `NOOP_STATE` stays exactly as built (a dead audit must not
   consume the exemption). If you judge this to be outside the armed text, say
   so on the record and route it rather than doing it — but say so, do not
   silently leave it.
3. **Give a dead run's routed artefacts the provenance its page gets (9.3).**
   When `lib_seal.sh` fires, stamp the entries that run created in
   `DECISIONS_NEEDED.md` / `REVIEW_QUEUE.md` / `CHAMPIONS.md` in the same commit
   — one line naming the incomplete run and its exit code. Start by
   back-stamping **`D23`**, whose default fires 09-11. Do not weaken, re-date or
   un-arm it: the entry's facts re-measure true today. This is a disclosure, not
   a reversal.
4. **`PROGRESS_LOG.md` lost its 09-05 row** because the Review died before
   appending. The append is at the end of a run that has now failed to complete
   **twice running**. Move it earlier, or have `lib_seal.sh` write a row marked
   `INCOMPLETE` — a trend file that only records the runs that finished
   over-reports the desk's own throughput, which is the exact quantity `D22`
   turns on.
5. **Standing prohibitions, unchanged and re-verified:** do not re-dispatch
   `D1.0`; `HR.1`–`HR.4` stay D19-held to 09-14, do not fetch a corpus; `HR.6`
   stays behind `HR.5`; `LF.01` attempt 2 waits for the 09-09 design; the ranker
   still cannot see `T2.01 → D1.0`. W36's 30 h are gated on today's two `d10-*`
   rows — **do not manufacture a dispatch to burn them**, as you correctly
   declined to do with W35's.

## FOR THE OWNER

1. **`D23` came from a run that never finished, and you should know that before
   2026-09-11.** No decision needed and nothing is wrong with the entry — I
   re-measured its arithmetic today and it holds. But its authoring Review run
   hit max turns and was sealed *"INCOMPLETE RUN — THIS IS A DRAFT, NOT A
   FINDING"*, and that qualification never travelled with the entry onto your
   desk. Its default fires by silence in five days. Builder item 3 back-stamps
   the provenance; nothing about the entry changes.

2. **The one thing that would move the goal is due in under six hours, and it is
   13 days late.** Four of your own constitutional commitments — *smell*,
   *balance*, *shelter/building*, *thermal (kills)* — have **zero passing
   claims**, and every claim spec behind them is parked or foreclosed. Not one
   died because Jack failed to learn; every one died on the **venue**.
   `GOAL.md`'s published roadmap past the jungle now cites four specs
   (`GEN.02/03/06/09`) that resolve to corpses. All of it is downstream of
   `w0-too-shallow` — the W1 design — which is **13 days old, past its own
   `DUE`, carries eleven instruments, and `D21`'s fired default has placed it as
   today's FULL's first design item, behind only the two cheap `d10-*` gate
   rows.** The previous Review's recommendation stands and I am repeating it
   because it rolled off that page unanswered: **W1 should stop being a queue
   row and become the project's stated stage.** We are at step 2 of GOAL.md's
   six-step path, building senses to point at a step-6 world.

3. **`D22` (due 09-08) is the arithmetic behind everything above.** The design
   desk takes 5.00 rows/cycle and disposes 0.29; 38 rows are live, the oldest
   13 days; **six of them promise today**, in a run that also owes Part 2 and
   both completeness audits, against a measured capacity of one. Every dated
   promise on the board is downstream of that ratio. Silence through 09-08 costs
   roughly 17 further net rows at the measured rate.

4. NO-DECISION: liveness. All four organs live, verified against
   `/data/jack-logs` mtimes rather than anyone's report — builder 00:18 (hourly,
   25 starts in the window, 25 `rc=0`, zero `PACING:`), overseer 00:37 (this
   run), review 06:52 on 09-05 (**died at max turns, second consecutive
   incomplete run**; next FULL 06:37 today), field watch 08-31 (Mondays; next
   fire 09-07, inside cadence). `lost_iterations.log` 0 bytes and still never
   exercised.
