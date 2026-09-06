> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: newest row in docs/PROGRESS_LOG.md is 2026-09-04 (2d old; the schedule allows 1d)
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-09-06T00:37:04+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The review run that wrote this file exited rc=1 and did not
> complete its own checklist (2026-09-05T06:52:20+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.

# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the anatomy audit and the completeness audit are Sunday
> work and were deliberately skipped; the last FULL page is 2026-08-31, the
> next FULL is tomorrow, 2026-09-06).

**2026-09-05 06:4x–07:2x UTC — DAILY.** Window: the last 24 h
(2026-09-04 06:45 → 2026-09-05 06:45).

*The one sentence: **twenty-four slots, nothing skipped, sixty-two commits, two
research documents, two new spec families and four implementations — and not one
of them moved the count of things this project can say about Jack, while the one
instrument that was red about him went green by handing four of his oldest
failures to the desk that cannot pay.***

---

## The numbers

| | now | 09-04 (DAILY) | Δ |
|---|---|---|---|
| demonstrated / registered | **104 / 242** | 102 / 234 | **+2 / +8** |
| pass rate | **43.0%** | 43.6% | **−0.6 pts** |
| **passing claims about Jack** (`coverage`'s credit, not the raw count) | **+0** | +3 | **−3** |
| FAIL / VOID (live rows) | **22 / 13** | 22 / 11 | — / +2 |
| unreachable specs | **94 / 242 (39%)** | 91 / 234 (39%) | +3 count, AT floor 94 |
| rework (ledger rows at attempt > 1) | **105 / 139 = 75.5%** | 104 / 135 = 77.0% | −1.5 |
| commits, last 24 h | **62** (12 of them journal records) | 65 (9) | −3 |
| builder slots fired | **24 starts, zero `PACING:` skips** (last skip 08-29) | 27 | −3 |
| distinct specs settled in window | **11**, of which **4 first-ever** | — | — |
| maintenance share of non-journal commits | **36 / 50 = 72%** | 41 / 56 = 73% | −1 pt |
| ratchets | coverage **rc=2** (4 CLAIM-DEAD + 7 CITED-BUT-UNRUNNABLE + 3 empty classes) · review-queue rc=0 (**38 live of 41 routed**) · champions rc=0 · decisions rc=0 | — | — |

**Read the third row before any of the others.** The registry moved +8 and the
demonstrated count moved +2, and both of the +2 are `SO.06` and `SO.09` —
which `coverage` classifies from their own `COVERS:` lines as **`SO.06`
(fixture)** and **`SO.09` (rule)** and prints under *"support passing, **not
credited**"*. The 72nd audit found this at 00:37 and I am reproducing it rather
than claiming it. **The count of passing claims about Jack moved by zero.**

**And the composition underneath is thinner than the settlement count suggests.**
Eleven distinct specs settled in the window. **Seven of them are Tier 0** —
`T0.17`, `T0.21`, `T0.28`, `T0.31`, `T0.33`, `T0.34` re-bought (several of them
more than once; `T0.28` three times, `T0.33` twice, `T0.21` twice) plus `T0.27`,
the deliberately-red gate held by `D16`. The other four are the window's
first-ever rows: `SO.06` PASS, `SO.09` PASS, and **two VOIDs** — `LG.03` on its
own liveness gate (`blind_calib_rate` 0.583 ± 0.312 against a 0.75 bar, and the
repair it pre-registered falsified by a 90-second probe within the hour) and
`SO.07` at its pre-registered reference lane after ~2.5 h of CPU (*the recording
worlds cannot produce the behaviour*).

**Goodhart: the rate fell, and this time the fall UNDERSTATES it.** −0.6 pts on
+8 registry against +2 demonstrated is the ordinary dilution reading. The honest
version is that the numerator's *creature* component moved zero, so on the only
quantity `GOAL.md` cares about the rate fell by the full width of the
registration. That is not an argument against registering — `LG.03`–`LG.06` and
`SO.06`–`SO.09` are both real families with real falsifiers, and both were
ordered from this desk. It is an argument against reading `104/242` as progress
this morning.

**On maintenance share I am reporting a hand classification, not a tool.** 36 of
50 non-journal commits carry an audit B-item, a certificate re-buy, a re-stamp,
a harvest or a routing repair; 13 serve the creature directly (the two research
passes, the two registrations, the four implementations and the three runs).
Flat against yesterday's 73%, and the method is mine, so treat the trend and not
the digit.

---

## The frontier, recomputed

`run blocked` is unchanged at the top and has been for weeks: **`T2.01` frees
35 / blocks 38**, settled FAIL since 08-12, **implementation unchanged 26 days**,
repair through `D1.0`. `NE.01` frees 8. `LT.01` frees 7. `UB.10` frees 4.
`T2.02` frees 3. `HR.1` frees 3 (D19-held). `HR.5` frees 2.

**One entry is new, and it is one day old: `LG.03` VOID, frees 3.** The family I
ordered on 09-04 was registered at 12:14 and by 17:20 its root had VOIDed and
become a terminal blocker in the ranker, with `LG.04`/`LG.05`/`LG.06` behind it.
That is not a failure of the order — a fixture that refuses to certify its venue
is doing precisely its job, and `LG.03`'s own `falsified_by` says so: *"this
world does not admit language-necessary commands at this horizon… the reading is
routed to `w0-too-shallow` as an instrument."* **It is the eleventh instrument on
that row.** `SO.07`'s VOID is arguably the twelfth; I am not counting it as one
until tomorrow's design reads it properly, because its VOID lane is about the
recording worlds rather than about W0 directly.

### The board has a tenth spec, and it is not like the other nine

Yesterday nine specs had no ledger row and all dependencies PASS, and all nine
were parked, pilot-blocked or decision-held. I re-derived the sweep this morning
rather than inheriting it — 242 registered, 139 with rows, 103 without, **ten
with all dependencies PASS** — and it agrees with the builder's 06:07 slot,
which ran the same sweep independently.

Nine are the same held set (`SH.01`, `SM.02`, `T2.11`, `T3.10` PARKED;
`SH.02`, `SM.03`, `DP.04`, `LC.07` PILOT-BLOCKED; `HR.1` D19-held). **The tenth
is `SO.08` — implemented, gates pre-registered from binomial arithmetic before
any run, and held by nothing but a clock.** Its cost class enumerates at
54,000 s against a 57,600 s day, so a full untouched day leaves 3,600 s of
slack; 09-05's allowance was already spent by `SO.07`'s honest VOID before
`SO.08` existed. It becomes runnable at 00:00 tonight and the window is one hour
wide. That is the first genuinely startable claim about Jack this desk has been
able to name in three days, and it needs nothing from me.

### The finding: a red ratchet about Jack went green in three minutes, and the
### instrument it went green into cannot go red

At **01:16** the 72nd audit shipped `FAIL-UNOWNED` — a counted, ratcheted class
for a settled FAIL with no repair owner — with an honest baseline of **4**,
correcting its own prose, which had said 3. It is good work and it found
something five instruments had missed for a fortnight: **`XL.01` — *"death does
not erase what he learned"* — has read FAIL for 17 days** with no owner, no
clock and no queue row, and `run blocked`, `coverage`, `review_queue`,
`champions` and `decisions` each reported it as fine, because every one of them
is keyed to a spec's REACHABILITY and none to its DISPOSITION.

At **01:19** the same audit routed the four orphans (`XL.01`, `T2.05`, `T4.02`,
`T2.15`) into `docs/REVIEW_QUEUE.md` at `DUE 2026-09-13`, and the baseline went
**4 → 0**. The class now reads *AT floor — ok*.

Every step of that was correct. Routing is the right response to an orphaned
FAIL; it is what the class exists to provoke; the rows are dated and reasoned;
`next_free_due` was used rather than piling. **The question is what the discharge
measured.** `coverage`'s printed definition is that a FAIL has an owner if it has
*"no `repaired_by`, no `REVIEW_QUEUE` mention, no `FAIL-DISPOSED` marker"* — so a
mention in the queue file is sufficient. And the queue, measured the same
morning from its own git history: **arrived 36 (5.14/cycle), disposed 1
(0.14/cycle), drain UNBOUNDED, 39 live rows, `0 violations`.**

So four of Jack's oldest settled failures moved from an instrument that was RED
about them to an instrument that reports zero violations by construction —
because the queue's violation classes fire on a promise BREAKING, not on a
promise being unpayable. **The debt did not shrink. It changed instruments, and
the one it moved to cannot go red about it.** Third appearance of this shape in
the record (08-26: three ratchets discharged by declaring claims rather than
passing them; 09-04: the divergence itself), and the first in which the
discharge flows *between* instruments, where no single tool is wrong and only
the composition is. Routed as **`D23`**, with the weakest of the three available
options as its default, for the reason given there.

### And the queue got worse, not better

I called this a finding about me yesterday and I will not soften it today.
Routed rows **35 → 41** in 24 hours; disposals in that window, before I sat
down: **zero**. Trailing-week drain still UNBOUNDED. I disposed one row this
morning (below) and re-dated one, which moves the trailing figure from
0.14/cycle to 0.29/cycle — against arrivals of 5.14. `D22` is on the owner's
desk with `decide_by` 09-08 and it is the only thing on this page that would
change that arithmetic.

---

## The honest paragraph

This was a day of competent, honest, well-sequenced work that gave Jack nothing,
and the sequencing is the reason rather than the excuse. The builder went where
it was sent, wrote up the owner's hands and the grounding of words in acts,
turned both into registered families with falsifiers, and then — correctly —
built the guards before the claims: the fixture that proves a hand can only
reach him through the world, and the accountant that would catch us puppeteering
him if we ever did. Both passed. Neither is a thing he can do. Then the two runs
that would have been about him came back empty in the two ways this project keeps
meeting: a certifier that could not prove its own blind twin was alive, and a
reference world that could not produce the behaviour it was supposed to
reference. Both were pre-registered lanes and both fired cleanly, which is the
machine working. The one unit that would have asked a real question about him
was written, gated and then refused by a clock, because an honest negative
earlier in the day had spent the day. And underneath all of it the instruments
read green — including one that was newly, correctly red about three of his
oldest failures, and that went to floor in the time it takes to write four dated
lines, by moving them to a desk whose own meter says it cannot pay. The most
important step toward Jack was building the accountant before the claim it
polices: choosing to make cheating detectable before making cheating possible is
the rarest thing this project does and it did it unprompted. The most concerning
drift is that *owned* and *being repaired* have quietly become the same reading
on our instruments while remaining opposites in fact — and the thing that makes
that dangerous is not that anybody lied, but that four separate tools each told
the truth and the truth they composed into was false.

---

## STEERING / STRENGTHENED

**No spec file was touched and no threshold moved in either direction.** Part 2
is tomorrow's.

- **`docs/CHAMPIONS.md` — the *Language grounding (word → lived skill)* seat:
  `ARENA: NONE` → `ARENA: LG.04, LG.05, LG.06`.** Part 2.5 duty 3 (seat whose
  arena context changed without a rematch). The seat has been declared
  `UNFALSIFIABLE` — *nothing that could be run would unseat the holder* — since
  the 51st audit, and the previous Review declined to name `LG.00` for a good
  reason: `LG.00` asks whether Jack's knowledge lives in his core and diary,
  while this seat contests *which grounding approach*. **That declination is not
  overturned; it is discharged.** On 09-04 the research pass this desk ordered
  registered `LG.04` (*the grounding bakeoff: five arms, one certified cell
  set*) and `LG.06` (*the ordering experiment: does skills-first buy anything*)
  — which are, by title, the two things the seat's own challenger cell has named
  since it was written: *"grounding approaches + the ordering experiment"*. So
  the ring exists and the seat did not know. `LG.00` is still not named here.
  **Stronger, and verified rather than asserted:** `champions --check` rc=0
  before and after, `UNFALSIFIABLE` **3 → 2**, uncontestable-in-total **4 → 3**,
  no ratchet raised, no seat added or removed, no holder unseated. Caveat
  recorded on the row: all three members are blocked behind `LG.03`'s VOID, so
  the ring is real but cannot be entered today — still strictly better than
  `NONE`, which asserts something that has been false since 09-04.
- **`docs/REVIEW_QUEUE.md` — `champions-language-grounding-arena` closed
  `ACTED`, two days early.** First disposal in seven days.
- **`docs/REVIEW_QUEUE.md` — `t027-preserved-failimpl-as-artifact` re-dated
  09-05 → 09-07 with the defect named.** This row came due **today** and was
  unclosable by anyone: its own gate line says the disposal is `D16`'s, and
  `D16`'s `decide_by` is 2026-09-05, so its default cannot fire until the first
  slot of 09-06 — *after* the row's date. A row dated before the default that
  disposes it. That is the 70th audit's `DEFAULT-ACTION-EXPIRED` arriving from
  the other side, and `review_queue.py` has no reader for that direction. Named
  on the row rather than routed as a new one, deliberately: the drain is the
  finding. Not re-dated onto 09-06, which already carries six rows.
- **`scripts/ladder_prompt.md`, four edits.** (i) `1''`/`2''` replaced by
  `1'''`/`2'''` — `2''` was executed in full inside 24 h, the **fifth
  consecutive day** a priority block was spent in one day, which left the block
  as prohibitions-only again, one day after I fixed exactly that. The
  replacement names the one live unit (`SO.08`), its 3,600-second window, and
  its ordering against the `D15`/`D16`/`D21` defaults at 00:0x. (ii) **I
  NARROWED MY OWN PROHIBITION.** Yesterday's *"no third increment of the CPU
  accountant"* would, read literally, have forbidden the 69th audit's B4
  (`9c5e74a`: the day gate now admits on a spec's MEASURED cost, self-inflicted
  foreclosures **53 → 36**) and the 70th's B4 (`40f6a32`: the meter prints slack
  rather than a bare count) — both repairs of the meter's own over-refusal, i.e.
  the exact harm the prohibition existed to prevent. It did no damage only
  because the overseer never reads that file, which is not a safeguard. The live
  rule is now: *a change that makes the meter refuse fewer runs, or tell the
  truth more plainly, is always allowed; what is prohibited is building more
  meter.* (iii) The GPU bullet rewritten — it named `W35` and expires at
  midnight; the standing rule now applies to `W36` by derivation, with `W36`'s
  named buyer and its gate stated. (iv) A **five-day-stale cached date**
  removed: the meter section said *"`week:Fable`… resets 2026-08-31 04:59
  UTC"*, on the one page that opens by promising no number is cached on it.
  Replaced with the tool-reading rule rather than a fresher date.
- **`docs/DECISIONS_NEEDED.md` — `D23` written** (see FOR THE OWNER 1). Its own
  dating was refused twice by `decisions --check`, once for
  `DEFAULT-ACTION-SAME-DAY` and once for `DEFAULT-ACTION-EXPIRED`; both refusals
  were correct, both are recorded in the entry, and `--check` is rc=0 with the
  ratchet clean as committed.

---

## FOR THE BUILDER

1. **At 2026-09-06 00:0x: fire `D15`/`D16`/`D21` first, then start `SO.08`
   before anything else bills a second.** The defaults are seconds of paperwork
   and `D21`'s is a race it must win before 06:37 (72nd audit finding 2). Then
   `SO.08`: 54,000 s of class enumeration against a 57,600 s day leaves 3,600 s
   of slack, so one certificate re-buy, one gate sweep, or one audit B-item that
   runs a spec takes the day. It is the only startable claim about Jack on the
   board and its window does not survive a distraction.
2. **After `SO.08`, if the board is empty again, say so and stop early.** Both
   research debts are discharged and there is no third queued. A slot that reads
   the board, verifies emptiness against `run blocked`/`run coverage`, writes
   what it checked and ends is a CORRECT slot and this page will not call it
   wasted. Do not write a fourth research doc because the previous two were
   accepted, and do not reach for a prohibition as cheap work.
3. **The CPU-accountant prohibition is NARROWER than you were told yesterday,
   and the correction is mine.** Repairs that make the meter refuse fewer runs
   or print more honestly need no permission — the 69th and 70th audits' B4s
   were both that, and both were good. What stays prohibited is new accounting
   *surface* while the meter's measured output is our own churn.
4. **Standing prohibitions, unchanged:** do not re-dispatch `D1.0` (the `d10-*`
   gate rows are owed here tomorrow); `HR.1`–`HR.4` stay D19-held to 09-14, do
   not fetch a corpus; `HR.6` stays blocked behind `HR.5`; `LF.01` attempt 2
   waits for the 09-09 design and `FIXTURE_VOID_CAP=3` is not permission; do not
   re-stagger the 09-06 docket a third time by hand; the ranker still cannot see
   `T2.01 → D1.0`.
5. **`W36` opens tonight with 30 h and the rule does not reset with the week.**
   Its named buyer is `D1.0`'s successor at ~16 h, and that is gated on the two
   `d10-*` rows I owe tomorrow. Until they land, `W36`'s hours are as
   unspendable as `W35`'s were, and that is not a fault to fix by dispatching
   something else. Derive the week from `%Y-W%U` against `gpu_budget.json`; do
   not read a week number off my page.

---

## FOR THE OWNER

1. **A ratchet about Jack's oldest failures went green in three minutes by
   routing them into the one queue that measures itself as unable to pay — and
   the honest recommendation is the weakest of the three fixes, because the two
   stronger ones both point a red light away from this desk.** Full arithmetic
   above and in **`D23`** (`class: goal`, `decide_by` 2026-09-11). The facts:
   `FAIL-UNOWNED` was shipped at 01:16 with baseline 4, found `XL.01` — *"death
   does not erase what he learned"* — sitting FAIL for **17 days** past five
   instruments that each called it fine, and went to floor 0 at 01:19 when the
   four orphans were routed to `DUE 09-13`; `coverage`'s definition accepts a
   `REVIEW_QUEUE` mention as an owner, and that file's own trailing-week reading
   is arrivals 5.14/cycle, disposals 0.14/cycle, **drain UNBOUNDED**, and
   `0 violations`.

   > **My recommendation: (iii) — keep counting the row, and print the drain
   > beside it.** `FAIL-UNOWNED` should go on accepting a `REVIEW_QUEUE` mention
   > as an owner, because the tighter reading punishes the one correct act
   > available to the organ that finds an orphan, and it would paint the ratchet
   > red for a reason the builder cannot fix and I can. What is missing is not a
   > stricter gate but a second number: **`FAIL-OWNED-BUT-UNDRAINED` — of the
   > settled FAILs this project calls owned, how many are owned by a desk whose
   > own instrument reads `drain UNBOUNDED`.** It moves no threshold, refuses no
   > run, weakens no control and fails no spec; it makes the difference between
   > *owned* and *being repaired* a printed integer instead of an inference a
   > reader has to make across two tools. I am recommending the weakest of the
   > three options on purpose, because the two stronger ones both end in a red
   > light pointed at somebody else, and the desk that would be exonerated by
   > that is mine.

2. NO-DECISION: self-report of a steering correction I made and then reversed
   myself on; nothing here to rule on, and it is already executed.
   Yesterday I put *"no third increment of the CPU accountant"* into the
   builder's file. Within nine hours the overseer shipped two edits to that
   instrument — both of them repairs that made it refuse **fewer** runs (53 → 36
   foreclosed) and print more honestly. Read literally my prohibition would have
   forbidden both. It did no harm only because the overseer does not read the
   builder's file. I have narrowed the rule to what the measurement actually
   supports, and I am recording the near-miss rather than asking for a wider
   channel to the other organs: **the lesson runs the other way.** A desk that
   watches an instrument once a day should not be writing blanket prohibitions
   about it, and the correct scope for my steering is the one organ whose output
   I read every morning.

3. **Tomorrow's FULL, 2026-09-06 — order unchanged and now published for the
   third day, cited to `D21`** (whose default fires at 00:0x tomorrow and is a
   same-day race against my own 06:37 start; the 72nd audit's finding 2 is
   correct and the builder has the handoff). Six live rows name the day: both
   `d10-*` gate rows first (cheap, and they release a ~16 h dispatch into
   `W36`'s 30 free hours), then **`w0-too-shallow` at twelve days and now
   eleven instruments** — `LG.03`'s venue VOID is the eleventh — then
   `lt01-c2-body-cannot-rise`, `lc07-checkpoint-branch`,
   `cross-organ-doc-race-voids-certificates`, then Part 2 at its minimum of 8
   plus both completeness audits. `me11-every-arm-hits-the-same-infeasible-
   branch` is already `ACTED`. If `D22` is granted before then, this docket is
   the first thing that changes shape.

4. NO-DECISION: liveness report, nothing here to rule on.
   All four organs live, verified against `/data/jack-logs` mtimes rather than
   against anyone's report: builder 06:12 (hourly, 24 starts in the window, zero
   `PACING:` — last skip 08-29), overseer 06:37 (6 h), field watch 08-31 05:53
   (Mondays — next fire 09-07, well inside cadence; wk5 was consumed on 08-31
   and `FIELD_WATCH.md` is unchanged since, so **Part 2.5 duty 2 has nothing to
   consume this run**), review 06:37 (this run). `lost_iterations.log` still
   0 bytes and still never exercised. No organ is silent. The overseer and I
   again shared the 06:37 minute and the git index; committed with an explicit
   pathspec.
